"""Serialized operator actions with interruptible Stop and fail-closed release guards."""
import asyncio
import copy
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import time
import uuid

LOAD_WAIT_SECONDS = 2.0
DROP_SETTLE_SECONDS = 2.0
SETTLE_SECONDS = 1.0
SAMPLE_SECONDS = 0.15
POST_RELEASE_SECONDS = 2.0
FRESH_SECONDS = 3.0
SETTLE_TIMEOUT_SECONDS = 15.0
RECORDING_TIMEOUT_SECONDS = 12.0
POSE_TOLERANCE_MM = 2.0
ORIENTATION_TOLERANCE_DEG = 1.0
BOARD_TOLERANCE_DEG = 0.5


def utc():
    return datetime.now(timezone.utc).isoformat()


def number(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise TrialError(f'{name} must be a finite number')
    return float(value)


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def orientation_distance(a, b):
    def ov(p):
        axis = [number(p[k], k) for k in ('o_x', 'o_y', 'o_z')]
        angle = number(p['theta'], 'theta')
        norm = math.sqrt(sum(v*v for v in axis))
        if norm < 1e-8:
            raise TrialError('Invalid orientation vector')
        return [v/norm for v in axis], angle
    # Viam OV is tool-Z direction plus roll, NOT axis-angle. This intentionally
    # conservative comparison requires the saved OV representation to match;
    # equivalent alternate representations may be rejected, never inferred safe.
    av, at = ov(a)
    bv, bt = ov(b)
    direction = math.degrees(math.acos(max(-1, min(1, sum(x*y for x,y in zip(av,bv))))))
    roll = abs((at - bt + 180) % 360 - 180)
    return direction + roll


def pose_matches(a, b):
    return (math.dist([number(a[k], k) for k in ('x', 'y', 'z')],
                      [number(b[k], k) for k in ('x', 'y', 'z')]) <= POSE_TOLERANCE_MM
            and orientation_distance(a, b) <= ORIENTATION_TOLERANCE_DEG)


class TrialError(ValueError):
    pass


class Controller:
    def __init__(self, adapter, root, mode='dry_run', audio=None):
        self.adapter, self.root, self.mode = adapter, Path(root), mode
        # Optional observer only. Its bounded enqueue methods never await a
        # network response, and no audio state participates in robot guards.
        self.audio = audio
        self.audio_clock_id = 'sircus-dashboard-' + uuid.uuid4().hex
        self.audio_observer_errors = 0
        self.root.mkdir(parents=True, exist_ok=True)
        self.config_path = self.root / 'calibration.json'
        self.calibration = json.loads(self.config_path.read_text()) if self.config_path.exists() else {}
        self.connected = False
        self.robot = {}
        self.snapshot_time = 0
        self.phase = 'disconnected'
        self.error = None
        self.task = None
        self.poll_task = None
        self.generation = 0
        self.stopping = False
        self.fault = False
        self.audit = []
        self.gripper_closed = False
        self.gripper_command_state = 'unknown'
        self.loading_deadline = None
        self.trial = None
        self.capabilities = {}
        self.recording = {'active': False, 'ready': False}
        self._journal_recovery()

    def _journal_recovery(self):
        paths = sorted((self.root / 'trials').glob('*/trial.json'))
        pending = []
        for path in paths:
            trial = json.loads(path.read_text())
            if not trial.get('saved'):
                trial.update(status='interrupted', interrupted_at=utc(),
                             interruption='Dashboard restarted; no motion or recording resumed')
                write_json(path, trial)
                pending.append(trial)
        if pending:
            self.trial = pending[-1]
            self.fault = True
            self.error = 'Interrupted trial recovered; use Stop to verify dropper state before further actions'
            self.phase = 'awaiting_outcome'

    async def start(self):
        try:
            self.capabilities = await self.adapter.connect()
            self.connected = True
            await self._snapshot()
            self.phase = 'awaiting_outcome' if self._unsaved() else 'idle'
        except Exception as exc:
            self.error = f'Connection/readback unavailable ({type(exc).__name__})'
            self.phase = 'disconnected'
        self.poll_task = asyncio.create_task(self._poll())

    async def close(self):
        # Application shutdown is not a physical command. Operators must Stop first.
        self._audio_cancel()
        for task in (self.poll_task, self.task):
            if task and not task.done():
                task.cancel()
        await asyncio.gather(*(t for t in (self.poll_task, self.task) if t), return_exceptions=True)
        await self.adapter.finish_recording()
        await self.adapter.close()

    async def _snapshot(self):
        try:
            state = await asyncio.wait_for(self.adapter.snapshot(), 5)
            for name in ('arm', 'arm-board'):
                arm = state['arms'][name]
                if not isinstance(arm['is_moving'], bool) or len(arm['joints']) != 6:
                    raise TrialError('Incomplete arm readback')
                for value in arm['joints']:
                    number(value, 'joint')
                for key in ('x', 'y', 'z', 'o_x', 'o_y', 'o_z', 'theta'):
                    number(arm['pose'][key], key)
            self.robot, self.snapshot_time, self.connected = state, time.monotonic(), True
            return state
        except BaseException:
            self.snapshot_time = 0
            self.connected = False
            self.gripper_command_state = 'unknown'
            raise

    async def _poll(self):
        while True:
            try:
                # Cloud signaling can legitimately take longer than a state
                # readback. Keep connection setup outside the 5-second sample
                # deadline so recovery does not continually cancel a good dial.
                await self.adapter.connect()
                await self._snapshot()
                self.recording = await self.adapter.recording_status()
                if self.phase == 'disconnected':
                    self.phase = 'awaiting_outcome' if self._unsaved() else 'idle'
                    self.capabilities = getattr(self.adapter, 'capabilities', self.capabilities)
                if self.error and self.error.startswith(('Connection/readback unavailable', 'Readback unavailable')):
                    self.error = None
                if self.phase == 'ready' and self._drop_blockers():
                    # Eligibility is recomputed, not latched. A click always revalidates.
                    self.error = '; '.join(self._drop_blockers())
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self.connected = False
                self.snapshot_time = 0
                self.error = f'Readback unavailable ({type(exc).__name__}); release disabled'
            await asyncio.sleep(0.5)

    def _unsaved(self):
        return self.trial is not None and not self.trial.get('saved')

    def calibration_missing(self):
        c = self.calibration
        missing = [label for key, label in [('loading_pose', 'Taught loading pose'),
            ('drop_pose', 'Taught fixed drop XY and orientation'),
            ('board_joints', 'Board-holder reference joints'),
            ('reference_z_mm', 'Calibrated release-height reference Z'),
            ('min_height_mm', 'Minimum permitted height'), ('max_height_mm', 'Maximum permitted height')]
            if c.get(key) is None]
        if not c.get('confirmed'):
            missing.append('Operator confirmation of calibration and fixed board/bucket placement')
        return missing

    def fixed_cycle_missing(self):
        missing = []
        for key in ('loading_pose', 'drop_pose'):
            try:
                if not pose_matches(self.calibration[key], self.calibration[key]):
                    raise TrialError('Invalid pose')
            except (KeyError, TypeError, ValueError):
                missing.append('Teach ' + key.replace('_', ' '))
        joints = self.calibration.get('board_joints')
        if not isinstance(joints, list) or len(joints) != 6:
            missing.append('Teach board reference with drop pose')
        return missing

    def _stationary(self, state):
        if any(state['arms'][n]['is_moving'] for n in ('arm', 'arm-board')):
            raise TrialError('Both arms must be stationary')

    def _board_fixed(self, state):
        ref = self.calibration.get('board_joints')
        actual = state['arms']['arm-board']['joints']
        if ref is None or len(ref) != 6 or max(abs(a-b) for a, b in zip(ref, actual)) > BOARD_TOLERANCE_DEG:
            raise TrialError('Board-holder pose changed; reteach and reconfirm the setup')
        if state['arms']['arm-board']['is_moving']:
            raise TrialError('Board-holder is moving')

    def _height(self, value):
        h = number(value, 'height_mm')
        if self.calibration_missing():
            raise TrialError('Calibration is incomplete')
        if not self.calibration['min_height_mm'] <= h <= self.calibration['max_height_mm']:
            raise TrialError('Height is outside the calibrated limits')
        return h

    def _drop_blockers(self):
        blockers = []
        if not self.connected or time.monotonic() - self.snapshot_time > FRESH_SECONDS:
            blockers.append('Fresh robot readback required')
        if self.trial and self.trial.get('behavior') == 'fixed_pose_cycle':
            if self.fixed_cycle_missing() or not self.trial.get('fixed_setup_confirmed'):
                blockers.append('Taught poses and fixed setup confirmation required')
        elif self.calibration_missing():
            blockers.append('Calibration incomplete')
        if not self.gripper_closed:
            blockers.append('Close the dropper gripper before the trial')
        if not self.trial or self.trial.get('status') != 'ready':
            blockers.append('Go to the requested drop position first')
        elif self.robot:
            try:
                self._stationary(self.robot)
                self._board_fixed(self.robot)
                if not pose_matches(self.robot['arms']['arm']['pose'], self.trial['target_pose']):
                    blockers.append('Dropper is not at the intended release pose')
            except (TrialError, KeyError) as exc:
                blockers.append(str(exc))
        rec = self.recording
        if (not rec.get('active') or not rec.get('ready') or rec.get('error') or
                time.monotonic() - rec.get('last_frame_monotonic', 0) > FRESH_SECONDS):
            blockers.append('Recording must be active with a fresh saved frame and robot state')
        return blockers

    def status_snapshot(self):
        busy = self.stopping or bool(self.task and not self.task.done())
        allowed = ['stop']
        fresh = self.connected and time.monotonic() - self.snapshot_time <= FRESH_SECONDS
        if not busy:
            if self._unsaved():
                if self.phase == 'ready' and not self._drop_blockers():
                    allowed += ['drop']
                if self.phase == 'awaiting_outcome':
                    allowed += ['save_trial', 'keep_unlabeled']
            elif fresh and not self.fault:
                allowed += ['teach_loading', 'teach_drop', 'configure']
                if not any(a['is_moving'] for a in self.robot['arms'].values()):
                    allowed += ['open_gripper', 'close_gripper']
                    if not self.fixed_cycle_missing():
                        try:
                            self._board_fixed(self.robot)
                            allowed += ['load_drop_cycle']
                        except TrialError:
                            pass
                    if not self.calibration_missing():
                        try:
                            self._board_fixed(self.robot)
                            allowed += ['go_loading']
                            if self.gripper_closed:
                                allowed += ['go_drop']
                        except TrialError:
                            pass
        return copy.deepcopy({'mode': self.mode, 'machine': 'sfhack10', 'connected': fresh,
            'phase': self.phase, 'busy': busy, 'allowed_actions': allowed, 'robot': self.robot,
            'calibration': self.calibration, 'calibration_missing': self.calibration_missing(),
            'recording': self.recording, 'trial': self.trial, 'error': self.error,
            'fault': self.fault,
            'fixed_cycle_missing': self.fixed_cycle_missing(),
            'loading_seconds_remaining': max(0, math.ceil(self.loading_deadline - time.monotonic())) if self.loading_deadline else None,
            'blockers': self._drop_blockers() if self.phase == 'ready' else [],
            'capabilities': self.capabilities, 'gripper_closed_commanded': self.gripper_closed,
            'gripper_command_state': self.gripper_command_state,
            'gripper_state_provenance': 'Last acknowledged panel command; not jaw-position sensing or external-controller tracking'})

    async def status(self):
        return self.status_snapshot()

    async def action(self, payload):
        name = payload.get('action')
        if name == 'stop':
            await self._stop()
            return {'accepted': True, **await self.status()}
        if name not in (await self.status())['allowed_actions']:
            raise TrialError('Action unavailable: another action/trial is pending, state is stale, or calibration is missing')
        if name == 'go_drop':
            self._height(payload.get('height_mm'))
        if name == 'load_drop_cycle' and payload.get('fixed_setup_confirmed') is not True:
            raise TrialError('Confirm fixed board/bucket, clear route and two-second automatic close')
        if name in ('save_trial', 'keep_unlabeled'):
            if payload.get('trial_id') != self.trial['id']:
                raise TrialError('Outcome trial ID does not match the completed trial')
            if name == 'save_trial' and payload.get('outcome') not in ('captured', 'missed', 'unsure'):
                raise TrialError('Choose Captured, Missed or Unsure')
            if len(str(payload.get('notes', ''))) > 4000 or len(str(payload.get('miss_direction', ''))) > 120:
                raise TrialError('Trial annotation is too long')
        if not self.fault:
            self.error = None
        generation = self.generation
        self._event(name, 'operator_requested')
        self.phase = name
        self.task = asyncio.create_task(self._run(name, copy.deepcopy(payload), generation))
        return {'accepted': True, **await self.status()}

    def _persist(self):
        if self.trial:
            write_json(self.root / 'trials' / self.trial['id'] / 'trial.json', self.trial)

    def _event(self, command, event, **extra):
        entry = {'event_id': uuid.uuid4().hex, 'command': command, 'event': event, 'timestamp': utc(),
                 'monotonic': time.monotonic(), 'trial_id':self.trial['id'] if self._unsaved() else None, **extra}
        self.audit.append(entry)
        self.audit = self.audit[-100:]
        with (self.root / 'operator-commands.jsonl').open('a') as stream:
            stream.write(json.dumps(entry, allow_nan=False) + '\n')
        if self._unsaved():
            self.trial['events'].append(entry)
            self._persist()
        self._audio_lifecycle(entry)

    def _audio_lifecycle(self, entry):
        """Observe explicit operator-requested stages, not inferred contacts.

        Use the persisted audit ID/time verbatim. A release cue means the open
        command was sent, not that separation, impact or capture was observed.
        Dry-run actions are deliberately silent rather than labelled physical.
        """
        if self.audio is None or self.mode != 'hardware' or not entry.get('trial_id'):
            return
        kinds = {('close_gripper', 'command_sent'): 'pickup',
                 ('load_drop_cycle', 'move_drop_sent'): 'carry_start',
                 ('load_drop_cycle', 'move_drop_returned'): 'carry_stop',
                 ('go_drop', 'requested'): 'carry_start',
                 ('go_drop', 'motion_returned'): 'carry_stop',
                 ('drop', 'command_sent'): 'release'}
        kind = kinds.get((entry['command'], entry['event']))
        if kind is None:
            return
        try:
            from sircus.audio.contract import AudioEvent
            cue = AudioEvent(event_id=entry['event_id'], trial_id=entry['trial_id'],
                source='operator', event_type=kind, contact_object='unknown',
                timestamp=datetime.fromisoformat(entry['timestamp']).timestamp(),
                time_basis='unix_s', clock_id=self.audio_clock_id,
                evidence='operator_action',
                provenance=f"Operator-requested controller stage {entry['command']}/{entry['event']}; command timing only, not pickup, ball separation, impact or outcome detection")
            self.audio.emit(cue.to_dict())
        except Exception:
            self.audio_observer_errors += 1

    def _audio_outcome(self):
        if self.audio is None or self.mode != 'hardware':
            return
        try:
            from sircus.audio.sources import operator_outcome
            cue = operator_outcome(self.trial)
            if cue.confirmed and cue.outcome in ('captured', 'missed'):
                self.audio.emit(cue.to_dict())
        except Exception:
            self.audio_observer_errors += 1

    def _audio_cancel(self):
        if self.audio is not None and self.trial is not None:
            try:
                self.audio.stop(self.trial['id'])
            except Exception:
                self.audio_observer_errors += 1

    async def _settle(self, target=None, board_guard=True, stable_seconds=None):
        stable_seconds = SETTLE_SECONDS if stable_seconds is None else stable_seconds
        deadline, stable_since, previous = time.monotonic() + SETTLE_TIMEOUT_SECONDS, None, None
        while time.monotonic() < deadline:
            state = await self._snapshot()
            if board_guard:
                self._board_fixed(state)
            stationary = not any(state['arms'][n]['is_moving'] for n in ('arm', 'arm-board'))
            if target is not None:
                stationary = stationary and pose_matches(state['arms']['arm']['pose'], target)
            if previous:
                stationary = stationary and all(max(abs(a-b) for a,b in zip(
                    state['arms'][n]['joints'], previous['arms'][n]['joints'])) <= 0.1 for n in ('arm', 'arm-board'))
            if stationary:
                stable_since = stable_since or time.monotonic()
                if time.monotonic() - stable_since >= stable_seconds:
                    return state
            else:
                stable_since = None
            previous = state
            await asyncio.sleep(SAMPLE_SECONDS)
        raise TrialError('Arm did not settle at the intended pose')

    async def _prepare_recording(self, still_current):
        await self.adapter.start_recording(self.trial['id'], self.root/'trials'/self.trial['id'], self.trial)
        still_current()
        deadline = time.monotonic() + RECORDING_TIMEOUT_SECONDS
        while time.monotonic() < deadline:
            self.recording = await self.adapter.recording_status()
            still_current()
            if self.recording.get('error'):
                raise TrialError('Recording failed; release remains disabled')
            if self.recording.get('ready'):
                if not self.recording.get('active') or time.monotonic() - self.recording.get('last_frame_monotonic', 0) > FRESH_SECONDS:
                    raise TrialError('Recorded frame is stale; release remains disabled')
                break
            await asyncio.sleep(0.2)
        else:
            raise TrialError('No recorded frame and robot state; release remains disabled')
        self.trial['status'] = self.phase = 'ready'
        self._event('recording', 'ready')

    async def _command_gripper(self, opening, still_current):
        # Invalidate before awaiting: failure, cancellation or late completion
        # must never leave a stale 'open' cache after a close attempt.
        self.gripper_command_state = 'unknown'
        self.gripper_closed = False
        result = await (self.adapter.open_gripper() if opening else self.adapter.close_gripper())
        still_current()
        self.gripper_command_state = 'open' if opening else 'closed'
        if not opening:
            self.gripper_closed = result.get('grabbed') is True if isinstance(result, dict) else result is True
        return result

    async def _release_current(self, still_current):
        settled = await self._settle(self.trial['target_pose'], stable_seconds=DROP_SETTLE_SECONDS if self.trial.get('behavior') == 'fixed_pose_cycle' else None)
        still_current()
        self.trial['settled_pose'] = settled['arms']['arm']['pose']
        self.recording = await self.adapter.recording_status()
        blockers = self._drop_blockers()
        if blockers:
            raise TrialError('; '.join(blockers))
        measured = await self._snapshot()
        self._stationary(measured)
        self._board_fixed(measured)
        if not pose_matches(measured['arms']['arm']['pose'], self.trial['target_pose']):
            raise TrialError('Release pose changed during final validation')
        still_current()
        self.recording = await self.adapter.recording_status()
        still_current()
        blockers = self._drop_blockers()
        if blockers:
            raise TrialError('; '.join(blockers))
        self.trial['measured_release_pose'] = measured['arms']['arm']['pose']
        self.trial['measured_release_state_before'] = measured
        self.trial['status'] = 'releasing'
        still_current()
        self._event('drop', 'command_sent')
        await self._command_gripper(True, still_current)
        self._event('drop', 'command_returned')
        self.trial['measured_release_state_after'] = await self._snapshot()
        still_current()
        self.phase = 'post_release_recording'
        await asyncio.sleep(POST_RELEASE_SECONDS)
        still_current()
        await self.adapter.finish_recording()
        still_current()
        self.recording = await self.adapter.recording_status()
        self.trial.update(status='completed', completed_at=utc())
        self._persist()
        self.phase = 'awaiting_outcome'

    async def _run(self, name, p, generation):
        def still_current():
            if generation != self.generation or self.stopping:
                raise asyncio.CancelledError()
        try:
            still_current()
            if name in ('teach_loading', 'teach_drop'):
                state = await self._settle(board_guard=False)
                still_current()
                pose = copy.deepcopy(state['arms']['arm']['pose'])
                self.calibration['loading_pose' if name == 'teach_loading' else 'drop_pose'] = pose
                if name == 'teach_drop':
                    self.calibration['board_joints'] = state['arms']['arm-board']['joints']
                    self.calibration['board_pose'] = state['arms']['arm-board']['pose']
                self.calibration.update(confirmed=False, taught_at=utc(), frame='world',
                                        pose_reference='arm flange as reported by motion.get_pose; not ball center')
                write_json(self.config_path, self.calibration)
                self.phase = 'idle'
            elif name == 'configure':
                c = copy.deepcopy(self.calibration)
                for key in ('reference_z_mm', 'min_height_mm', 'max_height_mm'):
                    c[key] = number(p.get(key), key)
                if not 0 <= c['min_height_mm'] < c['max_height_mm']:
                    raise TrialError('Require 0 <= minimum height < maximum height')
                if not c.get('loading_pose') or not c.get('drop_pose') or not c.get('board_joints'):
                    raise TrialError('Teach both poses before confirming calibration')
                if p.get('confirmed') is not True:
                    raise TrialError('Confirm calibrated poses/limits and fixed board/bucket placement')
                state = await self._snapshot()
                self._stationary(state)
                self._board_fixed(state)
                c.update(confirmed=True, confirmed_at=utc(), setup_revision=str(p.get('setup_revision',''))[:200],
                         height_definition='arm flange world Z minus calibrated reference Z; reference must include held-ball offset')
                still_current()
                self.calibration = c
                write_json(self.config_path, c)
                self.phase = 'idle'
            elif name in ('open_gripper', 'close_gripper'):
                self._stationary(await self._snapshot())
                still_current()
                await self._command_gripper(name == 'open_gripper', still_current)
                self.phase = 'idle'
            elif name == 'load_drop_cycle':
                await self._fixed_cycle(still_current, practice=p.get('practice_mode') is True)
            elif name == 'go_loading':
                state = await self._snapshot()
                self._stationary(state)
                self._board_fixed(state)
                still_current()
                await self.adapter.move_pose(copy.deepcopy(self.calibration['loading_pose']))
                await self._settle(self.calibration['loading_pose'])
                still_current()
                self.phase = 'idle'
            elif name == 'go_drop':
                height = self._height(p['height_mm'])
                state = await self._snapshot()
                self._stationary(state)
                self._board_fixed(state)
                target = copy.deepcopy(self.calibration['drop_pose'])
                target['z'] = self.calibration['reference_z_mm'] + height
                trial_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ') + '-' + uuid.uuid4().hex[:8]
                self._audio_cancel()  # Retire the previous trial before replacement.
                self.trial = {'schema':'sircus-physical-trial-v1', 'id':trial_id, 'machine':'sfhack10',
                    'mode':self.mode, 'status':'positioning', 'created_at':utc(), 'saved':False,
                    'requested_height_mm':height, 'height_mm':height, 'target_pose':target,
                    'calibration':copy.deepcopy(self.calibration), 'calibration_sha256':hashlib.sha256(
                        json.dumps(self.calibration,sort_keys=True).encode()).hexdigest(), 'events':[],
                    'pretrial_operator_events':copy.deepcopy([e for e in self.audit[-40:]
                        if e['command'] in ('open_gripper','close_gripper','go_loading','teach_loading','teach_drop','configure')]),
                    'synchronization_limits':['RPC observations have local request/response times; camera exposure and joint readbacks are not hardware synchronized.',
                        'Measured release pose is bracketed readback around gripper command, not measured ball separation time.',
                        'Board stability is joint readback; bucket placement is operator-confirmed, not visually localized.']}
                self._event(name, 'requested')
                still_current()
                await self.adapter.move_pose(target)
                still_current()
                self._event(name, 'motion_returned')
                settled = await self._settle(target)
                still_current()
                self.trial['settled_pose'] = settled['arms']['arm']['pose']
                self._event(name, 'settled')
                await self._prepare_recording(still_current)
            elif name == 'drop':
                await self._release_current(still_current)
            elif name == 'keep_unlabeled':
                still_current()
                self._keep_unlabeled()
            elif name == 'save_trial':
                still_current()
                self.trial.update(outcome=p['outcome'], miss_direction=str(p.get('miss_direction','')),
                    notes=str(p.get('notes','')), annotated_at=utc(), saved=True, saved_at=utc(),
                    cloud={'state':'pending'})
                self._persist()
                self._audio_outcome()  # Saved human label; cloud/audio readiness never gates it.
                try:
                    self.trial['cloud'] = await self.adapter.save_trial(self.root/'trials'/self.trial['id'],self.trial)
                except Exception as exc:
                    self.trial['cloud'] = {'state':'pending','error_type':type(exc).__name__}
                self._persist()
                self.phase = 'idle'
            if generation != self.generation:
                raise asyncio.CancelledError()
            self._event(name, 'action_completed')
        except asyncio.CancelledError:
            self._audio_cancel()
            raise
        except Exception as exc:
            if generation != self.generation:
                return
            self._audio_cancel()
            self.error = str(exc) if isinstance(exc, TrialError) else f'{name} failed ({type(exc).__name__}); check hardware/connection'
            if name in ('go_loading','go_drop','drop','load_drop_cycle'):
                try:
                    result = await self.adapter.stop()
                    if isinstance(result, dict) and result.get('planner_cancellation_verified') is False:
                        raise TrialError('Planner cancellation not verified')
                except Exception:
                    self.error += '; dropper stop not confirmed'
                    self.fault = True
            if self._unsaved():
                self.trial.update(status='aborted', error=self.error, completed_at=utc())
                self._event(name, 'failed')
                try:
                    await self.adapter.finish_recording()
                except Exception:
                    self.error += '; recording finalization failed; partial files retained'
                finally:
                    self.phase = 'awaiting_outcome'
            else:
                self.phase = 'idle'

    def _keep_unlabeled(self):
        if not self.trial or self.trial['status'] not in ('completed', 'aborted', 'interrupted'):
            raise TrialError('Only a finished or stopped trial can be kept unlabeled')
        self._event('keep_unlabeled', 'retained_locally')
        self.trial.update(outcome=None, label_status='unlabeled', saved=True, saved_at=utc(),
                          cloud={'state':'deferred', 'reason':'Practice: retained locally without outcome label or upload'})
        self._persist()
        self.phase = 'idle'

    async def _fixed_cycle(self, still_current, practice=False):
        state = await self._snapshot()
        self._stationary(state)
        self._board_fixed(state)
        still_current()
        config = copy.deepcopy(self.calibration)
        trial_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ') + '-' + uuid.uuid4().hex[:8]
        self._audio_cancel()
        self.trial = {'schema': 'sircus-physical-trial-v1', 'id': trial_id,
            'machine': 'sfhack10', 'mode': self.mode, 'behavior': 'fixed_pose_cycle',
            'status': 'positioning', 'created_at': utc(), 'saved': False,
            'requested_height_mm': None, 'height_mm': None,
            'height_status': 'unknown; exact taught flange pose, no height adjustment',
            'target_pose': config['drop_pose'], 'loading_pose': config['loading_pose'],
            'calibration': config, 'calibration_sha256': hashlib.sha256(
                json.dumps(config, sort_keys=True).encode()).hexdigest(),
            'fixed_setup_confirmed': True, 'loading_wait_seconds': LOAD_WAIT_SECONDS, 'drop_settle_seconds': DROP_SETTLE_SECONDS, 'post_release_seconds': POST_RELEASE_SECONDS, 'practice_mode': practice,
            'events': [], 'synchronization_limits': [
                'Independent camera/state RPCs; not hardware synchronized.',
                'Release pose brackets gripper command, not observed ball separation.',
                'Board constancy is joint readback; bucket placement is operator-confirmed.']}
        self._persist()
        self.phase = 'cycle_to_loading'
        self._event('load_drop_cycle', 'move_loading_sent')
        await self.adapter.move_pose(config['loading_pose'])
        still_current()
        await self._settle(config['loading_pose'])
        still_current()
        # Fail before opening for loading if a recording cannot be established.
        await self._prepare_recording(still_current)
        still_current()
        if self.gripper_command_state == 'open':
            self._event('open_gripper', 'loading_open_skipped', reason='Last panel command acknowledged open')
        else:
            self.phase = 'cycle_opening'
            self._event('open_gripper', 'loading_open_sent')
            await self._command_gripper(True, still_current)
        self.loading_deadline = time.monotonic() + LOAD_WAIT_SECONDS
        self.phase = 'cycle_loading_countdown'
        self._event('load_drop_cycle', 'loading_countdown_started', seconds=LOAD_WAIT_SECONDS)
        try:
            while time.monotonic() < self.loading_deadline:
                state = await self._snapshot()
                still_current()
                self._stationary(state)
                self._board_fixed(state)
                if not pose_matches(state['arms']['arm']['pose'], config['loading_pose']):
                    raise TrialError('Loading pose changed during countdown')
                await asyncio.sleep(min(.1, max(0, self.loading_deadline - time.monotonic())))
            still_current()
            self.phase = 'cycle_closing'
            self._event('close_gripper', 'command_sent')
            await self._command_gripper(False, still_current)
            self._event('close_gripper', 'command_returned', grabbed=self.gripper_closed)
            if not self.gripper_closed:
                raise TrialError('Gripper did not report a held ball; cycle aborted')
        finally:
            self.loading_deadline = None
        state = await self._snapshot()
        still_current()
        self._stationary(state)
        self._board_fixed(state)
        self.phase = 'cycle_to_drop'
        self.trial['status'] = 'positioning'
        self._event('load_drop_cycle', 'move_drop_sent')
        await self.adapter.move_pose(config['drop_pose'])
        still_current()
        self._event('load_drop_cycle', 'move_drop_returned')
        self.trial['status'] = 'ready'
        self.phase = 'cycle_release_checks'
        await self._release_current(still_current)
        still_current()
        if practice:
            self._keep_unlabeled()

    async def _stop(self):
        if self.stopping:
            return
        self.stopping = True
        self.gripper_command_state = 'unknown'
        self.generation += 1
        active = self.task
        if active and not active.done():
            active.cancel()
        self.phase = 'stopping'
        self._audio_cancel()  # Enqueue only; robot Stop below never awaits audio.
        self._event('stop', 'requested')
        try:
            # Independent of the busy action. Never stop_all(), gripper.open() or vacuum commands.
            result = await asyncio.wait_for(self.adapter.stop(), 15)
            if isinstance(result, dict) and result.get('planner_cancellation_verified') is False:
                raise TrialError('Pending Viam planner cancellation was not verified')
            self._event('stop', 'dropper_stop_returned')
            state = await self._snapshot()
            if state['arms']['arm']['is_moving']:
                raise TrialError('Dropper still reports moving after Stop')
            self.fault = False
            self.error = None
        except Exception as exc:
            self.fault = True
            self.error = f'Stop not confirmed ({type(exc).__name__}); use physical emergency stop if needed'
        finally:
            if active:
                try:
                    await asyncio.wait_for(asyncio.gather(active, return_exceptions=True), 2)
                except asyncio.TimeoutError:
                    self.fault = True
                    self.error = 'Pending action cancellation not confirmed; hardware Stop must be checked'
            try:
                await asyncio.wait_for(self.adapter.finish_recording(), 6)
                self.recording = await self.adapter.recording_status()
            except Exception:
                self.error = self.error or 'Recording finalization unconfirmed; local partial files retained'
            if self._unsaved():
                self.trial.update(status='aborted', completed_at=utc(), stop_error=self.error)
                self._persist()
                self.phase = 'awaiting_outcome'
            else:
                self.phase = 'idle'
            self.stopping = False
