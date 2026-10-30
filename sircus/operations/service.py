"""Threaded, simulation-only guided operation; there is no physical execution path."""
import copy
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import threading
import uuid

ROOT = Path(__file__).resolve().parents[2]
MISSING = [
    'Measured arm-base-to-arm-base transform and verified arm-14 joint readback',
    'Measured suction TCP-to-board impact-surface transform and compliance',
    'Camera-to-dropper-tool calibration, verified depth scale and RGB/depth alignment',
    'Validated bucket opening detector, world transform, scan pose and observation freshness',
    'Taught loading pose, ball-in-gripper offset, release orientation and height limits',
    'Matching v2 native adapter and collision-checked arm trajectories',
    'Real-board bounce/contact calibration and separately authorized physical trials',
    'Clarify whether reported 175 mm bucket height is overall height or interior depth',
]


def now():
    return datetime.now(timezone.utc).isoformat()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def vector(value, length, label, bound):
    if not isinstance(value, list) or len(value) != length or any(type(x) not in (int, float) or not math.isfinite(x) or abs(x) > bound for x in value):
        raise ValueError(f'{label} requires {length} finite values within +/-{bound}')
    return [float(x) for x in value]


class GuidedOperations:
    """All effects are local records and child numerical processes, never robots."""
    def __init__(self, root=None, data_root=None, *, worker_timeout=120, runner=None):
        self.root = Path(root or ROOT).resolve()
        self.data_root = Path(data_root or self.root/'.runtime/guided-trials')
        self.worker_timeout = worker_timeout
        self.runner = runner
        self.lock = threading.RLock()
        self._thread = self._process = None
        self._cancel = threading.Event()
        self._generation = 0
        self._closed = False
        self._default_scene = json.loads((self.root/'baselines/scene-conditioned-20261030/preview-scene.json').read_text())['scene']
        self._default_checkpoint = json.loads((self.root/'baselines/scene-conditioned-20261030/checkpoint.json').read_text())
        self._checkpoint = copy.deepcopy(self._default_checkpoint)
        self._state = {'scene': copy.deepcopy(self._default_scene), 'revision': 0, 'phase': 'position_board',
            'board_confirmed': False, 'bucket_confirmed': False, 'board_pose': None,
            'bucket_observation': None, 'plan': None, 'result': None, 'trial_id': None,
            'loaded': False, 'events': [], 'error': None, 'trial_counter': 0, 'arm10_loading_boxes': []}

    def _event(self, kind, **fields):
        row = {'event_id': str(uuid.uuid4()), 'trial_id': self._state['trial_id'], 'kind': kind,
            'timestamp_utc': now(), 'scene_revision': self._state['revision'],
            'scene_fingerprint': digest(self._state['scene']), 'data_origin': 'synthetic',
            'physically_verified': False, **fields}
        self._state['events'].append(row)
        self._state['events'] = self._state['events'][-100:]

    def _busy(self):
        return self._thread is not None and self._thread.is_alive()

    def _available(self):
        if self._closed:
            return []
        if self._busy():
            return ['stop']
        actions = ['set_board', 'set_bucket', 'select_checkpoint', 'reset']
        if self._state['board_confirmed'] and self._state['bucket_confirmed']:
            actions.append('scan_bucket')
        if self._state['bucket_observation'] is not None:
            actions.append('plan')
        if self._state['plan'] is not None:
            actions.append('confirm_loaded')
            if self._state['loaded'] and self._state['phase'] == 'ready_to_drop':
                actions.append('grip_drop')
        return actions

    def status(self):
        with self.lock:
            return copy.deepcopy({**self._state, 'busy': self._busy(), 'available_actions': self._available(),
                'scene_fingerprint': digest(self._state['scene']), 'mode': 'simulation_only',
                'hardware_enabled': False, 'native_paths_checked': False,
                'missing_calibrations': MISSING, 'checkpoint_id': self._checkpoint.get('checkpoint_id'),
                'checkpoint_validation': 'verified by numerical worker before learned planning',
                'scope': 'Numerical rehearsal only; arm poses are endpoint checks and scan is synthetic scene ground truth'})

    def _setup_phase(self):
        return ('position_board' if not self._state['board_confirmed'] else
                'position_bucket' if not self._state['bucket_confirmed'] else 'ready_to_scan')

    def _invalidate(self):
        self._generation += 1
        self._state['revision'] += 1
        self._state.update(board_pose=None, arm10_loading_boxes=[], bucket_observation=None, plan=None, result=None,
                           loaded=False, trial_id=None, error=None)
        self._state['phase'] = self._setup_phase()

    def _worker(self, payload, generation, cancel):
        if self.runner is not None:
            return self.runner(payload, cancel)
        # Child gets no inherited Viam/audio credentials and imports only numerical code.
        env = {'PATH': os.environ.get('PATH', ''), 'PYTHONDONTWRITEBYTECODE': '1',
               'OPENBLAS_NUM_THREADS': '1', 'OMP_NUM_THREADS': '1'}
        process = subprocess.Popen([sys.executable, str(self.root/'sircus/operations/numerical_worker.py')],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)
        with self.lock:
            if generation != self._generation or cancel.is_set():
                process.kill()
            self._process = process
        try:
            stdout, _ = process.communicate(json.dumps(payload, allow_nan=False), timeout=self.worker_timeout)
        except subprocess.TimeoutExpired:
            process.kill(); process.communicate()
            raise ValueError('Numerical operation timed out; no robot action was sent') from None
        finally:
            with self.lock:
                if self._process is process:
                    self._process = None
        if len(stdout) > 5_000_000:
            raise ValueError('Numerical response exceeded its size limit')
        try:
            result = json.loads(stdout)
        except (ValueError, TypeError):
            raise ValueError('Numerical worker unavailable; no robot action was sent') from None
        if process.returncode or result.get('error'):
            raise ValueError(str(result.get('error', 'Numerical operation rejected'))[:400])
        return result

    def _start(self, operation, payload, complete):
        self._generation += 1
        generation = self._generation
        self._cancel = threading.Event()
        cancel = self._cancel
        self._state.update(phase=operation, error=None)
        def run():
            try:
                result = self._worker(payload, generation, cancel)
                with self.lock:
                    if generation == self._generation and not cancel.is_set() and not self._closed:
                        complete(result)
            except Exception as exc:
                with self.lock:
                    if generation == self._generation and not cancel.is_set() and not self._closed:
                        self._state.update(phase='error', loaded=False, error=str(exc)[:400])
                        self._event('operation_failed', operation=operation, error=self._state['error'])
            finally:
                with self.lock:
                    if self._thread is threading.current_thread():
                        self._thread = None
        self._thread = threading.Thread(target=run, daemon=True, name='sircus-guided-numerical')
        self._thread.start()

    def _save(self):
        self.data_root.mkdir(parents=True, exist_ok=True)
        path = self.data_root/(self._state['trial_id']+'.json')
        value = {'schema': 'sircus-guided-synthetic-trial-v1', 'saved_at_utc': now(), **self.status(),
                 'busy': False, 'available_actions': [], 'record_scope': 'archival terminal trial snapshot'}
        temporary = path.with_suffix('.tmp')
        temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+'\n')
        temporary.replace(path)

    def action(self, payload):
        if not isinstance(payload, dict) or not isinstance(payload.get('action'), str):
            raise ValueError('Action JSON object required')
        name = payload['action']
        expected = {'set_board': {'action','center_mm','rpy_deg'}, 'set_bucket': {'action','center_xy_mm'},
            'scan_bucket': {'action'}, 'plan': {'action','mode','seed','budget'},
            'confirm_loaded': {'action'}, 'grip_drop': {'action'}, 'stop': {'action'},
            'reset': {'action'}, 'select_checkpoint': {'action','checkpoint'}}
        if name not in expected or set(payload)-expected[name]:
            raise ValueError('Unsupported action or fields; hardware execution is unavailable')
        with self.lock:
            if self._closed:
                raise ValueError('Guided rehearsal is closed')
            if name == 'stop':
                self._cancel.set(); self._generation += 1
                if self._process is not None and self._process.poll() is None:
                    self._process.terminate()
                self._state.update(phase='stopped', loaded=False, plan=None, error=None)
                self._event('cancelled', operation_scope='numerical_only')
                if self._state['trial_id']:
                    self._save()
                return self.status()
            if self._busy():
                raise ValueError('A numerical operation is pending; stop it before changing the setup')
            if name not in self._available():
                raise ValueError('Complete the preceding setup step before '+name)
            if name == 'reset':
                self._state.update(scene=copy.deepcopy(self._default_scene), board_confirmed=False, bucket_confirmed=False)
                self._invalidate(); self._event('reset')
            elif name == 'set_board':
                center = vector(payload.get('center_mm'), 3, 'Board center', 2000)
                angles = vector(payload.get('rpy_deg'), 3, 'Board angles', 180)
                if not 50 <= center[2] <= 900:
                    raise ValueError('Board height must be 50..900 mm in simulation')
                self._state['scene']['board'].update(center_mm=center, rpy_deg=angles)
                self._state['board_confirmed'] = True
                self._invalidate(); self._event('board_target_set', source='operator_simulation_input')
            elif name == 'set_bucket':
                position = vector(payload.get('center_xy_mm'), 2, 'Bucket position', 2000)
                self._state['scene']['bin']['center_xy_mm'] = position
                self._state['bucket_confirmed'] = True
                self._invalidate(); self._event('bucket_position_set', source='operator_simulation_input')
            elif name == 'select_checkpoint':
                value = payload.get('checkpoint')
                if not isinstance(value, dict) or value.get('schema') != 'sircus-scene-conditioned-ridge-v1':
                    raise ValueError('Scene-conditioned checkpoint required')
                if len(json.dumps(value, allow_nan=False).encode()) > 800_000:
                    raise ValueError('Checkpoint too large')
                self._checkpoint = copy.deepcopy(value)
                self._invalidate(); self._event('checkpoint_selected', checkpoint_id=value.get('checkpoint_id'), validation='pending plan worker')
            elif name == 'scan_bucket':
                self._state.update(plan=None, result=None, loaded=False, bucket_observation=None, trial_id=None)
                revision = self._state['revision']; fingerprint = digest(self._state['scene'])
                self._event('scan_requested', source='synthetic_scene_fixture', physical_scan_pose='uncalibrated; no arm moved')
                def scanned(result):
                    observation = {**result['bucket_observation'], 'observed_at_utc': now(),
                                   'scene_revision': revision, 'scene_fingerprint': fingerprint}
                    self._state.update(board_pose=result['board_pose'], arm10_loading_boxes=result.get('arm10_loading_boxes', []),
                                       bucket_observation=observation, phase='ready_to_plan')
                    self._event('bucket_scan_completed', observation=observation)
                self._start('scanning', {'op':'scan','scene':copy.deepcopy(self._state['scene'])}, scanned)
            elif name == 'plan':
                mode = payload.get('mode', 'fixed_board'); seed = payload.get('seed', 42); budget = payload.get('budget', 8)
                if mode not in ('fixed_board','learned_board') or type(seed) is not int or not 0 <= seed < 2**32 or type(budget) is not int or not 1 <= budget <= 32:
                    raise ValueError('Use fixed_board/learned_board, uint32 seed and budget 1..32')
                observation = self._state['bucket_observation']
                if observation['scene_fingerprint'] != digest(self._state['scene']):
                    raise ValueError('Bucket scan is stale; scan the current layout again')
                self._state.update(plan=None, result=None, loaded=False, trial_id=None)
                def planned(result):
                    result.update(scene_fingerprint=digest(self._state['scene']), scene_revision=self._state['revision'], mode=mode)
                    self._state.update(plan=result, phase='ready_to_load')
                    self._event('plan_ready', mode=mode, source='numerical_evaluation', predicted_outcome=result['result']['outcome'])
                self._start('planning', {'op':'plan','scene':copy.deepcopy(self._state['scene']),
                    'checkpoint':copy.deepcopy(self._checkpoint),'mode':mode,'seed':seed,'budget':budget}, planned)
            elif name == 'confirm_loaded':
                self._state.update(loaded=True, phase='ready_to_drop')
                self._event('ball_loaded_acknowledged', source='operator_simulation_acknowledgement', grasp_verified=False)
            elif name == 'grip_drop':
                plan = self._state['plan']
                if plan['scene_fingerprint'] != digest(self._state['scene']):
                    raise ValueError('Plan is stale; scan and plan again')
                self._state.update(trial_id=str(uuid.uuid4()), result=None, loaded=False)
                self._event('grip_requested', execution='simulated; no tool actuation')
                self._event('drop_position_requested', release_coordinates=plan['release_coordinates'], native_path_checked=False)
                self._state['trial_counter'] += 1
                trial_seed = (plan['seed']+104729*self._state['trial_counter']) % 2**32
                self._event('trial_seed_selected', seed=trial_seed, trial_counter=self._state['trial_counter'],
                            planning_mode=plan['mode'], checkpoint_id=plan.get('checkpoint_id'))
                def completed(result):
                    result.update(planning_mode=plan['mode'], checkpoint_id=plan.get('checkpoint_id'),
                                  planning_inference=plan.get('inference'), trial_counter=self._state['trial_counter'])
                    self._state.update(result=result, phase='complete')
                    self._event('release_simulated', release_coordinates=result['release_coordinates'],
                                board_pose_source=result['board_pose']['source'])
                    self._event('trial_completed', outcome=result['result']['outcome'],
                                success=result['result']['success'], source='versioned_synthetic_scoring')
                    self._save()
                self._start('running', {'op':'execute','scene':copy.deepcopy(plan['selected_scene']),
                    'release_action':copy.deepcopy(plan['release_action']),'seed':trial_seed}, completed)
            return self.status()

    def close(self):
        with self.lock:
            self._closed = True; self._cancel.set(); self._generation += 1
            process, thread = self._process, self._thread
            if process is not None and process.poll() is None:
                process.terminate()
        if thread is not None:
            thread.join(timeout=1)
        with self.lock:
            process = self._process or process
        # Terminate only the numerical process we own; do not leave a stuck child
        # behind if a C-extension delays SIGTERM during application shutdown.
        if process is not None and process.poll() is None:
            process.kill()
            try:
                process.wait(timeout=1)
            except subprocess.TimeoutExpired:
                pass
        if thread is not None:
            thread.join(timeout=.5)
