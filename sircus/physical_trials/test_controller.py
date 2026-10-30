"""Physical operator invariants exercised against an independent, in-memory adapter.

These tests never instantiate a Viam client or contact robot hardware.
"""
import asyncio
import copy
from datetime import datetime, timezone
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

import controller as controller_module
from controller import Controller, TrialError, pose_matches


def utc():
    return datetime.now(timezone.utc).isoformat()


class FakeAdapter:
    def __init__(self):
        self.calls = []
        self.arm_pose = dict(x=400.0, y=80.0, z=300.0,
                             o_x=0.0, o_y=1.0, o_z=0.0, theta=180.0)
        self.board_pose = dict(x=650.0, y=0.0, z=220.0,
                               o_x=0.0, o_y=0.0, o_z=1.0, theta=0.0)
        self.arm_joints = [0.0] * 6
        self.board_joints = [0.0] * 6
        self.arm_moving = False
        self.board_moving = False
        self.snapshot_error = None
        self.recording_error = None
        self.recording_ready = True
        self.frame_age = 0.0
        self.recording_active = False
        self.recording_trial_id = None
        self.move_gate = None
        self.move_started = asyncio.Event()
        self.release_gate = None
        self.release_started = asyncio.Event()
        self.saved_metadata = []
        self.close_result = True
        self.stop_result = {'planner_cancellation_verified': True}

    async def connect(self):
        self.calls.append(('connect',))
        return {'connected': True, 'machine': 'sfhack10', 'mock': True}

    async def snapshot(self):
        if self.snapshot_error:
            raise RuntimeError(self.snapshot_error)
        return {'captured_at': utc(), 'arms': {
            'arm': {'pose': copy.deepcopy(self.arm_pose),
                    'joints': list(self.arm_joints), 'is_moving': self.arm_moving},
            'arm-board': {'pose': copy.deepcopy(self.board_pose),
                          'joints': list(self.board_joints), 'is_moving': self.board_moving}}}

    async def camera_frame(self):
        return b'mock-jpeg-frame', 'image/jpeg', {'captured_at': utc()}

    async def move_pose(self, pose):
        self.calls.append(('move_pose', copy.deepcopy(pose)))
        self.arm_moving = True
        self.move_started.set()
        try:
            if self.move_gate is not None:
                await self.move_gate.wait()
            self.arm_pose = copy.deepcopy(pose)
            return True
        finally:
            self.arm_moving = False

    async def open_gripper(self):
        self.calls.append(('open_gripper',))
        self.release_started.set()
        if self.release_gate is not None:
            await self.release_gate.wait()
        return True

    async def close_gripper(self):
        self.calls.append(('close_gripper',))
        return copy.deepcopy(self.close_result)

    async def stop(self):
        self.calls.append(('stop_dropper',))
        self.arm_moving = False
        return copy.deepcopy(self.stop_result)

    async def start_recording(self, trial_id, trial_dir, metadata):
        self.calls.append(('start_recording', trial_id))
        self.recording_trial_id = trial_id
        self.recording_active = True
        self.trial_dir = Path(trial_dir)
        self.trial_dir.mkdir(parents=True, exist_ok=True)
        return await self.recording_status()

    async def recording_status(self):
        return {'trial_id':self.recording_trial_id,'active': self.recording_active,
                'ready': self.recording_active and self.recording_ready and not self.recording_error,
                'frame_count': int(self.recording_active and self.recording_ready),
                'last_frame_monotonic': time.monotonic() - self.frame_age,
                'error': self.recording_error}

    async def finish_recording(self):
        self.calls.append(('finish_recording',))
        self.recording_active = False
        return {'active': False, 'ready': False, 'frame_count': 1}

    async def save_trial(self, trial_dir, metadata):
        self.calls.append(('save_trial', str(trial_dir)))
        self.saved_metadata.append(copy.deepcopy(metadata))
        return {'local_saved': True, 'cloud_uploaded': False, 'mock': True}

    async def close(self):
        self.calls.append(('close_client',))


class ControllerSafetyTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.timings = patch.multiple(controller_module, SETTLE_SECONDS=0.02,
                                      SAMPLE_SECONDS=0.005, POST_RELEASE_SECONDS=0.01)
        self.timings.start()
        self.addCleanup(self.timings.stop)
        if hasattr(controller_module, 'SETTLE_TIMEOUT_SECONDS'):
            timeout = patch.object(controller_module, 'SETTLE_TIMEOUT_SECONDS', 0.2)
            timeout.start()
            self.addCleanup(timeout.stop)
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.adapter = FakeAdapter()
        self.controller = Controller(self.adapter, self.root, mode='dry_run')
        await self.controller.start()

    async def asyncTearDown(self):
        await self.controller.close()
        self.temp.cleanup()

    async def drain(self):
        task = getattr(self.controller, 'task', None)
        if task is not None:
            try:
                await asyncio.wait_for(asyncio.shield(task), timeout=8)
            except asyncio.CancelledError:
                pass

    async def command(self, action, **values):
        result = await self.controller.action({'action': action, **values})
        await self.drain()
        return result

    async def calibrate(self):
        await self.command('teach_loading')
        self.adapter.arm_pose['z'] = 450.0
        await self.command('teach_drop')
        await self.command('configure', reference_z_mm=100.0,
                           min_height_mm=50.0, max_height_mm=400.0,
                           confirmed=True)
        await self.command('close_gripper')

    async def ready(self, height=150.0):
        await self.calibrate()
        await self.command('go_drop', height_mm=height)
        state = await self.controller.status()
        self.assertIn('drop', state['allowed_actions'], state)
        return state

    async def assert_rejected(self, action, **values):
        before = list(self.adapter.calls)
        try:
            await self.controller.action({'action': action, **values})
            await self.drain()
        except TrialError:
            pass
        new = self.adapter.calls[len(before):]
        self.assertFalse(any(call[0] in ('move_pose', 'open_gripper') for call in new), new)

    async def test_missing_calibration_never_moves_or_releases(self):
        state = await self.controller.status()
        self.assertTrue(state['calibration_missing'])
        await self.assert_rejected('go_loading')
        await self.assert_rejected('go_drop', height_mm=150)
        await self.assert_rejected('drop')
        self.assertFalse(any(x[0] in ('move_pose', 'open_gripper') for x in self.adapter.calls))

    async def test_opposite_tool_directions_do_not_match_when_theta_is_zero(self):
        up = dict(x=1, y=2, z=3, o_x=0, o_y=0, o_z=1, theta=0)
        down = dict(up, o_z=-1)
        self.assertFalse(pose_matches(up, down), 'Viam orientation vector is not an axis-angle rotation')

    async def test_height_rejects_nonfinite_boolean_and_out_of_range(self):
        await self.calibrate()
        for value in (-1, 49.9, 400.1, float('nan'), float('inf'), -float('inf'), True):
            with self.subTest(height=value):
                await self.assert_rejected('go_drop', height_mm=value)
        self.assertFalse(any(x[0] == 'move_pose' for x in self.adapter.calls))

    async def test_height_changes_only_z_and_uses_calibrated_reference(self):
        await self.ready(height=123.0)
        requested = [x[1] for x in self.adapter.calls if x[0] == 'move_pose'][-1]
        self.assertEqual(requested['z'], 223.0)
        for key, value in dict(x=400.0, y=80.0, o_x=0.0, o_y=1.0, o_z=0.0, theta=180.0).items():
            self.assertEqual(requested[key], value)

    async def test_failed_grab_driver_report_does_not_enable_a_drop_trial(self):
        await self.calibrate()
        self.adapter.close_result = {'command': 'gripper.grab', 'grabbed': False}
        await self.command('close_gripper')
        state = await self.controller.status()
        self.assertFalse(state['gripper_closed_commanded'])
        self.assertNotIn('go_drop', state['allowed_actions'])
        await self.assert_rejected('go_drop', height_mm=150)
        self.assertFalse(any(x[0] == 'move_pose' for x in self.adapter.calls))

    async def test_drop_rechecks_intended_pose_instead_of_trusting_old_ready(self):
        await self.ready()
        self.adapter.arm_pose['x'] += 15.0
        await self.assert_rejected('drop')
        self.assertFalse(any(x[0] == 'open_gripper' for x in self.adapter.calls))

    async def test_drop_rejects_external_board_motion_or_changed_joints(self):
        await self.ready()
        self.adapter.board_joints[2] += 2.0
        await self.assert_rejected('drop')
        self.assertFalse(any(x[0] == 'open_gripper' for x in self.adapter.calls))

    async def test_drop_rejects_stale_or_failed_recording(self):
        await self.ready()
        self.adapter.frame_age = 30.0
        await self.assert_rejected('drop')
        self.adapter.frame_age = 0.0
        self.adapter.recording_error = 'camera disconnected'
        await self.assert_rejected('drop')
        self.assertFalse(any(x[0] == 'open_gripper' for x in self.adapter.calls))

    async def test_failed_readback_never_releases(self):
        await self.ready()
        self.adapter.snapshot_error = 'robot disconnected'
        await self.assert_rejected('drop')
        self.assertFalse(any(x[0] == 'open_gripper' for x in self.adapter.calls))

    async def test_new_external_dropper_motion_prevents_another_move_command(self):
        await self.calibrate()
        self.adapter.arm_moving = True
        # The controller's cached UI snapshot is still stationary here; the
        # command handler must fetch a new state before sending a trajectory.
        await self.assert_rejected('go_drop', height_mm=150)
        self.assertFalse(any(x[0] == 'move_pose' for x in self.adapter.calls))

    async def test_recording_failure_never_enables_drop(self):
        await self.calibrate()
        self.adapter.recording_error = 'camera stream failed'
        await self.command('go_drop', height_mm=150)
        state = await self.controller.status()
        self.assertNotIn('drop', state['allowed_actions'])
        self.assertIn('save_trial', state['allowed_actions'])
        self.assertEqual(state['trial']['status'], 'aborted')
        self.assertFalse(any(x[0] == 'open_gripper' for x in self.adapter.calls))

    async def test_restart_retains_interrupted_trial_without_resuming_actions(self):
        state = await self.ready()
        trial_id = state['trial']['id']
        await self.controller.close()
        self.adapter = FakeAdapter()
        self.controller = Controller(self.adapter, self.root, mode='dry_run')
        await self.controller.start()
        state = await self.controller.status()
        self.assertEqual(state['trial']['id'], trial_id)
        self.assertEqual(state['trial']['status'], 'interrupted')
        self.assertNotIn('drop', state['allowed_actions'])
        self.assertIn('save_trial', state['allowed_actions'])
        self.assertFalse(any(x[0] in ('move_pose', 'open_gripper', 'close_gripper', 'start_recording')
                             for x in self.adapter.calls))

    async def test_stop_cancels_pending_motion_and_never_opens_tools(self):
        await self.calibrate()
        self.adapter.move_gate = asyncio.Event()
        await self.controller.action({'action': 'go_drop', 'height_mm': 150})
        await asyncio.wait_for(self.adapter.move_started.wait(), 3)
        with self.assertRaises(TrialError):
            await self.controller.action({'action': 'go_loading'})
        await self.controller.action({'action': 'stop'})
        await self.drain()
        self.adapter.move_gate.set()
        await asyncio.sleep(0.02)
        names = [x[0] for x in self.adapter.calls]
        self.assertIn('stop_dropper', names)
        self.assertNotIn('open_gripper', names)
        self.assertNotIn('start_recording', names)
        state = await self.controller.status()
        self.assertNotIn('drop', state['allowed_actions'])

    async def test_stop_generation_blocks_late_move_continuation(self):
        await self.calibrate()
        entered = asyncio.Event()
        async def cancellation_resistant_move(pose):
            self.adapter.calls.append(('move_pose', dict(pose)))
            entered.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                # A transport finishing its RPC despite cancellation must not
                # resume the pending prepare/record/release workflow.
                self.adapter.arm_pose = dict(pose)
        self.adapter.move_pose = cancellation_resistant_move
        await self.controller.action({'action':'go_drop','height_mm':150})
        await entered.wait()
        await self.controller.action({'action':'stop'})
        await self.drain()
        self.assertNotIn('start_recording',[c[0] for c in self.adapter.calls])
        self.assertNotIn('open_gripper',[c[0] for c in self.adapter.calls])
        self.assertNotIn('drop',(await self.controller.status())['allowed_actions'])

    async def test_unverified_stop_latches_fault_until_a_verified_stop(self):
        state = await self.ready()
        trial_id = state['trial']['id']
        self.adapter.stop_result = {'planner_cancellation_verified': False,
                                    'cancel_errors': ['mock planner cancellation unavailable']}
        await self.controller.action({'action': 'stop'})
        state = await self.controller.status()
        self.assertTrue(self.controller.fault)
        self.assertIn('Stop not confirmed', state['error'])
        self.assertNotIn('drop', state['allowed_actions'])
        # Saving an annotation must not quietly unlock motion while the fault
        # remains. An explicit, newly verified Stop is required to recover.
        await self.command('save_trial', trial_id=trial_id, outcome='unsure')
        state = await self.controller.status()
        self.assertNotIn('go_loading', state['allowed_actions'])
        self.assertNotIn('go_drop', state['allowed_actions'])
        self.assertNotIn('open_gripper', state['allowed_actions'])
        self.adapter.stop_result = {'planner_cancellation_verified': True}
        await self.controller.action({'action': 'stop'})
        self.assertFalse(self.controller.fault)
        self.assertIn('go_loading', (await self.controller.status())['allowed_actions'])

    async def test_outcome_is_saved_only_against_completed_trial_identity(self):
        state = await self.ready(height=180)
        trial_id = state['trial']['id']
        with self.assertRaises(TrialError):
            await self.controller.action({'action': 'save_trial', 'trial_id': trial_id, 'outcome': 'captured'})
        await self.command('drop')
        with self.assertRaises(TrialError):
            await self.controller.action({'action': 'save_trial', 'trial_id': 'another-trial', 'outcome': 'captured'})
        await self.command('save_trial', trial_id=trial_id, outcome='missed',
                           miss_direction='left', notes='Observed one bounce left of the rim')
        self.assertEqual(len(self.adapter.saved_metadata), 1)
        metadata = self.adapter.saved_metadata[0]
        serialized = json.dumps(metadata)
        self.assertIn(trial_id, serialized)
        self.assertIn('missed', serialized)
        self.assertIn('Observed one bounce left of the rim', serialized)
        self.assertIn('180', serialized)
        self.assertIn('measured_release_pose', metadata)
        self.assertTrue(metadata['measured_release_pose'])
        self.assertTrue(any('timestamp' in key or 'utc' in key or key == 'events'
                            for key in metadata), metadata.keys())
        names = [x[0] for x in self.adapter.calls]
        self.assertLess(names.index('start_recording'), names.index('open_gripper'))
        self.assertLess(names.index('open_gripper'), names.index('finish_recording'))


if __name__ == '__main__':
    unittest.main()
