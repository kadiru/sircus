import asyncio
import copy
import unittest
from unittest.mock import patch
import controller as module
from controller import TrialError
import test_controller as fixtures


class FixedCycleTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        await fixtures.ControllerSafetyTests.asyncSetUp(self)
        self.drop_patch = patch.object(module, 'DROP_SETTLE_SECONDS', .01)
        self.drop_patch.start()
        self.addCleanup(self.drop_patch.stop)
    asyncTearDown = fixtures.ControllerSafetyTests.asyncTearDown
    drain = fixtures.ControllerSafetyTests.drain
    command = fixtures.ControllerSafetyTests.command

    async def teach(self):
        await self.command('teach_loading')
        self.adapter.arm_pose.update(x=600., y=-150., z=600.)
        await self.command('teach_drop')

    async def launch(self):
        return await self.controller.action({'action':'load_drop_cycle','fixed_setup_confirmed':True})

    async def wait_phase(self, phase):
        for _ in range(200):
            if self.controller.phase == phase:return
            await asyncio.sleep(.005)
        self.fail('Did not reach '+phase)

    async def test_fixed_pose_cycle_without_metric_calibration(self):
        await self.teach()
        self.assertEqual(module.LOAD_WAIT_SECONDS, 2.)
        with patch.object(module,'LOAD_WAIT_SECONDS',.06):
            await self.launch(); await self.drain()
        calls=[c[0] for c in self.adapter.calls if c[0] not in ('connect',)]
        self.assertEqual(calls,['move_pose','start_recording','open_gripper','close_gripper','move_pose','open_gripper','finish_recording'])
        trial=self.controller.trial
        self.assertIsNone(trial['height_mm'])
        self.assertEqual(trial['target_pose'],self.controller.calibration['drop_pose'])
        self.assertEqual(trial['measured_release_pose'],trial['target_pose'])
        self.assertEqual(trial['status'],'completed')
        self.assertFalse(self.controller.calibration['confirmed'])
        events=trial['events'];start=next(e['monotonic'] for e in events if e['event']=='loading_countdown_started');end=next(e['monotonic'] for e in events if e['command']=='close_gripper' and e['event']=='command_sent')
        self.assertGreaterEqual(end-start,.06)
        await self.command('save_trial',trial_id=trial['id'],outcome='captured',notes='mock cycle')
        self.assertEqual(self.adapter.saved_metadata[0]['behavior'],'fixed_pose_cycle')

    async def test_confirmation_and_taught_poses_required(self):
        with self.assertRaises(TrialError):await self.launch()
        await self.teach()
        with self.assertRaises(TrialError):await self.controller.action({'action':'load_drop_cycle'})
        self.assertFalse(any(c[0]=='move_pose' for c in self.adapter.calls))

    async def test_stop_during_countdown_prevents_close_and_drop(self):
        await self.teach()
        await self.launch();await self.wait_phase('cycle_loading_countdown')
        self.assertIsNotNone((await self.controller.status())['loading_seconds_remaining'])
        with self.assertRaises(TrialError):await self.launch()
        await self.command('stop');await asyncio.sleep(.03)
        calls=[c[0] for c in self.adapter.calls]
        self.assertNotIn('close_gripper',calls)
        self.assertEqual(calls.count('move_pose'),1)
        self.assertEqual(calls.count('open_gripper'),1)
        self.assertEqual(self.controller.trial['status'],'aborted')
        self.assertIsNone(self.controller.loading_deadline)

    async def test_no_ball_aborts_before_drop_motion(self):
        await self.teach();self.adapter.close_result={'grabbed':False}
        with patch.object(module,'LOAD_WAIT_SECONDS',.01):await self.launch();await self.drain()
        self.assertEqual([c[0] for c in self.adapter.calls].count('move_pose'),1)
        self.assertEqual([c[0] for c in self.adapter.calls].count('open_gripper'),1)
        self.assertEqual(self.controller.trial['status'],'aborted')

    async def test_board_change_during_loading_aborts(self):
        await self.teach();await self.launch();await self.wait_phase('cycle_loading_countdown')
        self.adapter.board_joints[0]=4
        await self.drain()
        self.assertNotIn('close_gripper',[c[0] for c in self.adapter.calls])
        self.assertEqual(self.controller.trial['status'],'aborted')

    async def test_recording_failure_never_opens_gripper(self):
        await self.teach();self.adapter.recording_error='camera unavailable'
        await self.launch();await self.drain()
        self.assertNotIn('open_gripper',[c[0] for c in self.adapter.calls])
        self.assertEqual(self.controller.trial['status'],'aborted')

    async def test_stale_frames_block_release(self):
        await self.teach();self.adapter.frame_age=30
        with patch.object(module,'LOAD_WAIT_SECONDS',.01):await self.launch();await self.drain()
        self.assertEqual([c[0] for c in self.adapter.calls].count('open_gripper'),0)
        self.assertEqual(self.controller.trial['status'],'aborted')

    async def test_stop_after_cancel_resistant_move_never_opens(self):
        await self.teach();gate=asyncio.Event();started=asyncio.Event()
        async def late_move(pose):
            self.adapter.calls.append(('move_pose',copy.deepcopy(pose)));started.set()
            try:await gate.wait()
            except asyncio.CancelledError:await gate.wait()
        self.adapter.move_pose=late_move
        await self.launch();await started.wait()
        stopping=asyncio.create_task(self.command('stop'));await asyncio.sleep(.02);gate.set();await stopping
        self.assertNotIn('open_gripper',[c[0] for c in self.adapter.calls])
        self.assertEqual(self.controller.trial['status'],'aborted')

    async def test_practice_repeats_without_labels_and_preserves_each_trial(self):
        import json
        await self.teach()
        ids=[]
        with patch.object(module,'LOAD_WAIT_SECONDS',.01):
            for _ in range(2):
                await self.controller.action({'action':'load_drop_cycle','fixed_setup_confirmed':True,'practice_mode':True})
                await self.drain()
                t=self.controller.trial;ids.append(t['id'])
                self.assertTrue(t['saved']);self.assertIsNone(t['outcome'])
                self.assertEqual(t['label_status'],'unlabeled')
                self.assertEqual(t['status'],'completed')
                self.assertIn('load_drop_cycle',(await self.controller.status())['allowed_actions'])
        self.assertNotEqual(*ids)
        for tid in ids:
            t=json.loads((self.controller.root/'trials'/tid/'trial.json').read_text())
            self.assertEqual(t['status'],'completed');self.assertTrue(t['saved'])
        self.assertEqual(self.adapter.saved_metadata,[])
        self.assertEqual([c[0] for c in self.adapter.calls].count('move_pose'),4)
        self.assertEqual([c[0] for c in self.adapter.calls].count('open_gripper'),3)
        self.assertTrue(any(e['event']=='loading_open_skipped' for e in self.controller.trial['events']))

    async def test_keep_unlabeled_requires_matching_finished_trial(self):
        await self.teach();await self.launch();await self.wait_phase('cycle_loading_countdown')
        with self.assertRaises(TrialError):
            await self.controller.action({'action':'keep_unlabeled','trial_id':self.controller.trial['id']})
        await self.command('stop')
        with self.assertRaises(TrialError):
            await self.command('keep_unlabeled',trial_id='wrong')
        await self.command('keep_unlabeled',trial_id=self.controller.trial['id'])
        self.assertTrue(self.controller.trial['saved'])
        self.assertEqual(self.controller.trial['status'],'aborted')
        self.assertIsNone(self.controller.trial['outcome'])

    async def test_two_second_drop_hold_before_release(self):
        import time
        await self.teach()
        original_move=self.adapter.move_pose;original_open=self.adapter.open_gripper
        times={}; moves=0
        async def move(pose):
            nonlocal moves
            await original_move(pose);moves+=1
            if moves==2:times['arrived']=time.monotonic()
        async def opened():
            if moves==2:times['release']=time.monotonic()
            await original_open()
        self.adapter.move_pose=move;self.adapter.open_gripper=opened
        with patch.object(module,'LOAD_WAIT_SECONDS',.01), patch.object(module,'DROP_SETTLE_SECONDS',2.), patch.object(module,'SETTLE_TIMEOUT_SECONDS',4.):
            await self.launch();await self.drain()
        self.assertGreaterEqual(times['release']-times['arrived'],2.)
        self.assertEqual(self.controller.trial['drop_settle_seconds'],2.)
        self.assertEqual(self.controller.trial['status'],'completed')

    async def test_unknown_and_failed_close_do_not_count_as_open(self):
        await self.teach()
        self.assertEqual(self.controller.gripper_command_state, 'unknown')
        await self.command('open_gripper')
        self.assertEqual(self.controller.gripper_command_state, 'open')
        async def failed_close():
            raise RuntimeError('uncertain close result')
        self.adapter.close_gripper=failed_close
        await self.command('close_gripper')
        self.assertEqual(self.controller.gripper_command_state, 'unknown')
        await self.launch();await self.wait_phase('cycle_loading_countdown')
        self.assertEqual([c[0] for c in self.adapter.calls].count('open_gripper'),2)
        await self.command('stop')

    async def test_readback_loss_invalidates_open_cache(self):
        await self.command('open_gripper')
        self.adapter.snapshot_error='connection lost'
        with self.assertRaises(RuntimeError):await self.controller._snapshot()
        self.assertEqual(self.controller.gripper_command_state,'unknown')
