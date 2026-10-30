"""Adversarial deterministic-controller/adapter regressions; all I/O is mocked."""
import asyncio
import copy
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import controller as module
from controller import Controller,TrialError
import test_controller as base
import test_fixed_cycle as cycle
import test_viam_adapter as sdk


class ReliabilityTests(unittest.IsolatedAsyncioTestCase):
    asyncSetUp=cycle.FixedCycleTests.asyncSetUp
    asyncTearDown=cycle.FixedCycleTests.asyncTearDown
    command=base.ControllerSafetyTests.command
    drain=base.ControllerSafetyTests.drain
    teach=cycle.FixedCycleTests.teach
    ready=base.ControllerSafetyTests.ready
    calibrate=base.ControllerSafetyTests.calibrate

    async def test_journal_failure_cannot_prevent_stop_or_leave_stopping_latched(self):
        await self.ready();before=len(self.adapter.calls)
        with patch.object(self.controller,'_event',side_effect=OSError('disk full')),patch.object(self.controller,'_persist',side_effect=OSError('disk full')):
            await self.controller.action({'action':'stop'})
        calls=self.adapter.calls[before:]
        self.assertIn(('stop_dropper',),calls)
        self.assertFalse(any(c[0] in ('open_gripper','close_gripper') for c in calls))
        state=self.controller.status_snapshot();self.assertFalse(state['busy']);self.assertFalse(self.controller.stopping)
        self.assertIsNotNone(state['journal_error']);self.assertEqual(state['trial']['status'],'aborted')

    async def test_fault_and_wrong_or_nonfinite_recording_identity_block_release(self):
        await self.ready();self.controller.fault=True
        self.assertNotIn('drop',self.controller.status_snapshot()['allowed_actions'])
        with self.assertRaises(TrialError):await self.controller.action({'action':'drop'})
        self.controller.fault=False;valid=copy.deepcopy(self.controller.recording)
        for update in ({'trial_id':'other-trial'},{'last_frame_monotonic':float('nan')},{'last_frame_monotonic':time.monotonic()+100},{'ready':'yes'}):
            self.controller.recording={**valid,**update}
            self.assertTrue(any('Recording' in x for x in self.controller._drop_blockers()))
        self.assertFalse(any(c[0]=='open_gripper' for c in self.adapter.calls))

    async def test_readback_and_connect_loss_invalidate_both_gripper_caches(self):
        await self.calibrate();self.assertTrue(self.controller.gripper_closed)
        self.adapter.snapshot_error='unavailable'
        with self.assertRaises(RuntimeError):await self.controller._snapshot()
        self.adapter.snapshot_error=None;await self.controller._snapshot()
        self.assertFalse(self.controller.gripper_closed);self.assertEqual(self.controller.gripper_command_state,'unknown')
        self.assertNotIn('go_drop',self.controller.status_snapshot()['allowed_actions'])
        self.controller.poll_task.cancel();await asyncio.gather(self.controller.poll_task,return_exceptions=True)
        self.controller.gripper_closed=True;self.controller.gripper_command_state='open'
        async def failed_connect():raise RuntimeError('offline')
        self.adapter.connect=failed_connect;poll=asyncio.create_task(self.controller._poll())
        await asyncio.sleep(0);poll.cancel();await asyncio.gather(poll,return_exceptions=True)
        self.assertFalse(self.controller.gripper_closed);self.assertEqual(self.controller.gripper_command_state,'unknown')

    async def test_stale_delayed_action_after_stop_is_rejected(self):
        epoch=self.controller.status_snapshot()['action_epoch']
        await self.controller.action({'action':'stop','expected_action_epoch':-1})
        with self.assertRaisesRegex(TrialError,'Stale action'):
            await self.controller.action({'action':'set_ball','ball_id':'ball-pink','expected_action_epoch':epoch})
        current=self.controller.status_snapshot()['action_epoch']
        await self.command('set_ball',ball_id='ball-pink',expected_action_epoch=current)
        self.assertGreater(self.controller.status_snapshot()['action_epoch'],current)

    async def test_cancel_resistant_action_cannot_block_stop_or_continue_cycle(self):
        await self.teach();started=asyncio.Event();release=asyncio.Event()
        async def stalled_move(pose):
            self.adapter.calls.append(('move_pose',copy.deepcopy(pose)));started.set()
            while not release.is_set():
                try:await release.wait()
                except asyncio.CancelledError:pass
        self.adapter.move_pose=stalled_move
        await self.controller.action({'action':'load_drop_cycle','fixed_setup_confirmed':True});await started.wait()
        try:
            with patch.object(module,'CANCEL_WAIT_SECONDS',.02):
                await asyncio.wait_for(self.controller.action({'action':'stop'}),.3)
            self.assertTrue(self.controller.fault);self.assertFalse(self.controller.stopping)
            self.assertEqual(self.controller.trial['status'],'aborted')
            self.assertNotIn('load_drop_cycle',self.controller.status_snapshot()['allowed_actions'])
        finally:release.set();await self.drain()
        self.assertFalse(any(c[0] in ('open_gripper','close_gripper') for c in self.adapter.calls))
        self.assertEqual(self.controller.trial['status'],'aborted')

    async def test_stop_rpc_timeout_and_cancelled_stop_never_claim_recovered(self):
        started=asyncio.Event();release=asyncio.Event()
        async def delayed_stop():
            started.set()
            while not release.is_set():
                try:await release.wait()
                except asyncio.CancelledError:pass
            return {'planner_cancellation_verified':True}
        self.adapter.stop=delayed_stop
        try:
            with patch.object(module,'STOP_TIMEOUT_SECONDS',.02):
                await asyncio.wait_for(self.controller.action({'action':'stop'}),.3)
            self.assertTrue(self.controller.fault);self.assertFalse(self.controller.stopping)
            task=asyncio.create_task(self.controller.action({'action':'stop'}));await asyncio.sleep(0)
            task.cancel();await asyncio.gather(task,return_exceptions=True)
            self.assertTrue(self.controller.fault);self.assertFalse(self.controller.stopping)
        finally:
            release.set();await asyncio.sleep(.01)

    async def test_automatic_abort_requires_stationary_stop_confirmation(self):
        await self.teach();self.adapter.close_result=False
        async def dishonest_stop():self.adapter.arm_moving=True;return {'planner_cancellation_verified':True}
        self.adapter.stop=dishonest_stop
        with patch.object(module,'LOAD_WAIT_SECONDS',.01):
            await self.command('load_drop_cycle',fixed_setup_confirmed=True)
        self.assertTrue(self.controller.fault);self.assertEqual(self.controller.trial['status'],'aborted')
        self.assertIn('stop not confirmed',self.controller.error)

    async def test_failed_calibration_writes_never_publish_unsaved_targets(self):
        await self.calibrate()
        previous=copy.deepcopy(self.controller.calibration)
        saved=self.controller.config_path.read_bytes()
        replacement=copy.deepcopy(previous['drop_pose']);replacement['z']+=10
        cases=[('configure_poses',dict(loading_pose=replacement,drop_pose=replacement,confirmed=True)),
               ('teach_loading',{}),('teach_drop',{}),
               ('configure',dict(reference_z_mm=10,min_height_mm=20,max_height_mm=300,confirmed=True))]
        self.adapter.arm_pose['z']+=20
        original=module.write_json
        def fail_calibration(path,value):
            if Path(path)==self.controller.config_path:raise OSError('disk full')
            return original(path,value)
        for name,payload in cases:
            with patch.object(module,'write_json',side_effect=fail_calibration):
                await self.command(name,**payload)
            self.assertEqual(self.controller.calibration,previous,name)
            self.assertEqual(self.controller.config_path.read_bytes(),saved,name)
            await self.command('stop')
            self.assertEqual(self.controller.calibration,previous,name)

    def test_corrupt_trial_journal_is_preserved_without_startup_crash(self):
        with tempfile.TemporaryDirectory() as directory:
            folder=Path(directory)/'trials'/'broken';folder.mkdir(parents=True)
            file=folder/'trial.json';file.write_text('{broken')
            controller=Controller(base.FakeAdapter(),directory)
            self.assertTrue(controller.fault);self.assertIsNotNone(controller.journal_error)
            self.assertEqual(file.read_text(),'{broken')


class AdapterReliabilityTests(unittest.IsolatedAsyncioTestCase):
    adapter=sdk.AdapterTests.adapter

    async def test_cancel_ack_with_remaining_owned_operation_is_not_verified(self):
        adapter,log=self.adapter()
        async def acknowledged_only(op):log.append('cancel:'+op)
        adapter.robot.cancel_operation=acknowledged_only
        task=asyncio.create_task(adapter.move_pose(dict(x=100.,y=0.,z=200.,o_x=0.,o_y=0.,o_z=-1.,theta=0.)))
        await asyncio.sleep(.01);result=await adapter.stop();await asyncio.gather(task,return_exceptions=True)
        self.assertFalse(result['planner_cancellation_verified'])
        self.assertEqual(result['remaining_owned_operations'],['only-panel-move'])
        self.assertTrue(adapter._move_active)

    async def test_delayed_connect_cannot_send_tool_command_after_stop(self):
        for method in ('open_gripper','close_gripper'):
            adapter,log=self.adapter();started=asyncio.Event();release=asyncio.Event();calls=0
            async def connect():
                nonlocal calls
                calls+=1
                if calls==1:started.set();await release.wait()
                return {}
            async def tool(**kwargs):log.append('forbidden_tool');return True
            adapter.connect=connect;adapter.gripper=SimpleNamespace(open=tool,grab=tool)
            task=asyncio.create_task(getattr(adapter,method)());await started.wait()
            await adapter.stop();release.set();await asyncio.gather(task,return_exceptions=True)
            self.assertNotIn('forbidden_tool',log)

    async def test_pending_cancel_resistant_tool_keeps_stop_unverified(self):
        adapter,log=self.adapter();started=asyncio.Event();release=asyncio.Event()
        async def late_open(**kwargs):
            started.set()
            while not release.is_set():
                try:await release.wait()
                except asyncio.CancelledError:pass
            log.append('late_already_issued_open')
        adapter.gripper=SimpleNamespace(open=late_open)
        task=asyncio.create_task(adapter.open_gripper());await started.wait();task.cancel()
        await asyncio.gather(task,return_exceptions=True)
        try:
            result=await adapter.stop()
            self.assertFalse(result['planner_cancellation_verified']);self.assertEqual(result['pending_owned_actuator_requests'],1)
            self.assertNotIn('late_already_issued_open',log)
        finally:release.set();await asyncio.sleep(.01)
        self.assertTrue((await adapter.stop())['planner_cancellation_verified'])

    async def test_completed_move_drops_old_operation_evidence(self):
        adapter,log=self.adapter()
        async def complete(**kwargs):adapter._move_operations.add('finished-old-move');return True
        adapter.motion.move=complete
        await adapter.move_pose(dict(x=100.,y=0.,z=200.,o_x=0.,o_y=0.,o_z=-1.,theta=0.))
        result=await adapter.stop()
        self.assertTrue(result['planner_cancellation_verified']);self.assertNotIn('cancel:finished-old-move',log)

if __name__=='__main__':unittest.main()
