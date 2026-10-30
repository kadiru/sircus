"""Explicit home reset: mock-only motion, preserved records, unchanged tools."""
import asyncio
import copy
import unittest
from controller import TrialError
import test_round_operator as rounds

class RoundHomeTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        await rounds.RoundOperatorTests.asyncSetUp(self)
        stop=self.adapter.stop
        async def recovery_stop(recover_idle=False):
            await stop()
            return {'planner_cancellation_verified':True,'recovery_evidence':{'mock':True}}
        self.adapter.stop=recovery_stop
    asyncTearDown=rounds.RoundOperatorTests.asyncTearDown
    teach=rounds.RoundOperatorTests.teach
    command=rounds.RoundOperatorTests.command
    drain=rounds.RoundOperatorTests.drain
    async def reset(self,**extra):
        return await self.controller.action({'action':'reset_round_home','confirmed':True,
            'expected_action_epoch':self.controller.action_epoch,**extra})
    def assert_tools_untouched(self):
        self.assertFalse(any(c[0] in ('open_gripper','close_gripper','move_board','start_recording') for c in self.adapter.calls))
    async def test_low_pickup_retracts_vertically_then_goes_home_and_retains_aborted_trial(self):
        grasp,approach=self.controller._pickup_poses('ball-green')
        self.adapter.arm_pose=copy.deepcopy(grasp)
        self.controller.trial={'id':'aborted-green','status':'aborted','saved':False,'events':[]}
        await self.reset();await self.drain()
        self.assertEqual(self.controller.phase,'idle',self.controller.error)
        self.assertEqual(self.controller.round_run['status'],'home_ready')
        self.assertEqual(self.controller.trial['status'],'aborted');self.assertTrue(self.controller.trial['saved'])
        moves=[c for c in self.adapter.calls if c[0].startswith('move_')]
        self.assertEqual(moves[0][0],'move_pose_linear');self.assertEqual(moves[0][1],approach)
        self.assertEqual(moves[-1][1],self.presets['rounds']['round-1']['dropper_arm']['pose'])
        self.assert_tools_untouched();self.assertFalse(self.audio_calls[-1]['enabled'])
        count=len(self.adapter.calls);await asyncio.sleep(.03);self.assertEqual(len(self.adapter.calls),count)
    async def test_approach_and_round2_drop_are_supported_empty_starts(self):
        _,approach=self.controller._pickup_poses('ball-green')
        for pose in (approach,self.presets['rounds']['round-2']['dropper_arm']['pose']):
            self.adapter.arm_pose=copy.deepcopy(pose)
            await self.reset();await self.drain()
            self.assertEqual(self.controller.round_run['status'],'home_ready',self.controller.error)
            self.assertEqual(self.adapter.arm_pose,self.presets['rounds']['round-1']['dropper_arm']['pose'])
        self.assert_tools_untouched()
    async def test_held_unknown_and_unrecognized_intermediate_pose_block_motion(self):
        for held,z in ((True,600),(None,600),(False,250)):
            self.held=held;self.adapter.arm_pose['z']=z;self.adapter.calls.clear()
            await self.reset();await self.drain()
            self.assertIsNone(self.controller.round_run)
            self.assertFalse(any(c[0].startswith('move_') for c in self.adapter.calls))
            self.assert_tools_untouched()
    async def test_failed_stop_missing_confirmation_and_stale_request_never_move(self):
        for extra in ({'confirmed':False},{'expected_action_epoch':-1}):
            with self.assertRaises(TrialError):await self.reset(**extra)
        async def fail(*a,**k):raise RuntimeError('mock stop failure')
        self.adapter.stop=fail
        with self.assertRaises(TrialError):await self.reset()
        self.assertTrue(self.controller.fault)
        self.assertFalse(any(c[0].startswith('move_') for c in self.adapter.calls))
        self.assert_tools_untouched()
    async def test_stop_during_home_move_cancels_without_starting_a_round(self):
        _,approach=self.controller._pickup_poses('ball-green');self.adapter.arm_pose=copy.deepcopy(approach)
        gate=asyncio.Event();started=asyncio.Event()
        async def move(pose):started.set();await gate.wait()
        self.adapter.move_pose=move
        await self.reset();await asyncio.wait_for(started.wait(),2)
        await self.command('stop')
        self.assertIsNone(self.controller.round_run)
        self.assert_tools_untouched()
    async def test_completed_unsaved_evidence_is_not_relabelled_aborted(self):
        self.controller.trial={'id':'completed-trial','status':'completed','saved':False,'events':[]}
        await self.reset();await self.drain()
        self.assertEqual(self.controller.trial['status'],'completed');self.assertTrue(self.controller.trial['saved'])
        self.assert_tools_untouched()
