"""Two-round UI backend tests. In-memory arms only; no SDK connection."""
import asyncio
import copy
import json
import unittest
from unittest.mock import patch
import controller as module
from controller import TrialError
import test_pickup_station as pickup_fixtures

class RoundOperatorTests(unittest.IsolatedAsyncioTestCase):
    asyncTearDown=pickup_fixtures.PickupStationTests.asyncTearDown
    drain=pickup_fixtures.PickupStationTests.drain
    command=pickup_fixtures.PickupStationTests.command
    teach=pickup_fixtures.PickupStationTests.teach
    async def asyncSetUp(self):
        await pickup_fixtures.PickupStationTests.asyncSetUp(self)
        await self.teach()
        self.board_gate=None;self.board_started=asyncio.Event();self.held=False;self.vacuum=True
        async def move_board(pose):
            self.adapter.calls.append(('move_board',copy.deepcopy(pose)))
            self.board_started.set()
            if self.board_gate:await self.board_gate.wait()
            self.adapter.board_pose=copy.deepcopy(pose)
        async def held():return self.held
        async def vacuum():return self.vacuum
        async def quiet():return {'method':'mock quiet'}
        self.adapter.move_board_pose=move_board;self.adapter.held_ball=held
        self.adapter.vacuum_holding=vacuum;self.adapter.verify_quiescent=quiet
        self.audio_calls=[]
        class Audio:
            def control(_,p):self.audio_calls.append(p)
            def stop(_, *a, **k):pass
            def emit(_, *a, **k):pass
        self.controller.audio=Audio()
        c=self.controller.calibration
        for i,ball in enumerate(('ball-gray','ball-purple','ball-pink','ball-green')):
            c['pickup_pose_provenance'][ball]=copy.deepcopy(c['pickup_pose_provenance']['ball-pink'])
            c['pickup_poses'][ball]={**c['pickup_poses']['ball-pink'],'y':-400.-70*i}
        rows={}
        for i,name in enumerate(('round-1','round-2')):
            rows[name]={'board_arm':{'pose':{**self.adapter.board_pose,'x':650.+30*i},'joints':[0.]*6,'pose_frame':'world','pose_component':'arm-board'},
                        'dropper_arm':{'pose':{**c['drop_pose'],'z':600.+100*i},'joints':[0.]*6,'pose_frame':'world','pose_component':'arm'}}
        self.presets={'schema':'sircus-taught-round-presets-v1','machine':'sfhack10','rounds':rows}
        self.path=self.root/'round-presets.pending.json';self.path.write_text(json.dumps(self.presets))
    async def start_round(self,name='round-1',**extra):
        return await self.controller.action({'action':'run_round','round_id':name,'sounds':True,'confirmed':True,
                'expected_action_epoch':self.controller.action_epoch,**extra})
    async def test_four_unique_trials_then_explicit_reload_second_round(self):
        await self.start_round();await self.drain()
        r=copy.deepcopy(self.controller.round_run)
        self.assertEqual(r['status'],'awaiting_reload',self.controller.error)
        self.assertEqual([t['ball_id'] for t in r['trials']],['ball-gray','ball-purple','ball-pink','ball-green'])
        self.assertEqual(len(set(t['trial_id'] for t in r['trials'])),4)
        self.assertTrue(all(t['outcome'] is None for t in r['trials']))
        self.assertEqual(self.audio_calls[-1]['enabled'],False)
        self.assertFalse(any(c[0]=='move_board' for c in self.adapter.calls))
        calls=len(self.adapter.calls);await asyncio.sleep(.02)
        self.assertEqual(len(self.adapter.calls),calls)  # No automatic Round 2.
        await self.start_round('round-2');await self.drain()
        self.assertEqual(self.controller.round_run['status'],'completed',self.controller.error)
        self.assertEqual(sum(c[0]=='move_board' for c in self.adapter.calls),1)
        self.assertEqual(sum(c[0]=='close_gripper' for c in self.adapter.calls),8)
        self.assertFalse(any(c[0]=='start_recording' for c in self.adapter.calls))
        self.assertEqual(self.controller.trial['round_id'],'round-2')
    async def test_missing_confirmation_stale_epoch_and_missing_presets_do_not_move(self):
        for extra in ({'confirmed':False},{'expected_action_epoch':-1},{'round_id':'round-3'}):
            with self.assertRaises(TrialError):await self.start_round(**extra)
        self.path.unlink()
        with self.assertRaises(TrialError):await self.start_round()
        self.assertFalse(any(c[0] in ('move_board','close_gripper') for c in self.adapter.calls))
    async def test_low_start_or_held_ball_or_lost_vacuum_blocks_before_transition(self):
        for problem in ('low','held','vacuum'):
            self.adapter.arm_pose=copy.deepcopy(self.controller.calibration['drop_pose'])
            self.held=problem=='held';self.vacuum=problem!='vacuum'
            if problem=='low':self.adapter.arm_pose['z']=200
            await self.start_round('round-2');await self.drain()
            self.assertEqual(self.controller.round_run['status'],'interrupted')
            self.assertFalse(any(c[0] in ('move_board','close_gripper','open_gripper') for c in self.adapter.calls))
    async def test_stop_during_board_transition_never_starts_pickup_and_blocks_conflicts(self):
        self.board_gate=asyncio.Event()
        await self.start_round('round-2');await asyncio.wait_for(self.board_started.wait(),2)
        with self.assertRaises(TrialError):await self.start_round()
        with self.assertRaises(TrialError):await self.controller.action({'action':'close_gripper'})
        await self.command('stop')
        self.assertEqual(self.controller.round_run['status'],'interrupted')
        self.assertFalse(any(c[0] in ('close_gripper','open_gripper') for c in self.adapter.calls))
        self.assertFalse(self.audio_calls[-1]['enabled'])
    async def test_failed_green_keeps_completed_trials_and_does_not_retry(self):
        original=self.adapter.move_pose
        async def fail_green(pose):
            if pose['y']==self.controller.calibration['pickup_poses']['ball-green']['y']:
                raise RuntimeError('mock readback unavailable')
            return await original(pose)
        self.adapter.move_pose=fail_green
        await self.start_round();await self.drain()
        self.assertEqual(self.controller.round_run['status'],'interrupted')
        self.assertEqual(len(self.controller.round_run['trials']),3)
        self.assertEqual(sum(c[0]=='close_gripper' for c in self.adapter.calls),3)
        self.assertEqual(self.controller.trial['ball_id'],'ball-green')
        self.assertEqual(self.controller.trial['status'],'aborted')
        self.assertIn('stop_dropper',[c[0] for c in self.adapter.calls])
    async def test_audio_failure_cannot_block_round_or_stop(self):
        def unavailable(p):raise RuntimeError('offline audio')
        self.controller.audio.control=unavailable
        await self.start_round();await self.drain()
        self.assertEqual(self.controller.round_run['status'],'awaiting_reload',self.controller.error)
        self.assertGreater(self.controller.audio_observer_errors,0)
    async def test_changed_joint_solution_stops_before_ball_commands(self):
        original=self.adapter.move_board_pose
        async def shifted(pose):
            await original(pose);self.adapter.board_joints[1]=3
        self.adapter.move_board_pose=shifted
        await self.start_round('round-2');await self.drain()
        self.assertEqual(self.controller.round_run['status'],'interrupted')
        self.assertFalse(any(c[0] in ('close_gripper','open_gripper') for c in self.adapter.calls))
    async def test_restart_never_resumes_a_pending_round(self):
        self.controller.round_run={'id':'mock-interrupted-round','status':'running','round_id':'round-2','trials':[]}
        self.controller._round_save()
        calls=len(self.adapter.calls)
        restarted=module.Controller(self.adapter,self.root,mode='dry_run')
        self.assertEqual(restarted.round_run['status'],'interrupted')
        self.assertTrue(restarted.fault)
        self.assertEqual(len(self.adapter.calls),calls)
        self.assertIsNone(restarted.task)
