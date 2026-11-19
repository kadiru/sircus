"""Station pickup behavior using in-memory adapters only; no SDK or devices."""
import asyncio
import copy
import json
import unittest
from unittest.mock import patch

import controller as module
from controller import TrialError
import test_controller as fixtures


class PickupStationTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        await fixtures.ControllerSafetyTests.asyncSetUp(self)
        timing=patch.object(module,'DROP_SETTLE_SECONDS',.01)
        timing.start();self.addCleanup(timing.stop)
        self.pose_frame='world';self.pose_component='arm'
        snapshot=self.adapter.snapshot
        async def referenced_snapshot():
            result=await snapshot()
            result['arms']['arm'].update(pose_frame=self.pose_frame,pose_component=self.pose_component)
            return result
        self.adapter.snapshot=referenced_snapshot
        self.linear_gate=None
        self.linear_started=asyncio.Event()
        self.linear_error=None
        async def linear(target,expected_start,line_tolerance_mm=2,orientation_tolerance_deg=1):
            self.adapter.calls.append(('move_pose_linear',copy.deepcopy(target),copy.deepcopy(expected_start),
                                       line_tolerance_mm,orientation_tolerance_deg))
            if not module.pose_matches(self.adapter.arm_pose,expected_start):
                raise ValueError('wrong mock linear start')
            self.assertEqual(target['x'],expected_start['x'])
            self.assertEqual(target['y'],expected_start['y'])
            self.assertEqual(module.orientation_distance(target,expected_start),0)
            self.assertNotEqual(target['z'],expected_start['z'])
            self.linear_started.set()
            if self.linear_gate is not None:await self.linear_gate.wait()
            if self.linear_error:raise RuntimeError(self.linear_error)
            self.adapter.arm_pose=copy.deepcopy(target)
            return {'linear_constraint_requested':True,'physical_path_verified':False}
        self.adapter.move_pose_linear=linear

    asyncTearDown=fixtures.ControllerSafetyTests.asyncTearDown
    drain=fixtures.ControllerSafetyTests.drain
    command=fixtures.ControllerSafetyTests.command

    async def teach(self):
        await self.command('teach_loading')
        self.adapter.arm_pose.update(x=600.,y=-150.,z=600.)
        await self.command('teach_drop')
        await self.command('set_ball',ball_id='ball-pink')
        self.adapter.arm_pose.update(x=150.,y=-600.,z=200.)
        await self.command('teach_pickup',ball_id='ball-pink')
        # Most tests begin from the previous high drop pose. Separate entry tests
        # exercise the low grasp and partially descended states explicitly.
        self.adapter.arm_pose=copy.deepcopy(self.controller.calibration['drop_pose'])

    async def launch(self,**extra):
        return await self.controller.action({'action':'pickup_drop_cycle','ball_id':'ball-pink',
            'pickup_setup_confirmed':True,'fixed_setup_confirmed':True,**extra})

    def commands(self):
        return [c[0] for c in self.adapter.calls if c[0]!='connect']

    async def test_teaching_is_read_only_and_preserves_handload_drop_and_board(self):
        await self.command('teach_loading')
        self.adapter.arm_pose.update(x=600.,z=600.)
        await self.command('teach_drop')
        baseline=copy.deepcopy(self.controller.calibration)
        await self.command('set_ball',ball_id='ball-pink')
        self.adapter.arm_pose.update(x=150.,y=-600.,z=200.)
        await self.command('teach_pickup',ball_id='ball-pink')
        self.assertEqual(self.commands(),[])
        for key,value in baseline.items():self.assertEqual(self.controller.calibration[key],value)
        pink=copy.deepcopy(self.controller.calibration['pickup_poses']['ball-pink'])
        await self.command('set_ball',ball_id='ball-purple')
        self.adapter.arm_pose['x']+=90
        await self.command('teach_pickup',ball_id='ball-purple')
        self.assertEqual(self.controller.calibration['pickup_poses']['ball-pink'],pink)
        self.assertEqual(self.controller.calibration['pickup_poses']['ball-purple']['x'],240)
        saved=json.loads(self.controller.config_path.read_text())
        self.assertEqual(saved['pickup_pose_provenance']['ball-purple']['frame'],'world')

    async def test_teaching_rejects_unknown_or_wrong_readback_reference(self):
        await self.teach()
        baseline=copy.deepcopy(self.controller.calibration)
        for frame,component in ((None,'arm'),('arm','arm'),('world',None),('world','arm-board')):
            with self.subTest(frame=frame,component=component):
                self.pose_frame=frame;self.pose_component=component
                self.adapter.arm_pose['x']+=10
                await self.command('teach_pickup',ball_id='ball-pink')
                self.assertEqual(self.controller.calibration,baseline)
                self.assertIn('world-frame arm readback',self.controller.error)
        self.assertEqual(self.commands(),[])

    async def test_full_sequence_uses_vertical_pickup_and_unchanged_saved_drop(self):
        await self.teach()
        original=copy.deepcopy(self.controller.calibration)
        await self.launch();await self.drain()
        self.assertEqual(self.commands(),['move_pose','start_recording','open_gripper',
            'move_pose_linear','close_gripper','move_pose_linear','move_pose','open_gripper','finish_recording'])
        linear=[c for c in self.adapter.calls if c[0]=='move_pose_linear']
        grasp=original['pickup_poses']['ball-pink'];approach={**grasp,'z':grasp['z']+100}
        self.assertEqual(linear[0][1:3],(grasp,approach))
        self.assertEqual(linear[1][1:3],(approach,grasp))
        self.assertEqual(linear[0][3:],(2.,1.))
        self.assertEqual(self.controller.calibration,original)
        trial=self.controller.trial
        self.assertEqual(trial['status'],'completed');self.assertEqual(trial['behavior'],'pickup_station_cycle')
        self.assertEqual(trial['target_pose'],original['drop_pose'])
        self.assertEqual(trial['measured_release_pose'],original['drop_pose'])
        self.assertEqual(trial['ball_id'],'ball-pink');self.assertTrue(trial['pickup_setup_confirmed'])
        self.assertIsNone(trial['height_mm']);self.assertNotIn('loading_countdown_started',[e['event'] for e in trial['events']])

    async def test_selected_pose_and_both_explicit_confirmations_required(self):
        with self.assertRaises(TrialError):await self.launch()
        await self.teach()
        for values in ({'pickup_setup_confirmed':False},{'fixed_setup_confirmed':False},{'ball_id':'ball-gray'}):
            with self.subTest(values=values),self.assertRaises(TrialError):await self.launch(**values)
        await self.command('set_ball',ball_id='ball-gray')
        status=await self.controller.status()
        self.assertNotIn('pickup_drop_cycle',status['allowed_actions'])
        self.assertIn('Teach the selected ball pickup pose',status['pickup_station_missing'])
        self.assertEqual(self.commands(),[])

    async def test_setup_approaches_never_open_descend_or_grip(self):
        await self.teach()
        with self.assertRaises(TrialError):await self.command('go_loading_approach')
        await self.command('go_loading_approach',pickup_setup_confirmed=True)
        expected={**self.controller.calibration['loading_pose'],'z':self.controller.calibration['loading_pose']['z']+100}
        self.assertEqual(self.adapter.arm_pose,expected)
        await self.command('go_pickup_approach',ball_id='ball-pink',pickup_setup_confirmed=True)
        self.assertEqual(self.commands(),['move_pose','move_pose'])
        self.assertEqual(self.adapter.arm_pose['z'],300)
        self.assertIsNone(self.controller.trial)

    async def test_initial_low_grasp_retreats_straight_up_before_opening(self):
        await self.teach()
        self.adapter.arm_pose=copy.deepcopy(self.controller.calibration['pickup_poses']['ball-pink'])
        await self.launch();await self.drain()
        calls=self.commands()
        self.assertEqual(calls,['move_pose_linear','start_recording','open_gripper',
            'move_pose_linear','close_gripper','move_pose_linear','move_pose','open_gripper','finish_recording'])
        first=next(c for c in self.adapter.calls if c[0]=='move_pose_linear')
        self.assertEqual(first[1]['z'],300);self.assertEqual(first[2]['z'],200)

    async def test_other_station_and_partial_descent_block_generic_low_transfer(self):
        await self.teach()
        await self.command('set_ball',ball_id='ball-purple')
        self.adapter.arm_pose.update(x=240.,y=-600.,z=200.)
        await self.command('teach_pickup',ball_id='ball-purple')
        await self.command('set_ball',ball_id='ball-pink')
        for x,z in ((240.,200.),(240.,250.),(150.,250.),(500.,250.)):
            with self.subTest(x=x,z=z):
                self.adapter.arm_pose.update(x=x,z=z)
                for action in ('go_pickup_approach','go_loading_approach'):
                    await self.command(action,ball_id='ball-pink',pickup_setup_confirmed=True)
                    self.assertIn('below pickup transfer clearance',self.controller.error)
        self.assertNotIn('move_pose',self.commands())
        self.assertNotIn('move_pose_linear',self.commands())
        self.assertNotIn('open_gripper',self.commands())

    async def test_retry_after_failed_grip_uses_vertical_retreat(self):
        await self.teach();self.adapter.close_result=False
        await self.launch();await self.drain()
        await self.command('keep_unlabeled',trial_id=self.controller.trial['id'])
        self.adapter.close_result=True;self.adapter.calls=[]
        await self.launch();await self.drain()
        self.assertEqual(self.commands()[0],'move_pose_linear')
        self.assertEqual(self.controller.trial['status'],'completed')

    async def teach_higher_neighbor(self):
        await self.teach()
        await self.command('set_ball',ball_id='ball-purple')
        self.adapter.arm_pose.update(x=240.,y=-600.,z=260.)
        await self.command('teach_pickup',ball_id='ball-purple')
        await self.command('set_ball',ball_id='ball-pink')
        self.adapter.arm_pose=copy.deepcopy(self.controller.calibration['drop_pose'])

    async def test_lower_station_entry_and_exit_use_high_transit_and_vertical_stages(self):
        await self.teach_higher_neighbor()
        await self.launch();await self.drain()
        self.assertEqual(self.controller.trial['status'],'completed')
        moves=[c for c in self.adapter.calls if c[0] in ('move_pose','move_pose_linear')]
        self.assertEqual([c[0] for c in moves],['move_pose','move_pose_linear','move_pose_linear',
            'move_pose_linear','move_pose_linear','move_pose'])
        self.assertEqual([c[1]['z'] for c in moves],[360.,300.,200.,300.,360.,600.])
        for c in moves[1:-1]:
            self.assertEqual(c[1]['x'],150.);self.assertEqual(c[1]['y'],-600.)
            self.assertEqual(c[1]['o_y'],c[2]['o_y'])
        events=[e['event'] for e in self.controller.trial['events']]
        self.assertIn('approach_vertical_entry_sent',events);self.assertIn('clearance_lift_sent',events)

    async def test_failed_high_transit_entry_blocks_open_and_grip(self):
        await self.teach_higher_neighbor();self.linear_error='entry plan failed'
        await self.launch();await self.drain()
        self.assertEqual(self.commands().count('move_pose'),1)
        self.assertEqual(self.commands().count('move_pose_linear'),1)
        self.assertNotIn('open_gripper',self.commands());self.assertNotIn('close_gripper',self.commands())
        self.assertEqual(self.controller.trial['status'],'aborted')

    async def test_missing_linear_capability_never_falls_back_to_generic_motion(self):
        await self.teach();self.adapter.move_pose_linear=None
        with self.assertRaises(TrialError):await self.launch()
        self.assertEqual(self.commands(),[])

    async def test_station_does_not_require_or_infer_a_handloading_pose(self):
        await self.teach();self.controller.calibration.pop('loading_pose')
        status=await self.controller.status()
        self.assertIn('pickup_drop_cycle',status['allowed_actions'])
        self.assertNotIn('load_drop_cycle',status['allowed_actions'])
        await self.launch();await self.drain()
        self.assertEqual(self.controller.trial['status'],'completed')

    async def test_unknown_pickup_or_drop_frame_blocks_motion(self):
        await self.teach()
        for section in ('pickup','drop'):
            with self.subTest(section=section):
                target=(self.controller.calibration['pickup_pose_provenance']['ball-pink']
                        if section=='pickup' else self.controller.calibration)
                target['frame']='unknown'
                with self.assertRaises(TrialError):await self.launch()
                target['frame']='world'
        self.assertEqual(self.commands(),[])

    async def test_failed_grip_blocks_lift_transfer_and_release(self):
        await self.teach();self.adapter.close_result={'grabbed':False}
        await self.launch();await self.drain()
        self.assertEqual(self.commands().count('move_pose_linear'),1)
        self.assertEqual(self.commands().count('move_pose'),1)
        self.assertEqual(self.commands().count('open_gripper'),1)
        self.assertIn('stop_dropper',self.commands())
        self.assertEqual(self.controller.trial['status'],'aborted')

    async def test_failed_descent_never_grips_or_lifts(self):
        await self.teach();self.linear_error='planner rejected constrained path'
        await self.launch();await self.drain()
        self.assertNotIn('close_gripper',self.commands())
        self.assertEqual(self.commands().count('move_pose_linear'),1)
        self.assertEqual(self.commands().count('move_pose'),1)
        self.assertEqual(self.controller.trial['status'],'aborted')

    async def test_stop_during_descent_prevents_grip_lift_and_transfer(self):
        await self.teach();self.linear_gate=asyncio.Event()
        await self.launch();await asyncio.wait_for(self.linear_started.wait(),2)
        await self.command('stop')
        self.assertNotIn('close_gripper',self.commands())
        self.assertEqual(self.commands().count('move_pose_linear'),1)
        self.assertEqual(self.commands().count('open_gripper'),1)
        self.assertEqual(self.controller.trial['status'],'aborted')

    async def test_late_cancelled_descent_cannot_resume_cycle(self):
        await self.teach();gate=asyncio.Event();started=asyncio.Event()
        async def late(target,start,**kwargs):
            self.adapter.calls.append(('move_pose_linear',target,start));started.set()
            try:await gate.wait()
            except asyncio.CancelledError:await gate.wait()
            self.adapter.arm_pose=copy.deepcopy(target)
        self.adapter.move_pose_linear=late
        await self.launch();await asyncio.wait_for(started.wait(),2)
        stopping=asyncio.create_task(self.command('stop'))
        await asyncio.sleep(.01);gate.set();await stopping
        self.assertNotIn('close_gripper',self.commands())
        self.assertEqual(self.commands().count('move_pose'),1)
        self.assertEqual(self.controller.trial['status'],'aborted')

    async def test_recording_failure_stops_before_open_or_descent(self):
        await self.teach();self.adapter.recording_error='no saved camera frames'
        await self.launch();await self.drain()
        self.assertNotIn('open_gripper',self.commands());self.assertNotIn('move_pose_linear',self.commands())

    async def test_recording_lost_during_open_blocks_descent(self):
        await self.teach();original=self.adapter.open_gripper
        async def opening():
            result=await original();self.adapter.frame_age=20;return result
        self.adapter.open_gripper=opening
        await self.launch();await self.drain()
        self.assertNotIn('move_pose_linear',self.commands());self.assertNotIn('close_gripper',self.commands())
        self.assertEqual(self.controller.trial['status'],'aborted')

    async def test_slow_readback_expiring_recording_blocks_vertical_motion(self):
        await self.teach();original=self.adapter.snapshot
        async def delayed():
            state=await original()
            if self.controller.phase=='pickup_descending':await asyncio.sleep(.05)
            return state
        self.adapter.snapshot=delayed
        with patch.object(module,'FRESH_SECONDS',.02):
            await self.controller._snapshot()
            await self.launch();await self.drain()
        self.assertNotIn('move_pose_linear',self.commands());self.assertNotIn('close_gripper',self.commands())
        self.assertIn('Recording became stale before pickup vertical motion',self.controller.error)

    async def test_slow_grasp_readback_expiring_recording_blocks_close(self):
        await self.teach();original=self.adapter.snapshot
        grasp=self.controller.calibration['pickup_poses']['ball-pink']
        async def delayed():
            state=await original()
            if self.controller.phase=='pickup_descending' and self.adapter.arm_pose==grasp:
                await asyncio.sleep(.05)
            return state
        self.adapter.snapshot=delayed
        with patch.object(module,'FRESH_SECONDS',.02):
            await self.controller._snapshot()
            await self.launch();await self.drain()
        self.assertEqual(self.commands().count('move_pose_linear'),1)
        self.assertNotIn('close_gripper',self.commands())
        self.assertIn('Recording became stale before station grip',self.controller.error)

    async def test_unknown_gripper_state_after_lift_blocks_transfer(self):
        await self.teach();original=self.adapter.move_pose_linear
        async def linear(target,start,**kwargs):
            result=await original(target,start,**kwargs)
            if target['z']>start['z']:self.controller._invalidate_gripper()
            return result
        self.adapter.move_pose_linear=linear
        await self.launch();await self.drain()
        self.assertEqual(self.commands().count('move_pose_linear'),2)
        self.assertEqual(self.commands().count('move_pose'),1)
        self.assertEqual(self.commands().count('open_gripper'),1)
        self.assertEqual(self.controller.trial['status'],'aborted')

    async def test_recording_loss_after_lift_blocks_drop_transfer(self):
        await self.teach();original=self.adapter.move_pose_linear
        async def linear(target,start,**kwargs):
            result=await original(target,start,**kwargs)
            if target['z']>start['z']:self.adapter.frame_age=20
            return result
        self.adapter.move_pose_linear=linear
        await self.launch();await self.drain()
        self.assertEqual(self.commands().count('move_pose_linear'),2)
        self.assertEqual(self.commands().count('move_pose'),1)
        self.assertEqual(self.commands().count('open_gripper'),1)
        self.assertEqual(self.controller.trial['status'],'aborted')
        self.assertIn('recording required before station transfer',self.controller.error)

    async def test_audio_lifecycle_preserves_trial_ball_and_command_provenance(self):
        class Audio:
            def __init__(self):self.events=[]
            def emit(self,event):self.events.append(event)
            def stop(self,trial):pass
        await self.teach();audio=Audio()
        # Hardware labels exercise lifecycle formatting only. The adapter is
        # still the in-memory fake and no SDK is imported by this test.
        self.controller.mode='hardware';self.controller.audio=audio
        await self.launch();await self.drain()
        self.assertEqual([e['event_type'] for e in audio.events],['pickup','carry_start','carry_stop','release'])
        self.assertTrue(all(e['ball_id']=='ball-pink' and e['trial_id']==self.controller.trial['id'] for e in audio.events))
        self.assertTrue(all(e['evidence']=='operator_action' and e['contact_object']=='unknown' for e in audio.events))
        ids={e['event_id'] for e in self.controller.trial['events']}
        self.assertTrue(all(e['event_id'] in ids for e in audio.events))

    async def test_changed_board_or_grasp_pose_blocks_lift(self):
        for change in ('board','grasp'):
            with self.subTest(change=change):
                await self.teach()
                async def moved_during_close():
                    self.adapter.calls.append(('close_gripper',))
                    if change=='board':self.adapter.board_joints[0]+=4
                    else:self.adapter.arm_pose['x']+=8
                    return {'grabbed':True}
                self.adapter.close_gripper=moved_during_close
                await self.launch();await self.drain()
                self.assertEqual(self.commands().count('move_pose_linear'),1)
                self.assertEqual(self.commands().count('move_pose'),1)
                await self.command('keep_unlabeled',trial_id=self.controller.trial['id'])
                self.adapter.calls=[]

    async def test_practice_retains_unknown_outcome_and_requires_another_press(self):
        await self.teach();await self.launch(practice_mode=True);await self.drain()
        trial=self.controller.trial
        self.assertTrue(trial['saved']);self.assertIsNone(trial['outcome'])
        self.assertEqual(trial['label_status'],'unlabeled')
        self.assertIn('pickup_drop_cycle',(await self.controller.status())['allowed_actions'])
        count=len(self.adapter.calls);await asyncio.sleep(.03)
        self.assertEqual(len(self.adapter.calls),count)

    async def test_pickup_reuses_two_second_drop_hold_and_recording_guards(self):
        import time
        await self.teach();original_move=self.adapter.move_pose;original_open=self.adapter.open_gripper
        times={}
        async def move(pose):
            await original_move(pose)
            if pose==self.controller.calibration['drop_pose']:times['arrived']=time.monotonic()
        async def opened():
            if 'arrived' in times:times['release']=time.monotonic()
            return await original_open()
        self.adapter.move_pose=move;self.adapter.open_gripper=opened
        with patch.object(module,'DROP_SETTLE_SECONDS',2.),patch.object(module,'SETTLE_TIMEOUT_SECONDS',4.):
            await self.launch();await self.drain()
        self.assertGreaterEqual(times['release']-times['arrived'],2)
        self.assertEqual(self.controller.trial['drop_settle_seconds'],2)


if __name__=='__main__':unittest.main()
