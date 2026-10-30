"""Bucket-facing geometry and state contracts, isolated from historical imports."""
import copy
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from sircus.operations.bucket_facing import (normal,heading,search_orientation,constrain_prediction,random_center,tilt_bounds)
from sircus.operations.service import GuidedOperations
from sircus.operations.tests.test_guided import fake,wait
from sircus.studies.profiles import front_scene
ROOT=Path(__file__).resolve().parents[3]


def release(scene,action):
    # Independent Rz Ry Rx implementation for transform-contract assertions.
    r,p,y=map(math.radians,scene['board']['rpy_deg']);x=action['u_mm'];v=action['v_mm']
    z=scene['board']['size_mm'][2]/2+scene['ball']['diameter_mm']/2
    rx=[x,math.cos(r)*v-math.sin(r)*z,math.sin(r)*v+math.cos(r)*z]
    ry=[math.cos(p)*rx[0]+math.sin(p)*rx[2],rx[1],-math.sin(p)*rx[0]+math.cos(p)*rx[2]]
    rz=[math.cos(y)*ry[0]-math.sin(y)*ry[1],math.sin(y)*ry[0]+math.cos(y)*ry[1],ry[2]]
    return [a+b for a,b in zip(scene['board']['center_mm'],rz[:2]+[rz[2]+action['gap_mm']])]


class FacingGeometryTests(unittest.TestCase):
    def test_quadrants_wrap_and_nonvertical_normal(self):
        scene=front_scene();center=scene['board']['center_mm']
        for dx,dy in [(0,-250),(200,100),(-200,100),(-200,-100),(200,-100),(-250,.001),(-250,-.001)]:
            scene['bin']['center_xy_mm']=[center[0]+dx,center[1]+dy]
            selected,info=search_orientation(scene,33.)
            self.assertAlmostEqual(info['heading_dot'],1.)
            self.assertAlmostEqual(info['tilt_deg'],33.)
            n=normal(selected['board']['rpy_deg'])
            self.assertAlmostEqual(n[0]*dy-n[1]*dx,0.,places=9)
            self.assertGreater(n[0]*dx+n[1]*dy,0.)
            self.assertGreater(n[2],0.)
        self.assertEqual(scene['board']['rpy_deg'],[0.,-30.,90.])

    def test_nonzero_roll_prediction_rotates_release_without_changing_uv_gap(self):
        scene=front_scene();scene['board']['rpy_deg']=[25.,-30.,90.];scene['bin']['center_xy_mm']=[750.,-400.]
        action={'u_mm':30.,'v_mm':40.,'gap_mm':200.}
        proposal={'selected_scene':scene,'release_action':action,'release_coordinates':{'commanded_release_ball_center_mm':release(scene,action)}}
        frozen=copy.deepcopy(proposal);selected,new_action,info=constrain_prediction(proposal)
        self.assertEqual(proposal,frozen);self.assertEqual(action,new_action)
        self.assertAlmostEqual(info['heading_dot'],1.)
        self.assertAlmostEqual(info['yaw_correction_deg'],-12.039314994843778,places=8)
        computed=release(selected,new_action)
        for a,b in zip(computed,info['expected_rotated_release_mm']):self.assertAlmostEqual(a,b,places=9)
        self.assertEqual(computed[2],proposal['release_coordinates']['commanded_release_ball_center_mm'][2])
        self.assertFalse(info['per_scene_search']);self.assertFalse(info['frozen_model_updated'])

    def test_bounds_degeneracy_and_seeded_box(self):
        scene=front_scene();scene['bin']['center_xy_mm']=scene['board']['center_mm'][:2]
        with self.assertRaises(ValueError):heading(scene)
        for value in ([0,80],[5,90],[80,5],[True,30],[5,float('nan')]):
            with self.assertRaises(ValueError):tilt_bounds(value)
        bounds={'min':[570,-180,330],'max':[630,-120,375]}
        self.assertEqual(random_center(bounds,42),random_center(bounds,42))
        self.assertNotEqual(random_center(bounds,42),random_center(bounds,43))
        for lo,x,hi in zip(bounds['min'],random_center(bounds,42),bounds['max']):self.assertLessEqual(lo,x);self.assertLessEqual(x,hi)
        with self.assertRaises(ValueError):random_center({'min':[0,0,0],'max':[600,20,20]},42)
        with self.assertRaises(ValueError):random_center(bounds,True)


class FacingWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.calls=[]
        def runner(payload,cancel):self.calls.append(copy.deepcopy(payload));return fake(payload,cancel)
        self.service=GuidedOperations(root=ROOT,data_root=self.temp.name,runner=runner,profile='front')
    def tearDown(self):self.service.close();self.temp.cleanup()
    def plan(self,mode='facing_search'):
        s=self.service.status()['scene'];self.service.action({'action':'set_board','center_mm':s['board']['center_mm'],'rpy_deg':s['board']['rpy_deg']})
        self.service.action({'action':'set_bucket','center_xy_mm':s['bin']['center_xy_mm']});self.service.action({'action':'scan_bucket'});wait(self.service)
        self.service.action({'action':'plan','mode':mode,'seed':42,'budget':2});return wait(self.service)
    def test_randomization_invalidates_scan_plan_loading_and_reproduces(self):
        self.plan();self.service.action({'action':'confirm_loaded','expected_mode':'facing_search'})
        bounds=self.service.status()['random_board_bounds_mm'];payload={'action':'randomize_board','bounds_mm':bounds,'seed':13}
        first=self.service.action(payload);second=self.service.action(payload)
        self.assertEqual(first['scene']['board']['center_mm'],second['scene']['board']['center_mm'])
        self.assertIsNone(second['plan']);self.assertIsNone(second['bucket_observation']);self.assertFalse(second['loaded'])
        self.assertNotIn('grip_drop',second['available_actions'])
        before=second['scene_fingerprint']
        with self.assertRaises(ValueError):self.service.action({**payload,'bounds_mm':{'min':[0,0,0],'max':[10,10,20]}})
        self.assertEqual(self.service.status()['scene_fingerprint'],before)
    def test_go_requires_ack_and_matching_mode_then_uses_exact_plan(self):
        plan=self.plan('facing_learned')['plan']
        with self.assertRaises(ValueError):self.service.action({'action':'grip_drop','expected_mode':'facing_learned'})
        with self.assertRaises(ValueError):self.service.action({'action':'confirm_loaded','expected_mode':'facing_search'})
        self.service.action({'action':'confirm_loaded','expected_mode':'facing_learned'})
        with self.assertRaises(ValueError):self.service.action({'action':'grip_drop','expected_mode':'facing_search'})
        self.service.action({'action':'grip_drop','expected_mode':'facing_learned'});state=wait(self.service)
        execute=self.calls[-1];self.assertEqual(execute['scene'],plan['selected_scene']);self.assertEqual(execute['release_action'],plan['release_action'])
        self.assertEqual(execute['seed'],42+104729);self.assertTrue(execute['choreography'])
        self.assertFalse(state['hardware_enabled']);self.assertEqual(state['phase'],'complete')


class FacingRealWorkerTests(unittest.TestCase):
    def worker(self,payload,script=None):
        command=[sys.executable,str(ROOT/'sircus/operations/numerical_worker.py')] if script is None else [sys.executable,'-c',script]
        p=subprocess.run(command,input=json.dumps(payload),text=True,capture_output=True,cwd=ROOT,timeout=60)
        self.assertTrue(p.stdout,p.stderr);return json.loads(p.stdout)
    def test_frozen_heading_has_no_search_and_preserves_checkpoint_mount_guards(self):
        scene=front_scene();scene['bin']['center_xy_mm']=[670,-400]
        path=ROOT/'baselines/front-bucket-20261030/checkpoint.json';raw=path.read_bytes();cp=json.loads(raw)
        script="""import json,sys
from sircus.operations import numerical_worker as w
def forbidden(*a,**k):raise AssertionError('Search invoked from frozen facing mode')
w.front_study.search_episode=w.study.search_episode=w.search_orientation=forbidden
print(json.dumps(w.handle(json.load(sys.stdin))))
"""
        payload={'op':'plan','scene':scene,'checkpoint':cp,'mode':'facing_learned','seed':42,'budget':1}
        result=self.worker(payload,script);self.assertNotIn('error',result)
        self.assertEqual(path.read_bytes(),raw);self.assertEqual(result['checkpoint_id'],cp['checkpoint_id'])
        self.assertEqual(result['search_budget'],0);self.assertAlmostEqual(result['board_facing']['heading_dot'],1.)
        self.assertLess(result['board_facing']['release_transform_error_mm'],1e-6)
        self.assertLess(result['board_pose']['reconstruction_position_error_mm'],2.)
        self.assertEqual(result['raw_prediction']['selected_scene']['board']['rpy_deg'][2],90.)
        self.assertNotEqual(result['selected_scene']['board']['rpy_deg'][2],90.)
        self.assertEqual(result['selected_scene']['robot14']['board_mount_yaw_deg'],90.)
        dot=sum(a*b for a,b in zip(result['board_pose']['normal'],result['board_facing']['normal']))
        self.assertLess(math.degrees(math.acos(max(-1.,min(1.,dot)))),1.2)
        altered=copy.deepcopy(payload);altered['scene']['board']['rpy_deg'][2]=100.
        self.assertIn('error',self.worker(altered))
        altered=copy.deepcopy(payload);altered['scene']['robot14']['board_mount_yaw_deg']=0.
        self.assertIn('error',self.worker(altered))
        altered=copy.deepcopy(payload);altered['checkpoint']['checkpoint_id']='tampered'
        self.assertIn('error',self.worker(altered))
        altered=copy.deepcopy(payload);altered['tilt_bounds_deg']=[1.,2.]
        self.assertIn('no clamp/search',self.worker(altered)['error'])
    def test_search_fresh_trajectory_and_two_second_choreography(self):
        scene=front_scene();plan=self.worker({'op':'plan','scene':scene,'mode':'facing_search','seed':42,'budget':2})
        self.assertNotIn('error',plan);self.assertEqual(len(plan['candidates']),2)
        self.assertAlmostEqual(plan['board_facing']['heading_dot'],1.)
        result=self.worker({'op':'execute','scene':plan['selected_scene'],'release_action':plan['release_action'],'seed':104771,'choreography':True,'initial_scene':scene})
        self.assertNotIn('error',result);timeline=result['choreography']
        self.assertEqual([s['stage'] for s in timeline['segments']],['grip','orient_board','move_dropper','settle','release','ball_flight'])
        settle=timeline['segments'][3];self.assertAlmostEqual(settle['end_s']-settle['start_s'],2.)
        physics=[f for f in timeline['ball_frames'] if f['source']=='versioned_numerical_ball_physics']
        self.assertEqual(len(physics),len(result['result']['frames']))
        for raw,shifted in zip(result['result']['frames'],physics):
            self.assertAlmostEqual(shifted['t'],raw['t']+timeline['release_time_s'])
            self.assertEqual(shifted['ball_xyz_mm'],raw['ball_xyz_mm'])
        self.assertFalse(timeline['native_paths_checked']);self.assertFalse(result['physically_verified'])

if __name__=='__main__':unittest.main()
