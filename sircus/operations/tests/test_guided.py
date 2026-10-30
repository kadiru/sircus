"""Guided workflow state/provenance/cancellation checks; no hardware or servers."""
import copy
import json
from pathlib import Path
import tempfile
import threading
import time
import unittest

from sircus.operations.service import GuidedOperations

ROOT = Path(__file__).resolve().parents[3]


def wait(service, limit=2):
    end = time.monotonic()+limit
    while service.status()['busy'] and time.monotonic()<end:
        time.sleep(.005)
    if service.status()['busy']:
        raise AssertionError('Numerical operation did not finish')
    return service.status()


def fake(payload, cancel):
    if payload['op'] == 'scan':
        return {'board_pose': {'source':'simulated_joint_kinematics','calibrated':False},
                'bucket_observation': {'source':'synthetic_scene_ground_truth','calibrated':False}}
    scene = copy.deepcopy(payload['scene'])
    action = payload.get('release_action', {'u_mm':0.,'v_mm':0.,'gap_mm':100.})
    return {'selected_scene':scene,'release_action':action,'seed':payload['seed'],
        'board_pose':{'source':'simulated_joint_kinematics'},
        'release_coordinates':{'commanded_release_ball_center_mm':[600,0,500]},
        'result':{'outcome':'captured','success':True,'frames':[{'t':0.,'ball_xyz_mm':[600,0,500]}]},
        'physically_verified':False,'native_paths_checked':False}


class GuidedTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.service = GuidedOperations(root=ROOT, data_root=self.temp.name, runner=fake)
    def tearDown(self):
        self.service.close(); self.temp.cleanup()
    def configure(self):
        scene=self.service.status()['scene']
        self.service.action({'action':'set_board','center_mm':scene['board']['center_mm'],'rpy_deg':scene['board']['rpy_deg']})
        self.service.action({'action':'set_bucket','center_xy_mm':scene['bin']['center_xy_mm']})
    def plan(self):
        self.configure();self.service.action({'action':'scan_bucket'});wait(self.service)
        self.service.action({'action':'plan','mode':'fixed_board','seed':42,'budget':1});return wait(self.service)

    def test_setup_scan_plan_load_drop_order_is_enforced_and_persisted(self):
        self.assertFalse(self.service.status()['hardware_enabled'])
        for action in ['scan_bucket','plan','confirm_loaded','grip_drop']:
            with self.assertRaises(ValueError):self.service.action({'action':action})
        state=self.plan()
        self.assertEqual(state['phase'],'ready_to_load')
        self.assertNotIn('grip_drop',state['available_actions'])
        self.service.action({'action':'confirm_loaded'})
        self.service.action({'action':'grip_drop'})
        final=wait(self.service)
        self.assertEqual(final['phase'],'complete')
        self.assertFalse(final['loaded'])
        self.assertFalse(final['hardware_enabled'])
        saved=json.loads((Path(self.temp.name)/(final['trial_id']+'.json')).read_text())
        self.assertEqual(saved['result']['seed'],42+104729)
        self.assertFalse(saved['busy']);self.assertEqual(saved['available_actions'],[])
        self.assertEqual(saved['result']['planning_mode'],'fixed_board')
        self.service.action({'action':'confirm_loaded'}); self.service.action({'action':'grip_drop'})
        repeated=wait(self.service)
        self.assertEqual(repeated['result']['seed'],42+2*104729)
        self.assertNotEqual(final['trial_id'],repeated['trial_id'])
        self.assertTrue(all(row['data_origin']=='synthetic' for row in saved['events']))
        self.assertTrue(all(row.get('event_id') for row in saved['events']))
        self.assertTrue(any(row['kind']=='ball_loaded_acknowledged' and row['grasp_verified'] is False for row in saved['events']))

    def test_geometry_changes_invalidate_scan_plan_and_load(self):
        self.plan(); self.service.action({'action':'confirm_loaded'})
        old=self.service.status()['scene_fingerprint']
        self.service.action({'action':'set_bucket','center_xy_mm':[700,210]})
        state=self.service.status()
        self.assertNotEqual(old,state['scene_fingerprint'])
        self.assertIsNone(state['bucket_observation']);self.assertIsNone(state['plan']);self.assertFalse(state['loaded'])
        with self.assertRaises(ValueError):self.service.action({'action':'grip_drop'})

    def test_checkpoint_replacement_invalidates_plan_even_same_scene(self):
        self.plan(); old=self.service.status()['revision']
        checkpoint=json.loads((ROOT/'baselines/scene-conditioned-20261030/checkpoint.json').read_text())
        self.service.action({'action':'select_checkpoint','checkpoint':checkpoint})
        self.assertGreater(self.service.status()['revision'],old)
        self.assertIsNone(self.service.status()['plan'])
        self.assertIsNone(self.service.status()['bucket_observation'])

    def test_invalid_inputs_do_not_mutate_scene_or_enable_hardware(self):
        before=self.service.status()['scene_fingerprint']
        bad=[{'action':'set_board','center_mm':[0,0,float('nan')],'rpy_deg':[0,0,0]},
             {'action':'set_bucket','center_xy_mm':[True,10]},
             {'action':'set_bucket','center_xy_mm':[0,10],'hardware':True},
             {'action':'hardware_enable'}, {'action':'select_checkpoint','checkpoint':{}}]
        for payload in bad:
            with self.subTest(payload=payload),self.assertRaises(ValueError):self.service.action(payload)
        self.assertEqual(before,self.service.status()['scene_fingerprint'])

    def test_pending_commands_reject_conflicts_and_stop_discards_late_results(self):
        started=threading.Event();release=threading.Event()
        def slow(payload,cancel):
            started.set();release.wait(2);return fake(payload,cancel)
        self.service.runner=slow;self.configure();self.service.action({'action':'scan_bucket'});self.assertTrue(started.wait(1))
        with self.assertRaisesRegex(ValueError,'pending'):
            self.service.action({'action':'set_bucket','center_xy_mm':[600,200]})
        self.service.action({'action':'stop'});release.set();state=wait(self.service)
        self.assertEqual(state['phase'],'stopped');self.assertIsNone(state['bucket_observation'])
        self.assertIsNone(state['plan']);self.assertFalse(state['loaded'])

    def test_worker_error_is_not_a_fabricated_scan_or_capture(self):
        def failed(payload,cancel):raise ValueError('unreachable endpoint')
        self.service.runner=failed;self.configure();self.service.action({'action':'scan_bucket'})
        result=wait(self.service)
        self.assertEqual(result['phase'],'error');self.assertIsNone(result['bucket_observation']);self.assertIsNone(result['result'])
        self.assertNotIn('grip_drop',result['available_actions'])

    def test_close_kills_only_its_own_unresponsive_numerical_child(self):
        class Child:
            def __init__(self):self.terminated=self.killed=False
            def poll(self):return 0 if self.killed else None
            def terminate(self):self.terminated=True
            def kill(self):self.killed=True
            def wait(self,timeout):return 0
        child=Child(); self.service._process=child
        self.service.close()
        self.assertTrue(child.terminated);self.assertTrue(child.killed)
        self.assertEqual(self.service.status()['available_actions'],[])

    def test_model_and_search_modes_are_explicit_and_seed_bounds_enforced(self):
        self.configure();self.service.action({'action':'scan_bucket'});wait(self.service)
        for kwargs in [{'mode':'automatic'}, {'seed':-1}, {'seed':2**32}, {'budget':0}, {'budget':33}]:
            with self.assertRaises(ValueError):self.service.action({'action':'plan',**kwargs})
        calls=[]
        def record(payload,cancel):calls.append(payload);return fake(payload,cancel)
        self.service.runner=record
        self.service.action({'action':'plan','mode':'learned_board','seed':2**32-1});wait(self.service)
        self.assertEqual(calls[0]['mode'],'learned_board');self.assertIn('checkpoint',calls[0])


class GuidedNumericalTests(unittest.TestCase):
    """Real bounded worker coverage; no server, training job, Viam or robot."""
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.service=GuidedOperations(root=ROOT, data_root=self.temp.name)
        self.scene=self.service.status()['scene']
        self.service.action({'action':'set_board','center_mm':self.scene['board']['center_mm'],
                             'rpy_deg':self.scene['board']['rpy_deg']})
        self.service.action({'action':'set_bucket','center_xy_mm':self.scene['bin']['center_xy_mm']})
    def tearDown(self):
        self.service.close();self.temp.cleanup()

    def test_frozen_prediction_real_fk_fresh_trial_and_fixed_pose(self):
        checkpoint_path=ROOT/'baselines/scene-conditioned-20261030/checkpoint.json'
        frozen=checkpoint_path.read_bytes()
        self.service.action({'action':'scan_bucket'});scan=wait(self.service,30)
        self.assertIsNone(scan['error'])
        self.assertEqual(scan['board_pose']['source'],'simulated_joint_kinematics')
        self.assertLess(scan['board_pose']['reconstruction_position_error_mm'],2)
        self.assertEqual(len(scan['board_pose']['joints_deg']),6)
        self.assertGreater(len(scan['arm10_loading_boxes']),0)
        self.service.action({'action':'plan','mode':'learned_board','seed':42});planned=wait(self.service,45)
        self.assertIsNone(planned['error'])
        reference=json.loads((ROOT/'baselines/scene-conditioned-20261030/preview.json').read_text())
        self.assertEqual(planned['plan']['action'],reference['action'])
        self.assertEqual(checkpoint_path.read_bytes(),frozen)
        self.service.action({'action':'confirm_loaded'});self.service.action({'action':'grip_drop'})
        trial=wait(self.service,30)
        self.assertEqual(trial['phase'],'complete')
        self.assertGreater(len(trial['result']['result']['frames']),0)
        self.assertEqual(trial['result']['planning_mode'],'learned_board')
        self.assertFalse(trial['result']['native_paths_checked'])
        self.assertFalse(trial['hardware_enabled'])
        self.service.action({'action':'plan','mode':'fixed_board','seed':42,'budget':1})
        fixed=wait(self.service,30)
        self.assertIsNone(fixed['error'])
        self.assertEqual(fixed['plan']['selected_scene']['board']['rpy_deg'],self.scene['board']['rpy_deg'])
        self.assertIn('per-scene',fixed['plan']['inference'])
        self.assertEqual(checkpoint_path.read_bytes(),frozen)

    def test_out_of_distribution_layout_cannot_be_silently_promoted_to_learned_plan(self):
        self.service.action({'action':'set_bucket','center_xy_mm':[900,400]})
        self.service.action({'action':'scan_bucket'});wait(self.service,30)
        self.service.action({'action':'plan','mode':'learned_board','seed':42})
        state=wait(self.service,30)
        self.assertEqual(state['phase'],'error')
        self.assertIn('outside trained distribution',state['error'])
        self.assertIsNone(state['plan'])
        self.assertNotIn('grip_drop',state['available_actions'])


if __name__ == '__main__':unittest.main()
