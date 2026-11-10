"""Exact target yaw and unchanged full-pose geometry guards, without dynamics."""
import copy
import math
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
import numpy as np
from sircus.mjlab import fixed35_imitation_native as N
from sircus.mjlab.learning_ik import shot_targets

SCENE={'board':{'mount_yaw_rad':math.pi/2,'tool_offset_m':.12},'arms':{'base_spacing_m':1.2,'board_base_yaw_rad':math.pi},'ball':{'radius_m':.0375}}


class Array:
    def __init__(self,x): self.x=np.asarray(x)
    def __getitem__(self,x): return Array(self.x[x])
    def cpu(self): return self
    def numpy(self): return self.x


class NativeTests(unittest.TestCase):
    def test_shared_behavior_rotates_only_gripper_and_faces_bucket(self):
        values=np.asarray([.6,-.1,.45,30.,.08])
        for bucket in ([1.1,-.54],[.2,-.7],[.6,.8]):
            bucket=np.asarray(bucket)
            _,target=N.yaw_target(SCENE,bucket,values)
            original={k:v[0] for k,v in shot_targets(SCENE,bucket[None],values[None,:3],[values[3]],[values[4]]).items()}
            direction=np.r_[bucket-values[:2],0.];direction/=np.linalg.norm(direction)
            np.testing.assert_allclose(target['dropper_flange_world'][:3,0],direction,atol=1e-15)
            np.testing.assert_allclose(target['dropper_flange_world'][:3,2],[0,0,-1],atol=1e-15)
            np.testing.assert_array_equal(original['dropper_flange_world'][:3,3],target['dropper_flange_world'][:3,3])
            for key in original:
                if key!='dropper_flange_world':np.testing.assert_array_equal(original[key],target[key])

    def fixture(self,*,valid=True,clear=True):
        pool=lambda: {k:Array(v) for k,v in dict(q=np.zeros((1,1,6)),valid=np.asarray([[valid]]),residual=np.asarray([[0.]])).items()}
        arms=[SimpleNamespace(solve_candidates=Mock(side_effect=lambda *a,**kw:pool())) for _ in range(2)]
        endpoint=SimpleNamespace(check=Mock(return_value=dict(reason=None if clear else 'blocked',independent_reject_arms=[])))
        solver=SimpleNamespace(scene=SCENE,bounds={},low=np.array([0,-1,0,0,0]),high=np.array([2,1,1,90,1]),
             initial=dict(dropper_joint_rad=[0]*6,board_joint_rad=[0]*6),ik=SimpleNamespace(dropper=arms[0],board=arms[1]),endpoint=endpoint,gate=None,max_pairs=1)
        base=SimpleNamespace(scene=SCENE,solver=solver)
        resolver=N.FixedYawResolver(base)
        hint=dict(id='geometry-only',values=[.6,-.1,.45,30.,.08],bucket_xy_m=[1.1,-.54],q10=[0.]*6,q14=[0.]*6,route_success=True,reward=100.)
        return resolver,hint,arms,endpoint

    def test_full_pose_and_geometry_only_hint_no_saved_test_branches(self):
        resolver,hint,arms,endpoint=self.fixture();before=copy.deepcopy(hint)
        with patch.object(N,'outer_geometry_mask',return_value=(np.ones((1,1),bool),np.ones((1,1),bool))),patch.object(N,'native_pose_error',return_value=dict(position_m=0.,rotation_rad=0.)):
            result=resolver.solve(hint['bucket_xy_m'],hint['values'],hint)
        self.assertTrue(result['passed']);self.assertEqual(hint,before)
        self.assertEqual(result['geometry_hint_sha256'],N.digest(N.source_geometry(hint)))
        self.assertFalse(result['selection_uses_flight_labels'])
        for arm in arms:
            kw=arm.solve_candidates.call_args.kwargs
            self.assertFalse(kw['normal_only']);self.assertFalse(kw['preserve_known'][0]);self.assertEqual((kw['starts'],kw['iterations']),(16,180))
        self.assertEqual(endpoint.check.call_count,1)

    def test_finite_ik_and_clearance_failures_remain_declines(self):
        for valid,clear,reason in ((False,True,'finite_ik_unresolved'),(True,False,'strict_endpoint_clearance_or_pose_rejected')):
            resolver,hint,_,_=self.fixture(valid=valid,clear=clear)
            with patch.object(N,'outer_geometry_mask',return_value=(np.ones((1,1),bool),np.ones((1,1),bool))):
                result=resolver.solve(hint['bucket_xy_m'],hint['values'],hint)
            self.assertFalse(result['passed']);self.assertEqual(result['reason'],reason)

    def test_strict_rotation_error_does_not_become_admitted(self):
        resolver,hint,_,_=self.fixture()
        with patch.object(N,'outer_geometry_mask',return_value=(np.ones((1,1),bool),np.ones((1,1),bool))),patch.object(N,'native_pose_error',return_value=dict(position_m=0.,rotation_rad=.01)):
            result=resolver.solve(hint['bucket_xy_m'],hint['values'],hint)
        self.assertFalse(result['passed']);self.assertTrue(result['pair_attempts'][0]['pose_rejected'])

    def test_close_target_is_a_recordable_decline_before_ik(self):
        resolver,hint,arms,_=self.fixture()
        with patch.object(N,'outer_geometry_mask',return_value=(np.ones((1,1),bool),np.ones((1,1),bool))):
            result=resolver.solve(hint['values'][:2],hint['values'],hint)
        self.assertEqual(result['reason'],'board_bucket_distance_constraint_rejected')
        for arm in arms:arm.solve_candidates.assert_not_called()

    def test_negative_gap_is_a_recordable_decline_before_target_building(self):
        resolver,hint,_,_=self.fixture()
        resolver.base.sources=[];resolver.base.home={'sha256':'home'}
        resolver.base._check_resolution_bytes=Mock()
        resolver.base._save=Mock(side_effect=lambda directory,key,record:dict(file=directory+'/'+key+'.json',sha256=N.digest(record)))
        anchor=dict(bucket_xy_m=[1.1,-.54],sha256='anchor',file='anchor.json',home_passed=True,hard_reject=False,anchor_values=hint['values'],anchor_derivation={},effective_source_hint=hint)
        values=list(hint['values']);values[4]=-.02
        with patch.object(N,'verify_resolution'),patch.object(N,'verify_connection'),patch.object(N,'outer_geometry_mask',return_value=(np.zeros((1,1),bool),np.ones((1,1),bool))):
            result=resolver.connect(anchor,values)
        self.assertFalse(result['passed']);self.assertEqual(result['reason'],'action_bounds_rejected')
        self.assertIsNone(result['release_behavior']['target_dropper_flange_world'])
        self.assertEqual(result['values'],values)

    def test_fixed_home_rejection_uses_new_behavior_key_and_no_solver(self):
        resolver,hint,_,_=self.fixture()
        resolver.base.sources=[];resolver.base.home={'sha256':'home'}
        resolver.base._check_resolution_bytes=Mock()
        resolver.base._save=Mock(side_effect=lambda directory,key,record:dict(file=directory+'/'+key+'.json',sha256=N.digest(record)))
        anchor=dict(bucket_xy_m=[1.1,-.54],sha256='anchor',file='anchor.json',home_passed=False,hard_reject=True,anchor_values=hint['values'],anchor_derivation={})
        with patch.object(N,'verify_resolution'),patch.object(N,'verify_connection'),patch.object(resolver,'solve') as solve:
            result=resolver.connect(anchor,hint['values'])
            cached=resolver.connect(anchor,hint['values'])
        solve.assert_not_called();self.assertFalse(result['passed']);self.assertEqual(result['reason'],'fixed_base_clearance_rejected')
        self.assertTrue(cached['cache_hit']);self.assertTrue(result['file'].startswith('fixed-yaw-connections/'))
        self.assertEqual(result['release_behavior']['version'],N.BEHAVIOR_VERSION)
        self.assertEqual(result['values'],hint['values'])

if __name__=='__main__':unittest.main()
