"""Endpoint and denominator semantics; mock results are not physics evidence."""
import unittest
from types import SimpleNamespace
import numpy as np

from sircus.mjlab.corrected_geometry import CorrectedGeometry, independent_rejections
from sircus.mjlab.learning_ik import shot_targets


class Array:
    def __init__(self,a):self.a=np.asarray(a)
    def __getitem__(self,index):return Array(self.a[index])
    def cpu(self):return self
    def numpy(self):return self.a


class FakeIK:
    def __init__(self,scene,valid=True):self.scene=scene;self.valid=valid;self.calls=[]
    def candidates(self,xy,xyz,tilt,gap,**kwargs):
        self.calls.append(kwargs);n=len(xy)
        q=np.broadcast_to(np.arange(3)[None,:,None],(n,3,6)).astype(float).copy()
        role=dict(q=Array(q),valid=Array(np.full((n,3),self.valid)),residual=Array(np.zeros((n,3))),query_sha256=["mock"]*n)
        return dict(dropper=role,board=role,targets=shot_targets(self.scene,xy,xyz,tilt,gap))


class FakeGate:
    def __init__(self,mode="accept"):self.mode=mode;self.calls=[]
    def check(self,q10,q14,*args):
        self.calls.append((q10.copy(),q14.copy()))
        fail=(self.mode=="independent" and q14[0]==0) or (self.mode=="cross" and q10[0]==0 and q14[0]==0) or self.mode=="reject_all"
        arms=[14] if self.mode in ("independent","reject_all") and fail else []
        return dict(reason=7 if fail else 0,independent_reject_arms=arms,pairs=[["Arm14/link4","Arm14/link6"]] if fail else [],minimum_clearance_lower_bound_m=.01)


class CorrectedGeometryTests(unittest.TestCase):
    def scene(self):
        return dict(arms=dict(base_spacing_m=1.33731,board_base_yaw_rad=np.pi),ball=dict(radius_m=.0375),
            board=dict(tool_offset_m=.1,mount_yaw_rad=np.pi/2))
    def setup(self,mode="accept",valid=True):
        scene=self.scene();ik=FakeIK(scene,valid);gate=FakeGate(mode)
        instance=CorrectedGeometry(scene,"/mock",ik=ik,gate=gate,initials=dict(dropper_joint_rad=[0]*6,board_joint_rad=[0]*6))
        return instance,ik,gate
    def request(self):return np.asarray([[.9,-.4]]),np.asarray([[[.67,-.1,.35,25,.2]]])

    def test_independent_roles_include_carried_ball(self):
        self.assertEqual(independent_rejections([["ball/geom","table"]]),{10})
        self.assertEqual(independent_rejections([["Arm14/link4","bucket/wall"]]),{14})
        self.assertEqual(independent_rejections([["ball/geom","Arm14/board"]]),set())

    def test_rejected_board_branch_prunes_only_that_independent_branch(self):
        instance,ik,gate=self.setup("independent")
        mask,result=instance.audit_values(*self.request())
        self.assertTrue(mask[0,0]);self.assertEqual(result["branch14"][0,0],1)
        self.assertEqual(len(gate.calls),2);self.assertEqual(result["independent_branch_pairs_pruned"][0,0],2)
        self.assertEqual(ik.calls[0]["starts"],16);self.assertEqual(ik.calls[0]["iterations"],180)

    def test_cross_arm_failure_does_not_prune_alternate_arm_pair(self):
        instance,_,gate=self.setup("cross")
        mask,result=instance.audit_values(*self.request())
        self.assertTrue(mask[0,0]);self.assertEqual(len(gate.calls),2)
        self.assertEqual(result["branch10"][0,0],1);self.assertEqual(result["branch14"][0,0],0)
        self.assertEqual(result["independent_branch_pairs_pruned"].sum(),0)

    def test_all_strict_rejects_remain_requests_with_no_flight_admission(self):
        instance,_,_=self.setup("reject_all")
        mask,result=instance.audit_values(*self.request())
        self.assertFalse(mask.any());self.assertEqual(result["reasons"][0,0],7)
        self.assertEqual(result["summary"]["total_requests"],1);self.assertEqual(result["summary"]["endpoint_admitted"],0)

    def test_ik_unresolved_and_bounds_rejected_are_separate(self):
        instance,_,_=self.setup(valid=False)
        _,result=instance.audit_values(*self.request())
        self.assertEqual(result["reasons"][0,0],3)
        instance,ik,_=self.setup();xy,values=self.request();values[0,0,2]=-1
        _,result=instance.audit_values(xy,values)
        self.assertEqual(result["reasons"][0,0],1);self.assertFalse(ik.calls)

    def test_nominal_preservation_needs_original_request_match(self):
        instance,ik,_=self.setup();xy,values=self.request();q=np.zeros((1,1,6))
        mask,result=instance.audit_values(xy,values,known_q10=q,known_q14=q,known_values=values.copy(),known_bucket_xy=xy)
        self.assertTrue(result["preserved_exact_nominal"][0,0]);self.assertTrue(ik.calls[-1]["preserve_known"][0])
        jitter=values.copy();jitter[0,0,0]+=.00005
        _,result=instance.audit_values(xy,jitter,known_q10=q,known_q14=q,known_values=values.copy(),known_bucket_xy=xy)
        self.assertFalse(result["preserved_exact_nominal"][0,0]);self.assertFalse(ik.calls[-1]["preserve_known"][0])

    def test_pinned_solver_budget_is_versioned(self):
        with self.assertRaises(ValueError):CorrectedGeometry(self.scene(),"/mock",starts=8)


if __name__=="__main__":unittest.main()
