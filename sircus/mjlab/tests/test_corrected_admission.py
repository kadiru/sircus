"""Common-home identity and local variation composition, never hardware."""
import copy
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np

from sircus.mjlab import admission as a
from sircus.mjlab.learning_contract import digest


class Endpoint:
    def __init__(self,reason=0):
        self.reason=reason;self.calls=0
        self.strict=SimpleNamespace(check=lambda q,**kwargs:dict(passed=True,reason=None,minimum_clearance_m=.01))
    def check(self,*args):self.calls+=1;return dict(reason=self.reason,pairs=[])


class AdmissionTests(unittest.TestCase):
    def scene(self):
        return dict(arms=dict(base_spacing_m=1.33731,board_base_yaw_rad=np.pi),ball=dict(radius_m=.0375),
            board=dict(tool_offset_m=.1,mount_yaw_rad=np.pi/2))
    def home(self,scene,passed=False):
        home=dict(version=a.HOME_VERSION,passed=passed,scene_sha256=digest(scene))
        if passed:home.update(q10=[0.]*6,q14=[0.]*6)
        home["sha256"]=digest(home);return home
    def instance(self,reason=0,passed=False):
        scene=self.scene();endpoint=Endpoint(reason)
        return a.TransitAdmission(scene,"/mock",home=self.home(scene,passed),ik=object(),gate=endpoint),endpoint
    def row(self):
        return dict(bucket_xy_m=[.9,-.4],values=[.67,-.1,.35,25.,.2],q10=[0.]*6,q14=[0.]*6)

    def test_home_record_rejects_tamper_or_other_scene(self):
        scene=self.scene();home=self.home(scene)
        changed=copy.deepcopy(home);changed["passed"]=True
        with self.assertRaises(ValueError):a.validate_home_record(changed,scene)
        changed_scene=copy.deepcopy(scene);changed_scene["ball"]["radius_m"]+=.001
        with self.assertRaises(ValueError):a.validate_home_record(home,changed_scene)

    def test_home_generation_ignores_injected_gates_previous_bucket(self):
        from sircus.mjlab.tests.test_corrected_geometry import Array
        class Arm:
            def solve_candidates(self,targets,**kwargs):
                n=len(targets)
                return dict(q=Array(np.zeros((n,1,6))),valid=Array(np.ones((n,1),bool)),residual=Array(np.zeros((n,1))))
        scene=self.scene();scene["bucket"]=dict(center_xy_m=[.67,-.42])
        def gate(previous):
            return SimpleNamespace(bucket_xy=np.asarray(previous),validation_model_sha256="mock",
                check=lambda q:dict(passed=True,reason=None,minimum_clearance_m=.01))
        first,second=gate([9.,9.]),gate([-9.,-9.])
        with mock.patch.object(a,"native_pose_error",return_value=dict(position_m=0.,rotation_rad=0.)):
            one=a.freeze_common_home(scene,"/mock",ik=SimpleNamespace(dropper=Arm(),board=Arm()),gate=first)
            two=a.freeze_common_home(scene,"/mock",ik=SimpleNamespace(dropper=Arm(),board=Arm()),gate=second)
        self.assertEqual(one["sha256"],two["sha256"])
        self.assertEqual(one["generation_bucket_xy_m"],[.67,-.42])

    def test_home_property_cannot_mutate_internal_frozen_identity(self):
        obj,_=self.instance(passed=True);before=obj.home
        altered=obj.home;altered["q10"][0]=100;altered["sha256"]="bad"
        self.assertEqual(obj.home,before)

    def test_home_unresolved_is_not_own_endpoint_cycle_success(self):
        obj,_=self.instance();result=obj.admit(self.row())
        self.assertFalse(result["passed"]);self.assertEqual(result["reason"],"common_home_not_admitted_in_layout")
        self.assertEqual(result["proof_scope"],a.PROOF_SCOPE);self.assertFalse(result["report"]["attempts"])

    def test_endpoint_rejection_precedes_any_transit(self):
        obj,_=self.instance(reason=7);result=obj.admit(self.row())
        self.assertFalse(result["passed"]);self.assertEqual(result["reason"],"selected_endpoint_not_admitted")

    def test_bounds_and_malformed_q_reject_before_native_gate(self):
        obj,endpoint=self.instance();row=self.row();row["values"][2]=-1
        result=obj.admit(row)
        self.assertEqual(result["reason"],"selected_request_bounds_or_workspace_rejected");self.assertEqual(endpoint.calls,0)
        row=self.row();row["q10"]=[0.]*5
        with self.assertRaises(ValueError):obj.admit(row)

    def test_request_identity_binds_exact_joints(self):
        obj,_=self.instance();row=self.row();first=obj.admit(row)
        row["q14"][0]=.00001;second=obj.admit(row)
        self.assertNotEqual(first["request_sha256"],second["request_sha256"])

    def test_variation_changes_bucket_requires_full_connection(self):
        obj,endpoint=self.instance();obj.admit=lambda row:dict(passed=True,request_sha256="nominal")
        source=self.row();jitter=copy.deepcopy(source);jitter["bucket_xy_m"][0]+=.001
        result=obj.connect_variation(source,jitter)
        self.assertFalse(result["passed"]);self.assertIn("layout_changed",result["reason"]);self.assertEqual(endpoint.calls,0)

    def test_variation_serializes_board_then_dropper_and_checks_reverse(self):
        obj,_=self.instance();obj.admit=lambda row:dict(passed=True,request_sha256="nominal")
        source=self.row();jitter=copy.deepcopy(source);jitter["q14"][1]=.01;jitter["q10"][2]=.02
        with mock.patch.object(a,"audit_segment",return_value=dict(passed=True,total_samples=3,minimum_clearance_m=.01)) as edge:
            result=obj.connect_variation(source,jitter)
        self.assertTrue(result["passed"]);self.assertEqual(edge.call_count,4)
        first,middle=edge.call_args_list[0].args[1:3]
        np.testing.assert_array_equal(first[:6],middle[:6]);self.assertEqual(middle[7],.01)
        start,end=edge.call_args_list[1].args[1:3]
        np.testing.assert_array_equal(start[6:],end[6:]);self.assertEqual(end[2],.02)
        self.assertEqual(result["nominal_request_sha256"],"nominal");self.assertEqual(result["checked_joint_configurations"],12)

    def test_failed_variation_path_stays_failed(self):
        obj,_=self.instance();obj.admit=lambda row:dict(passed=True,request_sha256="nominal")
        with mock.patch.object(a,"audit_segment",return_value=dict(passed=False,reason="clearance")):
            result=obj.connect_variation(self.row(),self.row())
        self.assertFalse(result["passed"]);self.assertEqual(result["reason"],"variation_local_sampled_path_rejected")


if __name__=="__main__":unittest.main()
