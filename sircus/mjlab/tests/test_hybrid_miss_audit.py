"""Reject causal confounds and preserve failed-request denominators."""
import copy
import unittest
import numpy as np
from sircus.mjlab.benchmark_verify import Checks
from sircus.mjlab.hybrid_miss_audit import expected_slots,unchanged_geometry,independent_summary,classify


def command():
    return dict(id='original',values=[.5,-.3,.4,28.,.3],bucket_xy_m=[1.,0.],
        board_heading_offset_deg=0.,board_twist_deg=0.,selected_q10=[0.]*6,selected_q14=[0.]*6)


def proof():
    return dict(targets=dict(board_world=np.eye(4).tolist(),dropper_flange_world=np.eye(4).tolist()))


class AuditTests(unittest.TestCase):
    def test_no_treatment_is_three_requested_declines_not_no_requests(self):
        plan={'cases':[dict(id='case',directory='native-case',original=command(),corrected=None)]}
        slots=expected_slots(plan)
        self.assertEqual(len(slots),6)
        self.assertEqual(sum(s['request'] is None for s in slots),3)
        summary=independent_summary(plan,[])[0]['results']['corrected']
        self.assertEqual((summary['requested'],summary['setup_rejected'],summary['released']),(3,3,0))

    def test_exact_targets_pass_for_materially_different_joints(self):
        old=command();new=copy.deepcopy(old);new['selected_q14'][1]=.3
        unchanged_geometry(old,new,proof(),proof(),{},Checks(),'case')

    def test_bucket_or_release_or_board_rotation_changes_are_rejected(self):
        for field in ('values','bucket_xy_m','selected_q10'):
            old=command();new=copy.deepcopy(old);new['selected_q14'][1]=.3;new[field][0]+=.1
            with self.assertRaises(ValueError):unchanged_geometry(old,new,proof(),proof(),{},Checks(),'case')
        old=command();new=copy.deepcopy(old);new['board_twist_deg']=180.
        with self.assertRaises(ValueError):unchanged_geometry(old,new,proof(),proof(),{},Checks(),'case')

    def test_same_joints_modulo_full_revolution_are_not_a_new_branch(self):
        old=command();new=copy.deepcopy(old);new['selected_q14'][0]=2*np.pi
        with self.assertRaises(ValueError):unchanged_geometry(old,new,proof(),proof(),{},Checks(),'case')

    def test_robot_assistance_remains_separate_and_invalid_physics_never_capture(self):
        row=dict(invalid_physics=False,route_success=False,experimental_route={'robot_assisted_capture':True})
        self.assertEqual(classify(row),'arm_assisted_capture')
        row.update(invalid_physics=True,route_success=True,shot_route='direct')
        self.assertEqual(classify(row),'invalid_physics')


if __name__=='__main__':unittest.main()
