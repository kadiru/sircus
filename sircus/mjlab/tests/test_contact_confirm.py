import unittest

import numpy as np

from sircus.mjlab.contact_confirm import confirmation_plan,paired_differences
from sircus.mjlab.contact_study import shot_plan


class ConfirmTests(unittest.TestCase):
    def test_new_seed_is_disjoint_from_diagnostic(self):
        source=dict(values=[.47,-.28,.4,36,.3],q10=[0]*6,q14=[1]*6)
        old=shot_plan(source,[],fresh=20,nominal=0)
        new=confirmation_plan(source)
        self.assertEqual(len(new),116)
        self.assertEqual(confirmation_plan(source),new)
        self.assertEqual(new[15]["values"],source["values"])
        self.assertTrue(set(tuple(x["values"]) for x in old).isdisjoint(tuple(x["values"]) for x in new[16:]))

    def test_differences_distinguish_late_and_ballistic(self):
        first=dict(position_m=np.zeros((10,1,3)),forces_N=np.zeros((10,1,6)))
        second={k:v.copy() for k,v in first.items()}
        first["forces_N"][3,0,1]=1;second["forces_N"][3,0,1]=1
        second["position_m"][7:,0]=2
        rows=[dict(physics_steps=5,route_success=True)]
        d=paired_differences(first,second,rows,rows)
        self.assertEqual(d["max_full_horizon_difference_m"],2)
        self.assertEqual(d["max_common_scored_prefix_difference_m"],0)
        self.assertEqual(d["max_before_first_bucket_contact_difference_m"],0)
        self.assertEqual(d["capture_disagreements"],0)

    def test_paired_capture_disagreement_is_counted(self):
        arrays=dict(position_m=np.zeros((10,1,3)),forces_N=np.zeros((10,1,6)))
        d=paired_differences(arrays,arrays,[dict(route_success=True)],[dict(route_success=False)])
        self.assertEqual(d["capture_disagreements"],1)


if __name__=="__main__":unittest.main()
