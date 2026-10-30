import unittest
from unittest.mock import patch
from sircus.studies import contact_interventions as ci
from sircus.studies import front_study as study
from sircus.studies.profiles import front_scene


class ContactInterventionTests(unittest.TestCase):
    def test_declared_override_preserves_predicted_angle_height(self):
        predicted={'roll_deg':3.,'pitch_deg':-31.,'u_mm':4.,'v_mm':7.,'gap_mm':170.}
        with patch.object(study,'predict',return_value={'action':predicted}):
            base,action=ci.intervened_action({},front_scene(),65,95)
        self.assertEqual(base,predicted)
        self.assertEqual(action,dict(predicted,u_mm=65.,v_mm=95.))
        self.assertEqual(predicted['u_mm'],4.)
        self.assertEqual(ci.METHOD,'forced_contact_intervention')

    def test_interventions_remain_within_explicit_bounds(self):
        for u,v in ci.CONTACTS:
            study.action_scene(front_scene(),dict(study.FIXED_ACTION,u_mm=u,v_mm=v))


if __name__=='__main__':unittest.main()
