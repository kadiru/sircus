"""Derived acos roundoff never weakens strict native admission evidence."""
import copy
import unittest
from unittest.mock import Mock
from sircus.mjlab import fixed35_bouncer_audit_v3 as V
from sircus.mjlab.benchmark_verify import Checks,EvidenceError


def fixture():
    targets=dict(dropper_flange_world=[[1.,0.],[0.,1.]],board_flange_local=[[1.,0.],[0.,1.]],release_center_m=[0.,0.,1.])
    saved=dict(reason=0,minimum_clearance_lower_bound_m=.007472308820424101,board_normal_error_deg=1.2074182697257333e-6,
        pairs=[],strict_geometry=dict(passed=True,minimum_clearance_m=.007472308820424101))
    independent=copy.deepcopy(saved);independent['board_normal_error_deg']=0.
    endpoint=dict(fresh_endpoint=saved,targets=copy.deepcopy(targets));gate=Mock();gate.check.return_value=copy.deepcopy(saved)
    return gate,dict(q10=[0]*6,q14=[1]*6),dict(id='test',values=[1,2,3,4,5],bucket_xy_m=[1,2]),targets,independent,endpoint


class NumericalCertificateTests(unittest.TestCase):
    def test_observed_microdegree_roundoff_accepts_with_entire_saved_certificate_exact(self):
        args=fixture();before=copy.deepcopy(args[4:]);receipt=V.compare_certificates(*args,Checks())
        self.assertEqual(receipt['absolute_difference_deg'],1.2074182697257333e-6)
        self.assertTrue(receipt['saved_target_certificate_exact']);self.assertTrue(receipt['other_independent_fields_exact'])
        self.assertEqual(args[4:],before);args[0].check.assert_called_once()

    def test_no_clearance_reason_contact_or_target_change_permitted(self):
        for mode in ('clearance','reason','contact','saved','target','large_angle','not_near_zero'):
            with self.subTest(mode=mode):
                a=list(fixture())
                if mode=='clearance':a[4]['minimum_clearance_lower_bound_m']-=1e-12
                elif mode=='reason':a[4]['reason']=7
                elif mode=='contact':a[4]['pairs']=['new']
                elif mode=='saved':a[0].check.return_value['reason']=7
                elif mode=='target':a[5]['targets']['release_center_m'][0]=1e-10
                elif mode=='large_angle':a[4]['board_normal_error_deg']=1e-5
                elif mode=='not_near_zero':a[4]['board_normal_error_deg']=1.;a[5]['fresh_endpoint']['board_normal_error_deg']=1.+1e-7;a[0].check.return_value=copy.deepcopy(a[5]['fresh_endpoint'])
                with self.assertRaises(EvidenceError):V.compare_certificates(*a,Checks())

    def test_zero_delta_unchanged(self):
        a=list(fixture());a[4]=copy.deepcopy(a[5]['fresh_endpoint'])
        self.assertEqual(V.compare_certificates(*a,Checks())['absolute_difference_deg'],0.)


if __name__=='__main__':unittest.main()
