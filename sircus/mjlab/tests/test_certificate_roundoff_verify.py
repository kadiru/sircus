import copy
import unittest
import numpy as np
from sircus.mjlab import certificate_roundoff_verify as V
from sircus.mjlab.replay650_verify import AuditChecks
from sircus.mjlab.benchmark_verify import EvidenceError


def fixture(a=.004654315848229726,b=.004654315769922895):
    targets={key:np.eye(4) for key in V.TARGET_FIELDS}
    targets['release_center_m']=np.array([.6,0,.4]);targets['board_normal_world']=np.array([0.,0.,1.])
    saved=dict(reason=4,pairs=[['Arm10/link1','table/top']],board_normal_error_deg=b,minimum_clearance_lower_bound_m=.005)
    independent=dict(saved,board_normal_error_deg=a)
    endpoint=dict(targets={k:v.tolist() for k,v in targets.items()},fresh_endpoint=copy.deepcopy(saved))
    class Gate:
        def check(self,*args):return copy.deepcopy(saved)
    return Gate(),dict(q10=[0]*6,q14=[0]*6),dict(id='actual-roundoff',bucket_xy_m=[.86,.025],values=[.6,0,.2,28,.1]),targets,independent,endpoint


class CertificateTests(unittest.TestCase):
    def compare(self,*args):return V.compare_certificates(*args,AuditChecks())
    def test_actual_nonzero_roundoff_keeps_complete_collision_rejection(self):
        values=fixture();before=copy.deepcopy(values[4:]);out=self.compare(*values)
        self.assertLess(out['absolute_difference_deg'],8e-11)
        self.assertEqual(out['reason'],4);self.assertTrue(out['saved_target_certificate_exact'])
        self.assertEqual(values[4:],before)
    def test_original_near_zero_roundoff_still_allowed(self):
        self.assertLess(self.compare(*fixture(0.,1e-6))['cosine_difference'],V.COSINE_TOLERANCE)
    def test_both_angle_and_cosine_limits_are_enforced(self):
        for a,b in ((0.,3e-6),(45.,45.+1e-7),(float('nan'),0.),(-1.,0.)):
            with self.subTest(a=a,b=b),self.assertRaises(EvidenceError):self.compare(*fixture(a,b))
    def test_no_change_to_acceptance_or_clearance_or_contact_pair(self):
        for key,value in (('reason',0),('minimum_clearance_lower_bound_m',.00499),('pairs',[])):
            values=list(fixture());values[4][key]=value
            with self.subTest(key=key),self.assertRaises(EvidenceError):self.compare(*values)
    def test_exact_saved_certificate_must_be_reproduced(self):
        values=list(fixture());values[5]['fresh_endpoint']['reason']=0
        with self.assertRaises(EvidenceError):self.compare(*values)
    def test_full_target_geometry_still_requires_original_tolerance(self):
        for key in V.TARGET_FIELDS:
            values=list(fixture());v=np.asarray(values[5]['targets'][key]);v.flat[0]+=1e-9;values[5]['targets'][key]=v.tolist()
            with self.subTest(key=key),self.assertRaises(EvidenceError):self.compare(*values)
    def test_equal_sentinel_diagnostic_does_not_invent_valid_geometry(self):
        out=self.compare(*fixture(-1.,-1.));self.assertEqual(out['reason'],4)

if __name__=='__main__':unittest.main()
