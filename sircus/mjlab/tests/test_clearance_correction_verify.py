import copy
from types import SimpleNamespace
import unittest

import numpy as np

from sircus.mjlab import clearance_correction_verify as audit
from sircus.mjlab.arm_clearance_distance import box_vertices,certify_path
from sircus.mjlab.arm_clearance_selector import box_planes


def requests():
    rows=[]
    for phase in ('controls_pre','diagnostic','controls_post'):
        if phase=='diagnostic':
            for context in audit.CONTEXTS:
                for method in audit.METHODS:
                    for i in range(3):rows.append(dict(id=f'{context}-{method}-{i}',context_id=context,method=method,repeat=i,role=phase,phase=phase))
        else:
            for role in ('positive_control','negative_control'):
                for i in range(3):rows.append(dict(id=f'{phase}-{role}-{i}',phase=phase,role=role,repeat=i))
    return rows


class BudgetTests(unittest.TestCase):
    def test_exact_development_comparison(self):
        audit.budget_checks(requests(),audit.replay.AuditChecks())

    def test_cannot_drop_a_refused_attempt(self):
        rows=requests();rows.pop(8)
        with self.assertRaises(Exception):audit.budget_checks(rows,audit.replay.AuditChecks())

    def test_replacing_case_after_outcomes_is_rejected(self):
        rows=requests();rows[6]['context_id']='arm-clearance-test-03'
        with self.assertRaises(Exception):audit.budget_checks(rows,audit.replay.AuditChecks())

    def test_extra_repeat_is_not_an_independent_new_position(self):
        rows=requests();rows[6]['repeat']=4
        with self.assertRaises(Exception):audit.budget_checks(rows,audit.replay.AuditChecks())


class CertificateTests(unittest.TestCase):
    def setUp(self):
        self.collider=dict(name='test-finger',planes=box_planes([0,0,0],[1,1,1]),vertices=box_vertices([0,0,0],[1,1,1]))
        self.gate=SimpleNamespace(radius=.06,colliders=lambda *args:[self.collider])
        self.origin=[1.05,1.05,0.];self.velocity=[0,0,0];self.gravity=[0,0,0];self.horizon=.01
        normal=np.array([1,1,0])/np.sqrt(2)
        provider=lambda c,p:np.tile(normal,(len(p),1))
        proof=certify_path(self.origin,self.velocity,self.horizon,self.gravity,[self.collider],self.gate.radius,provider)
        self.saved=dict(members=[dict(member=i,**copy.deepcopy(proof)) for i in range(5)],passed=proof['passed'],
            minimum_clearance_lower_bound_m=proof['minimum_clearance_lower_bound_m'],native_queries=0,uses_observed_rebound=False)
        self.diag=dict(members=[dict(member=i,rim_band_start_s=self.horizon,full_downward_rim_band_available=True,
            origin_m=self.origin,velocity_world_m_s=self.velocity) for i in range(5)])

    def check(self,saved=None):
        return audit.audit_support_certificate(self.gate,[],[],[],self.diag,self.gravity,
            saved or self.saved,audit.replay.AuditChecks(),'fixture')

    def test_corner_clearance_independently_passes_without_using_native_distance(self):
        self.assertTrue(self.saved['passed']);self.check()
        self.assertLess(self.saved['members'][0]['legacy_minimum_clearance_lower_bound_m'],0)

    def test_invented_positive_bound_is_rejected(self):
        bad=copy.deepcopy(self.saved);bad['members'][0]['minimum_clearance_lower_bound_m']+=.001
        with self.assertRaises(Exception):self.check(bad)

    def test_missing_refinement_cannot_exclude_a_finger(self):
        bad=copy.deepcopy(self.saved);bad['members'][0]['refinements']=[]
        with self.assertRaises(Exception):self.check(bad)

    def test_nonunit_directions_cannot_inflate_a_clearance(self):
        bad=copy.deepcopy(self.saved);bad['members'][0]['refinements'][0]['support_normals'][0]=[20,20,0]
        with self.assertRaises(Exception):self.check(bad)

    def test_horizon_is_not_silently_changed(self):
        bad=copy.deepcopy(self.saved);bad['members'][0]['horizon_s']+=.05
        with self.assertRaises(Exception):self.check(bad)

    def test_all_five_predictions_required(self):
        bad=copy.deepcopy(self.saved);bad['members'].pop()
        with self.assertRaises(Exception):self.check(bad)

    def test_safety_margin_is_not_reduced(self):
        bad=copy.deepcopy(self.saved);bad['members'][0]['required_margin_m']=0
        with self.assertRaises(Exception):self.check(bad)


if __name__=='__main__':unittest.main()
