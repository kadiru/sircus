import math
import unittest

from sircus.mjlab import horizon_window_verify as audit


def requests():
    rows=[]
    for phase in ('controls_pre','diagnostic','controls_post'):
        if phase=='diagnostic':
            for method in audit.METHODS:
                for repeat in range(3):
                    rows.append(dict(id=f'{method}-{repeat}',context_id=audit.CONTEXT,
                        method=method,repeat=repeat,role=phase,phase=phase))
        else:
            for role in ('positive_control','negative_control'):
                for repeat in range(3):
                    rows.append(dict(id=f'{phase}-{role}-{repeat}',phase=phase,role=role,repeat=repeat))
    return rows


class FrozenDenominatorTests(unittest.TestCase):
    def test_exact_case_and_controls(self):
        audit.budget_checks(requests(),audit.replay.AuditChecks())

    def test_refusals_cannot_be_removed(self):
        rows=requests();rows.pop(9)
        with self.assertRaises(Exception):audit.budget_checks(rows,audit.replay.AuditChecks())

    def test_no_case_replacement(self):
        rows=requests();rows[6]['context_id']='arm-clearance-test-07'
        with self.assertRaises(Exception):audit.budget_checks(rows,audit.replay.AuditChecks())

    def test_controls_must_surround_comparison(self):
        rows=requests();rows[0],rows[-1]=rows[-1],rows[0]
        with self.assertRaises(Exception):audit.budget_checks(rows,audit.replay.AuditChecks())


class IndependentBoundaryTests(unittest.TestCase):
    def test_simple_drop_with_known_exact_crossing(self):
        # z=.3375-5t², ball-center tangent at .0375 -> sqrt(.06).
        self.assertAlmostEqual(audit.bisected_horizon(.3375,0.,-10.,.0375,.2),math.sqrt(.06),places=13)

    def test_upward_initial_velocity_uses_descending_root(self):
        # z=.0375+t-5t² reaches tangent again at .2s, not t=0.
        self.assertAlmostEqual(audit.bisected_horizon(.0375,1.,-10.,.0375,.1),.2,places=13)

    def test_cannot_extend_backwards_or_clamp(self):
        for args in ((.1,0.,-10.,.0375,.2),(20.,0.,-10.,.0375,.2)):
            with self.assertRaises(ValueError):audit.bisected_horizon(*args)

    def test_invalid_trajectory_is_not_certified(self):
        for args in ((float('nan'),0.,-10.,.0375,.2),(.1,0.,0.,.0375,.2),
                     (.01,0.,-10.,.0375,.01),(.3,0.,-10.,.0375,0.)):
            with self.assertRaises(ValueError):audit.bisected_horizon(*args)


if __name__=='__main__':unittest.main()
