"""Force-balance regression independent of a GPU/native asset installation."""

from types import SimpleNamespace
import unittest

import numpy as np

from sircus.mjlab.gravity_audit import measure_condition, checks


class GravityAuditTests(unittest.TestCase):
    def setUp(self):
        self.data = SimpleNamespace(
            qfrc_applied=np.zeros(3), qfrc_bias=np.array([8., -3., 9.81]),
            qfrc_passive=np.array([8., -3., 0.]), qfrc_gravcomp=np.array([8., -3., 0.]),
            qacc=np.zeros(3), ncon=0,
        )
        self.jac = np.eye(3)
        self.calls = 0

    def forward(self, _, data):
        self.calls += 1
        data.qacc[:] = (data.qfrc_applied + data.qfrc_passive - data.qfrc_bias) / np.array([2., 3., 1.])

    def measure(self, extra):
        return measure_condition(None, self.data, [0, 1], self.jac, self.jac, self.forward, extra_bias=extra)

    def test_injection_only_robot_and_recomputation(self):
        old = self.measure(True)
        self.assertEqual(self.calls, 2)
        self.assertEqual(old["arm_qacc_rad_s2"], [4., -1.])
        self.assertEqual(self.data.qfrc_applied[2], 0)  # Free ball never compensated.
        corrected = self.measure(False)
        self.assertEqual(corrected["arm_qacc_rad_s2"], [0., 0.])
        self.assertEqual(corrected["arm_qfrc_applied_N_or_Nm"], [0., 0.])
        self.assertAlmostEqual(corrected["board_linear_acceleration_m_s2"][2], -9.81)

    def test_zero_and_double_gravity_conclusion(self):
        case = dict(without_extra_bias=self.measure(False), with_extra_bias=self.measure(True))
        self.assertTrue(all(checks(case).values()))

    def test_contact_contamination_fails(self):
        self.data.ncon = 1
        case = dict(without_extra_bias=self.measure(False), with_extra_bias=self.measure(True))
        self.assertFalse(checks(case)["contact_free"])

    def test_non_gravity_passive_force_not_claimed_compensation(self):
        self.data.qfrc_passive[0] += .01
        case = dict(without_extra_bias=self.measure(False), with_extra_bias=self.measure(True))
        result = checks(case)
        self.assertFalse(result["passive_already_cancels_bias"])
        self.assertFalse(result["passive_is_gravity_compensation"])
        self.assertFalse(result["no_extra_is_stationary"])


if __name__ == "__main__":
    unittest.main()
