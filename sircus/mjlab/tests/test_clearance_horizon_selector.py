import copy
import math
import unittest
from unittest.mock import Mock, patch

import numpy as np

from sircus.mjlab import arm_clearance_distance as precise
from sircus.mjlab import arm_clearance_selector as legacy
from sircus.mjlab import clearance_horizon_selector as extended


class TableHorizonTests(unittest.TestCase):
    def root(self, origin=(0, 0, .4), velocity=(0, 0, 0), **kwargs):
        return extended.descending_table_horizon(origin, velocity, (0, 0, -9.81),
            tabletop_z=kwargs.get("tabletop_z", 0.), ball_radius=.04)

    def test_root_reaches_ball_tangent_on_descending_flight(self):
        for vz in (-1., 0., 2.):
            t = self.root(velocity=(0, 0, vz))
            self.assertAlmostEqual(.4 + vz*t - .5*9.81*t*t, .04)
            self.assertLess(vz-9.81*t, 0.)
        self.assertAlmostEqual(self.root(), math.sqrt(2*.36/9.81))

    def test_table_and_origin_translation_preserves_duration(self):
        self.assertAlmostEqual(self.root(), self.root(origin=(0, 0, .9), tabletop_z=.5))

    def test_invalid_or_missing_crossing_is_not_a_valid_window(self):
        for p, v in (((0, 0, .03), (0, 0, 2)), ((0, 0, .04), (0, 0, -1)),
                     ((0, 0, float("nan")), (0, 0, 0))):
            with self.assertRaises(ValueError):
                self.root(p, v)
        with self.assertRaises(ValueError):
            extended.descending_table_horizon((0, 0, .4), (0, 0, 0), (0, 0, 0),
                tabletop_z=0., ball_radius=.04)

    def test_contact_after_old_window_is_seen_without_changing_forecast(self):
        collider = dict(name="finger", planes=legacy.box_planes([.285, 0., .10], [.01, .03, .03]),
            vertices=precise.box_vertices([.285, 0., .10], [.01, .03, .03]))
        def directions(c, points):
            planes = np.asarray(c["planes"])
            index = np.argmax(points @ planes[:, :3].T + planes[:, 3], axis=1)
            return planes[index, :3]
        origin, velocity, gravity = [0, 0, .5], [1, 0, 0], [0, 0, -9.81]
        before = precise.certify_path(origin, velocity, .2, gravity, [collider], .04, directions)
        horizon = self.root(origin, velocity)
        after = precise.certify_path(origin, velocity, horizon, gravity, [collider], .04, directions)
        self.assertTrue(before["passed"])
        self.assertFalse(after["passed"])
        self.assertGreater(after["closest"]["time_s"], .2)
        self.assertLessEqual(after["actual_spacing_s"], .002)
        self.assertEqual(after["required_margin_m"], .002)


class HorizonGateTests(unittest.TestCase):
    def setUp(self):
        self.gate = object.__new__(extended.HorizonArmClearanceGate)
        self.gate.radius, self.gate.tabletop_z = .04, 0.
        self.gate.colliders = Mock(return_value=["immutable geometry"])
        self.gate._directions = Mock()
        self.old = dict(passed=True, reason=None, rule_sha256=precise.fingerprint(precise.RULE),
            minimum_clearance_lower_bound_m=.01,
            members=[dict(member=i, horizon_s=.1) for i in range(5)])
        self.diagnostic = dict(launch=dict(origin_m=[0, 0, .4]),
            members=[dict(member=i, velocity_world_m_s=[.5, 0, .05*i]) for i in range(5)])

    def check(self, certify=None):
        if certify is None:
            certify = lambda p, v, h, g, c, r, d: dict(passed=True, horizon_s=h,
                minimum_clearance_lower_bound_m=.012, refinements=[])
        with patch.object(precise.PreciseArmClearanceGate, "check", return_value=self.old), \
                patch.object(precise, "certify_path", side_effect=certify) as certificate:
            result = self.gate.check([1]*6, [2]*6, [.8, .1], [0]*5, self.diagnostic)
        return result, certificate

    def test_same_forecasts_and_geometry_are_recertified_and_inputs_unchanged(self):
        old, diagnostic = copy.deepcopy(self.old), copy.deepcopy(self.diagnostic)
        result, certificate = self.check()
        self.assertTrue(result["passed"])
        self.assertEqual(certificate.call_count, 5)
        for i, call in enumerate(certificate.call_args_list):
            p, v, h, g, c, radius, directions = call.args
            self.assertEqual(p, diagnostic["launch"]["origin_m"])
            self.assertEqual(v, diagnostic["members"][i]["velocity_world_m_s"])
            self.assertGreater(h, .1)
            self.assertEqual(c, ["immutable geometry"])
            self.assertEqual(radius, .04)
            self.assertEqual(result["horizon_metadata"][i]["original_horizon_s"], .1)
            self.assertEqual(result["horizon_metadata"][i]["new_horizon_s"], h)
        self.assertEqual(self.old, old)
        self.assertEqual(self.diagnostic, diagnostic)
        self.assertEqual(result["original_rule_sha256"], old["rule_sha256"])
        self.assertNotEqual(result["rule_sha256"], old["rule_sha256"])

    def test_overlong_crossing_rejects_without_clamping_or_certifying(self):
        self.diagnostic["members"][0]["velocity_world_m_s"] = [0, 0, 10.]
        result, certificate = self.check()
        self.assertFalse(result["passed"])
        self.assertEqual(result["reason"], "overlong_or_shortened_table_tangent_horizon")
        self.assertGreater(result["horizon_metadata"][0]["new_horizon_s"], 1.5)
        self.assertIsNone(result["minimum_clearance_lower_bound_m"])
        certificate.assert_not_called()

    def test_incomplete_old_certificate_and_shortened_extension_reject(self):
        self.old["members"] = self.old["members"][:4]
        result, certificate = self.check()
        self.assertFalse(result["passed"])
        certificate.assert_not_called()
        self.old["members"] = [dict(member=i, horizon_s=.8) for i in range(5)]
        result, certificate = self.check()
        self.assertFalse(result["passed"])
        certificate.assert_not_called()

    def test_one_member_refusal_refuses_whole_candidate(self):
        def certificate(p, v, h, g, c, r, d):
            margin = -.001 if v[2] == .1 else .012
            return dict(passed=margin >= .002, horizon_s=h,
                minimum_clearance_lower_bound_m=margin, refinements=[])
        result, _ = self.check(certificate)
        self.assertFalse(result["passed"])
        self.assertEqual(result["minimum_clearance_lower_bound_m"], -.001)

    def test_extension_cannot_rescue_a_short_window_refusal_by_resampling(self):
        self.old.update(passed=False, reason="predicted_arm_clearance_unproven",
            minimum_clearance_lower_bound_m=.001)
        result, _ = self.check()
        self.assertTrue(result["extended_window_passed"])
        self.assertFalse(result["passed"])
        self.assertEqual(result["reason"], "original_window_clearance_unproven")

    def test_safety_and_forecast_rules_remain_fixed(self):
        for key in precise.RULE.keys() - {"version", "end", "missing_or_overlong_horizon"}:
            self.assertEqual(extended.RULE[key], precise.RULE[key], key)
        self.assertEqual(extended.RULE["maximum_horizon_s"], 1.5)
        self.assertEqual(extended.RULE["maximum_sample_spacing_s"], .002)


if __name__ == "__main__":
    unittest.main()
