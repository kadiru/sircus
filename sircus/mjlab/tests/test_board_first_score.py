"""Synthetic temporal controls, not physical validation or simulated captures."""
import copy
import hashlib
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np

from sircus.mjlab import calibrated_score as v2, table_assisted_score as v3
from sircus.mjlab import board_first_score as primary
from sircus.mjlab.tests.test_calibrated_score import config, trace, truncate


class BoardFirstControls(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = Path(__file__).resolve().parents[3]
        cls.scene = json.loads((cls.root / "sircus/isaac/scene.json").read_text())
        cls.cfg = config(cls.scene)
        cls.xy = cls.scene["bucket"]["center_xy_m"]
        cls.normal = [0., 0., 1.]

    def score(self, arrays):
        from sircus.mjlab import board_first_score_oracle as oracle
        before = copy.deepcopy(arrays)
        result = primary.score_trace(self.scene, self.xy, self.normal, self.cfg, arrays)
        other = oracle.score_trace(self.scene, self.xy, self.normal, self.cfg, arrays)
        self.assertEqual(set(result), set(other))
        for key in result:
            if isinstance(result[key], float):
                self.assertAlmostEqual(result[key], other[key], delta=1e-9, msg=key)
            else:
                self.assertEqual(result[key], other[key], key)
        for key in before:
            np.testing.assert_array_equal(before[key], arrays[key])
        return result

    def test_direct_capture_and_kernel_facts_are_unchanged(self):
        arrays = trace(self.scene)
        old = v2.score_trace(self.scene, self.xy, self.normal, self.cfg, arrays)
        out = self.score(arrays)
        for key in ("success", "whole_ball_entry", "max_retention_s", "physics_steps", "invalid_physics", "capture_time_s", "quality_score", "reward"):
            self.assertEqual(out[key], old[key], key)
        self.assertTrue(out["board_first_confirmed"])
        self.assertTrue(out["route_success"])
        self.assertEqual(out["route_class"], "direct")
        self.assertEqual(out["first_contact_surfaces"], ["board"])
        self.assertAlmostEqual(out["first_contact_start_s"], .05)
        self.assertAlmostEqual(out["first_board_rebound_s"], .052)

    def test_all_later_surface_types_are_permitted(self):
        for channel in range(1, 6):
            with self.subTest(channel=channel):
                arrays = trace(self.scene)
                arrays["forces_N"][500, channel] = 1.
                out = self.score(arrays)
                self.assertTrue(out["route_success"])
                self.assertTrue(out["board_first_confirmed"])
                self.assertEqual(out["forbidden_contact_steps"], 0)
                self.assertAlmostEqual(out["reward"], 4.55)

    def test_robot_then_table_after_board_retains_capture_and_multiple_flags(self):
        arrays = trace(self.scene)
        arrays["forces_N"][100, 4] = 1.
        arrays["forces_N"][500, 2] = 1.
        before = v3.score_trace(self.scene, self.xy, self.normal, self.cfg, arrays)
        out = self.score(arrays)
        self.assertFalse(before["route_success"])
        self.assertTrue(out["route_success"])
        self.assertTrue(out["robot_assisted_capture"])
        self.assertTrue(out["table_assisted_capture"])
        self.assertEqual(out["route_class"], "mixed_assisted")
        self.assertEqual(out["assistance_surfaces"], ["table", "robot"])

    def test_any_surface_before_board_rejects_even_when_old_v3_allowed_table(self):
        for channel in range(1, 6):
            with self.subTest(channel=channel):
                arrays = trace(self.scene)
                arrays["forces_N"][25, channel] = 1.
                out = self.score(arrays)
                self.assertTrue(out["success"])
                self.assertFalse(out["route_success"])
                self.assertEqual(out["board_first_status"], "non_board_first")
                self.assertEqual(out["forbidden_contact_steps"], 1)
                if channel == 2:
                    self.assertTrue(v3.score_trace(self.scene, self.xy, self.normal, self.cfg, arrays)["route_success"])

    def test_first_interval_ties_do_not_establish_contact_order(self):
        for channel in range(1, 6):
            arrays = trace(self.scene)
            arrays["forces_N"][50, channel] = 1.
            out = self.score(arrays)
            self.assertEqual(out["board_first_status"], "ambiguous_first_contact")
            self.assertFalse(out["route_success"])

    def test_board_touch_then_robot_before_rebound_cannot_attribute_robot_velocity_to_board(self):
        arrays = trace(self.scene)
        arrays["forces_N"][51, 4] = 1.
        out = self.score(arrays)
        self.assertTrue(out["board_bounce"])
        self.assertTrue(out["success"])
        self.assertFalse(out["route_success"])
        self.assertEqual(out["board_first_status"], "intervening_contact_before_board_rebound")

    def test_interval_start_at_confirmed_rebound_is_subsequent(self):
        arrays = trace(self.scene)
        arrays["forces_N"][52, 4] = 1.
        out = self.score(arrays)
        self.assertTrue(out["route_success"])
        self.assertTrue(out["robot_assisted_capture"])
        self.assertEqual(out["forbidden_contact_steps"], 0)

    def test_strict_force_threshold_and_union_count(self):
        arrays = trace(self.scene)
        arrays["forces_N"][25, 1:] = .01
        self.assertTrue(self.score(arrays)["route_success"])
        arrays["forces_N"][25, 2:] = np.nextafter(.01, 1.)
        out = self.score(arrays)
        self.assertFalse(out["route_success"])
        self.assertEqual(out["forbidden_contact_steps"], 1)

    def test_partial_exit_still_resets_one_continuous_second(self):
        arrays = trace(self.scene, end=2.2)
        arrays["forces_N"][100, 4] = 1.
        bucket = self.scene["bucket"]
        top = bucket["assumed_bottom_thickness_m"] + bucket["assumed_interior_depth_m"]
        r = self.scene["ball"]["radius_m"]
        arrays["position_m"][(arrays["time_s"] >= .9) & (arrays["time_s"] < .95), 2] = top - r + .001
        out = self.score(arrays)
        self.assertTrue(out["route_success"])
        self.assertAlmostEqual(out["capture_time_s"], 1.95)
        self.assertFalse(self.score(truncate(arrays, 1.949))["route_success"])

    def test_short_retention_does_not_become_capture(self):
        arrays = trace(self.scene)
        arrays["forces_N"][100, 4] = 1.
        self.assertFalse(self.score(truncate(arrays, 1.199))["route_success"])
        self.assertTrue(self.score(truncate(arrays, 1.2))["route_success"])

    def test_contacts_ending_terminal_count_starting_terminal_do_not(self):
        arrays = trace(self.scene)
        arrays["forces_N"][1200:, 2:] = 1.
        out = self.score(arrays)
        self.assertTrue(out["route_success"])
        self.assertEqual(out["assistance_surfaces"], [])
        arrays["forces_N"][1199, 4] = 1.
        out = self.score(arrays)
        self.assertTrue(out["robot_assisted_capture"])
        self.assertEqual(out["robot_contact_steps"], 1)

    def test_rim_contact_is_still_accepted_but_not_clean_entry(self):
        arrays = trace(self.scene)
        t = arrays["time_s"]
        arrays["position_m"][:, 0] += np.where(t <= .2, .05, np.maximum(0, .05 - (t - .2) * .25))
        arrays["forces_N"][199, 1] = 2.
        arrays["forces_N"][100, 4] = 1.
        out = self.score(arrays)
        self.assertTrue(out["route_success"])
        self.assertEqual(out["route_quality"], 1)

    def test_invalid_physics_and_missing_rebound_still_fail(self):
        arrays = trace(self.scene)
        arrays["forces_N"][100, 4] = 1.
        arrays["velocity_m_s"][1200, 0] = 6.3
        out = self.score(arrays)
        self.assertTrue(out["invalid_physics"])
        self.assertFalse(out["route_success"])
        self.assertEqual(out["reward"], -1.)
        arrays = trace(self.scene)
        arrays["velocity_m_s"][:, 2] = -.1
        out = self.score(arrays)
        self.assertEqual(out["board_first_status"], "board_contact_without_rebound")
        self.assertFalse(out["route_success"])
        arrays["forces_N"][:] = 0.
        self.assertEqual(self.score(arrays)["board_first_status"], "no_contact")

    def test_reward_requires_explicit_version_and_no_fabricated_rejection_trace(self):
        from sircus.mjlab import board_first_score_oracle as oracle
        old = v3.score_trace(self.scene, self.xy, self.normal, self.cfg, trace(self.scene))
        for implementation in (primary, oracle):
            with self.assertRaises(ValueError):
                implementation.reward(old)
            self.assertEqual(implementation.reward({"invalid": "setup_rejected"}), -1.)

    def test_oracle_is_independent_of_both_production_kernels(self):
        from sircus.mjlab import board_first_score_oracle as oracle
        with patch.object(primary, "score_trace", side_effect=AssertionError("new production called")), patch.object(v2, "score_trace", side_effect=AssertionError("old production called")):
            out = oracle.score_trace(self.scene, self.xy, self.normal, self.cfg, trace(self.scene))
        self.assertTrue(out["route_success"])


if __name__ == "__main__":
    unittest.main()
