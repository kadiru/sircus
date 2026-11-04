"""FP32 control bounds do not relax geometry, threshold masks or scoring."""

import copy
import unittest

import numpy as np

from sircus.mjlab.impact_pose_control import compare_observations, ordered_fp32


class ImpactControlTests(unittest.TestCase):
    def fixture(self):
        return dict(position_m=np.zeros((2, 1, 3), np.float32),
            velocity_m_s=np.full((2, 1, 3), .1, np.float32),
            forces_N=np.full((1, 1, 6), 2., np.float32),
            contact_distance_m=np.full((1, 1, 2), np.inf, np.float32),
            contact_geom_ids=np.zeros((1, 1, 2, 2), np.int32),
            time_s=np.array([0., 1/15360], np.float64))

    def test_ordered_float32_handles_negative_numbers_and_signed_zero(self):
        x = np.array([-2., -1., -0., 0., 1., 2.], np.float32)
        encoded = ordered_fp32(x)
        self.assertTrue((np.diff(encoded) >= 0).all())
        self.assertEqual(encoded[2], encoded[3])
        for a in (-2., -.1, .1, 2.):
            first = np.array([a], np.float32)
            next_value = np.nextafter(first, np.array([np.inf], np.float32))
            self.assertEqual(int(ordered_fp32(next_value)[0]-ordered_fp32(first)[0]), 1)

    def test_exact_control_and_additive_pose_arrays_pass(self):
        a = self.fixture(); b = copy.deepcopy(a)
        b["pose_joint_qpos_rad"] = np.zeros((2, 1, 12), np.float32)
        result = compare_observations(a, b, {"success": False}, {"success": False})
        self.assertEqual(result["status"], "passed")
        self.assertEqual(result["additive_pose_fields"], ["pose_joint_qpos_rad"])

    def test_eight_ulp_allowed_only_for_forces_and_velocity(self):
        a = self.fixture(); b = copy.deepcopy(a)
        for key in ("forces_N", "velocity_m_s"):
            for _ in range(8): b[key] = np.nextafter(b[key], np.full_like(b[key], np.inf))
        result = compare_observations(a, b, {"success": False}, {"success": False})
        self.assertEqual(result["status"], "passed")
        self.assertEqual(result["original_fields"]["forces_N"]["maximum_ulp_distance"], 8)
        b["velocity_m_s"] = np.nextafter(b["velocity_m_s"], np.full_like(b["velocity_m_s"], np.inf))
        self.assertEqual(compare_observations(a, b, {}, {})["status"], "failed")

    def test_one_ulp_geometry_or_clock_change_fails(self):
        a = self.fixture()
        for key in ("position_m", "time_s"):
            b = copy.deepcopy(a)
            b[key].flat[-1] = np.nextafter(b[key].flat[-1], np.array(np.inf, dtype=b[key].dtype))
            with self.subTest(key=key):
                self.assertEqual(compare_observations(a, b, {}, {})["status"], "failed")

    def test_threshold_crossing_fails_even_within_one_ulp(self):
        a = self.fixture()
        a["forces_N"][:] = np.float32(.01)
        b = copy.deepcopy(a)
        b["forces_N"] = np.nextafter(b["forces_N"], np.full_like(b["forces_N"], np.inf))
        result = compare_observations(a, b, {}, {})
        self.assertEqual(result["original_fields"]["forces_N"]["maximum_ulp_distance"], 1)
        self.assertFalse(result["force_category_threshold_masks_exact"])
        self.assertEqual(result["status"], "failed")

    def test_nonfinite_pattern_and_any_score_change_fail(self):
        a = self.fixture(); b = copy.deepcopy(a)
        b["contact_distance_m"].flat[0] = -np.inf
        self.assertEqual(compare_observations(a, b, {}, {})["status"], "failed")
        b = copy.deepcopy(a)
        result = compare_observations(a, b, {"reward": .1}, {"reward": .10000000001})
        self.assertFalse(result["complete_scalar_score_output_exact"])
        self.assertEqual(result["status"], "failed")

    def test_shape_dtype_unexpected_original_key_and_signed_geometry_zero_fail(self):
        a = self.fixture()
        for alter in (lambda b: b.update(velocity_m_s=b["velocity_m_s"].astype(np.float64)),
                      lambda b: b.update(unexpected=np.zeros(1)),
                      lambda b: b["position_m"].fill(-0.)):
            b = copy.deepcopy(a); alter(b)
            self.assertEqual(compare_observations(a, b, {}, {})["status"], "failed")


if __name__ == "__main__":
    unittest.main()
