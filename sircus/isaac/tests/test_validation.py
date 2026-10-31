import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from validation_contract import (
    CONDITIONS,
    layouts,
    noisy_depth,
    paired_source,
    summarize,
    verify_pair,
    visible_sphere_pixels,
)


class ValidationTests(unittest.TestCase):
    def test_noise_pair_guard_rejects_changed_color_or_truth(self):
        self.assertEqual(paired_source("noise_2mm"), "clean")
        base = {
            "rgb": np.zeros((3, 3, 3)),
            "intrinsics": np.eye(3),
            "camera_to_world": np.eye(4),
        }
        source = {"frame_id": "a", "objects": {"ball": [1, 2, 3]}}
        truth = {"source_render_frame_id": "a", "objects": {"ball": [1, 2, 3]}}
        verify_pair(base, base, truth, source)
        with self.assertRaises(ValueError):
            verify_pair({**base, "rgb": np.ones((3, 3, 3))}, base, truth, source)
        with self.assertRaises(ValueError):
            verify_pair(base, base, {**truth, "objects": {"ball": [1, 2, 4]}}, source)

    def test_seed_and_balanced_modes(self):
        a = list(layouts(12, 24))
        self.assertEqual(a, list(layouts(12, 24)))
        self.assertNotEqual(a, list(layouts(13, 24)))
        for mode in ["above_board", "between_board_bucket", "table", "inside_bucket"]:
            self.assertEqual(sum(x["ball_mode"] == mode for x in a), 6)
        self.assertEqual(len(CONDITIONS), 6)

    def test_noise_reproducible_and_does_not_fill_invalid(self):
        depth = np.full((500, 500), 2.0)
        depth[:10] = np.nan
        a = noisy_depth(depth, 0.002, 0.1, 10)
        b = noisy_depth(depth, 0.002, 0.1, 10)
        np.testing.assert_equal(a, b)
        self.assertTrue(np.isnan(a[:10]).all())
        self.assertAlmostEqual(np.nanstd(a), 0.002, delta=0.00002)
        self.assertAlmostEqual(np.isnan(a[10:]).mean(), 0.1, delta=0.003)
        np.testing.assert_equal(noisy_depth(depth, 0, 0, 9), depth)
        np.testing.assert_equal(depth[10:], 2.0)

    def test_visibility_requires_depth_agreement(self):
        k = np.array([[100, 0, 50], [0, 100, 50], [0, 0, 1.0]])
        v, u = np.indices((100, 100))
        a = ((u - 50) / 100) ** 2 + ((v - 50) / 100) ** 2 + 1
        disc = 1 - a * (1 - 0.1**2)
        depth = np.where(disc > 0, (1 - np.sqrt(np.maximum(disc, 0))) / a, np.nan)
        self.assertGreater(
            visible_sphere_pixels(depth, k, np.eye(4), [0, 0, 1], 0.1), 200
        )
        self.assertEqual(
            visible_sphere_pixels(
                np.full((100, 100), 0.5), k, np.eye(4), [0, 0, 1], 0.1
            ),
            0,
        )
        self.assertEqual(
            visible_sphere_pixels(
                np.full((100, 100), np.nan), k, np.eye(4), [0, 0, 1], 0.1
            ),
            0,
        )

    def test_missing_is_in_denominator_and_error_conditional(self):
        errors = {
            name: {"status": "missing"} for name in ["table", "board", "bucket", "ball"]
        }
        a = {"condition": "clean", "errors": errors, "ball_visible_pixels": 50}
        b = {
            "condition": "clean",
            "errors": {**errors, "ball": {"status": "estimated", "center_error_mm": 7}},
            "ball_visible_pixels": 50,
        }
        r = summarize([a, b], 144, [])["conditions"]["clean"]
        self.assertEqual(r["objects"]["ball"]["total"], 2)
        self.assertEqual(r["objects"]["ball"]["estimated"], 1)
        self.assertEqual(r["objects"]["ball"]["mean_mm"], 7)
        self.assertEqual(r["ball_visible"], 2)
        self.assertEqual(r["ball_localized_10mm"], 1)


if __name__ == "__main__":
    unittest.main()
