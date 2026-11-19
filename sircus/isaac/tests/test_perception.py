"""Sensor-only geometry and strict truth separation regressions."""

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from perception_estimator import cloud, estimate, fit_plane, fit_rim, fit_sphere
from perception_evaluate import compare


class PerceptionTests(unittest.TestCase):
    def test_axial_depth_unprojection_and_camera_transform(self):
        k = np.array([[2, 0, 1], [0, 2, 1], [0, 0, 1]])
        t = np.eye(4)
        t[:3, 3] = [1, 2, 3]
        xyz = cloud(np.full((3, 3), 2.0), k, t)
        np.testing.assert_allclose(xyz[1, 1], [1, 2, 5])
        np.testing.assert_allclose(xyz[0, 0], [0, 1, 5])

    def test_missing_depth_never_fills_from_priors(self):
        result, masks = estimate(
            np.zeros((30, 40, 3)), np.full((30, 40), np.nan), np.eye(3), np.eye(4)
        )
        self.assertTrue(
            all(o["status"] == "missing" for o in result["objects"].values())
        )
        self.assertFalse(result["ground_truth_used"])
        self.assertFalse(masks)
        json.dumps(result, allow_nan=False)

    def test_sphere_center_free_radius_and_translation(self):
        rng = np.random.default_rng(33)
        rays = rng.normal(size=(500, 3))
        rays[:, 2] = abs(rays[:, 2])
        rays /= np.linalg.norm(rays, axis=1)[:, None]
        center = np.array([0.43, -0.27, 0.68])
        radius = 0.032
        points = center + radius * rays + rng.normal(0, 0.0001, rays.shape)
        got, r, rms = fit_sphere(points)
        np.testing.assert_allclose(got, center, atol=0.0003)
        self.assertAlmostEqual(r, radius, delta=0.0003)
        self.assertLess(rms, 0.0003)
        offset = np.array([0.8, -0.6, 0.2])
        np.testing.assert_allclose(
            fit_sphere(points + offset)[0], got + offset, atol=1e-6
        )

    def test_planar_patch_not_accepted_as_ball(self):
        x, y = np.meshgrid(np.linspace(-0.03, 0.03, 15), np.linspace(-0.03, 0.03, 15))
        with self.assertRaises(ValueError):
            fit_sphere(np.c_[x.ravel(), y.ravel(), np.ones(x.size)])

    def test_plane_with_outliers(self):
        rng = np.random.default_rng(2)
        xy = rng.uniform(-0.1, 0.1, (700, 2))
        points = np.c_[xy, 0.4 + 0.3 * xy[:, 0]]
        points += rng.normal(0, 0.0002, points.shape)
        points = np.vstack([points, rng.uniform(-1, 1, (100, 3))])
        center, normal, _, used, rms = fit_plane(points)
        self.assertGreater(used.sum(), 680)
        self.assertLess(rms, 0.001)
        self.assertAlmostEqual(center[2] - 0.3 * center[0], 0.4, delta=0.001)
        self.assertGreater(normal @ np.array([-0.3, 0, 1]) / np.sqrt(1.09), 0.999)

    def test_rim_requires_broad_visible_arc(self):
        angles = np.linspace(0, 2 * np.pi, 200, endpoint=False)
        points = np.c_[
            0.09 * np.cos(angles) + 0.2, 0.09 * np.sin(angles) - 0.3, np.full(200, 0.18)
        ]
        center, _normal, r, _, _, _ = fit_rim(points, np.array([0, 0, 1]))
        np.testing.assert_allclose(center, [0.2, -0.3, 0.18], atol=0.001)
        self.assertAlmostEqual(r, 0.09, delta=0.001)
        with self.assertRaises(ValueError):
            fit_rim(points[:35], np.array([0, 0, 1]))

    def test_evaluator_rejects_mismatched_frames(self):
        self.assertIsNone(compare({"frame_id": "a"}, {"frame_id": "b"}))

    def test_evaluator_preserves_missing_and_uses_table_plane_error(self):
        observed = {
            "frame_id": "a",
            "timestamp_utc": "t",
            "objects": {
                "ball": {"status": "missing"},
                "table": {
                    "status": "estimated",
                    "center_m": [1, 2, 0.002],
                    "normal": [0, 0, 1],
                },
            },
        }
        truth = {
            "frame_id": "a",
            "objects": {
                "ball": {"center_m": [0, 0, 0]},
                "table": {"center_m": [0, 0, 0], "normal": [0, 0, 1]},
            },
        }
        report = compare(observed, truth)
        self.assertEqual(report["errors"]["ball"]["status"], "missing")
        self.assertAlmostEqual(report["errors"]["table"]["plane_distance_error_mm"], 2)

    def test_bundle_rejects_truth_fields(self):
        from perception_worker import KEYS, load_sensor

        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "sensor.npz"
            values = {k: np.array(0) for k in KEYS}
            np.savez(path, **values, ground_truth_ball=np.ones(3))
            with self.assertRaises(ValueError):
                load_sensor(path)
            np.savez(path, **values)
            self.assertEqual(set(load_sensor(path)), KEYS)


if __name__ == "__main__":
    unittest.main()
