import copy
import json
import math
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from prepare_assets import matrix
from shot_reach_diagnostic import (
    analyze_cell,
    grid,
    model_range,
    reach_sphere,
    release_ceiling,
)
from shot_sweep import ballistic_gap


def fixture_links():
    offsets = (
        [0, 0, 0.364],
        [0, 0, 0],
        [0.39, 0, 0],
        [0.451637, 0, 0],
        [0, 0, 0],
        [0, 0, 0.09],
    )
    return [(matrix(xyz), np.array([0, 0, 1]), -math.pi, math.pi) for xyz in offsets]


class ReachDiagnosticTests(unittest.TestCase):
    def setUp(self):
        self.scene = json.loads(
            (Path(__file__).resolve().parents[1] / "scene.json").read_text()
        )

    def test_terminal_normal_offset_tightens_downward_release_ceiling(self):
        sphere = reach_sphere(fixture_links())
        self.assertAlmostEqual(sphere["outer_radius_m"], 0.931637)
        self.assertAlmostEqual(sphere["normal_wrist_outer_radius_m"], 0.841637)
        self.assertAlmostEqual(sphere["terminal_normal_offset_m"], 0.09)
        normal = release_ceiling([0.6, 0], sphere)
        loose = release_ceiling([0.6, 0], sphere, normal_wrist_bound=False)
        self.assertAlmostEqual(
            normal, 0.364 + math.sqrt(0.841637**2 - 0.6**2) - 0.09 - 0.095
        )
        self.assertGreater(loose - normal, 0.18)
        self.assertIsNone(release_ceiling([0.9, 0], sphere))
        self.assertIsNotNone(
            release_ceiling([0.9, 0], sphere, normal_wrist_bound=False)
        )

    def test_perpendicular_terminal_residual_expands_rather_than_shrinks_bound(self):
        links = fixture_links()
        links[-1] = (matrix([0.001, 0, 0.09]), np.array([0, 0, 1]), -math.pi, math.pi)
        sphere = reach_sphere(links)
        self.assertAlmostEqual(sphere["normal_wrist_outer_radius_m"], 0.842637)
        self.assertAlmostEqual(sphere["terminal_perpendicular_offset_m"], 0.001)

    def test_inverse_model_matches_each_declared_rim_plane(self):
        for plane in (-1.0, 0.0, 1.0):
            gap = ballistic_gap(self.scene, 0.35, 0.4, 28, rim_offset_r=plane)
            self.assertIsNotNone(gap)
            self.assertAlmostEqual(
                float(model_range(self.scene, 0.35, gap, 28, plane)), 0.4, places=10
            )
        self.assertTrue(np.isnan(model_range(self.scene, 0.35, -1.0, 28)))

    def test_wider_model_grid_includes_current_bounds_and_does_not_modify_scene(self):
        original = copy.deepcopy(self.scene)
        self.scene["table"]["size_m"][:2] = [0.2, 0.2]
        sphere = reach_sphere(fixture_links())
        result = analyze_cell(
            self.scene,
            {"id": "fixture", "index": 0, "bucket_xy_m": [1.1, -0.1]},
            {"Arm10": sphere, "Arm14": sphere},
            xy_step=0.2,
            height_step=0.1,
            tilt_step=10,
            top=3,
        )
        if result["current"]["max_model_range_minus_required_distance_m"] is not None:
            self.assertGreaterEqual(
                result["wider_diagnostic"]["max_model_range_minus_required_distance_m"]
                + 1e-12,
                result["current"]["max_model_range_minus_required_distance_m"],
            )
        self.assertEqual(original["ball"], self.scene["ball"])
        self.assertNotIn("captures", result)
        self.assertNotIn("success", result["current"])

    def test_grid_preserves_endpoints_without_float_duplicate_strata(self):
        values = grid(0.14, 0.54, 0.02)
        self.assertEqual(values[0], 0.14)
        self.assertEqual(values[-1], 0.54)
        self.assertEqual(len(values), 21)


if __name__ == "__main__":
    unittest.main()
