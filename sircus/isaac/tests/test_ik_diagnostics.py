import json
import sys
import unittest
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dropper_diagnostics import solve_dropper
from ik_diagnostics import board_target, solve_board
from prepare_assets import fk, matrix


class IKDiagnosticTests(unittest.TestCase):
    def setUp(self):
        self.links = [
            (matrix([0.1, 0, 0]), np.array([0.0, 0.0, 1.0]), -np.pi, np.pi)
            for _ in range(6)
        ]
        self.initial = np.array([0.1, -0.2, 0.3, -0.4, 0.2, 0.1])

    def test_existing_solution_and_free_spin(self):
        target = fk(self.initial, self.links)
        target[:3, :3] = target[:3, :3] @ Rotation.from_euler("z", 1.5).as_matrix()
        result = solve_board(target, self.links, self.initial, extended=False)
        self.assertTrue(result["solved"])
        self.assertEqual(result["stage"], "original")
        self.assertLess(result["position_error_mm"], 1e-8)
        self.assertLess(result["normal_error_deg"], 1e-5)

    def test_dropper_preserves_full_orientation(self):
        target = fk(self.initial, self.links)
        result = solve_dropper(target, self.links, self.initial)
        self.assertTrue(result["solved"])
        self.assertEqual(result["stage"], "original")
        self.assertLess(result["rotation_error_deg"], 1e-5)

    def test_unresolved_does_not_claim_infeasibility(self):
        target = np.eye(4)
        target[0, 3] = 100
        result = solve_board(target, self.links, self.initial, extended=False)
        self.assertFalse(result["solved"])
        self.assertEqual(result["stage"], "unresolved")
        self.assertEqual(len(result["attempts"]), 2)
        self.assertGreater(result["position_error_mm"], 90000)

    def test_board_target_mount_and_heading(self):
        cfg = json.loads(
            (Path(__file__).resolve().parents[1] / "scene.json").read_text()
        )
        scene = {"board_center_m": [0.67, -0.1, 0.4], "bucket_xy_m": [0.67, -0.4]}
        t = board_target(cfg, scene, [20, 0, 0.15])
        base = matrix([cfg["arms"]["base_spacing_m"], 0, 0], [0, 0, np.pi])
        mount = matrix(
            [0, 0, cfg["board"]["tool_offset_m"]], [0, 0, cfg["board"]["mount_yaw_rad"]]
        )
        world = base @ t @ mount
        np.testing.assert_allclose(world[:3, 3], scene["board_center_m"])
        self.assertLess(world[1, 2], 0)
        self.assertAlmostEqual(world[2, 2], np.cos(np.deg2rad(20)))
        np.testing.assert_array_equal(t, board_target(cfg, scene, [20, 0, 0.25]))
