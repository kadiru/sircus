import sys
import unittest
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.optimize._numdiff import approx_derivative

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import json

from learning_contract import contract as legacy_contract
from pilot_contract import balanced_indices, contract, layouts
from prepare_assets import fk, matrix
from robust_ik import normal_jacobian, solve


class PilotTests(unittest.TestCase):
    def test_disjoint_seeded_balanced_layouts(self):
        pools = {k: layouts(k, 12) for k in ["train", "validation", "test"]}
        self.assertEqual(pools["train"], layouts("train", 12))
        ids = [r["id"] for p in pools.values() for r in p]
        self.assertEqual(len(ids), len(set(ids)))
        geometry = [
            tuple(r["board_center_m"] + r["bucket_xy_m"])
            for p in pools.values()
            for r in p
        ]
        self.assertEqual(len(geometry), len(set(geometry)))
        for p in pools.values():
            self.assertEqual(
                Counter(s["difficulty"] for s in p),
                {"near": 12, "medium": 12, "far": 12},
            )
        pool = layouts("train", 32)
        indices = balanced_indices(pool, 16, np.random.default_rng(43))
        self.assertEqual(len(indices), len(set(indices)))
        self.assertEqual(
            Counter(pool[i]["difficulty"] for i in indices),
            {"near": 16, "medium": 16, "far": 16},
        )

    def test_new_contract_does_not_relabel_legacy(self):
        cfg = json.loads(
            (Path(__file__).resolve().parents[1] / "scene.json").read_text()
        )
        legacy = legacy_contract(cfg)
        pilot = contract(cfg)
        self.assertNotEqual(pilot["fingerprint"], legacy["fingerprint"])
        self.assertEqual(pilot["legacy_reference_fingerprint"], legacy["fingerprint"])
        self.assertEqual(pilot["ik_acceptance"], 1e-4)
        self.assertTrue(pilot["settling"]["no_release_if_unsettled"])

    def test_analytic_jacobian_matches_finite_difference(self):
        axes = [np.array([1.0, 0, 0]), np.array([0.0, 1, 0]), np.array([0.0, 0, 1])] * 2
        links = [
            (matrix([0.07, 0.02, 0.03], [0.1, 0.2, 0.1]), axis, -3, 3) for axis in axes
        ]
        q = np.array([0.1, -0.5, 0.2, 0.3, -0.2, 0.6])

        def value(q):
            t = fk(q, links)
            return np.r_[t[:3, 3], 0.25 * t[:3, 2]]

        np.testing.assert_allclose(
            normal_jacobian(q, links), approx_derivative(value, q), atol=1e-8
        )
        for normal_only in [False, True]:
            r = solve(fk(q, links), links, q, normal_only)
            self.assertTrue(r["solved"])
            self.assertLess(r["residual"], 1e-4)

    def test_measured_pose_gate_and_feedback_direction(self):
        from scipy.spatial.transform import Rotation
        from settling_contract import accepted, correction, pose_errors

        target = matrix([0.67, -0.1, 0.4], [0.3, 0, 0])
        measured = target[:3, 3] + [0, 0, -0.008]
        normal = Rotation.from_euler("x", 0.32).as_matrix()[:, 2]
        record = pose_errors(target, measured, normal)
        self.assertFalse(
            accepted(
                dict(
                    record, joint_speed_rad_s=0, position_drift_mm=0, normal_drift_deg=0
                )
            )
        )
        command = correction(target, target, measured, normal)
        self.assertAlmostEqual(command[2, 3], 0.408)
        # Opposes the measured excess tilt, rather than reinforcing it.
        self.assertAlmostEqual(np.arccos(command[2, 2]), 0.28)
        self.assertTrue(
            accepted(
                {
                    "position_error_mm": 1,
                    "normal_error_deg": 0.2,
                    "joint_speed_rad_s": 0.001,
                    "position_drift_mm": 0,
                    "normal_drift_deg": 0,
                }
            )
        )
        self.assertFalse(
            accepted(
                {
                    "position_error_mm": 1,
                    "normal_error_deg": 0.2,
                    "joint_speed_rad_s": 0.04,
                    "position_drift_mm": 0,
                    "normal_drift_deg": 0,
                }
            )
        )
        self.assertFalse(
            accepted(
                {
                    "position_error_mm": float("nan"),
                    "normal_error_deg": 0,
                    "joint_speed_rad_s": 0,
                    "position_drift_mm": 0,
                    "normal_drift_deg": 0,
                }
            )
        )

    def test_fingerprint_matches_actual_release_limits(self):
        from settling_contract import (
            MAX_CORRECTIONS,
            NORMAL_TOL_DEG,
            POSITION_TOL_M,
            VELOCITY_TOL_RAD_S,
        )

        cfg = json.loads(
            (Path(__file__).resolve().parents[1] / "scene.json").read_text()
        )
        limits = contract(cfg)["settling"]
        self.assertEqual(limits["position_mm"], POSITION_TOL_M * 1000)
        self.assertEqual(limits["normal_deg"], NORMAL_TOL_DEG)
        self.assertEqual(limits["joint_speed_rad_s"], VELOCITY_TOL_RAD_S)
        self.assertEqual(limits["max_corrections"], MAX_CORRECTIONS)
