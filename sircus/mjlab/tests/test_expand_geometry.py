"""Finite geometry priors never substitute for native contact validation."""

import json
import importlib.util
from pathlib import Path
import unittest
import numpy as np

from sircus.mjlab.expand_geometry import (
    maximum_gap,
    proposals,
    proposal_mask,
    outer_geometry_mask,
    validate_bounds,
    OLD_BOUNDS,
    WIDE_BOUNDS,
)
from sircus.mjlab.learning_contract import LOW, HIGH

SCENE = json.loads(
    (Path(__file__).resolve().parents[2] / "isaac/scene.json").read_text()
)


class ExpandedProposalTests(unittest.TestCase):
    def test_seeded_finite_bounded_and_reach_dense(self):
        xy = np.repeat([[1.2, -0.15], [0.15, -0.63], [1.1, 0.2]], 64, axis=0)
        first = proposals(xy, np.random.default_rng(71), SCENE)
        np.testing.assert_array_equal(
            first, proposals(xy, np.random.default_rng(71), SCENE)
        )
        self.assertTrue(np.isfinite(first).all())
        self.assertTrue(((first >= LOW) & (first <= HIGH)).all())
        self.assertTrue(proposal_mask(xy, first, SCENE).all())
        self.assertLess(np.median(first[:, 2]), 0.30)

    def test_gap_bound_and_refinement(self):
        xy = np.repeat([[0.67, -0.5]], 30, axis=0)
        broad = proposals(xy, np.random.default_rng(5), SCENE)
        refined = proposals(
            xy, np.random.default_rng(6), SCENE, stage="refine", best=broad
        )
        self.assertTrue(proposal_mask(xy, refined, SCENE).all())
        self.assertTrue(
            (refined[:, 4] <= maximum_gap(refined[:, :3], SCENE) + 1e-12).all()
        )
        self.assertFalse(np.array_equal(broad, refined))

    def test_malformed_inputs(self):
        for bucket in ([1, 2], [[float("nan"), 2]]):
            with self.assertRaises(ValueError):
                proposals(bucket, np.random.default_rng(1), SCENE)
        with self.assertRaises(ValueError):
            proposals([[0.6, 0]], np.random.default_rng(1), SCENE, stage="refine")

    def test_explicit_wide_bounds_and_version_metadata(self):
        from sircus.mjlab.expanded_actions import (
            OLD_BOUNDS as ROOT_OLD,
            WIDE_BOUNDS as ROOT_WIDE,
        )

        self.assertEqual(OLD_BOUNDS, ROOT_OLD)
        self.assertEqual(WIDE_BOUNDS, ROOT_WIDE)
        xy = np.repeat([[0.15, -0.63], [1.20, -0.2]], 64, axis=0)
        values = proposals(xy, np.random.default_rng(71), SCENE, bounds=WIDE_BOUNDS)
        low, high = validate_bounds(WIDE_BOUNDS)
        self.assertTrue(((values >= low) & (values <= high)).all())
        self.assertTrue(proposal_mask(xy, values, SCENE, WIDE_BOUNDS).all())
        self.assertTrue((values[:, 2] < LOW[2]).any())
        with self.assertRaises(ValueError):
            validate_bounds(dict(WIDE_BOUNDS, unexpected=1))

    def test_default_outer_mask_matches_original_exactly(self):
        from sircus.mjlab.bank_geometry import outer_geometry_mask as original

        rng = np.random.default_rng(192)
        xy = rng.uniform([0.1, -0.65], [1.24, 0.25], (9, 2))
        values = rng.uniform(LOW, HIGH, (9, 50, 5))
        for a, b in zip(
            original(SCENE, xy, values), outer_geometry_mask(SCENE, xy, values)
        ):
            np.testing.assert_array_equal(a, b)

    @unittest.skipUnless(
        importlib.util.find_spec("torch")
        and importlib.util.find_spec("mujoco")
        and (Path.home() / "sircus-isaac/assets/dropper.urdf").exists(),
        "Native asset/CPU Torch check runs on study host",
    )
    def test_native_audit_preserves_original_accepted_joint_branches(self):
        import torch
        from sircus.mjlab.bank_geometry import audit_candidates as original
        from sircus.mjlab.expand_geometry import audit_values

        torch.set_num_threads(2)
        xy = np.array([[0.67, -0.5]])
        values = np.array(
            [
                [
                    [0.67, -0.1, 0.4, 21, 0.13],
                    [0.62, -0.15, 0.32, 26, 0.24],
                    [0.58, -0.1, 0.38, 26, 0.265],
                ]
            ]
        )
        first, a = original(SCENE, Path.home() / "sircus-isaac/assets", xy, values)
        second, b = audit_values(
            SCENE, Path.home() / "sircus-isaac/assets", xy, values, bounds=OLD_BOUNDS
        )
        self.assertGreater(first.sum(), 0)
        np.testing.assert_array_equal(first, second)
        for key in (
            "q10",
            "q14",
            "position_error_m",
            "board_normal_error_deg",
            "reasons",
        ):
            np.testing.assert_array_equal(a[key], b[key])


if __name__ == "__main__":
    unittest.main()
