"""Conservative proposal geometry is not a substitute for endpoint acceptance."""

import json
from pathlib import Path
import unittest

import numpy as np

from sircus.mjlab.learning_contract import HIGH, LOW, splits
from sircus.mjlab.learning_reach import feasible_mask, maximum_gap, sample_proposals

ROOT = Path(__file__).resolve().parents[2]


class ReachTests(unittest.TestCase):
    def setUp(self):
        self.scene = json.loads((ROOT / "isaac" / "scene.json").read_text())

    def test_known_endpoint_valid_family_is_admitted(self):
        values = np.array(
            [
                [0.67, -0.1, 0.4, 21, 0.13],
                [0.62, -0.15, 0.32, 26, 0.24],
                [0.58, -0.1, 0.38, 26, 0.265],
            ]
        )
        self.assertTrue(
            feasible_mask(
                values[:, :3],
                values[:, 3],
                [[0.67, -0.5], [0.62, -0.5], [0.58, -0.5]],
                values[:, 4],
                self.scene,
            ).all()
        )

    def test_high_release_is_filtered_and_gap_cap_is_consistent(self):
        board = np.array([[0.67, -0.1, 0.4]])
        cap = maximum_gap(board, self.scene)[0]
        self.assertGreater(cap, 0.13)
        self.assertLess(cap, 0.24)
        self.assertFalse(feasible_mask(board, 25, [[0.67, -0.5]], 0.52, self.scene)[0])
        self.assertTrue(feasible_mask(board, 25, [[0.67, -0.5]], cap, self.scene)[0])

    def test_board_wrist_filter_is_separate_from_dropper(self):
        self.assertFalse(
            feasible_mask([[0.32, 0, 0.4]], 25, [[0.32, -0.5]], 0.1, self.scene)[0]
        )

    def test_invalid_geometry_is_not_admitted(self):
        self.assertFalse(
            feasible_mask([[0.67, -0.1, 0.4]], 25, [[0.67, -0.1]], 0.1, self.scene)[0]
        )
        self.assertFalse(
            feasible_mask([[0.67, -0.1, 0.4]], 90, [[0.67, -0.5]], 0.1, self.scene)[0]
        )
        self.assertFalse(
            feasible_mask([[np.nan, -0.1, 0.4]], 25, [[0.67, -0.5]], 0.1, self.scene)[0]
        )

    def test_proposal_sampling_is_seeded_bounded_and_reports_unresolved(self):
        buckets = np.asarray(splits()["train"])
        first, failed = sample_proposals(
            buckets, np.random.default_rng(20261103), self.scene
        )
        second, again = sample_proposals(
            buckets, np.random.default_rng(20261103), self.scene
        )
        np.testing.assert_array_equal(first, second)
        np.testing.assert_array_equal(failed, again)
        self.assertTrue(np.isfinite(first).all())
        self.assertTrue((first >= LOW).all() and (first <= HIGH).all())
        self.assertTrue(
            feasible_mask(
                first[~failed, :3],
                first[~failed, 3],
                buckets[~failed],
                first[~failed, 4],
                self.scene,
            ).all()
        )
        self.assertGreater((~failed).sum(), 25)

    def test_exhausted_finite_search_does_not_claim_feasibility(self):
        self.scene["arms"]["base_spacing_m"] = 100
        values, failed = sample_proposals(
            [[0.67, -0.5]], np.random.default_rng(3), self.scene, max_tries=3
        )
        self.assertTrue(failed.all())
        self.assertTrue(np.isfinite(values).all())


if __name__ == "__main__":
    unittest.main()
