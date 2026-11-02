"""CPU-only proposal/acquisition checks; no native or GPU jobs."""

import copy
import unittest

import numpy as np

from sircus.mjlab.expand_far import far_proposals, proposal_rank, select_seeds
from sircus.mjlab.expanded_actions import WIDE_BOUNDS


def seed(**changes):
    row = dict(
        id="old-shot-1",
        phase="search",
        target_id="old-test-08",
        source_phase="targeted_development",
        values=[0.62, -0.1, 0.35, 22, 0.3],
        invalid=None,
        board_bounce=True,
        whole_ball_entry=False,
        max_retention_s=0.0,
        closest_opening_distance_m=0.03,
        contact_peaks_N={"table": 500.0, "robot": 0.0},
    )
    row.update(changes)
    return row


class FarProposalTests(unittest.TestCase):
    scene = dict(
        ball=dict(radius_m=0.0375),
        bucket=dict(assumed_bottom_thickness_m=0.008, assumed_interior_depth_m=0.175),
    )

    def test_late_table_misses_remain_useful_but_robot_assistance_is_penalized(self):
        table = seed()
        robot = seed(
            id="robot",
            closest_opening_distance_m=0.001,
            whole_ball_entry=True,
            contact_peaks_N={"robot": 300.0, "table": 0.0},
        )
        self.assertGreater(proposal_rank(table), 0.0)
        self.assertGreater(proposal_rank(table), proposal_rank(robot))
        self.assertEqual(
            select_seeds([robot, table], "old-test-08", 1)[0]["id"], table["id"]
        )
        self.assertEqual(proposal_rank(seed(invalid="endpoint")), -10.0)

    def test_seed_selection_is_deterministic_diverse_and_does_not_mutate_evidence(self):
        rows = [
            seed(id=f"old-{i}", values=[0.62, -0.1 + i * 0.01, 0.35, 22, 0.3])
            for i in range(20)
        ]
        rows.append(copy.deepcopy(rows[0]))
        rows.append(
            seed(
                id="wrong-target",
                target_id="old-test-09",
                closest_opening_distance_m=0.0,
            )
        )
        before = copy.deepcopy(rows)
        selected = select_seeds(rows, "old-test-08", 8)
        self.assertEqual(len(selected), 8)
        self.assertEqual(selected, select_seeds(rows, "old-test-08", 8))
        self.assertEqual(len({tuple(r["values"]) for r in selected}), 8)
        self.assertEqual(rows, before)
        self.assertTrue(all(r["target_id"] == "old-test-08" for r in selected))

    def test_proposals_cover_full_height_and_tilt_with_bounded_seeded_geometry(self):
        values, meta = far_proposals(
            [1.21, -0.4], 2048, np.random.default_rng(7), self.scene, [seed()]
        )
        repeated, _ = far_proposals(
            [1.21, -0.4], 2048, np.random.default_rng(7), self.scene, [seed()]
        )
        np.testing.assert_array_equal(values, repeated)
        self.assertTrue(np.isfinite(values).all())
        self.assertTrue((values >= WIDE_BOUNDS["low"]).all())
        self.assertTrue((values <= WIDE_BOUNDS["high"]).all())
        self.assertGreater(int((values[:, 2] > 0.40).sum()), 300)
        self.assertGreater(int((values[:, 3] > 48).sum()), 150)
        self.assertEqual(
            set(m["proposal_family"] for m in meta),
            {"full-height-global", "target-relative", "measured-near-miss-local"},
        )
        self.assertTrue(
            all(
                m.get("seed_source_record_sha256")
                for m in meta
                if m["proposal_family"] == "measured-near-miss-local"
            )
        )

    def test_global_only_proposals_do_not_invent_seed_ids_or_labels(self):
        values, meta = far_proposals(
            [1.2, 0.05], 256, np.random.default_rng(8), self.scene
        )
        self.assertEqual(values.shape, (256, 5))
        self.assertTrue(all(m["seed_source_id"] is None for m in meta))
        self.assertTrue(all("route_success" not in m for m in meta))
        for count in (0, 2049):
            with self.assertRaises(ValueError):
                far_proposals([1.2, 0.05], count, np.random.default_rng(8), self.scene)


if __name__ == "__main__":
    unittest.main()
