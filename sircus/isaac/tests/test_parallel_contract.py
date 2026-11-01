import copy
import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from parallel_contract import (
    BOARD_XY,
    SPACING_M,
    batch_actions,
    candidates,
    cells,
    select,
    specification,
)


class ParallelContractTests(unittest.TestCase):
    def test_distinct_targets_are_table_local_and_fixed_across_actions(self):
        grid = cells(32)
        before = copy.deepcopy(grid)
        self.assertEqual(len({tuple(c["bucket_xy_m"]) for c in grid}), 32)
        for batch in range(len(candidates())):
            actions = batch_actions(batch, grid)
            self.assertEqual(len(actions), 32)
            self.assertGreater(len({(a["height_m"], a["gap_m"]) for a in actions}), 1)
        self.assertEqual(grid, before)
        self.assertTrue(all(c["board_xy_m"] == BOARD_XY for c in grid))
        self.assertGreater(SPACING_M, 4.0)  # Floor footprints cannot overlap.

    def test_every_cell_receives_the_same_search_without_changing_its_bucket(self):
        grid = cells(32)
        planned = {a["id"] for a in candidates()}
        for cell in grid:
            actual = {batch_actions(i, [cell])[0]["id"] for i in range(len(planned))}
            self.assertEqual(planned, actual)

    def test_contract_is_bounded_and_compatible_with_serial_target_subset(self):
        cfg = json.loads(
            Path(__file__).resolve().parents[1].joinpath("scene.json").read_text()
        )
        a = specification(cfg, 32, [0], 0, 0)
        b = specification(cfg, 32, list(range(32)), len(candidates()), 10)
        self.assertEqual(a["cells"][0], b["cells"][0])
        self.assertNotEqual(a["fingerprint"], b["fingerprint"])
        for count, ids, batches, confirmation in [
            (32, [1, 1], 1, 1),
            (32, [32], 1, 1),
            (32, [0], 1000, 1),
            (32, [0], 1, 1000),
        ]:
            with self.assertRaises(ValueError):
                specification(cfg, count, ids, batches, confirmation)

    def test_selected_candidate_must_have_qualified_route(self):
        def row(i, route, single, quality, invalid=None):
            return {
                "id": i,
                "route_success": route,
                "single_bounce_success": single,
                "route_quality": quality,
                "reward": 1,
                "invalid": invalid,
            }

        rows = [
            row("a", False, False, 0),
            row("b", True, True, 2, "endpoint_ik"),
            row("c", True, False, 2),
            row("d", True, True, 1),
        ]
        self.assertEqual(select(rows)["id"], "d")
        self.assertIsNone(select(rows[:2]))
