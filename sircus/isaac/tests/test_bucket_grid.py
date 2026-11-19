import copy
import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from board_route import BoardRoute
from bucket_grid_contract import (
    candidates,
    cells,
    choose,
    contract,
    perturb,
    scene,
    select,
)


class BucketGridTests(unittest.TestCase):
    def test_route_rejects_table_and_tracks_second_board_bounce_not_rim(self):
        def forces(board=0, bucket=0, table=0):
            return {"board": board, "bucket": bucket, "table": table}

        for external in (False, True):
            d = BoardRoute(1 / 240)
            d.step(forces(board=1))
            for _ in range(15):
                d.step(forces())
            d.step(forces(bucket=2, table=1 if external else 0))
            self.assertEqual(d.result(True, 1)["route_success"], not external)
            d.step(forces(board=1))
            self.assertEqual(d.board_episodes, 2)
            self.assertEqual(d.result(True, 2)["route_success"], not external)
            self.assertFalse(d.result(True, 2)["single_bounce_success"])
        with self.assertRaises(ValueError):
            d.step(forces(table=float("nan")))
        self.assertFalse(BoardRoute(0.01).result(True, 2)["route_success"])

    def test_grid_inside_table_and_bucket_facing(self):
        cfg = json.loads(
            (Path(__file__).resolve().parents[1] / "scene.json").read_text()
        )
        self.assertEqual(len(cells()), 9)
        self.assertEqual(len({tuple(c["bucket_xy_m"]) for c in cells()}), 9)
        self.assertLess(cells()[0]["heading_deg"], 0)
        self.assertGreater(cells()[2]["heading_deg"], 0)
        self.assertEqual(
            scene(candidates()[0], cells()[0])["board_center_m"][:2], [0.67, -0.1]
        )
        self.assertEqual(contract(cfg), contract(copy.deepcopy(cfg)))
        cfg["bucket"]["outer_top_diameter_m"] = 2
        with self.assertRaises(AssertionError):
            contract(cfg)

    def test_seeds_bounds_and_frozen_selection(self):
        c = candidates()[0]
        seeds = set()
        for ci in range(9):
            for final, n in [(False, 10), (True, 30)]:
                for i in range(n):
                    p = perturb(c, ci, 0, i, final)
                    self.assertNotIn(p["seed"], seeds)
                    seeds.add(p["seed"])
                    self.assertLessEqual(abs(p["height_m"] - c["height_m"]), 0.002)
                    self.assertLessEqual(abs(p["gap_m"] - c["gap_m"]), 0.002)
                    self.assertLessEqual(abs(p["tilt_deg"] - c["tilt_deg"]), 0.5)
        rows = [
            dict(c, success=True, route_success=False, route_quality=0, reward=15),
            dict(
                candidates()[1],
                success=True,
                route_success=True,
                route_quality=1,
                reward=10,
            ),
        ]
        self.assertEqual(select(rows), [candidates()[1]])
        self.assertIsNone(choose([]))
        self.assertIsNone(
            choose([{"candidate": c, "summary": {"route_captures": 0, "clean": 0}}])
        )
