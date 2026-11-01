import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from distance_catalogue_contract import (
    DISTANCES,
    choose,
    contract,
    grid,
    jitter,
    refinement,
    scene,
)


class DistanceCatalogueTests(unittest.TestCase):
    def test_geometry_and_grid(self):
        cfg = json.loads(
            (Path(__file__).resolve().parents[1] / "scene.json").read_text()
        )
        spec = contract(cfg)
        self.assertAlmostEqual(spec["table_margin_mm"]["0.65"], 25)
        self.assertEqual(len(grid(0.4)), 1)
        for d in DISTANCES:
            rows = grid(d)
            self.assertEqual(len({r["id"] for r in rows}), len(rows))
            for c in rows:
                s = scene(c, d)
                self.assertAlmostEqual(s["board_center_m"][1] - s["bucket_xy_m"][1], d)
        cfg["table"]["size_m"][1] = 0.5
        with self.assertRaises(AssertionError):
            contract(cfg)

    def test_near_misses_refine_but_invalids_do_not(self):
        c = grid(0.5)[0]
        r = dict(
            c,
            invalid=None,
            quality_score=0,
            success=False,
            reward=2,
            minimum_mouth_clearance_mm=None,
        )
        self.assertEqual(len(refinement([r], 0.5)), 10)
        self.assertEqual(refinement([dict(r, invalid="IK")], 0.5), [])
        self.assertEqual(refinement([r], 0.4), [])

    def test_jitter_is_bounded_reproducible_and_separated(self):
        c = grid(0.4)[0]
        seen = set()
        for di in range(6):
            for final in [False, True]:
                for i in range(50 if final else 20):
                    p = jitter(c, di, 0, i, final)
                    self.assertEqual(p, jitter(c, di, 0, i, final))
                    self.assertLessEqual(abs(p["height_m"] - c["height_m"]), 0.002001)
                    self.assertLessEqual(abs(p["gap_m"] - c["gap_m"]), 0.002001)
                    self.assertLessEqual(abs(p["tilt_deg"] - c["tilt_deg"]), 0.500001)
                    values = (p["height_m"], p["gap_m"], p["tilt_deg"])
                    self.assertNotIn(values, seen)
                    seen.add(values)

    def test_selection_uses_screening_only(self):
        self.assertIsNone(choose([]))
        a = {
            "candidate": {"id": "a"},
            "perturbed_summary": {"captures": 18, "clean": 18},
            "nominal_summary": {"captures": 5},
        }
        b = {
            "candidate": {"id": "b"},
            "perturbed_summary": {"captures": 20, "clean": 10},
            "nominal_summary": {"captures": 5},
        }
        self.assertEqual(choose([a, b]), b["candidate"])
