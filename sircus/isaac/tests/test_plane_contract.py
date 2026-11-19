import copy
import json
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from plane_contract import action, candidates, contract, scene, select


class PlaneContractTests(unittest.TestCase):
    def test_aligned_plane_fixed_distance_and_positive_release_gap(self):
        rows = candidates()
        self.assertEqual(len(rows), 175)
        self.assertEqual(len({r["id"] for r in rows}), 175)
        for r in rows:
            s = scene(r)
            a = action(r)
            self.assertAlmostEqual(
                np.linalg.norm(np.array(s["bucket_xy_m"]) - s["board_center_m"][:2]),
                0.4,
            )
            self.assertEqual(s["bucket_xy_m"][0], s["board_center_m"][0])
            self.assertEqual(s["board_center_m"][1], -0.1)
            self.assertGreater(a[2], 0)
            self.assertEqual(a[1], 0)
            self.assertEqual(a[0], r["tilt_deg"])

    def test_selection_excludes_failed_and_invalid_candidates(self):
        rows = [
            {"id": "a", "success": False, "reward": 100},
            {"id": "b", "success": True, "reward": 15, "invalid": "endpoint_ik"},
            {"id": "c", "success": True, "reward": 14},
            {"id": "d", "success": True, "reward": 15},
        ]
        self.assertEqual([r["id"] for r in select(rows)], ["d", "c"])

    def test_geometry_uncertainty_and_distinct_contract(self):
        cfg = json.loads(
            (Path(__file__).resolve().parents[1] / "scene.json").read_text()
        )
        before = copy.deepcopy(cfg)
        c = contract(cfg)
        self.assertEqual(cfg, before)
        self.assertIn("unresolved", c["scene"]["bucket"]["height_interpretation"])
        self.assertNotEqual(c["fingerprint"], c["controller"])
