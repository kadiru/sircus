import json
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from learning_contract import contract, observation, outcome_summary, scenes


class LearningContractTests(unittest.TestCase):
    def setUp(self):
        self.cfg = json.loads(
            (Path(__file__).resolve().parents[1] / "scene.json").read_text()
        )

    def test_disjoint_reproducible_splits(self):
        pools = [
            scenes(k, 1000) for k in ["train", "validation", "test", "development"]
        ]
        self.assertEqual(pools[0], scenes("train", 1000))
        keys = [
            {tuple(s["board_center_m"] + s["bucket_xy_m"]) for s in p} for p in pools
        ]
        for i in range(4):
            for j in range(i):
                self.assertFalse(keys[i] & keys[j])

    def test_observation_has_geometry_and_no_outcome(self):
        a = scenes("train", 1)[0]
        x = observation(a, self.cfg)
        self.assertEqual(x.shape, (11,))
        self.assertTrue(np.isfinite(x).all())
        np.testing.assert_array_equal(
            x, observation({**a, "success": True, "reward": 99}, self.cfg)
        )
        b = {**a, "bucket_xy_m": [a["bucket_xy_m"][0] + 0.1, a["bucket_xy_m"][1]]}
        self.assertNotEqual(x[3], observation(b, self.cfg)[3])

    def test_failures_in_denominator(self):
        rows = [
            {"success": True, "reward": 15, "difficulty": "near"},
            {
                "success": False,
                "reward": -2,
                "difficulty": "far",
                "invalid": "endpoint_ik",
            },
        ]
        r = outcome_summary(rows)
        self.assertEqual(r["success_rate"], 0.5)
        self.assertEqual(r["invalid_endpoints"], 1)
        self.assertLess(r["wilson95"][0], 0.5)
        self.assertGreater(r["wilson95"][1], 0.5)

    def test_repeats_do_not_get_independent_sample_ci(self):
        row = {"scene_id": "same", "success": True, "reward": 15, "difficulty": "near"}
        result = outcome_summary([row, row])
        self.assertEqual(result["unique_layouts"], 1)
        self.assertIsNone(result["wilson95"])

    def test_fingerprint_changes_with_physics(self):
        before = contract(self.cfg)["fingerprint"]
        self.cfg["ball"]["restitution"] = 0.5
        self.assertNotEqual(before, contract(self.cfg)["fingerprint"])


if __name__ == "__main__":
    unittest.main()
