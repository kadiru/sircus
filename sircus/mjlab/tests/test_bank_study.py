"""Fresh study splits and cached native endpoint replay with CPU mocks."""

from pathlib import Path
import json
import sys
import unittest
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from sircus.mjlab.bank_study import new_splits, run_bank_rows, unresolved_row
from sircus.mjlab.learning_contract import splits, statistics


class BankStudyTests(unittest.TestCase):
    def test_fresh_seeded_splits_have_no_overlap(self):
        grid = splits()["train"]
        pools = new_splits(grid)
        self.assertEqual(pools, new_splits(grid))
        self.assertEqual(
            {k: len(v) for k, v in pools.items()},
            dict(train=99, validation=20, test=50),
        )
        self.assertEqual(pools["train"][:35], grid)
        self.assertEqual(len({tuple(p) for v in pools.values() for p in v}), 169)
        self.assertFalse(
            set(map(tuple, pools["test"])) & set(map(tuple, splits()["test"]))
        )

    def test_unresolved_layout_remains_in_complete_denominator(self):
        row = unresolved_row([0.5, -0.2])
        result = statistics([row, row])
        self.assertEqual(result["attempts"], 2)
        self.assertEqual(result["captures"], 0)
        self.assertEqual(result["invalid"], 2)
        self.assertFalse(row["native_release"])

    def test_cached_endpoint_context_restores_and_does_not_release_empty_masks(self):
        scene = json.loads((ROOT / "sircus/isaac/scene.json").read_text())

        class Fake:
            device = "cpu"

            def __init__(self):
                self.scene = scene
                self.ik = object()
                self.calls = 0

            def run(self, bucket, values, locks, record=False):
                self.calls += 1
                self.assert_shape = (len(bucket), record)
                return [dict(invalid=None, route_success=True) for _ in bucket], {}

        env = Fake()
        original = env.ik
        xy = np.array([[0.5, -0.4], [0.7, -0.4]])
        values = np.tile([0.65, -0.04, 0.36, 25, 0.12], (2, 1))
        q = np.zeros((2, 6))
        rows = run_bank_rows(env, xy, values, q, q, [True, False], record=True)
        self.assertIs(env.ik, original)
        self.assertEqual(env.calls, 1)
        self.assertEqual(env.assert_shape, (1, True))
        self.assertTrue(rows[0]["native_release"])
        self.assertFalse(rows[1]["native_release"])
        run_bank_rows(env, xy, values, q, q, [False, False])
        self.assertEqual(env.calls, 1)


if __name__ == "__main__":
    unittest.main()
