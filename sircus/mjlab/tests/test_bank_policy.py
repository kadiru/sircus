"""Masked categorical learning and immutable inference behavior (Torch CPU)."""

import copy
import importlib.util
from pathlib import Path
import sys
import unittest
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
HAVE_TORCH = importlib.util.find_spec("torch") is not None
if HAVE_TORCH:
    import torch
    from sircus.mjlab.bank_policy import Policy, features, predict, update, weight_hash


@unittest.skipUnless(HAVE_TORCH, "Torch tests run in pinned mjlab environment")
class BankPolicyTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(71)
        self.bucket = torch.tensor([[0.5, -0.4], [0.7, -0.2], [1.0, 0.1]])
        self.values = torch.tensor([[[0.65, -0.04, 0.36, 25.0, 0.12]] * 4] * 3)
        self.values[:, :, 0] += torch.arange(4) * 0.03
        self.mask = torch.tensor(
            [[True, False, True, False], [False] * 4, [False, True, False, False]]
        )

    def test_initial_uniform_valid_mask_and_unresolved_rows(self):
        policy = Policy()
        dist, eligible = policy.distribution(self.bucket, self.values, self.mask)
        torch.testing.assert_close(dist.probs[0], torch.tensor([0.5, 0, 0.5, 0]))
        self.assertEqual(eligible.tolist(), [True, False, True])
        choice = predict(policy, self.bucket, self.values, self.mask)
        self.assertEqual(choice["indices"][2], 1)
        self.assertFalse(choice["eligible"][1])
        self.assertEqual(features(self.bucket, self.values).shape, (3, 4, 9))

    def test_shared_scorer_respects_candidate_permutations(self):
        policy = Policy()
        torch.nn.init.normal_(policy.actor[-1].weight, std=0.3)
        p, _ = policy.distribution(self.bucket, self.values, self.mask)
        order = torch.tensor([3, 0, 1, 2])
        q, _ = policy.distribution(
            self.bucket, self.values[:, order], self.mask[:, order]
        )
        torch.testing.assert_close(q.probs[[0, 2]], p.probs[:, order][[0, 2]])

    def test_ppo_excludes_empty_rows_and_changes_weights(self):
        p = Policy()
        q = copy.deepcopy(p)
        with torch.no_grad():
            dist, _ = p.distribution(self.bucket, self.values, self.mask)
            indices = dist.sample()
            logp = dist.log_prob(indices)
        returns = torch.tensor([1.0, 1e8, -0.5])
        other = returns.clone()
        other[1] = -1e8
        log = update(
            p,
            torch.optim.Adam(p.parameters(), lr=0.0003),
            self.bucket,
            self.values,
            self.mask,
            indices,
            logp,
            returns,
            epochs=3,
        )
        update(
            q,
            torch.optim.Adam(q.parameters(), lr=0.0003),
            self.bucket,
            self.values,
            self.mask,
            indices,
            logp,
            other,
            epochs=3,
        )
        self.assertNotEqual(log["before"], log["after"])
        self.assertEqual(log["after"], weight_hash(q))
        self.assertEqual(log["optimizer_rows"], 2)
        self.assertEqual(log["excluded_all_invalid"], 1)
        self.assertTrue(all(np.isfinite(list(x.values())).all() for x in log["epochs"]))

    def test_all_empty_batch_does_not_update_weights(self):
        p = Policy()
        mask = torch.zeros_like(self.mask)
        idx = torch.zeros(3, dtype=torch.long)
        log = update(
            p,
            torch.optim.Adam(p.parameters()),
            self.bucket,
            self.values,
            mask,
            idx,
            torch.zeros(3),
            torch.ones(3),
        )
        self.assertEqual(log["before"], log["after"])
        self.assertEqual(log["optimizer_rows"], 0)


if __name__ == "__main__":
    unittest.main()
