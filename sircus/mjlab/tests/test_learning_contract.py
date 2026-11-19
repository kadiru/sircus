"""Learning splits, positional locks, reward gates and actual optimizer checks."""

import importlib.util
from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from sircus.mjlab.learning_contract import (
    HIGH,
    LOW,
    WARM,
    actions,
    constraints,
    latents,
    observations,
    project,
    proposals,
    reward,
    splits,
    statistics,
)

HAVE_TORCH = importlib.util.find_spec("torch") is not None
if HAVE_TORCH:
    import torch
    from sircus.mjlab.learning_policy import Policy, update, warm_start, weight_hash


class LearningContractTests(unittest.TestCase):
    def test_seeded_splits_are_fixed_disjoint_and_inside_declared_region(self):
        a, b = splits(), splits()
        self.assertEqual(a, b)
        self.assertEqual(
            {k: len(v) for k, v in a.items()}, dict(train=35, validation=20, test=50)
        )
        all_xy = [tuple(xy) for group in a.values() for xy in group]
        self.assertEqual(len(set(all_xy)), 105)
        xy = np.asarray(all_xy)
        self.assertTrue(((xy >= [0.1, -0.65]) & (xy <= [1.24, 0.25])).all())
        self.assertEqual(len(set(x[0] for x in a["train"])), 7)
        self.assertEqual(len(set(x[1] for x in a["train"])), 5)

    def test_normalized_actions_and_seeded_proposals_remain_bounded(self):
        rng = np.random.default_rng(42)
        values = rng.uniform(
            LOW + 0.01 * (HIGH - LOW), HIGH - 0.01 * (HIGH - LOW), (30, 5)
        )
        np.testing.assert_allclose(actions(latents(values)), values, atol=1e-12)
        bx = np.asarray(splits()["train"])
        first = proposals(bx, np.random.default_rng(42))
        second = proposals(bx, np.random.default_rng(42))
        np.testing.assert_array_equal(first, second)
        self.assertTrue(np.isfinite(first).all())
        self.assertTrue((first >= LOW).all() and (first <= HIGH).all())

    def test_board_lock_keeps_center_and_dropper_lock_keeps_release_center(self):
        values = np.tile(WARM, (3, 1))
        lock = constraints(3)
        lock["board"][0] = [0.62, -0.1, 0.33]
        lock["dropper"][1] = [0.68, 0.1, 0.7]
        lock["board"][2] = [0.6, -0.02, 0.35]
        lock["dropper"][2] = [0.6, -0.02, 0.7]
        projected, valid = project(values, lock)
        self.assertTrue(valid.all())
        np.testing.assert_array_equal(projected[0, :3], lock["board"][0])
        np.testing.assert_array_equal(projected[1, :2], lock["dropper"][1, :2])
        self.assertAlmostEqual(projected[1, 2] + 0.0375 + projected[1, 4], 0.7)
        np.testing.assert_array_equal(projected[2, :3], lock["board"][2])
        self.assertAlmostEqual(projected[2, 2] + 0.0375 + projected[2, 4], 0.7)
        # Projecting must not mutate the caller's proposal or locks.
        np.testing.assert_array_equal(values, np.tile(WARM, (3, 1)))

    def test_conflicting_locks_and_out_of_bounds_gap_are_rejected(self):
        lock = constraints(3)
        lock["board"][0] = [0.6, 0, 0.3]
        lock["dropper"][0] = [0.63, 0, 0.65]
        lock["dropper"][1] = [0.6, 0, 0.1]
        lock["board"][2] = [9, 0, 0.3]
        _, valid = project(np.tile(WARM, (3, 1)), lock)
        self.assertFalse(valid.any())

    def test_masks_distinguish_absent_from_zero_coordinates(self):
        lock = constraints(2)
        lock["board"][1] = [0, 0, 0.3]
        lock["dropper"][1] = [0, 0, 0.7]
        obs = observations([[0.6, -0.4]] * 2, lock)
        self.assertEqual(obs.shape, (2, 10))
        self.assertTrue(np.isfinite(obs).all())
        np.testing.assert_array_equal(obs[:, 2:4], [[0, 0], [1, 1]])
        np.testing.assert_array_equal(obs[0, 4:], [0] * 6)

    def test_partial_and_infinite_locks_cannot_silently_become_unlocked(self):
        for bad in ([0.6, np.nan, 0.3], [0.6, float("inf"), 0.3]):
            lock = constraints(1)
            lock["board"][0] = bad
            with self.assertRaises(ValueError):
                project(np.tile(WARM, (1, 1)), lock)
            with self.assertRaises(ValueError):
                observations([[0.6, -0.4]], lock)

    @staticmethod
    def row(**overrides):
        value = dict(
            invalid=None,
            board_bounce=True,
            closest_opening_distance_m=0.01,
            max_retention_s=1.0,
            route_success=True,
            route_quality=2,
            single_bounce_success=True,
            external_contact_steps=0,
        )
        value.update(overrides)
        return value

    def test_invalid_and_external_routes_cannot_receive_capture_bonus(self):
        clean = self.row()
        invalid = dict(clean, invalid="endpoint_ik")
        external = dict(clean, external_contact_steps=1)
        external_miss = dict(
            external,
            max_retention_s=0,
            route_success=False,
            route_quality=0,
            single_bounce_success=False,
        )
        self.assertEqual(reward(invalid), -1)
        self.assertEqual(reward(external), reward(external_miss))
        self.assertGreater(reward(clean), reward(external) + 3)
        stats = statistics([clean, invalid, external_miss])
        self.assertEqual(stats["attempts"], 3)
        self.assertEqual(stats["captures"], 1)
        self.assertEqual(stats["invalid"], 1)


@unittest.skipUnless(
    HAVE_TORCH, "Torch optimizer checks run in pinned mjlab environment"
)
class LearningOptimizerTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        torch.set_num_threads(1)

    def test_train_only_warm_start_changes_actor_and_reduces_fit_error(self):
        policy = Policy()
        rng = np.random.default_rng(42)
        obs = rng.normal(0, 0.4, (20, 10)).astype(np.float32)
        values = np.tile(WARM, (20, 1))
        values[:, 0] += obs[:, 0] * 0.04
        values[:, 2] += obs[:, 1] * 0.02
        x = torch.tensor(obs)
        target = torch.tensor(latents(values), dtype=torch.float32)
        before = float(((policy.actor(x) - target) ** 2).mean().detach())
        critic = {k: v.clone() for k, v in policy.critic.state_dict().items()}
        logstd = policy.logstd.detach().clone()
        warm_start(policy, obs, values, np.ones(20), steps=80)
        after = float(((policy.actor(x) - target) ** 2).mean().detach())
        self.assertLess(after, before * 0.2)
        for k, v in policy.critic.state_dict().items():
            self.assertTrue(torch.equal(v, critic[k]))
        self.assertTrue(torch.equal(policy.logstd.detach(), logstd))

    def test_terminal_ppo_performs_measured_finite_weight_update(self):
        policy = Policy()
        obs = torch.randn(32, 10) * 0.3
        with torch.no_grad():
            dist = policy.distribution(obs)
            z = dist.sample()
            old_logp = dist.log_prob(z).sum(-1)
            returns = (z[:, 0] - z[:, 2]).tanh()
        before = weight_hash(policy)
        optimizer = torch.optim.Adam(policy.parameters(), lr=0.0003)
        log = update(policy, optimizer, obs, z, old_logp, returns, epochs=3)
        self.assertEqual(log["before"], before)
        self.assertNotEqual(log["before"], log["after"])
        self.assertEqual(log["after"], weight_hash(policy))
        self.assertEqual(len(log["epochs"]), 3)
        self.assertTrue(
            all(np.isfinite(list(row.values())).all() for row in log["epochs"])
        )


if __name__ == "__main__":
    unittest.main()
