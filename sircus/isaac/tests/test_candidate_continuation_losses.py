"""Synthetic loss/backward checks only: no optimizer, native requests or training."""

from dataclasses import replace
import hashlib
import math
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

ISAAC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ISAAC))
import candidate_continuation_policy as categorical
import candidate_continuation_losses as objectives


@unittest.skipIf(categorical.torch is None, "PyTorch CPU environment is unavailable")
class ContinuationLossTests(unittest.TestCase):
    def setUp(self):
        self.torch = categorical.torch
        self.torch.set_num_threads(1)
        self.counts = {f"train-{i:03d}": 73 for i in range(21)}
        self.counts["train-000"] = 1
        self.counts["train-020"] = 146
        self.assertEqual(sum(self.counts.values()), 1534)

    def distribution(self, rows, *, contexts=None, values=None, gradients=False):
        contexts = tuple(contexts or (f"train-{i:03d}" for i in range(len(rows))))
        scores = tuple(self.torch.tensor(row, dtype=self.torch.float64,
                                        requires_grad=gradients) for row in rows)
        ids = tuple(tuple(f"{context}-candidate-{j}" for j in range(len(row)))
                    for context, row in zip(contexts, rows))
        fingerprints = tuple(hashlib.sha256(repr((context, choices)).encode()).hexdigest()
                             for context, choices in zip(contexts, ids))
        value_tensor = self.torch.tensor(values if values is not None else [0.] * len(rows),
                                         dtype=self.torch.float64, requires_grad=gradients)
        return categorical.CandidateDistribution(
            context_ids=contexts, support_fingerprints=fingerprints, candidate_ids=ids,
            scores=scores, log_probabilities=tuple(self.torch.log_softmax(row, 0) for row in scores),
            values=value_tensor)

    def subset(self, distribution, indices):
        return categorical.CandidateDistribution(
            context_ids=tuple(distribution.context_ids[i] for i in indices),
            support_fingerprints=tuple(distribution.support_fingerprints[i] for i in indices),
            candidate_ids=tuple(distribution.candidate_ids[i] for i in indices),
            scores=tuple(distribution.scores[i] for i in indices),
            log_probabilities=tuple(distribution.log_probabilities[i] for i in indices),
            values=distribution.values[self.torch.tensor(indices)])

    def assertFiniteGradients(self, distribution):
        for scores in distribution.scores:
            self.assertIsNotNone(scores.grad)
            self.assertTrue(self.torch.isfinite(scores.grad).all())

    def test_demo_weights_use_global_context_counts_and_actual_batch_mean(self):
        current = self.distribution([[0., 1.], [0., 1.], [0., 1.]],
                                    contexts=["train-000", "train-020", "train-020"], gradients=True)
        selected = [0, 0, 1]
        before = [value.detach().clone() for value in current.scores]
        actual = objectives.context_balanced_demo_nll(current, selected, self.counts)
        logps = current.log_prob(selected)
        weights = self.torch.tensor([1534 / 21, 1534 / (21 * 146), 1534 / (21 * 146)],
                                    dtype=self.torch.float64)
        self.torch.testing.assert_close(actual, -(weights * logps).sum() / 3)
        renormalized = -(weights * logps).sum() / weights.sum()
        self.assertGreater(abs(float((actual - renormalized).detach())), 1)
        actual.backward()
        self.assertFiniteGradients(current)
        for previous, scores in zip(before, current.scores):
            self.torch.testing.assert_close(previous, scores.detach(), rtol=0, atol=0)
        self.assertIsNone(current.values.grad)

    def test_complete_demo_pass_gives_each_success_context_equal_total_weight(self):
        full = self.distribution([[i / 10, -i / 15] for i in range(21)])
        indices = [i for i, context in enumerate(full.context_ids) for _ in range(self.counts[context])]
        batch = self.subset(full, indices)
        actual = objectives.context_balanced_demo_nll(batch, [0] * 1534, self.counts)
        expected = -self.torch.stack([row[0] for row in full.log_probabilities]).mean()
        self.torch.testing.assert_close(actual, expected)

    def test_demo_rejects_invented_targets_wrong_global_counts_and_noninteger_selection(self):
        current = self.distribution([[0., 1.]])
        for counts in ({"train-000": 1534}, {**self.counts, "train-020": 145},
                       {**self.counts, "train-020": 146.0}):
            with self.assertRaisesRegex(ValueError, "audited counts"):
                objectives.context_balanced_demo_nll(current, [0], counts)
        absent = self.distribution([[0., 1.]], contexts=["train-034"])
        with self.assertRaisesRegex(ValueError, "no audited success"):
            objectives.context_balanced_demo_nll(absent, [0], self.counts)
        for selected in ([True], [-1], [2], self.torch.tensor([0.] )):
            with self.assertRaises(ValueError):
                objectives.context_balanced_demo_nll(current, selected, self.counts)

    def test_forward_kl_is_exact_equal_context_mean_with_detached_parent(self):
        parent_rows = [[0., -109.]] + [[0., 1., -3.]] * 34
        current_rows = [[-1., 0.]] + [[1., -1., -2.]] * 34
        parent = self.distribution(parent_rows, gradients=True)
        current = self.distribution(current_rows, gradients=True)
        actual = objectives.exact_forward_kl(parent, current)
        expected = sum(sum(float((p.exp() * (p - q)).detach()) for p, q in zip(old, new))
                       for old, new in zip(parent.log_probabilities, current.log_probabilities)) / 35
        self.assertAlmostEqual(float(actual.detach()), expected, places=13)
        actual.backward()
        self.assertFiniteGradients(current)
        self.assertTrue(all(row.grad is None for row in parent.scores))
        self.assertIsNone(parent.values.grad)
        self.assertIsNone(current.values.grad)

    def test_anchor_requires_all_35_unique_identical_context_supports(self):
        parent = self.distribution([[0., 1.]] * 35)
        current = self.distribution([[1., 0.]] * 35)
        short_parent, short_current = self.subset(parent, list(range(34))), self.subset(current, list(range(34)))
        with self.assertRaisesRegex(ValueError, "all 35 distinct"):
            objectives.exact_forward_kl(short_parent, short_current)
        duplicate_indices = [0] * 35
        with self.assertRaisesRegex(ValueError, "all 35 distinct"):
            objectives.exact_forward_kl(self.subset(parent, duplicate_indices),
                                        self.subset(current, duplicate_indices))
        ids = list(current.candidate_ids)
        ids[0] = tuple(reversed(ids[0]))
        with patch.object(categorical.CandidateDistribution, "log_prob", side_effect=AssertionError("arithmetic")):
            with self.assertRaisesRegex(ValueError, "ordered candidate IDs changed"):
                objectives.exact_forward_kl(parent, replace(current, candidate_ids=tuple(ids)))

    def test_imitation_adds_full_35_context_anchor_once_and_trains_no_success_context_preference(self):
        parent = self.distribution([[0., 1.]] * 35, gradients=True)
        current = self.distribution([[1., 0.]] * 35, gradients=True)
        demos = self.subset(current, [0, 20])
        result = objectives.imitation_objective(demos, [0, 1], self.counts, parent, current)
        expected = objectives.context_balanced_demo_nll(demos, [0, 1], self.counts)
        expected += .05 * objectives.exact_forward_kl(parent, current)
        self.torch.testing.assert_close(result.total, expected)
        result.total.backward()
        self.assertFiniteGradients(current)
        self.assertGreater(float(current.scores[34].grad.abs().sum()), 0)
        self.assertTrue(all(row.grad is None for row in parent.scores))
        wrong_support = replace(demos, support_fingerprints=("a" * 64, demos.support_fingerprints[1]))
        with self.assertRaisesRegex(ValueError, "differs from TRAIN anchor"):
            objectives.imitation_objective(wrong_support, [0, 1], self.counts, parent, current)

    def test_ppo_uses_selected_categorical_ratios_frozen_advantages_exact_entropy_and_critic(self):
        old = self.distribution([[math.log(.25), math.log(.75)],
                                 [math.log(.8), math.log(.2)], [math.log(.5), math.log(.5)]],
                                values=[1., -1., 2.], gradients=True)
        returns = self.torch.tensor([3., -2., 5.], dtype=self.torch.float64, requires_grad=True)
        rollout = objectives.freeze_terminal_rollout(old, [0, 1, 1], returns)
        current = self.distribution([[math.log(.5), math.log(.5)],
                                     [math.log(.4), math.log(.6)], [math.log(.4), math.log(.6)]],
                                    values=[1.5, -.5, 4.], gradients=True)
        before = [value.detach().clone() for value in current.scores] + [current.values.detach().clone()]
        result = objectives.categorical_ppo_loss(current, rollout)
        expected_ratios = self.torch.tensor([2., 3., 1.2], dtype=self.torch.float64)
        advantages = self.torch.tensor([2., -1., 3.], dtype=self.torch.float64)
        advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-6)
        actor = -self.torch.minimum(expected_ratios * advantages,
                                    expected_ratios.clamp(.8, 1.2) * advantages).mean()
        critic = ((current.values - returns.detach()) ** 2).mean()
        entropy = sum(-sum(p * math.log(p) for p in row)
                      for row in ([.5, .5], [.4, .6], [.4, .6])) / 3
        self.torch.testing.assert_close(result.ratios, expected_ratios)
        self.torch.testing.assert_close(result.advantages, advantages)
        self.torch.testing.assert_close(result.actor_loss, actor)
        self.torch.testing.assert_close(result.critic_loss, critic)
        self.assertAlmostEqual(float(result.entropy.detach()), entropy, places=14)
        self.torch.testing.assert_close(result.total, actor + .5 * critic - .005 * entropy)
        self.assertFalse(result.advantages.requires_grad)
        self.assertFalse(result.old_selected_log_probabilities.requires_grad)
        result.total.backward()
        self.assertFiniteGradients(current)
        self.assertTrue(self.torch.isfinite(current.values.grad).all())
        self.assertTrue(all(scores.grad is None for scores in old.scores))
        self.assertIsNone(old.values.grad)
        self.assertIsNone(returns.grad)
        self.torch.testing.assert_close(current.values.grad, (current.values.detach() - returns.detach()) / 3)
        for previous, value in zip(before, [*current.scores, current.values]):
            self.torch.testing.assert_close(previous, value.detach(), rtol=0, atol=0)

    def test_rollout_snapshots_are_detached_immutable_and_reused_without_recomputation(self):
        old = self.distribution([[0., 1.], [1., 0.]], values=[1., 4.], gradients=True)
        returns = self.torch.tensor([-1., 7.], dtype=self.torch.float64)
        rollout = objectives.freeze_terminal_rollout(old, [0, 1], returns)
        saved = (rollout.old_values, rollout.old_selected_log_probabilities, rollout.terminal_returns,
                 rollout.advantages)
        with self.torch.no_grad():
            old.values.add_(100)
            returns.add_(100)
        self.assertEqual(saved, (rollout.old_values, rollout.old_selected_log_probabilities,
                                rollout.terminal_returns, rollout.advantages))
        changed_values = self.distribution([[.2, .8], [.8, .2]], values=[50., -50.])
        first = objectives.categorical_ppo_loss(changed_values, rollout)
        second = objectives.categorical_ppo_loss(changed_values, rollout)
        self.torch.testing.assert_close(first.advantages, second.advantages, rtol=0, atol=0)
        self.torch.testing.assert_close(first.advantages,
                                        self.torch.tensor(rollout.advantages, dtype=self.torch.float64))

    def test_singleton_support_has_zero_actor_gradient_but_real_terminal_critic_target(self):
        old = self.distribution([[3.]], values=[2.], gradients=True)
        rollout = objectives.freeze_terminal_rollout(old, [0], [7.])
        self.assertEqual(rollout.advantages, (5.,))
        current = self.distribution([[-109.]], values=[3.], gradients=True)
        result = objectives.categorical_ppo_loss(current, rollout)
        self.assertEqual(float(result.actor_loss.detach()), -5.)
        self.assertEqual(float(result.entropy.detach()), 0.)
        self.assertEqual(float(result.critic_loss.detach()), 16.)
        result.total.backward()
        self.torch.testing.assert_close(current.scores[0].grad, self.torch.zeros(1, dtype=self.torch.float64))
        self.torch.testing.assert_close(current.values.grad, self.torch.tensor([-4.], dtype=self.torch.float64))

    def test_rare_selected_probability_keeps_finite_ratios_losses_and_gradients(self):
        old = self.distribution([[0., -109.]], values=[0.], gradients=True)
        current = self.distribution([[0., -108.]], values=[.2], gradients=True)
        rollout = objectives.freeze_terminal_rollout(old, [1], [1.])
        result = objectives.categorical_ppo_loss(current, rollout)
        self.assertAlmostEqual(float(result.ratios.detach()[0]), math.e, places=13)
        result.total.backward()
        self.assertFiniteGradients(current)
        self.assertTrue(self.torch.isfinite(current.values.grad).all())

    def test_ppo_rejects_legacy_buffers_wrong_laws_and_changed_support_before_arithmetic(self):
        current = self.distribution([[0., 1.], [1., 0.]])
        rollout = objectives.freeze_terminal_rollout(current, [0, 1], [-1., 1.])
        bad_rollouts = [dict(latent=[[0.] * 5] * 2, old_logp=[-1., -1.]),
                        replace(rollout, probability_law="continuous-tanh-gaussian"),
                        replace(rollout, candidate_ids=()),
                        replace(rollout, support_fingerprints=("a" * 64,) * 2),
                        replace(rollout, candidate_ids=(tuple(reversed(rollout.candidate_ids[0])),
                                                       rollout.candidate_ids[1]))]
        with patch.object(categorical.CandidateDistribution, "log_prob", side_effect=AssertionError("arithmetic")):
            for invalid in bad_rollouts:
                with self.assertRaises(ValueError):
                    objectives.categorical_ppo_loss(current, invalid)
            with self.assertRaisesRegex(ValueError, "probability law/schema"):
                objectives.categorical_ppo_loss(replace(current, probability_law="continuous"), rollout)
            with self.assertRaisesRegex(ValueError, "not a legacy continuous buffer"):
                objectives.categorical_ppo_loss(dict(old_logp=[0., 0.]), rollout)

    def test_terminal_returns_and_nonfinite_ratios_fail_closed(self):
        current = self.distribution([[0., 1.]])
        for returns in ([], [float("nan")], [float("inf")], [1., 2.]):
            with self.assertRaisesRegex(ValueError, "finite terminal return"):
                objectives.freeze_terminal_rollout(current, [0], returns)
        rollout = objectives.freeze_terminal_rollout(current, [0], [1.])
        with self.assertRaisesRegex(ValueError, "Nonfinite categorical PPO"):
            objectives.categorical_ppo_loss(current, replace(rollout, old_selected_log_probabilities=(-1000.,)))
        with self.assertRaisesRegex(ValueError, "advantages differ"):
            objectives.categorical_ppo_loss(current, replace(rollout, advantages=(100.,)))


if __name__ == "__main__":
    unittest.main()
