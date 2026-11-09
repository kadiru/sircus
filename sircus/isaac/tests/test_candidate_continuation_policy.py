"""Synthetic no-update finite-law checks; no optimizer, simulator or native work."""

import copy
from dataclasses import FrozenInstanceError, replace
import json
import math
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np

SOURCE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOURCE))
import candidate_continuation_policy as candidate
from continuous_policy import ContinuousPolicy, torch


def action(latent, identifier="synthetic"):
    values = np.asarray(candidate.LOW) + (np.tanh(latent)+1)*(np.asarray(candidate.HIGH)-candidate.LOW)/2
    return dict(zip(candidate.ACTION_NAMES, values.tolist()), id=identifier)


def choice(identifier, latent):
    return dict(candidate_id=identifier, action=action(latent, identifier))


class FixedSupportTests(unittest.TestCase):
    def test_stdlib_contract_and_policy_use_identical_probability_law(self):
        from candidate_continuation_contract import PROBABILITY_LAW
        self.assertEqual(PROBABILITY_LAW, candidate.PROBABILITY_LAW)

    def test_parent_uses_exact_original_latent_and_others_use_affine_inverse(self):
        # Saved float32 tanh actions need not invert to the original latent.
        parent = action([0, .2, 0, 0, 0])
        original = [0, .20000001788139343, 0, 0, 0]
        rows = [dict(candidate_id="parent", action=parent), choice("other", [.4, -.2, .1, 0, 0])]
        support = candidate.freeze_support("train-000", rows, parent, original)
        self.assertEqual(support.latents[0], tuple(original))
        np.testing.assert_allclose(support.latents[1], [.4, -.2, .1, 0, 0], atol=1e-15, rtol=0)
        self.assertEqual(support.parent_index, 0)
        self.assertEqual(support.candidate_ids, ("parent", "other"))

    def test_actions_latents_and_order_are_immutable_and_serialization_is_bound(self):
        parent = action([0]*5)
        rows = [choice("z", [.1, 0, 0, 0, 0]), dict(candidate_id="a", action=parent)]
        support = candidate.freeze_support("train-000", rows, parent, [0]*5)
        before = support.to_dict()
        rows[0]["action"]["height_m"] = .59
        support.actions[0]["height_m"] = .58
        support.action(0)["height_m"] = .57
        self.assertEqual(support.to_dict(), before)
        with self.assertRaises(FrozenInstanceError):
            support.parent_index = 0
        restored = candidate.FrozenCandidateSupport.from_dict(json.loads(json.dumps(before)))
        self.assertEqual(restored, support)
        altered = copy.deepcopy(before)
        altered["latents"][0][0] += .01
        with self.assertRaisesRegex(ValueError, "fingerprint"):
            candidate.FrozenCandidateSupport.from_dict(altered)
        exact_rows = [dict(candidate_id=name, action=shot) for name, shot in
                      zip(support.candidate_ids, support.actions)]
        reordered = candidate.freeze_support("train-000", list(reversed(exact_rows)), parent, [0]*5)
        self.assertNotEqual(support.support_fingerprint, reordered.support_fingerprint)

    def test_empty_support_is_explicit_decline_without_an_action(self):
        with self.assertRaises(candidate.NoQualifiedCandidates) as caught:
            candidate.freeze_support("train-000", [], action([0]*5), [0]*5)
        self.assertEqual(caught.exception.reason, "no_qualified_candidate_support")
        self.assertEqual(caught.exception.context_id, "train-000")
        self.assertFalse(hasattr(caught.exception, "action"))

    def test_boundary_fails_closed_except_exact_parent_with_saved_latent(self):
        parent = action([0]*5)
        boundary = dict(zip(candidate.ACTION_NAMES, candidate.LOW), id="boundary")
        with self.assertRaisesRegex(ValueError, "saturated latent boundary"):
            candidate.freeze_support("train-000", [dict(candidate_id="edge", action=boundary)], parent, [0]*5)
        support = candidate.freeze_support("train-000", [dict(candidate_id="edge", action=boundary)], boundary, [-20]*5)
        self.assertEqual(support.latents, ((-20.,)*5,))

    def test_duplicate_identity_action_context_and_nonfinite_input_fail_closed(self):
        parent = action([0]*5)
        cases = [([choice("same", [.1, 0, 0, 0, 0]), choice("same", [.2, 0, 0, 0, 0])], "candidate identity"),
                 ([choice("a", [.1, 0, 0, 0, 0]), choice("b", [.1, 0, 0, 0, 0])], "physical candidate"),
                 ([dict(choice("a", [.1, 0, 0, 0, 0]), cell_id="test-000")], "context identity")]
        for rows, message in cases:
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                candidate.freeze_support("train-000", rows, parent, [0]*5)
        with self.assertRaisesRegex(ValueError, "frozen-parent latent"):
            candidate.freeze_support("train-000", [choice("a", [.1, 0, 0, 0, 0])], parent, [math.nan]*5)
        with self.assertRaisesRegex(ValueError, "out of bounds"):
            candidate.freeze_support("train-000", [choice("a", [.1, 0, 0, 0, 0])], parent, [0]*5).action(-1)


@unittest.skipIf(torch is None, "configured PyTorch environment unavailable")
class CandidateDistributionTests(unittest.TestCase):
    def make_policy(self, mean=None):
        policy = ContinuousPolicy()
        # Controlled synthetic initialization, never a learned/optimizer update.
        with torch.no_grad():
            for parameter in policy.actor.parameters():
                parameter.zero_()
            policy.logstd.zero_()
            if mean is not None:
                policy.actor[-1].bias.copy_(torch.tensor(mean, dtype=torch.float32))
        return policy

    def observations(self, radii=(.74,)):
        return torch.tensor([[0, 0, radius, 0, 0] for radius in radii], dtype=torch.float32)

    def support(self, latents, *, names=None, parent=None, original=None, context="train-000"):
        names = names or [f"candidate-{i}" for i in range(len(latents))]
        parent = [0]*5 if parent is None else parent
        rows = [choice(name, latent) for name, latent in zip(names, latents)]
        return candidate.freeze_support(context, rows, action(parent), parent if original is None else original)

    def test_scores_match_frozen_numpy_law_without_a_tanh_jacobian(self):
        policy = self.make_policy()
        observations = self.observations()
        normal = policy.distribution(observations)
        original = normal.loc[0].detach().numpy().astype(float)
        support = self.support([original, [.5, -.2, .4, 0, .1], [0, 0, 0, 0, 3]], parent=original)
        distribution = candidate.CandidateContinuationPolicy(policy).distribution(observations, [support])
        mean, scale = normal.loc[0].detach().numpy().astype(float), normal.scale[0].detach().numpy().astype(float)
        z = np.asarray(support.latents)
        expected = -.5*np.square((z-mean)/scale).sum(axis=-1)-np.log(scale).sum()-2.5*math.log(2*math.pi)
        np.testing.assert_allclose(distribution.scores[0].detach().numpy(), expected, atol=1e-12, rtol=0)
        expected_logp = expected-(expected.max()+np.log(np.exp(expected-expected.max()).sum()))
        np.testing.assert_allclose(distribution.log_probabilities[0].detach().numpy(), expected_logp, atol=1e-12, rtol=0)
        self.assertEqual(distribution.deterministic().tolist(), [0])
        self.assertEqual(distribution.probability_law, candidate.PROBABILITY_LAW)

    def test_real_parent_distribution_includes_state_dependent_radius_prior(self):
        policy = self.make_policy()
        observations = self.observations((.35, 1.10))
        supports = [self.support([[0]*5, [0, 1, 0, 0, 0]], context=f"train-{i:03d}") for i in range(2)]
        with patch.object(policy, "distribution", wraps=policy.distribution) as original_distribution:
            result = candidate.CandidateContinuationPolicy(policy).distribution(observations, supports)
        original_distribution.assert_called_once_with(observations)
        normal = policy.distribution(observations)
        self.assertLess(float(normal.loc[0, 1]), 0)
        self.assertGreater(float(normal.loc[1, 1]), 0)
        for index, support in enumerate(supports):
            expected = torch.distributions.Normal(normal.loc[index].double(), normal.scale[index].double()).log_prob(
                torch.tensor(support.latents, dtype=torch.float64)).sum(-1)
            torch.testing.assert_close(result.scores[index], expected, atol=1e-12, rtol=0)
        self.assertNotEqual(result.deterministic()[0].item(), result.deterministic()[1].item())

    def test_original_parent_latent_does_not_move_or_force_parent_maximal(self):
        support = self.support([[0]*5, [2, 0, 0, 0, 0]], names=["parent", "other"])
        original = support.latents
        first = candidate.CandidateContinuationPolicy(self.make_policy()).distribution(self.observations(), [support])
        shifted = candidate.CandidateContinuationPolicy(self.make_policy([2, 0, 0, 0, 0])).distribution(self.observations(), [support])
        self.assertEqual(first.deterministic().tolist(), [0])
        self.assertEqual(shifted.deterministic().tolist(), [1])
        self.assertGreater(float(shifted.log_probabilities[0][1]), float(shifted.log_probabilities[0][0]))
        self.assertEqual(support.latents, original)

    def test_ragged_singleton_and_rare_probabilities_have_exact_entropy_and_finite_gradients(self):
        policy = self.make_policy()
        supports = [self.support([[0]*5]),
                    self.support([[0]*5, [math.sqrt(218), 0, 0, 0, 0]], context="train-001")]
        distribution = candidate.CandidateContinuationPolicy(policy).distribution(self.observations((.74, .74)), supports)
        self.assertEqual(tuple(len(row) for row in distribution.log_probabilities), (1, 2))
        self.assertEqual(distribution.log_prob([0, 1]).dtype, torch.float64)
        self.assertEqual(float(distribution.entropy()[0]), 0)
        self.assertLess(float(distribution.log_probabilities[1][1]), -108)
        self.assertGreater(float(distribution.probabilities[1][1]), 0)
        expected_entropy = -(distribution.probabilities[1]*distribution.log_probabilities[1]).sum()
        torch.testing.assert_close(distribution.entropy()[1], expected_entropy, atol=0, rtol=0)
        loss = -distribution.log_prob([0, 1]).mean()-.005*distribution.entropy().mean()+distribution.values.square().mean()
        loss.backward()
        gradients = [parameter.grad for parameter in policy.parameters()]
        self.assertTrue(all(gradient is not None and torch.isfinite(gradient).all().item() for gradient in gradients))
        self.assertTrue(any(gradient.abs().sum().item() > 0 for gradient in gradients))
        self.assertTrue(all(parameter.dtype == torch.float32 for parameter in policy.parameters()))

    def test_singleton_zero_actor_gradient_still_has_real_critic_gradient(self):
        policy = self.make_policy()
        distribution = candidate.CandidateContinuationPolicy(policy).distribution(self.observations(), [self.support([[0]*5])])
        loss = -distribution.log_prob([0]).sum()+(distribution.values-100).square().sum()
        loss.backward()
        self.assertTrue(all(parameter.grad.abs().sum().item() == 0 for parameter in policy.actor.parameters()))
        self.assertEqual(policy.logstd.grad.abs().sum().item(), 0)
        self.assertTrue(any(parameter.grad.abs().sum().item() > 0 for parameter in policy.critic.parameters()))

    def test_batch_support_counts_cover_existing_maximum_without_padding(self):
        supports = [self.support([[0]*5]),
                    self.support([[0]*5, [.1, 0, 0, 0, 0]], context="train-001"),
                    self.support([[float(x), 0, 0, 0, 0] for x in np.linspace(-3, 3, 176)],
                                 context="train-002")]
        distribution = candidate.CandidateContinuationPolicy(self.make_policy()).distribution(
            self.observations((.74, .74, .74)), supports)
        self.assertEqual([len(row) for row in distribution.scores], [1, 2, 176])
        self.assertEqual(distribution.sample(generator=torch.Generator().manual_seed(1)).shape, (3,))
        for probability in distribution.probabilities:
            torch.testing.assert_close(probability.sum(), torch.ones((), dtype=torch.float64), atol=1e-12, rtol=0)

    def test_sampling_uses_same_finite_law_and_dedicated_generator(self):
        policy = self.make_policy()
        support = self.support([[0]*5, [.8, 0, 0, 0, 0], [-.8, 0, 0, 0, 0]])
        distribution = candidate.CandidateContinuationPolicy(policy).distribution(self.observations(), [support])
        rng = torch.Generator().manual_seed(12345)
        expected_rng = torch.Generator().manual_seed(12345)
        expected = torch.multinomial(distribution.probabilities[0], 1, generator=expected_rng)
        actual = distribution.sample(generator=rng)
        self.assertEqual(actual.tolist(), expected.tolist())
        self.assertEqual(actual.dtype, torch.int64)
        torch.testing.assert_close(distribution.log_prob(actual), distribution.log_probabilities[0][actual])
        many = candidate.CandidateContinuationPolicy(policy).distribution(
            self.observations((.74,)*2000), [support]*2000).sample(generator=torch.Generator().manual_seed(900))
        frequencies = torch.bincount(many, minlength=3).double()/len(many)
        torch.testing.assert_close(frequencies, distribution.probabilities[0], atol=.04, rtol=0)

    def test_deterministic_ties_use_parent_then_lexical_id_with_fixed_tolerance(self):
        def finite(scores, *, parent=None):
            scores = torch.tensor(scores, dtype=torch.float64)
            return candidate.CandidateDistribution(("train-000",), ("support",), (("z", "a", "b"),),
                (scores,), (scores.log_softmax(0),), torch.zeros(1), (parent,))
        self.assertEqual(finite([0, 0, 0], parent=0).deterministic().tolist(), [0])
        self.assertEqual(finite([0, 0, 0]).deterministic().tolist(), [1])
        self.assertEqual(finite([0, .5e-12, -1], parent=0).deterministic().tolist(), [0])
        self.assertEqual(finite([0, 2e-12, -1], parent=0).deterministic().tolist(), [1])
        self.assertEqual(finite([0, -.5e-12, -1]).deterministic().tolist(), [1])

    def test_invalid_indices_law_shapes_nonfinite_scores_and_empty_batches_fail(self):
        distribution = candidate.CandidateContinuationPolicy(self.make_policy()).distribution(
            self.observations(), [self.support([[0]*5])])
        for indices in ([], [1], [-1], [True], [0.], [0, 0], torch.tensor([0.])):
            with self.subTest(indices=indices), self.assertRaises(ValueError):
                distribution.log_prob(indices)
        with self.assertRaisesRegex(ValueError, "coverage"):
            candidate.CandidateContinuationPolicy(self.make_policy()).distribution(self.observations((.74, .74)), [self.support([[0]*5])])
        with self.assertRaises(ValueError):
            replace(distribution, scores=(torch.tensor([math.nan], dtype=torch.float64),))
        with self.assertRaises(ValueError):
            replace(distribution, log_probabilities=(torch.tensor([1.], dtype=torch.float64),))
        with self.assertRaisesRegex(ValueError, "critic values"):
            replace(distribution, values=torch.zeros(1, dtype=torch.int64))
        with self.assertRaises(ValueError):
            candidate.CandidateContinuationPolicy(self.make_policy()).distribution(self.observations(), [])


if __name__ == "__main__":
    unittest.main()
