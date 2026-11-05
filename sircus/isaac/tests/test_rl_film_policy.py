"""No-native tests for geometry-conditioned categorical shot PPO.

The learnable bandit below is synthetic unit evidence only. It is never used as
native training evidence, a successful shot catalogue or a simulator result.
"""

import copy
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ISAAC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ISAAC))
import rl_film_policy as module
from expanded_farm import ACTION_NAMES, HIGH, LOW, make_spec, shot


class GeometryInputsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        scene = json.loads((ISAAC / "scene.json").read_text())
        cls.cells = make_spec(scene, 10)["cells"][:2]
        cls.menus = []
        for cell in cls.cells:
            # Height and gap vary by small amounts; four other dimensions also
            # carry physical values, with a constant tilt coordinate here.
            options = []
            for index in range(24):
                values = [0, .4 + .0001 * index, .36 + .0005 * index,
                          .245 + .0002 * index, 26]
                options.append(dict(shot(cell, values), id=f"qualified-{index}"))
            cls.menus.append(options)

    def test_features_keep_small_physical_variation_and_constant_dimensions(self):
        obs, candidates = module.build_inputs(self.cells, self.menus)
        self.assertEqual(obs.shape, (2, 5))
        self.assertEqual(candidates.shape, (2, 24, 10))
        self.assertEqual(obs.dtype, np.float32)
        self.assertEqual(candidates.dtype, np.float32)
        self.assertTrue(np.isfinite(candidates).all())
        np.testing.assert_array_equal(candidates[:, :, 9], 0)
        np.testing.assert_array_equal(candidates[:, 0, 6:9], -1)
        np.testing.assert_array_equal(candidates[:, -1, 6:9], 1)
        self.assertLess(float(np.ptp(candidates[0, :, 2])), .06)
        self.assertEqual(float(np.ptp(candidates[0, :, 7])), 2)
        expected = 2 * (np.array([self.menus[0][0][k] for k in ACTION_NAMES]) - LOW) / (HIGH - LOW) - 1
        np.testing.assert_allclose(candidates[0, 0, :5], expected, atol=1e-7)

    def test_ids_and_stored_outcomes_never_enter_policy_inputs(self):
        changed = copy.deepcopy(self.menus)
        for options in changed:
            for index, option in enumerate(options):
                option.update(id=f"renamed-{23-index}", reward=10**6*index,
                              route_success=True, captures=index, trial_id="unrelated")
        before, after = module.build_inputs(self.cells, self.menus), module.build_inputs(self.cells, changed)
        for a, b in zip(before, after):
            np.testing.assert_array_equal(a, b)

    def test_exact_action_decoding_preserves_ids_and_copies_original(self):
        selected = module.actions_from_indices(self.cells, self.menus, np.array([4, 23]))
        self.assertEqual(selected, [self.menus[0][4], self.menus[1][23]])
        selected[0]["gap_m"] = -1
        self.assertGreater(self.menus[0][4]["gap_m"], 0)

    def test_missing_duplicated_or_inconsistent_native_geometry_is_rejected(self):
        for menus in (self.menus[:1], [self.menus[0][:-1], self.menus[1]]):
            with self.assertRaises(ValueError):
                module.build_inputs(self.cells, menus)
        changed = copy.deepcopy(self.menus)
        changed[0][-1] = dict(changed[0][0], id="different-id-same-geometry")
        with self.assertRaisesRegex(ValueError, "distinct physical"):
            module.build_inputs(self.cells, changed)
        changed = copy.deepcopy(self.menus)
        changed[0][0]["board_x_m"] += .01
        with self.assertRaises(ValueError):
            module.build_inputs(self.cells, changed)
        changed = copy.deepcopy(self.menus)
        changed[0][0]["gap_m"] = math.nan
        with self.assertRaises(ValueError):
            module.build_inputs(self.cells, changed)

    def test_invalid_category_indices_cannot_be_coerced_to_valid_actions(self):
        for indices in ([0, 24], [-1, 0], [0., 1.], [True, False], [0], [[0, 1]]):
            with self.subTest(indices=indices), self.assertRaises(ValueError):
                module.actions_from_indices(self.cells, self.menus, indices)


@unittest.skipIf(module.torch is None, "PyTorch CPU environment is unavailable")
class CategoricalPPOTests(unittest.TestCase):
    def setUp(self):
        self.torch = module.torch
        self.torch.set_num_threads(1)
        self.torch.manual_seed(20261110)
        self.policy = module.MenuPolicy()
        self.obs = self.torch.zeros((48, 5))
        self.features = self.torch.zeros((48, 24, 10))
        self.features[:, :, 5] = self.torch.linspace(-1, 1, 24)
        self.generator = self.torch.Generator().manual_seed(7)
        self.config = dict(module.PPO_DEFAULTS)

    def test_fresh_policy_is_exactly_uniform_and_state_dict_is_versioned(self):
        dist = self.policy.distribution(self.obs, self.features)
        self.torch.testing.assert_close(dist.probs, self.torch.full((48, 24), 1 / 24))
        self.torch.testing.assert_close(dist.entropy(), self.torch.full((48,), math.log(24)))
        self.assertTrue(all(self.torch.count_nonzero(p).item() == 0 for p in self.policy.actor[-1].parameters()))
        self.assertEqual(module.ARCHITECTURE["actor_input_dim"], 15)
        self.assertEqual(self.policy.value(self.obs).shape, (48,))

    def test_candidate_order_is_equivariant_without_category_id_biases(self):
        with self.torch.no_grad():
            self.policy.actor[-1].weight.normal_()
        permutation = self.torch.randperm(24)
        original = self.policy.logits(self.obs, self.features)
        permuted = self.policy.logits(self.obs, self.features[:, permutation])
        self.torch.testing.assert_close(permuted, original[:, permutation])

    def test_seeded_sampling_logp_entropy_and_rng_isolation(self):
        first = module.sample(self.policy, self.obs, self.features, self.generator)
        duplicate = module.sample(self.policy, self.obs, self.features,
                                  self.torch.Generator().manual_seed(7))
        self.torch.testing.assert_close(first["indices"], duplicate["indices"])
        dist = self.policy.distribution(self.obs, self.features)
        self.torch.testing.assert_close(first["logp"], dist.log_prob(first["indices"]))
        self.torch.testing.assert_close(first["entropy"], dist.entropy())
        self.assertFalse(first["logp"].requires_grad)
        self.assertFalse(first["value"].requires_grad)
        state = self.generator.get_state().clone()
        module.sample(self.policy, self.obs, self.features, self.torch.Generator().manual_seed(91))
        self.torch.testing.assert_close(state, self.generator.get_state())

    def test_ppo_consumes_rejections_and_changes_actor_as_well_as_critic(self):
        drawn = module.sample(self.policy, self.obs, self.features, self.generator)
        rewards = self.features[self.torch.arange(48), drawn["indices"], 5].clone()
        rewards[::3] = -1  # Admission rejects remain ordinary attempted-slot data.
        optimizer = self.torch.optim.Adam(self.policy.parameters(), lr=.003)
        full, actor = module.weight_hash(self.policy), module.actor_weight_hash(self.policy)
        result = module.update(self.policy, optimizer, self.obs, self.features,
                               drawn["indices"], drawn["logp"], rewards, self.config)
        self.assertEqual(result["samples"], 48)
        self.assertEqual(result["before"], full)
        self.assertEqual(result["actor_before"], actor)
        self.assertNotEqual(result["after"], full)
        self.assertNotEqual(result["actor_after"], actor)
        self.assertEqual(len(result["epochs"]), 4)
        for epoch in result["epochs"]:
            self.assertEqual(epoch["samples"], 48)
            self.assertTrue(all(math.isfinite(value) for value in epoch.values()))
            self.assertGreaterEqual(epoch["categorical_entropy"], 0)
            self.assertLessEqual(epoch["categorical_entropy"], math.log(24) + 1e-5)

    def test_critic_only_change_cannot_be_mistaken_for_actor_learning(self):
        full, actor = module.weight_hash(self.policy), module.actor_weight_hash(self.policy)
        optimizer = self.torch.optim.SGD(self.policy.critic.parameters(), lr=.1)
        loss = (self.policy.value(self.obs) - 1).square().mean()
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        self.assertNotEqual(module.weight_hash(self.policy), full)
        self.assertEqual(module.actor_weight_hash(self.policy), actor)

    def test_nonfinite_rewards_or_invalid_buffers_never_update_weights(self):
        optimizer = self.torch.optim.Adam(self.policy.parameters(), lr=.003)
        drawn = module.sample(self.policy, self.obs, self.features, self.generator)
        before = module.weight_hash(self.policy)
        for bad in (self.torch.full((48,), float("nan")), self.torch.zeros(47)):
            with self.assertRaises(ValueError):
                module.update(self.policy, optimizer, self.obs, self.features,
                              drawn["indices"], drawn["logp"], bad, self.config)
        self.assertEqual(module.weight_hash(self.policy), before)

    def test_synthetic_bandit_improves_without_any_native_success_claim(self):
        optimizer = self.torch.optim.Adam(self.policy.parameters(), lr=.003)
        initial = float((self.policy.distribution(self.obs, self.features).probs[0]
                         * self.features[0, :, 5]).sum())
        for _ in range(48):
            drawn = module.sample(self.policy, self.obs, self.features, self.generator)
            rewards = self.features[self.torch.arange(48), drawn["indices"], 5]
            module.update(self.policy, optimizer, self.obs, self.features,
                          drawn["indices"], drawn["logp"], rewards, self.config)
        final = float((self.policy.distribution(self.obs, self.features).probs[0]
                       * self.features[0, :, 5]).sum())
        self.assertGreater(final, initial + .65)

    def test_checkpoint_round_trip_and_hash_bindings(self):
        optimizer = self.torch.optim.Adam(self.policy.parameters(), lr=.003)
        bindings = dict(study_fingerprint="a"*64, geometry_fingerprint="b"*64, menu_sha256="c"*64)
        payload = module.checkpoint_payload(self.policy, optimizer, **bindings,
                                            updates=0, training_batches=1, seed=20261110)
        self.assertEqual(payload["policy_version"], module.POLICY_VERSION)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"policy.pt"; self.torch.save(payload, path)
            loaded, saved = module.load_checkpoint(path, **bindings, expected_updates=0,
                                                    expected_training_batches=1)
            self.assertEqual(module.weight_hash(loaded), module.weight_hash(self.policy))
            self.assertEqual(module.actor_weight_hash(loaded), payload["actor_weight_hash"])
            self.assertEqual(saved["training_batches"], 1)
            for key in bindings:
                changed = dict(bindings, **{key: "d"*64})
                with self.assertRaises(ValueError):
                    module.load_checkpoint(path, **changed)
            for key, value in (("policy_version", "other-policy"),
                               ("architecture", {}), ("weight_hash", "0"*64),
                               ("actor_weight_hash", "0"*64), ("updates", 1)):
                corrupted = copy.deepcopy(payload); corrupted[key] = value
                self.torch.save(corrupted, path)
                with self.subTest(key=key), self.assertRaises(ValueError):
                    module.load_checkpoint(path, **bindings)

    def test_checkpoint_assessment_load_preserves_global_training_rng(self):
        bindings = dict(study_fingerprint="a"*64, geometry_fingerprint="b"*64, menu_sha256="c"*64)
        payload = module.checkpoint_payload(self.policy, None, **bindings,
                                            updates=0, training_batches=0, seed=20261110)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"policy.pt"; self.torch.save(payload, path)
            state = self.torch.random.get_rng_state().clone()
            module.load_checkpoint(path, **bindings)
            self.torch.testing.assert_close(state, self.torch.random.get_rng_state())

    def test_checkpoint_payload_is_frozen_against_later_policy_changes(self):
        bindings = dict(study_fingerprint="a"*64, geometry_fingerprint="b"*64, menu_sha256="c"*64)
        payload = module.checkpoint_payload(self.policy, None, **bindings,
                                            updates=0, training_batches=0, seed=20261110)
        frozen = payload["state_dict"]["actor.4.bias"].clone()
        with self.torch.no_grad():
            self.policy.actor[-1].bias.add_(1)
        self.torch.testing.assert_close(payload["state_dict"]["actor.4.bias"], frozen)
        self.assertNotEqual(module.actor_weight_hash(self.policy), payload["actor_weight_hash"])
        with self.assertRaises(ValueError):
            module.checkpoint_payload(self.policy, None, **bindings,
                                      updates=1, training_batches=1, seed=20261110)


if __name__ == "__main__":
    unittest.main()
