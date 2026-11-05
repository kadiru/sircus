"""Unit-only continuous policy checks; synthetic rewards are not native evidence."""

import copy
import math
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ISAAC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ISAAC))
import continuous_policy as module


class GeometryTests(unittest.TestCase):
    def test_observation_features_ignore_ids_rewards_and_training_history(self):
        cells = [dict(id="one", bucket_xy_m=[.65, -.45]),
                 dict(id="two", bucket_xy_m=[.85, -.35])]
        before = module.build_observations(cells)
        changed = copy.deepcopy(cells)
        for cell in changed:
            cell.update(id="unknown", reward=100, captures=9, menu=[1,2,3])
        np.testing.assert_array_equal(before, module.build_observations(changed))
        self.assertEqual(before.shape, (2, 5))
        self.assertEqual(before.dtype, np.float32)
        with self.assertRaises(ValueError):
            module.build_observations([])
        with self.assertRaises(ValueError):
            module.build_observations([dict(bucket_xy_m=[float("nan"), 0])])

    def test_declared_initial_geometry_matches_signed_broad_domain(self):
        low = [-math.pi, .28, .14, .08, math.radians(12)]
        high = [math.pi, 1.20, .60, .56, math.radians(52)]
        decoded = module.physical_actions([module.INITIAL_NORMALIZED], low, high)
        # Radius here is the zero-residual coordinate; actual initial radius is
        # state-conditioned in distribution(), tested with Torch below.
        np.testing.assert_allclose(decoded[0][[0,2,3,4]], [0, .35, .24, math.radians(29)],
                                   atol=1e-14, rtol=0)
        self.assertEqual(module.ARCHITECTURE["initial_logstd"], -.9)
        self.assertEqual(module.ARCHITECTURE["initial_radius_prior"]["prior_clip_m"], [.29,1.19])
        self.assertEqual(module.ACTION_NAMES[0], "bearing_offset_rad")
        self.assertEqual(module.ACTION_NAMES[2], "height_m")

    def test_affine_actions_keep_continuous_resolution_without_projection(self):
        low = np.array([-math.pi, .22, .25, .08, math.radians(10)])
        high = np.array([math.pi, .8, .65, .55, math.radians(55)])
        raw = np.array([[-1]*5, [0]*5, [1]*5, [.00001]*5])
        actions = module.physical_actions(raw, low, high)
        np.testing.assert_allclose(actions[0], low)
        np.testing.assert_allclose(actions[1], (low+high)/2)
        np.testing.assert_allclose(actions[2], high)
        self.assertTrue((actions[3] > actions[1]).all())
        for bad in ([[0]*4], [[2]*5], [[float("nan")]*5]):
            with self.assertRaises(ValueError):
                module.physical_actions(bad, low, high)
        with self.assertRaises(ValueError):
            module.physical_actions(raw, high, low)


@unittest.skipIf(module.torch is None, "PyTorch CPU environment is unavailable")
class ContinuousPPOTests(unittest.TestCase):
    def setUp(self):
        self.torch = module.torch
        self.torch.set_num_threads(1)
        self.torch.manual_seed(20261111)
        self.policy = module.ContinuousPolicy()
        self.obs = self.torch.zeros((128, 5))
        self.generator = self.torch.Generator().manual_seed(7)
        self.config = dict(module.PPO_DEFAULTS)
        self.bindings = dict(study_fingerprint="a"*64, geometry_fingerprint="b"*64,
                             domain_fingerprint="c"*64)

    def test_fresh_actor_has_only_declared_geometry_initialization(self):
        dist = self.policy.distribution(self.obs)
        expected = self.torch.tensor(module.INITIAL_NORMALIZED).atanh().expand(128,5).clone()
        expected[:,1] = math.atanh(2*(.29-.28)/(.92)-1)
        self.torch.testing.assert_close(dist.loc, expected)
        self.torch.testing.assert_close(dist.scale, self.torch.full((128, 5), math.exp(-.9)))
        self.assertEqual(module.ACTION_DIM, 5)
        self.assertEqual(self.policy.value(self.obs).shape, (128,))

    def test_radius_prior_centers_bounce_toward_reference_without_outcome_input(self):
        cells = [dict(bucket_xy_m=[.2,-.5]), dict(bucket_xy_m=[1.1,-.5]),
                 dict(bucket_xy_m=[.67,-.10]), dict(bucket_xy_m=[2.,-.1])]
        obs = self.torch.from_numpy(module.build_observations(cells))
        picked = module.sample(self.policy, obs, deterministic=True)
        low = [-math.pi, .28, .14, .08, math.radians(12)]
        high = [math.pi,1.20,.60,.56,math.radians(52)]
        decoded = module.physical_actions(picked["actions_normalized"],low,high)
        distance = np.array([math.dist(c["bucket_xy_m"], [.67,-.1]) for c in cells])
        np.testing.assert_allclose(decoded[:,1], np.clip(distance,.29,1.19), atol=1e-7)
        for i in (0,1):
            bucket = np.array(cells[i]["bucket_xy_m"])
            direction = (np.array([.67,-.1])-bucket)/distance[i]
            np.testing.assert_allclose(bucket+decoded[i,1]*direction,[.67,-.1],atol=1e-7)
        # Radius remains learned and continuous, rather than nearest-option
        # projection: an actor residual can move it away from the prior.
        with self.torch.no_grad():
            self.policy.actor[-1].bias[1].add_(.25)
        changed = module.sample(self.policy,obs,deterministic=True)
        self.assertTrue((changed["actions_normalized"][:,1] > picked["actions_normalized"][:,1]).all())

    def test_likelihood_matches_exact_tanh_change_of_variables(self):
        policy = self.policy.double()
        obs = self.obs.double()
        latent = self.torch.linspace(-2, 2, 128*5, dtype=self.torch.float64).reshape(128,5)
        dist = policy.distribution(obs)
        expected = (dist.log_prob(latent) - self.torch.log1p(-latent.tanh().square())).sum(-1)
        self.torch.testing.assert_close(policy.log_prob(obs, latent), expected, rtol=1e-10, atol=1e-10)

    def test_saved_latent_likelihood_remains_finite_at_float32_saturation(self):
        latent = self.torch.ones(128,5)*20
        latent[::2] *= -1
        self.assertTrue((latent.tanh().abs() == 1).all())
        actual = self.policy.log_prob(self.obs, latent)
        self.assertTrue(self.torch.isfinite(actual).all())
        expected = (self.policy.distribution(self.obs).log_prob(latent)
                    -2*(math.log(2)-latent.abs())).sum(-1)
        self.torch.testing.assert_close(actual, expected)

    def test_seeded_sampling_and_deterministic_inference_preserve_rng(self):
        first = module.sample(self.policy, self.obs, self.generator)
        other = module.sample(self.policy, self.obs, self.torch.Generator().manual_seed(7))
        for key in ("latent", "actions_normalized", "logp", "value"):
            self.torch.testing.assert_close(first[key], other[key])
        self.torch.testing.assert_close(first["logp"], self.policy.log_prob(self.obs, first["latent"]))
        self.assertTrue((first["actions_normalized"].abs() <= 1).all())
        self.assertFalse(first["logp"].requires_grad)
        global_state = self.torch.random.get_rng_state().clone()
        generator_state = self.generator.get_state().clone()
        direct = module.sample(self.policy, self.obs, self.generator, deterministic=True)
        self.assertTrue(direct["deterministic"])
        expected = self.torch.tensor(module.INITIAL_NORMALIZED).expand(128,5).clone()
        expected[:,1] = 2*(.29-.28)/(.92)-1
        self.torch.testing.assert_close(direct["actions_normalized"], expected)
        self.torch.testing.assert_close(global_state, self.torch.random.get_rng_state())
        self.torch.testing.assert_close(generator_state, self.generator.get_state())

    def test_ppo_keeps_invalid_attempts_and_updates_actor_and_exploration(self):
        drawn = module.sample(self.policy, self.obs, self.generator)
        reward = drawn["actions_normalized"][:,0].clone()
        reward[::3] = -1
        optimizer = self.torch.optim.Adam(self.policy.parameters(), lr=3e-4)
        before, actor = module.weight_hash(self.policy), module.actor_weight_hash(self.policy)
        result = module.update(self.policy, optimizer, self.obs, drawn["latent"],
                               drawn["logp"], reward, self.config)
        self.assertEqual(result["samples"], 128)
        self.assertEqual(result["before"], before)
        self.assertEqual(result["actor_before"], actor)
        self.assertNotEqual(result["after"], before)
        self.assertNotEqual(result["actor_after"], actor)
        self.assertEqual(len(result["epochs"]), 4)
        self.assertTrue(all(math.isfinite(v) for row in result["epochs"] for v in row.values()))

    def test_actor_hash_detects_exploration_change_and_ignores_critic_only_change(self):
        actor = module.actor_weight_hash(self.policy)
        with self.torch.no_grad():
            self.policy.critic[-1].bias.add_(1)
        self.assertEqual(module.actor_weight_hash(self.policy), actor)
        with self.torch.no_grad():
            self.policy.logstd.add_(.1)
        self.assertNotEqual(module.actor_weight_hash(self.policy), actor)

    def test_invalid_buffers_fail_before_updates(self):
        drawn = module.sample(self.policy, self.obs, self.generator)
        optimizer = self.torch.optim.Adam(self.policy.parameters(), lr=3e-4)
        before = module.weight_hash(self.policy)
        for reward in (self.torch.zeros(127), self.torch.full((128,), float("nan"))):
            with self.assertRaises(ValueError):
                module.update(self.policy, optimizer, self.obs, drawn["latent"],
                              drawn["logp"], reward, self.config)
        self.assertEqual(module.weight_hash(self.policy), before)

    def test_terminal_synthetic_continuous_objective_is_learnable(self):
        # Center the geometric radius prior for this unit-only quadratic task.
        self.obs[:,2] = .74
        optimizer = self.torch.optim.Adam(self.policy.parameters(), lr=.003)
        for _ in range(40):
            drawn = module.sample(self.policy, self.obs, self.generator)
            reward = -(drawn["actions_normalized"] - .5).square().sum(-1)
            module.update(self.policy, optimizer, self.obs, drawn["latent"], drawn["logp"], reward, self.config)
        learned = module.sample(self.policy, self.obs, deterministic=True)["actions_normalized"]
        self.assertLess(float((learned-.5).square().mean()), .025)

    def test_checkpoint_reload_is_exact_and_deterministic_without_rng_change(self):
        optimizer = self.torch.optim.Adam(self.policy.parameters(), lr=3e-4)
        payload = module.checkpoint_payload(self.policy, optimizer, **self.bindings,
                                            update_count=2, attempted_shots=140, seed=20261111)
        before = module.sample(self.policy, self.obs, deterministic=True)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"policy.pt"
            self.torch.save(payload, path)
            state = self.torch.random.get_rng_state().clone()
            loaded, saved = module.load_checkpoint(path, **self.bindings, expected_updates=2,
                                                   expected_attempted_shots=140)
            self.torch.testing.assert_close(state, self.torch.random.get_rng_state())
            self.assertEqual(module.weight_hash(loaded), payload["weight_hash"])
            self.assertEqual(module.actor_weight_hash(loaded), payload["actor_weight_hash"])
            after = module.sample(loaded, self.obs, deterministic=True)
            for key in ("latent", "actions_normalized", "logp", "value"):
                self.torch.testing.assert_close(before[key], after[key])
            self.assertEqual(saved["attempted_shots"], 140)
            for key in self.bindings:
                changed = dict(self.bindings, **{key:"d"*64})
                with self.assertRaises(ValueError):
                    module.load_checkpoint(path, **changed)
            for key, value in (("architecture", {}), ("weight_hash", "0"*64),
                               ("actor_weight_hash", "0"*64), ("update_count", 141)):
                changed = copy.deepcopy(payload); changed[key] = value
                self.torch.save(changed, path)
                with self.subTest(key=key), self.assertRaises(ValueError):
                    module.load_checkpoint(path, **self.bindings)

    def test_checkpoint_is_immutable_snapshot_and_rejects_invalid_counters(self):
        payload = module.checkpoint_payload(self.policy, None, **self.bindings,
                                            update_count=0, attempted_shots=0, seed=20261111)
        frozen = payload["state_dict"]["actor.4.bias"].clone()
        with self.torch.no_grad():
            self.policy.actor[-1].bias.add_(1)
        self.torch.testing.assert_close(payload["state_dict"]["actor.4.bias"], frozen)
        self.assertNotEqual(module.actor_weight_hash(self.policy), payload["actor_weight_hash"])
        with self.assertRaises(ValueError):
            module.checkpoint_payload(self.policy, None, **self.bindings,
                                      update_count=2, attempted_shots=1, seed=20261111)


if __name__ == "__main__":
    unittest.main()
