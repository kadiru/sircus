"""Optimizer tests on fabricated models/actions only; no original checkpoint."""

import copy
from dataclasses import replace
import json
import math
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np

SOURCE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOURCE))
from candidate_continuation_checkpoint import state_hash
from candidate_continuation_policy import ACTION_NAMES, LOW, HIGH, freeze_support
from candidate_continuation_updates import LearningEngine
from continuous_policy import ContinuousPolicy, weight_hash, torch


def fake_inputs():
    """Exactly sized but wholly invented data, never capture observations."""
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(123)
        policy = ContinuousPolicy()
    with torch.no_grad():
        policy.actor[-1].bias.zero_()
        policy.logstd.zero_()
    observations = torch.tensor([[i/100, 0, .74, 0, 1] for i in range(35)], dtype=torch.float32)
    supports, demonstrations, counts = [], [], {}
    clean = 0
    for context in range(35):
        name = f"train-{context:03d}"
        count = (74 if context == 0 else 73) if context < 21 else 20
        candidates = []
        for index in range(count):
            latent = np.asarray([(index+1)/100, (context+1)/200, -.2, .1, -.3])
            physical = np.asarray(LOW)+(np.tanh(latent)+1)*(np.asarray(HIGH)-LOW)/2
            action = dict(zip(ACTION_NAMES, physical.tolist()))
            candidates.append(dict(candidate_id=f"synthetic-{context}-{index}", action=action))
        # Parent action is deliberately separate, not another candidate.
        midpoint = dict(zip(ACTION_NAMES, ((np.asarray(LOW)+HIGH)/2).tolist()))
        supports.append(freeze_support(name, candidates, midpoint, [0]*5))
        if context < 21:
            counts[name] = count
            for index in range(count):
                demonstrations.append(dict(context_id=name, candidate_id=candidates[index]["candidate_id"],
                    index=index, success_key=f"synthetic-success-{context}-{index}", clean=clean < 485))
                clean += 1
    return policy, observations, supports, demonstrations, counts


@unittest.skipIf(torch is None, "configured PyTorch environment unavailable")
class LearningEngineTests(unittest.TestCase):
    def engine(self):
        return LearningEngine(*fake_inputs())

    def ppo_engine(self):
        # Isolate PPO checks from the 240-step loop exercised separately below.
        engine = self.engine()
        engine._counters.update(imitation_steps=240, imitation_presentations=30680)
        engine.start_ppo()
        return engine

    def test_constructor_validates_counts_contexts_and_exact_labels_without_global_rng(self):
        inputs = fake_inputs()
        before = torch.get_rng_state().clone()
        engine = LearningEngine(*inputs)
        self.assertTrue(torch.equal(before, torch.get_rng_state()))
        self.assertEqual(weight_hash(engine.policy), weight_hash(inputs[0]))
        self.assertEqual(engine.snapshot_counters(), dict(imitation_steps=0, imitation_presentations=0,
                                                        ppo_updates=0, ppo_steps=0, ppo_samples=0))
        snapshots = engine.state_dict()
        self.assertEqual(set(snapshots["rng_states"]), {"shuffle", "sample"})
        self.assertEqual(set(snapshots["optimizer_states"]), {"imitation", "ppo"})
        self.assertIsNone(snapshots["optimizer_states"]["ppo"])
        snapshots["rng_states"]["shuffle"].zero_()
        self.assertFalse(torch.equal(snapshots["rng_states"]["shuffle"], engine.shuffle_generator.get_state()))
        engine.observations.zero_()
        self.assertTrue(torch.equal(engine.observations, inputs[1]))
        bad = list(fake_inputs())
        bad[3][0]["index"] = 1
        with self.assertRaisesRegex(ValueError, "candidate/index"):
            LearningEngine(*bad)
        bad = list(fake_inputs()); bad[3][1] = copy.deepcopy(bad[3][0])
        with self.assertRaisesRegex(ValueError, "Repeated"):
            LearningEngine(*bad)
        bad = list(fake_inputs()); bad[4]["train-000"] -= 1
        with self.assertRaisesRegex(ValueError, "counts"):
            LearningEngine(*bad)
        bad = list(fake_inputs()); bad[1][0][0] = float("nan")
        with self.assertRaisesRegex(ValueError, "finite"):
            LearningEngine(*bad)

    def test_imitation_changes_actor_only_and_records_exact_schedule_and_snapshots(self):
        engine = self.engine()
        critic = state_hash(engine.policy.critic.state_dict())
        scales = state_hash(engine.policy.logstd)
        before = weight_hash(engine.policy)
        global_rng = torch.get_rng_state().clone()
        receipt = engine.imitation_step()
        json.dumps(receipt, allow_nan=False)
        self.assertNotEqual(before, weight_hash(engine.policy))
        self.assertEqual(critic, state_hash(engine.policy.critic.state_dict()))
        self.assertEqual(scales, state_hash(engine.policy.logstd))
        self.assertTrue(torch.equal(global_rng, torch.get_rng_state()))
        self.assertEqual((receipt["epoch"], receipt["minibatch"], len(receipt["demo_indices"])), (0, 0, 128))
        self.assertEqual(receipt["counters_after"]["imitation_presentations"], 128)
        self.assertEqual(receipt["counters_after"]["imitation_steps"], 1)
        self.assertTrue(receipt["frozen_critic_and_logstd"])
        self.assertNotEqual(receipt["optimizer_hash_before"], receipt["optimizer_hash_after"])
        self.assertGreater(receipt["metrics"]["demonstration_nll"], 0)
        self.assertTrue(all(p.grad is None for p in engine.policy.critic.parameters()))
        self.assertIsNone(engine.policy.logstd.grad)
        another = self.engine().imitation_step()
        self.assertEqual(receipt, another)

    def test_complete_imitation_visits_each_example_once_per_epoch_and_stops_at_240(self):
        engine = self.engine()
        with self.assertRaisesRegex(ValueError, "exact completed"):
            engine.start_ppo()
        for epoch in range(20):
            seen = []
            for minibatch in range(12):
                receipt = engine.imitation_step()
                self.assertEqual((receipt["epoch"], receipt["minibatch"]), (epoch, minibatch))
                self.assertEqual(len(receipt["demo_indices"]), 128 if minibatch < 11 else 126)
                seen.extend(receipt["demo_indices"])
            self.assertEqual(sorted(seen), list(range(1534)))
        self.assertEqual(engine.snapshot_counters()["imitation_presentations"], 30680)
        with self.assertRaisesRegex(ValueError, "complete"):
            engine.imitation_step()
        old_optimizer_hash = engine.optimizer_state_hash()
        engine.start_ppo()
        self.assertEqual(engine.stage, "ppo")
        self.assertFalse(engine.optimizer.state)
        self.assertNotEqual(old_optimizer_hash, engine.optimizer_state_hash())
        self.assertTrue(all(p.requires_grad for p in engine.policy.parameters()))
        with self.assertRaisesRegex(ValueError, "stage"):
            engine.imitation_step()

    def test_ppo_uses_four_epochs_old_targets_all_returns_and_clipped_gradients(self):
        engine = self.ppo_engine()
        contexts = [engine.context_ids[i % 35] for i in range(140)]
        old = engine.distribution(contexts)
        indices = old.deterministic()
        returns = [-1. if i % 2 else 1. for i in range(140)]
        before = weight_hash(engine.policy)
        receipt = engine.ppo_update(contexts, indices, returns, old)
        json.dumps(receipt, allow_nan=False)
        self.assertEqual(receipt["request_count"], 140)
        self.assertEqual(receipt["rollout"]["terminal_returns"], returns)
        self.assertEqual(len(receipt["epochs"]), 4)
        self.assertEqual([r["epoch"] for r in receipt["epochs"]], [0, 1, 2, 3])
        self.assertEqual(receipt["counters_after"]["ppo_steps"], 4)
        self.assertEqual(receipt["counters_after"]["ppo_updates"], 1)
        self.assertEqual(receipt["counters_after"]["ppo_samples"], 140)
        self.assertNotEqual(before, weight_hash(engine.policy))
        self.assertTrue(torch.isfinite(engine.policy.logstd).all())
        self.assertTrue((engine.policy.logstd >= -3).all() and (engine.policy.logstd <= 1).all())
        self.assertNotEqual(receipt["critic_hash_before"], receipt["critic_hash_after"])
        self.assertNotEqual(receipt["logstd_hash_before"], receipt["logstd_hash_after"])
        with self.assertRaisesRegex(ValueError, "Stale PPO"):
            engine.ppo_update(contexts, indices, returns, old)

    def test_partial_singleton_final_buffer_has_raw_advantage_and_cannot_be_reused(self):
        engine = self.ppo_engine()
        contexts = [engine.context_ids[0]]
        old, indices = engine.sample(contexts)
        rng = torch.get_rng_state().clone()
        with self.assertRaisesRegex(ValueError, "partial"):
            engine.ppo_update(contexts, indices, [1.], old)
        receipt = engine.ppo_update(contexts, indices, [1.], old, final=True)
        self.assertEqual(receipt["rollout"]["advantages"], [1.-receipt["rollout"]["old_values"][0]])
        self.assertEqual(receipt["counters_after"]["ppo_samples"], 1)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        with self.assertRaisesRegex(ValueError, "Final PPO buffer"):
            engine.sample(contexts)
        with self.assertRaisesRegex(ValueError, "cap-stopped"):
            engine.finish_ppo()

    def test_sampling_is_dedicated_exactly_one_draw_per_context_without_updates(self):
        first, second = self.ppo_engine(), self.ppo_engine()
        contexts = list(first.context_ids)
        before = torch.get_rng_state().clone()
        one, indices = first.sample(contexts)
        torch.rand(7)  # Deliberately disturb unrelated global RNG.
        global_after_external_draw = torch.get_rng_state().clone()
        two, other = second.sample(contexts)
        self.assertTrue(torch.equal(indices, other))
        self.assertTrue(torch.equal(global_after_external_draw, torch.get_rng_state()))
        self.assertFalse(torch.equal(before, torch.get_rng_state()))
        self.assertTrue(all(not logp.requires_grad for logp in one.log_probabilities))
        self.assertFalse(one.values.requires_grad)
        self.assertEqual(first.snapshot_counters()["ppo_steps"], 0)
        self.assertEqual(weight_hash(first.policy), weight_hash(second.policy))
        with self.assertRaisesRegex(ValueError, "known TRAIN"):
            first.distribution(["test-000"])
        with self.assertRaisesRegex(ValueError, "at most 35"):
            first.sample(contexts+contexts)

    def test_invalid_nonfinite_targets_and_external_weight_mutations_fail_closed(self):
        engine = self.ppo_engine()
        contexts = [engine.context_ids[0]]
        old, indices = engine.sample(contexts)
        before = weight_hash(engine.policy)
        with self.assertRaisesRegex(ValueError, "finite terminal"):
            engine.ppo_update(contexts, indices, [float("nan")], old, final=True)
        self.assertEqual(engine.stage, "halted")
        self.assertEqual(weight_hash(engine.policy), before)
        self.assertEqual(engine.snapshot_counters()["ppo_steps"], 0)
        with self.assertRaisesRegex(ValueError, "stage"):
            engine.sample(contexts)
        engine = self.engine()
        with torch.no_grad():
            engine.policy.actor[-1].bias.add_(.1)
        with self.assertRaisesRegex(ValueError, "outside"):
            engine.imitation_step()

    def test_request_and_update_caps_prevent_steps_and_final_freezes_parameters(self):
        engine = self.ppo_engine()
        # Counter boundary fixtures, not fabricated completed-native evidence.
        engine._counters.update(ppo_samples=1749, ppo_updates=12, ppo_steps=48)
        contexts = [engine.context_ids[0]]
        old, indices = engine.sample(contexts)
        receipt = engine.ppo_update(contexts, indices, [-1.], old, final=True)
        self.assertEqual(receipt["counters_after"]["ppo_samples"], 1750)
        self.assertEqual(receipt["counters_after"]["ppo_steps"], 52)
        engine.finish_ppo()
        self.assertEqual(engine.stage, "final")
        self.assertTrue(all(not p.requires_grad and p.grad is None for p in engine.policy.parameters()))
        with self.assertRaisesRegex(ValueError, "stage"):
            engine.sample(contexts)
        engine = self.ppo_engine()
        engine._counters.update(ppo_samples=1750, ppo_updates=13, ppo_steps=52)
        old, indices = engine.sample(contexts)
        before = weight_hash(engine.policy)
        with self.assertRaisesRegex(ValueError, "cap"):
            engine.ppo_update(contexts, indices, [-1.], old, final=True)
        self.assertEqual(weight_hash(engine.policy), before)


if __name__ == "__main__":
    unittest.main()
