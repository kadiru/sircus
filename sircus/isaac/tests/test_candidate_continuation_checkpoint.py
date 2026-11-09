"""Synthetic checkpoint fixtures only; no real parent loading or weight updates."""

import copy
import hashlib
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

SOURCE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOURCE))
import candidate_continuation_checkpoint as checkpoint
from candidate_continuation_contract import PROTOCOL_SHA256
from candidate_continuation_ledger import PARENT_CHECKPOINT_SHA256
from continuous_policy import ContinuousPolicy, load_checkpoint, weight_hash, torch


@unittest.skipIf(torch is None, "configured PyTorch environment unavailable")
class ContinuationCheckpointTests(unittest.TestCase):
    def bindings(self):
        return dict(parent_checkpoint_sha256=PARENT_CHECKPOINT_SHA256,
            input_sha256="1"*64, support_sha256="2"*64, source_sha256="3"*64,
            learning_protocol_sha256=PROTOCOL_SHA256)

    def counters(self, stage="final"):
        return dict(imitation_steps=240, imitation_presentations=30680,
            ppo_updates=10 if stage == "final" else 0,
            ppo_steps=40 if stage == "final" else 0,
            ppo_samples=1400 if stage == "final" else 0)

    def optimizer(self, parameters, steps, lr):
        # Deliberately fabricated Adam moments, not a training run.
        parameters = list(parameters)
        state = torch.optim.Adam(parameters, lr=lr, foreach=False, fused=False).state_dict()
        if steps:
            state["state"] = {i:dict(step=torch.tensor(float(steps)),
                exp_avg=torch.zeros_like(p), exp_avg_sq=torch.zeros_like(p))
                for i,p in enumerate(parameters)}
        return state

    def fixture(self, stage="final"):
        policy = ContinuousPolicy()
        counters = self.counters(stage)
        rng = {name:torch.Generator(device="cpu").manual_seed(seed).get_state()
               for name,seed in (("shuffle", 21), ("sample", 22))}
        optimizers = dict(imitation=self.optimizer(policy.actor.parameters(), 240, 1e-4),
            ppo=None if stage == "imitation" else self.optimizer(policy.parameters(), counters["ppo_steps"], 3e-4))
        payload = checkpoint.checkpoint_payload(policy, stage=stage, counters=counters,
            bindings=self.bindings(), rng_states=rng, optimizer_states=optimizers,
            ledger_fingerprint="4"*64, journal_fingerprint="5"*64)
        return policy, payload

    def resign(self, payload):
        payload["optimizer_state_hash"] = checkpoint.state_hash(payload["optimizer_states"])
        payload["rng_state_hash"] = checkpoint.state_hash(payload["rng_states"])
        payload["fingerprint"] = checkpoint.state_hash({k:v for k,v in payload.items() if k != "fingerprint"})

    def load(self, path, payload, **kwargs):
        torch.save(payload, path)
        return checkpoint.load_continuation(path,
            expected_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            expected_bindings=kwargs.get("bindings", self.bindings()))

    def test_model_round_trip_keeps_exact_tensors_counters_and_global_rng(self):
        original, payload = self.fixture()
        before = torch.get_rng_state().clone()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"final.pt"
            sha = checkpoint.save_checkpoint_new(path, payload)
            loaded, saved = checkpoint.load_continuation(path, expected_sha256=sha,
                expected_bindings=self.bindings())
        self.assertTrue(torch.equal(before, torch.get_rng_state()))
        self.assertEqual(weight_hash(loaded), weight_hash(original))
        self.assertEqual(saved["counters"], self.counters())
        self.assertEqual(saved["inherited"], dict(seed=20261122, requests=10000, optimizer_updates=73))
        self.assertTrue(all(p.grad is None for p in loaded.parameters()))
        self.assertFalse(saved["execution_authorized"])
        self.assertFalse(saved["release_authorized"])

    def test_payload_is_deep_detached_snapshot_without_caller_aliases(self):
        policy, payload = self.fixture()
        before = checkpoint.state_hash(payload)
        with torch.no_grad():
            policy.actor[-1].bias.add_(.1)
        self.assertEqual(checkpoint.state_hash(payload), before)
        restored = copy.deepcopy(payload)
        payload["state_dict"]["actor.0.weight"].zero_()
        payload["optimizer_states"]["imitation"]["state"][0]["exp_avg"].add_(2)
        self.assertNotEqual(checkpoint.state_hash(payload), checkpoint.state_hash(restored))

    def test_different_stages_separate_optimizer_state_and_new_counters(self):
        for stage in ("imitation", "ppo", "final"):
            with self.subTest(stage=stage):
                _, payload = self.fixture(stage)
                with tempfile.TemporaryDirectory() as directory:
                    _, saved = self.load(Path(directory)/"stage.pt", payload)
                self.assertEqual(saved["stage"], stage)
                self.assertEqual(saved["optimizer_states"]["ppo"] is None, stage == "imitation")

    def test_counter_schedule_and_caps_fail_closed(self):
        _, good = self.fixture()
        cases = [dict(imitation_presentations=30679), dict(imitation_steps=241),
                 dict(ppo_samples=1399), dict(ppo_samples=1751), dict(ppo_updates=11),
                 dict(ppo_steps=39), dict(ppo_steps=True)]
        for changes in cases:
            bad = copy.deepcopy(good); bad["counters"].update(changes); self.resign(bad)
            with self.subTest(changes=changes), tempfile.TemporaryDirectory() as directory:
                with self.assertRaises(ValueError):
                    self.load(Path(directory)/"bad.pt", bad)
        # Nonterminal imitation also follows exact 128-sized presentations.
        _, partial = self.fixture("imitation")
        partial["counters"].update(imitation_steps=1, imitation_presentations=127)
        self.resign(partial)
        with self.assertRaisesRegex(ValueError, "presentations"):
            checkpoint._validate(partial, self.bindings())

    def test_schema_law_parent_protocol_architecture_and_permission_are_strict(self):
        _, good = self.fixture()
        changes = [("schema", "sircus-continuous-shot-ppo-v1"), ("probability_law", "Gaussian"),
            ("seed", 20261122), ("stage", "test"), ("execution_authorized", True),
            ("release_authorized", True), ("learning_protocol_sha256", "6"*64),
            ("inherited", dict(seed=20261122, requests=0, optimizer_updates=0)),
            ("architecture", {})]
        for key,value in changes:
            bad = copy.deepcopy(good); bad[key] = value; self.resign(bad)
            with self.subTest(key=key), self.assertRaises(ValueError):
                checkpoint._validate(bad, self.bindings())
        for key in ("parent_checkpoint_sha256", "learning_protocol_sha256"):
            bad = copy.deepcopy(good); bad["bindings"][key] = "6"*64; self.resign(bad)
            with self.subTest(binding=key), self.assertRaises(ValueError):
                checkpoint._validate(bad, bad["bindings"])

    def test_external_file_sha_and_expected_bindings_are_required(self):
        _, payload = self.fixture()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"model.pt"; torch.save(payload, path)
            with patch.object(torch, "load", side_effect=AssertionError("must not load unbound bytes")):
                with self.assertRaisesRegex(ValueError, "file SHA"):
                    checkpoint.load_continuation(path, expected_sha256="0"*64,
                        expected_bindings=self.bindings())
            changed = self.bindings(); changed["input_sha256"] = "8"*64
            with self.assertRaisesRegex(ValueError, "bindings"):
                self.load(path, payload, bindings=changed)

    def test_corrupt_model_hash_shape_dtype_or_nonfinite_fails(self):
        _, good = self.fixture()
        for change in ("hash", "shape", "dtype", "nonfinite", "logstd"):
            bad = copy.deepcopy(good)
            if change == "hash": bad["weight_hash"] = "0"*64
            if change == "shape": bad["state_dict"]["actor.0.weight"] = torch.zeros((2, 2))
            if change == "dtype": bad["state_dict"]["actor.0.weight"] = bad["state_dict"]["actor.0.weight"].double()
            if change == "nonfinite": bad["state_dict"]["logstd"][0] = float("nan")
            if change == "logstd": bad["state_dict"]["logstd"][0] = 1.1
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.resign(bad); checkpoint._validate(bad, self.bindings())

    def test_adam_phase_moments_settings_and_counts_fail_closed_even_when_resigned(self):
        _, good = self.fixture()
        for change in ("step", "shape", "lr", "negative", "missing", "phase"):
            bad = copy.deepcopy(good)
            state = bad["optimizer_states"]["ppo"]
            if change == "step": state["state"][0]["step"] = torch.tensor(39.)
            if change == "shape": state["state"][0]["exp_avg"] = torch.zeros((2,))
            if change == "lr": state["param_groups"][0]["lr"] = .01
            if change == "negative": state["state"][0]["exp_avg_sq"].fill_(-1)
            if change == "missing": state["state"].pop(1)
            if change == "phase": bad["optimizer_states"]["ppo"] = None
            self.resign(bad)
            with self.subTest(change=change), self.assertRaises(ValueError):
                checkpoint._validate(bad, self.bindings())

    def test_rng_and_payload_fingerprint_corruption_fail(self):
        _, good = self.fixture()
        bad = copy.deepcopy(good); bad["rng_states"]["sample"] = torch.zeros(3, dtype=torch.uint8)
        self.resign(bad)
        with self.assertRaisesRegex(ValueError, "RNG state"):
            checkpoint._validate(bad, self.bindings())
        bad = copy.deepcopy(good); bad["rng_states"]["sample"] = bad["rng_states"]["sample"].float()
        self.resign(bad)
        with self.assertRaisesRegex(ValueError, "generator state"):
            checkpoint._validate(bad, self.bindings())
        bad = copy.deepcopy(good); bad["journal_fingerprint"] = "9"*64
        with self.assertRaisesRegex(ValueError, "fingerprint"):
            checkpoint._validate(bad, self.bindings())

    def test_new_file_publish_refuses_replacement_and_cleans_owned_temporary(self):
        _, payload = self.fixture()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"final.pt"
            first = checkpoint.save_checkpoint_new(path, payload)
            with self.assertRaises(FileExistsError):
                checkpoint.save_checkpoint_new(path, payload)
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), first)
            self.assertEqual(list(Path(directory).iterdir()), [path])
            with patch.object(torch, "save", side_effect=RuntimeError("synthetic serialization failure")):
                with self.assertRaises(RuntimeError):
                    checkpoint.save_checkpoint_new(Path(directory)/"failed.pt", payload)
            self.assertEqual(list(Path(directory).iterdir()), [path])

    def test_original_continuous_loader_rejects_new_checkpoint(self):
        _, payload = self.fixture()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"new.pt"; torch.save(payload, path)
            with self.assertRaisesRegex(ValueError, "architecture mismatch"):
                load_checkpoint(path, study_fingerprint="1"*64,
                    geometry_fingerprint="2"*64, domain_fingerprint="3"*64)

    def test_type_aware_state_hash_binds_mapping_keys_tensor_bytes_and_dtype(self):
        a = {1:torch.tensor([1., 2.]), "1":(True, None)}
        self.assertEqual(checkpoint.state_hash(a), checkpoint.state_hash(copy.deepcopy(a)))
        self.assertNotEqual(checkpoint.state_hash({1:"v"}), checkpoint.state_hash({"1":"v"}))
        self.assertNotEqual(checkpoint.state_hash([1, 2]), checkpoint.state_hash((1, 2)))
        self.assertNotEqual(checkpoint.state_hash(torch.ones(2)), checkpoint.state_hash(torch.ones(2).double()))
        for invalid in (float("nan"), {object():1}, torch.tensor([float("inf")]), object()):
            with self.subTest(type=type(invalid).__name__), self.assertRaises(ValueError):
                checkpoint.state_hash(invalid)


if __name__ == "__main__":
    unittest.main()
