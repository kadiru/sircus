"""Synthetic learning-driver integration; never loads the real seed-22 model."""

import copy
import hashlib
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from candidate_continuation_contract import PROTOCOL_SHA256
from candidate_continuation_ledger import Ledger, SourceBindings, canonical_hash
from candidate_continuation_policy import freeze_support, ACTION_NAMES, LOW, HIGH, torch
from candidate_continuation_journal import read_events
from candidate_continuation_runner import ContinuationRunner, source_fingerprint
from continuous_policy import ContinuousPolicy, weight_hash


def fixture():
    from candidate_continuation_updates import LearningEngine
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(104)
        policy = ContinuousPolicy()
    supports, demonstrations, counts = [], [], {}
    for cell in range(35):
        context = f"train-{cell:03d}"
        number = (74 if cell == 0 else 73) if cell < 21 else 20
        rows = []
        parent = dict(zip(ACTION_NAMES, ((np.asarray(LOW)+np.asarray(HIGH))/2).tolist()))
        for index in range(number):
            latent = np.asarray([.01+(index/number)*.8, 0, 0, 0, 0])
            values = np.asarray(LOW)+(np.tanh(latent)+1)*(np.asarray(HIGH)-LOW)/2
            identifier = f"synthetic-{cell}-{index}"
            rows.append(dict(candidate_id=identifier, action=dict(zip(ACTION_NAMES, values.tolist()))))
            if cell < 21:
                demonstrations.append(dict(context_id=context, candidate_id=identifier, index=index,
                    success_key=identifier, clean=len(demonstrations) < 485))
        supports.append(freeze_support(context, rows, parent, [0.]*5))
        if cell < 21: counts[context] = number
    observations = torch.tensor([[0., 0., .74, 0., 1.]]*35)
    engine = LearningEngine(policy, observations, supports, demonstrations, counts)
    bindings = dict(parent_checkpoint_sha256=SourceBindings().parent_checkpoint_sha256,
        input_sha256="1"*64, support_sha256=canonical_hash([s.to_dict() for s in supports]),
        source_sha256=source_fingerprint(), learning_protocol_sha256=PROTOCOL_SHA256,
        demonstrations_sha256=canonical_hash(demonstrations),
        observations_sha256=canonical_hash(observations.tolist()), parent_weight_hash=weight_hash(policy))
    ledger = Ledger.new(tuple(s.context_id for s in supports),
        bindings=SourceBindings(train_support_sha256=bindings["support_sha256"]))
    return engine, ledger, bindings


@unittest.skipIf(torch is None, "configured PyTorch environment unavailable")
class RunnerTests(unittest.TestCase):
    def make(self, root, *, clock=lambda:0., free_bytes=lambda:100*1024**3):
        engine, ledger, bindings = fixture()
        runner = ContinuationRunner(Path(root)/"run", engine, ledger, bindings, synthetic=True,
                                    clock=clock, free_bytes=free_bytes)
        return runner

    def test_full_synthetic_schedule_checkpoint_and_independent_learning_audit(self):
        from candidate_continuation_audit import audit_events
        from candidate_continuation_checkpoint import load_continuation
        with tempfile.TemporaryDirectory() as temporary:
            runner = self.make(temporary)
            initial = weight_hash(runner.engine.policy)
            for _ in range(240): runner.imitation_step()
            self.assertNotEqual(weight_hash(runner.engine.policy), initial)
            runner.start_ppo()
            observed, captured = 0, 0
            while runner.ledger.next_batch() is not None:
                choices = runner.reserve_train_batch()
                event = read_events(runner.journal.directory)[-1]
                self.assertEqual(event["kind"], "reserve")
                self.assertEqual(event["payload"]["ledger"]["counters"]["train"]["unresolved"], len(choices))
                results = []
                for choice in choices:
                    capture = observed % 3 == 0
                    clean = capture and observed % 2 == 0
                    results.append(dict(request_id=choice["request_id"], released=True, decline_reason=None,
                        source_sha256=canonical_hash(choice), captured=capture, clean=clean,
                        **{"return":1.6 if clean else (1.5 if capture else -.8)}))
                    observed += 1; captured += capture
                # Persisting a prefix leaves all remaining slots charged.
                runner.record_results(results[:1])
                if len(results) > 1: runner.record_results(results[1:])
            result = runner.finish()
            self.assertEqual(result["counters"], dict(imitation_steps=240, imitation_presentations=30680,
                ppo_updates=10, ppo_steps=40, ppo_samples=1400))
            loaded, saved = load_continuation(runner.directory/"final.pt",
                expected_sha256=result["checkpoint_sha256"], expected_bindings=runner.bindings)
            self.assertEqual(weight_hash(loaded), weight_hash(runner.engine.policy))
            self.assertEqual(saved["inherited"]["requests"], 10000)
            events = read_events(runner.journal.directory)
            audit = audit_events(events, expected_bindings=runner.bindings,
                expected_head_sha256=result["journal_head_sha256"],
                train_context_ids=runner.ledger.train_context_ids, supports=runner.engine.supports,
                demonstrations=list(runner.engine.demonstrations))
            self.assertEqual(audit["status"], "passed")
            from candidate_continuation_verify import verify_run
            kwargs = dict(expected_result_sha256=hashlib.sha256((runner.directory/"result.json").read_bytes()).hexdigest(),
                expected_bindings=runner.bindings, train_context_ids=runner.ledger.train_context_ids,
                supports=runner.engine.supports, demonstrations=list(runner.engine.demonstrations))
            verified = verify_run(runner.directory, **kwargs)
            self.assertTrue(verified["checkpoint_bytes_verified"])
            self.assertFalse(verified["native_physics_or_capture_verified"])
            with self.assertRaisesRegex(ValueError, "External run-result"):
                verify_run(runner.directory, **dict(kwargs, expected_result_sha256="f"*64))
            final_path = runner.directory/"final.pt"
            original = final_path.read_bytes()
            final_path.write_bytes(original+b"synthetic tamper")
            with self.assertRaisesRegex(ValueError, "file SHA"):
                verify_run(runner.directory, **kwargs)
            final_path.write_bytes(original)  # Restore only the temporary fabricated model.
            head = runner.journal.head
            with self.assertRaises(ValueError): runner.reserve_train_batch()
            self.assertEqual(runner.journal.head, head)

    def test_new_run_bindings_and_real_model_identity_fail_before_directory(self):
        engine, ledger, bindings = fixture()
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)/"run"
            with self.assertRaisesRegex(ValueError, "exact seed-22"):
                ContinuationRunner(path, engine, ledger, bindings, synthetic=False)
            self.assertFalse(path.exists())
            for key in ("support_sha256", "source_sha256", "demonstrations_sha256", "observations_sha256", "parent_weight_hash"):
                changed = dict(bindings); changed[key] = "f"*64
                with self.assertRaises(ValueError): ContinuationRunner(path, engine, ledger, changed, synthetic=True)
                self.assertFalse(path.exists())

    def test_phase_wall_storage_external_mutation_and_no_resume_fail_closed(self):
        for mode in ("clock", "storage", "weights"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as temporary:
                now = [0.]; free = [100*1024**3]
                runner = self.make(temporary, clock=lambda:now[0], free_bytes=lambda:free[0])
                if mode == "clock": now[0] = 600.
                if mode == "storage": free[0] = 6*1024**3-1
                if mode == "weights":
                    with torch.no_grad(): next(runner.engine.policy.parameters()).add_(.01)
                with self.assertRaises(ValueError): runner.imitation_step()
                self.assertTrue(runner.failed)
                self.assertEqual(runner.engine.snapshot_counters()["imitation_steps"], 0)
                self.assertEqual(read_events(runner.journal.directory)[-1]["kind"], "failure")
                head = runner.journal.head
                with self.assertRaises(ValueError): runner.imitation_step()
                self.assertEqual(runner.journal.head, head)

    def test_early_ppo_and_journal_failure_do_not_start_execution(self):
        with tempfile.TemporaryDirectory() as temporary:
            runner = self.make(temporary)
            with self.assertRaises(ValueError): runner.start_ppo()
            self.assertTrue(runner.failed)
        with tempfile.TemporaryDirectory() as temporary:
            runner = self.make(temporary)
            with patch.object(runner.journal, "append", side_effect=OSError("synthetic write")):
                with self.assertRaises(OSError): runner.imitation_step()
            self.assertTrue(runner.failed)
            self.assertEqual(runner.ledger.summary()["train"]["requested"], 0)


if __name__ == "__main__":
    unittest.main()
