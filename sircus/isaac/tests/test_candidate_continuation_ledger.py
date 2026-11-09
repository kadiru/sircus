"""Synthetic pure accounting only: no policy, optimizer, server or native work."""

import copy
from dataclasses import FrozenInstanceError, replace
import json
from pathlib import Path
import subprocess
import sys
import unittest

SOURCE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOURCE))
import candidate_continuation_ledger as accounting


TRAIN = tuple(f"train-{i:03d}" for i in range(35))
VALIDATION = tuple(f"new-validation-{i:03d}" for i in range(35))
TEST = tuple(f"new-test-{i:03d}" for i in range(70))
FINAL_SHA = "f" * 64


def new_ledger(held_out=True):
    return accounting.Ledger.new(TRAIN, VALIDATION if held_out else None, TEST if held_out else None)


def terminal(batch, released=True, reason="settling_declined"):
    return tuple(accounting.Outcome(r.request_id, released, None if released else reason) for r in batch.requests)


def finish_train(ledger, released=True):
    while (batch := ledger.next_batch()) is not None:
        ledger = ledger.reserve_batch(batch).record_batch(terminal(batch, released))
    return ledger


def rehash(receipt):
    unsigned = dict(receipt)
    unsigned.pop("fingerprint", None)
    receipt["fingerprint"] = accounting.canonical_hash(unsigned)
    return receipt


class ContinuationLedgerTests(unittest.TestCase):
    def test_fixed_versioned_caps_are_phase_separate_and_not_caller_config(self):
        ledger = new_ledger(False)
        rules = accounting.protocol()
        self.assertEqual((rules["train"]["request_limit"], rules["train"]["release_limit"]), (1750, 1400))
        self.assertEqual((rules["assessment"]["request_limit"], rules["assessment"]["release_limit"]), (420, 420))
        self.assertEqual((rules["maximum_new_task_requests"], rules["maximum_new_task_releases"]), (2170, 1820))
        self.assertEqual((rules["inherited_parent_requests"], rules["prior_comparison_requests"]), (10000, 70))
        self.assertEqual(ledger.summary()["total_new_task_requests"], 0)
        self.assertEqual(accounting.canonical_hash(rules), accounting.PROTOCOL_SHA256)
        rules["train"]["request_limit"] = 99999
        self.assertEqual(accounting.protocol()["train"]["request_limit"], 1750)
        with self.assertRaises(TypeError):
            accounting.Ledger.new(TRAIN, request_limit=99999)
        for changes in (dict(schema="old-continuous-policy"), dict(protocol_sha256="0" * 64)):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                replace(ledger, **changes)

    def test_reservation_charges_requests_before_terminal_outcomes_and_never_refunds(self):
        empty = new_ledger()
        batch = empty.next_batch()
        ledger = empty.reserve_batch(batch)
        self.assertEqual(empty.summary()["train"]["requested"], 0)
        self.assertEqual((ledger.summary()["train"]["requested"], ledger.summary()["train"]["unresolved"]), (35, 35))
        self.assertEqual(ledger.summary()["train"]["released"], 0)
        for action in (ledger.next_batch, lambda: ledger.reserve_batch(batch), lambda: ledger.record_batch(())):
            with self.assertRaises(ValueError):
                action()
        restored = accounting.Ledger.from_dict(json.loads(json.dumps(ledger.to_dict())))
        self.assertEqual(restored, ledger)
        self.assertEqual(restored.summary()["train"]["requested"], 35)
        with self.assertRaises(ValueError):
            restored.next_batch()

    def test_each_decline_costs_a_request_zero_releases_and_advances_context(self):
        empty = new_ledger()
        batch = empty.next_batch()
        ledger = empty.reserve_batch(batch)
        ledger = ledger.record_outcomes((accounting.Outcome(batch.requests[0].request_id, False, "preflight_declined"),))
        self.assertEqual((ledger.summary()["train"]["requested"], ledger.summary()["train"]["declined"]), (35, 1))
        ledger = ledger.record_batch(tuple(accounting.Outcome(r.request_id, False, "no_qualified_support")
                                          for r in batch.requests[1:]))
        self.assertEqual((ledger.summary()["train"]["released"], ledger.summary()["train"]["declined"]), (0, 35))
        self.assertEqual(set(ledger.summary()["train"]["context_exposure"]), set(TRAIN))
        next_batch = ledger.next_batch()
        self.assertEqual(next_batch.requests[0].request_index, 35)
        self.assertTrue(set(r.request_id for r in batch.requests).isdisjoint(r.request_id for r in next_batch.requests))
        self.assertEqual(set(r.context_id for r in next_batch.requests), set(TRAIN))

    def test_deterministic_rounds_and_fingerprint_bind_ordered_contexts_rng_and_sources(self):
        first, second = new_ledger(), new_ledger()
        self.assertEqual(first.next_batch(), second.next_batch())
        self.assertNotEqual([r.context_id for r in first.next_batch().requests], list(TRAIN))
        reordered = accounting.Ledger.new(tuple(reversed(TRAIN)), VALIDATION, TEST)
        self.assertNotEqual(first.schedule_sha256, reordered.schedule_sha256)
        self.assertEqual(accounting.protocol()["train"]["schedule_seed"], 20261125)
        self.assertNotEqual(first.schedule_sha256, new_ledger(False).schedule_sha256)
        reserved = first.reserve_batch(first.next_batch())
        bindings = replace(first.bindings, train_bank_sha256="a" * 64)
        with self.assertRaisesRegex(ValueError, "source binding"):
            replace(reserved, bindings=bindings)
        with self.assertRaisesRegex(ValueError, "source binding"):
            replace(reserved, train_context_ids=tuple(reversed(TRAIN)))

    def test_out_of_order_duplicate_unknown_and_too_many_outcomes_fail_without_mutation(self):
        original = new_ledger()
        batch = original.next_batch()
        ledger = original.reserve_batch(batch)
        good = terminal(batch)
        invalid = (good[::-1], (good[0], good[0]) + good[2:],
            (replace(good[0], request_id="unknown"),) + good[1:], good + (good[-1],))
        for outcomes in invalid:
            with self.subTest(outcomes=outcomes), self.assertRaises(ValueError):
                ledger.record_batch(outcomes)
        ledger = ledger.record_batch(good)
        for action in (lambda: ledger.record_batch(good), lambda: ledger.reserve_batch(batch)):
            with self.assertRaises(ValueError):
                action()
        self.assertEqual(ledger.summary()["train"]["released"], 35)
        duplicate = replace(batch, batch_index=1)
        with self.assertRaisesRegex(ValueError, "Request IDs/order"):
            replace(ledger, batches=ledger.batches + (duplicate,))

    def test_release_cap_controls_partial_last_batch_and_inactive_padding_does_not_count(self):
        ledger = new_ledger()
        batch = ledger.next_batch()
        outcomes = (accounting.Outcome(batch.requests[0].request_id, False, "admission_declined"),) + terminal(batch)[1:]
        ledger = ledger.reserve_batch(batch).record_batch(outcomes)
        for _ in range(39):
            batch = ledger.next_batch()
            self.assertEqual(len(batch.requests), 35)
            ledger = ledger.reserve_batch(batch).record_batch(terminal(batch))
        self.assertEqual((ledger.summary()["train"]["requested"], ledger.summary()["train"]["released"]), (1400, 1399))
        final = ledger.next_batch()
        self.assertEqual(len(final.requests), 1)
        self.assertEqual(final.native_slots[1:], (None,) * 34)
        ledger = ledger.reserve_batch(final)
        with self.assertRaisesRegex(ValueError, "Inactive padding"):
            ledger.record_batch(terminal(final) + (terminal(final)[0],) + (None,) * 33)
        ledger = ledger.record_batch(terminal(final) + (None,) * 34)
        self.assertEqual((ledger.summary()["train"]["requested"], ledger.summary()["train"]["released"]), (1401, 1400))
        self.assertTrue(ledger.summary()["train"]["release_limit_reached"])
        self.assertIsNone(ledger.next_batch())
        with self.assertRaises(ValueError):
            ledger.reserve_batch(final)

    def test_batch_cannot_hide_release_overshoot_even_if_most_outcomes_decline(self):
        ledger = new_ledger()
        batch = ledger.next_batch()
        ledger = ledger.reserve_batch(batch).record_batch(
            (accounting.Outcome(batch.requests[0].request_id, False, "guard_declined"),) + terminal(batch)[1:])
        for _ in range(39):
            batch = ledger.next_batch()
            ledger = ledger.reserve_batch(batch).record_batch(terminal(batch))
        final = ledger.next_batch()
        extra = replace(final.requests[0], request_id="train-001401", request_index=1401)
        enlarged = replace(final, requests=final.requests + (extra,))
        with self.assertRaisesRegex(ValueError, "stale|out of order"):
            ledger.reserve_batch(enlarged)
        with self.assertRaisesRegex(ValueError, "overshoot"):
            replace(ledger, batches=ledger.batches + (enlarged,))

    def test_request_cap_stops_all_declines_and_reports_release_shortfall(self):
        ledger = finish_train(new_ledger(), released=False)
        summary = ledger.summary()["train"]
        self.assertEqual((summary["requested"], summary["released"], summary["declined"]), (1750, 0, 1750))
        self.assertEqual(summary["request_cap_release_target_shortfall"], 1400)
        self.assertEqual(set(summary["context_exposure"].values()), {50})
        self.assertTrue(summary["request_limit_reached"])
        self.assertFalse(summary["release_limit_reached"])
        self.assertIsNone(ledger.next_batch())

    def test_request_allowance_can_make_last_batch_smaller_than_release_allowance(self):
        ledger = new_ledger()
        batch = ledger.next_batch()
        outcomes = terminal(batch, released=False)[:-1] + terminal(batch)[-1:]
        ledger = ledger.reserve_batch(batch).record_batch(outcomes)
        for _ in range(39):
            batch = ledger.next_batch()
            ledger = ledger.reserve_batch(batch).record_batch(terminal(batch))
        self.assertEqual((ledger.summary()["train"]["requested"], ledger.summary()["train"]["released"]), (1400, 1366))
        for _ in range(10):
            batch = ledger.next_batch()
            self.assertEqual(len(batch.requests), 34)
            ledger = ledger.reserve_batch(batch).record_batch(terminal(batch, released=False))
        self.assertEqual((ledger.summary()["train"]["remaining_requests"], ledger.summary()["train"]["remaining_releases"]), (10, 34))
        batch = ledger.next_batch()
        self.assertEqual(len(batch.requests), 10)
        ledger = ledger.reserve_batch(batch).record_batch(terminal(batch, released=False) + (None,) * 25)
        self.assertEqual(ledger.summary()["train"]["requested"], 1750)
        self.assertEqual(ledger.summary()["train"]["request_cap_release_target_shortfall"], 34)
        self.assertIsNone(ledger.next_batch())

    def test_assessment_requires_train_stop_fresh_ids_and_one_frozen_artifact(self):
        with self.assertRaisesRegex(ValueError, "stopping rule"):
            new_ledger().begin_assessment(FINAL_SHA)
        with self.assertRaisesRegex(ValueError, "preparation is missing"):
            finish_train(new_ledger(False)).begin_assessment(FINAL_SHA)
        ledger = finish_train(new_ledger())
        for value in (None, True, 1, "invalid"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                ledger.begin_assessment(value)
        assessed = ledger.begin_assessment(FINAL_SHA)
        self.assertEqual(assessed.final_checkpoint_sha256, FINAL_SHA)
        self.assertEqual(assessed.summary()["assessment"]["requested"], 0)
        with self.assertRaisesRegex(ValueError, "checkpoint cannot change"):
            assessed.begin_assessment("a" * 64)
        # Rehashing a handcrafted phase transition must not hide unresolved
        # reservations, even when the request cap has already been consumed.
        request_capped = finish_train(new_ledger(), released=False)
        unresolved = replace(request_capped.batches[-1], outcomes=())
        request_capped = replace(request_capped, batches=request_capped.batches[:-1] + (unresolved,))
        self.assertEqual(request_capped.summary()["train"]["requested"], 1750)
        with self.assertRaisesRegex(ValueError, "terminal outcomes"):
            request_capped.begin_assessment(FINAL_SHA)
        with self.assertRaisesRegex(ValueError, "terminal outcomes"):
            replace(request_capped, phase="assessment", final_checkpoint_sha256=FINAL_SHA)

    def test_assessment_exact_420_split_method_rotation_and_no_replacements(self):
        ledger = finish_train(new_ledger(), released=False).begin_assessment(FINAL_SHA)
        requested = []
        for split, offset, method in accounting.ASSESSMENT_COHORTS:
            batch = ledger.next_batch()
            self.assertEqual(len(batch.requests), 35)
            self.assertEqual({r.split for r in batch.requests}, {split})
            self.assertEqual({r.method for r in batch.requests}, {method})
            expected_contexts = {"train": TRAIN, "validation": VALIDATION, "test": TEST}[split][offset:offset + 35]
            self.assertEqual(tuple(r.context_id for r in batch.requests), expected_contexts)
            self.assertEqual({r.final_checkpoint_sha256 for r in batch.requests}, {FINAL_SHA})
            requested.extend(batch.requests)
            # Empty-bank/guard declines remain in all requested denominators.
            ledger = ledger.reserve_batch(batch).record_batch(terminal(batch, released=False, reason="empty_bank"))
        self.assertEqual(len({r.request_id for r in requested}), 420)
        for method in accounting.METHODS:
            self.assertEqual({split: sum(r.method == method and r.split == split for r in requested)
                              for split in ("train", "validation", "test")}, dict(train=35, validation=35, test=70))
        self.assertEqual((ledger.summary()["total_new_task_requests"], ledger.summary()["total_new_task_releases"]), (2170, 0))
        self.assertTrue(ledger.summary()["assessment_complete"])
        self.assertIsNone(ledger.next_batch())
        with self.assertRaises(ValueError):
            ledger.reserve_batch(ledger.batches[-1])
        self.assertEqual(accounting.Ledger.from_dict(ledger.to_dict()), ledger)

    def test_assessment_release_maximum_is_separate_from_train(self):
        ledger = finish_train(new_ledger()).begin_assessment(FINAL_SHA)
        while (batch := ledger.next_batch()) is not None:
            ledger = ledger.reserve_batch(batch).record_batch(terminal(batch))
        self.assertEqual((ledger.summary()["train"]["released"], ledger.summary()["assessment"]["released"]), (1400, 420))
        self.assertEqual(ledger.summary()["total_new_task_releases"], 1820)

    def test_partial_test_preserves_reserved_denominator_and_stops_without_retry(self):
        ledger = finish_train(new_ledger()).begin_assessment(FINAL_SHA)
        for _ in range(6):
            batch = ledger.next_batch()
            ledger = ledger.reserve_batch(batch).record_batch(terminal(batch))
        test_batch = ledger.next_batch()
        self.assertEqual({r.split for r in test_batch.requests}, {"test"})
        ledger = ledger.reserve_batch(test_batch).record_outcomes(terminal(test_batch)[:2]).halt("source_integrity_failure")
        summary = ledger.summary()["assessment"]
        self.assertEqual((summary["requested"], summary["terminal"], summary["unresolved"]), (245, 212, 33))
        self.assertFalse(ledger.summary()["assessment_complete"])
        restored = accounting.Ledger.from_dict(ledger.to_dict())
        self.assertEqual(restored, ledger)
        for action in (restored.next_batch, lambda: restored.record_outcomes(terminal(test_batch)[2:]),
                       lambda: restored.begin_assessment(FINAL_SHA), lambda: restored.halt("another_failure")):
            with self.assertRaises(ValueError):
                action()

    def test_strict_bool_int_types_and_decline_semantics(self):
        request = new_ledger().next_batch().requests[0]
        for value in (True, False, 0.0, "0", -1):
            with self.subTest(value=value), self.assertRaises(ValueError):
                replace(request, request_index=value)
            with self.subTest(value=value), self.assertRaises(ValueError):
                accounting.Batch(value, "train", (request,))
        for value in (0, 1, "true", None):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "Boolean"):
                accounting.Outcome(request.request_id, value)
        for released, reason in ((False, None), (False, ""), (False, 1), (True, "declined")):
            with self.subTest(released=released, reason=reason), self.assertRaises(ValueError):
                accounting.Outcome(request.request_id, released, reason)

    def test_immutability_context_validation_and_unbound_sources_are_explicit(self):
        ledger = new_ledger()
        with self.assertRaises(FrozenInstanceError):
            ledger.phase = "assessment"
        with self.assertRaises(FrozenInstanceError):
            ledger.bindings.train_bank_sha256 = "a" * 64
        with self.assertRaises(FrozenInstanceError):
            ledger.next_batch().requests[0].request_index = 100
        for train, validation, test in ((TRAIN[:-1], VALIDATION, TEST),
            ((TRAIN[0],) * 35, VALIDATION, TEST), (TRAIN, TRAIN, TEST),
            (TRAIN, VALIDATION, TEST[:-1]), (TRAIN[:-1] + (True,), VALIDATION, TEST)):
            with self.subTest(train=train, validation=validation, test=test), self.assertRaises(ValueError):
                accounting.Ledger.new(train, validation, test)
        with self.assertRaisesRegex(ValueError, "context ID lists"):
            accounting.Ledger.new("abcdefghijklmnopqrstuvwxyz012345678")
        receipt = ledger.to_dict()
        self.assertFalse(receipt["source_bindings_complete"])
        self.assertFalse(receipt["execution_authorized"])
        self.assertFalse(receipt["release_authorized"])
        self.assertIsNone(receipt["bindings"]["producer_source_sha256"])
        receipt["bindings"]["train_bank_sha256"] = "f" * 64
        self.assertIsNone(ledger.bindings.train_bank_sha256)
        for value in (True, 1, "z" * 64, "a" * 63):
            with self.subTest(value=value), self.assertRaises(ValueError):
                accounting.SourceBindings(train_bank_sha256=value)
        with self.assertRaisesRegex(ValueError, "parent checkpoint"):
            accounting.SourceBindings(parent_checkpoint_sha256="a" * 64)

    def test_receipt_replay_rejects_tampered_schema_counters_ids_and_type_coercion(self):
        batch = new_ledger().next_batch()
        ledger = new_ledger().reserve_batch(batch).record_batch(terminal(batch))
        original = ledger.to_dict()
        unsigned = copy.deepcopy(original)
        unsigned["phase"] = "assessment"
        with self.assertRaisesRegex(ValueError, "fingerprint changed"):
            accounting.Ledger.from_dict(unsigned)
        changes = (
            lambda x: x.update(schema="old-schema"),
            lambda x: x.update(protocol_sha256="a" * 64),
            lambda x: x.update(release_authorized=True),
            lambda x: x.update(source_bindings_complete=True),
            lambda x: x.update(extra_budget=1000),
            lambda x: x["counters"]["train"].update(requested=0),
            lambda x: x["batches"][0]["requests"][0].update(request_index=False),
            lambda x: x["batches"][0]["outcomes"][0].update(released=1),
            lambda x: x["batches"][0]["requests"][1].update(request_id=x["batches"][0]["requests"][0]["request_id"]),
            lambda x: x["batches"][0].update(batch_index=1),
            lambda x: x["batches"][0]["requests"].reverse(),
            lambda x: x["bindings"].update(producer_source_sha256="a" * 64),
            lambda x: x.update(train_context_ids="abcdefghijklmnopqrstuvwxyz012345678"),
        )
        for change in changes:
            receipt = copy.deepcopy(original)
            change(receipt)
            with self.subTest(receipt=receipt), self.assertRaises(ValueError):
                accounting.Ledger.from_dict(rehash(receipt))
        self.assertEqual(accounting.Ledger.from_dict(original), ledger)

    def test_replay_rejects_unresolved_batch_followed_by_new_reservations(self):
        ledger = new_ledger()
        first = ledger.next_batch()
        completed = ledger.reserve_batch(first).record_batch(terminal(first))
        second = completed.next_batch()
        with self.assertRaisesRegex(ValueError, "Unresolved requests"):
            replace(ledger, batches=(first, second))

    def test_import_and_accounting_need_only_stdlib_and_start_no_runtime(self):
        code = ("import sys; sys.path.insert(0, %r); import candidate_continuation_ledger as m; "
            "x=m.Ledger.new(tuple('train-%%03d'%%i for i in range(35))); "
            "assert x.summary()['total_new_task_requests']==0; "
            "assert 'torch' not in sys.modules; assert 'numpy' not in sys.modules; "
            "assert not any(n.startswith(('isaacsim','isaaclab','viam')) for n in sys.modules)") % str(SOURCE)
        subprocess.run([sys.executable, "-I", "-S", "-c", code], check=True)


if __name__ == "__main__":
    unittest.main()
