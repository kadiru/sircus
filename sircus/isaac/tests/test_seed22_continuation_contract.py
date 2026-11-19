"""Synthetic contract/layout/accounting checks; no model or native execution."""

import copy
import math
from pathlib import Path
import random
import sys
import unittest
from collections import Counter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import seed22_continuation_contract as contract


def train_fixture():
    # One invented, well-separated geometry point at each tile center.
    return [dict(id=f"train-{i:03d}", split="train", tile_index=i,
                 grid_row=i // 7, grid_column=i % 7,
                 bucket_xy_m=[-.10 + (i % 7 + .5) * 1.50 / 7,
                              -.70 + (i // 7 + .5) * .90 / 5]) for i in range(35)]


def resign(value):
    return contract.signed({key: item for key, item in value.items() if key != "fingerprint"})


class ProtocolTests(unittest.TestCase):
    def bindings(self):
        return dict(parent_checkpoint_sha256=contract.PARENT_CHECKPOINT_SHA256,
                    original_experiment_sha256=contract.ORIGINAL_EXPERIMENT_SHA256,
                    original_split_sha256=contract.ORIGINAL_SPLIT_SHA256,
                    geometry_layer_sha256=contract.GEOMETRY_LAYER_SHA256,
                    source_sha256={"synthetic_contract.py": "a" * 64},
                    exclusions_fingerprint="b" * 64)

    def test_fixed_new_caps_parent_and_continuous_likelihood(self):
        value = contract.make_protocol(**self.bindings())
        self.assertEqual(contract.validate_protocol(value), value["fingerprint"])
        self.assertEqual(value["additional_train_request_cap"], 30000)
        self.assertEqual(value["absolute_wall_seconds"], 28800)
        self.assertEqual(value["assessment_request_cap"], 525)
        self.assertEqual(value["inherited"], dict(requests=10000, updates=73, optimizer_steps=292))
        self.assertTrue(value["learning"]["restore_parent_optimizer"])
        self.assertEqual(value["learning"]["imitation_steps"], 0)
        self.assertIn("Original sampled latent", value["learning"]["likelihood_action"])
        self.assertFalse(value["execution_authorized"])

    def test_resealed_cap_learning_or_parent_mutations_rejected(self):
        value = contract.make_protocol(**self.bindings())
        for field, replacement in (("additional_train_request_cap", 30001),
                                   ("absolute_wall_seconds", 28801),
                                   ("assessment_request_cap", 526),
                                   ("execution_authorized", True)):
            changed = copy.deepcopy(value)
            changed[field] = replacement
            with self.assertRaises(ValueError):
                contract.validate_protocol(resign(changed))
        for key in ("parent_checkpoint_sha256", "original_experiment_sha256",
                    "original_split_sha256", "geometry_layer_sha256"):
            changed = self.bindings()
            changed[key] = "f" * 64
            with self.assertRaises(ValueError):
                contract.make_protocol(**changed)

    def test_missing_or_malformed_source_binding_rejected(self):
        for source in ({}, {"../bad.py": "a" * 64}, {"/bad.py": "a" * 64}, {"x.py": "bad"}):
            changed = self.bindings()
            changed["source_sha256"] = source
            with self.assertRaises(ValueError):
                contract.make_protocol(**changed)


class FreshSplitTests(unittest.TestCase):
    def setUp(self):
        self.train = train_fixture()
        self.history = [[-.02, -.65], [.8, -.18], [1.7, 2.]]
        self.value = contract.create_fresh_splits(self.train, self.history)

    def test_exact_parent_preservation_and_new_stratified_counts(self):
        self.assertEqual(self.value["train"], self.train)
        self.assertEqual([len(self.value[role]) for role in ("train", "validation", "test")], [35, 35, 70])
        self.assertEqual(Counter(cell["tile_index"] for cell in self.value["validation"]), {i: 1 for i in range(35)})
        self.assertEqual(Counter(cell["tile_index"] for cell in self.value["test"]), {i: 2 for i in range(35)})
        self.assertEqual(contract.validate_splits(self.value, original_train=self.train,
                                                excluded_coordinates=self.history), self.value["fingerprint"])
        self.assertEqual(self.value["outcome_files_read"], 0)
        self.assertEqual(self.value["native_requests"], 0)

    def test_all_new_points_have_history_and_mutual_separation(self):
        previous = list(self.history) + [cell["bucket_xy_m"] for cell in self.train]
        for cell in self.value["validation"] + self.value["test"]:
            point = cell["bucket_xy_m"]
            self.assertTrue(all(math.dist(point, old) >= .025 for old in previous))
            previous.append(point)

    def test_deterministic_and_no_global_rng_or_input_mutation(self):
        state = random.getstate()
        before = copy.deepcopy((self.train, self.history))
        again = contract.create_fresh_splits(self.train, self.history + self.history)
        self.assertEqual(again, self.value)
        self.assertEqual(random.getstate(), state)
        self.assertEqual((self.train, self.history), before)

    def test_external_parent_and_history_cannot_be_resealed_away(self):
        changed = copy.deepcopy(self.value)
        changed["validation"][0]["bucket_xy_m"][0] += .0001
        with self.assertRaises(ValueError):
            contract.validate_splits(resign(changed))
        old = copy.deepcopy(self.train)
        old[0]["bucket_xy_m"][0] += .001
        with self.assertRaises(ValueError):
            contract.validate_splits(self.value, original_train=old, excluded_coordinates=self.history)
        with self.assertRaises(ValueError):
            contract.validate_splits(self.value, original_train=self.train,
                                     excluded_coordinates=self.history + [[.11, -.61]])

    def test_finite_exhaustion_never_relaxes_spacing_or_replaces_tile(self):
        # Fill the first tile densely enough that no permissible draw remains.
        dense = [[-.10 + x * .01, -.70 + y * .01] for x in range(24) for y in range(21)]
        with self.assertRaisesRegex(ValueError, "no spacing relaxation"):
            contract.create_fresh_splits(self.train, dense, max_draws=105)
        with self.assertRaises(ValueError):
            contract.create_fresh_splits(self.train, self.history, split_seed=20261131)

    def test_nonfinite_coordinates_extra_labels_and_duplicate_train_rejected(self):
        for history in ([[float("nan"), 0]], [[True, 0]], [[0, float("inf")]]):
            with self.assertRaises(ValueError):
                contract.create_fresh_splits(self.train, history)
        changed = copy.deepcopy(self.train)
        changed[0]["captured"] = True
        with self.assertRaises(ValueError):
            contract.create_fresh_splits(changed, self.history)
        changed = copy.deepcopy(self.train)
        changed[1]["bucket_xy_m"] = changed[0]["bucket_xy_m"]
        with self.assertRaises(ValueError):
            contract.create_fresh_splits(changed, self.history)


class ScheduleTests(unittest.TestCase):
    def test_exact_30k_paid_coverage_and_exact_milestone_cuts(self):
        rows = contract.batch_schedule()
        self.assertEqual(sum(row["attempts"] for row in rows), 30000)
        self.assertEqual([row["milestone_after"] for row in rows if row["milestone_after"]],
                         [1000, 2000, 5000, 10000, 20000, 30000])
        paid = []
        for index, row in enumerate(rows):
            self.assertEqual(row["call_index"], index)
            self.assertEqual(row["batch_index"], index)
            self.assertEqual(row["attempt_start"], len(paid))
            self.assertEqual(sum(row["enabled"]), row["attempts"])
            self.assertEqual(len(set(row["site_indices"])), row["attempts"])
            paid.extend(row["site_indices"])
            self.assertEqual(row["attempt_stop"], len(paid))
        for start in range(0, len(paid) - 34, 35):
            self.assertEqual(set(paid[start:start + 35]), set(range(35)))

    def test_fixed_order_is_repeatable_and_small_bounds_preserve_prefix(self):
        full = contract.batch_schedule()
        self.assertEqual(full, contract.batch_schedule())
        self.assertEqual(contract.batch_schedule(1000), full[:len(contract.batch_schedule(1000))])
        self.assertEqual(sum(row["attempts"] for row in contract.batch_schedule(1)), 1)
        for args in (dict(max_requests=0), dict(max_requests=30001), dict(max_requests=True),
                     dict(seed=20261121), dict(milestones=(0, 500, 30000))):
            with self.assertRaises(ValueError):
                contract.batch_schedule(**args)


class LedgerTests(unittest.TestCase):
    def test_reservation_first_exact_coverage_and_no_duplicate_dispatch(self):
        ledger = contract.Ledger("synthetic-session")
        with self.assertRaises(ValueError):
            ledger.complete(0, "train", 35)
        first = ledger.reserve(0, "train", 35)
        self.assertEqual(ledger.snapshot()["train_reserved"], 35)
        self.assertEqual(ledger.snapshot()["train_terminal"], 0)
        self.assertEqual(len(set(first["payload"]["request_ids"])), 35)
        with self.assertRaises(ValueError):
            ledger.reserve(1, "train", 35)
        for count in (0, 34, 36, True):
            with self.assertRaises(ValueError):
                ledger.complete(0, "train", count)
        with self.assertRaises(ValueError):
            ledger.complete(0, "test", 35)
        ledger.complete(0, "train", 35)
        with self.assertRaises(ValueError):
            ledger.reserve(0, "train", 35)
        self.assertEqual(ledger.snapshot()["train_terminal"], 35)

    def test_30k_train_cap_counts_every_request_and_inherited_separate(self):
        ledger = contract.Ledger("synthetic-cap")
        for batch in contract.batch_schedule():
            ledger.reserve(batch["batch_index"], "train", batch["attempts"])
            ledger.complete(batch["batch_index"], "train", batch["attempts"])
        state = ledger.snapshot()
        self.assertEqual(state["train_reserved"], 30000)
        self.assertEqual(state["train_unresolved"], 0)
        self.assertEqual(state["inherited"]["requests"], 10000)
        with self.assertRaises(ValueError):
            ledger.reserve(10000, "train", 1)
        ledger.finish("request_cap")
        with self.assertRaises(ValueError):
            ledger.reserve(0, "validation", 35)

    def test_assessment_roles_share_525_cap_without_using_train(self):
        ledger = contract.Ledger("synthetic-assessment")
        for i in range(15):
            role = "validation" if i < 11 else "test"
            ledger.reserve(i, role, 35)
            ledger.complete(i, role, 35)
        state = ledger.snapshot()
        self.assertEqual(state["assessment_reserved"], 525)
        self.assertEqual(state["train_reserved"], 0)
        with self.assertRaises(ValueError):
            ledger.reserve(16, "test", 1)

    def test_failure_keeps_partial_request_cost_and_forbids_resume(self):
        ledger = contract.Ledger("synthetic-failure")
        ledger.reserve(0, "train", 35)
        ledger.fail("owned_native_timeout", terminal_count=12)
        state = ledger.snapshot()
        self.assertEqual((state["train_reserved"], state["train_terminal"], state["train_unresolved"]), (35, 12, 23))
        self.assertTrue(state["closed"])
        with self.assertRaises(ValueError):
            ledger.complete(0, "train", 35)
        with self.assertRaises(ValueError):
            ledger.reserve(1, "train", 35)
        self.assertFalse(hasattr(contract.Ledger, "from_dict"))

    def test_receipts_are_chained_detached_and_session_bound(self):
        ledger = contract.Ledger("synthetic-chain")
        reserved = ledger.reserve(0, "train", 2)
        reserved["payload"]["request_ids"].clear()
        completed = ledger.complete(0, "train", 2)
        events = ledger.history()
        self.assertEqual(len(events[0]["payload"]["request_ids"]), 2)
        self.assertEqual(completed["previous_sha256"], events[0]["fingerprint"])
        for event in events:
            contract.validate_fingerprint(event)
        self.assertEqual(ledger.snapshot()["journal_head"], completed["fingerprint"])
        events.clear()
        self.assertEqual(len(ledger.history()), 2)
        other = contract.Ledger("synthetic-other").reserve(0, "train", 2)
        self.assertNotEqual(other["payload"]["request_ids"], ledger.history()[0]["payload"]["request_ids"])

    def test_invalid_roles_counts_sessions_and_unresolved_finish_rejected(self):
        for session in ("", "../bad", "a" * 97):
            with self.assertRaises(ValueError):
                contract.Ledger(session)
        ledger = contract.Ledger("synthetic-invalid")
        for args in ((0, "heldout", 1), (-1, "train", 1), (0, "train", 0), (0, "train", 36), (0, "train", True)):
            with self.assertRaises(ValueError):
                ledger.reserve(*args)
        ledger.reserve(0, "train", 35)
        with self.assertRaises(ValueError):
            ledger.finish("timer")
        with self.assertRaises(ValueError):
            ledger.fail("timeout", terminal_count=36)


if __name__ == "__main__":
    unittest.main()
