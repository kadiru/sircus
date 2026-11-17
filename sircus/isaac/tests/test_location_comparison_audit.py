"""Independent trace mutation checks using fabricated policy/physics results."""

import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import location_comparison as comparison
from location_comparison_audit import audit_trace, digest, read_trace
from location_comparison_bounds import Bounds
from location_comparison_mock import FakeClock, MockEngine
from location_comparison_runtime import ComparisonRunner, Journal
from test_location_comparison import fixture, bind


def synthetic_plan():
    train, history, placement = fixture()
    plan = comparison.build_comparison(train, history, placement, bind(train, history, placement))
    # A synthetic fixture with the production-sized local cohort and budget.
    # It is never a production input or a measured placement manifest.
    plan["cohorts"]["test_local"] = plan["cohorts"]["test_local"][:25]
    plan["protocol"]["test_positions"]["local"] = 25
    plan["protocol"]["final_assessment_requests"] = 285
    plan["protocol"]["total_assessment_requests"] = 600
    plan.pop("fingerprint")
    return comparison.signed(plan)


def synthetic_run(plan, engine_type=MockEngine):
    clock = FakeClock()
    engine = engine_type(clock)
    bounds = Bounds(plan, clock, lambda: dict(free_storage_bytes=10 * 1024**3))
    return ComparisonRunner(plan, engine, bounds, Journal(), clock=clock).run()


def reseal(trace):
    previous = "0" * 64
    for index, event in enumerate(trace["events"]):
        event["seq"], event["previous"] = index, previous
        event.pop("fingerprint", None)
        event["fingerprint"] = digest(event)
        previous = event["fingerprint"]
    return trace


def first(trace, kind, predicate=lambda payload: True):
    return next(event["payload"] for event in trace["events"]
                if event["kind"] == kind and predicate(event["payload"]))


def with_rng_receipts(original):
    """Fabricate consistent RNG identities while preserving all accounting."""
    trace = copy.deepcopy(original)
    current, batches, checkpoints = {}, {}, {}
    for event in trace["events"]:
        kind, payload = event["kind"], event["payload"]
        if kind == "restore":
            model = payload["model"]
            current[model] = dict(action_rng_hash="a" * 64, optimizer_rng_hash="b" * 64)
            payload["state"].update(current[model])
        elif kind == "reserve":
            batches[payload["batch_id"]] = (payload["model"], payload["deterministic"])
        elif kind == "terminal":
            model, deterministic = batches[payload["batch_id"]]
            payload["state_before"].update(current[model])
            if not deterministic:
                current[model]["action_rng_hash"] = digest(["synthetic action RNG", model, payload["batch_id"]])
            payload["state_after"].update(current[model])
        elif kind == "update":
            model = payload["model"]
            payload["before"].update(current[model])
            current[model]["optimizer_rng_hash"] = digest(["synthetic optimizer RNG", model, payload["after"]["updates"]])
            payload["after"].update(current[model])
        elif kind == "checkpoint":
            model = payload["model"]
            payload["receipt"]["state"].update(current[model])
            checkpoints[(model, payload["label"])] = copy.deepcopy(payload["receipt"])
        elif kind == "frozen":
            payload["checkpoints"] = {model: copy.deepcopy(checkpoints[(model, "initial" if model == "parent" else "final")])
                                      for model in current}
    return reseal(trace)


class AuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.plan = synthetic_plan()
        cls.trace = synthetic_run(cls.plan)
        cls.rng_trace = with_rng_receipts(cls.trace)

    def reject_change(self, mutate):
        trace = copy.deepcopy(self.trace)
        mutate(trace)
        with self.assertRaises(ValueError):
            audit_trace(self.plan, reseal(trace))

    def test_full_equal_budget_trace_counts_rejections_and_remains_synthetic(self):
        result = audit_trace(self.plan, self.trace)
        self.assertEqual(result["status"], "synthetic_verified")
        self.assertEqual((result["requested"], result["completed"], result["unresolved"]), (18520, 18520, 0))
        self.assertEqual(result["updates"], {"original_usable": 64, "broader": 64})
        self.assertFalse(result["native_physics_verified"])
        self.assertFalse(result["checkpoint_bytes_independently_verified"])
        self.assertEqual(result["native_evidence_batches_checked"], 0)
        for counts in result["train"].values():
            self.assertEqual(counts["requests"], 8960)
            self.assertGreater(counts["rejected"], 0)
            self.assertEqual(counts["captures"] + counts["misses"] + counts["rejected"], 8960)
        broad = [x for x in result["assessments"] if x["phase"] == "test_broad"]
        self.assertEqual(sum(x["requests"] for x in broad), 210)
        self.assertEqual({x["model"] for x in broad}, {"parent", "original_usable", "broader"})
        for model in ("parent", "original_usable", "broader"):
            self.assertEqual(result["final_tests"][model]["test_broad"]["requests"], 70)
            self.assertEqual(result["final_tests"][model]["test_local"]["requests"], 25)

    def test_recorded_compute_scopes_are_separated_by_model_and_phase(self):
        trace = copy.deepcopy(self.trace)
        # Explicit fabricated wall measurements, not a benchmark of compute.
        for event in trace["events"]:
            if event["kind"] == "restore":
                event["payload"]["restore_wall_s"] = .003
            elif event["kind"] == "checkpoint":
                event["payload"]["checkpoint_wall_s"] = .004
            elif event["kind"] == "update":
                event["payload"]["update_wall_s"] = .005
        result = audit_trace(self.plan, reseal(trace))
        self.assertAlmostEqual(result["costs"]["restore_wall_s"], .009)
        self.assertAlmostEqual(result["costs"]["update_wall_s"], 128 * .005)
        for model in ("original_usable", "broader"):
            costs = result["per_model_phase_costs"][model]
            self.assertAlmostEqual(costs["update"]["update_wall_s"], 64 * .005)
            self.assertAlmostEqual(costs["checkpoint"]["checkpoint_wall_s"], 6 * .004)
            self.assertIn("episode_wall_s", costs["train"])
            self.assertIn("geometry_wall_s", costs["train"])
            self.assertIn("test_broad", costs)

    def test_hash_chain_cannot_hide_missing_events(self):
        changed = copy.deepcopy(self.trace)
        changed["events"].pop(10)
        with self.assertRaisesRegex(ValueError, "Trace chain"):
            audit_trace(self.plan, changed)

    def test_parent_optimizer_or_independent_model_identity_change_is_rejected(self):
        self.reject_change(lambda t: first(t, "restore", lambda p: p["model"] == "broader")["state"].update(
            optimizer_hash="e" * 64))
        self.reject_change(lambda t: first(t, "restore", lambda p: p["model"] == "broader")["state"].update(
            model_id="synthetic-original_usable"))

    def test_optional_rng_receipts_allow_train_sampling_and_optimizer_randomness(self):
        result = audit_trace(self.plan, self.rng_trace)
        self.assertEqual(result["status"], "synthetic_verified")
        update = first(self.rng_trace, "update")
        self.assertEqual(update["before"]["action_rng_hash"], update["after"]["action_rng_hash"])
        self.assertNotEqual(update["before"]["optimizer_rng_hash"], update["after"]["optimizer_rng_hash"])
        terminal = first(self.rng_trace, "terminal", lambda p: p["state_before"]["model_id"] == "synthetic-original_usable")
        self.assertNotEqual(terminal["state_before"]["action_rng_hash"], terminal["state_after"]["action_rng_hash"])

    def test_rng_pairing_assessment_learning_and_checkpoint_mutations_fail(self):
        mutations = (
            lambda t: first(t, "restore", lambda p: p["model"] == "broader")["state"].update(action_rng_hash="c" * 64),
            lambda t: first(t, "restore", lambda p: p["model"] == "broader")["state"].update(optimizer_rng_hash="c" * 64),
            lambda t: first(t, "terminal")["state_after"].update(action_rng_hash="c" * 64),
            lambda t: first(t, "terminal")["state_after"].update(optimizer_rng_hash="c" * 64),
            lambda t: first(t, "update")["after"].update(action_rng_hash="c" * 64),
            lambda t: first(t, "checkpoint")["receipt"]["state"].update(action_rng_hash="c" * 64),
            lambda t: first(t, "terminal")["state_before"].pop("action_rng_hash"),
        )
        for index, mutate in enumerate(mutations):
            with self.subTest(mutation=index):
                trace = copy.deepcopy(self.rng_trace)
                mutate(trace)
                with self.assertRaises(ValueError):
                    audit_trace(self.plan, reseal(trace))

    def test_bucket_or_slot_or_paid_request_identity_change_is_rejected(self):
        for field, value in (("bucket_xy_m", [.5, -.5]), ("physical_slot", 2), ("target_id", "another-target")):
            with self.subTest(field=field):
                self.reject_change(lambda t, f=field, v=value: first(t, "reserve")["cells"][0].update({f: v}))

    def test_assessment_cannot_be_trained_or_dispatched_stochastically(self):
        self.reject_change(lambda t: first(t, "reserve")["cells"][0].update(split="train"))
        self.reject_change(lambda t: first(t, "reserve").update(deterministic=False))

    def test_unreserved_proposal_and_missing_terminal_rows_fail(self):
        self.reject_change(lambda t: first(t, "proposal").update(batch_id="not-reserved"))
        self.reject_change(lambda t: first(t, "terminal")["rows"].pop())

    def test_original_observation_and_latent_cannot_be_replaced(self):
        self.reject_change(lambda t: first(t, "proposal")["packet"]["rows"][0]["observation"].__setitem__(0, .75))
        self.reject_change(lambda t: first(t, "proposal")["packet"]["rows"][0]["actions_normalized"].__setitem__(0, .75))

    def test_padded_or_rejected_rows_cannot_be_refunded_or_clean(self):
        def bad_rejection(trace):
            rows = first(trace, "terminal")["rows"]
            next(row for row in rows if row["outcome"] == "rejected")["reward"] = 0
        self.reject_change(bad_rejection)
        self.reject_change(lambda t: first(t, "terminal")["rows"][0].update(outcome="miss", clean=True, reward=-.5))

    def test_update_must_use_exactly_original_train_examples_once(self):
        self.reject_change(lambda t: first(t, "update")["request_ids"].pop())
        self.reject_change(lambda t: first(t, "update").update(examples_sha256="0" * 64))
        self.reject_change(lambda t: first(t, "update")["after"].update(optimizer_steps=293))

    def test_assessment_cannot_modify_model_or_final_checkpoint_choice(self):
        self.reject_change(lambda t: first(t, "terminal")["state_after"].update(weight_hash="b" * 64))
        def select_initial(trace):
            initial = first(trace, "checkpoint", lambda p: p["model"] == "original_usable" and p["label"] == "initial")
            first(trace, "frozen")["checkpoints"]["original_usable"] = initial["receipt"]
        self.reject_change(select_initial)

    def test_finish_cannot_inflate_catches_or_hide_requests(self):
        self.reject_change(lambda t: first(t, "finish")["counts"]["train"]["broader"].update(captures=8960))
        self.reject_change(lambda t: first(t, "finish")["bounds"]["train_requested"].update(broader=8959))
        self.reject_change(lambda t: first(t, "finish")["bounds"]["phase_elapsed_s"].update(broader=28800))

    def test_synthetic_result_cannot_be_relabelled_as_native(self):
        def flip_mode(trace):
            trace["mode"] = "native"
            first(trace, "start")["mode"] = "native"
        self.reject_change(flip_mode)
        calls = []
        result = audit_trace(self.plan, self.trace, native_verifier=lambda *args: calls.append(args) or True)
        self.assertFalse(calls)
        self.assertEqual(result["status"], "synthetic_verified")

    def test_crash_prefix_preserves_unresolved_paid_reservation(self):
        trace = copy.deepcopy(self.trace)
        cutoff = next(i for i, event in enumerate(trace["events"]) if event["kind"] == "reserve"
                      and event["payload"]["phase"] == "train")
        trace["events"] = trace["events"][:cutoff + 1]
        result = audit_trace(self.plan, trace)
        self.assertEqual(result["status"], "incomplete_preserved_no_retry")
        self.assertEqual((result["requested"], result["completed"], result["unresolved"]), (70, 35, 35))

    def test_caught_native_failure_keeps_charged_rows_and_never_tests(self):
        class FailureEngine(MockEngine):
            def execute(self, cells, proposal, film=False):
                if cells[0]["cohort"] == "train":
                    raise RuntimeError("fabricated dispatch failure")
                return super().execute(cells, proposal, film)
        trace = synthetic_run(self.plan, FailureEngine)
        result = audit_trace(self.plan, trace)
        self.assertEqual(result["status"], "incomplete_preserved_no_retry")
        self.assertEqual(result["unresolved"], 35)
        self.assertFalse(any(item["phase"].startswith("test_") for item in result["assessments"]))

    def test_trace_reader_rejects_duplicate_keys_and_nonfinite_values(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.json"
            for raw in ('{"mode":"native","mode":"synthetic"}', '{"cost":NaN}'):
                path.write_text(raw)
                with self.assertRaises(ValueError):
                    read_trace(path)


if __name__ == "__main__":
    unittest.main()
