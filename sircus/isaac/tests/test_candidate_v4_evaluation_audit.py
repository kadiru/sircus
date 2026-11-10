"""Independent synthetic TEST accounting checks; no model or native work."""

import copy
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import candidate_v4_evaluation_audit as subject
from candidate_v4_evaluation_runner import run_mock_evaluation
from test_candidate_v4_evaluation import fixture


class MockRuntime:
    synthetic = True
    mock_only = True

    def __init__(self):
        self.closed, self.calls, self.summary = False, [], None

    def initialize(self, plan):
        self.calls.append("initialize")

    def verify_frozen(self, plan):
        return dict(final_checkpoint_sha256=plan["final_checkpoint_sha256"],
            parent_checkpoint_sha256=plan["parent_checkpoint_sha256"], bindings=copy.deepcopy(plan["bindings"]),
            model_hashes=copy.deepcopy(plan["original_plan"]["model_hashes"]))

    def execute(self, cohort, choices, reservation):
        self.calls.append(cohort["index"])
        rows = []
        for choice in choices:
            number = int(choice["context_id"].rsplit("-", 1)[-1])
            released = choice["enabled"]
            captured = released and number % dict(parent=3, candidate_bank=4, continued=2)[choice["method"]] == 0
            clean = captured and number % 5 == 0
            rows.append(dict(request_id=choice["request_id"], released=released,
                decline_reason=None if released else "no_qualified_candidate_support", captured=captured, clean=clean,
                source_sha256=hashlib.sha256(choice["request_id"].encode()).hexdigest(),
                **{"return": (1.6 if clean else 1.5) if captured else (0. if released else -1.)}))
        return rows

    def close(self):
        self.closed = True

    def audit(self, directory, plan, summary):
        if not self.closed:
            raise AssertionError("mock must close before independent audit")
        self.summary = copy.deepcopy(summary)
        return subject.audit_mock_evaluation(directory, plan, summary)


class EvaluationAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.prepared = fixture()
        cls.base_temp = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.base_temp.cleanup)
        root = Path(cls.base_temp.name).resolve()
        (root / "protected").mkdir()
        runtime = MockRuntime()
        cls.base_directory = root / "base"
        cls.base_result = run_mock_evaluation(cls.base_directory, cls.prepared, runtime,
            protected_roots=[root / "protected"], clock=lambda:0., free_bytes=lambda:10*1024**3,
            usage=lambda:dict(cpu_s=0., lifetime_peak_rss_mib=1.))
        cls.base_summary = copy.deepcopy(runtime.summary)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.protected = self.root / "preserved-v4"; self.protected.mkdir()
        self.sentinel = self.protected / "partial-assessment.json"
        self.sentinel.write_bytes(b'{"requested":140,"terminal":105,"unresolved":35}\n')
        self.before = self.sentinel.read_bytes()
        self.plan = copy.deepcopy(self.prepared)
        self.directory = self.root / "new-test-only"
        shutil.copytree(self.base_directory, self.directory)
        self.result = copy.deepcopy(self.base_result)
        self.summary = copy.deepcopy(self.base_summary)

    def audit(self, **changes):
        return subject.audit_mock_evaluation(changes.get("directory", self.directory),
            changes.get("plan", self.plan), changes.get("summary", self.summary))

    def rewrite(self, modify):
        paths = sorted((self.directory / "journal").glob("event-*.json"))
        events = [json.loads(path.read_bytes()) for path in paths]
        modify(events)
        head = "0" * 64
        for i, event in enumerate(events):
            event["seq"], event["previous_sha256"] = i, head
            event.pop("fingerprint", None)
            event["fingerprint"] = subject.canonical_hash(event)
            head = event["fingerprint"]
            paths[i].write_text(json.dumps(event, sort_keys=True, allow_nan=False))
        self.summary["journal_head_sha256"] = head
        return events

    def test_complete210_same70each_four_empty_declines_and_prior_unchanged(self):
        audit = self.audit()
        self.assertEqual(audit["status"], "passed_mock_only")
        self.assertFalse(audit["full_native_audit"])
        self.assertEqual(audit["counts"], dict(requested=210, terminal=210, released=206,
            rejected=4, captures=74, clean_captures=16, unresolved=0))
        methods = audit["comparison"]["methods"]
        self.assertEqual([methods[m]["requested"] for m in subject.METHODS], [70, 70, 70])
        self.assertEqual([methods[m]["rejected"] for m in subject.METHODS], [0, 2, 2])
        self.assertTrue(audit["synthetic"] and audit["mock_only"])
        self.assertFalse(audit["original_assessment_completed"])
        self.assertEqual(audit["original_assessment_status"], "incomplete_no_retry")
        self.assertEqual(self.sentinel.read_bytes(), self.before)
        self.assertEqual(audit["training_requests"], 0)
        self.assertEqual(audit["optimizer_steps"], 0)
        self.assertEqual(audit["native_requests_executed"], 0)

    def test_paired_counts_are_recomputed_by_context(self):
        pairs = self.audit()["comparison"]["test_pairs"]
        for control in ("parent", "candidate_bank"):
            pair = pairs["continued_minus_" + control]
            for metric in ("captured", "clean"):
                row = pair[metric]
                self.assertEqual(row["gains"] + row["losses"] + row["ties"], 70)
                self.assertEqual(row["gains"], len(row["gain_context_ids"]))
                self.assertEqual(row["losses"], len(row["loss_context_ids"]))
                self.assertFalse(set(row["gain_context_ids"]) & set(row["loss_context_ids"]))
                self.assertEqual(row["difference_rate"], (row["gains"] - row["losses"]) / 70.)

    def test_repeated_result_is_rejected_even_after_rehash(self):
        def edit(events):
            rows = events[2]["payload"]["results"]
            rows[-1]["request_id"] = rows[0]["request_id"]
        self.rewrite(edit)
        with self.assertRaisesRegex(ValueError, "outcome identity"):
            self.audit()

    def test_reservation_frozen_action_is_checked_independently(self):
        def edit(events):
            action = events[1]["payload"]["choices"][0]["action"]
            key = next(k for k,v in action.items() if type(v) is float)
            action[key] += .001
        self.rewrite(edit)
        with self.assertRaisesRegex(ValueError, "Reservation changed"):
            self.audit()

    def test_empty_support_cannot_be_counted_as_capture(self):
        def edit(events):
            for event in events:
                if event["kind"] != "outcomes":
                    continue
                for row in event["payload"]["results"]:
                    if row["decline_reason"] == "no_qualified_candidate_support":
                        row.update(released=True, captured=True, clean=False, decline_reason=None)
                        row["return"] = 1.5
                        return
        self.rewrite(edit)
        with self.assertRaisesRegex(ValueError, "empty support"):
            self.audit()

    def test_rehashed_source_swap_still_fails_against_prepared_binding(self):
        def edit(events):
            for event in (events[0], events[-1]):
                event["payload"]["frozen"]["bindings"]["evaluation_source_sha256"] = "e" * 64
        self.rewrite(edit)
        self.summary["frozen"]["bindings"]["evaluation_source_sha256"] = "e" * 64
        with self.assertRaisesRegex(ValueError, "immutable checkpoint/source/model"):
            self.audit()

    def test_rehashed_weight_swap_still_fails_against_prepared_model(self):
        def edit(events):
            for event in (events[0], events[-1]):
                event["payload"]["frozen"]["model_hashes"]["continued_weight_hash"] = "f" * 64
        self.rewrite(edit)
        self.summary["frozen"]["model_hashes"]["continued_weight_hash"] = "f" * 64
        with self.assertRaisesRegex(ValueError, "immutable checkpoint/source/model"):
            self.audit()

    def test_checkpoint_swap_is_rejected(self):
        self.summary["frozen"]["final_checkpoint_sha256"] = "f" * 64
        with self.assertRaisesRegex(ValueError, "Mock result identity"):
            self.audit()

    def test_mixed_provenance_and_fake_native_pass_are_rejected(self):
        for changed in (dict(synthetic=False), dict(mock_only=False), dict(full_native_audit=True), dict(model_promoted=True)):
            summary = dict(self.summary, **changed)
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                self.audit(summary=summary)

    def test_native_starts_and_permission_claims_cannot_enter_summary(self):
        for changed in (dict(native_application_starts=1), dict(native_application_starts=False),
                dict(execution_authorized=True), dict(release_authorized=True),
                dict(original_assessment_completed=True), dict(native_execution_ready=True)):
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                self.audit(summary=dict(self.summary, **changed))

    def test_saved_plan_cannot_be_changed_missing_or_linked(self):
        path = self.directory / "evaluation-plan.json"
        original = path.read_bytes()
        path.write_bytes(b'{}')
        with self.assertRaisesRegex(ValueError, "Durable prepared"):
            self.audit()
        path.unlink()
        with self.assertRaisesRegex(ValueError, "Durable prepared"):
            self.audit()
        target = self.root / "elsewhere-plan.json"; target.write_bytes(original)
        path.symlink_to(target)
        with self.assertRaisesRegex(ValueError, "linked evaluation artifacts"):
            self.audit()

    def test_failure_file_cannot_coexist_with_a_complete_journal(self):
        (self.directory / "evaluation-failure.json").write_text('{"status":"mock_incomplete_no_retry"}')
        with self.assertRaisesRegex(ValueError, "Interrupted, foreign"):
            self.audit()

    def test_missing_final_receipt_cannot_pass(self):
        paths = sorted((self.directory / "journal").glob("event-*.json"))
        paths[-1].unlink()
        with self.assertRaisesRegex(ValueError, "no complete final"):
            self.audit()

    def test_short_prefix_failure_never_passes_or_refunds(self):
        directory = self.root / "interrupted"
        class Partial(MockRuntime):
            def execute(self, cohort, choices, reservation):
                return super().execute(cohort, choices, reservation)[:7]
        with self.assertRaisesRegex(ValueError, "Partial cohort"):
            run_mock_evaluation(directory, self.plan, Partial(), protected_roots=[self.protected],
                clock=lambda:0., free_bytes=lambda:10*1024**3,
                usage=lambda:dict(cpu_s=0., lifetime_peak_rss_mib=1.))
        failure = json.loads((directory / "evaluation-failure.json").read_bytes())
        self.assertEqual(failure["counts"]["requested"], 35)
        self.assertEqual(failure["counts"]["terminal"], 7)
        self.assertEqual(failure["counts"]["unresolved"], 28)
        with self.assertRaisesRegex(ValueError, "Interrupted, foreign|failed evaluation event"):
            self.audit(directory=directory)
        self.assertFalse((directory / "evaluation-result.json").exists())

    def test_duplicate_json_keys_are_not_silently_accepted(self):
        path = self.directory / "journal/event-000000.json"
        raw = path.read_text()
        path.write_text(raw.replace('{', '{"seq":99,', 1))
        with self.assertRaisesRegex(ValueError, "Duplicate audit JSON"):
            self.audit()

    def test_native_folder_cannot_be_passed_off_as_mock_evidence(self):
        (self.directory / "native").mkdir()
        with self.assertRaisesRegex(ValueError, "cannot consume a native"):
            self.audit()

    def test_optimizer_and_training_counts_cannot_be_introduced(self):
        for field in ("training_requests", "optimizer_steps", "native_requests_executed"):
            with self.subTest(field=field):
                summary = dict(self.summary, **{field:1})
                with self.assertRaisesRegex(ValueError, "work boundary"):
                    self.audit(summary=summary)


if __name__ == "__main__":
    unittest.main()
