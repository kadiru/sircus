"""Fabricated recovery certificates test readers only; no native data or process."""
import copy
from pathlib import Path
import sys
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import location_comparison_publication as publication
import location_comparison_recovery_publication as recovery
import test_location_comparison_publication as canonical_tests
from test_location_comparison_publication import seal, write


def recovery_presentation(report):
    """Invented report for DOM/media consumer tests, not an audit certificate."""
    value = copy.deepcopy(report)
    value["status"] = recovery.STATUS
    value["audit"].update(status=recovery.STATUS, original_audit_status="not_passed",
        native_shutdown_status="failed", native_clean_shutdown=False,
        session_id="f"*32, recovery_session_id="e"*32)
    for key in ("manifest_fingerprint", "comparison_fingerprint", "audit_sha256", "audit_exit_sha256",
                "native_exit_sha256", "native_exit_fingerprint", "audit_exit_fingerprint", "trace_sha256",
                "recovery_request_fingerprint", "inventory_sha256", "retirement_proof_sha256"):
        value["audit"].setdefault(key, "a"*64)
    value.setdefault("costs", {}).update(native_watchdog_wall_s=1000,
        shutdown_failure=dict(status="failed", original_audit_status="not_passed", clean_shutdown=False,
            reason=recovery.FAILURE_REASON, native_watchdog_wall_s=1000, last_progress_shared_phase_s=100,
            after_last_progress_wall_s=400, shared_phase_limit_s=3600,
            scope="Last-progress shared time is stale. Tail includes shutdown wait, not isolated compute."),
        recovery_audit=dict(wall_s=2, cpu_user_s=1, cpu_system_s=.1, peak_rss_bytes=1024,
            scope="wait4_exact_audit_child_including_reaped_descendants"))
    return value


class RecoveryPublicationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # A distinct temporary tree avoids mutating the canonical suite's data.
        cls.fixture = type("RecoveryFixture", (canonical_tests.PublicationTests,), {})
        cls.fixture.setUpClass()
        f = cls.fixture
        cls.output, cls.original = f.output, f.audit
        cls.recovered = f.root / "recovered"; cls.recovered.mkdir()
        (cls.output / "child-exit.json").unlink()
        for file in ("audit.json", "comparison-audit-exit.json"):
            (cls.original / file).unlink()
        native = copy.deepcopy(f.native)
        native.update(status="stopped_no_retry", reason=recovery.FAILURE_REASON,
            termination="stopped", exit_code=-15, start_monotonic=100., wall_s=1000.)
        native["final_snapshot"]["observed_monotonic"] = 700.
        cls.native = seal(native, compact=True)
        cls.native_sha = write(cls.output / "comparison-watchdog-exit.json", cls.native)
        cls.bindings = dict(manifest_sha256="1"*64, native_session_id=f.session,
            runtime_fingerprint=f.manifest_id, comparison_fingerprint=f.plan["fingerprint"],
            recovery_request_fingerprint="2"*64, native_exit_fingerprint=cls.native["fingerprint"],
            native_exit_sha256=cls.native_sha, inventory_sha256="3"*64,
            retirement_proof_sha256="4"*64, source_commit="5"*40)
        cls.limits = dict(wall_s=7200, min_free_bytes=6*2**30, cpu_cores=4,
                          memory_bytes=8*2**30, cpu_seconds=28800)
        cls.session = "e"*32
        cls.owner = seal(dict(schema=recovery.EXIT_SCHEMA, **cls.bindings, session_id=cls.session,
            owner=dict(pid=1001, start_token="boot:10", cookie=cls.session, ppid=1, sid=1001),
            child_owner=dict(pid=1002, start_token="boot:11", cookie=cls.session, ppid=1001, sid=1002),
            command=["/mock/python", "/mock/audit.py"], limits=cls.limits,
            start_monotonic=2000., deadline_monotonic=9200., storage_root="/mock/storage",
            cuda_visible_devices="", visible_gpu_devices=0), compact=True)
        cls.owner_sha = write(cls.recovered / "recovery-audit-owner.json", cls.owner)
        report = copy.deepcopy(f.envelope["report"])
        report.update({k:v for k,v in cls.bindings.items() if k != "runtime_fingerprint"})
        report.update(schema=recovery.REPORT_SCHEMA, status=recovery.STATUS,
            original_audit_passed=False, clean_shutdown=False, model_promoted=False)
        proof = {k:report[k] for k in (*recovery.PIN_FIELDS, "manifest_sha256", "manifest_fingerprint",
            "comparison_fingerprint", "native_session_id", "source_commit", "recovery_request_fingerprint",
            "original_audit_passed", "clean_shutdown", "model_promoted")}
        proof.update(trace_sha256=f.trace_sha, inventory_before_after_verified=True,
            scientific_body_preserved=True,
            original_verifier_sha256="0e196852b48070d423d4db61b894b93b36edc655bc5f92c77adbf6b3815eccdd",
            recovery_scientific_body_sha256="6bef2f77aa1800047d4c1d4a0416a2cca1c3d2cb475cb7895fae470a5f1f8947",
            journal=dict(event_count=len(f.trace["events"]), file_count=10,
                trace_head=f.trace["events"][-1]["fingerprint"], journal_sha256="7"*64))
        report.update(recovery_proof=proof, exit_proof=copy.deepcopy(proof))
        cls.report = seal(report)
        cls.report_sha = write(cls.recovered / "recovery-audit.json", cls.report)
        cls.receipt = seal(dict(cls.owner, status="recovery_exit_zero_report_verified", exit_code=0,
            reason=None, error_type=None, termination=None, cleanup_safe=True,
            independent_report_verified=True, owner_receipt_sha256=cls.owner_sha,
            audit_process_starts=1, audit_child_start_attempts=1, native_application_starts=0,
            optimizer_steps=0, new_shots=0, model_promoted=False,
            wall_s=2., cpu_user_s=1., cpu_system_s=.1, peak_rss_bytes=1024,
            measurement_scope="wait4_exact_audit_child_including_reaped_descendants",
            measured_cgroup_cpu_s=1.2, sampled_peak_cgroup_memory_bytes=2048,
            cgroup_limits=dict(cgroup_memory_limit_bytes=8*2**30, cgroup_cpu_cores=4,
                memory_current_bytes=2048, cpu_usage_seconds=2, visible_gpu_devices=0),
            report_sha256=cls.report_sha, report_fingerprint=cls.report["fingerprint"],
            report_status=recovery.STATUS), compact=True)
        cls.receipt_sha = write(cls.recovered / "recovery-audit-exit.json", cls.receipt)

    @classmethod
    def tearDownClass(cls):
        cls.fixture.tearDownClass()

    def build(self, **kw):
        f = self.fixture
        args = dict(original_audit_output=self.original,
            expected_manifest_fingerprint=f.manifest_id, expected_comparison_fingerprint=f.plan["fingerprint"],
            expected_session_id=f.session, expected_recovery_session_id=self.session,
            expected_recovery_audit_sha256=self.report_sha, expected_recovery_exit_sha256=self.receipt_sha,
            expected_native_exit_sha256=self.native_sha)
        args.update(kw)
        return recovery.build_recovery_publication(self.output, self.recovered, **args)

    def changed(self, path, value):
        return self.fixture.changed(self, path, value)

    def test_explicit_gate_exposes_results_with_failed_shutdown_and_separate_costs(self):
        result = self.build()
        self.assertTrue(publication.recovery_publication_verified(result))
        self.assertEqual(result["status"], recovery.STATUS)
        self.assertEqual(result["audit"]["original_audit_status"], "not_passed")
        self.assertFalse(result["audit"]["native_clean_shutdown"])
        self.assertFalse(result["model_promoted"])
        self.assertEqual(result["costs"]["shutdown_failure"]["after_last_progress_wall_s"], 400)
        self.assertEqual(result["costs"]["recovery_audit"]["peak_rss_bytes"], 1024)
        self.assertEqual(len(result["videos"]), 10)
        self.assertEqual(sum(v["phase"] == "train" for v in result["videos"]), 8)
        self.assertTrue(all(v["model"] == "parent" for v in result["videos"] if v["phase"] != "train"))
        for model in publication.MODELS[1:]:
            self.assertEqual(len(result["train_curves"][model]), 256)
            self.assertEqual(result["train"][model]["requests"], 8960)
        for model in publication.MODELS:
            self.assertEqual(result["final_tests"][model]["test_broad"]["requests"], 70)
            self.assertEqual(result["final_tests"][model]["test_local"]["requests"], 25)

    def test_original_gate_still_rejects_failed_attempt(self):
        f = self.fixture
        result = publication.build_publication(self.output, self.original,
            expected_manifest_fingerprint=f.manifest_id, expected_comparison_fingerprint=f.plan["fingerprint"],
            expected_session_id=f.session, expected_audit_sha256=self.report_sha)
        self.assertEqual(result["status"], "stopped_no_retry")
        self.assertNotIn("final_tests", result)

    def test_missing_wrong_or_cross_run_external_pins_fail(self):
        for key in ("expected_recovery_audit_sha256", "expected_recovery_exit_sha256", "expected_native_exit_sha256",
                    "expected_manifest_fingerprint", "expected_comparison_fingerprint"):
            for value in (None, "bad", "0"*64):
                with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                    self.build(**{key:value})
        with self.assertRaises(ValueError):
            self.build(expected_recovery_session_id=self.fixture.session)

    def test_failed_or_malformed_receipt_never_reads_outcomes(self):
        mutations = [dict(status="recovery_stopped_no_retry"), dict(exit_code=False), dict(reason="failed"),
            dict(error_type="Error"), dict(termination="stopped"), dict(cleanup_safe=False),
            dict(independent_report_verified=False), dict(audit_process_starts=2), dict(new_shots=1),
            dict(optimizer_steps=1), dict(visible_gpu_devices=True), dict(cuda_visible_devices="0"),
            dict(model_promoted=True), dict(report_status="native_verified"), dict(report_sha256="0"*64),
            dict(native_session_id="c"*32), dict(owner_receipt_sha256="0"*64), dict(wall_s=7200),
            dict(peak_rss_bytes=-1), dict(measurement_scope="estimated"), dict(limits={}),
            dict(child_owner=dict(self.receipt["child_owner"], ppid=999)), dict(cgroup_limits={}),
            dict(deadline_monotonic=9999), dict(sampled_peak_cgroup_memory_bytes=9*2**30)]
        for change in mutations:
            with self.subTest(change=change):
                changed = seal(dict(self.receipt, **change), compact=True)
                with self.changed(self.recovered/"recovery-audit-exit.json", changed) as pin:
                    with mock.patch.object(recovery, "_present_verified", side_effect=AssertionError("Outcomes opened")):
                        with self.assertRaises(ValueError):
                            self.build(expected_recovery_exit_sha256=pin)

    def test_resealed_scientific_malformations_fail(self):
        for change in (dict(status="native_verified"), dict(synthetic=True), dict(original_audit_passed=True),
                       dict(clean_shutdown=True), dict(model_promoted=True), dict(update_artifacts=127),
                       dict(model_likelihoods_independently_verified=False), dict(manifest_sha256="0"*64),
                       dict(recovery_request_fingerprint="0"*64), dict(source_commit="0"*40)):
            report = seal(dict(self.report, **change))
            with self.subTest(change=change), self.changed(self.recovered/"recovery-audit.json", report) as pin:
                receipt = seal(dict(self.receipt, report_sha256=pin, report_fingerprint=report["fingerprint"]), compact=True)
                with self.changed(self.recovered/"recovery-audit-exit.json", receipt) as exit_pin:
                    with self.assertRaises(ValueError):
                        self.build(expected_recovery_audit_sha256=pin, expected_recovery_exit_sha256=exit_pin)

    def test_changed_parity_journal_and_inventory_proofs_fail(self):
        for change in (dict(scientific_body_preserved=False), dict(original_verifier_sha256="0"*64),
                       dict(recovery_scientific_body_sha256="0"*64), dict(inventory_before_after_verified=False),
                       dict(journal={}), dict(child_exit_sha256="0"*64), dict(manifest_sha256="0"*64)):
            proof = dict(self.report["recovery_proof"], **change)
            report = seal(dict(self.report, recovery_proof=proof, exit_proof=proof))
            with self.subTest(change=change), self.changed(self.recovered/"recovery-audit.json", report) as pin:
                receipt = seal(dict(self.receipt, report_sha256=pin, report_fingerprint=report["fingerprint"]), compact=True)
                with self.changed(self.recovered/"recovery-audit-exit.json", receipt) as exit_pin:
                    with self.assertRaises(ValueError):
                        self.build(expected_recovery_audit_sha256=pin, expected_recovery_exit_sha256=exit_pin)

    def test_added_original_clean_exit_or_audit_is_rejected(self):
        for root, name in ((self.output,"child-exit.json"),(self.original,"audit.json"),
                           (self.original,"comparison-audit-exit.json")):
            path = root/name; path.write_text("{}")
            try:
                with self.assertRaises(ValueError): self.build()
            finally: path.unlink()

    def test_missing_failure_presentation_pins_or_inconsistent_costs_withhold_results(self):
        good = self.build()
        for key in ("native_exit_sha256", "audit_exit_sha256", "inventory_sha256", "retirement_proof_sha256"):
            bad = copy.deepcopy(good); bad["audit"].pop(key)
            self.assertFalse(publication.recovery_publication_verified(bad))
        for change in (dict(status="passed"), dict(clean_shutdown=True), dict(after_last_progress_wall_s=1001),
                       dict(native_watchdog_wall_s=999), dict(scope=""), dict(shared_phase_limit_s=9999)):
            bad = copy.deepcopy(good); bad["costs"]["shutdown_failure"].update(change)
            self.assertFalse(publication.recovery_publication_verified(bad))


if __name__ == "__main__": unittest.main()
