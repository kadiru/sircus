"""Invented certificate/trace fixtures test publishing only, never native audit.

Native-looking certificates below are fabricated inside temporary test folders
to exercise an explicitly externally-pinned certificate consumer. They are NOT
produced by the native auditor, are never published and prove no policy result.
"""
import copy
from contextlib import contextmanager
import gzip
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import location_comparison_publication as publication
from location_comparison_audit import audit_trace
from test_location_comparison_audit import synthetic_plan, synthetic_run, reseal


def seal(value, compact=False):
    result = {k: v for k, v in value.items() if k != "fingerprint"}
    return dict(result, fingerprint=publication.digest(result, compact=compact))


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = json.dumps(value, sort_keys=True, allow_nan=False).encode()
    if str(path).endswith(".gz"):
        with gzip.open(path, "wb") as stream:
            stream.write(raw)
    else:
        path.write_bytes(raw)
    return hashlib.sha256(path.read_bytes()).hexdigest()


class PublicationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name).resolve()
        cls.output, cls.audit = cls.root / "native", cls.root / "audit"
        cls.output.mkdir(); cls.audit.mkdir()
        cls.plan = synthetic_plan()
        trace = synthetic_run(cls.plan)
        cls.accounting = audit_trace(cls.plan, trace)
        # Deliberately invented receipts for the consumer boundary, not an
        # injected native audit or a conversion of any measured experiment.
        trace["mode"] = "native"
        trace["events"][0]["payload"]["mode"] = "native"
        reserved, frame_index, checkpoint_hashes = {}, 0, {}
        for event in trace["events"]:
            payload = event["payload"]
            if event["kind"] == "reserve":
                reserved[payload["batch_id"]] = payload
            elif event["kind"] == "terminal":
                reservation = reserved[payload["batch_id"]]
                payload["costs"]["resources_before"] = dict(lifetime_peak_rss_mib=123., sampled_gpu_used_mib=456.)
                payload["costs"]["resources_after"] = dict(lifetime_peak_rss_mib=125., sampled_gpu_used_mib=450.)
                if reservation["film"]:
                    frames = []
                    for _ in range(2):
                        name = cls.output / f"native/frames/{frame_index:05d}.jpg"
                        name.parent.mkdir(parents=True, exist_ok=True)
                        name.write_bytes(b"FAKE JPEG FOR PUBLICATION MOCK")
                        frames.append(dict(frame=frame_index, jpeg_sha256=publication._file_hash(name)))
                        frame_index += 1
                    name = f"comparison-native/trial-{int(payload['batch_id'].split('-')[1]):05d}.json.gz"
                    checksum = write(cls.output / name, dict(schema="sircus-location-comparison-native-trial-v1",
                        synthetic=False, filming=True, frames=frames, raw_physics="must not appear in publication"))
                    payload["evidence"] = dict(file=name, sha256=checksum)
            elif event["kind"] == "frozen":
                for model, cp in payload["checkpoints"].items():
                    cp["file"] = f"checkpoints/{model}-final.fake"
                    cp["sha256"] = write(cls.output / cp["file"], {"synthetic_fixture": True, "model": model})
                    checkpoint_hashes[cp["file"]] = cp["sha256"]
        cls.trace = reseal(trace)
        cls.manifest_id, cls.session = "a" * 64, "f" * 32
        cls.identity = dict(runtime_fingerprint=cls.manifest_id,
            comparison_fingerprint=cls.plan["fingerprint"], session_id=cls.session)
        cls.trace_sha = write(cls.output / "comparison-trace.json", cls.trace)
        child = dict(cls.identity, status="native_complete_pending_audit", synthetic=False,
            app_closed=True, error=None, trace_sha256=cls.trace_sha)
        child_sha = write(cls.output / "child-exit.json", child)
        cls.native = seal(dict(cls.identity, schema="sircus-location-comparison-watchdog-v1",
            status="native_exit_zero_independent_audit_required", exit_code=0, reason=None,
            lock_release_safe=True, native_application_starts=1, wall_s=100.,
            final_snapshot=cls.trace["events"][-1]["payload"]["bounds"]), compact=True)
        write(cls.output / "comparison-watchdog-exit.json", cls.native)
        cls.audit_exit = seal(dict(cls.identity, schema="sircus-location-comparison-audit-watchdog-v1",
            status="audit_exit_zero_report_verified", exit_code=0, reason=None, cleanup_safe=True,
            independent_report_verified=True, cuda_visible_devices="", model_promoted=False,
            native_exit_fingerprint=cls.native["fingerprint"], audit_process_starts=1,
            audit_child_start_attempts=1, native_application_starts=0, wall_s=2.), compact=True)
        write(cls.audit / "comparison-audit-exit.json", cls.audit_exit)
        accounting = copy.deepcopy(cls.accounting)
        accounting.update(status="native_accounting_verified_physics_pending", synthetic=False,
            trace_head=cls.trace["events"][-1]["fingerprint"])
        report = seal(dict(schema="sircus-location-comparison-independent-audit-v1", status="native_verified",
            synthetic=False, manifest_fingerprint=cls.manifest_id, comparison_fingerprint=cls.plan["fingerprint"],
            trace_canonical_sha256=publication.digest(cls.trace), trace_head=cls.trace["events"][-1]["fingerprint"],
            checkpoint_bytes_independently_verified=True, model_likelihoods_independently_verified=True,
            native_physics_verified=True, paid_policy_samples=18520, terminal_batches=539,
            update_artifacts=128, checkpoint_count=144, startup=dict(status="passed"),
            accounting=seal(accounting), checkpoint_sha256=checkpoint_hashes,
            exit_proof=dict(native_exit_fingerprint=cls.native["fingerprint"], trace_sha256=cls.trace_sha, child_exit_sha256=child_sha)))
        cls.envelope = seal(dict(schema="sircus-location-comparison-audit-envelope-v1", status="native_verified",
            synthetic=False, session_id=cls.session, manifest_fingerprint=cls.manifest_id,
            comparison_fingerprint=cls.plan["fingerprint"], native_exit_fingerprint=cls.native["fingerprint"],
            trace_sha256=cls.trace_sha, report=report))
        cls.audit_sha = write(cls.audit / "audit.json", cls.envelope)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def build(self, **kwargs):
        args = dict(expected_manifest_fingerprint=self.manifest_id,
            expected_comparison_fingerprint=self.plan["fingerprint"], expected_session_id=self.session,
            expected_audit_sha256=self.audit_sha)
        args.update(kwargs)
        return publication.build_publication(self.output, self.audit, **args)

    @contextmanager
    def changed(self, path, value):
        old = path.read_bytes()
        checksum = write(path, value)
        try:
            yield checksum
        finally:
            path.write_bytes(old)

    def test_completed_certificate_exposes_exact_denominators_complete_curves_and_scopes(self):
        result = self.build()
        self.assertEqual(result["status"], "completed_audited")
        self.assertTrue(result["audit"]["complete"])
        self.assertFalse(result["model_promoted"])
        for model in publication.MODELS[1:]:
            self.assertEqual(result["train"][model]["requests"], 8960)
            curve = result["train_curves"][model]
            self.assertEqual([p["train_requests"] for p in curve], list(range(35, 8961, 35)))
            self.assertEqual(sum(p["captures"] for p in curve), result["train"][model]["captures"])
            monitor = result["monitor_curves"][model]
            self.assertEqual([p["train_requests"] for p in monitor], [0,2240,4480,6720,8960])
            self.assertEqual(monitor[0]["source_model"], "parent")
        for model in publication.MODELS:
            for phase, denominator in (("test_broad", 70), ("test_local", 25)):
                row = result["final_tests"][model][phase]
                self.assertEqual(row["requests"], denominator)
                self.assertEqual(row["capture_rate"], row["captures"] / denominator)
                self.assertEqual(row["captures"]+row["misses"]+row["rejected"], denominator)
        costs = result["costs"]
        self.assertEqual(len(costs["memory_samples"]), 1078)
        self.assertTrue(costs["geometry_nested_in_episode"])
        self.assertEqual(max(s["lifetime_peak_rss_mib"] for s in costs["memory_samples"]), 125.)
        self.assertIn("whole-device", costs["memory_scope"])
        self.assertNotIn("audit_cpu_s", costs)
        self.assertNotIn("raw_physics", json.dumps(result))
        self.assertNotIn(str(self.root), json.dumps(result))

    def test_recording_coverage_labels_and_inactive_slots_are_factual(self):
        videos = self.build()["videos"]
        self.assertEqual(len(videos), 10)
        self.assertEqual(sum(v["phase"] == "train" for v in videos), 8)
        tests = [v for v in videos if v["phase"] != "train"]
        self.assertEqual([(v["model"], v["active_slots"], v["parked_slots"]) for v in tests],
                         [("parent",24,11),("parent",9,26)])
        self.assertTrue(all(v["media"] is None and v["frames"] == 2 for v in videos))
        self.assertIn("requests 1–35", videos[0]["label"])
        self.assertTrue(all(v["chapters"][0]["start_s"] == 0 for v in videos))

    def test_no_external_pin_never_reads_outcome_files(self):
        original = publication._read
        def safe(root, name, *args):
            self.assertNotIn(name, ("audit.json", "comparison-trace.json"))
            self.assertFalse(name.startswith("comparison-native/"))
            return original(root, name, *args)
        with mock.patch.object(publication, "_read", side_effect=safe):
            result = self.build(expected_audit_sha256=None)
        self.assertEqual(result["status"], "awaiting_audit")
        for key in ("train", "final_tests", "train_curves", "costs", "videos"):
            self.assertNotIn(key, result)

    def test_missing_audit_completion_never_reads_trace_or_envelope(self):
        path = self.audit / "comparison-audit-exit.json"
        old = path.read_bytes(); path.unlink()
        try:
            with mock.patch.object(publication, "_gate", side_effect=AssertionError("Outcome gate called")):
                result = self.build()
            self.assertEqual(result["status"], "awaiting_audit")
        finally:
            path.write_bytes(old)

    def test_failed_audit_withholds_all_results_even_with_pin(self):
        value = seal(dict(self.audit_exit, status="audit_stopped_no_retry", independent_report_verified=False), compact=True)
        with self.changed(self.audit / "comparison-audit-exit.json", value):
            result = self.build()
        self.assertEqual(result["status"], "stopped_no_retry")
        self.assertFalse(result["audit"]["complete"])
        self.assertNotIn("final_tests", result)

    def test_safe_progress_excludes_raw_error_outcomes_and_coordinates(self):
        snapshot = copy.deepcopy(self.trace["events"][-1]["payload"]["bounds"])
        snapshot["leaked_data"] = dict(bucket_xy_m=[1,2], captures=99)
        progress = seal(dict(self.identity, schema="sircus-location-comparison-process-progress-v1",
            snapshot=snapshot, running=True, status="running", arbitrary_secret="do not publish"), compact=True)
        path = self.output / "comparison-progress.json"
        write(path, progress)
        try:
            result = self.build(expected_audit_sha256=None)
            self.assertEqual(result["progress"]["broader"]["requested"], 8960)
            self.assertNotIn("leaked_data", json.dumps(result))
            self.assertNotIn("do not publish", json.dumps(result))
        finally:
            path.unlink()

    def test_wrong_pin_or_session_rejected_before_outcomes(self):
        with self.assertRaisesRegex(ValueError, "Changed evidence bytes"):
            self.build(expected_audit_sha256="0"*64)
        with self.assertRaisesRegex(ValueError, "Cross-run"):
            self.build(expected_session_id="0"*32)

    def test_synthetic_or_accounting_only_certificate_cannot_pass(self):
        for mutation in ({"synthetic": True}, {"native_physics_verified": False},
                         {"paid_policy_samples": 18519}, {"startup": {"status": "incomplete"}}):
            value = copy.deepcopy(self.envelope)
            value["report"] = seal(dict(value["report"], **mutation))
            value = seal(value)
            with self.changed(self.audit / "audit.json", value) as checksum:
                with self.assertRaisesRegex(ValueError, "Complete independent native audit"):
                    self.build(expected_audit_sha256=checksum)

    def test_native_cleanup_and_cpu_audit_gates_fail_closed(self):
        for mutation in ({"cleanup_safe": False}, {"cuda_visible_devices": "0"}, {"audit_process_starts": 2}):
            value = seal(dict(self.audit_exit, **mutation), compact=True)
            with self.changed(self.audit / "comparison-audit-exit.json", value):
                with self.assertRaisesRegex(ValueError, "complete safely"):
                    self.build()

    def test_trace_checkpoint_and_recorded_frame_tampering_fail(self):
        paths = [self.output / "comparison-trace.json", self.output / "checkpoints/parent-final.fake",
                 self.output / "native/frames/00000.jpg"]
        for path in paths:
            old = path.read_bytes(); path.write_bytes(old + b"CHANGED")
            try:
                with self.assertRaises(ValueError):
                    self.build()
            finally:
                path.write_bytes(old)

    def test_resealed_certificate_cannot_replace_test_denominator(self):
        value = copy.deepcopy(self.envelope)
        value["report"]["accounting"]["final_tests"]["broader"]["test_broad"]["requests"] = 69
        value["report"]["accounting"] = seal(value["report"]["accounting"])
        value["report"] = seal(value["report"]); value = seal(value)
        with self.changed(self.audit / "audit.json", value) as checksum:
            with self.assertRaisesRegex(ValueError, "TEST/accounting"):
                self.build(expected_audit_sha256=checksum)

    def test_duplicate_json_and_nonfinite_measurements_fail_closed(self):
        path = self.audit / "audit.json"
        old = path.read_bytes()
        try:
            path.write_text('{"report": {}, "report": {}}')
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                self.build(expected_audit_sha256=publication._file_hash(path))
            path.write_text('{"report": NaN}')
            with self.assertRaisesRegex(ValueError, "Nonfinite"):
                self.build(expected_audit_sha256=publication._file_hash(path))
        finally:
            path.write_bytes(old)

    def test_broken_process_fingerprint_cannot_unlock_outcomes(self):
        value = copy.deepcopy(self.audit_exit)
        value["wall_s"] += 1
        with self.changed(self.audit / "comparison-audit-exit.json", value):
            with mock.patch.object(publication, "_gate", side_effect=AssertionError("Outcome gate called")):
                with self.assertRaisesRegex(ValueError, "fingerprint"):
                    self.build()

    def test_accounting_memory_omission_does_not_hide_trace_memory(self):
        self.assertNotIn("resources_before", self.accounting["costs"])
        result = self.build()
        self.assertEqual(result["costs"]["memory_by_model_phase"][0]["lifetime_peak_rss_mib"], 125.)
        self.assertEqual(result["costs"]["memory_by_model_phase"][0]["sampled_gpu_used_mib"], 456.)

    def test_missing_runtime_does_not_claim_running_or_passed(self):
        with tempfile.TemporaryDirectory() as directory:
            empty = Path(directory).resolve()
            result = publication.build_publication(empty, empty,
                expected_manifest_fingerprint=self.manifest_id,
                expected_comparison_fingerprint=self.plan["fingerprint"], expected_session_id=self.session)
            self.assertEqual(result["status"], "unknown")
            self.assertFalse(result["audit"]["complete"])
            self.assertEqual(result["progress"], {})

    def test_publication_is_read_only(self):
        before = {str(p.relative_to(self.root)): publication._file_hash(p)
                  for p in self.root.rglob("*") if p.is_file()}
        self.build()
        after = {str(p.relative_to(self.root)): publication._file_hash(p)
                 for p in self.root.rglob("*") if p.is_file()}
        self.assertEqual(after, before)

    def test_linked_or_escaped_evidence_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Escaped"):
            publication._file(self.output, "../anything")
        path = self.audit / "audit.json"
        old = path.read_bytes(); other = self.root / "linked.json"
        other.write_bytes(old); path.unlink(); path.symlink_to(other)
        try:
            with self.assertRaisesRegex(ValueError, "linked"):
                self.build()
        finally:
            path.unlink(); path.write_bytes(old); other.unlink()


if __name__ == "__main__":
    unittest.main()
