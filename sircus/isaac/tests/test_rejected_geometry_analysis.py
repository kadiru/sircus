"""Synthetic rejected-proposal accounting and durable CPU resume checks."""

import copy
import fcntl
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

SOURCE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOURCE))
import rejected_geometry_analysis as analysis
from continuous_feasibility import decode_action


class RejectedGeometryAnalysisTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.protected = self.root/"original.pt"
        self.protected.write_bytes(b"synthetic model bytes; must never be loaded")
        from test_continuous_geometry_audit import GeometryAuditTests
        self.toy = GeometryAuditTests()
        self.toy.setUp()
        self.toy.bad_action = decode_action(self.toy.cell, [0.,.4,.351,.24,.5061454830783556], "exact")
        cases = []
        for i, classification in enumerate(("unchanged", "adjusted", "rejected", "unchanged")):
            result = self.toy.result(classification)
            result["binding"]["protocol"]["max_normalized_adjustment"] = .10
            if result["repair"]["attempted"]:
                result["repair"]["optimizer"] = dict(termination="synthetic_optimizer")
            result.pop("fingerprint")
            result = analysis.signed(result)
            reason = "setup_transit_incomplete" if i < 2 else "outer_link_length_bound"
            cases.append(dict(case_id="synthetic-"+str(i), cell=self.toy.cell,
                action=result["original_action"], starts=self.toy.starts, seeds=self.toy.starts,
                original_reason=reason, rejection_stage="native_motion_or_settling" if i<2 else "geometry_or_path",
                original_geometry_admitted=i<2, home_path_eligible=True, synthetic_result=result))
        self.inputs = dict(synthetic=True, cases=cases, scene=self.toy.scene, links=self.toy.links,
            expected_counts=dict(requested=4), bindings=dict(synthetic=True),
            protected_before=analysis.success_analysis.snapshot([self.protected]),
            protected_roots=[str(self.root/"frozen-benchmark")])
        self.calls = []

    def evaluate(self, job):
        _, _, case, binding = job
        self.calls.append(case["case_id"])
        return analysis.signed(dict(case_id=case["case_id"], cell_id=case["cell"]["id"],
            original_reason=case["original_reason"], rejection_stage=case["rejection_stage"],
            home_path_eligible=case["home_path_eligible"], result=copy.deepcopy(case["synthetic_result"]),
            analysis_binding=binding, wall_s=.2, cpu_s=.1))

    def run_analysis(self, **kwargs):
        args = dict(workers=1, evaluate=self.evaluate)
        args.update(kwargs)
        return analysis.analyze_cases(self.inputs, self.root/"output", **args)

    def test_aggregate_three_outcomes_keeps_native_failure_distinct(self):
        report = self.run_analysis()
        self.assertEqual(report["counts"], dict(requested=4, unchanged=2, adjusted=1, rejected=1))
        self.assertEqual(report["unchanged_native_rejections"], 1)
        self.assertEqual(report["adjusted_native_rejections"], 1)
        self.assertEqual(report["remaining_geometry_reasons"], dict(outer_link_length_bound=1))
        self.assertFalse(report["native_execution_verified"])
        self.assertFalse(report["capture_replayed"])
        self.assertEqual(len(report["durable_row_sha256"]), 4)

    def test_profile_selects_first_per_reason_and_resume_never_recomputes_saved_rows(self):
        partial = self.run_analysis(profile=True)
        self.assertEqual(partial["status"], "geometry_profile_partial")
        self.assertEqual(self.calls, ["synthetic-0", "synthetic-2"])
        self.calls.clear()
        final = self.run_analysis(resume=True)
        self.assertEqual(self.calls, ["synthetic-1", "synthetic-3"])
        self.assertEqual(final["rows_reused"], 2)
        self.assertEqual([r["case_id"] for r in final["rows"]], [c["case_id"] for c in self.inputs["cases"]])
        self.assertEqual(final["row_worker_cpu_s"], .4)
        self.assertEqual(len(list((self.root/"output/invocations").glob("*.json"))), 2)

    def test_existing_output_requires_explicit_resume_and_completed_output_cannot_reopen(self):
        report = self.run_analysis()
        with self.assertRaisesRegex(ValueError, "explicitly resumed"):
            self.run_analysis()
        from rejected_geometry_audit import audit_analysis
        audit = audit_analysis(report, self.inputs)
        self.assertEqual(audit["status"], "passed_synthetic")
        analysis.write_results(self.root/"output", report, audit)
        with self.assertRaisesRegex(ValueError, "cannot be reopened"):
            self.run_analysis(resume=True)

    def test_changed_source_or_case_binding_cannot_resume(self):
        self.run_analysis(profile=True)
        self.inputs["cases"][1]["action"] = dict(self.inputs["cases"][1]["action"], height_m=.36)
        with self.assertRaisesRegex(ValueError, "manifest changed"):
            self.run_analysis(resume=True)
        self.protected.write_bytes(b"changed original bytes")
        with self.assertRaisesRegex(ValueError, "Protected input"):
            self.run_analysis(resume=True)

    def test_corrupt_completed_record_is_preserved_and_never_silently_recomputed(self):
        self.run_analysis(profile=True)
        path = self.root/"output/rows/000000.json"
        row = json.loads(path.read_bytes())
        row["result"]["classification"] = "rejected"
        path.write_text(json.dumps(row))
        self.calls.clear()
        with self.assertRaisesRegex(ValueError, "fingerprint"):
            self.run_analysis(resume=True)
        self.assertEqual(self.calls, [])
        self.assertEqual(json.loads(path.read_bytes())["result"]["classification"], "rejected")

    def test_nonblocking_writer_lock_prevents_concurrent_output_mutation(self):
        self.run_analysis(profile=True)
        with (self.root/"output/writer.lock").open("a+") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with self.assertRaisesRegex(ValueError, "active writer"):
                self.run_analysis(resume=True)

    def test_outputs_cannot_overlap_original_roots(self):
        output = self.root/"frozen-benchmark/new-analysis"
        with self.assertRaisesRegex(ValueError, "overlaps"):
            analysis.analyze_cases(self.inputs, output, workers=1, evaluate=self.evaluate)
        self.assertFalse(output.exists())

    def test_resume_cannot_follow_redirected_row_directory_or_record(self):
        self.run_analysis(profile=True)
        rows = self.root/"output/rows"
        moved = self.root/"saved-rows"
        rows.rename(moved)
        rows.symlink_to(moved, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "redirected"):
            self.run_analysis(resume=True)
        rows.unlink()
        moved.rename(rows)
        row = rows/"000000.json"
        copied = self.root/"saved-row.json"
        row.rename(copied)
        row.symlink_to(copied)
        with self.assertRaisesRegex(ValueError, "Unexpected"):
            self.run_analysis(resume=True)

    def test_new_record_publish_never_replaces_a_concurrent_destination(self):
        path = self.root/"row.json"
        import os
        original_link = os.link
        def raced_link(pending, destination):
            Path(destination).write_text("preserved concurrent evidence")
            return original_link(pending, destination)
        with patch.object(analysis.os, "link", side_effect=raced_link), self.assertRaises(FileExistsError):
            analysis._write_new(path, dict(synthetic=True))
        self.assertEqual(path.read_text(), "preserved concurrent evidence")

    def test_real_evidence_disallows_custom_geometry_hooks_and_witness_seeds(self):
        real = dict(self.inputs, synthetic=False)
        with self.assertRaisesRegex(ValueError, "cannot be replaced"):
            analysis.analyze_cases(real, self.root/"out", evaluate=self.evaluate)
        self.inputs["cases"][0]["seeds"] = {arm:[.01]*6 for arm in self.toy.starts}
        with self.assertRaisesRegex(ValueError, "Historical IK"):
            self.run_analysis()

    def test_duplicate_cases_unbounded_workers_and_unknown_record_files_fail(self):
        for workers in (0, 5, True):
            with self.subTest(workers=workers), self.assertRaises(ValueError):
                self.run_analysis(workers=workers)
        self.run_analysis(profile=True)
        (self.root/"output/rows/unexpected.json").write_text("{}")
        with self.assertRaisesRegex(ValueError, "Unexpected"):
            self.run_analysis(resume=True)


class RejectedGeometryLoaderTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        scene = json.loads((SOURCE/"scene.json").read_text())
        cell = dict(id="synthetic-train", bucket_xy_m=[.27, -.1], split="train")
        self.success = dict(source=dict(commit="synthetic"), source_bindings=dict(identity="synthetic"),
            scene=scene, cells=[cell], training_totals=dict(rejected=2),
            train_trial_sha256={"trials/train-00000.json.gz":"a"*64})
        self.success_path = self.root/"success.json"
        self.success_path.write_text(json.dumps(self.success))
        rejected = []
        for index, reason in enumerate(("outer_link_length_bound", "setup_transit_incomplete")):
            rejected.append(dict(rejection_key="trials/train-00000.json.gz#"+str(index),
                trial_file="trials/train-00000.json.gz", trial_sha256="a"*64, slot=index,
                cell=cell, action=decode_action(cell, [0.,.4,.35,.24,.5], "case-"+str(index)),
                original_reason=reason, rejection_stage="geometry_or_path" if index==0 else "native_motion_or_settling",
                home_path_eligible=True, qualification=dict(accepted=index==1), path_proxy=dict(accepted=index==1),
                recorded_starts_rad={arm:[0.]*6 for arm in ("Arm10","Arm14")}))
        self.export = dict(schema="sircus-seed22-paid-train-rejections-v1", seed=20261122,
            rejected_count=2, rejection_reasons={reason:1 for reason in ("outer_link_length_bound","setup_transit_incomplete")},
            evaluated_trial_files_read=0, new_native_shots=0, optimizer_updates=0,
            rejections=rejected, **self.success)
        self.success["fingerprint"] = analysis.success_analysis.TRAIN_EXPORT_FINGERPRINT
        self.success_path.write_text(json.dumps(self.success))
        self.export.update(successful_train_export_sha256=analysis.success_analysis.TRAIN_EXPORT_SHA256,
            successful_train_export_fingerprint=self.success["fingerprint"],
            rejection_stages=dict(geometry_or_path=1, native_motion_or_settling=1))
        self.path = self.root/"rejected.json"
        self.base = dict(scene=scene, links={}, protected_before=analysis.success_analysis.snapshot([self.success_path]),
            protected_roots=[str(self.root/"benchmark")], bindings=dict(synthetic=True))

    def write(self):
        self.path.write_text(json.dumps(analysis.signed(self.export)))
        return analysis.digest(self.path)

    def load(self, sha=None):
        sha = self.write() if sha is None else sha
        with patch.object(analysis, "REJECTED_COUNT", 2), \
             patch.object(analysis, "EXPECTED_REASONS", self.export["rejection_reasons"]), \
             patch.object(analysis, "REJECTED_EXPORT_SHA256", sha), \
             patch.object(analysis, "REJECTED_EXPORT_FINGERPRINT", json.loads(self.path.read_bytes())["fingerprint"]), \
             patch.object(analysis.success_analysis, "load_inputs", return_value=self.base):
            return analysis.load_inputs(self.path, sha, self.success_path,
                self.root/"benchmark", self.root/"assets", self.root/"published.json")

    def test_loader_keeps_actual_starts_and_old_motion_causes_without_endpoint_lookup(self):
        self.export["rejections"][0]["qualification"] = None
        inputs = self.load()
        self.assertEqual(len(inputs["cases"]), 2)
        self.assertTrue(all(c["starts"] == c["seeds"] for c in inputs["cases"]))
        self.assertTrue(all("qualification" not in c for c in inputs["cases"]))
        self.assertEqual(inputs["cases"][1]["rejection_stage"], "native_motion_or_settling")
        self.assertEqual(inputs["train_trial_sha256"], self.success["train_trial_sha256"])

    def test_changed_raw_export_and_case_geometry_fail_closed(self):
        self.write()
        with self.assertRaisesRegex(ValueError, "SHA"):
            self.load("0"*64)
        self.export["rejections"][0]["action"]["board_x_m"] += .01
        with self.assertRaisesRegex(ValueError, "geometry changed"):
            self.load()

    def test_native_rejection_must_have_original_admitted_path(self):
        self.export["rejections"][1]["path_proxy"]["accepted"] = False
        with self.assertRaisesRegex(ValueError, "original admitted geometry"):
            self.load()


if __name__ == "__main__":
    unittest.main()
