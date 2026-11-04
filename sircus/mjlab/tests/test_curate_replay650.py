"""Publication curation fails closed; synthetic files, no native dependencies."""

import csv
import json
from pathlib import Path
import tempfile
import unittest

from sircus.mjlab import curate_replay650 as export


def fixture(base, opened):
    study = base / "replay650-fixture"
    study.mkdir()
    names = [("positive-control", "positive_control", .3868), ("candidate-650", "candidate", .3868),
             ("negative-control", "negative_control", .3868)]
    names += [("window-gap-" + str(round(g * 10000)), "window", g) for g in export.GAPS]
    groups = []
    for ident, role, gap in names:
        attempts = 0 if role == "window" and not opened else 3
        captures = 0 if role == "negative_control" else attempts
        values = [.4708, -.2778, .38, 28., gap]
        groups.append(dict(id=ident, role=role, source_record_id="source-"+ident,
            source_record_sha256="a"*64, values=values, board_xy_m=values[:2], bucket_xy_m=[-.03, -.69],
            release_center_height_m=.38+gap+.0375, distance_m=.65, eligible=True,
            sensitivity=dict(paired_attempts=attempts, capture_agreement=attempts,
                             nominal_position_delta_350ms_max_m=.001 if attempts else None),
            by_substeps={str(s):dict(attempts=attempts, native_releases=attempts, captures=captures,
                invalid=0, clean=captures, whole_entries=captures, max_retention_s=1. if captures else 0.) for s in export.STEPS}))
    gate = dict(status="passed" if opened else "failed", positive_control=True, negative_control=True,
                candidate=opened, physics_valid=True, oracle_agreement=True, max_drift_m=.001, window_opened=opened)
    settings = dict(substeps=list(export.STEPS), gaps_m=list(export.GAPS), repeats_per_group=3, max_requests=48)
    spec = dict(version=export.STUDY_VERSION, settings=settings, source_sha256={"source/module.py":"b"*64},
                asset_sha256={"assets/arm.urdf":"c"*64})
    spec["fingerprint"] = export.digest(spec)
    export.write_json(study/"contract.json", spec)
    # A raw trajectory and a private field are intentionally excluded from the
    # publication output, although the audit binds their source report bytes.
    report = dict(version=export.STUDY_VERSION, status="completed", phase="completed", settings=settings,
                  contract_fingerprint=spec["fingerprint"], groups=groups, gate=gate,
                  trace_examples=[dict(position_m=[[12, 34, 56]], q10=[1]*6)],
                  private_runtime_path="/home/private/runtime", secret_extra="DO-NOT-EXPORT")
    report["counts"] = {key:sum(g["by_substeps"][str(s)][key] for g in groups for s in export.STEPS)
                        for key in export.STAT_FIELDS[:-1]}
    report["counts"]["max_retention_s"] = 1.
    export.write_json(study/"records.jsonl", {"raw": "fixture bytes, never exported"})
    manifest = {p.name:export.sha(p) for p in study.iterdir()}
    export.write_json(study/"evidence-manifest.json", manifest)
    report["evidence_sha256"] = manifest
    export.write_json(study/"report.json", report)
    audit = dict(version=export.AUDIT_VERSION, status="passed", checks_count=1, auditor_sha256="d"*64,
                 checks=[dict(label="Synthetic byte-binding control", passed=True)], failures=[],
                 contract_fingerprint=spec["fingerprint"], report_sha256=export.sha(study/"report.json"),
                 evidence_manifest_sha256=export.sha(study/"evidence-manifest.json"), gate=gate,
                 requested_trials=report["counts"]["attempts"], native_releases=report["counts"]["native_releases"])
    export.write_json(study/"audit-replay650.json", audit)
    return study


def rewrite_report(study, edit):
    report = export.read(study/"report.json")
    edit(report)
    export.write_json(study/"report.json", report)
    audit = export.read(study/"audit-replay650.json")
    audit["report_sha256"] = export.sha(study/"report.json")
    export.write_json(study/"audit-replay650.json", audit)


class CurationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)

    def run_export(self, study):
        return export.curate(study, study/"audit-replay650.json", self.base/"completed")

    def test_closed_gate_retains_unexecuted_declared_window_and_privacy(self):
        summary = self.run_export(fixture(self.base, False))
        self.assertEqual(summary["counts"]["attempts"], 18)
        self.assertEqual(summary["window"], dict(status="skipped_gate_closed", opened=False, requested_trials=0))
        self.assertEqual(len(summary["groups"]), 8)
        with (self.base/"completed/configurations.csv").open() as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(len(rows), 16)
        self.assertEqual(sum(r["status"] == "skipped_gate_closed" for r in rows), 10)
        all_text = "\n".join(p.read_text() for p in (self.base/"completed").iterdir())
        for forbidden in ("q10", "trajectory", "position_m", "/home/", "DO-NOT-EXPORT", "private_runtime_path"):
            self.assertNotIn(forbidden, all_text)
        manifest = export.read(self.base/"completed/checksums.json")
        self.assertEqual(set(manifest), {"summary.json", "configurations.csv"})
        for name, expected in manifest.items():
            self.assertEqual(export.sha(self.base/"completed"/name), expected)

    def test_open_gate_exports_all_48_requested_denominators(self):
        summary = self.run_export(fixture(self.base, True))
        self.assertEqual(summary["counts"]["attempts"], 48)
        self.assertEqual(summary["window"]["requested_trials"], 30)
        self.assertEqual(summary["window"]["status"], "completed")
        self.assertTrue(all(v["status"] == "executed_nominal" for g in summary["groups"] for v in g["by_substeps"].values()))

    def test_empty_or_failed_audit_checks_rejected_before_output(self):
        study = fixture(self.base, False)
        original = export.read(study/"audit-replay650.json")
        for checks, count in (([], 0), ([dict(label="failed", passed=False)], 1),
                              ([dict(label="passed", passed=True)], 2)):
            audit = dict(original, checks=checks, checks_count=count)
            export.write_json(study/"audit-replay650.json", audit)
            with self.subTest(checks=checks), self.assertRaises(ValueError):
                self.run_export(study)
        self.assertFalse((self.base/"completed").exists())

    def test_changed_report_and_changed_raw_evidence_rejected(self):
        study = fixture(self.base, False)
        (study/"report.json").write_text((study/"report.json").read_text()+"\n")
        with self.assertRaisesRegex(ValueError, "report bytes"):
            self.run_export(study)
        audit = export.read(study/"audit-replay650.json")
        audit["report_sha256"] = export.sha(study/"report.json")
        export.write_json(study/"audit-replay650.json", audit)
        (study/"records.jsonl").write_text("changed")
        with self.assertRaisesRegex(ValueError, "Changed evidence"):
            self.run_export(study)

    def test_running_report_and_disagreeing_gate_rejected(self):
        study = fixture(self.base, False)
        rewrite_report(study, lambda r:r.update(status="running"))
        with self.assertRaisesRegex(ValueError, "not completed"):
            self.run_export(study)
        rewrite_report(study, lambda r:r.update(status="completed", gate=dict(r["gate"], candidate=True)))
        with self.assertRaisesRegex(ValueError, "gate differs"):
            self.run_export(study)

    def test_skipped_window_cannot_report_attempts(self):
        study = fixture(self.base, False)
        rewrite_report(study, lambda r:r["groups"][-1]["by_substeps"]["64"].update(attempts=3))
        with self.assertRaisesRegex(ValueError, "conditional denominator"):
            self.run_export(study)

    def test_existing_output_and_frozen_evidence_output_rejected(self):
        study = fixture(self.base, False)
        with self.assertRaisesRegex(ValueError, "outside frozen"):
            export.curate(study, study/"audit-replay650.json", study/"curated")
        self.run_export(study)
        with self.assertRaisesRegex(ValueError, "overwrite"):
            self.run_export(study)

    def test_manifest_cannot_read_outside_study(self):
        study = fixture(self.base, False)
        (self.base/"outside").write_text("outside")
        manifest = export.read(study/"evidence-manifest.json")
        manifest["../outside"] = export.sha(self.base/"outside")
        export.write_json(study/"evidence-manifest.json", manifest)
        rewrite_report(study, lambda r:r.update(evidence_sha256=manifest))
        audit = export.read(study/"audit-replay650.json")
        audit["evidence_manifest_sha256"] = export.sha(study/"evidence-manifest.json")
        export.write_json(study/"audit-replay650.json", audit)
        with self.assertRaisesRegex(ValueError, "Unsafe provenance"):
            self.run_export(study)


if __name__ == "__main__":
    unittest.main()
