"""Synthetic controller fixtures test export boundaries, never native evidence."""

import copy
import csv
import json
from pathlib import Path
import unittest

from sircus.mjlab import curate_clean650 as export
from sircus.mjlab.tests import test_clean650_study as controller_fixture


class CleanExportTests(unittest.TestCase):
    def fixture(self, blocked=False,profile='coarse-v1'):
        controller = controller_fixture.CleanControllerTests()
        self.addCleanup(controller.doCleanups)
        study, report, _, _ = controller.run_fixture(fail_initial=blocked,profile=profile)
        self.study = study
        self.report = report
        self.audit_path = study/"audit-clean650.json"
        self.audit = dict(version=export.VERSION_PAIRS[report['version']], status="passed", checks_count=1,
                          checks=[dict(label="Synthetic export fixture; not a native audit", passed=True)], failures=[],
                          auditor_sha256="a"*64, auditor_identity=dict(dependencies_sha256={"synthetic-checker.py":"b"*64}))
        self.bind()
        return study

    def bind(self):
        # Explicitly re-sign synthetic fixtures to reach semantic guards. These
        # signatures never constitute real independent native verification.
        spec = export.read(self.study/"contract.json")
        manifest = {str(p.relative_to(self.study)):export.sha(p) for p in self.study.rglob("*") if p.is_file()
                    and p.name not in ("report.json", "evidence-manifest.json") and not p.name.startswith("audit")}
        export.write_json(self.study/"evidence-manifest.json", manifest)
        self.report["evidence_sha256"] = manifest
        export.write_json(self.study/"report.json", self.report)
        self.audit.update(report_sha256=export.sha(self.study/"report.json"), evidence_manifest_sha256=export.sha(self.study/"evidence-manifest.json"),
                          contract_fingerprint=spec["fingerprint"], requested_trials=self.report["counts"]["attempts"],
                          native_releases=self.report["counts"]["native_releases"], gate=copy.deepcopy(self.report["gate"]))
        export.write_json(self.audit_path, self.audit)

    def run_export(self, name="publication"):
        return export.curate(self.study, self.audit_path, self.study.parent/name)

    def test_full_finite_controller_exports_all_requests_lf_and_checksums(self):
        self.fixture()
        summary = self.run_export()
        output = self.study.parent/"publication"
        self.assertEqual(summary["counts"]["attempts"], 496)
        self.assertEqual(summary["counts"]["phase_counts"], dict(controls_pre=12, coarse=361, refinement=81, verification=30, controls_post=12))
        self.assertEqual(summary["gate"]["qualified_settings"], 5)
        self.assertEqual(summary["settings"]["verification_dedup_decimals"], 9)
        self.assertEqual(len(summary["verification"]), 5)
        self.assertEqual(len(summary["controls"]), 8)
        content = (output/"requested-shots.csv").read_bytes()
        self.assertNotIn(b"\r", content)
        with (output/"requested-shots.csv").open() as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(len(rows), 496)
        self.assertEqual(rows[0]["phase"], "controls_pre")
        self.assertEqual(rows[-1]["phase"], "controls_post")
        self.assertEqual(rows[0]["source_record_id"], "historical-positive-control")
        for name, value in export.read(output/"checksums.json").items():
            self.assertEqual(export.sha(output/name), value)

    def test_fine_profile_exports_211_only_with_matching_v2_audit(self):
        self.fixture(profile='fine-v2')
        summary=self.run_export()
        self.assertEqual(summary['study_version'],export.FINE_STUDY_VERSION)
        self.assertEqual(summary['counts']['attempts'],211)
        self.assertEqual(summary['settings']['grid']['tilts_deg'],[27.,27.5,28.,28.5,29.])
        self.audit['version']=export.AUDIT_VERSION;self.bind()
        with self.assertRaisesRegex(ValueError,'Full independent audit'):self.run_export('wrong-pair')

    def test_blocked_controls_export_only_executed_requests_and_no_qualification(self):
        self.fixture(blocked=True)
        summary = self.run_export()
        self.assertEqual(summary["counts"]["attempts"], 12)
        self.assertEqual(summary["gate"]["status"], "blocked_controls")
        self.assertEqual(summary["control_gates"]["controls_post"]["status"], "not_run")
        self.assertEqual(summary["verification"], [])
        self.assertEqual(summary["gate"]["qualified_settings"], 0)
        self.assertEqual(summary["declared_groups"], 363)
        self.assertEqual(summary["executed_groups"], 2)

    def test_rejected_records_remain_denominators_with_unknown_contact_values(self):
        self.fixture(blocked=True)
        path = self.study/"records.jsonl"
        records = [json.loads(line) for line in path.read_text().splitlines()]
        for row in records:
            if row["role"] != "negative_control": continue
            row.update(eligible=False, native_release=False, invalid="strict_endpoint_rejected", scalar_score_audit_passed=False)
            row.pop("rim_band_contact"); row.pop("minimum_mouth_clearance_mm")
        path.write_text("".join(json.dumps(r)+"\n" for r in records))
        for g in self.report["groups"]:
            if g["role"] == "negative_control": g["eligible"] = False
        self.report["counts"].update(export.statistics(records))
        for c in self.report["controls"]:
            chosen = [r for r in records if r["phase"] == c["phase"] and r["group_id"] == c["group_id"] and r["substeps"] == c["substeps"]]
            c["stats"] = export.statistics(chosen)
        self.bind()
        summary = self.run_export()
        self.assertEqual(summary["counts"]["geometry_rejections"], 6)
        self.assertEqual(summary["counts"]["attempts"], 12)
        self.assertEqual(summary["counts"]["native_releases"], 6)
        with (self.study.parent/"publication/requested-shots.csv").open() as stream:
            rejected = [r for r in csv.DictReader(stream) if r["native_release"] == "False"]
        self.assertEqual(len(rejected), 6)
        self.assertTrue(all(r["invalid"] == "strict_endpoint_rejected" and r["rim_band_contact"] == "" for r in rejected))

    def test_only_allowlisted_configuration_and_metrics_leave_evidence(self):
        self.fixture()
        secret = "DO-NOT-PUBLISH-secret-host-path"
        self.report.update(local_path="/private/"+secret, q10=[secret], position_m=[[secret]], credentials=dict(token=secret))
        self.report["settings"]["q14"] = [secret]
        spec = export.read(self.study/"contract.json")
        spec["settings"]["q14"] = [secret]
        unsigned = dict(spec); unsigned.pop("fingerprint")
        spec["fingerprint"] = export.digest(unsigned)
        self.report["contract_fingerprint"] = spec["fingerprint"]
        export.write_json(self.study/"contract.json", spec)
        for value in self.report["groups"]+self.report["top_settings"]+self.report["verification"]+self.report["controls"]:
            value.update(q10=[secret], link_positions14=[[secret]], native_pose=dict(raw=secret))
        self.bind()
        self.run_export()
        for path in (self.study.parent/"publication").iterdir():
            self.assertNotIn(secret, path.read_text())

    def test_missing_empty_failed_or_wrong_version_audit_cannot_export(self):
        self.fixture(blocked=True)
        original = copy.deepcopy(self.audit)
        for edit in (dict(version="wrong"), dict(checks=[], checks_count=0), dict(checks=[dict(label="failed", passed=False)]),
                     dict(checks_count=2), dict(status="failed")):
            self.audit = dict(original, **edit); export.write_json(self.audit_path, self.audit)
            with self.assertRaises(ValueError): self.run_export()
        self.audit_path.unlink()
        with self.assertRaises(OSError): self.run_export()
        self.assertFalse((self.study.parent/"publication").exists())

    def test_report_and_manifest_artifact_tampering_are_detected(self):
        self.fixture(blocked=True)
        original = (self.study/"report.json").read_bytes()
        (self.study/"report.json").write_bytes(original+b" ")
        with self.assertRaisesRegex(ValueError, "Audit/report bytes"): self.run_export()
        (self.study/"report.json").write_bytes(original)
        (self.study/"calibration.json").write_text('{"changed":true}')
        with self.assertRaisesRegex(ValueError, "Changed evidence bytes"): self.run_export()

    def test_escaping_manifest_incomplete_state_and_output_overwrite_fail(self):
        self.fixture(blocked=True)
        self.report["status"] = "running"; self.bind()
        with self.assertRaisesRegex(ValueError, "not completed"): self.run_export()
        self.report["status"] = "completed"; self.bind()
        with self.assertRaisesRegex(ValueError, "outside evidence"): export.curate(self.study, self.audit_path, self.study/"publication")
        self.run_export()
        with self.assertRaisesRegex(ValueError, "New publication directory"): self.run_export()
        manifest = export.read(self.study/"evidence-manifest.json"); manifest["../outside"] = "c"*64
        export.write_json(self.study/"evidence-manifest.json", manifest)
        self.report["evidence_sha256"] = manifest; export.write_json(self.study/"report.json", self.report)
        self.audit.update(report_sha256=export.sha(self.study/"report.json"), evidence_manifest_sha256=export.sha(self.study/"evidence-manifest.json"))
        export.write_json(self.audit_path, self.audit)
        with self.assertRaisesRegex(ValueError, "Unsafe provenance filename"): self.run_export("second")

    def test_non_finite_metrics_or_changed_requested_denominator_fail_closed(self):
        self.fixture()
        self.report["counts"]["attempts"] = 497; self.bind()
        with self.assertRaisesRegex(ValueError, "trial accounting"): self.run_export()
        self.report["counts"]["attempts"] = 496
        self.report["verification"][0]["sensitivity"]["nominal_position_delta_350ms_max_m"] = {"private": "path"}
        self.bind()
        with self.assertRaisesRegex(ValueError, "finite numerical metric"): self.run_export()


if __name__ == "__main__": unittest.main()
