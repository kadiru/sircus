"""Synthetic file fixtures for publication gates; no native simulation runs."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

from sircus.mjlab import prepare_gap_flight_report as publisher


def write(path, value):
    path.write_text(json.dumps(value, sort_keys=True, allow_nan=False)+"\n")


def diagnosis_fixture(root):
    folder = root/"diagnosis"; folder.mkdir()
    proof = dict(status="passed", checks_count=1, checks=[dict(label="synthetic publication fixture", passed=True)])
    write(folder/"checks.json", proof)
    value = dict(version=publisher.DIAGNOSIS_VERSION, status="completed",
        processing=dict(source_sha256=publisher.sha(Path(publisher.__file__).with_name("gap_flight_diagnosis.py"))),
        audit=dict(source_sha256=publisher.sha(folder/"checks.json")),
        source={key+"_sha256": val for key, val in publisher.SOURCE_HASHES.items()},
        requests=[dict(id="request-"+str(i)) for i in range(50)], misses=[{} for _ in range(18)], rejects=[{} for _ in range(8)])
    write(folder/"diagnosis.json", value); write(folder/"summary.json", value)
    write(folder/"checksums.json", {name: publisher.sha(folder/name) for name in ("diagnosis.json", "summary.json", "checks.json")})
    return folder


def correction_fixture(root, status="completed", audit=True):
    study = root/"study"; study.mkdir()
    spec = dict(version=publisher.VERSION, maximum_requests=30, contexts=["12", "24", "48"],
                gap_delta_m=.06, substeps=64, repeats=3, training=False)
    spec["fingerprint"] = publisher.digest(spec)
    write(study/"contract.json", spec)
    write(study/"prediction-plan.json", dict(version=publisher.VERSION, contract_fingerprint=spec["fingerprint"], fixture=True))
    rows = []
    for variant, amount in (("original", 9), ("gap_plus60", 9), ("control_exact", 12)):
        for i in range(amount):
            capture = variant == "gap_plus60" or variant == "control_exact" and i < 6
            rows.append(dict(id=variant+str(i), variant=variant, native_release=True, invalid=None,
                invalid_physics=False, route_success=capture, route_quality=1 if capture else 0))
    (study/"records.jsonl").write_text("".join(json.dumps(r)+"\n" for r in rows))
    (study/"raw-fixture.dat").write_bytes(b"Synthetic publication bytes; not simulation evidence.\n")
    manifest = {p.name: publisher.sha(p) for p in study.iterdir() if p.is_file()}
    write(study/"evidence-manifest.json", manifest)
    report = dict(version=publisher.VERSION, contract_fingerprint=spec["fingerprint"], status=status,
        phase="completed" if status == "completed" else status, plan_sha256=publisher.sha(study/"prediction-plan.json"),
        evidence_sha256=manifest, control_gates={k: dict(status="passed") for k in ("pre", "post")})
    if status == "queued": report["queue"] = dict(wait_s=25)
    write(study/"report.json", report)
    audit_path = root/"audit.json"
    proof = dict(version="sircus-gap-flight-correction-audit-v1", status="passed", failures=[], checks_count=2,
        checks=[dict(label="synthetic proof fixture A", passed=True), dict(label="synthetic proof fixture B", passed=True)],
        auditor_sha256=publisher.sha(Path(publisher.__file__).with_name("gap_flight_correction_verify.py")),
        contract_fingerprint=spec["fingerprint"], report_sha256=publisher.sha(study/"report.json"),
        plan_sha256=publisher.sha(study/"prediction-plan.json"), evidence_manifest_sha256=publisher.sha(study/"evidence-manifest.json"),
        requested_trials=30, control_gates=copy.deepcopy(report["control_gates"]))
    if audit: write(audit_path, proof)
    return study, audit_path, proof


def snapshot(folder):
    return {str(p.relative_to(folder)): p.read_bytes() for p in folder.rglob("*") if p.is_file()}


class GapFlightPublisherTests(unittest.TestCase):
    def test_completed_without_audit_waits_and_does_not_show_outcomes(self):
        with tempfile.TemporaryDirectory() as temp:
            study, audit, _ = correction_fixture(Path(temp), audit=False)
            value = publisher.correction(study, audit)
            self.assertEqual(value["status"], "awaiting_audit")
            self.assertEqual(value["comparisons"], [])
            self.assertIsNone(value["audit"]["passed"])

    def test_full_bound_proof_exposes_exact_nine_plus_nine(self):
        with tempfile.TemporaryDirectory() as temp:
            study, audit, _ = correction_fixture(Path(temp))
            before = snapshot(study)
            value = publisher.correction(study, audit)
            self.assertEqual(value["status"], "passed")
            self.assertEqual([c["counts"]["requested"] for c in value["comparisons"]], [9, 9])
            self.assertEqual([c["counts"]["captures"] for c in value["comparisons"]], [0, 9])
            self.assertEqual(value["audit"]["version"], "sircus-gap-flight-correction-checks-v1")
            self.assertTrue(value["audit"]["full_checks_verified"])
            self.assertEqual(before, snapshot(study))

    def test_each_raw_artifact_tamper_fails_closed(self):
        for name in ("report.json", "prediction-plan.json", "records.jsonl", "raw-fixture.dat", "evidence-manifest.json"):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as temp:
                study, audit, _ = correction_fixture(Path(temp))
                target = study/name
                target.write_bytes(target.read_bytes()+b" ")
                with self.assertRaises(ValueError): publisher.correction(study, audit)

    def test_incomplete_wrong_or_stale_proofs_never_show_success(self):
        changes = (("auditor_sha256", "0"*64), ("report_sha256", "0"*64), ("plan_sha256", "0"*64),
            ("evidence_manifest_sha256", "0"*64), ("contract_fingerprint", "0"*64), ("checks_count", 3),
            ("checks", []), ("failures", ["failed"]), ("status", "pending"), ("requested_trials", 29),
            ("version", "some-other-audit"))
        for key, changed in changes:
            with self.subTest(key=key), tempfile.TemporaryDirectory() as temp:
                study, audit, proof = correction_fixture(Path(temp)); proof[key] = changed; write(audit, proof)
                with self.assertRaises(ValueError): publisher.correction(study, audit)
        with tempfile.TemporaryDirectory() as temp:
            study, audit, proof = correction_fixture(Path(temp)); proof["checks"][1]["passed"] = False; write(audit, proof)
            with self.assertRaises(ValueError): publisher.correction(study, audit)

    def test_negative_control_failure_rejected_even_when_rebound_to_new_hash(self):
        with tempfile.TemporaryDirectory() as temp:
            study, audit, proof = correction_fixture(Path(temp))
            report = publisher.read(study/"report.json")
            report["control_gates"]["post"]["status"] = "failed"
            write(study/"report.json", report)
            proof.update(report_sha256=publisher.sha(study/"report.json"), control_gates=copy.deepcopy(report["control_gates"]))
            write(audit, proof)
            with self.assertRaisesRegex(ValueError, "matched controls"): publisher.correction(study, audit)

    def test_missing_diagnostic_denominator_rejected_with_bound_proof(self):
        with tempfile.TemporaryDirectory() as temp:
            study, audit, proof = correction_fixture(Path(temp))
            rows = [json.loads(line) for line in (study/"records.jsonl").read_text().splitlines()]
            rows[0]["variant"] = "control_exact"
            (study/"records.jsonl").write_text("".join(json.dumps(r)+"\n" for r in rows))
            manifest = publisher.read(study/"evidence-manifest.json"); manifest["records.jsonl"] = publisher.sha(study/"records.jsonl")
            write(study/"evidence-manifest.json", manifest)
            report = publisher.read(study/"report.json"); report["evidence_sha256"] = manifest; write(study/"report.json", report)
            proof.update(report_sha256=publisher.sha(study/"report.json"), evidence_manifest_sha256=publisher.sha(study/"evidence-manifest.json"))
            write(audit, proof)
            with self.assertRaisesRegex(ValueError, "18 diagnostic slots"): publisher.correction(study, audit)

    def test_prepared_queued_running_have_no_fabricated_outcomes(self):
        for raw, expected in (("prepared", "pending"), ("queued", "queued"), ("running", "running")):
            with self.subTest(raw=raw), tempfile.TemporaryDirectory() as temp:
                study, audit, _ = correction_fixture(Path(temp), status=raw, audit=False)
                value = publisher.correction(study, audit)
                self.assertEqual(value["status"], expected)
                self.assertEqual(value["comparisons"], [])
                self.assertIsNone(value["audit"]["passed"])
                if raw == "queued": self.assertEqual(value["queue"]["wait_s"], 25)

    def test_failed_independent_proof_shows_failure_without_outcomes(self):
        with tempfile.TemporaryDirectory() as temp:
            study, audit, proof = correction_fixture(Path(temp)); proof["status"] = "failed"; write(audit, proof)
            value = publisher.correction(study, audit)
            self.assertEqual(value["status"], "failed")
            self.assertFalse(value["audit"]["passed"])
            self.assertEqual(value["comparisons"], [])

    def test_pipeline_timeout_or_abnormal_exit_surfaces_without_mutating_study(self):
        for raw, failure in (("completed", "audit_failed"), ("running", "failed")):
            with self.subTest(raw=raw), tempfile.TemporaryDirectory() as temp:
                root = Path(temp); folder = diagnosis_fixture(root)
                study, audit, _ = correction_fixture(root, status=raw, audit=False)
                before = snapshot(study)
                display = root/"display.json"; original = dict(other_study=dict(status="passed", counts=[1, 2, 3]), unrelated="preserve")
                write(display, original)
                result = publisher.publish(display, folder, study, audit, pipeline=dict(status=failure, error="Own worker stopped"))
                value = result[publisher.CORRECTION_KEY]
                self.assertEqual(value["status"], "failed")
                self.assertEqual(value["error"], "Own worker stopped")
                self.assertEqual(value["comparisons"], [])
                self.assertFalse(value["audit"]["passed"])
                self.assertEqual(result["other_study"], original["other_study"])
                self.assertEqual(result["unrelated"], original["unrelated"])
                self.assertEqual(snapshot(study), before)
                self.assertEqual(result, publisher.read(display))

    def test_diagnosis_file_tampering_rejected_before_publishing(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); folder = diagnosis_fixture(root); display = root/"display.json"
            write(display, dict(untouched=True)); original = display.read_bytes()
            path = folder/"diagnosis.json"; path.write_bytes(path.read_bytes()+b" ")
            with self.assertRaises(ValueError): publisher.publish(display, folder)
            self.assertEqual(display.read_bytes(), original)


if __name__ == "__main__": unittest.main()
