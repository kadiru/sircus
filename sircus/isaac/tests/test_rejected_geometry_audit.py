"""Independent rejection audit tests; chains and outcomes are synthetic only."""

from collections import Counter
import copy
import hashlib
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch


SOURCE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOURCE))
import rejected_geometry_audit as subject
from continuous_feasibility import canonical_hash, decode_action


def sign(value):
    result = copy.deepcopy(value)
    result.pop("fingerprint", None)
    result["fingerprint"] = canonical_hash(result)
    return result


class RejectedGeometryAuditTests(unittest.TestCase):
    def setUp(self):
        from test_continuous_geometry_audit import GeometryAuditTests
        self.toy = GeometryAuditTests()
        self.toy.setUp()
        self.toy.bad_action = decode_action(self.toy.cell,
            [0., .41, .35, .24, self.toy.action["tilt_rad"]], "exact")

    def population(self):
        rows, cases = [], []
        reasons = ("setup_transit_incomplete", "endpoint_ik_unresolved_Arm10", "outer_link_length_bound")
        for i, (classification, reason) in enumerate(zip(subject.CLASSIFICATIONS, reasons)):
            result = self.toy.result(classification)
            result["binding"]["protocol"]["max_normalized_adjustment"] = .10
            if result["repair"]["attempted"]:
                result["repair"]["optimizer"] = dict(termination="optimizer_returned", nfev=1,
                                                       success=classification == "adjusted")
            result = sign(result)
            stage = "native_motion_or_settling" if reason in subject.NATIVE_REASONS else "geometry_or_path"
            identifier = "synthetic-"+str(i)
            case = dict(case_id=identifier, cell=self.toy.cell, action=result["original_action"],
                        starts=self.toy.starts, seeds=self.toy.starts, original_reason=reason,
                        rejection_stage=stage, home_path_eligible=True,
                        original_geometry_admitted=stage == "native_motion_or_settling")
            cases.append(copy.deepcopy(case))
            rows.append(sign(dict(case_id=identifier, cell_id=self.toy.cell["id"],
                                  original_reason=reason, rejection_stage=stage,
                                  home_path_eligible=True, result=result, wall_s=0., cpu_s=0.)))
        by_original = {reason:dict(requested=1, **{key:int(key == classification)
                         for key in subject.CLASSIFICATIONS})
                       for reason, classification in zip(reasons, subject.CLASSIFICATIONS)}
        inputs = dict(synthetic=True, cases=cases, scene=self.toy.scene, links=self.toy.links,
                      expected_counts=dict(requested=3), bindings=dict(synthetic=True),
                      protected_before={"synthetic-input": "0"*64}, protected_roots=[])
        report = sign(dict(schema=subject.ANALYSIS_VERSION, seed=20261122, synthetic=True,
                    rows=rows, counts=dict(requested=3, unchanged=1, adjusted=1, rejected=1),
                    by_original_reason=by_original,
                    first_geometry_reasons={"admitted":1, "outer_link_length_bound":2},
                    remaining_geometry_reasons={"outer_link_length_bound":1},
                    repair_terminations={"not_attempted":1, "optimizer_returned":2},
                    unchanged_native_rejections=1, adjusted_native_rejections=0,
                    bindings=inputs["bindings"], protected_before=inputs["protected_before"],
                    protected_after=inputs["protected_before"], protected_roots=[],
                    optimizer_updates=0, new_native_shots=0, held_out_shot_files_read=0,
                    candidate_bank_read=False, model_loaded=False, release_authorized=False,
                    capture_replayed=False, native_execution_verified=False))
        return report, inputs

    def check(self, report, inputs, **kwargs):
        return subject.audit_analysis(report, inputs, verify_files=False, **kwargs)

    def test_all_three_geometry_results_verify_without_native_success_claim(self):
        report, inputs = self.population()
        audited = self.check(report, inputs)
        self.assertEqual(audited["status"], "passed_synthetic")
        self.assertEqual(audited["counts"], dict(requested=3, unchanged=1, adjusted=1, rejected=1))
        self.assertEqual(audited["unchanged_native_rejections"], 1)
        self.assertFalse(audited["native_execution_verified"])
        self.assertFalse(audited["capture_replayed"])
        self.assertFalse(audited["release_authorized"])

    def test_analysis_and_row_fingerprints_are_mandatory(self):
        report, inputs = self.population()
        report["counts"]["adjusted"] = 99
        with self.assertRaisesRegex(ValueError, "Analysis fingerprint"):
            self.check(report, inputs)
        report, inputs = self.population()
        report["rows"][0]["original_reason"] = "forged"
        with self.assertRaisesRegex(ValueError, "Row fingerprint"):
            self.check(sign(report), inputs)

    def test_omissions_duplicates_and_reordering_fail(self):
        mutators = (lambda x: x["rows"].pop(), lambda x: x["rows"].reverse(),
                    lambda x: x["rows"].__setitem__(1, x["rows"][0]))
        for mutate in mutators:
            report, inputs = self.population(); mutate(report)
            with self.subTest(mutate=mutate), self.assertRaises(ValueError):
                self.check(sign(report), inputs)
        report, inputs = self.population()
        inputs["cases"][1] = inputs["cases"][0]
        with self.assertRaisesRegex(ValueError, "duplicate"):
            self.check(report, inputs)

    def test_original_native_labels_cannot_be_reclassified_as_geometry_failures(self):
        report, inputs = self.population()
        inputs["cases"][0]["rejection_stage"] = "geometry_or_path"
        with self.assertRaisesRegex(ValueError, "stage"):
            self.check(report, inputs)
        report, inputs = self.population()
        report["rows"][0]["original_reason"] = "endpoint_ik_unresolved_Arm10"
        report["rows"][0] = sign(report["rows"][0])
        with self.assertRaisesRegex(ValueError, "identity"):
            self.check(sign(report), inputs)
        report, inputs = self.population()
        inputs["cases"][0]["original_geometry_admitted"] = False
        with self.assertRaisesRegex(ValueError, "provenance"):
            self.check(report, inputs)

    def test_original_actual_starts_are_required_without_historical_witnesses(self):
        report, inputs = self.population()
        inputs["cases"][1]["seeds"] = copy.deepcopy(self.toy.starts)
        inputs["cases"][1]["seeds"]["Arm10"][0] += .01
        with self.assertRaisesRegex(ValueError, "historical endpoint"):
            self.check(report, inputs)
        report, inputs = self.population()
        inputs["cases"][0]["starts"] = copy.deepcopy(self.toy.starts)
        inputs["cases"][0]["starts"]["Arm14"][0] += .01
        with self.assertRaises(ValueError):
            self.check(report, inputs)

    def test_same_default_repair_budgets_are_enforced(self):
        for key, value in (("enabled", False), ("max_nfev", 25),
                           ("max_residual_evaluations", 513), ("max_normalized_adjustment", .11)):
            report, inputs = self.population()
            result = report["rows"][1]["result"]
            result["binding"]["protocol"][key] = value
            report["rows"][1]["result"] = sign(result)
            report["rows"][1] = sign(report["rows"][1])
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "defaults"):
                self.check(sign(report), inputs)

    def test_forged_accepted_repair_cannot_bypass_final_geometry(self):
        report, inputs = self.population()
        result = report["rows"][2]["result"]
        result.update(classification="adjusted", accepted=True, action=self.toy.action,
                      final_preflight=result["original_preflight"])
        report["rows"][2]["result"] = sign(result)
        report["rows"][2] = sign(report["rows"][2])
        with self.assertRaises(ValueError):
            self.check(sign(report), inputs)

    def test_aggregate_counts_reasons_and_native_subtotals_are_recomputed(self):
        mutations = [lambda x:x["counts"].update(adjusted=2),
                     lambda x:x.update(by_original_reason={}),
                     lambda x:x.update(first_geometry_reasons={}),
                     lambda x:x.update(remaining_geometry_reasons={}),
                     lambda x:x.update(repair_terminations={}),
                     lambda x:x.update(unchanged_native_rejections=0),
                     lambda x:x.update(adjusted_native_rejections=1)]
        for mutate in mutations:
            report, inputs = self.population(); mutate(report)
            with self.subTest(mutate=mutate), self.assertRaisesRegex(ValueError, "aggregate"):
                self.check(sign(report), inputs)

    def test_geometry_pass_cannot_claim_native_execution_or_captures(self):
        for key, value in (("capture_replayed", True), ("native_execution_verified", True),
                           ("model_loaded", True), ("candidate_bank_read", True),
                           ("new_native_shots", 1), ("optimizer_updates", 1),
                           ("held_out_shot_files_read", 1), ("release_authorized", True)):
            report, inputs = self.population(); report[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.check(sign(report), inputs)

    def test_synthetic_population_cannot_pass_as_full_real_historical_evidence(self):
        report, inputs = self.population()
        report["synthetic"] = False
        with self.assertRaisesRegex(ValueError, "mode"):
            self.check(sign(report), inputs)
        inputs["synthetic"] = False
        with self.assertRaisesRegex(ValueError, "skip"):
            self.check(sign(report), inputs)

    def test_current_protected_inputs_and_original_roots_are_checked(self):
        report, inputs = self.population()
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)/"synthetic-parent.bin"
            path.write_bytes(b"unchanged synthetic bytes")
            identities = {str(path):hashlib.sha256(path.read_bytes()).hexdigest()}
            inputs["protected_before"] = identities
            report["protected_before"] = report["protected_after"] = identities
            self.assertEqual(subject.audit_analysis(sign(report), inputs)["status"], "passed_synthetic")
            path.write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "Protected input bytes"):
                subject.audit_analysis(sign(report), inputs)
        report, inputs = self.population()
        report["protected_roots"] = ["/changed-root"]
        with self.assertRaisesRegex(ValueError, "roots"):
            self.check(sign(report), inputs)

    def test_frozen_layer_or_original_auditor_cannot_be_replaced(self):
        report, inputs = self.population()
        with patch.dict(subject.FROZEN_SOURCE_SHA256, {"continuous_geometry_layer.py":"0"*64}):
            with self.assertRaisesRegex(ValueError, "frozen geometry source"):
                self.check(report, inputs)

    def test_reused_durable_row_has_the_same_independent_geometry_checks(self):
        report, inputs = self.population()
        for row, case in zip(report["rows"], inputs["cases"]):
            self.assertEqual(subject.audit_row(row, case, inputs), row["result"]["classification"])
        row = copy.deepcopy(report["rows"][0]); row["home_path_eligible"] = False
        with self.assertRaisesRegex(ValueError, "identity"):
            subject.audit_row(sign(row), inputs["cases"][0], inputs)

    def test_row_source_input_manifest_cannot_be_changed_on_resume(self):
        report, inputs = self.population()
        manifest = subject._manifest_fingerprint(inputs)
        report["analysis_manifest_fingerprint"] = manifest
        for row in report["rows"]:
            row["analysis_binding"] = manifest
            row.update(sign(row))
        self.assertEqual(self.check(sign(report), inputs)["status"], "passed_synthetic")
        report["rows"][0]["analysis_binding"] = "0"*64
        report["rows"][0] = sign(report["rows"][0])
        with self.assertRaisesRegex(ValueError, "manifest"):
            self.check(sign(report), inputs)


if __name__ == "__main__":
    unittest.main()
