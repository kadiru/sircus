"""Synthetic CPU controls for the independent30-request correction audit."""
import copy
import unittest

from sircus.mjlab import gap_flight_correction_verify as audit
from sircus.mjlab.benchmark_verify import Checks, EvidenceError
from sircus.mjlab.tests.test_policy64_diagnosis_native import source_row


def parent_rows():
    result = []
    for i, context in enumerate(audit.CONTEXTS):
        row = source_row("consistent-"+context, context=context, method="consistent", role="test", capture=False)
        row.update(values=[.58+i*.01, -.04, .38, 39., .056+i*.007], phase="test", observation=[0.]*9)
        result.append(row)
    for role in ("positive_control", "negative_control"):
        result.append(source_row("controls_pre-"+role+"-r0", context=role, role=role, capture=role == "positive_control"))
    return result


def plan():
    from sircus.mjlab.gap_flight_correction import make_plan
    return make_plan(parent_rows())


class GapCorrectionAuditTests(unittest.TestCase):
    def test_independent_plan_agrees_with_worker_exactly30(self):
        rows = parent_rows(); before = copy.deepcopy(rows)
        from sircus.mjlab.gap_flight_correction import make_plan
        actual = make_plan(rows)
        expected = audit.audit_plan(actual, rows, Checks())
        self.assertEqual(len(actual), 30)
        self.assertEqual(rows, before)
        self.assertEqual([r["phase"] for r in actual], ["controls_pre"]*6+["diagnostic"]*18+["controls_post"]*6)
        for produced, independent in zip(actual, expected):
            self.assertEqual({key: produced[key] for key in independent}, independent)
        self.assertEqual(sum(r["kind"] == "coordinate_swap" for r in actual), 9)
        self.assertEqual(sum(r["role"] == "positive_control" for r in actual), 6)
        self.assertEqual(sum(r["role"] == "negative_control" for r in actual), 6)

    def test_only_gap_changes_by60mm_and_new_joints_not_inherited(self):
        rows = {r["id"]: r for r in parent_rows()}
        for request in plan():
            source = rows[request["parent_record_id"]]
            self.assertEqual(request["values"][:4], source["values"][:4])
            self.assertEqual(request["bucket_xy_m"], source["bucket_xy_m"])
            self.assertEqual(request["parent_record_sha256"], audit.digest(source))
            if request["variant"] == "gap_plus60":
                self.assertEqual(request["values"][4], source["values"][4]+.06)
                self.assertEqual(request["swapped_coordinates"], [4])
                self.assertNotIn("q10", request)
                self.assertNotIn("q14", request)
                self.assertNotIn("connection_file", request)
            else:
                self.assertEqual(request["values"], source["values"])
                self.assertEqual(request["q10"], source["q10"])
                self.assertEqual(request["q14"], source["q14"])

    def test_reject_missing_duplicate_reordered_or_extra_requests(self):
        for mutation in ("missing", "duplicate", "order", "extra"):
            changed = plan()
            if mutation == "missing": changed.pop()
            elif mutation == "duplicate": changed[-1]["id"] = changed[-2]["id"]
            elif mutation == "order": changed[6], changed[7] = changed[7], changed[6]
            else: changed.append(copy.deepcopy(changed[-1]))
            with self.assertRaises(EvidenceError): audit.audit_plan(changed, parent_rows(), Checks())

    def test_reject_hidden_coordinate_or_physics_or_proof_changes(self):
        for mutation in ("height", "tilt", "gap", "bucket", "branch", "proof", "substeps", "source"):
            changed = plan(); row = changed[9]
            if mutation == "height": row["values"][2] += .001
            elif mutation == "tilt": row["values"][3] += .1
            elif mutation == "gap": row["values"][4] += .001
            elif mutation == "bucket": row["bucket_xy_m"][0] += .001
            elif mutation == "branch": row["q10"] = [0.]*6
            elif mutation == "proof": row["connection_file"] = "old-proof.json"
            elif mutation == "substeps": row["substeps"] = 128
            else: row["parent_record_sha256"] = "x"*64
            with self.assertRaises(EvidenceError): audit.audit_plan(changed, parent_rows(), Checks())

    def test_reject_duplicate_selected_source_and_missing_source(self):
        for rows in (parent_rows()[1:], parent_rows()+[parent_rows()[0]]):
            with self.assertRaises(EvidenceError): audit.expected_plan_core(rows)

    def test_counts_distinguish_bucket_retention_from_valid_board_route(self):
        rows = []
        for capture, retained, native in ((True, True, True), (False, True, True), (False, False, False)):
            rows.append(dict(native_release=native, route_success=capture, route_quality=2 if capture else 0,
                             success=retained, invalid_physics=False))
        self.assertEqual(audit.native_counts(rows), dict(attempts=3, native_releases=2, captures=1, clean=1, rejected=1, invalid_physics=0))

    def test_controls_require_actual_native_negative_and_positive_repeats(self):
        rows = []
        for role in ("positive_control", "negative_control"):
            for _ in range(3):
                rows.append(dict(role=role, native_release=True, route_success=role == "positive_control",
                    route_quality=1 if role == "positive_control" else 0, whole_ball_entry=role == "positive_control",
                    invalid=None, invalid_physics=False, scalar_score_audit_passed=True))
        self.assertEqual(audit.native_control_gate(rows)["status"], "passed")
        for field, value in (("native_release", False), ("whole_ball_entry", True), ("invalid_physics", True), ("scalar_score_audit_passed", False)):
            changed = copy.deepcopy(rows); changed[-1][field] = value
            self.assertEqual(audit.native_control_gate(changed)["status"], "failed")
        self.assertEqual(audit.native_control_gate(rows[:-1])["status"], "failed")

    def test_booleans_cannot_be_passed_as_numbers(self):
        with self.assertRaises(EvidenceError): audit.same(1, True, Checks(), "explicit")
        with self.assertRaises(EvidenceError): audit.same(dict(a=1, hidden=2), dict(a=1), Checks(), "fields")

    def test_contract_rejects_more_budget_geometry_or_training_change(self):
        requests = plan(); rows = parent_rows()
        parent = dict(scene={"fixture": True}, identity={"score": "frozen"}, packages={}, native_source_sha256={}, fingerprint="a"*64)
        manifest = {"contract.json": "b"*64, "records.jsonl": "c"*64}
        bindings = dict(parent_report_sha256=audit.PARENT_REPORT_SHA, parent_contract_sha256=manifest["contract.json"],
            parent_manifest_sha256=audit.PARENT_MANIFEST_SHA, parent_audit_sha256=audit.PARENT_AUDIT_SHA,
            parent_contract_fingerprint=parent["fingerprint"], parent_rows_sha256=manifest["records.jsonl"], parent_evidence_sha256=manifest,
            parent_records_sha256={r["parent_record_id"]: r["parent_record_sha256"] for r in requests})
        spec = dict(version=audit.STUDY_VERSION, scope="Fixed64 development intervention", substeps=64,
            maximum_requests=30, maximum_acquired_s=3600, contexts=["12", "24", "48"], gap_delta_m=.06, repeats=3,
            training=False, noise=False, parent_bindings=bindings, plan=requests,
            **{k: parent[k] for k in ("scene", "identity", "packages", "native_source_sha256")})
        spec["fingerprint"] = audit.digest(spec)
        report = dict(version=audit.STUDY_VERSION, status="completed", phase="completed", contract_fingerprint=spec["fingerprint"])
        audit.audit_contract(spec, report, parent, rows, manifest, Checks())
        for mutation in ("budget", "gap", "geometry", "training", "substeps"):
            changed = copy.deepcopy(spec)
            if mutation == "budget": changed["maximum_requests"] = 31
            elif mutation == "gap": changed["gap_delta_m"] = .08
            elif mutation == "geometry": changed["scene"]["fixture"] = False
            elif mutation == "training": changed["training"] = True
            else: changed["substeps"] = 128
            changed.pop("fingerprint"); changed["fingerprint"] = audit.digest(changed)
            with self.assertRaises(EvidenceError):
                audit.audit_contract(changed, dict(report, contract_fingerprint=changed["fingerprint"]), parent, rows, manifest, Checks())


if __name__ == "__main__": unittest.main()
