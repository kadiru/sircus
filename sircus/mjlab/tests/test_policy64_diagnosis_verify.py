"""Synthetic CPU-only controls for the post-hoc diagnosis auditor."""
import copy
import unittest

from sircus.mjlab import policy64_diagnosis_verify as audit
from sircus.mjlab.benchmark_verify import Checks, EvidenceError


def rows():
    lost = {4, 7, 13, 17, 21, 26, 43}
    gained = {1, 25, 29, 32, 37}
    both = set(index for index in range(50) if index not in lost|gained)
    both = set(sorted(both)[:13])
    result = []
    for method in ("bc", "ppo"):
        for index in range(50):
            capture = index in both or index in (lost if method == "bc" else gained)
            native = not (method == "ppo" and index in {4, 13, 21, 26, 43})
            result.append(dict(id=f"test-{method}-{'0000' if method=='bc' else '1024'}-policy64-test-{index:02d}", role="test", method=method,
                context_id=f"policy64-test-{index:02d}", bucket_xy_m=[.1+index*.01, -.3], values=[.5, -.1, .3, 28., .4] if method=="bc" else [.51, -.11, .31, 29., .41],
                native_release=native, route_success=capture, route_quality=1 if capture else 0,
                invalid=None if native else "finite_ik_unresolved", invalid_physics=False, board_bounce=native, whole_ball_entry=capture,
                max_retention_s=1. if capture else 0., closest_opening_distance_m=.01 if capture else 2., external_contact_steps=0,
                single_bounce_success=capture, q10=[.1]*6 if native else None, q14=[.2]*6 if native else None))
    return result


def complete_rows():
    from sircus.mjlab.tests.test_policy64_diagnosis_native import source_row
    result = []
    fields = ("source_teacher_id", "source_hint", "effective_source_hint", "context_resolution_file", "context_resolution_sha256",
              "context_home_sha256", "context_layout_eligible", "context_anchor_admitted", "context_hard_reject", "connection_file",
              "connection_sha256", "native_cache_file", "native_cache_sha256", "observation")
    for row in rows():
        template = source_row(row["id"], context=row["context_id"], method=row["method"], role="test", native=row["native_release"], capture=row["route_success"])
        row.update({key: copy.deepcopy(template[key]) for key in fields});row["phase"] = "test";result.append(row)
    result += [source_row("controls_pre-positive_control-r0"), source_row("controls_pre-negative_control-r0", role="negative_control", capture=False)]
    return result


class DiagnosisAuditTests(unittest.TestCase):
    def test_original_fifty_pair_counts_and_all_denominators(self):
        summary = audit.pair_summary(rows())
        self.assertEqual([summary[key] for key in ("layouts", "both_capture", "bc_only", "ppo_only", "neither")], [50, 13, 7, 5, 25])
        self.assertEqual((summary["bc"]["attempts"], summary["bc"]["captures"], summary["ppo"]["attempts"], summary["ppo"]["captures"]), (50, 20, 50, 18))

    def test_pairing_rejects_missing_duplicate_and_moved_buckets(self):
        for mutation in ("missing", "duplicate", "moved"):
            changed = rows()
            if mutation == "missing": changed.pop()
            elif mutation == "duplicate": changed[-1]["context_id"] = changed[-2]["context_id"]
            else: changed[-1]["bucket_xy_m"][0] += .001
            with self.assertRaises(EvidenceError): audit.paired_rows(changed)

    def test_literal_seven_losses_generate84_requests_without_reselecting_outcomes(self):
        original = rows();variants = audit.expected_variants(original)
        self.assertEqual(len(variants), 84)
        self.assertEqual(list(dict.fromkeys(row["context_id"] for row in variants)), list(audit.CONTEXTS))
        for context in audit.CONTEXTS:
            group = [row for row in variants if row["context_id"] == context]
            self.assertEqual([row["variant"] for row in group[:6]], ["bc_exact"]*3+["ppo_exact"]*3)
            self.assertEqual([row["variant"] for row in group[6:]], [name for name, _ in audit.COORDINATES])
        changed = copy.deepcopy(original)
        next(row for row in changed if row["method"] == "bc" and row["context_id"] == audit.CONTEXTS[0])["route_success"] = False
        with self.assertRaisesRegex(EvidenceError, "seven"):
            audit.expected_variants(changed)

    def test_coordinate_swaps_change_only_prescribed_dimensions_from_bc(self):
        original = rows();by = audit.paired_rows(original)
        variants = audit.expected_variants(original)
        for group in variants:
            source = by[group["origin_method"]][group["context_id"]]
            self.assertEqual(group["parent_record_id"], source["id"])
            self.assertEqual(group["parent_record_sha256"], audit.digest(source))
            if group["kind"] == "coordinate_swap":
                coordinates = dict(audit.COORDINATES)[group["variant"]]
                for index in range(5):
                    expected = by["bc"][group["context_id"]]["values"][index] if index in coordinates else source["values"][index]
                    self.assertEqual(group["values"][index], expected)
            elif group["kind"] == "historical_rejection":
                self.assertFalse(source["native_release"])
        self.assertEqual(sum(row["kind"] == "historical_rejection" for row in variants), 15)

    def test_deep_equality_requires_complete_fields_and_explicit_booleans(self):
        audit.same(dict(a=[1., False]), dict(a=[1.+1e-12, False]), Checks(), "fixture")
        with self.assertRaises(EvidenceError): audit.same(dict(a=[1., 0]), dict(a=[1., False]), Checks(), "fixture")
        with self.assertRaises(EvidenceError): audit.same(dict(a=1., hidden=1), dict(a=1.), Checks(), "fixture")

    def test_full_independent96_plan_matches_worker_and_freezes_joints_and_sources(self):
        from sircus.mjlab.policy64_diagnosis_native import make_plan
        original = complete_rows();before = copy.deepcopy(original)
        expected = audit.expected_plan(original)
        self.assertEqual(expected, make_plan(original));self.assertEqual(original, before)
        self.assertEqual((len(expected), len({r["id"] for r in expected})), (96, 96))
        self.assertEqual(sum(row["kind"] == "historical_rejection" for row in expected), 15)
        self.assertEqual(sum(row["kind"] == "coordinate_swap" for row in expected), 42)
        for row in expected:
            if row["kind"] == "historical_rejection":
                self.assertNotIn("q10", row);self.assertNotIn("q14", row);self.assertNotIn("native_cache_file", row)
            if row["kind"] == "exact_replay":
                self.assertEqual(row["q10"], next(r for r in original if r["id"] == row["parent_record_id"])["q10"])

    def test_geometry_evidence_is_measured_clearance_not_unreachability_claim(self):
        from sircus.mjlab.policy64_diagnosis import geometry_evidence
        connection = dict(reason="finite_ik_unresolved", passed=False, endpoint_attempts=dict(valid_ik_branches=2, reason="budget",
            pair_attempts=[dict(checked=dict(strict_geometry=dict(minimum_clearance_m=.003, closest_pair=["arm/a", "board/b"]))),
                           dict(checked=dict(strict_geometry=dict(minimum_clearance_m=.004, closest_pair=["arm/a", "board/b"]))) ]))
        result = audit.geometry_evidence(connection)
        self.assertEqual(result, geometry_evidence(connection))
        self.assertEqual((result["best_checked_clearance_m"], result["minimum_required_clearance_m"], result["checked_branch_pairs"]), (.004, .005, 2))
        self.assertIn("not a proof of unreachability", result["interpretation"])

    def test_frozen_contract_rejects_action_tampering_or_more_requests(self):
        original = complete_rows();plan = audit.expected_plan(original)
        parent = dict(scene={}, identity={}, packages={}, source_sha256={})
        manifest = {"contract.json": "a"*64, "rows.jsonl": "b"*64}
        analysis_sha = "c"*64
        bindings = dict(parent_contract_fingerprint=audit.PARENT_FP, parent_report_sha256=audit.PARENT_REPORT_SHA,
            parent_contract_sha256=manifest["contract.json"], parent_manifest_sha256=audit.PARENT_MANIFEST_SHA,
            parent_audit_sha256=audit.PARENT_AUDIT_SHA, parent_checks=346485, parent_rows_sha256=manifest["rows.jsonl"],
            parent_evidence_sha256=manifest, analysis_sha256=analysis_sha,
            parent_records_sha256={row["parent_record_id"]: row["parent_record_sha256"] for row in plan})
        spec = dict(version=audit.NATIVE_VERSION, scope="Post-hoc. No jitter or physical claim.", substeps=64, maximum_requests=96,
            maximum_acquired_s=3600, contexts=[key.removeprefix("policy64-test-") for key in audit.CONTEXTS],
            swaps=[dict(variant=v, coordinates=list(i)) for v, i in audit.COORDINATES], training=False, noise=False,
            analysis_sha256=analysis_sha, plan=plan, parent_bindings=bindings, **parent, native_source_sha256={})
        spec["fingerprint"] = audit.digest(spec)
        report = dict(version=audit.NATIVE_VERSION, status="completed", phase="completed", contract_fingerprint=spec["fingerprint"],
            analysis_sha256=analysis_sha, training=False, substeps=64,
            gate=dict(status="passed", controls_passed=True, complete=True, scope="Diagnostic execution completion only; no new policy success-rate claim"))
        audit.audit_contract(spec, report, parent, {}, original, manifest, analysis_sha, Checks())
        for mutation in ("extra", "action", "noise"):
            changed = copy.deepcopy(spec)
            if mutation == "extra": changed["maximum_requests"] = 97
            elif mutation == "action": changed["plan"][6]["values"][2] += .001
            else: changed["noise"] = True
            changed.pop("fingerprint");changed["fingerprint"] = audit.digest(changed)
            altered_report = dict(report, contract_fingerprint=changed["fingerprint"])
            with self.assertRaises(EvidenceError):
                audit.audit_contract(changed, altered_report, parent, {}, original, manifest, analysis_sha, Checks())


if __name__ == "__main__":
    unittest.main()
