"""CPU controls for replay accounting; synthetic rows are never native evidence."""

import copy
import json
from pathlib import Path
import tempfile
from unittest.mock import patch
import unittest

import numpy as np

from sircus.mjlab import replay650_verify as verify
from sircus.mjlab.benchmark_verify import Checks, EvidenceError


def rows():
    result = []
    for step in (64, 128):
        for role in ("positive_control", "candidate", "negative_control"):
            for index in range(3):
                result.append(dict(id=f"{step}-{role}-{index}", pair_id=f"{role}-{index}",
                    group_id=role, role=role, repeat_index=index, substeps=step,
                    q10=[0.]*6, q14=[0.]*6, values=[.47, -.27, .38, 28., .3868],
                    bucket_xy_m=[.1, -.6], native_release=True, invalid=None,
                    route_success=role != "negative_control", invalid_physics=False,
                    scalar_score_audit_passed=True, route_quality=2 if role != "negative_control" else 0,
                    whole_ball_entry=role != "negative_control", max_retention_s=1. if role != "negative_control" else 0.))
    return result


def options():
    result = {}
    for step in (64, 128):
        native = dict(timestep=1/(240*step), solver=2, iterations=100, tolerance=1e-8,
                      cone=0, integrator=3, ccd_iterations=50, ccd_tolerance=1e-6,
                      disableflags=0, enableflags=0, gravity=[0., 0., -9.81])
        warp = {k: (np.asarray(1e-6 if k == "tolerance" else v, dtype=np.float32).astype(float).tolist()
                    if k in {"timestep", "tolerance", "ccd_tolerance", "gravity"} else v)
                for k, v in native.items()}
        result[str(step)] = dict(version="sircus-calibrated-native-warp-options-v1", substeps=step, native=native, warp=warp)
    return result


def group_fixture():
    scene = dict(table=dict(center_m=[0., 0., 0.], size_m=[4., 4., .1]),
                 bucket=dict(outer_top_diameter_m=.2, outer_bottom_diameter_m=.158), ball=dict(radius_m=.0375))
    original = [dict(id=ident, role=role, source_record_id="source-"+ident, source_record_sha256="source-hash-"+ident,
                     values=[.47, -.27, .38, 28., .3868], bucket_xy_m=[.1, -.6], q10=[0.]*6, q14=[0.]*6)
                for ident, role in (("positive-control", "positive_control"), ("candidate-650", "candidate"),
                                    ("negative-control", "negative_control"))]
    groups = copy.deepcopy(original)
    for gap in verify.WINDOW_GAPS_M:
        group = copy.deepcopy(original[1])
        group.update(id="window-gap-"+str(round(gap*10000)), role="window")
        group["values"][4] = gap
        if gap != .3868:
            group["q10"][0] = .001
        groups.append(group)
    proofs = []
    for index, group in enumerate(groups):
        xy = np.asarray(group["bucket_xy_m"])
        edges = np.r_[xy+2-.1, 2-xy-.1]
        proof = dict(group_id=group["id"], values=group["values"], bucket_xy_m=group["bucket_xy_m"], eligible=True,
                     held_endpoints_only=True, preserved_source=index < 3, bounds_passed=True,
                     bucket_footprint=dict(outer_radius_m=.1, edge_clearance_m=edges.tolist(),
                                           minimum_edge_clearance_m=float(min(edges)), passed=True),
                     attempts=[dict(q10=group["q10"], q14=group["q14"], checked=dict(reason=0, minimum_clearance_lower_bound_m=.006))],
                     minimum_clearance_lower_bound_m=.006)
        group.update(eligible=True, geometry_row=index, geometry_sha256=verify.digest(proof))
        proofs.append(proof)
    return scene, original, groups, proofs


class ReplayAccountingTests(unittest.TestCase):
    def test_gate_requires_both_resolutions_and_all_three_copies(self):
        data = rows()
        self.assertTrue(verify.replay_gate(data, .005)["window_opened"])
        self.assertFalse(verify.replay_gate(data[:-1], .001)["window_opened"])
        changed = copy.deepcopy(data)
        next(r for r in changed if r["role"] == "candidate" and r["substeps"] == 128)["route_success"] = False
        self.assertFalse(verify.replay_gate(changed, .001)["candidate"])
        self.assertFalse(verify.replay_gate(data, np.nextafter(.005, np.inf))["window_opened"])

    def test_negative_must_be_valid_native_miss_not_endpoint_rejection(self):
        data = rows()
        negative = next(r for r in data if r["role"] == "negative_control")
        negative.update(native_release=False, invalid="endpoint_rejected")
        out = verify.replay_gate(data, 0.)
        self.assertFalse(out["negative_control"])
        self.assertFalse(out["physics_valid"])
        self.assertFalse(out["window_opened"])
        self.assertEqual(verify.statistics(data)["attempts"], 18)
        self.assertEqual(verify.statistics(data)["native_releases"], 17)

    def test_physics_and_actual_oracle_checks_cannot_be_omitted(self):
        for field, value, failure in (("scalar_score_audit_passed", False, "oracle_agreement"),
                                      ("invalid_physics", True, "physics_valid")):
            data = rows()
            data[0][field] = value
            gate = verify.replay_gate(data, .001)
            self.assertFalse(gate[failure])
            self.assertFalse(gate["window_opened"])

    def test_exact_pairing_prevents_different_joints_or_actions_at_fine_step(self):
        data = rows()
        self.assertEqual(len(verify.check_pairing(data, Checks())), 9)
        for field, value in (("q14", [0., 0., 0., 0., 0., 1e-10]),
                             ("values", [.47, -.27, .38, 28., .386800001])):
            changed = copy.deepcopy(data)
            changed[-1][field] = value
            with self.assertRaisesRegex(EvidenceError, "same frozen input"):
                verify.check_pairing(changed, Checks())
        with self.assertRaisesRegex(EvidenceError, "one request"):
            verify.check_pairing(data + [data[0]], Checks())

    def test_sensitivity_uses_observed_positions_and_retains_outcome_disagreement(self):
        data = [r for r in rows() if r["role"] == "candidate"]
        position = {r["id"]: [0., 0., .38] for r in data}
        position[data[-1]["id"]] = [.003, .004, .38]
        data[-1]["route_success"] = False
        value = verify.paired_sensitivity(data, position)
        self.assertEqual(value["paired_attempts"], 3)
        self.assertEqual(value["capture_agreement"], 2)
        self.assertAlmostEqual(value["nominal_position_delta_350ms_max_m"], .005)
        with self.assertRaisesRegex(EvidenceError, "Unpaired"):
            verify.paired_sensitivity(data[:-1], position)

    def test_dense_interval_schema_changes_cadence_without_refitting_calibration(self):
        a, b = (verify.native_trace_spec(s) for s in (64, 128))
        self.assertEqual(a["native_dt_s"], 1/15360)
        self.assertEqual(b["native_dt_s"], 1/30720)
        self.assertEqual(a["horizon_s"], b["horizon_s"])
        self.assertEqual(a["force_names"], b["force_names"])

    def test_options_pin_both_native_and_vendor_floor_exactly(self):
        verify.audit_options(options(), Checks())
        for engine, field, value in (("native", "tolerance", np.nextafter(1e-8, np.inf)),
                                     ("warp", "tolerance", 1e-8), ("warp", "timestep", 1/15360)):
            changed = options()
            changed["128"][engine][field] = value
            with self.assertRaisesRegex(EvidenceError, "options"):
                verify.audit_options(changed, Checks())

    def test_dependency_ancestry_explicitly_includes_scipy_and_rejects_changed_pin(self):
        from sircus.mjlab.corrected_contract import PINNED_PACKAGES
        declared = dict(PINNED_PACKAGES, scipy="1.15.3")
        declared["torch"] += "+cu128"
        with patch.object(verify.importlib.metadata, "version", side_effect=declared.__getitem__):
            self.assertEqual(verify.audit_packages(declared, Checks()), declared)
            missing = dict(declared); missing.pop("scipy")
            with self.assertRaisesRegex(EvidenceError, "complete native ancestry"):
                verify.audit_packages(missing, Checks())
        changed = dict(declared, scipy="1.15.4")
        with patch.object(verify.importlib.metadata, "version", side_effect=changed.__getitem__):
            with self.assertRaisesRegex(EvidenceError, "Pinned package/scipy"):
                verify.audit_packages(changed, Checks())

    def test_actual_frozen_calibration_schema_proves_cadence_through_fixture_metadata(self):
        from sircus.mjlab.calibrated_benchmark_contract import checked_calibration
        source = Path(__file__).resolve().parents[3]/"baselines/mjlab-calibrated-benchmark-20261104"
        declared = checked_calibration(source/"calibration.json", source/"verification.json")
        self.assertNotIn("native_dt_s", declared["score_config"])
        self.assertEqual(verify.check_calibration(source, declared, Checks()), declared["score_config"])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            calibration = json.loads((source/"calibration.json").read_text())
            verification = json.loads((source/"verification.json").read_text())
            calibration["fit_evidence"]["effective_options"]["timestep"] = 1/30720
            (root/"calibration.json").write_text(json.dumps(calibration))
            verification["calibration_sha256"] = verify.sha(root/"calibration.json")
            (root/"verification.json").write_text(json.dumps(verification))
            modified = checked_calibration(root/"calibration.json", root/"verification.json")
            with self.assertRaisesRegex(EvidenceError, "Actual frozen fit/verification native options"):
                verify.check_calibration(root, modified, Checks())

    def test_revised_auditor_is_separate_but_actual_oracle_dependency_must_match_producer(self):
        # Import the actual oracle; this test does no dynamics or GPU work.
        from sircus.mjlab import calibrated_score_oracle
        root = Path(__file__).resolve().parents[3]
        declared = {str(p.relative_to(root)): verify.sha(p) for p in (root/"sircus/mjlab").glob("*.py")}
        declared.update({str(p.relative_to(root)): verify.sha(p) for p in (root/"sircus/isaac").glob("*.py")})
        own = "sircus/mjlab/replay650_verify.py"
        with tempfile.TemporaryDirectory() as directory:
            producer = Path(directory)
            path = producer/own; path.parent.mkdir(parents=True)
            path.write_text("# Immutable historical checker bytes, not executed.\n")
            declared[own] = verify.sha(path)
            identity = verify.auditor_identity(producer, declared, Checks())
            self.assertNotEqual(identity["auditor_sha256"], identity["original_producer_auditor_sha256"])
            self.assertIn("sircus/mjlab/calibrated_score_oracle.py", identity["dependencies_sha256"])
            dirty = dict(declared)
            dirty["sircus/mjlab/calibrated_score_oracle.py"] = "0"*64
            with self.assertRaisesRegex(EvidenceError, "actual auditor core/oracle dependency"):
                verify.auditor_identity(producer, dirty, Checks())

    def test_exact_groups_and_real_proof_branch_detect_joint_and_clearance_tampering(self):
        scene, original, groups, proofs = group_fixture()
        verify.audit_groups(groups, proofs, original, scene, Checks())
        changed = copy.deepcopy(groups)
        changed[3]["q14"][0] += 1e-10
        with self.assertRaisesRegex(EvidenceError, "board branch"):
            verify.audit_groups(changed, proofs, original, scene, Checks())
        changed = copy.deepcopy(proofs)
        changed[3]["minimum_clearance_lower_bound_m"] = .00499
        dirty = copy.deepcopy(groups)
        dirty[3]["geometry_sha256"] = verify.digest(changed[3])
        with self.assertRaisesRegex(EvidenceError, "strict endpoint clearance"):
            verify.audit_groups(dirty, changed, original, scene, Checks())

    def test_saved_requests_and_actual_native_initial_joints_are_independently_bound(self):
        _, _, groups, _ = group_fixture()
        plan = verify.expected_plan(groups[:3], "replay")
        actual = []
        for request in plan:
            row = dict(request, native_release=True, native_directory="native/"+str(request["substeps"]),
                       native_initial_q10=list(request["q10"]), native_initial_q14=list(request["q14"]),
                       contact_protocol=dict(substeps=request["substeps"], board_mode="articulated", gravity_mode="passive_only"),
                       scalar_score_audit_passed=True, endpoint_collision_pairs=[])
            actual.append(row)
        verify.audit_rows(actual, plan, groups, Checks())
        changed = copy.deepcopy(actual)
        changed[-1]["native_initial_q10"][0] = .01
        with self.assertRaisesRegex(EvidenceError, "actual float32 initial joint"):
            verify.audit_rows(changed, plan, groups, Checks())

    def test_evidence_manifest_detects_unassigned_or_changed_native_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/"cache.bin").write_bytes(b"synthetic byte identity fixture")
            (root/"evidence-manifest.json").write_text(json.dumps({"cache.bin": verify.sha(root/"cache.bin")}))
            verify.check_manifest(root, Checks())
            (root/"unassigned.npz").write_bytes(b"not native evidence")
            with self.assertRaisesRegex(EvidenceError, "covers every"):
                verify.check_manifest(root, Checks())

    def test_cpu_endpoints_are_rechecked_against_saved_proofs(self):
        scene, _, groups, proofs = group_fixture()
        # Real shot-target construction remains active; only the expensive
        # collision query is replaced with an explicit CPU test dependency.
        source_scene = json.loads((Path(__file__).resolve().parents[3]/"sircus/isaac/scene.json").read_text())
        class Gate:
            clearance = .006
            def __init__(self, *args): pass
            def check(self, *args):
                return dict(reason=0, minimum_clearance_lower_bound_m=self.clearance)
        with patch("sircus.mjlab.corrected_geometry.StrictEndpointGate", Gate):
            verify.audit_cpu_endpoints(groups, proofs, source_scene, Path("unused"), Checks())
            Gate.clearance = .0049
            with self.assertRaisesRegex(EvidenceError, "fresh actual strict5mm"):
                verify.audit_cpu_endpoints(groups, proofs, source_scene, Path("unused"), Checks())

    def test_actual_producer_gate_matches_independent_nominal_gate(self):
        from sircus.mjlab import replay650_contract as producer
        data = rows()
        for row in data:
            row["phase"] = "replay"
        arrays = {r["id"]: np.zeros((round(.35*240*r["substeps"])+1, 3)) for r in data}
        self.assertEqual(producer.replay_gate(data, arrays), verify.replay_gate(data, 0.))
        for field, value in (("route_success", False), ("invalid_physics", True),
                             ("scalar_score_audit_passed", False)):
            dirty = copy.deepcopy(data)
            changed = next(r for r in dirty if r["role"] == "candidate")
            changed[field] = value
            if field == "invalid_physics":
                changed["invalid"] = "native_physics_outside_calibrated_domain"
            self.assertEqual(producer.replay_gate(dirty, arrays), verify.replay_gate(dirty, 0.))

    def test_closed_window_cannot_be_fabricated_as_executed_or_omitted_after_success(self):
        _, _, groups, _ = group_fixture()
        initial = verify.expected_plan(groups[:3], "replay")
        window = verify.expected_plan(groups[3:], "window")
        self.assertEqual(len(initial), 18)
        self.assertEqual(len(window), 30)
        rejected = [dict(r, native_release=False, invalid="strict_endpoint_or_local_branch_rejected",
                         route_success=False, whole_ball_entry=False) for r in initial]
        rejected_groups = copy.deepcopy(groups)
        for group in rejected_groups:
            group["eligible"] = False
        for row in rejected:
            row["eligible"] = False
        expected = verify.expected_plan(rejected_groups[:3], "replay")
        verify.audit_rows(rejected, expected, rejected_groups, Checks())
        with self.assertRaisesRegex(EvidenceError, "actually opened requests"):
            verify.audit_rows(rejected + rejected[:1], expected, rejected_groups, Checks())
        with self.assertRaisesRegex(EvidenceError, "actually opened requests"):
            verify.audit_rows(rejected, expected+window, rejected_groups, Checks())

    def test_display_aggregate_matches_actual_saved_requests_in_both_conditional_cases(self):
        from collections import Counter
        from sircus.mjlab import replay650_contract as producer
        scene, _, groups, _ = group_fixture()
        for opened in (False, True):
            plan = verify.expected_plan(groups[:3], "replay")
            if opened:
                plan += verify.expected_plan(groups[3:], "window")
            data = [dict(r, native_release=True, invalid=None, invalid_physics=False,
                         route_success=r["role"] != "negative_control" and (opened or r["role"] != "candidate"),
                         whole_ball_entry=r["role"] != "negative_control", max_retention_s=1.,
                         route_quality=1, scalar_score_audit_passed=True) for r in plan]
            arrays = {r["id"]: np.zeros((round(.35*240*r["substeps"])+1, 3)) for r in data}
            point = {r["id"]: [0., 0., 0.] for r in data}
            gate = producer.replay_gate(data, arrays)
            report = dict(gate=gate, counts=dict(**producer.statistics(data),
                phase_counts=dict(Counter(r["phase"] for r in data)), role_counts=dict(Counter(r["role"] for r in data)),
                substep_counts=dict(Counter(str(r["substeps"]) for r in data))), groups=[])
            for group in groups:
                selected = [r for r in data if r["group_id"] == group["id"]]
                report["groups"].append(dict(**producer.metadata(group, scene), eligible=group["eligible"],
                    by_substeps={str(s): producer.statistics([r for r in selected if r["substeps"] == s]) for s in (64, 128)},
                    sensitivity=producer.sensitivity(selected, arrays)))
            verify.audit_report(report, data, groups, point, gate, Checks())
            dirty = copy.deepcopy(report)
            dirty["groups"][1]["by_substeps"]["128"]["captures"] += 1
            with self.assertRaisesRegex(EvidenceError, "captures"):
                verify.audit_report(dirty, data, groups, point, gate, Checks())


if __name__ == "__main__":
    unittest.main()
