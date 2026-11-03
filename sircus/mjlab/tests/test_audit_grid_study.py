"""Read-only auditor checks with synthetic evidence; no GPU or native meshes."""

import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from sircus.mjlab import audit_grid_study as audit
from sircus.mjlab.contract import FORCE_NAMES, ScoreStream
from sircus.mjlab.corrected_contract import NOISE, sha
from sircus.mjlab.learning_contract import digest
from sircus.mjlab import grid_study_contract as display


ROOT = Path(__file__).resolve().parents[3]
BUCKET = [0.3, -0.5]
HOME = "a" * 64
VALUES = [0.67, -0.1, 0.32, 28.0, 0.3]


def motion(source):
    request = dict(bucket_xy_m=BUCKET, **source)
    return dict(
        passed=True,
        common_home_sha256=HOME,
        request_sha256=digest(request),
        report=dict(
            request=request,
            endpoint=dict(reason=0, strict_geometry=dict(passed=True)),
            home_geometry=dict(passed=True),
            minimum_clearance_lower_bound_m=0.006,
            exact_selected_joint_max_error_rad=0.0,
            returned_home_joint_max_error_rad=0.0,
        ),
    )


def fixture(*, purpose="qualification", rejected=(), nominal_failure=False):
    n = 30 if purpose == "frozen" else 20
    source = dict(values=VALUES, q10=[0.0] * 6, q14=[1.0] * 6)
    proof = motion(source)
    record = dict(
        source_id="source",
        source_values=source["values"],
        source_q10=source["q10"],
        source_q14=source["q14"],
        purpose=purpose,
        seed=456,
        perturbation_std=NOISE.tolist(),
        motion=proof,
        nominal=[],
        fresh=[],
        variations=[],
        qualified=False,
    )
    rows, geometry = {}, []
    for index in range(3):
        name = f"source-{purpose}-nominal-{index}"
        row = dict(
            source,
            id=name,
            phase="nominal",
            bucket_xy_m=BUCKET,
            native_release=True,
            route_success=not (nominal_failure and index == 2),
            invalid=None,
            scalar_score_audit_passed=True,
            route_quality=2,
        )
        rows[name] = row
        record["nominal"].append(name)
    record["nominal_counts"] = audit.cohort_counts(list(rows.values()))
    if nominal_failure and purpose == "qualification":
        return record, rows, geometry
    jitter = np.asarray(VALUES) + np.random.default_rng(456).normal(0, NOISE, (n, 5))
    for index, value in enumerate(jitter):
        name = f"source-{purpose}-fresh-{index}"
        variation = dict(id=name, values=value.tolist(), q10=[0.0] * 6, q14=[1.0] * 6)
        admitted = index not in rejected
        request = dict(
            bucket_xy_m=BUCKET, **{k: variation[k] for k in ("values", "q10", "q14")}
        )
        local = dict(
            passed=admitted,
            request=request,
            request_sha256=digest(request),
            nominal_request_sha256=proof["request_sha256"],
            common_home_sha256=HOME,
            endpoint=dict(reason=0, strict_geometry=dict(passed=True)),
            local_edges=[dict(passed=True)] * 4,
            minimum_clearance_lower_bound_m=0.006,
            returned_nominal_joint_max_error_rad=0.0,
        )
        variation.update(admitted=admitted, local_path=local)
        geometry.append(copy.deepcopy(variation))
        rows[name] = dict(
            variation,
            phase="fresh",
            bucket_xy_m=BUCKET,
            native_release=admitted,
            route_success=admitted,
            invalid=None if admitted else "path_rejected",
            scalar_score_audit_passed=admitted,
            route_quality=2 if admitted else 0,
        )
        record["fresh"].append(name)
        record["variations"].append(variation)
    record["fresh_counts"] = audit.cohort_counts(
        [rows[name] for name in record["fresh"]]
    )
    record["quality"] = dict(clean=n - len(rejected), assisted=0, cohort="fresh")
    record["qualified"] = bool(
        not nominal_failure and n == 20 and n - len(rejected) >= 18
    )
    return record, rows, geometry


def parent_fixture(root):
    """Compact parent archive and an exact byte-preserving saved wrapper."""
    identity = dict(
        contract_fingerprint="a" * 64,
        scene_sha256="b" * 64,
        asset_sha256={"mesh": "c" * 64},
        source_sha256={"worker": "d" * 64},
        physics=copy.deepcopy(display.PHYSICS),
    )
    layout = display.grid_layouts()[0]
    cell_id = layout["id"]
    request = dict(
        request_id="parent-shot", mode="search", bucket_xy=layout["bucket_xy_m"]
    )
    result = dict(
        request_id=request["request_id"],
        bucket_xy_m=layout["bucket_xy_m"],
        success=True,
        qualified=True,
        status_native="verified_shot",
        method="sweep_planner",
        configuration=dict(
            board_center=[0.6, -0.3, 0.4],
            release_center=[0.6, -0.3, 0.7],
            tilt_deg=32.0,
            gap_m=0.2625,
        ),
        trajectory=[],
        motion=dict(passed=True),
        constraints=None,
        metrics=dict(
            cpu_candidates=84,
            geometry_rejected=53,
            search_cpu_candidates=64,
            search_geometry_rejected=52,
            search_requests=12,
            search_captures=2,
            confirmation_requests=23,
            nominal_captures=3,
            nominal_attempts=3,
            fresh_captures=20,
            fresh_attempts=20,
            scalar_score_audit_passed=True,
            invalid=None,
        ),
        provenance=dict(
            identity, policy_used=False, saved_evidence=False, request_sha256="e" * 64
        ),
    )
    stage = root / "cells" / cell_id / "baseline"
    stage.mkdir(parents=True)
    for kind, value in (("request", request), ("result", result)):
        (stage / ("original-" + kind + ".json")).write_text(
            json.dumps(value, indent=2) + "\n"
        )
    attempt = dict(
        stage="baseline",
        request=request,
        result=result,
        result_sha256=sha(stage / "original-result.json"),
        evidence_sha256={},
        result_url="/api/planner-grid/result/" + cell_id,
    )
    config_sha = digest(result["configuration"])
    report = display.make_report(
        {cell_id: [attempt]},
        identity=identity,
        protocol={"fixture": True},
        status="completed",
        states={
            c["id"]: ("verified_fresh" if c["id"] == cell_id else "unresolved")
            for c in display.grid_layouts()
        },
        validation_by_cell={
            cell_id: dict(
                status="completed",
                configuration_sha256=config_sha,
                seed=123,
                nominal=dict(captures=3, attempts=3),
                fresh=dict(captures=29, attempts=30),
            )
        },
    )
    parent_contract = dict(identity=identity, protocol={"fixture": True}, packages={})
    parent_contract["fingerprint"] = digest(parent_contract)
    archive = root / "parent"
    archive.mkdir()
    files = {
        "contract.json": parent_contract,
        "report.json": report,
        "frozen-selection.json": {cell_id: report["cells"][0]["selected"]},
    }
    for name, payload in files.items():
        (archive / name).write_text(json.dumps(payload, indent=2) + "\n")
    original_manifest = {name: sha(archive / name) for name in files}
    for kind in ("request", "result"):
        original_manifest[f"cells/{cell_id}/baseline/{kind}.json"] = sha(
            stage / ("original-" + kind + ".json")
        )
    (archive / "manifest.json").write_text(
        json.dumps(original_manifest, indent=2) + "\n"
    )
    parent_audit = dict(
        status="passed",
        checks=123,
        failures=[],
        contract_fingerprint=parent_contract["fingerprint"],
        study_manifest_sha256=sha(archive / "manifest.json"),
        report_sha256=sha(archive / "report.json"),
        frozen_selection_sha256=sha(archive / "frozen-selection.json"),
        source_sha256=identity["source_sha256"],
        asset_sha256=identity["asset_sha256"],
        physics=identity["physics"],
        packages={},
    )
    (archive / "audit.json").write_text(json.dumps(parent_audit, indent=2) + "\n")
    metadata = dict(
        version="sircus-grid35-parent-evidence-v1",
        contract_fingerprint=parent_contract["fingerprint"],
        report_fingerprint=report["fingerprint"],
        study_manifest_sha256=sha(archive / "manifest.json"),
        report_sha256=sha(archive / "report.json"),
        frozen_selection_sha256=sha(archive / "frozen-selection.json"),
        audit_sha256=sha(archive / "audit.json"),
        source_sha256=identity["source_sha256"],
        asset_sha256=identity["asset_sha256"],
        physics=identity["physics"],
    )
    child_identity = copy.deepcopy(identity)
    child_identity["source_sha256"]["new-driver"] = "f" * 64
    child_contract = dict(identity=child_identity, parent=metadata, packages={})
    evidence = dict(
        parent_contract_fingerprint=metadata["contract_fingerprint"],
        parent_report_fingerprint=metadata["report_fingerprint"],
        parent_manifest_sha256=metadata["study_manifest_sha256"],
        parent_audit_sha256=metadata["audit_sha256"],
        original_request_id=request["request_id"],
        original_stage="baseline",
        original_request_payload_sha256=digest(request),
        original_result_payload_sha256=digest(result),
        original_request_file=f"cells/{cell_id}/baseline/request.json",
        original_result_file=f"cells/{cell_id}/baseline/result.json",
        original_result_sha256=sha(stage / "original-result.json"),
        original_provenance=copy.deepcopy(result["provenance"]),
        original_metrics=copy.deepcopy(result["metrics"]),
    )
    wrapper = copy.deepcopy(result)
    wrapper["method"] = "catalog_lookup"
    wrapper["provenance"] = dict(
        child_identity, parent_evidence=evidence, saved_evidence=True
    )
    for key in (
        "cpu_candidates",
        "geometry_rejected",
        "search_requests",
        "confirmation_requests",
        "search_cpu_candidates",
        "search_geometry_rejected",
        "search_captures",
    ):
        wrapper["metrics"][key] = 0
    child_attempt = dict(stage="baseline", request=request, result=wrapper)
    return child_contract, child_identity, child_attempt, report["cells"][0], stage


def diversity_fixture(root, *, indices=(0, 1, 4, 5)):
    rows = []
    for i in range(8):
        rows.append(
            dict(
                id=f"candidate-{i}",
                values=[0.6 + i * 0.001, -0.2, 0.3, 28.0 + i, 0.2 + i * 0.005],
                q10=[0.0] * 6,
                q14=[1.0] * 6,
                geometry_eligible=True,
                proposal=dict(
                    method="broad_space_filling" if i < 4 else "local_interleaved_sweep"
                ),
            )
        )
    chosen = [rows[i] for i in indices]

    def summary(values):
        a = np.asarray([r["values"] for r in values])
        return dict(
            count=len(values),
            ranges={
                key: [float(a[:, i].min()), float(a[:, i].max())]
                for i, key in enumerate(
                    ("board_x_m", "board_y_m", "board_height_m", "tilt_deg", "gap_m")
                )
            },
        )

    selection = dict(
        planned_cpu_candidates=8,
        checked_cpu_candidates=8,
        all_proposals_audited=True,
        native_budget=4,
        requested_broad_quota=2,
        available_broad=4,
        selected_broad=sum(i < 4 for i in indices),
        quota_satisfied=True,
        selected_indices=list(indices),
        selected_ids=[r["id"] for r in chosen],
        selection_sha256=digest(chosen),
        checked=summary(rows),
        eligible=summary(rows),
        selected=summary(chosen),
    )
    (root / "diversity.json").write_text(json.dumps(selection))
    (root / "proposals.json").write_text(
        json.dumps(
            dict(
                values=[r["values"] for r in rows],
                metadata=[r["proposal"] for r in rows],
            )
        )
    )
    shots = [dict(r, phase="search") for r in chosen]
    return rows, shots, dict(search_cpu_candidates=8, search_geometry_rejected=0)


class GridAuditTests(unittest.TestCase):
    def test_manifest_covers_all_files_and_detects_mutation(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "row.json").write_text('{"ok":true}')
            (root / "manifest.json").write_text(
                json.dumps({"row.json": sha(root / "row.json")})
            )
            self.assertEqual(
                set(audit.check_manifest(root, audit.Checks())), {"row.json"}
            )
            (root / "row.json").write_text('{"ok":false}')
            with self.assertRaisesRegex(audit.EvidenceError, "hash changed"):
                audit.check_manifest(root, audit.Checks())

    def test_manifest_rejects_unlisted_files_and_path_escape(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "row").write_text("one")
            (root / "hidden").write_text("two")
            (root / "manifest.json").write_text(json.dumps({"row": sha(root / "row")}))
            with self.assertRaisesRegex(audit.EvidenceError, "every evidence file"):
                audit.check_manifest(root, audit.Checks())
            with self.assertRaisesRegex(audit.EvidenceError, "escapes"):
                audit.contained(root, "../outside")

    def test_twenty_qualification_keeps_rejects_in_denominator(self):
        record, rows, geometry = fixture(rejected=(2, 5))
        result = audit.check_confirmation(
            record, rows, geometry, BUCKET, HOME, audit.Checks(), "mock"
        )
        self.assertEqual(
            audit.cohort_counts(result["fresh"]), dict(captures=18, attempts=20)
        )
        self.assertTrue(record["qualified"])

    def test_frozen_thirty_is_not_reclassified_as_qualification(self):
        record, rows, geometry = fixture(purpose="frozen")
        result = audit.check_confirmation(
            record, rows, geometry, BUCKET, HOME, audit.Checks(), "mock"
        )
        self.assertEqual(len(result["fresh"]), 30)
        self.assertFalse(record["qualified"])

    def test_nominal_failure_skips_fresh_qualification(self):
        record, rows, geometry = fixture(nominal_failure=True)
        result = audit.check_confirmation(
            record, rows, geometry, BUCKET, HOME, audit.Checks(), "mock"
        )
        self.assertEqual(
            audit.cohort_counts(result["nominal"]), dict(captures=2, attempts=3)
        )
        self.assertEqual(result["fresh"], [])

    def test_seeded_value_or_branch_tamper_is_rejected(self):
        for field in ("values", "q14"):
            record, rows, geometry = fixture()
            record["variations"][0][field] = [
                v + 0.01 for v in record["variations"][0][field]
            ]
            with self.assertRaises(audit.EvidenceError):
                audit.check_confirmation(
                    record, rows, geometry, BUCKET, HOME, audit.Checks(), "mock"
                )

    def test_path_cannot_admit_wrong_nominal_or_clearance(self):
        for field, value in (
            ("nominal_request_sha256", "z" * 64),
            ("minimum_clearance_lower_bound_m", 0.004),
        ):
            record, rows, geometry = fixture()
            record["variations"][0]["local_path"][field] = value
            with self.assertRaises(audit.EvidenceError):
                audit.check_confirmation(
                    record, rows, geometry, BUCKET, HOME, audit.Checks(), "mock"
                )

    def trace_fixture(self, root):
        scene = json.loads((ROOT / "sircus/isaac/scene.json").read_text())
        position = np.zeros((960, 1, 3))
        position[:, :, 0] = 0.6
        position[:, :, 1] = 0.1
        position[:, :, 2] = -0.3
        velocity = np.zeros_like(position)
        forces = np.zeros((960, 1, len(FORCE_NAMES)))
        scorer = ScoreStream(scene, BUCKET, [0.0, 0.0, 1.0])
        for p, v, f in zip(position[:, 0], velocity[:, 0], forces[:, 0]):
            scorer.step(p, v, dict(zip(FORCE_NAMES, f)))
        row = dict(
            scorer.result(),
            id="mock-shot",
            bucket_xy_m=BUCKET,
            board_normal=[0.0, 0.0, 1.0],
            native_release=True,
            invalid=None,
            scalar_score_audit_passed=True,
            native_cache_file="native-0000.npz",
            native_cache_row=0,
        )
        np.savez_compressed(
            root / "native-0000.npz",
            position_m=position,
            velocity_m_s=velocity,
            forces_N=forces,
            record_ids=np.asarray(["mock-shot"]),
        )
        return scene, row

    def test_native_trace_is_independently_rescored_and_id_bound(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            scene, row = self.trace_fixture(root)
            traces = audit.rescore_rows(root, [row], scene, audit.Checks(), "mock")
            self.assertEqual(len(traces["mock-shot"]), 240)
            row["id"] = "another-shot"
            with self.assertRaisesRegex(audit.EvidenceError, "trace ID"):
                audit.rescore_rows(root, [row], scene, audit.Checks(), "mock")

    def test_wrong_score_or_unassigned_arrays_are_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            scene, row = self.trace_fixture(root)
            row["route_success"] = True
            with self.assertRaisesRegex(audit.EvidenceError, "route_success"):
                audit.rescore_rows(root, [row], scene, audit.Checks(), "mock")
            with self.assertRaisesRegex(
                audit.EvidenceError, "unassigned native arrays"
            ):
                audit.rescore_rows(root, [], scene, audit.Checks(), "mock")

    def test_archived_parent_and_wrapper_preserve_original_ids_and_final_rates(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            contract, identity, attempt, cell, path = parent_fixture(root)
            checks = audit.Checks()
            parent = audit.parent_snapshot(root, contract, identity, checks)
            original = audit.check_parent_wrapper(
                path, attempt, parent, checks, cell["id"]
            )
            self.assertEqual(original["request_id"], "parent-shot")
            self.assertEqual(attempt["result"]["metrics"]["search_requests"], 0)
            current = copy.deepcopy(cell)
            current["selected"]["source"] = "saved_reference"
            inherited = audit.inherited_validation(current, parent, checks)
            self.assertEqual(inherited["fresh"]["captures"], 29)
            self.assertEqual(inherited["fresh"]["attempts"], 30)

    def test_parent_report_bytes_or_native_source_change_is_rejected(self):
        for corruption in ("bytes", "source"):
            with tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                contract, identity, _, _, _ = parent_fixture(root)
                if corruption == "bytes":
                    with (root / "parent" / "report.json").open("a") as stream:
                        stream.write("\n")
                else:
                    identity["source_sha256"]["worker"] = "9" * 64
                with self.assertRaises(audit.EvidenceError):
                    audit.parent_snapshot(root, contract, identity, audit.Checks())

    def test_parent_wrapper_cannot_claim_new_trials_or_replace_original_provenance(
        self,
    ):
        for corruption in ("counter", "provenance", "bytes"):
            with tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                contract, identity, attempt, cell, path = parent_fixture(root)
                parent = audit.parent_snapshot(root, contract, identity, audit.Checks())
                if corruption == "counter":
                    attempt["result"]["metrics"]["confirmation_requests"] = 23
                elif corruption == "provenance":
                    attempt["result"]["provenance"]["parent_evidence"][
                        "original_provenance"
                    ] = {}
                else:
                    with (path / "original-result.json").open("a") as stream:
                        stream.write("\n")
                with self.assertRaises(audit.EvidenceError):
                    audit.check_parent_wrapper(
                        path, attempt, parent, audit.Checks(), cell["id"]
                    )

    def test_inherited_final_rate_cannot_attach_to_a_new_configuration(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            contract, identity, _, cell, _ = parent_fixture(root)
            parent = audit.parent_snapshot(root, contract, identity, audit.Checks())
            cell = copy.deepcopy(cell)
            cell["selected"]["configuration_sha256"] = "8" * 64
            with self.assertRaisesRegex(audit.EvidenceError, "frozen configuration"):
                audit.inherited_validation(cell, parent, audit.Checks())

    def test_followup_targets_include_unresolved_and_fragile_final_recipes(self):
        criterion = dict(
            nominal_required=3,
            nominal_attempts=3,
            fresh_required=27,
            fresh_attempts=30,
            missing_or_blocked=True,
        )
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            contract, identity, _, cell, _ = parent_fixture(root)
            parent = audit.parent_snapshot(root, contract, identity, audit.Checks())
            targets = audit.followup_targets(
                parent["report"], criterion, audit.Checks()
            )
            self.assertEqual(len(targets), 34)
            self.assertNotIn(cell["id"], targets)
            parent["report"]["cells"][0]["frozen_validation"]["fresh"]["captures"] = 26
            targets = audit.followup_targets(
                parent["report"], criterion, audit.Checks()
            )
            self.assertEqual(len(targets), 35)
            self.assertIn(cell["id"], targets)
            criterion["fresh_required"] = 26
            with self.assertRaisesRegex(audit.EvidenceError, "criterion"):
                audit.followup_targets(parent["report"], criterion, audit.Checks())

    def test_diversity_binds_all_audited_proposals_and_native_selected_rows(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            geometry, shots, metrics = diversity_fixture(root)
            result = audit.check_diversity(
                root, geometry, shots, metrics, audit.Checks(), "mock"
            )
            self.assertEqual(result["checked"], 8)
            self.assertEqual(result["native_selected"], 4)
            self.assertEqual(result["broad_selected"], 2)

    def test_diversity_rejects_prefix_only_admission_and_missing_broad_quota(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            geometry, shots, metrics = diversity_fixture(root)
            metrics["search_cpu_candidates"] = 6
            with self.assertRaisesRegex(audit.EvidenceError, "every proposal audited"):
                audit.check_diversity(
                    root, geometry, shots, metrics, audit.Checks(), "mock"
                )
            geometry, shots, metrics = diversity_fixture(root, indices=(4, 5, 6, 7))
            with self.assertRaisesRegex(audit.EvidenceError, "broad family quota"):
                audit.check_diversity(
                    root, geometry, shots, metrics, audit.Checks(), "mock"
                )

    def test_diversity_rejects_changed_native_branch_or_false_parameter_range(self):
        for corruption in ("branch", "range"):
            with tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                geometry, shots, metrics = diversity_fixture(root)
                if corruption == "branch":
                    shots[0]["q14"] = [2.0] * 6
                else:
                    selection = json.loads((root / "diversity.json").read_text())
                    selection["selected"]["ranges"]["gap_m"] = [0.0, 1.0]
                    (root / "diversity.json").write_text(json.dumps(selection))
                with self.assertRaises(audit.EvidenceError):
                    audit.check_diversity(
                        root, geometry, shots, metrics, audit.Checks(), "mock"
                    )


if __name__ == "__main__":
    unittest.main()
