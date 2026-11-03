"""Read-only audit of frozen 35-cell search and native repeat evidence.

This rescoring tool never starts a GPU environment or changes source evidence.
Optional CPU geometry checks repeat strict endpoints and sampled local paths.
Neither operation is a new physics rollout or a continuous path certificate.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import time

import numpy as np

from .corrected_contract import PHYSICS, NOISE, geometry_hint, grid_layouts, sha
from .learning_contract import digest

VERSION = "sircus-independent-grid35-evidence-audit-v1"
SCORE_FIELDS = (
    "success",
    "route_success",
    "route_quality",
    "complete_success",
    "board_bounce",
    "whole_ball_entry",
    "external_contact_steps",
    "board_contact_episodes",
    "single_bounce_success",
    "physics_steps",
)
FLOAT_FIELDS = (
    "closest_opening_distance_m",
    "max_retention_s",
    "minimum_mouth_clearance_mm",
)
SELECTED_SHOT_RULE = dict(
    candidate_limit=6,
    ranking=[
        "qualification_fresh_captures",
        "qualification_fresh_clean_captures",
        "native_search_rank",
    ],
    native_search_rank="sweep_contract.rank_row(search_source)",
    early_stop=dict(
        nominal_captures=3,
        nominal_attempts=3,
        fresh_captures=20,
        fresh_attempts=20,
        fresh_clean_captures=20,
    ),
    frozen_validation="Never used for recipe selection or fallback",
)


class EvidenceError(ValueError):
    pass


class Checks:
    def __init__(self):
        self.count = 0
        self.float_maximum_error = Counter()

    def require(self, passed, label):
        self.count += 1
        if not passed:
            raise EvidenceError(label)

    def equal(self, first, second, label):
        self.require(first == second, label)

    def numeric(self, first, second, label, tolerance=1e-10):
        if first is None or second is None:
            self.equal(first, second, label)
            return
        error = abs(float(first) - float(second))
        self.require(np.isfinite(error) and error <= tolerance, label)
        key = label.rsplit("/", 1)[-1]
        self.float_maximum_error[key] = max(self.float_maximum_error[key], error)


def contained(root, name):
    root = Path(root).resolve()
    path = (root / name).resolve()
    if Path(name).is_absolute() or not path.is_relative_to(root):
        raise EvidenceError(
            "Evidence path escapes its declared directory: " + str(name)
        )
    return path


def read(path):
    return json.loads(Path(path).read_text())


def check_manifest(directory, checks, *, exact=True):
    directory = Path(directory)
    manifest = read(directory / "manifest.json")
    checks.require(
        isinstance(manifest, dict) and bool(manifest),
        str(directory) + ": empty manifest",
    )
    for name, expected in manifest.items():
        checks.equal(
            sha(contained(directory, name)), expected, "File hash changed: " + name
        )
    if exact:
        actual = {
            str(p.relative_to(directory))
            for p in directory.rglob("*")
            if p.is_file() and p.name != "manifest.json"
        }
        checks.equal(
            set(manifest), actual, "Manifest does not cover every evidence file"
        )
    return manifest


def capture(row):
    return bool(
        row.get("native_release")
        and row.get("route_success")
        and not row.get("invalid")
        and row.get("scalar_score_audit_passed")
    )


def cohort_counts(rows):
    return dict(captures=sum(map(capture, rows)), attempts=len(rows))


def signature(row):
    return tuple(tuple(row[key]) for key in ("values", "q10", "q14"))


def parent_snapshot(study, contract, identity, checks):
    """Validate an archived, fully audited parent without crediting new trials."""
    metadata = contract.get("parent")
    if metadata is None:
        return None
    checks.equal(
        metadata.get("version"),
        "sircus-grid35-parent-evidence-v1",
        "Parent evidence version",
    )
    path = Path(study) / "parent"
    files = {
        name: read(path / name)
        for name in (
            "contract.json",
            "report.json",
            "frozen-selection.json",
            "manifest.json",
            "audit.json",
        )
    }
    for field, name in (
        ("study_manifest_sha256", "manifest.json"),
        ("report_sha256", "report.json"),
        ("frozen_selection_sha256", "frozen-selection.json"),
        ("audit_sha256", "audit.json"),
    ):
        checks.equal(sha(path / name), metadata.get(field), "Parent file/" + name)
    parent_contract = dict(files["contract.json"])
    fingerprint = parent_contract.pop("fingerprint")
    checks.equal(digest(parent_contract), fingerprint, "Parent contract fingerprint")
    checks.equal(
        metadata.get("contract_fingerprint"), fingerprint, "Parent contract binding"
    )
    parent_identity = parent_contract["identity"]
    parent_report = files["report.json"]
    from .grid_study_contract import validate_report

    validate_report(parent_report)
    checks.equal(parent_report["status"], "completed", "Parent study is complete")
    checks.equal(
        parent_report["fingerprint"],
        metadata.get("report_fingerprint"),
        "Parent report fingerprint",
    )
    checks.equal(parent_report["identity"], parent_identity, "Parent report identity")
    checks.equal(
        parent_report["protocol"], parent_contract["protocol"], "Parent protocol"
    )
    checks.equal(
        files["frozen-selection.json"],
        {c["id"]: c["selected"] for c in parent_report["cells"] if c.get("selected")},
        "Parent immutable selection",
    )
    # The original manifest references raw arrays outside this compact archive.
    # Only archived critical bytes and original request/result sidecars are
    # checked here. The full parent audit binds its original native arrays.
    original_manifest = files["manifest.json"]
    for name in ("contract.json", "report.json", "frozen-selection.json"):
        checks.equal(
            sha(path / name),
            original_manifest.get(name),
            "Parent original file/" + name,
        )
    parent_audit = files["audit.json"]
    checks.equal(
        parent_audit.get("status"), "passed", "Parent independent audit passed"
    )
    checks.equal(parent_audit.get("failures"), [], "Parent audit has no failures")
    checks.require(parent_audit.get("checks", 0) > 0, "Parent audit records checks")
    for key, value in (
        ("contract_fingerprint", fingerprint),
        ("study_manifest_sha256", metadata["study_manifest_sha256"]),
        ("report_sha256", metadata["report_sha256"]),
        ("frozen_selection_sha256", metadata["frozen_selection_sha256"]),
    ):
        checks.equal(parent_audit.get(key), value, "Parent audit binding/" + key)
    for key in ("source_sha256", "asset_sha256", "physics"):
        checks.equal(metadata.get(key), parent_identity[key], "Parent metadata/" + key)
        checks.equal(parent_audit.get(key), parent_identity[key], "Parent audit/" + key)
    for key in ("contract_fingerprint", "scene_sha256", "asset_sha256", "physics"):
        checks.equal(
            parent_identity[key], identity[key], "Parent compatible identity/" + key
        )
    for name, expected in parent_identity["source_sha256"].items():
        checks.equal(
            identity["source_sha256"].get(name),
            expected,
            "Parent compatible native/source dependency/" + name,
        )
    checks.equal(
        parent_audit.get("packages"),
        parent_contract["packages"],
        "Parent audited packages",
    )
    checks.equal(
        contract["packages"], parent_contract["packages"], "Parent compatible packages"
    )
    return dict(
        metadata=metadata,
        report=parent_report,
        manifest=original_manifest,
        contract=files["contract.json"],
        audit=parent_audit,
        cells={c["id"]: c for c in parent_report["cells"]},
    )


def check_parent_wrapper(path, attempt, parent, checks, cell_id):
    """A new-reader wrapper must preserve original bytes, IDs and provenance."""
    result = attempt["result"]
    evidence = result.get("provenance", {}).get("parent_evidence", {})
    original_request = read(Path(path) / "original-request.json")
    original_result = read(Path(path) / "original-result.json")
    original_request_id = original_request["request_id"]
    parent_cell = parent["cells"][cell_id]
    parent_attempt = next(
        a for a in parent_cell["attempts"] if a["request_id"] == original_request_id
    )
    if parent_cell.get("selected"):
        checks.equal(
            original_request_id,
            parent_cell["selected"]["request_id"],
            cell_id + "/parent selected request",
        )
    else:
        checks.equal(
            parent_attempt,
            parent_cell["attempts"][-1],
            cell_id + "/parent last unresolved attempt",
        )
    original_stage = parent_attempt["stage"]
    directory = "saved" if original_stage == "saved_reference" else original_stage
    for kind, payload in (("request", original_request), ("result", original_result)):
        filename = "original-" + kind + ".json"
        original_file = f"cells/{cell_id}/{directory}/{kind}.json"
        checks.equal(
            sha(Path(path) / filename),
            parent["manifest"].get(original_file),
            cell_id + "/parent original byte identity/" + kind,
        )
        checks.equal(
            digest(payload),
            parent_attempt[kind + "_payload_sha256"],
            cell_id + "/parent original payload/" + kind,
        )
        checks.equal(
            evidence.get("original_" + kind + "_payload_sha256"),
            digest(payload),
            cell_id + "/nested parent payload/" + kind,
        )
        checks.equal(
            evidence.get("original_" + kind + "_file"),
            original_file,
            cell_id + "/nested original path/" + kind,
        )
    expected = {
        "parent_contract_fingerprint": parent["metadata"]["contract_fingerprint"],
        "parent_report_fingerprint": parent["metadata"]["report_fingerprint"],
        "parent_manifest_sha256": parent["metadata"]["study_manifest_sha256"],
        "parent_audit_sha256": parent["metadata"]["audit_sha256"],
        "original_request_id": original_request_id,
        "original_result_sha256": sha(Path(path) / "original-result.json"),
        "original_stage": original_stage,
        "original_provenance": original_result["provenance"],
        "original_metrics": original_result["metrics"],
    }
    for key, value in expected.items():
        checks.equal(
            evidence.get(key), value, cell_id + "/nested parent identity/" + key
        )
    checks.equal(
        original_result["request_id"],
        original_request_id,
        cell_id + "/original request/result ID",
    )
    checks.equal(
        attempt["request"]["bucket_xy"],
        original_request["bucket_xy"],
        cell_id + "/parent fixed bucket",
    )
    for key in (
        "configuration",
        "trajectory",
        "motion",
        "success",
        "qualified",
        "status_native",
        "bucket_xy_m",
        "constraints",
    ):
        checks.equal(
            result.get(key),
            original_result.get(key),
            cell_id + "/preserved parent result/" + key,
        )
    checks.equal(
        result.get("method"), "catalog_lookup", cell_id + "/parent wrapper method"
    )
    checks.require(
        result["provenance"].get("saved_evidence") is True,
        cell_id + "/parent saved evidence",
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
        checks.equal(
            result["metrics"].get(key), 0, cell_id + "/no new parent work/" + key
        )
    for key in (
        "nominal_attempts",
        "nominal_captures",
        "fresh_attempts",
        "fresh_captures",
    ):
        checks.equal(
            result["metrics"].get(key),
            original_result["metrics"].get(key),
            cell_id + "/saved parent cohort/" + key,
        )
    checks.require(
        not (Path(path) / "shots.jsonl").exists(),
        cell_id + "/parent wrapper has no new trial file",
    )
    return original_result


def inherited_validation(cell, parent, checks):
    """Return parent's already audited final rates without new trial credit."""
    selected = cell["selected"]
    previous = parent["cells"][cell["id"]]
    checks.require(
        previous.get("selected") is not None,
        cell["id"] + "/inherited selected recipe exists",
    )
    checks.equal(
        selected["configuration_sha256"],
        previous["selected"]["configuration_sha256"],
        cell["id"] + "/inherited frozen configuration",
    )
    validation = previous.get("frozen_validation")
    checks.require(
        validation is not None,
        cell["id"] + "/parent independent repeat evidence exists",
    )
    checks.equal(
        validation["configuration_sha256"],
        selected["configuration_sha256"],
        cell["id"] + "/parent final rates exact configuration",
    )
    return validation


def followup_targets(parent_report, criterion, checks):
    expected = dict(
        nominal_required=3,
        nominal_attempts=3,
        fresh_required=27,
        fresh_attempts=30,
        missing_or_blocked=True,
    )
    checks.equal(
        criterion, expected, "Prespecified fragile/unresolved target criterion"
    )
    targets = []
    for cell in parent_report["cells"]:
        validation = cell.get("frozen_validation") or {}
        nominal, fresh = validation.get("nominal", {}), validation.get("fresh", {})
        if (
            not cell.get("selected")
            or validation.get("status") != "completed"
            or nominal.get("captures") != 3
            or nominal.get("attempts") != 3
            or fresh.get("attempts") != 30
            or fresh.get("captures") is None
            or fresh["captures"] < 27
        ):
            targets.append(cell["id"])
    return targets


def check_followup_contract(study, contract, parent, checks):
    protocol = contract["protocol"]
    checks.require(
        protocol.get("inherited_baseline") is True, "Explicit inherited-baseline scope"
    )
    checks.equal(
        protocol.get("frozen_scope"),
        "new_selected_only",
        "New independent repeats scope",
    )
    checks.equal(
        protocol.get("selected_shot_rule"),
        SELECTED_SHOT_RULE,
        "Declared best-cohort shot selection",
    )
    targets = followup_targets(
        parent["report"], protocol.get("target_selection"), checks
    )
    checks.equal(
        protocol.get("targeted_cell_ids"),
        targets,
        "Every unresolved or fragile parent cell is targeted",
    )
    hints = read(Path(study) / "parent-hints.json")
    checks.equal(
        set(hints), set(targets), "Warm-start map includes every targeted cell"
    )
    checks.equal(
        digest(hints), contract["parent_hints_sha256"], "Parent geometry hints identity"
    )
    row_manifest = read(Path(study) / "parent-row-manifest.json")
    checks.equal(
        row_manifest, contract["parent_row_manifest"], "Parent row manifest identity"
    )
    for name, expected in row_manifest.items():
        checks.require(
            any(name.startswith("cells/" + cell + "/") for cell in targets)
            and name.endswith("/shots.jsonl")
            and "/frozen-repeat/" not in name,
            "Only declared parent development rows become warm starts",
        )
        checks.equal(
            parent["manifest"].get(name),
            expected,
            "Parent warm-start file identity/" + name,
        )
    layouts = {row["id"]: row for row in grid_layouts()}
    allowed = {
        "source_id",
        "source_namespace",
        "source_record_sha256",
        "values",
        "q10",
        "q14",
        "bucket_xy_m",
        "use",
    }
    for cell_id, rows in hints.items():
        for hint in rows:
            checks.equal(
                set(hint), allowed, cell_id + "/geometry-only parent hint fields"
            )
            checks.equal(
                hint["source_namespace"],
                "audited_parent_geometry_only",
                cell_id + "/parent hint namespace",
            )
            checks.equal(
                hint["bucket_xy_m"],
                layouts[cell_id]["bucket_xy_m"],
                cell_id + "/parent hint exact layout",
            )
            for key, size in (("values", 5), ("q10", 6), ("q14", 6)):
                array = np.asarray(hint[key], float)
                checks.require(
                    array.shape == (size,) and np.isfinite(array).all(),
                    cell_id + "/finite parent hint/" + key,
                )
            source_hash = hint["source_record_sha256"]
            checks.require(
                isinstance(source_hash, str)
                and len(source_hash) == 64
                and all(c in "0123456789abcdef" for c in source_hash),
                cell_id + "/parent source record hash",
            )
    return targets


def check_diversity(path, geometry, shots, metrics, checks, label):
    """All proposals precede native selection; broad quota cannot be a prefix."""
    path = Path(path)
    selection = read(path / "diversity.json")
    proposals = read(path / "proposals.json")
    n = metrics["search_cpu_candidates"]
    checked = geometry[:n]
    checks.equal(n, len(proposals["values"]), label + "/every proposal audited")
    checks.equal(len(proposals["metadata"]), n, label + "/proposal metadata count")
    checks.equal(selection["planned_cpu_candidates"], n, label + "/planned candidates")
    checks.equal(selection["checked_cpu_candidates"], n, label + "/checked candidates")
    checks.require(
        selection["all_proposals_audited"] is True, label + "/full pool audited"
    )
    for row, value, meta in zip(checked, proposals["values"], proposals["metadata"]):
        checks.equal(row["values"], value, label + "/geometry proposal values")
        checks.equal(row["proposal"], meta, label + "/geometry proposal metadata")
    checks.equal(
        metrics["search_geometry_rejected"],
        sum(not row["geometry_eligible"] for row in checked),
        label + "/search geometry rejects",
    )
    indices = selection["selected_indices"]
    checks.require(
        all(
            isinstance(i, int) and not isinstance(i, bool) and 0 <= i < n
            for i in indices
        ),
        label + "/selection indices",
    )
    checks.equal(len(indices), len(set(indices)), label + "/unique selected candidates")
    chosen = [checked[i] for i in indices]
    checks.require(
        all(row["geometry_eligible"] for row in chosen),
        label + "/selected admitted candidates",
    )
    checks.equal(
        selection["selected_ids"],
        [row["id"] for row in chosen],
        label + "/selected IDs",
    )
    checks.equal(
        selection["selection_sha256"],
        digest(chosen),
        label + "/exact geometry selection hash",
    )
    native = [row for row in shots if row["phase"] == "search"]
    checks.equal(
        [row["id"] for row in native],
        selection["selected_ids"],
        label + "/native selection IDs",
    )
    for row, original in zip(native, chosen):
        checks.equal(
            signature(row), signature(original), label + "/selected native branches"
        )
    eligible = [row for row in checked if row["geometry_eligible"]]
    checks.equal(
        len(chosen),
        min(selection["native_budget"], len(eligible)),
        label + "/finite native budget",
    )
    broad_methods = {"broad_space_filling", "inverse_ballistic_ring"}
    broad_available = sum(
        row.get("proposal", {}).get("method") in broad_methods for row in eligible
    )
    broad_selected = sum(
        row.get("proposal", {}).get("method") in broad_methods for row in chosen
    )
    checks.equal(
        selection["available_broad"], broad_available, label + "/eligible broad count"
    )
    checks.equal(
        selection["selected_broad"], broad_selected, label + "/selected broad count"
    )
    checks.require(
        broad_selected >= min(selection["requested_broad_quota"], broad_available),
        label + "/native broad family quota",
    )
    checks.require(selection["quota_satisfied"] is True, label + "/quota evidence")
    for key, rows in (
        ("checked", checked),
        ("eligible", eligible),
        ("selected", chosen),
    ):
        summary = selection[key]
        checks.equal(summary["count"], len(rows), label + "/range count/" + key)
        if rows:
            values = np.asarray([row["values"] for row in rows], float)
            for i, name in enumerate(
                ("board_x_m", "board_y_m", "board_height_m", "tilt_deg", "gap_m")
            ):
                checks.equal(
                    summary["ranges"][name],
                    [float(values[:, i].min()), float(values[:, i].max())],
                    label + "/actual parameter range/" + key + "/" + name,
                )
    return dict(
        checked=n,
        native_selected=len(chosen),
        broad_available=broad_available,
        broad_selected=broad_selected,
        selection_sha256=selection["selection_sha256"],
    )


def native_rank(row):
    """Reconstruct the declared search-source ordering from scalar-audited data."""
    valid = bool(row.get("native_release")) and not row.get("invalid")
    captured = valid and bool(row.get("route_success"))
    distance = row.get("closest_opening_distance_m", 2.0)
    if not isinstance(distance, (int, float)) or not np.isfinite(distance):
        distance = 2.0
    return (
        captured,
        int(row.get("route_quality", 0)) if captured else 0,
        valid and bool(row.get("board_bounce")),
        -float(distance),
        str(row.get("id", "")),
    )


def qualification_score(cohort, rows):
    return (
        sum(map(capture, cohort["fresh"])),
        sum(capture(row) and row.get("route_quality") == 2 for row in cohort["fresh"]),
        native_rank(rows[cohort["source_id"]]),
    )


def check_selected_qualification(result, cohorts, rows, rule, checks, label):
    """The final recipe uses the best tested cohort, without final-test feedback."""
    checks.equal(rule, SELECTED_SHOT_RULE, label + "/predeclared cohort selection rule")
    candidates = sorted(
        [row for row in rows.values() if row.get("phase") == "search" and capture(row)],
        key=native_rank,
        reverse=True,
    )[: rule["candidate_limit"]]
    checks.require(
        len(cohorts) <= len(candidates), label + "/candidate qualification limit"
    )
    checks.equal(
        [cohort["source_id"] for cohort in cohorts],
        [row["id"] for row in candidates[: len(cohorts)]],
        label + "/tested native-ranked candidate prefix",
    )

    def qualified(cohort):
        return (
            len(cohort["nominal"]) == 3
            and all(map(capture, cohort["nominal"]))
            and len(cohort["fresh"]) == 20
            and sum(map(capture, cohort["fresh"])) >= 18
        )

    def perfect(cohort):
        return qualified(cohort) and qualification_score(cohort, rows)[:2] == (20, 20)

    if len(cohorts) < len(candidates):
        checks.require(
            bool(cohorts) and perfect(cohorts[-1]),
            label + "/early stop requires perfect clean repeats",
        )
    eligible = [cohort for cohort in cohorts if qualified(cohort)]
    checks.equal(
        bool(result.get("success")),
        bool(eligible),
        label + "/qualified result availability",
    )
    if not eligible:
        return None
    best = max(eligible, key=lambda cohort: qualification_score(cohort, rows))
    config = result["configuration"]
    checks.equal(config["q10"], best["source"]["q10"], label + "/best-cohort q10")
    checks.equal(config["q14"], best["source"]["q14"], label + "/best-cohort q14")
    checks.equal(
        [config["tilt_deg"], config["gap_m"]],
        best["source"]["values"][3:],
        label + "/best-cohort exact action",
    )
    measured = best["nominal"][0]
    checks.equal(
        config["board_center"],
        measured["board_center_m"],
        label + "/best-cohort board pose",
    )
    checks.equal(
        config["release_center"],
        measured["release_ball_center_m"],
        label + "/best-cohort release pose",
    )
    for prefix in ("nominal", "fresh"):
        counts = cohort_counts(best[prefix])
        for name in ("captures", "attempts"):
            checks.equal(
                result["metrics"][prefix + "_" + name],
                counts[name],
                label + "/best-cohort displayed " + prefix + " " + name,
            )
    checks.equal(
        result["metrics"]["fresh_clean_captures"],
        qualification_score(best, rows)[1],
        label + "/best-cohort displayed clean captures",
    )
    checks.equal(
        result["metrics"]["fresh_assisted_captures"],
        sum(capture(row) and row.get("route_quality") == 1 for row in best["fresh"]),
        label + "/best-cohort displayed assisted captures",
    )
    return dict(
        selected_source_id=best["source_id"],
        tested_candidates=len(cohorts),
        fresh_captures=qualification_score(best, rows)[0],
        fresh_clean_captures=qualification_score(best, rows)[1],
    )


def check_motion(proof, source, bucket, home_sha, checks, label):
    if not proof.get("passed"):
        return False
    checks.equal(proof.get("common_home_sha256"), home_sha, label + "/home identity")
    record = proof.get("report", {})
    request = record.get("request", {})
    for name in ("values", "q10", "q14"):
        checks.equal(request.get(name), source[name], label + "/" + name)
    checks.equal(request.get("bucket_xy_m"), bucket, label + "/bucket")
    checks.equal(
        proof.get("request_sha256"), digest(request), label + "/request identity"
    )
    endpoint = record.get("endpoint", {})
    checks.equal(endpoint.get("reason"), 0, label + "/native endpoint")
    checks.require(
        endpoint.get("strict_geometry", {}).get("passed") is True,
        label + "/strict geometry",
    )
    checks.require(
        record.get("home_geometry", {}).get("passed") is True, label + "/home clearance"
    )
    checks.require(
        record.get("minimum_clearance_lower_bound_m", -1) >= 0.005 - 1e-10,
        label + "/path clearance",
    )
    checks.equal(
        record.get("exact_selected_joint_max_error_rad"),
        0.0,
        label + "/selected exact branch",
    )
    checks.equal(
        record.get("returned_home_joint_max_error_rad"),
        0.0,
        label + "/home exact branch",
    )
    return True


def check_local_path(proof, variation, bucket, nominal_proof, checks, label):
    if not proof.get("passed"):
        return False
    request = proof.get("request", {})
    expected = dict(
        bucket_xy_m=bucket, **{k: variation[k] for k in ("values", "q10", "q14")}
    )
    checks.equal(request, expected, label + "/exact variation")
    checks.equal(
        proof.get("request_sha256"), digest(request), label + "/request identity"
    )
    checks.equal(
        proof.get("nominal_request_sha256"),
        nominal_proof["request_sha256"],
        label + "/nominal identity",
    )
    checks.equal(
        proof.get("common_home_sha256"),
        nominal_proof["common_home_sha256"],
        label + "/home identity",
    )
    endpoint = proof.get("endpoint", {})
    checks.equal(endpoint.get("reason"), 0, label + "/native endpoint")
    checks.require(
        endpoint.get("strict_geometry", {}).get("passed") is True,
        label + "/strict geometry",
    )
    edges = proof.get("local_edges", [])
    checks.require(
        len(edges) == 4 and all(e.get("passed") for e in edges),
        label + "/four local edges",
    )
    checks.require(
        proof.get("minimum_clearance_lower_bound_m", -1) >= 0.005 - 1e-10,
        label + "/local clearance",
    )
    checks.equal(
        proof.get("returned_nominal_joint_max_error_rad"),
        0.0,
        label + "/reverse exact branch",
    )
    return True


def check_confirmation(record, rows, geometry, bucket, home_sha, checks, label):
    """Check candidate-specific IDs, exact actions, RNG draws and path proofs."""
    source = dict(
        values=record["source_values"],
        q10=record["source_q10"],
        q14=record["source_q14"],
    )
    purpose = record["purpose"]
    checks.require(purpose in {"qualification", "frozen"}, label + "/purpose")
    checks.equal(record.get("perturbation_std"), NOISE.tolist(), label + "/noise")
    motion_ok = check_motion(record["motion"], source, bucket, home_sha, checks, label)
    nominal = [rows[name] for name in record["nominal"]]
    fresh = [rows[name] for name in record["fresh"]]
    checks.equal(
        len(set(record["nominal"] + record["fresh"])),
        len(nominal) + len(fresh),
        label + "/unique rows",
    )
    if not motion_ok:
        checks.require(
            not nominal and not fresh, label + "/no release after rejected staging"
        )
        checks.equal(
            record["qualified"], False, label + "/rejected staging qualification"
        )
        return dict(
            source=source,
            source_id=record["source_id"],
            nominal=nominal,
            fresh=fresh,
            purpose=purpose,
        )
    checks.equal(len(nominal), 3, label + "/nominal denominator")
    for index, row in enumerate(nominal):
        checks.equal(
            row["id"],
            record["source_id"] + f"-{purpose}-nominal-{index}",
            label + "/nominal ID",
        )
        checks.equal(
            signature(row),
            signature(source),
            label + "/nominal exact action and branches",
        )
        checks.equal(row["phase"], "nominal", label + "/nominal phase")
    checks.equal(
        record.get("nominal_counts"), cohort_counts(nominal), label + "/nominal counts"
    )
    expected_count = 30 if purpose == "frozen" else 20
    if not all(map(capture, nominal)) and purpose == "qualification":
        checks.require(not fresh, label + "/nominal failure does not qualify")
        checks.equal(
            record["qualified"], False, label + "/failed nominal qualification"
        )
        return dict(
            source=source,
            source_id=record["source_id"],
            nominal=nominal,
            fresh=fresh,
            purpose=purpose,
        )
    checks.equal(len(fresh), expected_count, label + "/fresh denominator")
    variations = record.get("variations", [])
    checks.equal(len(variations), expected_count, label + "/variation proof count")
    generated = np.asarray(source["values"]) + np.random.default_rng(
        record["seed"]
    ).normal(0, NOISE, (expected_count, 5))
    known_geometry = {signature(row) for row in geometry}
    for index, (row, variation) in enumerate(zip(fresh, variations)):
        expected_id = record["source_id"] + f"-{purpose}-fresh-{index}"
        checks.equal(row["id"], expected_id, label + "/fresh ID")
        checks.equal(variation["id"], expected_id, label + "/proof ID")
        checks.equal(row["phase"], "fresh", label + "/fresh phase")
        checks.equal(
            variation["values"],
            generated[index].tolist(),
            label + "/seeded fresh values",
        )
        checks.equal(
            signature(row), signature(variation), label + "/fresh exact branch binding"
        )
        checks.require(
            signature(variation) in known_geometry, label + "/fresh geometry evidence"
        )
        admitted = variation.get("admitted") is True
        checks.equal(
            bool(row.get("native_release")), admitted, label + "/native admission"
        )
        if admitted:
            checks.require(
                check_local_path(
                    variation["local_path"],
                    variation,
                    bucket,
                    record["motion"],
                    checks,
                    label + f"/variation-{index}",
                ),
                label + "/local path admitted",
            )
        else:
            checks.require(
                not capture(row), label + "/rejected variation cannot capture"
            )
    checks.equal(
        record.get("fresh_counts"), cohort_counts(fresh), label + "/fresh counts"
    )
    quality = dict(
        clean=sum(capture(r) and r.get("route_quality") == 2 for r in fresh),
        assisted=sum(capture(r) and r.get("route_quality") == 1 for r in fresh),
        cohort="fresh",
    )
    checks.equal(record.get("quality"), quality, label + "/quality counts")
    qualified = (
        all(map(capture, nominal))
        and len(fresh) == 20
        and sum(map(capture, fresh)) >= 18
    )
    checks.equal(record["qualified"], qualified, label + "/qualification")
    return dict(
        source=source,
        source_id=record["source_id"],
        nominal=nominal,
        fresh=fresh,
        purpose=purpose,
    )


def rescore_rows(directory, rows, scene, checks, label):
    """Rescore every saved native trace and reject unused/misassigned columns."""
    from .contract import ScoreStream, FORCE_NAMES

    directory = Path(directory)
    traces, by_file = {}, {}
    for row in rows:
        if not row.get("native_release"):
            checks.require(not capture(row), label + "/rejected row cannot capture")
            continue
        checks.require(not row.get("invalid"), label + "/released invalid row")
        filename, index = row["native_cache_file"], row["native_cache_row"]
        checks.require(
            isinstance(index, int) and not isinstance(index, bool) and index >= 0,
            label + "/trace row index",
        )
        by_file.setdefault(filename, []).append((index, row))
    for filename, members in by_file.items():
        with np.load(contained(directory, filename), allow_pickle=False) as arrays:
            ids = arrays["record_ids"].tolist()
            checks.require(len(ids) == len(set(ids)), label + "/unique trace IDs")
            checks.equal(
                {i for i, _ in members},
                set(range(len(ids))),
                label + "/all trace columns used exactly once",
            )
            checks.equal(len(members), len(ids), label + "/no duplicated trace columns")
            for name, width in (
                ("position_m", 3),
                ("velocity_m_s", 3),
                ("forces_N", len(FORCE_NAMES)),
            ):
                a = arrays[name]
                checks.equal(
                    a.shape, (960, len(ids), width), label + "/trace dimensions/" + name
                )
                checks.require(np.isfinite(a).all(), label + "/finite trace/" + name)
            checks.require(
                (arrays["forces_N"] >= 0).all(), label + "/nonnegative force norms"
            )
            for index, row in members:
                row_label = label + "/" + row["id"]
                checks.equal(str(ids[index]), row["id"], row_label + "/trace ID")
                scorer = ScoreStream(scene, row["bucket_xy_m"], row["board_normal"])
                for p, v, f in zip(
                    arrays["position_m"][:, index],
                    arrays["velocity_m_s"][:, index],
                    arrays["forces_N"][:, index],
                ):
                    scorer.step(p, v, dict(zip(FORCE_NAMES, f)))
                measured = scorer.result()
                for name in SCORE_FIELDS:
                    checks.equal(
                        measured[name], row[name], row_label + "/score/" + name
                    )
                for name in FLOAT_FIELDS:
                    if name in row or name in measured:
                        checks.numeric(
                            measured.get(name),
                            row.get(name),
                            row_label + "/score/" + name,
                        )
                checks.require(
                    row.get("scalar_score_audit_passed") is True,
                    row_label + "/recorded scalar check",
                )
                traces[row["id"]] = arrays["position_m"][3::4, index].tolist()
    checks.equal(
        set(by_file),
        {p.name for p in directory.glob("native-*.npz")},
        label + "/no unassigned native arrays",
    )
    return traces


def audit(study, reference, assets, *, repository=None, recheck_frozen_geometry=False):
    from .grid_study_contract import make_report, validate_report
    from .sweep_worker import load_reference

    begin = time.monotonic()
    study, assets = Path(study).resolve(), Path(assets).resolve()
    repository = Path(repository or Path(__file__).resolve().parents[2]).resolve()
    checks = Checks()
    manifest = check_manifest(study, checks)
    contract, ledger, report = (
        read(study / name) for name in ("contract.json", "ledger.json", "report.json")
    )
    unsigned = dict(contract)
    fingerprint = unsigned.pop("fingerprint")
    checks.equal(digest(unsigned), fingerprint, "Contract fingerprint")
    checks.equal(
        report["status"], "completed", "Audit requires a completed immutable study"
    )
    ref = load_reference(reference, assets, repository=repository)
    identity = contract["identity"]
    checks.equal(identity["physics"], PHYSICS, "Physics contract")
    checks.equal(
        identity["contract_fingerprint"],
        ref["contract"]["fingerprint"],
        "Parent contract",
    )
    checks.equal(identity["scene_sha256"], digest(ref["scene"]), "Scene identity")
    checks.equal(identity["asset_sha256"], ref["asset_hashes"], "Asset identities")
    checks.equal(contract["packages"], ref["packages"], "Pinned native dependencies")
    for name, expected in identity["source_sha256"].items():
        checks.equal(
            sha(contained(repository, name)), expected, "Frozen source: " + name
        )
    for name, expected in ref["source_hashes"].items():
        checks.equal(
            identity["source_sha256"].get(name), expected, "Native dependency: " + name
        )
    parent = parent_snapshot(study, contract, identity, checks)
    followup_cell_ids = (
        check_followup_contract(study, contract, parent, checks)
        if parent is not None
        else None
    )
    raw_hints_path = Path(reference) / "shots.jsonl"
    checks.equal(
        sha(raw_hints_path),
        ref["public"]["artifacts_sha256"]["shots.jsonl"],
        "Historical hint file",
    )
    hints = [
        geometry_hint(item["source"], "qualified_reference_geometry")
        for item in ref["catalogue"]["sources"]
        if item.get("qualified")
    ]
    for line in raw_hints_path.read_text().splitlines():
        row = json.loads(line)
        if (
            row.get("phase") == "search"
            and row.get("native_release")
            and row.get("route_success")
            and not row.get("invalid")
        ):
            hints.append(geometry_hint(row, "corrected_grid_search"))
    checks.equal(
        digest([h for h in hints if h is not None]),
        contract["hints_sha256"],
        "Geometry hint identities",
    )
    validate_report(report)
    expected_report = make_report(
        ledger["attempts"],
        identity=identity,
        protocol=contract["protocol"],
        status=report["status"],
        states=ledger["states"],
        started_at=ledger["started_at"],
        updated_at=report["updated_at"],
        validation_by_cell=ledger["validation"],
    )
    checks.equal(
        report, expected_report, "Display is exactly reconstructed from its ledger"
    )
    selected = {c["id"]: c["selected"] for c in report["cells"] if c.get("selected")}
    if followup_cell_ids is not None:
        for layout in grid_layouts():
            stages = [a["stage"] for a in ledger["attempts"].get(layout["id"], [])]
            checks.require(
                set(stages).issubset({"baseline", "expanded"}),
                layout["id"] + "/only declared follow-up stages",
            )
            checks.equal(
                stages.count("baseline"), 1, layout["id"] + "/one archived baseline"
            )
            checks.equal(
                stages.count("expanded"),
                int(layout["id"] in followup_cell_ids),
                layout["id"] + "/complete prespecified follow-up target coverage",
            )
    checks.equal(
        read(study / "frozen-selection.json"),
        selected,
        "No reselection after frozen validation",
    )
    checks.equal(
        contract["protocol"]["grid"], grid_layouts(), "All35 distinct fixed targets"
    )
    checks.equal(
        contract["protocol"]["qualification"],
        dict(nominal=3, fresh=20, required_fresh=18),
        "Qualification budgets",
    )
    checks.equal(
        contract["protocol"]["frozen_validation"]["nominal"],
        3,
        "Independent nominal budget",
    )
    checks.equal(
        contract["protocol"]["frozen_validation"]["fresh"],
        30,
        "Independent fresh budget",
    )
    checks.equal(
        contract["protocol"]["perturbation_std"],
        NOISE.tolist(),
        "Perturbation contract",
    )
    home_sha = ref["contract"]["common_home"]["sha256"]
    all_ids, totals, cell_reports = set(), Counter(), []
    accounted_shot_files = set()
    admission = None
    if recheck_frozen_geometry:
        from .corrected_admission import TransitAdmission

        admission = TransitAdmission(
            ref["scene"], assets, home=ref["contract"]["common_home"]
        )
    for cell in report["cells"]:
        cell_id, bucket = cell["id"], cell["bucket_xy_m"]
        cell_report = dict(
            id=cell_id,
            bucket_xy_m=bucket,
            stage_checks=[],
            native_rows=0,
            frozen_geometry_rechecked=0,
        )
        for attempt in ledger["attempts"].get(cell_id, []):
            stage = attempt["stage"]
            path = (
                study
                / "cells"
                / cell_id
                / ("saved" if stage == "saved_reference" else stage)
            )
            stage_label = cell_id + "/" + stage
            stage_manifest = check_manifest(path, checks)
            checks.equal(
                stage_manifest,
                attempt["evidence_sha256"],
                stage_label + "/ledger manifest",
            )
            request, result = read(path / "request.json"), read(path / "result.json")
            checks.equal(request, attempt["request"], stage_label + "/request payload")
            checks.equal(result, attempt["result"], stage_label + "/result payload")
            checks.equal(
                sha(path / "result.json"),
                attempt["result_sha256"],
                stage_label + "/result file identity",
            )
            checks.equal(
                result["provenance"]["study_fingerprint"],
                fingerprint,
                stage_label + "/study identity",
            )
            checks.equal(
                result["provenance"]["request_sha256"],
                digest(
                    dict(request=request, study_fingerprint=fingerprint, stage=stage)
                ),
                stage_label + "/request identity",
            )
            if result["provenance"].get("parent_evidence") is not None:
                checks.require(
                    parent is not None, stage_label + "/parent snapshot required"
                )
                check_parent_wrapper(path, attempt, parent, checks, cell_id)
                cell_report["stage_checks"].append(
                    dict(
                        stage=stage,
                        requests=0,
                        native_releases=0,
                        inherited_parent=True,
                    )
                )
                continue
            if stage == "saved_reference":
                checks.require(
                    result["provenance"].get("saved_evidence") is True,
                    stage_label + "/saved provenance",
                )
                checks.equal(
                    result["metrics"].get("search_requests"),
                    0,
                    stage_label + "/no new saved search trials",
                )
                continue
            shots = (
                [
                    json.loads(line)
                    for line in (path / "shots.jsonl").read_text().splitlines()
                ]
                if (path / "shots.jsonl").exists()
                else []
            )
            if (path / "shots.jsonl").exists():
                accounted_shot_files.add(path / "shots.jsonl")
            rows = {row["id"]: row for row in shots}
            checks.equal(len(rows), len(shots), stage_label + "/unique shot IDs")
            checks.require(
                not (all_ids & set(rows)), stage_label + "/global unique shot IDs"
            )
            all_ids.update(rows)
            for row in shots:
                checks.equal(row["bucket_xy_m"], bucket, stage_label + "/fixed bucket")
            geometry = (
                read(path / "geometry.json")
                if (path / "geometry.json").exists()
                else []
            )
            traces = rescore_rows(path, shots, ref["scene"], checks, stage_label)
            metrics = result["metrics"]
            search = [r for r in shots if r["phase"] == "search"]
            checks.equal(
                metrics["search_requests"],
                len(search),
                stage_label + "/search request count",
            )
            checks.equal(
                metrics["confirmation_requests"],
                len(shots) - len(search),
                stage_label + "/all confirmation requests",
            )
            checks.equal(
                metrics["search_captures"],
                sum(map(capture, search)),
                stage_label + "/search captures",
            )
            checks.equal(
                metrics["cpu_candidates"],
                len(geometry),
                stage_label + "/all CPU admission counts",
            )
            checks.equal(
                metrics["geometry_rejected"],
                sum(not r["geometry_eligible"] for r in geometry),
                stage_label + "/endpoint rejects",
            )
            if (
                parent is not None
                and stage == "expanded"
                and result["status_native"] != "common_home_unresolved"
            ):
                checks.require(
                    (path / "diversity.json").is_file(),
                    stage_label + "/required diverse selection evidence",
                )
                checks.equal(
                    metrics["search_cpu_candidates"],
                    request["cpu_candidates"],
                    stage_label + "/entire expanded geometry budget checked",
                )
            if (path / "diversity.json").exists():
                cell_report["diversity"] = check_diversity(
                    path, geometry, shots, metrics, checks, stage_label
                )
                selection = read(path / "diversity.json")
                expanded_protocol = next(
                    item
                    for item in contract["protocol"]["stages"]
                    if item["name"] == "expanded"
                )
                checks.equal(
                    selection["requested_broad_quota"],
                    expanded_protocol["broad_native_quota"],
                    stage_label + "/predeclared broad native quota",
                )
                checks.equal(
                    selection["method"],
                    contract["protocol"]["selection_method"],
                    stage_label + "/predeclared diversity selection",
                )
                checks.equal(
                    selection["native_budget"],
                    request["search_budget"],
                    stage_label + "/selection native budget",
                )
                checks.equal(
                    selection["seed"],
                    contract["protocol"]["seed"]
                    + (cell["row"] * 7 + cell["column"]) * 100,
                    stage_label + "/selection seed",
                )
            parsed = []
            for ordinal, confirmation_path in enumerate(
                sorted(path.glob("qualification-*.json"))
            ):
                if (path / "diversity.json").exists():
                    checks.equal(
                        read(confirmation_path)["seed"],
                        selection["seed"] + 10000 + ordinal,
                        stage_label + "/independent candidate qualification seed",
                    )
                parsed.append(
                    check_confirmation(
                        read(confirmation_path),
                        rows,
                        geometry,
                        bucket,
                        home_sha,
                        checks,
                        stage_label + "/" + confirmation_path.name,
                    )
                )
            if (
                parent is not None
                and stage == "expanded"
                and (path / "diversity.json").exists()
            ):
                cell_report["candidate_selection"] = check_selected_qualification(
                    result,
                    parsed,
                    rows,
                    contract["protocol"]["selected_shot_rule"],
                    checks,
                    stage_label,
                )
            if result["configuration"]:
                config = result["configuration"]
                matches = [
                    r
                    for r in shots
                    if r.get("native_release")
                    and r["q10"] == config["q10"]
                    and r["q14"] == config["q14"]
                    and r["values"][3:] == [config["tilt_deg"], config["gap_m"]]
                    and r["board_center_m"] == config["board_center"]
                    and r["release_ball_center_m"] == config["release_center"]
                ]
                checks.require(
                    bool(matches), stage_label + "/configuration native source"
                )
                displayed = [x["ball"] for x in result["trajectory"]]
                checks.require(
                    any(displayed == traces[r["id"]] for r in matches),
                    stage_label + "/display trajectory",
                )
                if result.get("success"):
                    matching = [
                        p
                        for p in parsed
                        if signature(p["source"]) == signature(matches[0])
                        and len(p["fresh"]) == 20
                        and all(map(capture, p["nominal"]))
                        and sum(map(capture, p["fresh"])) >= 18
                    ]
                    checks.require(
                        bool(matching), stage_label + "/selected qualified cohort"
                    )
            for row in shots:
                totals["native_releases"] += bool(row.get("native_release"))
                totals["requested_rows"] += 1
                totals["captures"] += capture(row)
            cell_report["native_rows"] += len(traces)
            cell_report["stage_checks"].append(
                dict(
                    stage=stage,
                    requests=len(shots),
                    native_releases=len(traces),
                    confirmation_sets=len(parsed),
                )
            )
        frozen_path = study / "cells" / cell_id / "frozen-repeat"
        if (
            cell_id in selected
            and parent is not None
            and selected[cell_id]["source"] == "saved_reference"
        ):
            checks.require(
                not frozen_path.exists(),
                cell_id + "/inherited recipe has no duplicate new repeat run",
            )
            checks.require(
                cell_id not in ledger["validation"],
                cell_id + "/inherited final rates are not new validation",
            )
            previous = inherited_validation(cell, parent, checks)
            cell_report["parent_frozen"] = previous
            totals["inherited_selected"] += 1
        elif cell_id in selected:
            check_manifest(frozen_path, checks)
            validation = read(frozen_path / "validation.json")
            checks.equal(
                validation,
                ledger["validation"][cell_id],
                cell_id + "/frozen validation ledger",
            )
            checks.equal(
                validation["configuration_sha256"],
                selected[cell_id]["configuration_sha256"],
                cell_id + "/frozen config",
            )
            expected_seed = (
                contract["protocol"]["frozen_validation"]["seed"]
                + cell["row"] * 7
                + cell["column"]
            )
            checks.equal(
                validation["seed"], expected_seed, cell_id + "/frozen independent seed"
            )
            files = list(frozen_path.glob("frozen-*.json"))
            checks.equal(len(files), 1, cell_id + "/one frozen cohort; no fallback")
            shots = (
                [
                    json.loads(line)
                    for line in (frozen_path / "shots.jsonl").read_text().splitlines()
                ]
                if (frozen_path / "shots.jsonl").exists()
                else []
            )
            if (frozen_path / "shots.jsonl").exists():
                accounted_shot_files.add(frozen_path / "shots.jsonl")
            rows = {row["id"]: row for row in shots}
            checks.equal(len(rows), len(shots), cell_id + "/frozen unique shot IDs")
            checks.require(not (all_ids & set(rows)), cell_id + "/frozen IDs disjoint")
            all_ids.update(rows)
            geometry = (
                read(frozen_path / "geometry.json")
                if (frozen_path / "geometry.json").exists()
                else []
            )
            record = read(files[0])
            checks.equal(record["seed"], expected_seed, cell_id + "/frozen actual seed")
            parsed = check_confirmation(
                record, rows, geometry, bucket, home_sha, checks, cell_id + "/frozen"
            )
            checks.equal(
                validation.get("nominal"),
                (
                    cohort_counts(parsed["nominal"])
                    if parsed["nominal"]
                    else dict(captures=None, attempts=None)
                ),
                cell_id + "/frozen nominal rate",
            )
            checks.equal(
                validation.get("fresh"),
                (
                    cohort_counts(parsed["fresh"])
                    if parsed["fresh"]
                    else dict(captures=None, attempts=None)
                ),
                cell_id + "/frozen fresh rate",
            )
            traces = rescore_rows(
                frozen_path, shots, ref["scene"], checks, cell_id + "/frozen"
            )
            checks.equal(
                set(rows),
                set(record["nominal"] + record["fresh"]),
                cell_id + "/all frozen requests accounted",
            )
            chosen_attempt = next(
                a
                for a in ledger["attempts"][cell_id]
                if a["request"]["request_id"] == selected[cell_id]["request_id"]
            )
            config = chosen_attempt["result"]["configuration"]
            checks.equal(
                record["source_q10"], config["q10"], cell_id + "/frozen q10 selection"
            )
            checks.equal(
                record["source_q14"], config["q14"], cell_id + "/frozen q14 selection"
            )
            checks.equal(
                record["source_values"][3:],
                [config["tilt_deg"], config["gap_m"]],
                cell_id + "/frozen tilt gap",
            )
            if chosen_attempt["stage"] == "saved_reference":
                source_rows = [item["source"] for item in ref["catalogue"]["sources"]]
            else:
                selected_stage = study / "cells" / cell_id / chosen_attempt["stage"]
                source_rows = [
                    json.loads(line)
                    for line in (selected_stage / "shots.jsonl")
                    .read_text()
                    .splitlines()
                ]
            exact_actions = [
                row["values"]
                for row in source_rows
                if row.get("q10") == config["q10"]
                and row.get("q14") == config["q14"]
                and row["values"][3:] == [config["tilt_deg"], config["gap_m"]]
            ]
            checks.require(
                record["source_values"] in exact_actions,
                cell_id + "/frozen exact original action",
            )
            if "quality" in validation:
                checks.equal(
                    validation["quality"],
                    record["quality"],
                    cell_id + "/frozen validation quality",
                )
            checks.equal(
                validation["status"],
                (
                    "completed"
                    if len(parsed["nominal"]) == 3 and len(parsed["fresh"]) == 30
                    else "blocked"
                ),
                cell_id + "/frozen status from requested denominator",
            )
            if admission is not None and parsed["fresh"]:
                source = dict(
                    parsed["source"], id=record["source_id"], bucket_xy_m=bucket
                )
                for variation in record["variations"]:
                    if variation["admitted"]:
                        query = dict(variation, bucket_xy_m=bucket)
                        checked = admission.connect_variation(source, query)
                        checks.require(
                            checked.get("passed") is True,
                            cell_id + "/independent frozen strict/local path",
                        )
                        cell_report["frozen_geometry_rechecked"] += 1
            cell_report["native_rows"] += len(traces)
            cell_report["frozen"] = dict(
                nominal=validation["nominal"],
                fresh=validation["fresh"],
                configuration_sha256=validation["configuration_sha256"],
            )
            for row in shots:
                totals["frozen_requests"] += 1
                totals["frozen_releases"] += bool(row.get("native_release"))
                totals["frozen_captures"] += capture(row)
        else:
            checks.require(
                not frozen_path.exists(), cell_id + "/no frozen run for unknown recipe"
            )
        cell_reports.append(cell_report)
    checks.equal(
        accounted_shot_files,
        set(study.rglob("shots.jsonl")),
        "Every new trial file belongs to an accounted stage or frozen cohort",
    )
    return dict(
        version=VERSION,
        status="passed",
        checks=checks.count,
        failures=[],
        study_manifest_sha256=sha(study / "manifest.json"),
        contract_fingerprint=fingerprint,
        report_sha256=sha(study / "report.json"),
        frozen_selection_sha256=sha(study / "frozen-selection.json"),
        auditor_sha256=sha(Path(__file__)),
        source_sha256=identity["source_sha256"],
        asset_sha256=identity["asset_sha256"],
        packages=contract["packages"],
        physics=PHYSICS,
        totals=dict(totals),
        parent_evidence=(parent["metadata"] if parent is not None else None),
        cells=cell_reports,
        maximum_numeric_error=dict(checks.float_maximum_error),
        elapsed_s=time.monotonic() - begin,
        scope="Read-only independent scalar rescoring of saved native arrays and evidence/selection/cohort checks; no new physics rollout",
        geometry_scope=(
            "CPU strict endpoints and sampled local paths rechecked for admitted frozen perturbations"
            if recheck_frozen_geometry
            else "Recorded endpoint/local-path proof identities checked; geometry was not independently rerun"
        ),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("study", "reference", "assets", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--repository", type=Path)
    parser.add_argument("--recheck-frozen-geometry", action="store_true")
    args = parser.parse_args()
    if args.output.resolve().is_relative_to(args.study.resolve()):
        raise ValueError("Write the audit outside immutable source evidence")
    try:
        result = audit(
            args.study,
            args.reference,
            args.assets,
            repository=args.repository,
            recheck_frozen_geometry=args.recheck_frozen_geometry,
        )
    except Exception as error:
        result = dict(
            version=VERSION,
            status="failed",
            failures=[str(error)],
            auditor_sha256=sha(Path(__file__)),
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, allow_nan=False, indent=2) + "\n")
    print(
        json.dumps(
            {
                k: result[k]
                for k in ("version", "status", "checks", "failures", "totals")
                if k in result
            }
        )
    )
    if result["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
