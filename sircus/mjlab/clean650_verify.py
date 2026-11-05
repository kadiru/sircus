"""Independent CPU-only audit of the bounded clean 650 mm development sweep.

Every native trace is rescored using the independent calibrated contact oracle.
This checker reconstructs the adaptive search from saved earlier observations;
it never launches dynamics, selects a learned policy, or changes old evidence.
"""
from __future__ import annotations

import argparse
import copy
from collections import Counter
import itertools
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import numpy as np

from .benchmark_verify import EvidenceError, contained, digest, read, sha
from . import replay650_verify as replay

VERSION = "sircus-clean650-independent-audit-v1"
STUDY_VERSION = "sircus-clean650-height-tilt-gap-v1"
HEIGHTS = (.22, .26, .30, .34, .38, .42)
TILTS = (16., 20., 24., 28., 32., 36.)
GAPS = (.20, .24, .28, .32, .36, .40, .44, .48, .52, .56)
STEPS = (64, 128)
MAX_REQUESTS = 496
AuditChecks = replay.AuditChecks


def statistics(rows):
    value = replay.statistics(rows)
    value["geometry_rejections"] = sum(not r.get("native_release") for r in rows)
    value["rim_assisted"] = sum(replay.captured(r) and r.get("rim_band_contact") is True for r in rows)
    return value


def native_valid(row):
    return bool(row.get("native_release") and not row.get("invalid")
                and row.get("invalid_physics") is False)


def clean(row):
    return bool(native_valid(row) and row.get("route_success") and row.get("route_quality") == 2)


def order_key(row):
    """Literal descending quality/margin ranking with a stable ascending ID."""
    margin = row.get("minimum_mouth_clearance_mm")
    margin = float(margin) if isinstance(margin, (int, float)) and math.isfinite(margin) else -1e9
    distance = row.get("closest_opening_distance_m")
    distance = float(distance) if isinstance(distance, (int, float)) and math.isfinite(distance) else 1e6
    return (-int(clean(row)), -int(bool(row.get("route_success"))),
            -int(bool(row.get("whole_ball_entry") and not row.get("external_contact_steps"))),
            -margin, -float(row.get("max_retention_s", 0.)), distance, row["id"])


def select(rows, limit):
    return sorted((r for r in rows if native_valid(r)), key=order_key)[:limit]


def select_unique(rows, limit):
    """Rank earlier outcomes, then keep distinct configured geometry targets.

    Nanometre-scale key rounding collapses floating addition representations;
    executed actions and joint arrays are never rounded or replaced.
    """
    chosen, seen = [], set()
    for row in select(rows, len(rows)):
        key = tuple(round(float(v), 9) for v in [*row["values"], *row["bucket_xy_m"]])
        if key in seen:
            continue
        seen.add(key);chosen.append(row)
        if len(chosen) == limit:
            break
    return chosen


def coarse_values(seed):
    xy = seed["values"][:2]
    return [[*xy, h, t, g] for h, t, g in itertools.product(HEIGHTS, TILTS, GAPS)] + [copy.deepcopy(seed["values"])]


def local_values(source):
    v = source["values"]
    return [[v[0], v[1], round(v[2]+h, 8), round(v[3]+t, 8), round(v[4]+g, 8)]
            for h, t, g in itertools.product((-.02, 0., .02), (-2., 0., 2.), (-.02, 0., .02))]


def control_gate(rows):
    present = len(rows) == 12 and all(
        len([r for r in rows if r["role"] == role and r["substeps"] == step]) == 3
        for role in ("positive_control", "negative_control") for step in STEPS)
    positive = present and all(clean(r) for r in rows if r["role"] == "positive_control")
    negative = present and all(native_valid(r) and not r.get("route_success")
                               for r in rows if r["role"] == "negative_control")
    physics = present and all(native_valid(r) for r in rows)
    oracle = present and all(r.get("scalar_score_audit_passed") is True for r in rows)
    passed = bool(positive and negative and physics and oracle)
    return dict(status="passed" if passed else "failed", positive_control=bool(positive),
                negative_control=bool(negative), physics_valid=bool(physics), oracle_agreement=bool(oracle))


def qualify(rows, positions, *, controls_passed=True, source_clean=True):
    """Nominal clean-at-both-resolutions gate; no noise robustness claim."""
    present = len(rows) == 6 and all(len([r for r in rows if r["substeps"] == s]) == 3 for s in STEPS)
    paired = replay.paired_sensitivity(rows, positions) if present else dict(
        paired_attempts=0, capture_agreement=0, nominal_position_delta_350ms_max_m=None)
    delta = paired["nominal_position_delta_350ms_max_m"]
    clean_both = present and all(clean(r) for r in rows)
    physics = present and all(native_valid(r) for r in rows)
    oracle = present and all(r.get("scalar_score_audit_passed") is True for r in rows)
    passed = bool(controls_passed and source_clean and clean_both and physics and oracle and delta is not None and delta <= .005)
    return dict(status="qualified" if passed else "failed", clean_both=bool(clean_both),
                physics_valid=bool(physics), oracle_agreement=bool(oracle), max_drift_m=delta, qualified=passed)


def audit_geometry(groups, proofs, scene, assets, checks):
    """Recheck every actually tested branch with the strict CPU geometry model."""
    checks.equal(len(groups), len(proofs), "Exactly one geometry proof per requested group")
    checks.equal(len({g["id"] for g in groups}), len(groups), "Unique immutable group identities")
    center = np.asarray(scene["table"]["center_m"][:2]);half = np.asarray(scene["table"]["size_m"][:2])/2
    radius = max(scene["bucket"]["outer_top_diameter_m"], scene["bucket"]["outer_bottom_diameter_m"])/2
    for index, (group, proof) in enumerate(zip(groups, proofs)):
        label = group["id"]
        checks.equal(group["geometry_row"], index, label+"/ordered proof row")
        checks.equal(group["geometry_sha256"], digest(proof), label+"/immutable geometry proof")
        checks.equal(proof["group_id"], label, label+"/exact proof group")
        checks.require(proof["held_endpoints_only"] is True, label+"/endpoint scope explicit")
        for key in ("values", "bucket_xy_m", "eligible"):
            checks.equal(group[key], proof[key], label+"/exact proof input/"+key)
        xy = np.asarray(group["bucket_xy_m"])
        edges = np.r_[xy-(center-half)-radius, (center+half)-xy-radius]
        footprint = proof["bucket_footprint"]
        checks.numeric(footprint["outer_radius_m"], radius, label+"/whole outer bucket radius", 0)
        checks.numeric(footprint["edge_clearance_m"], edges, label+"/actual bucket edge margins", 1e-12)
        checks.numeric(footprint["minimum_edge_clearance_m"], min(edges), label+"/smallest table margin", 1e-12)
        checks.equal(footprint["passed"], bool(min(edges) >= .005-1e-12), label+"/5mm table-edge rule")
        checks.require(footprint["passed"] or not group["eligible"], label+"/off-table groups cannot release")
        passed = [a for a in proof["attempts"] if a["checked"]["reason"] == 0]
        checks.equal(group["eligible"], bool(passed), label+"/endpoint admission has a passing branch")
        checks.require(len(proof["attempts"]) <= 17*17, label+"/bounded branch-pair budget")
        if passed:
            checks.equal(proof["attempts"][-1], passed[0], label+"/first strictly admitted pair selected")
            for key in ("q10", "q14"):
                checks.equal(passed[0][key], group[key], label+"/actual selected joint branch/"+key)
            checks.require(proof["minimum_clearance_lower_bound_m"] >= .005-1e-12,
                           label+"/minimum admitted clearance")
    replay.audit_cpu_endpoints(groups, proofs, scene, assets, checks)


def audit_native_rows(rows, plan, groups, checks):
    """Retain rejections and bind actual initialized joints to exact requests."""
    lookup = {g["id"]: g for g in groups}
    checks.equal(len(rows), len(plan), "All opened requests remain in the denominator")
    checks.equal(len({r["id"] for r in rows}), len(rows), "No hidden duplicate or retried request ID")
    for row, expected in zip(rows, plan):
        label = row["id"]
        for key, value in expected.items():
            checks.equal(row[key], value, label+"/exact frozen request/"+key)
        checks.equal(row["eligible"], lookup[row["group_id"]]["eligible"], label+"/geometry eligibility")
        checks.equal(bool(row["native_release"]), bool(row["eligible"]), label+"/every admitted request released")
        if row["native_release"]:
            checks.equal(row["native_directory"], "native/"+str(row["substeps"]), label+"/actual step directory")
            for key in ("q10", "q14"):
                checks.numeric(row["native_initial_"+key], np.asarray(row[key], dtype=np.float32), label+"/native actual float32 joints/"+key, 0)
            checks.equal(row["contact_protocol"], dict(substeps=row["substeps"], board_mode="articulated", gravity_mode="passive_only"), label+"/held-endpoint native controls")
            checks.require(row.get("scalar_score_audit_passed") is True, label+"/online scalar agreement")
            checks.equal(row["endpoint_collision_pairs"], [], label+"/native endpoint collision check")
        else:
            checks.require(bool(row.get("invalid")) and not row.get("route_success") and not row.get("whole_ball_entry"), label+"/rejected request never becomes capture")
            checks.require("native_cache_file" not in row, label+"/rejected request has no invented native trace")


def auditor_identity(producer_source, producer_hashes, checks):
    """Separate checker identity; imported core dependencies stay source-bound."""
    root = Path(__file__).resolve().parents[2];own = "sircus/mjlab/clean650_verify.py";dependencies = {}
    for module in tuple(sys.modules.values()):
        name = getattr(module, "__name__", "");filename = getattr(module, "__file__", None)
        if not filename or not (name.startswith("sircus.mjlab.") or name.startswith("sircus.isaac.")):
            continue
        path = Path(filename).resolve()
        checks.require(path.is_relative_to(root), "One explicit auditor source snapshot/"+name)
        relative = str(path.relative_to(root))
        if len(Path(relative).parts) != 3 or path.suffix != ".py" or relative == own:
            continue
        if relative.startswith("sircus/isaac/") and relative not in producer_hashes:
            continue
        checks.require(relative in producer_hashes, "Imported core dependency declared by original producer/"+relative)
        actual = sha(path);checks.equal(actual, producer_hashes[relative], "Actual unchanged auditor dependency/"+relative)
        dependencies[relative] = actual
    checks.require("sircus/mjlab/calibrated_score_oracle.py" in dependencies, "Independent scorer identity retained")
    checks.equal(sha(contained(producer_source, own)), producer_hashes[own], "Original producer checker bytes preserved")
    return dict(version=VERSION, auditor_sha256=sha(Path(__file__)), original_producer_auditor_sha256=producer_hashes[own],
                dependencies_sha256=dependencies)


def make_plan(groups, phase, *, steps, repeats):
    """Independent literal order; every geometry reject is still requested."""
    return [dict(id=f'{phase}-{step}-{g["id"]}-r{i}', pair_id=f'{phase}-{g["id"]}-r{i}',
                 group_id=g["id"], role=g["role"], substeps=step, repeat_index=i, phase=phase,
                 values=copy.deepcopy(g["values"]), bucket_xy_m=copy.deepcopy(g["bucket_xy_m"]),
                 q10=copy.deepcopy(g["q10"]), q14=copy.deepcopy(g["q14"]), eligible=g["eligible"],
                 geometry_row=g["geometry_row"])
            for step in steps for g in groups for i in range(repeats)]


def audit_prior_replay(study, bindings, checks):
    expected_names = {"contract.json", "report.json", "audit-replay650.json", "records.jsonl", "evidence-manifest.json"}
    checks.equal(set(bindings), expected_names, "All prior replay inputs explicitly byte-bound")
    for name, expected in bindings.items():
        checks.equal(sha(contained(study, name)), expected, "Exact prior calibrated replay/"+name)
    spec, report, proof = (read(study/name) for name in ("contract.json", "report.json", "audit-replay650.json"))
    unsigned = dict(spec);fp = unsigned.pop("fingerprint", None)
    checks.equal(digest(unsigned), fp, "Prior complete contract fingerprint")
    checks.equal(fp, "8ebbd58394180bbf3540228fd9725608fa0f6fdff2db799036f4f3ce875f348b", "Exact original650 matched replay")
    checks.equal((report["version"], report["status"], report["phase"]),
                 ("sircus-calibrated-650-replay-v1", "completed", "completed"), "Prior native replay completed")
    checks.equal(proof["version"], "sircus-calibrated-650-independent-audit-v2", "Prior revised independent checker")
    checks.equal(proof["status"], "passed", "Prior full audit passed")
    checks.require(not proof.get("failures") and proof["checks_count"] == len(proof["checks"]) > 0
                   and all(c.get("passed") is True for c in proof["checks"]), "Prior audit complete strictly passed checks")
    checks.equal(proof["contract_fingerprint"], fp, "Prior audit exact contract")
    checks.equal(report["contract_fingerprint"], fp, "Prior report exact contract")
    checks.equal(proof["report_sha256"], sha(study/"report.json"), "Prior audit binds exact report")
    checks.equal(proof["evidence_manifest_sha256"], sha(study/"evidence-manifest.json"), "Prior audit binds every native artifact")
    manifest = replay.check_manifest(study, checks)
    checks.equal(report["evidence_sha256"], manifest, "Prior report binds complete raw evidence")
    return spec, report


def source_fields(actual, expected, checks, label):
    for key in ("values", "bucket_xy_m"):
        checks.numeric(actual[key], expected[key], label+"/unchanged configured geometry/"+key, 1e-12)
    for key in ("source_record_id", "source_record_sha256"):
        if key in expected:
            checks.equal(actual[key], expected[key], label+"/source provenance/"+key)


def audit_initial(groups, seed, controls, checks):
    checks.equal(len(groups), 363, "Two fixed controls plus360grid and1reference groups")
    for actual, source in zip(groups[:2], controls):
        checks.equal(actual["id"], source["id"], "Original control group identity")
        checks.equal(actual["role"], source["role"], "Original control role")
        source_fields(actual, source, checks, actual["id"])
        for key in ("q10", "q14"):
            checks.equal(actual[key], source[key], "Control exact historical joint branch/"+key)
    for index, (actual, value) in enumerate(zip(groups[2:], coarse_values(seed))):
        ident = f"coarse-{index:04d}" if index < 360 else "reference-650"
        checks.equal(actual["id"], ident, "Frozen H/T/G lexicographic grid identity")
        checks.equal(actual["role"], "discovery", ident+"/coarse role")
        checks.equal(actual["phase"], "coarse", ident+"/coarse phase")
        checks.numeric(actual["values"], value, ident+"/literal grid geometry", 1e-12)
        checks.equal(actual["bucket_xy_m"], seed["bucket_xy_m"], ident+"/fixed650mm bucket")
        checks.numeric(np.linalg.norm(np.asarray(actual["bucket_xy_m"])-actual["values"][:2]), .65, ident+"/fixed650mm plane distance", 1e-12)
        if index == 360:
            for key in ("q10", "q14"):
                if actual["eligible"]:
                    checks.equal(actual[key], seed[key], "Exact original650 reference joints/"+key)


def audit_refinement(groups, selected, checks):
    checks.equal(len(groups), len(selected)*27, "All27 conditional refinements per selected source retained")
    for s, source in enumerate(selected):
        for index, (actual, value) in enumerate(zip(groups[s*27:(s+1)*27], local_values(source))):
            i, j, k = np.unravel_index(index, (3, 3, 3))
            checks.equal(actual["id"], f"refine-s{s:02d}-h{i}t{j}g{k}", "Fixed local-offset identity")
            checks.equal(actual["role"], "discovery", actual["id"]+"/refinement discovery role")
            checks.equal(actual["phase"], "refinement", actual["id"]+"/refinement phase")
            checks.numeric(actual["values"], value, actual["id"]+"/literal local H/T/G offsets", 1e-12)
            checks.equal(actual["bucket_xy_m"], source["bucket_xy_m"], actual["id"]+"/unchanged650bucket")
            checks.equal(actual["source_record_id"], source["id"], actual["id"]+"/earlier selected source")
            checks.equal(actual["source_record_sha256"], digest(source), actual["id"]+"/exact earlier source bytes")


def audit_verification(groups, selected, checks):
    checks.equal(len(groups), len(selected), "One frozen verification group per selected earlier record")
    for index, (actual, source) in enumerate(zip(groups, selected)):
        checks.equal(actual["id"], f"verify-{index:02d}", "Declared verification order")
        checks.equal(actual["role"], "verification", actual["id"]+"/verification role")
        checks.require(actual["freeze_joints"] is True, actual["id"]+"/verification must preserve source joints")
        for key in ("values", "bucket_xy_m", "q10", "q14"):
            checks.equal(actual[key], source[key], actual["id"]+"/exact source field/"+key)
        checks.equal(actual["source_record_id"], source["id"], actual["id"]+"/source record ID")
        checks.equal(actual["source_record_sha256"], digest(source), actual["id"]+"/source record bytes")


def audit_group_metadata(actual, group, scene, checks, label):
    for key in ("id", "role", "source_record_id", "source_record_sha256", "values", "bucket_xy_m", "eligible"):
        checks.equal(actual[key], group[key], label+"/actual report group/"+key)
    v = group["values"]
    for key, expected in dict(board_xy_m=v[:2], board_height_m=v[2], tilt_deg=v[3], gap_m=v[4],
                              release_center_height_m=v[2]+v[4]+scene["ball"]["radius_m"],
                              distance_m=float(np.linalg.norm(np.asarray(group["bucket_xy_m"])-v[:2]))).items():
        checks.equal(actual[key], expected, label+"/actual configured geometry/"+key)


def audit_traces(study, report, rows, groups, scene, checks):
    by_id = {r["id"]: r for r in rows};by_group = {g["id"]: g for g in groups}
    expected = [r for r in rows if r["phase"] == "controls_pre" and r["repeat_index"] == 0 and r["native_release"]]
    expected += select([r for r in rows if r["phase"] in ("coarse", "refinement")], 3)
    expected += [r for r in rows if r["phase"] == "verification" and r["repeat_index"] == 0 and r["native_release"]]
    checks.equal([t["record_id"] for t in report["trace_examples"]], [r["id"] for r in expected],
                 "Only prespecified first nominal controls/verification and ranked development illustrations")
    checks.require(len(report["trace_examples"]) <= 24, "Bounded illustrative native trajectories")
    seen = set()
    for trace in report["trace_examples"]:
        row = by_id[trace["record_id"]];group = by_group[row["group_id"]]
        checks.require(row["native_release"], "Displayed native record actually released")
        checks.require(row["id"] not in seen, "No duplicated displayed native record")
        seen.add(row["id"])
        for key in ("group_id", "role", "substeps", "whole_ball_entry", "max_retention_s", "external_contact_steps",
                    "route_success", "route_quality", "rim_band_contact", "invalid", "invalid_physics", "scalar_score_audit_passed"):
            checks.equal(trace[key], row[key], row["id"]+"/native display field/"+key)
        v = group["values"]
        for key, expected in dict(board_xy_m=v[:2], bucket_xy_m=group["bucket_xy_m"], board_height_m=v[2], tilt_deg=v[3], gap_m=v[4],
                                  release_center_height_m=v[2]+v[4]+scene["ball"]["radius_m"],
                                  distance_m=float(np.linalg.norm(np.asarray(group["bucket_xy_m"])-v[:2]))).items():
            checks.equal(trace[key], expected, row["id"]+"/actual trace geometry/"+key)
        with np.load(contained(study, row["native_directory"])/row["native_cache_file"], allow_pickle=False) as archive:
            points = archive["position_m"][::row["substeps"]*4, row["native_cache_row"]]
        checks.numeric(trace["position_m"], points, row["id"]+"/actual saved native positions", 0)
        checks.numeric(trace["sample_dt_s"], 1/60, row["id"]+"/native display cadence", 0)
        checks.require(isinstance(trace.get("selection_scope"), str) and bool(trace["selection_scope"]), "Native illustration selection scope explicit")


def audit_statistics(actual, rows, checks, label):
    for key, expected in statistics(rows).items():
        if isinstance(expected, float):
            checks.numeric(actual[key], expected, label+"/"+key, 1e-10)
        else:
            checks.equal(actual[key], expected, label+"/"+key)


def audit_settings(settings, checks):
    expected = dict(grid=dict(heights_m=list(HEIGHTS), tilts_deg=list(TILTS), gaps_m=list(GAPS), reference=[.38, 28., .3868]),
                    coarse_requests=361, refinement_seeds=3,
                    refinement_offsets=dict(height_m=[-.02, 0., .02], tilt_deg=[-2., 0., 2.], gap_m=[-.02, 0., .02]),
                    refinement_max_requests=81, verification_sources=5, repeats_per_resolution=3,
                    substeps=list(STEPS), max_requests=MAX_REQUESTS, minimum_clearance_m=.005, max_drift_350ms_m=.005,
                    verification_dedup_decimals=9)
    for key, value in expected.items():
        checks.equal(settings[key], value, "Literal prespecified sweep protocol/"+key)
    checks.require("no" in settings["verification_scope"].lower() and "noise" in settings["verification_scope"].lower(),
                   "No fresh-noise qualification claimed")


def audit_adaptive_stage(study, stages, index, name, groups, selected, plan, rows_before, proofs, checks):
    checks.equal(read(study/(name+"-selection.json")), dict(source_ids=[r["id"] for r in selected]),
                 name+"/only preceding observed source IDs determine the stage")
    checks.equal(read(study/(name+"-plan.json")), plan, name+"/complete conditional plan saved")
    stage = stages[index]
    checks.equal(stage["phase"], name, "Adaptive stage order")
    raw_lines = (study/"records.jsonl").read_bytes().splitlines(keepends=True)
    expected_hash = hashlib.sha256(b"".join(raw_lines[:len(rows_before)])).hexdigest()
    checks.equal(stage["prior_records_sha256"], expected_hash, name+"/exact earlier observations frozen before adaptive stage")
    names = {name+suffix for suffix in ("-selection.json", "-groups.json", "-plan.json")}
    checks.equal(set(stage["files_sha256"]), names, name+"/all conditional files declared")
    for filename, expected in stage["files_sha256"].items():
        checks.equal(sha(contained(study, filename)), expected, name+"/conditional file bytes/"+filename)
    prefix = max((g["geometry_row"] for g in groups), default=(362 if index == 0 else 362)) + 1
    checks.equal(stage["geometry_sha256"], digest(proofs[:prefix]), name+"/full actual geometry-proof prefix frozen")


def audit_report(report, rows, groups, verification_groups, sources, positions, pre, post, scene, checks):
    audit_statistics(report["counts"], rows, checks, "All requested shot accounting")
    for field, key in (("phase_counts", "phase"), ("substep_counts", "substeps")):
        checks.equal(report["counts"][field], dict(Counter(str(r[key]) for r in rows)), "Report complete accounting/"+field)
    checks.equal(report["control_gates"], dict(controls_pre=pre, controls_post=post), "Both independent control gates reported")
    checks.equal(len(report["groups"]), len(groups), "All geometry groups visible, including rejects")
    for actual, group in zip(report["groups"], groups):
        audit_group_metadata(actual, group, scene, checks, group["id"])
        checks.equal(actual["phase"], group["phase"], group["id"]+"/actual development/control phase")
        for step in STEPS:
            audit_statistics(actual["by_substeps"][str(step)], [r for r in rows if r["group_id"] == group["id"] and r["substeps"] == step], checks,
                             group["id"]+"/actual group outcome/"+str(step))
    control_rows = report["controls"]
    keys = [(phase, g["id"], g["role"], step) for phase in ("controls_pre", "controls_post") for g in groups[:2] for step in STEPS]
    checks.equal([(r["phase"], r["group_id"], r["role"], r["substeps"]) for r in control_rows], keys, "All control phases/roles/resolutions visible")
    for actual in control_rows:
        selected = [r for r in rows if r["phase"] == actual["phase"] and r["group_id"] == actual["group_id"] and r["substeps"] == actual["substeps"]]
        audit_statistics(actual["stats"], selected, checks, "Actual control cohort")
    selected = select([r for r in rows if r["phase"] in ("coarse", "refinement")], 10)
    checks.equal([r["id"] for r in report["top_settings"]], [r["id"] for r in selected], "Exact prespecified top10 observed development records")
    by_group = {g["id"]: g for g in groups}
    for actual, source in zip(report["top_settings"], selected):
        group = by_group[source["group_id"]]
        # A top row's ID is the actual record ID; group identity remains explicit.
        metadata = dict(actual, id=group["id"])
        audit_group_metadata(metadata, group, scene, checks, source["id"])
        checks.equal(actual["group_id"], group["id"], source["id"]+"/actual parent group")
        checks.equal(actual["substeps"], source["substeps"], source["id"]+"/actual search resolution")
        audit_statistics(actual["stats"], [source], checks, source["id"]+"/actual single observed trial")
        for key in ("minimum_mouth_clearance_mm", "closest_opening_distance_m"):
            checks.equal(actual[key], source.get(key), source["id"]+"/actual quality metric/"+key)
    checks.equal(len(report["verification"]), len(verification_groups), "Every selected verification group reported")
    qualified = 0
    for actual, group, source in zip(report["verification"], verification_groups, sources):
        audit_group_metadata(actual, group, scene, checks, group["id"])
        selected = [r for r in rows if r["group_id"] == group["id"]]
        for step in STEPS:
            audit_statistics(actual["by_substeps"][str(step)], [r for r in selected if r["substeps"] == step], checks, group["id"]+"/verification/"+str(step))
        replay.check_pairing(selected, checks)
        sensitivity = replay.paired_sensitivity(selected, positions)
        checks.equal(actual["sensitivity"], sensitivity, group["id"]+"/actual matched position/outcome sensitivity")
        result = qualify(selected, positions, controls_passed=pre["status"] == post["status"] == "passed", source_clean=clean(source))
        checks.equal(actual["qualification"], result, group["id"]+"/independent clean source/control/timestep qualification")
        qualified += int(result["qualified"])
    status = "blocked_controls" if pre["status"] != "passed" or post["status"] != "passed" else "passed" if qualified else "failed"
    gate = dict(status=status, controls_pre_passed=pre["status"] == "passed", controls_post_passed=post["status"] == "passed",
                qualified_settings=qualified, more_ppo_started=False)
    checks.equal(report["gate"], gate, "Final clean650 gate independently reconstructed")
    return gate


def audit(study, source, assets, *, shallow_study, positive_study, positive_audit, replay_study):
    started, checks = time.monotonic(), AuditChecks()
    study, source, assets = map(Path, (study, source, assets))
    manifest = replay.check_manifest(study, checks)
    spec, report = read(study/"contract.json"), read(study/"report.json")
    unsigned = dict(spec);fp = unsigned.pop("fingerprint", None)
    checks.equal(digest(unsigned), fp, "Complete clean650 contract fingerprint")
    checks.equal(spec["version"], STUDY_VERSION, "Versioned calibrated clean650 study")
    checks.equal(report["version"], STUDY_VERSION, "Versioned clean650 report")
    checks.equal(report["contract_fingerprint"], fp, "Report exact contract identity")
    checks.equal((report["status"], report["phase"]), ("completed", "completed"), "Finite study completed")
    checks.equal(spec["evidence_kind"], "native_calibrated_clean650_sweep", "Actual simulated native evidence")
    checks.equal(report["evidence_sha256"], manifest, "Report binds complete raw evidence")
    replay.check_source_files(source, spec["source_sha256"], checks, "Immutable actual native source")
    paths = list((source/"sircus/mjlab").glob("*.py")) + [source/"sircus/mjlab/requirements.txt"]
    paths += [source/"sircus/isaac"/n for n in ("geometry.py", "rl_contract.py", "entry_quality.py", "board_route.py")]
    checks.equal(set(spec["source_sha256"]), {str(p.relative_to(source)) for p in paths}, "All current execution/audit modules declared")
    from .learning_solve import required_asset_paths
    assets_actual = {n: sha(contained(assets.parent, n)) for n in required_asset_paths(assets)}
    checks.equal(spec["asset_sha256"], assets_actual, "Actual native collision assets")
    replay.audit_packages(spec["packages"], checks)
    calibration = replay.check_calibration(study, spec["calibration"], checks)
    parent, original = replay.audit_input_sources(shallow_study, positive_study, positive_audit, spec["input_bindings"], checks, spec["calibration"])
    prior, _ = audit_prior_replay(Path(replay_study), spec["input_bindings"]["replay"], checks)
    checks.equal(spec["scene"], parent["scene"], "Original scene and all provisional measurements preserved")
    checks.equal(spec["scene"], prior["scene"], "Same scene as prior matched650 replay")
    checks.equal(spec["calibration"], prior["calibration"], "Calibration not refitted")
    checks.equal(spec["packages"], parent["packages"], "Complete eight-package historical ancestry")
    checks.equal(spec["asset_sha256"], parent["asset_sha256"], "Exact original assets preserved")
    frozen = {"initial-groups.json", "initial-geometry.json", "initial-plan.json", "calibration.json", "verification.json"}
    checks.equal(set(spec["frozen_files_sha256"]), frozen, "All preoutcome inputs frozen")
    for name, expected in spec["frozen_files_sha256"].items():
        checks.equal(sha(contained(study, name)), expected, "Original preoutcome file bytes/"+name)
    audit_settings(spec["settings"], checks)
    checks.equal(report["settings"], spec["settings"], "Report exact frozen experiment settings")
    initial_groups, initial_proofs = read(study/"initial-groups.json"), read(study/"initial-geometry.json")
    seed = next(g for g in original if g["role"] == "candidate")
    controls = [g for g in original if g["role"].endswith("_control")]
    audit_initial(initial_groups, seed, controls, checks)
    before = make_plan(initial_groups[:2], "controls_pre", steps=STEPS, repeats=3)
    coarse = make_plan(initial_groups[2:], "coarse", steps=(64,), repeats=1)
    after = make_plan(initial_groups[:2], "controls_post", steps=STEPS, repeats=3)
    checks.equal(read(study/"initial-plan.json"), dict(controls_pre=before, coarse=coarse, controls_post=after), "Complete literal initial/control/coarse plans")
    groups, proofs = read(study/"groups.json"), read(study/"geometry.json")
    checks.equal(groups[:363], initial_groups, "Initial groups never mutated after outcomes")
    checks.equal(proofs[:363], initial_proofs, "Initial geometry proofs never mutated after outcomes")
    audit_geometry(groups, proofs, spec["scene"], assets, checks)
    rows = replay.read_records(study/"records.jsonl")
    checks.require(12 <= len(rows) <= MAX_REQUESTS, "Finite requested attempt budget including rejects")
    checks.equal([r["id"] for r in rows[:12]], [r["id"] for r in before], "Controls precede every development request")
    pre = control_gate(rows[:12])
    checks.equal(read(study/"initial-control-gate.json"), pre, "Initial independently reconstructed control gate")
    selected_coarse = select([r for r in rows if r["phase"] == "coarse"], 3) if pre["status"] == "passed" else []
    refined = read(study/"refinement-groups.json")
    audit_refinement(refined, selected_coarse, checks)
    refine_plan = make_plan(refined, "refinement", steps=(64,), repeats=1)
    selected_verify = select_unique([r for r in rows if r["phase"] in ("coarse", "refinement")], 5) if pre["status"] == "passed" else []
    verified = read(study/"verification-groups.json")
    audit_verification(verified, selected_verify, checks)
    verification_plan = make_plan(verified, "verification", steps=STEPS, repeats=3)
    checks.equal(groups, initial_groups+refined+verified, "Final groups exactly concatenate opened phases")
    stages = read(study/"adaptive-stages.json")
    checks.equal(len(stages), 2, "Exactly two declared adaptive stages, no retries")
    prior_rows = rows[:12+len(coarse)] if pre["status"] == "passed" else rows[:12]
    audit_adaptive_stage(study, stages, 0, "refinement", refined, selected_coarse, refine_plan, prior_rows, proofs, checks)
    prior_rows = rows[:12+len(coarse)+len(refine_plan)] if pre["status"] == "passed" else rows[:12]
    audit_adaptive_stage(study, stages, 1, "verification", verified, selected_verify, verification_plan, prior_rows, proofs, checks)
    expected = before+coarse+refine_plan+verification_plan+after if pre["status"] == "passed" else before
    audit_native_rows(rows, expected, groups, checks)
    replay.audit_batches(study, rows, checks)
    post = control_gate([r for r in rows if r["phase"] == "controls_post"])
    if pre["status"] != "passed":
        post["status"] = "not_run"
    checks.equal(read(study/"final-control-gate.json"), post, "Final independent control gate retained")
    actual_steps = {r["substeps"] for r in rows if r["native_release"]}
    checks.require(actual_steps in (set(), set(STEPS)), "Controls allocate both resolutions together or release none")
    if actual_steps:
        options = read(study/"effective-options.json")
        replay.audit_options(options, checks)
        replay.audit_compiled(study, spec["scene"], assets, calibration, options, checks)
        positions, traces = replay.audit_native(study, rows, spec["scene"], calibration, assets, checks)
    else:
        checks.require(not (study/"effective-options.json").exists(), "No invented native options for wholly rejected controls")
        checks.require(not list((study/"native").rglob("*.npz")), "No invented native caches for wholly rejected controls")
        positions, traces = {}, {}
    gate = audit_report(report, rows, groups, verified, selected_verify, positions, pre, post, spec["scene"], checks)
    audit_traces(study, report, rows, groups, spec["scene"], checks)
    identity = auditor_identity(source, spec["source_sha256"], checks)
    return dict(version=VERSION, status="passed", contract_fingerprint=fp, checks_count=checks.count,
                checks=checks.entries, failures=[], auditor_sha256=sha(Path(__file__)),
                report_sha256=sha(study/"report.json"), evidence_manifest_sha256=sha(study/"evidence-manifest.json"),
                auditor_identity=identity, producer_source_sha256=spec["source_sha256"],
                requested_trials=len(rows), native_releases=len(traces), gate=gate, elapsed_s=time.monotonic()-started,
                limitations=["Fixed650mm development search; no held-out or general workspace coverage.",
                             "Nominal repeats only; noise qualification and new PPO explicitly excluded.",
                             "Calibrated simulated contacts and held endpoints; no grasp/transit/actuator/perception/physical validation."])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("study", "source", "assets", "shallow-study", "positive-study", "positive-audit", "replay-study", "output"):
        parser.add_argument("--"+name, type=Path, required=True)
    args = parser.parse_args()
    if args.output.resolve().is_relative_to(args.study.resolve()) and args.output.name != "audit-clean650.json":
        raise ValueError("Only designated new audit may be written inside frozen evidence")
    try:
        result = audit(args.study, args.source, args.assets, shallow_study=args.shallow_study,
                       positive_study=args.positive_study, positive_audit=args.positive_audit, replay_study=args.replay_study)
    except (EvidenceError, ValueError, KeyError, OSError) as error:
        result = dict(version=VERSION, status="failed", failures=[str(error)])
        args.output.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
        raise
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    print(json.dumps({k:result[k] for k in ("status", "checks_count", "requested_trials", "native_releases", "gate")}))


if __name__ == "__main__":
    main()
