"""Independent CPU audit of the conditional calibrated 650 mm replay.

This module only reads frozen evidence. It neither launches dynamics nor alters
the calibration, scoring contract, requests, or original historical outcomes.
"""

from __future__ import annotations

import argparse
import copy
from collections import Counter
import importlib.metadata
import json
import math
from pathlib import Path
import time

import numpy as np

from .benchmark_verify import Checks, EvidenceError, contained, digest, read, sha

VERSION = "sircus-calibrated-650-independent-audit-v1"
STUDY_VERSION = "sircus-calibrated-650-replay-v1"
STEPS = (64, 128)
WINDOW_GAPS_M = (.3768, .3808, .3848, .3868, .3888)
REPEATS = 3
MAX_REQUESTS = 48
DRIFT_TIME_S = .350
MAX_DRIFT_M = .005
FORCE_NAMES = ["board", "bucket", "table", "floor", "robot", "unattributed"]
CALIBRATED_SCORER = "sircus-calibrated-native-capture-v2"


class AuditChecks(Checks):
    def __init__(self):
        super().__init__()
        self.entries = []

    def require(self, passed, label):
        self.entries.append(dict(label=label, passed=bool(passed)))
        super().require(passed, label)


def captured(row):
    return bool(row.get("native_release") and row.get("route_success")
                and not row.get("invalid") and not row.get("invalid_physics"))


def statistics(rows):
    """Requested denominators include every rejected endpoint."""
    return dict(attempts=len(rows),
                native_releases=sum(bool(r.get("native_release")) for r in rows),
                captures=sum(captured(r) for r in rows),
                invalid=sum(bool(r.get("invalid") or r.get("invalid_physics")) for r in rows),
                clean=sum(captured(r) and r.get("route_quality") == 2 for r in rows),
                whole_entries=sum(bool(r.get("native_release") and r.get("whole_ball_entry")) for r in rows),
                max_retention_s=max((float(r.get("max_retention_s", 0.)) for r in rows), default=0.))


def replay_gate(rows, max_drift_m):
    """Independent literal gate; no outcome-dependent retries or fresh noise."""
    groups = {role: {step: [r for r in rows if r["role"] == role
                            and int(r["substeps"]) == step] for step in STEPS}
              for role in ("positive_control", "negative_control", "candidate")}
    present = all(len(groups[role][step]) == REPEATS for role in groups for step in STEPS)
    positive = present and all(all(captured(r) for r in groups["positive_control"][s]) for s in STEPS)
    negative = present and all(all(r.get("native_release") and not captured(r)
                                  and not r.get("invalid") and not r.get("invalid_physics")
                                  for r in groups["negative_control"][s]) for s in STEPS)
    candidate = present and all(all(captured(r) for r in groups["candidate"][s]) for s in STEPS)
    physics = present and all(r.get("native_release") and not r.get("invalid")
                             and not r.get("invalid_physics") for r in rows)
    oracle = present and all(r.get("scalar_score_audit_passed") is True for r in rows)
    drift = isinstance(max_drift_m, (int, float)) and math.isfinite(max_drift_m) and 0 <= max_drift_m <= MAX_DRIFT_M
    passed = bool(positive and negative and candidate and physics and oracle and drift)
    return dict(status="passed" if passed else "failed", positive_control=bool(positive),
                negative_control=bool(negative), candidate=bool(candidate),
                physics_valid=bool(physics), oracle_agreement=bool(oracle),
                max_drift_m=max_drift_m, window_opened=passed)


def paired_sensitivity(rows, positions):
    """Compare actual native positions at the same canonical 350 ms tick."""
    by_pair = {}
    for row in rows:
        step = int(row["substeps"])
        if step not in STEPS:
            raise EvidenceError("Unexpected sensitivity timestep")
        pair = row["pair_id"]
        if step in by_pair.setdefault(pair, {}):
            raise EvidenceError("Repeated sensitivity pair/step")
        by_pair[pair][step] = row
    differences, agreement = [], 0
    for pair, members in by_pair.items():
        if set(members) != set(STEPS):
            raise EvidenceError("Unpaired sensitivity request: " + pair)
        a, b = members[64], members[128]
        if a.get("native_release") and b.get("native_release"):
            pa, pb = (np.asarray(positions[r["id"]], dtype=float) for r in (a, b))
            if pa.shape != (3,) or pb.shape != (3,) or not np.isfinite([pa, pb]).all():
                raise EvidenceError("Missing finite paired native 350 ms position")
            differences.append(float(np.linalg.norm(pa-pb)))
        agreement += captured(a) == captured(b)
    return dict(paired_attempts=len(by_pair), capture_agreement=agreement,
                nominal_position_delta_350ms_max_m=max(differences) if differences and len(differences) == len(by_pair) else None)


def check_pairing(rows, checks):
    pairs = {}
    for row in rows:
        key = row["pair_id"]
        members = pairs.setdefault(key, {})
        step = int(row["substeps"])
        checks.require(step in STEPS and step not in members, key+"/one request at each resolution")
        members[step] = row
    for key, members in pairs.items():
        checks.equal(set(members), set(STEPS), key+"/matched two-resolution pair")
        a, b = members[64], members[128]
        for field in ("group_id", "role", "repeat_index", "values", "bucket_xy_m", "q10", "q14"):
            checks.equal(a[field], b[field], key+"/same frozen input/"+field)
    return pairs


def check_statistics(actual, rows, checks, label):
    for key, expected in statistics(rows).items():
        if isinstance(expected, float):
            checks.numeric(actual[key], expected, label+"/"+key, 1e-10)
        else:
            checks.equal(actual[key], expected, label+"/"+key)


def check_source_files(source, declared, checks, label):
    checks.require(isinstance(declared, dict) and bool(declared), label+"/nonempty source binding")
    for name, expected in declared.items():
        checks.equal(sha(contained(source, name)), expected, label+"/"+name)


def audit_packages(declared, checks):
    """Retain the parent's full dependency identity, including CPU hull math."""
    from .corrected_contract import PINNED_PACKAGES
    expected = dict(PINNED_PACKAGES, scipy="1.15.3")
    actual = {name: importlib.metadata.version(name) for name in expected}
    checks.equal(declared, actual, "Actual CPU audit package versions match complete native ancestry")
    for name, pinned in expected.items():
        public = actual[name].split("+")[0] if name == "torch" else actual[name]
        checks.equal(public, pinned, "Pinned package/"+name)
    return actual


def check_calibration(study, declared, checks):
    """Frozen independently verified calibration is consumed, never refitted."""
    from .calibrated_score_oracle import validate_calibration
    calibration, verification = read(study/"calibration.json"), read(study/"verification.json")
    for name, value in (("calibration", calibration), ("verification", verification)):
        checks.equal(value["status"], "passed", name+"/completed passing controls")
        checks.require(not value.get("failures"), name+"/no control failures")
        checks.require(not value.get("synthetic_fixture", False), name+"/native controls")
        count = value.get("checks", value.get("checks_count"))
        checks.require(type(count) is int and count > 0, name+"/positive actual checks")
        checks.equal(sha(study/(name+".json")), declared[name+"_sha256"], name+"/exact frozen bytes")
    checks.equal(verification["calibration_sha256"], sha(study/"calibration.json"), "Independent verification binds exact calibration")
    cfg = validate_calibration(calibration["score_config"])
    checks.equal(verification["score_config_sha256"], digest(cfg), "Independent verification binds exact score configuration")
    checks.equal(declared["score_config"], cfg, "Executed frozen score configuration")
    checks.equal(declared["score_config_sha256"], digest(cfg), "Executed calibration identity")
    checks.numeric(cfg["retention_s"], 1., "One continuous second unchanged", 0)
    checks.numeric(cfg["native_dt_s"], 1/15360, "Frozen fit used canonical64 dynamics", 0)
    return cfg


def read_records(path):
    raw = Path(path).read_text()
    if raw and not raw.endswith("\n"):
        raise EvidenceError("Incomplete final JSONL record")
    return [json.loads(line) for line in raw.splitlines() if line.strip()]


def native_trace_spec(substeps):
    return dict(version="sircus-dense-native-interval-trace-v2",
                native_dt_s=1/(240*substeps), scoring_dt_s=1/(240*substeps),
                horizon_s=4., force_names=FORCE_NAMES)


def audit_options(options, checks):
    checks.equal(set(options), {"64", "128"}, "Both actually used native resolutions recorded")
    for step in STEPS:
        value = options[str(step)]
        checks.equal(value["version"], "sircus-calibrated-native-warp-options-v1", "Allocated option record version")
        checks.equal(value["substeps"], step, "Actual allocated option resolution")
        expected = dict(timestep=1/(240*step), solver=2, iterations=100, tolerance=1e-8,
                        cone=0, integrator=3, ccd_iterations=50, ccd_tolerance=1e-6,
                        disableflags=0, enableflags=0, gravity=[0., 0., -9.81])
        checks.equal(value["native"], expected, "Exact allocated native options/"+str(step))
        warp = {k: (np.asarray(1e-6 if k == "tolerance" else v, dtype=np.float32).astype(float).tolist()
                    if k in {"timestep", "tolerance", "ccd_tolerance", "gravity"} else v)
                for k, v in expected.items()}
        checks.equal(value["warp"], warp, "Exact allocated float32 Warp options/vendor tolerance floor/"+str(step))


def check_manifest(study, checks):
    manifest = read(study/"evidence-manifest.json")
    checks.require(isinstance(manifest, dict) and bool(manifest), "Nonempty complete evidence manifest")
    for name, expected in manifest.items():
        checks.equal(sha(contained(study, name)), expected, "Frozen evidence bytes/"+name)
    actual = {str(p.relative_to(study)) for p in study.rglob("*") if p.is_file()
              and p.name not in {"report.json", "evidence-manifest.json"} and not p.name.startswith("audit")}
    checks.equal(set(manifest), actual, "Manifest covers every saved native and metadata artifact")
    return manifest


def audit_input_sources(old_study, positive_study, positive_audit, bindings, checks, calibration):
    """Resolve geometry directly from original files, independently of the runner."""
    old_study, positive_study, positive_audit = map(Path, (old_study, positive_study, positive_audit))
    for root, key, manifest_name in ((old_study, "old", "evidence-manifest.json"),
                                      (positive_study, "positive", "manifest.json")):
        for name, expected in bindings[key].items():
            checks.equal(sha(contained(root, name)), expected, "Original input bytes/"+key+"/"+name)
        manifest = read(root/manifest_name)
        checks.require(isinstance(manifest, dict) and bool(manifest), "Original input manifest present/"+key)
        for name, expected in manifest.items():
            checks.equal(sha(contained(root, name)), expected, "Original manifest member/"+key+"/"+name)
    checks.equal(sha(positive_audit), bindings["positive_audit_sha256"], "Exact independent positive-control audit")
    parent = read(old_study/"contract.json")
    payload = dict(parent); fp = payload.pop("fingerprint", None)
    expected_fp = "c4e59119e71f724b1e1941ae080e40c8aa001d954356715188895803fcd41a84"
    checks.equal(digest(payload), expected_fp, "Exact original shallow candidate contract")
    checks.equal(fp, expected_fp, "Original shallow fingerprint")
    old_report, old_audit = read(old_study/"report.json"), read(old_study/"audit-shallow.json")
    checks.equal(old_report["contract_fingerprint"], fp, "Original report identity")
    checks.equal(old_report["status"], "completed", "Original shallow study completed")
    checks.equal(old_report["phase"], "completed", "Original shallow phase completed")
    checks.equal(old_audit["contract_fingerprint"], fp, "Original independent audit identity")
    checks.equal(old_audit["status"], "passed", "Original independent audit passed")
    checks.require(not old_audit.get("failures"), "Original audit has no failures")
    checks.equal(old_audit["checks_count"], len(old_audit["checks"]), "Original complete check accounting")
    checks.require(bool(old_audit["checks"]) and all(c.get("passed") is True for c in old_audit["checks"]), "All original shallow checks passed")
    historical = {r["id"]: r for r in read_records(old_study/"records.jsonl")}
    selected = []
    for ident, role, group, distance in (("shallow-shot-000024", "candidate", "candidate-650", .65),
                                         ("shallow-shot-000026", "negative_control", "negative-control", .75)):
        row = historical[ident]
        checks.equal(row["values"][2:], [.38, 28., .3868], ident+"/exact rounded historical candidate")
        checks.require(row["native_release"] and not row.get("invalid"), ident+"/actual historical release")
        checks.numeric(np.linalg.norm(np.asarray(row["bucket_xy_m"])-row["values"][:2]), distance, ident+"/original target distance", 1e-12)
        selected.append(dict(id=group, role=role, source_record_id=ident, source_record_sha256=digest(row),
                             **{k: copy.deepcopy(row[k]) for k in ("values", "bucket_xy_m", "q10", "q14")}))
    positive_report, proof = read(positive_study/"report.json"), read(positive_audit)
    checks.equal(positive_report["status"], "passed", "Original calibrated positive preflight passed")
    checks.equal(positive_report["synthetic_fixture"], False, "Positive control is native evidence")
    checks.equal(positive_report["controls_evaluated"], 17, "All original calibrated controls requested")
    checks.equal(positive_report["controls_captured"], 17, "All original calibrated controls captured")
    checks.equal(proof["status"], "passed", "Independent positive-source audit passed")
    checks.require(not proof.get("failures") and type(proof.get("checks")) is int and proof["checks"] > 0,
                   "Positive-source audit has completed passing checks")
    checks.equal(proof["identity"], positive_report["identity"], "Positive-source audit binds exact runtime identity")
    checks.equal(proof["report_sha256"], sha(positive_study/"report.json"), "Positive audit binds exact original report")
    checks.equal(proof["study_manifest_sha256"], sha(positive_study/"manifest.json"), "Positive audit binds all original evidence")
    actual_calibration = positive_report["identity"]["calibration"]
    for key in ("calibration_sha256", "verification_sha256", "score_config_sha256"):
        checks.equal(actual_calibration[key], calibration[key], "Positive-control matched frozen calibration/"+key)
    checks.equal(actual_calibration, calibration, "Positive-control complete frozen calibration identity")
    reference_scene = copy.deepcopy(parent["scene"])
    reference_scene["bucket"]["center_xy_m"] = [.67, -.42]
    checks.equal(positive_report["identity"]["scene_sha256"], digest(reference_scene),
                 "Only exact unused default bucket translation differs between historical and control scenes")
    checks.equal(positive_report["identity"]["asset_sha256"], parent["asset_sha256"],
                 "Positive and historical control assets exactly match")
    translation = bindings["scene_translation_metadata"]
    for key, expected in dict(historical_scene_sha256=digest(parent["scene"]),
                              control_reference_scene_sha256=digest(reference_scene),
                              historical_default_bucket_xy_m=parent["scene"]["bucket"]["center_xy_m"],
                              control_reference_default_bucket_xy_m=[.67, -.42]).items():
        checks.equal(translation[key], expected, "Explicit single-field scene metadata exception/"+key)
    positive = read(positive_study/"controls.json")["rows"]
    checks.equal(len(positive), 17, "Original ordered positive controls")
    checks.require(all(captured(r) for r in positive), "Every original positive is a valid capture")
    row = positive[0]
    selected.insert(0, dict(id="positive-control", role="positive_control", source_record_id=row["id"], source_record_sha256=digest(row),
                           **{k: copy.deepcopy(row[k]) for k in ("values", "bucket_xy_m", "q10", "q14")}))
    return parent, selected


def expected_plan(groups, phase):
    """Independent fixed order and literal request identities."""
    return [dict(id=f'replay650-{step}-{g["id"]}-r{i}', pair_id=f'{g["id"]}-r{i}',
                 group_id=g["id"], role=g["role"], substeps=step, repeat_index=i, phase=phase,
                 values=g["values"], bucket_xy_m=g["bucket_xy_m"], q10=g["q10"], q14=g["q14"],
                 eligible=g["eligible"], geometry_row=g["geometry_row"])
            for step in STEPS for g in groups for i in range(REPEATS)]


def audit_groups(groups, proofs, original, scene, checks):
    checks.equal(len(groups), 8, "Three matched groups plus five fixed conditional gaps")
    checks.equal(len(proofs), 8, "One frozen geometry proof per group")
    checks.equal([g["id"] for g in groups[:3]], [g["id"] for g in original], "Original positive/candidate/negative order")
    for actual, source in zip(groups[:3], original):
        for key, expected in source.items():
            checks.equal(actual[key], expected, actual["id"]+"/exact historical geometry/"+key)
    candidate = groups[1]
    for index, gap in enumerate(WINDOW_GAPS_M, start=3):
        group = groups[index]
        checks.equal(group["id"], "window-gap-"+str(round(gap*10000)), "Fixed conditional gap identity")
        checks.equal(group["role"], "window", "Conditional group role")
        values = list(candidate["values"]); values[4] = gap
        checks.equal(group["values"], values, "Only release gap changes in conditional window")
        checks.equal(group["bucket_xy_m"], candidate["bucket_xy_m"], "Conditional bucket fixed")
        for key in ("source_record_id", "source_record_sha256"):
            checks.equal(group[key], candidate[key], "Conditional original source provenance/"+key)
        checks.equal(group["q14"], candidate["q14"], "Conditional board branch exactly frozen")
        checks.require(np.max(np.abs(np.asarray(group["q10"])-candidate["q10"])) <= .10,
                       "Conditional dropper stays on local <=0.10rad branch")
        if gap == .3868:
            checks.equal(group["q10"], candidate["q10"], "Central conditional action preserves exact source dropper")
    table = scene["table"]
    center, half = np.asarray(table["center_m"][:2]), np.asarray(table["size_m"][:2])/2
    radius = max(scene["bucket"]["outer_top_diameter_m"], scene["bucket"]["outer_bottom_diameter_m"])/2
    for index, (group, proof) in enumerate(zip(groups, proofs)):
        label = group["id"]
        checks.equal(group["geometry_row"], index, label+"/unique geometry proof row")
        checks.equal(group["geometry_sha256"], digest(proof), label+"/frozen geometry proof bytes")
        for key in ("values", "bucket_xy_m", "eligible"):
            checks.equal(proof[key], group[key], label+"/proof input binding/"+key)
        checks.equal(proof["group_id"], label, label+"/geometry group identity")
        checks.equal(proof["held_endpoints_only"], True, label+"/no unsupported transit proof")
        checks.equal(proof["preserved_source"], index < 3, label+"/declared source versus local gap")
        xy = np.asarray(group["bucket_xy_m"])
        edges = np.r_[xy-(center-half)-radius, (center+half)-xy-radius]
        fp = proof["bucket_footprint"]
        checks.numeric(fp["outer_radius_m"], radius, label+"/full outside bucket footprint", 0)
        checks.numeric(fp["edge_clearance_m"], edges, label+"/all four table-edge clearances", 1e-12)
        checks.numeric(fp["minimum_edge_clearance_m"], min(edges), label+"/minimum table clearance", 1e-12)
        checks.equal(fp["passed"], bool(min(edges) >= .005-1e-12), label+"/5mm table-edge criterion")
        checks.require(fp["passed"] or not group["eligible"], label+"/no off-table native eligibility")
        passed = [a for a in proof["attempts"] if a["checked"]["reason"] == 0]
        checks.equal(group["eligible"], bool(passed), label+"/no silent geometry admission")
        if passed:
            checks.equal(proof["attempts"][-1], passed[0], label+"/first admitted endpoint used")
            for key in ("q10", "q14"):
                checks.equal(passed[0][key], group[key], label+"/actual chosen joint branch/"+key)
            checks.require(proof["minimum_clearance_lower_bound_m"] >= .005-1e-12, label+"/strict endpoint clearance proof")


def audit_cpu_endpoints(groups, proofs, scene, assets, checks):
    from .corrected_geometry import StrictEndpointGate
    from .learning_ik import shot_targets
    from .expanded_actions import WIDE_BOUNDS
    gate = StrictEndpointGate(scene, assets, .005)
    for group, proof in zip(groups, proofs):
        values, xy = np.asarray(group["values"]), np.asarray(group["bucket_xy_m"])
        bounded = bool((values >= WIDE_BOUNDS["low"]).all() and (values <= WIDE_BOUNDS["high"]).all())
        checks.equal(proof["bounds_passed"], bounded, group["id"]+"/actual unchanged action bounds")
        checks.require(bounded or not group["eligible"], group["id"]+"/out-of-bounds requests cannot be admitted")
        checks.require(np.linalg.norm(xy-values[:2]) > .15, group["id"]+"/>150mm board/bucket separation")
        target = {k: value[0] for k, value in shot_targets(scene, xy[None], values[None, :3], [values[3]], [values[4]]).items()}
        for index, attempt in enumerate(proof["attempts"]):
            actual = gate.check(attempt["q10"], attempt["q14"], xy, values, target)
            checks.equal(actual["reason"], attempt["checked"]["reason"], group["id"]+"/independently rechecked endpoint/"+str(index))
            if actual["reason"] == 0:
                checks.require(actual["minimum_clearance_lower_bound_m"] >= .005-1e-12,
                               group["id"]+"/fresh actual strict5mm clearance")
                checks.numeric(attempt["checked"]["minimum_clearance_lower_bound_m"], actual["minimum_clearance_lower_bound_m"],
                               group["id"]+"/fresh clearance identity", 1e-10)


def audit_compiled(study, scene, assets, calibration, options, checks):
    """Recompile geometry on CPU; independently inspect hulls and materials."""
    import mujoco
    from .model import build_scene
    from .verify_capture_calibration import compile_cpu_model, inspect_compiled_geometry
    full = mujoco.MjModel.from_xml_string(build_scene(scene, assets, solref=(.004, .15), contact_friction=.35))
    expected_ids = dict(ball=int(full.geom("ball/geom").id), base=int(full.geom("bucket/base").id),
                        walls=[int(full.geom(f"bucket/wall{i:02d}").id) for i in range(scene["bucket"]["segments"])])
    outputs = {}
    for step in STEPS:
        directory = study/"native"/str(step)
        checks.equal(read(directory/"effective-options.json"), options[str(step)], "Per-resolution actual options binding")
        compiled = read(directory/"native-compiled-geometry.json")
        checks.equal(compiled["contact_geom_ids"], expected_ids, "CPU actual full-model contact category IDs")
        checks.numeric(compiled["compiled_planes"], calibration["compiled_planes"], "Unchanged calibrated actual collision planes", calibration["geometry_epsilon_m"])
        identity = dict(planes=compiled["compiled_planes"], floor=compiled["compiled_floor_m"], rim=compiled["compiled_rim_m"], radius=compiled["ball_radius_m"])
        checks.equal(compiled["geometry_sha256"], digest(identity), "Compiled geometry digest independently reconstructed")
        checks.equal(compiled["effective_options"], options[str(step)]["native"], "Compiled model actual options")
        cpu, _ = compile_cpu_model(scene, assets, step)
        isolated = copy.deepcopy(compiled)
        isolated["contact_geom_ids"] = dict(ball=int(cpu.geom("ball/geom").id), base=int(cpu.geom("bucket/base").id),
            walls=[int(cpu.geom(f"bucket/wall{i:02d}").id) for i in range(scene["bucket"]["segments"])])
        class GeometryChecks:
            require = checks.require
            def close(self, actual, expected, label, *, atol=1e-12):
                checks.numeric(actual, expected, label, atol)
        inspect_compiled_geometry(cpu, scene, isolated, GeometryChecks())
        outputs[step] = compiled
    for key in ("compiled_planes", "compiled_floor_m", "compiled_rim_m", "ball_radius_m", "ball_contact", "geometry_sha256"):
        checks.equal(outputs[64][key], outputs[128][key], "Same compiled geometry/material at both resolutions/"+key)


def audit_native(study, rows, scene, calibration, assets, checks):
    from .calibrated_benchmark_verify import rescore_rows
    from .calibrated_score_oracle import score_trace
    positions, trace_evidence = {}, {}
    for step in STEPS:
        chosen = [r for r in rows if r["substeps"] == step]
        directory = study/"native"/str(step)
        trace_evidence.update(rescore_rows(directory, chosen, scene, calibration, checks,
                             "dense native "+str(step), native_trace_spec(step), assets=assets))
        by_file = {}
        for row in chosen:
            if row.get("native_release"):
                by_file.setdefault(row["native_cache_file"], []).append(row)
        for name, members in by_file.items():
            with np.load(contained(directory, name), allow_pickle=False) as archive:
                arrays = {k: archive[k] for k in archive.files}
                for row in members:
                    index = row["native_cache_row"]
                    trace = {k: (v[:, index] if v.ndim >= 2 else v) for k, v in arrays.items()
                             if k != "record_ids"}
                    actual = score_trace(scene, row["bucket_xy_m"], row["board_normal"], calibration, trace)
                    for key, expected in actual.items():
                        if key == "invalid" and expected:
                            expected = "native_physics_outside_calibrated_domain"
                        checks.require(key in row, row["id"]+"/complete oracle field/"+key)
                        if isinstance(expected, float):
                            checks.numeric(row[key], expected, row["id"]+"/complete independent oracle/"+key, 1e-8)
                        else:
                            checks.equal(row[key], expected, row["id"]+"/complete independent oracle/"+key)
                    tick = round(DRIFT_TIME_S*240*step)
                    checks.numeric(arrays["time_s"][tick], DRIFT_TIME_S, "Paired canonical350ms tick", 1e-12)
                    positions[row["id"]] = arrays["position_m"][tick, index].copy()
    return positions, trace_evidence


def audit_rows(rows, plan, groups, checks):
    lookup = {g["id"]: g for g in groups}
    checks.equal(len(rows), len(plan), "All and only actually opened requests are saved")
    checks.equal(len({r["id"] for r in rows}), len(rows), "Every requested ID occurs exactly once")
    for row, expected in zip(rows, plan):
        label = row["id"]
        for key, value in expected.items():
            checks.equal(row[key], value, label+"/frozen request field/"+key)
        group = lookup[row["group_id"]]
        checks.equal(row["eligible"], group["eligible"], label+"/frozen endpoint eligibility")
        checks.equal(bool(row["native_release"]), group["eligible"], label+"/all admitted requests executed exactly once")
        if row["native_release"]:
            checks.equal(row["native_directory"], "native/"+str(row["substeps"]), label+"/resolution cache directory")
            for field in ("q10", "q14"):
                checks.numeric(row["native_initial_"+field], np.asarray(row[field], dtype=np.float32), label+"/actual float32 initial joint state/"+field, 0)
            checks.equal(row["contact_protocol"], dict(substeps=row["substeps"], board_mode="articulated", gravity_mode="passive_only"), label+"/actual prescribed endpoint contact protocol")
            checks.require(row["scalar_score_audit_passed"] is True, label+"/producer scalar agreement recorded")
            checks.require(row["endpoint_collision_pairs"] == [], label+"/actual native endpoint collision check")
        else:
            checks.equal(row["invalid"], "strict_endpoint_or_local_branch_rejected", label+"/rejection retained explicitly")
            checks.require(not row["route_success"] and not row["whole_ball_entry"], label+"/rejection cannot become capture")
            checks.require("native_cache_file" not in row, label+"/rejection has no fabricated trace")
    check_pairing(rows, checks)


def audit_batches(study, rows, checks):
    batches = read(study/"batches.json")
    observed = []
    by_id = {r["id"]: r for r in rows}
    cache_files = set()
    for index, batch in enumerate(batches):
        checks.equal(batch["packet"], index, "Sequential non-retried native packet")
        checks.require(bool(batch["record_ids"]), "No empty native packet")
        checks.require(all(i in by_id for i in batch["record_ids"]), "Packet IDs are actual requested rows")
        actual = [by_id[i] for i in batch["record_ids"]]
        checks.equal({r["substeps"] for r in actual}, {batch["substeps"]}, "One native resolution per packet")
        checks.equal({r["phase"] for r in actual}, {batch["phase"]}, "One phase per packet")
        checks.equal(batch["native_directory"], "native/"+str(batch["substeps"]), "Packet resolution directory")
        released = [r for r in actual if r["native_release"]]
        checks.equal(batch["native_record_ids"], [r["id"] for r in released], "Every native column retained in request order")
        if released:
            name = batch["cache_file"]
            key = (batch["native_directory"], name)
            checks.require(key not in cache_files, "No reused native cache packet")
            cache_files.add(key)
            checks.equal([r["native_cache_row"] for r in released], list(range(len(released))), "Contiguous actual native columns")
            checks.equal({r["native_cache_file"] for r in released}, {name}, "Rows bind their actual packet cache")
        else:
            checks.equal(batch["cache_file"], None, "Rejected-only packet has no native cache")
        observed.extend(batch["record_ids"])
    checks.equal(observed, [r["id"] for r in rows], "Packets account for every completed request exactly in order")


def audit_report(report, rows, groups, positions, gate, checks):
    checks.equal(report["gate"], gate, "Final gate independently reconstructed")
    expected_counts = statistics(rows)
    for key, expected in expected_counts.items():
        if isinstance(expected, float):
            checks.numeric(report["counts"][key], expected, "Final counts/"+key, 1e-10)
        else:
            checks.equal(report["counts"][key], expected, "Final counts/"+key)
    for field, key in (("phase_counts", "phase"), ("role_counts", "role"), ("substep_counts", "substeps")):
        checks.equal(report["counts"][field], dict(Counter(str(r[key]) for r in rows)), "Final complete accounting/"+field)
    checks.equal(len(report["groups"]), len(groups), "All conditional groups displayed even when unexecuted")
    for actual, source in zip(report["groups"], groups):
        selected = [r for r in rows if r["group_id"] == source["id"]]
        for key in ("id", "role", "source_record_id", "source_record_sha256", "values", "bucket_xy_m", "eligible"):
            checks.equal(actual[key], source[key], source["id"]+"/display actual group/"+key)
        values = source["values"]
        for key, value in dict(gap_m=values[4], board_height_m=values[2], tilt_deg=values[3],
                               board_xy_m=values[:2],
                               distance_m=float(np.linalg.norm(np.asarray(source["bucket_xy_m"])-values[:2]))).items():
            checks.equal(actual[key], value, source["id"]+"/display actual geometry/"+key)
        for step in STEPS:
            check_statistics(actual["by_substeps"][str(step)], [r for r in selected if r["substeps"] == step], checks,
                             source["id"]+"/display actual outcome/"+str(step))
        sensitivity = paired_sensitivity(selected, positions) if selected else dict(paired_attempts=0, capture_agreement=0, nominal_position_delta_350ms_max_m=None)
        checks.equal(actual["sensitivity"], sensitivity, source["id"]+"/display actual paired sensitivity")


def audit_traces(study, report, rows, groups, scene, checks):
    selected = []
    for group in groups:
        for step in STEPS:
            selected.extend(r for r in rows if r["group_id"] == group["id"] and r["substeps"] == step
                            and r["repeat_index"] == 0 and r["native_release"])
    checks.equal([t["record_id"] for t in report["trace_examples"]], [r["id"] for r in selected],
                 "Only declared first nominal actual native traces displayed, never outcome-selected")
    lookup = {g["id"]: g for g in groups}
    for trace, row in zip(report["trace_examples"], selected):
        group = lookup[row["group_id"]]
        for key in ("group_id", "role", "substeps", "whole_ball_entry", "max_retention_s", "external_contact_steps",
                    "route_success", "invalid", "invalid_physics", "scalar_score_audit_passed"):
            checks.equal(trace[key], row[key], row["id"]+"/display actual trace metadata/"+key)
        values = group["values"]
        for key, expected in dict(board_xy_m=values[:2], bucket_xy_m=group["bucket_xy_m"], board_height_m=values[2],
                                  tilt_deg=values[3], gap_m=values[4],
                                  release_center_height_m=values[2]+values[4]+scene["ball"]["radius_m"],
                                  distance_m=float(np.linalg.norm(np.asarray(group["bucket_xy_m"])-values[:2]))).items():
            checks.equal(trace[key], expected, row["id"]+"/display actual trace geometry/"+key)
        path = contained(study, row["native_directory"])/row["native_cache_file"]
        with np.load(path, allow_pickle=False) as archive:
            points = archive["position_m"][::row["substeps"]*4, row["native_cache_row"]]
        checks.numeric(trace["position_m"], points, row["id"]+"/display actual saved trajectory points", 0)
        checks.numeric(trace["sample_dt_s"], 1/60, row["id"]+"/display actual trajectory cadence", 0)


def audit(study, source, assets, *, shallow_study, positive_study, positive_audit):
    started, checks = time.monotonic(), AuditChecks()
    study, source, assets = map(Path, (study, source, assets))
    manifest = check_manifest(study, checks)
    spec, report = read(study/"contract.json"), read(study/"report.json")
    unsigned = dict(spec); fp = unsigned.pop("fingerprint", None)
    checks.equal(digest(unsigned), fp, "Full replay contract fingerprint")
    checks.equal(spec["version"], STUDY_VERSION, "Versioned new calibrated replay")
    checks.equal(report["version"], STUDY_VERSION, "Report replay version")
    checks.equal(report["contract_fingerprint"], fp, "Report bound to exact replay contract")
    checks.equal((report["status"], report["phase"]), ("completed", "completed"), "Completed finite replay")
    checks.equal(spec["evidence_kind"], "native_current_ball_calibrated_replay", "Native simulated evidence only")
    checks.equal(report["evidence_sha256"], manifest, "Published report binds every raw artifact")
    check_source_files(source, spec["source_sha256"], checks, "Actual executed source bytes")
    paths = list((source/"sircus/mjlab").glob("*.py")) + [source/"sircus/mjlab/requirements.txt"]
    paths += [source/"sircus/isaac"/n for n in ("geometry.py", "rl_contract.py", "entry_quality.py", "board_route.py")]
    checks.equal(set(spec["source_sha256"]), {str(p.relative_to(source)) for p in paths}, "Source manifest includes complete current execution/audit module set")
    from .learning_solve import required_asset_paths
    actual_assets = {n: sha(contained(assets.parent, n)) for n in required_asset_paths(assets)}
    checks.equal(spec["asset_sha256"], actual_assets, "Actual robot collision assets and poses")
    audit_packages(spec["packages"], checks)
    calibration = check_calibration(study, spec["calibration"], checks)
    parent, original = audit_input_sources(shallow_study, positive_study, positive_audit, spec["input_bindings"], checks, spec["calibration"])
    checks.equal(spec["scene"], parent["scene"], "Historical scene geometry and provisional measurements preserved")
    checks.equal(spec["asset_sha256"], parent["asset_sha256"], "Original exact collision asset ancestry")
    checks.equal(spec["packages"], parent["packages"], "Original native package ancestry")
    for name, expected in spec["frozen_files_sha256"].items():
        checks.equal(sha(contained(study, name)), expected, "Preoutcome frozen inputs/"+name)
    checks.equal(set(spec["frozen_files_sha256"]), {"groups.json", "geometry.json", "plan.json", "calibration.json", "verification.json"}, "All preoutcome group/geometry/plan/calibration files frozen")
    settings = spec["settings"]
    checks.equal(settings["candidate"], dict(board_height_m=.38, tilt_deg=28., gap_m=.3868, release_center_height_m=.8043, distance_m=.65), "Exact single historical candidate, no outcome search")
    checks.equal(settings["substeps"], list(STEPS), "Only canonical64 and fine128 requested")
    checks.equal(settings["gaps_m"], list(WINDOW_GAPS_M), "Five predeclared conditional gap values")
    checks.equal(settings["repeats_per_group"], REPEATS, "Three nominal copies without fresh noise")
    checks.equal(settings["max_requests"], MAX_REQUESTS, "Finite48 request ceiling")
    checks.equal(settings["gate"], dict(positive_captures=3, negative_captures=0, candidate_captures=3,
                 each_resolution=True, required_clearance_m=.005, max_drift_350ms_m=.005, valid_physics=True, oracle_agreement=True), "Literal prespecified conditional gate")
    checks.equal(report["settings"], settings, "Display uses exact executed settings")
    groups, proofs = read(study/"groups.json"), read(study/"geometry.json")
    audit_groups(groups, proofs, original, spec["scene"], checks)
    audit_cpu_endpoints(groups, proofs, spec["scene"], assets, checks)
    initial, window = expected_plan(groups[:3], "replay"), expected_plan(groups[3:], "window")
    checks.equal(read(study/"plan.json"), initial+window, "Complete literal48 request plan frozen before outcomes")
    rows = read_records(study/"records.jsonl")
    checks.require(len(rows) in (18, 48), "Only matched replay or gated full window executed")
    options = read(study/"effective-options.json")
    audit_options(options, checks)
    audit_compiled(study, spec["scene"], assets, calibration, options, checks)
    positions, traces = audit_native(study, rows, spec["scene"], calibration, assets, checks)
    replay = [r for r in rows if r["phase"] == "replay"]
    candidate = [r for r in replay if r["role"] == "candidate"]
    sensitivity = paired_sensitivity(candidate, positions)
    gate = replay_gate(replay, sensitivity["nominal_position_delta_350ms_max_m"])
    checks.equal(read(study/"replay-gate.json"), gate, "Frozen prewindow gate uses actual matched outcomes")
    expected = initial+window if gate["window_opened"] else initial
    audit_rows(rows, expected, groups, checks)
    audit_batches(study, rows, checks)
    audit_report(report, rows, groups, positions, gate, checks)
    audit_traces(study, report, rows, groups, spec["scene"], checks)
    return dict(version=VERSION, status="passed", contract_fingerprint=fp, checks_count=checks.count,
                checks=checks.entries, failures=[], auditor_sha256=sha(Path(__file__)),
                report_sha256=sha(study/"report.json"), evidence_manifest_sha256=sha(study/"evidence-manifest.json"),
                requested_trials=len(rows), native_releases=len(traces), gate=gate,
                elapsed_s=time.monotonic()-started,
                limitations=["Nominal engineering repeats only; no fresh-noise robustness or learned-policy claim.",
                             "Calibrated simulated contact envelope; no real-world calibration or deployment.",
                             "Strict held endpoints only; no swept transit, grasp, perception or actuator validation.",
                             "350ms position check is a declared sensitivity diagnostic, not a full numerical convergence proof."])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("study", "source", "assets", "shallow-study", "positive-study", "positive-audit", "output"):
        parser.add_argument("--"+name, type=Path, required=True)
    args = parser.parse_args()
    if args.output.resolve().is_relative_to(args.study.resolve()) and args.output.name != "audit-replay650.json":
        raise ValueError("Only the designated new audit may be written inside frozen evidence")
    try:
        result = audit(args.study, args.source, args.assets, shallow_study=args.shallow_study,
                       positive_study=args.positive_study, positive_audit=args.positive_audit)
    except (EvidenceError, ValueError, KeyError, OSError) as error:
        result = dict(version=VERSION, status="failed", failures=[str(error)])
        args.output.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
        raise
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    print(json.dumps({k: result[k] for k in ("status", "checks_count", "requested_trials", "native_releases", "gate")}))


if __name__ == "__main__":
    main()
