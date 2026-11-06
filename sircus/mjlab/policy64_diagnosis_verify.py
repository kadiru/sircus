"""Independent CPU audit of post-hoc fixed64 policy diagnostics.

Original held-out outcomes stay immutable. Reusing their positions for diagnosis
does not turn those positions into untouched future test data. This module only
reads saved inputs and dense traces; it never steps a native environment.
"""
from __future__ import annotations

import argparse
from collections import Counter
import copy
import importlib.metadata
import json
import math
from pathlib import Path
import sys
import time

import numpy as np

from . import benchmark_verify as common
from . import calibrated_benchmark_verify as calibrated
from . import replay650_verify as replay
from . import policy64_verify as policy_audit
from . import xy650_verify as xy_audit
from .benchmark_verify import EvidenceError, contained, digest, read, sha


VERSION = "sircus-policy64-diagnosis-audit-v1"
ANALYSIS_VERSION = "sircus-policy64-diagnosis-v1"
PARENT_VERSION = "sircus-policy64-bc-ppo-v1"
PARENT_FP = "d30ad91111cb8beb26438f2271357a103daf50d0399ac3646f3b7f163d87f02e"
PARENT_AUDIT_SHA = "4dc317e0697e426a4708c3c997f8671d97b1f49de6bd4450c56e6487a6e93ac3"
PARENT_REPORT_SHA = "637731856cba291b57a1119b32b56fe97a3c4cbc000d55db4bfb1793c6757cc3"
PARENT_MANIFEST_SHA = "632ce6cd692ea68d862243deb07c71cb605994fd017f7f15a603aa7995bd3514"
NATIVE_SOURCE_COMMIT = "f00a29a42cac69221b0896173f9be785f113a8a6"
CONTEXTS = tuple(f"policy64-test-{index:02d}" for index in (4, 7, 13, 17, 21, 26, 43))
COORDINATES = (("swap_x", (0,)), ("swap_y", (1,)), ("swap_height", (2,)),
               ("swap_tilt", (3,)), ("swap_gap", (4,)), ("swap_height_gap", (2, 4)))
MAX_REQUESTS = 96
NATIVE_VERSION = "sircus-policy64-paired-native-diagnosis-v1"
AuditChecks = replay.AuditChecks


def same(actual, expected, checks, label, tolerance=1e-9):
    if isinstance(expected, dict):
        checks.require(isinstance(actual, dict), label+"/dictionary")
        checks.equal(set(actual), set(expected), label+"/complete fields")
        for key in expected:
            same(actual[key], expected[key], checks, label+"/"+str(key), tolerance)
    elif isinstance(expected, (list, tuple)):
        checks.require(isinstance(actual, (list, tuple)), label+"/sequence")
        checks.equal(len(actual), len(expected), label+"/length")
        for index, value in enumerate(expected):
            same(actual[index], value, checks, label+"/"+str(index), tolerance)
    elif type(expected) is bool:
        checks.require(type(actual) is bool and actual is expected, label+"/explicit boolean")
    elif isinstance(expected, (float, np.floating)):
        checks.numeric(actual, expected, label, tolerance)
    else:
        checks.equal(actual, expected, label)


def audit_parent(study, audit_path, checks):
    """Authenticate all consumed parent bytes against the completed full proof."""
    study, audit_path = Path(study), Path(audit_path)
    for path, expected in ((audit_path, PARENT_AUDIT_SHA), (study/"report.json", PARENT_REPORT_SHA),
                            (study/"evidence-manifest.json", PARENT_MANIFEST_SHA)):
        checks.equal(sha(path), expected, "Exact immutable original fixed64 artifact/"+path.name)
    spec, report, proof = read(study/"contract.json"), read(study/"report.json"), read(audit_path)
    unsigned = dict(spec);fp = unsigned.pop("fingerprint", None)
    checks.equal((fp, digest(unsigned)), (PARENT_FP, PARENT_FP), "Exact original complete contract")
    checks.equal((spec["version"], report["version"]), (PARENT_VERSION, PARENT_VERSION), "Original study version unchanged")
    checks.equal((report["status"], report["phase"]), ("completed", "completed"), "Original finite study completed")
    checks.equal(proof["version"], "sircus-policy64-independent-audit-v1", "Original independent policy auditor")
    checks.equal(proof["status"], "passed", "Original full audit passed")
    checks.require(proof["checks_count"] == len(proof["checks"]) == 346485 and not proof.get("failures")
                   and all(check.get("passed") is True for check in proof["checks"]), "Every original346485 audit check strictly passed")
    checks.equal((proof["report_sha256"], proof["evidence_manifest_sha256"]), (PARENT_REPORT_SHA, PARENT_MANIFEST_SHA), "Original proof binds report and full manifest")
    checks.equal((proof["contract_fingerprint"], report["contract_fingerprint"]), (PARENT_FP, PARENT_FP), "Original report/proof exact protocol")
    manifest = read(study/"evidence-manifest.json")
    checks.equal(report["evidence_sha256"], manifest, "Original report complete raw evidence manifest")
    consumed = ("contract.json", "rows.jsonl", "splits.json", "prior.json", "setup-home.json", "calibration.json", "verification.json",
                "bootstrap/dataset.json", "bootstrap/teachers.json", "bootstrap/fit.json", "selected.json", "updates.jsonl", "test-plan.json")
    for name in consumed:
        checks.equal(sha(contained(study, name)), manifest[name], "Consumed original artifact bytes/"+name)
    rows = replay.read_records(study/"rows.jsonl")
    checks.equal((len(rows), sum(bool(row["native_release"]) for row in rows)), (1284, 1103), "Original requested/native denominators preserved")
    checks.equal(proof["producer_source_sha256"], spec["source_sha256"], "Original executed source bindings unchanged")
    checks.equal(proof["selected"], report["selected"], "Original selected checkpoint unchanged")
    for key in ("bc", "selected"):
        record = report["artifacts"][key]
        checks.equal(sha(contained(study, record["path"])), record["sha256"], "Exact original checkpoint bytes/"+key)
        checks.equal(record["sha256"], manifest[record["path"]], "Original model manifest binding/"+key)
    return spec, report, rows, manifest


def paired_rows(rows):
    result = {}
    for method in ("bc", "ppo"):
        cohort = [row for row in rows if row["role"] == "test" and row["method"] == method]
        if len(cohort) != 50 or len({row["context_id"] for row in cohort}) != 50:
            raise EvidenceError("Exactly50 original nominal rows per paired method required")
        result[method] = {row["context_id"]: row for row in cohort}
    if set(result["bc"]) != set(result["ppo"]):
        raise EvidenceError("BC and PPO layout identities are not paired")
    for key in result["bc"]:
        if result["bc"][key]["bucket_xy_m"] != result["ppo"][key]["bucket_xy_m"]:
            raise EvidenceError("Original paired bucket position differs")
    return result


def pair_summary(rows):
    """Unconditional rates and exact paired deltas; no admission filtering."""
    by = paired_rows(rows)
    transitions = Counter((policy_audit.captured(by["bc"][key]), policy_audit.captured(by["ppo"][key])) for key in by["bc"])
    return dict(layouts=50, bc=calibrated.counts(list(by["bc"].values())), ppo=calibrated.counts(list(by["ppo"].values())),
        both_capture=transitions[(True, True)], bc_only=transitions[(True, False)], ppo_only=transitions[(False, True)], neither=transitions[(False, False)])


def expected_variants(rows):
    """Seven prescribed BC-only contexts, exact commands and six counterfactuals."""
    by = paired_rows(rows)
    selected = tuple(sorted(key for key in by["bc"] if policy_audit.captured(by["bc"][key]) and not policy_audit.captured(by["ppo"][key])))
    if selected != CONTEXTS:
        raise EvidenceError("The literal seven original BC-only layouts changed")
    result = []
    for context in CONTEXTS:
        bc, ppo = by["bc"][context], by["ppo"][context]
        for variant, source in (("bc_exact", bc), ("ppo_exact", ppo)):
            for repeat in range(3):
                result.append(dict(context_id=context, variant=variant, repeat=repeat, values=copy.deepcopy(source["values"]),
                                   bucket_xy_m=copy.deepcopy(source["bucket_xy_m"]), parent_record_id=source["id"], parent_record_sha256=digest(source),
                                   kind="exact_replay" if source["native_release"] else "historical_rejection", origin_method="bc" if variant == "bc_exact" else "ppo"))
        for variant, coordinates in COORDINATES:
            values = copy.deepcopy(ppo["values"])
            for index in coordinates:
                values[index] = bc["values"][index]
            result.append(dict(context_id=context, variant=variant, repeat=0, values=values, bucket_xy_m=copy.deepcopy(ppo["bucket_xy_m"]),
                               parent_record_id=ppo["id"], parent_record_sha256=digest(ppo), kind="coordinate_swap", origin_method="ppo"))
    return result


def outcome(row):
    if row.get("native_release") is False:
        return "rejected"
    if row.get("native_release") is not True:
        return "unknown"
    if row.get("invalid") or row.get("invalid_physics"):
        return "invalid"
    return "capture" if policy_audit.captured(row) else "miss"


def analysis_statistics(rows):
    return dict(requested=len(rows), native=sum(row.get("native_release") is True for row in rows),
        captures=sum(policy_audit.captured(row) for row in rows),
        clean=sum(policy_audit.captured(row) and row.get("route_quality") == 2 for row in rows),
        rejected=sum(row.get("native_release") is False for row in rows),
        native_misses=sum(outcome(row) == "miss" for row in rows), invalid=sum(outcome(row) == "invalid" for row in rows),
        mean_reward=math.fsum(row["learning_reward"] for row in rows)/len(rows) if rows else None)


def geometry_evidence(connection):
    endpoint = connection.get("endpoint_attempts") or {}
    attempted = [pair.get("checked", {}) for pair in endpoint.get("pair_attempts", []) if isinstance(pair, dict)]
    strict = [value.get("strict_geometry") for value in attempted if isinstance(value.get("strict_geometry"), dict)]
    margins = [value["minimum_clearance_m"] for value in strict if type(value.get("minimum_clearance_m")) in (int, float)
               and math.isfinite(value["minimum_clearance_m"])]
    pairs = sorted({tuple(value["closest_pair"]) for value in strict if isinstance(value.get("closest_pair"), list) and len(value["closest_pair"]) == 2})
    return dict(reason=connection.get("reason"), passed=connection.get("passed"), minimum_required_clearance_m=.005,
        checked_branch_pairs=len(attempted), valid_ik_branches=endpoint.get("valid_ik_branches"),
        best_checked_clearance_m=max(margins) if margins else None, limiting_pairs=[list(value) for value in pairs],
        proof_minimum_clearance_m=(connection.get("proof") or {}).get("minimum_clearance_m"), endpoint_reason=endpoint.get("reason"),
        interpretation="No admitted solution found within the fixed solver budget; not a proof of unreachability."
        if connection.get("reason") == "finite_ik_unresolved" else "Endpoint or sampled setup proof; does not establish driven actuator transit.")


def compact_row(row, connection):
    names = ("id", "values", "native_release", "invalid", "invalid_physics", "learning_reward", "board_bounce", "whole_ball_entry",
             "max_retention_s", "closest_opening_distance_m", "external_contact_steps", "board_contact_episodes", "artifact_sha256", "connection_sha256", "source_teacher_id",
             "rim_band_contact", "bucket_contact_before_entry", "entry_class", "minimum_mouth_clearance_mm", "whole_ball_entry_time_s", "capture_time_s", "retention_resets", "first_mouth_passage_time_s")
    result = {key: copy.deepcopy(row.get(key)) for key in names}
    result.update(record_id=row["id"], record_sha256=digest(row), capture=policy_audit.captured(row),
                  clean=policy_audit.captured(row) and row.get("route_quality") == 2, outcome=outcome(row),
                  setup_stage=row.get("geometry_reason"), geometry_evidence=geometry_evidence(connection))
    return result


def original_flights(study, rows, spec, manifest, checks):
    """Reconstruct first-flight diagnostics from parent bytes, not stored labels."""
    grouped, result = {}, {}
    for row in rows:
        if row.get("role") == "test" and row.get("method") in ("bc", "ppo") and row.get("native_release"):
            grouped.setdefault(row["native_cache_file"], []).append(row)
    config = spec["identity"]["calibration"]["score_config"]
    for filename, cohort in sorted(grouped.items()):
        path = contained(study/"native", filename)
        checks.equal(sha(path), manifest["native/"+filename], "Original trace bytes authenticated/"+filename)
        with np.load(path, allow_pickle=False) as archive:
            arrays = {key: archive[key] for key in ("position_m", "velocity_m_s", "forces_N", "time_s", "dt_s", "contact_distance_m") if key in archive.files}
            ids = archive["record_ids"].tolist()
            for row in cohort:
                column = row["native_cache_row"];label = row["id"]
                checks.equal(ids[column], label, label+"/parent native cache column")
                checks.equal(row["native_cache_sha256"], manifest["native/"+filename], label+"/parent record cache bytes")
                trace = {key: value[:, column] if value.ndim >= 2 and key != "time_s" else value for key, value in arrays.items()}
                metric = xy_audit.first_flight(spec["scene"], row["bucket_xy_m"], row["board_normal"], config, trace, valid=not row.get("invalid_physics"))
                if metric["available"]:
                    start, end = metric["start_index"], metric["end_index"]
                    metric.update(rebound_position_m=trace["position_m"][start].tolist(), rebound_velocity_m_s=trace["velocity_m_s"][start].tolist(),
                        first_contact_position_m=trace["position_m"][end].tolist() if metric["cutoff_reason"] == "contact" else None,
                        first_contact_velocity_before_m_s=trace["velocity_m_s"][end].tolist() if metric["cutoff_reason"] == "contact" else None)
                direction = np.asarray(row["bucket_xy_m"])-np.asarray(row["values"][:2]);direction /= np.linalg.norm(direction)
                if metric["first_crossing"] is not None:
                    offset = np.asarray(metric["first_crossing"]["xy_m"])-row["bucket_xy_m"]
                    metric.update(forward_error_m=float(offset@direction), lateral_error_m=float(offset@np.asarray([-direction[1], direction[0]])))
                else:
                    metric.update(forward_error_m=None, lateral_error_m=None)
                checks.numeric(row["trajectory_m"], arrays["position_m"][256::256, column], label+"/original displayed measured points", 0)
                result[label] = dict(record_id=label, record_sha256=digest(row), context_id=row["context_id"], method=row["method"],
                    bucket_xy_m=row["bucket_xy_m"], values=row["values"], board_center_m=row["board_center_m"], board_normal=row["board_normal"], substeps=64,
                    position_m=row["trajectory_m"], sample_dt_s=row["trajectory_dt_s"], capture=policy_audit.captured(row),
                    clean=policy_audit.captured(row) and row.get("route_quality") == 2, whole_ball_entry=row["whole_ball_entry"], max_retention_s=row["max_retention_s"],
                    native_cache_sha256=row["native_cache_sha256"], native_cache_row=column, first_flight=metric,
                    provenance="Observed original native trace; no generated trajectory or new rollout.")
    return result


def optimizer_analysis(study, manifest, checks):
    """Auditor-owned NumPy Gaussian/critic diagnostics over the frozen32 buffers."""
    import torch
    updates = common.lines(study/"updates.jsonl");result = []
    checks.equal([row["index"] for row in updates], list(range(32)), "Every original saved update included in optimization diagnosis")
    for record in updates:
        states = []
        for key in ("behavior", "after_artifact"):
            relative = record[key]["path"];path = contained(study, relative)
            checks.equal(sha(path), manifest[relative], "Original optimization model bytes/"+relative)
            checks.equal(record[key]["sha256"], manifest[relative], "Original update checkpoint reference/"+relative)
            state = common.load_checkpoint(path, checks, relative)
            states.append(common.numpy_state(state["policy"]))
        relative = record["buffer"];path = contained(study, relative)
        checks.equal(sha(path), manifest[relative], "Original optimization buffer bytes/"+relative)
        checks.equal(record["buffer_sha256"], manifest[relative], "Original update buffer reference")
        with np.load(path, allow_pickle=False) as archive:
            observation, latent, old, reward = (archive[key].astype(np.float32) for key in ("observation", "latent", "old_logp", "reward"))
        before, after = states
        mean_before, mean_after = (common.network_forward(state, observation) for state in states)
        std_before, std_after = np.exp(before["logstd"]), np.exp(after["logstd"])
        log_probability = (-.5*((latent-mean_after)/std_after)**2-after["logstd"]-.5*np.log(2*np.pi)).sum(axis=-1)
        ratio = np.exp(log_probability-old)
        kl = (after["logstd"]-before["logstd"]+(std_before**2+(mean_before-mean_after)**2)/(2*std_after**2)-.5).sum(axis=-1)
        mse_before, mse_after = (float(np.mean((common.network_forward(state, observation, prefix="critic").squeeze(-1)-reward)**2)) for state in states)
        result.append(dict(index=record["index"], samples=(record["index"]+1)*32,
            analytical_kl_mean=float(kl.mean()), analytical_kl_max=float(kl.max()), sampled_ratio_clip_fraction=float(((ratio<.8)|(ratio>1.2)).mean()),
            critic_mse_before=mse_before, critic_mse_after=mse_after,
            standard_deviation_before=std_before.astype(float).tolist(), standard_deviation_after=std_after.astype(float).tolist(),
            gradient_norm_max=max(epoch["gradient_norm"] for epoch in record["epochs"]), before_sha256=record["behavior"]["sha256"], after_sha256=record["after_artifact"]["sha256"]))
    return dict(updates=result, bc_retention_penalty=False,
        mean_post_update_kl=math.fsum(row["analytical_kl_mean"] for row in result)/len(result),
        mean_post_update_ratio_clip_fraction=math.fsum(row["sampled_ratio_clip_fraction"] for row in result)/len(result),
        metric_definition="KL and likelihood-ratio fractions compare each saved behavior model with its final model after six epochs on that update buffer. Ratio outside[0.8,1.2] is not an invalid physical action.",
        caveat="PPO clipping compares each update with its own behavior policy, not with the original BC policy. These diagnostics do not establish causality.")


def audit_analysis(study, original, rows, manifest, analysis, checks):
    spec, report = original
    checks.equal((analysis["version"], analysis["status"], analysis["phase"]), (ANALYSIS_VERSION, "completed", "original_analysis"), "Separately frozen original-data CPU diagnosis")
    scope = analysis["scope"].lower()
    checks.require(all(word in scope for word in ("post-hoc", "development", "fresh holdout", "no noise", "128")), "Posthoc reuse and fixed64 limitations explicit")
    source = dict(study_version=PARENT_VERSION, contract_fingerprint=PARENT_FP, report_sha256=PARENT_REPORT_SHA,
        manifest_sha256=PARENT_MANIFEST_SHA, original_audit_sha256=PARENT_AUDIT_SHA, original_audit_checks=346485,
        rows_sha256=manifest["rows.jsonl"], substeps=64, selected=report["selected"], bc=report["artifacts"]["bc"])
    same(analysis["source"], source, checks, "CPU diagnosis exact original parent identity")
    by = paired_rows(rows);ids = sorted(by["bc"]);connections = {}
    for method in ("bc", "ppo"):
        for row in by[method].values():
            name = row["connection_file"]
            checks.equal(sha(contained(study/"native", name)), row["connection_sha256"], row["id"]+"/original exact-action proof bytes")
            checks.equal(row["connection_sha256"], manifest["native/"+name], row["id"]+"/proof authenticated by parent")
            connections[row["id"]] = read(contained(study/"native", name))
    flights = original_flights(study, rows, spec, manifest, checks)
    pairs, transitions, admission = [], Counter(), Counter()
    for key in ids:
        bc, ppo = by["bc"][key], by["ppo"][key]
        delta = (np.asarray(ppo["values"])-np.asarray(bc["values"])).tolist()
        values = {}
        for method, row in (("bc", bc), ("ppo", ppo)):
            values[method] = compact_row(row, connections[row["id"]])
            values[method]["first_flight"] = flights.get(row["id"], {}).get("first_flight")
        pair = dict(context_id=key, bucket_xy_m=bc["bucket_xy_m"], bc=values["bc"], ppo=values["ppo"], delta_values=delta,
            delta_release_height_m=delta[2]+delta[4], shared_anchor=dict(same_id=bc.get("source_teacher_id") == ppo.get("source_teacher_id"),
            same_values=bc.get("effective_source_hint", {}).get("values") == ppo.get("effective_source_hint", {}).get("values"),
            same_observation=bc.get("observation") == ppo.get("observation"), bc_id=bc.get("source_teacher_id"), ppo_id=ppo.get("source_teacher_id")))
        pairs.append(pair);transitions[outcome(bc), outcome(ppo)] += 1
        admission["both_native" if bc["native_release"] and ppo["native_release"] else "bc_only_native" if bc["native_release"] else "ppo_only_native" if ppo["native_release"] else "neither_native"] += 1
    counts = pair_summary(rows)
    counts["bc"], counts["ppo"] = (analysis_statistics([by[method][key] for key in ids]) for method in ("bc", "ppo"))
    counts["transitions"] = [dict(bc_outcome=bc, ppo_outcome=ppo, count=count) for (bc, ppo), count in sorted(transitions.items())]
    jointly = [key for key in ids if by["bc"][key]["native_release"] and by["ppo"][key]["native_release"]]
    counts.update(admission_cohorts=dict(admission), jointly_admitted=dict(layouts=len(jointly),
        bc_captures=sum(policy_audit.captured(by["bc"][key]) for key in jointly), ppo_captures=sum(policy_audit.captured(by["ppo"][key]) for key in jointly),
        caveat="Conditioning on joint admission is descriptive and does not remove rejected requests from the primary denominator."))
    same(analysis["paired_counts"], counts, checks, "Independent unconditional50-layout pairing")
    same(analysis["pairs"], pairs, checks, "Independent exact paired requests, geometry proof summaries and first flights")
    parameters = ["board_x_m", "board_y_m", "board_height_m", "board_tilt_deg", "nominal_plane_gap_m"]
    checks.equal(analysis["parameter_order"], parameters, "Five actual requested action dimensions")
    deltas = []
    for name, cohort in (("all", pairs), ("bc_only", [p for p in pairs if p["bc"]["capture"] and not p["ppo"]["capture"]]),
                          ("ppo_only", [p for p in pairs if p["ppo"]["capture"] and not p["bc"]["capture"]])):
        if not cohort:
            continue
        columns = np.asarray([pair["delta_values"] for pair in cohort])
        deltas.append(dict(group=name, layouts=len(cohort), parameters=parameters, mean_delta=columns.mean(axis=0).tolist(),
            mean_abs_delta=np.abs(columns).mean(axis=0).tolist(), max_abs_delta=np.abs(columns).max(axis=0).tolist(),
            min_delta=columns.min(axis=0).tolist(), max_delta=columns.max(axis=0).tolist()))
    same(analysis["delta_summary"], deltas, checks, "Independent action-change ranges and means")
    reasons = []
    for method in ("bc", "ppo"):
        for reason in sorted({by[method][key].get("invalid") for key in ids if not by[method][key]["native_release"]}):
            cases = [key for key in ids if not by[method][key]["native_release"] and by[method][key].get("invalid") == reason]
            reasons.append(dict(method=method, reason=reason, count=len(cases), context_ids=cases))
    same(analysis["rejection_reasons"], reasons, checks, "All actual setup rejections remain visible")
    same(analysis["actor_shift"], dict(board_height_lower_count=sum(pair["delta_values"][2] < 0 for pair in pairs),
         release_gap_higher_count=sum(pair["delta_values"][4] > 0 for pair in pairs)), checks, "Measured actor shifts, not causal diagnosis")
    expected_examples = [dict(context_id=pair["context_id"], selection_rule="All7 BC-only losses and all5 PPO-only gains, selected post-hoc from the original paired50.",
         basis="original_TEST_now_diagnosis", projection_reference=dict(origin_xy_m=pair["bucket_xy_m"], direction_from_xy_m=pair["bc"]["values"][:2]),
         bc_trace=flights.get(pair["bc"]["record_id"]), ppo_trace=flights.get(pair["ppo"]["record_id"]))
         for pair in pairs if pair["bc"]["capture"] != pair["ppo"]["capture"]]
    same(analysis["examples"], expected_examples, checks, "All12 discordant observed trace examples without cherry-picking")
    same(analysis["optimization_history"], optimizer_analysis(study, manifest, checks), checks, "Independent saved Gaussian/critic update diagnostics", tolerance=1e-4)
    same(analysis["original_comparison"], report["test_comparison"], checks, "Historical four-method results never replaced")
    same(analysis["validation"], report["curves"], checks, "Original checkpoint selection evidence")
    checks.require(analysis["model_options"]["status"] == "proposed_not_trained", "Proposed dynamics model is not represented as trained")
    checks.require("newly frozen future holdouts" in analysis["model_options"]["evaluation"], "Future models need fresh excluded evaluation positions")
    checks.equal(analysis["native_study"], dict(status="pending", plan_requests=96, observed_requests=0), "Frozen analysis precedes counterfactual outcomes")
    checks.equal([item["status"] for item in analysis["hypotheses"]], ["observed_association_counterfactuals_pending", "hypothesis_not_proven", "not_supported_by_this_comparison"], "Associations and proposed mechanisms are not relabeled causal proof")
    return counts


def expected_plan(rows):
    by = paired_rows(rows);expected_variants(rows)
    plan = []
    def request(source, context, variant, phase, repeat, coordinates=()):
        values = copy.deepcopy(source["values"])
        kind = "exact_replay" if source["native_release"] else "historical_rejection"
        if coordinates:
            bc = by["bc"][context]
            for index in coordinates:
                values[index] = bc["values"][index]
            kind = "coordinate_swap"
        result = dict(id=f"diag64-{phase}-{context}-{variant}-r{repeat}", phase=phase,
            role=source["role"] if phase.startswith("controls_") else "diagnostic", context_id=context,
            variant=variant, kind=kind, repeat=repeat, substeps=64, origin_method=source["method"],
            parent_record_id=source["id"], parent_record_sha256=digest(source), bucket_xy_m=copy.deepcopy(source["bucket_xy_m"]),
            values=values, original_native_release=source["native_release"] is True, swapped_coordinates=list(coordinates),
            selection_or_learning=False, source_teacher_id=source["source_teacher_id"], observation=copy.deepcopy(source["observation"]))
        result["command_values"] = copy.deepcopy(values)
        for key in ("source_hint", "effective_source_hint", "context_resolution_file", "context_resolution_sha256", "context_home_sha256",
                    "context_layout_eligible", "context_anchor_admitted", "context_hard_reject"):
            result[key] = copy.deepcopy(source[key])
        if kind == "exact_replay":
            for key in ("q10", "q14", "connection_file", "connection_sha256"):
                result[key] = copy.deepcopy(source[key])
            result.update(parent_native_cache_file=source["native_cache_file"], parent_native_cache_sha256=source["native_cache_sha256"])
        elif kind == "historical_rejection":
            result["historical_rejection_reason"] = source["invalid"]
        return result
    controls = [next(row for row in rows if row["phase"] == "controls_pre" and row["role"] == role and row.get("repeat") == 0)
                for role in ("positive_control", "negative_control")]
    for phase in ("controls_pre", "diagnostic", "controls_post"):
        if phase.startswith("controls_"):
            plan.extend(request(source, source["role"], "control_exact", phase, repeat) for source in controls for repeat in range(3))
            continue
        for context in CONTEXTS:
            for method in ("bc", "ppo"):
                plan.extend(request(by[method][context], context, method+"_exact", phase, repeat) for repeat in range(3))
            plan.extend(request(by["ppo"][context], context, variant, phase, 0, coordinates) for variant, coordinates in COORDINATES)
    return plan


def native_counts(rows):
    return dict(attempts=len(rows), native_releases=sum(bool(row.get("native_release")) for row in rows),
        captures=sum(policy_audit.captured(row) for row in rows), clean=sum(policy_audit.captured(row) and row.get("route_quality") == 2 for row in rows),
        rejected=sum(not row.get("native_release") for row in rows), invalid_physics=sum(bool(row.get("invalid_physics")) for row in rows))


def native_control_gate(rows):
    positive = [row for row in rows if row["role"] == "positive_control"]
    negative = [row for row in rows if row["role"] == "negative_control"]
    good = (len(rows) == 6 and len(positive) == len(negative) == 3 and all(policy_audit.captured(row) for row in positive)
        and all(row.get("native_release") and not policy_audit.captured(row) and not row.get("whole_ball_entry") for row in negative)
        and all(not row.get("invalid") and not row.get("invalid_physics") and row.get("scalar_score_audit_passed") is True for row in rows))
    return dict(status="passed" if good else "failed", counts=native_counts(rows))


def audit_contract(spec, report, parent, parent_report, original_rows, parent_manifest, frozen_analysis_sha, checks):
    checks.equal(spec["version"], NATIVE_VERSION, "New posthoc native diagnostic protocol")
    unsigned = dict(spec);fp = unsigned.pop("fingerprint", None)
    checks.equal(digest(unsigned), fp, "Full preoutcome96 protocol fingerprint")
    checks.equal(report["version"], NATIVE_VERSION, "New native report version")
    checks.equal((report["status"], report["phase"]), ("completed", "completed"), "All96 diagnostic requests completed")
    checks.equal(report["contract_fingerprint"], fp, "Native report exact frozen plan")
    for key, value in dict(substeps=64, maximum_requests=96, maximum_acquired_s=3600, contexts=[key.removeprefix("policy64-test-") for key in CONTEXTS],
                           swaps=[dict(variant=variant, coordinates=list(coordinates)) for variant, coordinates in COORDINATES], training=False, noise=False).items():
        checks.equal(spec[key], value, "Literal no-training finite diagnostic/"+key)
    checks.equal(spec["analysis_sha256"], frozen_analysis_sha, "CPU posthoc analysis frozen before native outcomes")
    checks.equal(report["analysis_sha256"], frozen_analysis_sha, "Native report retains same frozen CPUanalysis")
    checks.equal((report["training"], report["substeps"]), (False, 64), "Native diagnosis is not new policy training")
    checks.require("post-hoc" in spec["scope"].lower() and "no jitter" in spec["scope"].lower(), "Nominal diagnostic scope remains explicit")
    checks.equal(spec["scene"], parent["scene"], "No quiet geometry/material change")
    checks.equal(spec["identity"], parent["identity"], "Same original exact calibration, home and physics identity")
    checks.equal(spec["packages"], parent["packages"], "Same pinned package ancestry")
    checks.equal(spec["native_source_sha256"], parent["source_sha256"], "All155 original executed native modules preserved")
    expected = expected_plan(original_rows)
    same(spec["plan"], expected, checks, "Independent exact96 requests; no outcome-adaptive resampling", 0)
    bindings = spec["parent_bindings"]
    required = dict(parent_contract_fingerprint=PARENT_FP, parent_report_sha256=PARENT_REPORT_SHA,
        parent_contract_sha256=parent_manifest["contract.json"], parent_manifest_sha256=PARENT_MANIFEST_SHA,
        parent_audit_sha256=PARENT_AUDIT_SHA, parent_checks=346485, parent_rows_sha256=parent_manifest["rows.jsonl"],
        parent_evidence_sha256=parent_manifest, analysis_sha256=frozen_analysis_sha,
        parent_records_sha256={row["parent_record_id"]: row["parent_record_sha256"] for row in expected})
    same(bindings, required, checks, "Complete consumed parent/analysis provenance", 0)
    checks.equal(report["gate"], dict(status="passed", controls_passed=True, complete=True,
        scope="Diagnostic execution completion only; no new policy success-rate claim"), "Execution completeness is not policy improvement")
    return expected


def audit_requests(diagnosis, parent_study, rows, plan, parent, original_rows, checks):
    from .calibrated_admission import verify_resolution, verify_connection
    checks.equal(len(rows), MAX_REQUESTS, "Every planned96 slot retained including historical rejects")
    checks.equal([row["id"] for row in rows], [request["id"] for request in plan], "Exact predeclared request order and unique IDs")
    checks.equal(replay.read_records(diagnosis/"native/shots.jsonl"), rows, "Native request log duplicates every observed/rejected slot without omission")
    source = {row["id"]: row for row in original_rows}
    config, home = parent["identity"]["calibration"]["score_config"], parent["identity"]["setup"]["home"]
    prior = read(parent_study/"prior.json")["records"]
    for row, request in zip(rows, plan):
        label = row["id"]
        for key, value in request.items():
            same(row[key], value, checks, label+"/exact requested field/"+key, 0)
        checks.require(type(row.get("native_release")) is bool and row.get("action_substituted") is False, label+"/explicit release and no action replacement")
        resolution_path = contained(diagnosis/"native", row["context_resolution_file"])
        checks.equal(sha(resolution_path), row["context_resolution_sha256"], label+"/new copied parent-resolution bytes")
        parent_resolution = contained(parent_study/"native", row["context_resolution_file"])
        checks.equal(sha(parent_resolution), row["context_resolution_sha256"], label+"/exact original resolution bytes")
        resolution = read(resolution_path);resolution.update(file=row["context_resolution_file"], sha256=row["context_resolution_sha256"], cache_hit=True)
        checks.require(verify_resolution(resolution, prior, home), label+"/unchanged geometry-only source/home resolution")
        if row["kind"] == "historical_rejection":
            checks.require(not row["native_release"], label+"/old rejected request cannot be silently repaired")
            checks.equal(row["invalid"], source[row["parent_record_id"]]["invalid"], label+"/same historical reject reason")
            checks.require(not any(key in row for key in ("q10", "q14", "native_cache_file", "native_cache_sha256", "native_initial_q10", "native_initial_q14")), label+"/no fabricated branch or observed flight for old rejection")
        else:
            path = contained(diagnosis/"native", row["connection_file"])
            checks.equal(sha(path), row["connection_sha256"], label+"/exact same-action connection proof bytes")
            connection = read(path)
            checks.require(verify_connection(connection, resolution, home, row["values"]), label+"/unchanged strict endpoint/sampled setup verification")
            if row["kind"] == "exact_replay":
                checks.equal(sha(contained(parent_study/"native", request["connection_file"])), row["connection_sha256"], label+"/frozen original admitted proof")
                checks.require(row["native_release"], label+"/every original admitted branch actually replayed")
                checks.require(connection["passed"], label+"/original strict setup pass retained")
            else:
                checks.equal(row["kind"], "coordinate_swap", label+"/known controlled intervention")
                checks.equal(row["native_release"], bool(connection["passed"]), label+"/fresh exact-action admission determines native release")
            if row["native_release"]:
                for key in ("q10", "q14"):
                    checks.numeric(row[key], connection[key], label+"/exact admitted joint branch/"+key, 0)
        if row["native_release"]:
            checks.equal(row["contact_protocol"], dict(substeps=64, board_mode="articulated", gravity_mode="passive_only"), label+"/only unchanged prescribed64 physics")
            checks.equal(row["scoring_version"], "sircus-calibrated-native-capture-v2", label+"/same complete-shot scorer")
            checks.require(row.get("scalar_score_audit_passed") is True, label+"/online independent oracle")
            checks.equal(row["adapter_directory"], "native", label+"/native adapter directory")
            checks.equal((row["effective_options_file"], row["effective_options_sha256"]), ("effective-options.json", sha(diagnosis/"native/effective-options.json")), label+"/actual allocated options bytes")
            for key in ("q10", "q14"):
                checks.numeric(row["native_initial_"+key], np.asarray(row[key], dtype=np.float32).astype(float), label+"/observed exact float32 initial joints/"+key, 0)
        else:
            checks.require(not row.get("route_success") and not row.get("invalid_physics") and bool(row.get("invalid")), label+"/geometry rejection remains a failed attempt, not a physics failure")
            missing = xy_audit.first_flight(parent["scene"], row["bucket_xy_m"], [0., 0., 1.], config, {})
            same(row["first_flight"], missing, checks, label+"/unavailable first flight marked missing", 0)
    checks.equal(sum(row["kind"] == "historical_rejection" for row in rows), 15, "All15 retained historical-reject slots in denominator")


def audit_first_flights(diagnosis, rows, parent, checks):
    grouped = {}
    for row in rows:
        if row["native_release"]:
            grouped.setdefault(row["native_cache_file"], []).append(row)
    for filename, cohort in grouped.items():
        with np.load(contained(diagnosis/"native", filename), allow_pickle=False) as archive:
            arrays = {key: archive[key] for key in ("position_m", "velocity_m_s", "forces_N", "time_s", "dt_s", "contact_distance_m") if key in archive.files}
            for row in cohort:
                column = row["native_cache_row"]
                trace = {key: value[:, column] if value.ndim >= 2 and key != "time_s" else value for key, value in arrays.items()}
                expected = xy_audit.first_flight(parent["scene"], row["bucket_xy_m"], row["board_normal"], parent["identity"]["calibration"]["score_config"], trace,
                                                valid=row.get("invalid_physics") is False)
                same(row["first_flight"], expected, checks, row["id"]+"/independent new native first-flight diagnosis")
                checks.numeric(row["trajectory_m"], arrays["position_m"][256::256, column], row["id"]+"/actual saved sparse trajectory", 0)
                checks.numeric(row["trajectory_dt_s"], 1/60, row["id"]+"/actual saved trajectory cadence", 0)


def audit_native_report(report, rows, checks):
    same(report["counts"], native_counts(rows), checks, "Complete96 diagnostic accounting")
    gates = {key: native_control_gate([row for row in rows if row["phase"] == phase]) for key, phase in (("pre", "controls_pre"), ("post", "controls_post"))}
    checks.equal(report["control_gates"], gates, "Both before/after native control gates independently reconstructed")
    checks.require(all(gate["status"] == "passed" for gate in gates.values()), "Fresh exact positive/native-miss controls passed before and after")
    contexts = []
    for context in CONTEXTS:
        cohort = [row for row in rows if row["context_id"] == context]
        contexts.append(dict(context_id=context, variants={variant: native_counts([row for row in cohort if row["variant"] == variant])
            for variant in ["bc_exact", "ppo_exact", *[name for name, _ in COORDINATES]]}))
    same(report["contexts"], contexts, checks, "Every context/variant shown including unchanged historical rejections")
    checks.require(type(report["reported_worker_elapsed_s"]) in (int, float) and 0 <= report["reported_worker_elapsed_s"] <= 3600,
                   "Finite worker-time ceiling, not a speed benchmark")
    return gates


def audit_sources(source, native_source, spec, parent, checks):
    source, native_source = Path(source).resolve(), Path(native_source).resolve()
    expected = {"sircus/mjlab/policy64_diagnosis.py", "sircus/mjlab/policy64_diagnosis_native.py", "sircus/mjlab/policy64_diagnosis_verify.py"}
    checks.equal(set(spec["diagnostic_source_sha256"]), expected, "All three new diagnostic producer/analyzer/checker modules frozen")
    for name, value in spec["diagnostic_source_sha256"].items():
        checks.equal(sha(contained(source, name)), value, "Actually executed new diagnostic source/"+name)
    checks.equal(spec["worker_sha256"], spec["diagnostic_source_sha256"]["sircus/mjlab/policy64_diagnosis_native.py"], "Standalone worker exact frozen byte identity")
    for name, value in spec["native_source_sha256"].items():
        checks.equal(sha(contained(native_source, name)), value, "Original f00a29a native source unchanged/"+name)
    checks.equal(spec["native_source_sha256"], parent["source_sha256"], "Existing155-module native runtime preserved")
    dependencies = {}
    for module in tuple(sys.modules.values()):
        name, filename = getattr(module, "__name__", ""), getattr(module, "__file__", None)
        if not filename or not (name.startswith("sircus.mjlab.") or name.startswith("sircus.isaac.")):
            continue
        path = Path(filename).resolve()
        root = source if path.is_relative_to(source) else native_source if path.is_relative_to(native_source) else None
        checks.require(root is not None, "Only explicit new-audit/original-native source roots imported/"+name)
        relative = str(path.relative_to(root))
        if len(Path(relative).parts) != 3 or path.suffix != ".py":
            continue
        hashes = spec["diagnostic_source_sha256"] if relative in expected else spec["native_source_sha256"]
        if relative.startswith("sircus/isaac/") and relative not in hashes:
            continue
        checks.require(relative in hashes, "Every actually imported core dependency byte-bound/"+relative)
        checks.equal(sha(path), hashes[relative], "Actual independent imported core/oracle bytes/"+relative)
        dependencies[relative] = sha(path)
    checks.require("sircus/mjlab/calibrated_score_oracle.py" in dependencies, "Independent dense calibrated scorer actually imported")
    checks.equal({name: importlib.metadata.version(name) for name in spec["packages"]}, spec["packages"], "Actual CPU audit package versions equal pinned native source")
    return dict(version=VERSION, auditor_sha256=sha(Path(__file__)), dependencies_sha256=dependencies,
                diagnostic_source_sha256=spec["diagnostic_source_sha256"], native_source_sha256=spec["native_source_sha256"])


def audit(study, diagnosis, source, native_source, assets, *, parent_audit, analysis):
    start, checks = time.monotonic(), AuditChecks()
    study, diagnosis, source, native_source, assets, parent_audit, analysis = map(Path, (study, diagnosis, source, native_source, assets, parent_audit, analysis))
    parent, parent_report, original_rows, parent_manifest = audit_parent(study, parent_audit, checks)
    manifest = replay.check_manifest(diagnosis, checks)
    spec, report, frozen_analysis, combined = read(diagnosis/"contract.json"), read(diagnosis/"report.json"), read(diagnosis/"analysis.json"), read(analysis)
    same(report["evidence_sha256"], manifest, checks, "New native report binds complete raw evidence")
    frozen_sha = sha(diagnosis/"analysis.json")
    plan = audit_contract(spec, report, parent, parent_report, original_rows, parent_manifest, frozen_sha, checks)
    same(read(diagnosis/"source-bindings.json"), spec["parent_bindings"], checks, "Copied full parent/analysis provenance binding")
    for filename in ("calibration.json", "verification.json", "setup-home.json", "prior.json"):
        checks.equal(sha(diagnosis/filename), parent_manifest[filename], "Copied unchanged original native input/"+filename)
        checks.equal(sha(study/filename), parent_manifest[filename], "Original native input still unchanged/"+filename)
    expected_combined = copy.deepcopy(frozen_analysis)
    expected_combined.update(contract_fingerprint=spec["fingerprint"], phase="completed", status="completed", native_study=report, evidence_sha256=manifest)
    same(combined, expected_combined, checks, "Combined publication is an exact external merge, never native evidence mutation", 0)
    paired = audit_analysis(study, (parent, parent_report), original_rows, parent_manifest, frozen_analysis, checks)
    rows = replay.read_records(diagnosis/"records.jsonl")
    audit_requests(diagnosis, study, rows, plan, parent, original_rows, checks)
    calibrated.check_rows(rows, checks)
    calibration = replay.check_calibration(diagnosis, spec["identity"]["calibration"], checks)
    from .learning_solve import required_asset_paths
    actual_assets = {name: sha(contained(assets.parent, name)) for name in required_asset_paths(assets)}
    checks.equal(actual_assets, spec["identity"]["asset_sha256"], "Actual original native collision assets and pose files")
    policy_audit.audit_native_options(diagnosis/"native", parent["scene"], calibration, assets, checks)
    traces = calibrated.rescore_rows(diagnosis/"native", rows, parent["scene"], calibration, checks, "New96 fixed64 diagnostic", parent["identity"]["trace_schema"], assets=assets)
    audit_first_flights(diagnosis, rows, parent, checks)
    gates = audit_native_report(report, rows, checks)
    identity = audit_sources(source, native_source, spec, parent, checks)
    return dict(version=VERSION, status="passed", contract_fingerprint=spec["fingerprint"], checks_count=checks.count, checks=checks.entries, failures=[],
        report_sha256=sha(analysis), evidence_manifest_sha256=sha(diagnosis/"evidence-manifest.json"), native_report_sha256=sha(diagnosis/"report.json"),
        analysis_report_sha256=sha(analysis), frozen_analysis_sha256=frozen_sha, auditor_sha256=sha(Path(__file__)), auditor_identity=identity,
        source=copy.deepcopy(frozen_analysis["source"]), source_parent=dict(contract_fingerprint=PARENT_FP, report_sha256=PARENT_REPORT_SHA,
            manifest_sha256=PARENT_MANIFEST_SHA, original_audit_sha256=PARENT_AUDIT_SHA),
        producer_source_sha256=spec["diagnostic_source_sha256"], native_source_sha256=spec["native_source_sha256"],
        requested_trials=len(rows), native_releases=len(traces), paired_counts=paired, counts=native_counts(rows), control_gates=gates,
        elapsed_s=time.monotonic()-start, maximum_numeric_error=checks.numeric_error,
        limitations=["Post-hoc development diagnosis on seven selected original TEST losses; the original50 positions are not untouched future holdouts.",
                     "Nominal64-only replay and coordinate interventions; historical128 failures and original policies/results remain unchanged.",
                     "Three exact repeats are reproducibility checks, not noise robustness or independent training seeds.",
                     "Finite strict IK/clearance rejection does not prove physical unreachability; sampled setup paths do not validate driven actuator motion.",
                     "No retraining, inference activation, perception or physical hardware validation."])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("study", "diagnosis", "source", "native-source", "assets", "parent-audit", "analysis", "output"):
        parser.add_argument("--"+name, type=Path, required=True)
    args = parser.parse_args()
    if any(args.output.resolve().is_relative_to(path.resolve()) for path in (args.study, args.diagnosis)):
        raise ValueError("Write the independent new proof outside both immutable studies")
    try:
        result = audit(args.study, args.diagnosis, args.source, args.native_source, args.assets, parent_audit=args.parent_audit, analysis=args.analysis)
    except Exception as error:
        args.output.write_text(json.dumps(dict(version=VERSION, status="failed", failures=[type(error).__name__+": "+str(error)]), indent=2, allow_nan=False)+"\n")
        raise
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    print(json.dumps({key: result[key] for key in ("status", "checks_count", "requested_trials", "native_releases")}))


if __name__ == "__main__":
    main()
