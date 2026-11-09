"""Independent CPU audit of the frozen TRAIN-only candidate comparison.

Every admitted ball flight is reconstructed by the unchanged benchmark scorer
from the full-rate native observations. This verifier never loads a policy,
constructs an optimizer, searches IK, starts a simulator or authorizes release.
"""

import argparse
from collections import Counter
import copy
import hashlib
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

from continuous_feasibility import canonical_hash
from continuous_native_audit import same as original_same
from diverse_pilot_audit import verify_settling_history
from diverse_pilot_curate import contact_receipts
from expanded_pose_menu import endpoint_targets
from joint_feasibility import preflight
from prepare_assets import chain
from settled_transit import protocol as setup_protocol
from shot_benchmark_audit import (read_trial, verify_native_trial, verify_home,
                                  summary as native_summary)
from shot_benchmark_reach_audit import fit_details
from shot_benchmark_report import safe_path

SCHEMA = "sircus-candidate-native-independent-audit-v1"
PLAN_SCHEMA = "sircus-frozen-candidate-comparison-v1"
RESULT_SCHEMA = "sircus-candidate-native-result-v1"
METHODS = ("parent", "candidate_bank")
ARMS = ("Arm10", "Arm14")
SOURCE = Path(__file__).resolve().parent
SCORE_FILES = ("rl_contract.py", "board_route.py", "entry_quality.py", "farm_contract.py",
               "settling_contract.py", "shot_benchmark_reward.py")


def require(value, message):
    if not value:
        raise ValueError(message)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_bytes(), parse_constant=lambda x: (_ for _ in ()).throw(ValueError("Nonfinite JSON " + x)))


def unsigned(value, label):
    require(isinstance(value, dict), "Expected " + label)
    payload = copy.deepcopy(value)
    fingerprint = payload.pop("fingerprint", None)
    require(fingerprint == canonical_hash(payload), label + " fingerprint changed")
    return payload


def validate_plan(plan):
    value = unsigned(plan, "Comparison plan")
    require(value.get("schema") == PLAN_SCHEMA and value.get("seed") == 20261122
            and value.get("optimizer_updates") == value.get("native_shots_observed") == 0
            and value.get("held_out_shot_files_read") == 0 and value.get("release_authorized") is False,
            "Not a frozen zero-update comparison plan")
    cells = value["cells"]
    require(len(cells) == len({c["id"] for c in cells}) == len({tuple(c["bucket_xy_m"]) for c in cells}) == 35
            and all(c.get("split") == "train" for c in cells), "Comparison requires 35 distinct TRAIN contexts")
    require(value.get("cohort_order") == list(METHODS) and len(value["cohorts"]) == 2
            and value.get("request_count") == len(value["requests"]) == 70, "Comparison budget/order changed")
    for index, cohort in enumerate(value["cohorts"]):
        require(cohort.get("method") == METHODS[index] and cohort.get("request_count") == 35
                and all(len(cohort[k]) == 35 for k in ("actions", "native_enabled", "decline_reasons", "ik_seeds")),
                "Native cohort shape changed")
        for i, cell in enumerate(cells):
            enabled = cohort["native_enabled"][i]
            require(type(enabled) is bool, "Native enabled flags must be Boolean")
            require((cohort["decline_reasons"][i] is None) if enabled else
                    isinstance(cohort["decline_reasons"][i], str) and bool(cohort["decline_reasons"][i]),
                    "Predeclined request needs explicit reason")
            request = value["requests"][index*35+i]
            require(request.get("request_index") == index*35+i and request.get("cohort") == index
                    and request.get("method") == METHODS[index] and request.get("slot") == i
                    and request.get("cell_id") == cell["id"] and request.get("action") == cohort["actions"][i]
                    and request.get("predeclined") is (not enabled)
                    and request.get("decline_reason") == cohort["decline_reasons"][i],
                    "Prescheduled request membership changed")
    require(value["cohorts"][0]["native_enabled"] == [True]*35
            and value["cohorts"][0]["ik_seeds"] == [None]*35, "Parent baseline was changed by geometry filtering")
    return value


def same_path(actual, expected):
    """Keep gate/sample identity exact; bound only numeric readback roundoff.

    Original NumPy/SciPy producers and current CPU readers can differ in the
    last few bits of reconstructed distance metrics. The inherited numerical
    comparison tolerance is 1e-7 (0.1 micrometre for path distances in metres).
    Geometry acceptance is recomputed independently with its original margins;
    this comparison never changes a decision, threshold or release guard.
    """
    if actual is None or expected is None:
        return actual is None and expected is None
    if not isinstance(actual, dict) or not isinstance(expected, dict):
        return False
    for name in ("accepted", "bucket_proxy_checked"):
        if type(actual.get(name)) is not bool or type(expected.get(name)) is not bool or actual[name] is not expected[name]:
            return False
    if actual.get("reasons") != expected.get("reasons") or actual.get("scope") != expected.get("scope"):
        return False
    checks, saved_checks = actual.get("checks"), expected.get("checks")
    if not isinstance(checks, list) or not isinstance(saved_checks, list) or len(checks) != len(saved_checks):
        return False
    for check, saved in zip(checks, saved_checks):
        if not isinstance(check, dict) or not isinstance(saved, dict):
            return False
        if (type(check.get("fraction")) not in (int, float) or type(saved.get("fraction")) not in (int, float)
                or check["fraction"] != saved["fraction"] or check.get("reasons") != saved.get("reasons")):
            return False
    return original_same(actual, expected, tolerance=1e-7)


def verify_initial_geometry(trial, scene, cells, links, seeds):
    """Recheck saved fits and path without invoking an IK optimizer."""
    accepted = 0
    for i, (cell, action, row, enabled) in enumerate(zip(cells, trial["actions"], trial["rows"], trial["enabled"])):
        if not enabled:
            require(row["invalid"] and not row["qualification"]["accepted"], "Predeclined request bypassed admission")
            continue
        starts = {n: np.asarray(trial["home"]["measured_joint_rad"][n][i], dtype=float) for n in ARMS}
        seed = starts if seeds[i] is None else {n: seeds[i][n]["q"] for n in ARMS}
        saved = row["qualification"]
        targets = endpoint_targets(scene, cell, action)
        for name, fit in saved.get("ik", {}).items():
            require(name in ARMS, "Unknown native IK arm")
            fit_details(targets[name], fit, links[name], np.eye(4), name == "Arm14")
        called = []

        def replay(target, robot_links, actual_seed, normal_only):
            name = "Arm14" if normal_only else "Arm10"
            require(name in saved.get("ik", {}) and np.array_equal(target, targets[name])
                    and np.array_equal(actual_seed, seed[name]), "Native saved IK objective/seed changed")
            called.append(name)
            return copy.deepcopy(saved["ik"][name])

        prepared = preflight(scene, cell, action, links, starts, seeds=seed, solver=replay,
                             solver_protocol="native-comparison-saved-fit-recheck-v1")
        require(called == list(saved.get("ik", {})), "Native IK gate sequence changed")
        require(prepared["qualification"]["accepted"] == saved["accepted"]
                and prepared["qualification"]["reason"] == saved["reason"], "Native geometry qualification changed")
        require(same_path(prepared["path"], row["path_proxy"]), "Native path proxy changed")
        if prepared["accepted"]:
            require(row.get("path_rejection") is None, "Admitted path has rejection reason")
            accepted += 1
        else:
            require(row["invalid"] is not None, "Native release bypassed rejected geometry/path")
    return dict(requests=len(cells), initial_geometry_path_admitted=accepted,
                initial_geometry_path_declined=sum(trial["enabled"])-accepted,
                scope="Original saved-fit FK and actual-start path reconstruction; no IK optimization or collision certification")


def verify_controls(run, scene, expected_hashes):
    require(set(expected_hashes) == {"probes.json", "rim-rays.json"}, "Missing startup control byte bindings")
    for name, expected in expected_hashes.items():
        require(digest(safe_path(run, name)) == expected, "Startup control bytes changed: " + name)
    contact, rim = contact_receipts(run)
    raw = read(safe_path(run, "rim-rays.json"))
    rays = raw["rays"]
    angles = [0, 90, 180, 270, 45, 135, 225, 315]
    require(isinstance(rays, list) and 1 <= len(rays) <= len(angles)
            and [r["azimuth_deg"] for r in rays] == angles[:len(rays)], "Startup rim-ray sequence changed")
    found = np.zeros(35, bool)
    directions = np.zeros((35, 2))
    for index, ray in enumerate(rays):
        forces = np.asarray(ray["first_contact_N"], dtype=float)
        require(forces.shape == (35, 6) and np.isfinite(forces).all() and (forces >= 0).all(), "Invalid startup rim contact readback")
        clear = (forces[:,1] > .01) & (forces[:,[0,2,3,4,5]].max(axis=1) <= .01)
        theta = math.radians(ray["azimuth_deg"])
        directions[clear & ~found] = [math.cos(theta), math.sin(theta)]
        found |= clear
        require(not found.all() or index == len(rays)-1, "Startup rim search continued after all cells qualified")
    require(found.tolist() == raw["passed"] == [True]*35
            and np.allclose(directions, raw["directions"], atol=1e-12, rtol=0)
            and abs(raw["clearance_m"]-(scene["ball"]["radius_m"]+.012)) <= 1e-12,
            "Startup rim qualification differs from measured force gates")
    return dict(contact=contact, rim=rim, per_ray_force_gates_recomputed=True,
                scope="Shared startup diagnostics, not part of the 70 method requests")


def verify_metrics(metrics):
    require(isinstance(metrics, dict), "Native metrics missing")
    for name in ("wall_s", "cpu_s", "peak_rss_mib"):
        require(name in metrics, "Missing native resource measurement: " + name)
    for name, value in metrics.items():
        if name.endswith(("_s", "_mib")) and value is not None:
            require(type(value) in (int, float) and math.isfinite(value) and value >= 0,
                    "Invalid native resource measurement: " + name)
    return copy.deepcopy(metrics)


def audit_run(run, comparison_plan_path, *, checkpoint_path, assets_path):
    run, comparison_plan_path, assets_path = Path(run).resolve(), Path(comparison_plan_path).resolve(), Path(assets_path)
    plan = read(comparison_plan_path)
    checked = validate_plan(plan)
    result, spec, binding = (read(safe_path(run, name)) for name in ("result.json", "contract.json", "experiment-binding.json"))
    expected_sha = checked["input_bindings"]["original"]["checkpoint_sha256"]
    require(result.get("schema") == RESULT_SCHEMA and result.get("status") == "completed"
            and result.get("comparison_fingerprint") == plan["fingerprint"]
            and result.get("optimizer_updates") == 0 and result.get("native_requests") == 70,
            "Completed zero-update 70-request native result required")
    require(digest(checkpoint_path) == result.get("checkpoint_sha256_before") == result.get("checkpoint_sha256_after") == expected_sha,
            "Frozen parent checkpoint changed")
    require(binding.get("comparison_sha256") == digest(comparison_plan_path)
            and binding.get("comparison_fingerprint") == plan["fingerprint"]
            and binding.get("setup_controller") == setup_protocol(), "Comparison/controller binding changed")
    source = checked["native_binding"]["original_source"]
    require(binding.get("original_source") == source, "Original native source identity changed")
    require(unsigned(spec, "Native original geometry") is not None and spec["fingerprint"] == source["geometry_fingerprint"],
            "Native scene geometry fingerprint changed")
    require(digest(safe_path(run, "selected-camera.json")) == source["camera_sha256"], "Original selected camera changed")
    actual_sources = {p.name:digest(p) for p in sorted(SOURCE.glob("*.py"))}
    require(binding.get("source_sha256") == result.get("source_sha256") == actual_sources, "Executed source bytes differ from auditor source")
    for name in SCORE_FILES:
        expected = checked["native_binding"].get("original_scoring_source_sha256", {}).get(name)
        require(expected == actual_sources[name], "Original scorer/reward source changed: " + name)
    links = {}
    for name, filename in (("Arm10", "dropper.urdf"), ("Arm14", "board_arm.urdf")):
        path = assets_path/filename
        require(digest(path) == checked["input_bindings"]["assets_sha256"][name], "Original robot URDF changed")
        links[name] = chain(ET.fromstring(path.read_bytes()))
    controls = verify_controls(run, spec["scene"], result["control_files"])
    manifests = result.get("trials")
    require(isinstance(manifests, list) and len(manifests) == 2
            and [m["method"] for m in manifests] == list(METHODS), "Native cohort count/order changed")
    require({m["file"] for m in manifests} == {str(p.relative_to(run)) for p in (run/"trials").glob("*.json.gz")},
            "Unlisted, duplicated or missing native trial")
    outcomes, trials = {}, []
    for index, (cohort, receipt) in enumerate(zip(checked["cohorts"], manifests)):
        method = METHODS[index]
        require(receipt["file"] == f"trials/{method}-{index:05d}.json.gz", "Native trial identity changed")
        trial, source_receipt = read_trial(run, receipt["file"], receipt["sha256"])
        require(trial.get("schema") == "sircus-shot-benchmark-native-trial-v1"
                and trial.get("comparison_fingerprint") == plan["fingerprint"]
                and trial.get("mode") == method and trial.get("batch") == index
                and trial.get("optimizer_updates") == 0 and trial.get("training_request_order") is None
                and trial.get("learner_packet") is None, "Native request became training or changed identity")
        require(trial.get("cells") == checked["cells"] and trial.get("actions") == cohort["actions"]
                and trial.get("enabled") == cohort["native_enabled"]
                and trial.get("proposal_ik_seeds") == cohort["ik_seeds"], "Native actions/TRAIN membership/IK seeds changed")
        require(trial.get("source") == source
                and trial.get("split_manifest_sha256") == checked["input_bindings"]["original"]["split_manifest_sha256"]
                and all(row.get("split") == "train" for row in trial["rows"]), "Native trial original source/split changed")
        verify_home(trial)
        geometry = verify_initial_geometry(trial, spec["scene"], checked["cells"], links, cohort["ik_seeds"])
        native = verify_native_trial(run, trial, spec["scene"])
        settling = verify_settling_history(trial, spec["scene"], checked["cells"], dict(settling_protocol=setup_protocol()))
        counts = native_summary(trial["rows"], trial["rewards"])
        counts.update(requests=counts["attempts"], released=counts["attempts"]-counts["rejected"],
                      capture_rate=counts["captures"]/35, clean_rate=counts["clean"]/35)
        outcomes[method] = dict(counts=counts, metrics=verify_metrics(receipt["metrics"]))
        trials.append(dict(method=method, file=receipt["file"], **source_receipt,
                           geometry=geometry, native=native, settlement=settling))
    require(sum(m["counts"]["requests"] for m in outcomes.values()) == 70, "Requested-position denominator changed")
    value = dict(schema=SCHEMA, status="passed", comparison_fingerprint=plan["fingerprint"],
                 comparison_sha256=digest(comparison_plan_path), source_result_sha256=digest(run/"result.json"),
                 source_contract_sha256=digest(run/"contract.json"), source_binding_sha256=digest(run/"experiment-binding.json"),
                 checkpoint_sha256=expected_sha, audit_source_sha256=digest(__file__), source_sha256=actual_sources,
                 trials=trials, methods=outcomes, startup_controls=controls, native_requests=70,
                 rescored_releases=sum(t["native"]["rescored_releases"] for t in trials),
                 full_rate_steps=sum(t["native"]["native_steps"] for t in trials),
                 optimizer_updates=0, held_out_shot_files_read=0,
                 scope="Frozen TRAIN-informed bank versus unchanged parent on the same 35 known TRAIN positions; no held-out generalization, capture guarantee or physical validation.")
    value["fingerprint"] = canonical_hash(value)
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--assets", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    require(not args.output.exists(), "Refuse to replace an audit receipt")
    value = audit_run(args.run, args.plan, checkpoint_path=args.checkpoint, assets_path=args.assets)
    with args.output.open("x") as out:
        json.dump(value, out, indent=2, allow_nan=False)
        out.write("\n")
    print(json.dumps(dict(status=value["status"], native_requests=value["native_requests"], rescored_releases=value["rescored_releases"])))


if __name__ == "__main__":
    main()
