"""Read-only CPU diagnosis of a frozen prefix of the live shot benchmark.

No solver, simulator, policy inference, optimizer, network, native reset or robot
command runs here. Geometry/FK is recomputed from the exact saved request and
URDF. A numerical IK failure is unresolved, never a proof of impossibility.
"""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import gzip
import hashlib
import json
import math
from pathlib import Path
import re
import sys
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation

import continuous_feasibility
import expanded_pose_menu
import feasibility_contract
import prepare_assets
from continuous_feasibility import ACTION_NAMES, decode_action, joint_points
from expanded_pose_menu import endpoint_targets, outer_bounds
from prepare_assets import chain, fk, matrix

SCHEMA = "sircus-shot-benchmark-reach-audit-v1"
DEPENDENCY_MODULES = (continuous_feasibility, expanded_pose_menu, feasibility_contract, prepare_assets)
PARAMETER_EDGES = {
    "radius_m": [.28, .45, .60, .80, 1.0, 1.20],
    "height_m": [.14, .25, .35, .45, .60],
    "gap_m": [.08, .20, .32, .44, .56],
    "tilt_deg": [12., 22., 32., 42., 52.],
}
CATEGORIES = {
    "outer_reach": ("Outside conservative outer reach", "The requested flange lies beyond an arm's URDF origin-length bound. This is a necessary reach bound; being inside does not prove a pose reachable."),
    "dropper_ik": ("Dropper endpoint IK unresolved", "The finite multistart solver did not satisfy the position and full-orientation residual gate. This does not prove that no solution exists."),
    "board_ik": ("Board endpoint IK unresolved", "The finite multistart solver did not satisfy the position and board-normal residual gate. This does not prove that no solution exists."),
    "clearance_proxy": ("Clearance proxy rejection", "A conservative board/bucket/table or arm capsule proxy overlaps. Such proxies can reject safe poses; they are not exact mesh-collision certificates."),
    "transit": ("Path or transit rejection", "Endpoint qualification is separate from the sampled path proxy and actual driven transit."),
    "settling": ("Settling or correction rejection", "Actual driven readback did not satisfy the original endpoint/quiet release gates or a correction could not qualify."),
    "other": ("Other setup rejection", "The recorded primary rejection remains explicit; no missing category is silently treated as unreachable."),
}
SCOPE = [
    "Frozen-prefix TRAIN-only audit. No held-out shot files, rewards or predictions enter diagnosis or selection.",
    "Primary recorded rejection reasons are mutually exclusive; gates short-circuit, so later checks may be unevaluated.",
    "All active TRAIN requests, including rejects, remain in the original denominator. Inactive cells and evaluation requests are excluded.",
    "Requested targets and numerical best-IK chains are reconstructions in a single workcell's Arm10-base frame, not screenshots or achieved simulator poses.",
    "Outer spheres are necessary reach bounds, not feasible-workspace volumes. Inside the sphere does not certify joint-limited pose reachability.",
    "No new IK optimization or ball-flight search; saved candidates and residuals are independently recomputed with FK only.",
    "Arm14 endpoint IK constrains board center and normal, not the board's full in-plane twist/yaw; the requested rectangle is not a certified achieved rectangle pose.",
    "This diagnostic does not change the running benchmark, guards, sampling, material assumptions or trial budget.",
]


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(), parse_constant=lambda x: (_ for _ in ()).throw(ValueError("Nonfinite JSON " + x)))


def category(reason):
    if reason == "outer_link_length_bound": return "outer_reach"
    if reason == "endpoint_ik_unresolved_Arm10": return "dropper_ik"
    if reason == "endpoint_ik_unresolved_Arm14": return "board_ik"
    if "proxy" in reason and not any(x in reason for x in ("path", "correction", "home")): return "clearance_proxy"
    if any(x in reason for x in ("not_quiet", "not_settled", "settling", "correction")): return "settling"
    if any(x in reason for x in ("path", "transit", "home")): return "transit"
    return "other"


def stats(values):
    values = np.asarray(values, dtype=float)
    return None if not len(values) else dict(n=len(values), minimum=float(values.min()), median=float(np.median(values)), p95=float(np.percentile(values, 95)), maximum=float(values.max()))


def bases_for(scene):
    return {"Arm10": np.eye(4), "Arm14": matrix([scene["arms"]["base_spacing_m"], 0, 0], [0, 0, scene["arms"]["board_base_yaw_rad"]])}


def fit_details(target, fit, links, base, normal_only):
    """Recompute the saved finite-solver candidate without solving again."""
    q = np.asarray(fit["q"], dtype=float)
    require(q.shape == (6,) and np.isfinite(q).all(), "Invalid saved IK joints")
    lo, hi = np.array([l[2] for l in links]), np.array([l[3] for l in links])
    require(np.all(q >= lo - 1e-6) and np.all(q <= hi + 1e-6), "Saved IK candidate outside joint limits")
    current = fk(q, links)
    angular = current[:3, 2] - target[:3, 2] if normal_only else Rotation.from_matrix(target[:3, :3] @ current[:3, :3].T).as_rotvec()
    residual = np.r_[current[:3, 3] - target[:3, 3], .25 * angular]
    norm = float(np.linalg.norm(residual))
    position_mm = float(np.linalg.norm(residual[:3]) * 1000)
    angle = math.degrees(float(np.arccos(np.clip(current[:3, 2] @ target[:3, 2], -1, 1)))) if normal_only else math.degrees(float(np.linalg.norm(angular)))
    require(abs(norm - float(fit["residual"])) <= 1e-7, "Saved IK residual differs from independent FK")
    require(abs(position_mm - float(fit["position_error_mm"])) <= 1e-4, "Saved IK position error differs from independent FK")
    require(abs(angle - float(fit["orientation_error_deg"])) <= 2e-4, "Saved IK angle differs from independent FK")
    require(type(fit["solved"]) is bool and fit["solved"] == (norm < 1e-4), "Saved IK success differs from unchanged residual gate")
    attempts = fit.get("attempts", [])
    require(isinstance(attempts, list), "Invalid solver attempt log")
    active_limits = []
    for i, (value, low, high) in enumerate(zip(q, lo, hi)):
        if min(value - low, high - value) <= 1e-4:
            active_limits.append(dict(joint=i + 1, side="lower" if value - low <= high - value else "upper", margin_rad=float(min(value - low, high - value))))
    return dict(q=q.tolist(), ik_solved=fit["solved"], residual=norm,
        position_error_mm=position_mm, orientation_error_deg=angle,
        best_joint_points_world_m=joint_points(q, links, base).tolist(),
        best_flange_world_m=(base @ current)[:3, 3].tolist(),
        active_joint_limits=active_limits, starts=len(attempts), nfev_total=sum(int(x["nfev"]) for x in attempts),
        joint_limit_threshold_rad=1e-4,
        candidate_scope="Saved numerical endpoint candidate; not an achieved simulator pose",
        orientation_scope="Board normal only; in-plane twist unconstrained" if normal_only else "Full flange orientation")


def geometry_record(scene, cell, action, links, qualification):
    values = [action[k] for k in ACTION_NAMES]
    expected = decode_action(cell, values, action.get("id", "audit"))
    for key in ("board_x_m", "board_y_m", "tilt_deg", "azimuth_deg"):
        require(abs(expected[key] - action[key]) < 1e-8, "Saved action decoding differs")
    targets = endpoint_targets(scene, cell, action)
    bounds = outer_bounds(targets, links)
    saved = qualification.get("outer_bounds")
    if saved is not None:
        require(saved["eligible"] == bounds["eligible"], "Saved outer-bound eligibility differs")
        for name in ("Arm10", "Arm14"):
            for key in ("distance_m", "outer_radius_m"):
                require(abs(saved["arms"][name][key] - bounds["arms"][name][key]) < 1e-8, "Saved outer-bound arithmetic differs")
            require(saved["arms"][name]["within"] == bounds["arms"][name]["within"], "Saved outer-bound flag differs")
    if qualification.get("reason") == "outer_link_length_bound":
        require(not bounds["eligible"], "Outer-bound rejection has no violated necessary bound")
    bases = bases_for(scene)
    arms = {}
    for name in ("Arm10", "Arm14"):
        b = bounds["arms"][name]
        pivot = bases[name] @ np.r_[links[name][0][0][:3, 3], 1]
        arm = dict(pivot_world_m=pivot[:3].tolist(), outer_radius_m=b["outer_radius_m"],
            target_world_m=(bases[name] @ targets[name])[:3, 3].tolist(),
            target_normal_world=(bases[name] @ targets[name])[:3, 2].tolist(),
            distance_m=b["distance_m"], reach_margin_m=b["outer_radius_m"] - b["distance_m"], within_outer_bound=b["within"])
        fit = qualification.get("ik", {}).get(name)
        if fit is not None:
            arm.update(fit_details(targets[name], fit, links[name], bases[name], name == "Arm14"))
        else:
            arm["candidate_scope"] = "Not evaluated after an earlier gate rejected this request"
        arms[name] = arm
    half = np.asarray(scene["board"]["size_m"]) / 2
    corners = [[x, y, half[2], 1] for x, y in ((-half[0], -half[1]), (half[0], -half[1]), (half[0], half[1]), (-half[0], half[1]))]
    return dict(bucket_center_m=list(cell["bucket_xy_m"]) + [0.],
        board_center_m=targets["board_world"][:3, 3].tolist(),
        board_corners_m=[(targets["board_world"] @ c)[:3].tolist() for c in corners],
        board_normal_m=targets["board_normal_world"].tolist(),
        release_center_m=targets["release_center_m"].tolist(),
        arm_bases_m={n: b[:3, 3].tolist() for n, b in bases.items()}), arms


def select_cases(records, limit=14):
    """Stable median and extreme diagnostics, never reward-selected successes."""
    chosen, ids = [], set()
    def add(pool, metric, label, extreme=False):
        if not pool or len(chosen) >= limit: return
        ranked = sorted(pool, key=lambda r: (metric(r), r["id"]))
        row = ranked[-1] if extreme else ranked[(len(ranked) - 1) // 2]
        if row["id"] not in ids:
            ids.add(row["id"]);chosen.append(dict(row, label=label))
    outer = [r for r in records if r["reason"] == "outer_link_length_bound"]
    for names in (("Arm10",), ("Arm14",), ("Arm10", "Arm14")):
        pool = [r for r in outer if tuple(n for n in ("Arm10", "Arm14") if not r["arms"][n]["within_outer_bound"]) == names]
        metric = lambda r: max(-r["arms"][n]["reach_margin_m"] for n in names)
        label = " + ".join(names) + " outside outer reach"
        add(pool, metric, label + " · median overshoot")
        add(pool, metric, label + " · largest overshoot", True)
    for name in ("Arm10", "Arm14"):
        pool = [r for r in records if r["reason"] == "endpoint_ik_unresolved_" + name]
        metric = lambda r: r["arms"][name]["residual"]
        add(pool, metric, name + " IK unresolved · median residual")
        add(pool, metric, name + " IK unresolved · largest residual", True)
    for key in ("clearance_proxy", "transit", "settling", "other"):
        pool = [r for r in records if r["classification"] == key]
        add(pool, lambda r: r["batch"], CATEGORIES[key][0] + " · median batch")
    return chosen


def dependency_receipts(binding):
    result = {}
    for module in DEPENDENCY_MODULES:
        path = Path(module.__file__)
        actual = sha(path)
        require(binding["source_sha256"].get(path.name) == actual, "Audit geometry dependency differs from native source: " + path.name)
        result[path.name] = actual
    return result


def probe_selection(candidates):
    """Preselect a small counterfactual numerical diagnostic before re-solving."""
    result = []
    for arm in ("Arm10", "Arm14"):
        pool = sorted([c for c in candidates if c["arm"] == arm and not c["saved_fit"]["solved"]], key=lambda c:(c["saved_fit"]["residual"],c["case_id"]))
        chosen = []
        for group in ([c for c in pool if c["saved_fit"]["residual"] < 1e-3], [c for c in pool if c["saved_fit"]["residual"] >= 1e-3]):
            for i in sorted(set([0,len(group)-1])) if group else []:
                chosen.append(group[i])
        for c in pool:
            if len(chosen) >= 4: break
            if c not in chosen: chosen.append(c)
        result.extend(chosen[:4])
        controls = sorted([c for c in candidates if c["arm"] == arm and c["saved_fit"]["solved"]],key=lambda c:(c["saved_fit"]["residual"],c["case_id"]))
        if controls: result.append(controls[(len(controls)-1)//2])
    return result


def audit(progress_snapshot, run, assets, max_cases=14):
    progress_snapshot, run, assets = map(Path, (progress_snapshot, run, assets))
    require(not progress_snapshot.resolve().is_relative_to(run.resolve().parents[1]), "Capture progress outside the live experiment before auditing")
    progress = read(progress_snapshot)
    require(progress.get("schema") == "sircus-shot-benchmark-progress-v1", "Unexpected benchmark progress schema")
    require(progress.get("smoke") is False, "Production TRAIN audit required; smoke is separate")
    binding = read(run / "experiment-binding.json")
    require(progress["source"] == binding["source"], "Snapshot source differs from native binding")
    deps = dependency_receipts(binding)
    original_hashes = {"contract.json":sha(run/"contract.json"),"experiment-binding.json":sha(run/"experiment-binding.json")}
    asset_hashes = {f"assets/{f}":sha(assets/f) for f in ("dropper.urdf","board_arm.urdf")}
    contract = read(run / "contract.json")
    scene = contract["scene"]
    links = {name: chain(ET.parse(assets / file).getroot()) for name, file in (("Arm10", "dropper.urdf"), ("Arm14", "board_arm.urdf"))}
    records, reasons, cells, phases = [], Counter(), {}, {}
    probe_candidates, inventory = [], []
    counters = dict(attempts=0, rejected=0, released=0, captures=0, clean=0)
    gates, ik_metrics, outer_metrics = Counter(), defaultdict(lambda: defaultdict(list)), defaultdict(list)
    source_files, request_indices = {}, []
    for filename, expected_sha in sorted(progress["source_trial_sha256"].items()):
        if not re.fullmatch(r"trials/train-\d{5}\.json\.gz", filename): continue
        path = run / filename
        require(path.resolve().is_relative_to(run.resolve()) and not path.is_symlink(), "Unsafe native trial path")
        require(sha(path) == expected_sha, "Native trial bytes changed: " + filename)
        source_files[filename] = expected_sha
        with gzip.open(path, "rt") as stream: trial = json.load(stream)
        require(trial["schema"] == "sircus-shot-benchmark-native-trial-v1" and trial["mode"] == "train", "Expected original native TRAIN trial")
        require(trial["source"] == progress["source"] and trial["split_manifest_sha256"] == progress["split_manifest_sha256"], "Native trial belongs to another source/split")
        enabled, rows, contexts, actions = (trial[k] for k in ("enabled", "rows", "cells", "actions"))
        require(len(enabled) == len(rows) == len(contexts) == len(actions) == 35 and all(type(x) is bool for x in enabled), "Invalid native enabled mask")
        order = trial["training_request_order"]
        require(isinstance(order, list) and len(order) == sum(enabled) and len(set(order)) == len(order) and set(order) == {i for i, e in enumerate(enabled) if e}, "TRAIN paid request order differs from active cells")
        meta = trial.get("proposal_metadata")
        if meta is not None: require(len(meta) == len(order), "Missing sweep proposal metadata")
        for ordinal, i in enumerate(order):
            row, cell, action = rows[i], contexts[i], actions[i]
            require(cell["split"] == row["split"] == "train" and cell["id"] == row["cell_id"], "Non-TRAIN or changed context in paid requests")
            reason = row["invalid"]
            require(reason is None or (isinstance(reason, str) and reason), "Invalid recorded primary rejection")
            captured = row["route_success"]
            require(type(captured) is bool and not (reason and captured), "Rejected request cannot capture")
            phase = meta[ordinal]["phase"] if meta is not None else "ppo_sample"
            index = int(meta[ordinal]["global_request_index"]) if meta is not None else counters["attempts"]
            if meta is not None:
                require(meta[ordinal]["cell_id"] == cell["id"] and meta[ordinal]["bucket_xy_m"] == cell["bucket_xy_m"], "Proposal context/order differs")
            request_indices.append(index)
            counters["attempts"] += 1;counters["rejected"] += bool(reason);counters["released"] += not bool(reason)
            counters["captures"] += captured;counters["clean"] += row.get("route_quality") == 2
            require(row.get("route_quality") != 2 or captured, "Clean outcome without qualified capture")
            inventory.append((action,reason))
            if reason: reasons[reason] += 1
            for collection, key in ((cells, cell["id"]), (phases, phase)):
                if key not in collection: collection[key] = dict(id=key, attempts=0, rejected=0, captures=0, reason_counts=Counter())
                collection[key]["attempts"] += 1;collection[key]["rejected"] += bool(reason);collection[key]["captures"] += captured
                if reason: collection[key]["reason_counts"][reason] += 1
            cells[cell["id"]]["bucket_xy_m"] = cell["bucket_xy_m"]
            qual = row["qualification"]
            geometry, arms = geometry_record(scene, cell, action, links, qual)
            targets = endpoint_targets(scene,cell,action)
            if qual.get("geometry") is not None:
                gates["geometry_evaluated"] += 1;gates["geometry_accepted"] += bool(qual["geometry"]["accepted"])
            if qual.get("outer_bounds") is not None:
                gates["outer_bound_evaluated"] += 1;gates["outer_bound_accepted"] += bool(qual["outer_bounds"]["eligible"])
            for name in ("Arm10", "Arm14"):
                fit = qual.get("ik", {}).get(name)
                if fit:
                    gates[name + "_ik_evaluated"] += 1;gates[name + "_ik_solved"] += fit["solved"]
                    if not fit["solved"]:
                        for k in ("position_error_mm", "orientation_error_deg", "residual", "nfev_total"):
                            ik_metrics[name][k].append(arms[name][k])
                        ik_metrics[name]["near_joint_limit"].append(bool(arms[name]["active_joint_limits"]))
                    if fit["solved"] or reason == "endpoint_ik_unresolved_" + name:
                        probe_candidates.append(dict(case_id=f"batch{trial['batch']:05d}-{cell['id']}-{name}",split="train",arm=name,
                            target_arm_base=targets[name].tolist(),saved_fit=fit,
                            home_q=trial["home"]["measured_joint_rad"][name][i],
                            provenance=dict(trial_file=filename,trial_sha256=expected_sha,batch=trial["batch"],cell_id=cell["id"],cell_index=i,
                                paid_request_index=index,phase=phase,reason=reason,action=action,bucket_xy_m=cell["bucket_xy_m"])))
                if reason == "outer_link_length_bound" and not arms[name]["within_outer_bound"]:
                    outer_metrics[name].append(-arms[name]["reach_margin_m"])
            if qual.get("ik", {}).get("Arm10", {}).get("solved") is False:
                gates["Arm14_not_evaluated_after_Arm10_failure"] += int("Arm14" not in qual.get("ik", {}))
            if reason:
                compact_action = {k: action[k] for k in (*ACTION_NAMES, "board_x_m", "board_y_m", "tilt_deg", "azimuth_deg")}
                records.append(dict(id=f"batch{trial['batch']:05d}-{cell['id']}", reason=reason,
                    classification=category(reason), trial_file=filename, trial_sha256=expected_sha,
                    batch=trial["batch"], paid_request_index=index, phase=phase, cell_id=cell["id"],
                    bucket_xy_m=cell["bucket_xy_m"], action=compact_action, geometry=geometry, arms=arms,
                    native_release=False,
                    executed_pose_scope="Requested target and numerical IK candidate only; rejected before release. Some later-stage failures include attempted driven setup.",
                    measured=dict(board_center_m=row.get("settled_board_center_m"),board_normal=row.get("settled_board_normal"),dropper_flange_m=row.get("settled",{}).get("dropper_flange_m"))))
    require(counters["attempts"] > 0 and request_indices == sorted(request_indices) and request_indices == list(range(counters["attempts"])), "Missing/duplicated/out-of-order paid TRAIN request indices")
    for key in ("attempts", "captures", "clean", "rejected"):
        require(counters[key] == progress["train"][key], "Snapshot TRAIN counts differ from immutable trials: " + key)
    require(sum(reasons.values()) == counters["rejected"], "Rejection categories changed denominator")
    require(all(progress["train"].get("failures",{}).get(reason)==n for reason,n in reasons.items()), "Primary reason counts differ from frozen aggregate")
    require(sum(cells[k]["attempts"] for k in cells) == counters["attempts"], "Per-cell denominator mismatch")
    categories = []
    for key, (label, interpretation) in CATEGORIES.items():
        count = sum(n for reason, n in reasons.items() if category(reason) == key)
        if count: categories.append(dict(id=key, label=label, count=count, fraction_of_attempts=count/counters["attempts"], fraction_of_rejected=count/counters["rejected"] if counters["rejected"] else 0, interpretation=interpretation))
    bins = {}
    # Use every verified paid request, not only the rejected population.
    for key, edges in PARAMETER_EDGES.items():
        bins[key] = []
        for j, (low, high) in enumerate(zip(edges, edges[1:])):
            items = [(a, reason) for a, reason in inventory if low-1e-10 <= a[key] and (a[key] < high if j<len(edges)-2 else a[key] <= high+1e-10)]
            rc = Counter(reason for _, reason in items if reason)
            bins[key].append(dict(lo=low, hi=high, attempts=len(items), rejected=sum(rc.values()), reason_counts=dict(rc)))
        require(sum(b["attempts"] for b in bins[key]) == counters["attempts"], "Parameter bins changed denominator: " + key)
    require(original_hashes == {n:sha(run/n) for n in original_hashes}, "Native scene/binding changed during read-only audit")
    require(asset_hashes == {f"assets/{f}":sha(assets/f) for f in ("dropper.urdf","board_arm.urdf")}, "URDF assets changed during read-only audit")
    require(deps == dependency_receipts(binding), "Geometry dependency bytes changed during read-only audit")
    return dict(schema=SCHEMA,
        snapshot=dict(progress_sha256=sha(progress_snapshot),source=progress["source"],method=progress["method"],seed=progress["seed"],train_attempts=counters["attempts"],trial_count=len(source_files),created_utc=datetime.now(timezone.utc).isoformat(),native_elapsed_s=progress.get("elapsed_s")),
        totals=dict(counters, reason_counts=dict(reasons), rejection_rate=counters["rejected"]/counters["attempts"]),
        scene=scene,categories=categories,cells=[dict(v,reason_counts=dict(v["reason_counts"])) for _,v in sorted(cells.items())],
        phases=[dict(v,reason_counts=dict(v["reason_counts"])) for _,v in sorted(phases.items())],
        parameter_bins=bins,gate_stats=dict(gates),
        outer_reach_overshoot_m={n:stats(v) for n,v in outer_metrics.items()},
        unresolved_ik_summary={n:{k:stats(v) for k,v in metrics.items()} for n,metrics in ik_metrics.items()},
        cases=select_cases(records,max_cases),probe_candidates=probe_selection(probe_candidates),source_trial_sha256=source_files,
        dependency_sha256=dict(deps,**asset_hashes,**original_hashes),
        audit_source_sha256=sha(__file__),scope=SCOPE,
        selection="Deterministic median and extreme saved reach/IK failures plus median-stage proxy/transit/settling cases; no outcome-selected success substitution")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--progress-snapshot", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-cases", type=int, default=14)
    args = parser.parse_args()
    require(1 <= args.max_cases <= 24, "Bounded case count required")
    require(not args.output.resolve().is_relative_to(args.run.resolve().parents[1]), "Audit output must stay outside the live experiment")
    require(not args.output.exists(), "Audit output already exists; preserve snapshots")
    result = audit(args.progress_snapshot,args.run,args.assets,args.max_cases)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open("x") as stream: json.dump(result,stream,indent=2,allow_nan=False);stream.write("\n")
    print(json.dumps(dict(schema=SCHEMA,attempts=result["totals"]["attempts"],rejected=result["totals"]["rejected"],cases=len(result["cases"]),sha256=sha(args.output))))


if __name__ == "__main__": main()
