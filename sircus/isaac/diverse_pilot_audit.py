"""Independent CPU audit for geometry-region hybrid PPO and measured setup.

Historical study validators remain intact. New protocol identities explicitly
admit the new bounded setup controller while preserving scoring/release limits.
"""

import argparse
import ast
import copy
import hashlib
import json
import math
import tarfile
from pathlib import Path

import numpy as np

import continuous_pilot_audit as chain
from continuous_pilot_audit import (require, read, digest, canonical_hash, signed, finite,
                                    evidence_file, home_comparison, LOGP_TOLERANCE, HOME_PROTOCOL)

SCHEMA = "sircus-diverse-settled-checkpoint-audit-v1"
STUDY_SCHEMA = "sircus-native-diverse-settled-pilot-v1"
DECODER_V1 = "geometry-only region choice + local affine residual-v1"
DECODER = "geometry-only region choice + affine residual-v2"
SCORING_FILES = ("rl_contract.py", "board_route.py", "entry_quality.py", "farm_contract.py", "settling_contract.py")


def validate_support_evidence(support, feasibility, spec):
    """Reconstruct every geometry-only retained region, without outcome labels.

    This repeats ranking, fixed widths, coverage arithmetic and candidate
    membership. It does not repeat the original IK/path simulation/proxy work.
    """
    from continuous_feasibility import ACTION_NAMES, LOW, HIGH, decode_action

    signed(support, "Diverse support")
    signed(feasibility, "Geometry map")
    geometry = signed(spec, "Source geometry")
    half = np.array([math.radians(7.5), .040, .040, .080, math.radians(8)])
    v2 = support.get("schema") == "sircus-diverse-shot-support-v2"
    require(support.get("schema") in ("sircus-diverse-shot-support-v1", "sircus-diverse-shot-support-v2")
        and support.get("policy_geometry_only") is True
        and support.get("source_fingerprint") == geometry
        and support.get("feasibility_fingerprint") == feasibility["fingerprint"]
        and support.get("asset_sha256") == feasibility.get("asset_sha256")
        and feasibility.get("source_fingerprint") == geometry
        and feasibility.get("ik_protocol") == "full_robust_v2", "Diverse map/source binding changed")
    require(support.get("fixed_half_width") == half.tolist()
        and support.get("anchor_path_samples") == 16
        and support.get("action_names") == list(ACTION_NAMES)
        and support.get("global_low") == LOW.tolist() and support.get("global_high") == HIGH.tolist(),
        "Diverse fixed-width geometry protocol changed")
    home = support.get("canonical_home_joint_rad", {})
    require(set(home) == {"Arm10", "Arm14"}, "Missing diverse canonical home")
    home = {name: finite(q, (6,), "support home").astype(float) for name, q in home.items()}
    by_cell = {c["id"]: [] for c in spec["cells"]}
    for record in feasibility["records"]:
        require(record["cell_id"] in by_cell, "Unexpected geometry-map cell")
        if record["accepted"]:
            by_cell[record["cell_id"]].append(record)

    def coverage(regions):
        if not regions:
            return None
        return dict(low=np.min([r["low"] for r in regions], axis=0).tolist(),
            high=np.max([r["high"] for r in regions], axis=0).tolist(),
            anchor_low=np.min([[r["anchor_action"][key] for key in ACTION_NAMES] for r in regions], axis=0).tolist(),
            anchor_high=np.max([[r["anchor_action"][key] for key in ACTION_NAMES] for r in regions], axis=0).tolist(),
            scope="Coordinate-wise envelope of separate boxes, not a filled feasible box")

    require(len(support["cells"]) == 35, "Diverse support cell count changed")
    all_regions, counts = [], []
    for index, (row, cell) in enumerate(zip(support["cells"], spec["cells"])):
        require(row.get("cell_id") == cell["id"] and row.get("cell_index") == index
            and row.get("bucket_xy_m") == cell["bucket_xy_m"], "Diverse support context changed")
        candidates, checks = by_cell[cell["id"]], row.get("anchor_checks", [])
        require(row.get("map_candidates") == len(candidates) == len(checks), "Geometry candidate omitted from support checks")
        distance = np.linalg.norm(np.array([.67, -.10])-cell["bucket_xy_m"])
        prior = np.array([0., np.clip(distance, .29, 1.19), .35, .24, math.radians(29)])
        retained = []
        for original, check in zip(candidates, checks):
            action = original["action"]
            values = np.array([action[key] for key in ACTION_NAMES])
            expected = decode_action(cell, values, action["id"])
            travel = np.mean([np.square(np.asarray(original["ik"][name]["q"])-home[name]).sum()
                              for name in ("Arm10", "Arm14")])
            score = float(np.square((values-prior)/(HIGH-LOW)).sum()+.01*travel)
            require(check.get("proposal") == original["proposal"]
                and check.get("action_id") == action["id"] and abs(check.get("score", math.inf)-score) < 2e-7
                and type(check.get("path", {}).get("accepted")) is bool, "Diverse geometry-only anchor ranking changed")
            if check["path"]["accepted"]:
                retained.append((score, int(original["proposal"]), original, expected, check["path"]))
        retained.sort(key=lambda value: value[:2])
        local_regions = row["regions"][:-1] if v2 and row["regions"] else row["regions"]
        global_count = int(bool(retained)) if v2 else 0
        require(len(local_regions) == len(retained) and row.get("region_count") == len(retained)+global_count,
                "Path-qualified geometry anchor was pruned or duplicated")
        require(row.get("status") == ("supported" if retained else "unresolved"), "Empty diverse support relabeled")
        for region_id, (region, (score, proposal, original, action, path)) in enumerate(zip(local_regions, retained)):
            center = np.array([action[key] for key in ACTION_NAMES])
            seeds = {name: dict(q=original["ik"][name]["q"]) for name in ("Arm10", "Arm14")}
            require(region.get("id") == region_id and region.get("proposal") == proposal
                and region.get("anchor_action") == action and region.get("anchor_ik") == seeds
                and region.get("anchor_path") == path and abs(region.get("anchor_score", math.inf)-score) < 2e-7
                and region.get("low") == np.maximum(LOW, center-half).tolist()
                and region.get("high") == np.minimum(HIGH, center+half).tolist(), "Diverse retained region/action/width changed")
            if v2:
                require(region.get("kind") == "local_geometry" and region.get("interior_qualified") is False,
                        "Local geometry region mislabeled")
        if v2:
            require(row.get("local_region_count") == len(retained) and row.get("global_region_count") == global_count
                and row.get("local_coverage") == coverage(local_regions), "Local/global geometry count or envelope changed")
            if retained:
                global_ = row["regions"][-1]
                require(global_.get("id") == len(retained) and global_.get("kind") == "global_exploration"
                    and global_.get("proposal") == -1 and global_.get("interior_qualified") is False
                    and global_.get("low") == LOW.tolist() and global_.get("high") == HIGH.tolist()
                    and global_.get("anchor_action") == local_regions[0]["anchor_action"]
                    and global_.get("anchor_ik") == local_regions[0]["anchor_ik"]
                    and global_.get("qualification_scope") == "Full original action domain is unqualified. First geometry anchor supplies only an IK warm start; every requested action still needs fresh IK/path/settlement gates.",
                    "Missing or falsely qualified full-domain escape region")
        require(row.get("coverage") == coverage(row["regions"]), "Diverse per-cell coverage envelope changed")
        all_regions.extend(row["regions"])
        counts.append(dict(cell_id=cell["id"], map_candidates=len(candidates), regions=len(row["regions"]),
                           local_regions=len(retained), global_regions=global_count, coverage=row["coverage"]))
    require(support.get("region_count") == len(all_regions)
        and support.get("supported_cells") == sum(c["regions"] > 0 for c in counts)
        and support.get("max_regions_per_cell") == max(c["regions"] for c in counts)
        and support.get("coverage") == coverage(all_regions), "Diverse aggregate geometry accounting changed")
    if v2:
        require(support.get("local_region_count") == sum(c["local_regions"] for c in counts)
            and support.get("global_region_count") == sum(c["global_regions"] for c in counts)
            and support.get("local_coverage") == coverage([r for r in all_regions if r["kind"] == "local_geometry"]),
            "Diverse local/global aggregate changed")
    return dict(supported_cells=sum(c["regions"] > 0 for c in counts), unresolved_cells=sum(c["regions"] == 0 for c in counts),
        retained_regions=len(all_regions), local_regions=sum(c["local_regions"] for c in counts),
        global_regions=sum(c["global_regions"] for c in counts), original_domain_escape_available=v2,
        all_path_qualified_anchors_retained=True, per_cell=counts,
        scope="Independent candidate membership, geometry-prior ranking and fixed-box arithmetic; original map IK/path proxy results not replayed, region interiors not certified")


def independent_physical_actions(normalized, indices, support, cells):
    from continuous_feasibility import LOW, HIGH
    normalized = np.asarray(normalized, dtype=float)
    indices = np.asarray(indices)
    require(normalized.shape == (len(cells), 5) and np.isfinite(normalized).all()
        and (np.abs(normalized) <= 1).all() and indices.shape == (len(cells),)
        and np.isfinite(indices).all() and (indices == np.floor(indices)).all(), "Invalid independent hybrid action")
    by_id = {r["cell_id"]: r for r in support["cells"]}
    bounds = []
    for cell, index in zip(cells, indices.astype(int)):
        require(cell["id"] in by_id and by_id[cell["id"]]["bucket_xy_m"] == cell["bucket_xy_m"], "Hybrid decoder context changed")
        row = by_id[cell["id"]]
        if row["regions"]:
            require(0 <= index < len(row["regions"]), "Hybrid category outside support")
            region = row["regions"][index]
            bounds.append((region["low"], region["high"]))
        else:
            require(index == 0, "Unresolved accounting category changed")
            bounds.append((LOW, HIGH))
    low, high = (np.array([bounds[i][j] for i in range(len(cells))]) for j in (0, 1))
    return low+(normalized+1)*(high-low)/2


def validate_diverse_study(study, spec, support, support_sha256, feasibility_sha256):
    from continuous_policy import ARCHITECTURE as OLD_ARCH, POLICY_VERSION as OLD_VERSION
    from diverse_policy import ARCHITECTURE, POLICY_VERSION, ARCHITECTURE_V1, POLICY_VERSION_V1
    from diverse_support import validate_support
    from settled_transit import protocol
    study_hash, geometry_hash, domain_hash = signed(study, "Diverse study"), signed(spec, "Source geometry"), signed(study["domain"], "Diverse domain")
    validate_support(support, spec["cells"])
    v2 = support["schema"] == "sircus-diverse-shot-support-v2"
    require(study.get("schema") == STUDY_SCHEMA and study.get("architecture") == (ARCHITECTURE if v2 else ARCHITECTURE_V1)
        and study.get("policy_version") == (POLICY_VERSION if v2 else POLICY_VERSION_V1) and study.get("support_fingerprint") == support["fingerprint"]
        and study.get("support_sha256") == support_sha256 and study.get("feasibility_sha256") == feasibility_sha256
        and study["domain"].get("decoder") == (DECODER if v2 else DECODER_V1) and study["domain"].get("support_fingerprint") == support["fingerprint"]
        and support.get("source_fingerprint") == geometry_hash, "Diverse study/policy/support identity changed")
    require(study.get("settling_protocol") == protocol() and study.get("setup") == protocol()["scope"]
        and study.get("ik_protocol") == "full_robust_v2-equivalent-fk-v1"
        and study.get("measurement_window_s") == .25, "Diverse driven-settling controller protocol changed")
    shared = copy.deepcopy(study)
    shared.update(schema="sircus-native-continuous-pilot-v1", architecture=OLD_ARCH,
                  policy_version=OLD_VERSION, ik_protocol="full_robust_v2")
    shared.pop("fingerprint")
    shared["fingerprint"] = canonical_hash(shared)
    chain.validate_study(shared, spec)
    return study_hash, geometry_hash, domain_hash


class _Backend:
    def __init__(self, run):
        from diverse_policy import DiversePolicy, load_checkpoint
        self.support_path, self.map_path = evidence_file(run, "support.json"), evidence_file(run, "feasibility.json")
        self.support, self.map = read(self.support_path), read(self.map_path)
        self.spec = read(evidence_file(run, "contract.json"))
        self.support_evidence = validate_support_evidence(self.support, self.map, self.spec)
        self.policy_factory = lambda: DiversePolicy(self.support)
        self._load_checkpoint = load_checkpoint

    def validate_study(self, study, spec):
        require(spec == self.spec, "Diverse audit geometry source changed")
        return validate_diverse_study(study, spec, self.support, digest(self.support_path), digest(self.map_path))

    def load_checkpoint(self, path, **kwargs):
        return self._load_checkpoint(path, support=self.support, support_fingerprint=self.support["fingerprint"], **kwargs)

    def physical_actions(self, normalized, indices):
        from diverse_support import physical_actions
        expected = independent_physical_actions(normalized, indices, self.support, self.spec["cells"])
        actual = physical_actions(normalized, self.support, self.spec["cells"], indices, allow_unresolved=True)
        require(np.array_equal(actual, expected), "Diverse producer physical decoder changed")
        return expected


def backend_for_run(run):
    return _Backend(run)


def verify_joint_components(policy, observations, latent, data):
    """Recompute constituent densities without calling the producer logp helper."""
    import torch
    with torch.no_grad():
        features, mask = policy._features(observations)
        logits = policy.region_actor(features).squeeze(-1).masked_fill(~mask, -torch.inf)
        indices = latent[:, 0].to(torch.int64)
        batch = torch.arange(len(observations))
        require(((latent[:, 0] == indices) & (indices >= 0) & (indices < mask.shape[1])).all().item()
            and mask[batch, indices].all().item(), "Hybrid sampled unavailable geometry category")
        region = torch.log_softmax(logits, -1)[batch, indices]
        mean = policy.residual_actor(features)[batch, indices]
        logstd = policy.logstd.clamp(-3, 1)
        z = latent[:, 1:]
        jacobian = 2*(math.log(2)-z-torch.nn.functional.softplus(-2*z))
        residual = (-.5*((z-mean)/logstd.exp()).square()-logstd-.5*math.log(2*math.pi)-jacobian).sum(-1)
        expected = {"region_logp": region.numpy(), "residual_logp": residual.numpy(), "logp": (region+residual).numpy()}
    for name, value in expected.items():
        stored = finite(data.get(name), (35,), "joint density "+name)
        require(np.allclose(stored, value, atol=LOGP_TOLERANCE, rtol=LOGP_TOLERANCE), "Hybrid constituent likelihood changed: "+name)


def independent_feedback_ready(metric):
    """Quiet readbacks alone never establish readiness to correct steady bias."""
    lag = max(abs(p-t) for state in metric["arms"].values()
              for p, t in zip(state["position_rad"], state["target_rad"]))
    return (metric["joint_speed_rad_s"] <= .005 and all_articulation_speed(metric) <= .005 and metric["position_drift_mm"] <= .1
        and metric["normal_drift_deg"] <= .05 and metric["dropper_drift_mm"] <= .1 and lag <= .02)


def all_articulation_speed(metric):
    """Legacy numeric gate covers six rotary joints AND dropper fingers.

    The historical scalar name is joint_speed_rad_s. Prismatic finger channels
    retain their native m/s values under that same legacy numeric threshold.
    """
    names, velocities = metric.get("articulation_joint_names", {}), metric.get("all_articulation_joint_velocity", {})
    require(set(names) == set(velocities) == {"Arm10", "Arm14"}, "Missing all-articulation velocity/name readbacks")
    robot_names = ["joint"+str(i) for i in range(1, 7)]
    require(isinstance(names["Arm10"], list) and len(names["Arm10"]) == 8 and names["Arm10"][:6] == robot_names
        and set(names["Arm10"][6:]) == {"finger_left_joint", "finger_right_joint"}
        and names["Arm14"] == robot_names, "Articulation joint inventory/order changed")
    maximum = 0.
    for arm, count in (("Arm10", 8), ("Arm14", 6)):
        values = finite(velocities[arm], (count,), "full articulation velocity")
        robot = finite(metric.get("arms", {}).get(arm, {}).get("velocity_rad_s"), (6,), "six-joint FK velocity")
        require(np.array_equal(values[:6], robot), "Robot FK velocities differ from full articulation readback")
        maximum = max(maximum, float(np.abs(values).max()))
    return maximum


def verify_settling_history(trial, scene, cells, study):
    """Check measured windows and quiet-before-bias feedback, never relax gates."""
    from scipy.spatial.transform import Rotation
    from expanded_pose_menu import endpoint_targets
    from prepare_assets import matrix
    from settled_transit import protocol
    from settling_contract import accepted

    expected = protocol()
    require(trial.get("setup_controller") == study.get("settling_protocol") == expected,
            "Trial setup controller binding changed")
    records = trial.get("settling_history")
    transit = trial.get("transit_history", [])
    corrections = trial.get("correction_history", [])
    require(isinstance(records, list) and len(records) == 35 and all(isinstance(r, list) and r for r in records),
            "Missing measured settling windows")
    global_clock = [(m.get("iteration"), m.get("measurement_start_step"), m.get("physics_step")) for m in records[0]]
    require(len(global_clock) <= 256 and all([(m.get("iteration"), m.get("measurement_start_step"), m.get("physics_step"))
            for m in history] == global_clock for history in records), "Global settling measurement clock changed or exceeded finite bound")
    require(len(transit) >= 3 and transit[0].get("phase") == "start"
        and transit[-1].get("phase") == "restored_original_hold"
        and all(type(t.get("step")) is int for t in transit)
        and all(a["step"] < b["step"] for a, b in zip(transit, transit[1:])),
        "Driven transit/restoration evidence changed")
    for point in transit:
        require(set(point.get("state", {})) == {"Arm10", "Arm14"}, "Incomplete driven transit state")
        for state in point["state"].values():
            require(set(state) == {"position_rad", "velocity_rad_s", "target_rad"}, "Transit state columns changed")
            for key in state:
                finite(state[key], (35, 6), "transit "+key)
    original, restored = transit[0].get("original_gains", {}), transit[-1].get("restored_gains", {})
    require(set(original) == set(restored) == {"Arm10", "Arm14"}, "Missing original/restored gain readback")
    for arm in original:
        require(set(original[arm]) == set(restored[arm]) == {"effort", "stiffness", "damping"}, "Gain readback columns changed")
        for key in original[arm]:
            a, b = finite(original[arm][key], (35, 6), "original gain"), finite(restored[arm][key], (35, 6), "restored gain")
            require(np.array_equal(a, b), "Original holding gains/caps were not restored")
    require(transit[-1]["step"]-transit[-2]["step"] >= 240, "Holding gain restore ramp shortened")
    base = matrix([scene["arms"]["base_spacing_m"], 0, 0], [0, 0, np.pi])
    mount = matrix([0, 0, scene["board"]["tool_offset_m"]], [0, 0, scene["board"]["mount_yaw_rad"]])
    by_id = {cell["id"]: i for i, cell in enumerate(cells)}
    windows = 0
    for i, (history, cell, action, row) in enumerate(zip(records, cells, trial["actions"], trial["rows"])):
        target = endpoint_targets(scene, cell, action)
        board = base@target["Arm14"]@mount
        drop = target["Arm10"][:3, 3]
        require(history[0]["physics_step"] == transit[-1]["step"]+60,
                "First measurement does not follow restored holding drives")
        for j, metric in enumerate(history):
            require(type(metric.get("physics_step")) is int and 0 <= metric.get("iteration", -1) <= 3,
                    "Invalid settling window step/iteration")
            require(metric.get("measurement_start_step") == metric["physics_step"]-60,
                    "Settling measurement duration changed")
            if j:
                require(metric["physics_step"] == history[j-1]["physics_step"]+60
                    and metric["iteration"] in (history[j-1]["iteration"], history[j-1]["iteration"]+1),
                    "Measured stability window cadence changed")
            for key in ("board_center_m", "board_normal", "dropper_flange_m"):
                finite(metric.get(key), (3,), key)
            center = np.asarray(metric["board_center_m"], dtype=float)
            normal = np.asarray(metric["board_normal"], dtype=float)
            flange = np.asarray(metric["dropper_flange_m"], dtype=float)
            require(abs(np.linalg.norm(normal)-1) < 1e-4, "Settling normal not normalized")
            pose = dict(position_error_mm=float(np.linalg.norm(center-board[:3, 3])*1000),
                normal_error_deg=float(np.rad2deg(np.arccos(np.clip(normal@board[:3, 2], -1, 1)))),
                dropper_position_error_mm=float(np.linalg.norm(flange-drop)*1000))
            # Native arrays and targets are float32. Millimetre errors retain a
            # tight readback tolerance, separate from the unchanged 2mm gate.
            for key, value in pose.items():
                require(abs(metric.get(key, math.inf)-value) <= 1e-3, "Measured settling pose metric differs: "+key)
            require(set(metric.get("arms", {})) == {"Arm10", "Arm14"}, "Incomplete settling joint state")
            for state in metric["arms"].values():
                require(set(state) == {"position_rad", "velocity_rad_s", "target_rad"}, "Settling arm columns changed")
                for key in state:
                    values = finite(state[key], (6,), "settling "+key)
            max_speed = all_articulation_speed(metric)
            require(abs(metric.get("joint_speed_rad_s", math.inf)-max_speed) <= 1e-7, "Measured joint-speed gate differs from readback")
            for key in ("previous_board_center_m", "previous_board_normal", "previous_dropper_flange_m"):
                finite(metric.get(key), (3,), key)
            pos_drift = np.linalg.norm(center-metric["previous_board_center_m"])*1000
            drop_drift = np.linalg.norm(flange-metric["previous_dropper_flange_m"])*1000
            normal_drift = np.rad2deg(np.arccos(np.clip(normal@metric["previous_board_normal"], -1, 1)))
            require(abs(metric["position_drift_mm"]-pos_drift) <= 1e-5
                and abs(metric["dropper_drift_mm"]-drop_drift) <= 1e-5
                and abs(metric["normal_drift_deg"]-normal_drift) <= 1e-5,
                "Measured settling drift differs from window readbacks")
            if j:
                require(all(np.array_equal(metric["previous_"+key], history[j-1][key])
                            for key in ("board_center_m", "board_normal", "dropper_flange_m")),
                        "Settling window previous pose differs from preceding readback")
            windows += 1
        for iteration in {m["iteration"] for m in history}:
            count = sum(m["iteration"] == iteration for m in history)
            require(7 <= count <= (160 if iteration == 0 else 32),
                    "Measured stabilization hold outside declared initial/correction bound")
        require(row.get("settled") == history[-1], "Release row does not use final measured stability window")
        if not row["invalid"]:
            require(accepted(history[-1]) and all_articulation_speed(history[-1]) <= .005
                    and history[-1]["dropper_position_error_mm"] <= 2,
                    "Admitted release violates original measured gate")
        per_cell = [c for c in corrections if c.get("cell_id") == cell["id"]]
        require(len(per_cell) <= 3 and len({c["iteration"] for c in per_cell}) == len(per_cell), "Excessive or repeated bias corrections")
        for correction in per_cell:
            metric = correction.get("measured", {})
            require(metric in history and metric["iteration"] == correction.get("iteration")
                and metric == [m for m in history if m["iteration"] == correction["iteration"]][-1],
                "Correction not bound to final prior measured window")
            require(metric["joint_speed_rad_s"] <= .005 and all_articulation_speed(metric) <= .005
                and metric["position_drift_mm"] <= .1
                and metric["normal_drift_deg"] <= .05 and metric["dropper_drift_mm"] <= .1,
                "Correction attempted before measured motion was quiet")
            lag = max(abs(p-t) for state in metric["arms"].values() for p, t in zip(state["position_rad"], state["target_rad"]))
            require(lag <= .02, "Correction chased unfinished joint transit")
            expected_arms = []
            if metric["position_error_mm"] > 2 or metric["normal_error_deg"] > .5:
                expected_arms.append("Arm14")
            if metric["dropper_position_error_mm"] > 2:
                expected_arms.append("Arm10")
            require(correction.get("arms") == expected_arms and set(correction.get("fits", {})).issubset(expected_arms),
                    "Bias correction moved an arm already within tolerance")
            require(correction.get("invalid") in (None, "settling_correction_ik", "settling_correction_path_proxy"),
                    "Unexpected correction-stage rejection reason")
            if correction.get("invalid") is None:
                require(set(correction.get("fits", {})) == set(expected_arms)
                    and all(fit.get("solved") is True for fit in correction["fits"].values()),
                    "Applied correction lacks solved arm targets")
    require(all(c.get("cell_id") in by_id for c in corrections), "Unknown correction workcell")
    # Native and setup-only rows both retain initial qualification/path
    # admission, allowing complete reconstruction of the common wait barrier.
    complete_eligibility = (isinstance(trial.get("enabled"), list) and len(trial["enabled"]) == 35
        and all("qualification" in row and "path_proxy" in row for row in trial["rows"]))
    require(complete_eligibility, "Audit requires complete initial qualification/path eligibility records")
    require(all(isinstance(row["qualification"], dict) and type(row["qualification"].get("accepted")) is bool
        and (not row["qualification"]["accepted"] or isinstance(row["path_proxy"], dict)
             and type(row["path_proxy"].get("accepted")) is bool) for row in trial["rows"]),
        "Missing native initial qualification/path eligibility flags")
    eligible = [bool(trial["enabled"][i] and row["qualification"].get("accepted")
        and row["path_proxy"] and row["path_proxy"].get("accepted") and not row.get("path_rejection"))
        for i, row in enumerate(trial["rows"])]
    short_windows = eligible_checks = 0
    for iteration in sorted({m["iteration"] for m in records[0]}):
        indices = [j for j, m in enumerate(records[0]) if m["iteration"] == iteration]
        maximum = 160 if iteration == 0 else 32
        if len(indices) < maximum:
            short_windows += 1
            for i, use in enumerate(eligible):
                prior_failure = any(c["cell_id"] == cells[i]["id"] and c["iteration"] < iteration
                                    and c.get("invalid") is not None for c in corrections)
                if use and not prior_failure:
                    metric = records[i][indices[-1]]
                    require((accepted(metric) and all_articulation_speed(metric) <= .005 and metric["dropper_position_error_mm"] <= 2)
                        or independent_feedback_ready(metric),
                        "Stabilization exited early before release or joint-lag-aware feedback readiness")
                    eligible_checks += 1
    require(max(history[-1]["physics_step"] for history in records) <= trial["release_step"],
            "Release preceded final stability measurement")
    return dict(measured_windows=windows, bias_corrections=len(corrections), original_holding_gains_restored=True,
                quiet_before_every_correction_verified=True,
                full_articulation_velocity_gate_verified=True,
                early_exit=dict(short_global_windows=short_windows, eligible_cell_stage_checks=eligible_checks,
                    complete_initial_qualification_path_eligibility_logged=complete_eligibility,
                    readiness="Release-ready OR quiet plus maximum arm joint lag <=0.02 rad",
                    scope="Full currently eligible-cell reconstruction from initial qualification/path rows and prior correction failures"),
                scope="Readback arithmetic and controller cadence/gates, not collision-certified paths or optimizer arithmetic")

def verify_checkpoint_chain(run, executed_source=None):
    """Check every continuous trial against its frozen CPU-loaded checkpoint."""
    import torch
    from continuous_feasibility import LOW, HIGH, decode_action, geometry_coverage
    from diverse_policy import (build_observations, weight_hash, actor_weight_hash, sample)
    from farm_contract import reward, statistics

    backend = backend_for_run(run)
    validate_plan = backend.validate_study
    ContinuousPolicy = backend.policy_factory
    load_checkpoint = backend.load_checkpoint

    run = Path(run)
    spec, study, report = (read(run/name) for name in ("contract.json", "study.json", "result.json"))
    study_hash, geometry_hash, domain_hash = validate_plan(study,spec)
    require(report.get("schema") == study["schema"] and report.get("status") == "completed"
            and report.get("study_fingerprint") == study_hash
            and report.get("geometry_fingerprint") == geometry_hash
            and report.get("workcells") == 35 and report.get("articulations") == 70,
            "Continuous result identity changed")
    batches = study["training_batches"]
    repeats = study["assessment_batches"]
    require(len(report["batches"]) == batches and len(report["updates"]) == batches//4
            and len(report["assessments"]) == 2*repeats, "Incomplete continuous pilot")
    if executed_source is not None:
        executed_source = Path(executed_source)
        require(isinstance(report.get("source_sha256"),dict) and report["source_sha256"],
                "Missing native source hashes")
        for name, expected in report["source_sha256"].items():
            require(Path(name).name == name and name.endswith(".py"), "Unsafe native source filename")
            require(digest(executed_source/name) == expected, "Executed native source changed: " + name)
    torch.set_num_threads(2)
    expected_obs = build_observations(spec["cells"])
    observations = torch.from_numpy(expected_obs)
    train_indices = [i for i,c in enumerate(study["cells"]) if c["split"] == "train"]
    expected_train_ids = [study["cells"][i]["id"] for i in train_indices]
    checkpoint_hashes, trial_hashes, policies = {}, {}, {}
    max_logp_error = 0.0
    training_rows, evaluation_rows = [], {"initial":[],"final":[]}
    membership, coverage_rows, home_readbacks = [], [], []
    coverage_batches = study.get("coverage_batches",0)
    require(len(report.get("coverage",[])) == coverage_batches, "Incomplete geometry coverage")
    coverage_plans = []
    if coverage_batches:
        path = evidence_file(run,"feasibility.json")
        require(digest(path) == study["feasibility_sha256"], "Geometry feasibility map changed")
        feasibility = read(path)
        signed(feasibility,"Geometry feasibility map")
        require(feasibility.get("source_fingerprint") == geometry_hash,
                "Geometry coverage uses a different source")
        if study.get("ik_protocol") == "full_robust_v2":
            require(feasibility.get("ik_protocol") == "full_robust_v2",
                    "Native robust IK and feasibility-map solver protocols differ")
        coverage_plans = geometry_coverage(feasibility,spec["cells"],batches=coverage_batches)

    def checkpoint(reference, updates, attempts):
        path = evidence_file(run, reference["file"], suffix=".pt")
        require(digest(path) == reference["sha256"], "Checkpoint checksum changed")
        checkpoint_hashes[path.name] = digest(path)
        if path.name not in policies:
            policies[path.name] = load_checkpoint(path, study_fingerprint=study_hash,
                geometry_fingerprint=geometry_hash, domain_fingerprint=domain_hash,
                device="cpu", expected_updates=updates, expected_attempted_shots=attempts)
        policy, saved = policies[path.name]
        require(saved["seed"] == study["seed"] and saved["update_count"] == updates
                and saved["attempted_shots"] == attempts,
                "Sampling checkpoint counters changed")
        require(weight_hash(policy) == reference["policy_hash"]
                and actor_weight_hash(policy) == reference["actor_hash"],
                "Sampling checkpoint weights changed")
        return policy

    initial = checkpoint(report["initial"],0,0)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(study["seed"])
        fresh=ContinuousPolicy()
    require(weight_hash(fresh)==weight_hash(initial),
            "Initial checkpoint differs from declared fresh seeded initialization")
    final = checkpoint(report["final"],batches//4,batches*25)
    require(report["initial_hash"] == weight_hash(initial)
            and report["final_hash"] == weight_hash(final), "Initial/final policy identity changed")
    expected_hash, expected_actor = weight_hash(initial), actor_weight_hash(initial)

    def trial(summary, mode, batch, expected_reference, updates, attempts, coverage_plan=None):
        nonlocal max_logp_error
        path = evidence_file(run,summary["file"],trial=True)
        require(digest(path) == summary["sha256"], "Trial checksum changed")
        trial_hashes[summary["file"]] = digest(path)
        data = read(path)
        policy = checkpoint(data["checkpoint"],updates,attempts)
        require(data["checkpoint"] == expected_reference
                and data.get("schema") == study["schema"]
                and data.get("study_fingerprint") == study_hash
                and data.get("mode") == summary.get("mode") == mode
                and data.get("batch") == summary.get("batch") == batch
                and data.get("update_count") == summary.get("update_count") == updates
                and data.get("policy_hash") == weight_hash(policy),
                "Trial identity or policy binding changed")
        enabled = ([c["split"] == "train" for c in study["cells"]] if mode == "train"
                   else coverage_plan["enabled"] if mode == "coverage" else [True]*35)
        active = [i for i,value in enumerate(enabled) if value]
        active_ids = [study["cells"][i]["id"] for i in active]
        require(data.get("enabled") == enabled and data.get("active_cell_ids") == active_ids
                and data.get("optimizer_samples") == (25 if mode == "train" else 0)
                and data.get("deterministic") is (mode == "eval")
                and data.get("policy_action_used") is (mode != "coverage"),
                "Training membership or frozen-evaluation protocol changed")
        home = data.get("home",{})
        require(home.get("protocol") == HOME_PROTOCOL and home.get("episode_reset") is True
                and isinstance(home.get("path_eligible"),list)
                and len(home["path_eligible"]) == 35
                and home["path_eligible"] == [True]*35
                and len(home.get("path_checks",[])) == 35
                and all(check.get("accepted") is use for check,use in zip(home["path_checks"],home["path_eligible"]))
                and type(home.get("physics_step")) is int
                and home["physics_step"] <= data.get("release_step",-1),
                "Common-home path/readback protocol changed")
        home_arrays = {}
        for key in ("target_joint_rad","measured_joint_rad","measured_joint_velocity_rad_s"):
            require(set(home.get(key,{})) == {"Arm10","Arm14"}, "Incomplete home arm readback")
            home_arrays[key] = np.stack([finite(home[key][name],(35,6),"home "+key)
                                          for name in ("Arm10","Arm14")])
        for arm_index, arm in enumerate(("Arm10", "Arm14")):
            require(np.allclose(home_arrays["target_joint_rad"][arm_index],
                np.asarray(backend.support["canonical_home_joint_rad"][arm])[None], atol=1e-7, rtol=0),
                "Diverse support canonical home differs from commanded episode home")
        home_readbacks.append(dict(mode=mode,updates=updates,batch=batch,
                                   eligible=np.asarray(home["path_eligible"],dtype=bool),**home_arrays))
        for row,use in zip(data["rows"],home["path_eligible"]):
            require(use or row.get("invalid"), "Blocked common-home path admitted a release")
        actual_obs = finite(data["observations"],(35,5),"policy observations")
        latent = finite(data["latent"],(35,6),"hybrid region/Gaussian action latent")
        region_indices = finite(data["region_indices"],(35,),"categorical region indices")
        require(np.array_equal(latent[:,0],region_indices), "Saved categorical region differs from hybrid latent")
        normalized = finite(data["actions_normalized"],(35,5),"bounded action")
        logp = finite(data["logp"],(35,),"saved log probability")
        require(np.array_equal(actual_obs,expected_obs), "Policy observations differ from bucket geometry")
        with torch.no_grad():
            latent_t = torch.from_numpy(latent)
            actual_normalized = latent_t[:,1:].tanh().numpy()
            actual_logp = policy.log_prob(observations,latent_t).numpy()
            if mode == "eval":
                require(np.allclose(latent,sample(policy,observations,deterministic=True)["latent"].numpy(),atol=2e-6,rtol=2e-6),
                        "Frozen deterministic assessment action changed")
        require(np.allclose(normalized,actual_normalized,atol=2e-6,rtol=2e-6),
                "Saved bounded action does not match sampled latent")
        require(np.allclose(logp,actual_logp,atol=LOGP_TOLERANCE,rtol=LOGP_TOLERANCE),
                "Saved log probability differs from frozen sampling checkpoint")
        max_logp_error = max(max_logp_error,float(np.abs(logp-actual_logp).max()))
        verify_joint_components(policy, observations, latent_t, data)
        require(data.get("support_fingerprint") == backend.support["fingerprint"], "Trial diverse support binding changed")
        physical = backend.physical_actions(normalized,region_indices)
        expected_actions = (coverage_plan["actions"] if mode == "coverage" else
                            [decode_action(c,a,f"{mode}-{batch}-{i}")
                             for i,(c,a) in enumerate(zip(spec["cells"],physical))])
        require(data.get("actions") == expected_actions and len(data["rows"]) == 35,
                "Executed continuous action differs from direct policy decoding")
        for i,(row,action) in enumerate(zip(data["rows"],expected_actions)):
            require(row["cell_id"] == study["cells"][i]["id"]
                    and row["split"] == study["cells"][i]["split"]
                    and all(row.get(k) == v for k,v in action.items()),
                    "Native outcome is not bound to its requested action/layout")
            if mode != "coverage" and enabled[i] and not backend.support["cells"][i]["regions"]:
                require(row.get("invalid") == "no_geometry_support", "Unsupported context was silently skipped or released")
        rewards = finite(data["rewards"],(35,),"terminal reward")
        expected_rewards = np.asarray([reward(r) for r in data["rows"]],dtype=np.float32)
        require(np.array_equal(rewards,expected_rewards), "Saved rewards differ from native scalar reward rule")
        stats = statistics([data["rows"][i] for i in active])
        for key,value in stats.items():
            require(key in summary and np.isclose(summary[key],value,rtol=0,atol=1e-7),
                    "Trial summary counts changed: " + key)
        if mode == "train":
            membership.append(dict(batch=batch, cell_ids=active_ids,
                includes_invalid_attempts=True, samples=len(active),
                input_sha256=canonical_hash(dict(observations=actual_obs[active].tolist(),
                    latent=latent[active].tolist(), logp=logp[active].tolist(), rewards=rewards[active].tolist()))))
        return data,policy

    for batch,(summary,plan) in enumerate(zip(report.get("coverage",[]),coverage_plans)):
        data,_ = trial(summary,"coverage",batch,report["initial"],0,0,plan)
        coverage_rows.extend(row for row,use in zip(data["rows"],plan["enabled"]) if use)
    for batch,summary in enumerate(report["batches"]):
        reference = read(evidence_file(run,summary["file"],trial=True))["checkpoint"]
        require(reference["policy_hash"] == expected_hash and reference["actor_hash"] == expected_actor,
                "Training policy update chain changed")
        data,policy = trial(summary,"train",batch,reference,batch//4,batch*25)
        training_rows.extend(data["rows"][i] for i in train_indices)
        if batch%4 == 3:
            change = report["updates"][batch//4]
            require(change.get("update_count") == batch//4+1
                    and change.get("training_batches") == list(range(batch-3,batch+1))
                    and change.get("before") == expected_hash and change.get("actor_before") == expected_actor
                    and change.get("sample_count") == change.get("samples") == 100
                    and len(change.get("epochs",[])) == study["ppo"]["epochs"],
                    "PPO update membership or hash chain changed")
            for epoch in change["epochs"]:
                require(epoch.get("samples") == 100
                        and all(type(v) in (int,float) and np.isfinite(v) for v in epoch.values()),
                        "Nonfinite or incomplete PPO diagnostics")
            expected_hash,expected_actor = change["after"],change["actor_after"]
    require(expected_hash == weight_hash(final) and expected_actor == actor_weight_hash(final),
            "Final checkpoint does not follow logged PPO updates")
    for stage,label in enumerate(("initial","final")):
        reference = report[label]
        after_batches = stage*batches
        for repeat in range(repeats):
            summary = report["assessments"][stage*repeats+repeat]
            data,_ = trial(summary,"eval",repeat,reference,after_batches//4,after_batches*25)
            evaluation_rows[label].extend(data["rows"])
    require(len(training_rows) == 25*batches and len(membership) == batches
            and all(m["cell_ids"] == expected_train_ids and m["samples"] == 25 for m in membership),
            "Held-out layouts entered optimization or rejected attempts were filtered")
    with torch.no_grad():
        initial_pick=sample(initial,observations,deterministic=True)
        final_pick=sample(final,observations,deterministic=True)
        initial_mean=initial_pick["actions_normalized"].numpy()
        final_mean=final_pick["actions_normalized"].numpy()
    initial_physical=backend.physical_actions(initial_mean,initial_pick["region_indices"].numpy())
    final_physical=backend.physical_actions(final_mean,final_pick["region_indices"].numpy())
    action_delta=final_physical-initial_physical
    mean_action_change=dict(action_names=study["domain"]["names"],
        changed_cells=int((np.abs(final_physical-initial_physical).max(axis=1)>1e-7).sum()),
        changed_region_cells=int((initial_pick["region_indices"]!=final_pick["region_indices"]).sum()),
        max_absolute_physical_delta=np.abs(action_delta).max(axis=0).tolist(),
        mean_absolute_physical_delta=np.abs(action_delta).mean(axis=0).tolist(),
        per_cell=[dict(cell_id=cell["id"],split=cell["split"],initial_region=int(initial_pick["region_indices"][i]), final_region=int(final_pick["region_indices"][i]), initial=initial_physical[i].tolist(),
                       final=final_physical[i].tolist(),delta=action_delta[i].tolist())
                  for i,cell in enumerate(study["cells"])],
        scope="Direct deterministic network outputs before IK; changes alone do not prove a capture-rate improvement")
    return dict(schema=SCHEMA,status="passed",study_fingerprint=study_hash,
        geometry_fingerprint=geometry_hash,domain_fingerprint=domain_hash,
        ik_protocol=study.get("ik_protocol","not_declared_in_historical_smoke"),
        source_result_sha256=digest(run/"result.json"), study_sha256=digest(run/"study.json"),
        checkpoints=checkpoint_hashes,trials=trial_hashes,updates=len(report["updates"]),
        optimizer_samples=len(training_rows), training_batches=batches,
        train_cell_ids=expected_train_ids,
        held_out_cell_ids={label:[c["id"] for c in study["cells"] if c["split"] == label]
                           for label in ("validation","test")},
        optimizer_membership_sha256=canonical_hash(membership),
        max_logp_error=max_logp_error,logp_tolerance=LOGP_TOLERANCE,
        fresh_seeded_initialization_verified=True,
        support_fingerprint=backend.support["fingerprint"], support_sha256=digest(backend.support_path),
        support_evidence=backend.support_evidence,
        actor_changed=actor_weight_hash(initial)!=actor_weight_hash(final),
        deterministic_mean_action_change=mean_action_change,
        train_statistics=statistics(training_rows),
        geometry_coverage_attempts=len(coverage_rows),
        geometry_coverage_excluded_from_optimizer=True,
        home_comparison=home_comparison(home_readbacks,study["cells"]),
        assessment_statistics={label:{split:statistics([r for r in rows if r["split"] == split])
                                      for split in ("train","validation","test")}
                               for label,rows in evaluation_rows.items()},
        scope="CPU hybrid region/residual checkpoint/input/joint likelihood/reward/update-membership audit; native telemetry rescoring and optimizer arithmetic are separate")


def verify_first_update(run,executed_source=None):
    """Audit the first 100 on-policy attempts while the native pilot continues.

    This reads immutable completed trial/checkpoint files and one progress
    snapshot. It never rewrites the run, creates a GPU session or extrapolates
    a final score from partial training. The full completed-run audit is still
    required before publishing final results.
    """
    import torch
    from continuous_feasibility import LOW,HIGH,decode_action
    from diverse_policy import build_observations,weight_hash,actor_weight_hash
    from farm_contract import reward,statistics

    backend=backend_for_run(run)
    validate_plan=backend.validate_study
    load_checkpoint=backend.load_checkpoint

    run=Path(run)
    spec,study=read(run/"contract.json"),read(run/"study.json")
    study_hash,geometry_hash,domain_hash=validate_plan(study,spec)
    snapshot=(run/"progress.json").read_bytes()
    progress=json.loads(snapshot)
    require(progress.get("schema")==study["schema"]
            and progress.get("study_fingerprint")==study_hash
            and len(progress.get("batches",[]))>=4 and len(progress.get("updates",[]))>=1,
            "First complete PPO update is not available")
    change=progress["updates"][0]
    require(change.get("update_count")==1 and change.get("training_batches")==[0,1,2,3]
            and change.get("sample_count")==change.get("samples")==100
            and len(change.get("epochs",[]))==study["ppo"]["epochs"],
            "First PPO update membership changed")
    obs=build_observations(spec["cells"]); obs_t=torch.from_numpy(obs)
    train=[i for i,c in enumerate(study["cells"]) if c["split"]=="train"]
    expected_enabled=[i in train for i in range(35)]
    ids=[study["cells"][i]["id"] for i in train]
    trials,checkpoints,rows,membership={}, {}, [], []
    max_logp_error=0.

    def checkpoint(path,updates,attempts):
        checksum=digest(path)
        model,saved=load_checkpoint(path,study_fingerprint=study_hash,geometry_fingerprint=geometry_hash,
                                   domain_fingerprint=domain_hash,expected_updates=updates,
                                   expected_attempted_shots=attempts)
        require(digest(path)==checksum,"Checkpoint is still being written")
        require(saved["seed"]==study["seed"],"First-update policy seed changed")
        checkpoints[path.name]=checksum
        return model

    before=None
    for batch,summary in enumerate(progress["batches"][:4]):
        path=evidence_file(run,summary["file"],trial=True)
        require(digest(path)==summary["sha256"],"First-update trial checksum changed")
        trials[summary["file"]]=digest(path)
        data=read(path);reference=data["checkpoint"]
        cp=evidence_file(run,reference["file"],suffix=".pt")
        model=checkpoint(cp,0,batch*25)
        require(reference["sha256"]==checkpoints[cp.name]
                and reference["policy_hash"]==data.get("policy_hash")==change["before"]==weight_hash(model)
                and reference["actor_hash"]==change["actor_before"]==actor_weight_hash(model),
                "First-update sampling checkpoint changed")
        require(data.get("schema")==study["schema"] and data.get("study_fingerprint")==study_hash
                and data.get("mode")==summary.get("mode")=="train"
                and data.get("batch")==summary.get("batch")==batch
                and data.get("update_count")==summary.get("update_count")==0
                and data.get("enabled")==expected_enabled and data.get("active_cell_ids")==ids
                and data.get("optimizer_samples")==25 and data.get("deterministic") is False
                and data.get("policy_action_used") is True,
                "First-update train/held-out membership changed")
        actual_obs=finite(data["observations"],(35,5),"first-update observations")
        latent=finite(data["latent"],(35,6),"first-update hybrid latent")
        regions=finite(data["region_indices"],(35,),"first-update region indices")
        require(np.array_equal(latent[:,0],regions),"First-update categorical latent mismatch")
        normalized=finite(data["actions_normalized"],(35,5),"first-update actions")
        stored_logp=finite(data["logp"],(35,),"first-update likelihood")
        require(np.array_equal(actual_obs,obs),"First-update observation geometry changed")
        with torch.no_grad():
            latent_t=torch.from_numpy(latent)
            actual_logp=model.log_prob(obs_t,latent_t).numpy()
            require(np.allclose(normalized,latent_t[:,1:].tanh().numpy(),rtol=2e-6,atol=2e-6),
                    "First-update bounded action changed")
        require(np.allclose(actual_logp,stored_logp,rtol=LOGP_TOLERANCE,atol=LOGP_TOLERANCE),
                "First-update stored likelihood differs from sampling model")
        max_logp_error=max(max_logp_error,float(np.abs(actual_logp-stored_logp).max()))
        verify_joint_components(model,obs_t,latent_t,data)
        actions=[decode_action(c,a,f"train-{batch}-{i}")
                 for i,(c,a) in enumerate(zip(spec["cells"],backend.physical_actions(normalized,regions)))]
        require(data["actions"]==actions and len(data["rows"])==35,"First-update direct action decoding changed")
        for i,(row,action) in enumerate(zip(data["rows"],actions)):
            require(row["cell_id"]==study["cells"][i]["id"] and row["split"]==study["cells"][i]["split"]
                    and all(row.get(k)==v for k,v in action.items()),"First-update outcome/action binding changed")
        rewards=finite(data["rewards"],(35,),"first-update reward")
        require(np.array_equal(rewards,np.asarray([reward(r) for r in data["rows"]],dtype=np.float32)),
                "First-update terminal reward changed")
        selected=[data["rows"][i] for i in train]
        for key,value in statistics(selected).items():
            require(np.isclose(summary[key],value,rtol=0,atol=1e-7),"First-update counts changed")
        rows.extend(selected)
        membership.append(dict(batch=batch,ids=ids,latent=latent[train].tolist(),
                               logp=stored_logp[train].tolist(),rewards=rewards[train].tolist()))
        before=model
    next_name="policy-final.pt" if study["training_batches"]==4 else "policy-before-004.pt"
    after=checkpoint(evidence_file(run,next_name,suffix=".pt"),1,100)
    require(change["after"]==weight_hash(after) and change["actor_after"]==actor_weight_hash(after),
            "First PPO update does not match the next actual checkpoint")
    for epoch in change["epochs"]:
        require(epoch.get("samples")==100 and all(type(v) in (int,float) and np.isfinite(v) for v in epoch.values()),
                "Invalid first-update optimization diagnostic")
    return dict(schema="sircus-diverse-settled-first-update-audit-v1",status="passed",
                study_fingerprint=study_hash,geometry_fingerprint=geometry_hash,
                domain_fingerprint=domain_hash,ik_protocol=study.get("ik_protocol"),
                updates=1,optimizer_samples=100,train_cell_ids=ids,
                checkpoints=checkpoints,trials=trials,first_update=change,
                optimizer_membership_sha256=canonical_hash(membership),
                progress_snapshot_sha256=hashlib.sha256(snapshot).hexdigest(),
                source_py_sha256={p.name:digest(p) for p in sorted(Path(executed_source).glob("*.py"))} if executed_source else {},
                max_logp_error=max_logp_error,train_statistics=statistics(rows),
                actor_changed=actor_weight_hash(before)!=actor_weight_hash(after),
                scope="Partial CPU audit of first 100 attempted train shots and one checkpoint update; no final score or generalization claim. Completed-run native and checkpoint audits remain required.")


def compare_runs(broad_run, guided_run, diverse_run):
    """Independently count matched frozen outcomes, allowing the declared driver change."""
    from farm_contract import statistics
    from feasibility_guided_audit import backend_for_run as guided_backend
    roots = {"broad": Path(broad_run), "guided": Path(guided_run), "diverse": Path(diverse_run)}
    studies, specs, results = {}, {}, {}
    for name, root in roots.items():
        studies[name], specs[name], results[name] = (read(root/file) for file in ("study.json", "contract.json", "result.json"))
        if name == "broad":
            chain.validate_study(studies[name], specs[name])
        elif name == "guided":
            guided_backend(root).validate_study(studies[name], specs[name])
        else:
            backend_for_run(root).validate_study(studies[name], specs[name])
        result, study = results[name], studies[name]
        require(result.get("status") == "completed" and result.get("schema") == study["schema"]
            and result.get("study_fingerprint") == study["fingerprint"]
            and result.get("geometry_fingerprint") == study["geometry_fingerprint"], "Comparison requires completed bound native results")
        require(specs[name] == specs["broad"] and study["cells"] == studies["broad"]["cells"], "Comparison layouts or split changed")
        for key in ("training_batches", "batches_per_update", "assessment_batches", "seed", "ppo",
                    "measurement_hold_s", "measurement_window_s", "home_hold_s", "episode_start", "filmed_training_batches"):
            require(study.get(key) == studies["broad"].get(key), "Comparison budget/protocol changed: "+key)
        require(study["source"]["camera_sha256"] == studies["broad"]["source"]["camera_sha256"], "Comparison camera changed")
        for filename in SCORING_FILES:
            require(result.get("source_sha256", {}).get(filename)
                    == results["broad"].get("source_sha256", {}).get(filename)
                    == digest(Path(__file__).with_name(filename)), "Comparison scoring/release gate source changed: "+filename)

    def method(name):
        root, result, study = roots[name], results[name], studies[name]
        def rows(summary, mode):
            path = evidence_file(root, summary["file"], trial=True)
            require(digest(path) == summary["sha256"], "Comparison trial checksum changed")
            data = read(path)
            require(data.get("mode") == summary.get("mode") == mode
                and data.get("schema") == study["schema"] and data.get("study_fingerprint") == study["fingerprint"]
                and [(r["cell_id"], r["split"]) for r in data.get("rows", [])]
                == [(c["id"], c["split"]) for c in study["cells"]], "Comparison trial membership changed")
            expected_enabled = [c["split"] == "train" for c in study["cells"]] if mode == "train" else [True]*35
            require(data.get("enabled") == expected_enabled and data.get("optimizer_samples") == (25 if mode == "train" else 0),
                    "Comparison enabled/optimizer membership changed")
            return data["rows"]
        def counts(values):
            stats = statistics(values)
            return dict(attempts=stats["attempts"], captures=stats["captures"], clean=stats["clean"], rejected=stats["invalid"])
        require(len(result["batches"]) == study["training_batches"]
            and len(result["assessments"]) == 2*study["assessment_batches"], "Incomplete matched comparison budget")
        train = [r for summary in result["batches"] for r in rows(summary, "train") if r["split"] == "train"]
        require(len(train) == 25*study["training_batches"], "Comparison train sample count changed")
        frozen = {split: {} for split in ("train", "validation", "test")}
        repeats = study["assessment_batches"]
        for stage, label in enumerate(("initial", "final")):
            values = [r for summary in result["assessments"][stage*repeats:(stage+1)*repeats] for r in rows(summary, "eval")]
            for split in frozen:
                frozen[split][label] = counts([r for r in values if r["split"] == split])
        return dict(source_result_sha256=digest(root/"result.json"), training=counts(train), frozen=frozen)
    methods = {name: method(name) for name in roots}
    return dict(schema="sircus-diverse-settled-comparison-v1", status="completed", methods=methods,
        source_result_sha256=methods["diverse"]["source_result_sha256"],
        scoring_source_sha256={name: digest(Path(__file__).with_name(name)) for name in SCORING_FILES},
        changed_factors=["geometry regions", "hybrid policy/action support and exploration", "bounded driven setup controller and equivalent joint representation"],
        scope="Matched single-seed attempted-shot budgets at inherited 25/5/5 coordinates, not an isolated PPO ablation. Geometry assists held-out contexts. Controller/precompute/native elapsed costs differ. Nominal repeats are not independent noise/layout robustness trials.")


def audit_setup_check(run, original_run):
    """Audit the fixed-request controller check without inventing shot releases.

    Only the first four original *training* requests are replayed. A derived
    decision cutoff permits reuse of measurement-history checks; it is explicitly
    not a measured release event. Producer code identity is a separate launch
    receipt, while this receipt binds all raw request/readback bytes directly.
    """
    from collections import Counter
    from settled_transit import protocol
    from settling_contract import accepted
    expected_original = "cd90e6ec9c1e1bc3729ca5113b67d98c6ebf680f0964cec68065acb1f98138ca"
    run, original_run = Path(run), Path(original_run)
    original_result_path = evidence_file(original_run, "result.json")
    require(digest(original_result_path) == expected_original, "Setup audit requires the frozen original guided run")
    original = read(original_result_path)
    spec, old_study = read(evidence_file(original_run, "contract.json")), read(evidence_file(original_run, "study.json"))
    from feasibility_guided_audit import backend_for_run as guided_backend
    guided_backend(original_run).validate_study(old_study, spec)
    summary_path, raw_path = evidence_file(run, "setup-check.json"), evidence_file(run, "setup-check-raw.json")
    summary, raw = read(summary_path), read(raw_path)
    require(summary.get("schema") == "sircus-fixed-request-setup-check-v1" and summary.get("status") == "completed"
        and summary.get("source_result_sha256") == expected_original
        and summary.get("geometry_fingerprint") == signed(spec, "Setup-check geometry")
        and summary.get("settling_protocol") == protocol() and summary.get("ball_releases") == 0,
        "Setup-check identity, geometry or no-release scope changed")
    records = raw.get("records")
    require(isinstance(records, list) and len(records) == 4 and len(original.get("batches", [])) >= 4,
            "Setup-check must contain exactly the first four training batches")
    before, after, pairs, history_receipts, source_trials = [], [], [], [], {}
    cells = old_study["cells"]
    for index, (record, source_summary) in enumerate(zip(records, original["batches"][:4])):
        source_path = evidence_file(original_run, source_summary["file"], trial=True)
        source_sha = digest(source_path)
        old = read(source_path)
        require(source_sha == source_summary["sha256"] == record.get("source_trial_sha256")
            and record.get("source_trial") == source_summary["file"]
            and old.get("schema") == old_study["schema"]
            and old.get("study_fingerprint") == old_study["fingerprint"]
            and old.get("mode") == source_summary.get("mode") == "train"
            and old.get("batch") == source_summary.get("batch") == index,
            "Setup-check source training order/hash changed")
        expected_enabled = [cell["split"] == "train" for cell in cells]
        require(record.get("actions") == old.get("actions") and record.get("enabled") == old.get("enabled") == expected_enabled
            and len(old.get("rows", [])) == 35 and len(record.get("invalid", [])) == 35,
            "Setup-check requests, membership or status vector changed")
        require(all(isinstance(record.get(key), list) and len(record[key]) == 35
                    for key in ("qualifications", "path_checks", "path_reasons")),
                "Setup-check missing initial qualification/path eligibility readbacks")
        histories = record.get("settling_history", [])
        require(len(histories) == 35 and all(isinstance(h, list) and h for h in histories),
                "Setup-check missing measured histories")
        # This is a history-validation cutoff, not an invented ball release.
        decision_cutoff = max(h[-1]["physics_step"] for h in histories)
        virtual = dict(record, setup_controller=summary["settling_protocol"],
            rows=[dict(invalid=invalid, settled=history[-1], qualification=record["qualifications"][i],
                       path_proxy=record["path_checks"][i], path_rejection=record["path_reasons"][i])
                  for i, (invalid, history) in enumerate(zip(record["invalid"], histories))], release_step=decision_cutoff)
        history_receipt = verify_settling_history(virtual, spec["scene"], cells,
                                                dict(settling_protocol=summary["settling_protocol"]))
        history_receipt.update(source_trial=source_summary["file"], setup_decision_step=decision_cutoff,
            cutoff_scope="Final setup-decision clock only; no physical ball release or scoring event")
        history_receipts.append(history_receipt)
        source_trials[source_summary["file"]] = source_sha
        for i, use in enumerate(expected_enabled):
            require(old["rows"][i].get("cell_id") == cells[i]["id"]
                and old["rows"][i].get("split") == cells[i]["split"], "Setup-check source row/layout association changed")
            if not use:
                require(record["invalid"][i] == "inactive_split", "Setup check moved a held-out context into active status")
                continue
            metric = histories[i][-1]
            require(record["invalid"][i] is not None
                or accepted(metric) and metric["dropper_position_error_mm"] <= 2,
                "Setup admitted a request outside unchanged release-ready thresholds")
            before.append(old["rows"][i]["invalid"])
            after.append(record["invalid"][i])
            pairs.append(dict(cell_id=cells[i]["id"], batch=index, before_invalid=before[-1], after_invalid=after[-1],
                position_error_mm=metric["position_error_mm"], position_drift_mm=metric["position_drift_mm"],
                joint_speed_rad_s=metric["joint_speed_rad_s"]))
    expected = dict(attempts=100, before_admitted=before.count(None), after_admitted=after.count(None),
        before_failures=dict(Counter(x for x in before if x)), after_failures=dict(Counter(x for x in after if x)),
        repaired=sum(a is not None and b is None for a, b in zip(before, after)),
        regressions=sum(a is None and b is not None for a, b in zip(before, after)), paired=pairs)
    require(len(before) == len(after) == 100 and all(summary.get(key) == value for key, value in expected.items()),
            "Setup-check summary/paired admission accounting changed")
    require(summary.get("selection") == "First four saved training batches in original order, all 25 training contexts; no held-out/outcome filtering",
            "Setup-check prespecified selection changed")
    return dict(schema="sircus-fixed-request-setup-audit-v1", status="passed", audit_source_sha256=digest(__file__),
        source_result_sha256=expected_original, source_setup_sha256=digest(summary_path), source_raw_sha256=digest(raw_path),
        geometry_fingerprint=spec["fingerprint"], source_trial_sha256=source_trials,
        checked_training_batches=4, checked_attempts=100, ball_releases=0,
        **{key: value for key, value in expected.items() if key != "paired"},
        history_audits=history_receipts, scope="Fixed first-four-training-batch setup-only comparison; independent raw request/history/counter checks. No capture measurement, policy training or physical release is claimed. Producer source identity remains separately bound by its launch receipt.")


GUARD_PREFLIGHT_COMMIT = "5e0b664a05534d83ceb1948651dfabdd2191ef40"
GUARD_PREFLIGHT_ARCHIVE_SHA256 = "e5076ff392360946fd2368fa590dc8a25e2eaa4cacd04a5e78c3c3b5f8a6e6ee"
GUARD_UNCHANGED_FILES = ("continuous_pilot.py", "diverse_pilot.py", "film_cylinder_farm.py", "feasibility_contract.py")


def _ast_key(node):
    return ast.dump(node, include_attributes=False)


def _method(tree, class_name, method_name):
    classes = [node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name]
    require(len(classes) == 1, "Expected one source class: "+class_name)
    methods = [node for node in classes[0].body if isinstance(node, ast.FunctionDef) and node.name == method_name]
    require(len(methods) == 1, "Expected one source method: "+method_name)
    return methods[0]


def verify_guard_only_controller_ast(old_text, new_text):
    """Accept exactly the two restored guard nodes and no other AST change."""
    old, new = ast.parse(old_text), ast.parse(new_text)
    home_guard = ast.parse('if not self.home_readback["path_eligible"][i]:\n reason="common_home_path_proxy_rejected"').body[0]
    distance_guard = ast.parse('for i,cell in enumerate(self.spec["cells"]):\n if cell["distance_m"] < .28:\n  invalid[i]="board_bucket_separation"').body[0]
    keys = [_ast_key(home_guard), _ast_key(distance_guard)]
    require(all(sum(_ast_key(n) == key for n in ast.walk(old)) == 0 for key in keys), "Old controller already contains a claimed new guard")
    require(all(sum(_ast_key(n) == key for n in ast.walk(new)) == 1 for key in keys), "Expected exactly the two literal restored guards")
    old_prepare, new_prepare = _method(old, "SettledTransitShots", "prepare"), _method(new, "SettledTransitShots", "prepare")
    require(len(new_prepare.body) >= 2 and _ast_key(new_prepare.body[-2]) == keys[1]
        and isinstance(new_prepare.body[-1], ast.Return) and _ast_key(new_prepare.body[-1]) == _ast_key(old_prepare.body[-1]),
        "Distance guard is not immediately before the unchanged final return")
    # Assignment/iteration targets have Store context; compare identifiers
    # explicitly instead of expression contexts when locating the source loop.
    loops = [node for node in new_prepare.body if isinstance(node, ast.For)
        and isinstance(node.target, ast.Tuple) and [getattr(x, "id", None) for x in node.target.elts] == ["i", "action"]
        and _ast_key(node.iter) == _ast_key(ast.parse("enumerate(actions)", mode="eval").body)]
    require(len(loops) == 1 and sum(_ast_key(n) == keys[0] for n in ast.walk(loops[0])) == 1,
        "Home-eligibility guard is not in the initial indexed action loop")
    removed = [0, 0]
    class Strip(ast.NodeTransformer):
        def visit_If(self, node):
            if _ast_key(node) == keys[0]:
                removed[0] += 1
                return None
            return self.generic_visit(node)
        def visit_For(self, node):
            if _ast_key(node) == keys[1]:
                removed[1] += 1
                return None
            return self.generic_visit(node)
    stripped = Strip().visit(copy.deepcopy(new))
    require(removed == [1, 1] and _ast_key(stripped) == _ast_key(old),
            "Controller AST differs beyond the two restored guard nodes")
    return dict(removed_guard_nodes=removed, whole_controller_ast_equal_after_exact_removal=True,
        old_controller_ast_sha256=hashlib.sha256(_ast_key(old).encode()).hexdigest(),
        stripped_new_controller_ast_sha256=hashlib.sha256(_ast_key(stripped).encode()).hexdigest())


def verify_home_source_invariant(continuous_text, runner_text):
    """Prove the literal return and assignment order, not an observed new flag."""
    home = _method(ast.parse(continuous_text), "ContinuousShots", "home")
    returns = [node for node in ast.walk(home) if isinstance(node, ast.Return)]
    require(len(returns) == 1 and isinstance(returns[0].value, ast.Call)
        and isinstance(returns[0].value.func, ast.Name) and returns[0].value.func.id == "dict", "Home return is not the fixed dictionary source invariant")
    eligible = [keyword.value for keyword in returns[0].value.keywords if keyword.arg == "path_eligible"]
    require(len(eligible) == 1 and _ast_key(eligible[0]) == _ast_key(ast.parse("[True]*self.n", mode="eval").body),
            "Home path eligibility is not literal True times workcell count")
    functions = [node for node in ast.parse(runner_text).body if isinstance(node, ast.FunctionDef) and node.name == "run_setup_check"]
    require(len(functions) == 1, "Expected one setup-check source function")
    function = functions[0]
    home_assignments = [node for node in ast.walk(function) if isinstance(node, ast.Assign)
        and any(_ast_key(t) == _ast_key(ast.Attribute(value=ast.Name(id="env", ctx=ast.Load()), attr="home_readback", ctx=ast.Store())) for t in node.targets)]
    require(len(home_assignments) == 1 and _ast_key(home_assignments[0].value) == _ast_key(ast.parse("env.home()", mode="eval").body),
            "Setup-check home_readback assignment is not the unchanged env.home() call")
    loops = [node for node in function.body if isinstance(node, ast.For)
        and isinstance(node.target, ast.Name) and node.target.id == "item"]
    require(len(loops) == 1, "Setup-check training-request loop changed")
    loop = loops[0]
    assignment_index = next((i for i, node in enumerate(loop.body) if node is home_assignments[0]), None)
    prepare_nodes = [(i, node) for i, node in enumerate(loop.body) if isinstance(node, ast.Assign)
        and isinstance(node.value, ast.Call) and _ast_key(node.value.func) == _ast_key(ast.parse("env.prepare", mode="eval").body)]
    require(assignment_index is not None and len(prepare_nodes) == 1 and assignment_index < prepare_nodes[0][0]
        and _ast_key(prepare_nodes[0][1].value) == _ast_key(ast.parse('env.prepare(old["actions"])', mode="eval").body),
        "Setup-check home assignment does not precede prepare of the saved request")
    for node in loop.body[assignment_index+1:prepare_nodes[0][0]]:
        require(not any(isinstance(child, ast.Attribute) and child.attr == "home_readback" for child in ast.walk(node)),
                "Setup-check changes home_readback before prepare")
    return dict(home_return_path_eligible_literal_true_times_n=True, home_readback_assigned_from_home_before_each_prepare=True,
        basis="Source-invariant home eligibility, not an observed newly added flag")


def audit_guard_restoration_applicability(run, old_source, new_source, *, original_run=None, prior_setup_audit=None, old_archive=None):
    """Bounded inert-guard proof for exactly the already measured 100 requests.

    This is neither a resimulation nor validation of newly proposed actions.
    Native preflight provenance remains the old 5e0b664 producer throughout.
    """
    run, old_source, new_source = Path(run), Path(old_source), Path(new_source)
    original_run = Path(original_run) if original_run is not None else run.parent/"feasibility-guided-pilot-20261105-v1"
    prior_setup_audit = Path(prior_setup_audit) if prior_setup_audit is not None else Path(str(run)+"-independent-audit.json")
    old_archive = Path(old_archive) if old_archive is not None else Path(str(old_source.parents[1])+".tar.gz")
    require(old_archive.is_file() and digest(old_archive) == GUARD_PREFLIGHT_ARCHIVE_SHA256,
            "Old preflight source archive identity changed")
    filenames = ("settled_transit.py",)+GUARD_UNCHANGED_FILES
    old_hashes, new_hashes = {}, {}
    with tarfile.open(old_archive, "r:gz") as archive:
        members = archive.getmembers()
        for name in filenames:
            old_file, new_file = old_source/name, new_source/name
            require(old_file.is_file() and new_file.is_file() and not old_file.is_symlink() and not new_file.is_symlink(),
                    "Missing or linked guard-proof source file")
            selected = [m for m in members if m.isfile() and m.name.endswith("sircus/isaac/"+name)]
            require(len(selected) == 1, "Old source archive member missing or ambiguous: "+name)
            data = archive.extractfile(selected[0]).read()
            require(data == old_file.read_bytes(), "Old source snapshot differs from pinned archive: "+name)
            old_hashes[name], new_hashes[name] = digest(old_file), digest(new_file)
    require(all(old_hashes[name] == new_hashes[name] for name in GUARD_UNCHANGED_FILES),
            "Runtime/source geometry files changed beyond restored guards")
    ast_receipt = verify_guard_only_controller_ast((old_source/"settled_transit.py").read_text(),
                                                 (new_source/"settled_transit.py").read_text())
    invariant = verify_home_source_invariant((new_source/"continuous_pilot.py").read_text(), (new_source/"diverse_pilot.py").read_text())
    launch_path, status_path = Path(str(run)+".launch.json"), Path(str(run)+".supervisor.json")
    require(launch_path.is_file() and status_path.is_file(), "Missing measured preflight launch/exit receipt")
    launch, status = read(launch_path), read(status_path)
    require(launch.get("source_commit") == status.get("source_commit") == GUARD_PREFLIGHT_COMMIT
        and status.get("status") == "completed" and status.get("returncode") == 0,
        "Guard applicability requires completed 5e preflight provenance")
    integrity = launch.get("integrity", {})
    require(integrity.get("archive_sha256") == GUARD_PREFLIGHT_ARCHIVE_SHA256
        and all(integrity.get("source_members_sha256", {}).get("sircus/isaac/"+name) == sha for name, sha in old_hashes.items()),
        "Measured preflight launch does not bind the pinned source archive members")
    measured = audit_setup_check(run, original_run)
    require(prior_setup_audit.is_file() and not prior_setup_audit.is_symlink(), "Missing prior independent V5 setup audit")
    prior = read(prior_setup_audit)
    require(prior.get("schema") == "sircus-fixed-request-setup-audit-v1" and prior.get("status") == "passed"
        and all(prior.get(key) == measured.get(key) for key in ("source_setup_sha256", "source_raw_sha256", "source_result_sha256",
            "geometry_fingerprint", "checked_attempts", "ball_releases", "source_trial_sha256", "before_admitted", "after_admitted", "repaired", "regressions")),
        "Prior independent preflight audit does not bind these measured requests")
    spec = read(evidence_file(original_run, "contract.json"))
    raw = read(evidence_file(run, "setup-check-raw.json"))
    require(len(spec["cells"]) == 35, "Guard proof requires measured 35-cell setup")
    geometry_tree = ast.parse((new_source/"feasibility_contract.py").read_text())
    definitions = [n for n in geometry_tree.body if isinstance(n, ast.FunctionDef) and n.name == "dynamic_cell"]
    require(len(definitions) == 1, "Expected one real dynamic_cell source function")
    namespace = dict(copy=copy, math=math)
    exec(compile(ast.Module(body=[copy.deepcopy(definitions[0])], type_ignores=[]), str(new_source/"feasibility_contract.py"), "exec"), namespace)
    dynamic_cell = namespace["dynamic_cell"]
    distances = []
    for batch, record in enumerate(raw["records"]):
        require(len(record["actions"]) == len(record["enabled"]) == 35, "Guard proof workcell count changed")
        for i, use in enumerate(record["enabled"]):
            if use:
                cell = dynamic_cell(spec["cells"][i], record["actions"][i])
                distance = cell["distance_m"]
                require(math.isfinite(distance) and distance >= .28, "Restored recomputed-distance guard changes a measured request")
                distances.append(dict(batch=batch, cell_id=spec["cells"][i]["id"], recomputed_distance_m=distance))
    require(len(distances) == 100, "Guard proof must cover all 100 active measured requests")
    return dict(schema="sircus-setup-guard-restoration-applicability-v1", status="passed", audit_source_sha256=digest(__file__),
        measured_preflight_source_commit=GUARD_PREFLIGHT_COMMIT, measured_preflight_source_archive_sha256=GUARD_PREFLIGHT_ARCHIVE_SHA256,
        measured_setup_sha256=measured["source_setup_sha256"], measured_raw_sha256=measured["source_raw_sha256"],
        prior_independent_audit_sha256=digest(prior_setup_audit), launch_sha256=digest(launch_path), supervisor_sha256=digest(status_path),
        source_trial_sha256=measured["source_trial_sha256"], geometry_fingerprint=measured["geometry_fingerprint"],
        old_source_sha256=old_hashes, new_source_sha256=new_hashes, controller_ast=ast_receipt, home_eligibility=invariant,
        checked_active_requests=100, observed_workcell_count=35, ball_releases=0, resimulation=False,
        recomputed_distance_min_m=min(r["recomputed_distance_m"] for r in distances),
        recomputed_distance_max_m=max(r["recomputed_distance_m"] for r in distances), distances=distances,
        scope="Bounded source-inertness proof for exactly the previously measured 100 requests. Home eligibility is a SOURCE INVARIANT, not an observed new flag. Preflight remains producer5e; new guards are false on these inputs. No resimulation, new proposal validation, capture measurement or new producer execution is claimed.")


def audit(run, executed_source=None, *, first_update_only=False):
    return verify_first_update(run,executed_source) if first_update_only else verify_checkpoint_chain(run, executed_source)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--executed-source", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--first-update-only", action="store_true")
    parser.add_argument("--setup-check-source", type=Path,
                        help="Audit setup-check files against this frozen prior guided run, without policy/result requirements")
    parser.add_argument("--guard-source-old", type=Path)
    parser.add_argument("--guard-source-new", type=Path)
    parser.add_argument("--prior-setup-audit", type=Path)
    parser.add_argument("--guard-old-archive", type=Path)
    parser.add_argument("--compare-original", type=Path)
    parser.add_argument("--compare-guided", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Choose a new audit output; old evidence is preserved")
    require(bool(args.compare_original) == bool(args.compare_guided), "Supply both prior runs for three-method comparison")
    require(not args.first_update_only or not args.compare_original, "Partial pilot cannot produce final comparison")
    require(not args.setup_check_source or not (args.first_update_only or args.compare_original or args.compare_guided),
            "Setup-only audit cannot produce policy/full-study comparisons")
    require(bool(args.guard_source_old) == bool(args.guard_source_new), "Supply both guard source snapshots")
    if args.guard_source_old:
        require(not (args.first_update_only or args.compare_original or args.compare_guided), "Guard applicability proof is separate from policy/full-study audits")
        result = audit_guard_restoration_applicability(args.run, args.guard_source_old, args.guard_source_new,
            original_run=args.setup_check_source, prior_setup_audit=args.prior_setup_audit, old_archive=args.guard_old_archive)
    elif args.setup_check_source:
        result = audit_setup_check(args.run, args.setup_check_source)
    else:
        result = audit(args.run, args.executed_source, first_update_only=args.first_update_only)
    if args.compare_original:
        result["comparison"] = compare_runs(args.compare_original, args.compare_guided, args.run)
    with args.output.open("x") as file:
        json.dump(result, file, allow_nan=False)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
