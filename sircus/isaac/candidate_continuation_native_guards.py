"""Read-only native guard reconstruction; never starts IK or a simulator.

This verifies saved request geometry, measured setup windows and correction
commands. The inherited path check remains a sampled capsule proxy. It does not
certify mesh collision freedom or unrecorded motion between readbacks.
"""

import copy

import numpy as np

from candidate_native_audit import verify_initial_geometry, same_path
from continuous_feasibility import approximate_path_check, canonical_hash
from diverse_pilot_audit import verify_settling_history
from expanded_pose_menu import endpoint_targets
from feasibility_contract import dynamic_cell
from prepare_assets import matrix
from settled_transit import protocol, quiet, release_ready
from settling_contract import correction
from shot_benchmark_audit import verify_home
from shot_benchmark_reach_audit import fit_details, geometry_record

ARMS = ("Arm10", "Arm14")


def require(value, message):
    if not value:
        raise ValueError(message)


def _same_vector(actual, expected, label, tolerance=1e-6):
    a, b = np.asarray(actual, dtype=float), np.asarray(expected, dtype=float)
    require(a.shape == b.shape and np.isfinite(a).all() and np.isfinite(b).all()
            and np.allclose(a, b, rtol=0, atol=tolerance), label)


def _verify_preflight_records(packet, scene, cells, seeds):
    records = packet.get("preflight_records")
    require(isinstance(records, list) and len(records) == 35,
            "Missing complete native preflight records")
    for i, (cell, action, row, use, record) in enumerate(zip(
            cells, packet["actions"], packet["rows"], packet["enabled"], records)):
        if not use:
            require(record is None, "Inactive slot fabricated a preflight")
            continue
        require(isinstance(record, dict), "Missing active native preflight")
        unsigned = copy.deepcopy(record)
        fingerprint = unsigned.pop("fingerprint", None)
        require(fingerprint == canonical_hash(unsigned), "Native preflight fingerprint changed")
        require(record.get("action") == action
                and record.get("qualification") == row["qualification"]
                and same_path(record.get("path"), row.get("path_proxy"))
                and record.get("release_authorized") is False
                and record.get("bucket_impossibility_proven") is False,
                "Native row differs from its preflight")
        binding = record.get("binding", {})
        expected_starts = {n: packet["home"]["measured_joint_rad"][n][i] for n in ARMS}
        expected_seeds = expected_starts if seeds[i] is None else {n: seeds[i][n]["q"] for n in ARMS}
        require(binding.get("context") == dict(id=cell["id"], bucket_xy_m=cell["bucket_xy_m"])
                and binding.get("action") == action
                and binding.get("measured_starts_rad") == expected_starts
                and binding.get("ik_seeds_rad") == expected_seeds
                and binding.get("path_samples") == 16,
                "Native preflight context/start/seed binding changed")
        targets = endpoint_targets(scene, cell, action)
        require(record.get("targets") == {n: np.asarray(v).tolist() for n, v in targets.items()},
                "Native preflight endpoint targets changed")
        admitted = bool(row["qualification"]["accepted"] and row.get("path_proxy")
                        and row["path_proxy"]["accepted"])
        require(type(record.get("accepted")) is bool and record["accepted"] == admitted,
                "Native preflight admission changed")


def _verify_corrections(packet, scene, cells, links):
    """Reconstruct each saved feedback target and path without solving IK."""
    base = matrix([scene["arms"]["base_spacing_m"], 0, 0], [0, 0, np.pi])
    mount = matrix([0, 0, scene["board"]["tool_offset_m"]], [0, 0, scene["board"]["mount_yaw_rad"]])
    by_id = {cell["id"]: i for i, cell in enumerate(cells)}
    corrections = packet["correction_history"]
    order = [(c["iteration"], by_id[c["cell_id"]]) for c in corrections]
    require(order == sorted(order) and len(set(order)) == len(order),
            "Correction application order changed")
    fit_count = path_count = 0
    for i, (cell, action, row, enabled) in enumerate(zip(
            cells, packet["actions"], packet["rows"], packet["enabled"])):
        eligible = bool(enabled and row["qualification"]["accepted"]
                        and row.get("path_proxy") and row["path_proxy"]["accepted"]
                        and not row.get("path_rejection"))
        targets = endpoint_targets(scene, cell, action)
        board_target = base @ targets["Arm14"] @ mount
        board_command, drop_command = board_target.copy(), targets["Arm10"].copy()
        commanded = ({n: np.asarray(row["qualification"]["ik"][n]["q"], dtype=float) for n in ARMS}
                     if eligible else {n: np.asarray(packet["home"]["measured_joint_rad"][n][i]) for n in ARMS})
        for name in ARMS:
            _same_vector(packet["transit_history"][-1]["state"][name]["target_rad"][i], commanded[name],
                         "Restored transit target differs from initial qualified/parked target")
        local = {c["iteration"]: c for c in corrections if c["cell_id"] == cell["id"]}
        require(eligible or not local, "Correction applied to initially ineligible request")
        failed = False
        for iteration in sorted({m["iteration"] for m in packet["settling_history"][i]}):
            history = [m for m in packet["settling_history"][i] if m["iteration"] == iteration]
            for metric in history:
                for name in ARMS:
                    _same_vector(metric["arms"][name]["target_rad"], commanded[name],
                                 "Measured hold target changed without a qualified correction")
            if iteration not in local:
                continue
            require(not failed and iteration < 3, "Correction followed a failed or final correction stage")
            entry = local[iteration]
            metric = entry["measured"]
            commands = {}
            if "Arm14" in entry["arms"]:
                board_command = correction(board_command, board_target,
                    np.asarray(metric["board_center_m"]), np.asarray(metric["board_normal"]))
                commands["Arm14"] = np.linalg.inv(base) @ board_command @ np.linalg.inv(mount)
            if "Arm10" in entry["arms"]:
                drop_command[:3, 3] += targets["Arm10"][:3, 3] - np.asarray(metric["dropper_flange_m"])
                commands["Arm10"] = drop_command.copy()
            require(list(commands) == entry["arms"] and commands,
                    "Correction arm order or empty correction changed")
            solved = {n: np.asarray(metric["arms"][n]["target_rad"]) for n in ARMS}
            failed_ik = False
            expected_fit_names = []
            for name, target in commands.items():
                expected_fit_names.append(name)
                require(name in entry["fits"], "Missing correction saved IK fit")
                fit = entry["fits"][name]
                fit_details(target, fit, links[name], np.eye(4), name == "Arm14")
                fit_count += 1
                if not fit["solved"]:
                    failed_ik = True
                    break
                solved[name] = np.asarray(fit["q"], dtype=float)
            require(list(entry["fits"]) == expected_fit_names, "Correction IK gate order changed")
            if failed_ik:
                require(entry["invalid"] == "settling_correction_ik", "Failed correction IK was admitted")
                failed = True
            else:
                starts = {n: np.asarray(metric["arms"][n]["position_rad"]) for n in ARMS}
                path = approximate_path_check(scene, links, starts, solved, cell=cell, samples=16)
                path_count += 1
                require(entry["invalid"] == (None if path["accepted"] else "settling_correction_path_proxy"),
                        "Correction rejection differs from independently rebuilt path")
                if path["accepted"]:
                    commanded = {n: solved[n].copy() for n in ARMS}
                else:
                    failed = True
            require(not failed or row["invalid"] == entry["invalid"], "Failed correction rejection changed or was released")
            require(any(m["iteration"] == iteration + 1 for m in packet["settling_history"][i]),
                    "Correction has no following measured hold stage")
    return dict(saved_ik_fits_recomputed=fit_count, sampled_paths_recomputed=path_count,
                command_target_continuity_verified=True,
                scope="Saved FK, accumulated Cartesian feedback and 16-sample path proxies; no IK optimization or mesh collision certificate")



def _verify_rejections(packet, cells):
    """Rebuild first-cause declines instead of trusting a rejection label."""
    for i, (cell, action, row, enabled) in enumerate(zip(
            cells, packet["actions"], packet["rows"], packet["enabled"])):
        qualification, path = row["qualification"], row.get("path_proxy")
        if not enabled:
            reason = "inactive_split"
        elif not qualification["accepted"]:
            reason = qualification["reason"]
        elif not path["accepted"]:
            reason = path["reasons"][0]
        else:
            failed = [c for c in packet["correction_history"]
                      if c["cell_id"] == cell["id"] and c.get("invalid")]
            if failed:
                reason = failed[0]["invalid"]
            else:
                metric = packet["settling_history"][i][-1]
                if release_ready(metric):
                    reason = "board_bucket_separation" if dynamic_cell(cell, action)["distance_m"] < .28 else None
                else:
                    require(metric["iteration"] == 3, "Unfinished settling was reported as a terminal decline")
                    lag = max(abs(p-t) for state in metric["arms"].values()
                              for p, t in zip(state["position_rad"], state["target_rad"]))
                    reason = ("setup_motion_not_quiet" if not quiet(metric) else
                              "setup_transit_incomplete" if lag > .02 else "endpoints_not_settled")
        require(row["invalid"] == reason, "Native rejection first cause changed")
    return dict(first_cause_rejections_reconstructed=True)


def verify_guards(packet, scene, cells, links, seeds):
    """Verify one 35-slot packet after the caller binds its exact reservation.

    ``scene``, ``cells``, ``links`` and ``seeds`` must come from independently
    checked inputs. This routine does not authenticate files or source hashes.
    """
    require(isinstance(cells, list) and len(cells) == 35
            and len({c["id"] for c in cells}) == 35
            and packet.get("cells") == cells and set(links) == set(ARMS),
            "Native guard workcell/arm association changed")
    require(all(isinstance(packet.get(k), list) and len(packet[k]) == 35
                for k in ("actions", "rows", "enabled"))
            and all(type(v) is bool for v in packet["enabled"])
            and isinstance(seeds, list) and len(seeds) == 35
            and packet.get("proposal_ik_seeds") == seeds,
            "Native guard row/action/seed coverage changed")
    require(all(row.get("invalid") is None or
                isinstance(row.get("invalid"), str) and row["invalid"] for row in packet["rows"]),
            "Invalid native rejection reason")
    verify_home(packet)
    canonical = packet.get("native_binding", {}).get("canonical_home_rad", {})
    require(set(canonical) == set(ARMS), "Missing bound canonical home targets")
    for name in ARMS:
        require(np.asarray(canonical[name]).shape == (6,), "Canonical home target shape changed")
        _same_vector(packet["home"]["target_joint_rad"][name], np.tile(canonical[name], (35, 1)),
                     "Home target differs from bound canonical reset")
        start = packet["transit_history"][0]["state"][name]
        _same_vector(start["position_rad"], packet["home"]["measured_joint_rad"][name],
                     "Driven setup start differs from measured episode home", tolerance=0)
        _same_vector(start["velocity_rad_s"], packet["home"]["measured_joint_velocity_rad_s"][name],
                     "Driven setup velocity differs from measured episode home", tolerance=0)
    _verify_preflight_records(packet, scene, cells, seeds)
    geometry = verify_initial_geometry(packet, scene, cells, links, seeds)
    for cell, action, row, enabled in zip(cells, packet["actions"], packet["rows"], packet["enabled"]):
        if enabled:
            geometry_record(scene, cell, action, links, row["qualification"])
    settlement = verify_settling_history(packet, scene, cells, dict(settling_protocol=protocol()))
    corrections = _verify_corrections(packet, scene, cells, links)
    rejections = _verify_rejections(packet, cells)
    return dict(home=dict(canonical_target_verified=True, measured_start_association_verified=True),
                initial_geometry=geometry, settlement=settlement, corrections=corrections, rejections=rejections,
                scope="Six rotary-axis positions/targets and all 14 articulation velocity channels verified from saved evidence; finger positions, between-readback motion, exact mesh collisions and physical execution are not certified")
