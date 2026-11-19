"""Independent CPU evidence audit for stable IK and ten-target native search.

This checks saved evidence and endpoint geometry. It neither starts a GPU job nor
replays dynamics, trains a policy, changes an existing study, or deploys a model.
"""

import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np

from .verify_expansion import Audit, digest, sha256, statistics

VERSION = "sircus-stable-ik-evidence-audit-v1"
STUDY_VERSION = "sircus-stable-ik-ten-target-study-v1"
STABLE_MODES = ("stable_matched", "stable_free", "stable_seeded")
CONTROL_MODES = ("legacy_fresh", "saved", "stable_matched", "stable_free", "stable_seeded")
EXPECTED_INDICES = (8, 9, 10, 12, 21, 23, 26, 35, 39, 41)
JOINT_TOLERANCE = 1e-9


def read(path):
    return json.loads(Path(path).read_text())


def lines(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def _fingerprint(value, key):
    payload = dict(value)
    recorded = payload.pop(key)
    return digest(payload) == recorded


def _stats(audit, name, actual, rows):
    expected = statistics(rows)
    audit.check(name, all(k in actual and np.isclose(actual[k], v, atol=1e-9, rtol=1e-9)
        for k, v in expected.items()), dict(actual=actual, expected=expected))


def captured_rank(row):
    """Independent reconstruction of the captured-source acquisition branch."""
    margin = float(row.get("minimum_mouth_clearance_mm") or 0)
    return (100. + 2 * (row["route_quality"] == 2) + min(margin, 50.) / 50., row["id"])


def audit_strict_endpoints(audit, assets, spec, rows):
    """Independent NumPy full-pose residuals plus direct native static contacts."""
    import mujoco
    from scipy.spatial.transform import Rotation
    from .learning_ik import read_chain, shot_targets
    from .model import build_scene

    assets = Path(assets)
    chains = {key: read_chain(assets / filename) for key, filename in
              (("q10", "dropper.urdf"), ("q14", "board_arm.urdf"))}
    model = mujoco.MjModel.from_xml_string(build_scene(spec["scene"], assets, solref=(.004, .15), contact_friction=.35))
    data = mujoco.MjData(model)
    indices = {key: np.asarray([model.jnt_qposadr[model.joint(f"{role}/joint{i}").id] for i in range(1, 7)])
               for key, role in (("q10", "Arm10"), ("q14", "Arm14"))}
    fingers = [model.jnt_qposadr[model.joint(f"Arm10/finger_{side}_joint").id] for side in ("left", "right")]
    ball = int(model.jnt_qposadr[model.joint("ball/free").id])
    mocap = int(model.body_mocapid[model.body("bucket").id])
    geoms = np.asarray([i for i in range(model.ngeom) if
        (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i) or "").startswith("bucket/")])
    math_bad, contact_bad, collider_bad = [], [], []
    maxima = dict(position_m=0., weighted_pose_residual=0., collider_error_m=0.)
    for row in rows:
        if not row["native_release"]:
            continue
        value = np.asarray(row["values"])
        targets = shot_targets(spec["scene"], [row["bucket_xy_m"]], [value[:3]], [value[3]], [value[4]])
        for key, target_key in (("q10", "dropper_flange_world"), ("q14", "board_flange_local")):
            origins, axes, lower, upper = chains[key]
            q = np.asarray(row[key])
            pose = np.eye(4)
            for origin, axis, angle in zip(origins, axes, q):
                rotation = np.eye(4); rotation[:3, :3] = Rotation.from_rotvec(axis * angle).as_matrix()
                pose = pose @ origin @ rotation
            target = targets[target_key][0]
            distance = float(np.linalg.norm(pose[:3, 3] - target[:3, 3]))
            angle = float(Rotation.from_matrix(target[:3, :3] @ pose[:3, :3].T).magnitude())
            residual = float(np.hypot(distance, .25 * angle))
            maxima["position_m"] = max(maxima["position_m"], distance)
            maxima["weighted_pose_residual"] = max(maxima["weighted_pose_residual"], residual)
            if not np.isfinite(q).all() or (q < lower - 1e-9).any() or (q > upper + 1e-9).any() or distance >= 1e-4 + 1e-10 or residual >= 1e-4 + 1e-10:
                math_bad.append((row["id"], key))
        data.qpos[:] = model.qpos0
        for key in indices:
            data.qpos[indices[key]] = row[key]
        data.qpos[fingers] = .025
        data.qpos[ball:ball + 3] = [0, 0, 2]
        data.qpos[ball + 3:ball + 7] = [1, 0, 0, 0]
        data.qvel[:] = 0
        data.mocap_pos[mocap] = [*row["bucket_xy_m"], 0]
        data.mocap_quat[mocap] = [1, 0, 0, 0]
        mujoco.mj_forward(model, data)
        collider = float(np.linalg.norm(data.geom_xpos[geoms] - (np.r_[row["bucket_xy_m"], 0] + model.geom_pos[geoms]), axis=1).max())
        maxima["collider_error_m"] = max(maxima["collider_error_m"], collider)
        if collider > 1e-8:
            collider_bad.append(row["id"])
        for contact in data.contact[:data.ncon]:
            names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(g)) or "" for g in contact.geom]
            if contact.dist < -.002 and not any(name.startswith("ball/") for name in names):
                contact_bad.append((row["id"], names)); break
    audit.check("independent_original_full_pose_tolerance", not math_bad, dict(maxima=maxima, failures=math_bad[:10]))
    audit.check("independent_native_endpoint_contacts", not contact_bad, contact_bad[:10])
    audit.check("independent_bucket_collision_transforms", not collider_bad, collider_bad[:10])
    return maxima


def audit_solver_arrays(audit, arrays, queries, claimed):
    """Recompute invariance/preservation from arrays, not runner pass flags."""
    n = len(queries)
    base_shape = arrays.get("bucket", np.empty(0)).shape == (n, 2) and arrays.get("values", np.empty(0)).shape == (n, 5)
    audit.check("solver_query_shapes", base_shape and n > 0)
    if not base_shape:
        return
    audit.check("solver_query_geometry", np.array_equal(arrays["bucket"], [q["bucket_xy_m"] for q in queries])
        and np.array_equal(arrays["values"], [q["values"] for q in queries]))
    order = np.asarray(arrays.get("reverse_order", []), dtype=int)
    good_order = np.array_equal(np.sort(order), np.arange(n))
    audit.check("solver_permutation_is_bijection", good_order)
    if not good_order:
        return
    inverse = np.argsort(order)
    comparisons = {r["method"]: r for r in claimed.get("comparison", [])}
    passed = []
    for mode in ("old",) + STABLE_MODES:
        names = [mode + suffix + "_" + key for suffix in ("", "_reverse", "_partition")
                 for key in ("mask", "reasons", "q10", "q14")]
        missing = [name for name in names if name not in arrays]
        audit.check(mode + "_audit_variants_present", not missing, missing)
        if missing:
            continue
        mask = np.asarray(arrays[mode + "_mask"], dtype=bool)
        reasons = arrays[mode + "_reasons"]
        shape = mask.shape == reasons.shape == (n,) and all(arrays[mode + "_" + k].shape == (n, 6) for k in ("q10", "q14"))
        audit.check(mode + "_variant_shapes", shape)
        if not shape:
            continue
        audit.check(mode + "_mask_reason_identity", np.array_equal(mask, reasons == 0))
        deltas, changes = [], []
        for suffix, index in (("_reverse", inverse), ("_partition", np.arange(n))):
            other_mask = arrays[mode + suffix + "_mask"][index].astype(bool)
            other_reasons = arrays[mode + suffix + "_reasons"][index]
            accepted = mask & other_mask
            delta = max(float(np.max(np.abs(arrays[mode + "_" + key][accepted] -
                arrays[mode + suffix + "_" + key][index][accepted]))) if accepted.any() else 0.
                for key in ("q10", "q14"))
            valid = np.array_equal(mask, other_mask) and np.array_equal(reasons, other_reasons) and delta <= JOINT_TOLERANCE
            changes.append(int((mask != other_mask).sum())); deltas.append(delta)
            if mode != "old":
                audit.check(mode + suffix + "_invariance", valid,
                    dict(mask_changes=changes[-1], max_q_delta_rad=delta))
                passed.append(valid)
            for key in ("branch10", "branch14", "query10_sha256", "query14_sha256"):
                name = mode + "_" + key
                if mode != "old":
                    audit.check(mode + suffix + "_" + key,
                        name in arrays and mode + suffix + "_" + key in arrays
                        and np.array_equal(arrays[name], arrays[mode + suffix + "_" + key][index]))
        row = comparisons.get(mode, {})
        audit.check(mode + "_reported_counts", row.get("queries") == n and row.get("accepted") == int(mask.sum())
            and row.get("order_mask_changes") == changes[0] and row.get("batch_mask_changes") == changes[1]
            and np.isclose(row.get("max_q_delta_rad", float("inf")), max(deltas), atol=1e-12, rtol=0))
    known = np.asarray([q.get("known_valid") is True for q in queries])
    hints = all(name in arrays and arrays[name].shape == (n, 6) for name in ("known_q10", "known_q14"))
    audit.check("known_joint_shapes", hints)
    preserved = np.zeros(n, dtype=bool)
    if hints and "stable_seeded_mask" in arrays:
        preserved = arrays["stable_seeded_mask"].astype(bool).copy()
        for key in ("q10", "q14"):
            audit.check("known_" + key + "_identity", np.array_equal(arrays["known_" + key], [q[key] for q in queries]))
            preserved &= (arrays["stable_seeded_" + key] == arrays["known_" + key]).all(1)
    audit.check("known_valid_joint_pairs_preserved_exactly", known.any() and preserved[known].all())
    audit.check("saved_preservation_flags_truthful", "seeded_preserved" in arrays and np.array_equal(arrays["seeded_preserved"], preserved))
    claim = claimed.get("known_branch_preservation", {})
    audit.check("reported_preservation_counts", claim.get("known_valid_queries") == int(known.sum())
        and claim.get("preserved") == int(preserved[known].sum()) and claim.get("passed") == bool(preserved[known].all()))
    audit.check("reported_stability_pass", bool(passed) and all(passed)
        and claimed.get("order_batch_invariance", {}).get("passed") is True
        and claimed.get("order_batch_invariance", {}).get("tolerance_rad", JOINT_TOLERANCE) <= JOINT_TOLERANCE)


def audit_records(spec, report, rows, bank, original):
    audit = Audit()
    audit.check("study_version_and_no_training", spec.get("version") == STUDY_VERSION
        and spec.get("no_training") is True and report.get("no_training") is True)
    audit.check("contract_fingerprint", _fingerprint(spec, "fingerprint") and report.get("contract_fingerprint") == spec["fingerprint"])
    audit.check("completed", report.get("status") == report.get("phase") == "completed")
    audit.check("base_bank_fingerprint", _fingerprint(original, "sha256") and original["sha256"] == spec["base_bank_sha256"])
    audit.check("confirmed_bank_fingerprint", _fingerprint(bank, "sha256") and bank["sha256"] == report["proposal_bank_sha256"])
    n = len(original["templates"])
    audit.check("unchanged_base_template_prefix", bank["templates"][:n] == original["templates"])
    audit.check("bank_contract_association", bank.get("parent_bank_sha256") == original["sha256"]
        and bank.get("stable_ik_contract_fingerprint") == spec["fingerprint"])
    targets = {t["id"]: t for t in spec["development_layouts"]}
    full_targets = {t["id"]: t for t in spec["original_development_layouts"]}
    audit.check("fixed_ten_development_targets", tuple(t["original_test_index"] for t in spec["development_layouts"]) == EXPECTED_INDICES
        and len(targets) == 10 and all(t["role"] == "targeted_development" for t in targets.values()))
    audit.check("confirmation_protocol_unchanged", spec["confirmations_per_candidate"] == 10
        and spec["final_confirmations_per_candidate"] == 20 and spec["minimum_confirmation_rate"] == .8)
    audit.check("sequential_trial_ids", [r["id"] for r in rows] == [f"stable-shot-{i:07d}" for i in range(len(rows))])
    audit.check("record_count", report["records"] == len(rows))
    phases = Counter(r["phase"] for r in rows)
    allowed = {"search", "confirmation", "final_confirmation"} | {"control_" + m for m in CONTROL_MODES}
    audit.check("only_declared_phases", set(phases) <= allowed)
    for phase, key in (("search", "native_search_attempts"), ("confirmation", "confirmation_attempts"),
        ("final_confirmation", "final_confirmation_attempts")):
        audit.check(phase + "_attempt_count", report[key] == phases[phase])
    audit.check("control_attempt_count", report["control_attempts"] == sum(v for k, v in phases.items() if k.startswith("control_")))
    for mode in CONTROL_MODES:
        controls = [r for r in rows if r["phase"] == "control_" + mode]
        expected_controls = Counter({sid: spec["control_repeats"] for sid in spec["control_source_ids"]})
        audit.check(mode + "_control_denominators", Counter(r.get("control_source_id") for r in controls) == expected_controls)
        if "control_statistics" in report:
            _stats(audit, mode + "_control_statistics", report["control_statistics"].get(mode, {}), controls)
    _stats(audit, "search_statistics", report["native_search_statistics"], [r for r in rows if r["phase"] == "search"])
    _stats(audit, "confirmation_statistics", report["confirmation_statistics"], [r for r in rows if r["phase"] == "confirmation"])
    _stats(audit, "final_confirmation_statistics", report["final_confirmation_statistics"], [r for r in rows if r["phase"] == "final_confirmation"])
    counts = Counter(r["target_id"] for r in rows if r["phase"] == "search")
    audit.check("bounded_ten_target_native_budget", set(counts) <= set(targets)
        and all(x <= spec["native_per_target"] for x in counts.values()))
    bad_targets, bad_bounds, bad_scores, bad_poses, bad_scalar, bad_timing = [], [], [], [], [], []
    low, high = np.asarray(spec["action_bounds"]["low"]), np.asarray(spec["action_bounds"]["high"])
    for row in rows:
        expected = full_targets if row["phase"].startswith("control_") else targets
        target = expected.get(row["target_id"])
        if target is None or not np.array_equal(row["bucket_xy_m"], target["bucket_xy_m"]) or row.get("source_phase") != "targeted_development":
            bad_targets.append(row["id"])
        value = np.asarray(row.get("values", []))
        if value.shape != (5,) or not np.isfinite(value).all() or ((value < low - 1e-9) | (value > high + 1e-9)).any():
            bad_bounds.append(row["id"])
        if row["route_success"] and (row["invalid"] or not row["board_bounce"] or not row["whole_ball_entry"]
            or row["max_retention_s"] < 1 - 1e-9 or row["external_contact_steps"] or row["board_contact_episodes"] < 1):
            bad_scores.append(row["id"])
        if row["route_quality"] not in (0, 1, 2) or (row["route_quality"] and not row["route_success"]):
            bad_scores.append(row["id"])
        if not row["invalid"]:
            gate = row.get("pose_gate", {})
            if (gate.get("board_position_error_mm", float("inf")) >= 2 or gate.get("dropper_error_mm", float("inf")) >= 2
                or gate.get("board_normal_error_deg", float("inf")) >= .5 or row.get("endpoint_collision_pairs")
                or not row.get("native_release") or not row.get("geometry_eligible")):
                bad_poses.append(row["id"])
            if row["phase"].startswith("control_") and row.get("scalar_score_audit_passed") is not True:
                bad_scalar.append(row["id"])
        if row.get("first_contact_s") is not None:
            times = row["first_contact_s"]
            external = [times[k] for k in ("table", "floor", "robot", "unattributed") if times.get(k) is not None]
            earliest = min(external) if external else None
            bucket = times.get("bucket")
            before = bucket is not None and (earliest is None or bucket < earliest)
            distance = row.get("airborne_closest_opening_distance_m")
            if (row.get("first_external_contact_s") != earliest
                or bool(row["external_contact_steps"]) != bool(external)
                or row.get("bucket_contact_before_external") != before
                or any(v is not None and (not np.isfinite(v) or v <= 0 or v > row["physics_steps"] / 240 + 1e-12) for v in times.values())
                or (distance is not None and (not np.isfinite(distance) or distance < 0))):
                bad_timing.append(row["id"])
    audit.check("declared_development_coordinates", not bad_targets, bad_targets[:10])
    audit.check("explicit_action_envelope", not bad_bounds, bad_bounds[:10])
    audit.check("unchanged_direct_capture_invariants", not bad_scores, bad_scores[:10])
    audit.check("accepted_native_pose_and_collision_gates", not bad_poses, bad_poses[:10])
    audit.check("released_controls_pass_scalar_scoring", not bad_scalar, bad_scalar[:10])
    audit.check("pre_external_flight_diagnostics_consistent", not bad_timing, bad_timing[:10])
    audit.check("native_release_accounting", all(bool(r["native_release"]) ==
        (bool(r["geometry_eligible"]) and not bool(r["invalid"])) for r in rows))
    lookup = {r["id"]: r for r in rows}
    first, final, association_bad = {}, {}, []
    for row in rows:
        if row["phase"] not in ("confirmation", "final_confirmation"):
            continue
        source = lookup.get(row.get("confirmation_source_id"))
        if source is None or source["phase"] != "search" or source["target_id"] != row["target_id"] or source["invalid"] or not source["route_success"]:
            association_bad.append(row["id"])
        (first if row["phase"] == "confirmation" else final).setdefault(row.get("confirmation_source_id"), []).append(row)
    audit.check("confirmation_source_associations", not association_bad, association_bad[:10])
    audit.check("first_confirmation_denominators", all(len(rr) == 10 for rr in first.values()))
    audit.check("final_confirmation_denominators", all(len(rr) == 20 for rr in final.values()))
    audit.check("only_screened_sources_receive_final_confirmations", all(sid in first and statistics(first[sid])["captures"] >= 8 for sid in final))
    expected_screen, expected_final = [], []
    for tid in targets:
        sources = sorted([r for r in rows if r["phase"] == "search" and r["target_id"] == tid
            and r["route_success"] and not r["invalid"]], key=captured_rank, reverse=True)[:spec["top_candidates"]]
        expected_screen.extend(r["id"] for r in sources)
        screened = [r for r in sources if r["id"] in first and len(first[r["id"]]) == 10
            and statistics(first[r["id"]])["captures"] >= 8]
        if screened:
            chosen = max(screened, key=lambda r: (statistics(first[r["id"]])["captures"], *captured_rank(r)))
            expected_final.append(chosen["id"])
    audit.check("all_selected_sources_receive_screening", set(first) == set(expected_screen))
    audit.check("final_sources_selected_without_final_outcomes", set(final) == set(expected_final))
    phase_ids = {phase: [i for i, r in enumerate(rows) if r["phase"] == phase]
                 for phase in ("search", "confirmation", "final_confirmation")}
    audit.check("recorded_search_screen_final_order", all(not phase_ids[a] or not phase_ids[b]
        or max(phase_ids[a]) < min(phase_ids[b]) for a, b in (("search", "confirmation"), ("confirmation", "final_confirmation"))))
    promoted = bank["templates"][n:]
    audit.check("one_promotion_per_target", len({p["source_target_id"] for p in promoted}) == len(promoted))
    audit.check("promotion_counts", report["additions"] == len(promoted)
        and report["base_templates"] == n and report["confirmed_templates"] == n + len(promoted))
    qualified_ids = {sid for sid, rr in final.items() if statistics(rr)["captures"] >= 16}
    audit.check("all_and_only_final_qualified_sources_promoted", {p["source_trial_id"] for p in promoted} == qualified_ids)
    for template in promoted:
        sid = template["source_trial_id"]
        source = lookup.get(sid)
        valid = source is not None and sid in qualified_ids
        if valid:
            value, xy = np.asarray(source["values"]), np.asarray(source["bucket_xy_m"])
            valid = (template["source_target_id"] == source["target_id"] and template["source_phase"] == "targeted_development"
                and template.get("source_record_sha256") == digest(source)
                and np.array_equal(template["relative_board_xy_m"], value[:2] - xy)
                and template["height_m"] == value[2] and template["tilt_deg"] == value[3] and template["gap_m"] == value[4])
            screen = template.get("first_confirmation", {})
            final_counts = template.get("final_confirmation", template.get("confirmation", {}))
            valid = valid and screen.get("attempts") == 10 and screen.get("captures") == statistics(first[sid])["captures"]
            valid = valid and final_counts.get("attempts") == 20 and final_counts.get("captures") == statistics(final[sid])["captures"]
            valid = valid and final_counts.get("invalid") == statistics(final[sid])["invalid"]
            nominal = template.get("nominal_source", {})
            valid = valid and nominal.get("requested_values") == source["values"] and all(nominal.get(k) == source.get(k) for k in (
                "bucket_xy_m", "q10", "q14", "branch10", "branch14", "geometry_cache_file", "geometry_cache_row"))
        audit.check(template["source_target_id"] + "_native_promotion_provenance", valid)
    cells = {row["id"]: row for row in report["cells"]}
    audit.check("all_ten_cells_reported", set(cells) == set(targets))
    by_target = {t["source_target_id"]: t for t in promoted}
    for tid in targets:
        if tid not in cells:
            continue
        rr = [r for r in rows if r["phase"] == "search" and r["target_id"] == tid]
        _stats(audit, tid + "_search_statistics", cells[tid]["search"], rr)
        assisted = sum(bool(r["whole_ball_entry"]) and bool(r["external_contact_steps"]) for r in rr)
        audit.check(tid + "_assisted_entries_separate", cells[tid]["observed_assisted_entries"] == assisted)
        audit.check(tid + "_promotion_identity", cells[tid].get("promotion") == by_target.get(tid))
    return audit


def audit_endpoint_caches(audit, rows, caches, prior_sources, solver, spec):
    """Audit emitted joints, measured hint provenance and non-absorbed jitter."""
    lookup = {r["id"]: r for r in prior_sources + rows}
    fresh_ids = {r["id"] for r in rows}
    bad_cache, bad_hints, bad_continuation, bad_controls = [], [], [], []
    for row in rows:
        name, i = row.get("geometry_cache_file"), row.get("geometry_cache_row")
        data = caches.get(name)
        valid = data is not None and isinstance(i, int) and 0 <= i < len(data["eligible"])
        if not valid:
            bad_cache.append(row["id"]); continue
        valid = np.array_equal(data["bucket"][i], row["bucket_xy_m"]) and np.array_equal(data["values"][i], row["values"])
        valid = valid and bool(data["eligible"][i]) == row["geometry_eligible"]
        if row["native_release"]:
            valid = valid and data["eligible"][i] and all(np.array_equal(data[k][i], row[k]) for k in ("q10", "q14"))
        for key in ("branch10", "branch14", "known_hint_used", "known_pair_preserved", "known_request_match", "preserved_exact_nominal", "refined_continuation"):
            valid = valid and key in data and data[key][i] == row.get(key)
        if not valid:
            bad_cache.append(row["id"])
        source_id = row.get("confirmation_source_id", row.get("seed_source_id", row.get("control_source_id")))
        source = lookup.get(source_id)
        used = bool(row.get("known_hint_used"))
        control = row["phase"].startswith("control_")
        if source is not None:
            hints_good = all(np.array_equal(data["known_" + k][i], source[k]) for k in ("q10", "q14", "values"))
            hints_good = hints_good and np.array_equal(data["known_bucket_xy"][i], source["bucket_xy_m"])
            if not hints_good:
                bad_hints.append(row["id"])
            if "seed_source_record_sha256" in row and row["seed_source_record_sha256"] != digest(source):
                bad_hints.append(row["id"])
            if source_id in fresh_ids and row.get("seed_source_id") and int(source_id.rsplit("-", 1)[1]) >= int(row["id"].rsplit("-", 1)[1]):
                bad_hints.append(row["id"])
        elif used or source_id is not None:
            bad_hints.append(row["id"])
        elif not control and not all(np.isnan(data[key][i]).all() for key in ("known_q10", "known_q14", "known_values", "known_bucket_xy")):
            bad_hints.append(row["id"])
        match = used and source is not None and np.array_equal(row["values"], source["values"]) and np.array_equal(row["bucket_xy_m"], source["bucket_xy_m"])
        same_joints = source is not None and all(np.array_equal(data[k][i], source[k]) for k in ("q10", "q14"))
        expected_preserved = used and bool(data["eligible"][i]) and same_joints
        if (row.get("known_request_match") != match
            or row.get("known_pair_preserved") != expected_preserved
            or row.get("preserved_exact_nominal") != (expected_preserved and match)
            or (used and not match and row["native_release"] and same_joints)
            or (row.get("refined_continuation") and (not used or same_joints or row["branch10"] != 0 or row["branch14"] != 0))):
            bad_continuation.append(row["id"])
        if control and source is not None:
            index = spec["control_source_ids"].index(source_id)
            mode = row["phase"][len("control_"):]
            if mode == "saved":
                expected = source
                good = bool(data["eligible"][i])
            else:
                mode = "old" if mode == "legacy_fresh" else mode
                expected = {k: solver[mode + "_" + k][index] for k in ("q10", "q14")}
                good = bool(data["eligible"][i]) == bool(solver[mode + "_mask"][index])
            good = good and all(np.array_equal(data[k][i], expected[k]) for k in ("q10", "q14"))
            if not good:
                bad_controls.append(row["id"])
    audit.check("exact_cached_native_branches", not bad_cache, bad_cache[:10])
    audit.check("measured_continuation_hint_provenance", not bad_hints, bad_hints[:10])
    audit.check("nominal_preservation_and_jitter_not_absorbed", not bad_continuation, bad_continuation[:10])
    audit.check("native_controls_reuse_corresponding_audited_branches", not bad_controls, bad_controls[:10])


def verify(study, *, source=None, assets=None, prior_study=None, prior_expansion=None, parent_study=None):
    study = Path(study)
    spec, report = read(study / "contract.json"), read(study / "report.json")
    prior_study = Path(prior_study or spec["prior_study_directory"])
    prior_expansion = Path(prior_expansion or spec["prior_expansion_directory"])
    parent_study = Path(parent_study or spec["parent_study_directory"])
    original = read(prior_study / "proposal-bank.json")
    bank, rows = read(study / "confirmed-bank.json"), lines(study / "shots.jsonl")
    audit = audit_records(spec, report, rows, bank, original)
    prior_report, prior_contract = read(prior_study / "report.json"), read(prior_study / "contract.json")
    expansion = read(prior_expansion / "contract.json")
    geometry = prior_study / "geometry-expanded-development.npz"
    with np.load(geometry, allow_pickle=False) as previous:
        xy, mask = previous["bucket"], previous["mask"]
        selected = [t for t, m in zip(expansion["development_layouts"], mask) if not m.any()]
        audit.check("remaining_targets_derived_from_original_cache", len(xy) == 12
            and np.array_equal(xy, [t["bucket_xy_m"] for t in expansion["development_layouts"]])
            and selected == spec["development_layouts"])
    audit.check("prior_study_hashes_and_identity", prior_report["status"] == "completed"
        and sha256(prior_study / "report.json") == spec["prior_study_report_sha256"]
        and sha256(prior_study / "contract.json") == spec["prior_study_contract_sha256"]
        and sha256(geometry) == spec["prior_geometry_sha256"]
        and prior_report["artifacts_sha256"][geometry.name] == sha256(geometry)
        and prior_report["proposal_bank_sha256"] == original["sha256"]
        and _fingerprint(prior_contract, "fingerprint")
        and prior_report["contract_fingerprint"] == prior_contract["fingerprint"]
        and prior_contract["parent_checkpoint_sha256"] == spec["parent_checkpoint_sha256"]
        and prior_contract["parent_contract_fingerprint"] == spec["parent_contract_fingerprint"]
        and prior_contract["scene"] == spec["scene"])
    audit.check("preserved_prior_expansion_and_bank_files", spec["base_bank_file"] == "proposal-bank.json"
        and sha256(prior_study / "proposal-bank.json") == spec["base_bank_file_sha256"]
        and sha256(prior_expansion / "expanded-bank.json") == spec["prior_expansion_bank_sha256"]
        and read(prior_expansion / "expanded-bank.json") == original
        and _fingerprint(expansion, "fingerprint") and expansion["scene"] == spec["scene"]
        and read(prior_expansion / "report.json")["contract_fingerprint"] == expansion["fingerprint"])
    sources = lines(study / "prior-sources.jsonl")
    source_lookup = {r["id"]: r for r in sources}
    audit.check("selected_prior_sources", spec["selected_prior_source_ids"] == [r["id"] for r in sources]
        and spec["selected_prior_source_hashes"] == [digest(r) for r in sources])
    originals = {}
    for origin in spec["source_origins"]:
        directory = Path(origin["directory"])
        audit.check(directory.name + "_prior_files", sha256(directory / "report.json") == origin["report_sha256"]
            and sha256(directory / "contract.json") == origin["contract_sha256"]
            and sha256(directory / "shots.jsonl") == origin["shots_sha256"])
        originals[str(directory)] = {r["id"]: r for r in lines(directory / "shots.jsonl")}
    bad_sources = []
    for row in sources:
        origin = row["prior_origin"]
        original_row = originals.get(origin["study_directory"], {}).get(row["prior_record_id"])
        payload = dict(row); payload.pop("prior_record_id"); payload.pop("prior_origin"); payload["id"] = row["prior_record_id"]
        if original_row != payload or original_row is None or digest(original_row) != origin["record_sha256"]:
            bad_sources.append(row["id"])
    audit.check("exact_prior_source_records", not bad_sources, bad_sources[:10])
    queries = read(study / "audit-queries.json")
    audit.check("audit_query_sources", all(q["source_id"] in source_lookup
        and all(q[k] == source_lookup[q["source_id"]][k] for k in ("bucket_xy_m", "values", "q10", "q14")) for q in queries))
    with np.load(study / "solver-audit.npz", allow_pickle=False) as value:
        solver = {k: value[k].copy() for k in value.files}
    audit_solver_arrays(audit, solver, queries, report["solver_audit"])
    cache, bad_cache = {}, []
    for row in rows:
        name = row.get("geometry_cache_file", "")
        if Path(name).name != name or not name.startswith("endpoint-batch-") or not name.endswith(".npz"):
            bad_cache.append(row["id"]); continue
        if name not in cache:
            with np.load(study / name, allow_pickle=False) as value:
                cache[name] = {k: value[k].copy() for k in value.files}
    audit.check("cache_files_are_local_basenames", not bad_cache, bad_cache[:10])
    audit_endpoint_caches(audit, rows, cache, sources, solver, spec)
    frozen = read(study / "frozen-final-sources.json")
    final_ids = {r["confirmation_source_id"] for r in rows if r["phase"] == "final_confirmation"}
    lookup = {r["id"]: r for r in rows}
    audit.check("frozen_final_source_identity", {r["id"] for r in frozen} == final_ids
        and all(r["sha256"] == digest(lookup[r["id"]]) for r in frozen))
    bad_hashes = [name for name, expected in report["artifacts_sha256"].items()
        if Path(name).name != name or not (study / name).is_file() or sha256(study / name) != expected]
    audit.check("all_saved_artifact_hashes", not bad_hashes, bad_hashes)
    audit.check("confirmed_bank_artifact_hash", sha256(study / "confirmed-bank.json") == report["confirmed_bank_sha256"])
    if source is not None:
        source = Path(source)
        bad = [name for name, expected in report["source_sha256"].items()
            if Path(name).name != name or not (source / name).is_file() or sha256(source / name) != expected]
        audit.check("immutable_executed_source_hashes", bool(report["source_sha256"]) and not bad, bad)
    fk = None
    if assets is not None:
        from .verify_learning import audit_fk
        accepted = [dict(r, locks=dict(board=None, dropper=None)) for r in rows if r["native_release"]]
        fk = audit_fk(Path(assets), spec, accepted, audit)
        fk["strict_and_native"] = audit_strict_endpoints(audit, assets, spec, accepted)
    if parent_study is not None and assets is not None:
        from .learning_solve import validate_artifacts
        _, _, cp_sha, fp = validate_artifacts(Path(parent_study), Path(assets))
        audit.check("parent_source_asset_and_scorer_guards", cp_sha == spec["parent_checkpoint_sha256"] and fp == spec["parent_contract_fingerprint"])
    warnings = [
            "This audits stored evidence and endpoint FK, not an independent ball-dynamics rerun.",
            "All ten positions are development; finite failed search is not impossibility.",
            "No policy training, inference deployment, grasping or collision-free arm transit is validated.",
            "Nominal numerical repeatability and perturbation confirmation are different measurements.",
        ]
    if source is None:
        warnings.append("Executed source bytes not supplied: immutable-source verification missing.")
    if assets is None:
        warnings.append("Native assets not supplied: independent strict FK, contacts, collision transforms and parent asset/scorer guards missing.")
    return dict(version=VERSION, passed=all(x["passed"] for x in audit.checks), checks=audit.checks,
        records=len(rows), independent_fk=fk, warnings=warnings)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("study", type=Path)
    for name in ("source", "assets", "prior-study", "prior-expansion", "parent-study", "output"):
        parser.add_argument("--" + name, type=Path)
    args = parser.parse_args()
    result = verify(args.study, source=args.source, assets=args.assets, prior_study=args.prior_study,
        prior_expansion=args.prior_expansion, parent_study=args.parent_study)
    if args.output:
        args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result, indent=2, allow_nan=False))
    raise SystemExit(not result["passed"])
