"""Read-only validation-only diagnostics for the frozen calibrated benchmark.

No policy prediction, search, rollout, hardware, network or service is started.
Only VAL rows and VAL-only dense packets are inspected. TRAIN/TEST trace access
is refused, including packets containing mixed-role IDs. Original evidence is
never modified. Compact recorded replays contain ball samples and INITIAL held
FK geometry, not invented arm trajectories or measured settled/impact poses.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import tempfile

import numpy as np

VERSION = "sircus-calibrated-validation-failure-audit-v1"
REPLAY_VERSION = "sircus-recorded-validation-replay-v1"
CATEGORIES = ("captured_clean", "captured_assisted", "setup_rejected", "invalid_physics",
              "external_disqualified_retention", "no_board_bounce", "entry_not_retained", "no_whole_ball_entry")
MISS_REGIONS = ("short", "long", "lateral", "over_mouth", "no_crossing", "unknown")
POSE_SCOPE = ("Initial native reset+forward FK; joints prescribed each native interval. "
              "Cached board translation drift sampled240Hz; no arm/board pose time history, "
              "dynamic settled pose or impact orientation is recorded. Ball is spawned in free flight; "
              "there is no simulated gripper release. Requested gap is ball-bottom above board CENTER plane.")
SOURCE_AUDIT_VERSION = "sircus-calibrated-sweep-rl-independent-audit-v2"
STUDY_VERSION = "sircus-calibrated-sweep-rl-benchmark-v2"


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for data in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(data)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def require(condition, message):
    if not condition:
        raise ValueError(message)


def signed(value):
    value = copy.deepcopy(value)
    value.pop("fingerprint", None)
    value["fingerprint"] = digest(value)
    return value


def verify_signed(value, label):
    unsigned = dict(value)
    require(unsigned.pop("fingerprint", None) == digest(unsigned), label + " fingerprint changed")


def member(root, name):
    require(isinstance(name, str) and name, "Nonempty relative evidence path required")
    relative = PurePosixPath(name)
    require(not relative.is_absolute() and ".." not in relative.parts and "\\" not in name, "Unsafe evidence path")
    root = Path(root).resolve()
    path = root / name
    require(not path.is_symlink() and path.resolve().is_relative_to(root), "Evidence path leaves declared root")
    require(path.is_file(), "Missing evidence file: " + name)
    return path


def bound_file(study, manifest, name):
    path = member(study, name)
    require(manifest.get(name) == sha(path), "Original manifest checksum changed: " + name)
    return path


def finite_vector(value, size, label):
    out = np.asarray(value, dtype=float)
    require(out.shape == (size,) and np.isfinite(out).all(), "Finite vector required: " + label)
    return out


def expected_normal(bucket, values):
    direction = np.asarray(bucket) - np.asarray(values[:2])
    length = np.linalg.norm(direction)
    require(length > 0, "Requested board/bucket heading is undefined")
    angle = np.deg2rad(values[3])
    return np.r_[np.sin(angle) * direction / length, np.cos(angle)]


def pose_record(row, scene, initial_ball=None):
    values = finite_vector(row["values"], 5, "shot parameters")
    normal = expected_normal(row["bucket_xy_m"], values)
    release = values[:3].copy()
    release[2] += values[4] + scene["ball"]["radius_m"]
    result = dict(scope=POSE_SCOPE, available=False,
                  requested=dict(board_center_m=values[:3].tolist(), board_normal=normal.tolist(),
                                 release_ball_center_m=release.tolist(), bucket_center_m=[*row["bucket_xy_m"], 0.],
                                 gap_above_center_plane_m=float(values[4]),
                                 q10=copy.deepcopy(row.get("q10")), q14=copy.deepcopy(row.get("q14"))),
                  native_initial=None, errors=None, sampled_cached_board_drift_mm=None,
                  dynamic_settled_pose="not_recorded", impact_board_orientation="not_recorded")
    if not row.get("native_release"):
        return result
    center = finite_vector(row["board_center_m"], 3, "initial board center")
    observed_normal = finite_vector(row["board_normal"], 3, "initial board normal")
    norm = np.linalg.norm(observed_normal)
    require(abs(norm - 1.) < 1e-4, "Initial board normal is not a rotation axis")
    observed_ball = finite_vector(row["native_initial_ball_center_m"], 3, "initial ball")
    if initial_ball is not None:
        require(np.allclose(observed_ball, initial_ball, atol=1e-9, rtol=0), "Row initial ball differs from dense time zero")
    bucket = finite_vector(row["native_initial_bucket_center_m"], 3, "initial bucket")
    q10 = finite_vector(row["native_initial_q10"], 6, "initial dropper joints")
    q14 = finite_vector(row["native_initial_q14"], 6, "initial board-arm joints")
    target_q = np.r_[finite_vector(row["q10"], 6, "certified dropper joints"), finite_vector(row["q14"], 6, "certified board joints")]
    angle = float(np.rad2deg(np.arccos(np.clip(np.dot(observed_normal / norm, normal), -1, 1))))
    errors = dict(board_position_mm=float(1000 * np.linalg.norm(center - values[:3])), board_normal_deg=angle,
                  release_position_mm=float(1000 * np.linalg.norm(observed_ball - release)),
                  bucket_position_mm=float(1000 * np.linalg.norm(bucket - np.r_[row["bucket_xy_m"], 0.])),
                  joint_max_rad=float(np.abs(np.r_[q10, q14] - target_q).max()),
                  flange_error_mm=float(row["pose_gate"]["dropper_error_mm"]))
    require(errors["board_position_mm"] < 2. and angle < .5 and errors["release_position_mm"] <= .002
            and errors["bucket_position_mm"] <= .002 and errors["joint_max_rad"] <= 2e-6
            and 0 <= errors["flange_error_mm"] < 2., "Recorded initial held pose violates frozen accuracy gates")
    result.update(available=True,
                  native_initial=dict(board_center_m=center.tolist(), board_normal=observed_normal.tolist(),
                                      board_quaternion_wxyz=copy.deepcopy(row.get("board_quaternion_wxyz")),
                                      release_ball_center_m=observed_ball.tolist(), bucket_center_m=bucket.tolist(),
                                      q10=q10.tolist(), q14=q14.tolist()),
                  errors=errors, sampled_cached_board_drift_mm=float(row["max_board_drift_mm"]),
                  max_integrated_joint_excursion_rad=float(row["max_integrated_joint_excursion_rad"]),
                  peak_integrated_joint_velocity_rad_s=float(row["peak_integrated_joint_velocity_rad_s"]))
    require(all(np.isfinite(v) and v >= 0 for v in errors.values()), "Nonfinite or negative pose errors")
    return result


def aggregate_pose(records):
    available = [r["pose_audit"] for r in records if r["pose_audit"]["available"]]
    def stats(values):
        return dict(mean=float(np.mean(values)), max=float(np.max(values)), p95=float(np.quantile(values, .95))) if values else None
    return dict(scope=POSE_SCOPE, attempts=len(records), native_releases=len(available),
                errors={key: stats([p["errors"][key] for p in available]) for key in
                        ("board_position_mm", "board_normal_deg", "release_position_mm", "bucket_position_mm", "joint_max_rad", "flange_error_mm")},
                sampled_cached_board_drift_mm=stats([p["sampled_cached_board_drift_mm"] for p in available]),
                dynamic_settled_pose="not_recorded", impact_board_orientation="not_recorded")


def direction_at_crossing(scene, calibration, row, arrays, stop, bounce_index):
    """Observed downward crossing only; no ballistic fitting or counterfactual."""
    output = dict(region="unknown", geometric_region=None, reason="no_accepted_rebound", crossing=None,
                  scope="Observed descending ball-center crossing at rim minus radius; nominal compiled whole-sphere XY clearance, not a cause attribution")
    if bounce_index is None:
        return output
    p, times = arrays["position_m"], arrays["time_s"]
    radius = float(scene["ball"]["radius_m"])
    top = float(scene["bucket"]["assumed_bottom_thickness_m"] + scene["bucket"]["assumed_interior_depth_m"])
    height = top - radius
    output["plane_center_z_m"] = height
    selected = np.flatnonzero((p[:-1, 2] > height) & (p[1:, 2] <= height) & (np.arange(len(p) - 1) >= bounce_index)
                              & (np.arange(len(p) - 1) < stop))
    if not len(selected):
        output.update(region="no_crossing", reason="no_recorded_post_rebound_descending_target_height_crossing")
        return output
    left = int(selected[0])
    fraction = float((p[left, 2] - height) / (p[left, 2] - p[left + 1, 2]))
    point = p[left] + fraction * (p[left + 1] - p[left])
    time = float(times[left] + fraction * (times[left + 1] - times[left]))
    bucket = np.asarray(row["bucket_xy_m"], float)
    direction = bucket - np.asarray(row["board_center_m"], float)[:2]
    length = np.linalg.norm(direction)
    if length <= 1e-9:
        output.update(reason="undefined_observed_board_bucket_heading")
        return output
    direction /= length
    side = np.array([-direction[1], direction[0]])
    u, v = float(np.dot(point[:2] - bucket, direction)), float(np.dot(point[:2] - bucket, side))
    planes = np.asarray(calibration["compiled_planes"], float)
    a = planes[:, :2] @ direction
    rhs = radius - planes[:, 2] * height - planes[:, 3] - (planes[:, :2] @ side) * v
    low, high = -math.inf, math.inf
    empty = bool(((np.abs(a) < 1e-12) & (rhs > 1e-9)).any())
    if (a > 1e-12).any(): low = float(np.max(rhs[a > 1e-12] / a[a > 1e-12]))
    if (a < -1e-12).any(): high = float(np.min(rhs[a < -1e-12] / a[a < -1e-12]))
    empty |= low > high + 1e-9
    region = "lateral" if empty else "short" if u < low - 1e-9 else "long" if u > high + 1e-9 else "over_mouth"
    intervening = bool((arrays["forces_N"][bounce_index:left + 1, 1:] > .01).any()
                       or (arrays["forces_N"][bounce_index:left + 1, 0] > .01).any())
    output.update(region="unknown" if intervening else region, geometric_region=region,
                  reason="intervening_contact_before_crossing" if intervening else "observed_target_height_crossing",
                  crossing=dict(time_s=time, ball_center_m=point.tolist(), along_bucket_axis_m=u, lateral_m=v,
                                nominal_allowed_along_interval_m=None if empty else [low, high],
                                later_contacts_do_not_change_this_observation=True))
    return output


def diagnose_native(row, scene, calibration, arrays):
    from .calibrated_score_oracle import score_trace
    measured = score_trace(scene, row["bucket_xy_m"], row["board_normal"], calibration, arrays)
    for key in ("route_success", "route_quality", "board_bounce", "whole_ball_entry", "max_retention_s", "invalid_physics",
                "external_contact_steps", "physics_steps", "retention_resets", "success"):
        require(key in row and key in measured, "Missing canonical score field: " + key)
        if isinstance(measured[key], float):
            require(abs(float(row[key]) - measured[key]) <= 1e-8, "Independent VAL oracle disagreement: " + key)
        else:
            require(row[key] == measured[key], "Independent VAL oracle disagreement: " + key)
    stop = int(measured["physics_steps"])
    p, v, f, t = (arrays[k] for k in ("position_m", "velocity_m_s", "forces_N", "time_s"))
    board = f[:stop, 0] > .01
    last = np.maximum.accumulate(np.where(board, t[1:stop + 1], -np.inf))
    normal = np.asarray(row["board_normal"], float)
    rebound = (~board) & (t[1:stop + 1] - last <= .05 + 1e-12) & ((v[1:stop + 1] @ normal) > .05)
    accepted = int(np.flatnonzero(rebound)[0]) + 1 if rebound.any() else None
    require(bool(accepted is not None) == bool(row["board_bounce"]), "Accepted rebound reconstruction differs")
    if row["invalid_physics"]: category = "invalid_physics"
    elif row["route_success"]: category = "captured_clean" if row["route_quality"] == 2 else "captured_assisted"
    elif row["success"]: category = "external_disqualified_retention"
    elif not row["board_bounce"]: category = "no_board_bounce"
    elif row["whole_ball_entry"]: category = "entry_not_retained"
    else: category = "no_whole_ball_entry"
    current = p[stop]
    radius, bucket = scene["ball"]["radius_m"], scene["bucket"]
    local = current.copy(); local[:2] -= row["bucket_xy_m"]
    planes = np.asarray(calibration["compiled_planes"])
    eps = calibration["geometry_epsilon_m"]
    inside_at_stop = bool(current[2] + radius <= bucket["assumed_bottom_thickness_m"] + bucket["assumed_interior_depth_m"] + eps
                          and (planes[:, :3] @ local + planes[:, 3] >= radius - calibration["wall_tolerance_m"] - eps).all()
                          and current[2] - radius >= bucket["assumed_bottom_thickness_m"] - calibration["floor_tolerance_m"] - eps)
    retention_status = None
    if category == "entry_not_retained":
        retention_status = "interrupted" if row["retention_resets"] else "horizon_censored" if stop == len(f) and inside_at_stop else "terminated_before_retention" if stop < len(f) else "unknown"
    contact_times = {}
    for i, name in enumerate(("board", "bucket", "table", "floor", "robot", "unattributed")):
        indexes = np.flatnonzero(f[:stop, i] > .01)
        contact_times[name] = float(t[indexes[0]]) if len(indexes) else None
    return dict(category=category, pose_audit=pose_record(row, scene, p[0]),
                miss_region=direction_at_crossing(scene, calibration, row, arrays, stop, accepted),
                retention_status=retention_status,
                observed_events=dict(first_contact_interval_start_s=contact_times, accepted_rebound_sample_time_s=float(t[accepted]) if accepted is not None else None,
                    board_contact_observed=bool(board.any()), board_contact_episodes=row["board_contact_episodes"],
                    whole_ball_entry_time_s=row.get("whole_ball_entry_time_s"), capture_time_s=row.get("capture_time_s"),
                    retention_resets=row["retention_resets"], max_retention_s=row["max_retention_s"],
                    rim_band_contact=bool(row.get("rim_band_contact")), bucket_contact_before_entry=bool(row.get("bucket_contact_before_entry")),
                    scorer_terminal_time_s=float(t[stop]), terminal_prefix_only=True,
                    containment_interruption_observed=bool(row["retention_resets"]),
                    inside_calibrated_envelope_at_terminal=inside_at_stop,
                    recording_horizon_reached=bool(stop == len(f)),
                    retention_horizon_censored=bool(category == "entry_not_retained" and stop == len(f) and inside_at_stop),
                    force_observation="Net-force-derived category magnitudes; bucket nearest contacts only, no measured board impact normal"))


def counts(records):
    categories = {key: sum(r["category"] == key for r in records) for key in CATEGORIES}
    regions = {key: sum(r["miss_region"]["region"] == key for r in records if r["category"] in ("no_whole_ball_entry", "entry_not_retained")) for key in MISS_REGIONS}
    reasons = {}
    for row in records:
        if row["category"] == "setup_rejected":
            reason = str(row.get("rejection_reason") or "unspecified")
            reasons[reason] = reasons.get(reason, 0) + 1
    return dict(attempts=len(records), captures=categories["captured_clean"] + categories["captured_assisted"],
                native_releases=len(records) - categories["setup_rejected"], categories=categories,
                miss_regions=regions, rejection_reasons=reasons)


def compact_replay(row, record, arrays, source):
    step = max(1, int(round((1 / 60) / float(arrays["time_s"][1] - arrays["time_s"][0]))))
    indices = np.arange(0, len(arrays["time_s"]), step)
    if indices[-1] != len(arrays["time_s"]) - 1: indices = np.r_[indices, len(arrays["time_s"]) - 1]
    trajectory = [dict(time=float(arrays["time_s"][i]), ball=arrays["position_m"][i].tolist()) for i in indices]
    configuration = dict(board_center=copy.deepcopy(row["board_center_m"]), board_normal=copy.deepcopy(row["board_normal"]),
        board_quaternion_wxyz=copy.deepcopy(row.get("board_quaternion_wxyz")), release_center=arrays["position_m"][0].tolist(),
        q10=copy.deepcopy(row["native_initial_q10"]), q14=copy.deepcopy(row["native_initial_q14"]),
        link_positions10=copy.deepcopy(row.get("link_positions10")), link_positions14=copy.deepcopy(row.get("link_positions14")))
    return dict(version=REPLAY_VERSION, id=record["replay_id"], role="validation", source=source,
                source_row_id=row["id"], method=row["method"], seed=row["seed"], samples=row["samples"], context_id=row["context_id"],
                category=record["category"], pose_audit=record["pose_audit"], observed_events=record["observed_events"],
                result=dict(request_id=row["id"], method="validation_recorded_replay", success=bool(row["route_success"]),
                    configuration=configuration, trajectory=trajectory, bucket_xy_m=copy.deepcopy(row["bucket_xy_m"]),
                    metrics=dict(route_quality=row["route_quality"],max_retention_s=row["max_retention_s"],category=record["category"],
                                 observed_miss_region=record["miss_region"],pose_audit=record["pose_audit"])),
                scope="Recorded dense ball samples at about60Hz, including time zero; static initial native FK arms only. " + POSE_SCOPE)


def read_validation_rows(path, method, seed, contract):
    """Stream the mixed metadata log, retaining VAL only; no other trace opened."""
    result = []
    layouts = {c["id"]: c for c in contract["splits"]["validation"]}
    milestones = contract["protocol"]["milestones"]
    with Path(path).open() as stream:
        for line in stream:
            row = json.loads(line)
            if row.get("role") != "validation":
                continue
            require(row.get("method") == method and row.get("seed") == seed, "VAL worker identity changed")
            require(row.get("samples") in milestones and row.get("context_id") in layouts and row.get("repeat") in (0, 1, 2), "VAL cohort identity changed")
            expected = f"{method}-{seed}-validation-{row['samples']:04d}-{row['context_id']}-r{row['repeat']}"
            require(row.get("id") == expected, "TRAIN/TEST or forged ID cannot be accessed as VAL")
            require(row["bucket_xy_m"] == layouts[row["context_id"]]["bucket_xy_m"], "VAL bucket differs from frozen layout")
            result.append(row)
    require(len(result) == 144 and len({r["id"] for r in result}) == 144, "Expected all144 unique VAL requests per worker")
    expected = {(n, c, k) for n in milestones for c in layouts for k in range(3)}
    require({(r["samples"], r["context_id"], r["repeat"]) for r in result} == expected, "VAL schedule missing or repeated")
    return result


def verify_source(study, audit_path, repository, expected_contract_fingerprint, expected_audit_sha256, *, synthetic=False):
    require(sha(audit_path) == expected_audit_sha256, "Pinned full-study audit bytes changed")
    audit = read(audit_path)
    require(audit.get("version") == SOURCE_AUDIT_VERSION and audit.get("status") == "passed"
            and type(audit.get("checks")) is int and audit["checks"] > 0 and not audit.get("failures"), "Passed independent whole-study audit required")
    contract, report, manifest = (read(study / n) for n in ("contract.json", "report.json", "manifest.json"))
    verify_signed(contract, "Frozen contract"); verify_signed(report, "Frozen report")
    require(contract.get("version") == STUDY_VERSION and report.get("version") == STUDY_VERSION, "Source benchmark version mismatch")
    require(contract["fingerprint"] == expected_contract_fingerprint == audit.get("contract_fingerprint") == report.get("contract_fingerprint"), "Exact frozen source contract required")
    require(audit.get("report_sha256") == sha(study / "report.json") and audit.get("report_fingerprint") == report["fingerprint"]
            and audit.get("study_manifest_sha256") == sha(study / "manifest.json"), "Full audit is stale relative to source bytes")
    require(report.get("status") == "completed" and report.get("phase") == "completed", "Complete source study required")
    require(report["protocol"] == contract["protocol"], "Report protocol differs")
    kind = "synthetic_fixture" if synthetic else "native_current_ball_v2"
    require(contract["protocol"].get("study_kind") == kind and audit.get("study_kind") == kind
            and bool(audit.get("synthetic_fixture")) == synthetic, "Evidence kind differs from explicit diagnostic mode")
    require(contract["protocol"]["milestones"] == [0, 256, 512, 1024] and len(contract["protocol"]["seeds"]) == 3
            and len(contract["splits"]["validation"]) == 12, "Original four-milestone, three-seed VAL protocol required")
    require(audit.get("source_sha256") == contract["identity"]["source_sha256"], "Full audit source identity differs")
    if not synthetic:
        require(set(audit.get("auditor_dependencies", {})) == {"sircus/mjlab/benchmark_verify.py", "sircus/mjlab/verify_capture_calibration.py"}, "Exact independent auditor dependencies required")
    for name, expected in audit["source_sha256"].items():
        require(sha(member(repository, name)) == expected, "Frozen producer source changed: " + name)
    require(sha(member(repository, "sircus/mjlab/calibrated_benchmark_verify.py")) == audit["auditor_sha256"], "Full-study auditor source changed")
    for name, expected in audit.get("auditor_dependencies", {}).items():
        require(sha(member(repository, name)) == expected, "Full-study audit dependency changed: " + name)
    for name in ("contract.json", "report.json", "calibration.json", "verification.json", "setup-home.json", "splits.json"):
        bound_file(study, manifest, name)
    require(read(study / "splits.json") == contract["splits"], "VAL split copy changed")
    require(audit["role_counts"]["validation"]["attempts"] == 864, "Whole audit did not cover all864 VAL requests")
    scene = read(member(repository, "sircus/isaac/scene.json"))
    require(digest(scene) == contract["identity"]["scene_sha256"], "Exact source scene required")
    calibration = read(study / "calibration.json")
    require(calibration["score_config"] == contract["identity"]["calibration"]["score_config"], "Executed score calibration differs")
    source = dict(contract_fingerprint=contract["fingerprint"], report_sha256=sha(study / "report.json"),
                  audit_sha256=sha(audit_path), study_manifest_sha256=sha(study / "manifest.json"),
                  source_sha256=copy.deepcopy(audit["source_sha256"]), auditor_sha256=audit["auditor_sha256"],
                  auditor_dependencies=copy.deepcopy(audit.get("auditor_dependencies", {})),
                  full_audit_checks=audit["checks"], original_role="validation")
    return contract, report, manifest, audit, scene, calibration["score_config"], source


def inspect_worker(study, manifest, worker, contract, scene, calibration, source):
    method, seed = worker["method"], worker["seed"]
    prefix = f"runs/{method}-seed{seed}"
    selection = read(bound_file(study, manifest, prefix + "/selected.json"))
    require(selection["selected_samples"] == worker["selected_samples"] and selection["frozen_before_test"] is True,
            "Frozen selected checkpoint identity changed")
    worker_manifest_path = bound_file(study, manifest, prefix + "/manifest.json")
    worker_manifest = read(worker_manifest_path)
    rows_path = bound_file(study, manifest, prefix + "/rows.jsonl")
    require(worker_manifest.get("rows.jsonl") == sha(rows_path), "Worker manifest does not bind VAL row metadata")
    rows = read_validation_rows(rows_path, method, seed, contract)
    by_file, records, replay_payloads = {}, [], {}
    source_files = {prefix + "/rows.jsonl": sha(rows_path), prefix + "/manifest.json": sha(worker_manifest_path),
                    prefix + "/selected.json": sha(study / prefix / "selected.json")}
    for row in rows:
        common = dict(source_row_id=row["id"], replay_id="val-" + hashlib.sha256(row["id"].encode()).hexdigest()[:32],
                      method=method, seed=seed, samples=row["samples"], context_id=row["context_id"],
                      bucket_xy_m=row["bucket_xy_m"], repeat=row["repeat"], primary_selected=row["samples"] == selection["selected_samples"],
                      selected_samples=selection["selected_samples"], native_release=bool(row.get("native_release")))
        if not row.get("native_release"):
            require(bool(row.get("invalid")) and not row.get("route_success"), "Rejected VAL request cannot be a capture")
            records.append(dict(common, category="setup_rejected", rejection_reason=row.get("invalid"),
                                pose_audit=pose_record(row, scene), retention_status=None, observed_events={},
                                miss_region=dict(region="unknown", reason="no_native_trace", crossing=None)))
        else:
            name = row.get("native_cache_file")
            require(isinstance(name, str) and name.endswith(".npz") and "/" not in name and "\\" not in name, "Simple VAL native packet name required")
            by_file.setdefault(name, []).append((row, common))
    for filename, members in by_file.items():
        # Every row was role/ID checked BEFORE opening this file. Read packet IDs
        # first and reject mixed/TRAIN/TEST packets before accessing dense arrays.
        packet_name = prefix + "/native/" + filename
        path = bound_file(study, manifest, packet_name)
        packet_sha = sha(path)
        require(worker_manifest.get("native/" + filename) == packet_sha, "Worker manifest native packet mismatch")
        require(all(row["native_cache_sha256"] == packet_sha for row, _ in members), "VAL row native checksum mismatch")
        source_files[packet_name] = packet_sha
        with np.load(path, allow_pickle=False) as archive:
            ids = archive["record_ids"].tolist()
            eligible = {row["id"] for row, _ in members}
            require(isinstance(ids, list) and set(ids) == eligible and len(ids) == len(eligible), "TRAIN/TEST/mixed trace packet refused")
            require(all("-validation-" in name and "-train-" not in name and "-test-" not in name for name in ids), "Only VAL native IDs allowed")
            arrays = {key: archive[key] for key in ("time_s", "position_m", "velocity_m_s", "forces_N", "contact_distance_m", "native_time_s")}
        time = np.asarray(arrays["time_s"], float)
        dt = contract["identity"]["trace_schema"]["native_dt_s"]
        horizon = contract["identity"]["trace_schema"]["horizon_s"]
        steps = int(round(horizon / dt))
        require(time.shape == (steps + 1,) and np.array_equal(time, np.arange(steps + 1) * dt), "Dense canonical VAL times changed")
        for key, shape in (("position_m", (steps + 1, len(ids), 3)), ("velocity_m_s", (steps + 1, len(ids), 3)),
                           ("forces_N", (steps, len(ids), 6)), ("contact_distance_m", (steps, len(ids), 2))):
            require(arrays[key].shape == shape, "Dense VAL packet shape mismatch: " + key)
        for row, common in members:
            index = row["native_cache_row"]
            require(type(index) is int and 0 <= index < len(ids) and ids[index] == row["id"], "VAL native column association changed")
            trace = {key: value[:, index] for key, value in arrays.items() if key != "time_s"}
            trace["time_s"] = time
            record = dict(common, **diagnose_native(row, scene, calibration, trace))
            records.append(record)
            replay_payloads[row["id"]] = compact_replay(row, record, trace, source)
        # Compact replay payloads contain Python scalars/lists only. Release all
        # dense packet arrays before loading the next packet (no dense cache).
        del trace, arrays
    return records, replay_payloads, source_files


def pick_representatives(records, cells):
    """One native failure-first example per displayed cell, plus rare categories."""
    selected = set()
    def rank(row):
        return (row["category"] in ("captured_clean", "captured_assisted"), not row["native_release"], row["seed"], row["repeat"], row["source_row_id"])
    for cell in cells:
        pool = [r for r in records if r["method"] == cell["method"] and r["context_id"] == cell["id"]
                and (r["primary_selected"] if cell["milestone"] == "selected" else r["samples"] == cell["milestone"])]
        require(pool, "A displayed VAL cell has no requests")
        selected.add(min(pool, key=rank)["source_row_id"])
    for method in ("sweep", "ppo"):
        for category in CATEGORIES:
            pool = [r for r in records if r["method"] == method and r["category"] == category]
            if pool: selected.add(min(pool, key=lambda r: (not r["primary_selected"], *rank(r)))["source_row_id"])
    return selected


def build(study, audit_path, repository, output, *, expected_contract_fingerprint, expected_audit_sha256, synthetic=False):
    study, audit_path, repository, output = (Path(p).resolve() for p in (study, audit_path, repository, output))
    require(not output.exists() and not output.is_relative_to(study), "New diagnostic output must be outside immutable study")
    contract, benchmark, manifest, audit, scene, calibration, source = verify_source(
        study, audit_path, repository, expected_contract_fingerprint, expected_audit_sha256, synthetic=synthetic)
    pairs = {(m, s) for m in ("sweep", "ppo") for s in contract["protocol"]["seeds"]}
    require(len(benchmark["runs"]) == 6 and {(r["method"], r["seed"]) for r in benchmark["runs"]} == pairs, "All six completed source workers required")
    records, replays, source_files = [], {}, {}
    for worker in sorted(benchmark["runs"], key=lambda r: (r["method"], r["seed"])):
        require(worker["status"] == "completed", "Incomplete source worker")
        observed, payloads, hashes = inspect_worker(study, manifest, worker, contract, scene, calibration, source)
        records.extend(observed); replays.update(payloads); source_files.update(hashes)
    require(len(records) == 864 and len({r["source_row_id"] for r in records}) == 864, "All864 VAL requests required")
    all_counts = counts(records)
    for key in ("attempts", "captures", "native_releases"):
        require(all_counts[key] == audit["role_counts"]["validation"][key], "VAL diagnostic counts disagree with passed source audit")
    selected = [r for r in records if r["primary_selected"]]
    require(len(selected) == 216, "Selected primary VAL cohort must have216 requests")
    layouts = copy.deepcopy(contract["splits"]["validation"])
    cohorts = dict(selected=dict(**counts(selected), by_method={m: counts([r for r in selected if r["method"] == m]) for m in ("sweep", "ppo")}),
                   milestones=[dict(samples=n, **counts([r for r in records if r["samples"] == n]),
                                    by_method={m: counts([r for r in records if r["samples"] == n and r["method"] == m]) for m in ("sweep", "ppo")})
                               for n in contract["protocol"]["milestones"]])
    cells = []
    for method in ("sweep", "ppo"):
        for milestone in ("selected", *contract["protocol"]["milestones"]):
            for layout in layouts:
                pool = [r for r in records if r["method"] == method and r["context_id"] == layout["id"]
                        and (r["primary_selected"] if milestone == "selected" else r["samples"] == milestone)]
                require(len(pool) == 9, "Every method/layout/cohort must retain all9 seed/repeat requests")
                cells.append(dict(id=layout["id"], bucket_xy_m=layout["bucket_xy_m"], method=method, milestone=milestone,
                                  **counts(pool), selected_samples_by_seed={str(r["seed"]): r["selected_samples"] for r in pool},
                                  pose_audit=aggregate_pose(pool), representative_ids=[]))
    chosen = pick_representatives(records, cells)
    representatives, payload_bytes = [], {}
    for row in sorted((r for r in records if r["source_row_id"] in chosen), key=lambda r: r["replay_id"]):
        entry = dict(id=row["replay_id"], replay_id=row["replay_id"], source_row_id=row["source_row_id"],
                     method=row["method"], seed=row["seed"], samples=row["samples"], context_id=row["context_id"],
                     bucket_xy_m=row["bucket_xy_m"], category=row["category"], miss_region=row["miss_region"],
                     retention_status=row["retention_status"], primary_selected=row["primary_selected"],
                     recorded_replay_available=row["native_release"], pose_audit=row["pose_audit"], observed_events=row["observed_events"])
        if row["native_release"]:
            name = "replays/" + row["replay_id"] + ".json"
            raw = (json.dumps(replays[row["source_row_id"]], indent=2, allow_nan=False) + "\n").encode()
            payload_bytes[name] = raw
            entry.update(path=name, sha256=hashlib.sha256(raw).hexdigest())
        representatives.append(entry)
    for cell in cells:
        pool = [r for r in representatives if r["method"] == cell["method"] and r["context_id"] == cell["id"]
                and (r["primary_selected"] if cell["milestone"] == "selected" else r["samples"] == cell["milestone"])]
        cell["representative_ids"] = [r["id"] for r in pool]
    report = signed(dict(version=VERSION, status="passed", synthetic_fixture=bool(synthetic), source=source,
        diagnostic_source_sha256=sha(Path(__file__)), inspected_validation_files_sha256=source_files,
        layouts=layouts, cohorts=cohorts, cells=cells, representatives=representatives,
        pose_audit=aggregate_pose(records),
        summary=dict(all_milestone_requests=all_counts, primary_selected_requests=counts(selected),
                     selected_samples=[dict(method=w["method"], seed=w["seed"], samples=w["selected_samples"]) for w in benchmark["runs"]]),
        definitions=dict(categories=list(CATEGORIES), miss_regions=list(MISS_REGIONS),
            cohort_scope="All864 VAL requests at four separately labeled milestones; primary216 selected-checkpoint requests are a SUBSET and must never be added to milestone totals",
            capture_rule="Accepted board rebound, observed whole-ball mouth entry, and one continuous calibrated second; rim assistance counts",
            miss_plane="Descending observed ball CENTER z=rim−radius; compiled sphere-clearance polygon at this height. Missing/intervening-contact trajectories remain unknown/no_crossing; no speed/tilt cause inferred",
            pose_scope=POSE_SCOPE, contact_scope="Net-force-derived category magnitudes, nearest bucket wall/base observations; board impact normal and point are not recorded"),
        limitations=["Validation selected checkpoints; this diagnostic is not an additional held-out TEST estimate.",
                     "Same12 layouts and paired perturbations across seeds/milestones; cells are descriptive counts, not independent trials.",
                     "No TRAIN/TEST native arrays read; original benchmark, policies, calibrations and outcome definitions unchanged.",
                     "Recorded ball replay with static INITIAL held FK arm geometry; no driven-motion or compliant-impact replay.",
                     "No new physics, search, training, sensing, policy changes or material inference."]))
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".validation-failure-audit-", dir=output.parent) as temporary:
        staged = Path(temporary) / "diagnostic"; staged.mkdir()
        for name, raw in payload_bytes.items():
            path = staged / name; path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(raw)
        (staged / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        files = {str(p.relative_to(staged)): sha(p) for p in staged.rglob("*") if p.is_file()}
        (staged / "manifest.json").write_text(json.dumps(files, indent=2, allow_nan=False) + "\n")
        require(sha(audit_path) == expected_audit_sha256 and sha(study / "report.json") == source["report_sha256"]
                and sha(study / "manifest.json") == source["study_manifest_sha256"], "Immutable source changed during diagnostic")
        require(not output.exists(), "Diagnostic output appeared during build")
        staged.rename(output)
    return dict(status="passed", version=VERSION, validation_requests=len(records), primary_selected_requests=len(selected),
                report_sha256=sha(output / "report.json"), representatives=len(representatives), recorded_replays=len(payload_bytes))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("study", "audit", "repository", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--expected-contract-fingerprint", required=True)
    parser.add_argument("--expected-audit-sha256", required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.study, args.audit, args.repository, args.output,
                           expected_contract_fingerprint=args.expected_contract_fingerprint,
                           expected_audit_sha256=args.expected_audit_sha256), allow_nan=False))


if __name__ == "__main__":
    main()
