"""Versioned contact-calibrated scoring for dense native traces.

Positions include the state at time zero. Force row k describes the interval
[t[k], t[k+1]], rather than an instantaneous force paired with the final pose.
The caller binds this configuration to independently frozen calibration and
verification reports. Historical scorers and labels are never imported here.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import re

import numpy as np

VERSION = "sircus-calibrated-native-capture-v2"
CALIBRATION_VERSION = "sircus-contact-capture-calibration-v1"
REWARD_VERSION = "sircus-calibrated-shot-reward-v2"
ROUTE_VERSION = "sircus-calibrated-board-route-v2"
FORCE_NAMES = ("board", "bucket", "table", "floor", "robot", "unattributed")
FORCE_THRESHOLD_N = 0.01
REBOUND_NORMAL_SPEED_M_S = 0.05
CONTACT_FREE_S = 0.05
TIME_EPS_S = 1e-12


def calibration_fingerprint(value):
    unsigned = dict(value)
    unsigned.pop("fingerprint", None)
    return hashlib.sha256(
        json.dumps(unsigned, sort_keys=True, allow_nan=False).encode()
    ).hexdigest()


def validate_calibration(value):
    """Validate the score configuration; report/verification bindings are external."""
    if not isinstance(value, dict) or value.get("version") != CALIBRATION_VERSION:
        raise ValueError("Unsupported capture calibration version")
    result = copy.deepcopy(value)
    fingerprint = result.get("fingerprint")
    if not isinstance(fingerprint, str) or not re.fullmatch(r"[0-9a-f]{64}", fingerprint):
        raise ValueError("Calibration fingerprint is missing")
    if calibration_fingerprint(result) != fingerprint:
        raise ValueError("Calibration fingerprint changed")
    for name in ("wall_tolerance_m", "floor_tolerance_m"):
        number = result.get(name)
        if isinstance(number, bool) or not isinstance(number, (int, float)) or not math.isfinite(number) or number < 0:
            raise ValueError("Calibration contact tolerances must be finite and nonnegative")
    for name in ("geometry_epsilon_m", "max_speed_m_s", "retention_s"):
        number = result.get(name)
        if isinstance(number, bool) or not isinstance(number, (int, float)) or not math.isfinite(number) or number <= 0:
            raise ValueError("Calibration scales must be finite and positive")
    if result["retention_s"] != 1.0:
        raise ValueError("Capture requires one continuous second")
    planes = np.asarray(result.get("compiled_planes"), dtype=float)
    if planes.ndim != 2 or planes.shape[1] != 4 or len(planes) < 3 or not np.isfinite(planes).all():
        raise ValueError("Expected finite inward bucket-local compiled wall planes")
    if not np.allclose(np.linalg.norm(planes[:, :3], axis=1), 1, atol=1e-8, rtol=0):
        raise ValueError("Compiled wall plane normals must have unit length")
    if (np.linalg.norm(planes[:, :2], axis=1) < 0.5).any():
        raise ValueError("Compiled wall planes must describe side walls")
    if "native_dt_s" in result:
        dt = result["native_dt_s"]
        if isinstance(dt, bool) or not isinstance(dt, (int, float)) or not math.isfinite(dt) or dt <= 0:
            raise ValueError("Invalid calibrated native time step")
    return result


def reward(row):
    """The fixed shaped reward, explicitly bound to the new capture version."""
    if row.get("invalid") or row.get("invalid_physics"):
        return -1.0
    shaped = 0.15 * float(row["board_bounce"]) + 0.5 * math.exp(
        -float(row["closest_opening_distance_m"]) / 0.12
    )
    if row["external_contact_steps"]:
        return float(shaped - 0.5)
    return float(
        shaped + 0.5 * min(float(row["max_retention_s"]), 1.0)
        + 3 * bool(row["route_success"])
        + 0.3 * (row["route_quality"] == 2)
        + 0.1 * bool(row["single_bounce_success"])
    )


def _trace(arrays):
    position = np.asarray(arrays["position_m"], dtype=float)
    velocity = np.asarray(arrays["velocity_m_s"], dtype=float)
    force = np.asarray(arrays["forces_N"], dtype=float)
    if position.ndim != 2 or position.shape[1] != 3 or len(position) < 2 or velocity.shape != position.shape or force.shape != (len(position) - 1, 6):
        raise ValueError("Dense trace requires T+1 positions/velocities and T force intervals")
    if not np.isfinite(position).all() or not np.isfinite(velocity).all() or not np.isfinite(force).all() or (force < 0).any():
        raise ValueError("Invalid dense trace values")
    if "time_s" in arrays:
        times = np.asarray(arrays["time_s"], dtype=float)
        if times.shape != (len(position),) or not np.isfinite(times).all() or abs(times[0]) > TIME_EPS_S or (np.diff(times) <= 0).any():
            raise ValueError("Time samples must begin at zero and increase strictly")
        if "dt_s" in arrays:
            dt = float(np.asarray(arrays["dt_s"]).item())
            if not math.isfinite(dt) or dt <= 0 or not np.allclose(np.diff(times), dt, atol=1e-12, rtol=1e-9):
                raise ValueError("Trace time samples contradict dt_s")
    else:
        dt = float(np.asarray(arrays.get("dt_s", float("nan"))).item())
        if not math.isfinite(dt) or dt <= 0:
            raise ValueError("Explicit time_s or positive dt_s is required")
        times = np.arange(len(position), dtype=float) * dt
    contact = None
    if "contact_distance_m" in arrays:
        contact = np.asarray(arrays["contact_distance_m"], dtype=float)
        if contact.shape != (len(force), 2) or np.isnan(contact).any() or np.isneginf(contact).any():
            raise ValueError("Contact distances require T by wall/base values; +inf means absent")
    return position, velocity, force, times, contact


def _geometry(scene, bucket_xy_m, calibration, position):
    bucket = scene["bucket"]
    radius = float(scene["ball"]["radius_m"])
    floor = float(bucket["assumed_bottom_thickness_m"])
    top = floor + float(bucket["assumed_interior_depth_m"])
    if radius <= 0 or not math.isfinite(radius) or not 0 <= floor < top:
        raise ValueError("Invalid sphere or bucket dimensions")
    if max(calibration["wall_tolerance_m"], calibration["floor_tolerance_m"], calibration["geometry_epsilon_m"]) >= radius:
        raise ValueError("Contact envelope cannot exceed the sphere radius")
    planes = np.asarray(calibration["compiled_planes"], dtype=float)
    if len(planes) != int(bucket["segments"]):
        raise ValueError("Compiled wall count differs from scene")
    eps = calibration["geometry_epsilon_m"]
    horizontal = np.linalg.norm(planes[:, :2], axis=1)
    cosine = math.cos(math.pi / int(bucket["segments"]))
    angles = np.sort(np.mod(np.arctan2(-planes[:, 1], -planes[:, 0]), 2 * math.pi))
    expected_angles = (np.arange(len(planes)) + 0.5) * 2 * math.pi / len(planes)
    if np.max(np.abs(angles - expected_angles)) * (bucket["outer_top_diameter_m"] / 2) > eps:
        raise ValueError("Compiled wall directions differ from scene")
    for height, expected in ((floor, bucket["inner_bottom_diameter_m"] / 2 * cosine), (top, bucket["inner_top_diameter_m"] / 2 * cosine)):
        inferred = (planes[:, 2] * height + planes[:, 3]) / horizontal
        if np.max(np.abs(inferred - expected)) > eps:
            raise ValueError("Compiled wall geometry differs from scene")
    for key, expected in (("compiled_floor_m", floor), ("compiled_rim_m", top)):
        if key in calibration and abs(float(calibration[key]) - expected) > eps:
            raise ValueError("Compiled vertical geometry differs from scene")
    xy = np.asarray(bucket_xy_m, dtype=float)
    if xy.shape != (2,) or not np.isfinite(xy).all():
        raise ValueError("Expected finite bucket XY")
    local = position.copy()
    local[:, :2] -= xy
    wall = np.empty(len(position))
    mouth = np.empty(len(position))
    # Bound temporary matrices even for the full 61,441-state native trace.
    for start in range(0, len(position), 8192):
        stop = min(start + 8192, len(position))
        distances = local[start:stop] @ planes[:, :3].T + planes[:, 3]
        wall[start:stop] = distances.min(axis=1)
        at_rim = (distances - (local[start:stop, 2] - top)[:, None] * planes[:, 2]) / horizontal
        mouth[start:stop] = at_rim.min(axis=1)
    return local, wall, mouth, radius, floor, top, planes, horizontal


def _runs(mask):
    padded = np.r_[False, np.asarray(mask, bool), False]
    change = np.diff(padded.astype(np.int8))
    return list(zip(np.flatnonzero(change == 1), np.flatnonzero(change == -1) - 1))


def _first(mask, default):
    found = np.flatnonzero(mask)
    return int(found[0]) if len(found) else default


def score_trace(scene, bucket_xy_m, normal, calibration, arrays):
    """Score one dense trace using calibrated boundary contact, never old labels.

    A below-rim run can become an entry only after a measured board rebound and
    a downward upper-cap crossing whose interpolated XY lies in the open mouth.
    Initial-inside states and later side-wall arrivals do not create an entry.
    Contact envelope slack never relaxes the upper rim. Retention starts at the
    first observed valid inside timestamp, without credit for an earlier interval.
    """
    cfg = validate_calibration(calibration)
    p, v, f, t, contact_distance = _trace(arrays)
    normal = np.asarray(normal, dtype=float)
    if normal.shape != (3,) or not np.isfinite(normal).all() or not np.isclose(np.linalg.norm(normal), 1, atol=1e-6, rtol=0):
        raise ValueError("Expected a finite unit board normal")
    local, wall, mouth, r, floor, top, planes, horizontal = _geometry(scene, bucket_xy_m, cfg, p)
    eps = cfg["geometry_epsilon_m"]
    below = p[:, 2] + r <= top + eps
    centre_side = (wall >= -eps) & (p[:, 2] >= floor - eps)
    candidate = below & centre_side
    inside = below & (wall >= r - cfg["wall_tolerance_m"] - eps) & (p[:, 2] - r >= floor - cfg["floor_tolerance_m"] - eps)
    wall_overlap = np.where(candidate, np.maximum(0, r - wall), 0)
    floor_overlap = np.where(candidate, np.maximum(0, floor + r - p[:, 2]), 0)

    board = f[:, 0] > FORCE_THRESHOLD_N
    last_contact_end = np.maximum.accumulate(np.where(board, t[1:], -np.inf))
    rebound = (~board) & (t[1:] - last_contact_end <= CONTACT_FREE_S + TIME_EPS_S) & ((v[1:] @ normal) > REBOUND_NORMAL_SPEED_M_S)
    bounce_index = _first(rebound, len(f)) + 1
    has_bounce = bounce_index < len(p)

    passage = np.zeros(len(p), dtype=bool)
    passage_starts = []
    for start, end in _runs(below):
        if start == 0 or start < bounce_index:
            continue
        previous, current = local[start - 1], local[start]
        denominator = previous[2] - current[2]
        if denominator <= 0:
            continue
        fraction = np.clip((previous[2] + r - top - eps) / denominator, 0, 1)
        crossing = previous + fraction * (current - previous)
        clearance = ((planes[:, :2] @ crossing[:2]) + planes[:, 2] * top + planes[:, 3]) / horizontal
        if clearance.min() >= -eps:
            passage[start:end + 1] = True
            passage_starts.append(int(start))
    eligible = passage & inside
    entry_index = _first(eligible, len(p))
    true_runs = _runs(eligible)
    capture_index = len(p)
    for start, end in true_runs:
        target = t[start] + cfg["retention_s"]
        if t[end] >= target - TIME_EPS_S:
            index = int(np.searchsorted(t, target - TIME_EPS_S, side="left"))
            if index <= end:
                capture_index = index
                break

    contact_domain = np.zeros(len(p), dtype=bool)
    active_contact = f[:, 1] > FORCE_THRESHOLD_N
    if contact_distance is not None:
        active_contact |= np.isfinite(contact_distance).any(axis=1)
    contact_domain[:-1] |= active_contact
    contact_domain[1:] |= active_contact
    speed = np.linalg.norm(v, axis=1)
    physics_masks = {
        "bucket_contact_speed_outside_calibration": (candidate | contact_domain) & (speed > cfg["max_speed_m_s"] + 1e-9),
        "wall_overlap_outside_calibration": candidate & (wall_overlap > cfg["wall_tolerance_m"] + eps),
        "floor_overlap_outside_calibration": candidate & (floor_overlap > cfg["floor_tolerance_m"] + eps),
    }
    if entry_index < len(p):
        after = np.arange(len(p)) >= entry_index
        physics_masks["through_wall_after_entry"] = after & below & (wall < -eps)
        physics_masks["through_base_after_entry"] = after & below & (p[:, 2] < floor - eps)
    if contact_distance is not None:
        physics_masks["contact_wall_penetration_outside_calibration"] = np.r_[False, -contact_distance[:, 0] > cfg["wall_tolerance_m"] + eps]
        physics_masks["contact_floor_penetration_outside_calibration"] = np.r_[False, -contact_distance[:, 1] > cfg["floor_tolerance_m"] + eps]
    bad = np.logical_or.reduce(list(physics_masks.values()))
    physics_index = _first(bad, len(p))
    fall_index = _first(p[:, 2] < -0.2, len(p))
    stop = min(capture_index, physics_index, fall_index, len(p) - 1)
    invalid_physics = physics_index <= stop
    success = capture_index <= stop and capture_index < len(p) and not invalid_physics
    reasons = [name for name, mask in physics_masks.items() if mask[:stop + 1].any()] if invalid_physics else []
    # The original success terminal freezes labels. Later escapes/contacts or
    # uncalibrated states remain in raw evidence but cannot revoke a completed shot.
    entry_seen = entry_index <= stop
    max_retention = 0.0
    retention_resets = 0
    for start, end in true_runs:
        if start > stop:
            break
        max_retention = max(max_retention, float(t[min(end, stop)] - t[start]))
        retention_resets += int(end < stop)
    interval_f = f[:stop]
    active_board = np.flatnonzero(interval_f[:, 0] > FORCE_THRESHOLD_N)
    episodes = 0
    if len(active_board):
        episodes = 1 + int(np.count_nonzero(t[active_board[1:]] - t[active_board[:-1] + 1] >= CONTACT_FREE_S - TIME_EPS_S))
    external = int(np.count_nonzero((interval_f[:, 2:] > FORCE_THRESHOLD_N).any(axis=1)))
    route = bool(success and episodes >= 1 and external == 0)
    bounced = bool(has_bounce and bounce_index <= stop)
    closest = 2.0
    if bounced:
        delta_xy = local[bounce_index:stop + 1, :2]
        dz = p[bounce_index:stop + 1, 2] - (top - r)
        closest = min(closest, float(np.sqrt((delta_xy * delta_xy).sum(axis=1) + dz * dz).min()))

    bucket_intervals = interval_f[:, 1] > FORCE_THRESHOLD_N
    before_end = min(entry_index, stop) if entry_seen else stop
    contact_before = bool(bucket_intervals[:before_end].any())
    peak_before = float(f[:before_end, 1].max()) if before_end else 0.0
    rim_contact = bool((bucket_intervals & (np.maximum(p[:stop, 2], p[1:stop + 1, 2]) + r >= top - 0.002)).any())
    exited = bool(entry_seen and (~below[entry_index:stop + 1]).any())
    # Mouth-quality clearance is restricted to the downward passage associated
    # with the actual entry, not an earlier same-height flight beside the bucket.
    mouth_margin = None
    saw_mouth = False
    if entry_seen:
        start = bounce_index
        for left, right in _runs(p[:, 2] - r <= top + eps):
            if left <= entry_index <= right:
                start = max(int(left), bounce_index)
                break
        ids = np.arange(start, entry_index + 1)
        ids = ids[np.abs(p[ids, 2] - top) <= r + eps]
        if len(ids):
            cross_radius = np.sqrt(np.maximum(0, r * r - (p[ids, 2] - top) ** 2))
            mouth_margin = float((mouth[ids] - cross_radius).min())
            saw_mouth = True
    clean = bool(success and saw_mouth and not contact_before and not rim_contact and not exited and mouth_margin >= 0.001)
    quality = 2 if clean else 1 if success else 0
    contact_peaks = {name: float(interval_f[:, i].max()) if stop else 0.0 for i, name in enumerate(FORCE_NAMES)}
    score = dict(
        success=bool(success), complete_success=route, route_success=route,
        route_quality=quality if route else 0, single_bounce_success=bool(route and episodes == 1),
        board_bounce=bounced, whole_ball_entry=bool(entry_seen),
        max_retention_s=float(min(max_retention, cfg["retention_s"])),
        closest_opening_distance_m=closest, quality_score=quality,
        entry_class="clean_capture" if clean else "assisted_or_marginal_capture" if success else "miss",
        bucket_contact_before_entry=contact_before, rim_band_contact=rim_contact,
        exited_after_entry=exited, peak_bucket_force_before_entry_N=peak_before,
        minimum_mouth_clearance_mm=mouth_margin * 1000 if saw_mouth else None,
        board_contact_episodes=episodes, external_contact_steps=external, contact_peaks_N=contact_peaks,
        physics_steps=int(stop), scoring_samples=int(stop + 1),
        invalid="uncalibrated_contact_dynamics" if invalid_physics else None,
        invalid_physics=bool(invalid_physics), invalid_physics_reasons=reasons,
        first_mouth_passage_time_s=float(t[passage_starts[0]]) if passage_starts and passage_starts[0] <= stop else None,
        whole_ball_entry_time_s=float(t[entry_index]) if entry_seen else None,
        capture_time_s=float(t[capture_index]) if success else None,
        retention_resets=retention_resets,
        max_speed_m_s=float(speed[:stop + 1].max()),
        max_bucket_domain_speed_m_s=float(speed[:stop + 1][(candidate | contact_domain)[:stop + 1]].max()) if (candidate | contact_domain)[:stop + 1].any() else 0.0,
        max_nominal_wall_overlap_m=float(wall_overlap[:stop + 1].max()),
        max_nominal_floor_overlap_m=float(floor_overlap[:stop + 1].max()),
        max_contact_wall_penetration_m=float(np.maximum(0, -contact_distance[:stop, 0]).max()) if contact_distance is not None and stop else None,
        max_contact_floor_penetration_m=float(np.maximum(0, -contact_distance[:stop, 1]).max()) if contact_distance is not None and stop else None,
        contact_distance_evidence=bool(contact_distance is not None),
        scoring_version=VERSION, reward_version=REWARD_VERSION, route_version=ROUTE_VERSION,
        calibration_fingerprint=cfg["fingerprint"],
        quality_scope="Dense native interval contacts; entry passage mouth clearance >=1mm and no bucket/rim contact for clean tier; rim assistance remains a capture",
        retention_scope="Whole sphere below rim, contact-calibrated wall/base envelope, observed mouth entry and one continuous second from first inside timestamp",
        physics_guard_scope="Verified bucket-contact and inside-cavity speed/penetration domain; ordinary outside free fall is not invalidated",
    )
    score["reward"] = reward(score)
    return score
