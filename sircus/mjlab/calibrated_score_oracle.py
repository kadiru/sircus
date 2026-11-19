"""Independent CPU oracle for calibrated dense capture traces.

No production geometry, event, duration or reward implementation is imported.
Each compiled plane is evaluated separately; temporal state is reconstructed by
last-crossing and last-invalid-state arrays rather than the production run scan.
This audits saved native data and never performs a physics rollout.
"""

from __future__ import annotations

import hashlib
import json
import math
import re

import numpy as np

VERSION = "sircus-calibrated-native-capture-v2"
CALIBRATION_VERSION = "sircus-contact-capture-calibration-v1"
REWARD_VERSION = "sircus-calibrated-shot-reward-v2"
ROUTE_VERSION = "sircus-calibrated-board-route-v2"
_NAMES = ("board", "bucket", "table", "floor", "robot", "unattributed")
_EPS_TIME = 1e-12


def validate_calibration(value):
    if not isinstance(value, dict) or value.get("version") != CALIBRATION_VERSION:
        raise ValueError("Unsupported capture calibration")
    signed = dict(value)
    expected = signed.pop("fingerprint", None)
    if not isinstance(expected, str) or re.fullmatch("[0-9a-f]{64}", expected) is None:
        raise ValueError("Missing calibration digest")
    actual = hashlib.sha256(json.dumps(signed, sort_keys=True, allow_nan=False).encode()).hexdigest()
    if actual != expected:
        raise ValueError("Changed calibration digest")
    for name, zero_ok in (("wall_tolerance_m", True), ("floor_tolerance_m", True), ("geometry_epsilon_m", False), ("max_speed_m_s", False), ("retention_s", False)):
        number = value.get(name)
        if type(number) not in (int, float) or not math.isfinite(number) or number < 0 or (not zero_ok and number == 0):
            raise ValueError("Invalid calibration bound")
    if value["retention_s"] != 1:
        raise ValueError("Wrong retention duration")
    coefficients = np.asarray(value.get("compiled_planes"), dtype=float)
    if coefficients.ndim != 2 or coefficients.shape[1:] != (4,) or len(coefficients) < 3 or not np.isfinite(coefficients).all():
        raise ValueError("Invalid compiled planes")
    for x, y, z, _ in coefficients:
        if abs(math.sqrt(x * x + y * y + z * z) - 1) > 1e-8 or math.hypot(x, y) < 0.5:
            raise ValueError("Invalid compiled plane normalization")
    if "native_dt_s" in value:
        dt = value["native_dt_s"]
        if type(dt) not in (int, float) or not math.isfinite(dt) or dt <= 0:
            raise ValueError("Invalid native dt")
    return value


def reward(row):
    if row.get("invalid") or row.get("invalid_physics"):
        return -1.0
    proximity = math.exp(-float(row["closest_opening_distance_m"]) / 0.12) / 2
    base = 0.15 * int(bool(row["board_bounce"])) + proximity
    if int(row["external_contact_steps"]) > 0:
        return float(base - 0.5)
    return float(base + min(float(row["max_retention_s"]), 1) / 2 + 3 * int(bool(row["route_success"])) + (0.3 if row["route_quality"] == 2 else 0) + (0.1 if row["single_bounce_success"] else 0))


def score_trace(scene, bucket_xy_m, normal, calibration, arrays):
    cfg = validate_calibration(calibration)
    position = np.asarray(arrays["position_m"], dtype=float)
    velocity = np.asarray(arrays["velocity_m_s"], dtype=float)
    force = np.asarray(arrays["forces_N"], dtype=float)
    if position.ndim != 2 or position.shape[1:] != (3,) or len(position) < 2 or velocity.shape != position.shape or force.shape != (len(position) - 1, 6):
        raise ValueError("Wrong dense trace shape")
    if any(not np.isfinite(a).all() for a in (position, velocity, force)) or (force < 0).any():
        raise ValueError("Nonfinite or negative force trace")
    size = len(position)
    if "time_s" in arrays:
        times = np.asarray(arrays["time_s"], dtype=float)
        if times.shape != (size,) or not np.isfinite(times).all() or abs(times[0]) > _EPS_TIME or np.min(np.diff(times)) <= 0:
            raise ValueError("Invalid trace timestamps")
        if "dt_s" in arrays:
            step = float(np.asarray(arrays["dt_s"]).item())
            if not math.isfinite(step) or step <= 0 or not np.allclose(np.diff(times), step, atol=1e-12, rtol=1e-9):
                raise ValueError("Contradictory trace cadence")
    else:
        step = float(np.asarray(arrays.get("dt_s", float("nan"))).item())
        if not math.isfinite(step) or step <= 0:
            raise ValueError("Missing trace cadence")
        times = np.arange(size, dtype=float) * step
    contacts = np.asarray(arrays["contact_distance_m"], dtype=float) if "contact_distance_m" in arrays else None
    if contacts is not None and (contacts.shape != (size - 1, 2) or np.isnan(contacts).any() or np.isneginf(contacts).any()):
        raise ValueError("Invalid contact distance evidence")
    axis = np.asarray(normal, dtype=float)
    xy = np.asarray(bucket_xy_m, dtype=float)
    if xy.shape != (2,) or not np.isfinite(xy).all() or axis.shape != (3,) or not np.isfinite(axis).all() or abs(math.sqrt(float((axis * axis).sum())) - 1) > 1e-6:
        raise ValueError("Invalid bucket/board reference")
    bucket = scene["bucket"]
    radius = float(scene["ball"]["radius_m"])
    floor = float(bucket["assumed_bottom_thickness_m"])
    rim = floor + float(bucket["assumed_interior_depth_m"])
    if not math.isfinite(radius) or radius <= 0 or not 0 <= floor < rim or max(cfg["wall_tolerance_m"], cfg["floor_tolerance_m"], cfg["geometry_epsilon_m"]) >= radius:
        raise ValueError("Invalid capture geometry")
    coefficients = np.asarray(cfg["compiled_planes"], dtype=float)
    if len(coefficients) != int(bucket["segments"]):
        raise ValueError("Compiled wall count differs")
    epsilon = float(cfg["geometry_epsilon_m"])
    polygon_cos = math.cos(math.pi / int(bucket["segments"]))
    directions = sorted(math.atan2(-row[1], -row[0]) % (2 * math.pi) for row in coefficients)
    for index, direction in enumerate(directions):
        if abs(direction - (index + 0.5) * 2 * math.pi / len(coefficients)) * bucket["outer_top_diameter_m"] / 2 > epsilon:
            raise ValueError("Compiled wall directions disagree")
    for height, diameter in ((floor, bucket["inner_bottom_diameter_m"]), (rim, bucket["inner_top_diameter_m"])):
        for nx, ny, nz, b in coefficients:
            if abs((nz * height + b) / math.hypot(nx, ny) - diameter * polygon_cos / 2) > epsilon:
                raise ValueError("Compiled walls disagree with scene")
    for name, number in (("compiled_floor_m", floor), ("compiled_rim_m", rim)):
        if name in cfg and abs(float(cfg[name]) - number) > epsilon:
            raise ValueError("Compiled vertical geometry disagrees")
    x = position[:, 0] - xy[0]
    y = position[:, 1] - xy[1]
    z = position[:, 2]
    distance = np.full(size, np.inf)
    mouth_distance = np.full(size, np.inf)
    for nx, ny, nz, b in coefficients:
        distance = np.minimum(distance, nx * x + ny * y + nz * z + b)
        mouth_distance = np.minimum(mouth_distance, (nx * x + ny * y + nz * rim + b) / math.hypot(nx, ny))
    under = z + radius <= rim + epsilon
    cavity_side = (distance >= -epsilon) & (z >= floor - epsilon)
    domain = under & cavity_side
    full = under & (distance >= radius - cfg["wall_tolerance_m"] - epsilon) & (z - radius >= floor - cfg["floor_tolerance_m"] - epsilon)
    wall_depth = np.zeros(size)
    base_depth = np.zeros(size)
    wall_depth[domain] = np.maximum(0, radius - distance[domain])
    base_depth[domain] = np.maximum(0, floor + radius - z[domain])

    on_board = force[:, 0] > 0.01
    observed_contact_ends = times[1:][on_board]
    rebound = np.zeros(size, dtype=bool)
    if len(observed_contact_ends):
        match = np.searchsorted(observed_contact_ends, times[1:] + _EPS_TIME, side="right") - 1
        recent = np.zeros(size - 1, dtype=bool)
        recent[match >= 0] = times[1:][match >= 0] - observed_contact_ends[match[match >= 0]] <= 0.05 + _EPS_TIME
        rebound[1:] = recent & ~on_board & ((velocity[1:] * axis).sum(axis=1) > 0.05)
    rebound_indices = np.flatnonzero(rebound)
    bounce = int(rebound_indices[0]) if len(rebound_indices) else size
    state_ids = np.arange(size)
    crossing = under & ~np.r_[under[0], under[:-1]]
    accepted = np.zeros(size, dtype=bool)
    for index in np.flatnonzero(crossing):
        if index < bounce or index == 0:
            continue
        dz = z[index - 1] - z[index]
        if dz <= 0:
            continue
        alpha = max(0.0, min(1.0, (z[index - 1] + radius - rim - epsilon) / dz))
        cross_x = x[index - 1] + alpha * (x[index] - x[index - 1])
        cross_y = y[index - 1] + alpha * (y[index] - y[index - 1])
        edge_distances = [(nx * cross_x + ny * cross_y + nz * rim + b) / math.hypot(nx, ny) for nx, ny, nz, b in coefficients]
        accepted[index] = min(edge_distances) >= -epsilon
    previous_crossing = np.maximum.accumulate(np.where(crossing, state_ids, -1))
    passed_mouth = np.zeros(size, dtype=bool)
    found = previous_crossing >= 0
    passed_mouth[found] = accepted[previous_crossing[found]]
    retained = under & passed_mouth & full
    entries = np.flatnonzero(retained)
    entry = int(entries[0]) if len(entries) else size
    previous_bad = np.maximum.accumulate(np.where(~retained, state_ids, -1))
    duration = np.zeros(size)
    valid_ids = np.flatnonzero(retained)
    duration[valid_ids] = times[valid_ids] - times[previous_bad[valid_ids] + 1]
    capture_ids = np.flatnonzero(duration >= cfg["retention_s"] - _EPS_TIME)
    capture = int(capture_ids[0]) if len(capture_ids) else size

    bucket_intervals_all = force[:, 1] > 0.01
    if contacts is not None:
        bucket_intervals_all |= np.isfinite(contacts).any(axis=1)
    contact_domain = np.r_[bucket_intervals_all, False] | np.r_[False, bucket_intervals_all]
    speed = np.sqrt((velocity * velocity).sum(axis=1))
    violations = {
        "bucket_contact_speed_outside_calibration": (domain | contact_domain) & (speed > cfg["max_speed_m_s"] + 1e-9),
        "wall_overlap_outside_calibration": domain & (wall_depth > cfg["wall_tolerance_m"] + epsilon),
        "floor_overlap_outside_calibration": domain & (base_depth > cfg["floor_tolerance_m"] + epsilon),
    }
    if entry < size:
        violations["through_wall_after_entry"] = (state_ids >= entry) & under & (distance < -epsilon)
        violations["through_base_after_entry"] = (state_ids >= entry) & under & (z < floor - epsilon)
    if contacts is not None:
        violations["contact_wall_penetration_outside_calibration"] = np.r_[False, -contacts[:, 0] > cfg["wall_tolerance_m"] + epsilon]
        violations["contact_floor_penetration_outside_calibration"] = np.r_[False, -contacts[:, 1] > cfg["floor_tolerance_m"] + epsilon]
    bad = np.zeros(size, dtype=bool)
    for values in violations.values():
        bad |= values
    bad_ids = np.flatnonzero(bad)
    physics = int(bad_ids[0]) if len(bad_ids) else size
    fall_ids = np.flatnonzero(z < -0.2)
    fall = int(fall_ids[0]) if len(fall_ids) else size
    terminal = min(capture, physics, fall, size - 1)
    invalid = physics <= terminal
    success = capture <= terminal and capture < size and not invalid
    entry_observed = entry <= terminal
    reasons = [name for name in violations if violations[name][:terminal + 1].any()] if invalid else []
    max_retention = min(float(duration[:terminal + 1].max()), float(cfg["retention_s"]))
    resets = int(np.count_nonzero(retained[:terminal] & ~retained[1:terminal + 1]))

    prefix_force = force[:terminal]
    board_ids = np.flatnonzero(prefix_force[:, 0] > 0.01)
    episodes = int(bool(len(board_ids)))
    if len(board_ids) > 1:
        gaps = times[board_ids[1:]] - times[board_ids[:-1] + 1]
        episodes += int(np.count_nonzero(gaps >= 0.05 - _EPS_TIME))
    external = int(np.count_nonzero(np.max(prefix_force[:, 2:], axis=1) > 0.01)) if terminal else 0
    bounced = bounce <= terminal
    route = bool(success and episodes > 0 and external == 0)
    nearest = 2.0
    if bounced:
        radial = np.hypot(x[bounce:terminal + 1], y[bounce:terminal + 1])
        nearest = min(nearest, float(np.hypot(radial, z[bounce:terminal + 1] - rim + radius).min()))
    cutoff = min(entry, terminal) if entry_observed else terminal
    contact_before = bool((force[:cutoff, 1] > 0.01).any())
    peak_before = float(force[:cutoff, 1].max()) if cutoff else 0.0
    rim_contact = bool(((prefix_force[:, 1] > 0.01) & ((z[:terminal] + radius >= rim - 0.002) | (z[1:terminal + 1] + radius >= rim - 0.002))).any())
    exited = bool(entry_observed and (~under[entry:terminal + 1]).any())
    minimum_margin = None
    if entry_observed:
        previous_above = np.flatnonzero(z[bounce:entry] - radius > rim + epsilon) + bounce
        start = int(previous_above[-1] + 1) if len(previous_above) else bounce
        mouth_ids = np.arange(start, entry + 1)
        mouth_ids = mouth_ids[np.abs(z[mouth_ids] - rim) <= radius + epsilon]
        if len(mouth_ids):
            circles = np.sqrt(np.maximum(0, radius * radius - (z[mouth_ids] - rim) * (z[mouth_ids] - rim)))
            minimum_margin = float(np.min(mouth_distance[mouth_ids] - circles))
    saw_mouth = minimum_margin is not None
    clean = bool(success and saw_mouth and not contact_before and not rim_contact and not exited and minimum_margin >= 0.001)
    quality = 2 if clean else 1 if success else 0
    peaks = {name: float(prefix_force[:, k].max()) if terminal else 0.0 for k, name in enumerate(_NAMES)}
    relevant = (domain | contact_domain)[:terminal + 1]
    result = dict(
        success=bool(success), complete_success=route, route_success=route,
        route_quality=quality if route else 0, single_bounce_success=bool(route and episodes == 1),
        board_bounce=bool(bounced), whole_ball_entry=bool(entry_observed),
        max_retention_s=max_retention, closest_opening_distance_m=nearest,
        quality_score=quality,
        entry_class="clean_capture" if clean else "assisted_or_marginal_capture" if success else "miss",
        bucket_contact_before_entry=contact_before, rim_band_contact=rim_contact,
        exited_after_entry=exited, peak_bucket_force_before_entry_N=peak_before,
        minimum_mouth_clearance_mm=minimum_margin * 1000 if saw_mouth else None,
        board_contact_episodes=episodes, external_contact_steps=external, contact_peaks_N=peaks,
        physics_steps=int(terminal), scoring_samples=int(terminal + 1),
        invalid="uncalibrated_contact_dynamics" if invalid else None,
        invalid_physics=bool(invalid), invalid_physics_reasons=reasons,
        first_mouth_passage_time_s=float(times[np.flatnonzero(accepted)[0]]) if accepted[:terminal + 1].any() else None,
        whole_ball_entry_time_s=float(times[entry]) if entry_observed else None,
        capture_time_s=float(times[capture]) if success else None,
        retention_resets=resets, max_speed_m_s=float(speed[:terminal + 1].max()),
        max_bucket_domain_speed_m_s=float(speed[:terminal + 1][relevant].max()) if relevant.any() else 0.0,
        max_nominal_wall_overlap_m=float(wall_depth[:terminal + 1].max()),
        max_nominal_floor_overlap_m=float(base_depth[:terminal + 1].max()),
        max_contact_wall_penetration_m=float(np.maximum(0, -contacts[:terminal, 0]).max()) if contacts is not None and terminal else None,
        max_contact_floor_penetration_m=float(np.maximum(0, -contacts[:terminal, 1]).max()) if contacts is not None and terminal else None,
        contact_distance_evidence=bool(contacts is not None),
        scoring_version=VERSION, reward_version=REWARD_VERSION, route_version=ROUTE_VERSION,
        calibration_fingerprint=cfg["fingerprint"],
        quality_scope="Dense native interval contacts; entry passage mouth clearance >=1mm and no bucket/rim contact for clean tier; rim assistance remains a capture",
        retention_scope="Whole sphere below rim, contact-calibrated wall/base envelope, observed mouth entry and one continuous second from first inside timestamp",
        physics_guard_scope="Verified bucket-contact and inside-cavity speed/penetration domain; ordinary outside free fall is not invalidated",
    )
    result["reward"] = reward(result)
    return result
