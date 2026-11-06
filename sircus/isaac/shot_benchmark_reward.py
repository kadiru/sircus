"""Versioned miss feedback from the native trajectory before external contact.

This module does not replace the capture scorer or historical farm reward.
Cell-local samples use the unchanged ShotScore and BoardRoute. The complete
per-cell scoring window ends at the original success/floor termination; the
shaping prefix excludes the entire first external-contact sample and everything
after it. Bucket/rim and board contacts are not external route contacts.
"""

import copy
import math
from collections.abc import Mapping
from numbers import Integral, Real

import numpy as np
from board_route import BoardRoute, FORCE_THRESHOLD_N, VERSION as ROUTE_VERSION
from rl_contract import ShotScore, VERSION as SCORE_VERSION

VERSION = "sircus-shot-benchmark-preexternal-reward-v1"
FEATURE_VERSION = "sircus-preexternal-shot-features-v1"
FORCE_NAMES = ("board", "bucket", "table", "floor", "robot", "unattributed")
DISTANCE_SCALE_M = .20


def protocol():
    return dict(version=VERSION, feature_version=FEATURE_VERSION,
        scorer_version=SCORE_VERSION, route_version=ROUTE_VERSION,
        invalid=-1., no_preexternal_board_bounce=-.8,
        miss_base=-.60, proximity_weight=.45, distance_scale_m=DISTANCE_SCALE_M,
        retention_weight=.15, retention_scale_s=1., external_route_penalty=.10,
        qualified_capture=1.5, clean_capture_bonus=.10,
        miss_bounds=[-.70, 0.], external_miss_upper_bound=-.10,
        first_external_step="1-based relative scoring sample; that sample and all later samples excluded from shaping",
        prefix="Unchanged ShotScore state after qualified board bounce, strictly before first external force >0.01N; no global closest/retention fallback",
        distance_geometry="Unchanged ShotScore.closest: Euclidean ball-center distance to bucket center XY at Z=top minus ball radius, capped by its historical 2m initialization; not an aperture-clearance measurement",
        capture="Unchanged board bounce + whole-ball entry + continuous one second inside + BoardRoute without external contact; rim-assisted captures count",
        scope="Terminal training reward only. Capture labels, physics, material assumptions and historical returns are not changed.")


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _number(value, label, minimum=0.):
    _require(isinstance(value, Real) and not isinstance(value, (bool, np.bool_))
        and math.isfinite(float(value)) and float(value) >= minimum,
        "Invalid " + label)
    return float(value)


def _count(value, label, minimum=0):
    _require(isinstance(value, Integral) and not isinstance(value, (bool, np.bool_))
        and int(value) >= minimum, "Invalid " + label)
    return int(value)


def _flag(value, label):
    _require(isinstance(value, (bool, np.bool_)), "Invalid " + label)
    return bool(value)


def _array(value, shape, label):
    try:
        result = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError("Invalid " + label) from error
    _require(result.shape == shape and np.isfinite(result).all(), "Invalid " + label)
    return result


def build_features(scene, bucket_xy, positions, velocities, forces, dt, board_normal):
    """Extract one cell's immutable shaping prefix from its actual native trace.

    positions/velocities are Nx3 in cell-local/world-aligned coordinates; forces
    are nonnegative Nx6 magnitudes in FORCE_NAMES order. board_normal is the
    fixed measured 3-vector used by the original score, not an estimated target.
    Full-batch tails beyond this cell's success or Z < -0.2 are ignored.
    """
    _require(isinstance(scene, Mapping), "Expected scene configuration")
    dt = _number(dt, "physics dt", minimum=1e-12)
    configured_dt = _number(scene["simulation"]["physics_dt_s"], "configured dt", minimum=1e-12)
    _require(abs(dt-configured_dt) <= 1e-12, "Trace dt differs from scene")
    raw = np.asarray(positions)
    _require(raw.ndim == 2 and raw.shape[1:] == (3,) and len(raw) > 0,
        "Expected nonempty Nx3 position trace")
    count = len(raw)
    positions = _array(positions, (count, 3), "positions")
    velocities = _array(velocities, (count, 3), "velocities")
    forces = _array(forces, (count, len(FORCE_NAMES)), "forces")
    _require((forces >= 0).all(), "Negative contact force magnitude")
    normal = _array(board_normal, (3,), "measured scoring board normal")
    _require(abs(float(np.linalg.norm(normal))-1.) <= 1e-4, "Scoring normal is not unit length")
    xy = _array(bucket_xy, (2,), "bucket center")
    radius = _number(scene["ball"]["radius_m"], "ball radius", minimum=1e-12)
    bucket = copy.deepcopy(scene["bucket"])
    bucket["center_xy_m"] = xy.tolist()
    for name in ("assumed_interior_depth_m", "inner_top_diameter_m", "inner_bottom_diameter_m"):
        _number(bucket[name], name, minimum=1e-12)
    _number(bucket["assumed_bottom_thickness_m"], "bucket bottom thickness")
    _count(bucket["segments"], "bucket segment count", minimum=3)
    score, route = ShotScore(bucket, radius), BoardRoute(dt)
    first_external = None
    prefix_steps = 0
    bounced, closest, retained = False, None, 0.
    consumed = 0
    for index, (position, velocity, force) in enumerate(zip(positions, velocities, forces)):
        external = bool(np.any(force[2:] > FORCE_THRESHOLD_N))
        if first_external is None and external:
            first_external = index+1
        score.step(position, float(force[0]), dt, float(velocity @ normal))
        route.step(dict(zip(FORCE_NAMES, map(float, force))))
        consumed = index+1
        if first_external is None:
            prefix_steps = consumed
            bounced = bool(score.bounced)
            closest = float(score.closest) if bounced else None
            retained = float(score.max_retention)
        if score.success or position[2] < -.2:
            break
    observed = score.result()
    # Quality is deliberately not redefined here: route_success is independent
    # of the quality argument; the unchanged producer supplies route_quality.
    route_result = route.result(observed["success"], 0)
    observed = {name: observed[name] for name in (
        "board_bounce", "closest_opening_distance_m", "max_retention_s",
        "whole_ball_entry", "success")}
    observed.update(external_contact_steps=route_result["external_contact_steps"],
                    route_success=route_result["route_success"])
    return dict(feature_version=FEATURE_VERSION, scorer_version=SCORE_VERSION,
        route_version=ROUTE_VERSION, dt_s=dt, trace_steps=count,
        steps_seen=consumed, prefix_steps=prefix_steps,
        first_external_step=first_external,
        board_bounce_preexternal=bounced,
        closest_opening_distance_preexternal_m=closest,
        max_retention_preexternal_s=retained, observed=observed)


def _validated(row, features):
    _require(isinstance(features, Mapping), "Native preexternal feature packet required; no legacy fallback")
    _require(features.get("feature_version") == FEATURE_VERSION
        and features.get("scorer_version") == SCORE_VERSION
        and features.get("route_version") == ROUTE_VERSION,
        "Incompatible preexternal feature/scoring version")
    steps = _count(features.get("steps_seen"), "steps seen", minimum=1)
    _require(_count(features.get("trace_steps"), "trace steps", minimum=1) >= steps,
        "Scoring window exceeds trace")
    prefix = _count(features.get("prefix_steps"), "prefix steps")
    external_step = features.get("first_external_step")
    external = _count(row.get("external_contact_steps"), "external contact steps") > 0
    if external_step is None:
        _require(not external and prefix == steps, "Missing or inconsistent external cutoff")
    else:
        cutoff = _count(external_step, "first external step", minimum=1)
        _require(external and cutoff <= steps and prefix == cutoff-1,
            "External-contact sample entered shaping prefix")
    _number(features.get("dt_s"), "feature dt", minimum=1e-12)
    bounced = _flag(features.get("board_bounce_preexternal"), "prefix board bounce")
    distance = features.get("closest_opening_distance_preexternal_m")
    retention = _number(features.get("max_retention_preexternal_s"), "prefix retention")
    _require(retention <= prefix*features["dt_s"]+1e-9, "Prefix retention exceeds observed prefix time")
    if bounced:
        distance = _number(distance, "prefix closest-opening distance")
        _require(prefix > 0, "Empty prefix cannot contain a board bounce")
    else:
        _require(distance is None and retention == 0., "No-bounce prefix cannot claim progress")
    observed = features.get("observed")
    _require(isinstance(observed, Mapping), "Missing native score evidence")
    for name in ("board_bounce", "route_success"):
        _require(_flag(row.get(name), name) == _flag(observed.get(name), "observed "+name),
            "Feature packet belongs to a different native outcome: "+name)
    for name in ("closest_opening_distance_m", "max_retention_s"):
        _require(abs(_number(row.get(name), name)-_number(observed.get(name), "observed "+name)) <= 1e-7,
            "Feature packet belongs to a different native outcome: "+name)
    _require(_count(row.get("external_contact_steps"), "external steps") ==
        _count(observed.get("external_contact_steps"), "observed external steps"),
        "Feature packet has different external-contact evidence")
    for name in ("whole_ball_entry", "success"):
        if name in row:
            _require(_flag(row[name], name) == _flag(observed.get(name), "observed "+name),
                "Feature packet has different "+name+" evidence")
    _require(not bounced or observed["board_bounce"], "Prefix bounce absent from full score")
    _require(retention <= observed["max_retention_s"]+1e-7, "Prefix retention exceeds full score")
    if bounced:
        _require(distance+1e-7 >= observed["closest_opening_distance_m"],
            "Prefix distance is inconsistent with full closest approach")
    quality = _count(row.get("route_quality"), "qualified route quality")
    _require(quality in ((1, 2) if row["route_success"] else (0,)),
        "Route quality contradicts qualified capture")
    if row["route_success"]:
        _require(not external and observed["board_bounce"] and observed.get("success")
            and observed.get("whole_ball_entry") and observed["max_retention_s"] >= 1.-1e-9,
            "Qualified capture lacks the unchanged capture evidence")
    return bounced, distance, retention, external, quality


def reward(row, features=None):
    """Return the new training reward; never modify row/capture labels."""
    _require(isinstance(row, Mapping) and "invalid" in row, "Expected native outcome row")
    if row["invalid"]:
        _require(not row.get("route_success", False), "Rejected action cannot capture")
        return -1.
    bounced, distance, retention, external, quality = _validated(row, features)
    if row["route_success"]:
        return 1.5 + .10*(quality == 2)
    if not bounced:
        return -.8
    value = -.60 + .45*math.exp(-distance/DISTANCE_SCALE_M) + .15*min(retention, 1.) - .10*external
    # Numerical boundary clipping only; the formula itself has these bounds.
    return float(np.clip(value, -.70, -.10 if external else 0.))


def statistics(rows, features):
    """Qualified counts remain the producer's unchanged labels; rewards differ."""
    rows, features = list(rows), list(features)
    _require(len(rows) == len(features), "One feature packet per attempted shot required")
    values = [reward(row, packet) for row, packet in zip(rows, features)]
    return dict(reward_version=VERSION, attempts=len(rows),
        captures=sum(bool(row.get("route_success", False)) for row in rows),
        clean=sum(row.get("route_quality") == 2 for row in rows),
        single_bounce=sum(bool(row.get("single_bounce_success", False)) for row in rows),
        invalid=sum(bool(row["invalid"]) for row in rows),
        mean_reward=math.fsum(values)/len(values) if values else None,
        min_reward=min(values) if values else None,
        max_reward=max(values) if values else None)
