"""Pure CPU contract and proposals for explicit native shot search.

Proposals are not success labels. Exact cache identity includes scene, assets,
executed sources, contact physics and the full query; distance alone is not an
identity. Native endpoint, flight, confirmation and path gates remain separate.
"""

from __future__ import annotations

from collections.abc import Mapping
import copy
import hashlib
import json
import math
from numbers import Real
import uuid

import numpy as np

from .corrected_contract import PHYSICS
from .expanded_actions import WIDE_BOUNDS

VERSION = "sircus-explicit-sweep-planner-v1"
PROPOSAL_VERSION = "sircus-sphere-plane-and-broad-proposals-v1"
NAMES = ("board_x_m", "board_y_m", "board_height_m", "tilt_deg", "gap_m")
BOUNDS = copy.deepcopy(WIDE_BOUNDS)
SEED = 20261129
MAX_CPU_CANDIDATES = 1024
MAX_NATIVE_ATTEMPTS = 128
NOMINAL_REPEATS = 3
FRESH_REPEATS = 20
FRESH_REQUIRED = 18
NOISE_STD = (0.001, 0.001, 0.002, 0.35, 0.002)


def digest(value):
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def _vector(value, count, label):
    if not isinstance(value, (list, tuple, np.ndarray)) or len(value) != count:
        raise ValueError(f"{label} requires {count} finite numbers")
    if any(
        isinstance(v, (bool, np.bool_))
        or not isinstance(v, Real)
        or not math.isfinite(v)
        for v in value
    ):
        raise ValueError(f"{label} requires {count} finite numbers")
    return [float(v) for v in value]


def bucket_bounds(scene):
    """Whole outer top footprint must fit on the provisional tabletop."""
    center = _vector(scene["table"]["center_m"], 3, "table center")
    size = _vector(scene["table"]["size_m"], 3, "table size")
    radius = float(scene["bucket"]["outer_top_diameter_m"]) / 2
    if (
        not math.isfinite(radius)
        or radius <= 0
        or any(s <= 2 * radius for s in size[:2])
    ):
        raise ValueError("Malformed bucket/table footprint")
    return {
        axis: [center[i] - size[i] / 2 + radius, center[i] + size[i] / 2 - radius]
        for i, axis in enumerate(("x", "y"))
    }


def _integer(value, lower, upper, label):
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or not lower <= value <= upper
    ):
        raise ValueError(f"{label} must be an integer from {lower} to {upper}")
    return value


def normalize_constraints(value=None):
    value = {} if value is None else value
    if not isinstance(value, Mapping) or set(value) - {
        "board_position",
        "dropper_position",
    }:
        raise ValueError("Unknown positional constraints")
    return {
        name: None if value.get(name) is None else _vector(value[name], 3, name)
        for name in ("board_position", "dropper_position")
    }


def validate_request(payload, scene):
    """Validate an explicit search independently of any learned policy files."""
    allowed = {
        "request_id",
        "mode",
        "bucket_xy",
        "constraints",
        "search_budget",
        "cpu_candidates",
    }
    if not isinstance(payload, Mapping) or set(payload) - allowed:
        raise ValueError("Unknown search request fields")
    if payload.get("mode", "search") not in ("search", "sweep"):
        raise ValueError("This planner supports explicit search only")
    xy = _vector(payload.get("bucket_xy"), 2, "bucket_xy")
    limits = bucket_bounds(scene)
    for name, coordinate in zip(("x", "y"), xy):
        if not limits[name][0] - 1e-12 <= coordinate <= limits[name][1] + 1e-12:
            raise ValueError(f"Bucket {name} footprint lies outside the table")
    locks = normalize_constraints(payload.get("constraints"))
    lo, hi = np.asarray(BOUNDS["low"]), np.asarray(BOUNDS["high"])
    board, dropper = locks["board_position"], locks["dropper_position"]
    if board is not None and not ((lo[:3] <= board).all() and (board <= hi[:3]).all()):
        raise ValueError("Locked board center lies outside this search family")
    if dropper is not None:
        if not ((lo[:2] <= dropper[:2]).all() and (dropper[:2] <= hi[:2]).all()):
            raise ValueError("Locked release XY lies outside this search family")
        radius = float(scene["ball"]["radius_m"])
        minimum_h, maximum_h = max(lo[2], dropper[2] - radius - hi[4]), min(
            hi[2], dropper[2] - radius - lo[4]
        )
        if minimum_h > maximum_h or (
            board is not None and not minimum_h <= board[2] <= maximum_h
        ):
            raise ValueError("Locked release height conflicts with height/gap bounds")
    if board is not None and dropper is not None and board[:2] != dropper[:2]:
        raise ValueError("Locked XY conflict with vertical center-impact shots")
    native = _integer(
        payload.get("search_budget", 64), 16, MAX_NATIVE_ATTEMPTS, "search_budget"
    )
    cpu = _integer(
        payload.get("cpu_candidates", max(256, native)),
        1,
        MAX_CPU_CANDIDATES,
        "cpu_candidates",
    )
    if cpu < native:
        raise ValueError("cpu_candidates must cover the native search budget")
    identity = payload.get("request_id", str(uuid.uuid4()))
    if not isinstance(identity, str) or not 1 <= len(identity) <= 128:
        raise ValueError("Invalid request_id")
    return dict(
        request_id=identity,
        mode="search",
        bucket_xy=xy,
        constraints=locks,
        search_budget=native,
        cpu_candidates=cpu,
    )


def _hashes(value, label):
    if not isinstance(value, Mapping) or not value:
        raise ValueError(f"Nonempty {label} hashes required for cache identity")
    result = {}
    for name, sha in value.items():
        if (
            not isinstance(name, str)
            or not name
            or not isinstance(sha, str)
            or len(sha) != 64
            or any(c not in "0123456789abcdef" for c in sha)
        ):
            raise ValueError(f"Malformed {label} hash")
        result[name] = sha
    return result


def request_key(scene, asset_hashes, request, *, source_hashes, physics=None):
    """No rounded coordinates or approximate/distance-only cache matching."""
    clean = validate_request(request, scene)
    clean.pop("request_id")
    return digest(
        dict(
            version=VERSION,
            scene=scene,
            assets=_hashes(asset_hashes, "asset"),
            sources=_hashes(source_hashes, "source"),
            physics=PHYSICS if physics is None else physics,
            request=clean,
            bounds=BOUNDS,
            nominal_repeats=NOMINAL_REPEATS,
            fresh_repeats=FRESH_REPEATS,
            fresh_required=FRESH_REQUIRED,
            perturbation_std=list(NOISE_STD),
        )
    )


def inverse_gap(scene, height, distance, tilt_deg, restitution=None):
    """Ideal descending center crossing at rim+radius; proposal, never capture.

    Includes sphere radius and finite board thickness at vertical first contact.
    A missing ideal solution is not a proof of native infeasibility.
    """
    e = scene["ball"].get("restitution", 0.65) if restitution is None else restitution
    inputs = (height, distance, tilt_deg, e)
    if not all(isinstance(v, Real) and math.isfinite(v) for v in inputs):
        return None
    if distance <= 0 or not 0 < tilt_deg < 90 or not 0 <= e <= 1:
        return None
    theta = math.radians(tilt_deg)
    radius = float(scene["ball"]["radius_m"])
    impact = height + (radius + scene["board"]["size_m"][2] / 2) / math.cos(theta)
    target = (
        scene["bucket"]["assumed_interior_depth_m"]
        + scene["bucket"]["assumed_bottom_thickness_m"]
        + radius
    )
    horizontal = (1 + e) * math.sin(theta) * math.cos(theta)
    vertical = (1 + e) * math.cos(theta) ** 2 - 1
    denominator = impact + vertical * distance / horizontal - target
    if denominator <= 0:
        return None
    fall = distance * distance / (4 * horizontal * horizontal * denominator)
    speed = math.sqrt(2 * 9.81 * fall)
    arrival = distance / (horizontal * speed)
    if vertical * speed - 9.81 * arrival >= 0:
        return None
    gap = fall + impact - height - radius
    return float(gap) if math.isfinite(gap) and gap > 0 else None


def _halton(index, base):
    value, weight = 0.0, 1.0
    while index:
        index, digit = divmod(index, base)
        weight /= base
        value += digit * weight
    return value


def _hint(row):
    if not isinstance(row, Mapping):
        raise ValueError("Source hint must be a record")
    values = _vector(row.get("values"), 5, "source values")
    xy = _vector(row.get("bucket_xy_m"), 2, "source bucket")
    source = dict(values=values, bucket_xy_m=xy)
    for key in ("q10", "q14"):
        if row.get(key) is not None:
            source[key] = _vector(row[key], 6, f"source {key}")
    return dict(
        source_id=str(row.get("source_id", row.get("id", "unspecified"))),
        source_record_sha256=digest(dict(row)),
        source_hint=source,
    )


def proposals(scene, bucket_xy, limit=256, seed=SEED, hints=(), *, constraints=None):
    """Deterministic bounded geometric proposals; no success data inherited.

    Action order is X/Y/H/tilt/G. Exact hints lead, then every zero-rotation
    translated hint, followed by interleaved rotated-relative hints, corrected
    inverse ballistics and full-box/ring
    space-filling samples. Out-of-range proposals are omitted, never clipped.
    Returned values have not passed IK, clearance, native contact or path checks.
    """
    _integer(limit, 1, MAX_CPU_CANDIDATES, "limit")
    _integer(seed, 0, 2**63 - 1, "seed")
    xy = np.asarray(_vector(bucket_xy, 2, "bucket_xy"))
    locks = normalize_constraints(constraints)
    board, dropper = locks["board_position"], locks["dropper_position"]
    if board is not None and dropper is not None and board[:2] != dropper[:2]:
        raise ValueError("Locked XY conflict with vertical center-impact shots")
    low, high = np.asarray(BOUNDS["low"]), np.asarray(BOUNDS["high"])
    radius = float(scene["ball"]["radius_m"])
    rng = np.random.default_rng(seed)
    shift = rng.random(5)
    rows, metadata, seen = [], [], set()
    hint_rows = [_hint(row) for row in hints]

    def constrained(value):
        value = np.asarray(value, float).copy()
        if board is not None:
            value[:3] = board
        if dropper is not None:
            value[:2] = dropper[:2]
            if board is None:
                hmin = max(low[2], dropper[2] - radius - high[4])
                hmax = min(high[2], dropper[2] - radius - low[4])
                if hmin > hmax:
                    return None
                value[2] = hmin + ((value[2] - low[2]) / (high[2] - low[2])) * (
                    hmax - hmin
                )
            value[4] = dropper[2] - value[2] - radius
        return value

    def add(value, meta, *, already_constrained=False):
        if len(rows) >= limit:
            return
        value = np.asarray(value, float) if already_constrained else constrained(value)
        if (
            value is None
            or not np.isfinite(value).all()
            or not ((value >= low) & (value <= high)).all()
        ):
            return
        if np.linalg.norm(xy - value[:2]) <= 0.15:
            return
        key = tuple(value)
        if key in seen:
            return
        seen.add(key)
        rows.append(value)
        metadata.append(
            dict(
                candidate_index=len(rows) - 1,
                proposal_version=PROPOSAL_VERSION,
                scope="Proposal only; no inherited capture or admission",
                **meta,
            )
        )

    for hint in hint_rows:
        source = hint["source_hint"]
        if np.array_equal(xy, source["bucket_xy_m"]):
            add(source["values"], dict(method="exact_source_geometry", **hint))

    # Preserve the relative shot geometry of every supplied recipe before
    # spending a small native budget on broad exploration. This does not carry
    # over capture/clearance labels: changed absolute arm poses need fresh gates.
    for hint in hint_rows:
        source = hint["source_hint"]
        if np.array_equal(xy, source["bucket_xy_m"]):
            continue  # Already inserted exactly; avoid subtract/add round-off.
        relative = np.array(source["values"], float)
        relative[:2] = xy + (np.asarray(source["values"][:2]) - source["bucket_xy_m"])
        add(
            relative,
            dict(method="relative_source_geometry", rotation_deg=0.0, **hint),
        )

    for index in range(1, limit * 128 + 1):
        if len(rows) == limit:
            break
        unit = (
            np.array([_halton(index, base) for base in (2, 3, 5, 7, 11)]) + shift
        ) % 1
        value = low + unit * (high - low)
        family = index % 3
        if hint_rows and family == 0:
            hint = hint_rows[(index // 3) % len(hint_rows)]
            source = hint["source_hint"]
            angle = 2 * math.pi * ((index // (3 * len(hint_rows))) % 16) / 16
            offset = np.asarray(source["values"][:2]) - source["bucket_xy_m"]
            rotation = np.array(
                [
                    [math.cos(angle), -math.sin(angle)],
                    [math.sin(angle), math.cos(angle)],
                ]
            )
            relative = np.array(source["values"], float)
            relative[:2] = xy + rotation @ offset
            if angle != 0.0 or not np.array_equal(xy, source["bucket_xy_m"]):
                add(
                    relative,
                    dict(
                        method="relative_source_geometry",
                        rotation_deg=math.degrees(angle),
                        **hint,
                    ),
                )
        if family == 1:
            distance, bearing = 0.16 + 1.04 * unit[0], 2 * math.pi * unit[1]
            value[:2] = xy - distance * np.array([math.cos(bearing), math.sin(bearing)])
            value = constrained(value)
            if value is None:
                continue
            gap = inverse_gap(
                scene, value[2], float(np.linalg.norm(xy - value[:2])), value[3]
            )
            if gap is not None:
                if dropper is None:
                    value[4] = gap * (0.75, 1.0, 1.25)[(index // 3) % 3]
                add(
                    value,
                    dict(
                        method="inverse_ballistic_ring",
                        energy_scale=(0.75, 1.0, 1.25)[(index // 3) % 3],
                        assumed_restitution=float(
                            scene["ball"].get("restitution", 0.65)
                        ),
                    ),
                    already_constrained=True,
                )
        else:
            add(value, dict(method="broad_space_filling"))
    return np.asarray(rows, dtype=float).reshape(-1, 5), metadata


def rank_row(row):
    """Native acquisition rank only; repeat/path qualification is independent."""
    valid = bool(row.get("native_release")) and not row.get("invalid")
    capture = valid and bool(row.get("route_success"))
    distance = row.get("closest_opening_distance_m", 2.0)
    distance = (
        float(distance)
        if isinstance(distance, Real) and math.isfinite(distance)
        else 2.0
    )
    return (
        capture,
        int(row.get("route_quality", 0)) if capture else 0,
        valid and bool(row.get("board_bounce")),
        -distance,
        str(row.get("id", "")),
    )
