"""Rank archived rebounds and propose new shots for the fixed (1050, 25) mm cell."""
from __future__ import annotations

from collections import defaultdict
import hashlib
from itertools import count
import json
import math
from pathlib import Path

import numpy as np

from .benchmark_verify import contained, digest, read, sha
from .calibrated_score import FORCE_NAMES, FORCE_THRESHOLD_N, validate_calibration
from .expanded_actions import WIDE_BOUNDS

VERSION = "sircus-cell1050-measured-rebound-plan-v1"
BUCKET_XY_M = (1.05, 0.025)
CONTEXT_ID = "fixed35-dev-grid-03-05"
SEARCH_RELEASE_LIMIT = 2000
SEED = 202611121050
GROUPS = (
    "bucket_facing",
    "steered_clearance",
    "board_twist",
    "robot_bank",
    "steered_clearance",
    "robot_bank",
    "oblique_board",
    "board_twist",
)


def _coordinates_match(xy):
    return isinstance(xy, (tuple, list)) and len(xy) == 2 and all(
        abs(float(value) - expected) < 1e-9
        for value, expected in zip(xy, BUCKET_XY_M)
    )


def load_history(output_root):
    """Return lightweight lineage and command records for saved target releases."""
    history = []
    for ledger in sorted(Path(output_root).glob("*/rows.jsonl")):
        with ledger.open(encoding="utf-8") as stream:
            for line in stream:
                row = json.loads(line)
                if not row.get("native_release") or not _coordinates_match(row.get("bucket_xy_m")):
                    continue
                fields = (
                    "id", "values", "bucket_xy_m", "q10", "q14", "selected_q10", "selected_q14",
                    "board_heading_offset_deg", "board_twist_deg", "native_cache_file",
                    "native_cache_row", "native_cache_sha256", "adapter_directory",
                    "board_bounce", "invalid", "invalid_physics", "board_normal",
                )
                record = {key: row[key] for key in fields if key in row}
                record.update(study=ledger.parent.name, source_row_sha256=hashlib.sha256(line.encode()).hexdigest())
                history.append(record)
    return history


def _joints(row, name):
    value = row.get("selected_" + name)
    if value is None:
        value = row[name]
    vector = np.asarray(value, dtype=float)
    if vector.shape != (6,) or not np.isfinite(vector).all():
        raise ValueError(f"Expected a finite six-joint command for {name}")
    return vector


def command_key(row):
    """Deduplicate pose commands, including equivalent whole joint turns."""
    values = np.asarray(row["values"], dtype=float)
    if values.shape != (5,) or not np.isfinite(values).all():
        raise ValueError("Expected five finite shot coordinates")
    angles = (float(row.get("board_heading_offset_deg", 0)), float(row.get("board_twist_deg", 0)))
    if not all(math.isfinite(angle) for angle in angles) or not _coordinates_match(row["bucket_xy_m"]):
        raise ValueError("Expected finite board angles at the fixed bucket cell")
    radians = np.concatenate([_joints(row, name) for name in ("q10", "q14")])
    joints = np.round((radians + np.pi) % (2 * np.pi) - np.pi, 4)
    # Rounding must map both sides of the periodic boundary to the same bin.
    joints[joints == round(np.pi, 4)] = round(-np.pi, 4)
    coordinates = [round(float(v), 4 if i == 3 else 6) for i, v in enumerate(values)]
    return digest({
        "bucket_xy_m": list(BUCKET_XY_M),
        "values": [v + 0.0 for v in coordinates],
        "board_angles_deg": [round((a + 180) % 360 - 180, 4) + 0.0 for a in angles],
        "joints_rad": (joints + 0.0).tolist(),
    })


def _mouth_projection(position, velocity, bucket_xy, planes, rim_z, radius):
    """Continue the measured ball state under gravity to the whole-ball entry height."""
    gravity = 9.81
    target_z = rim_z - radius
    discriminant = float(velocity[2] ** 2 + 2 * gravity * (position[2] - target_z))
    if discriminant <= 0:
        return None
    t = (float(velocity[2]) + math.sqrt(discriminant)) / gravity
    if t <= 0:
        return None
    xy = position[:2] + velocity[:2] * t
    local = np.r_[xy - bucket_xy, target_z]
    return {
        "flight_s": t,
        "xy_m": xy.tolist(),
        "side_clearance_m": float(np.min(planes[:, :3] @ local + planes[:, 3]) - radius),
        "center_error_m": float(np.linalg.norm(local[:2])),
    }


def _robot_events(forces, first_robot, last):
    """Yield ends of robot-contact runs, with adjacent force intervals aligned to state indices."""
    active = forces[first_robot:last, FORCE_NAMES.index("robot")] > FORCE_THRESHOLD_N
    ends = np.flatnonzero(active & ~np.r_[active[1:], False])
    for index in ends:
        yield first_robot + int(index) + 1


def _measure_seed(row, positions, velocities, forces, times, planes, rim_z, radius):
    board, bucket, table, floor, robot, unattributed = (
        forces[:, FORCE_NAMES.index(name)] > FORCE_THRESHOLD_N for name in FORCE_NAMES
    )
    robot_steps = np.flatnonzero(robot)
    if not len(robot_steps):
        return None
    first_robot = int(robot_steps[0])
    if not board[:first_robot].any() or (bucket | table | floor | unattributed)[:first_robot].any():
        return None
    pre = _mouth_projection(
        positions[first_robot], velocities[first_robot], row["bucket_xy_m"], planes, rim_z, radius
    )
    if pre is None:
        return None

    # Use only flight immediately after a robot impulse and before the first
    # table, floor or bucket contact. Record later bucket touches separately.
    subsequent_other = np.flatnonzero((bucket | table | floor | unattributed)[first_robot:])
    cutoff = first_robot + int(subsequent_other[0]) if len(subsequent_other) else len(forces)
    post = []
    for tick in _robot_events(forces, first_robot, cutoff):
        if tick == cutoff and cutoff < len(forces) and robot[cutoff]:
            continue
        projection = _mouth_projection(
            positions[tick], velocities[tick], row["bucket_xy_m"], planes, rim_z, radius
        )
        if projection is not None:
            post.append(dict(projection, state_index=tick))
    best_post = max(post, key=lambda projection: projection["side_clearance_m"], default=None)
    touches = np.flatnonzero(bucket[first_robot + 1:])
    after_robot = positions[first_robot + 1:]
    rim_band = after_robot[np.abs(after_robot[:, 2] - rim_z) <= radius]
    closest = min(
        np.linalg.norm(rim_band[:, :2] - row["bucket_xy_m"], axis=1),
        default=float("inf"),
    )
    return {
        "first_robot_interval": first_robot,
        "first_robot_s": float(times[first_robot]),
        "first_bucket_after_robot_s": float(times[first_robot + 1 + touches[0]]) if len(touches) else None,
        "observed_rim_band_center_error_m": float(closest) if math.isfinite(closest) else None,
        "pre_robot_projection": pre,
        "post_robot_projection": best_post,
    }


def _diverse_settings(ranked, limit):
    """Use distinct numeric shots and defer almost identical board placements."""
    selected, postponed, seen = [], [], set()
    scales = np.array([0.022, 0.022, 0.025, 3.0, 0.025])
    for row in ranked:
        values = tuple(row["hint"]["values"])
        if values in seen:
            continue
        seen.add(values)
        value = np.asarray(values)
        if any(
            np.linalg.norm((value - previous["hint"]["values"]) / scales) < 0.9
            for previous in selected
        ):
            postponed.append(row)
        else:
            selected.append(row)
            if len(selected) == limit:
                break
    return (selected + postponed)[:limit]


def load_seed_evidence(study, history_rows, max_seeds=24):
    """Select warm IK hints from hash-matched dense traces of the previous search."""
    study = Path(study)
    if max_seeds < 2:
        raise ValueError("Both clearance and robot bank seeds are required")
    contract = read(study / "contract.json")
    calibration = validate_calibration(read(study / "calibration.json")["score_config"])
    radius = float(contract["scene"]["ball"]["radius_m"])
    rim_z = float(calibration["compiled_rim_m"])
    planes = np.asarray(calibration["compiled_planes"])
    archives = defaultdict(list)
    for row in history_rows:
        if (
            row["study"] == study.name
            and row.get("board_bounce")
            and not row.get("invalid")
            and not row.get("invalid_physics")
        ):
            path = contained(study, str(Path(row["adapter_directory"]) / row["native_cache_file"]))
            archives[path].append(row)
    measured = []
    for path, rows in archives.items():
        packet_hash = sha(path)
        if any(row["native_cache_sha256"] != packet_hash for row in rows):
            raise ValueError(f"Archived trace hash changed in {path.name}")
        with np.load(path, allow_pickle=False) as trace:
            positions, velocities, forces = (trace[name] for name in ("position_m", "velocity_m_s", "forces_N"))
            times, identifiers = trace["time_s"], trace["record_ids"]
        for row in rows:
            column = int(row["native_cache_row"])
            if identifiers[column] != row["id"]:
                raise ValueError(f"Archived trace identity changed for {row['id']}")
            observation = _measure_seed(
                row, positions[:, column], velocities[:, column], forces[:, column], times, planes, rim_z, radius
            )
            if observation is not None:
                hint = {key: row[key] for key in ("id", "values", "bucket_xy_m")}
                hint.update({name: _joints(row, name).tolist() for name in ("q10", "q14")})
                measured.append({
                    "hint": hint,
                    "source_study": study.name,
                    "source_row_sha256": row["source_row_sha256"],
                    "source_trace": f"{row['adapter_directory']}/{row['native_cache_file']}",
                    "source_trace_sha256": packet_hash,
                    "source_trace_column": column,
                    "prediction": observation,
                })
    clearance = sorted(
        measured,
        key=lambda row: (-row["prediction"]["pre_robot_projection"]["side_clearance_m"], row["hint"]["id"]),
    )
    bank = [row for row in measured if (
        row["prediction"]["first_bucket_after_robot_s"] is not None
        or row["prediction"]["post_robot_projection"] is not None
    )]

    def bank_rank(row):
        observation = row["prediction"]
        contact = observation["first_bucket_after_robot_s"]
        projection = observation["post_robot_projection"]
        clearance = projection["side_clearance_m"] if projection else -math.inf
        rim_error = observation["observed_rim_band_center_error_m"]
        return (contact is None, rim_error if rim_error is not None else math.inf, -clearance, row["hint"]["id"])

    bank.sort(key=bank_rank)
    bank_limit = max(1, max_seeds // 3)
    selected = [dict(row, group="clearance") for row in _diverse_settings(clearance, max_seeds - bank_limit)]
    selected += [dict(row, group="robot_bank") for row in _diverse_settings(bank, bank_limit)]
    if {row["group"] for row in selected} != {"clearance", "robot_bank"}:
        raise ValueError("Saved dense evidence did not yield both clearance and robot bank seeds")
    return selected


def _proposal(seed, group, ordinal, pool_size, rng):
    hint = seed["hint"]
    values = np.asarray(hint["values"], dtype=float).copy()
    cycle = ordinal // pool_size
    heading = 0.0
    twist = 0.0
    if group == "bucket_facing" and cycle == 0:
        return values, heading, twist

    if group == "board_twist":
        # The first series changes only rotation about the board normal.
        twists = (-15, 15, -30, 30, -45, 45, -70, 70, -110, 110, 180)
        twist = float(twists[cycle % len(twists)])
        if cycle < len(twists):
            return values, heading, twist
        values += rng.uniform(-1, 1, 5) * [.028, .032, .025, 2.0, .024]
        heading = float(rng.uniform(-4, 4))
    elif group == "robot_bank":
        # A saved rim graze may have a narrow local window. Bracket it first
        # with single-coordinate moves, then explore wider combinations.
        sign = (-1, 1)[cycle % 2]
        if cycle < 4:
            return values, sign * (1.0, 3.0)[cycle // 2], twist
        if cycle < 14:
            axis = (cycle - 4) // 2
            values[axis] += sign * (.005, .005, .004, .5, .005)[axis]
            return values, heading, twist
        if cycle < 16:
            return values, heading, sign * 10.0
        values += rng.uniform(-1, 1, 5) * [.026, .035, .038, 3.0, .030]
        heading = float(rng.choice([-1, 1]) * rng.uniform(1.5, 12))
        twist = float(rng.choice([0, 0, -18, 18, -36, 36]))
    else:
        width = .085 if group == "oblique_board" else .045
        if group == "oblique_board":
            heading = float(rng.choice([-1, 1]) * rng.uniform(12, 45))
        elif group == "steered_clearance":
            heading = float(rng.choice([-1, 1]) * rng.uniform(1.5, 8))
        values += rng.uniform(-1, 1, 5) * [width, width, .7 * width, width * 100, .15 * width]
        # Match the local flight distance approximately when moving the board.
        # The real trajectory, including tilt, spin and contacts, comes from native drops.
        target = np.asarray(hint["bucket_xy_m"])
        ratio = np.linalg.norm(target - values[:2]) / np.linalg.norm(target - hint["values"][:2])
        values[4] *= ratio ** 2
    return values, heading, twist


def candidate_stream(seeds, *, seed=SEED):
    """Yield a fixed, interleaved order without using any new flight outcomes."""
    by_group = {group: [row for row in seeds if row["group"] == group] for group in ("clearance", "robot_bank")}
    if not all(by_group.values()):
        raise ValueError("Provide archived seeds for both intended route families")
    rng = np.random.default_rng(seed)
    used = set()
    ordinals = defaultdict(int)
    low, high = np.asarray(WIDE_BOUNDS["low"]), np.asarray(WIDE_BOUNDS["high"])
    for index in count():
        group = GROUPS[index % len(GROUPS)]
        ordinal = ordinals[group]
        ordinals[group] += 1
        pool = by_group["robot_bank"] if group == "robot_bank" else by_group["clearance"]
        if group == "bucket_facing":
            pool = by_group["clearance"] + by_group["robot_bank"]
        source = pool[ordinal % len(pool)]
        hint = source["hint"]
        values, heading, twist = _proposal(source, group, ordinal, len(pool), rng)
        if not (low <= values).all() or not (values <= high).all():
            continue
        if np.linalg.norm(values[:2] - hint["bucket_xy_m"]) <= .15:
            continue
        geometry = (*values.tolist(), heading, twist)
        if geometry in used:
            continue
        used.add(geometry)
        yield {
            "id": f"cell1050-p{index:06d}",
            "proposal_index": index,
            "context_id": CONTEXT_ID,
            "bucket_xy_m": list(hint["bucket_xy_m"]),
            "values": values.tolist(),
            "board_heading_offset_deg": heading,
            "board_twist_deg": twist,
            "group": group,
            "seed_id": hint["id"],
            "seed_q10": list(hint["q10"]),
            "seed_q14": list(hint["q14"]),
            "hint": hint,
        }
