"""Propose follow-up shots around recorded arm-clear misses at (1050, 25) mm."""
from __future__ import annotations

from collections import Counter
from itertools import count, cycle, product
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from .benchmark_verify import contained, digest, read, sha
from .calibrated_score import FORCE_NAMES, FORCE_THRESHOLD_N, validate_calibration
from .cell1050_plan import (
    BUCKET_XY_M,
    CONTEXT_ID,
    _coordinates_match,
    _joints,
    _mouth_projection,
    candidate_stream,
)
from .expanded_actions import WIDE_BOUNDS

PREVIOUS_VERSION = "cell1050-rebound-study-v1"
MAX_CLOSEST_SEEDS = 4
MAX_HEADING_DEG = 45.0


def previous_inputs(previous: Path, previous_audit: Path):
    """Verify the completed report and the ledger bound by its passed audit."""
    previous = Path(previous)
    paths = {
        "report": previous / "report.json",
        "audit": Path(previous_audit),
        "manifest": previous / "evidence-manifest.json",
        "ledger": previous / "rows.jsonl",
        "contract": previous / "contract.json",
    }
    binding = {key: sha(path) for key, path in paths.items()}
    report, audit, manifest, spec = (
        read(paths[key]) for key in ("report", "audit", "manifest", "contract")
    )
    if (
        spec.get("version") != PREVIOUS_VERSION
        or spec.get("fingerprint") != digest({k: v for k, v in spec.items() if k != "fingerprint"})
        or report.get("status") != "completed"
        or audit.get("status") != "passed"
        or audit.get("report_sha256") != binding["report"]
        or audit.get("evidence_manifest_sha256") != binding["manifest"]
        or report.get("evidence_manifest_sha256") != binding["manifest"]
        or manifest.get("rows.jsonl") != binding["ledger"]
        or manifest.get("contract.json") != binding["contract"]
    ):
        raise ValueError("The completed prior study and passed audit must bind the same evidence")

    with paths["ledger"].open(encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream]
    search = [row for row in rows if row["phase"] == "search"]
    routes = dict(Counter(row["experimental_route"]["route_class"] for row in search))
    if (
        not search
        or len({row["id"] for row in rows}) != len(rows)
        or len(rows) != audit.get("total_releases")
        or len(search) != spec.get("target_releases")
        or len(search) != report.get("search_releases")
        or len(search) != audit.get("search_releases")
        or routes != report.get("route_counts")
        or routes != audit.get("route_counts")
        or any(
            row["native_release"] is not True or not _coordinates_match(row["bucket_xy_m"])
            for row in search
        )
    ):
        raise ValueError("The prior ledger does not match the audited fixed-bucket search")
    return spec, report, rows, manifest, binding


def geometry_key(row):
    """Identify a shot geometry independently of which arm joints realize it."""
    coordinates = tuple(round(float(v), 4 if i == 3 else 6) for i, v in enumerate(row["values"]))
    angles = tuple(
        round((float(row.get(name, 0.0)) + 180.0) % 360.0 - 180.0, 4)
        for name in ("board_heading_offset_deg", "board_twist_deg")
    )
    return coordinates + angles


def _recorded_departure(previous, row, manifest, calibration, radius):
    """Measure heading and range after the last board contact before another surface."""
    name = (Path(row["adapter_directory"]) / row["native_cache_file"]).as_posix()
    path = contained(previous, name)
    if manifest.get(name) != row["native_cache_sha256"] or sha(path) != row["native_cache_sha256"]:
        raise ValueError(f"The prior native recording changed for {row['id']}")
    column = row["native_cache_row"]
    with np.load(path, allow_pickle=False) as cache:
        identifiers = cache["record_ids"]
        if type(column) is not int or not 0 <= column < len(identifiers) or identifiers[column] != row["id"]:
            raise ValueError(f"The prior native recording does not contain {row['id']}")
        stop = row["physics_steps"]
        forces = cache["forces_N"][:stop, column]
        contacts = forces > FORCE_THRESHOLD_N
        if contacts[:, FORCE_NAMES.index("robot")].any():
            raise ValueError(f"Expected an arm-clear native recording for {row['id']}")
        other = np.flatnonzero(contacts[:, 1:].any(axis=1))
        board = np.flatnonzero(contacts[:int(other[0]) if len(other) else stop, 0])
        if not len(board):
            raise ValueError(f"No board rebound before other contacts for {row['id']}")
        tick = int(board[-1]) + 1
        position = np.asarray(cache["position_m"][tick, column], dtype=float)
        velocity = np.asarray(cache["velocity_m_s"][tick, column], dtype=float)
        departure_s = float(cache["time_s"][tick])
    toward_bucket = np.asarray(BUCKET_XY_M) - position[:2]
    length = float(np.linalg.norm(toward_bucket))
    if length == 0:
        raise ValueError(f"Board departure is already centered over the bucket for {row['id']}")
    toward_bucket /= length
    across = np.array([-toward_bucket[1], toward_bucket[0]])
    projection = _mouth_projection(
        position, velocity, np.asarray(BUCKET_XY_M),
        np.asarray(calibration["compiled_planes"]), calibration["compiled_rim_m"], radius,
    )
    if projection is not None:
        delta = np.asarray(projection["xy_m"]) - np.asarray(BUCKET_XY_M)
        projection["along_error_m"] = float(delta @ toward_bucket)
        projection["across_error_m"] = float(delta @ across)
    return {
        "departure_time_s": departure_s,
        "departure_position_m": position.tolist(),
        "departure_velocity_m_s": velocity.tolist(),
        "signed_heading_error_deg": math.degrees(math.atan2(velocity[:2] @ across, velocity[:2] @ toward_bucket)),
        "unobstructed_projection": projection,
    }


def load_followup_seeds(previous: Path, previous_audit: Path):
    """Choose closest distinct arm-clear misses and preserve their measured provenance."""
    previous = Path(previous)
    spec, report, rows, manifest, binding = previous_inputs(previous, previous_audit)
    calibration_path = previous / "calibration.json"
    if sha(calibration_path) != manifest.get("calibration.json"):
        raise ValueError("The prior capture calibration changed")
    calibration = validate_calibration(read(calibration_path)["score_config"])
    radius = float(spec["scene"]["ball"]["radius_m"])
    arm_clear = [
        row for row in rows
        if row["phase"] == "search" and not row["experimental_route"]["robot_contact_steps"]
    ]
    if len(arm_clear) != report["search_releases"] - report["contact_counts"]["robot_contacts"]:
        raise ValueError("The prior arm-clear count differs from its audited report")
    misses = [
        row for row in arm_clear
        if row["board_bounce"] and not row["invalid"] and not row["invalid_physics"]
        and row["experimental_route"]["route_class"] == "not_captured"
        and not row["experimental_route"]["floor_contact_steps"]
        and not row["experimental_route"]["unattributed_contact_steps"]
        and math.isfinite(row["closest_opening_distance_m"])
    ]
    ranked = sorted(misses, key=lambda row: (row["closest_opening_distance_m"], row["id"]))
    if not ranked:
        raise ValueError("The prior study has no usable arm-clear misses")
    chosen, used = [], set()
    for row in ranked:
        key = geometry_key(row)
        if key not in used:
            used.add(key)
            chosen.append((row, "closest_arm_clear"))
        if len(chosen) == MAX_CLOSEST_SEEDS:
            break
    if all(row["contact_peaks_N"]["bucket"] > FORCE_THRESHOLD_N for row, _ in chosen):
        no_touch = next(
            (row for row in ranked if row["contact_peaks_N"]["bucket"] <= FORCE_THRESHOLD_N),
            None,
        )
        if no_touch is not None:
            chosen.append((no_touch, "closest_without_bucket_contact"))

    selected_ids = {row["id"] for row, _ in chosen}
    row_hashes = {}
    with (previous / "rows.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if row["id"] in selected_ids:
                row_hashes[row["id"]] = hashlib.sha256(line.rstrip("\n").encode()).hexdigest()
    if sha(previous / "rows.jsonl") != binding["ledger"]:
        raise ValueError("The prior ledger changed during seed selection")

    seeds = []
    for row, selection in chosen:
        trace = _recorded_departure(previous, row, manifest, calibration, radius)
        hint = {key: row[key] for key in ("id", "values", "bucket_xy_m")}
        hint.update({key: _joints(row, key).tolist() for key in ("q10", "q14")})
        seeds.append({
            "hint": hint,
            "source_study": previous.name,
            "source_selection": selection,
            "source_row_sha256": row_hashes[row["id"]],
            "source_trace": f"{row['adapter_directory']}/{row['native_cache_file']}",
            "source_trace_sha256": row["native_cache_sha256"],
            "source_trace_column": row["native_cache_row"],
            "source_proposal_id": row["proposal_id"],
            "closest_opening_distance_m": row["closest_opening_distance_m"],
            "board_heading_offset_deg": row["board_heading_offset_deg"],
            "board_twist_deg": row["board_twist_deg"],
            "prediction": trace,
        })
    return seeds, binding


def _signed_heading_step(source):
    error = source["prediction"]["signed_heading_error_deg"]
    return math.copysign(4.0 if abs(error) < 10 else 5.0, -error)


def _candidate(source, values, heading, twist, group):
    hint = source["hint"]
    return {
        "context_id": CONTEXT_ID,
        "bucket_xy_m": list(hint["bucket_xy_m"]),
        "values": np.asarray(values, dtype=float).tolist(),
        "board_heading_offset_deg": float(heading),
        "board_twist_deg": float(twist),
        "group": group,
        "seed_id": hint["id"],
        "seed_q10": list(hint["q10"]),
        "seed_q14": list(hint["q14"]),
        "hint": hint,
        "source_row_sha256": source["source_row_sha256"],
        "source_trace_sha256": source["source_trace_sha256"],
    }


def _controlled_probes(sources):
    for selection in ("closest_arm_clear", "closest_without_bucket_contact"):
        group = [source for source in sources if source["source_selection"] == selection]
        for offset, gap_level in product((1, 2, 3), range(3)):
            for source in group:
                values = np.array(source["hint"]["values"], dtype=float)
                if selection == "closest_without_bucket_contact":
                    gaps = (0.0, .015, .035)
                else:
                    gaps = (.015, .035, .060) if values[2] >= .50 else (.020, .040, .070)
                values[4] += gaps[gap_level]
                heading = source["board_heading_offset_deg"] + _signed_heading_step(source) * offset
                yield _candidate(source, values, heading, source["board_twist_deg"], "arm_clear_heading_gap")
    # Single-variable comparisons indicate whether heading, gap or their combination matters.
    for source in sources:
        heading, twist = source["board_heading_offset_deg"], source["board_twist_deg"]
        values = np.array(source["hint"]["values"], dtype=float)
        for factor in (1, 2, 3):
            yield _candidate(source, values, heading + factor * _signed_heading_step(source), twist, "arm_clear_heading_only")
        for gap in (.020, .040, .060):
            adjusted = values.copy()
            adjusted[4] += gap
            yield _candidate(source, adjusted, heading, twist, "arm_clear_gap_only")


def _local_and_broader(sources, fallback_seeds, seed):
    rng = np.random.default_rng(seed)
    broader = candidate_stream(fallback_seeds, seed=seed) if fallback_seeds else None
    local = cycle(sources)
    for index in count():
        if broader is not None and index % 5 == 4:
            yield next(broader)
            continue
        source = next(local)
        values = np.array(source["hint"]["values"], dtype=float)
        values += rng.uniform(-1, 1, 5) * [.035, .035, .035, 4.0, .055]
        values[4] += .035
        heading = source["board_heading_offset_deg"] + 2 * _signed_heading_step(source) + rng.uniform(-15, 15)
        twist = source["board_twist_deg"] + rng.choice([0, 0, 0, -12, 12, -24, 24])
        yield _candidate(source, values, heading, twist, "arm_clear_local")


def followup_candidates(seeds, fallback_seeds, *, seed, exclude_geometry=()):
    """Run matched heading/gap probes first, then mix local and wider search."""
    if not seeds or any(not _coordinates_match(source["hint"]["bucket_xy_m"]) for source in seeds):
        raise ValueError("Arm-clear seeds must belong to the fixed bucket position")
    low, high = np.asarray(WIDE_BOUNDS["low"]), np.asarray(WIDE_BOUNDS["high"])
    used = set(exclude_geometry)
    used.update(geometry_key(dict(source, values=source["hint"]["values"])) for source in seeds)
    index = 0
    streams = _controlled_probes(seeds), _local_and_broader(seeds, fallback_seeds, seed)
    for stream in streams:
        for candidate in stream:
            values = np.asarray(candidate["values"], dtype=float)
            if (
                not ((low <= values) & (values <= high)).all()
                or np.linalg.norm(values[:2] - BUCKET_XY_M) <= .15
                or abs(candidate["board_heading_offset_deg"]) > MAX_HEADING_DEG
            ):
                continue
            key = geometry_key(candidate)
            if key in used:
                continue
            used.add(key)
            yield dict(candidate, id=f"cell1050-followup-p{index:06d}", proposal_index=index)
            index += 1
