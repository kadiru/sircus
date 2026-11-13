"""Build branch-aware training records from saved native64 grid drops."""
from __future__ import annotations

from collections import Counter
import hashlib
import json
import math
from pathlib import Path

from .benchmark_verify import read, sha
from .calibrated_score import FORCE_THRESHOLD_N
from .corrected_contract import grid_layouts
from .parallel_capture_grid_plan import BASELINE_STUDIES, FIXED_BASE_BLOCKS, _archive_manifest
from .table_assisted_score import ROUTE_VERSION, VERSION as SCORING_VERSION


VERSION = "sircus-intergrid-native64-dataset-v1"
GRID_STUDY_VERSION = "parallel-capture-grid-v1"
NATIVE_PROTOCOL = {"substeps": 64, "board_mode": "articulated", "gravity_mode": "passive_only"}


def _finite_vector(value, count):
    return isinstance(value, list) and len(value) == count and all(
        type(number) in (float, int) and math.isfinite(number) for number in value
    )


def _bucket_key(xy):
    if not _finite_vector(xy, 2):
        return None
    # The original grid is written with either literal or linspace floats.
    return tuple(round(coordinate, 10) for coordinate in xy)


def _grid():
    return {
        _bucket_key(layout["bucket_xy_m"]): (
            "fixed35-" + layout["id"], layout["bucket_xy_m"],
        )
        for layout in grid_layouts()
        if "fixed35-" + layout["id"] not in FIXED_BASE_BLOCKS
    }


def _grid_audit(study, audit_path, manifest, binding, grid):
    report = read(study / "report.json")
    audit = read(audit_path)
    contract = read(study / "contract.json")
    planned = contract.get("cells")
    if not isinstance(planned, dict) or len(planned) != 13 or any(
        grid.get(_bucket_key(xy)) != (cell, xy) for cell, xy in planned.items()
    ):
        raise ValueError("The new sweep must contain thirteen canonical bucket positions")
    if (
        contract.get("version") != GRID_STUDY_VERSION
        or contract.get("target_per_cell") != 1000
        or manifest.get("contract.json") != sha(study / "contract.json")
        or audit.get("status") != "passed"
        or type(audit.get("checks_count")) is not int
        or audit["checks_count"] <= 0
        or audit.get("report_sha256") != binding["report_sha256"]
        or audit.get("evidence_manifest_sha256") != binding["manifest_sha256"]
        or set(audit["cells"]) != set(report["cells"]) or set(audit["cells"]) != set(planned)
        or audit.get("search_releases") != 13000
        or report.get("search_releases") != 13000
        or audit.get("total_releases") != 13018
    ):
        raise ValueError("A completed and independently audited 13,000-drop sweep is required")
    for cell, result in audit["cells"].items():
        expected = report["cells"][cell]
        if result.get("search_releases") != 1000 or any(expected.get(key) != result.get(key) for key in (
            "search_releases", "route_counts", "attempts", "preparation_declines",
        )):
            raise ValueError(f"Report and independent audit differ at {cell}")
    return audit


def _reject(row, bucket):
    if (str(row.get("phase", "")).startswith("controls") or row.get("method") == "control"
            or str(row.get("role", "")).endswith("_control")):
        return "control"
    if bucket is None:
        return "outside_usable_grid"
    if row.get("native_release") is not True:
        return "setup_declined"
    if row.get("invalid_physics") or row.get("invalid"):
        return "invalid"
    if (row.get("contact_protocol") != NATIVE_PROTOCOL
            or row.get("route_version") != ROUTE_VERSION
            or row.get("scoring_version") != SCORING_VERSION):
        return "different_physics_or_score"
    return None


def _route(row, robot_contact):
    capture = row["route_success"]
    quality = row["route_quality"]
    retention = row["max_retention_s"]
    distance = row["closest_opening_distance_m"]
    if (
        type(capture) is not bool or row.get("complete_success") is not capture
        or type(quality) is not int or quality not in (0, 1, 2)
        or not _finite_vector([retention, distance], 2) or not 0 <= retention <= 1.000000001
        or distance < 0 or row.get("scalar_score_audit_passed") is not True
        or (capture and (robot_contact or not row["board_bounce"] or not row["whole_ball_entry"]
                        or retention < 1 - 1e-9 or quality < 1))
    ):
        raise ValueError(f"Invalid standard capture evidence for {row['id']}")
    experimental = row.get("experimental_route", {})
    arm_capture = experimental.get("robot_assisted_capture") is True
    if arm_capture and (capture or not robot_contact or not row["whole_ball_entry"] or retention < 1 - 1e-9):
        raise ValueError(f"Invalid experimental robot-assisted evidence for {row['id']}")
    if capture:
        route = "standard_capture"
    elif arm_capture:
        route = "robot_assisted_capture"
    else:
        route = experimental.get("route_class", "not_captured")
    return capture, arm_capture, route


def _observation(row, directory, raw, manifest, bucket):
    name = row["id"]
    context, xy = bucket
    vectors = {key: row.get(key) for key in ("values", "q10", "q14")}
    angles = [row.get(key, 0.0) for key in ("board_heading_offset_deg", "board_twist_deg")]
    if not (
        _finite_vector(vectors["values"], 5) and _finite_vector(vectors["q10"], 6)
        and _finite_vector(vectors["q14"], 6) and _finite_vector(angles, 2)
    ):
        raise ValueError(f"Missing exact shot or joint coordinates for {directory.name}/{name}")
    for joint in ("q10", "q14"):
        if row.get("selected_" + joint, vectors[joint]) != vectors[joint]:
            raise ValueError(f"Selected and measured joint commands disagree for {directory.name}/{name}")
    file, adapter = row["native_cache_file"], row["adapter_directory"]
    archive = (Path(adapter) / file).as_posix()
    trace_hash, column = row["native_cache_sha256"], row["native_cache_row"]
    if manifest.get(archive) != trace_hash or type(column) is not int or column < 0:
        raise ValueError(f"Unbound native trace for {directory.name}/{name}")
    peak = row["contact_peaks_N"]["robot"]
    if not _finite_vector([peak], 1) or peak < 0:
        raise ValueError(f"Invalid robot contact force for {directory.name}/{name}")
    robot_contact = peak > FORCE_THRESHOLD_N
    capture, arm_capture, route = _route(row, robot_contact)
    lineage = {
        "source_study": directory.name,
        "source_record_id": name,
        "source_row_sha256": hashlib.sha256(raw.rstrip(b"\r\n")).hexdigest(),
        "source_context_id": row["context_id"],
        "source_bucket_xy_m": row["bucket_xy_m"],
        "source_trace": archive,
        "source_trace_sha256": trace_hash,
        "source_trace_column": column,
        "phase": row["phase"],
        "method": row.get("method"),
        "group": row.get("group"),
        "seed_id": row.get("seed_id"),
        "seed_source_study": row.get("source_study"),
        "seed_source_row_sha256": row.get("source_row_sha256"),
        "seed_source_trace_sha256": row.get("source_trace_sha256"),
        "source_teacher_id": row.get("source_teacher_id"),
        "proposal_id": row.get("proposal_id"),
        "selected_from_id": row.get("selected_from_id"),
        "connection_file": row.get("connection_file"),
        "connection_sha256": row.get("connection_sha256"),
        "release_behavior": row.get("release_behavior"),
        "standard_capture": capture,
        "experimental_robot_assisted_capture": arm_capture,
        "route_class": route,
        "route_quality": row["route_quality"],
        "board_bounce": row["board_bounce"],
        "whole_ball_entry": row["whole_ball_entry"],
        "max_retention_s": row["max_retention_s"],
        "closest_opening_distance_m": row["closest_opening_distance_m"],
        "robot_contact": robot_contact,
    }
    shot = dict(lineage, id=f"{directory.name}:{name}", context_id=context, bucket_xy_m=xy,
                **vectors, board_heading_offset_deg=angles[0], board_twist_deg=angles[1],
                route_success=capture, native_release=True, invalid=None, invalid_physics=False,
                route_version=ROUTE_VERSION, scoring_version=SCORING_VERSION,
                lineage=[lineage], observations=dict(
                    standard_capture=int(capture), standard_miss=int(not capture),
                    robot_assisted_capture=int(arm_capture),
                ), label_conflict=False)
    # Exact pose values and joint commands are significant, even for nearby settings.
    key = context, tuple(vectors["values"]), tuple(angles), tuple(vectors["q10"]), tuple(vectors["q14"])
    return key, shot


def _add_observation(records, key, shot):
    previous = records.get(key)
    if previous is None:
        records[key] = shot
        return
    # Retain the first miss as the representative when the exact command has
    # conflicting results. Every original label is still carried in lineage.
    selected = shot if previous["standard_capture"] and not shot["standard_capture"] else previous
    counts = {label: previous["observations"][label] + count for label, count in shot["observations"].items()}
    records[key] = dict(
        selected,
        observations=counts,
        lineage=previous["lineage"] + shot["lineage"],
        label_conflict=bool(counts["standard_capture"] and counts["standard_miss"]),
    )


def load_dataset(history_root, study, audit_path):
    """Return compact rows and provenance for all 33 exact bucket positions.

    Each row represents one exact shot geometry and pair of joint commands.
    Repeated observations remain in its lineage. A command with both a
    standard capture and a miss gets a conservative negative training label.
    """
    history_root, study, audit_path = map(Path, (history_root, study, audit_path))
    if study.name in BASELINE_STUDIES:
        raise ValueError("The new sweep cannot duplicate a historical input")
    grid = _grid()
    if len(grid) != 33:
        raise ValueError("Expected 33 available fixed bucket positions")
    records, sources, skipped = {}, {}, Counter()
    new_counts, new_routes = Counter(), {}
    new_audit = None
    for directory in (*[history_root / name for name in BASELINE_STUDIES], study):
        ledger = directory / "rows.jsonl"
        manifest, binding = _archive_manifest(directory, sha(ledger))
        if directory == study:
            new_audit = _grid_audit(study, audit_path, manifest, binding, grid)
        counts = Counter()
        identifiers = set()
        with ledger.open("rb") as stream:
            for raw in stream:
                row = json.loads(raw)
                if row["id"] in identifiers:
                    raise ValueError(f"Duplicate source record in {directory.name}: {row['id']}")
                identifiers.add(row["id"])
                bucket = grid.get(_bucket_key(row.get("bucket_xy_m")))
                if directory == study and row["phase"] == "search":
                    if (bucket is None or row["context_id"] != bucket[0]
                            or row["bucket_xy_m"] != bucket[1] or row.get("native_release") is not True):
                        raise ValueError(f"New sweep row changed its exact grid position or release: {row['id']}")
                    new_counts[bucket[0]] += 1
                    new_routes.setdefault(bucket[0], Counter())[row["experimental_route"]["route_class"]] += 1
                reason = _reject(row, bucket)
                if reason is not None:
                    counts[reason] += 1
                    continue
                key, shot = _observation(row, directory, raw, manifest, bucket)
                _add_observation(records, key, shot)
                counts["valid_observations"] += 1
                counts["standard_captures"] += int(shot["standard_capture"])
        sources[directory.name] = dict(binding, counts=dict(counts))
        skipped.update({key: count for key, count in counts.items() if key not in (
            "valid_observations", "standard_captures",
        )})

    for cell, result in new_audit["cells"].items():
        if new_counts[cell] != 1000 or dict(new_routes.get(cell, {})) != result["route_counts"]:
            raise ValueError(f"New observation ledger disagrees with the independent audit at {cell}")
    rows = list(records.values())
    cell_counts = {context: Counter() for context, _ in grid.values()}
    for row in rows:
        counts = cell_counts[row["context_id"]]
        counts["commands"] += 1
        counts["standard_captures"] += int(row["standard_capture"])
        counts["standard_misses"] += int(not row["standard_capture"])
        counts["robot_assisted_commands"] += int(bool(row["observations"]["robot_assisted_capture"]))
        counts["label_conflicts"] += int(row["label_conflict"])
        counts["valid_observations"] += len(row["lineage"])
    if any(not count["standard_captures"] for count in cell_counts.values()):
        raise ValueError("All 33 training positions require at least one valid positive command")
    sources[study.name]["audit_sha256"] = sha(audit_path)
    provenance = dict(
        version=VERSION,
        scoring_version=SCORING_VERSION,
        native_protocol=dict(NATIVE_PROTOCOL),
        sources=sources,
        cells={cell: dict(counts) for cell, counts in cell_counts.items()},
        excluded=dict(skipped),
        new_search_releases=sum(new_counts.values()),
        new_standard_capture_observations=sum(c["standard_capture"] for c in new_routes.values()),
        new_robot_assisted_capture_observations=sum(c["robot_assisted_capture"] for c in new_routes.values()),
        total_unique_commands=len(rows),
        total_valid_observations=sum(len(row["lineage"]) for row in rows),
        label_conflicts=sum(row["label_conflict"] for row in rows),
        label_policy="Standard captures are positive; robot-assisted captures are standard misses. Any standard label conflict is negative with all evidence retained.",
        deduplication="Exact bucket cell, five shot values, board heading and twist, and both six-joint commands.",
    )
    return rows, provenance
