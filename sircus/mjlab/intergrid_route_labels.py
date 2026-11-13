"""Verify direct and table rebound labels without changing native success scoring."""
from __future__ import annotations

from collections import defaultdict
import hashlib
import json
import math

import numpy as np

from .benchmark_verify import contained, sha
from .calibrated_score import (
    CONTACT_FREE_S, FORCE_NAMES, FORCE_THRESHOLD_N, REBOUND_NORMAL_SPEED_M_S, TIME_EPS_S,
)

LABEL_VERSION = "sircus-pre-entry-table-rebound-v1"
TABLE_SPEED_M_S = 0.05
ARRAYS = ("record_ids", "time_s", "forces_N", "velocity_m_s")


def label_capture(row, arrays, world):
    """Recognize force episodes that reverse vertical velocity before whole-ball entry."""
    if row["route_success"] is not True or row["scalar_score_audit_passed"] is not True:
        raise ValueError("Route labels require a verified standard capture")
    if arrays["record_ids"][world] != row["id"]:
        raise ValueError("Native trace column has a different shot identity")
    terminal = row["physics_steps"]
    time = np.asarray(arrays["time_s"][:terminal + 1], float)
    force = arrays["forces_N"][:terminal, world]
    velocity = arrays["velocity_m_s"][:terminal + 1, world]
    if (len(time) != terminal + 1 or terminal <= 0
            or force.shape != (terminal, len(FORCE_NAMES)) or velocity.shape != (terminal + 1, 3)
            or not np.isfinite(force).all() or (force < 0).any() or not np.isfinite(velocity).all()
            or not np.isfinite(time).all() or (np.diff(time) <= 0).any()):
        raise ValueError("Invalid native route evidence")
    board, bucket, table, floor, robot, other = (force > FORCE_THRESHOLD_N).T
    normal = np.asarray(row["board_normal"], float)
    if normal.shape != (3,) or not math.isclose(float(np.linalg.norm(normal)), 1.0, abs_tol=1e-6):
        raise ValueError("Expected a unit board normal")
    last = np.maximum.accumulate(np.where(board, time[1:], -np.inf))
    rebound = np.flatnonzero((~board) & (time[1:] - last <= CONTACT_FREE_S + TIME_EPS_S)
                             & (velocity[1:].astype(float) @ normal > REBOUND_NORMAL_SPEED_M_S))
    if not len(rebound) or (floor | robot | other).any():
        raise ValueError("Native evidence contradicts standard capture")
    if not table.any():
        if row["route_class"] != "direct":
            raise ValueError("Native evidence contradicts direct route")
        return "direct", 0
    if row["route_class"] != "table_assisted":
        raise ValueError("Native evidence contradicts table contact")
    start, entry = time[rebound[0] + 1], row["whole_ball_entry_time_s"]
    edges = np.flatnonzero(np.diff(np.r_[False, table, False]))
    rebounds = sum(start <= time[s] < time[e] <= entry
                   and not (board[s:e] | bucket[s:e]).any()
                   and velocity[s, 2] < -TABLE_SPEED_M_S and velocity[e, 2] > TABLE_SPEED_M_S
                   for s, e in zip(edges[::2], edges[1::2], strict=True))
    return ("table" if rebounds else None), int(rebounds)


def trace_labels(bindings):
    """Check each referenced archive once; bindings are (path, hash, column, native row)."""
    packets = defaultdict(list)
    for index, (path, trace_sha, column, row) in enumerate(bindings):
        packets[path].append((index, trace_sha, column, row))
    results = {}
    for path, members in sorted(packets.items()):
        if {member[1] for member in members} != {sha(path)}:
            raise ValueError(f"Native trace bytes changed: {path.name}")
        with np.load(path, allow_pickle=False) as archive:
            arrays = {name: archive[name] for name in ARRAYS}
        for index, _, column, row in members:
            results[index] = label_capture(row, arrays, column)
    return [results[index] for index in range(len(bindings))]


def annotate_routes(rows, directory):
    captures = [row for row in rows if row["route_success"] and not row["invalid_physics"]]
    labels = trace_labels([(contained(directory, row["native_cache_file"]), row["native_cache_sha256"],
                            row["native_cache_row"], row) for row in captures])
    for row in rows:
        row.update(shot_route=None, table_rebounds_before_entry=None)
    for row, (route, rebounds) in zip(captures, labels, strict=True):
        row.update(shot_route=route, table_rebounds_before_entry=rebounds)
    return rows


def historical_labels(rows, history_root):
    """Recover labels omitted from the compact dataset, binding back to native evidence."""
    studies = defaultdict(list)
    for row in rows:
        if row["route_success"] and not row["label_conflict"]:
            studies[row["source_study"]].append(row)
    labels = {}
    for name, sources in sorted(studies.items()):
        directory = contained(history_root, name)
        wanted = {row["source_record_id"]: row for row in sources}
        if len(wanted) != len(sources):
            raise ValueError("Historical source IDs are not unique")
        found = set()
        table_sources = []
        with (directory / "rows.jsonl").open("rb") as stream:
            for raw in stream:
                native = json.loads(raw)
                key = native["id"]
                source = wanted.get(key)
                if source is None:
                    continue
                if (key in found or hashlib.sha256(raw.rstrip(b"\r\n")).hexdigest() != source["source_row_sha256"]
                        or not native["route_success"] or native["scalar_score_audit_passed"] is not True
                        or native["native_cache_row"] != source["source_trace_column"]
                        or native["native_cache_sha256"] != source["source_trace_sha256"]):
                    raise ValueError(f"Historical route evidence is inconsistent: {name}/{key}")
                found.add(key)
                if native["route_class"] == "direct":
                    labels[source["id"]] = ("direct", 0)
                elif native["route_class"] == "table_assisted":
                    table_sources.append((source, native))
                else:
                    raise ValueError(f"Unknown historical standard route: {native['route_class']}")
        if found != wanted.keys():
            raise ValueError(f"Historical ledger is missing required training shots: {name}")
        labels.update((source["id"], result) for (source, _), result in zip(
            table_sources, trace_labels([
                (contained(directory, source["source_trace"]), source["source_trace_sha256"],
                 source["source_trace_column"], native) for source, native in table_sources
            ]), strict=True))
    return [dict(row, shot_route=labels.get(row["id"], (None, None))[0],
                 table_rebounds_before_entry=labels.get(row["id"], (None, None))[1]) for row in rows]
