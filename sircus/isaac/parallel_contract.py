"""Frozen fixed-target work cells and high-level experiments; no simulator imports."""

import hashlib
import itertools
import json
import math

from board_route import VERSION as ROUTE_VERSION
from bucket_grid_contract import contract as grid_contract

VERSION = "sircus-parallel-workcells-v1"
BOARD_XY = [0.67, -0.10]
SPACING_M = 4.5


def cells(count=32):
    if count not in (4, 32, 64):
        raise ValueError("Grid must have 4, 32 or 64 distinct targets")
    xs = [0.47, 0.57, 0.67, 0.87]
    rows = count // len(xs)
    result = []
    for row in range(rows):
        y = -0.4 - (0.3 * row / (rows - 1) if rows > 1 else 0)
        for col, x in enumerate(xs):
            dx, dy = x - BOARD_XY[0], y - BOARD_XY[1]
            result.append(
                {
                    "id": f"cell-{len(result):02d}",
                    "index": len(result),
                    "grid_row": row,
                    "grid_column": col,
                    "bucket_xy_m": [x, y],
                    "board_xy_m": list(BOARD_XY),
                    "distance_m": math.hypot(dx, dy),
                    "heading_deg": math.degrees(math.atan2(dy, dx) + math.pi / 2),
                }
            )
    return result


def candidates():
    values = [(0.35, 0.24, 24.0), (0.35, 0.20, 24.0)]
    values += list(
        itertools.product([0.35, 0.40, 0.45], [0.12, 0.18, 0.24], [18.0, 24.0, 30.0])
    )
    return [
        {"id": f"height-{i:02d}", "height_m": h, "gap_m": g, "tilt_deg": t}
        for i, (h, g, t) in enumerate(dict.fromkeys(values))
    ]


def batch_actions(batch, active_cells):
    """Rotate the same finite search so simultaneous cells try different heights."""
    plan = candidates()
    if not 0 <= batch < len(plan):
        raise ValueError("Batch outside the frozen search")
    return [dict(plan[(batch + c["index"]) % len(plan)]) for c in active_cells]


def select(rows):
    valid = [r for r in rows if r.get("route_success") and not r.get("invalid")]
    return min(
        valid,
        key=lambda r: (
            -int(r["single_bounce_success"]),
            -r["route_quality"],
            -r["reward"],
            r["id"],
        ),
        default=None,
    )


def specification(cfg, count, active_indices, batches, confirmations):
    grid = cells(count)
    if len(set(active_indices)) != len(active_indices) or not active_indices:
        raise ValueError("Each work cell must have a distinct target")
    if any(i < 0 or i >= count for i in active_indices):
        raise ValueError("Unknown target index")
    if not 0 <= batches <= len(candidates()) or not 0 <= confirmations <= 30:
        raise ValueError("Unbounded experiment")
    spec = {
        "version": VERSION,
        "parent_grid": grid_contract(cfg)["fingerprint"],
        "route_version": ROUTE_VERSION,
        "scene": cfg,
        "cells": [grid[i] for i in active_indices],
        "grid_count": count,
        "num_envs": len(active_indices),
        "spacing_m": SPACING_M,
        "candidates": candidates(),
        "batches": batches,
        "confirmation_trials_per_selected_cell": confirmations,
        "perturbations": {"height_m": 0.002, "gap_m": 0.002, "tilt_deg": 0.5},
        "bucket_rule": "Distinct table-local XY; each stays fixed across all height/tilt trials. Distances differ between cells, never within one cell.",
        "physics": "One GPU PhysX scene, replicated articulations, isolated environment collision groups; same 240 Hz contact scoring.",
        "scope": "Parallel endpoint-initialized parameter experiments, not learned policy, grasp/transit control or physical validation.",
    }
    spec["fingerprint"] = hashlib.sha256(
        json.dumps(spec, sort_keys=True).encode()
    ).hexdigest()
    return spec
