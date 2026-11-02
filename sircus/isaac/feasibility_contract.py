"""Finite development search; analytic ranking is a proposal, never a capture label."""

import copy
import hashlib
import itertools
import json
import math
import random

from board_route import VERSION as ROUTE_VERSION
from parallel_contract import SPACING_M, candidates

VERSION = "sircus-workspace-feasibility-v1"
SEED = 20261105
BOARD_POINTS = [(0.67, -0.10)] + list(
    itertools.product([0.52, 0.62, 0.72, 0.82], [-0.30, -0.15, 0.0, 0.15])
)
ACTION_KEYS = ("board_x_m", "board_y_m", "height_m", "gap_m", "tilt_deg")
PERTURBATIONS = dict(zip(ACTION_KEYS, [0.003, 0.003, 0.002, 0.002, 0.5]))


def cells():
    return [
        {
            "id": f"target-{i:02d}",
            "index": i,
            "grid_row": i // 8,
            "grid_column": i % 8,
            "bucket_xy_m": [x, y],
            "board_xy_m": [0.67, -0.10],
            "distance_m": math.hypot(x - 0.67, y + 0.10),
            "heading_deg": math.degrees(math.atan2(y + 0.10, x - 0.67) + math.pi / 2),
        }
        for i, (y, x) in enumerate(
            itertools.product(
                [-0.72, -0.47, -0.22, 0.23],
                [-0.10, 0.10, 0.30, 0.50, 0.70, 0.90, 1.10, 1.40],
            )
        )
    ]


def dynamic_cell(cell, action):
    c = copy.deepcopy(cell)
    c["board_xy_m"] = [action["board_x_m"], action["board_y_m"]]
    dx, dy = [b - a for a, b in zip(c["board_xy_m"], c["bucket_xy_m"])]
    c["distance_m"] = math.hypot(dx, dy)
    c["heading_deg"] = math.degrees(math.atan2(dy, dx) + math.pi / 2)
    return c


def proposed_range(cfg, h, gap, tilt):
    """Frictionless single-plane estimate, used only to rank trial proposals."""
    theta = math.radians(tilt)
    r = cfg["ball"]["radius_m"]
    z = h + (r + cfg["board"]["size_m"][2] / 2) / math.cos(theta)
    fall = h + r + gap - z
    if fall <= 0:
        return 0.0
    v = math.sqrt(2 * 9.81 * fall)
    e = cfg["ball"]["restitution"]
    horizontal = (1 + e) * v * math.sin(theta) * math.cos(theta)
    vertical = v * ((1 + e) * math.cos(theta) ** 2 - 1)
    rim = (
        cfg["bucket"]["assumed_interior_depth_m"]
        + cfg["bucket"]["assumed_bottom_thickness_m"]
        + r
    )
    discriminant = vertical**2 + 2 * 9.81 * (z - rim)
    return (
        horizontal * (vertical + math.sqrt(discriminant)) / 9.81
        if discriminant >= 0
        else 0.0
    )


def baseline_actions():
    return [dict(a, board_x_m=0.67, board_y_m=-0.10) for a in candidates()]


def search_actions(cfg, cell):
    """Eight proposals per admissible board XY, balanced across height and gap.

    Alternate analytic proposals and seeded broad proposals. A poor analytic model
    therefore cannot exclude a region or establish that it is impossible.
    """
    rng = random.Random(SEED + cell["index"])
    portfolios = []
    for x, y in BOARD_POINTS:
        distance = math.hypot(cell["bucket_xy_m"][0] - x, cell["bucket_xy_m"][1] - y)
        if distance < 0.28:
            continue  # Conservative board/bucket footprint separation, not arm-path proof.
        proposals = []
        for h in [0.30, 0.35, 0.40, 0.45]:
            for gaps in [[0.08, 0.10, 0.12, 0.14], [0.16, 0.20, 0.24, 0.28]]:
                pool = list(itertools.product(gaps, range(10, 43, 2)))
                if len(proposals) % 2:
                    # Broad half of the portfolio includes the high-energy gap range.
                    g, t = rng.choice(pool)
                else:
                    g, t = min(
                        pool, key=lambda v: abs(proposed_range(cfg, h, *v) - distance)
                    )
                proposals.append(
                    {
                        "board_x_m": x,
                        "board_y_m": y,
                        "height_m": h,
                        "gap_m": g,
                        "tilt_deg": float(t),
                    }
                )
        portfolios.append(proposals)
    # Visit all XY choices before returning to their other height/gap strata.
    result = [p[k] for k in range(8) for p in portfolios]
    for i, a in enumerate(result):
        a["id"] = f"expanded-{i:03d}"
    return result


def rank(row):
    return (
        int(not row.get("invalid")),
        int(row.get("route_success", False)),
        int(row.get("single_bounce_success", False)),
        row.get("route_quality", 0),
        float(row.get("max_retention_s", 0)),
        float(row.get("reward", -1)),
    )


def select(rows):
    return max(
        (r for r in rows if not r.get("invalid") and r.get("route_success")),
        key=rank,
        default=None,
    )


def refinement_actions(row):
    if not row:
        return []
    result = []
    for dh, dg, dt in itertools.product(
        [-0.025, 0, 0.025], [-0.02, 0, 0.02], [-2, 0, 2]
    ):
        a = {k: row[k] for k in ACTION_KEYS}
        a.update(
            height_m=a["height_m"] + dh,
            gap_m=a["gap_m"] + dg,
            tilt_deg=a["tilt_deg"] + dt,
        )
        a["id"] = f"refined-{len(result):02d}"
        result.append(a)
    return result


def perturb(action, index, repeat, nominal=False):
    seed = SEED + 10000 + index * 100 + repeat
    rng = random.Random(seed)
    a = {k: action[k] for k in ACTION_KEYS}
    for key, delta in PERTURBATIONS.items():
        a[key] += 0 if nominal else rng.uniform(-delta, delta)
    return dict(a, id=action["id"], seed=seed)


def specification(cfg, indices, smoke=False):
    all_cells = cells()
    if (
        not indices
        or len(set(indices)) != len(indices)
        or any(i not in range(32) for i in indices)
    ):
        raise ValueError("Choose distinct target indices in 0..31")
    result = {
        "version": VERSION,
        "scene": copy.deepcopy(cfg),
        "seed": SEED,
        "num_envs": len(indices),
        "cells": [all_cells[i] for i in indices],
        "spacing_m": SPACING_M,
        "route_version": ROUTE_VERSION,
        "smoke": smoke,
        "board_points_m": BOARD_POINTS,
        "baseline_batches": len(baseline_actions()),
        "expanded_proposals_per_xy": 8,
        "refinement_limit_per_target": 27,
        "nominal_repeats": 3,
        "fixed_xy_ablation_repeats": 3,
        "direct_drop_audit_ball_heights_m": [0.30, 0.45, 0.60],
        "confirmation_repeats": 30,
        "perturbations": PERTURBATIONS,
        "selection": "Search+baseline+refinement, freeze by valid route/single-bounce/clean/retention/reward. No confirmation-driven re-selection.",
        "scope": "Development feasibility search in articulated Isaac Lab/PhysX, not RL or held-out generalization. Endpoints are initialized; no pickup, transit or arm-path collision certification.",
        "unknown_rule": "No success found means unresolved within this finite search, never impossible.",
        "geometry_status": "Rigid mount, contact properties and tool proxy provisional; 175 mm bucket measurement still reported height with unresolved interpretation.",
    }
    result["fingerprint"] = hashlib.sha256(
        json.dumps(result, sort_keys=True).encode()
    ).hexdigest()
    return result
