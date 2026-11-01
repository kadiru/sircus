"""Small spatial development grid, frozen search and independent repeats; no RL."""

import hashlib
import itertools
import json
import math

import numpy as np
from board_route import CONTACT_FREE_S, FORCE_THRESHOLD_N, VERSION
from catalogue_contract import candidate, specification

CENTER = [0.67, -0.10]
XS = [0.47, 0.67, 0.87]
YS = [-0.40, -0.55, -0.70]
SEED = 20261103


def cells():
    return [
        {
            "id": f"x{x:.2f}-y{y:.2f}",
            "bucket_xy_m": [x, y],
            "distance_m": math.dist(CENTER, [x, y]),
            "heading_deg": math.degrees(math.atan2(x - CENTER[0], CENTER[1] - y)),
        }
        for y in YS
        for x in XS
    ]


def candidates():
    triples = [
        (0.35, 0.2, 24),
        (0.4, 0.13, 21),
        (0.45, 0.15, 19.5),
        (0.55, 0.08, 13.5),
        (0.45, 0.13, 13.5),
        (0.5, 0.08, 18),
    ]
    triples += list(
        itertools.product([0.35, 0.45, 0.55], [0.08, 0.16, 0.24], [12, 18, 24, 30])
    )
    return [candidate(*t, prefix="grid") for t in dict.fromkeys(triples)]


def scene(c, cell):
    return dict(
        cell,
        id=cell["id"] + ":" + c["id"],
        board_center_m=[*CENTER, c["height_m"]],
        difficulty="grid_development",
    )


def select(rows, count=2):
    ordered = sorted(
        [r for r in rows if r.get("route_success") and not r.get("invalid")],
        key=lambda r: (
            -int(r.get("single_bounce_success", False)),
            -r["route_quality"],
            -r["reward"],
            r["id"],
        ),
    )
    return [
        {k: r[k] for k in ("id", "height_m", "gap_m", "tilt_deg")}
        for r in ordered[:count]
    ]


def refine(rows):
    valid = sorted(
        [r for r in rows if not r.get("invalid")],
        key=lambda r: (
            -int(r.get("route_success", False)),
            -r.get("route_quality", 0),
            bool(r.get("external_contact_steps")),
            -r["reward"],
            r["id"],
        ),
    )[:2]
    seen = {(r["height_m"], r["gap_m"], r["tilt_deg"]) for r in rows}
    new = []
    for r in valid:
        for dg, dt in itertools.product([-0.01, 0.01], [-1.5, 1.5]):
            triple = (r["height_m"], round(r["gap_m"] + dg, 6), r["tilt_deg"] + dt)
            if triple not in seen:
                seen.add(triple)
                new.append(candidate(*triple, prefix="grid-local"))
    return new


def perturb(c, ci, candidate_index, i, final=False):
    seed = SEED + (200000 if final else 100000) + ci * 1000 + candidate_index * 100 + i
    rng = np.random.default_rng(seed)
    r = dict(c)
    for k, spread in [("height_m", 0.002), ("gap_m", 0.002), ("tilt_deg", 0.5)]:
        r[k] = float(c[k] + rng.uniform(-spread, spread))
    r["seed"] = seed
    return r


def summary(rows):
    return {
        "trials": len(rows),
        "historical_captures": sum(r["success"] for r in rows),
        "route_captures": sum(r.get("route_success", False) for r in rows),
        "single_bounce_captures": sum(
            bool(r.get("single_bounce_success")) for r in rows
        ),
        "clean": sum(r.get("route_quality") == 2 for r in rows),
        "assisted": sum(r.get("route_quality") == 1 for r in rows),
        "external_contact": sum(bool(r.get("external_contact_steps")) for r in rows),
        "invalid": sum(bool(r.get("invalid")) for r in rows),
    }


def choose(screened):
    if not screened:
        return None
    best = min(
        screened,
        key=lambda s: (
            -s["summary"]["route_captures"],
            -s["summary"].get("single_bounce_captures", 0),
            -s["summary"]["clean"],
            s["candidate"]["id"],
        ),
    )
    return best["candidate"] if best["summary"]["route_captures"] else None


def contract(cfg, smoke=False):
    for cell in cells():
        for xy, center, size in zip(
            cell["bucket_xy_m"], cfg["table"]["center_m"], cfg["table"]["size_m"]
        ):
            assert (
                abs(xy - center) + cfg["bucket"]["outer_top_diameter_m"] / 2 <= size / 2
            )
    spec = {
        "version": "sircus-bucket-grid-v1",
        "parent_quality": specification(cfg),
        "cells": cells(),
        "candidates": candidates(),
        "board_xy_m": CENTER,
        "screening_per_candidate": 10,
        "fresh_trials_per_cell": 30,
        "seed": SEED,
        "route_version": VERSION,
        "supports": "table/floor are stationary kinematic rigid bodies for supported GPU contact filtering; geometry unchanged; rechecked controls",
        "contact_threshold_N": FORCE_THRESHOLD_N,
        "board_contact_episode_gap_s": CONTACT_FREE_S,
        "smoke": smoke,
        "selection": "up to two board-only captures by single-bounce/tier/reward; 10 screening perturbations each; freeze best by captures/single-bounce/clean/id; 30 fresh seeds, no retuning",
        "perturbations": {"board_height_m": 0.002, "gap_m": 0.002, "tilt_deg": 0.5},
        "scope": "development search, not RL or held-out generalization; fixed board XY, bucket-facing heading; endpoint initialization; no robot path validation",
        "route_definition": "historical board-bounce/whole-ball/1s capture plus one or more board contact episodes (single-bounce captures counted separately) and no detected table/floor/robot/other contact during flight; bucket rim allowed",
        "sensor_limits": "240Hz normal contact force threshold; aggregate simultaneous forces may cancel; subthreshold contacts not certified absent",
    }
    spec["fingerprint"] = hashlib.sha256(
        json.dumps(spec, sort_keys=True).encode()
    ).hexdigest()
    return spec
