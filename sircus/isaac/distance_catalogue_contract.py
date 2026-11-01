"""Frozen distance-search protocol; pure functions, no simulator dependencies."""

import hashlib
import itertools
import json

from catalogue_contract import candidate, finalists, perturb, specification
from plane_contract import CENTER_XY

DISTANCES = [0.4, 0.3, 0.5, 0.55, 0.6, 0.65]
HEIGHTS = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55]
GAPS = [0.08, 0.12, 0.16, 0.20, 0.24]
TILTS = [12, 18, 24, 30, 36, 42, 48]
ANCHORS = [
    (0.4, 0.13, 21),
    (0.45, 0.12, 21),
    (0.375, 0.13, 24),
    (0.475, 0.12, 27),
    (0.35, 0.19, 15),
    (0.325, 0.23, 12),
]


def grid(distance):
    if distance == 0.4:
        return [candidate(*ANCHORS[0], prefix="control")]
    seen = set()
    result = []
    for triple in [*ANCHORS, *itertools.product(HEIGHTS, GAPS, TILTS)]:
        if triple not in seen:
            result.append(candidate(*triple, prefix="distance"))
            seen.add(triple)
    return result


def scene(c, distance):
    return {
        "id": c["id"],
        "board_center_m": [*CENTER_XY, c["height_m"]],
        "bucket_xy_m": [CENTER_XY[0], round(CENTER_XY[1] - distance, 6)],
        "distance_m": distance,
        "difficulty": "distance_catalogue",
    }


def refinement(rows, distance):
    if distance == 0.4:
        return []
    # Reward only breaks miss ties when there are insufficient captures.
    seeds = sorted(
        [r for r in rows if not r.get("invalid")],
        key=lambda r: (
            -r["quality_score"],
            -(
                r.get("minimum_mouth_clearance_mm")
                if r.get("minimum_mouth_clearance_mm") is not None and r["success"]
                else -999
            ),
            -r["reward"],
            r["id"],
        ),
    )[:6]
    seen = {(r["height_m"], r["gap_m"], r["tilt_deg"]) for r in rows}
    result = []
    offsets = [
        (0, dg, dt)
        for dg, dt in itertools.product([-0.01, 0, 0.01], [-1.5, 0, 1.5])
        if dg or dt
    ] + [(-0.0125, 0, 0), (0.0125, 0, 0)]
    for r in seeds:
        for dh, dg, dt in offsets:
            c = candidate(
                r["height_m"] + dh, r["gap_m"] + dg, r["tilt_deg"] + dt, "local"
            )
            key = (c["height_m"], c["gap_m"], c["tilt_deg"])
            if key not in seen:
                result.append(c)
                seen.add(key)
    return result


def selected(rows):
    return finalists(rows, 3)


def jitter(c, distance_index, candidate_index, trial, final=False):
    index = (
        (200000 if final else 100000)
        + distance_index * 1000
        + candidate_index * 100
        + trial
    )
    return perturb(c, index)


def choose(screened):
    if not screened:
        return None
    return max(
        screened,
        key=lambda e: (
            e["perturbed_summary"]["captures"],
            e["perturbed_summary"]["clean"],
            e["nominal_summary"]["captures"],
            e["candidate"]["id"],
        ),
    )["candidate"]


def table_margin(cfg, d):
    xy = scene(candidate(*ANCHORS[0]), d)["bucket_xy_m"]
    t = cfg["table"]
    radius = cfg["bucket"]["outer_top_diameter_m"] / 2
    return min(
        t["size_m"][i] / 2 - abs(xy[i] - t["center_m"][i]) - radius for i in range(2)
    )


def contract(cfg):
    assert all(table_margin(cfg, d) >= 0 for d in DISTANCES), (
        "Bucket footprint outside table"
    )
    d = {
        "version": "sircus-distance-catalogue-v1",
        "parent_quality": specification(cfg),
        "distances_m": DISTANCES,
        "heights_m": HEIGHTS,
        "gaps_m": GAPS,
        "tilts_deg": TILTS,
        "anchors": ANCHORS,
        "control": "400mm rechecks the prior strongest setting; other distances get broad grids",
        "refinement": "top6 valid search rows by tier/clearance/reward: gap +/-10mm x tilt +/-1.5deg, plus H +/-12.5mm; deduplicate; no capture required to refine near misses",
        "screening": "up to3 geometrically separated captures; freeze before5 nominal and20 perturbed trials each",
        "confirmation": "freeze screening winner by perturbed captures, clean captures, nominal captures, ID;50 disjoint perturbed trials without retuning",
        "perturbations": {
            "H_mm": 2,
            "gap_mm": 2,
            "tilt_deg": 0.5,
            "seed_base": 20261102,
            "screen_index_start": 100000,
            "final_index_start": 200000,
        },
        "table_margin_mm": {
            str(v): round(table_margin(cfg, v) * 1000, 3) for v in DISTANCES
        },
        "scope": "endpoint-initialized synthetic search; not RL, path validation or physical reliability; no feasible setting found is not proof of impossibility",
    }
    d["fingerprint"] = hashlib.sha256(
        json.dumps(d, sort_keys=True).encode()
    ).hexdigest()
    return d
