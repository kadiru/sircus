"""Prespecified dense search and nominal/perturbed confirmation protocol."""

import hashlib
import itertools
import json

import numpy as np
from plane_contract import contract as plane_contract

HEIGHTS = [0.30, 0.325, 0.35, 0.375, 0.40, 0.425, 0.45, 0.475, 0.50]
GAPS = [0.08, 0.12, 0.16, 0.20, 0.24]
TILTS = list(range(12, 43, 3))


def candidate(h, g, t, prefix="dense"):
    return {
        "id": f"{prefix}-h{round(h * 10000)}-g{round(g * 10000)}-t{round(t * 100)}",
        "height_m": round(h, 6),
        "gap_m": round(g, 6),
        "tilt_deg": round(t, 6),
    }


def grid():
    return [candidate(h, g, t) for h, g, t in itertools.product(HEIGHTS, GAPS, TILTS)]


def rank(rows, n):
    valid = [r for r in rows if r["success"] and not r.get("invalid")]
    return sorted(
        valid,
        key=lambda r: (
            -r["quality_score"],
            -(r.get("minimum_mouth_clearance_mm") or -999),
            r["id"],
        ),
    )[:n]


def refinement(rows):
    seen = {(r["height_m"], r["gap_m"], r["tilt_deg"]) for r in rows}
    result = []
    for r in rank(rows, 12):
        for dg, dt in itertools.product([-0.01, 0, 0.01], [-1.5, 0, 1.5]):
            c = candidate(r["height_m"], r["gap_m"] + dg, r["tilt_deg"] + dt, "refine")
            key = (c["height_m"], c["gap_m"], c["tilt_deg"])
            if key not in seen:
                result.append(c)
                seen.add(key)
    return result


def finalists(rows, n=6):
    # Avoid filling the catalogue with practically identical neighbors.
    picked = []
    for r in rank(rows, len(rows)):
        if all(
            abs(r["height_m"] - s["height_m"]) >= 0.024
            or abs(r["gap_m"] - s["gap_m"]) >= 0.019
            or abs(r["tilt_deg"] - s["tilt_deg"]) >= 2.9
            for s in picked
        ):
            picked.append(r)
        if len(picked) == n:
            break
    return picked


def perturb(c, index):
    rng = np.random.default_rng(20261102 + index)
    return candidate(
        c["height_m"] + rng.uniform(-0.002, 0.002),
        c["gap_m"] + rng.uniform(-0.002, 0.002),
        c["tilt_deg"] + rng.uniform(-0.5, 0.5),
        "perturbed",
    )


def specification(cfg):
    d = {
        "version": "sircus-shot-catalogue-v1",
        "parent": plane_contract(cfg),
        "grid": {"heights_m": HEIGHTS, "gaps_m": GAPS, "tilts_deg": TILTS},
        "refinement": "top12 captures by quality/margin: gap +/-10mm, tilt +/-1.5deg; deduplicate",
        "confirmation": "six geometrically separated finalists; each20 nominal +20 independently seeded bounded H/G/tilt perturbations; frozen selection before confirmation",
        "perturbations": {
            "height_mm": 2,
            "gap_mm": 2,
            "tilt_deg": 0.5,
            "seed": 20261102,
        },
        "quality": {
            "2": "capture with no observed bucket force >0.01N before first entry or near rim, no subsequent partial exit, and >=1mm mouth clearance",
            "1": "capture with bucket contact or marginal/unknown entry clearance",
            "0": "miss; invalid separately flagged",
        },
        "capture": "existing board bounce + whole ball entry + continuous1second retention",
        "scope": "fixed400mm distance; development search, not RL; ideal repeated simulation is not physical reliability",
    }
    d["fingerprint"] = hashlib.sha256(
        json.dumps(d, sort_keys=True).encode()
    ).hexdigest()
    return d
