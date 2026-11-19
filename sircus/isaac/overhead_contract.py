"""Bounded presentation replay: frozen settings, orthographic cameras, no training."""

import hashlib
import json

from parallel_contract import cells

SELECTED = (0, 2, 3, 6)
CAPTURE_STRIDE = 2  # 240 Hz physics -> 120 Hz recorded frames.


def selection(report, cfg):
    required = (
        "contract",
        "plans_and_seeds",
        "raw_rows",
        "executed_source_hashes",
        "distinct_fixed_targets",
        "contact_probes",
    )
    if report.get("phase") != "completed" or not report.get("validation_passed"):
        raise ValueError("Replay requires a completed, validated source study")
    if any(report.get("verification", {}).get(k) is not True for k in required):
        raise ValueError("Source evidence has not passed verification")
    if report["contract"]["scene"] != cfg or report["contract"]["grid_count"] != 32:
        raise ValueError("Scene changed; do not relabel historical settings")
    indices, actions = [], []
    for index in SELECTED:
        cell = next(c for c in report["cells"] if c["index"] == index)
        expected = cells(32)[index]
        if any(cell[k] != v for k, v in expected.items()):
            raise ValueError("Source cell geometry does not match")
        winner = cell.get("winner")
        if not winner or len(cell["confirmations"]) != 10:
            raise ValueError("Selected cell lacks its frozen confirmed setting")
        if not all(
            r["single_bounce_success"] and not r["invalid"]
            for r in cell["confirmations"]
        ):
            raise ValueError("Selected cell did not pass 10/10 reference repeats")
        indices.append(index)
        actions.append(dict(winner))
    return indices, actions


def cameras():
    """Four matched oblique views plus plan/side details of the first cell.

    Aperture is the actual orthographic width in metres (USD stores tenths).
    Fixed cameras only change presentation, never simulation or policy inputs.
    """
    result = [
        {
            "name": f"cell{i}",
            "cell": i,
            "width": 1280,
            "height": 800,
            "span_m": 2.1,
            "eye": [0.67, -2.3, 3.5],
            "look": [0.67, -0.20, 0.35],
        }
        for i in range(4)
    ]
    result += [
        {
            "name": "top",
            "cell": 0,
            "width": 1280,
            "height": 800,
            "span_m": 1.35,
            "eye": [0.67, -0.28, 4.0],
            "look": [0.67, -0.28, 0.0],
        },
        {
            "name": "side",
            "cell": 0,
            "width": 1280,
            "height": 800,
            "span_m": 2.10,
            "eye": [0.67, -3.0, 0.65],
            "look": [0.67, -0.25, 0.40],
        },
    ]
    return result


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()
