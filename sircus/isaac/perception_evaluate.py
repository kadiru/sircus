"""Separate evaluator. Truth never goes back to the perception process."""

import argparse
import json
import time
from pathlib import Path

import numpy as np


def compare(estimate, truth):
    if estimate["frame_id"] != truth["frame_id"]:
        return None
    metrics = {}
    for name, actual in truth["objects"].items():
        seen = estimate["objects"][name]
        if seen["status"] != "estimated":
            metrics[name] = {"status": "missing"}
            continue
        error = np.array(seen["center_m"]) - actual["center_m"]
        # Table center is explicitly only the visible patch, so measure plane error.
        entry = {"status": "estimated"}
        if name == "table":
            entry["plane_distance_error_mm"] = float(
                abs(error @ np.array(actual["normal"])) * 1000
            )
        else:
            entry["center_error_mm"] = float(np.linalg.norm(error) * 1000)
        if "normal" in seen:
            dot = abs(np.array(seen["normal"]) @ np.array(actual["normal"]))
            entry["normal_error_deg"] = float(
                np.degrees(np.arccos(np.clip(dot, -1, 1)))
            )
        metrics[name] = entry
    return {
        "frame_id": estimate["frame_id"],
        "timestamp_utc": estimate["timestamp_utc"],
        "label": "Ground-truth evaluation only; not estimator inputs",
        "errors": metrics,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", required=True)
    a = p.parse_args()
    out = Path(a.output)
    # Latest files may advance independently. Only evaluate exact matching frame IDs.
    estimates = {}
    truths = {}
    processed = {}
    while True:
        try:
            est = json.loads((out / "perception.json").read_text())
            gt = json.loads((out / "evaluation_truth/latest.json").read_text())
            estimates[est["frame_id"]] = est
            truths[gt["frame_id"]] = gt
            for key in list(estimates):
                if key in truths and key not in processed:
                    report = compare(estimates[key], truths[key])
                    temp = out / "perception-evaluation.tmp"
                    temp.write_text(json.dumps(report, indent=2, allow_nan=False))
                    temp.replace(out / "perception-evaluation.json")
                    processed[key] = None
                    estimates.pop(key)
                    truths.pop(key)
            while len(processed) > 512:
                processed.pop(next(iter(processed)))
            while len(estimates) > 20:
                estimates.pop(next(iter(estimates)))
            while len(truths) > 20:
                truths.pop(next(iter(truths)))
        except (OSError, ValueError, KeyError):
            pass
        time.sleep(0.04)


if __name__ == "__main__":
    main()
