"""Validate completed aggregate evidence and export a compact, non-host catalogue."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

from catalogue_contract import finalists, grid, perturb, refinement
from settling_contract import accepted


def validate(d):
    if d["phase"] != "completed":
        raise ValueError("Incomplete catalogue")
    rows = d["rows"]
    expected_grid = grid()
    assert len(rows) >= len(expected_grid)
    for expected, actual in zip(expected_grid, rows):
        assert all(actual[k] == v for k, v in expected.items())
    expected_refine = refinement(rows[: len(expected_grid)])
    assert len(rows) == len(expected_grid) + len(expected_refine)
    for expected, actual in zip(expected_refine, rows[len(expected_grid) :]):
        assert all(actual[k] == v for k, v in expected.items())
    selected = finalists(rows)
    assert len(selected) == len(d["confirmed"]) > 0
    all_trials = list(rows)
    for idx, (candidate, entry) in enumerate(zip(selected, d["confirmed"])):
        assert entry["candidate"] == candidate
        for mode in ("nominal", "perturbed"):
            assert len(entry[mode]) == 20
            for i, row in enumerate(entry[mode]):
                expected = (
                    candidate if mode == "nominal" else perturb(candidate, idx * 20 + i)
                )
                for key in ("height_m", "gap_m", "tilt_deg"):
                    assert expected[key] == row[key]
            all_trials.extend(entry[mode])
    ids = [r["trial_id"] for r in all_trials]
    assert len(ids) == len(set(ids))
    measured = []
    for r in all_trials:
        assert r["quality_score"] in (0, 1, 2)
        assert bool(r["quality_score"]) == bool(r["success"])
        if r["success"]:
            assert (
                r["board_bounce"]
                and r["whole_ball_entry"]
                and r["retention_s"] >= 1 - 1e-9
            )
        if r["quality_score"] == 2:
            assert r["minimum_mouth_clearance_mm"] >= 1
            assert not any(
                r[k]
                for k in (
                    "bucket_contact_before_entry",
                    "rim_band_contact",
                    "exited_after_entry",
                )
            )
        if not r.get("invalid"):
            assert accepted(r["settled"])
            measured.append(r["settled"])
    return {
        "validated_trial_count": len(all_trials),
        "released_trials": len(measured),
        "max_settled_position_error_mm": max(r["position_error_mm"] for r in measured),
        "max_settled_normal_error_deg": max(r["normal_error_deg"] for r in measured),
        "max_settled_joint_speed_rad_s": max(r["joint_speed_rad_s"] for r in measured),
        "max_settled_position_drift_mm": max(r["position_drift_mm"] for r in measured),
        "max_settled_normal_drift_deg": max(r["normal_drift_deg"] for r in measured),
    }


def export(source, destination):
    d = json.loads(source.read_text())
    verified = validate(d)
    destination.mkdir(parents=True, exist_ok=True)
    d["verification"] = verified
    d["source_summary_sha256"] = hashlib.sha256(source.read_bytes()).hexdigest()
    (destination / "results.json").write_text(json.dumps(d, indent=2) + "\n")
    fields = [
        "trial_id",
        "board_height_mm",
        "drop_gap_mm",
        "ball_bottom_Z_mm",
        "ball_center_Z_mm",
        "dropper_flange_Z_mm",
        "tilt_deg",
        "distance_mm",
        "quality_score",
        "success",
        "invalid",
        "minimum_mouth_clearance_mm",
    ]
    with (destination / "searched-settings.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        radius_mm = d["contract"]["parent"]["scene"]["ball"]["radius_m"] * 1000
        for r in d["rows"]:
            bottom = (r["height_m"] + r["gap_m"]) * 1000
            writer.writerow(
                {
                    "trial_id": r["trial_id"],
                    "board_height_mm": r["height_m"] * 1000,
                    "drop_gap_mm": r["gap_m"] * 1000,
                    "ball_bottom_Z_mm": bottom,
                    "ball_center_Z_mm": bottom + radius_mm,
                    "dropper_flange_Z_mm": bottom + radius_mm + 95,
                    "tilt_deg": r["tilt_deg"],
                    "distance_mm": 400,
                    "quality_score": r["quality_score"],
                    "success": r["success"],
                    "invalid": r["invalid"],
                    "minimum_mouth_clearance_mm": r["minimum_mouth_clearance_mm"],
                }
            )
    print(json.dumps(verified, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("source", type=Path)
    p.add_argument("destination", type=Path)
    args = p.parse_args()
    export(args.source, args.destination)
