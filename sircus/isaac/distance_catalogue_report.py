"""Validate completed distance evidence and export selected settings without raw state."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

from distance_catalogue_contract import (
    DISTANCES,
    choose,
    grid,
    jitter,
    refinement,
    selected,
)
from settling_contract import accepted


def counts(rows):
    n = len(rows)
    captures = sum(r["success"] for r in rows)
    clean = sum(r["quality_score"] == 2 for r in rows)
    invalid = sum(bool(r.get("invalid")) for r in rows)
    return {
        "trials": n,
        "captures": captures,
        "clean": clean,
        "assisted": captures - clean,
        "misses": n - captures - invalid,
        "invalid": invalid,
    }


def check_summary(rows, reported):
    assert all(reported[k] == v for k, v in counts(rows).items()), (
        "Aggregate differs from trials"
    )


def params(expected, actual):
    for key in ("height_m", "gap_m", "tilt_deg"):
        assert actual[key] == expected[key], f"Parameter differs: {key}"


def validate(d):
    if d["phase"] != "completed":
        raise ValueError("Distance study is incomplete")
    assert [e["distance_m"] for e in d["distances"]] == DISTANCES
    all_rows = []
    for di, e in enumerate(d["distances"]):
        distance = e["distance_m"]
        rows = e["rows"]
        coarse = grid(distance)
        assert len(rows) >= len(coarse)
        for a, b in zip(coarse, rows):
            params(a, b)
        refined = refinement(rows[: len(coarse)], distance)
        assert len(rows) == len(coarse) + len(refined)
        for a, b in zip(refined, rows[len(coarse) :]):
            params(a, b)
        check_summary(rows, e["search_summary"])
        all_rows.extend(rows)
        selected_rows = selected(rows)
        assert [x["candidate"] for x in e["screened"]] == selected_rows
        for ci, s in enumerate(e["screened"]):
            for mode, n in [("nominal", 5), ("perturbed", 20)]:
                assert len(s[mode]) == n
                check_summary(s[mode], s[mode + "_summary"])
                for i, r in enumerate(s[mode]):
                    params(
                        s["candidate"]
                        if mode == "nominal"
                        else jitter(s["candidate"], di, ci, i),
                        r,
                    )
                all_rows.extend(s[mode])
        assert e["winner"] == choose(e["screened"])
        if e["winner"]:
            assert e["phase"] == "completed" and len(e["confirmation"]) == 50
            for i, r in enumerate(e["confirmation"]):
                params(jitter(e["winner"], di, 0, i, final=True), r)
            check_summary(e["confirmation"], e["confirmation_summary"])
            all_rows.extend(e["confirmation"])
        else:
            assert e["phase"] == "no_capture_found" and not e["confirmation"]
        assert len(e["probes"]) == 3
        assert (
            e["probes"][0]["peak_bucket_force_N"] > 0.01
            and e["probes"][1]["peak_bucket_force_N"] > 0.01
            and e["probes"][2]["peak_bucket_force_N"] < 0.01
        )
    assert len(all_rows) == d["attempts"]
    assert len({r["trial_id"] for r in all_rows}) == len(all_rows)
    released = []
    for r in all_rows:
        assert r["distance_m"] in DISTANCES and r["quality_score"] in (0, 1, 2)
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
            released.append(r)
    return {
        **counts(all_rows),
        "released": len(released),
        "max_settled_position_error_mm": max(
            r["settled"]["position_error_mm"] for r in released
        ),
        "max_settled_normal_error_deg": max(
            r["settled"]["normal_error_deg"] for r in released
        ),
        "max_settled_joint_speed_rad_s": max(
            r["settled"]["joint_speed_rad_s"] for r in released
        ),
    }


def export(source, destination):
    d = json.loads(source.read_text())
    d["verification"] = validate(d)
    d["source_summary_sha256"] = hashlib.sha256(source.read_bytes()).hexdigest()
    destination.mkdir(parents=True, exist_ok=True)
    # Compact aggregate trial parameters only; raw trajectories/joints remain outside Git.
    (destination / "results.json").write_text(
        json.dumps(d, separators=(",", ":")) + "\n"
    )
    fields = [
        "distance_mm",
        "status",
        "board_height_mm",
        "tilt_deg",
        "gap_mm",
        "ball_bottom_Z_mm",
        "ball_center_Z_mm",
        "flange_Z_mm",
        "fresh_trials",
        "captures",
        "clean",
        "assisted",
        "misses",
        "invalid",
    ]
    with (destination / "selected-settings.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for e in sorted(d["distances"], key=lambda e: e["distance_m"]):
            r = {"distance_mm": round(e["distance_m"] * 1000), "status": e["phase"]}
            c = e["winner"]
            if c:
                bottom = (c["height_m"] + c["gap_m"]) * 1000
                r.update(
                    board_height_mm=round(c["height_m"] * 1000, 3),
                    tilt_deg=c["tilt_deg"],
                    gap_mm=round(c["gap_m"] * 1000, 3),
                    ball_bottom_Z_mm=round(bottom, 3),
                    ball_center_Z_mm=round(bottom + 37.5, 3),
                    flange_Z_mm=round(bottom + 132.5, 3),
                    fresh_trials=len(e["confirmation"]),
                )
                r.update(
                    {
                        k: e["confirmation_summary"][k]
                        for k in ("captures", "clean", "assisted", "misses", "invalid")
                    }
                )
            writer.writerow(r)
    print(json.dumps(d["verification"], indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("source", type=Path)
    p.add_argument("destination", type=Path)
    a = p.parse_args()
    export(a.source, a.destination)
