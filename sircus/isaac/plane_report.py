"""Verify complete fixed-plane evidence and export aggregates without raw states."""

import collections
import hashlib
import json
from pathlib import Path

from plane_contract import candidates
from settling_contract import accepted


def verify(run):
    run = Path(run)
    d = json.loads((run / "summary.json").read_text())
    if d["phase"] != "completed" or d["completed"] != 175:
        raise ValueError("Incomplete fixed-plane sweep")
    expected = {c["id"]: c for c in candidates()}
    raw = [json.loads(s) for s in (run / "grid.jsonl").read_text().splitlines()]
    if len(raw) != 175 or {x["candidate"]["id"] for x in raw} != set(expected):
        raise ValueError("Incomplete or duplicated grid")
    for x in raw:
        if x["candidate"] != expected[x["candidate"]["id"]]:
            raise ValueError("Unexpected grid configuration")
    rows = {x["id"]: x for x in d["rows"]}
    if len(rows) != 175:
        raise ValueError("Missing compact results")
    for x in raw:
        r = x["result"]
        if not r.get("invalid") and (
            not r.get("settled_board") or not accepted(r["settled_board"][-1])
        ):
            raise ValueError("Released shot lacked accepted settling")
        if rows[x["candidate"]["id"]]["success"] != r["success"]:
            raise ValueError("Grid outcome mismatch")
        if r["success"] and (
            not r["board_bounce"]
            or not r["whole_ball_entry"]
            or r["max_retention_s"] < 1 - 1e-9
        ):
            raise ValueError("Capture contract mismatch")
    if d["captures"] != sum(x["result"]["success"] for x in raw):
        raise ValueError("Capture count mismatch")
    for entry in d["repeats"]:
        if len(entry["trials"]) != 5:
            raise ValueError("Incomplete selected repeats")
    if len(d["replays"]) != len(d["repeats"]):
        raise ValueError("Incomplete replay")
    for name, value in d["artifact_sha256"].items():
        if hashlib.sha256((run / name).read_bytes()).hexdigest() != value:
            raise ValueError("Artifact hash mismatch")
    measured = [
        x["result"]["settled_board"][-1]
        for x in raw
        if x["result"].get("settled_board")
    ]
    d["pose_summary"] = {
        "measured_grid_shots": len(measured),
        "max_position_error_mm": max(
            (x["position_error_mm"] for x in measured), default=None
        ),
        "max_normal_error_deg": max(
            (x["normal_error_deg"] for x in measured), default=None
        ),
        "max_joint_speed_rad_s": max(
            (x["joint_speed_rad_s"] for x in measured), default=None
        ),
        "max_position_drift_mm": max(
            (x["position_drift_mm"] for x in measured), default=None
        ),
        "max_normal_drift_deg": max(
            (x["normal_drift_deg"] for x in measured), default=None
        ),
    }
    d["rejection_reasons"] = dict(
        collections.Counter(
            str(r.get("invalid")) + ":" + str(r.get("invalid_component"))
            for r in d["rows"]
            if r.get("invalid")
        )
    )
    d["by_height"] = [
        {
            "height_m": h,
            "trials": sum(r["height_m"] == h for r in d["rows"]),
            "captures": sum(r["height_m"] == h and r["success"] for r in d["rows"]),
            "rejected": sum(
                r["height_m"] == h and bool(r.get("invalid")) for r in d["rows"]
            ),
        }
        for h in d["contract"]["heights_m"]
    ]
    return d


def aggregates(d):
    out = {k: v for k, v in d.items() if k not in ("rows", "replays", "repeats")}
    out["selected_combinations"] = [
        {
            "id": e["candidate"]["id"],
            "height_m": e["candidate"]["height_m"],
            "gap_m": e["candidate"]["gap_m"],
            "tilt_deg": e["candidate"]["tilt_deg"],
            "repeat_captures": sum(t["success"] for t in e["trials"]),
            "repeat_trials": len(e["trials"]),
            "replay_capture": d["replays"][i]["result"]["success"],
        }
        for i, e in enumerate(d["repeats"])
    ]
    return out
