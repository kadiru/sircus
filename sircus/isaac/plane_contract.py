"""Fixed vertical release plane; bounded development search, not policy training."""

import hashlib
import itertools
import json

from pilot_contract import contract as settled_contract

HEIGHTS = [0.30, 0.35, 0.40, 0.45, 0.50]
GAPS = [0.08, 0.12, 0.16, 0.20, 0.24]
TILTS = [12.0, 17.0, 22.0, 27.0, 32.0, 37.0, 42.0]
DISTANCE = 0.40
CENTER_XY = [0.67, -0.10]


def candidates():
    return [
        {"id": f"plane-{i:03d}", "height_m": h, "gap_m": g, "tilt_deg": t}
        for i, (h, g, t) in enumerate(itertools.product(HEIGHTS, GAPS, TILTS))
    ]


def scene(candidate):
    return {
        "id": candidate["id"],
        "board_center_m": [*CENTER_XY, candidate["height_m"]],
        "bucket_xy_m": [CENTER_XY[0], CENTER_XY[1] - DISTANCE],
        "distance_m": DISTANCE,
        "difficulty": "fixed_distance",
    }


def action(candidate):
    return [candidate["tilt_deg"], 0.0, candidate["gap_m"]]


def contract(cfg):
    spec = {
        "version": "sircus-fixed-plane-grid-v1",
        "controller": settled_contract(cfg)["fingerprint"],
        "scene": cfg,
        "distance_m": DISTANCE,
        "board_xy_m": CENTER_XY,
        "heights_m": HEIGHTS,
        "gaps_m": GAPS,
        "tilts_deg": TILTS,
        "heading_offset_deg": 0,
        "release": "ball center Z = board center H + ball radius + gap; fixed XY",
        "plane": "Y=-0.10 m; normal toward bucket along -Y; board and release centers share X/Y",
        "selection": "top three captures by reward, deterministic ID tie break; five fresh repeats each",
        "scope": "development grid search, no RL training, no generalization test; endpoint initialization, not path validation",
    }
    spec["fingerprint"] = hashlib.sha256(
        json.dumps(spec, sort_keys=True).encode()
    ).hexdigest()
    return spec


def compact(candidate, result):
    record = dict(
        candidate,
        success=bool(result["success"]),
        reward=result["reward"],
        invalid=result.get("invalid"),
        invalid_component=result.get("invalid_component"),
        board_bounce=result.get("board_bounce", False),
        whole_ball_entry=result.get("whole_ball_entry", False),
        retention_s=result.get("max_retention_s", 0),
    )
    measured = result.get("settled_board", [])
    if measured:
        record["settled"] = {
            k: measured[-1][k]
            for k in (
                "position_error_mm",
                "normal_error_deg",
                "joint_speed_rad_s",
                "position_drift_mm",
                "normal_drift_deg",
            )
        }
    return record


def select(rows):
    return sorted(
        [r for r in rows if r["success"] and not r.get("invalid")],
        key=lambda r: (-r["reward"], r["id"]),
    )[:3]
