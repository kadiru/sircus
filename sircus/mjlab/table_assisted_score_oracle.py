"""Independent opt-in v3 oracle built on the unchanged independent v2 kernel.

This module never imports the v3 production scorer. Containment and terminal
come from calibrated_score_oracle, while route counts and reward arithmetic
are reconstructed here separately. No rollout or material recalibration occurs.
"""
from __future__ import annotations

import math
import numpy as np

from . import calibrated_score_oracle as old_oracle

VERSION = "sircus-table-assisted-native-capture-v3"
ROUTE_VERSION = "sircus-table-assisted-board-route-v3"
REWARD_VERSION = "sircus-table-assisted-shot-reward-v3"
CONTAINMENT_VERSION = "sircus-calibrated-native-capture-v2"
FORCE_NAMES = ("board", "bucket", "table", "floor", "robot", "unattributed")
FORCE_THRESHOLD_N = .01
FORBIDDEN_CONTACT_NAMES = ("floor", "robot", "unattributed")
validate_calibration = old_oracle.validate_calibration


def reward(row):
    if row.get("invalid") or row.get("invalid_physics"):
        return -1.0
    if row.get("scoring_version") != VERSION or row.get("route_version") != ROUTE_VERSION or row.get("reward_version") != REWARD_VERSION:
        raise ValueError("Explicit v3 score/route/reward identity required")
    total = math.exp(-float(row["closest_opening_distance_m"]) / .12) / 2
    total += .15 if row["board_bounce"] else 0.
    if int(row["forbidden_contact_steps"]) > 0:
        return float(total - .5)
    total += min(float(row["max_retention_s"]), 1.) / 2
    if row["route_success"]:
        total += 3.
    if row["route_quality"] == 2:
        total += .3
    if row["single_board_bounce_success"]:
        total += .1
    return float(total)


def score_trace(scene, bucket_xy_m, normal, calibration, arrays):
    result = old_oracle.score_trace(scene, bucket_xy_m, normal, calibration, arrays)
    prefix = np.asarray(arrays["forces_N"], dtype=float)[:int(result["physics_steps"])]
    blocked = (prefix[:, 3] > .01) | (prefix[:, 4] > .01) | (prefix[:, 5] > .01)
    forbidden = int(len(np.flatnonzero(blocked)))
    table = int((prefix[:, 2] > .01).sum())
    accepted = bool(result["success"] and result["board_contact_episodes"] > 0 and not forbidden)
    one_board = bool(accepted and result["board_contact_episodes"] == 1)
    category = "not_captured"
    if accepted:
        category = "table_assisted" if table else "direct"
    result.update(
        route_success=accepted, complete_success=accepted,
        route_quality=result["quality_score"] if accepted else 0,
        single_board_bounce_success=one_board, single_bounce_success=one_board,
        forbidden_contact_steps=forbidden, table_contact_steps=table,
        direct_capture=bool(accepted and not table), table_assisted_capture=bool(accepted and table),
        route_class=category,
        scoring_version=VERSION, reward_version=REWARD_VERSION, route_version=ROUTE_VERSION,
        containment_version=CONTAINMENT_VERSION,
        forbidden_contact_names=list(FORBIDDEN_CONTACT_NAMES),
        route_scope="Measured board rebound, whole-ball mouth entry and one continuous second retained; table and bucket contacts permitted; floor/robot/unattributed contacts excluded through the unchanged scoring terminal",
        table_contact_scope="Any table force interval >0.01N before the unchanged terminal, including before the board; presence is not proof of a table rebound or causal assistance",
        single_bounce_scope="single_bounce_success is a compatibility alias of single_board_bounce_success: exactly one board-contact episode in an accepted route, not one total physical bounce",
    )
    result["reward"] = reward(result)
    return result
