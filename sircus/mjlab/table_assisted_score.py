"""Opt-in v3 task score: permit table contact while retaining v2 containment.

The calibrated whole-ball entry, continuous retention, physics guards, and
first terminal are the unchanged v2 kernel. Only the route contact exclusion
and its reward gate change. Historical v2 code and observations stay intact.
"""
from __future__ import annotations

import math
import numpy as np

from . import calibrated_score as capture_v2

VERSION = "sircus-table-assisted-native-capture-v3"
ROUTE_VERSION = "sircus-table-assisted-board-route-v3"
REWARD_VERSION = "sircus-table-assisted-shot-reward-v3"
CONTAINMENT_VERSION = capture_v2.VERSION
FORCE_NAMES = capture_v2.FORCE_NAMES
FORCE_THRESHOLD_N = capture_v2.FORCE_THRESHOLD_N
FORBIDDEN_CONTACT_NAMES = ("floor", "robot", "unattributed")
validate_calibration = capture_v2.validate_calibration


def reward(row):
    """Keep v2 weights; only floor/robot/unattributed block capture bonuses.

Rejected setups retain -1 without requiring a fabricated native score. A valid
row must explicitly carry v3 versions, so old reward rows cannot silently opt in.
The 0.1 bonus is for one BOARD contact episode, not one total physical bounce.
"""
    if row.get("invalid") or row.get("invalid_physics"):
        return -1.0
    if (row.get("scoring_version"), row.get("route_version"), row.get("reward_version")) != (VERSION, ROUTE_VERSION, REWARD_VERSION):
        raise ValueError("Explicit v3 score/route/reward identity required")
    shaped = .15 * float(row["board_bounce"]) + .5 * math.exp(-float(row["closest_opening_distance_m"]) / .12)
    if row["forbidden_contact_steps"]:
        return float(shaped - .5)
    return float(shaped + .5 * min(float(row["max_retention_s"]), 1.)
                 + 3 * bool(row["route_success"]) + .3 * (row["route_quality"] == 2)
                 + .1 * bool(row["single_board_bounce_success"]))


def score_trace(scene, bucket_xy_m, normal, calibration, arrays):
    """Re-score native evidence without rerunning physics or altering old labels.

All contact counts use force intervals [0, physics_steps), exactly as v2.
Table contact anywhere in that prefix is allowed; no new contact-order rule is
introduced. `table_assisted_capture` reports table-contact presence, not proof
of a rebound velocity reversal or causal assistance. Rim quality is separate.
"""
    score = capture_v2.score_trace(scene, bucket_xy_m, normal, calibration, arrays)
    force = np.asarray(arrays["forces_N"], dtype=float)[:score["physics_steps"]]
    forbidden = int(np.count_nonzero((force[:, 3:] > FORCE_THRESHOLD_N).any(axis=1)))
    table = int(np.count_nonzero(force[:, 2] > FORCE_THRESHOLD_N))
    route = bool(score["success"] and score["board_contact_episodes"] >= 1 and forbidden == 0)
    one_board = bool(route and score["board_contact_episodes"] == 1)
    score.update(
        complete_success=route, route_success=route,
        route_quality=score["quality_score"] if route else 0,
        single_board_bounce_success=one_board, single_bounce_success=one_board,
        forbidden_contact_steps=forbidden, table_contact_steps=table,
        direct_capture=bool(route and table == 0), table_assisted_capture=bool(route and table > 0),
        route_class="table_assisted" if route and table else "direct" if route else "not_captured",
        scoring_version=VERSION, reward_version=REWARD_VERSION, route_version=ROUTE_VERSION,
        containment_version=CONTAINMENT_VERSION,
        forbidden_contact_names=list(FORBIDDEN_CONTACT_NAMES),
        route_scope="Measured board rebound, whole-ball mouth entry and one continuous second retained; table and bucket contacts permitted; floor/robot/unattributed contacts excluded through the unchanged scoring terminal",
        table_contact_scope="Any table force interval >0.01N before the unchanged terminal, including before the board; presence is not proof of a table rebound or causal assistance",
        single_bounce_scope="single_bounce_success is a compatibility alias of single_board_bounce_success: exactly one board-contact episode in an accepted route, not one total physical bounce",
    )
    score["reward"] = reward(score)
    return score
