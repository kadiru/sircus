"""Opt-in board-first capture: later ball contacts may involve any surface.

Historical v2 containment/physics and v3 route definitions stay unchanged.
Contact channels establish an observed sequence, not the cause of a rebound.
A board-only first contact must be followed by a measured board rebound before
any other surface is contacted. Simultaneous/unresolved ordering is not proof.
"""
from __future__ import annotations

import math
import numpy as np

from . import calibrated_score as capture_v2

VERSION = "sircus-board-first-native-capture-v4"
ROUTE_VERSION = "sircus-board-first-board-route-v4"
REWARD_VERSION = "sircus-board-first-shot-reward-v4"
CONTAINMENT_VERSION = capture_v2.VERSION
FORCE_NAMES = capture_v2.FORCE_NAMES
FORCE_THRESHOLD_N = capture_v2.FORCE_THRESHOLD_N
FORBIDDEN_CONTACT_NAMES = ()
validate_calibration = capture_v2.validate_calibration

ROUTE_SCOPE = "Measured board rebound before any other surface contact, whole-ball mouth entry and one continuous second retained; all subsequent surface contacts permitted through the unchanged scoring terminal"
ORDER_SCOPE = "Force intervals >0.01N start at t[k]; first contact must be board-only and no non-board interval may start before the first observed board rebound timestamp. Same-interval or intervening contact order is not confirmed. Later contact presence is not proof of a causal rebound"
SINGLE_SCOPE = "single_bounce_success is a compatibility alias of single_board_bounce_success: exactly one board-contact episode in an accepted route, not one total physical bounce"


def reward(row):
    """Keep prior shaping weights, without penalizing permitted later contacts."""
    if row.get("invalid") or row.get("invalid_physics"):
        return -1.0
    if (row.get("scoring_version"), row.get("route_version"), row.get("reward_version")) != (VERSION, ROUTE_VERSION, REWARD_VERSION):
        raise ValueError("Explicit v4 score/route/reward identity required")
    shaped = .15 * float(row["board_bounce"]) + .5 * math.exp(-float(row["closest_opening_distance_m"]) / .12)
    if row["forbidden_contact_steps"]:
        return float(shaped - .5)
    return float(shaped + .5 * min(float(row["max_retention_s"]), 1.)
                 + 3 * bool(row["route_success"]) + .3 * (row["route_quality"] == 2)
                 + .1 * bool(row["single_board_bounce_success"]))


def score_trace(scene, bucket_xy_m, normal, calibration, arrays):
    """Score immutable post-release dense evidence; never rerun or alter physics."""
    score = capture_v2.score_trace(scene, bucket_xy_m, normal, calibration, arrays)
    _, velocity, force, times, _ = capture_v2._trace(arrays)
    stop = score["physics_steps"]
    force, times, velocity = force[:stop], times[:stop + 1], velocity[:stop + 1]
    active = force > FORCE_THRESHOLD_N
    ticks = np.flatnonzero(active.any(axis=1))
    first = int(ticks[0]) if len(ticks) else None
    first_names = [name for i, name in enumerate(FORCE_NAMES) if first is not None and active[first, i]]
    board_ticks = np.flatnonzero(active[:, 0])
    first_board = int(board_ticks[0]) if len(board_ticks) else None
    last_board_end = np.maximum.accumulate(np.where(active[:, 0], times[1:], -np.inf))
    rebound = (~active[:, 0]) & (times[1:] - last_board_end <= capture_v2.CONTACT_FREE_S + capture_v2.TIME_EPS_S) & ((velocity[1:] @ np.asarray(normal)) > capture_v2.REBOUND_NORMAL_SPEED_M_S)
    rebounds = np.flatnonzero(rebound) + 1
    rebound_index = int(rebounds[0]) if len(rebounds) else None
    if bool(len(rebounds)) != score["board_bounce"]:
        raise ValueError("Board rebound reconstruction disagrees with containment kernel")
    cutoff = rebound_index if rebound_index is not None else stop
    forbidden = int(np.count_nonzero(active[:cutoff, 1:].any(axis=1)))
    if first is None:
        status = "no_contact"
    elif "board" not in first_names:
        status = "non_board_first"
    elif len(first_names) > 1:
        status = "ambiguous_first_contact"
    elif forbidden:
        status = "intervening_contact_before_board_rebound"
    elif rebound_index is None:
        status = "board_contact_without_rebound"
    else:
        status = "board_first_confirmed"
    confirmed = status == "board_first_confirmed"
    accepted = bool(score["success"] and confirmed)
    counts = {name: int(active[:, i].sum()) for i, name in enumerate(FORCE_NAMES)}
    assistance = [name for name in FORCE_NAMES[2:] if counts[name]]
    category = "not_captured"
    if accepted:
        category = ("direct" if not assistance else "mixed_assisted" if len(assistance) > 1
                    else "table_assisted" if assistance == ["table"] else "robot_assisted" if assistance == ["robot"] else "other_assisted")
    one_board = bool(accepted and score["board_contact_episodes"] == 1)
    reason = ("invalid_physics" if score["invalid_physics"] else status if not confirmed
              else "not_retained" if not accepted else None)
    score.update(
        complete_success=accepted, route_success=accepted,
        route_quality=score["quality_score"] if accepted else 0,
        single_board_bounce_success=one_board, single_bounce_success=one_board,
        forbidden_contact_steps=forbidden, forbidden_contact_names=[],
        table_contact_steps=counts["table"], robot_contact_steps=counts["robot"],
        floor_contact_steps=counts["floor"], unattributed_contact_steps=counts["unattributed"],
        direct_capture=bool(accepted and not assistance),
        table_assisted_capture=bool(accepted and counts["table"]),
        robot_assisted_capture=bool(accepted and counts["robot"]),
        other_assisted_capture=bool(accepted and (counts["floor"] or counts["unattributed"])),
        assistance_surfaces=assistance, route_class=category,
        board_first_confirmed=confirmed, board_first_status=status,
        first_contact_interval=first, first_contact_surfaces=first_names,
        first_contact_start_s=float(times[first]) if first is not None else None,
        first_contact_end_s=float(times[first + 1]) if first is not None else None,
        first_board_contact_start_s=float(times[first_board]) if first_board is not None else None,
        first_board_rebound_s=float(times[rebound_index]) if rebound_index is not None else None,
        route_failure_reason=reason,
        scoring_version=VERSION, route_version=ROUTE_VERSION, reward_version=REWARD_VERSION,
        containment_version=CONTAINMENT_VERSION,
        route_scope=ROUTE_SCOPE, contact_order_scope=ORDER_SCOPE,
        forbidden_contact_scope="Non-board force intervals starting before the first measured board rebound; if no rebound is observed, all non-board intervals through the terminal. No surface is forbidden after a confirmed board-first rebound",
        assistance_scope="Contact presence before the unchanged terminal; table/robot/other flags may overlap and do not establish causal assistance",
        single_bounce_scope=SINGLE_SCOPE,
    )
    score["reward"] = reward(score)
    return score
