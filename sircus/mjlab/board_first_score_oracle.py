"""Independent board-first route/reward oracle over the v2 containment oracle.

No production board-first scorer import. Order is reconstructed by scanning
force intervals and velocity samples in time order, including terminal bounds.
"""
from __future__ import annotations

import math
import numpy as np

from . import calibrated_score_oracle as containment

VERSION = "sircus-board-first-native-capture-v4"
ROUTE_VERSION = "sircus-board-first-board-route-v4"
REWARD_VERSION = "sircus-board-first-shot-reward-v4"
CONTAINMENT_VERSION = "sircus-calibrated-native-capture-v2"
FORCE_NAMES = ("board", "bucket", "table", "floor", "robot", "unattributed")
FORCE_THRESHOLD_N = .01
FORBIDDEN_CONTACT_NAMES = ()
validate_calibration = containment.validate_calibration


def reward(row):
    if row.get("invalid") or row.get("invalid_physics"):
        return -1.
    if (row.get("scoring_version"), row.get("route_version"), row.get("reward_version")) != (VERSION, ROUTE_VERSION, REWARD_VERSION):
        raise ValueError("Explicit v4 score/route/reward identity required")
    total = math.exp(-float(row["closest_opening_distance_m"])/.12)/2
    total += .15 if row["board_bounce"] else 0
    if int(row["forbidden_contact_steps"]) > 0:
        return float(total-.5)
    total += min(float(row["max_retention_s"]),1)/2
    total += 3 if row["route_success"] else 0
    total += .3 if row["route_quality"] == 2 else 0
    total += .1 if row["single_board_bounce_success"] else 0
    return float(total)


def score_trace(scene, bucket_xy_m, normal, calibration, arrays):
    out = containment.score_trace(scene,bucket_xy_m,normal,calibration,arrays)
    times = (np.asarray(arrays["time_s"],dtype=float) if "time_s" in arrays
             else np.arange(len(arrays["position_m"])) * float(np.asarray(arrays["dt_s"]).item()))
    forces = np.asarray(arrays["forces_N"],dtype=float)[:out["physics_steps"]]
    velocity = np.asarray(arrays["velocity_m_s"],dtype=float)
    axis = np.asarray(normal,dtype=float)
    active = forces > .01
    any_contact = np.flatnonzero(np.sum(active,axis=1))
    first = int(any_contact[0]) if len(any_contact) else None
    names = [name for i,name in enumerate(FORCE_NAMES) if first is not None and active[first,i]]
    board_ticks = np.flatnonzero(active[:,0])
    first_board = int(board_ticks[0]) if len(board_ticks) else None
    rebound = None
    state_times = times[1:len(forces)+1]
    if len(board_ticks):
        contact_ends = times[board_ticks+1]
        nearest = np.searchsorted(contact_ends,state_times+1e-12,side="right")-1
        recent = np.zeros(len(forces),dtype=bool)
        matched = nearest >= 0
        recent[matched] = state_times[matched]-contact_ends[nearest[matched]] <= .05+1e-12
        positive = np.sum(velocity[1:len(forces)+1]*axis,axis=1) > .05
        candidates = np.flatnonzero(recent & ~active[:,0] & positive)
        if len(candidates): rebound = int(candidates[0])+1
    counts = {name:int(np.count_nonzero(active[:,i])) for i,name in enumerate(FORCE_NAMES)}
    nonboard_intervals = np.flatnonzero(np.sum(active[:,1:],axis=1))
    if (rebound is not None) != out["board_bounce"]:
        raise ValueError("Independent board rebound differs from containment oracle")
    stop = out["physics_steps"] if rebound is None else rebound
    forbidden = int(np.count_nonzero(nonboard_intervals < stop))
    if first is None: status = "no_contact"
    elif "board" not in names: status = "non_board_first"
    elif len(names) != 1: status = "ambiguous_first_contact"
    elif forbidden: status = "intervening_contact_before_board_rebound"
    elif rebound is None: status = "board_contact_without_rebound"
    else: status = "board_first_confirmed"
    confirmed = status == "board_first_confirmed"
    hit = bool(out["success"] and confirmed)
    surfaces = [name for name in FORCE_NAMES[2:] if counts[name] > 0]
    category = "not_captured"
    if hit:
        if not surfaces: category = "direct"
        elif len(surfaces)>1: category = "mixed_assisted"
        elif surfaces[0]=="table": category = "table_assisted"
        elif surfaces[0]=="robot": category = "robot_assisted"
        else: category = "other_assisted"
    once = bool(hit and out["board_contact_episodes"]==1)
    failure = "invalid_physics" if out["invalid_physics"] else status if not confirmed else "not_retained" if not hit else None
    out.update(
        complete_success=hit,route_success=hit,route_quality=out["quality_score"] if hit else 0,
        single_board_bounce_success=once,single_bounce_success=once,
        forbidden_contact_steps=forbidden,forbidden_contact_names=[],
        table_contact_steps=counts["table"],robot_contact_steps=counts["robot"],
        floor_contact_steps=counts["floor"],unattributed_contact_steps=counts["unattributed"],
        direct_capture=bool(hit and not surfaces),table_assisted_capture=bool(hit and counts["table"]),
        robot_assisted_capture=bool(hit and counts["robot"]),
        other_assisted_capture=bool(hit and (counts["floor"] or counts["unattributed"])),
        assistance_surfaces=surfaces,route_class=category,
        board_first_confirmed=confirmed,board_first_status=status,
        first_contact_interval=first,first_contact_surfaces=names,
        first_contact_start_s=float(times[first]) if first is not None else None,
        first_contact_end_s=float(times[first+1]) if first is not None else None,
        first_board_contact_start_s=float(times[first_board]) if first_board is not None else None,
        first_board_rebound_s=float(times[rebound]) if rebound is not None else None,
        route_failure_reason=failure,scoring_version=VERSION,route_version=ROUTE_VERSION,
        reward_version=REWARD_VERSION,containment_version=CONTAINMENT_VERSION,
        route_scope="Measured board rebound before any other surface contact, whole-ball mouth entry and one continuous second retained; all subsequent surface contacts permitted through the unchanged scoring terminal",
        contact_order_scope="Force intervals >0.01N start at t[k]; first contact must be board-only and no non-board interval may start before the first observed board rebound timestamp. Same-interval or intervening contact order is not confirmed. Later contact presence is not proof of a causal rebound",
        forbidden_contact_scope="Non-board force intervals starting before the first measured board rebound; if no rebound is observed, all non-board intervals through the terminal. No surface is forbidden after a confirmed board-first rebound",
        assistance_scope="Contact presence before the unchanged terminal; table/robot/other flags may overlap and do not establish causal assistance",
        single_bounce_scope="single_bounce_success is a compatibility alias of single_board_bounce_success: exactly one board-contact episode in an accepted route, not one total physical bounce",
    )
    out["reward"] = reward(out)
    return out
