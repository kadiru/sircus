"""Read robot-contact routes from a verified v3 native shot and its dense trace."""
from __future__ import annotations

import math

import numpy as np

from . import table_assisted_score as v3
from .calibrated_score import (
    CONTACT_FREE_S,
    FORCE_NAMES,
    FORCE_THRESHOLD_N,
    REBOUND_NORMAL_SPEED_M_S,
    TIME_EPS_S,
)

VERSION = "cell1050-robot-rebound-v1"
RETENTION_S = 1.0


def score_recorded_outcome(row, arrays, *, world=None):
    """Return an additive experimental outcome for one independently scored shot.

    Pass a single-world trace, or a packet with its native world index. Contact
    times are interval boundaries. A robot-assisted capture requires the first
    robot interval to follow the first board rebound, the last robot interval
    to end before or at the retained run, and calibrated whole-ball retention.
    The force channel establishes the order of contacts. Velocity differences
    across the first robot episode report the observed change in motion.
    """
    if row.get("native_release") is not True or row.get("scalar_score_audit_passed") is not True:
        raise ValueError("Experimental scoring requires a verified native release")
    if (row.get("scoring_version"), row.get("route_version"), row.get("containment_version")) != (
        v3.VERSION, v3.ROUTE_VERSION, v3.CONTAINMENT_VERSION,
    ):
        raise ValueError("Experimental scoring requires the current v3 containment")

    force = np.asarray(arrays["forces_N"])
    velocity = np.asarray(arrays["velocity_m_s"])
    position = np.asarray(arrays["position_m"])
    time = np.asarray(arrays["time_s"], dtype=float)
    if world is not None:
        if (isinstance(world, bool) or not isinstance(world, (int, np.integer))
                or force.ndim != 3 or velocity.ndim != 3 or position.ndim != 3
                or not 0 <= world < min(force.shape[1], velocity.shape[1], position.shape[1])):
            raise ValueError("Expected a valid native packet world index")
        if "record_ids" in arrays and arrays["record_ids"][world] != row["id"]:
            raise ValueError("Native packet and shot identity disagree")
        force, velocity, position = force[:, world], velocity[:, world], position[:, world]
    if (time.ndim != 1 or len(time) < 2 or force.shape != (len(time) - 1, len(FORCE_NAMES))
            or position.shape != (len(time), 3) or velocity.shape != position.shape):
        raise ValueError("Expected aligned dense time, position, velocity and force arrays")
    if not np.array_equal(position[0], row["native_initial_ball_center_m"]):
        raise ValueError("Dense trace and recorded release center disagree")
    terminal = row["physics_steps"]
    if type(terminal) is not int or not 0 <= terminal < len(time):
        raise ValueError("Invalid native scoring terminal")
    time, force, velocity = time[:terminal + 1], force[:terminal], velocity[:terminal + 1]
    if (not np.isfinite(time).all() or abs(time[0]) > TIME_EPS_S or (np.diff(time) <= 0).any()
            or not np.isfinite(force).all() or (force < 0).any() or not np.isfinite(velocity).all()):
        raise ValueError("Invalid dense samples in the scoring interval")

    normal = np.asarray(row["board_normal"], dtype=float)
    if normal.shape != (3,) or not math.isclose(float(np.linalg.norm(normal)), 1.0, abs_tol=1e-6):
        raise ValueError("Expected a unit board normal")
    board, _, table, floor, robot, unattributed = (force > FORCE_THRESHOLD_N).T
    last_board_end = np.maximum.accumulate(np.where(board, time[1:], -np.inf))
    rebound = (~board
               & (time[1:] - last_board_end <= CONTACT_FREE_S + TIME_EPS_S)
               & ((np.asarray(velocity[1:], dtype=float) @ normal) > REBOUND_NORMAL_SPEED_M_S))
    rebound_ticks = np.flatnonzero(rebound) + 1
    if bool(len(rebound_ticks)) != row["board_bounce"]:
        raise ValueError("Trace and recorded board rebound disagree")
    board_rebound_s = float(time[rebound_ticks[0]]) if len(rebound_ticks) else None

    contact_ticks = np.flatnonzero(robot)
    first_robot_s = last_robot_end_s = first_robot_end_s = velocity_change = None
    episodes = 0
    if len(contact_ticks):
        splits = np.flatnonzero(np.diff(contact_ticks) > 1)
        first, last = contact_ticks[0], contact_ticks[splits[0]] if len(splits) else contact_ticks[-1]
        first_robot_s, first_robot_end_s = float(time[first]), float(time[last + 1])
        last_robot_end_s = float(time[contact_ticks[-1] + 1])
        velocity_change = (velocity[last + 1].astype(float) - velocity[first]).tolist()
        episodes = len(splits) + 1

    retained = bool(row["success"] and not row.get("invalid") and not row["invalid_physics"])
    retained_start_s = None
    if retained:
        capture_s = row["capture_time_s"]
        entry_s = row["whole_ball_entry_time_s"]
        if (not row["whole_ball_entry"] or board_rebound_s is None or capture_s is None or entry_s is None
                or terminal == 0 or not math.isclose(capture_s, float(time[-1]), abs_tol=TIME_EPS_S, rel_tol=0)
                or not math.isclose(row["max_retention_s"], RETENTION_S, abs_tol=1e-9, rel_tol=0)):
            raise ValueError("Capture lacks verified whole-ball retention")
        # v3 accepts the first sample at least one second after a run starts.
        # Use the earliest sample compatible with that terminal even when sample spacing varies.
        start_index = np.searchsorted(time, time[-2] - RETENTION_S + TIME_EPS_S, side="right")
        retained_start_s = float(time[start_index])
        if entry_s > capture_s - RETENTION_S + TIME_EPS_S or retained_start_s > capture_s - RETENTION_S + TIME_EPS_S:
            raise ValueError("Capture terminal contradicts the retained run")

    standard = bool(row["route_success"])
    if standard != (retained and not (robot | floor | unattributed).any()):
        raise ValueError("Trace and standard capture label disagree")
    assisted = bool(retained and len(contact_ticks) and not (floor | unattributed).any()
                    and first_robot_s >= board_rebound_s - TIME_EPS_S
                    and last_robot_end_s <= retained_start_s + TIME_EPS_S)
    if standard:
        category = "standard_capture"
    elif assisted:
        category = "robot_assisted_capture"
    else:
        category = "uncategorized_capture" if retained else "not_captured"
    return dict(
        version=VERSION, route_class=category, experimental_route_success=standard or assisted,
        standard_capture=standard, robot_assisted_capture=assisted, retained_capture=retained,
        whole_ball_entry=bool(row["whole_ball_entry"]), whole_ball_entry_s=row["whole_ball_entry_time_s"],
        retention_s=float(row["max_retention_s"]), capture_s=row["capture_time_s"],
        retained_run_earliest_start_s=retained_start_s, first_board_rebound_s=board_rebound_s,
        first_robot_contact_s=first_robot_s, first_robot_contact_end_s=first_robot_end_s,
        last_robot_contact_end_s=last_robot_end_s, robot_contact_steps=int(len(contact_ticks)),
        robot_contact_episodes=int(episodes), first_robot_episode_velocity_change_m_s=velocity_change,
        peak_robot_force_N=float(force[:, FORCE_NAMES.index("robot")].max()) if terminal else 0.0,
        table_contact_steps=int(np.count_nonzero(table)), floor_contact_steps=int(np.count_nonzero(floor)),
        unattributed_contact_steps=int(np.count_nonzero(unattributed)),
    )
