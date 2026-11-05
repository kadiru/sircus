"""First-flight diagnostics for translated 650 mm development shots.

These metrics rank unassisted flight observations; they do not alter capture,
whole-ball entry, clean-tier, retention or reward definitions. Force row k
belongs to [time[k], time[k+1]]. A cutoff at interval k includes position[k]
and excludes position[k+1], so a later table roll cannot improve the metric.
"""

from collections.abc import Mapping

import numpy as np

from .calibrated_score import (
    CONTACT_FREE_S, FORCE_NAMES, FORCE_THRESHOLD_N, REBOUND_NORMAL_SPEED_M_S,
    TIME_EPS_S, _geometry, _trace, validate_calibration,
)


VERSION = "sircus-xy650-first-flight-aperture-v1"
CROSSING_PLANE = "ball_center_at_rim"


def _unavailable(reason):
    return dict(version=VERSION, crossing_plane=CROSSING_PLANE, available=False,
                reason=reason, start_index=None, end_index=None,
                start_time_s=None, end_time_s=None, cutoff_reason=None,
                cutoff_contact_channels=[], first_crossing=None,
                mouth_clearance_m=None, closest_opening_distance_m=None,
                sample_count=0)


def first_flight(scene, bucket_xy, normal, calibration, arrays, *, valid=True):
    """Summarize only the first measured rebound's subsequent free flight.

    A descending *ball-center* rim-plane crossing supplies a signed necessary
    aperture clearance: minimum distance to the compiled rim polygon minus
    the full ball radius. This projected clearance is not a capture or a clean
    passage label. If no crossing occurs, sampled 3D distance to the opening
    center still ranks the observed first flight. Contact-distance arrays are
    validated when present; only measured forces >0.01 N cut this diagnostic.
    """
    if type(valid) is not bool:
        raise ValueError("Explicit valid-physics boolean required")
    if not valid:
        return _unavailable("invalid_physics")
    if not isinstance(arrays, Mapping) or not all(
        key in arrays for key in ("position_m", "velocity_m_s", "forces_N")
    ) or not ("time_s" in arrays or "dt_s" in arrays):
        return _unavailable("missing_trace")
    cfg = validate_calibration(calibration)
    p, v, force, times, _ = _trace(arrays)
    axis = np.asarray(normal, dtype=float)
    if axis.shape != (3,) or not np.isfinite(axis).all() or not np.isclose(
        np.linalg.norm(axis), 1., atol=1e-6, rtol=0
    ):
        raise ValueError("Expected a finite unit board normal")
    # The shared geometry validator needs one position to verify the complete
    # compiled facet geometry; no T-by-facet matrix is needed for this metric.
    _, _, _, radius, _, scene_rim, planes, horizontal = _geometry(
        scene, bucket_xy, cfg, p[:1]
    )
    rim = float(cfg.get("compiled_rim_m", scene_rim))
    if not np.isfinite(radius) or not np.isfinite(rim):
        raise ValueError("Finite sphere radius and compiled rim required")
    board = force[:, 0] > FORCE_THRESHOLD_N
    last_board_end = np.maximum.accumulate(np.where(board, times[1:], -np.inf))
    rebound = (~board) & (
        times[1:] - last_board_end <= CONTACT_FREE_S + TIME_EPS_S
    ) & ((v[1:] @ axis) > REBOUND_NORMAL_SPEED_M_S)
    indices = np.flatnonzero(rebound)
    if not len(indices):
        return _unavailable("no_board_rebound")
    start = int(indices[0]) + 1
    active = force > FORCE_THRESHOLD_N
    contacts = np.flatnonzero(active.any(axis=1) & (np.arange(len(force)) >= start))
    end = int(contacts[0]) if len(contacts) else len(p)-1
    channels = [FORCE_NAMES[i] for i in np.flatnonzero(active[end])] if len(contacts) else []
    xy = np.asarray(bucket_xy, dtype=float)
    segment = p[start:end+1]
    center = np.r_[xy, rim]
    closest = float(np.linalg.norm(segment-center, axis=1).min())
    crossing = None
    clearance = None
    lefts = np.arange(start, end)
    descending = (p[lefts, 2] > rim) & (p[lefts+1, 2] <= rim)
    found = np.flatnonzero(descending)
    if len(found):
        left = int(lefts[found[0]])
        fraction = float((p[left, 2]-rim)/(p[left, 2]-p[left+1, 2]))
        point_xy = p[left, :2] + fraction*(p[left+1, :2]-p[left, :2])
        local_xy = point_xy-xy
        distances = (planes[:, :2] @ local_xy + planes[:, 2]*rim + planes[:, 3])/horizontal
        clearance = float(distances.min()-radius)
        crossing = dict(time_s=float(times[left]+fraction*(times[left+1]-times[left])),
                        interval_index=left, fraction=fraction, xy_m=point_xy.tolist())
    return dict(version=VERSION, crossing_plane=CROSSING_PLANE, available=True,
                reason=None, start_index=start, end_index=end,
                start_time_s=float(times[start]), end_time_s=float(times[end]),
                cutoff_reason="contact" if len(contacts) else "horizon",
                cutoff_contact_channels=channels, first_crossing=crossing,
                mouth_clearance_m=clearance, closest_opening_distance_m=closest,
                sample_count=end-start+1)
