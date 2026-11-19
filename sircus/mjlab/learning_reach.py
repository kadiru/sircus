"""Conservative UF850 wrist-distance proposal filter, not an IK certificate.

These bounds improve exploration efficiency. They cannot establish that a
rejected target is impossible, nor certify joint limits, collisions or arm paths.
Every selected proposal still needs the existing endpoint and native-contact
gates. Do not use this filter to change historical episode labels.
"""

import numpy as np

from .learning_contract import HIGH, LOW, proposals

VERSION = "sircus-mjlab-wrist-proposal-v1"
SHOULDER_Z_M = 0.364
WRIST_FLANGE_M = 0.09
DROPPER_FLANGE_TO_BALL_M = 0.095
# Actual two-link sum is .390 + hypot(.150,.426) = .84166 m. This 31.7 mm
# margin is an exploration prior, deliberately stricter than that outer bound.
WRIST_RADIUS_M = 0.81
MIN_BOARD_BUCKET_DISTANCE_M = 0.15


def maximum_gap(board_xyz, scene):
    """Gap cap from the dropper's fixed downward orientation and wrist bound."""
    xyz = np.asarray(board_xyz, dtype=float)
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError("Expected boardXYZ [N,3]")
    horizontal_squared = np.sum(xyz[:, :2] ** 2, axis=1)
    vertical = np.sqrt(np.maximum(WRIST_RADIUS_M**2 - horizontal_squared, 0))
    fixed = scene["ball"]["radius_m"] + DROPPER_FLANGE_TO_BALL_M + WRIST_FLANGE_M
    result = SHOULDER_Z_M + vertical - xyz[:, 2] - fixed
    return np.where(horizontal_squared <= WRIST_RADIUS_M**2, result, -np.inf)


def feasible_mask(board_xyz, tilt, bucket, gap, scene):
    """Return a conservative geometric proposal mask; acceptance needs actual IK.

    Inputs are metres and degrees, boardXYZ [N,3] and bucketXY [N,2]. Board +Z
    points towards the bucket with the requested tilt from horizontal. The board
    and release share XY; the dropper flange normal is fixed vertically down.
    """
    xyz = np.asarray(board_xyz, dtype=float)
    xy = np.asarray(bucket, dtype=float)
    if xyz.ndim != 2 or xyz.shape[1] != 3 or xy.shape != (len(xyz), 2):
        raise ValueError("Expected boardXYZ [N,3] and bucketXY [N,2]")
    angle = np.broadcast_to(np.asarray(tilt, dtype=float), (len(xyz),))
    gap = np.broadcast_to(np.asarray(gap, dtype=float), (len(xyz),))
    delta = xy - xyz[:, :2]
    distance = np.linalg.norm(delta, axis=1)
    direction = delta / np.maximum(distance[:, None], 1e-12)
    angle_rad = np.deg2rad(angle)
    normal = np.column_stack(
        (direction * np.sin(angle_rad)[:, None], np.cos(angle_rad))
    )
    wrist14 = xyz - (scene["board"]["tool_offset_m"] + WRIST_FLANGE_M) * normal
    shoulder14 = [scene["arms"]["base_spacing_m"], 0, SHOULDER_Z_M]
    wrist10 = xyz.copy()
    wrist10[:, 2] += (
        scene["ball"]["radius_m"] + gap + DROPPER_FLANGE_TO_BALL_M + WRIST_FLANGE_M
    )
    finite = (
        np.isfinite(xyz).all(1)
        & np.isfinite(xy).all(1)
        & np.isfinite(angle)
        & np.isfinite(gap)
    )
    return (
        finite
        & (angle >= 0)
        & (angle < 90)
        & (gap > 0)
        & (distance > MIN_BOARD_BUCKET_DISTANCE_M)
        & (
            np.linalg.norm(wrist10 - [0, 0, SHOULDER_Z_M], axis=1)
            <= WRIST_RADIUS_M + 1e-12
        )
        & (np.linalg.norm(wrist14 - shoulder14, axis=1) <= WRIST_RADIUS_M + 1e-12)
    )


def sample_proposals(bucket, rng, scene, *, max_tries=64):
    """Return ``(values[N,5], unresolved[N])`` after finite geometry resampling.

    Most candidates use the existing ballistic prior. Half also explore a wider
    XY/height family within the declared action limits. The gap is capped by the
    downward dropper wrist bound; this can reduce the ballistic energy estimate,
    so it does not predict success. No simulation outcome enters this proposal.
    Failed rows remain finite and explicitly unresolved rather than being called
    impossible or silently reusing a successful layout.
    """
    bucket = np.asarray(bucket, dtype=float)
    if bucket.ndim != 2 or bucket.shape[1] != 2 or not np.isfinite(bucket).all():
        raise ValueError("Expected finite bucketXY [N,2]")
    if not 1 <= max_tries <= 1024:
        raise ValueError("max_tries must be between 1 and 1024")
    values = np.zeros((len(bucket), 5))
    unresolved = np.ones(len(bucket), dtype=bool)
    for _ in range(max_tries):
        ids = np.flatnonzero(unresolved)
        if not len(ids):
            break
        candidate = proposals(bucket[ids], rng)
        broad = rng.random(len(ids)) < 0.5
        count = int(broad.sum())
        if count:
            candidate[broad, 0] = rng.uniform(0.43, 0.85, count)
            candidate[broad, 1] = rng.uniform(LOW[1], HIGH[1], count)
            candidate[broad, 2] = rng.uniform(0.22, 0.43, count)
            candidate[broad, 3] = rng.uniform(14, 43, count)
        cap = np.minimum(HIGH[4], maximum_gap(candidate[:, :3], scene))
        # Broad proposals deliberately retain a uniform energy component. The
        # rest preserve the analytic prior up to a conservative reach cap.
        if count:
            candidate[broad, 4] = LOW[4] + rng.random(count) * np.maximum(
                cap[broad] - LOW[4], 0
            )
        candidate[:, 4] = np.minimum(candidate[:, 4], cap)
        geometric = feasible_mask(
            candidate[:, :3], candidate[:, 3], bucket[ids], candidate[:, 4], scene
        )
        geometric &= (candidate >= LOW).all(1) & (candidate <= HIGH).all(1)
        # Keep a finite diagnostic action for exhausted rows without relaxing the
        # acceptance mask when even the minimum gap exceeds the wrist bound.
        values[ids] = np.nan_to_num(
            np.clip(candidate, LOW, HIGH), nan=LOW[4], neginf=LOW[4], posinf=HIGH[4]
        )
        unresolved[ids[geometric]] = False
    return values, unresolved
