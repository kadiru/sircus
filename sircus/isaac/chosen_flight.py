"""Sixteen-second farm orbit/fly-in, then the user's fixed close-up camera."""

from math import atan2, cos, pi, sin

import numpy as np
from dark_flight import Shot, ease

PLAN = "chosen-farm-v1"
SHOTS = tuple(Shot(f"orbit-{i+1:02}", 6, None, 120, 6) for i in range(4)) + tuple(
    Shot(f"station-chosen-{i+1:02}", source, 1, 165, 4, 36)
    for i, source in enumerate((4, 3, 6, 0) * 2)
)


def camera_pose(shot, frame, cells, origins, ball_world=None, *, camera):
    """The chosen close-up never depends on a ball, joint, outcome or scene action."""
    selected_eye = np.asarray(camera["eye_m"], dtype=float)
    selected_aim = np.asarray(camera["target_m"], dtype=float)
    if shot.name.startswith("station-"):
        return selected_eye.copy(), selected_aim.copy()
    global_frame = SHOTS.index(shot) * 120 + frame
    center = np.array([0.67, -0.25, 0.15])
    bearing = atan2(selected_eye[1] - center[1], selected_eye[0] - center[0])
    if global_frame < 420:
        angle = bearing + 2 * pi * global_frame / 419
        return center + [14 * cos(angle), 14 * sin(angle), 6.8], center
    # Finish the full circle before the two-second approach. Every frame remains
    # camera-directed; no object tracking or keyframes derived from ball motion.
    far = center + [14 * cos(bearing), 14 * sin(bearing), 6.8]
    t = ease((global_frame - 419) / 60)
    return (1 - t) * far + t * selected_eye, (1 - t) * center + t * selected_aim
