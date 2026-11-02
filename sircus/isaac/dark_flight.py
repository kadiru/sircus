"""Native perspective camera for the 60-second compact farm replay, in metres."""

from dataclasses import dataclass
from math import cos, pi, sin

import numpy as np


@dataclass(frozen=True)
class Shot:
    name: str
    source_batch: int
    focus: int | None
    frames: int
    stride: int
    move_frames: int = 0


# Each station shot repeats row 1, column 2 at the same bucket, with
# four distinct paired endpoints selected from the original training batches.
# The outcomes referenced during selection are not labels for the new run.
SHOTS = (
    Shot("farm-orbit", 9, None, 180, 4),
    Shot("aisle", 2, None, 120, 4),
    Shot("station-short-gap", 4, 1, 198, 4, 36),
    Shot("station-wide-bearing", 3, 1, 198, 4, 36),
    Shot("station-raised-board", 6, 1, 198, 4, 36),
    Shot("station-side-release", 0, 1, 198, 4, 36),
    Shot("front-column-three", 5, 2, 198, 4, 36),
    Shot("front-column-five", 2, 4, 198, 4, 36),
    Shot("overhead-ball", 3, 4, 144, 6, 36),
    Shot("overhead-farm", 8, None, 168, 4),
)


def ease(t):
    t = float(np.clip(t, 0, 1))
    return t * t * (3 - 2 * t)


def camera_pose(shot, frame, cells, origins, ball_world=None):
    """World-space eye and aim. Flight tracking uses the live rigid body's position."""
    t = frame / (shot.frames - 1)
    u = ease(t)
    center = np.array([0.67, -0.25, 0.15])
    if shot.name == "farm-orbit":
        # A low first pass keeps the foreground distinct from the brighter arms
        # behind it. Ease out to the entrance of the central aisle.
        a = -pi / 2 + 2 * pi * t
        eye = center + [17.5 * cos(a), 17.5 * sin(a), 8.7 + sin(pi * t)]
        return eye, center
    if shot.name == "aisle":
        # Center halfway between the central and next column. Stay above tabletop.
        x = (origins[3, 0] + origins[4, 0]) / 2 + center[0]
        eye = np.array([x, origins[3, 1] - 2.45 + u * 7.30, 1.18 - 0.29 * sin(pi * t)])
        aim = eye + [-0.62 * cos(pi * t), 2.45, -0.49]
        return eye, aim
    if shot.name == "overhead-farm":
        return center + [0.25, -1.35 - 1.1 * u, 17.2 - 1.4 * u], center

    origin = np.asarray(origins[shot.focus])
    cell = cells[shot.focus]
    bucket = np.asarray(cell["bucket_xy_m"])
    board = np.asarray(cell["board_xy_m"])
    if shot.name.startswith("station-"):
        # A fixed XY and axis make successive poses at this station comparable.
        xy = bucket + [0.22, -0.11]
        offset = np.array([0.70, -1.85, 0.85])
    else:
        xy = 0.54 * board + 0.46 * bucket
        offset = np.array([-0.75 if shot.focus == 2 else 0.95, -1.19, 0.69])
    anchor = origin + [*xy, 0.32]
    target = anchor.copy()
    eye = anchor + offset
    overhead = shot.name == "overhead-ball"
    if overhead:
        eye = anchor + np.array([0.12, -0.19, 2.30])
    if ball_world is None or frame < shot.move_frames:
        return eye, target

    ball = np.asarray(ball_world)
    elapsed = (frame - shot.move_frames) * shot.stride / 240
    follow = ease((elapsed - 0.28) / 0.34)
    delta = ball[:2] - anchor[:2]
    # Keep the board and bucket steady during the drop. Track an escaping ball
    # across the table edge rather than continuing to frame an empty board.
    outside = ease((np.linalg.norm(delta) - 0.14) / 0.40)
    shift = follow * outside * delta
    target[:2] += shift
    eye[:2] += shift
    low = follow * ease((0.14 - (ball[2] - origin[2])) / 0.64)
    if overhead:
        target[2] -= 0.80 * low
        eye[2] -= 0.64 * low
    else:
        target[2] -= 0.92 * low
        eye[2] -= (offset[2] + 0.32 + 0.33) * low
        # Remain in the open front aisle on the low follow-through, so a
        # neighboring tabletop cannot pass between the lens and a floor impact.
        eye[1] -= low * max(0, eye[1] - (origin[1] - 1.00))
    return eye, target
