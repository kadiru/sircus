"""A single perspective camera, in metres; the front row looks into the farm."""

import math

import numpy as np
from cylinder_farm import COLS, origins

# name, number of distinct rendered frames, 240 Hz physics steps per output frame.
# Playback is 30 fps. Two steps/frame shows the close-ups at quarter speed.
SHOTS = (
    ("orbit-start", 135, 4),
    ("orbit-back", 135, 4),
    ("orbit-return", 135, 4),
    ("descend", 120, 4),
    ("through-aisle", 150, 4),
    ("front-left", 144, 2),
    ("front-right", 144, 2),
    ("overhead-field", 105, 4),
    ("overhead-bounce", 132, 2),
    ("aerial-exit", 105, 4),
)


def choose_front(valid, shot, previous=None):
    if shot not in ("front-left", "front-right", "overhead-bounce"):
        return None
    order = [2, 3, 1, 4, 5, 0, 6] if shot == "front-left" else [4, 3, 5, 2, 1, 6, 0]
    candidates = [i for i in order if i < COLS and valid[i] and i != previous]
    if not candidates:
        raise ValueError(f"No settled front-row release for {shot}")
    return candidates[0]


def camera_pose(index, frame, cells, focus=None):
    name, count, _ = SHOTS[index]
    t = np.clip(frame / (count - 1), 0, 1)
    u = t * t * (3 - 2 * t)
    center = np.array([0.67, -0.25, 0.34])
    entry = np.array([2.92, -13.2, 1.65])
    forward = np.array([2.0, -7.7, 0.58])
    if index < 3:
        # Same angular velocity across the two batch boundaries.
        theta = -math.pi / 2 + 2 * math.pi * (index * count + frame) / (3 * count - 1)
        eye = center + [
            31 * math.cos(theta),
            31 * math.sin(theta),
            13 + 1.6 * math.sin(theta + math.pi / 2),
        ]
        return eye, center
    if name == "descend":
        return (1 - u) * (center + [0, -31, 13]) + u * entry, (
            1 - u
        ) * center + u * forward
    if name == "through-aisle":
        eye = entry + [0, 16 * u, -0.32 * math.sin(math.pi * u)]
        return eye, forward + [1.55 * math.sin(math.pi * u), 16 * u, 0]
    if name == "overhead-field":
        return center + [1.6 * u, -0.45, 34.0 - 3 * u], center + [1.6 * u, 0, 0]
    if name == "aerial-exit":
        return center + [3 + 5 * u, -2 - 15 * u, 26 - 9 * u], center
    cell = cells[focus]
    board, bucket = np.array(cell["board_xy_m"]), np.array(cell["bucket_xy_m"])
    xy = 0.62 * board + 0.38 * bucket
    target = origins()[focus] + [*xy, 0.42]
    if name == "overhead-bounce":
        return target + [0.13 + 0.24 * u, -0.16, 2.22 - 0.26 * u], target
    side = -1 if name == "front-left" else 1
    return (
        target + [side * (1.15 - 0.18 * u), -1.80 + 0.18 * u, 0.69 - 0.07 * u],
        target,
    )
