"""Fixed camera and repeated foreground drops for the next station recording."""

import numpy as np
from dark_flight import Shot
from dark_stage import origins as farm_origins

FOREGROUND = 1  # Row 1, column 2.
MIN_FOREGROUND_DROPS = 10
SOURCE_BATCHES = (4, 3, 6, 0) * 3
SHOTS = tuple(
    Shot(f"station-{i + 1:02}", batch, FOREGROUND, 252, 6, 36)
    for i, batch in enumerate(SOURCE_BATCHES)
)


def camera_pose(shot, frame, cells, origins=None, ball_world=None):
    """Hold the same world-space view through every move, release and landing.

    The viewpoint includes the featured tabletop, nearby floor and the farm
    behind it. Ball position, shot configuration and playback time never steer
    this camera.
    """
    world = farm_origins() if origins is None else np.asarray(origins)
    anchor = world[FOREGROUND]
    return anchor + [1.95, -3.7, 1.85], anchor + [0.67, -0.2, -0.13]
