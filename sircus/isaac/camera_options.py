"""Ten fixed native-camera compositions of one unchanged silver-gray farm.

The validated source workcell remains r1c2. Exchanging its entire world origin
with r1c7 puts it at the right-front display corner without changing its local
bucket, arms, board or release geometry. This is a composition preview, not a
new rollout or a successful shot at r1c7's original target.
"""

import copy

import numpy as np
from dark_stage import origins as farm_origins

SOURCE_FOCUS = 1
DISPLAY_SLOT = 6
SOURCE_BATCH = 6
FOCAL_LENGTH_MM = 23

# Offsets are metres from the focused workcell's display origin. Neither scene
# state nor the ball position changes these camera poses.
_OPTIONS = (
    (
        "Original perspective, right-front corner",
        (1.95, -3.70, 1.85),
        (0.67, -0.20, -0.13),
    ),
    ("Close diagonal", (2.55, -3.20, 1.45), (0.35, 0.10, 0.15)),
    ("Low diagonal", (2.85, -3.25, 1.15), (0.35, 0.10, 0.20)),
    ("Balanced diagonal", (3.00, -3.60, 1.65), (0.35, 0.10, 0.15)),
    ("Inward diagonal", (3.35, -3.40, 1.60), (0.10, 0.25, 0.20)),
    ("Across the front row", (3.60, -2.50, 1.35), (0.15, 0.05, 0.18)),
    ("Along the right aisle", (2.20, -4.40, 1.40), (0.30, 0.35, 0.20)),
    ("Elevated close diagonal", (2.80, -3.40, 2.40), (0.40, 0.00, 0.05)),
    ("Wide diagonal", (4.10, -4.50, 2.20), (0.15, 0.35, 0.20)),
    ("Low wide diagonal", (4.20, -4.80, 1.50), (0.20, 0.40, 0.20)),
)


def preview_origins():
    """Return source-indexed world origins with the two whole cells exchanged."""
    world = farm_origins()
    world[[SOURCE_FOCUS, DISPLAY_SLOT]] = world[[DISPLAY_SLOT, SOURCE_FOCUS]]
    return world


def camera_options():
    """Return fresh JSON-ready fixed camera choices; no simulator is required."""
    anchor = preview_origins()[SOURCE_FOCUS]
    return [
        dict(
            id=index,
            name=name,
            eye_m=(anchor + eye).tolist(),
            target_m=(anchor + target).tolist(),
            eye_offset_m=list(eye),
            target_offset_m=list(target),
            focus_source="r1c2",
            focus_display="r1c7",
        )
        for index, (name, eye, target) in enumerate(_OPTIONS, 1)
    ]


def adjusted_camera(option, right_m=0.0, z_m=None):
    """Shift in the original camera's horizontal right direction; retain its aim."""
    if not np.isfinite(right_m) or (z_m is not None and not np.isfinite(z_m)):
        raise ValueError("Camera adjustments must be finite")
    result = copy.deepcopy(option)
    eye = np.array(option["eye_m"], dtype=float)
    direction = np.array(option["target_m"]) - eye
    right = np.cross(direction, [0, 0, 1])
    norm = np.linalg.norm(right)
    if norm < 1e-8:
        raise ValueError("Cannot define horizontal right for a vertical camera")
    eye += right_m * right / norm
    if z_m is not None:
        eye[2] = z_m
    anchor = preview_origins()[SOURCE_FOCUS]
    result.update(
        eye_m=eye.tolist(),
        eye_offset_m=(eye - anchor).tolist(),
        adjustment=dict(
            right_m=right_m,
            z_m=z_m,
            right_reference="Original camera horizontal screen-right",
        ),
    )
    return result
