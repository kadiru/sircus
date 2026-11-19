"""Validate a camera-scout export before using it in a native Isaac recording.

Only the viewpoint is imported. Scene identity and the fixed film intrinsics
must match; camera files never replace geometry, scoring or robot state.
"""

import json
import math
from pathlib import Path

SCHEMA = "sircus-camera-scout-camera-v1"
INTRINSICS = {
    "focal_length_mm": 23,
    "horizontal_aperture_mm": 36,
    "vertical_aperture_mm": 20.25,
}
RESOLUTION = [1920, 1080]


def _reject_constant(value):
    raise ValueError(f"Camera JSON contains non-finite constant {value}")


def _vector(camera, name):
    value = camera.get(name)

    def finite_number(number):
        if isinstance(number, bool) or not isinstance(number, (int, float)):
            return False
        try:
            return math.isfinite(number)
        except OverflowError:
            return False

    if (
        not isinstance(value, list)
        or len(value) != 3
        or not all(finite_number(v) for v in value)
    ):
        raise ValueError(f"Camera {name} must contain three finite numbers")
    return value


def load_camera(path, expected_fingerprint):
    """Return the complete JSON-ready camera after strict compatibility checks.

    ``expected_fingerprint`` must come from the source scene contract, rather
    than the camera file itself. Invalid camera metadata raises ``ValueError``;
    ordinary file errors propagate to the caller.
    """
    if not isinstance(expected_fingerprint, str) or not expected_fingerprint:
        raise ValueError("A nonempty expected source fingerprint is required")
    camera = json.loads(
        Path(path).read_text(encoding="utf-8"), parse_constant=_reject_constant
    )
    if not isinstance(camera, dict) or camera.get("schema") != SCHEMA:
        raise ValueError(f"Camera schema must be {SCHEMA}")
    if camera.get("source_fingerprint") != expected_fingerprint:
        raise ValueError("Camera source fingerprint does not match the scene")
    if camera.get("up_axis") != "Z":
        raise ValueError("Camera up_axis must be Z")
    eye, target = _vector(camera, "eye_m"), _vector(camera, "target_m")
    distance = math.dist(eye, target)
    if not math.isfinite(distance) or distance <= 0.02:
        raise ValueError("Camera eye and target must be more than 0.02 m apart")
    for name, expected in INTRINSICS.items():
        value = camera.get(name)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or value != expected
        ):
            raise ValueError(f"Camera {name} must be {expected}")
    resolution = camera.get("resolution")
    if (
        not isinstance(resolution, list)
        or any(type(value) is not int for value in resolution)
        or resolution != RESOLUTION
    ):
        raise ValueError("Camera resolution must be [1920, 1080]")
    # Overflowed numeric literals can occur even in otherwise valid JSON (1e999).
    # Ensure the full returned metadata, including additional provenance fields,
    # remains strict JSON, without silently discarding or rewriting fields.
    try:
        json.dumps(camera, allow_nan=False)
    except ValueError as error:
        raise ValueError(
            "Camera metadata must contain only finite JSON values"
        ) from error
    return camera
