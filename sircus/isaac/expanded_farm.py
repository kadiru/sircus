"""Opt-in five-parameter shots with the original farm geometry and saved camera.

Only the proposal/action contract changes. Physics, bucket definitions, scoring,
privileged observations and the selected source-indexed display layout do not.
"""

import copy
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from camera_options import preview_origins as origins
from cylinder_farm import COLS, REFERENCE, ROWS, SEED, observation
from cylinder_farm import specification as legacy_specification
from saved_camera import load_camera

VERSION = "sircus-expanded-cylinder-film-v1"
CAMERA_BINDING_SCHEMA = "sircus-expanded-camera-binding-v1"
CAMERA_BINDING_FILENAME = "camera-binding.json"
CAMERA_SOURCE_BATCHES = 10
CAMERA_SOURCE_FINGERPRINT = (
    "98ebe47b48a01b9028747ab03bc913a033d90611e44f03b912e58f41c87ff79a"
)
CAMERA_SOURCE_CONTRACT = Path(__file__).with_name("camera-source-contract.json")
ACTION_NAMES = ("azimuth_offset_deg", "radius_m", "height_m", "gap_m", "tilt_deg")
ACTION_DIM = len(ACTION_NAMES)
LOW = np.array([-180.0, 0.28, 0.14, 0.08, 12.0])
HIGH = np.array([180.0, 1.20, 0.54, 0.56, 52.0])
WARM = np.array([0.0, 0.40, 0.32, 0.24, 29.0])
LOGSTD = (-1.25, -0.80, -0.32, -0.85, -0.92)


def _json(value):
    return json.dumps(value, sort_keys=True, allow_nan=False)


def _hash(value):
    return hashlib.sha256(_json(value).encode()).hexdigest()


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _canonical_scene():
    return json.loads(Path(__file__).with_name("scene.json").read_text())


def _same_legacy(actual, expected):
    """Permit only numerical ULPs while independently checking the frozen layout."""
    if isinstance(expected, dict):
        return (
            isinstance(actual, dict)
            and set(actual) == set(expected)
            and all(_same_legacy(actual[key], value) for key, value in expected.items())
        )
    if isinstance(expected, (list, tuple)):
        return (
            isinstance(actual, (list, tuple))
            and len(actual) == len(expected)
            and all(_same_legacy(a, b) for a, b in zip(actual, expected))
        )
    if isinstance(expected, (int, float)) and not isinstance(expected, bool):
        return (
            isinstance(actual, (int, float))
            and not isinstance(actual, bool)
            and math.isfinite(actual)
            and abs(actual - expected) <= 1e-12
        )
    return actual == expected


def _legacy(scene):
    _require(
        _json(scene) == _json(_canonical_scene()),
        "Expanded farm requires the unchanged original static scene geometry",
    )
    legacy = json.loads(CAMERA_SOURCE_CONTRACT.read_bytes())
    unsigned = copy.deepcopy(legacy)
    fingerprint = unsigned.pop("fingerprint")
    _require(
        fingerprint == CAMERA_SOURCE_FINGERPRINT and _hash(unsigned) == fingerprint,
        "Pinned original camera source contract fingerprint mismatch",
    )
    _require(
        _json(legacy["scene"]) == _json(scene),
        "Pinned original camera source static geometry mismatch",
    )
    regenerated = legacy_specification(copy.deepcopy(scene), CAMERA_SOURCE_BATCHES)
    regenerated.pop("fingerprint")
    _require(
        _same_legacy(unsigned, regenerated),
        "Pinned camera source differs from the original layout beyond numerical ULPs",
    )
    return legacy


def _geometry(legacy):
    return dict(
        version=legacy["version"],
        fingerprint=legacy["fingerprint"],
        camera_source_batches=CAMERA_SOURCE_BATCHES,
        scene_sha256=_hash(legacy["scene"]),
        original_cells_sha256=_hash(legacy["cells"]),
        source_contract_file_sha256=hashlib.sha256(
            CAMERA_SOURCE_CONTRACT.read_bytes()
        ).hexdigest(),
        actual_display_origins_sha256=_hash(origins().tolist()),
    )


def is_expanded(spec):
    return isinstance(spec, dict) and spec.get("version") == VERSION


def make_spec(scene, batches):
    """Create a new experiment identity without relabeling the old camera scene."""
    _require(type(batches) is int and batches > 0, "Expected a positive batch count")
    legacy = _legacy(scene)
    spec = copy.deepcopy(legacy)
    spec.pop("fingerprint")
    spec.pop("radius_m")
    spec.update(
        version=VERSION,
        camera_source_fingerprint=legacy["fingerprint"],
        base_geometry=_geometry(legacy),
        origins_m=origins().tolist(),
        action_dim=ACTION_DIM,
        action_names=list(ACTION_NAMES),
        action_low=LOW.tolist(),
        action_high=HIGH.tolist(),
        initial_action=WARM.tolist(),
        initial_logstd=list(LOGSTD),
        radius_mode="variable board-to-bucket horizontal distance",
        radius_bounds_m=[float(LOW[1]), float(HIGH[1])],
        training_batches=batches,
        preflight_menu=dict(
            version="sircus-expanded-pose-menu-v1",
            qualification="Native endpoint IK and measured settling, without release outcomes or rewards.",
            projection="Closest qualified prototype in five-dimensional latent coordinates; empty menus are rejected.",
            minimum_qualified_options_per_cell=2,
        ),
        scope="Privileged one-shot PPO over azimuth, radius, board height, drop gap and tilt; native-qualified endpoints, fixed inward heading, scripted transit and initialized ball releases.",
    )
    spec["fingerprint"] = _hash(spec)
    return spec


def _vector(value, shape, label):
    result = np.asarray(value, dtype=float)
    _require(result.shape == shape and np.isfinite(result).all(), f"Invalid {label}")
    return result


def _values(value):
    result = _vector(value, (ACTION_DIM,), "five-dimensional shot parameters")
    _require(
        np.all(result >= LOW) and np.all(result <= HIGH),
        "Expanded shot parameters are outside their bounds",
    )
    return result


def shot(cell, values):
    """Decode bucket-relative azimuth/radius while keeping the board aimed inward."""
    offset, radius, height, gap, tilt = _values(values)
    xy = _vector(cell["bucket_xy_m"], (2,), "bucket position")
    dx, dy = REFERENCE - xy
    bearing = math.atan2(dy, dx) + math.radians(offset)
    return dict(
        id="expanded-cylinder-ppo",
        azimuth_offset_deg=float(offset),
        azimuth_deg=math.degrees(bearing) % 360,
        radius_m=float(radius),
        board_x_m=float(xy[0] + radius * math.cos(bearing)),
        board_y_m=float(xy[1] + radius * math.sin(bearing)),
        height_m=float(height),
        gap_m=float(gap),
        tilt_deg=float(tilt),
    )


def actions(cells, latents):
    latent = _vector(latents, (len(cells), ACTION_DIM), "five-dimensional latents")
    values = LOW + (np.tanh(latent) + 1) * (HIGH - LOW) / 2
    return [shot(cell, value) for cell, value in zip(cells, values)]


def validate_action(cell, action):
    """Reject a menu entry whose radius or bearing disagrees with its board XY."""
    values = _values([action[name] for name in ACTION_NAMES])
    expected = shot(cell, values)
    actual_xy = _vector(
        [action["board_x_m"], action["board_y_m"]], (2,), "menu board position"
    )
    bucket_xy = _vector(cell["bucket_xy_m"], (2,), "bucket position")
    _require(
        abs(float(np.linalg.norm(actual_xy - bucket_xy)) - values[1]) < 2e-7,
        "Qualified action radius disagrees with its board-to-bucket distance",
    )
    _require(
        np.allclose(
            actual_xy, [expected["board_x_m"], expected["board_y_m"]], atol=2e-7, rtol=0
        ),
        "Qualified action bearing disagrees with its board position",
    )
    azimuth = float(action["azimuth_deg"])
    _require(
        math.isfinite(azimuth)
        and abs((azimuth - expected["azimuth_deg"] + 180) % 360 - 180) < 2e-5,
        "Qualified action azimuth disagrees with its bucket-relative offset",
    )
    if "distance_m" in action:
        distance = float(action["distance_m"])
        _require(
            math.isfinite(distance) and abs(distance - values[1]) < 2e-7,
            "Qualified action distance disagrees with its radius",
        )
    return values


def project_actions(cells, latents, menus):
    """Select only native-qualified menu entries, never an unqualified fallback."""
    latent = _vector(latents, (len(cells), ACTION_DIM), "five-dimensional latents")
    _require(len(menus) == len(cells), "Expected one qualified menu per workcell")
    projected = []
    for cell, proposal, options in zip(cells, latent, menus):
        _require(
            bool(options), f"No qualified action menu for {cell.get('id', 'cell')}"
        )
        values = np.array([validate_action(cell, option) for option in options])
        scaled = 2 * (values - LOW) / (HIGH - LOW) - 1
        menu_latents = np.arctanh(np.clip(scaled, -1 + 1e-6, 1 - 1e-6))
        nearest = int(np.argmin(np.sum((menu_latents - proposal) ** 2, axis=1)))
        projected.append(copy.deepcopy(options[nearest]))
    return projected


def _validate_binding_spec(spec, actual_origins):
    _require(
        is_expanded(spec), "Camera binding requires the expanded experiment contract"
    )
    unsigned = copy.deepcopy(spec)
    fingerprint = unsigned.pop("fingerprint")
    _require(_hash(unsigned) == fingerprint, "Expanded experiment fingerprint mismatch")
    legacy = _legacy(spec["scene"])
    _require(
        spec["camera_source_fingerprint"] == legacy["fingerprint"]
        and _json(spec["base_geometry"]) == _json(_geometry(legacy)),
        "Original camera source geometry fingerprint mismatch",
    )
    _require(
        spec["num_envs"] == ROWS * COLS == 35
        and spec["rows"] == ROWS
        and spec["columns"] == COLS
        and _json(spec["cells"]) == _json(legacy["cells"]),
        "Expanded experiment changed the original 35 bucket definitions",
    )
    _require(
        spec["action_dim"] == ACTION_DIM
        and spec["action_names"] == list(ACTION_NAMES)
        and spec["action_low"] == LOW.tolist()
        and spec["action_high"] == HIGH.tolist(),
        "Expanded experiment action contract mismatch",
    )
    actual = _vector(actual_origins, (35, 3), "actual display origins")
    _require(
        np.array_equal(actual, origins())
        and np.array_equal(np.asarray(spec["origins_m"]), origins()),
        "Actual display origins do not match the unchanged selected-camera farm",
    )
    return legacy, actual


def bind_camera(path, spec, actual_origins, destination=None):
    """Load against the original fingerprint and optionally copy exact camera bytes.

    The camera remains bound to its original scene; a separate binding records
    this expanded experiment. No fingerprint in the original JSON is rewritten.
    """
    legacy, actual = _validate_binding_spec(spec, actual_origins)
    path = Path(path)
    raw = path.read_bytes()
    camera = load_camera(path, legacy["fingerprint"])
    _require(path.read_bytes() == raw, "Selected camera changed during validation")
    binding = dict(
        schema=CAMERA_BINDING_SCHEMA,
        camera_sha256=hashlib.sha256(raw).hexdigest(),
        camera_source_fingerprint=legacy["fingerprint"],
        experiment_fingerprint=spec["fingerprint"],
        base_geometry=copy.deepcopy(spec["base_geometry"]),
        actual_origins_m=actual.tolist(),
        scope="Original selected camera bytes and static farm retained; only the five-dimensional shot experiment has a new fingerprint.",
    )
    if destination is not None:
        destination = Path(destination)
        if destination.exists():
            _require(
                destination.read_bytes() == raw,
                "Camera destination already has different bytes",
            )
        else:
            destination.write_bytes(raw)
    return camera, binding


def verify_camera_binding(run, spec, actual_origins):
    """Recompute the binding of a saved run without modifying any artifact."""
    run = Path(run)
    camera, expected = bind_camera(run / "selected-camera.json", spec, actual_origins)
    saved = json.loads((run / CAMERA_BINDING_FILENAME).read_text())
    _require(
        _json(saved) == _json(expected), "Saved camera experiment binding mismatch"
    )
    return camera, expected
