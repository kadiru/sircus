"""One table-local bucket per tile. One shared policy chooses a point on its cylinder."""

import hashlib
import json
import math

import numpy as np
from board_route import VERSION as ROUTE_VERSION
from feasibility_contract import dynamic_cell
from parallel_contract import SPACING_M

ROWS, COLS = 5, 7
RADIUS = 0.400
SEED = 20261106
# Azimuth offset in degrees, board center Z, ball-bottom gap in metres, board tilt.
LOW = np.array([-180, 0.105, 0.10, 10.0])
HIGH = np.array([180, 0.445, 0.425, 42.0])
WARM = np.array([0, 0.285, 0.255, 26.0])
LOGSTD = (-1.25, -0.32, -0.85, -0.92)
REFERENCE = np.array([0.67, -0.10])
# Angle is measured from the direction from the bucket toward REFERENCE.
RING_OFFSETS = (0, -15, 15, -30, 30, -50, 50, -90, 90, -135, 135, 180)
POSES = (
    (0.380, 0.265, 26),
    (0.320, 0.240, 29),
    (0.320, 0.145, 24),
    (0.150, 0.235, 22),
)
# Additional bearings point into the overlap of the two arms' workspaces.
OVERLAP_POINTS = ((0.58, -0.10), (0.64, 0.01), (0.73, -0.16))


def origins():
    return np.array(
        [
            [(c - (COLS - 1) / 2) * SPACING_M, (r - (ROWS - 1) / 2) * SPACING_M, 0]
            for r in range(ROWS)
            for c in range(COLS)
        ],
        dtype=np.float32,
    )


def layouts(scene):
    center = np.array(scene["table"]["center_m"][:2])
    size = np.array(scene["table"]["size_m"][:2])
    lower = center - size / 2
    tile = size / [COLS, ROWS]
    bucket_radius = scene["bucket"]["outer_top_diameter_m"] / 2
    bases = np.array([[0, 0], [scene["arms"]["base_spacing_m"], 0]])
    result = []
    for row in range(ROWS):
        for col in range(COLS):
            tile_row = ROWS - row - 1
            corner = lower + [col, tile_row] * tile
            midpoint = corner + tile / 2
            # Pick the closest point to the tile center with its footprint on the
            # tabletop and a 170 mm center-to-base exclusion for the site layout.
            lo = np.maximum(corner + 0.001, lower + bucket_radius + 0.001)
            hi = np.minimum(corner + tile - 0.001, lower + size - bucket_radius - 0.001)
            candidates = [midpoint] + [
                np.array([x, y])
                for x in np.linspace(lo[0], hi[0], 9)
                for y in np.linspace(lo[1], hi[1], 9)
            ]
            allowed = [
                p
                for p in candidates
                if np.all(p >= lo)
                and np.all(p <= hi)
                and np.linalg.norm(bases - p, axis=1).min() >= 0.170
            ]
            if not allowed:
                raise ValueError(f"No bucket site in tile {(row, col)}")
            xy = min(allowed, key=lambda p: np.linalg.norm(p - midpoint))
            cell = dict(
                id=f"r{row+1}c{col+1}",
                index=row * COLS + col,
                grid_row=row,
                grid_column=col,
                bucket_tile_row=tile_row,
                bucket_xy_m=xy.tolist(),
            )
            result.append(dynamic_cell(cell, shot(cell, WARM)))
    return result


def observation(cell):
    # Absolute location matters because the two arm bases are fixed.
    x, y = cell["bucket_xy_m"]
    dx, dy = REFERENCE - [x, y]
    bearing = math.atan2(dy, dx)
    return [
        float((x - 0.67) / 0.9),
        float((y + 0.25) / 0.625),
        math.hypot(dx, dy),
        math.sin(bearing),
        math.cos(bearing),
    ]


def shot(cell, values):
    offset, height, gap, tilt = map(float, values)
    xy = cell["bucket_xy_m"]
    dx, dy = REFERENCE - xy
    bearing = math.atan2(dy, dx) + math.radians(offset)
    return dict(
        id="cylinder-ppo",
        azimuth_offset_deg=offset,
        azimuth_deg=math.degrees(bearing) % 360,
        board_x_m=xy[0] + RADIUS * math.cos(bearing),
        board_y_m=xy[1] + RADIUS * math.sin(bearing),
        height_m=height,
        gap_m=gap,
        tilt_deg=tilt,
    )


def actions(cells, latents):
    values = LOW + (np.tanh(latents) + 1) * (HIGH - LOW) / 2
    if values.shape != (len(cells), 4) or not np.isfinite(values).all():
        raise ValueError("Expected four finite actions for every workcell")
    return [shot(c, v) for c, v in zip(cells, values)]


def menu_batches(cells):
    """Geometry-only candidates. No simulation outcome or reward is consulted."""
    offsets = []
    for cell in cells:
        xy = np.asarray(cell["bucket_xy_m"])
        base_dx, base_dy = REFERENCE - xy
        base = math.atan2(base_dy, base_dx)
        towards_overlap = []
        for target in OVERLAP_POINTS:
            dx, dy = np.asarray(target) - xy
            delta = math.degrees(math.atan2(dy, dx) - base)
            towards_overlap.append((delta + 180) % 360 - 180)
        offsets.append([0, *towards_overlap, *RING_OFFSETS[1:]])
    for j in range(len(RING_OFFSETS) + len(OVERLAP_POINTS)):
        for pose_index, pose in enumerate(POSES):
            proposal_id = f"ring-{j:02d}-pose-{pose_index}"
            yield [
                dict(shot(c, [angles[j], *pose]), id=proposal_id)
                for c, angles in zip(cells, offsets)
            ]


def project_actions(cells, latents, menus):
    """Select the closest qualified prototype in the PPO's four latent coordinates."""
    raw = actions(cells, latents)
    if len(menus) != len(cells):
        raise ValueError("Expected a preflight menu for every workcell")
    result = []
    for proposal, latent, options in zip(raw, latents, menus):
        if not options:
            result.append(proposal)
            continue
        values = np.array(
            [
                [a["azimuth_offset_deg"], a["height_m"], a["gap_m"], a["tilt_deg"]]
                for a in options
            ]
        )
        # Clipping gives the +/-180-degree ring endpoints finite latent images.
        scaled = (values - LOW) * 2 / (HIGH - LOW) - 1
        menu_latents = np.arctanh(np.clip(scaled, -1 + 1e-6, 1 - 1e-6))
        best = np.argmin(np.sum((menu_latents - latent) ** 2, axis=1))
        result.append(options[best].copy())
    return result


def specification(scene, batches):
    spec = dict(
        version="sircus-cylinder-film-v2",
        route_version=ROUTE_VERSION,
        cells=layouts(scene),
        scene=scene,
        seed=SEED,
        radius_m=RADIUS,
        num_envs=ROWS * COLS,
        rows=ROWS,
        columns=COLS,
        origins_m=origins().tolist(),
        bucket_assignment="Physical front row maps to the +Y edge of the tabletop.",
        preflight_menu=dict(
            angle_offsets_deg=list(RING_OFFSETS),
            extra_aim_points_m=list(OVERLAP_POINTS),
            height_gap_tilt=list(POSES),
            projection="Closest qualified action in latent space, unprojected proposal if none qualify.",
        ),
        action_names=[
            "azimuth_offset_deg",
            "board_height_m",
            "ball_bottom_gap_m",
            "tilt_deg",
        ],
        action_low=LOW.tolist(),
        action_high=HIGH.tolist(),
        initial_action=WARM.tolist(),
        initial_logstd=list(LOGSTD),
        training_batches=batches,
        batches_per_update=2,
        epochs=4,
        lr=0.0003,
        clip=0.2,
        value_coef=0.5,
        entropy_coef=0.001,
        grad_norm=1.0,
        scope="Privileged one-shot PPO, solved endpoints, fixed radius and inward heading.",
    )
    spec["fingerprint"] = hashlib.sha256(
        json.dumps(spec, sort_keys=True).encode()
    ).hexdigest()
    return spec
