"""Dependency-light geometry for the Isaac foundation (SI units, no hardware)."""

import math


def board_world_transform(board):
    """Landscape rotation about the board normal, then tilt toward world -Y."""
    import numpy as np

    tilt = math.radians(board["initial_tilt_deg"])
    yaw = board["mount_yaw_rad"]
    rx = np.array(
        [
            [1, 0, 0],
            [0, math.cos(tilt), -math.sin(tilt)],
            [0, math.sin(tilt), math.cos(tilt)],
        ]
    )
    rz = np.array(
        [
            [math.cos(yaw), -math.sin(yaw), 0],
            [math.sin(yaw), math.cos(yaw), 0],
            [0, 0, 1],
        ]
    )
    transform = np.eye(4)
    transform[:3, :3] = rx @ rz
    transform[:3, 3] = board["initial_center_m"]
    return transform


def bucket_wedge(bucket, index):
    """Eight corners of one convex tapered wall segment; open top stays empty."""
    n = bucket["segments"]
    a, b = (2 * math.pi * i / n for i in (index, index + 1))
    z0 = bucket["assumed_bottom_thickness_m"]
    z1 = z0 + bucket["assumed_interior_depth_m"]
    radii = [
        (bucket["inner_bottom_diameter_m"] / 2, z0),
        (bucket["outer_bottom_diameter_m"] / 2, 0.0),
        (bucket["inner_top_diameter_m"] / 2, z1),
        (bucket["outer_top_diameter_m"] / 2, z1),
    ]
    return [(r * math.cos(t), r * math.sin(t), z) for r, z in radii for t in (a, b)]


WEDGE_FACES = [
    (0, 1, 5, 4),
    (2, 6, 7, 3),
    (0, 2, 3, 1),
    (4, 5, 7, 6),
    (0, 4, 6, 2),
    (1, 3, 7, 5),
]


def depth_to_points(depth, intrinsics):
    """Axial depth in metres -> optical camera XYZ (+X right,+Y down,+Z forward)."""
    import numpy as np

    v, u = np.indices(depth.shape)
    valid = np.isfinite(depth) & (depth > 0)
    z = depth[valid]
    return np.column_stack(
        (
            (u[valid] - intrinsics[0, 2]) * z / intrinsics[0, 0],
            (v[valid] - intrinsics[1, 2]) * z / intrinsics[1, 1],
            z,
        )
    ), valid
