"""Pure measured board-settling contract; no simulator imports."""

import numpy as np
from scipy.spatial.transform import Rotation

POSITION_TOL_M = 0.002
NORMAL_TOL_DEG = 0.5
VELOCITY_TOL_RAD_S = 0.005
MAX_CORRECTIONS = 3


def pose_errors(target, center, normal):
    return {
        "position_error_mm": float(np.linalg.norm(target[:3, 3] - center) * 1000),
        "normal_error_deg": float(
            np.rad2deg(np.arccos(np.clip(target[:3, 2] @ normal, -1, 1)))
        ),
    }


def correction(command, target, measured_center, measured_normal):
    result = command.copy()
    result[:3, 3] += target[:3, 3] - measured_center
    axis = np.cross(measured_normal, target[:3, 2])
    n = np.linalg.norm(axis)
    if n > 1e-12:
        angle = np.arctan2(n, measured_normal @ target[:3, 2])
        result[:3, :3] = (
            Rotation.from_rotvec(axis * angle / n).as_matrix() @ command[:3, :3]
        )
    return result


def accepted(record):
    return (
        record["position_error_mm"] <= POSITION_TOL_M * 1000
        and record["normal_error_deg"] <= NORMAL_TOL_DEG
        and record["joint_speed_rad_s"] <= VELOCITY_TOL_RAD_S
        and record["position_drift_mm"] <= 0.1
        and record["normal_drift_deg"] <= 0.05
    )
