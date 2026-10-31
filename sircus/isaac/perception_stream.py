"""Bounded, atomic camera observations plus a separate evaluation-only truth file."""

import datetime
import json
import uuid
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

STREAM_ID = uuid.uuid4().hex


def publish_sensor(output, frame, rgb, depth, camera):
    output = Path(output)
    sensor = output / "sensors"
    sensor.mkdir(exist_ok=True)
    q = camera.data.quat_w_ros[0].cpu().numpy()
    transform = np.eye(4)
    transform[:3, :3] = Rotation.from_quat(q[[1, 2, 3, 0]]).as_matrix()
    transform[:3, 3] = camera.data.pos_w[0].cpu().numpy()
    stamp = datetime.datetime.now(datetime.UTC).isoformat()
    frame_id = f"{STREAM_ID}:{frame}"
    # One bundle prevents mixed RGB/depth/calibration snapshots; overwrite atomically.
    with (sensor / "latest.tmp").open("wb") as f:
        np.savez_compressed(
            f,
            rgb=rgb,
            depth_m=depth,
            intrinsics=camera.data.intrinsic_matrices[0].cpu().numpy(),
            camera_to_world=transform,
            frame_id=frame_id,
            timestamp_utc=stamp,
        )
    (sensor / "latest.tmp").replace(sensor / "latest.npz")
    return frame_id


def publish_truth(output, frame_id, cfg, arms, ball):
    board = arms["Arm14"]
    idx = board.body_names.index("impact_board")
    q = board.data.body_quat_w[0, idx].cpu().numpy()
    normal = Rotation.from_quat(q[[1, 2, 3, 0]]).apply([0, 0, 1])
    center = (
        board.data.body_pos_w[0, idx].cpu().numpy()
        + normal * cfg["board"]["size_m"][2] / 2
    )
    table = cfg["table"]
    tc = list(table["center_m"])
    tc[2] += table["size_m"][2] / 2
    b = cfg["bucket"]
    data = {
        "frame_id": frame_id,
        "provenance": "simulator truth; evaluation only",
        "objects": {
            "table": {"center_m": tc, "normal": [0, 0, 1]},
            "board": {"center_m": center.tolist(), "normal": normal.tolist()},
            "bucket": {
                "center_m": [
                    *b["center_xy_m"],
                    b["assumed_interior_depth_m"] + b["assumed_bottom_thickness_m"],
                ],
                "normal": [0, 0, 1],
            },
            "ball": {"center_m": ball.data.root_pos_w[0].cpu().tolist()},
        },
    }
    target = Path(output) / "evaluation_truth"
    target.mkdir(exist_ok=True)
    (target / "latest.tmp").write_text(json.dumps(data, allow_nan=False))
    (target / "latest.tmp").replace(target / "latest.json")
