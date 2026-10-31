"""Verify a saved synthetic RGB-D capture without starting Isaac or hardware."""

import argparse
import json
from pathlib import Path

import numpy as np
from geometry import depth_to_points
from PIL import Image
from scipy.spatial.transform import Rotation


def inspect(path):
    path = Path(path)
    m = json.loads((path / "metadata.json").read_text())
    rgb = np.asarray(Image.open(path / "rgb.png"))
    depth = np.load(path / "depth_m.npy", allow_pickle=False)
    c = m["camera"]
    k = np.asarray(c["intrinsics"])
    assert rgb.shape == (c["height"], c["width"], 3)
    assert depth.shape == rgb.shape[:2]
    assert float(rgb.std()) > 5, "Rendered RGB is blank"
    points, _valid = depth_to_points(depth, k)
    assert len(points) == c["valid_depth_pixels"] and len(points) > 1000
    w, x, y, z = c["world_quaternion_wxyz_ros"]
    world = points @ Rotation.from_quat([x, y, z, w]).as_matrix().T + np.asarray(
        c["world_position_m"]
    )
    # Test the known synthetic tabletop, not a claimed perception algorithm.
    table = m["geometry"]["table"]
    cx, cy, cz = table["center_m"]
    sx, sy, sz = table["size_m"]
    visible_table = (
        (abs(world[:, 0] - cx) < sx * 0.45)
        & (abs(world[:, 1] - cy) < sy * 0.45)
        & (abs(world[:, 2] - (cz + sz / 2)) < 0.002)
    )
    count = int(visible_table.sum())
    assert count > 500, "Depth unprojection/extrinsics do not recover the known table"
    assert len(m["arms"]["Arm10"]["joint_names"]) == 8
    assert len(m["arms"]["Arm14"]["joint_names"]) == 6
    report = {
        "passed": True,
        "valid_depth_points": len(points),
        "known_table_points_within_2mm": count,
        "rgb_std": float(rgb.std()),
        "depth_range_m": [float(np.nanmin(depth)), float(np.nanmax(depth))],
        "note": "Synthetic geometric consistency test, not learned perception or physical calibration",
    }
    (path / "inspection.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("capture")
    a = p.parse_args()
    print(json.dumps(inspect(a.capture), indent=2))
