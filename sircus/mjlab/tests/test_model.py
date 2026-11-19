"""Geometry and articulation contracts that can run without installing MuJoCo."""

import copy
import json
from pathlib import Path
import tempfile
import unittest
import xml.etree.ElementTree as ET

import numpy as np
from sircus.isaac.geometry import bucket_wedge
from sircus.mjlab.model import build_scene

ROOT = Path(__file__).resolve().parents[2]


def fixture(path, board):
    root = ET.Element("robot", name="arm")
    for i, name in enumerate(["link_base"] + [f"link{i}" for i in range(1, 7)]):
        link = ET.SubElement(root, "link", name=name)
        inertial = ET.SubElement(link, "inertial")
        ET.SubElement(inertial, "origin", xyz="0 0 0", rpy="0 0 1.5707963267948966")
        ET.SubElement(inertial, "mass", value="1")
        ET.SubElement(
            inertial, "inertia", ixx="1", iyy="2", izz="3", ixy="0", ixz="0", iyz="0"
        )
        geo = ET.SubElement(ET.SubElement(link, "collision"), "geometry")
        ET.SubElement(geo, "box", size=".04 .05 .1")
        if i:
            j = ET.SubElement(root, "joint", name=f"joint{i}", type="revolute")
            ET.SubElement(j, "parent", link="link_base" if i == 1 else f"link{i-1}")
            ET.SubElement(j, "child", link=name)
            ET.SubElement(j, "origin", xyz="0 0 .1", rpy="0 0 0")
            ET.SubElement(j, "axis", xyz="0 0 1")
            ET.SubElement(j, "limit", lower="-3", upper="3", effort="100", velocity="1")
    if board:
        link = ET.SubElement(root, "link", name="impact_board")
        geo = ET.SubElement(ET.SubElement(link, "collision"), "geometry")
        ET.SubElement(geo, "box", size=".2 .3 .005")
        j = ET.SubElement(root, "joint", name="board_mount", type="fixed")
        ET.SubElement(j, "parent", link="link6")
        ET.SubElement(j, "child", link="impact_board")
        ET.SubElement(j, "origin", xyz="0 0 .1", rpy="0 0 1.5707963267948966")
    ET.ElementTree(root).write(path)


class ModelTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.assets = Path(self.temp.name)
        fixture(self.assets / "dropper.urdf", False)
        fixture(self.assets / "board_arm.urdf", True)
        self.cfg = json.loads((ROOT / "isaac" / "scene.json").read_text())

    def tearDown(self):
        self.temp.cleanup()

    def test_open_taper_uses_separate_shared_wall_vertices_and_floor(self):
        root = ET.fromstring(build_scene(self.cfg, self.assets))
        body = root.find(".//body[@name='bucket']")
        # Per-world static body transforms can leave cached GPU colliders behind.
        self.assertEqual(body.get("mocap"), "true")
        walls = [
            g for g in body.findall("geom") if g.get("name").startswith("bucket/wall")
        ]
        self.assertEqual(len(walls), 64)
        self.assertEqual(len(body.findall("geom")), 65)
        for i, w in enumerate(walls):
            mesh = root.find(f".//mesh[@name='{w.get('mesh')}']")
            vertices = np.array([float(v) for v in mesh.get("vertex").split()]).reshape(
                -1, 3
            )
            np.testing.assert_allclose(vertices, bucket_wedge(self.cfg["bucket"], i))
        base = body.find("geom[@name='bucket/base']")
        np.testing.assert_allclose(
            np.fromstring(base.get("size"), sep=" "), [0.079, 0.004]
        )
        self.assertEqual(
            self.cfg["bucket"]["height_interpretation"],
            "unresolved: overall height or interior depth",
        )

    def test_two_articulations_preserve_joints_and_rigid_board_mount(self):
        root = ET.fromstring(build_scene(self.cfg, self.assets))
        for arm in ("Arm10", "Arm14"):
            joints = [
                j.get("name")
                for j in root.findall(".//joint")
                if j.get("name", "").startswith(arm + "/")
            ]
            self.assertEqual(joints, [f"{arm}/joint{i}" for i in range(1, 7)])
        mount = root.find(".//body[@name='Arm14/impact_board']")
        np.testing.assert_allclose(
            np.fromstring(mount.get("pos"), sep=" "), [0, 0, 0.1]
        )
        np.testing.assert_allclose(
            np.fromstring(mount.get("quat"), sep=" "), [2**-0.5, 0, 0, 2**-0.5]
        )
        self.assertIsNone(mount.find("joint"))
        excludes = root.findall("contact/exclude")
        self.assertTrue(
            all(
                x.get("body1").split("/")[0] == x.get("body2").split("/")[0]
                for x in excludes
            )
        )
        board = root.find(".//geom[@name='Arm14/impact_board/collision/0']")
        self.assertNotEqual(board.get("contype"), "0")

    def test_inertial_rotation_and_camera_coordinates_are_preserved(self):
        before = copy.deepcopy(self.cfg)
        root = ET.fromstring(
            build_scene(
                self.cfg,
                self.assets,
                bucket_xy=[0.3, -0.4],
                ball_position=[0.7, -0.1, 0.8],
            )
        )
        inertial = root.find(".//body[@name='Arm10/link1']/inertial")
        np.testing.assert_allclose(
            np.fromstring(inertial.get("fullinertia"), sep=" "),
            [2, 1, 3, 0, 0, 0],
            atol=1e-12,
        )
        self.assertIsNone(inertial.get("quat"))
        cam = root.find(".//camera[@name='front']")
        axes = np.fromstring(cam.get("xyaxes"), sep=" ").reshape(2, 3)
        look = np.cross(*axes) * -1
        target = (
            np.array(self.cfg["camera"]["look_at_m"]) - self.cfg["camera"]["position_m"]
        )
        np.testing.assert_allclose(look, target / np.linalg.norm(target), atol=1e-12)
        self.assertEqual(self.cfg, before)
        np.testing.assert_allclose(
            np.fromstring(root.find(".//body[@name='bucket']").get("pos"), sep=" "),
            [0.3, -0.4, 0],
        )


if __name__ == "__main__":
    unittest.main()
