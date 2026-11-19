"""Endpoint geometry tests; Torch/native tests run when their dependencies exist."""

import json
import math
import os
from pathlib import Path
import tempfile
import unittest
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation

from sircus.mjlab.learning_ik import BatchIK, ShotIK, read_chain, shot_targets

try:
    import torch
except ImportError:
    torch = None

ROOT = Path(__file__).resolve().parents[2]


def fixture(path):
    root = ET.Element("robot", name="six-axis-test")
    for i, axis in enumerate(["0 0 1", "0 1 0", "0 1 0", "1 0 0", "0 1 0", "1 0 0"], 1):
        joint = ET.SubElement(root, "joint", name=f"joint{i}", type="revolute")
        ET.SubElement(joint, "origin", xyz=".1 0 .12", rpy=".03 -.02 .01")
        ET.SubElement(joint, "axis", xyz=axis)
        ET.SubElement(joint, "limit", lower="-3", upper="3")
    ET.ElementTree(root).write(path)


class TargetTests(unittest.TestCase):
    def setUp(self):
        self.cfg = json.loads((ROOT / "isaac" / "scene.json").read_text())

    def test_board_normal_faces_bucket_and_rotated_mount_is_preserved(self):
        boards = np.array([[0.6, -0.1, 0.3], [0.7, -0.2, 0.4], [0.5, 0.0, 0.2]])
        buckets = np.array([[0.6, -0.5], [1, -0.2], [0.2, 0.3]])
        tilts = np.array([25, 40, 12])
        out = shot_targets(self.cfg, buckets, boards, tilts, [0.2, 0.3, 0.1])
        normals = out["board_normal_world"]
        direction = buckets - boards[:, :2]
        direction /= np.linalg.norm(direction, axis=1)[:, None]
        np.testing.assert_allclose(
            normals[:, :2], direction * np.sin(np.deg2rad(tilts))[:, None], atol=1e-14
        )
        np.testing.assert_allclose(normals[:, 2], np.cos(np.deg2rad(tilts)), atol=1e-14)
        base = np.eye(4)
        base[:3, :3] = Rotation.from_euler("z", math.pi).as_matrix()
        base[0, 3] = self.cfg["arms"]["base_spacing_m"]
        mount = np.eye(4)
        mount[:3, :3] = Rotation.from_euler(
            "z", self.cfg["board"]["mount_yaw_rad"]
        ).as_matrix()
        mount[2, 3] = self.cfg["board"]["tool_offset_m"]
        np.testing.assert_allclose(
            base @ out["board_flange_local"] @ mount, out["board_world"], atol=1e-14
        )

    def test_vertical_release_height_and_downward_tool_orientation(self):
        board = np.array([[0.6, -0.1, 0.3]])
        out = shot_targets(self.cfg, [[0.6, -0.5]], board, 25, 0.2)
        np.testing.assert_allclose(out["release_center_m"], [[0.6, -0.1, 0.5375]])
        np.testing.assert_allclose(
            out["dropper_flange_world"][0, :3, 3], [0.6, -0.1, 0.6325]
        )
        np.testing.assert_allclose(out["dropper_flange_world"][0, :3, 2], [0, 0, -1])

    def test_invalid_geometry_is_rejected_before_ik(self):
        for bucket, board, tilt, gap in [
            ([[0.6, -0.1]], [[0.6, -0.1, 0.3]], 25, 0.2),
            ([[0.6, -0.5]], [[0.6, -0.1, 0.3]], 90, 0.2),
            ([[0.6, -0.5]], [[0.6, -0.1, 0.3]], 25, 0),
            ([[np.nan, -0.5]], [[0.6, -0.1, 0.3]], 25, 0.2),
        ]:
            with self.subTest(bucket=bucket, tilt=tilt, gap=gap):
                with self.assertRaises(ValueError):
                    shot_targets(self.cfg, bucket, board, tilt, gap)

    def test_chain_has_six_origins_axes_and_explicit_limits(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "arm.urdf"
            fixture(path)
            origins, axes, lo, hi = read_chain(path)
            self.assertEqual(origins.shape, (6, 4, 4))
            self.assertEqual(axes.shape, (6, 3))
            np.testing.assert_allclose(lo, -3)
            np.testing.assert_allclose(hi, 3)


@unittest.skipIf(
    torch is None, "Torch unavailable; run in pinned Ubuntu mjlab environment"
)
class TorchIKTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        path = Path(self.temp.name) / "arm.urdf"
        fixture(path)
        self.solver = BatchIK.from_urdf(path)
        self.rng = np.random.default_rng(7221)

    def tearDown(self):
        self.temp.cleanup()

    def test_fk_matches_independent_scipy_chain(self):
        q = self.rng.uniform(-2, 2, (16, 6))
        actual = self.solver.fk(q).numpy()
        for i, values in enumerate(q):
            expected = np.eye(4)
            for j in range(6):
                rotate = np.eye(4)
                rotate[:3, :3] = Rotation.from_rotvec(
                    self.solver.axes[j].numpy() * values[j]
                ).as_matrix()
                expected = expected @ self.solver.origins[j].numpy() @ rotate
            np.testing.assert_allclose(actual[i], expected, atol=2e-14)

    def test_nearby_targets_solve_with_strict_residual_and_limits(self):
        q = self.rng.uniform(-1, 1, (12, 6))
        targets = self.solver.fk(q)
        result = self.solver.solve(targets, q + 0.05, starts=3, iterations=100)
        self.assertTrue(bool(result["valid"].all()), result["residual"])
        self.assertLess(float(result["position_error_m"].max()), 1e-4)
        self.assertTrue(bool((result["q"] >= self.solver.lower).all()))
        self.assertTrue(bool((result["q"] <= self.solver.upper).all()))

    def test_rotation_error_retains_near_pi_error(self):
        matrices = Rotation.from_rotvec(
            np.array([[math.pi, 0, 0], [0, math.pi - 1e-7, 0], [0.1, -0.2, 0.3]])
        ).as_matrix()
        actual = self.solver._rotvec(self.solver.tensor(matrices)).numpy()
        np.testing.assert_allclose(
            np.linalg.norm(actual, axis=1),
            [math.pi, math.pi - 1e-7, np.sqrt(0.14)],
            atol=1e-10,
        )

    def test_unreachable_finite_budget_is_not_accepted(self):
        target = np.eye(4)[None]
        target[0, 0, 3] = 100
        result = self.solver.solve(target, starts=2, iterations=30)
        self.assertFalse(bool(result["valid"].any()))
        self.assertGreater(float(result["position_error_m"][0]), 90)

    def test_normal_only_spin_policy_is_explicit(self):
        q = np.zeros((1, 6))
        result = self.solver.solve(self.solver.fk(q), q, starts=1, normal_only=True)
        self.assertTrue(bool(result["valid"][0]))
        self.assertEqual(result["spin_policy"], "unconstrained_about_normal")

    def test_nonrigid_or_nonfinite_target_is_rejected(self):
        for target in [np.zeros((1, 4, 4)), np.full((1, 4, 4), np.nan)]:
            with self.assertRaises(ValueError):
                self.solver.solve(target)

    def test_position_and_normal_jacobian_match_finite_differences(self):
        q = self.solver.tensor(self.rng.uniform(-1, 1, (3, 6)))
        target = self.solver.fk(q)
        _, jacobian, _ = self.solver._error_jacobian(q, target, True)
        for j in range(6):
            step = torch.zeros_like(q)
            step[:, j] = 1e-6
            plus, minus = self.solver.fk(q + step), self.solver.fk(q - step)
            derivative = torch.cat(
                (
                    (plus[:, :3, 3] - minus[:, :3, 3]) / 2e-6,
                    0.25 * (plus[:, :3, 2] - minus[:, :3, 2]) / 2e-6,
                ),
                -1,
            )
            np.testing.assert_allclose(
                jacobian[:, :, j].numpy(), derivative.numpy(), atol=1e-9
            )


@unittest.skipUnless(
    torch is not None and os.environ.get("SIRCUS_IK_ASSETS"),
    "Optional generated native assets required",
)
class NativeIKTests(unittest.TestCase):
    def test_generated_urdf_endpoints_agree_with_mujoco(self):
        import mujoco
        from sircus.mjlab.model import build_scene

        assets = Path(os.environ["SIRCUS_IK_ASSETS"])
        cfg = json.loads((ROOT / "isaac" / "scene.json").read_text())
        ik = ShotIK(cfg, assets)
        model = mujoco.MjModel.from_xml_string(build_scene(cfg, assets))
        data = mujoco.MjData(model)
        rng = np.random.default_rng(91823)
        for role, solver in [("Arm10", ik.dropper), ("Arm14", ik.board)]:
            q = rng.uniform(solver.lower.numpy(), solver.upper.numpy(), (12, 6))
            predicted = solver.fk(q).numpy()
            for values, expected in zip(q, predicted):
                for j, value in enumerate(values, 1):
                    data.qpos[model.joint(f"{role}/joint{j}").qposadr[0]] = value
                mujoco.mj_forward(model, data)
                observed = data.body(f"{role}/link6")
                base = np.eye(4)
                if role == "Arm14":
                    base[:3, :3] = Rotation.from_euler("z", math.pi).as_matrix()
                    base[0, 3] = cfg["arms"]["base_spacing_m"]
                world = base @ expected
                np.testing.assert_allclose(observed.xpos, world[:3, 3], atol=1e-10)
                np.testing.assert_allclose(
                    observed.xmat.reshape(3, 3), world[:3, :3], atol=1e-10
                )


if __name__ == "__main__":
    unittest.main()
