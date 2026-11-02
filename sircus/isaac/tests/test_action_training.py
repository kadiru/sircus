"""No-GPU checks for training-film provenance, parking and exact fixed timing."""

import copy
import hashlib
import importlib.util
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np

SOURCE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOURCE))
from action_training import (
    ActionTrainingShots,
    hold_disabled_targets,
    load_geometry_menu,
)
from cylinder_farm import menu_batches, specification


class Array:
    def __init__(self, data):
        self.value = np.asarray(data)

    def cpu(self):
        return self

    def numpy(self):
        return self.value


class Tensor(np.ndarray):
    def clone(self):
        return self.copy().view(Tensor)

    def abs(self):
        return np.abs(self).view(Tensor)

    def amax(self, dim):
        return np.asarray(np.max(np.asarray(self), axis=dim)).view(Tensor)

    def cpu(self):
        return self

    def numpy(self):
        return np.asarray(self)


def tensor(data):
    return np.asarray(data, dtype=float).view(Tensor)


class ActionTrainingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.source = Path(self.temp.name)
        cfg = json.loads((SOURCE / "scene.json").read_text())
        self.spec = specification(cfg, 10)
        candidates = list(menu_batches(self.spec["cells"]))
        # A synthetic endpoint qualification fixture, never a success fixture.
        checks = [
            [None if i == 1 else "endpoint_ik_fixture" for i in range(35)]
            for _ in candidates
        ]
        self.menu = dict(
            menus=[
                [batch[i] for batch in candidates] if i == 1 else [] for i in range(35)
            ],
            checks=checks,
            rule="Both endpoints pass IK and measured settling before release.",
        )
        (self.source / "contract.json").write_text(json.dumps(self.spec))
        self.report = dict(
            fingerprint=self.spec["fingerprint"], workcells=35, articulations=70
        )
        self.write_menu()

    def write_menu(self):
        data = json.dumps(self.menu).encode()
        (self.source / "menu.json").write_bytes(data)
        self.report["menu_sha256"] = hashlib.sha256(data).hexdigest()
        (self.source / "result.json").write_text(json.dumps(self.report))

    def test_menu_reuse_verifies_geometry_without_consulting_outcomes(self):
        menus, audit = load_geometry_menu(self.source, self.spec)
        self.assertEqual(menus, self.menu["menus"])
        self.assertEqual(audit["accepted_per_cell"][1], 60)
        self.report.update(
            captures=9999, rewards=[10000], successful_actions=["malicious"]
        )
        self.write_menu()
        again, _ = load_geometry_menu(self.source, self.spec)
        self.assertEqual(again, menus)

    def test_rejects_changed_source_contract_menu_checksum_or_proposals(self):
        changed = copy.deepcopy(self.spec)
        changed["fingerprint"] = "wrong"
        with self.assertRaises(ValueError):
            load_geometry_menu(self.source, changed)
        self.report["menu_sha256"] = "wrong"
        (self.source / "result.json").write_text(json.dumps(self.report))
        with self.assertRaises(ValueError):
            load_geometry_menu(self.source, self.spec)
        self.menu["menus"][1][0]["height_m"] += 0.001
        self.write_menu()
        with self.assertRaises(ValueError):
            load_geometry_menu(self.source, self.spec)

    def test_rejects_wrong_source_cell_count_and_mutated_contract_content(self):
        self.report["articulations"] = 68
        self.write_menu()
        with self.assertRaises(ValueError):
            load_geometry_menu(self.source, self.spec)
        self.report["articulations"] = 70
        self.write_menu()
        altered = dict(self.spec, radius_m=0.5)
        (self.source / "contract.json").write_text(json.dumps(altered))
        with self.assertRaises(ValueError):
            load_geometry_menu(self.source, self.spec)

    def test_menu_empty_cells_are_held_before_drive_commands(self):
        starts = {
            name: np.arange(18).reshape(3, 6).copy() for name in ("Arm10", "Arm14")
        }
        targets = {name: value + 100 for name, value in starts.items()}
        expected = copy.deepcopy(targets)
        hold_disabled_targets(targets, starts, np.array([False, True, False]))
        for name in targets:
            np.testing.assert_array_equal(targets[name][[0, 2]], starts[name][[0, 2]])
            np.testing.assert_array_equal(targets[name][1], expected[name][1])

    def test_fixed_view_never_uses_ball_configuration_or_frame(self):
        env = ActionTrainingShots()
        env.saved_camera = dict(
            eye_m=[10.9706, -6.1943, 2.5406], target_m=[7.8307, -3.4877, 0.3904]
        )
        first = env.view_pose()
        for frame in (0, 100, 999):
            for ball in ([100, -200, 300], [-4, 8, -9]):
                actual = env.view_pose(
                    None, frame, [{"board_xy_m": ball[:2]}], None, ball
                )
                np.testing.assert_array_equal(actual, first)
        first[0][:] = 0
        self.assertEqual(env.saved_camera["eye_m"][0], 10.9706)

    def test_disabled_cell_never_retargeted_during_active_settling_correction(self):
        # Execute the actual production prepare method, including its correction
        # loop, using CPU arrays and arm stubs instead of the Isaac runtime.
        module_spec = importlib.util.spec_from_file_location(
            "training_prepare_test", SOURCE / "parallel_runtime.py"
        )
        module = importlib.util.module_from_spec(module_spec)
        with patch.dict(
            sys.modules,
            {
                "torch": types.SimpleNamespace(),
                "PIL": types.SimpleNamespace(Image=None),
            },
        ):
            module_spec.loader.exec_module(module)

        class Arm:
            joint_names = [f"joint{i}" for i in range(1, 7)]

            def __init__(self, flange):
                self.writes = []
                self.data = types.SimpleNamespace(
                    default_joint_pos=tensor(np.zeros((2, 6))),
                    joint_pos=tensor([[7] * 6, [0] * 6]),
                    joint_vel=tensor(np.zeros((2, 6))),
                    body_pos_w=tensor(np.tile(flange, (2, 1, 1))),
                )

            def set_joint_position_target(self, q):
                self.writes.append(np.array(q).copy())

        class Environment(module.ParallelShots):
            def approach_endpoints(self, q):
                starts = {n: a.data.joint_pos.clone() for n, a in self.arms.items()}
                hold_disabled_targets(q, starts, self.preparation_enabled)
                for n, arm in self.arms.items():
                    arm.set_joint_position_target(q[n])

            def _ik(self, target, name, seed=None):
                self.ik_calls.append((name, seed is not None))
                return dict(solved=True, q=np.ones(6))

            def tick(self):
                pass

            def set_balls(self, xyz):
                pass

            def assert_buckets_fixed(self):
                pass

            def board_poses(self):
                self.pose_reads += 1
                # Active cell deliberately fails the first settling test, then
                # passes after one correction. Disabled cell remains far away.
                offset = 0.03 if self.pose_reads <= 2 else 0
                return np.array([[9, 9, 9], [0.5, 0, 0.3 + offset]]), np.array(
                    [[0, 0, 1]] * 2
                )

        env = Environment.__new__(Environment)
        env.cfg = json.loads((SOURCE / "scene.json").read_text())
        env.n, env.origins, env.flange_idx = 2, np.zeros((2, 3)), 0
        env.preparation_enabled = np.array([False, True])
        env.spec = dict(cells=[dict(board_xy_m=[0.5, 0], heading_deg=0)] * 2)
        flange = [0.5, 0, 0.3 + env.cfg["ball"]["radius_m"] + 0.2 + 0.095]
        env.arms = {name: Arm(flange) for name in ("Arm10", "Arm14")}
        env.ik_calls, env.pose_reads = [], 0
        actions = [dict(height_m=0.3, gap_m=0.2, tilt_deg=0)] * 2
        _, _, _, invalid = env.prepare(actions)
        self.assertEqual(invalid, ["geometry_menu_empty", None])
        self.assertEqual(
            env.ik_calls, [("Arm14", False), ("Arm10", False), ("Arm14", True)]
        )
        self.assertGreaterEqual(len(env.arms["Arm14"].writes), 2)
        for arm in env.arms.values():
            for write in arm.writes:
                np.testing.assert_array_equal(write[0], [7] * 6)

    def test_telemetry_is_full_rate_while_rgb_has_exact_stride(self):
        env = ActionTrainingShots()
        env.training_active, env.phase, env.steps, env.release_step = (
            True,
            "release",
            0,
            0,
        )
        env.focus_telemetry = []
        env.ball = types.SimpleNamespace(
            data=types.SimpleNamespace(
                root_pos_w=Array(np.ones((35, 3))),
                root_lin_vel_w=Array(np.zeros((35, 3))),
            )
        )
        env.forces = lambda: np.ones((35, 6))
        env.sample = Mock()
        fake = types.SimpleNamespace(
            ParallelShots=types.SimpleNamespace(
                tick=lambda e: setattr(e, "steps", e.steps + 1)
            )
        )
        with patch.dict(sys.modules, {"parallel_runtime": fake}):
            for _ in range(17):
                env.tick()
        self.assertEqual(
            [v["physics_step"] for v in env.focus_telemetry], list(range(1, 18))
        )
        self.assertEqual(env.sample.call_count, 2)
        self.assertEqual(env.focus_telemetry[-1]["activity"]["board"], list(range(35)))
        self.assertEqual(env.focus_telemetry[-1]["forces_N"]["bucket"], 1)


if __name__ == "__main__":
    unittest.main()
