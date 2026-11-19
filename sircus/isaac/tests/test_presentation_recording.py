"""No-GPU tests for geometry-only native filming identity and evidence."""

import copy
import json
import shutil
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np

SOURCE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOURCE))
from action_training import ActionTrainingShots
from presentation_recording import (
    PresentationShots,
    canonical_hash,
    geometry_order,
    load_source,
    make_plan,
    save_physics,
    seed_verified_ik,
    validate_options,
)

BASELINE = SOURCE.parents[1] / "baselines/isaac-expanded-farm-20261102"


class Tensor(np.ndarray):
    def cpu(self):
        return self

    def numpy(self):
        return np.asarray(self)


def tensor(value):
    return np.asarray(value).view(Tensor)


class PresentationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bundle = load_source(BASELINE, BASELINE / "selected-camera.json")

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.dest = Path(self.tmp.name)

    def copied_source(self):
        for name in (
            "contract.json", "result.json", "selected-camera.json",
            "camera-binding.json", "expanded-menu-qualified.json",
            "probes.json", "rim-rays.json",
        ):
            shutil.copyfile(BASELINE / name, self.dest / name)
        return self.dest

    def test_verifies_real_frozen_source_without_changing_training_contract(self):
        bundle = self.bundle
        self.assertEqual(bundle["spec"]["training_batches"], 10)
        self.assertEqual([len(m) for m in bundle["menus"]], [24] * 35)
        self.assertEqual(len(bundle["source"]["original_source_sha256"]), 92)
        self.assertEqual(bundle["source"]["camera_sha256"],
                         bundle["report"]["selected_camera_sha256"])

    def test_rejects_camera_edits_even_when_json_is_semantically_identical(self):
        source = self.copied_source()
        camera = source / "selected-camera.json"
        camera.write_text(json.dumps(json.loads(camera.read_text()), indent=4))
        with self.assertRaises(ValueError):
            load_source(source, camera)

    def test_rejects_source_result_menu_probe_or_contract_tampering(self):
        changes = (
            ("result.json", lambda x: x.update(menu_sha256="bad")),
            ("result.json", lambda x: x.update(updates=[])),
            ("result.json", lambda x: x["source_sha256"].update(rl_contract__fixture="bad")),
            ("contract.json", lambda x: x.update(training_batches=30)),
            ("rim-rays.json", lambda x: x["passed"].__setitem__(0, False)),
            ("expanded-menu-qualified.json", lambda x: x["menus"][0][0].update(gap_m=0.9)),
        )
        for name, change in changes:
            with self.subTest(name=name):
                source = self.copied_source()
                data = json.loads((source / name).read_text())
                change(data)
                (source / name).write_text(json.dumps(data))
                with self.assertRaises((ValueError, FileNotFoundError)):
                    load_source(source, source / "selected-camera.json")

    def test_rejects_changed_physics_dependency(self):
        source = self.copied_source()
        report = json.loads((source / "result.json").read_text())
        report["source_sha256"]["parallel_runtime.py"] = "incorrect"
        (source / "result.json").write_text(json.dumps(report))
        with self.assertRaisesRegex(ValueError, "Historical native dependency changed"):
            load_source(source, source / "selected-camera.json")

    def test_plan_has_all35_distinct_buckets_and_full24_geometry_coverage(self):
        plan = make_plan(self.bundle, 30)
        self.assertEqual(plan["original_training_batches"], 10)
        self.assertEqual(plan["optimizer_updates"], 0)
        self.assertEqual(plan["presentation_batches"], 30)
        self.assertEqual(len({tuple(c["bucket_xy_m"]) for c in plan["cells"]}), 35)
        self.assertEqual(plan, make_plan(self.bundle, 30))
        for i in range(35):
            self.assertEqual(len({c["menu_indices"][i] for c in plan["cycles"][:24]}), 24)
            for c in plan["cycles"]:
                self.assertEqual(c["actions"][i], self.bundle["menus"][i][c["menu_indices"][i]])
        self.assertEqual(canonical_hash(plan), canonical_hash(copy.deepcopy(plan)))
        plan["cycles"][0]["actions"][0]["gap_m"] += 0.01
        self.assertNotEqual(plan, make_plan(self.bundle, 30))

    def test_plan_selection_cannot_use_saved_capture_rewards_or_ids(self):
        before = make_plan(self.bundle)
        changed = copy.deepcopy(self.bundle)
        changed["report"].update(captures=99999, rewards=[9999], successful_actions=["bad"])
        after = make_plan(changed)
        self.assertEqual(before, after)
        options = copy.deepcopy(self.bundle["menus"][1])
        order = geometry_order(options)
        for index, option in enumerate(options):
            option.update(id=f"renamed-{index}", capture=index == 0, reward=99999 - index)
        self.assertEqual(order, geometry_order(options))
        # Numeric geometry defines tie breaks, independent of menu list order.
        reversed_options = options[::-1]
        self.assertEqual(
            [options[i]["height_m"] for i in order],
            [reversed_options[i]["height_m"] for i in geometry_order(reversed_options)],
        )

    def test_bounded_presentation_options_preserve_old_mode(self):
        self.assertFalse(validate_options(None, None, 30, appearance="dark"))
        self.assertTrue(validate_options("source", "camera", 30, appearance="silver"))
        for count in (0, 61, 30.0, True):
            with self.assertRaises(ValueError):
                make_plan(self.bundle, count)
        for source, camera, count, appearance, incompatible in (
            (None, "camera", 30, "silver", False),
            (None, None, 2, "dark", False),
            ("source", None, 30, "silver", False),
            ("source", "camera", 30, "dark", False),
            ("source", "camera", 30, "silver", True),
        ):
            with self.assertRaises(ValueError):
                validate_options(source, camera, count, appearance=appearance, incompatible=incompatible)

    def test_cache_population_reverifies_all1680_endpoint_fits(self):
        fake = types.ModuleType("expanded_pose_native")
        fake.verified_fit = Mock(side_effect=lambda target, record, links, normal: record)
        env = types.SimpleNamespace(n=35, links="fixture", ik_cache={})
        with patch.dict(sys.modules, expanded_pose_native=fake):
            count = seed_verified_ik(env, self.bundle)
        self.assertEqual(count, 35 * 24 * 2)
        self.assertEqual(fake.verified_fit.call_count, count)
        self.assertTrue(any(k[0] == "Arm10" for k in env.ik_cache))
        self.assertTrue(any(k[0] == "Arm14" for k in env.ik_cache))
        self.assertEqual(sum(c.args[3] for c in fake.verified_fit.call_args_list), 35 * 24)

    def test_camera_is_exact_and_does_not_follow_balls(self):
        env = PresentationShots()
        env.saved_camera = self.bundle["camera"]
        expected = env.view_pose()
        for ball in ([1000, -999, -200], [0, 0, 8]):
            actual = env.view_pose(frame=900, ball_world=ball)
            np.testing.assert_array_equal(actual, expected)

    def test_full_rate_all35_telemetry_stays_native_and_saves_without_pickle(self):
        env = PresentationShots()
        env.training_active, env.phase, env.steps = True, "release", 0
        env.physics_samples = []
        env.board_idx = 2
        env.ball = types.SimpleNamespace(data=types.SimpleNamespace(
            root_pos_w=tensor(np.zeros((35, 3))), root_lin_vel_w=tensor(np.ones((35, 3))),
        ))
        env.arms = {name: types.SimpleNamespace(data=types.SimpleNamespace(
            joint_pos=tensor(np.zeros((35, 6))), joint_vel=tensor(np.ones((35, 6))),
            body_pos_w=tensor(np.zeros((35, 3, 3))), body_quat_w=tensor(np.zeros((35, 3, 4))),
        )) for name in ("Arm10", "Arm14")}
        env.forces = lambda: np.zeros((35, 6))
        def native_step(_):
            env.steps += 1
            env.ball.data.root_pos_w[:, 2] = env.steps / 240
        with patch.object(ActionTrainingShots, "tick", native_step):
            for _ in range(17):
                env.tick()
        save_physics(self.dest / "physics.npz", env.physics_samples)
        with np.load(self.dest / "physics.npz", allow_pickle=False) as data:
            np.testing.assert_array_equal(data["time_step"], np.arange(1, 18))
            self.assertEqual(data["position_m"].shape, (17, 35, 3))
            self.assertEqual(data["joint_position_rad"].shape, (17, 35, 2, 6))
            self.assertEqual(data["board_quaternion_wxyz"].shape, (17, 35, 4))
            np.testing.assert_allclose(data["position_m"][:, 0, 2], np.arange(1, 18) / 240)
        env.physics_samples[2]["time_step"] += 1
        with self.assertRaisesRegex(RuntimeError, "Missing full-rate"):
            save_physics(self.dest / "bad.npz", env.physics_samples)

    def test_presentation_entrypoint_leaves_historical_contracts_intact(self):
        code = (SOURCE / "film_cylinder_farm.py").read_text()
        self.assertIn("run_presentation(env, presentation_bundle, args.presentation_batches)", code)
        self.assertIn('if presentation:\n        shutil.copyfile', code)
        self.assertIn('if training_camera:\n        from action_training import run_training', code)
        module_code = (SOURCE / "presentation_recording.py").read_text()
        self.assertNotIn("torch.optim", module_code)
        self.assertNotIn("from farm_policy", module_code)


if __name__ == "__main__":
    unittest.main()
