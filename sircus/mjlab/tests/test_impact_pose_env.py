"""CPU-only reconstruction, alignment and logger-restoration checks."""

import importlib.util
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np


NATIVE = all(importlib.util.find_spec(name) for name in ("torch", "mujoco", "warp"))


@unittest.skipUnless(NATIVE, "Pinned native dependencies required; no GPU is used")
class ImpactPoseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import mujoco

        bodies = []
        for role in (10, 14):
            tree = ""
            for k in range(1, 7):
                tree += (f'<body name="Arm{role}/link{k}" pos=".03 0 .04">'
                         f'<joint name="Arm{role}/joint{k}" axis="0 1 0"/>'
                         '<geom type="sphere" size=".01" mass=".1"/>')
            if role == 14:
                tree += '<body name="Arm14/impact_board" pos="0 0 .07"><geom type="box" size=".15 .1 .0025" mass=".1"/></body>'
            bodies.append(tree + "</body>"*6)
        cls.model = mujoco.MjModel.from_xml_string('<mujoco><worldbody>'+"".join(bodies)+'</worldbody></mujoco>')

    def arrays(self):
        from sircus.mjlab.impact_pose_env import reconstruct_pose_arrays

        times = np.arange(2049)/15360
        q = np.zeros((2049, 2, 12), dtype=np.float32)
        # A measured integrated excursion at first impact, independently
        # different on each arm, must not be confused with commanded hold.
        q[234, 0, 5] = .03
        q[234, 0, 11] = .08
        v = np.zeros_like(q)
        v[234, 0, 5] = .9
        v[234, 0, 11] = 1.2
        forces = np.zeros((2048, 2, 6))
        forces[233, 0, 0] = 5.
        ball = np.zeros((2049, 2, 3))
        ball[:, :, 2] = 1-times[:, None]
        velocity = np.zeros_like(ball)
        velocity[:, :, 2] = -1
        out = dict(time_s=times, forces_N=forces, position_m=ball, velocity_m_s=velocity)
        out.update(reconstruct_pose_arrays(self.model, q, v, times, forces, ball, velocity))
        return out

    def test_native_interval_both_endpoints_and_impact_window_preserved(self):
        from sircus.mjlab.impact_pose_env import sample_indices

        times = np.arange(2049)/15360
        forces = np.zeros(2048)
        forces[233] = 1
        selected, hit = sample_indices(times, forces)
        self.assertEqual(hit, 233)
        self.assertIn(233, selected)
        self.assertIn(234, selected)
        required = np.flatnonzero((times >= times[233]-.01) & (times <= times[234]+.01))
        self.assertTrue(set(required) <= set(selected))
        self.assertTrue(set(range(0, 2049, 64)) <= set(selected))
        self.assertIn(2048, selected)

    def test_actual_integrated_excursion_not_stale_held_transform(self):
        from sircus.mjlab.impact_pose_env import pose_summary, verify_pose_arrays

        arrays = self.arrays()
        result = verify_pose_arrays(self.model, arrays)
        self.assertEqual(result["status"], "passed")
        self.assertEqual(result["maximum_numeric_difference"], 0.)
        row = pose_summary(arrays, 0)
        self.assertEqual(row["first_impact_interval"]["index"], 233)
        self.assertEqual(row["impact_after"]["native_state_index"], 234)
        self.assertGreater(row["maximum_postintegration_board_drift_mm"], 1)
        self.assertAlmostEqual(row["maximum_postintegration_board_normal_error_deg"], np.degrees(.08), places=5)
        self.assertGreater(row["arm14_maximum_integrated_joint_excursion_rad"], .079)
        self.assertGreater(row["arm10_maximum_integrated_joint_excursion_rad"], .029)
        self.assertNotEqual(row["impact_after"]["board_normal"], row["held_pre_step_board_pose"]["board_normal"])
        self.assertEqual(row["settling_status"], "not_applicable_prescribed_endpoints_reset_each_native_interval")
        self.assertEqual(row["physical_release_status"], "not_tested_ball_spawned_free_flight")

    def test_absent_impact_remains_unknown_and_padding_explicit(self):
        from sircus.mjlab.impact_pose_env import pose_summary, verify_pose_arrays

        arrays = self.arrays()
        row = pose_summary(arrays, 1)
        self.assertIsNone(row["first_impact_interval"])
        self.assertIsNone(row["impact_before"])
        self.assertIsNone(row["impact_after"])
        self.assertTrue((arrays["pose_sample_state_index"][~arrays["pose_sample_valid"]] == -1).all())
        self.assertEqual(verify_pose_arrays(self.model, arrays)["status"], "passed")

    def test_independent_verifier_rejects_transform_and_clock_tampering(self):
        from sircus.mjlab.impact_pose_env import verify_pose_arrays

        arrays = self.arrays()
        arrays["pose_board_center_m"][0, 0, 0] += .001
        with self.assertRaisesRegex(ValueError, "FK evidence mismatch"):
            verify_pose_arrays(self.model, arrays)
        arrays = self.arrays()
        arrays["pose_sample_time_s"][1, 0] += 1/15360
        with self.assertRaisesRegex(ValueError, "identity changed"):
            verify_pose_arrays(self.model, arrays)

    def test_independent_verifier_rejects_missing_dense_impact_sample(self):
        from sircus.mjlab.impact_pose_env import verify_pose_arrays

        arrays = self.arrays()
        slot = np.flatnonzero(arrays["pose_sample_state_index"][:, 0] == 234)[0]
        arrays["pose_sample_valid"][slot, 0] = False
        arrays["pose_sample_state_index"][slot, 0] = -1
        with self.assertRaisesRegex(ValueError, "identity changed"):
            verify_pose_arrays(self.model, arrays)

    def test_scope_restores_original_logger_after_failure(self):
        from sircus.mjlab import calibrated_native_log
        from sircus.mjlab.impact_pose_env import pose_logger_scope

        original = calibrated_native_log.DenseNativeLog
        simulation = object()
        with self.assertRaisesRegex(RuntimeError, "deliberate failure"):
            with pose_logger_scope(simulation):
                self.assertIsNot(calibrated_native_log.DenseNativeLog, original)
                raise RuntimeError("deliberate failure")
        self.assertIs(calibrated_native_log.DenseNativeLog, original)

    def test_scope_rejects_other_simulation_instead_of_silent_instrumentation(self):
        from sircus.mjlab import calibrated_native_log
        from sircus.mjlab.impact_pose_env import pose_logger_scope

        original = calibrated_native_log.DenseNativeLog
        with pose_logger_scope(object()):
            with self.assertRaisesRegex(RuntimeError, "unexpected simulation"):
                calibrated_native_log.DenseNativeLog(object(), 1, 1)
        self.assertIs(calibrated_native_log.DenseNativeLog, original)

    def test_nonfinite_pose_samples_rejected(self):
        from sircus.mjlab.impact_pose_env import reconstruct_pose_arrays

        q = np.zeros((2, 1, 12))
        q[1, 0, 3] = np.nan
        with self.assertRaisesRegex(ValueError, "finite"):
            reconstruct_pose_arrays(self.model, q, np.zeros_like(q), np.arange(2)/15360,
                                    np.zeros((1, 1, 6)), np.zeros((2, 1, 3)), np.zeros((2, 1, 3)))

    def test_logger_keeps_copied_states_without_changing_simulation(self):
        import torch
        from sircus.mjlab.impact_pose_env import _PoseLog, model_ids

        class OriginalLog:
            def __init__(self, *args): self.calls = []
            def step(self, index, force): self.calls.append(index)
            def arrays(self, count):
                return dict(time_s=np.arange(3)/15360,
                    position_m=np.zeros((3, 1, 3)), velocity_m_s=np.zeros((3, 1, 3)),
                    forces_N=np.zeros((2, 1, 6)))

        simulation = SimpleNamespace(device="cpu", mj_model=self.model,
            data=SimpleNamespace(qpos=torch.zeros((1, self.model.nq)),
                                 qvel=torch.zeros((1, self.model.nv))))
        log = _PoseLog(OriginalLog, simulation, 1, 2)
        ids = model_ids(self.model)
        simulation.data.qpos[0, ids["q"][6]] = .05
        simulation.data.qvel[0, ids["v"][6]] = .2
        log.step(0, torch.zeros((1, 6)))
        simulation.data.qpos[0, ids["q"][6]] = .08
        log.step(1, torch.zeros((1, 6)))
        before_q, before_v = simulation.data.qpos.clone(), simulation.data.qvel.clone()
        arrays = log.arrays(1)
        self.assertTrue(torch.equal(before_q, simulation.data.qpos))
        self.assertTrue(torch.equal(before_v, simulation.data.qvel))
        self.assertEqual(log.base.calls, [0, 1])
        np.testing.assert_allclose(arrays["pose_joint_qpos_rad"][:, 0, 6], [0, .05, .08])
        np.testing.assert_allclose(arrays["pose_joint_qvel_rad_s"][:, 0, 6], [0, .2, .2])


if __name__ == "__main__":
    unittest.main()
