"""Native-constructor contract checks with mocked simulation, without a GPU."""

from contextlib import contextmanager
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from sircus.mjlab.tests.test_calibrated_score import config


def simulation_options(substeps):
    """Allocated native doubles and Warp float32 arrays, without allocation."""
    values = dict(timestep=1 / (240 * substeps), solver=2, iterations=100,
                  tolerance=1e-8, cone=0, integrator=3, ccd_iterations=50,
                  ccd_tolerance=1e-6, disableflags=0, enableflags=0,
                  gravity=np.asarray([0.0, 0.0, -9.81]))
    warp = dict(values)
    for key in ("timestep", "tolerance", "ccd_tolerance", "gravity"):
        warp[key] = np.asarray([1e-6 if key == "tolerance" else values[key]], dtype=np.float32)
    return SimpleNamespace(mj_model=SimpleNamespace(opt=SimpleNamespace(**values)),
                           wp_model=SimpleNamespace(opt=SimpleNamespace(**warp)))


@unittest.skipUnless(importlib.util.find_spec("torch") and importlib.util.find_spec("mujoco"),
                     "Pinned native CPU dependencies required; no GPU is used")
class NativeContractTests(unittest.TestCase):
    def test_reset_mask_and_rollout_share_native_stream(self):
        from sircus.mjlab.calibrated_env import native_reset_scope

        events = []
        @contextmanager
        def scope(stream):
            self.assertEqual(stream, "native-stream")
            events.append("enter-native")
            yield
            events.append("exit-native")
        sim = SimpleNamespace(wp_device="device", reset=lambda:events.append("reset"))
        with patch("warp.get_stream", return_value=SimpleNamespace(cuda_stream=123)), \
             patch("torch.cuda.ExternalStream", return_value="native-stream"), \
             patch("torch.cuda.stream", side_effect=scope):
            with native_reset_scope(sim):
                events.append("initialize-and-rollout")
        self.assertEqual(events, ["enter-native", "reset", "initialize-and-rollout", "exit-native"])

    def test_constructor_preserves_articulated_contact_model(self):
        from sircus.mjlab.calibrated_env import CalibratedEnvironment
        from sircus.mjlab.contact_env import ContactEnvironment
        from sircus.mjlab.corrected_contract import PHYSICS

        scene = json.loads((Path(__file__).resolve().parents[3]/"sircus/isaac/scene.json").read_text())
        cfg = config(scene)
        calls = []
        def base_init(instance, scene, assets, **kwargs):
            calls.append(kwargs)
            instance.sim = simulation_options(kwargs["substeps"])
        with patch.object(ContactEnvironment, "__init__", base_init), \
             patch("sircus.mjlab.capture_calibration.compiled_geometry", return_value={
                 "compiled_planes": cfg["compiled_planes"], "contact_geom_ids": {}}):
            CalibratedEnvironment(scene, Path("assets"), cfg)
        self.assertEqual(PHYSICS["board_mode"], "articulated")
        self.assertEqual(calls[0]["board_mode"], PHYSICS["board_mode"])
        self.assertEqual(calls[0]["gravity_mode"], PHYSICS["gravity_mode"])
        self.assertEqual(calls[0]["substeps"], PHYSICS["substeps"])

    def test_fine_constructor_passes_step_before_allocation(self):
        from sircus.mjlab.calibrated_env import CalibratedEnvironment
        from sircus.mjlab.contact_env import ContactEnvironment

        scene = json.loads((Path(__file__).resolve().parents[3]/"sircus/isaac/scene.json").read_text())
        cfg, calls = config(scene), []
        def base_init(instance, scene, assets, **kwargs):
            calls.append(kwargs)
            instance.sim = simulation_options(kwargs["substeps"])
        with patch.object(ContactEnvironment, "__init__", base_init), \
             patch("sircus.mjlab.capture_calibration.compiled_geometry", return_value={
                 "compiled_planes": cfg["compiled_planes"], "contact_geom_ids": {}}):
            env = CalibratedEnvironment(scene, Path("assets"), cfg, substeps=128)
        self.assertEqual(calls[0]["substeps"], 128)
        self.assertEqual(env.effective_options["native"]["timestep"], 1/30720)
        self.assertEqual(env.effective_options["warp"]["timestep"], float(np.float32(1/30720)))

    def test_unsupported_step_rejected_before_allocation(self):
        from sircus.mjlab.calibrated_env import CalibratedEnvironment
        from sircus.mjlab.contact_env import ContactEnvironment

        with patch.object(ContactEnvironment, "__init__") as allocate:
            for step in (32, 256, 64.0, True):
                with self.subTest(step=step), self.assertRaises(ValueError):
                    CalibratedEnvironment({}, Path("assets"), {}, substeps=step)
            allocate.assert_not_called()

    def test_option_verifier_reads_both_engines_without_mutating(self):
        from sircus.mjlab.calibrated_env import verified_native_options

        for step in (64, 128):
            sim = simulation_options(step)
            native_dt = sim.mj_model.opt.timestep
            warp_dt = sim.wp_model.opt.timestep.copy()
            actual = verified_native_options(sim, step)
            self.assertEqual(actual["native"]["tolerance"], 1e-8)
            self.assertEqual(actual["warp"]["tolerance"], float(np.float32(1e-6)))
            self.assertEqual(sim.mj_model.opt.timestep, native_dt)
            np.testing.assert_array_equal(sim.wp_model.opt.timestep, warp_dt)

    def test_option_verifier_fails_closed_for_wrong_step_or_physics(self):
        from sircus.mjlab.calibrated_env import verified_native_options

        for engine, key, wrong in (
            ("native", "timestep", 1/15360),
            ("warp", "timestep", np.asarray([1/15360], dtype=np.float32)),
            ("warp", "timestep", np.asarray([1/30720, 1/15360], dtype=np.float32)),
            ("native", "tolerance", 1e-6),
            ("warp", "tolerance", np.asarray([1e-8], dtype=np.float32)),
            ("warp", "gravity", np.asarray([[0, 0, 0]], dtype=np.float32)),
            ("warp", "solver", 1),
            ("warp", "ccd_iterations", 49),
            ("native", "disableflags", 1),
        ):
            sim = simulation_options(128)
            options = sim.mj_model.opt if engine == "native" else sim.wp_model.opt
            setattr(options, key, wrong)
            with self.subTest(engine=engine, key=key), self.assertRaises(ValueError):
                verified_native_options(sim, 128)

    def test_contact_environment_allocates_requested_native_step(self):
        import sys
        from sircus.mjlab.contact_env import ContactEnvironment

        class Allocated(Exception):
            pass
        calls = []
        def allocate(**kwargs):
            calls.append(kwargs)
            raise Allocated
        module = SimpleNamespace(Simulation=allocate,
                                 SimulationCfg=lambda **kw:SimpleNamespace(**kw),
                                 MujocoCfg=lambda **kw:SimpleNamespace(**kw))
        spec = SimpleNamespace(compile=lambda:object())
        with patch.dict(sys.modules, {"mjlab.sim": module}), \
             patch("torch.cuda.device_count", return_value=1), \
             patch("torch.cuda.get_device_name", return_value="NVIDIA GeForce RTX 4090"), \
             patch("sircus.mjlab.contact_model.make_model", return_value=(spec, object())), \
             patch("sircus.mjlab.run.sensor_config", return_value=[]):
            for step in (8, 64, 128):
                with self.assertRaises(Allocated):
                    ContactEnvironment({}, Path("assets"), substeps=step)
                self.assertEqual(calls[-1]["cfg"].mujoco.timestep, 1/(240*step))


if __name__ == "__main__":
    unittest.main()
