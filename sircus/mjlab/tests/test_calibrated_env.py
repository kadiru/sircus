"""Native-constructor contract checks with mocked simulation, without a GPU."""

from contextlib import contextmanager
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from sircus.mjlab.tests.test_calibrated_score import config


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
            instance.sim = SimpleNamespace(mj_model=object())
        with patch.object(ContactEnvironment, "__init__", base_init), \
             patch("sircus.mjlab.capture_calibration.compiled_geometry", return_value={
                 "compiled_planes": cfg["compiled_planes"], "contact_geom_ids": {}}):
            CalibratedEnvironment(scene, Path("assets"), cfg)
        self.assertEqual(PHYSICS["board_mode"], "articulated")
        self.assertEqual(calls[0]["board_mode"], PHYSICS["board_mode"])
        self.assertEqual(calls[0]["gravity_mode"], PHYSICS["gravity_mode"])
        self.assertEqual(calls[0]["substeps"], PHYSICS["substeps"])


if __name__ == "__main__":
    unittest.main()
