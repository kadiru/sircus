import copy
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from sircus.mjlab import expanded_selector_video as V


class DeviceSelectionTests(unittest.TestCase):
    def setUp(self):
        self.inventory = [dict(uuid=V.GPU_UUID, name="NVIDIA GeForce RTX 4090", pci="00000000:01:00.0"),
                          dict(uuid="other", name="NVIDIA GeForce RTX 4070", pci="00000000:04:00.0")]
        self.devices = [dict(index=0, extensions="EGL_NV_device_cuda", pci="0000:04:00.0"),
                        dict(index=1, extensions="EGL_MESA_device", pci="0000:01:00.0"),
                        dict(index=2, extensions="EGL_NV_device_cuda EGL_EXT_device_drm", pci="0000:01:00.0")]

    def test_maps_pci_not_cuda_or_physical_index(self):
        self.assertEqual(V.select_device(self.devices, self.inventory)["index"], 2)

    def test_refuses_unknown_or_ambiguous_mapping(self):
        with self.assertRaisesRegex(ValueError, "Exactly one"):
            V.select_device(self.devices[:2], self.inventory)
        with self.assertRaisesRegex(ValueError, "Exactly one"):
            V.select_device(self.devices + [dict(self.devices[2], index=3)], self.inventory)

    def test_backend_rejects_4070_and_software(self):
        device = V.select_device(self.devices, self.inventory)
        for renderer, vendor in (("NVIDIA GeForce RTX 4070", "NVIDIA"), ("llvmpipe", "Mesa")):
            with self.assertRaises(ValueError):
                V.validate_backend(renderer, vendor, device)
        self.assertEqual(V.validate_backend("NVIDIA GeForce RTX 4090", "NVIDIA Corporation", device)["device"]["uuid"], V.GPU_UUID)


class FrameTests(unittest.TestCase):
    def bundles(self):
        times = np.arange(61441) / 15360
        return {m:dict(time_s=times, scored_until_s=times[stop], events={},
                       row=dict(bucket_xy_m=[.9, -.2], board_center_m=[.5, -.3, .4],
                                native_initial_ball_center_m=[.5, -.3, .8]))
                for m,stop in (("current", 24576), ("expanded", 61440))}

    def test_each_frame_is_actual_recorded_tick_and_explicit_terminal_hold(self):
        bundles = self.bundles()
        frames = V.frame_plan(bundles)
        for f in frames:
            for m,b in bundles.items():
                s = f["methods"][m]
                self.assertEqual(s["source_time_s"], b["time_s"][s["source_tick"]])
                self.assertLessEqual(s["source_time_s"], b["scored_until_s"])
                self.assertEqual(s["terminal_hold"], f["common_time_s"] >= b["scored_until_s"])
        self.assertEqual(frames[-1]["methods"]["current"]["source_tick"], 24576)
        self.assertEqual(frames[-1]["methods"]["expanded"]["source_tick"], 61440)
        slow = [f for f in frames if f["segment"] == "slow"]
        self.assertEqual(slow[1]["methods"]["expanded"]["source_tick"], 128)
        self.assertEqual(len(frames)/V.FPS, 17.5)

    def test_status_does_not_assume_entry_means_retention_or_contact_means_failure(self):
        bundle = dict(events={"board": .2, "robot": .3, "whole_ball_entry": .7, "capture": 1.7})
        self.assertEqual(V.status_line(dict(source_time_s=.4, terminal_hold=False),bundle), "Recorded contact: robot")
        self.assertEqual(V.status_line(dict(source_time_s=.9, terminal_hold=False),bundle), "Whole-ball entry recorded")
        self.assertIn("Recording ended", V.status_line(dict(source_time_s=2,terminal_hold=True),bundle))

    def test_one_camera_covers_both_methods_and_invalid_interval_rejected(self):
        bundles = self.bundles()
        camera = V.common_camera(bundles)
        self.assertTrue(np.isfinite(list(camera["lookat"]) + [camera["distance"],camera["azimuth"],camera["elevation"]]).all())
        bundles["current"]["scored_until_s"] = 5
        with self.assertRaisesRegex(ValueError, "outside"):
            V.frame_plan(bundles)


class VersionedScoreTests(unittest.TestCase):
    def test_pair_uses_v4_without_mutating_historical_score_or_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            study = Path(directory)
            (study/"choices.json").write_text("{}")
            for name in V.METHODS:
                (study/name).mkdir()
                (study/name/"options.json").write_text("{}")
            choices = [dict(id=f"expanded-selector-test-04-{m}", method=m, bucket_xy_m=[.5,-.2]) for m in V.METHODS]
            manifest = {"choices.json":"hash", **{m+"/options.json":"hash" for m in V.METHODS}}
            envelope = ({}, dict(methods=list(V.METHODS),scoring_version=V.SCORING_VERSION), manifest,dict(choices_sha256="hash"))
            bundles={}
            for i,m in enumerate(V.METHODS):
                score=dict(scoring_version=V.SCORING_VERSION, invalid_physics=False, route_success=bool(i),physics_steps=100,
                           whole_ball_entry_time_s=.4 if i else None,capture_time_s=1.4 if i else None)
                row=dict(id=choices[i]["id"],bucket_xy_m=[.5,-.2], route_success=not bool(i),board_first_score=score,
                         physics_steps=100,adapter_directory=m,effective_options_file="options.json",effective_options_sha256="hash")
                bundles[m]=dict(row=row,record_sha256="original-row-hash",events={"board":.1,"capture":.8},scored_until_s=2.,input_sha256={})
            originals=copy.deepcopy(bundles)
            def loaded(a):return bundles[a.record.rsplit("-",1)[1]]
            def reading(path):return dict(slots=choices) if Path(path).name=="choices.json" else {}
            args=SimpleNamespace(study=study,audit=study/"audit",assets=study/"assets",case="04")
            with patch.object(V.recorded,"verify_envelope",return_value=envelope), patch.object(V.recorded,"load_recording",side_effect=loaded), patch.object(V,"sha",return_value="hash"), patch.object(V,"read",side_effect=reading):
                out=V.load_pair(args)
            self.assertFalse(out["current"]["score"]["route_success"])
            self.assertTrue(out["expanded"]["score"]["route_success"])
            self.assertNotIn("capture",out["current"]["events"])
            self.assertEqual(out["expanded"]["events"]["capture"],1.4)
            for m in V.METHODS:
                self.assertEqual(out[m]["row"],originals[m]["row"])
                self.assertEqual(out[m]["record_sha256"],"original-row-hash")

    def test_renderer_never_calls_dynamics_step(self):
        import ast
        module=ast.parse(Path(V.__file__).read_text())
        called={x.func.attr for x in ast.walk(module) if isinstance(x,ast.Call) and isinstance(x.func,ast.Attribute)}
        self.assertIn("mj_forward",called)
        self.assertFalse(called & {"mj_step","mj_step1","mj_step2"})


if __name__ == "__main__":
    unittest.main()
