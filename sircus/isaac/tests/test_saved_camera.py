"""Saved camera imports preserve the scene contract and exact selected view."""

import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from saved_camera import load_camera


class SavedCameraTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "camera.json"
        self.fingerprint = "a" * 64
        self.camera = dict(
            schema="sircus-camera-scout-camera-v1",
            source_fingerprint=self.fingerprint,
            scope="Static camera choice; validate native rendering separately",
            eye_m=[10.9706, -6.1943, 2.5406],
            target_m=[7.8307, -3.4877, 0.3904],
            up_axis="Z",
            focal_length_mm=23,
            horizontal_aperture_mm=36,
            vertical_aperture_mm=20.25,
            resolution=[1920, 1080],
        )

    def load(self, value=None, expected=None):
        self.path.write_text(json.dumps(self.camera if value is None else value))
        return load_camera(
            self.path, self.fingerprint if expected is None else expected
        )

    def test_round_trip_preserves_exact_view_and_all_provenance(self):
        self.camera["provenance"] = dict(source_batch=6, note="user-selected")
        result = self.load()
        self.assertEqual(result, self.camera)
        self.assertEqual(json.loads(json.dumps(result, allow_nan=False)), result)
        result["eye_m"][0] = 0
        self.assertEqual(load_camera(str(self.path), self.fingerprint), self.camera)

    def test_rejects_foreign_contract_and_unsupported_camera_geometry(self):
        for key, value in (
            ("schema", "other-camera-v1"),
            ("source_fingerprint", "b" * 64),
            ("up_axis", "Y"),
            ("focal_length_mm", 24),
            ("horizontal_aperture_mm", 35),
            ("vertical_aperture_mm", 20),
            ("resolution", [1280, 720]),
            ("resolution", [1920.0, 1080.0]),
        ):
            with self.subTest(key=key, value=value):
                changed = dict(self.camera, **{key: value})
                with self.assertRaises(ValueError):
                    self.load(changed)
        for key in self.camera:
            with self.subTest(missing=key):
                if key == "scope":
                    continue
                changed = copy.deepcopy(self.camera)
                del changed[key]
                with self.assertRaises(ValueError):
                    self.load(changed)
        for expected in ("", False, 1, "A" * 64):
            with self.subTest(expected=expected), self.assertRaises(ValueError):
                self.load(expected=expected)

    def test_rejects_malformed_nonfinite_and_degenerate_views(self):
        for key in ("eye_m", "target_m"):
            for value in (
                None,
                {},
                [0, 1],
                [0, 1, 2, 3],
                ["0", 1, 2],
                [False, 1, 2],
                [float("nan"), 1, 2],
                [float("inf"), 1, 2],
                [10**400, 1, 2],
            ):
                with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                    self.load(dict(self.camera, **{key: value}))
        for distance in (0, 0.019, 0.020):
            with self.subTest(distance=distance), self.assertRaises(ValueError):
                self.load(dict(self.camera, eye_m=[0, 0, 0], target_m=[distance, 0, 0]))
        self.load(dict(self.camera, eye_m=[0, 0, 0], target_m=[0.020001, 0, 0]))
        with self.assertRaises(ValueError):
            self.load(dict(self.camera, eye_m=[1e308, 0, 0], target_m=[-1e308, 0, 0]))

    def test_rejects_nonobject_invalid_json_and_nonfinite_extra_metadata(self):
        for value in (
            [],
            "camera",
            42,
            True,
            dict(self.camera, extra=float("inf")),
        ):
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.load(value)
        for text in (
            "{",
            "null",
            json.dumps(dict(self.camera, extra=1)).replace(
                '"extra": 1', '"extra": 1e999'
            ),
        ):
            with self.subTest(text=text), self.assertRaises(ValueError):
                self.path.write_text(text)
                load_camera(self.path, self.fingerprint)


if __name__ == "__main__":
    unittest.main()
