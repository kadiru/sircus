"""Chosen camera identity and eight-drop one-minute export contracts."""

import copy
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dark_export import encode_all, selected_camera, validate_chosen_farm
from saved_camera import INTRINSICS, RESOLUTION, SCHEMA


def chapters():
    orbit = [
        dict(
            shot=f"orbit-{i}",
            start_frame=120 * i,
            frames=120,
            station=False,
            action_only=False,
            focus=None,
        )
        for i in range(4)
    ]
    station = [
        dict(
            shot=f"station-{i}",
            start_frame=480 + 165 * i,
            frames=165,
            station=True,
            action_only=True,
            focus="r1c2",
            visible_board_frame=530 + 165 * i,
            background_drops_in_view=2,
        )
        for i in range(8)
    ]
    return orbit + station


class ChosenExportTests(unittest.TestCase):
    def test_accepts_eight_measured_events_without_applying_ten_x_minimum(self):
        validate_chosen_farm(chapters(), 1800)

    def test_rejects_missing_repeated_or_unmeasured_station_events(self):
        with self.assertRaisesRegex(ValueError, "1800"):
            validate_chosen_farm(chapters(), 1799)
        with self.assertRaisesRegex(ValueError, "four orbit and eight"):
            validate_chosen_farm(chapters()[:-1], 1800)
        for field, value in (
            ("start_frame", 480),
            ("focus", "r1c3"),
            ("visible_board_frame", None),
            ("background_drops_in_view", 1),
            ("station", False),
            ("action_only", False),
        ):
            with self.subTest(field=field):
                changed = chapters()
                changed[-1][field] = value
                with self.assertRaises(ValueError):
                    validate_chosen_farm(changed, 1800)
        changed = chapters()
        changed[0]["focus"] = "r1c2"
        with self.assertRaisesRegex(ValueError, "orbit"):
            validate_chosen_farm(changed, 1800)

    def test_original_camera_bytes_json_and_source_scene_must_agree(self):
        camera = dict(
            schema=SCHEMA,
            source_fingerprint="verified-source",
            up_axis="Z",
            eye_m=[10.0, -8.0, 1.8],
            target_m=[8.0, -4.0, 0.2],
            resolution=RESOLUTION,
            provenance=dict(selected=True),
            **INTRINSICS,
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "selected-camera.json"
            raw = (json.dumps(camera, indent=2) + "\n").encode()
            path.write_bytes(raw)
            result = dict(
                selected_camera=camera,
                selected_camera_sha256=hashlib.sha256(raw).hexdigest(),
            )
            self.assertEqual(selected_camera(root, result, "verified-source"), camera)
            path.write_bytes(raw + b"\n")
            with self.assertRaisesRegex(ValueError, "digest"):
                selected_camera(root, result, "verified-source")
            path.write_bytes(raw)
            with self.assertRaisesRegex(ValueError, "fingerprint"):
                selected_camera(root, result, "another-scene")
            altered = copy.deepcopy(result)
            altered["selected_camera"]["eye_m"][0] = 11.0
            with self.assertRaisesRegex(ValueError, "JSON"):
                selected_camera(root, altered, "verified-source")

    def test_chosen_export_encodes_all_1800_frames_and_untrimmed_front_row_only(self):
        scene = chapters()
        summary = dict(
            recording_plan="chosen-farm-v1", chapters=scene, resolution=[1920, 1080]
        )
        calls = []

        def encode(frames, path, clips, resolution):
            calls.append((path.name, clips))
            return dict(file=path.name, frames=sum(c["frames"] for c in clips))

        with tempfile.TemporaryDirectory() as tmp, patch(
            "dark_export.encode_movie", side_effect=encode
        ), patch(
            "dark_export.render_cut",
            side_effect=AssertionError("unexpected trimming/10x"),
        ), patch(
            "dark_export.verify_fast_events",
            side_effect=AssertionError("unexpected 10x gate"),
        ), patch(
            "dark_export.contact_sheet"
        ):
            root = Path(tmp)
            encode_all(root, summary, root)
            self.assertEqual(
                [c[0] for c in calls], ["farm-one-minute.mp4", "front-row.mp4"]
            )
            self.assertEqual(summary["full_movie"]["frames"], 1800)
            self.assertEqual(summary["front_row_movie"]["frames"], 1320)
            self.assertEqual(calls[1][1][0]["start_frame"], 480)
            self.assertNotIn("station_10x_movie", summary)
            self.assertTrue((root / "film.json").is_file())


if __name__ == "__main__":
    unittest.main()
