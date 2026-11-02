"""A floor miss must be measured after the board bounce and filmed in the foreground."""

import sys
import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

ISAAC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ISAAC))
from dark_export import (
    assess_focus,
    event_frame,
    inside_frame,
    render_cut,
    highlight_span,
    verify_resume,
    validate_station_series,
    verify,
)
from dark_stage import FLOOR_Z_M
from export_cylinder_film import read


class DarkExportTests(unittest.TestCase):
    def test_station_export_counts_actual_visible_rebounds_with_active_background(self):
        chapters = [
            dict(
                focus="r1c2",
                start_frame=i * 252,
                station=True,
                visible_board_frame=i * 252 + 50,
                background_drops_in_view=2,
            )
            for i in range(10)
        ]
        validate_station_series(chapters)
        with self.assertRaisesRegex(ValueError, "at least 10"):
            validate_station_series(chapters[:9])
        for field, value in (
            ("visible_board_frame", None),
            ("background_drops_in_view", 1),
            ("station", False),
            ("focus", "r1c3"),
            ("start_frame", 0),
        ):
            with self.subTest(field=field):
                altered = copy.deepcopy(chapters)
                altered[-1][field] = value
                with self.assertRaises(ValueError):
                    validate_station_series(altered)

    def test_historical_export_without_plan_metadata_selects_original_camera(self):
        # Stop before artifact loading: this tests the dispatch of old result.json.
        with patch("dark_export.read", return_value={}), patch(
            "dark_export.recording_plan", side_effect=RuntimeError("selected plan")
        ) as choose:
            with self.assertRaisesRegex(RuntimeError, "selected plan"):
                verify(Path("run"), Path("source"))
            choose.assert_called_once_with("dark-tour-v1")

    def test_resume_requires_unchanged_original_rgb_and_shot_records(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            previous, resumed = base / "previous", base / "resumed"
            clip = json.dumps(dict(frames=[dict(frame=0)])).encode()
            for folder in (previous, resumed):
                (folder / "frames").mkdir(parents=True)
                (folder / "frames" / "00000.jpg").write_bytes(b"native-rgb")
                (folder / "clip-00.json").write_bytes(clip)
                (folder / "probes.json").write_text(json.dumps([dict(all_passed=True)]))
            info = dict(
                previous_run=previous.name,
                completed_chapters=1,
                clip_sha256={"clip-00.json": hashlib.sha256(clip).hexdigest()},
                frames_sha256=hashlib.sha256(b"00000.jpgnative-rgb").hexdigest(),
            )
            verify_resume(resumed, dict(resumed=info))
            (resumed / "frames" / "00000.jpg").write_bytes(b"replacement-rgb")
            with self.assertRaisesRegex(ValueError, "RGB changed"):
                verify_resume(resumed, dict(resumed=info))

    def setUp(self):
        self.scene = read(ISAAC / "scene.json")
        self.radius = self.scene["ball"]["radius_m"]
        self.floor = [0, 0, FLOOR_Z_M + self.radius]
        self.positions = [
            [0, 0, 0.6],
            [0, 0, 0.36],
            [0, 0.02, 0.40],
            self.floor,
            self.floor,
        ]
        self.frames = [
            dict(
                frame=20 + i,
                physics_step=100 + 4 * i,
                eye_m=[0, -1.9, 0],
                target_m=[0, 0, 0],
                focus_ball_world_m=p,
            )
            for i, p in enumerate(self.positions)
        ]
        self.contacts = dict(
            table=None,
            bucket=None,
            board=dict(physics_step=104, force_N=14.0, ball_world_m=self.positions[1]),
            floor=dict(physics_step=111, force_N=11.0, ball_world_m=self.floor),
        )
        self.chapter = dict(
            shot="test",
            move_frames=0,
            action_only=True,
            board_bounce=True,
            captured=False,
            literal_floor=True,
        )
        self.clip = dict(frames=self.frames, focus_contacts=self.contacts)
        self.row = dict(
            invalid=None,
            board_bounce=True,
            route_success=False,
            release_ball_center_m=self.positions[0],
        )

    def check_focus(self):
        return assess_focus(
            self.chapter,
            self.clip,
            self.row,
            self.scene["bucket"],
            self.radius,
            np.zeros(3),
        )

    def test_requires_first_visible_measured_floor_after_board(self):
        outcome = self.check_focus()
        self.assertEqual(outcome["visible_floor_frame"], 23)
        self.assertEqual(outcome["visible_board_frame"], 21)
        self.contacts["floor"]["physics_step"] = 200
        with self.assertRaisesRegex(ValueError, "outside shot"):
            self.check_focus()
        self.contacts["floor"]["physics_step"] = 103
        with self.assertRaisesRegex(ValueError, "outside shot"):
            self.check_focus()

    def test_a_falling_ball_is_insufficient_floor_evidence(self):
        self.contacts["floor"]["force_N"] = 0.001
        with self.assertRaisesRegex(ValueError, "no measured contact"):
            self.check_focus()
        self.contacts["floor"]["force_N"] = 12.0
        self.contacts["floor"]["ball_world_m"] = [0, 0, 0.0375]
        with self.assertRaisesRegex(ValueError, "floor height"):
            self.check_focus()

    def test_point_projection_requires_in_front_and_in_the_shot(self):
        frame = dict(eye_m=[0, -2, 0], target_m=[0, 0, 0])
        self.assertTrue(inside_frame(frame, [0, 0, 0.2]))
        self.assertFalse(inside_frame(frame, [0, -3, 0]))
        self.assertFalse(inside_frame(frame, [5, 0, 0]))
        self.contacts["floor"]["ball_world_m"] = [10, 0, FLOOR_Z_M + self.radius]
        self.assertIsNone(event_frame(self.contacts["floor"], self.frames, "Floor"))

    def test_two_disjoint_shots_are_reindexed_for_encoder(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            raw = folder / "frames"
            raw.mkdir()
            for i in range(40):
                (raw / f"{i:05d}.jpg").write_bytes(bytes([i]))
            chapters = [
                dict(shot="a", start_frame=0, frames=20, move_frames=16),
                dict(shot="b", start_frame=29, frames=3, move_frames=0),
            ]

            def inspect(numbered, path, stitched, resolution):
                sources = [p.read_bytes()[0] for p in sorted(numbered.glob("*.jpg"))]
                return dict(
                    source_start_frame=0,
                    sources=sources,
                    chapter_frames=[(c["start_frame"], c["frames"]) for c in stitched],
                )

            with patch("dark_export.encode_movie", side_effect=inspect):
                normal = render_cut(raw, folder / "unused.mp4", chapters, [1920, 1080])
                self.assertEqual(normal["sources"], list(range(20)) + [29, 30, 31])
                self.assertEqual(normal["chapter_frames"], [(0, 20), (20, 3)])
                fast = render_cut(
                    raw, folder / "fast.mp4", chapters, [1920, 1080], step=10
                )
                self.assertEqual(fast["sources"], normal["sources"][::10])
                self.assertEqual(fast["playback_multiple"], 10)

    def test_highlights_end_after_impact_with_a_short_hold(self):
        c = dict(
            start_frame=300,
            frames=198,
            move_frames=36,
            visible_board_frame=357,
            visible_floor_frame=400,
            captured=False,
        )
        self.assertEqual(highlight_span(c), (318, 416))
        c.update(
            visible_floor_frame=None,
            visible_table_contact_frame=370,
            visible_bucket_contact_frame=372,
        )
        self.assertEqual(highlight_span(c), (318, 388))
        c.update(captured=True, visible_bucket_entry_frame=364)
        self.assertEqual(highlight_span(c), (318, 380))


if __name__ == "__main__":
    unittest.main()
