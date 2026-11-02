"""Portable style metadata and visible events in an accelerated native recording."""

import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dark_export import verify_fast_events
from dark_stage import stage_settings as original_stage_settings
from silver_style import stage_settings


class SilverRecordingTests(unittest.TestCase):
    def test_selected_style_roundtrips_without_changing_geometry(self):
        selected = stage_settings()
        portable = json.loads(json.dumps(selected))
        original = json.loads(json.dumps(original_stage_settings()))
        for key in (
            "columns",
            "rows",
            "column_pitch_m",
            "row_pitch_m",
            "floor_surface_z_m",
            "floor_visual_span_m",
            "floor_collision",
        ):
            self.assertEqual(portable[key], original[key])
        self.assertEqual(portable["floor_linear_rgb"], portable["seam_linear_rgb"])
        self.assertLess(
            max(portable["floor_linear_rgb"]) - min(portable["floor_linear_rgb"]), 0.03
        )
        self.assertGreater(
            portable["table_linear_rgb"][2], portable["table_linear_rgb"][0]
        )
        self.assertGreater(portable["table_roughness"], 0)
        self.assertLessEqual(portable["table_roughness"], 1)
        self.assertFalse(portable["ambient_occlusion"])
        self.assertGreater(portable["lighting"]["dome_fill"], 900)
        self.assertLess(portable["lighting"]["key"]["intensity"], 360)
        selected["lighting"]["fills"][0]["intensity"] = -1
        self.assertGreater(stage_settings()["lighting"]["fills"][0]["intensity"], 0)

    def fast_fixture(self):
        chapters, spans = [], []
        for index in range(12):
            start = index * 200
            captured = bool(index % 2)
            chapters.append(
                dict(
                    start_frame=start,
                    move_frames=36,
                    visible_board_frame=start + 55,
                    visible_floor_frame=None if captured else start + 95,
                    visible_bucket_entry_frame=start + 95 if captured else None,
                    visible_bucket_contact_frame=None,
                    visible_table_contact_frame=None,
                    captured=captured,
                )
            )
            spans.append(
                dict(
                    first_sampled_frame=start,
                    end_source_frame_exclusive=start + 120,
                )
            )
        return chapters, spans

    def test_true_tenfold_sampling_retains_twelve_falls_and_rebounds(self):
        chapters, spans = self.fast_fixture()
        self.assertEqual(verify_fast_events(chapters, spans), 12)

    def test_one_falling_sample_does_not_count_as_a_visible_drop(self):
        chapters, spans = self.fast_fixture()
        # Sampled frames 40 and 50 show descent before contact at 55. Moving
        # contact to 45 leaves only frame 40, so three chapters must stop counting.
        for chapter in chapters[:3]:
            chapter["visible_board_frame"] = chapter["start_frame"] + 45
        with self.assertRaisesRegex(ValueError, "Fewer than ten drops"):
            verify_fast_events(chapters, spans)

    def test_descent_without_a_retained_rebound_is_rejected(self):
        chapters, spans = self.fast_fixture()
        for chapter in chapters[:3]:
            landing = (
                "visible_bucket_entry_frame"
                if chapter["captured"]
                else "visible_floor_frame"
            )
            chapter[landing] = chapter["visible_board_frame"]
        with self.assertRaisesRegex(ValueError, "Fewer than ten drops"):
            verify_fast_events(chapters, spans)


if __name__ == "__main__":
    unittest.main()
