"""Outcome trims preserve unsuccessful attempts and full scoring provenance."""

import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

ISAAC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ISAAC))
from action_training_export import (
    MOVE_FRAMES,
    score_focus,
    terminal_cut,
    render_action_cut,
)
from board_route import BoardRoute
from entry_quality import EntryQuality
from rl_contract import ShotScore


class ActionTrainingExportTests(unittest.TestCase):
    def setUp(self):
        self.scene = json.loads((ISAAC / "scene.json").read_text())
        self.origin = np.array([7.5, -4.2, 0.0])
        self.xy = self.scene["bucket"]["center_xy_m"]
        self.row = dict(invalid=None, route_success=False, max_retention_s=0)

    def clip(self, samples):
        telemetry = []
        for step, pos, forces in samples:
            contacts = dict.fromkeys(
                ("board", "bucket", "table", "floor", "robot", "unattributed"), 0.0
            )
            contacts.update(forces)
            telemetry.append(
                dict(
                    physics_step=step,
                    ball_world_m=(np.array(pos) + self.origin).tolist(),
                    ball_velocity_m_s=[0, 0, 0.5],
                    forces_N=contacts,
                )
            )
        frames = [
            dict(
                frame=k,
                phase="move",
                physics_step=-36 + k,
                eye_m=[7.5, -6.2, 1],
                target_m=[7.5, -4.2, 0],
            )
            for k in range(MOVE_FRAMES)
        ]
        for step in [0, *range(8, samples[-1][0] + 1, 8)]:
            frames.append(
                dict(
                    frame=len(frames),
                    phase="release",
                    physics_step=step,
                    eye_m=[7.5, -6.2, 1],
                    target_m=[7.5, -4.2, 0],
                )
            )
        return dict(frames=frames, focus_telemetry=telemetry, release_step=0)

    def audit(self, captured=False, entry=None):
        return dict(
            physical_retained_capture=captured,
            successful_retention_start_step=entry,
            first_board_contact_step=3,
            scoring_stop_step=300,
        )

    def test_rim_then_entry_cuts_at_final_successful_interval_not_first_contact(self):
        samples = [
            (i, [*self.xy, 0.30], {"bucket": 1.0} if i == 40 else {})
            for i in range(1, 401)
        ]
        clip = self.clip(samples)
        row = dict(self.row, route_success=True, max_retention_s=1.0)
        cut = terminal_cut(clip, row, self.audit(True, 120), self.origin)
        self.assertEqual(cut["terminal_reason"], "retained_bucket_entry")
        self.assertEqual(cut["terminal_physics_step"], 120)
        self.assertEqual(cut["source_end_frame_inclusive"], MOVE_FRAMES + 21)
        self.assertEqual(cut["source_start_frame"], 0)
        self.assertLess(cut["source_frames"], len(clip["frames"]))

    def test_miss_keeps_rim_contact_and_ends_after_measured_floor(self):
        samples = [
            (
                i,
                [0, 0, -0.73 if i >= 160 else 0.25],
                {"bucket": 5} if i == 80 else {"floor": 20} if i == 160 else {},
            )
            for i in range(1, 241)
        ]
        clip = self.clip(samples)
        cut = terminal_cut(clip, self.row, self.audit(), self.origin)
        self.assertEqual(cut["terminal_reason"], "floor_contact")
        self.assertEqual(cut["terminal_physics_step"], 160)
        self.assertEqual(cut["source_end_frame_inclusive"], MOVE_FRAMES + 26)

    def test_table_then_rim_then_floor_does_not_hide_later_bucket_interaction(self):
        samples = [
            (
                i,
                [0, 0, 0.25],
                (
                    {"table": 1}
                    if i == 30
                    else {"bucket": 1} if i == 80 else {"floor": 1} if i == 120 else {}
                ),
            )
            for i in range(1, 201)
        ]
        cut = terminal_cut(self.clip(samples), self.row, self.audit(), self.origin)
        self.assertEqual(cut["terminal_reason"], "floor_contact")
        self.assertEqual(cut["terminal_physics_step"], 120)

    def test_unresolved_rim_only_keeps_all_frames_and_never_labels_miss(self):
        samples = [
            (i, [*self.xy, 0.18], {"bucket": 3} if i >= 100 else {})
            for i in range(1, 961)
        ]
        clip = self.clip(samples)
        cut = terminal_cut(clip, self.row, self.audit(), self.origin)
        self.assertEqual(cut["terminal_reason"], "timeout_unresolved")
        self.assertEqual(cut["source_frames"], len(clip["frames"]))

    def test_explicit_below_table_exit_does_not_claim_floor_contact(self):
        samples = [(i, [0, 0, -0.25 if i >= 80 else 0.3], {}) for i in range(1, 161)]
        cut = terminal_cut(self.clip(samples), self.row, self.audit(), self.origin)
        self.assertEqual(cut["terminal_reason"], "workspace_exit_below_table")
        self.assertEqual(cut["terminal_physics_step"], 80)

    def test_invalid_foreground_is_retained_with_reason_and_complete_setup(self):
        samples = [(i, [0.1, -0.8, 0.04], {}) for i in range(1, 81)]
        clip = self.clip(samples)
        row = dict(self.row, invalid="endpoint_ik_Arm10")
        cut = terminal_cut(clip, row, self.audit(), self.origin)
        self.assertEqual(cut["terminal_reason"], "endpoint_rejected")
        self.assertEqual(cut["invalid"], row["invalid"])
        self.assertGreaterEqual(cut["source_frames"], MOVE_FRAMES + 1)

    def test_full_rate_score_resets_retention_after_partial_exit(self):
        samples = []
        for i in range(1, 501):
            pos = [*self.xy, 0.10]
            if i <= 3 or i == 100:
                pos = [*self.xy, 0.25]
            samples.append((i, pos, {"board": 1} if i == 1 else {}))
        clip = self.clip(samples)
        score = ShotScore(self.scene["bucket"], self.scene["ball"]["radius_m"])
        quality = EntryQuality(self.scene["bucket"], self.scene["ball"]["radius_m"])
        route = BoardRoute(1 / 240)
        for item in clip["focus_telemetry"]:
            score.step(
                np.array(item["ball_world_m"]) - self.origin,
                item["forces_N"]["board"],
                1 / 240,
                0.5,
            )
            quality.step(
                np.array(item["ball_world_m"]) - self.origin,
                item["forces_N"]["bucket"],
                score.entered,
            )
            route.step(item["forces_N"])
            if score.success:
                break
        row = dict(
            self.row,
            **score.result(),
            **quality.result(score.success),
            **route.result(
                score.success, quality.result(score.success)["quality_score"]
            ),
            bucket_xy_m=self.xy,
            settled_board_normal=[0, 0, 1],
        )
        audit = score_focus(clip, row, self.scene, self.origin)
        self.assertEqual(audit["successful_retention_start_step"], 101)
        self.assertEqual(audit["scoring_stop_step"], 340)
        row["max_retention_s"] = 0.5
        with self.assertRaisesRegex(ValueError, "retention"):
            score_focus(clip, row, self.scene, self.origin)
        clip["focus_telemetry"].pop(10)
        with self.assertRaisesRegex(ValueError, "continuous"):
            score_focus(clip, row, self.scene, self.origin)

    def test_action_cut_reindexes_only_original_pixels_and_keeps_every_batch(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            (run / "frames").mkdir()
            for i in range(20):
                (run / "frames" / f"{i:05d}.jpg").write_bytes(bytes([i]))
            chapters = [
                dict(
                    title=f"Batch {i}",
                    source_start_frame=i * 10,
                    source_end_frame_inclusive=i * 10 + 3,
                )
                for i in range(2)
            ]

            def inspect(frames, path, chapters, res):
                self.assertEqual(
                    [p.read_bytes()[0] for p in sorted(frames.glob("*.jpg"))],
                    [0, 1, 2, 3, 10, 11, 12, 13],
                )
                self.assertEqual([c["start_frame"] for c in chapters], [0, 4])
                return dict(frames=8)

            with patch("action_training_export.encode_movie", side_effect=inspect):
                result = render_action_cut(
                    run, run / "unused.mp4", chapters, [1920, 1080]
                )
            self.assertEqual(len(result["chapters"]), 2)
            self.assertTrue(result["native_pixels_only"])


if __name__ == "__main__":
    unittest.main()
