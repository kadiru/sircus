"""Regression: small endpoint changes must not stall camera capture readiness."""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from recording_plans import setup_capture_progress, transition_group_ready


class SetupCaptureTests(unittest.TestCase):
    def test_tiny_move_with_residual_still_records_all_setup_frames(self):
        # Native failing case was already within the joint-readiness tolerance,
        # but its relative measured progress plateaued after three RGB frames.
        thresholds = np.linspace(0.03, 0.995, 36)
        captured = []
        for step in range(4, 241, 4):
            progress = setup_capture_progress(step, 240, 0.09, training=True)
            if len(captured) < 36 and progress >= thresholds[len(captured)]:
                captured.append(step)
        self.assertEqual(len(captured), 36)
        self.assertEqual(len(set(captured)), 36)
        self.assertLessEqual(captured[-1], 240)

    def test_legacy_relative_progress_is_not_changed(self):
        for residual_limited in (0.0, 0.09, 0.97, 1.01):
            self.assertEqual(
                setup_capture_progress(1000, 240, residual_limited),
                residual_limited,
            )

    def test_training_camera_progress_caps_at_end_of_command(self):
        self.assertEqual(setup_capture_progress(400, 240, 0.1, training=True), 1.0)

    def test_all_cell_training_does_not_finish_with_two_ready_backgrounds(self):
        ready = [True, True, True, False]
        self.assertTrue(transition_group_ready(ready, True, 2))
        self.assertFalse(transition_group_ready(ready, True, 2, require_all=True))

    def test_all_qualified_cells_and_foreground_must_be_ready(self):
        self.assertTrue(transition_group_ready([True] * 35, True, 0, require_all=True))
        self.assertFalse(
            transition_group_ready([True] * 35, False, 34, require_all=True)
        )


if __name__ == "__main__":
    unittest.main()
