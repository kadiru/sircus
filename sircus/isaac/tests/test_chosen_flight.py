"""Exact one-minute framing, continuous orbit and a motion-independent close view."""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from chosen_flight import SHOTS, camera_pose
from recording_plans import plan_origins, recording_plan
from dark_stage import origins


CAMERA = dict(eye_m=[10.9706, -6.1943, 2.5406], target_m=[7.8307, -3.4877, 0.3904])


class ChosenFlightTests(unittest.TestCase):
    def test_one_minute_has_an_orbit_then_eight_same_table_drops_with_motion(self):
        self.assertEqual(sum(s.frames for s in SHOTS), 60 * 30)
        self.assertEqual(sum(s.frames for s in SHOTS[:4]), 16 * 30)
        self.assertTrue(all(s.focus is None for s in SHOTS[:4]))
        self.assertEqual(len(SHOTS[4:]), 8)
        self.assertEqual({s.focus for s in SHOTS[4:]}, {1})
        self.assertEqual({s.move_frames for s in SHOTS[4:]}, {36})
        self.assertEqual([s.source_batch for s in SHOTS[4:]], [4, 3, 6, 0] * 2)

    def test_close_view_never_tracks_ball_or_configuration(self):
        for shot in SHOTS[4:]:
            for frame, ball in (
                (0, None),
                (36, [100, -100, 3]),
                (164, [-300, 900, -10]),
            ):
                eye, aim = camera_pose(
                    shot, frame, [dict(arbitrary=frame)], origins(), ball, camera=CAMERA
                )
                np.testing.assert_array_equal(eye, CAMERA["eye_m"])
                np.testing.assert_array_equal(aim, CAMERA["target_m"])

    def test_orbit_completes_circle_then_continuously_enters_selected_view(self):
        points = [
            camera_pose(s, f, [], origins(), camera=CAMERA)
            for s in SHOTS[:4]
            for f in range(s.frames)
        ]
        np.testing.assert_allclose(points[0][0], points[419][0], atol=1e-12)
        np.testing.assert_array_equal(points[-1][0], CAMERA["eye_m"])
        np.testing.assert_array_equal(points[-1][1], CAMERA["target_m"])
        self.assertLess(
            np.max(np.linalg.norm(np.diff([p[0] for p in points], axis=0), axis=1)), 0.5
        )
        self.assertTrue(all(p[0][2] >= CAMERA["eye_m"][2] for p in points))
        np.testing.assert_array_equal(
            points[-1], camera_pose(SHOTS[4], 0, [], origins(), camera=CAMERA)
        )

    def test_saved_camera_is_required_and_scene_permutation_preserves_all_sites(self):
        with self.assertRaisesRegex(ValueError, "saved camera"):
            recording_plan("chosen-farm-v1")
        plan, view = recording_plan("chosen-farm-v1", CAMERA)
        self.assertEqual(plan, SHOTS)
        world = plan_origins("chosen-farm-v1")
        self.assertEqual({tuple(v) for v in world}, {tuple(v) for v in origins()})
        np.testing.assert_array_equal(world[1], origins()[6])
        np.testing.assert_array_equal(
            view(plan[-1], 0, [], world), [CAMERA["eye_m"], CAMERA["target_m"]]
        )
        np.testing.assert_array_equal(plan_origins("dark-tour-v1"), origins())


if __name__ == "__main__":
    unittest.main()
