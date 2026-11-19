"""Keep ten-plus real releases in one framing, including the background farm."""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dark_stage import origins
from station_flight import FOREGROUND, MIN_FOREGROUND_DROPS, SHOTS, camera_pose


def project(eye, aim, points):
    forward = aim - eye
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, [0, 0, 1])
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    rays = np.asarray(points) - eye
    depth = rays @ forward
    uv = rays @ np.stack([right, up]).T * 23 / (depth[:, None] * [18, 10.125])
    return depth, uv


class StationFlightTests(unittest.TestCase):
    def test_camera_ignores_ball_positions_frames_and_changed_board_configuration(self):
        world = origins()
        expected = camera_pose(SHOTS[0], 0, [], world)
        balls = (None, [-100, 5, 20], [20, -90, -4], [0, 0, 0])
        for shot in SHOTS:
            for frame, ball in zip((0, 36, 120, 251), balls):
                cells = [dict(board_xy_m=[frame, -frame], bucket_xy_m=[frame / 9, 8])]
                actual = camera_pose(shot, frame, cells, world, ball)
                np.testing.assert_array_equal(actual, expected)
        np.testing.assert_array_equal(camera_pose(None, 0, None), expected)

    def test_plan_has_more_than_ten_separate_releases_at_one_foreground_station(self):
        self.assertGreaterEqual(len(SHOTS), MIN_FOREGROUND_DROPS)
        self.assertEqual(len({shot.name for shot in SHOTS}), len(SHOTS))
        self.assertEqual({shot.focus for shot in SHOTS}, {FOREGROUND})
        self.assertGreaterEqual(len({shot.source_batch for shot in SHOTS}), 4)
        for shot in SHOTS:
            flight_seconds = (shot.frames - shot.move_frames - 1) * shot.stride / 240
            self.assertGreater(flight_seconds, 2)
            self.assertGreater(shot.move_frames, 0)

    def test_fixed_view_contains_release_board_bucket_floor_and_background_rows(self):
        world = origins()
        eye, aim = camera_pose(SHOTS[0], 0, [], world)
        # Include the previously observed leftward floor miss without moving the lens.
        foreground = world[FOREGROUND] + [
            [0.45, 0.10, 0.38],
            [0.55, 0.36, 0.70],
            [0.155, 0.25, 0.18],
            [-1.20, -0.10, -0.7375],
        ]
        depth, uv = project(eye, aim, foreground)
        self.assertTrue(np.all(depth > 0.04))
        self.assertLess(np.max(abs(uv)), 0.95)
        for row in range(1, 5):
            points = world[row * 7 : (row + 1) * 7] + [0.67, -0.25, 0.45]
            depth, uv = project(eye, aim, points)
            visible = (depth > 0.04) & np.all(abs(uv) < 0.95, axis=1)
            self.assertGreaterEqual(np.count_nonzero(visible), 1)


if __name__ == "__main__":
    unittest.main()
