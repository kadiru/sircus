"""Frame coverage and motion/release timing for the compact Isaac farm film."""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dark_flight import SHOTS, camera_pose
from dark_stage import origins


def project(eye, aim, points):
    forward = aim - eye
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, [0, 0, 1])
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    relative = np.asarray(points) - eye
    depth = relative @ forward
    uv = relative @ np.stack([right, up]).T * 23 / (depth[:, None] * [18, 10.125])
    return depth, uv


class DarkFlightTests(unittest.TestCase):
    def test_shot_clock_includes_moves_and_actual_fall_time(self):
        self.assertEqual(sum(s.frames for s in SHOTS), 30 * 60)
        front = [s for s in SHOTS if s.name.startswith(("station-", "front-"))]
        self.assertGreaterEqual(len(front), 6)
        self.assertGreaterEqual(len({s.focus for s in front}), 3)
        self.assertTrue(all(0 <= s.focus < 7 and s.move_frames == 36 for s in front))
        station = [s for s in front if s.name.startswith("station-")]
        self.assertEqual([s.focus for s in station], [1] * 4)
        self.assertEqual([s.source_batch for s in station], [4, 3, 6, 0])
        for s in SHOTS:
            if s.focus is not None:
                self.assertGreater((s.frames - s.move_frames - 1) * s.stride / 240, 2.0)

    def test_orbit_and_overhead_keep_all_35_tables_in_frame(self):
        world = origins()
        corners = [
            [x, y, z] for x in (-0.23, 1.57) for y in (-0.875, 0.375) for z in (0, 1.0)
        ]
        points = (world[:, None] + corners).reshape(-1, 3)
        for shot in (SHOTS[0], SHOTS[-1]):
            for frame in range(0, shot.frames, 5):
                eye, aim = camera_pose(shot, frame, [], world)
                depth, uv = project(eye, aim, points)
                self.assertTrue(np.all(depth > 0))
                self.assertLess(np.max(abs(uv)), 0.97)

    def test_station_holds_same_angle_then_tracks_floor_beyond_table(self):
        world = origins()
        station = [s for s in SHOTS if s.name.startswith("station-")]
        cells = [dict(bucket_xy_m=[0.156, 0.25], board_xy_m=[0.46, -0.005]) for _ in world]
        reference = camera_pose(station[0], 0, cells, world)
        for shot in station:
            origin = world[shot.focus]
            eye, aim = camera_pose(shot, 0, cells, world)
            np.testing.assert_allclose((eye, aim), reference)
            for row in range(1, 5):
                pts = world[row * 7 : (row + 1) * 7] + [0.67, -0.25, 0.45]
                d, uv = project(eye, aim, pts)
                self.assertTrue(np.any((d > 0) & np.all(abs(uv) < 0.95, axis=1)))
            # The release, board and bucket stay visible as a group.
            pts = origin + [[0.55, 0.36, 0.68], [0.55, 0.36, 0.32], [0.156, 0.25, 0.183]]
            d, uv = project(eye, aim, pts)
            self.assertTrue(np.all(d > 0))
            self.assertLess(np.max(abs(uv)), 0.8)
            # r1c2 misses continue off the left edge, below the tabletop.
            floor = origin + [-1.18, 0.32, -0.7375]
            eye, aim = camera_pose(shot, shot.move_frames + 85, cells, world, floor)
            self.assertLess(eye[2], 0)
            self.assertLess(eye[1], origin[1] - 0.875)
            d, uv = project(eye, aim, floor[None])
            self.assertGreater(d[0], 0)
            self.assertLess(np.max(abs(uv)), 0.65)


if __name__ == "__main__":
    unittest.main()
