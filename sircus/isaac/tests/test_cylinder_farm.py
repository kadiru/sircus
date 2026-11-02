"""Spatial and framing contracts for the 35-table Isaac training shoot."""

import json
import sys
import unittest
from pathlib import Path

import numpy as np

ISAAC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ISAAC))
from cylinder_farm import (
    HIGH,
    LOW,
    actions,
    layouts,
    menu_batches,
    origins,
    project_actions,
)
from farm_flight import SHOTS, camera_pose, choose_front


class CylinderFarmTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.scene = json.loads((ISAAC / "scene.json").read_text())
        cls.cells = layouts(cls.scene)

    def test_bucket_tiles_correspond_to_actual_clone_origins(self):
        world = origins()
        self.assertEqual(world.shape, (35, 3))
        columns = np.unique(world[:, 0])
        rows = np.unique(world[:, 1])
        self.assertEqual((len(columns), len(rows)), (7, 5))
        self.assertEqual(len({tuple(p) for p in world}), 35)
        table = self.scene["table"]
        size = np.array(table["size_m"][:2])
        lower = np.array(table["center_m"][:2]) - size / 2
        bucket_radius = self.scene["bucket"]["outer_top_diameter_m"] / 2
        self.assertEqual(len(self.cells), 35)
        for cell, (x, y, _) in zip(self.cells, world):
            bucket = np.array(cell["bucket_xy_m"])
            tile = np.floor((bucket - lower) / (size / [7, 5])).astype(int)
            coord = [np.searchsorted(columns, x), np.searchsorted(rows, y)]
            np.testing.assert_array_equal(tile, [coord[0], 4 - coord[1]])
            self.assertEqual(cell["bucket_tile_row"], tile[1])
            self.assertTrue(np.all(bucket - bucket_radius >= lower))
            self.assertTrue(np.all(bucket + bucket_radius <= lower + size))

    def test_action_changes_azimuth_and_allows_a_low_board_with_high_release(self):
        latents = np.tile([0.0, -4, 4, 0], (35, 1))
        first = actions(self.cells, latents)
        latents[:, 0] = np.arctanh(0.5)  # quarter turn
        second = actions(self.cells, latents)
        bucket = self.scene["bucket"]
        rim = bucket["assumed_bottom_thickness_m"] + bucket["assumed_interior_depth_m"]
        for cell, a, b in zip(self.cells, first, second):
            xy = np.array(cell["bucket_xy_m"])
            u = np.array([a["board_x_m"], a["board_y_m"]]) - xy
            v = np.array([b["board_x_m"], b["board_y_m"]]) - xy
            self.assertAlmostEqual(np.linalg.norm(u), 0.400)
            np.testing.assert_allclose(v, [-u[1], u[0]], atol=1e-7)
            self.assertLess(b["height_m"], rim)
            self.assertGreater(b["height_m"] + b["gap_m"], rim)

    def test_projection_preserves_radius_and_cannot_choose_rejected_candidates(self):
        bank = list(menu_batches(self.cells))
        self.assertEqual(len(bank), 60)
        arrays = []
        menus = []
        for i, cell in enumerate(self.cells):
            available = [bank[j][i] for j in [0, 1, 2, 3]]
            heights = [p["height_m"] for p in available]
            self.assertLess(min(heights), 0.175)
            for batch in bank:
                p = batch[i]
                delta = np.array(cell["bucket_xy_m"]) - [p["board_x_m"], p["board_y_m"]]
                self.assertAlmostEqual(np.linalg.norm(delta), 0.4)
            p = available[2]
            vals = [
                p[k] for k in ("azimuth_offset_deg", "height_m", "gap_m", "tilt_deg")
            ]
            arrays.append(np.arctanh(2 * (vals - LOW) / (HIGH - LOW) - 1))
            menus.append(available)
        latents = np.asarray(arrays)
        menus[0] = []
        menus[1] = [bank[1][1]]
        selected = project_actions(self.cells, latents, menus)
        self.assertEqual(selected[0], actions(self.cells, latents)[0])
        self.assertEqual(selected[1], bank[1][1])
        self.assertEqual(selected[2:], bank[2][2:])

    def test_camera_orbits_once_then_enters_between_tables(self):
        names = [s[0] for s in SHOTS]
        angles = []
        for i, (name, count, _) in enumerate(SHOTS):
            if name.startswith("orbit-"):
                for frame in range(count):
                    eye, aim = camera_pose(i, frame, self.cells)
                    angles.append(np.arctan2(eye[1] - aim[1], eye[0] - aim[0]))
        sweep = np.unwrap(angles)
        self.assertAlmostEqual(abs(sweep[-1] - sweep[0]), 2 * np.pi)
        self.assertLess(np.max(np.abs(np.diff(sweep))), np.deg2rad(2))
        aisle = names.index("through-aisle")
        start, _ = camera_pose(aisle, 0, self.cells)
        end, _ = camera_pose(aisle, SHOTS[aisle][1] - 1, self.cells)
        table_centers_x = origins()[:, 0] + self.scene["table"]["center_m"][0]
        self.assertEqual(start[0], end[0])
        self.assertGreater(
            np.min(abs(start[0] - table_centers_x)),
            self.scene["table"]["size_m"][0] / 2,
        )
        self.assertLess(start[1], origins()[:, 1].min())
        self.assertGreater(end[1], 0)

    def test_closeups_frame_front_release_board_bucket_and_other_rows(self):
        valid = np.ones(35, dtype=bool)
        first = choose_front(valid, "front-left")
        second = choose_front(valid, "front-right", previous=first)
        self.assertNotEqual(first, second)
        valid[:7] = False
        with self.assertRaises(ValueError):
            choose_front(valid, "front-right")
        for name, focus in [("front-left", first), ("front-right", second)]:
            self.assertLess(focus, 7)
            i = [s[0] for s in SHOTS].index(name)
            cell = self.cells[focus]
            origin = origins()[focus]
            subjects = origin + np.array(
                [
                    [*cell["board_xy_m"], 0.105],
                    [*cell["board_xy_m"], 0.908],
                    [*cell["bucket_xy_m"], 0.183],
                ]
            )
            # Include representative arm centers on all four rows behind the subject.
            backdrop = origins()[7:] + [0.67, -0.25, 0.35]
            for frame in [0, SHOTS[i][1] // 2, SHOTS[i][1] - 1]:
                eye, aim = camera_pose(i, frame, self.cells, focus)
                forward = (aim - eye) / np.linalg.norm(aim - eye)
                right = np.cross(forward, [0, 0, 1])
                right /= np.linalg.norm(right)
                up = np.cross(right, forward)
                points = np.vstack([subjects, backdrop]) - eye
                depth = points @ forward
                xy = points @ np.stack([right, up], axis=1)
                # 23 mm perspective lens, 36 x 20.25 mm aperture from the film.
                projected = xy * 23 / (depth[:, None] * [18, 10.125])
                visible = (depth > 0) & (np.max(abs(projected), axis=1) < 0.94)
                self.assertTrue(visible[:3].all())
                self.assertGreaterEqual(sum(visible[3:]), 4)


if __name__ == "__main__":
    unittest.main()
