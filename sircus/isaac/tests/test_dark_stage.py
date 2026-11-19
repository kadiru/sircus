"""Layout, visibility and footprint of the compact, dark farm floor."""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dark_stage import (
    COL_PITCH_M,
    FLOOR_RGB,
    FLOOR_Z_M,
    ROW_PITCH_M,
    SEAM_RGB,
    SEAM_WIDTH_M,
    STAGE_HALF_M,
    floor_mesh,
    origins,
)


class DarkStageTests(unittest.TestCase):
    def test_origins_are_tighter_with_clear_table_aisles(self):
        cells = origins().reshape(5, 7, 3)
        np.testing.assert_array_equal(cells[2, 3], [0, 0, 0])
        np.testing.assert_allclose(np.diff(cells[:, :, 0], axis=1), 2.5)
        np.testing.assert_allclose(np.diff(cells[:, :, 1], axis=0), 2.1)
        self.assertAlmostEqual(COL_PITCH_M - 1.8, 0.70)
        self.assertAlmostEqual(ROW_PITCH_M - 1.25, 0.85)

    def test_single_continuous_surface_seams_only_in_aisles(self):
        xy = (0.67, -0.25)
        points, floor, seams = floor_mesh(xy)
        points = np.array(points)
        self.assertTrue((points[:, 2] == FLOOR_Z_M).all())
        total_area = 0
        for faces in (floor, seams):
            quads = points[np.reshape(faces, (-1, 4))]
            widths = quads[:, 1, 0] - quads[:, 0, 0]
            heights = quads[:, 2, 1] - quads[:, 1, 1]
            self.assertTrue((widths > 0).all() and (heights > 0).all())
            total_area += (widths * heights).sum()
            if faces is seams:
                self.assertTrue(np.allclose(np.minimum(widths, heights), SEAM_WIDTH_M))
                centers = quads.mean(axis=1)[:, :2] - xy
                scaled = centers / [COL_PITCH_M, ROW_PITCH_M]
                self.assertTrue(
                    np.isclose(np.abs(scaled - np.round(scaled)), 0.5).any(axis=1).all()
                )
        self.assertAlmostEqual(total_area, (2 * STAGE_HALF_M) ** 2)
        self.assertLess(max(FLOOR_RGB), min(SEAM_RGB))
        self.assertLess(max(SEAM_RGB), 0.025)
