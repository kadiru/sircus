import json
import math
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from geometry import WEDGE_FACES, board_world_transform, bucket_wedge, depth_to_points


class GeometryTest(unittest.TestCase):
    def setUp(self):
        self.c = json.loads(
            (Path(__file__).resolve().parents[1] / "scene.json").read_text()
        )

    def test_measured_dimensions_preserved(self):
        b = self.c["bucket"]
        self.assertEqual(
            [
                b[k]
                for k in (
                    "inner_top_diameter_m",
                    "outer_top_diameter_m",
                    "inner_bottom_diameter_m",
                    "outer_bottom_diameter_m",
                )
            ],
            [0.177, 0.2, 0.15, 0.158],
        )
        self.assertIn("unresolved", b["height_interpretation"])
        self.assertEqual(self.c["board"]["size_m"], [0.2, 0.3, 0.005])

    def test_wall_ring_is_closed_but_top_not_capped(self):
        b = self.c["bucket"]
        for i in range(b["segments"]):
            w = bucket_wedge(b, i)
            nxt = bucket_wedge(b, (i + 1) % b["segments"])
            for j in range(4):
                for p, q in zip(w[j * 2 + 1], nxt[j * 2]):
                    self.assertAlmostEqual(p, q, places=12)
            for j in (4, 5):
                self.assertAlmostEqual(math.hypot(*w[j][:2]), 0.0885)
            self.assertEqual(len(WEDGE_FACES), 6)
        self.assertGreater(
            min(math.hypot(x, y) for x, y, z in bucket_wedge(b, 0)), 0.07
        )

    def test_landscape_board_normal_faces_bucket_negative_y(self):
        import numpy as np

        transform = board_world_transform(self.c["board"])
        normal = transform[:3, 2]
        np.testing.assert_allclose(
            normal,
            [0, -math.sin(math.radians(25)), math.cos(math.radians(25))],
            atol=1e-12,
        )
        # Long (300 mm) local-Y edge lies along world X in landscape mode.
        np.testing.assert_allclose(abs(transform[:3, 1]), [1, 0, 0], atol=1e-12)

    def test_axial_depth_optical_frame_and_missing_pixels(self):
        import numpy as np

        d = np.array([[1.0, 2.0], [float("nan"), 0.0]])
        k = np.array([[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 1.0]])
        pts, mask = depth_to_points(d, k)
        np.testing.assert_allclose(pts, [[0, 0, 1], [1, 0, 2]])
        self.assertEqual(int(mask.sum()), 2)


if __name__ == "__main__":
    unittest.main()
