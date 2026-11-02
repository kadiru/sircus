"""Pure geometry checks for frozen-state, right-front camera comparisons."""

import itertools
import json
import subprocess
import sys
import unittest
from pathlib import Path

import numpy as np

ISAAC_SOURCE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ISAAC_SOURCE))
from camera_options import (
    DISPLAY_SLOT,
    FOCAL_LENGTH_MM,
    SOURCE_BATCH,
    SOURCE_FOCUS,
    adjusted_camera,
    camera_options,
    preview_origins,
)
from dark_stage import origins


def project(option, points):
    eye, target = np.array(option["eye_m"]), np.array(option["target_m"])
    forward = target - eye
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, [0, 0, 1])
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    rays = np.asarray(points) - eye
    depth = rays @ forward
    uv = (
        rays
        @ np.stack([right, up]).T
        * FOCAL_LENGTH_MM
        / (depth[:, None] * [18, 10.125])
    )
    return depth, uv


class CameraOptionsTests(unittest.TestCase):
    def test_adjustment_moves_horizontally_right_keeps_aim_and_sets_world_z(self):
        original = camera_options()[5]
        saved = json.loads(json.dumps(original))
        result = adjusted_camera(original, 0.30, 2.0)
        delta = np.array(result["eye_m"]) - original["eye_m"]
        self.assertAlmostEqual(np.linalg.norm(delta[:2]), 0.30)
        forward_xy = (
            np.array(original["target_m"])[:2] - np.array(original["eye_m"])[:2]
        )
        self.assertAlmostEqual(delta[:2] @ forward_xy, 0)
        self.assertGreater(delta[0], 0)
        self.assertGreater(delta[1], 0)
        self.assertEqual(result["eye_m"][2], 2.0)
        self.assertEqual(result["target_m"], original["target_m"])
        self.assertEqual(original, saved)
        for bad in (float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                adjusted_camera(original, bad, 2.0)

    def test_selected_adjustment_retains_focused_table_and_background(self):
        option = adjusted_camera(camera_options()[5], 0.30, 2.0)
        world = preview_origins()
        envelope = list(itertools.product((-0.23, 1.57), (-0.875, 0.375), (0, 0.95)))
        depth, uv = project(option, world[SOURCE_FOCUS] + envelope)
        self.assertTrue(np.all(depth > 0.04))
        self.assertLess(np.max(abs(uv)), 0.95)
        depth, uv = project(option, world + [0.67, -0.25, 0.45])
        visible = (depth > 0.04) & np.all(abs(uv) < 0.95, axis=1)
        visible[SOURCE_FOCUS] = False
        self.assertGreaterEqual(np.count_nonzero(visible), 25)

    def test_mapping_preserves_all_cells_and_the_focused_shot_geometry(self):
        original, preview = origins(), preview_origins()
        self.assertEqual((SOURCE_FOCUS, DISPLAY_SLOT, SOURCE_BATCH), (1, 6, 6))
        self.assertEqual(preview.shape, (35, 3))
        self.assertEqual({tuple(v) for v in original}, {tuple(v) for v in preview})
        np.testing.assert_array_equal(preview[SOURCE_FOCUS], original[DISPLAY_SLOT])
        np.testing.assert_array_equal(preview[DISPLAY_SLOT], original[SOURCE_FOCUS])
        others = [i for i in range(35) if i not in (SOURCE_FOCUS, DISPLAY_SLOT)]
        np.testing.assert_array_equal(preview[others], original[others])
        # A common translation moves the board/release/bucket together; copying
        # just arm poses into the original corner case would fail this invariant.
        local_shot = np.array(
            [[0.55, 0.36, 0.70], [0.45, 0.10, 0.38], [0.155, 0.25, 0.18]]
        )
        before = original[SOURCE_FOCUS] + local_shot
        after = preview[SOURCE_FOCUS] + local_shot
        np.testing.assert_allclose(after[:, None] - after, before[:, None] - before)
        preview[:] = 0
        np.testing.assert_array_equal(origins(), original)
        self.assertEqual(len({tuple(v) for v in preview_origins()}), 35)

    def test_options_are_distinct_fixed_json_ready_and_not_shared_mutable_state(self):
        options = camera_options()
        self.assertEqual([o["id"] for o in options], list(range(1, 11)))
        self.assertEqual(len({o["name"] for o in options}), 10)
        self.assertEqual(len({tuple(o["eye_m"]) for o in options}), 10)
        self.assertEqual(json.loads(json.dumps(options)), options)
        anchor = origins()[DISPLAY_SLOT]
        for option in options:
            self.assertEqual(
                (option["focus_source"], option["focus_display"]), ("r1c2", "r1c7")
            )
            for world, offset in (
                ("eye_m", "eye_offset_m"),
                ("target_m", "target_offset_m"),
            ):
                np.testing.assert_allclose(option[world], anchor + option[offset])
                self.assertTrue(np.isfinite(option[world]).all())
        saved = camera_options()
        options[0]["eye_m"][0] = 1000
        options[0]["eye_offset_m"][0] = 1000
        np.testing.assert_equal(camera_options(), saved)

    def test_every_view_contains_foreground_envelope_and_background_farm(self):
        world = preview_origins()
        anchor = world[SOURCE_FOCUS]
        # Conservative tabletop/arm envelope plus known shot and nearby floor.
        envelope = list(itertools.product((-0.23, 1.57), (-0.875, 0.375), (0, 0.95)))
        envelope += [(0.55, 0.36, 0.70), (0.45, 0.10, 0.38), (0.155, 0.25, 0.18)]
        envelope += [(-0.8, -0.5, -0.7375), (2.4, 0.7, -0.7375)]
        for option in camera_options():
            with self.subTest(camera=option["id"]):
                depth, uv = project(option, anchor + envelope)
                self.assertTrue(np.all(depth > 0.04))
                self.assertLess(np.max(abs(uv)), 0.95)
                depth, uv = project(option, world + [0.67, -0.25, 0.45])
                visible = (depth > 0.04) & np.all(abs(uv) < 0.95, axis=1)
                visible[SOURCE_FOCUS] = False
                self.assertGreaterEqual(np.count_nonzero(visible), 15)
                for row in range(1, 5):
                    self.assertGreaterEqual(
                        np.count_nonzero(visible[row * 7 : (row + 1) * 7]), 1
                    )

    def test_import_requires_no_isaac_or_gpu_runtime(self):
        script = (
            "import sys; "
            f"sys.path.insert(0, {str(ISAAC_SOURCE)!r}); "
            "import camera_options; "
            "assert len(camera_options.camera_options()) == 10; "
            "assert not any(n.split('.')[0] in ('isaaclab', 'omni', 'pxr', 'torch') for n in sys.modules)"
        )
        subprocess.run(
            [sys.executable, "-c", script], check=True, capture_output=True, text=True
        )


if __name__ == "__main__":
    unittest.main()
