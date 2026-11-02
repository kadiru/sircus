"""Exact scalar/batched scoring parity, including retention and terminal edges."""

import importlib.util
import json
from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT.parent))
from sircus.mjlab.contract import FORCE_NAMES, ScoreStream

HAVE_TORCH = importlib.util.find_spec("torch") is not None
if HAVE_TORCH:
    import torch
    from sircus.mjlab.learning_score import BatchScore


@unittest.skipUnless(HAVE_TORCH, "Torch scorer parity also runs in mjlab environment")
class BatchedScoreTests(unittest.TestCase):
    def setUp(self):
        self.scene = json.loads((ROOT / "isaac/scene.json").read_text())
        b = self.scene["bucket"]
        self.top = b["assumed_bottom_thickness_m"] + b["assumed_interior_depth_m"]

    def assert_rows_equal(self, expected, actual):
        for a, b in zip(expected, actual):
            self.assertEqual(set(a), set(b))
            for key in a:
                if isinstance(a[key], float):
                    self.assertAlmostEqual(a[key], b[key], places=10, msg=key)
                else:
                    self.assertEqual(a[key], b[key], key)

    def parity(self, positions, velocities, forces, xy, normals):
        scalar = [ScoreStream(self.scene, x, n) for x, n in zip(xy, normals)]
        batch = BatchScore(self.scene, xy, normals)
        for p, v, f in zip(positions, velocities, forces):
            batch.step(p, v, f)
            for score, pi, vi, fi in zip(scalar, p, v, f):
                score.step(pi, vi, dict(zip(FORCE_NAMES, fi)))
        self.assert_rows_equal([score.result() for score in scalar], batch.rows())
        return batch.rows()

    def test_mixed_capture_rim_exit_external_multi_bounce_and_freeze(self):
        n, t = 8, 500
        xy = np.tile([0.67, -0.5], (n, 1))
        normals = np.tile([0, 0, 1], (n, 1))
        p = np.tile([0.67, -0.5, 0.08], (t, n, 1))
        v = np.zeros_like(p)
        f = np.zeros((t, n, 6))
        p[0, :, :] = [0.67, -0.1, 0.36]
        v[0, :, 2] = -1
        f[0, :, 0] = 100
        p[1, :, :] = [0.67, -0.1, 0.37]
        v[1, :, 2] = 1
        p[2, :, 2] = self.top + 0.02
        v[2, :, 2] = -1
        f[2, 1, 1] = 20  # Rim-assisted capture still counts.
        f[12, 2, 2] = 10  # Table-assisted historical success, route failure.
        f[:, 3, :] = 0  # No board contact, so no score despite being inside.
        p[123, 4, 2] = self.top - 0.02  # Partial exit resets continuous retention.
        p[5:, 5, 2] = -0.75  # Terminal fall freezes pre-entry state.
        f[5:, 5, 3] = 50
        f[20, 6, 0] = 5  # Second separated board contact episode.
        v[21, 6, 2] = 1
        # A capture freezes before this later contact/fall can alter it.
        p[450:, 0, 2] = -0.75
        f[450:, 0, 3] = 70
        # Near wall: the polygon planes, rather than a circular proxy, matter.
        p[3:, 7, 0] += 0.04
        rows = self.parity(p, v, f, xy, normals)
        self.assertTrue(rows[0]["complete_success"])
        self.assertEqual(rows[0]["route_quality"], 2)
        self.assertEqual(rows[1]["route_quality"], 1)
        self.assertTrue(rows[2]["success"])
        self.assertFalse(rows[2]["complete_success"])
        self.assertFalse(rows[3]["success"])
        self.assertTrue(rows[4]["success"])
        self.assertEqual(rows[4]["route_quality"], 1)
        self.assertFalse(rows[5]["success"])
        self.assertEqual(rows[5]["physics_steps"], 6)
        self.assertEqual(rows[6]["board_contact_episodes"], 2)
        self.assertFalse(rows[6]["single_bounce_success"])

    def test_seeded_jitter_and_differing_bucket_frames(self):
        rng = np.random.default_rng(20261102)
        n, t = 12, 400
        xy = rng.uniform([0.3, -0.6], [1, -0.2], (n, 2))
        normals = rng.normal(size=(n, 3))
        normals[:, 2] = np.abs(normals[:, 2]) + 1
        normals /= np.linalg.norm(normals, axis=1, keepdims=True)
        p = rng.normal(0, 0.007, (t, n, 3))
        p[:, :, :2] += xy
        p[:, :, 2] += 0.08
        p[:20, :, 2] += np.linspace(0.3, 0.1, 20)[:, None]
        v = rng.normal(0, 0.2, p.shape)
        v[1, :, :] = normals
        f = np.zeros((t, n, 6))
        f[0, :, 0] = rng.uniform(1, 100, n)
        f[3, ::3, 1] = 10
        f[14, ::4, 4] = 1
        p[100, ::2, 2] = self.top - 0.02
        self.parity(p, v, f, xy, normals)

    def test_input_validation(self):
        xy, normal = [[0.5, -0.4]], [[0, 0, 1]]
        with self.assertRaises(ValueError):
            BatchScore(self.scene, xy, [[0, 0, 2]])
        with self.assertRaises(ValueError):
            BatchScore(
                dict(self.scene, simulation=dict(physics_dt_s=1 / 120)), xy, normal
            )
        batch = BatchScore(self.scene, xy, normal)
        with self.assertRaises(ValueError):
            batch.step([[0, 0, 0]], [[0, 0, 0]], [[-1, 0, 0, 0, 0, 0]])
        with self.assertRaises(ValueError):
            batch.step([[float("nan"), 0, 0]], [[0, 0, 0]], [[0] * 6])
        with self.assertRaises(ValueError):
            batch.step([[0, 0, 0]], [[0, 0, 0]], [[0] * 5])

    def test_exact_retention_boundary_and_inactive_input_freeze(self):
        batch = BatchScore(self.scene, [[0.67, -0.5]], [[0, 0, 1]])
        force = torch.zeros((1, 6), dtype=torch.float64)
        force[0, 0] = 100
        batch.step([[0.67, -0.1, 0.36]], [[0, 0, -1]], force)
        force.zero_()
        batch.step([[0.67, -0.1, 0.37]], [[0, 0, 1]], force)
        batch.step([[0.67, -0.5, self.top + 0.02]], [[0, 0, -1]], force)
        for _ in range(239):
            batch.step([[0.67, -0.5, 0.08]], [[0, 0, 0]], force)
        self.assertFalse(batch.rows()[0]["success"])
        batch.step([[0.67, -0.5, 0.08]], [[0, 0, 0]], force)
        before = batch.rows()
        self.assertTrue(before[0]["success"])
        # Scalar ScoreStream returns before validating data after it finishes.
        batch.step([[float("nan")] * 3], [[float("nan")] * 3], force)
        self.assertEqual(before, batch.rows())


if __name__ == "__main__":
    unittest.main()
