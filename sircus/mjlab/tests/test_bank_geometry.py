"""Stable action slots and cached endpoint replay, without GPU work."""

import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np

from sircus.mjlab.bank_geometry import (
    FixedEndpointIK,
    audit_candidates,
    build_candidates,
    candidate_features,
    fixed_endpoints,
    outer_geometry_mask,
)
from sircus.mjlab.learning_contract import digest
from sircus.mjlab.learning_reach import feasible_mask
from sircus.mjlab.learning_search import VERSION as BANK_VERSION

ROOT = Path(__file__).resolve().parents[2]
HAVE_TORCH = importlib.util.find_spec("torch") is not None


def bank(count=2):
    result = dict(
        version=BANK_VERSION,
        templates=[
            dict(
                source_trial_id=f"shot-{i:07d}",
                source_cell_id=f"train-{i}",
                relative_board_xy_m=[0, 0.4 + i * 0.005],
                height_m=0.4,
                tilt_deg=21.0,
                gap_m=0.13,
                qualification="train_confirmation_3_of_3",
            )
            for i in range(count)
        ],
    )
    result["sha256"] = digest(result)
    return result


class BankGeometryTests(unittest.TestCase):
    def setUp(self):
        self.scene = json.loads((ROOT / "isaac" / "scene.json").read_text())

    def test_every_rotation_and_template_keeps_its_slot_across_layouts(self):
        values, metadata = build_candidates(
            bank(16), [[0.67, -0.5], [0.1, 0.25]], self.scene
        )
        self.assertEqual(values.shape, (2, 256, 5))
        self.assertEqual(len(metadata), 256)
        self.assertEqual(len({r["candidate_id"] for r in metadata}), 256)
        self.assertEqual(metadata[0]["rotation_index"], 0)
        self.assertEqual(metadata[16]["rotation_index"], 1)
        self.assertEqual(metadata[255]["template_index"], 15)
        np.testing.assert_allclose(
            values[1, :, :2] - values[0, :, :2], [[-0.57, 0.75]] * 256
        )
        # Even positions outside action bounds remain present with stable IDs.
        self.assertGreater(
            ((values[:, :, 1] > 0.25) | (values[:, :, 0] < 0.32)).sum(), 0
        )

    def test_quarter_turn_and_geometry_only_features_are_correct(self):
        values, metadata = build_candidates(bank(1), [[0.67, -0.5]], self.scene)
        np.testing.assert_allclose(values[0, 0, :2], [0.67, -0.1], atol=1e-14)
        np.testing.assert_allclose(values[0, 4, :2], [0.27, -0.5], atol=1e-14)
        self.assertEqual(metadata[4]["rotation_deg"], 90)
        features = candidate_features([[0.67, -0.5]], values)
        self.assertEqual(features.shape, (1, 16, 12))
        self.assertTrue(np.isfinite(features).all())

    def test_outer_bound_retains_a_pose_beyond_the_old_conservative_prior(self):
        xyz = np.array([[0.6, -0.1, 0.4]])
        gap = 0.1415 + np.sqrt(0.83**2 - 0.6**2 - 0.1**2) - 0.4
        values = np.array([[[0.6, -0.1, 0.4, 25, gap]]])
        within, reach = outer_geometry_mask(self.scene, [[0.6, -0.5]], values)
        self.assertTrue(within[0, 0] and reach[0, 0])
        self.assertFalse(feasible_mask(xyz, 25, [[0.6, -0.5]], gap, self.scene)[0])

    def test_corrupt_bank_and_wrong_bucket_geometry_are_rejected(self):
        broken = bank()
        broken["templates"][0]["gap_m"] += 0.01
        with self.assertRaises(ValueError):
            build_candidates(broken, [[0.67, -0.5]], self.scene)
        self.scene["bucket"]["segments"] = 32
        with self.assertRaises(ValueError):
            build_candidates(bank(), [[0.67, -0.5]], self.scene)


@unittest.skipUnless(HAVE_TORCH, "Torch CPU provider test in pinned mjlab environment")
class FixedEndpointTests(unittest.TestCase):
    def setUp(self):
        self.scene = json.loads((ROOT / "isaac" / "scene.json").read_text())
        self.xy = np.array([[0.67, -0.5]])
        self.values = np.array([[0.67, -0.1, 0.4, 21, 0.13]])
        self.q = np.array([[0, -0.5, -1.5, 0, 0.5, 0]])

    def test_provider_preserves_cached_joint_branch_when_padding(self):
        provider = FixedEndpointIK(
            self.scene, self.xy, self.values, self.q, self.q, [True]
        )
        result = provider.solve(
            np.repeat(self.xy, 3, axis=0),
            np.repeat(self.values[:, :3], 3, axis=0),
            [21] * 3,
            [0.13] * 3,
        )
        np.testing.assert_array_equal(
            result["q10"].numpy(), np.repeat(self.q, 3, axis=0)
        )
        self.assertTrue(result["valid"].all())
        self.assertEqual(result["targets"]["release_center_m"].shape, (3, 3))

    def test_provider_rejects_changed_valid_geometry(self):
        provider = FixedEndpointIK(
            self.scene, self.xy, self.values, self.q, self.q, [True]
        )
        with self.assertRaises(RuntimeError):
            provider.solve(self.xy, self.values[:, :3] + [0.01, 0, 0], [21], [0.13])

    def test_invalid_provider_slot_stays_invalid_after_sanitization(self):
        provider = FixedEndpointIK(
            self.scene, self.xy, self.values, self.q, self.q, [False]
        )
        result = provider.solve(self.xy, [[0.65, -0.04, 0.36]], [25], [0.24])
        self.assertFalse(result["valid"][0])

    def test_context_restores_original_solver_after_error(self):
        original = object()
        env = SimpleNamespace(ik=original, scene=self.scene, device="cpu")
        with self.assertRaises(RuntimeError):
            with fixed_endpoints(env, self.xy, self.values, self.q, self.q, [True]):
                self.assertIsInstance(env.ik, FixedEndpointIK)
                raise RuntimeError("test cancellation")
        self.assertIs(env.ik, original)


@unittest.skipUnless(
    HAVE_TORCH and os.environ.get("SIRCUS_IK_ASSETS"),
    "Optional generated native assets required",
)
class NativeBankTests(unittest.TestCase):
    def test_known_control_keeps_fixed_slot_and_passes_native_gates(self):
        scene = json.loads((ROOT / "isaac" / "scene.json").read_text())
        xy = np.array([[0.67, -0.5]])
        values, metadata = build_candidates(bank(1), xy, scene)
        mask, details = audit_candidates(
            scene, Path(os.environ["SIRCUS_IK_ASSETS"]), xy, values, chunk_size=32
        )
        self.assertEqual(mask.shape, (1, 16))
        self.assertTrue(mask[0, 0], details["summary"])
        self.assertEqual(details["q10"].shape, (1, 16, 6))
        self.assertLess(details["position_error_m"][0, 0].max(), 0.002)
        self.assertEqual(details["summary"]["total_slots"], 16)


if __name__ == "__main__":
    unittest.main()
