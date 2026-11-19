"""Expanded shot decoding and camera/source identity remain separate contracts."""

import copy
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

SOURCE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOURCE))
from cylinder_farm import observation as original_observation
from cylinder_farm import specification as original_specification
from expanded_farm import (
    ACTION_DIM,
    ACTION_NAMES,
    CAMERA_SOURCE_CONTRACT,
    CAMERA_SOURCE_FINGERPRINT,
    HIGH,
    LOW,
    VERSION,
    WARM,
    actions,
    bind_camera,
    is_expanded,
    make_spec,
    observation,
    origins,
    project_actions,
    shot,
    validate_action,
    verify_camera_binding,
)
from saved_camera import INTRINSICS, RESOLUTION, SCHEMA, load_camera


def rehash(spec):
    unsigned = copy.deepcopy(spec)
    unsigned.pop("fingerprint")
    spec["fingerprint"] = hashlib.sha256(
        json.dumps(unsigned, sort_keys=True, allow_nan=False).encode()
    ).hexdigest()


class ExpandedFarmTests(unittest.TestCase):
    def setUp(self):
        self.scene = json.loads((SOURCE / "scene.json").read_text())
        self.legacy = original_specification(self.scene, 10)
        self.pinned = json.loads(CAMERA_SOURCE_CONTRACT.read_bytes())
        self.spec = make_spec(self.scene, 10)
        self.cells = self.spec["cells"]

    def test_new_learning_identity_keeps_old_geometry_and_camera_source_identity(self):
        self.assertEqual(self.pinned["fingerprint"], CAMERA_SOURCE_FINGERPRINT)
        self.assertTrue(is_expanded(self.spec))
        self.assertFalse(is_expanded(self.legacy))
        self.assertEqual(self.spec["version"], VERSION)
        self.assertNotEqual(self.spec["fingerprint"], CAMERA_SOURCE_FINGERPRINT)
        self.assertEqual(
            self.spec["camera_source_fingerprint"], CAMERA_SOURCE_FINGERPRINT
        )
        self.assertEqual(self.spec["scene"], self.scene)
        self.assertEqual(self.cells, self.pinned["cells"])
        np.testing.assert_allclose(
            [c["bucket_xy_m"] for c in self.cells],
            [c["bucket_xy_m"] for c in self.legacy["cells"]],
            rtol=0,
            atol=1e-12,
        )
        self.assertEqual(
            self.spec["base_geometry"]["source_contract_file_sha256"],
            hashlib.sha256(CAMERA_SOURCE_CONTRACT.read_bytes()).hexdigest(),
        )
        self.assertEqual(len({tuple(c["bucket_xy_m"]) for c in self.cells}), 35)
        self.assertEqual(self.spec["action_dim"], ACTION_DIM)
        self.assertEqual(self.spec["action_names"], list(ACTION_NAMES))
        self.assertNotIn("radius_m", self.spec)
        np.testing.assert_array_equal(self.spec["origins_m"], origins())
        self.assertEqual(
            self.spec["scene"]["bucket"]["height_interpretation"],
            "unresolved: overall height or interior depth",
        )
        later = make_spec(self.scene, 20)
        self.assertEqual(
            later["camera_source_fingerprint"], self.spec["camera_source_fingerprint"]
        )
        self.assertNotEqual(later["fingerprint"], self.spec["fingerprint"])
        self.assertEqual(original_specification(self.scene, 10), self.legacy)

    def test_observations_remain_the_same_five_privileged_bucket_features(self):
        for cell in self.cells:
            np.testing.assert_array_equal(observation(cell), original_observation(cell))
            self.assertEqual(len(observation(cell)), 5)

    def test_radius_and_azimuth_control_board_xy_and_allow_low_board_high_release(self):
        cell = self.cells[1]
        first = shot(cell, [0, 0.3, 0.14, 0.56, 25])
        second = shot(cell, [90, 1.1, 0.14, 0.56, 25])
        bucket = np.array(cell["bucket_xy_m"])
        a = np.array([first["board_x_m"], first["board_y_m"]]) - bucket
        b = np.array([second["board_x_m"], second["board_y_m"]]) - bucket
        self.assertAlmostEqual(np.linalg.norm(a), 0.3)
        self.assertAlmostEqual(np.linalg.norm(b), 1.1)
        np.testing.assert_allclose(b / 1.1, [-a[1] / 0.3, a[0] / 0.3], atol=1e-12)
        rim = (
            self.scene["bucket"]["assumed_interior_depth_m"]
            + self.scene["bucket"]["assumed_bottom_thickness_m"]
        )
        self.assertLess(first["height_m"], rim)
        self.assertGreater(first["height_m"] + first["gap_m"], rim)
        np.testing.assert_array_equal(
            validate_action(cell, second), [90, 1.1, 0.14, 0.56, 25]
        )

    def test_decoder_accepts_exactly_five_finite_latents_and_enforces_bounds(self):
        for fill, expected in ((-100, LOW), (100, HIGH)):
            decoded = actions(self.cells, np.full((35, 5), fill))
            for action in decoded:
                np.testing.assert_allclose([action[k] for k in ACTION_NAMES], expected)
        for value in (np.zeros((35, 4)), np.zeros((34, 5)), np.full((35, 5), np.nan)):
            with self.subTest(shape=np.shape(value)), self.assertRaises(ValueError):
                actions(self.cells, value)
        for index in range(5):
            values = WARM.copy()
            values[index] = HIGH[index] + 0.001
            with self.subTest(index=index), self.assertRaises(ValueError):
                shot(self.cells[0], values)

    def test_nearest_qualified_menu_uses_radius_dimension_and_copies_entry(self):
        cell = self.cells[2]
        near = dict(shot(cell, [0, 0.35, 0.32, 0.24, 29]), id="qualified-near")
        far = dict(shot(cell, [0, 1.1, 0.32, 0.24, 29]), id="qualified-far")
        values = np.array([far[k] for k in ACTION_NAMES])
        latent = np.arctanh(2 * (values - LOW) / (HIGH - LOW) - 1)
        menus = [[near, far]]
        selected = project_actions([cell], latent[None], menus)
        self.assertEqual(selected, [far])
        selected[0]["radius_m"] = 0
        self.assertEqual(menus[0][1]["radius_m"], 1.1)
        with self.assertRaisesRegex(ValueError, "No qualified action menu"):
            project_actions([cell], latent[None], [[]])

    def test_qualified_menu_cannot_lie_about_radius_or_bearing(self):
        cell = self.cells[1]
        qualified = shot(cell, WARM)
        bad = dict(qualified, radius_m=0.45)
        with self.assertRaisesRegex(ValueError, "radius disagrees"):
            project_actions([cell], np.zeros((1, 5)), [[bad]])
        bad = dict(
            qualified,
            board_x_m=2 * cell["bucket_xy_m"][0] - qualified["board_x_m"],
            board_y_m=2 * cell["bucket_xy_m"][1] - qualified["board_y_m"],
        )
        with self.assertRaisesRegex(ValueError, "bearing disagrees"):
            validate_action(cell, bad)
        with self.assertRaisesRegex(ValueError, "azimuth disagrees"):
            validate_action(
                cell, dict(qualified, azimuth_deg=qualified["azimuth_deg"] + 5)
            )
        with self.assertRaisesRegex(ValueError, "distance disagrees"):
            validate_action(cell, dict(qualified, distance_m=0.8))

    def test_changed_static_geometry_is_not_an_expanded_action_experiment(self):
        changed = copy.deepcopy(self.scene)
        changed["bucket"]["inner_top_diameter_m"] += 0.001
        with self.assertRaisesRegex(ValueError, "unchanged original static scene"):
            make_spec(changed, 10)
        for count in (0, -1, 2.0, True):
            with self.assertRaises(ValueError):
                make_spec(self.scene, count)

    def test_pinned_source_hash_and_ulp_boundary_are_both_checked(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "changed-source.json"
            changed = copy.deepcopy(self.pinned)
            changed["cells"][0]["bucket_xy_m"][0] += 1e-14
            path.write_text(json.dumps(changed))
            with patch("expanded_farm.CAMERA_SOURCE_CONTRACT", path):
                with self.assertRaisesRegex(ValueError, "source contract fingerprint"):
                    make_spec(self.scene, 10)
        regenerated = copy.deepcopy(self.legacy)
        regenerated["cells"][0]["bucket_xy_m"][0] += 2e-12
        with patch("expanded_farm.legacy_specification", return_value=regenerated):
            with self.assertRaisesRegex(ValueError, "beyond numerical ULPs"):
                make_spec(self.scene, 10)


class ExpandedCameraBindingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.scene = json.loads((SOURCE / "scene.json").read_text())
        self.spec = make_spec(self.scene, 10)
        self.camera = dict(
            schema=SCHEMA,
            source_fingerprint=self.spec["camera_source_fingerprint"],
            up_axis="Z",
            eye_m=[10.9706, -6.1943, 2.5406],
            target_m=[7.8307, -3.4877, 0.3904],
            resolution=RESOLUTION,
            provenance={"selected": True},
            **INTRINSICS,
        )
        self.raw = ("\n" + json.dumps(self.camera, indent=3) + "\n\n").encode()
        self.source = self.root / "original-camera.json"
        self.source.write_bytes(self.raw)

    def test_binding_preserves_exact_original_bytes_and_separates_fingerprints(self):
        target = self.root / "selected-camera.json"
        camera, binding = bind_camera(self.source, self.spec, origins(), target)
        self.assertEqual(camera, self.camera)
        self.assertEqual(self.source.read_bytes(), self.raw)
        self.assertEqual(target.read_bytes(), self.raw)
        self.assertEqual(binding["camera_sha256"], hashlib.sha256(self.raw).hexdigest())
        self.assertEqual(
            binding["camera_source_fingerprint"], self.camera["source_fingerprint"]
        )
        self.assertEqual(binding["experiment_fingerprint"], self.spec["fingerprint"])
        with self.assertRaisesRegex(ValueError, "fingerprint"):
            load_camera(target, self.spec["fingerprint"])
        (self.root / "camera-binding.json").write_text(json.dumps(binding))
        self.assertEqual(
            verify_camera_binding(self.root, self.spec, origins()), (camera, binding)
        )
        self.assertEqual(target.read_bytes(), self.raw)

    def test_geometry_bucket_site_and_display_transform_cannot_be_rebound(self):
        changed = copy.deepcopy(self.spec)
        changed["scene"]["board"]["size_m"][0] += 0.001
        rehash(changed)
        with self.assertRaisesRegex(ValueError, "static scene geometry"):
            bind_camera(self.source, changed, origins())
        changed = copy.deepcopy(self.spec)
        changed["cells"][0]["bucket_xy_m"][0] += 0.001
        rehash(changed)
        with self.assertRaisesRegex(ValueError, "35 bucket definitions"):
            bind_camera(self.source, changed, origins())
        shifted = origins()
        shifted[1, 0] += 0.01
        with self.assertRaisesRegex(ValueError, "display origins"):
            bind_camera(self.source, self.spec, shifted)

    def test_rejects_rewritten_camera_identity_stale_spec_and_replaced_binding(self):
        changed_camera = dict(self.camera, source_fingerprint=self.spec["fingerprint"])
        self.source.write_text(json.dumps(changed_camera))
        with self.assertRaisesRegex(ValueError, "source fingerprint"):
            bind_camera(self.source, self.spec, origins())
        self.source.write_bytes(self.raw)
        changed = copy.deepcopy(self.spec)
        changed["training_batches"] += 2
        with self.assertRaisesRegex(ValueError, "experiment fingerprint"):
            bind_camera(self.source, changed, origins())
        _, binding = bind_camera(
            self.source, self.spec, origins(), self.root / "selected-camera.json"
        )
        binding["experiment_fingerprint"] = "other-experiment"
        (self.root / "camera-binding.json").write_text(json.dumps(binding))
        with self.assertRaisesRegex(ValueError, "binding mismatch"):
            verify_camera_binding(self.root, self.spec, origins())

    def test_does_not_overwrite_an_existing_different_camera_artifact(self):
        target = self.root / "selected-camera.json"
        target.write_text("previous camera")
        with self.assertRaisesRegex(ValueError, "different bytes"):
            bind_camera(self.source, self.spec, origins(), target)
        self.assertEqual(target.read_text(), "previous camera")


if __name__ == "__main__":
    unittest.main()
