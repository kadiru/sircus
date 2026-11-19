"""Geometry-only expansion and strict native-proof gates, without GPU work."""

import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from scipy.spatial.transform import Rotation

SOURCE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOURCE))
from expanded_farm import make_spec, validate_action
from feasibility_contract import dynamic_cell
from prepare_assets import matrix
from expanded_pose_menu import (
    canonical_hash,
    cell_identity,
    endpoint_targets,
    geometry_reason,
    load_native_menu,
    make_manifest,
    outer_bounds,
    proposals,
    qualify,
)


class ExpandedPoseMenuTests(unittest.TestCase):
    def setUp(self):
        self.scene = json.loads((SOURCE / "scene.json").read_text())
        self.spec = make_spec(self.scene, 10)
        self.cells = self.spec["cells"]

    def test_grid_stays_distinct_and_proposals_vary_distance_height_gap_tilt(self):
        before = copy.deepcopy(self.cells)
        self.assertEqual(len({tuple(c["bucket_xy_m"]) for c in self.cells}), 35)
        for i in (0, 6, 14, 28, 34):
            group = proposals(self.scene, self.cells[i], 128)
            self.assertEqual(group, proposals(self.scene, self.cells[i], 128))
            self.assertEqual(len(group), 128)
            self.assertEqual(len({a["id"] for a in group}), 128)
            self.assertGreater(len({round(a["radius_m"], 3) for a in group}), 5)
            self.assertGreaterEqual(len({a["height_m"] for a in group}), 4)
            self.assertGreaterEqual(len({a["gap_m"] for a in group}), 3)
            self.assertGreaterEqual(len({a["tilt_deg"] for a in group}), 3)
            for action in group:
                validate_action(self.cells[i], action)
                self.assertIsNone(geometry_reason(self.scene, self.cells[i], action))
        self.assertEqual(self.cells, before)

    def test_targets_preserve_vertical_release_and_bucket_facing_normal(self):
        cell = self.cells[0]
        action = proposals(self.scene, cell, 1)[0]
        targets = endpoint_targets(self.scene, cell, action)
        release = targets["release_center_m"]
        np.testing.assert_allclose(
            release[:2], [action["board_x_m"], action["board_y_m"]]
        )
        self.assertAlmostEqual(
            release[2],
            action["height_m"] + action["gap_m"] + self.scene["ball"]["radius_m"],
        )
        np.testing.assert_allclose(targets["Arm10"][:3, 3], release + [0, 0, 0.095])
        direction = np.array(cell["bucket_xy_m"]) - release[:2]
        self.assertGreater(targets["board_normal_world"][:2] @ direction, 0)
        np.testing.assert_allclose(
            targets["Arm10"][:3, :3], np.diag([1, -1, -1]), atol=1e-14
        )

    def test_targets_match_exact_native_prepare_cache_keys(self):
        for cell in self.cells:
            for action in proposals(self.scene, cell, 10):
                native = dynamic_cell(cell, action)
                board = np.eye(4)
                board[:3, :3] = (
                    Rotation.from_euler(
                        "z", native["heading_deg"], degrees=True
                    ).as_matrix()
                    @ Rotation.from_euler(
                        "x", action["tilt_deg"], degrees=True
                    ).as_matrix()
                    @ Rotation.from_euler(
                        "z", self.scene["board"]["mount_yaw_rad"]
                    ).as_matrix()
                )
                board[:3, 3] = [*native["board_xy_m"], action["height_m"]]
                base = matrix(
                    [self.scene["arms"]["base_spacing_m"], 0, 0], [0, 0, np.pi]
                )
                mount = matrix(
                    [0, 0, self.scene["board"]["tool_offset_m"]],
                    [0, 0, self.scene["board"]["mount_yaw_rad"]],
                )
                expected = np.linalg.inv(base) @ board @ np.linalg.inv(mount)
                actual = endpoint_targets(self.scene, cell, action)["Arm14"]
                self.assertEqual(tuple(actual.ravel()), tuple(expected.ravel()))

    def test_link_bound_is_only_a_conservative_geometric_reject(self):
        joint = lambda xyz: (
            np.block(
                [
                    [np.eye(3), np.array(xyz)[:, None]],
                    [np.zeros((1, 3)), np.ones((1, 1))],
                ]
            ),
            np.array([0, 0, 1]),
            -4,
            4,
        )
        links = [joint([0, 0, 0.3]), joint([0.4, 0, 0]), joint([0.4, 0, 0])]
        targets = {name: np.eye(4) for name in ("Arm10", "Arm14")}
        for target in targets.values():
            target[:3, 3] = [0.5, 0, 0.3]
        self.assertTrue(outer_bounds(targets, links)["eligible"])
        targets["Arm10"][0, 3] = 1.0
        self.assertFalse(outer_bounds(targets, links)["eligible"])

    def fixture(self, qualification="native_settled_endpoints"):
        menus = [proposals(self.scene, c, 2) for c in self.cells]
        settled = dict(
            position_error_mm=0.1,
            normal_error_deg=0.01,
            joint_speed_rad_s=0.001,
            position_drift_mm=0.01,
            normal_drift_deg=0.01,
            dropper_position_error_mm=0.1,
        )
        native = [
            [
                dict(
                    action_id=a["id"],
                    action=copy.deepcopy(a),
                    accepted=True,
                    invalid=None,
                    settled=copy.deepcopy(settled),
                )
                for a in group
            ]
            for group in menus
        ]
        return make_manifest(
            self.spec, menus, native_checks=native, qualification=qualification
        )

    def load_fixture(self, value):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "menu.json"
            path.write_text(json.dumps(value))
            return load_native_menu(path, self.spec)

    def test_only_complete_native_measured_menus_can_train(self):
        good = self.fixture()
        menus, audit = self.load_fixture(good)
        self.assertEqual([len(m) for m in menus], [2] * 35)
        self.assertEqual(audit["source_fingerprint"], self.spec["fingerprint"])
        with self.assertRaises(ValueError):
            self.load_fixture(self.fixture("offline_endpoint_ik"))
        for bad in ("stale_action", "unstable", "missing_cell", "duplicate_pose"):
            value = copy.deepcopy(good)
            if bad == "stale_action":
                value["native_checks"][0][0]["action"]["gap_m"] += 0.01
            if bad == "unstable":
                value["native_checks"][0][0]["settled"]["joint_speed_rad_s"] = 0.006
            if bad == "missing_cell":
                value["menus"][0] = []
            if bad == "duplicate_pose":
                value["menus"][0][1] = dict(value["menus"][0][0], id="another-id")
                value["native_checks"][0][1].update(
                    action_id="another-id", action=value["menus"][0][1]
                )
            value["fingerprint"] = canonical_hash(
                {k: v for k, v in value.items() if k != "fingerprint"}
            )
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                self.load_fixture(value)

    def test_cpu_search_reuses_targets_and_resumes_without_repeating_candidates(self):
        # Exercise orchestration with deterministic IK stand-ins. This is not
        # native qualification, and it cannot pass load_native_menu.
        cells = copy.deepcopy(self.cells[:2])
        small = copy.deepcopy(self.spec)
        small["cells"] = cells
        small["fingerprint"] = canonical_hash(
            {k: v for k, v in small.items() if k != "fingerprint"}
        )
        pool = [proposals(self.scene, c, 4) for c in cells]
        with tempfile.TemporaryDirectory() as directory:
            assets = Path(directory)
            urdf = (
                "<robot>"
                + "".join(
                    f'<joint name="joint{i}" type="revolute"><origin xyz="0.4 0 0" rpy="0 0 0"/><axis xyz="0 0 1"/><limit lower="-4" upper="4"/></joint>'
                    for i in range(1, 7)
                )
                + "</robot>"
            )
            for name in ("dropper.urdf", "board_arm.urdf"):
                (assets / name).write_text(urdf)
            (assets / "poses.json").write_text(
                json.dumps(dict(dropper_joint_rad=[0] * 6, board_joint_rad=[0] * 6))
            )
            calls = []

            def solve(target, links, seed, normal_only):
                calls.append((target.copy(), normal_only))
                return dict(solved=True, q=[0.2] * 6, residual=0.0, stage="fixture")

            with patch("robust_ik.solve", solve):
                result = qualify(
                    self.scene,
                    cells,
                    assets,
                    pool,
                    spec=small,
                    max_checks_per_cell=2,
                    accepted_per_cell=2,
                )
                first = len(calls)
                resumed = qualify(
                    self.scene,
                    cells,
                    assets,
                    pool,
                    spec=small,
                    max_checks_per_cell=2,
                    accepted_per_cell=2,
                    previous=result,
                )
            self.assertEqual(len(calls), first)
            self.assertEqual(resumed["menus"], result["menus"])
            self.assertEqual(result["qualification"], "offline_endpoint_ik")
            self.assertTrue(all(len(m) == 2 for m in result["menus"]))


if __name__ == "__main__":
    unittest.main()
