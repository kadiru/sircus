"""Broad shot coverage and analytic-prior correctness, without GPU or hardware."""

import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

SOURCE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOURCE))
from expanded_farm import make_spec, validate_action
from expanded_pose_menu import _action, canonical_hash, geometry_reason as old_reason
from expanded_pose_menu import make_manifest
from feasibility_contract import proposed_range
from shot_sweep import (
    VERSION,
    analytic_ballistics,
    ballistic_gap,
    board_geometry_diagnostics,
    fitted_geometry_reason,
    filter_manifest,
    geometry_reason,
    proposals,
    qualify,
    read_seed_manifest,
    workspace_bounds,
)


class ShotSweepTests(unittest.TestCase):
    def setUp(self):
        self.scene = json.loads((SOURCE / "scene.json").read_text())
        self.spec = make_spec(self.scene, 10)

    def test_inverse_model_roundtrips_existing_thickness_aware_descending_range(self):
        checked = 0
        for height in (0.14, 0.26, 0.42, 0.54):
            for tilt in (12, 20, 28, 36, 44, 52):
                for radius in (0.28, 0.45, 0.70, 1.1):
                    gap = ballistic_gap(self.scene, height, radius, tilt)
                    if gap is None:
                        continue
                    self.assertAlmostEqual(
                        proposed_range(self.scene, height, gap, tilt), radius, places=11
                    )
                    # Gap refers to ball bottom relative to board center, not
                    # fall distance; radius and board thickness matter.
                    impact = height + (0.0375 + 0.0025) / np.cos(np.deg2rad(tilt))
                    self.assertGreater(height + 0.0375 + gap, impact)
                    checked += 1
        self.assertGreater(checked, 50)

    def test_inverse_prior_does_not_assign_impossible_or_nonfinite_solutions(self):
        for values in ((0.14, 0.8, 52), (0.3, 0, 28), (0.3, 0.4, 0), (np.nan, 0.4, 28)):
            self.assertIsNone(ballistic_gap(self.scene, *values))
        self.assertIsNone(ballistic_gap(self.scene, 0.3, 0.4, 28, restitution=1.1))

    def test_full_table_proposals_extend_old_box_without_changing_scene(self):
        before = copy.deepcopy(self.spec)
        cell = self.spec["cells"][30]
        action = _action(cell, 0.7, -0.65, 0.3, 0.2, 28, "outside-old-box")
        self.assertEqual(old_reason(self.scene, cell, action), "proposal_bounds")
        self.assertIsNone(geometry_reason(self.scene, cell, action, "table"))
        overhang = _action(cell, 0.7, -0.86, 0.3, 0.2, 28, "overhang")
        self.assertIsNone(geometry_reason(self.scene, cell, overhang, "table"))
        outside = _action(cell, 0.7, -0.87, 0.3, 0.2, 28, "outside-center")
        self.assertEqual(
            geometry_reason(self.scene, cell, outside, "table"),
            "board_center_workspace",
        )
        self.assertEqual(self.spec, before)
        x, y = workspace_bounds(self.scene, "table")
        np.testing.assert_allclose(x, [-0.23, 1.57])
        np.testing.assert_allclose(y, [-0.875, 0.375])

    def test_low_discrepancy_proposals_are_reproducible_and_broad_in_all_dimensions(
        self,
    ):
        original = copy.deepcopy(self.spec["cells"])
        for index in (0, 6, 14, 28, 34):
            cell = self.spec["cells"][index]
            group = proposals(self.scene, cell, 512, workspace="table")
            self.assertEqual(group, proposals(self.scene, cell, 512, workspace="table"))
            self.assertEqual(len(group), 512)
            self.assertEqual(len({a["id"] for a in group}), 512)
            for name, minimum in (
                ("height_m", 0.35),
                ("gap_m", 0.40),
                ("tilt_deg", 30),
                ("radius_m", 0.45),
            ):
                values = [a[name] for a in group]
                self.assertGreater(max(values) - min(values), minimum)
            self.assertTrue(
                any(old_reason(self.scene, cell, a) == "proposal_bounds" for a in group)
            )
            for action in group:
                self.assertIsNone(geometry_reason(self.scene, cell, action, "table"))
                validate_action(cell, action)
            self.assertNotEqual(
                group, proposals(self.scene, cell, 512, seed=123, workspace="table")
            )
        self.assertEqual(self.spec["cells"], original)

    def test_diagnostics_remain_approximate_without_success_labels(self):
        action = proposals(self.scene, self.spec["cells"][1], 1)[0]
        value = analytic_ballistics(self.scene, action)
        self.assertNotIn("success", value)
        self.assertIn("Proposal-only", value["scope"])
        self.assertTrue(np.isfinite(value["predicted_range_m"]))

    def test_actual_fk_overhang_is_diagnostic_but_center_and_bottom_are_gates(self):
        # Choose a board close to the positive-Y edge. Rotating its in-plane
        # rectangle by90degrees leaves the normal unchanged but adds50mm reach.
        from prepare_assets import matrix

        scene = copy.deepcopy(self.scene)
        base = matrix([scene["arms"]["base_spacing_m"], 0, 0], [0, 0, np.pi])
        mount = matrix(
            [0, 0, scene["board"]["tool_offset_m"]],
            [0, 0, scene["board"]["mount_yaw_rad"]],
        )
        diagnostics = []
        for yaw in (np.pi / 2, 0):
            board = matrix([0.67, 0.24, 0.3], [0, 0, yaw])
            actual_flange = np.linalg.inv(base) @ board @ np.linalg.inv(mount)
            with patch("shot_sweep.fk", return_value=actual_flange):
                self.assertEqual(
                    fitted_geometry_reason(scene, dict(q=[0] * 6), [], "table"),
                    None,
                )
                diagnostics.append(
                    board_geometry_diagnostics(scene, dict(q=[0] * 6), [])
                )
        self.assertEqual(diagnostics[0]["overhang_xy_m"][1][1], 0)
        self.assertGreater(diagnostics[1]["overhang_xy_m"][1][1], 0)
        for xyz, reason in (
            ([0.67, 0.37, 0.3], "actual_board_center_workspace"),
            ([0.67, 0, 0.01], "actual_board_table_clearance"),
        ):
            board = matrix(xyz, [0, 0, 0])
            actual_flange = np.linalg.inv(base) @ board @ np.linalg.inv(mount)
            with patch("shot_sweep.fk", return_value=actual_flange):
                self.assertEqual(
                    fitted_geometry_reason(scene, dict(q=[0] * 6), [], "table"), reason
                )
        board = matrix([0.671, 0, 0.3], [0, 0, 0])
        actual_flange = np.linalg.inv(base) @ board @ np.linalg.inv(mount)
        with patch("shot_sweep.fk", return_value=actual_flange):
            self.assertEqual(
                fitted_geometry_reason(
                    scene,
                    dict(q=[0] * 6),
                    [],
                    "table",
                    dict(board_x_m=0.67, board_y_m=0, height_m=0.3),
                ),
                "actual_board_center_error",
            )

    def test_inverse_prior_samples_three_rim_height_planes_without_changing_physics(
        self,
    ):
        original = copy.deepcopy(self.scene)
        gaps = [
            ballistic_gap(self.scene, 0.3, 0.4, 28, rim_offset_r=x) for x in (-1, 0, 1)
        ]
        self.assertTrue(all(g is not None for g in gaps))
        self.assertLess(gaps[0], gaps[1])
        self.assertLess(gaps[1], gaps[2])
        self.assertEqual(self.scene, original)

    def fixture_assets(self, assets):
        urdf = (
            "<robot>"
            + "".join(
                f'<joint name="joint{i}" type="revolute"><origin xyz="0.4 0 0" rpy="0 0 0"/><axis xyz="0 0 1"/><limit lower="-6" upper="6"/></joint>'
                for i in range(1, 7)
            )
            + "</robot>"
        )
        for name in ("dropper.urdf", "board_arm.urdf"):
            (assets / name).write_text(urdf)
        (assets / "poses.json").write_text(
            json.dumps(dict(dropper_joint_rad=[0] * 6, board_joint_rad=[0] * 6))
        )

    def test_ik_search_is_bounded_keeps_controls_and_requires_native_followup(self):
        menus = [proposals(self.scene, cell, 4) for cell in self.spec["cells"]]
        fit = {
            name: dict(q=[0] * 6, solved=True, residual=0.0)
            for name in ("Arm10", "Arm14")
        }
        seeded = make_manifest(
            self.spec,
            menus,
            fits=[[copy.deepcopy(fit) for _ in group] for group in menus],
        )
        with tempfile.TemporaryDirectory() as directory:
            assets = Path(directory)
            self.fixture_assets(assets)
            native = assets / "seed.json"
            native.write_text(json.dumps(seeded))
            calls = []

            def solve(target, links, initial, normal_only):
                calls.append((target.copy(), normal_only))
                return dict(q=[0] * 6, solved=True, residual=0.0)

            # Stand-ins check orchestration only, not correctness of native IK.
            with patch("robust_ik.solve", solve), patch(
                "expanded_pose_native.verified_fit",
                side_effect=lambda target, record, *a: record,
            ), patch("shot_sweep.fitted_geometry_reason", return_value=None):
                result = qualify(
                    self.spec,
                    assets,
                    seed_manifest=native,
                    cell_indices=[0],
                    max_checks_per_cell=3,
                    accepted_per_cell=5,
                    controls=1,
                    candidates_per_cell=32,
                )
            self.assertEqual(len(result["menus"][0]), 4)
            self.assertEqual(result["menus"][0][0], menus[0][0])
            self.assertEqual(result["menus"][1:], menus[1:])
            self.assertEqual(result["summary"]["ik_checks_per_cell"][0], 3)
            self.assertLessEqual(len(calls), 6)
            self.assertEqual(result["qualification"], "offline_endpoint_ik")
            self.assertEqual(result["search_proposal_version"], VERSION)
            self.assertEqual(result["proposal_protocol"]["workspace"], "table")
            self.assertEqual(result["source_fingerprint"], self.spec["fingerprint"])
            self.assertEqual(
                result["fingerprint"],
                canonical_hash({k: v for k, v in result.items() if k != "fingerprint"}),
            )

    def test_sealed_seed_identity_rejects_changed_scene_or_missing_fits(self):
        value = make_manifest(self.spec, [[] for _ in range(35)])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "menu.json"
            path.write_text(json.dumps(value))
            self.assertEqual(read_seed_manifest(path, self.spec)[0], value)
            changed = copy.deepcopy(value)
            changed["spec"]["cells"][0]["bucket_xy_m"][0] += 0.01
            changed["fingerprint"] = canonical_hash(
                {k: v for k, v in changed.items() if k != "fingerprint"}
            )
            path.write_text(json.dumps(changed))
            with self.assertRaises(ValueError):
                read_seed_manifest(path, self.spec)
            with self.assertRaises(ValueError):
                workspace_bounds(self.scene, "unbounded")

    def test_frozen_bank_filter_binds_input_and_does_not_restore_rejected_controls(
        self,
    ):
        menus = [proposals(self.scene, cell, 2) for cell in self.spec["cells"]]
        fit = {
            name: dict(q=[0] * 6, solved=True, residual=0.0)
            for name in ("Arm10", "Arm14")
        }
        source = make_manifest(
            self.spec,
            menus,
            fits=[[copy.deepcopy(fit) for _ in group] for group in menus],
        )
        original = copy.deepcopy(source)
        with tempfile.TemporaryDirectory() as directory:
            self.fixture_assets(Path(directory))
            with patch(
                "shot_sweep.check_fitted_geometry",
                side_effect=lambda scene, cell, action, *args: (
                    "actual_board_table_clearance" if cell["index"] == 0 else None
                ),
            ):
                result = filter_manifest(source, Path(directory))
        self.assertEqual(result["menus"][0], [])
        self.assertEqual(result["fits"][0], [])
        self.assertEqual(result["menus"][1:], menus[1:])
        self.assertEqual(
            result["filter_protocol"]["source_manifest_fingerprint"],
            source["fingerprint"],
        )
        self.assertEqual(result["filter_protocol"]["source_counts"], [2] * 35)
        self.assertEqual(result["filter_protocol"]["retained_counts"], [0] + [2] * 34)
        self.assertEqual(result["qualification"], "offline_endpoint_ik")
        self.assertEqual(source, original)
        bad = copy.deepcopy(source)
        bad["menus"][0][0]["gap_m"] += 0.01
        with self.assertRaises(ValueError):
            filter_manifest(bad, Path("missing"))


if __name__ == "__main__":
    unittest.main()
