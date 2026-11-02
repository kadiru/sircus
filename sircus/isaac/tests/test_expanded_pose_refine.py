"""Hold refinement uses measured geometry proof and the original IK branch."""

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
from expanded_farm import make_spec
from expanded_pose_menu import _action, canonical_hash, endpoint_targets, make_manifest
from expanded_pose_refine import held_seed, refine, refinement_actions


class HoldRefinementTests(unittest.TestCase):
    def setUp(self):
        self.scene = json.loads((SOURCE / "scene.json").read_text())
        self.spec = make_spec(self.scene, 10)
        self.cell = self.spec["cells"][5]
        self.seed = _action(
            self.cell,
            0.6933219218391121,
            0.07866829100703607,
            0.30,
            0.22,
            52.0,
            "expanded-0163",
        )

    def test_gap_first_variants_hold_board_exactly_and_are_distinct(self):
        actions = refinement_actions(self.scene, self.cell, self.seed)
        self.assertGreater(len(actions), 12)
        fixed = endpoint_targets(self.scene, self.cell, self.seed)["Arm14"]
        gaps = []
        for action in actions[:6]:
            np.testing.assert_array_equal(
                endpoint_targets(self.scene, self.cell, action)["Arm14"], fixed
            )
            gaps.append(action["gap_m"])
        np.testing.assert_allclose(gaps, [0.21, 0.23, 0.20, 0.24, 0.18, 0.26])
        self.assertEqual(len({a["id"] for a in actions}), len(actions))

    def fixture(self):
        menus = [
            [_action(c, 0.66, -0.10, 0.30, 0.22, 28.0, "seed")]
            for c in self.spec["cells"]
        ]
        menus[5] = [self.seed]
        fits = [
            [
                dict(
                    Arm10=dict(solved=True, q=[0.1] * 6, residual=0),
                    Arm14=dict(solved=True, q=[0.2] * 6, residual=0),
                )
            ]
            for _ in menus
        ]
        measured = dict(
            position_error_mm=0.1,
            normal_error_deg=0.01,
            joint_speed_rad_s=0.001,
            position_drift_mm=0.01,
            normal_drift_deg=0.01,
            dropper_position_error_mm=0.1,
        )
        proofs = [
            [
                dict(
                    action_id=a["id"],
                    action=a,
                    accepted=True,
                    invalid=None,
                    settled=measured,
                )
            ]
            for group in menus
            for a in group
        ]
        native = make_manifest(
            self.spec,
            menus,
            fits=fits,
            native_checks=proofs,
            qualification="native_settled_endpoints",
        )
        offline = make_manifest(self.spec, menus, fits=fits)
        return offline, native

    def test_rejects_stale_or_unsteady_seed_proof(self):
        _, native = self.fixture()
        self.assertEqual(held_seed(native, 5)[0], self.seed)
        bad = copy.deepcopy(native)
        bad["native_checks"][5][0]["settled"]["position_error_mm"] = 3
        with self.assertRaises(ValueError):
            held_seed(bad, 5)
        bad = copy.deepcopy(native)
        bad["native_checks"][5][0]["action"] = dict(self.seed, gap_m=0.3)
        with self.assertRaises(ValueError):
            held_seed(bad, 5)

    def test_board_first_local_order_stays_small_and_interleaves_gap(self):
        actions = refinement_actions(
            self.scene, self.cell, self.seed, local_only=True, board_first=True
        )
        self.assertNotEqual(actions[0]["height_m"], self.seed["height_m"])
        self.assertEqual(actions[0]["gap_m"], self.seed["gap_m"])
        self.assertEqual(actions[2]["height_m"], self.seed["height_m"])
        self.assertNotEqual(actions[2]["gap_m"], self.seed["gap_m"])
        for action in actions:
            for key, maximum in (
                ("board_x_m", 0.01),
                ("board_y_m", 0.01),
                ("height_m", 0.02),
                ("gap_m", 0.02),
                ("tilt_deg", 2.0),
            ):
                self.assertLessEqual(abs(action[key] - self.seed[key]), maximum + 1e-12)

    def test_refinement_never_warmstarts_from_last_fit_and_preserves_other_cells(self):
        offline, native = self.fixture()
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            for name, value in (("offline", offline), ("native", native)):
                (base / (name + ".json")).write_text(json.dumps(value))
            urdf = (
                "<robot>"
                + "".join(
                    f'<joint name="joint{i}" type="revolute"><origin xyz=".4 0 0" rpy="0 0 0"/><axis xyz="0 0 1"/><limit lower="-4" upper="4"/></joint>'
                    for i in range(1, 7)
                )
                + "</robot>"
            )
            for name in ("dropper.urdf", "board_arm.urdf"):
                (base / name).write_text(urdf)
            seen = []

            def solve(target, links, seed, normal_only):
                seen.append((list(seed), normal_only))
                return dict(solved=True, q=[9] * 6, residual=0)

            with patch("robust_ik.solve", solve), patch(
                "expanded_pose_native.verified_fit",
                lambda target, record, links, normal: record,
            ):
                result = refine(
                    base / "offline.json", base / "native.json", base, [5], append=12
                )
        self.assertEqual(result["qualification"], "offline_endpoint_ik")
        self.assertEqual(len(result["menus"][5]), 13)
        for index in range(35):
            if index != 5:
                self.assertEqual(result["menus"][index], native["menus"][index])
        self.assertTrue(seen)
        for seed, normal in seen:
            self.assertEqual(seed, [0.2] * 6 if normal else [0.1] * 6)
        self.assertEqual(result["refinement"][0]["seed_action"], self.seed)
        self.assertEqual(
            canonical_hash({k: v for k, v in result.items() if k != "fingerprint"}),
            result["fingerprint"],
        )

    def test_local_only_discards_old_branches_and_rejects_new_joint_jump(self):
        offline, native = self.fixture()
        old = dict(self.seed, id="old-branch", gap_m=0.25)
        native["menus"][5].append(old)
        native["fits"][5].append(
            dict(
                Arm10=dict(solved=True, q=[1.5] * 6),
                Arm14=dict(solved=True, q=[1.6] * 6),
            )
        )
        native["fingerprint"] = canonical_hash(
            {k: v for k, v in native.items() if k != "fingerprint"}
        )
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            for name, value in (("offline", offline), ("native", native)):
                (base / (name + ".json")).write_text(json.dumps(value))
            urdf = (
                "<robot>"
                + "".join(
                    f'<joint name="joint{i}" type="revolute"><origin xyz=".4 0 0" rpy="0 0 0"/><axis xyz="0 0 1"/><limit lower="-4" upper="4"/></joint>'
                    for i in range(1, 7)
                )
                + "</robot>"
            )
            for name in ("dropper.urdf", "board_arm.urdf"):
                (base / name).write_text(urdf)
            board_calls = 0

            def solve(target, links, seed, normal_only):
                nonlocal board_calls
                q = np.array(seed, dtype=float)
                if normal_only:
                    board_calls += 1
                    q[0] += 0.60 if board_calls == 1 else 0.10
                else:
                    q[0] += 0.05
                return dict(solved=True, q=q.tolist(), residual=0)

            with patch("robust_ik.solve", solve), patch(
                "expanded_pose_native.verified_fit",
                lambda target, record, links, normal: record,
            ):
                result = refine(
                    base / "offline.json",
                    base / "native.json",
                    base,
                    [5],
                    append=2,
                    max_checks=3,
                    local_only=True,
                    max_joint_delta=0.35,
                    board_first=True,
                )
                legacy = refine(
                    base / "offline.json",
                    base / "native.json",
                    base,
                    [5],
                    append=1,
                    max_checks=2,
                )
        self.assertEqual(len(result["menus"][5]), 3)
        self.assertNotIn("old-branch", [a["id"] for a in result["menus"][5]])
        self.assertIn("old-branch", [a["id"] for a in legacy["menus"][5]])
        self.assertTrue(result["local_only"])
        self.assertEqual(result["max_joint_delta_rad"], 0.35)
        self.assertIn(
            "joint_branch_delta_Arm14", [r["reason"] for r in result["checks"][5]]
        )
        for fit in result["fits"][5]:
            for name, initial in (("Arm10", 0.1), ("Arm14", 0.2)):
                self.assertLessEqual(
                    float(np.max(np.abs(np.array(fit[name]["q"]) - initial))), 0.35
                )
        for index in range(35):
            if index != 5:
                self.assertEqual(result["menus"][index], native["menus"][index])


if __name__ == "__main__":
    unittest.main()
