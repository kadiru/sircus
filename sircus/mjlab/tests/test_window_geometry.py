"""Fixed-board endpoint protocol tests. Mocks provide no dynamics evidence."""

import copy
import json
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from sircus.mjlab.admission import REFERENCE_BUCKET, REFERENCE_VALUES
from sircus.mjlab.expand_geometry import WIDE_BOUNDS
from sircus.mjlab.learning_ik import shot_targets
from sircus.mjlab.stable_ik import target_digest
from sircus.mjlab.window_geometry import BOUNDS, WindowGeometry


class MockArm:
    def __init__(self, role):
        self.role, self.calls, self.valid = role, [], True

    def solve_candidates(self, targets, initial, **kwargs):
        self.calls.append((targets.copy(), copy.deepcopy(kwargs)))
        known = np.asarray(kwargs["known"])
        keep = np.asarray(kwargs["preserve_known"])
        q = np.empty((len(targets), 3, 6))
        for i, target in enumerate(targets):
            # A deterministic stand-in for refinement of the fixed hint. Every
            # non-nominal target changes it; no global/query-order RNG is used.
            q[i, 0] = known if keep[i] else known + .01 + abs(target[2, 3]) * .01
            q[i, 1] = known + 1
            q[i, 2] = known + 2
        return dict(q=q, valid=np.full((len(targets), 3), self.valid),
                    residual=np.zeros((len(targets), 3)),
                    query_sha256=[target_digest(t, role=self.role) for t in targets])


class MockGate:
    def __init__(self, reject=None):
        self.calls, self.reject = [], reject

    def check(self, q10, q14, bucket, values, targets):
        self.calls.append(dict(q10=q10.copy(), q14=q14.copy(), values=values.copy(),
                               bucket=bucket.copy(), targets=copy.deepcopy(targets)))
        failed = self.reject(q10, q14, values) if self.reject else False
        return dict(reason=7 if failed else 0, pairs=[],
                    minimum_clearance_lower_bound_m=.006,
                    strict_geometry=dict(passed=not failed))


class WindowGeometryTests(unittest.TestCase):
    def scene(self):
        return dict(arms=dict(base_spacing_m=1.33731, board_base_yaw_rad=np.pi),
                    ball=dict(radius_m=.0375),
                    board=dict(tool_offset_m=.1, mount_yaw_rad=np.pi / 2))

    def anchor(self):
        return dict(values=list(REFERENCE_VALUES), bucket_xy_m=list(REFERENCE_BUCKET),
                    q10=[.10, .20, .30, .40, .50, .60],
                    q14=[.15, .25, .35, .45, .55, .65])

    def setup(self, *, reject=None, board_check=None, anchor=None):
        ik = SimpleNamespace(board=MockArm("board"), dropper=MockArm("dropper"))
        gate = MockGate(reject)
        geometry = WindowGeometry(self.scene(), "/not-native-assets", anchor or self.anchor(),
            ik=ik, gate=gate, initials=dict(board_joint_rad=[0] * 6, dropper_joint_rad=[0] * 6),
            board_checker=board_check or (lambda q, target: dict(passed=True, minimum_clearance_m=.007)),
            pose_checker=lambda *args: {str(role): dict(position_m=0., rotation_rad=0.) for role in (10, 14)})
        return geometry, ik, gate

    def context(self, name="anchor", height=None, tilt=None):
        return dict(id=name, board_height_m=REFERENCE_VALUES[2] if height is None else height,
                    tilt_deg=REFERENCE_VALUES[3] if tilt is None else tilt)

    def test_board_is_exactly_frozen_and_query_order_does_not_change_branches(self):
        one, ik, gate = self.setup()
        two, _, _ = self.setup()
        contexts = [self.context("raised", height=.41), self.context()]
        a, b = one.prepare_contexts(contexts), two.prepare_contexts(contexts[::-1])
        self.assertEqual(a, b[::-1])
        gaps = [.16, .24, .32]
        first = one.audit(["raised"] * 3, gaps)
        second = two.audit(["raised"] * 3, gaps[::-1])
        self.assertEqual(first, second[::-1])
        self.assertTrue(all(row["eligible"] for row in first))
        self.assertEqual(len({row["board_branch_sha256"] for row in first}), 1)
        self.assertEqual(len({np.asarray(row["q14"]).tobytes() for row in first}), 1)
        for call in gate.calls:
            np.testing.assert_array_equal(call["q14"], a[0]["q14"])
        self.assertEqual(len(ik.board.calls), 1)
        # Dropper continuation is always the immutable anchor, not the last gap.
        np.testing.assert_array_equal(ik.dropper.calls[0][1]["known"], self.anchor()["q10"])

    def test_exact_anchor_requires_and_preserves_known_branches(self):
        one, ik, _ = self.setup()
        prepared = one.prepare_contexts([self.context(), self.context("raised", height=.41)])
        np.testing.assert_array_equal(prepared[0]["q14"], self.anchor()["q14"])
        self.assertTrue(prepared[0]["preserved_anchor_board"])
        rows = one.audit(["anchor", "anchor", "raised"],
                         [REFERENCE_VALUES[4], REFERENCE_VALUES[4] + 1e-7, REFERENCE_VALUES[4]])
        self.assertTrue(rows[0]["preserved_anchor_dropper"])
        np.testing.assert_array_equal(rows[0]["q10"], self.anchor()["q10"])
        self.assertFalse(rows[1]["preserved_anchor_dropper"])
        self.assertFalse(rows[2]["preserved_anchor_dropper"])
        np.testing.assert_array_equal(ik.dropper.calls[0][1]["preserve_known"], [True, False, False])

    def test_failed_anchor_control_cannot_silently_switch_to_another_board(self):
        original = self.anchor()["q14"]
        one, _, _ = self.setup(board_check=lambda q, target:
                              dict(passed=not np.array_equal(q, original), reason="mock_collision"))
        prepared = one.prepare_contexts([self.context()])[0]
        self.assertFalse(prepared["board_admitted"])
        self.assertEqual(len(prepared["board_branch_checks"]), 1)
        self.assertEqual(prepared["reason"], "board_at_common_park_mock_collision")

    def test_geometry_selects_board_before_gaps_without_outcome_based_substitution(self):
        checked = []
        def board_check(q, target):
            checked.append(q.copy())
            return dict(passed=len(checked) > 1, reason="mock_self_collision")
        one, ik, gate = self.setup(board_check=board_check,
                                  reject=lambda q10, q14, values: values[4] < .2)
        prepared = one.prepare_contexts([self.context("new", height=.41)])[0]
        self.assertEqual(prepared["board_branch_id"], 1)
        rows = one.audit(["new", "new"], [.16, .32])
        self.assertFalse(rows[0]["eligible"])
        self.assertTrue(rows[1]["eligible"])
        self.assertEqual(rows[0]["reason"], "strict_clearance_reject")
        self.assertEqual(len(checked), 2)
        self.assertEqual(rows[0]["q14"], rows[1]["q14"])
        self.assertEqual(len(gate.calls), 4)
        self.assertFalse(rows[1]["native_release"])
        self.assertIsNone(rows[1]["path_admitted"])
        self.assertNotIn("success", rows[1])
        self.assertEqual(len(ik.board.calls), 1)

    def test_dropper_collision_can_try_dropper_branches_but_never_change_board(self):
        one, _, gate = self.setup(reject=lambda q10, q14, values: q10[0] < .5)
        prepared = one.prepare_contexts([self.context("raised", height=.41)])[0]
        row = one.audit(["raised"], [.25])[0]
        self.assertTrue(row["eligible"])
        self.assertEqual(row["dropper_branch_id"], 1)
        self.assertEqual(len(row["branch_checks"]), 2)
        for call in gate.calls:
            np.testing.assert_array_equal(call["q14"], prepared["q14"])
        # At the exact control, replacing a rejected historical dropper would
        # invalidate the comparison, even if another mathematical branch fits.
        one.prepare_contexts([self.context()])
        row = one.audit(["anchor"], [REFERENCE_VALUES[4]])[0]
        self.assertFalse(row["eligible"])
        self.assertEqual(len(row["branch_checks"]), 1)

    def test_missing_ik_remains_unresolved_without_claiming_impossibility(self):
        one, ik, _ = self.setup()
        ik.board.valid = False
        context = one.prepare_contexts([self.context()])[0]
        self.assertEqual(context["reason"], "anchor_board_branch_unresolved")
        row = one.audit(["anchor"], [.25])[0]
        self.assertEqual(row["reason"], "board_context_unresolved")
        self.assertFalse(ik.dropper.calls)
        one, ik, _ = self.setup()
        one.prepare_contexts([self.context()])
        ik.dropper.valid = False
        row = one.audit(["anchor"], [.25])[0]
        self.assertEqual(row["reason"], "dropper_ik_unresolved")
        self.assertFalse(row["eligible"])

    def test_versioned_bounds_do_not_mutate_older_contract(self):
        older = copy.deepcopy(WIDE_BOUNDS)
        self.assertEqual(BOUNDS["low"][2:], [.04, 8., .02])
        self.assertEqual(BOUNDS["high"][2:], [.56, 64., .65])
        one, ik, _ = self.setup()
        bad = one.prepare_contexts([self.context("too_low", height=.039), self.context("too_tilted", tilt=65)])
        self.assertEqual([c["reason"] for c in bad], ["context_bounds_rejected"] * 2)
        self.assertFalse(ik.board.calls)
        one.prepare_contexts([self.context()])
        rows = one.audit(["anchor"] * 2, [.019, .651])
        self.assertEqual([r["reason"] for r in rows], ["action_bounds_or_invalid"] * 2)
        self.assertFalse(ik.dropper.calls)
        self.assertEqual(WIDE_BOUNDS, older)
        json.dumps(rows, allow_nan=False)

    def test_context_and_cache_results_are_copies_and_conflicting_ids_reject(self):
        one, _, _ = self.setup()
        with self.assertRaises(ValueError):
            one.prepare_contexts([self.context(), self.context()])
        one.prepare_contexts([self.context()])
        with self.assertRaises(ValueError):
            one.prepare_contexts([self.context(height=.4)])
        with self.assertRaises(ValueError):
            one.audit(["unknown"], [.3])
        with self.assertRaises(ValueError):
            one.audit(["anchor"], [np.nan])
        with self.assertRaises(ValueError):
            one.audit(["anchor"], [.2, .3])
        contexts = one.contexts
        contexts[0]["q14"][0] = 999
        first = one.audit(["anchor"], [.25])
        second = copy.deepcopy(first)
        first[0]["q14"][0] = 999
        self.assertEqual(one.audit(["anchor"], [.25]), second)
        self.assertNotEqual(one.contexts[0]["q14"][0], 999)
        json.dumps(second, allow_nan=False)

    def test_anchor_only_reads_geometry_fields_not_source_outcomes(self):
        class GeometryOnly(dict):
            def __getitem__(self, key):
                if key not in {"q10", "q14", "values", "bucket_xy_m"}:
                    raise AssertionError("Outcome/provenance fields must not select branches")
                return super().__getitem__(key)
        raw = GeometryOnly(self.anchor(), success=object(), reward=object(), outcomes=object())
        one, _, _ = self.setup(anchor=raw)
        two, _, _ = self.setup()
        one.prepare_contexts([self.context()]); two.prepare_contexts([self.context()])
        self.assertEqual(one.audit(["anchor"], [.25]), two.audit(["anchor"], [.25]))

    def test_native_board_gate_checks_parked_pair_and_full_pose(self):
        one, _, gate = self.setup()
        one.board_checker = None
        calls = []
        gate.strict = SimpleNamespace(bucket_xy=np.array([999., 999.]),
            check=lambda q: (calls.append(q.copy()) or dict(passed=True, minimum_clearance_m=.006)))
        values = np.asarray([REFERENCE_VALUES])
        world = shot_targets(self.scene(), np.asarray([REFERENCE_BUCKET]), values[:, :3],
                             values[:, 3], values[:, 4])["board_world"][0]
        with patch("sircus.mjlab.window_geometry.native_pose_error", return_value=dict(position_m=.0002, rotation_rad=0)) as error:
            result = one._check_board(np.asarray(self.anchor()["q14"]), world)
        self.assertFalse(result["passed"])
        self.assertEqual(result["reason"], "native_board_full_pose_mismatch")
        np.testing.assert_array_equal(calls[0][:6], one.park_q10)
        np.testing.assert_array_equal(calls[0][6:], self.anchor()["q14"])
        np.testing.assert_array_equal(gate.strict.bucket_xy, REFERENCE_BUCKET)
        self.assertEqual(error.call_args.args[1], 14)

    def test_full_flange_rotation_errors_cannot_pass_position_normal_only_gate(self):
        one, _, _ = self.setup()
        one.prepare_contexts([self.context("raised", height=.41)])
        one.pose_checker = lambda *args: {
            "10": dict(position_m=0., rotation_rad=.001),
            "14": dict(position_m=0., rotation_rad=0.),
        }
        row = one.audit(["raised"], [.25])[0]
        self.assertFalse(row["eligible"])
        self.assertEqual(row["reason"], "native_pose_mismatch")
        self.assertEqual(len(row["branch_checks"]), 3)

    def test_clearance_contract_cannot_be_relaxed(self):
        one, ik, gate = self.setup()
        gate.clearance_m = .001
        with self.assertRaises(ValueError):
            WindowGeometry(self.scene(), "/mock", self.anchor(), ik=ik, gate=gate,
                           initials=one.initials)


if __name__ == "__main__":
    unittest.main()
