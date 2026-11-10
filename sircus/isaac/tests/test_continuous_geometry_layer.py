"""CPU-only geometry adapter checks with explicitly synthetic kinematic chains."""

import copy
import json
import math
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch

import numpy as np

SOURCE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOURCE))
import continuous_geometry_layer as layer
import continuous_feasibility as geometry
import joint_feasibility as joint
from expanded_pose_menu import endpoint_targets


class ContinuousGeometryLayerTests(unittest.TestCase):
    def setUp(self):
        self.scene = json.loads((SOURCE / "scene.json").read_text())
        self.cell = dict(id="synthetic-context", bucket_xy_m=[.27, -.10])
        self.values = [0., .4, .35, .24, math.radians(29)]
        self.action = geometry.decode_action(self.cell, self.values, "direct-neural-output")
        targets = endpoint_targets(self.scene, self.cell, self.action)
        self.links = {name: [(targets[name].copy() if i == 0 else np.eye(4),
                             np.array([0., 0., 1.]), -2*math.pi, 2*math.pi)
                            for i in range(6)] for name in joint.ARMS}
        self.starts = {name: [0.]*6 for name in joint.ARMS}
        self.calls = []

    def solver(self, target, links, seed, normal_only):
        self.calls.append((np.asarray(target).copy(), np.asarray(seed).copy(), normal_only))
        error = layer._pose_error(target, np.zeros(6), links, normal_only)
        residual = float(np.linalg.norm(error))
        return dict(q=[0.]*6, solved=residual < 1e-4, residual=residual)

    def run_filter(self, **kwargs):
        args = dict(scene=self.scene, cell=self.cell, action=self.action, links=self.links,
                    starts=self.starts, solver=self.solver, solver_protocol="synthetic-FK-only")
        args.update(kwargs)
        return layer.filter_shot(**args)

    def validate(self, result, **kwargs):
        args = dict(scene=self.scene, cell=self.cell, action=self.action, links=self.links,
                    starts=self.starts, solver_protocol="synthetic-FK-only")
        args.update(kwargs)
        return layer.validate_result(result, **args)

    def resign(self, record):
        record = copy.deepcopy(record)
        record.pop("fingerprint")
        return dict(record, fingerprint=geometry.canonical_hash(record))

    def test_originally_admitted_action_stays_exactly_unchanged_without_repair_work(self):
        snapshot = copy.deepcopy((self.action, self.starts, self.cell))
        with patch.object(layer, "_repair_proposal", side_effect=AssertionError("no repair allowed")):
            result = self.run_filter()
        self.assertEqual(result["classification"], "unchanged")
        self.assertTrue(result["accepted"])
        self.assertEqual(result["action"], self.action)
        self.assertEqual(result["final_preflight"], result["original_preflight"])
        self.assertEqual(result["delta_physical"], [0.]*5)
        self.assertEqual(result["normalized_adjustment_l2"], 0.)
        self.assertEqual(result["repair"]["residual_evaluations"], 0)
        self.assertFalse(result["repair"]["attempted"])
        self.assertEqual(result["preflight_calls"], 1)
        self.assertEqual((self.action, self.starts, self.cell), snapshot)
        self.assertEqual([row[2] for row in self.calls], [False, True])
        self.assertEqual(self.validate(result), "unchanged")
        self.assertFalse(result["release_authorized"])
        self.assertEqual(result["training_updates"], result["native_shots"], 0)

    def test_actual_continuous_optimizer_repairs_nearby_unreachable_height(self):
        # Opposing tiny origins preserve exact FK at zero while giving the
        # synthetic chain a nonzero necessary outer bound.
        for name in joint.ARMS:
            self.links[name][1][0][0, 3] = .001
            self.links[name][2][0][0, 3] = -.001
        bad = geometry.decode_action(self.cell, [0., .4, .351, .24, math.radians(29)], self.action["id"])
        result = self.run_filter(action=bad)
        self.assertEqual(result["classification"], "adjusted")
        self.assertTrue(result["accepted"])
        self.assertEqual(result["original_reason"], "endpoint_ik_unresolved_Arm10")
        self.assertEqual(result["preflight_calls"], 2)
        self.assertTrue(result["repair"]["attempted"])
        self.assertLessEqual(result["repair"]["residual_evaluations"], 512)
        self.assertLess(max(abs(x) for x in result["delta_normalized"]), .1 + 1e-12)
        self.assertAlmostEqual(result["action"]["height_m"], .35, delta=1e-4)
        self.assertNotEqual(result["action"], bad)
        self.assertEqual(self.validate(result, action=bad), "adjusted")
        self.assertFalse(result["bucket_impossibility_proven"])
        self.assertFalse(result["release_authorized"])

    def test_disabled_repair_leaves_declined_request_unresolved_without_retry(self):
        bad = geometry.decode_action(self.cell, [0., .4, .37, .24, math.radians(29)], self.action["id"])
        with patch.object(layer, "_repair_proposal", side_effect=AssertionError("disabled")):
            result = self.run_filter(action=bad, repair=False)
        self.assertEqual(result["classification"], "rejected")
        self.assertFalse(result["accepted"])
        self.assertIsNone(result["action"])
        self.assertIsNone(result["final_preflight"])
        self.assertEqual(result["repair"]["reason"], "repair_disabled")
        self.assertEqual(result["preflight_calls"], 1)
        self.assertIsNone(result["delta_physical"])
        self.assertFalse(result["bucket_impossibility_proven"])

    def test_hard_residual_budget_stops_optimizer_and_still_requires_final_gates(self):
        bad = geometry.decode_action(self.cell, [0., .4, .37, .24, math.radians(29)], self.action["id"])
        result = self.run_filter(action=bad, repair_max_evaluations=1)
        self.assertEqual(result["classification"], "rejected")
        self.assertEqual(result["repair"]["residual_evaluations"], 1)
        self.assertEqual(result["repair"]["optimizer"]["termination"], "residual_budget_exhausted")
        self.assertLessEqual(result["preflight_calls"], 2)
        self.assertIsNone(result["action"])

    def test_far_target_outside_local_adjustment_box_remains_rejected(self):
        bad = geometry.decode_action(self.cell, [0., .4, .52, .24, math.radians(29)], self.action["id"])
        result = self.run_filter(action=bad, max_normalized_adjustment=.01, repair_max_nfev=4,
                                 repair_max_evaluations=96)
        self.assertEqual(result["classification"], "rejected")
        self.assertFalse(result["accepted"])
        self.assertLessEqual(result["repair"]["residual_evaluations"], 96)
        candidate = result["repair"]["candidate"]
        deltas = [(candidate[k] - bad[k]) / (geometry.HIGH[i] - geometry.LOW[i])
                  for i, k in enumerate(geometry.ACTION_NAMES)]
        self.assertLessEqual(max(abs(x) for x in deltas), .01 + 1e-12)
        self.assertFalse(result["bucket_impossibility_proven"])

    def test_optimizer_success_flag_is_never_sufficient_for_acceptance(self):
        bad = geometry.decode_action(self.cell, [0., .4, .52, .24, math.radians(29)], self.action["id"])
        candidate = geometry.decode_action(self.cell, [0., .4, .51, .24, math.radians(29)], self.action["id"])
        proposal = dict(candidate=candidate, ik_seeds_rad=self.starts,
                        optimizer=dict(success=True, termination="synthetic-success", nfev=1),
                        residual_evaluations=1, reason=None)
        with patch.object(layer, "_repair_proposal", return_value=proposal):
            result = self.run_filter(action=bad)
        self.assertEqual(result["classification"], "rejected")
        self.assertTrue(result["repair"]["optimizer"]["success"])
        self.assertFalse(result["repair"]["candidate_preflight"]["accepted"])

    def test_original_path_gate_still_rejects_after_endpoints_pass(self):
        path = dict(accepted=False, reasons=["synthetic-path-overlap"], checks=[])
        with patch.object(geometry, "approximate_path_check", return_value=path):
            result = self.run_filter(repair=False)
        self.assertEqual(result["classification"], "rejected")
        self.assertTrue(result["original_preflight"]["qualification"]["accepted"])
        self.assertEqual(result["original_reason"], "synthetic-path-overlap")

    def test_invalid_action_or_mismatched_derived_geometry_fails_closed(self):
        for bad in ([0., .4, .35, .07999999, math.radians(29)], [0., .4, .35, float("nan"), .5], [0.]*4):
            with self.subTest(action=bad), self.assertRaises(ValueError):
                self.run_filter(action=bad)
        bad = dict(self.action, board_x_m=self.action["board_x_m"] + .001)
        with self.assertRaisesRegex(ValueError, "geometry disagrees"):
            self.run_filter(action=bad)

    def test_minimum_gap_keeps_original_ball_bottom_definition(self):
        action = geometry.decode_action(self.cell, [0., .4, .35, .08, math.radians(29)], "min-gap")
        result = self.run_filter(action=action, repair=False)
        targets = result["original_preflight"]["targets"]
        self.assertAlmostEqual(targets["release_center_m"][2] - targets["board_world"][2][3], .08 + .0375)
        self.assertAlmostEqual(targets["Arm10"][2][3] - targets["board_world"][2][3], .08 + .0375 + .095)

    def test_protocol_budgets_are_explicit_and_reject_invalid_limits(self):
        cases = [dict(repair=1), dict(repair_max_nfev=0), dict(repair_max_nfev=257),
                 dict(repair_max_evaluations=0), dict(repair_max_evaluations=4097),
                 dict(max_normalized_adjustment=0), dict(max_normalized_adjustment=.251),
                 dict(max_normalized_adjustment=float("nan"))]
        for args in cases:
            with self.subTest(args=args), self.assertRaises(ValueError):
                self.run_filter(**args)

    def test_fixed_seeds_are_only_ik_initializations_and_actual_starts_bind_paths(self):
        seeds = {name: [.001]*6 for name in joint.ARMS}
        result = self.run_filter(seeds=seeds)
        self.assertEqual(result["classification"], "unchanged")
        for call in self.calls:
            np.testing.assert_array_equal(call[1], [.001]*6)
        self.assertEqual(result["original_preflight"]["binding"]["measured_starts_rad"], self.starts)
        self.assertEqual(self.validate(result, seeds=seeds), "unchanged")
        with self.assertRaisesRegex(ValueError, "Stale"):
            self.validate(result)

    def test_changed_state_and_model_free_source_bindings_invalidate_reuse(self):
        result = self.run_filter()
        starts = copy.deepcopy(self.starts); starts["Arm10"][0] += .001
        with self.assertRaisesRegex(ValueError, "Stale"):
            self.validate(result, starts=starts)
        scene = copy.deepcopy(self.scene); scene["ball"]["radius_m"] += .001
        with self.assertRaisesRegex(ValueError, "Stale"):
            self.validate(result, scene=scene)
        with patch.object(layer, "_sources", return_value={"changed.py": "0"*64}):
            with self.assertRaisesRegex(ValueError, "Stale geometry-layer source"):
                self.validate(result)

    def test_tampered_classification_delta_or_release_authority_fails_closed(self):
        result = self.run_filter()
        unsigned_tamper = copy.deepcopy(result); unsigned_tamper["classification"] = "adjusted"
        with self.assertRaisesRegex(ValueError, "fingerprint"):
            self.validate(unsigned_tamper)
        changes = [dict(classification="adjusted"), dict(delta_physical=[1.]*5),
                   dict(release_authorized=True), dict(native_shots=1),
                   dict(preflight_calls=2), dict(bucket_impossibility_proven=True)]
        for change in changes:
            tamper = copy.deepcopy(result); tamper.update(change)
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.validate(self.resign(tamper))

    def test_repair_may_not_move_an_originally_valid_shot_even_with_a_resigned_record(self):
        result = self.run_filter()
        result["action"]["height_m"] += .001
        with self.assertRaisesRegex(ValueError, "exactly unchanged"):
            self.validate(self.resign(result))

    def test_outcome_metadata_does_not_change_the_geometric_decision(self):
        a = self.run_filter()
        b = self.run_filter(cell=dict(self.cell, reward=99, split="test"),
                            action=dict(self.action, capture=True))
        self.assertEqual(a["classification"], b["classification"])
        self.assertEqual(a["action"], b["action"])
        self.assertEqual(a["original_preflight"]["binding"], b["original_preflight"]["binding"])
        self.assertNotIn("reward", b["action"])
        self.assertNotIn("capture", b["action"])

    def test_import_is_model_and_simulator_free(self):
        code = ("import sys; sys.path.insert(0, %r); import continuous_geometry_layer; "
                "assert 'torch' not in sys.modules; "
                "assert not any(x.startswith(('isaacsim','isaaclab')) for x in sys.modules)") % str(SOURCE)
        subprocess.run([sys.executable, "-c", code], check=True)


if __name__ == "__main__":
    unittest.main()
