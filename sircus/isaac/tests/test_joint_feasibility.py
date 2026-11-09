"""Pure CPU contract tests; synthetic chains are not robot validation evidence."""

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
import joint_feasibility as layer
from continuous_feasibility import canonical_hash, decode_action
from expanded_pose_menu import endpoint_targets


class JointFeasibilityTests(unittest.TestCase):
    def setUp(self):
        self.scene = json.loads((SOURCE / "scene.json").read_text())
        self.cell = dict(id="synthetic-context", bucket_xy_m=[.27, -.10])
        self.values = [0., .4, .35, .24, math.radians(29)]
        self.action = decode_action(self.cell, self.values, "exact-request")
        targets = endpoint_targets(self.scene, self.cell, self.action)
        # Exactly solvable synthetic kinematics exercise the inherited FK,
        # capsule and path checks without importing Isaac or downloaded URDFs.
        self.links = {name: [(targets[name].copy() if i == 0 else np.eye(4),
                             np.array([0., 0., 1.]), -2*math.pi, 2*math.pi)
                            for i in range(6)] for name in layer.ARMS}
        self.starts = {name: [0.]*6 for name in layer.ARMS}
        self.calls = []

    def solve(self, target, links, seed, normal_only):
        self.calls.append((target.copy(), np.asarray(seed).copy(), normal_only))
        return dict(q=[0.]*6, solved=True, residual=0.)

    def preflight(self, **kwargs):
        args = dict(scene=self.scene, cell=self.cell, action=self.action,
                    links=self.links, starts=self.starts, solver=self.solve,
                    solver_protocol="synthetic-exact-FK-test")
        args.update(kwargs)
        return layer.preflight(**args)

    def validate(self, record, **kwargs):
        args = dict(scene=self.scene, cell=self.cell, action=self.action,
                    links=self.links, starts=self.starts,
                    solver_protocol="synthetic-exact-FK-test")
        args.update(kwargs)
        return layer.validate_preflight(record, **args)

    def test_coupled_targets_use_distinct_board_release_and_flange_heights(self):
        self.action = decode_action(self.cell, [0., .4, .35, .08, math.radians(29)], "min-gap")
        # The endpoint targets are exposed even if an old synthetic chain does
        # not solve this changed request. No second ball placement is assumed.
        r = self.preflight(solver=lambda *a: dict(q=[0.]*6, solved=False, residual=.1))
        targets = r["targets"]
        self.assertAlmostEqual(targets["board_world"][2][3], .35)
        self.assertAlmostEqual(targets["release_center_m"][2], .35 + .08 + .0375)
        self.assertAlmostEqual(targets["Arm10"][2][3], .35 + .08 + .0375 + .095)
        self.assertEqual(r["action"], self.action)
        self.assertFalse(r["release_authorized"])

    def test_admission_checks_both_arms_and_preserves_action_and_input_state(self):
        before = copy.deepcopy((self.action, self.starts))
        r = self.preflight()
        self.assertTrue(r["accepted"])
        self.assertEqual(r["status"], "admitted")
        self.assertIsNone(r["reason"])
        self.assertEqual([c[2] for c in self.calls], [False, True])
        self.assertEqual(r["joint_targets_rad"], self.starts)
        self.assertEqual(len(r["path"]["checks"]), 16)
        self.assertEqual((self.action, self.starts), before)
        self.assertTrue(self.validate(r))
        self.assertFalse(r["release_authorized"])
        self.assertFalse(r["bucket_impossibility_proven"])
        self.assertIn("in-plane twist", r["scope"])

    def test_outcomes_and_split_metadata_do_not_enter_inputs_or_actions(self):
        first = self.preflight()
        cell = dict(self.cell, reward=9, route_success=True, split="test")
        action = dict(self.action, reward=9, route_success=True)
        second = self.preflight(cell=cell, action=action)
        self.assertEqual(first["binding"], second["binding"])
        self.assertEqual(first["action"], second["action"])
        self.assertNotIn("reward", second["action"])
        self.assertTrue(second["accepted"])

    def test_finite_support_retains_all_candidates_and_never_selects_a_winner(self):
        actions = [dict(self.action, id="a"), dict(self.action, id="b")]
        r = layer.prepare_candidates(self.scene, self.cell, actions, self.links,
                                     self.starts, solver=self.solve, solver_protocol="test")
        self.assertEqual((r["requests"], r["admitted"]), (2, 2))
        self.assertEqual(r["admitted_indices"], [0, 1])
        self.assertEqual([x["action"] for x in r["records"]], actions)
        self.assertNotIn("selected", r)
        self.assertFalse(r["release_authorized"])
        raw = layer.prepare_candidates(self.scene, self.cell, [self.values], self.links,
                                       self.starts, solver=self.solve, solver_protocol="test")
        self.assertEqual(raw["records"][0]["action"]["id"], "candidate-0000")

    def test_empty_support_is_unresolved_and_bound_to_state_not_impossible(self):
        r = layer.prepare_candidates(self.scene, self.cell, [], self.links, self.starts)
        self.assertEqual(r["status"], "unresolved")
        self.assertEqual(r["requests"], 0)
        self.assertIn("finite_budget", r["reason"])
        self.assertFalse(r["bucket_impossibility_proven"])
        self.assertEqual(r["binding"]["context"], self.cell)

    def test_outer_bound_rejects_exact_candidate_without_solving_or_claiming_bad_bucket(self):
        links = copy.deepcopy(self.links)
        links["Arm10"][0][0][0, 3] += .2
        r = self.preflight(links=links)
        self.assertFalse(r["accepted"])
        self.assertEqual(r["reason"], "outer_link_length_bound")
        self.assertTrue(r["candidate_outside_necessary_reach_bound"])
        self.assertFalse(r["bucket_impossibility_proven"])
        self.assertEqual(self.calls, [])
        self.assertIsNone(r["path"])

    def test_board_arm_failure_is_unresolved_not_a_successful_dropper_only_admission(self):
        def solver(target, links, seed, normal_only):
            return dict(q=[0.]*6, solved=not normal_only, residual=.1 if normal_only else 0.)
        r = self.preflight(solver=solver)
        self.assertFalse(r["accepted"])
        self.assertEqual(r["status"], "unresolved")
        self.assertEqual(r["reason"], "endpoint_ik_unresolved_Arm14")
        self.assertIsNone(r["path"])

    def test_path_rejection_remains_separate_after_good_endpoints(self):
        check = dict(accepted=False, reasons=["inter_arm_proxy_overlap"], checks=[])
        with patch.object(layer.geometry, "approximate_path_check", return_value=check):
            r = self.preflight()
        self.assertTrue(r["qualification"]["accepted"])
        self.assertFalse(r["accepted"])
        self.assertEqual(r["reason"], "inter_arm_proxy_overlap")
        self.assertEqual(r["path"], check)

    def test_original_fk_gate_rejects_a_lying_solver(self):
        bad = dict(q=[.2, 0., 0., 0., 0., 0.], solved=True, residual=0.)
        with self.assertRaisesRegex(ValueError, "original residual"):
            self.preflight(solver=lambda *a: bad)
        with self.assertRaisesRegex(ValueError, "acceptance disagrees"):
            self.preflight(solver=lambda *a: dict(q=[0.]*6, solved=True, residual=.1))

    def test_invalid_request_state_chain_or_solver_metadata_fails_closed(self):
        for bad in ([0., .4, .35, .079999, math.radians(29)],
                    [0., .4, .35, float("nan"), math.radians(29)], [0.]*4):
            with self.subTest(action=bad), self.assertRaises(ValueError):
                self.preflight(action=bad)
        changed = dict(self.action, board_x_m=self.action["board_x_m"]+.001)
        with self.assertRaisesRegex(ValueError, "geometry disagrees"):
            self.preflight(action=changed)
        for starts in ({"Arm10": [0.]*6}, {n: [99.]*6 for n in layer.ARMS}):
            with self.assertRaises(ValueError):
                self.preflight(starts=starts)
        links = copy.deepcopy(self.links); links["Arm10"][0][1][2] = 2.
        with self.assertRaisesRegex(ValueError, "unit joint axis"):
            self.preflight(links=links)
        with self.assertRaisesRegex(ValueError, "explicit solver_protocol"):
            self.preflight(solver_protocol=None)
        with self.assertRaises(ValueError):
            layer.prepare_candidates(self.scene, self.cell, [self.values]*4097, self.links, self.starts)

    def test_changed_action_scene_chain_starts_seeds_and_protocol_invalidate_reuse(self):
        r = self.preflight()
        changes = []
        scene = copy.deepcopy(self.scene); scene["ball"]["radius_m"] += .001
        changes.append(dict(scene=scene))
        starts = copy.deepcopy(self.starts); starts["Arm10"][0] += .001
        changes.append(dict(starts=starts))
        seeds = copy.deepcopy(self.starts); seeds["Arm14"][1] += .001
        changes.append(dict(seeds=seeds))
        links = copy.deepcopy(self.links)
        item = list(links["Arm10"][0]); item[2] += .001; links["Arm10"][0] = tuple(item)
        changes.append(dict(links=links))
        changes.append(dict(action=decode_action(self.cell, [0., .4, .351, .24, math.radians(29)], "exact-request")))
        changes.append(dict(solver_protocol="other-producer"))
        for changed in changes:
            with self.subTest(change=list(changed)), self.assertRaisesRegex(ValueError, "Stale"):
                self.validate(r, **changed)

    def test_altered_source_fingerprint_or_signed_payload_cannot_be_reused(self):
        r = self.preflight()
        tampered = copy.deepcopy(r); tampered["accepted"] = False
        with self.assertRaisesRegex(ValueError, "fingerprint mismatch"):
            self.validate(tampered)
        with patch.object(layer, "_source_hashes", return_value={"changed.py": "0"*64}):
            with self.assertRaisesRegex(ValueError, "Stale"):
                self.validate(r)
        tampered = copy.deepcopy(r); tampered["joint_targets_rad"]["Arm10"][0] = .1
        tampered.pop("fingerprint"); tampered["fingerprint"] = canonical_hash(tampered)
        with self.assertRaisesRegex(ValueError, "Changed prepared joint targets"):
            self.validate(tampered)

    def test_replay_cannot_call_an_absent_saved_arm_or_silently_fall_back(self):
        def saved(target, links, seed, normal_only):
            if normal_only:
                raise ValueError("Saved Arm14 fit was not evaluated")
            return dict(q=[0.]*6, solved=True, residual=0.)
        with self.assertRaisesRegex(ValueError, "not evaluated"):
            self.preflight(solver=saved, solver_protocol="saved-fit-replay")

    def test_module_and_source_binding_do_not_import_torch_or_isaac(self):
        script = ("import sys; sys.path.insert(0, %r); import joint_feasibility as f; "
                  "f._source_hashes(); assert 'torch' not in sys.modules; "
                  "assert not any(n.startswith(('isaacsim','isaaclab')) for n in sys.modules)") % str(SOURCE)
        subprocess.run([sys.executable, "-c", script], check=True)


if __name__ == "__main__":
    unittest.main()
