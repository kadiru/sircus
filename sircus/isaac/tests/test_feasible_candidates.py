"""CPU-only support generation tests; no trained model or native rollout."""

import copy
import json
import math
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np

SOURCE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOURCE))
import feasible_candidates as fc
from expanded_pose_menu import endpoint_targets


class CandidateTests(unittest.TestCase):
    def setUp(self):
        self.cells = [dict(id="train-a", bucket_xy_m=[.27, -.1], split="train")]
        self.values = [0., .4, .35, .24, math.radians(29)]
        self.parent = {"train-a":fc.decode_action(self.cells[0], self.values, "parent-exact")}
        self.means = {"train-a":[0.]*5}
        self.scales = {"train-a":[.2]*5}
        self.scene = json.loads((SOURCE/"scene.json").read_text())
        targets = endpoint_targets(self.scene, self.cells[0], self.parent["train-a"])
        self.links = {name:[(targets[name].copy() if i == 0 else np.eye(4),
                             np.array([0.,0.,1.]), -2*math.pi, 2*math.pi)
                           for i in range(6)] for name in fc.ARMS}
        self.starts = {"train-a":{name:[0.]*6 for name in fc.ARMS}}
        self.fit = dict(q=[0.]*6, solved=True, residual=0., position_error_mm=0.,
                        orientation_error_deg=0., attempts=[])

    def success(self, case="history-1", action=None):
        return dict(case_id=case, cell=copy.deepcopy(self.cells[0]),
                    action=copy.deepcopy(self.parent["train-a"] if action is None else action),
                    ik={name:copy.deepcopy(self.fit) for name in fc.ARMS})

    def plan(self, successes=(), **kwargs):
        return fc.generate_candidates(self.cells, self.parent, self.means, self.scales,
                                      list(successes), **kwargs)

    def solve(self, *args):
        return copy.deepcopy(self.fit)

    def test_union_counts_exact_aliases_and_all_successes_preserved(self):
        # Two historical cases with the exact parent physical action collapse
        # only geometrically; both original cases remain explicit aliases.
        history = [self.success("h1"), self.success("h2")]
        value = self.plan(history, nearby_per_cell=2, global_per_cell=4)
        self.assertEqual(value["source_requests"], {"parent":1, "nearby":2, "global":4, "historical_train_success":2})
        self.assertEqual(value["requests"], 9)
        self.assertEqual(value["unique_candidates"], 7)
        first = value["records"][0]
        self.assertEqual(first["action"], self.parent["train-a"])
        self.assertEqual([s["kind"] for s in first["sources"]], ["parent", "historical_train_success", "historical_train_success"])
        self.assertEqual([w["case_id"] for w in first["witnesses"]], ["h1", "h2"])
        self.assertTrue(value["all_historical_actions_included"])
        fc.validate_plan(value)

    def test_no_rounding_or_nearest_action_replacement(self):
        changed = self.values.copy()
        changed[2] = float(np.nextafter(changed[2], 1.))
        action = fc.decode_action(self.cells[0], changed, "one-ulp-different")
        value = self.plan([self.success(action=action)], nearby_per_cell=0, global_per_cell=0)
        self.assertEqual(value["unique_candidates"], 2)
        self.assertEqual(value["records"][1]["action"], action)
        self.assertNotEqual(value["records"][0]["candidate_id"], value["records"][1]["candidate_id"])

    def test_deterministic_generation_full_domain_and_nearby_decode(self):
        one = self.plan(nearby_per_cell=16, global_per_cell=64)
        self.assertEqual(one, self.plan(nearby_per_cell=16, global_per_cell=64))
        other = self.plan(nearby_per_cell=16, global_per_cell=64, random_seed=20261125)
        self.assertNotEqual(one["fingerprint"], other["fingerprint"])
        global_rows = [r for r in one["records"] if r["sources"][0]["kind"] == "global"]
        unit = (np.array([r["values"] for r in global_rows])-fc.LOW)/(fc.HIGH-fc.LOW)
        self.assertEqual(unit.shape, (64,5))
        for dim in range(5):
            self.assertEqual(set((unit[:,dim]*8).astype(int)), set(range(8)))
        nearby = [r for r in one["records"] if r["sources"][0]["kind"] == "nearby"]
        rng = np.random.default_rng(20261124)
        physical = fc.LOW+(np.tanh(.2*rng.standard_normal((16,5)))+1)*(fc.HIGH-fc.LOW)/2
        np.testing.assert_array_equal([r["values"] for r in nearby], physical)

    def test_parent_arrays_and_mapping_inputs_have_identical_content(self):
        array = fc.generate_candidates(self.cells, [self.parent["train-a"]], [[0.]*5], [[.2]*5], [])
        self.assertEqual(array, self.plan())

    def test_outcome_metadata_cannot_rank_or_change_generator(self):
        history = self.success()
        original = self.plan([history])
        changed = copy.deepcopy(history)
        changed.update(reward=999., route_quality=0, route_success=False)
        changed["cell"].update(reward=-1, capture=True)
        changed["action"].update(reward=1000, captured=True)
        self.assertEqual(original, self.plan([changed]))
        self.assertNotIn("reward", original["records"][0]["action"])

    def test_heldout_changed_or_duplicate_contexts_fail_closed(self):
        for split in ("validation", "test", None):
            with self.subTest(split=split), self.assertRaisesRegex(ValueError,"TRAIN"):
                fc.generate_candidates([dict(self.cells[0], split=split)], self.parent, self.means, self.scales, [])
        bad = self.success();bad["cell"]["bucket_xy_m"][0] += .001
        with self.assertRaisesRegex(ValueError,"context changed"):
            self.plan([bad])
        bad = self.success();bad["cell"]["split"] = "test"
        with self.assertRaisesRegex(ValueError,"TRAIN"):
            self.plan([bad])
        with self.assertRaisesRegex(ValueError,"Duplicate"):
            fc.generate_candidates(self.cells*2, [self.parent["train-a"]]*2, [[0.]*5]*2, [[.2]*5]*2, [])

    def test_bad_distribution_budget_action_or_success_witness_is_rejected(self):
        for kwargs in (dict(nearby_per_cell=-1), dict(global_per_cell=3), dict(random_seed=True)):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.plan(**kwargs)
        bad = copy.deepcopy(self.parent);bad["train-a"]["gap_m"] = .079
        with self.assertRaisesRegex(ValueError,"bounds"):
            fc.generate_candidates(self.cells,bad,self.means,self.scales,[])
        bad = copy.deepcopy(self.scales);bad["train-a"][0] = 0.
        with self.assertRaisesRegex(ValueError,"Gaussian"):
            fc.generate_candidates(self.cells,self.parent,self.means,bad,[])
        with self.assertRaisesRegex(ValueError,"duplicate historical"):
            self.plan([self.success(), self.success()])
        bad = self.success();bad["ik"].pop("Arm14")
        with self.assertRaisesRegex(ValueError,"both historical"):
            self.plan([bad])

    def test_modified_plan_or_source_binding_is_rejected(self):
        plan = self.plan()
        plan["records"][0]["values"][0] += .1
        with self.assertRaisesRegex(ValueError,"fingerprint"):
            fc.validate_plan(plan)
        plan = self.plan()
        with patch.object(fc,"source_hashes",return_value={"changed":"0"*64}):
            with self.assertRaisesRegex(ValueError,"source version"):
                fc.validate_plan(plan)

    def test_resigning_cannot_hide_missing_parent_or_historical_alias(self):
        plan = self.plan([self.success()],nearby_per_cell=0,global_per_cell=0)
        raw = fc.unsigned(plan)
        raw["records"][0]["sources"] = raw["records"][0]["sources"][1:]
        raw["source_requests"]["parent"] = 0
        raw["requests"] -= 1
        with self.assertRaisesRegex(ValueError,"union incomplete"):
            fc.validate_plan(fc.signed(raw))
        raw = fc.unsigned(plan)
        raw["records"][0]["witnesses"] = []
        with self.assertRaisesRegex(ValueError,"witness aliases"):
            fc.validate_plan(fc.signed(raw))

    def test_real_inherited_geometry_checks_preserve_good_historical_witness(self):
        history = [self.success("h1"), self.success("h2")]
        plan = self.plan(history,nearby_per_cell=0,global_per_cell=0)
        calls = []
        def solve(target,links,seed,normal_only):
            calls.append((np.asarray(seed).tolist(),normal_only))
            return self.solve()
        result = fc.qualify_candidates(plan,self.scene,self.links,self.starts,solver=solve,solver_protocol="synthetic-test")
        self.assertEqual((result["requests"],result["admitted"]),(1,1))
        self.assertEqual(result["source_survival"]["historical_train_success"]["admitted_aliases"],2)
        self.assertEqual(len(result["history_survival"]),2)
        self.assertTrue(all(h["generator_retained"] and h["accepted"] for h in result["history_survival"]))
        row = result["records"][0]
        self.assertEqual(row["witness_count"],2)
        self.assertEqual(row["unneeded_witness_cases"],["h2"])
        self.assertEqual(row["attempted_seed_count"],1)
        self.assertEqual(len(row["decision"]["path"]["checks"]),16)
        self.assertEqual(calls,[([0.]*6,False),([0.]*6,True)])
        self.assertFalse(result["release_authorized"])

    def test_all_witnesses_verified_even_when_first_witness_admits(self):
        one,two = self.success("h1"),self.success("h2")
        two["ik"]["Arm10"]["residual"] = .1
        plan = self.plan([one,two],nearby_per_cell=0,global_per_cell=0)
        with self.assertRaisesRegex(ValueError,"residual"):
            fc.qualify_candidates(plan,self.scene,self.links,self.starts,solver=self.solve,solver_protocol="test")

    def test_finite_alternate_witnesses_preserved_and_attempted_without_action_change(self):
        plan = self.plan([self.success("h1"),self.success("h2")],nearby_per_cell=0,global_per_cell=0)
        denied = dict(accepted=False,status="unresolved",reason="endpoint_ik_unresolved_Arm14")
        admitted = dict(accepted=True,status="admitted",reason=None)
        with patch.object(fc,"preflight",side_effect=[denied,admitted]) as p:
            result = fc.qualify_candidates(plan,self.scene,self.links,self.starts)
        row = result["records"][0]
        self.assertEqual(row["attempted_seed_count"],2)
        self.assertTrue(row["accepted"])
        self.assertEqual([a["seed_source"]["case_id"] for a in row["attempts"]],["h1","h2"])
        self.assertEqual(p.call_args_list[0].args[2],p.call_args_list[1].args[2])

    def test_reasons_empty_cell_support_and_history_exclusions_are_explicit(self):
        plan = self.plan([self.success()],nearby_per_cell=0,global_per_cell=0)
        denied = dict(accepted=False,status="rejected_candidate",reason="board_table_proxy_overlap")
        with patch.object(fc,"preflight",return_value=denied):
            result = fc.qualify_candidates(plan,self.scene,self.links,self.starts)
        self.assertEqual(result["per_cell"][0]["status"],"unresolved")
        self.assertEqual(result["history_survival"][0]["reason"],"board_table_proxy_overlap")
        self.assertTrue(result["history_survival"][0]["generator_retained"])
        self.assertEqual(result["source_survival"]["historical_train_success"]["exclusion_reasons"],dict(board_table_proxy_overlap=1))
        for kwargs in (dict(workers=0),dict(workers=9),dict(workers=2,solver=self.solve,solver_protocol="test")):
            with self.assertRaises(ValueError):
                fc.qualify_candidates(plan,self.scene,self.links,self.starts,**kwargs)

    def test_changed_common_state_membership_and_nonfinite_joints_fail_closed(self):
        plan = self.plan(nearby_per_cell=0,global_per_cell=0)
        for starts in ({}, {"test-a":self.starts["train-a"]},
                       {"train-a":dict(self.starts["train-a"],Arm10=[float("nan")]*6)}):
            with self.subTest(starts=starts), self.assertRaises(ValueError):
                fc.qualify_candidates(plan,self.scene,self.links,starts)


if __name__ == "__main__":
    unittest.main()
