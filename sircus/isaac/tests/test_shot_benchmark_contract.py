"""Pure accounting/layout tests, not simulation or policy-performance evidence."""

import copy
import sys
import unittest
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import shot_benchmark_contract as contract


class ContractTests(unittest.TestCase):
    def test_fixed_paid_protocol_and_balanced_method_order(self):
        value = contract.make_protocol()
        self.assertEqual(value["total_train_attempts"], 60000)
        self.assertEqual(value["train_budget_per_method_seed"], 10000)
        self.assertEqual(value["run_order"], [
            dict(method="sweep", seed=20261121), dict(method="ppo", seed=20261121),
            dict(method="ppo", seed=20261122), dict(method="sweep", seed=20261122),
            dict(method="sweep", seed=20261123), dict(method="ppo", seed=20261123)])
        self.assertEqual(value["evaluation"]["total_requested_per_method_seed"], 420)
        self.assertIn("Final 10000", value["evaluation"]["selection"])
        self.assertEqual(value["domain"]["low"], [-np.pi, .28, .14, .08, np.radians(12.)])
        self.assertEqual(value["domain"]["high"], [np.pi, 1.2, .6, .56, np.radians(52.)])
        for key, replacement in [("train_budget_per_method_seed", 1600), ("total_train_attempts", 10000)]:
            altered = {k: v for k, v in value.items() if k != "fingerprint"}
            altered[key] = replacement
            with self.assertRaises(ValueError):
                contract.validate_protocol(contract.signed(altered))

    def test_exact_milestone_cuts_never_spend_future_samples(self):
        plans = []
        for seed in contract.SEEDS:
            schedule = contract.training_schedule(seed)
            plan = contract.batch_plan(contract.make_protocol(), seed)
            self.assertEqual(len(plan), 289)
            self.assertEqual(plan[-1]["attempts"], 25)
            self.assertEqual([i for row in plan for i in row["site_indices"]], schedule)
            self.assertEqual(Counter(Counter(schedule).values()), {286: 25, 285: 10})
            self.assertEqual([r["milestone_after"] for r in plan if r["milestone_after"]], [1000, 2500, 5000, 10000])
            end = 0
            for call, row in enumerate(plan):
                self.assertEqual(row["call_index"], call)
                self.assertEqual(row["attempt_start"], end)
                self.assertEqual(row["attempts"], sum(row["enabled"]))
                self.assertEqual(len(set(row["site_indices"])), len(row["site_indices"]))
                self.assertTrue(all(row["enabled"][i] for i in row["site_indices"]))
                self.assertFalse(any(row["attempt_start"] < m < row["attempt_stop"] for m in contract.MILESTONES))
                end = row["attempt_stop"]
            self.assertEqual(end, 10000)
            plans.append(schedule)
        self.assertNotEqual(plans[0], plans[1])
        self.assertEqual(plans[0], contract.training_schedule(contract.SEEDS[0]))

    def test_fresh_stratified_layouts_and_all_historical_exclusions(self):
        excluded = [[.5, -.2], [.8, -.3], [2., 2.]]
        splits = contract.make_splits(excluded)
        self.assertEqual(splits, contract.make_splits(list(reversed(excluded)) + excluded[:1]))
        contract.validate_splits(splits, excluded_coordinates=excluded)
        points = []
        for role, count in contract.SITE_COUNTS.items():
            self.assertEqual(len(splits[role]), count)
            self.assertEqual(Counter(c["tile_index"] for c in splits[role]), {i: count // 35 for i in range(35)})
            points.extend(c["bucket_xy_m"] for c in splits[role])
        points = np.asarray(points)
        self.assertGreaterEqual(np.linalg.norm(points[:, None] - np.asarray(excluded)[None, :], axis=2).min(), .025)
        distances = np.linalg.norm(points[:, None] - points[None, :], axis=2)
        np.fill_diagonal(distances, np.inf)
        self.assertGreaterEqual(distances.min(), .025)
        cohorts = contract.evaluation_cohorts(splits, "test")
        self.assertEqual([len(c) for c in cohorts], [35, 35])
        self.assertTrue(all([c["tile_index"] for c in cohort] == list(range(35)) for cohort in cohorts))

    def test_no_silent_spacing_relaxation_or_admission_filtering(self):
        with self.assertRaisesRegex(ValueError, "no constraint was relaxed"):
            contract.make_splits([[0., 0.]], bounds=[[0., 0.], [.001, .001]], max_draws=140)
        for value in (.024, float("nan")):
            with self.assertRaises(ValueError):
                contract.make_splits([[0., 0.]], minimum_separation_m=value)
        with self.assertRaises(ValueError):
            contract.make_splits([[float("nan"), 0.]])

    def test_signed_layout_tampering_fails_even_after_rehashing(self):
        splits = contract.make_splits([[2., 2.]])
        changed = copy.deepcopy(splits)
        changed["train"][0]["bucket_xy_m"] = changed["train"][1]["bucket_xy_m"]
        with self.assertRaises(ValueError):
            contract.validate_splits(changed)
        changed.pop("fingerprint")
        with self.assertRaises(ValueError):
            contract.validate_splits(contract.signed(changed))
        with self.assertRaises(ValueError):
            contract.validate_splits(splits, excluded_coordinates=[[3., 3.]])
        changed = copy.deepcopy(splits)
        changed.pop("fingerprint")
        changed["test"][0]["capture"] = True
        with self.assertRaises(ValueError):
            contract.validate_splits(contract.signed(changed))

    def test_local_generators_do_not_change_caller_rng(self):
        before = np.random.get_state()
        contract.make_splits([[2., 2.]])
        contract.training_schedule(contract.SEEDS[0])
        after = np.random.get_state()
        self.assertEqual(before[0], after[0])
        np.testing.assert_array_equal(before[1], after[1])
        self.assertEqual(before[2:], after[2:])


if __name__ == "__main__":
    unittest.main()
