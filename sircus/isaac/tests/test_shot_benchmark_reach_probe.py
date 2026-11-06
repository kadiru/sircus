"""Synthetic CPU diagnostics only; no native or capture evidence is generated."""

import copy
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import shot_benchmark_reach_probe as probe
from prepare_assets import fk, matrix


def chain_fixture():
    axes = [[0, 0, 1], [0, 1, 0], [0, 1, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]]
    translations = [[0, 0, .1], [.1, 0, 0], [.1, 0, 0], [0, 0, .1], [0, 0, .1], [0, 0, .05]]
    return [(matrix(xyz), np.array(axis), -3., 3.) for xyz, axis in zip(translations, axes)]


def case(identifier="failure", arm="Arm10", residual=.01, solved=False):
    joints = [0.] * 6 if solved else [.2, -.2, .2, -.2, .2, -.2]
    return dict(case_id=identifier, arm=arm, split="train",
        target_arm_base=fk([0.] * 6, chain_fixture()).tolist(),
        saved_fit=dict(q=joints, residual=residual, solved=solved), home_q=[0.] * 6,
        provenance=dict(trial_file="synthetic-only.json", trial_sha256="a" * 64))


class ProbeTests(unittest.TestCase):
    def setUp(self):
        self.links = {name:chain_fixture() for name in ("Arm10", "Arm14")}

    def test_selection_precedes_results_and_stratifies_both_arms(self):
        candidates = []
        for arm in ("Arm10", "Arm14"):
            for index, residual in enumerate([.0002, .0003, .0004, .02, .03, .04]):
                candidates.append(case(f"{arm}-failure-{index}", arm, residual))
            candidates.append(case(f"{arm}-control", arm, 0., True))
        selected, receipt = probe.select_cases(dict(probe_candidates=list(reversed(candidates)),
                                                   cases=[dict(visualization_only=True)]))
        self.assertEqual(len(selected), 10)
        for arm in self.links:
            failures = [row for row in selected if row["arm"] == arm and not row["saved_fit"]["solved"]]
            self.assertEqual(len(failures), 4)
            self.assertEqual(sum(row["saved_fit"]["residual"] < .001 for row in failures), 2)
        again, _ = probe.select_cases(dict(probe_candidates=candidates))
        self.assertEqual(selected, again)
        self.assertEqual(receipt["selection_sha256"], probe.canonical_hash(selected))

    def test_input_validation_excludes_heldout_and_invalid_frames(self):
        for change in [dict(split="test"), dict(split="validation"), dict(target_arm_base=np.zeros((4, 4)).tolist()),
                       dict(saved_fit=dict(q=[0.] * 6, residual=.01, solved=True))]:
            with self.assertRaises(ValueError):
                probe.select_cases(dict(cases=[dict(case(), **change)]))
        with self.assertRaises(ValueError):
            probe.select_cases(dict(cases=[case(str(i)) for i in range(11)]))
        with self.assertRaises(ValueError):
            probe.select_cases(dict(cases=[case(), case()]))

    def test_independent_starts_are_reproducible_and_inside_limits(self):
        row = probe.validate_case(case())
        starts, seed = probe.starts_for(row, self.links["Arm10"])
        other, repeat_seed = probe.starts_for(row, self.links["Arm10"])
        self.assertEqual(seed, repeat_seed)
        self.assertEqual(len(starts), 10)
        self.assertEqual([name for name, _ in starts[:2]], ["saved_best", "saved_home"])
        np.testing.assert_array_equal(np.array([q for _, q in starts]), np.array([q for _, q in other]))
        self.assertTrue(all(np.all(np.abs(q) < 3.) for _, q in starts))
        row["home_q"] = None
        self.assertEqual(probe.starts_for(row, self.links["Arm10"])[0][1][0], "midpoint_home_unavailable")

    def test_real_cpu_optimizer_recovers_saved_endpoint_and_keeps_input(self):
        payload = dict(cases=[case(), case("board", "Arm14"), case("control", residual=0., solved=True)])
        saved = copy.deepcopy(payload)
        result = probe.run_probe(payload, self.links, wall_s=5., max_nfev=100)
        self.assertEqual(result["counts"]["recovered_endpoint"], 2)
        self.assertEqual(result["counts"]["control_verified"], 1)
        self.assertTrue(all(row["best"]["residual"] < 1e-4 for row in result["cases"]))
        self.assertEqual(payload, saved)
        self.assertFalse(result["native_trial_labels_changed"])
        self.assertFalse(result["learner_feedback"])
        self.assertFalse(result["simulator_started"])
        self.assertEqual(result["cases"][1]["objective"], "position plus normal only")

    def test_optimizer_claim_cannot_bypass_independent_fk_gate(self):
        def bogus(*args, **kwargs):
            return SimpleNamespace(x=np.array([.7] * 6), fun=np.zeros(6), nfev=1)
        with patch.object(probe, "least_squares", bogus):
            result = probe.run_probe(dict(cases=[case()]), self.links, wall_s=5.)
        row = result["cases"][0]
        self.assertEqual(row["status"], "still_unresolved")
        self.assertEqual(len(row["attempts"]), 10)
        self.assertFalse(row["endpoint_verified"])
        self.assertGreater(row["best"]["residual"], 1e-4)

    def test_evaluation_budget_interrupts_current_and_preserves_remaining_not_run(self):
        result = probe.run_probe(dict(cases=[case(), case("remaining")]), self.links,
                                 wall_s=5., evaluations=1)
        self.assertEqual(result["residual_evaluations"], 1)
        self.assertEqual(result["cases"][0]["status"], "budget_exhausted")
        self.assertEqual(result["cases"][1]["status"], "not_run")
        self.assertEqual(result["cases"][1]["reason"], "residual_evaluation_budget")

    def test_wall_timeout_marks_unrun_instead_of_impossible(self):
        class Clock:
            def __init__(self):
                self.now = -1
            def __call__(self):
                self.now += 1
                return float(self.now)
        result = probe.run_probe(dict(cases=[case(), case("remaining")]), self.links,
                                 wall_s=.5, clock=Clock())
        self.assertEqual([row["status"] for row in result["cases"]], ["not_run", "not_run"])
        self.assertEqual(result["residual_evaluations"], 0)

    def test_hard_limits_cannot_be_expanded_through_function_arguments(self):
        for parameters in [dict(wall_s=181), dict(evaluations=100001), dict(max_nfev=1001)]:
            with self.assertRaises(ValueError):
                probe.run_probe(dict(cases=[case()]), self.links, **parameters)


if __name__ == "__main__":
    unittest.main()
