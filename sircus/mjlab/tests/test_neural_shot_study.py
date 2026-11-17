"""Learning labels must retain exact commands and never turn unrun shots into misses."""
from copy import deepcopy
import unittest

from sircus.mjlab.neural_shot_study import learning_rows, GPU_CAP_S, STAGE_CAPS


class LearningEvidenceTests(unittest.TestCase):
    def fixture(self):
        proposals = [dict(id=f"p{i}", values=[.5, -.2, .3, 30., .3], bucket_xy_m=[.8, -.4])
                     for i in range(3)]
        admitted = [dict(row, id=row["id"] + "-b0", proposal_id=row["id"]) for row in proposals[:2]]
        sites = [dict(layout=dict(split="train"), proposals=proposals, admitted=admitted,
                      requests=admitted[:1])]
        observations = {admitted[0]["id"]: dict(admitted[0], native_release=True)}
        return sites, observations

    def test_setup_only_and_rejection_remain_distinct(self):
        sites, observations = self.fixture()
        rows = learning_rows(sites, observations)["train"]
        self.assertTrue(rows[0]["setup_ok"])
        self.assertIsNotNone(rows[0]["observation"])
        self.assertTrue(rows[1]["setup_ok"])
        self.assertIsNone(rows[1]["observation"])
        self.assertFalse(rows[2]["setup_ok"])
        self.assertIsNone(rows[2]["observation"])

    def test_wrong_native_command_is_not_a_label(self):
        sites, observations = self.fixture()
        observations["p0-b0"]["values"] = [1., 0., .2, 30., .3]
        with self.assertRaisesRegex(ValueError, "exact proposed"):
            learning_rows(sites, observations)

    def test_missing_extra_or_duplicate_selected_drops_rejected(self):
        sites, observations = self.fixture()
        for observed in ({}, dict(observations, extra=observations["p0-b0"])):
            with self.assertRaises(ValueError):
                learning_rows(sites, observed)
        sites[0]["requests"].append(deepcopy(sites[0]["requests"][0]))
        with self.assertRaises(ValueError):
            learning_rows(sites, observations)

    def test_test_rows_never_enter_training(self):
        sites, observations = self.fixture()
        sites[0]["layout"]["split"] = "test"
        with self.assertRaisesRegex(ValueError, "TEST"):
            learning_rows(sites, observations)

    def test_reserved_test_budget_keeps_total_within_authorization(self):
        self.assertEqual(sum(STAGE_CAPS.values()), GPU_CAP_S)
        self.assertEqual(GPU_CAP_S, 7 * 3600)
        self.assertGreaterEqual(STAGE_CAPS["test"], 2 * 3600)


if __name__ == "__main__":
    unittest.main()
