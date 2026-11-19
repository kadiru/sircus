"""Learned ordering reduces setup work without replacing exact safety admission."""
from contextlib import ExitStack
from copy import deepcopy
import importlib.util
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch


@unittest.skipUnless(importlib.util.find_spec("torch"), "Pinned native planner imports need torch")
class NeuralShotPlanTests(unittest.TestCase):
    def setUp(self):
        from sircus.mjlab import neural_shot_plan as plan
        from sircus.mjlab import neural_shot_model as model
        self.plan, self.model = plan, model
        self.layout = dict(id="site", context_id="site", split="test", bucket_xy_m=[.8, -.5], stratum="interior")
        self.ranker = Mock()
        self.ranker.order.side_effect = lambda rows: sorted(rows, key=lambda row: -row.get("score", -.01))
        self.ranker.predict.side_effect = lambda row: dict(score_m=row.get("score", -.01))
        self.engine = SimpleNamespace(adapter=SimpleNamespace(env=None))

    def row(self, index, score=-.01):
        return dict(id=f"proposal-{index}", context_id="site", proposal_id=f"proposal-{index}",
                    bucket_xy_m=[.8, -.5], values=[.4, -.3, .38, 28., .2 + index * .001],
                    intended_route="direct", board_heading_offset_deg=0., board_twist_deg=0.,
                    seed_q10=[0.] * 6, seed_q14=[0.] * 6,
                    selected_q10=[.1] * 6, selected_q14=[.2] * 6, score=score)

    def prepared(self, rows, *, worker, branches=None, rank_order=None):
        stack = ExitStack()
        self.addCleanup(stack.close)
        stack.enter_context(patch.object(self.plan, "menu", return_value=rows))
        stack.enter_context(patch.object(self.plan, "_ranker", return_value=self.ranker))
        stack.enter_context(patch.object(self.plan.workers, "_engine", self.engine))
        checked = stack.enter_context(patch.object(self.plan.workers, "prepare_proposals", side_effect=worker))
        ordering = stack.enter_context(patch.object(self.model, "order", side_effect=rank_order or (lambda _, menu: list(reversed(menu)))))
        stack.enter_context(patch.object(self.model, "predict", return_value={"capture_probability": .5}))
        branch = stack.enter_context(patch.object(self.plan, "_prepare_branches", return_value=branches or self.plan._empty()))
        proposal = stack.enter_context(patch.object(self.plan, "branch_proposals", return_value=[]))
        return checked, ordering, branch, proposal

    def test_menu_is_cheap_unique_and_keeps_validation_identity(self):
        original = [self.row(i) for i in range(8)]
        transferred = [self.row(i) for i in range(8, 40)]
        libraries = {"original": {"direct": []}, "expanded": {"table": []}}
        layout = dict(self.layout, split="validation")
        snapshot = deepcopy(original)
        with patch.object(self.plan, "base_proposals", return_value=original) as base, \
             patch.object(self.plan, "collection_geometry", return_value=transferred), \
             patch.object(self.plan.workers, "prepare_proposals", side_effect=AssertionError("No IK in menu")):
            rows = self.plan.menu(layout, libraries, {})
        self.assertEqual(len(rows), self.plan.MENU_LIMIT)
        self.assertEqual(len({self.plan._geometry(row) for row in rows}), len(rows))
        self.assertEqual({row["dataset_split"] for row in rows}, {"validation"})
        self.assertEqual(base.call_args.args[0]["split"], "train")
        self.assertEqual(original, snapshot)

    def test_rank_before_ik_and_stop_after_first_safe_batch(self):
        rows = [self.row(i, .02) for i in range(40)]
        order_seen = []
        def rank(_, menu):
            order_seen.append(True)
            return list(reversed(menu))
        def prepare(batch, warm):
            self.assertTrue(order_seen)
            self.assertTrue(warm)
            return batch, []
        checked, _, branch, _ = self.prepared(rows, worker=prepare, rank_order=rank)
        result = self.plan.prepare_neural(self.layout, {}, {}, [], {"assets": "/assets"}, {})
        self.assertEqual(checked.call_count, 1)
        self.assertEqual(checked.call_args.args[0], list(reversed(rows))[:4])
        self.assertEqual(result["choices"][0]["checked_proposals"], 4)
        self.assertFalse(result["choices"][0]["predicted_flight_risk"])
        branch.assert_not_called()

    def test_rejected_top_proposal_cannot_be_selected_for_release(self):
        rows = [self.row(i, .02) for i in range(8)]
        def prepare(batch, warm):
            return batch[1:], [dict(id=batch[0]["id"], rejection="IK")]
        checked, _, branch, _ = self.prepared(rows, worker=prepare)
        result = self.plan.prepare_neural(self.layout, {}, {}, [], {"assets": "/assets"}, {})
        self.assertEqual(result["choices"][0]["request"]["proposal_id"], rows[-2]["id"])
        self.assertNotEqual(result["choices"][0]["request"]["proposal_id"], rows[-1]["id"])
        self.assertEqual(checked.call_count, 1)
        branch.assert_not_called()

    def test_all_rejections_exhaust_fixed_budget_without_branch_or_drop(self):
        rows = [self.row(i) for i in range(40)]
        checked, _, branch, _ = self.prepared(rows, worker=lambda batch, warm: ([], []))
        result = self.plan.prepare_neural(self.layout, {}, {}, [], {"assets": "/assets"}, {})
        self.assertEqual(checked.call_count, 4)
        self.assertEqual(sum(len(call.args[0]) for call in checked.call_args_list), self.plan.CHECK_LIMIT)
        self.assertIsNone(result["choices"][0]["request"])
        self.assertTrue(result["choices"][0]["predicted_flight_risk"])
        branch.assert_not_called()

    def test_branch_search_receives_only_checked_admitted_geometry(self):
        rows = [self.row(i) for i in range(40)]
        changed = dict(rows[-1], id="alternate", selected_q14=[.8] * 6, score=.03)
        checked, _, branch, proposed = self.prepared(rows, worker=lambda batch, warm: (batch, []),
                             branches=dict(proposals=[], admitted=[changed], records=[]))
        result = self.plan.prepare_neural(self.layout, {}, {}, [], {"assets": "/assets"}, {})
        checked_ids = {row["id"] for call in checked.call_args_list for row in call.args[0]}
        branch_input = proposed.call_args.args[1]
        self.assertEqual({row["id"] for row in branch_input["base"]["admitted"]}, checked_ids)
        self.assertEqual(set(branch.call_args.args[2]), checked_ids)
        self.assertEqual(len(checked_ids), self.plan.CHECK_LIMIT)
        self.assertEqual(result["choices"][0]["request"]["selected_q14"], [.8] * 6)
        self.assertFalse(result["choices"][0]["predicted_flight_risk"])

    def test_collection_is_bounded_preserves_all_admission_evidence_and_reproducible(self):
        rows = [self.row(i, i / 1000.) for i in range(50)]
        with patch.object(self.plan, "menu", return_value=rows), \
             patch.object(self.plan.workers, "prepare_proposals", return_value=(rows, [])), \
             patch.object(self.plan, "_ranker", return_value=self.ranker), \
             patch.object(self.plan.workers, "_engine", self.engine):
            train = self.plan.prepare_collection(dict(self.layout, split="train"), {}, {}, {})
            repeated = self.plan.prepare_collection(dict(self.layout, split="train"), {}, {}, {})
            validation = self.plan.prepare_collection(dict(self.layout, split="validation"), {}, {}, {})
        self.assertEqual(len(train["admitted"]), 50)
        self.assertEqual(len(train["requests"]), self.plan.TRAIN_RELEASE_LIMIT)
        self.assertEqual(len(validation["requests"]), self.plan.VALIDATION_RELEASE_LIMIT)
        self.assertEqual(train["requests"], repeated["requests"])
        self.assertEqual(train["requests"][:4], list(reversed(rows))[:4])

    def test_native_allocation_in_cpu_preparation_fails(self):
        self.engine.adapter.env = object()
        rows = [self.row(i, .02) for i in range(4)]
        self.prepared(rows, worker=lambda batch, warm: (batch, []))
        with self.assertRaisesRegex(RuntimeError, "allocated native CUDA"):
            self.plan.prepare_neural(self.layout, {}, {}, [], {"assets": "/assets"}, {})

    def test_dataset_mapping_keeps_unreleased_admissions_and_rejects_missing_or_wrong_labels(self):
        from sircus.mjlab.neural_shot_study import learning_rows

        proposals = [self.row(index) for index in range(3)]
        released = dict(proposals[0], id="native-release")
        unreleased = dict(proposals[1], id="unreleased")
        site = dict(layout=dict(self.layout, split="train"), proposals=proposals,
                    admitted=[released, unreleased], requests=[released])
        rows = learning_rows([site], {released["id"]: released})["train"]
        self.assertEqual([row["setup_ok"] for row in rows], [True, True, False])
        self.assertEqual(rows[0]["request"]["id"], proposals[0]["id"])
        self.assertEqual(rows[0]["observation"]["id"], released["id"])
        self.assertIsNone(rows[1]["observation"])
        self.assertIsNone(rows[2]["observation"])
        with self.assertRaisesRegex(ValueError, "Every and only"):
            learning_rows([site], {})
        changed = dict(released, bucket_xy_m=[.1, .2])
        with self.assertRaisesRegex(ValueError, "exact proposed shot"):
            learning_rows([site], {released["id"]: changed})
        site["layout"]["split"] = "test"
        with self.assertRaisesRegex(ValueError, "TEST cannot"):
            learning_rows([site], {released["id"]: released})


if __name__ == "__main__":
    unittest.main()
