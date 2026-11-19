"""Deadline enforcement discards unfinished shots without skipping admission."""
from contextlib import ExitStack
from copy import deepcopy
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from sircus.mjlab import planning_budget_plan as budget


def _snapshot(name):
    return dict(choice=dict(request=dict(id=name)), completed_proposal_ids=[name])


def _fast(*, publish, deadline):
    publish(_snapshot("complete"))


def _hang(*, publish, deadline):
    while True:
        time.sleep(.02)


def _complete_then_hang(*, publish, deadline):
    publish(_snapshot("safe"))
    _hang(publish=publish, deadline=deadline)


def _late(*, publish, deadline):
    publish(_snapshot("safe"))
    time.sleep(max(0., deadline - time.perf_counter()) + .05)
    publish(_snapshot("late"))


def _bad(*, publish, deadline):
    publish(_snapshot("safe"))
    raise ValueError("invalid candidate")


def _masked(*, publish, deadline):
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("CUDA device visible")
    publish(_snapshot("cpu"))


def _slow_init():
    time.sleep(.15)


def _tamper(journal, *, publish, deadline):
    publish(_snapshot("safe"))
    (Path(journal) / "snapshot-00000.json").write_text("{}")


class BudgetSupervisorTests(unittest.TestCase):
    def test_fast_complete_choice_is_preserved(self):
        result = budget.supervise(_fast, budget_s=.5)
        self.assertEqual(result["status"], "completed")
        self.assertFalse(result["deadline_reached"])
        self.assertEqual(result["snapshot"]["choice"]["request"]["id"], "complete")
        receipt = result["accepted_receipts"][0]
        self.assertLessEqual(receipt["elapsed_s"], receipt["received_elapsed_s"])
        self.assertLessEqual(result["decision_latency_s"], .5)

    def test_hard_hang_terminates_owned_child_with_no_choice(self):
        result = budget.supervise(_hang, budget_s=.1)
        self.assertEqual(result["status"], "deadline")
        self.assertTrue(result["deadline_reached"])
        self.assertIsNone(result["snapshot"])
        self.assertEqual(result["decision_latency_s"], .1)
        self.assertLess(result["stop_overhead_s"], 1.)
        with self.assertRaises(ProcessLookupError):
            os.kill(result["worker_pid"], 0)

    def test_timeout_salvages_only_previously_complete_choice(self):
        result = budget.supervise(_complete_then_hang, budget_s=.15)
        self.assertTrue(result["deadline_reached"])
        self.assertEqual(result["snapshot"]["choice"]["request"]["id"], "safe")
        self.assertEqual(len(result["accepted_receipts"]), 1)

    def test_late_choice_cannot_replace_predeadline_choice(self):
        result = budget.supervise(_late, budget_s=.15)
        self.assertTrue(result["deadline_reached"])
        self.assertEqual(result["snapshot"]["choice"]["request"]["id"], "safe")
        self.assertTrue(all(r["received_elapsed_s"] <= .15 for r in result["accepted_receipts"]))

    def test_exception_fails_closed_even_with_earlier_choice(self):
        with self.assertRaisesRegex(RuntimeError, "invalid candidate"):
            budget.supervise(_bad, budget_s=.5)

    def test_common_initialization_is_separately_reported(self):
        result = budget.supervise(_fast, initializer=_slow_init, budget_s=.1)
        self.assertGreaterEqual(result["startup_s"], .15)
        self.assertEqual(result["status"], "completed")
        self.assertLess(result["decision_latency_s"], .1)

    def test_cpu_child_hides_cuda(self):
        with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "visible-in-parent"}):
            result = budget.supervise(_masked, budget_s=.5)
            self.assertEqual(os.environ["CUDA_VISIBLE_DEVICES"], "visible-in-parent")
        self.assertEqual(result["snapshot"]["choice"]["request"]["id"], "cpu")

    def test_immutable_journal_detects_tampering(self):
        with tempfile.TemporaryDirectory() as directory:
            journal = Path(directory) / "journal"
            with self.assertRaisesRegex(ValueError, "changed after publication"):
                budget.supervise(_tamper, (str(journal),), journal_directory=journal, budget_s=.5)

    def test_budget_and_existing_journal_validation(self):
        for value in (0, -1, float("nan"), float("inf"), True):
            with self.assertRaises(ValueError):
                budget.supervise(_fast, budget_s=value)
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(FileExistsError):
                budget.supervise(_fast, journal_directory=directory, budget_s=.5)


@unittest.skipUnless(importlib.util.find_spec("torch"), "Pinned planner imports require torch")
class IncrementalPlannerTests(unittest.TestCase):
    def setUp(self):
        from sircus.mjlab import neural_shot_plan, neural_shot_model
        from sircus.mjlab import clearance_expansion_plan, intergrid_system_plan
        self.neural, self.model = neural_shot_plan, neural_shot_model
        self.expansion, self.system = clearance_expansion_plan, intergrid_system_plan
        self.layout = dict(id="site", context_id="site", split="test", bucket_xy_m=[.8, -.5], stratum="interior")
        self.engine = SimpleNamespace(adapter=SimpleNamespace(env=None))
        self.ranker = Mock()
        self.ranker.order.side_effect = lambda rows: sorted(rows, key=lambda row: -row["score"])
        self.ranker.predict.side_effect = lambda row: dict(score_m=row["score"])
        self.history = [dict(id=f"history-{index}", bucket_xy_m=[.4 + index * .01, -.5],
                             direct_capture=True, table_capture=False) for index in range(5)]

    def row(self, index, score=.02):
        return dict(id=f"proposal-{index}", proposal_id=f"proposal-{index}", context_id="site",
                    bucket_xy_m=self.layout["bucket_xy_m"], values=[.4, -.3, .38, 28., .2 + index * .001],
                    intended_route="direct", method="direct", phase="test", role="test",
                    board_heading_offset_deg=0., board_twist_deg=0.,
                    selected_q10=[.1]*6, selected_q14=[.2]*6, score=score)

    def run_plan(self, method, rows, worker=None, branch_rows=()):
        snapshots = []
        with ExitStack() as stack:
            stack.enter_context(patch.object(self.neural.workers, "_engine", self.engine))
            prepare = stack.enter_context(patch.object(self.neural.workers, "prepare_proposals", side_effect=worker or (lambda batch, _: (batch, []))))
            stack.enter_context(patch.object(self.expansion, "_ranker", return_value=self.ranker))
            stack.enter_context(patch.object(self.neural, "menu", return_value=rows))
            stack.enter_context(patch.object(self.model, "order", side_effect=lambda _, menu: list(reversed(menu))))
            stack.enter_context(patch.object(self.system, "base_proposals", return_value=rows))
            stack.enter_context(patch.object(self.system, "fallback_proposals", return_value=[]))
            branch = stack.enter_context(patch.object(self.expansion, "branch_proposals", return_value=list(branch_rows)))
            branch_prepare = stack.enter_context(patch.object(self.expansion, "_prepare_branches", return_value=dict(admitted=[], records=[])))
            def publish(value):
                snapshots.append(deepcopy(value))
                return True
            budget.incremental_plan(self.layout, method, {}, {}, self.history, {"assets": "/assets"}, {},
                                    publish=publish, deadline=time.perf_counter() + 10.)
        return snapshots, prepare, branch, branch_prepare

    def test_neural_publishes_single_candidates_but_stops_after_safe_batch(self):
        rows = [self.row(i) for i in range(8)]
        snapshots, prepared, branch, _ = self.run_plan("neural_selector", rows)
        self.assertEqual(prepared.call_count, 4)
        self.assertTrue(all(len(call.args[0]) == 1 for call in prepared.call_args_list))
        self.assertEqual(snapshots[-1]["completed_proposal_ids"], [row["id"] for row in reversed(rows)][0:4])
        self.assertEqual(len(snapshots[-1]["plan"]["neural"]["admitted"]), 4)
        self.assertEqual(snapshots[-1]["active_stage"], "finished")
        branch.assert_not_called()

    def test_rejection_cannot_become_selected_choice(self):
        rows = [self.row(i) for i in range(8)]
        def prepare(batch, warm):
            return ([], [{"id": batch[0]["id"], "reason": "IK"}]) if batch[0]["id"] == rows[-1]["id"] else (batch, [])
        snapshots, _, _, _ = self.run_plan("neural_selector", rows, worker=prepare)
        self.assertIsNone(snapshots[0]["choice"]["request"])
        self.assertNotEqual(snapshots[-1]["choice"]["request"]["proposal_id"], rows[-1]["id"])

    def test_neural_exhaustion_preserves_generated_prefix(self):
        rows = [self.row(i, -.1) for i in range(20)]
        snapshots, prepared, branch, _ = self.run_plan("neural_selector", rows, worker=lambda batch, _: ([], []))
        self.assertEqual(prepared.call_count, 16)
        self.assertEqual(len(snapshots[-1]["generated"]["neural"]), 16)
        self.assertEqual(snapshots[-1]["completed_proposal_ids"], [row["id"] for row in reversed(rows)][:16])
        self.assertIsNone(snapshots[-1]["choice"]["request"])
        branch.assert_not_called()

    def test_current_incremental_final_command_matches_existing_selection(self):
        rows = [self.row(i, i / 100.) for i in range(4)]
        snapshots, prepared, _, _ = self.run_plan("current_planner", rows)
        from sircus.mjlab.intergrid_candidate_replay_plan import command_key
        old = self.expansion.current_selection(self.layout, snapshots[-1]["plan"], self.history, self.ranker)
        self.assertEqual(command_key(old["request"]), command_key(snapshots[-1]["choice"]["request"]))
        self.assertEqual(snapshots[-1]["choice"]["request"]["method"], "current_planner")
        self.assertEqual(prepared.call_count, 4)

    def test_native_allocation_rejected_before_publication(self):
        def allocate(batch, warm):
            self.engine.adapter.env = object()
            return batch, []
        with self.assertRaisesRegex(RuntimeError, "native CUDA"):
            self.run_plan("neural_selector", [self.row(i) for i in range(4)], worker=allocate)

    def test_cached_physics_preserves_score_ties_and_isolation(self):
        cached = budget.CachedRanker(self.ranker)
        first, second = self.row(0), self.row(1)
        renamed = dict(first, id="z-renamed")
        ordered = cached.order([renamed, second, first])
        self.assertEqual([row["id"] for row in ordered], [first["id"], second["id"], renamed["id"]])
        self.assertEqual(self.ranker.predict.call_count, 2)
        altered = cached.predict(first)
        altered["score_m"] = -999
        self.assertEqual(cached.predict(first)["score_m"], .02)
        self.assertEqual(self.ranker.predict.call_count, 2)
        cached.predict(dict(first, selected_q14=[.5]*6))
        cached.predict(dict(first, bucket_xy_m=[.81, -.5]))
        self.assertEqual(self.ranker.predict.call_count, 4)

    def test_incremental_publications_do_not_repeat_physics_work(self):
        self.run_plan("neural_selector", [self.row(i) for i in range(8)])
        self.assertEqual(self.ranker.predict.call_count, 4)


if __name__ == "__main__":
    unittest.main()
