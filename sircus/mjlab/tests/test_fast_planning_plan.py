"""Owned CPU lifecycle, exclusive timing, and unchanged exact candidate checks."""
from collections import deque
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

from sircus.mjlab import fast_planning_plan as fast
from sircus.mjlab import planning_budget_plan as old


_initialized = 0


def _initialize():
    global _initialized
    _initialized += 1
    time.sleep(.04)


def _snapshot(name):
    return dict(choice=dict(request=dict(id=name)), completed_proposal_ids=[name],
                initializer_calls=_initialized, cuda_visible=os.environ.get("CUDA_VISIBLE_DEVICES"))


def _job(layout, method, *, publish, deadline, profile):
    with profile.stage("menu"):
        time.sleep(.005)
    with profile.stage("arm_setup_checks"):
        if layout.get("error"):
            raise ValueError("injected failure")
        if layout.get("crash"):
            os._exit(9)
        if layout.get("hang"):
            publish(_snapshot("safe"))
            while True:
                time.sleep(.02)
        time.sleep(.005)
    publish(_snapshot(layout["id"]))
    return "completed"


def _late(layout, method, *, publish, deadline, profile):
    publish(_snapshot("safe"))
    with profile.stage("physics_ranking"):
        time.sleep(max(0., deadline-time.perf_counter())+.1)
    publish(_snapshot("late"))
    return "completed"


def _tamper(layout, method, *, publish, deadline, profile):
    publish(_snapshot("safe"))
    (Path(layout["journal"])/"snapshot-00000.json").write_text("{}")
    return "completed"


def _engine():
    base = SimpleNamespace(_connected={"old": 1}, _resolved={"old": 1})
    resolver = SimpleNamespace(pending=deque(), connections={"old": 1}, base=base)
    adapter = SimpleNamespace(env=None, resolver=resolver, _ids={"old"}, packet=4,
                              orientation_by_id={"old": 1})
    return SimpleNamespace(adapter=adapter, prepared_ids={"old"}, _anchors={"old": 1})


class LifecycleTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)

    def worker(self, function=_job, budget=.2):
        worker = fast.OwnedWorker(function, initializer=_initialize, budget_s=budget)
        self.addCleanup(worker.close)
        return worker

    def run_job(self, worker, name, method="warm_30", **values):
        return worker.run(dict(id=name, **values), method, self.root/name)

    def assert_profile(self, result):
        profile = result["profile"]
        self.assertAlmostEqual(sum(profile["stage_seconds"].values()), result["decision_latency_s"])
        cursor = 0.
        for block in profile["blocks"]:
            self.assertEqual(block["start_s"], cursor)
            self.assertGreaterEqual(block["end_s"], block["start_s"])
            cursor = block["end_s"]
        self.assertEqual(cursor, profile["decision_observed_s"])
        self.assertGreaterEqual(result["request_wall_s"], result["startup_s"]+result["wall_planning_s"]+result["receipt_verification_s"])

    def test_reuse_only_initialized_clean_worker(self):
        worker = self.worker()
        first, second = self.run_job(worker, "first"), self.run_job(worker, "second")
        self.assertEqual(first["worker_pid"], second["worker_pid"])
        self.assertEqual([first["worker_job_index"], second["worker_job_index"]], [0, 1])
        self.assertEqual(first["worker_generation"], second["worker_generation"])
        self.assertFalse(first["worker_reused"])
        self.assertTrue(second["worker_reused"])
        self.assertEqual(second["prior_worker_status"], "completed")
        self.assertEqual(second["startup_s"], 0.)
        self.assertGreaterEqual(first["startup_s"], .04)
        self.assertEqual(second["snapshot"]["initializer_calls"], 1)
        self.assertEqual(second["snapshot"]["cuda_visible"], "")
        self.assert_profile(first)
        self.assert_profile(second)

    def test_cold_always_reinitializes(self):
        worker = self.worker()
        first, second = self.run_job(worker, "a", "cold_30"), self.run_job(worker, "b", "cold_30")
        self.assertEqual(second["worker_generation"], first["worker_generation"]+1)
        self.assertEqual(second["worker_job_index"], 0)
        self.assertFalse(second["worker_reused"])
        self.assertIsNone(worker.process)
        self.assertEqual(first["snapshot"]["initializer_calls"], 1)
        self.assertEqual(second["snapshot"]["initializer_calls"], 1)

    def test_timeout_profiles_interrupted_work_and_respawns(self):
        worker = self.worker(budget=.12)
        timed = self.run_job(worker, "timeout", hang=True)
        self.assertEqual(timed["status"], "deadline")
        self.assertEqual(timed["snapshot"]["choice"]["request"]["id"], "safe")
        self.assertEqual(timed["profile"]["interrupted_stage"], "arm_setup_checks")
        self.assertGreater(timed["profile"]["interrupted_stage_s"], .07)
        self.assertIsNone(worker.process)
        with self.assertRaises(ProcessLookupError):
            os.kill(timed["worker_pid"], 0)
        next_job = self.run_job(worker, "after")
        self.assertEqual(next_job["worker_generation"], timed["worker_generation"]+1)
        self.assertFalse(next_job["worker_reused"])
        self.assertEqual(next_job["prior_worker_status"], "deadline")
        self.assertEqual(next_job["worker_job_index"], 0)
        self.assert_profile(timed)

    def test_error_and_crash_destroy_owned_worker(self):
        for kind in ("error", "crash"):
            worker = self.worker()
            with self.assertRaises((RuntimeError, EOFError)):
                self.run_job(worker, kind, **{kind: True})
            self.assertIsNone(worker.process)
            result = self.run_job(worker, kind+"-after")
            self.assertFalse(result["worker_reused"])
            self.assertEqual(result["prior_worker_status"], "error")

    def test_late_snapshot_cannot_replace_complete_choice(self):
        result = self.run_job(self.worker(_late, .1), "late")
        self.assertEqual(result["snapshot"]["choice"]["request"]["id"], "safe")
        self.assertEqual(len(result["accepted_receipts"]), 1)
        self.assertEqual(result["profile"]["interrupted_stage"], "physics_ranking")

    def test_snapshots_are_immutable_and_verification_failure_discards_worker(self):
        worker = self.worker(_tamper)
        journal = self.root/"tamper"
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            self.run_job(worker, "tamper", journal=str(journal))
        self.assertIsNone(worker.process)
        self.assertEqual(worker.prior_status, "error")

    def test_separate_pools_do_not_reuse_workers_or_journals(self):
        left, right = self.worker(), self.worker()
        a, b = self.run_job(left, "left"), self.run_job(right, "right", "warm_early")
        self.assertNotEqual(a["worker_pool_id"], b["worker_pool_id"])
        self.assertNotEqual(a["worker_pid"], b["worker_pid"])
        self.assertFalse(a["worker_reused"] or b["worker_reused"])
        with self.assertRaises(FileExistsError):
            self.run_job(left, "left")

    def test_invalid_timeouts_and_nonmonotonic_profiles_fail(self):
        for value in (0, -1, True, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                fast.OwnedWorker(_job, budget_s=value)
        with self.assertRaises(ValueError):
            fast.build_profile([dict(stage="other", elapsed_s=0.),
                                dict(stage="menu", elapsed_s=.1),
                                dict(stage="other", elapsed_s=.05)], .2, False)


class ResetTests(unittest.TestCase):
    def test_all_query_state_clears_but_geometry_object_persists(self):
        engine = _engine()
        resolver, base = engine.adapter.resolver, engine.adapter.resolver.base
        geometry = object()
        resolver.models = geometry
        fast._reset_query_state(engine)
        for value in (engine.prepared_ids, engine._anchors, engine.adapter._ids,
                      engine.adapter.orientation_by_id, resolver.connections,
                      base._connected, base._resolved):
            self.assertFalse(value)
        self.assertEqual(engine.adapter.packet, 0)
        self.assertIs(resolver.models, geometry)

    def test_dirty_native_or_pending_state_cannot_reuse(self):
        for dirty in ("native", "pending"):
            engine = _engine()
            if dirty == "native":
                engine.adapter.env = object()
            else:
                engine.adapter.resolver.pending.append({})
            with self.assertRaises(RuntimeError):
                fast._reset_query_state(engine)
            self.assertEqual(engine.prepared_ids, {"old"})


@unittest.skipUnless(importlib.util.find_spec("torch"), "Pinned planner imports require torch")
class SelectionTests(unittest.TestCase):
    def setUp(self):
        from sircus.mjlab import neural_shot_plan, neural_shot_model, clearance_expansion_plan
        self.neural, self.model, self.expansion = neural_shot_plan, neural_shot_model, clearance_expansion_plan
        self.layout = dict(id="site", context_id="site", split="test", bucket_xy_m=[.8, -.5], stratum="interior")
        self.engine = _engine()
        self.ranker = Mock()
        self.ranker.predict.side_effect = lambda row: dict(score_m=row["score"])

    def row(self, index, score=.02):
        return dict(id=f"proposal-{index}", proposal_id=f"proposal-{index}", context_id="site",
                    bucket_xy_m=self.layout["bucket_xy_m"], values=[.4, -.3, .38, 28., .2+index*.001],
                    intended_route="direct", method="direct", phase="test", role="test",
                    board_heading_offset_deg=0., board_twist_deg=0.,
                    selected_q10=[.1]*6, selected_q14=[.2]*6, score=score)

    def run_plan(self, method, rows, prepare=None, branch_rows=()):
        snapshots, events = [], []
        with ExitStack() as stack:
            stack.enter_context(patch.object(self.neural.workers, "_engine", self.engine))
            checked = stack.enter_context(patch.object(self.neural.workers, "prepare_proposals", side_effect=prepare or (lambda batch, _: (batch, []))))
            stack.enter_context(patch.object(self.expansion, "_ranker", return_value=self.ranker))
            stack.enter_context(patch.object(self.neural, "menu", return_value=rows))
            stack.enter_context(patch.object(self.model, "order", side_effect=lambda _, menu: menu))
            branches = stack.enter_context(patch.object(self.expansion, "branch_proposals", return_value=list(branch_rows)))
            branch_prepare = stack.enter_context(patch.object(self.expansion, "_prepare_branches", side_effect=lambda _, batch, *args: dict(admitted=batch, records=[])))
            def publish(value):
                snapshots.append(deepcopy(value))
                return True
            args = (self.layout, method, {}, {}, [], {"assets": "/assets"}, {})
            if method == "neural_selector":
                reason = old.incremental_plan(*args, publish=publish, deadline=time.perf_counter()+10.)
            else:
                reason = fast.incremental_fast_plan(*args, publish=publish, deadline=time.perf_counter()+10.,
                    profile=fast.StageRecorder(events.append, time.perf_counter(), "test"))
        return snapshots, checked, branches, branch_prepare, reason

    def test_cold_and_warm_preserve_previous_exact_selection_and_checks(self):
        from sircus.mjlab.intergrid_candidate_replay_plan import command_key
        rows = [self.row(i, .01*i) for i in range(8)]
        original, checked, *_ = self.run_plan("neural_selector", rows)
        for method in ("cold_30", "warm_30"):
            result, actual, _, _, reason = self.run_plan(method, rows)
            self.assertEqual(actual.call_args_list, checked.call_args_list)
            self.assertEqual(result[-1]["plan"], original[-1]["plan"])
            self.assertEqual(result[-1]["generated"], original[-1]["generated"])
            self.assertEqual(command_key(result[-1]["choice"]["request"]), command_key(original[-1]["choice"]["request"]))
            self.assertEqual(reason, "completed")

    def test_early_stops_at_first_fully_checked_threshold_not_later_better_shot(self):
        rows = [self.row(0, -.01), self.row(1, .005), self.row(2, .1)]
        snapshots, checked, branches, _, reason = self.run_plan("warm_early", rows)
        self.assertEqual(checked.call_count, 2)
        self.assertEqual(reason, "early_good_shot")
        self.assertEqual(snapshots[-1]["active_stage"], "early_good_shot")
        self.assertEqual(snapshots[-1]["completed_proposal_ids"], ["proposal-0", "proposal-1"])
        self.assertEqual(snapshots[-1]["choice"]["request"]["proposal_id"], "proposal-1")
        branches.assert_not_called()

    def test_rejected_proposal_cannot_trigger_early_even_if_neural_score_high(self):
        rows = [self.row(0, .9), self.row(1, .0049), self.row(2, .005)]
        def prepare(batch, _):
            return ([], [dict(reason="IK failed")]) if batch[0]["id"] == "proposal-0" else (batch, [])
        snapshots, checked, _, _, reason = self.run_plan("warm_early", rows, prepare=prepare)
        self.assertEqual(checked.call_count, 3)
        self.assertIsNone(snapshots[0]["choice"]["request"])
        self.assertEqual(reason, "early_good_shot")

    def test_threshold_below_5mm_preserves_original_batch_stop(self):
        snapshots, checked, _, _, reason = self.run_plan("warm_early", [self.row(i, .004) for i in range(8)])
        self.assertEqual(checked.call_count, 4)
        self.assertEqual(reason, "completed")
        self.assertEqual(snapshots[-1]["active_stage"], "finished")

    def test_early_can_follow_exact_branch_checks(self):
        rows = [self.row(i, -.1) for i in range(4)]
        branch = self.row(40, .006)
        snapshots, checked, branches, prepared, reason = self.run_plan("warm_early", rows, branch_rows=[branch])
        self.assertEqual(checked.call_count, 4)
        self.assertEqual(prepared.call_count, 1)
        self.assertEqual(reason, "early_good_shot")
        self.assertEqual(snapshots[-1]["completed_proposal_ids"][-1], branch["id"])

    def test_physics_cache_is_per_job_and_setup_checks_repeat(self):
        rows = [self.row(i) for i in range(4)]
        self.run_plan("warm_30", rows)
        self.assertEqual(self.ranker.predict.call_count, 4)
        self.run_plan("warm_30", rows)
        self.assertEqual(self.ranker.predict.call_count, 8)

    def test_native_allocation_fails_before_choice_publication(self):
        def allocate(batch, _):
            self.engine.adapter.env = object()
            return batch, []
        with self.assertRaisesRegex(RuntimeError, "native environment"):
            self.run_plan("warm_early", [self.row(0)], prepare=allocate)


if __name__ == "__main__":
    unittest.main()
