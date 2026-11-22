"""Bounded hybrid switching, immutable receipts and owned CPU-worker lifecycle."""
from collections import deque
from contextlib import nullcontext
import copy
import json
import os
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from sircus.mjlab import stalled_reference_plan as hybrid
from sircus.mjlab import parallel_capture_grid_workers as workers


class Clock:
    def __init__(self):
        self.value = 100.

    def __call__(self):
        return self.value

    def advance(self, duration):
        self.value += duration


def resolver_fixture(clock, *, steps=10, step_s=1., endpoint_s=0., paths=1):
    def check(*_):
        clock.advance(step_s)
        return {"passed": True}

    gate = SimpleNamespace(check=check)

    def plan(*_):
        for _ in range(steps):
            gate.check(None)
        return {"passed": True, "proof": "unchanged"}

    planner = SimpleNamespace(plan=plan)

    def resolve(*_):
        clock.advance(endpoint_s)
        for _ in range(paths):
            result = planner.plan(None)
        return result

    return SimpleNamespace(resolve=resolve, planner=planner,
                           solver=SimpleNamespace(gate=gate))


def fake_engine(directory, mode):
    base = SimpleNamespace(directory=Path(directory), _resolved={}, _connected={})
    resolver = SimpleNamespace(base=base, pending=deque(), connections={})
    adapter = SimpleNamespace(resolver=resolver, directory=Path(directory), env=None,
                              packet=0, _ids=set(), orientation_by_id={})
    return SimpleNamespace(adapter=adapter, directory=Path(directory), prepared_ids=set(),
                           _anchors={}, mode=mode)


def prefix_fixture(*_):
    return dict(menu_sha256=hybrid._digest([]), menu_count=0,
                ranked_proposal_ids=[], ranked_proposal_ids_sha256=hybrid._digest([]))


def snapshot_fixture(layout, mode):
    folder = "resolutions" if mode == "reference" else "direct-contexts"
    return dict(layout=layout, method="warm_early", generated={"menu": []},
        ranked_proposal_ids=[], completed_proposal_ids=["checked"],
        choice=dict(request=dict(id="checked", context_resolution_file=f"{folder}/proof.json")))


class ClockTests(unittest.TestCase):
    def timer(self, resolver, clock, budget=5.):
        events = []
        timer = hybrid.ReferencePathClock(budget, clock=clock).install(resolver)
        timer.begin(clock(), lambda name: events.append((name, timer.state())))
        return timer, events

    def test_ordinary_candidate_checks_do_not_consume_reference_budget(self):
        clock = Clock()
        resolver = resolver_fixture(clock, steps=8)
        timer, events = self.timer(resolver, clock)
        self.assertEqual(resolver.planner.plan(None), {"passed": True, "proof": "unchanged"})
        self.assertEqual(timer.path_s, 0.)
        self.assertEqual(events, [])

    def test_endpoint_time_is_excluded_and_reference_path_is_interrupted(self):
        clock = Clock()
        resolver = resolver_fixture(clock, endpoint_s=12.)
        timer, events = self.timer(resolver, clock)
        with self.assertRaises(hybrid.ReferenceSearchStalled):
            resolver.resolve([.7, -.3])
        self.assertEqual(timer.path_s, 5.)
        self.assertEqual(timer.intervals, [dict(index=0, start_s=12., end_s=17.,
            duration_s=5., completed=False, interrupted=True)])
        self.assertEqual(timer.reference_depth, 0)
        self.assertIsNone(timer.active)
        self.assertEqual([e[0] for e in events], ["reference_path_started", "reference_path_interrupted"])

    def test_reference_budget_accumulates_across_path_attempts(self):
        clock = Clock()
        resolver = resolver_fixture(clock, steps=3, paths=3)
        timer, _ = self.timer(resolver, clock)
        with self.assertRaises(hybrid.ReferenceSearchStalled):
            resolver.resolve([.7, -.3])
        self.assertEqual([r["duration_s"] for r in timer.intervals], [3., 2.])
        self.assertEqual([r["completed"] for r in timer.intervals], [True, False])
        self.assertEqual(timer.path_s, 5.)

    def test_fast_reference_preserves_result_and_timer_resets_per_query(self):
        clock = Clock()
        resolver = resolver_fixture(clock, steps=2)
        timer, _ = self.timer(resolver, clock)
        self.assertEqual(resolver.resolve(None), {"passed": True, "proof": "unchanged"})
        self.assertEqual(timer.path_s, 2.)
        timer.begin(clock(), lambda _: None)
        self.assertEqual(timer.path_s, 0.)
        resolver.resolve(None)
        self.assertEqual(timer.path_s, 2.)

    def test_other_resolver_is_not_patched(self):
        clock = Clock()
        ours, other = resolver_fixture(clock), resolver_fixture(clock)
        self.timer(ours, clock)
        self.assertTrue(other.resolve(None)["passed"])
        self.assertFalse(hasattr(other, "_stalled_reference_clock"))

    def test_invalid_thresholds_clocks_and_double_install_fail(self):
        for budget in (0, -1, True, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                hybrid.ReferencePathClock(budget)
        clock = Clock()
        resolver = resolver_fixture(clock)
        timer, _ = self.timer(resolver, clock)
        with self.assertRaises(ValueError):
            hybrid.ReferencePathClock().install(resolver)
        for value in (99., float("nan"), float("inf")):
            clock.value = value
            with self.assertRaises(ValueError):
                timer.now()


class WrapperTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.directory = Path(temp.name)
        self.clock = Clock()
        self.layout = dict(id="test", bucket_xy_m=[.7, -.3])
        self.profile = SimpleNamespace(started=self.clock(), job_id="job", stage=lambda _: nullcontext())
        self.reference = fake_engine(self.directory, "reference")
        self.direct = fake_engine(self.directory, "direct_first")
        resolver = resolver_fixture(self.clock, endpoint_s=2.)
        self.timer = hybrid.ReferencePathClock(5., clock=self.clock).install(resolver)
        self.reference.resolve = resolver.resolve
        self.patches = [patch.object(hybrid, "_reference_engine", self.reference),
            patch.object(hybrid, "_direct_engine", self.direct),
            patch.object(hybrid, "_timer", self.timer),
            patch.object(hybrid, "_engine_inputs", ({}, "assets", self.directory, {})),
            patch.object(hybrid, "_selection_prefix", prefix_fixture),
            patch.object(workers, "_engine", None)]
        for value in self.patches:
            value.start()
            self.addCleanup(value.stop)

    def invoke(self, inner, budget=30.):
        self.snapshots = []
        self.inner_calls = []
        def run(layout, method, *args, **kwargs):
            self.inner_calls.append((workers._engine.mode, kwargs["deadline"]))
            return inner(layout, kwargs["publish"])
        with patch.object(hybrid, "incremental_fast_plan", run):
            return hybrid.incremental_stalled_reference_plan(self.layout, "warm_early",
                *([{}]*5), deadline=self.profile.started+budget, profile=self.profile,
                publish=lambda row: self.snapshots.append(row) or True)

    def result(self):
        return dict(job_id="job", layout=self.layout,
                    choice=self.snapshots[-1]["choice"] if self.snapshots else {"request": None},
                    accepted_snapshot_sha256="snapshot" if self.snapshots else None)

    def test_fast_reference_never_initializes_or_runs_direct_engine(self):
        def inner(layout, publish):
            publish(snapshot_fixture(layout, workers._engine.mode))
            return "early_good_shot"
        with patch.object(hybrid, "_get_direct_engine", side_effect=AssertionError("unexpected fallback")):
            self.assertEqual(self.invoke(inner), "early_good_shot")
        receipt = hybrid.read_receipt(self.directory, self.result())
        self.assertEqual(receipt["selected_adapter"], "reference")
        self.assertIsNone(receipt["fallback"])
        self.assertEqual(self.inner_calls, [("reference", 130.)])

    def test_stall_switches_only_before_publication_with_same_deadline(self):
        def inner(layout, publish):
            if workers._engine.mode == "reference":
                self.reference.resolve(None)
            publish(snapshot_fixture(layout, workers._engine.mode))
            return "early_good_shot"
        self.assertEqual(self.invoke(inner), "early_good_shot")
        receipt = hybrid.read_receipt(self.directory, self.result())
        self.assertEqual(self.inner_calls, [("reference", 130.), ("direct_first", 130.)])
        self.assertEqual(receipt["fallback"]["remaining_budget_s"], 23.)
        self.assertEqual(receipt["reference_path_s"], 5.)
        self.assertEqual(receipt["selected_adapter"], "direct_first")
        self.assertEqual(receipt["fallback"]["published_before_switch"], 0)
        self.assertFalse(receipt["fallback"]["reference_search_complete"])
        self.assertEqual(self.snapshots[0]["stalled_reference_snapshot"]["engine_mode"], "direct_first")
        self.assertEqual(receipt["event"], "query_completed")

    def test_changed_menu_cannot_be_published(self):
        def inner(layout, publish):
            snapshot = snapshot_fixture(layout, "reference")
            snapshot["generated"]["menu"] = [{"id": "changed"}]
            publish(snapshot)
        with self.assertRaisesRegex(ValueError, "frozen menu"):
            self.invoke(inner)
        self.assertEqual(self.snapshots, [])

    def test_switch_after_any_published_candidate_fails_closed(self):
        def inner(layout, publish):
            publish(snapshot_fixture(layout, "reference"))
            self.reference.resolve(None)
        with self.assertRaisesRegex(RuntimeError, "completed candidate"):
            self.invoke(inner)
        self.assertEqual(len(self.inner_calls), 1)

    def test_unrelated_error_cannot_trigger_fallback(self):
        with self.assertRaisesRegex(ValueError, "bad inputs"):
            self.invoke(lambda *_: (_ for _ in ()).throw(ValueError("bad inputs")))
        self.assertEqual(len(self.inner_calls), 1)

    def test_expired_original_budget_never_initializes_direct_engine(self):
        with patch.object(hybrid, "_get_direct_engine", side_effect=AssertionError("past deadline")):
            self.invoke(lambda *_: self.reference.resolve(None), budget=6.)
        receipt = hybrid.read_receipt(self.directory, self.result())
        self.assertIsNone(receipt["selected_adapter"])
        self.assertEqual(receipt["fallback"]["remaining_budget_s"], 0.)
        self.assertEqual(receipt["event"], "decision_budget_exhausted_before_fallback")

    def test_lazy_initialization_cost_uses_original_remaining_budget(self):
        def initialize():
            self.clock.advance(25.)
            return self.direct
        with patch.object(hybrid, "_direct_engine", None), patch.object(hybrid, "_get_direct_engine", initialize):
            self.invoke(lambda *_: self.reference.resolve(None))
        receipt = hybrid.read_receipt(self.directory, self.result())
        self.assertEqual(len(self.inner_calls), 1)
        self.assertEqual(receipt["direct_engine_initialization_s"], 25.)
        self.assertTrue(receipt["direct_engine_initialized_this_query"])
        self.assertEqual(receipt["event"], "decision_budget_exhausted_after_initialization")

    def test_receipt_detects_cross_job_and_file_tampering(self):
        self.invoke(lambda layout, publish: publish(snapshot_fixture(layout, "reference")))
        receipt = hybrid.read_receipt(self.directory, self.result())
        first = self.directory/receipt["receipt_files"][0]["file"]
        row = json.loads(first.read_text())
        row["layout_sha256"] = "changed"
        first.write_text(json.dumps(row))
        with self.assertRaisesRegex(ValueError, "binding"):
            hybrid.read_receipt(self.directory, self.result())

    def test_live_binding_resets_both_engines_and_lazy_destination(self):
        directory = self.directory/"requests"/"fresh"
        self.reference.adapter.resolver.base._resolved["prior"] = "evidence"
        self.direct.adapter.resolver.base._connected["prior"] = "evidence"
        hybrid.bind_live_request(directory)
        for engine in (self.reference, self.direct):
            self.assertEqual(engine.directory, directory)
            self.assertEqual(engine.adapter.resolver.base.directory, directory)
            self.assertEqual(engine.adapter.resolver.base._resolved, {})
            self.assertEqual(engine.adapter.resolver.base._connected, {})
        self.assertEqual(hybrid._engine_inputs[2], directory)
        with self.assertRaises(FileExistsError):
            hybrid.bind_live_request(directory)


def process_initializer(directory):
    """Mock only initialization/work, retain the real owned-worker deadline."""
    hybrid._reference_engine = fake_engine(directory, "reference")
    hybrid._direct_engine = fake_engine(directory, "direct_first")
    gate = SimpleNamespace(check=lambda *_: time.sleep(.006))
    planner = SimpleNamespace(plan=lambda *_: [gate.check(None) for _ in range(100)])
    resolver = SimpleNamespace(resolve=lambda *_: planner.plan(None), planner=planner,
                               solver=SimpleNamespace(gate=gate))
    hybrid._timer = hybrid.ReferencePathClock(.03).install(resolver)
    hybrid._reference_engine.resolve = resolver.resolve
    hybrid._engine_inputs = ({}, "unused", Path(directory), {})
    hybrid._selection_prefix = prefix_fixture
    hybrid.incremental_fast_plan = process_inner


def process_inner(layout, method, *_, publish, deadline, profile):
    if workers._engine.mode == "reference" and layout.get("stall"):
        hybrid._reference_engine.resolve(None)
    if workers._engine.mode == "direct_first" and layout.get("hang"):
        time.sleep(2.)
    publish(snapshot_fixture(layout, workers._engine.mode))
    return "early_good_shot"


class ProcessTests(unittest.TestCase):
    def test_one_owned_deadline_bounds_both_lanes_and_next_query_respawns(self):
        with tempfile.TemporaryDirectory() as directory:
            planner = hybrid.StalledReferencePlanner("warm_early", *([{}]*5),
                reference={}, assets="unused", directory=directory, engine_kwargs={},
                budget_s=.13, reference_path_budget_s=.03)
            self.addCleanup(planner.close)
            planner.worker.initializer = process_initializer
            planner.worker.initargs = (directory,)
            timed = planner.plan(dict(id="stalled", bucket_xy_m=[.1, -.1], stratum="test", stall=True, hang=True))
            self.assertEqual(timed["status"], "deadline")
            self.assertIsNone(timed["choice"]["request"])
            self.assertLess(timed["decision_latency_s"], .2)
            self.assertEqual(timed["stalled_reference"]["event"], "fallback_started")
            self.assertEqual(timed["stalled_reference"]["published_snapshots"], 0)
            self.assertIsNone(planner.worker.process)
            with self.assertRaises(ProcessLookupError):
                os.kill(timed["worker_pid"], 0)
            first = planner.plan(dict(id="fresh", bucket_xy_m=[.2, -.1], stratum="test"))
            second = planner.plan(dict(id="warm", bucket_xy_m=[.3, -.1], stratum="test"))
            self.assertEqual(first["worker_generation"], timed["worker_generation"]+1)
            self.assertEqual(first["stalled_reference"]["selected_adapter"], "reference")
            self.assertTrue(second["worker_reused"])
            self.assertEqual(first["worker_pid"], second["worker_pid"])
            self.assertEqual(second["stalled_reference"]["reference_path_intervals"], [])


class DispatchTests(unittest.TestCase):
    def test_only_known_frozen_context_formats_dispatch(self):
        self.assertEqual(hybrid.request_mode({"context_resolution_file": "resolutions/a.json"}), "reference")
        self.assertEqual(hybrid.request_mode({"context_resolution_file": "direct-contexts/a.json"}), "direct_first")
        for value in (None, {}, {"context_resolution_file": "../direct-contexts/a.json"},
                      {"context_resolution_file": "/resolutions/a.json"},
                      {"context_resolution_file": "fake/a.json"}):
            with self.assertRaises(ValueError):
                hybrid.request_mode(value)


if __name__ == "__main__":
    unittest.main()
