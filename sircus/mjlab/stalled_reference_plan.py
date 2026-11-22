"""Reference-first planning with a bounded, explicitly recorded escape hatch.

Only time in the reference resolver's path search counts toward the 5 s trigger.
Ordinary candidate IK, endpoint and path checks retain their historical rules.
After an interrupted reference search, the existing direct-first planner receives
the *remaining* original decision budget. No second worker or second 30 s clock
is created. All accepted choices still use the immutable warm_early snapshots.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path
import time
from types import MethodType

from .fast_planning_plan import ReusablePlanner, incremental_fast_plan
from .planning_budget_plan import _json_bytes

VERSION = "sircus-stalled-reference-fallback-v1"
REFERENCE_PATH_BUDGET_S = 5.0
FEATURES = ["sircus-vectorized-arm-broad-phase-v1",
            "sircus-arm-path-rejection-short-circuit-v1", VERSION]


def _digest(value):
    return hashlib.sha256(_json_bytes(value)).hexdigest()


def _finite_positive(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"Finite positive {name} required")
    return float(value)


class ReferenceSearchStalled(RuntimeError):
    """Cooperative interruption, never an exhaustive/no-solution certificate."""


class ReferencePathClock:
    """Per-instance guard armed only inside GeometryResolver.resolve -> plan.

    gate.check remains the original exact check; it is merely a cancellation
    point. Checks before/after it cannot turn a rejected path into an admission.
    """

    def __init__(self, budget_s=REFERENCE_PATH_BUDGET_S, *, clock=time.perf_counter):
        self.budget_s = _finite_positive(budget_s, "reference path budget")
        self.clock = clock
        self.started = None
        self.last_clock = None
        self.reference_depth = 0
        self.active = None
        self.intervals = []
        self.path_s = 0.
        self.interrupted = False
        self.notify = lambda event: None

    def begin(self, started, notify):
        if self.reference_depth or self.active is not None:
            raise RuntimeError("Reference clock still belongs to an active query")
        if not math.isfinite(started):
            raise ValueError("Finite decision start required")
        self.started, self.last_clock = float(started), float(started)
        self.intervals, self.path_s, self.interrupted = [], 0., False
        self.notify = notify

    def now(self):
        value = self.clock()
        if not math.isfinite(value) or self.last_clock is None or value < self.last_clock:
            raise ValueError("Nonfinite or nonmonotonic reference clock")
        self.last_clock = value
        return value

    def elapsed(self):
        return self.now()-self.started

    def state(self):
        return dict(reference_path_s=self.path_s,
                    reference_path_intervals=copy.deepcopy(self.intervals),
                    active_reference_path=copy.deepcopy(self.active),
                    reference_search_interrupted=self.interrupted)

    def check(self):
        if self.active is None:
            return
        elapsed = self.now()-self.started-self.active["start_s"]
        if self.path_s+elapsed >= self.budget_s:
            self.interrupted = True
            raise ReferenceSearchStalled("Reference path budget exhausted; search is incomplete")

    def install(self, resolver):
        """Wrap only this resolver/planner/gate, after speed optimizations."""
        if getattr(resolver, "_stalled_reference_clock", None) is not None:
            raise ValueError("Reference timer already installed")
        original_resolve = resolver.resolve
        original_plan = resolver.planner.plan
        original_check = resolver.solver.gate.check
        timer = self

        def resolve(_self, *args, **kwargs):
            timer.reference_depth += 1
            try:
                return original_resolve(*args, **kwargs)
            finally:
                timer.reference_depth -= 1

        def plan(_self, *args, **kwargs):
            if not timer.reference_depth:
                return original_plan(*args, **kwargs)
            if timer.active is not None:
                raise RuntimeError("Nested reference path search is unsupported")
            # Persist a start marker before work, so a hard deadline also leaves
            # an honest unfinished interval. This is instrumented path-phase
            # wall time, including its small start-marker publication cost.
            timer.active = dict(index=len(timer.intervals), start_s=timer.elapsed())
            timer.notify("reference_path_started")
            completed, was_stalled = False, False
            try:
                timer.check()
                result = original_plan(*args, **kwargs)
                timer.check()
                completed = True
                return result
            except ReferenceSearchStalled:
                was_stalled = True
                raise
            finally:
                end = timer.elapsed()
                interval = dict(timer.active, end_s=end,
                    duration_s=end-timer.active["start_s"], completed=completed,
                    interrupted=was_stalled)
                timer.path_s += interval["duration_s"]
                timer.intervals.append(interval)
                timer.active = None
                timer.notify("reference_path_interrupted" if was_stalled else "reference_path_finished")

        def check(_self, *args, **kwargs):
            timer.check()
            result = original_check(*args, **kwargs)
            timer.check()
            return result

        resolver.resolve = MethodType(resolve, resolver)
        resolver.planner.plan = MethodType(plan, resolver.planner)
        resolver.solver.gate.check = MethodType(check, resolver.solver.gate)
        resolver._stalled_reference_clock = self
        return self


class QueryReceipt:
    """Small immutable hash-chained events, durable even with no shot selected."""

    def __init__(self, directory, layout, profile, deadline, timer):
        self.directory = Path(directory)/"stalled-reference-receipts"/profile.job_id
        self.directory.mkdir(parents=True, exist_ok=False)
        self.timer, self.profile, self.deadline = timer, profile, float(deadline)
        self.binding = dict(version=VERSION, job_id=profile.job_id, layout_id=layout["id"],
            layout_sha256=_digest(layout), decision_budget_s=deadline-profile.started,
            reference_path_budget_s=timer.budget_s, method="warm_early",
            selection_uses_flight_labels=False,
            initialization_policy="Reference worker startup; first direct engine initialization consumes remaining decision time")
        self.sequence = 0
        self.previous = None
        self.mode = "reference"
        self.fallback = None
        self.direct_initialized = False
        self.initialization_s = 0.
        self.published = 0
        self.selection_prefix = None

    def emit(self, event):
        elapsed = self.timer.elapsed()
        row = dict(self.binding, sequence=self.sequence, event=event,
            elapsed_s=elapsed, engine_mode=self.mode,
            remaining_budget_s=max(0., self.deadline-self.profile.started-elapsed),
            fallback=copy.deepcopy(self.fallback),
            direct_engine_initialized_this_query=self.direct_initialized,
            direct_engine_initialization_s=self.initialization_s,
            published_snapshots=self.published,
            selection_prefix=copy.deepcopy(self.selection_prefix),
            previous_event_sha256=self.previous, **self.timer.state())
        raw = _json_bytes(row)
        path = self.directory/f"event-{self.sequence:05d}.json"
        # Atomic publication: a killed worker may leave a .tmp, never a partial
        # immutable JSON event. The caller ignores those uncommitted temporaries.
        temporary = path.with_suffix(".tmp")
        with temporary.open("xb") as stream:
            stream.write(raw)
            stream.flush()
        temporary.rename(path)
        self.previous = hashlib.sha256(raw).hexdigest()
        self.sequence += 1
        return row


_reference_engine = None
_direct_engine = None
_timer = None
_engine_inputs = None


def initialize_worker(reference, assets, directory, engine_kwargs,
                      reference_path_budget_s=REFERENCE_PATH_BUDGET_S):
    """CPU-only initializer; no CUDA/environment creation, and no global patch."""
    global _reference_engine, _direct_engine, _timer, _engine_inputs
    from . import parallel_capture_grid_workers as workers
    from .arm_check_speed import install_vectorized_gate
    from .arm_path_speed import install_path_short_circuit
    from .sweep_worker import atomic
    workers._initialize_worker(reference, assets, directory, engine_kwargs)
    _reference_engine, _direct_engine = workers._engine, None
    base = _reference_engine.adapter.resolver.base
    install_vectorized_gate(base.solver.gate)
    install_path_short_circuit(base.planner)
    _timer = ReferencePathClock(reference_path_budget_s).install(base)
    _engine_inputs = (reference, assets, Path(directory), copy.deepcopy(engine_kwargs))
    atomic(Path(directory)/"arm-check-features.json", dict(variant="hybrid", features=FEATURES))


def _get_direct_engine():
    global _direct_engine
    if _direct_engine is None:
        from .direct_path_first import DirectPathEngine
        reference, assets, directory, kwargs = _engine_inputs
        _direct_engine = DirectPathEngine(reference, assets, directory, **kwargs)
    return _direct_engine


def bind_live_request(directory):
    """Rebind both warm CPU engines to an exclusive request's empty evidence."""
    global _engine_inputs
    from .fast_planning_plan import _reset_query_state
    if _engine_inputs is None or _reference_engine is None:
        raise RuntimeError("Hybrid CPU worker has not been initialized")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    for engine in (_reference_engine, _direct_engine):
        if engine is None:
            continue
        _reset_query_state(engine)
        engine.directory = engine.adapter.directory = directory
        if "directory" in vars(engine.adapter.resolver):
            raise RuntimeError("Unexpected shadow resolver directory")
        engine.adapter.resolver.base.directory = directory
    reference, assets, _, kwargs = _engine_inputs
    _engine_inputs = (reference, assets, directory, kwargs)


def _selection_prefix(layout, common_args, profile):
    """Freeze the deterministic menu before any reference search is attempted.

    The historical selector computes the same finite menu itself. Rather than
    patch its globals, bind that deterministic prefix and verify every published
    inner snapshot against it. This small extra computation consumes the same
    decision budget and is included in the ordinary timing profile.
    """
    from .neural_shot_plan import menu
    from .neural_shot_model import order
    libraries, checkpoint, _, _, neural_checkpoint = common_args
    with profile.stage("menu"):
        proposals = menu(layout, libraries, checkpoint)
    with profile.stage("neural_order"):
        ordered = order(neural_checkpoint, proposals)
    return dict(menu_sha256=_digest(proposals), menu_count=len(proposals),
                ranked_proposal_ids=[row["id"] for row in ordered],
                ranked_proposal_ids_sha256=_digest([row["id"] for row in ordered]))


def incremental_stalled_reference_plan(layout, method, *common_args, publish, deadline, profile):
    """Preserve the inner selector and one absolute deadline across the switch."""
    from . import parallel_capture_grid_workers as workers
    if method != "warm_early":
        raise ValueError("Stalled-reference fallback is the warm_early selector only")
    if _reference_engine is None or _timer is None or _engine_inputs is None:
        raise RuntimeError("Hybrid CPU worker has not been initialized")
    _timer.begin(profile.started, lambda event: None)
    receipt = QueryReceipt(_engine_inputs[2], layout, profile, deadline, _timer)
    _timer.notify = receipt.emit
    receipt.emit("query_started")
    workers._engine = _reference_engine
    receipt.selection_prefix = _selection_prefix(layout, common_args, profile)
    receipt.emit("selection_prefix_frozen")

    def published(snapshot):
        prefix = receipt.selection_prefix
        if (_digest(snapshot["generated"]["menu"]) != prefix["menu_sha256"]
                or snapshot["ranked_proposal_ids"] != prefix["ranked_proposal_ids"]):
            raise ValueError("Hybrid rerun changed the frozen menu or neural ordering")
        result = publish(dict(snapshot, stalled_reference_snapshot=dict(
            version=VERSION, job_id=profile.job_id, engine_mode=receipt.mode,
            selection_prefix=copy.deepcopy(prefix), fallback=copy.deepcopy(receipt.fallback))))
        if result:
            receipt.published += 1
        return result

    try:
        result = incremental_fast_plan(layout, method, *common_args,
            publish=published, deadline=deadline, profile=profile)
    except ReferenceSearchStalled:
        # A switch only precedes the first complete candidate. This cannot erase
        # an already checked choice or use its result to change the search rule.
        if receipt.published:
            raise RuntimeError("Refusing a fallback after a completed candidate snapshot")
        now = _timer.now()
        receipt.fallback = dict(reason="reference_path_budget_exhausted",
            trigger_elapsed_s=now-profile.started, reference_path_s=_timer.path_s,
            threshold_s=_timer.budget_s, remaining_budget_s=max(0., deadline-now),
            published_before_switch=receipt.published, reference_search_complete=False)
        receipt.emit("fallback_triggered")
        if now >= deadline:
            receipt.emit("decision_budget_exhausted_before_fallback")
            return "completed"
        # Lazy initialization is intentionally charged to this same decision.
        initial = _timer.now()
        receipt.direct_initialized = _direct_engine is None
        workers._engine = _get_direct_engine()
        receipt.initialization_s = _timer.now()-initial
        receipt.mode = "direct_first"
        receipt.emit("fallback_started")
        if _timer.now() >= deadline:
            receipt.emit("decision_budget_exhausted_after_initialization")
            return "completed"
        result = incremental_fast_plan(layout, method, *common_args,
            publish=published, deadline=deadline, profile=profile)
    receipt.emit("query_completed")
    return result


def request_mode(request):
    """Native replay must choose its adapter from the frozen proof context."""
    path = request.get("context_resolution_file") if isinstance(request, dict) else None
    if not isinstance(path, str):
        raise ValueError("Prepared request lacks a frozen context resolution")
    parts = Path(path).parts
    if len(parts) != 2 or any(part in (".", "..") for part in parts):
        raise ValueError("Unsafe or unrecognized context resolution")
    if parts[0] == "direct-contexts":
        return "direct_first"
    if parts[0] == "resolutions":
        return "reference"
    raise ValueError("Unknown context proof format")


def read_receipt(directory, result):
    """Freeze/hash all committed events after the owned job has completed/stopped."""
    root = Path(directory)
    folder = root/"stalled-reference-receipts"/result["job_id"]
    if not folder.is_dir():
        identifier = result["layout"]["id"]
        if not isinstance(identifier, str) or Path(identifier).name != identifier or identifier in ("", ".", ".."):
            raise ValueError("Unsafe live request identifier")
        folder = root/"requests"/identifier/"stalled-reference-receipts"/result["job_id"]
    previous, events, paths = None, [], []
    for index, path in enumerate(sorted(folder.glob("event-*.json"))):
        raw = path.read_bytes()
        row = json.loads(raw)
        if (row.get("version") != VERSION or row.get("sequence") != index
                or row.get("job_id") != result["job_id"]
                or row.get("layout_sha256") != _digest(result["layout"])
                or row.get("previous_event_sha256") != previous):
            raise ValueError("Hybrid receipt binding or immutable event chain changed")
        previous = hashlib.sha256(raw).hexdigest()
        paths.append(dict(file=str(path.relative_to(root)), sha256=previous))
        events.append(row)
    if not events:
        raise ValueError("Hybrid worker left no durable query receipt")
    request = result.get("choice", {}).get("request")
    mode = request_mode(request) if request else None
    if mode is not None and mode != events[-1]["engine_mode"]:
        raise ValueError("Chosen proof adapter differs from frozen fallback mode")
    return dict(events[-1], events=events, receipt_files=paths, selected_adapter=mode,
                frozen_choice_sha256=_digest(result["choice"]),
                accepted_snapshot_sha256=result.get("accepted_snapshot_sha256"))


class StalledReferencePlanner(ReusablePlanner):
    """Reusable warm_early API with explicit per-query hybrid evidence."""

    def __init__(self, method, *args, reference_path_budget_s=REFERENCE_PATH_BUDGET_S, **kwargs):
        if method != "warm_early":
            raise ValueError("Stalled-reference fallback requires warm_early")
        threshold = _finite_positive(reference_path_budget_s, "reference path budget")
        super().__init__(method, *args, **kwargs)
        if threshold >= self.worker.budget_s:
            self.worker.close()
            raise ValueError("Reference path threshold must be less than the total decision budget")
        self.worker.function = incremental_stalled_reference_plan
        self.worker.initializer = initialize_worker
        self.worker.initargs = (*self.worker.initargs, threshold)

    def plan(self, layout):
        started = time.perf_counter()
        result = super().plan(layout)
        result["stalled_reference"] = read_receipt(self.directory, result)
        result["request_wall_s"] = time.perf_counter()-started
        return result


# Entry point used by the request-isolated live wrapper.
stalled_reference_plan = incremental_stalled_reference_plan
