"""Profile the frozen neural selector with owned, reusable CPU processes.

The decision clock starts after common initialization. A worker is reusable only
after a clean, on-time completion and verified immutable snapshots. A deadline or
exception destroys that worker. Reuse retains initialized geometry, never native
outcomes, query-specific IK/path proofs, or cached flight predictions.
"""
from __future__ import annotations

import copy
from contextlib import contextmanager
import hashlib
import json
import math
import multiprocessing
import os
from pathlib import Path
import threading
import time
import uuid

from .planning_budget_plan import (
    CachedRanker, DEFAULT_BUDGET_S, MAX_RECEIPT_BYTES, STARTUP_TIMEOUT_S,
    _empty_choice, _json_bytes, _send, _sha, _stop, _wrap_choice,
)

VERSION = "sircus-fast-planning-plan-v1"
METHODS = ("cold_30", "warm_30", "warm_early")
EARLY_SCORE_M = .005
STAGES = ("menu", "neural_order", "arm_setup_checks", "physics_ranking",
          "evidence_publication", "other")


class StageRecorder:
    """Exclusive intervals; a parent closes an interrupted interval at cutoff."""

    def __init__(self, send, started, job_id):
        self.send, self.started, self.job_id = send, started, job_id
        self.active = "other"
        self.events = [dict(stage="other", elapsed_s=0.)]

    def transition(self, stage):
        if stage not in STAGES:
            raise ValueError("Unknown profiling stage")
        event = dict(stage=stage, elapsed_s=max(0., time.perf_counter() - self.started))
        self.events.append(event)
        self.active = stage
        self.send(dict(kind="stage", job_id=self.job_id, **event))

    @contextmanager
    def stage(self, name):
        previous = self.active
        self.transition(name)
        try:
            yield
        finally:
            self.transition(previous)


def build_profile(events, observed_s, interrupted):
    """Turn the independently received event ledger into a contiguous partition."""
    if not math.isfinite(observed_s) or observed_s < 0:
        raise ValueError("Finite nonnegative profile duration required")
    clean = []
    previous = 0.
    for event in events:
        elapsed = float(event["elapsed_s"])
        if event["stage"] not in STAGES or not math.isfinite(elapsed) or elapsed < previous:
            raise ValueError("Invalid or nonmonotonic stage ledger")
        if elapsed <= observed_s:
            clean.append(dict(stage=event["stage"], elapsed_s=elapsed))
        previous = elapsed
    if not clean or clean[0] != dict(stage="other", elapsed_s=0.):
        raise ValueError("Stage ledger must begin at zero")
    intervals, seconds = [], dict.fromkeys(STAGES, 0.)
    for index, event in enumerate(clean):
        end = clean[index + 1]["elapsed_s"] if index + 1 < len(clean) else observed_s
        duration = end - event["elapsed_s"]
        intervals.append(dict(stage=event["stage"], start_s=event["elapsed_s"], end_s=end,
                              duration_s=duration, interrupted=bool(interrupted and index == len(clean)-1)))
        seconds[event["stage"]] += duration
    return dict(stage_seconds=seconds, blocks=intervals, events=clean,
                decision_observed_s=observed_s,
                interrupted_stage=clean[-1]["stage"] if interrupted else None,
                interrupted_stage_s=intervals[-1]["duration_s"] if interrupted else 0.,
                scope="Exclusive decision-clock wall time including unfinished active work; startup, parent verification and stopping measured separately")


def _worker(connection, function, common_args, initializer, initargs):
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[key] = "1"
    try:
        if initializer is not None:
            initializer(*initargs)
        _send(connection, dict(kind="ready", pid=os.getpid()))
        while True:
            instruction = json.loads(connection.recv_bytes(MAX_RECEIPT_BYTES))
            if instruction.get("kind") == "close":
                return
            if instruction.get("kind") != "start":
                raise ValueError("Expected explicit job start")
            started, deadline = instruction["started"], instruction["deadline"]
            job_id, journal = instruction["job_id"], Path(instruction["journal"])
            profile = StageRecorder(lambda row: _send(connection, row), started, job_id)
            sequence = 0

            def publish(snapshot):
                nonlocal sequence
                if time.perf_counter() >= deadline:
                    return False
                with profile.stage("evidence_publication"):
                    # Store the ledger before serialization, avoiding self-reference.
                    snapshot = dict(snapshot, profile_events_before_publication=copy.deepcopy(profile.events))
                    raw = _json_bytes(snapshot)
                    target = journal / f"snapshot-{sequence:05d}.json"
                    temporary = target.with_suffix(".tmp")
                    with temporary.open("xb") as handle:
                        handle.write(raw)
                        handle.flush()
                    if target.exists():
                        raise ValueError("Refusing to overwrite a planning snapshot")
                    os.replace(temporary, target)
                    elapsed = time.perf_counter() - started
                    if time.perf_counter() >= deadline:
                        return False
                    _send(connection, dict(kind="snapshot", job_id=job_id, sequence=sequence,
                        file=target.name, sha256=hashlib.sha256(raw).hexdigest(), elapsed_s=elapsed,
                        choice_sha256=_sha(snapshot["choice"]),
                        completed_proposals=len(snapshot["completed_proposal_ids"])))
                    sequence += 1
                return True

            reason = function(instruction["layout"], instruction["method"], *common_args,
                              publish=publish, deadline=deadline, profile=profile)
            _send(connection, dict(kind="done", job_id=job_id,
                                   elapsed_s=time.perf_counter()-started,
                                   return_reason=reason or "completed"))
    except BaseException as error:
        try:
            _send(connection, dict(kind="error", error_type=type(error).__name__, message=str(error)[:1500]))
        except (BrokenPipeError, EOFError, OSError):
            pass
    finally:
        connection.close()


class OwnedWorker:
    """Small generic process owner, also used by CPU-only lifecycle tests."""

    def __init__(self, function, common_args=(), *, initializer=None, initargs=(),
                 budget_s=DEFAULT_BUDGET_S, startup_timeout_s=STARTUP_TIMEOUT_S):
        for value in (budget_s, startup_timeout_s):
            if type(value) not in (float, int) or not math.isfinite(value) or value <= 0:
                raise ValueError("Positive finite timeout required")
        self.function, self.common_args = function, common_args
        self.initializer, self.initargs = initializer, initargs
        self.budget_s, self.startup_timeout_s = float(budget_s), float(startup_timeout_s)
        self.pool_id = uuid.uuid4().hex
        self.process = self.connection = None
        self.generation, self.job_index = -1, -1
        self.prior_status = None
        self.readiness_receipt = None
        self._call_lock = threading.Lock()

    def _discard(self):
        if self.process is not None:
            _stop(self.process)
        if self.connection is not None:
            self.connection.close()
        self.process = self.connection = None

    def close(self):
        self._discard()

    def _start(self):
        started = time.perf_counter()
        context = multiprocessing.get_context("spawn")
        parent, child = context.Pipe(duplex=True)
        process = context.Process(target=_worker,
            args=(child, self.function, self.common_args, self.initializer, self.initargs))
        process.start()
        child.close()
        self.process, self.connection = process, parent
        self.generation += 1
        self.job_index = -1
        try:
            if not parent.poll(self.startup_timeout_s):
                raise TimeoutError("Planning worker initialization exceeded its deadline")
            ready = json.loads(parent.recv_bytes(MAX_RECEIPT_BYTES))
            if ready.get("kind") != "ready" or ready.get("pid") != process.pid:
                raise RuntimeError(f"Planning worker initialization failed: {ready}")
            self.readiness_receipt = ready
        except BaseException:
            self._discard()
            self.prior_status = "error"
            raise
        return time.perf_counter() - started

    def run(self, layout, method, journal_directory):
        if not self._call_lock.acquire(blocking=False):
            raise RuntimeError("A planner lane accepts one job at a time")
        try:
            return self._run(layout, method, journal_directory)
        finally:
            self._call_lock.release()

    def _run(self, layout, method, journal_directory):
        request_started = time.perf_counter()
        if method not in METHODS:
            raise ValueError("Unknown fast-planning method")
        journal = Path(journal_directory)
        journal.mkdir(parents=True, exist_ok=False)
        reused = self.process is not None and self.process.is_alive() and self.prior_status == "completed"
        prior_status = self.prior_status
        if method == "cold_30" or not reused:
            self._discard()
            startup_s = self._start()
            reused = False
        else:
            startup_s = 0.
        process, connection = self.process, self.connection
        self.job_index += 1
        job_id = f"{self.pool_id}-{self.generation}-{self.job_index}"
        started = time.perf_counter()
        deadline = started + self.budget_s
        receipts, events = [], [dict(stage="other", elapsed_s=0.)]
        status, reason, error = "deadline", "deadline", None
        completion_receipt = None
        decision_s = self.budget_s
        try:
            _send(connection, dict(kind="start", layout=layout, method=method, job_id=job_id,
                                    journal=str(journal.resolve()), started=started, deadline=deadline))
            while True:
                remaining = deadline - time.perf_counter()
                if remaining <= 0 or not connection.poll(max(0., remaining)):
                    break
                message = json.loads(connection.recv_bytes(MAX_RECEIPT_BYTES))
                received = time.perf_counter()
                if received >= deadline:
                    break
                if message.get("kind") == "error":
                    raise RuntimeError(f"Planning worker failed: {message['error_type']}: {message['message']}")
                if message.get("job_id") != job_id:
                    raise ValueError("Planning receipt belongs to a different job")
                elapsed = message.get("elapsed_s")
                if type(elapsed) not in (float, int) or not math.isfinite(elapsed) or not 0 <= elapsed <= received-started:
                    raise ValueError("Invalid planning receipt timestamp")
                kind = message.get("kind")
                if kind == "stage":
                    if message.get("stage") not in STAGES or elapsed < events[-1]["elapsed_s"]:
                        raise ValueError("Nonmonotonic stage event")
                    events.append(dict(stage=message["stage"], elapsed_s=elapsed))
                elif kind == "snapshot":
                    if message.get("sequence") != len(receipts) or message.get("file") != f"snapshot-{len(receipts):05d}.json":
                        raise ValueError("Snapshot sequence mismatch")
                    receipts.append(dict(message, received_elapsed_s=received-started))
                elif kind == "done":
                    status, reason, decision_s = "completed", message.get("return_reason"), received-started
                    if reason not in ("completed", "early_good_shot"):
                        raise ValueError("Unexpected completed-worker return reason")
                    completion_receipt = dict(message, received_elapsed_s=decision_s)
                    break
                else:
                    raise ValueError("Unknown planning IPC message")
        except BaseException as caught:
            error = caught
        stop_started = time.perf_counter()
        if status != "completed" or method == "cold_30" or error is not None:
            self._discard()
        stop_s = time.perf_counter()-stop_started
        if error is not None:
            self.prior_status = "error"
            raise error
        wall_planning_s = time.perf_counter()-started
        verification_started = time.perf_counter()
        snapshot = None
        try:
            for receipt in receipts:
                raw = (journal / receipt["file"]).read_bytes()
                if hashlib.sha256(raw).hexdigest() != receipt["sha256"]:
                    raise ValueError("Planning snapshot hash mismatch")
                candidate = json.loads(raw)
                if _sha(candidate["choice"]) != receipt["choice_sha256"] or len(candidate["completed_proposal_ids"]) != receipt["completed_proposals"]:
                    raise ValueError("Planning snapshot contents do not match receipt")
                snapshot = candidate
            if status == "completed" and snapshot is None:
                raise ValueError("Completed planner did not publish a complete snapshot")
        except BaseException:
            self._discard()
            self.prior_status = "error"
            raise
        verification_s = time.perf_counter()-verification_started
        self.prior_status = status
        profile = build_profile(events, decision_s, status == "deadline")
        result = dict(version=VERSION, status=status, return_reason=reason,
            budget_s=self.budget_s, deadline_reached=status == "deadline",
            decision_latency_s=decision_s, startup_s=startup_s, stop_overhead_s=stop_s,
            receipt_verification_s=verification_s, wall_planning_s=wall_planning_s,
            worker_pid=process.pid, worker_exitcode=process.exitcode,
            worker_pool_id=self.pool_id, worker_generation=self.generation,
            worker_job_index=self.job_index, worker_reused=reused, prior_worker_status=prior_status,
            readiness_receipt=self.readiness_receipt if not reused else None,
            completion_receipt=completion_receipt,
            job_id=job_id, accepted_receipts=receipts,
            accepted_snapshot_file=receipts[-1]["file"] if receipts else None,
            accepted_snapshot_sha256=receipts[-1]["sha256"] if receipts else None,
            journal_directory=str(journal.resolve()), profile=profile, snapshot=snapshot,
            request_timing_scope="Parent call through verified snapshot receipt and result assembly; excludes caller result-file write, native drop and HTTP/UI latency")
        result["request_wall_s"] = time.perf_counter()-request_started
        return result


def _reset_query_state(engine):
    """Only immutable construction state survives; all per-query caches clear."""
    adapter, resolver = engine.adapter, engine.adapter.resolver
    if adapter.env is not None or resolver.pending:
        raise RuntimeError("Cannot reuse a worker with native or pending evaluation state")
    engine.prepared_ids.clear()
    engine._anchors.clear()
    adapter._ids.clear()
    adapter.packet = 0
    if hasattr(adapter, "orientation_by_id"):
        adapter.orientation_by_id.clear()
    resolver.connections.clear()
    resolver.base._connected.clear()
    resolver.base._resolved.clear()


def early_eligible(choice):
    from .clearance_expansion_plan import needs_expansion
    score = (choice.get("prediction") or {}).get("score_m")
    return (choice.get("request") is not None and type(score) in (float, int)
            and math.isfinite(score) and score >= EARLY_SCORE_M and not needs_expansion(choice))


def incremental_fast_plan(layout, method, libraries, source_checkpoint, history, ranker_config,
                          neural_checkpoint, *, publish, deadline, profile):
    """The previous neural selector, plus an explicit per-candidate early rule."""
    from . import parallel_capture_grid_workers as workers
    from . import neural_shot_plan as neural
    from .neural_shot_model import order
    from .clearance_expansion_plan import _empty, _ranker, _prepare_branches, needs_expansion, branch_proposals
    from .intergrid_candidate_replay_plan import command_key

    if method not in METHODS or workers._engine is None:
        raise ValueError("Initialized fast-planning CPU worker required")
    with profile.stage("other"):
        _reset_query_state(workers._engine)
    with profile.stage("physics_ranking"):
        # FlightRanker geometry can persist, but no per-shot prediction cache does.
        ranker = CachedRanker(_ranker(ranker_config))
    generated, completed, ranked_ids = {}, [], []
    plan = {key: _empty() for key in ("base", "fallback", "neural", "branches")}

    def choice():
        with profile.stage("physics_ranking"):
            admitted = list({command_key(row): row for name in ("neural", "branches")
                             for row in plan[name]["admitted"]}.values())
            selected = _wrap_choice(neural._choice(layout, admitted, ranker,
                                       len(plan["neural"]["proposals"]), ranked_ids), method)
            selected["predicted_flight_risk"] = needs_expansion(selected)
            return selected

    def snapshot(stage, selected=None):
        if workers._engine.adapter.env is not None:
            raise RuntimeError("CPU preparation allocated a native environment")
        selected = choice() if selected is None else selected
        with profile.stage("evidence_publication"):
            return dict(layout=layout, method=method, active_stage=stage, generated=copy.deepcopy(generated),
                        plan=copy.deepcopy(plan), completed_proposal_ids=list(completed),
                        ranked_proposal_ids=list(ranked_ids), choice=selected)

    def prepare(stage, rows, branch=False):
        if stage == "neural":
            generated.setdefault(stage, []).extend(rows)
        else:
            generated[stage] = rows
        for proposal in rows:
            if time.perf_counter() >= deadline:
                return "deadline"
            with profile.stage("arm_setup_checks"):
                if branch:
                    originals = {row["id"]: row for row in plan["neural"]["admitted"]}
                    batch = _prepare_branches(workers._engine, [proposal], originals, ranker_config["assets"])
                    admitted, records = batch["admitted"], batch["records"]
                else:
                    admitted, records = workers.prepare_proposals([proposal], True)
            if workers._engine.adapter.env is not None:
                raise RuntimeError("CPU preparation allocated a native environment")
            if time.perf_counter() >= deadline:
                return "deadline"
            plan[stage]["proposals"].append(proposal)
            plan[stage]["admitted"].extend(admitted)
            plan[stage]["records"].extend(records)
            completed.append(proposal["id"])
            selected = choice()
            early = method == "warm_early" and early_eligible(selected)
            if not publish(snapshot("early_good_shot" if early else stage, selected)):
                return "deadline"
            if early:
                return "early_good_shot"
        return None

    with profile.stage("menu"):
        proposals = neural.menu(layout, libraries, source_checkpoint)
    with profile.stage("neural_order"):
        ordered = order(neural_checkpoint, proposals)
    generated["menu"] = proposals
    ranked_ids.extend(row["id"] for row in ordered)
    for start in range(0, min(neural.CHECK_LIMIT, len(ordered)), neural.CHECK_BATCH):
        reason = prepare("neural", ordered[start:start+neural.CHECK_BATCH])
        if reason:
            return reason
        if not needs_expansion(choice()):
            break
    if needs_expansion(choice()) and plan["neural"]["admitted"]:
        branch_input = dict(base=plan["neural"], fallback=_empty(), expansion=_empty(), branches=_empty())
        with profile.stage("physics_ranking"):
            branches = branch_proposals(neural.proposal_layout(layout), branch_input, ranker)
        reason = prepare("branches", branches, branch=True)
        if reason:
            return reason
    if not publish(snapshot("finished")):
        return "deadline"
    return "completed"


class ReusablePlanner:
    """One lane per method, one active call at a time; the caller owns its pool."""

    def __init__(self, method, libraries, source_checkpoint, history, ranker_config,
                 neural_checkpoint, *, reference, assets, directory, engine_kwargs,
                 budget_s=DEFAULT_BUDGET_S, startup_timeout_s=STARTUP_TIMEOUT_S):
        if method not in METHODS:
            raise ValueError("Unknown fast-planning method")
        from .parallel_capture_grid_workers import _initialize_worker
        self.method, self.directory = method, Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.worker = OwnedWorker(incremental_fast_plan,
            (libraries, source_checkpoint, history, ranker_config, neural_checkpoint),
            initializer=_initialize_worker, initargs=(reference, assets, self.directory, engine_kwargs),
            budget_s=budget_s, startup_timeout_s=startup_timeout_s)
        # Prevent two lanes, processes, or resumed runs from silently sharing
        # method-specific resolver files. A new run requires a fresh directory.
        with (self.directory / "fast-planner-owner.json").open("x") as handle:
            json.dump(dict(version=VERSION, method=method, worker_pool_id=self.worker.pool_id), handle)
        self.seen_positions = set()

    def plan(self, layout):
        request_started = time.perf_counter()
        # Method-local evidence is never reused at a second position/request.
        xy = tuple(layout["bucket_xy_m"])
        if xy in self.seen_positions:
            raise ValueError("A lane requires unique bucket positions; use a new proof directory for repeats")
        identifier = layout["id"]
        if not isinstance(identifier, str) or not identifier or Path(identifier).name != identifier or identifier in (".", ".."):
            raise ValueError("Safe unique layout ID required")
        self.seen_positions.add(xy)
        result = self.worker.run(layout, self.method, self.directory / "fast-journals" / identifier)
        snapshot = result.pop("snapshot")
        result.update(layout=layout, method=self.method, choice=_empty_choice(layout, self.method),
            plan={key: dict(proposals=[], admitted=[], records=[]) for key in ("base", "fallback", "neural", "branches")},
            generated={}, completed_proposal_ids=[], ranked_proposal_ids=[], active_stage="no_complete_snapshot")
        if snapshot is not None:
            result.update(snapshot)
        result["choices"] = [result["choice"]]
        result["latency_s"] = result["decision_latency_s"]
        result["request_wall_s"] = time.perf_counter()-request_started
        return result

    def close(self):
        self.worker.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
