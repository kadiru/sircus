"""Independent original/board-geometry CPU lanes sharing one decision deadline.

The original hybrid is unchanged. A second spawned process starts only after a
verified risky original publication. Its startup consumes the remaining clock.
Final selection rechecks the alternate against the final original publication.
Native outcomes never enter either lane. This module does not activate a live
planner or allocate a native simulation environment.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import multiprocessing
from multiprocessing.connection import wait
import os
from pathlib import Path
import threading
import time
import uuid

from .board_fallback_plan import fallback_menu, needs_board_fallback
from .fast_planning_plan import CachedRanker, StageRecorder, build_profile, early_eligible
from .guarded_board_fallback_plan import RULE as GUARDED_RULE, replacement_decision
from .planning_budget_plan import (
    MAX_RECEIPT_BYTES, STARTUP_TIMEOUT_S, _empty_choice, _json_bytes, _send, _sha, _stop, _wrap_choice,
)

VERSION = "sircus-parallel-board-planner-v1"
RULE = dict(copy.deepcopy(GUARDED_RULE), version=VERSION,
    lanes="Fresh independent spawned CPU processes; no shared engine or resolver state",
    trigger="First verified original publication with checked request and predicted arm clearance below 5 mm or unavailable",
    clock="Original initialization precedes the shared decision clock; alternate startup consumes that same clock",
    selection="Recheck qualified alternate against final accepted original; preserve original on ties, invalid scores, or worker error",
    termination="Original is never interrupted by the alternate; stop after both lanes finish or the shared deadline",
    original_search="Unchanged hybrid runs concurrently; no soft handoff",
    timing_comparison="Prototype starts fresh original/alternate workers per query; baseline may reuse clean workers. Full request time includes this extra startup/CPU cost",
)


def _immutable(path, value):
    path = Path(path)
    raw = _json_bytes(value)
    with path.open("xb") as stream:
        stream.write(raw)
    return hashlib.sha256(raw).hexdigest()


def _process(connection, function, common_args, initializer, initargs):
    """One owned CPU job with small receipts and immutable on-disk snapshots."""
    cpu_started = time.process_time()
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[key] = "1"
    def send(row):
        _send(connection, dict(row, cpu_s=time.process_time()-cpu_started))
    try:
        if initializer is not None:
            initializer(*initargs)
        send(dict(kind="ready", pid=os.getpid()))
        instruction = json.loads(connection.recv_bytes(MAX_RECEIPT_BYTES))
        if instruction.get("kind") != "start":
            raise ValueError("Expected explicit shared-clock start")
        started, deadline = instruction["started"], instruction["deadline"]
        job_id, journal = instruction["job_id"], Path(instruction["journal"])
        profile = StageRecorder(send, started, job_id)
        sequence = 0
        def publish(snapshot):
            nonlocal sequence
            if time.perf_counter() >= deadline:
                return False
            with profile.stage("evidence_publication"):
                snapshot = dict(snapshot, profile_events_before_publication=copy.deepcopy(profile.events))
                raw = _json_bytes(snapshot)
                target = journal/f"snapshot-{sequence:05d}.json"
                temporary = target.with_suffix(".tmp")
                with temporary.open("xb") as stream:
                    stream.write(raw)
                    stream.flush()
                if target.exists():
                    raise ValueError("Refusing to overwrite a planning snapshot")
                os.replace(temporary, target)
                elapsed = time.perf_counter()-started
                if time.perf_counter() >= deadline:
                    return False
                send(dict(kind="snapshot", job_id=job_id, sequence=sequence,
                    file=target.name, sha256=hashlib.sha256(raw).hexdigest(), elapsed_s=elapsed,
                    choice_sha256=_sha(snapshot["choice"]),
                    completed_proposals=len(snapshot["completed_proposal_ids"])))
                sequence += 1
            return True
        reason = function(instruction["layout"], instruction["method"], *common_args,
            publish=publish, deadline=deadline, profile=profile)
        send(dict(kind="done", job_id=job_id, elapsed_s=time.perf_counter()-started,
                  return_reason=reason or "completed"))
    except BaseException as error:
        try:
            send(dict(kind="error", error_type=type(error).__name__, message=str(error)[:1000]))
        except (BrokenPipeError, EOFError, OSError):
            pass
    finally:
        connection.close()


class _Lane:
    def __init__(self, name, directory, layout, method, function, common_args, initializer, initargs):
        self.name, self.directory = name, Path(directory)
        self.journal = self.directory/"fast-journals"/layout["id"]
        self.journal.mkdir(parents=True, exist_ok=False)
        self.layout, self.method = layout, method
        self.job_id = uuid.uuid4().hex
        self.receipts, self.events = [], [dict(stage="other", elapsed_s=0.)]
        self.snapshot = self.ready = self.completion = self.error = None
        self.status, self.reason, self.cpu_s = "initializing", "initializing", 0.
        self.ready_at = self.launch_at = self.finished_at = None
        self.stop_s = self.verification_s = 0.
        self.closed_at = None
        context = multiprocessing.get_context("spawn")
        self.connection, child = context.Pipe(duplex=True)
        self.process = context.Process(target=_process,
            args=(child, function, common_args, initializer, initargs))
        self.launch_at = time.perf_counter()
        self.process.start()
        child.close()

    def start_job(self, started, deadline):
        self.status = "running"
        _send(self.connection, dict(kind="start", layout=self.layout, method=self.method,
            job_id=self.job_id, journal=str(self.journal.resolve()), started=started, deadline=deadline))

    def receive(self, started=None, deadline=None):
        try:
            message = json.loads(self.connection.recv_bytes(MAX_RECEIPT_BYTES))
        except EOFError:
            self.error = dict(error_type="UnexpectedEOF", message="Worker exited without completion")
            self.status, self.reason = "error", "worker_error"
            return None
        received = time.perf_counter()
        if deadline is not None and received >= deadline:
            return None
        cpu = message.get("cpu_s")
        if type(cpu) not in (float, int) or not math.isfinite(cpu) or cpu < self.cpu_s:
            raise ValueError("Invalid worker CPU ledger")
        self.cpu_s = float(cpu)
        kind = message.get("kind")
        if kind == "error":
            self.error = message
            self.status, self.reason, self.finished_at = "error", "worker_error", received
            return kind
        if kind == "ready":
            if self.ready is not None or message.get("pid") != self.process.pid:
                raise ValueError("Invalid worker readiness")
            self.ready, self.ready_at = message, received
            return kind
        if self.ready is None or message.get("job_id") != self.job_id:
            raise ValueError("Planning receipt belongs to a different job")
        elapsed = message.get("elapsed_s")
        if type(elapsed) not in (float, int) or not math.isfinite(elapsed) or not 0 <= elapsed <= received-started:
            raise ValueError("Invalid shared-clock timestamp")
        if kind == "stage":
            if message.get("stage") not in self.events_stage_names or elapsed < self.events[-1]["elapsed_s"]:
                raise ValueError("Invalid profiling event")
            self.events.append(dict(stage=message["stage"], elapsed_s=elapsed))
        elif kind == "snapshot":
            sequence = len(self.receipts)
            if message.get("sequence") != sequence or message.get("file") != f"snapshot-{sequence:05d}.json":
                raise ValueError("Snapshot sequence mismatch")
            before = time.perf_counter()
            raw = (self.journal/message["file"]).read_bytes()
            snapshot = json.loads(raw)
            if (hashlib.sha256(raw).hexdigest() != message["sha256"]
                    or _sha(snapshot["choice"]) != message["choice_sha256"]
                    or len(snapshot["completed_proposal_ids"]) != message["completed_proposals"]):
                raise ValueError("Immutable planning receipt mismatch")
            after = time.perf_counter()
            self.verification_s += after-before
            if after >= deadline:
                return None
            self.receipts.append(dict(message, received_elapsed_s=received-started,
                                      validated_elapsed_s=after-started))
            self.snapshot = snapshot
        elif kind == "done":
            if message.get("return_reason") not in ("completed", "early_good_shot", "deadline"):
                raise ValueError("Unexpected worker completion")
            self.completion = dict(message, received_elapsed_s=received-started)
            self.status, self.reason, self.finished_at = "completed", message["return_reason"], received
        else:
            raise ValueError("Unknown planning IPC message")
        return kind

    @property
    def events_stage_names(self):
        from .fast_planning_plan import STAGES
        return STAGES

    def close(self):
        before = time.perf_counter()
        _stop(self.process)
        self.connection.close()
        self.stop_s = time.perf_counter()-before
        self.closed_at = time.perf_counter()

    def result(self, started, decision_s, budget_s):
        snapshot = self.snapshot or dict(layout=self.layout, method=self.method,
            choice=_empty_choice(self.layout,self.method), active_stage="no_complete_snapshot",
            plan={key:dict(proposals=[],admitted=[],records=[]) for key in ("base","fallback","neural","branches")},
            generated={}, completed_proposal_ids=[], ranked_proposal_ids=[])
        observed = min(decision_s, self.finished_at-started) if self.finished_at is not None else decision_s
        status = self.status if self.status in ("completed","error") else "deadline"
        result = dict(version=VERSION, status=status, return_reason=self.reason if status != "deadline" else "deadline",
            budget_s=budget_s, deadline_reached=status=="deadline", decision_latency_s=observed,
            startup_s=(self.ready_at or started+decision_s)-self.launch_at,
            launch_offset_s=self.launch_at-started,
            readiness_offset_s=self.ready_at-started if self.ready_at is not None else None,
            available_budget_at_launch_s=max(0., min(budget_s,started+budget_s-self.launch_at)),
            stop_overhead_s=self.stop_s, receipt_verification_s=0.,
            online_receipt_verification_s=self.verification_s,
            receipt_verification_scope="Hashes verified online inside the shared decision clock; no post-decision verification phase",
            wall_planning_s=self.closed_at-started,
            request_wall_s=self.closed_at-self.launch_at,latency_s=observed,
            worker_pid=self.process.pid, worker_exitcode=self.process.exitcode,
            worker_pool_id=self.job_id, worker_generation=0, worker_job_index=0,
            worker_reused=False, prior_worker_status=None, lifecycle="fresh_worker_per_request",
            readiness_receipt=self.ready, completion_receipt=self.completion, error=self.error,
            job_id=self.job_id, accepted_receipts=self.receipts,
            accepted_snapshot_file=self.receipts[-1]["file"] if self.receipts else None,
            accepted_snapshot_sha256=self.receipts[-1]["sha256"] if self.receipts else None,
            journal_directory=str(self.journal.resolve()),
            profile=build_profile(self.events,observed,status=="deadline"),
            cpu_s_last_receipt=self.cpu_s, cpu_measurement_complete=status=="completed", **snapshot)
        result["choices"] = [result["choice"]]
        return result


def select_final(original, alternative, *, worker_error=False):
    """The trigger only starts work; it never substitutes for the final original."""
    decision = replacement_decision((alternative or {}).get("prediction"), original.get("prediction"))
    accepted = (not worker_error and original.get("request") is not None
                and alternative is not None and alternative.get("request") is not None and decision["accepted"])
    return dict(original_choice_sha256=_sha(original),
        alternative_choice_sha256=_sha(alternative) if alternative is not None else None,
        replacement_decision=decision, worker_error=bool(worker_error),
        selected_lane="alternative" if accepted else "original")


def supervise_parallel(layout, method, common_args, *, directory, original_function,
                       original_initializer=None, original_initargs=(), alternative_factory,
                       budget_s=30., startup_timeout_s=STARTUP_TIMEOUT_S):
    """Auditable two-process owner; injectable CPU functions support lifecycle tests."""
    for value in (budget_s,startup_timeout_s):
        if type(value) not in (float,int) or not math.isfinite(value) or value<=0:
            raise ValueError("Finite positive planning timeout required")
    request_started = time.perf_counter()
    root = Path(directory); root.mkdir(parents=True,exist_ok=True)
    original = _Lane("original",root/"original",layout,method,original_function,common_args,
                     original_initializer,original_initargs)
    alternate = None
    started = trigger = None
    try:
        if not original.connection.poll(startup_timeout_s):
            raise TimeoutError("Original initialization timed out")
        if original.receive() != "ready":
            raise RuntimeError("Original initialization failed")
        started = time.perf_counter(); deadline = started+budget_s
        original.start_job(started,deadline)
        while time.perf_counter() < deadline:
            lanes = [lane for lane in (original,alternate)
                     if lane is not None and lane.status not in ("completed","error")]
            if not lanes:
                break
            ready = wait([lane.connection for lane in lanes],timeout=max(0.,min(.05,deadline-time.perf_counter())))
            for lane in lanes:
                if lane.connection not in ready or time.perf_counter() >= deadline:
                    continue
                kind = lane.receive(started,deadline)
                if kind == "ready":
                    lane.start_job(started,deadline)
                if (lane is original and kind == "snapshot" and alternate is None
                        and needs_board_fallback(original.snapshot["choice"])):
                    receipt = original.receipts[-1]
                    source = copy.deepcopy(original.snapshot["choice"])
                    trigger = dict(original_receipt_sequence=receipt["sequence"],file=receipt["file"],
                        sha256=receipt["sha256"],choice_sha256=receipt["choice_sha256"],
                        elapsed_s=time.perf_counter()-started, selection_uses_flight_labels=False)
                    proof = root/"parallel-triggers"/(layout["id"]+".json")
                    proof.parent.mkdir(parents=True,exist_ok=True)
                    trigger["binding_file"] = str(proof.relative_to(root))
                    trigger["binding_sha256"] = _immutable(proof,dict(trigger=trigger,source_choice=source))
                    if time.perf_counter() >= deadline:
                        break
                    spec = alternative_factory(source,root/"original",root/"alternative")
                    if time.perf_counter() >= deadline:
                        break
                    alternate = _Lane("alternative",root/"alternative",layout,method,
                        spec["function"],spec["common_args"],spec.get("initializer"),spec.get("initargs",()))
        decision_s = min(budget_s,time.perf_counter()-started)
    finally:
        for lane in (original,alternate):
            if lane is not None:
                lane.close()
    stop_finished = time.perf_counter()
    primary = original.result(started,decision_s,budget_s)
    secondary = alternate.result(started,decision_s,budget_s) if alternate is not None else None
    # Alternative publications initially carry the trigger choice; only a newly
    # admitted and selected alternate can compete in the final comparison.
    candidate = (secondary["choice"] if secondary is not None
        and secondary.get("board_alternative",{}).get("selected_candidate_id") else None)
    selection = select_final(primary["choice"],candidate,
        worker_error=primary["error"] is not None or (secondary is not None and secondary["error"] is not None))
    selected = primary if selection["selected_lane"]=="original" else secondary
    result = dict(version=VERSION,protocol=copy.deepcopy(RULE),layout=layout,method=method,
        status="deadline" if decision_s>=budget_s else "completed",budget_s=float(budget_s),
        deadline_reached=decision_s>=budget_s,decision_latency_s=decision_s,
        startup_s=original.ready_at-original.launch_at,stop_overhead_s=stop_finished-started-decision_s,
        trigger=trigger,selection=selection,selected_lane=selection["selected_lane"],
        selected_adapter_directory=selection["selected_lane"],choice=copy.deepcopy(selected["choice"]),
        lanes=dict(original=primary,alternative=secondary),
        shared_clock=dict(started=started,deadline=deadline,cutoff_elapsed_s=decision_s,
                          original_interrupted_by_alternative=False),
        cpu_s_last_receipts=original.cpu_s+(alternate.cpu_s if alternate else 0.))
    path = root/"parallel-selections"/(layout["id"]+".json")
    path.parent.mkdir(parents=True,exist_ok=True)
    result["selection_receipt_file"] = str(path.relative_to(root))
    result["selection_receipt_sha256"] = _immutable(path,dict(selection=selection,trigger=trigger,
        decision_latency_s=decision_s,budget_s=budget_s,
        original_snapshot_sha256=primary["accepted_snapshot_sha256"],
        alternative_snapshot_sha256=secondary["accepted_snapshot_sha256"] if secondary else None))
    result["request_wall_s"] = time.perf_counter()-request_started
    return result


_alternate_engine = None
_alternate_context_manifest = None


def initialize_alternative(reference, assets, directory, engine_kwargs, source, source_directory):
    """Initialize private CPU geometry from one immutable original context."""
    global _alternate_engine, _alternate_context_manifest
    from . import parallel_capture_grid_workers as workers
    from .stalled_reference_plan import request_mode
    from .direct_path_first import initialize_direct_first, hydrate_direct_resolution
    from .arm_check_speed import install_vectorized_gate
    from .arm_path_speed import install_path_short_circuit
    from .table_correction_native import copy_bound
    from .benchmark_verify import sha
    request = source["request"]
    mode = request_mode(request)
    if mode == "direct_first":
        initialize_direct_first(reference,assets,directory,engine_kwargs)
    else:
        workers._initialize_worker(reference,assets,directory,engine_kwargs)
        base = workers._engine.adapter.resolver.base
        install_vectorized_gate(base.solver.gate)
        install_path_short_circuit(base.planner)
    _alternate_engine = workers._engine
    relative = request["context_resolution_file"]
    expected = request["context_resolution_sha256"]
    copy_bound(source_directory,directory,relative,expected)
    _alternate_context_manifest = dict(adapter_mode=mode,source_directory=str(source_directory),
        destination_directory=str(directory),files=[dict(file=relative,expected_sha256=expected,
            source_sha256=sha(Path(source_directory)/relative),destination_sha256=sha(Path(directory)/relative))])
    if mode == "direct_first":
        hydrate_direct_resolution(_alternate_engine,request["bucket_xy_m"])
    elif workers.load_resolution(_alternate_engine,request["bucket_xy_m"]) is None:
        raise ValueError("Copied original context was not hydrated")
    anchor,_ = _alternate_engine.adapter.resolve_context(request)
    if anchor["sha256"] != expected or _alternate_engine.adapter.env is not None:
        raise ValueError("Alternate context changed or allocated native simulation")
    path = Path(directory)/"parallel-context-copies"/(request["context_id"]+".json")
    path.parent.mkdir(parents=True,exist_ok=True)
    _immutable(path,_alternate_context_manifest)


def incremental_alternative(layout, method, ranker_config, source, *, publish, deadline, profile):
    from .board_pose_recovery_study import nominal_prediction,wrist_screen,diversified
    from .clearance_expansion_plan import _ranker,needs_expansion
    engine = _alternate_engine
    if engine is None:
        raise RuntimeError("Private alternate engine was not initialized")
    best = copy.deepcopy(source)
    state = dict(version=VERSION,protocol=copy.deepcopy(RULE),source_choice=copy.deepcopy(source),
        source_choice_sha256=_sha(source),context_copy_manifest=copy.deepcopy(_alternate_context_manifest),
        bounds=dict(low=list(map(float,engine.adapter.resolver.solver.low)),
                    high=list(map(float,engine.adapter.resolver.solver.high))),
        proposals=[],nominal_predictions={},wrist_screen={},check_order=[],records=[],admitted=[],
        predictions={},replacement_decisions=[],selected_candidate_id=None,completed=False)
    def snapshot(stage):
        return dict(layout=layout,method=method,active_stage=stage,choice=copy.deepcopy(best),
            board_alternative=copy.deepcopy(state),completed_proposal_ids=[r["proposal_id"] for r in state["records"]],
            ranked_proposal_ids=list(state["check_order"]),generated={},
            plan={key:dict(proposals=[],admitted=[],records=[]) for key in ("base","fallback","neural","branches")})
    if not publish(snapshot("alternate_initialized")):
        return "completed"
    with profile.stage("physics_ranking"):
        ranker = CachedRanker(_ranker(ranker_config))
    with profile.stage("menu"):
        state["proposals"] = fallback_menu(layout,source["request"],state["bounds"]["low"],state["bounds"]["high"])
    eligible = []
    with profile.stage("physics_ranking"):
        for proposal in state["proposals"]:
            if time.perf_counter() >= deadline:
                return "completed"
            screen = wrist_screen(ranker_config["scene"],proposal,engine.adapter.resolver.models)
            state["wrist_screen"][proposal["id"]] = screen
            if screen["passed"]:
                state["nominal_predictions"][proposal["id"]] = nominal_prediction(ranker.ranker,ranker_config["scene"],proposal)
                eligible.append(proposal)
    shortlist = diversified(eligible,state["nominal_predictions"],RULE["full_checks"])
    state["check_order"] = [r["id"] for r in shortlist]
    if not publish(snapshot("alternate_menu")):
        return "completed"
    rejected = []
    def prepare(proposal,multistart=False):
        nonlocal best
        if time.perf_counter() >= deadline:
            return False
        with profile.stage("arm_setup_checks"):
            admitted,records = engine.prepare([proposal],one_per_setting=not multistart)
        if engine.adapter.env is not None:
            raise RuntimeError("Alternate CPU checks allocated native simulation")
        if time.perf_counter() >= deadline:
            return False
        decisions = []
        with profile.stage("physics_ranking"):
            predictions = {r["id"]:ranker.predict(r) for r in admitted}
            for request in admitted:
                prediction = predictions[request["id"]]
                decision = replacement_decision(prediction,best.get("prediction"))
                decisions.append(dict(request_id=request["id"],**decision))
                if decision["accepted"]:
                    best = copy.deepcopy(source)
                    best.update(prediction=prediction,candidate_pool="parallel_board_geometry",
                        intended_route=request["intended_route"],request=copy.deepcopy(request),
                        available_routes=sorted(set(source.get("available_routes",[]))|{request["intended_route"]}))
                    best = _wrap_choice(best,method)
                    best["predicted_flight_risk"] = needs_expansion(best)
                    state["selected_candidate_id"] = request["id"]
        if time.perf_counter() >= deadline:
            return False
        state["records"].extend(records); state["admitted"].extend(admitted)
        state["predictions"].update(predictions);state["replacement_decisions"].extend(decisions)
        early = state["selected_candidate_id"] is not None and early_eligible(best)
        state["completed"] = bool(early)
        if not publish(snapshot("early_good_shot" if early else "alternate_check")):
            return False
        if early:
            return "early_good_shot"
        if not admitted and not multistart:
            rejected.append(proposal)
        return True
    for proposal in shortlist:
        status = prepare(proposal)
        if status == "early_good_shot":
            return status
        if not status:
            return "completed"
    for original in rejected[:RULE["multistart_retry_limit"]]:
        proposal = dict(copy.deepcopy(original),id=original["id"]+"-multistart",retry_of=original["id"],ik_search="multistart16")
        state["proposals"].append(proposal);state["check_order"].append(proposal["id"])
        status = prepare(proposal,True)
        if status == "early_good_shot":
            return status
        if not status:
            return "completed"
    state["completed"] = True
    publish(snapshot("finished"))
    return "completed"


class ParallelBoardPlanner:
    """Reusable API; each query owns fresh, isolated process/evidence lanes."""
    def __init__(self,method,libraries,source_checkpoint,history,ranker_config,neural_checkpoint,
                 *,reference,assets,directory,engine_kwargs,budget_s=30.,startup_timeout_s=STARTUP_TIMEOUT_S):
        if method != "warm_early":
            raise ValueError("Parallel board planning requires warm_early")
        self.method,self.directory = method,Path(directory)
        self.directory.mkdir(parents=True,exist_ok=True)
        self.common_args = (libraries,source_checkpoint,history,ranker_config,neural_checkpoint)
        self.reference,self.assets,self.engine_kwargs = reference,assets,engine_kwargs
        self.budget_s,self.startup_timeout_s = budget_s,startup_timeout_s
        self.seen_positions = set()
        self.call_lock = threading.Lock()
        _immutable(self.directory/"parallel-planner-owner.json",dict(version=VERSION,method=method,protocol=RULE))

    def plan(self,layout):
        from .stalled_reference_plan import initialize_worker,incremental_stalled_reference_plan,read_receipt
        if not self.call_lock.acquire(blocking=False):
            raise RuntimeError("Only one active request per parallel planner")
        try:
            identifier = layout["id"]
            if not isinstance(identifier,str) or not identifier or Path(identifier).name != identifier or identifier in (".",".."):
                raise ValueError("Safe unique layout ID required")
            xy = tuple(layout["bucket_xy_m"])
            if xy in self.seen_positions:
                raise ValueError("Unique target positions required; use new evidence for repeats")
            self.seen_positions.add(xy)
            def factory(source,original_directory,alternative_directory):
                return dict(function=incremental_alternative,common_args=(self.common_args[3],source),
                    initializer=initialize_alternative,initargs=(self.reference,self.assets,alternative_directory,
                    self.engine_kwargs,source,original_directory))
            result = supervise_parallel(layout,self.method,self.common_args,directory=self.directory,
                original_function=incremental_stalled_reference_plan,original_initializer=initialize_worker,
                original_initargs=(self.reference,self.assets,self.directory/"original",self.engine_kwargs),
                alternative_factory=factory,budget_s=self.budget_s,startup_timeout_s=self.startup_timeout_s)
            primary = result["lanes"]["original"]
            if primary["accepted_receipts"]:
                primary["stalled_reference"] = read_receipt(self.directory/"original",primary)
            return result
        finally:
            self.call_lock.release()

    def close(self):
        # supervise_parallel owns and closes both processes on every exit.
        pass

    def __enter__(self):
        return self

    def __exit__(self,*_):
        self.close()
