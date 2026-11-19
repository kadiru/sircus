"""Give frozen planners a hard wall-time budget and keep complete choices only.

The supervisor never kills a shared worker or interrupts native simulation.
Each call owns one disposable CPU process; setup happens before its deadline.
Small pipe receipts refer to immutable JSON snapshots so a large partial pipe
write cannot block deadline enforcement. Unfinished/late proposals never release.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import multiprocessing
import os
from pathlib import Path
import tempfile
import time

VERSION = "sircus-planning-budget-plan-v1"
METHODS = ("current_planner", "neural_selector")
DEFAULT_BUDGET_S = 30.
STARTUP_TIMEOUT_S = 180.
STOP_GRACE_S = .5
MAX_RECEIPT_BYTES = 4096


def _json_bytes(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _sha(value):
    return hashlib.sha256(_json_bytes(value)).hexdigest()


def _send(connection, value):
    raw = _json_bytes(value)
    if len(raw) > MAX_RECEIPT_BYTES:
        raise ValueError("Budget IPC receipts must remain small")
    connection.send_bytes(raw)


def _child(connection, function, args, kwargs, initializer, initargs, journal):
    """Child entrypoint: no inherited or allocated CUDA runtime is permitted."""
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[name] = "1"
    try:
        if initializer is not None:
            initializer(*initargs)
        _send(connection, dict(kind="ready", pid=os.getpid()))
        instruction = json.loads(connection.recv_bytes(MAX_RECEIPT_BYTES))
        if instruction.get("kind") != "start":
            raise ValueError("Expected explicit planning start")
        started, deadline = instruction["started"], instruction["deadline"]
        sequence = 0

        def publish(snapshot):
            nonlocal sequence
            if time.perf_counter() >= deadline:
                return False
            # Complete candidate and ranking evidence is serialized before a
            # receipt is sent. Atomic rename never exposes half a JSON object.
            raw = _json_bytes(snapshot)
            target = Path(journal) / f"snapshot-{sequence:05d}.json"
            temporary = target.with_suffix(".tmp")
            with temporary.open("xb") as handle:
                handle.write(raw)
                handle.flush()
            os.replace(temporary, target)
            elapsed = time.perf_counter() - started
            if time.perf_counter() >= deadline:
                return False
            receipt = dict(kind="snapshot", sequence=sequence, file=target.name,
                           sha256=hashlib.sha256(raw).hexdigest(), elapsed_s=elapsed,
                           choice_sha256=_sha(snapshot["choice"]),
                           completed_proposals=len(snapshot["completed_proposal_ids"]))
            _send(connection, receipt)
            sequence += 1
            return True

        function(*args, publish=publish, deadline=deadline, **kwargs)
        _send(connection, dict(kind="done", elapsed_s=time.perf_counter() - started))
    except BaseException as error:
        # Exception text is bounded; no arbitrary object/command dump in IPC.
        try:
            _send(connection, dict(kind="error", error_type=type(error).__name__,
                                   message=str(error)[:1500]))
        except (BrokenPipeError, EOFError, OSError):
            pass
    finally:
        connection.close()


def _stop(process):
    if process.is_alive():
        process.terminate()
        process.join(STOP_GRACE_S)
    if process.is_alive():
        process.kill()
        process.join(STOP_GRACE_S)
    if process.is_alive():
        raise RuntimeError("Owned planning worker could not be stopped")
    process.join()


def supervise(function, args=(), kwargs=None, *, initializer=None, initargs=(),
              journal_directory=None, budget_s=DEFAULT_BUDGET_S,
              startup_timeout_s=STARTUP_TIMEOUT_S):
    """Run a callback with a hard deadline; return only received complete snapshots.

    The callback accepts ``publish(snapshot), deadline``. The initialization
    phase is excluded and reported separately. Production callers supply a new
    journal directory; tests may use temporary storage. Never call from a daemon
    process. A ThreadPoolExecutor can supervise independent jobs concurrently.
    """
    for name, value in (("budget", budget_s), ("startup timeout", startup_timeout_s)):
        if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
            raise ValueError(name + " must be finite and positive")
    temporary = tempfile.TemporaryDirectory(prefix="sircus-budget-") if journal_directory is None else None
    journal = Path(temporary.name if temporary else journal_directory)
    if temporary is None:
        journal.mkdir(parents=True, exist_ok=False)
    context = multiprocessing.get_context("spawn")
    parent, child = context.Pipe(duplex=True)
    process = context.Process(target=_child,
                              args=(child, function, args, kwargs or {}, initializer, initargs, str(journal)))
    launched = time.perf_counter()
    started = None
    receipts, finished, reached = [], False, False
    try:
        process.start()
        child.close()
        startup_deadline = launched + startup_timeout_s
        while not parent.poll(.02):
            if not process.is_alive():
                raise RuntimeError("Planning worker exited before readiness")
            if time.perf_counter() >= startup_deadline:
                raise RuntimeError("Planning worker initialization timed out")
        ready = json.loads(parent.recv_bytes(MAX_RECEIPT_BYTES))
        if ready.get("kind") != "ready":
            raise RuntimeError("Planning initialization failed: " + str(ready))
        started = time.perf_counter()
        deadline = started + budget_s
        _send(parent, dict(kind="start", started=started, deadline=deadline))
        while True:
            remaining = deadline - time.perf_counter()
            if remaining <= 0:
                reached = True
                break
            if not parent.poll(min(.02, remaining)):
                if not process.is_alive():
                    raise RuntimeError("Planning worker exited without completion receipt")
                continue
            message = json.loads(parent.recv_bytes(MAX_RECEIPT_BYTES))
            received = time.perf_counter() - started
            if received > budget_s:
                reached = True
                break
            kind = message.get("kind")
            if kind == "error":
                raise RuntimeError("Planning worker failed: " + str(message))
            if kind == "done":
                if not 0 <= message["elapsed_s"] <= budget_s:
                    raise ValueError("Late completion receipt")
                finished = True
                break
            if kind != "snapshot" or message.get("sequence") != len(receipts):
                raise ValueError("Unexpected planning snapshot receipt")
            if not 0 <= message["elapsed_s"] <= received <= budget_s:
                raise ValueError("Invalid planning snapshot timing")
            receipts.append(dict(message, received_elapsed_s=received))
        decision = min(time.perf_counter() - started, budget_s)
        stop_started = time.perf_counter()
        _stop(process)
        stopped = time.perf_counter()
        snapshot = None
        for receipt in receipts:
            name = receipt["file"]
            if name != f"snapshot-{receipt['sequence']:05d}.json":
                raise ValueError("Unexpected planning snapshot path")
            raw = (journal / name).read_bytes()
            if hashlib.sha256(raw).hexdigest() != receipt["sha256"]:
                raise ValueError("Planning snapshot changed after publication")
            value = json.loads(raw)
            if (_sha(value["choice"]) != receipt["choice_sha256"] or
                    len(value["completed_proposal_ids"]) != receipt["completed_proposals"]):
                raise ValueError("Planning receipt and complete snapshot disagree")
            snapshot = value
        return dict(version=VERSION, status="completed" if finished else "deadline",
                    budget_s=float(budget_s), deadline_reached=reached,
                    decision_latency_s=decision, wall_planning_s=stopped - started,
                    startup_s=started - launched, stop_overhead_s=stopped - stop_started,
                    worker_pid=process.pid, worker_exitcode=process.exitcode,
                    accepted_receipts=receipts, accepted_snapshot_file=receipts[-1]["file"] if receipts else None,
                    accepted_snapshot_sha256=receipts[-1]["sha256"] if receipts else None,
                    journal_directory=str(journal) if temporary is None else None, snapshot=snapshot)
    finally:
        if process.pid is not None:
            _stop(process)
        parent.close()
        child.close()
        if temporary is not None:
            temporary.cleanup()


def _empty_choice(layout, method):
    identifier = f"{layout['id']}-{method}"
    return dict(id=identifier, context_id=layout["id"], bucket_xy_m=layout["bucket_xy_m"],
                stratum=layout["stratum"], method=method, intended_route=None,
                available_routes=[], prediction=None, candidate_pool=None, request=None,
                predicted_flight_risk=True)


def _wrap_choice(choice, method):
    result = copy.deepcopy(choice)
    identifier = f"{choice['context_id']}-{method}"
    result.update(id=identifier, method=method)
    if result["request"] is not None:
        result["request"].update(id=identifier, method=method)
    return result


class CachedRanker:
    """Memoize exact physics predictions without changing scores or tie order.

    Incremental snapshots revisit previously admitted commands. Cache keys bind
    bucket, shot geometry and both solved joint vectors; proposal names and
    intended route do not affect FlightRanker's physical prediction.
    """
    def __init__(self, ranker):
        self.ranker = ranker
        self.cache = {}

    def predict(self, request):
        from .intergrid_candidate_replay_plan import command_key

        key = command_key(request)
        if key not in self.cache:
            self.cache[key] = copy.deepcopy(self.ranker.predict(request))
        return copy.deepcopy(self.cache[key])

    def order(self, admitted):
        from .intergrid_flight_ranker import rank_key

        ranked = [(request, self.predict(request)["score_m"]) for request in admitted]
        return [request for request, _ in sorted(ranked, key=lambda item: rank_key(item[0], item[1]))]


def incremental_plan(layout, method, libraries, source_checkpoint, history, ranker_config,
                     neural_checkpoint, *, publish, deadline):
    """Keep the frozen candidate rules, publishing after each complete candidate."""
    from . import parallel_capture_grid_workers as workers
    from . import neural_shot_plan as neural
    from .neural_shot_model import order
    from .clearance_expansion_plan import (_empty, _ranker, _prepare_branches,
                                          current_selection, needs_expansion, branch_proposals)
    from .intergrid_system_plan import base_proposals, fallback_proposals
    from .intergrid_candidate_replay_plan import command_key

    if method not in METHODS:
        raise ValueError("Unknown budgeted planner")
    if workers._engine is None or workers._engine.adapter.env is not None:
        raise RuntimeError("Budgeted planning requires initialized CPU-only engine")
    ranker = CachedRanker(_ranker(ranker_config))
    generated, completed, ranked_ids = {}, [], []
    plan = {name: _empty() for name in ("base", "fallback", "neural", "branches")}

    def admitted_rows():
        if method == METHODS[0]:
            return plan["base"]["admitted"] + plan["fallback"]["admitted"]
        return list({command_key(row): row for name in ("neural", "branches")
                     for row in plan[name]["admitted"]}.values())

    def choice():
        if method == METHODS[0]:
            selected = _wrap_choice(current_selection(layout, plan, history, ranker), method)
        else:
            selected = neural._choice(layout, admitted_rows(), ranker,
                                      len(plan["neural"]["proposals"]), ranked_ids)
        selected["predicted_flight_risk"] = needs_expansion(selected)
        return selected

    def snapshot(stage):
        if workers._engine.adapter.env is not None:
            raise RuntimeError("Budgeted preparation allocated native CUDA")
        selected = choice()  # Ranking must finish before the snapshot is eligible.
        return dict(layout=layout, method=method, active_stage=stage,
                    generated=copy.deepcopy(generated), plan=copy.deepcopy(plan),
                    completed_proposal_ids=list(completed),
                    ranked_proposal_ids=list(ranked_ids), choice=selected)

    def prepare(stage, rows, branch=False):
        if stage == "neural":
            generated.setdefault(stage, []).extend(rows)
        else:
            generated[stage] = rows
        for proposal in rows:
            if time.perf_counter() >= deadline:
                return False
            if branch:
                originals = {row["id"]: row for row in plan["neural"]["admitted"]}
                batch = _prepare_branches(workers._engine, [proposal], originals, ranker_config["assets"])
                admitted, records = batch["admitted"], batch["records"]
            else:
                admitted, records = workers.prepare_proposals([proposal], True)
            if workers._engine.adapter.env is not None:
                raise RuntimeError("Budgeted preparation allocated native CUDA")
            # An expensive preparation that ends after the deadline is discarded.
            if time.perf_counter() >= deadline:
                return False
            plan[stage]["proposals"].append(proposal)
            plan[stage]["admitted"].extend(admitted)
            plan[stage]["records"].extend(records)
            completed.append(proposal["id"])
            if not publish(snapshot(stage)):
                return False
        return True

    if method == METHODS[0]:
        proposals = base_proposals(layout, libraries, source_checkpoint)
        if not prepare("base", proposals):
            return
        fallback = fallback_proposals(layout, libraries, proposals, plan["base"]["admitted"])
        if not prepare("fallback", fallback):
            return
    else:
        proposals = neural.menu(layout, libraries, source_checkpoint)
        ordered = order(neural_checkpoint, proposals)
        generated["menu"] = proposals
        ranked_ids.extend(row["id"] for row in ordered)
        for start in range(0, min(neural.CHECK_LIMIT, len(ordered)), neural.CHECK_BATCH):
            batch = ordered[start:start + neural.CHECK_BATCH]
            if not prepare("neural", batch):
                return
            if not needs_expansion(choice()):
                break
        if needs_expansion(choice()) and plan["neural"]["admitted"]:
            branch_input = dict(base=plan["neural"], fallback=_empty(), expansion=_empty(), branches=_empty())
            branches = branch_proposals(neural.proposal_layout(layout), branch_input, ranker)
            if not prepare("branches", branches, branch=True):
                return
    # Covers empty menus and records the completed whole-stage state.
    publish(snapshot("finished"))


def run_budget_job(layout, method, libraries, source_checkpoint, history, ranker_config,
                   neural_checkpoint, *, reference, assets, directory, engine_kwargs,
                   budget_s=DEFAULT_BUDGET_S, startup_timeout_s=STARTUP_TIMEOUT_S):
    """Prepare one layout/method independently; suitable for four parent threads."""
    if method not in METHODS:
        raise ValueError("Unknown budgeted planner")
    from .parallel_capture_grid_workers import _initialize_worker

    journal = Path(directory) / "budget-journals" / layout["id"]
    result = supervise(
        incremental_plan,
        (layout, method, libraries, source_checkpoint, history, ranker_config, neural_checkpoint),
        initializer=_initialize_worker, initargs=(reference, assets, Path(directory), engine_kwargs),
        journal_directory=journal, budget_s=budget_s, startup_timeout_s=startup_timeout_s,
    )
    snapshot = result.pop("snapshot")
    result.update(layout=layout, method=method, choice=_empty_choice(layout, method),
                  plan={name: dict(proposals=[], admitted=[], records=[]) for name in ("base", "fallback", "neural", "branches")},
                  generated={}, completed_proposal_ids=[], ranked_proposal_ids=[], active_stage="no_complete_snapshot")
    if snapshot is not None:
        result.update(snapshot)
    result["choices"] = [result["choice"]]
    result["latency_s"] = result["decision_latency_s"]
    return result
