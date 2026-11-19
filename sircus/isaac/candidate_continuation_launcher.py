"""Mock-tested bounds for a single future continuation application start.

This module has no simulator, subprocess, GPU query or production runtime
backend. ``supervise`` accepts only an explicitly mock-only runtime. It tests
the ownership/accounting contract that a separately reviewed backend must
implement; passing these tests grants no execution permission. Runtime calls
must themselves be bounded, and a future backend needs an external hard
watchdog because this synchronous driver cannot preempt a blocked callback.

Receipts are new-only and preserve partial work. An interrupted start is never
retried. Success here means the mock control flow completed, not that a native
run passed the independent trace/checkpoint auditor or that a model is promoted.
"""

from dataclasses import dataclass
import json
import math
from pathlib import Path
import re
from typing import Protocol
import uuid

from candidate_continuation_contract import PROTOCOL_SHA256 as LEARNING_SHA256
from candidate_continuation_journal import write_new_json, _sync_directory
from candidate_continuation_ledger import (
    ASSESSMENT_REQUEST_LIMIT, PARENT_CHECKPOINT_SHA256, TRAIN_RELEASE_LIMIT,
    TRAIN_REQUEST_LIMIT, canonical_hash,
)

SCHEMA = "sircus-candidate-continuation-bounded-launch-v1"
GPU_UUID = "GPU-ab1950b0-01dd-0fab-407f-2d17936b27f4"
CONTAINER = "sircus_isaac_4090"
WAIT_SECONDS = 7200
NATIVE_SECONDS = 10800
MIN_FREE_BYTES = 6 * 1024 ** 3
POLL_SECONDS = 1
PHASE_LIMITS = (("initialization", 600), ("imitation", 600), ("train", 7200),
                ("assessment", 1800), ("audit", 600))
PHASES = tuple(name for name, _ in PHASE_LIMITS)
STARTUP_HELPERS = {
    "film_cylinder_farm.py": "849b1af800a0f13a73af8845d39ec30117f8f01648effeb9698b5891af4efe8d",
    "parallel_runtime.py": "33261c6eadbc7d0f4f823caaedb80f8e95f08229da5d545c688043d945b1b25d",
}
BINDING_NAMES = frozenset(("parent_checkpoint_sha256", "producer_source_sha256",
    "native_binding_sha256", "fresh_split_sha256", "train_bank_sha256",
    "learning_protocol_sha256", *STARTUP_HELPERS))
RIM_ANGLES = (0, 90, 180, 270, 45, 135, 225, 315)
CONTACT_KINDS = ("bucket", "rim", "table", "floor", "robot")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def _sha(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def _preserve_invalid(value):
    """Keep failed mock observations without emitting nonfinite JSON numbers."""
    if isinstance(value, dict):
        return {str(k):_preserve_invalid(v) for k,v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_preserve_invalid(v) for v in value]
    if type(value) is float and not math.isfinite(value):
        return dict(nonfinite_numeric_repr=repr(value))
    if value is None or type(value) in (bool, int, float, str):
        return value
    return dict(invalid_mock_value_type=type(value).__name__)


def _path(value):
    p = Path(value)
    require(p.is_absolute(), "Launch paths must be absolute")
    return p.resolve()


def _bindings(value):
    require(type(value) is dict and set(value) == BINDING_NAMES,
            "Expected the complete fixed launch input binding set")
    require(all(_sha(v) for v in value.values()), "Invalid launch input digest")
    require(value["parent_checkpoint_sha256"] == PARENT_CHECKPOINT_SHA256,
            "Launch parent differs from frozen seed 22")
    require(value["learning_protocol_sha256"] == LEARNING_SHA256,
            "Learning protocol differs from the reviewed continuation")
    require(all(value[k] == v for k, v in STARTUP_HELPERS.items()),
            "Original startup helper identity changed")
    return tuple(sorted(value.items()))


@dataclass(frozen=True)
class LaunchPlan:
    """Immutable, hashable mock contract; no executable command or overrides."""

    output: str
    protected_roots: tuple[str, ...]
    scene_lock: str
    bindings: tuple[tuple[str, str], ...]
    session: str

    def __post_init__(self):
        out, lock = _path(self.output), _path(self.scene_lock)
        require(str(out) == self.output and str(lock) == self.scene_lock,
                "Launch paths must be canonical")
        require(type(self.protected_roots) is tuple and self.protected_roots,
                "Source/evidence roots must be explicitly protected")
        require(all(type(p) is str and str(_path(p)) == p for p in self.protected_roots),
                "Protected roots must be canonical absolute paths")
        require(all(not out.is_relative_to(Path(p)) and not Path(p).is_relative_to(out)
                    for p in self.protected_roots),
                "Output cannot be inside source or frozen evidence")
        require(not lock.is_relative_to(out), "Shared lock cannot be inside new output")
        require(type(self.bindings) is tuple and _bindings(dict(self.bindings)) == self.bindings,
                "Launch bindings must be immutable and sorted")
        require(isinstance(self.session, str)
                and re.fullmatch(r"[0-9a-f]{32}", self.session) is not None,
                "Expected a unique 128-bit session token")

    @classmethod
    def create(cls, output, *, protected_roots, scene_lock, input_sha256, session=None):
        return cls(str(_path(output)), tuple(str(_path(p)) for p in protected_roots),
                   str(_path(scene_lock)), _bindings(input_sha256), session or uuid.uuid4().hex)

    def to_dict(self):
        result = dict(schema=SCHEMA, output=self.output,
            protected_roots=list(self.protected_roots), scene_lock=self.scene_lock,
            input_sha256=dict(self.bindings), session=self.session,
            gpu_uuid=GPU_UUID, container=CONTAINER,
            queue_timeout_s=WAIT_SECONDS, native_timeout_s=NATIVE_SECONDS,
            phase_wall_limits_s=dict(PHASE_LIMITS), min_free_bytes=MIN_FREE_BYTES,
            max_application_starts=1, max_diagnostic_cohorts=14,
            max_diagnostic_placements=490, max_diagnostic_physics_steps=1824,
            train_request_limit=TRAIN_REQUEST_LIMIT, train_release_limit=TRAIN_RELEASE_LIMIT,
            assessment_request_limit=ASSESSMENT_REQUEST_LIMIT,
            mock_only=True, production_backend_implemented=False,
            execution_authorized=False, release_authorized=False)
        result["fingerprint"] = canonical_hash(result)
        return result

    @classmethod
    def from_dict(cls, value):
        require(type(value) is dict, "Expected launch plan object")
        plan = cls.create(value["output"], protected_roots=value["protected_roots"],
            scene_lock=value["scene_lock"], input_sha256=value["input_sha256"],
            session=value["session"])
        require(plan.to_dict() == value, "Launch plan bounds or fingerprint changed")
        return plan


@dataclass(frozen=True)
class ChildIdentity:
    pid: int
    start_token: str
    session: str
    gpu_uuid: str = GPU_UUID
    container: str = CONTAINER

    def valid_for(self, plan):
        return (type(self.pid) is int and self.pid > 1
                and isinstance(self.start_token, str) and bool(self.start_token)
                and self.session == plan.session and self.gpu_uuid == GPU_UUID
                and self.container == CONTAINER)

    def to_dict(self):
        return dict(pid=self.pid, start_token=self.start_token, session=self.session,
                    gpu_uuid=self.gpu_uuid, container=self.container)


class Runtime(Protocol):
    """Only mock implementations are accepted by this development driver.

    Query only the declared 4090. Lock methods operate on the pre-existing shared
    lock without replacing it. ``start_native`` must not restart a container or
    service; it must enforce the supplied external absolute watchdog. Every
    runtime call must return promptly. ``terminate_owned`` must recheck the full
    identity atomically at the final signal boundary, not trust an earlier poll.
    It returns ``stopped``, ``already_exited`` or ``ownership_mismatch``.
    """

    mock_only: bool

    def monotonic(self) -> float: ...
    def sleep(self, seconds: float) -> None: ...
    def source_snapshot(self, plan: LaunchPlan) -> dict: ...
    def free_bytes(self, output_parent: Path) -> int: ...
    def acquire_scene_lock(self, lock: str, session: str) -> bool: ...
    def release_scene_lock(self, lock: str, session: str) -> None: ...
    def gpu_state(self, gpu_uuid: str) -> dict: ...
    def start_native(self, plan: LaunchPlan, absolute_deadline: float) -> ChildIdentity: ...
    def identity(self, pid: int) -> ChildIdentity | None: ...
    def poll(self, owner: ChildIdentity) -> dict: ...
    def terminate_owned(self, owner: ChildIdentity) -> str: ...


class DiagnosticLedger:
    """Append-only finite startup schedule, distinct from task requests."""

    def __init__(self):
        self.records = []
        self.rims = 0
        self.rim_found = False
        self.contacts = 0
        self.isolation = False
        self.parking_resets = 0

    def observe(self, records, parking_resets):
        require(type(records) is list and records[:len(self.records)] == self.records,
                "Startup diagnostics were removed, replaced or repeated")
        require(type(parking_resets) is int and parking_resets >= self.parking_resets,
                "Parking reset counter changed or went backwards")
        for record in records[len(self.records):]:
            require(type(record) is dict, "Malformed diagnostic cohort")
            expected = {"kind", "placements", "active_probes", "physics_steps",
                        "wall_s", "process_cpu_s", "initialized_states_sha256", "passed"}
            require(set(record) == expected | ({"azimuth_deg"} if record.get("kind") == "rim_ray" else set()),
                    "Unexpected diagnostic fields")
            require(record["placements"] == 35 and type(record["placements"]) is int,
                    "Every diagnostic cohort must log all 35 initialized balls")
            require(_sha(record["initialized_states_sha256"]), "Missing initialized ball-state digest")
            require(all(_finite(record[k]) and record[k] >= 0 for k in ("wall_s", "process_cpu_s")),
                    "Nonfinite or negative diagnostic phase cost")
            require(type(record["passed"]) is bool, "Expected diagnostic result boolean")
            require(type(record["active_probes"]) is int and type(record["physics_steps"]) is int,
                    "Diagnostic probe and step counts must be integers")
            kind = record["kind"]
            if kind == "rim_ray":
                require(not self.rim_found and self.contacts == 0 and not self.isolation
                        and self.rims < len(RIM_ANGLES), "Repeated/out-of-order rim diagnostic")
                require(record["azimuth_deg"] == RIM_ANGLES[self.rims]
                        and type(record["azimuth_deg"]) is int,
                        "Rim diagnostic azimuth differs from the original schedule")
                require(record["active_probes"] == 35 and record["physics_steps"] == 48,
                        "Rim diagnostic probe/step budget changed")
                self.rims += 1
                self.rim_found = record["passed"]
                require(self.rim_found or self.rims < 8, "Eight rays exhausted without all cells clear")
            elif kind in CONTACT_KINDS:
                require(self.rim_found and not self.isolation and self.contacts < 5
                        and kind == CONTACT_KINDS[self.contacts], "Contact diagnostics out of order")
                require(record["active_probes"] == 35 and record["physics_steps"] == 240,
                        "Contact diagnostic probe/step budget changed")
                require(record["passed"], "Contact attribution diagnostic failed")
                self.contacts += 1
            elif kind == "foreign_cell_collision_filter":
                require(self.contacts == 5 and not self.isolation, "Repeated/out-of-order isolation diagnostic")
                require(record["active_probes"] == 1 and record["physics_steps"] == 240,
                        "Isolation diagnostic must activate only env0 for 240 steps")
                require(record["passed"], "Foreign-cell isolation diagnostic failed")
                self.isolation = True
            else:
                raise ValueError("Unreviewed startup diagnostic or smoke release")
            self.records.append(json.loads(json.dumps(record, allow_nan=False)))
        self.parking_resets = parking_resets
        require(len(self.records) <= 14 and sum(r["placements"] for r in self.records) <= 490
                and sum(r["physics_steps"] for r in self.records) <= 1824,
                "Startup diagnostic budget exceeded")

    @property
    def complete(self):
        return self.rim_found and self.contacts == 5 and self.isolation

    def summary(self):
        return dict(complete=self.complete, cohorts=len(self.records),
            initialized_trajectories=sum(r["placements"] for r in self.records),
            active_probes=sum(r["active_probes"] for r in self.records),
            physics_steps=sum(r["physics_steps"] for r in self.records),
            wall_s=sum(r["wall_s"] for r in self.records),
            process_cpu_s=sum(r["process_cpu_s"] for r in self.records),
            parking_resets=self.parking_resets, task_requests=0)


def _gpu_idle(state):
    require(type(state) is dict and set(state) == {"gpu_uuid", "container", "container_running",
            "container_gpu_uuids", "compute_pids"}, "Malformed 4090 readiness response")
    require(state["gpu_uuid"] == GPU_UUID and state["container"] == CONTAINER
            and state["container_running"] is True and state["container_gpu_uuids"] == [GPU_UUID],
            "Existing container must already be running with only the declared 4090")
    require(type(state["compute_pids"]) is list and all(type(pid) is int and pid > 1
            for pid in state["compute_pids"]), "Invalid 4090 owner process list")
    return not state["compute_pids"]


def supervise(plan, runtime):
    """Exercise one bounded mock launch; all partial evidence remains new-only.

    Ownership uncertainty retains the injected scene-lock lease for inspection.
    No second start is possible in the same output, including after a failed
    start whose child identity could not be returned. Results never substitute
    for the independent native audit. The final audit phase is CPU-only and its
    externally produced result must still be verified by the independent reader.
    """
    require(isinstance(plan, LaunchPlan) and getattr(runtime, "mock_only", None) is True,
            "Only explicit mock-only runtimes are implemented; native execution is unavailable")
    # Revalidate even a dataclass built by a nonstandard caller.
    LaunchPlan.from_dict(plan.to_dict())
    output = Path(plan.output)
    require(output.parent.is_dir() and not output.exists() and not output.is_symlink(),
            "Expected untouched output path with an existing parent; no resume/retry")
    output.mkdir()
    _sync_directory(output.parent)
    write_new_json(output / "launch-plan.json", plan.to_dict())
    initial = runtime.monotonic()
    require(_finite(initial) and initial >= 0, "Invalid mock monotonic clock")
    last_clock = initial
    sequence = 0
    observation_sequence = 0
    owner = None
    lock_held = False
    attempted_start = False
    safe_to_unlock = True
    diagnostics = DiagnosticLedger()
    phase_cost = {name: 0. for name in PHASES}
    counters = dict(train_requests=0, train_releases=0, assessment_requests=0, assessment_releases=0)
    last_progress = None
    started = None
    phase_index = 0
    phase_started = None

    def clock():
        nonlocal last_clock
        value = runtime.monotonic()
        require(_finite(value) and value >= last_clock, "Mock monotonic clock is nonfinite or went backwards")
        last_clock = value
        return value

    def publish(kind, payload):
        nonlocal sequence
        value = dict(schema=SCHEMA, session=plan.session, seq=sequence, kind=kind,
                     elapsed_s=clock()-initial, mock_only=True, **payload)
        write_new_json(output / f"launch-{sequence:06d}.json", value)
        sequence += 1

    def check_inputs():
        require(runtime.source_snapshot(plan) == dict(plan.bindings),
                "Source/input identity changed; run cannot continue")
        free = runtime.free_bytes(output.parent)
        require(type(free) is int and free >= MIN_FREE_BYTES,
                "Storage fell below the original 6 GiB floor; evidence preserved")

    try:
        check_inputs()
        publish("waiting", {})
        while True:
            elapsed = clock()-initial
            require(elapsed < WAIT_SECONDS, "4090/shared-lock wait exceeded two hours")
            check_inputs()
            acquired = runtime.acquire_scene_lock(plan.scene_lock, plan.session)
            require(type(acquired) is bool, "Invalid shared-lock acquisition response")
            if acquired:
                lock_held = True
                if _gpu_idle(runtime.gpu_state(GPU_UUID)):
                    break
                runtime.release_scene_lock(plan.scene_lock, plan.session)
                lock_held = False
            runtime.sleep(min(POLL_SECONDS, WAIT_SECONDS - elapsed))
        # Any queue wait, lock acquisition or GPU query may have raced a source
        # change. Bind again immediately before the one permitted start.
        check_inputs()
        started = clock()
        require(started-initial < WAIT_SECONDS, "4090/shared-lock wait exceeded two hours")
        phase_started = started
        publish("start_intent", dict(absolute_deadline=started+NATIVE_SECONDS))
        attempted_start = True
        safe_to_unlock = False
        owner = runtime.start_native(plan, started+NATIVE_SECONDS)
        require(isinstance(owner, ChildIdentity) and owner.valid_for(plan),
                "Native start returned no matching child identity; never retry")
        require(runtime.identity(owner.pid) == owner, "Child identity changed at native start")
        publish("started", dict(owner=owner.to_dict()))
        while True:
            current = clock()
            require(current-started <= NATIVE_SECONDS, "Absolute three-hour native wall limit exceeded")
            require(current-phase_started <= dict(PHASE_LIMITS)[PHASES[phase_index]],
                    "Phase wall limit exceeded: " + PHASES[phase_index])
            check_inputs()
            require(runtime.identity(owner.pid) == owner, "Child PID/start identity changed; refusing foreign signal")
            snapshot = runtime.poll(owner)
            # Persist even malformed/failed observations before interpreting
            # counters, diagnostics or exit status; never replace a prior one.
            write_new_json(output / f"launch-observation-{observation_sequence:06d}.json",
                dict(schema=SCHEMA, session=plan.session, mock_only=True,
                     observation=_preserve_invalid(snapshot)))
            observation_sequence += 1
            current = clock()
            require(current-started <= NATIVE_SECONDS, "Absolute three-hour native wall limit exceeded")
            require(current-phase_started <= dict(PHASE_LIMITS)[PHASES[phase_index]],
                    "Phase wall limit exceeded: " + PHASES[phase_index])
            expected = {"phase", "diagnostics", "parking_resets", "train_requests", "train_releases",
                        "assessment_requests", "assessment_releases", "running", "exit_code", "audit_cpu_only"}
            require(type(snapshot) is dict and set(snapshot) == expected, "Malformed child progress receipt")
            require(type(snapshot["running"]) is bool, "Invalid child running state")
            require(snapshot["phase"] in PHASES, "Unknown child phase")
            next_index = PHASES.index(snapshot["phase"])
            require(next_index in (phase_index, phase_index+1), "Skipped, repeated or reversed native phase")
            require(phase_index == 0 or snapshot["diagnostics"] == diagnostics.records,
                    "Startup diagnostics changed after initialization")
            diagnostics.observe(snapshot["diagnostics"], snapshot["parking_resets"])
            if next_index > 0:
                require(diagnostics.complete, "Startup diagnostic schedule incomplete before task work")
            if next_index != phase_index:
                require(next_index != PHASES.index("assessment")
                        or counters["train_requests"] == TRAIN_REQUEST_LIMIT
                        or counters["train_releases"] == TRAIN_RELEASE_LIMIT,
                        "Assessment began before a TRAIN stop budget was reached")
                require(next_index != PHASES.index("audit")
                        or counters["assessment_requests"] == ASSESSMENT_REQUEST_LIMIT,
                        "Audit began before all fixed assessment requests were resolved")
                phase_cost[PHASES[phase_index]] = current-phase_started
                phase_index, phase_started = next_index, current
            for key, before in counters.items():
                value = snapshot[key]
                require(type(value) is int and value >= before, "Request/release counters moved backwards")
            require(snapshot["train_requests"] <= TRAIN_REQUEST_LIMIT
                    and snapshot["train_releases"] <= min(TRAIN_RELEASE_LIMIT, snapshot["train_requests"])
                    and snapshot["assessment_requests"] <= ASSESSMENT_REQUEST_LIMIT
                    and snapshot["assessment_releases"] <= snapshot["assessment_requests"],
                    "Task request/release budget exceeded")
            require(next_index >= PHASES.index("train")
                    or snapshot["train_requests"] == snapshot["train_releases"] == 0,
                    "Native task request before TRAIN phase")
            require(next_index >= PHASES.index("assessment")
                    or snapshot["assessment_requests"] == snapshot["assessment_releases"] == 0,
                    "Held-out request before assessment phase")
            require(next_index <= PHASES.index("train")
                    or all(snapshot[k] == counters[k] for k in ("train_requests", "train_releases")),
                    "TRAIN requests changed after the final checkpoint boundary")
            require(next_index <= PHASES.index("assessment")
                    or all(snapshot[k] == counters[k] for k in ("assessment_requests", "assessment_releases")),
                    "Assessment requests changed during audit")
            require(type(snapshot["audit_cpu_only"]) is bool
                    and (next_index != PHASES.index("audit") or snapshot["audit_cpu_only"]),
                    "Independent audit phase must be CPU-only")
            counters = {k:snapshot[k] for k in counters}
            if snapshot != last_progress:
                publish("progress", dict(snapshot=snapshot))
                last_progress = json.loads(json.dumps(snapshot, allow_nan=False))
            if not snapshot["running"]:
                require(type(snapshot["exit_code"]) is int and snapshot["exit_code"] == 0,
                        "Native child failed; partial evidence preserved")
                safe_to_unlock = True
                require(next_index == len(PHASES)-1, "Native child exited before all bounded phases")
                phase_cost[PHASES[phase_index]] = clock()-phase_started
                result = dict(status="mock_complete_independent_audit_required",
                    independent_audit_passed=False, model_promoted=False,
                    native_application_starts=1, native_wall_s=clock()-started,
                    queue_wall_s=started-initial, phase_wall_s=phase_cost,
                    phase_wall_scope="Observed non-overlapping supervisor phase intervals; mock only",
                    diagnostics=diagnostics.summary(), counters=counters, owner=owner.to_dict())
                publish("complete", result)
                write_new_json(output / "launch-result.json", dict(schema=SCHEMA,
                    session=plan.session, mock_only=True, **result))
                return result
            require(snapshot["exit_code"] is None, "Running child claimed an exit code")
            runtime.sleep(POLL_SECONDS)
    except BaseException as exc:
        termination = "not_started"
        termination_error = None
        if attempted_start:
            termination = "unknown_child_identity"
            if isinstance(owner, ChildIdentity) and owner.valid_for(plan):
                try:
                    if runtime.identity(owner.pid) == owner:
                        termination = runtime.terminate_owned(owner)
                        require(termination in ("stopped", "already_exited", "ownership_mismatch"),
                                "Invalid owned termination response")
                        safe_to_unlock = termination in ("stopped", "already_exited")
                    else:
                        termination = "ownership_mismatch"
                except BaseException as stop_error:
                    termination, termination_error = "termination_failed", repr(stop_error)
                    safe_to_unlock = False
        if started is not None and phase_started is not None:
            phase_cost[PHASES[phase_index]] = max(0., last_clock-phase_started)
        result = dict(schema=SCHEMA, session=plan.session, status="mock_failed",
            reason=str(exc) or type(exc).__name__, exception_type=type(exc).__name__,
            mock_only=True, independent_audit_passed=False, model_promoted=False,
            native_application_starts=int(attempted_start), termination=termination,
            termination_error=termination_error,
            scene_lock_retained=lock_held and not safe_to_unlock, phase_wall_s=phase_cost,
            diagnostics=diagnostics.summary(), counters=counters)
        write_new_json(output / "launch-result.json", result)
        raise
    finally:
        if lock_held and safe_to_unlock:
            runtime.release_scene_lock(plan.scene_lock, plan.session)
