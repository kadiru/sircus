"""Owned-process deadlines for the comparison; imports and mocks start nothing.

Only explicit approved callers can enter the Linux operations. Native progress
is an atomic, bound snapshot, never permission to extend a phase deadline. The
existing reviewed pidfd operations signal identified descendants, never a PID or
process group. Zero native exit still needs the separate model/physics audit.
"""

import json
import math
import os
from pathlib import Path
import re
import stat

from candidate_continuation_journal import write_new_json
from candidate_continuation_ledger import canonical_hash
from candidate_continuation_watchdog import (
    COOKIE, GPU_UUID, LinuxProcessOps, same_process, stop_owned, strict_json,
)
from location_comparison_bounds import (
    ARMS, ARM_WALL_S, SHARED_WALL_S, TRAIN_CAP, MIN_FREE_BYTES,
    SnapshotWatchdog, _limits,
)

SCHEMA = "sircus-location-comparison-watchdog-v1"
PROGRESS_SCHEMA = "sircus-location-comparison-process-progress-v1"
AUDIT_SCHEMA = "sircus-location-comparison-audit-watchdog-v1"
MAX_PROGRESS_BYTES = 65536
AUDIT_WALL_S = 7200
NATIVE_WALL_S = 2 * ARM_WALL_S + SHARED_WALL_S
_RETAINED_LOCKS = []


def require(value, message):
    if not value:
        raise ValueError(message)


def signed(value):
    return dict(value, fingerprint=canonical_hash(value))


def _clock(value):
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


def _final_wall(ops, start):
    try:
        now = ops.monotonic()
        return now-start if _clock(now) and _clock(start) and now >= start else None
    except BaseException:
        return None


def _identity(session_id, runtime_fingerprint, comparison_fingerprint):
    require(type(session_id) is str and re.fullmatch(r"[0-9a-f]{32}", session_id),
            "Invalid process session")
    for value in (runtime_fingerprint, comparison_fingerprint):
        require(type(value) is str and re.fullmatch(r"[0-9a-f]{64}", value),
                "Invalid process input fingerprint")


def _absolute(value, label):
    path = Path(value)
    require(path.is_absolute() and ".." not in path.parts and path == path.resolve()
            and not any(p.is_symlink() for p in (path, *path.parents)),
            "Expected canonical unlinked " + label)
    return path


def _sync_directory(path):
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


class ProgressWriter:
    """One child writes a new progress stream; existing streams cannot reopen."""

    def __init__(self, output, session_id, runtime_fingerprint, comparison_fingerprint):
        _identity(session_id, runtime_fingerprint, comparison_fingerprint)
        self.path = _absolute(output, "progress directory") / "comparison-progress.json"
        require(self.path.parent.is_dir() and not self.path.exists(), "Progress already exists or output missing")
        self.session, self.runtime, self.comparison = session_id, runtime_fingerprint, comparison_fingerprint
        self.sequence, self.finished = 0, False

    def publish(self, snapshot, *, running=True, status="running"):
        require(not self.finished and type(running) is bool, "Progress is already closed or running flag invalid")
        require((running and status == "running") or
                (not running and status in ("runner_complete_pending_audit", "failed_no_retry")),
                "Unrecognized child progress status")
        require(type(snapshot) is dict and snapshot.get("comparison_fingerprint") == self.comparison,
                "Progress snapshot comparison changed")
        value = signed(dict(schema=PROGRESS_SCHEMA, session_id=self.session,
            runtime_fingerprint=self.runtime, comparison_fingerprint=self.comparison,
            sequence=self.sequence, snapshot=snapshot, running=running, status=status))
        temporary = self.path.parent / (".comparison-progress-%06d.tmp" % self.sequence)
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "wb") as stream:
            stream.write((json.dumps(value, sort_keys=True, allow_nan=False) + "\n").encode())
            stream.flush()
            os.fsync(stream.fileno())
        require(not self.path.is_symlink(), "Progress destination became linked")
        os.replace(temporary, self.path)
        _sync_directory(self.path.parent)
        self.sequence += 1
        self.finished = not running
        return value


def read_progress(output, *, session_id, runtime_fingerprint, comparison_fingerprint, previous=None):
    path = Path(output) / "comparison-progress.json"
    require(not path.is_symlink(), "Linked progress file")
    if not path.exists():
        require(previous is None, "Published progress disappeared")
        return None
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    descriptor = os.open(path, flags)
    with os.fdopen(descriptor, "rb") as stream:
        require(stat.S_ISREG(os.fstat(stream.fileno()).st_mode), "Progress must be a regular file")
        raw = stream.read(MAX_PROGRESS_BYTES + 1)
    require(len(raw) <= MAX_PROGRESS_BYTES, "Progress exceeds 64 KiB")
    value = strict_json(raw)
    require(type(value) is dict and set(value) == {"schema", "session_id", "runtime_fingerprint",
            "comparison_fingerprint", "sequence", "snapshot", "running", "status", "fingerprint"},
            "Progress fields changed")
    require(value["schema"] == PROGRESS_SCHEMA and value["session_id"] == session_id
            and value["runtime_fingerprint"] == runtime_fingerprint
            and value["comparison_fingerprint"] == comparison_fingerprint,
            "Progress process/input binding changed")
    require(value["fingerprint"] == canonical_hash({k: v for k, v in value.items() if k != "fingerprint"}),
            "Progress fingerprint changed")
    require(type(value["sequence"]) is int and value["sequence"] >= 0
            and type(value["running"]) is bool and type(value["snapshot"]) is dict,
            "Invalid progress sequence or shape")
    require((value["running"] and value["status"] == "running") or
            (not value["running"] and value["status"] in ("runner_complete_pending_audit", "failed_no_retry")),
            "Invalid progress completion status")
    if previous is not None:
        require(value["sequence"] >= previous["sequence"], "Progress sequence moved backwards")
        require(value["sequence"] != previous["sequence"] or value == previous,
                "Progress sequence was rewritten")
        require(previous["running"] or value == previous, "Closed progress was changed")
    return value


def _arguments(command, output, session_id, runtime_fingerprint, comparison_fingerprint,
               integrity_check, poll_s, execute_approved):
    require(execute_approved is True, "Explicit approved execution required")
    _identity(session_id, runtime_fingerprint, comparison_fingerprint)
    require(type(command) is list and command and all(type(x) is str and x and "\0" not in x for x in command)
            and Path(command[0]).is_absolute(), "Explicit argv with absolute executable required")
    require(callable(integrity_check), "Actual input integrity check required")
    require(type(poll_s) in (int, float) and math.isfinite(poll_s) and 0 < poll_s <= 1,
            "Invalid bounded poll interval")
    output = _absolute(output, "output")
    require(output.parent.is_dir() and not output.exists(), "Refuse existing output or missing parent")
    return output


def _terminal(progress, assessment_cap):
    require(progress is not None and progress["running"] is False
            and progress["status"] == "runner_complete_pending_audit", "Native exit lacks final progress")
    value = progress["snapshot"]
    require(value.get("phase_index") == 3 and value.get("active_phase") == "shared"
            and value.get("pending") is None and value.get("stopped") is None
            and value.get("train_requested") == {arm: TRAIN_CAP for arm in ARMS}
            and value.get("train_completed") == {arm: TRAIN_CAP for arm in ARMS}
            and value.get("assessment_requested") == value.get("assessment_completed") == assessment_cap,
            "Native exit has incomplete comparison counters")


def _phase_deadline(observer, start):
    value = observer.previous if observer is not None else None
    if value is None:
        return start + SHARED_WALL_S
    phase = value["active_phase"]
    limit = SHARED_WALL_S if phase == "shared" else ARM_WALL_S
    return value["observed_monotonic"] + limit - value["phase_elapsed_s"][phase]


def _phase_accounting(previous, current):
    """Completed/future phases cannot absorb current time to evade their cap."""
    value = current["snapshot"]
    index = value["phase_index"]
    elapsed = value["phase_elapsed_s"]
    for phase, first in (("original_usable", 1), ("broader", 2)):
        require(index >= first or elapsed[phase] == 0, "Future phase was charged elapsed time")
    if previous is not None:
        before = previous["snapshot"]
        prior_index = before["phase_index"]
        for phase, occurrences in (("shared", (0, 3)), ("original_usable", (1,)), ("broader", (2,))):
            if not any(prior_index <= position <= index for position in occurrences):
                require(elapsed[phase] == before["phase_elapsed_s"][phase],
                        "Inactive phase elapsed time changed")


def _environment(session, runtime, comparison, start, deadline, owner_file, *, audit=False):
    return dict(os.environ, **{COOKIE: session,
        "CUDA_VISIBLE_DEVICES": "" if audit else GPU_UUID, "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
        "SIRCUS_ISAAC_SOURCE": str(Path(__file__).resolve().parent),
        "PYTHONDONTWRITEBYTECODE": "1", "SIRCUS_COMPARISON_SESSION": session,
        "SIRCUS_COMPARISON_RUNTIME_FINGERPRINT": runtime,
        "SIRCUS_COMPARISON_FINGERPRINT": comparison,
        "SIRCUS_COMPARISON_STARTED_MONOTONIC": str(start),
        "SIRCUS_COMPARISON_DEADLINE_MONOTONIC": str(deadline),
        "SIRCUS_COMPARISON_OWNER_FILE": str(owner_file)})


def supervise(command, *, plan, output, lock, session_id, runtime_fingerprint,
              integrity_check, ops=None, poll_s=.5, execute_approved=False):
    """One native child; a blocked call cannot pause the independent phase clock.

    Startup and closure count against the cumulative shared hour. Each arm has
    its cumulative eight hours. The fixed 17-hour overall deadline is a second
    ceiling, not additional time. Missing/stale progress remains in the last
    observed phase; there is no retry, resume, refund or PID-only signalling.
    """
    comparison, assessment_cap = _limits(plan)
    output = _arguments(command, output, session_id, runtime_fingerprint, comparison,
                        integrity_check, poll_s, execute_approved)
    lock = _absolute(lock, "shared lock")
    require(lock.is_file() and not lock.is_relative_to(output), "Existing shared scene lock required outside output")
    import fcntl
    descriptor = os.open(lock, os.O_RDWR | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0))
    if not stat.S_ISREG(os.fstat(descriptor).st_mode):
        os.close(descriptor)
        raise ValueError("Scene lock must be regular")
    lease = os.fdopen(descriptor, "r+")
    lock_identity = (os.fstat(lease.fileno()).st_dev, os.fstat(lease.fileno()).st_ino)
    try:
        fcntl.flock(lease.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BaseException:
        lease.close()
        raise
    ops = ops or LinuxProcessOps()
    start = ops.monotonic()
    owner = watchdog = progress = cleanup = error = reason = code = observer = None
    starts, process_starts, safe, created = 0, 0, True, False
    deadline = start + NATIVE_WALL_S if _clock(start) else 0
    try:
        require(_clock(start), "Invalid independent clock")
        output.mkdir(mode=0o700)
        _sync_directory(output.parent)
        created = True
        observer = SnapshotWatchdog(plan, started_monotonic=start, clock=ops.monotonic,
            resources=lambda: dict(free_storage_bytes=ops.free_bytes(output)))
        watchdog = ops.identity(ops.own_pid())
        require(watchdog is not None and watchdog.get("cookie") == session_id,
                "Watchdog session cookie missing")
        require(integrity_check() is True, "Actual input integrity check did not pass")
        observer.check()
        ops.prepare()
        observer.check()
        environment = _environment(session_id, runtime_fingerprint, comparison, start, deadline,
                                   output / "comparison-watchdog-owner.json")
        starts = 1
        pid = ops.start(command, environment)
        process_starts = 1
        owner = ops.identity(pid)
        require(owner is not None and owner.get("cookie") == session_id
                and owner.get("ppid") == ops.own_pid() and owner.get("sid") == pid,
                "Owned child identity unavailable after sole start")
        write_new_json(output / "comparison-watchdog-owner.json", signed(dict(schema=SCHEMA,
            session_id=session_id, runtime_fingerprint=runtime_fingerprint, comparison_fingerprint=comparison,
            owner=watchdog, native_owner=owner, command=command, gpu_uuid=GPU_UUID,
            shared_lock=str(lock), shared_lock_identity=list(lock_identity), start_monotonic=start,
            absolute_deadline_monotonic=deadline,
            phase_limits=dict(shared=SHARED_WALL_S, original_usable=ARM_WALL_S, broader=ARM_WALL_S))))
        while True:
            require(not lock.is_symlink() and (lock.stat().st_dev, lock.stat().st_ino) == lock_identity,
                    "Shared scene lock identity changed")
            require(integrity_check() is True, "Actual input integrity check did not pass")
            current = read_progress(output, session_id=session_id, runtime_fingerprint=runtime_fingerprint,
                                    comparison_fingerprint=comparison, previous=progress)
            observer.check(None if current is None else current["snapshot"])
            if current is not None:
                _phase_accounting(progress, current)
            progress = current
            now = ops.monotonic()
            require(_clock(now) and observer.last_time <= now < deadline,
                    "Independent absolute native deadline/clock failed")
            require(same_process(ops.identity(ops.own_pid()), watchdog), "Watchdog ownership changed")
            code = ops.poll(pid)
            if code is not None:
                require(type(code) is int and code == 0, "Native child exited unsuccessfully")
                require(not ops.owned(pid, session_id), "Native child left owned workers alive")
                _terminal(progress, assessment_cap)
                break
            require(same_process(ops.identity(pid), owner), "Native child ownership changed")
            ops.sleep(poll_s)
    except BaseException as failure:
        error, reason = type(failure).__name__, str(failure) or type(failure).__name__
        if starts:
            safe = False
            try:
                cleanup = stop_owned(ops, owner, session_id,
                    deadline=min(deadline, _phase_deadline(observer, start), ops.monotonic() + 35)) \
                    if owner is not None else "unknown_child_identity"
                safe = cleanup in ("stopped", "already_exited")
            except BaseException as stop_error:
                cleanup = "termination_unconfirmed:" + type(stop_error).__name__
    finally:
        result = signed(dict(schema=SCHEMA, session_id=session_id,
            runtime_fingerprint=runtime_fingerprint, comparison_fingerprint=comparison,
            owner=watchdog, native_owner=owner,
            status="native_exit_zero_independent_audit_required" if reason is None else "stopped_no_retry",
            exit_code=code, error_type=error, reason=reason, termination=cleanup,
            native_application_starts=starts, native_child_start_attempts=starts,
            native_child_process_starts=process_starts,
            start_counter_scope="native_application_starts is the legacy child-start-intent alias; it does not prove simulator initialization",
            last_progress_fingerprint=(progress or {}).get("fingerprint"),
            final_snapshot=(progress or {}).get("snapshot"), start_monotonic=start if _clock(start) else None,
            absolute_deadline_monotonic=deadline, wall_s=_final_wall(ops, start),
            lock_release_safe=safe, independent_audit_passed=False, model_promoted=False,
            ambiguous_lock_scope=None if safe else "retained only while watchdog remains alive"))
        try:
            if created:
                write_new_json(output / "comparison-watchdog-exit.json", result)
        finally:
            if safe:
                fcntl.flock(lease.fileno(), fcntl.LOCK_UN)
                lease.close()
            else:
                _RETAINED_LOCKS.append(lease)
    return result


def supervise_audit(command, *, output, native_exit, session_id, runtime_fingerprint,
                    comparison_fingerprint, integrity_check, verify_report,
                    ops=None, poll_s=.5, execute_approved=False):
    """One fresh CPU-only audit child, at most two hours, after safe native exit.

    ``verify_report(output, native_exit_fingerprint)`` must independently verify
    the exact complete report and all input bindings; process exit cannot stand
    in for that check. The outer entry supplies the real verifier, never a stub.
    """
    output = _arguments(command, output, session_id, runtime_fingerprint, comparison_fingerprint,
                        integrity_check, poll_s, execute_approved)
    require(callable(verify_report), "Independent native report verifier required")
    require(type(native_exit) is dict and native_exit.get("schema") == SCHEMA
            and native_exit.get("fingerprint") == canonical_hash({k: v for k, v in native_exit.items() if k != "fingerprint"})
            and native_exit.get("session_id") == session_id
            and native_exit.get("runtime_fingerprint") == runtime_fingerprint
            and native_exit.get("comparison_fingerprint") == comparison_fingerprint
            and native_exit.get("status") == "native_exit_zero_independent_audit_required"
            and type(native_exit.get("exit_code")) is int and native_exit["exit_code"] == 0
            and native_exit.get("lock_release_safe") is True
            and type(native_exit.get("native_application_starts")) is int
            and native_exit["native_application_starts"] == 1,
            "Audit requires the exact successful native exit receipt")
    ops = ops or LinuxProcessOps()
    start, owner, watchdog = ops.monotonic(), None, None
    require(_clock(start), "Invalid independent audit clock")
    deadline, previous = start + AUDIT_WALL_S, start
    starts, process_starts, code, reason, error, cleanup, safe = 0, 0, None, None, None, None, True
    output.mkdir(mode=0o700)
    _sync_directory(output.parent)
    def check():
        nonlocal previous
        now = ops.monotonic()
        require(_clock(now) and previous <= now < deadline, "Independent audit deadline/clock failed")
        previous = now
        free = ops.free_bytes(output)
        require(type(free) is int and free >= MIN_FREE_BYTES,
                "Independent audit storage floor reached")
        require(integrity_check() is True, "Audit input integrity did not pass")
    try:
        watchdog = ops.identity(ops.own_pid())
        require(watchdog is not None and watchdog.get("cookie") == session_id, "Audit watchdog cookie missing")
        check()
        ops.prepare()
        check()
        environment = _environment(session_id, runtime_fingerprint, comparison_fingerprint, start, deadline,
                                   output / "comparison-audit-owner.json", audit=True)
        environment["SIRCUS_COMPARISON_NATIVE_EXIT_FINGERPRINT"] = native_exit["fingerprint"]
        starts = 1
        pid = ops.start(command, environment)
        process_starts = 1
        owner = ops.identity(pid)
        require(owner is not None and owner.get("cookie") == session_id
                and owner.get("ppid") == ops.own_pid() and owner.get("sid") == pid,
                "Audit child identity unavailable after sole start")
        write_new_json(output / "comparison-audit-owner.json", signed(dict(schema=AUDIT_SCHEMA,
            session_id=session_id, runtime_fingerprint=runtime_fingerprint,
            comparison_fingerprint=comparison_fingerprint, native_exit_fingerprint=native_exit["fingerprint"],
            owner=watchdog, native_owner=owner, command=command, start_monotonic=start,
            deadline_monotonic=deadline, cuda_visible_devices="")))
        while True:
            check()
            require(same_process(ops.identity(ops.own_pid()), watchdog), "Audit watchdog ownership changed")
            code = ops.poll(pid)
            if code is not None:
                require(type(code) is int and code == 0, "Audit child exited unsuccessfully")
                require(not ops.owned(pid, session_id), "Audit child left owned workers alive")
                require(verify_report(output, native_exit["fingerprint"]) is True,
                        "Independent audit report did not verify")
                check()
                break
            require(same_process(ops.identity(pid), owner), "Audit child ownership changed")
            ops.sleep(poll_s)
    except BaseException as failure:
        error, reason = type(failure).__name__, str(failure) or type(failure).__name__
        if starts:
            safe = False
            try:
                cleanup = stop_owned(ops, owner, session_id, deadline=min(deadline, ops.monotonic() + 35)) \
                    if owner is not None else "unknown_child_identity"
                safe = cleanup in ("stopped", "already_exited")
            except BaseException as stop_error:
                cleanup = "termination_unconfirmed:" + type(stop_error).__name__
    result = signed(dict(schema=AUDIT_SCHEMA, session_id=session_id,
        runtime_fingerprint=runtime_fingerprint, comparison_fingerprint=comparison_fingerprint,
        native_exit_fingerprint=native_exit["fingerprint"], owner=watchdog, native_owner=owner,
        status="audit_exit_zero_report_verified" if reason is None else "audit_stopped_no_retry",
        audit_process_starts=process_starts, audit_child_start_attempts=starts,
        exit_code=code, reason=reason, error_type=error,
        termination=cleanup, cleanup_safe=safe, start_monotonic=start, deadline_monotonic=deadline,
        wall_s=_final_wall(ops, start), cuda_visible_devices="", native_application_starts=0,
        independent_report_verified=reason is None, model_promoted=False))
    write_new_json(output / "comparison-audit-exit.json", result)
    return result
