"""Independent bounds for one explicitly invoked continuous seed-22 child.

Import is inert and needs only the standard library. The child must separately
validate its source/model inputs, reserve requests before execution, and verify
that its sole CUDA device is the prescribed 4090. This supervisor is not a
physics or learning auditor. It does not restart a container or any service.
"""

import argparse
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

SCHEMA = "sircus-seed22-continuation-watchdog-v1"
MAX_WALL_S = 8 * 60 * 60
MAX_REQUESTS = 30000
MIN_FREE_BYTES = 6 * 1024 ** 3
MAX_PROGRESS_BYTES = 65536
# An ambiguous cleanup must retain its flock while this process remains alive.
# The exit receipt makes that scope explicit; a host must also refuse another
# launch until an unsafe receipt has been manually resolved.
_RETAINED_LOCKS = []


def require(value, message):
    if not value:
        raise ValueError(message)


def signed(value):
    return dict(value, fingerprint=canonical_hash(value))


def _number(value, lower, upper, label):
    require(type(value) in (int, float) and math.isfinite(value)
            and lower < value <= upper, "Invalid " + label)
    return value


def _absolute(value, label):
    path = Path(value)
    require(path.is_absolute() and ".." not in path.parts, "Expected absolute " + label)
    require(not any(p.is_symlink() for p in (path, *path.parents)), "Linked " + label)
    require(path == path.resolve(), "Noncanonical " + label)
    return path


def read_progress(path, *, session_id, runtime_fingerprint, previous_requests=0,
                  previous_completed=0, max_requests=MAX_REQUESTS):
    """Read one atomic, bound progress snapshot; missing startup progress is OK.

    Required fields are session_id, runtime_fingerprint, train_requested and a
    fingerprint over the complete object except fingerprint. Additional finite
    JSON fields are retained. Writers must use atomic replacement, not append.
    """
    path = Path(path)
    require(not path.is_symlink(), "Linked progress file")
    if not path.exists():
        return None
    require(path.is_file(), "Progress must be a plain file")
    with path.open("rb") as stream:
        raw = stream.read(MAX_PROGRESS_BYTES + 1)
    require(len(raw) <= MAX_PROGRESS_BYTES, "Progress exceeds 64 KiB")
    value = strict_json(raw)
    require(type(value) is dict and value.get("session_id") == session_id
            and value.get("runtime_fingerprint") == runtime_fingerprint,
            "Progress session or runtime binding changed")
    require(value.get("fingerprint") == canonical_hash(
        {k: v for k, v in value.items() if k != "fingerprint"}),
        "Progress fingerprint changed")
    requested = value.get("train_requested")
    require(type(requested) is int and previous_requests <= requested <= max_requests,
            "TRAIN requests exceeded cap or moved backwards")
    if "train_completed" in value:
        completed = value["train_completed"]
        require(type(completed) is int and previous_completed <= completed <= requested,
                "TRAIN completion counter invalid or moved backwards")
    return value


def supervise(command, *, output, lock, session_id, runtime_fingerprint,
              ops=None, wall_s=MAX_WALL_S, max_requests=MAX_REQUESTS,
              storage_floor_bytes=MIN_FREE_BYTES, poll_s=.5):
    """Start at most one child, with an independent wall clock and shared lock.

    The eight-hour interval begins on entry and includes child initialization,
    training, assessments and cleanup. Reaching the request cap does not kill a
    child that is saving/finally assessing; no TRAIN counter may exceed the cap.
    A zero process exit is reported only as awaiting independent audit. No old
    output directory or receipt can be reopened. Tests inject fake process ops.
    """
    ops = ops or LinuxProcessOps()
    start = ops.monotonic()
    _number(start, -1, float("inf"), "start clock")
    _number(wall_s, 0, MAX_WALL_S, "wall limit")
    _number(poll_s, 0, 1, "poll interval")
    require(type(max_requests) is int and 0 < max_requests <= MAX_REQUESTS,
            "Invalid TRAIN request cap")
    require(type(storage_floor_bytes) is int and storage_floor_bytes >= MIN_FREE_BYTES,
            "Storage floor cannot be below 6 GiB")
    require(type(session_id) is str and re.fullmatch(r"[0-9a-f]{32}", session_id),
            "Invalid session")
    require(type(runtime_fingerprint) is str and re.fullmatch(r"[0-9a-f]{64}", runtime_fingerprint),
            "Invalid runtime fingerprint")
    require(type(command) is list and command and all(type(x) is str and x and "\0" not in x for x in command)
            and Path(command[0]).is_absolute(), "Expected explicit argv with an absolute executable")
    output, lock = _absolute(output, "output"), _absolute(lock, "shared lock")
    require(output != lock and not lock.is_relative_to(output), "Shared lock must be outside new output")
    require(output.parent.is_dir() and lock.parent.is_dir(), "Output and lock parents must exist")
    require(not output.exists(), "Refuse existing output: this run cannot retry or overwrite")
    require(lock.is_file(), "Existing shared scene lock file required")
    deadline = start + wall_s
    import fcntl
    lock_flags = os.O_RDWR | getattr(os, "O_NOFOLLOW", 0)
    lock_descriptor = os.open(lock, lock_flags)
    if not stat.S_ISREG(os.fstat(lock_descriptor).st_mode):
        os.close(lock_descriptor)
        raise ValueError("Existing shared scene lock must be a regular file")
    lock_stream = os.fdopen(lock_descriptor, "r+")
    lock_identity = (os.fstat(lock_stream.fileno()).st_dev, os.fstat(lock_stream.fileno()).st_ino)
    try:
        fcntl.flock(lock_stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BaseException:
        lock_stream.close()
        raise
    owner, code, error, reason, cleanup = None, None, None, None, None
    last_time, requested, completed, last_progress = start, 0, 0, None
    child_starts, safe_to_unlock, created_output = 0, True, False
    try:
        output.mkdir(mode=0o700)
        created_output = True
        watchdog = ops.identity(ops.own_pid())
        require(watchdog is not None and watchdog.get("cookie") == session_id,
                "Watchdog must start with its session cookie in its environment")
        require(ops.free_bytes(output) >= storage_floor_bytes, "6 GiB storage floor reached before start")
        ops.prepare()
        now = ops.monotonic()
        require(math.isfinite(now) and start <= now < deadline, "Wall limit reached before child start")
        environment = dict(os.environ, **{
            COOKIE: session_id, "CUDA_VISIBLE_DEVICES": GPU_UUID,
            "CUDA_DEVICE_ORDER": "PCI_BUS_ID", "PYTHONDONTWRITEBYTECODE": "1",
            "SIRCUS_SEED22_DEADLINE_MONOTONIC": str(deadline),
            "SIRCUS_CONTINUATION_DEADLINE_MONOTONIC": str(deadline),
            "SIRCUS_CONTINUATION_STARTED_MONOTONIC": str(start),
            "SIRCUS_SEED22_SESSION": session_id,
            "SIRCUS_SEED22_RUNTIME_FINGERPRINT": runtime_fingerprint,
        })
        child_starts = 1
        pid = ops.start(command, environment)
        owner = ops.identity(pid)
        require(owner is not None and owner.get("cookie") == session_id
                and owner.get("ppid") == ops.own_pid() and owner.get("sid") == pid,
                "Child identity unavailable after sole start; no retry")
        write_new_json(output / "watchdog-owner.json", signed(dict(
            schema=SCHEMA, session_id=session_id, runtime_fingerprint=runtime_fingerprint,
            owner=watchdog, native_owner=owner, command=command, gpu_uuid=GPU_UUID,
            shared_lock=str(lock),
            shared_lock_identity=list(lock_identity),
            start_monotonic=start, deadline_monotonic=deadline,
            max_requests=max_requests, storage_floor_bytes=storage_floor_bytes)))
        while True:
            require(not lock.is_symlink() and (lock.stat().st_dev, lock.stat().st_ino) == lock_identity,
                    "Shared scene lock identity changed")
            now = ops.monotonic()
            require(type(now) in (int, float) and math.isfinite(now) and now >= last_time,
                    "Watchdog monotonic clock moved backwards")
            last_time = now
            require(now < deadline, "Independent eight-hour wall limit reached")
            require(ops.free_bytes(output) >= storage_floor_bytes, "Independent 6 GiB storage floor reached")
            progress = read_progress(output / "progress.json", session_id=session_id,
                runtime_fingerprint=runtime_fingerprint, previous_requests=requested,
                previous_completed=completed, max_requests=max_requests)
            require(progress is not None or last_progress is None,
                    "Previously published progress disappeared")
            if progress is not None:
                last_progress = progress
                requested = progress["train_requested"]
                completed = progress.get("train_completed", completed)
            now = ops.monotonic()
            require(type(now) in (int, float) and math.isfinite(now) and now >= last_time,
                    "Watchdog monotonic clock moved backwards")
            last_time = now
            require(now < deadline, "Independent eight-hour wall limit reached")
            code = ops.poll(pid)
            if code is not None:
                require(type(code) is int and code == 0, "Native child exited unsuccessfully")
                require(not ops.owned(pid, session_id), "Native child exited with owned workers still alive")
                require(last_progress is not None, "Native child exited without bound progress")
                break
            require(same_process(ops.identity(pid), owner), "Native child ownership changed")
            require(same_process(ops.identity(ops.own_pid()), watchdog), "Watchdog ownership changed")
            ops.sleep(min(poll_s, max(0, deadline - now)))
    except BaseException as failure:
        error = type(failure).__name__
        reason = str(failure) or error
        if child_starts:
            safe_to_unlock = False
            try:
                if owner is not None:
                    cleanup = stop_owned(ops, owner, session_id, deadline=deadline)
                    safe_to_unlock = cleanup in ("stopped", "already_exited")
                else:
                    cleanup = "unknown_child_identity"
            except BaseException as stop_error:
                cleanup = "termination_unconfirmed:" + type(stop_error).__name__
    finally:
        final_clock = ops.monotonic()
        elapsed = final_clock - start if type(final_clock) in (int, float) and math.isfinite(final_clock) else None
        result = signed(dict(schema=SCHEMA, session_id=session_id,
            runtime_fingerprint=runtime_fingerprint, native_owner=owner,
            status="native_exit_zero_independent_audit_required" if reason is None else "stopped_no_retry",
            exit_code=code, error_type=error, reason=reason, termination=cleanup,
            native_application_starts=child_starts, train_requested=requested,
            train_completed=completed, last_progress_fingerprint=(last_progress or {}).get("fingerprint"),
            start_monotonic=start, deadline_monotonic=deadline, wall_s=elapsed,
            lock_release_safe=safe_to_unlock,
            ambiguous_lock_scope="retained only while this watchdog process remains alive" if not safe_to_unlock else None,
            independent_audit_passed=False, model_promoted=False))
        try:
            if created_output:
                write_new_json(output / "watchdog-exit.json", result)
        finally:
            if safe_to_unlock:
                fcntl.flock(lock_stream.fileno(), fcntl.LOCK_UN)
                lock_stream.close()
            else:
                _RETAINED_LOCKS.append(lock_stream)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--lock", required=True)
    parser.add_argument("--session-id", required=True)
    parser.add_argument("--runtime-fingerprint", required=True)
    parser.add_argument("--wall-s", type=float, default=MAX_WALL_S)
    parser.add_argument("--max-requests", type=int, default=MAX_REQUESTS)
    parser.add_argument("--execute-approved", action="store_true")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    require(args.execute_approved, "Explicit approved execution flag required")
    require(Path(args.lock) == Path("/workspace/scene.lock"), "Production must use the existing shared scene lock")
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    result = supervise(command, output=args.output, lock=args.lock,
        session_id=args.session_id, runtime_fingerprint=args.runtime_fingerprint,
        wall_s=args.wall_s, max_requests=args.max_requests)
    print(json.dumps(result, sort_keys=True, allow_nan=False), flush=True)
    return 0 if result["status"] == "native_exit_zero_independent_audit_required" else 1


if __name__ == "__main__":
    raise SystemExit(main())
