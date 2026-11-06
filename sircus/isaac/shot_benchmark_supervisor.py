#!/usr/bin/env python3
"""Bounded 4090-only supervisor for one versioned Sircus benchmark run.

No simulator is started by importing this module or using --plan-only. The normal
entry point detaches a supervisor that holds the existing read-opened scene lock
until its docker exec exits. Only the declared 4090 container may be used.
"""

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import signal
import subprocess
import sys
import tarfile
import time
import uuid

GPU_UUID = "GPU-ab1950b0-01dd-0fab-407f-2d17936b27f4"
CONTAINER = "sircus_isaac_4090"
WAIT_SECONDS = 43200
NATIVE_SECONDS = 57600


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def container_path(root, path):
    root, path = Path(root).resolve(), Path(path).resolve()
    try:
        relative = path.relative_to(root)
    except ValueError as exc:
        raise ValueError("Every mapped path must remain inside the existing workspace mount") from exc
    return str(PurePosixPath("/workspace") / relative.as_posix())


def verify_archive(archive, source, commit):
    """Bind a Git archive's commit and member bytes to the extracted source."""
    if not re.fullmatch(r"[a-f0-9]{40}", commit):
        raise ValueError("An exact 40-character Git commit is required")
    source = Path(source).resolve()
    files = {}
    with tarfile.open(archive, "r:*") as tar:
        if tar.pax_headers.get("comment") != commit:
            raise ValueError("Git archive PAX commit differs from the requested source commit")
        for member in tar:
            name = PurePosixPath(member.name)
            if name.is_absolute() or ".." in name.parts or "\\" in member.name:
                raise ValueError("Archive contains an unsafe source path")
            target = source / member.name
            if not target.resolve().is_relative_to(source):
                raise ValueError("Extracted source member leaves its snapshot")
            if member.isdir():
                continue
            if member.issym():
                if not target.is_symlink() or os.readlink(target) != member.linkname:
                    raise ValueError("Extracted source symlink differs from archive")
                files[member.name] = "symlink:" + member.linkname
                continue
            if not member.isfile() or target.is_symlink() or not target.is_file():
                raise ValueError("Extracted source member is missing or unsupported")
            expected = hashlib.sha256(tar.extractfile(member).read()).hexdigest()
            if sha(target) != expected:
                raise ValueError("Extracted source bytes changed: " + member.name)
            files[member.name] = expected
    if not files:
        raise ValueError("Empty source archive")
    return dict(archive_sha256=sha(archive), verified_source_files=len(files),
                source_members_sha256=files)


def load_native_args(path, replacements):
    values = json.loads(Path(path).read_text())
    if not isinstance(values, list) or not all(isinstance(v, str) and v for v in values):
        raise ValueError("Native arguments must be a JSON array of nonempty strings")
    protected = {"--assets", "--output", "--device", "--headless", "--enable_cameras"}
    result = []
    for value in values:
        if value.split("=", 1)[0] in protected:
            raise ValueError("Native arguments cannot override bounded launch settings")
        if value in replacements:
            value = replacements[value]
        elif "{" in value or "}" in value:
            raise ValueError("Unknown native argument placeholder")
        result.append(value)
    return result


def make_plan(args, session=None):
    root = args.workspace_host.resolve()
    source = args.source_host.resolve()
    output = args.output_host.resolve()
    helper = args.helper_host.resolve()
    isaac = source / "sircus/isaac"
    entry = isaac / args.entrypoint
    if (PurePosixPath(args.entrypoint).name != args.entrypoint or not args.entrypoint.endswith(".py")
            or not entry.is_file() or not (isaac / "python.sh").is_file()):
        raise ValueError("Native entry point must be an existing script in snapshot/sircus/isaac")
    if output.exists() or not output.parent.is_dir():
        raise ValueError("A new output path with an existing parent is required")
    for readonly in (source, args.source_run_host.resolve(), root / "assets"):
        if output.is_relative_to(readonly):
            raise ValueError("Output cannot be inside source, source data or camera directories")
    if not isinstance(args.batches, int) or not 1 <= args.batches <= 512:
        raise ValueError("Bounded benchmark requires 1..512 native TRAIN batches")
    integrity = verify_archive(args.archive_host, source, args.source_commit)
    required = [args.camera_host, args.native_args_json, helper,
                args.source_run_host / "result.json", args.source_run_host / "contract.json",
                root / "assets/poses.json", root / "assets/dropper.urdf", root / "assets/board_arm.urdf"]
    if not all(p.is_file() for p in required):
        raise ValueError("Existing camera, source run, launcher and generated assets are required")
    source_result = json.loads((args.source_run_host / "result.json").read_text())
    if source_result.get("workcells") != 35 or source_result.get("articulations") != 70:
        raise ValueError("Source must describe the existing 35-table/two-arm farm")
    mapped = {"source": container_path(root, source), "output": container_path(root, output),
              "camera": container_path(root, args.camera_host),
              "source_run": container_path(root, args.source_run_host),
              "helper": container_path(root, helper)}
    extra = load_native_args(args.native_args_json, {
        "{camera}": mapped["camera"], "{source_run}": mapped["source_run"],
        "{batches}": str(args.batches)})
    native = ["/usr/bin/timeout", "--signal=TERM", "--kill-after=30", str(NATIVE_SECONDS),
              "/bin/bash", mapped["source"] + "/sircus/isaac/python.sh",
              mapped["source"] + "/sircus/isaac/" + args.entrypoint,
              "--headless", "--enable_cameras", "--device", "cuda:0", "--assets", "/workspace/assets",
              "--output", mapped["output"], *extra]
    session = session or uuid.uuid4().hex
    native_receipt = output.with_name(output.name + ".native-exec.json")
    payload = dict(session=session, receipt=container_path(root, native_receipt),
                   native_argv=native, source_commit=args.source_commit)
    command = ["docker", "exec", "-u", "0", "--env", "SIRCUS_ISAAC_SOURCE=" + mapped["source"] + "/sircus/isaac",
               "--env", "PYTHONDONTWRITEBYTECODE=1", "--env", "CUDA_VISIBLE_DEVICES=0",
               "--env", "SIRCUS_CAPTURE_SESSION=" + session,
               "--workdir", mapped["source"] + "/sircus/isaac", CONTAINER,
               "/workspace/venv/bin/python", mapped["helper"], "--container-exec", json.dumps(payload)]
    return dict(session=session, created_at=now(), source_commit=args.source_commit,
                source_host=str(source), output_host=str(output), camera_host=str(args.camera_host.resolve()),
                source_run_host=str(args.source_run_host.resolve()), batches=args.batches,
                integrity=integrity, input_sha256={str(p.resolve()): sha(p) for p in required},
                lock_host=str(root / "scene.lock"), native_exec_receipt=str(native_receipt),
                container=CONTAINER, gpu_uuid=GPU_UUID, queue_timeout_s=WAIT_SECONDS,
                native_timeout_s=NATIVE_SECONDS, command=command,
                scope="One isolated native Isaac benchmark run; no other workload is signaled or restarted")


def require_running_container(run=subprocess.run):
    value = run(["docker", "inspect", CONTAINER], check=True, capture_output=True, text=True, timeout=10)
    data = json.loads(value.stdout)[0]
    devices = [d for request in data["HostConfig"].get("DeviceRequests", [])
               for d in request.get("DeviceIDs", [])]
    if not data["State"].get("Running") or devices != [GPU_UUID]:
        raise RuntimeError("Existing container must be running and assigned only the 4090 UUID")
    run(["docker", "exec", CONTAINER, "/bin/bash", "-c",
         "test -x /usr/bin/timeout && test -x /workspace/venv/bin/python"], check=True, timeout=10)


def foreign_processes(run=subprocess.run):
    result = run(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader"],
                 check=True, capture_output=True, text=True, timeout=10)
    return sorted(int(line.split(",")[1]) for line in result.stdout.splitlines()
                  if line.split(",")[0].strip() == GPU_UUID)


def atomic(path, value):
    path = Path(path)
    temp = path.with_name(path.name + ".tmp-" + str(os.getpid()))
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def wait_for_slot(lock, publish, *, busy=foreign_processes, monotonic=time.monotonic, sleep=time.sleep):
    started = monotonic()
    while True:
        acquired = False
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            acquired = True
            active = busy()
        except BlockingIOError:
            active = ["shared-lock"]
        if acquired and not active:
            return
        if acquired:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
        elapsed = monotonic() - started
        publish(dict(status="waiting_for_4090", elapsed_s=elapsed, blocking=active))
        if elapsed >= WAIT_SECONDS:
            raise TimeoutError("4090 remained busy; no existing job was interrupted")
        sleep(min(2, WAIT_SECONDS - elapsed))


def terminate_own_native(plan):
    """Signal only our mapped timeout PID after matching the session cookie."""
    receipt = Path(plan["native_exec_receipt"])
    if not receipt.exists():
        return False
    record = json.loads(receipt.read_text())
    if record.get("session") != plan["session"]:
        raise RuntimeError("Refusing to signal a mismatched native session")
    payload = json.dumps(dict(pid=record["native_timeout_pid"], session=plan["session"]))
    subprocess.run(["docker", "exec", "-u", "0", CONTAINER, "/workspace/venv/bin/python",
                    container_path(Path(plan["lock_host"]).parent, Path(__file__)),
                    "--container-terminate", payload], check=True, timeout=20)
    return True


def supervise(args):
    plan = make_plan(args, args.session)
    status_path = args.output_host.with_name(args.output_host.name + ".supervisor.json")
    def publish(value):
        atomic(status_path, dict(session=plan["session"], supervisor_pid=os.getpid(), updated_at=now(), **value))
    require_running_container()
    child = None
    try:
        # Open read-only: the existing root-owned lock is shared with other jobs.
        with Path(plan["lock_host"]).open("rb") as lock:
            wait_for_slot(lock, publish)
            # Recheck every source/input byte after any queue wait.
            current = make_plan(args, plan["session"])
            if current["integrity"] != plan["integrity"] or current["input_sha256"] != plan["input_sha256"]:
                raise RuntimeError("Frozen inputs changed while waiting")
            publish(dict(status="running", source_commit=plan["source_commit"], command=plan["command"]))
            child = subprocess.Popen(plan["command"], stdin=subprocess.DEVNULL)
            publish(dict(status="running", source_commit=plan["source_commit"],
                         docker_cli_pid=child.pid, native_exec_receipt=plan["native_exec_receipt"]))
            def cancel(signum, frame):
                terminate_own_native(plan)
            signal.signal(signal.SIGTERM, cancel)
            signal.signal(signal.SIGINT, cancel)
            try:
                # The timeout is inside the container, so a dead docker CLI
                # cannot leave an unbounded orphaned Isaac process.
                code = child.wait(timeout=NATIVE_SECONDS + 180)
            except subprocess.TimeoutExpired:
                terminate_own_native(plan)
                code = child.wait(timeout=90)
            native = json.loads(Path(plan["native_exec_receipt"]).read_text()) if Path(plan["native_exec_receipt"]).exists() else None
            publish(dict(status="completed" if code == 0 else "failed", returncode=code,
                         docker_cli_pid=child.pid, native_exec=native, source_commit=plan["source_commit"]))
            return code
    except Exception as exc:
        if child is not None and child.poll() is None:
            terminate_own_native(plan)
            child.wait(timeout=90)
        publish(dict(status="failed", error=type(exc).__name__ + ": " + str(exc)))
        raise


def container_exec(raw):
    value = json.loads(raw)
    if os.environ.get("SIRCUS_CAPTURE_SESSION") != value["session"]:
        raise RuntimeError("Native session cookie mismatch")
    path = Path(value["receipt"])
    if path.exists():
        raise RuntimeError("Refusing to replace native PID evidence")
    atomic(path, dict(session=value["session"], native_timeout_pid=os.getpid(), parent_pid=os.getppid(),
                      source_commit=value["source_commit"], started_at=now(), command=value["native_argv"]))
    os.execv(value["native_argv"][0], value["native_argv"])


def container_terminate(raw):
    value = json.loads(raw)
    pid = int(value["pid"])
    proc = Path("/proc") / str(pid)
    if not proc.exists():
        return
    environment = (proc / "environ").read_bytes().split(b"\0")
    if ("SIRCUS_CAPTURE_SESSION=" + value["session"]).encode() not in environment:
        raise RuntimeError("Refusing to signal another native process")
    command = (proc / "cmdline").read_bytes().split(b"\0")
    if not command or command[0] != b"/usr/bin/timeout":
        raise RuntimeError("Mapped native process is no longer the owned timeout")
    os.kill(pid, signal.SIGTERM)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source-host", "output-host", "camera-host", "source-run-host", "archive-host", "native-args-json"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--workspace-host", type=Path, default=Path("/home/kadir/sircus-isaac"))
    parser.add_argument("--helper-host", type=Path, default=Path(__file__).resolve())
    parser.add_argument("--entrypoint", default="film_cylinder_farm.py")
    parser.add_argument("--batches", type=int, default=30)
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--supervise", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--session", help=argparse.SUPPRESS)
    return parser.parse_args()


def main():
    if len(sys.argv) == 3 and sys.argv[1] == "--container-exec":
        return container_exec(sys.argv[2])
    if len(sys.argv) == 3 and sys.argv[1] == "--container-terminate":
        return container_terminate(sys.argv[2])
    args = parse_args()
    if args.supervise:
        return supervise(args)
    plan = make_plan(args)
    if args.plan_only:
        print(json.dumps(plan, indent=2))
        return 0
    require_running_container()
    receipt = args.output_host.with_name(args.output_host.name + ".launch.json")
    log = args.output_host.with_name(args.output_host.name + ".native.log")
    if receipt.exists() or log.exists() or Path(plan["native_exec_receipt"]).exists():
        raise ValueError("All launch/evidence paths must be new")
    with receipt.open("x") as stream:
        json.dump(dict(plan, status="launching"), stream, indent=2)
    with log.open("xb") as stream:
        child = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), *sys.argv[1:],
                                  "--supervise", "--session", plan["session"]],
                                 stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT,
                                 start_new_session=True)
    atomic(receipt, dict(plan, status="supervisor_started", supervisor_pid=child.pid,
                         supervisor_log=str(log), launch_helper_sha256=sha(Path(__file__))))
    print(json.dumps(dict(supervisor_pid=child.pid, receipt=str(receipt), log=str(log),
                         source_commit=plan["source_commit"], archive_sha256=plan["integrity"]["archive_sha256"],
                         queue_timeout_s=WAIT_SECONDS, native_timeout_s=NATIVE_SECONDS)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
