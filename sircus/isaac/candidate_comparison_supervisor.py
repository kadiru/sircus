"""Bounded native comparison using the existing exclusive 4090 supervisor.

The wrapper binds external comparison/model bytes in addition to finite
queue/native deadlines. Existing GPU owners, containers and services are never
signaled; termination remains restricted to the owned timeout session.
"""

import json
import os
from pathlib import Path, PurePosixPath
import subprocess
import sys

import shot_benchmark_supervisor as supervisor
from candidate_selector import validate_comparison

QUEUE_SECONDS = 7200
NATIVE_SECONDS = 1800
EXPECTED_PLAN_ENV = "SIRCUS_CANDIDATE_COMPARISON_SHA256"
_BASE_MAKE_PLAN = supervisor.make_plan


def _mapped_input(root, value):
    name = PurePosixPath(value)
    if (not name.is_absolute() or name.parts[:2] != ("/", "workspace")
            or ".." in name.parts or "\\" in value):
        raise ValueError("Comparison inputs must be absolute paths inside /workspace")
    root = Path(root).resolve()
    path = root.joinpath(*name.parts[2:])
    if path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(root):
        raise ValueError("Comparison input is missing or leaves the workspace")
    return path.resolve()


def make_plan(args, session=None):
    """Extend inherited pre/post-queue hashing to the plan and parent model."""
    plan = _BASE_MAKE_PLAN(args, session)
    values = json.loads(args.native_args_json.read_text())
    paths = []
    for index, value in enumerate(values):
        if value == "--candidate-comparison-plan":
            if index+1 >= len(values) or values[index+1].startswith("--"):
                raise ValueError("Missing comparison plan argument")
            paths.append(values[index+1])
        elif value.startswith("--candidate-comparison-plan="):
            paths.append(value.split("=",1)[1])
    if len(paths) != 1:
        raise ValueError("Exactly one immutable candidate comparison plan is required")
    comparison_path = _mapped_input(args.workspace_host, paths[0])
    comparison = json.loads(comparison_path.read_text())
    validate_comparison(comparison)
    binding = comparison.get("native_binding")
    if not isinstance(binding,dict):
        raise ValueError("Native comparison needs its complete producer binding before launch")
    if (binding["source_commit"] != args.source_commit
            or binding["source_archive_sha256"] != supervisor.sha(args.archive_host)):
        raise ValueError("Comparison plan producer/archive identity changed")
    checkpoint_path = _mapped_input(args.workspace_host, binding["checkpoint_path"])
    if supervisor.sha(checkpoint_path) != binding["checkpoint_sha256"]:
        raise ValueError("Frozen parent checkpoint bytes changed")
    comparison_sha = supervisor.sha(comparison_path)
    expected = os.environ.get(EXPECTED_PLAN_ENV)
    if expected is not None and expected != comparison_sha:
        raise ValueError("Comparison plan changed after the detached launch receipt")
    plan["input_sha256"].update({str(comparison_path):comparison_sha,
                                 str(checkpoint_path):binding["checkpoint_sha256"]})
    plan.update(comparison_plan_host=str(comparison_path), comparison_plan_sha256=comparison_sha,
                comparison_fingerprint=comparison["fingerprint"],
                frozen_checkpoint_host=str(checkpoint_path))
    return plan


def _run():
    if len(sys.argv) == 3 and sys.argv[1] == "--container-exec":
        return supervisor.container_exec(sys.argv[2])
    if len(sys.argv) == 3 and sys.argv[1] == "--container-terminate":
        return supervisor.container_terminate(sys.argv[2])
    args = supervisor.parse_args()
    if args.helper_host.resolve() != Path(__file__).resolve():
        raise ValueError("Comparison launcher requires its pinned wrapper as --helper-host")
    if args.batches != 2 or args.entrypoint != "film_cylinder_farm.py":
        raise ValueError("Only the two fixed comparison cohorts are permitted")
    if args.supervise:
        # supervise resolves its module-global planner at BOTH queue boundaries.
        # Restore it on return so mock/test callers cannot affect legacy tasks.
        original = supervisor.make_plan
        supervisor.make_plan = make_plan
        try:
            return supervisor.supervise(args)
        finally:
            supervisor.make_plan = original
    plan = make_plan(args)
    if args.plan_only:
        print(json.dumps(plan, indent=2))
        return 0
    supervisor.require_running_container()
    receipt = args.output_host.with_name(args.output_host.name + ".launch.json")
    log = args.output_host.with_name(args.output_host.name + ".native.log")
    if receipt.exists() or log.exists() or Path(plan["native_exec_receipt"]).exists():
        raise ValueError("All launch/evidence paths must be new; no partial run retries")
    with receipt.open("x") as stream:
        json.dump(dict(plan, status="launching"), stream, indent=2)
    with log.open("xb") as stream:
        environment = dict(os.environ)
        environment[EXPECTED_PLAN_ENV] = plan["comparison_plan_sha256"]
        child = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), *sys.argv[1:],
                                  "--supervise", "--session", plan["session"]],
                                 stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT,
                                 start_new_session=True, env=environment)
    supervisor.atomic(receipt, dict(plan, status="supervisor_started", supervisor_pid=child.pid,
        supervisor_log=str(log), launch_helper_sha256=supervisor.sha(Path(__file__))))
    print(json.dumps(dict(status="supervisor_started", pid=child.pid, session=plan["session"])))
    return 0


def main():
    previous = supervisor.WAIT_SECONDS, supervisor.NATIVE_SECONDS
    supervisor.WAIT_SECONDS = QUEUE_SECONDS
    supervisor.NATIVE_SECONDS = NATIVE_SECONDS
    try:
        return _run()
    finally:
        supervisor.WAIT_SECONDS, supervisor.NATIVE_SECONDS = previous


if __name__ == "__main__":
    sys.exit(main())
