"""External, finite recovery of the already-authorized six-run benchmark.

The frozen native source is never edited. Its original supervisor still owns
GPU admission, locking, timeouts and cancellation. Its original independent
auditor runs CPU-only in the producer container, preserving numerical IDs.
No launch occurs on import; previous failed coordinator evidence is retained.
"""
import argparse
from datetime import datetime, timezone
import fcntl
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import shot_benchmark_contract as contract
import shot_benchmark_supervisor as supervisor

SCHEMA = "sircus-shot-benchmark-recovery-v1"
STATUS_SCHEMA = "sircus-shot-benchmark-recovery-coordinator-v1"
AUDIT_SCHEMA = "sircus-shot-benchmark-independent-audit-v1"
AUDIT_SECONDS = 7200
TOTAL_SECONDS = 8 * 86400
FREE_BYTES = 6 * 1024**3


def require(value, message):
    if not value:
        raise ValueError(message)


def read(path):
    def invalid(value):
        raise ValueError("Nonfinite JSON: " + value)
    return json.loads(Path(path).read_bytes(), parse_constant=invalid)


def inside(root, name):
    path = Path(root) / name
    require(not path.is_symlink() and path.resolve().is_relative_to(Path(root).resolve()),
            "Evidence path leaves its original root")
    return path


def absent_run(dest):
    """An un-audited partial run is never resumed or paid for twice."""
    for path in [dest] + [dest.with_name(dest.name + suffix) for suffix in
            (".launch.json", ".native.log", ".native-exec.json", ".supervisor.json")]:
        require(not path.exists() and not path.is_symlink(),
                "Existing/partial run requires another explicit recovery: " + str(path))


def finite_deadline(failed):
    require(failed.get("status") == "failed", "Original coordinator did not fail")
    elapsed = failed.get("elapsed_s")
    require(type(elapsed) in (int, float) and math.isfinite(elapsed) and elapsed >= 0,
            "Original finite elapsed time is missing")
    updated = datetime.fromisoformat(failed["updated_utc"])
    require(updated.tzinfo is not None, "Original coordinator time lacks timezone")
    return updated.timestamp() - elapsed + TOTAL_SECONDS


def require_dead(pid, kill=os.kill):
    require(type(pid) is int and pid > 1, "Original coordinator owner PID is invalid")
    try:
        kill(pid, 0)
    except ProcessLookupError:
        return
    # Permission errors cannot establish absence; PID reuse also refuses safely.
    raise ValueError("Original coordinator PID is still present; refusing concurrent recovery")


def verify_inputs(args):
    root = args.experiment.resolve().parent
    workspace = args.workspace.resolve()
    require(root.is_relative_to(workspace), "Experiment must use the original workspace mount")
    experiment = read(args.experiment)
    contract.validate_protocol(experiment["protocol"])
    require(experiment.get("schema") == "sircus-shot-benchmark-experiment-v1"
            and experiment["source"]["commit"] == args.source_commit
            and experiment["source"]["archive_sha256"] == supervisor.sha(args.archive),
            "Frozen experiment/source identity changed")
    supervisor.verify_archive(args.archive, args.source, args.source_commit)
    # The external coordinator's imported validation helpers must be byte-equal
    # to C, even though its own source is independently identified.
    for module in (contract, supervisor):
        require(supervisor.sha(module.__file__) == supervisor.sha(
            args.source / "sircus/isaac" / Path(module.__file__).name),
            "Recovery helper differs from frozen native helper")
    declarations = experiment["runs"]
    order = experiment["protocol"]["run_order"]
    require(len(declarations) == len(order) == 6, "Exactly six declared method/seed runs required")
    indexed = {(item["method"], item["seed"]): item for item in declarations}
    require(len(indexed) == 6 and set(indexed) == {(r["method"], r["seed"]) for r in order},
            "Declared run identities differ from frozen protocol")
    for (method, seed), item in indexed.items():
        require(item["path"] == f"runs/{method}-seed{seed}", "Unexpected native run path")
        inside(root, item["path"])
    for path in (args.source, args.archive, args.camera, args.source_run, args.experiment):
        supervisor.container_path(workspace, path)
    require(supervisor.sha(args.camera) == experiment["source"]["camera_sha256"],
            "Original camera bytes changed")
    failed = read(root / "coordinator.json")
    require(failed.get("source_commit") == args.source_commit
            and failed.get("total_authorized_train_attempts") == 60000,
            "Original failed coordinator identity changed")
    require_dead(failed["pid"])
    deadline = finite_deadline(failed)
    require(time.time() < deadline, "Original eight-day authorization deadline expired")
    return root, experiment, indexed, failed, deadline


def verify_audit(path, dest, experiment, args, method, seed):
    """Rebind a passed original audit to current immutable evidence bytes."""
    audit = read(path)
    require(audit.get("schema") == AUDIT_SCHEMA and audit.get("status") == "passed"
            and audit.get("smoke_only") is False and audit.get("method") == method
            and audit.get("seed") == seed and audit.get("paid_train_attempts") == 10000
            and audit.get("evaluation_attempts") == 420,
            "Passed full-budget audit for this method/seed required")
    require(audit.get("source") == experiment["source"]
            and audit.get("experiment_sha256") == supervisor.sha(args.experiment)
            and audit.get("split_manifest_sha256") == experiment["split_manifest_sha256"],
            "Audit plan/source/split binding changed")
    split = inside(args.experiment.parent, experiment["split_manifest_file"])
    require(supervisor.sha(split) == audit["split_manifest_sha256"], "Split manifest bytes changed")
    for name, field in [("result.json", "source_result_sha256"),
                        ("progress.json", "source_progress_sha256"),
                        ("experiment-binding.json", "source_binding_sha256"),
                        ("contract.json", "source_contract_sha256")]:
        require(supervisor.sha(inside(dest, name)) == audit.get(field), "Audited " + name + " changed")
    isaac = args.source / "sircus/isaac"
    require(audit.get("source_sha256") == {p.name: supervisor.sha(p) for p in sorted(isaac.glob("*.py"))}
            and audit.get("audit_source_sha256") == supervisor.sha(isaac / "shot_benchmark_audit.py"),
            "Audit did not use unchanged original native source")
    for field, folder in (("source_trial_sha256", "trials"), ("checkpoint_sha256", "checkpoints")):
        references = audit.get(field)
        require(isinstance(references, dict) and bool(references), "Missing audited file bindings")
        for name, digest in references.items():
            candidate = inside(dest, name)
            require(candidate.parent == dest / folder and supervisor.sha(candidate) == digest,
                    "Audited trial/checkpoint bytes changed")
    require(set(audit["source_trial_sha256"]) == {
        str(p.relative_to(dest)) for p in (dest / "trials").glob("*.json.gz")},
        "Unlisted native trial appeared after audit")
    nodes = audit.get("trials")
    require(isinstance(nodes, list), "Audited native trial summaries missing")
    by_trial = {node["trial_file"]: node for node in nodes}
    require(len(by_trial) == len(nodes) and set(by_trial) == set(audit["source_trial_sha256"]),
            "Audited native trial summary membership changed")
    for name, compressed_sha in audit["source_trial_sha256"].items():
        node = by_trial[name]
        compressed = inside(dest, name)
        # In original C, checked.update(**receipt) overwrites node.sha256 with
        # the COMPRESSED trial digest while node.file still names its NPZ.
        # Preserve C's receipt exactly; recover the NPZ digest from the verified
        # lossless trial rather than confusing these two provenance fields.
        require(node.get("sha256") == compressed_sha and node.get("bytes") == compressed.stat().st_size,
                "Audited compressed trial summary changed")
        packet_receipt = inside(dest, name + ".receipt.json")
        require(supervisor.sha(packet_receipt) == node.get("receipt_sha256"),
                "Lossless trial receipt bytes changed")
        saved_receipt = read(packet_receipt)
        require(saved_receipt.get("file") == compressed.name
                and saved_receipt.get("sha256") == compressed_sha
                and saved_receipt.get("bytes") == compressed.stat().st_size,
                "Lossless trial receipt binding changed")
        with gzip.open(compressed, "rb") as stream:
            raw = stream.read(256 * 1024**2 + 1)
        require(len(raw) <= 256 * 1024**2, "Lossless trial exceeds original audit bound")
        raw_sha = hashlib.sha256(raw).hexdigest()
        require(raw_sha == node.get("uncompressed_sha256") == saved_receipt.get("uncompressed_sha256"),
                "Audited uncompressed trial bytes changed")
        trial = json.loads(raw)
        physics = trial.get("physics")
        if physics is None:
            require("file" not in node, "Rejected-only trial acquired physics after audit")
        else:
            require(isinstance(physics, dict) and physics.get("file") == node.get("file")
                    and supervisor.sha(inside(dest, physics["file"])) == physics.get("sha256"),
                    "Audited full-rate physics bytes changed")
    progress, result = read(dest / "progress.json"), read(dest / "result.json")
    require(progress.get("status") == result.get("status") == "completed"
            and progress.get("source") == result.get("source") == experiment["source"]
            and progress.get("method") == method and progress.get("seed") == seed
            and progress.get("smoke") is False and progress["train"]["attempts"] == 10000,
            "Native completed run identity changed")
    launch = read(dest.with_name(dest.name + ".launch.json"))
    native = read(dest.with_name(dest.name + ".native-exec.json"))
    finished = read(dest.with_name(dest.name + ".supervisor.json"))
    require(launch.get("source_commit") == native.get("source_commit") == finished.get("source_commit") == args.source_commit
            and launch.get("container") == supervisor.CONTAINER
            and launch.get("gpu_uuid") == supervisor.GPU_UUID
            and launch.get("source_host") == str(args.source.resolve())
            and launch.get("integrity", {}).get("archive_sha256") == experiment["source"]["archive_sha256"]
            and launch.get("session") == native.get("session") == finished.get("session")
            and finished.get("status") == "completed" and finished.get("returncode") == 0,
            "Original native completion/launch receipts changed")
    return audit


def cpu_audit_command(args, dest, output):
    mapped = lambda path: supervisor.container_path(args.workspace, path)
    source = mapped(args.source) + "/sircus/isaac"
    return ["docker", "exec", "-u", "0", "--env", "CUDA_VISIBLE_DEVICES=",
        "--env", "PYTHONDONTWRITEBYTECODE=1", "--env", "OMP_NUM_THREADS=1",
        "--env", "OPENBLAS_NUM_THREADS=1", "--env", "MKL_NUM_THREADS=1",
        "--env", "SIRCUS_ISAAC_SOURCE=" + source, "--workdir", source,
        supervisor.CONTAINER, "/usr/bin/timeout", "--signal=TERM", "--kill-after=30",
        str(AUDIT_SECONDS), "/bin/bash", source + "/python.sh", source + "/shot_benchmark_audit.py",
        "--run", mapped(dest), "--plan", mapped(args.experiment), "--output", mapped(output)]


def write_new_bytes(path, data):
    with Path(path).open("xb") as stream:
        stream.write(data)


def run(args, *, execute=subprocess.run, sleep=time.sleep):
    root, experiment, indexed, failed, deadline = verify_inputs(args)
    controls = root / "control"
    controls.mkdir(exist_ok=True)
    lock_path = controls / "recovery.lock"
    require(not lock_path.is_symlink(), "Recovery lock must be a local file")
    with lock_path.open("a+b") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        return run_locked(args, root, experiment, indexed, failed, deadline, execute, sleep)


def run_locked(args, root, experiment, indexed, failed, deadline, execute, sleep):
    controls = root / "control"
    receipt_path = controls / "recovery-launch.json"
    status_path = root / "recovery-coordinator.json"
    require(not receipt_path.exists() and not status_path.exists(), "A recovery already exists; refusing another owner")
    original_status = root / "coordinator.json"
    original_log = root / "coordinator.log"
    first = experiment["protocol"]["run_order"][0]
    dest = root / indexed[(first["method"], first["seed"])]["path"]
    canonical = dest.with_name(dest.name + ".audit.json")
    require(not canonical.exists(), "First canonical audit already exists; explicit recovery required")
    verify_audit(args.adopt_audit, dest, experiment, args, first["method"], first["seed"])
    # Refuse the entire recovery before spending anything if any later run is
    # already partially present. This is not a partial-native resume mechanism.
    for item in experiment["protocol"]["run_order"][1:]:
        absent_run(root / indexed[(item["method"], item["seed"])]["path"])
    require(shutil.disk_usage(root).free >= FREE_BYTES, "Less than 6 GiB free; no data are deleted")
    frozen_status = controls / "recovery-original-coordinator.json"
    frozen_log = controls / "recovery-original-coordinator.log"
    status_bytes, log_bytes = original_status.read_bytes(), original_log.read_bytes()
    require(read(original_status) == failed, "Failed coordinator evidence changed during preflight")
    write_new_bytes(frozen_status, status_bytes)
    write_new_bytes(frozen_log, log_bytes)
    write_new_bytes(canonical, args.adopt_audit.read_bytes())
    receipt = dict(schema=SCHEMA, source_commit=args.source_commit,
        source_archive_sha256=supervisor.sha(args.archive), experiment_sha256=supervisor.sha(args.experiment),
        original_failed_status_sha256=supervisor.sha(frozen_status),
        original_failed_log_sha256=supervisor.sha(frozen_log), recovery_script_sha256=supervisor.sha(__file__),
        source_host=str(args.source.resolve()), source_archive_host=str(args.archive.resolve()),
        frozen_failed_status_file=str(frozen_status.relative_to(root)),
        frozen_failed_log_file=str(frozen_log.relative_to(root)),
        deadline_utc=datetime.fromtimestamp(deadline, timezone.utc).isoformat(),
        adopted_audit=dict(method=first["method"], seed=first["seed"], source_file=str(args.adopt_audit.resolve()),
            source_sha256=supervisor.sha(args.adopt_audit), canonical_file=str(canonical.relative_to(root)),
            canonical_sha256=supervisor.sha(canonical)),
        scope="External finite recovery only; original native code, audit checks, budgets and run order unchanged")
    write_new_bytes(receipt_path, (json.dumps(receipt, indent=2, allow_nan=False) + "\n").encode())
    started = time.monotonic()
    def publish(state, **extra):
        supervisor.atomic(status_path, dict(schema=STATUS_SCHEMA, status=state, pid=os.getpid(),
            source_commit=args.source_commit, experiment_sha256=receipt["experiment_sha256"],
            original_failed_status_sha256=receipt["original_failed_status_sha256"],
            recovery_receipt_sha256=supervisor.sha(receipt_path), total_authorized_train_attempts=60000,
            updated_utc=datetime.now(timezone.utc).isoformat(), elapsed_s=time.monotonic()-started,
            deadline_utc=receipt["deadline_utc"], **extra))
    def check_deadline():
        require(time.time() < deadline, "Original eight-day deadline reached; no next work launched")
    try:
        publish("recovered", method=first["method"], seed=first["seed"])
        for ordered in experiment["protocol"]["run_order"][1:]:
            method, seed = ordered["method"], ordered["seed"]
            dest = root / indexed[(method, seed)]["path"]
            check_deadline()
            absent_run(dest)
            require(shutil.disk_usage(root).free >= FREE_BYTES, "Less than 6 GiB free; no data are deleted")
            supervisor.verify_archive(args.archive, args.source, args.source_commit)
            require(supervisor.sha(args.experiment) == receipt["experiment_sha256"], "Frozen experiment changed")
            dest.parent.mkdir(exist_ok=True)
            af = controls / f"{method}-seed{seed}-recovery-args.json"
            write_new_bytes(af, json.dumps(["--appearance", "silver", "--continuous-source", "{source_run}",
                "--continuous-camera-json", "{camera}", "--shot-benchmark-plan",
                supervisor.container_path(args.workspace, args.experiment), "--shot-benchmark-method", method,
                "--shot-benchmark-seed", str(seed)]).encode())
            helper = args.source / "sircus/isaac/shot_benchmark_supervisor.py"
            command = [sys.executable, str(helper), "--source-host", str(args.source), "--output-host", str(dest),
                "--camera-host", str(args.camera), "--source-run-host", str(args.source_run),
                "--archive-host", str(args.archive), "--native-args-json", str(af),
                "--source-commit", args.source_commit, "--workspace-host", str(args.workspace),
                "--helper-host", str(helper), "--batches", "289"]
            publish("launching", method=method, seed=seed)
            execute(command, check=True, timeout=60)
            state_path = dest.with_name(dest.name + ".supervisor.json")
            while True:
                check_deadline()
                state = read(state_path) if state_path.exists() else {"status": "queued"}
                publish("running", method=method, seed=seed, native_status=state)
                if state["status"] in ("completed", "failed"):
                    require(state["status"] == "completed" and state.get("returncode") == 0,
                            "Native run failed; later runs were not launched")
                    break
                sleep(15)
            done = dest.with_name(dest.name + ".audit.json")
            require(not done.exists(), "New audit destination already exists")
            publish("auditing", method=method, seed=seed)
            # The unchanged original auditor gets its original numerical runtime,
            # never a host-side approximation or a weakened ID/hash comparison.
            command = cpu_audit_command(args, dest, done)
            log_path = dest.with_name(dest.name + ".recovery-audit.log")
            with log_path.open("xb") as stream:
                execute(command, check=True, timeout=AUDIT_SECONDS + 60,
                        stdout=stream, stderr=subprocess.STDOUT)
            verify_audit(done, dest, experiment, args, method, seed)
        publish("completed")
    except BaseException as exc:
        publish("failed", error=type(exc).__name__ + ": " + str(exc))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("experiment", "source", "archive", "camera", "source-run", "workspace", "adopt-audit"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
