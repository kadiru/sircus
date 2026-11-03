"""Bounded sequential 4090 comparison, live progress, then independent audit."""

import argparse
import datetime
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from .benchmark_contract import (
    VERSION,
    SEEDS,
    MILESTONES,
    load_prior,
    make_contract,
    study_identity,
    validate_report,
)
from .benchmark_run import signed
from .corrected_contract import sha
from .learning_contract import digest
from .sweep_worker import atomic, load_reference
from .gpu_slot import GPU_UUID


def read(path):
    return json.loads(Path(path).read_text())


def run(args):
    if os.environ.get("CUDA_VISIBLE_DEVICES") != GPU_UUID:
        raise RuntimeError("Expose only the authorized4090 UUID")
    output = args.output
    source = Path(__file__).resolve().parents[2]
    ref = load_reference(args.reference, args.assets)
    prior = load_prior(args.prior_bundle)
    spec = make_contract(
        ref, prior, seeds=args.seeds, budget=args.budget, milestones=args.milestones
    )
    controls = read(args.preflight_report)
    if (
        controls.get("status") != "passed"
        or controls.get("identity") != spec["identity"]
        or controls.get("prior_sha256") != digest(prior)
        or controls.get("source_sha256")
        != sha(Path(__file__).with_name("benchmark_smoke.py"))
        or controls.get("requests") != 50
        or controls.get("controls_captured") != 17
        or controls.get("deliberate_rejects") != 1
        or controls.get("discarded_ppo_requests") != 32
        or controls.get("actor_changed") is not True
        or controls.get("behavior_logp_max_error", 1.0) > 1e-6
    ):
        raise ValueError(
            "Matching native control/PPO smoke must pass before the benchmark"
        )
    control_directory = args.preflight_report.parent.resolve()
    for name, expected in read(control_directory / "manifest.json").items():
        path = (control_directory / name).resolve()
        if not path.is_relative_to(control_directory) or sha(path) != expected:
            raise ValueError("Native preflight manifest changed")
    spec.pop("fingerprint")
    spec["preflight"] = dict(
        report_sha256=sha(args.preflight_report),
        manifest_sha256=sha(control_directory / "manifest.json"),
        driver_sha256=controls["source_sha256"],
        requests=controls["requests"],
        native_releases=controls["native_releases"],
        scope="Separate TRAIN-only engineering controls; never training-budget or test evidence",
    )
    spec["fingerprint"] = digest(spec)
    if args.resume:
        if (
            not output.is_dir()
            or read(output / "contract.json") != spec
            or read(output / "prior.json") != prior
        ):
            raise ValueError(
                "Resume requires identical source, prior, layouts, packages and protocol"
            )
        if (
            output / "preflight.json"
        ).read_bytes() != args.preflight_report.read_bytes():
            raise ValueError("Copied native preflight evidence changed")
        if (
            sha(output / "preflight-manifest.json")
            != spec["preflight"]["manifest_sha256"]
        ):
            raise ValueError("Copied native preflight manifest changed")
    else:
        output.mkdir(parents=True, exist_ok=False)
        atomic(output / "contract.json", spec)
        atomic(output / "prior.json", prior)
        atomic(output / "splits.json", spec["splits"])
        (output / "preflight.json").write_bytes(args.preflight_report.read_bytes())
        (output / "preflight-manifest.json").write_bytes(
            (control_directory / "manifest.json").read_bytes()
        )
    lock = (output / "controller.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    logs = output / "logs"
    logs.mkdir(exist_ok=True)
    report = dict(
        version=VERSION,
        contract_fingerprint=spec["fingerprint"],
        protocol=spec["protocol"],
        status="queued" if args.prepare_only else "running",
        phase="prepared",
        curves=[],
        runs=[],
        final_test=[],
        scope=spec["protocol"]["scope"],
    )
    pairs = [
        (method, seed)
        for i, seed in enumerate(spec["protocol"]["seeds"])
        for method in (["sweep", "ppo"] if i % 2 == 0 else ["ppo", "sweep"])
    ]
    directories = [output / "runs" / f"{method}-seed{seed}" for method, seed in pairs]

    def publish(phase, *, completed=False):
        runs = []
        for directory in directories:
            path = directory / "report.json"
            if not path.is_file():
                continue
            value = read(path)
            unsigned = dict(value)
            actual = unsigned.pop("fingerprint", None)
            if (
                actual != digest(unsigned)
                or value["contract_fingerprint"] != spec["fingerprint"]
            ):
                raise ValueError("Worker report identity changed")
            runs.append(value)
        report["runs"] = runs
        report["curves"] = [point for item in runs for point in item["curves"]]
        report["final_test"] = [
            item["final_test"] for item in runs if item.get("final_test")
        ]
        report["updated_at"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        report["phase"] = phase
        if completed:
            report["status"] = "completed"
        value = signed(report)
        validate_report(value)
        atomic(output / "report.json", value)
        return value

    publish("prepared")
    if args.prepare_only:
        fcntl.flock(lock, fcntl.LOCK_UN)
        lock.close()
        return
    process = None
    try:
        for (method, seed), directory in zip(pairs, directories):
            existing = directory / "report.json"
            if existing.exists():
                value = read(existing)
                if (
                    value.get("status") == "completed"
                    and (directory / "manifest.json").exists()
                ):
                    for name, expected in read(directory / "manifest.json").items():
                        if sha(directory / name) != expected:
                            raise ValueError("Completed worker evidence changed")
                    continue
                raise ValueError(
                    "Incomplete worker exists; preserve it and use a new study"
                )
            command = [
                sys.executable,
                "-u",
                "-m",
                "sircus.mjlab.benchmark_run",
                "--study",
                str(output),
                "--reference",
                str(args.reference),
                "--assets",
                str(args.assets),
                "--gpu-lock",
                str(args.gpu_lock),
                "--method",
                method,
                "--seed",
                str(seed),
                "--queue-timeout",
                str(args.queue_timeout),
                "--wall-time-limit",
                str(args.worker_wall_time_limit),
            ]
            atomic(
                output / "active-worker.json",
                dict(method=method, seed=seed, command=command),
            )
            with (logs / f"{method}-seed{seed}.log").open("x") as log:
                process = subprocess.Popen(
                    command, cwd=source, stdout=log, stderr=subprocess.STDOUT
                )
                atomic(
                    output / "active-worker.json",
                    dict(method=method, seed=seed, pid=process.pid, command=command),
                )
                deadline = (
                    time.monotonic()
                    + args.queue_timeout
                    + args.worker_wall_time_limit
                    + 300
                )
                while process.poll() is None:
                    publish(f"{method} · seed {seed}")
                    if time.monotonic() > deadline:
                        process.terminate()
                        try:
                            process.wait(timeout=30)
                        except subprocess.TimeoutExpired:
                            process.kill()
                            process.wait()
                        raise TimeoutError("Own worker exceeded its finite deadline")
                    time.sleep(5)
                publish(f"{method} · seed {seed} exited")
                if process.returncode:
                    raise RuntimeError(
                        f"{method} seed {seed} exited {process.returncode}"
                    )
        # Complete bytes are frozen BEFORE audit. Controller state tracks the
        # audit separately, so its report SHA never changes after certification.
        publish("completed", completed=True)
        manifest = {
            str(p.relative_to(output)): sha(p)
            for p in output.rglob("*")
            if p.is_file()
            and p
            not in {
                output / "manifest.json",
                output / "controller.lock",
                output / "active-worker.json",
            }
        }
        atomic(output / "manifest.json", manifest)
        state = output.with_name(output.name + "-controller.json")
        atomic(
            state,
            dict(
                status="auditing",
                pid=os.getpid(),
                report_sha256=sha(output / "report.json"),
            ),
        )
        env = dict(os.environ, CUDA_VISIBLE_DEVICES="")
        audit = output.with_name(output.name + "-audit.json")
        command = [
            sys.executable,
            "-m",
            "sircus.mjlab.benchmark_verify",
            "--study",
            str(output),
            "--reference",
            str(args.reference),
            "--assets",
            str(args.assets),
            "--repository",
            str(source),
            "--output",
            str(audit),
        ]
        with output.with_name(output.name + "-audit.log").open("x") as log:
            checked = subprocess.run(
                command,
                cwd=source,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=3600,
            )
        if checked.returncode:
            atomic(state, dict(status="audit_failed", exit_code=checked.returncode))
            raise RuntimeError("Benchmark completed but independent audit failed")
        atomic(
            state,
            dict(
                status="completed",
                audit_sha256=sha(audit),
                report_sha256=sha(output / "report.json"),
            ),
        )
    except Exception as exc:
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        if report.get("status") != "completed":
            report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
            publish("failed")
        raise
    finally:
        fcntl.flock(lock, fcntl.LOCK_UN)
        lock.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "reference",
        "assets",
        "gpu-lock",
        "prior-bundle",
        "preflight-report",
        "output",
    ):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    parser.add_argument("--budget", type=int, default=1024)
    parser.add_argument("--milestones", type=int, nargs="+", default=MILESTONES)
    parser.add_argument("--queue-timeout", type=float, default=900)
    parser.add_argument("--worker-wall-time-limit", type=float, default=14400)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
