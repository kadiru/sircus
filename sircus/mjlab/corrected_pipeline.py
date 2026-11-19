"""Finite study then independent CPU audit, without restarting any service."""

import argparse
import datetime
import json
import os
from pathlib import Path
import subprocess
import sys

from .gpu_slot import GPU_UUID


def commands(args, executable=sys.executable, source=None):
    source = Path(source) if source else Path(__file__).resolve().parents[2]
    common = []
    for name in ("parent_study", "bank", "contact_reference", "assets"):
        common += ["--" + name.replace("_", "-"), str(getattr(args, name))]
    for name in ("source_study", "historical_study"):
        for value in getattr(args, name):
            common += ["--" + name.replace("_", "-"), str(value)]
    grid = [executable, "-u", "-m", "sircus.mjlab.corrected_grid", *common,
        "--output", str(args.output), "--gpu-lock", str(args.gpu_lock),
        "--queue-timeout", str(args.queue_timeout)]
    audit = [executable, "-u", "-m", "sircus.mjlab.corrected_verify", *common,
        "--study", str(args.output), "--source", str(source),
        "--output", str(args.output / "audit-corrected.json")]
    return grid, audit


def run(args):
    if os.environ.get("CUDA_VISIBLE_DEVICES") != GPU_UUID:
        raise RuntimeError("Expose only the authorized RTX4090 UUID")
    grid, audit = commands(args)
    status_file = args.output.with_name(args.output.name + "-pipeline.json")
    if status_file.exists() or args.output.exists():
        raise ValueError("Use a new study/output; preserve earlier runs")
    status = dict(version="sircus-corrected-grid-pipeline-v1", status="running",
        timestamp_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        grid_command=grid, audit_command=audit, grid_exit_code=None, audit_exit_code=None)
    def publish():
        temporary = status_file.with_suffix(".tmp")
        temporary.write_text(json.dumps(status, indent=2) + "\n")
        temporary.replace(status_file)
    publish()
    try:
        result = subprocess.run(grid, timeout=args.queue_timeout + 7200, check=False)
        status["grid_exit_code"] = result.returncode
        publish()
        if result.returncode:
            raise RuntimeError(f"Grid exited with status {result.returncode}; no audit or training started")
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = ""
        result = subprocess.run(audit, env=env, timeout=1200, check=False)
        status["audit_exit_code"] = result.returncode
        if result.returncode:
            raise RuntimeError(f"Independent CPU audit exited with status {result.returncode}")
        status["status"] = "completed"
    except Exception as exc:
        status.update(status="failed", error=str(exc))
        # Pipeline/audit failure is separate from native completion. Never
        # rewrite native evidence to imply its integration failed afterward.
        raise
    finally:
        publish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("parent-study", "bank", "contact-reference", "assets", "output", "gpu-lock"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--source-study", type=Path, action="append", required=True)
    parser.add_argument("--historical-study", type=Path, action="append", default=[])
    parser.add_argument("--queue-timeout", type=float, default=21600)
    args = parser.parse_args()
    if not 1 <= args.queue_timeout <= 21600:
        parser.error("Queue timeout must be finite and at most six hours")
    run(args)
