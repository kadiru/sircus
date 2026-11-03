"""Backend-only, bounded child adapter for the independent sweep planner."""

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

from .learning_interface import _terminate_group
from .gpu_slot import GPU_UUID

TIMEOUT_S = 600


def solve(request):
    if request.get("mode") != "search":
        raise ValueError(
            "This simulation endpoint serves sweep search only; no learned policy is configured"
        )
    names = {
        key: os.environ.get("SIRCUS_SWEEP_" + key.upper())
        for key in ("reference", "assets", "lock", "cache")
    }
    if not all(names.values()):
        raise RuntimeError(
            "Backend needs SIRCUS_SWEEP_REFERENCE, ASSETS, LOCK and CACHE"
        )
    with tempfile.TemporaryDirectory(prefix="sircus-sweep-request-") as temp:
        source, output = Path(temp) / "request.json", Path(temp) / "result.json"
        child_request = dict(request)
        # Scale CPU proposals with the selected native budget while preserving
        # an explicitly supplied backend override.
        child_request.setdefault(
            "cpu_candidates", max(64, 4 * request.get("search_budget", 64))
        )
        source.write_text(json.dumps(child_request, allow_nan=False))
        env = dict(
            os.environ,
            CUDA_VISIBLE_DEVICES=GPU_UUID,
            WANDB_MODE="disabled",
            OMP_NUM_THREADS="2",
            OPENBLAS_NUM_THREADS="2",
            MKL_NUM_THREADS="2",
        )
        command = [
            sys.executable,
            "-m",
            "sircus.mjlab.sweep_worker",
            "--request",
            str(source),
            "--output",
            str(output),
        ]
        for key, flag in (
            ("reference", "reference"),
            ("assets", "assets"),
            ("lock", "gpu-lock"),
            ("cache", "cache"),
        ):
            command += ["--" + flag, str(Path(names[key]).resolve())]
        process = subprocess.Popen(
            command,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
        try:
            try:
                stdout, _ = process.communicate(timeout=TIMEOUT_S)
            except subprocess.TimeoutExpired:
                _terminate_group(process)
                raise RuntimeError(
                    "Sweep planner exceeded its 600-second request budget; its child process was cancelled"
                ) from None
            if process.returncode or not output.is_file():
                raise RuntimeError("Sweep planner failed: " + (stdout or "")[-1400:])
            result = json.loads(output.read_text())
            if (
                not isinstance(result, dict)
                or result.get("request_id") != request["request_id"]
            ):
                raise RuntimeError("Sweep planner returned a mismatched request")
            return result
        finally:
            _terminate_group(process)
