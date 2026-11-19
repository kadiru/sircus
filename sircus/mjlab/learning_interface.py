"""HTTP adapter: a bounded child owns/releases only the explicitly selected 4090."""

import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile

GPU_UUID = "GPU-ab1950b0-01dd-0fab-407f-2d17936b27f4"
SOLVE_TIMEOUT_S = 240


def _terminate_group(process):
    """Cancel native work too, rather than leaving an orphan after HTTP timeout."""
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        process.communicate(timeout=5)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.communicate(timeout=5)


def solve(request):
    study = os.environ.get("SIRCUS_MJLAB_STUDY")
    assets = os.environ.get("SIRCUS_MJLAB_ASSETS")
    lock = os.environ.get("SIRCUS_MJLAB_GPU_LOCK")
    bank_study = os.environ.get("SIRCUS_MJLAB_BANK_STUDY")
    if not all((study, assets, lock)):
        raise RuntimeError(
            "Configure SIRCUS_MJLAB_STUDY, SIRCUS_MJLAB_ASSETS and "
            "SIRCUS_MJLAB_GPU_LOCK on the backend"
        )
    with tempfile.TemporaryDirectory(prefix="sircus-mjlab-request-") as directory:
        source, output = (
            Path(directory) / "request.json",
            Path(directory) / "result.json",
        )
        source.write_text(json.dumps(request, allow_nan=False))
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=GPU_UUID, WANDB_MODE="disabled")
        use_bank = bool(bank_study and request.get("mode")=="policy" and
                        not any(v is not None for v in request.get("constraints",{}).values()))
        command = [
                sys.executable,
                "-m",
                "sircus.mjlab.bank_solve" if use_bank else "sircus.mjlab.learning_solve",
                "--study",
                str(Path(bank_study if use_bank else study).resolve()),
                "--assets",
                str(Path(assets).resolve()),
                "--gpu-lock",
                str(Path(lock).resolve()),
                "--request",
                str(source),
                "--output",
                str(output),
            ]
        if use_bank:
            command.extend(["--parent-study",str(Path(study).resolve())])
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
                stdout, _ = process.communicate(timeout=SOLVE_TIMEOUT_S)
            except subprocess.TimeoutExpired:
                _terminate_group(process)
                raise RuntimeError(
                    f"Native simulation solve exceeded {SOLVE_TIMEOUT_S}s; "
                    "its process group was cancelled and the GPU lock released"
                ) from None
            if process.returncode or not output.is_file():
                raise RuntimeError(
                    "Native simulation solve failed: " + (stdout or "")[-1600:]
                )
            result = json.loads(output.read_text())
            if (
                not isinstance(result, dict)
                or result.get("request_id") != request["request_id"]
            ):
                raise RuntimeError("Native solver returned a mismatched request result")
            return result
        finally:
            # Also handles interruption of the HTTP worker itself.
            _terminate_group(process)
