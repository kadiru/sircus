"""Observed process resource use; sampled NVIDIA memory is not an exact peak."""

from contextlib import contextmanager
import os
import json
from pathlib import Path
import resource
import subprocess
import sys
import threading
import time


class ResourceMeter:
    def __init__(self, *, gpu_uuid, sample_s=1.0, log_path=None):
        self.gpu_uuid, self.sample_s = gpu_uuid, sample_s
        self.started = time.monotonic()
        self.wall, self.cpu = {"setup": 0.0, "train": 0.0, "evaluation": 0.0}, {
            "setup": 0.0,
            "train": 0.0,
            "evaluation": 0.0,
        }
        self.geometry_s, self.gpu_s, self.native_init_s = 0.0, 0.0, 0.0
        self.peak_vram_mb, self.samples, self.sampler_errors = None, 0, 0
        self.log_path = Path(log_path) if log_path is not None else None
        self.log_lock = threading.Lock()
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._watch, daemon=True)
        self.thread.start()

    def _watch(self):
        while not self.stop_event.is_set():
            observed = None
            try:
                raw = subprocess.check_output(
                    [
                        "nvidia-smi",
                        "--query-compute-apps=gpu_uuid,pid,used_memory",
                        "--format=csv,noheader,nounits",
                    ],
                    text=True,
                    timeout=5,
                )
                for line in raw.splitlines():
                    fields = [x.strip() for x in line.split(",")]
                    if (
                        len(fields) == 3
                        and fields[0] == self.gpu_uuid
                        and int(fields[1]) == os.getpid()
                    ):
                        value = float(fields[2])
                        observed = value
                        self.peak_vram_mb = (
                            value
                            if self.peak_vram_mb is None
                            else max(self.peak_vram_mb, value)
                        )
                        self.samples += 1
            except (OSError, subprocess.SubprocessError, ValueError):
                self.sampler_errors += 1
            self._record("nvidia_process_sample", dict(used_vram_mib=observed))
            self.stop_event.wait(self.sample_s)

    def _record(self, event, values):
        if self.log_path is not None:
            with self.log_lock, self.log_path.open("a") as stream:
                stream.write(
                    json.dumps(
                        dict(event=event, monotonic_s=time.monotonic(), values=values),
                        allow_nan=False,
                    )
                    + "\n"
                )

    @contextmanager
    def phase(self, phase):
        before, cpu = time.monotonic(), time.process_time()
        try:
            yield
        finally:
            self.wall[phase] += time.monotonic() - before
            self.cpu[phase] += time.process_time() - cpu

    def add_adapter(self, timing, *, training):
        self.native_init_s += float(timing.get("native_init_s", 0.0))
        if training:
            self.geometry_s += float(timing.get("geometry_s", 0.0))
            self.gpu_s += float(timing.get("gpu_s", 0.0))

    def snapshot(self):
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        rss_mb = rss / (1024**2 if sys.platform == "darwin" else 1024)
        allocated = reserved = None
        try:
            import torch

            if torch.cuda.is_initialized():
                allocated = torch.cuda.max_memory_allocated() / 1024**2
                reserved = torch.cuda.max_memory_reserved() / 1024**2
        except ImportError:
            pass
        result = dict(
            wall_s=self.wall["setup"] + self.wall["train"],
            cpu_s=self.cpu["setup"] + self.cpu["train"],
            geometry_s=self.geometry_s,
            gpu_s=self.gpu_s,
            setup_s=self.wall["setup"],
            evaluation_wall_s=self.wall["evaluation"],
            gpu_reservation_s=time.monotonic() - self.started,
            native_init_s=self.native_init_s,
            peak_rss_mb=rss_mb,
            peak_vram_mb=self.peak_vram_mb,
            torch_allocated_mb=allocated,
            torch_reserved_mb=reserved,
            vram_samples=self.samples,
            vram_sampler_errors=self.sampler_errors,
        )
        self._record("snapshot", result)
        return result

    def close(self):
        self.stop_event.set()
        self.thread.join(timeout=6)
        if self.thread.is_alive():
            raise RuntimeError("Resource sampler did not stop; evidence not sealed")
