"""Bounded cooperative 4090 scheduling. Never signal or alter another job."""

from contextlib import contextmanager
import fcntl
import os
import subprocess
import time

GPU_UUID = "GPU-ab1950b0-01dd-0fab-407f-2d17936b27f4"


def foreign_processes():
    if os.environ.get("CUDA_VISIBLE_DEVICES")!=GPU_UUID:
        raise RuntimeError("Select only the authorized RTX4090 UUID")
    raw=subprocess.check_output(["nvidia-smi","--query-compute-apps=gpu_uuid,pid","--format=csv,noheader"],text=True)
    return [int(line.split(',')[1]) for line in raw.splitlines()
            if line.split(',')[0].strip()==GPU_UUID and int(line.split(',')[1])!=os.getpid()]


@contextmanager
def gpu_slot(path, timeout_s=1800, poll_s=5, notify=None):
    start=time.monotonic()
    with open(path,'r') as lock:
        while True:
            owned=False
            try:
                fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);owned=True
                busy=foreign_processes()
            except BlockingIOError:
                busy=['shared-lock']
            if owned and not busy:
                break
            if owned:fcntl.flock(lock,fcntl.LOCK_UN)
            elapsed=time.monotonic()-start
            if notify:notify(dict(status='waiting_for_4090',wait_s=elapsed,reason=busy))
            if elapsed>=timeout_s:
                raise TimeoutError("4090 remained busy; no existing workload was interrupted")
            time.sleep(min(poll_s,timeout_s-elapsed))
        try:
            yield lock
        finally:
            fcntl.flock(lock,fcntl.LOCK_UN)
