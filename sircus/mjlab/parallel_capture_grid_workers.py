"""Prepare independent bucket layouts with CPU workers and shared immutable proofs."""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager
import hashlib
import json
import multiprocessing
import os

MAX_WORKERS = 8
_engine = None


def load_resolution(engine, bucket_xy_m):
    """Hydrate an existing exact resolver record without repeating the search."""
    from .benchmark_verify import contained, digest
    from .calibrated_adapter import vector
    from .calibrated_admission import VERSION, anchor_search_spec, verify_resolution

    resolver = engine.adapter.resolver.base
    bucket = vector(bucket_xy_m, 2)
    key = digest(dict(
        version=VERSION, bucket_xy_m=bucket, home_sha256=resolver.home["sha256"],
        sources=resolver.sources, bounds=resolver.bounds, anchor_search=anchor_search_spec(),
    ))
    if key in resolver._resolved:
        cached = resolver._resolved[key]
        resolver._check_resolution_bytes(cached)
        return cached
    relative = f"resolutions/{key}.json"
    path = contained(resolver.directory, relative)
    if not path.exists():
        return None
    raw = path.read_bytes()
    record = dict(json.loads(raw), file=relative, sha256=hashlib.sha256(raw).hexdigest())
    if record.get("key") != key or record.get("bucket_xy_m") != bucket:
        raise ValueError("Stored setup resolution does not match the requested geometry")
    verify_resolution(record, resolver.sources, resolver.home)
    resolver._check_resolution_bytes(record)
    resolver._resolved[key] = record
    return record


def _initialize_worker(ref, assets, directory, engine_kwargs):
    global _engine

    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.nice(10)

    import torch

    if torch.cuda.is_initialized():
        raise RuntimeError("Preparation workers require an uninitialized CUDA runtime")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    from .cell1050_heading_native import HeadingEngine

    _engine = HeadingEngine(ref, assets, directory, **engine_kwargs)


@contextmanager
def open_workers(count, ref, assets, directory, engine_kwargs):
    """Start bounded CPU processes independently of the parent CUDA context."""
    if type(count) is not int or not 1 <= count <= MAX_WORKERS:
        raise ValueError(f"Use between 1 and {MAX_WORKERS} preparation workers")
    pool = ProcessPoolExecutor(
        max_workers=count, mp_context=multiprocessing.get_context("spawn"),
        initializer=_initialize_worker, initargs=(ref, assets, directory, engine_kwargs),
    )
    try:
        yield pool
    finally:
        pool.shutdown(cancel_futures=True)


def prepare_proposals(proposals, warm):
    """Prepare one fixed-bucket batch without allocating the native environment."""
    if _engine is None:
        raise RuntimeError("The CPU preparation worker was not initialized")
    if not proposals:
        return [], []
    load_resolution(_engine, proposals[0]["bucket_xy_m"])
    result = _engine.prepare(proposals, one_per_setting=warm)
    if _engine.adapter.env is not None:
        raise RuntimeError("CPU preparation unexpectedly allocated a native environment")
    return result
