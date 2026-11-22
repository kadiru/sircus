"""Opt-in live stalled-reference fallback, gated by audited matched evidence.

Uses new proof directories per request; retains classic and fast planners.
Native release consumes only the exact saved preview, once, under the 4090 lease.
"""
from __future__ import annotations
import copy
import math
import time
import multiprocessing
import os
from pathlib import Path
import threading
from uuid import uuid4

from .interactive_fast_planner import FastPhysicsPlanner, verified_inputs
from .interactive_physics import _PreparedShot, _status, _trace, _configuration
from .stalled_reference_plan import StalledReferencePlanner

VERSION = "sircus-interactive-stalled-planner-v1"


def verified_stalled_inputs(args, classic):
    """Require exact audited selector/geometry lineage and a passing rollout rule."""
    from .benchmark_verify import digest, sha
    from .fast_planning_study import COPIED_INPUTS
    from .intergrid_route_refit_inputs import _audited_source
    from .stalled_reference_study import VERSION as STUDY_VERSION, promotion_decision
    from .stalled_reference_audit import VERSION as AUDIT_VERSION

    common, binding = verified_inputs(args, classic)
    report, manifest, audit, artifacts, contract = _audited_source(
        args.stalled_study, args.stalled_audit, STUDY_VERSION,
        (*COPIED_INPUTS, "choices.json", "layouts.json", "setup-home.json"))
    decision = promotion_decision(report["comparison"], report["diagnostic"])
    if (audit["version"] != AUDIT_VERSION or audit["comparison"] != report["comparison"]
            or report.get("decision") != decision or decision["eligible"] is not True
            or contract["methods"] != ["baseline", "hybrid"] or contract["test_positions"] != 20
            or contract["planning_budget_s"] != 30. or contract["reference_path_s"] != 5.
            or contract["promotion_time_ratio"] != 1.10 or contract["early_score_m"] != .005
            or contract["attempts_per_method_per_position"] != 1
            or report["budgets"] != dict(positions=20, planning_s=30., reference_path_s=5.)):
        raise ValueError("Hybrid planner has not passed its frozen rollout criteria")
    source = Path(__file__).resolve().parents[2]
    for name, expected in contract["source_sha256"].items():
        if sha(source / name) != expected:
            raise ValueError("Audited hybrid computational source changed: " + name)
    for name, expected in contract["frozen_inputs"].items():
        if sha(args.stalled_study / name) != expected:
            raise ValueError("Frozen hybrid input changed: " + name)
    for name in COPIED_INPUTS:
        if sha(args.stalled_study / name) != sha(args.fast_study / name):
            raise ValueError("Hybrid and existing planner input differs: " + name)
    from .benchmark_verify import read
    if (contract["scene"] != classic._native["ref"]["scene"]
            or contract["identity"]["scene_sha256"] != classic.provenance["scene_sha256"]
            or read(args.stalled_study / "setup-home.json") != classic._native["home"]
            or contract["flight_ranking"]["rule_sha256"] != classic.provenance["flight_rule_sha256"]):
        raise ValueError("Hybrid and existing planner geometry differs")
    binding.update(stalled_study_report_sha256=artifacts["report.json"],
        stalled_study_audit_sha256=artifacts["audit"],
        stalled_contract_fingerprint=contract["fingerprint"],
        stalled_choices_sha256=artifacts["choices.json"],
        stalled_source_manifest_sha256=digest(contract["source_sha256"]),
        reference_path_budget_s=5., promotion_decision=decision,
        live_adapter_sha256=sha(Path(__file__)))
    return common, binding


def _isolated_plan(layout, method, directory, *args, **kwargs):
    from .stalled_reference_plan import bind_live_request, stalled_reference_plan
    identifier = layout["id"]
    if not isinstance(identifier, str) or Path(identifier).name != identifier or identifier in ("", ".", ".."):
        raise ValueError("Fresh safe request identifier required")
    bind_live_request(Path(directory) / "requests" / identifier)
    return stalled_reference_plan(layout, method, *args, **kwargs)


def copy_prepared_proofs(source, destination, request):
    """Copy the selected context/connection and any bound reference fallback."""
    from .benchmark_verify import contained, read
    from .table_correction_native import copy_bound
    for name in ("context_resolution", "prepared_connection"):
        copy_bound(source, destination, request[name + "_file"], request[name + "_sha256"])
    record = read(contained(destination, request["prepared_connection_file"]))
    fallback = record.get("fallback_resolution")
    if fallback is not None:
        copy_bound(source, destination, fallback["file"], fallback["sha256"])


class LiveStalledPlanner(StalledReferencePlanner):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.worker.function = _isolated_plan
        self.worker.common_args = (self.directory, *self.worker.common_args)
        self._live_lock = threading.Lock()
        self._closed = False

    def plan(self, layout):
        if not self._live_lock.acquire(blocking=False):
            raise RuntimeError("Only one live planning request is allowed")
        try:
            if self._closed:
                raise RuntimeError("Planner has closed")
            self.seen_positions.clear()
            return super().plan(layout)
        except BaseException:
            self.worker.close()
            raise
        finally:
            self._live_lock.release()

    def close(self):
        with self._live_lock:
            self._closed = True
            super().close()


def _hybrid_native_drop(native, expected, directory, request_sha256, gpu_lock, gpu_wait_s, notify):
    from .benchmark_verify import read, sha
    from .calibrated_adapter import timing_record
    from .cell1050_heading_native import HeadingEngine
    from .cell1050_rebound_study import annotate
    from .gpu_slot import foreign_processes, gpu_slot
    from .intergrid_matched_route_study import outcome
    from .intergrid_route_labels import annotate_routes
    from .intergrid_route_selector_plan import STANDARD_CAPTURES
    from .intergrid_system_study import require_storage
    from .parallel_capture_grid_workers import load_resolution
    from .sweep_worker import atomic

    request_file = directory / "request.json"
    if sha(request_file) != request_sha256:
        raise ValueError("The prepared release request changed")
    request = read(request_file)
    native_dir = directory / "native-heading"
    require_storage(directory)
    from .stalled_reference_plan import request_mode
    from .direct_path_first import DirectPathEngine, hydrate_direct_resolution
    if request_mode(request) == 'direct_first':
        engine = DirectPathEngine(directory=native_dir, batch=1, **native)
        resolution = hydrate_direct_resolution(engine, request["bucket_xy_m"])
    else:
        engine = HeadingEngine(directory=native_dir, batch=1, **native)
        resolution = load_resolution(engine, request["bucket_xy_m"])
    if resolution is None or resolution["sha256"] != request["context_resolution_sha256"]:
        raise ValueError("The exact prepared bucket resolution is unavailable")
    def waiting(event):
        notify(dict(status=event["status"], wait_s=event["wait_s"],
                    message="Waiting for the 4090 to become available"))

    with gpu_slot(gpu_lock, timeout_s=gpu_wait_s, poll_s=1, notify=waiting):
        if foreign_processes():
            raise RuntimeError("The 4090 is busy")
        notify(dict(status="simulating", message="Simulating the previewed drop"))
        env = engine.adapter._environment(timing_record())
        if any(getattr(env, key) != value for key, value in expected.items()):
            raise RuntimeError("The native options or geometry differ from the audited setup")
        rows, timing = engine.evaluate([request], "interactive")
        if (len(rows) != 1 or rows[0]["id"] != request["id"] or rows[0]["native_release"] is not True
                or rows[0]["bucket_xy_m"] != request["bucket_xy_m"]
                or rows[0]["values"] != request["values"]):
            raise RuntimeError("The exact prepared request did not release")
    notify(dict(status="scoring", message="Verifying the recorded flight and capture"))
    annotate(rows, native_dir)
    annotate_routes(rows, native_dir)
    row = rows[0]
    actual = outcome(row)
    atomic(directory / "measurement.json", dict(request_sha256=request_sha256, row=row, timing=timing))
    return dict(
        status="completed", outcome=actual, capture=actual in STANDARD_CAPTURES,
        retention_s=row["max_retention_s"], trajectory=_trace(native_dir, row),
        native_cache_sha256=row["native_cache_sha256"], native_release_id=row["id"],
        scoring_version=row["scoring_version"], scalar_score_audit_passed=row["scalar_score_audit_passed"],
    )


def _hybrid_drop_worker(native, expected, directory, request_sha256, gpu_lock, gpu_wait_s, channel):
    """The one-shot worker owns its CUDA context, so exiting releases the GPU."""
    from .gpu_slot import GPU_UUID

    os.environ["CUDA_VISIBLE_DEVICES"] = GPU_UUID
    with channel:
        try:
            result = _hybrid_native_drop(native, expected, directory, request_sha256, gpu_lock, gpu_wait_s,
                                  lambda event: channel.send(("progress", event)))
        except TimeoutError as exc:
            channel.send(("busy", str(exc)))
        except Exception as exc:
            channel.send(("error", f"{type(exc).__name__}: {exc}"))
        else:
            channel.send(("result", result))


class StalledPhysicsPlanner(FastPhysicsPlanner):
    """Keep exact-preview native release, with the audited bounded CPU fallback."""
    def __init__(self, args, evidence_dir, *, classic):
        common, binding = verified_stalled_inputs(args, classic)
        self.__dict__.update(classic.__dict__)
        self._directory = Path(evidence_dir).resolve()
        self.provenance = dict(classic.provenance, **binding, version=VERSION,
            planner="frozen checked selector with bounded reference-path fallback")
        self._lane = self._directory / ("cpu-lane-" + uuid4().hex)
        native = self._native
        self._reusable = LiveStalledPlanner("warm_early", *common,
            reference=native["ref"], assets=native["assets"], directory=self._lane,
            engine_kwargs={key: native[key] for key in ("prior", "calibration", "home")},
            budget_s=30., reference_path_budget_s=5.)
        self._plan_lock = threading.Lock()

    def _plan(self, bucket_xy_m, progress):
        from .benchmark_verify import sha
        from .sweep_worker import atomic

        started = time.perf_counter()
        if (not isinstance(bucket_xy_m, (list, tuple)) or len(bucket_xy_m) != 2
                or any(type(v) not in (int, float) or not math.isfinite(v) for v in bucket_xy_m)):
            raise ValueError("Place the bucket at two finite coordinates in metres")
        xy = list(map(float, bucket_xy_m))
        preview = dict(bucket_xy_m=xy, status="blocked", planner="hybrid", candidate_pool=None,
                       route_intent=None, configuration=None, prediction=None, candidate_counts={},
                       provenance=dict(self.provenance), reason=self._placement_block(xy))
        if preview["reason"] is not None:
            preview["timing"] = dict(planning_wall_s=time.perf_counter()-started)
            return preview, None
        identity = "interactive-" + uuid4().hex
        layout = dict(id=identity, context_id=identity, bucket_xy_m=xy, split="test", stratum="interactive")
        _status(progress, "checking_hybrid", message="Checking ranked shots with bounded fallback (30-second decision limit)")
        site = self._reusable.plan(layout)
        choice = site["choice"]
        request = choice["request"]
        preview.update(status="feasible" if request else "unresolved",
                       candidate_pool=choice["candidate_pool"], route_intent=choice["intended_route"],
                       prediction=choice["prediction"],
                       candidate_counts={key: dict(proposed=len(stage["proposals"]), admitted=len(stage["admitted"]))
                                         for key, stage in site["plan"].items()},
                       reason=None if request else "No completely checked shot was ready within the planning limit.")
        directory = self._directory / identity
        directory.mkdir(parents=True, exist_ok=False)
        atomic(directory / "plan.json", site)
        preview["timing"] = dict(planning_wall_s=time.perf_counter()-started,
                                 decision_latency_s=site["decision_latency_s"], startup_latency_s=site["startup_s"],
                                 worker_reused=site["worker_reused"], termination=site["status"],
                                 return_reason=site["return_reason"], budget_s=30.,
                                 stalled_reference=site.get("stalled_reference"))
        if request is None:
            atomic(directory / "preview.json", preview)
            return preview, None
        if request["bucket_xy_m"] != xy:
            raise ValueError("Hybrid selector returned another bucket's shot")
        copy_prepared_proofs(self._lane / "requests" / identity,
                             directory / "native-heading", request)
        preview["configuration"] = _configuration(self._engine, request)
        atomic(directory / "request.json", request)
        request_sha256 = sha(directory / "request.json")
        preview["provenance"]["request_sha256"] = request_sha256
        preview["timing"]["planning_wall_s"] = time.perf_counter()-started
        atomic(directory / "preview.json", preview)
        return preview, _PreparedShot(directory, request_sha256, copy.deepcopy(preview))

    def drop(self, internal_plan, progress=None):
        from .sweep_worker import atomic

        if (not isinstance(internal_plan, _PreparedShot)
                or internal_plan.directory.parent != self._directory):
            raise ValueError("A prepared plan from this planner is required")
        directory = internal_plan.directory
        with (directory / "drop-started").open("x") as marker:
            marker.write(internal_plan.request_sha256 + "\n")
        _status(progress, "starting_native", message="Starting the isolated simulator")
        context = multiprocessing.get_context("spawn")
        incoming, outgoing = context.Pipe(duplex=False)
        proc = context.Process(target=_hybrid_drop_worker, args=(
            self._native, self._expected, directory, internal_plan.request_sha256,
            self._gpu_lock, self._gpu_wait_s, outgoing,
        ))
        with incoming, outgoing:
            proc.start()
            outgoing.close()
            try:
                measured = self._wait_for_release(proc, incoming, progress)
            finally:
                proc.join(timeout=2)
                if proc.is_alive():
                    proc.terminate()
                    proc.join(timeout=5)
                if proc.is_alive():
                    proc.kill()
                    proc.join()
                proc.close()
        provenance = dict(internal_plan.preview["provenance"],
                          native_cache_sha256=measured.pop("native_cache_sha256"),
                          native_release_id=measured.pop("native_release_id"))
        result = dict(internal_plan.preview, **measured, provenance=provenance)
        atomic(directory / "result.json", result)
        return result
