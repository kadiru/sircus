"""Matched v2 requested-action admission, native execution and raw provenance."""

from __future__ import annotations

import copy
import json
from pathlib import Path
import shutil
import time

import numpy as np

from .calibrated_admission import GeometryResolver, source_geometry, verify_connection, verify_resolution
from .calibrated_score import validate_calibration
from .corrected_contract import sha
from .expanded_actions import WIDE_BOUNDS
from .learning_contract import digest


VERSION = "sircus-calibrated-requested-action-adapter-v2"


def vector(value, size):
    out = np.asarray(value, dtype=float)
    if out.shape != (size,) or not np.isfinite(out).all():
        raise ValueError("Finite native request vector required")
    return out.tolist()


def timing_record():
    return dict(geometry_s=0., path_s=0., gpu_s=0., postprocess_s=0., native_init_s=0., wall_s=0.,
                requested=0, native_releases=0, resolver_candidates=0, resolver_cache_hits=0,
                scope="Measured setup/admission/native pipeline wall time; not GPU kernel-active time")


def add_geometry_timing(target, record):
    if record.get("cache_hit"):
        target["resolver_cache_hits"] += 1
        return
    timing = record.get("timing", {})
    target["geometry_s"] += float(timing.get("home_s", 0.))+float(timing.get("endpoint_s", 0.))
    target["path_s"] += float(timing.get("path_s", 0.))
    target["resolver_candidates"] += len(record.get("attempts", []))


class CalibratedAdapter:
    def __init__(self, ref, assets, directory, *, prior_records, calibration, home, batch=32,
                 resolver=None, environment_factory=None, native_executor=None):
        self.ref, self.assets, self.directory = ref, Path(assets), Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.batch, self.bounds = batch, copy.deepcopy(WIDE_BOUNDS)
        self.calibration = validate_calibration(calibration)
        self.sources = copy.deepcopy(prior_records)
        self.resolver = resolver or GeometryResolver(ref, self.assets, self.directory, self.sources, home=home)
        self.home = self.resolver.home
        self.environment_factory, self.native_executor = environment_factory, native_executor
        self.env, self.packet, self._ids = None, 0, set()

    def storage_check(self):
        # Dense audit data are intentionally written on the roomy root output
        # filesystem. Refuse before a partial packet if headroom is exhausted.
        if shutil.disk_usage(self.directory).free < 2*1024**3:
            raise RuntimeError("Insufficient storage for dense calibrated native evidence")

    def resolve_context(self, context):
        begin = time.monotonic()
        bucket = vector(context["bucket_xy_m"], 2)
        record = self.resolver.resolve(bucket)
        verify_resolution(record, self.sources, self.home)
        measured = timing_record()
        add_geometry_timing(measured, record)
        measured["wall_s"] = time.monotonic()-begin
        return record, measured

    def _environment(self, timing):
        if self.env is None:
            start = time.monotonic()
            if self.environment_factory is not None:
                self.env = self.environment_factory()
            else:
                from .calibrated_env import CalibratedEnvironment
                self.env = CalibratedEnvironment(self.ref["scene"], self.assets, self.calibration,
                                                  n=self.batch, bounds=self.bounds)
            from .benchmark_adapter import BenchmarkAdapter
            BenchmarkAdapter.check_native_options(self, self.env)
            timing["native_init_s"] += time.monotonic()-start
            geometry = getattr(self.env, "compiled_geometry", None)
            if geometry is not None:
                path = self.directory/"native-compiled-geometry.json"
                raw = json.dumps(geometry, indent=2, allow_nan=False)+"\n"
                if path.exists() and path.read_text() != raw:
                    raise RuntimeError("Native compiled geometry changed")
                path.write_text(raw)
        return self.env

    def _native(self, candidates, timing):
        if self.native_executor is not None:
            return self.native_executor(candidates)
        from .bank_geometry import fixed_endpoints
        from .learning_contract import constraints
        env = self._environment(timing)
        bucket = np.asarray([r["bucket_xy_m"] for r in candidates])
        values = np.asarray([r["values"] for r in candidates])
        q10, q14 = (np.asarray([r[k] for r in candidates]) for k in ("q10", "q14"))
        with fixed_endpoints(env, bucket, values, q10, q14, np.ones(len(candidates), dtype=bool)):
            rows, metadata = env.run(bucket, values, constraints(len(candidates)), record=True)
        return rows, env.last_arrays, metadata

    def evaluate(self, requests, phase):
        begin, timing = time.monotonic(), timing_record()
        if not isinstance(phase, str) or not phase:
            raise ValueError("Nonempty calibrated trial phase required")
        normalized = []
        for request in requests:
            if not isinstance(request.get("id"), str) or not request["id"] or request["id"] in self._ids or any(r["id"] == request["id"] for r in normalized):
                raise ValueError("Unique nonempty requested trial ID required")
            value = copy.deepcopy(request)
            value["bucket_xy_m"] = vector(request.get("bucket_xy_m"), 2)
            value["values"] = vector(request.get("values"), 5)
            normalized.append(value)
        self._ids.update(r["id"] for r in normalized)
        self.storage_check()
        candidates = []
        for request in normalized:
            resolution = self.resolver.resolve(request["bucket_xy_m"])
            verify_resolution(resolution, self.sources, self.home)
            # Normally a cache hit, since the worker resolved before predicting.
            # A direct engineering call must still pay any newly performed work.
            add_geometry_timing(timing, resolution)
            for key, expected in (("context_resolution_file", resolution["file"]),
                                  ("context_resolution_sha256", resolution["sha256"]),
                                  ("context_home_sha256", self.home["sha256"])):
                if request.get(key) != expected:
                    raise ValueError("Requested action lost its pre-policy setup identity")
            if source_geometry(request.get("source_hint")) != source_geometry(resolution["source_hint"]):
                raise ValueError("Requested source differs from pre-policy resolver")
            if request.get("effective_source_hint") != resolution["effective_source_hint"]:
                raise ValueError("Requested effective anchor changed after prediction")
            if request.get("source_teacher_id") != resolution["source_hint"]["id"]:
                raise ValueError("Requested source teacher identity changed before prediction")
            for key, expected in (("context_layout_eligible", resolution["layout_eligible"]),
                                  ("context_anchor_admitted", resolution["anchor_admitted"]),
                                  ("context_hard_reject", resolution["hard_reject"])):
                if request.get(key) is not expected:
                    raise ValueError("Requested action changed its pre-policy setup status")
            connection = self.resolver.connect(resolution, request["values"])
            verify_connection(connection, resolution, self.home, request["values"])
            add_geometry_timing(timing, connection)
            row = dict(
                id=request["id"], bucket_xy_m=request["bucket_xy_m"], values=request["values"], phase=phase,
                native_release=False, invalid=connection.get("reason") or "requested_setup_not_admitted",
                success=False, complete_success=False, route_success=False, route_quality=0,
                single_bounce_success=False, board_bounce=False, whole_ball_entry=False,
                max_retention_s=0., closest_opening_distance_m=2., external_contact_steps=0,
                board_contact_episodes=0, invalid_physics=False,
                geometry_eligible=bool(connection["passed"]), motion_admitted=bool(connection["passed"]),
                endpoint_eligible=connection["endpoint_eligible"],
                geometry_reason=connection.get("reason"), q10=connection["q10"], q14=connection["q14"],
                connection_file=connection["file"], connection_sha256=connection["sha256"],
                context_resolution_file=resolution["file"], context_resolution_sha256=resolution["sha256"],
                context_home_sha256=self.home["sha256"], context_layout_eligible=resolution["layout_eligible"],
                context_anchor_admitted=resolution["anchor_admitted"], context_hard_reject=resolution["hard_reject"],
                source_hint=copy.deepcopy(request["source_hint"]), effective_source_hint=copy.deepcopy(request["effective_source_hint"]),
                source_teacher_id=request["source_teacher_id"],
                action_substituted=False, adapter_version=VERSION,
            )
            candidates.append(row)
        for offset in range(0, len(candidates), self.batch):
            group = candidates[offset:offset+self.batch]
            active = [i for i, row in enumerate(group) if row["geometry_eligible"]]
            if active:
                self.storage_check()
                native, arrays, metadata = self._native([group[i] for i in active], timing)
                if len(native) != len(active):
                    raise RuntimeError("Native engine changed requested-action count")
                timing["gpu_s"] += float(metadata.get("native_rollout_s", metadata.get("seconds", 0.)))
                post_start = time.monotonic()
                filename = f"native-{self.packet:04d}.npz"
                self.packet += 1
                native_path = self.directory/filename
                if native_path.exists():
                    raise ValueError("Refusing to replace native evidence")
                np.savez_compressed(native_path, **arrays, record_ids=np.asarray([group[i]["id"] for i in active]))
                native_sha256 = sha(native_path)
                for column, (index, measured) in enumerate(zip(active, native)):
                    row = group[index]
                    if measured.get("native_release") is not True or measured.get("values") != row["values"] or measured.get("bucket_xy_m") != row["bucket_xy_m"]:
                        raise RuntimeError("Native observation/action association changed")
                    for key in ("q10", "q14"):
                        if not np.allclose(measured[key], row[key], atol=1e-12, rtol=0):
                            raise RuntimeError("Native joint branch differs from exact path proof")
                    row.update(copy.deepcopy(measured), native_cache_file=filename, native_cache_row=column,
                               native_cache_sha256=native_sha256)
                    # The native row's invalid value denotes numerical-domain
                    # failures; geometry admission was already proven above.
                    if row.get("invalid_physics"):
                        row.update(invalid="native_physics_outside_calibrated_domain", route_success=False,
                                   success=False, complete_success=False, route_quality=0, single_bounce_success=False)
                    else:
                        row["invalid"] = None
                    timing["native_releases"] += 1
                timing["postprocess_s"] += time.monotonic()-post_start
                timing["postprocess_s"] += float(metadata.get("score_and_oracle_s", 0.))+float(metadata.get("raw_transfer_s", 0.))
        with (self.directory/"shots.jsonl").open("a") as stream:
            for row in candidates:
                stream.write(json.dumps({k: v for k, v in row.items() if k != "trajectory_m"}, allow_nan=False)+"\n")
        timing.update(requested=len(candidates), wall_s=time.monotonic()-begin)
        return candidates, timing
