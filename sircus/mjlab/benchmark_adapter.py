"""Shared corrected-contact execution for the sweep/RL benchmark.

Both methods pay the same endpoint and sampled setup-path gates. A request's
TRAIN-derived teacher supplies geometry/joint continuation hints only. The
teacher's original bucket pose is retained, so a new query first resolves its
changed bucket-facing heading and proves its own nominal setup path.

Every requested action produces one result, including anchor, endpoint and path
rejections. There is no outcome search, retry, projection or fallback action.
"""

from __future__ import annotations

import copy
import math
from pathlib import Path
import shutil
import time

import numpy as np

from .corrected_contract import PHYSICS, sha
from .expanded_actions import WIDE_BOUNDS, checked_bounds
from .learning_contract import digest
from .sweep_worker import NativePlanner, atomic


VERSION = "sircus-sweep-rl-corrected-anchor-adapter-v1"
RESIDUAL_HALF_RANGE = np.asarray([0.04, 0.04, 0.05, 8.0, 0.12])
TIMING_SCOPE = (
    "gpu_s is synchronized native rollout wall time, not active GPU-utilization "
    "seconds; postprocess_s includes independent scalar scoring and evidence IO"
)


def jsonable(value):
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    if isinstance(value, np.generic):
        return jsonable(value.item())
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def vector(value, count, name):
    result = np.asarray(value, dtype=float)
    if result.shape != (count,) or not np.isfinite(result).all():
        raise ValueError(f"Finite {name}[{count}] required")
    return result.tolist()


def source_hint(value):
    """Strip flight labels: only exact teacher geometry may enter continuation."""
    if not isinstance(value, dict):
        raise ValueError("A shared TRAIN-derived source_hint is required")
    result = {
        key: vector(value.get(key), count, "source_hint." + key)
        for key, count in (("values", 5), ("bucket_xy_m", 2), ("q10", 6), ("q14", 6))
    }
    for key in (
        "id",
        "source_id",
        "source_record_sha256",
        "source_namespace",
        "configuration_sha256",
        "source_study",
    ):
        if key in value:
            if not isinstance(value[key], str) or not value[key]:
                raise ValueError("Nonempty source-hint provenance required")
            result[key] = value[key]
    return result


class BenchmarkAdapter(NativePlanner):
    """Per-method CPU/path caches and lazy native 4090 environment.

    Requests have ``id``, ``bucket_xy_m``, ``values[5]`` and ``source_hint``.
    ``evaluate`` returns ``(rows, timing)``. Callers own source/split/teacher-bank
    bindings, nominal residual-family bounds, GPU locking, resource sampling and
    phase budgets. Realized evaluation perturbations may leave the nominal
    residual envelope; physical global bounds and all path gates still apply.
    Construct one fresh adapter per method/seed so cache warm-up costs are not
    transferred.
    """

    def __init__(
        self,
        ref,
        assets,
        directory,
        *,
        batch=32,
        bounds=None,
        geometry=None,
        admission=None,
        environment=None,
        environment_factory=None,
        include_trajectory=False,
        max_evidence_bytes=4 * 1024**3,
        min_free_bytes=512 * 1024**2,
    ):
        if not 1 <= batch <= 32:
            raise ValueError("Native benchmark batch must be between 1 and 32")
        if geometry is None:
            from .corrected_geometry import CorrectedGeometry

            geometry = CorrectedGeometry(ref["scene"], assets)
        if admission is None:
            from .corrected_admission import TransitAdmission

            admission = TransitAdmission(
                ref["scene"], assets, home=ref["contract"]["common_home"]
            )
        self.ref, self.assets, self.directory = ref, Path(assets), Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.geometry, self.admission, self.batch = geometry, admission, batch
        self.bounds = copy.deepcopy(WIDE_BOUNDS if bounds is None else bounds)
        checked_bounds(self.bounds)
        self.env = environment
        self.environment_factory = environment_factory
        self.include_trajectory = bool(include_trajectory)
        self.max_evidence_bytes, self.min_free_bytes = (
            max_evidence_bytes,
            min_free_bytes,
        )
        self.packet, self.preparation_packet, self.rows = 0, 0, []
        self.counts = dict(
            cpu_candidates=0,
            geometry_rejected=0,
            search_requests=0,
            confirmation_requests=0,
        )
        self.totals = dict(
            requested=0,
            native_releases=0,
            anchors_prepared=0,
            anchor_cache_hits=0,
            local_path_cache_hits=0,
        )
        self._anchors, self._local_paths, self._ids = {}, {}, set()

    def storage_check(self):
        free = shutil.disk_usage(self.directory).free
        used = sum(p.stat().st_size for p in self.directory.rglob("*") if p.is_file())
        if free < self.min_free_bytes or used > self.max_evidence_bytes:
            raise RuntimeError(
                "Benchmark evidence storage limit reached; no files deleted"
            )

    def _save(self, name, value):
        path = self.directory / name
        if path.exists():
            raise ValueError("Refusing to overwrite benchmark evidence: " + name)
        atomic(path, jsonable(value))
        return dict(file=name, sha256=sha(path))

    def _geometry(self, requests, hints):
        from .corrected_geometry import REASONS

        xy = np.asarray([r["bucket_xy_m"] for r in requests])
        values = np.asarray([r["values"] for r in requests])
        mask, detail = self.geometry.audit_values(
            xy,
            values[:, None],
            bounds=self.bounds,
            known_q10=np.asarray([h["q10"] for h in hints]),
            known_q14=np.asarray([h["q14"] for h in hints]),
            known_values=np.asarray([h["values"] for h in hints]),
            known_bucket_xy=np.asarray([h["bucket_xy_m"] for h in hints]),
            chunk_size=32,
        )
        rows = [
            dict(
                id=r["id"],
                bucket_xy_m=r["bucket_xy_m"],
                values=r["values"],
                q10=detail["q10"][i, 0].tolist(),
                q14=detail["q14"][i, 0].tolist(),
                geometry_eligible=bool(mask[i, 0]),
                geometry_reason=REASONS[int(detail["reasons"][i, 0])],
            )
            for i, r in enumerate(requests)
        ]
        return rows, detail

    def _anchor(self, request, timing):
        hint, bucket = request["source_hint"], request["bucket_xy_m"]
        identity = dict(bucket_xy_m=bucket, source_hint=hint)
        key = digest(identity)
        if key in self._anchors:
            self.totals["anchor_cache_hits"] += 1
            return self._anchors[key]
        self.storage_check()
        record = dict(version=VERSION, key=key, request=identity, passed=False)
        self.totals["anchors_prepared"] += 1
        begin = time.monotonic()
        home = self.admission.check_home(bucket)
        timing["geometry_s"] += time.monotonic() - begin
        record["home_geometry"] = home
        if not home.get("passed"):
            record["reason"] = "anchor_common_home_clearance_unresolved"
        else:
            anchor_request = dict(
                id="anchor-" + key, bucket_xy_m=bucket, values=hint["values"]
            )
            begin = time.monotonic()
            rows, detail = self._geometry([anchor_request], [hint])
            timing["geometry_s"] += time.monotonic() - begin
            row = rows[0]
            record.update(geometry=detail, resolved=row)
            if not row["geometry_eligible"]:
                record["reason"] = "anchor_" + row["geometry_reason"]
            else:
                begin = time.monotonic()
                proof = self.admission.admit(row)
                timing["path_s"] += time.monotonic() - begin
                record["path"] = proof
                record["passed"] = bool(proof.get("passed"))
                record["reason"] = (
                    None if record["passed"] else "anchor_common_home_path_unresolved"
                )
        record.update(self._save("anchors/" + key + ".json", record))
        self._anchors[key] = record
        return record

    def _local_path(self, anchor, row, timing):
        identity = dict(
            anchor_sha256=anchor["sha256"],
            bucket_xy_m=row["bucket_xy_m"],
            values=row["values"],
            q10=row["q10"],
            q14=row["q14"],
        )
        key = digest(identity)
        if key in self._local_paths:
            self.totals["local_path_cache_hits"] += 1
            return self._local_paths[key]
        begin = time.monotonic()
        proof = self.admission.connect_variation(anchor["resolved"], row)
        timing["path_s"] += time.monotonic() - begin
        record = dict(version=VERSION, request=identity, proof=proof)
        record.update(self._save("local-paths/" + key + ".json", record))
        self._local_paths[key] = record
        return record

    @staticmethod
    def _synchronize(env):
        if str(getattr(env, "device", "")).startswith("cuda"):
            import torch

            torch.cuda.synchronize(env.device)

    def _environment(self, timing):
        if self.env is None:
            begin = time.monotonic()
            if self.environment_factory is None:
                from .contact_env import ContactEnvironment

                self.env = ContactEnvironment(
                    self.ref["scene"],
                    self.assets,
                    n=self.batch,
                    bounds=self.bounds,
                    substeps=PHYSICS["substeps"],
                    board_mode=PHYSICS["board_mode"],
                    gravity_mode=PHYSICS["gravity_mode"],
                )
            else:
                self.env = self.environment_factory()
            self._synchronize(self.env)
            timing["native_init_s"] += time.monotonic() - begin
        self.check_native_options(self.env)
        return self.env

    def check_native_options(self, env):
        options = env.sim.mj_model.opt
        actual = dict(
            timestep=float(options.timestep),
            solver=int(options.solver),
            cone=int(options.cone),
            integrator=int(options.integrator),
            iterations=int(options.iterations),
            tolerance=float(options.tolerance),
            gravity=np.asarray(options.gravity).tolist(),
        )
        if actual != self.ref["public"]["effective_native_options"]:
            raise ValueError("Effective native benchmark options changed")
        if (
            env.substeps != PHYSICS["substeps"]
            or env.board_mode != PHYSICS["board_mode"]
            or env.gravity_mode != PHYSICS["gravity_mode"]
        ):
            raise ValueError("Native benchmark contact protocol changed")

    def evaluate(self, requests, phase):
        """Return every requested trial in order plus measured phase timings."""
        begin = time.monotonic()
        timing = dict(
            geometry_s=0.0,
            path_s=0.0,
            gpu_s=0.0,
            postprocess_s=0.0,
            native_init_s=0.0,
            wall_s=0.0,
            requested=0,
            native_releases=0,
            scope=TIMING_SCOPE,
        )
        if not isinstance(phase, str) or not phase:
            raise ValueError("A nonempty benchmark phase is required")
        normalized = []
        for request in requests:
            if (
                not isinstance(request, dict)
                or not isinstance(request.get("id"), str)
                or not request["id"]
            ):
                raise ValueError("Each requested trial requires a nonempty ID")
            if request["id"] in self._ids or any(
                r["id"] == request["id"] for r in normalized
            ):
                raise ValueError("Benchmark trial ID must be unique")
            hint = source_hint(request.get("source_hint"))
            values = vector(request.get("values"), 5, "values")
            item = dict(
                id=request["id"],
                bucket_xy_m=vector(request.get("bucket_xy_m"), 2, "bucket_xy_m"),
                values=values,
                source_hint=hint,
            )
            for key in (
                "command_values",
                "perturbation_mixed_units",
                "normalized_residual",
            ):
                if key in request:
                    item[key] = vector(request[key], 5, key)
            normalized.append(item)
        if not normalized:
            timing["wall_s"] = time.monotonic() - begin
            return [], timing
        self.storage_check()
        self._ids.update(r["id"] for r in normalized)
        anchors = [self._anchor(r, timing) for r in normalized]
        candidates, details, live = [], None, []
        for i, (request, anchor) in enumerate(zip(normalized, anchors)):
            candidate = dict(
                id=request["id"],
                bucket_xy_m=request["bucket_xy_m"],
                values=request["values"],
                q10=[0.0] * 6,
                q14=[0.0] * 6,
                geometry_eligible=False,
                endpoint_eligible=None,
                geometry_row=None,
                anchor_admitted=anchor["passed"],
                geometry_reason=anchor.get("reason") or "request_not_checked",
                proposal=dict(
                    method="shared_anchor_residual_benchmark",
                    adapter_version=VERSION,
                    source_hint=request["source_hint"],
                    anchor_file=anchor["file"],
                    anchor_sha256=anchor["sha256"],
                    path_scope="fixed common home via query-specific nominal anchor and exact sampled local edges",
                ),
            )
            candidates.append(candidate)
            if anchor["passed"]:
                live.append(i)
        if live:
            start = time.monotonic()
            audited, details = self._geometry(
                [normalized[i] for i in live],
                [anchors[i]["resolved"] for i in live],
            )
            timing["geometry_s"] += time.monotonic() - start
            for geometry_index, (index, row) in enumerate(zip(live, audited)):
                candidates[index].update(
                    row,
                    endpoint_eligible=row["geometry_eligible"],
                    geometry_row=geometry_index,
                )
                if row["geometry_eligible"]:
                    path = self._local_path(anchors[index], row, timing)
                    proof = path["proof"]
                    candidates[index]["proposal"].update(
                        local_path_file=path["file"], local_path_sha256=path["sha256"]
                    )
                    candidates[index]["geometry_eligible"] = bool(proof.get("passed"))
                    if not proof.get("passed"):
                        candidates[index]["geometry_reason"] = (
                            proof.get("reason") or "local_path_unresolved"
                        )
        preparation_name = f"preparations/{self.preparation_packet:05d}.json"
        self.preparation_packet += 1
        preparation = self._save(
            preparation_name,
            dict(
                version=VERSION,
                phase=phase,
                requests=normalized,
                candidates=candidates,
                geometry_request_indices=live,
                geometry=details,
                common_home_sha256=self.admission.home["sha256"],
            ),
        )
        for i, candidate in enumerate(candidates):
            candidate["proposal"].update(
                preparation_file=preparation["file"],
                preparation_sha256=preparation["sha256"],
                preparation_row=i,
                geometry_row=candidate["geometry_row"],
            )
        self.counts["cpu_candidates"] += len(candidates)
        self.counts["geometry_rejected"] += sum(
            not c["geometry_eligible"] for c in candidates
        )
        env = (
            self._environment(timing)
            if any(c["geometry_eligible"] for c in candidates)
            else self.env
        )
        original_run = env.run if env is not None else None
        if original_run is not None:

            def measured_run(*args, **kwargs):
                self._synchronize(env)
                start = time.monotonic()
                try:
                    return original_run(*args, **kwargs)
                finally:
                    self._synchronize(env)
                    timing["gpu_s"] += time.monotonic() - start

            env.run = measured_run
        start = time.monotonic()
        try:
            rows = super().execute(env, candidates, phase)
        finally:
            if original_run is not None:
                env.run = original_run
        timing["postprocess_s"] = max(0.0, time.monotonic() - start - timing["gpu_s"])
        for row, candidate in zip(rows, candidates):
            row.update(
                endpoint_eligible=candidate["endpoint_eligible"],
                anchor_admitted=candidate["anchor_admitted"],
                geometry_eligible=candidate["geometry_eligible"],
                motion_evaluated="local_path_file" in candidate["proposal"],
                motion_admitted=candidate["geometry_eligible"],
                preparation_file=preparation["file"],
                preparation_sha256=preparation["sha256"],
                geometry_row=candidate["geometry_row"],
                anchor_file=candidate["proposal"]["anchor_file"],
                anchor_sha256=candidate["proposal"]["anchor_sha256"],
                adapter_version=VERSION,
            )
            for key in ("local_path_file", "local_path_sha256"):
                if key in candidate["proposal"]:
                    row[key] = candidate["proposal"][key]
            if not candidate["geometry_eligible"]:
                row["invalid"] = candidate["geometry_reason"]
            if not self.include_trajectory:
                row.pop("trajectory_m", None)
        self.totals["requested"] += len(rows)
        releases = sum(bool(r.get("native_release")) for r in rows)
        self.totals["native_releases"] += releases
        timing.update(
            requested=len(rows),
            native_releases=releases,
            wall_s=time.monotonic() - begin,
        )
        # NativePlanner's rows were durably written before returning; callers
        # receive complete gate provenance without retaining every trajectory.
        self.rows.clear()
        return rows, timing
