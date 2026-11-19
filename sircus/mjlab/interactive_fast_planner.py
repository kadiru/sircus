"""Live adapter for the unchanged, audited early-return CPU shot selector.

Each request receives a new proof directory, including repeated bucket XY.
Only initialized kinematics/models survive a clean worker completion. Native
release remains the existing isolated, single-use, exact-preview contract.
"""
from __future__ import annotations

import copy
import hashlib
import math
from pathlib import Path
import threading
import time
from uuid import uuid4

from .fast_planning_plan import ReusablePlanner, _reset_query_state, incremental_fast_plan
from .interactive_physics import PhysicsPlanner, _PreparedShot, _configuration, _status

VERSION = "sircus-interactive-fast-planner-v1"
METHOD = "warm_early"
# The published comparison selected for this deployment, not any mutable UI copy.
PUBLISHED = {
    "report.json": "9d2c0fdf022580e78bbb5368eab5012e164f827aa3914e128194cbea29cb66bb",
    "choices.json": "baa949b70fd51864449e73cc6e60754da3541a69ec33f840c5980605a3f2c4b6",
    "evidence-manifest.json": "7aa188566b3d9af7c7366ec62acd6f7568192634f36b197eb366cea69d035d51",
    "audit": "f0ef2fa51f1564844cbcb4a19c5a75582712708434b6aff72ad344b57ec489ca",
}


def _verify_published(archive, audit):
    for name, expected in PUBLISHED.items():
        path = audit if name == "audit" else archive / name
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"Published fast-planning evidence changed: {name}")


def verified_inputs(args, classic):
    """Fail closed on changed evidence, model lineage, physics or helper source."""
    _verify_published(args.fast_study, args.fast_audit)
    from .benchmark_verify import digest, read, sha
    from .fast_planning_study import COPIED_INPUTS, VERSION as STUDY_VERSION, audited_neural, audited_budget
    from .fast_planning_audit import VERSION as AUDIT_VERSION
    from .intergrid_route_labels import LABEL_VERSION
    from .intergrid_route_refit_inputs import _audited_source
    from .intergrid_library_plan import libraries

    archive = args.fast_study
    report, manifest, audit, artifacts, contract = _audited_source(
        archive, args.fast_audit, STUDY_VERSION, (*COPIED_INPUTS, "choices.json", "layouts.json"))
    if (audit["version"] != AUDIT_VERSION or audit["comparison"] != report["comparison"]
            or audit["choices_sha256"] != artifacts["choices.json"]
            or contract["methods"] != ["cold_30", "warm_30", METHOD]
            or contract["planning_budget_s"] != 30. or contract["early_score_m"] != .005
            or contract["physics_substeps"] != 64 or contract["label_version"] != LABEL_VERSION
            or contract["attempts_per_method_per_position"] != 1
            or contract["neural_parent"] != audited_neural(args)
            or contract["budget_parent"] != audited_budget(args)):
        raise ValueError("Fast planner differs from its audited method or model lineage")
    source = Path(__file__).resolve().parents[2]
    # The HTTP router changes in this integration; the planner, IK, physics,
    # scoring and all other previously recorded sources must remain unchanged.
    for name, expected in contract["source_sha256"].items():
        if name == "sircus/mjlab/interactive_physics_server.py":
            continue
        if sha(source / name) != expected:
            raise ValueError(f"Audited computational source changed: {name}")
    for name, expected in contract["frozen_inputs"].items():
        if sha(archive / name) != expected:
            raise ValueError(f"Frozen fast-planning input changed: {name}")
    for name in COPIED_INPUTS:
        if sha(archive / name) != sha(args.neural_study / name):
            raise ValueError(f"Frozen neural lineage differs: {name}")
    raw = read(archive / "shot-library-rows.json")
    libs = libraries(raw["original"], raw["additions"])
    native = classic._native
    system_contract = read(args.system_study / "contract.json")
    model, history = read(archive / "route-models.json"), read(archive / "route-history.json")
    if (contract["parent"] != system_contract["parent"]
            or contract["identity"] != system_contract["identity"]
            or contract["scene"] != native["ref"]["scene"]
            or contract["identity"]["scene_sha256"] != classic.provenance["scene_sha256"]
            or contract["identity"]["asset_sha256"] != native["ref"]["asset_hashes"]
            or libs != classic._libraries or model != classic._checkpoint or history != classic._history
            or read(archive / "training-prior.json") != native["prior"]
            or read(archive / "calibration.json") != native["calibration"]
            or contract["flight_ranking"]["rule_sha256"] != classic.provenance["flight_rule_sha256"]):
        raise ValueError("Fast planner and classic planner use different geometry or frozen inputs")
    provenance = dict(
        fast_study_report_sha256=artifacts["report.json"], fast_study_audit_sha256=artifacts["audit"],
        fast_choices_sha256=artifacts["choices.json"], fast_evidence_manifest_sha256=artifacts["evidence-manifest.json"],
        fast_contract_sha256=artifacts["contract.json"], fast_contract_fingerprint=contract["fingerprint"],
        neural_checkpoint_sha256=artifacts["neural-checkpoint.json"],
        neural_study_audit_sha256=sha(args.neural_audit), fast_method=METHOD,
        fast_source_manifest_sha256=digest(contract["source_sha256"]),
        live_adapter_sha256=sha(Path(__file__)), planning_budget_s=30., early_margin_m=.005,
    )
    config = dict(scene=contract["scene"], calibration=native["calibration"]["score_config"],
                  assets=str(args.assets), rule=contract["flight_ranking"]["rule"])
    return (libs, model, history, config, read(archive / "neural-checkpoint.json")), provenance


def _bind_fresh_request(engine, directory):
    """Preserve immutable CPU models, never per-request resolution/path evidence."""
    directory = Path(directory)
    _reset_query_state(engine)
    directory.mkdir(parents=True, exist_ok=False)
    engine.directory = directory
    engine.adapter.directory = directory
    # HeadingResolver delegates directory reads to its base resolver.
    engine.adapter.resolver.base.directory = directory
    if "directory" in vars(engine.adapter.resolver):
        raise RuntimeError("Unexpected shadow resolver directory")


def _isolated_plan(layout, method, directory, *args, **kwargs):
    from .parallel_capture_grid_workers import _engine

    identifier = layout["id"]
    if not isinstance(identifier, str) or Path(identifier).name != identifier or identifier in ("", ".", ".."):
        raise ValueError("Fresh safe request identifier required")
    _bind_fresh_request(_engine, Path(directory) / "requests" / identifier)
    return incremental_fast_plan(layout, method, *args, **kwargs)


class LiveReusablePlanner(ReusablePlanner):
    """Audited selector with request-scoped proofs instead of study-only XY IDs."""

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
            # New request IDs and exclusive proof-directory creation preserve
            # the frozen study's no-evidence-reuse guarantee even at repeated XY.
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


class FastPhysicsPlanner(PhysicsPlanner):
    """Reuse classic native release, with separately verified early-return plans."""

    def __init__(self, args, evidence_dir, *, classic):
        common, binding = verified_inputs(args, classic)
        # Share immutable reference/kinematics for preview rendering only. The
        # actual fast IK worker has its own process and per-request evidence.
        self.__dict__.update(classic.__dict__)
        self._directory = Path(evidence_dir).resolve()
        self.provenance = dict(classic.provenance, **binding, version=VERSION,
                               planner="frozen neural ordering with checked early return")
        self._lane = self._directory / ("cpu-lane-" + uuid4().hex)
        native = self._native
        self._reusable = LiveReusablePlanner(
            METHOD, *common, reference=native["ref"], assets=native["assets"], directory=self._lane,
            engine_kwargs={key: native[key] for key in ("prior", "calibration", "home")}, budget_s=30.)
        self._plan_lock = threading.Lock()

    def plan(self, bucket_xy_m, progress=None):
        if not self._plan_lock.acquire(blocking=False):
            raise RuntimeError("Only one live planning request is allowed")
        try:
            return self._plan(bucket_xy_m, progress)
        except BaseException:
            # A failed proof copy/preview also invalidates an otherwise idle
            # worker; the next request must initialize a clean one.
            self._reusable.worker.close()
            raise
        finally:
            self._plan_lock.release()

    def _plan(self, bucket_xy_m, progress):
        from .benchmark_verify import sha
        from .sweep_worker import atomic
        from .table_correction_native import copy_bound

        started = time.perf_counter()
        if (not isinstance(bucket_xy_m, (list, tuple)) or len(bucket_xy_m) != 2
                or any(type(v) not in (int, float) or not math.isfinite(v) for v in bucket_xy_m)):
            raise ValueError("Place the bucket at two finite coordinates in metres")
        xy = list(map(float, bucket_xy_m))
        preview = dict(bucket_xy_m=xy, status="blocked", planner="fast", candidate_pool=None,
                       route_intent=None, configuration=None, prediction=None, candidate_counts={},
                       provenance=dict(self.provenance), reason=self._placement_block(xy))
        if preview["reason"] is not None:
            preview["timing"] = dict(planning_wall_s=time.perf_counter()-started)
            return preview, None
        identity = "interactive-" + uuid4().hex
        layout = dict(id=identity, context_id=identity, bucket_xy_m=xy, split="test", stratum="interactive")
        _status(progress, "checking_fast", message="Checking ranked shots (30-second decision limit)")
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
                                 return_reason=site["return_reason"], budget_s=30.)
        if request is None:
            atomic(directory / "preview.json", preview)
            return preview, None
        if request["bucket_xy_m"] != xy:
            raise ValueError("Fast selector returned another bucket's shot")
        for name in ("context_resolution", "prepared_connection"):
            copy_bound(self._lane / "requests" / identity, directory / "native-heading",
                       request[name + "_file"], request[name + "_sha256"])
        preview["configuration"] = _configuration(self._engine, request)
        atomic(directory / "request.json", request)
        request_sha256 = sha(directory / "request.json")
        preview["provenance"]["request_sha256"] = request_sha256
        preview["timing"]["planning_wall_s"] = time.perf_counter()-started
        atomic(directory / "preview.json", preview)
        return preview, _PreparedShot(directory, request_sha256, copy.deepcopy(preview))

    def close(self):
        with self._plan_lock:
            self._reusable.close()
