"""Preview and release one exact audited mjlab shot under the 4090 lease."""

from __future__ import annotations

from dataclasses import dataclass
import importlib.metadata
import math
import multiprocessing
import os
from pathlib import Path
import time
from uuid import uuid4


VERSION = "sircus-interactive-physics-v1"
SUBSTEPS = 64
DROP_TIMEOUT_S = 10 * 60


@dataclass(frozen=True)
class _PreparedShot:
    directory: Path
    request_sha256: str
    preview: dict


def _status(progress, status, **details):
    if progress is not None:
        progress(dict(status=status, **details))


def _ui_scene(scene, bounds):
    """Map the pinned simulator dimensions to the existing browser renderer."""
    bucket, board, table = (scene[name] for name in ("bucket", "board", "table"))
    return dict(
        version=VERSION, units="m", simulation_only=True,
        table=dict(center=table["center_m"], size=table["size_m"]),
        bases=[[0, 0, 0], [scene["arms"]["base_spacing_m"], 0, 0]],
        board=dict(size=board["size_m"], center=board["initial_center_m"]),
        ball=dict(radius=scene["ball"]["radius_m"]),
        bucket=dict(
            xy=bucket["center_xy_m"], inside_top_diameter=bucket["inner_top_diameter_m"],
            inside_bottom_diameter=bucket["inner_bottom_diameter_m"],
            outside_top_diameter=bucket["outer_top_diameter_m"],
            outside_bottom_diameter=bucket["outer_bottom_diameter_m"],
            reported_height=bucket["assumed_interior_depth_m"],
            height_interpretation=bucket["height_interpretation"],
            bottom_thickness=bucket["assumed_bottom_thickness_m"],
        ),
        bucket_bounds={axis: [bounds[0][i], bounds[1][i]] for i, axis in enumerate(("x", "y"))},
        release_center=[*board["initial_center_m"][:2], scene["dropper"]["flange_target_m"][2]],
        pose_display="CPU-forward-kinematics preview and native ball flight from held endpoints",
    )


def _load_verified_inputs(args):
    from . import intergrid_flight_ranking_study as baseline
    from . import intergrid_system_audit as study_audit
    from . import intergrid_system_study as study_runner
    from .benchmark_verify import digest, read, sha
    from .cell1050_rebound_study import parent_inputs
    from .intergrid_library_plan import libraries
    from .intergrid_route_labels import LABEL_VERSION
    from .intergrid_route_refit_inputs import _audited_source
    from .intergrid_system_plan import BASE_PER_ROUTE, FALLBACK_PER_ROUTE, METHODS
    from .policy64_contract import load_runtime_reference

    parent = args.parent_study
    parent_spec, parent_binding, _ = parent_inputs(parent, args.parent_audit)
    source = Path(__file__).resolve().parents[2]
    ref = load_runtime_reference(args.reference, args.assets, source)
    if (digest(ref["scene"]) != parent_spec["identity"]["scene_sha256"]
            or ref["asset_hashes"] != parent_spec["identity"]["asset_sha256"]
            or any(importlib.metadata.version(name) != pinned for name, pinned in parent_spec["packages"].items())):
        raise ValueError("Scene, assets or dependencies differ from the audited simulator")

    study = args.system_study
    report, _, audit, artifacts, contract = _audited_source(
        study, args.system_audit, study_runner.VERSION,
        ("choices.json", "learning.json", "training-prior.json", "route-models.json", "route-history.json"),
    )
    if (audit["version"] != study_audit.VERSION or audit["comparison"] != report["comparison"]
            or audit["choices_sha256"] != artifacts["choices.json"]
            or audit["learning_sha256"] != artifacts["learning.json"]
            or (contract["parent"], contract["scene"], contract["identity"]) != (
                parent_binding, ref["scene"], parent_spec["identity"])
            or (contract["physics_substeps"], contract["label_version"], contract["methods"],
                contract["candidate_budget_per_route"], contract["fallback_budget_per_route"],
                contract["attempts_per_method_per_position"]) != (
                    SUBSTEPS, LABEL_VERSION, list(METHODS), BASE_PER_ROUTE, FALLBACK_PER_ROUTE, 1)):
        raise ValueError("The system planner does not match the completed independent audit")

    checkpoint, original, additions, _, binding, history = study_runner.inputs(args)
    if (any(contract[key] != value for key, value in binding.items())
            or artifacts["route-models.json"] != binding["selector_study"]["checkpoint_sha256"]
            or artifacts["training-prior.json"] != binding["model_study"]["geometry_prior"]
            or read(study / "route-models.json") != checkpoint
            or read(study / "route-history.json") != history):
        raise ValueError("Frozen candidate sources differ from the audited system planner")
    for name in ("intergrid_system_plan.py", "intergrid_route_plan.py", "intergrid_route_selector_plan.py",
                 "intergrid_flight_ranker.py", "cell1050_heading_native.py"):
        relative = "sircus/mjlab/" + name
        if contract["source_sha256"][relative] != sha(source / relative):
            raise ValueError(f"Audited planner source changed: {name}")

    native = dict(
        ref=ref, assets=args.assets, prior=read(study / "training-prior.json"),
        calibration=read(parent / "calibration.json"), home=read(parent / "setup-home.json"),
    )
    expected = dict(
        effective_options=read(parent / "native-yaw/effective-options.json"),
        compiled_geometry=read(parent / "native-yaw/native-compiled-geometry.json"),
    )
    provenance = dict(
        version=VERSION, substeps=SUBSTEPS, engine="mjlab / native MuJoCo Warp",
        planner="physics ranking with a setup-only fallback", simulation_only=True,
        scope="Geometry-checked setup and release from held arm endpoints",
        scene_sha256=digest(ref["scene"]), study_fingerprint=contract["fingerprint"],
        study_audit_sha256=artifacts["audit"], flight_rule_sha256=binding["flight_ranking"]["rule_sha256"],
        route_checkpoint_sha256=artifacts["route-models.json"],
        calibration_sha256=sha(parent / "calibration.json"),
    )
    return native, expected, provenance, checkpoint, libraries(original, additions), history, baseline._ranker(
        args, ref["scene"], binding,
    )


def _configuration(engine, request):
    import numpy as np
    from scipy.spatial.transform import Rotation

    from .cell1050_heading_native import heading_targets

    q = np.r_[request["selected_q10"], request["selected_q14"]]
    gate = engine.adapter.resolver.planner.gate
    gate.bucket_xy = np.asarray(request["bucket_xy_m"])
    frame = gate.frame(q, "held_endpoint")
    rotation = Rotation.from_quat(np.roll(frame["board_quaternion_wxyz"], -1))
    values = request["values"]
    targets = heading_targets(engine.adapter.ref["scene"], request["bucket_xy_m"], values,
                              request["board_heading_offset_deg"], request["board_twist_deg"])
    return dict(
        board_center_m=frame["board_center_m"], board_normal=rotation.as_matrix()[:, 2].tolist(),
        board_quaternion_wxyz=frame["board_quaternion_wxyz"], release_center_m=targets["release_center_m"].tolist(),
        q10=frame["q10"], q14=frame["q14"], link_positions10=frame["link_positions10"],
        link_positions14=frame["link_positions14"], tilt_deg=values[3], gap_m=values[4],
    )


def _trace(directory, row):
    import numpy as np

    from .benchmark_verify import contained, sha

    path = contained(directory, row["native_cache_file"])
    if sha(path) != row["native_cache_sha256"]:
        raise RuntimeError("The native trajectory evidence changed")
    col = row["native_cache_row"]
    with np.load(path, allow_pickle=False) as data:
        if str(data["record_ids"][col]) != row["id"]:
            raise RuntimeError("The native trajectory belongs to another release")
        cadence = SUBSTEPS * 4
        times = data["time_s"][::cadence]
        positions = data["position_m"][::cadence, col]
    if (positions.shape != (len(times), 3) or len(times) < 2
            or not np.all(np.isfinite(positions)) or not np.isfinite(times).all()
            or times[0] != 0.0 or not np.allclose(np.diff(times), 1 / 60, atol=1e-12, rtol=0)
            or positions[0].tolist() != row["native_initial_ball_center_m"]):
        raise RuntimeError("The measured trajectory lost its release binding or sampling cadence")
    return [dict(time=float(t), ball=position.tolist()) for t, position in zip(times, positions, strict=True)]


def _native_drop(native, expected, directory, request_sha256, gpu_lock, gpu_wait_s, notify):
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


def _drop_worker(native, expected, directory, request_sha256, gpu_lock, gpu_wait_s, channel):
    """The one-shot worker owns its CUDA context, so exiting releases the GPU."""
    from .gpu_slot import GPU_UUID

    os.environ["CUDA_VISIBLE_DEVICES"] = GPU_UUID
    with channel:
        try:
            result = _native_drop(native, expected, directory, request_sha256, gpu_lock, gpu_wait_s,
                                  lambda event: channel.send(("progress", event)))
        except TimeoutError as exc:
            channel.send(("busy", str(exc)))
        except Exception as exc:
            channel.send(("error", f"{type(exc).__name__}: {exc}"))
        else:
            channel.send(("result", result))


class PhysicsPlanner:
    """Service-side CPU planning with exactly one isolated native drop per preview."""

    def __init__(self, args, evidence_dir):
        import torch

        from .cell1050_heading_native import HeadingEngine
        from .intergrid_layouts import _placement_geometry
        from .table_assisted_score import VERSION as SCORE_VERSION

        if torch.cuda.is_initialized():
            raise RuntimeError("The interactive service must keep native simulation in its child worker")
        self._native, self._expected, self.provenance, self._checkpoint, self._libraries, self._history, self._ranker = (
            _load_verified_inputs(args)
        )
        self.provenance.update(scoring_version=SCORE_VERSION, required_retention_s=1.0)
        self.scene = self._native["ref"]["scene"]
        geometry = _placement_geometry(self.scene, args.assets)
        self._radius = geometry["bucket_radius_m"]
        self._base_margin = geometry["minimum_base_clearance_m"]
        self.fixed_arm_bounds = geometry["fixed_arm_bounds"]
        low, high = geometry["tabletop_xy_bounds_m"]
        self.planning_bucket_bounds = [[v + self._radius for v in low], [v - self._radius for v in high]]
        self.ui_scene = _ui_scene(self.scene, self.planning_bucket_bounds)
        self._gpu_lock = args.gpu_lock
        self._gpu_wait_s = getattr(args, "gpu_wait_s", 15)
        if type(self._gpu_wait_s) is not int or not 0 <= self._gpu_wait_s <= 120:
            raise ValueError("Interactive GPU wait must be between 0 and 120 seconds")
        self._directory = Path(evidence_dir).resolve()
        self._preparation = self._directory / "preparation"
        self._engine = HeadingEngine(directory=self._preparation, **self._native)

    def _placement_block(self, xy):
        low, high = self.planning_bucket_bounds
        if any(not low[i] <= value <= high[i] for i, value in enumerate(xy)):
            return "Place the whole bucket on the tabletop."
        for arm, bounds in self.fixed_arm_bounds.items():
            if math.dist(xy, bounds["center_xy_m"]) < bounds["radius_m"] + self._radius + self._base_margin:
                return f"The bucket is within the fixed-base clearance zone for {arm}."
        return None

    def plan(self, bucket_xy_m, progress=None):
        from .benchmark_verify import sha
        from .intergrid_system_plan import base_proposals, fallback_proposals, select_requests
        from .parallel_capture_grid_workers import load_resolution
        from .sweep_worker import atomic
        from .table_correction_native import copy_bound

        if (not isinstance(bucket_xy_m, (list, tuple)) or len(bucket_xy_m) != 2
                or any(type(v) not in (int, float) or not math.isfinite(v) for v in bucket_xy_m)):
            raise ValueError("Place the bucket at two finite coordinates in metres")
        xy = [float(v) for v in bucket_xy_m]
        preview = dict(bucket_xy_m=xy, status="blocked", candidate_pool=None, route_intent=None,
                       configuration=None, prediction=None, candidate_counts={}, provenance=dict(self.provenance),
                       reason=self._placement_block(xy))
        if preview["reason"] is not None:
            return preview, None
        identity = f"interactive-{uuid4().hex}"
        layout = dict(id=identity, context_id=identity, bucket_xy_m=xy, split="test", stratum="interactive")
        load_resolution(self._engine, xy)

        def stage(name, proposals):
            label = "Checking the initial shot candidates" if name == "base" else "Checking additional candidates"
            _status(progress, f"checking_{name}", proposed=len(proposals), message=label)
            admitted, records = self._engine.prepare(proposals, one_per_setting=True)
            if self._engine.adapter.env is not None:
                raise RuntimeError("CPU preparation initialized a native environment")
            return dict(proposals=proposals, admitted=admitted, records=records)

        base = stage("base", base_proposals(layout, self._libraries, self._checkpoint))
        proposals = fallback_proposals(layout, self._libraries, base["proposals"], base["admitted"])
        fallback = stage("fallback", proposals) if not base["admitted"] else dict(proposals=[], admitted=[], records=[])
        plan = dict(base=base, fallback=fallback)
        _, selection = select_requests(layout, plan, self._history, self._ranker)
        request = selection["request"]
        preview.update(
            status="feasible" if request else "unresolved", candidate_pool=selection["candidate_pool"],
            route_intent=selection["intended_route"], prediction=selection["prediction"],
            candidate_counts={name: dict(proposed=len(value["proposals"]), admitted=len(value["admitted"]))
                              for name, value in plan.items()},
            reason=None if request else "No tested candidate passed the arm pose and path checks.",
        )
        directory = self._directory / identity
        directory.mkdir(exist_ok=False)
        atomic(directory / "plan.json", dict(layout=layout, stages=plan, selection=selection))
        if request is None:
            return preview, None
        for name in ("context_resolution", "prepared_connection"):
            copy_bound(self._preparation, directory / "native-heading",
                       request[name + "_file"], request[name + "_sha256"])
        preview["configuration"] = _configuration(self._engine, request)
        atomic(directory / "request.json", request)
        request_sha256 = sha(directory / "request.json")
        preview["provenance"]["request_sha256"] = request_sha256
        atomic(directory / "preview.json", preview)
        return preview, _PreparedShot(directory, request_sha256, preview)

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
        proc = context.Process(target=_drop_worker, args=(
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

    def _wait_for_release(self, proc, incoming, progress):
        deadline = time.monotonic() + DROP_TIMEOUT_S + self._gpu_wait_s
        while time.monotonic() < deadline:
            if incoming.poll(0.5):
                try:
                    tag, value = incoming.recv()
                except EOFError:
                    break
                if tag == "progress":
                    if progress is not None:
                        progress(value)
                elif tag == "result":
                    return value
                elif tag == "busy":
                    raise TimeoutError(value)
                else:
                    raise RuntimeError(value)
            elif not proc.is_alive():
                break
        if time.monotonic() >= deadline:
            raise RuntimeError("The interactive simulation exceeded its time limit")
        raise RuntimeError("The isolated native simulation exited before recording a result")
