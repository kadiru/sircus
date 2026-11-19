"""Fresh native farm filming from qualified geometry, without policy updates.

The original ten-batch training contract and every release/scoring gate remain
unchanged. A separate signed filming plan schedules geometry-only endpoint
choices. RGB and telemetry come from newly stepped PhysX, never saved ball paths.
"""

import copy
import hashlib
import json
import shutil
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from action_training import (
    ActionTrainingShots,
    FOCUS,
    FORCE_NAMES,
    MOVE_FRAMES,
    RELEASE_STRIDE,
    digest,
    load_expanded_training_menu,
    verify_expanded_binding,
)

SCHEMA = "sircus-native-presentation-recording-v1"
PROTOCOL = "sircus-fixed-camera-qualified-geometry-presentation-v1"
DEFAULT_BATCHES = 30
MAX_BATCHES = 60
SCOPE = (
    "Fresh native Isaac/PhysX simulation of geometry-only qualified endpoint "
    "trials. Scripted articulation drives and initialized ball releases; no new "
    "optimizer updates, learned grasp, learned joint paths or held-out evaluation. "
    "All attempts and misses are retained in the source recording."
)


def canonical_hash(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, allow_nan=False).encode()
    ).hexdigest()


def validate_options(source, camera, batches, *, appearance, incompatible=False):
    """A separate opt-in route; never relax historical CLI combinations."""
    active = source is not None
    if not active:
        if camera is not None or batches != DEFAULT_BATCHES:
            raise ValueError("Presentation options require --presentation-source")
        return False
    if (
        camera is None
        or type(batches) is not int
        or not 1 <= batches <= MAX_BATCHES
        or appearance != "silver"
        or incompatible
    ):
        raise ValueError(
            "Presentation requires its exact camera JSON, silver appearance, "
            "1..60 batches and no training/replay/preview options"
        )
    return True


def load_source(source, camera_path, *, source_root=None):
    """Verify the complete native source and immutable physics dependencies.

    The only historical file allowed to change is the additive CLI entrypoint.
    Its old and new hashes are both reported. Every old physics, scoring,
    controls, camera and asset helper must still match the completed source.
    """
    from expanded_farm import VERSION, make_spec

    source, camera_path = Path(source), Path(camera_path)
    root = Path(source_root) if source_root else Path(__file__).parent
    spec = json.loads((source / "contract.json").read_text())
    report = json.loads((source / "result.json").read_text())
    unsigned = dict(spec)
    fingerprint = unsigned.pop("fingerprint", None)
    if (
        spec.get("version") != VERSION
        or canonical_hash(unsigned) != fingerprint
        or spec.get("training_batches") != 10
        or spec != make_spec(spec["scene"], 10)
        or report.get("schema") != "sircus-action-training-film-v2"
        or report.get("source_fingerprint") != fingerprint
        or report.get("fingerprint") != fingerprint
        or report.get("workcells") != 35
        or report.get("articulations") != 70
        or len(report.get("batches", [])) != 10
        or len(report.get("updates", [])) != 5
    ):
        raise ValueError("Expected a completed, unchanged native expanded ten-batch source")
    if any(b.get("supported_cells") != 35 for b in report["batches"]):
        raise ValueError("Source must support all 35 original bucket positions")
    binding = verify_expanded_binding(source, spec, report)
    if camera_path.read_bytes() != (source / "selected-camera.json").read_bytes():
        raise ValueError("Presentation camera must preserve the exact selected source bytes")
    menu_path = source / "expanded-menu-qualified.json"
    menus, audit = load_expanded_training_menu(menu_path, spec)
    if report.get("menu_sha256") != digest(menu_path):
        raise ValueError("Source result does not bind the native qualified menu")
    recorded = report.get("source_sha256", {})
    required = {
        "parallel_runtime.py", "feasibility_runtime.py", "rl_contract.py",
        "entry_quality.py", "board_route.py", "settling_contract.py",
        "expanded_farm.py", "expanded_pose_menu.py", "expanded_pose_native.py",
        "action_training.py", "dark_replay.py", "parallel_assets.py",
        "silver_style.py", "dark_stage.py", "camera_options.py", "saved_camera.py",
    }
    if not required.issubset(recorded):
        raise ValueError("Source is missing physics/control dependency hashes")
    actual = {}
    for name, expected in recorded.items():
        if Path(name).name != name or not name.endswith(".py"):
            raise ValueError("Unsafe recorded dependency path")
        actual[name] = digest(root / name)
        if name != "film_cylinder_farm.py" and actual[name] != expected:
            raise ValueError(f"Historical native dependency changed: {name}")
    for filename in ("probes.json", "rim-rays.json"):
        evidence = json.loads((source / filename).read_text())
        passed = (
            all(row.get("all_passed") for row in evidence)
            if filename == "probes.json"
            else len(evidence.get("passed", [])) == 35 and all(evidence["passed"])
        )
        if not passed:
            raise ValueError(f"Completed native source lacks passing {filename}")
    return dict(
        path=source,
        spec=spec,
        report=report,
        menus=menus,
        menu_audit=audit,
        camera=binding["camera"],
        camera_binding=binding["binding"],
        source=dict(
            run=source.name,
            contract_sha256=digest(source / "contract.json"),
            result_sha256=digest(source / "result.json"),
            menu_sha256=digest(menu_path),
            camera_sha256=digest(camera_path),
            camera_binding_sha256=digest(source / "camera-binding.json"),
            probes_sha256=digest(source / "probes.json"),
            rim_rays_sha256=digest(source / "rim-rays.json"),
            original_source_sha256=recorded,
            current_original_dependencies_sha256=actual,
            intentionally_updated_entrypoints=["film_cylinder_farm.py"],
        ),
    )


def geometry_order(options):
    """Deterministic farthest-first coverage of five normalized action axes.

    IDs, rewards and saved rollouts cannot affect this order. After exhausting
    the menu, repeat the same physical coverage without selecting by success.
    """
    from expanded_farm import ACTION_NAMES, HIGH, LOW

    values = np.asarray([[a[k] for k in ACTION_NAMES] for a in options], dtype=float)
    if len(values) < 2 or not np.isfinite(values).all():
        raise ValueError("Every table needs two finite qualified geometries")
    points = (values - LOW) / (HIGH - LOW)
    if len({tuple(p) for p in points}) != len(points):
        raise ValueError("Repeated physical menu geometry")
    # Start nearest the geometry centroid, then maximize distance to the entire
    # selected set, with the numeric geometry tuple as a stable tie breaker.
    centroid = points.mean(axis=0)
    remaining = set(range(len(points)))
    first = min(remaining, key=lambda i: (np.linalg.norm(points[i] - centroid), tuple(points[i])))
    ordered = [first]
    remaining.remove(first)
    while remaining:
        nxt = min(
            remaining,
            key=lambda i: (
                -float(np.min(np.linalg.norm(points[ordered] - points[i], axis=1))),
                tuple(points[i]),
            ),
        )
        ordered.append(nxt)
        remaining.remove(nxt)
    return ordered


def make_plan(bundle, batches=DEFAULT_BATCHES):
    if type(batches) is not int or not 1 <= batches <= MAX_BATCHES:
        raise ValueError("Presentation batch count must be 1..60")
    spec, menus = bundle["spec"], bundle["menus"]
    if len(menus) != 35 or len(spec["cells"]) != 35:
        raise ValueError("Presentation requires all 35 source cells")
    buckets = [tuple(c["bucket_xy_m"]) for c in spec["cells"]]
    if len(set(buckets)) != 35:
        raise ValueError("Each workcell must retain a distinct local bucket position")
    orders = [geometry_order(menu) for menu in menus]
    cycles = []
    for cycle in range(batches):
        indices = [order[cycle % len(order)] for order in orders]
        cycles.append(dict(
            cycle=cycle,
            menu_indices=indices,
            action_ids=[menus[i][j]["id"] for i, j in enumerate(indices)],
            actions=[copy.deepcopy(menus[i][j]) for i, j in enumerate(indices)],
        ))
    return dict(
        schema="sircus-native-presentation-plan-v1",
        filming_protocol=PROTOCOL,
        source_fingerprint=spec["fingerprint"],
        source=bundle["source"],
        camera=bundle["camera"],
        camera_binding=bundle["camera_binding"],
        original_training_batches=spec["training_batches"],
        presentation_batches=batches,
        workcells=35,
        focus=FOCUS,
        cells=[dict(id=c["id"], bucket_xy_m=c["bucket_xy_m"]) for c in spec["cells"]],
        selection="Five-axis normalized farthest-first geometry coverage; no saved outcomes or policy weights used.",
        menu_order=orders,
        cycles=cycles,
        move_frames=MOVE_FRAMES,
        release_stride=RELEASE_STRIDE,
        physics_dt_s=spec["scene"]["simulation"]["physics_dt_s"],
        camera_motion="None; exact selected camera remains fixed for every frame.",
        optimizer_updates=0,
        scope=SCOPE,
    )


def seed_verified_ik(env, bundle):
    """Reuse original endpoint branches only after independent FK verification."""
    from expanded_pose_menu import endpoint_targets
    from expanded_pose_native import verified_fit

    manifest = json.loads((bundle["path"] / "expanded-menu-qualified.json").read_text())
    fits = manifest.get("fits", [])
    if len(fits) != env.n or env.n != 35:
        raise ValueError("Native qualified menu is missing endpoint fits")
    count = 0
    for cell, options, solutions in zip(bundle["spec"]["cells"], bundle["menus"], fits):
        if len(options) != len(solutions):
            raise ValueError("Native menu solutions do not match its geometry")
        for action, by_arm in zip(options, solutions):
            targets = endpoint_targets(bundle["spec"]["scene"], cell, action)
            for name in ("Arm10", "Arm14"):
                target = targets[name]
                env.ik_cache[(name, tuple(target.ravel()))] = verified_fit(
                    target, by_arm[name], env.links, name == "Arm14"
                )
                count += 1
    return count


class PresentationShots(ActionTrainingShots):
    """Existing native driven setup plus full-rate, all-cell evidence capture."""

    def tick(self):
        super().tick()
        if not self.training_active or self.phase != "release":
            return
        self.physics_samples.append(dict(
            time_step=self.steps,
            position_m=self.ball.data.root_pos_w.cpu().numpy().copy(),
            velocity_m_s=self.ball.data.root_lin_vel_w.cpu().numpy().copy(),
            forces_N=self.forces().copy(),
            joint_position_rad=np.stack(
                [a.data.joint_pos[:, :6].cpu().numpy() for a in self.arms.values()], axis=1
            ).copy(),
            joint_velocity_rad_s=np.stack(
                [a.data.joint_vel[:, :6].cpu().numpy() for a in self.arms.values()], axis=1
            ).copy(),
            board_position_m=self.arms["Arm14"].data.body_pos_w[:, self.board_idx].cpu().numpy().copy(),
            board_quaternion_wxyz=self.arms["Arm14"].data.body_quat_w[:, self.board_idx].cpu().numpy().copy(),
        ))

    def sample(self, phase, elapsed):
        super().sample(phase, elapsed)
        self.frames[-1].update(
            cycle=self.presentation_cycle,
            source_fingerprint=self.presentation_fingerprint,
            plan_sha256=self.presentation_plan_sha256,
            all_joints_rad={name: a.data.joint_pos[:, :6].cpu().tolist() for name, a in self.arms.items()},
            camera_motion="fixed",
        )
        if not (self.dest / "preview.jpg").exists():
            shutil.copyfile(
                self.frames_path / f"{self.frames[-1]['frame']:05d}.jpg",
                self.dest / "preview.jpg",
            )


def save_physics(path, samples):
    if not samples:
        raise RuntimeError("Native release produced no physics telemetry")
    steps = np.asarray([s["time_step"] for s in samples], dtype=np.int64)
    if not np.all(np.diff(steps) == 1):
        raise RuntimeError("Missing full-rate native physics steps")
    data = {key: np.asarray([s[key] for s in samples]) for key in samples[0]}
    if any(not np.isfinite(value).all() for value in data.values()):
        raise RuntimeError("Nonfinite native physics telemetry")
    np.savez_compressed(path, **data)


def run_presentation(env, bundle, batches=DEFAULT_BATCHES):
    """Record new driven attempts. Import no actor, optimizer or training loop."""
    import torch
    from dark_flight import Shot
    from farm_contract import statistics
    from parallel_runtime import atomic

    spec = bundle["spec"]
    if env.n != 35 or env.spec != spec or not np.array_equal(env.origins, bundle["report"]["origins_m"]):
        raise ValueError("Native presentation scene differs from the verified source")
    plan = make_plan(bundle, batches)
    plan_sha = canonical_hash(plan)
    atomic(env.dest / "plan.json", plan)
    shutil.copyfile(bundle["path"] / "expanded-menu-qualified.json", env.dest / "expanded-menu-qualified.json")
    atomic(env.dest / "menu-provenance.json", bundle["menu_audit"])
    fit_count = seed_verified_ik(env, bundle)
    enabled = np.ones(env.n, dtype=bool)
    env.source_valid = torch.tensor(enabled, dtype=torch.bool, device=env.sim.device)
    env.preparation_enabled = enabled.copy()
    env.require_all_transit_ready = True
    model = ET.parse(Path(env.args.assets) / "dropper.urdf")
    env.transit_limits = torch.tensor([
        float(model.find(f".//joint[@name='joint{i + 1}']/limit").get("effort"))
        for i in range(6)
    ], device=env.sim.device)
    env.presentation_plan_sha256 = plan_sha
    env.presentation_fingerprint = spec["fingerprint"]
    summaries, all_rows = [], []
    started = time.monotonic()
    for cycle_plan in plan["cycles"]:
        cycle = cycle_plan["cycle"]
        if shutil.disk_usage(env.dest).free < 3 * 1024**3:
            raise RuntimeError("Less than 3 GiB free during native presentation capture")
        env.presentation_cycle = cycle
        env.shot = Shot(f"presentation-{cycle + 1:02}", cycle, FOCUS, 10000, RELEASE_STRIDE, MOVE_FRAMES)
        env.focus, env.frames, env.focus_telemetry = FOCUS, [], []
        env.physics_samples = []
        # This existing internal flag enables live scripted transit/frame capture;
        # no policy/optimizer is instantiated by this presentation route.
        env.phase, env.training_active = "ready", True
        start = env.next_frame
        rows, raw = env.shot_batch(cycle_plan["actions"], record_video=True, enabled=enabled)
        all_rows.extend(rows)
        score_end_step = env.steps
        if env.steps - env.release_step < 960:
            for _ in range(round(0.2 / env.dt)):
                env.tick()
        env.phase, env.training_active = None, False
        physics_path = env.dest / f"physics-{cycle:02}.npz"
        save_physics(physics_path, env.physics_samples)
        clip = dict(
            schema="sircus-native-presentation-clip-v1", filming_protocol=PROTOCOL,
            cycle=cycle, batch=cycle, focus=FOCUS, shot=env.shot.name,
            source_fingerprint=spec["fingerprint"], plan_sha256=plan_sha,
            source=bundle["source"], source_cycle=cycle_plan,
            selected_camera_sha256=bundle["source"]["camera_sha256"],
            menu_sha256=bundle["source"]["menu_sha256"],
            actions=cycle_plan["actions"], release_enabled=enabled.tolist(),
            rollout=rows, trajectory_m=raw["trajectory_m"], frames=env.frames,
            focus_telemetry=env.focus_telemetry,
            physics_file=physics_path.name, physics_sha256=digest(physics_path),
            physics_cell_ids=[c["id"] for c in spec["cells"]],
            physics_columns=dict(
                cell_ids=[c["id"] for c in spec["cells"]],
                arm_names=list(env.arms), force_names=list(FORCE_NAMES),
                coordinate_frame="world", position_units="metres",
                time_step="Native physics-step index; sample at release_step + 1 through telemetry_end_step inclusive.",
            ),
            release_step=env.release_step, score_end_step=score_end_step,
            capture_end_step=env.steps, telemetry_end_step=env.steps,
            score_freeze_rule="Per cell: success OR local ball center Z < -0.2 OR 960 steps. Post-score native ticks never update scores.",
            physics_dt_s=env.dt, release_stride=RELEASE_STRIDE,
            transit_joint_error_rad=env.transit_joint_error_rad,
            transit_ready_after_drive=env.transit_ready_after_drive,
            optimizer_updates=0, scope=SCOPE,
        )
        atomic(env.dest / f"clip-{cycle:02}.json", clip)
        summary = dict(
            cycle=cycle, batch=cycle, file=f"clip-{cycle:02}.json",
            start_frame=start, frames=len(env.frames),
            release_step=env.release_step, score_end_step=score_end_step,
            capture_end_step=env.steps, supported_cells=env.n,
            transit_ready_cells=sum(env.transit_ready_after_drive),
            physics_file=physics_path.name, physics_sha256=digest(physics_path),
            **statistics(rows),
        )
        summaries.append(summary)
        atomic(env.dest / "progress.json", dict(
            schema=SCHEMA, phase="recording", batches=summaries,
            frames=env.next_frame, plan_sha256=plan_sha, optimizer_updates=0,
        ))
        print(json.dumps(summary), flush=True)
    report = dict(
        schema=SCHEMA, status="completed", filming_protocol=PROTOCOL,
        source_fingerprint=spec["fingerprint"], fingerprint=spec["fingerprint"],
        source=bundle["source"], filming_plan=plan, plan_sha256=plan_sha,
        task_version=spec["version"], action_dim=spec["action_dim"],
        workcells=env.n, articulations=2 * env.n, origins_m=env.origins.tolist(),
        camera=env.saved_camera, camera_binding=env.camera_binding,
        selected_camera_sha256=bundle["source"]["camera_sha256"],
        camera_binding_sha256=digest(env.dest / "camera-binding.json"),
        stage=env.stage_metadata, frames=env.next_frame, fps=30,
        resolution=[1920, 1080], physics_dt_s=env.dt,
        batches=summaries, attempts=sum(b["attempts"] for b in summaries),
        captures=sum(b["captures"] for b in summaries),
        invalid=sum(b["invalid"] for b in summaries),
        counts=statistics(all_rows),
        verified_cached_endpoint_fits=fit_count, optimizer_updates=0,
        max_bucket_drift_m=env.peak_bucket_drift,
        elapsed_s=time.monotonic() - started,
        source_sha256={p.name: digest(p) for p in Path(__file__).parent.glob("*.py")},
        scope=SCOPE,
    )
    atomic(env.dest / "result.json", report)
    atomic(env.dest / "progress.json", dict(report, phase="completed"))
    return report
