"""Actual on-policy shot training, recorded from one fixed native camera.

The policy selects high-level endpoint parameters. Joint travel is scripted,
balls are initialized at release, and all original settling/scoring guards stay
in force. A geometry-only menu is reused; no saved outcomes choose actions.
"""

import hashlib
import json
import shutil
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

BATCHES = 10
FOCUS = 1
MOVE_FRAMES = 36
RELEASE_STRIDE = 8
PROTOCOL = "sircus-fixed-camera-on-policy-film-v2"
FORCE_NAMES = ("board", "bucket", "table", "floor", "robot", "unattributed")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_geometry_menu(source, spec):
    """Verify source identity and accepted endpoint proposals, never rewards."""
    from export_cylinder_film import verified_menus

    source = Path(source)
    contract = json.loads((source / "contract.json").read_text())
    fingerprint = contract.pop("fingerprint")
    actual = hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest()
    report = json.loads((source / "result.json").read_text())
    if actual != fingerprint or fingerprint != spec["fingerprint"]:
        raise ValueError("Geometry menu source contract does not match")
    if report["fingerprint"] != fingerprint:
        raise ValueError("Geometry menu result has a different contract")
    if report.get("articulations") != 70 or report.get("workcells") != 35:
        raise ValueError("Geometry menu source must contain 35 two-arm workcells")
    menus, audit = verified_menus(source, spec["cells"])
    if report["menu_sha256"] != audit["file_sha256"]:
        raise ValueError("Geometry menu checksum does not match the source result")
    return menus, dict(
        audit,
        source_run=source.name,
        source_fingerprint=fingerprint,
        source_contract_sha256=digest(source / "contract.json"),
        source_result_sha256=digest(source / "result.json"),
        scope="Geometry-only endpoint qualification; no saved rollout outcomes used.",
    )


def task_dispatch(spec):
    """Whitelist historical and expanded task contracts without guessing shapes."""
    import cylinder_farm

    if spec.get("version") == "sircus-cylinder-film-v2":
        return cylinder_farm, False
    if spec.get("version") == "sircus-expanded-cylinder-film-v1":
        import expanded_farm

        return expanded_farm, True
    raise ValueError("Unsupported training task version")


def training_formats(spec):
    task, expanded = task_dispatch(spec)
    return dict(
        task_version=spec["version"],
        action_dim=len(task.LOW),
        report_schema=(
            "sircus-action-training-film-v2"
            if expanded
            else "sircus-action-training-film-v1"
        ),
        clip_schema=(
            "sircus-action-training-clip-v2"
            if expanded
            else "sircus-action-training-clip-v1"
        ),
        audit_schema=(
            "sircus-action-training-checkpoint-audit-v2"
            if expanded
            else "sircus-action-training-checkpoint-audit-v1"
        ),
        export_schema=(
            "sircus-action-training-export-v2"
            if expanded
            else "sircus-action-training-export-v1"
        ),
        protocol="sircus-fixed-camera-expanded-pose-film-v1" if expanded else PROTOCOL,
    )


def verify_expanded_binding(run, spec, report):
    """Verify original camera bytes bound explicitly to a new action contract."""
    task, expanded = task_dispatch(spec)
    if not expanded:
        return None
    run = Path(run)
    camera, binding = task.bind_camera(
        run / "selected-camera.json", spec, report["origins_m"]
    )
    stored = json.loads((run / "camera-binding.json").read_text())
    formats = training_formats(spec)
    if (
        stored != binding
        or report["camera"] != camera
        or report["camera_source_fingerprint"] != camera["source_fingerprint"]
        or report["camera_binding_sha256"] != digest(run / "camera-binding.json")
        or report["task_version"] != spec["version"]
        or report["action_dim"] != formats["action_dim"]
    ):
        raise ValueError("Expanded camera/task binding mismatch")
    return dict(
        camera=camera, binding=binding, sha256=digest(run / "camera-binding.json")
    )


def load_expanded_training_menu(path, spec):
    """Only native settled endpoint proofs may enable the expanded 35-cell task."""
    from expanded_pose_menu import load_native_menu

    menus, audit = load_native_menu(Path(path), spec)
    if len(menus) != 35 or any(len(menu) < 2 for menu in menus):
        raise ValueError(
            "Expanded training needs two native-qualified actions at all 35 sites"
        )
    return menus, audit


def hold_disabled_targets(targets, starts, enabled):
    """Keep menu-empty stations stationary before any articulated drive command."""
    for name in targets:
        targets[name][~enabled] = starts[name][~enabled]


class ActionTrainingShots:
    """Mixin placed before FilmShots; imports Isaac/Torch only on native calls."""

    training_active = False
    phase = None

    def view_pose(
        self, shot=None, frame=None, cells=None, origins=None, ball_world=None
    ):
        return (
            np.array(self.saved_camera["eye_m"], dtype=float),
            np.array(self.saved_camera["target_m"], dtype=float),
        )

    def prepare(self, actions):
        from feasibility_runtime import FeasibilityShots

        return FeasibilityShots.prepare(self, actions)

    def approach_endpoints(self, targets):
        from parallel_runtime import ParallelShots

        if not self.training_active:
            return ParallelShots.approach_endpoints(self, targets)
        from dark_replay import ReplayShots

        starts = {name: arm.data.joint_pos.clone() for name, arm in self.arms.items()}
        hold_disabled_targets(targets, starts, self.source_valid)
        # The existing progress-sampled drives and final release tolerances are
        # reused. This flag selects foreground + visible background readiness;
        # it never replaces the independent final measured settling checks.
        original_plan = self.args.recording_plan
        self.args.recording_plan = "fixed-station-v1"
        try:
            result = ReplayShots.approach_endpoints(self, targets)
            if getattr(self, "require_all_transit_ready", False):
                errors = np.maximum.reduce(
                    [
                        (arm.data.joint_pos[:, :6] - targets[name][:, :6])
                        .abs()
                        .amax(dim=1)
                        .cpu()
                        .numpy()
                        for name, arm in self.arms.items()
                    ]
                )
                self.transit_joint_error_rad = errors.tolist()
                self.transit_ready_after_drive = (errors < 0.001).tolist()
            return result
        finally:
            self.args.recording_plan = original_plan

    def tick(self):
        from parallel_runtime import ParallelShots

        ParallelShots.tick(self)
        if not self.training_active or self.phase != "release":
            return
        forces = self.forces()
        positions = self.ball.data.root_pos_w.cpu().numpy()
        velocities = self.ball.data.root_lin_vel_w.cpu().numpy()
        self.focus_telemetry.append(
            dict(
                physics_step=self.steps,
                ball_world_m=positions[FOCUS].tolist(),
                ball_velocity_m_s=velocities[FOCUS].tolist(),
                forces_N=dict(zip(FORCE_NAMES, forces[FOCUS].tolist())),
                activity={
                    name: np.flatnonzero(forces[:, j] > 0.01).tolist()
                    for j, name in enumerate(FORCE_NAMES)
                },
            )
        )
        if (self.steps - self.release_step) % RELEASE_STRIDE == 0:
            self.sample("release", RELEASE_STRIDE)

    def render(self):
        if self.training_active and self.record_video and self.phase == "ready":
            self.phase = "release"
            self.release_step = self.steps
            self.sample("release", RELEASE_STRIDE)

    def sample(self, phase, elapsed):
        eye, aim = self.view_pose()
        self.write_rgb(eye, aim, elapsed)
        frame = self.frames[-1]
        frame.update(
            phase=phase,
            focus_ball_world_m=self.ball.data.root_pos_w[FOCUS].cpu().tolist(),
            all_ball_world_m=self.ball.data.root_pos_w.cpu().tolist(),
        )
        if phase == "move":
            frame["focus_joints_rad"] = {
                name: arm.data.joint_pos[FOCUS, :6].cpu().tolist()
                for name, arm in self.arms.items()
            }


def run_training(env, spec, menu_source):
    """Run ten fresh 35-cell batches and five PPO updates, with bounded capture."""
    import torch
    from dark_flight import Shot
    from farm_contract import reward, statistics
    from farm_policy import Policy, update, weight_hash
    from parallel_runtime import atomic

    if env.n != 35 or spec["training_batches"] != BATCHES:
        raise ValueError("This bounded training film requires ten 35-cell batches")
    task, expanded = task_dispatch(spec)
    env.require_all_transit_ready = expanded
    formats = training_formats(spec)
    LOW, HIGH, WARM, LOGSTD = task.LOW, task.HIGH, task.WARM, task.LOGSTD
    observation, project_actions, SEED = (
        task.observation,
        task.project_actions,
        task.SEED,
    )
    extra = {}
    if expanded:
        qualified = getattr(env, "expanded_menu_path", None)
        if qualified is None:
            raise ValueError(
                "Expanded training requires a separately native-qualified menu artifact"
            )
        target = env.dest / "expanded-menu-qualified.json"
        if Path(qualified).resolve() != target.resolve():
            shutil.copyfile(qualified, target)
        menus, menu_audit = load_expanded_training_menu(target, spec)
        camera, binding = task.bind_camera(
            env.dest / "selected-camera.json", spec, env.origins
        )
        if camera != env.saved_camera:
            raise ValueError("Native camera differs from expanded experiment binding")
        atomic(env.dest / "camera-binding.json", binding)
        extra = dict(
            task_version=spec["version"],
            action_dim=formats["action_dim"],
            camera_source_fingerprint=camera["source_fingerprint"],
            camera_binding_sha256=digest(env.dest / "camera-binding.json"),
            transition_policy="all-qualified-sites-ready-v1",
        )
    else:
        menus, menu_audit = load_geometry_menu(menu_source, spec)
        shutil.copyfile(Path(menu_source) / "menu.json", env.dest / "menu.json")
    enabled = np.array([bool(menu) for menu in menus])
    if not enabled[FOCUS]:
        raise ValueError("The chosen foreground has no qualified geometry menu")
    atomic(env.dest / "menu-provenance.json", menu_audit)
    env.source_valid = torch.tensor(enabled, dtype=torch.bool, device=env.sim.device)
    env.preparation_enabled = enabled.copy()
    model = ET.parse(Path(env.args.assets) / "dropper.urdf")
    env.transit_limits = torch.tensor(
        [
            float(model.find(f".//joint[@name='joint{i+1}']/limit").get("effort"))
            for i in range(6)
        ],
        device=env.sim.device,
    )
    torch.manual_seed(SEED)
    policy = Policy(LOW, HIGH, WARM, LOGSTD).to(env.sim.device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=spec["lr"])
    obs = torch.tensor(
        [observation(cell) for cell in spec["cells"]],
        dtype=torch.float32,
        device=env.sim.device,
    )
    initial = weight_hash(policy)
    buffer, changes, batches = [], [], []
    started = time.monotonic()
    camera_hash = digest(env.dest / "selected-camera.json")
    for batch in range(BATCHES):
        if shutil.disk_usage(env.dest).free < 3 * 1024**3:
            raise RuntimeError("Less than 3 GiB free during native training capture")
        before = weight_hash(policy)
        checkpoint = env.dest / f"policy-{batch:02}.pt"
        torch.save(
            dict(
                state_dict=policy.state_dict(),
                optimizer=optimizer.state_dict(),
                source_fingerprint=spec["fingerprint"],
                updates=len(changes),
                menu_sha256=menu_audit["file_sha256"],
                weight_hash=before,
                **extra,
            ),
            checkpoint,
        )
        with torch.no_grad():
            distribution = policy.distribution(obs)
            latent = distribution.sample()
            logp = distribution.log_prob(latent).sum(-1)
        actions = project_actions(spec["cells"], latent.cpu().numpy(), menus)
        env.shot = Shot(
            f"training-{batch+1:02}", batch, FOCUS, 10000, RELEASE_STRIDE, MOVE_FRAMES
        )
        env.focus, env.frames, env.focus_telemetry = FOCUS, [], []
        env.phase, env.training_active = "ready", True
        start = env.next_frame
        rows, raw = env.shot_batch(actions, record_video=True, enabled=enabled)
        score_end_step = env.steps
        # Continue real native physics briefly when every scorer has already
        # frozen early. This can show an immediate visible consequence; it never
        # modifies terminal scores or duplicates/pads static RGB frames.
        if env.steps - env.release_step < 960:
            for _ in range(round(0.2 / env.dt)):
                env.tick()
        env.phase, env.training_active = None, False
        returns = torch.tensor([reward(row) for row in rows], device=env.sim.device)
        for i, row in enumerate(rows):
            row.update(
                geometry_menu_empty=not bool(enabled[i]), ppo_reward=float(returns[i])
            )
        # Preserve the original optimizer methodology for this filming pass:
        # all 35 attempted slots, including invalid rewards, enter each update.
        buffer.append((obs, latent, logp, returns))
        clip = dict(
            schema=formats["clip_schema"],
            filming_protocol=formats["protocol"],
            **extra,
            batch=batch,
            focus=FOCUS,
            shot=env.shot.name,
            source_fingerprint=spec["fingerprint"],
            menu_sha256=menu_audit["file_sha256"],
            policy_hash=before,
            checkpoint_file=checkpoint.name,
            checkpoint_sha256=digest(checkpoint),
            selected_camera_sha256=camera_hash,
            actions=actions,
            latent=latent.cpu().tolist(),
            old_logp=logp.cpu().tolist(),
            rewards=returns.cpu().tolist(),
            release_enabled=enabled.tolist(),
            observation=obs.cpu().tolist(),
            rollout=rows,
            trajectory_m=raw["trajectory_m"],
            frames=env.frames,
            focus_telemetry=env.focus_telemetry,
            release_step=env.release_step,
            score_end_step=score_end_step,
            capture_end_step=env.steps,
            telemetry_end_step=env.steps,
            score_freeze_rule="Per cell: success OR local ball center Z < -0.2 OR 960 steps. Post-score native ticks do not update rewards.",
            physics_dt_s=env.dt,
            release_stride=RELEASE_STRIDE,
            scope="Actual on-policy data consumed by PPO; scripted joint transit and initialized ball release.",
        )
        if expanded:
            clip["transit_joint_error_rad"] = env.transit_joint_error_rad
            clip["transit_ready_after_drive"] = env.transit_ready_after_drive
        atomic(env.dest / f"clip-{batch:02}.json", clip)
        summary = dict(
            batch=batch,
            file=f"clip-{batch:02}.json",
            policy_hash=before,
            start_frame=start,
            frames=len(env.frames),
            release_step=env.release_step,
            score_end_step=score_end_step,
            capture_end_step=env.steps,
            supported_cells=int(enabled.sum()),
            optimizer_samples=env.n,
            **statistics(rows),
        )
        if expanded:
            summary["transit_ready_cells"] = sum(env.transit_ready_after_drive)
        batches.append(summary)
        if len(buffer) == spec["batches_per_update"]:
            changes.append(
                update(policy, optimizer, *[torch.cat(v) for v in zip(*buffer)], spec)
            )
            buffer.clear()
        atomic(
            env.dest / "progress.json",
            dict(
                schema="sircus-action-training-progress-v1",
                phase="training",
                batches=batches,
                updates=changes,
                frames=env.next_frame,
            ),
        )
        print(json.dumps(summary), flush=True)
    final = weight_hash(policy)
    if len(changes) != 5 or buffer or final == initial:
        raise RuntimeError("Ten batches did not produce five measured PPO updates")
    torch.save(
        dict(
            state_dict=policy.state_dict(),
            optimizer=optimizer.state_dict(),
            contract=spec,
            menu_sha256=menu_audit["file_sha256"],
            weight_hash=final,
            **extra,
        ),
        env.dest / "policy-final.pt",
    )
    report = dict(
        schema=formats["report_schema"],
        filming_protocol=formats["protocol"],
        **extra,
        source_fingerprint=spec["fingerprint"],
        fingerprint=spec["fingerprint"],
        menu_sha256=menu_audit["file_sha256"],
        menu_provenance=menu_audit,
        initial_hash=initial,
        final_hash=final,
        final_checkpoint_sha256=digest(env.dest / "policy-final.pt"),
        updates=changes,
        batches=batches,
        workcells=env.n,
        articulations=2 * env.n,
        origins_m=env.origins.tolist(),
        camera=env.saved_camera,
        selected_camera_sha256=camera_hash,
        stage=env.stage_metadata,
        frames=env.next_frame,
        fps=30,
        resolution=[1920, 1080],
        physics_dt_s=env.dt,
        attempts=sum(b["attempts"] for b in batches),
        optimizer_samples=sum(b["optimizer_samples"] for b in batches),
        captures=sum(b["captures"] for b in batches),
        elapsed_s=time.monotonic() - started,
        source_sha256={p.name: digest(p) for p in Path(__file__).parent.glob("*.py")},
        scope=(
            "Fresh five-dimensional ten-batch PPO training visualization, with two or more native-qualified actions at all 35 fixed bucket sites. Original per-attempt release guards remain active; actual releases and invalids are reported. Scripted transit and initialized release; no held-out evaluation or learned grasp/joint path."
            if expanded
            else "Fresh ten-batch PPO training visualization; no held-out evaluation, learned grasp or learned joint path. Empty-menu cells stay parked/release-disabled; all 350 attempted slots remain in the original PPO optimizer."
        ),
    )
    atomic(env.dest / "result.json", report)
    atomic(env.dest / "progress.json", dict(report, phase="completed"))
    return report
