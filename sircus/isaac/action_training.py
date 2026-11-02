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
            return ReplayShots.approach_endpoints(self, targets)
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
    from cylinder_farm import (
        HIGH,
        LOGSTD,
        LOW,
        SEED,
        WARM,
        observation,
        project_actions,
    )
    from dark_flight import Shot
    from farm_contract import reward, statistics
    from farm_policy import Policy, update, weight_hash
    from parallel_runtime import atomic

    if env.n != 35 or spec["training_batches"] != BATCHES:
        raise ValueError("This bounded training film requires ten 35-cell batches")
    menus, menu_audit = load_geometry_menu(menu_source, spec)
    enabled = np.array([bool(menu) for menu in menus])
    if not enabled[FOCUS]:
        raise ValueError("The chosen foreground has no qualified geometry menu")
    shutil.copyfile(Path(menu_source) / "menu.json", env.dest / "menu.json")
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
            schema="sircus-action-training-clip-v1",
            filming_protocol=PROTOCOL,
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
        ),
        env.dest / "policy-final.pt",
    )
    report = dict(
        schema="sircus-action-training-film-v1",
        filming_protocol=PROTOCOL,
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
        scope="Fresh ten-batch PPO training visualization; no held-out evaluation, learned grasp or learned joint path. Empty-menu cells stay parked/release-disabled; all 350 attempted slots remain in the original PPO optimizer.",
    )
    atomic(env.dest / "result.json", report)
    atomic(env.dest / "progress.json", dict(report, phase="completed"))
    return report
