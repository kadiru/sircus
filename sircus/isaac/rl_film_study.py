"""A separate, finite categorical-PPO study with native training footage.

The original geometry/scoring contract is retained. Learning, assessment and
filming budgets have a separate fingerprint. All releases share the same
canonical endpoint initialization and unchanged measured-settling checks.
"""

import copy
import hashlib
import json
import shutil
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from action_training import ActionTrainingShots, FOCUS, FORCE_NAMES, MOVE_FRAMES, RELEASE_STRIDE
from presentation_recording import canonical_hash, digest, save_physics, seed_verified_ik

SCHEMA = "sircus-native-categorical-ppo-study-v1"
RESULT_SCHEMA = "sircus-native-rl-film-result-v1"
TRIAL_SCHEMA = "sircus-native-rl-film-trial-v1"
CLIP_SCHEMA = "sircus-native-rl-film-clip-v1"
SETUP_PROTOCOL = "scripted-visual-transit-then-common-canonical-endpoint-init-v1"
TRAIN_BATCHES = 128
ASSESSMENT_STAGES = [0, 32, 64, 128]
ASSESSMENT_BATCHES = 8
FILMED_BATCHES = list(range(8)) + list(range(60, 68)) + list(range(120, 128))
SEED = 20261110
SCOPE = (
    "Fresh categorical clipped-PPO over 24 geometry-only native-qualified shot "
    "options at each of 35 known fixed layouts. Privileged geometric inputs; "
    "board bearing/radius/height, release gap and tilt selected jointly. Native "
    "PhysX rewards, scripted visual transit, common canonical episode endpoint "
    "initialization and initialized ball release. Assessment uses the same known "
    "layouts with frozen weights and independent common sampling seeds; it is "
    "not unseen-layout evaluation, learned grasp/path or physical validation."
)


def make_study(bundle):
    from rl_film_policy import ARCHITECTURE, POLICY_VERSION

    menus = bundle["menus"]
    if len(menus) != 35 or any(len(m) != 24 for m in menus):
        raise ValueError("This study requires the original 24-option menus at all 35 sites")
    value = dict(
        schema=SCHEMA, geometry_fingerprint=bundle["spec"]["fingerprint"],
        source=bundle["source"], menu_sha256=bundle["source"]["menu_sha256"],
        camera_sha256=bundle["source"]["camera_sha256"],
        workcells=35, options_per_cell=24, training_batches=TRAIN_BATCHES,
        batches_per_update=2, assessment_stages=ASSESSMENT_STAGES,
        assessment_batches=ASSESSMENT_BATCHES, filmed_batches=FILMED_BATCHES,
        seed=SEED, assessment_seed_base=SEED + 100000,
        policy_version=POLICY_VERSION, architecture=copy.deepcopy(ARCHITECTURE),
        policy_precision=dict(dtype="float32", allow_tf32=False,
                              matmul_precision="highest"),
        ppo=dict(lr=0.003, epochs=4, clip=0.2, value_coef=0.5,
                 entropy_coef=0.01, grad_norm=1.0),
        setup_protocol=SETUP_PROTOCOL,
        assessment_protocol="Frozen stochastic policy; 8 common seeded batches on the same known 35 layouts; no optimizer or search",
        video_selection="Prespecified training batch indices only; no outcome-dependent selection",
        camera_motion="Fixed exact user-selected camera; no ball following",
        scope=SCOPE,
    )
    value["fingerprint"] = canonical_hash(value)
    return value


def validate_options(source, camera, *, appearance, incompatible=False):
    active = source is not None
    if not active:
        if camera is not None:
            raise ValueError("RL camera requires --rl-study-source")
        return False
    if camera is None or appearance != "silver" or incompatible:
        raise ValueError("RL study requires its exact camera, silver appearance and no other recording mode")
    return True


def wilson(captures, attempts):
    if not attempts:
        return [0.0, 1.0]
    p, z = captures / attempts, 1.959963984540054
    den = 1 + z*z/attempts
    center = (p + z*z/(2*attempts))/den
    half = z*np.sqrt(p*(1-p)/attempts + z*z/(4*attempts*attempts))/den
    return [max(0.0, float(center-half)), min(1.0, float(center+half))]


class RLStudyShots(ActionTrainingShots):
    """Native camera frames plus 240 Hz evidence for filmed and unfilmed throws."""

    capture_trial = False
    film_trial = False

    def approach_endpoints(self, targets):
        from parallel_runtime import ParallelShots

        if self.film_trial:
            ActionTrainingShots.approach_endpoints(self, targets)
        # Common episode initialization in EVERY mode. The optional preceding
        # camera-visible motion is not an alternative release setup protocol.
        ParallelShots.approach_endpoints(self, targets)

    def render(self):
        if self.capture_trial and self.phase == "ready":
            self.release_step = self.steps
            self.phase = "release"
            if self.film_trial:
                self.sample("release", RELEASE_STRIDE)

    def tick(self):
        from parallel_runtime import ParallelShots

        ParallelShots.tick(self)
        if not self.capture_trial or self.phase != "release":
            return
        forces = self.forces().copy()
        pos = self.ball.data.root_pos_w.cpu().numpy().copy()
        vel = self.ball.data.root_lin_vel_w.cpu().numpy().copy()
        self.focus_telemetry.append(dict(
            physics_step=self.steps, ball_world_m=pos[FOCUS].tolist(),
            ball_velocity_m_s=vel[FOCUS].tolist(),
            forces_N=dict(zip(FORCE_NAMES, forces[FOCUS].tolist())),
            activity={name: np.flatnonzero(forces[:, j] > 0.01).tolist()
                      for j, name in enumerate(FORCE_NAMES)},
        ))
        self.physics_samples.append(dict(
            time_step=self.steps, position_m=pos, velocity_m_s=vel, forces_N=forces,
            joint_position_rad=np.stack([a.data.joint_pos[:, :6].cpu().numpy()
                                         for a in self.arms.values()], axis=1).copy(),
            joint_velocity_rad_s=np.stack([a.data.joint_vel[:, :6].cpu().numpy()
                                           for a in self.arms.values()], axis=1).copy(),
            board_position_m=self.arms["Arm14"].data.body_pos_w[:, self.board_idx].cpu().numpy().copy(),
            board_quaternion_wxyz=self.arms["Arm14"].data.body_quat_w[:, self.board_idx].cpu().numpy().copy(),
        ))
        if self.film_trial and (self.steps-self.release_step) % RELEASE_STRIDE == 0:
            self.sample("release", RELEASE_STRIDE)

    def sample(self, phase, elapsed):
        ActionTrainingShots.sample(self, phase, elapsed)
        self.frames[-1].update(
            cycle=self.film_cycle, batch=self.train_batch,
            source_fingerprint=self.spec["fingerprint"],
            study_fingerprint=self.study["fingerprint"],
            all_joints_rad={n:a.data.joint_pos[:, :6].cpu().tolist()
                            for n,a in self.arms.items()}, camera_motion="fixed",
        )
        if not (self.dest/"preview.jpg").exists():
            shutil.copyfile(self.frames_path/f"{self.frames[-1]['frame']:05d}.jpg",
                            self.dest/"preview.jpg")


def run_study(env, bundle):
    import torch
    from dark_flight import Shot
    from farm_contract import reward, statistics
    from parallel_runtime import atomic
    from rl_film_policy import (MenuPolicy, POLICY_VERSION, actions_from_indices,
                                actor_weight_hash, build_inputs, checkpoint_payload,
                                sample, update, weight_hash)

    study = make_study(bundle)
    spec, menus = bundle["spec"], bundle["menus"]
    if env.n != 35 or env.spec != spec or not np.array_equal(env.origins, bundle["report"]["origins_m"]):
        raise ValueError("Native RL scene differs from the original qualified source")
    atomic(env.dest/"study.json", study)
    shutil.copyfile(bundle["path"]/"expanded-menu-qualified.json", env.dest/"expanded-menu-qualified.json")
    atomic(env.dest/"menu-provenance.json", bundle["menu_audit"])
    fit_count = seed_verified_ik(env, bundle)
    env.study = study
    enabled = np.ones(35, dtype=bool)
    env.source_valid = torch.tensor(enabled, dtype=torch.bool, device=env.sim.device)
    env.preparation_enabled = enabled.copy()
    env.require_all_transit_ready = True
    model = ET.parse(Path(env.args.assets)/"dropper.urdf")
    env.transit_limits = torch.tensor([float(model.find(f".//joint[@name='joint{i+1}']/limit").get("effort"))
                                      for i in range(6)], device=env.sim.device)
    torch.manual_seed(SEED)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    policy = MenuPolicy().to(env.sim.device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=study["ppo"]["lr"])
    obs_np, candidate_np = build_inputs(spec["cells"], menus)
    obs = torch.as_tensor(obs_np, device=env.sim.device)
    candidates = torch.as_tensor(candidate_np, device=env.sim.device)
    initial_hash, initial_actor_hash = weight_hash(policy), actor_weight_hash(policy)
    changes, batches, assessments, clips, buffer = [], [], [], [], []
    started = time.monotonic()

    def checkpoint(name, after_batches):
        p = env.dest/name
        torch.save(checkpoint_payload(
            policy, optimizer, study_fingerprint=study["fingerprint"],
            geometry_fingerprint=spec["fingerprint"], menu_sha256=study["menu_sha256"],
            updates=len(changes), training_batches=after_batches, seed=SEED,
        ), p)
        return dict(checkpoint_file=p.name, checkpoint_sha256=digest(p),
                    policy_hash=weight_hash(policy), actor_hash=actor_weight_hash(policy))

    def trial(name, mode, batch, cp, *, film=False, repeat=None, generator=None):
        if shutil.disk_usage(env.dest).free < 3*1024**3:
            raise RuntimeError("Less than 3 GiB free during native RL study")
        before = weight_hash(policy)
        with torch.no_grad():
            picked = sample(policy, obs, candidates, generator=generator)
        indices = picked["indices"].cpu().tolist()
        actions = actions_from_indices(spec["cells"], menus, indices)
        env.train_batch, env.film_cycle = batch, len(clips)
        env.shot = Shot(name, batch, FOCUS, 10000, RELEASE_STRIDE, MOVE_FRAMES)
        env.focus, env.frames, env.focus_telemetry, env.physics_samples = FOCUS, [], [], []
        env.film_trial, env.training_active = film, film
        env.capture_trial, env.phase = True, "ready"
        start_frame = env.next_frame
        rows, raw = env.shot_batch(actions, record_video=film, enabled=enabled)
        score_end_step = env.steps
        if env.steps-env.release_step < 960:
            for _ in range(round(0.2/env.dt)):
                env.tick()
        env.capture_trial, env.training_active, env.phase = False, False, None
        if weight_hash(policy) != before or before != cp["policy_hash"]:
            raise RuntimeError("Policy changed while collecting a native trial")
        returns = torch.tensor([reward(r) for r in rows], dtype=torch.float32, device=env.sim.device)
        for r, value in zip(rows, returns.cpu().tolist()):
            r["ppo_reward"] = value
        physics = env.dest/f"{name}-physics.npz"
        save_physics(physics, env.physics_samples)
        payload = dict(
            schema=TRIAL_SCHEMA, mode=mode, batch=batch,
            policy_version=POLICY_VERSION, update_count_before=len(changes),
            source_fingerprint=spec["fingerprint"], study_fingerprint=study["fingerprint"],
            menu_sha256=study["menu_sha256"], selected_camera_sha256=study["camera_sha256"],
            **cp, selected_indices=indices, old_logp=picked["logp"].cpu().tolist(),
            entropy=picked["entropy"].cpu().tolist(),
            observations=obs_np.tolist(), candidate_features=candidate_np.tolist(),
            actions=actions, rewards=returns.cpu().tolist(), rollout=rows,
            release_enabled=enabled.tolist(), trajectory_m=raw["trajectory_m"],
            focus=FOCUS, frames=env.frames, focus_telemetry=env.focus_telemetry,
            physics_file=physics.name, physics_sha256=digest(physics),
            physics_cell_ids=[c["id"] for c in spec["cells"]],
            physics_columns=dict(cell_ids=[c["id"] for c in spec["cells"]],
                arm_names=list(env.arms), force_names=list(FORCE_NAMES),
                coordinate_frame="world", position_units="metres"),
            release_step=env.release_step, score_end_step=score_end_step,
            capture_end_step=env.steps, telemetry_end_step=env.steps,
            physics_dt_s=env.dt, release_stride=RELEASE_STRIDE,
            score_freeze_rule="Per cell: success OR local ball center Z < -0.2 OR 960 steps; post-score ticks do not update reward",
            setup_protocol=SETUP_PROTOCOL,
            setup_visualization="driven" if film else "not-recorded",
            optimizer_samples=35 if mode == "train" else 0,
            scope=SCOPE,
        )
        if mode == "assessment":
            payload.update(after_train_batches=batch, repeat=repeat,
                           assessment_seed=study["assessment_seed_base"]+repeat)
        path = env.dest/f"{name}.json"
        atomic(path, payload)
        summary = dict(batch=batch, file=path.name, file_sha256=digest(path), **cp,
                       update_count_before=len(changes), frames=len(env.frames),
                       optimizer_samples=35 if mode == "train" else 0,
                       start_frame=start_frame, mean_entropy=float(picked["entropy"].mean()),
                       selected_indices=indices, **statistics(rows))
        if film:
            clip = dict(payload, schema=CLIP_SCHEMA, cycle=len(clips))
            clip_path = env.dest/f"clip-{len(clips):02d}.json"
            atomic(clip_path, clip)
            clips.append(dict(summary, cycle=len(clips), file=clip_path.name,
                              file_sha256=digest(clip_path)))
        return summary, (obs, candidates, picked["indices"], picked["logp"], returns)

    def assess(after_batches):
        from farm_contract import statistics

        if buffer:
            raise RuntimeError("Assessment cannot interrupt a partly filled PPO update")
        cp = checkpoint(f"assessment-policy-{after_batches:03d}.pt", after_batches)
        files, rows = [], []
        before = weight_hash(policy)
        for rep in range(ASSESSMENT_BATCHES):
            g = torch.Generator(device=env.sim.device)
            g.manual_seed(study["assessment_seed_base"]+rep)
            summary, _ = trial(f"assessment-{after_batches:03d}-{rep:02d}",
                               "assessment", after_batches, cp, repeat=rep, generator=g)
            files.append(summary["file"])
            rows.extend(json.loads((env.dest/summary["file"]).read_text())["rollout"])
        if weight_hash(policy) != before:
            raise RuntimeError("Assessment changed policy weights")
        values = statistics(rows)
        summary = dict(after_train_batches=after_batches, update_count_before=len(changes),
                       files=files, **cp, **values,
                       capture_rate=values["captures"]/values["attempts"],
                       wilson95=wilson(values["captures"], values["attempts"]),
                       scope=study["assessment_protocol"])
        assessments.append(summary)
        atomic(env.dest/f"assessment-{after_batches:03d}-summary.json", summary)
        print(json.dumps(dict(phase="assessment", **summary)), flush=True)

    assess(0)
    for batch in range(TRAIN_BATCHES):
        cp = checkpoint(f"policy-{batch:03d}.pt", batch)
        summary, tensors = trial(f"train-batch-{batch:03d}", "train", batch, cp,
                                 film=batch in FILMED_BATCHES)
        batches.append(summary)
        buffer.append(tensors)
        if len(buffer) == 2:
            changed = update(policy, optimizer, *[torch.cat(v) for v in zip(*buffer)], study["ppo"])
            changed.update(optimizer_batches=[batch-1, batch], optimizer_samples=70)
            changes.append(changed)
            buffer.clear()
        atomic(env.dest/"progress.json", dict(
            schema=RESULT_SCHEMA, phase="training", study_fingerprint=study["fingerprint"],
            training_batches=batches, updates=changes, assessments=assessments,
            clips=clips, frames=env.next_frame,
        ))
        print(json.dumps(dict(phase="training", **summary)), flush=True)
        if batch+1 in ASSESSMENT_STAGES:
            assess(batch+1)
    if buffer or len(changes) != 64 or weight_hash(policy) == initial_hash or actor_weight_hash(policy) == initial_actor_hash:
        raise RuntimeError("Expected 64 measured PPO updates and a changed actor")
    final_cp = checkpoint("policy-final.pt", TRAIN_BATCHES)
    report = dict(
        schema=RESULT_SCHEMA, status="completed", study=study,
        study_fingerprint=study["fingerprint"], source_fingerprint=spec["fingerprint"],
        source=bundle["source"], menu_sha256=study["menu_sha256"],
        policy_version=POLICY_VERSION, initial_hash=initial_hash,
        initial_actor_hash=initial_actor_hash, final_actor_hash=actor_weight_hash(policy),
        final_checkpoint_file=final_cp["checkpoint_file"],
        final_checkpoint_sha256=final_cp["checkpoint_sha256"],
        final_policy_hash=final_cp["policy_hash"], final_hash=final_cp["policy_hash"],
        training_batches=batches, filmed_batches=FILMED_BATCHES, clips=clips,
        assessments=assessments, updates=changes,
        training_attempts=sum(b["attempts"] for b in batches),
        optimizer_samples=sum(u["optimizer_samples"] for u in changes),
        workcells=35, articulations=70, origins_m=env.origins.tolist(),
        camera=env.saved_camera, camera_binding=env.camera_binding,
        selected_camera_sha256=study["camera_sha256"],
        camera_binding_sha256=digest(env.dest/"camera-binding.json"),
        stage=env.stage_metadata, frames=env.next_frame, fps=30,
        resolution=[1920,1080], physics_dt_s=env.dt,
        setup_protocol=SETUP_PROTOCOL, verified_cached_endpoint_fits=fit_count,
        max_bucket_drift_m=env.peak_bucket_drift,
        elapsed_s=time.monotonic()-started,
        source_sha256={p.name:digest(p) for p in Path(__file__).parent.glob("*.py")},
        scope=SCOPE,
    )
    atomic(env.dest/"result.json", report)
    atomic(env.dest/"progress.json", dict(report, phase="completed"))
    return report
