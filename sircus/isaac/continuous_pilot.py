"""Finite continuous-shot PPO pilot with native driven setup and fixed camera.

The actor learns one cooperative shot decision, not the arm trajectories. Every
sampled attempt is retained. Endpoint qualification never replaces the action
with a catalogue entry. Each episode resets to the same home before filming; all subsequent shot
setup uses live articulation drives without endpoint joint-state resets.
"""

import copy
import hashlib
import json
import math
import shutil
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from continuous_feasibility import (LOW, HIGH, ACTION_NAMES, decode_action,
                                    qualify_action, split_for_index, canonical_hash)
from presentation_recording import digest, save_physics

SCHEMA = "sircus-native-continuous-pilot-v1"
SEED = 20261112
FOCUS = 1
FORCE_NAMES = ("board", "bucket", "table", "floor", "robot", "unattributed")


def make_study(bundle, batches=64):
    from continuous_policy import ARCHITECTURE, POLICY_VERSION

    if batches not in (4, 64):
        raise ValueError("Only four-batch smoke or 64-batch pilot is supported")
    cells = [dict(c, split=split_for_index(i)) for i, c in enumerate(bundle["spec"]["cells"])]
    domain = dict(names=list(ACTION_NAMES), low=LOW.tolist(), high=HIGH.tolist(),
                  reference_xy_m=[.67, -.10],
                  gap_definition="Ball-bottom release Z minus board-center Z; ball center adds radius")
    domain["fingerprint"] = canonical_hash(domain)
    study = dict(schema=SCHEMA, geometry_fingerprint=bundle["spec"]["fingerprint"],
                 source=bundle["source"], cells=cells, domain=domain, seed=SEED,
                 training_batches=batches, batches_per_update=4,
                 assessment_stages=[0, batches], assessment_batches=2 if batches == 4 else 8,
                 ppo=dict(lr=.0003, epochs=4, clip=.2, value_coef=.5,
                          entropy_coef=.005, grad_norm=1.),
                 policy_version=POLICY_VERSION, architecture=ARCHITECTURE,
                 observations="Five privileged geometric features from original fixed bucket layouts",
                 training_split="25 fixed bucket sites; five validation and five test sites excluded from optimization",
                 evaluation="Frozen deterministic direct inference, repeated native trials; identical repeats measure numerical repeatability, not independent layout generalization",
                 setup="Canonical episode home reset before visible setup; live scripted articulation drive to chosen shot; no endpoint reset; coarse swept proxy and final measured settlement gates",
                 measurement_hold_s=1.75,
                 measurement_window_s=.25,
                 ik_protocol="full_robust_v2",
                 ik_scope="Unchanged robust_ik.py v2 with strict and deterministic multistarts; action unchanged, independent FK verification, failures retained",
                 episode_start="Canonical episode reset to original saved home joints before filming, six seconds original hold, then live shot drive; same reset for coverage/train/eval; no endpoint reset",
                 home_hold_s=6.0,
                 motion_scope="Self collision disabled in source asset; coarse path proxy is not collision certification; initialized ball release, no grasp/path learning",
                 filmed_training_batches=list(range(batches)) if batches == 4 else list(range(4))+list(range(16,20))+list(range(32,36))+list(range(60,64)),
                 video="Prescheduled early/middle/late training batches, normal-time 30fps setup/release; episode reset happens before visible setup; exact fixed saved camera, no ball following",
                 scope="Single-seed continuous-action terminal contextual PPO pilot; sampled feasibility, not exhaustive reachability or physical validation")
    study["fingerprint"] = canonical_hash(study)
    return study


def validate_options(source, camera, batches, *, appearance, incompatible=False):
    if source is None:
        if camera is not None or batches != 64:
            raise ValueError("Continuous options require --continuous-source")
        return False
    if camera is None or appearance != "silver" or incompatible or batches not in (4, 64):
        raise ValueError("Continuous pilot requires exact camera, silver, 4 or 64 batches and no other mode")
    return True


class ContinuousShots:
    capture_trial = False
    film_trial = False
    phase = None

    def view_pose(self, *args, **kwargs):
        return np.asarray(self.saved_camera["eye_m"]), np.asarray(self.saved_camera["target_m"])

    def prepare(self, actions):
        from feasibility_runtime import FeasibilityShots
        from expanded_pose_menu import endpoint_targets
        from continuous_feasibility import approximate_path_check
        from robust_ik import solve

        self.board_pose_reads = 0
        eligible, qualifications = [], []
        self.path_reasons = [None] * self.n
        self.path_checks = [None] * self.n
        for i, action in enumerate(actions):
            if not self.active_mask[i]:
                qualifications.append(dict(accepted=False, reason="inactive_split"))
                eligible.append(False)
                continue
            if not self.home_readback["path_eligible"][i]:
                qualifications.append(dict(accepted=False, reason="common_home_path_proxy_rejected"))
                eligible.append(False)
                continue
            seeds = {name: arm.data.joint_pos[i, :6].cpu().numpy().copy()
                     for name, arm in self.arms.items()}
            if self.trial_mode == "coverage" and self.coverage_fits[i] is not None:
                seeds = {n: self.coverage_fits[i][n]["q"] for n in self.arms}
            fit = qualify_action(self.cfg, self.base_cells[i], action, self.continuous_links, seeds, solver=solve)
            qualifications.append(fit)
            if fit["accepted"]:
                targets = endpoint_targets(self.cfg, self.base_cells[i], action)
                for name in self.arms:
                    self.ik_cache[(name, tuple(targets[name].ravel()))] = fit["ik"][name]
                actual_start = {n: arm.data.joint_pos[i, :6].cpu().numpy().copy() for n,arm in self.arms.items()}
                check = approximate_path_check(self.cfg, self.continuous_links, actual_start,
                    {n: fit["ik"][n]["q"] for n in self.arms}, cell=self.base_cells[i], samples=16)
                self.path_checks[i] = check
                if not check["accepted"]:
                    self.path_reasons[i] = check["reasons"][0]
            eligible.append(fit["accepted"] and self.path_reasons[i] is None)
        self.qualifications = qualifications
        self.preparation_enabled = np.asarray(eligible)
        releases, normals, records, invalid = FeasibilityShots.prepare(self, actions)
        self.scoring_normals = normals.copy()
        for i, fit in enumerate(qualifications):
            if not fit["accepted"]:
                invalid[i] = fit["reason"]
            elif self.path_reasons[i] is not None:
                invalid[i] = self.path_reasons[i]
        return releases, normals, records, invalid

    def home(self):
        from parallel_runtime import ParallelShots

        # Standard episodic reset happens before any visible setup, never at a
        # selected shot endpoint. All later arm motion is driven and measured.
        ParallelShots._initialize_states(self)
        for _ in range(round(6/self.dt)):
            self.tick()
        return dict(protocol="canonical-episode-home-reset-before-visible-drive-v1",
            episode_reset=True, target_joint_rad={n:a.data.default_joint_pos[:,:6].cpu().tolist() for n,a in self.arms.items()},
            measured_joint_rad={n:a.data.joint_pos[:,:6].cpu().tolist() for n,a in self.arms.items()},
            measured_joint_velocity_rad_s={n:a.data.joint_vel[:,:6].cpu().tolist() for n,a in self.arms.items()},
            path_eligible=[True]*self.n,
            path_checks=[dict(accepted=True, scope="Canonical simulation episode reset; no physical home path claimed") for _ in range(self.n)],
            physics_step=self.steps)

    def approach_endpoints(self, targets):
        starts = {n: a.data.joint_pos.clone() for n, a in self.arms.items()}
        self.set_balls(np.tile([.1, -.8, .04], (self.n, 1)))
        for i in range(self.n):
            if not self.preparation_enabled[i]:
                for name in targets:
                    targets[name][i] = starts[name][i]
                continue
        deltas = {n: targets[n]-starts[n] for n in targets}
        longest = max(float(d.abs().max()) for d in deltas.values())
        steps = max(120, int(math.ceil(1.5*longest/(.85*self.dt))))
        original = {}
        for name, arm in self.arms.items():
            ids = [arm.joint_names.index(f"joint{i+1}") for i in range(6)]
            original[name] = (ids, arm.data.joint_effort_limits[:, ids].clone(),
                              arm.data.joint_stiffness[:, ids].clone(), arm.data.joint_damping[:, ids].clone())
            arm.write_joint_effort_limit_to_sim(self.transit_limits.expand(self.n, 6), ids)
            arm.write_joint_stiffness_to_sim(6000., ids)
            arm.write_joint_damping_to_sim(180., ids)
        try:
            for step in range(1, steps+round(2/self.dt)+1):
                if step <= steps:
                    t = step/steps
                    t = t*t*(3-2*t)
                    for name, arm in self.arms.items():
                        arm.set_joint_position_target(starts[name]+t*deltas[name])
                self.tick()
                if step >= steps and step % 60 == 0:
                    errors = np.maximum.reduce([(arm.data.joint_pos[:, :6]-targets[n][:, :6]).abs().amax(dim=1).cpu().numpy()
                                                for n, arm in self.arms.items()])
                    if np.all(errors[self.preparation_enabled] < .001):
                        break
        finally:
            for name, arm in self.arms.items():
                ids, effort, stiffness, damping = original[name]
                arm.write_joint_effort_limit_to_sim(effort, ids)
                arm.write_joint_stiffness_to_sim(stiffness, ids)
                arm.write_joint_damping_to_sim(damping, ids)
        self.drive_ticks = step
        self.transit_joint_error_rad = np.maximum.reduce([
            (arm.data.joint_pos[:, :6]-targets[n][:, :6]).abs().amax(dim=1).cpu().numpy()
            for n, arm in self.arms.items()]).tolist()
        self.transit_ready_after_drive = (np.asarray(self.transit_joint_error_rad) < .001).tolist()
        self.assert_buckets_fixed()

    def _ik(self, target, name, seed=None):
        from robust_ik import solve
        from expanded_pose_native import verified_fit
        key = (name, tuple(target.ravel()))
        if seed is None and key in self.ik_cache:
            return self.ik_cache[key]
        if seed is None:
            seed = self.poses["board_joint_rad" if name == "Arm14" else "dropper_joint_rad"]
        fit = solve(target, self.continuous_links[name], seed, name == "Arm14")
        if fit["solved"]:
            verified_fit(target, fit, self.continuous_links[name], name == "Arm14")
        return fit

    def board_poses(self):
        from parallel_runtime import ParallelShots

        # The endpoint-initialized historical task did not include large driven
        # transitions. Allow the restored original hold drives to dissipate
        # motion before each stability window starts. No velocity or
        # pose is overwritten, and no acceptance threshold is relaxed.
        if self.capture_trial and self.phase == "setup":
            if self.board_pose_reads % 2 == 0:
                for _ in range(round(1.75/self.dt)):
                    self.tick()
            self.board_pose_reads += 1
        return ParallelShots.board_poses(self)

    def tick(self):
        from parallel_runtime import ParallelShots

        ParallelShots.tick(self)
        if not self.capture_trial:
            return
        if self.phase == "release":
            self.physics_samples.append(dict(
                time_step=self.steps,
                position_m=self.ball.data.root_pos_w.cpu().numpy().copy(),
                velocity_m_s=self.ball.data.root_lin_vel_w.cpu().numpy().copy(),
                forces_N=self.forces().copy(),
                joint_position_rad=np.stack([a.data.joint_pos[:, :6].cpu().numpy() for a in self.arms.values()], axis=1).copy(),
                joint_velocity_rad_s=np.stack([a.data.joint_vel[:, :6].cpu().numpy() for a in self.arms.values()], axis=1).copy(),
                board_position_m=self.arms["Arm14"].data.body_pos_w[:, self.board_idx].cpu().numpy().copy(),
                board_quaternion_wxyz=self.arms["Arm14"].data.body_quat_w[:, self.board_idx].cpu().numpy().copy()))
        if self.film_trial and (self.steps-self.trial_start_step) % 8 == 0:
            self.sample(self.phase, 8)

    def render(self):
        if self.capture_trial and self.phase == "setup":
            self.release_step = self.steps
            self.phase = "release"

    def sample(self, phase, elapsed):
        eye, aim = self.view_pose()
        self.write_rgb(eye, aim, elapsed)
        self.frames[-1].update(phase=phase,
            all_joints_rad={n: a.data.joint_pos[:, :6].cpu().tolist() for n,a in self.arms.items()},
            all_ball_world_m=self.ball.data.root_pos_w.cpu().tolist(),
            mode=self.trial_mode, batch=self.trial_batch)
        if not (self.dest/"preview.jpg").exists():
            shutil.copyfile(self.frames_path/f"{self.frames[-1]['frame']:05d}.jpg", self.dest/"preview.jpg")


def run_study(env, bundle, batches=64, feasibility_path=None):
    import torch
    from continuous_policy import (ContinuousPolicy, build_observations, sample, physical_actions,
                                   weight_hash, actor_weight_hash, checkpoint_payload, update)
    from farm_contract import reward, statistics
    from parallel_runtime import atomic

    study = make_study(bundle, batches)
    feasibility = json.loads(Path(feasibility_path).read_text())
    if feasibility.get("source_fingerprint") != study["geometry_fingerprint"]:
        raise ValueError("Feasibility map belongs to a different farm geometry")
    unsigned = {k:v for k,v in feasibility.items() if k != "fingerprint"}
    if canonical_hash(unsigned) != feasibility.get("fingerprint"):
        raise ValueError("Feasibility map fingerprint changed")
    for filename, expected in feasibility["asset_sha256"].items():
        if digest(Path(env.args.assets)/filename) != expected:
            raise ValueError("Feasibility assets differ from native scene")
    shutil.copyfile(feasibility_path, env.dest/"feasibility.json")
    study["feasibility_sha256"] = digest(feasibility_path)
    study["coverage_batches"] = 4
    study["fingerprint"] = canonical_hash({k:v for k,v in study.items() if k != "fingerprint"})
    atomic(env.dest/"study.json", study)
    env.study = study
    env.focus = FOCUS
    env.active_mask = np.ones(35, dtype=bool)
    from prepare_assets import chain
    env.continuous_links = {name: chain(ET.parse(Path(env.args.assets)/filename).getroot())
        for name, filename in (("Arm10", "dropper.urdf"), ("Arm14", "board_arm.urdf"))}
    model = ET.parse(Path(env.args.assets)/"dropper.urdf")
    env.transit_limits = torch.tensor([float(model.find(f".//joint[@name='joint{i+1}']/limit").get("effort")) for i in range(6)], device=env.sim.device)
    torch.manual_seed(SEED)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    policy = ContinuousPolicy().to(env.sim.device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=study["ppo"]["lr"])
    observations = torch.as_tensor(build_observations(bundle["spec"]["cells"]), device=env.sim.device)
    train_mask = np.array([c["split"] == "train" for c in study["cells"]])
    if int(train_mask.sum()) != 25 or any(sum(c["split"] == label for c in study["cells"]) != 5 for label in ("validation", "test")):
        raise ValueError("Continuous pilot split changed")
    train_ids = torch.tensor(np.flatnonzero(train_mask), device=env.sim.device)
    initial_hash = weight_hash(policy)
    changes, summaries, assessments, coverages, clips, buffer = [], [], [], [], [], []
    (env.dest/"trials").mkdir()
    started = time.monotonic()

    def checkpoint(name, after_batches):
        path = env.dest/name
        payload = checkpoint_payload(policy, optimizer,
            study_fingerprint=study["fingerprint"], geometry_fingerprint=study["geometry_fingerprint"],
            domain_fingerprint=study["domain"]["fingerprint"], update_count=len(changes),
            attempted_shots=after_batches*int(train_mask.sum()), seed=SEED)
        torch.save(payload, path)
        return dict(file=name, sha256=digest(path), policy_hash=weight_hash(policy), actor_hash=actor_weight_hash(policy))

    def trial(mode, batch, cp, deterministic=False, film=False, coverage=None):
        if shutil.disk_usage(env.dest).free < 4*1024**3:
            raise RuntimeError("Less than 4GiB free during continuous pilot")
        env.active_mask = train_mask.copy() if mode == "train" else np.ones(35, dtype=bool)
        with torch.no_grad():
            picked = sample(policy, observations, deterministic=deterministic)
        physical = physical_actions(picked["actions_normalized"], LOW, HIGH)
        actions = [decode_action(c, a, f"{mode}-{batch}-{i}") for i,(c,a) in enumerate(zip(bundle["spec"]["cells"], physical))]
        if coverage is not None:
            actions = coverage["actions"]
            env.active_mask = np.asarray(coverage["enabled"], dtype=bool)
            env.coverage_fits = coverage["fits"]
        env.capture_trial, env.film_trial, env.phase = False, False, None
        home=env.home()
        env.home_readback=home
        env.trial_mode, env.trial_batch = mode, batch
        env.frames, env.physics_samples = [], []
        env.capture_trial, env.film_trial, env.phase = True, film, "setup"
        env.trial_start_step = env.steps
        first = env.next_frame
        rows, raw = env.shot_batch(actions, record_video=film, enabled=env.active_mask)
        env.capture_trial, env.film_trial, env.phase = False, False, None
        active = [i for i in range(35) if env.active_mask[i]]
        for i,r in enumerate(rows):
            r["split"] = study["cells"][i]["split"]
            r["qualification"] = env.qualifications[i]
            r["path_rejection"] = env.path_reasons[i]
            r["path_proxy"] = env.path_checks[i]
            r["scoring_board_normal"] = env.scoring_normals[i].tolist()
        returns = torch.tensor([reward(r) for r in rows], dtype=torch.float32, device=env.sim.device)
        name = f"{mode}-{len(summaries)+len(assessments)+len(coverages):04d}"
        physics = None
        if env.physics_samples:
            path = env.dest/f"{name}-physics.npz"
            save_physics(path, env.physics_samples)
            physics = dict(file=path.name, sha256=digest(path), samples=len(env.physics_samples))
        payload = dict(schema=SCHEMA, mode=mode, batch=batch, update_count=len(changes),
            study_fingerprint=study["fingerprint"], checkpoint=cp, policy_hash=weight_hash(policy),
            observations=observations.cpu().tolist(), latent=picked["latent"].cpu().tolist(),
            actions_normalized=picked["actions_normalized"].cpu().tolist(), logp=picked["logp"].cpu().tolist(),
            actions=actions, rows=rows, rewards=returns.cpu().tolist(),
            active_cell_ids=[study["cells"][i]["id"] for i in active],
            enabled=env.active_mask.tolist(), deterministic=deterministic,
            optimizer_samples=len(active) if mode == "train" else 0,
            policy_action_used=mode != "coverage",
            home=home,
            release_step=env.release_step, score_end_step=env.steps, physics=physics,
            trajectory_m=raw["trajectory_m"], wall_s=raw["wall_s"],
            transit_joint_error_rad=env.transit_joint_error_rad)
        filename=f"trials/{name}.json"
        atomic(env.dest/filename, payload)
        summary = dict(mode=mode, batch=batch, update_count=len(changes), file=filename,
                       sha256=digest(env.dest/filename), **statistics([rows[i] for i in active]))
        if film:
            clip = dict(mode=mode, batch=batch, update_count=len(changes), trial_file=filename,
                frames=env.frames, actions=actions, rows=rows, fps=30,
                start_frame=first, end_frame=env.next_frame,
                release_step=env.release_step, camera_sha256=bundle["source"]["camera_sha256"])
            clips.append(clip)
            atomic(env.dest/"clips.json", dict(clips=clips))
        print(json.dumps(summary), flush=True)
        return summary, (observations[train_ids], picked["latent"][train_ids],
                         picked["logp"][train_ids], returns[train_ids])

    initial = checkpoint("policy-initial.pt", 0)
    from continuous_feasibility import geometry_coverage
    for batch, plan in enumerate(geometry_coverage(feasibility, bundle["spec"]["cells"], batches=4)):
        row,_ = trial("coverage", batch, initial, film=True, coverage=plan)
        coverages.append(row)
    for repeat in range(study["assessment_batches"]):
        row,_ = trial("eval", repeat, initial, deterministic=True)
        assessments.append(row)
    for batch in range(batches):
        cp = checkpoint(f"policy-before-{batch:03d}.pt", batch)
        row, data = trial("train", batch, cp, film=batch in study["filmed_training_batches"])
        summaries.append(row)
        buffer.append(data)
        if len(buffer) == study["batches_per_update"]:
            values = update(policy, optimizer, *[torch.cat(group) for group in zip(*buffer)], study["ppo"])
            values.update(update_count=len(changes)+1, training_batches=list(range(batch-3,batch+1)),
                          sample_count=4*int(train_mask.sum()))
            changes.append(values)
            buffer.clear()
        atomic(env.dest/"progress.json", dict(schema=SCHEMA, status="training", study_fingerprint=study["fingerprint"],
            batches=summaries, assessments=assessments, coverage=coverages, updates=changes, frames=env.next_frame))
    final=checkpoint("policy-final.pt", batches)
    for repeat in range(study["assessment_batches"]):
        row,_=trial("eval", repeat, final, deterministic=True)
        assessments.append(row)
    result=dict(schema=SCHEMA, status="completed", study_fingerprint=study["fingerprint"],
        geometry_fingerprint=study["geometry_fingerprint"], batches=summaries, assessments=assessments,
        updates=changes, coverage=coverages, initial=initial, final=final, initial_hash=initial_hash,
        final_hash=weight_hash(policy), seconds=time.monotonic()-started, frames=env.next_frame,
        origins_m=env.origins.tolist(), camera=env.saved_camera, stage=env.stage_metadata,
        source_sha256={p.name:digest(p) for p in sorted(Path(__file__).parent.glob("*.py"))},
        gpu=torch.cuda.get_device_name(), workcells=35, articulations=70,
        peak_bucket_drift_m=env.peak_bucket_drift, scope=study["scope"])
    atomic(env.dest/"result.json", result)
    return result
