"""Episodic policy-gradient RL over an articulated Isaac Lab / PhysX scene.

One cooperative action per shot. Arms initialize at IK solutions and settle;
this trains shot parameters, NOT joint trajectories or physical grasping.
"""

import datetime
import json
import shutil
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import torch
from perception_stream import publish_sensor, publish_truth
from PIL import Image
from prepare_assets import chain, fk, matrix
from rl_contract import (
    HIGH,
    INITIAL_MEAN,
    LOW,
    VERSION,
    ShotScore,
    action_values,
    fingerprint,
)
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation


class FixedShot:
    def __init__(self, sim, arms, ball, camera, contact, cfg, poses, args, atomic):
        self.sim, self.arms, self.ball, self.camera, self.contact = (
            sim,
            arms,
            ball,
            camera,
            contact,
        )
        self.cfg, self.poses, self.args, self.atomic = cfg, poses, args, atomic
        self.out = Path(args.output)
        self.links = chain(ET.parse(Path(args.assets) / "board_arm.urdf").getroot())
        self.lo = np.array([l[2] for l in self.links])
        self.hi = np.array([l[3] for l in self.links])
        self.q = {
            "Arm10": np.array(poses["dropper_joint_rad"]),
            "Arm14": np.array(poses["board_joint_rad"]),
        }
        self.frames = 0
        self.episode = 0
        self.phase = "initializing"
        self.metrics = {}
        self.dt = cfg["simulation"]["physics_dt_s"]

    def ik(self, target, name):
        def error(q):
            cur = fk(q, self.links)
            return np.r_[
                cur[:3, 3] - target[:3, 3],
                0.25 * Rotation.from_matrix(target[:3, :3] @ cur[:3, :3].T).as_rotvec(),
            ]

        for seed in [
            np.array(
                self.poses[
                    "dropper_joint_rad" if name == "Arm10" else "board_joint_rad"
                ]
            ),
        ]:
            fit = least_squares(
                error,
                seed,
                bounds=(self.lo, self.hi),
                max_nfev=100,
                ftol=1e-8,
                gtol=1e-8,
                xtol=1e-8,
            )
            if np.linalg.norm(fit.fun) < 1e-4:
                return fit.x
        raise ValueError("unreachable endpoint")

    def tick(self, render=False):
        for a in self.arms.values():
            a.write_data_to_sim()
        self.sim.step(render=False)
        for a in self.arms.values():
            a.update(self.dt)
        self.ball.update(self.dt)
        self.contact.update(self.dt, force_recompute=True)
        self.frames += 1
        if render:
            self.sim.render()
            self.camera.update(self.dt * 24, force_recompute=True)
            self.publish()

    def publish(self):
        rgb = self.camera.data.output["rgb"][0, :, :, :3].cpu().numpy().astype(np.uint8)
        depth = (
            self.camera.data.output["distance_to_image_plane"][0, :, :, 0].cpu().numpy()
        )
        frame_id = publish_sensor(self.out, self.frames, rgb, depth, self.camera)
        publish_truth(self.out, frame_id, self.cfg, self.arms, self.ball)
        valid = np.isfinite(depth) & (depth > 0) & (depth < self.cfg["camera"]["far_m"])
        Image.fromarray(rgb).save(self.out / "latest.tmp.jpg")
        (self.out / "latest.tmp.jpg").replace(self.out / "latest.jpg")
        Image.fromarray(
            np.uint8(np.clip(1 - np.nan_to_num(depth, nan=5, posinf=5) / 5, 0, 1) * 255)
        ).save(self.out / "depth-preview.png")
        errs = [
            float((a.data.joint_pos - a.data.joint_pos_target).abs().max())
            for a in self.arms.values()
        ]
        self.atomic(
            self.out / "status.json",
            {
                "timestamp_utc": datetime.datetime.now(datetime.UTC).isoformat(),
                "frame": self.frames,
                "state": self.phase,
                "gpu": torch.cuda.get_device_name(0),
                "max_joint_target_error_rad": max(errs),
                "camera": {"valid_depth_pixels": int(valid.sum())},
                "arms": {
                    n: {"joint_names": a.joint_names} for n, a in self.arms.items()
                },
                "training": {
                    "episode": self.episode,
                    "phase": self.phase,
                    **self.metrics,
                },
            },
        )

    def before_release(self, target_board_world):
        """Optional versioned controller gate; legacy behavior remains unchanged."""
        return {}

    def board_contact_force(self):
        return float(torch.linalg.vector_norm(self.contact.data.force_matrix_w))

    def start_flight_observation(self):
        pass

    def observe_flight(self, pos, score):
        pass

    def finish_flight_observation(self, score):
        return {}

    def shot(self, action):
        self.episode += 1
        tilt, heading, gap = map(float, action)
        c = self.cfg
        center = np.array(c["board"]["initial_center_m"])
        delta = np.array(c["bucket"]["center_xy_m"]) - center[:2]
        yaw = np.arctan2(delta[1], delta[0]) + np.pi / 2 + np.deg2rad(heading)
        rot = (
            Rotation.from_euler("z", yaw).as_matrix()
            @ Rotation.from_euler("x", tilt, degrees=True).as_matrix()
            @ Rotation.from_euler("z", c["board"]["mount_yaw_rad"]).as_matrix()
        )
        target14 = np.eye(4)
        target14[:3, :3] = rot
        target14[:3, 3] = center
        base = matrix([c["arms"]["base_spacing_m"], 0, 0], [0, 0, np.pi])
        mount = matrix(
            [0, 0, c["board"]["tool_offset_m"]], [0, 0, c["board"]["mount_yaw_rad"]]
        )
        release = center + np.array([0, 0, c["ball"]["radius_m"] + gap])
        target10 = matrix(release + [0, 0, 0.095], [np.pi, 0, 0])
        try:
            q14 = self.ik(
                np.linalg.inv(base) @ target14 @ np.linalg.inv(mount), "Arm14"
            )
            q10 = self.ik(target10, "Arm10")
        except ValueError:
            return {
                "reward": -2.0,
                "success": False,
                "invalid": "endpoint_ik",
                "action": action.tolist(),
            }
        self.q = {"Arm10": q10, "Arm14": q14}
        for n, a in self.arms.items():
            q = a.data.default_joint_pos.clone()
            for i in range(6):
                q[0, a.joint_names.index(f"joint{i + 1}")] = float(self.q[n][i])
            a.write_joint_state_to_sim(q, torch.zeros_like(q))
            a.set_joint_position_target(q)
            a.reset()
        parked = self.ball.data.default_root_state.clone()
        parked[0, :3] = torch.tensor([0.1, -0.8, 0.04], device=self.sim.device)
        self.ball.write_root_state_to_sim(parked)
        self.ball.reset()
        self.contact.reset()
        for step in range(120):
            self.tick(render=step % 24 == 0)
        settled = self.before_release(target14)
        if settled.get("invalid"):
            return {
                "reward": -2.0,
                "success": False,
                "action": action.tolist(),
                **settled,
            }
        scoring_normal = np.array(settled.get("release_normal_world", rot[:, 2]))
        # Set up release state in the open proxy fingers; grasping is a separate task.
        state = self.ball.data.default_root_state.clone()
        state[0, :3] = torch.tensor(release, device=self.sim.device)
        state[0, 7:] = 0
        self.ball.write_root_state_to_sim(state)
        self.ball.reset()
        self.contact.reset()
        self.start_flight_observation()
        score = ShotScore(c["bucket"], c["ball"]["radius_m"])
        trail = []
        peak = 0.0
        for step in range(960):
            self.tick(render=step % 24 == 0)
            pos = self.ball.data.root_pos_w[0].cpu().numpy()
            force = self.board_contact_force()
            normal_speed = float(
                self.ball.data.root_lin_vel_w[0].cpu().numpy() @ scoring_normal
            )
            peak = max(peak, force)
            score.step(pos, force, self.dt, normal_speed)
            self.observe_flight(pos, score)
            if step % 8 == 0:
                trail.append(pos.tolist())
            if not np.isfinite(pos).all():
                raise RuntimeError("nonfinite PhysX state")
            if score.success or pos[2] < -0.2:
                break
        result = {
            **score.result(),
            **self.finish_flight_observation(score),
            **settled,
            "action": action.tolist(),
            "board_contact_peak_N": peak,
            "release_center_world_m": release.tolist(),
            "trajectory_m": trail,
            "arm_joint_targets": {n: q.tolist() for n, q in self.q.items()},
        }
        return result


def train(sim, arms, ball, camera, contact, cfg, poses, args, atomic):
    torch.manual_seed(args.rl_seed)
    np.random.seed(args.rl_seed)
    env = FixedShot(sim, arms, ball, camera, contact, cfg, poses, args, atomic)
    stamp = datetime.datetime.now(datetime.UTC).strftime("%Y%m%dT%H%M%SZ")
    run = Path(args.output) / ("rl-" + stamp)
    run.mkdir()
    contract = {
        "version": VERSION,
        "fingerprint": fingerprint(cfg),
        "scene": cfg,
        "seed": args.rl_seed,
        "action_names": ["tilt_deg", "bucket_heading_offset_deg", "release_gap_m"],
        "low": LOW.tolist(),
        "high": HIGH.tolist(),
        "algorithm": "REINFORCE with within-batch return baseline; one decision per episode",
        "observation": "fixed scene privileged constants; state-independent Gaussian actor",
        "limits": [
            "not joint trajectory learning",
            "reset at IK endpoints, no grasp learning",
            "rigid board",
            "ideal depth recorded for preview, not policy input",
        ],
        "physics": "PhysX, ball restitution max-combine; separate from foundation average-combine",
    }
    atomic(run / "contract.json", contract)
    atomic(
        Path(args.output) / "training.json",
        {"run": run.name, "phase": "initializing", "episodes": 0, "version": VERSION},
    )
    mean = torch.nn.Parameter(
        torch.tensor(INITIAL_MEAN, dtype=torch.float32, device=sim.device)
    )
    logstd = torch.nn.Parameter(torch.full((3,), -0.35, device=sim.device))
    optimizer = torch.optim.Adam([mean, logstd], lr=0.06)

    def deterministic():
        return action_values(mean.detach().cpu().numpy())

    if args.rl_checkpoint:
        checkpoint = json.loads(Path(args.rl_checkpoint).read_text())
        if checkpoint["contract"]["fingerprint"] != fingerprint(cfg):
            raise ValueError("Checkpoint scene/physics/scoring fingerprint mismatch")
        with torch.no_grad():
            mean.copy_(torch.tensor(checkpoint["mean"], device=sim.device))
        env.phase = "frozen_checkpoint_evaluation"
        results = [env.shot(deterministic()) for _ in range(args.rl_episodes)]
        summary = {
            "run": run.name,
            "phase": env.phase,
            "successes": sum(r["success"] for r in results),
            "trials": len(results),
            "action": deterministic().tolist(),
            "source_checkpoint": str(args.rl_checkpoint),
        }
        atomic(run / "evaluation.json", results)
        atomic(run / "summary.json", summary)
        atomic(Path(args.output) / "result.json", {"passed": True, **summary})
        print("RL_FROZEN_EVAL", summary, flush=True)
        return
    env.phase = "midpoint_heuristic_baseline"
    heuristic = [env.shot(action_values(np.zeros(3))) for _ in range(2)]
    atomic(run / "heuristic_baseline.json", heuristic)
    env.phase = "baseline"
    baseline = [env.shot(deterministic()) for _ in range(2)]
    atomic(run / "baseline.json", baseline)
    samples = []
    returns = []
    records = []
    started = time.monotonic()
    env.phase = "training"
    for episode in range(args.rl_episodes):
        if shutil.disk_usage(run).free < 3 * 1024**3:
            raise RuntimeError("less than 3GiB free")
        dist = torch.distributions.Normal(mean, logstd.exp())
        z = dist.sample()
        action = action_values(z.cpu().numpy())
        result = env.shot(action)
        samples.append(z)
        returns.append(result["reward"])
        record = {"episode": episode + 1, "phase": "training", **result}
        records.append(record)
        with (run / "episodes.jsonl").open("a") as f:
            f.write(json.dumps(record) + "\n")
        env.metrics = {
            "last_reward": result["reward"],
            "last_success": result["success"],
            "action": action.tolist(),
            "recent_success_rate": float(
                np.mean([r["success"] for r in records[-32:]])
            ),
        }
        if len(samples) == 16 or episode == args.rl_episodes - 1:
            ret = torch.tensor(returns, device=sim.device)
            adv = (ret - ret.mean()) / (ret.std(unbiased=False) + 1e-6)
            dist = torch.distributions.Normal(mean, logstd.exp())
            loss = (
                -(dist.log_prob(torch.stack(samples)).sum(-1) * adv.detach()).mean()
                - 0.005 * dist.entropy().sum()
            )
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_([mean, logstd], 1.0)
            optimizer.step()
            with torch.no_grad():
                logstd.clamp_(-2.0, 0.3)
            samples = []
            returns = []
            checkpoint = {
                "contract": contract,
                "mean": mean.detach().cpu().tolist(),
                "logstd": logstd.detach().cpu().tolist(),
                "episodes": episode + 1,
                "deterministic_action": deterministic().tolist(),
                "initial_mean": INITIAL_MEAN.tolist(),
            }
            atomic(run / "checkpoint.json", checkpoint)
            atomic(
                Path(args.output) / "training.json",
                {
                    **env.metrics,
                    "episodes": episode + 1,
                    "run": run.name,
                    "phase": "training",
                },
            )
            print(
                "RL_UPDATE",
                episode + 1,
                env.metrics,
                "mean_action",
                deterministic().tolist(),
                flush=True,
            )
    # Freeze actor; no optimization/search during evaluation.
    env.phase = "frozen_evaluation"
    evaluation = [env.shot(deterministic()) for _ in range(8)]
    atomic(run / "evaluation.json", evaluation)
    summary = {
        "run": run.name,
        "training_episodes": args.rl_episodes,
        "baseline_successes": sum(r["success"] for r in baseline),
        "baseline_trials": len(baseline),
        "heuristic_successes": sum(r["success"] for r in heuristic),
        "heuristic_trials": len(heuristic),
        "frozen_successes": sum(r["success"] for r in evaluation),
        "frozen_trials": len(evaluation),
        "final_action": deterministic().tolist(),
        "elapsed_s": time.monotonic() - started,
        "scope": "same fixed scene, deterministic repeats; not generalization or physical validation",
    }
    atomic(run / "summary.json", summary)
    atomic(Path(args.output) / "training.json", summary)
    atomic(
        Path(args.output) / "result.json",
        {"passed": True, "state": "rl_complete", **summary},
    )
    print("RL_COMPLETE", summary, flush=True)
