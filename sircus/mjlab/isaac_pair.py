"""Fresh Isaac PhysX counterpart to the fixed-endpoint MuJoCo contact benchmark."""

import argparse
import copy
import json
import os
import time
import traceback
from pathlib import Path

BOOT_STARTED = time.monotonic()

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--benchmark", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--assets", default="/workspace/assets")
parser.add_argument("--substeps", type=int, default=1)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.output.exists():
    parser.error("Use a fresh result file")
if args.substeps < 1:
    parser.error("--substeps must be positive")
args.output.parent.mkdir(parents=True, exist_ok=True)
app = AppLauncher(args).app

import isaaclab
import isaaclab.sim as sim_utils
import numpy as np
import torch
from contract import FORCE_NAMES, ScoreStream, verify_manifest
from isaaclab.scene import InteractiveScene
from parallel_assets import bucket_asset, robot_body_filters, scene_config
from parallel_runtime import ParallelShots, atomic

# The original native source stored q in these observed articulation array orders.
SOURCE_JOINT_ORDER = dict(
    Arm10=[f"joint{i+1}" for i in range(6)]
    + ["finger_left_joint", "finger_right_joint"],
    Arm14=[f"joint{i+1}" for i in range(6)],
)


class FixedEndpoints(ParallelShots):
    targets = None

    def tick(self):
        if self.targets is not None:
            for name, arm in self.arms.items():
                arm.write_joint_state_to_sim(
                    self.targets[name], torch.zeros_like(self.targets[name])
                )
                arm.set_joint_position_target(self.targets[name])
        super().tick()


def run():
    torch.set_num_threads(2)
    manifest = verify_manifest(json.loads(args.benchmark.read_text()))
    shots = manifest["shots"]
    cfg = manifest["scene"]
    runtime_cfg = copy.deepcopy(cfg)
    runtime_cfg["simulation"]["physics_dt_s"] = manifest["dt_s"] / args.substeps
    poses = json.loads((Path(args.assets) / "poses.json").read_text())
    stage_dir = args.output.parent / (args.output.stem + "-scene")
    stage_dir.mkdir(exist_ok=False)
    spec = dict(
        num_envs=len(shots), cells=[dict(bucket_xy_m=s["bucket_xy_m"]) for s in shots]
    )
    sim = sim_utils.SimulationContext(
        sim_utils.SimulationCfg(
            dt=runtime_cfg["simulation"]["physics_dt_s"],
            render_interval=8,
            device=args.device,
            physx=sim_utils.PhysxCfg(enable_ccd=False),
        )
    )
    bucket = stage_dir / "bucket.usda"
    bucket_asset(bucket, cfg["bucket"])
    scfg, filters = scene_config(
        cfg, poses, args.assets, bucket, len(shots), robot_body_filters(args.assets)
    )
    scene = InteractiveScene(scfg)
    sim.reset()
    env = FixedEndpoints(
        sim,
        scene,
        None,
        None,
        runtime_cfg,
        poses,
        args,
        copy.deepcopy(spec),
        filters,
        stage_dir,
    )
    env.targets = {}
    for name, arm in env.arms.items():
        targets = arm.data.default_joint_pos.clone()
        for i, s in enumerate(shots):
            q = s["final_joint_positions_rad"][name]
            if len(q) != len(SOURCE_JOINT_ORDER[name]):
                raise ValueError(f"Unknown source joint order for {name}")
            for joint, value in zip(SOURCE_JOINT_ORDER[name], q):
                targets[i, arm.joint_names.index(joint)] = value
        env.targets[name] = targets
    for _ in range(120 * args.substeps):
        env.tick()
    rows = []
    rollout_seconds = 0.0
    warmup_seconds = 0.0
    setup_seconds = None
    for repeat in range(manifest["repeats"]):
        warmup_started = time.monotonic()
        for _ in range(120 * args.substeps):
            env.tick()
        warmup_seconds += time.monotonic() - warmup_started
        if setup_seconds is None:
            setup_seconds = time.monotonic() - BOOT_STARTED
        centers, normals = env.board_poses()
        position_errors = (
            np.linalg.norm(
                centers - np.array([s["settled_board_center_m"] for s in shots]), axis=1
            )
            * 1000
        )
        normal_errors = np.rad2deg(
            np.arccos(
                np.clip(
                    np.sum(
                        normals * np.array([s["settled_board_normal"] for s in shots]),
                        axis=1,
                    ),
                    -1,
                    1,
                )
            )
        )
        flange = (
            env.arms["Arm10"].data.body_pos_w[:, env.flange_idx].cpu().numpy()
            - env.origins
        )
        release = np.array([s["release_ball_center_m"] for s in shots])
        dropper_errors = (
            np.linalg.norm(flange - (release + [0, 0, 0.095]), axis=1) * 1000
        )
        invalid = [
            None if p <= 2 and n <= 0.5 and d <= 2 else "source_endpoint_pose_mismatch"
            for p, n, d in zip(position_errors, normal_errors, dropper_errors)
        ]
        xyz = release.copy()
        xyz[[i for i, reason in enumerate(invalid) if reason]] = [0.1, -0.8, 0.04]
        env.set_balls(xyz)
        streams = [
            ScoreStream(cfg, s["bucket_xy_m"], normal)
            for s, normal in zip(shots, normals)
        ]
        tracks = [[] for _ in shots]
        times = []
        max_drift = np.zeros(len(shots))
        first_contact = [dict.fromkeys(FORCE_NAMES) for _ in shots]
        peaks = np.zeros((len(shots), len(FORCE_NAMES)))
        rollout_started = time.monotonic()
        for step in range(manifest["horizon_steps"]):
            forces = np.zeros((len(shots), len(FORCE_NAMES)))
            for _ in range(args.substeps):
                env.tick()
                forces = np.maximum(forces, env.forces())
                board_centers, _ = env.board_poses()
                max_drift = np.maximum(
                    max_drift, np.linalg.norm(board_centers - centers, axis=1) * 1000
                )
            pos = (env.ball.data.root_pos_w - env.origins_t).cpu().numpy()
            vel = env.ball.data.root_lin_vel_w.cpu().numpy()
            peaks = np.maximum(peaks, forces)
            for i, stream in enumerate(streams):
                if invalid[i] is None:
                    stream.step(
                        pos[i], vel[i], dict(zip(FORCE_NAMES, map(float, forces[i])))
                    )
                for j, key in enumerate(FORCE_NAMES):
                    if first_contact[i][key] is None and forces[i, j] > 0.01:
                        first_contact[i][key] = step + 1
            if (step + 1) % 4 == 0:
                times.append(step + 1)
                for i, p in enumerate(pos):
                    tracks[i].append(p.tolist())
        rollout_seconds += time.monotonic() - rollout_started
        for i, (s, stream) in enumerate(zip(shots, streams)):
            rows.append(
                dict(
                    shot_id=s["shot_id"],
                    repeat=repeat,
                    invalid=invalid[i],
                    **stream.result(),
                    board_center_m=centers[i].tolist(),
                    board_normal=normals[i].tolist(),
                    source_position_error_mm=float(position_errors[i]),
                    source_normal_error_deg=float(normal_errors[i]),
                    source_dropper_error_mm=float(dropper_errors[i]),
                    max_board_drift_mm=float(max_drift[i]),
                    trajectory_steps=list(times),
                    trajectory_m=tracks[i],
                    first_contact_step=first_contact[i],
                    all_horizon_contact_peaks_N=dict(
                        zip(FORCE_NAMES, map(float, peaks[i]))
                    ),
                )
            )
        atomic(
            args.output.with_suffix(".progress.json"),
            dict(completed_repeats=repeat + 1, rows=len(rows)),
        )
    atomic(
        args.output,
        dict(
            benchmark_fingerprint=manifest["fingerprint"],
            engine="isaac_physx",
            comparison_protocol="fixed_endpoint_contact_replay",
            versions=dict(
                isaaclab=getattr(isaaclab, "__version__", "unknown"),
                torch=torch.__version__,
                gpu=torch.cuda.get_device_name(),
                physics_substeps=args.substeps,
                physics_dt_s=runtime_cfg["simulation"]["physics_dt_s"],
                scoring_dt_s=manifest["dt_s"],
            ),
            source_joint_order=SOURCE_JOINT_ORDER,
            timing=dict(
                setup_and_compile_s=setup_seconds,
                rollout_with_shared_scoring_s=rollout_seconds,
                repeat_warmup_s=warmup_seconds,
                kernel_only_s=None,
                scope="Setup includes application startup, scene construction and first warmup. Rollout sums only three 960-tick loops with CPU shared scoring, state/force reads and trajectory sampling; all warmup is excluded. Kernel-only throughput was not measured.",
            ),
            joint_intervention="Set identical recorded source joint angles and zero joint velocities before every physics tick; physics integrates each tick normally.",
            contact_sampling="240 Hz scores receive the per-category maximum force norm across the inner physics steps and the final position/velocity of each scoring interval.",
            shots=rows,
        ),
    )
    print(
        json.dumps(
            dict(
                result=str(args.output),
                trials=len(rows),
                invalid=sum(bool(r["invalid"]) for r in rows),
                captures=sum(r["route_success"] for r in rows),
            )
        ),
        flush=True,
    )


try:
    run()
except BaseException:
    atomic(args.output.with_suffix(".failure.json"), dict(error=traceback.format_exc()))
    traceback.print_exc()
    os._exit(1)
app.close(skip_cleanup=True)
