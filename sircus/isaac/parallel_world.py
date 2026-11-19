"""Run isolated two-arm tables in one GPU physics scene; never connects to hardware."""

import argparse
import datetime
import hashlib
import json
import os
import shutil
import sys
import time
import traceback
from pathlib import Path

from isaaclab.app import AppLauncher
from parallel_contract import candidates, select, specification

parser = argparse.ArgumentParser()
parser.add_argument("--assets", default="/workspace/assets")
parser.add_argument("--output", default="/workspace/output")
parser.add_argument("--grid-count", type=int, choices=[4, 32, 64], default=32)
parser.add_argument("--num-envs", type=int, default=32)
parser.add_argument("--cell-start", type=int, default=0)
parser.add_argument("--batches", type=int, default=len(candidates()))
parser.add_argument("--confirmations", type=int, default=10)
parser.add_argument("--smoke", action="store_true")
parser.add_argument("--video-batches", type=int, default=1)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if not 1 <= args.num_envs <= 64 or args.num_envs + args.cell_start > args.grid_count:
    parser.error("Choose 1..64 active cells within the grid")
if args.smoke:
    args.batches, args.confirmations, args.video_batches = 0, 0, 0
if args.video_batches not in (0, 1, 2):
    parser.error("Video recording is bounded to 0..2 batches")
root = Path(args.output)
root.mkdir(parents=True, exist_ok=True)
if shutil.disk_usage(root).free < 3 * 1024**3:
    parser.error("Free disk below 3 GiB")
stamp = datetime.datetime.now(datetime.UTC).strftime("%Y%m%dT%H%M%SZ")
dest = root / f"parallel-world-{stamp}-n{args.num_envs}"
dest.mkdir()
app = AppLauncher(args).app

import isaaclab.sim as sim_utils
import numpy as np
import torch
from isaaclab.scene import InteractiveScene
from isaaclab.sensors import Camera, CameraCfg
from parallel_assets import bucket_asset, robot_body_filters, scene_config
from parallel_contract import batch_actions
from parallel_runtime import ParallelShots, atomic


def main():
    started = time.monotonic()
    cfg = json.loads(Path(__file__).with_name("scene.json").read_text())
    poses = json.loads((Path(args.assets) / "poses.json").read_text())
    indices = list(range(args.cell_start, args.cell_start + args.num_envs))
    spec = specification(
        cfg, args.grid_count, indices, args.batches, args.confirmations
    )
    spec["source_sha256"] = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(Path(__file__).parent.glob("*.py"))
    }
    spec["gpu"] = torch.cuda.get_device_name(0)
    spec["smoke"] = args.smoke
    atomic(dest / "contract.json", spec)
    sim = sim_utils.SimulationContext(
        sim_utils.SimulationCfg(
            dt=cfg["simulation"]["physics_dt_s"],
            render_interval=8,
            device=args.device,
            physx=sim_utils.PhysxCfg(enable_ccd=False),
        )
    )
    bucket_usd = dest / "bucket.usda"
    bucket_asset(bucket_usd, cfg["bucket"])
    scene_cfg, filters = scene_config(
        cfg,
        poses,
        args.assets,
        bucket_usd,
        args.num_envs,
        robot_body_filters(args.assets),
    )
    scene = InteractiveScene(scene_cfg)
    cameras = []
    for name, width, height, far in [
        ("Overview", 1440, 900, 150.0),
        ("Close", 720, 900, 8.0),
    ]:
        cameras.append(
            Camera(
                CameraCfg(
                    prim_path="/World/" + name,
                    update_period=0,
                    update_latest_camera_pose=True,
                    width=width,
                    height=height,
                    data_types=["rgb", "distance_to_image_plane"],
                    spawn=sim_utils.PinholeCameraCfg(
                        focal_length=20,
                        horizontal_aperture=36,
                        clipping_range=(0.05, far),
                    ),
                )
            )
        )
    sim.reset()
    origins = scene.env_origins.cpu().numpy()
    center = origins.mean(axis=0) + [0.67, -0.25, 0.1]
    span = max(np.ptp(origins[:, 0]), np.ptp(origins[:, 1]), 2.5)
    for camera, eye, look in [
        (cameras[0], center + [span * 0.85, -span * 1.10, span * 1.15], center),
        (cameras[1], origins[0] + [1.9, -2.2, 1.7], origins[0] + [0.65, -0.20, 0.35]),
    ]:
        camera.set_world_poses_from_view(
            torch.tensor(np.array([eye]), dtype=torch.float32, device=sim.device),
            torch.tensor(np.array([look]), dtype=torch.float32, device=sim.device),
        )
        camera.reset()
    sim.set_camera_view(
        (center + [span * 0.85, -span * 1.10, span * 1.15]).tolist(), center.tolist()
    )
    env = ParallelShots(sim, scene, *cameras, cfg, poses, args, spec, filters, dest)
    for _ in range(8):
        sim.render()
    state = {
        "run": dest.name,
        "phase": "validating",
        "timestamp_utc": "",
        "contract": spec,
        "cells": [
            dict(c, rows=[], confirmations=[], winner=None) for c in spec["cells"]
        ],
        "controls": [],
        "probes": [],
        "batches_completed": 0,
        "attempts": 0,
        "scope": spec["scope"],
    }

    def publish(phase):
        state.update(
            phase=phase,
            timestamp_utc=datetime.datetime.now(datetime.UTC).isoformat(),
            elapsed_s=time.monotonic() - started,
            max_bucket_drift_m=env.peak_bucket_drift,
            view=dict(
                env.last_view,
                overview=f"/parallel-media/{dest.name}/overview.jpg",
                close=f"/parallel-media/{dest.name}/close.jpg",
            ),
            gpu_allocated_bytes=torch.cuda.max_memory_allocated(),
            video_frames=env.video_frames,
        )
        atomic(dest / "summary.json", state)
        if not args.smoke:
            atomic(root / "parallel-world.json", state)

    def attempt(actions, stage, record=False, enabled=None):
        before = time.monotonic()
        rows, raw = env.shot_batch(actions, record_video=record, enabled=enabled)
        for i, row in enumerate(rows):
            if enabled is not None and not enabled[i]:
                continue
            state["attempts"] += 1
            row.update(trial_id=f"{dest.name}-{state['attempts']:05d}", stage=stage)
        with (dest / "trials.jsonl").open("a") as f:
            f.write(
                json.dumps({"stage": stage, "rows": rows, "raw": raw}, allow_nan=False)
                + "\n"
            )
        print(
            json.dumps(
                {
                    "stage": stage,
                    "wall_s": time.monotonic() - before,
                    "route": sum(r["route_success"] for r in rows),
                    "invalid": sum(bool(r["invalid"]) for r in rows),
                }
            ),
            flush=True,
        )
        return rows, raw["wall_s"]

    try:
        env.render()
        publish("validating")
        state["probes"] = env.probes()
        for ci, a in enumerate(candidates()[:2]):
            rows, elapsed = attempt([dict(a) for _ in indices], f"control-{ci}")
            state["controls"].append({"rows": rows, "wall_s": elapsed})
            publish("controls")
        if args.smoke:
            state["validation_passed"] = True
            publish("completed")
            return
        for batch in range(args.batches):
            if shutil.disk_usage(root).free < 3 * 1024**3:
                raise RuntimeError("Free disk below 3 GiB")
            rows, elapsed = attempt(
                batch_actions(batch, spec["cells"]),
                f"search-{batch}",
                record=batch < args.video_batches,
            )
            for cell, row in zip(state["cells"], rows):
                cell["rows"].append(row)
            state["batches_completed"] += 1
            publish("searching")
        enabled = []
        for i, cell in enumerate(state["cells"]):
            # Both control settings are declared development candidates, not held-out tests.
            winner = select(cell["rows"] + [c["rows"][i] for c in state["controls"]])
            cell["winner"] = (
                {k: winner[k] for k in ("id", "height_m", "gap_m", "tilt_deg")}
                if winner
                else None
            )
            enabled.append(bool(winner))
        publish("confirming")
        for rep in range(args.confirmations):
            if not any(enabled):
                break
            actions = []
            for cell in state["cells"]:
                a = dict(cell["winner"] or candidates()[0])
                seed = 20261102 + 1000 * cell["index"] + rep
                rng = np.random.default_rng(seed)
                for key, delta in spec["perturbations"].items():
                    a[key] += float(rng.uniform(-delta, delta))
                a["seed"] = seed
                actions.append(a)
            rows, elapsed = attempt(actions, f"confirmation-{rep}", enabled=enabled)
            for i, cell in enumerate(state["cells"]):
                if enabled[i]:
                    cell["confirmations"].append(rows[i])
            publish("confirming")
        state["validation_passed"] = True
        state["artifact_sha256"] = {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (dest / "contract.json", dest / "trials.jsonl", bucket_usd)
        }
        publish("completed")
        print("PARALLEL_WORLD_COMPLETE " + str(dest), flush=True)
    except Exception as exc:
        state["error"] = f"{type(exc).__name__}: {exc}"
        publish("failed")
        raise


try:
    main()
except BaseException as exc:  # noqa: BLE001 -- Kit shutdown would hide the failure status.
    traceback.print_exc()
    atomic(
        dest / "failure.json",
        {"error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()},
    )
    # Kit shutdown exits with code zero; preserve batch failure for callers.
    sys.stderr.flush()
    os._exit(1)
finally:
    app.close(skip_cleanup=True)
