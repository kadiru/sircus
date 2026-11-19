"""Render one bounded, frozen four-cell replay on the simulation GPU only."""

# Isaac imports must follow AppLauncher.
import argparse
import datetime
import hashlib
import json
import os
import shutil
import traceback
from pathlib import Path

from isaaclab.app import AppLauncher
from overhead_contract import CAPTURE_STRIDE, cameras, selection

p = argparse.ArgumentParser()
p.add_argument("--source-summary", type=Path, required=True)
p.add_argument("--assets", default="/workspace/assets")
p.add_argument("--output", type=Path, default=Path("/workspace/output"))
p.add_argument("--preview-only", action="store_true")
AppLauncher.add_app_launcher_args(p)
a = p.parse_args()
if not a.enable_cameras:
    p.error("--enable_cameras is required")
a.output.mkdir(parents=True, exist_ok=True)
if shutil.disk_usage(a.output).free < 3 * 1024**3:
    p.error("Less than 3 GiB free")
stamp = datetime.datetime.now(datetime.UTC).strftime("%Y%m%dT%H%M%SZ")
dest = a.output / f"overhead-replay-{stamp}"
dest.mkdir()
app = AppLauncher(a).app

import isaaclab.sim as sim_utils
import numpy as np
import torch
from isaaclab.scene import InteractiveScene
from isaaclab.sensors import Camera, CameraCfg
from parallel_assets import bucket_asset, robot_body_filters, scene_config
from parallel_contract import specification
from parallel_runtime import ParallelShots, atomic
from PIL import Image
from pxr import UsdGeom
from scipy.spatial.transform import Rotation


class Replay(ParallelShots):
    def __init__(self, *args, views, **kwargs):
        self.views = views
        self.samples = []
        self.capture_start = None
        self.last_capture_step = None
        self.contact_start = None
        self.contact_samples = []
        super().__init__(*args, **kwargs)

    def tick(self):
        super().tick()
        if self.record_video:
            self.contact_samples.append(
                {"step": self.steps, "forces_N": self.forces().tolist()}
            )
            if (self.steps - self.capture_start) % CAPTURE_STRIDE == 0:
                self.render()

    def render(self):
        if self.record_video:
            if self.capture_start is None:
                self.capture_start = self.steps
            if self.steps == self.last_capture_step:
                return
            # Base runner also asks for renders on its older 8-step cadence.
            if (self.steps - self.capture_start) % CAPTURE_STRIDE:
                return
        for _ in range(3):
            self.sim.render()
        sample = {
            "step": self.steps,
            "time_s": (self.steps - (self.capture_start or self.steps)) * self.dt,
            "positions_m": self.ball.data.root_pos_w.cpu().numpy().tolist(),
            "forces_N": self.forces().tolist(),
            "cameras": {},
        }
        for spec, camera in self.views:
            camera.update(self.dt * CAPTURE_STRIDE, force_recompute=True)
            rgb = camera.data.output["rgb"][0, :, :, :3].cpu().numpy().astype(np.uint8)
            image = Image.fromarray(rgb)
            image.save(self.dest / (spec["name"] + ".jpg"), quality=94)
            if self.record_video:
                image.save(
                    self.frames_path / f"{spec['name']}-{self.video_frames:04d}.jpg",
                    quality=94,
                )
            quat = camera.data.quat_w_ros[0].cpu().numpy()
            sample["cameras"][spec["name"]] = {
                "position_m": camera.data.pos_w[0].cpu().numpy().tolist(),
                "rotation_ros": Rotation.from_quat(quat[[1, 2, 3, 0]])
                .as_matrix()
                .tolist(),
            }
        if self.record_video:
            self.samples.append(sample)
            self.last_capture_step = self.steps
            self.video_frames += 1
            if self.video_frames % 60 == 0:
                print(
                    json.dumps(
                        {"frames": self.video_frames, "time_s": sample["time_s"]}
                    ),
                    flush=True,
                )
        self.render_count += 1


def main():
    cfg = json.loads(Path(__file__).with_name("scene.json").read_text())
    source = json.loads(a.source_summary.read_text())
    indices, actions = selection(source, cfg)
    poses = json.loads((Path(a.assets) / "poses.json").read_text())
    spec = specification(cfg, 32, indices, 0, 0)
    meta = {
        "run": dest.name,
        "source_run": source["run"],
        "source_fingerprint": source["contract"]["fingerprint"],
        "source_summary_sha256": hashlib.sha256(
            a.source_summary.read_bytes()
        ).hexdigest(),
        "scene": cfg,
        "cells": spec["cells"],
        "actions": actions,
        "cameras": cameras(),
        "projection": "orthographic",
        "dt_s": cfg["simulation"]["physics_dt_s"],
        "capture_stride": CAPTURE_STRIDE,
        "phase": "preparing",
        "scope": "Frozen endpoint-initialized PhysX replay. No pickup, arm transit, gripper actuation or new RL training.",
        "source_sha256": {
            f.name: hashlib.sha256(f.read_bytes()).hexdigest()
            for f in Path(__file__).parent.glob("*.py")
        },
    }
    atomic(dest / "replay.json", meta)
    sim = sim_utils.SimulationContext(
        sim_utils.SimulationCfg(
            dt=meta["dt_s"],
            render_interval=CAPTURE_STRIDE,
            device=a.device,
            physx=sim_utils.PhysxCfg(enable_ccd=False),
        )
    )
    bucket_asset(dest / "bucket.usda", cfg["bucket"])
    scfg, filters = scene_config(
        cfg, poses, a.assets, dest / "bucket.usda", 4, robot_body_filters(a.assets)
    )
    scene = InteractiveScene(scfg)
    views = []
    for c in meta["cameras"]:
        path = "/World/Replay_" + c["name"]
        camera = Camera(
            CameraCfg(
                prim_path=path,
                width=c["width"],
                height=c["height"],
                update_period=0,
                update_latest_camera_pose=True,
                data_types=["rgb"],
                spawn=sim_utils.PinholeCameraCfg(
                    focal_length=20,
                    horizontal_aperture=c["span_m"] * 10,
                    vertical_aperture=c["span_m"] * c["height"] / c["width"] * 10,
                    clipping_range=(
                        0.05,
                        float(np.linalg.norm(np.array(c["eye"]) - c["look"])) + 1.0,
                    ),
                ),
            )
        )
        # USD orthographic projection is renderer-supported. Isaac's convenience
        # spawner only names pinhole; no pinhole intrinsics/depth are used here.
        usd = UsdGeom.Camera.Get(sim.stage, path)
        usd.GetProjectionAttr().Set(UsdGeom.Tokens.orthographic)
        assert str(usd.GetProjectionAttr().Get()) == "orthographic"
        views.append((c, camera))
    sim.reset()
    origins = scene.env_origins.cpu().numpy()
    for c, camera in views:
        eye = origins[c["cell"]] + c["eye"]
        look = origins[c["cell"]] + c["look"]
        if c["name"] == "top":
            # ROS camera x=world +X, y=world -Y, z=world -Z; true plan view.
            camera.set_world_poses(
                torch.tensor(np.array([eye]), device=sim.device, dtype=torch.float32),
                torch.tensor([[0.0, 1.0, 0.0, 0.0]], device=sim.device),
                convention="ros",
            )
        else:
            camera.set_world_poses_from_view(
                torch.tensor(np.array([eye]), device=sim.device, dtype=torch.float32),
                torch.tensor(np.array([look]), device=sim.device, dtype=torch.float32),
            )
        camera.reset()
    env = Replay(
        sim,
        scene,
        views[0][1],
        views[1][1],
        cfg,
        poses,
        a,
        spec,
        filters,
        dest,
        views=views,
    )
    for _ in range(12):
        sim.render()
    if a.preview_only:
        xyz, _, records, invalid = env.prepare(actions)
        if any(invalid):
            raise RuntimeError(f"Invalid preview endpoints: {invalid}")
        env.set_balls(xyz)
        env.render()
        meta.update(phase="preview_only", settling=records)
    else:
        meta["probes"] = env.probes()
        rows, raw = env.shot_batch(actions, record_video=True)
        atomic(dest / "samples.json", env.samples)
        atomic(dest / "contact-samples.json", env.contact_samples)
        atomic(dest / "raw.json", raw)
        meta.update(
            phase="completed",
            rows=rows,
            frames=env.video_frames,
            max_bucket_drift_m=env.peak_bucket_drift,
            raw_sha256={
                f: hashlib.sha256((dest / f).read_bytes()).hexdigest()
                for f in ("samples.json", "contact-samples.json", "raw.json")
            },
        )
    atomic(dest / "replay.json", meta)
    print(
        json.dumps(
            {
                "run": str(dest),
                "phase": meta["phase"],
                "captures": sum(r["route_success"] for r in meta.get("rows", [])),
            }
        ),
        flush=True,
    )


try:
    main()
except BaseException:  # noqa: BLE001 -- Kit teardown can mask failures; preserve exit status.
    traceback.print_exc()
    atomic(dest / "failure.json", {"error": traceback.format_exc()})
    os._exit(1)
app.close(skip_cleanup=True)
