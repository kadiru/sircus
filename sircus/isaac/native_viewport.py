"""Bounded native Isaac GUI demonstration, recorded externally from its X11 window.

Uses the same 32-cell PhysX task and PPO implementation. No offline image compositor,
camera-frame synthesis or trained arm-transit claim. Marker files synchronize the
window-only recorder without connecting to physical hardware or cloud services.
"""

import argparse
import datetime
import hashlib
import json
import os
import time
import traceback
from pathlib import Path

from isaaclab.app import AppLauncher

p = argparse.ArgumentParser()
p.add_argument("--assets", default="/workspace/assets")
p.add_argument("--output", type=Path, required=True)
p.add_argument(
    "--view", choices=("orthographic", "perspective"), default="orthographic"
)
AppLauncher.add_app_launcher_args(p)
a = p.parse_args()
if a.headless or a.livestream > 0:
    p.error("This demonstration requires the native desktop window")
if not a.enable_cameras:
    p.error("This demonstration requires --enable_cameras")
a.output.mkdir(parents=True, exist_ok=False)
app = AppLauncher(a, window_width=2400, window_height=1340, width=2048, height=1152).app

import isaaclab.sim as sim_utils
import numpy as np
import omni.kit.viewport.utility
import omni.ui as ui
import torch
from farm_contract import actions, contract, observation, reward, statistics
from farm_policy import Policy, update, weight_hash
from isaaclab.scene import InteractiveScene
from isaaclab.sensors import Camera, CameraCfg
from parallel_assets import bucket_asset, robot_body_filters, scene_config
from parallel_contract import specification
from parallel_runtime import ParallelShots, atomic
from pxr import UsdGeom


class LiveShots(ParallelShots):
    pace = False
    deadline = None

    def tick(self):
        super().tick()
        if self.pace and self.steps % 8 == 0:
            self.sim.render()
            now = time.monotonic()
            target = (self.deadline or now) + 8 * self.dt
            if target > now:
                time.sleep(target - now)
            self.deadline = max(target, time.monotonic())

    def render(self):
        # The native viewport draws the live USD/PhysX scene directly.
        self.sim.render()


def idle(sim, seconds):
    until = time.monotonic() + seconds
    while app.is_running() and time.monotonic() < until:
        sim.render()
        time.sleep(0.03)


def main():
    torch.set_num_threads(2)
    torch.manual_seed(20261104)
    cfg = json.loads(Path(__file__).with_name("scene.json").read_text())
    poses = json.loads((Path(a.assets) / "poses.json").read_text())
    task = specification(cfg, 32, list(range(32)), 0, 0)
    ppo = contract(cfg, 2, 2, 32)
    ppo["seed"] = 20261104  # Separate demonstration, not a replacement study.
    ppo["fingerprint"] = hashlib.sha256(
        json.dumps(
            {k: v for k, v in ppo.items() if k != "fingerprint"}, sort_keys=True
        ).encode()
    ).hexdigest()
    sim = sim_utils.SimulationContext(
        sim_utils.SimulationCfg(
            dt=cfg["simulation"]["physics_dt_s"],
            render_interval=8,
            device=a.device,
            physx=sim_utils.PhysxCfg(enable_ccd=False),
        )
    )
    bucket_asset(a.output / "bucket.usda", cfg["bucket"])
    scfg, filters = scene_config(
        cfg, poses, a.assets, a.output / "bucket.usda", 32, robot_body_filters(a.assets)
    )
    scene = InteractiveScene(scfg)
    perspective = a.view == "perspective"
    view = Camera(
        CameraCfg(
            prim_path="/World/FarmViewport",
            width=1920,
            height=1440,
            update_period=0,
            data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=24 if perspective else 20,
                horizontal_aperture=36 if perspective else 400,
                vertical_aperture=20.25 if perspective else 300,
                clipping_range=(0.05, 200),
            ),
        )
    )
    camera = UsdGeom.Camera.Get(sim.stage, "/World/FarmViewport")
    camera.GetProjectionAttr().Set(
        UsdGeom.Tokens.perspective if perspective else UsdGeom.Tokens.orthographic
    )
    sim.reset()
    env = LiveShots(sim, scene, None, None, cfg, poses, a, task, filters, a.output)
    center = env.origins.mean(axis=0) + [0.67, -0.25, 0.35]
    eye = center + ([18, -23, 20] if perspective else [0, -24, 48])
    view.set_world_poses_from_view(
        torch.tensor(np.array([eye]), dtype=torch.float32, device=sim.device),
        torch.tensor(np.array([center]), dtype=torch.float32, device=sim.device),
    )
    view.reset()
    viewport = omni.kit.viewport.utility.get_active_viewport()
    viewport.camera_path = "/World/FarmViewport"
    for name in (
        "Content",
        "Content Browser",
        "Console",
        "Property",
        "Semantics Schema Editor",
    ):
        window = ui.Workspace.get_window(name)
        if window:
            window.visible = False
    if perspective:
        for name in ("Stage", "Layer", "Render Settings"):
            window = ui.Workspace.get_window(name)
            if window:
                window.visible = False
    camera_metadata = dict(
        projection=a.view,
        eye_world_m=eye.tolist(),
        target_world_m=center.tolist(),
        focal_length=float(camera.GetFocalLengthAttr().Get()),
        horizontal_aperture=float(camera.GetHorizontalApertureAttr().Get()),
        vertical_aperture=float(camera.GetVerticalApertureAttr().Get()),
    )
    policy = Policy().to(sim.device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=ppo["lr"])
    report = dict(
        scope="Native Isaac desktop recording of 32-cell live PPO; endpoint resets, no pickup/transit learning.",
        seed=20261104,
        camera=camera_metadata,
        contract=ppo,
        workcells=32,
        arms=64,
        initial_weight_hash=weight_hash(policy),
        probes=env.probes(),
        updates=[],
    )
    x = torch.tensor(
        np.array([observation(c) for c in task["cells"]]), device=sim.device
    )
    with torch.no_grad():
        env.prepare(actions(policy.actor(x).cpu().numpy()))
    idle(sim, 2)
    atomic(
        a.output / "ready.json",
        dict(
            phase="ready",
            camera=str(viewport.camera_path),
            view=camera_metadata,
            workcells=32,
        ),
    )
    until = time.monotonic() + 300
    while not (a.output / "record").exists():
        if time.monotonic() > until or not app.is_running():
            raise RuntimeError("Recorder did not start within five minutes")
        idle(sim, 0.1)
    env.pace = True
    started = time.monotonic()
    report["started_utc"] = datetime.datetime.now(datetime.UTC).isoformat()
    idle(sim, 2)
    for iteration in range(1, 3):
        bx, bz, blp, br = [], [], [], []
        for batch in range(2):
            with torch.no_grad():
                distribution = policy.distribution(x)
                z = distribution.sample()
                lp = distribution.log_prob(z).sum(-1)
            rows, _ = env.shot_batch(actions(z.cpu().numpy()))
            bx.append(x)
            bz.append(z)
            blp.append(lp)
            br.append(torch.tensor([reward(r) for r in rows], device=sim.device))
            atomic(
                a.output / f"batch-{iteration}-{batch}.json",
                dict(iteration=iteration, batch=batch, rows=rows),
            )
            print(
                json.dumps(dict(iteration=iteration, batch=batch, **statistics(rows))),
                flush=True,
            )
            idle(sim, 1)
        result = update(
            policy,
            optimizer,
            torch.cat(bx),
            torch.cat(bz),
            torch.cat(blp),
            torch.cat(br),
            ppo,
        )
        report["updates"].append(result)
    report.update(
        phase="completed",
        throws=128,
        wall_s=time.monotonic() - started,
        final_weight_hash=weight_hash(policy),
    )
    if report["initial_weight_hash"] == report["final_weight_hash"]:
        raise RuntimeError("No policy update occurred")
    torch.save(
        dict(state_dict=policy.state_dict(), contract=ppo), a.output / "policy.pt"
    )
    atomic(a.output / "result.json", report)
    idle(sim, 5)


try:
    main()
except BaseException:
    atomic(a.output / "failure.json", dict(error=traceback.format_exc()))
    traceback.print_exc()
    os._exit(1)
app.close(skip_cleanup=True)
