"""32-cell cooperative PPO in Isaac Lab/PhysX, with actual training-rollout video."""

import argparse
import copy
import datetime
import hashlib
import json
import os
import shutil
import time
import traceback
from pathlib import Path

from isaaclab.app import AppLauncher
from farm_contract import (
    FIXED,
    SEED,
    actions,
    contract,
    observation,
    reward,
    splits,
    statistics,
)

p = argparse.ArgumentParser()
p.add_argument("--assets", default="/workspace/assets")
p.add_argument("--output", type=Path, default=Path("/workspace/output"))
p.add_argument("--updates", type=int, default=16)
p.add_argument("--batches-per-update", type=int, default=4)
p.add_argument("--smoke", action="store_true")
AppLauncher.add_app_launcher_args(p)
a = p.parse_args()
if not a.enable_cameras:
    p.error("--enable_cameras is required")
if a.smoke:
    a.updates, a.batches_per_update = 1, 1
n = 4 if a.smoke else 32
a.output.mkdir(parents=True, exist_ok=True)
if shutil.disk_usage(a.output).free < 3 * 1024**3:
    p.error("Less than 3 GiB free")
stamp = datetime.datetime.now(datetime.UTC).strftime("%Y%m%dT%H%M%SZ")
dest = a.output / f"farm-ppo-{stamp}-n{n}"
dest.mkdir()
app = AppLauncher(a).app

import isaaclab.sim as sim_utils
import numpy as np
import torch
from farm_policy import Policy, update, weight_hash
from isaaclab.scene import InteractiveScene
from isaaclab.sensors import Camera, CameraCfg
from parallel_assets import bucket_asset, robot_body_filters, scene_config
from parallel_contract import specification
from parallel_runtime import ParallelShots, atomic
from PIL import Image
from pxr import UsdGeom
from scipy.spatial.transform import Rotation


class FarmShots(ParallelShots):
    def __init__(self, *args, views, **kwargs):
        self.views = views
        self.clip = None
        self.frame_start = None
        self.last_frame = None
        self.clip_samples = []
        super().__init__(*args, **kwargs)

    def layouts(self, cells):
        self.spec["cells"] = copy.deepcopy(cells)
        self.bucket_xy = np.array([c["bucket_xy_m"] for c in cells])
        root = self.bucket.data.default_root_state.clone()
        root[:, :3] = self.origins_t + torch.tensor(
            np.column_stack([self.bucket_xy, np.zeros(self.n)]), device=self.sim.device
        )
        root[:, 7:] = 0
        self.bucket.write_root_state_to_sim(root)
        self.bucket.reset()
        for _ in range(8):
            self.tick()
        self.assert_buckets_fixed()

    def start_clip(self, meta):
        meta["cell_origins_m"] = self.origins.tolist()
        self.clip = dest / "clips" / f"iteration-{meta['iteration']:03d}"
        self.clip.mkdir(parents=True)
        self.frame_start, self.last_frame = None, None
        self.clip_samples = []
        self.video_frames = 0
        atomic(self.clip / "clip.json", meta)

    def tick(self):
        super().tick()
        if self.record_video and (self.steps - self.frame_start) % 4 == 0:
            self.render()

    def render(self):
        if not self.record_video:
            return
        if self.frame_start is None:
            self.frame_start = self.steps
        if self.last_frame == self.steps or (self.steps - self.frame_start) % 4:
            return
        for _ in range(3):
            self.sim.render()
        sample = dict(
            step=self.steps,
            time_s=(self.steps - self.frame_start) * self.dt,
            positions_m=self.ball.data.root_pos_w.cpu().numpy().tolist(),
            cameras={},
        )
        for spec, camera in self.views:
            camera.update(4 * self.dt, force_recompute=True)
            rgb = camera.data.output["rgb"][0, :, :, :3].cpu().numpy().astype(np.uint8)
            Image.fromarray(rgb).save(
                self.clip / f"{spec['name']}-{self.video_frames:04d}.jpg", quality=92
            )
            q = camera.data.quat_w_ros[0].cpu().numpy()
            sample["cameras"][spec["name"]] = dict(
                position_m=camera.data.pos_w[0].cpu().numpy().tolist(),
                rotation_ros=Rotation.from_quat(q[[1, 2, 3, 0]]).as_matrix().tolist(),
            )
        self.clip_samples.append(sample)
        self.last_frame = self.steps
        self.video_frames += 1


def main():
    torch.set_num_threads(2)
    torch.manual_seed(SEED)
    cfg = json.loads(Path(__file__).with_name("scene.json").read_text())
    spec = contract(cfg, a.updates, a.batches_per_update, n)
    spec["source_sha256"] = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in Path(__file__).parent.glob("*.py")
    }
    spec["gpu"] = torch.cuda.get_device_name(0)
    atomic(dest / "contract.json", spec)
    pools = splits(n)
    atomic(dest / "splits.json", pools)
    poses = json.loads((Path(a.assets) / "poses.json").read_text())
    sim = sim_utils.SimulationContext(
        sim_utils.SimulationCfg(
            dt=cfg["simulation"]["physics_dt_s"],
            render_interval=4,
            device=a.device,
            physx=sim_utils.PhysxCfg(enable_ccd=False),
        )
    )
    bucket_asset(dest / "bucket.usda", cfg["bucket"])
    scfg, filters = scene_config(
        cfg, poses, a.assets, dest / "bucket.usda", n, robot_body_filters(a.assets)
    )
    scene = InteractiveScene(scfg)
    origins = scene.env_origins.cpu().numpy()
    center = origins.mean(axis=0) + [0.67, -0.25, 0.35]
    span = max(float(np.ptp(origins[:, 0])), float(np.ptp(origins[:, 1]))) + 9.5
    viewspec = [
        dict(
            name="farm",
            cell=None,
            width=1920,
            height=1440,
            span_m=span,
            eye=(center + [0, -0.6 * span, 1.2 * span]).tolist(),
            look=center.tolist(),
            far=3 * span,
        )
    ]
    for name, index in [("near", 0), ("middle", min(13, n - 1)), ("far", n - 1)]:
        eye = origins[index] + [0.67, -2.3, 3.5]
        look = origins[index] + [0.67, -0.20, 0.35]
        viewspec.append(
            dict(
                name=name,
                cell=index,
                width=640,
                height=400,
                span_m=2.1,
                eye=eye.tolist(),
                look=look.tolist(),
                far=float(np.linalg.norm(eye - look)) + 1,
            )
        )
    views = []
    for c in viewspec:
        path = "/World/Training_" + c["name"]
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
                    clipping_range=(0.05, c["far"]),
                ),
            )
        )
        usd = UsdGeom.Camera.Get(sim.stage, path)
        usd.GetProjectionAttr().Set(UsdGeom.Tokens.orthographic)
        assert str(usd.GetProjectionAttr().Get()) == "orthographic"
        views.append((c, camera))
    sim.reset()
    for c, camera in views:
        camera.set_world_poses_from_view(
            torch.tensor([c["eye"]], dtype=torch.float32, device=sim.device),
            torch.tensor([c["look"]], dtype=torch.float32, device=sim.device),
        )
        camera.reset()
    env = FarmShots(
        sim,
        scene,
        views[0][1],
        views[1][1],
        cfg,
        poses,
        a,
        specification(cfg, 32, list(range(n)), 0, 0),
        filters,
        dest,
        views=views,
    )
    for _ in range(12):
        sim.render()
    policy = Policy().to(sim.device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=spec["lr"])
    initial = copy.deepcopy(policy.state_dict())
    curves, training, validation, clips = [], [], [], []
    started = time.monotonic()
    report = dict(
        run=dest.name,
        phase="validating",
        contract=spec,
        training_throws=0,
        budget=spec["training_throws"],
        updates_completed=0,
        learning_curve=curves,
        validation=validation,
        evaluation={},
        clips=clips,
        camera_specs=viewspec,
        scope=spec["scope"],
    )

    def publish(phase):
        report.update(
            phase=phase,
            elapsed_s=time.monotonic() - started,
            training_throws=len(training),
            timestamp_utc=datetime.datetime.now(datetime.UTC).isoformat(),
            max_bucket_drift_m=env.peak_bucket_drift,
        )
        atomic(dest / "progress.json", report)
        if not a.smoke:
            atomic(a.output / "farm-training.json", report)

    def save(name, iteration):
        torch.save(
            dict(
                state_dict=policy.state_dict(),
                optimizer=optimizer.state_dict(),
                contract=spec,
                updates=iteration,
                training_throws=len(training),
            ),
            dest / (name + ".tmp"),
        )
        (dest / (name + ".tmp")).replace(dest / name)

    def evaluate(label, pool, mode="policy"):
        env.layouts(pool)
        x = torch.tensor(np.array([observation(c) for c in pool]), device=sim.device)
        with torch.no_grad():
            chosen = actions(policy.actor(x).cpu().numpy())
        if mode == "fixed":
            chosen = [
                dict(id="fixed", height_m=FIXED[0], gap_m=FIXED[1], tilt_deg=FIXED[2])
                for _ in pool
            ]
        before = weight_hash(policy)
        rows, _ = env.shot_batch(chosen)
        if weight_hash(policy) != before:
            raise RuntimeError("Evaluation modified the policy")
        atomic(dest / (label + ".json"), dict(weight_hash=before, rows=rows))
        return statistics(rows)

    publish("contact_validation")
    report["probes"] = env.probes()
    save("initial.pt", 0)
    report["initial_weight_hash"] = weight_hash(policy)
    validation.append(
        dict(updates=0, **evaluate("validation-000", pools["validation"]))
    )
    best_key = (validation[-1]["captures"], validation[-1]["mean_reward"])
    best_state = copy.deepcopy(policy.state_dict())
    best_update = 0
    for iteration in range(1, a.updates + 1):
        if shutil.disk_usage(dest).free < 3 * 1024**3:
            raise RuntimeError("Free disk below 3 GiB")
        env.layouts(pools["train"])
        obs = torch.tensor(
            np.array([observation(c) for c in pools["train"]]), device=sim.device
        )
        batch_x, batch_z, batch_lp, batch_ret = [], [], [], []
        rows_update = []
        publish("training")
        for batch in range(a.batches_per_update):
            with torch.no_grad():
                distribution = policy.distribution(obs)
                z = distribution.sample()
                logp = distribution.log_prob(z).sum(-1)
            chosen = actions(z.cpu().numpy())
            record = batch == 0 and iteration in {1, max(1, a.updates // 2), a.updates}
            if record:
                save(f"rollout-policy-{iteration:03d}.pt", iteration - 1)
                env.start_clip(
                    dict(
                        run=dest.name,
                        iteration=iteration,
                        policy_updates_applied=iteration - 1,
                        throws_before=len(training),
                        total_updates=a.updates,
                        weight_hash=weight_hash(policy),
                        actions=chosen,
                        cells=pools["train"],
                        views=viewspec,
                        dt_s=env.dt,
                        capture_stride=4,
                        scope="Actual on-policy training rollout, later consumed by its PPO update.",
                    )
                )
            rows, raw = env.shot_batch(chosen, record_video=record)
            for j, r in enumerate(rows):
                r.update(
                    episode=len(training) + 1,
                    iteration=iteration,
                    batch=batch,
                    observation=obs[j].cpu().tolist(),
                    latent=z[j].cpu().tolist(),
                    old_logp=float(logp[j]),
                    ppo_reward=reward(r),
                )
                training.append(r)
                rows_update.append(r)
            with (dest / "episodes.jsonl").open("a") as f:
                f.write(
                    json.dumps(
                        dict(iteration=iteration, batch=batch, rows=rows),
                        allow_nan=False,
                    )
                    + "\n"
                )
            if record:
                atomic(env.clip / "samples.json", env.clip_samples)
                meta = json.loads((env.clip / "clip.json").read_text())
                meta.update(
                    rows=rows,
                    frames=env.video_frames,
                    statistics=statistics(rows),
                    samples_sha256=hashlib.sha256(
                        (env.clip / "samples.json").read_bytes()
                    ).hexdigest(),
                )
                atomic(env.clip / "clip.json", meta)
                clips.append(
                    dict(
                        path=str(env.clip.relative_to(dest)),
                        iteration=iteration,
                        frames=env.video_frames,
                        statistics=meta["statistics"],
                    )
                )
            batch_x.append(obs)
            batch_z.append(z)
            batch_lp.append(logp)
            batch_ret.append(
                torch.tensor([r["ppo_reward"] for r in rows], device=sim.device)
            )
            print(
                json.dumps(dict(iteration=iteration, batch=batch, **statistics(rows))),
                flush=True,
            )
        optimization = update(
            policy,
            optimizer,
            torch.cat(batch_x),
            torch.cat(batch_z),
            torch.cat(batch_lp),
            torch.cat(batch_ret),
            spec,
        )
        curve = dict(
            iteration=iteration,
            throws=len(training),
            **statistics(rows_update),
            optimization=optimization,
        )
        curves.append(curve)
        report["updates_completed"] = iteration
        save("latest.pt", iteration)
        atomic(dest / f"update-{iteration:03d}.json", curve)
        if iteration % 4 == 0 or iteration == a.updates:
            publish("validation")
            stats = evaluate(f"validation-{iteration:03d}", pools["validation"])
            validation.append(dict(updates=iteration, **stats))
            key = (stats["captures"], stats["mean_reward"])
            if key > best_key:
                best_key, best_update, best_state = (
                    key,
                    iteration,
                    copy.deepcopy(policy.state_dict()),
                )
        publish("training")
    final_state = copy.deepcopy(policy.state_dict())
    report["final_weight_hash"] = weight_hash(policy)
    report["actor_change_l1"] = float(
        sum(
            (v - initial[k]).abs().sum()
            for k, v in final_state.items()
            if k.startswith("actor.")
        )
    )
    if report["actor_change_l1"] <= 0:
        raise RuntimeError("No actor learning occurred")
    publish("frozen_test")
    for label, state in [
        ("initial", initial),
        ("final", final_state),
        ("selected", best_state),
    ]:
        policy.load_state_dict(state)
        report["evaluation"][label] = evaluate("test-" + label, pools["test"])
    report["evaluation"]["fixed"] = evaluate("test-fixed", pools["test"], mode="fixed")
    # Selected artifact is evaluation-only, without an unrelated final optimizer.
    torch.save(
        dict(
            state_dict=best_state,
            contract=spec,
            updates=best_update,
            training_budget_used=len(training),
        ),
        dest / "selected.pt",
    )
    report.update(
        selected_update=best_update,
        selected_sha256=hashlib.sha256((dest / "selected.pt").read_bytes()).hexdigest(),
        final_sha256=hashlib.sha256((dest / "latest.pt").read_bytes()).hexdigest(),
        episodes_sha256=hashlib.sha256(
            (dest / "episodes.jsonl").read_bytes()
        ).hexdigest(),
        splits_sha256=hashlib.sha256((dest / "splits.json").read_bytes()).hexdigest(),
    )
    publish("completed")
    print(
        json.dumps(
            dict(run=str(dest), phase="completed", evaluation=report["evaluation"])
        ),
        flush=True,
    )


try:
    main()
except BaseException:  # noqa: BLE001 -- preserve failure instead of Kit swallowing it.
    traceback.print_exc()
    atomic(dest / "failure.json", dict(error=traceback.format_exc()))
    os._exit(1)
app.close(skip_cleanup=True)
