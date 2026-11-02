"""Replay measured training actions on a compact cinema stage with driven arm transitions."""

import hashlib
import json
import os
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import torch
from board_route import FORCE_THRESHOLD_N
from recording_plans import recording_plan
from dark_stage import stage_settings
from dark_export import inside_frame
from feasibility_runtime import FeasibilityShots
from parallel_runtime import ParallelShots, atomic

ACTION_FIELDS = (
    "id",
    "azimuth_deg",
    "azimuth_offset_deg",
    "board_x_m",
    "board_y_m",
    "height_m",
    "gap_m",
    "tilt_deg",
)


class ReplayShots:
    # FilmShots supplies the native-camera readback/encoding and rim ray probes.
    phase = None
    next_frame = 0

    def prepare(self, actions):
        return FeasibilityShots.prepare(self, actions)

    def approach_endpoints(self, targets):
        """Move from current measured joint angles with the live articulation drives."""
        starts = {k: arm.data.joint_pos.clone() for k, arm in self.arms.items()}
        self.set_balls(np.tile([0.1, -0.8, 0.04], (self.n, 1)))
        deltas = {k: end - starts[k] for k, end in targets.items()}
        longest = max(float(d.abs().max()) for d in deltas.values())
        # The peak speed of a cubic ease is 1.5 times its mean speed.
        # The earlier still-pose benchmark uses a 100 Nm uniform hold limit.
        # During transit use the source UF850 joint effort limits, then restore
        # the original properties before settlement, release and contact scoring.
        old_drives = {}
        for name, arm in self.arms.items():
            ids = [arm.joint_names.index(f"joint{i+1}") for i in range(6)]
            old_drives[name] = (
                ids,
                arm.data.joint_effort_limits[:, ids].clone(),
                arm.data.joint_stiffness[:, ids].clone(),
                arm.data.joint_damping[:, ids].clone(),
            )
            limits = self.transit_limits.expand(self.n, 6)
            arm.write_joint_effort_limit_to_sim(limits, ids)
            arm.write_joint_stiffness_to_sim(6000.0, ids)
            arm.write_joint_damping_to_sim(180.0, ids)
        steps = max(240, int(np.ceil(1.5 * longest / (0.85 * self.dt))))
        if self.focus is not None:
            visible = torch.cat([d[self.focus, :6] for d in deltas.values()])
            norm = float(torch.linalg.vector_norm(visible))
        else:
            norm = 0
        thresholds = np.linspace(0.03, 0.995, self.shot.move_frames)
        self.drive_ticks = 0
        self.latest_focus_error_deg = None
        previous_frame = self.steps
        for step in range(1, steps + round(85 / self.dt) + 1):
            if step <= steps:
                t = step / steps
                t = t * t * (3 - 2 * t)
                for name, arm in self.arms.items():
                    arm.set_joint_position_target(starts[name] + t * deltas[name])
            self.tick()
            if step % 4:
                continue
            if self.focus is not None:
                diff = torch.cat(
                    [
                        (arm.data.joint_pos[self.focus] - targets[name][self.focus])[:6]
                        for name, arm in self.arms.items()
                    ]
                )
                self.latest_focus_error_deg = float(torch.rad2deg(diff.abs().max()))
                progress = (
                    1 - float(torch.linalg.vector_norm(diff)) / norm
                    if norm > 1e-5
                    else min(1, step / steps)
                )
                k = len(self.frames)
                if k < len(thresholds) and progress >= thresholds[k]:
                    self.sample("move", self.steps - previous_frame)
                    previous_frame = self.steps
            if step < steps or step % 120 or len(self.frames) < self.shot.move_frames:
                continue
            # Some original proposals were outside the joint workspace. Do not
            # wait on them to authorize a previously reachable shot.
            ready = torch.ones(self.n, dtype=torch.bool, device=self.sim.device)
            for name, arm in self.arms.items():
                err = (
                    (arm.data.joint_pos[:, :6] - targets[name][:, :6]).abs().amax(dim=1)
                )
                ready &= err < (
                    0.001
                    if getattr(self.args, "recording_plan", None) == "fixed-station-v1"
                    else 0.00065
                )
            supported = ready[self.source_valid]
            foreground_ready = self.focus is None or bool(ready[self.focus])
            if getattr(self.args, "recording_plan", None) == "fixed-station-v1":
                eye, aim = self.view_pose(
                    self.shot, 0, self.spec["cells"], self.origins
                )
                frame = dict(eye_m=eye, target_m=aim)
                prospective = self.arms["Arm10"].data.body_pos_w[
                    :, self.flange_idx
                ].cpu().numpy() - [0, 0, 0.095]
                visible_ready = ready.cpu().numpy() & self.source_valid.cpu().numpy()
                visible_ready[self.focus] = False
                background_ready = sum(
                    ok and inside_frame(frame, point)
                    for ok, point in zip(visible_ready, prospective)
                )
                if foreground_ready and background_ready >= 2:
                    break
            elif foreground_ready and int(supported.sum()) >= min(12, len(supported)):
                break
        self.drive_ticks = step
        for name, arm in self.arms.items():
            ids, caps, stiffness, damping = old_drives[name]
            arm.write_joint_effort_limit_to_sim(caps, ids)
            arm.write_joint_stiffness_to_sim(stiffness, ids)
            arm.write_joint_damping_to_sim(damping, ids)
        if len(self.frames) != self.shot.move_frames:
            raise RuntimeError(
                f"Drive failed to reach filmed pose in {self.shot.name} "
                f"(joint error {self.latest_focus_error_deg} degrees, {len(self.frames)} motion frames)"
            )
        self.assert_buckets_fixed()

    def render(self):
        if self.record_video and self.phase == "ready":
            self.phase = "release"
            self.release_step = self.steps
            self.sample("release", self.shot.stride)

    def sample(self, phase, elapsed):
        k = len(self.frames)
        ball = None
        if phase == "release" and self.focus is not None:
            ball = self.ball.data.root_pos_w[self.focus].cpu().numpy()
        eye, aim = self.view_pose(self.shot, k, self.spec["cells"], self.origins, ball)
        self.write_rgb(eye, aim, elapsed)
        frame = self.frames[-1]
        frame["phase"] = phase
        if phase == "move" and self.focus is not None:
            frame["focus_joints_rad"] = {
                name: arm.data.joint_pos[self.focus, :6].cpu().tolist()
                for name, arm in self.arms.items()
            }

    def tick(self):
        ParallelShots.tick(self)
        if self.phase != "release" or len(self.frames) >= self.shot.frames:
            return
        if self.focus is not None:
            forces = self.forces()[self.focus]
            for j, name in enumerate(("board", "bucket", "table", "floor")):
                if self.contacts[name] is None and forces[j] > FORCE_THRESHOLD_N:
                    self.contacts[name] = dict(
                        physics_step=self.steps,
                        force_N=float(forces[j]),
                        ball_world_m=self.ball.data.root_pos_w[self.focus]
                        .cpu()
                        .tolist(),
                    )
        if (self.steps - self.release_step) % self.shot.stride == 0:
            self.sample("release", self.shot.stride)

    def replay(self, shot, actions):
        self.shot, self.frames, self.focus = shot, [], shot.focus
        self.contacts = dict.fromkeys(("board", "bucket", "table", "floor"))
        self.phase = "ready"
        rows, _ = self.shot_batch(actions, record_video=True)
        if self.focus is not None and rows[self.focus]["invalid"]:
            raise RuntimeError(
                f"Focused release refused in {shot.name}: {rows[self.focus]}"
            )
        while len(self.frames) < shot.frames:
            self.tick()
        self.phase = None
        return rows


def resume_prefix(env, previous, shots, plan_name):
    """Restore a completed camera-cut prefix; preserve its original RGB and logs."""
    progress = json.loads((previous / "progress.json").read_text())
    if progress.get("recording_plan", "dark-tour-v1") != plan_name:
        raise ValueError("Resume requires the same recording plan")
    chapters = progress["chapters"]
    count = len(chapters)
    if not 0 < count < len(shots) or not all(
        json.loads((previous / "probes.json").read_text())[i]["all_passed"]
        for i in range(6)
    ):
        raise ValueError("Resume requires a verified, incomplete recording prefix")
    manifest = {}
    frame_hash = hashlib.sha256()
    for i, c in enumerate(chapters):
        plan = shots[i]
        if (c["shot"], c["focus"], c["source_batch"]) != (
            plan.name,
            env.spec["cells"][plan.focus]["id"] if plan.focus is not None else None,
            plan.source_batch,
        ):
            raise ValueError("Resume prefix disagrees with the camera plan")
        path = previous / f"clip-{i:02}.json"
        data = path.read_bytes()
        clip = json.loads(data)
        if clip["chapter"] != c or len(clip["frames"]) != plan.frames:
            raise ValueError("Incomplete resume clip")
        (env.dest / path.name).write_bytes(data)
        manifest[path.name] = hashlib.sha256(data).hexdigest()
        for frame in clip["frames"]:
            filename = f"{frame['frame']:05d}.jpg"
            src, dest = previous / "frames" / filename, env.frames_path / filename
            frame_hash.update(filename.encode() + src.read_bytes())
            try:
                os.link(src, dest)
            except OSError:
                shutil.copyfile(src, dest)
    last = json.loads((previous / f"clip-{count-1:02}.json").read_text())
    for name, arm in env.arms.items():
        q = torch.tensor(
            [r["final_joint_positions_rad"][name] for r in last["rollout"]],
            device=env.sim.device,
        )
        arm.write_joint_state_to_sim(q, torch.zeros_like(q))
        arm.set_joint_position_target(q)
        arm.reset()
    env.next_frame = chapters[-1]["start_frame"] + chapters[-1]["frames"]
    for _ in range(120):
        env.tick()
    info = dict(
        previous_run=previous.name,
        completed_chapters=count,
        clip_sha256=manifest,
        frames_sha256=frame_hash.hexdigest(),
        restored="Measured final joint positions, zero velocity at camera cut.",
    )
    atomic(env.dest / "resume.json", info)
    return chapters, info


def run_replay(env, source, spec):
    plan_name = getattr(env.args, "recording_plan", "dark-tour-v1")
    shots, env.view_pose = recording_plan(plan_name)
    parent = json.loads((source / "result.json").read_text())
    if parent["fingerprint"] != spec["fingerprint"] or parent["articulations"] != 70:
        raise ValueError("Replay source does not match the 35-table training contract")
    model = ET.parse(Path(env.args.assets) / "dropper.urdf")
    env.transit_limits = torch.tensor(
        [
            float(model.find(f".//joint[@name='joint{i+1}']/limit").get("effort"))
            for i in range(6)
        ],
        device=env.sim.device,
    )
    chapters, resumed = (
        resume_prefix(env, env.args.resume_filming, shots, plan_name)
        if env.args.resume_filming
        else ([], None)
    )
    skip = len(chapters)
    for i, shot in enumerate(shots):
        if i < skip:
            continue
        encoding = (source / f"clip-{shot.source_batch:02}.json").read_bytes()
        old = json.loads(encoding)
        records = old["rollout"]
        if [r["cell_id"] for r in records] != [c["id"] for c in spec["cells"]]:
            raise ValueError("Source bucket sites do not match the replay")
        actions = [{k: r[k] for k in ACTION_FIELDS} for r in records]
        env.source_valid = torch.tensor(
            [not r["invalid"] for r in records], device=env.sim.device
        )
        start = env.next_frame
        rows = env.replay(shot, actions)
        focus = rows[shot.focus] if shot.focus is not None else {}
        chapter = dict(
            shot=shot.name,
            title=shot.name.replace("-", " ").title(),
            start_frame=start,
            frames=len(env.frames),
            move_frames=shot.move_frames,
            stride=shot.stride,
            source_batch=shot.source_batch,
            focus=focus.get("cell_id"),
            captured=focus.get("route_success"),
            board_bounce=focus.get("board_bounce"),
            literal_floor=(env.contacts["floor"] is not None) if focus else None,
            action_only=bool(focus),
            station=shot.name.startswith("station-"),
        )
        excursions = {}
        for name in env.arms:
            samples = [
                f["focus_joints_rad"][name]
                for f in env.frames
                if "focus_joints_rad" in f
            ]
            if samples:
                excursions[name] = float(np.rad2deg(np.ptp(samples, axis=0)).max())
        atomic(
            env.dest / f"clip-{i:02}.json",
            dict(
                chapter=chapter,
                rollout=rows,
                frames=env.frames,
                focus_contacts=env.contacts,
                motion=dict(
                    max_joint_excursion_deg=excursions,
                    drive_ticks=env.drive_ticks,
                    final_focus_error_deg=env.latest_focus_error_deg,
                ),
                source_clip_sha256=hashlib.sha256(encoding).hexdigest(),
                action_source_batches=[shot.source_batch] * env.n,
            ),
        )
        chapters.append(chapter)
        atomic(
            env.dest / "progress.json",
            dict(chapters=chapters, frames=env.next_frame, recording_plan=plan_name),
        )
    atomic(
        env.dest / "result.json",
        dict(
            version="sircus-dark-farm-replay-v1",
            recording_plan=plan_name,
            source_fingerprint=parent["fingerprint"],
            source_weights_sha256=parent["final_hash"],
            stage=getattr(env, "stage_metadata", stage_settings()),
            appearance=getattr(env.args, "appearance", "dark"),
            origins_m=env.origins.tolist(),
            physics_dt_s=env.dt,
            workcells=env.n,
            articulations=2 * env.n,
            fps=30,
            resolution=[1920, 1080],
            frames=env.next_frame,
            chapters=chapters,
            gpu=torch.cuda.get_device_name(),
            mode="Previously sampled PPO actions replayed with articulated joint drives; no optimizer updates.",
            resumed=resumed,
            source_sha256={
                p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                for p in sorted(Path(__file__).parent.glob("*.py"))
            },
        ),
    )
