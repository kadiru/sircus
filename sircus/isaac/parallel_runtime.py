"""Batched articulated PhysX shots with the existing per-shot scoring definitions."""

import json
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import torch
from board_route import BoardRoute
from entry_quality import EntryQuality
from PIL import Image
from prepare_assets import chain, matrix
from rl_contract import ShotScore
from robust_ik import solve
from scipy.spatial.transform import Rotation
from settling_contract import MAX_CORRECTIONS, accepted, correction, pose_errors


def atomic(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, allow_nan=False))
    temp.replace(path)


class ParallelShots:
    def __init__(
        self, sim, scene, camera, close_camera, cfg, poses, args, spec, filters, dest
    ):
        self.sim, self.scene = sim, scene
        self.camera, self.close_camera = camera, close_camera
        self.cfg, self.poses, self.args, self.spec = cfg, poses, args, spec
        self.filters, self.dest = filters, dest
        self.arms = {n: scene[n] for n in ("Arm10", "Arm14")}
        self.ball, self.bucket, self.contact = (
            scene["Ball"],
            scene["Bucket"],
            scene["contacts"],
        )
        self.n, self.dt, self.steps = (
            spec["num_envs"],
            cfg["simulation"]["physics_dt_s"],
            0,
        )
        self.origins = scene.env_origins.cpu().numpy().copy()
        self.origins_t = scene.env_origins.clone()
        self.links = chain(ET.parse(Path(args.assets) / "board_arm.urdf").getroot())
        self.ik_cache = {}
        self.board_idx = self.arms["Arm14"].body_names.index("impact_board")
        self.flange_idx = self.arms["Arm10"].body_names.index("link6")
        self.bucket_xy = np.array([c["bucket_xy_m"] for c in spec["cells"]])
        self.peak_bucket_drift = 0.0
        self.render_count = 0
        self.frames_path = dest / "frames"
        self.frames_path.mkdir()
        self.record_video = False
        self.video_frames = 0
        self.last_view = {}
        self._initialize_states()

    def _initialize_states(self):
        for arm in self.arms.values():
            root = arm.data.default_root_state.clone()
            root[:, :3] += self.origins_t
            arm.write_root_pose_to_sim(root[:, :7])
            arm.write_root_velocity_to_sim(root[:, 7:])
            arm.write_joint_state_to_sim(
                arm.data.default_joint_pos, arm.data.default_joint_vel
            )
            arm.set_joint_position_target(arm.data.default_joint_pos)
        for name in ("Table", "Floor", "Bucket", "Ball"):
            obj = self.scene[name]
            root = obj.data.default_root_state.clone()
            root[:, :3] += self.origins_t
            if name == "Bucket":
                root[:, :2] = self.origins_t[:, :2] + torch.tensor(
                    self.bucket_xy, device=self.sim.device
                )
            obj.write_root_state_to_sim(root)
        self.scene.reset()
        for _ in range(120):
            self.tick()
        for name, arm in self.arms.items():
            assert arm.num_instances == self.n, f"Wrong articulation count: {name}"
            expected = self.origins + (
                [0, 0, 0]
                if name == "Arm10"
                else [self.cfg["arms"]["base_spacing_m"], 0, 0]
            )
            assert (
                np.max(
                    np.linalg.norm(arm.data.root_pos_w.cpu().numpy() - expected, axis=1)
                )
                < 1e-5
            )
        assert self.contact.data.force_matrix_w.shape == (
            self.n,
            1,
            len(self.filters),
            3,
        ), "Contact-filter indexing mismatch"
        self.assert_buckets_fixed()

    def assert_buckets_fixed(self):
        observed = (self.bucket.data.root_pos_w - self.origins_t).cpu().numpy()
        target = np.column_stack([self.bucket_xy, np.zeros(self.n)])
        drift = float(np.linalg.norm(observed - target, axis=1).max())
        self.peak_bucket_drift = max(self.peak_bucket_drift, drift)
        if drift > 1e-5:
            raise RuntimeError(
                f"Bucket moved from its assigned cell position: {drift} m"
            )
        return drift

    def tick(self):
        for arm in self.arms.values():
            arm.set_joint_effort_target(
                arm.root_physx_view.get_gravity_compensation_forces()
            )
        self.scene.write_data_to_sim()
        self.sim.step(render=False)
        self.scene.update(self.dt)
        self.contact.update(0, force_recompute=True)
        self.steps += 1

    def set_balls(self, xyz, world=False):
        state = self.ball.data.default_root_state.clone()
        state[:, :3] = torch.tensor(np.array(xyz), device=self.sim.device)
        if not world:
            state[:, :3] += self.origins_t
        state[:, 7:] = 0
        self.ball.write_root_state_to_sim(state)
        self.ball.reset()
        self.contact.reset()

    def forces(self):
        f = self.contact.data.force_matrix_w[:, 0]
        net = self.contact.data.net_forces_w[:, 0]
        values = torch.stack(
            [torch.linalg.vector_norm(f[:, i], dim=-1) for i in range(4)]
            + [
                torch.linalg.vector_norm(f[:, 4:], dim=-1).sum(dim=-1),
                torch.linalg.vector_norm(net - f.sum(dim=1), dim=-1),
            ],
            dim=-1,
        )
        a = values.cpu().numpy()
        if not np.isfinite(a).all():
            raise RuntimeError("Nonfinite contact data")
        return a

    def probes(self):
        b = self.cfg["bucket"]
        top = b["assumed_interior_depth_m"] + b["assumed_bottom_thickness_m"]
        results = []
        names = ["board", "bucket", "table", "floor", "robot", "unattributed"]
        for kind in ("bucket", "rim", "table", "floor", "robot"):
            if kind in ("bucket", "rim"):
                xyz = np.column_stack([self.bucket_xy, np.full(self.n, top + 0.15)])
                if kind == "rim":
                    directions = getattr(self, "rim_probe_directions", (1.0, 0.0))
                    xyz[:, :2] += np.asarray(directions) * b["inner_top_diameter_m"] / 2
                    xyz[:, 2] = top + getattr(self, "rim_probe_height_m", 0.15)
            else:
                xyz = np.tile(
                    getattr(
                        self,
                        "probe_positions",
                        {
                            "table": [-0.05, -0.65, 0.25],
                            "floor": [1.8, -0.65, -0.5],
                            "robot": [0, 0, 0.75],
                        },
                    )[kind],
                    (self.n, 1),
                )
            self.set_balls(xyz)
            peaks = np.zeros((self.n, 6))
            for _ in range(240):
                self.tick()
                peaks = np.maximum(peaks, self.forces())
            target = names.index("bucket" if kind == "rim" else kind)
            passed = peaks[:, target] > 0.01
            if kind in ("table", "floor"):
                passed &= (peaks[:, 0] <= 0.01) & (peaks[:, 1] <= 0.01)
            results.append(
                {
                    "kind": kind,
                    "all_passed": bool(passed.all()),
                    "per_cell_passed": passed.tolist(),
                    "peak_N": peaks.tolist(),
                }
            )
            if not passed.all():
                atomic(self.dest / "failed-probes.json", results)
                raise RuntimeError(f"{kind} contact attribution failed in some cells")
        if self.n > 1:
            # Deliberately put env0's ball over env1's bucket. It must pass through
            # foreign bucket/table/floor, rather than report an env0 bucket capture.
            xyz = self.origins + np.tile([0.1, -0.8, 0.04], (self.n, 1))
            other = getattr(self, "foreign_probe_index", 1)
            xyz[0] = self.origins[other] + [*self.bucket_xy[other], top + 0.15]
            self.set_balls(xyz, world=True)
            peak = np.zeros(6)
            for _ in range(240):
                self.tick()
                peak = np.maximum(peak, self.forces()[0])
            z = float(self.ball.data.root_pos_w[0, 2])
            passed = bool(peak.max() <= 0.01 and z < -1)
            results.append(
                {
                    "kind": "foreign_cell_collision_filter",
                    "all_passed": passed,
                    "peak_N": peak.tolist(),
                    "foreign_ball_final_z_m": z,
                }
            )
            if not passed:
                raise RuntimeError("Foreign-cell contact isolation failed")
        self.assert_buckets_fixed()
        self.set_balls(np.tile([0.1, -0.8, 0.04], (self.n, 1)))
        return results

    def _ik(self, target, name, seed=None):
        key = (name, tuple(target.ravel()))
        if seed is None and key in self.ik_cache:
            return self.ik_cache[key]
        qkey = "board_joint_rad" if name == "Arm14" else "dropper_joint_rad"
        result = solve(
            target,
            self.links,
            self.poses[qkey] if seed is None else seed,
            name == "Arm14",
        )
        if seed is None:
            self.ik_cache[key] = result
        return result

    def prepare(self, actions):
        cfg, n = self.cfg, self.n
        targets, releases, drop_targets = [], [], []
        q = {
            name: arm.data.default_joint_pos.clone() for name, arm in self.arms.items()
        }
        invalid = [None] * n
        base = matrix([cfg["arms"]["base_spacing_m"], 0, 0], [0, 0, np.pi])
        mount = matrix(
            [0, 0, cfg["board"]["tool_offset_m"]], [0, 0, cfg["board"]["mount_yaw_rad"]]
        )
        for i, (a, cell) in enumerate(zip(actions, self.spec["cells"])):
            center = np.array([*cell["board_xy_m"], a["height_m"]])
            rot = (
                Rotation.from_euler("z", cell["heading_deg"], degrees=True).as_matrix()
                @ Rotation.from_euler("x", a["tilt_deg"], degrees=True).as_matrix()
                @ Rotation.from_euler("z", cfg["board"]["mount_yaw_rad"]).as_matrix()
            )
            target = np.eye(4)
            target[:3, :3] = rot
            target[:3, 3] = center
            targets.append(target)
            release = center + [0, 0, cfg["ball"]["radius_m"] + a["gap_m"]]
            releases.append(release)
            drop_target = matrix(release + [0, 0, 0.095], [np.pi, 0, 0])
            drop_targets.append(drop_target)
            for name, endpoint in [
                ("Arm14", np.linalg.inv(base) @ target @ np.linalg.inv(mount)),
                ("Arm10", drop_target),
            ]:
                r = self._ik(endpoint, name)
                if not r["solved"]:
                    invalid[i] = "endpoint_ik_" + name
                    continue
                for j in range(6):
                    q[name][i, self.arms[name].joint_names.index(f"joint{j + 1}")] = r[
                        "q"
                    ][j]
        self.approach_endpoints(q)
        self.set_balls(np.tile([0.1, -0.8, 0.04], (n, 1)))
        for _ in range(120):
            self.tick()
        commands = np.array(targets).copy()
        records = [[] for _ in range(n)]
        normals = np.zeros((n, 3))
        for iteration in range(MAX_CORRECTIONS + 1):
            prev, prev_norm = self.board_poses()
            for _ in range(60):
                self.tick()
            centers, normals = self.board_poses()
            speed = np.maximum(
                *[
                    a.data.joint_vel.abs().amax(dim=1).cpu().numpy()
                    for a in self.arms.values()
                ]
            )
            drop = (
                self.arms["Arm10"].data.body_pos_w[:, self.flange_idx].cpu().numpy()
                - self.origins
            )
            bad = []
            for i in range(n):
                m = dict(
                    pose_errors(targets[i], centers[i], normals[i]),
                    joint_speed_rad_s=float(speed[i]),
                    position_drift_mm=float(
                        np.linalg.norm(centers[i] - prev[i]) * 1000
                    ),
                    normal_drift_deg=float(
                        np.rad2deg(np.arccos(np.clip(normals[i] @ prev_norm[i], -1, 1)))
                    ),
                    dropper_position_error_mm=float(
                        np.linalg.norm(drop[i] - drop_targets[i][:3, 3]) * 1000
                    ),
                    iteration=iteration,
                )
                records[i].append(m)
                if invalid[i]:
                    continue
                if accepted(m) and m["dropper_position_error_mm"] <= 2:
                    continue
                bad.append(i)
            if not bad:
                break
            if iteration == MAX_CORRECTIONS:
                for i in bad:
                    invalid[i] = "endpoints_not_settled"
                break
            for i in bad:
                commands[i] = correction(
                    commands[i], targets[i], centers[i], normals[i]
                )
                endpoint = np.linalg.inv(base) @ commands[i] @ np.linalg.inv(mount)
                seed = q["Arm14"][i, :6].cpu().numpy()
                r = self._ik(endpoint, "Arm14", seed)
                if not r["solved"]:
                    invalid[i] = "settling_correction_ik"
                    continue
                for j in range(6):
                    q["Arm14"][
                        i, self.arms["Arm14"].joint_names.index(f"joint{j + 1}")
                    ] = r["q"][j]
            self.arms["Arm14"].set_joint_position_target(q["Arm14"])
            for _ in range(120):
                self.tick()
        self.assert_buckets_fixed()
        return np.array(releases), normals, records, invalid

    def approach_endpoints(self, q):
        for name, arm in self.arms.items():
            arm.write_joint_state_to_sim(q[name], torch.zeros_like(q[name]))
            arm.set_joint_position_target(q[name])
            arm.reset()

    def board_poses(self):
        a = self.arms["Arm14"]
        centers = a.data.body_pos_w[:, self.board_idx].cpu().numpy() - self.origins
        quats = a.data.body_quat_w[:, self.board_idx].cpu().numpy()
        normals = Rotation.from_quat(quats[:, [1, 2, 3, 0]]).as_matrix()[:, :, 2]
        return centers, normals

    def shot_batch(self, actions, record_video=False, enabled=None):
        started = time.monotonic()
        releases, normals, settled, invalid = self.prepare(actions)
        if enabled is not None:
            invalid = [
                reason if use else "no_selected_setting"
                for reason, use in zip(invalid, enabled)
            ]
        scores, qualities, routes = [], [], []
        for c in self.spec["cells"]:
            b = dict(self.cfg["bucket"], center_xy_m=c["bucket_xy_m"])
            scores.append(ShotScore(b, self.cfg["ball"]["radius_m"]))
            qualities.append(EntryQuality(b, self.cfg["ball"]["radius_m"]))
            routes.append(BoardRoute(self.dt))
        xyz = releases.copy()
        active = np.array([v is None for v in invalid])
        xyz[~active] = [0.1, -0.8, 0.04]
        self.set_balls(xyz)
        self.record_video = record_video
        self.render()
        trails = [[] for _ in range(self.n)]
        force_names = ["board", "bucket", "table", "floor", "robot", "unattributed"]
        for step in range(960):
            if not active.any():
                break
            self.tick()
            positions = (self.ball.data.root_pos_w - self.origins_t).cpu().numpy()
            velocities = self.ball.data.root_lin_vel_w.cpu().numpy()
            forces = self.forces()
            if not np.isfinite(positions).all():
                raise RuntimeError("Nonfinite ball states")
            for i in np.flatnonzero(active):
                p, s = positions[i], scores[i]
                s.step(
                    p, float(forces[i, 0]), self.dt, float(velocities[i] @ normals[i])
                )
                qualities[i].step(p, float(forces[i, 1]), s.entered)
                routes[i].step(dict(zip(force_names, map(float, forces[i]))))
                if step % 8 == 0:
                    trails[i].append(p.tolist())
                if s.success or p[2] < -0.2:
                    active[i] = False
            if record_video and step % 8 == 0:
                self.render()
        self.record_video = False
        self.render()
        self.assert_buckets_fixed()
        rows = []
        for i, a in enumerate(actions):
            result = scores[i].result()
            q = qualities[i].result(result["success"])
            route = routes[i].result(result["success"], q["quality_score"])
            rows.append(
                dict(
                    a,
                    **result,
                    **q,
                    **route,
                    cell_id=self.spec["cells"][i]["id"],
                    invalid=invalid[i],
                    settled=settled[i][-1],
                    release_ball_center_m=releases[i].tolist(),
                )
            )
        return rows, {
            "wall_s": time.monotonic() - started,
            "trajectory_m": trails,
            "settling": settled,
        }

    def render(self):
        # Flush temporal render history after endpoint resets / headless physics.
        for _ in range(3):
            self.sim.render()
        labels = []
        for camera, name in [(self.camera, "overview"), (self.close_camera, "close")]:
            camera.update(self.dt * 8, force_recompute=True)
            rgb = camera.data.output["rgb"][0, :, :, :3].cpu().numpy().astype(np.uint8)
            temp = self.dest / f"{name}.tmp.jpg"
            Image.fromarray(rgb).save(temp, quality=88)
            temp.replace(self.dest / f"{name}.jpg")
            if self.record_video:
                Image.fromarray(rgb).save(
                    self.frames_path / f"{name}-{self.video_frames:04d}.jpg", quality=88
                )
            if name == "overview":
                pos = camera.data.pos_w[0].cpu().numpy()
                quat = camera.data.quat_w_ros[0].cpu().numpy()
                k = camera.data.intrinsic_matrices[0].cpu().numpy()
                r = Rotation.from_quat(quat[[1, 2, 3, 0]]).as_matrix()
                for cell, origin in zip(self.spec["cells"], self.origins):
                    p = r.T @ (origin + [0.67, -0.25, 0.65] - pos)
                    uv = k @ p
                    labels.append(
                        {
                            "id": cell["id"],
                            "x": float(uv[0] / uv[2] / rgb.shape[1]),
                            "y": float(uv[1] / uv[2] / rgb.shape[0]),
                        }
                    )
        if self.record_video:
            self.video_frames += 1
        self.render_count += 1
        self.last_view = {
            "render_frame": self.render_count,
            "physics_step": self.steps,
            "labels": labels,
            "cell_origins_m": self.origins.tolist(),
        }
