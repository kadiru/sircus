"""Batched endpoint IK for simulated shot configuration, never trajectory planning.

URDF transforms are read once. The solver accepts only finite joint-limit-valid
endpoints at the original 0.1 mm / normal-weighted residual tolerance. A failure
means unresolved by this finite solver, not proof that the target is impossible.
Torch is imported lazily so geometry helpers remain usable on a small CPU install.
"""

import math
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation

VERSION = "sircus-batched-endpoint-ik-v1"
TOLERANCE = 1e-4


def _origin(node):
    result = np.eye(4)
    if node is not None:
        result[:3, 3] = np.fromstring(node.get("xyz", "0 0 0"), sep=" ")
        result[:3, :3] = Rotation.from_euler(
            "xyz", np.fromstring(node.get("rpy", "0 0 0"), sep=" ")
        ).as_matrix()
    return result


def read_chain(path):
    """Return six UF850 link origin, axis and limit arrays from a generated URDF."""
    root = ET.parse(path).getroot()
    origins, axes, lo, hi = [], [], [], []
    for i in range(1, 7):
        joint = root.find(f"joint[@name='joint{i}']")
        if joint is None or joint.get("type") != "revolute":
            raise ValueError(f"Missing revolute joint{i}")
        origins.append(_origin(joint.find("origin")))
        axis = np.fromstring(joint.find("axis").get("xyz"), sep=" ")
        if len(axis) != 3 or abs(np.linalg.norm(axis) - 1) > 1e-8:
            raise ValueError("Expected unit joint axis")
        axes.append(axis)
        limit = joint.find("limit")
        lo.append(float(limit.get("lower")))
        hi.append(float(limit.get("upper")))
    if not np.all(np.asarray(hi) > lo):
        raise ValueError("Invalid joint limits")
    return np.asarray(origins), np.asarray(axes), np.asarray(lo), np.asarray(hi)


def shot_targets(cfg, bucket_xy, board_xyz, tilt_deg, gap_m):
    """Build exact nominal world board/flange targets and release ball centers.

    Tilt is from horizontal. The board's +Z normal tilts towards the bucket;
    landscape spin is the shared mount yaw. Gap is vertical ball-bottom clearance
    above the board's *center plane*, matching historical shot definitions.
    """
    bucket_xy = np.asarray(bucket_xy, dtype=float)
    board_xyz = np.asarray(board_xyz, dtype=float)
    if board_xyz.ndim != 2 or board_xyz.shape[1] != 3:
        raise ValueError("Expected board_xyz with shape [N,3]")
    n = len(board_xyz)
    if bucket_xy.shape != (n, 2):
        raise ValueError("Expected bucket_xy with shape [N,2]")
    tilt = np.broadcast_to(np.asarray(tilt_deg, dtype=float), (n,))
    gap = np.broadcast_to(np.asarray(gap_m, dtype=float), (n,))
    delta = bucket_xy - board_xyz[:, :2]
    if (
        not all(np.isfinite(x).all() for x in (bucket_xy, board_xyz, tilt, gap))
        or np.any(np.linalg.norm(delta, axis=1) < 1e-8)
        or np.any(tilt < 0)
        or np.any(tilt >= 90)
        or np.any(gap <= 0)
    ):
        raise ValueError(
            "Finite positions, separated XY, 0<=tilt<90 and gap>0 required"
        )
    heading = np.arctan2(delta[:, 1], delta[:, 0]) + math.pi / 2
    world = np.broadcast_to(np.eye(4), (n, 4, 4)).copy()
    rz = Rotation.from_euler("z", heading[:, None]).as_matrix()
    rx = Rotation.from_euler("x", tilt[:, None], degrees=True).as_matrix()
    mount_rotation = Rotation.from_euler("z", cfg["board"]["mount_yaw_rad"]).as_matrix()
    world[:, :3, :3] = rz @ rx @ mount_rotation
    world[:, :3, 3] = board_xyz
    mount = np.eye(4)
    mount[:3, :3] = mount_rotation
    mount[2, 3] = cfg["board"]["tool_offset_m"]
    base14 = np.diag([-1.0, -1.0, 1.0, 1.0])
    base14[0, 3] = cfg["arms"]["base_spacing_m"]
    target14 = np.linalg.inv(base14) @ world @ np.linalg.inv(mount)
    release = board_xyz.copy()
    release[:, 2] += cfg["ball"]["radius_m"] + gap
    target10 = np.broadcast_to(np.eye(4), (n, 4, 4)).copy()
    target10[:, :3, :3] = np.diag([1.0, -1.0, -1.0])
    target10[:, :3, 3] = release + [0, 0, 0.095]
    return {
        "board_world": world,
        "board_flange_local": target14,
        "dropper_flange_world": target10,
        "release_center_m": release,
        "board_normal_world": world[:, :3, 2],
    }


class BatchIK:
    """Vectorized damped least-squares with deterministic multistarts and gates."""

    def __init__(self, origins, axes, lower, upper, *, device="cpu", dtype=None):
        import torch

        self.torch = torch
        self.dtype = dtype or torch.float64
        self.device = torch.device(device)
        tensor = lambda x: torch.as_tensor(x, device=self.device, dtype=self.dtype)
        self.origins, self.axes = tensor(origins), tensor(axes)
        self.lower, self.upper = tensor(lower), tensor(upper)
        skew = torch.zeros((6, 3, 3), dtype=self.dtype, device=self.device)
        x, y, z = self.axes.unbind(-1)
        skew[:, 0, 1], skew[:, 0, 2] = -z, y
        skew[:, 1, 0], skew[:, 1, 2] = z, -x
        skew[:, 2, 0], skew[:, 2, 1] = -y, x
        self.skew, self.skew2 = skew, skew @ skew

    @classmethod
    def from_urdf(cls, path, **kwargs):
        return cls(*read_chain(path), **kwargs)

    def tensor(self, values):
        return self.torch.as_tensor(values, device=self.device, dtype=self.dtype)

    def _kinematics(self, q):
        t = self.torch
        pose = (
            t.eye(4, dtype=self.dtype, device=self.device)
            .expand(*q.shape[:-1], 4, 4)
            .clone()
        )
        positions, axes = [], []
        eye3 = t.eye(3, dtype=self.dtype, device=self.device)
        for j in range(6):
            pose = pose @ self.origins[j]
            positions.append(pose[..., :3, 3].clone())
            axes.append((pose[..., :3, :3] @ self.axes[j, :, None])[..., 0])
            theta = q[..., j, None, None]
            rotation = (
                eye3 + t.sin(theta) * self.skew[j] + (1 - t.cos(theta)) * self.skew2[j]
            )
            next_pose = pose.clone()
            next_pose[..., :3, :3] = pose[..., :3, :3] @ rotation
            pose = next_pose
        return pose, t.stack(positions, -2), t.stack(axes, -2)

    def fk(self, q):
        return self._kinematics(self.tensor(q))[0]

    def _rotvec(self, matrix):
        # Branch on the largest quaternion component. Unlike asin(skew), this
        # retains a meaningful error at rotations near 180 degrees.
        t = self.torch
        m00, m01, m02 = matrix[..., 0, :].unbind(-1)
        m10, m11, m12 = matrix[..., 1, :].unbind(-1)
        m20, m21, m22 = matrix[..., 2, :].unbind(-1)
        magnitudes = (
            t.stack(
                (
                    1 + m00 + m11 + m22,
                    1 + m00 - m11 - m22,
                    1 - m00 + m11 - m22,
                    1 - m00 - m11 + m22,
                ),
                -1,
            )
            .clamp_min(0)
            .sqrt()
        )
        candidates = (
            t.stack(
                (
                    t.stack(
                        (magnitudes[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01), -1
                    ),
                    t.stack(
                        (m21 - m12, magnitudes[..., 1] ** 2, m10 + m01, m02 + m20), -1
                    ),
                    t.stack(
                        (m02 - m20, m10 + m01, magnitudes[..., 2] ** 2, m12 + m21), -1
                    ),
                    t.stack(
                        (m10 - m01, m02 + m20, m12 + m21, magnitudes[..., 3] ** 2), -1
                    ),
                ),
                -2,
            )
            / (2 * magnitudes.clamp_min(0.1))[..., :, None]
        )
        chosen = magnitudes.argmax(-1)
        quat = candidates.gather(
            -2, chosen[..., None, None].expand(*chosen.shape, 1, 4)
        )[..., 0, :]
        quat = quat / t.linalg.vector_norm(quat, dim=-1, keepdim=True).clamp_min(1e-15)
        quat = t.where(quat[..., :1] < 0, -quat, quat)
        length = t.linalg.vector_norm(quat[..., 1:], dim=-1, keepdim=True)
        angle = 2 * t.atan2(length, quat[..., :1].clamp_min(0))
        return quat[..., 1:] * t.where(
            length > 1e-10, angle / length.clamp_min(1e-15), 2.0
        )

    def _error_jacobian(self, q, target, normal_only):
        t = self.torch
        pose, origins, axes = self._kinematics(q)
        position_error = target[..., :3, 3] - pose[..., :3, 3]
        jp = t.linalg.cross(axes, pose[..., None, :3, 3] - origins, dim=-1).transpose(
            -1, -2
        )
        if normal_only:
            angular_error = target[..., :3, 2] - pose[..., :3, 2]
            ja = t.linalg.cross(
                axes, pose[..., None, :3, 2].expand_as(axes), dim=-1
            ).transpose(-1, -2)
        else:
            angular_error = self._rotvec(
                target[..., :3, :3] @ pose[..., :3, :3].transpose(-1, -2)
            )
            ja = axes.transpose(-1, -2)
        return (
            t.cat((position_error, 0.25 * angular_error), -1),
            t.cat((jp, 0.25 * ja), -2),
            pose,
        )

    def solve(
        self,
        targets,
        initial=None,
        *,
        normal_only=False,
        starts=4,
        iterations=100,
        seed=20261102,
    ):
        t = self.torch
        target = self.tensor(targets)
        if (
            target.ndim != 3
            or not len(target)
            or target.shape[1:] != (4, 4)
            or not t.isfinite(target).all()
        ):
            raise ValueError("Expected finite targets [N,4,4]")
        rotation = target[:, :3, :3]
        identity = t.eye(3, device=self.device, dtype=self.dtype)
        if (
            not t.allclose(
                rotation @ rotation.transpose(-1, -2),
                identity.expand_as(rotation),
                atol=2e-5,
                rtol=0,
            )
            or not t.allclose(
                t.linalg.det(rotation),
                t.ones(len(target), device=self.device, dtype=self.dtype),
                atol=2e-5,
                rtol=0,
            )
            or not t.allclose(
                target[:, 3],
                self.tensor([0, 0, 0, 1]).expand(len(target), -1),
                atol=1e-8,
                rtol=0,
            )
        ):
            raise ValueError("Targets must contain proper homogeneous rigid transforms")
        if starts < 1 or iterations < 1:
            raise ValueError("Positive starts/iterations required")
        n = target.shape[0]
        if initial is None:
            initial = self.tensor([0, -0.5, -1.5, 0, 0.5, 0]).expand(n, -1)
        else:
            initial = self.tensor(initial).expand(n, 6)
        if not t.isfinite(initial).all():
            raise ValueError("Initial joints must be finite")
        generator = t.Generator(device=self.device).manual_seed(seed)
        q = self.lower + (self.upper - self.lower) * t.rand(
            (n, starts, 6), generator=generator, device=self.device, dtype=self.dtype
        )
        q[:, 0] = initial
        if starts > 1:
            q[:, 1] = (self.lower + self.upper) / 2
        if starts > 2:
            q[:, 2] = initial + self.tensor([0.1, -0.1, 0.1, 0, 0, 0])
        q = q.clamp(self.lower + 1e-7, self.upper - 1e-7).reshape(-1, 6)
        expanded = target[:, None].expand(n, starts, 4, 4).reshape(-1, 4, 4)
        best_q = q.clone()
        best_error = t.full(
            (n * starts,), float("inf"), dtype=self.dtype, device=self.device
        )
        damping = t.full_like(best_error, 0.01)
        eye = t.eye(6, dtype=self.dtype, device=self.device)
        with t.no_grad():
            for iteration in range(iterations):
                error, jac, _ = self._error_jacobian(q, expanded, normal_only)
                norm = t.linalg.vector_norm(error, dim=-1)
                improve = norm < best_error
                best_q = t.where(improve[:, None], q, best_q)
                best_error = t.minimum(norm, best_error)
                if iteration % 10 == 0 and bool(
                    (best_error.reshape(n, starts).min(-1).values < TOLERANCE).all()
                ):
                    break
                update = jac.transpose(-1, -2) @ t.linalg.solve(
                    jac @ jac.transpose(-1, -2) + damping[:, None, None] ** 2 * eye,
                    error[..., None],
                )
                update = update[..., 0]
                scale = (0.35 / update.abs().amax(-1).clamp_min(0.35)).unsqueeze(-1)
                trial = (q + update * scale).clamp(self.lower + 1e-7, self.upper - 1e-7)
                trial_error = self._error_jacobian(trial, expanded, normal_only)[0]
                accept = t.linalg.vector_norm(trial_error, dim=-1) < norm
                q = t.where(accept[:, None], trial, q)
                damping = t.where(accept, damping * 0.7, damping * 2).clamp(0.0001, 1)
            # Include the last update, even when the iteration budget ends there.
            error, _, _ = self._error_jacobian(q, expanded, normal_only)
            norm = t.linalg.vector_norm(error, dim=-1)
            best_q = t.where((norm < best_error)[:, None], q, best_q)
            best_error = t.minimum(norm, best_error)
            chosen = best_error.reshape(n, starts).argmin(-1)
            selected = best_q.reshape(n, starts, 6)[
                t.arange(n, device=self.device), chosen
            ]
            error, _, pose = self._error_jacobian(selected, target, normal_only)
            position = t.linalg.vector_norm(error[:, :3], dim=-1)
            cosine = (pose[:, :3, 2] * target[:, :3, 2]).sum(-1).clamp(-1, 1)
            orientation = (
                t.acos(cosine)
                if normal_only
                else t.linalg.vector_norm(error[:, 3:], dim=-1) / 0.25
            )
            residual = t.linalg.vector_norm(error, dim=-1)
            finite = t.isfinite(selected).all(-1) & t.isfinite(residual)
            valid = finite & (residual < TOLERANCE) & (position < TOLERANCE)
            valid &= (selected >= self.lower).all(-1) & (selected <= self.upper).all(-1)
        return dict(
            q=selected,
            valid=valid,
            position_error_m=position,
            orientation_error_rad=orientation,
            residual=residual,
            flange_local=pose,
            normal_only=normal_only,
            spin_policy=(
                "unconstrained_about_normal"
                if normal_only
                else "full_rotation_constrained"
            ),
            starts=starts,
            iterations=iteration + 1,
        )


class ShotIK:
    """Cached pair of arm solvers and shared world transform construction."""

    def __init__(self, cfg, assets, *, device="cpu", dtype=None):
        self.cfg = cfg
        self.dropper = BatchIK.from_urdf(
            Path(assets) / "dropper.urdf", device=device, dtype=dtype
        )
        self.board = BatchIK.from_urdf(
            Path(assets) / "board_arm.urdf", device=device, dtype=dtype
        )

    def solve(
        self,
        bucket_xy,
        board_xyz,
        tilt_deg,
        gap_m,
        *,
        initial10=None,
        initial14=None,
        normal_only=False,
        starts=4,
        iterations=100,
    ):
        targets = shot_targets(self.cfg, bucket_xy, board_xyz, tilt_deg, gap_m)
        board = self.board.solve(
            targets["board_flange_local"],
            initial14,
            normal_only=normal_only,
            starts=starts,
            iterations=iterations,
        )
        dropper = self.dropper.solve(
            targets["dropper_flange_world"],
            initial10,
            starts=starts,
            iterations=iterations,
        )
        return dict(
            targets=targets,
            board=board,
            dropper=dropper,
            valid=board["valid"] & dropper["valid"],
            q10=dropper["q"],
            q14=board["q"],
            version=VERSION,
        )


def solve_shots(cfg, assets, bucket_xy, board_xyz, tilt_deg, gap_m, **kwargs):
    """One-off convenience wrapper; use ShotIK for repeated training batches."""
    device, dtype = kwargs.pop("device", "cpu"), kwargs.pop("dtype", None)
    return ShotIK(cfg, assets, device=device, dtype=dtype).solve(
        bucket_xy, board_xyz, tilt_deg, gap_m, **kwargs
    )
