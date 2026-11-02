"""Query-stable IK branches for simulated shot endpoints.

The historical solver remains unchanged. This version reuses its kinematics,
Jacobian and 0.1 mm/weighted-orientation tolerance, but has target-hashed starts
and independent branch convergence. No random state or other query determines
the starts, stopping time, or branch ordering of a target. Collision selection
belongs to stable_geometry, which examines the returned branch pool.
"""

import hashlib
from pathlib import Path

import numpy as np

from .learning_ik import BatchIK, TOLERANCE, shot_targets

VERSION = "sircus-query-stable-endpoint-ik-v1"
SEED = 20261110
POLISH_TOLERANCE = 1e-8


def target_digest(target, *, role="arm", seed=SEED):
    """An exact target identity independent of batch index/order/global RNG."""
    a = np.asarray(target, dtype="<f8").copy()
    if a.shape != (4, 4) or not np.isfinite(a).all():
        raise ValueError("Expected one finite target matrix")
    a[a == 0] = 0  # Canonicalize signed zero, without rounding the request.
    payload = VERSION.encode() + role.encode() + str(int(seed)).encode() + a.tobytes()
    return hashlib.sha256(payload).hexdigest()


def normalize_known(value, count):
    """Finite six-joint rows or all-NaN absent rows; never partial hints."""
    if value is None:
        return np.full((count, 6), np.nan)
    a = np.asarray(value, dtype=float)
    if a.shape == (6,):
        a = np.broadcast_to(a, (count, 6)).copy()
    if a.shape != (count, 6) or not (
        np.isfinite(a).all(-1) | np.isnan(a).all(-1)
    ).all():
        raise ValueError("Known joints must be finite [N,6] or all-NaN rows")
    return a.copy()


class StableBatchIK(BatchIK):
    """Expose all deterministic converged branches, without residual-only choice."""

    def __init__(self, *args, role="arm", **kwargs):
        super().__init__(*args, **kwargs)
        self.role = str(role)

    def starts_for(self, targets, initial=None, *, known=None, starts=8, seed=SEED):
        if not 1 <= starts <= 32:
            raise ValueError("A finite 1..32 canonical/hash start budget is required")
        target = self.tensor(targets)
        if target.ndim != 3 or target.shape[1:] != (4, 4) or not len(target):
            raise ValueError("Expected finite targets [N,4,4]")
        a = target.detach().cpu().numpy()
        if not np.isfinite(a).all():
            raise ValueError("Targets must be finite")
        n = len(a)
        lo, hi = self.lower.cpu().numpy(), self.upper.cpu().numpy()
        if initial is None:
            initial = [0, -0.5, -1.5, 0, 0.5, 0]
        initial = np.broadcast_to(np.asarray(initial, dtype=float), (n, 6)).copy()
        if not np.isfinite(initial).all():
            raise ValueError("Initial joints must be finite")
        known = normalize_known(known, n)
        present = np.isfinite(known).all(-1)
        # A permanent optional-hint slot keeps canonical IDs stable whether or
        # not a caller supplies continuation. Absent hints cannot be selected.
        q = np.empty((n, starts + 1, 6))
        q[:, 0] = np.where(present[:, None], known, initial)
        q[:, 1] = initial
        if starts > 1:
            q[:, 2] = (lo + hi) / 2
        if starts > 2:
            q[:, 3] = initial + [0.1, -0.1, 0.1, 0, 0, 0]
        identities = []
        for i, matrix in enumerate(a):
            identity = target_digest(matrix, role=self.role, seed=seed)
            identities.append(identity)
            for j in range(4, starts + 1):
                # Per-target and per-start independent NumPy generators. This
                # neither consumes nor seeds a process-wide NumPy/Torch RNG.
                entropy = int(hashlib.sha256(f"{identity}:{j}".encode()).hexdigest(), 16)
                q[i, j] = np.random.default_rng(entropy).uniform(lo, hi)
        q[:, 1:] = np.clip(q[:, 1:], lo + 1e-7, hi - 1e-7)
        # Out-of-limit continuation may be used as a repaired seed, but cannot
        # be reported as preserved. In-limit hints are kept bit-for-bit.
        q[:, 0] = np.clip(q[:, 0], lo, hi)
        available = np.ones(q.shape[:2], dtype=bool)
        available[:, 0] = present
        return self.tensor(q), self.torch.as_tensor(available, device=self.device), identities

    def solve_candidates(
        self, targets, initial=None, *, known=None, normal_only=False,
        starts=8, iterations=120, seed=SEED, preserve_known=None,
    ):
        t = self.torch
        target = self.tensor(targets)
        q, available, identities = self.starts_for(
            target, initial, known=known, starts=starts, seed=seed
        )
        if not 1 <= iterations <= 2000:
            raise ValueError("A finite 1..2000 iteration budget is required")
        n, branches = q.shape[:2]
        if preserve_known is None:
            preserve_known = np.zeros(n, dtype=bool)
        preserve_known = np.asarray(preserve_known, dtype=bool)
        if preserve_known.shape != (n,):
            raise ValueError("preserve_known must have shape[N]")
        # Ordinary warm starts keep iterating below the historical acceptance
        # tolerance so even sub0.1mm requested perturbations are realized.
        # Only an explicitly source-query-matched continuation may retain an
        # already valid historical endpoint without polishing it.
        freeze_tolerance = t.full((n, branches), POLISH_TOLERANCE,
            device=self.device, dtype=self.dtype)
        freeze_tolerance[:, 0] = t.where(t.as_tensor(preserve_known,device=self.device),
            freeze_tolerance.new_full((), TOLERANCE), freeze_tolerance.new_full((), POLISH_TOLERANCE))
        freeze_tolerance = freeze_tolerance.reshape(-1)
        rotation = target[:, :3, :3]
        identity = t.eye(3, device=self.device, dtype=self.dtype)
        if not (
            t.allclose(rotation @ rotation.transpose(-1, -2), identity.expand_as(rotation), atol=2e-5, rtol=0)
            and t.allclose(t.linalg.det(rotation), t.ones(n, device=self.device, dtype=self.dtype), atol=2e-5, rtol=0)
            and t.allclose(target[:, 3], self.tensor([0, 0, 0, 1]).expand(n, -1), atol=1e-8, rtol=0)
        ):
            raise ValueError("Targets must be proper homogeneous rigid transforms")
        q = q.reshape(-1, 6)
        expanded = target[:, None].expand(n, branches, 4, 4).reshape(-1, 4, 4)
        best_q = q.clone()
        best_error = t.full((len(q),), float("inf"), device=self.device, dtype=self.dtype)
        damping = t.full_like(best_error, 0.01)
        frozen = t.zeros(len(q), dtype=t.bool, device=self.device)
        solved_at = t.full((len(q),), -1, dtype=t.int64, device=self.device)
        eye = t.eye(6, device=self.device, dtype=self.dtype)
        with t.no_grad():
            for iteration in range(iterations):
                error, jac, _ = self._error_jacobian(q, expanded, normal_only)
                norm = t.linalg.vector_norm(error, dim=-1)
                improve = (~frozen) & (norm < best_error)
                best_q = t.where(improve[:, None], q, best_q)
                best_error = t.minimum(best_error, norm)
                converged = (~frozen) & t.isfinite(norm) & (norm < freeze_tolerance)
                solved_at = t.where(converged, solved_at.new_full((), iteration), solved_at)
                frozen |= converged
                # Deliberately no global or batch-wide early termination.
                # Frozen rows receive zero updates and never change their q.
                update = (jac.transpose(-1, -2) @ t.linalg.solve(
                    jac @ jac.transpose(-1, -2) + damping[:, None, None] ** 2 * eye,
                    error[..., None],
                ))[..., 0]
                scale = (0.35 / update.abs().amax(-1).clamp_min(0.35)).unsqueeze(-1)
                trial = (q + update * scale).clamp(self.lower + 1e-7, self.upper - 1e-7)
                trial_error = self._error_jacobian(trial, expanded, normal_only)[0]
                accept = (~frozen) & (t.linalg.vector_norm(trial_error, dim=-1) < norm)
                q = t.where(accept[:, None], trial, q)
                changed_damping = t.where(accept, damping * 0.7, damping * 2).clamp(0.0001, 1)
                damping = t.where(frozen, damping, changed_damping)
            error, _, _ = self._error_jacobian(q, expanded, normal_only)
            norm = t.linalg.vector_norm(error, dim=-1)
            improve = (~frozen) & (norm < best_error)
            best_q = t.where(improve[:, None], q, best_q)
            error, _, pose = self._error_jacobian(best_q, expanded, normal_only)
            residual = t.linalg.vector_norm(error, dim=-1)
            position = t.linalg.vector_norm(error[:, :3], dim=-1)
            orientation = t.acos((pose[:, :3, 2] * expanded[:, :3, 2]).sum(-1).clamp(-1, 1)) if normal_only else t.linalg.vector_norm(error[:, 3:], dim=-1) / 0.25
            finite = t.isfinite(best_q).all(-1) & t.isfinite(residual)
            valid = finite & (residual < TOLERANCE) & (position < TOLERANCE)
            valid &= (best_q >= self.lower).all(-1) & (best_q <= self.upper).all(-1)
            valid = valid.reshape(n, branches) & available
        reshape = lambda v: v.reshape(n, branches, *v.shape[1:])
        return dict(q=reshape(best_q), valid=valid, residual=reshape(residual),
            position_error_m=reshape(position), orientation_error_rad=reshape(orientation),
            flange_local=reshape(pose), converged_at=reshape(solved_at),
            start_available=available, query_sha256=identities,
            branch_ids=["continuation"] + [f"canonical-{i}" for i in range(starts)],
            iterations=iterations, normal_only=normal_only, version=VERSION,
            polishing_tolerance=POLISH_TOLERANCE, preserve_known=preserve_known.tolist())

    def solve(self, targets, initial=None, **kwargs):
        """Deterministic mathematical choice; native collision selection is separate."""
        pool = self.solve_candidates(targets, initial, **kwargs)
        t = self.torch
        indices = []
        for good, error in zip(pool["valid"].cpu().numpy(), pool["residual"].cpu().numpy()):
            ids = np.flatnonzero(good)
            indices.append(int(ids[0]) if len(ids) else int(np.argmin(error)))
        ii = t.as_tensor(indices, device=self.device)
        jj = t.arange(len(ii), device=self.device)
        result = {k: pool[k][jj, ii] for k in ("q", "valid", "residual", "position_error_m", "orientation_error_rad", "flange_local")}
        result.update(version=VERSION, branch_index=ii, pool=pool)
        return result


class StableShotIK:
    def __init__(self, cfg, assets, *, device="cpu", dtype=None):
        self.cfg = cfg
        self.dropper = StableBatchIK.from_urdf(Path(assets) / "dropper.urdf", role="dropper", device=device, dtype=dtype)
        self.board = StableBatchIK.from_urdf(Path(assets) / "board_arm.urdf", role="board", device=device, dtype=dtype)

    def candidates(self, bucket_xy, board_xyz, tilt_deg, gap_m, *, initial10=None,
        initial14=None, known_q10=None, known_q14=None, starts=8, iterations=120,
        preserve_known=None):
        targets = shot_targets(self.cfg, bucket_xy, board_xyz, tilt_deg, gap_m)
        first = self.dropper.solve_candidates(targets["dropper_flange_world"], initial10,
            known=known_q10, starts=starts, iterations=iterations, preserve_known=preserve_known)
        second = self.board.solve_candidates(targets["board_flange_local"], initial14,
            known=known_q14, starts=starts, iterations=iterations, preserve_known=preserve_known)
        return dict(targets=targets, dropper=first, board=second, version=VERSION)
