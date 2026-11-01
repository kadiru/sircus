"""Versioned deterministic endpoint solver; never relaxes acceptance or plans paths."""

import numpy as np
from prepare_assets import fk
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

VERSION = "sircus-endpoint-ik-v2"
TOLERANCE = 1e-4


def normal_jacobian(q, links):
    t = np.eye(4)
    origins, axes = [], []
    for angle, (origin, axis, *_) in zip(q, links):
        t = t @ origin
        origins.append(t[:3, 3].copy())
        axes.append(t[:3, :3] @ axis)
        r = np.eye(4)
        r[:3, :3] = Rotation.from_rotvec(axis * angle).as_matrix()
        t = t @ r
    return np.column_stack(
        [
            np.r_[np.cross(a, t[:3, 3] - o), 0.25 * np.cross(a, t[:3, 2])]
            for o, a in zip(origins, axes)
        ]
    )


def solve(target, links, initial, normal_only):
    lo, hi = np.array([l[2] for l in links]), np.array([l[3] for l in links])
    initial = np.clip(initial, lo + 1e-6, hi - 1e-6)
    nearby = np.clip(initial + [0.1, -0.1, 0.1, 0, 0, 0], lo + 1e-4, hi - 1e-4)
    original = [initial, nearby] if normal_only else [initial]
    starts = [("original", q, 80 if normal_only else 100, 1e-8) for q in original]
    starts += [("strict", q, 500, 1e-12) for q in original]
    rng = np.random.default_rng(20261101)
    starts += [
        ("multistart", q, 400, 1e-12)
        for q in [(lo + hi) / 2] + [rng.uniform(lo + 1e-4, hi - 1e-4) for _ in range(7)]
    ]

    def residual(q):
        cur = fk(q, links)
        angular = (
            (cur[:3, 2] - target[:3, 2])
            if normal_only
            else Rotation.from_matrix(target[:3, :3] @ cur[:3, :3].T).as_rotvec()
        )
        return np.r_[cur[:3, 3] - target[:3, 3], 0.25 * angular]

    best, log = None, []
    for stage, q, budget, stop in starts:
        # The UF850's last joint spins about the flange normal without changing
        # either residual. Remove that null variable rather than chase numerical
        # noise in a rank-deficient six-variable optimization.
        reduce_spin = (
            normal_only
            and stage != "original"
            and np.linalg.norm(normal_jacobian(q, links)[:, -1]) < 1e-10
        )
        if reduce_spin:
            unpack = lambda v, spin=q[-1]: np.r_[v, spin]
            fit = least_squares(
                lambda v, unpack=unpack: residual(unpack(v)),
                q[:-1],
                bounds=(lo[:-1], hi[:-1]),
                jac=lambda v, unpack=unpack: normal_jacobian(unpack(v), links)[:, :-1],
                max_nfev=budget,
                ftol=stop,
                gtol=stop,
                xtol=stop,
            )
            fit.x = unpack(fit.x)
        else:
            jac = (
                (lambda v: normal_jacobian(v, links))
                if normal_only and stage != "original"
                else "2-point"
            )
            fit = least_squares(
                residual,
                q,
                bounds=(lo, hi),
                jac=jac,
                max_nfev=budget,
                ftol=stop,
                gtol=stop,
                xtol=stop,
            )
        norm = float(np.linalg.norm(fit.fun))
        cur = fk(fit.x, links)
        angle = (
            np.arccos(np.clip(cur[:3, 2] @ target[:3, 2], -1, 1))
            if normal_only
            else np.linalg.norm(fit.fun[3:]) / 0.25
        )
        record = {
            "q": fit.x.tolist(),
            "residual": norm,
            "solved": norm < TOLERANCE,
            "stage": stage,
            "position_error_mm": float(np.linalg.norm(fit.fun[:3]) * 1000),
            "orientation_error_deg": float(np.rad2deg(angle)),
        }
        log.append({"stage": stage, "nfev": fit.nfev, "residual": norm})
        if best is None or norm < best["residual"]:
            best = record
        if record["solved"]:
            return dict(record, attempts=log)
    return dict(best, stage="unresolved", attempts=log)
