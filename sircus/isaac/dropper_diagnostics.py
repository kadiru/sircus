"""Offline full-orientation audit of dropper targets; no simulator/hardware commands."""

import argparse
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from prepare_assets import chain, fk, matrix
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation


def solve_dropper(target, links, initial):
    lo = np.array([l[2] for l in links])
    hi = np.array([l[3] for l in links])
    rng = np.random.default_rng(20261101)
    attempts = [("original", initial, 100), ("iteration", initial, 400)]
    attempts += [("seed", (lo + hi) / 2, 300)]
    attempts += [("seed", rng.uniform(lo + 1e-4, hi - 1e-4), 300) for _ in range(7)]

    def residual(q):
        cur = fk(q, links)
        return np.r_[
            cur[:3, 3] - target[:3, 3],
            0.25 * Rotation.from_matrix(target[:3, :3] @ cur[:3, :3].T).as_rotvec(),
        ]

    best = None
    for stage, seed, budget in attempts:
        fit = least_squares(
            residual,
            seed,
            bounds=(lo, hi),
            max_nfev=budget,
            ftol=1e-8,
            gtol=1e-8,
            xtol=1e-8,
        )
        norm = float(np.linalg.norm(fit.fun))
        record = {
            "stage": stage,
            "residual": norm,
            "solved": norm < 1e-4,
            "position_error_mm": float(np.linalg.norm(fit.fun[:3]) * 1000),
            "rotation_error_deg": float(np.rad2deg(np.linalg.norm(fit.fun[3:]) / 0.25)),
            "joint_limit_margin_rad": float(np.minimum(fit.x - lo, hi - fit.x).min()),
        }
        if best is None or norm < best["residual"]:
            best = record
        if record["solved"]:
            return record
    return dict(best, stage="unresolved")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--assets", default="/workspace/assets")
    p.add_argument("--output", required=True)
    a = p.parse_args()
    assets = Path(a.assets)
    links = chain(ET.parse(assets / "board_arm.urdf").getroot())
    poses = json.loads((assets / "poses.json").read_text())
    cfg = json.loads((Path(__file__).parent / "scene.json").read_text())
    rows = []
    for gap in [0.06, 0.10, 0.15, 0.20, 0.25]:
        release = np.array([0.67, -0.1, 0.4 + cfg["ball"]["radius_m"] + gap])
        target = matrix(release + [0, 0, 0.095], [np.pi, 0, 0])
        rows.append(
            dict(
                gap_m=gap,
                target_flange_m=target[:3, 3].tolist(),
                **solve_dropper(target, links, poses["dropper_joint_rad"]),
            )
        )
    Path(a.output).write_text(
        json.dumps(
            {
                "schema": "sircus-dropper-ik-audit-v1",
                "cases": rows,
                "tolerance": 1e-4,
                "seed": 20261101,
                "scope": "Full flange orientation; unresolved is not proof of geometric infeasibility. No motion or changed sweep results.",
            },
            indent=2,
        )
    )
