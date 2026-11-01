"""Deterministic endpoint IK diagnostics; no motion/path/feasibility claims."""

import argparse
import hashlib
import json
import time
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

import numpy as np
from prepare_assets import chain, fk, matrix
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation


def board_target(cfg, scene, action):
    center = np.array(scene["board_center_m"])
    delta = np.array(scene["bucket_xy_m"]) - center[:2]
    yaw = np.arctan2(delta[1], delta[0]) + np.pi / 2 + np.deg2rad(action[1])
    rot = (
        Rotation.from_euler("z", yaw).as_matrix()
        @ Rotation.from_euler("x", action[0], degrees=True).as_matrix()
        @ Rotation.from_euler("z", cfg["board"]["mount_yaw_rad"]).as_matrix()
    )
    world = np.eye(4)
    world[:3, :3] = rot
    world[:3, 3] = center
    base = matrix([cfg["arms"]["base_spacing_m"], 0, 0], [0, 0, np.pi])
    mount = matrix(
        [0, 0, cfg["board"]["tool_offset_m"]], [0, 0, cfg["board"]["mount_yaw_rad"]]
    )
    return np.linalg.inv(base) @ world @ np.linalg.inv(mount)


def solve_board(target, links, initial, extended=True):
    lo = np.array([l[2] for l in links])
    hi = np.array([l[3] for l in links])
    initial = np.array(initial)
    nearby = np.clip(initial + [0.1, -0.1, 0.1, 0, 0, 0], lo + 1e-4, hi - 1e-4)
    attempts = [("original", initial, 80), ("original", nearby, 80)]
    if extended:
        attempts += [("iteration", initial, 300), ("iteration", nearby, 300)]
        rng = np.random.default_rng(20261101)
        attempts += [("seed", (lo + hi) / 2, 200)]
        attempts += [("seed", rng.uniform(lo + 1e-4, hi - 1e-4), 200) for _ in range(7)]

    def residual(q):
        cur = fk(q, links)
        return np.r_[cur[:3, 3] - target[:3, 3], 0.25 * (cur[:3, 2] - target[:3, 2])]

    log = []
    best = None
    started = time.monotonic()
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
        cur = fk(fit.x, links)
        norm = float(np.linalg.norm(fit.fun))
        log.append({"stage": stage, "nfev": fit.nfev, "residual": norm})
        record = {
            "q": fit.x.tolist(),
            "position_error_mm": float(
                np.linalg.norm(cur[:3, 3] - target[:3, 3]) * 1000
            ),
            "normal_error_deg": float(
                np.rad2deg(np.arccos(np.clip(cur[:3, 2] @ target[:3, 2], -1, 1)))
            ),
            "joint_limit_margin_rad": float(np.minimum(fit.x - lo, hi - fit.x).min()),
            "residual": norm,
        }
        if best is None or norm < best["residual"]:
            best = record
        if norm < 1e-4:
            return {
                **record,
                "solved": True,
                "stage": stage,
                "attempts": log,
                "wall_s": time.monotonic() - started,
            }
    return {
        **best,
        "solved": False,
        "stage": "unresolved",
        "attempts": log,
        "wall_s": time.monotonic() - started,
    }


def audit(runs, assets, output):
    cfg = json.loads((Path(__file__).parent / "scene.json").read_text())
    links = chain(ET.parse(Path(assets) / "board_arm.urdf").getroot())
    poses = json.loads((Path(assets) / "poses.json").read_text())
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    cases = []
    for run in runs:
        run = Path(run)
        rows = json.loads((run / "test.json").read_text())
        splits = json.loads((run / "splits.json").read_text())
        layouts = {s["id"]: s for s in splits["test"]}
        controls = [r for r in rows if not r.get("invalid")][:8]
        for row in [
            r for r in rows if r.get("invalid_component") == "Arm14"
        ] + controls:
            cases.append((run.name, row, layouts[row["scene_id"]]))
    results = []

    def publish(state):
        value = {
            "state": state,
            "completed": len(results),
            "planned": len(cases),
            "cases": results,
            "stage_counts": dict(
                Counter(r["solve"]["stage"] for r in results if r["historical_failure"])
            ),
            "historical_failures": sum(bool(r["historical_failure"]) for r in results),
            "scope": "Retrospective old-test diagnosis plus successful controls; not a fresh heldout score. IK endpoints only; unresolved does not prove infeasible.",
        }
        temp = out / "ik-audit.tmp"
        temp.write_text(json.dumps(value, indent=2))
        temp.replace(out / "ik-audit.json")

    publish("running")
    for run, row, scene in cases:
        result = solve_board(
            board_target(cfg, scene, row["action"]), links, poses["board_joint_rad"]
        )
        results.append(
            {
                "run": run,
                "scene": scene,
                "action": row["action"],
                "historical_failure": bool(row.get("invalid")),
                "solve": result,
            }
        )
        publish("running")
        print(
            "IK_AUDIT",
            len(results),
            len(cases),
            row["scene_id"],
            result["stage"],
            flush=True,
        )
    publish("completed")
    (out / "provenance.json").write_text(
        json.dumps(
            {
                "source_sha256": hashlib.sha256(
                    Path(__file__).read_bytes()
                ).hexdigest(),
                "robot_urdf_sha256": hashlib.sha256(
                    (Path(assets) / "board_arm.urdf").read_bytes()
                ).hexdigest(),
                "solver_seed": 20261101,
                "tolerance": 1e-4,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("runs", nargs="+")
    p.add_argument("--assets", default="/workspace/assets")
    p.add_argument("--output", required=True)
    args = p.parse_args()
    audit(args.runs, args.assets, args.output)
