"""Bounded multi-position search on one GPU; endpoint poses, not motion policies."""

import argparse
import copy
import datetime
import hashlib
import json
import os
import shutil
import sys
import time
import traceback
from pathlib import Path

from feasibility_contract import (
    ACTION_KEYS,
    baseline_actions,
    perturb,
    rank,
    refinement_actions,
    search_actions,
    select,
    specification,
)
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--assets", default="/workspace/assets")
parser.add_argument("--output", default="/workspace/output")
parser.add_argument("--indices", default=",".join(map(str, range(32))))
parser.add_argument("--smoke", action="store_true")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
indices = [int(i) for i in args.indices.split(",")]
root = Path(args.output)
root.mkdir(parents=True, exist_ok=True)
if shutil.disk_usage(root).free < 3 * 1024**3:
    parser.error("Free disk below 3 GiB")
stamp = datetime.datetime.now(datetime.UTC).strftime("%Y%m%dT%H%M%SZ")
dest = root / f"feasibility-{stamp}-n{len(indices)}"
dest.mkdir()
app = AppLauncher(args).app

import isaaclab.sim as sim_utils
import torch
from feasibility_runtime import FeasibilityShots
from isaaclab.scene import InteractiveScene
from parallel_assets import bucket_asset, robot_body_filters, scene_config
from parallel_runtime import atomic


def main():
    started = time.monotonic()
    cfg = json.loads(Path(__file__).with_name("scene.json").read_text())
    poses = json.loads((Path(args.assets) / "poses.json").read_text())
    spec = specification(cfg, indices, args.smoke)
    spec["source_sha256"] = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(Path(__file__).parent.glob("*.py"))
    }
    spec["gpu"] = torch.cuda.get_device_name(0)
    atomic(dest / "contract.json", spec)
    sim = sim_utils.SimulationContext(
        sim_utils.SimulationCfg(
            dt=cfg["simulation"]["physics_dt_s"],
            render_interval=8,
            device=args.device,
            physx=sim_utils.PhysxCfg(enable_ccd=False),
        )
    )
    bucket_usd = dest / "bucket.usda"
    bucket_asset(bucket_usd, cfg["bucket"])
    scene_cfg, filters = scene_config(
        cfg,
        poses,
        args.assets,
        bucket_usd,
        len(indices),
        robot_body_filters(args.assets),
    )
    scene = InteractiveScene(scene_cfg)
    sim.reset()
    env = FeasibilityShots(
        sim, scene, None, None, cfg, poses, args, copy.deepcopy(spec), filters, dest
    )
    state = {
        "run": dest.name,
        "contract": spec,
        "probes": [],
        "attempts": 0,
        "cells": [
            dict(
                c,
                baseline=[],
                search=[],
                refinement=[],
                nominal=[],
                fixed_xy_ablation=[],
                confirmations=[],
                winner=None,
            )
            for c in spec["cells"]
        ],
    }

    def publish(phase):
        state.update(
            phase=phase,
            elapsed_s=time.monotonic() - started,
            timestamp_utc=datetime.datetime.now(datetime.UTC).isoformat(),
            max_bucket_drift_m=env.peak_bucket_drift,
        )
        atomic(dest / "summary.json", state)
        if not args.smoke:
            atomic(root / "feasibility-map.json", state)

    def attempt(actions, stage, enabled=None):
        if shutil.disk_usage(root).free < 3 * 1024**3:
            raise RuntimeError("Free disk below 3 GiB")
        rows, raw = env.shot_batch(actions, enabled=enabled)
        for i, row in enumerate(rows):
            row["trial_id"] = f"{dest.name}-{state['attempts']:06d}"
            row["stage"] = stage
            row["included"] = enabled is None or enabled[i]
            if row["included"]:
                state["attempts"] += 1
                state["cells"][i][stage].append(row)
        with (dest / "trials.jsonl").open("a") as f:
            f.write(
                json.dumps({"stage": stage, "rows": rows, "raw": raw}, allow_nan=False)
                + "\n"
            )
        print(
            json.dumps(
                {
                    "stage": stage,
                    "attempts": state["attempts"],
                    "wall_s": raw["wall_s"],
                    "captures": sum(r["route_success"] and r["included"] for r in rows),
                    "invalid": sum(bool(r["invalid"]) and r["included"] for r in rows),
                }
            ),
            flush=True,
        )
        publish(stage)
        return rows, raw

    try:
        publish("probes")
        state["probes"] = env.probes()
        if not args.smoke:
            state["direct_drop_audit"] = env.direct_drop_audit(
                spec["direct_drop_audit_ball_heights_m"]
            )
            publish("reachability")
        plans = [search_actions(cfg, c) for c in spec["cells"]]
        if args.smoke:
            for a in baseline_actions()[:2]:
                attempt([dict(a) for _ in indices], "baseline")
            for b in [0, 1]:
                attempt([dict(p[b]) for p in plans], "search")
            state["validation_passed"] = True
            publish("completed")
            print("FEASIBILITY_SMOKE_COMPLETE " + str(dest), flush=True)
            return
        for a in baseline_actions():
            attempt([dict(a) for _ in indices], "baseline")
        for b in range(max(map(len, plans))):
            enabled = [b < len(p) for p in plans]
            attempt(
                [dict(p[b] if b < len(p) else p[0]) for p in plans], "search", enabled
            )
        refinements = []
        for c in state["cells"]:
            # Unsuccessful best valid shot can also seed local refinement.
            rows = [r for r in c["baseline"] + c["search"] if not r["invalid"]]
            refinements.append(refinement_actions(max(rows, key=rank, default=None)))
        for b in range(27):
            enabled = [b < len(p) for p in refinements]
            attempt(
                [dict(p[b] if p else baseline_actions()[0]) for p in refinements],
                "refinement",
                enabled,
            )
        for c in state["cells"]:
            winner = select(c["baseline"] + c["search"] + c["refinement"])
            c["winner"] = (
                {k: winner[k] for k in (*ACTION_KEYS, "id", "trial_id")}
                if winner
                else None
            )
        atomic(
            dest / "frozen-settings.json",
            {c["id"]: c["winner"] for c in state["cells"]},
        )
        publish("frozen")
        enabled = [bool(c["winner"]) for c in state["cells"]]
        if any(enabled):
            for stage, count, nominal in [
                ("nominal", 3, True),
                ("fixed_xy_ablation", 3, True),
                ("confirmations", 30, False),
            ]:
                for rep in range(count):
                    actions = [
                        perturb(
                            c["winner"] or baseline_actions()[0],
                            c["index"],
                            rep,
                            nominal,
                        )
                        for c in state["cells"]
                    ]
                    if stage == "fixed_xy_ablation":
                        for a in actions:
                            a.update(board_x_m=0.67, board_y_m=-0.10)
                    rows, raw = attempt(actions, stage, enabled)
                    if rep == 0:
                        for i, c in enumerate(state["cells"]):
                            if enabled[i]:
                                c[stage + "_replay"] = {
                                    "trial_id": rows[i]["trial_id"],
                                    "trajectory_m": raw["trajectory_m"][i],
                                    "sample_dt_s": 8 * env.dt,
                                }
        state["validation_passed"] = True
        state["artifact_sha256"] = {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (
                dest / "contract.json",
                dest / "trials.jsonl",
                dest / "frozen-settings.json",
                bucket_usd,
            )
        }
        publish("completed")
        print("FEASIBILITY_COMPLETE " + str(dest), flush=True)
    except Exception as exc:
        state["error"] = f"{type(exc).__name__}: {exc}"
        publish("failed")
        raise


try:
    main()
except BaseException as exc:  # noqa: BLE001 -- Kit shutdown can hide a nonzero exit.
    traceback.print_exc()
    atomic(
        dest / "failure.json", {"error": str(exc), "traceback": traceback.format_exc()}
    )
    sys.stderr.flush()
    os._exit(1)
finally:
    app.close(skip_cleanup=True)
