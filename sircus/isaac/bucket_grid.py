"""Finite spatial catalogue with contact-instrumented board-only route evaluation."""

import copy
import datetime
import hashlib
import json
import shutil
import time
from pathlib import Path

import numpy as np
import torch
from board_route import BoardRoute
from bucket_grid_contract import (
    candidates,
    cells,
    choose,
    contract,
    perturb,
    refine,
    scene,
    select,
    summary,
)
from plane_contract import action, compact
from shot_catalogue import QualityShot


class RouteShot(QualityShot):
    def forces(self):
        matrix = self.contact.data.force_matrix_w[0, 0].cpu().numpy()
        paths = self.sim._sircus_grid_filters
        if len(matrix) != len(paths):
            raise RuntimeError("Contact filter cardinality mismatch")
        net = self.contact.data.net_forces_w[0, 0].cpu().numpy()
        values = {
            "board": float(np.linalg.norm(matrix[0])),
            "bucket": float(np.linalg.norm(matrix[1])),
            "table": float(np.linalg.norm(matrix[2])),
            "floor": float(np.linalg.norm(matrix[3])),
            "robot": float(np.linalg.norm(matrix[4:], axis=-1).sum()),
            "unattributed": float(np.linalg.norm(net - matrix.sum(axis=0))),
        }
        if not all(np.isfinite(v) for v in values.values()):
            raise RuntimeError("Nonfinite contact readback")
        return values

    def start_flight_observation(self):
        super().start_flight_observation()
        self.direct = BoardRoute(self.dt)

    def observe_flight(self, pos, score):
        super().observe_flight(pos, score)
        self.direct.step(self.forces())

    def finish_flight_observation(self, score):
        quality = super().finish_flight_observation(score)
        return dict(
            quality, **self.direct.result(score.success, quality["quality_score"])
        )

    def shot(self, action):
        r = super().shot(action)
        self.last_trajectory = r.get("trajectory_m", [])
        return r


def probes(env, xy):
    env.place_bucket(xy)
    b = env.cfg["bucket"]
    top = b["assumed_interior_depth_m"] + b["assumed_bottom_thickness_m"]
    trials = [
        ("bucket", [*xy, top + 0.15]),
        ("rim", [xy[0] + b["inner_top_diameter_m"] / 2, xy[1], top + 0.15]),
        ("table", [-0.05, -0.65, 0.25]),
        ("floor", [1.8, -0.65, -0.50]),
        ("robot", [0.0, 0.0, 0.75]),
    ]
    results = []
    for label, xyz in trials:
        state = env.ball.data.default_root_state.clone()
        state[0, :3] = torch.tensor(xyz, device=env.sim.device)
        state[0, 7:] = 0
        env.ball.write_root_state_to_sim(state)
        env.ball.reset()
        env.contact.reset()
        peak = {}
        for _ in range(240):
            env.tick()
            for k, v in env.forces().items():
                peak[k] = max(peak.get(k, 0), v)
        target = "bucket" if label == "rim" else label
        passed = peak[target] > 0.01
        if label in ("table", "floor"):
            passed &= peak["bucket"] < 0.01 and peak["board"] < 0.01
        results.append({"kind": label, "peak_N": peak, "passed": bool(passed)})
    return results


def run(sim, arms, ball, camera, contact, cfg, poses, args, atomic):
    out = Path(args.output)
    dest = out / (
        "bucket-grid-" + datetime.datetime.now(datetime.UTC).strftime("%Y%m%dT%H%M%SZ")
    )
    dest.mkdir()
    spec = contract(copy.deepcopy(cfg), args.bucket_grid_smoke)
    spec["source_sha256"] = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in Path(__file__).parent.glob("*.py")
    }
    spec["contact_filters"] = sim._sircus_grid_filters
    atomic(dest / "contract.json", spec)
    a = copy.copy(args)
    a.output = str(dest)
    env = RouteShot(
        sim, arms, ball, camera, contact, copy.deepcopy(cfg), poses, a, atomic
    )
    started, number = time.monotonic(), 0
    entries, checks, controls = [], [], []

    def publish(phase):
        d = {
            "run": dest.name,
            "phase": phase,
            "elapsed_s": time.monotonic() - started,
            "timestamp_utc": datetime.datetime.now(datetime.UTC).isoformat(),
            "contract": spec,
            "cells": entries,
            "probes": checks,
            "controls": controls,
            "attempts": number,
        }
        atomic(dest / "summary.json", d)
        if not args.bucket_grid_smoke:
            atomic(out / "bucket-grid.json", d)
        return d

    def shot(c, cell, stage):
        nonlocal number
        if shutil.disk_usage(dest).free < 3 * 1024**3:
            raise RuntimeError("Free disk below 3 GiB")
        r = env.scene_shot(scene(c, cell), np.array(action(c)))
        number += 1
        row = compact(c, r)
        row.update(
            trial_id=f"{dest.name}-{number:05d}", cell_id=cell["id"], stage=stage
        )
        for k in (
            "quality_score",
            "route_success",
            "single_bounce_success",
            "route_quality",
            "route_version",
            "board_contact_episodes",
            "external_contact_steps",
            "contact_peaks_N",
        ):
            row[k] = r.get(k)
        if r.get("invalid"):
            row.update(
                route_success=False,
                single_bounce_success=False,
                route_quality=0,
                quality_score=0,
            )
        with (dest / "trials.jsonl").open("a") as f:
            f.write(
                json.dumps(
                    {
                        "trial_id": row["trial_id"],
                        "candidate": c,
                        "cell": cell,
                        "stage": stage,
                        "result": r,
                        "trajectory_m": env.last_trajectory
                        if stage in ("confirmation", "control")
                        else [],
                    },
                    allow_nan=False,
                )
                + "\n"
            )
        return row

    try:
        publish("probes")
        checks.extend(probes(env, [0.67, -0.5]))
        atomic(dest / "contact-probes.json", checks)
        if not all(p["passed"] for p in checks):
            raise RuntimeError("Direct-contact probes failed: " + str(checks))
        # Controls exercise both a known board-only candidate and the old table-assisted candidate.
        for c, distance in [
            (
                {"id": "control-near", "height_m": 0.4, "gap_m": 0.13, "tilt_deg": 21},
                0.4,
            ),
            (
                {
                    "id": "control-far",
                    "height_m": 0.45,
                    "gap_m": 0.13,
                    "tilt_deg": 13.5,
                },
                0.6,
            ),
        ]:
            cell = {
                "id": c["id"],
                "bucket_xy_m": [0.67, -0.1 - distance],
                "distance_m": distance,
                "difficulty": "control",
            }
            controls.append(shot(c, cell, "control"))
        if (
            not controls[0]["route_success"]
            or not controls[1]["external_contact_steps"]
            or controls[1]["route_success"]
        ):
            raise RuntimeError(
                "Board-only route positive/negative controls failed: " + str(controls)
            )
        if args.bucket_grid_smoke:
            publish("smoke_completed")
            return
        for ci, cell in enumerate(cells()):
            e = dict(
                cell,
                phase="searching",
                rows=[],
                screened=[],
                confirmation=[],
                winner=None,
            )
            entries.append(e)
            for c in candidates():
                e["rows"].append(shot(c, cell, "grid"))
                e["search_summary"] = summary(e["rows"])
                publish("searching")
            e["refinements"] = refine(e["rows"])
            for c in e["refinements"]:
                e["rows"].append(shot(c, cell, "refinement"))
                e["search_summary"] = summary(e["rows"])
                publish("searching")
            for index, c in enumerate(select(e["rows"])):
                group = {"candidate": c, "trials": []}
                e["screened"].append(group)
                for i in range(10):
                    group["trials"].append(
                        shot(perturb(c, ci, index, i), cell, "screening")
                    )
                    group["summary"] = summary(group["trials"])
                    e["phase"] = "screening"
                    publish("screening")
            e["winner"] = choose(e["screened"])
            if e["winner"]:
                for i in range(30):
                    e["confirmation"].append(
                        shot(perturb(e["winner"], ci, 0, i, True), cell, "confirmation")
                    )
                    e["confirmation_summary"] = summary(e["confirmation"])
                    e["phase"] = "confirming"
                    publish("confirming")
            e["phase"] = "completed" if e["winner"] else "no_route_setting_found"
            publish("cell_completed")
            print(
                "GRID_CELL_COMPLETE",
                cell["id"],
                e.get("confirmation_summary"),
                flush=True,
            )
        d = publish("completed")
        d["artifact_sha256"] = {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in dest.iterdir()
            if p.is_file() and p.name != "summary.json"
        }
        atomic(dest / "summary.json", d)
        atomic(out / "bucket-grid.json", d)
    except BaseException as exc:
        d = publish("failed")
        d["error_type"] = type(exc).__name__
        atomic(dest / "summary.json", d)
        if not args.bucket_grid_smoke:
            atomic(out / "bucket-grid.json", d)
        raise
