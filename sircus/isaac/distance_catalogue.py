"""Finite multi-distance search, screening and fresh confirmation on one GPU."""

import copy
import datetime
import hashlib
import json
import shutil
import time
from pathlib import Path

import numpy as np
from distance_catalogue_contract import (
    DISTANCES,
    choose,
    contract,
    grid,
    jitter,
    refinement,
    scene,
    selected,
)
from plane_contract import action, compact
from shot_catalogue import QualityShot, contact_probe, summary


class DiagnosticShot(QualityShot):
    def shot(self, action):
        r = super().shot(action)
        self.last_trajectory = r.get("trajectory_m", [])
        return r


def run(sim, arms, ball, camera, contact, cfg, poses, args, atomic):
    out = Path(args.output)
    dest = out / (
        "distance-catalogue-"
        + datetime.datetime.now(datetime.UTC).strftime("%Y%m%dT%H%M%SZ")
    )
    dest.mkdir()
    spec = contract(copy.deepcopy(cfg))
    spec["source_sha256"] = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in Path(__file__).parent.glob("*.py")
    }
    atomic(dest / "contract.json", spec)
    a = copy.copy(args)
    a.output = str(dest)
    env = DiagnosticShot(
        sim, arms, ball, camera, contact, copy.deepcopy(cfg), poses, a, atomic
    )
    entries = []
    started = time.monotonic()
    number = 0

    def publish(phase):
        d = {
            "phase": phase,
            "run": dest.name,
            "timestamp_utc": datetime.datetime.now(datetime.UTC).isoformat(),
            "elapsed_s": time.monotonic() - started,
            "contract": spec,
            "distances": entries,
            "attempts": number,
        }
        atomic(dest / "summary.json", d)
        atomic(out / "distance-catalogue.json", d)
        return d

    def shot(c, distance, stage):
        nonlocal number
        if shutil.disk_usage(dest).free < 3 * 1024**3:
            raise RuntimeError("Free disk below3GiB")
        r = env.scene_shot(scene(c, distance), np.array(action(c)))
        number += 1
        row = compact(c, r)
        row.update(trial_id=f"{dest.name}-{number:05d}", distance_m=distance)
        for k in (
            "quality_score",
            "entry_class",
            "bucket_contact_before_entry",
            "rim_band_contact",
            "exited_after_entry",
            "minimum_mouth_clearance_mm",
        ):
            row[k] = r.get(k, 0 if k == "quality_score" else None)
        with (dest / "trials.jsonl").open("a") as f:
            record = {
                "stage": stage,
                "candidate": c,
                "distance_m": distance,
                "trial_id": row["trial_id"],
                "result": r,
            }
            if stage == "confirmation":
                record["trajectory_m"] = env.last_trajectory
            f.write(json.dumps(record, allow_nan=False) + "\n")
        return row

    try:
        publish("starting")
        for di, distance in enumerate(DISTANCES):
            entry = {
                "distance_m": distance,
                "phase": "probes",
                "grid_planned": len(grid(distance)),
                "refinement_planned": 0,
                "rows": [],
                "screened": [],
                "winner": None,
                "confirmation": [],
                "probes": [],
            }
            entries.append(entry)
            entry["probes"] = contact_probe(
                env, scene(grid(distance)[0], distance)["bucket_xy_m"]
            )
            atomic(dest / f"probes-{di}.json", entry["probes"])
            for c in grid(distance):
                entry["rows"].append(shot(c, distance, "grid"))
                entry["search_summary"] = summary(entry["rows"])
                entry["phase"] = "grid"
                publish("searching")
            refine = refinement(entry["rows"], distance)
            entry["refinement_planned"] = len(refine)
            atomic(dest / f"refinement-{di}.json", refine)
            for c in refine:
                entry["rows"].append(shot(c, distance, "refinement"))
                entry["search_summary"] = summary(entry["rows"])
                entry["phase"] = "refinement"
                publish("searching")
            candidates = selected(entry["rows"])
            atomic(dest / f"selected-{di}.json", candidates)
            for ci, c in enumerate(candidates):
                e = {"candidate": c, "nominal": [], "perturbed": []}
                entry["screened"].append(e)
                for mode, count in [("nominal", 5), ("perturbed", 20)]:
                    for i in range(count):
                        sample = c if mode == "nominal" else jitter(c, di, ci, i)
                        e[mode].append(shot(sample, distance, mode))
                        e[mode + "_summary"] = summary(e[mode])
                        entry["phase"] = "screening"
                        publish("screening")
            winner = choose(entry["screened"])
            entry["winner"] = winner
            atomic(dest / f"winner-{di}.json", winner)
            if winner:
                for i in range(50):
                    entry["confirmation"].append(
                        shot(
                            jitter(winner, di, 0, i, final=True),
                            distance,
                            "confirmation",
                        )
                    )
                    entry["confirmation_summary"] = summary(entry["confirmation"])
                    entry["phase"] = "confirmation"
                    publish("confirming")
            entry["phase"] = "completed" if winner else "no_capture_found"
            publish("distance_completed")
            print(
                "DISTANCE_COMPLETE",
                distance,
                entry.get("confirmation_summary"),
                flush=True,
            )
        d = publish("completed")
        d["artifact_sha256"] = {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in dest.iterdir()
            if p.is_file() and p.name != "summary.json"
        }
        atomic(dest / "summary.json", d)
        atomic(out / "distance-catalogue.json", d)
    except BaseException as exc:
        d = publish("failed")
        d["error_type"] = type(exc).__name__
        atomic(dest / "summary.json", d)
        atomic(out / "distance-catalogue.json", d)
        raise
