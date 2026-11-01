"""Dense fixed-plane catalogue with measured bucket-contact quality and confirmation."""

import copy
import datetime
import hashlib
import json
import shutil
import time
from pathlib import Path

import numpy as np
import torch
from catalogue_contract import finalists, grid, perturb, refinement, specification
from entry_quality import EntryQuality
from plane_contract import action, compact, scene
from settled_shot import SettledShot


class QualityShot(SettledShot):
    def board_contact_force(self):
        return float(
            torch.linalg.vector_norm(self.contact.data.force_matrix_w[..., 0, :])
        )

    def start_flight_observation(self):
        self.quality = EntryQuality(self.cfg["bucket"], self.cfg["ball"]["radius_m"])

    def observe_flight(self, pos, score):
        f = float(torch.linalg.vector_norm(self.contact.data.force_matrix_w[..., 1, :]))
        self.quality.step(pos, f, score.entered)

    def finish_flight_observation(self, score):
        return self.quality.result(score.success)


def contact_probe(env):
    xy = [0.67, -0.50]
    env.place_bucket(xy)
    b = env.cfg["bucket"]
    top = b["assumed_interior_depth_m"] + b["assumed_bottom_thickness_m"]
    results = []
    for label, dx in [
        ("floor", 0.0),
        ("rim", b["inner_top_diameter_m"] / 2),
        ("outside", 0.3),
    ]:
        state = env.ball.data.default_root_state.clone()
        state[0, :3] = torch.tensor(
            [xy[0] + dx, xy[1], top + 0.15], device=env.sim.device
        )
        state[0, 7:] = 0
        env.ball.write_root_state_to_sim(state)
        env.ball.reset()
        env.contact.reset()
        peak = 0.0
        for _ in range(240):
            env.tick()
            forces = env.contact.data.force_matrix_w
            if forces.shape[-2] != 2:
                raise RuntimeError("Expected board and bucket contact filters")
            peak = max(peak, float(torch.linalg.vector_norm(forces[..., 1, :])))
        results.append({"kind": label, "peak_bucket_force_N": peak})
    if not (
        results[0]["peak_bucket_force_N"] > 0.01
        and results[1]["peak_bucket_force_N"] > 0.01
        and results[2]["peak_bucket_force_N"] < 0.01
    ):
        raise RuntimeError(
            "Bucket contact filter failed physical probes: " + str(results)
        )
    return results


def summary(trials):
    n = len(trials)
    captured = sum(t["success"] for t in trials)
    invalid = sum(bool(t.get("invalid")) for t in trials)
    clean = sum(t["quality_score"] == 2 for t in trials)
    z = 1.96
    p = captured / n if n else 0
    lower = (
        (p + z * z / (2 * n) - z * ((p * (1 - p) + z * z / (4 * n)) / n) ** 0.5)
        / (1 + z * z / n)
        if n
        else None
    )
    return {
        "trials": n,
        "captures": captured,
        "clean": clean,
        "assisted": captured - clean,
        "misses": n - captured - invalid,
        "invalid": invalid,
        "mean_tier_score": sum(t["quality_score"] for t in trials) / n if n else None,
        "wilson95_lower": lower,
    }


def run(sim, arms, ball, camera, contact, cfg, poses, args, atomic):
    out = Path(args.output)
    stamp = datetime.datetime.now(datetime.UTC).strftime("%Y%m%dT%H%M%SZ")
    dest = out / ("shot-catalogue-" + stamp)
    dest.mkdir()
    spec = specification(copy.deepcopy(cfg))
    spec["source_sha256"] = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in Path(__file__).parent.glob("*.py")
    }
    atomic(dest / "contract.json", spec)
    run_args = copy.copy(args)
    run_args.output = str(dest)
    env = QualityShot(
        sim, arms, ball, camera, contact, copy.deepcopy(cfg), poses, run_args, atomic
    )
    started = time.monotonic()
    rows = []
    confirmed = []
    probes = []
    refine = []
    trial_number = 0

    def publish(phase):
        d = {
            "run": dest.name,
            "phase": phase,
            "timestamp_utc": datetime.datetime.now(datetime.UTC).isoformat(),
            "elapsed_s": time.monotonic() - started,
            "contract": spec,
            "probes": probes,
            "grid_planned": len(grid()),
            "search_completed": len(rows),
            "refinement_planned": len(refine),
            "search_summary": summary(rows),
            "rows": rows,
            "confirmed": confirmed,
        }
        atomic(dest / "summary.json", d)
        atomic(out / "shot-catalogue.json", d)
        return d

    def shot(c, stage):
        nonlocal trial_number
        trial_number += 1
        trial_id = f"{dest.name}-{trial_number:04d}"
        if shutil.disk_usage(dest).free < 3 * 1024**3:
            raise RuntimeError("Free disk below3GiB")
        r = env.scene_shot(scene(c), np.array(action(c)))
        row = compact(c, r)
        row["trial_id"] = trial_id
        for k in [
            "quality_score",
            "entry_class",
            "bucket_contact_before_entry",
            "rim_band_contact",
            "exited_after_entry",
            "peak_bucket_force_before_entry_N",
            "minimum_mouth_clearance_mm",
        ]:
            row[k] = r.get(k, 0 if k == "quality_score" else None)
        with (dest / "trials.jsonl").open("a") as f:
            f.write(
                json.dumps(
                    {"trial_id": trial_id, "stage": stage, "candidate": c, "result": r},
                    allow_nan=False,
                )
                + "\n"
            )
        return row

    try:
        publish("contact_probes")
        probes = contact_probe(env)
        atomic(dest / "contact-probes.json", probes)
        if args.catalogue_probe_only:
            publish("probe_completed")
            return
        for c in grid():
            rows.append(shot(c, "grid"))
            publish("grid")
            if len(rows) % 50 == 0:
                print("CATALOGUE_GRID", len(rows), summary(rows), flush=True)
        refine = refinement(rows)
        atomic(dest / "refinement.json", refine)
        for c in refine:
            rows.append(shot(c, "refinement"))
            publish("refinement")
        selected = finalists(rows)
        atomic(dest / "selected.json", selected)
        for idx, c in enumerate(selected):
            ball_z = c["height_m"] + c["gap_m"] + cfg["ball"]["radius_m"]
            entry = {
                "candidate": c,
                "distance_mm": 400,
                "board_height_mm": c["height_m"] * 1000,
                "drop_gap_mm": c["gap_m"] * 1000,
                "ball_center_release_height_mm": ball_z * 1000,
                "ball_bottom_release_height_mm": (ball_z - cfg["ball"]["radius_m"])
                * 1000,
                "dropper_flange_height_mm": (ball_z + 0.095) * 1000,
                "nominal": [],
                "perturbed": [],
            }
            confirmed.append(entry)
            for mode in ["nominal", "perturbed"]:
                for i in range(20):
                    sample = c if mode == "nominal" else perturb(c, idx * 20 + i)
                    entry[mode].append(shot(sample, mode))
                    entry[mode + "_summary"] = summary(entry[mode])
                    publish("confirmation")
        d = publish("completed")
        d["artifact_sha256"] = {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in dest.iterdir()
            if p.is_file() and p.name != "summary.json"
        }
        atomic(dest / "summary.json", d)
        atomic(out / "shot-catalogue.json", d)
        print("CATALOGUE_COMPLETE", summary(rows), flush=True)
    except BaseException:
        publish("failed")
        raise
