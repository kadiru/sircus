"""Finite PhysX height/gap/tilt sweep and repeated selected-shot replay."""

import copy
import datetime
import hashlib
import json
import shutil
import time
from pathlib import Path

import numpy as np
from PIL import Image
from plane_contract import action, candidates, compact, contract, scene, select
from settled_shot import SettledShot


class PlaneShot(SettledShot):
    capture_setup = False

    def before_release(self, target):
        result = super().before_release(target)
        if self.capture_setup and not result.get("invalid"):
            self.sim.render()
            self.camera.update(self.dt * 24, force_recompute=True)
            rgb = self.camera.data.output["rgb"][0, :, :, :3].cpu().numpy()
            Image.fromarray(rgb).save(self.out / "setup.jpg")
        return result


def run(sim, arms, ball, camera, contact, cfg, poses, args, atomic):
    out = Path(args.output)
    stamp = datetime.datetime.now(datetime.UTC).strftime("%Y%m%dT%H%M%SZ")
    dest = out / ("plane-sweep-" + stamp)
    dest.mkdir()
    spec = contract(copy.deepcopy(cfg))
    spec["source_sha256"] = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in Path(__file__).parent.glob("*.py")
    }
    atomic(dest / "contract.json", spec)
    run_args = copy.copy(args)
    run_args.output = str(dest)
    env = PlaneShot(
        sim, arms, ball, camera, contact, copy.deepcopy(cfg), poses, run_args, atomic
    )
    started = time.monotonic()
    rows, repeats, replays = [], [], []

    def publish(phase):
        report = {
            "run": dest.name,
            "phase": phase,
            "contract": spec,
            "timestamp_utc": datetime.datetime.now(datetime.UTC).isoformat(),
            "elapsed_s": time.monotonic() - started,
            "completed": len(rows),
            "planned": len(candidates()),
            "captures": sum(r["success"] for r in rows),
            "invalid": sum(bool(r.get("invalid")) for r in rows),
            "rows": rows,
            "repeats": repeats,
            "replays": replays,
            "scope": spec["scope"],
        }
        atomic(dest / "summary.json", report)
        atomic(out / "plane-sweep.json", report)
        return report

    def shot(c, preview=False):
        if shutil.disk_usage(dest).free < 3 * 1024**3:
            raise RuntimeError("Free space below 3 GiB")
        return env.scene_shot(scene(c), np.array(action(c)), preview=preview)

    try:
        publish("grid")
        for c in candidates():
            result = shot(c)
            with (dest / "grid.jsonl").open("a") as f:
                f.write(
                    json.dumps({"candidate": c, "result": result}, allow_nan=False)
                    + "\n"
                )
            rows.append(compact(c, result))
            publish("grid")
            if len(rows) % 25 == 0:
                print(
                    "PLANE_GRID", len(rows), sum(r["success"] for r in rows), flush=True
                )
        selected = select(rows)
        atomic(dest / "selected.json", selected)
        for c in selected:
            entry = {"candidate": c, "trials": []}
            repeats.append(entry)
            for _ in range(5):
                result = shot(c)
                entry["trials"].append(compact(c, result))
                publish("repeats")
        # Fresh measured trajectories; display each separately from searched captures.
        for i, c in enumerate(selected):
            env.capture_setup = i == 0
            result = shot(c, preview=True)
            atomic(dest / f"replay-{i}.json", result)
            replays.append(
                {
                    "candidate": c,
                    "result": compact(c, result),
                    "trajectory_m": result.get("trajectory_m", []),
                    "release_center_world_m": result.get("release_center_world_m"),
                }
            )
            if i == 0 and (dest / "setup.jpg").exists():
                shutil.copyfile(dest / "setup.jpg", out / "plane-setup.tmp.jpg")
                (out / "plane-setup.tmp.jpg").replace(out / "plane-setup.jpg")
            publish("replay")
        report = publish("completed")
        report["artifact_sha256"] = {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(dest.iterdir())
            if p.is_file() and p.name != "summary.json"
        }
        atomic(dest / "summary.json", report)
        atomic(out / "plane-sweep.json", report)
        print(
            "PLANE_COMPLETE",
            len(rows),
            report["captures"],
            report["invalid"],
            flush=True,
        )
    except BaseException:
        publish("failed")
        raise
