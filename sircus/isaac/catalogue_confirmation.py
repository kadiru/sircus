"""Independent bounded confirmation of the best catalogue candidate, no search."""

import copy
import datetime
import hashlib
import json
import shutil
import time
from pathlib import Path

import numpy as np
from catalogue_contract import perturb
from plane_contract import action, compact, scene
from shot_catalogue import QualityShot, contact_probe, summary


def choose(report):
    if report["phase"] != "completed":
        raise ValueError("Source catalogue must be complete")
    return max(
        report["confirmed"],
        key=lambda e: (
            e["perturbed_summary"]["captures"],
            e["perturbed_summary"]["clean"],
            e["nominal_summary"]["captures"],
            e["candidate"]["id"],
        ),
    )["candidate"]


class DiagnosticShot(QualityShot):
    def shot(self, action):
        r = super().shot(action)
        self.last_trajectory = r.get("trajectory_m", [])
        return r


def run(sim, arms, ball, camera, contact, cfg, poses, args, atomic):
    source = Path(args.catalogue_confirm_source)
    original = json.loads(source.read_text())
    candidate = choose(original)
    out = Path(args.output)
    dest = out / (
        "catalogue-confirmation-"
        + datetime.datetime.now(datetime.UTC).strftime("%Y%m%dT%H%M%SZ")
    )
    dest.mkdir()
    spec = {
        "version": "sircus-catalogue-independent-confirmation-v1",
        "source_run": original["run"],
        "source_summary_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "source_contract": original["contract"]["fingerprint"],
        "candidate": candidate,
        "selection": "highest prior perturbed captures, then clean captures, nominal captures, ID tie break",
        "trials": 100,
        "perturbation_index_start": 10000,
        "scope": "same bounded H +/-2mm, G +/-2mm, tilt +/-0.5deg; fresh seeds; no retuning or material/perception noise",
        "source_sha256": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in Path(__file__).parent.glob("*.py")
        },
    }
    atomic(dest / "contract.json", spec)
    a = copy.copy(args)
    a.output = str(dest)
    env = DiagnosticShot(
        sim, arms, ball, camera, contact, copy.deepcopy(cfg), poses, a, atomic
    )
    rows = []
    started = time.monotonic()
    probes = []

    def publish(phase):
        d = {
            "phase": phase,
            "run": dest.name,
            "timestamp_utc": datetime.datetime.now(datetime.UTC).isoformat(),
            "contract": spec,
            "rows": rows,
            "summary": summary(rows),
            "elapsed_s": time.monotonic() - started,
            "probes": probes,
        }
        atomic(dest / "summary.json", d)
        atomic(out / "catalogue-confirmation.json", d)
        return d

    try:
        publish("contact_probes")
        probes = contact_probe(env)
        for i in range(100):
            if shutil.disk_usage(dest).free < 3 * 1024**3:
                raise RuntimeError("Free space below3GiB")
            c = perturb(candidate, 10000 + i)
            result = env.scene_shot(scene(c), np.array(action(c)))
            row = compact(c, result)
            row["trial_id"] = f"{dest.name}-{i + 1:04d}"
            for k in (
                "quality_score",
                "entry_class",
                "bucket_contact_before_entry",
                "rim_band_contact",
                "exited_after_entry",
                "minimum_mouth_clearance_mm",
            ):
                row[k] = result.get(k, 0 if k == "quality_score" else None)
            rows.append(row)
            with (dest / "trials.jsonl").open("a") as f:
                f.write(
                    json.dumps(
                        {
                            "trial_id": row["trial_id"],
                            "candidate": c,
                            "result": result,
                            "trajectory_m": env.last_trajectory,
                        },
                        allow_nan=False,
                    )
                    + "\n"
                )
            publish("confirming")
        d = publish("completed")
        d["artifact_sha256"] = {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in dest.iterdir()
            if p.is_file() and p.name != "summary.json"
        }
        atomic(dest / "summary.json", d)
        atomic(out / "catalogue-confirmation.json", d)
        print("INDEPENDENT_CONFIRMATION", summary(rows), flush=True)
    except BaseException:
        publish("failed")
        raise
