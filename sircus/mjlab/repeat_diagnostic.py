"""Bounded native repeatability diagnostic; never promotes teachers or policies."""

import argparse
from collections import Counter
import datetime
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from .learning_contract import digest, statistics

VERSION = "sircus-mjlab-native-repeat-diagnostic-v1"
CAPTURED = (10, 19, 21, 26, 39, 48)
EXTERNAL = (8, 9, 12, 23, 35, 41)
SEED = 20261119


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic(path, value):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def select_sources(rows, targets=CAPTURED, *, external=False):
    """Freeze native search sources by observed mouth margin, never a new label."""
    chosen = []
    for index in targets:
        target = f"old-test-{index:02d}"
        candidates = [
            r
            for r in rows
            if r["target_id"] == target
            and r["phase"] == "search"
            and not r["invalid"]
            and (r["success"] if external else r["route_success"])
        ]
        if not candidates:
            raise ValueError("No qualifying native source for " + target)
        chosen.append(
            max(
                candidates,
                key=lambda r: (
                    (
                        r.get("minimum_mouth_clearance_mm")
                        if r.get("minimum_mouth_clearance_mm") is not None
                        else -1e9
                    ),
                    r["route_quality"],
                    r["max_retention_s"],
                ),
            )
        )
    return chosen


def plans(captured, external, source_contract):
    """Five reset batches: six cached, six fresh, twelve jittered shots each."""
    rng = np.random.default_rng(SEED)
    result = []
    for repeat in range(5):
        for mode, repetitions in (
            ("nominal_cached", 1),
            ("nominal_fresh_ik", 1),
            ("jitter_fresh_ik", 2),
        ):
            for source in captured:
                for local in range(repetitions):
                    value = np.asarray(source["values"], dtype=float)
                    bounds = source_contract[source["bound_phase"] + "_bounds"]
                    if mode == "jitter_fresh_ik":
                        value = np.clip(
                            value
                            + rng.normal(0, source_contract["confirmation_noise_std"]),
                            bounds["low"],
                            bounds["high"],
                        )
                    result.append(
                        dict(
                            source_id=source["id"],
                            target_id=source["target_id"],
                            mode=mode,
                            repeat=repeat,
                            local_repeat=local,
                            reset_batch=repeat,
                            bound_phase=source["bound_phase"],
                            bucket_xy_m=source["bucket_xy_m"],
                            requested_values=value.tolist(),
                            source_q10=source["q10"],
                            source_q14=source["q14"],
                        )
                    )
    for repeat in range(2):
        for source in external:
            result.append(
                dict(
                    source_id=source["id"],
                    target_id=source["target_id"],
                    mode="external_nominal_cached",
                    repeat=repeat,
                    local_repeat=0,
                    reset_batch=5 + repeat,
                    bound_phase=source["bound_phase"],
                    bucket_xy_m=source["bucket_xy_m"],
                    requested_values=source["values"],
                    source_q10=source["q10"],
                    source_q14=source["q14"],
                )
            )
    return result


def branch_delta(first, second):
    raw = np.asarray(first) - np.asarray(second)
    wrapped = (raw + np.pi) % (2 * np.pi) - np.pi
    return dict(
        max_abs_rad=float(np.abs(raw).max()),
        max_wrapped_rad=float(np.abs(wrapped).max()),
        wrapped_l2_rad=float(np.linalg.norm(wrapped)),
    )


def contact_category(row):
    if row["invalid"]:
        return "geometry_rejected"
    if row["external_contact_steps"]:
        peaks = row.get("contact_peaks_N", {})
        names = [
            k
            for k in ("table", "floor", "robot", "unattributed")
            if peaks.get(k, 0) > 0.01
        ]
        return "external_" + ("_".join(names) or "unspecified")
    if row["route_success"]:
        return "clean_capture" if row["route_quality"] == 2 else "assisted_capture"
    if not row["board_bounce"]:
        return "no_board_bounce"
    return "retention_miss" if row["whole_ball_entry"] else "opening_miss"


class ContactRecorder:
    """Read-only observer after the last sensor update; control math is untouched."""

    def __init__(self, environment):
        self.environment = environment
        self.tick = 0
        self.force = []
        self.position = []
        self.pending = []

    def reset(self):
        self.tick = 0
        self.force = []
        self.position = []
        self.pending = []

    def sample(self):
        import torch

        e = self.environment
        vectors = [s.data.force for s in e.sensors]
        norms = [torch.linalg.vector_norm(v, dim=-1).sum(1) for v in vectors[:5]]
        residual = torch.linalg.vector_norm(
            vectors[-1].sum(1) + sum(v.sum(1) for v in vectors[:5]), dim=-1
        )
        self.pending.append(torch.stack([*norms, residual], -1).clone())
        self.tick += 1
        if self.tick % 8 == 0:
            self.force.append(torch.stack(self.pending).max(0).values)
            self.position.append(e.sim.data.qpos[:, e.ball_q : e.ball_q + 3].clone())
            self.pending = []

    def arrays(self):
        import torch

        return (
            torch.stack(self.force).cpu().numpy(),
            torch.stack(self.position).cpu().numpy(),
        )


class ObservedSensor:
    def __init__(self, sensor, recorder):
        self.sensor, self.recorder = sensor, recorder

    def __getattr__(self, name):
        return getattr(self.sensor, name)

    def reset(self):
        self.sensor.reset()
        self.recorder.reset()

    def update(self, dt):
        self.sensor.update(dt)
        self.recorder.sample()


def run(a):
    from .learning_solve import validate_artifacts

    _, parent_cp, _, _ = validate_artifacts(a.parent_study, a.assets)
    source_contract = json.loads((a.source_study / "contract.json").read_text())
    source_report = json.loads((a.source_study / "report.json").read_text())
    if (
        source_report.get("status") != "completed"
        or source_contract["scene"] != parent_cp["contract"]["scene"]
    ):
        raise RuntimeError("Completed source and unchanged scene required")
    previous = [
        json.loads(line)
        for line in (a.source_study / "shots.jsonl").read_text().splitlines()
    ]
    captured = select_sources(previous)
    external = select_sources(previous, EXTERNAL, external=True)
    sources = captured + external
    plan = plans(captured, external, source_contract)
    scene = source_contract["scene"]
    from .expanded_actions import WIDE_BOUNDS

    spec = dict(
        version=VERSION,
        seed=SEED,
        scene=scene,
        source_contract_fingerprint=source_contract["fingerprint"],
        source_report_sha256=sha(a.source_study / "report.json"),
        source_records_sha256=sha(a.source_study / "shots.jsonl"),
        action_bounds=WIDE_BOUNDS,
        selected_source_ids=[s["id"] for s in sources],
        confirmation_noise_std=source_contract["confirmation_noise_std"],
        protocol="Six high-mouth-margin route captures: 5 nominal cached-branch, 5 nominal fresh CPU IK, 10 perturbed fresh CPU IK each. Six external-contact entry controls: 2 nominal cached-branch each. Five independent native resets plus two external-control resets. Rejections remain in denominator.",
        contact_observation="Read-only last-sensor observer, max force over eight native substeps and ball XYZ at 240Hz; no force, control, scoring or endpoint gate is changed.",
        scope="Development diagnostic; no teacher promotion, new training or held-out evidence",
    )
    spec["fingerprint"] = digest(spec)
    a.output.mkdir(parents=True, exist_ok=False)
    atomic(a.output / "contract.json", spec)
    atomic(a.output / "sources.json", sources)
    atomic(a.output / "plan.json", plan)
    source_hashes = {p.name: sha(p) for p in Path(__file__).parent.glob("*.py")}
    report = dict(
        status="running",
        phase="cpu-geometry",
        contract_fingerprint=spec["fingerprint"],
        records=0,
    )
    started = time.monotonic()

    def publish(phase):
        report.update(
            phase=phase,
            elapsed_s=time.monotonic() - started,
            timestamp_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        )
        atomic(a.output / "report.json", report)
        print(
            json.dumps({k: report[k] for k in ("phase", "records", "elapsed_s")}),
            flush=True,
        )

    publish("cpu-geometry")
    n = len(plan)
    eligible = np.ones(n, dtype=bool)
    q10 = np.asarray([p["source_q10"] for p in plan])
    q14 = np.asarray([p["source_q14"] for p in plan])
    reasons = np.zeros(n, dtype=np.uint8)
    from .expand_geometry import audit_values

    ids = np.asarray(
        [
            i
            for i, p in enumerate(plan)
            if p["mode"] in ("nominal_fresh_ik", "jitter_fresh_ik")
        ]
    )
    # This is fresh endpoint IK, not trajectory search or a successful-outcome label.
    mask, detail = audit_values(
        scene,
        a.assets,
        np.asarray([plan[i]["bucket_xy_m"] for i in ids]),
        np.asarray([plan[i]["requested_values"] for i in ids])[:, None, :],
        bounds=WIDE_BOUNDS,
    )
    eligible[ids] = mask[:, 0]
    q10[ids] = detail["q10"][:, 0]
    q14[ids] = detail["q14"][:, 0]
    reasons[ids] = detail["reasons"][:, 0]
    np.savez_compressed(
        a.output / "geometry.npz", eligible=eligible, q10=q10, q14=q14, reasons=reasons
    )
    from .gpu_slot import gpu_slot
    from .bank_study import run_bank_rows

    with gpu_slot(
        a.gpu_lock, timeout_s=1800, notify=lambda info: publish("waiting-for-4090")
    ):
        import torch
        from .expanded_env import ExpandedEnvironment

        torch.set_num_threads(2)
        env = ExpandedEnvironment(scene, a.assets, n=24, bounds=WIDE_BOUNDS)
        recorder = ContactRecorder(env)
        env.sensors[-1] = ObservedSensor(env.sensors[-1], recorder)
        all_rows = []
        by_source = {s["id"]: s for s in sources}
        for batch in range(7):
            packet = np.asarray(
                [i for i, p in enumerate(plan) if p["reset_batch"] == batch]
            )
            xy = np.asarray([plan[i]["bucket_xy_m"] for i in packet])
            values = np.asarray([plan[i]["requested_values"] for i in packet])
            rows = run_bank_rows(
                env, xy, values, q10[packet], q14[packet], eligible[packet], record=True
            )
            force, position = recorder.arrays()
            native_ids = np.flatnonzero(eligible[packet])
            np.savez_compressed(
                a.output / f"contact-batch-{batch:02d}.npz",
                forces_N=force[:, : len(native_ids)],
                ball_position_m=position[:, : len(native_ids)],
                plan_indices=packet[native_ids],
                dt_s=np.asarray(1 / 240),
            )
            for j, (index, row) in enumerate(zip(packet, rows)):
                p = plan[index]
                source = by_source[p["source_id"]]
                row.update(
                    p,
                    id=f"repeat-shot-{len(all_rows):05d}",
                    geometry_reason=int(reasons[index]),
                    source_q10_delta=branch_delta(q10[index], source["q10"]),
                    source_q14_delta=branch_delta(q14[index], source["q14"]),
                    category=contact_category(row),
                )
                if row["native_release"]:
                    native_index = int(np.flatnonzero(native_ids == j)[0])
                    timeline = []
                    # Contact data after terminal scoring is excluded explicitly.
                    for step in range(min(len(force), int(row["physics_steps"]))):
                        names = [
                            name
                            for name, value in zip(
                                (
                                    "board",
                                    "bucket",
                                    "table",
                                    "floor",
                                    "robot",
                                    "unattributed",
                                ),
                                force[step, native_index],
                            )
                            if value > 0.01
                        ]
                        if names:
                            timeline.append(
                                dict(
                                    time_s=(step + 1) / 240,
                                    ball_position_m=position[
                                        step, native_index
                                    ].tolist(),
                                    forces_N={
                                        name: float(value)
                                        for name, value in zip(
                                            (
                                                "board",
                                                "bucket",
                                                "table",
                                                "floor",
                                                "robot",
                                                "unattributed",
                                            ),
                                            force[step, native_index],
                                        )
                                        if value > 0.01
                                    },
                                )
                            )
                    row["contact_timeline"] = timeline
                all_rows.append(row)
            with (a.output / "shots.jsonl").open("a") as stream:
                for row in rows:
                    stream.write(json.dumps(row, allow_nan=False) + "\n")
            report["records"] = len(all_rows)
            publish("native-repeat")
        groups = {}
        for row in all_rows:
            groups.setdefault((row["target_id"], row["mode"]), []).append(row)
        report["results"] = [
            dict(
                target_id=target,
                mode=mode,
                **statistics(rr),
                native_releases=sum(r["native_release"] for r in rr),
                categories=dict(Counter(r["category"] for r in rr)),
                max_source_q10_wrapped_rad=max(
                    r["source_q10_delta"]["max_wrapped_rad"] for r in rr
                ),
                max_source_q14_wrapped_rad=max(
                    r["source_q14_delta"]["max_wrapped_rad"] for r in rr
                ),
            )
            for (target, mode), rr in groups.items()
        ]
    if any(
        sha(Path(__file__).parent / name) != value
        for name, value in source_hashes.items()
    ):
        raise RuntimeError("Diagnostic executed source changed while running")
    report.update(
        status="completed",
        source_sha256=source_hashes,
        artifacts_sha256={
            p.name: sha(p)
            for p in a.output.iterdir()
            if p.is_file() and p.name != "report.json"
        },
    )
    publish("completed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source-study", "parent-study", "assets", "output", "gpu-lock"):
        parser.add_argument("--" + name, type=Path, required=True)
    run(parser.parse_args())
