"""Sweep the saved (100, -650) mm capture in the audited native simulator."""
from __future__ import annotations

import argparse
from collections import Counter
import importlib.metadata
from itertools import product
import json
from pathlib import Path
import signal
import shutil
import time

from .benchmark_verify import digest, read, sha
from .corrected_imitation_study import mirror_control_proofs
from .corrected_grid import source_manifest
from .coverage_training_study import append, share_native_environment
from .fixed35_recovery_contract import HIGH, LOW, request
from .fixed35_recovery_native import RecoveryEngine
from .gpu_slot import GPU_UUID, foreign_processes, gpu_slot
from .policy64_contract import load_runtime_reference
from .sweep_worker import atomic
from .table_correction_native import META, MatchedEngine
from .table_training_contract import captured, statistics
from .table_training_study import control_gate

CONTEXT = "fixed35-dev-grid-00-00"
PARAMETERS = ("board_x", "board_y", "height", "tilt", "gap")
PARENT_FILES = ("report.json", "rows.jsonl", "evidence-manifest.json", "contract.json")


def plan_search(seed):
    offsets = []
    for axis, parameter in enumerate(PARAMETERS):
        magnitudes = (0.5, 1.5, 3.0) if parameter == "tilt" else (0.005, 0.015, 0.030)
        for magnitude, sign in product(magnitudes, (-1, 1)):
            offset = [0.0] * 5
            offset[axis] = sign * magnitude
            offsets.append((parameter, offset))
    for tilt, gap in product((-3.0, -1.0, 1.0, 3.0), (-0.025, -0.010, 0.010, 0.025)):
        offsets.append(("tilt_gap", [0.0, 0.0, 0.0, tilt, gap]))
    plan = []
    for index, (family, offset) in enumerate(offsets):
        values = [value + delta for value, delta in zip(seed["values"], offset)]
        if any(not low <= value <= high for low, value, high in zip(LOW, values, HIGH)):
            raise ValueError("A planned corner offset exceeds the audited action bounds")
        trial = request(seed, identity=f"corner-search-{index:02d}", phase="search",
                        method="corner_sweep", values=values, family=family)
        trial["perturbation_mixed_units"] = offset
        plan.append(trial)
    return plan


def run(args):
    parent = args.parent_study.resolve()
    output = args.output.resolve()
    started = time.monotonic()
    audit = read(args.parent_audit)
    hashes = {name: sha(parent / name) for name in PARENT_FILES}
    hashes["audit"] = sha(args.parent_audit)
    if (audit["status"] != "passed" or audit["full_checks_verified"] is not True
            or audit["native_checks_verified"] is not True
            or audit["report_sha256"] != hashes["report.json"]
            or audit["evidence_manifest_sha256"] != hashes["evidence-manifest.json"]):
        raise ValueError("A full audit bound to the exact completed parent is required")
    manifest = read(parent / "evidence-manifest.json")
    for name in ("rows.jsonl", "contract.json", "calibration.json", "verification.json", "setup-home.json",
                 "prior.json", "old-prior.json", "control-replays.json",
                 "native-yaw/effective-options.json", "native-yaw/native-compiled-geometry.json"):
        if sha(parent / name) != manifest[name]:
            raise ValueError(f"Parent evidence changed: {name}")
    spec = read(parent / "contract.json")
    if audit["contract_fingerprint"] != spec["fingerprint"]:
        raise ValueError("Parent contract differs from the audited contract")
    with (parent / "rows.jsonl").open() as stream:
        seeds = [row for row in map(json.loads, stream)
                 if row["context_id"] == CONTEXT and row["phase"] == "preserved" and captured(row)]
    if len(seeds) != 1 or seeds[0]["bucket_xy_m"] != [0.1, -0.65]:
        raise ValueError("Expected exactly one preserved capture at the corner")
    seed = seeds[0]
    root = Path(__file__).resolve().parents[2]
    ref = load_runtime_reference(args.reference, args.assets, root)
    if (digest(ref["scene"]) != spec["identity"]["scene_sha256"]
            or ref["asset_hashes"] != spec["identity"]["asset_sha256"]
            or any(importlib.metadata.version(name) != version for name, version in spec["packages"].items())):
        raise ValueError("Scene, assets or packages differ from the audited parent")

    sources = read(parent / "control-replays.json")
    controls = [
        dict({key: source[key] for key in META}, id=f"corner-{phase}-{source['role']}-{i}",
             phase=phase, method="control", role=source["role"], context_id=source["context_id"],
             bucket_xy_m=source["bucket_xy_m"], values=source["values"], command_values=source["values"],
             source_record_id=source["id"], source_record_sha256=digest(source),
             variant="control_exact", repeat=i)
        for phase in ("controls_pre", "controls_post") for source in sources for i in range(3)
    ]
    seed_trials = [
        request(seed, identity=f"corner-{phase}-{i}", phase=phase, method="seed_replay",
                exact_parent_action=True)
        for phase in ("seed_pre", "seed_post") for i in range(3)
    ]
    search = plan_search(seed)
    output.mkdir(parents=True, exist_ok=False)
    (output / "requests").mkdir()
    calibration = read(parent / "calibration.json")
    home = read(parent / "setup-home.json")
    for name in ("calibration.json", "verification.json"):
        shutil.copyfile(parent / name, output / name)
    atomic(output / "plan.json", dict(seed=seed_trials, search=search, controls=controls,
                                     repeat_count=3, repeat_limit=6))
    contract = dict(
        version="corner-sweep-v1", source_sha256=source_manifest(root),
        parent_sha256=hashes, parent_seed_id=seed["id"], parent_seed_sha256=digest(seed),
        plan_sha256=sha(output / "plan.json"), scene=spec["scene"], identity=spec["identity"],
        wall_time_limit_s=args.wall_time_limit, substeps=64, bucket_xy_m=seed["bucket_xy_m"],
    )
    contract["fingerprint"] = digest(contract)
    atomic(output / "contract.json", contract)
    engine = RecoveryEngine(ref, args.assets, output / "native-yaw", parent=parent,
                            sources=seeds, prior=read(parent / "prior.json"),
                            calibration=calibration, home=home)
    mirror = mirror_control_proofs(parent, output, sources)
    control_engine = MatchedEngine(ref, args.assets, output / "native-controls",
                                  parent=mirror, sources=sources, prior=read(parent / "old-prior.json"),
                                  calibration=calibration, home=home)
    share_native_environment(engine, control_engine)
    rows = []
    report = {"status": "running", "confirmed": [], "repeat_results": []}
    packet_count = 0

    def publish(phase):
        report.update(phase=phase, elapsed_s=time.monotonic() - started)
        for name in ("seed_pre", "seed_post", "search", "repeat"):
            report[name] = statistics([row for row in rows if row["phase"] == name])
        atomic(output / "report.json", report)
        print(json.dumps(dict(phase=phase, search=report["search"], elapsed_s=report["elapsed_s"])), flush=True)

    def execute(trials, phase):
        nonlocal packet_count
        if foreign_processes():
            raise RuntimeError("Another workload is using the 4090")
        packet = f"requests/batch-{packet_count:03d}.json"
        packet_count += 1
        atomic(output / packet, dict(phase=phase, rows=trials))
        control = phase.startswith("controls_")
        measured, _ = (control_engine if control else engine).evaluate(trials, phase)
        if [row["id"] for row in measured] != [trial["id"] for trial in trials]:
            raise RuntimeError("Native trial identities or order changed")
        for row in measured:
            row.update(request_packet=packet, request_packet_sha256=sha(output / packet),
                       adapter_directory="native-controls" if control else "native-yaw")
        append(output / "rows.jsonl", measured)
        rows.extend(measured)
        publish(phase)
        return measured

    def run_controls(phase):
        report[phase] = control_gate(execute([r for r in controls if r["phase"] == phase], phase))
        if report[phase]["status"] != "passed":
            raise RuntimeError(f"{phase} failed")

    def run_seed(phase):
        outcomes = execute([r for r in seed_trials if r["phase"] == phase], phase)
        if not all(captured(row) for row in outcomes):
            raise RuntimeError(f"The saved corner shot failed during {phase}")

    run_controls("controls_pre")
    run_seed("seed_pre")
    successes = []
    for offset in range(0, len(search), 8):
        successes.extend(row for row in execute(search[offset:offset + 8], "search") if captured(row))
    # Select the first capture per offset family, then fill the remaining slots in plan order.
    families = set()
    diverse = []
    for row in successes:
        if row["family"] not in families:
            diverse.append(row)
            families.add(row["family"])
    selected = (diverse + [row for row in successes if row not in diverse])[:6]
    repeats = [
        request(seed, identity=f"{row['id']}-repeat-{i}", phase="repeat",
                method="corner_repeat", values=row["values"], selected_from_id=row["id"],
                selected_from_sha256=digest(row), selected_q10=row["q10"], selected_q14=row["q14"],
                selected_connection_sha256=row["connection_sha256"], repeat=i)
        for row in selected for i in range(3)
    ]
    atomic(output / "repeat-plan.json", dict(rows=repeats, selected_from_ids=[row["id"] for row in selected]))
    repeat_rows = execute(repeats, "repeat") if repeats else []
    counts = Counter(r["selected_from_id"] for r in repeat_rows if captured(r))
    report["repeat_results"] = [
        dict(selected_from_id=row["id"], values=row["values"], captures=counts[row["id"]], requests=3)
        for row in selected
    ]
    report["confirmed"] = [result for result in report["repeat_results"] if result["captures"] == 3]
    run_seed("seed_post")
    run_controls("controls_post")
    for name in PARENT_FILES:
        if sha(parent / name) != hashes[name]:
            raise RuntimeError(f"Parent evidence changed during the sweep: {name}")
    atomic(output / "evidence-manifest.json", {
        str(path.relative_to(output)): sha(path) for path in sorted(output.rglob("*"))
        if path.is_file() and path.name not in ("report.json", "evidence-manifest.json")
    })
    report.update(status="completed", evidence_manifest_sha256=sha(output / "evidence-manifest.json"))
    publish("completed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("parent-study", "parent-audit", "reference", "assets", "output", "gpu-lock"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--wall-time-limit", type=int, default=1200)
    args = parser.parse_args()
    if args.output.exists() or not 60 <= args.wall_time_limit <= 1800:
        parser.error("Use a fresh output directory and a finite 60–1800 second worker limit")

    def expire(_signal, _frame):
        raise TimeoutError("Corner sweep reached its worker limit")

    with gpu_slot(args.gpu_lock, timeout_s=900, notify=lambda value: print(json.dumps(value), flush=True)):
        signal.signal(signal.SIGALRM, expire)
        signal.alarm(args.wall_time_limit)
        try:
            run(args)
        finally:
            signal.alarm(0)


if __name__ == "__main__":
    main()
