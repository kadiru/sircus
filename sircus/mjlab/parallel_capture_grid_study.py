"""Search underrepresented bucket locations in mixed native simulation batches."""
from __future__ import annotations

import argparse
from collections import Counter, deque
import importlib.metadata
import json
import math
from pathlib import Path
import shutil
import time

from .benchmark_verify import digest, read, sha
from .cell1050_followup_plan import geometry_key
from .cell1050_rebound_study import annotate, control_requests, parent_inputs
from .corrected_grid import source_manifest
from .corrected_imitation_study import mirror_control_proofs
from .coverage_training_study import append, share_native_environment
from .gpu_slot import foreign_processes, gpu_slot
from .parallel_capture_grid_workers import load_resolution, open_workers, prepare_proposals
from .policy64_contract import load_runtime_reference
from .sweep_worker import atomic
from .table_training_study import control_gate

VERSION = "parallel-capture-grid-v1"
NATIVE_BATCH = 32
STORAGE_RESERVE_BYTES = 8 * 1024**3
SCHEDULER_FILES = {
    "sircus/mjlab/parallel_capture_grid_study.py",
    "sircus/mjlab/parallel_capture_grid_workers.py",
    "sircus/mjlab/parallel_capture_grid_resume.py",
}


def take_packet(pending, order, limit):
    """Round-robin over the ready layouts, carrying the starting layout forward."""
    packet = []
    while len(packet) < limit and any(pending.values()):
        cell = order[0]
        order.rotate(-1)
        if pending[cell]:
            packet.append(pending[cell].popleft())
    return packet


def frozen_plan(args, output, spec, binding, source):
    from .parallel_capture_grid_plan import load_grid_plan

    current_source = source_manifest(source)
    if getattr(args, "resume", False):
        contract = read(output / "contract.json")
        if contract["fingerprint"] != digest({k: v for k, v in contract.items() if k != "fingerprint"}):
            raise ValueError("Frozen grid contract fingerprint changed")
        expected = dict(version=VERSION, target_per_cell=args.target_per_cell,
                        native_batch=NATIVE_BATCH, generator_seed=args.seed,
                        preparation_batch=args.prepare_batch, parent=binding,
                        identity=spec["identity"], scene=spec["scene"])
        if any(contract[key] != value for key, value in expected.items()):
            raise ValueError("Continuation must keep the original study settings")
        if any(sha(output / name) != contract[key] for name, key in (
            ("plan.json", "plan_sha256"), ("historical-geometries.json", "history_sha256"),
        )):
            raise ValueError("Frozen grid plan or history changed")
        original_source = contract["source_sha256"]
        changed = {name for name in original_source.keys() | current_source.keys()
                   if original_source.get(name) != current_source.get(name)}
        if changed - SCHEDULER_FILES:
            raise ValueError("Continuation changed source outside the CPU scheduler")
        plan, history = read(output / "plan.json"), read(output / "historical-geometries.json")
        if ({cell: data["bucket_xy_m"] for cell, data in plan.items()} != contract["cells"]
                or set(history) != set(plan)):
            raise ValueError("Frozen grid plan lost a bucket location")
        return {cell: dict(data, excluded_geometry=history[cell]) for cell, data in plan.items()}, history

    cells, source_bindings = load_grid_plan(args.history_root)
    if len(cells) != 13:
        raise ValueError(f"Expected the 13 underrepresented bucket locations, found {len(cells)}")
    output.mkdir(parents=True, exist_ok=False)
    for name in ("calibration.json", "verification.json"):
        shutil.copyfile(args.parent_study / name, output / name)
    historical = {cell: sorted(data["excluded_geometry"]) for cell, data in cells.items()}
    plan = {cell: {key: value for key, value in data.items() if key != "excluded_geometry"}
            for cell, data in cells.items()}
    atomic(output / "historical-geometries.json", historical)
    atomic(output / "plan.json", plan)
    contract = dict(
        version=VERSION, target_per_cell=args.target_per_cell, native_batch=NATIVE_BATCH,
        cells={cell: data["bucket_xy_m"] for cell, data in cells.items()},
        history_sha256=sha(output / "historical-geometries.json"),
        plan_sha256=sha(output / "plan.json"), source_bindings=source_bindings,
        generator_seed=args.seed, preparation_batch=args.prepare_batch,
        parent=binding, identity=spec["identity"], scene=spec["scene"], source_sha256=current_source,
    )
    contract["fingerprint"] = digest(contract)
    atomic(output / "contract.json", contract)
    return cells, historical


def run(args):
    from .cell1050_heading_native import HeadingEngine
    from .parallel_capture_grid_plan import cell_candidates
    from .table_correction_native import MatchedEngine

    started = time.monotonic()
    output = args.output.resolve()
    parent = args.parent_study.resolve()
    spec, binding, anchor = parent_inputs(parent, args.parent_audit)
    source = Path(__file__).resolve().parents[2]
    ref = load_runtime_reference(args.reference, args.assets, source)
    if (digest(ref["scene"]) != spec["identity"]["scene_sha256"]
            or ref["asset_hashes"] != spec["identity"]["asset_sha256"]
            or any(importlib.metadata.version(name) != version for name, version in spec["packages"].items())):
        raise ValueError("Scene, assets or pinned dependencies differ from the audited parent")

    print(json.dumps({"status": "loading_archived_shots"}), flush=True)
    cells, historical = frozen_plan(args, output, spec, binding, source)
    cpu_workers = getattr(args, "cpu_workers", 1)

    summaries = {
        cell: dict(bucket_xy_m=data["bucket_xy_m"], prior_count=data["prior_count"],
                   search_releases=0, attempts=0, preparation_declines=0, route_counts={})
        for cell, data in cells.items()
    }
    report = dict(status="running", controls={}, cells=summaries,
                  target_per_cell=args.target_per_cell, search_releases=0,
                  search_packets=0, mixed_packets=0, max_cells_in_packet=0)
    order = deque(cells)
    pending = {cell: deque() for cell in cells}
    seen = {cell: set(map(tuple, keys)) for cell, keys in historical.items()}
    outcomes = {cell: Counter() for cell in cells}
    dry_preparations = Counter()
    generators = {cell: iter(cell_candidates(data, seed=args.seed))
                  for cell, data in cells.items()}
    packets = 0
    drafts = []
    elapsed_before_resume = 0.0
    if getattr(args, "resume", False):
        from .parallel_capture_grid_resume import recover

        packets, drafts = recover(output, report, summaries, outcomes, pending, order, generators,
                                  seen, dry_preparations, geometry_key)
        elapsed_before_resume = read(output / "report.json")["elapsed_s"]
        append(output / "continuations.jsonl", [
            dict(starting_releases=report["search_releases"], cpu_workers=cpu_workers,
                 previous_report_sha256=sha(output / "report.json"),
                 source_sha256=source_manifest(source))
        ])
    report["cpu_workers"] = cpu_workers

    def publish(phase):
        report.update(phase=phase, elapsed_s=round(elapsed_before_resume + time.monotonic() - started, 3))
        for cell, counts in outcomes.items():
            summaries[cell]["route_counts"] = dict(counts)
        atomic(output / "report.json", report)
        print(json.dumps(dict(status=report["status"], phase=phase,
                              search_releases=report["search_releases"], mixed_packets=report["mixed_packets"],
                              elapsed_s=report["elapsed_s"])), flush=True)

    def execute(requests, phase, *, control=False):
        nonlocal packets
        if foreign_processes():
            raise RuntimeError("Another process is using the leased 4090")
        if shutil.disk_usage(output).free < STORAGE_RESERVE_BYTES:
            raise RuntimeError("Less than 8 GiB remains for immutable simulation evidence")
        directory = "native-controls" if control else "native-heading"
        packet = f"requests/batch-{packets:05d}.json"
        packets += 1
        atomic(output / packet, dict(phase=phase, rows=requests))
        used_engine = controls if control else engine
        measured, timing = used_engine.evaluate(requests, phase)
        if [row["id"] for row in measured] != [row["id"] for row in requests]:
            raise RuntimeError("Native trial IDs changed")
        if not all(row["native_release"] is True for row in measured):
            raise RuntimeError("A prepared drop did not release")
        adapter = controls.engine.adapter if control else engine.adapter
        if adapter.env.effective_options != expected_options or adapter.env.compiled_geometry != expected_geometry:
            raise RuntimeError("Native physics options or collision geometry changed")
        if not control:
            annotate(measured, output / directory)
        packet_hash = sha(output / packet)
        for row in measured:
            row.update(request_packet=packet, request_packet_sha256=packet_hash, adapter_directory=directory)
        append(output / "rows.jsonl", measured)
        append(output / "timing.jsonl", [dict(packet=packet, phase=phase, timing=timing)])
        return measured

    def run_controls(phase):
        gate = control_gate(execute(control_requests(control_sources, phase), phase, control=True))
        report["controls"][phase] = gate
        if gate["status"] != "passed":
            raise RuntimeError(f"{phase} failed")
        publish(phase)

    def reserve(cell, count):
        summary = summaries[cell]
        proposals = []
        for _ in range(count):
            candidate = next(generators[cell])
            if candidate["bucket_xy_m"] != cells[cell]["bucket_xy_m"]:
                raise ValueError("Candidate changed the fixed bucket location")
            key = geometry_key(candidate)
            if key in seen[cell]:
                raise ValueError("Candidate repeated a historical or already proposed geometry")
            seen[cell].add(key)
            number = summary["attempts"]
            summary["attempts"] += 1
            proposals.append(dict(candidate, id=f"parallel-grid-{cell[-5:]}-{number:06d}",
                                  phase="search", role="development", context_id=cell,
                                  method="parallel_capture_grid"))
        filename = output / "preparations" / f"batch-{proposals[0]['id']}.json"
        warm = dry_preparations[cell] < 2
        atomic(filename, dict(proposals=proposals, warm_start=warm))
        return filename, proposals, warm

    def register(job, result):
        filename, proposals, warm = job
        cell = proposals[0]["context_id"]
        admitted, records = result
        atomic(filename, dict(proposals=proposals, records=records, admitted=admitted, warm_start=warm))
        summaries[cell]["preparation_declines"] += sum(not row["admitted_request_ids"] for row in records)
        first_branches = {}
        for row in admitted:
            first_branches.setdefault(row["proposal_id"], row)
        pending[cell].extend(first_branches.values())
        dry_preparations[cell] = 0 if first_branches else dry_preparations[cell] + 1
        report["last_prepared_cell"] = cell

    def release(packet):
        packet_cells = {row["context_id"] for row in packet}
        if cpu_workers > 1 or getattr(args, "resume", False):
            for cell in packet_cells:
                load_resolution(engine, cells[cell]["bucket_xy_m"])
        measured = execute(packet, "search")
        report["search_packets"] += 1
        report["mixed_packets"] += int(len(packet_cells) > 1)
        report["max_cells_in_packet"] = max(report["max_cells_in_packet"], len(packet_cells))
        for row in measured:
            cell = row["context_id"]
            outcomes[cell][row["experimental_route"]["route_class"]] += 1
            summaries[cell]["search_releases"] += 1
        report["search_releases"] += len(measured)
        publish("search")

    def next_jobs(reserved):
        remaining = {cell: args.target_per_cell - summaries[cell]["search_releases"]
                     - len(pending[cell]) - reserved[cell] for cell in order}
        active = [cell for cell, count in remaining.items() if count > 0]
        if not active:
            return []
        capacity = max(args.prepare_batch, math.ceil(NATIVE_BATCH / len(active)))
        jobs = []
        for cell in active:
            count = min(args.prepare_batch, capacity - len(pending[cell]), remaining[cell])
            if count > 0:
                jobs.append(reserve(cell, count))
        return jobs

    try:
        publish("initializing")
        calibration, home = read(parent / "calibration.json"), read(parent / "setup-home.json")
        engine_kwargs = dict(prior=read(parent / "prior.json"), calibration=calibration, home=home)
        engine = HeadingEngine(ref, args.assets, output / "native-heading", **engine_kwargs, anchor_source=anchor,
                               anchor_directory=parent / "native-yaw")
        control_sources = read(parent / "control-replays.json")
        mirror = mirror_control_proofs(parent, output, control_sources)
        controls = MatchedEngine(ref, args.assets, output / "native-controls", parent=mirror,
                                 sources=control_sources, prior=read(parent / "old-prior.json"),
                                 calibration=calibration, home=home)
        share_native_environment(engine, controls)
        expected_options = read(parent / "native-yaw/effective-options.json")
        expected_geometry = read(parent / "native-yaw/native-compiled-geometry.json")
        if getattr(args, "resume", False):
            for adapter in (engine.adapter, controls.engine.adapter):
                adapter.packet = len(list(adapter.directory.glob("native-*.npz")))
        if "controls_pre" not in report["controls"]:
            run_controls("controls_pre")

        target = len(cells) * args.target_per_cell
        if cpu_workers == 1:
            jobs = drafts
            while report["search_releases"] < target:
                for job in jobs or next_jobs(Counter()):
                    register(job, engine.prepare(job[1], one_per_setting=job[2]))
                jobs = []
                publish("preparing")
                ready = sum(map(len, pending.values()))
                while ready >= NATIVE_BATCH:
                    release(take_packet(pending, order, NATIVE_BATCH))
                    ready -= NATIVE_BATCH
                if ready and report["search_releases"] + ready == target:
                    release(take_packet(pending, order, ready))
        else:
            with open_workers(cpu_workers, ref, args.assets, output / "native-heading", engine_kwargs) as pool:
                while report["search_releases"] < target:
                    ready = sum(map(len, pending.values()))
                    batch_sizes = [NATIVE_BATCH] * (ready // NATIVE_BATCH)
                    if ready % NATIVE_BATCH and report["search_releases"] + ready == target:
                        batch_sizes.append(ready % NATIVE_BATCH)
                    selected = [take_packet(pending, order, size) for size in batch_sizes]
                    reserved = Counter(row["context_id"] for packet in selected for row in packet)
                    jobs = drafts or next_jobs(reserved)
                    drafts = []
                    futures = [(job, pool.submit(prepare_proposals, job[1], job[2])) for job in jobs]
                    if not futures and not selected:
                        raise RuntimeError("Grid scheduler has neither preparation work nor ready drops")
                    for packet in selected:
                        release(packet)
                    for job, future in futures:
                        register(job, future.result())
                    if futures:
                        publish("preparing")

        if "controls_post" not in report["controls"]:
            run_controls("controls_post")
        if (sha(parent / "report.json") != binding["report"] or sha(args.parent_audit) != binding["audit"]
                or sha(parent / "evidence-manifest.json") != binding["manifest"]):
            raise RuntimeError("Parent evidence changed during the study")
        atomic(output / "evidence-manifest.json", {
            str(path.relative_to(output)): sha(path) for path in sorted(output.rglob("*"))
            if path.is_file() and path.name not in ("report.json", "evidence-manifest.json")
        })
        report.update(status="completed", evidence_manifest_sha256=sha(output / "evidence-manifest.json"))
        publish("completed")
    except Exception as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        publish("failed")
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("parent-study", "parent-audit", "reference", "assets", "history-root", "output", "gpu-lock"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--target-per-cell", type=int, default=1000)
    parser.add_argument("--prepare-batch", type=int, default=8)
    parser.add_argument("--seed", type=int, default=202611141050)
    parser.add_argument("--cpu-workers", type=int, default=1)
    parser.add_argument("--resume", action="store_true", help="Continue a stopped, intact evidence directory")
    args = parser.parse_args()
    if args.output.exists() != args.resume or (args.resume and not args.output.is_dir()):
        parser.error("Provide a fresh output directory or use --resume with a stopped study directory")
    if (not 1 <= args.target_per_cell <= 1000 or not 3 <= args.prepare_batch <= 32
            or not 1 <= args.cpu_workers <= 8):
        parser.error("Use 1–1000 drops per location, 3–32 preparations per batch and 1–8 CPU workers")
    with gpu_slot(args.gpu_lock, timeout_s=900, notify=lambda value: print(json.dumps(value), flush=True)):
        run(args)


if __name__ == "__main__":
    main()
