"""Frozen 30-second shot planning, then one native64 drop per new position."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import random
import shutil
import time

from . import clearance_expansion_study as parent
from .benchmark_verify import digest, read, sha
from .cell1050_rebound_study import annotate, control_requests
from .corrected_grid import source_manifest
from .coverage_training_study import append
from .gpu_slot import foreign_processes, gpu_slot
from .intergrid_library_plan import libraries
from .intergrid_proposer_plan import make_splits, sample_splits
from .intergrid_route_labels import LABEL_VERSION, annotate_routes
from .intergrid_system_study import gpu_deadline, require_storage
from .parallel_capture_grid_study import NATIVE_BATCH, control_mirror
from .parallel_capture_grid_workers import load_resolution
from .sweep_worker import atomic
from .table_training_study import control_gate

VERSION = "sircus-planning-budget-study-v1"
METHODS = ("current_planner", "neural_selector")
POSITION_COUNT = 100
PLANNING_BUDGET_S = 30.
SEED = 2026113001
WORKERS = 4
GPU_CAP_S = 3600
COPIED_INPUTS = ("shot-library-rows.json", "route-history.json", "training-prior.json",
                 "route-models.json", "calibration.json", "verification.json", "neural-checkpoint.json")


def fresh_layouts(excluded, geometry, grid, *, seed=SEED):
    """Two stratified batches, separated from history and each other by 5 mm."""
    points, rows = list(map(tuple, excluded)), []
    for batch in range(2):
        selected = sample_splits(points, geometry, grid, seed=seed + batch, include_train=False)["test"]
        rows.extend(selected)
        points.extend(tuple(row["bucket_xy_m"]) for row in selected)
    return [dict(row, id=f"planning-budget-test-{i:03d}", context_id=f"planning-budget-test-{i:03d}",
                 split="test") for i, row in enumerate(rows)]


def publish(args, report, phase=None):
    if phase is not None:
        report["phase"] = phase
    report["updated_at_unix"] = time.time()
    atomic(args.output / "report.json", report)
    if args.display_report:
        display = read(args.display_report) if args.display_report.exists() else {}
        display["planning_budget_study"] = report
        atomic(args.display_report, display)
    print(json.dumps({k: report[k] for k in ("status", "phase", "progress", "gpu_wait", "error")
                      if k in report}), flush=True)


def audited_neural(args):
    archive, receipt = args.neural_study, read(args.neural_audit)
    report = read(archive / "report.json")
    manifest = read(archive / "evidence-manifest.json")
    if (receipt.get("status") != "passed" or report.get("status") != "completed"
            or receipt["report_sha256"] != sha(archive / "report.json")
            or receipt["evidence_manifest_sha256"] != sha(archive / "evidence-manifest.json")
            or receipt["comparison"] != report["comparison"]):
        raise ValueError("The learned selector needs its unchanged passed comparison audit")
    for name in (*COPIED_INPUTS, "layouts.json", "contract.json"):
        if sha(archive / name) != manifest[name]:
            raise ValueError(f"Audited neural input changed: {name}")
    return dict(study_report_sha256=sha(archive / "report.json"), audit_sha256=sha(args.neural_audit),
                checkpoint_sha256=sha(archive / "neural-checkpoint.json"),
                layouts_sha256=sha(archive / "layouts.json"))


def initialize(args):
    spec, parent_binding, source, ref = parent.native_inputs(args)
    _, original, additions, previous, binding, history = parent.inputs(args)
    neural_binding = audited_neural(args)
    previous += [row for split in read(args.neural_study / "layouts.json").values() for row in split["layouts"]]
    previous += read(args.expansion_study / "layouts.json")["layouts"]
    grid = read(args.parent_study / "grid.json")
    _, provenance = make_splits(args.history_root, grid, scene=ref["scene"], assets=args.assets,
                                previous_layouts=previous, seed=SEED, include_train=False)
    layouts = fresh_layouts(provenance["excluded_coordinates_m"], provenance["geometry"], grid)
    if len(layouts) != POSITION_COUNT:
        raise ValueError("Expected exactly 100 fresh test placements")
    args.output.mkdir(parents=True, exist_ok=False)
    require_storage(args.output)
    atomic(args.output / "layouts.json", dict(layouts=layouts, provenance=dict(provenance, layouts_sha256=digest(layouts))))
    for name in COPIED_INPUTS:
        shutil.copyfile(args.neural_study / name, args.output / name)
    if (read(args.output / "shot-library-rows.json") != dict(original=original, additions=additions)
            or read(args.output / "route-history.json") != history):
        raise ValueError("Frozen library/history differ from the current planner's audited inputs")
    frozen = {p.name: sha(p) for p in args.output.glob("*.json")}
    contract = dict(version=VERSION, parent=parent_binding, identity=spec["identity"], scene=spec["scene"],
                    **binding, neural_parent=neural_binding, methods=list(METHODS), test_positions=POSITION_COUNT,
                    seed=SEED, planning_budget_s=PLANNING_BUDGET_S, physics_substeps=64, label_version=LABEL_VERSION,
                    attempts_per_method_per_position=1, gpu_cap_s=GPU_CAP_S, workers=WORKERS,
                    frozen_inputs=frozen, source_sha256=source_manifest(source),
                    budget_scope="Menu, inference, IK/path checks and ranking; common initialization separately measured",
                    timeout_rule="Last fully checked choice received before deadline; none means no release",
                    scope="Frozen models, privileged geometry, endpoint-held native64; no live promotion")
    contract["fingerprint"] = digest(contract)
    atomic(args.output / "contract.json", contract)
    report = dict(version=VERSION, status="running", phase="initialized", audit_status="pending",
                  budgets=dict(planning_s=PLANNING_BUDGET_S, positions=POSITION_COUNT), controls={},
                  gpu_lease_s=0., gpu_cap_s=GPU_CAP_S,
                  progress=dict(prepared=0, planned=POSITION_COUNT * len(METHODS), completed_releases=0,
                                planned_releases=None),
                  next="Prepare both frozen planners under the same 30-second decision deadline.")
    publish(args, report)


def frozen_check(args):
    contract = read(args.output / "contract.json")
    for name, expected in contract["frozen_inputs"].items():
        if sha(args.output / name) != expected:
            raise ValueError(f"Frozen input changed: {name}")
    source = Path(__file__).resolve().parents[2]
    for name, expected in contract["source_sha256"].items():
        if sha(source / name) != expected:
            raise ValueError(f"Frozen computational source changed: {name}")
    return contract


def prepare(args):
    from .planning_budget_plan import run_budget_job

    contract = frozen_check(args)
    _, _, _, ref = parent.native_inputs(args)
    output = args.output
    if (output / "plans.json").exists():
        raise ValueError("Preparation is already frozen; preserve it")
    layouts = read(output / "layouts.json")["layouts"]
    raw = read(output / "shot-library-rows.json")
    libs = libraries(raw["original"], raw["additions"])
    model, history, neural = (read(output / name) for name in ("route-models.json", "route-history.json", "neural-checkpoint.json"))
    kwargs = dict(prior=read(output / "training-prior.json"), calibration=read(output / "calibration.json"),
                  home=read(args.parent_study / "setup-home.json"))
    config = dict(scene=ref["scene"], calibration=kwargs["calibration"]["score_config"],
                  assets=str(args.assets), rule=contract["flight_ranking"]["rule"])
    report = read(output / "report.json")
    tasks = [(row, method) for row in layouts for method in METHODS]
    random.Random(SEED).shuffle(tasks)
    sites = {}
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        jobs = {pool.submit(run_budget_job, row, method, libs, model, history, config, neural,
                            reference=ref, assets=args.assets, directory=output / ("native-" + method),
                            engine_kwargs=kwargs, budget_s=PLANNING_BUDGET_S): (row, method)
                for row, method in tasks}
        for future in as_completed(jobs):
            row, method = jobs[future]
            site = future.result()
            sites[row["id"], method] = site
            atomic(output / "preparation" / method / (row["id"] + ".json"), site)
            report["progress"]["prepared"] = len(sites)
            require_storage(output)
            publish(args, report, "preparing")
    ordered = [sites[row["id"], method] for row in layouts for method in METHODS]
    slots = [site["choice"] for site in ordered]
    atomic(output / "plans.json", ordered)
    bindings = {name: sha(output / name) for name in (*contract["frozen_inputs"], "plans.json")}
    atomic(output / "choices.json", dict(slots=slots, bindings=bindings))
    report["progress"]["planned_releases"] = sum(slot["request"] is not None for slot in slots)
    publish(args, report, "prepared")


def run_native(args):
    from .cell1050_heading_native import HeadingEngine
    from .table_correction_native import MatchedEngine

    contract = frozen_check(args)
    _, _, _, ref = parent.native_inputs(args)
    output = args.output
    report = read(output / "report.json")
    if report["phase"] != "prepared" or report["status"] != "running":
        raise ValueError("Need complete frozen preparation before any drop")
    choices = read(output / "choices.json")
    bindings = dict(choices["bindings"], **{"choices.json": sha(output / "choices.json")})
    if any(sha(output / name) != value for name, value in bindings.items()):
        raise ValueError("Pre-release choices or their inputs changed")
    planned = {method: [slot["request"] for slot in choices["slots"]
                        if slot["method"] == method and slot["request"] is not None] for method in METHODS}
    receipt = output / "native-started.json"
    with receipt.open("x") as stream:
        json.dump(dict(bindings=bindings, requests_sha256=digest(planned), timestamp_unix=time.time()), stream)
    kwargs = dict(prior=read(output / "training-prior.json"), calibration=read(output / "calibration.json"),
                  home=read(args.parent_study / "setup-home.json"))
    expected_options = read(args.parent_study / "native-yaw/effective-options.json")
    expected_geometry = read(args.parent_study / "native-yaw/native-compiled-geometry.json")
    packet_number = 0

    def execute(engine, batch, phase, directory, control=False):
        nonlocal packet_number
        if foreign_processes():
            raise RuntimeError("A foreign process appeared on the leased 4090")
        require_storage(output)
        if any(sha(output / name) != value for name, value in bindings.items()):
            raise ValueError("Frozen selections changed during native execution")
        if not control:
            for xy in {tuple(row["bucket_xy_m"]) for row in batch}:
                load_resolution(engine, list(xy))
        packet = f"requests/batch-{packet_number:05d}.json"
        packet_number += 1
        atomic(output / packet, dict(phase=phase, control=control, adapter_directory=directory, rows=batch))
        measured, timing = engine.evaluate(batch, phase)
        if [row["id"] for row in measured] != [row["id"] for row in batch] or any(
                row["native_release"] is not True for row in measured):
            raise RuntimeError("A frozen admitted command did not release exactly once")
        adapter = engine.engine.adapter if control else engine.adapter
        if adapter.env.effective_options != expected_options or adapter.env.compiled_geometry != expected_geometry:
            raise RuntimeError("Native options or geometry differ from audited controls")
        if not control:
            annotate(measured, output / directory)
            annotate_routes(measured, output / directory)
            for row, request in zip(measured, batch, strict=True):
                row["intended_route"] = request["intended_route"]
        for row in measured:
            row.update(request_packet=packet, request_packet_sha256=sha(output / packet), adapter_directory=directory)
        append(output / "rows.jsonl", measured)
        append(output / "timing.jsonl", [dict(packet=packet, phase=phase, timing=timing)])
        return measured

    def waiting(status):
        report["gpu_wait"] = status
        publish(args, report, "waiting_for_4090")

    with gpu_slot(args.gpu_lock, timeout_s=args.gpu_wait_s, poll_s=30, notify=waiting):
        started = time.monotonic()
        with gpu_deadline(GPU_CAP_S):
            report.pop("gpu_wait", None)
            engines = {method: HeadingEngine(ref, args.assets, output / ("native-" + method), **kwargs) for method in METHODS}
            sources = read(args.parent_study / "control-replays.json")
            mirror = control_mirror(args.parent_study, output, sources, False)
            controls = MatchedEngine(ref, args.assets, output / "native-controls", parent=mirror, sources=sources,
                                    prior=read(args.parent_study / "old-prior.json"),
                                    calibration=kwargs["calibration"], home=kwargs["home"])
            adapters = [engine.adapter for engine in engines.values()] + [controls.engine.adapter]
            factory, allocated = adapters[0].environment_factory, []
            def shared_factory():
                if not allocated:
                    allocated.append(factory())
                return allocated[0]
            for adapter in adapters:
                adapter.environment_factory = shared_factory
            for phase in ("controls_pre", "test", "controls_post"):
                if phase != "test":
                    gate = control_gate(execute(controls, control_requests(sources, phase), phase, "native-controls", True))
                    report["controls"][phase] = gate
                    if gate["status"] != "passed":
                        raise RuntimeError(f"Matched control gate failed: {phase}")
                    publish(args, report, phase)
                    continue
                # Alternate method batches without feeding any outcome back into selection.
                for offset in range(0, max(map(len, planned.values())), NATIVE_BATCH):
                    for method in METHODS:
                        batch = planned[method][offset:offset + NATIVE_BATCH]
                        if not batch:
                            continue
                        rows = execute(engines[method], batch, "test_" + method, "native-" + method)
                        report["progress"]["completed_releases"] += len(rows)
                        report["gpu_lease_s"] = time.monotonic() - started
                        publish(args, report, "test_" + method)
        report["gpu_lease_s"] = time.monotonic() - started
    report["native_completed"] = True
    publish(args, report, "native_completed")


def finalize(args):
    from .parallel_capture_grid_audit import observed_rows
    from .planning_budget_audit import summarize

    frozen_check(args)
    output = args.output
    report = read(output / "report.json")
    if report.get("native_completed") is not True:
        raise ValueError("Complete native drops and both controls before summarizing")
    rows = {row["id"]: row for row in observed_rows(output / "rows.jsonl")
            if row["adapter_directory"] != "native-controls"}
    report["comparison"] = summarize(read(output / "layouts.json")["layouts"], read(output / "choices.json")["slots"],
                                      rows, read(output / "plans.json"))
    atomic(output / "evidence-manifest.json", {str(path.relative_to(output)): sha(path)
           for path in sorted(output.rglob("*")) if path.is_file() and path.name not in {"report.json", "evidence-manifest.json"}})
    report.update(status="completed", audit_status="pending", evidence_manifest_sha256=sha(output / "evidence-manifest.json"),
                  next="Independently audit the bounded comparison before publishing capture scores.")
    publish(args, report, "completed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("history-root", "parent-study", "parent-audit", "reference", "assets", "gpu-lock", "output",
                 "rule", "system-study", "system-audit", "interactive-history", "expansion-study", "expansion-audit",
                 "neural-study", "neural-audit"):
        parser.add_argument("--" + name, type=Path, required=True)
    for name in (*parent.SOURCE_NAMES, *parent.baseline.EXTRA_SOURCES, "prior-comparison"):
        for kind in ("study", "audit"):
            parser.add_argument(f"--{name}-{kind}", type=Path, required=True)
    parser.add_argument("--display-report", type=Path)
    parser.add_argument("--gpu-wait-s", type=int, default=12 * 3600)
    parser.add_argument("--stage", required=True, choices=("initialize", "prepare", "run", "finalize", "audit"))
    parser.add_argument("--audit-output", type=Path)
    args = parser.parse_args()
    if args.gpu_wait_s <= 0:
        parser.error("Use a positive GPU wait limit")
    try:
        if args.stage == "initialize":
            initialize(args)
        elif args.stage == "prepare":
            prepare(args)
        elif args.stage == "run":
            run_native(args)
        elif args.stage == "finalize":
            finalize(args)
        else:
            from .planning_budget_audit import audit
            if args.audit_output is None or args.audit_output.exists() or args.audit_output.is_relative_to(args.output):
                parser.error("Use a fresh audit path outside the study evidence")
            result = audit(args)
            atomic(args.audit_output, result)
            if result["status"] != "passed":
                raise RuntimeError("Independent audit failed")
            if args.display_report:
                display = read(args.display_report)
                display["planning_budget_study"] = dict(read(args.output / "report.json"), audit_status="passed",
                    next="Compare capture and timeouts before selecting an interactive default.")
                atomic(args.display_report, display)
            print(json.dumps({k:v for k,v in result.items() if k != "comparison"}), flush=True)
    except Exception as exc:
        if (args.output / "report.json").exists() and args.stage != "audit":
            report = read(args.output / "report.json")
            report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
            publish(args, report, "failed_" + args.stage)
        raise


if __name__ == "__main__":
    main()
