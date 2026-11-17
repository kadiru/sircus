"""Compare conditional flight-clearance expansion with the current one-drop planner."""
from __future__ import annotations

import argparse
from concurrent.futures import as_completed
import importlib.metadata
import json
from pathlib import Path
import shutil
import time

from . import intergrid_flight_ranking_study as baseline
from . import intergrid_system_study as parent_runner
from .benchmark_verify import digest, read, sha
from .cell1050_rebound_study import annotate, control_requests, parent_inputs
from .corrected_grid import source_manifest
from .coverage_training_study import append, share_native_environment
from .gpu_slot import foreign_processes, gpu_slot
from .interactive_physics_check import historical_layouts
from .intergrid_library_plan import libraries
from .intergrid_matched_route_plan import make_layouts as make_matched_layouts
from .intergrid_route_labels import LABEL_VERSION, annotate_routes
from .intergrid_route_selector_study import SOURCE_NAMES
from .intergrid_system_study import gpu_deadline, require_storage
from .parallel_capture_grid_study import NATIVE_BATCH, control_mirror
from .parallel_capture_grid_workers import load_resolution, open_workers
from .policy64_contract import load_runtime_reference
from .sweep_worker import atomic
from .table_training_study import control_gate

VERSION = "sircus-clearance-expansion-study-v1"
SEED = 2026112801
POSITION_COUNT = 50
POSITIONS = POSITION_COUNT
WORKERS = 4
GPU_CAP_S = 3600


def inputs(args):
    """Preserve the audited model and exclude all known interactive placements."""
    model, original, additions, previous, binding, history = parent_runner.inputs(args)
    _, _, _, system_binding = historical_layouts(args.system_study, args.system_audit)
    archive = read(args.system_study / "layouts.json")
    previous = previous + [row for split in archive.values() for row in split["layouts"]]
    live_files = sorted({path for folder in ("evidence", "evidence-acceptance")
                         for path in (args.interactive_history / folder).glob("interactive-*/result.json")})
    if not live_files:
        raise ValueError("Expected preserved interactive evidence for spatial exclusions")
    live_binding = []
    for path in live_files:
        result = read(path)
        xy = result["bucket_xy_m"]
        previous.append(dict(bucket_xy_m=xy))
        live_binding.append(dict(file=path.relative_to(args.interactive_history).as_posix(),
                                 sha256=sha(path), bucket_xy_m=xy))
    return model, original, additions, previous, dict(binding, system_comparison=system_binding,
                                                     interactive_history=live_binding), history


def layout_ids(layouts):
    return [dict(row, id=f"clearance-expansion-test-{i:02d}",
                 context_id=f"clearance-expansion-test-{i:02d}", split="test")
            for i, row in enumerate(layouts)]


def make_layouts(args, scene, previous):
    rows, provenance = make_matched_layouts(args.history_root, read(args.parent_study / "grid.json"),
                                            scene=scene, assets=args.assets,
                                            previous_layouts=previous, seed=SEED)
    rows = layout_ids(rows)
    if len(rows) != POSITION_COUNT:
        raise ValueError("Expected 50 new evaluation placements")
    return dict(layouts=rows, provenance=dict(provenance, layouts_sha256=digest(rows)))


def native_inputs(args):
    spec, parent_binding, _ = parent_inputs(args.parent_study, args.parent_audit)
    source = Path(__file__).resolve().parents[2]
    ref = load_runtime_reference(args.reference, args.assets, source)
    if (digest(ref["scene"]) != spec["identity"]["scene_sha256"]
            or ref["asset_hashes"] != spec["identity"]["asset_sha256"]
            or any(importlib.metadata.version(name) != version for name, version in spec["packages"].items())):
        raise ValueError("Scene, assets or dependency versions differ from audited controls")
    return spec, parent_binding, source, ref


def prepare(args):
    from .clearance_expansion_plan import METHODS, prepare_layout, protocol

    spec, parent_binding, source, ref = native_inputs(args)
    model, original, additions, previous, binding, history = inputs(args)
    layouts = make_layouts(args, ref["scene"], previous)
    output = args.output
    output.mkdir(parents=True, exist_ok=False)
    require_storage(output)
    atomic(output / "layouts.json", layouts)
    shutil.copyfile(args.model_study / "training-prior.json", output / "training-prior.json")
    shutil.copyfile(args.refit_study / "route-models.json", output / "route-models.json")
    atomic(output / "route-history.json", history)
    frozen = {name: sha(output / name) for name in
              ("layouts.json", "training-prior.json", "route-models.json", "route-history.json")}
    if (frozen["route-models.json"] != binding["refit_study"]["checkpoint_sha256"]
            or frozen["training-prior.json"] != binding["model_study"]["geometry_prior"]):
        raise ValueError("Copied model or geometry prior differs from audited source")
    contract = dict(version=VERSION, parent=parent_binding, identity=spec["identity"], scene=spec["scene"],
                    **binding, methods=list(METHODS), test_positions=POSITION_COUNT, seed=SEED,
                    physics_substeps=64, label_version=LABEL_VERSION, attempts_per_method_per_position=1,
                    protocol=protocol(), source_sha256=source_manifest(source),
                    frozen_inputs=frozen, gpu_cap_s=args.gpu_cap_s)
    contract["fingerprint"] = digest(contract)
    atomic(output / "contract.json", contract)
    for name in ("calibration.json", "verification.json"):
        shutil.copyfile(args.parent_study / name, output / name)
    report = dict(version=VERSION, status="running", phase="preparing", controls={},
                  prepared_positions=0, total_positions=POSITION_COUNT, test_releases=0,
                  completed_releases=0, planned_releases=None)
    atomic(output / "report.json", report)
    kwargs = dict(prior=read(output / "training-prior.json"),
                  calibration=read(output / "calibration.json"),
                  home=read(args.parent_study / "setup-home.json"))
    ranker_config = dict(scene=ref["scene"], calibration=kwargs["calibration"]["score_config"],
                         assets=str(args.assets), rule=binding["flight_ranking"]["rule"])
    shot_libraries = libraries(original, additions)
    prepared = {}
    try:
        with open_workers(WORKERS, ref, args.assets, output / "native-heading", kwargs) as workers:
            futures = {workers.submit(prepare_layout, row, shot_libraries, model, history, ranker_config): row
                       for row in layouts["layouts"]}
            for future in as_completed(futures):
                row = futures[future]
                site = future.result()
                prepared[row["id"]] = site
                atomic(output / "preparation" / (row["id"] + ".json"), site)
                report["prepared_positions"] = len(prepared)
                require_storage(output)
                atomic(output / "report.json", report)
                print(json.dumps(dict(phase="preparing", completed=len(prepared), total=POSITION_COUNT)), flush=True)
        sites = [prepared[row["id"]] for row in layouts["layouts"]]
        atomic(output / "plans.json", sites)
        slots = [slot for site in sites for slot in site["choices"]]
        atomic(output / "choices.json", dict(plans_sha256=sha(output / "plans.json"), slots=slots))
        report.update(phase="prepared", choices_sha256=sha(output / "choices.json"),
                      test_slots=len(slots), test_admitted=sum(slot["request"] is not None for slot in slots),
                      planned_releases=sum(slot["request"] is not None for slot in slots))
        atomic(output / "report.json", report)
        print(json.dumps(report), flush=True)
    except Exception as exc:
        report.update(status="failed", phase="preparation_failed", error=f"{type(exc).__name__}: {exc}")
        atomic(output / "report.json", report)
        raise


def run_native(args):
    from .cell1050_heading_native import HeadingEngine
    from .clearance_expansion_audit import summarize
    from .table_correction_native import MatchedEngine

    output, parent = args.output, args.parent_study
    spec, _, source, ref = native_inputs(args)
    contract = read(output / "contract.json")
    report = read(output / "report.json")
    if report["phase"] != "prepared" or report["status"] != "running":
        raise ValueError("Native execution requires a prepared, unexecuted comparison")
    for name in ("clearance_expansion_plan.py", "clearance_expansion_study.py"):
        if contract["source_sha256"]["sircus/mjlab/" + name] != sha(source / "sircus/mjlab" / name):
            raise ValueError("Prepared planner or runner source changed")
    frozen = dict(contract["frozen_inputs"], **{name: sha(output / name) for name in ("plans.json", "choices.json")})
    if frozen["choices.json"] != report["choices_sha256"]:
        raise ValueError("Frozen choices changed before release")
    sites = read(output / "plans.json")
    choices = read(output / "choices.json")
    if choices["plans_sha256"] != frozen["plans.json"]:
        raise ValueError("Frozen preparation changed before release")
    slots = choices["slots"]
    requests = [slot["request"] for slot in slots if slot["request"] is not None]
    kwargs = dict(prior=read(output / "training-prior.json"), calibration=read(output / "calibration.json"),
                  home=read(parent / "setup-home.json"))
    expected_options = read(parent / "native-yaw/effective-options.json")
    expected_geometry = read(parent / "native-yaw/native-compiled-geometry.json")
    packet_number = 0

    def publish(phase):
        report["phase"] = phase
        atomic(output / "report.json", report)
        print(json.dumps({k: v for k, v in report.items() if k != "comparison"}), flush=True)

    def waiting(status):
        report["gpu_wait"] = status
        publish("waiting_for_4090")

    def execute(engine, batch, phase):
        nonlocal packet_number
        if foreign_processes():
            raise RuntimeError("Another process is using the leased 4090")
        require_storage(output)
        if any(sha(output / name) != expected for name, expected in frozen.items()):
            raise ValueError("Frozen inputs changed during execution")
        control = phase.startswith("controls_")
        directory = "native-controls" if control else "native-heading"
        if not control:
            for xy in {tuple(request["bucket_xy_m"]) for request in batch}:
                load_resolution(engine, list(xy))
        packet = f"requests/batch-{packet_number:05d}.json"
        packet_number += 1
        atomic(output / packet, dict(phase=phase, rows=batch))
        measured, timing = engine.evaluate(batch, phase)
        if [row["id"] for row in measured] != [row["id"] for row in batch] or any(
                row["native_release"] is not True for row in measured):
            raise RuntimeError("An admitted frozen request failed to release")
        adapter = engine.engine.adapter if control else engine.adapter
        if adapter.env.effective_options != expected_options or adapter.env.compiled_geometry != expected_geometry:
            raise RuntimeError("Native settings differ from pinned controls")
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

    try:
        with (output / "native-started").open("x") as started_file:
            started_file.write(frozen["choices.json"] + "\n")
        with gpu_slot(args.gpu_lock, timeout_s=args.gpu_wait_s, poll_s=15, notify=waiting):
            with gpu_deadline(contract["gpu_cap_s"]):
                started = time.monotonic()
                report.pop("gpu_wait", None)
                heading = HeadingEngine(ref, args.assets, output / "native-heading", **kwargs)
                control_sources = read(parent / "control-replays.json")
                mirror = control_mirror(parent, output, control_sources, False)
                controls = MatchedEngine(ref, args.assets, output / "native-controls", parent=mirror,
                                        sources=control_sources, prior=read(parent / "old-prior.json"),
                                        calibration=kwargs["calibration"], home=kwargs["home"])
                share_native_environment(heading, controls)
                for phase in ("controls_pre", "test", "controls_post"):
                    if phase != "test":
                        result = control_gate(execute(controls, control_requests(control_sources, phase), phase))
                        report["controls"][phase] = result
                        if result["status"] != "passed":
                            raise RuntimeError(f"{phase} failed")
                        publish(phase)
                        continue
                    observations = {}
                    for start in range(0, len(requests), NATIVE_BATCH):
                        measured = execute(heading, requests[start:start + NATIVE_BATCH], phase)
                        observations.update((row["id"], row) for row in measured)
                        report.update(test_releases=len(observations), completed_releases=len(observations))
                        publish(phase)
                    report["comparison"] = summarize(read(output / "layouts.json")["layouts"], slots, observations, sites)
                report["gpu_lease_s"] = time.monotonic() - started
        atomic(output / "evidence-manifest.json", {
            str(path.relative_to(output)): sha(path) for path in sorted(output.rglob("*"))
            if path.is_file() and path.name != "report.json"
        })
        report.update(status="completed", evidence_manifest_sha256=sha(output / "evidence-manifest.json"))
        publish("completed")
    except Exception as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        publish("failed")
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("history-root", "parent-study", "parent-audit", "reference", "assets", "gpu-lock", "output",
                 "rule", "system-study", "system-audit", "interactive-history"):
        parser.add_argument("--" + name, type=Path, required=True)
    for name in (*SOURCE_NAMES, *baseline.EXTRA_SOURCES, "prior-comparison"):
        for kind in ("study", "audit"):
            parser.add_argument(f"--{name}-{kind}", type=Path, required=True)
    parser.add_argument("--gpu-wait-s", type=int, default=7 * 3600)
    parser.add_argument("--gpu-cap-s", type=int, default=GPU_CAP_S)
    stages = parser.add_mutually_exclusive_group()
    stages.add_argument("--prepare-only", action="store_true")
    stages.add_argument("--run-prepared", action="store_true")
    stages.add_argument("--audit", type=Path)
    args = parser.parse_args()
    if args.audit:
        if args.audit.exists() or args.audit.resolve().is_relative_to(args.output.resolve()):
            parser.error("Use a fresh audit path outside frozen evidence")
        from .clearance_expansion_audit import audit
        try:
            result = audit(args)
        except Exception as exc:
            result = dict(status="failed", error=f"{type(exc).__name__}: {exc}")
        atomic(args.audit, result)
        print(json.dumps(result), flush=True)
        if result["status"] != "passed":
            raise SystemExit(1)
        return
    if not 0 < args.gpu_cap_s <= GPU_CAP_S or args.gpu_wait_s <= 0:
        parser.error("Use a positive wait and at most one hour of leased GPU time")
    if not args.run_prepared:
        if args.output.exists():
            parser.error("Preparation requires a fresh output directory")
        prepare(args)
    if not args.prepare_only:
        run_native(args)


if __name__ == "__main__":
    main()
