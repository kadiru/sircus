"""Compare two table-shot libraries with a single frozen specialist."""
from __future__ import annotations

import argparse
from concurrent.futures import as_completed
import importlib.metadata
import json
from pathlib import Path
import shutil

from .benchmark_verify import digest, read, sha
from .cell1050_rebound_study import annotate, control_requests, parent_inputs
from .corrected_grid import source_manifest
from .coverage_training_study import append, share_native_environment
from .gpu_slot import foreign_processes, gpu_slot
from .intergrid_library_inputs import fresh_layouts, load_library_inputs
from .intergrid_library_plan import METHODS, ROUTE, libraries, select_requests, summarize, test_proposals
from .intergrid_route_labels import LABEL_VERSION, annotate_routes
from .intergrid_route_study import PER_ROUTE_TEST
from .parallel_capture_grid_study import NATIVE_BATCH, STORAGE_RESERVE_BYTES, control_mirror
from .parallel_capture_grid_workers import load_resolution, open_workers, prepare_proposals
from .policy64_contract import load_runtime_reference
from .sweep_worker import atomic
from .table_training_study import control_gate

VERSION = "sircus-intergrid-table-library-study-v1"
SPLIT_SEED = 2026112001
POSITIONS = 50
WORKERS = 8


def library_summary(shot_libraries):
    return {
        method: dict(shots=sum(len(shots) for shots in library[ROUTE].values()),
                     positions=len(library[ROUTE]))
        for method, library in shot_libraries.items()
    }


def run(args):
    from .cell1050_heading_native import HeadingEngine
    from .table_correction_native import MatchedEngine

    output, parent = args.output, args.parent_study
    spec, parent_binding, _ = parent_inputs(parent, args.parent_audit)
    source = Path(__file__).resolve().parents[2]
    ref = load_runtime_reference(args.reference, args.assets, source)
    if (digest(ref["scene"]) != spec["identity"]["scene_sha256"]
            or ref["asset_hashes"] != spec["identity"]["asset_sha256"]
            or any(importlib.metadata.version(name) != version for name, version in spec["packages"].items())):
        raise ValueError("Scene, assets or dependencies differ from the audited controls")
    checkpoint, original_rows, new_rows, previous, binding = load_library_inputs(
        args.history_root, args.model_study, args.model_audit, args.paired_study, args.paired_audit,
        args.matched_study, args.matched_audit, args.refit_study, args.refit_audit)
    refit_contract = read(args.refit_study / "contract.json")
    if (refit_contract["parent"] != parent_binding or refit_contract["identity"] != spec["identity"]
            or refit_contract["physics_substeps"] != 64 or refit_contract["label_version"] != LABEL_VERSION):
        raise ValueError("Frozen specialist uses a different simulation contract")
    prior = read(args.model_study / "training-prior.json")
    layouts, provenance = fresh_layouts(args.history_root, parent / "grid.json", ref["scene"], args.assets,
                                       previous, SPLIT_SEED)
    if len(layouts) != POSITIONS or len({tuple(row["bucket_xy_m"]) for row in layouts}) != POSITIONS:
        raise ValueError("Expected 50 distinct test positions")
    output.mkdir(parents=True, exist_ok=False)
    report = dict(status="running", controls={}, test_prepared=0, test_releases=0)
    packet_number = 0

    def publish(phase):
        report["phase"] = phase
        atomic(output / "report.json", report)
        print(json.dumps(report), flush=True)

    def execute(engine, requests, phase):
        nonlocal packet_number
        if foreign_processes():
            raise RuntimeError("Another process is using the leased 4090")
        if shutil.disk_usage(output).free < STORAGE_RESERVE_BYTES:
            raise RuntimeError("Insufficient space for native trace evidence")
        control = phase.startswith("controls_")
        directory = "native-controls" if control else "native-heading"
        if not control:
            for xy in {tuple(request["bucket_xy_m"]) for request in requests}:
                load_resolution(engine, list(xy))
        packet = f"requests/batch-{packet_number:05d}.json"
        packet_number += 1
        atomic(output / packet, dict(phase=phase, rows=requests))
        measured, timing = engine.evaluate(requests, phase)
        if ([row["id"] for row in measured] != [row["id"] for row in requests]
                or any(row["native_release"] is not True for row in measured)):
            raise RuntimeError("An exact admitted request did not release")
        adapter = engine.engine.adapter if control else engine.adapter
        if (adapter.env.effective_options != expected_options
                or adapter.env.compiled_geometry != expected_geometry):
            raise RuntimeError("Native physics differs from the pinned controls")
        if not control:
            annotate(measured, output / directory)
            annotate_routes(measured, output / directory)
            for row, request in zip(measured, requests, strict=True):
                row["intended_route"] = request["intended_route"]
        packet_hash = sha(output / packet)
        for row in measured:
            row.update(request_packet=packet, request_packet_sha256=packet_hash, adapter_directory=directory)
        append(output / "rows.jsonl", measured)
        append(output / "timing.jsonl", [dict(packet=packet, phase=phase, timing=timing)])
        return measured

    def run_controls(phase):
        gate = control_gate(execute(controls, control_requests(control_sources, phase), phase))
        report["controls"][phase] = gate
        if gate["status"] != "passed":
            raise RuntimeError(f"{phase} failed")
        publish(phase)

    try:
        atomic(output / "layouts.json", dict(layouts=layouts, provenance=provenance))
        shutil.copyfile(args.model_study / "training-prior.json", output / "training-prior.json")
        shutil.copyfile(args.refit_study / "route-models.json", output / "route-models.json")
        checkpoint_sha256 = sha(output / "route-models.json")
        if checkpoint_sha256 != binding["refit_study"]["checkpoint_sha256"]:
            raise ValueError("Frozen specialist checkpoint changed during copy")
        report["checkpoint_sha256"] = checkpoint_sha256
        contract = dict(version=VERSION, parent=parent_binding, identity=spec["identity"], scene=spec["scene"],
                        **binding, test_positions=POSITIONS, methods=list(METHODS), routes=[ROUTE],
                        attempts_per_method_per_route_per_position=1, candidate_budget_per_route=PER_ROUTE_TEST,
                        physics_substeps=64, label_version=LABEL_VERSION, split_seed=SPLIT_SEED,
                        source_sha256=source_manifest(source),
                        frozen_inputs={name: sha(output / name) for name in (
                            "layouts.json", "training-prior.json", "route-models.json")})
        contract["fingerprint"] = digest(contract)
        atomic(output / "contract.json", contract)
        setup, calibration = read(parent / "setup-home.json"), read(parent / "calibration.json")
        for name in ("calibration.json", "verification.json"):
            shutil.copyfile(parent / name, output / name)
        kwargs = dict(prior=prior, calibration=calibration, home=setup)
        shot_libraries = libraries(original_rows, new_rows)
        report["library"] = library_summary(shot_libraries)

        def waiting(status):
            report["gpu_wait"] = status
            publish("waiting_for_4090")

        observations = {}
        prepared = {}
        publish("preparing_test")
        with open_workers(WORKERS, ref, args.assets, output / "native-heading", kwargs) as workers:
            pending = {}
            for layout in layouts:
                proposals = test_proposals(layout, shot_libraries, checkpoint)
                pending[workers.submit(prepare_proposals, proposals, True)] = (layout, proposals)
            for future in as_completed(pending):
                layout, proposals = pending[future]
                admitted, records = future.result()
                atomic(output / "test" / (layout["id"] + ".json"),
                       dict(proposals=proposals, admitted=admitted, records=records))
                prepared[layout["id"]] = select_requests(layout, proposals, admitted)
                report["test_prepared"] = len(prepared)
                if len(prepared) % 5 == 0:
                    publish("preparing_test")
        slots = [slot for layout in layouts for slot in prepared[layout["id"]]]
        if len(slots) != POSITIONS * len(METHODS):
            raise RuntimeError("The frozen comparison is missing an attempt slot")
        atomic(output / "choices.json", dict(checkpoint_sha256=checkpoint_sha256, slots=slots))
        report["choices_sha256"] = sha(output / "choices.json")
        requests = [slot["request"] for slot in slots if slot["request"] is not None]
        report.update(test_slots=len(slots), test_admitted=len(requests))
        publish("choices_frozen")
        with gpu_slot(args.gpu_lock, timeout_s=args.gpu_wait_s, poll_s=15, notify=waiting):
            report.pop("gpu_wait", None)
            heading = HeadingEngine(ref, args.assets, output / "native-heading", **kwargs)
            control_sources = read(parent / "control-replays.json")
            mirror = control_mirror(parent, output, control_sources, False)
            controls = MatchedEngine(ref, args.assets, output / "native-controls", parent=mirror,
                                     sources=control_sources, prior=read(parent / "old-prior.json"),
                                     calibration=calibration, home=setup)
            share_native_environment(heading, controls)
            expected_options = read(parent / "native-yaw/effective-options.json")
            expected_geometry = read(parent / "native-yaw/native-compiled-geometry.json")
            run_controls("controls_pre")
            for offset in range(0, len(requests), NATIVE_BATCH):
                if (sha(output / "route-models.json") != checkpoint_sha256
                        or sha(output / "choices.json") != report["choices_sha256"]):
                    raise ValueError("Checkpoint or choices changed during the test")
                batch = requests[offset:offset + NATIVE_BATCH]
                observations.update((row["id"], row) for row in execute(heading, batch, "test"))
                report["test_releases"] = len(observations)
                publish("testing")
            report["comparison"] = summarize(layouts, slots, observations)
            run_controls("controls_post")
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
    for name in ("history-root", "model-study", "model-audit", "parent-study", "parent-audit",
                 "reference", "assets", "gpu-lock", "output", "paired-study", "paired-audit",
                 "matched-study", "matched-audit", "refit-study", "refit-audit"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--gpu-wait-s", type=int, default=7 * 3600)
    args = parser.parse_args()
    if args.output.exists() or args.gpu_wait_s <= 0:
        parser.error("Provide a fresh output directory and a positive GPU wait bound")
    run(args)


if __name__ == "__main__":
    main()
