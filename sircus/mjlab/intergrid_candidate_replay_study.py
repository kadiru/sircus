"""Replay admitted commands from the route selector's prior ordinary flight misses."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path
import shutil

from .benchmark_verify import contained, digest, read, sha
from .cell1050_rebound_study import annotate, control_requests, parent_inputs
from .corrected_grid import source_manifest
from .coverage_training_study import append, share_native_environment
from .gpu_slot import foreign_processes, gpu_slot
from .intergrid_candidate_replay_plan import load_replay_plan
from .intergrid_candidate_replay_summary import summarize
from .intergrid_route_labels import LABEL_VERSION, annotate_routes
from .parallel_capture_grid_study import NATIVE_BATCH, STORAGE_RESERVE_BYTES, control_mirror
from .parallel_capture_grid_workers import load_resolution
from .policy64_contract import load_runtime_reference
from .sweep_worker import atomic
from .table_training_study import control_gate


VERSION = "sircus-intergrid-candidate-replay-v1"


def copy_setup_proofs(fallback, output, manifest, requests):
    """Copy only the exact resolution and connection used by each replay."""
    from .table_correction_native import copy_bound

    files = {}
    for request in requests:
        for name in ("context_resolution", "prepared_connection"):
            path, expected = request[f"{name}_file"], request[f"{name}_sha256"]
            if manifest["native-heading/" + path] != expected:
                raise ValueError("Setup proof does not match the fallback manifest: " + path)
            files[path] = expected
    for path, expected in files.items():
        copy_bound(fallback / "native-heading", output / "native-heading", path, expected)
    return len(files)


def run(args):
    from .cell1050_heading_native import HeadingEngine
    from .table_correction_native import MatchedEngine

    output, parent, fallback = args.output, args.parent_study, args.fallback_study
    spec, parent_binding, _ = parent_inputs(parent, args.parent_audit)
    source = Path(__file__).resolve().parents[2]
    ref = load_runtime_reference(args.reference, args.assets, source)
    if (digest(ref["scene"]) != spec["identity"]["scene_sha256"]
            or ref["asset_hashes"] != spec["identity"]["asset_sha256"]
            or any(importlib.metadata.version(name) != pinned for name, pinned in spec["packages"].items())):
        raise ValueError("Scene, assets or dependencies differ from the audited controls")
    plan, binding = load_replay_plan(fallback, args.fallback_audit)
    fallback_spec = read(fallback / "contract.json")
    fallback_manifest = read(fallback / "evidence-manifest.json")
    if (fallback_spec["parent"] != parent_binding or fallback_spec["identity"] != spec["identity"]
            or fallback_spec["physics_substeps"] != 64 or fallback_spec["label_version"] != LABEL_VERSION):
        raise ValueError("Replay requires the original simulation and scoring contract")
    requests = [request for case in plan["cases"] for request in case["requests"]]
    if not requests or len({request["id"] for request in requests}) != len(requests):
        raise ValueError("Replay requires distinct frozen requests")
    output.mkdir(parents=True, exist_ok=False)
    report = dict(status="running", controls={}, test_releases=0, test_requests=len(requests),
                  flight_miss_cases=len(plan["cases"]), setup_failure_cases=len(plan["setup_failures"]))
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
        atomic(output / "replay-plan.json", plan)
        prior_file = contained(fallback, "training-prior.json")
        if sha(prior_file) != fallback_manifest["training-prior.json"]:
            raise ValueError("Control geometry prior differs from the fallback study")
        shutil.copyfile(prior_file, output / "training-prior.json")
        prior = read(output / "training-prior.json")
        contract = dict(version=VERSION, parent=parent_binding, identity=spec["identity"], scene=spec["scene"],
                        fallback_study=binding, physics_substeps=64, label_version=LABEL_VERSION,
                        source_sha256=source_manifest(source),
                        frozen_inputs={name: sha(output / name) for name in ("replay-plan.json", "training-prior.json")})
        contract["fingerprint"] = digest(contract)
        atomic(output / "contract.json", contract)
        setup, calibration = read(parent / "setup-home.json"), read(parent / "calibration.json")
        for name in ("calibration.json", "verification.json"):
            shutil.copyfile(parent / name, output / name)
        report["setup_proofs"] = copy_setup_proofs(fallback, output, fallback_manifest, requests)
        publish("replay_frozen")

        def waiting(status):
            report["gpu_wait"] = status
            publish("waiting_for_4090")

        observations = {}
        with gpu_slot(args.gpu_lock, timeout_s=args.gpu_wait_s, poll_s=15, notify=waiting):
            report.pop("gpu_wait", None)
            heading = HeadingEngine(ref, args.assets, output / "native-heading", prior=prior,
                                    calibration=calibration, home=setup)
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
                if sha(output / "replay-plan.json") != contract["frozen_inputs"]["replay-plan.json"]:
                    raise ValueError("Frozen replay plan bytes changed")
                batch = requests[offset:offset + NATIVE_BATCH]
                observations.update((row["id"], row) for row in execute(heading, batch, "test"))
                report["test_releases"] = len(observations)
                publish("replaying")
            report["diagnosis"] = summarize(plan, observations)
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
    for name in ("parent-study", "parent-audit", "fallback-study", "fallback-audit", "reference", "assets",
                 "gpu-lock", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--gpu-wait-s", type=int, default=7 * 3600)
    args = parser.parse_args()
    if args.output.exists() or args.gpu_wait_s <= 0:
        parser.error("Provide a fresh output directory and a positive GPU wait bound")
    run(args)


if __name__ == "__main__":
    main()
