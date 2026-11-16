"""Test the 43 previously admitted recovery shots and repeat every capture."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path
import shutil

from .benchmark_verify import digest, read, sha
from .cell1050_rebound_study import annotate, control_requests, parent_inputs
from .corrected_grid import source_manifest
from .coverage_training_study import append, share_native_environment
from .gpu_slot import foreign_processes, gpu_slot
from .intergrid_candidate_replay_study import copy_setup_proofs
from .intergrid_matched_route_study import outcome
from .intergrid_route_labels import LABEL_VERSION, annotate_routes
from .intergrid_route_selector_plan import STANDARD_CAPTURES
from .intergrid_setup_drop_plan import load_setup_drop_plan
from .parallel_capture_grid_study import NATIVE_BATCH, STORAGE_RESERVE_BYTES, control_mirror
from .parallel_capture_grid_workers import load_resolution
from .policy64_contract import load_runtime_reference
from .sweep_worker import atomic
from .table_training_study import control_gate


VERSION = "sircus-intergrid-setup-drop-v1"
REPEAT_COUNT = 3


def repeat_requests(requests, observations):
    """Retain every first-pass capture's literal configuration for fresh drops."""
    return [
        dict(request, id=f"{request['id']}-repeat-{index:02d}")
        for request in requests if outcome(observations[request["id"]]) in STANDARD_CAPTURES
        for index in range(1, REPEAT_COUNT + 1)
    ]


def summarize(plan, first, repeats):
    cases = []
    for case in plan["cases"]:
        shots = []
        for request in case["requests"]:
            identifier = request["id"]
            first_outcome = outcome(first[identifier])
            repeat_outcomes = [
                outcome(repeats[f"{identifier}-repeat-{index:02d}"])
                for index in range(1, REPEAT_COUNT + 1)
            ] if first_outcome in STANDARD_CAPTURES else []
            shots.append(dict(
                id=identifier, intended_route=request["intended_route"], first_outcome=first_outcome,
                repeat_outcomes=repeat_outcomes,
                confirmed=bool(repeat_outcomes) and all(label in STANDARD_CAPTURES for label in repeat_outcomes),
            ))
        layout = case["layout"]
        cases.append(dict(
            id=layout["id"], bucket_xy_m=layout["bucket_xy_m"], shots=shots,
            first_pass_drops=len(shots),
            first_pass_captures=sum(shot["first_outcome"] in STANDARD_CAPTURES for shot in shots),
            repeat_drops=sum(len(shot["repeat_outcomes"]) for shot in shots),
            repeat_captures=sum(label in STANDARD_CAPTURES for shot in shots for label in shot["repeat_outcomes"]),
            confirmed_shots=sum(shot["confirmed"] for shot in shots),
        ))
    totals = {key: sum(case[key] for case in cases) for key in (
        "first_pass_drops", "first_pass_captures", "repeat_drops", "repeat_captures", "confirmed_shots",
    )}
    return dict(
        positions=len(cases), positions_with_capture=sum(case["first_pass_captures"] > 0 for case in cases),
        positions_confirmed=sum(case["confirmed_shots"] > 0 for case in cases), cases=cases, **totals,
    )


def run(args):
    plan, binding = load_setup_drop_plan(args.setup_study, args.setup_audit, args.fallback_study, args.parent_study)
    if sum(len(case["requests"]) for case in plan["cases"]) != 43:
        raise ValueError("Expected exactly 43 archived admitted requests")
    run_plan(args, plan, binding, args.setup_study, args.fallback_study / "training-prior.json",
             version=VERSION, source_key="setup_source")


def run_plan(args, plan, binding, proof_study, prior_source, *, version, source_key):
    """Replay frozen prepared commands before scheduling repeats and controls."""
    from .cell1050_heading_native import HeadingEngine
    from .table_correction_native import MatchedEngine

    output, parent = args.output, args.parent_study
    spec, parent_binding, _ = parent_inputs(parent, args.parent_audit)
    source = Path(__file__).resolve().parents[2]
    ref = load_runtime_reference(args.reference, args.assets, source)
    if (binding["parent"] != parent_binding or digest(ref["scene"]) != spec["identity"]["scene_sha256"]
            or ref["asset_hashes"] != spec["identity"]["asset_sha256"]
            or any(importlib.metadata.version(name) != pinned for name, pinned in spec["packages"].items())):
        raise ValueError("Scene, setup lineage, assets or dependencies differ from the audited controls")
    requests = [request for case in plan["cases"] for request in case["requests"]]
    if not requests or len({request["id"] for request in requests}) != len(requests):
        raise ValueError("Expected a nonempty plan of distinct admitted requests")
    output.mkdir(parents=True, exist_ok=False)
    report = dict(status="running", controls={}, test_requests=len(requests), test_releases=0,
                  repeat_requests=0, repeat_releases=0)
    packet_number = 0

    def publish(phase):
        report["phase"] = phase
        atomic(output / "report.json", report)
        print(json.dumps({key: value for key, value in report.items() if key != "result"}), flush=True)

    def execute(engine, rows, phase):
        nonlocal packet_number
        if foreign_processes():
            raise RuntimeError("Another process is using the leased 4090")
        if shutil.disk_usage(output).free < STORAGE_RESERVE_BYTES:
            raise RuntimeError("Insufficient space for native trace evidence")
        control = phase.startswith("controls_")
        directory = "native-controls" if control else "native-heading"
        if not control:
            for xy in sorted({tuple(row["bucket_xy_m"]) for row in rows}):
                load_resolution(engine, list(xy))
        packet = f"requests/batch-{packet_number:05d}.json"
        packet_number += 1
        atomic(output / packet, dict(phase=phase, rows=rows))
        measured, timing = engine.evaluate(rows, phase)
        if ([row["id"] for row in measured] != [row["id"] for row in rows]
                or any(row["native_release"] is not True for row in measured)):
            raise RuntimeError("A frozen admitted request did not release")
        adapter = engine.engine.adapter if control else engine.adapter
        if adapter.env.effective_options != expected_options or adapter.env.compiled_geometry != expected_geometry:
            raise RuntimeError("Native physics differs from the pinned controls")
        if not control:
            annotate(measured, output / directory)
            annotate_routes(measured, output / directory)
            for row, request in zip(measured, rows, strict=True):
                row["intended_route"] = request["intended_route"]
        packet_hash = sha(output / packet)
        for row in measured:
            row.update(request_packet=packet, request_packet_sha256=packet_hash, adapter_directory=directory)
        append(output / "rows.jsonl", measured)
        append(output / "timing.jsonl", [dict(packet=packet, phase=phase, timing=timing)])
        return measured

    def controls(phase):
        gate = control_gate(execute(matched, control_requests(control_sources, phase), phase))
        report["controls"][phase] = gate
        if gate["status"] != "passed":
            raise RuntimeError(f"{phase} failed")
        publish(phase)

    def run_test(rows, name, plan_file, expected_hash):
        observations = {}
        for offset in range(0, len(rows), NATIVE_BATCH):
            if sha(output / plan_file) != expected_hash:
                raise ValueError(f"Frozen {plan_file} changed")
            batch = rows[offset:offset + NATIVE_BATCH]
            observations.update((row["id"], row) for row in execute(heading, batch, "test"))
            report["test_releases"] += len(batch)
            if name == "repeating":
                report["repeat_releases"] = len(observations)
            publish(name)
        return observations

    try:
        atomic(output / "drop-plan.json", plan)
        prior = output / "training-prior.json"
        shutil.copyfile(prior_source, prior)
        if sha(prior) != binding["training_prior_sha256"]:
            raise ValueError("Geometry prior differs from audited setup search")
        contract = dict(
            version=version, parent=parent_binding, identity=spec["identity"], scene=spec["scene"],
            physics_substeps=64, label_version=LABEL_VERSION,
            repeat_count=REPEAT_COUNT, source_sha256=source_manifest(source),
            frozen_inputs={name: sha(output / name) for name in ("drop-plan.json", "training-prior.json")},
        )
        contract[source_key] = binding
        contract["fingerprint"] = digest(contract)
        atomic(output / "contract.json", contract)
        for name in ("calibration.json", "verification.json"):
            shutil.copyfile(parent / name, output / name)
        report["setup_proofs"] = copy_setup_proofs(proof_study, output, binding["proof_manifest"], requests)
        publish("drop_plan_frozen")

        def waiting(status):
            report["gpu_wait"] = status
            publish("waiting_for_4090")

        with gpu_slot(args.gpu_lock, timeout_s=args.gpu_wait_s, poll_s=15, notify=waiting):
            report.pop("gpu_wait", None)
            publish("running_controls_pre")
            setup, calibration = read(parent / "setup-home.json"), read(parent / "calibration.json")
            heading = HeadingEngine(ref, args.assets, output / "native-heading", prior=read(prior),
                                    calibration=calibration, home=setup)
            control_sources = read(parent / "control-replays.json")
            mirror = control_mirror(parent, output, control_sources, False)
            matched = MatchedEngine(ref, args.assets, output / "native-controls", parent=mirror,
                                    sources=control_sources, prior=read(parent / "old-prior.json"),
                                    calibration=calibration, home=setup)
            share_native_environment(heading, matched)
            expected_options = read(parent / "native-yaw/effective-options.json")
            expected_geometry = read(parent / "native-yaw/native-compiled-geometry.json")
            controls("controls_pre")
            first = run_test(requests, "first_pass", "drop-plan.json", contract["frozen_inputs"]["drop-plan.json"])
            repeats = repeat_requests(requests, first)
            atomic(output / "repeat-plan.json", dict(requests=repeats))
            report.update(repeat_requests=len(repeats), repeat_plan_sha256=sha(output / "repeat-plan.json"))
            publish("repeats_frozen")
            confirmed = run_test(repeats, "repeating", "repeat-plan.json", report["repeat_plan_sha256"])
            report["result"] = summarize(plan, first, confirmed)
            controls("controls_post")
        atomic(output / "evidence-manifest.json", {
            path.relative_to(output).as_posix(): sha(path) for path in sorted(output.rglob("*"))
            if path.is_file() and path.name != "report.json"
        })
        report.update(status="completed", evidence_manifest_sha256=sha(output / "evidence-manifest.json"))
        publish("completed")
    except Exception as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        publish("failed")
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("parent-study", "parent-audit", "fallback-study", "setup-study", "setup-audit",
                 "reference", "assets", "gpu-lock", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--gpu-wait-s", type=int, default=7 * 3600)
    args = parser.parse_args()
    if args.output.exists() or args.gpu_wait_s <= 0:
        parser.error("Provide a fresh output directory and positive GPU wait bound")
    run(args)


if __name__ == "__main__":
    main()
