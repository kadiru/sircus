"""Evaluate frozen bounce models once per route at the same 50 unseen positions."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import as_completed
import importlib.metadata
from itertools import combinations
import json
from pathlib import Path
import shutil

from .benchmark_verify import digest, read, sha
from .cell1050_rebound_study import annotate, control_requests, parent_inputs
from .corrected_grid import source_manifest
from .coverage_training_study import append, share_native_environment
from .gpu_slot import foreign_processes, gpu_slot
from .intergrid_matched_route_plan import make_layouts
from .intergrid_route_labels import LABEL_VERSION, annotate_routes
from .intergrid_route_models import MODELS, load_checkpoint
from .intergrid_route_plan import ROUTES
from .intergrid_route_study import PER_ROUTE_TEST, route_library, test_proposals
from .parallel_capture_grid_study import NATIVE_BATCH, STORAGE_RESERVE_BYTES, control_mirror
from .parallel_capture_grid_workers import load_resolution, open_workers, prepare_proposals
from .policy64_contract import load_runtime_reference
from .sweep_worker import atomic
from .table_training_study import control_gate

VERSION = "sircus-intergrid-matched-route-study-v1"
POSITIONS = 50
WORKERS = 8


def load_model_inputs(study, audit_path):
    """Bind the previous audit, checkpoint, training sources and layouts."""
    names = dict(report="report.json", manifest="evidence-manifest.json",
                 checkpoint="route-models.json", dataset="training-dataset.jsonl",
                 geometry_prior="training-prior.json", splits="splits.json")
    binding = {key: sha(study / name) for key, name in names.items()}
    binding["audit"] = sha(audit_path)
    audit, report, manifest = (read(audit_path), read(study / names["report"]), read(study / names["manifest"]))
    if (audit["status"] != "passed" or report["status"] != "completed"
            or audit["report_sha256"] != binding["report"]
            or audit["evidence_manifest_sha256"] != binding["manifest"]
            or report["evidence_manifest_sha256"] != binding["manifest"]
            or audit["checkpoint_sha256"] != binding["checkpoint"]
            or any(manifest[names[key]] != binding[key]
                   for key in ("checkpoint", "dataset", "geometry_prior", "splits"))):
        raise ValueError("Frozen route inputs differ from the previously audited run")
    checkpoint = load_checkpoint(study / names["checkpoint"])
    if checkpoint["routes"] != list(ROUTES) or checkpoint["model_names"] != list(MODELS):
        raise ValueError("Expected the frozen pooled, conditioned and specialist models")
    with (study / names["dataset"]).open() as stream:
        rows = [json.loads(line) for line in stream]
    if len(rows) != checkpoint["training_rows"]:
        raise ValueError("Frozen route dataset length differs from the checkpoint")
    previous = read(study / names["splits"])["splits"]
    training_xy = sorted({tuple(row["bucket_xy_m"]) for row in rows})
    layouts = previous["train"] + previous["test"] + [{"bucket_xy_m": list(xy)} for xy in training_xy]
    return checkpoint, rows, read(study / names["geometry_prior"]), layouts, binding


def choose(layout, proposals, admitted):
    by_id = {row["proposal_id"]: row for row in admitted}
    if len(by_id) != len(admitted):
        raise ValueError("Expected at most one feasible branch per proposal")
    slots = []
    for method in MODELS:
        for route in ROUTES:
            identifier = f"{layout['id']}-{method}-{route}"
            chosen = next((by_id[row["id"]] for row in proposals
                           if row["method"] == method and row["intended_route"] == route and row["id"] in by_id), None)
            slots.append(dict(id=identifier, method=method, context_id=layout["id"],
                              bucket_xy_m=layout["bucket_xy_m"], stratum=layout["stratum"], intended_route=route,
                              request=None if chosen is None else dict(chosen, id=identifier, stratum=layout["stratum"])))
    return slots


def outcome(row):
    if row is None:
        return "setup_rejected"
    if row["invalid_physics"]:
        return "invalid_physics"
    if row["route_success"]:
        return row["shot_route"] or "unclassified_capture"
    if row["experimental_route"]["robot_assisted_capture"]:
        return "arm_assisted_capture"
    return "flight_miss"


def summarize(slots, observations):
    if (len({slot["id"] for slot in slots}) != len(slots)
            or set(observations) != {slot["id"] for slot in slots if slot["request"] is not None}):
        raise ValueError("Expected one observation per admitted frozen choice")
    counts = {method: {route: Counter() for route in ROUTES} for method in MODELS}
    success = defaultdict(dict)
    for slot in slots:
        method, route = slot["method"], slot["intended_route"]
        actual = outcome(observations.get(slot["id"]))
        counts[method][route][actual] += 1
        success[slot["context_id"], route][method] = actual == route
    paired = {route: {f"{a}_vs_{b}": Counter() for a, b in combinations(MODELS, 2)} for route in ROUTES}
    for (_, route), models in success.items():
        if set(models) != set(MODELS):
            raise ValueError("Each requested route needs all models at each position")
        for a, b in combinations(MODELS, 2):
            label = "both" if models[a] and models[b] else f"{a}_only" if models[a] else f"{b}_only" if models[b] else "neither"
            paired[route][f"{a}_vs_{b}"][label] += 1
    methods = {}
    for method, routes in counts.items():
        methods[method] = {}
        for route, outcomes in routes.items():
            attempts = sum(outcomes.values())
            if attempts != POSITIONS:
                raise ValueError("Each model and route needs exactly 50 positions")
            methods[method][route] = dict(attempts=attempts, releases=attempts - outcomes["setup_rejected"],
                                          captures=outcomes[route], capture_rate=outcomes[route] / attempts,
                                          outcomes=dict(outcomes))
    return dict(methods=methods, paired={route: {pair: dict(values) for pair, values in groups.items()}
                                        for route, groups in paired.items()})


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
    checkpoint, rows, prior, previous, binding = load_model_inputs(args.model_study, args.model_audit)
    prior_contract = read(args.model_study / "contract.json")
    if (prior_contract["parent"] != parent_binding or prior_contract["identity"] != spec["identity"]
            or prior_contract["physics_substeps"] != 64 or prior_contract["label_version"] != LABEL_VERSION):
        raise ValueError("The frozen models were evaluated with a different simulation contract")
    layouts, provenance = make_layouts(args.history_root, read(parent / "grid.json"),
                                       scene=ref["scene"], assets=args.assets, previous_layouts=previous)
    if len(layouts) != POSITIONS or len({row["id"] for row in layouts}) != POSITIONS:
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
        atomic(output / "training-prior.json", prior)
        shutil.copyfile(args.model_study / "route-models.json", output / "route-models.json")
        if sha(output / "route-models.json") != binding["checkpoint"]:
            raise ValueError("Frozen checkpoint bytes changed during copy")
        contract = dict(version=VERSION, parent=parent_binding, identity=spec["identity"], scene=spec["scene"],
                        model_study=binding, test_positions=POSITIONS, methods=list(MODELS), routes=list(ROUTES),
                        attempts_per_method_per_route_per_position=1, candidate_budget_per_route=PER_ROUTE_TEST,
                        physics_substeps=64, label_version=LABEL_VERSION, source_sha256=source_manifest(source),
                        frozen_inputs={name: sha(output / name) for name in ("layouts.json", "training-prior.json")})
        contract["fingerprint"] = digest(contract)
        atomic(output / "contract.json", contract)
        setup, calibration = read(parent / "setup-home.json"), read(parent / "calibration.json")
        for name in ("calibration.json", "verification.json"):
            shutil.copyfile(parent / name, output / name)
        kwargs = dict(prior=prior, calibration=calibration, home=setup)
        library = route_library(rows)
        prepared = {}
        publish("preparing_test")
        with open_workers(WORKERS, ref, args.assets, output / "native-heading", kwargs) as workers:
            pending = {}
            for layout in layouts:
                proposals = test_proposals(layout, library, checkpoint)
                pending[workers.submit(prepare_proposals, proposals, True)] = (layout, proposals)
            for future in as_completed(pending):
                layout, proposals = pending[future]
                admitted, records = future.result()
                atomic(output / "test" / (layout["id"] + ".json"),
                       dict(proposals=proposals, admitted=admitted, records=records))
                prepared[layout["id"]] = choose(layout, proposals, admitted)
                report["test_prepared"] = len(prepared)
                if len(prepared) % 5 == 0:
                    publish("preparing_test")
        slots = [slot for layout in layouts for slot in prepared[layout["id"]]]
        atomic(output / "choices.json", dict(checkpoint_sha256=binding["checkpoint"], slots=slots))
        report["choices_sha256"] = sha(output / "choices.json")
        publish("choices_frozen")

        def waiting(status):
            report["gpu_wait"] = status
            publish("waiting_for_4090")

        observations = {}
        requests = [slot["request"] for slot in slots if slot["request"] is not None]
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
                if (sha(output / "route-models.json") != binding["checkpoint"]
                        or sha(output / "choices.json") != report["choices_sha256"]):
                    raise ValueError("Checkpoint or choices changed during the test")
                batch = requests[offset:offset + NATIVE_BATCH]
                observations.update((row["id"], row) for row in execute(heading, batch, "test"))
                report["test_releases"] = len(observations)
                publish("testing")
            report["comparison"] = summarize(slots, observations)
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
                 "reference", "assets", "gpu-lock", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--gpu-wait-s", type=int, default=7 * 3600)
    args = parser.parse_args()
    if args.output.exists() or args.gpu_wait_s <= 0:
        parser.error("Provide a fresh output directory and a positive GPU wait bound")
    run(args)


if __name__ == "__main__":
    main()
