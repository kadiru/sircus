"""Collect new shots around verified direct and table captures at identical positions."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import as_completed
import importlib.metadata
import json
from pathlib import Path
import shutil

from .benchmark_verify import digest, read, sha
from .cell1050_followup_plan import geometry_key
from .cell1050_rebound_study import annotate, control_requests, parent_inputs
from .corrected_grid import source_manifest
from .coverage_training_study import append, share_native_environment
from .gpu_slot import foreign_processes, gpu_slot
from .intergrid_dataset import _observation, _reject
from .intergrid_matched_route_study import load_model_inputs
from .intergrid_paired_collection_plan import proposals_for
from .intergrid_route_labels import LABEL_VERSION, annotate_routes
from .intergrid_route_plan import ROUTES
from .parallel_capture_grid_study import NATIVE_BATCH, STORAGE_RESERVE_BYTES, control_mirror
from .parallel_capture_grid_workers import load_resolution, open_workers, prepare_proposals
from .policy64_contract import load_runtime_reference
from .sweep_worker import atomic
from .table_training_study import control_gate

VERSION = "sircus-intergrid-paired-route-collection-v1"
BUDGET_PER_ROUTE = 16
WORKERS = 8


def collection_plan(rows, matched, audit_path, model_binding, parent_binding):
    """Authenticate the matched test and select only positions with both measured routes."""
    audit, report, spec = read(audit_path), read(matched / "report.json"), read(matched / "contract.json")
    manifest = read(matched / "evidence-manifest.json")
    binding = {name: sha(matched / name)
               for name in ("report.json", "contract.json", "evidence-manifest.json", "rows.jsonl", "choices.json")}
    binding["audit"] = sha(audit_path)
    if (audit["status"] != "passed" or report["status"] != "completed"
            or audit["report_sha256"] != binding["report.json"]
            or audit["evidence_manifest_sha256"] != binding["evidence-manifest.json"]
            or report["evidence_manifest_sha256"] != binding["evidence-manifest.json"]
            or audit["checkpoint_sha256"] != model_binding["checkpoint"]
            or spec["model_study"] != model_binding or spec["parent"] != parent_binding
            or any(binding[name] != manifest[name] for name in ("contract.json", "rows.jsonl", "choices.json"))):
        raise ValueError("Matched-study inputs do not match the previously audited evidence")

    seeds = defaultdict(lambda: {route: [] for route in ROUTES})
    known = defaultdict(set)
    original_tests = set()
    for row in rows:
        xy = tuple(row["bucket_xy_m"])
        known[xy].add(geometry_key(row))
        if row["shot_route"] in ROUTES and row["route_success"] and not row["label_conflict"]:
            seeds[xy][row["shot_route"]].append(row)
    choices = read(matched / "choices.json")["slots"]
    frozen = {slot["id"]: slot for slot in choices}
    if len(frozen) != 300:
        raise ValueError("Expected the 300 authenticated matched decisions")
    for slot in choices:
        original_tests.add(tuple(slot["bucket_xy_m"]))
        if slot["request"] is not None:
            known[tuple(slot["bucket_xy_m"])].add(geometry_key(slot["request"]))
    with (matched / "rows.jsonl").open("rb") as stream:
        for raw in stream:
            row = json.loads(raw)
            if row["phase"] != "test":
                continue
            slot = frozen[row["id"]]
            if row["bucket_xy_m"] != slot["bucket_xy_m"] or row["intended_route"] != slot["intended_route"]:
                raise ValueError("Recorded shot differs from its frozen matched layout")
            xy = tuple(row["bucket_xy_m"])
            if row["shot_route"] in ROUTES and _reject(row, (row["context_id"], list(xy))) is None:
                _, sample = _observation(row, matched, raw, manifest, (row["context_id"], list(xy)))
                seeds[xy][row["shot_route"]].append(dict(sample, shot_route=row["shot_route"]))

    plan = []
    for xy, routes in sorted(seeds.items()):
        if not all(routes.values()):
            continue
        layout = dict(id=f"intergrid-paired-route-train-{len(plan):02d}", bucket_xy_m=list(xy),
                      previously_matched_test=xy in original_tests)
        proposals = proposals_for(layout, routes, budget=BUDGET_PER_ROUTE, known_geometry=known[xy])
        plan.append(dict(layout=layout, proposals=proposals,
                         prior_captures={route: len(shots) for route, shots in routes.items()}))
    if not plan:
        raise ValueError("No position has a verified capture by both routes")
    return plan, binding


def collection_rows(output, plan, trace_hashes):
    """Export valid new native releases with original evidence links and measured labels."""
    positions = {item["layout"]["id"]: item["layout"]["bucket_xy_m"] for item in plan}
    fresh, excluded, seen = [], Counter(), set()
    with (output / "rows.jsonl").open("rb") as stream:
        for raw in stream:
            row = json.loads(raw)
            if row["phase"] != "train":
                continue
            context, xy = row["context_id"], row["bucket_xy_m"]
            if positions[context] != xy:
                raise ValueError("Collection release differs from its frozen bucket position")
            reason = _reject(row, (context, xy))
            if reason:
                excluded[reason] += 1
                continue
            key, sample = _observation(row, output, raw, trace_hashes, (context, xy))
            if key in seen:
                raise ValueError("Duplicate new collection command")
            seen.add(key)
            fresh.append(dict(sample, intended_route=row["intended_route"], shot_route=row["shot_route"],
                              table_rebounds_before_entry=row["table_rebounds_before_entry"],
                              proposal_id=row["proposal_id"]))
    return fresh, dict(excluded)


def outcome(row):
    if row["invalid_physics"]:
        return "invalid_physics"
    if row["route_success"]:
        return row["shot_route"] or "unclassified_capture"
    if row["experimental_route"]["robot_assisted_capture"]:
        return "arm_assisted_capture"
    return "flight_miss"


def summarize(plan, prepared, observations):
    if set(observations) != {row["id"] for artifact in prepared.values() for row in artifact["admitted"]}:
        raise ValueError("Every admitted collection request must have exactly one release")
    cells, totals, captured_both = [], {route: Counter() for route in ROUTES}, 0
    for item in plan:
        layout = item["layout"]
        artifact = prepared[layout["id"]]
        realized = Counter()
        by_intent = {}
        for route in ROUTES:
            count = Counter(proposals=sum(row["intended_route"] == route for row in artifact["proposals"]))
            for request in artifact["admitted"]:
                if request["intended_route"] == route:
                    measured = outcome(observations[request["id"]])
                    realized[measured] += 1
                    count.update({"releases": 1, measured: 1})
            count["setup_rejected"] = count["proposals"] - count["releases"]
            by_intent[route] = dict(count)
            totals[route].update(count)
        captured_both += int(all(realized[route] > 0 for route in ROUTES))
        cells.append(dict(layout, prior_captures=item["prior_captures"], by_intent=by_intent,
                          realized=dict(realized)))
    return dict(cells=cells, positions_with_both=captured_both, totals={route: dict(count) for route, count in totals.items()})


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
    _, rows, prior, _, binding = load_model_inputs(args.model_study, args.model_audit)
    prior_contract = read(args.model_study / "contract.json")
    if (prior_contract["parent"] != parent_binding or prior_contract["identity"] != spec["identity"]
            or prior_contract["physics_substeps"] != 64 or prior_contract["label_version"] != LABEL_VERSION):
        raise ValueError("The frozen models were evaluated with a different simulation contract")
    plan, matched_binding = collection_plan(rows, args.matched_study, args.matched_audit, binding, parent_binding)
    output.mkdir(parents=True, exist_ok=False)
    report = dict(status="running", controls={}, collection_prepared=0, train_releases=0)
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
        atomic(output / "collection-plan.json", plan)
        atomic(output / "training-prior.json", prior)
        contract = dict(version=VERSION, parent=parent_binding, identity=spec["identity"], scene=spec["scene"],
                        model_study=binding, matched_study=matched_binding, positions=len(plan),
                        routes=list(ROUTES), budget_per_route=BUDGET_PER_ROUTE, physics_substeps=64,
                        label_version=LABEL_VERSION, source_sha256=source_manifest(source),
                        frozen_inputs={name: sha(output / name)
                                       for name in ("collection-plan.json", "training-prior.json")})
        contract["fingerprint"] = digest(contract)
        atomic(output / "contract.json", contract)
        setup, calibration = read(parent / "setup-home.json"), read(parent / "calibration.json")
        for name in ("calibration.json", "verification.json"):
            shutil.copyfile(parent / name, output / name)
        kwargs = dict(prior=prior, calibration=calibration, home=setup)
        prepared = {}
        publish("preparing_collection")
        with open_workers(WORKERS, ref, args.assets, output / "native-heading", kwargs) as workers:
            pending = {}
            for item in plan:
                layout, proposals = item["layout"], item["proposals"]
                pending[workers.submit(prepare_proposals, proposals, True)] = (layout, proposals)
            for future in as_completed(pending):
                layout, proposals = pending[future]
                admitted, records = future.result()
                artifact = dict(proposals=proposals, admitted=admitted, records=records)
                atomic(output / "collection" / (layout["id"] + ".json"), artifact)
                prepared[layout["id"]] = artifact
                report["collection_prepared"] = len(prepared)
                publish("preparing_collection")
        requests = [row for item in plan for row in prepared[item["layout"]["id"]]["admitted"]]
        if not requests:
            raise RuntimeError("No new collection shot passed setup")
        atomic(output / "admitted.json", requests)
        report["admitted_sha256"] = sha(output / "admitted.json")
        report["admitted"] = len(requests)
        publish("requests_frozen")

        def waiting(status):
            report["gpu_wait"] = status
            publish("waiting_for_4090")

        observations = {}
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
                if sha(output / "admitted.json") != report["admitted_sha256"]:
                    raise ValueError("Collection requests changed after the first release")
                batch = requests[offset:offset + NATIVE_BATCH]
                observations.update((row["id"], row) for row in execute(heading, batch, "train"))
                report["train_releases"] = len(observations)
                publish("collecting")
            report["collection"] = summarize(plan, prepared, observations)
            run_controls("controls_post")
        trace_hashes = {f"native-heading/{row['native_cache_file']}": row["native_cache_sha256"]
                        for row in observations.values()}
        fresh, excluded = collection_rows(output, plan, trace_hashes)
        (output / "new-training-dataset.jsonl").touch(exist_ok=False)
        append(output / "new-training-dataset.jsonl", fresh)
        report.update(valid_training_rows=len(fresh), excluded_from_dataset=excluded)
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
    for name in ("model-study", "model-audit", "matched-study", "matched-audit", "parent-study", "parent-audit",
                 "reference", "assets", "gpu-lock", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--gpu-wait-s", type=int, default=7 * 3600)
    args = parser.parse_args()
    if args.output.exists() or args.gpu_wait_s <= 0:
        parser.error("Provide a fresh output directory and a positive GPU wait bound")
    run(args)


if __name__ == "__main__":
    main()
