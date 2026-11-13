"""Collect paired route training shots and compare three frozen model designs."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from itertools import combinations
import math
import importlib.metadata
import json
from pathlib import Path
import shutil

from .benchmark_verify import digest, read, sha
from .cell1050_rebound_study import annotate, control_requests, parent_inputs
from .corrected_grid import source_manifest
from .coverage_training_study import append, share_native_environment
from .gpu_slot import foreign_processes, gpu_slot
from .intergrid_dataset import _observation, _reject
from .intergrid_route_labels import LABEL_VERSION, annotate_routes, historical_labels
from .intergrid_route_models import MODELS, fit_route_models, load_checkpoint, propose_route_candidates, save_checkpoint
from .intergrid_route_plan import PER_ROUTE_LIMIT, ROUTES, collection_proposals, make_splits
from .parallel_capture_grid_study import NATIVE_BATCH, STORAGE_RESERVE_BYTES, control_mirror
from .parallel_capture_grid_workers import load_resolution, open_workers, prepare_proposals
from .policy64_contract import load_runtime_reference
from .sweep_worker import atomic
from .table_training_study import control_gate

VERSION = "sircus-intergrid-route-study-v1"
METHODS = MODELS
MODEL_SEED = 50284
PER_ROUTE_TEST = 4
CANDIDATE_BUDGET = 2 * PER_ROUTE_TEST
WORKERS = 6


def load_prior(study, audit_path, history_root):
    manifest_path = study / "evidence-manifest.json"
    report, audit, manifest = read(study / "report.json"), read(audit_path), read(manifest_path)
    binding = dict(report=sha(study / "report.json"), audit=sha(audit_path), manifest=sha(manifest_path),
                   dataset=sha(study / "training-dataset.jsonl"), splits=sha(study / "splits.json"),
                   geometry_prior=sha(study / "training-prior.json"))
    if (report["status"] != "completed" or audit["status"] != "passed"
            or audit["report_sha256"] != binding["report"]
            or audit["evidence_manifest_sha256"] != binding["manifest"]
            or report["evidence_manifest_sha256"] != binding["manifest"]
            or manifest["training-dataset.jsonl"] != binding["dataset"]
            or manifest["splits.json"] != binding["splits"]
            or manifest["training-prior.json"] != binding["geometry_prior"]):
        raise ValueError("Expected the completed and independently audited intergrid proposer")
    with (study / "training-dataset.jsonl").open() as stream:
        rows = [json.loads(line) for line in stream]
    # Earlier experiments also marked a small number of fixed-grid trials TEST.
    # Keep their original split out of both the training data and source menu.
    eligible = [row for row in rows if not any(row.get(key) in {"test", "validation"} for key in ("phase", "role"))]
    return historical_labels(eligible, history_root), binding


def fresh_rows(output, layouts, trace_hashes):
    positions = {row["id"]: row["bucket_xy_m"] for row in layouts}
    fresh, exclusions, seen = [], Counter(), set()
    if not (output / "rows.jsonl").exists():
        return fresh, {}
    with (output / "rows.jsonl").open("rb") as stream:
        for raw in stream:
            row = json.loads(raw)
            if row["phase"] != "train":
                continue
            context = row["context_id"]
            if row["bucket_xy_m"] != positions[context] or row["role"] != "train":
                raise ValueError("Collection shot differs from the frozen TRAIN layout")
            reason = _reject(row, (context, positions[context]))
            if reason:
                exclusions[reason] += 1
                continue
            key, sample = _observation(row, output, raw, trace_hashes, (context, positions[context]))
            if key in seen:
                raise ValueError("Duplicate fresh training command")
            seen.add(key)
            fresh.append(dict(sample, intended_route=row["intended_route"], shot_route=row["shot_route"],
                              table_rebounds_before_entry=row["table_rebounds_before_entry"]))
    return fresh, dict(exclusions)


def route_library(rows):
    result = {route: defaultdict(list) for route in ROUTES}
    for row in rows:
        route = row["shot_route"]
        if route in result and row["route_success"] and not row["label_conflict"]:
            result[route][tuple(row["bucket_xy_m"])].append(row)
    return {route: {xy: sorted(shots, key=lambda row: (
        -len(row["lineage"]), -row["route_quality"], row["id"])) for xy, shots in locations.items()}
            for route, locations in result.items()}


def source_menu(layout, library):
    """Four known shots per route, spread over nearby locations if possible."""
    result = []
    target = layout["bucket_xy_m"]
    for route in ROUTES:
        locations = library[route]
        order = sorted(locations, key=lambda xy: (math.dist(xy, target), xy))
        selected = []
        seen = set()
        # Start with two from each of two nearest locations; fill from farther locations only if needed.
        for limit in (2, PER_ROUTE_TEST):
            for xy in order:
                for row in locations[xy][:limit]:
                    if row["id"] in seen:
                        continue
                    selected.append(dict(row, intended_route=route))
                    seen.add(row["id"])
                    if len(selected) == PER_ROUTE_TEST:
                        break
                if len(selected) == PER_ROUTE_TEST:
                    break
            if len(selected) == PER_ROUTE_TEST:
                break
        if len(selected) != PER_ROUTE_TEST:
            raise ValueError(f"Insufficient verified training sources for {route}")
        result.extend(selected)
    return result


def test_proposals(layout, library, checkpoint):
    sources = source_menu(layout, library)
    return [row for method in METHODS for row in propose_route_candidates(
        checkpoint, method, layout["bucket_xy_m"], sources, context_id=layout["id"], limit=CANDIDATE_BUDGET)]


def fit_gate(layouts, packets, train_observations):
    """Fit a common pre-release route preference from all paired collection attempts."""
    sites = []
    for layout in layouts:
        packet = packets[layout["id"]]
        by_proposal = {row["proposal_id"]: row for row in packet["admitted"]}
        counts = {route: dict(proposals=0, releases=0, intended_captures=0) for route in ROUTES}
        realized = Counter()
        for proposal in packet["proposals"]:
            route = proposal["intended_route"]
            counts[route]["proposals"] += 1
            request = by_proposal.get(proposal["id"])
            if request is None:
                continue
            observation = train_observations[request["id"]]
            if observation["intended_route"] != route:
                raise ValueError("Measured and planned route intents differ")
            counts[route]["releases"] += 1
            measured_route = observation["shot_route"]
            if measured_route in ROUTES:
                realized[measured_route] += 1
                counts[route]["intended_captures"] += int(measured_route == route)
            elif observation["route_success"]:
                realized["unclassified"] += 1
        if any(value["proposals"] != PER_ROUTE_LIMIT for value in counts.values()):
            raise ValueError("TRAIN route proposal budgets differ")
        sites.append(dict(id=layout["id"], bucket_xy_m=layout["bucket_xy_m"], counts=counts,
                          realized=dict(realized)))
    gate = dict(version="sircus-route-gate-v1", sites=sites, neighbors=5, prior_alpha=1, prior_beta=1,
                metric="measured capture of intended route per planned proposal; setup rejects count against route",
                tie_break="direct")
    return dict(gate, fingerprint=digest(gate))


def summarize_collection(gate):
    totals = {route: Counter() for route in ROUTES}
    realized = Counter()
    for site in gate["sites"]:
        for route in ROUTES:
            totals[route].update(site["counts"][route])
        realized.update(site["realized"])
    return dict(positions=len(gate["sites"]), attempted_both_routes=len(gate["sites"]),
                released_both_routes=sum(all(site["counts"][route]["releases"] for route in ROUTES)
                                         for site in gate["sites"]),
                captured_both_routes=sum(all(site["realized"].get(route) for route in ROUTES)
                                         for site in gate["sites"]),
                positions_with_capture={route: sum(bool(site["realized"].get(route)) for site in gate["sites"])
                                        for route in ROUTES},
                intended={route: dict(total) for route, total in totals.items()}, realized=dict(realized))


def training_summary(checkpoint, canonical, fresh, excluded, checkpoint_hash, gate_hash):
    return dict(prior_commands=len(canonical), new_valid_commands=len(fresh),
                new_standard_captures=sum(row["route_success"] for row in fresh), exclusions=excluded,
                training_positions=checkpoint["training_positions"], training_captures=checkpoint["training_captures"],
                training_pairs=checkpoint["training_pairs"],
                support={route: {key: value for key, value in support.items() if key != "locations"}
                         for route, support in checkpoint["route_support"].items()},
                checkpoint_sha256=checkpoint_hash, gate_sha256=gate_hash)


def choose(layout, proposed, admitted, gate):
    neighbors = sorted(gate["sites"], key=lambda row: (
        math.dist(layout["bucket_xy_m"], row["bucket_xy_m"]), row["id"]))[:gate["neighbors"]]
    scores = {route: (gate["prior_alpha"] + sum(site["counts"][route]["intended_captures"] for site in neighbors)) /
              (gate["prior_alpha"] + gate["prior_beta"] + sum(site["counts"][route]["proposals"] for site in neighbors))
              for route in ROUTES}
    order = sorted(ROUTES, key=lambda route: (-scores[route], ROUTES.index(route)))
    by_proposal = {row["proposal_id"]: row for row in admitted}
    if len(by_proposal) != len(admitted):
        raise ValueError("Each proposal can have only one admitted branch")
    slots = []
    for method in METHODS:
        choice = next((by_proposal[row["id"]] for route in order for row in proposed
                       if row["method"] == method and row["intended_route"] == route and row["id"] in by_proposal), None)
        identifier = f"{layout['id']}-{method}"
        slots.append(dict(id=identifier, method=method, context_id=layout["id"],
                          bucket_xy_m=layout["bucket_xy_m"], stratum=layout["stratum"],
                          intended_route=None if choice is None else choice["intended_route"], route_priority=order,
                          gate_scores=scores,
                          request=None if choice is None else dict(choice, id=identifier, stratum=layout["stratum"])))
    return slots


def outcome(row):
    if row is None:
        return "setup_rejected"
    if row["invalid_physics"]:
        return "invalid_physics"
    if row["route_success"]:
        return row["shot_route"] or "unclassified_capture"
    return "flight_miss"


def summarize(slots, observations):
    if (len({slot["id"] for slot in slots}) != len(slots)
            or set(observations) != {slot["id"] for slot in slots if slot["request"] is not None}):
        raise ValueError("Expected one observation per admitted frozen choice")
    counts = {method: Counter() for method in METHODS}
    intent = {method: Counter() for method in METHODS}
    by_intent = {method: {route: Counter() for route in ROUTES} for method in METHODS}
    strata = defaultdict(lambda: {method: Counter() for method in METHODS})
    paired = {f"{a}_vs_{b}": Counter() for a, b in combinations(METHODS, 2)}
    for offset in range(0, len(slots), len(METHODS)):
        group = slots[offset:offset + len(METHODS)]
        if tuple(slot["method"] for slot in group) != METHODS or len({slot["context_id"] for slot in group}) != 1:
            raise ValueError("Expected three frozen slots per position")
        success = {}
        for slot in group:
            method = slot["method"]
            result = outcome(observations.get(slot["id"]))
            counts[method][result] += 1
            strata[slot["stratum"]][method][result] += 1
            success[method] = result in (*ROUTES, "unclassified_capture")
            if slot["intended_route"] is not None:
                by_intent[method][slot["intended_route"]][result] += 1
                intent[method][slot["intended_route"] + "_releases"] += 1
                intent[method]["intended_captures"] += int(slot["intended_route"] == result)
                intent[method]["other_route_captures"] += int(result in ROUTES and result != slot["intended_route"])
        for a, b in combinations(METHODS, 2):
            label = "both" if success[a] and success[b] else a + "_only" if success[a] else b + "_only" if success[b] else "neither"
            paired[f"{a}_vs_{b}"][label] += 1
    return dict(methods={method: dict(counts[method]) for method in METHODS},
                intent={method: dict(intent[method]) for method in METHODS},
                by_intent={method: {route: dict(results) for route, results in routes.items()}
                           for method, routes in by_intent.items()},
                strata={key: {method: dict(values[method]) for method in METHODS} for key, values in strata.items()},
                paired={key: dict(value) for key, value in paired.items()})


def run(args):
    from .cell1050_heading_native import HeadingEngine
    from .table_correction_native import MatchedEngine

    parent, output = args.parent_study, args.output
    spec, parent_binding, _ = parent_inputs(parent, args.parent_audit)
    source = Path(__file__).resolve().parents[2]
    ref = load_runtime_reference(args.reference, args.assets, source)
    if (digest(ref["scene"]) != spec["identity"]["scene_sha256"]
            or ref["asset_hashes"] != spec["identity"]["asset_sha256"]
            or any(importlib.metadata.version(name) != version for name, version in spec["packages"].items())):
        raise ValueError("Scene, assets or pinned dependencies differ from the audited parent")
    canonical, binding = load_prior(args.prior_study, args.prior_audit, args.history_root)
    previous = read(args.prior_study / "splits.json")["splits"]
    splits, provenance = make_splits(args.history_root, read(parent / "grid.json"), scene=ref["scene"],
                                    assets=args.assets, previous_layouts=previous["train"] + previous["test"])
    output.mkdir(parents=True, exist_ok=False)
    report = dict(status="running", controls={}, train_releases=0, test_releases=0)
    packet_number = 0
    trace_hashes = {}

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
            raise RuntimeError("Native physics differs from the pinned source study")
        if not control:
            annotate(measured, output / directory)
            annotate_routes(measured, output / directory)
            for row, request in zip(measured, requests, strict=True):
                row["intended_route"] = request["intended_route"]
        packet_hash = sha(output / packet)
        for row in measured:
            row.update(request_packet=packet, request_packet_sha256=packet_hash, adapter_directory=directory)
            trace_hashes[directory + "/" + row["native_cache_file"]] = row["native_cache_sha256"]
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
        publish("inputs")
        if len(splits["train"]) != 24 or len(splits["test"]) != 50:
            raise ValueError("Expected 24 collection and 50 final test positions")
        if len({row["id"] for row in splits["train"] + splits["test"]}) != 74:
            raise ValueError("Split contains duplicate placement IDs")
        atomic(output / "splits.json", dict(splits=splits, provenance=provenance))
        seed_library = route_library(canonical)
        menu = [collection_proposals(layout, seed_library) for layout in splits["train"]]
        atomic(output / "collection-menu.json", menu)
        prior = read(args.prior_study / "training-prior.json")
        atomic(output / "training-prior.json", prior)
        contract = dict(version=VERSION, parent=parent_binding, identity=spec["identity"],
                        scene=spec["scene"], prior_proposer=binding, canonical_positions=33,
                        collection_positions=24, test_positions=50, methods=list(METHODS),
                        attempts_per_method_per_position=1, physics_substeps=64, seed=MODEL_SEED,
                        candidate_budget_per_method=CANDIDATE_BUDGET,
                        collection_budget_per_route=PER_ROUTE_LIMIT, label_version=LABEL_VERSION,
                        routes=list(ROUTES),
                        source_sha256=source_manifest(source),
                        frozen_inputs={name: sha(output / name) for name in
                                       ("splits.json", "collection-menu.json", "training-prior.json")})
        contract["fingerprint"] = digest(contract)
        atomic(output / "contract.json", contract)

        setup, calibration = read(parent / "setup-home.json"), read(parent / "calibration.json")
        for name in ("calibration.json", "verification.json"):
            shutil.copyfile(parent / name, output / name)
        kwargs = dict(prior=prior, calibration=calibration, home=setup)
        heading = HeadingEngine(ref, args.assets, output / "native-heading", **kwargs)
        control_sources = read(parent / "control-replays.json")
        mirror = control_mirror(parent, output, control_sources, False)
        controls = MatchedEngine(ref, args.assets, output / "native-controls", parent=mirror,
                                 sources=control_sources, prior=read(parent / "old-prior.json"),
                                 calibration=calibration, home=setup)
        share_native_environment(heading, controls)
        expected_options = read(parent / "native-yaw/effective-options.json")
        expected_geometry = read(parent / "native-yaw/native-compiled-geometry.json")
        requests = []
        slots, packets, collected = [], {}, {}
        with open_workers(WORKERS, ref, args.assets, output / "native-heading", kwargs) as workers:
            futures = [workers.submit(prepare_proposals, proposals, True) for proposals in menu]
            run_controls("controls_pre")
            for layout, proposals, future in zip(splits["train"], menu, futures, strict=True):
                admitted, records = future.result()
                packet = dict(proposals=proposals, admitted=admitted, records=records)
                packets[layout["id"]] = packet
                atomic(output / "collection" / (layout["id"] + ".json"), packet)
                report["collection_prepared"] = report.get("collection_prepared", 0) + 1
                requests.extend(admitted)
                if len(requests) >= NATIVE_BATCH:
                    measured = execute(heading, requests[:NATIVE_BATCH], "train")
                    collected.update((row["id"], row) for row in measured)
                    del requests[:NATIVE_BATCH]
                    report["train_releases"] = len(collected)
                publish("collecting")
            if requests:
                measured = execute(heading, requests, "train")
                collected.update((row["id"], row) for row in measured)
                report["train_releases"] = len(collected)
                publish("collecting")

            fresh, excluded = fresh_rows(output, splits["train"], trace_hashes)
            fit_rows = canonical + fresh
            append(output / "training-dataset.jsonl", fit_rows)
            library = route_library(fit_rows)
            gate = fit_gate(splits["train"], packets, collected)
            report["collection"] = summarize_collection(gate)
            atomic(output / "route-gate.json", gate)
            gate_hash = sha(output / "route-gate.json")
            if foreign_processes():
                raise RuntimeError("Another process is using the leased 4090")
            checkpoint = fit_route_models(fit_rows, seed=MODEL_SEED, device="cuda:0", routes=ROUTES)
            save_checkpoint(output / "route-models.json", checkpoint)
            checkpoint = load_checkpoint(output / "route-models.json")
            checkpoint_hash = sha(output / "route-models.json")
            report["training"] = training_summary(checkpoint, canonical, fresh, excluded, checkpoint_hash, gate_hash)
            publish("checkpoint_frozen")

            menus = [test_proposals(layout, library, checkpoint) for layout in splits["test"]]
            futures = [workers.submit(prepare_proposals, proposals, True) for proposals in menus]
            for layout, proposals, future in zip(splits["test"], menus, futures, strict=True):
                admitted, records = future.result()
                atomic(output / "test" / (layout["id"] + ".json"),
                       dict(proposals=proposals, admitted=admitted, records=records))
                slots.extend(choose(layout, proposals, admitted, gate))
                report["test_prepared"] = len(slots) // len(METHODS)
                if report["test_prepared"] % 5 == 0:
                    publish("preparing_test")
        atomic(output / "choices.json", dict(checkpoint_sha256=checkpoint_hash, gate_sha256=gate_hash, slots=slots))
        report["choices_sha256"] = sha(output / "choices.json")
        publish("choices_frozen")
        observations = {}
        requests = [slot["request"] for slot in slots if slot["request"] is not None]
        for offset in range(0, len(requests), NATIVE_BATCH):
            if (sha(output / "route-models.json") != checkpoint_hash or sha(output / "route-gate.json") != gate_hash
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
    for name in ("history-root", "prior-study", "prior-audit", "parent-study", "parent-audit",
                 "reference", "assets", "gpu-lock", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--gpu-wait-s", type=int, default=7 * 3600)
    # Parse the queue bound before starting the job.
    args = parser.parse_args()
    if args.output.exists() or args.gpu_wait_s <= 0:
        parser.error("Provide a fresh output directory and a positive GPU wait bound")
    with gpu_slot(args.gpu_lock, timeout_s=args.gpu_wait_s, poll_s=15,
                  notify=lambda row: print(json.dumps(row), flush=True)):
        run(args)


if __name__ == "__main__":
    main()
