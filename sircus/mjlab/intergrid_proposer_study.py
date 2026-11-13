"""Collect intermediate training shots and test a frozen continuous shot model."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
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
from .intergrid_study import proposals_at, stable_captures
from .parallel_capture_grid_study import NATIVE_BATCH, STORAGE_RESERVE_BYTES, control_mirror
from .parallel_capture_grid_workers import load_resolution, open_workers, prepare_proposals
from .policy64_contract import load_runtime_reference
from .sweep_worker import atomic
from .table_training_study import control_gate

VERSION = "sircus-intergrid-proposer-study-v1"
METHODS = ("nearest_feasible", "learned")
MODEL_SEED = 50283
WORKERS = 6


def load_prior(study, audit_path):
    report, audit = read(study / "report.json"), read(audit_path)
    manifest_path = study / "evidence-manifest.json"
    manifest = read(manifest_path)
    path = study / "dataset.jsonl"
    binding = dict(report=sha(study / "report.json"), audit=sha(audit_path),
                   manifest=sha(manifest_path), dataset=sha(path))
    if (report["status"] != "completed" or audit["status"] != "passed"
            or audit["report_sha256"] != binding["report"]
            or audit["evidence_manifest_sha256"] != binding["manifest"]
            or report["evidence_manifest_sha256"] != binding["manifest"]
            or audit["train_source_reconstructed"] is not True
            or manifest["dataset.jsonl"] != binding["dataset"]):
        raise ValueError("Expected the completed and independently audited fixed-grid training dataset")
    with path.open() as stream:
        return [json.loads(line) for line in stream], binding


def training_rows(output, layouts, trace_hashes):
    """Convert only valid TRAIN flights and preserve original trace lineage."""
    positions = {row["id"]: row["bucket_xy_m"] for row in layouts}
    fresh, exclusions, seen = [], Counter(), set()
    with (output / "rows.jsonl").open("rb") as stream:
        for raw in stream:
            row = json.loads(raw)
            if row["phase"] != "train":
                continue
            context = row["context_id"]
            if row["bucket_xy_m"] != positions[context] or row["role"] != "train":
                raise ValueError("Collection shot differs from the frozen training layout")
            reason = _reject(row, (context, positions[context]))
            if reason:
                exclusions[reason] += 1
                continue
            key, sample = _observation(row, output, raw, trace_hashes, (context, positions[context]))
            if key in seen:
                raise ValueError("Duplicate fresh training command")
            seen.add(key)
            fresh.append(sample)
    return fresh, dict(exclusions)


def expanded_library(canonical, fresh):
    library = stable_captures(canonical)
    groups = defaultdict(list)
    for row in fresh:
        if row["route_success"] and not row["label_conflict"]:
            groups[tuple(row["bucket_xy_m"])].append(row)
    for xy, rows in groups.items():
        if xy in library:
            raise ValueError("Intermediate collection overlaps a canonical position")
        library[xy] = sorted(rows, key=lambda row: (-len(row["lineage"]), -row["route_quality"], row["id"]))
    return library


def geometry_prior(library):
    return dict(records=[dict(
        {key: rows[0][key] for key in ("id", "bucket_xy_m", "values", "q10", "q14")},
        source_record_sha256=rows[0]["source_row_sha256"],
    ) for rows in library.values()])


def test_proposals(layout, library, checkpoint, saved_geometry):
    """Use the same ordered recipes as inputs to both test methods."""
    from .intergrid_proposer import propose_candidates

    baseline = [dict(row, method=METHODS[0]) for row in proposals_at(layout, library)]
    recipes = []
    for row in baseline:
        source = next(recipe for recipe in library[tuple(row["training_bucket_xy_m"])]
                      if recipe["id"] == row["training_recipe_id"])
        recipes.append(source)
    seen = set()
    learned = []
    for row in propose_candidates(checkpoint, layout["bucket_xy_m"], recipes, context_id=layout["id"]):
        key = geometry_key(row)
        if key in saved_geometry or key in seen:
            continue
        seen.add(key)
        learned.append(dict(row, phase="test", role="test", context_id=layout["id"], method=METHODS[1]))
    return baseline + learned


def choose(layout, proposed, admitted):
    """Select one feasible command in source order for each method."""
    by_proposal = {row["proposal_id"]: row for row in admitted}
    if len(by_proposal) != len(admitted):
        raise ValueError("Each proposed shot can have only one admitted branch")
    slots = []
    for method in METHODS:
        choice = next((by_proposal[row["id"]] for row in proposed
                       if row["method"] == method and row["id"] in by_proposal), None)
        identifier = f"{layout['id']}-{method}"
        slots.append(dict(id=identifier, method=method, context_id=layout["id"],
                          bucket_xy_m=layout["bucket_xy_m"], stratum=layout["stratum"],
                          request=None if choice is None else dict(choice, id=identifier, stratum=layout["stratum"])))
    return slots


def outcome(row):
    if row is None:
        return "setup_rejected"
    return "invalid_physics" if row["invalid_physics"] else row["experimental_route"]["route_class"]


def summarize(slots, observations):
    if (len({slot["id"] for slot in slots}) != len(slots)
            or set(observations) != {slot["id"] for slot in slots if slot["request"] is not None}):
        raise ValueError("Expected one observation per admitted frozen choice")
    counts = {method: Counter() for method in METHODS}
    strata = defaultdict(lambda: {method: Counter() for method in METHODS})
    paired, both_released = Counter(), Counter()
    for i in range(0, len(slots), 2):
        pair = slots[i:i + 2]
        if (tuple(slot["method"] for slot in pair) != METHODS
                or pair[0]["context_id"] != pair[1]["context_id"]):
            raise ValueError("Expected one pair of attempts per placement")
        results = []
        for slot in pair:
            value = outcome(observations.get(slot["id"]))
            counts[slot["method"]][value] += 1
            strata[slot["stratum"]][slot["method"]][value] += 1
            results.append(value == "standard_capture")
        nearest, learned = results
        label = "both" if nearest and learned else "nearest_only" if nearest else "learned_only" if learned else "neither"
        paired[label] += 1
        if all(slot["id"] in observations for slot in pair):
            both_released[label] += 1
    return dict(methods={method: dict(counts[method]) for method in METHODS},
                strata={key: {method: dict(values[method]) for method in METHODS} for key, values in strata.items()},
                paired=dict(paired), both_released=dict(both_released))


def run(args):
    from .cell1050_heading_native import HeadingEngine
    from .intergrid_proposer import fit_proposer, load_checkpoint, save_checkpoint
    from .intergrid_proposer_plan import collection_proposals, make_splits
    from .table_correction_native import MatchedEngine

    parent, output = args.parent_study, args.output
    spec, parent_binding, _ = parent_inputs(parent, args.parent_audit)
    source = Path(__file__).resolve().parents[2]
    ref = load_runtime_reference(args.reference, args.assets, source)
    if (digest(ref["scene"]) != spec["identity"]["scene_sha256"]
            or ref["asset_hashes"] != spec["identity"]["asset_sha256"]
            or any(importlib.metadata.version(name) != version for name, version in spec["packages"].items())):
        raise ValueError("Scene, assets or pinned dependencies differ from the audited parent")
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
        canonical, binding = load_prior(args.prior_study, args.prior_audit)
        old_test = read(args.prior_study / "test-layouts.json")["rows"]
        splits, provenance = make_splits(args.history_root, read(parent / "grid.json"),
                                        scene=ref["scene"], assets=args.assets, previous_layouts=old_test)
        if len(splits["train"]) != 24 or len(splits["test"]) != 50:
            raise ValueError("Expected 24 collection and 50 final test positions")
        if len({row["id"] for row in splits["train"] + splits["test"]}) != 74:
            raise ValueError("Split contains duplicate placement IDs")
        atomic(output / "splits.json", dict(splits=splits, provenance=provenance))
        base_library = stable_captures(canonical)
        menu = [collection_proposals(layout, base_library) for layout in splits["train"]]
        atomic(output / "collection-menu.json", menu)
        prior = geometry_prior(base_library)
        atomic(output / "training-prior.json", prior)
        contract = dict(version=VERSION, parent=parent_binding, identity=spec["identity"],
                        scene=spec["scene"], prior_comparison=binding, canonical_positions=33,
                        collection_positions=24, test_positions=50, methods=list(METHODS),
                        attempts_per_method_per_position=1, physics_substeps=64, seed=MODEL_SEED,
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
        slots = []
        with open_workers(WORKERS, ref, args.assets, output / "native-heading", kwargs) as workers:
            futures = [workers.submit(prepare_proposals, proposals, True) for proposals in menu]
            run_controls("controls_pre")
            for layout, proposals, future in zip(splits["train"], menu, futures, strict=True):
                admitted, records = future.result()
                atomic(output / "collection" / (layout["id"] + ".json"),
                       dict(proposals=proposals, admitted=admitted, records=records))
                report["collection_prepared"] = report.get("collection_prepared", 0) + 1
                requests.extend(admitted)
                if len(requests) >= NATIVE_BATCH:
                    count = len(execute(heading, requests[:NATIVE_BATCH], "train"))
                    del requests[:NATIVE_BATCH]
                    report["train_releases"] += count
                publish("collecting")
            if requests:
                report["train_releases"] += len(execute(heading, requests, "train"))
                publish("collecting")

            fresh, excluded = training_rows(output, splits["train"], trace_hashes)
            fit_rows = canonical + fresh
            append(output / "training-dataset.jsonl", fit_rows)
            library = expanded_library(canonical, fresh)
            novel_positions = len(library) - len(base_library)
            if novel_positions == 0:
                raise RuntimeError("Collection found no intermediate standard captures to fit")
            if foreign_processes():
                raise RuntimeError("Another process is using the leased 4090")
            checkpoint = fit_proposer(fit_rows, seed=MODEL_SEED, device="cuda:0")
            save_checkpoint(output / "proposer.json", checkpoint)
            checkpoint = load_checkpoint(output / "proposer.json")
            checkpoint_hash = sha(output / "proposer.json")
            report["training"] = dict(canonical_commands=len(canonical), new_valid_commands=len(fresh),
                                      new_standard_captures=sum(row["route_success"] for row in fresh),
                                      new_positions_with_capture=novel_positions, exclusions=excluded,
                                      checkpoint_sha256=checkpoint_hash)
            publish("checkpoint_frozen")

            saved_geometry = {geometry_key(row) for row in fit_rows}
            menus = [test_proposals(layout, library, checkpoint, saved_geometry) for layout in splits["test"]]
            futures = [workers.submit(prepare_proposals, proposals, True) for proposals in menus]
            for layout, proposals, future in zip(splits["test"], menus, futures, strict=True):
                admitted, records = future.result()
                atomic(output / "test" / (layout["id"] + ".json"),
                       dict(proposals=proposals, admitted=admitted, records=records))
                slots.extend(choose(layout, proposals, admitted))
                report["test_prepared"] = len(slots) // len(METHODS)
                if report["test_prepared"] % 5 == 0:
                    publish("preparing_test")
        atomic(output / "choices.json", dict(checkpoint_sha256=checkpoint_hash, slots=slots))
        report["choices_sha256"] = sha(output / "choices.json")
        publish("choices_frozen")
        observations = {}
        requests = [slot["request"] for slot in slots if slot["request"] is not None]
        for offset in range(0, len(requests), NATIVE_BATCH):
            if sha(output / "proposer.json") != checkpoint_hash or sha(output / "choices.json") != report["choices_sha256"]:
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
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Provide a fresh output directory")
    with gpu_slot(args.gpu_lock, timeout_s=1800, notify=lambda row: print(json.dumps(row), flush=True)):
        run(args)


if __name__ == "__main__":
    main()
