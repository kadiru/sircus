"""Fit an equal-position capture model and compare one shot at new bucket positions."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import importlib.metadata
import json
import math
from pathlib import Path
import shutil

from .benchmark_verify import digest, read, sha
from .cell1050_followup_plan import geometry_key
from .cell1050_rebound_study import annotate, control_requests, parent_inputs
from .corrected_grid import source_manifest
from .coverage_training_study import append, share_native_environment
from .gpu_slot import foreign_processes, gpu_slot
from .parallel_capture_grid_study import NATIVE_BATCH, STORAGE_RESERVE_BYTES, control_mirror
from .parallel_capture_grid_workers import load_resolution, open_workers, prepare_proposals
from .policy64_contract import load_runtime_reference
from .sweep_worker import atomic
from .table_training_study import control_gate

METHODS = ("nearest", "learned")
VERSION = "sircus-intergrid-comparison-v1"
MODEL_SEED = 50283
STEPS = 120
WORKERS = 6


def stable_captures(rows):
    """Rank stored working shots with repeat and entry quality as tie breakers."""
    cells = defaultdict(list)
    for row in rows:
        if row["route_success"] and not row.get("label_conflict"):
            cells[tuple(row["bucket_xy_m"])].append(row)
    if len(cells) != 33:
        raise ValueError("Known-shot library requires a standard capture at all 33 positions")

    def quality(row):
        return (-len(row["lineage"]), -row["route_quality"], row["id"])

    return {xy: sorted(entries, key=quality) for xy, entries in cells.items()}


def proposals_at(layout, library):
    """Share eight candidates from the four nearest training positions."""
    target = layout["bucket_xy_m"]
    selected = []
    seen = set()
    for xy in sorted(library, key=lambda xy: (math.dist(xy, target), xy))[:4]:
        count = 0
        for recipe in library[xy]:
            key = geometry_key(recipe)
            if key in seen:
                continue
            seen.add(key)
            proposal = dict(
                id=f"{layout['id']}-candidate-{len(selected):02d}",
                context_id=layout["id"], phase="test", role="test",
                bucket_xy_m=target, values=recipe["values"],
                board_heading_offset_deg=recipe["board_heading_offset_deg"],
                board_twist_deg=recipe["board_twist_deg"],
                seed_q10=recipe["q10"], seed_q14=recipe["q14"],
                training_recipe_id=recipe["id"], training_bucket_xy_m=list(xy),
                training_distance_m=math.dist(xy, target),
            )
            selected.append(proposal)
            count += 1
            if count == 2:
                break
    return selected


def select(layout, admitted, proposals, scores):
    """Freeze both choices before either method gets a ball flight."""
    if len(admitted) != len(scores):
        raise ValueError("Candidate scores must match the admitted menu")
    admitted_by_proposal = {row["proposal_id"]: row for row in admitted}
    if len(admitted_by_proposal) != len(admitted):
        raise ValueError("Expected at most one admitted branch per proposal")
    nearest = admitted_by_proposal.get(proposals[0]["id"])
    best = max(range(len(admitted)), key=lambda index: scores[index], default=None)
    learned = None if best is None else admitted[best]
    return [
        dict(id=f"{layout['id']}-{method}", method=method, context_id=layout["id"],
             bucket_xy_m=layout["bucket_xy_m"], stratum=layout["stratum"],
             request=None if choice is None else dict(choice, id=f"{layout['id']}-{method}",
                                                       method=method, stratum=layout["stratum"]),
             score=None if choice is None else float(scores[admitted.index(choice)]),
             first_known_admitted=proposals[0]["id"] in admitted_by_proposal)
        for method, choice in zip(METHODS, (nearest, learned), strict=True)
    ]


def compare(slots, observations):
    if len({slot["id"] for slot in slots}) != len(slots):
        raise ValueError("Duplicate frozen test attempt")
    expected = {slot["id"] for slot in slots if slot["request"] is not None}
    if set(observations) != expected:
        raise ValueError("Observed test releases differ from frozen choices")
    methods = {method: Counter() for method in METHODS}
    strata = defaultdict(lambda: {method: Counter() for method in METHODS})
    paired = Counter()
    both_released = Counter()
    changed_settings = 0
    for i in range(0, len(slots), 2):
        pair = slots[i:i + 2]
        if (len(pair) != 2 or tuple(slot["method"] for slot in pair) != METHODS
                or pair[0]["context_id"] != pair[1]["context_id"]):
            raise ValueError("Expected one paired result per bucket position")
        capture = []
        for slot in pair:
            row = observations.get(slot["id"])
            category = ("setup_rejected" if row is None else "invalid_physics" if row["invalid_physics"]
                        else row["experimental_route"]["route_class"])
            methods[slot["method"]][category] += 1
            strata[slot["stratum"]][slot["method"]][category] += 1
            capture.append(category == "standard_capture")
        result = (("both" if capture[0] else "learned_only") if capture[1]
                  else ("nearest_only" if capture[0] else "neither"))
        paired[result] += 1
        if all(slot["id"] in observations for slot in pair):
            both_released[result] += 1
            changed_settings += pair[0]["request"]["proposal_id"] != pair[1]["request"]["proposal_id"]
    return dict(methods={key: dict(value) for key, value in methods.items()},
                strata={key: {method: dict(counts) for method, counts in value.items()}
                        for key, value in strata.items()}, paired=dict(paired),
                both_released=dict(both_released), different_settings_when_both_released=changed_settings)


def run(args):
    from .cell1050_heading_native import HeadingEngine
    from .intergrid_dataset import load_dataset
    from .intergrid_layouts import make_layouts
    from .intergrid_ranker import fit_ranker, load_checkpoint, predict_scores, save_checkpoint
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
    report = dict(status="running", phase="inputs", controls={}, measured_test_releases=0)

    def publish(phase):
        report["phase"] = phase
        atomic(output / "report.json", report)
        print(json.dumps(report), flush=True)

    packet_number = 0

    def execute(engine, directory, requests, phase):
        nonlocal packet_number
        if foreign_processes():
            raise RuntimeError("Another process is using the leased 4090")
        if shutil.disk_usage(output).free < STORAGE_RESERVE_BYTES:
            raise RuntimeError("Insufficient space for native trace evidence")
        packet = f"requests/batch-{packet_number:05d}.json"
        packet_number += 1
        atomic(output / packet, dict(phase=phase, rows=requests))
        measured, timing = engine.evaluate(requests, phase)
        if ([row["id"] for row in measured] != [row["id"] for row in requests]
                or any(row["native_release"] is not True for row in measured)):
            raise RuntimeError("An exact admitted test request did not release")
        adapter = engine.adapter if phase == "test" else engine.engine.adapter
        if (adapter.env.effective_options != expected_options
                or adapter.env.compiled_geometry != expected_geometry):
            raise RuntimeError("Native physics differs from the pinned source study")
        if phase == "test":
            annotate(measured, output / directory)
        packet_hash = sha(output / packet)
        for row in measured:
            row.update(request_packet=packet, request_packet_sha256=packet_hash, adapter_directory=directory)
        append(output / "rows.jsonl", measured)
        append(output / "timing.jsonl", [dict(packet=packet, phase=phase, timing=timing)])
        return measured

    def run_controls(phase):
        rows = execute(controls, "native-controls", control_requests(control_sources, phase), phase)
        gate = control_gate(rows)
        report["controls"][phase] = gate
        if gate["status"] != "passed":
            raise RuntimeError(f"{phase} failed")
        publish(phase)

    try:
        publish("inputs")
        rows, provenance = load_dataset(args.history_root, args.grid_study, args.grid_audit)
        layouts, placement_provenance = make_layouts(args.history_root, read(parent / "grid.json"),
                                                     scene=ref["scene"], assets=args.assets)
        if len(layouts) != 50 or len({row["id"] for row in layouts}) != len(layouts):
            raise ValueError("Expected exactly 50 distinct test positions")
        atomic(output / "test-layouts.json", dict(rows=layouts, provenance=placement_provenance))
        atomic(output / "dataset-provenance.json", provenance)
        append(output / "dataset.jsonl", rows)
        library = stable_captures(rows)
        candidate_menu = [proposals_at(layout, library) for layout in layouts]
        atomic(output / "candidate-menu.json", candidate_menu)
        contract = dict(version=VERSION, parent=parent_binding, identity=spec["identity"],
                        scene=spec["scene"], train_positions=33, test_positions=len(layouts),
                        methods=list(METHODS), attempts_per_method_per_position=1,
                        candidate_rule="Two known geometries per each of four nearest training positions",
                        nearest_rule="First known setting at the nearest training bucket, ranked by repeat evidence and route quality; setup rejection counts as failure",
                        learned_rule="Highest frozen score over the same admitted candidates",
                        seed=MODEL_SEED, training_steps=STEPS, physics_substeps=64,
                        source_sha256=source_manifest(source), grid_audit_sha256=sha(args.grid_audit),
                        frozen_inputs={name: sha(output / name) for name in (
                            "dataset.jsonl", "dataset-provenance.json", "test-layouts.json", "candidate-menu.json",
                        )})
        contract["fingerprint"] = digest(contract)
        atomic(output / "contract.json", contract)
        publish("training")
        checkpoint = fit_ranker(rows, seed=MODEL_SEED, device="cuda:0", epochs=STEPS)
        save_checkpoint(output / "ranker.json", checkpoint)
        checkpoint = load_checkpoint(output / "ranker.json")
        checkpoint_hash = sha(output / "ranker.json")
        report["training"] = {key: checkpoint[key] for key in (
            "training_rows", "excluded_conflicts", "steps", "device", "curve",
        )}
        publish("checkpoint_frozen")

        setup = read(parent / "setup-home.json")
        calibration = read(parent / "calibration.json")
        for name in ("calibration.json", "verification.json"):
            shutil.copyfile(parent / name, output / name)
        prior = dict(records=[dict(
            {key: recipes[0][key] for key in ("id", "bucket_xy_m", "values", "q10", "q14")},
            source_record_sha256=recipes[0]["source_row_sha256"],
        ) for recipes in library.values()])
        atomic(output / "training-prior.json", prior)
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
        slots = []
        with open_workers(WORKERS, ref, args.assets, output / "native-heading", kwargs) as workers:
            futures = [workers.submit(prepare_proposals, menu, True) for menu in candidate_menu]
            run_controls("controls_pre")
            for layout, menu, future in zip(layouts, candidate_menu, futures, strict=True):
                admitted, records = future.result()
                scores = predict_scores(checkpoint, admitted)
                atomic(output / "preparations" / (layout["id"] + ".json"),
                       dict(proposals=menu, admitted=admitted, records=records, scores=scores.tolist()))
                slots.extend(select(layout, admitted, menu, scores))
                report["prepared_positions"] = len(slots) // 2
                if report["prepared_positions"] % 5 == 0:
                    publish("preparing")
        atomic(output / "choices.json", dict(checkpoint_sha256=checkpoint_hash, slots=slots))
        report["choices_sha256"] = sha(output / "choices.json")
        publish("choices_frozen")
        observations = {}
        requests = [slot["request"] for slot in slots if slot["request"] is not None]
        for i in range(0, len(requests), NATIVE_BATCH):
            if sha(output / "ranker.json") != checkpoint_hash or sha(output / "choices.json") != report["choices_sha256"]:
                raise ValueError("Checkpoint or choices changed during test")
            batch = requests[i:i + NATIVE_BATCH]
            for xy in {tuple(request["bucket_xy_m"]) for request in batch}:
                load_resolution(heading, list(xy))
            observations.update((row["id"], row) for row in execute(heading, "native-heading", batch, "test"))
            report["measured_test_releases"] = len(observations)
            publish("test")
        report["comparison"] = compare(slots, observations)
        report["nearest_first_choice_admitted"] = sum(
            slot["first_known_admitted"] for slot in slots if slot["method"] == "nearest"
        )
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
    for name in ("history-root", "grid-study", "grid-audit", "parent-study", "parent-audit",
                 "reference", "assets", "gpu-lock", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Provide a fresh output directory")
    with gpu_slot(args.gpu_lock, timeout_s=1800, notify=lambda row: print(json.dumps(row), flush=True)):
        run(args)


if __name__ == "__main__":
    main()
