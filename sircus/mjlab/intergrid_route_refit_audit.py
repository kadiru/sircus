"""Reproduce the route refit, held-out decisions and capture results on CPU."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

from . import replay650_verify as replay
from .benchmark_verify import Checks, digest, fingerprint, read, sha
from .cell1050_rebound_study import parent_inputs
from .intergrid_layouts import MIN_SEPARATION_M, _placement_geometry
from .intergrid_proposer_audit import audit_packets, audit_preparation
from .intergrid_route_audit import (
    verify_candidate_budget, verify_checkpoint, verify_history_files,
    verify_realized_routes, verify_separation,
)
from .intergrid_route_labels import LABEL_VERSION
from .intergrid_route_models import MODELS, _valid_release
from .intergrid_route_plan import ROUTES
from .intergrid_route_refit_inputs import PLAN_VERSION, load_inputs, sample_layouts
from .intergrid_route_refit_plan import METHODS, test_proposals
from .intergrid_route_refit_study import VERSION as STUDY_VERSION
from .intergrid_route_study import MODEL_SEED, PER_ROUTE_TEST, route_library, source_menu
from .sweep_worker import atomic


VERSION = "sircus-intergrid-route-refit-independent-audit-v1"
SPLIT_SEED = 2026111901
POSITION_COUNT = 50


def verify_layouts(study, grid, scene, assets, history_root, previous, checks):
    artifact = read(study / "layouts.json")
    layouts, provenance = artifact["layouts"], artifact["provenance"]
    checks.equal(len(layouts), POSITION_COUNT, "50 fresh bucket positions")
    checks.equal((provenance["version"], provenance["seed"], provenance["minimum_separation_m"]),
                 (PLAN_VERSION, SPLIT_SEED, MIN_SEPARATION_M), "Prespecified placement protocol")
    geometry = _placement_geometry(scene, assets)
    checks.equal(provenance["geometry"], geometry, "Reconstructed tabletop and fixed-base geometry")
    exclusions = [tuple(xy) for xy in provenance["excluded_coordinates_m"]]
    checks.equal(exclusions, sorted(set(exclusions)), "Sorted unique excluded coordinates")
    checks.equal(provenance["excluded_coordinates_sha256"], digest(exclusions), "Frozen exclusions")
    checks.equal(provenance["layouts_sha256"], digest(layouts), "Frozen held-out layouts")
    previous_xy = {tuple(row["bucket_xy_m"]) for row in previous}
    checks.equal(provenance["declared_previous_coordinates_m"], [list(xy) for xy in sorted(previous_xy)],
                 "Exact original and added training positions and prior test positions")
    historical = verify_history_files(provenance, history_root, study, checks)
    known = historical | previous_xy | {tuple(row["bucket_xy_m"]) for row in grid}
    checks.require(known.issubset(exclusions), "Every known placement excluded")
    checks.equal(layouts, sample_layouts(exclusions, geometry, grid, seed=SPLIT_SEED),
                 "Reproduce all held-out positions from the frozen seed")
    verify_separation(layouts, [], exclusions, checks)
    return layouts


def verify_choices(study, layouts, library, checkpoints, hashes, prior, home, checks):
    folder = study / "test"
    checks.equal({path.stem for path in folder.glob("*.json")}, {row["id"] for row in layouts},
                 "One complete candidate artifact per held-out position")
    resolutions, slots = {}, []
    totals = {method: {route: Counter() for route in ROUTES} for method in METHODS}
    for layout in layouts:
        identity = layout["id"]
        artifact = read(folder / (identity + ".json"))
        checks.equal(artifact["proposals"], test_proposals(layout, library, checkpoints["original"],
                                                          checkpoints["refit"]),
                     identity + "/reproduced frozen predictions from the common original source menu")
        proposals, admitted = audit_preparation(artifact, layout, "test", study / "native-heading",
                                                prior, home, resolutions, checks)
        sources = source_menu(layout, library)
        for generation, checkpoint in checkpoints.items():
            suffix = "_" + generation
            groups = []
            for candidates in (proposals, admitted):
                group = []
                for row in candidates:
                    if row["method"].endswith(suffix):
                        checks.equal(row["route_models_fingerprint"], checkpoint["fingerprint"],
                                     row["id"] + "/exact model generation")
                        group.append(dict(row, method=row["method"].removesuffix(suffix)))
                groups.append(group)
            counts = verify_candidate_budget(identity, sources, *groups, checks)
            for model in MODELS:
                for route in ROUTES:
                    totals[model + suffix][route].update(counts[model][route])

        for method in METHODS:
            for route in ROUTES:
                name = f"{identity}-{method}-{route}"
                chosen = next((row for row in admitted
                               if row["method"] == method and row["intended_route"] == route), None)
                request = None if chosen is None else dict(chosen, id=name, stratum=layout["stratum"])
                slots.append(dict(id=name, method=method, context_id=identity,
                                  bucket_xy_m=layout["bucket_xy_m"], stratum=layout["stratum"],
                                  intended_route=route, request=request))
    checks.equal(read(study / "choices.json"), dict(checkpoint_sha256=hashes, slots=slots),
                 "First admitted proposal per model and requested route was frozen before release")
    return slots, {method: {route: dict(counter) for route, counter in by_route.items()}
                   for method, by_route in totals.items()}


def account_shots(layouts, slots, observations, checks):
    counts = {method: {route: Counter() for route in ROUTES} for method in METHODS}
    outcomes_by_layout = {layout["id"]: {} for layout in layouts}
    checks.equal(len({slot["id"] for slot in slots}), len(slots), "Unique frozen slot IDs")
    checks.equal(set(observations), {slot["id"] for slot in slots if slot["request"] is not None},
                 "Exactly one release for each admitted slot")
    for slot in slots:
        name = slot["id"]
        method, route = slot["method"], slot["intended_route"]
        outcome = "setup_rejected"
        if slot["request"] is not None:
            row, measured = observations[name]
            checks.equal(row["intended_route"], route, name + "/release retains the requested route")
            if measured == "standard_capture":
                outcome = row["shot_route"] or "unclassified_capture"
            elif measured == "robot_assisted_capture":
                outcome = "arm_assisted_capture"
            elif measured == "invalid_physics":
                outcome = "invalid_physics"
            else:
                outcome = "flight_miss"
        counts[method][route][outcome] += 1
        at_layout = outcomes_by_layout[slot["context_id"]]
        checks.require((method, route) not in at_layout, name + "/one attempt per model and route")
        at_layout[method, route] = outcome == route

    expected = {(method, route) for method in METHODS for route in ROUTES}
    paired = {route: {model: dict(both=0, original_only=0, refit_only=0, neither=0) for model in MODELS}
              for route in ROUTES}
    paired_labels = {(True, True): "both", (True, False): "original_only",
                     (False, True): "refit_only", (False, False): "neither"}
    for identity, results in outcomes_by_layout.items():
        checks.equal(set(results), expected, identity + "/all twelve paired slots accounted for")
        for route in ROUTES:
            for model in MODELS:
                key = (results[model + "_original", route], results[model + "_refit", route])
                paired[route][model][paired_labels[key]] += 1

    methods = {}
    for method, by_route in counts.items():
        methods[method] = {}
        for route, outcomes in by_route.items():
            attempts, captures = sum(outcomes.values()), outcomes[route]
            checks.equal(attempts, POSITION_COUNT, f"{method}/{route}/all positions in denominator")
            methods[method][route] = dict(attempts=attempts, releases=attempts - outcomes["setup_rejected"],
                                          captures=captures, capture_rate=captures / attempts,
                                          outcomes=dict(outcomes))
    return dict(methods=methods, paired=paired)


def audit(study, parent, parent_audit, model_study, model_audit, paired_study, paired_audit,
          matched_study, matched_audit, history_root, assets):
    checks = Checks()
    spec, report = read(study / "contract.json"), read(study / "report.json")
    parent_spec, parent_binding, _ = parent_inputs(parent, parent_audit)
    checks.equal((spec["version"], report["status"], report["phase"]),
                 (STUDY_VERSION, "completed", "completed"), "Completed paired route refit comparison")
    checks.equal(report["test_prepared"], POSITION_COUNT, "Every held-out position prepared")
    fingerprint(spec, checks, "Route refit contract")
    checks.equal((spec["parent"], spec["identity"], spec["scene"]),
                 (parent_binding, parent_spec["identity"], parent_spec["scene"]), "Exact audited native64 parent")
    checks.equal((spec["test_positions"], spec["methods"], spec["routes"],
                  spec["attempts_per_method_per_route_per_position"], spec["candidate_budget_per_route"],
                  spec["physics_substeps"], spec["label_version"], spec["seed"], spec["split_seed"]),
                 (POSITION_COUNT, list(METHODS), list(ROUTES), 1, PER_ROUTE_TEST, 64, LABEL_VERSION,
                  MODEL_SEED, SPLIT_SEED),
                 "Matched attempt budgets, fresh split seed and unchanged route scoring")

    manifest = replay.check_manifest(study, checks)
    checks.equal(report["evidence_manifest_sha256"], sha(study / "evidence-manifest.json"),
                 "Report binds all saved evidence")
    checks.equal(report["choices_sha256"], manifest["choices.json"], "Report binds frozen choices")
    checks.equal(spec["frozen_inputs"], {name: manifest[name] for name in
                 ("layouts.json", "training-prior.json", "training-dataset.jsonl")}, "Exact frozen input bytes")
    original, original_rows, added_rows, previous, binding = load_inputs(
        history_root, model_study, model_audit, paired_study, paired_audit, matched_study, matched_audit)
    for key, expected in binding.items():
        checks.equal(spec[key], expected, "Exact audited input binding/" + key)
    checks.equal(manifest["training-prior.json"], binding["model_study"]["geometry_prior"],
                 "Original audited arm geometry prior")
    hashes = dict(original=binding["model_study"]["checkpoint"], refit=manifest["route-models.json"])
    checks.equal(manifest["route-models-original.json"], hashes["original"], "Original checkpoint bytes unchanged")
    with (study / "training-dataset.jsonl").open() as stream:
        training = [json.loads(line) for line in stream]
    checks.equal(training, original_rows + added_rows, "Exact original training rows plus audited paired releases")
    checks.equal(len({row["id"] for row in training}), len(training), "Unique training observations")
    refit = verify_checkpoint(study / "route-models.json", training, ROUTES, MODEL_SEED, checks)
    checks.require(refit["dataset_sha256"] != original["dataset_sha256"], "Refit used additional measured captures")
    positive_positions = {route: set() for route in ROUTES}
    for row in training:
        route = row["shot_route"]
        if route in positive_positions and _valid_release(row):
            positive_positions[route].add(tuple(row["bucket_xy_m"]))
    training_summary = dict(
        original_rows=len(original_rows), new_rows=len(added_rows), total_rows=len(training),
        positive_rows=refit["training_captures"],
        positive_by_route={route: refit["route_support"][route]["captures"] for route in ROUTES},
        positive_positions_by_route={route: len(positions) for route, positions in positive_positions.items()},
        checkpoint_sha256=hashes,
    )
    checks.equal(report["training"], training_summary, "Report matches independently reconstructed training counts")

    grid_path = parent / "grid.json"
    checks.equal(sha(grid_path), read(parent / "evidence-manifest.json")["grid.json"], "Exact original bucket grid")
    layouts = verify_layouts(study, read(grid_path), spec["scene"], assets, history_root, previous, checks)
    prior, home = read(study / "training-prior.json"), read(parent / "setup-home.json")
    slots, candidate_counts = verify_choices(study, layouts, route_library(original_rows),
                                              dict(original=original, refit=refit), hashes, prior, home, checks)
    requests = [slot["request"] for slot in slots if slot["request"] is not None]
    measured, controls = audit_packets(study, parent, assets, spec, report, manifest,
                                        {"test": requests}, prior, home, checks)
    verify_realized_routes(study, measured, checks)
    comparison = account_shots(layouts, slots, measured["test"], checks)
    checks.equal(report["comparison"], comparison, "Exact paired comparison from rescored native evidence")
    return dict(version=VERSION, status="passed", checks_count=checks.count, test_positions=len(layouts),
                test_attempts=len(slots), test_releases=len(requests), control_releases=controls,
                comparison=comparison, candidate_counts=candidate_counts, training=training_summary,
                report_sha256=sha(study / "report.json"), evidence_manifest_sha256=report["evidence_manifest_sha256"],
                checkpoint_sha256=hashes, choices_sha256=manifest["choices.json"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("study", "parent-study", "parent-audit", "model-study", "model-audit", "paired-study",
                 "paired-audit", "matched-study", "matched-audit", "history-root", "assets", "audit"):
        parser.add_argument("--" + name, required=True, type=Path)
    args = parser.parse_args()
    if args.audit.exists() or args.audit.resolve().is_relative_to(args.study.resolve()):
        parser.error("Write a fresh audit outside the frozen evidence directory")
    try:
        result = audit(args.study, args.parent_study, args.parent_audit, args.model_study, args.model_audit,
                       args.paired_study, args.paired_audit, args.matched_study, args.matched_audit,
                       args.history_root, args.assets)
    except Exception as error:
        result = dict(version=VERSION, status="failed", error=f"{type(error).__name__}: {error}")
    atomic(args.audit, result)
    print(json.dumps(result), flush=True)
    if result["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
