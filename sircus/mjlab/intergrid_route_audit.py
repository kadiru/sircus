"""Audit frozen route selection and every saved native64 comparison shot on CPU."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import copy
from itertools import combinations
import json
import math
from pathlib import Path

import numpy as np

from . import replay650_verify as replay
from .benchmark_verify import Checks, contained, digest, fingerprint, read, sha
from .cell1050_rebound_study import parent_inputs
from .intergrid_layouts import MIN_SEPARATION_M, _placement_geometry
from .intergrid_proposer_audit import audit_packets, audit_preparation
from .intergrid_route_labels import LABEL_VERSION, annotate_routes
from .intergrid_route_models import MODELS, fit_route_models, load_checkpoint
from .intergrid_route_plan import PER_ROUTE_LIMIT, ROUTES, SEED, collection_proposals, sample_splits
from .intergrid_route_study import (
    CANDIDATE_BUDGET, MODEL_SEED, PER_ROUTE_TEST, VERSION as STUDY_VERSION,
    fit_gate, fresh_rows, load_prior, route_library, source_menu, summarize_collection, test_proposals, training_summary,
)
from .parallel_capture_grid_audit import observed_rows
from .sweep_worker import atomic


VERSION = "sircus-intergrid-route-independent-audit-v1"


def verify_separation(layouts, train_xy, previous_xy, checks):
    seen_ids = set()
    occupied = set(map(tuple, train_xy)) | set(map(tuple, previous_xy))
    for layout in layouts:
        identity = layout["id"]
        xy = tuple(layout["bucket_xy_m"])
        checks.require(len(xy) == 2 and all(math.isfinite(value) for value in xy), identity + "/finite XY")
        checks.equal((layout["context_id"], layout["split"]), (identity, "test"), identity + "/unique frozen TEST identity")
        checks.require(identity not in seen_ids, identity + "/duplicate TEST identity")
        checks.require(all(math.dist(xy, other) >= MIN_SEPARATION_M for other in occupied),
                       identity + "/at least 5 mm from TRAIN, prior TEST and all earlier TEST placements")
        seen_ids.add(identity)
        occupied.add(xy)


def verify_splits(study, grid, scene, assets, historical, previous, checks):
    artifact = read(study / "splits.json")
    split, provenance = artifact["splits"], artifact["provenance"]
    checks.equal(set(split), {"train", "test"}, "Exact frozen TRAIN and TEST splits")
    checks.equal((len(split["train"]), len(split["test"])), (24, 50), "24 collection and 50 holdout locations")
    checks.equal(provenance["seed"], SEED, "Exact prespecified placement seed")
    checks.equal(provenance["minimum_separation_m"], MIN_SEPARATION_M, "Prespecified novelty threshold")
    geometry = _placement_geometry(scene, assets)
    checks.equal(provenance["geometry"], geometry, "Reconstructed tabletop and fixed-base geometry")
    exclusions = [tuple(xy) for xy in provenance["excluded_coordinates_m"]]
    checks.equal(exclusions, sorted(set(exclusions)), "Sorted unique historical exclusion coordinates")
    checks.equal(digest(exclusions), provenance["excluded_coordinates_sha256"], "Exact frozen exclusion coordinates")
    checks.equal(digest(split), provenance["splits_sha256"], "Exact frozen TRAIN and TEST layouts")
    prior_xy = sorted({tuple(row["bucket_xy_m"]) for row in previous})
    checks.equal(provenance["declared_previous_coordinates_m"], [list(xy) for xy in prior_xy],
                 "Authenticated prior placement exclusions")
    known = set(map(tuple, historical)) | set(prior_xy) | {tuple(row["bucket_xy_m"]) for row in grid}
    checks.require(known.issubset(exclusions), "Prior training, prior tests and original grid excluded")
    checks.equal(split, sample_splits(exclusions, geometry, grid, seed=SEED),
                 "Reproduce 74 placements from seed, geometry and frozen exclusions")
    for index, layout in enumerate(split["train"]):
        identity = f"intergrid-route-train-{index:02d}"
        checks.equal((layout["id"], layout["context_id"], layout["split"]), (identity, identity, "train"),
                     identity + "/unique training identity")
    verify_separation(split["test"], [row["bucket_xy_m"] for row in split["train"]], exclusions, checks)
    return split


def account_shots(slots, observations, checks):
    counts = {method: Counter() for method in MODELS}
    intent = {method: Counter() for method in MODELS}
    by_intent = {method: {route: Counter() for route in ROUTES} for method in MODELS}
    strata = defaultdict(lambda: {method: Counter() for method in MODELS})
    by_layout = defaultdict(dict)
    for slot in slots:
        method, layout = slot["method"], slot["context_id"]
        identity = slot["id"]
        checks.require(method not in by_layout[layout], identity + "/one attempt for method and layout")
        measurement = observations.get(identity)
        checks.equal(measurement is None, slot["request"] is None, identity + "/exactly one admitted release or a setup decline")
        if measurement is None:
            result = "setup_rejected"
            checks.require(slot["intended_route"] is None, identity + "/setup reject does not have an executed route")
        else:
            row, native_outcome = measurement
            route = slot["intended_route"]
            checks.equal((row["intended_route"], slot["request"]["intended_route"]), (route, route),
                         identity + "/measured request matches frozen intended route")
            if native_outcome == "standard_capture":
                result = row["shot_route"] or "unclassified_capture"
            else:
                result = "invalid_physics" if native_outcome == "invalid_physics" else "flight_miss"
            by_intent[method][route][result] += 1
            intent[method][route + "_releases"] += 1
            intent[method]["intended_captures"] += int(result == route)
            intent[method]["other_route_captures"] += int(result in ROUTES and result != route)
        counts[method][result] += 1
        strata[slot["stratum"]][method][result] += 1
        by_layout[layout][method] = result
    paired = {f"{a}_vs_{b}": Counter() for a, b in combinations(MODELS, 2)}
    captures = {*ROUTES, "unclassified_capture"}
    for layout, outcomes in by_layout.items():
        checks.equal(set(outcomes), set(MODELS), layout + "/all comparison methods accounted for")
        for a, b in combinations(MODELS, 2):
            a_captured, b_captured = outcomes[a] in captures, outcomes[b] in captures
            result = "both" if a_captured and b_captured else a + "_only" if a_captured else b + "_only" if b_captured else "neither"
            paired[f"{a}_vs_{b}"][result] += 1
    return dict(methods={method: dict(value) for method, value in counts.items()},
                intent={method: dict(value) for method, value in intent.items()},
                by_intent={method: {route: dict(results) for route, results in routes.items()}
                           for method, routes in by_intent.items()},
                strata={key: {method: dict(value) for method, value in group.items()} for key, group in strata.items()},
                paired={key: dict(value) for key, value in paired.items()})


def verify_checkpoint(checkpoint_path, training, routes, seed, checks):
    checkpoint = load_checkpoint(checkpoint_path)
    checks.equal((checkpoint["seed"], checkpoint["device"], set(checkpoint["routes"])),
                 (seed, "cuda:0", set(routes)), "Exact frozen seed, device and route set")
    reconstructed = fit_route_models(training, seed=seed, device="cpu", routes=routes)

    def compare(actual, expected, label):
        if isinstance(expected, dict):
            checks.equal(set(actual), set(expected), label + "/exact model fields")
            for key, value in expected.items():
                compare(actual[key], value, label + "/" + key)
        else:
            checks.numeric(np.asarray(actual), np.asarray(expected), label, 1e-9)

    for key, value in reconstructed.items():
        label = "CPU-reconstructed common training pairs and models/" + key
        if key in {"coefficients", "training_rmse"}:
            compare(checkpoint[key], value, label)
        elif key not in {"device", "fingerprint"}:
            checks.equal(checkpoint[key], value, label)
    return checkpoint


def verify_previous_layouts(prior_study, checks):
    path = prior_study / "splits.json"
    manifest = read(prior_study / "evidence-manifest.json")
    checks.equal(sha(path), manifest["splits.json"], "Previous TRAIN and TEST placements match audited bytes")
    split = read(path)["splits"]
    return split["train"] + split["test"]


def verify_history_files(provenance, history_root, study, checks):
    """Re-read the exact historical inventory and its previously recorded coordinates."""
    from .table_training_contract import _coordinates

    repository = Path(__file__).resolve().parents[2]
    sources = {"output": Path(history_root), "baselines": repository / "baselines"}
    files = provenance["historical_files_sha256"]
    checks.require(bool(files), "Nonempty historical placement inventory")
    points = set()
    for name, expected in files.items():
        namespace, relative = name.split("/", 1)
        source = contained(sources[namespace], relative)
        checks.require(not source.is_relative_to(study.resolve()), name + "/current study excluded from historical inventory")
        checks.equal(sha(source), expected, name + "/authenticated historical placement bytes")
        if source.suffix == ".jsonl":
            with source.open() as stream:
                for line in stream:
                    if line.strip():
                        _coordinates(json.loads(line), points)
            continue
        payload = read(source)
        _coordinates(payload, points, coordinate_list="exclusion" in source.name)
        if source.name in ("splits.json", "contract.json") and isinstance(payload, dict):
            for part in (payload, payload.get("splits"), payload.get("future_splits")):
                if isinstance(part, dict):
                    for role in ("train", "validation", "test"):
                        _coordinates(part.get(role), points, coordinate_list=True)
    checks.equal(len(points), provenance["historical_coordinate_count"], "Exact historical coordinate inventory count")
    checks.require(points.issubset(set(map(tuple, provenance["excluded_coordinates_m"]))),
                   "Every historical coordinate is excluded from this comparison")
    return points


def verify_collection(study, layouts, library, prior, home, resolutions, checks):
    menus = read(study / "collection-menu.json")
    checks.equal(menus, [collection_proposals(layout, library) for layout in layouts],
                 "Collection menu comes only from authenticated earlier TRAIN captures")
    folder = study / "collection"
    checks.equal({path.stem for path in folder.glob("*.json")}, {row["id"] for row in layouts},
                 "One preparation record for each frozen collection layout")
    requests, packets = [], {}
    for layout, menu in zip(layouts, menus, strict=True):
        identity = layout["id"]
        artifact = read(folder / (identity + ".json"))
        checks.equal(artifact["proposals"], menu, identity + "/exact frozen collection menu")
        checks.equal(dict(Counter(row["intended_route"] for row in menu)),
                     {route: PER_ROUTE_LIMIT for route in ROUTES}, identity + "/equal direct and table collection quotas")
        _, admitted = audit_preparation(artifact, layout, "train", study / "native-heading", prior, home, resolutions, checks)
        requests.extend(admitted)
        packets[identity] = artifact
    return requests, packets


def verify_training_rows(training, prior, fresh, splits, checks):
    checks.equal(training, prior + fresh, "Exact earlier TRAIN rows plus valid new native collection releases")
    checks.equal(len({row["id"] for row in training}), len(training), "Unique training sample identities")
    positions = {tuple(row["bucket_xy_m"]): row["id"] for row in splits["train"]}
    for row in fresh:
        checks.equal((row["phase"], row["context_id"]), ("train", positions[tuple(row["bucket_xy_m"])]),
                     row["id"] + "/exact frozen collection placement")
    for row in training:
        identity = row["id"]
        checks.require(row["phase"] not in {"test", "validation"}, identity + "/earlier or current TRAIN release")
        checks.require(row["native_release"] is True and not row["invalid"] and row["invalid_physics"] is False,
                       identity + "/valid measured release")


def verify_realized_routes(study, measurements, checks):
    originals = [row for phase in measurements.values() for row, _ in phase.values()]
    reproduced = annotate_routes(copy.deepcopy(originals), study / "native-heading")
    for original, actual in zip(originals, reproduced, strict=True):
        for key in ("shot_route", "table_rebounds_before_entry"):
            checks.equal(original[key], actual[key], original["id"] + "/native observed " + key)


def verify_candidate_budget(identity, source_menu, proposals, admitted, checks):
    checks.equal(dict(Counter(row["shot_route"] for row in source_menu)), {route: PER_ROUTE_TEST for route in ROUTES},
                 identity + "/four source slots per route for every model")
    checks.equal(len({row["id"] for row in source_menu}), len(source_menu), identity + "/unique shared source IDs")
    counts = {method: {route: dict(proposed=0, admitted=0) for route in ROUTES} for method in MODELS}
    for label, candidates in (("proposed", proposals), ("admitted", admitted)):
        seen = set()
        for row in candidates:
            method, index, route = (row[key] for key in ("method", "source_index", "intended_route"))
            name = row["id"]
            checks.require(type(index) is int and 0 <= index < len(source_menu), name + "/within eight common slots")
            checks.require((method, index) not in seen, name + "/at most one candidate per model and source slot")
            seen.add((method, index))
            source = source_menu[index]
            checks.equal((row["training_recipe_id"], row["training_bucket_xy_m"], route),
                         (source["id"], source["bucket_xy_m"], source["shot_route"]), name + "/exact shared source and route")
            counts[method][route][label] += 1
    return counts


def verify_test_choices(study, layouts, library, checkpoint, gate, prior, home, resolutions, checks):
    folder = study / "test"
    checks.equal({path.stem for path in folder.glob("*.json")}, {row["id"] for row in layouts},
                 "One complete candidate record per TEST layout")
    selections = []
    totals = {method: {route: Counter() for route in ROUTES} for method in MODELS}
    for layout in layouts:
        identity = layout["id"]
        artifact = read(folder / (identity + ".json"))
        checks.equal(artifact["proposals"], test_proposals(layout, library, checkpoint),
                     identity + "/exact frozen model predictions on the common source menu")
        proposals, admitted = audit_preparation(artifact, layout, "test", study / "native-heading", prior, home, resolutions, checks)
        counts = verify_candidate_budget(identity, source_menu(layout, library), proposals, admitted, checks)
        for method in MODELS:
            for route in ROUTES:
                totals[method][route].update(counts[method][route])
        neighbors = sorted(gate["sites"], key=lambda site: (math.dist(site["bucket_xy_m"], layout["bucket_xy_m"]), site["id"]))[:5]
        scores = {route: (1 + sum(site["counts"][route]["intended_captures"] for site in neighbors)) /
                  (2 + sum(site["counts"][route]["proposals"] for site in neighbors)) for route in ROUTES}
        order = sorted(ROUTES, key=lambda route: -scores[route])
        for method in MODELS:
            identifier = f"{identity}-{method}"
            request = next((row for route in order for row in admitted
                            if row["method"] == method and row["intended_route"] == route), None)
            selections.append(dict(id=identifier, method=method, context_id=identity, bucket_xy_m=layout["bucket_xy_m"],
                                   stratum=layout["stratum"], route_priority=order, gate_scores=scores,
                                   intended_route=None if request is None else request["intended_route"],
                                   request=None if request is None else dict(request, id=identifier, stratum=layout["stratum"])))
    choices = read(study / "choices.json")
    checks.equal((choices["checkpoint_sha256"], choices["gate_sha256"]),
                 (sha(study / "route-models.json"), sha(study / "route-gate.json")), "Choices bind both frozen models and route gate")
    checks.equal(choices["slots"], selections, "Reconstructed all frozen one-shot decisions")
    return selections, {method: {route: dict(counts) for route, counts in by_route.items()}
                        for method, by_route in totals.items()}


def audit(study, parent, parent_audit, prior_study, prior_audit, history_root, assets):
    checks = Checks()
    spec, report = read(study / "contract.json"), read(study / "report.json")
    parent_spec, parent_binding, _ = parent_inputs(parent, parent_audit)
    checks.equal((spec["version"], report["status"], report["phase"]), (STUDY_VERSION, "completed", "completed"),
                 "Completed versioned route comparison")
    fingerprint(spec, checks, "Frozen route comparison contract")
    checks.equal((spec["identity"], spec["scene"], spec["parent"]),
                 (parent_spec["identity"], parent_spec["scene"], parent_binding), "Exact audited native64 parent")
    checks.equal((spec["canonical_positions"], spec["collection_positions"], spec["test_positions"], spec["methods"],
                  spec["attempts_per_method_per_position"], spec["physics_substeps"], spec["seed"],
                  spec["candidate_budget_per_method"], spec["collection_budget_per_route"], spec["label_version"], spec["routes"]),
                 (33, 24, 50, list(MODELS), 1, 64, MODEL_SEED, CANDIDATE_BUDGET, PER_ROUTE_LIMIT, LABEL_VERSION, list(ROUTES)),
                 "Frozen routes, equal preparation budgets and one native attempt per method")
    manifest = replay.check_manifest(study, checks)
    checks.equal(report["evidence_manifest_sha256"], sha(study / "evidence-manifest.json"), "Report binds complete evidence bytes")
    frozen = ("splits.json", "collection-menu.json", "training-prior.json")
    checks.equal(spec["frozen_inputs"], {name: manifest[name] for name in frozen}, "Exact inputs frozen before collection")
    checks.equal(report["choices_sha256"], manifest["choices.json"], "Report binds frozen choices")
    canonical, binding = load_prior(prior_study, prior_audit, history_root)
    checks.equal(binding, spec["prior_proposer"], "Exact earlier audited training evidence and route labels")
    previous = verify_previous_layouts(prior_study, checks)
    historical = verify_history_files(read(study / "splits.json")["provenance"], history_root, study, checks)
    grid_path = parent / "grid.json"
    checks.equal(sha(grid_path), read(parent / "evidence-manifest.json")["grid.json"], "Exact original bucket grid")
    grid = read(grid_path)
    split = verify_splits(study, grid, spec["scene"], assets,
                          historical | {tuple(row["bucket_xy_m"]) for row in canonical}, previous, checks)
    prior, home = read(study / "training-prior.json"), read(parent / "setup-home.json")
    checks.equal(prior, read(prior_study / "training-prior.json"), "Reuse exact audited TRAIN geometry anchors")
    resolutions = {}
    train_requests, packets = verify_collection(study, split["train"], route_library(canonical), prior, home, resolutions, checks)
    fresh, excluded = fresh_rows(study, split["train"], manifest)
    training = list(observed_rows(study / "training-dataset.jsonl"))
    verify_training_rows(training, canonical, fresh, split, checks)
    checkpoint = verify_checkpoint(study / "route-models.json", training, ROUTES, MODEL_SEED, checks)
    gate = read(study / "route-gate.json")
    fingerprint(gate, checks, "Frozen common route gate")
    selections, candidate_counts = verify_test_choices(study, split["test"], route_library(training), checkpoint,
                                                        gate, prior, home, resolutions, checks)
    expected = dict(train=train_requests, test=[slot["request"] for slot in selections if slot["request"] is not None])
    observations, controls = audit_packets(study, parent, assets, spec, report, manifest, expected, prior, home, checks)
    verify_realized_routes(study, observations, checks)
    train_intents = {request["id"]: request["intended_route"] for request in train_requests}
    for row, _ in observations["train"].values():
        checks.equal(row["intended_route"], train_intents[row["id"]], row["id"] + "/exact declared TRAIN route")
    collected = {key: row for key, (row, _) in observations["train"].items()}
    checks.equal(gate, fit_gate(split["train"], packets, collected), "Exact shared gate from pre-test collection outcomes")
    checks.equal(report["collection"], summarize_collection(gate), "Exact per-route collection accounting")
    checks.equal(report["training"], training_summary(checkpoint, canonical, fresh, excluded, manifest["route-models.json"],
                                                       manifest["route-gate.json"]), "Exact common TRAIN dataset and model support")
    comparison = account_shots(selections, observations["test"], checks)
    checks.equal(report["comparison"], comparison, "Exact paired comparison from rescored native evidence")
    checks.equal((report["collection_prepared"], report["test_prepared"]), (24, 50), "Every frozen layout prepared")
    return dict(version=VERSION, status="passed", checks_count=checks.count, control_releases=controls,
                train_releases=len(observations["train"]), test_releases=len(observations["test"]),
                training=report["training"], collection=report["collection"], comparison=comparison,
                candidate_counts=candidate_counts, report_sha256=sha(study / "report.json"),
                evidence_manifest_sha256=report["evidence_manifest_sha256"], checkpoint_sha256=manifest["route-models.json"],
                gate_sha256=manifest["route-gate.json"], choices_sha256=manifest["choices.json"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("study", "parent-study", "parent-audit", "prior-study", "prior-audit", "history-root", "assets", "audit"):
        parser.add_argument("--" + name, required=True, type=Path)
    args = parser.parse_args()
    if args.audit.exists() or args.audit.resolve().is_relative_to(args.study.resolve()):
        parser.error("Write a fresh audit outside the frozen evidence directory")
    try:
        result = audit(args.study, args.parent_study, args.parent_audit, args.prior_study, args.prior_audit,
                       args.history_root, args.assets)
    except Exception as error:
        result = dict(version=VERSION, status="failed", error=f"{type(error).__name__}: {error}")
    atomic(args.audit, result)
    print(json.dumps(result), flush=True)
    if result["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
