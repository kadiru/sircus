"""Audit the route selection comparison and rescore native evidence on CPU."""
from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path

from . import replay650_verify as replay
from .benchmark_verify import Checks, digest, fingerprint, read, sha
from .cell1050_rebound_study import parent_inputs
from .intergrid_layouts import MIN_SEPARATION_M, _placement_geometry
from .intergrid_library_audit import verify_library
from .intergrid_proposer_audit import audit_packets, audit_preparation
from .intergrid_route_audit import verify_history_files, verify_realized_routes, verify_separation
from .intergrid_route_fallback_inputs import PLAN_VERSION, load_fallback_inputs, sample_layouts
from .intergrid_route_fallback_study import VERSION as STUDY_VERSION
from .intergrid_route_labels import LABEL_VERSION
from .intergrid_route_selector_audit import ROUTES, STANDARD_CAPTURES, verify_proposals
from .intergrid_route_study import PER_ROUTE_TEST
from .sweep_worker import atomic


VERSION = "sircus-intergrid-route-fallback-independent-audit-v1"
SPLIT_SEED = 2026112201
PASS_SEEDS = [SPLIT_SEED, SPLIT_SEED + 1]
POSITION_COUNT = 100
METHODS = ("direct_first", "route_selector")


def verify_layouts(study, grid, scene, assets, history_root, previous, checks):
    artifact = read(study / "layouts.json")
    layouts, provenance = artifact["layouts"], artifact["provenance"]
    checks.equal(len(layouts), POSITION_COUNT, "100 fresh bucket positions")
    checks.equal((provenance["version"], provenance["seed"], provenance["pass_seeds"],
                  provenance["minimum_separation_m"]),
                 (PLAN_VERSION, SPLIT_SEED, PASS_SEEDS, MIN_SEPARATION_M), "Prespecified placement protocol")
    geometry = _placement_geometry(scene, assets)
    checks.equal(provenance["geometry"], geometry, "Reconstructed tabletop and fixed-base geometry")
    exclusions = [tuple(xy) for xy in provenance["excluded_coordinates_m"]]
    checks.equal(exclusions, sorted(set(exclusions)), "Sorted unique excluded coordinates")
    checks.equal(provenance["excluded_coordinates_sha256"], digest(exclusions), "Frozen exclusion coordinates")
    checks.equal(provenance["layouts_sha256"], digest(layouts), "Frozen held-out layouts")
    previous_xy = {tuple(row["bucket_xy_m"]) for row in previous}
    checks.equal(provenance["declared_previous_coordinates_m"], [list(xy) for xy in sorted(previous_xy)],
                 "Exact earlier training, collection and evaluation positions")
    historical = verify_history_files(provenance, history_root, study, checks)
    known = historical | previous_xy | {tuple(row["bucket_xy_m"]) for row in grid}
    checks.require(known.issubset(exclusions), "Every known placement excluded")
    checks.equal(layouts, sample_layouts(exclusions, geometry, grid, seed=SPLIT_SEED),
                 "Reproduce both sampling passes from the frozen seeds")
    verify_separation(layouts, [], exclusions, checks)
    return layouts


def verify_choices(study, layouts, libraries, checkpoint, checkpoint_hash, history, prior, home, checks):
    folder = study / "test"
    checks.equal({path.stem for path in folder.glob("*.json")}, {row["id"] for row in layouts},
                 "One candidate record per held-out position")
    checks.require(len({row["id"] for row in history}) == len(history) >= 5, "Unique historical paired sites")
    resolutions, slots = {}, []
    candidate_counts = {route: Counter() for route in ROUTES}
    for layout in layouts:
        identity, xy = layout["id"], layout["bucket_xy_m"]
        artifact = read(folder / (identity + ".json"))
        verify_proposals(artifact, layout, libraries, checkpoint, checks)
        proposals, admitted = audit_preparation(artifact, layout, "test", study / "native-heading",
                                                prior, home, resolutions, checks)
        first = {}
        for row in proposals:
            candidate_counts[row["intended_route"]]["proposed"] += 1
        for row in admitted:
            route = row["intended_route"]
            candidate_counts[route]["admitted"] += 1
            first.setdefault(route, row)

        neighbors = sorted(history, key=lambda row: (math.dist(xy, row["bucket_xy_m"]), row["id"]))[:5]
        scores = {route: (1 + sum(row[route + "_capture"] for row in neighbors)) / (2 + len(neighbors))
                  for route in ROUTES}
        selector_order = ["table", "direct"] if scores["table"] > scores["direct"] else list(ROUTES)
        available = [route for route in ROUTES if route in first]
        for method, priority, neighbor_ids, gate_scores in (
            ("direct_first", list(ROUTES), [], None),
            ("route_selector", selector_order, [row["id"] for row in neighbors], scores),
        ):
            name = f"{identity}-{method}"
            route = next((route for route in priority if route in first), None)
            chosen = None if route is None else dict(first[route], id=name, method=method, stratum=layout["stratum"])
            slots.append(dict(id=name, method=method, context_id=identity, bucket_xy_m=xy,
                              stratum=layout["stratum"], intended_route=route, route_priority=priority,
                              available_routes=available, neighbor_ids=neighbor_ids,
                              gate_scores=gate_scores, request=chosen))
    checks.equal(read(study / "choices.json"), dict(checkpoint_sha256=checkpoint_hash, slots=slots),
                 "Reconstructed first admitted choices for direct-first and frozen selector")
    return slots, {route: dict(counts) for route, counts in candidate_counts.items()}


def _released_outcome(row, measured):
    if measured == "standard_capture":
        return row["shot_route"] or "unclassified_capture"
    if measured == "robot_assisted_capture":
        return "arm_assisted_capture"
    return "invalid_physics" if measured == "invalid_physics" else "flight_miss"


def account_shots(layouts, slots, observations, checks):
    by_layout = {layout["id"]: {} for layout in layouts}
    counts = {method: Counter() for method in METHODS}
    selected_routes = {method: dict.fromkeys(ROUTES, 0) for method in METHODS}
    requested_route_captures = Counter()
    checks.equal(len({slot["id"] for slot in slots}), len(slots), "Unique frozen slot IDs")
    checks.equal(set(observations), {slot["id"] for slot in slots if slot["request"] is not None},
                 "Exactly one release for each admitted slot")
    for slot in slots:
        name, method, identity, route = (slot[key] for key in ("id", "method", "context_id", "intended_route"))
        request = slot["request"]
        checks.require(method not in by_layout[identity], name + "/one attempt per method")
        outcome = "setup_rejected"
        if request is None:
            checks.equal((route, slot["available_routes"]), (None, []), name + "/no feasible route for refusal")
        else:
            row, measured = observations[name]
            checks.equal((row["intended_route"], request["intended_route"]), (route, route),
                         name + "/release retains its frozen selected route")
            outcome = _released_outcome(row, measured)
            selected_routes[method][route] += 1
            requested_route_captures[method] += int(outcome == route)
        counts[method][outcome] += 1
        by_layout[identity][method] = (slot, outcome)

    labels = {(True, True): "both", (True, False): "direct_first_only",
              (False, True): "route_selector_only", (False, False): "neither"}
    paired = {metric: dict.fromkeys(labels.values(), 0) for metric in ("setup", "capture")}
    by_feasibility = {category: dict.fromkeys(labels.values(), 0) for category in ("both_routes_feasible", "other")}
    same = dict(attempts=0, outcome_disagreements=0, capture_disagreements=0)
    different = dict.fromkeys(labels.values(), 0)
    for identity, results in by_layout.items():
        checks.equal(set(results), set(METHODS), identity + "/both methods represented")
        direct, direct_outcome = results["direct_first"]
        selector, selector_outcome = results["route_selector"]
        available = direct["available_routes"]
        checks.equal(selector["available_routes"], available, identity + "/both methods share route candidates")
        direct_route = direct["intended_route"]
        selector_route = selector["intended_route"]
        checks.equal(direct_route, next(iter(available), None), identity + "/direct-first priority and fallback")
        checks.equal((direct_route is not None, selector_route is not None), (bool(available), bool(available)),
                     identity + "/same setup availability for both methods")
        paired["setup"][labels[direct_route is not None, selector_route is not None]] += 1
        capture_pair = labels[direct_outcome in STANDARD_CAPTURES, selector_outcome in STANDARD_CAPTURES]
        paired["capture"][capture_pair] += 1
        group = "both_routes_feasible" if available == list(ROUTES) else "other"
        by_feasibility[group][capture_pair] += 1
        if direct_route != selector_route:
            checks.equal((available, direct_route, selector_route), (list(ROUTES), "direct", "table"),
                         identity + "/choice differs only for a both-feasible switch to table")
            different[capture_pair] += 1
        elif direct_route is not None:
            checks.equal({key: value for key, value in direct["request"].items() if key not in {"id", "method"}},
                         {key: value for key, value in selector["request"].items() if key not in {"id", "method"}},
                         identity + "/same pose, path and full release command")
            same["attempts"] += 1
            same["outcome_disagreements"] += int(direct_outcome != selector_outcome)
            same["capture_disagreements"] += int(capture_pair in {"direct_first_only", "route_selector_only"})

    methods = {}
    for method, outcomes in counts.items():
        attempts = sum(outcomes.values())
        checks.equal(attempts, POSITION_COUNT, method + "/all 100 positions count")
        releases = attempts - outcomes["setup_rejected"]
        captures = sum(outcomes[outcome] for outcome in STANDARD_CAPTURES)
        methods[method] = dict(attempts=attempts, releases=releases, setup_rate=releases / attempts,
                               captures=captures, capture_rate=captures / attempts, outcomes=dict(outcomes),
                               selected_routes=selected_routes[method],
                               requested_route_captures=requested_route_captures[method])
    return dict(methods=methods, paired=paired, capture_by_feasibility=by_feasibility,
                same_command=same, different_command=different)


def audit(study, parent, parent_audit, model_study, model_audit, paired_study, paired_audit,
          matched_study, matched_audit, refit_study, refit_audit, library_study, library_audit,
          selector_study, selector_audit, history_root, assets):
    checks = Checks()
    spec, report = read(study / "contract.json"), read(study / "report.json")
    parent_spec, parent_binding, _ = parent_inputs(parent, parent_audit)
    checks.equal((spec["version"], report["status"], report["phase"]),
                 (STUDY_VERSION, "completed", "completed"), "Completed direct-first comparison")
    fingerprint(spec, checks, "Direct-first comparison contract")
    checks.equal((spec["parent"], spec["identity"], spec["scene"]),
                 (parent_binding, parent_spec["identity"], parent_spec["scene"]), "Exact audited native64 parent")
    checks.equal((spec["test_positions"], spec["methods"], spec["routes"],
                  spec["attempts_per_method_per_position"], spec["candidate_budget_per_route"],
                  spec["physics_substeps"], spec["label_version"], spec["split_seed"], spec["split_seeds"]),
                 (POSITION_COUNT, list(METHODS), list(ROUTES), 1, PER_ROUTE_TEST, 64, LABEL_VERSION,
                  SPLIT_SEED, PASS_SEEDS), "Matched one-shot budget, fresh split and unchanged native scoring")
    checks.equal(report["test_prepared"], POSITION_COUNT, "Every fresh position prepared")

    manifest = replay.check_manifest(study, checks)
    checks.equal(report["evidence_manifest_sha256"], sha(study / "evidence-manifest.json"),
                 "Report binds all saved evidence")
    checks.equal(report["choices_sha256"], manifest["choices.json"], "Report binds frozen choices")
    checks.equal(spec["frozen_inputs"], {name: manifest[name] for name in
                 ("layouts.json", "training-prior.json", "route-models.json", "route-history.json")},
                 "Exact frozen input bytes")
    checkpoint, original, additions, previous, binding, history = load_fallback_inputs(
        history_root, model_study, model_audit, paired_study, paired_audit, matched_study, matched_audit,
        refit_study, refit_audit, library_study, library_audit, selector_study, selector_audit)
    for key, expected in binding.items():
        checks.equal(spec[key], expected, "Exact audited input binding/" + key)
    selector_spec = read(selector_study / "contract.json")
    checks.equal((selector_spec["parent"], selector_spec["identity"]), (parent_binding, spec["identity"]),
                 "Same audited native64 scene as the prior selector")
    checkpoint_hash = binding["selector_study"]["checkpoint_sha256"]
    checks.equal((manifest["route-models.json"], report["checkpoint_sha256"]), (checkpoint_hash, checkpoint_hash),
                 "Both methods use the exact audited refit checkpoint bytes")
    checks.equal(manifest["training-prior.json"], binding["model_study"]["geometry_prior"],
                 "Original audited arm geometry prior")
    checks.equal(manifest["route-history.json"], binding["selector_study"]["route-history.json"],
                 "Exact historical selection data bytes from the original selector")
    checks.equal(read(study / "route-history.json"), history, "Exact paired historic specialist evidence")
    checks.equal(report["selection_source_positions"], len(history), "Historical paired sites counted")

    libraries, summary = verify_library(original, additions, checks)
    checks.equal(report["library"], summary, "Exact starting-shot library counts")
    grid_path = parent / "grid.json"
    checks.equal(sha(grid_path), read(parent / "evidence-manifest.json")["grid.json"], "Exact original bucket grid")
    layouts = verify_layouts(study, read(grid_path), spec["scene"], assets, history_root, previous, checks)
    prior, home = read(study / "training-prior.json"), read(parent / "setup-home.json")
    slots, candidate_counts = verify_choices(study, layouts, libraries, checkpoint, checkpoint_hash,
                                              history, prior, home, checks)
    requests = [slot["request"] for slot in slots if slot["request"] is not None]
    checks.equal((report["test_slots"], report["test_admitted"]), (len(slots), len(requests)),
                 "Exact counts of frozen attempts and admitted releases")
    measured, controls = audit_packets(study, parent, assets, spec, report, manifest,
                                        {"test": requests}, prior, home, checks)
    verify_realized_routes(study, measured, checks)
    comparison = account_shots(layouts, slots, measured["test"], checks)
    checks.equal(report["comparison"], comparison, "Exact paired and route switch results from rescored evidence")
    return dict(version=VERSION, status="passed", checks_count=checks.count, test_positions=len(layouts),
                test_attempts=len(slots), test_releases=len(requests), control_releases=controls,
                comparison=comparison, candidate_counts=candidate_counts, library=summary,
                selection_source_positions=len(history), report_sha256=sha(study / "report.json"),
                evidence_manifest_sha256=report["evidence_manifest_sha256"], checkpoint_sha256=checkpoint_hash,
                choices_sha256=manifest["choices.json"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("study", "parent-study", "parent-audit", "model-study", "model-audit", "paired-study",
                 "paired-audit", "matched-study", "matched-audit", "refit-study", "refit-audit",
                 "library-study", "library-audit", "selector-study", "selector-audit",
                 "history-root", "assets", "audit"):
        parser.add_argument("--" + name, required=True, type=Path)
    args = parser.parse_args()
    if args.audit.exists() or args.audit.resolve().is_relative_to(args.study.resolve()):
        parser.error("Write a fresh audit outside the frozen evidence directory")
    try:
        result = audit(args.study, args.parent_study, args.parent_audit, args.model_study, args.model_audit,
                       args.paired_study, args.paired_audit, args.matched_study, args.matched_audit,
                       args.refit_study, args.refit_audit, args.library_study, args.library_audit,
                       args.selector_study, args.selector_audit, args.history_root, args.assets)
    except Exception as error:
        result = dict(version=VERSION, status="failed", error=f"{type(error).__name__}: {error}")
    atomic(args.audit, result)
    print(json.dumps(result), flush=True)
    if result["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
