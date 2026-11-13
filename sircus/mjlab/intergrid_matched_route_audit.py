"""Reproduce a frozen matched route comparison from its saved evidence on CPU."""
from __future__ import annotations

import argparse
from collections import Counter
from itertools import combinations
import json
from pathlib import Path

from . import replay650_verify as replay
from .benchmark_verify import Checks, digest, fingerprint, read, sha
from .cell1050_rebound_study import parent_inputs
from .intergrid_layouts import MIN_SEPARATION_M, _placement_geometry
from .intergrid_matched_route_plan import POSITION_COUNT, SEED, VERSION as PLAN_VERSION, sample_layouts
from .intergrid_matched_route_study import VERSION as STUDY_VERSION, load_model_inputs
from .intergrid_proposer_audit import audit_packets, audit_preparation
from .intergrid_route_audit import verify_candidate_budget, verify_history_files, verify_realized_routes, verify_separation
from .intergrid_route_labels import LABEL_VERSION
from .intergrid_route_models import MODELS
from .intergrid_route_plan import ROUTES
from .intergrid_route_study import PER_ROUTE_TEST, route_library, source_menu, test_proposals
from .sweep_worker import atomic


VERSION = "sircus-intergrid-matched-route-independent-audit-v1"


def verify_layouts(study, grid, scene, assets, history_root, previous, checks):
    artifact = read(study / "layouts.json")
    layouts, provenance = artifact["layouts"], artifact["provenance"]
    checks.equal(len(layouts), POSITION_COUNT, "Fifty paired held-out placements")
    checks.equal((provenance["version"], provenance["seed"], provenance["minimum_separation_m"]),
                 (PLAN_VERSION, SEED, MIN_SEPARATION_M), "Prespecified placement protocol")
    geometry = _placement_geometry(scene, assets)
    checks.equal(provenance["geometry"], geometry, "CPU-reconstructed tabletop and fixed-base geometry")
    exclusions = [tuple(xy) for xy in provenance["excluded_coordinates_m"]]
    checks.equal(exclusions, sorted(set(exclusions)), "Sorted unique exclusion coordinates")
    checks.equal(provenance["excluded_coordinates_sha256"], digest(exclusions), "Frozen exclusion coordinates")
    checks.equal(provenance["layouts_sha256"], digest(layouts), "Frozen held-out layouts")
    previous_xy = sorted({tuple(row["bucket_xy_m"]) for row in previous})
    checks.equal(provenance["declared_previous_coordinates_m"], [list(xy) for xy in previous_xy],
                 "Exact frozen-model training and previous test locations")
    historical = verify_history_files(provenance, history_root, study, checks)
    known = historical | set(previous_xy) | {tuple(row["bucket_xy_m"]) for row in grid}
    checks.require(known.issubset(exclusions), "All known placements excluded")
    checks.equal(layouts, sample_layouts(exclusions, geometry, grid, seed=SEED),
                 "Reproduce placements from frozen seed, geometry and exclusions")
    verify_separation(layouts, [], exclusions, checks)
    return layouts


def verify_choices(study, layouts, library, checkpoint, checkpoint_hash, prior, home, checks):
    folder = study / "test"
    checks.equal({path.stem for path in folder.glob("*.json")}, {row["id"] for row in layouts},
                 "One complete candidate record per held-out layout")
    resolutions = {}
    slots = []
    totals = {method: {route: Counter() for route in ROUTES} for method in MODELS}
    for layout in layouts:
        identity = layout["id"]
        artifact = read(folder / (identity + ".json"))
        checks.equal(artifact["proposals"], test_proposals(layout, library, checkpoint),
                     identity + "/frozen predictions from the common source menu")
        proposals, admitted = audit_preparation(artifact, layout, "test", study / "native-heading",
                                                prior, home, resolutions, checks)
        counts = verify_candidate_budget(identity, source_menu(layout, library), proposals, admitted, checks)
        for method in MODELS:
            for route in ROUTES:
                totals[method][route].update(counts[method][route])
                identifier = f"{identity}-{method}-{route}"
                request = next((row for row in admitted
                                if row["method"] == method and row["intended_route"] == route), None)
                slots.append(dict(id=identifier, method=method, context_id=identity,
                                  bucket_xy_m=layout["bucket_xy_m"], stratum=layout["stratum"],
                                  intended_route=route,
                                  request=None if request is None else dict(request, id=identifier,
                                                                            stratum=layout["stratum"])))
    choices = read(study / "choices.json")
    checks.equal(choices, dict(checkpoint_sha256=checkpoint_hash, slots=slots),
                 "Every slot uses its first admitted proposal for the requested route")
    return slots, {method: {route: dict(counts) for route, counts in by_route.items()}
                   for method, by_route in totals.items()}


def account_shots(slots, observations, checks):
    counts = {method: {route: Counter() for route in ROUTES} for method in MODELS}
    pairs_by_layout = {}
    admitted_ids = {slot["id"] for slot in slots if slot["request"] is not None}
    checks.equal(set(observations), admitted_ids, "Exactly one release for each admitted slot")
    for slot in slots:
        identifier = slot["id"]
        method, route = slot["method"], slot["intended_route"]
        layout = pairs_by_layout.setdefault(slot["context_id"], {})
        key = method, route
        checks.require(key not in layout, identifier + "/one attempt per model and route")
        if slot["request"] is None:
            outcome = "setup_rejected"
        else:
            row, native_outcome = observations[identifier]
            checks.equal(row["intended_route"], route, identifier + "/release retains the required route")
            if native_outcome == "standard_capture":
                outcome = row["shot_route"] or "unclassified_capture"
            elif native_outcome in {"invalid_physics", "robot_assisted_capture"}:
                outcome = "arm_assisted_capture" if native_outcome == "robot_assisted_capture" else native_outcome
            else:
                outcome = "flight_miss"
        counts[method][route][outcome] += 1
        layout[key] = outcome == route

    expected = {(method, route) for method in MODELS for route in ROUTES}
    pairs = {route: {f"{a}_vs_{b}": Counter() for a, b in combinations(MODELS, 2)} for route in ROUTES}
    checks.equal(len(pairs_by_layout), POSITION_COUNT, "Every held-out placement represented")
    for identity, layout in pairs_by_layout.items():
        checks.equal(set(layout), expected, identity + "/all six comparison slots represented")
        for route in ROUTES:
            for a, b in combinations(MODELS, 2):
                success_a, success_b = layout[a, route], layout[b, route]
                if success_a and success_b:
                    label = "both"
                elif success_a:
                    label = a + "_only"
                elif success_b:
                    label = b + "_only"
                else:
                    label = "neither"
                pairs[route][f"{a}_vs_{b}"][label] += 1

    methods = {method: {} for method in MODELS}
    for method, routes in counts.items():
        for route, outcomes in routes.items():
            attempts, captures = sum(outcomes.values()), outcomes[route]
            checks.equal(attempts, POSITION_COUNT, f"{method}/{route}/all positions in denominator")
            methods[method][route] = dict(attempts=attempts, releases=attempts - outcomes["setup_rejected"],
                                          captures=captures, capture_rate=captures / attempts,
                                          outcomes=dict(outcomes))
    paired = {route: {pair: dict(counter) for pair, counter in items.items()} for route, items in pairs.items()}
    return dict(methods=methods, paired=paired)


def audit(study, parent, parent_audit, model_study, model_audit, history_root, assets):
    checks = Checks()
    spec, report = read(study / "contract.json"), read(study / "report.json")
    parent_spec, parent_binding, _ = parent_inputs(parent, parent_audit)
    checks.equal((spec["version"], report["status"], report["phase"]),
                 (STUDY_VERSION, "completed", "completed"), "Completed matched-route comparison")
    checks.equal(report["test_prepared"], POSITION_COUNT, "Every held-out placement was prepared")
    fingerprint(spec, checks, "Matched route contract")
    checks.equal((spec["parent"], spec["identity"], spec["scene"]),
                 (parent_binding, parent_spec["identity"], parent_spec["scene"]), "Exact audited native64 parent")
    checks.equal((spec["test_positions"], spec["methods"], spec["routes"],
                  spec["attempts_per_method_per_route_per_position"], spec["candidate_budget_per_route"],
                  spec["physics_substeps"], spec["label_version"]),
                 (POSITION_COUNT, list(MODELS), list(ROUTES), 1, PER_ROUTE_TEST, 64, LABEL_VERSION),
                 "Matched attempt budgets and original route scoring")
    manifest = replay.check_manifest(study, checks)
    checks.equal(report["evidence_manifest_sha256"], sha(study / "evidence-manifest.json"),
                 "Report binds complete evidence")
    checks.equal(spec["frozen_inputs"], {name: manifest[name] for name in ("layouts.json", "training-prior.json")},
                 "Exact frozen inputs")
    checks.equal(report["choices_sha256"], manifest["choices.json"], "Report binds frozen choices")

    checkpoint, rows, prior, previous, binding = load_model_inputs(model_study, model_audit)
    checks.equal(spec["model_study"], binding, "Exact independently audited checkpoint and source data")
    checks.equal(manifest["route-models.json"], binding["checkpoint"], "Exact copy of the frozen checkpoint")
    checks.equal(manifest["training-prior.json"], binding["geometry_prior"], "Exact audited setup geometry prior")
    grid_path = parent / "grid.json"
    checks.equal(sha(grid_path), read(parent / "evidence-manifest.json")["grid.json"], "Exact original bucket grid")
    layouts = verify_layouts(study, read(grid_path), spec["scene"], assets, history_root, previous, checks)
    home = read(parent / "setup-home.json")
    slots, candidate_counts = verify_choices(study, layouts, route_library(rows), checkpoint, binding["checkpoint"],
                                              prior, home, checks)
    requests = [slot["request"] for slot in slots if slot["request"] is not None]
    measured, controls = audit_packets(study, parent, assets, spec, report, manifest, {"test": requests}, prior, home,
                                        checks)
    verify_realized_routes(study, measured, checks)
    comparison = account_shots(slots, measured["test"], checks)
    checks.equal(report["comparison"], comparison, "Exact matched comparison from rescored native evidence")
    return dict(version=VERSION, status="passed", checks_count=checks.count, test_attempts=len(slots),
                test_releases=len(requests), control_releases=controls, comparison=comparison,
                candidate_counts=candidate_counts, report_sha256=sha(study / "report.json"),
                evidence_manifest_sha256=report["evidence_manifest_sha256"], checkpoint_sha256=binding["checkpoint"],
                choices_sha256=manifest["choices.json"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("study", "parent-study", "parent-audit", "model-study", "model-audit", "history-root", "assets", "audit"):
        parser.add_argument("--" + name, required=True, type=Path)
    args = parser.parse_args()
    if args.audit.exists() or args.audit.resolve().is_relative_to(args.study.resolve()):
        parser.error("Write a fresh audit outside the frozen evidence directory")
    try:
        result = audit(args.study, args.parent_study, args.parent_audit, args.model_study, args.model_audit,
                       args.history_root, args.assets)
    except Exception as error:
        result = dict(version=VERSION, status="failed", error=f"{type(error).__name__}: {error}")
    atomic(args.audit, result)
    print(json.dumps(result), flush=True)
    if result["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
