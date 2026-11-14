"""Reproduce the frozen table-shot library comparison and score native traces on CPU."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

from . import replay650_verify as replay
from .benchmark_verify import Checks, digest, fingerprint, read, sha
from .cell1050_rebound_study import parent_inputs
from .intergrid_layouts import MIN_SEPARATION_M, _placement_geometry
from .intergrid_library_inputs import PLAN_VERSION, load_library_inputs, sample_layouts
from .intergrid_library_study import VERSION as STUDY_VERSION
from .intergrid_proposer_audit import audit_packets, audit_preparation
from .intergrid_route_audit import verify_history_files, verify_realized_routes, verify_separation
from .intergrid_route_labels import LABEL_VERSION
from .intergrid_route_models import _valid_release, propose_route_candidates
from .intergrid_route_study import PER_ROUTE_TEST, route_library, source_menu
from .sweep_worker import atomic


VERSION = "sircus-intergrid-table-library-independent-audit-v1"
SPLIT_SEED = 2026112001
POSITION_COUNT = 50
METHODS = ("original", "expanded")
ROUTE = "table"
MODEL = "specialist"


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
    checks.equal(provenance["excluded_coordinates_sha256"], digest(exclusions), "Frozen exclusion coordinates")
    checks.equal(provenance["layouts_sha256"], digest(layouts), "Frozen held-out layouts")
    previous_xy = {tuple(row["bucket_xy_m"]) for row in previous}
    checks.equal(provenance["declared_previous_coordinates_m"], [list(xy) for xy in sorted(previous_xy)],
                 "Exact declared earlier training, collection and evaluation positions")
    historical = verify_history_files(provenance, history_root, study, checks)
    known = historical | previous_xy | {tuple(row["bucket_xy_m"]) for row in grid}
    checks.require(known.issubset(exclusions), "Every known placement excluded")
    checks.equal(layouts, sample_layouts(exclusions, geometry, grid, seed=SPLIT_SEED),
                 "Reproduce held-out positions from the frozen seed")
    verify_separation(layouts, [], exclusions, checks)
    return layouts


def verify_library(original_rows, added_rows, checks):
    table_additions = [row for row in added_rows if row["shot_route"] == ROUTE and _valid_release(row)]
    checks.require(bool(table_additions), "The expansion includes measured new table captures")
    for row in table_additions:
        checks.require(row["phase"] == "train" and row.get("role") not in {"test", "validation"},
                       row["id"] + "/added starting shot is a training capture")
    libraries = dict(original=route_library(original_rows), expanded=route_library(original_rows + table_additions))
    checks.equal(libraries["original"]["direct"], libraries["expanded"]["direct"], "Direct sources remain identical")

    def table_ids(library):
        return {row["id"] for rows in library[ROUTE].values() for row in rows}

    old_ids, expanded_ids = map(table_ids, libraries.values())
    additions = {row["id"] for row in table_additions}
    checks.equal(len(additions), len(table_additions), "Added starting shot IDs are unique")
    checks.require(old_ids.isdisjoint(additions), "Starting shots added without replacing original rows")
    checks.equal(expanded_ids, old_ids | additions, "Expanded library adds only verified new table captures")
    return libraries, {
        method: dict(shots=sum(len(rows) for rows in library[ROUTE].values()), positions=len(library[ROUTE]))
        for method, library in libraries.items()
    }


def verify_choices(study, layouts, libraries, checkpoint, checkpoint_hash, prior, home, checks):
    folder = study / "test"
    checks.equal({path.stem for path in folder.glob("*.json")}, {row["id"] for row in layouts},
                 "One candidate record per held-out position")
    resolutions, slots = {}, []
    totals = {method: Counter() for method in METHODS}
    for layout in layouts:
        identity = layout["id"]
        artifact = read(folder / (identity + ".json"))
        menus = {method: source_menu(layout, libraries[method])[PER_ROUTE_TEST:] for method in METHODS}
        expected = []
        for method, sources in menus.items():
            checks.equal([source["shot_route"] for source in sources], [ROUTE] * PER_ROUTE_TEST,
                         identity + "/" + method + "/four table source slots")
            checks.equal(len({source["id"] for source in sources}), PER_ROUTE_TEST,
                         identity + "/" + method + "/distinct sources")
            expected.extend(dict(row, id=f"{row['id']}-{method}", method=method)
                            for row in propose_route_candidates(
                                checkpoint, MODEL, layout["bucket_xy_m"], sources,
                                context_id=identity, limit=PER_ROUTE_TEST))
        checks.equal(artifact["proposals"], expected,
                     identity + "/same frozen specialist evaluated with each library")
        proposals, admitted = audit_preparation(artifact, layout, "test", study / "native-heading",
                                                prior, home, resolutions, checks)
        for label, candidates in (("proposed", proposals), ("admitted", admitted)):
            used = set()
            for row in candidates:
                method, index, name = row["method"], row["source_index"], row["id"]
                checks.require(type(index) is int and 0 <= index < PER_ROUTE_TEST,
                               name + "/within four table source slots")
                checks.require((method, index) not in used, name + "/at most one candidate per source slot")
                used.add((method, index))
                source = menus[method][index]
                checks.equal((row["training_recipe_id"], row["training_bucket_xy_m"], row["intended_route"],
                              row["route_models_fingerprint"]),
                             (source["id"], source["bucket_xy_m"], ROUTE, checkpoint["fingerprint"]),
                             name + "/exact library source and fixed checkpoint")
                totals[method][label] += 1

        for method in METHODS:
            name = f"{identity}-{method}-{ROUTE}"
            chosen = next((row for row in admitted if row["method"] == method), None)
            slots.append(dict(id=name, method=method, context_id=identity, bucket_xy_m=layout["bucket_xy_m"],
                              stratum=layout["stratum"], intended_route=ROUTE,
                              request=None if chosen is None else dict(chosen, id=name, stratum=layout["stratum"])))
    checks.equal(read(study / "choices.json"), dict(checkpoint_sha256=checkpoint_hash, slots=slots),
                 "First admitted candidate per library frozen before any release")
    return slots, {method: dict(counter) for method, counter in totals.items()}


def account_shots(layouts, slots, observations, checks):
    by_layout = {layout["id"]: {} for layout in layouts}
    counts = {method: Counter() for method in METHODS}
    checks.equal(len({slot["id"] for slot in slots}), len(slots), "Unique frozen slot IDs")
    checks.equal(set(observations), {slot["id"] for slot in slots if slot["request"] is not None},
                 "Exactly one release for each admitted slot")
    for slot in slots:
        name, method, identity = slot["id"], slot["method"], slot["context_id"]
        checks.equal(slot["intended_route"], ROUTE, name + "/table route slot")
        checks.require(method not in by_layout[identity], name + "/one attempt per library at each position")
        admitted = slot["request"] is not None
        outcome = "setup_rejected"
        if admitted:
            row, measured = observations[name]
            checks.equal(row["intended_route"], ROUTE, name + "/release retains the requested route")
            if measured == "standard_capture":
                outcome = row["shot_route"] or "unclassified_capture"
            elif measured == "robot_assisted_capture":
                outcome = "arm_assisted_capture"
            elif measured == "invalid_physics":
                outcome = "invalid_physics"
            else:
                outcome = "flight_miss"
        counts[method][outcome] += 1
        by_layout[identity][method] = (admitted, outcome == ROUTE)

    labels = {(True, True): "both", (True, False): "original_only",
              (False, True): "expanded_only", (False, False): "neither"}
    paired = {metric: dict.fromkeys(labels.values(), 0) for metric in ("setup", "capture")}
    for identity, results in by_layout.items():
        checks.equal(set(results), set(METHODS), identity + "/both libraries represented")
        for index, metric in enumerate(paired):
            paired[metric][labels[tuple(results[method][index] for method in METHODS)]] += 1

    methods = {}
    for method, outcomes in counts.items():
        attempts, captures = sum(outcomes.values()), outcomes[ROUTE]
        checks.equal(attempts, POSITION_COUNT, method + "/all 50 positions count")
        releases = attempts - outcomes["setup_rejected"]
        methods[method] = dict(attempts=attempts, releases=releases, setup_rate=releases / attempts,
                               captures=captures, capture_rate=captures / attempts, outcomes=dict(outcomes))
    return dict(methods=methods, paired=paired)


def audit(study, parent, parent_audit, model_study, model_audit, paired_study, paired_audit,
          matched_study, matched_audit, refit_study, refit_audit, history_root, assets):
    checks = Checks()
    spec, report = read(study / "contract.json"), read(study / "report.json")
    parent_spec, parent_binding, _ = parent_inputs(parent, parent_audit)
    checks.equal((spec["version"], report["status"], report["phase"]),
                 (STUDY_VERSION, "completed", "completed"), "Completed frozen library comparison")
    checks.equal(report["test_prepared"], POSITION_COUNT, "Every fresh position prepared")
    fingerprint(spec, checks, "Table library comparison contract")
    checks.equal((spec["parent"], spec["identity"], spec["scene"]),
                 (parent_binding, parent_spec["identity"], parent_spec["scene"]), "Exact audited native64 parent")
    checks.equal((spec["test_positions"], spec["methods"], spec["routes"],
                  spec["attempts_per_method_per_route_per_position"], spec["candidate_budget_per_route"],
                  spec["physics_substeps"], spec["label_version"], spec["split_seed"]),
                 (POSITION_COUNT, list(METHODS), [ROUTE], 1, PER_ROUTE_TEST, 64, LABEL_VERSION, SPLIT_SEED),
                 "Matched attempt budgets, fresh split seed and unchanged route scoring")

    manifest = replay.check_manifest(study, checks)
    checks.equal(report["evidence_manifest_sha256"], sha(study / "evidence-manifest.json"),
                 "Report binds all saved evidence")
    checks.equal(report["choices_sha256"], manifest["choices.json"], "Report binds frozen choices")
    checks.equal(spec["frozen_inputs"], {name: manifest[name] for name in
                 ("layouts.json", "training-prior.json", "route-models.json")}, "Exact frozen input bytes")
    checkpoint, original, additions, previous, binding = load_library_inputs(
        history_root, model_study, model_audit, paired_study, paired_audit,
        matched_study, matched_audit, refit_study, refit_audit)
    for key, expected in binding.items():
        checks.equal(spec[key], expected, "Exact audited input binding/" + key)
    refit_spec = read(refit_study / "contract.json")
    checks.equal((refit_spec["parent"], refit_spec["identity"]), (parent_binding, spec["identity"]),
                 "Same audited native64 scene as refit")
    checkpoint_hash = binding["refit_study"]["checkpoint_sha256"]
    checks.equal((manifest["route-models.json"], report["checkpoint_sha256"]), (checkpoint_hash, checkpoint_hash),
                 "Both variants use exact audited refit checkpoint bytes")
    checks.equal(manifest["training-prior.json"], binding["model_study"]["geometry_prior"],
                 "Original audited arm geometry prior")

    libraries, summary = verify_library(original, additions, checks)
    checks.equal(report["library"], summary, "Exact starting-shot library counts")
    grid_path = parent / "grid.json"
    checks.equal(sha(grid_path), read(parent / "evidence-manifest.json")["grid.json"], "Exact original bucket grid")
    layouts = verify_layouts(study, read(grid_path), spec["scene"], assets, history_root, previous, checks)
    prior, home = read(study / "training-prior.json"), read(parent / "setup-home.json")
    slots, candidate_counts = verify_choices(study, layouts, libraries, checkpoint, checkpoint_hash,
                                              prior, home, checks)
    requests = [slot["request"] for slot in slots if slot["request"] is not None]
    checks.equal((report["test_slots"], report["test_admitted"]), (len(slots), len(requests)),
                 "Exact counts of frozen attempts and admitted releases")
    measured, controls = audit_packets(study, parent, assets, spec, report, manifest,
                                        {"test": requests}, prior, home, checks)
    verify_realized_routes(study, measured, checks)
    comparison = account_shots(layouts, slots, measured["test"], checks)
    checks.equal(report["comparison"], comparison, "Exact paired setup and capture results from rescored evidence")
    return dict(version=VERSION, status="passed", checks_count=checks.count, test_positions=len(layouts),
                test_attempts=len(slots), test_releases=len(requests), control_releases=controls,
                comparison=comparison, candidate_counts=candidate_counts, library=summary,
                report_sha256=sha(study / "report.json"), evidence_manifest_sha256=report["evidence_manifest_sha256"],
                checkpoint_sha256=checkpoint_hash, choices_sha256=manifest["choices.json"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("study", "parent-study", "parent-audit", "model-study", "model-audit", "paired-study",
                 "paired-audit", "matched-study", "matched-audit", "refit-study", "refit-audit",
                 "history-root", "assets", "audit"):
        parser.add_argument("--" + name, required=True, type=Path)
    args = parser.parse_args()
    if args.audit.exists() or args.audit.resolve().is_relative_to(args.study.resolve()):
        parser.error("Write a fresh audit outside the frozen evidence directory")
    try:
        result = audit(args.study, args.parent_study, args.parent_audit, args.model_study, args.model_audit,
                       args.paired_study, args.paired_audit, args.matched_study, args.matched_audit,
                       args.refit_study, args.refit_audit, args.history_root, args.assets)
    except Exception as error:
        result = dict(version=VERSION, status="failed", error=f"{type(error).__name__}: {error}")
    atomic(args.audit, result)
    print(json.dumps(result), flush=True)
    if result["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
