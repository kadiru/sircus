"""Reconstruct the frozen one-drop comparison and rescore its native traces on CPU."""
from __future__ import annotations

import math
from pathlib import Path
from statistics import median

from . import replay650_verify as replay
from .benchmark_verify import Checks, digest, fingerprint, read, sha
from .cell1050_rebound_study import parent_inputs
from .intergrid_candidate_replay_plan import command_key
from .intergrid_flight_ranking_study import _ranker
from .intergrid_layouts import MIN_SEPARATION_M, _placement_geometry
from .intergrid_library_audit import verify_library
from .intergrid_matched_route_plan import VERSION as PLAN_VERSION, sample_layouts
from .intergrid_proposer_audit import audit_packets, audit_preparation
from .intergrid_proposer_plan import _grid_points
from .intergrid_route_audit import verify_history_files, verify_realized_routes, verify_separation
from .intergrid_route_labels import LABEL_VERSION
from .intergrid_route_selector_plan import STANDARD_CAPTURES, _account
from .intergrid_system_learn import fit_ranker
from .intergrid_system_plan import BASE_PER_ROUTE, FALLBACK_PER_ROUTE, METHODS, ROUTES, base_proposals, fallback_proposals


VERSION = "sircus-intergrid-system-independent-audit-v1"
BASELINE, FALLBACK = METHODS


def _preparation_seconds(preparation):
    durations = [row["timing"]["wall_s"] for row in preparation["records"]]
    if not all(type(value) in (int, float) and math.isfinite(value) and value >= 0 for value in durations):
        raise ValueError("Setup decision timing must contain finite nonnegative wall durations")
    return math.fsum(durations)


def _latency(values):
    ordered = sorted(values)
    return dict(positions=len(ordered), median_s=median(ordered), p90_s=ordered[math.ceil(len(ordered) * .9) - 1],
                max_s=ordered[-1], aggregate_s=math.fsum(ordered))


def summarize(layouts, slots, records, sites, methods):
    """Report paired outcomes and measured CPU preparation latency per bucket position."""
    identities, performance, attempts = _account(layouts, slots, records, methods)
    if [site["layout"] for site in sites] != layouts:
        raise ValueError("Preparation artifacts must follow the frozen bucket order")

    paired, latency = {}, {}
    outcomes = {(True, True): "both", (True, False): "baseline_only",
                (False, True): "comparison_only", (False, False): "neither"}
    setup_timings = {site["layout"]["id"]: (
        _preparation_seconds(site["plan"]["base"]), _preparation_seconds(site["plan"]["fallback"]),
    ) for site in sites}
    latency[BASELINE] = _latency([base for base, _ in setup_timings.values()])
    for method in methods:
        if method == BASELINE:
            continue
        setup = dict.fromkeys(outcomes.values(), 0)
        capture = dict.fromkeys(outcomes.values(), 0)
        shared = dict(attempts=0, outcome_disagreements=0, capture_disagreements=0)
        extra = dict(positions=0, setups=0, captures=0)
        durations = []
        for identity in sorted(identities):
            baseline, baseline_outcome = attempts[identity, BASELINE]
            comparison, comparison_outcome = attempts[identity, method]
            left, right = baseline["request"], comparison["request"]
            left_hit, right_hit = baseline_outcome in STANDARD_CAPTURES, comparison_outcome in STANDARD_CAPTURES
            setup[outcomes[left is not None, right is not None]] += 1
            capture[outcomes[left_hit, right_hit]] += 1
            base_seconds, fallback_seconds = setup_timings[identity]
            durations.append(base_seconds + (fallback_seconds if left is None else 0))
            if left is None:
                extra["positions"] += 1
                extra["setups"] += int(right is not None)
                extra["captures"] += int(right_hit)
                continue
            if method == FALLBACK and (right is None or dict(left, id=right["id"], method=right["method"]) != right):
                raise ValueError("The setup-only fallback changed an already admitted baseline request")
            if right is not None and command_key(left) == command_key(right):
                shared["attempts"] += 1
                shared["outcome_disagreements"] += int(baseline_outcome != comparison_outcome)
                shared["capture_disagreements"] += int(left_hit != right_hit)
        paired[method] = dict(setup=setup, capture=capture, same_command=shared, recovery=extra)
        latency[method] = _latency(durations)
    return dict(methods=performance, paired_against_baseline=paired, setup_cpu_worker_wall_time=latency)


def verify_layouts(study, parent, history_root, scene, assets, previous, checks):
    """Reproduce both geometry-based samples from authenticated prior coordinates."""
    from .intergrid_system_study import POSITIONS, TEST_SEED, TRAIN_SEED, layout_ids

    archive = read(study / "layouts.json")
    checks.equal(set(archive), {"train", "test"}, "Exact development and held-out splits")
    grid = read(parent / "grid.json")
    geometry = _placement_geometry(scene, assets)
    canonical, _ = _grid_points(grid)
    prior = list(previous)
    result = {}
    for split, seed in (("train", TRAIN_SEED), ("test", TEST_SEED)):
        layouts, provenance = archive[split]["layouts"], archive[split]["provenance"]
        exclusions = [tuple(xy) for xy in provenance["excluded_coordinates_m"]]
        declared = sorted({tuple(row["bucket_xy_m"]) for row in prior})
        checks.equal(len(layouts), POSITIONS, split + "/50 frozen positions")
        checks.equal((provenance["version"], provenance["seed"], provenance["minimum_separation_m"], provenance["geometry"]),
                     (PLAN_VERSION, seed, MIN_SEPARATION_M, geometry), split + "/frozen spatial sampling contract")
        checks.equal(provenance["declared_previous_coordinates_m"], [list(xy) for xy in declared],
                     split + "/all supplied historical and development exclusions")
        checks.equal(exclusions, sorted(set(exclusions)), split + "/sorted unique exclusions")
        checks.equal(provenance["excluded_coordinates_sha256"], digest(exclusions), split + "/exclusion digest")
        checks.equal(provenance["layouts_sha256"], digest(layouts), split + "/layout digest")
        historical = verify_history_files(provenance, history_root, study, checks)
        known = historical | set(declared) | set(canonical.values())
        checks.equal(set(exclusions), known, split + "/all and only known positions excluded")
        expected = layout_ids(sample_layouts(exclusions, geometry, grid, seed=seed), split)
        checks.equal(layouts, expected, split + "/reconstructed placements from geometry and seed")
        verify_separation([dict(row, split="test") for row in layouts], [], exclusions, checks)
        prior.extend(layouts)
        result[split] = layouts
    checks.require(not {row["id"] for row in result["train"]} & {row["id"] for row in result["test"]},
                   "Held-out and development position IDs are disjoint")
    return result


def verify_plans(study, splits, libraries, checkpoint, ranker, prior, home, checks):
    """Rebuild each pre-release candidate list and verify each saved admission proof."""
    plans = read(study / "plans.json")
    checks.equal(set(plans), set(splits), "Preparation recorded for both splits")
    native = study / "native-heading"
    resolutions = {}
    for phase, layouts in splits.items():
        checks.equal([site["layout"] for site in plans[phase]], layouts,
                     phase + "/one ordered preparation record per position")
        for site in plans[phase]:
            layout, plan = site["layout"], site["plan"]
            identity = layout["id"]
            checks.equal(set(plan), {"base", "fallback"}, identity + "/two setup stages")
            checks.equal(plan["base"]["proposals"], base_proposals(layout, libraries, checkpoint),
                         identity + "/unchanged specialist proposals")
            base, admitted = audit_preparation(plan["base"], layout, phase, native, prior, home, resolutions, checks)
            checks.equal(plan["fallback"]["proposals"], fallback_proposals(layout, libraries, base, admitted),
                         identity + "/exact conditional fallback proposals")
            _, fallback = audit_preparation(plan["fallback"], layout, phase, native, prior, home, resolutions, checks)
            expected = [dict(id=row["id"], **ranker.predict(row)) for row in admitted + fallback]
            checks.equal(site["predictions"], expected, identity + "/reproduced pre-release flight predictions")
    return plans


def audit(args):
    """Verify the comparison without starting any simulator or GPU workload."""
    from . import intergrid_system_study as runner

    study, parent = args.output, args.parent_study
    spec, report = read(study / "contract.json"), read(study / "report.json")
    parent_spec, parent_binding, _ = parent_inputs(parent, args.parent_audit)
    checks = Checks()
    fingerprint(spec, checks, "Frozen system comparison")
    checks.equal((spec["version"], report["status"], report["phase"]), (runner.VERSION, "completed", "completed"),
                 "Completed study uses the frozen protocol")
    checks.equal((spec["parent"], spec["identity"], spec["scene"]),
                 (parent_binding, parent_spec["identity"], parent_spec["scene"]), "Exact audited native scene")
    checks.equal((spec["train_positions"], spec["test_positions"], spec["physics_substeps"], spec["label_version"],
                  spec["split_seeds"]), (runner.POSITIONS, runner.POSITIONS, 64, LABEL_VERSION,
                                         [runner.TRAIN_SEED, runner.TEST_SEED]), "Frozen split sizes and physics")
    checks.equal((spec["methods"], spec["routes"], spec["attempts_per_method_per_position"],
                  spec["candidate_budget_per_route"], spec["fallback_budget_per_route"], spec["train_max_per_site"]),
                 (list(METHODS), list(ROUTES), 1, BASE_PER_ROUTE, FALLBACK_PER_ROUTE, runner.TRAIN_MAX_PER_SITE),
                 "Frozen planner methods and candidate budgets")
    checks.require(type(spec["gpu_cap_s"]) is int and 0 < report["gpu_lease_s"] <= spec["gpu_cap_s"] <= runner.GPU_CAP_S,
                   "Completed within the GPU lease budget")
    source = Path(__file__).resolve().parent
    for name in ("intergrid_system_study.py", "intergrid_system_plan.py", "intergrid_system_learn.py"):
        checks.equal(spec["source_sha256"][f"sircus/mjlab/{name}"], sha(source / name), name + "/exact producer code")

    manifest = replay.check_manifest(study, checks)
    checks.equal(report["evidence_manifest_sha256"], sha(study / "evidence-manifest.json"), "Complete evidence binding")
    checks.equal(report["choices_sha256"], manifest["choices.json"], "Report binds choices bytes")
    frozen = spec["frozen_inputs"]
    checks.require({"layouts.json", "training-prior.json", "route-models.json", "route-history.json"} <= frozen.keys(),
                   "The contract freezes spatial splits, geometry priors and candidate sources")
    checks.equal(frozen, {name: manifest[name] for name in frozen}, "Original input digests are unchanged")

    checkpoint, original, additions, previous, binding, history = runner.inputs(args)
    checks.equal({key: spec[key] for key in binding}, binding, "Audited baseline and source evidence")
    checks.equal(manifest["route-models.json"], binding["selector_study"]["checkpoint_sha256"],
                 "Exact prior specialist checkpoint")
    checks.equal(manifest["training-prior.json"], binding["model_study"]["geometry_prior"], "Exact prior geometry")
    checks.equal(read(study / "route-history.json"), history, "Exact prior paired route outcomes")
    libraries, _ = verify_library(original, additions, checks)
    splits = verify_layouts(study, parent, args.history_root, spec["scene"], args.assets, previous, checks)
    ranker = _ranker(args, spec["scene"], binding)
    prior, home = read(study / "training-prior.json"), read(parent / "setup-home.json")
    plans = verify_plans(study, splits, libraries, checkpoint, ranker, prior, home, checks)

    cases = [runner.training_requests(site, ranker) for site in plans["train"]]
    checks.equal(read(study / "train_cases.json"), cases, "Exact frozen development requests and order")
    expected = {"train": [row for case in cases for row in case["requests"]]}
    for case in cases:
        checks.require(len(case["requests"]) <= runner.TRAIN_MAX_PER_SITE,
                       case["layout"]["id"] + "/bounded releases per development position")
    checks.equal(report["train_admitted"], len(expected["train"]), "Every admitted development drop accounted for")
    predictions = runner.training_predictions(cases, plans["train"])

    learning = read(study / "learning.json")
    slots = [slot for site in plans["test"] for slot in runner.test_requests(site, history, ranker, learning["checkpoint"])]
    choices = read(study / "choices.json")
    checks.equal(choices, dict(learning_sha256=manifest["learning.json"], slots=slots),
                 "Exact frozen model binding and reproducible test commands")
    expected["test"] = [slot["request"] for slot in slots if slot["request"] is not None]
    advanced = learning["checkpoint"] is not None
    methods = (*METHODS, runner.LEARNED) if advanced else METHODS
    checks.equal([(slot["context_id"], slot["method"]) for slot in slots],
                 [(layout["id"], method) for layout in splits["test"] for method in methods],
                 "One slot for each method and held-out position")
    checks.equal((report["test_slots"], report["test_admitted"]), (len(slots), len(expected["test"])),
                 "All frozen slots and admitted eval drops accounted for")
    checks.equal(len({row["id"] for row in expected["train"] + expected["test"]}),
                 len(expected["train"]) + len(expected["test"]), "Development and evaluation request IDs are unique")

    measured, controls = audit_packets(study, parent, args.assets, spec, report, manifest,
                                        expected, prior, home, checks)
    verify_realized_routes(study, measured, checks)
    train_records = {name: row for name, (row, _) in measured["train"].items()}
    checks.equal(learning, fit_ranker(cases, predictions, train_records),
                 "Reproduced site-held-out comparison, advancement decision and exact frozen weights")
    checks.equal((report["validation"], report["learned_advance"], report["prepared"]),
                 (learning["validation"], advanced, dict(train=runner.POSITIONS, test=runner.POSITIONS)),
                 "Reported advancement and preparation coverage match the frozen evidence")
    test_records = {name: row for name, (row, _) in measured["test"].items()}
    comparison = summarize(splits["test"], slots, test_records, plans["test"], methods)
    checks.equal(report["comparison"], comparison, "Capture and setup results from independently rescored traces")
    return dict(version=VERSION, status="passed", checks_count=checks.count,
                train_releases=len(train_records), test_releases=len(test_records), control_releases=controls,
                comparison=comparison, learning=learning["validation"], report_sha256=sha(study / "report.json"),
                learning_sha256=manifest["learning.json"], choices_sha256=manifest["choices.json"],
                evidence_manifest_sha256=report["evidence_manifest_sha256"])
