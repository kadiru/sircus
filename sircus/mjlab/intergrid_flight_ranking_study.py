"""Compare a frozen pre-release flight ranker with direct-first on fresh positions."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

from . import replay650_verify as replay
from .benchmark_verify import Checks, digest, fingerprint, read, sha
from .cell1050_rebound_study import parent_inputs
from .intergrid_candidate_replay_audit import STUDY_VERSION as REPLAY_VERSION
from .intergrid_candidate_replay_audit import VERSION as REPLAY_AUDIT_VERSION
from .intergrid_candidate_replay_plan import command_key
from .intergrid_layouts import MIN_SEPARATION_M, _placement_geometry
from .intergrid_library_audit import verify_library
from .intergrid_matched_route_plan import POSITION_COUNT
from .intergrid_matched_route_plan import make_layouts as make_matched_layouts
from .intergrid_matched_route_plan import sample_layouts as sample_matched_layouts
from .intergrid_proposer_audit import audit_packets, audit_preparation
from .intergrid_route_audit import verify_history_files, verify_realized_routes, verify_separation
from .intergrid_route_fallback_audit import VERSION as FALLBACK_AUDIT_VERSION
from .intergrid_route_fallback_inputs import load_fallback_inputs
from .intergrid_route_fallback_study import POSITIONS as FALLBACK_POSITIONS
from .intergrid_route_fallback_study import VERSION as FALLBACK_VERSION
from .intergrid_route_labels import LABEL_VERSION
from .intergrid_route_plan import ROUTES
from .intergrid_route_refit_inputs import _audited_source
from .intergrid_route_selector_audit import verify_proposals
from .intergrid_route_selector_plan import STANDARD_CAPTURES, _account, select_fallback_requests
from .intergrid_route_selector_study import SOURCE_NAMES, run_comparison
from .intergrid_route_study import PER_ROUTE_TEST
from .sweep_worker import atomic

VERSION = "sircus-intergrid-flight-ranking-study-v1"
PLAN_VERSION = "sircus-intergrid-flight-ranking-plan-v1"
SCREEN_VERSION = "sircus-intergrid-flight-ranking-screen-v1"
AUDIT_VERSION = "sircus-intergrid-flight-ranking-independent-audit-v1"
SPLIT_SEED = 2026112301
METHODS = ("direct_first", "physics_ranker")
EXTRA_SOURCES = ("selector", "fallback", "replay")


def load_comparison_inputs(history_root, *sources, screen, screen_hash, parent, parent_audit):
    """Bind the development result to its audited replay and the original candidate sources."""
    *selector_sources, fallback, fallback_audit, replay_study, replay_audit = sources
    if (screen.get("version") != SCREEN_VERSION or screen.get("advance") is not True
            or not isinstance(screen.get("rule"), dict) or not screen["rule"]
            or screen.get("rule_sha256") != digest(screen["rule"])):
        raise ValueError("The replay screen must pass before starting a fresh comparison")

    replay_report, _, replay_receipt, replay_binding, replay_contract = _audited_source(
        replay_study, replay_audit, REPLAY_VERSION, ("replay-plan.json", "training-prior.json", "rows.jsonl"),
    )
    if (replay_receipt["version"] != REPLAY_AUDIT_VERSION
            or replay_receipt["diagnosis"] != replay_report["diagnosis"]
            or replay_receipt["replay_plan_sha256"] != replay_binding["replay-plan.json"]
            or screen.get("replay_binding") != {
                "evidence_manifest_sha256": replay_binding["evidence-manifest.json"],
                "report_sha256": replay_binding["report.json"],
                "audit_sha256": replay_binding["audit"],
            }):
        raise ValueError("The saved ranking rule does not bind the audited replay evidence")

    fallback_files = ("layouts.json", "choices.json", "route-models.json", "training-prior.json",
                      "route-history.json", "rows.jsonl")
    fallback_report, _, fallback_receipt, fallback_binding, fallback_contract = _audited_source(
        fallback, fallback_audit, FALLBACK_VERSION, fallback_files,
    )
    if (fallback_receipt["version"] != FALLBACK_AUDIT_VERSION
            or fallback_receipt["comparison"] != fallback_report["comparison"]
            or any(replay_contract["fallback_study"][key] != value for key, value in fallback_binding.items())):
        raise ValueError("The fallback archive does not match the replay source")

    parent_spec, parent_binding, _ = parent_inputs(parent, parent_audit)
    for contract in (fallback_contract, replay_contract):
        if (contract["parent"], contract["identity"], contract["scene"]) != (
                parent_binding, parent_spec["identity"], parent_spec["scene"]):
            raise ValueError("Study sources do not share the audited native scene")

    checkpoint, original, new, previous, binding, history = load_fallback_inputs(history_root, *selector_sources)
    if (any(fallback_contract[key] != value for key, value in binding.items())
            or fallback_binding["route-models.json"] != binding["selector_study"]["checkpoint_sha256"]
            or fallback_binding["training-prior.json"] != replay_binding["training-prior.json"]):
        raise ValueError("The fallback archive does not bind the same frozen candidate sources")

    archive = read(fallback / "layouts.json")
    prior_layouts = archive["layouts"]
    fallback_xy = {tuple(layout["bucket_xy_m"]) for layout in prior_layouts}
    replay_plan = read(replay_study / "replay-plan.json")
    cases = replay_plan["cases"]
    gate = {key: screen.get(key) for key in ("captures", "uniform_expected_captures", "positions", "attempts")}
    captures, uniform = gate["captures"], gate["uniform_expected_captures"]
    if (type(captures) is not int or type(uniform) not in (int, float)
            or not math.isfinite(uniform) or not 0 <= uniform < captures <= len(cases)
            or (gate["positions"], gate["attempts"]) != (len(cases), sum(len(case["requests"]) for case in cases))):
        raise ValueError("The frozen ranker must beat uniform selection on the complete replay menu")
    replay_xy = {
        tuple(case["layout"]["bucket_xy_m"])
        for name in ("cases", "setup_failures") for case in replay_plan[name]
    }
    if (len(prior_layouts) != FALLBACK_POSITIONS or len(fallback_xy) != FALLBACK_POSITIONS
            or not replay_xy.issubset(fallback_xy)
            or digest(prior_layouts) != archive["provenance"]["layouts_sha256"]):
        raise ValueError("The archived fallback positions must cover every replay development position")

    binding.update(fallback_study=fallback_binding, replay_study=replay_binding,
                   flight_ranking={"rule": screen["rule"], "rule_sha256": screen["rule_sha256"],
                                   "screen_sha256": screen_hash, "replay_gate": gate})
    return checkpoint, original, new, previous + prior_layouts, binding, history


def _layout_ids(layouts):
    renamed = []
    for index, layout in enumerate(layouts):
        identity = f"intergrid-flight-ranking-test-{index:02d}"
        renamed.append(dict(layout, id=identity, context_id=identity))
    return renamed


def fresh_layouts(history_root, grid_path, scene, assets, previous, seed):
    layouts, provenance = make_matched_layouts(
        history_root, read(grid_path), scene=scene, assets=assets, previous_layouts=previous, seed=seed,
    )
    layouts = _layout_ids(layouts)
    provenance.update(version=PLAN_VERSION, layouts_sha256=digest(layouts))
    return layouts, provenance


def select_requests(layout, proposals, admitted, history, ranker):
    """Give both methods the same admitted candidates and one selection each."""
    baseline, _ = select_fallback_requests(layout, proposals, admitted, history)
    known = {row["id"]: digest(row) for row in admitted}
    ordered = ranker.order(admitted)
    if len(ordered) != len(known) or {row["id"]: digest(row) for row in ordered} != known:
        raise ValueError("The frozen ranker must order all and only admitted commands")
    selected = ordered[0] if ordered else None
    method = "physics_ranker"
    identifier = f"{layout['id']}-{method}"
    ranked = dict(
        id=identifier, method=method, context_id=layout["id"], bucket_xy_m=layout["bucket_xy_m"],
        stratum=layout["stratum"], available_routes=baseline["available_routes"],
        intended_route=selected["intended_route"] if selected else None,
        admitted_distinct_commands=len({command_key(row) for row in admitted}),
        ranked_candidate_ids=[row["id"] for row in ordered],
        prediction=ranker.predict(selected) if selected else None,
        request=dict(selected, id=identifier, method=method, stratum=layout["stratum"]) if selected else None,
    )
    return [baseline, ranked]


def summarize(layouts, slots, records):
    """Account for every refusal and distinguish equal, same-route and switched commands."""
    identities, methods, paired_slots = _account(layouts, slots, records, METHODS)
    outcomes = {(True, True): "both", (True, False): "direct_first_only",
                (False, True): "physics_ranker_only", (False, False): "neither"}
    paired = dict.fromkeys(outcomes.values(), 0)
    changed = {name: dict.fromkeys(outcomes.values(), 0) for name in ("same_route", "different_route")}
    same_command = dict(attempts=0, capture_disagreements=0)
    setup_refusals = 0
    for identity in identities:
        (baseline, first), (ranked, second) = (paired_slots[identity, name] for name in METHODS)
        first_req, second_req = baseline["request"], ranked["request"]
        first_hit, second_hit = first in STANDARD_CAPTURES, second in STANDARD_CAPTURES
        result = outcomes[first_hit, second_hit]
        paired[result] += 1
        if (first_req is None) != (second_req is None) or baseline["available_routes"] != ranked["available_routes"]:
            raise ValueError("Both methods must use the same setup availability")
        if first_req is None:
            setup_refusals += 1
        elif command_key(first_req) == command_key(second_req):
            same_command["attempts"] += 1
            same_command["capture_disagreements"] += int(first_hit != second_hit)
        else:
            kind = "same_route" if baseline["intended_route"] == ranked["intended_route"] else "different_route"
            changed[kind][result] += 1
    return dict(methods=methods, setup_refusals=setup_refusals, paired_capture=paired,
                same_command=same_command, different_commands=changed)


def _inputs(args):
    raw = args.rule.read_bytes()
    screen = json.loads(raw)
    screen_hash = hashlib.sha256(raw).hexdigest()
    source_args = [getattr(args, f"{name}_{kind}") for name in (*SOURCE_NAMES, *EXTRA_SOURCES)
                   for kind in ("study", "audit")]
    return load_comparison_inputs(args.history_root, *source_args, screen=screen, screen_hash=screen_hash,
                                  parent=args.parent_study, parent_audit=args.parent_audit)


def _ranker(args, scene, binding):
    from .intergrid_flight_ranker import FlightRanker

    calibration = read(args.parent_study / "calibration.json")["score_config"]
    return FlightRanker(scene, calibration, args.assets, binding["flight_ranking"]["rule"])


def run(args):
    inputs = _inputs(args)
    scene = read(args.parent_study / "contract.json")["scene"]
    ranker = _ranker(args, scene, inputs[4])
    run_comparison(
        args, version=VERSION, split_seeds=(SPLIT_SEED,), positions=POSITION_COUNT,
        methods=METHODS, load_inputs=lambda *_: inputs, make_layouts=fresh_layouts,
        select=lambda layout, proposals, admitted, history: select_requests(
            layout, proposals, admitted, history, ranker,
        ),
        summarize_outcomes=summarize, extra_source_names=EXTRA_SOURCES,
    )


def audit(args):
    """Rebuild the candidate choices and independently rescore the saved trajectories."""
    checks = Checks()
    study, parent = args.output, args.parent_study
    spec, report = read(study / "contract.json"), read(study / "report.json")
    parent_spec, parent_binding, _ = parent_inputs(parent, args.parent_audit)
    fingerprint(spec, checks, "Flight ranking comparison")
    checks.equal((spec["version"], report["status"], report["phase"]), (VERSION, "completed", "completed"),
                 "Completed frozen comparison")
    checks.equal((spec["parent"], spec["identity"], spec["scene"]),
                 (parent_binding, parent_spec["identity"], parent_spec["scene"]), "Audited native scene")
    checks.equal((spec["test_positions"], spec["attempts_per_method_per_position"], spec["methods"], spec["routes"],
                  spec["candidate_budget_per_route"], spec["physics_substeps"], spec["label_version"], spec["split_seed"]),
                 (POSITION_COUNT, 1, list(METHODS), list(ROUTES), PER_ROUTE_TEST, 64, LABEL_VERSION, SPLIT_SEED),
                 "One shot per method and position with matched candidate budgets")
    manifest = replay.check_manifest(study, checks)
    checks.equal(report["evidence_manifest_sha256"], sha(study / "evidence-manifest.json"), "All evidence bound")
    checks.equal(report["choices_sha256"], manifest["choices.json"], "Frozen choice bytes")
    checks.equal(spec["frozen_inputs"], {name: manifest[name] for name in
                 ("layouts.json", "training-prior.json", "route-models.json", "route-history.json")}, "Frozen inputs")

    checkpoint, original, additions, previous, binding, history = _inputs(args)
    checks.equal({key: spec[key] for key in binding}, binding, "Frozen rule and audited source evidence")
    checkpoint_sha = binding["selector_study"]["checkpoint_sha256"]
    checks.equal((report["checkpoint_sha256"], manifest["route-models.json"]), (checkpoint_sha, checkpoint_sha),
                 "Exact frozen specialist weights")
    checks.equal(manifest["training-prior.json"], binding["model_study"]["geometry_prior"], "Original arm prior")
    checks.equal(read(study / "route-history.json"), history, "Stored selection history")
    libraries, library_summary = verify_library(original, additions, checks)
    checks.equal(report["library"], library_summary, "Candidate library")
    checks.equal(report["selection_source_positions"], len(history), "Original paired history count")

    archive = read(study / "layouts.json")
    layouts, provenance = archive["layouts"], archive["provenance"]
    grid = read(parent / "grid.json")
    geometry = _placement_geometry(spec["scene"], args.assets)
    previous_xy = {tuple(row["bucket_xy_m"]) for row in previous}
    exclusions = [tuple(xy) for xy in provenance["excluded_coordinates_m"]]
    checks.equal((len(layouts), provenance["version"], provenance["seed"], provenance["minimum_separation_m"]),
                 (POSITION_COUNT, PLAN_VERSION, SPLIT_SEED, MIN_SEPARATION_M), "Fresh layout protocol")
    checks.equal(provenance["geometry"], geometry, "Compiled placement geometry")
    checks.equal(provenance["declared_previous_coordinates_m"], [list(xy) for xy in sorted(previous_xy)],
                 "Explicitly excluded earlier coordinates")
    checks.equal(exclusions, sorted(set(exclusions)), "Sorted distinct placement exclusions")
    historical = verify_history_files(provenance, args.history_root, study, checks)
    known = historical | previous_xy | {tuple(row["bucket_xy_m"]) for row in grid}
    checks.require(known.issubset(exclusions), "Every known position is excluded")
    checks.equal(provenance["excluded_coordinates_sha256"], digest(exclusions), "Placement exclusion digest")
    checks.equal(provenance["layouts_sha256"], digest(layouts), "Frozen position digest")
    checks.equal(layouts, _layout_ids(sample_matched_layouts(exclusions, geometry, grid, seed=SPLIT_SEED)),
                 "Recomputed 50 held-out bucket positions")
    verify_separation(layouts, [], exclusions, checks)

    prior, home = read(study / "training-prior.json"), read(parent / "setup-home.json")
    ranker = _ranker(args, spec["scene"], binding)
    slots, resolutions = [], {}
    checks.equal({path.stem for path in (study / "test").glob("*.json")}, {row["id"] for row in layouts},
                 "One candidate record per position")
    for layout in layouts:
        artifact = read(study / "test" / (layout["id"] + ".json"))
        verify_proposals(artifact, layout, libraries, checkpoint, checks)
        proposals, admitted = audit_preparation(artifact, layout, "test", study / "native-heading",
                                                prior, home, resolutions, checks)
        slots.extend(select_requests(layout, proposals, admitted, history, ranker))
    checks.equal(read(study / "choices.json"), dict(checkpoint_sha256=checkpoint_sha, slots=slots),
                 "Exact direct-first selections and all predicted ranks")
    requests = [slot["request"] for slot in slots if slot["request"] is not None]
    checks.equal((report["test_prepared"], report["test_slots"], report["test_admitted"], report["test_releases"]),
                 (len(layouts), len(slots), len(requests), len(requests)), "All matched attempt slots accounted for")
    measured, controls = audit_packets(study, parent, args.assets, spec, report, manifest,
                                        {"test": requests}, prior, home, checks)
    verify_realized_routes(study, measured, checks)
    expected = {key: row for key, (row, _) in measured["test"].items()}
    comparison = summarize(layouts, slots, expected)
    checks.equal(report["comparison"], comparison, "Capture rates and paired outcomes from rescored evidence")
    return dict(version=AUDIT_VERSION, status="passed", checks_count=checks.count, test_attempts=len(slots),
                test_releases=len(requests), control_releases=controls, comparison=comparison,
                report_sha256=sha(study / "report.json"), choices_sha256=manifest["choices.json"],
                evidence_manifest_sha256=report["evidence_manifest_sha256"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("history-root", "parent-study", "parent-audit", "reference", "assets", "gpu-lock", "output", "rule"):
        parser.add_argument("--" + name, type=Path, required=True)
    for source in (*SOURCE_NAMES, *EXTRA_SOURCES):
        for kind in ("study", "audit"):
            parser.add_argument(f"--{source}-{kind}", type=Path, required=True)
    parser.add_argument("--gpu-wait-s", type=int, default=7 * 3600)
    parser.add_argument("--audit", type=Path, help="Audit a completed output on CPU and write a new receipt")
    args = parser.parse_args()
    if args.audit is None:
        if args.output.exists() or args.gpu_wait_s <= 0:
            parser.error("Provide a fresh output directory and a positive GPU wait bound")
        run(args)
        return
    if args.audit.exists() or args.audit.resolve().is_relative_to(args.output.resolve()):
        parser.error("Write a fresh audit outside the frozen evidence directory")
    try:
        result = audit(args)
    except Exception as error:
        result = dict(version=AUDIT_VERSION, status="failed", error=f"{type(error).__name__}: {error}")
    atomic(args.audit, result)
    print(json.dumps(result), flush=True)
    if result["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
