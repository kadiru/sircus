"""Audit the fixed-center case-33 tilt sweep and rescore every repeat on CPU."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path

from . import replay650_verify as replay
from .benchmark_verify import Checks, contained, digest, fingerprint, read, sha
from .cell1050_heading_native import BRANCH_RULE, heading_targets
from .cell1050_rebound_study import parent_inputs
from .fixed35_broad_native import reach_models, wrist_bound
from .intergrid_joint_branch_audit import verify_repeat_boundary
from .intergrid_proposer_audit import audit_packets
from .intergrid_route_audit import verify_realized_routes
from .intergrid_route_fallback_audit import _released_outcome
from .intergrid_route_labels import LABEL_VERSION
from .intergrid_route_selector_plan import STANDARD_CAPTURES
from .intergrid_setup_drop_audit import REPEATS, repeats_for, verify_result, verify_setup_proofs
from .sweep_worker import atomic


VERSION = "sircus-intergrid-case33-tilt-independent-audit-v1"
STUDY_VERSION = "sircus-intergrid-case33-tilt-v1"
SEED_ID = "intergrid-route-fallback-test-33-table-14-b0-joint-seed-2026111502-b0"
SOURCE_AUDIT_SHA256 = "c59380b20065798f815204608845f2671ee3d2a368a62b08cacc6d07f172e029"
TILTS = (25, 28, 31)
FIXED_FIELDS = ("bucket_xy_m", "context_id", "board_heading_offset_deg", "board_twist_deg",
                "intended_route", "phase", "role", "method", "stratum")


def reachable_grid(scene, models, seed, tilt, *, higher_than_m=0.):
    """Apply the arm wrist-sphere tests directly to the entire registered grid."""
    values = seed["values"][:3]
    heading, twist = seed["board_heading_offset_deg"], seed["board_twist_deg"]
    sample = heading_targets(scene, seed["bucket_xy_m"], [*values, tilt, .05], heading, twist)
    board = wrist_bound(models["bouncer"], sample["board_flange_local"])
    grid = []
    if board["passed"]:
        for gap_mm in range(650, 49, -5):
            gap_m = gap_mm / 1000
            if gap_m <= higher_than_m:
                break
            targets = heading_targets(scene, seed["bucket_xy_m"], [*values, tilt, gap_m], heading, twist)
            if wrist_bound(models["dropper"], targets["dropper_flange_world"])["passed"]:
                grid.append(gap_mm)
    return grid, 1000 * board["outside_by_m"]


def verify_tilt_trials(case, seed, scene, models, checks, *, relief=False):
    tilt = case["tilt_deg"]
    label = "Higher release at the original tilt" if relief else f"{tilt} degree sweep"
    layout_id, variant = ("case33-release-up", "release_up") if relief else (f"case33-tilt-{tilt}", f"tilt{tilt}")
    checks.equal(case["layout"]["id"], layout_id, label + "/layout identity")
    checks.equal(case["layout"]["bucket_xy_m"], seed["bucket_xy_m"], label + "/fixed bucket")
    grid, outside_mm = reachable_grid(scene, models, seed, tilt, higher_than_m=seed["values"][4] if relief else 0.)
    checks.equal(case["gap_grid_mm"], grid, label + "/reconstructed 5 mm reach grid")
    checks.numeric(case["board_wrist_outside_mm"], outside_mm, label + "/reconstructed board reach margin", 1e-9)
    trials, requests = case["trials"], case["requests"]
    tried = [trial["gap_mm"] for trial in trials]
    checks.equal(tried, grid[:len(tried)], label + "/descending grid with no skipped setting")
    checks.require(len(requests) <= BRANCH_RULE["maximum_admitted_branches"], label + "/admitted branch budget")
    if requests:
        checks.require(bool(trials), label + "/selected a tested gap")
        checks.equal(case["selected_gap_mm"], tried[-1], label + "/highest admitted gap")
    else:
        checks.equal((tried, case["selected_gap_mm"]), (grid, None), label + "/exhausted without an admission")

    for index, trial in enumerate(trials):
        proposal, record = trial["proposal"], trial["record"]
        name = f"{label}/{trial['gap_mm']} mm"
        expected_values = [*seed["values"][:3], tilt, trial["gap_mm"] / 1000]
        checks.equal(tuple(proposal[key] for key in FIXED_FIELDS), tuple(seed[key] for key in FIXED_FIELDS),
                     name + "/fixed requested settings")
        checks.equal(proposal["values"], expected_values, name + "/only tilt and gap change")
        checks.equal((proposal["seed_q10"], proposal["seed_q14"], proposal["source_request_id"]),
                     (seed["selected_q10"], seed["selected_q14"], SEED_ID), name + "/archived IK seed")
        checks.equal(proposal["variant"], variant, name + "/planned intervention")
        checks.equal((record["proposal_id"], record["proposal_sha256"]),
                     (proposal["id"], digest(proposal)), name + "/exact proposal binding")
        scheduled = requests if index == len(trials) - 1 else []
        checks.equal((record["admitted"], record["admitted_request_ids"]),
                     (len(scheduled), [row["id"] for row in scheduled]), name + "/one audited admission decision")
        checks.require(record["reason"] is None if scheduled else bool(record["reason"]),
                       name + "/admission or explicit refusal")
        for request in scheduled:
            checks.equal(tuple(request[key] for key in FIXED_FIELDS), tuple(seed[key] for key in FIXED_FIELDS),
                         request["id"] + "/fixed tested settings")
            checks.equal((request["proposal_id"], request["variant"], request["source_request_id"]),
                         (proposal["id"], variant, SEED_ID), request["id"] + "/selected proposal")
            checks.equal((request["values"], request["command_values"]),
                         (expected_values, expected_values), request["id"] + "/exact requested geometry")


def verify_sweep(plan, source, receipt, output, scene, assets, checks):
    checks.equal(sha(receipt), SOURCE_AUDIT_SHA256, "Published joint branch audit receipt")
    source_audit = read(receipt)
    checks.equal(source_audit["status"], "passed", "Audited clear source study")
    source_manifest_file = source / "evidence-manifest.json"
    checks.equal(sha(source_manifest_file), source_audit["evidence_manifest_sha256"], "Archived evidence manifest")
    source_manifest = read(source_manifest_file)
    source_plan = source / "drop-plan.json"
    checks.equal(sha(source_plan), source_manifest["drop-plan.json"], "Archived drop plan")
    seeds = [row for case in read(source_plan)["cases"] for row in case["requests"] if row["id"] == SEED_ID]
    checks.equal(len(seeds), 1, "One literal clear seed")
    seed = seeds[0]
    checks.equal([round(value * 1000, 1) for value in seed["bucket_xy_m"]], [1028.9, -19.0], "Target bucket")

    baseline, *cases = plan["cases"]
    checks.equal(baseline["layout"]["id"], "case33-tilt-baseline", "Exact baseline layout identity")
    checks.equal(baseline["layout"]["bucket_xy_m"], seed["bucket_xy_m"], "Baseline fixed bucket")
    checks.equal(baseline["requests"], [dict(seed, variant="baseline", source_request_id=SEED_ID)],
                 "Exact archived clear setup replays as the baseline")
    sweeps, relief = cases[:len(TILTS)], cases[len(TILTS):]
    checks.equal([case["tilt_deg"] for case in sweeps], list(TILTS), "All three tilts in registered order")

    models = reach_models(str(Path(assets).resolve()))
    for case in sweeps:
        verify_tilt_trials(case, seed, scene, models, checks)
    board_blocks_all = all(case["board_wrist_outside_mm"] > 0 for case in sweeps)
    checks.equal(len(relief), int(board_blocks_all), "Higher release only when the board cannot reach all requested tilts")
    if relief:
        checks.equal(relief[0]["tilt_deg"], seed["values"][3], "Higher-release control keeps the original tilt")
        verify_tilt_trials(relief[0], seed, scene, models, checks, relief=True)
    for case in plan["cases"]:
        for request in case["requests"]:
            proof = contained(output / "native-heading", request["prepared_connection_file"])
            checks.equal(sha(proof), request["prepared_connection_sha256"], request["id"] + "/saved path proof")
            targets = heading_targets(scene, request["bucket_xy_m"], request["values"],
                                      request["board_heading_offset_deg"], request["board_twist_deg"])
            recorded = read(proof)["targets"]
            checks.equal(recorded.keys(), targets.keys(), request["id"] + "/complete commanded targets")
            for field, target in targets.items():
                checks.numeric(recorded[field], target, request["id"] + "/" + field, 1e-12)


def audit(args):
    from .intergrid_case33_tilt_plan import load_tilt_plan

    study, parent = args.output, args.parent_study
    checks = Checks()
    spec, report = read(study / "contract.json"), read(study / "report.json")
    parent_spec, parent_binding, _ = parent_inputs(parent, args.parent_audit)
    fingerprint(spec, checks, "Frozen tilt study contract")
    checks.equal((spec["version"], report["status"], report["phase"]),
                 (STUDY_VERSION, "completed", "completed"), "Completed tilt study")
    checks.equal((spec["parent"], spec["identity"], spec["scene"], spec["physics_substeps"],
                  spec["label_version"], spec["repeat_count"]),
                 (parent_binding, parent_spec["identity"], parent_spec["scene"], 64, LABEL_VERSION, REPEATS),
                 "Original native64 physics and scoring with three repeats")
    plan, binding = load_tilt_plan(args.preparation, args.source_study, args.source_audit, parent, args.parent_audit)
    checks.equal(spec["tilt_source"], binding, "Exact source and frozen CPU preparation")
    checks.equal(binding["parent"], parent_binding, "Tilt results use the same original controls")
    manifest = replay.check_manifest(study, checks)
    checks.equal(report["evidence_manifest_sha256"], sha(study / "evidence-manifest.json"),
                 "Report binds complete evidence")
    checks.equal(spec["frozen_inputs"], {name: manifest[name] for name in ("drop-plan.json", "training-prior.json")},
                 "Exact frozen plan and geometry prior")
    checks.equal(manifest["training-prior.json"], binding["training_prior_sha256"], "Archived geometry prior")
    checks.equal(read(study / "drop-plan.json"), plan, "Every frozen tilt request was scheduled")
    verify_sweep(plan, args.source_study, args.source_audit, study, spec["scene"], args.assets, checks)

    initial = [request for case in plan["cases"] for request in case["requests"]]
    repeat_plan = read(study / "repeat-plan.json")
    repeats = repeat_plan["requests"]
    checks.require(isinstance(repeats, list) and len(repeats) <= len(initial) * REPEATS, "Bounded repeat queue")
    checks.equal(len({row["id"] for row in initial + repeats}), len(initial) + len(repeats), "Unique native IDs")
    proofs = verify_setup_proofs(initial, binding["proof_manifest"], manifest, checks)
    checks.equal((report["test_requests"], report["repeat_requests"], report["repeat_releases"]),
                 (len(initial), len(repeats), len(repeats)), "Exact first-pass and repeat counts")
    checks.equal(report["setup_proofs"], proofs, "Exact certified arm path count")
    checks.equal(report["repeat_plan_sha256"], manifest["repeat-plan.json"], "Exact repeat plan hash")
    prior, home = read(study / "training-prior.json"), read(parent / "setup-home.json")
    verify_repeat_boundary(study, len(initial), checks)
    measured, controls = audit_packets(study, parent, args.assets, spec, report, manifest,
                                       {"test": initial + repeats}, prior, home, checks)
    verify_realized_routes(study, measured, checks)
    outcomes = {}
    for request in initial + repeats:
        row, native_outcome = measured["test"][request["id"]]
        checks.equal(row["intended_route"], request["intended_route"], request["id"] + "/intended route")
        outcomes[request["id"]] = _released_outcome(row, native_outcome)
    expected = [repeat for request in initial if outcomes[request["id"]] in STANDARD_CAPTURES
                for repeat in repeats_for(request)]
    checks.equal(repeat_plan, {"requests": expected}, "Exactly three replays for every first-pass capture")
    result = verify_result(plan, outcomes, report, checks)
    first_pass_by_variant = {}
    for case in plan["cases"]:
        requests = case["requests"]
        label = case["layout"]["id"]
        rows = [measured["test"][request["id"]][0] for request in requests]
        first_pass_by_variant[label] = dict(
            selected_gap_mm=case.get("selected_gap_mm"),
            board_wrist_outside_mm=case.get("board_wrist_outside_mm"),
            attempted_gaps=len(case.get("trials", [])),
            outcomes=dict(Counter(outcomes[request["id"]] for request in requests)),
            robot_contacts=sum(row["experimental_route"]["robot_contact_steps"] > 0 for row in rows),
        )
    return dict(version=VERSION, status="passed", checks_count=checks.count, first_pass_releases=len(initial),
                repeat_releases=len(repeats), control_releases=controls, first_pass_by_variant=first_pass_by_variant,
                report_sha256=sha(study / "report.json"), repeat_plan_sha256=manifest["repeat-plan.json"],
                evidence_manifest_sha256=report["evidence_manifest_sha256"], result=result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("parent-study", "parent-audit", "preparation", "source-study", "source-audit",
                 "assets", "output", "audit"):
        parser.add_argument("--" + name, required=True, type=Path)
    args = parser.parse_args()
    if args.audit.exists() or args.audit.resolve().is_relative_to(args.output.resolve()):
        parser.error("Write a fresh audit outside the frozen evidence directory")
    try:
        result = audit(args)
    except Exception as error:
        result = dict(version=VERSION, status="failed", error=f"{type(error).__name__}: {error}")
    atomic(args.audit, result)
    print(json.dumps(result), flush=True)
    if result["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
