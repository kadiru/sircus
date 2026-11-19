"""Verify the fixed-pose branch drops and rescore every confirmation on CPU."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path

from . import replay650_verify as replay
from .benchmark_verify import Checks, contained, fingerprint, read, sha
from .cell1050_rebound_study import parent_inputs
from .intergrid_proposer_audit import audit_packets
from .intergrid_route_audit import verify_realized_routes
from .intergrid_route_fallback_audit import _released_outcome
from .intergrid_route_labels import LABEL_VERSION
from .intergrid_route_selector_plan import STANDARD_CAPTURES
from .intergrid_setup_drop_audit import REPEATS, repeats_for, verify_result, verify_setup_proofs
from .sweep_worker import atomic


VERSION = "sircus-intergrid-joint-branch-independent-audit-v1"
STUDY_VERSION = "sircus-intergrid-joint-branch-v1"
SOURCE_CASE = "intergrid-route-fallback-test-33"
POSE_FIELDS = ("context_id", "bucket_xy_m", "values", "command_values", "board_heading_offset_deg",
               "board_twist_deg", "intended_route")


def verify_fixed_poses(plan, source, output, checks):
    source_cases = read(source / "drop-plan.json")["cases"]
    selected = [case for case in source_cases if case["layout"]["id"] == SOURCE_CASE]
    checks.equal(len(selected), 1, "One archived layout at the target bucket coordinates")
    original = selected[0]["requests"]
    checks.equal(len(original), 4, "All four previously blocked shots remain as baseline requests")
    checks.equal([case["layout"]["id"] for case in plan["cases"]], [row["id"] for row in original],
                 "Branch test groups follow the exact original request order")
    for case, parent in zip(plan["cases"], original, strict=True):
        parent_id = parent["id"]
        checks.equal(case["layout"]["bucket_xy_m"], parent["bucket_xy_m"], parent_id + "/fixed bucket")
        baseline, *alternates = case["requests"]
        checks.equal(baseline, dict(parent, variant="baseline", source_request_id=parent_id),
                     parent_id + "/exact archived baseline command")
        parent_proof = contained(source / "native-heading", parent["prepared_connection_file"])
        checks.equal(sha(parent_proof), parent["prepared_connection_sha256"], parent_id + "/source pose proof bytes")
        expected_targets = read(parent_proof)["targets"]
        previous = [parent["selected_q10"] + parent["selected_q14"]]
        for request in alternates:
            label = request["id"]
            checks.equal((request["source_request_id"], request["variant"], request["phase"], request["role"]),
                         (parent_id, "alternate", "test", "test"), label + "/belongs to the parent pose")
            checks.equal(tuple(request[key] for key in POSE_FIELDS), tuple(parent[key] for key in POSE_FIELDS),
                         label + "/same board pose and release point")
            joints = request["selected_q10"] + request["selected_q14"]
            checks.require(len(joints) == 12 and all(math.isfinite(value) for value in joints),
                           label + "/finite six-joint settings for both arms")
            checks.require(all(
                max(abs(math.remainder(a - b, math.tau)) for a, b in zip(joints, seen, strict=True)) >= .05 - 1e-12
                for seen in previous
            ), label + "/materially different joint branch")
            previous.append(joints)
            proof = contained(output / "native-heading", request["prepared_connection_file"])
            checks.equal(sha(proof), request["prepared_connection_sha256"], label + "/saved branch path proof")
            checks.equal(read(proof)["targets"], expected_targets, label + "/exact commanded flange and release targets")


def verify_repeat_boundary(study, initial_count, checks):
    test_drops = 0
    for path in sorted((study / "requests").glob("*.json")):
        packet = read(path)
        if packet["phase"] == "test":
            end = test_drops + len(packet["rows"])
            checks.require(not test_drops < initial_count < end,
                           "First-pass and confirmation drops use separate native packets")
            test_drops = end


def audit(args):
    from .intergrid_joint_branch_search import load_branch_plan

    study, parent = args.output, args.parent_study
    checks = Checks()
    spec, report = read(study / "contract.json"), read(study / "report.json")
    parent_spec, parent_binding, _ = parent_inputs(parent, args.parent_audit)
    fingerprint(spec, checks, "Frozen joint branch contract")
    checks.equal((spec["version"], report["status"], report["phase"]),
                 (STUDY_VERSION, "completed", "completed"), "Completed joint branch study")
    checks.equal((spec["parent"], spec["identity"], spec["scene"], spec["physics_substeps"],
                  spec["label_version"], spec["repeat_count"]),
                 (parent_binding, parent_spec["identity"], parent_spec["scene"], 64, LABEL_VERSION, REPEATS),
                 "Original native64 physics and scoring with three confirmation drops")
    plan, binding = load_branch_plan(args.preparation, args.source_study, args.source_audit, parent, args.parent_audit)
    checks.equal(spec["branch_source"], binding, "Exact source study and frozen CPU preparations")
    checks.equal(binding["parent"], parent_binding, "Joint branch study uses the same original controls")
    manifest = replay.check_manifest(study, checks)
    checks.equal(report["evidence_manifest_sha256"], sha(study / "evidence-manifest.json"),
                 "Report binds complete frozen evidence")
    checks.equal(spec["frozen_inputs"], {name: manifest[name] for name in ("drop-plan.json", "training-prior.json")},
                 "Exact pre-release input hashes")
    checks.equal(manifest["training-prior.json"], binding["training_prior_sha256"],
                 "Exact prior geometry from the source study")
    checks.equal(read(study / "drop-plan.json"), plan, "All frozen prepared joint branches were scheduled")
    verify_fixed_poses(plan, args.source_study, study, checks)

    initial = [request for case in plan["cases"] for request in case["requests"]]
    repeat_plan = read(study / "repeat-plan.json")
    repeats = repeat_plan["requests"]
    checks.require(isinstance(repeats, list) and len(repeats) <= len(initial) * REPEATS,
                   "Bounded confirmation drop plan")
    checks.equal(len({row["id"] for row in initial + repeats}), len(initial) + len(repeats),
                 "Unique native drop identifiers")
    proofs = verify_setup_proofs(initial, binding["proof_manifest"], manifest, checks)
    checks.equal((report["test_requests"], report["repeat_requests"], report["repeat_releases"]),
                 (len(initial), len(repeats), len(repeats)), "Exact first-pass and confirmation counts")
    checks.equal(report["setup_proofs"], proofs, "Exact arm path and endpoint proof count")
    checks.equal(report["repeat_plan_sha256"], manifest["repeat-plan.json"], "Exact confirmation plan hash")
    prior, home = read(study / "training-prior.json"), read(parent / "setup-home.json")
    verify_repeat_boundary(study, len(initial), checks)
    measured, controls = audit_packets(study, parent, args.assets, spec, report, manifest,
                                       {"test": initial + repeats}, prior, home, checks)
    verify_realized_routes(study, measured, checks)
    outcomes = {}
    for request in initial + repeats:
        row, native_outcome = measured["test"][request["id"]]
        checks.equal(row["intended_route"], request["intended_route"], request["id"] + "/original intended route")
        outcomes[request["id"]] = _released_outcome(row, native_outcome)
    expected = [repeat for request in initial if outcomes[request["id"]] in STANDARD_CAPTURES
                for repeat in repeats_for(request)]
    checks.equal(repeat_plan, {"requests": expected}, "Every first-pass capture receives three exact repeats")
    result = verify_result(plan, outcomes, report, checks)
    by_variant = {variant: Counter() for variant in ("baseline", "alternate")}
    with_robot_contact = dict.fromkeys(by_variant, 0)
    for request in initial:
        variant, identifier = request["variant"], request["id"]
        by_variant[variant][outcomes[identifier]] += 1
        observation, _ = measured["test"][identifier]
        with_robot_contact[variant] += int(observation["experimental_route"]["robot_contact_steps"] > 0)
    return dict(
        version=VERSION, status="passed", checks_count=checks.count, first_pass_releases=len(initial),
        repeat_releases=len(repeats), control_releases=controls,
        first_pass_outcomes={variant: dict(counts) for variant, counts in by_variant.items()},
        first_pass_with_robot_contact=with_robot_contact,
        report_sha256=sha(study / "report.json"), repeat_plan_sha256=manifest["repeat-plan.json"],
        evidence_manifest_sha256=report["evidence_manifest_sha256"], result=result,
    )


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
