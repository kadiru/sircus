"""Verify the setup-recovery drops and every nominal repeat on CPU."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

from . import replay650_verify as replay
from .benchmark_verify import Checks, fingerprint, read, sha
from .cell1050_rebound_study import parent_inputs
from .intergrid_proposer_audit import audit_packets
from .intergrid_route_audit import verify_realized_routes
from .intergrid_route_fallback_audit import _released_outcome
from .intergrid_route_labels import LABEL_VERSION
from .intergrid_route_selector_plan import STANDARD_CAPTURES
from .intergrid_setup_drop_plan import load_setup_drop_plan
from .sweep_worker import atomic


VERSION = "sircus-intergrid-setup-drop-independent-audit-v1"
STUDY_VERSION = "sircus-intergrid-setup-drop-v1"
INITIAL_REQUESTS = 43
POSITIONS = 5
REPEATS = 3


def verify_setup_proofs(requests, source_proofs, manifest, checks):
    copied = {}
    for request in requests:
        for kind in ("context_resolution", "prepared_connection"):
            path = "native-heading/" + request[kind + "_file"]
            expected = request[kind + "_sha256"]
            checks.equal((source_proofs[path], manifest[path]), (expected, expected),
                         request["id"] + "/copied audited " + kind)
            copied[path] = expected
    return len(copied)


def repeats_for(req):
    return [dict(req, id=f"{req['id']}-repeat-{i:02d}") for i in range(1, REPEATS + 1)]


def verify_repeat_boundary(study, checks):
    test_drops = 0
    for path in sorted((study / "requests").glob("*.json")):
        packet = read(path)
        if packet["phase"] == "test":
            end = test_drops + len(packet["rows"])
            checks.require(not test_drops < INITIAL_REQUESTS < end,
                           "A submission packet must belong to either first-pass or confirmation drops")
            test_drops = end


def verify_result(plan, outcomes, report, checks):
    cases = []
    totals = Counter()
    for case in plan["cases"]:
        layout = case["layout"]
        counts = dict.fromkeys(("first_pass_drops", "first_pass_captures", "repeat_drops",
                               "repeat_captures", "confirmed_shots"), 0)
        shots = []
        for request in case["requests"]:
            name = request["id"]
            first = outcomes[name]
            accepted = first in STANDARD_CAPTURES
            repeats = [outcomes[repeat["id"]] for repeat in repeats_for(request)] if accepted else []
            captured = sum(outcome in STANDARD_CAPTURES for outcome in repeats)
            confirmed = captured == REPEATS
            counts["first_pass_drops"] += 1
            counts["first_pass_captures"] += int(accepted)
            counts["repeat_drops"] += len(repeats)
            counts["repeat_captures"] += captured
            counts["confirmed_shots"] += int(confirmed)
            shots.append(dict(id=name, intended_route=request["intended_route"], first_outcome=first,
                              repeat_outcomes=repeats, confirmed=confirmed))
        totals.update(counts)
        cases.append(dict(id=layout["id"], bucket_xy_m=layout["bucket_xy_m"], **counts, shots=shots))
    result = dict(positions=len(cases),
                  positions_with_capture=sum(case["first_pass_captures"] > 0 for case in cases),
                  positions_confirmed=sum(case["confirmed_shots"] > 0 for case in cases), **totals, cases=cases)
    checks.equal(report["result"], result, "Capture and confirmation results by exact recovered setup")
    return result


def audit(args):
    checks = Checks()
    study, parent = args.output, args.parent_study
    spec, report = read(study / "contract.json"), read(study / "report.json")
    parent_spec, parent_binding, _ = parent_inputs(parent, args.parent_audit)
    fingerprint(spec, checks, "Frozen setup drop contract")
    checks.equal((spec["version"], report["status"], report["phase"]),
                 (STUDY_VERSION, "completed", "completed"), "Completed setup drop study")
    checks.equal((spec["parent"], spec["identity"], spec["scene"], spec["physics_substeps"],
                  spec["label_version"], spec["repeat_count"]),
                 (parent_binding, parent_spec["identity"], parent_spec["scene"], 64, LABEL_VERSION, REPEATS),
                 "Original native64 physics and scoring with three confirmation drops")
    plan, binding = load_setup_drop_plan(args.setup_study, args.setup_audit, args.fallback_study, parent)
    checks.equal(spec["setup_source"], binding, "Exact audited pose recovery source")
    checks.equal(binding["parent"], parent_binding, "Pose recovery uses the same audited parent and controls")
    manifest = replay.check_manifest(study, checks)
    checks.equal(report["evidence_manifest_sha256"], sha(study / "evidence-manifest.json"),
                 "Report binds complete frozen evidence")
    checks.equal(spec["frozen_inputs"], {name: manifest[name] for name in ("drop-plan.json", "training-prior.json")},
                 "Exact pre-release input hashes")
    checks.equal(manifest["training-prior.json"], binding["training_prior_sha256"],
                 "Exact geometry prior of the audited adjustment")
    checks.equal(read(study / "drop-plan.json"), plan, "All and only the audited setups were scheduled")
    checks.equal(len({tuple(case["layout"]["bucket_xy_m"]) for case in plan["cases"]}), POSITIONS,
                 "Distinct recovery bucket positions")

    initial = [request for case in plan["cases"] for request in case["requests"]]
    repeat_plan = read(study / "repeat-plan.json")
    repeats = repeat_plan["requests"]
    checks.equal(len(initial), INITIAL_REQUESTS, "Exactly 43 admitted first-pass drops")
    checks.require(isinstance(repeats, list) and len(repeats) <= INITIAL_REQUESTS * REPEATS,
                   "Bounded confirmation drop plan")
    checks.equal(len({item["id"] for item in initial + repeats}), len(initial) + len(repeats),
                 "Unique initial and repeat identities")
    proofs = verify_setup_proofs(initial, binding["proof_manifest"], manifest, checks)
    checks.equal((report["test_requests"], report["repeat_requests"], report["repeat_releases"]),
                 (len(initial), len(repeats), len(repeats)), "Exact first-pass and confirmation counts")
    checks.equal(report["setup_proofs"], proofs, "Exact copied arm path and endpoint proof count")
    checks.equal(report["repeat_plan_sha256"], manifest["repeat-plan.json"], "Exact confirmation plan hash")
    prior, home = read(study / "training-prior.json"), read(parent / "setup-home.json")
    verify_repeat_boundary(study, checks)
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
    checks.equal(repeat_plan, {"requests": expected}, "Every accepted initial capture receives three exact repeats")
    result = verify_result(plan, outcomes, report, checks)
    return dict(version=VERSION, status="passed", checks_count=checks.count,
                first_pass_releases=len(initial), repeat_releases=len(repeats), control_releases=controls,
                report_sha256=sha(study / "report.json"), repeat_plan_sha256=manifest["repeat-plan.json"],
                evidence_manifest_sha256=report["evidence_manifest_sha256"], result=result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("parent-study", "parent-audit", "fallback-study", "setup-study", "setup-audit",
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
