"""Audit diagnostic replays of previously admitted bounce commands on CPU."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

from . import replay650_verify as replay
from .benchmark_verify import Checks, fingerprint, read, sha
from .cell1050_rebound_study import parent_inputs
from .intergrid_candidate_replay_plan import command_key, load_replay_plan
from .intergrid_proposer_audit import audit_packets, audit_preparation
from .intergrid_route_audit import verify_realized_routes
from .intergrid_route_fallback_audit import _released_outcome
from .intergrid_route_labels import LABEL_VERSION
from .intergrid_route_plan import ROUTES
from .parallel_capture_grid_audit import observed_rows
from .sweep_worker import atomic


VERSION = "sircus-intergrid-candidate-replay-independent-audit-v1"
STUDY_VERSION = "sircus-intergrid-candidate-replay-v1"
STANDARD_CAPTURES = {"direct", "table", "unclassified_capture"}


def verify_commands(fallback, plan, manifest, prior, home, checks):
    """Confirm requests are exact archived commands with archived pose and path proofs."""
    fallback_manifest = read(fallback / "evidence-manifest.json")
    slots = {(slot["context_id"], slot["method"]): slot for slot in read(fallback / "choices.json")["slots"]}
    prior_observed = {row["id"]: row for row in observed_rows(fallback / "rows.jsonl") if row["phase"] == "test"}
    resolutions, expected_cases, expected_setup = {}, [], []
    for layout in read(fallback / "layouts.json")["layouts"]:
        identity = layout["id"]
        selector = slots[identity, "route_selector"]
        if selector["request"] is None:
            artifact = read(fallback / "test" / (identity + ".json"))
            checks.equal(artifact["admitted"], [], identity + "/no archived feasible command")
            proposals, records = artifact["proposals"], artifact["records"]
            blocked = {route: Counter() for route in ROUTES}
            for proposal, record in zip(proposals, records, strict=True):
                blocked[proposal["intended_route"]][record["reason"]] += 1
            expected_setup.append(dict(
                layout=layout, reason="all_proposals_rejected" if proposals else "no_proposals",
                routes={route: dict(proposed=sum(reasons.values()), reasons=dict(reasons))
                        for route, reasons in blocked.items()},
            ))
            continue
        row = prior_observed[selector["id"]]
        if not row["route_success"] and not row["invalid_physics"] and not row["experimental_route"]["robot_assisted_capture"]:
            expected_cases.append(identity)
    checks.equal([case["layout"]["id"] for case in plan["cases"]], expected_cases,
                 "Exactly the prior selector's valid released misses, in archived order")
    checks.equal(plan["setup_failures"], expected_setup,
                 "Exactly the prior selector's setup failures, in archived order")

    requests, proof_paths = [], set()
    for case in plan["cases"]:
        layout = case["layout"]
        identity = layout["id"]
        selector = slots[identity, "route_selector"]
        direct = slots[identity, "direct_first"]
        checks.equal((case["prior_selector_id"], case["prior_direct_first_id"]),
                     (selector["id"], direct["id"]), identity + "/exact prior policy slots")
        artifact = read(fallback / "test" / (identity + ".json"))
        _, admitted = audit_preparation(artifact, layout, "test", fallback / "native-heading",
                                        prior, home, resolutions, checks)
        prior_selected = selector["request"]
        by_id = {row["id"]: row for row in admitted}
        original = by_id[f"{prior_selected['proposal_id']}-b0"]
        checks.equal(dict(original, id=selector["id"], method="route_selector", stratum=layout["stratum"]),
                     prior_selected, identity + "/prior selector command binds an admitted branch")
        direct_key = command_key(direct["request"]) if direct["request"] is not None else None
        direct_row = prior_observed.get(direct["id"])
        keys = set()
        expected_requests = []
        for admitted_command in (original, *admitted):
            key = command_key(admitted_command)
            if key in keys:
                continue
            keys.add(key)
            index = len(expected_requests)
            expected_requests.append(dict(
                admitted_command, id=f"{identity}-replay-{index:02d}",
                method="baseline_repeat" if index == 0 else "alternative",
                phase="test", role="test", stratum=layout["stratum"],
            ))
        checks.equal(case["requests"], expected_requests,
                     identity + "/exact admitted commands with one repeat and distinct alternatives")
        metadata = []
        for request in expected_requests:
            direct_match = command_key(request) == direct_key
            capture = bool(direct_row["route_success"] and not direct_row["invalid_physics"]) if direct_match else None
            metadata.append(dict(id=request["id"], already_direct_first=direct_match, prior_direct_first_capture=capture))
            for file_key, sha_key in (
                ("context_resolution_file", "context_resolution_sha256"),
                ("prepared_connection_file", "prepared_connection_sha256"),
            ):
                path = "native-heading/" + request[file_key]
                proof_paths.add(path)
                checks.equal((fallback_manifest[path], manifest[path]),
                             (request[sha_key], request[sha_key]), request["id"] + "/exact copied proof " + file_key)
        checks.equal(case["candidates"], metadata, identity + "/prior direct-first lineage")
        requests.extend(expected_requests)

    checks.equal(len({request["id"] for request in requests}), len(requests), "Unique diagnostic request IDs")
    return requests, len(proof_paths)


def account_replays(plan, observations, checks):
    names = [request["id"] for case in plan["cases"] for request in case["requests"]]
    checks.equal(len(set(names)), len(names), "Unique frozen diagnostic releases")
    checks.equal(set(observations), set(names), "All and only frozen replay requests were released")
    groups = {name: Counter() for name in ("baseline_repeat", "novel_alternatives", "prior_direct_first_replays")}
    novel_positions = Counter()
    direct_first = Counter()
    untested = Counter()
    paired_labels = {(True, True): "both_capture", (True, False): "repeat_only",
                     (False, True): "novel_only", (False, False): "neither_capture"}
    paired = dict.fromkeys(paired_labels.values(), 0)
    cases = []
    for case in plan["cases"]:
        layout = case["layout"]
        shots = []
        same_route_capture = other_route_capture = False
        novel = []
        for index, (request, lineage) in enumerate(zip(case["requests"], case["candidates"], strict=True)):
            name = request["id"]
            row, native_outcome = observations[name]
            route = request["intended_route"]
            checks.equal((lineage["id"], row["intended_route"]), (name, route), name + "/frozen route and lineage")
            outcome = _released_outcome(row, native_outcome)
            capture = outcome in STANDARD_CAPTURES
            already_tested = lineage["already_direct_first"]
            prior_capture = lineage["prior_direct_first_capture"]
            shots.append(dict(id=name, intended_route=route, outcome=outcome, already_direct_first=already_tested,
                              prior_direct_first_capture=prior_capture))
            if index == 0:
                groups["baseline_repeat"][outcome] += 1
                direct_first["matching_baselines"] += int(already_tested)
            elif already_tested:
                groups["prior_direct_first_replays"][outcome] += 1
                direct_first["previous_captures"] += int(prior_capture)
                direct_first["disagreements"] += int(prior_capture != capture)
            else:
                groups["novel_alternatives"][outcome] += 1
                novel.append(capture)
                if capture:
                    if route == case["requests"][0]["intended_route"]:
                        same_route_capture = True
                    else:
                        other_route_capture = True
        if novel:
            captured = any(novel)
            paired[paired_labels[shots[0]["outcome"] in STANDARD_CAPTURES, captured]] += 1
            novel_positions["positions_tested"] += 1
            novel_positions["positions_captured"] += int(captured)
            novel_positions["same_route_capture_positions"] += int(same_route_capture)
            novel_positions["other_route_capture_positions"] += int(other_route_capture)
        untested["any"] += int(len(shots) == 1)
        untested["novel"] += int(not novel)
        cases.append(dict(layout=layout, prior_selector_id=case["prior_selector_id"],
                          prior_direct_first_id=case["prior_direct_first_id"], baseline_repeat=shots[0],
                          alternatives=shots[1:]))

    def counts(group):
        outcomes = groups[group]
        return dict(releases=sum(outcomes.values()), captures=sum(outcomes[value] for value in STANDARD_CAPTURES),
                    outcomes=dict(outcomes))

    baseline = dict(counts("baseline_repeat"), matches_prior_direct_first=direct_first["matching_baselines"])
    novel = dict(counts("novel_alternatives"), **{key: novel_positions[key] for key in (
        "positions_tested", "positions_captured", "same_route_capture_positions", "other_route_capture_positions",
    )})
    prior = dict(counts("prior_direct_first_replays"), previous_captures=direct_first["previous_captures"],
                 capture_disagreements=direct_first["disagreements"])
    return dict(positions=len(cases), no_alternative_positions=untested["any"],
                no_novel_alternative_positions=untested["novel"], baseline_repeat=baseline,
                novel_alternatives=novel, prior_direct_first_replays=prior, paired_novel=paired,
                setup_failures=plan["setup_failures"], cases=cases)


def audit(study, fallback, fallback_audit, parent, parent_audit, assets):
    checks = Checks()
    spec, report = read(study / "contract.json"), read(study / "report.json")
    parent_spec, parent_binding, _ = parent_inputs(parent, parent_audit)
    checks.equal((spec["version"], report["status"], report["phase"]),
                 (STUDY_VERSION, "completed", "completed"), "Completed diagnostic replays")
    fingerprint(spec, checks, "Candidate replay contract")
    checks.equal((spec["parent"], spec["identity"], spec["scene"], spec["physics_substeps"], spec["label_version"]),
                 (parent_binding, parent_spec["identity"], parent_spec["scene"], 64, LABEL_VERSION),
                 "Exact audited native64 parent and scoring rules")
    manifest = replay.check_manifest(study, checks)
    checks.equal(report["evidence_manifest_sha256"], sha(study / "evidence-manifest.json"),
                 "Report binds all saved evidence")
    checks.equal(spec["frozen_inputs"], {name: manifest[name] for name in
                 ("replay-plan.json", "training-prior.json")}, "Exact frozen input bytes")

    plan, binding = load_replay_plan(fallback, fallback_audit)
    checks.equal(spec["fallback_study"], binding, "Exact previous one-drop comparison and evidence")
    source = read(fallback / "contract.json")
    checks.equal((source["parent"], source["identity"], source["scene"], source["physics_substeps"],
                  source["label_version"]),
                 (parent_binding, spec["identity"], spec["scene"], 64, LABEL_VERSION),
                 "Replays use unchanged previous simulation and scoring")
    checks.equal(manifest["training-prior.json"], sha(fallback / "training-prior.json"),
                 "Same audited arm geometry prior as the archived admissions")
    checks.equal(read(study / "replay-plan.json"), plan, "All and only the prespecified archived misses are diagnosed")
    prior, home = read(study / "training-prior.json"), read(parent / "setup-home.json")
    requests, proofs = verify_commands(fallback, plan, manifest, prior, home, checks)
    checks.equal((report["test_requests"], report["flight_miss_cases"], report["setup_failure_cases"],
                  report["setup_proofs"]), (len(requests), len(plan["cases"]), len(plan["setup_failures"]), proofs),
                 "Every frozen request, diagnosed position and copied proof accounted for")
    measured, controls = audit_packets(study, parent, assets, spec, report, manifest,
                                        {"test": requests}, prior, home, checks)
    verify_realized_routes(study, measured, checks)
    diagnosis = account_replays(plan, measured["test"], checks)
    checks.equal(report["diagnosis"], diagnosis, "Independent diagnosis from rescored replay evidence")
    return dict(version=VERSION, status="passed", checks_count=checks.count,
                test_releases=len(requests), control_releases=controls, diagnosis=diagnosis,
                report_sha256=sha(study / "report.json"), evidence_manifest_sha256=report["evidence_manifest_sha256"],
                replay_plan_sha256=manifest["replay-plan.json"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("study", "fallback-study", "fallback-audit", "parent-study", "parent-audit", "assets", "audit"):
        parser.add_argument("--" + name, required=True, type=Path)
    args = parser.parse_args()
    if args.audit.exists() or args.audit.resolve().is_relative_to(args.study.resolve()):
        parser.error("Write a fresh audit outside the frozen evidence directory")
    try:
        result = audit(args.study, args.fallback_study, args.fallback_audit,
                       args.parent_study, args.parent_audit, args.assets)
    except Exception as error:
        result = dict(version=VERSION, status="failed", error=f"{type(error).__name__}: {error}")
    atomic(args.audit, result)
    print(json.dumps(result), flush=True)
    if result["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
