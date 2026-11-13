"""Audit matched direct and table collection from saved MuJoCo evidence on CPU."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

from . import replay650_verify as replay
from .benchmark_verify import Checks, fingerprint, read, sha
from .cell1050_followup_plan import geometry_key
from .cell1050_rebound_study import parent_inputs
from .intergrid_matched_route_study import load_model_inputs
from .intergrid_paired_collection_study import BUDGET_PER_ROUTE, VERSION as STUDY_VERSION, collection_plan, collection_rows
from .intergrid_proposer_audit import audit_packets, audit_preparation
from .intergrid_route_audit import verify_realized_routes
from .intergrid_route_labels import LABEL_VERSION
from .intergrid_route_plan import ROUTES
from .sweep_worker import atomic

VERSION = "sircus-intergrid-paired-route-collection-independent-audit-v1"


def verify_requests(study, plan, prior, home, checks):
    folder = study / "collection"
    expected = {item["layout"]["id"] for item in plan}
    checks.equal({path.stem for path in folder.glob("*.json")}, expected, "One setup file per collection position")
    checks.equal(len(expected), len(plan), "Distinct collection identifiers")
    checks.equal(len({tuple(item["layout"]["bucket_xy_m"]) for item in plan}), len(plan),
                 "Each collection position uses a distinct exact bucket coordinate")
    resolutions, requests = {}, []
    for item in plan:
        layout = item["layout"]
        identity = layout["id"]
        artifact = read(folder / (identity + ".json"))
        checks.equal(artifact["proposals"], item["proposals"], identity + "/frozen ordered menu")
        proposals, admitted = audit_preparation(artifact, layout, "train", study / "native-heading",
                                                prior, home, resolutions, checks)
        checks.equal(Counter(row["intended_route"] for row in proposals),
                     Counter({route: BUDGET_PER_ROUTE for route in ROUTES}), identity + "/equal route budgets")
        checks.equal(len({geometry_key(row) for row in proposals}), len(proposals), identity + "/unique shot geometry")
        checks.require(all(item["prior_captures"][route] > 0 for route in ROUTES),
                       identity + "/both routes captured previously")
        requests.extend(admitted)
    checks.equal(read(study / "admitted.json"), requests, "All and only admitted requests in frozen plan order")
    checks.equal(len({row["id"] for row in requests}), len(requests), "Distinct admitted request identities")
    return requests


def account_collection(plan, requests, measured, checks):
    proposal_results = {}
    for request in requests:
        row, result = measured[request["id"]]
        checks.equal(row["intended_route"], request["intended_route"], request["id"] + "/frozen intended route")
        if result == "standard_capture":
            outcome = row["shot_route"] or "unclassified_capture"
        elif result == "robot_assisted_capture":
            outcome = "arm_assisted_capture"
        elif result == "invalid_physics":
            outcome = result
        else:
            outcome = "flight_miss"
        proposal_id = request["proposal_id"]
        checks.require(proposal_id not in proposal_results, proposal_id + "/one native release")
        proposal_results[proposal_id] = outcome

    cells, totals, paired = [], {route: Counter() for route in ROUTES}, 0
    for item in plan:
        observed = Counter()
        counts = {route: Counter() for route in ROUTES}
        for proposal in item["proposals"]:
            route_counts = counts[proposal["intended_route"]]
            route_counts["proposals"] += 1
            outcome = proposal_results.get(proposal["id"])
            if outcome is not None:
                route_counts["releases"] += 1
                route_counts[outcome] += 1
                observed[outcome] += 1
        for route, count in counts.items():
            count["setup_rejected"] = count["proposals"] - count["releases"]
            totals[route].update(count)
        paired += int(all(observed[route] > 0 for route in ROUTES))
        cells.append(dict(item["layout"], prior_captures=item["prior_captures"], realized=dict(observed),
                          by_intent={route: dict(count) for route, count in counts.items()}))
    return dict(cells=cells, positions_with_both=paired, totals={route: dict(count) for route, count in totals.items()})


def audit(study, parent, parent_audit, model_study, model_audit, matched_study, matched_audit, assets):
    checks = Checks()
    spec, report = read(study / "contract.json"), read(study / "report.json")
    parent_spec, parent_binding, _ = parent_inputs(parent, parent_audit)
    checks.equal((spec["version"], report["status"], report["phase"]),
                 (STUDY_VERSION, "completed", "completed"), "Completed paired collection")
    fingerprint(spec, checks, "Paired collection contract")
    checks.equal((spec["parent"], spec["identity"], spec["scene"]),
                 (parent_binding, parent_spec["identity"], parent_spec["scene"]), "Exact audited native64 parent")
    checks.equal((spec["routes"], spec["budget_per_route"], spec["physics_substeps"], spec["label_version"]),
                 (list(ROUTES), BUDGET_PER_ROUTE, 64, LABEL_VERSION), "Exact route quotas and native scoring")
    manifest = replay.check_manifest(study, checks)
    checks.equal(report["evidence_manifest_sha256"], sha(study / "evidence-manifest.json"),
                 "Report binds complete evidence")
    checks.equal(spec["frozen_inputs"], {name: manifest[name] for name in ("collection-plan.json", "training-prior.json")},
                 "Exact frozen inputs")
    checks.equal(report["admitted_sha256"], manifest["admitted.json"], "Report binds the ordered release list")

    _, rows, prior, _, model_binding = load_model_inputs(model_study, model_audit)
    plan, matched_binding = collection_plan(rows, matched_study, matched_audit, model_binding, parent_binding)
    checks.equal((spec["model_study"], spec["matched_study"]), (model_binding, matched_binding),
                 "Exact previously audited source records")
    checks.equal(manifest["training-prior.json"], model_binding["geometry_prior"], "Exact audited setup geometry prior")
    checks.equal(read(study / "collection-plan.json"), plan, "Reproduced balanced plan at every paired position")
    checks.equal((spec["positions"], report["collection_prepared"]), (len(plan), len(plan)), "Every position prepared")
    home = read(parent / "setup-home.json")
    requests = verify_requests(study, plan, prior, home, checks)
    checks.equal(report["admitted"], len(requests), "Exact number of admitted shots")
    measured, controls = audit_packets(study, parent, assets, spec, report, manifest, {"train": requests}, prior, home,
                                        checks)
    verify_realized_routes(study, measured, checks)
    summary = account_collection(plan, requests, measured["train"], checks)
    checks.equal(report["collection"], summary, "Independent collection counts and verified route labels")

    fresh, excluded = collection_rows(study, plan, manifest)
    with (study / "new-training-dataset.jsonl").open() as stream:
        exported = [json.loads(line) for line in stream]
    checks.equal(exported, fresh, "Exact rebuild of new training rows from native evidence")
    checks.equal((report["valid_training_rows"], report["excluded_from_dataset"]), (len(fresh), excluded),
                 "Every new release accounted for in the training dataset")
    return dict(version=VERSION, status="passed", checks_count=checks.count, positions=len(plan),
                train_releases=len(requests), control_releases=controls, collection=summary,
                valid_training_rows=len(fresh), excluded_from_dataset=excluded,
                report_sha256=sha(study / "report.json"), evidence_manifest_sha256=report["evidence_manifest_sha256"],
                training_dataset_sha256=manifest["new-training-dataset.jsonl"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("study", "parent-study", "parent-audit", "model-study", "model-audit",
                 "matched-study", "matched-audit", "assets", "audit"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    if args.audit.exists() or args.audit.resolve().is_relative_to(args.study.resolve()):
        parser.error("Write a fresh audit outside the frozen evidence directory")
    try:
        result = audit(args.study, args.parent_study, args.parent_audit, args.model_study, args.model_audit,
                       args.matched_study, args.matched_audit, args.assets)
    except Exception as error:
        result = dict(version=VERSION, status="failed", error=f"{type(error).__name__}: {error}")
    atomic(args.audit, result)
    print(json.dumps(result), flush=True)
    if result["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
