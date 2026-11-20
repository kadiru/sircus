"""Screen wider shot proposals at the five audited no-setup bucket positions."""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import as_completed
import importlib.metadata
import json
from pathlib import Path

from .benchmark_verify import Checks, digest, read, sha
from .cell1050_followup_plan import geometry_key
from .cell1050_rebound_study import parent_inputs
from .intergrid_candidate_replay_plan import load_replay_plan
from .intergrid_proposer_audit import audit_preparation
from .intergrid_route_plan import PER_ROUTE_LIMIT, ROUTES, collection_proposals
from .intergrid_route_refit_inputs import _audited_source
from .intergrid_route_refit_study import VERSION as REFIT_VERSION
from .intergrid_route_study import route_library
from .parallel_capture_grid_workers import open_workers, prepare_proposals
from .policy64_contract import load_runtime_reference
from .sweep_worker import atomic


VERSION = "sircus-intergrid-setup-recovery-v1"


def inputs(args):
    spec, parent_binding, _ = parent_inputs(args.parent_study, args.parent_audit)
    replay, fallback_binding = load_replay_plan(args.fallback_study, args.fallback_audit)
    _, _, _, refit_binding, refit = _audited_source(
        args.refit_study, args.refit_audit, REFIT_VERSION, ("training-dataset.jsonl",),
    )
    fallback = read(args.fallback_study / "contract.json")
    for contract in (fallback, refit):
        if contract["parent"] != parent_binding or contract["identity"] != spec["identity"]:
            raise ValueError("Setup inputs require the same audited scene and reference")
    repository = Path(__file__).resolve().parents[2]
    ref = load_runtime_reference(args.reference, args.assets, repository)
    if (digest(ref["scene"]) != spec["identity"]["scene_sha256"]
            or ref["asset_hashes"] != spec["identity"]["asset_sha256"]
            or any(importlib.metadata.version(name) != pinned for name, pinned in spec["packages"].items())):
        raise ValueError("Setup geometry or dependencies differ from the audited reference")
    with (args.refit_study / "training-dataset.jsonl").open(encoding="utf-8") as stream:
        library = route_library([json.loads(line) for line in stream])
    plan = []
    for previous in replay["setup_failures"]:
        layout = dict(previous["layout"], split="train")
        proposals = collection_proposals(layout, library)
        plan.append(dict(layout=layout, proposals=proposals, previous=previous))
    plan_document = dict(version=VERSION, parent=parent_binding, fallback=fallback_binding,
                         refit=refit_binding, source=sha(__file__), proposals_per_route=PER_ROUTE_LIMIT, cases=plan)
    prior = read(args.fallback_study / "training-prior.json")
    home = read(args.parent_study / "setup-home.json")
    calibration = read(args.parent_study / "calibration.json")
    return plan_document, ref, dict(prior=prior, calibration=calibration, home=home)


def verify(args, plan, setup):
    checks = Checks()
    checks.equal(read(args.output / "plan.json"), plan, "Exact authenticated proposal plan")
    resolutions = {}
    cases = []
    for case in plan["cases"]:
        layout = case["layout"]
        artifact = read(args.output / "preparations" / f"{layout['id']}.json")
        checks.equal(artifact["proposals"], case["proposals"], "Exact generated proposals")
        proposals, admitted = audit_preparation(
            artifact, layout, "train", args.output / "native-heading", setup["prior"], setup["home"],
            resolutions, checks,
        )
        checks.equal(len({geometry_key(proposal) for proposal in proposals}), len(proposals),
                     "Distinct proposal geometries")
        routes = {}
        for route in ROUTES:
            counts = Counter()
            ids = []
            for proposal, record in zip(proposals, artifact["records"], strict=True):
                if proposal["intended_route"] != route:
                    continue
                checks.equal(record["reason"] is None, record["admitted"] == 1, "Admission or rejection reason")
                counts[record["reason"] or "admitted"] += 1
                ids.extend(record["admitted_request_ids"])
            checks.equal(sum(counts.values()), PER_ROUTE_LIMIT, "Complete per-route proposal budget")
            routes[route] = dict(outcomes=dict(counts), admitted_ids=ids)
        cases.append(dict(id=layout["id"], bucket_xy_m=layout["bucket_xy_m"],
                          admitted=len(admitted), routes=routes))
    return dict(version=VERSION, status="passed", checks_count=checks.count, cases=cases,
                plan_sha256=sha(args.output / "plan.json"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("parent-study", "parent-audit", "fallback-study", "fallback-audit", "refit-study",
                 "refit-audit", "reference", "assets", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--audit", type=Path, help="Verify the saved preparation records and write an audit receipt")
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    plan, ref, setup = inputs(args)
    if args.audit is not None:
        result = verify(args, plan, setup)
        atomic(args.audit, result)
        print(json.dumps(result), flush=True)
        return
    args.output.mkdir(parents=True, exist_ok=False)
    atomic(args.output / "plan.json", plan)
    with open_workers(args.workers, ref, args.assets, args.output / "native-heading", setup) as workers:
        jobs = {workers.submit(prepare_proposals, case["proposals"], True): case for case in plan["cases"]}
        for job in as_completed(jobs):
            case = jobs[job]
            admitted, records = job.result()
            identity = case["layout"]["id"]
            atomic(args.output / "preparations" / f"{identity}.json",
                   dict(proposals=case["proposals"], admitted=admitted, records=records))
            print(f"{identity}: {len(admitted)} certified setups", flush=True)


if __name__ == "__main__":
    main()
