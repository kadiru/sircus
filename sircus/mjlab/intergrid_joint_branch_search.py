"""Prepare fixed-task-pose joint alternatives for the four case-33 misses."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
from pathlib import Path

import numpy as np

from .benchmark_verify import contained, digest, read, sha
from .calibrated_admission import verify_connection, verify_resolution
from .cell1050_heading_native import BRANCH_RULE, HeadingEngine, HeadingResolver
from .cell1050_rebound_study import parent_inputs
from .fixed35_bouncer_native import wrapped_difference
from .fixed35_broad_native import distinct_indices
from .intergrid_candidate_replay_study import copy_setup_proofs
from .intergrid_route_labels import LABEL_VERSION
from .intergrid_setup_drop_audit import VERSION as SOURCE_AUDIT_VERSION
from .intergrid_setup_drop_study import VERSION as SOURCE_VERSION
from .policy64_contract import load_runtime_reference
from .sweep_worker import atomic


VERSION = "sircus-intergrid-fixed-pose-joint-branch-plan-v1"
LAYOUT = "intergrid-route-fallback-test-33"
SOURCE_REPORT_SHA256 = "4bfbe28870c0abcbbe854b1a11abdfc215415bc30e8677a50aceceb1874d37aa"
SOURCE_AUDIT_SHA256 = "64de6a57c730c40249875a4245db3b52549f5071974b3abdb39b8b461965974c"
SEEDS = (2026111501, 2026111502, 2026111503)
STARTS_PER_ARM = 32
MINIMUM_DELTA_RAD = BRANCH_RULE["minimum_joint_difference_rad"]
FIXED_FIELDS = ("values", "bucket_xy_m", "board_heading_offset_deg", "board_twist_deg",
                "context_id", "intended_route", "phase", "role", "stratum")
PORTABLE_PROPOSAL_FIELDS = (*FIXED_FIELDS, "method")


def source_case(source_study, source_audit, parent_study, parent_audit):
    """Authenticate the archived study, exact misses and unchanged simulations."""
    study, receipt_path = Path(source_study), Path(source_audit)
    report_hash = sha(study / "report.json")
    audit_hash = sha(receipt_path)
    if (report_hash, audit_hash) != (SOURCE_REPORT_SHA256, SOURCE_AUDIT_SHA256):
        raise ValueError("Use the published native setup-drop report and its audit")
    report, receipt = read(study / "report.json"), read(receipt_path)
    manifest_path = study / "evidence-manifest.json"
    manifest_hash = sha(manifest_path)
    if (report["status"] != "completed" or receipt["status"] != "passed"
            or receipt["version"] != SOURCE_AUDIT_VERSION
            or receipt["report_sha256"] != report_hash or receipt["result"] != report["result"]
            or report["evidence_manifest_sha256"] != manifest_hash
            or receipt["evidence_manifest_sha256"] != manifest_hash):
        raise ValueError("The four parent misses require independently audited evidence")
    manifest = read(manifest_path)
    names = ("contract.json", "drop-plan.json", "rows.jsonl", "training-prior.json")
    bindings = {name: sha(study / name) for name in names}
    if any(bindings[name] != manifest[name] for name in names):
        raise ValueError("An archived setup-drop input changed")
    source_contract = read(study / "contract.json")
    spec, parent, _ = parent_inputs(Path(parent_study), Path(parent_audit))
    if (source_contract["version"] != SOURCE_VERSION or source_contract["parent"] != parent
            or source_contract["identity"] != spec["identity"] or source_contract["scene"] != spec["scene"]
            or source_contract["physics_substeps"] != 64 or source_contract["label_version"] != LABEL_VERSION):
        raise ValueError("The archived misses do not use the pinned native64 scene and scorer")
    cases = [case for case in read(study / "drop-plan.json")["cases"] if case["layout"]["id"] == LAYOUT]
    if len(cases) != 1 or len(cases[0]["requests"]) != 4:
        raise ValueError("Expected the four original requests for case 33")
    case = cases[0]
    xyz = case["layout"]["bucket_xy_m"]
    if [round(value * 1000, 1) for value in xyz] != [1028.9, -19.0]:
        raise ValueError("The archived bucket position differs")
    misses = {item["id"]: item for group in receipt["result"]["cases"] if group["id"] == LAYOUT
              for item in group["shots"]}
    original_ids = {row["id"] for row in case["requests"]}
    if misses.keys() != original_ids or any(
            item["first_outcome"] != "flight_miss" or item["repeat_outcomes"] for item in misses.values()):
        raise ValueError("Require the four exact audited flight misses")
    rows = {}
    with (study / "rows.jsonl").open() as stream:
        for raw in stream:
            row = json.loads(raw)
            if row["id"] in original_ids:
                if row["id"] in rows:
                    raise ValueError("Duplicate archived native observation")
                rows[row["id"]] = row
    if rows.keys() != original_ids:
        raise ValueError("An archived miss was not released")
    for request in case["requests"]:
        row = rows[request["id"]]
        contact = row["experimental_route"]
        if (any(row[key] != request[key] for key in FIXED_FIELDS) or not row["native_release"]
                or row["invalid_physics"] or row["route_success"] or not row["board_bounce"]
                or contact["first_robot_contact_s"] <= contact["first_board_rebound_s"]
                or request["selected_q10"] != row["q10"] or request["selected_q14"] != row["q14"]):
            raise ValueError("Expected the archived board rebound, robot contact and exact joints")
    return case, spec, dict(
        parent=parent, source_report_sha256=report_hash, source_audit_sha256=audit_hash,
        source_manifest_sha256=manifest_hash, source_files_sha256=bindings,
        training_prior_sha256=bindings["training-prior.json"],
    ), manifest


def circular_delta(first, second):
    return float(wrapped_difference(first, second).max())


class BoardFirstResolver(HeadingResolver):
    """Keep the original dropper branch when examining board-arm alternatives."""

    def __init__(self, resolver, assets):
        super().__init__(resolver.base, assets)
        self.original = None
        self.accepted = []

    def _branches(self, first, second, hint, *, prefer_saved=False):
        def distinct(pool):
            indices = distinct_indices(pool)
            if pool["valid"][0]:
                indices = [0] + [index for index in indices
                                 if circular_delta(pool["q"][index], pool["q"][0]) >= MINIMUM_DELTA_RAD]
            return indices

        def key(pair):
            i, j = pair
            q10, q14 = first["q"][i], second["q"][j]
            original = self.original
            same_dropper = np.array_equal(q10, original["selected_q10"])
            proximal = circular_delta(q14[:3], original["selected_q14"][:3])
            board_delta = circular_delta(q14, original["selected_q14"])
            if proximal >= MINIMUM_DELTA_RAD:
                priority = 0 if same_dropper else 1
            elif board_delta >= MINIMUM_DELTA_RAD:
                priority = 2 if same_dropper else 3
            else:
                priority = 4
            return (priority, -proximal, -board_delta, i, j)

        def novel(i, j):
            return all(
                joint_distance(first["q"][i], second["q"][j], old) >= MINIMUM_DELTA_RAD
                for old in (self.original, *self.accepted)
            )

        return sorted(((i, j) for i in distinct(first) for j in distinct(second) if novel(i, j)), key=key)


def joint_distance(q10, q14, request):
    return max(circular_delta(q10, request["selected_q10"]),
               circular_delta(q14, request["selected_q14"]))


def proposal_for(request, seed):
    return dict(
        {key: request[key] for key in PORTABLE_PROPOSAL_FIELDS},
        id=f"{request['id']}-joint-seed-{seed}", source_request_id=request["id"], variant="alternate",
        seed_q10=request["selected_q10"], seed_q14=request["selected_q14"],
    )


def proof_inventory(preparation, plan, source_manifest, prior, home):
    """Verify the saved setup proofs against the original full Cartesian poses."""
    directory = preparation / "native-heading"
    inventory = {}
    resolutions = {}
    for case in plan["cases"]:
        baseline, *alternatives = case["requests"]
        original_proof = read(contained(directory, baseline["prepared_connection_file"]))
        previous = []
        if len(alternatives) > len(SEEDS) * BRANCH_RULE["maximum_admitted_branches"]:
            raise ValueError("The alternative trial budget changed")
        for request in case["requests"]:
            name = request["context_resolution_file"]
            if name not in resolutions:
                anchor_path = contained(directory, name)
                resolutions[name] = dict(read(anchor_path), file=name, sha256=sha(anchor_path))
                verify_resolution(resolutions[name], prior["records"], home)
            anchor = resolutions[name]
            path = contained(directory, request["prepared_connection_file"])
            proof = dict(read(path), file=request["prepared_connection_file"], sha256=sha(path))
            verify_connection(proof, anchor, home, request["values"])
            if (not proof["passed"] or proof["sha256"] != request["prepared_connection_sha256"]
                    or anchor["sha256"] != request["context_resolution_sha256"]
                    or proof["targets"] != original_proof["targets"] or proof["q10"] != request["selected_q10"]
                    or proof["q14"] != request["selected_q14"] or proof["bucket_xy_m"] != request["bucket_xy_m"]
                    or proof["heading_key"] != request["heading_key"]
                    or any(proof[key] != request[key] for key in ("board_heading_offset_deg", "board_twist_deg"))
                    or any(request[key] != baseline[key] for key in (*FIXED_FIELDS, "method"))):
                raise ValueError("A prepared branch differs from its archived Cartesian targets or proof")
            for entry in (anchor, proof):
                relative = f"native-heading/{entry['file']}"
                if request["variant"] == "baseline" and source_manifest[relative] != entry["sha256"]:
                    raise ValueError("An exact baseline proof differs from the archived observation")
                inventory[relative] = entry["sha256"]
            if any(joint_distance(request["selected_q10"], request["selected_q14"], old)
                   < MINIMUM_DELTA_RAD for old in previous):
                raise ValueError("A proposed joint variation duplicates an earlier shot at this pose")
            previous.append(request)
    return inventory


def load_branch_plan(preparation, source_study, source_audit, parent_study, parent_audit):
    """Consume the frozen CPU output after independently verifying every command binding."""
    preparation, source_study, parent_study = map(Path, (preparation, source_study, parent_study))
    case, _, binding, source_manifest = source_case(source_study, source_audit, parent_study, parent_audit)
    path = preparation / "branch-plan.json"
    plan = read(path)
    if (plan["version"] != VERSION or plan["source"] != binding or plan["seeds"] != list(SEEDS)
            or plan["starts_per_arm"] != STARTS_PER_ARM or len(plan["cases"]) != len(case["requests"])):
        raise ValueError("The prepared plan does not match the frozen source or search budget")
    ids = set()
    for prepared, original in zip(plan["cases"], case["requests"], strict=True):
        baseline, *alternatives = prepared["requests"]
        if (prepared["layout"] != dict(case["layout"], id=original["id"])
                or baseline != dict(original, variant="baseline", source_request_id=original["id"])):
            raise ValueError("The prepared baseline changed from the audited request")
        for row in prepared["requests"]:
            if row["source_request_id"] != original["id"] or row["id"] in ids:
                raise ValueError("Each run needs one distinct command ID and original request identity")
            ids.add(row["id"])
        if any(row["variant"] != "alternate" for row in alternatives):
            raise ValueError("Only alternate joint branches may follow each exact baseline")
    prior = read(source_study / "training-prior.json")
    home = read(Path(parent_study) / "setup-home.json")
    proof_manifest = proof_inventory(preparation, plan, source_manifest, prior, home)
    verification = read(preparation / "preparation-manifest.json")
    plan_sha = sha(path)
    if verification != {"branch_plan_sha256": plan_sha, "proof_manifest": proof_manifest}:
        raise ValueError("Frozen preparation plan or certified arm paths differ")
    return plan, dict(binding, prepared_plan_sha256=plan_sha, proof_manifest=proof_manifest,
                      preparation_manifest_sha256=sha(preparation / "preparation-manifest.json"))


def prepare(args):
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ValueError("Run preparation with both GPUs hidden")
    import torch

    torch.set_num_threads(1)
    case, spec, binding, source_manifest = source_case(
        args.source_study, args.source_audit, args.parent_study, args.parent_audit,
    )
    root = Path(__file__).resolve().parents[2]
    ref = load_runtime_reference(args.reference, args.assets, root)
    if (digest(ref["scene"]) != spec["identity"]["scene_sha256"]
            or ref["asset_hashes"] != spec["identity"]["asset_sha256"]
            or any(importlib.metadata.version(name) != version for name, version in spec["packages"].items())):
        raise ValueError("Use the pinned CPU dependencies, scene and assets")
    output = args.output
    output.mkdir(parents=True, exist_ok=False)
    originals = case["requests"]
    copy_setup_proofs(args.source_study, output, source_manifest, originals)
    prior = read(args.source_study / "training-prior.json")
    home = read(args.parent_study / "setup-home.json")
    calibration = read(args.parent_study / "calibration.json")
    engine = HeadingEngine(ref, args.assets, output / "native-heading", prior=prior, calibration=calibration, home=home,
                           anchor_source=originals[0], anchor_directory=args.source_study / "native-heading")
    resolver = BoardFirstResolver(engine.adapter.resolver, args.assets)
    engine.adapter.resolver = resolver
    arms = (resolver.solver.ik.dropper, resolver.solver.ik.board)
    solve = [arm.solve_candidates for arm in arms]
    plan = dict(version=VERSION, source=binding, seeds=list(SEEDS), starts_per_arm=STARTS_PER_ARM, cases=[])
    for request in originals:
        resolver.original = request
        resolver.accepted = []
        logs = []
        for seed in SEEDS:
            for arm, implementation in zip(arms, solve, strict=True):
                def solve_seeded(targets, initial, *, starts, iterations,
                                 implementation=implementation, seed=seed, **kwargs):
                    return implementation(targets, initial, starts=STARTS_PER_ARM, iterations=iterations, seed=seed,
                                          preserve_known=np.ones(len(targets), dtype=bool), **kwargs)

                arm.solve_candidates = solve_seeded
            accepted, records = engine.prepare([proposal_for(request, seed)], one_per_setting=False)
            resolver.accepted.extend(accepted)
            logs.append(dict(seed=seed, **records[0]))
        if engine.adapter.env is not None or torch.cuda.is_initialized():
            raise RuntimeError("CPU preparation unexpectedly allocated a native environment")
        baseline = dict(request, variant="baseline", source_request_id=request["id"])
        plan["cases"].append(dict(layout=dict(case["layout"], id=request["id"]),
                                  requests=[baseline, *resolver.accepted], preparation=logs))
        print(json.dumps({"original_id": request["id"], "alternatives": len(resolver.accepted)}), flush=True)
    atomic(output / "branch-plan.json", plan)
    proofs = proof_inventory(output, plan, source_manifest, prior, home)
    atomic(output / "preparation-manifest.json", dict(branch_plan_sha256=sha(output / "branch-plan.json"),
                                                     proof_manifest=proofs))
    load_branch_plan(output, args.source_study, args.source_audit, args.parent_study, args.parent_audit)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source-study", "source-audit", "parent-study", "parent-audit", "reference", "assets", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    prepare(parser.parse_args())


if __name__ == "__main__":
    main()
