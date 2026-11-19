"""Prepare fixed-center tilt trials near the reachable drop-height limit."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import os
from pathlib import Path

import numpy as np

from .benchmark_verify import contained, digest, read, sha
from .calibrated_admission import verify_connection, verify_resolution
from .cell1050_heading_native import BRANCH_RULE, HeadingEngine, HeadingResolver, heading_targets
from .cell1050_rebound_study import parent_inputs
from .fixed35_broad_native import distinct_indices, reach_models, wrist_bound
from .intergrid_candidate_replay_study import copy_setup_proofs
from .intergrid_joint_branch_search import PORTABLE_PROPOSAL_FIELDS, circular_delta, joint_distance
from .intergrid_route_labels import LABEL_VERSION
from .policy64_contract import load_runtime_reference
from .sweep_worker import atomic


VERSION = "sircus-case33-tilt-plan-v1"
SOURCE_VERSION = "sircus-intergrid-joint-branch-v1"
SOURCE_AUDIT_VERSION = "sircus-intergrid-joint-branch-independent-audit-v1"
SOURCE_AUDIT_SHA256 = "c59380b20065798f815204608845f2671ee3d2a368a62b08cacc6d07f172e029"
SEED_ID = "intergrid-route-fallback-test-33-table-14-b0-joint-seed-2026111502-b0"
TILTS_DEG = (25, 28, 31)
GAP_BOUNDS_M = (.05, .65)
GAP_STEP_MM = 5


def source_seed(study, source_audit, parent, parent_audit):
    """Bind the archived arm-clear seed and its geometry to the audited study."""
    study, source_audit, parent, parent_audit = map(Path, (study, source_audit, parent, parent_audit))
    report, audit = read(study / "report.json"), read(source_audit)
    manifest = read(study / "evidence-manifest.json")
    binding = dict(
        source_report_sha256=sha(study / "report.json"), source_audit_sha256=sha(source_audit),
        source_manifest_sha256=sha(study / "evidence-manifest.json"),
    )
    if (binding["source_audit_sha256"] != SOURCE_AUDIT_SHA256
            or (audit["version"], audit["status"], report["status"]) != (SOURCE_AUDIT_VERSION, "passed", "completed")
            or audit["report_sha256"] != binding["source_report_sha256"]
            or audit["result"] != report["result"]
            or audit["evidence_manifest_sha256"] != binding["source_manifest_sha256"]
            or report["evidence_manifest_sha256"] != binding["source_manifest_sha256"]):
        raise ValueError("Use the exact audited native64 joint branch study")
    names = ("contract.json", "drop-plan.json", "rows.jsonl", "training-prior.json")
    files = {name: sha(study / name) for name in names}
    if any(files[name] != manifest[name] for name in names):
        raise ValueError("An archived joint branch input changed")
    contract = read(study / "contract.json")
    spec, parent_binding, _ = parent_inputs(parent, parent_audit)
    if (contract["version"] != SOURCE_VERSION or contract["parent"] != parent_binding
            or contract["identity"] != spec["identity"] or contract["scene"] != spec["scene"]
            or contract["physics_substeps"] != 64 or contract["label_version"] != LABEL_VERSION):
        raise ValueError("Archived seed and parent do not share a native64 scene and scorer")
    matches = [(case["layout"], request) for case in read(study / "drop-plan.json")["cases"]
               for request in case["requests"] if request["id"] == SEED_ID]
    ((layout, seed),) = matches
    with (study / "rows.jsonl").open() as stream:
        row, = (row for line in stream if (row := json.loads(line))["id"] == SEED_ID)
    route = row["experimental_route"]
    if (any(row[key] != seed[key] for key in PORTABLE_PROPOSAL_FIELDS)
            or not row["native_release"] or row["invalid_physics"] or row["route_success"]
            or not row["board_bounce"] or route["robot_contact_steps"] != 0
            or row["q10"] != seed["selected_q10"] or row["q14"] != seed["selected_q14"]
            or layout["bucket_xy_m"] != seed["bucket_xy_m"]
            or [round(value * 1000, 1) for value in seed["bucket_xy_m"]] != [1028.9, -19.0]):
        raise ValueError("The archived seed does not match the arm-clear case-33 rebound")
    binding.update(parent=parent_binding, source_files_sha256=files,
                   training_prior_sha256=files["training-prior.json"])
    return seed, layout, spec, binding, manifest


def reach_test(scene, models, seed, tilt):
    """Bound both wrists with the exact heading and board orientation."""
    lower, upper = GAP_BOUNDS_M
    targets = heading_targets(scene, seed["bucket_xy_m"], [*seed["values"][:3], tilt, lower],
                              seed["board_heading_offset_deg"], seed["board_twist_deg"])
    board_outside_mm = 1000 * wrist_bound(models["bouncer"], targets["board_flange_local"])["outside_by_m"]
    if board_outside_mm > 0:
        return None, board_outside_mm
    model = models["dropper"]
    target = targets["dropper_flange_world"]
    wrist = target[:3, 3] - target[:3, 2] * model["terminal_axial_m"]
    offset = wrist - np.asarray(model["shoulder_m"])
    square = model["outer_radius_m"] ** 2 - float(offset[:2] @ offset[:2])
    if square < 0:
        return None, board_outside_mm
    radius = math.sqrt(square)
    bottom = max(lower, lower - offset[2] - radius)
    top = min(upper, lower - offset[2] + radius)
    return ([float(bottom), float(top)] if bottom <= top else None), board_outside_mm


def gap_grid_mm(interval):
    if interval is None:
        return []
    lower, upper = interval
    start = GAP_STEP_MM * math.floor(1000 * upper / GAP_STEP_MM)
    stop = GAP_STEP_MM * math.ceil(1000 * lower / GAP_STEP_MM)
    return list(range(start, stop - 1, -GAP_STEP_MM))


class SeededResolver(HeadingResolver):
    """Search the branches nearest the known clear arm poses first."""

    def _branches(self, first, second, hint, *, prefer_saved=False):
        board_indices = distinct_indices(second)
        if second["valid"][0]:
            board_indices = [0, *[index for index in board_indices
                                 if circular_delta(second["q"][index], second["q"][0])
                                 >= BRANCH_RULE["minimum_joint_difference_rad"]]]

        def priority(pair):
            i, j = pair
            dropper, board = first["q"][i], second["q"][j]
            return (circular_delta(board[:3], hint["q14"][:3]),
                    circular_delta(board, hint["q14"]), circular_delta(dropper, hint["q10"]), i, j)

        return sorted(((i, j) for i in distinct_indices(first) for j in board_indices),
                      key=priority)


def proposal_at(seed, tilt, gap_mm):
    fallback = tilt == seed["values"][3]
    case = "case33-release-up" if fallback else f"case33-tilt-{tilt}"
    return dict(
        {key: seed[key] for key in PORTABLE_PROPOSAL_FIELDS},
        id=f"{case}-gap-{gap_mm:03}",
        variant="release_up" if fallback else f"tilt{tilt}",
        values=[*seed["values"][:3], float(tilt), gap_mm / 1000],
        source_request_id=seed["id"], seed_q10=seed["selected_q10"], seed_q14=seed["selected_q14"],
    )


def baseline_case(seed, layout):
    return dict(layout=dict(layout, id="case33-tilt-baseline"),
                requests=[dict(seed, variant="baseline", source_request_id=seed["id"])])


def verify_schedule(plan, seed, layout):
    baseline, *cases = plan["cases"]
    if baseline != baseline_case(seed, layout) or [case["tilt_deg"] for case in cases[:3]] != list(TILTS_DEG):
        raise ValueError("Frozen plan changed its baseline or tilt treatments")
    expected_fallback = all(case["board_wrist_outside_mm"] > 0 for case in cases[:3])
    if [case["tilt_deg"] for case in cases[3:]] != ([seed["values"][3]] if expected_fallback else []):
        raise ValueError("Use the original tilt as a fallback only when all requested tilts exceed wrist reach")
    for case in cases:
        tilt = case["tilt_deg"]
        fallback = tilt == seed["values"][3]
        case_id = "case33-release-up" if fallback else f"case33-tilt-{tilt}"
        if case["layout"] != dict(layout, id=case_id):
            raise ValueError("A tilt case changed the bucket or layout")
        grid = gap_grid_mm(case["reach_interval_m"])
        if fallback:
            grid = [gap for gap in grid if gap > 1000 * seed["values"][4]]
        trials = case["trials"]
        if case["gap_grid_mm"] != grid or [trial["gap_mm"] for trial in trials] != grid[:len(trials)]:
            raise ValueError("Trials must be a strict descending prefix of the reachable grid")
        requests = case["requests"]
        ids = [request["id"] for request in requests]
        if len(ids) != len(set(ids)) or len(ids) > BRANCH_RULE["maximum_admitted_branches"]:
            raise ValueError("Selected joint branch IDs must be unique and bounded")
        if requests:
            selected = trials[-1]["gap_mm"]
            if case["selected_gap_mm"] != selected:
                raise ValueError("Selected release must be the highest admitted grid point")
        elif len(trials) != len(grid) or case["selected_gap_mm"] is not None:
            raise ValueError("An empty treatment must exhaust the reachable grid")
        for index, trial in enumerate(trials):
            proposal = proposal_at(seed, tilt, trial["gap_mm"])
            record = trial["record"]
            admitted_ids = ids if index == len(trials) - 1 else []
            if (trial["proposal"] != proposal or record["proposal_id"] != proposal["id"]
                    or record["proposal_sha256"] != digest(proposal) or record["admitted"] != len(admitted_ids)
                    or record["admitted_request_ids"] != admitted_ids or bool(record["reason"]) == bool(admitted_ids)):
                raise ValueError("Saved admission records disagree with the descending treatment schedule")
        seen = []
        for request in requests:
            proposal = trials[-1]["proposal"]
            if (request["proposal_id"] != proposal["id"] or request["command_values"] != proposal["values"]
                    or request["native_prepared"] is not True
                    or any(request[key] != proposal[key] for key in proposal if key != "id")):
                raise ValueError("A native request changed the admitted treatment")
            if any(joint_distance(request["selected_q10"], request["selected_q14"], old)
                   < BRANCH_RULE["minimum_joint_difference_rad"] for old in seen):
                raise ValueError("The treatment contains duplicate joint branches")
            seen.append(request)


def proof_inventory(preparation, plan, spec, source_manifest, prior, home):
    directory = preparation / "native-heading"
    inventory, anchors = {}, {}
    for case in plan["cases"]:
        for request in case["requests"]:
            name = request["context_resolution_file"]
            if name not in anchors:
                path = contained(directory, name)
                anchor = dict(read(path), file=name, sha256=sha(path))
                verify_resolution(anchor, prior["records"], home)
                anchors[name] = anchor
            anchor = anchors[name]
            path = contained(directory, request["prepared_connection_file"])
            proof = dict(read(path), file=request["prepared_connection_file"], sha256=sha(path))
            verify_connection(proof, anchor, home, request["values"])
            expected = heading_targets(spec["scene"], request["bucket_xy_m"], request["values"],
                                       request["board_heading_offset_deg"], request["board_twist_deg"])
            if (proof["passed"] is not True or anchor["sha256"] != request["context_resolution_sha256"]
                    or proof["sha256"] != request["prepared_connection_sha256"]
                    or proof["targets"] != {key: value.tolist() for key, value in expected.items()}
                    or proof["heading_key"] != request["heading_key"]
                    or proof["q10"] != request["selected_q10"] or proof["q14"] != request["selected_q14"]
                    or proof["bucket_xy_m"] != request["bucket_xy_m"]):
                raise ValueError("A selected branch disagrees with its exact admitted targets")
            for entry in (anchor, proof):
                relative = f"native-heading/{entry['file']}"
                if request["variant"] == "baseline" and source_manifest[relative] != entry["sha256"]:
                    raise ValueError("The baseline proof changed from the source study")
                inventory[relative] = entry["sha256"]
    return inventory


def load_tilt_plan(preparation, source_study, source_audit, parent_study, parent_audit):
    """Read the frozen preparation and verify selected requests and proofs."""
    preparation, source_study, parent_study = map(Path, (preparation, source_study, parent_study))
    seed, layout, spec, binding, manifest = source_seed(source_study, source_audit, parent_study, parent_audit)
    path = preparation / "tilt-plan.json"
    plan = read(path)
    if plan["version"] != VERSION or plan["source"] != binding or plan["gap_step_mm"] != GAP_STEP_MM:
        raise ValueError("Prepared plan disagrees with the frozen study inputs")
    verify_schedule(plan, seed, layout)
    proofs = proof_inventory(preparation, plan, spec, manifest, read(source_study / "training-prior.json"),
                             read(parent_study / "setup-home.json"))
    inventory = preparation / "preparation-manifest.json"
    if read(inventory) != {"tilt_plan_sha256": sha(path), "proof_manifest": proofs}:
        raise ValueError("Frozen tilt plan or selected proof bytes changed")
    return plan, dict(binding, prepared_plan_sha256=sha(path), proof_manifest=proofs,
                      preparation_manifest_sha256=sha(inventory))


def prepare(args):
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ValueError("Run pose preparation with GPUs hidden")
    import torch

    torch.set_num_threads(1)
    seed, layout, spec, binding, manifest = source_seed(
        args.source_study, args.source_audit, args.parent_study, args.parent_audit,
    )
    ref = load_runtime_reference(args.reference, args.assets, Path(__file__).resolve().parents[2])
    if (digest(ref["scene"]) != spec["identity"]["scene_sha256"]
            or ref["asset_hashes"] != spec["identity"]["asset_sha256"]
            or any(importlib.metadata.version(name) != version for name, version in spec["packages"].items())):
        raise ValueError("Use the pinned simulation assets, scene and Python dependencies")
    output = args.output
    output.mkdir(parents=True, exist_ok=False)
    copy_setup_proofs(args.source_study, output, manifest, [seed])
    prior = read(args.source_study / "training-prior.json")
    home = read(args.parent_study / "setup-home.json")
    engine = HeadingEngine(
        ref, args.assets, output / "native-heading", prior=prior, home=home,
        calibration=read(args.parent_study / "calibration.json"), anchor_source=seed,
        anchor_directory=args.source_study / "native-heading",
    )
    engine.adapter.resolver = SeededResolver(engine.adapter.resolver, args.assets)
    solve_board = engine.adapter.resolver.solver.ik.board.solve_candidates

    def solve_board_preserving_seed(targets, *args, **kwargs):
        return solve_board(targets, *args, preserve_known=np.ones(len(targets), dtype=bool), **kwargs)

    engine.adapter.resolver.solver.ik.board.solve_candidates = solve_board_preserving_seed
    models = reach_models(str(args.assets.resolve()))
    plan = dict(version=VERSION, source=binding, gap_step_mm=GAP_STEP_MM, cases=[baseline_case(seed, layout)])
    for tilt in (*TILTS_DEG, seed["values"][3]):
        fallback = tilt == seed["values"][3]
        if fallback and not all(case["board_wrist_outside_mm"] > 0 for case in plan["cases"][1:]):
            break
        interval, board_outside_mm = reach_test(ref["scene"], models, seed, tilt)
        grid = gap_grid_mm(interval)
        if fallback:
            grid = [gap for gap in grid if gap > 1000 * seed["values"][4]]
        case_id = "case33-release-up" if fallback else f"case33-tilt-{tilt}"
        case = dict(layout=dict(layout, id=case_id), tilt_deg=tilt, reach_interval_m=interval,
                    board_wrist_outside_mm=board_outside_mm,
                    gap_grid_mm=grid, trials=[], selected_gap_mm=None, requests=[])
        if not grid:
            print(json.dumps(dict(tilt_deg=tilt, board_wrist_outside_mm=board_outside_mm,
                                  reachable_grid_points=0)), flush=True)
        for gap in grid:
            proposal = proposal_at(seed, tilt, gap)
            requests, (record,) = engine.prepare([proposal])
            case["trials"].append(dict(gap_mm=gap, proposal=proposal, record=record))
            print(json.dumps(dict(tilt_deg=tilt, gap_mm=gap, admitted=len(requests), reason=record["reason"])),
                  flush=True)
            if requests:
                case.update(selected_gap_mm=gap, requests=requests)
                break
        plan["cases"].append(case)
    if engine.adapter.env is not None or torch.cuda.is_initialized():
        raise RuntimeError("Pose preparation allocated a native environment")
    path = output / "tilt-plan.json"
    atomic(path, plan)
    proofs = proof_inventory(output, plan, spec, manifest, prior, home)
    atomic(output / "preparation-manifest.json", dict(tilt_plan_sha256=sha(path), proof_manifest=proofs))
    load_tilt_plan(output, args.source_study, args.source_audit, args.parent_study, args.parent_audit)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source-study", "source-audit", "parent-study", "parent-audit", "reference", "assets", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    prepare(parser.parse_args())


if __name__ == "__main__":
    main()
