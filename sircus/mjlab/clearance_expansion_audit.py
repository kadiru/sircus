"""Independently rescore and account for the frozen clearance-expansion comparison."""
from __future__ import annotations

import math
from pathlib import Path
from statistics import median

from . import replay650_verify as replay
from .benchmark_verify import Checks, contained, digest, fingerprint, read, sha
from .calibrated_admission import verify_connection, verify_resolution
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
from .intergrid_system_plan import base_proposals, fallback_proposals

VERSION = "sircus-clearance-expansion-independent-audit-v1"
METHODS = ("current_planner", "clearance_expansion")
STAGES = ("base", "fallback", "expansion", "branches")


def _seconds(stage):
    values = [row["timing"]["wall_s"] for row in stage["records"]]
    if not all(type(value) in (int, float) and math.isfinite(value) and value >= 0 for value in values):
        raise ValueError("Setup timing must be finite and nonnegative")
    return math.fsum(values)


def _latency(values):
    ordered = sorted(values)
    return dict(positions=len(ordered), median_s=median(ordered),
                p90_s=ordered[math.ceil(.9 * len(ordered)) - 1], max_s=ordered[-1],
                aggregate_s=math.fsum(ordered))


def summarize(layouts, slots, records, sites):
    """Keep every placement in the denominator and separate reused commands."""
    ids, performance, attempts = _account(layouts, slots, records, METHODS)
    if [site["layout"] for site in sites] != layouts:
        raise ValueError("Preparation must follow the frozen placement order")
    timings, expanded, wall_latency = {}, {}, {}
    for site in sites:
        identity = site["layout"]["id"]
        stage = {name: _seconds(site["plan"][name]) for name in STAGES}
        timings[identity] = (stage["base"] + stage["fallback"], sum(stage.values()))
        expanded[identity] = bool(site["trigger"])
        measured = site["latency_s"]
        if (set(measured) != set(METHODS) or any(type(value) not in (int, float) or not math.isfinite(value)
                                              or value < 0 for value in measured.values())
                or measured[METHODS[1]] < measured[METHODS[0]]):
            raise ValueError("Planner latency must be finite, nonnegative and include baseline preparation")
        wall_latency[identity] = measured
    labels = {(True, True): "both", (True, False): "current_only",
              (False, True): "expanded_only", (False, False): "neither"}
    paired = {name: dict.fromkeys(labels.values(), 0) for name in ("setup", "capture")}
    same = dict(positions=0, capture_disagreements=0, outcome_disagreements=0)
    triggered = dict(positions=0, current_captures=0, expanded_captures=0, changed_commands=0)
    per_position = []
    for identity in sorted(ids):
        (left, left_outcome), (right, right_outcome) = (attempts[identity, name] for name in METHODS)
        lreq, rreq = left["request"], right["request"]
        lhit, rhit = left_outcome in STANDARD_CAPTURES, right_outcome in STANDARD_CAPTURES
        paired["setup"][labels[lreq is not None, rreq is not None]] += 1
        paired["capture"][labels[lhit, rhit]] += 1
        identical = lreq is not None and rreq is not None and command_key(lreq) == command_key(rreq)
        if identical:
            same["positions"] += 1
            same["capture_disagreements"] += int(lhit != rhit)
            same["outcome_disagreements"] += int(left_outcome != right_outcome)
        if expanded[identity]:
            triggered["positions"] += 1
            triggered["current_captures"] += int(lhit)
            triggered["expanded_captures"] += int(rhit)
            triggered["changed_commands"] += int(not identical and rreq is not None)
        elif not identical and (lreq is not None or rreq is not None):
            raise ValueError("Non-triggered placement changed its baseline command")
        per_position.append(dict(
            id=identity, bucket_xy_m=left["bucket_xy_m"], expanded=expanded[identity],
            current_outcome=left_outcome, expanded_outcome=right_outcome, same_command=identical,
        ))
    contacts = {}
    for method in METHODS:
        selected = [records[slot["id"]] for slot in slots
                    if slot["method"] == method and slot["request"] is not None]
        contacts[method] = sum(row.get("experimental_route", {}).get("robot_contact_steps", 0) > 0
                               for row in selected)
        counts = performance[method]
        counts.update(positions=counts["attempts"], admitted=len(selected), released=len(selected),
                      unresolved=counts["attempts"] - len(selected),
                      flight_misses=counts["outcomes"].get("flight_miss", 0),
                      robot_contacts=contacts[method])
        selected_stages = dict.fromkeys((*STAGES, "unresolved"), 0)
        for slot in slots:
            if slot["method"] == method:
                stage = slot["candidate_pool"] if slot["request"] is not None else "unresolved"
                selected_stages[stage] += 1
        counts["selected_stages"] = selected_stages
    paired.update(gained=paired["capture"]["expanded_only"], lost=paired["capture"]["current_only"],
                  both=paired["capture"]["both"], neither=paired["capture"]["neither"],
                  changed_positions=sum(not row["same_command"] and
                                        any(attempts[row["id"], name][0]["request"] is not None for name in METHODS)
                                        for row in per_position))
    latency = {name: _latency([wall_latency[identity][name] for identity in ids]) for name in METHODS}
    return dict(methods=performance, paired=paired, same_command=same, triggered=triggered,
                robot_contact_releases=contacts, positions=per_position,
                prepared_candidates={name: dict(
                    proposals=sum(len(site["plan"][name]["proposals"]) for site in sites),
                    admitted=sum(len(site["plan"][name]["admitted"]) for site in sites),
                ) for name in STAGES},
                setup_cpu_worker_wall_time={name: _latency([timings[identity][i] for identity in ids])
                                            for i, name in enumerate(METHODS)}, latency=latency)


def verify_layouts(study, parent, history_root, scene, assets, previous, checks):
    from .clearance_expansion_study import POSITION_COUNT, SEED, layout_ids

    archive = read(study / "layouts.json")
    checks.equal(set(archive), {"layouts", "provenance"}, "One frozen TEST placement set")
    layouts, provenance = archive["layouts"], archive["provenance"]
    grid = read(parent / "grid.json")
    geometry = _placement_geometry(scene, assets)
    canonical, _ = _grid_points(grid)
    exclusions = [tuple(xy) for xy in provenance["excluded_coordinates_m"]]
    declared = sorted({tuple(row["bucket_xy_m"]) for row in previous})
    checks.equal(len(layouts), POSITION_COUNT, "50 new bucket positions")
    checks.equal((provenance["version"], provenance["seed"], provenance["minimum_separation_m"],
                  provenance["geometry"]), (PLAN_VERSION, SEED, MIN_SEPARATION_M, geometry),
                 "Frozen spatial sampling contract")
    checks.equal(provenance["declared_previous_coordinates_m"], [list(xy) for xy in declared],
                 "Every supplied prior study and interactive position is excluded")
    checks.equal(exclusions, sorted(set(exclusions)), "Sorted unique spatial exclusions")
    checks.equal(provenance["excluded_coordinates_sha256"], digest(exclusions), "Exclusion digest")
    checks.equal(provenance["layouts_sha256"], digest(layouts), "Frozen placement digest")
    historical = verify_history_files(provenance, history_root, study, checks)
    checks.equal(set(exclusions), historical | set(declared) | set(canonical.values()),
                 "All and only authenticated historical, declared and grid positions excluded")
    checks.equal(layouts, layout_ids(sample_layouts(exclusions, geometry, grid, seed=SEED)),
                 "Reproduce fresh placements from seed and geometry")
    verify_separation(layouts, [], exclusions, checks)
    return layouts


def _proof(request, layout, directory, prior, home, resolutions, checks):
    """Verify path, endpoint, frame and file provenance for any prepared branch."""
    label = request["id"]
    name = request["context_resolution_file"]
    if name not in resolutions:
        path = contained(directory, name)
        anchor = dict(read(path), file=name, sha256=sha(path))
        verify_resolution(anchor, prior["records"], home)
        resolutions[name] = anchor
    anchor = resolutions[name]
    checks.equal((anchor["bucket_xy_m"], anchor["sha256"]),
                 (layout["bucket_xy_m"], request["context_resolution_sha256"]), label + "/exact anchor")
    path = contained(directory, request["prepared_connection_file"])
    proof = dict(read(path), file=request["prepared_connection_file"], sha256=sha(path))
    verify_connection(proof, anchor, home, request["values"])
    checks.equal((proof["passed"], proof["sha256"], proof["q10"], proof["q14"], proof["heading_key"],
                  proof["board_heading_offset_deg"], proof["board_twist_deg"]),
                 (True, request["prepared_connection_sha256"], request["selected_q10"], request["selected_q14"],
                  request["heading_key"], request["board_heading_offset_deg"], request["board_twist_deg"]),
                 label + "/verified exact arm paths and endpoint")
    return proof


def audit_branches(artifact, layout, originals, directory, prior, home, resolutions, checks):
    """Bind multiple board-only joint alternatives to unchanged Cartesian targets."""
    from .clearance_expansion_plan import BRANCH_SETTINGS, MAX_BRANCHES_PER_SETTING, MINIMUM_JOINT_DELTA_RAD

    proposals, admitted, records = (artifact[name] for name in ("proposals", "admitted", "records"))
    identity = layout["id"]
    checks.require(len(proposals) <= BRANCH_SETTINGS, identity + "/bounded branch source poses")
    checks.equal(len(records), len(proposals), identity + "/one branch setup record per proposal")
    checks.equal(len({row["id"] for row in proposals}), len(proposals), identity + "/unique branch proposals")
    by_id = {row["id"]: row for row in admitted}
    checks.equal(len(by_id), len(admitted), identity + "/unique admitted alternative identities")
    ordered = []
    for proposal, record in zip(proposals, records, strict=True):
        label = proposal["id"]
        checks.equal((proposal["context_id"], proposal["bucket_xy_m"], proposal["phase"]),
                     (identity, layout["bucket_xy_m"], "test"), label + "/fixed held-out placement")
        checks.equal((record["proposal_id"], record["proposal_sha256"]), (label, digest(proposal)),
                     label + "/frozen branch source proposal")
        ids = record["admitted_request_ids"]
        checks.equal(len(ids), record["admitted"], label + "/exact admitted branch count")
        checks.require(len(ids) <= MAX_BRANCHES_PER_SETTING, label + "/bounded admitted alternatives")
        checks.equal(ids, [label + f"-b{i}" for i in range(len(ids))], label + "/ordered branch identities")
        original = originals[proposal["source_request_id"]]
        original_proof = _proof(original, layout, directory, prior, home, resolutions, checks)
        checks.equal((proposal["seed_q10"], proposal["seed_q14"]),
                     (original["selected_q10"], original["selected_q14"]), label + "/exact source joint seed")
        checks.equal((proposal["values"], proposal["board_heading_offset_deg"], proposal["board_twist_deg"]),
                     (original["values"], original["board_heading_offset_deg"], original["board_twist_deg"]),
                     label + "/same shot geometry as admitted source")
        previous = [original["selected_q14"]]
        for branch_id in ids:
            request = by_id[branch_id]
            checks.equal({key: request[key] for key in proposal if key != "id"},
                         {key: value for key, value in proposal.items() if key != "id"},
                         branch_id + "/exact proposal retained by admitted branch")
            checks.equal((request["proposal_id"], request["native_prepared"], request["command_values"]),
                         (label, True, proposal["values"]), branch_id + "/admitted native command")
            checks.equal(request["selected_q10"], original["selected_q10"],
                         branch_id + "/dropper joints unchanged")
            joints = request["selected_q14"]
            checks.require(len(joints) == 6 and all(math.isfinite(value) for value in joints),
                           branch_id + "/finite six-joint board arm")
            checks.require(all(max(abs(math.remainder(a - b, math.tau))
                                   for a, b in zip(joints, prior_joints, strict=True))
                               >= MINIMUM_JOINT_DELTA_RAD - 1e-12 for prior_joints in previous),
                           branch_id + "/distinct board configuration modulo full revolutions")
            previous.append(joints)
            proof = _proof(request, layout, directory, prior, home, resolutions, checks)
            checks.equal(proof["targets"], original_proof["targets"],
                         branch_id + "/unchanged full board and release targets")
            ordered.append(branch_id)
    checks.equal(ordered, list(by_id), identity + "/complete ordered alternative admissions")
    return admitted


def verify_plans(study, layouts, libraries, checkpoint, history, ranker, prior, home, checks):
    """Recompute proposals, predictions and choices without any native rollout."""
    from .clearance_expansion_plan import (
        _all_admitted, branch_proposals, current_selection, expansion_proposals, needs_expansion, select_requests,
    )

    sites = read(study / "plans.json")
    checks.equal([site["layout"] for site in sites], layouts, "One ordered preparation per frozen placement")
    checks.equal({path.stem for path in (study / "preparation").glob("*.json")},
                 {layout["id"] for layout in layouts}, "Complete per-placement preparation inventory")
    native, resolutions = study / "native-heading", {}
    for site in sites:
        layout, plan = site["layout"], site["plan"]
        identity = layout["id"]
        checks.equal(read(study / "preparation" / (identity + ".json")), site,
                     identity + "/per-placement artifact agrees with assembled plans")
        checks.equal(set(plan), set(STAGES), identity + "/exact four preparation stages")
        checks.require(type(site["trigger"]) is bool, identity + "/explicit expansion decision")
        checks.equal(plan["base"]["proposals"], base_proposals(layout, libraries, checkpoint),
                     identity + "/unchanged baseline proposal menu")
        base, admitted = audit_preparation(plan["base"], layout, "test", native, prior, home, resolutions, checks)
        checks.equal(plan["fallback"]["proposals"], fallback_proposals(layout, libraries, base, admitted),
                     identity + "/unchanged setup-only baseline fallback")
        audit_preparation(plan["fallback"], layout, "test", native, prior, home, resolutions, checks)
        baseline = current_selection(layout, plan, history, ranker)
        checks.equal(site["trigger"], needs_expansion(baseline),
                     identity + "/expansion uses only missing setup or nonpositive predicted clearance")
        if site["trigger"]:
            checks.equal(plan["expansion"]["proposals"], expansion_proposals(layout, libraries, plan),
                         identity + "/exact conditional wider geometry menu")
        else:
            for name in STAGES[2:]:
                checks.equal(plan[name], dict(proposals=[], admitted=[], records=[]),
                             identity + "/no extra search for positive baseline clearance")
        audit_preparation(plan["expansion"], layout, "test", native, prior, home, resolutions, checks)
        expected_branches = branch_proposals(layout, plan, ranker) if site["trigger"] else []
        checks.equal(plan["branches"]["proposals"], expected_branches,
                     identity + "/exact physics-ranked alternative-joint source poses")
        originals = {row["id"]: row for row in _all_admitted(plan, STAGES[:3])}
        audit_branches(plan["branches"], layout, originals, native, prior, home, resolutions, checks)
        checks.equal(site["predictions"], [dict(id=row["id"], **ranker.predict(row)) for row in _all_admitted(plan)],
                     identity + "/independently reproduced pre-release predictions")
        choices = select_requests(layout, plan, history, ranker)
        checks.equal(site["choices"], choices, identity + "/reconstructed frozen one-drop decisions")
        checks.equal([(row["context_id"], row["method"]) for row in choices],
                     [(identity, method) for method in METHODS], identity + "/exact two paired slots")
        if baseline["request"] is not None:
            checks.equal(command_key(choices[0]["request"]), command_key(baseline["request"]),
                         identity + "/current command exactly preserved")
            checks.require(choices[1]["request"] is not None, identity + "/expansion cannot delete available baseline")
        else:
            checks.require(choices[0]["request"] is None, identity + "/unresolved baseline retained in denominator")
    return sites


def bound_inputs(args, spec, checks):
    """Reopen only the interactive inventory frozen before this experiment."""
    from .intergrid_system_study import inputs
    from .interactive_physics_check import historical_layouts

    checkpoint, original, additions, previous, binding, history = inputs(args)
    _, _, _, system_binding = historical_layouts(args.system_study, args.system_audit)
    archive = read(args.system_study / "layouts.json")
    previous += [row for split in archive.values() for row in split["layouts"]]
    inventory = spec["interactive_history"]
    checks.require(isinstance(inventory, list) and bool(inventory), "Frozen interactive placement inventory")
    names = [item["file"] for item in inventory]
    # The producer sorts Path objects by components before serializing them.
    # Plain string ordering reverses evidence/ and evidence-acceptance/.
    checks.equal(names, sorted(set(names), key=Path), "Sorted unique frozen interactive files")
    for item in inventory:
        name = item["file"]
        path = contained(args.interactive_history, name)
        checks.require(len(Path(name).parts) == 3 and Path(name).parts[0] in {"evidence", "evidence-acceptance"}
                       and Path(name).parts[1].startswith("interactive-") and Path(name).parts[2] == "result.json",
                       "Declared input is an interactive result")
        xy = read(path)["bucket_xy_m"]
        checks.require(len(xy) == 2 and all(math.isfinite(value) for value in xy), name + "/finite placement")
        checks.equal(item, dict(file=name, sha256=sha(path), bucket_xy_m=xy), name + "/frozen interactive evidence")
        previous.append(dict(bucket_xy_m=xy))
    binding = dict(binding, system_comparison=system_binding, interactive_history=inventory)
    checks.equal({key: spec[key] for key in binding}, binding, "Audited baseline and historical evidence binding")
    return checkpoint, original, additions, previous, binding, history


def audit(args):
    """Check complete native64 evidence, unchanged scoring and one drop per method."""
    from . import clearance_expansion_study as runner
    from .clearance_expansion_plan import protocol

    study, parent, checks = args.output, args.parent_study, Checks()
    spec, report = read(study / "contract.json"), read(study / "report.json")
    parent_spec, parent_binding, _ = parent_inputs(parent, args.parent_audit)
    fingerprint(spec, checks, "Frozen clearance-expansion comparison")
    checks.equal((spec["version"], report["status"], report["phase"]), (runner.VERSION, "completed", "completed"),
                 "Completed frozen experiment")
    checks.equal((spec["parent"], spec["identity"], spec["scene"]),
                 (parent_binding, parent_spec["identity"], parent_spec["scene"]), "Exact audited native scene")
    checks.equal((spec["test_positions"], spec["seed"], spec["physics_substeps"], spec["label_version"],
                  spec["methods"], spec["attempts_per_method_per_position"], spec["protocol"]),
                 (runner.POSITION_COUNT, runner.SEED, 64, LABEL_VERSION, list(METHODS), 1, protocol()),
                 "Frozen positions, native64 scoring, methods and search budget")
    checks.require(type(spec["gpu_cap_s"]) is int and 0 < report["gpu_lease_s"] <= spec["gpu_cap_s"] <= runner.GPU_CAP_S,
                   "Completed within one authorized GPU-hour")
    source = Path(__file__).resolve().parent
    for name in ("clearance_expansion_plan.py", "clearance_expansion_study.py", "intergrid_system_plan.py",
                 "intergrid_flight_ranker.py"):
        checks.equal(spec["source_sha256"]["sircus/mjlab/" + name], sha(source / name), name + "/exact producer code")
    manifest = replay.check_manifest(study, checks)
    checks.equal(report["evidence_manifest_sha256"], sha(study / "evidence-manifest.json"), "Complete evidence binding")
    checks.equal(report["choices_sha256"], manifest["choices.json"], "Report binds exact pre-release choices")
    checks.equal(spec["frozen_inputs"], {name: manifest[name] for name in
                 ("layouts.json", "training-prior.json", "route-models.json", "route-history.json")},
                 "Exact immutable input inventory")
    checks.equal((study / "native-started").read_text().strip(), manifest["choices.json"],
                 "Native execution began with these exact frozen choices")
    checkpoint, original, additions, previous, binding, history = bound_inputs(args, spec, checks)
    checks.equal(manifest["route-models.json"], binding["refit_study"]["checkpoint_sha256"], "No model retraining")
    checks.equal(manifest["training-prior.json"], binding["model_study"]["geometry_prior"], "Unchanged geometry prior")
    checks.equal(read(study / "route-history.json"), history, "Unchanged prior route outcomes")
    libraries, _ = verify_library(original, additions, checks)
    layouts = verify_layouts(study, parent, args.history_root, spec["scene"], args.assets, previous, checks)
    ranker = _ranker(args, spec["scene"], binding)
    prior, home = read(study / "training-prior.json"), read(parent / "setup-home.json")
    sites = verify_plans(study, layouts, libraries, checkpoint, history, ranker, prior, home, checks)
    slots = [slot for site in sites for slot in site["choices"]]
    checks.equal(read(study / "choices.json"), dict(plans_sha256=manifest["plans.json"], slots=slots),
                 "Exact frozen plan binding and one-drop command inventory")
    requests = [slot["request"] for slot in slots if slot["request"] is not None]
    checks.equal(len({row["id"] for row in requests}), len(requests), "Unique native release identifiers")
    checks.equal((report["test_slots"], report["test_admitted"], report["prepared_positions"], report["total_positions"],
                  report["planned_releases"], report["completed_releases"]),
                 (2 * runner.POSITION_COUNT, len(requests), runner.POSITION_COUNT, runner.POSITION_COUNT,
                  len(requests), len(requests)), "Every position and admitted release counted")
    measured, controls = audit_packets(study, parent, args.assets, spec, report, manifest,
                                       {"test": requests}, prior, home, checks)
    verify_realized_routes(study, measured, checks)
    records = {identity: row for identity, (row, _) in measured["test"].items()}
    comparison = summarize(layouts, slots, records, sites)
    checks.equal(report["comparison"], comparison, "Summary agrees with independently rescored native traces")
    return dict(version=VERSION, status="passed", checks_count=checks.count, test_releases=len(records),
                control_releases=controls, comparison=comparison, report_sha256=sha(study / "report.json"),
                choices_sha256=manifest["choices.json"], plans_sha256=manifest["plans.json"],
                evidence_manifest_sha256=report["evidence_manifest_sha256"])
