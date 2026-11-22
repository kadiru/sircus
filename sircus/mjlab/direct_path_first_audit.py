"""Read-only independent audit of candidate-first paths with unchanged native64 scoring.

The inner reusable-planner receipts retain their original warm_early identity.
Only outer variant slots use baseline/direct_first names. No audited historical
module or global constant is patched by this verifier.
"""
from __future__ import annotations

from collections import Counter
import copy
import math
import numpy as np
from scipy.spatial.transform import Rotation
from pathlib import Path
import random

from .benchmark_verify import Checks, contained, digest, fingerprint, read, sha
from .fast_planning_audit import (
    EARLY_SCORE_M, STANDARD_CAPTURES, latency, seconds, verify_bindings,
    verify_deadline, verify_early_rule,
)

VERSION = "sircus-direct-path-first-independent-audit-v2"
STUDY_VERSION = "sircus-direct-path-first-study-v1"
METHODS = ("baseline", "direct_first")



# The new version explicitly admits a checked home-to-candidate path without
# claiming that an exhaustive legacy reference search has already happened.
def verify_resolution(record, sources, home):
    from .calibrated_admission import verify_resolution as historical
    from .direct_path_first import VERSION as DIRECT_VERSION, verify_resolution as direct
    return (direct if record.get("version") == DIRECT_VERSION else historical)(record, sources, home)


def verify_connection(record, anchor, home, values):
    from .calibrated_admission import verify_connection as historical
    from .direct_path_first import VERSION as DIRECT_VERSION, verify_connection as direct
    return (direct if record.get("version") == DIRECT_VERSION else historical)(record, anchor, home, values)


def verify_fallback_bytes(proof, directory, prior, home, checks):
    fallback = proof.get("fallback_resolution")
    if fallback is None:
        return
    path = contained(directory, fallback["file"])
    record = dict(read(path), file=fallback["file"], sha256=sha(path))
    checks.equal(record, {key: value for key, value in fallback.items() if key != "cache_hit"},
                 "Reference fallback binds exact immutable resolution bytes")
    verify_resolution(record, prior["records"], home)


def verify_seed(proof, proposal, checks):
    """A changed IK hint is explicit; it never rewrites the frozen shot proposal."""
    from .direct_path_first import VERSION as DIRECT_VERSION
    if proof.get("version") != DIRECT_VERSION:
        return
    seed = proof["ik_seed"]
    checks.equal(seed["proposal_sha256"], digest(proposal), "Direct-first IK seed binds exact proposal")
    explicit = "seed_q10" in proposal or "seed_q14" in proposal
    checks.equal(seed["kind"], "explicit_proposal_joints" if explicit else "deferred_source_geometry",
                 "Direct-first IK hint source explicitly declared")
    if explicit:
        checks.equal((seed["q10"], seed["q14"]), (proposal["seed_q10"], proposal["seed_q14"]),
                     "Explicit alternative joint hints unchanged")


def verify_heading(row, request, directory, scene, home, prior, resolutions, checks):
    name = row["id"]
    resolution_name = row["context_resolution_file"]
    if resolution_name not in resolutions:
        path = contained(directory, resolution_name)
        resolution = dict(read(path), file=resolution_name, sha256=sha(path))
        verify_resolution(resolution, prior["records"], home)
        resolutions[resolution_name] = resolution
    resolution = resolutions[resolution_name]
    checks.equal(row["context_resolution_sha256"], resolution["sha256"], name + "/frozen resolution")
    file = request["prepared_connection_file"]
    path = contained(directory, file)
    proof = dict(read(path), file=file, sha256=sha(path))
    checks.equal((row["connection_file"], row["connection_sha256"]),
                 (file, proof["sha256"]), name + "/frozen connection")
    checks.equal(proof["sha256"], request["prepared_connection_sha256"], name + "/request proof bytes")
    verify_connection(proof, resolution, home, request["values"])
    verify_fallback_bytes(proof, directory, prior, home, checks)
    checks.require(proof["passed"], name + "/passed path proof")
    for key in ("board_heading_offset_deg", "board_twist_deg", "heading_key"):
        checks.equal((row[key], proof[key]), (request[key], request[key]), name + "/" + key)
    for key in ("q10", "q14"):
        checks.equal((row[key], proof[key]), (request["selected_" + key], request["selected_" + key]),
                     name + "/requested joint branch/" + key)
        checks.equal(row["native_initial_" + key], np.asarray(row[key], np.float32).astype(float).tolist(),
                     name + "/observed exact reset joints/" + key)
    values = np.asarray(request["values"])
    delta = np.asarray(request["bucket_xy_m"]) - values[:2]
    azimuth = np.arctan2(delta[1], delta[0]) + np.pi / 2 + np.deg2rad(request["board_heading_offset_deg"])
    spin = scene["board"]["mount_yaw_rad"] + np.deg2rad(request["board_twist_deg"])
    expected = Rotation.from_euler("z", azimuth) * Rotation.from_euler("x", values[3], degrees=True) * Rotation.from_euler("z", spin)
    actual = Rotation.from_quat(np.roll(row["board_quaternion_wxyz"], -1))
    checks.require((actual * expected.inv()).magnitude() < 4e-4, name + "/measured full board orientation")
    checks.numeric(row["board_normal"], actual.as_matrix()[:, 2], name + "/measured board normal", 2e-6)
    checks.require(np.linalg.norm(np.asarray(row["board_center_m"]) - values[:3]) < .002,
                   name + "/measured board position")


def audit_preparation(artifact, layout, phase, directory, prior, home, resolutions, checks):
    identity = layout["id"]
    proposals, admitted, records = (artifact[key] for key in ("proposals", "admitted", "records"))
    checks.equal(len(records), len(proposals), identity + "/one setup record per proposal")
    checks.equal(len({item["id"] for item in proposals}), len(proposals), identity + "/unique proposal IDs")
    branches = {item["id"]: item for item in admitted}
    checks.equal(len(branches), len(admitted), identity + "/unique admitted branch IDs")
    ordered = []
    for proposal, record in zip(proposals, records, strict=True):
        label = proposal["id"]
        checks.equal((proposal["context_id"], proposal["bucket_xy_m"], proposal["phase"]),
                     (identity, layout["bucket_xy_m"], phase), label + "/exact frozen layout and role")
        checks.equal((record["proposal_id"], record["proposal_sha256"]), (label, digest(proposal)),
                     label + "/frozen proposal geometry")
        ids = record["admitted_request_ids"]
        checks.equal(len(ids), record["admitted"], label + "/exact admission count")
        checks.require(len(ids) <= 1, label + "/at most one admitted arm branch")
        for branch_id in ids:
            request = branches[branch_id]
            checks.equal(branch_id, label + "-b0", label + "/admitted branch identity")
            checks.equal({key: request[key] for key in proposal if key != "id"},
                         {key: value for key, value in proposal.items() if key != "id"},
                         label + "/admitted request retains exact proposal")
            checks.equal((request["proposal_id"], request["native_prepared"], request["command_values"]),
                         (label, True, proposal["values"]), label + "/exact admission command")
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
            verify_fallback_bytes(proof, directory, prior, home, checks)
            verify_seed(proof, proposal, checks)
            checks.equal((proof["passed"], proof["sha256"], proof["q10"], proof["q14"], proof["heading_key"],
                          proof["board_heading_offset_deg"], proof["board_twist_deg"]),
                         (True, request["prepared_connection_sha256"], request["selected_q10"], request["selected_q14"],
                          request["heading_key"], request["board_heading_offset_deg"], request["board_twist_deg"]),
                         label + "/verified exact arm paths and endpoint")
            ordered.append(branch_id)
    checks.equal(ordered, list(branches), identity + "/complete ordered admissions")
    return proposals, admitted


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
    verify_fallback_bytes(proof, directory, prior, home, checks)
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
            verify_seed(proof, proposal, checks)
            checks.equal(proof["targets"], original_proof["targets"],
                         branch_id + "/unchanged full board and release targets")
            ordered.append(branch_id)
    checks.equal(ordered, list(by_id), identity + "/complete ordered alternative admissions")
    return admitted



def summarize(layouts, slots, observations, wrappers, *, variants=METHODS):
    """Count no-setup positions in the denominator; timing includes startup/IO."""
    from .intergrid_route_selector_plan import _account
    from .planning_budget_plan import _wrap_choice

    _, methods, attempts = _account(layouts, slots, observations, variants)
    expected = [(layout["id"], variant) for layout in layouts for variant in variants]
    actual = [(item["site"]["layout"]["id"], item["variant"]) for item in wrappers]
    if actual != expected:
        raise ValueError("One ordered original plan receipt per placement and variant is required")
    indexed = {(item["site"]["layout"]["id"], item["variant"]): item["site"] for item in wrappers}
    by_id = {layout["id"]: layout for layout in layouts}
    for slot in slots:
        site = indexed[slot["context_id"], slot["method"]]
        if (site["layout"] != by_id[slot["context_id"]]
                or _wrap_choice(site["choice"], slot["method"]) != slot):
            raise ValueError("Each outer slot must exactly wrap its immutable inner choice")
    for variant, counts in methods.items():
        counts.update(
            positions=len(layouts), admitted=counts["releases"],
            flight_misses=counts["outcomes"].get("flight_miss", 0),
            no_candidate=len(layouts) - counts["releases"],
        )
    paired = dict(both=0, gained=0, lost=0, neither=0)
    positions = []
    for layout in layouts:
        outcomes = {variant: attempts[layout["id"], variant][1] for variant in variants}
        before, after = (outcomes[variant] in STANDARD_CAPTURES for variant in variants)
        paired["both" if before and after else "lost" if before else "gained" if after else "neither"] += 1
        positions.append(dict(id=layout["id"], bucket_xy_m=layout["bucket_xy_m"], outcomes=outcomes))
    return dict(
        methods=methods, paired=paired, positions=positions,
        latency={variant: latency([item["site"]["request_wall_s"] for item in wrappers
                                   if item["variant"] == variant]) for variant in variants},
        decision_latency={variant: latency([item["site"]["decision_latency_s"] for item in wrappers
                                            if item["variant"] == variant]) for variant in variants},
    )


def verify_slots(study, layouts, wrappers, manifest, checks, *, variants=METHODS):
    """Bind releases to the complete pre-native plan, not relabeled receipts."""
    from .planning_budget_plan import _wrap_choice

    artifact = read(study / "choices.json")
    bindings = artifact["bindings"]
    checks.require({"plans.json", "layouts.json", "neural-checkpoint.json", "contract.json"} <= set(bindings),
                   "Choices freeze positions, plans, exact model and optimization contract")
    verify_bindings(study, bindings, checks, "Pre-release selection")
    checks.require(all(manifest.get(name) == expected for name, expected in bindings.items()),
                   "Final evidence retains every pre-release selection dependency")
    slots = [_wrap_choice(item["site"]["choice"], item["variant"]) for item in wrappers]
    checks.equal(artifact["slots"], slots, "Outer choices exactly wrap original completed preparations")
    checks.equal([(slot["context_id"], slot["method"]) for slot in slots],
                 [(layout["id"], variant) for layout in layouts for variant in variants],
                 "One attempted drop per method at every fresh position")
    checks.equal(len({slot["id"] for slot in slots}), len(slots), "Unique outer attempted-shot IDs")
    for slot in slots:
        if slot["request"] is not None:
            request = slot["request"]
            checks.equal((request["id"], request["context_id"], request["method"], request["bucket_xy_m"]),
                         (slot["id"], slot["context_id"], slot["method"], slot["bucket_xy_m"]),
                         slot["id"] + "/exact outer release identity")
    receipt = read(study / "native-started.json")
    checks.require({"choices.json", "neural-checkpoint.json", "contract.json"} <= set(receipt["bindings"]),
                   "Native start binds frozen choices, model and optimization contract")
    verify_bindings(study, receipt["bindings"], checks, "Native start")
    checks.require(all(manifest.get(name) == expected for name, expected in receipt["bindings"].items()),
                   "Native start bindings remain in final evidence")
    expected = {"test_" + variant: [slot["request"] for slot in slots
                                   if slot["method"] == variant and slot["request"] is not None]
                for variant in variants}
    checks.equal(receipt["requests_sha256"], digest({variant: expected["test_" + variant] for variant in variants}),
                 "Every native request was frozen before test drops")
    return slots, expected


def verify_workers(wrappers, layouts, spec, checks, *, variants=METHODS):
    """Keep separate persistent worker lifecycles despite identical inner names."""
    order = [layout["id"] for layout in layouts]
    random.Random(spec["processing_seed"]).shuffle(order)
    checks.equal(spec["processing_order"], order, "Seeded processing order")
    checks.equal(spec["methods"], list(variants), "Independent baseline and optimized lanes")
    indexed = {(item["site"]["layout"]["id"], item["variant"]): item["site"] for item in wrappers}
    checks.equal(len(indexed), len(layouts) * len(variants), "One lifecycle per requested job")
    pool_ids, job_ids = set(), set()
    for variant in variants:
        previous, pool = None, None
        for identity in order:
            site = indexed[identity, variant]
            label = identity + "/" + variant + "/worker"
            generation, job = site["worker_generation"], site["worker_job_index"]
            checks.require(type(generation) is int and generation >= 0 and type(job) is int and job >= 0,
                           label + "/nonnegative generation and job")
            checks.require(type(site["worker_pid"]) is int and site["worker_pid"] > 0, label + "/process identity")
            checks.require(type(site["worker_reused"]) is bool, label + "/explicit reuse")
            checks.require(isinstance(site["job_id"], str) and site["job_id"] not in job_ids,
                           label + "/globally unique job")
            checks.equal(site["job_id"], f"{site['worker_pool_id']}-{generation}-{job}", label + "/bound pool identity")
            checks.equal(site["readiness_receipt"], None if site["worker_reused"] else
                         dict(kind="ready", pid=site["worker_pid"]), label + "/initialization receipt")
            job_ids.add(site["job_id"])
            if previous is None:
                pool = site["worker_pool_id"]
                checks.require(isinstance(pool, str) and bool(pool) and pool not in pool_ids,
                               label + "/separate method pool")
                pool_ids.add(pool)
                checks.equal((generation, job, site["worker_reused"], site["prior_worker_status"]),
                             (0, 0, False, None), label + "/first initialization")
            else:
                checks.equal(site["worker_pool_id"], pool, label + "/persistent variant lane")
                reuse = previous["status"] == "completed"
                checks.equal(site["worker_reused"], reuse, label + "/reuse only after clean completion")
                checks.equal(site["prior_worker_status"], previous["status"], label + "/previous status")
                if reuse:
                    checks.equal((generation, job, site["worker_pid"]),
                                 (previous["worker_generation"], previous["worker_job_index"] + 1, previous["worker_pid"]),
                                 label + "/next job in same process")
                    checks.equal(site["startup_s"], 0., label + "/reused job has no startup")
                else:
                    checks.equal((generation, job), (previous["worker_generation"] + 1, 0),
                                 label + "/new generation after timeout")
                    checks.require(site["worker_pid"] != previous["worker_pid"], label + "/old timed-out process replaced")
            previous = site

def audit_packets(study, parent, assets, spec, report, manifest, expected, prior, home, checks, *, variants=METHODS):
    """Reconstruct contacts, whole-ball retention and routes from native arrays."""
    from . import replay650_verify as replay
    from .cell1050_rebound_study import control_requests
    from .intergrid_route_labels import annotate_routes
    from .parallel_capture_grid_audit import observed_rows, rescore_packet, verify_release
    from .policy64_verify import audit_native_options
    from .table_training_verify import control_gate, rescore_rows

    calibration = replay.check_calibration(study, spec["identity"]["calibration"], checks)
    sources = read(parent / "control-replays.json")
    ledger, timings = observed_rows(study / "rows.jsonl"), observed_rows(study / "timing.jsonl")
    adapters = {"test_" + method: "native-" + method for method in variants}
    adapters.update({"diagnostic_" + method: "diagnostic/native-" + method for method in variants})
    observed = {phase: {} for phase in expected}
    used = {directory: set() for directory in adapters.values()}
    resolutions = {directory: {} for directory in adapters.values()}
    controls, identifiers, phases = {}, set(), []
    control_counts = Counter()
    for index, path in enumerate(sorted((study / "requests").glob("*.json"))):
        relative = f"requests/batch-{index:05d}.json"
        checks.equal(path.relative_to(study).as_posix(), relative, "Contiguous native packet ledger")
        packet = read(path)
        phase, requests = packet["phase"], packet["rows"]
        control = phase in {"controls_pre", "controls_post"}
        checks.require(control or phase in expected, relative + "/known native phase")
        directory = "native-controls" if control else adapters[phase]
        checks.equal((packet["control"], packet["adapter_directory"]), (control, directory),
                     relative + "/separate native adapter namespace")
        checks.require(0 < len(requests) <= 32, relative + "/bounded native batch")
        timing = next(timings, None)
        checks.require(timing is not None, relative + "/saved timing exists")
        checks.equal((timing["packet"], timing["phase"]), (relative, phase), relative + "/matching timing")
        if control:
            checks.equal(requests, control_requests(sources, phase), relative + "/exact matched controls")
            control_counts[phase] += 1
        else:
            start = len(observed[phase])
            checks.equal(requests, expected[phase][start:start + len(requests)],
                         relative + "/ordered choices frozen before release")
        phases.append(0 if phase == "controls_pre" else 2 if phase == "controls_post" else 1)
        rows = []
        for request in requests:
            row = next(ledger, None)
            checks.require(row is not None, request["id"] + "/saved observation exists")
            verify_release(row, dict(request, phase=phase), relative, manifest[relative], phase, checks)
            checks.require(row["id"] not in identifiers, row["id"] + "/unique native observation")
            identifiers.add(row["id"])
            checks.equal(row["adapter_directory"], directory, row["id"] + "/fixed adapter namespace")
            if not control:
                checks.equal((request["phase"], row["proposal_id"], row["method"]),
                             ("test", request["proposal_id"], request["method"]),
                             row["id"] + "/exact held-out prepared command")
                verify_heading(row, request, study / directory, spec["scene"], home, prior,
                               resolutions[directory], checks)
            rows.append(row)
        if control:
            controls[phase] = rows
        else:
            alias = {"native-heading/" + name[len(directory) + 1:]: value
                     for name, value in manifest.items() if name.startswith(directory + "/")}
            classes = list(rescore_packet(study / directory, rows, spec["scene"], calibration,
                                          alias, used[directory], checks))
            actual_routes = annotate_routes(copy.deepcopy(rows), study / directory)
            for row, actual, _ in zip(rows, actual_routes, classes, strict=True):
                checks.equal((row["shot_route"], row["table_rebounds_before_entry"]),
                             (actual["shot_route"], actual["table_rebounds_before_entry"]),
                             row["id"] + "/independently observed bounce route")
                observed[phase][row["id"]] = row
    checks.equal(phases, sorted(phases), "Controls bracket all frozen test attempts")
    checks.equal(control_counts, Counter(controls_pre=1, controls_post=1), "One pre and one post control block")
    checks.require(next(ledger, None) is None and next(timings, None) is None,
                   "No observations or timings outside frozen packets")
    for phase, requests in expected.items():
        checks.equal(list(observed[phase]), [row["id"] for row in requests], phase + "/all requests accounted for")
        directory = adapters[phase]
        if requests:
            audit_native_options(study / directory, spec["scene"], calibration, assets, checks)
        checks.equal(used[directory], {path.relative_to(study / directory).as_posix()
                                      for path in (study / directory).rglob("*.npz")},
                     directory + "/complete native array inventory")
    control_rows = [row for rows in controls.values() for row in rows]
    directory = study / "native-controls"
    audit_native_options(directory, spec["scene"], calibration, assets, checks)
    rescore_rows(directory, control_rows, spec["scene"], calibration,
                 spec["identity"]["trace_schema"], assets, checks)
    checks.equal({row["native_cache_file"] for row in control_rows},
                 {path.relative_to(directory).as_posix() for path in directory.rglob("*.npz")},
                 "Complete control array inventory")
    checks.equal(set(report["controls"]), set(controls), "No omitted or extra control gates")
    for phase, rows in controls.items():
        gate = control_gate(rows)
        checks.equal(gate["status"], "passed", phase + "/matched direct, table, and negative controls")
        checks.equal(report["controls"][phase], gate, phase + "/exact independently rescored control gate")
    return {key: row for phase in observed.values() for key, row in phase.items()}, len(control_rows)

def verify_plans(study, layouts, spec, assets, prior, home, manifest, checks, *, artifact_root=None, variants=METHODS, plan_hook=None):
    """Rebuild finite menus, checked prefixes, full proof bindings and choices."""
    from .clearance_expansion_plan import _empty, branch_proposals, needs_expansion
    from .intergrid_candidate_replay_plan import command_key
    from .intergrid_flight_ranker import FlightRanker
    from .intergrid_library_plan import libraries
    from .neural_shot_model import order
    from .neural_shot_plan import CHECK_BATCH, CHECK_LIMIT, _choice, menu

    artifact_root = study if artifact_root is None else artifact_root
    wrappers = read(artifact_root / "plans.json")
    expected = [(layout, method) for layout in layouts for method in variants]
    checks.equal([(item["site"]["layout"], item["variant"]) for item in wrappers], expected,
                 "Exact ordered preparation for every placement and both variants")
    checks.require(all(set(item) == {"variant", "site"} for item in wrappers), "Exact outer wrapper schema")
    checks.require(all(item["site"]["method"] == "warm_early" for item in wrappers),
                   "Original warm_early inner identity retained without rewriting receipts")
    raw = read(study / "shot-library-rows.json")
    libs = libraries(raw["original"], raw["additions"])
    source_checkpoint, checkpoint, history = (read(study / name) for name in
                                            ("route-models.json", "neural-checkpoint.json", "route-history.json"))
    ranker = FlightRanker(spec["scene"], read(study / "calibration.json")["score_config"], assets,
                         spec["flight_ranking"]["rule"])
    # Recompute from the independent physics implementation once per exact
    # command, then reuse only this audit's values across immutable prefixes.
    implementation, predictions = ranker.predict, {}
    def independent_prediction(request):
        key = command_key(request)
        if key not in predictions:
            predictions[key] = implementation(request)
        return predictions[key]
    ranker.predict = independent_prediction
    for method in variants:
        checks.equal({path.stem for path in (artifact_root / "preparation" / method).glob("*.json")},
                     {layout["id"] for layout in layouts}, method + "/exact preparation file inventory")
    for wrapper in wrappers:
        site = wrapper["site"]
        layout, method, plan = site["layout"], wrapper["variant"], site["plan"]
        identity = layout["id"] + "/" + method
        directory = artifact_root / ("native-" + method)
        checks.equal(site, read(artifact_root / "preparation" / method / (layout["id"] + ".json")),
                     identity + "/assembled plan retains exact per-job receipt")
        prefix = artifact_root.relative_to(study).as_posix()
        own_manifest = manifest if prefix == "." else {
            name[len(prefix) + 1:]: value for name, value in manifest.items() if name.startswith(prefix + "/")}
        snapshots = verify_deadline(site, directory, own_manifest, checks, spec["planning_budget_s"])
        proposals = menu(layout, libs, source_checkpoint)
        ordered = order(checkpoint, proposals)
        if plan_hook is not None:
            plan_hook(site=site, snapshots=snapshots, directory=directory, manifest=own_manifest,
                      checks=checks, spec=spec, variant=method, expected_menu=proposals, expected_order=ordered)
        checks.equal(set(plan), {"base", "fallback", "neural", "branches"}, identity + "/exact preparation stages")
        if not snapshots:
            verify_early_rule(site, snapshots, checks, spec["early_score_m"])
            continue
        resolutions = {}
        for stage in ("base", "fallback", "neural"):
            audit_preparation(plan[stage], layout, "test", directory, prior, home, resolutions, checks)
        # Check each accepted publication, not a command chosen after observing
        # a simulated outcome. Full endpoint proofs need only be checked once.
        for snapshot in snapshots:
            p, generated = snapshot["plan"], snapshot["generated"]
            for stage in p:
                # Every earlier publication is an immutable prefix of the final
                # accepted complete evidence; late candidate files do not enter it.
                for key in ("proposals", "admitted", "records"):
                    checks.equal(plan[stage][key][:len(p[stage][key])], p[stage][key],
                                 identity + "/immutable complete stage prefix " + stage + "/" + key)
            completed = [row["id"] for stage in ("base", "fallback", "neural", "branches")
                         for row in p[stage]["proposals"]]
            checks.equal(snapshot["completed_proposal_ids"], completed, identity + "/only complete candidates count")
            checks.require(set(generated) <= {"menu", "neural", "branches"}, identity + "/learned menu stages only")
            checks.equal(generated["menu"], proposals, identity + "/frozen pre-IK proposal menu")
            checks.equal(snapshot["ranked_proposal_ids"], [row["id"] for row in ordered],
                         identity + "/frozen checkpoint orders proposals before IK")
            checked = p["neural"]["proposals"]
            checks.require(len(checked) <= min(CHECK_LIMIT, len(ordered)), identity + "/bounded learned prefix")
            checks.equal(checked, ordered[:len(checked)], identity + "/no skipped candidate or hidden full-menu search")
            generated_neural = generated.get("neural", [])
            checks.equal(generated_neural, ordered[:len(generated_neural)], identity + "/generated checked prefix")
            checks.require(len(checked) <= len(generated_neural) <= min(CHECK_LIMIT, len(ordered))
                           and len(generated_neural) - len(checked) < CHECK_BATCH,
                           identity + "/only current bounded candidate batch may be incomplete")
            checks.require(not any(p[stage][key] for stage in ("base", "fallback") for key in p[stage]),
                           identity + "/learned planner does not borrow baseline admissions")
            for stop in range(CHECK_BATCH, len(checked), CHECK_BATCH):
                ids = {row["id"] for row in checked[:stop]}
                prefix = [row for row in p["neural"]["admitted"] if row["proposal_id"] in ids]
                checks.require(needs_expansion(_choice(layout, prefix, ranker, stop, snapshot["ranked_proposal_ids"])),
                               identity + "/more checks only after nonpositive predicted clearance")
            if "branches" in generated:
                checks.equal(len(checked), min(CHECK_LIMIT, len(ordered)),
                             identity + "/joint alternatives follow exhausted neural budget")
                initial = _choice(layout, p["neural"]["admitted"], ranker, len(checked), snapshot["ranked_proposal_ids"])
                checks.require(needs_expansion(initial) and bool(p["neural"]["admitted"]),
                               identity + "/alternative joints require an admitted poor-clearance shot")
                branch_plan = dict(base=p["neural"], fallback=_empty(), expansion=_empty(), branches=_empty())
                branches = branch_proposals(layout, branch_plan, ranker)
                checks.equal(generated["branches"], branches, identity + "/frozen alternative-arm menu")
                checks.equal(p["branches"]["proposals"], branches[:len(p["branches"]["proposals"])],
                             identity + "/only prefix of alternative-arm checks")
            else:
                checks.require(not any(p["branches"].values()), identity + "/no undeclared alternative arm checks")
            admitted = list({command_key(row): row for stage in ("neural", "branches")
                             for row in p[stage]["admitted"]}.values())
            choice = _choice(layout, admitted, ranker, len(checked), snapshot["ranked_proposal_ids"])
            choice = copy.deepcopy(choice)
            identifier = layout["id"] + "-" + site["method"]
            choice.update(id=identifier, method=site["method"])
            if choice["request"] is not None:
                choice["request"].update(id=identifier, method=site["method"])
            choice["predicted_flight_risk"] = needs_expansion(choice)
            checks.equal(snapshot["choice"], choice, identity + "/pre-deadline command independently reconstructed")
            if snapshot["active_stage"] == "finished":
                limit = min(CHECK_LIMIT, len(ordered))
                checks.require(len(checked) == limit or (len(checked) > 0 and len(checked) % CHECK_BATCH == 0),
                               identity + "/normal stop follows completed batch")
                initial = _choice(layout, p["neural"]["admitted"], ranker, len(checked), snapshot["ranked_proposal_ids"])
                checks.require(len(checked) == limit or not needs_expansion(initial),
                               identity + "/normal stop needs positive clearance or exhausted candidates")
                if needs_expansion(initial) and p["neural"]["admitted"]:
                    checks.require("branches" in generated, identity + "/required alternative joints considered")
                    checks.equal(p["branches"]["proposals"], generated["branches"],
                                 identity + "/normal completion finishes joint alternatives")
        verify_early_rule(site, snapshots, checks, spec["early_score_m"])
        audit_branches(plan["branches"], layout, {row["id"]: row for row in plan["neural"]["admitted"]},
                           directory, prior, home, resolutions, checks)
    worker_spec = spec
    if artifact_root != study:
        order = [layout["id"] for layout in layouts]
        random.Random(spec["processing_seed"]).shuffle(order)
        worker_spec = dict(spec, processing_order=order)
    verify_workers(wrappers, layouts, worker_spec, checks, variants=variants)
    return wrappers


def verify_layouts(args, spec, checks, *, layout_prefix="direct-path-first-test", extra_history=()):
    """Rebuild 20 placements from authenticated history and deterministic seeds."""
    from .interactive_fast_check import coordinates, fast_inventory
    from .interactive_physics_check import _digest, sample_positions
    from .intergrid_layouts import _placement_geometry
    from .intergrid_route_audit import verify_separation

    study = Path(args.output)
    artifact = read(study / "layouts.json")
    layouts, provenance = artifact["layouts"], artifact["provenance"]
    known, binding, fast_spec = fast_inventory(Path(args.fast_study), Path(args.fast_audit))
    checks.equal(spec["fast_parent"], binding, "Authenticated frozen fast-planner parent")
    checks.equal((spec["identity"], spec["scene"], spec["flight_ranking"]),
                 (fast_spec["identity"], fast_spec["scene"], fast_spec["flight_ranking"]),
                 "Same native scene, scoring identity and flight-ranking rule")
    acceptance = Path(args.acceptance_study)
    prior_report = read(acceptance / "report.json")
    prior_positions = read(acceptance / "positions.json")
    receipt = read(args.acceptance_audit)
    checks.equal((receipt["status"], prior_report["status"]), ("passed", "completed"),
                 "Earlier interactive acceptance completed and passed")
    checks.equal(receipt["results_sha256"], _digest(prior_report["results"]),
                 "Prior acceptance outcomes match independent receipt")
    checks.equal(receipt["positions_sha256"], sha(acceptance / "positions.json"),
                 "Prior acceptance coordinates remain unchanged")
    checks.equal(spec["acceptance_parent"], dict(
        report_sha256=sha(acceptance / "report.json"),
        positions_sha256=sha(acceptance / "positions.json"),
        audit_sha256=sha(args.acceptance_audit)), "Exact acceptance-parent bytes")
    speed = Path(args.speed_study)
    speed_report, speed_receipt = read(speed / "report.json"), read(args.speed_audit)
    checks.equal((speed_report["status"], speed_receipt["status"]), ("completed", "passed"),
                 "Previous optimized-check study completed and passed")
    checks.equal(speed_receipt["report_sha256"], sha(speed / "report.json"),
                 "Previous optimized-check report remains audited")
    checks.equal(spec["speed_parent"], dict(report_sha256=sha(speed / "report.json"),
                                          layouts_sha256=sha(speed / "layouts.json"),
                                          audit_sha256=sha(args.speed_audit)), "Exact optimized-check parent")
    known.update(coordinates(read(speed / "layouts.json")))
    known.update(coordinates(prior_positions))
    checks.equal((provenance["geometry"], provenance["grid"], provenance["bounds"]),
                 (prior_positions["geometry"], prior_positions["grid"], prior_positions["bounds"]),
                 "Same authenticated interactive workspace and allowed placements")
    checks.equal(provenance["geometry"], _placement_geometry(spec["scene"], args.assets),
                 "Independent table and fixed-base geometry")
    history = provenance["history_files"]
    required = {str(acceptance / "positions.json"), str(Path(args.fast_study) / "layouts.json"),
                str(speed / "layouts.json")}
    required.update(str(Path(name)) for name in extra_history)
    checks.require(isinstance(history, dict) and required <= set(history),
                   "Historical inventory includes both previous comparisons")
    allowed = {Path(name).resolve() for name in required}
    root = Path(args.evidence_root).resolve()
    for name, expected in history.items():
        path = Path(name)
        checks.require(path.resolve() in allowed or (
            path.resolve().is_relative_to(root)
            and path.name in {"request.json", "preview.json", "plan.json", "positions.json"}),
            "Declared history belongs to the recorded interactive evidence/" + name)
        checks.require(not path.resolve().is_relative_to(study.resolve()), "New study never contributes history")
        checks.equal(sha(path), expected, "Frozen historical bytes/" + name)
        known.update(coordinates(read(path)))
    exclusions = list(map(tuple, provenance["excluded_coordinates_m"]))
    checks.equal(exclusions, sorted(known), "All inventoried prior positions excluded without additions")
    checks.equal(provenance["excluded_coordinates_sha256"], digest(exclusions), "Historical coordinate binding")
    occupied = list(exclusions)
    reconstructed = []
    for offset in range(2):
        rows = sample_positions(provenance["geometry"], provenance["grid"], occupied,
                                provenance["bounds"], spec["seed"] + offset)
        for row in rows:
            identity = f"{layout_prefix}-{len(reconstructed):02d}"
            reconstructed.append(dict(row, id=identity, context_id=identity, split="test", stratum=row["region"]))
        occupied.extend(tuple(row["bucket_xy_m"]) for row in rows)
    checks.equal(len(layouts), 20, "20 fresh paired positions")
    checks.equal(layouts, reconstructed, "Placements independently reconstructed from frozen seeds/history")
    verify_separation(layouts, [], exclusions, checks)
    return layouts



def verify_diagnostic(study, spec, assets, prior, home, manifest, checks, *, variants=METHODS, plan_hook=None):
    """Audit the known difficult target separately from all fresh placements."""
    from .planning_budget_plan import _wrap_choice
    layout = dict(id="diagnostic-old-case02", context_id="diagnostic-old-case02", split="test",
                  stratum="development", bucket_xy_m=[0.9712010092480674, -0.2853881822068356])
    root = study / "diagnostic"
    wrappers = verify_plans(study, [layout], spec, assets, prior, home, manifest, checks, artifact_root=root,
                            variants=variants, plan_hook=plan_hook)
    artifact = read(root / "choices.json")
    checks.equal(artifact["bindings"], {"plans.json": sha(root / "plans.json")},
                 "Diagnostic choices bind the exact pre-native plans")
    slots = [_wrap_choice(item["site"]["choice"], item["variant"]) for item in wrappers]
    checks.equal(artifact["slots"], slots, "Diagnostic releases exactly wrap completed plan choices")
    receipt = read(study / "native-started.json")
    for name in ("diagnostic/plans.json", "diagnostic/choices.json"):
        checks.equal(receipt["bindings"].get(name), sha(study / name),
                     "Diagnostic choices frozen before every native drop/" + name)
    planned = {variant: [slot["request"] for slot in slots
                          if slot["method"] == variant and slot["request"] is not None] for variant in variants}
    checks.equal(receipt["diagnostic_requests_sha256"], digest(planned),
                 "Exact diagnostic requests frozen before native drops")
    for variant in variants:
        features = read(root / ("native-" + variant) / "arm-check-features.json")
        checks.equal(features, dict(variant=variant, features=spec["features"][variant]),
                     "Diagnostic lane uses declared planner features/" + variant)
    return layout, wrappers, slots, {"diagnostic_" + method: rows for method, rows in planned.items()}


def summarize_diagnostic(layout, wrappers, slots, observations):
    from .intergrid_route_selector_plan import outcome
    result = dict(bucket_xy_m=layout["bucket_xy_m"], methods={})
    for wrapper, slot in zip(wrappers, slots, strict=True):
        variant, site = wrapper["variant"], wrapper["site"]
        if slot["method"] != variant or slot["context_id"] != layout["id"]:
            raise ValueError("Diagnostic slot order or target changed")
        released = slot["request"] is not None
        result["methods"][variant] = dict(planning_s=site["request_wall_s"], checked_shot=released,
            outcome=outcome(observations[slot["id"]]) if released else "setup_rejected")
    if set(observations) != {slot["id"] for slot in slots if slot["request"] is not None}:
        raise ValueError("Unaccounted diagnostic observation")
    return result

def audit(args):
    """Validate finished plans and traces without training, simulation or writes."""
    from . import replay650_verify as replay
    from .cell1050_rebound_study import parent_inputs
    from .fast_planning_study import COPIED_INPUTS
    from .intergrid_route_labels import LABEL_VERSION

    study, parent, checks = Path(args.output), Path(args.parent_study), Checks()
    spec, report = read(study / "contract.json"), read(study / "report.json")
    fingerprint(spec, checks, "Frozen candidate-direct-path-first comparison")
    checks.equal((spec["version"], report["version"], report["status"], report["phase"]),
                 (STUDY_VERSION, STUDY_VERSION, "completed", "completed"), "Completed declared comparison")
    checks.equal((spec["physics_substeps"], spec["methods"], spec["label_version"],
                  spec["test_positions"], spec["planning_budget_s"], spec["attempts_per_method_per_position"]),
                 (64, list(METHODS), LABEL_VERSION, 20, 30., 1),
                 "Same scoring, decision budget and single-drop denominator")
    checks.equal((spec["early_score_m"], spec["seed"], spec["processing_seed"]),
                 (EARLY_SCORE_M, 2026112117, 2026112118), "Frozen early-return threshold and seeds")
    checks.equal(spec["features"], dict(
        baseline=["sircus-vectorized-arm-broad-phase-v1", "sircus-arm-path-rejection-short-circuit-v1"],
        direct_first=["sircus-vectorized-arm-broad-phase-v1", "sircus-arm-path-rejection-short-circuit-v1",
                      "sircus-candidate-direct-path-first-v1"]),
        "Only declared per-instance CPU optimizations differ")
    checks.require(0 < seconds(report["gpu_lease_s"]) <= spec["gpu_cap_s"] <= 3600,
                   "Native allocation remains within one GPU-hour")
    checks.equal(report.get("native_completed"), True, "Native comparison finished before audit")
    manifest = replay.check_manifest(study, checks)
    checks.equal(sha(study / "evidence-manifest.json"), report["evidence_manifest_sha256"],
                 "Complete immutable evidence inventory")
    required_inputs = set(COPIED_INPUTS) | {"setup-home.json", "layouts.json", "cpu-equivalence.json"}
    checks.require(required_inputs <= set(spec["frozen_inputs"]), "All models, geometry, layouts and controls frozen")
    verify_bindings(study, spec["frozen_inputs"], checks, "Original frozen study inputs")
    # An external verifier revision audits the originally frozen producer tree;
    # it must not rewrite that tree merely to repair audit reporting code.
    declared_source = getattr(args, "frozen_source_root", None)
    source = Path(declared_source).resolve() if declared_source is not None else Path(__file__).resolve().parents[2]
    checks.require(source.is_dir(), "Declared frozen producer source tree exists")
    source_names = set(spec["source_sha256"])
    checks.require({"sircus/mjlab/arm_check_speed.py", "sircus/mjlab/arm_path_speed.py",
                    "sircus/mjlab/direct_path_first.py", "sircus/mjlab/direct_path_first_study.py",
                    "sircus/mjlab/direct_path_first_audit.py"} <= source_names,
                   "Producer and independent audit sources frozen before preparation")
    for name, expected in spec["source_sha256"].items():
        checks.equal(sha(contained(source, name)), expected, "Exact frozen source/" + name)
    parent_spec, parent_binding, _ = parent_inputs(parent, args.parent_audit)
    checks.equal((spec["parent"], spec["identity"], spec["scene"]),
                 (parent_binding, parent_spec["identity"], parent_spec["scene"]), "Exact original calibrated native scene")
    checks.equal(sha(study / "setup-home.json"), sha(parent / "setup-home.json"),
                 "Canonical home unchanged from audited parent")
    fast_manifest = read(Path(args.fast_study) / "evidence-manifest.json")
    for name in COPIED_INPUTS:
        checks.equal(sha(Path(args.fast_study) / name), fast_manifest[name], "Audited parent input unchanged/" + name)
        checks.equal(sha(study / name), fast_manifest[name], "Same frozen model/library/calibration/" + name)
    equivalence = read(study / "cpu-equivalence.json")
    checks.equal(sha(study / "cpu-equivalence.json"), sha(args.equivalence_report),
                 "Exact CPU-equivalence prerequisite report")
    checks.equal((equivalence.get("exact_full_record_equivalence"), equivalence.get("mismatch_count"),
                  equivalence.get("segment_mismatch_count"), equivalence.get("cpu_only")),
                 (True, 0, 0, True), "No measured CPU check/certificate differences")
    checks.require(type(equivalence.get("samples")) is int and equivalence["samples"] >= 64,
                   "At least 64 native-scene CPU states compared")
    for variant in METHODS:
        features = read(study / ("native-" + variant) / "arm-check-features.json")
        checks.equal(features, dict(variant=variant, features=spec["features"][variant]),
                     variant + "/initialized worker feature receipt")
    layouts = verify_layouts(args, spec, checks)
    prior, home = read(study / "training-prior.json"), read(study / "setup-home.json")
    wrappers = verify_plans(study, layouts, spec, args.assets, prior, home, manifest, checks)
    slots, expected = verify_slots(study, layouts, wrappers, manifest, checks)
    diagnostic_layout, diagnostic_wrappers, diagnostic_slots, diagnostic_expected = verify_diagnostic(
        study, spec, args.assets, prior, home, manifest, checks)
    expected.update(diagnostic_expected)
    observations, controls = audit_packets(study, parent, args.assets, spec, report, manifest, expected, prior, home, checks)
    fresh_observations = {identity: row for identity, row in observations.items()
                          if not row["adapter_directory"].startswith("diagnostic/")}
    diagnostic_observations = {identity: row for identity, row in observations.items()
                               if row["adapter_directory"].startswith("diagnostic/")}
    checks.equal(report["progress"], dict(prepared=40, planned=40, completed_releases=len(fresh_observations),
                                         planned_releases=len(fresh_observations)),
                 "Every deadline and completed release accounted for")
    checks.equal(controls, 18, "Nine matched controls before and after all test drops")
    checks.equal(report["budgets"], dict(planning_s=30., positions=20), "Report reflects frozen denominator/budget")
    comparison = summarize(layouts, slots, fresh_observations, wrappers)
    checks.equal(report["comparison"], comparison, "Independent capture and timing summary matches report")
    diagnostic = summarize_diagnostic(diagnostic_layout, diagnostic_wrappers, diagnostic_slots, diagnostic_observations)
    checks.equal(report["diagnostic"], diagnostic, "Independent known-target result stays separate from fresh tests")
    return dict(version=VERSION, verifier_source_sha256=sha(Path(__file__)),
                frozen_source_root=str(source), status="passed", checks_count=checks.count,
                comparison=comparison, diagnostic=diagnostic,
                test_releases=len(fresh_observations), diagnostic_releases=len(diagnostic_observations),
                controls=controls, report_sha256=sha(study / "report.json"),
                choices_sha256=sha(study / "choices.json"), evidence_manifest_sha256=sha(study / "evidence-manifest.json"))
