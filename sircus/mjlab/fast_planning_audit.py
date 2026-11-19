"""Independent evidence audit for cold, warm and early-return shot planning."""
from __future__ import annotations

from collections import Counter
import copy
import math
from pathlib import Path
from statistics import median

from .benchmark_verify import Checks, contained, digest, fingerprint, read, sha

VERSION = "sircus-fast-planning-independent-audit-v1"
STUDY_VERSION = "sircus-fast-planning-study-v1"
METHODS = ("cold_30", "warm_30", "warm_early")
PROFILE_STAGES = ("menu", "neural_order", "arm_setup_checks", "physics_ranking", "evidence_publication", "other")
EARLY_SCORE_M = .005
PAIRS = {"warm_vs_cold": ("cold_30", "warm_30"),
         "early_vs_warm": ("warm_30", "warm_early"),
         "early_vs_cold": ("cold_30", "warm_early")}
STANDARD_CAPTURES = {"direct", "table", "unclassified_capture"}


def seconds(value):
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        raise ValueError("Timing must be finite and nonnegative")
    return value


def latency(values):
    values = sorted(seconds(value) for value in values)
    if not values:
        raise ValueError("Timing requires every frozen placement")
    return dict(positions=len(values), mean_s=math.fsum(values) / len(values), median_s=median(values),
                p90_s=values[math.ceil(.9 * len(values)) - 1], max_s=values[-1],
                aggregate_s=math.fsum(values))


def summarize(layouts, slots, observations, sites):
    """Count every placement, including no safe shot within the deadline."""
    from .intergrid_candidate_replay_plan import command_key
    from .intergrid_route_selector_plan import _account

    _, performance, attempts = _account(layouts, slots, observations, METHODS)
    expected = [(layout["id"], method) for layout in layouts for method in METHODS]
    if [(site["layout"]["id"], site["method"]) for site in sites] != expected:
        raise ValueError("Exactly one ordered preparation per placement and method is required")
    by_layout = {layout["id"]: layout for layout in layouts}
    by_site = {(site["layout"]["id"], site["method"]): site for site in sites}
    for slot in slots:
        layout = by_layout[slot["context_id"]]
        site = by_site[slot["context_id"], slot["method"]]
        if site["layout"] != layout or slot["bucket_xy_m"] != layout["bucket_xy_m"] or site["choice"] != slot:
            raise ValueError("Frozen layouts and complete choices must match each slot")
        if type(site["deadline_reached"]) is not bool:
            raise ValueError("Deadline status must be explicit")
        request = slot["request"]
        if request is not None and (request["id"], request["context_id"], request["bucket_xy_m"], request["method"]) != (
                slot["id"], layout["id"], layout["bucket_xy_m"], slot["method"]):
            raise ValueError("Selected command identity must match the frozen slot")
    for method, counts in performance.items():
        selected = [observations[slot["id"]] for slot in slots
                    if slot["method"] == method and slot["request"] is not None]
        own = [site for site in sites if site["method"] == method]
        counts.update(positions=len(layouts), admitted=len(selected), released=len(selected),
                      no_candidate=len(layouts) - len(selected),
                      deadline_reached=sum(site["deadline_reached"] for site in own),
                      deadline_with_candidate=sum(site["deadline_reached"] and site["choice"]["request"] is not None
                                                  for site in own),
                      deadline_no_candidate=sum(site["deadline_reached"] and site["choice"]["request"] is None
                                                for site in own),
                      early_return=sum(site["return_reason"] == "early_good_shot" for site in own),
                      flight_misses=counts["outcomes"].get("flight_miss", 0),
                      robot_contacts=sum(row.get("experimental_route", {}).get("robot_contact_steps", 0) > 0
                                         for row in selected))
    paired, same = {}, {}
    for name, (before, after) in PAIRS.items():
        pair = dict(both=0, gained=0, lost=0, neither=0)
        shared = dict(positions=0, capture_disagreements=0, outcome_disagreements=0)
        for layout in layouts:
            (left, lout), (right, rout) = (attempts[layout["id"], method] for method in (before, after))
            lhit, rhit = lout in STANDARD_CAPTURES, rout in STANDARD_CAPTURES
            pair["both" if lhit and rhit else "lost" if lhit else "gained" if rhit else "neither"] += 1
            if left["request"] is not None and right["request"] is not None and (
                    command_key(left["request"]) == command_key(right["request"])):
                shared["positions"] += 1
                shared["capture_disagreements"] += int(lhit != rhit)
                shared["outcome_disagreements"] += int(lout != rout)
        paired[name], same[name] = pair, shared
    timings = (("startup", "startup_s"), ("wall_planning", "wall_planning_s"),
               ("stop_overhead", "stop_overhead_s"), ("receipt_verification", "receipt_verification_s"),
               ("request", "request_wall_s"))
    profile, workers = {}, {}
    for method in METHODS:
        own = [site for site in sites if site["method"] == method]
        total = math.fsum(seconds(site["profile"]["decision_observed_s"]) for site in own)
        stages = {}
        for key in PROFILE_STAGES:
            aggregate = math.fsum(seconds(site["profile"]["stage_seconds"][key]) for site in own)
            stages[key] = dict(aggregate_s=aggregate, mean_s=aggregate / len(own),
                               fraction=aggregate / total if total else 0.)
        profile[method] = dict(stages=stages, decision_observed_s=total,
                               interrupted_requests=sum(site["profile"]["interrupted_stage"] is not None for site in own),
                               interrupted_s=math.fsum(site["profile"]["interrupted_stage_s"] for site in own))
        starts = sum(not site["worker_reused"] for site in own)
        workers[method] = dict(cold_starts=starts, reused_requests=len(own) - starts,
                               restarts=starts - 1 if method != "cold_30" else 0)
    return dict(methods=performance, paired=paired, same_command=same,
                positions=[dict(id=layout["id"], bucket_xy_m=layout["bucket_xy_m"],
                                outcomes={method: attempts[layout["id"], method][1] for method in METHODS},
                                deadline_reached={method: by_site[layout["id"], method]["deadline_reached"] for method in METHODS})
                           for layout in layouts],
                latency={method: latency([site["latency_s"] for site in sites if site["method"] == method]) for method in METHODS},
                timing={method: {name: latency([site[key] for site in sites if site["method"] == method])
                                 for name, key in timings} for method in METHODS}, profile=profile, workers=workers)


def verify_bindings(study, bindings, checks, label):
    checks.require(isinstance(bindings, dict) and bool(bindings), label + "/nonempty immutable bindings")
    for name, expected in bindings.items():
        checks.equal(sha(contained(study, name)), expected, label + "/" + name)


def verify_slots(study, layouts, sites, manifest, checks):
    artifact = read(study / "choices.json")
    bindings = artifact["bindings"]
    checks.require({"plans.json", "layouts.json", "neural-checkpoint.json"} <= set(bindings),
                   "Choices freeze layouts, full plans and the learned checkpoint")
    verify_bindings(study, bindings, checks, "Pre-release selection")
    checks.require(all(manifest[name] == expected for name, expected in bindings.items()),
                   "Final evidence retains every pre-release selection dependency")
    slots = [site["choice"] for site in sites]
    checks.equal(artifact["slots"], slots, "Exact choices from all deadline-limited preparations")
    checks.equal([(slot["context_id"], slot["method"]) for slot in slots],
                 [(layout["id"], method) for layout in layouts for method in METHODS],
                 "One attempted drop per method at every new position")
    checks.equal(len({slot["id"] for slot in slots}), len(slots), "Globally unique attempted shot IDs")
    receipt = read(study / "native-started.json")
    checks.require({"choices.json", "neural-checkpoint.json"} <= set(receipt["bindings"]),
                   "Native start binds exact frozen choices and model")
    verify_bindings(study, receipt["bindings"], checks, "Native start")
    checks.require(all(manifest[name] == expected for name, expected in receipt["bindings"].items()),
                   "Start bindings remain in final evidence")
    expected = {"test_" + method: [slot["request"] for slot in slots
                                  if slot["method"] == method and slot["request"] is not None]
                for method in METHODS}
    checks.equal(receipt["requests_sha256"], digest({method: expected["test_" + method] for method in METHODS}),
                 "Native requests frozen before all test drops")
    return slots, expected


def audit_packets(study, parent, assets, spec, report, manifest, expected, prior, home, checks):
    """Reconstruct contacts, whole-ball retention and routes from native arrays."""
    from . import replay650_verify as replay
    from .cell1050_rebound_audit import verify_heading
    from .cell1050_rebound_study import control_requests
    from .intergrid_route_labels import annotate_routes
    from .parallel_capture_grid_audit import observed_rows, rescore_packet, verify_release
    from .policy64_verify import audit_native_options
    from .table_training_verify import control_gate, rescore_rows

    calibration = replay.check_calibration(study, spec["identity"]["calibration"], checks)
    sources = read(parent / "control-replays.json")
    ledger, timings = observed_rows(study / "rows.jsonl"), observed_rows(study / "timing.jsonl")
    adapters = {"test_" + method: "native-" + method for method in METHODS}
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


def verify_profile(site, checks):
    """Account for the entire decision, including work interrupted at cutoff."""
    label = site["layout"]["id"] + "/" + site["method"] + "/profile"
    profile = site["profile"]
    checks.equal(set(profile["stage_seconds"]), set(PROFILE_STAGES), label + "/exact stage names")
    observed = seconds(profile["decision_observed_s"])
    checks.numeric(observed, site["decision_latency_s"], label + "/decision time fully accounted", tolerance=1e-6)
    previous, totals = 0., dict.fromkeys(PROFILE_STAGES, 0.)
    blocks = profile["blocks"]
    checks.require(bool(blocks), label + "/time blocks present")
    events = profile["events"]
    checks.require(bool(events), label + "/received stage ledger present")
    checks.equal(events[0], dict(stage="other", elapsed_s=0.), label + "/ledger begins at job start")
    checks.equal(len(events), len(blocks), label + "/every transition has one interval")
    for index, block in enumerate(blocks):
        stage, start, end = block["stage"], seconds(block["start_s"]), seconds(block["end_s"])
        checks.require(stage in PROFILE_STAGES, label + "/known block stage")
        checks.equal(events[index], dict(stage=stage, elapsed_s=start), label + "/interval matches event")
        checks.numeric(end, events[index + 1]["elapsed_s"] if index + 1 < len(events) else observed,
                       label + "/interval ends at next event or cutoff", tolerance=1e-7)
        checks.numeric(block["duration_s"], end - start, label + "/reported duration", tolerance=1e-7)
        checks.equal(block["interrupted"], bool(site["deadline_reached"] and index == len(blocks) - 1),
                     label + "/only cutoff interval interrupted")
        checks.numeric(start, previous, label + f"/block-{index} contiguous", tolerance=1e-7)
        checks.require(end >= start and end <= observed + 1e-7, label + "/ordered bounded intervals")
        totals[stage] += end - start
        previous = end
    checks.numeric(previous, observed, label + "/coverage reaches decision cutoff", tolerance=1e-7)
    for stage, duration in totals.items():
        checks.numeric(seconds(profile["stage_seconds"][stage]), duration,
                       label + "/exclusive sum " + stage, tolerance=1e-6)
    checks.numeric(math.fsum(totals.values()), observed, label + "/no missing or overlapping time", tolerance=1e-6)
    if site["deadline_reached"]:
        checks.equal(profile["interrupted_stage"], blocks[-1]["stage"], label + "/unfinished stage retained")
        checks.numeric(seconds(profile["interrupted_stage_s"]), blocks[-1]["end_s"] - blocks[-1]["start_s"],
                       label + "/unfinished work duration included", tolerance=1e-7)
    else:
        checks.equal((profile["interrupted_stage"], profile["interrupted_stage_s"]),
                     (None, 0.), label + "/clean completion has no interrupted work")


def verify_workers(sites, layouts, spec, checks):
    """A warm process can follow only its own successfully completed previous job."""
    import random

    order = [layout["id"] for layout in layouts]
    random.Random(spec["processing_seed"]).shuffle(order)
    checks.equal(spec["processing_order"], order, "Frozen seeded processing order")
    checks.equal(spec["methods"], list(METHODS), "Three independent method lanes")
    checks.equal(len(sites), len(layouts) * len(METHODS), "One lifecycle per requested job")
    indexed = {(site["layout"]["id"], site["method"]): site for site in sites}
    checks.equal(len(indexed), len(sites), "No duplicate lifecycle job")
    pool_ids, job_ids = set(), set()
    for method in METHODS:
        previous, pool = None, None
        for identity in order:
            site = indexed[identity, method]
            label = identity + "/" + method + "/worker"
            generation, job = site["worker_generation"], site["worker_job_index"]
            checks.require(type(generation) is int and generation >= 0 and type(job) is int and job >= 0,
                           label + "/finite nonnegative generation and index")
            checks.require(type(site["worker_pid"]) is int and site["worker_pid"] > 0, label + "/recorded process identity")
            checks.require(type(site["worker_reused"]) is bool, label + "/explicit reuse status")
            checks.require(isinstance(site["job_id"], str) and site["job_id"] not in job_ids, label + "/unique job identity")
            checks.equal(site["job_id"], f"{site['worker_pool_id']}-{generation}-{job}", label + "/bound pool generation and job")
            checks.equal(site["readiness_receipt"], None if site["worker_reused"] else
                         dict(kind="ready", pid=site["worker_pid"]), label + "/initialization readiness receipt")
            job_ids.add(site["job_id"])
            if pool is None:
                pool = site["worker_pool_id"]
                checks.require(isinstance(pool, str) and bool(pool) and pool not in pool_ids,
                               label + "/independent method worker pool")
                pool_ids.add(pool)
                checks.equal((generation, job, site["worker_reused"], site["prior_worker_status"]),
                             (0, 0, False, None), label + "/first job initializes a process")
            else:
                checks.equal(site["worker_pool_id"], pool, label + "/persistent method lane")
                reuse = method != "cold_30" and previous["status"] == "completed"
                checks.equal(site["worker_reused"], reuse, label + "/reuse only after clean completion")
                checks.equal(site["prior_worker_status"], previous["status"], label + "/previous status retained")
                if reuse:
                    checks.equal((generation, job, site["worker_pid"]),
                                 (previous["worker_generation"], previous["worker_job_index"] + 1, previous["worker_pid"]),
                                 label + "/same initialized process and next job")
                    checks.equal(site["startup_s"], 0., label + "/reused job has no initialization charge")
                else:
                    checks.equal((generation, job), (previous["worker_generation"] + 1, 0),
                                 label + "/new generation after cold request or timeout")
                    checks.require(site["worker_pid"] != previous["worker_pid"], label + "/owned timed out process replaced")
            previous = site


def verify_early_rule(site, snapshots, checks, threshold=EARLY_SCORE_M):
    """Use independently reconstructed snapshots, never a claimed good-shot flag."""
    from .clearance_expansion_plan import needs_expansion

    label = site["layout"]["id"] + "/" + site["method"]
    checks.require(site["return_reason"] in {"early_good_shot", "completed", "deadline"}, label + "/known return reason")
    if site["method"] != "warm_early":
        checks.equal(site["return_reason"], "deadline" if site["deadline_reached"] else "completed",
                     label + "/non-early method keeps its frozen stop rule")
        return
    first = None
    previous_count = 0
    for snapshot in snapshots:
        count = len(snapshot["completed_proposal_ids"])
        checks.require(previous_count <= count <= previous_count + 1,
                       label + "/early eligibility checked after each completed candidate")
        previous_count = count
        prediction = snapshot["choice"].get("prediction") or {}
        score = prediction.get("score_m")
        eligible = (type(score) in (int, float) and math.isfinite(score) and score >= threshold
                    and not needs_expansion(snapshot["choice"]))
        if first is None and eligible:
            first = snapshot
        if first is not None:
            checks.equal(snapshot["completed_proposal_ids"], first["completed_proposal_ids"],
                         label + "/no later checks after first qualifying completion")
            checks.equal(snapshot["choice"], first["choice"], label + "/earliest qualifying choice retained")
    if first is None:
        checks.require(site["return_reason"] != "early_good_shot", label + "/early return needs independently positive margin")
    else:
        # A complete early-good snapshot can precede cutoff while its done
        # receipt arrives just after it. Keep the checked shot but report the
        # real deadline; this cannot authorize another candidate check.
        checks.equal(site["return_reason"], "deadline" if site["deadline_reached"] else "early_good_shot",
                     label + "/first qualifying shot ends search")




def verify_deadline(site, native, manifest, checks, budget_s=30.):
    """Only immutable complete snapshots received before the deadline can count."""
    import hashlib
    import json

    identity = site["layout"]["id"] + "/" + site["method"]
    checks.equal(site["budget_s"], budget_s, identity + "/same decision budget")
    for name in ("decision_latency_s", "wall_planning_s", "startup_s", "stop_overhead_s",
                 "receipt_verification_s", "request_wall_s"):
        seconds(site[name])
    checks.equal(site["latency_s"], site["decision_latency_s"], identity + "/reported decision latency")
    checks.require(site["decision_latency_s"] <= budget_s, identity + "/bounded decision time")
    checks.require(site["wall_planning_s"] >= site["decision_latency_s"],
                   identity + "/shutdown cannot shorten decision time")
    checks.require(site["wall_planning_s"] + 1e-9 >= site["decision_latency_s"] + site["stop_overhead_s"],
                   identity + "/wall time includes the separately reported shutdown cost")
    checks.require(site["request_wall_s"] + 1e-6 >= site["startup_s"] + site["wall_planning_s"]
                   + site["receipt_verification_s"], identity + "/request includes startup, planning and verification")
    checks.require(type(site["deadline_reached"]) is bool, identity + "/explicit deadline result")
    checks.equal(site["status"], "deadline" if site["deadline_reached"] else "completed",
                 identity + "/worker failures cannot masquerade as planning timeouts")
    if site["deadline_reached"]:
        checks.equal(site["decision_latency_s"], budget_s, identity + "/deadline uses full decision allowance")
        checks.equal(site["completion_receipt"], None, identity + "/no on-time completion at deadline")
    else:
        completion = site["completion_receipt"]
        checks.require(isinstance(completion, dict), identity + "/completion receipt required before reuse")
        checks.equal((completion["kind"], completion["job_id"], completion["return_reason"]),
                     ("done", site["job_id"], site["return_reason"]), identity + "/same-job completion receipt")
        checks.equal(completion["received_elapsed_s"], site["decision_latency_s"], identity + "/decision ends on receipt")
        checks.require(0 <= seconds(completion["elapsed_s"]) <= completion["received_elapsed_s"] <= budget_s,
                       identity + "/complete before decision cutoff")
    directory = native / "fast-journals" / site["layout"]["id"]
    checks.equal(Path(site["journal_directory"]).resolve(), directory.resolve(), identity + "/owned journal directory")
    previous, received, snapshots = [], -1., []
    for index, receipt in enumerate(site["accepted_receipts"]):
        label = identity + f"/snapshot-{index}"
        checks.equal((receipt["kind"], receipt["sequence"], receipt["file"]),
                     ("snapshot", index, f"snapshot-{index:05d}.json"), label + "/ordered immutable receipt")
        checks.equal(receipt["job_id"], site["job_id"], label + "/receipt belongs to current job")
        emitted, arrival = seconds(receipt["elapsed_s"]), seconds(receipt["received_elapsed_s"])
        checks.require(0 <= emitted <= arrival <= budget_s and received <= arrival,
                       label + "/complete snapshot emitted and received before deadline")
        received = arrival
        checks.require(arrival <= site["decision_latency_s"], label + "/publication precedes final decision")
        path = directory / receipt["file"]
        raw = path.read_bytes()
        checks.equal(hashlib.sha256(raw).hexdigest(), receipt["sha256"], label + "/published bytes unchanged")
        relative = path.relative_to(native.parent).as_posix()
        checks.equal(manifest[relative], receipt["sha256"], label + "/snapshot in final evidence manifest")
        snapshot = json.loads(raw)
        encoded = json.dumps(snapshot["choice"], sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        checks.equal(hashlib.sha256(encoded).hexdigest(), receipt["choice_sha256"], label + "/choice bound before deadline")
        profile_events = snapshot["profile_events_before_publication"]
        checks.equal(site["profile"]["events"][:len(profile_events)], profile_events,
                     label + "/immutable profiling prefix retained")
        checks.require(all(seconds(event["elapsed_s"]) <= emitted for event in profile_events),
                       label + "/profile prefix predates publication")
        completed = snapshot["completed_proposal_ids"]
        checks.equal((snapshot["layout"], snapshot["method"]), (site["layout"], site["method"]),
                     label + "/exact placement and planner")
        checks.equal((receipt["completed_proposals"], len(set(completed))), (len(completed), len(completed)),
                     label + "/complete distinct candidate inventory")
        checks.equal(completed[:len(previous)], previous, label + "/completed prefix cannot be rewritten")
        previous = completed
        snapshots.append(snapshot)
    if not snapshots:
        checks.equal(site["status"], "deadline", identity + "/normal completion publishes a final snapshot")
        checks.equal((site["accepted_snapshot_file"], site["accepted_snapshot_sha256"], site["choice"]["request"]),
                     (None, None, None), identity + "/no unreceived candidate may release")
        checks.equal(site["completed_proposal_ids"], [], identity + "/no candidate completion without receipt")
        checks.require(all(not value for stage in site["plan"].values() for value in stage.values()),
                       identity + "/no unreceived preparation evidence used")
        checks.equal(site["generated"], {}, identity + "/no unreceived menu used")
    else:
        latest, receipt = snapshots[-1], site["accepted_receipts"][-1]
        checks.equal((site["accepted_snapshot_file"], site["accepted_snapshot_sha256"]),
                     (receipt["file"], receipt["sha256"]), identity + "/last complete snapshot selected")
        for key, expected in latest.items():
            checks.equal(site[key], expected, identity + "/selected complete snapshot field " + key)
        if site["status"] == "completed":
            checks.equal(latest["active_stage"], "early_good_shot" if site["return_reason"] == "early_good_shot" else "finished",
                         identity + "/completed planner follows its frozen stop rule")
    checks.equal(site["choices"], [site["choice"]], identity + "/one selected slot")
    verify_profile(site, checks)
    return snapshots



def verify_plans(study, layouts, spec, assets, prior, home, manifest, checks):
    """Rebuild finite menus, checked prefixes, full proof bindings and choices."""
    from .clearance_expansion_audit import audit_branches
    from .clearance_expansion_plan import _empty, branch_proposals, needs_expansion
    from .intergrid_candidate_replay_plan import command_key
    from .intergrid_flight_ranker import FlightRanker
    from .intergrid_library_plan import libraries
    from .intergrid_proposer_audit import audit_preparation
    from .neural_shot_model import order
    from .neural_shot_plan import CHECK_BATCH, CHECK_LIMIT, _choice, menu

    sites = read(study / "plans.json")
    expected = [(layout, method) for layout in layouts for method in METHODS]
    checks.equal([(site["layout"], site["method"]) for site in sites], expected,
                 "Exact ordered preparation for every placement and all three methods")
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
    for method in METHODS:
        checks.equal({path.stem for path in (study / "preparation" / method).glob("*.json")},
                     {layout["id"] for layout in layouts}, method + "/exact preparation file inventory")
    for site in sites:
        layout, method, plan = site["layout"], site["method"], site["plan"]
        identity = layout["id"] + "/" + method
        directory = study / ("native-" + method)
        checks.equal(site, read(study / "preparation" / method / (layout["id"] + ".json")),
                     identity + "/assembled plan retains exact per-job receipt")
        snapshots = verify_deadline(site, directory, manifest, checks, spec["planning_budget_s"])
        checks.equal(set(plan), {"base", "fallback", "neural", "branches"}, identity + "/exact preparation stages")
        if not snapshots:
            verify_early_rule(site, snapshots, checks, spec["early_score_m"])
            continue
        resolutions = {}
        for stage in ("base", "fallback", "neural"):
            audit_preparation(plan[stage], layout, "test", directory, prior, home, resolutions, checks)
        # Check each accepted publication, not a command chosen after observing
        # a simulated outcome. Full endpoint proofs need only be checked once.
        proposals = menu(layout, libs, source_checkpoint)
        ordered = order(checkpoint, proposals)
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
            identifier = layout["id"] + "-" + method
            choice.update(id=identifier, method=method)
            if choice["request"] is not None:
                choice["request"].update(id=identifier, method=method)
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
    verify_workers(sites, layouts, spec, checks)
    return sites



def audit(args):
    """Validate finished evidence without launching native simulation or fitting."""
    from . import replay650_verify as replay
    from .cell1050_rebound_study import parent_inputs
    from .intergrid_layouts import MIN_SEPARATION_M, _placement_geometry
    from .intergrid_route_audit import verify_history_files, verify_separation
    from .intergrid_route_labels import LABEL_VERSION
    from .fast_planning_study import COPIED_INPUTS, fresh_layouts

    study, parent, checks = Path(args.output), Path(args.parent_study), Checks()
    spec, report = read(study / "contract.json"), read(study / "report.json")
    fingerprint(spec, checks, "Frozen fast-planning comparison")
    checks.equal((spec["version"], report["status"], report["phase"]),
                 (STUDY_VERSION, "completed", "completed"), "Completed declared comparison")
    checks.equal((spec["physics_substeps"], spec["methods"], spec["label_version"],
                  spec["test_positions"], spec["planning_budget_s"], spec["attempts_per_method_per_position"]),
                 (64, list(METHODS), LABEL_VERSION, 50, 30., 1), "Fixed scoring, deadline and one-drop budget")
    checks.equal((spec["early_score_m"], spec["seed"], spec["processing_seed"], spec["workers"]),
                 (EARLY_SCORE_M, 2026113002, 2026113003, 3), "Frozen early threshold, seeds and one worker per lane")
    parent_spec, parent_binding, _ = parent_inputs(parent, args.parent_audit)
    checks.equal((spec["parent"], spec["identity"], spec["scene"]),
                 (parent_binding, parent_spec["identity"], parent_spec["scene"]), "Exact audited native scene")
    checks.require(0 < seconds(report["gpu_lease_s"]) <= spec["gpu_cap_s"] <= 3600,
                   "Native allocation remains within one GPU-hour")
    manifest = replay.check_manifest(study, checks)
    checks.equal(sha(study / "evidence-manifest.json"), report["evidence_manifest_sha256"], "Complete evidence manifest")
    verify_bindings(study, spec["frozen_inputs"], checks, "Frozen original inputs")
    source = Path(__file__).resolve().parents[2]
    for name, expected in spec["source_sha256"].items():
        checks.equal(sha(contained(source, name)), expected, "Exact producer source/" + name)
    original, audit_receipt = Path(args.neural_study), read(args.neural_audit)
    original_report = read(original / "report.json")
    original_manifest = read(original / "evidence-manifest.json")
    checks.equal((audit_receipt["status"], original_report["status"]), ("passed", "completed"),
                 "Frozen selector derives from completed audited training")
    checks.equal(audit_receipt["report_sha256"], sha(original / "report.json"), "Prior report unchanged")
    checks.equal(audit_receipt["evidence_manifest_sha256"], sha(original / "evidence-manifest.json"),
                 "Prior evidence inventory unchanged")
    checks.equal(audit_receipt["comparison"], original_report["comparison"], "Prior published results agree with audit")
    checks.equal(spec["neural_parent"], dict(study_report_sha256=sha(original / "report.json"),
                 audit_sha256=sha(args.neural_audit), checkpoint_sha256=sha(original / "neural-checkpoint.json"),
                 layouts_sha256=sha(original / "layouts.json")), "Exact prior comparison and checkpoint binding")
    original_spec = read(original / "contract.json")
    checks.equal((spec["identity"], spec["scene"], spec["flight_ranking"]),
                 (original_spec["identity"], original_spec["scene"], original_spec["flight_ranking"]),
                 "Frozen physics prediction rule and simulator scene are unchanged")
    for name in (*COPIED_INPUTS, "layouts.json", "contract.json"):
        checks.equal(sha(original / name), original_manifest[name], "Prior audited artifact/" + name)
        if name in COPIED_INPUTS:
            checks.equal(sha(study / name), original_manifest[name], "Unchanged frozen comparison input/" + name)
    budget = Path(args.budget_study)
    budget_receipt, budget_report = read(args.budget_audit), read(budget / "report.json")
    budget_manifest = read(budget / "evidence-manifest.json")
    checks.equal((budget_receipt["status"], budget_report["status"]), ("passed", "completed"),
                 "Prior bounded comparison completed and independently passed")
    checks.equal(budget_receipt["report_sha256"], sha(budget / "report.json"), "Budget parent report unchanged")
    checks.equal(budget_receipt["evidence_manifest_sha256"], sha(budget / "evidence-manifest.json"),
                 "Budget parent evidence unchanged")
    checks.equal(budget_receipt["comparison"], budget_report["comparison"], "Budget parent results match audit")
    checks.equal(spec["budget_parent"], dict(study_report_sha256=sha(budget / "report.json"),
                 audit_sha256=sha(args.budget_audit), layouts_sha256=sha(budget / "layouts.json"),
                 checkpoint_sha256=sha(budget / "neural-checkpoint.json")), "Exact prior budget study binding")
    for name in (*COPIED_INPUTS, "layouts.json", "contract.json"):
        checks.equal(sha(budget / name), budget_manifest[name], "Budget parent audited input/" + name)
        if name in COPIED_INPUTS:
            checks.equal(sha(study / name), budget_manifest[name], "Same frozen budget-comparison input/" + name)
    archive = read(study / "layouts.json")
    layouts, provenance = archive["layouts"], archive["provenance"]
    checks.equal(len(layouts), 50, "50 fresh held-out positions")
    checks.equal(provenance["minimum_separation_m"], MIN_SEPARATION_M, "Frozen spatial separation threshold")
    geometry = _placement_geometry(spec["scene"], args.assets)
    checks.equal(provenance["geometry"], geometry, "Independent table and fixed-base geometry")
    checks.equal(provenance["layouts_sha256"], digest(layouts), "Frozen new test positions")
    exclusions = list(map(tuple, provenance["excluded_coordinates_m"]))
    checks.equal(exclusions, sorted(set(exclusions)), "Sorted distinct historical exclusions")
    checks.equal(provenance["excluded_coordinates_sha256"], digest(exclusions), "Exact historical exclusion binding")
    verify_history_files(provenance, args.history_root, study, checks)
    prior_positions = [row["bucket_xy_m"] for split in read(original / "layouts.json").values() for row in split["layouts"]]
    checks.equal(len(prior_positions), 400, "All previous neural train, validation and test positions retained")
    checks.require(set(map(tuple, prior_positions)) <= set(exclusions), "Every earlier neural position excluded")
    budget_positions = [row["bucket_xy_m"] for row in read(budget / "layouts.json")["layouts"]]
    checks.equal(len(budget_positions), 100, "All 100 previous bounded test positions retained")
    checks.require(set(map(tuple, budget_positions)) <= set(exclusions), "Every earlier bounded test position excluded")
    prior_positions += budget_positions
    checks.equal(layouts, fresh_layouts(exclusions, geometry, read(parent / "grid.json"), seed=spec["seed"]),
                 "New placements reconstructed from frozen geometry and seed")
    verify_separation(layouts, prior_positions, exclusions, checks)
    prior, home = read(study / "training-prior.json"), read(parent / "setup-home.json")
    sites = verify_plans(study, layouts, spec, args.assets, prior, home, manifest, checks)
    slots, expected = verify_slots(study, layouts, sites, manifest, checks)
    observed, controls = audit_packets(study, parent, args.assets, spec, report, manifest, expected, prior, home, checks)
    checks.equal(report["progress"], dict(prepared=150, planned=150, completed_releases=len(observed),
                                         planned_releases=len(observed)), "Every timeout and completed release accounted for")
    checks.equal(report["budgets"], dict(planning_s=30., positions=50), "UI reflects exact frozen budgets")
    comparison = summarize(layouts, slots, observed, sites)
    checks.equal(report["comparison"], comparison, "Independent outcome and latency summary matches final report")
    return dict(version=VERSION, status="passed", checks_count=checks.count, comparison=comparison,
                test_releases=len(observed), controls=controls, report_sha256=sha(study / "report.json"),
                choices_sha256=sha(study / "choices.json"), evidence_manifest_sha256=sha(study / "evidence-manifest.json"))
