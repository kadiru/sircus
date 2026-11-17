"""Read-only audit of frozen, deadline-limited one-drop planning comparisons."""
from __future__ import annotations

from collections import Counter
import copy
import math
from pathlib import Path
from statistics import median

from .benchmark_verify import Checks, contained, digest, fingerprint, read, sha

VERSION = "sircus-planning-budget-independent-audit-v1"
STUDY_VERSION = "sircus-planning-budget-study-v1"
METHODS = ("current_planner", "neural_selector")
STANDARD_CAPTURES = {"direct", "table", "unclassified_capture"}


def seconds(value):
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        raise ValueError("Timing must be finite and nonnegative")
    return value


def latency(values):
    values = sorted(seconds(value) for value in values)
    if not values:
        raise ValueError("Timing requires every frozen placement")
    return dict(positions=len(values), median_s=median(values),
                p90_s=values[math.ceil(.9 * len(values)) - 1], max_s=values[-1],
                aggregate_s=math.fsum(values))


def summarize(layouts, slots, observations, sites):
    """A timeout without a safe command stays in the first-drop denominator."""
    from .intergrid_candidate_replay_plan import command_key
    from .intergrid_route_selector_plan import _account

    identifiers, performance, attempts = _account(layouts, slots, observations, METHODS)
    expected = [(layout["id"], method) for layout in layouts for method in METHODS]
    if [(site["layout"]["id"], site["method"]) for site in sites] != expected:
        raise ValueError("Exactly one ordered preparation per placement and planner is required")
    by_layout = {layout["id"]: layout for layout in layouts}
    by_site = {(site["layout"]["id"], site["method"]): site for site in sites}
    for slot in slots:
        layout = by_layout[slot["context_id"]]
        site = by_site[slot["context_id"], slot["method"]]
        if site["layout"] != layout or slot["bucket_xy_m"] != layout["bucket_xy_m"]:
            raise ValueError("Frozen layout coordinates must match every site and slot")
        if site["choice"] != slot:
            raise ValueError("Slot must retain the exact completed planner choice")
        if type(site["deadline_reached"]) is not bool:
            raise ValueError("Deadline status must be explicit")
        request = slot["request"]
        if request is not None and (request["id"] != slot["id"]
                                   or request["context_id"] != layout["id"]
                                   or request["bucket_xy_m"] != layout["bucket_xy_m"]
                                   or request["method"] != slot["method"]):
            raise ValueError("Selected command identity and bucket must match its frozen slot")
    for method, counts in performance.items():
        selected = [observations[slot["id"]] for slot in slots
                    if slot["method"] == method and slot["request"] is not None]
        own_sites = [site for site in sites if site["method"] == method]
        counts.update(positions=len(layouts), admitted=len(selected), released=len(selected),
                      no_candidate=len(layouts) - len(selected),
                      deadline_reached=sum(site["deadline_reached"] for site in own_sites),
                      deadline_with_candidate=sum(site["deadline_reached"] and site["choice"]["request"] is not None
                                                  for site in own_sites),
                      deadline_no_candidate=sum(site["deadline_reached"] and site["choice"]["request"] is None
                                                for site in own_sites),
                      flight_misses=counts["outcomes"].get("flight_miss", 0),
                      robot_contacts=sum(row.get("experimental_route", {}).get("robot_contact_steps", 0) > 0
                                         for row in selected))
    paired = dict(both=0, gained=0, lost=0, neither=0)
    same = dict(positions=0, capture_disagreements=0, outcome_disagreements=0)
    positions = []
    for layout in layouts:
        identity = layout["id"]
        (left, lout), (right, rout) = (attempts[identity, method] for method in METHODS)
        lhit, rhit = lout in STANDARD_CAPTURES, rout in STANDARD_CAPTURES
        paired["both" if lhit and rhit else "lost" if lhit else "gained" if rhit else "neither"] += 1
        if left["request"] is not None and right["request"] is not None and (
                command_key(left["request"]) == command_key(right["request"])):
            same["positions"] += 1
            same["capture_disagreements"] += int(lhit != rhit)
            same["outcome_disagreements"] += int(lout != rout)
        positions.append(dict(id=identity, bucket_xy_m=layout["bucket_xy_m"],
                              outcomes=dict(zip(METHODS, (lout, rout), strict=True)),
                              deadline_reached={method: by_site[identity, method]["deadline_reached"]
                                                for method in METHODS}))
    return dict(methods=performance, paired=paired, same_command=same, positions=positions,
                latency={method: latency([site["latency_s"] for site in sites if site["method"] == method])
                         for method in METHODS},
                timing={method: {name: latency([site[key] if key is not None else
                                                site["startup_s"] + site["wall_planning_s"]
                                                for site in sites if site["method"] == method])
                                 for name, key in (("startup", "startup_s"), ("wall_planning", "wall_planning_s"),
                                                   ("stop_overhead", "stop_overhead_s"), ("end_to_end", None))}
                        for method in METHODS})


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


def verify_deadline(site, native, manifest, checks, budget_s=30.):
    """Only immutable complete snapshots received before the deadline can count."""
    import hashlib
    import json

    identity = site["layout"]["id"] + "/" + site["method"]
    checks.equal(site["budget_s"], budget_s, identity + "/same decision budget")
    for name in ("decision_latency_s", "wall_planning_s", "startup_s", "stop_overhead_s"):
        seconds(site[name])
    checks.equal(site["latency_s"], site["decision_latency_s"], identity + "/reported decision latency")
    checks.require(site["decision_latency_s"] <= budget_s, identity + "/bounded decision time")
    checks.require(site["wall_planning_s"] >= site["decision_latency_s"],
                   identity + "/shutdown cannot shorten decision time")
    checks.require(site["wall_planning_s"] + 1e-9 >= site["decision_latency_s"] + site["stop_overhead_s"],
                   identity + "/wall time includes the separately reported shutdown cost")
    checks.require(type(site["deadline_reached"]) is bool, identity + "/explicit deadline result")
    checks.equal(site["status"], "deadline" if site["deadline_reached"] else "completed",
                 identity + "/worker failures cannot masquerade as planning timeouts")
    if site["deadline_reached"]:
        checks.equal(site["decision_latency_s"], budget_s, identity + "/deadline uses full decision allowance")
    directory = native / "budget-journals" / site["layout"]["id"]
    checks.equal(Path(site["journal_directory"]).resolve(), directory.resolve(), identity + "/owned journal directory")
    previous, received, snapshots = [], -1., []
    for index, receipt in enumerate(site["accepted_receipts"]):
        label = identity + f"/snapshot-{index}"
        checks.equal((receipt["kind"], receipt["sequence"], receipt["file"]),
                     ("snapshot", index, f"snapshot-{index:05d}.json"), label + "/ordered immutable receipt")
        emitted, arrival = seconds(receipt["elapsed_s"]), seconds(receipt["received_elapsed_s"])
        checks.require(0 <= emitted <= arrival <= budget_s and received <= arrival,
                       label + "/complete snapshot emitted and received before deadline")
        received = arrival
        path = directory / receipt["file"]
        raw = path.read_bytes()
        checks.equal(hashlib.sha256(raw).hexdigest(), receipt["sha256"], label + "/published bytes unchanged")
        relative = path.relative_to(native.parent).as_posix()
        checks.equal(manifest[relative], receipt["sha256"], label + "/snapshot in final evidence manifest")
        snapshot = json.loads(raw)
        encoded = json.dumps(snapshot["choice"], sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        checks.equal(hashlib.sha256(encoded).hexdigest(), receipt["choice_sha256"], label + "/choice bound before deadline")
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
            checks.equal(latest["active_stage"], "finished", identity + "/completed planner exhausted its frozen rule")
    checks.equal(site["choices"], [site["choice"]], identity + "/one selected slot")
    return snapshots


def verify_plans(study, layouts, spec, assets, prior, home, manifest, checks):
    """Rebuild finite menus, checked prefixes, full proof bindings and choices."""
    from .clearance_expansion_audit import audit_branches
    from .clearance_expansion_plan import _empty, branch_proposals, current_selection, needs_expansion
    from .intergrid_candidate_replay_plan import command_key
    from .intergrid_flight_ranker import FlightRanker
    from .intergrid_library_plan import libraries
    from .intergrid_proposer_audit import audit_preparation
    from .intergrid_system_plan import base_proposals, fallback_proposals
    from .neural_shot_model import order
    from .neural_shot_plan import CHECK_BATCH, CHECK_LIMIT, _choice, menu

    sites = read(study / "plans.json")
    expected = [(layout, method) for layout in layouts for method in METHODS]
    checks.equal([(site["layout"], site["method"]) for site in sites], expected,
                 "Exact ordered preparation for every placement and both planners")
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
            continue
        resolutions = {}
        for stage in ("base", "fallback", "neural"):
            audit_preparation(plan[stage], layout, "test", directory, prior, home, resolutions, checks)
        # Check each accepted publication, not a command chosen after observing
        # a simulated outcome. Full endpoint proofs need only be checked once.
        base_menu = base_proposals(layout, libs, source_checkpoint) if method == METHODS[0] else []
        proposals = menu(layout, libs, source_checkpoint) if method == METHODS[1] else []
        ordered = order(checkpoint, proposals) if method == METHODS[1] else []
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
            if method == METHODS[0]:
                checks.require(set(generated) <= {"base", "fallback"}, identity + "/current menu stages only")
                checks.equal(generated["base"], base_menu, identity + "/unchanged baseline candidate menu")
                checks.equal(p["base"]["proposals"], base_menu[:len(p["base"]["proposals"])],
                             identity + "/baseline checks an ordered prefix")
                checks.require(not any(p[stage][key] for stage in ("neural", "branches") for key in p[stage]),
                               identity + "/baseline cannot use learned or alternative-arm menus")
                if "fallback" in generated:
                    checks.equal(p["base"]["proposals"], base_menu, identity + "/fallback follows complete base checks")
                    fallback = fallback_proposals(layout, libs, base_menu, p["base"]["admitted"])
                    checks.equal(generated["fallback"], fallback, identity + "/unchanged setup-only fallback menu")
                    checks.equal(p["fallback"]["proposals"], fallback[:len(p["fallback"]["proposals"])],
                                 identity + "/fallback checks an ordered prefix")
                else:
                    checks.require(not any(p["fallback"].values()), identity + "/no undeclared fallback checks")
                choice = copy.deepcopy(current_selection(layout, p, history, ranker))
                identifier = layout["id"] + "-" + method
                choice.update(id=identifier, method=method)
                if choice["request"] is not None:
                    choice["request"].update(id=identifier, method=method)
            else:
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
            choice["predicted_flight_risk"] = needs_expansion(choice)
            checks.equal(snapshot["choice"], choice, identity + "/pre-deadline command independently reconstructed")
        if method == METHODS[1]:
            audit_branches(plan["branches"], layout, {row["id"]: row for row in plan["neural"]["admitted"]},
                           directory, prior, home, resolutions, checks)
    return sites


def audit(args):
    """Validate finished evidence without launching native simulation or fitting."""
    from . import replay650_verify as replay
    from .cell1050_rebound_study import parent_inputs
    from .intergrid_layouts import MIN_SEPARATION_M, _placement_geometry
    from .intergrid_route_audit import verify_history_files, verify_separation
    from .intergrid_route_labels import LABEL_VERSION
    from .planning_budget_study import COPIED_INPUTS, fresh_layouts

    study, parent, checks = Path(args.output), Path(args.parent_study), Checks()
    spec, report = read(study / "contract.json"), read(study / "report.json")
    fingerprint(spec, checks, "Frozen planning-budget comparison")
    checks.equal((spec["version"], report["status"], report["phase"]),
                 (STUDY_VERSION, "completed", "completed"), "Completed declared comparison")
    checks.equal((spec["physics_substeps"], spec["methods"], spec["label_version"],
                  spec["test_positions"], spec["planning_budget_s"], spec["attempts_per_method_per_position"]),
                 (64, list(METHODS), LABEL_VERSION, 100, 30., 1), "Fixed scoring, deadline and one-drop budget")
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
    archive = read(study / "layouts.json")
    layouts, provenance = archive["layouts"], archive["provenance"]
    checks.equal(len(layouts), 100, "100 fresh held-out positions")
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
    checks.equal(layouts, fresh_layouts(exclusions, geometry, read(parent / "grid.json"), seed=spec["seed"]),
                 "New placements reconstructed from frozen geometry and seed")
    verify_separation(layouts, prior_positions, exclusions, checks)
    prior, home = read(study / "training-prior.json"), read(parent / "setup-home.json")
    sites = verify_plans(study, layouts, spec, args.assets, prior, home, manifest, checks)
    slots, expected = verify_slots(study, layouts, sites, manifest, checks)
    observed, controls = audit_packets(study, parent, args.assets, spec, report, manifest, expected, prior, home, checks)
    checks.equal(report["progress"], dict(prepared=200, planned=200, completed_releases=len(observed),
                                         planned_releases=len(observed)), "Every timeout and completed release accounted for")
    checks.equal(report["budgets"], dict(planning_s=30., positions=100), "UI reflects exact frozen budgets")
    comparison = summarize(layouts, slots, observed, sites)
    checks.equal(report["comparison"], comparison, "Independent outcome and latency summary matches final report")
    return dict(version=VERSION, status="passed", checks_count=checks.count, comparison=comparison,
                test_releases=len(observed), controls=controls, report_sha256=sha(study / "report.json"),
                choices_sha256=sha(study / "choices.json"), evidence_manifest_sha256=sha(study / "evidence-manifest.json"))
