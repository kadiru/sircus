"""Independent audit of reference-first planning with a bounded stalled-path fallback.

The original receipt, full accepted-path, immutable menu, native64 and scoring
checks are reused with explicit dependencies. No historical module globals or
saved study artifacts are rewritten by this verifier.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

from . import direct_path_first_audit as shared
from .benchmark_verify import Checks, contained, digest, fingerprint, read, sha
from .fast_planning_audit import EARLY_SCORE_M, seconds, verify_bindings

VERSION = "sircus-stalled-reference-independent-audit-v1"
STUDY_VERSION = "sircus-stalled-reference-study-v1"
METHODS = ("baseline", "hybrid")


def summarize(layouts, slots, observations, wrappers):
    return shared.summarize(layouts, slots, observations, wrappers, variants=METHODS)


def verify_layouts(args, spec, checks):
    """Require the previous direct-first experiment and exclude its 20 positions."""
    previous = Path(args.direct_study)
    report, receipt = read(previous / "report.json"), read(args.direct_audit)
    checks.equal((report["status"], receipt["status"]), ("completed", "passed"),
                 "Previous candidate-first comparison completed and independently passed")
    checks.equal(receipt["report_sha256"], sha(previous / "report.json"),
                 "Previous candidate-first results remain independently bound")
    checks.equal(receipt["choices_sha256"], sha(previous / "choices.json"),
                 "Previous candidate-first frozen choices remain independently bound")
    checks.equal(spec["direct_parent"], dict(report_sha256=sha(previous / "report.json"),
                                          layouts_sha256=sha(previous / "layouts.json"),
                                          audit_sha256=sha(args.direct_audit)),
                 "Exact candidate-first parent binding")
    return shared.verify_layouts(args, spec, checks, layout_prefix="stalled-reference-test",
                                 extra_history=(previous / "layouts.json",))


def verify_plans(study, layouts, spec, assets, prior, home, manifest, checks, *, artifact_root=None):
    return shared.verify_plans(study, layouts, spec, assets, prior, home, manifest, checks,
                               artifact_root=artifact_root, variants=METHODS, plan_hook=verify_switch)


def _compact_digest(value):
    """Independent encoding used by immutable worker receipts."""
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                      allow_nan=False).encode()).hexdigest()


def verify_switch(*, site, snapshots, directory, manifest, checks, spec, variant,
                  expected_menu, expected_order):
    """Require an interrupted reference search before any direct-first choice."""
    from .calibrated_admission import VERSION as REFERENCE_VERSION
    from .direct_path_first import VERSION as DIRECT_VERSION

    if variant == "baseline":
        checks.require("stalled_reference" not in site, "Baseline has no fallback mechanism")
        checks.require(all("stalled_reference_snapshot" not in row for row in snapshots),
                       "Baseline immutable snapshots remain unchanged")
        for stage in site["plan"].values():
            for request in stage["admitted"]:
                record = read(contained(directory, request["context_resolution_file"]))
                checks.equal(record["version"], REFERENCE_VERSION, "Baseline keeps reference-first admission")
        return
    label = site["layout"]["id"] + "/hybrid"
    receipt = site["stalled_reference"]
    events, files = receipt["events"], receipt["receipt_files"]
    checks.require(bool(events) and len(events) == len(files), label + "/complete durable event chain")
    extras = {"events", "receipt_files", "selected_adapter", "frozen_choice_sha256", "accepted_snapshot_sha256"}
    checks.equal({key: value for key, value in receipt.items() if key not in extras}, events[-1],
                 label + "/outer summary equals the last immutable event")
    checks.equal(receipt["frozen_choice_sha256"], _compact_digest(site["choice"]),
                 label + "/fallback receipt binds exact selected choice")
    checks.equal(receipt["accepted_snapshot_sha256"], site["accepted_snapshot_sha256"],
                 label + "/fallback receipt binds accepted immutable snapshot")
    prefix = dict(menu_sha256=_compact_digest(expected_menu), menu_count=len(expected_menu),
                  ranked_proposal_ids=[row["id"] for row in expected_order],
                  ranked_proposal_ids_sha256=_compact_digest([row["id"] for row in expected_order]))
    version = "sircus-stalled-reference-fallback-v1"
    budget, threshold = spec["planning_budget_s"], spec["reference_path_s"]
    previous_hash, previous_elapsed, previous_intervals, active = None, -1., [], None
    trigger, fallback_start, last_published = None, None, 0
    frozen_prefix = False
    directory = Path(directory)
    expected_files = []
    known_events = {"query_started", "selection_prefix_frozen", "reference_path_started",
                    "reference_path_finished", "reference_path_interrupted", "fallback_triggered",
                    "fallback_started", "decision_budget_exhausted_before_fallback",
                    "decision_budget_exhausted_after_initialization", "query_completed"}
    for index, (event, file) in enumerate(zip(events, files, strict=True)):
        event_label = label + f"/event-{index}"
        relative = f"stalled-reference-receipts/{site['job_id']}/event-{index:05d}.json"
        checks.equal(file["file"], relative, event_label + "/ordered job-owned event path")
        path = contained(directory, relative)
        checks.equal(sha(path), file["sha256"], event_label + "/immutable event bytes")
        checks.equal(read(path), event, event_label + "/event summary matches original bytes")
        expected_files.append(path)
        checks.equal(manifest[path.relative_to(directory.parent).as_posix()], file["sha256"],
                     event_label + "/event retained in final manifest")
        checks.equal((event["version"], event["job_id"], event["layout_id"], event["layout_sha256"],
                      event["sequence"], event["previous_event_sha256"], event["method"]),
                     (version, site["job_id"], site["layout"]["id"], _compact_digest(site["layout"]),
                      index, previous_hash, "warm_early"), event_label + "/exact same-job hash-chain binding")
        checks.equal((event["decision_budget_s"], event["reference_path_budget_s"],
                      event["selection_uses_flight_labels"]), (budget, threshold, False),
                     event_label + "/one original decision clock and outcome-blind trigger")
        checks.equal(event["initialization_policy"],
                     "Reference worker startup; first direct engine initialization consumes remaining decision time",
                     event_label + "/no extra initialization decision allowance")
        elapsed = seconds(event["elapsed_s"])
        checks.require(previous_elapsed <= elapsed <= site["wall_planning_s"] + 1e-6,
                       event_label + "/monotonic timing within owned process lifetime")
        checks.numeric(event["remaining_budget_s"], max(0., budget-elapsed), event_label + "/same remaining budget")
        checks.require(event["event"] in known_events, event_label + "/known lifecycle event")
        if index == 0:
            checks.equal(event["event"], "query_started", event_label + "/query begins before path work")
        else:
            checks.require(event["event"] != "query_started", event_label + "/no second decision clock")
        if event["event"] == "selection_prefix_frozen":
            checks.require(not frozen_prefix and index == 1, event_label + "/menu frozen once before path search")
            frozen_prefix = True
        checks.equal(event["selection_prefix"], prefix if frozen_prefix else None,
                     event_label + "/identical independently rebuilt menu and order across switch")
        intervals = event["reference_path_intervals"]
        checks.equal(intervals[:len(previous_intervals)], previous_intervals,
                     event_label + "/completed reference intervals never rewritten")
        checks.require(len(previous_intervals) <= len(intervals) <= len(previous_intervals)+1,
                       event_label + "/one path completion per event")
        end = 0.
        for position, interval in enumerate(intervals):
            start, stop = seconds(interval["start_s"]), seconds(interval["end_s"])
            checks.require(interval["index"] == position and end <= start <= stop <= elapsed,
                           event_label + "/ordered nonoverlapping actual reference-path intervals")
            checks.numeric(interval["duration_s"], stop-start, event_label + "/measured path duration")
            checks.require(type(interval["completed"]) is bool and type(interval["interrupted"]) is bool
                           and interval["completed"] != interval["interrupted"],
                           event_label + "/interruption cannot claim a completed reference search")
            blocks = site["profile"]["blocks"]
            clipped_stop = min(stop, site["decision_latency_s"])
            checks.require(any(block["stage"] == "arm_setup_checks" and
                               block["start_s"] <= start <= clipped_stop <= block["end_s"] + 1e-6
                               for block in blocks), event_label + "/reference timer excludes other planning phases")
            end = stop
        checks.numeric(event["reference_path_s"], math.fsum(row["duration_s"] for row in intervals),
                       event_label + "/reference-path budget counts only recorded intervals")
        if event["event"] == "reference_path_started":
            checks.require(frozen_prefix and active is None and trigger is None,
                           event_label + "/reference phase starts before any switch")
            active = event["active_reference_path"]
            checks.require(active is not None and active["index"] == len(intervals)
                           and end <= seconds(active["start_s"]) <= elapsed,
                           event_label + "/durable active reference phase")
        elif event["event"] in {"reference_path_finished", "reference_path_interrupted"}:
            checks.require(active is not None and len(intervals) == len(previous_intervals)+1,
                           event_label + "/path completion follows a durable start")
            checks.equal({key: intervals[-1][key] for key in ("index", "start_s")}, active,
                         event_label + "/timed path start is unchanged")
            checks.equal(intervals[-1]["interrupted"], event["event"] == "reference_path_interrupted",
                         event_label + "/explicit interrupted path event")
            active = None
        else:
            checks.equal(intervals, previous_intervals, event_label + "/no hidden path intervals")
        checks.equal(event["active_reference_path"], active, event_label + "/active phase state")
        interrupted = any(row["interrupted"] for row in intervals)
        checks.equal(event["reference_search_interrupted"], interrupted, event_label + "/honest incomplete-search state")
        checks.require(sum(row["interrupted"] for row in intervals) <= 1, event_label + "/at most one interruption")
        if not interrupted:
            checks.require(event["reference_path_s"] < threshold, event_label + "/uninterrupted path budget not exhausted")
        if event["event"] == "fallback_triggered":
            checks.require(trigger is None and interrupted and active is None and last_published == 0,
                           event_label + "/fallback follows actual stall before any completed candidate")
            trigger = event["fallback"]
            checks.equal((trigger["reason"], trigger["threshold_s"], trigger["published_before_switch"],
                          trigger["reference_search_complete"]),
                         ("reference_path_budget_exhausted", threshold, 0, False),
                         event_label + "/interruption never masquerades as exhaustive reference failure")
            checks.numeric(trigger["reference_path_s"], event["reference_path_s"], event_label + "/actual trigger path time")
            checks.require(threshold <= seconds(trigger["reference_path_s"]) <= seconds(trigger["trigger_elapsed_s"])
                           and end <= trigger["trigger_elapsed_s"] <= elapsed,
                           event_label + "/trigger happens only after the measured path budget")
            checks.numeric(trigger["remaining_budget_s"], max(0., budget-trigger["trigger_elapsed_s"]),
                           event_label + "/fallback receives only remaining original time")
        checks.equal(event["fallback"], trigger, event_label + "/immutable outcome-independent trigger")
        if event["event"] == "fallback_started":
            checks.require(trigger is not None and fallback_start is None and trigger["trigger_elapsed_s"] < budget,
                           event_label + "/single fallback after stall within original budget")
            fallback_start = elapsed
        mode = "direct_first" if fallback_start is not None else "reference"
        checks.equal(event["engine_mode"], mode, event_label + "/engine changes only at the recorded switch")
        checks.require(type(event["direct_engine_initialized_this_query"]) is bool,
                       event_label + "/explicit initialization state")
        initialization = seconds(event["direct_engine_initialization_s"])
        checks.require(initialization <= elapsed, event_label + "/initialization charged inside total elapsed time")
        published = event["published_snapshots"]
        checks.require(type(published) is int and published >= last_published,
                       event_label + "/monotonic completed publication count")
        if trigger is not None and fallback_start is None:
            checks.equal(published, 0, event_label + "/no candidate completed before switching")
        if event["event"] == "query_completed":
            checks.equal(index, len(events)-1, event_label + "/completion terminates event chain")
        previous_hash, previous_elapsed, previous_intervals, last_published = file["sha256"], elapsed, intervals, published
    folder = directory / "stalled-reference-receipts" / site["job_id"]
    checks.equal(sorted(folder.glob("event-*.json")), expected_files, label + "/no omitted committed events")
    if site["status"] == "completed":
        checks.equal(events[-1]["event"], "query_completed", label + "/completed job has a terminal durable event")
        checks.equal(last_published, len(snapshots), label + "/all completed publications accepted before clean finish")
    final_mode = events[-1]["engine_mode"]
    for snapshot, publication in zip(snapshots, site["accepted_receipts"], strict=True):
        checks.equal(snapshot["stalled_reference_snapshot"], dict(version=version, job_id=site["job_id"],
                     engine_mode=final_mode, selection_prefix=prefix, fallback=trigger),
                     label + "/accepted snapshot binds the unchanged engine and trigger")
        if final_mode == "direct_first":
            checks.require(fallback_start is not None and fallback_start <= publication["elapsed_s"] <= budget,
                           label + "/no direct-first accepted snapshot before the actual stall and switch")
    requests = [row for stage in site["plan"].values() for row in stage["admitted"]]
    for request in requests:
        resolution = read(contained(directory, request["context_resolution_file"]))
        checks.equal(resolution["version"], DIRECT_VERSION if final_mode == "direct_first" else REFERENCE_VERSION,
                     label + "/every admitted full path belongs to the declared engine")
    selected = final_mode if site["choice"]["request"] is not None else None
    checks.equal(receipt["selected_adapter"], selected, label + "/native adapter follows the frozen selected proof")



def audit(args):
    """Read saved evidence only; never run drops, train, acquire a GPU or edit it."""
    from . import replay650_verify as replay
    from .cell1050_rebound_study import parent_inputs
    from .fast_planning_study import COPIED_INPUTS
    from .intergrid_route_labels import LABEL_VERSION

    study, parent, checks = Path(args.output), Path(args.parent_study), Checks()
    spec, report = read(study / "contract.json"), read(study / "report.json")
    fingerprint(spec, checks, "Frozen stalled-reference comparison")
    checks.equal((spec["version"], report["version"], report["status"], report["phase"]),
                 (STUDY_VERSION, STUDY_VERSION, "completed", "completed"), "Completed declared comparison")
    checks.equal((spec["physics_substeps"], spec["methods"], spec["label_version"],
                  spec["test_positions"], spec["planning_budget_s"], spec["attempts_per_method_per_position"]),
                 (64, list(METHODS), LABEL_VERSION, 20, 30., 1),
                 "Same scoring, total decision budget and single-drop denominator")
    checks.equal((spec["early_score_m"], spec["seed"], spec["processing_seed"], spec["reference_path_s"]),
                 (EARLY_SCORE_M, 2026112127, 2026112128, 5.),
                 "Frozen early-return, spatial seeds and reference-path stall threshold")
    from .stalled_reference_plan import FEATURES
    checks.equal(spec["features"], dict(
        baseline=["sircus-vectorized-arm-broad-phase-v1", "sircus-arm-path-rejection-short-circuit-v1"],
        hybrid=list(FEATURES)), "Only declared planner features differ")
    checks.require(0 < seconds(report["gpu_lease_s"]) <= spec["gpu_cap_s"] <= 3600,
                   "Native allocation remains within one GPU-hour")
    checks.equal(report.get("native_completed"), True, "Native comparison finished before audit")
    manifest = replay.check_manifest(study, checks)
    checks.equal(sha(study / "evidence-manifest.json"), report["evidence_manifest_sha256"],
                 "Complete immutable evidence inventory")
    required_inputs = set(COPIED_INPUTS) | {"setup-home.json", "layouts.json", "cpu-equivalence.json"}
    checks.require(required_inputs <= set(spec["frozen_inputs"]), "All models, geometry, layouts and controls frozen")
    verify_bindings(study, spec["frozen_inputs"], checks, "Original frozen study inputs")
    declared_source = getattr(args, "frozen_source_root", None)
    source = Path(declared_source).resolve() if declared_source is not None else Path(__file__).resolve().parents[2]
    checks.require(source.is_dir(), "Declared frozen producer source tree exists")
    checks.require({"sircus/mjlab/arm_check_speed.py", "sircus/mjlab/arm_path_speed.py",
                    "sircus/mjlab/direct_path_first.py", "sircus/mjlab/direct_path_first_audit.py",
                    "sircus/mjlab/stalled_reference_plan.py", "sircus/mjlab/stalled_reference_study.py",
                    "sircus/mjlab/stalled_reference_audit.py"} <= set(spec["source_sha256"]),
                   "Producer and independent reusable audit sources frozen before preparation")
    for name, expected in spec["source_sha256"].items():
        checks.equal(sha(contained(source, name)), expected, "Exact frozen source/" + name)
    parent_spec, parent_binding, _ = parent_inputs(parent, args.parent_audit)
    checks.equal((spec["parent"], spec["identity"], spec["scene"]),
                 (parent_binding, parent_spec["identity"], parent_spec["scene"]),
                 "Exact original calibrated native scene")
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
    slots, expected = shared.verify_slots(study, layouts, wrappers, manifest, checks, variants=METHODS)
    diagnostic_layout, diagnostic_wrappers, diagnostic_slots, diagnostic_expected = shared.verify_diagnostic(
        study, spec, args.assets, prior, home, manifest, checks, variants=METHODS, plan_hook=verify_switch)
    expected.update(diagnostic_expected)
    observations, controls = shared.audit_packets(study, parent, args.assets, spec, report, manifest,
                                                  expected, prior, home, checks, variants=METHODS)
    fresh_observations = {identity: row for identity, row in observations.items()
                          if not row["adapter_directory"].startswith("diagnostic/")}
    diagnostic_observations = {identity: row for identity, row in observations.items()
                               if row["adapter_directory"].startswith("diagnostic/")}
    checks.equal(report["progress"], dict(prepared=40, planned=40, completed_releases=len(fresh_observations),
                                         planned_releases=len(fresh_observations)),
                 "Every deadline and completed release accounted for")
    checks.equal(controls, 18, "Nine matched controls before and after all test drops")
    checks.equal(report["budgets"], dict(planning_s=30., positions=20, reference_path_s=5.), "Report reflects frozen denominator/budget")
    comparison = summarize(layouts, slots, fresh_observations, wrappers)
    checks.equal(report["comparison"], comparison, "Independent capture and timing summary matches report")
    diagnostic = shared.summarize_diagnostic(diagnostic_layout, diagnostic_wrappers,
                                             diagnostic_slots, diagnostic_observations)
    checks.equal(report["diagnostic"], diagnostic, "Known-target result stays separate from fresh tests")
    checks.equal(spec["promotion_time_ratio"], 1.10, "Predeclared UI promotion time tolerance")
    before, after = (comparison["methods"][variant] for variant in METHODS)
    before_time, after_time = (comparison["latency"][variant]["mean_s"] for variant in METHODS)
    gates = dict(captures_preserved=after["captures"] >= before["captures"],
                 mean_planning_within_tolerance=after_time <= 1.10 * before_time,
                 old_target_captured=diagnostic["methods"]["hybrid"]["outcome"] in shared.STANDARD_CAPTURES)
    decision = dict(eligible=all(gates.values()), gates=gates, mean_time_ratio_limit=1.10,
                    requires_independent_audit=True)
    checks.equal(report["decision"], decision, "Independent promotion decision uses frozen capture/time gates")
    return dict(version=VERSION, verifier_source_sha256=sha(Path(__file__)),
                frozen_source_root=str(source), status="passed", checks_count=checks.count,
                comparison=comparison, diagnostic=diagnostic, decision=decision, test_releases=len(fresh_observations),
                diagnostic_releases=len(diagnostic_observations), controls=controls,
                report_sha256=sha(study / "report.json"), choices_sha256=sha(study / "choices.json"),
                evidence_manifest_sha256=sha(study / "evidence-manifest.json"))
