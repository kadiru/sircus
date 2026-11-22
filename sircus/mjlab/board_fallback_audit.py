"""Read-only audit of the frozen board-pose fallback comparison.

Rebuilds deterministic target selection and predictions, verifies immutable
pre-deadline choices and full pose/path certificates, then independently scores
all saved native64 contacts and retained-ball trajectories. No native execution.
"""
from __future__ import annotations
import copy
import hashlib
import json
import math
from pathlib import Path
import random

from .benchmark_verify import Checks, contained, digest, fingerprint, read, sha
from .direct_path_first_audit import (audit_preparation, audit_branches, verify_workers,
    _proof, verify_seed)
from .fast_planning_audit import verify_bindings, verify_deadline, verify_early_rule, seconds

VERSION = "sircus-board-fallback-independent-audit-v1"
STUDY_VERSION = "sircus-board-fallback-study-v1"
METHODS = ("baseline", "fallback")

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
    from .fixed35_broad_native import reach_models
    ranker._audit_reach_models = reach_models(str(Path(assets).resolve()))
    ranker._audit_screen_cache = {}
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
        verify_hybrid_events(site, snapshots, directory, own_manifest, checks, spec, proposals, ordered)
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
            if method == "fallback":
                completed += [r["proposal_id"] for r in snapshot["board_fallback"]["records"]]
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
            if method == "fallback":
                choice = verify_fallback_snapshot(snapshot, choice, layout, directory, prior, home, resolutions, ranker, checks, spec)
            checks.equal(snapshot["choice"], choice, identity + "/pre-deadline command independently reconstructed")
            if snapshot["active_stage"] == "finished" and not snapshot.get("board_fallback", {}).get("triggered"):
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


def _compact_digest(value):
    """Independent encoding used by immutable worker receipts."""
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                      allow_nan=False).encode()).hexdigest()


def _verify_hybrid_chain(*, site, snapshots, directory, manifest, checks, spec, variant,
                  expected_menu, expected_order, geometry_triggered=False):
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
    if site["status"] == "completed" and not geometry_triggered:
        checks.equal(events[-1]["event"], "query_completed", label + "/completed job has a terminal durable event")
        checks.equal(last_published, len(snapshots), label + "/all completed publications accepted before clean finish")
    if geometry_triggered:
        checks.require(last_published <= len(snapshots), label + "/prefix event cannot claim later geometry publications")
    final_mode = events[-1]["engine_mode"]
    for snapshot, publication in zip(snapshots, site["accepted_receipts"][:len(snapshots)], strict=True):
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


def verify_hybrid_events(site, snapshots, directory, manifest, checks, spec, proposals, ordered):
    triggered = any(s.get('board_fallback', {}).get('triggered') for s in snapshots)
    prefix = [s for s in snapshots if not s.get('board_fallback', {}).get('triggered')]
    checks.require(snapshots[:len(prefix)] == prefix, 'Geometry fallback follows a contiguous hybrid prefix')
    _verify_hybrid_chain(site=site, snapshots=prefix, directory=directory, manifest=manifest,
        checks=checks, spec=spec, variant='hybrid', expected_menu=proposals, expected_order=ordered,
        geometry_triggered=triggered)
    if triggered:
        checks.require(bool(prefix), 'Fallback requires an accepted checked prefix')
        fixed = prefix[-1]
        extension = snapshots[len(prefix):]
        first_state = extension[0]['board_fallback']
        if site.get('accepted_receipts') is not None:
            before = site['accepted_receipts'][len(prefix)-1]['elapsed_s']
            takeover = site['accepted_receipts'][len(prefix)]['elapsed_s']
            checks.require(before <= first_state['trigger']['elapsed_s'] <= takeover,
                           'Takeover occurs between accepted prefix and its durable trigger publication')
        previous_state = first_state
        for snapshot in extension:
            for key in ('plan', 'generated', 'ranked_proposal_ids', 'stalled_reference_snapshot'):
                checks.equal(snapshot[key], fixed[key], 'Geometry fallback preserves original prefix/'+key)
            state = snapshot['board_fallback']
            checks.equal(state['baseline_choice'], fixed['choice'],
                         'Geometry fallback uses the last accepted prefix choice')
            for key in ('version', 'protocol', 'trigger', 'baseline_choice'):
                checks.equal(state[key], first_state[key], 'Immutable geometry takeover/'+key)
            for key in ('proposals', 'records', 'admitted', 'check_order'):
                checks.equal(state[key][:len(previous_state[key])], previous_state[key],
                             'Completed geometry evidence cannot be rewritten/'+key)
            for key in ('nominal_predictions', 'wrist_screen', 'predictions'):
                checks.require(all(state[key].get(k) == v for k,v in previous_state[key].items()),
                               'Recorded predictions cannot be rewritten/'+key)
            if previous_state['bounds'] is not None:
                checks.equal(state['bounds'], previous_state['bounds'], 'Frozen geometry bounds remain fixed')
            previous_state = state
    checks.equal(site.get('board_fallback'), snapshots[-1].get('board_fallback') if snapshots else None,
                 'Outer fallback state equals accepted immutable snapshot')


def verify_fallback_snapshot(snapshot, base, layout, directory, prior, home, resolutions, ranker, checks, spec):
    from .board_fallback_plan import VERSION as PLAN_VERSION, RULE, fallback_menu
    from .board_pose_recovery_audit import recompute_screen, ranked_diverse, verify_admissions
    from .fixed35_broad_native import reach_models
    from .expand_geometry import WIDE_BOUNDS
    state=snapshot['board_fallback']; label=layout['id']+'/board-fallback'
    checks.equal((state['version'],state['protocol']), (PLAN_VERSION,spec['fallback_rule']),label+'/frozen protocol')
    checks.equal(spec['fallback_rule'],RULE,label+'/declared rule unchanged')
    checks.require(type(state['triggered']) is bool and type(state['completed']) is bool,label+'/explicit state')
    if not state['triggered']:
        checks.equal(state,dict(version=PLAN_VERSION,protocol=RULE,triggered=False,trigger=None,
            baseline_choice=None,bounds=None,proposals=[],nominal_predictions={},wrist_screen={},check_order=[],
            records=[],admitted=[],predictions={},selected_fallback_candidate_id=None,completed=False),
            label+'/no geometry work before takeover')
        return base
    checks.equal(state['baseline_choice'],base,label+'/independently reconstructed checked prefix')
    prediction=base.get('prediction') or {}; route=prediction.get('predicted_route')
    margin=(prediction.get(route) or {}).get('arm_clearance_m') if route in ('direct','table') else None
    margin=float(margin) if type(margin) in (int,float) and math.isfinite(margin) else None
    checks.require(base['request'] is not None and (margin is None or margin<RULE['arm_clearance_trigger_m']),
                   label+'/only a risky fully checked source can trigger geometry search')
    trigger=state['trigger']; elapsed=seconds(trigger['elapsed_s'])
    checks.equal((trigger['baseline_choice_sha256'],trigger['selection_uses_flight_labels'],trigger['arm_clearance_m']),
                 (digest(base),False,margin),label+'/outcome-blind original source binding')
    checks.require(trigger['reason'] in ('risky_prefix_soft_boundary','risky_prefix_finished'),label+'/known trigger')
    checks.require(elapsed<=spec['planning_budget_s']-RULE['minimum_fallback_s'],label+'/remaining original budget')
    if trigger['reason']=='risky_prefix_soft_boundary':
        checks.require(elapsed>=RULE['primary_prefix_s'],label+'/soft boundary cannot fire early')
    if state['bounds'] is None:
        checks.require(not any(state[k] for k in ('proposals','nominal_predictions','wrist_screen','check_order','records','admitted','predictions')),
                       label+'/no work before bounds frozen')
        checks.equal((state['selected_fallback_candidate_id'],state['completed']),(None,False),label+'/trigger-only state')
        return base
    checks.equal(state['bounds'],{k:WIDE_BOUNDS[k] for k in ('low','high')},label+'/unchanged calibrated action bounds')
    from .clearance_expansion_plan import needs_expansion
    from .planning_budget_plan import _wrap_choice
    proposals=fallback_menu(layout,base['request'],state['bounds']['low'],state['bounds']['high'])
    initial=[p for p in state['proposals'] if 'retry_of' not in p]
    retries=[p for p in state['proposals'] if 'retry_of' in p]
    checks.equal(initial,proposals,label+'/deterministic same relative menu for every target')
    checks.require(len(retries)<=RULE['multistart_retry_limit'],label+'/bounded multistart retries')
    # ranker carries the immutable URDF-derived reach models attached by verify_plans.
    models=ranker._audit_reach_models
    screens={}; nominal={}
    for proposal in proposals:
        key=digest(proposal)
        if key not in ranker._audit_screen_cache:
            ranker._audit_screen_cache[key]=recompute_screen(ranker,spec['scene'],proposal,models)
        reach,pred=ranker._audit_screen_cache[key]
        screens[proposal['id']]=reach
        if reach['passed']:nominal[proposal['id']]=pred
    checks.equal(state['wrist_screen'],screens,label+'/necessary reach screening independently recomputed')
    checks.equal(state['nominal_predictions'],nominal,label+'/nominal ballistic scores independently recomputed')
    selected=ranked_diverse([p for p in proposals if screens[p['id']]['passed']],nominal,RULE['full_checks'])
    records=state['records']; warm=[r for r in records if not r['proposal_id'].endswith('-multistart')]
    checks.equal([r['proposal_id'] for r in warm],[p['id'] for p in selected[:len(warm)]],label+'/completed shortlist prefix')
    rejected=[r['proposal_id'] for r in warm if r['admitted']==0]
    source_by_id={p['id']:p for p in proposals}
    checks.equal(retries,[dict(copy.deepcopy(source_by_id[x]),id=x+'-multistart',retry_of=x,ik_search='multistart16')
                         for x in rejected[:len(retries)]],label+'/only earlier rejected shortlisted geometries retried')
    checks.require(not retries or len(warm)==len(selected),label+'/retries follow all shortlisted checks')
    order=[p['id'] for p in selected]+[p['id'] for p in retries]
    checks.equal(state['check_order'],order,label+'/complete deterministic check order')
    checks.equal([r['proposal_id'] for r in records],order[:len(records)],label+'/no skipping incomplete checks')
    checks.require(len(records)<=len(order),label+'/bounded completed checks')
    case=dict(id=layout['id'],original=base['request'],**{k:state[k] for k in ('proposals','records','admitted')})
    verify_admissions(case,directory,prior,home,resolutions,checks)
    predictions={r['id']:ranker.predict(r) for r in state['admitted']}
    checks.equal(state['predictions'],predictions,label+'/full-pose flight predictions independently recomputed')
    best=copy.deepcopy(base); winner=None
    for request in state['admitted']:
        candidate=predictions[request['id']]; new=candidate.get('score_m'); old=(best.get('prediction') or {}).get('score_m')
        better=type(new) in (float,int) and math.isfinite(new) and (type(old) not in (float,int) or not math.isfinite(old) or new>old)
        if better:
            best=copy.deepcopy(base)
            best.update(prediction=candidate,candidate_pool='board_pose_fallback',intended_route=request['intended_route'],
                        available_routes=sorted(set(base.get('available_routes',[]))|{request['intended_route']}),request=copy.deepcopy(request))
            best=_wrap_choice(best,'warm_early');best['predicted_flight_risk']=needs_expansion(best);winner=request['id']
    checks.equal(state['selected_fallback_candidate_id'],winner,label+'/strictly better checked scores only; ties retain prefix')
    if snapshot['active_stage']=='finished':
        checks.equal((len(records),state['completed']),(len(order),True),label+'/normal finish exhausts declared candidates')
    return best

def verify_layouts(args,spec,checks):
    from .interactive_fast_check import coordinates
    from .interactive_physics_check import sample_positions
    from .intergrid_layouts import _placement_geometry
    from .intergrid_route_audit import verify_separation
    from .board_fallback_study import recovery_binding, HISTORY_NAMES
    study=Path(args.output); artifact=read(study/'layouts.json'); provenance=artifact['provenance']
    source=Path(args.source_study); old=read(source/'layouts.json')
    checks.equal(provenance['source_layouts_sha256'],sha(source/'layouts.json'),'Source history is audited exact bytes')
    checks.equal(provenance['recovery'],recovery_binding(Path(args.recovery_study),Path(args.recovery_audit)),
                 'Known-target recovery remains separate and excluded')
    checks.equal(provenance['recovery'],spec['recovery_parent'],'Recovery contract binding')
    known=set(map(tuple,old['provenance']['excluded_coordinates_m']))
    known.update(coordinates(old['layouts']));known.update(coordinates(read(Path(args.recovery_study)/'preparation.json')))
    checks.equal({k:provenance[k] for k in ('geometry','grid','bounds')},
                 {k:old['provenance'][k] for k in ('geometry','grid','bounds')},'Unchanged audited workspace')
    checks.equal(provenance['geometry'],_placement_geometry(spec['scene'],args.assets),'Independent table/base geometry')
    for name,expected in provenance['history_files'].items():
        path=Path(name)
        checks.require(path.resolve().is_relative_to(Path(args.evidence_root).resolve())
                       and path.name in HISTORY_NAMES and not path.resolve().is_relative_to(study.resolve()),
                       'Only prior live observations contribute exclusions')
        checks.equal(sha(path),expected,'Immutable prior live observation/'+name)
        known.update(coordinates(read(path)))
    exclusions=list(map(tuple,provenance['excluded_coordinates_m']))
    checks.equal(exclusions,sorted(known),'Complete prior coordinates excluded without target-dependent additions')
    checks.equal(provenance['excluded_coordinates_sha256'],digest(exclusions),'Exclusion list hash')
    expected=[]; occupied=list(exclusions)
    for offset in range(5):
        batch=sample_positions(provenance['geometry'],provenance['grid'],occupied,provenance['bounds'],spec['seed']+offset)
        for row in batch:
            identity=f'board-fallback-test-{len(expected):02d}'
            expected.append(dict(row,id=identity,context_id=identity,split='test',stratum=row['region']))
        occupied.extend(tuple(row['bucket_xy_m']) for row in batch)
    layouts=artifact['layouts']
    checks.equal(len(layouts),50,'Exactly fifty new paired bucket positions')
    checks.equal(layouts,expected,'Independent seed/history reconstruction')
    verify_separation(layouts,[],exclusions,checks)
    return layouts


def audit(args):
    """Read-only CPU verification; no native rollouts or result mutation."""
    from . import replay650_verify as replay
    from . import direct_path_first_audit as shared
    from .cell1050_rebound_study import parent_inputs
    from .fast_planning_study import COPIED_INPUTS
    from .intergrid_route_refit_inputs import _audited_source
    from .intergrid_route_labels import LABEL_VERSION
    from .stalled_reference_plan import FEATURES
    from .board_fallback_plan import RULE
    study,parent=Path(args.output),Path(args.parent_study);checks=Checks()
    spec,report=read(study/'contract.json'),read(study/'report.json')
    fingerprint(spec,checks,'Frozen board-fallback comparison')
    checks.equal((spec['version'],report['version'],report['status'],report['phase']),
                 (STUDY_VERSION,STUDY_VERSION,'completed','completed'),'Completed declared comparison')
    checks.equal((spec['physics_substeps'],spec['methods'],spec['label_version'],spec['test_positions'],
                  spec['planning_budget_s'],spec['attempts_per_method_per_position']),
                 (64,list(METHODS),LABEL_VERSION,50,30.,1),'Matched scoring/decision budget/single-drop denominator')
    checks.equal((spec['seed'],spec['processing_seed'],spec['reference_path_s'],spec['early_score_m'],spec['fallback_rule']),
                 (2026112217,2026112218,5.,.005,RULE),'Predeclared frozen search and spatial seeds')
    checks.require(0<seconds(report['gpu_lease_s'])<=spec['gpu_cap_s']<=3600,'Within one GPU-hour native cap')
    checks.equal(report['native_completed'],True,'All native commands complete')
    manifest=replay.check_manifest(study,checks)
    checks.equal(report['evidence_manifest_sha256'],sha(study/'evidence-manifest.json'),'Complete evidence binding')
    required=set(COPIED_INPUTS)|{'setup-home.json','layouts.json','cpu-equivalence.json'}
    checks.require(required<=set(spec['frozen_inputs']),'Every policy/calibration/control/layout frozen before selection')
    verify_bindings(study,spec['frozen_inputs'],checks,'Frozen study inputs')
    source=Path(getattr(args,'frozen_source_root',None) or Path(__file__).resolve().parents[2]).resolve()
    checks.require({'sircus/mjlab/board_fallback_plan.py','sircus/mjlab/board_fallback_study.py',
                    'sircus/mjlab/board_fallback_audit.py'}<=set(spec['source_sha256']),'New producer/auditor source frozen')
    for name,expected in spec['source_sha256'].items():checks.equal(sha(contained(source,name)),expected,'Exact frozen source/'+name)
    native,binding,_=parent_inputs(parent,args.parent_audit)
    checks.equal((spec['parent'],spec['identity'],spec['scene']),
                 (binding,native['identity'],native['scene']),'Exact parent calibrated physics and geometry')
    _,_,_,binding,previous=_audited_source(Path(args.source_study),Path(args.source_audit),
        'sircus-stalled-reference-study-v1',(*COPIED_INPUTS,'layouts.json','setup-home.json','cpu-equivalence.json'))
    checks.equal(spec['source_parent'],binding,'Original audited planner/model/library source')
    checks.equal((spec['identity'],spec['scene'],spec['flight_ranking']),
                 (previous['identity'],previous['scene'],previous['flight_ranking']),'Unchanged frozen predictor/scorer')
    for name in (*COPIED_INPUTS,'setup-home.json','cpu-equivalence.json'):
        checks.equal(sha(study/name),sha(Path(args.source_study)/name),'Copied exact source/'+name)
    checks.equal(sha(study/'setup-home.json'),sha(parent/'setup-home.json'),'Canonical original home retained')
    equivalence=read(study/'cpu-equivalence.json')
    checks.equal((equivalence.get('exact_full_record_equivalence'),equivalence.get('mismatch_count'),
                  equivalence.get('segment_mismatch_count'),equivalence.get('cpu_only')),
                 (True,0,0,True),'Full CPU check equivalence prerequisite')
    checks.require(type(equivalence.get('samples')) is int and equivalence['samples']>=64,'At least64 CPU states checked')
    for method in METHODS:
        checks.equal(read(study/('native-'+method)/'arm-check-features.json'),
                     dict(variant='hybrid',features=list(FEATURES)),method+'/same underlying checked hybrid engine')
    layouts=verify_layouts(args,spec,checks)
    prior,home=read(study/'training-prior.json'),read(study/'setup-home.json')
    wrappers=verify_plans(study,layouts,spec,args.assets,prior,home,manifest,checks)
    slots,expected=shared.verify_slots(study,layouts,wrappers,manifest,checks,variants=METHODS)
    rows,controls=shared.audit_packets(study,parent,args.assets,spec,report,manifest,expected,prior,home,checks,variants=METHODS)
    checks.equal(controls,18,'Nine controls bracket each side of comparison')
    checks.equal(report['progress'],dict(prepared=100,planned=100,completed_releases=len(rows),planned_releases=len(rows)),
                 'All fifty attempts per method counted, including no-shot decisions')
    checks.equal(report['budgets'],dict(positions=50,planning_s=30.),'Unchanged report budget')
    comparison=shared.summarize(layouts,slots,rows,wrappers,variants=METHODS)
    checks.equal(report['comparison'],comparison,'Independent captures and timing match published summary')
    states=[w['site'].get('board_fallback',{}) for w in wrappers if w['variant']=='fallback']
    fallback=dict(activated=sum(s.get('triggered') is True for s in states),
                  changed_choice=sum(s.get('selected_fallback_candidate_id') is not None for s in states))
    checks.equal(report['fallback'],fallback,'Actual takeover and changed-choice counts')
    return dict(version=VERSION,status='passed',checks_count=checks.count,verifier_source_sha256=sha(Path(__file__)),
        frozen_source_root=str(source),comparison=comparison,fallback=fallback,controls=controls,test_releases=len(rows),
        report_sha256=sha(study/'report.json'),choices_sha256=sha(study/'choices.json'),
        evidence_manifest_sha256=sha(study/'evidence-manifest.json'))
