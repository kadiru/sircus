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

VERSION = "sircus-guarded-board-fallback-independent-audit-v1"
STUDY_VERSION = "sircus-guarded-board-fallback-study-v1"
METHODS = ("baseline", "guarded")

from .board_fallback_audit import _verify_hybrid_chain

def verify_plans(study, layouts, spec, assets, prior, home, manifest, checks, *, artifact_root=None, variants=METHODS, plan_hook=None, selected_wrappers=None):
    """Rebuild finite menus, checked prefixes, full proof bindings and choices."""
    from .clearance_expansion_plan import _empty, branch_proposals, needs_expansion
    from .intergrid_candidate_replay_plan import command_key
    from .intergrid_flight_ranker import FlightRanker
    from .intergrid_library_plan import libraries
    from .neural_shot_model import order
    from .neural_shot_plan import CHECK_BATCH, CHECK_LIMIT, _choice, menu

    artifact_root = study if artifact_root is None else artifact_root
    wrappers = read(artifact_root / "plans.json") if selected_wrappers is None else selected_wrappers
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
        inventory = {path.stem for path in (artifact_root / "preparation" / method).glob("*.json")}
        expected_inventory = {layout["id"] for layout in layouts}
        if selected_wrappers is None:
            checks.equal(inventory, expected_inventory, method + "/exact preparation file inventory")
        else:
            checks.require(expected_inventory <= inventory, method + "/selected preflight preparations exist")
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
            if method == "guarded":
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
            if method == "guarded":
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
    if selected_wrappers is None:
        verify_workers(wrappers, layouts, worker_spec, checks, variants=variants)
    return wrappers


def verify_fallback_snapshot(snapshot, base, layout, directory, prior, home, resolutions, ranker, checks, spec):
    from .guarded_board_fallback_plan import VERSION as PLAN_VERSION, RULE, fallback_menu
    from .board_pose_recovery_audit import recompute_screen, ranked_diverse, verify_admissions
    from .fixed35_broad_native import reach_models
    from .expand_geometry import WIDE_BOUNDS
    state=snapshot['board_fallback']; label=layout['id']+'/board-fallback'
    checks.equal((state['version'],state['protocol']), (PLAN_VERSION,spec['fallback_rule']),label+'/frozen protocol')
    checks.equal(spec['fallback_rule'],RULE,label+'/declared rule unchanged')
    checks.require(type(state['triggered']) is bool and type(state['completed']) is bool,label+'/explicit state')
    checks.equal(state['original_interrupted_by_fallback'],False,label+'/original search never interrupted')
    if not state['triggered']:
        expected=dict(version=PLAN_VERSION,protocol=RULE,triggered=False,trigger=None,
            original_completion=state['original_completion'],original_interrupted_by_fallback=False,
            baseline_choice=None,bounds=None,proposals=[],nominal_predictions={},wrist_screen={},check_order=[],
            records=[],admitted=[],predictions={},replacement_decisions=[],selected_fallback_candidate_id=None,
            completed=state['original_completion'] is not None)
        checks.equal(state,expected,label+'/no geometry work unless original risky search completed')
        if state['original_completion'] is not None:
            verify_original_completion(state,base,spec,checks,label)
        return base
    checks.equal(state['baseline_choice'],base,label+'/independently reconstructed checked prefix')
    prediction=base.get('prediction') or {}; route=prediction.get('predicted_route')
    margin=(prediction.get(route) or {}).get('arm_clearance_m') if route in ('direct','table') else None
    margin=float(margin) if type(margin) in (int,float) and math.isfinite(margin) else None
    checks.require(base['request'] is not None and (margin is None or margin<RULE['arm_clearance_trigger_m']),
                   label+'/only a risky fully checked source can trigger geometry search')
    verify_original_completion(state,base,spec,checks,label)
    trigger=state['trigger']; elapsed=seconds(trigger['elapsed_s'])
    checks.equal((trigger['baseline_choice_sha256'],trigger['selection_uses_flight_labels'],trigger['arm_clearance_m']),
                 (digest(base),False,margin),label+'/outcome-blind original source binding')
    checks.equal(trigger['reason'],'risky_original_natural_completion',label+'/only natural completion triggers fallback')
    checks.require(state['original_completion']['elapsed_s'] <= elapsed <= spec['planning_budget_s']-RULE['minimum_fallback_s'],
                   label+'/fallback uses only unused original decision time')
    if state['bounds'] is None:
        checks.require(not any(state[k] for k in ('proposals','nominal_predictions','wrist_screen','check_order','records','admitted','predictions','replacement_decisions')),
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
    best=copy.deepcopy(base); winner=None; decisions=[]
    for request in state['admitted']:
        candidate=predictions[request['id']]
        decision=independent_replacement_decision(candidate,best.get('prediction'))
        decisions.append(dict(request_id=request['id'],**decision))
        better=decision['accepted']
        if better:
            best=copy.deepcopy(base)
            best.update(prediction=candidate,candidate_pool='guarded_board_pose_fallback',intended_route=request['intended_route'],
                        available_routes=sorted(set(base.get('available_routes',[]))|{request['intended_route']}),request=copy.deepcopy(request))
            best=_wrap_choice(best,'warm_early');best['predicted_flight_risk']=needs_expansion(best);winner=request['id']
    checks.equal(state['replacement_decisions'],decisions,label+'/independently reconstructed finite-clearance replacement decisions')
    checks.equal(state['selected_fallback_candidate_id'],winner,label+'/strictly better checked scores only; ties retain prefix')
    if snapshot['active_stage']=='finished':
        checks.equal((len(records),state['completed']),(len(order),True),label+'/normal finish exhausts declared candidates')
    return best

def verify_layouts(args,spec,checks):
    from .interactive_fast_check import coordinates
    from .interactive_physics_check import sample_positions
    from .intergrid_layouts import _placement_geometry
    from .intergrid_route_audit import verify_separation
    from .guarded_board_fallback_study import HISTORY_NAMES
    study=Path(args.output); artifact=read(study/'layouts.json'); provenance=artifact['provenance']
    source=Path(args.source_study); old=read(source/'layouts.json')
    checks.equal(provenance['source_layouts_sha256'],sha(source/'layouts.json'),'Source history is audited exact bytes')
    known=set(map(tuple,old['provenance']['excluded_coordinates_m']))
    known.update(coordinates(old['layouts']))
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
    for offset in range(2):
        batch=sample_positions(provenance['geometry'],provenance['grid'],occupied,provenance['bounds'],spec['seed']+offset)
        for row in batch:
            identity=f'guarded-board-test-{len(expected):02d}'
            expected.append(dict(row,id=identity,context_id=identity,split='test',stratum=row['region']))
        occupied.extend(tuple(row['bucket_xy_m']) for row in batch)
    layouts=artifact['layouts']
    checks.equal(len(layouts),20,'Exactly twenty new paired bucket positions')
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
    from .guarded_board_fallback_plan import RULE
    study,parent=Path(args.output),Path(args.parent_study);checks=Checks()
    spec,report=read(study/'contract.json'),read(study/'report.json')
    fingerprint(spec,checks,'Frozen board-fallback comparison')
    checks.equal((spec['version'],report['version'],report['status'],report['phase']),
                 (STUDY_VERSION,STUDY_VERSION,'completed','completed'),'Completed declared comparison')
    checks.equal((spec['physics_substeps'],spec['methods'],spec['label_version'],spec['test_positions'],
                  spec['planning_budget_s'],spec['attempts_per_method_per_position']),
                 (64,list(METHODS),LABEL_VERSION,20,30.,1),'Matched scoring/decision budget/single-drop denominator')
    checks.equal((spec['seed'],spec['processing_seed'],spec['reference_path_s'],spec['early_score_m'],spec['fallback_rule']),
                 (2026112307,2026112308,5.,.005,RULE),'Predeclared frozen search and spatial seeds')
    checks.require(0<seconds(report['gpu_lease_s'])<=spec['gpu_cap_s']<=1800,'Within thirty minutes of leased native time')
    checks.equal(report['native_completed'],True,'All native commands complete')
    manifest=replay.check_manifest(study,checks)
    checks.equal(report['evidence_manifest_sha256'],sha(study/'evidence-manifest.json'),'Complete evidence binding')
    required=set(COPIED_INPUTS)|{'setup-home.json','layouts.json','cpu-equivalence.json','diagnostic-layout.json'}
    checks.require(required<=set(spec['frozen_inputs']),'Every policy/calibration/control/layout frozen before selection')
    verify_bindings(study,spec['frozen_inputs'],checks,'Frozen study inputs')
    source=Path(getattr(args,'frozen_source_root',None) or Path(__file__).resolve().parents[2]).resolve()
    checks.require({'sircus/mjlab/guarded_board_fallback_plan.py','sircus/mjlab/guarded_board_fallback_study.py',
                    'sircus/mjlab/guarded_board_fallback_audit.py'}<=set(spec['source_sha256']),'New producer/auditor source frozen')
    for name,expected in spec['source_sha256'].items():checks.equal(sha(contained(source,name)),expected,'Exact frozen source/'+name)
    native,binding,_=parent_inputs(parent,args.parent_audit)
    checks.equal((spec['parent'],spec['identity'],spec['scene']),
                 (binding,native['identity'],native['scene']),'Exact parent calibrated physics and geometry')
    _,_,_,binding,previous=_audited_source(Path(args.source_study),Path(args.source_audit),
        'sircus-board-fallback-study-v1',(*COPIED_INPUTS,'layouts.json','setup-home.json','cpu-equivalence.json','choices.json','plans.json'))
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
    diagnostic_layout,diagnostic_artifact,diagnostic_expected=verify_diagnostic(args,spec,manifest,prior,home,checks)
    expected.update(diagnostic_expected)
    rows,controls=shared.audit_packets(study,parent,args.assets,spec,report,manifest,expected,prior,home,checks,variants=METHODS)
    checks.equal(controls,18,'Nine controls bracket each side of comparison')
    fresh={k:r for k,r in rows.items() if not r['adapter_directory'].startswith('diagnostic/')}
    diagnostic_rows={k:r for k,r in rows.items() if r['adapter_directory'].startswith('diagnostic/')}
    checks.equal(report['progress'],dict(prepared=40,planned=40,completed_releases=len(fresh),planned_releases=len(fresh)),
                 'All twenty attempts per method counted, including no-shot decisions')
    checks.equal(report['diagnostic_progress'],dict(prepared=2,planned=2,completed_releases=len(diagnostic_rows),planned_releases=len(diagnostic_rows)),
                 'Diagnostic planning and releases remain separate from fresh denominator')
    checks.require(len(rows)+controls<=64,'Bounded forty fresh plus six diagnostic plus eighteen control releases')
    checks.equal(report['budgets'],dict(positions=20,planning_s=30.),'Unchanged report budget')
    comparison=shared.summarize(layouts,slots,fresh,wrappers,variants=METHODS)
    checks.equal(report['comparison'],comparison,'Independent captures and timing match published summary')
    diagnostic=summarize_diagnostic(diagnostic_layout,diagnostic_artifact,diagnostic_rows)
    checks.equal(report['diagnostic'],diagnostic,'Independent diagnostic accounting; not fresh evaluation')
    states=[w['site'].get('board_fallback',{}) for w in wrappers if w['variant']=='guarded']
    fallback=dict(activated=sum(s.get('triggered') is True for s in states),
                  changed_choice=sum(s.get('selected_fallback_candidate_id') is not None for s in states))
    checks.equal(report['fallback'],fallback,'Actual takeover and changed-choice counts')
    return dict(version=VERSION,status='passed',checks_count=checks.count,verifier_source_sha256=sha(Path(__file__)),
        frozen_source_root=str(source),comparison=comparison,diagnostic=diagnostic,fallback=fallback,controls=controls,test_releases=len(fresh),diagnostic_releases=len(diagnostic_rows),
        report_sha256=sha(study/'report.json'),choices_sha256=sha(study/'choices.json'),
        evidence_manifest_sha256=sha(study/'evidence-manifest.json'))

def verify_original_completion(state, base, spec, checks, label):
    value=state['original_completion']
    checks.require(isinstance(value,dict),label+'/natural original completion receipt required')
    checks.equal((value['returned_naturally'],value['interrupted_by_fallback'],value['choice_sha256']),
                 (True,False,digest(base)),label+'/original choice completed without wrapper interruption')
    checks.require(value['return_reason'] in ('completed','early_good_shot'),label+'/original natural return reason')
    checks.require(0<=seconds(value['elapsed_s'])<=spec['planning_budget_s'],label+'/original completion uses shared clock')


def verify_hybrid_events(site,snapshots,directory,manifest,checks,spec,proposals,ordered):
    prefix=[s for s in snapshots if s.get('board_fallback',{}).get('original_completion') is None]
    extension=snapshots[len(prefix):]
    checks.equal(snapshots[:len(prefix)],prefix,'Original planning is an uninterrupted contiguous prefix')
    _verify_hybrid_chain(site=site,snapshots=prefix,directory=directory,manifest=manifest,checks=checks,spec=spec,
        variant='hybrid',expected_menu=proposals,expected_order=ordered,geometry_triggered=bool(extension))
    if extension:
        checks.require(bool(prefix),'Extension must follow at least one accepted original snapshot')
        event=site['stalled_reference']['events'][-1]
        checks.equal((event['event'],event['published_snapshots']),('query_completed',len(prefix)),
                     'Hybrid must durably complete normally before wrapper extension')
        last=prefix[-1];first=extension[0]['board_fallback'];previous=first
        verify_original_completion(first,last['choice'],spec,checks,'Original completion')
        checks.require(event['elapsed_s']<=first['original_completion']['elapsed_s']<=site['accepted_receipts'][len(prefix)]['elapsed_s'],
                       'Original completion predates first extension publication')
        checks.equal(first['original_completion']['return_reason'],
                     'early_good_shot' if last['active_stage']=='early_good_shot' else 'completed','Honest original return reason')
        for snapshot in extension:
            state=snapshot['board_fallback']
            for key in ('plan','generated','ranked_proposal_ids','stalled_reference_snapshot'):
                checks.equal(snapshot[key],last[key],'Original completed evidence unchanged/'+key)
            for key in ('version','protocol','original_completion','original_interrupted_by_fallback','trigger','triggered','baseline_choice'):
                checks.equal(state[key],first[key],'Immutable extension origin/'+key)
            for key in ('proposals','records','admitted','check_order','replacement_decisions'):
                checks.equal(state[key][:len(previous[key])],previous[key],'Only append completed extension evidence/'+key)
            for key in ('nominal_predictions','wrist_screen','predictions'):
                checks.require(all(state[key].get(k)==v for k,v in previous[key].items()),'Immutable earlier predictions/'+key)
            if previous['bounds'] is not None:checks.equal(state['bounds'],previous['bounds'],'Stable geometry bounds')
            if state['triggered']:checks.equal(state['baseline_choice'],last['choice'],'Fallback anchors natural completed original choice')
            previous=state
    checks.equal(site.get('board_fallback'),snapshots[-1].get('board_fallback') if snapshots else None,
                 'Outer guard evidence is exactly last accepted snapshot')

def verify_diagnostic(args,spec,manifest,prior,home,checks,*,require_native_start=True):
    from .planning_budget_plan import _wrap_choice
    study=Path(args.output);artifact=read(study/'diagnostic-layout.json')
    source=read(Path(args.source_study)/'layouts.json')['layouts']
    old=next(r for r in source if r['id']=='board-fallback-test-23')
    historical=next(r for r in read(Path(args.source_study)/'report.json')['comparison']['positions'] if r['id']==old['id'])
    checks.equal(historical['outcomes'],dict(baseline='direct',fallback='flight_miss'),'Known previously lost capture, not a fresh test')
    layout=dict(old,id='guarded-board-known-regression',context_id='guarded-board-known-regression',split='test',
                stratum='known regression; excluded from fresh comparison')
    checks.equal(artifact,dict(layout=layout,source_id=old['id'],repeats=3),'Exact source-derived known diagnostic coordinate')
    checks.equal(spec['diagnostic_repeats'],3,'Three matched repetitions of each independently planned command')
    wrappers=verify_plans(study,[layout],spec,args.assets,prior,home,manifest,checks,artifact_root=study/'diagnostic')
    value=read(study/'diagnostic/choices.json')
    expected_bindings={name:sha(study/name) for name in ('contract.json','diagnostic-layout.json','diagnostic/plans.json')}
    checks.equal(value['bindings'],expected_bindings,'Diagnostic choices bind completed independent planning')
    checks.require(all(manifest[name]==v for name,v in expected_bindings.items()),'Diagnostic dependencies remain immutable')
    slots=[_wrap_choice(w['site']['choice'],w['variant']) for w in wrappers]
    checks.equal(value['slots'],slots,'Exact original planned diagnostic choice per method')
    requests=verify_repeat_requests(slots,value['requests'],checks)
    if require_native_start:
        started=read(study/'native-started.json')
        checks.equal(started['diagnostic_requests_sha256'],digest(requests),'All diagnostic choices frozen before any native release')
    return layout,value,{'diagnostic_'+method:requests[method] for method in METHODS}


def summarize_diagnostic(layout,artifact,rows):
    from .hybrid_miss_audit import classify
    from .intergrid_route_selector_plan import STANDARD_CAPTURES
    methods={}
    for slot in artifact['slots']:
        method=slot['method'];requests=artifact['requests'][method]
        outcomes=[classify(rows[request['id']]) for request in requests]
        methods[method]=dict(requested=3,checked_shot=slot['request'] is not None,released=len(outcomes),
                             captures=sum(v in STANDARD_CAPTURES for v in outcomes),outcomes=outcomes)
    return dict(bucket_xy_m=layout['bucket_xy_m'],repeats=3,methods=methods)

def independent_replacement_decision(candidate,incumbent):
    candidate=candidate or {};incumbent=incumbent or {}
    def finite(value):return float(value) if type(value) in (float,int) and math.isfinite(value) else None
    new,old=finite(candidate.get('score_m')),finite(incumbent.get('score_m'))
    route=candidate.get('predicted_route')
    margin=finite((candidate.get(route) or {}).get('arm_clearance_m')) if route in ('direct','table') else None
    reason=('unknown_arm_clearance' if margin is None else 'insufficient_arm_clearance' if margin<.005
            else 'nonfinite_score_comparison' if new is None or old is None else 'not_strictly_better' if new<=old
            else 'clearance_qualified_improvement')
    return dict(accepted=reason=='clearance_qualified_improvement',reason=reason,predicted_route=route,
                arm_clearance_m=margin,minimum_arm_clearance_m=.005,candidate_score_m=new,incumbent_score_m=old)


def preflight(args):
    """CPU preparation verification before native-start exists; never a final result."""
    from .guarded_board_fallback_study import frozen_check
    from .fast_planning_study import COPIED_INPUTS
    from .intergrid_route_refit_inputs import _audited_source
    from .planning_budget_plan import _wrap_choice
    study=Path(args.output);checks=Checks();spec=frozen_check(args)
    _,_,_,binding,_=_audited_source(Path(args.source_study),Path(args.source_audit),'sircus-board-fallback-study-v1',
        (*COPIED_INPUTS,'layouts.json','setup-home.json','cpu-equivalence.json','choices.json','plans.json'))
    checks.equal(spec['source_parent'],binding,'Audited prior study remains unchanged')
    layouts=verify_layouts(args,spec,checks)
    manifest={p.relative_to(study).as_posix():sha(p) for p in study.rglob('*') if p.is_file() and p.name!='report.json'}
    prior,home=read(study/'training-prior.json'),read(study/'setup-home.json')
    wrappers=verify_plans(study,layouts,spec,args.assets,prior,home,manifest,checks)
    _,diagnostic,_=verify_diagnostic(args,spec,manifest,prior,home,checks,require_native_start=False)
    choices=read(study/'choices.json')
    checks.equal(choices['slots'],[_wrap_choice(w['site']['choice'],w['variant']) for w in wrappers],
                 'Fresh slots exactly preserve all independently checked choices')
    verify_bindings(study,choices['bindings'],checks,'Pre-native frozen choice dependencies')
    frozen_check(args)
    return dict(version='sircus-guarded-board-fallback-preparation-audit-v1',status='passed',
        scope='CPU preparation only; not the completed-study audit',checks_count=checks.count,
        fresh_positions=len(layouts),fresh_plans=len(wrappers),diagnostic_plans=len(diagnostic['slots']),
        deferred=['Native-start binding','Native64 trajectory/contact scoring','Pre/post control outcomes','Final report/evidence audit'],
        plans_sha256=sha(study/'plans.json'),diagnostic_plans_sha256=sha(study/'diagnostic/plans.json'),
        choices_sha256=sha(study/'choices.json'),contract_sha256=sha(study/'contract.json'),
        verifier_source_sha256=sha(Path(__file__)),native_execution_performed=False)


def verify_repeat_requests(slots,actual,checks):
    requests={}
    for slot in slots:
        request=slot['request']
        requests[slot['method']]=[] if request is None else [dict(copy.deepcopy(request),id=request['id']+f'-repeat-{n}',
            repeat=n,source_request_id=request['id']) for n in (1,2,3)]
    checks.equal(actual,requests,'Three exact repeats; no geometry/joint/proof changes after outcomes')
    return requests
