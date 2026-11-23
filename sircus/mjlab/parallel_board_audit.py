"""Independent CPU audit of the six-target concurrent planning development test."""
from __future__ import annotations
import copy
import hashlib
import json
import math
from pathlib import Path
import random
from .benchmark_verify import Checks, contained, digest, fingerprint, read, sha
from .direct_path_first_audit import audit_preparation, audit_branches, verify_workers, _proof, verify_seed
from .fast_planning_audit import verify_bindings, verify_deadline, verify_early_rule, seconds
from .board_fallback_audit import _verify_hybrid_chain
from .guarded_board_fallback_audit import independent_replacement_decision
VERSION="sircus-parallel-board-independent-audit-v1"
STUDY_VERSION="sircus-parallel-board-study-v1"
METHODS=("baseline","parallel")
NATIVE_CHANNELS=("baseline","parallel/original","parallel/alternative")

def verify_original_plans(study, layouts, spec, assets, prior, home, manifest, checks):
    """Rebuild finite menus, checked prefixes, full proof bindings and choices."""
    from .clearance_expansion_plan import _empty, branch_proposals, needs_expansion
    from .intergrid_candidate_replay_plan import command_key
    from .intergrid_flight_ranker import FlightRanker
    from .intergrid_library_plan import libraries
    from .neural_shot_model import order
    from .neural_shot_plan import CHECK_BATCH, CHECK_LIMIT, _choice, menu

    artifact_root=study; variants=METHODS
    outer=read(study / "plans.json")
    wrappers=[dict(variant=w["variant"], site=w["site"] if w["variant"]=="baseline" else w["site"]["lanes"]["original"]) for w in outer]
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
        checks.equal(inventory,expected_inventory,method+"/exact preparation file inventory")
    snapshot_sets={}
    for wrapper in wrappers:
        site = wrapper["site"]
        layout, method, plan = site["layout"], wrapper["variant"], site["plan"]
        identity = layout["id"] + "/" + method
        directory = study / ("native-"+method)
        saved=read(study / "preparation" / method / (layout["id"]+".json"))
        if method=="parallel":
            directory=directory / "original";saved=saved["lanes"]["original"]
        checks.equal(site,saved,identity+"/exact original per-job receipt")
        prefix=directory.parent.relative_to(study).as_posix()
        own_manifest=manifest if prefix=="." else {k[len(prefix)+1:]:v for k,v in manifest.items() if k.startswith(prefix+"/")}
        snapshots=verify_deadline(site,directory,own_manifest,checks,spec["planning_budget_s"])
        snapshot_sets[layout["id"],method]=snapshots
        proposals = menu(layout, libs, source_checkpoint)
        ordered = order(checkpoint, proposals)
        _verify_hybrid_chain(site=site,snapshots=snapshots,directory=directory,manifest=own_manifest,checks=checks,spec=spec,
            variant="hybrid",expected_menu=proposals,expected_order=ordered,geometry_triggered=False)
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
    verify_worker_lifecycles(wrappers,layouts,spec,checks)
    return outer,snapshot_sets,ranker


def verify_context(args,checks):
    from .cell1050_rebound_study import parent_inputs
    from .fast_planning_study import COPIED_INPUTS
    from .intergrid_route_refit_inputs import _audited_source
    from .intergrid_route_labels import LABEL_VERSION
    from .parallel_board_plan import RULE
    study,parent=Path(args.output),Path(args.parent_study)
    spec=read(study/'contract.json');fingerprint(spec,checks,'Frozen development protocol')
    checks.equal(spec['source_paths'],dict(study=str(Path(args.source_study).resolve()),audit=str(Path(args.source_audit).resolve()),
        parent_study=str(parent.resolve()),parent_audit=str(Path(args.parent_audit).resolve())),'Exact source locations for preservation')
    checks.equal((spec['version'],spec['scope'],spec['methods'],spec['test_positions'],spec['repeats'],
                  spec['physics_substeps'],spec['label_version'],spec['planning_budget_s'],spec['max_native_releases']),
                 (STUDY_VERSION,'development',list(METHODS),6,3,64,LABEL_VERSION,30.,54),'Declared six-target repeated development experiment')
    checks.equal((spec['processing_seed'],spec['reference_path_s'],spec['early_score_m'],spec['parallel_rule']),
                 (2026112309,5.,.005,RULE),'Immutable planning and processing rule')
    checks.equal(spec['gpu_cap_s'],1800,'Native lease capped at thirty minutes')
    source=Path(getattr(args,'frozen_source_root',None) or Path(__file__).resolve().parents[2]).resolve()
    required={'sircus/mjlab/parallel_board_plan.py','sircus/mjlab/parallel_board_study.py','sircus/mjlab/parallel_board_audit.py'}
    checks.require(required<=set(spec['source_sha256']),'Producer and verifier computational source frozen')
    for name,value in spec['source_sha256'].items():checks.equal(sha(contained(source,name)),value,'Exact source/'+name)
    verify_bindings(study,spec['frozen_inputs'],checks,'Frozen inputs')
    checks.require(set(COPIED_INPUTS)|{'setup-home.json','cpu-equivalence.json','layouts.json'}<=set(spec['frozen_inputs']),
                   'Every model, prior, scene and target frozen before planning')
    native,binding,_=parent_inputs(parent,args.parent_audit)
    checks.equal((spec['parent'],spec['identity'],spec['scene']),(binding,native['identity'],native['scene']),
                 'Original calibrated native parent unchanged')
    files=(*COPIED_INPUTS,'layouts.json','diagnostic-layout.json','setup-home.json','cpu-equivalence.json','choices.json','plans.json')
    _,_,_,binding,previous=_audited_source(Path(args.source_study),Path(args.source_audit),'sircus-guarded-board-fallback-study-v1',files)
    checks.equal(spec['source_parent'],binding,'Passed guarded study remains exact selection source')
    checks.equal((spec['identity'],spec['scene'],spec['flight_ranking']),
                 (previous['identity'],previous['scene'],previous['flight_ranking']),'No hidden predictor or simulator change')
    for name in (*COPIED_INPUTS,'setup-home.json','cpu-equivalence.json'):
        checks.equal(sha(study/name),sha(Path(args.source_study)/name),'Copied exact immutable input/'+name)
    checks.equal(sha(study/'setup-home.json'),sha(parent/'setup-home.json'),'Canonical home')
    equivalence=read(study/'cpu-equivalence.json')
    checks.equal((equivalence.get('exact_full_record_equivalence'),equivalence.get('mismatch_count'),
                  equivalence.get('segment_mismatch_count'),equivalence.get('cpu_only')),
                 (True,0,0,True),'Existing independent CPU-check equivalence prerequisite')
    checks.require(type(equivalence.get('samples')) is int and equivalence['samples']>=64,'At least64 CPU check states')
    return spec,source


def verify_layouts(args,spec,checks):
    source=Path(args.source_study)
    old=read(source/'layouts.json')['layouts'];report=read(source/'report.json')
    results={r['id']:r for r in report['comparison']['positions']}
    expected=[]
    for suffix in ('03','09','13','15','19'):
        identity='guarded-board-test-'+suffix
        checks.equal(results[identity]['outcomes'],dict(baseline='flight_miss',guarded='flight_miss'),
                     identity+'/known paired miss, not new holdout')
        row=next(r for r in old if r['id']==identity);new='parallel-board-miss-'+suffix
        expected.append(dict(row,id=new,context_id=new,split='test',source_id=identity,role='previous_miss',stratum='selected development miss'))
    row=read(source/'diagnostic-layout.json')['layout']
    for method in ('baseline','guarded'):
        previous=report['diagnostic']['methods'][method]
        checks.equal((previous['requested'],previous['released'],previous['captures'],previous['outcomes']),
                     (3,3,3,['direct']*3),'Previously successful diagnostic control retained')
    expected.append(dict(row,id='parallel-board-known-regression',context_id='parallel-board-known-regression',
        source_id=row['id'],role='known_regression',split='test',stratum='known development regression control'))
    checks.equal(read(Path(args.output)/'layouts.json'),dict(layouts=expected,scope='development',source_ids=[r['source_id'] for r in expected]),
                 'Exactly the five known misses plus successful regression control')
    order=[r['id'] for r in expected];random.Random(spec['processing_seed']).shuffle(order)
    checks.equal(spec['processing_order'],order,'Target processing order frozen')
    return expected


def verify_slots(study,layouts,wrappers,manifest,checks,*,native_started):
    from .planning_budget_plan import _wrap_choice
    value=read(study/'choices.json')
    expected_slots=[_wrap_choice(w['site']['choice'],w['variant']) for w in wrappers]
    checks.equal(value['slots'],expected_slots,'One independently reconstructed decision per target and method')
    checks.equal([(s['context_id'],s['method']) for s in expected_slots],
                 [(r['id'],v) for r in layouts for v in METHODS],'Twelve decisions including setup declines')
    requests={v:[] for v in METHODS};channels={v:[] for v in NATIVE_CHANNELS}
    for wrapper,slot in zip(wrappers,expected_slots,strict=True):
        method=wrapper['variant'];site=wrapper['site'];request=slot['request']
        lane='baseline' if method=='baseline' else 'parallel/'+site['selected_adapter_directory']
        checks.require(lane in channels,'Selected proof namespace is declared')
        repeated=[] if request is None else [dict(copy.deepcopy(request),id=request['id']+f'-repeat-{n}',repeat=n,source_request_id=request['id']) for n in (1,2,3)]
        requests[method].extend(repeated);channels[lane].extend(repeated)
    checks.equal(value['requests'],requests,'Exactly three unchanged repeats; unique identity only')
    checks.equal(value['channels'],channels,'Each repeated command uses the immutable admitting lane')
    ids=[r['id'] for rs in channels.values() for r in rs]
    checks.equal(len(ids),len(set(ids)),'No duplicate release IDs')
    checks.require(len(ids)<=36,'At most36 decision repeats')
    spec=read(study/'contract.json')
    expected_bindings={n:sha(study/n) for n in (*spec['frozen_inputs'],'contract.json','plans.json')}
    checks.equal(value['bindings'],expected_bindings,'Exact frozen-choice dependency set')
    verify_bindings(study,value['bindings'],checks,'Pre-release dependencies')
    checks.require(all(manifest[n]==h for n,h in value['bindings'].items()),'Frozen dependencies in final manifest')
    if native_started:
        started=read(study/'native-started.json')
        checks.equal(started['bindings'],dict(value['bindings'],**{'choices.json':sha(study/'choices.json')}),
                     'All completed choices sealed before first native release')
        checks.equal(started['requests_sha256'],digest(channels),'Actual native plan matches frozen repeats and namespaces')
        verify_bindings(study,started['bindings'],checks,'Native-start exact bytes')
    return value,{'test_'+v:rs for v,rs in channels.items()}


def summarize_independent(layouts,wrappers,rows):
    from .planning_budget_plan import _wrap_choice
    from .hybrid_miss_audit import classify
    from .intergrid_route_selector_plan import STANDARD_CAPTURES
    from .fast_planning_audit import latency
    sites={(w['site']['layout']['id'],w['variant']):w['site'] for w in wrappers}
    result=dict(positions=[],methods={},latency={},decision_latency={})
    for layout in layouts:
        row=dict(id=layout['id'],source_id=layout['source_id'],bucket_xy_m=layout['bucket_xy_m'],role=layout['role'],methods={})
        for method in METHODS:
            site=sites[layout['id'],method];slot=_wrap_choice(site['choice'],method)
            outcomes=[] if slot['request'] is None else [classify(rows[slot['id']+f'-repeat-{n}']) for n in (1,2,3)]
            row['methods'][method]=dict(requested=3,checked_shot=slot['request'] is not None,released=len(outcomes),
                captures=sum(o in STANDARD_CAPTURES for o in outcomes),outcomes=outcomes,
                request_wall_s=site['request_wall_s'],decision_latency_s=site['decision_latency_s'])
        parallel=sites[layout['id'],'parallel'];alternative=parallel['lanes'].get('alternative') or {}
        state=alternative.get('board_alternative') or {}
        row['parallel_lane']=dict(triggered=parallel['trigger'] is not None,changed_choice=parallel['selected_lane']=='alternative',
            checked_candidates=len(state.get('records',[])),admitted_candidates=len(state.get('admitted',[])))
        result['positions'].append(row)
    for method in METHODS:
        values=[r['methods'][method] for r in result['positions']]
        result['methods'][method]=dict(positions=len(layouts),requested=3*len(layouts),checked_positions=sum(v['checked_shot'] for v in values),
            releases=sum(v['released'] for v in values),captures=sum(v['captures'] for v in values),reliable_positions=sum(v['captures']==3 for v in values))
        result['latency'][method]=latency([v['request_wall_s'] for v in values]);result['decision_latency'][method]=latency([v['decision_latency_s'] for v in values])
    return result


def verify_preparation(args,spec,manifest,checks):
    study=Path(args.output);layouts=verify_layouts(args,spec,checks)
    prior,home=read(study/'training-prior.json'),read(study/'setup-home.json')
    wrappers,snapshots,ranker=verify_original_plans(study,layouts,spec,args.assets,prior,home,manifest,checks)
    for wrapper in wrappers:
        if wrapper['variant']=='parallel':
            verify_parallel_site(wrapper['site'],snapshots[wrapper['site']['layout']['id'],'parallel'],study/'native-parallel',
                                 manifest,prior,home,ranker,spec,checks)
    return layouts,wrappers,prior,home


def preflight(args):
    import os
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('Preparation audit requires hidden GPUs')
    study=Path(args.output);checks=Checks();spec,_=verify_context(args,checks)
    manifest={p.relative_to(study).as_posix():sha(p) for p in study.rglob('*') if p.is_file() and p.name!='report.json'}
    layouts,wrappers,_,_=verify_preparation(args,spec,manifest,checks)
    verify_slots(study,layouts,wrappers,manifest,checks,native_started=False)
    return dict(version='sircus-parallel-board-preparation-audit-v1',status='passed',scope='CPU preparation only',
        checks_count=checks.count,development_positions=6,plans=12,native_execution_performed=False,
        deferred=['Native-start binding','Native64 trajectory/contact scoring','Pre/post controls','Final report/evidence binding'],
        plans_sha256=sha(study/'plans.json'),choices_sha256=sha(study/'choices.json'),contract_sha256=sha(study/'contract.json'),
        verifier_source_sha256=sha(Path(__file__)))


def audit(args):
    from . import replay650_verify as replay
    from . import direct_path_first_audit as shared
    study,parent=Path(args.output),Path(args.parent_study);checks=Checks();spec,source=verify_context(args,checks)
    report=read(study/'report.json')
    checks.equal((report['version'],report['status'],report['phase'],report['scope'],report['physics_substeps']),
                 (STUDY_VERSION,'completed','completed','development',64),'Completed development study only')
    checks.equal(report['budgets'],dict(positions=6,repeats=3,planning_s=30.),'Declared target/repeat/budget denominator')
    checks.require(0<seconds(report['gpu_lease_s'])<=1800,'Within authorized native lease cap')
    checks.equal(report['native_completed'],True,'Native runner completed every requested packet')
    manifest=replay.check_manifest(study,checks)
    checks.equal(report['evidence_manifest_sha256'],sha(study/'evidence-manifest.json'),'Report binds complete raw evidence')
    layouts,wrappers,prior,home=verify_preparation(args,spec,manifest,checks)
    choices,expected=verify_slots(study,layouts,wrappers,manifest,checks,native_started=True)
    rows,controls=shared.audit_packets(study,parent,args.assets,spec,report,manifest,expected,prior,home,checks,variants=NATIVE_CHANNELS)
    checks.equal(controls,18,'Three direct, table and negative controls before and after')
    checks.require(len(rows)+controls<=54,'Maximum36 repeated decisions plus18 controls')
    checks.equal(report['progress'],dict(prepared=12,planned=12,completed_releases=len(rows),planned_releases=len(rows)),
                 'All twelve decisions and their exact repeats accounted')
    comparison=summarize_independent(layouts,wrappers,rows)
    checks.equal(report['comparison'],comparison,'Independently rescored captures, repeats, and latency')
    states=[r['parallel_lane'] for r in comparison['positions']]
    parallel=dict(activated=sum(s['triggered'] for s in states),changed_choice=sum(s['changed_choice'] for s in states))
    checks.equal(report['parallel'],parallel,'Actual concurrent-lane activity')
    return dict(version=VERSION,status='passed',scope='development',checks_count=checks.count,comparison=comparison,parallel=parallel,
        controls=controls,test_releases=len(rows),frozen_source_root=str(source),verifier_source_sha256=sha(Path(__file__)),
        report_sha256=sha(study/'report.json'),choices_sha256=sha(study/'choices.json'),evidence_manifest_sha256=sha(study/'evidence-manifest.json'))


def verify_alternative_snapshot(snapshot,base,layout,directory,prior,home,resolutions,ranker,spec,checks):
    from .parallel_board_plan import VERSION as PLAN_VERSION,RULE
    from .board_fallback_plan import fallback_menu
    from .board_pose_recovery_audit import recompute_screen,ranked_diverse,verify_admissions
    from .clearance_expansion_plan import needs_expansion
    from .planning_budget_plan import _wrap_choice
    from .expand_geometry import WIDE_BOUNDS
    state=snapshot['board_alternative'];label=layout['id']+'/alternate'
    checks.equal((state['version'],state['protocol']),(PLAN_VERSION,spec['parallel_rule']),label+'/exact frozen protocol')
    checks.equal((state['source_choice'],state['source_choice_sha256']),(base,choice_hash(base)),label+'/authenticated trigger source')
    checks.equal(state['bounds'],{k:WIDE_BOUNDS[k] for k in ('low','high')},label+'/unchanged action bounds')
    checks.require(type(state['completed']) is bool,label+'/explicit completion')
    checks.equal(snapshot['generated'],{},label+'/no hidden learned menu in geometric lane')
    checks.require(all(not value for stage in snapshot['plan'].values() for value in stage.values()),label+'/original plan not fabricated in alternate')
    checks.equal(snapshot['completed_proposal_ids'],[r['proposal_id'] for r in state['records']],label+'/complete records only')
    checks.equal(snapshot['ranked_proposal_ids'],state['check_order'],label+'/declared deterministic order')
    if snapshot['active_stage']=='alternate_initialized':
        checks.require(not any(state[k] for k in ('proposals','nominal_predictions','wrist_screen','check_order','records','admitted','predictions','replacement_decisions')),label+'/initialization has no unperformed checks')
        checks.equal((state['completed'],state['selected_candidate_id']),(False,None),label+'/initial state')
        return base
    proposals=fallback_menu(layout,base['request'],state['bounds']['low'],state['bounds']['high'])
    initial=[p for p in state['proposals'] if 'retry_of' not in p]
    retries=[p for p in state['proposals'] if 'retry_of' in p]
    checks.equal(initial,proposals,label+'/independent deterministic geometry menu')
    checks.require(len(retries)<=RULE['multistart_retry_limit'],label+'/bounded alternate joint retries')
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
            best.update(prediction=candidate,candidate_pool='parallel_board_geometry',intended_route=request['intended_route'],
                        available_routes=sorted(set(base.get('available_routes',[]))|{request['intended_route']}),request=copy.deepcopy(request))
            best=_wrap_choice(best,'warm_early');best['predicted_flight_risk']=needs_expansion(best);winner=request['id']
    checks.equal(state['replacement_decisions'],decisions,label+'/independently reconstructed finite-clearance replacement decisions')
    checks.equal(state['selected_candidate_id'],winner,label+'/strictly better checked scores only; ties retain prefix')
    if snapshot['active_stage']=='finished':
        checks.equal((len(records),state['completed']),(len(order),True),label+'/normal finish exhausts declared candidates')
    return best


def risky(choice):
    if choice.get('request') is None:return False
    prediction=choice.get('prediction') or {};route=prediction.get('predicted_route')
    value=(prediction.get(route) or {}).get('arm_clearance_m') if route in ('direct','table') else None
    return type(value) not in (int,float) or not math.isfinite(value) or value<.005


def verify_parallel_site(site,original_snapshots,directory,manifest,prior,home,ranker,spec,checks):
    from .parallel_board_plan import VERSION as PLAN_VERSION,RULE
    from .stalled_reference_plan import request_mode
    verify_shared_timing(site,checks)
    label=site['layout']['id']+'/parallel';original=site['lanes']['original'];alternative=site['lanes']['alternative']
    checks.equal((site['version'],site['protocol'],site['method'],site['budget_s']),
                 (PLAN_VERSION,RULE,'warm_early',30.),label+'/exact parallel protocol')
    checks.equal(set(site['lanes']),{'original','alternative'},label+'/two owned lanes only')
    clock=site['shared_clock'];decision=seconds(site['decision_latency_s'])
    checks.numeric(clock['deadline']-clock['started'],30.,label+'/same absolute deadline',1e-6)
    checks.equal((clock['cutoff_elapsed_s'],clock['original_interrupted_by_alternative']),(decision,False),label+'/original never interrupted by alternative')
    checks.require(decision<=30.,label+'/bounded shared decision')
    checks.equal((site['deadline_reached'],site['status']),(decision==30.,'deadline' if decision==30. else 'completed'),label+'/honest outer completion')
    checks.require(seconds(site['request_wall_s'])+1e-6>=seconds(site['startup_s'])+decision+seconds(site['stop_overhead_s']),label+'/full request includes startup and shutdown')
    checks.require(original['launch_offset_s']<=original['readiness_offset_s']<=0.,label+'/original initialization precedes clock')
    checks.equal(original['available_budget_at_launch_s'],30.,label+'/original has full decision budget')
    checks.require(original['decision_latency_s']<=decision,label+'/original independent progress within shared budget')
    eligible=[(i,s) for i,s in enumerate(original_snapshots) if risky(s['choice'])]
    trigger=site['trigger'];source=None
    if trigger is None:
        checks.require(not eligible,label+'/every accepted risky source starts alternate search')
        checks.equal(alternative,None,label+'/no alternate without authenticated source')
    else:
        checks.require(bool(eligible),label+'/trigger requires checked risky snapshot')
        index,snapshot=eligible[0];receipt=original['accepted_receipts'][index];source=snapshot['choice']
        checks.equal((trigger['original_receipt_sequence'],trigger['file'],trigger['sha256'],trigger['choice_sha256'],trigger['selection_uses_flight_labels']),
                     (index,receipt['file'],receipt['sha256'],choice_hash(source),False),label+'/first authenticated risky publication only')
        checks.require(receipt['received_elapsed_s']<=receipt.get('validated_elapsed_s',receipt['received_elapsed_s'])<=seconds(trigger['elapsed_s'])<=decision,
                       label+'/authenticate source before concurrent work begins')
        path=contained(directory,trigger['binding_file'])
        checks.equal(sha(path),trigger['binding_sha256'],label+'/immutable trigger receipt bytes')
        checks.equal(manifest[path.relative_to(directory.parent).as_posix()],sha(path),label+'/trigger in final manifest')
        expected_trigger={k:v for k,v in trigger.items() if k!='binding_sha256'}
        checks.equal(read(path),dict(trigger=expected_trigger,source_choice=source),label+'/trigger freezes source before alternate work')
        if alternative is None:
            checks.require(site['deadline_reached'] and site['selected_lane']=='original',label+'/trigger without launch only when shared cutoff prevented startup')
    alternate_snapshots=[]
    if alternative is not None:
        checks.require(trigger is not None,label+'/alternate has source')
        checks.require(0<=trigger['elapsed_s']<=seconds(alternative['launch_offset_s'])<=decision,label+'/alternate launch charged to original clock')
        checks.numeric(alternative['available_budget_at_launch_s'],30.-alternative['launch_offset_s'],label+'/no fresh alternate budget',1e-6)
        if alternative['readiness_offset_s'] is not None:
            checks.require(alternative['launch_offset_s']<=alternative['readiness_offset_s']<=decision,label+'/alternate initialization inside decision budget')
            checks.numeric(alternative['startup_s'],alternative['readiness_offset_s']-alternative['launch_offset_s'],label+'/alternate startup charged honestly',1e-6)
        checks.require(alternative['worker_pid']!=original['worker_pid'],label+'/independent process and engine state')
        verify_fresh_lifecycle(alternative,checks,ready_required=False)
        submanifest={k[len('native-parallel/'):]:v for k,v in manifest.items() if k.startswith('native-parallel/')}
        alternate_snapshots=verify_alternate_receipts(alternative,directory/'alternative',submanifest,checks,30.)
        resolutions={};previous=None;copy_receipt=None
        for snapshot,receipt in zip(alternate_snapshots,alternative['accepted_receipts'],strict=True):
            checks.require(alternative['readiness_offset_s']<=receipt['elapsed_s'],label+'/no geometry snapshot before engine ready')
            state=snapshot['board_alternative']
            if previous is not None:
                for key in ('version','protocol','source_choice','source_choice_sha256','context_copy_manifest','bounds'):
                    checks.equal(state[key],previous[key],label+'/immutable concurrent source/'+key)
                for key in ('proposals','check_order','records','admitted','replacement_decisions'):
                    checks.equal(state[key][:len(previous[key])],previous[key],label+'/append-only checked evidence/'+key)
                for key in ('nominal_predictions','wrist_screen','predictions'):
                    checks.require(all(state[key].get(k)==v for k,v in previous[key].items()),label+'/immutable prior prediction/'+key)
            copy_receipt=state['context_copy_manifest'];request=source['request'];relative=request['context_resolution_file'];expected_hash=request['context_resolution_sha256']
            expected_copy=dict(adapter_mode=request_mode(request),source_directory=str(directory/'original'),destination_directory=str(directory/'alternative'),
                files=[dict(file=relative,expected_sha256=expected_hash,source_sha256=sha(contained(directory/'original',relative)),
                            destination_sha256=sha(contained(directory/'alternative',relative)))])
            checks.equal(copy_receipt,expected_copy,label+'/one exact original resolution copied; no mutable tree transfer')
            checks.equal(expected_copy['files'][0]['source_sha256'],expected_hash,label+'/source context exact hash')
            checks.equal(expected_copy['files'][0]['destination_sha256'],expected_hash,label+'/destination context exact hash')
            copy_file=directory/'alternative'/'parallel-context-copies'/(request['context_id']+'.json')
            checks.equal(read(copy_file),copy_receipt,label+'/durable context-copy receipt')
            choice=verify_alternative_snapshot(snapshot,source,site['layout'],directory/'alternative',prior,home,resolutions,ranker,spec,checks)
            checks.equal(snapshot['choice'],choice,label+'/independent alternate choice reconstruction')
            previous=state
        verify_alternative_early_rule(alternative,alternate_snapshots,checks)
        if alternate_snapshots:checks.equal(alternative['board_alternative'],alternate_snapshots[-1]['board_alternative'],label+'/last accepted alternate state')
    original_choice=original['choice']
    candidate=alternative['choice'] if alternative is not None and alternative.get('board_alternative',{}).get('selected_candidate_id') else None
    worker_error=original.get('error') is not None or (alternative is not None and alternative.get('error') is not None)
    expected_selection=independent_final_selection(original_choice,candidate,worker_error)
    selected=expected_selection['selected_lane']
    checks.equal(site['selection'],expected_selection,label+'/final alternate compares with FINAL original, never stale trigger')
    checks.equal((site['selected_lane'],site['selected_adapter_directory'],site['choice']),
                 (selected,selected,original_choice if selected=='original' else candidate),label+'/preserve final original unless finite qualified improvement')
    path=contained(directory,site['selection_receipt_file'])
    checks.equal(sha(path),site['selection_receipt_sha256'],label+'/immutable final selection receipt')
    checks.equal(manifest[path.relative_to(directory.parent).as_posix()],sha(path),label+'/selection in complete manifest')
    checks.equal(read(path),dict(selection=expected_selection,trigger=trigger,decision_latency_s=decision,budget_s=30.,
        original_snapshot_sha256=original['accepted_snapshot_sha256'],alternative_snapshot_sha256=alternative['accepted_snapshot_sha256'] if alternative else None),
        label+'/selection binds both last accepted snapshots')
    for lane in (original,alternative):
        if lane is None:continue
        checks.require(seconds(lane['cpu_s_last_receipt'])>=seconds((lane.get('readiness_receipt') or {}).get('cpu_s',0.)),label+'/CPU receipt accounting')
        checks.equal(lane['cpu_measurement_complete'],lane['status']=='completed',label+'/deadline CPU sample explicitly incomplete')
    checks.numeric(site['cpu_s_last_receipts'],sum(lane['cpu_s_last_receipt'] for lane in (original,alternative) if lane),label+'/total sampled CPU',1e-6)


def independent_final_selection(original,candidate,worker_error=False):
    decision=independent_replacement_decision((candidate or {}).get('prediction'),original.get('prediction'))
    accepted=(not worker_error and original['request'] is not None and candidate is not None
              and candidate['request'] is not None and decision['accepted'])
    return dict(original_choice_sha256=choice_hash(original),alternative_choice_sha256=choice_hash(candidate) if candidate is not None else None,
        replacement_decision=decision,worker_error=bool(worker_error),selected_lane='alternative' if accepted else 'original')


def verify_shared_timing(site,checks):
    clock=site['shared_clock'];decision=seconds(site['decision_latency_s'])
    checks.numeric(clock['deadline']-clock['started'],30.,'Exactly one30s absolute decision window',1e-6)
    checks.require(decision<=30.,'Shared decision cutoff within budget')
    checks.equal((clock['cutoff_elapsed_s'],clock['original_interrupted_by_alternative']),(decision,False),'Original never sacrificed to alternate')
    for name,lane in site['lanes'].items():
        if lane is None:continue
        checks.equal(lane['budget_s'],30.,'Lane receipts use the same shared clock')
        checks.require(seconds(lane['decision_latency_s'])<=decision,'No lane runs after shared selection')
        for receipt in lane.get('accepted_receipts',[]):
            checks.require(seconds(receipt['received_elapsed_s'])<=seconds(receipt['validated_elapsed_s'])<=lane['decision_latency_s'],
                           'Only fully hash-validated snapshots inside the decision window may count')
        if name=='alternative':
            launch=seconds(lane['launch_offset_s'])
            checks.require(site['trigger'] is not None and site['trigger']['elapsed_s']<=launch<=decision,'Source before alternate launch')
            checks.numeric(lane['available_budget_at_launch_s'],30.-launch,'Alternate startup receives only remaining time',1e-6)
            if lane['readiness_offset_s'] is not None:
                checks.require(launch<=seconds(lane['readiness_offset_s'])<=decision,'Alternate initialization charged inside window')


def choice_hash(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def verify_alternate_receipts(site, native, manifest, checks, budget_s=30.):
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
    checks.numeric(site['request_wall_s']+site['launch_offset_s'],site['wall_planning_s'],identity+'/actual lane lifetime maps to shared clock',1e-6)
    checks.equal(site['receipt_verification_s'],0.,identity+'/receipt verification already charged online')
    checks.require(seconds(site['online_receipt_verification_s'])<=site['request_wall_s'],identity+'/bounded measured online verification')
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
    from .fast_planning_audit import verify_profile
    verify_profile(site, checks)
    return snapshots


def verify_alternative_early_rule(site,snapshots,checks):
    from .clearance_expansion_plan import needs_expansion
    first=None;previous=0
    for snapshot in snapshots:
        count=len(snapshot['completed_proposal_ids'])
        checks.require(previous<=count<=previous+1,'Alternate tests early return after every completed geometry')
        previous=count;state=snapshot['board_alternative'];score=(snapshot['choice'].get('prediction') or {}).get('score_m')
        eligible=(state['selected_candidate_id'] is not None and type(score) in (int,float) and math.isfinite(score)
                  and score>=.005 and not needs_expansion(snapshot['choice']))
        if first is None and eligible:first=snapshot
        if first is not None:
            checks.equal((snapshot['completed_proposal_ids'],snapshot['choice']),
                         (first['completed_proposal_ids'],first['choice']),'No alternate checks after first new eligible shot')
    if first is None:checks.require(site['return_reason']!='early_good_shot','Alternate early return requires newly admitted eligible shot')
    else:checks.equal(site['return_reason'],'deadline' if site['deadline_reached'] else 'early_good_shot','Alternate honors own early-return rule')


def verify_fresh_lifecycle(site,checks,*,ready_required=True):
    checks.equal((site['worker_generation'],site['worker_job_index'],site['worker_reused'],site['prior_worker_status'],site['lifecycle']),
                 (0,0,False,None,'fresh_worker_per_request'),'Explicit fresh parallel worker lifecycle')
    checks.require(type(site['worker_pid']) is int and site['worker_pid']>0,'Fresh worker process identity')
    checks.require(isinstance(site['job_id'],str) and bool(site['job_id']),'Fresh worker job identity')
    checks.equal(site['worker_pool_id'],site['job_id'],'Fresh process owns exactly one job')
    ready=site['readiness_receipt']
    if ready_required:checks.require(isinstance(ready,dict),'Original process initialized before shared clock')
    if ready is not None:
        checks.equal(set(ready),{'kind','pid','cpu_s'},'Exact fresh worker initialization receipt')
        checks.equal((ready['kind'],ready['pid']),('ready',site['worker_pid']),'Initialization binds correct process')
        seconds(ready['cpu_s'])


def verify_worker_lifecycles(wrappers,layouts,spec,checks):
    """Validate two lifecycle contracts without mutating the declared protocol."""
    baseline=[w for w in wrappers if w['variant']=='baseline']
    verify_workers(baseline,layouts,dict(spec,methods=['baseline']),checks,variants=('baseline',))
    original=[w['site'] for w in wrappers if w['variant']=='parallel']
    checks.equal(len(original),len(layouts),'One original process per parallel decision')
    checks.equal(len({s['worker_pid'] for s in original}),len(original),'Fresh original process per parallel target')
    checks.equal(len({s['job_id'] for s in original}),len(original),'Distinct fresh original jobs')
    for site in original:verify_fresh_lifecycle(site,checks)
