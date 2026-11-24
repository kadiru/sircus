"""Frozen development correction for three rising bucket-wall misses.

CPU candidate preparation is bounded and completes before native64 trials.
No fitting, scorer changes, live promotion or physical execution.
"""
from __future__ import annotations
import argparse
import copy
import json
import math
import os
from pathlib import Path
import shutil
import time

from .benchmark_verify import Checks, contained, digest, read, sha
from .sweep_worker import atomic
from .hybrid_miss_study import engine_for, native_inputs, verify_frozen

VERSION = 'sircus-bucket-approach-study-v1'
TARGETS = ((4, 'expanded'), (10, 'current'), (19, 'expanded'))
PRESERVED = ((4, 'current'), (10, 'expanded'), (19, 'current'), (47, 'expanded'))
REPEATS = 3
MAX_CHECKS = 40
CASE_LIMIT_S = 600
PROPOSAL_LIMIT_S = 30
GPU_CAP_S = 3600
SEARCH_RULE = dict(tilt_delta_deg=[-8., -4., 0., 4., 8.],
    gap_delta_m=[-.04, -.02, 0., .02, .04, .06, .10],
    maximum_setup_checks=MAX_CHECKS, proposal_limit_s=PROPOSAL_LIMIT_S,
    case_limit_s=CASE_LIMIT_S, groups=['local_tilt_gap', 'training_library'],
    ordering='Alternate nominal rim-ranked local and training-library candidates; necessary wrist screen first',
    selection='Highest full bucket-approach score, stable ID tie-break; require strict improvement over original; no contact ban',
    attempts_per_candidate=1, repeats=REPEATS)


def lines(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def publish(a, report):
    report['updated_at_unix'] = time.time()
    atomic(a.output/'report.json', report)
    print(json.dumps({k:report[k] for k in ('status','phase','progress','error','gpu_wait') if k in report}), flush=True)


def source_inputs(a):
    from .expanded_selector_diagnosis import verify_parent
    checks = Checks()
    report, audit, bound, verified = verify_parent(a.source_study, a.source_audit, checks)
    spec = read(bound('contract.json'))
    choices = {(r['context_id'], r['method']): r['request'] for r in read(bound('choices.json'))['slots']}
    rows = {r['id']:r for r in lines(bound('rows.jsonl'))}
    diagnosis = read(a.diagnosis)
    if (diagnosis['status'] != 'completed' or diagnosis['source']['report_sha256'] != sha(a.source_study/'report.json')
        or diagnosis['source']['audit_sha256'] != sha(a.source_audit)):
        raise ValueError('Use the bound completed four-pair diagnosis')
    cases, preserved, diagnostic = [], [], []
    for selections, output, success in ((TARGETS,cases,False),(PRESERVED,preserved,True),(((47,'current'),),diagnostic,False)):
        for number, method in selections:
            key = f'expanded-selector-test-{number:02d}'
            request = choices[key,method]
            row = rows[request['id']]
            checks.equal(row['board_first_score']['route_success'], success, 'Expected historical outcome')
            checks.require(row['native_release'] and not row['invalid_physics'], 'Valid original native flight')
            checks.equal(row['values'], request['values'], 'Original action')
            for prefix in ('context_resolution','prepared_connection'):
                relative = row['adapter_directory']+'/'+request[prefix+'_file']
                checks.equal(sha(bound(relative)),request[prefix+'_sha256'],'Original admission proof')
            checks.equal(sha(bound(row['adapter_directory']+'/'+row['native_cache_file'])),row['native_cache_sha256'],'Original trace')
            label = key+'-'+method if success or output is diagnostic else key
            output.append(dict(id=label, original=request, historical=row, source_method=method,
                source_directory=row['adapter_directory'], directory=('preserved-' if success else 'diagnostic-' if output is diagnostic else 'native-')+f'{number:02d}'))
    for name in ('training-prior.json','calibration.json','setup-home.json','candidate-sources.json','shot-library-rows.json','route-models.json'):
        bound(name)
    binding = dict(report_sha256=sha(a.source_study/'report.json'),audit_sha256=sha(a.source_audit),
        evidence_manifest_sha256=sha(a.source_study/'evidence-manifest.json'),files=verified,
        diagnosis_sha256=sha(a.diagnosis))
    return cases, preserved, diagnostic, spec, binding


def proposal_menu(case, sources, library_rows, checkpoint, low, high):
    from .expanded_selector_plan import expanded_menu, VERSION as MENU_VERSION
    from .intergrid_library_plan import libraries
    original=case['original']; rows=[]
    # The exact original proof is already copied into the immutable namespace.
    # The training library can offer that same command again under a new ID;
    # reserve its geometry before preparing any alternatives.
    seen={digest([original['values'],original['board_heading_offset_deg'],original['board_twist_deg']])}
    def add(row,group):
        values=row['values']; key=digest([values,row['board_heading_offset_deg'],row['board_twist_deg']])
        if (key in seen or any(not lo<=v<=hi for v,lo,hi in zip(values,low,high))
                or math.dist(values[:2], original['bucket_xy_m'])<=.15): return
        seen.add(key)
        rows.append(dict(row,id=f"{case['id']}-approach-{len(rows):03d}",context_id=case['id'],
                         phase='diagnostic',role='diagnostic',method='bucket_approach',group=group))
    for tilt in SEARCH_RULE['tilt_delta_deg']:
        for gap in SEARCH_RULE['gap_delta_m']:
            if tilt==0 and gap==0: continue
            values=list(original['values']); values[3]+=tilt; values[4]+=gap
            add(dict(values=values,bucket_xy_m=original['bucket_xy_m'],
                board_heading_offset_deg=original['board_heading_offset_deg'],board_twist_deg=original['board_twist_deg'],
                intended_route=original['intended_route'],seed_q10=original['selected_q10'],seed_q14=original['selected_q14'],
                source_request_id=original['id']), 'local_tilt_gap')
    layout=dict(id=case['id'],bucket_xy_m=original['bucket_xy_m'],split='development')
    libs=dict(expanded_menu_version=MENU_VERSION,expanded_sources=sources,
        baseline_libraries=libraries(library_rows['original'],library_rows['additions']))
    for row in expanded_menu(layout,libs,checkpoint): add(row,'training_library')
    return rows


def ordered_proposals(proposals, nominal, wrist):
    def key(p):
        score=nominal[p['id']].get('score_m')
        return (not isinstance(score,(int,float)) or not math.isfinite(score),-score if score is not None and math.isfinite(score) else 0.,p['id'])
    groups=[sorted([p for p in proposals if p['group']==group and wrist[p['id']]['passed']],key=key)
            for group in SEARCH_RULE['groups']]
    return [group[i] for i in range(max(map(len,groups),default=0)) for group in groups if i<len(group)][:MAX_CHECKS]


def approach_score(value):
    score=value.get('predicted_score_m')
    return float(score) if type(score) in (int,float) and math.isfinite(score) else -math.inf


def prediction_score(prediction):
    value=prediction.get('score_m')
    return float(value) if type(value) in (int,float) and math.isfinite(value) else -math.inf


def choose_correction(original_approach, candidates, approaches):
    ordered=sorted(candidates,key=lambda row:(-approach_score(approaches[row['id']]),row['id']))
    return ordered[0] if ordered and approach_score(approaches[ordered[0]['id']])>approach_score(original_approach)+1e-9 else None


def _prepare_case(args, case, source_contract, reference):
    import torch
    torch.set_num_threads(1)
    from .board_first_selector_plan import BoardFirstFlightRanker
    from .bucket_approach import BucketApproach
    from .board_pose_recovery_study import nominal_prediction,wrist_screen
    from .cell1050_rebound_study import preparation_time_limit,SearchPreparationExpired
    from .interactive_stalled_planner import copy_prepared_proofs
    from .arm_check_speed import install_vectorized_gate
    from .arm_path_speed import install_path_short_circuit
    from .parallel_capture_grid_workers import load_resolution
    a=argparse.Namespace(**args); original=case['original']; directory=a.output/case['directory']
    copy_prepared_proofs(a.source_study/case['source_directory'],directory,original)
    def rebuild():
        engine=engine_for(a,original,directory,reference)
        load_resolution(engine,original['bucket_xy_m'])
        install_vectorized_gate(engine.adapter.resolver.planner.gate)
        install_path_short_circuit(engine.adapter.resolver.planner)
        return engine
    engine=rebuild()
    ranker=BoardFirstFlightRanker(reference['scene'],read(a.output/'calibration.json')['score_config'],a.assets,source_contract['flight_ranking']['rule'])
    checker=BucketApproach(reference['scene'])
    original_prediction=ranker.predict(original)
    original_approach=checker.assess(original,ranker,prediction=original_prediction)
    low,high=engine.adapter.resolver.solver.low.tolist(),engine.adapter.resolver.solver.high.tolist()
    proposals=proposal_menu(case,read(a.output/'candidate-sources.json'),read(a.output/'shot-library-rows.json'),
                            read(a.output/'route-models.json'),low,high)
    nominal={p['id']:nominal_prediction(ranker,reference['scene'],p) for p in proposals}
    wrist={p['id']:wrist_screen(reference['scene'],p,engine.adapter.resolver.models) for p in proposals}
    ordered=ordered_proposals(proposals,nominal,wrist)
    checked=[]; admitted=[]; records=[]; predictions={}; approaches={}; begin=time.monotonic()
    result=dict(case,proposals=proposals,nominal_predictions=nominal,wrist_reach=wrist,
        preparation_order=[p['id'] for p in ordered],checked=checked,candidates=admitted,records=records,
        predictions=predictions,approaches=approaches,original_prediction=original_prediction,
        original_approach=original_approach,bounds=dict(low=low,high=high),preparation_s=0.)
    for proposal in ordered:
        remaining=CASE_LIMIT_S-(time.monotonic()-begin)
        if remaining<=0: break
        checked.append(proposal)
        try:
            with preparation_time_limit(min(PROPOSAL_LIMIT_S,remaining)):
                added,logs=engine.prepare([proposal],one_per_setting=True)
            for row in added:
                predictions[row['id']]=ranker.predict(row)
                approaches[row['id']]=checker.assess(row,ranker,prediction=predictions[row['id']])
            admitted.extend(added);records.extend(logs)
        except SearchPreparationExpired:
            records.append(dict(proposal_id=proposal['id'],proposal_sha256=digest(proposal),admitted=0,
                admitted_request_ids=[],reason='bounded_setup_timeout',timing=dict(budget_s=min(PROPOSAL_LIMIT_S,remaining))))
            engine=rebuild()
        result=dict(case,proposals=proposals,nominal_predictions=nominal,wrist_reach=wrist,
            preparation_order=[p['id'] for p in ordered],checked=checked,candidates=admitted,records=records,
            predictions=predictions,approaches=approaches,original_prediction=original_prediction,
            original_approach=original_approach,bounds=dict(low=low,high=high),preparation_s=time.monotonic()-begin)
        atomic(a.output/f"preparation-{case['id'][-2:]}.json",result)
    without_bucket = max([original]+admitted,key=lambda r:(prediction_score(original_prediction if r is original else predictions[r['id']]),r['id']))
    result.update(corrected=choose_correction(original_approach,admitted,approaches),
        selected_without_bucket_check=without_bucket,status='prepared')
    if engine.adapter.env is not None or torch.cuda.is_initialized():raise RuntimeError('CPU preparation allocated CUDA')
    atomic(a.output/f"preparation-{case['id'][-2:]}.json",result)
    return result


def prepare(a):
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('CPU preparation requires CUDA_VISIBLE_DEVICES empty')
    from concurrent.futures import ProcessPoolExecutor,as_completed
    import multiprocessing
    from .board_first_selector_plan import BoardFirstFlightRanker
    from .bucket_approach import BucketApproach
    from .interactive_stalled_planner import copy_prepared_proofs
    from .intergrid_system_study import require_storage
    cases,preserved,diagnostic,source_contract,binding=source_inputs(a)
    spec,parent_binding,_,reference=native_inputs(a)
    if spec['identity']!=source_contract['identity']:raise ValueError('Scene identity differs from frozen controls')
    a.output.mkdir(parents=True,exist_ok=False); require_storage(a.output)
    for name in ('training-prior.json','calibration.json','setup-home.json','candidate-sources.json','shot-library-rows.json','route-models.json'):
        shutil.copyfile(a.source_study/name,a.output/name)
    shutil.copyfile(a.parent_study/'verification.json',a.output/'verification.json')
    shutil.copyfile(a.diagnosis,a.output/'diagnosis.json')
    report=dict(version=VERSION,status='running',phase='preparing',audit_status='pending',controls={},
        physics_substeps=64,progress=dict(completed=0,planned=33),scope='selected development positions')
    publish(a,report)
    plan=dict(version=VERSION,source=binding,parent=parent_binding,identity=spec['identity'],scene=spec['scene'],
        source_contract_flight_ranking=source_contract['flight_ranking'],repeats=REPEATS,search_rule=SEARCH_RULE,cases=[],preserved_cases=[],diagnostic_cases=[])
    ranker=BoardFirstFlightRanker(reference['scene'],read(a.output/'calibration.json')['score_config'],a.assets,source_contract['flight_ranking']['rule'])
    checker=BucketApproach(reference['scene'])
    for case in preserved+diagnostic:
        copy_prepared_proofs(a.source_study/case['source_directory'],a.output/case['directory'],case['original'])
        prediction=ranker.predict(case['original'])
        plan['preserved_cases' if case in preserved else 'diagnostic_cases'].append(dict(case,original_prediction=prediction,original_approach=checker.assess(case['original'],ranker,prediction=prediction)))
    with ProcessPoolExecutor(max_workers=2,mp_context=multiprocessing.get_context('spawn')) as pool:
        futures=[pool.submit(_prepare_case,vars(a),case,source_contract,reference) for case in cases]
        for future in as_completed(futures):
            plan['cases'].append(future.result());plan['cases'].sort(key=lambda c:c['id'])
            atomic(a.output/'preparation.json',plan)
    report.update(phase='prepared',prepared_cases=len(plan['cases']));publish(a,report)


def schedule(plan):
    slots=[]
    for case in plan['preserved_cases']+plan['diagnostic_cases']+plan['cases']:
        variants=('preserved',) if case in plan['preserved_cases'] else ('diagnostic_control',) if case in plan['diagnostic_cases'] else ('original','corrected')
        for repeat in range(REPEATS):
            for variant in variants if repeat%2==0 else tuple(reversed(variants)):
                source=case['original'] if variant in ('preserved','diagnostic_control') else case[variant]
                request=None if source is None else dict(copy.deepcopy(source),id=f"{case['id']}-{variant}-{repeat}",
                    context_id=case['id'],variant=variant,repeat=repeat,method=variant,phase='preservation' if variant=='preserved' else 'diagnostic',source_request_id=source['id'])
                slots.append(dict(case_id=case['id'],variant=variant,repeat=repeat,directory=case['directory'],request=request))
    return slots


def freeze(a):
    from .corrected_grid import source_manifest
    plan=read(a.output/'preparation.json')
    if len(plan['cases'])!=3 or len(plan['preserved_cases'])!=4 or len(plan['diagnostic_cases'])!=1 or (a.output/'contract.json').exists():raise ValueError('Fresh complete preparation required')
    slots=schedule(plan); atomic(a.output/'slots.json',slots)
    files={str(p.relative_to(a.output)):sha(p) for p in a.output.rglob('*') if p.is_file() and p.name!='report.json'}
    spec=dict(version=VERSION,identity=plan['identity'],scene=plan['scene'],parent=plan['parent'],source=plan['source'],
        source_sha256=source_manifest(Path(__file__).resolve().parents[2]),frozen_files=files,physics_substeps=64,
        repeats=3,controls=18,preservation_requests=12,planned_requests=33,maximum_native_releases=51,diagnostic_control_requests=3,gpu_cap_s=GPU_CAP_S,
        search_rule=SEARCH_RULE,scoring_version='sircus-board-first-native-capture-v4',
        scope='Three previously inspected wall-miss positions; frozen prediction correction, no retraining or generalization claim')
    spec['fingerprint']=digest(spec); atomic(a.output/'contract.json',spec)
    report=read(a.output/'report.json');report.update(phase='frozen',progress=dict(completed=0,planned=33));publish(a,report)


def counts(source, found):
    scores=[row['board_first_score'] for row in found]
    return dict(requested=3,completed=len(found),released=len(found),setup_rejected=3 if source is None else 0,
        captures=sum(score['route_success'] for score in scores),invalid_physics=sum(score['invalid_physics'] for score in scores),
        robot_contacts=sum(score['robot_contact_steps']>0 for score in scores),
        bucket_contacts=sum(score['bucket_contact_before_entry'] for score in scores),
        outcomes=[score['route_class'] if score['route_success'] else 'invalid_physics' if score['invalid_physics'] else 'miss' for score in scores],rows=found)


def summarize(plan,rows):
    cases=[]; preservation=[];diagnostic_controls=[]
    for case in plan['cases']:
        groups={variant:counts(case[variant],[r for r in rows if r.get('context_id')==case['id'] and r.get('variant')==variant]) for variant in ('original','corrected')}
        cases.append(dict(id=case['id'],bucket_xy_m=case['original']['bucket_xy_m'],results=groups))
    for case in plan['preserved_cases']:
        found=[r for r in rows if r.get('context_id')==case['id'] and r.get('variant')=='preserved']
        preservation.append(dict(id=case['id'],bucket_xy_m=case['original']['bucket_xy_m'],**counts(case['original'],found)))
    for case in plan['diagnostic_cases']:
        found=[r for r in rows if r.get('context_id')==case['id'] and r.get('variant')=='diagnostic_control']
        diagnostic_controls.append(dict(id=case['id'],bucket_xy_m=case['original']['bucket_xy_m'],**counts(case['original'],found)))
    return dict(cases=cases,preservation=preservation,diagnostic_controls=diagnostic_controls)


def run(a):
    from .cell1050_rebound_study import annotate,control_requests
    from .coverage_training_study import append
    from .expanded_selector_study import attach_native_scores,board_first_controls
    from .gpu_slot import foreign_processes,gpu_slot
    from .intergrid_route_labels import annotate_routes
    from .intergrid_system_study import gpu_deadline,require_storage
    from .parallel_capture_grid_study import control_mirror
    from .table_correction_native import MatchedEngine
    spec=verify_frozen(a);_,_,_,reference=native_inputs(a)
    if a.preflight_audit is None:raise ValueError('Independent preparation audit required')
    audit=read(a.preflight_audit)
    if (audit.get('status')!='passed' or audit.get('version')!='sircus-bucket-approach-preparation-audit-v1'
            or audit.get('contract_sha256')!=sha(a.output/'contract.json') or audit.get('slots_sha256')!=sha(a.output/'slots.json')):
        raise ValueError('Independent preparation audit must bind exact frozen choices')
    if (a.output/'native-started.json').exists():raise ValueError('Never silently repeat native execution')
    plan=read(a.output/'preparation.json');slots=read(a.output/'slots.json');report=read(a.output/'report.json')
    calibration=read(a.output/'calibration.json');home=read(a.output/'setup-home.json')
    def waiting(event):report.update(phase='waiting_for_4090',gpu_wait=event);publish(a,report)
    with gpu_slot(a.gpu_lock,timeout_s=a.gpu_wait_s,poll_s=30,notify=waiting):
        if foreign_processes():raise RuntimeError('4090 has protected foreign work')
        verify_frozen(a)
        with (a.output/'native-started.json').open('x') as stream:
            json.dump(dict(contract_sha256=sha(a.output/'contract.json'),slots_sha256=sha(a.output/'slots.json'),
                preparation_sha256=sha(a.output/'preparation.json'),preflight_audit_sha256=sha(a.preflight_audit),started_unix=time.time()),stream)
        begin=time.monotonic()
        with gpu_deadline(GPU_CAP_S):
            allcases=plan['preserved_cases']+plan['diagnostic_cases']+plan['cases']
            engines={case['id']:engine_for(a,case['original'],a.output/case['directory'],reference) for case in allcases}
            sources=read(a.parent_study/'control-replays.json');mirror=control_mirror(a.parent_study,a.output,sources,False)
            control=MatchedEngine(reference,a.assets,a.output/'native-controls',parent=mirror,sources=sources,
                prior=read(a.parent_study/'old-prior.json'),calibration=calibration,home=home)
            adapters=[engine.adapter for engine in engines.values()]+[control.engine.adapter]
            factory,allocated=adapters[0].environment_factory,[]
            def shared():
                if not allocated:allocated.append(factory())
                return allocated[0]
            for adapter in adapters:adapter.environment_factory=shared
            options=read(a.parent_study/'native-yaw/effective-options.json');geometry=read(a.parent_study/'native-yaw/native-compiled-geometry.json')
            observations=[];gates={};packet=0
            def execute(engine,batch,phase,directory,is_control=False):
                nonlocal packet
                if foreign_processes():raise RuntimeError('Foreign workload on leased4090')
                verify_frozen(a);require_storage(a.output)
                name=f'requests/batch-{packet:04d}.json';packet+=1
                atomic(a.output/name,dict(rows=batch,phase=phase,adapter_directory=directory,control=is_control))
                rows,timing=engine.evaluate(batch,phase)
                if [r['id'] for r in rows]!=[r['id'] for r in batch] or any(not r['native_release'] for r in rows):raise ValueError('Expected one native release per admitted command')
                adapter=engine.engine.adapter if is_control else engine.adapter
                if adapter.env.effective_options!=options or adapter.env.compiled_geometry!=geometry:raise ValueError('Pinned native64 controls differ')
                if not is_control:annotate(rows,a.output/directory);annotate_routes(rows,a.output/directory)
                attach_native_scores(reference['scene'],calibration['score_config'],rows,adapter.env.last_arrays)
                for row in rows:row.update(adapter_directory=directory,request_packet=name,request_packet_sha256=sha(a.output/name))
                observations.extend(rows);append(a.output/'rows.jsonl',rows);append(a.output/'timing.jsonl',[dict(packet=name,phase=phase,timing=timing)])
                report.update(phase=phase,progress=dict(completed=sum(r.get('variant') in ('original','corrected','preserved','diagnostic_control') for r in observations),planned=33),controls=gates,**summarize(plan,observations))
                publish(a,report);return rows
            for phase in ('controls_pre','test','controls_post'):
                if phase!='test':
                    gates[phase]=board_first_controls(execute(control,control_requests(sources,phase),phase,'native-controls',True))
                    if gates[phase]['status']!='passed':raise RuntimeError('Control gate failed')
                else:
                    for case in allcases:
                        batch=[s['request'] for s in slots if s['case_id']==case['id'] and s['request'] is not None]
                        execute(engines[case['id']],batch,'preservation' if case in plan['preserved_cases'] else 'diagnostic',case['directory'])
        report.update(native_completed=True,phase='native_completed',gpu_lease_s=time.monotonic()-begin,native_releases=len(observations),controls=gates)
    report.pop('gpu_wait',None);publish(a,report)


def finalize(a):
    verify_frozen(a)
    report=read(a.output/'report.json')
    if report.get('native_completed') is not True:raise ValueError('Native completion required')
    rows=lines(a.output/'rows.jsonl');plan=read(a.output/'preparation.json')
    expected={slot['request']['id'] for slot in schedule(plan) if slot['request'] is not None}
    observed=[r for r in rows if r['adapter_directory']!='native-controls']
    if {r['id'] for r in observed}!=expected or len(observed)!=len(expected):raise ValueError('Every frozen command exactly once')
    atomic(a.output/'evidence-manifest.json',{str(p.relative_to(a.output)):sha(p) for p in sorted(a.output.rglob('*')) if p.is_file() and p.name not in ('report.json','evidence-manifest.json')})
    report.update(status='completed',phase='completed',audit_status='pending',**summarize(plan,rows),
        evidence_manifest_sha256=sha(a.output/'evidence-manifest.json'))
    publish(a,report)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('source-study','source-audit','diagnosis','parent-study','parent-audit','reference','assets','output','gpu-lock'):
        parser.add_argument('--'+key,type=Path,required=True)
    parser.add_argument('--preflight-audit',type=Path)
    parser.add_argument('--gpu-wait-s',type=int,default=1800)
    parser.add_argument('--stage',choices=('prepare','freeze','run','finalize'),required=True)
    args=parser.parse_args()
    output_preexisted=args.output.exists()
    try:globals()[args.stage](args)
    except TimeoutError:
        if (args.output/'report.json').exists() and not (args.output/'native-started.json').exists():
            report=read(args.output/'report.json');report.update(status='waiting',phase='frozen',error='GPU wait expired; immutable preparation remains reusable');publish(args,report)
        raise
    except Exception as error:
        # Only mark this invocation's newly created output. A mistakenly reused
        # older path must remain unchanged, including its status report.
        if args.stage=='prepare' and not output_preexisted and (args.output/'report.json').exists():
            report=read(args.output/'report.json')
            report.update(status='failed',phase='preparation_failed',error=f'{type(error).__name__}: {error}')
            publish(args,report)
        raise


if __name__=='__main__':main()
