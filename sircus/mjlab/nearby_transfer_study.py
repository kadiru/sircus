"""Bounded nearby-successful-shot transfer; CPU preparation then frozen native64.

Five known development misses. No retraining, live promotion or service action.
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

from .benchmark_verify import digest, read, sha
from .sweep_worker import atomic
from .hybrid_miss_study import native_inputs, engine_for, verify_frozen
from .board_pose_recovery_study import nominal_prediction, wrist_screen, score_key
from .nearby_transfer_plan import (VERSION, CASE_IDS, SEARCH_RULE, source_inputs,
    proposal_menu, preparation_selection, candidate_selection, bank_summary, captured)

REPEATS=3
MAX_CANDIDATES=20
PREPARE_LIMIT_S=600
GPU_CAP_S=3600
SEED=2026112304


def publish(a,report):
    report['updated_at_unix']=time.time()
    atomic(a.output/'report.json',report)
    print(json.dumps({k:report[k] for k in ('status','phase','progress','error') if k in report}),flush=True)


def verify_preflight(a):
    if a.preflight_audit is None:raise ValueError('Independent preflight receipt required before native drops')
    audit=read(a.preflight_audit)
    if (audit.get('status')!='passed' or audit.get('version')!='sircus-nearby-transfer-preparation-audit-v1'
        or audit.get('native_execution_performed') is not False
        or type(audit.get('checks_count')) is not int or audit['checks_count']<=0
        or any(audit.get(key)!=sha(a.output/name) for key,name in
        (('contract_sha256','contract.json'),('preparation_sha256','preparation.json'),('slots_sha256','slots.json')))):
        raise ValueError('Independent preflight does not bind this frozen study')
    return audit


def _prepare_case(args,case,bank,source_contract,ref):
    import torch
    torch.set_num_threads(1)
    from .cell1050_rebound_study import preparation_time_limit,SearchPreparationExpired
    from .interactive_stalled_planner import copy_prepared_proofs
    from .intergrid_flight_ranker import FlightRanker
    from .arm_check_speed import install_vectorized_gate
    from .arm_path_speed import install_path_short_circuit
    a=argparse.Namespace(**args);original=case['original'];directory=a.output/('native-'+case['id'][-2:])
    begin=time.monotonic()
    copy_prepared_proofs(a.target_study/case['source_directory'],directory,original)
    engine=engine_for(a,original,directory,ref)
    install_vectorized_gate(engine.adapter.resolver.planner.gate)
    install_path_short_circuit(engine.adapter.resolver.planner)
    ranker=FlightRanker(ref['scene'],read(a.output/'calibration.json')['score_config'],a.assets,source_contract['flight_ranking']['rule'])
    proposals=proposal_menu(case,bank,engine.adapter.resolver.solver.low,engine.adapter.resolver.solver.high)
    wrist={p['id']:wrist_screen(ref['scene'],p,engine.adapter.resolver.models) for p in proposals}
    nominal={p['id']:nominal_prediction(ranker,ref['scene'],p) for p in proposals}
    ordered=preparation_selection(case,bank,proposals,nominal,wrist)
    admitted=[];records=[];predictions={};timed_out=False
    for offset in range(0,len(ordered),4):
        batch=ordered[offset:offset+4]
        try:
            with preparation_time_limit(PREPARE_LIMIT_S-(time.monotonic()-begin)):
                added,logs=engine.prepare(batch,one_per_setting=True)
                scored={r['id']:ranker.predict(r) for r in added}
            admitted.extend(added);records.extend(logs);predictions.update(scored)
        except SearchPreparationExpired:
            timed_out=True
            records.extend(dict(proposal_id=p['id'],proposal_sha256=digest(p),admitted=0,admitted_request_ids=[],
                reason='case_preparation_time_limit',timing={}) for p in batch)
            break
        atomic(a.output/f"preparation-{case['id'][-2:]}.json",dict(case,proposals=proposals,nominal_predictions=nominal,
            wrist_reach=wrist,preparation_order=[p['id'] for p in ordered],records=records,admitted=admitted,predictions=predictions,status='preparing'))
    rejected={r['proposal_id'] for r in records if not r['admitted'] and r['reason'] in
        ('finite_ik_unresolved','strict_endpoint_clearance_or_pose_rejected','finite_requested_setup_path_unresolved')}
    # Retry in the same source-balanced frozen order, not only one best rim score.
    retry_sources=[p for p in ordered if p['id'] in rejected][:SEARCH_RULE['multistart_retry_limit']]
    for source in retry_sources:
        if timed_out:break
        p=dict(copy.deepcopy(source),id=source['id']+'-multistart',retry_of=source['id'],ik_search='multistart16')
        proposals.append(p);nominal[p['id']]=nominal[source['id']];wrist[p['id']]=wrist[source['id']];ordered.append(p)
        try:
            with preparation_time_limit(PREPARE_LIMIT_S-(time.monotonic()-begin)):
                added,logs=engine.prepare([p],one_per_setting=False)
                scored={r['id']:ranker.predict(r) for r in added}
            admitted.extend(added);records.extend(logs);predictions.update(scored)
        except SearchPreparationExpired:
            timed_out=True;records.append(dict(proposal_id=p['id'],proposal_sha256=digest(p),admitted=0,
                admitted_request_ids=[],reason='case_preparation_time_limit',timing={}))
            break
    candidates=candidate_selection(admitted,predictions,MAX_CANDIDATES)
    counts={}
    for candidate in candidates:counts[candidate['source_bank_id']]=counts.get(candidate['source_bank_id'],0)+1
    result=dict(case,directory=directory.name,proposals=proposals,nominal_predictions=nominal,wrist_reach=wrist,
        preparation_order=[p['id'] for p in ordered],records=records,admitted=admitted,candidates=candidates,predictions=predictions,
        original_prediction=ranker.predict(original),preparation_s=time.monotonic()-begin,preparation_timed_out=timed_out,
        selection_order=[p['id'] for p in candidates],source_counts=counts,
        nearest_seed_distance_m=min(math.dist(original['bucket_xy_m'],s['bucket_xy_m']) for s in bank),
        bounds=dict(low=engine.adapter.resolver.solver.low.tolist(),high=engine.adapter.resolver.solver.high.tolist()),status='prepared')
    if engine.adapter.env is not None or torch.cuda.is_initialized():raise RuntimeError('CPU preparation allocated CUDA')
    atomic(a.output/f"preparation-{case['id'][-2:]}.json",result)
    return result


def prepare(a):
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('CPU preparation requires hidden GPUs')
    from concurrent.futures import ProcessPoolExecutor
    import multiprocessing
    from .intergrid_system_study import require_storage
    from .intergrid_route_labels import LABEL_VERSION
    cases,bank,source_contract,bindings=source_inputs(a)
    spec,parent_binding,source,ref=native_inputs(a)
    if spec['identity']!=source_contract['identity']:raise ValueError('Scene identity changed')
    a.output.mkdir(parents=True,exist_ok=False);require_storage(a.output)
    for name in ('training-prior.json','calibration.json','setup-home.json'):
        shutil.copyfile(a.target_study/name,a.output/name)
    shutil.copyfile(a.parent_study/'verification.json',a.output/'verification.json')
    atomic(a.output/'source-bank.json',bank)
    paths={name:str(getattr(a,name).resolve()) for name in ('target_study','target_audit','recovery_study','recovery_audit',
        'guarded_study','guarded_audit','parent_study','parent_audit')}
    report=dict(version=VERSION,status='running',phase='preparing',physics_substeps=64,audit_status='pending',scope='development',
        source_bank_summary=bank_summary(bank),progress=dict(completed=0,planned=0),gpu_cap_s=GPU_CAP_S)
    publish(a,report)
    plan=dict(version=VERSION,repeats=REPEATS,seed=SEED,search_rule=SEARCH_RULE,source=bindings,parent=parent_binding,
        source_paths=paths,identity=spec['identity'],scene=spec['scene'],label_version=LABEL_VERSION,
        source_bank_sha256=sha(a.output/'source-bank.json'),source_bank_summary=bank_summary(bank),cases=[])
    with ProcessPoolExecutor(max_workers=5,mp_context=multiprocessing.get_context('spawn')) as pool:
        futures=[pool.submit(_prepare_case,vars(a),case,bank,source_contract,ref) for case in cases]
        for future in futures:
            plan['cases'].append(future.result());atomic(a.output/'preparation.json',plan)
    report.update(phase='prepared',prepared_cases=len(plan['cases']),cases=summarize(plan,[]));publish(a,report)


def make_slot(case,source,variant,repeat):
    suffix=source['id'] if variant!='original' else case['id']
    request=dict(copy.deepcopy(source),id=f"{suffix}-{variant}-{repeat}",context_id=case['id'],
        candidate_id=source['id'],variant=variant,repeat=repeat,method=variant,
        phase='confirmation' if variant=='repeat' else 'diagnostic',source_request_id=source['id'])
    return dict(case_id=case['id'],variant=variant,repeat=repeat,candidate_id=source['id'],directory=case['directory'],request=request)


def schedule(plan):
    slots=[]
    for case in plan['cases']:
        slots.extend(make_slot(case,case['original'],'original',r) for r in range(REPEATS))
        slots.extend(make_slot(case,c,'search',0) for c in case['candidates'])
    return slots


def successful(row):
    return captured(row)


def repeat_schedule(plan,rows):
    winners={r['candidate_id'] for r in rows if r.get('variant')=='search' and successful(r)}
    return [make_slot(case,c,'repeat',i) for case in plan['cases'] for c in case['candidates'] if c['id'] in winners for i in range(1,REPEATS+1)]


def summarize(plan,rows):
    # Summary generation must also work in the lightweight, Torch-free CI
    # environment. These labels preserve the established route precedence.
    def outcome(row):
        if row['invalid_physics']:return 'invalid_physics'
        if row['route_success']:return row['shot_route'] or 'unclassified_capture'
        if row['experimental_route']['robot_assisted_capture']:return 'arm_assisted_capture'
        return 'flight_miss'
    result=[]
    for case in plan['cases']:
        groups={};foundcase=[r for r in rows if r.get('context_id')==case['id']]
        winners={r['candidate_id'] for r in foundcase if r.get('variant')=='search' and successful(r)}
        for variant,requested in (('original',3),('search',len(case['candidates'])),('repeat',3*len(winners))):
            found=[r for r in foundcase if r.get('variant')==variant];outcomes=[outcome(r) for r in found]
            groups[variant]=dict(requested=requested,completed=len(found),released=sum(r['native_release'] for r in found),setup_rejected=0,
                captures=sum(successful(r) for r in found),direct=outcomes.count('direct'),table=outcomes.count('table'),
                robot_contacts=sum(r['experimental_route']['robot_contact_steps']>0 for r in found),outcomes=outcomes,rows=found)
        confirmed=[]
        for candidate in case['candidates']:
            repeats=[r for r in foundcase if r.get('variant')=='repeat' and r.get('candidate_id')==candidate['id']]
            if candidate['id'] in winners:
                confirmed.append(dict(candidate_id=candidate['id'],request=candidate,repeat_captures=sum(successful(r) for r in repeats),repeat_completed=len(repeats)))
        ordered=sorted(confirmed,key=lambda r:(-r['repeat_captures'],score_key(r['request'],case['predictions'][r['candidate_id']])))
        best=ordered[0]['request'] if ordered else (case['candidates'][0] if case['candidates'] else None)
        result.append(dict(id=case['id'],bucket_xy_m=case['original']['bucket_xy_m'],results=groups,
            proposed=len(case['proposals']),checked=len(case['records']),admitted=len(case['admitted']),selected=len(case['candidates']),
            nearest_seed_distance_m=case.get('nearest_seed_distance_m'),source_counts=case.get('source_counts',{}),repeated_settings=confirmed,confirmed_settings=[r for r in confirmed if r['repeat_captures']==3 and r['repeat_completed']==3],best_candidate=best))
    return result


def freeze(a):
    from .corrected_grid import source_manifest
    plan=read(a.output/'preparation.json')
    if [c['id'] for c in plan['cases']]!=list(CASE_IDS) or (a.output/'contract.json').exists():raise ValueError('Fresh complete preparation required')
    atomic(a.output/'slots.json',schedule(plan))
    files={str(p.relative_to(a.output)):sha(p) for p in a.output.rglob('*') if p.is_file() and p.name!='report.json'}
    count=len(schedule(plan));searches=sum(len(c['candidates']) for c in plan['cases'])
    if any(len(c['candidates'])>MAX_CANDIDATES for c in plan['cases']) or count+searches*3+18>433:
        raise ValueError('Frozen selection exceeds the native release budget')
    spec=dict(version=VERSION,identity=plan['identity'],scene=plan['scene'],parent=plan['parent'],plan_sha256=sha(a.output/'preparation.json'),
        frozen_files=files,source_sha256=source_manifest(Path(__file__).resolve().parents[2]),physics_substeps=64,repeats=3,controls=18,
        initial_requests=count,maximum_confirmation_requests=searches*3,maximum_total_requests=count+searches*3+18,
        search_rule=SEARCH_RULE,gpu_cap_s=GPU_CAP_S,source_paths=plan['source_paths'],source_bindings=plan['source'],label_version=plan['label_version'],
        scope='Selected development geometry search; no policy fit, promotion, noise or driven-arm native validation')
    spec['fingerprint']=digest(spec);atomic(a.output/'contract.json',spec)
    report=read(a.output/'report.json');report.update(phase='frozen',progress=dict(completed=0,planned=count));publish(a,report)


def run(a):
    from .cell1050_rebound_study import annotate,control_requests
    from .coverage_training_study import append
    from .gpu_slot import foreign_processes,gpu_slot
    from .intergrid_route_labels import annotate_routes
    from .intergrid_system_study import gpu_deadline,require_storage
    from .parallel_capture_grid_study import control_mirror
    from .table_correction_native import MatchedEngine
    from .table_training_study import control_gate
    verify_frozen(a);verify_preflight(a);_,_,_,ref=native_inputs(a)
    plan=read(a.output/'preparation.json');slots=read(a.output/'slots.json')
    if (a.output/'native-started.json').exists():raise ValueError('Never repeat a started native study')
    report=read(a.output/'report.json')
    def waiting(event):report.update(phase='waiting_for_4090',gpu_wait=event);publish(a,report)
    with gpu_slot(a.gpu_lock,timeout_s=a.gpu_wait_s,poll_s=15,notify=waiting):
        if foreign_processes():raise RuntimeError('4090 still has a foreign workload')
        verify_frozen(a)
        verify_preflight(a)
        begin=time.monotonic()
        with (a.output/'native-started.json').open('x') as marker:
            json.dump(dict(slots_sha256=sha(a.output/'slots.json'),started_unix=time.time(),
                preflight_audit=str(a.preflight_audit.resolve()),preflight_audit_sha256=sha(a.preflight_audit)),marker)
        with gpu_deadline(GPU_CAP_S):
            engines={c['id']:engine_for(a,c['original'],a.output/c['directory'],ref) for c in plan['cases']}
            sources=read(a.parent_study/'control-replays.json');mirror=control_mirror(a.parent_study,a.output,sources,False)
            controls=MatchedEngine(ref,a.assets,a.output/'native-controls',parent=mirror,sources=sources,
                prior=read(a.parent_study/'old-prior.json'),calibration=read(a.output/'calibration.json'),home=read(a.output/'setup-home.json'))
            adapters=[e.adapter for e in engines.values()]+[controls.engine.adapter]
            factory,allocated=adapters[0].environment_factory,[]
            def shared():
                if not allocated:allocated.append(factory())
                return allocated[0]
            for adapter in adapters:adapter.environment_factory=shared
            rows=[];gates={};packet=0;repeat_slots=[]
            expected_options=read(a.parent_study/'native-yaw/effective-options.json');expected_geometry=read(a.parent_study/'native-yaw/native-compiled-geometry.json')
            def execute(engine,batch,phase,directory,control=False):
                nonlocal packet
                if foreign_processes():raise RuntimeError('Foreign workload appeared on leased4090')
                verify_frozen(a);require_storage(a.output)
                name=f'requests/batch-{packet:04d}.json';packet+=1
                atomic(a.output/name,dict(phase=phase,control=control,adapter_directory=directory,rows=batch))
                measured,timing=engine.evaluate(batch,phase)
                if [r['id'] for r in measured]!=[r['id'] for r in batch] or any(not r['native_release'] for r in measured):raise RuntimeError('Frozen request did not release')
                adapter=engine.engine.adapter if control else engine.adapter
                if adapter.env.effective_options!=expected_options or adapter.env.compiled_geometry!=expected_geometry:raise RuntimeError('Pinned native64 geometry/options changed')
                if not control:annotate(measured,a.output/directory);annotate_routes(measured,a.output/directory)
                for row,request in zip(measured,batch,strict=True):
                    row.update(adapter_directory=directory,request_packet=name,request_packet_sha256=sha(a.output/name))
                    if not control:row.update(**{k:request[k] for k in ('variant','repeat','context_id','candidate_id')});row['intended_route']=request['intended_route']
                rows.extend(measured);append(a.output/'rows.jsonl',measured);append(a.output/'timing.jsonl',[dict(packet=name,phase=phase,timing=timing)])
                report.update(progress=dict(completed=sum(r.get('variant') in ('original','search','repeat') for r in rows),planned=len(slots)+len(repeat_slots)),phase=phase,
                    gpu_lease_s=time.monotonic()-begin,controls=gates,cases=summarize(plan,rows));publish(a,report)
                return measured
            def execute_slots(selected,phase):
                for case in plan['cases']:
                    batch=[s['request'] for s in selected if s['case_id']==case['id']]
                    for start in range(0,len(batch),16):execute(engines[case['id']],batch[start:start+16],phase,case['directory'])
            gates['controls_pre']=control_gate(execute(controls,control_requests(sources,'controls_pre'),'controls_pre','native-controls',True))
            if gates['controls_pre']['status']!='passed':raise RuntimeError('Pre-controls failed')
            execute_slots(slots,'diagnostic')
            repeat_slots=repeat_schedule(plan,rows);atomic(a.output/'repeat-slots.json',repeat_slots)
            execute_slots(repeat_slots,'confirmation')
            gates['controls_post']=control_gate(execute(controls,control_requests(sources,'controls_post'),'controls_post','native-controls',True))
            if gates['controls_post']['status']!='passed':raise RuntimeError('Post-controls failed')
            report.update(status='completed',phase='completed',audit_status='pending',controls=gates,cases=summarize(plan,rows),gpu_lease_s=time.monotonic()-begin,
                native_releases=len(rows),initial_requests=len(slots),confirmation_requests=len(repeat_slots))
            manifest={str(p.relative_to(a.output)):sha(p) for p in a.output.rglob('*') if p.is_file() and p.name not in ('report.json','evidence-manifest.json')}
            atomic(a.output/'evidence-manifest.json',manifest);report['evidence_manifest_sha256']=sha(a.output/'evidence-manifest.json');publish(a,report)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('target-study','target-audit','recovery-study','recovery-audit','guarded-study','guarded-audit',
                'parent-study','parent-audit','reference','assets','output','gpu-lock'):
        p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--preflight-audit',type=Path)
    p.add_argument('--gpu-wait-s',type=int,default=1800)
    p.add_argument('--stage',choices=('prepare','freeze','run'),required=True)
    a=p.parse_args()
    try:globals()[a.stage](a)
    except TimeoutError:
        if (a.output/'report.json').exists() and not (a.output/'native-started.json').exists():
            report=read(a.output/'report.json');report.update(status='waiting',phase='prepared',error='Bounded GPU wait expired; frozen preparation can resume without any repeated releases')
            publish(a,report)
        raise


if __name__=='__main__':main()
