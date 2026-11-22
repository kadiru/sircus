"""Bounded geometry search for three diagnosed misses, followed by native64 repeats.

Privileged geometry and endpoint-held ball dynamics; no policy training or
interactive planner promotion. CPU geometry proposals freeze before any drops.
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
from .hybrid_miss_study import CASES, source_inputs, native_inputs, engine_for, verify_frozen

VERSION = 'sircus-board-pose-recovery-v1'
REPEATS = 3
MAX_CANDIDATES = 40
PREPARE_LIMIT_S = 900
GPU_CAP_S = 3600
SEED = 2026112207
SEARCH_RULE = dict(
    radial_offsets_m=[-.15, -.10, -.05, 0., .05, .10, .15],
    tangent_offsets_m=[-.15, -.10, -.05, 0., .05, .10, .15],
    heading_offsets_deg=[-30., -16., -8., 0., 8., 16., 30.],
    twist_deg=[0., 180.], height_delta_m=[-.12, .08],
    tilt_deg=[12., 44.], gap_m=[.08, .55], tuned_proposals=4096,
    analytic_candidates_per_group=96, maximum_native_candidates_per_case=MAX_CANDIDATES,
    necessary_wrist_screen_before_shortlist=True, multistart_retry_limit=24,
    multistart_starts_per_arm=16, multistart_iterations=180,
    cpu_case_limit_s=PREPARE_LIMIT_S, repeats_per_success=REPEATS,
    selection='Best precise min(arm,rim) margin per 50mm XY/40mm Z/8deg tilt/80mm gap/16deg heading/twist cell; remaining by score and stable ID',
    repeat_rule='Every valid direct/table capture in initial search gets three fresh identical repeats',
)


def geometry_key(row):
    return tuple(round(float(x), 10) for x in (*row['values'], row['board_heading_offset_deg'], row['board_twist_deg']))


def proposal_menu(case, low, high):
    """Fixed Cartesian offsets, then a deterministic space-filling tuning menu."""
    import numpy as np
    from scipy.stats import qmc
    original = case['original']; values = np.asarray(original['values'], float)
    bucket = np.asarray(original['bucket_xy_m']); delta = bucket-values[:2]
    direction = delta/np.linalg.norm(delta); lateral = np.array([-direction[1], direction[0]])
    proposals=[]; seen={geometry_key(original)}
    def add(v, heading, twist, group):
        if np.any(v < low-1e-9) or np.any(v > high+1e-9) or np.linalg.norm(bucket-v[:2]) <= .15:
            return
        row=dict(id=f"{case['id']}-pose-{len(proposals):05d}",context_id=case['id'],
            bucket_xy_m=bucket.tolist(),values=v.tolist(),board_heading_offset_deg=float(heading),
            board_twist_deg=float(twist),phase='search',role='diagnostic',method='geometry_search',
            intended_route=original['intended_route'],group=group,
            seed_q10=original['selected_q10'],seed_q14=original['selected_q14'],source_request_id=original['id'])
        key=geometry_key(row)
        if key not in seen: seen.add(key); proposals.append(row)
    for radial in SEARCH_RULE['radial_offsets_m']:
        for tangent in SEARCH_RULE['tangent_offsets_m']:
            for heading in SEARCH_RULE['heading_offsets_deg']:
                for twist in SEARCH_RULE['twist_deg']:
                    v=values.copy(); v[:2]+=radial*direction+tangent*lateral
                    add(v,heading,twist,'position_heading')
    samples=qmc.Halton(7,scramble=True,seed=SEED+int(case['id'][-2:])).random(SEARCH_RULE['tuned_proposals'])
    for u in samples:
        v=values.copy();v[:2]+=(.30*u[0]-.15)*direction+(.30*u[1]-.15)*lateral
        v[2]+= -.12+.20*u[2];v[3]=12+32*u[3];v[4]=.08+.47*u[4]
        add(v,60*u[5]-30,180. if u[6]>=.5 else 0.,'position_height_tilt_gap')
    return proposals


def nominal_prediction(ranker, scene, request):
    """Fast rim geometry screen only. Arm clearance is checked after actual IK."""
    import numpy as np
    from .cell1050_heading_native import heading_targets
    from .intergrid_flight_ranker import board_launch, rebound
    from .consistent_flight_selector import member_flight, _descending_time
    targets=heading_targets(scene,request['bucket_xy_m'],request['values'],request['board_heading_offset_deg'],request['board_twist_deg'])
    release=np.asarray(request['values'][:3],float);release[2]+=request['values'][4]+ranker.radius
    be,bt,te,tt=ranker.coefficients
    launch=board_launch(targets['board_world'],ranker.board_size,release,ranker.radius,ranker.gravity[2],be,bt)
    result=dict(direct=None,table=None,predicted_route=None,score_m=None)
    if launch is None:return result
    p,v=launch; bucket=request['bucket_xy_m']
    direct=member_flight(p,v,bucket,ranker.geom)['conservative_rim_band_clearance_m']
    if direct is not None:result['direct']=float(direct)
    t=_descending_time(p[2],v[2],ranker.table_center[2]+ranker.table_half[2]+ranker.radius,ranker.gravity[2])
    if t is not None and t<=2.:
        impact=p+v*t+.5*ranker.gravity*t*t
        edge=float(min(ranker.table_half[:2]-np.abs(impact[:2]-ranker.table_center[:2])))
        if edge>=0:
            velocity=rebound(v+ranker.gravity*t,np.array([0.,0.,1.]),te,tt)
            table=member_flight(impact,velocity,bucket,ranker.geom)['conservative_rim_band_clearance_m']
            if table is not None:result['table']=float(min(table,edge))
    routes=[r for r in ('direct','table') if result[r] is not None]
    if routes:
        route=max(routes,key=lambda r:result[r]);result.update(predicted_route=route,score_m=result[route])
    return result


def wrist_screen(scene, request, models):
    from .cell1050_heading_native import heading_targets
    from .fixed35_broad_native import wrist_bound
    targets=heading_targets(scene,request['bucket_xy_m'],request['values'],request['board_heading_offset_deg'],request['board_twist_deg'])
    arms=dict(dropper=wrist_bound(models['dropper'],targets['dropper_flange_world']),
              bouncer=wrist_bound(models['bouncer'],targets['board_flange_local']))
    return dict(passed=all(v['passed'] for v in arms.values()),arms=arms,
                scope='Necessary outer wrist reach only; not IK or collision/path admission')


def score_key(row, prediction):
    score=prediction.get('score_m')
    return (score is None or not math.isfinite(score), -score if score is not None and math.isfinite(score) else 0., row['id'])


def diversity_key(row):
    v=row['values']
    return tuple(int(round(x/s)) for x,s in zip(v,(.05,.05,.04,8.,.08)))+(int(round(row['board_heading_offset_deg']/16)),int(round(row['board_twist_deg']/180)))


def diversified(rows, predictions, limit):
    ordered=sorted(rows,key=lambda r:score_key(r,predictions[r['id']]))
    first=[];rest=[];seen=set();geometries=set()
    for row in ordered:
        geom=geometry_key(row)
        if geom in geometries:continue
        geometries.add(geom)
        key=diversity_key(row)
        if key not in seen:seen.add(key);first.append(row)
        else:rest.append(row)
    return (first+rest)[:limit]


def publish(a,report):
    report['updated_at_unix']=time.time();atomic(a.output/'report.json',report)
    print(json.dumps({k:report[k] for k in ('status','phase','progress','error') if k in report}),flush=True)


def _prepare_case(args,case,source_contract,ref):
    """One process per case; all native/GPU initialization is forbidden here."""
    import torch
    torch.set_num_threads(1)
    from .cell1050_rebound_study import preparation_time_limit,SearchPreparationExpired
    from .interactive_stalled_planner import copy_prepared_proofs
    from .intergrid_flight_ranker import FlightRanker
    a=argparse.Namespace(**args);original=case['original'];directory=a.output/('native-'+case['id'][-2:])
    copy_prepared_proofs(a.source_study/'native-hybrid',directory,original)
    engine=engine_for(a,original,directory,ref)
    from .arm_check_speed import install_vectorized_gate
    from .arm_path_speed import install_path_short_circuit
    install_vectorized_gate(engine.adapter.resolver.planner.gate)
    install_path_short_circuit(engine.adapter.resolver.planner)
    ranker=FlightRanker(ref['scene'],read(a.output/'calibration.json')['score_config'],a.assets,source_contract['flight_ranking']['rule'])
    begin=time.monotonic();proposals=proposal_menu(case,engine.adapter.resolver.solver.low,engine.adapter.resolver.solver.high)
    wrist_reach={p['id']:wrist_screen(ref['scene'],p,engine.adapter.resolver.models) for p in proposals}
    nominal={p['id']:nominal_prediction(ranker,ref['scene'],p) for p in proposals}
    shortlist=[]
    for group in ('position_heading','position_height_tilt_gap'):
        shortlist.extend(diversified([p for p in proposals if p['group']==group and wrist_reach[p['id']]['passed']],nominal,SEARCH_RULE['analytic_candidates_per_group']))
    # Interleave preserving each group's analytic ranking, so a timeout cannot
    # spend all preparation exclusively on the unchanged tilt/gap family.
    groups=[[p for p in shortlist if p['group']==g] for g in ('position_heading','position_height_tilt_gap')]
    ordered=[g[i] for i in range(max(map(len,groups))) for g in groups if i<len(g)]
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
            records.extend(dict(proposal_id=p['id'],proposal_sha256=digest(p),admitted=0,
                admitted_request_ids=[],reason='case_preparation_time_limit',timing={}) for p in batch)
            break
        # Save an evidence checkpoint after every complete batch.
        atomic(a.output/f"preparation-{case['id'][-2:]}.json",dict(case,proposals=proposals,
            nominal_predictions=nominal,wrist_reach=wrist_reach,preparation_order=[p['id'] for p in ordered],
            records=records,admitted=admitted,predictions=predictions,status='preparing'))
    # Warm-start IK is fast but not exhaustive. Give a fixed small set of failed
    # geometries the established multistart branch budget before declaring it
    # unresolved. These decisions use only CPU preparation, never drop results.
    rejected={r['proposal_id'] for r in records if not r['admitted'] and r['reason'] in
              ('finite_ik_unresolved','strict_endpoint_clearance_or_pose_rejected','finite_requested_setup_path_unresolved')}
    retry_sources=sorted([p for p in ordered if p['id'] in rejected],key=lambda p:score_key(p,nominal[p['id']]))[:SEARCH_RULE['multistart_retry_limit']]
    for source in retry_sources:
        if timed_out:break
        p=dict(copy.deepcopy(source),id=source['id']+'-multistart',retry_of=source['id'],ik_search='multistart16')
        proposals.append(p);nominal[p['id']]=nominal[source['id']];wrist_reach[p['id']]=wrist_reach[source['id']];ordered.append(p)
        try:
            with preparation_time_limit(PREPARE_LIMIT_S-(time.monotonic()-begin)):
                added,logs=engine.prepare([p],one_per_setting=False)
                scored={r['id']:ranker.predict(r) for r in added}
            admitted.extend(added);records.extend(logs);predictions.update(scored)
        except SearchPreparationExpired:
            timed_out=True;records.append(dict(proposal_id=p['id'],proposal_sha256=digest(p),admitted=0,
                admitted_request_ids=[],reason='case_preparation_time_limit',timing={}))
            break
    candidates=diversified(admitted,predictions,MAX_CANDIDATES)
    result=dict(case,directory=directory.name,proposals=proposals,nominal_predictions=nominal,wrist_reach=wrist_reach,
        preparation_order=[p['id'] for p in ordered],records=records,admitted=admitted,
        candidates=candidates,predictions=predictions,original_prediction=ranker.predict(original),
        preparation_s=time.monotonic()-begin,preparation_timed_out=timed_out,
        selection_order=[r['id'] for r in candidates],bounds=dict(low=engine.adapter.resolver.solver.low.tolist(),high=engine.adapter.resolver.solver.high.tolist()),status='prepared')
    if engine.adapter.env is not None or torch.cuda.is_initialized():raise RuntimeError('CPU preparation allocated CUDA')
    atomic(a.output/f"preparation-{case['id'][-2:]}.json",result)
    return result


def prepare(a):
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('CPU preparation requires hidden GPUs')
    from concurrent.futures import ProcessPoolExecutor
    import multiprocessing
    from .intergrid_system_study import require_storage
    selected,source_contract,binding=source_inputs(a)
    spec,parent_binding,source,ref=native_inputs(a)
    if spec['identity']!=source_contract['identity']:raise ValueError('Scene identity changed')
    a.output.mkdir(parents=True,exist_ok=False);require_storage(a.output)
    for name in ('training-prior.json','calibration.json','setup-home.json'):
        shutil.copyfile(a.source_study/name,a.output/name)
    shutil.copyfile(a.parent_study/'verification.json',a.output/'verification.json')
    report=dict(version=VERSION,status='running',phase='preparing',physics_substeps=64,audit_status='pending',progress=dict(completed=0,planned=0))
    publish(a,report)
    plan=dict(version=VERSION,repeats=REPEATS,seed=SEED,search_rule=SEARCH_RULE,source=binding,parent=parent_binding,
        identity=spec['identity'],scene=spec['scene'],changed_parameters=['board center','board heading/twist','height','tilt','drop gap'],cases=[])
    with ProcessPoolExecutor(max_workers=3,mp_context=multiprocessing.get_context('spawn')) as pool:
        futures=[pool.submit(_prepare_case,vars(a),case,source_contract,ref) for case in selected]
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
    return bool(row.get('native_release') and not row.get('invalid_physics') and row.get('route_success'))


def repeat_schedule(plan,rows):
    winners={r['candidate_id'] for r in rows if r.get('variant')=='search' and successful(r)}
    return [make_slot(case,c,'repeat',i) for case in plan['cases'] for c in case['candidates'] if c['id'] in winners for i in range(1,REPEATS+1)]


def summarize(plan,rows):
    from .intergrid_route_selector_plan import outcome,STANDARD_CAPTURES
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
            repeated_settings=confirmed,confirmed_settings=[r for r in confirmed if r['repeat_captures']==3 and r['repeat_completed']==3],best_candidate=best))
    return result


def freeze(a):
    from .corrected_grid import source_manifest
    plan=read(a.output/'preparation.json')
    if [c['id'] for c in plan['cases']]!=list(CASES) or (a.output/'contract.json').exists():raise ValueError('Fresh complete preparation required')
    atomic(a.output/'slots.json',schedule(plan))
    files={str(p.relative_to(a.output)):sha(p) for p in a.output.rglob('*') if p.is_file() and p.name!='report.json'}
    count=len(schedule(plan));searches=sum(len(c['candidates']) for c in plan['cases'])
    spec=dict(version=VERSION,identity=plan['identity'],scene=plan['scene'],parent=plan['parent'],plan_sha256=sha(a.output/'preparation.json'),
        frozen_files=files,source_sha256=source_manifest(Path(__file__).resolve().parents[2]),physics_substeps=64,repeats=3,controls=18,
        initial_requests=count,maximum_confirmation_requests=searches*3,maximum_total_requests=count+searches*3+18,
        search_rule=SEARCH_RULE,gpu_cap_s=GPU_CAP_S,
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
    verify_frozen(a);_,_,_,ref=native_inputs(a)
    plan=read(a.output/'preparation.json');slots=read(a.output/'slots.json')
    if (a.output/'native-started.json').exists():raise ValueError('Never repeat a started native study')
    report=read(a.output/'report.json')
    def waiting(event):report.update(phase='waiting_for_4090',gpu_wait=event);publish(a,report)
    with gpu_slot(a.gpu_lock,timeout_s=a.gpu_wait_s,poll_s=15,notify=waiting):
        begin=time.monotonic();atomic(a.output/'native-started.json',dict(slots_sha256=sha(a.output/'slots.json'),started_unix=time.time()))
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
                    if not control:row.update(**{k:request[k] for k in ('variant','repeat','context_id','candidate_id')})
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
    for key in ('source-study','source-audit','parent-study','parent-audit','reference','assets','output','gpu-lock'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--gpu-wait-s',type=int,default=1800)
    p.add_argument('--stage',choices=('prepare','freeze','run'),required=True)
    a=p.parse_args();globals()[a.stage](a)


if __name__=='__main__':main()
