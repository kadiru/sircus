"""Finite63-slot height-restoration diagnosis; no training or deployment."""
from __future__ import annotations
import argparse
import copy
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import signal
import time
from . import recovery_regression_native as N
from .benchmark_verify import contained, read, sha, digest
from .corrected_grid import source_manifest
from .gpu_slot import GPU_UUID, gpu_slot, foreign_processes
from .policy64_contract import load_runtime_reference
from .sweep_worker import atomic
from .table_correction_study import attach_robot_counts, control_gate
from .table_training_contract import statistics, captured, reward

VERSION=N.VERSION
PARENT_REPORT='6a255223820656059fe16a5de2944ba2593f7ac3603b287f4955979cd4677504'
PARENT_AUDIT='e2ba653792b37d302425ba4af6cb22460c3d5b426274b3c1b3cbce270a857a9b'
PARENT_FP='f54c341eba5258b427e4e1eb05a1b2e93bf76518d9f94a2fe8cf6bad744f6082'
INPUT_KEYS=('parent-study','parent-audit','diagnosis','rule','reference','assets')


def append(path,rows):
    with Path(path).open('a') as stream:
        for row in rows:stream.write(json.dumps(row,allow_nan=False)+'\n')


def share_native_environment(primary,controls):
    """Reuse the established shared-factory design without importing trainers."""
    adapters=(primary.adapter,controls.engine.adapter)
    if any(adapter.env is not None for adapter in adapters):raise ValueError('Share factory before allocating either environment')
    factory=primary.adapter.environment_factory
    if not callable(factory):raise ValueError('Pinned native factory required')
    allocated=[]
    def shared():
        if not allocated:allocated.append(factory())
        return allocated[0]
    for adapter in adapters:adapter.environment_factory=shared


def protocol():
    return dict(substeps=64,maximum_requests=63,repeats=3,diagnostic_requests=45,control_requests=18,
        maximum_acquired_s=3600,maximum_queue_s=900,maximum_pipeline_s=6300,audit_reserve_s=1800,
        storage_reserve_bytes=8*1024**3,training=False,noise=False,posthoc_development=True,
        variants=list(N.VARIANTS),contexts=list(N.CASES),changed_parameter='board_height_m',parameter_index=2,
        mode='restore_matched_parent_coordinate',
        old_refusals='Cached authenticated parent refusals count as requests; no fresh IK or placeholder-joint release.',
        correction='One strict changed-action proof per case, reused unchanged across three nominal requests.',
        derived_release='Fixed drop gap means release-center height changes by the same board-height delta; absolute release Z is not held fixed.',
        selection='All63 slots frozen before outcomes; no search, retries, winner selection or policy promotion.',
        controls='Original17 direct/table/miss before and after; parent captures are diagnostic outcomes, not a hard gate.')


def authenticate_parent(parent,audit):
    parent,audit=Path(parent),Path(audit)
    if sha(parent/'report.json')!=PARENT_REPORT or sha(audit)!=PARENT_AUDIT:raise ValueError('Exact audited recovery parent required')
    report,spec,proof,manifest=(read(p) for p in (parent/'report.json',parent/'contract.json',audit,parent/'evidence-manifest.json'))
    unsigned=dict(spec);fp=unsigned.pop('fingerprint')
    if (digest(unsigned)!=fp or fp!=PARENT_FP or report['contract_fingerprint']!=fp or proof['contract_fingerprint']!=fp
            or report['status']!='completed' or report['phase']!='completed' or report['evidence_sha256']!=manifest
            or proof['status']!='passed' or proof['full_checks_verified'] is not True or proof['native_checks_verified'] is not True
            or proof['failures'] or proof['checks_count']!=283251 or len(proof['checks'])!=283251
            or not all(c.get('passed') is True for c in proof['checks']) or proof['report_sha256']!=PARENT_REPORT
            or proof['evidence_manifest_sha256']!=sha(parent/'evidence-manifest.json')):
        raise ValueError('Complete bound recovery proof required')
    files={}
    def verify(name):
        path=contained(parent,name)
        if sha(path)!=manifest[name]:raise ValueError('Parent artifact changed: '+name)
        files[name]=manifest[name];return path
    rows=[json.loads(x) for x in verify('rows.jsonl').read_text().splitlines()]
    if len(rows)!=970 or len({r['id'] for r in rows})!=970:raise ValueError('Complete970-row parent ledger required')
    pairs=N.source_pairs(rows);sources=[pair[m] for pair in pairs for m in ('parent_bc','recovery_bc')]
    controls=read(verify('control-replays.json'))
    config={name:read(verify(name)) for name in ('contract.json','prior.json','old-prior.json','calibration.json','verification.json','setup-home.json')}
    if len(config['prior.json']['records'])!=19 or len(config['old-prior.json']['records'])!=17:raise ValueError('Separate19/17 geometry priors required')
    for namespace,members in (('native',sources),('control-parent/native',controls)):
        for name in ('effective-options.json','native-compiled-geometry.json'):verify(namespace+'/'+name)
        for row in members:
            for key,hkey in (('context_resolution_file','context_resolution_sha256'),('connection_file','connection_sha256')):
                name=namespace+'/'+row[key];verify(name)
                if files[name]!=row[hkey]:raise ValueError('Parent row proof differs: '+name)
    return dict(spec=spec,rows=rows,sources=sources,controls=controls,config=config,
        bindings=dict(report_sha256=PARENT_REPORT,audit_sha256=PARENT_AUDIT,contract_fingerprint=fp,
            manifest_sha256=sha(parent/'evidence-manifest.json'),files_sha256=files))


def authenticate_diagnosis(path,rule):
    if sha(path)!=rule['diagnosis_sha256']:raise ValueError('Reviewed diagnosis bytes differ')
    diagnosis=read(path);unsigned=dict(diagnosis);fp=unsigned.pop('diagnosis_fingerprint')
    if digest(unsigned)!=fp or diagnosis['version']!='sircus-coverage-regression-diagnosis-v1':raise ValueError('Diagnosis fingerprint/version differs')
    source=diagnosis['source']
    for key,value in dict(report_sha256=PARENT_REPORT,audit_sha256=PARENT_AUDIT,contract_fingerprint=PARENT_FP,
                          full_checks_verified=True,native_checks_verified=True,checks_count=283251).items():
        if source.get(key)!=value:raise ValueError('Diagnosis parent binding differs: '+key)
    if [x['context_id'] for x in diagnosis['losses']]!=list(N.CASES):raise ValueError('Diagnosis five-case order differs')
    return diagnosis


def summarize(rows):
    result=statistics(rows)
    result.update(invalid_physics=sum(bool(r.get('invalid_physics')) for r in rows),
        robot_contact_shots=sum(bool(r.get('diagnostic_robot_contact_steps')) for r in rows),
        cached_refusal_requests=sum(r.get('admission_mode')=='cached_parent_refusal' for r in rows))
    return result


def case_results(plan,rows):
    cases=[]
    for context in N.CASES:
        requests=[r for r in plan if r['context_id']==context];group=[r for r in rows if r['context_id']==context]
        def values(variant):return next(r['values'] for r in requests if r['variant']==variant)
        outcomes=[]
        for row in group:
            outcome={k:copy.deepcopy(row.get(k)) for k in ('id','variant','repeat','values','native_release','invalid',
                'invalid_physics','connection_sha256','native_cache_file','native_cache_row','native_cache_sha256','admission_mode')}
            outcome.update(capture=captured(row),table_contact_steps=row.get('table_contact_steps',0),
                robot_contact_steps=row.get('diagnostic_robot_contact_steps',0));outcomes.append(outcome)
        cases.append(dict(context_id=context,bucket_xy_m=requests[0]['bucket_xy_m'],parent_values=values('parent_exact'),
            recovery_values=values('recovery_exact'),corrected_values=values('corrected'),
            variants={v:summarize([r for r in group if r['variant']==v]) for v in N.VARIANTS},outcomes=outcomes))
    return cases


def run(a):
    root=Path(__file__).resolve().parents[2];started=getattr(a,'acquired_monotonic',time.monotonic())
    elapsed=lambda:time.monotonic()-started
    loaded=authenticate_parent(a.parent_study,a.parent_audit);parent=loaded['spec']
    rule=N.validate_rule(read(a.rule));diagnosis=authenticate_diagnosis(a.diagnosis,rule)
    if diagnosis['source']['rows_sha256']!=loaded['bindings']['files_sha256']['rows.jsonl']:raise ValueError('Diagnosis row ledger differs')
    plan=N.make_plan(loaded['rows'],loaded['controls'],rule)
    ref=load_runtime_reference(a.reference,a.assets,root)
    if digest(ref['scene'])!=parent['identity']['scene_sha256'] or ref['asset_hashes']!=parent['identity']['asset_sha256']:raise ValueError('Scene/assets differ')
    if {n:importlib.metadata.version(n) for n in parent['packages']}!=parent['packages']:raise ValueError('Pinned package versions differ')
    output=a.output.resolve()
    if output.exists() or any(output.is_relative_to(Path(x).resolve()) for x in (a.parent_study,root,a.assets)):
        raise ValueError('Fresh separate output required')
    output.mkdir(parents=True);(output/'requests').mkdir()
    for name in ('prior.json','old-prior.json','calibration.json','verification.json','setup-home.json'):
        shutil.copyfile(a.parent_study/name,output/name)
    shutil.copyfile(a.diagnosis,output/'diagnosis.json')
    atomic(output/'rule.json',rule);atomic(output/'plan.json',plan)
    atomic(output/'parent-rows.json',loaded['sources']);atomic(output/'control-replays.json',loaded['controls'])
    spec=dict(version=VERSION,protocol=protocol(),rule=rule,diagnosis_sha256=sha(output/'diagnosis.json'),
        plan_sha256=sha(output/'plan.json'),parent_rows_sha256=sha(output/'parent-rows.json'),
        control_replays_sha256=sha(output/'control-replays.json'),rule_sha256=sha(output/'rule.json'),
        source_sha256=source_manifest(root),parent_bindings=loaded['bindings'],scene=parent['scene'],
        identity=parent['identity'],packages=parent['packages'],runtime_reference=ref['runtime_reference'],created_unix_s=time.time())
    spec['fingerprint']=digest(spec);atomic(output/'contract.json',spec)
    rows=[];number=0
    report=dict(version=VERSION,status='running',phase='frozen_plan',contract_fingerprint=spec['fingerprint'],rule=rule,
        protocol=protocol(),expected_requests=63,control_gates={},training=False,posthoc_development=True,
        deadlines=dict(worker_limit_s=a.wall_time_limit,effective_worker_limit_s=a.effective_worker_limit,
            global_deadline_unix_s=a.global_deadline_unix),gate=dict(status='pending'))
    def publish(phase):
        report.update(phase=phase,elapsed_s=elapsed(),counts=summarize(rows),
            counts_by_variant={v:summarize([r for r in rows if r['variant']==v]) for v in N.VARIANTS},cases=case_results(plan,rows))
        atomic(output/'report.json',report);print(json.dumps(dict(status=report['status'],phase=phase,counts=report['counts'],elapsed_s=elapsed())),flush=True)
    def boundary(phase):
        append(output/'phase-boundaries.jsonl',[dict(phase=phase,acquired_elapsed_s=elapsed(),rows_count=len(rows),
            rows_sha256=sha(output/'rows.jsonl') if rows else None,plan_sha256=sha(output/'plan.json'))])
    publish('frozen_plan');boundary('plan-frozen')
    try:
        engine=N.MatchedEngine(ref,a.assets,output/'native',parent=a.parent_study,sources=loaded['sources'],
            prior=read(output/'prior.json'),calibration=read(output/'calibration.json'),home=read(output/'setup-home.json'),rule=rule)
        controls=N.MatchedEngine(ref,a.assets,output/'native-controls',parent=a.parent_study/'control-parent',sources=loaded['controls'],
            prior=read(output/'old-prior.json'),calibration=read(output/'calibration.json'),home=read(output/'setup-home.json'),rule=rule)
        share_native_environment(engine.engine,controls)
        for phase in ('controls_pre','diagnostic','controls_post'):
            requests=[r for r in plan if r['phase']==phase];observed_phase=[]
            for offset in range(0,len(requests),32):
                chunk=requests[offset:offset+32]
                if len(rows)+len(chunk)>63:raise RuntimeError('Frozen63-request cap exceeded')
                if elapsed()>a.effective_worker_limit-30:raise TimeoutError('Own finite acquired budget reached')
                if shutil.disk_usage(output).free<8*1024**3:raise RuntimeError('8GiB reserve reached; evidence retained')
                if foreign_processes():raise RuntimeError('Foreign4090 workload appeared; untouched')
                packet=f'requests/batch-{number:05d}.json';number+=1
                atomic(output/packet,dict(phase=phase,rows=chunk,pre_outcome=True));begin=elapsed()
                wrapper=controls if phase.startswith('controls_') else engine
                observed,timing=wrapper.evaluate(chunk,phase)
                namespace='native-controls' if phase.startswith('controls_') else 'native'
                attach_robot_counts(observed,output/namespace)
                for row in observed:row.update(adapter_directory=namespace,learning_reward=reward(row),request_packet=packet,request_packet_sha256=sha(output/packet))
                append(output/'rows.jsonl',observed);rows.extend(observed);observed_phase.extend(observed)
                append(output/'timing.jsonl',[dict(phase=phase,**timing,request_packet=packet,request_packet_sha256=sha(output/packet),acquired_start_s=begin,acquired_end_s=elapsed())]);publish(phase)
            if phase.startswith('controls_'):
                gate=control_gate(observed_phase);report['control_gates']['pre' if phase=='controls_pre' else 'post']=gate
                if gate['status']!='passed':raise RuntimeError('Original matched direct/table/miss control gate failed')
            boundary(phase+'-complete')
        if len(rows)!=63:raise RuntimeError('Complete63-slot denominator required')
        report.update(status='completed',gate=dict(status='passed',substeps=64,controls_passed=True,complete_plan=True,no_policy_promotion=True))
        publish('completed')
        manifest={str(p.relative_to(output)):sha(p) for p in output.rglob('*') if p.is_file() and p.name not in ('report.json','evidence-manifest.json')}
        atomic(output/'evidence-manifest.json',manifest);report['evidence_sha256']=manifest;atomic(output/'report.json',report)
        return report
    except Exception as error:
        report.update(status='failed',error=type(error).__name__+': '+str(error));publish('failed');raise


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in (*INPUT_KEYS,'output','gpu-lock'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--wall-time-limit',type=float,default=3600);p.add_argument('--queue-timeout',type=float,default=900)
    p.add_argument('--global-deadline-unix',type=float,required=True);a=p.parse_args()
    if not 60<=a.wall_time_limit<=3600 or not 0<a.queue_timeout<=900:raise ValueError('Finite approved limits required')
    if os.environ.get('CUDA_VISIBLE_DEVICES')!=GPU_UUID:raise RuntimeError('Only authorized4090 UUID permitted')
    remaining=a.global_deadline_unix-time.time()-1800
    if remaining<=60:raise TimeoutError('No bounded worker window remains')
    def deadline(signum,frame):raise TimeoutError('Own acquired-slot deadline')
    with gpu_slot(a.gpu_lock,timeout_s=min(a.queue_timeout,remaining-60),notify=lambda r:print(json.dumps(r),flush=True)):
        a.acquired_monotonic=time.monotonic();a.effective_worker_limit=min(a.wall_time_limit,a.global_deadline_unix-time.time()-1800)
        if a.effective_worker_limit<=60:raise TimeoutError('CPU audit reserve leaves no worker time')
        budget=a.output.with_name(a.output.name+'-budget.json')
        if budget.exists():raise ValueError('Fresh budget receipt required')
        atomic(budget,dict(acquired_unix_s=time.time(),limit_s=a.wall_time_limit,effective_worker_limit_s=a.effective_worker_limit,
            global_deadline_unix_s=a.global_deadline_unix,audit_reserve_s=1800,gpu_uuid=GPU_UUID))
        signal.signal(signal.SIGALRM,deadline);signal.setitimer(signal.ITIMER_REAL,a.effective_worker_limit-10)
        try:run(a)
        finally:signal.setitimer(signal.ITIMER_REAL,0)

if __name__=='__main__':main()
