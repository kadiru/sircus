"""Bounded49-shot fixed-yaw sweep and prespecified fresh confirmation repeats."""
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
from . import yaw_tilt_gap_contract as C
from .yaw_tilt_gap_native import SweepEngine
from .benchmark_verify import read,sha,digest
from .corrected_grid import source_manifest
from .gpu_slot import GPU_UUID,gpu_slot,foreign_processes
from .policy64_contract import load_runtime_reference
from .sweep_worker import atomic
from .table_correction_study import attach_robot_counts,control_gate
from .table_training_contract import reward
from .gripper_yaw_study import summarize
from .recovery_regression_study import append


def case_results(plan,rows,selection):
    source=next(r for r in plan if r['variant']=='far_rotated');observed={r['id']:r for r in rows}
    cells=[]
    selected={r['candidate_id']:r for r in selection}
    for r in plan:
        if r['phase']!='discovery':continue
        row=observed.get(r['id']);admission=None;flight=None
        if row:
            admission=dict(passed=bool(row['native_release']),reason=row.get('invalid'),
                setup_proof=row.get('setup_proof'),connection_file=row['connection_file'],connection_sha256=row['connection_sha256'])
            flight=dict(capture=C.captured(row),native_release=row['native_release'],invalid_physics=row.get('invalid_physics'),
                robot_contact_steps=row.get('diagnostic_robot_contact_steps',0),whole_ball_entry=row.get('whole_ball_entry'),
                max_retention_s=row.get('max_retention_s'),table_contact_steps=row.get('table_contact_steps',0),
                route_quality=row.get('route_quality'),record_sha256=digest(row))
        cells.append(dict(candidate_id=r['id'],grid_index=r['grid_index'],values=r['values'],bucket_xy_m=r['bucket_xy_m'],
            admission=admission,discovery=flight,selected=r['id'] in selected,
            repeats=summarize([x for x in rows if x.get('selected_from_id')==r['id']])))
    return [dict(context_id=C.FAR,bucket_xy_m=source['bucket_xy_m'],baseline_values=source['values'],
        target_yaw_rad=source['requested_gripper_yaw_rad'],grid=cells,selection=selection,
        candidate_repeats=[dict(**s,summary=summarize([r for r in rows if r.get('selected_from_id')==s['candidate_id']])) for s in selection],
        variants={v:summarize([r for r in rows if r['variant']==v]) for v in C.VARIANTS})]


def run(a):
    root=Path(__file__).resolve().parents[2];started=getattr(a,'acquired_monotonic',time.monotonic());elapsed=lambda:time.monotonic()-started
    loaded=C.authenticate_inputs(a.parent_study,a.parent_audit,a.yaw_study,a.yaw_audit);parent=loaded['spec']
    rule=C.validate_rule(read(a.rule));plan=C.make_plan(loaded);ref=load_runtime_reference(a.reference,a.assets,root)
    if digest(ref['scene'])!=parent['identity']['scene_sha256'] or ref['asset_hashes']!=parent['identity']['asset_sha256']:raise ValueError('Scene/assets changed')
    if {n:importlib.metadata.version(n) for n in parent['packages']}!=parent['packages']:raise ValueError('Pinned packages changed')
    output=a.output.resolve()
    if output.exists() or any(output.is_relative_to(Path(x).resolve()) for x in (a.parent_study,a.yaw_study,root,a.assets)):
        raise ValueError('Fresh separate output required')
    output.mkdir(parents=True);(output/'requests').mkdir()
    for name in ('prior.json','old-prior.json','calibration.json','verification.json','setup-home.json'):shutil.copyfile(a.parent_study/name,output/name)
    atomic(output/'rule.json',rule);atomic(output/'plan.json',plan);atomic(output/'parent-rows.json',loaded['sources'])
    atomic(output/'control-replays.json',loaded['controls']);atomic(output/'yaw-rows.json',loaded['yaw_rows'])
    spec=dict(version=C.VERSION,protocol=C.protocol(),rule=rule,
        **{k:sha(output/n) for k,n in dict(plan_sha256='plan.json',rule_sha256='rule.json',parent_rows_sha256='parent-rows.json',
            control_replays_sha256='control-replays.json',yaw_rows_sha256='yaw-rows.json').items()},
        source_sha256=source_manifest(root),parent_bindings=loaded['bindings'],yaw_bindings=loaded['yaw_bindings'],
        scene=parent['scene'],identity=parent['identity'],packages=parent['packages'],runtime_reference=ref['runtime_reference'],created_unix_s=time.time())
    spec['fingerprint']=digest(spec);atomic(output/'contract.json',spec)
    rows=[];selection=[];number=0
    report=dict(version=C.VERSION,status='running',phase='frozen_plan',contract_fingerprint=spec['fingerprint'],rule=rule,
        protocol=C.protocol(),expected_requests=76,maximum_requests=C.MAXIMUM_REQUESTS,control_gates={},training=False,noise=False,posthoc_development=True,
        deadlines=dict(worker_limit_s=a.wall_time_limit,effective_worker_limit_s=a.effective_worker_limit,global_deadline_unix_s=a.global_deadline_unix),gate=dict(status='pending'))
    def publish(phase):
        report.update(phase=phase,elapsed_s=elapsed(),expected_requests=76+3*len(selection),counts=summarize(rows),selection=selection,
            counts_by_variant={v:summarize([r for r in rows if r['variant']==v]) for v in C.VARIANTS},cases=case_results(plan,rows,selection))
        atomic(output/'report.json',report);print(json.dumps(dict(status=report['status'],phase=phase,counts=report['counts'],elapsed_s=elapsed())),flush=True)
    def boundary(phase):
        append(output/'phase-boundaries.jsonl',[dict(phase=phase,acquired_elapsed_s=elapsed(),rows_count=len(rows),
            rows_sha256=sha(output/'rows.jsonl') if rows else None,plan_sha256=sha(output/'plan.json'))])
    publish('frozen_plan');boundary('plan-frozen')
    try:
        kwargs=dict(ref=ref,assets=a.assets,prior=read(output/'prior.json'),calibration=read(output/'calibration.json'),home=read(output/'setup-home.json'),plan=plan)
        engine=SweepEngine(directory=output/'native',parent=a.parent_study,sources=loaded['sources'],namespace='native',**kwargs)
        yaw=SweepEngine(directory=output/'native-yaw',parent=a.parent_study,sources=loaded['sources'],namespace='native-yaw',yaw_study=a.yaw_study,yaw_rows=loaded['yaw_rows'],**kwargs)
        ck=dict(kwargs,prior=read(output/'old-prior.json'))
        controls=SweepEngine(directory=output/'native-controls',parent=a.parent_study/'control-parent',sources=loaded['controls'],namespace='native-controls',**ck)
        wrappers={'native':engine,'native-yaw':yaw,'native-controls':controls};adapters=[x.engine.adapter for x in wrappers.values()]
        if any(x.env is not None for x in adapters):raise RuntimeError('Factory sharing must precede allocation')
        factory=engine.engine.adapter.environment_factory;allocated=[]
        def shared_factory():
            if not allocated:allocated.append(factory())
            return allocated[0]
        for adapter in adapters:adapter.environment_factory=shared_factory
        def execute(requests,phase):
            nonlocal number
            chunks=[];observed_phase=[]
            # One seven-gap tilt row per discovery batch bounds unsaved admission
            # work and checks the storage/deadline/foreign-GPU guards more often.
            chunk_limit=7 if phase=='discovery' else 32
            for request in requests:
                if not chunks or chunks[-1][0]['adapter_directory']!=request['adapter_directory'] or len(chunks[-1])>=chunk_limit:chunks.append([])
                chunks[-1].append(request)
            for chunk in chunks:
                if len(rows)+len(chunk)>C.MAXIMUM_REQUESTS:raise RuntimeError('223-request cap exceeded')
                if elapsed()>a.effective_worker_limit-30:raise TimeoutError('Finite acquired worker budget reached')
                if shutil.disk_usage(output).free<8*1024**3:raise RuntimeError('8GiB reserve reached')
                if foreign_processes():raise RuntimeError('Foreign4090 workload appeared; untouched')
                packet=f'requests/batch-{number:05d}.json';number+=1
                atomic(output/packet,dict(phase=phase,rows=chunk,pre_outcome=True));begin=elapsed()
                namespace=chunk[0]['adapter_directory'];observed,timing=wrappers[namespace].evaluate(chunk,phase)
                attach_robot_counts(observed,output/namespace)
                for row in observed:row.update(adapter_directory=namespace,learning_reward=reward(row),request_packet=packet,request_packet_sha256=sha(output/packet))
                append(output/'rows.jsonl',observed);rows.extend(observed);observed_phase.extend(observed)
                append(output/'timing.jsonl',[dict(phase=phase,**timing,request_packet=packet,request_packet_sha256=sha(output/packet),acquired_start_s=begin,acquired_end_s=elapsed())]);publish(phase)
            if phase.startswith('controls_'):
                gate=control_gate(observed_phase);report['control_gates']['pre' if phase=='controls_pre' else 'post']=gate
                if gate['status']!='passed':raise RuntimeError('Original direct/table/miss control gate failed')
            boundary(phase+'-complete')
        for phase in ('controls_pre','references','discovery'):execute([r for r in plan if r['phase']==phase],phase)
        selection=C.select_captures(rows);selected_plan=C.selected_requests(plan,selection)
        atomic(output/'selection.json',selection);atomic(output/'selected-plan.json',selected_plan);boundary('selection-frozen')
        execute(selected_plan,'selected_repeats');execute([r for r in plan if r['phase']=='controls_post'],'controls_post')
        expected=[r for r in plan if r['phase']!='controls_post']+selected_plan+[r for r in plan if r['phase']=='controls_post']
        if [r['id'] for r in rows]!=[r['id'] for r in expected]:raise RuntimeError('Complete frozen-plan denominator required')
        report.update(status='completed',gate=dict(status='passed',substeps=64,controls_passed=True,complete_plan=True,no_policy_promotion=True));publish('completed')
        manifest={str(p.relative_to(output)):sha(p) for p in output.rglob('*') if p.is_file() and p.name not in ('report.json','evidence-manifest.json')}
        atomic(output/'evidence-manifest.json',manifest);report['evidence_sha256']=manifest;atomic(output/'report.json',report)
        return report
    except Exception as error:
        report.update(status='failed',error=type(error).__name__+': '+str(error));publish('failed');raise


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in (*C.INPUT_KEYS,'output','gpu-lock'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--wall-time-limit',type=float,default=3600);p.add_argument('--queue-timeout',type=float,default=900)
    p.add_argument('--global-deadline-unix',type=float,required=True);a=p.parse_args()
    if not 60<=a.wall_time_limit<=3600 or not 0<a.queue_timeout<=900:raise ValueError('Finite approved limits required')
    if os.environ.get('CUDA_VISIBLE_DEVICES')!=GPU_UUID:raise RuntimeError('Only4090 UUID permitted')
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
