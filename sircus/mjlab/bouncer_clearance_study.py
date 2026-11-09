"""Separate bounded CPU admission and leased4090 native phases for a fixed grid."""
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
from . import bouncer_clearance_contract as C
from .bouncer_clearance_native import BouncerClearanceEngine
from .benchmark_verify import read,sha,digest,contained
from .corrected_grid import source_manifest
from .gpu_slot import GPU_UUID,gpu_slot,foreign_processes
from .policy64_contract import load_runtime_reference
from .sweep_worker import atomic
from .table_correction_study import control_gate,summarize,attach_robot_counts
from .table_training_contract import reward


PARENT_REPORT='a93d348dd055fde3ec1e9dc660c278fdc3e5f38742bce93abd71da03f0d066c7'
PARENT_AUDIT='38f0553a0cf8ca3948f8b9624d55e05ce07b2d9c1c0b9934c22d7e0f8e8bbe00'


def authenticate_parent(parent,audit):
    parent=Path(parent);audit=Path(audit)
    if sha(parent/'report.json')!=PARENT_REPORT or sha(audit)!=PARENT_AUDIT:
        raise ValueError('Exact completed height/tilt parent report/audit required')
    proof=read(audit);report=read(parent/'report.json');spec=read(parent/'contract.json')
    if (proof.get('status')!='passed' or proof.get('full_checks_verified') is not True
        or proof.get('native_checks_verified') is not True or proof.get('failures')
        or proof.get('checks_count')!=36371 or len(proof.get('checks',[]))!=36371
        or not all(x.get('passed') is True for x in proof['checks'])
        or proof.get('report_sha256')!=PARENT_REPORT
        or proof.get('contract_fingerprint')!=spec['fingerprint']
        or report.get('status')!='completed'):
        raise ValueError('Completed bound parent proof failed')
    unsigned=dict(spec);fp=unsigned.pop('fingerprint')
    if digest(unsigned)!=fp:raise ValueError('Parent contract fingerprint differs')
    manifest=read(parent/'evidence-manifest.json')
    if sha(parent/'evidence-manifest.json')!=proof['evidence_manifest_sha256']:
        raise ValueError('Parent evidence manifest differs')
    for name in ('rows.jsonl','contract.json','calibration.json','verification.json','setup-home.json','prior.json',
                 'native/effective-options.json','native/native-compiled-geometry.json'):
        if sha(parent/name)!=manifest[name]:raise ValueError('Bound parent evidence changed: '+name)
    rows=[json.loads(line) for line in (parent/'rows.jsonl').read_text().splitlines()]
    return spec,rows,dict(report_sha256=PARENT_REPORT,audit_sha256=PARENT_AUDIT,
        manifest_sha256=sha(parent/'evidence-manifest.json'),rows_sha256=sha(parent/'rows.jsonl'),
        contract_sha256=sha(parent/'contract.json'),contract_fingerprint=fp)


def parent_inputs(a,root):
    parent,rows,bindings=authenticate_parent(a.parent_study,a.parent_audit)
    ref=load_runtime_reference(a.reference,a.assets,root)
    if digest(ref['scene'])!=parent['identity']['scene_sha256'] or ref['asset_hashes']!=parent['identity']['asset_sha256']:
        raise ValueError('Historical scene or native assets changed')
    if {n:importlib.metadata.version(n) for n in parent['packages']}!=parent['packages']:
        raise ValueError('Pinned native dependencies changed')
    return parent,rows,bindings,ref


def bound_engine(a,ref,sources,plan):
    return BouncerClearanceEngine(ref,a.assets,a.output/'native',parent=a.parent_study,sources=sources,
        prior=read(a.output/'prior.json'),calibration=read(a.output/'calibration.json'),
        home=read(a.output/'setup-home.json'),plan=plan)


def manifest(output):
    result={str(p.relative_to(output)):sha(p) for p in sorted(output.rglob('*'))
            if p.is_file() and p.name not in ('report.json','evidence-manifest.json')}
    atomic(output/'evidence-manifest.json',result);return result


def report_state(output,spec,matrix,rows,selection,*,status,phase,cpu_elapsed_s,native_elapsed_s=0.,gates=None,error=None):
    sources=read(output/'parent-rows.json');by_context={r['context_id']:r for r in sources}
    observed={r['id']:r for r in rows};selected={s['context_id']:s['selected'] for s in selection}
    cases=[]
    for context in C.CONTEXTS:
        cells=[]
        for cell in matrix:
            if cell['context_id']!=context:continue
            row=observed.get(cell['candidate_id']);discovery=None
            if row is not None:
                discovery=dict(record_id=row['id'],record_sha256=digest(row),native_release=row['native_release'],captured=C.captured(row),
                    route_quality=row['route_quality'],direct_capture=row['direct_capture'],table_assisted_capture=row['table_assisted_capture'],
                    invalid_physics=row.get('invalid_physics',False),invalid=row.get('invalid'),
                    robot_contact_steps=row['diagnostic_robot_contact_steps'],forbidden_contact_steps=row['forbidden_contact_steps'],
                    reward=row['learning_reward'])
            admission={k:copy.deepcopy(v) for k,v in cell.items() if k not in ('values','bucket_xy_m','candidate_id','context_id','grid_index','case_grid_index','q10','q14')}
            cells.append(dict(candidate_id=cell['candidate_id'],grid_index=cell['grid_index'],case_grid_index=cell['case_grid_index'],
                values=cell['values'],bucket_xy_m=cell['bucket_xy_m'],admission=admission,discovery=discovery,
                selected=bool(selected.get(context) and selected[context]['candidate_id']==cell['candidate_id'])))
        cases.append(dict(context_id=context,source_record_id=by_context[context]['id'],source_record_sha256=digest(by_context[context]),baseline_values=by_context[context]['values'],
            bucket_xy_m=by_context[context]['bucket_xy_m'],grid=cells,selection=selected.get(context),
            repeats={v:summarize([r for r in rows if r['context_id']==context and r['variant']==v]) for v in ('baseline','selected')}))
    admission=dict(proposals=75,completed=sum(r['status']=='completed' for r in matrix),
        admitted=sum(r['passed'] is True for r in matrix),declined=sum(r['passed'] is False for r in matrix),
        unattempted=sum(r['status']!='completed' for r in matrix))
    discovery=[r for r in rows if r['phase']=='discovery']
    result=dict(version=C.VERSION,status=status,phase=phase,contract_fingerprint=spec.get('fingerprint'),
        parent_bindings=spec['parent_bindings'],grid_protocol=C.protocol(),budget=C.budget(),
        admission_matrix_sha256=spec.get('admission_matrix_sha256'),admission_summary=admission,
        discovery_summary=dict(proposals=75,admitted=admission['admitted'],**summarize(discovery)),
        counts=summarize(rows),native_requested=len(rows),
        requested_configurations=admission['completed']+sum(r['phase']!='discovery' for r in rows),
        cases=cases,selection=selection,repeats={v:summarize([r for r in rows if r['variant']==v]) for v in ('baseline','selected')},
        control_gates=gates or {},cpu_elapsed_s=cpu_elapsed_s,native_elapsed_s=native_elapsed_s,
        training=False,noise=False,posthoc_development=True)
    if error:result['error']=error
    atomic(output/'report.json',result)
    print(json.dumps(dict(status=status,phase=phase,admission=admission,native_requested=len(rows),repeats=result['repeats'])),flush=True)
    return result


def finish_admission(output,base,plan,sources,matrix,elapsed,*,complete,error=None):
    for entry in matrix:
        if entry['status']!='completed':entry['reason']='admission_budget_or_error_before_attempt'
    atomic(output/'admission-matrix.json',matrix)
    lock=dict(version=C.VERSION,admission_matrix_sha256=sha(output/'admission-matrix.json'),
        request_contract_sha256=sha(output/'request-contract.json'),complete=complete,
        completed=sum(r['status']=='completed' for r in matrix),cpu_elapsed_s=elapsed,locked_unix_s=time.time())
    atomic(output/'admission-lock.json',lock)
    admitted={r['candidate_id'] for r in matrix if r['passed'] is True}
    pre=C.fixed_requests(sources,'controls_pre')+C.fixed_requests(sources,'baseline')+[r for r in plan if r['id'] in admitted]
    atomic(output/'pre-native-plan.json',pre)
    spec=dict(base,request_contract_sha256=sha(output/'request-contract.json'),admission_matrix_sha256=sha(output/'admission-matrix.json'),
        admission_lock_sha256=sha(output/'admission-lock.json'),admission_complete=complete,
        pre_native_plan_sha256=sha(output/'pre-native-plan.json'))
    spec.pop('fingerprint',None);spec['fingerprint']=digest(spec);atomic(output/'contract.json',spec)
    report=report_state(output,spec,matrix,[],[],status='admission_complete' if complete else 'admission_stopped',
        phase='awaiting_gpu' if complete else 'admission_stopped',cpu_elapsed_s=elapsed,error=error)
    report['evidence_sha256']=manifest(output);atomic(output/'report.json',report)


def admit(a):
    root=Path(__file__).resolve().parents[2];started=time.monotonic()
    parent,rows,bindings,ref=parent_inputs(a,root);sources=C.sources_from_parent(rows);plan=C.grid_plan(sources)
    output=a.output.resolve()
    if output.exists() or any(output.is_relative_to(Path(x).resolve()) for x in (a.parent_study,root,a.assets)):
        raise ValueError('Fresh separate evidence namespace required')
    output.mkdir(parents=True)
    for name in ('calibration.json','verification.json','setup-home.json','prior.json'):shutil.copyfile(a.parent_study/name,output/name)
    atomic(output/'plan.json',plan);atomic(output/'parent-rows.json',sources)
    base=dict(version=C.VERSION,grid_protocol=C.protocol(),budget=C.budget(),parent_bindings=bindings,
        source_sha256=source_manifest(root),scene=parent['scene'],identity=parent['identity'],packages=parent['packages'],
        plan_sha256=sha(output/'plan.json'),parent_rows_sha256=sha(output/'parent-rows.json'),created_unix_s=time.time(),
        training=False,noise=False,posthoc_development=True)
    base['fingerprint']=digest(base);atomic(output/'request-contract.json',base)
    matrix=C.empty_matrix(plan);atomic(output/'admission-matrix.json',matrix)
    report_state(output,base,matrix,[],[],status='admitting',phase='cpu_admission',cpu_elapsed_s=time.monotonic()-started)
    try:
        engine=bound_engine(a,ref,sources,plan)
        for index,request in enumerate(plan):
            if time.monotonic()-started>=a.cpu_limit-15:raise TimeoutError('Finite CPU admission cap reached')
            if shutil.disk_usage(output).free<C.budget()['storage_reserve_bytes']:raise RuntimeError('8GiB storage reserve reached')
            matrix[index]=engine.admit(request);atomic(output/'admission-matrix.json',matrix)
            if (index+1)%5==0:report_state(output,base,matrix,[],[],status='admitting',phase='cpu_admission',cpu_elapsed_s=time.monotonic()-started)
        finish_admission(output,base,plan,sources,matrix,time.monotonic()-started,complete=True)
    except Exception as error:
        finish_admission(output,base,plan,sources,matrix,time.monotonic()-started,complete=False,error=type(error).__name__+': '+str(error));raise


def native(a):
    root=Path(__file__).resolve().parents[2]
    parent,original,bindings,ref=parent_inputs(a,root)
    output=a.output;spec=read(output/'contract.json');unsigned=dict(spec);fp=unsigned.pop('fingerprint')
    if digest(unsigned)!=fp or not spec['admission_complete']:raise ValueError('Complete frozen admission required before GPU')
    if spec['source_sha256']!=source_manifest(root) or spec['parent_bindings']!=bindings:
        raise ValueError('Frozen producer source or original evidence changed between phases')
    for key in ('scene','identity','packages'):
        if spec[key]!=parent[key]:raise ValueError('Original runtime identity changed between phases: '+key)
    for name,key in (('plan.json','plan_sha256'),('parent-rows.json','parent_rows_sha256'),
                     ('request-contract.json','request_contract_sha256'),('admission-matrix.json','admission_matrix_sha256'),
                     ('admission-lock.json','admission_lock_sha256'),('pre-native-plan.json','pre_native_plan_sha256')):
        if sha(output/name)!=spec[key]:raise ValueError('Frozen admission input bytes changed: '+name)
    sources=read(output/'parent-rows.json');plan=read(output/'plan.json');matrix=read(output/'admission-matrix.json')
    if sources!=C.sources_from_parent(original) or plan!=C.grid_plan(sources):raise ValueError('Frozen inputs no longer reconstruct')
    pre=read(output/'pre-native-plan.json');lock=read(output/'admission-lock.json')
    if not lock['complete'] or lock['admission_matrix_sha256']!=spec['admission_matrix_sha256']:
        raise ValueError('Immutable matrix lock mismatch')
    if (output/'rows.jsonl').exists():raise ValueError('Never restart or overwrite an existing native attempt')
    engine=bound_engine(a,ref,sources,plan);engine.hydrate_matrix(matrix)
    rows=[];selection=[];gates={};started=None
    def deadline(signum,frame):raise TimeoutError('Own acquired4090 native cap reached')
    with gpu_slot(a.gpu_lock,timeout_s=a.queue_timeout,notify=lambda x:print(json.dumps(x),flush=True)):
        started=time.monotonic()
        atomic(output.with_name(output.name+'-native-budget.json'),dict(acquired_unix_s=time.time(),limit_s=a.native_limit))
        signal.signal(signal.SIGALRM,deadline);signal.setitimer(signal.ITIMER_REAL,a.native_limit-10)
        def publish(status,phase,error=None):
            return report_state(output,spec,matrix,rows,selection,status=status,phase=phase,
                cpu_elapsed_s=lock['cpu_elapsed_s'],native_elapsed_s=time.monotonic()-started,gates=gates,error=error)
        def execute(requests,phase):
            if not requests:return []
            if time.monotonic()-started>=a.native_limit-15:raise TimeoutError('Finite native wall cap reached')
            if 75+sum(r['phase']!='discovery' for r in rows+requests)>111:raise RuntimeError('Total111 candidate/repeat budget exceeded')
            if shutil.disk_usage(output).free<C.budget()['storage_reserve_bytes']:raise RuntimeError('8GiB storage reserve reached')
            if foreign_processes():raise RuntimeError('Foreign4090 workload appeared; leaving it untouched')
            result,timing=engine.evaluate(requests,phase);attach_robot_counts(result,output/'native')
            for row in result:row['learning_reward']=reward(row)
            with (output/'rows.jsonl').open('a') as f:
                for row in result:f.write(json.dumps(row,allow_nan=False)+'\n')
            with (output/'timing.jsonl').open('a') as f:f.write(json.dumps(dict(phase=phase,**timing),allow_nan=False)+'\n')
            rows.extend(result);publish('running',phase);return result
        try:
            for phase in ('controls_pre','baseline','discovery'):
                requests=[r for r in pre if r['phase']==phase]
                results=[]
                for index in range(0,len(requests),32):results+=execute(requests[index:index+32],phase)
                if phase=='controls_pre':
                    gates['pre']=control_gate(results)
                    if gates['pre']['status']!='passed':raise RuntimeError('Initial direct/table/miss controls failed')
            selection=C.select_clear(rows);atomic(output/'selection.json',selection)
            post=C.selected_requests(plan,selection)+C.fixed_requests(sources,'controls_post')
            atomic(output/'post-selection-plan.json',post)
            for phase in ('selected_repeats','controls_post'):
                results=execute([r for r in post if r['phase']==phase],phase)
                if phase=='controls_post':
                    gates['post']=control_gate(results)
                    if gates['post']['status']!='passed':raise RuntimeError('Final direct/table/miss controls failed')
            if [r['id'] for r in rows]!=[r['id'] for r in pre+post]:raise RuntimeError('Frozen native request accounting differs')
            report=publish('completed','completed');report['evidence_sha256']=manifest(output);atomic(output/'report.json',report)
        except Exception as error:
            report=publish('native_stopped','native_stopped',type(error).__name__+': '+str(error))
            report['evidence_sha256']=manifest(output);atomic(output/'report.json',report);raise
        finally:signal.setitimer(signal.ITIMER_REAL,0)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--phase',choices=('admit','native'),required=True)
    for key in ('parent-study','parent-audit','reference','assets','output','gpu-lock'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--cpu-limit',type=float,default=3600);p.add_argument('--native-limit',type=float,default=3600)
    p.add_argument('--queue-timeout',type=float,default=7200);a=p.parse_args()
    if not 60<=a.cpu_limit<=3600 or not 60<=a.native_limit<=3600 or not 0<a.queue_timeout<=7200:
        raise ValueError('Finite requested CPU/native/queue caps required')
    if a.phase=='admit':
        if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise RuntimeError('CPU admission requires CUDA_VISIBLE_DEVICES empty')
        atomic(a.output.with_name(a.output.name+'-cpu-budget.json'),dict(started_unix_s=time.time(),limit_s=a.cpu_limit))
        def deadline(signum,frame):raise TimeoutError('Own CPU admission deadline reached')
        signal.signal(signal.SIGALRM,deadline);signal.setitimer(signal.ITIMER_REAL,a.cpu_limit-10)
        try:admit(a)
        finally:signal.setitimer(signal.ITIMER_REAL,0)
    else:
        if os.environ.get('CUDA_VISIBLE_DEVICES')!=GPU_UUID:raise RuntimeError('Only authorized4090 UUID may execute native phase')
        native(a)

if __name__=='__main__':main()
