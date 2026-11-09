"""Bounded post-hoc v3 correction test; no fitting or active-policy replacement."""
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
import numpy as np

from . import table_correction_native as N
from .benchmark_verify import contained, read, sha, digest
from .corrected_grid import source_manifest
from .gpu_slot import GPU_UUID, gpu_slot, foreign_processes
from .policy64_contract import load_runtime_reference
from .sweep_worker import atomic
from .table_training_contract import statistics, reward, captured

VERSION=N.VERSION
PARENT_REPORT='4605bcb801351ae91ad16b71a033ceb3866027db621f9d84227f23b1ef257e95'
PARENT_AUDIT='6e50d87ddb7b2d4b37debbc2a5ac96507ff40bc236223277121acd385b8ec91b'


def control_gate(rows):
    direct,table,negative=([r for r in rows if r['role']==role] for role in N.CONTROL_ROLES)
    common=all(r.get('native_release') is True and not r.get('invalid')
        and not r.get('invalid_physics') and r.get('scalar_score_audit_passed') is True
        and r.get('contact_protocol',{}).get('substeps')==64 for r in rows)
    good=(len(rows)==9 and all(len(g)==3 for g in (direct,table,negative)) and common
        and all(captured(r) and r.get('direct_capture') is True and r.get('table_contact_steps')==0
                and r.get('historical_v2_score',{}).get('route_success') is True for r in direct)
        and all(captured(r) and r.get('table_assisted_capture') is True and r.get('table_contact_steps',0)>0
                and r.get('historical_v2_score',{}).get('route_success') is False for r in table)
        and all(not captured(r) and not r.get('whole_ball_entry')
                and r.get('historical_v2_score',{}).get('route_success') is False for r in negative))
    return dict(status='passed' if good else 'failed',counts=statistics(rows),
        direct_captures=sum(captured(r) for r in direct),table_assisted_captures=sum(captured(r) for r in table),
        negative_captures=sum(captured(r) for r in negative))


def authenticate_parent(parent,audit):
    parent=Path(parent);audit=Path(audit)
    if sha(parent/'report.json')!=PARENT_REPORT or sha(audit)!=PARENT_AUDIT:
        raise ValueError('Exact completed parent report/audit required')
    proof=read(audit);report=read(parent/'report.json');spec=read(parent/'contract.json')
    if (proof.get('status')!='passed' or proof.get('full_checks_verified') is not True
        or proof.get('failures') or proof.get('checks_count')!=823442
        or len(proof.get('checks',[]))!=823442
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


def summarize(rows):
    result=statistics(rows)
    result['invalid_physics']=sum(bool(r.get('invalid_physics')) for r in rows)
    result['robot_contact_shots']=sum(bool(r.get('diagnostic_robot_contact_steps')) for r in rows)
    return result


def attach_robot_counts(rows,directory):
    """Keep source scores intact; expose robot-only evidence separately."""
    files={}
    for row in rows:
        row['diagnostic_robot_contact_steps']=0
        if row.get('native_release'):
            files.setdefault(row['native_cache_file'],[]).append(row)
    for name,members in files.items():
        with np.load(contained(directory,name),allow_pickle=False) as trace:
            force=trace['forces_N']
            for row in members:
                row['diagnostic_robot_contact_steps']=int(np.count_nonzero(
                    force[:row['physics_steps'],row['native_cache_row'],4]>.01))


def run(a):
    root=Path(__file__).resolve().parents[2]
    parent,original,bindings=authenticate_parent(a.parent_study,a.parent_audit)
    rule=N.validate_rule(read(a.rule));plan=N.make_plan(original,rule)
    selected={r['source_record_id'] for r in plan};sources=[r for r in original if r['id'] in selected]
    ref=load_runtime_reference(a.reference,a.assets,root)
    if digest(ref['scene'])!=parent['identity']['scene_sha256'] or ref['asset_hashes']!=parent['identity']['asset_sha256']:
        raise ValueError('Scene or robot assets differ from audited parent')
    if {n:importlib.metadata.version(n) for n in parent['packages']}!=parent['packages']:
        raise ValueError('Pinned parent package versions differ')
    output=a.output.resolve()
    if output.exists() or any(output.is_relative_to(Path(x).resolve()) for x in (a.parent_study,root,a.assets)):
        raise ValueError('Separate fresh evidence namespace required')
    output.mkdir(parents=True);started=time.monotonic()
    for name in ('calibration.json','verification.json','setup-home.json','prior.json'):
        shutil.copyfile(a.parent_study/name,output/name)
    atomic(output/'rule.json',rule);atomic(output/'plan.json',plan);atomic(output/'parent-rows.json',sources)
    spec=dict(version=VERSION,rule=rule,plan_sha256=sha(output/'plan.json'),parent_rows_sha256=sha(output/'parent-rows.json'),
        rule_sha256=sha(output/'rule.json'),source_sha256=source_manifest(root),parent_bindings=bindings,
        scene=parent['scene'],identity=parent['identity'],packages=parent['packages'],substeps=64,
        maximum_requests=42,actual_planned_requests=len(plan),maximum_acquired_s=3600,repeats=3,
        training=False,noise=False,posthoc_development=True,created_unix_s=time.time())
    spec['fingerprint']=digest(spec);atomic(output/'contract.json',spec)
    report=dict(version=VERSION,status='running',phase='preparing',contract_fingerprint=spec['fingerprint'],
        description=rule['description'],changed_parameter=rule['parameter'],target_value=rule['target_value'],
        unchanged=[x for x in N.PARAMETERS if x!=rule['parameter']]+['bucket','v3 scorer','64 substeps','admission guards'],
        counts_by_variant={},cases=[],control_gates={},posthoc_development=True,training=False,expected_requests=42)
    rows=[]
    def publish(phase):
        report.update(phase=phase,elapsed_s=time.monotonic()-started,counts=summarize(rows),
            counts_by_variant={v:summarize([r for r in rows if r['variant']==v]) for v in ('baseline','corrected')})
        report['cases']=[]
        for context in rule['contexts']:
            cohort=[r for r in rows if r['context_id']==context and r['phase']=='diagnostic']
            requests=[r for r in plan if r['context_id']==context]
            report['cases'].append(dict(context_id=context,source_record_id=requests[0]['source_record_id'],
                bucket_xy_m=requests[0]['bucket_xy_m'],baseline_values=requests[0]['values'],
                corrected_values=next(r['values'] for r in requests if r['variant']=='corrected'),
                variants={v:summarize([r for r in cohort if r['variant']==v]) for v in ('baseline','corrected')}))
        atomic(output/'report.json',report)
        print(json.dumps(dict(status=report['status'],phase=phase,counts=report['counts'])),flush=True)
    def limits():
        if len(rows)>len(plan) or time.monotonic()-started>a.wall_time_limit-15:
            raise TimeoutError('Own finite correction budget reached')
        if shutil.disk_usage(output).free<8*1024**3:raise RuntimeError('8 GiB reserve reached; evidence retained')
        if foreign_processes():raise RuntimeError('Foreign4090 job appeared; leaving it untouched')
    publish('frozen_plan')
    engine=N.MatchedEngine(ref,a.assets,output/'native',parent=a.parent_study,sources=sources,
        prior=read(output/'prior.json'),calibration=read(output/'calibration.json'),home=read(output/'setup-home.json'))
    def execute(requests,phase):
        limits()
        if len(rows)+len(requests)>len(plan):raise RuntimeError('Prescribed request budget exceeded')
        observed,timing=engine.evaluate(requests,phase)
        attach_robot_counts(observed,output/'native')
        for row in observed:row['learning_reward']=reward(row)
        with (output/'rows.jsonl').open('a') as f:
            for row in observed:f.write(json.dumps(row,allow_nan=False)+'\n')
        with (output/'timing.jsonl').open('a') as f:f.write(json.dumps(dict(phase=phase,**timing),allow_nan=False)+'\n')
        rows.extend(observed);publish(phase);return observed
    try:
        for phase,key,requests in [('controls_pre','pre',plan[:9])]:
            gate=control_gate(execute(requests,phase));report['control_gates'][key]=gate
            if gate['status']!='passed':raise RuntimeError('Initial direct/table/miss controls failed')
        for context in rule['contexts']:
            execute([r for r in plan if r['phase']=='diagnostic' and r['context_id']==context],'diagnostic')
        gate=control_gate(execute(plan[-9:],'controls_post'));report['control_gates']['post']=gate
        if gate['status']!='passed':raise RuntimeError('Final direct/table/miss controls failed')
        if len(rows)!=len(plan):raise RuntimeError('Complete frozen-plan accounting required')
        limits();report['status']='completed';publish('completed')
        manifest={str(p.relative_to(output)):sha(p) for p in sorted(output.rglob('*'))
                  if p.is_file() and p.name not in ('report.json','evidence-manifest.json')}
        atomic(output/'evidence-manifest.json',manifest);report['evidence_sha256']=manifest;atomic(output/'report.json',report)
        return report
    except Exception as error:
        report.update(status='failed',error=type(error).__name__+': '+str(error));publish('failed');raise


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('parent-study','parent-audit','reference','assets','rule','output','gpu-lock'):
        p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--wall-time-limit',type=float,default=3600);p.add_argument('--queue-timeout',type=float,default=1800);a=p.parse_args()
    if not 60<=a.wall_time_limit<=3600 or not 0<a.queue_timeout<=7200:raise ValueError('Finite caps required')
    if os.environ.get('CUDA_VISIBLE_DEVICES')!=GPU_UUID:raise RuntimeError('Only authorized4090UUID may execute')
    def deadline(signum,frame):raise TimeoutError('Own correction acquired-slot deadline')
    with gpu_slot(a.gpu_lock,timeout_s=a.queue_timeout,notify=lambda r:print(json.dumps(r),flush=True)):
        atomic(a.output.with_name(a.output.name+'-budget.json'),dict(acquired_unix_s=time.time(),limit_s=a.wall_time_limit))
        signal.signal(signal.SIGALRM,deadline);signal.setitimer(signal.ITIMER_REAL,a.wall_time_limit-10)
        try:run(a)
        finally:signal.setitimer(signal.ITIMER_REAL,0)

if __name__=='__main__':main()
