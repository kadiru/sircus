"""Bounded four-pose/branch search plus two CPU clearance audits; no training or UI."""
from __future__ import annotations
import argparse
import copy
import datetime
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import signal
import time
from . import fixed35_six_contract as C
from .benchmark_verify import digest, sha, read
from .corrected_grid import source_manifest
from .coverage_training_study import append, share_native_environment
from .gpu_slot import GPU_UUID, gpu_slot, foreign_processes
from .policy64_contract import load_runtime_reference
from .table_correction_native import MatchedEngine, META
from .fixed35_six_native import SixEngine
from .fixed35_six_geometry import audit_geometry
from .corrected_imitation_study import mirror_control_proofs
from .table_training_study import control_gate
from .sweep_worker import atomic

INPUT_KEYS=C.INPUT_KEYS


def run(a):
    root=Path(__file__).resolve().parents[2];started=time.monotonic();acquired=getattr(a,'acquired_monotonic',started)
    elapsed=lambda:time.monotonic()-acquired
    loaded=C.load_inputs(a,root);parent=loaded['parent'];ref=load_runtime_reference(a.reference,a.assets,root)
    C.require(digest(ref['scene'])==parent['identity']['scene_sha256'] and ref['asset_hashes']==parent['identity']['asset_sha256'],'Parent scene/assets differ')
    C.require({n:importlib.metadata.version(n) for n in parent['packages']}==parent['packages'],'Pinned dependencies differ')
    output=Path(a.output).resolve()
    C.require(not output.exists() and not any(output.is_relative_to(Path(p).resolve()) for p in (root,a.parent_study,a.original_study,a.assets)),'Fresh separate evidence directory required')
    output.mkdir(parents=True)
    for name in ('bootstrap','requests','repeat-plans'):(output/name).mkdir()
    for name in ('calibration.json','verification.json','setup-home.json'):shutil.copyfile(a.parent_study/name,output/name)
    for name,key in (('prior.json','prior'),('old-prior.json','old_prior'),('controls.json','controls'),('control-replays.json','control_replays')):atomic(output/name,loaded[key])
    for name,key in (('preserved','preserved'),('cases','cases'),('blocked','blocked'),('parent-current','current'),('parent-refit','refit')):
        atomic(output/f'bootstrap/{name}.json',loaded[key])
    atomic(output/'diagnoses.json',loaded['diagnoses']);atomic(output/'source-bindings.json',loaded['bindings'])
    atomic(output/'grid.json',loaded['grid'])
    geometry_cases=[(r['context_id'],r['bucket_xy_m']) for r in loaded['blocked']]
    geometry_control=next(r for r in loaded['grid'] if r['id']=='fixed35-dev-grid-02-06')
    geometry_cases.append((geometry_control['id'],geometry_control['bucket_xy_m']))
    geometry=audit_geometry(ref['scene'],a.assets,cases=geometry_cases)
    atomic(output/'geometry-audit.json',geometry)
    plan=C.make_plan(loaded['cases'],loaded['refit'],loaded['preserved']);preserved=C.preserved_plan(loaded['preserved'])
    atomic(output/'search-plan.json',dict(rows=plan,frozen_before_outcomes=True,maximum_requests=C.SEARCH_REQUESTS))
    atomic(output/'preserved-plan.json',dict(rows=preserved,frozen_before_outcomes=True))
    rule=read(Path(__file__).with_name('fixed35_six_rule.json'))
    C.require(rule['version']==C.VERSION and rule['protocol']==C.protocol(),'Checked-in frozen rule differs')
    atomic(output/'rule.json',rule)
    spec=dict(version=C.VERSION,scope=C.SCOPE,protocol=C.protocol(),identity=copy.deepcopy(parent['identity']),parent_identity=copy.deepcopy(parent['identity']),
        scene=ref['scene'],packages=parent['packages'],runtime_reference=ref['runtime_reference'],source_sha256=source_manifest(root),
        source_bindings=loaded['bindings'],prior_sha256=digest(loaded['prior']),old_prior_sha256=digest(loaded['old_prior']),
        geometry_audit_sha256=sha(output/'geometry-audit.json'),
        search_plan_sha256=sha(output/'search-plan.json'),preserved_plan_sha256=sha(output/'preserved-plan.json'),rule_sha256=sha(output/'rule.json'),
        grid=loaded['grid'],parent_artifacts=loaded['artifacts'],release_behavior=C.ROTATION,
        created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    spec['identity']['source_sha256']=spec['source_sha256'];spec['fingerprint']=digest(spec);atomic(output/'contract.json',spec)
    rows=[];report=dict(version=C.VERSION,status='running',phase='setup',scope=C.SCOPE,contract_fingerprint=spec['fingerprint'],protocol=C.protocol(),
        expected_requests=C.MAXIMUM_REQUESTS,maximum_requests=C.MAXIMUM_REQUESTS,training=False,posthoc_development=True,evaluation_scope=C.protocol()['evaluation_scope'],
        parent_captures=29,historical_policy_captures=24,parent_positions=35,preserved_recipes=29,control_gates={},cases=[],budget_truncated=False,completion_reason=None,
        artifacts=copy.deepcopy(loaded['artifacts']),deadlines=dict(worker_limit_s=a.wall_time_limit,effective_worker_limit_s=getattr(a,'effective_worker_limit',a.wall_time_limit),
            global_deadline_unix_s=getattr(a,'global_deadline_unix',None)),gate=dict(status='pending'))
    def publish(phase):
        report.update(phase=phase,elapsed_s=elapsed(),counts=C.statistics(rows),actual_requests=len(rows),
            stage_counts={p:C.statistics([r for r in rows if r['phase']==p]) for p in ('controls_pre','preserved','search','repeat','controls_post')},
            cases=C.case_results(rows,loaded['cases'],loaded['blocked'],finished=report['status']=='completed',budget_truncated=report['budget_truncated']))
        confirmed=[r for r in report['cases'] if r['confirmed_recipe'] is not None]
        fresh=[r for r in rows if r['phase']=='preserved'];kept=sum(C.captured(r) for r in fresh)
        report.update(recovered_layouts=len(confirmed),preservation=dict(requests=len(fresh),captures=kept,expected=29,all_replayed=len(fresh)==29,all_recaptured=len(fresh)==29 and kept==29),
            demonstrated_recipe_coverage=dict(historical_recipe_coverage=29,historical_policy_captures=24,new_confirmed_layouts=len(confirmed),fresh_preserved_captures=kept,
                fresh_total=kept+len(confirmed),positions=35,not_a_policy_score=True))
        atomic(output/'report.json',report)
        print(json.dumps(dict(phase=phase,status=report['status'],counts=report['counts'],recovered_layouts=len(confirmed),elapsed_s=elapsed())),flush=True)
    def boundary(phase,**extra):
        append(output/'phase-boundaries.jsonl',[dict(phase=phase,acquired_elapsed_s=elapsed(),rows_count=len(rows),rows_sha256=sha(output/'rows.jsonl') if rows else None,**extra)])
    publish('setup');boundary('inputs-frozen')
    engine=SixEngine(ref,a.assets,output/'native-yaw',parent=a.parent_study,sources=loaded['sources'],source_roots=loaded['source_roots'],
        prior=loaded['prior'],calibration=loaded['calibration'],home=loaded['home'])
    atomic(output/'native-source-receipts.json',engine.source_receipts)
    parent_options=read(a.parent_study/'native-yaw/effective-options.json');parent_geometry=read(a.parent_study/'native-yaw/native-compiled-geometry.json')
    mirror=mirror_control_proofs(a.parent_study,output,loaded['control_replays'])
    controls=MatchedEngine(ref,a.assets,output/'native-controls',parent=mirror,sources=loaded['control_replays'],prior=loaded['old_prior'],calibration=loaded['calibration'],home=loaded['home'])
    share_native_environment(engine,controls)
    def limits(n=0):
        C.require(len(rows)+n<=C.MAXIMUM_REQUESTS,'851-request cap exceeded')
        if elapsed()>getattr(a,'effective_worker_limit',a.wall_time_limit)-30:raise TimeoutError('Finite own acquired budget reached')
        C.require(shutil.disk_usage(output).free>=C.protocol()['storage_reserve_bytes'],'8GiB reserve reached; evidence retained')
        C.require(not foreign_processes(),'Foreign4090 workload appeared; untouched')
    number=0
    def execute(requests,phase,*,control=False):
        nonlocal number
        result=[]
        for offset in range(0,len(requests),32):
            chunk=requests[offset:offset+32];limits(len(chunk));packet=f'requests/batch-{number:05d}.json';number+=1
            atomic(output/packet,dict(phase=phase,rows=chunk,pre_outcome=True));start=elapsed()
            observed,timing=(controls if control else engine).evaluate(chunk,phase)
            C.require([r['id'] for r in observed]==[r['id'] for r in chunk],'Requested order changed')
            if engine.adapter.env is not None:C.require(engine.adapter.env.effective_options==parent_options and engine.adapter.env.compiled_geometry==parent_geometry,'Native options/geometry changed')
            for r in observed:r.update(learning_reward=C.reward(r),request_packet=packet,request_packet_sha256=sha(output/packet),adapter_directory='native-controls' if control else 'native-yaw')
            append(output/'rows.jsonl',observed);rows.extend(observed);result.extend(observed)
            append(output/'timing.jsonl',[dict(phase=phase,**timing,acquired_start_s=start,acquired_end_s=elapsed(),request_packet=packet,request_packet_sha256=sha(output/packet))]);publish(phase)
        return result
    def control_block(phase):
        requests=[]
        for original,source,role in zip(loaded['controls'],loaded['control_replays'],('direct_control','table_control','negative_control')):
            C.require(source['role']==role and source['values']==original['values'] and source['bucket_xy_m']==original['bucket_xy_m'],'Original control inputs differ')
            for i in range(3):
                requests.append(dict({k:copy.deepcopy(source[k]) for k in META},id=f'fixed35-six-{phase}-{role}-r{i}',phase=phase,
                    role=role,method='control',context_id=source['context_id'],bucket_xy_m=source['bucket_xy_m'],values=source['values'],command_values=source['values'],
                    source_record_id=source['id'],source_record_sha256=digest(source),variant='control_exact',adapter_directory='native-controls',
                    control_source_id=original['id'],control_source_sha256=digest(original),repeat=i,perturbation_mixed_units=[0.]*5))
        gate=control_gate(execute(requests,phase,control=True));report['control_gates']['pre' if phase=='controls_pre' else 'post']=gate
        C.require(gate['status']=='passed','Exact original17 control gate failed');boundary(phase+'-complete')
    control_block('controls_pre')
    execute(preserved,'preserved');boundary('preserved-complete')
    pools={r['context_id']:[] for r in loaded['cases']}
    for r in plan:pools[r['context_id']].append(r)
    stop={};confirm_counts={c:0 for c in pools}
    for index in range(192):
        active=[c for c in pools if c not in stop]
        if not active:break
        if elapsed()>=C.protocol()['search_limit_s']:
            report['budget_truncated']=True;report['completion_reason']='search_time_limit';break
        requested=[pools[c][index] for c in active]
        boundary('search-round-frozen',proposal_index=index,contexts=active)
        discovered=execute([dict(r,search_plan_sha256=spec['search_plan_sha256']) for r in requested],'search')
        selected=[r for r in discovered if C.captured(r)]
        if selected:
            repeats=[]
            for r in selected:repeats.extend(C.repeat_plan(r,confirm_counts[r['context_id']]))
            packet=f'repeat-plans/round-{index:02d}.json'
            atomic(output/packet,dict(proposal_index=index,rows=repeats,selected=[dict(id=r['id'],sha256=digest(r)) for r in selected],
                rows_count_before=len(rows),rows_sha256_before=sha(output/'rows.jsonl'),frozen_before_repeat_outcomes=True))
            boundary('repeat-plan-frozen',proposal_index=index,path=packet,sha256=sha(output/packet))
            outcomes=execute([dict(r,repeat_plan_file=packet,repeat_plan_sha256=sha(output/packet)) for r in repeats],'repeat')
            for d in selected:
                c=d['context_id'];group=[r for r in outcomes if r['selected_from_id']==d['id']]
                C.require(len(group)==3,'Every selected discovery requires exactly3 fresh repeats')
                confirm_counts[c]+=1
                if all(C.captured(r) for r in group):stop[c]='confirmed'
                elif confirm_counts[c]>=3:stop[c]='confirmation_limit'
        boundary('search-round-complete',proposal_index=index,stopped=copy.deepcopy(stop))
    if report['completion_reason'] is None:
        report['completion_reason']='all_searchable_cases_confirmed' if len(stop)==4 and set(stop.values())=={'confirmed'} else 'finite_candidate_or_confirmation_caps'
    boundary('search-complete',completion_reason=report['completion_reason'],budget_truncated=report['budget_truncated'])
    control_block('controls_post')
    C.require(len([r for r in rows if r['phase']=='preserved'])==29 and len([r for r in rows if r['phase'].startswith('controls_')])==18,'Required preservation/control replays incomplete')
    report.update(status='completed');report['gate']=dict(status='passed',substeps=64,controls_passed=True,required_replays_complete=True,
        search_cap_completed=not report['budget_truncated'],no_policy_promotion=True)
    publish('completed')
    recipes=dict(preserved_parent=[dict(id=r['id'],sha256=digest(r),values=r['values'],q10=r['q10'],q14=r['q14'],bucket_xy_m=r['bucket_xy_m']) for r in loaded['preserved']],
        new_confirmed=[dict(r['confirmed_recipe'],confirmation_groups=r['confirmation_groups']) for r in report['cases'] if r['confirmed_recipe'] is not None],
        policy_unchanged=True,preservation=report['preservation'])
    atomic(output/'recipes.json',recipes)
    report['evidence_sha256']={str(p.relative_to(output)):sha(p) for p in sorted(output.rglob('*')) if p.is_file() and p.name not in ('report.json','evidence-manifest.json')}
    atomic(output/'evidence-manifest.json',report['evidence_sha256']);publish('completed');return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in (*INPUT_KEYS,'output','gpu-lock'):parser.add_argument('--'+key,type=Path,required=True)
    parser.add_argument('--wall-time-limit',type=float,default=7200);parser.add_argument('--queue-timeout',type=float,default=900)
    parser.add_argument('--global-deadline-unix',type=float,required=True);a=parser.parse_args()
    C.require(60<=a.wall_time_limit<=7200 and 0<a.queue_timeout<=900 and not a.output.exists(),'Fresh output and finite caps required')
    C.require(os.environ.get('CUDA_VISIBLE_DEVICES')==GPU_UUID,'Only authorized4090 UUID may execute')
    def deadline(signum,frame):raise TimeoutError('Own acquired-slot hard deadline')
    try:
        with gpu_slot(a.gpu_lock,timeout_s=a.queue_timeout,notify=lambda v:print(json.dumps(v),flush=True)):
            a.acquired_monotonic=time.monotonic();a.effective_worker_limit=min(a.wall_time_limit,a.global_deadline_unix-time.time()-1800)
            if a.effective_worker_limit<=60:raise TimeoutError('CPU audit reserve leaves no worker window')
            budget=a.output.with_name(a.output.name+'-budget.json');C.require(not budget.exists(),'Fresh budget receipt required')
            atomic(budget,dict(acquired_unix_s=time.time(),limit_s=a.wall_time_limit,effective_worker_limit_s=a.effective_worker_limit,
                global_deadline_unix_s=a.global_deadline_unix,audit_reserve_s=1800,gpu_uuid=GPU_UUID))
            signal.signal(signal.SIGALRM,deadline);signal.setitimer(signal.ITIMER_REAL,a.effective_worker_limit-10)
            try:run(a)
            finally:signal.setitimer(signal.ITIMER_REAL,0)
    except Exception as error:
        if a.output.exists():
            path=a.output/'report.json';report=read(path) if path.exists() else {};report.update(status='failed',error=type(error).__name__+': '+str(error));atomic(path,report)
        raise

if __name__=='__main__':main()
