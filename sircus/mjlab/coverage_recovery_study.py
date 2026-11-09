"""Bounded recovery demonstrations and actor-only imitation; no PPO or deployment."""
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
import numpy as np
import torch
from . import coverage_recovery_contract as C
from .benchmark_contract import observation, action_from_residual
from .benchmark_policy import ResidualPolicy
from .corrected_contract import sha
from .corrected_grid import source_manifest
from .coverage_training_study import append, share_native_environment
from .gpu_slot import GPU_UUID, gpu_slot, foreign_processes
from .learning_contract import digest
from .learning_policy import weight_hash
from .policy64_contract import load_runtime_reference
from .policy64_model import bootstrap
from .table_correction_native import MatchedEngine, META
from .table_training_study import control_gate
from .sweep_worker import atomic

INPUT_KEYS=('parent-study','parent-audit','history-root','reference','assets')


def mirror_control_proofs(parent,output,sources):
    """Only immutable admitted-control proofs are mirrored for the old helper API."""
    origin=Path(parent)/'native-controls';target=Path(output)/'control-parent/native';target.mkdir(parents=True)
    names={'effective-options.json','native-compiled-geometry.json'}
    for row in sources:names.update((row['context_resolution_file'],row['connection_file']))
    manifest=C.read(Path(parent)/'evidence-manifest.json');copied={}
    for name in sorted(names):
        original=C.contained(origin,name);expected=manifest['native-controls/'+name]
        if sha(original)!=expected:raise ValueError('Matched control proof changed')
        destination=target/name;destination.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(original,destination)
        if sha(destination)!=expected:raise ValueError('Matched control copy changed')
        copied[name]=expected
    atomic(Path(output)/'control-parent/mirror.json',copied)
    return target.parent


def run(a):
    root=Path(__file__).resolve().parents[2];started=time.monotonic();acquired=getattr(a,'acquired_monotonic',started)
    elapsed=lambda:time.monotonic()-acquired
    loaded=C.load_inputs(a,root);parent=loaded['parent'];ref=load_runtime_reference(a.reference,a.assets,root)
    if digest(ref['scene'])!=parent['identity']['scene_sha256'] or ref['asset_hashes']!=parent['identity']['asset_sha256']:
        raise ValueError('Parent scene/assets differ')
    if {n:importlib.metadata.version(n) for n in parent['packages']}!=parent['packages']:raise ValueError('Pinned packages differ')
    excluded=C.collect_exclusions(a.history_root,root);test=C.test_split(excluded['coordinates'])
    plan=C.make_plan(loaded['cases'],loaded['rows'])
    a.output.mkdir(parents=True,exist_ok=False)
    for name in ('bootstrap','checkpoints','predictions','requests'):(a.output/name).mkdir()
    for name in ('calibration.json','verification.json','setup-home.json'):shutil.copyfile(a.parent_study/name,a.output/name)
    for name,key in (('prior.json','prior'),('old-prior.json','old_prior'),('controls.json','controls'),('control-replays.json','control_replays')):
        atomic(a.output/name,loaded[key])
    atomic(a.output/'source-bindings.json',loaded['bindings']);atomic(a.output/'historical-coordinate-bindings.json',excluded)
    atomic(a.output/'splits.json',dict(development=[dict(id=r['context_id'],bucket_xy_m=r['bucket_xy_m'],split='train',original_role='test') for r in loaded['cases']],test=test))
    atomic(a.output/'blocked.json',[dict(id=r['id'],context_id=r['context_id'],bucket_xy_m=r['bucket_xy_m'],reason=r['invalid'],parent_row_sha256=digest(r)) for r in loaded['blocked']])
    atomic(a.output/'search-plan.json',dict(rows=plan,frozen_before_outcomes=True))
    artifacts={}
    for method,original in loaded['artifacts'].items():
        path='checkpoints/'+method+'.pt';shutil.copyfile(original['source_path'],a.output/path)
        if sha(a.output/path)!=original['sha256']:raise ValueError('Frozen parent weights changed')
        artifacts[method]=dict(path=path,sha256=original['sha256'],weight_sha256=original['weight_sha256'],samples=original['samples'],seed=original.get('seed'))
    spec=dict(version=C.VERSION,scope=C.SCOPE,protocol=C.protocol(),identity=copy.deepcopy(parent['identity']),
        parent_identity=parent['identity'],scene=ref['scene'],packages=parent['packages'],runtime_reference=ref['runtime_reference'],
        source_sha256=source_manifest(root),source_bindings=loaded['bindings'],prior_sha256=digest(loaded['prior']),
        old_prior_sha256=digest(loaded['old_prior']),splits=C.read(a.output/'splits.json'),old_artifacts=artifacts,
        plans=dict(search_sha256=sha(a.output/'search-plan.json')),created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    spec['identity']['source_sha256']=spec['source_sha256'];spec['fingerprint']=digest(spec);atomic(a.output/'contract.json',spec)
    rows=[];reasons=[];search=[];repeats=[];selected=[];confirmed=[];comparisons=[];positions=[]
    report=dict(version=C.VERSION,status='running',phase='setup',scope=C.SCOPE,contract_fingerprint=spec['fingerprint'],protocol=C.protocol(),
        search_requests=0,search_completed_rounds=0,search_partial_round_requests=0,repeat_requests=0,selected=selected,
        confirmed_teachers=0,base_teachers=68,fit=None,artifacts=copy.deepcopy(artifacts),test_comparison=comparisons,
        test_positions=positions,budget_reasons=reasons,control_gates={},blocked=C.read(a.output/'blocked.json'),
        deadlines=dict(worker_limit_s=a.wall_time_limit,search_limit_s=a.search_limit,repeat_cut_s=a.repeat_cut,
            effective_worker_limit_s=getattr(a,'effective_worker_limit',a.wall_time_limit),global_deadline_unix_s=getattr(a,'global_deadline_unix',None)),gate=dict(status='pending'))
    def publish(phase):
        report.update(phase=phase,elapsed_s=elapsed(),counts=C.statistics(rows),stage_counts={p:C.statistics([r for r in rows if r['phase']==p]) for p in ('controls_pre','search','repeat','test','controls_post')})
        report['search_coverage']=[dict(context_id=c['context_id'],bucket_xy_m=c['bucket_xy_m'],counts=C.statistics([r for r in search if r['context_id']==c['context_id']]),
            confirmed=any(t['context_id']==c['context_id'] for t in confirmed)) for c in loaded['cases']]
        atomic(a.output/'report.json',report);print(json.dumps(dict(phase=phase,counts=report['counts'],elapsed_s=elapsed())),flush=True)
    def boundary(phase):append(a.output/'phase-boundaries.jsonl',[dict(phase=phase,acquired_elapsed_s=elapsed(),rows_count=len(rows),rows_sha256=sha(a.output/'rows.jsonl') if rows else None)])
    publish('setup')
    case_ids={r['context_id'] for r in loaded['cases']}
    source_rows=[r for r in loaded['rows'] if r.get('phase')=='test' and r['context_id'] in case_ids and (r['method']=='selected_ppo' or C.captured(r))]
    hydrated=MatchedEngine(ref,a.assets,a.output/'native',parent=a.parent_study,sources=source_rows,prior=loaded['prior'],calibration=loaded['calibration'],home=loaded['home'])
    engine=hydrated.engine
    mirror=mirror_control_proofs(a.parent_study,a.output,loaded['control_replays'])
    controls=MatchedEngine(ref,a.assets,a.output/'native-controls',parent=mirror,sources=loaded['control_replays'],prior=loaded['old_prior'],calibration=loaded['calibration'],home=loaded['home'])
    share_native_environment(engine,controls)
    def limits(n=0):
        if len(rows)+n>1000:raise RuntimeError('Frozen1000-request cap exceeded')
        if elapsed()>a.wall_time_limit-30:raise TimeoutError('Own acquired-slot budget exceeded')
        if shutil.disk_usage(a.output).free<8*1024**3:raise RuntimeError('8GiB reserve reached; all evidence retained')
        if foreign_processes():raise RuntimeError('Foreign4090 workload appeared; left untouched')
    number=0
    def execute(requests,phase,*,control=False):
        nonlocal number
        result=[]
        for offset in range(0,len(requests),32):
            chunk=requests[offset:offset+32];limits(len(chunk));packet=f'requests/batch-{number:05d}.json';number+=1
            atomic(a.output/packet,dict(phase=phase,rows=chunk,pre_outcome=True));start=elapsed()
            observed,timing=(controls if control else engine).evaluate(chunk,phase)
            if [r['id'] for r in observed]!=[r['id'] for r in chunk]:raise RuntimeError('Requested order changed')
            if engine.adapter.env is not None and (engine.adapter.env.effective_options!=hydrated.parent_options or engine.adapter.env.compiled_geometry!=hydrated.parent_geometry):
                raise RuntimeError('Pinned native options/geometry changed')
            for r in observed:r.update(learning_reward=C.reward(r),request_packet=packet,request_packet_sha256=sha(a.output/packet),adapter_directory='native-controls' if control else 'native')
            append(a.output/'rows.jsonl',observed);rows.extend(observed);result.extend(observed)
            append(a.output/'timing.jsonl',[dict(phase=phase,**timing,acquired_start_s=start,acquired_end_s=elapsed(),request_packet=packet,request_packet_sha256=sha(a.output/packet))]);publish(phase)
        return result
    def control_block(phase):
        requests=[]
        for original,source,role in zip(loaded['controls'],loaded['control_replays'],('direct_control','table_control','negative_control')):
            if source['role']!=role or source['values']!=original['values'] or source['bucket_xy_m']!=original['bucket_xy_m']:
                raise ValueError('Exact original control identity/inputs changed')
            for i in range(3):
                requests.append(dict({k:copy.deepcopy(source[k]) for k in META},id=f'recovery-{phase}-{source["role"]}-r{i}',phase=phase,
                    role=source['role'],method='control',context_id=source['context_id'],bucket_xy_m=source['bucket_xy_m'],values=source['values'],command_values=source['values'],
                    source_record_id=source['id'],source_record_sha256=digest(source),variant='control_exact',adapter_directory='native-controls',
                    control_source_id=original['id'],control_source_sha256=digest(original),repeat=i,perturbation_mixed_units=[0.]*5))
        gate=control_gate(execute(requests,phase,control=True));report['control_gates']['pre' if phase=='controls_pre' else 'post']=gate
        if gate['status']!='passed':publish('control_failed');raise RuntimeError('Matching original17 control block failed')
        boundary(phase+'-complete')
    control_block('controls_pre');search_started=elapsed();report['search_started_elapsed_s']=search_started
    search_sha=sha(a.output/'search-plan.json')
    for offset in range(0,len(plan),32):
        if elapsed()>=a.search_limit:
            reasons.append(dict(stage='search',reason='search_deadline',completed=len(search),elapsed_s=elapsed(),search_elapsed_s=elapsed()-search_started));break
        chunk=[dict(r,search_plan_sha256=search_sha) for r in plan[offset:offset+32]]
        search.extend(execute(chunk,'search'));report.update(search_requests=len(search),search_completed_rounds=len(search)//39,search_partial_round_requests=len(search)%39);publish('search')
    boundary('search-complete');selected[:],repeat_requests=C.repeat_plan(search,loaded['cases'])
    atomic(a.output/'repeat-plan.json',dict(selected=selected,rows=repeat_requests,frozen_before_outcomes=True));repeat_sha=sha(a.output/'repeat-plan.json')
    for offset in range(0,len(repeat_requests),2):
        if elapsed()>=a.repeat_cut:
            reasons.append(dict(stage='repeat',reason='repeat_deadline',completed=len(repeats),planned=len(repeat_requests),elapsed_s=elapsed()));break
        actual=execute([dict(r,repeat_plan_sha256=repeat_sha) for r in repeat_requests[offset:offset+2]],'repeat')
        for r in actual:
            source=next(x for x in search if x['id']==r['source_discovery_id'])
            if any(r.get(k)!=source.get(k) for k in ('q10','q14','connection_file','connection_sha256')):raise RuntimeError('Nominal repeat changed admitted branch')
        repeats.extend(actual);report['repeat_requests']=len(repeats);publish('repeat')
    confirmed[:]=C.confirmed_teachers(search,repeats,selected);report['confirmed_teachers']=len(confirmed);boundary('repeat-complete')
    teachers=sorted(copy.deepcopy(loaded['teachers'])+confirmed,key=lambda r:r['context_id'])
    atomic(a.output/'bootstrap/teachers.json',teachers);atomic(a.output/'bootstrap/recovery-teachers.json',confirmed)
    atomic(a.output/'bootstrap/source-bindings.json',dict(parent_teacher_ids=[r['id'] for r in loaded['teachers']],
        parent_teacher_sha256={r['id']:digest(r) for r in loaded['teachers']},new_teacher_ids=[r['id'] for r in confirmed],
        teacher_file_sha256=sha(a.output/'bootstrap/teachers.json'),repeat_plan_sha256=repeat_sha,
        lineage='Parent68 captured TRAIN teachers plus only new2/2-confirmed development discoveries; no new TEST or validation outcomes'))
    torch.set_num_threads(2);state=torch.load(a.output/artifacts['parent_bc']['path'],map_location='cpu',weights_only=True)
    model=ResidualPolicy();model.load_state_dict(state['policy']);before_critic=weight_hash(model.critic);before_logstd=model.logstd.detach().clone()
    publish('fit')
    path='checkpoints/recovery_bc.pt'
    if confirmed:
        fit=bootstrap(model,teachers,teachers,actor_steps=800,critic_steps=0)
        if weight_hash(model.critic)!=before_critic or not torch.equal(model.logstd.detach(),before_logstd):raise RuntimeError('Actor-only fitting changed critic/logstd')
        torch.save(dict(policy=model.state_dict(),samples=len(teachers),seed=state.get('seed'),method='actor_only_recovery_bc',
            contract_fingerprint=spec['fingerprint'],weight_sha256=weight_hash(model),actor_sha256=weight_hash(model.actor)),a.output/path)
        artifact=dict(path=path,sha256=sha(a.output/path),weight_sha256=weight_hash(model),samples=len(teachers),seed=state.get('seed'))
        fit['status']='actor_updated'
    else:
        shutil.copyfile(a.output/artifacts['parent_bc']['path'],a.output/path);artifact=dict(artifacts['parent_bc'],path=path)
        fit=dict(status='no_confirmed_new_teacher',actor_steps=0,critic_steps=0,before=weight_hash(model),after=weight_hash(model),byte_identical_parent=True)
    fit.update(critic_before_sha256=before_critic,critic_after_sha256=weight_hash(model.critic),parent_teacher_count=68,new_teacher_count=len(confirmed))
    atomic(a.output/'fit.json',fit);report['fit']=fit;report['artifacts']['recovery_bc']=artifact;boundary('fit-complete')
    def proposals(method,artifact):
        if sha(a.output/artifact['path'])!=artifact['sha256']:raise ValueError('Frozen model file changed')
        saved=torch.load(a.output/artifact['path'],map_location='cpu',weights_only=True);model=ResidualPolicy();model.load_state_dict(saved['policy']);model.eval()
        if weight_hash(model)!=artifact['weight_sha256']:raise ValueError('Frozen weights differ')
        predictions=[];requests=[]
        for context in test:
            limits();anchor,meta,_=engine.resolve(context);obs=observation(context['bucket_xy_m'],anchor)
            with torch.no_grad():latent=model.actor(torch.tensor(obs).unsqueeze(0))[0].numpy()
            residual=np.tanh(latent.astype(float));values=action_from_residual(anchor['values'],residual).tolist()
            prediction=dict(context_id=context['id'],bucket_xy_m=context['bucket_xy_m'],method=method,command_values=values,
                observation=obs.tolist(),normalized_residual=residual.tolist(),latent_mean=latent.tolist(),source_training_ids=[],
                source_teacher_id=anchor['id'],**meta,artifact_sha256=artifact['sha256'],ball_flight_search_at_inference=False,
                geometry_search_at_inference=True,selection_or_learning=False)
            predictions.append(prediction);requests.append(dict(prediction,id=f'recovery-test-{method}-{context["id"]}',phase='test',role='test',values=values,
                frozen_artifact_sha256=artifact['sha256'],perturbation_mixed_units=[0.]*5))
        packet=dict(method=method,role='test',artifact=artifact,rows=predictions,one_prediction_per_layout=True,selection_or_learning=False)
        atomic(a.output/f'predictions/test-{method}.json',packet);return requests,packet
    plans=[]
    for method in C.METHODS:
        artifact=report['artifacts'][method];requests,packet=proposals(method,artifact);plans.append((method,artifact,requests,packet))
    atomic(a.output/'test-plan.json',dict(methods=[p[3] for p in plans],frozen_before_outcomes=True));test_sha=sha(a.output/'test-plan.json');boundary('test-plan-frozen')
    for method,artifact,requests,_ in plans:
        result=execute([dict(r,test_plan_sha256=test_sha) for r in requests],'test')
        if sha(a.output/artifact['path'])!=artifact['sha256']:raise RuntimeError('Frozen TEST model mutated')
        comparisons.append(dict(method=method,counts=C.statistics(result),artifact_sha256=artifact['sha256']))
        for r in result:positions.append(dict(context_id=r['context_id'],bucket_xy_m=r['bucket_xy_m'],method=method,
            capture=C.captured(r),clean=C.clean(r),native_release=r['native_release'],invalid=r.get('invalid'),
            forbidden_contact_steps=r.get('forbidden_contact_steps',0),values=r['values'],record_id=r['id']))
        publish('test')
    control_block('controls_post')
    if sum(r['phase']=='test' for r in rows)!=280 or sum(r['phase'].startswith('controls_') for r in rows)!=18:raise RuntimeError('Full frozen comparison and controls required')
    report.update(status='completed',search_budget_complete=len(search)==624,repeat_budget_complete=len(repeats)==len(repeat_requests))
    report['gate']=dict(status='passed',substeps=64,controls_passed=True,complete_test=True,no_policy_promotion=True)
    report['evidence_sha256']={str(p.relative_to(a.output)):sha(p) for p in sorted(a.output.rglob('*')) if p.is_file() and p.name not in ('report.json','evidence-manifest.json')}
    atomic(a.output/'evidence-manifest.json',report['evidence_sha256']);publish('completed');return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in (*INPUT_KEYS,'output','gpu-lock'):parser.add_argument('--'+key,type=Path,required=True)
    parser.add_argument('--wall-time-limit',type=float,default=7200);parser.add_argument('--search-limit',type=float,default=3600)
    parser.add_argument('--repeat-cut',type=float,default=4500);parser.add_argument('--queue-timeout',type=float,default=900)
    parser.add_argument('--global-deadline-unix',type=float,default=None);a=parser.parse_args()
    if not (60<=a.search_limit<=3600 and a.search_limit<a.repeat_cut<=4500 and a.repeat_cut<a.wall_time_limit<=7200):raise ValueError('Bounded ordered worker caps required')
    if not 0<a.queue_timeout<=900 or a.output.exists():raise ValueError('Fresh output and finite queue required')
    if os.environ.get('CUDA_VISIBLE_DEVICES')!=GPU_UUID:raise RuntimeError('Only authorized4090 may execute')
    def deadline(signum,frame):raise TimeoutError('Own acquired-slot hard deadline')
    try:
        with gpu_slot(a.gpu_lock,timeout_s=a.queue_timeout,notify=lambda v:print(json.dumps(v),flush=True)):
            a.acquired_monotonic=time.monotonic();effective=a.wall_time_limit if a.global_deadline_unix is None else min(a.wall_time_limit,a.global_deadline_unix-time.time()-1800)
            if effective<=60:raise TimeoutError('No own worker budget remains after audit reserve')
            a.effective_worker_limit=effective
            atomic(a.output.with_name(a.output.name+'-budget.json'),dict(acquired_unix_s=time.time(),limit_s=a.wall_time_limit,
                effective_worker_limit_s=effective,global_deadline_unix_s=a.global_deadline_unix,audit_reserve_s=1800))
            signal.signal(signal.SIGALRM,deadline);signal.setitimer(signal.ITIMER_REAL,effective-10)
            try:run(a)
            finally:signal.setitimer(signal.ITIMER_REAL,0)
    except Exception as error:
        if a.output.exists():
            path=a.output/'report.json';report=C.read(path) if path.exists() else {};report.update(status='failed',error=type(error).__name__+': '+str(error));atomic(path,report)
        raise

if __name__=='__main__':main()
