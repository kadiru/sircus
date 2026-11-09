"""Bounded multi-position search, fresh imitation and two-seed terminal PPO.

All policies use the same explicit expanded geometric prior and native64/v3
engine. Search outcomes may teach only TRAIN; TEST predictions are frozen first.
"""
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
from . import coverage_training_contract as C
from .policy64_contract import load_runtime_reference
from .benchmark_contract import observation, action_from_latent, action_from_residual
from .benchmark_policy import ResidualPolicy
from .corrected_contract import sha
from .corrected_grid import source_manifest
from .gpu_slot import GPU_UUID, gpu_slot, foreign_processes
from .learning_contract import digest
from .learning_policy import update, weight_hash
from .policy64_model import initialized, bootstrap
from .table_training_native import Engine
from .table_training_study import control_gate
from .sweep_worker import atomic

INPUT_KEYS=('current-study','current-audit','case00-study','case00-audit','history-root','reference','assets')
MAX_REQUESTS=2757


def append(path,rows):
    with Path(path).open('a') as stream:
        for row in rows:stream.write(json.dumps(row,allow_nan=False)+'\n')


def teacher_rank(row):
    return (-int(C.captured(row)),-int(C.clean(row)),-float(row['learning_reward']),row['id'])


def select_checkpoint(curves):
    if not curves:raise ValueError('A completed validation milestone is required')
    return min(curves,key=lambda p:(-p['validation']['captures'],-p['validation']['mean_reward'],p['samples'],p['seed']))


def run(a):
    root=Path(__file__).resolve().parents[2]
    started=time.monotonic();acquired=getattr(a,'acquired_monotonic',started)
    elapsed=lambda:time.monotonic()-acquired
    loaded=C.load_inputs(a,root)
    parent,prior,controls=(loaded[k] for k in ('parent','prior','controls'))
    ref=load_runtime_reference(a.reference,a.assets,root)
    if digest(ref['scene'])!=parent['identity']['scene_sha256'] or ref['asset_hashes']!=parent['identity']['asset_sha256']:
        raise ValueError('Parent scene/assets differ')
    if {n:importlib.metadata.version(n) for n in parent['packages']}!=parent['packages']:
        raise ValueError('Exact pinned historical packages required')
    exclusion=C.collect_exclusions(a.history_root,root)
    splits,excluded=C.make_splits(loaded['train_splits'],exclusion['coordinates'])
    a.output.mkdir(parents=True,exist_ok=False)
    for name in ('bootstrap','checkpoints','behavior','after','buffers','predictions','requests'):(a.output/name).mkdir()
    for name in ('calibration.json','verification.json','setup-home.json'):shutil.copyfile(a.current_study/name,a.output/name)
    atomic(a.output/'prior.json',prior);atomic(a.output/'old-prior.json',loaded['old_prior'])
    atomic(a.output/'controls.json',controls);atomic(a.output/'splits.json',splits);atomic(a.output/'exclusions.json',excluded)
    atomic(a.output/'historical-coordinate-bindings.json',exclusion)
    atomic(a.output/'source-bindings.json',loaded['bindings'])
    current_artifacts={}
    for method in ('current_bc','current_ppo'):
        original=loaded['old_artifacts'][method];source=Path(original['path']);path='checkpoints/'+method+'.pt'
        if sha(source)!=original['sha256']:raise ValueError('Current checkpoint bytes changed before copy')
        shutil.copyfile(source,a.output/path)
        if sha(a.output/path)!=original['sha256']:raise ValueError('Current artifact copy changed bytes')
        current_artifacts[method]=dict(path=path,sha256=original['sha256'],weight_sha256=original['weight_sha256'],
            samples=original['samples'],seed=original.get('seed'),historical_relative_path=original.get('relative_path'),
            evaluated_with_shared_expanded_prior=True)
    atomic(a.output/'old-artifacts.json',current_artifacts)
    identity=copy.deepcopy(parent['identity']);identity['source_sha256']=source_manifest(root)
    spec=dict(version=C.VERSION,scope=C.SCOPE,protocol=C.protocol(),scene=ref['scene'],identity=identity,parent_identity=parent['identity'],
        runtime_reference=ref['runtime_reference'],packages=dict(parent['packages'],scipy=importlib.metadata.version('scipy')),
        splits=splits,exclusions=excluded,source_bindings=loaded['bindings'],source_sha256=source_manifest(root),
        prior_sha256=digest(prior),old_prior_sha256=digest(loaded['old_prior']),old_artifacts=current_artifacts,
        controls_sha256=sha(a.output/'controls.json'),historical_coordinate_bindings_sha256=sha(a.output/'historical-coordinate-bindings.json'),
        evidence_kind='native_fixed64_expanded_prior_fresh_train_search_imitation_ppo',
        created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    spec['fingerprint']=digest(spec);atomic(a.output/'contract.json',spec)
    rows=[];curves=[];test_comparison=[];test_positions=[];stop_reasons=[]
    report=dict(version=C.VERSION,scope=C.SCOPE,status='running',phase='preparing',contract_fingerprint=spec['fingerprint'],
        protocol=spec['protocol'],search_requests=0,search_completed_rounds=0,search_partial_round_requests=0,
        train_requests=0,updates=0,elapsed_s=elapsed(),counts={},curves=curves,selected=None,
        test_comparison=test_comparison,test_positions=test_positions,trace_examples=[],control_gates={},
        artifacts=copy.deepcopy(current_artifacts),bootstrap_summary={},per_seed={},budget_reasons=stop_reasons,
        deadlines=dict(worker_limit_s=a.wall_time_limit,training_cut_s=a.training_cut,search_limit_s=a.search_limit,
            effective_worker_limit_s=getattr(a,'effective_worker_limit',a.wall_time_limit),global_deadline_unix_s=getattr(a,'global_deadline_unix',None)),
        inference_scope='All four weights use the same expanded19 geometry-only prior; current weights are not a historical-score rerun',
        gate=dict(status='pending'),timing_scope='Acquired-slot wall time, including CPU setup; not isolated GPU kernel time')
    def publish(phase):
        report.update(phase=phase,elapsed_s=elapsed(),counts=C.statistics(rows),
            stage_counts={p:C.statistics([r for r in rows if r['phase']==p]) for p in
                          ('controls_pre','search','train','validation','test','controls_post')})
        report['search_coverage']=[dict(context_id=context['id'],bucket_xy_m=context['bucket_xy_m'],
            counts=C.statistics([r for r in rows if r['phase']=='search' and r['context_id']==context['id']]),
            has_teacher=any(C.captured(r) for r in rows if r['phase']=='search' and r['context_id']==context['id']))
            for context in splits['train']]
        atomic(a.output/'report.json',report)
        print(json.dumps(dict(phase=phase,search_requests=report['search_requests'],train_requests=report['train_requests'],
            counts=report['counts'],elapsed_s=round(elapsed(),1))),flush=True)
    def boundary(phase):
        append(a.output/'phase-boundaries.jsonl',[dict(phase=phase,acquired_elapsed_s=elapsed(),rows_count=len(rows),
            rows_sha256=sha(a.output/'rows.jsonl') if (a.output/'rows.jsonl').exists() else None)])
    publish('setup')
    engine=Engine(ref,a.assets,a.output/'native',prior=prior,calibration=loaded['calibration'],home=loaded['home'])
    def limits(requests=0):
        if len(rows)+requests>MAX_REQUESTS:raise RuntimeError('Would exceed frozen2757-request maximum')
        if elapsed()>a.wall_time_limit-30:raise TimeoutError('Own finite acquired-slot budget exceeded')
        if shutil.disk_usage(a.output).free<8*1024**3:raise RuntimeError('8GiB reserve reached; all evidence retained')
        if foreign_processes():raise RuntimeError('Foreign4090 workload appeared; leaving it untouched')
    request_number=0
    def execute(requests,phase):
        nonlocal request_number
        result=[]
        for index in range(0,len(requests),32):
            chunk=requests[index:index+32];limits(len(chunk))
            packet=f'requests/batch-{request_number:05d}.json';request_number+=1
            atomic(a.output/packet,dict(phase=phase,rows=chunk,pre_outcome=True))
            began=elapsed();observed,timing=engine.evaluate(chunk,phase);finished=elapsed()
            if [r['id'] for r in observed]!=[r['id'] for r in chunk]:raise RuntimeError('Native order/count changed')
            for row in observed:row.update(learning_reward=C.reward(row),request_packet=packet,request_packet_sha256=sha(a.output/packet))
            append(a.output/'rows.jsonl',observed);rows.extend(observed);result.extend(observed)
            append(a.output/'timing.jsonl',[dict(phase=phase,**timing,acquired_start_s=began,acquired_end_s=finished,
                request_packet=packet,request_packet_sha256=sha(a.output/packet))]);publish(phase)
        return result
    def resolved(context):
        limits();return engine.resolve(context)
    def control_block(phase):
        requests=[]
        for source,role in zip(controls,('direct_control','table_control','negative_control')):
            context=dict(id='control-'+role,split='control',bucket_xy_m=source['bucket_xy_m'])
            anchor,meta,_=resolved(context);obs=observation(context['bucket_xy_m'],anchor).tolist()
            for repeat in range(3):
                requests.append(dict(id=f'coverage-{phase}-{role}-r{repeat}',phase=phase,role=role,method='control',
                    context_id=context['id'],bucket_xy_m=context['bucket_xy_m'],values=source['values'],command_values=source['values'],
                    observation=obs,source_teacher_id=anchor['id'],**meta,control_source_id=source['id'],
                    control_source_sha256=digest(source),repeat=repeat,perturbation_mixed_units=[0.]*5))
        gate=control_gate(execute(requests,phase));report['control_gates']['pre' if phase=='controls_pre' else 'post']=gate
        if gate['status']!='passed':publish('control_failed');raise RuntimeError('Matching64 v3 control block failed: '+phase)
        boundary(phase+'-complete')
    control_block('controls_pre')
    best={};search_rows=[];search_started=elapsed();search_stopped=False
    report['search_started_elapsed_s']=search_started
    for round_index in range(12):
        completed=0
        for offset in range(0,len(splits['train']),32):
            if elapsed()-search_started>=a.search_limit or elapsed()>=a.training_cut:
                stop_reasons.append(dict(stage='search',reason='search_or_training_deadline',elapsed_s=elapsed(),
                    round_index=round_index,completed_in_round=completed,search_elapsed_s=elapsed()-search_started,
                    limit_s=a.search_limit,training_cut_s=a.training_cut));search_stopped=True;break
            requests=[]
            for context in splits['train'][offset:offset+32]:
                anchor,meta,_=resolved(context);obs=observation(context['bucket_xy_m'],anchor)
                residual,search_meta=C.search_residual(context,round_index,best.get(context['id']))
                residual=np.asarray(residual,float);values=action_from_residual(anchor['values'],residual)
                index=round_index*len(splits['train'])+offset+len(requests)
                requests.append(dict(id=f'coverage-search-{index:05d}',role='train',phase='search',method='guided_search',
                    round_index=round_index,train_index=index,context_id=context['id'],bucket_xy_m=context['bucket_xy_m'],
                    values=values.tolist(),command_values=values.tolist(),observation=obs.tolist(),normalized_residual=residual.tolist(),
                    source_teacher_id=anchor['id'],**meta,search_metadata=search_meta,
                    source_training_ids=search_meta.get('source_training_ids',[]),perturbation_mixed_units=[0.]*5))
            result=execute(requests,'search');search_rows.extend(result);completed+=len(result)
            for row in result:
                if row['context_id'] not in best or teacher_rank(row)<teacher_rank(best[row['context_id']]):best[row['context_id']]=row
            report.update(search_requests=len(search_rows),search_partial_round_requests=completed);publish('search')
        if search_stopped:break
        report.update(search_completed_rounds=round_index+1,search_partial_round_requests=0)
    boundary('search-complete');teachers=C.select_teachers(search_rows)
    atomic(a.output/'bootstrap/dataset.json',search_rows);atomic(a.output/'bootstrap/teachers.json',teachers)
    summary=dict(requests=len(search_rows),captures=sum(C.captured(r) for r in search_rows),
        native_releases=sum(bool(r.get('native_release')) for r in search_rows),teacher_count=len(teachers),
        successful_layouts=len(teachers),layouts_requested=len({r['context_id'] for r in search_rows}),
        source_scope='Fresh TRAIN search outcomes only; no historical or held-out outcomes in new actor/critic fitting')
    report['bootstrap_summary']=summary
    atomic(a.output/'bootstrap/source-bindings.json',dict(dataset_sha256=sha(a.output/'bootstrap/dataset.json'),
        teachers_sha256=sha(a.output/'bootstrap/teachers.json'),train_ids=[r['id'] for r in search_rows],summary=summary))
    if not teachers:raise RuntimeError('No fresh TRAIN capture teacher; cannot manufacture an imitation checkpoint')
    torch.set_num_threads(2);policy=initialized(C.BC_SEED)
    def snapshot(model,optimizer,samples,seed,path):
        state=dict(policy=model.state_dict(),samples=samples,seed=seed,method='ppo',contract_fingerprint=spec['fingerprint'],
            weight_sha256=weight_hash(model),actor_sha256=weight_hash(model.actor))
        if optimizer is not None:state['optimizer']=optimizer.state_dict()
        torch.save(state,a.output/path)
        return dict(path=path,sha256=sha(a.output/path),weight_sha256=weight_hash(model),samples=samples,seed=seed)
    report['artifacts']['initial']=snapshot(policy,None,0,C.BC_SEED,'checkpoints/initial.pt')
    publish('bootstrap');fit=bootstrap(policy,search_rows,teachers);atomic(a.output/'bootstrap/fit.json',fit)
    bc=snapshot(policy,None,0,C.BC_SEED,'checkpoints/bc.pt');report['artifacts']['bc']=bc;boundary('bootstrap-complete')
    def proposals(role,method,artifact):
        if sha(a.output/artifact['path'])!=artifact['sha256']:raise ValueError('Frozen checkpoint bytes changed')
        state=torch.load(a.output/artifact['path'],map_location='cpu',weights_only=True)
        frozen=ResidualPolicy();frozen.load_state_dict(state['policy']);frozen.eval()
        if weight_hash(frozen)!=artifact['weight_sha256']:raise ValueError('Frozen model state differs')
        predictions=[];requests=[];samples=artifact['samples'];seed=artifact.get('seed')
        for context in splits[role]:
            anchor,meta,_=resolved(context);obs=observation(context['bucket_xy_m'],anchor)
            with torch.no_grad():latent=frozen.actor(torch.tensor(obs).unsqueeze(0))[0].numpy()
            residual=np.tanh(latent.astype(float));values=action_from_residual(anchor['values'],residual)
            prediction=dict(context_id=context['id'],bucket_xy_m=context['bucket_xy_m'],method=method,samples=samples,seed=seed,
                command_values=values.tolist(),observation=obs.tolist(),normalized_residual=residual.tolist(),latent_mean=latent.tolist(),
                source_training_ids=[],artifact_sha256=artifact['sha256'],source_teacher_id=anchor['id'],**meta,
                ball_flight_search_at_inference=False,geometry_search_at_inference=True,selection_or_learning=False)
            predictions.append(prediction);requests.append(dict(prediction,id=f'coverage-{role}-{method}-seed{seed}-{samples:04d}-{context["id"]}',
                role=role,phase=role,values=values.tolist(),frozen_artifact_sha256=artifact['sha256'],perturbation_mixed_units=[0.]*5))
        packet=dict(role=role,method=method,samples=samples,seed=seed,artifact=artifact,rows=predictions,
                    selection_or_learning=False,one_prediction_per_layout=True)
        atomic(a.output/f'predictions/{role}-{method}-seed{seed}-{samples:04d}.json',packet)
        return requests,packet
    def validation(method,artifact):
        requests,_=proposals('validation',method,artifact);result=execute(requests,'validation')
        curves.append(dict(method=method,seed=artifact['seed'],samples=artifact['samples'],validation=C.statistics(result),artifact=artifact))
        publish('validation')
    validation('bc',bc)
    states={}
    for seed in C.SEEDS:
        model=initialized(seed);model.load_state_dict(copy.deepcopy(policy.state_dict()))
        optimizer=torch.optim.Adam(model.parameters(),lr=.0003);torch.manual_seed(seed)
        states[seed]=dict(model=model,optimizer=optimizer,rng=torch.get_rng_state().clone(),samples=0,schedule=C.schedule(seed))
        atomic(a.output/f'schedule-seed{seed}.json',states[seed]['schedule'])
        report['per_seed'][str(seed)]=dict(samples=0,updates=0,artifacts={})
    stop_training=False
    for offset in range(0,512,32):
        if elapsed()>=a.training_cut:
            stop_reasons.append(dict(stage='train',reason='training_deadline_reserve_test',elapsed_s=elapsed(),samples_per_seed=offset))
            stop_training=True;break
        for seed in C.SEEDS:
            state=states[seed];model=state['model'];optimizer=state['optimizer']
            contexts=[splits['train'][i] for i in state['schedule'][offset:offset+32]]
            context_data=[resolved(c) for c in contexts]
            obs=np.asarray([observation(c['bucket_xy_m'],anchor) for c,(anchor,_,_) in zip(contexts,context_data)])
            behavior=snapshot(model,optimizer,offset,seed,f'behavior/seed{seed}-update-{offset//32:04d}.pt')
            torch.set_rng_state(state['rng']);x=torch.tensor(obs)
            with torch.no_grad():distribution=model.distribution(x);z=distribution.sample();logp=distribution.log_prob(z).sum(-1)
            state['rng']=torch.get_rng_state().clone();requests=[]
            for i,(context,(anchor,meta,_)) in enumerate(zip(contexts,context_data)):
                values=action_from_latent(anchor['values'],z[i].numpy())
                requests.append(dict(id=f'coverage-train-seed{seed}-{offset+i:05d}',role='train',phase='train',method='ppo',seed=seed,
                    train_index=offset+i,samples=offset+i+1,context_id=context['id'],bucket_xy_m=context['bucket_xy_m'],
                    values=values.tolist(),command_values=values.tolist(),observation=obs[i].tolist(),source_teacher_id=anchor['id'],**meta,
                    normalized_residual=np.tanh(z[i].numpy().astype(float)).tolist(),latent=z[i].tolist(),old_logp=float(logp[i]),
                    behavior_checkpoint=behavior['path'],behavior_checkpoint_sha256=behavior['sha256'],
                    source_training_ids=[],perturbation_mixed_units=[0.]*5))
            result=execute(requests,'train');returns=torch.tensor([r['learning_reward'] for r in result],dtype=torch.float32)
            buffer=f'buffers/seed{seed}-update-{offset//32:04d}.npz'
            np.savez_compressed(a.output/buffer,observation=obs,latent=z.numpy(),old_logp=logp.numpy(),reward=returns.numpy())
            changed=update(model,optimizer,x,z,logp,returns,epochs=6)
            after=snapshot(model,optimizer,offset+32,seed,f'after/seed{seed}-update-{offset//32:04d}.pt')
            append(a.output/'updates.jsonl',[dict(seed=seed,index=offset//32,train_ids=[r['id'] for r in result],
                buffer=buffer,buffer_sha256=sha(a.output/buffer),behavior=behavior,after_artifact=after,acquired_elapsed_s=elapsed(),**changed)])
            state['samples']=offset+32;local=report['per_seed'][str(seed)];local.update(samples=offset+32,updates=offset//32+1)
            report.update(train_requests=report['train_requests']+32,updates=report['updates']+1)
            if offset+32 in C.MILESTONES:
                artifact=snapshot(model,optimizer,offset+32,seed,f'checkpoints/seed{seed}-milestone-{offset+32:04d}.pt')
                local['artifacts'][str(offset+32)]=artifact;validation('ppo',artifact)
            publish('train')
        if stop_training:break
    best_point=select_checkpoint(curves);selected=copy.deepcopy(best_point['artifact']);report['selected']=selected
    report['artifacts']['selected']=selected
    atomic(a.output/'selected.json',dict(selected=selected,method=best_point['method'],validation=best_point['validation'],
        rule='Completed validation captures, mean reward, earliest samples, seed; freeze before TEST'))
    plans=[]
    for method in C.METHODS:
        artifact=selected if method=='selected_ppo' else bc if method=='bc' else current_artifacts[method]
        requests,packet=proposals('test',method,artifact);plans.append((method,artifact,requests,packet))
    atomic(a.output/'test-plan.json',dict(selected=selected,methods=[p[3] for p in plans],frozen_before_outcomes=True))
    plan_sha256=sha(a.output/'test-plan.json')
    for _,_,requests,_ in plans:
        for request in requests:request['test_plan_sha256']=plan_sha256
    boundary('test-plan-frozen');publish('test-frozen')
    for method,artifact,requests,_ in plans:
        before=sha(a.output/artifact['path']);result=execute(requests,'test')
        if sha(a.output/artifact['path'])!=before:raise RuntimeError('Frozen test artifact mutated')
        test_comparison.append(dict(method=method,counts=C.statistics(result),samples=artifact['samples'],seed=artifact.get('seed'),artifact_sha256=before))
        for row in result:
            test_positions.append(dict(context_id=row['context_id'],bucket_xy_m=row['bucket_xy_m'],method=method,values=row['values'],
                board_bucket_distance_m=float(np.linalg.norm(np.asarray(row['values'][:2])-row['bucket_xy_m'])),
                capture=C.captured(row),clean=C.clean(row),direct_capture=bool(row.get('direct_capture')),
                table_assisted_capture=bool(row.get('table_assisted_capture')),native_release=row['native_release'],
                invalid=row.get('invalid'),invalid_physics=row.get('invalid_physics'),record_id=row['id']))
        publish('test')
    control_block('controls_post')
    phase_counts={p:sum(r['phase']==p for r in rows) for p in ('search','train','validation','test','controls_pre','controls_post')}
    if phase_counts['test']!=280 or phase_counts['controls_pre']!=9 or phase_counts['controls_post']!=9:
        raise RuntimeError('Complete frozen TEST/control accounting required')
    if phase_counts['train']%32 or phase_counts['train']!=32*report['updates']:
        raise RuntimeError('PPO samples lack completed update accounting')
    full=phase_counts==dict(search=1260,train=1024,validation=175,test=280,controls_pre=9,controls_post=9)
    if not full and not stop_reasons:raise RuntimeError('Partial budget lacks an explicit stopping reason')
    report['gate']=dict(status='passed',controls_passed=True,complete_test=True,full_training_budget=full,substeps=64,
        scope='Complete frozen comparison; not128 convergence, guaranteed capture rate or automatic policy promotion')
    report['training_budget_complete']=full
    report['evidence_sha256']={str(p.relative_to(a.output)):sha(p) for p in sorted(a.output.rglob('*'))
        if p.is_file() and p.name not in ('report.json','evidence-manifest.json')}
    atomic(a.output/'evidence-manifest.json',report['evidence_sha256']);report['status']='completed';publish('completed')
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in (*INPUT_KEYS,'output','gpu-lock'):parser.add_argument('--'+key,type=Path,required=True)
    parser.add_argument('--wall-time-limit',type=float,default=19800)
    parser.add_argument('--training-cut',type=float,default=16200)
    parser.add_argument('--search-limit',type=float,default=7200)
    parser.add_argument('--queue-timeout',type=float,default=1800)
    parser.add_argument('--global-deadline-unix',type=float,default=None);a=parser.parse_args()
    if not (60<=a.search_limit<=7200 and a.search_limit<=a.training_cut<=16200 and a.training_cut<a.wall_time_limit<=19800):
        raise ValueError('Bounded search/training/worker deadlines required')
    if not 0<a.queue_timeout<=1800 or a.output.exists():raise ValueError('Fresh output and finite queue required')
    if os.environ.get('CUDA_VISIBLE_DEVICES')!=GPU_UUID:raise RuntimeError('Only authorized4090 may execute')
    def deadline(signum,frame):raise TimeoutError('Own acquired-slot hard deadline reached')
    try:
        with gpu_slot(a.gpu_lock,timeout_s=a.queue_timeout,notify=lambda value:print(json.dumps(value),flush=True)):
            a.acquired_monotonic=time.monotonic()
            effective_limit=a.wall_time_limit if a.global_deadline_unix is None else min(a.wall_time_limit,a.global_deadline_unix-time.time()-3600)
            if effective_limit<=60:raise TimeoutError('Global deadline leaves no worker budget after audit reserve')
            a.effective_worker_limit=effective_limit
            atomic(a.output.with_name(a.output.name+'-budget.json'),dict(acquired_unix_s=time.time(),limit_s=a.wall_time_limit,
                training_cut_s=a.training_cut,search_limit_s=a.search_limit,effective_worker_limit_s=effective_limit,
                global_deadline_unix_s=a.global_deadline_unix,audit_reserve_s=3600))
            signal.signal(signal.SIGALRM,deadline);signal.setitimer(signal.ITIMER_REAL,effective_limit-10)
            try:run(a)
            finally:signal.setitimer(signal.ITIMER_REAL,0)
    except Exception as error:
        if a.output.exists():
            path=a.output/'report.json';report=json.loads(path.read_text()) if path.exists() else {}
            report.update(status='failed',error=type(error).__name__+': '+str(error));atomic(path,report)
        raise

if __name__=='__main__':main()
