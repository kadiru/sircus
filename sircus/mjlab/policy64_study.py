"""One fixed64 success-bootstrapped PPO worker; no hardware or service changes."""
import argparse
import copy
import datetime
import importlib.metadata
import json
from pathlib import Path
import shutil
import signal
import time
import numpy as np
import torch

from . import policy64_contract as C
from .benchmark_contract import observation, action_from_latent, action_from_residual
from .benchmark_policy import ResidualPolicy
from .corrected_contract import sha
from .corrected_grid import source_manifest
from .gpu_slot import GPU_UUID, gpu_slot, foreign_processes
from .learning_contract import digest
from .learning_policy import update, weight_hash
from .policy64_model import initialized, bootstrap
from .policy64_native import Engine
from .sweep_worker import atomic

def append(path,rows):
    with Path(path).open('a') as stream:
        for r in rows: stream.write(json.dumps(r,allow_nan=False)+'\n')

def control_gate(rows):
    positive=[r for r in rows if r['role']=='positive_control']
    negative=[r for r in rows if r['role']=='negative_control']
    good=(len(rows)==6 and len(positive)==len(negative)==3
          and all(C.captured(r) for r in positive)
          and all(r.get('native_release') and not C.captured(r) and not r.get('whole_ball_entry') for r in negative)
          and all(not r.get('invalid') and not r.get('invalid_physics') and r.get('scalar_score_audit_passed') is True for r in rows))
    return dict(status='passed' if good else 'failed',counts=C.statistics(rows),
                positive_captures=sum(C.captured(r) for r in positive),negative_captures=sum(C.captured(r) for r in negative))

def run(a):
    root=Path(__file__).resolve().parents[2]
    parent,prior,dataset,teachers,bootstrap_summary,bindings,controls=C.load_parent(a.parent_study,a.parent_audit,root)
    ref=C.load_runtime_reference(a.reference,a.assets,root)
    if digest(ref['scene'])!=parent['identity']['scene_sha256'] or ref['asset_hashes']!=parent['identity']['asset_sha256']:
        raise ValueError('Parent reference scene/assets differ')
    if {n:importlib.metadata.version(n) for n in parent['packages']}!=parent['packages']:
        raise ValueError('Exact pinned historical packages required')
    historical=C.read(root/'baselines/mjlab-sweep-rl-benchmark-20261103/splits.json')
    splits,excluded=C.make_splits(parent['splits'],historical)
    a.output.mkdir(parents=True,exist_ok=False)
    for name in ('bootstrap','checkpoints','behavior','after','buffers','predictions'): (a.output/name).mkdir()
    calibration=parent['identity']['calibration'];home=parent['identity']['setup']['home']
    for name in ('calibration.json','verification.json','setup-home.json'): shutil.copyfile(a.parent_study/name,a.output/name)
    atomic(a.output/'prior.json',prior);atomic(a.output/'bootstrap/dataset.json',dataset)
    atomic(a.output/'bootstrap/teachers.json',teachers);atomic(a.output/'bootstrap/source-bindings.json',bindings)
    atomic(a.output/'controls.json',controls);atomic(a.output/'splits.json',splits);atomic(a.output/'exclusions.json',excluded)
    current_identity=copy.deepcopy(parent['identity']);current_identity['source_sha256']=source_manifest(root)
    spec=dict(version=C.VERSION,protocol=C.protocol(),scope=C.SCOPE,scene=ref['scene'],identity=current_identity,
        parent_identity=parent['identity'],runtime_reference=ref['runtime_reference'],
        packages=dict(parent['packages'],scipy=importlib.metadata.version('scipy')),splits=splits,exclusions=excluded,bootstrap_summary=bootstrap_summary,
        source_bindings=bindings,source_sha256=source_manifest(root),prior_sha256=digest(prior),
        bootstrap_dataset_sha256=sha(a.output/'bootstrap/dataset.json'),teachers_sha256=sha(a.output/'bootstrap/teachers.json'),
        controls_sha256=sha(a.output/'controls.json'),evidence_kind='native_fixed64_success_bootstrap_ppo',
        created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    spec['fingerprint']=digest(spec);atomic(a.output/'contract.json',spec)
    started=time.monotonic();rows=[];artifacts={};curves=[];test_comparison=[];test_positions=[]
    report=dict(version=C.VERSION,status='running',phase='preparing',scope=C.SCOPE,contract_fingerprint=spec['fingerprint'],
        protocol=spec['protocol'],bootstrap_summary=bootstrap_summary,train_requests=0,updates=0,elapsed_s=0.,counts={},
        curves=curves,selected=None,test_comparison=test_comparison,test_positions=test_positions,trace_examples=[],
        control_gates={},artifacts={},gate=dict(status='pending'),timing_scope='Reported run time after input verification; not isolated kernel time')
    def publish(phase):
        report.update(phase=phase,elapsed_s=time.monotonic()-started,counts=C.statistics(rows))
        atomic(a.output/'report.json',report)
        print(json.dumps(dict(phase=phase,train_requests=report['train_requests'],updates=report['updates'],
            counts=report['counts'],elapsed_s=round(report['elapsed_s'],1))),flush=True)
    publish('setup')
    engine=Engine(ref,a.assets,a.output/'native',prior=prior,calibration=calibration,home=home)
    def limits():
        if len(rows)>1284 or time.monotonic()-started>a.wall_time_limit-15: raise TimeoutError('Own finite policy64 budget exceeded')
        if shutil.disk_usage(a.output).free<8*1024**3: raise RuntimeError('8 GiB output reserve reached; no evidence deleted')
        if foreign_processes(): raise RuntimeError('Foreign4090 process appeared; leaving it untouched')
    def resolved(context):
        anchor,meta,timing=engine.resolve(context)
        return anchor,meta,timing
    def execute(requests,phase):
        result=[]
        for i in range(0,len(requests),32):
            limits();chunk,timing=engine.evaluate(requests[i:i+32],phase)
            for r in chunk:r['learning_reward']=C.reward(r)
            append(a.output/'rows.jsonl',chunk);rows.extend(chunk);result.extend(chunk)
            append(a.output/'timing.jsonl',[dict(phase=phase,**timing)])
            publish(phase)
        return result
    def control_block(phase):
        requests=[]
        for source,role in zip(controls,('positive_control','negative_control')):
            context=dict(id='control-'+role,split='control',bucket_xy_m=source['bucket_xy_m'])
            anchor,meta,_=resolved(context);obs=observation(context['bucket_xy_m'],anchor).tolist()
            for repeat in range(3):
                requests.append(dict(id=f'{phase}-{role}-r{repeat}',phase=phase,role=role,method='control',
                    context_id=context['id'],bucket_xy_m=context['bucket_xy_m'],values=source['values'],
                    command_values=source['values'],observation=obs,source_teacher_id=anchor['id'],**meta,
                    control_source_id=source['id'],control_source_sha256=digest(source),repeat=repeat,
                    perturbation_mixed_units=[0.]*5))
        result=execute(requests,phase);gate=control_gate(result)
        report['control_gates']['pre' if phase=='controls_pre' else 'post']=gate
        if gate['status']!='passed': raise RuntimeError('Matching64 control block failed: '+phase)
    control_block('controls_pre')
    torch.set_num_threads(2);np.random.seed(C.SEED)
    policy=initialized(C.SEED);optimizer=torch.optim.Adam(policy.parameters(),lr=.0003)
    def snapshot(samples,filename=None):
        path=filename or f'checkpoints/milestone-{samples:04d}.pt'
        torch.save(dict(policy=policy.state_dict(),optimizer=optimizer.state_dict(),samples=samples,seed=C.SEED,
            method='ppo',contract_fingerprint=spec['fingerprint'],weight_sha256=weight_hash(policy),
            actor_sha256=weight_hash(policy.actor)),a.output/path)
        return dict(path=path,sha256=sha(a.output/path),weight_sha256=weight_hash(policy),samples=samples)
    initial=snapshot(0,'checkpoints/initial.pt');report['artifacts']['initial']=initial
    publish('bootstrap')
    fit=bootstrap(policy,dataset,teachers);atomic(a.output/'bootstrap/fit.json',fit)
    artifacts[0]=snapshot(0);report['artifacts']['bc']=artifacts[0]
    def proposals(role,method,samples,artifact):
        frozen=None
        if method in ('bc','ppo'):
            if sha(a.output/artifact['path'])!=artifact['sha256']: raise ValueError('Frozen model bytes changed')
            state=torch.load(a.output/artifact['path'],map_location='cpu',weights_only=True)
            frozen=ResidualPolicy();frozen.load_state_dict(state['policy']);frozen.eval()
            if weight_hash(frozen)!=artifact['weight_sha256']:raise ValueError('Frozen actor state differs')
        predictions,requests=[],[]
        for context in splits[role]:
            anchor,meta,_=resolved(context);obs=observation(context['bucket_xy_m'],anchor)
            teacher_ids=[];latent=None
            if method=='prior': residual=np.zeros(5)
            elif method=='nearest':
                teacher=C.nearest_teacher(context,teachers);residual=np.asarray(teacher['normalized_residual']);teacher_ids=[teacher['id']]
            else:
                with torch.no_grad():latent=frozen.actor(torch.tensor(obs).unsqueeze(0))[0].numpy()
                residual=np.tanh(latent.astype(float))
            values=action_from_residual(anchor['values'],residual)
            prediction=dict(context_id=context['id'],bucket_xy_m=context['bucket_xy_m'],method=method,samples=samples,
                command_values=values.tolist(),observation=obs.tolist(),normalized_residual=residual.tolist(),
                latent_mean=None if latent is None else latent.tolist(),source_training_ids=teacher_ids,
                artifact_sha256=artifact['sha256'],source_teacher_id=anchor['id'],**meta,
                ball_flight_search_at_inference=False,geometry_search_at_inference=True,selection_or_learning=False)
            predictions.append(prediction)
            requests.append(dict(prediction,id=f'{role}-{method}-{samples:04d}-{context["id"]}',role=role,phase=role,
                values=values.tolist(),frozen_artifact_sha256=artifact['sha256'],perturbation_mixed_units=[0.]*5))
        packet=dict(role=role,method=method,samples=samples,artifact=artifact,rows=predictions,
                    selection_or_learning=False,one_prediction_per_layout=True)
        atomic(a.output/f'predictions/{role}-{method}-{samples:04d}.json',packet)
        return requests,packet
    def validation(samples):
        request,_=proposals('validation','ppo',samples,artifacts[samples])
        result=execute(request,'validation');point=dict(samples=samples,validation=C.statistics(result),artifact=artifacts[samples])
        curves.append(point);publish('validation')
    validation(0)
    train_schedule=C.schedule();atomic(a.output/'schedule.json',train_schedule)
    for offset in range(0,1024,32):
        limits();contexts=[splits['train'][i] for i in train_schedule[offset:offset+32]]
        resolved_contexts=[resolved(c) for c in contexts]
        obs=np.asarray([observation(c['bucket_xy_m'],a0) for c,(a0,_,_) in zip(contexts,resolved_contexts)])
        behavior=snapshot(offset,f'behavior/update-{offset//32:04d}.pt')
        x=torch.tensor(obs)
        with torch.no_grad():
            distribution=policy.distribution(x);z=distribution.sample();logp=distribution.log_prob(z).sum(-1)
        requests=[]
        for i,(context,(anchor,meta,_)) in enumerate(zip(contexts,resolved_contexts)):
            values=action_from_latent(anchor['values'],z[i].numpy())
            requests.append(dict(id=f'policy64-train-{offset+i:05d}',role='train',phase='train',method='ppo',seed=C.SEED,
                train_index=offset+i,samples=offset+i+1,context_id=context['id'],bucket_xy_m=context['bucket_xy_m'],
                values=values.tolist(),command_values=values.tolist(),observation=obs[i].tolist(),
                source_teacher_id=anchor['id'],**meta,normalized_residual=np.tanh(z[i].numpy().astype(float)).tolist(),
                latent=z[i].tolist(),old_logp=float(logp[i]),behavior_checkpoint=behavior['path'],
                behavior_checkpoint_sha256=behavior['sha256'],source_training_ids=[],perturbation_mixed_units=[0.]*5))
        result=execute(requests,'train');returns=torch.tensor([r['learning_reward'] for r in result],dtype=torch.float32)
        buffer=f'buffers/update-{offset//32:04d}.npz'
        np.savez_compressed(a.output/buffer,observation=obs,latent=z.numpy(),old_logp=logp.numpy(),reward=returns.numpy())
        changed=update(policy,optimizer,x,z,logp,returns,epochs=6)
        after=snapshot(offset+32,f'after/update-{offset//32:04d}.pt')
        append(a.output/'updates.jsonl',[dict(index=offset//32,train_ids=[r['id'] for r in result],
            buffer=buffer,buffer_sha256=sha(a.output/buffer),behavior=behavior,after_artifact=after,**changed)])
        report.update(train_requests=offset+32,updates=offset//32+1)
        if offset+32 in C.MILESTONES:
            artifacts[offset+32]=snapshot(offset+32);validation(offset+32)
        publish('train')
    best=max(curves,key=lambda p:(p['validation']['captures'],p['validation']['mean_reward'],-p['samples']))
    selected=copy.deepcopy(artifacts[best['samples']]);report['selected']=selected;report['artifacts']['selected']=selected
    atomic(a.output/'selected.json',dict(selected=selected,rule=C.protocol()['selection'],validation=best['validation']))
    for method in ('prior','nearest'):
        path=f'checkpoints/{method}.json';atomic(a.output/path,dict(method=method,contract_fingerprint=spec['fingerprint'],
            teachers_sha256=spec['teachers_sha256'],rule=C.protocol()['nearest'] if method=='nearest' else 'zero residual'))
        report['artifacts'][method]=dict(path=path,sha256=sha(a.output/path),samples=0)
    plans=[]
    for method in C.METHODS:
        artifact=selected if method=='ppo' else artifacts[0] if method=='bc' else report['artifacts'][method]
        requests,packet=proposals('test',method,artifact['samples'],artifact);plans.append((method,artifact,requests,packet))
    atomic(a.output/'test-plan.json',dict(selected=selected,methods=[p[3] for p in plans],frozen_before_outcomes=True))
    publish('test-frozen')
    for method,artifact,requests,_ in plans:
        before=sha(a.output/artifact['path']);result=execute(requests,'test')
        if sha(a.output/artifact['path'])!=before:raise RuntimeError('Frozen test artifact mutated')
        test_comparison.append(dict(method=method,counts=C.statistics(result),samples=artifact['samples'],artifact_sha256=before))
        for r in result:
            test_positions.append(dict(context_id=r['context_id'],bucket_xy_m=r['bucket_xy_m'],method=method,values=r['values'],
                board_bucket_distance_m=float(np.linalg.norm(np.asarray(r['values'][:2])-r['bucket_xy_m'])),
                capture=C.captured(r),clean=C.clean(r),native_release=r['native_release'],invalid=r.get('invalid'),invalid_physics=r.get('invalid_physics')))
        publish('test')
    control_block('controls_post')
    if len(rows)!=1284:raise RuntimeError('Literal1284 requested decisions must remain in ledger')
    examples=[]
    for method in ('bc','ppo'):
        for success in (True,False):
            group=[r for r in rows if r['role']=='test' and r['method']==method and r.get('native_release') and C.captured(r)==success]
            if group:
                r=group[0];examples.append(dict(id=r['id'],phase='test',method=method,substeps=64,
                    values=r['values'],bucket_xy_m=r['bucket_xy_m'],board_center_m=r['board_center_m'],board_normal=r['board_normal'],
                    trajectory_m=r['trajectory_m'],trajectory_dt_s=r['trajectory_dt_s'],capture=C.captured(r),clean=C.clean(r),
                    whole_ball_entry=r['whole_ball_entry'],max_retention_s=r['max_retention_s'],invalid=r.get('invalid'),
                    invalid_physics=r.get('invalid_physics'),scalar_score_audit_passed=r.get('scalar_score_audit_passed'),
                    native_cache_sha256=r['native_cache_sha256'],native_cache_file=r['native_cache_file'],native_cache_row=r['native_cache_row']))
    report['trace_examples']=examples;report['gate']=dict(status='passed',controls_passed=True,complete=True,substeps=64,
        scope='Protocol/controls completion, not a128convergence or minimum success-rate gate')
    report['evidence_sha256']={str(p.relative_to(a.output)):sha(p) for p in sorted(a.output.rglob('*'))
        if p.is_file() and p.name not in ('report.json','evidence-manifest.json')}
    atomic(a.output/'evidence-manifest.json',report['evidence_sha256']);report['status']='completed';publish('completed')
    return report

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('parent-study','parent-audit','reference','assets','output','gpu-lock'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--wall-time-limit',type=float,default=7200);p.add_argument('--queue-timeout',type=float,default=1800)
    a=p.parse_args()
    if not 60<=a.wall_time_limit<=7200:raise ValueError('Bounded worker limit must be60..7200seconds')
    import os
    if os.environ.get('CUDA_VISIBLE_DEVICES')!=GPU_UUID:raise RuntimeError('Only authorized4090UUID may execute')
    def deadline(signum,frame):raise TimeoutError('Own policy64 acquired-slot deadline reached')
    try:
        with gpu_slot(a.gpu_lock,timeout_s=a.queue_timeout,notify=lambda r:print(json.dumps(r),flush=True)):
            atomic(a.output.with_name(a.output.name+'-budget.json'),dict(acquired_unix_s=time.time(),limit_s=a.wall_time_limit))
            signal.signal(signal.SIGALRM,deadline);signal.setitimer(signal.ITIMER_REAL,a.wall_time_limit-10)
            try:run(a)
            finally:signal.setitimer(signal.ITIMER_REAL,0)
    except Exception as error:
        if a.output.exists():
            path=a.output/'report.json';report=C.read(path) if path.exists() else {};report.update(status='failed',error=type(error).__name__+': '+str(error));atomic(path,report)
        raise

if __name__=='__main__':main()
