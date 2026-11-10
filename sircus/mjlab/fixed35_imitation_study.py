"""Fixed-grid imitation refit followed by 70 frozen native64 policy requests."""
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
from . import fixed35_imitation_contract as C
from .benchmark_contract import observation, action_from_residual
from .benchmark_policy import ResidualPolicy
from .benchmark_verify import digest, sha, read
from .corrected_grid import source_manifest
from .coverage_training_study import append, share_native_environment
from .gpu_slot import GPU_UUID, gpu_slot, foreign_processes
from .learning_policy import weight_hash
from .policy64_contract import load_runtime_reference
from .policy64_model import bootstrap
from .table_correction_native import MatchedEngine, META
from .fixed35_imitation_native import FixedYawEngine
from .corrected_imitation_study import mirror_control_proofs
from .table_training_study import control_gate
from .sweep_worker import atomic

INPUT_KEYS=C.INPUT_KEYS


def paired_results(positions):
    by={}
    for p in positions:by.setdefault(p['context_id'],{})[p['method']]=p
    complete=[v for v in by.values() if set(v)==set(C.METHODS)]
    gains=sum(v['fixed35_bc']['capture'] and not v['current_bc']['capture'] for v in complete)
    losses=sum(v['current_bc']['capture'] and not v['fixed35_bc']['capture'] for v in complete)
    return dict(paired_positions=len(complete),both_capture=sum(all(r['capture'] for r in v.values()) for v in complete),
        fixed35_only=gains,current_only=losses,both_fail=sum(not any(r['capture'] for r in v.values()) for v in complete),net_captures=gains-losses)


def run(a):
    root=Path(__file__).resolve().parents[2];started=time.monotonic();acquired=getattr(a,'acquired_monotonic',started);elapsed=lambda:time.monotonic()-acquired
    loaded=C.load_inputs(a,root);parent=loaded['parent'];ref=load_runtime_reference(a.reference,a.assets,root)
    C.require(digest(ref['scene'])==parent['identity']['scene_sha256'] and ref['asset_hashes']==parent['identity']['asset_sha256'],'Parent scene/assets differ')
    C.require({n:importlib.metadata.version(n) for n in parent['packages']}==parent['packages'],'Pinned dependencies differ')
    test=C.fixed_grid()
    output=Path(a.output).resolve()
    C.require(not output.exists() and not any(output.is_relative_to(Path(p).resolve()) for p in (root,a.corrected_study,a.yaw_study,a.assets)),'Fresh separate evidence directory required')
    output.mkdir(parents=True)
    for name in ('bootstrap','checkpoints','predictions','requests'):(output/name).mkdir()
    for name in ('calibration.json','verification.json','setup-home.json'):shutil.copyfile(a.corrected_study/name,output/name)
    for name,key in (('prior.json','prior'),('old-prior.json','old_prior'),('controls.json','controls'),('control-replays.json','control_replays')):atomic(output/name,loaded[key])
    atomic(output/'source-bindings.json',loaded['bindings'])
    atomic(output/'splits.json',dict(train=[dict(id=r['context_id'],bucket_xy_m=r['bucket_xy_m'],split='train') for r in loaded['all_teachers']],test=test))
    atomic(output/'bootstrap/teachers.json',loaded['all_teachers']);atomic(output/'bootstrap/original-teachers.json',loaded['teachers'])
    atomic(output/'bootstrap/correction-teachers.json',loaded['corrected_teachers'])
    atomic(output/'bootstrap/source-bindings.json',dict(parent_teacher_ids=[r['id'] for r in loaded['teachers']],
        parent_teacher_sha256={r['id']:digest(r) for r in loaded['teachers']},new_teacher_ids=[r['id'] for r in loaded['corrected_teachers']],
        teacher_file_sha256=sha(output/'bootstrap/teachers.json'),teacher_lineage=loaded['teacher_lineage'],
        lineage=C.protocol()['teacher_semantics'], weighting='one equal-weight teacher per distinct bucket XY' ))
    original=loaded['artifacts']['current_bc'];path='checkpoints/current_bc.pt';shutil.copyfile(original['source_path'],output/path)
    C.require(sha(output/path)==original['sha256'],'Current model bytes changed')
    artifact={k:copy.deepcopy(original[k]) for k in ('sha256','weight_sha256','samples','seed')};artifact['path']=path
    artifacts={'current_bc':artifact}
    spec=dict(version=C.VERSION,scope=C.SCOPE,protocol=C.protocol(),identity=copy.deepcopy(parent['identity']),parent_identity=copy.deepcopy(parent['identity']),scene=ref['scene'],
        packages=parent['packages'],runtime_reference=ref['runtime_reference'],source_sha256=source_manifest(root),source_bindings=loaded['bindings'],
        prior_sha256=digest(loaded['prior']),old_prior_sha256=digest(loaded['old_prior']),splits=read(output/'splits.json'),old_artifacts=copy.deepcopy(artifacts),
        teacher_file_sha256=sha(output/'bootstrap/teachers.json'),release_behavior=C.ROTATION, evaluation_scope=C.protocol()['evaluation_scope'],
        created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    spec['identity']['source_sha256']=spec['source_sha256'];spec['fingerprint']=digest(spec);atomic(output/'contract.json',spec)
    rows=[];comparisons=[];positions=[]
    teacher_corrections=[]
    for r in loaded['corrected_teachers']:
        teacher_corrections.append(dict(context_id=r['context_id'],bucket_xy_m=r['bucket_xy_m'],values=r['values'],source_study='yaw',
            source_record_id=r['source_correction_record_id'],source_record_sha256=r['source_correction_record_sha256'],
            source_report_sha256=loaded['bindings']['yaw']['report_sha256'],confirmation_captures=3,confirmation_requests=3))
    report=dict(version=C.VERSION,status='running',phase='setup',scope=C.SCOPE,contract_fingerprint=spec['fingerprint'],protocol=C.protocol(),
        expected_requests=88,base_teachers=97,new_teachers=2,teacher_count=99,teacher_corrections=teacher_corrections,
        fit=None,artifacts=copy.deepcopy(artifacts),test_comparison=comparisons,test_positions=positions,control_gates={},
        training=True,posthoc_development=False,evaluation_scope=C.protocol()["evaluation_scope"],deadlines=dict(worker_limit_s=a.wall_time_limit,
            effective_worker_limit_s=getattr(a,'effective_worker_limit',a.wall_time_limit),global_deadline_unix_s=getattr(a,'global_deadline_unix',None)),gate=dict(status='pending'))
    def publish(phase):
        report.update(phase=phase,elapsed_s=elapsed(),counts=C.statistics(rows),
            stage_counts={p:C.statistics([r for r in rows if r['phase']==p]) for p in ('controls_pre','test','controls_post')},paired_comparison=paired_results(positions))
        atomic(output/'report.json',report);print(json.dumps(dict(phase=phase,status=report['status'],counts=report['counts'],elapsed_s=elapsed())),flush=True)
    def boundary(phase):append(output/'phase-boundaries.jsonl',[dict(phase=phase,acquired_elapsed_s=elapsed(),rows_count=len(rows),rows_sha256=sha(output/'rows.jsonl') if rows else None)])
    publish('setup');boundary('inputs-frozen')
    engine=FixedYawEngine(ref,a.assets,output/'native-yaw',prior=loaded['prior'],calibration=loaded['calibration'],home=loaded['home'])
    parent_options=read(a.corrected_study/'native/effective-options.json')
    parent_geometry=read(a.corrected_study/'native/native-compiled-geometry.json')
    mirror=mirror_control_proofs(a.corrected_study,output,loaded['control_replays'])
    controls=MatchedEngine(ref,a.assets,output/'native-controls',parent=mirror,sources=loaded['control_replays'],prior=loaded['old_prior'],calibration=loaded['calibration'],home=loaded['home'])
    share_native_environment(engine,controls)
    def limits(n=0):
        C.require(len(rows)+n<=88,'88-slot cap exceeded')
        if elapsed()>getattr(a,'effective_worker_limit',a.wall_time_limit)-30:raise TimeoutError('Finite own acquired budget reached')
        C.require(shutil.disk_usage(output).free>=8*1024**3,'8GiB reserve reached; evidence retained')
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
                requests.append(dict({k:copy.deepcopy(source[k]) for k in META},id=f'fixed35-imitation-{phase}-{role}-r{i}',phase=phase,
                    role=role,method='control',context_id=source['context_id'],bucket_xy_m=source['bucket_xy_m'],values=source['values'],command_values=source['values'],
                    source_record_id=source['id'],source_record_sha256=digest(source),variant='control_exact',adapter_directory='native-controls',
                    control_source_id=original['id'],control_source_sha256=digest(original),repeat=i,perturbation_mixed_units=[0.]*5))
        gate=control_gate(execute(requests,phase,control=True));report['control_gates']['pre' if phase=='controls_pre' else 'post']=gate
        C.require(gate['status']=='passed','Exact original17 control gate failed');boundary(phase+'-complete')
    control_block('controls_pre');limits()
    torch.set_num_threads(2);state=torch.load(output/artifact['path'],map_location='cpu',weights_only=True)
    model=ResidualPolicy();model.load_state_dict(state['policy']);C.require(weight_hash(model)==artifact['weight_sha256'],'Current state weights differ')
    before_critic=weight_hash(model.critic);before_logstd=model.logstd.detach().clone();publish('fit')
    fit=bootstrap(model,loaded['all_teachers'],loaded['all_teachers'],actor_steps=800,critic_steps=0)
    C.require(weight_hash(model.critic)==before_critic and torch.equal(model.logstd.detach(),before_logstd),'Actor-only fit changed critic/logstd')
    path='checkpoints/fixed35_bc.pt';torch.save(dict(policy=model.state_dict(),samples=99,seed=state.get('seed'),method='actor_only_fixed35_bc',
        contract_fingerprint=spec['fingerprint'],weight_sha256=weight_hash(model),actor_sha256=weight_hash(model.actor)),output/path)
    report['artifacts']['fixed35_bc']=dict(path=path,sha256=sha(output/path),weight_sha256=weight_hash(model),samples=99,seed=state.get('seed'))
    fit.update(status='actor_updated',critic_before_sha256=before_critic,critic_after_sha256=weight_hash(model.critic),parent_teacher_count=97,new_teacher_count=2)
    atomic(output/'fit.json',fit);report['fit']=fit;boundary('fit-complete')
    def proposals(method,artifact):
        C.require(sha(output/artifact['path'])==artifact['sha256'],'Frozen model bytes changed')
        saved=torch.load(output/artifact['path'],map_location='cpu',weights_only=True);model=ResidualPolicy();model.load_state_dict(saved['policy']);model.eval()
        C.require(weight_hash(model)==artifact['weight_sha256'],'Frozen policy weights differ')
        predictions=[];requests=[]
        for context in test:
            limits();anchor,meta,_=engine.resolve(context);obs=observation(context['bucket_xy_m'],anchor)
            with torch.no_grad():latent=model.actor(torch.tensor(obs).unsqueeze(0))[0].numpy()
            residual=np.tanh(latent.astype(float));values=action_from_residual(anchor['values'],residual).tolist()
            prediction=dict(context_id=context['id'],bucket_xy_m=context['bucket_xy_m'],method=method,command_values=values,
                observation=obs.tolist(),normalized_residual=residual.tolist(),latent_mean=latent.tolist(),source_training_ids=[],
                source_teacher_id=anchor['id'],**meta,artifact_sha256=artifact['sha256'],ball_flight_search_at_inference=False,
                geometry_search_at_inference=True,selection_or_learning=False)
            predictions.append(prediction);requests.append(dict(prediction,id=f'fixed35-imitation-test-{method}-{context["id"]}',phase='test',role='test',values=values,
                frozen_artifact_sha256=artifact['sha256'],perturbation_mixed_units=[0.]*5))
        packet=dict(method=method,role='test',artifact=artifact,rows=predictions,one_prediction_per_layout=True,selection_or_learning=False)
        atomic(output/f'predictions/test-{method}.json',packet);return requests,packet
    plans=[]
    for method in C.METHODS:
        artifact=report['artifacts'][method];requests,packet=proposals(method,artifact);plans.append((method,artifact,requests,packet))
    atomic(output/'test-plan.json',dict(methods=[p[3] for p in plans],frozen_before_outcomes=True));test_sha=sha(output/'test-plan.json');boundary('test-plan-frozen')
    for method,artifact,requests,_ in plans:
        result=execute([dict(r,test_plan_sha256=test_sha) for r in requests],'test')
        C.require(sha(output/artifact['path'])==artifact['sha256'],'Frozen TEST model changed')
        comparisons.append(dict(method=method,counts=C.statistics(result),artifact_sha256=artifact['sha256']))
        for r in result:positions.append(dict(context_id=r['context_id'],bucket_xy_m=r['bucket_xy_m'],method=method,capture=C.captured(r),clean=C.clean(r),
            native_release=r['native_release'],invalid=r.get('invalid'),invalid_physics=bool(r.get('invalid_physics')),forbidden_contact_steps=r.get('forbidden_contact_steps',0),
            direct_capture=r.get('direct_capture',False),table_assisted_capture=r.get('table_assisted_capture',False),values=r['values'],record_id=r['id']))
        publish('test')
    boundary('test-complete');control_block('controls_post')
    C.require(sum(r['phase']=='test' for r in rows)==70 and len(rows)==88,'Full88-slot comparison required')
    report.update(status='completed');report['gate']=dict(status='passed',substeps=64,controls_passed=True,complete_test=True,no_policy_promotion=True)
    report['evidence_sha256']={str(p.relative_to(output)):sha(p) for p in sorted(output.rglob('*')) if p.is_file() and p.name not in ('report.json','evidence-manifest.json')}
    atomic(output/'evidence-manifest.json',report['evidence_sha256']);publish('completed');return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in (*INPUT_KEYS,'output','gpu-lock'):parser.add_argument('--'+key,type=Path,required=True)
    parser.add_argument('--wall-time-limit',type=float,default=3600);parser.add_argument('--queue-timeout',type=float,default=900)
    parser.add_argument('--global-deadline-unix',type=float,required=True);a=parser.parse_args()
    C.require(60<=a.wall_time_limit<=3600 and 0<a.queue_timeout<=900 and not a.output.exists(),'Fresh output and finite caps required')
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
