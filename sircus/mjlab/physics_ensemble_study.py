"""CPU-fit learned shot physics, then queue bounded fixed64 native evaluation.

Run this file by absolute path. An explicit original native source supplies the
unchanged Engine/scorer/geometry; only the new contract/model are loaded from
this snapshot. Preparation and every frozen prediction precede GPU acquisition.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import shutil
import signal
import sys
import time
from types import SimpleNamespace

import numpy as np

GPU_UUID='GPU-ab1950b0-01dd-0fab-407f-2d17936b27f4'
METHODS=('bc','ppo','ensemble')
MAX_REQUESTS=264


def read(path):return json.loads(Path(path).read_text())


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda:stream.read(1024*1024),b''):h.update(part)
    return h.hexdigest()


def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,allow_nan=False).encode()).hexdigest()


def atomic(path,value):
    path=Path(path);temporary=path.with_name(path.name+'.tmp')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');temporary.replace(path)


def append(path,rows):
    with Path(path).open('a') as stream:
        for row in rows:stream.write(json.dumps(row,allow_nan=False)+'\n')


def vector(value,n):
    a=np.asarray(value,dtype=float)
    if a.shape!=(n,) or not np.isfinite(a).all():raise ValueError('Exact finite vector required')
    return a.tolist()


def captured(row):
    return bool(row.get('native_release') and row.get('route_success') and not row.get('invalid') and row.get('invalid_physics') is False)


def counts(rows):
    cap=sum(captured(r) for r in rows);native=sum(r.get('native_release') is True for r in rows)
    invalid=sum(r.get('native_release') is True and (bool(r.get('invalid')) or bool(r.get('invalid_physics'))) for r in rows)
    return dict(requested=len(rows),native_releases=native,captures=cap,
        clean=sum(captured(r) and r.get('route_quality')==2 for r in rows),
        rejected=len(rows)-native,native_misses=native-cap-invalid,invalid=invalid)


def relative(root,path):
    root=Path(root).resolve();path=Path(path).resolve()
    if not path.is_relative_to(root) or not path.is_file():raise ValueError('Artifact must exist within new study')
    return str(path.relative_to(root))


def load_api(args):
    """Import new modules explicitly; never replace files in the old package."""
    sys.dont_write_bytecode=True
    source=Path(args.native_source).resolve();repository=Path(__file__).resolve().parents[2]
    public=read(repository/'baselines/mjlab-policy64-20261105/summary.json')['provenance']
    if sha(source/'sircus/mjlab/gpu_slot.py')!=public['executed_source_sha256']['sircus/mjlab/gpu_slot.py']:
        raise ValueError('Original scheduler source differs')
    for name,module in tuple(sys.modules.items()):
        filename=getattr(module,'__file__',None)
        if (name=='sircus' or name.startswith('sircus.')) and filename and not Path(filename).resolve().is_relative_to(source):
            raise RuntimeError('Run standalone without another imported sircus source root')
    sys.path.insert(0,str(source));importlib.import_module('sircus.mjlab')
    def new_module(name):
        qualified='sircus.mjlab.'+name;path=repository/'sircus/mjlab'/(''+name+'.py')
        spec=importlib.util.spec_from_file_location(qualified,path)
        module=importlib.util.module_from_spec(spec);sys.modules[qualified]=module;spec.loader.exec_module(module)
        return module
    C=new_module('physics_ensemble_contract')
    bundle=C.authenticate(args.parent_study,args.parent_audit,args.native_source,
                          repository/'baselines/mjlab-policy64-20261105/summary.json')
    parent=bundle['spec']
    if digest(parent['scene'])!=parent['identity']['scene_sha256']:
        raise ValueError('Authenticated native scene digest differs')
    for name,expected in parent['packages'].items():
        if importlib.metadata.version(name)!=expected:raise ValueError('Pinned native package differs: '+name)
    asset_root=Path(args.assets).resolve().parent
    for name,expected in parent['identity']['asset_sha256'].items():
        if Path(name).is_absolute() or '..' in Path(name).parts:raise ValueError('Relative native asset binding required')
        path=(asset_root/name).resolve()
        if not path.is_relative_to(asset_root) or not path.is_file() or sha(path)!=expected:
            raise ValueError('Actual native asset bytes differ: '+name)
    M=new_module('physics_ensemble_model')
    native=importlib.import_module('sircus.mjlab.policy64_native')
    benchmark=importlib.import_module('sircus.mjlab.benchmark_contract')
    policy=importlib.import_module('sircus.mjlab.benchmark_policy')
    admission=importlib.import_module('sircus.mjlab.calibrated_admission')
    scorer=importlib.import_module('sircus.mjlab.calibrated_score')
    return SimpleNamespace(C=C,M=M,Engine=native.Engine,observation=benchmark.observation,
        ResidualPolicy=policy.ResidualPolicy,torch=importlib.import_module('torch'),
        verify_resolution=admission.verify_resolution,verify_connection=admission.verify_connection,
        reward=scorer.reward,gpu=importlib.import_module('sircus.mjlab.gpu_slot')),bundle


def load_actor(api,path):
    state=api.torch.load(path,map_location='cpu',weights_only=True)
    actor=api.ResidualPolicy();actor.load_state_dict(state['policy']);actor.cpu();actor.eval()
    return actor


def actor_values(api,actor,obs,anchor):
    with api.torch.no_grad():latent=actor.actor(api.torch.tensor(obs,dtype=api.torch.float32).unsqueeze(0))[0].numpy()
    residual=np.tanh(latent.astype(float))
    values=np.asarray(anchor['values'],float)+residual*np.asarray([.04,.04,.05,8.,.12])
    return values.tolist(),residual.tolist(),latent.tolist()


def prepare(args,*,api=None,bundle=None):
    """CPU only. Freeze every request/ranking before any native outcome."""
    started=time.monotonic()
    if api is None:api,bundle=load_api(args)
    if bundle is None:raise ValueError('Authenticated input bundle required')
    C,M=api.C,api.M;parent=bundle['spec'];output=Path(args.output)
    if len(bundle['train_rows'])!=1024 or any(r.get('role')!='train' for r in bundle['train_rows']):
        raise ValueError('Only exact1024 current parent TRAIN labels may fit physics')
    reserve=output.parent
    while not reserve.exists():reserve=reserve.parent
    if shutil.disk_usage(reserve).free<8*1024**3:raise RuntimeError('EightGiB CPU preparation reserve required')
    output.mkdir(parents=True,exist_ok=False)
    for name in ('model','predictions'):(output/name).mkdir()
    for name in ('calibration.json','verification.json','setup-home.json','prior.json'):
        shutil.copyfile(Path(args.parent_study)/name,output/name)
    split_record=C.make_splits(parent);splits=split_record['splits'];excluded=split_record['excluded']
    if [len(splits[k]) for k in ('validation','test')]!=[24,60]:raise ValueError('Literal24VAL60TEST split required')
    atomic(output/'splits.json',splits);atomic(output/'exclusions.json',excluded)
    dataset=M.build_dataset(bundle['train_rows'],Path(args.parent_study)/'native',parent['scene'],bundle['calibration']['score_config'])
    atomic(output/'training-dataset.json',dataset);atomic(output/'source-bindings.json',bundle['bindings'])
    protocol=C.protocol();repository=Path(__file__).resolve().parents[2]
    source_names=tuple('sircus/mjlab/'+name+'.py' for name in ('physics_ensemble_contract','physics_ensemble_model','physics_ensemble_study','physics_ensemble_verify'))
    options_path=Path(args.parent_study)/'native/effective-options.json';options=read(options_path)
    if (options.get('version')!='sircus-calibrated-native-warp-options-v1'
        or type(options.get('substeps')) is not int or options['substeps']!=64):
        raise ValueError('Authenticated parent effective64 options required')
    native_options={key:options['native'][key] for key in ('timestep','solver','cone','integrator','iterations','tolerance','gravity')}
    native_reference=dict(scene=parent['scene'],public=dict(effective_native_options=native_options))
    contract=dict(version=C.VERSION,protocol=protocol,scope=C.SCOPE,scene=parent['scene'],identity=parent['identity'],
        parent_contract_fingerprint=parent['fingerprint'],parent_bindings=bundle['bindings'],
        packages=parent['packages'],native_source_sha256=parent['source_sha256'],
        source_sha256={name:sha(repository/name) for name in source_names},
        splits=splits,excluded=excluded,training_rows_sha256=digest(bundle['train_rows']),
        dataset_sha256=sha(output/'training-dataset.json'),substeps=64,maximum_requests=264,
        cpu_fit=True,native_gpu=GPU_UUID,maximum_acquired_s=7200,maximum_queue_s=86400,
        validation_selects_model=False,training_native_requests=0,
        native_reference=dict(parent_options_file='native/effective-options.json',parent_options_sha256=sha(options_path),
            allocated_options=options,public=native_reference['public']))
    contract['fingerprint']=digest(contract);atomic(output/'contract.json',contract)
    report=dict(version=C.VERSION,status='preparing',phase='cpu_fit',scope=C.SCOPE,
        contract_fingerprint=contract['fingerprint'],protocol=protocol,counts=counts([]),control_gates={},
        comparisons=[],test_positions=[],trace_examples=[],artifacts={},gate=dict(status='pending'),
        cpu_fit=True,native_substeps=64,native_requests=0,train_requests=0,validation_selects_model=False,
        source_sha256=contract['source_sha256'],fit_status='fitting',test_comparison=[],prediction_metrics=[],
        phase_counts={p:counts([]) for p in ('controls_pre','validation','test','controls_post')},
        dataset=dict(train_requests=len(bundle['train_rows']),train_native_releases=sum(r.get('native_release') is True for r in bundle['train_rows']),
            train_labels=len(dataset),train_layouts=len({r['context_id'] for r in bundle['train_rows']}),
            source_version=parent.get('version'),source_contract_fingerprint=parent['fingerprint'],
            source_report_sha256=bundle['bindings'].get('parent_report_sha256'),source_audit_sha256=bundle['bindings'].get('parent_audit_sha256'),
            excluded_prior_layouts=len(excluded),split_hashes={'splits.json':sha(output/'splits.json'),'exclusions.json':sha(output/'exclusions.json')},
            target_support=np.asarray([r['mask'] for r in dataset],float).sum(0).astype(int).tolist(),
            labels_scope='Only1024 current parent TRAIN requests; no historical6144, held-out, control or diagnosis labels fitted',
            train_split_scope='Existing35 TRAIN layouts; new24VAL60TEST excluded from all prior layouts'),
        training=dict(status='fitting',ensemble_members=5,fit_device='cpu',fit_steps=600,
            selection_basis='Prespecified final fixed ensemble; no validation tuning',curves=[]))
    def publish(phase):
        report.update(phase=phase,cpu_preparation_elapsed_s=time.monotonic()-started)
        atomic(output/'report.json',report)
        print(json.dumps(dict(status=report['status'],phase=phase,native_requests=report['native_requests'])),flush=True)
    publish('cpu_fit')
    artifact=M.fit(dataset,output/'model',contract['fingerprint'])
    # The model API returns a checkpoint path within the supplied directory.
    checkpoint=Path(artifact['path'])
    if not checkpoint.is_absolute():checkpoint=output/'model'/checkpoint
    ensemble_artifact=dict(artifact,path=relative(output,checkpoint),sha256=sha(checkpoint))
    predictor=M.load(checkpoint);report['artifacts']['ensemble']=ensemble_artifact
    report['training'].update(status='completed',artifact=ensemble_artifact,curves=copy.deepcopy(artifact.get('curves',[])),
        input_features=artifact.get('input_features',[]),target_features=artifact.get('target_features',[]))
    report['fit_status']='model_fit_completed'
    publish('model_fit_completed')
    baselines={}
    for method,key in (('bc','bc'),('ppo','selected')):
        original=bundle['report']['artifacts'][key];src=Path(args.parent_study)/original['path']
        if sha(src)!=original['sha256']:raise ValueError('Authenticated saved baseline checkpoint changed')
        destination=output/'model'/(method+'.pt');shutil.copyfile(src,destination)
        report['artifacts'][method]=dict(path=relative(output,destination),sha256=sha(destination),samples=original['samples'])
        baselines[method]=load_actor(api,destination)
    engine=api.Engine(native_reference,args.assets,output/'native',prior=bundle['prior'],
        calibration=bundle['calibration'],home=bundle['home'],batch=32)
    requests=[];rankings=[]
    def context_meta(context):
        if shutil.disk_usage(output).free<8*1024**3:raise RuntimeError('EightGiB CPU planning reserve reached')
        anchor,meta,timing=engine.resolve(context)
        append(output/'resolution-timing.jsonl',[dict(context_id=context['id'],**timing)])
        obs=vector(api.observation(context['bucket_xy_m'],anchor),9)
        return anchor,meta,obs
    def request(context,method,values,meta,obs,artifact,*,extra=None):
        row=dict(id=f'physics-ensemble-{context["split"]}-{method}-{context["id"]}',role=context['split'],phase=context['split'],
            method=method,context_id=context['id'],bucket_xy_m=vector(context['bucket_xy_m'],2),values=vector(values,5),
            command_values=vector(values,5),observation=obs,source_teacher_id=meta['source_hint']['id'],**copy.deepcopy(meta),
            frozen_artifact_sha256=artifact['sha256'],selection_or_learning=False,
            ball_flight_search_at_inference=False,geometry_search_at_inference=True,
            perturbation_mixed_units=[0.]*5)
        if extra:row.update(copy.deepcopy(extra))
        return row
    def control_requests(phase):
        result=[]
        for source,role in zip(bundle['controls'],('positive_control','negative_control')):
            context=dict(id='ensemble-'+role,split='control',bucket_xy_m=source['bucket_xy_m'])
            anchor,meta,obs=context_meta(context)
            for repeat in range(3):
                row=request(context,'control',source['values'],meta,obs,dict(sha256=digest(source)))
                row.update(id=f'physics-ensemble-{phase}-{role}-r{repeat}',role=role,phase=phase,repeat=repeat,
                    control_source_id=source['id'],control_source_sha256=digest(source))
                result.append(row)
        return result
    before=control_requests('controls_pre');after=control_requests('controls_post')
    publish('cpu_prediction_planning')
    plans={}
    for role in ('validation','test'):
        by_method={m:[] for m in METHODS}
        for context in splits[role]:
            anchor,meta,obs=context_meta(context)
            base={}
            for method in ('bc','ppo'):
                values,residual,latent=actor_values(api,baselines[method],obs,anchor);base[method]=values
                prediction=predictor.predict_inputs([M.inputs(context['bucket_xy_m'],values)])[0]
                by_method[method].append(request(context,method,values,meta,obs,report['artifacts'][method],
                    extra=dict(normalized_residual=residual,latent_mean=latent,model_prediction=prediction,
                        prediction_model_sha256=ensemble_artifact['sha256'])))
            candidates=M.candidates(context,anchor,base['bc'],base['ppo'],bundle['train_rows'])
            if len(candidates)!=16:raise ValueError('Literal16 model candidates required')
            candidates=copy.deepcopy(candidates)
            for candidate in candidates:candidate['values']=vector(candidate['values'],5)
            ranking=M.rank(predictor,context,anchor,candidates)
            order=ranking['order']
            if sorted(order)!=list(range(16)) or any(type(i) is not int for i in order):
                raise ValueError('Model ranking must retain all16 candidate identities')
            resolver=engine.adapter.resolver
            resolution_path=engine.directory/meta['context_resolution_file']
            if sha(resolution_path)!=meta['context_resolution_sha256']:raise ValueError('Preprediction context proof changed')
            resolution=dict(read(resolution_path),file=meta['context_resolution_file'],sha256=meta['context_resolution_sha256'],cache_hit=True)
            api.verify_resolution(resolution,engine.sources,resolver.home)
            checked=[];selected=None
            for index in order[:4]:
                connection=resolver.connect(resolution,candidates[index]['values'])
                api.verify_connection(connection,resolution,resolver.home,candidates[index]['values'])
                checked.append(dict(candidate_index=index,candidate_id=candidates[index].get('id'),
                    passed=bool(connection['passed']),reason=connection.get('reason'),
                    connection_file=connection['file'],connection_sha256=connection['sha256']))
                if connection['passed']:selected=index;break
            if selected is None:selected=order[0]
            log=dict(context_id=context['id'],role=role,bucket_xy_m=context['bucket_xy_m'],observation=obs,
                anchor=anchor,context_metadata=meta,candidates=candidates,ranking=ranking,checks=checked,
                selected_index=selected,selected_values=candidates[selected]['values'],native_queries=0,
                model_candidates=16,maximum_geometry_checks=4,geometry_checks=len(checked),
                all_geometry_checks_rejected=not any(c['passed'] for c in checked),
                selection='First strict-admitted of model top4; if none, preserve modeltop1 rejected request')
            rankings.append(log)
            by_method['ensemble'].append(request(context,'ensemble',candidates[selected]['values'],meta,obs,ensemble_artifact,
                extra=dict(candidate_index=selected,candidate_id=candidates[selected].get('id'),
                    ranking_sha256=digest(log),model_candidates=16,geometry_checks=len(checked),native_planning_queries=0,
                    model_prediction=ranking['predictions'][selected],prediction_model_sha256=ensemble_artifact['sha256'])))
        plans[role]=[r for method in METHODS for r in by_method[method]]
    requests=before+plans['validation']+plans['test']+after
    if len(requests)!=264 or len({r['id'] for r in requests})!=264:raise RuntimeError('Literal264 unique slots required')
    atomic(output/'predictions/rankings.json',rankings)
    packet=dict(version=C.VERSION,contract_fingerprint=contract['fingerprint'],requests=requests,
        ensemble_artifact=ensemble_artifact,rankings_sha256=sha(output/'predictions/rankings.json'),
        frozen_before_any_native_outcome=True,validation_selects_model=False,native_planning_queries=0)
    atomic(output/'prediction-plan.json',packet)
    report.update(status='queued',phase='waiting_for_4090',prediction_plan_sha256=sha(output/'prediction-plan.json'),
        rankings_sha256=packet['rankings_sha256'],planned_requests=264,planned_validation=72,planned_test=180,
        planned_controls=12,geometry_planning_checks=sum(r['geometry_checks'] for r in rankings))
    if hasattr(api.torch,'cuda') and api.torch.cuda.is_initialized():
        raise RuntimeError('CPU preparation unexpectedly initialized CUDA before GPU acquisition')
    publish('waiting_for_4090')
    return dict(output=output,engine=engine,requests=requests,report=report,contract=contract,
        predictor=predictor,prediction_sha256=report['prediction_plan_sha256'],preparation_started=started),api


def execute_prepared(state,api,args):
    """Caller holds the shared4090 lock; all future requests are already fixed."""
    output=state['output'];engine=state['engine'];report=state['report'];rows=[];started=time.monotonic()
    def publish(phase):
        report.update(phase=phase,counts=counts(rows),native_requests=len(rows),acquired_elapsed_s=time.monotonic()-started)
        report['elapsed_s']=time.monotonic()-state['preparation_started']
        report['phase_counts']={p:counts([r for r in rows if r['phase']==p]) for p in ('controls_pre','validation','test','controls_post')}
        atomic(output/'report.json',report)
        print(json.dumps(dict(phase=phase,**report['counts'])),flush=True)
    def limits():
        if len(rows)>264 or time.monotonic()-started>min(args.wall_time_limit,7200)-10:raise TimeoutError('Own finite acquired native budget exceeded')
        if sha(output/'prediction-plan.json')!=state['prediction_sha256']:raise ValueError('Preoutcome frozen predictions changed')
        for artifact in report['artifacts'].values():
            if sha(output/artifact['path'])!=artifact['sha256']:raise ValueError('Frozen inference checkpoint changed')
        if shutil.disk_usage(output).free<8*1024**3:raise RuntimeError('EightGiB output reserve reached')
        if api.gpu.foreign_processes():raise RuntimeError('Foreign4090 workload appeared; leaving it untouched')
    def execute(requests,phase):
        result=[]
        for start in range(0,len(requests),32):
            limits();chunk,timing=engine.evaluate(requests[start:start+32],phase)
            if len(chunk)!=len(requests[start:start+32]):raise RuntimeError('Exact requested trial count changed')
            for r in chunk:r['learning_reward']=api.reward(r)
            append(output/'records.jsonl',chunk);append(output/'timing.jsonl',[dict(phase=phase,**timing)])
            rows.extend(chunk);result.extend(chunk);publish(phase)
        return result
    def gate(block):
        positive=[r for r in block if r['role']=='positive_control'];negative=[r for r in block if r['role']=='negative_control']
        good=(len(block)==6 and len(positive)==len(negative)==3 and all(captured(r) for r in positive)
              and all(r.get('native_release') and not captured(r) and not r.get('whole_ball_entry') for r in negative)
              and all(not r.get('invalid') and not r.get('invalid_physics') and r.get('scalar_score_audit_passed') is True for r in block))
        return dict(status='passed' if good else 'failed',counts=counts(block))
    report.update(status='running',acquired_unix_s=time.time());publish('controls_pre')
    try:
        before=execute(state['requests'][:6],'controls_pre');report['control_gates']['pre']=gate(before)
        if report['control_gates']['pre']['status']!='passed':raise RuntimeError('Initial native controls failed')
        for role in ('validation','test'):
            for method in METHODS:
                requests=[r for r in state['requests'] if r['role']==role and r['method']==method]
                result=execute(requests,role)
                report['comparisons'].append(dict(role=role,method=method,counts=counts(result),
                    artifact_sha256=report['artifacts'][method]['sha256'],
                    mean_reward=float(np.mean([r['learning_reward'] for r in result]))))
                if role=='test':
                    report['test_comparison'].append(copy.deepcopy(report['comparisons'][-1]))
                    for row in result:report['test_positions'].append(dict(context_id=row['context_id'],method=method,
                        bucket_xy_m=row['bucket_xy_m'],values=row['values'],record_id=row['id'],
                        board_bucket_distance_m=float(np.linalg.norm(np.asarray(row['values'][:2])-row['bucket_xy_m'])),
                        admission=bool(row.get('geometry_eligible',row['native_release'])),artifact_sha256=row['frozen_artifact_sha256'],
                        model_prediction=copy.deepcopy(row['model_prediction']),
                        actual=dict(native_release=row['native_release'],capture=captured(row),clean=captured(row) and row['route_quality']==2,
                            invalid=row.get('invalid'),invalid_physics=bool(row.get('invalid_physics')),
                            whole_entry=bool(row.get('whole_ball_entry')),retention_s=float(row.get('max_retention_s',0.)),
                            entry_class='clean_capture' if captured(row) and row['route_quality']==2 else 'captured_nonclean' if captured(row) else 'no_capture')))
                publish(role)
        after=execute(state['requests'][-6:],'controls_post');report['control_gates']['post']=gate(after)
        if report['control_gates']['post']['status']!='passed':raise RuntimeError('Final native controls failed')
        if len(rows)!=264:raise RuntimeError('All264 requested slots must remain in denominator')
        limits()
        # Outcomes from these positions are used for held-out diagnostics only,
        # after every native decision, never for fitting/ranking or tuning.
        evaluated=[r for r in rows if r['role'] in ('validation','test')]
        targets=api.M.build_dataset(evaluated,output/'native',state['contract']['scene'],
            read(output/'calibration.json')['score_config'],require_train=False)
        atomic(output/'evaluation-targets.json',targets)
        byid={r['id']:r for r in evaluated};report['prediction_metrics']=[]
        for role in ('validation','test'):
            group=[t for t in targets if byid[t['id']]['role']==role]
            report['prediction_metrics'].extend(api.M.prediction_metrics(state['predictor'],group,role))
        for method in METHODS:
            for success in (True,False):
                examples=[r for r in rows if r['role']=='test' and r['method']==method
                          and r.get('native_release') and captured(r)==success and r.get('trajectory_m')]
                if examples:
                    r=examples[0]
                    report['trace_examples'].append(dict(id=r['id'],record_id=r['id'],context_id=r['context_id'],method=method,
                        bucket_xy_m=r['bucket_xy_m'],values=r['values'],board_center_m=r['board_center_m'],board_normal=r['board_normal'],
                        position_m=r['trajectory_m'],sample_dt_s=r['trajectory_dt_s'],capture_time_s=r.get('capture_time_s'),
                        capture=captured(r),clean=captured(r) and r['route_quality']==2,whole_ball_entry=r.get('whole_ball_entry'),
                        max_retention_s=r.get('max_retention_s'),invalid=r.get('invalid'),invalid_physics=r.get('invalid_physics'),
                        native_cache_sha256=r['native_cache_sha256'],native_cache_row=r['native_cache_row'],
                        selection_scope='First actual TEST native flight for each method/outcome; measured trace, not model trajectory'))
        report.update(status='completed',gate=dict(status='passed',complete=True,controls_passed=True,
            predictions_frozen_before_any_evaluation=True,no_test_flight_search=True,
            scope='Protocol completion only, not all-table or physical reliability'))
        manifest={str(p.relative_to(output)):sha(p) for p in sorted(output.rglob('*'))
            if p.is_file() and p.name not in ('report.json','evidence-manifest.json')}
        atomic(output/'evidence-manifest.json',manifest);report['evidence_sha256']=manifest;publish('completed')
        return report
    except Exception as error:
        report.update(status='failed',error=type(error).__name__+': '+str(error));publish('failed');raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('parent-study','parent-audit','native-source','assets','output','gpu-lock'):
        parser.add_argument('--'+key,type=Path,required=True)
    parser.add_argument('--wall-time-limit',type=float,default=7200)
    parser.add_argument('--queue-timeout',type=float,default=86400)
    args=parser.parse_args()
    if not 60<=args.wall_time_limit<=7200 or not 0<=args.queue_timeout<=86400:raise ValueError('Bounded queue/acquired limits required')
    if os.environ.get('CUDA_VISIBLE_DEVICES')!=GPU_UUID:raise RuntimeError('Only authorized4090 UUID may execute')
    try:state,api=prepare(args)
    except Exception as error:
        path=args.output/'report.json'
        if path.is_file():
            report=read(path);report.update(status='failed',error=type(error).__name__+': '+str(error));atomic(path,report)
        raise
    def waiting(update):
        state['report'].update(status='queued',phase='waiting_for_4090',queue=update)
        atomic(state['output']/'report.json',state['report']);print(json.dumps(update),flush=True)
    def deadline(signum,frame):raise TimeoutError('Own acquired native deadline reached')
    try:
        with api.gpu.gpu_slot(args.gpu_lock,timeout_s=args.queue_timeout,notify=waiting):
            atomic(args.output.with_name(args.output.name+'-budget.json'),dict(acquired_unix_s=time.time(),limit_s=args.wall_time_limit))
            signal.signal(signal.SIGALRM,deadline);signal.setitimer(signal.ITIMER_REAL,args.wall_time_limit-5)
            try:execute_prepared(state,api,args)
            finally:signal.setitimer(signal.ITIMER_REAL,0)
    except Exception as error:
        state['report'].update(status='failed',error=type(error).__name__+': '+str(error));atomic(state['output']/'report.json',state['report']);raise


if __name__=='__main__':main()
