"""Frozen 112-request arm-clearance ablation; unchanged native64 engine and models.

Run by absolute path. CPU preparation freezes all decisions before acquiring the
shared 4090 slot. Never imports another engine or changes the interactive solver.
"""
import argparse
import copy
import importlib.util
import json
import os
from pathlib import Path
import shutil
import signal
import sys
import time

import numpy as np

VERSION='sircus-arm-clearance-study-v1'
METHODS=('consistent','arm_clearance')
GPU_UUID='GPU-ab1950b0-01dd-0fab-407f-2d17936b27f4'
MAX_REQUESTS=112
SEED=20261808
SCOPE=('Frozen-model paired arm-clearance filter at 50 fresh bucket positions, fixed64 only. '
       'All16 proposals, top4-geometry budgets, model weights and flight ranking remain fixed. '
       'Each admitted candidate must also pass the frozen predicted post-rebound arm-clearance rule. '
       'If no checked candidate passes, the new selector abstains; the request stays in the denominator. '
       'No fitting, native flight search, noise,128 convergence, driven transit or physical validation. '
       'Negative aperture remains a proxy; original capture and route scoring is unchanged.')


def read(path):return json.loads(Path(path).read_text())
def sha(path):
    import hashlib
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    return h.hexdigest()
def digest(value):
    import hashlib
    return hashlib.sha256(json.dumps(value,sort_keys=True,allow_nan=False).encode()).hexdigest()
def atomic(path,value):
    path=Path(path);tmp=path.with_name(path.name+'.tmp')
    tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');tmp.replace(path)
def append(path,rows):
    with Path(path).open('a') as f:
        for row in rows:f.write(json.dumps(row,allow_nan=False)+'\n')


def fresh_splits(old):
    excluded={tuple(map(float,xy)) for xy in old['excluded']}
    for contexts in old['splits'].values():excluded.update(tuple(map(float,c['bucket_xy_m'])) for c in contexts)
    used=np.array(sorted(excluded),float);rng=np.random.default_rng(SEED);contexts=[]
    for _ in range(100000):
        if len(contexts)==50:break
        xy=rng.uniform([.1,-.65],[1.24,.25])
        if len(used) and np.linalg.norm(used-xy,axis=1).min()<.01:continue
        contexts.append(dict(id=f'arm-clearance-test-{len(contexts):02d}',split='test',bucket_xy_m=xy.tolist()))
        used=np.vstack((used,xy))
    if len(contexts)!=50:raise ValueError('Could not freeze50 disjoint layouts')
    return dict(test=contexts),[list(x) for x in sorted(excluded)]


def protocol():
    return dict(substeps=64,methods=list(METHODS),test_layouts=50,test_seed=SEED,
        maximum_fresh_requests=112,control_requests=12,new_train_requests=0,
        minimum_split_distance_m=.01,no_test_flight_search=True,candidates=16,
        maximum_geometry_checks=4,noise=False,model_weights_unchanged=True,
        no_hard_aperture_veto=True,post_bounce_robot_clearance_added=True,
        changed_candidate_indices=[],unchanged_candidate_count=16,
        clearance_rejection_is_abstention=True,clearance_margin_m=.002,
        clearance_horizon_s=1.5,clearance_members=5)


def overlay(repository,name):
    qualified='sircus.mjlab.'+name;path=repository/'sircus/mjlab'/(name+'.py')
    spec=importlib.util.spec_from_file_location(qualified,path);m=importlib.util.module_from_spec(spec)
    sys.modules[qualified]=m;spec.loader.exec_module(m);return m


def load_dependencies(args):
    repository=Path(__file__).resolve().parents[2]
    old=read(Path(args.ensemble_study)/'contract.json'); unsigned=dict(old); fp=unsigned.pop('fingerprint')
    if digest(unsigned)!=fp or fp!='90a7d5b7e3d1fb1e738a36a7cf30930116516da0d6d6435c06b98eb97be8c376':
        raise ValueError('Exact original ensemble contract required before loading')
    for name,expected in old['source_sha256'].items():
        if sha(repository/name)!=expected:raise ValueError('Borrowed original source changed: '+name)
    # Original loader authenticates native source/packages/assets and parent data.
    loader_spec=importlib.util.spec_from_file_location('_frozen_ensemble_loader',repository/'sircus/mjlab/physics_ensemble_study.py')
    loader=importlib.util.module_from_spec(loader_spec);loader_spec.loader.exec_module(loader)
    api,bundle=loader.load_api(args)
    curator=overlay(repository,'curate_physics_ensemble')
    selector=overlay(repository,'consistent_flight_selector')
    return repository,loader,api,bundle,curator,selector


def prepare(args):
    started=time.monotonic();repository,S,api,bundle,C,selector=load_dependencies(args)
    old_study,old_report,old_spec,_,_,old_packet,old_rows=C.verified_inputs(args.ensemble_study,args.ensemble_audit)
    overlay(repository,'ensemble_flight_diagnosis')
    D=overlay(repository,'consistent_flight_diagnosis'); G=overlay(repository,'arm_clearance_selector')
    F=overlay(repository,'gap_flight_diagnosis')
    reference,reference_report,reference_spec,_,_,_,_=D.verify_inputs(args.consistent_study,args.consistent_audit)
    previous,_,previous_spec,_,_,_,_=F.verify_inputs(args.gap_study,args.gap_audit)
    api.torch.set_num_threads(2)
    output=Path(args.output).resolve()
    if output.exists() or any(output.is_relative_to(Path(root).resolve()) for root in (old_study,reference,previous,args.parent_study,args.native_source,args.assets,repository)):
        raise ValueError('New study namespace outside all immutable inputs required')
    output.parent.mkdir(parents=True,exist_ok=True)
    if shutil.disk_usage(output.parent).free<8*1024**3:raise RuntimeError('EightGiB reserve required')
    output.mkdir();(output/'model').mkdir()
    for name in ('calibration.json','verification.json','setup-home.json','prior.json'):
        shutil.copyfile(Path(args.parent_study)/name,output/name)
    calibration=read(output/'calibration.json')['score_config']
    artifacts={}
    for name,expected in C.ARTIFACT_SHA.items():
        original=old_report['artifacts'][name]
        src=old_study/original['path'];dst=output/'model'/(name+'.pt')
        if sha(src)!=expected:raise ValueError('Frozen checkpoint changed: '+name)
        shutil.copyfile(src,dst);artifacts[name]=dict(path='model/'+dst.name,sha256=expected)
    artifacts['consistent']=copy.deepcopy(artifacts['ensemble'])
    artifacts['arm_clearance']=copy.deepcopy(artifacts['ensemble'])
    predictor=api.M.load(output/artifacts['ensemble']['path'])
    actors={m:S.load_actor(api,output/artifacts[m]['path']) for m in ('bc','ppo')}
    splits,excluded=fresh_splits(previous_spec)
    source_names=['arm_clearance_study','arm_clearance_selector','arm_clearance_verify',
                  'arm_clearance_pipeline','prepare_arm_clearance_report','consistent_flight_study',
                  'consistent_flight_selector','consistent_flight_verify','consistent_flight_diagnosis','ensemble_flight_diagnosis',
                  'curate_physics_ensemble','physics_ensemble_contract','physics_ensemble_model',
                  'physics_ensemble_study','physics_ensemble_verify','gap_flight_diagnosis',
                  'gap_flight_results','gap_coverage_verify']
    contract=dict(version=VERSION,protocol=protocol(),scope=SCOPE,scene=old_spec['scene'],identity=old_spec['identity'],
        packages=old_spec['packages'],native_source_sha256=old_spec['native_source_sha256'],
        native_reference=old_spec['native_reference'],substeps=64,maximum_requests=112,
        source_sha256={'sircus/mjlab/'+name+'.py':sha(repository/'sircus/mjlab'/(name+'.py')) for name in source_names},
        splits=splits,excluded=excluded,artifacts=artifacts,ensemble_report_sha256=C.REPORT_SHA,ensemble_audit_sha256=C.AUDIT_SHA,
        consistent_report_sha256=D.SOURCE_HASHES['report'],consistent_audit_sha256=D.SOURCE_HASHES['audit'],
        consistent_manifest_sha256=D.SOURCE_HASHES['manifest'],
        gap_report_sha256=F.SOURCE_HASHES['report'],gap_audit_sha256=F.SOURCE_HASHES['audit'],
        gap_manifest_sha256=F.SOURCE_HASHES['manifest'],clearance_rule=copy.deepcopy(G.RULE),
        parent_contract_fingerprint=bundle['spec']['fingerprint'],parent_bindings=bundle['bindings'],
        native_gpu=GPU_UUID,maximum_acquired_s=3600,maximum_queue_s=86400,new_training_requests=0)
    contract['fingerprint']=digest(contract);atomic(output/'contract.json',contract)
    atomic(output/'splits.json',splits);atomic(output/'exclusions.json',excluded)
    report=dict(version=VERSION,status='preparing',phase='cpu_planning',scope=SCOPE,protocol=protocol(),
        contract_fingerprint=contract['fingerprint'],source_sha256=contract['source_sha256'],artifacts=artifacts,
        counts=S.counts([]),comparisons=[],test_positions=[],control_gates={},new_training_requests=0,
        source_report_sha256=D.SOURCE_HASHES['report'],
        selector_sha256=sha(repository/'sircus/mjlab/consistent_flight_selector.py'))
    atomic(output/'report.json',report)
    ref=dict(scene=contract['scene'],public=contract['native_reference']['public'])
    engine=api.Engine(ref,args.assets,output/'native',prior=bundle['prior'],calibration=bundle['calibration'],home=bundle['home'],batch=32)
    clearance=G.ArmClearanceGate(contract['scene'],args.assets)
    rankings=[];requests=[]
    def meta(context):
        if shutil.disk_usage(output).free<8*1024**3:raise RuntimeError('EightGiB planning reserve required')
        anchor,metadata,timing=engine.resolve(context)
        append(output/'planning-timing.jsonl',[dict(context_id=context['id'],**timing)])
        return anchor,metadata,S.vector(api.observation(context['bucket_xy_m'],anchor),9)
    def request(context,method,values,metadata,obs,extra=None):
        return dict(id=f'arm-clearance-{method}-{context["id"]}',role=context['split'],phase=context['split'],method=method,
            context_id=context['id'],bucket_xy_m=context['bucket_xy_m'],values=S.vector(values,5),command_values=S.vector(values,5),
            observation=obs,source_teacher_id=metadata['source_hint']['id'],**copy.deepcopy(metadata),
            frozen_artifact_sha256=artifacts[method]['sha256'] if method in artifacts else digest(values),
            selection_or_learning=False,ball_flight_search_at_inference=False,geometry_search_at_inference=True,
            perturbation_mixed_units=[0.]*5,**(extra or {}))
    def controls(phase):
        result=[]
        for source,role in zip(bundle['controls'],('positive_control','negative_control')):
            context=dict(id='arm-clearance-'+role,split='control',bucket_xy_m=source['bucket_xy_m'])
            _,metadata,obs=meta(context)
            for i in range(3):
                row=request(context,'control',source['values'],metadata,obs)
                row.update(id=f'arm-clearance-{phase}-{role}-r{i}',role=role,phase=phase,repeat=i,
                    control_source_id=source['id'],control_source_sha256=digest(source),frozen_artifact_sha256=digest(source))
                result.append(row)
        return result
    pre,post=controls('controls_pre'),controls('controls_post')
    bymethod={m:[] for m in METHODS}
    for context in splits['test']:
        anchor,metadata,obs=meta(context);actors_log={}
        for m in ('bc','ppo'):
            values,residual,latent=S.actor_values(api,actors[m],obs,anchor)
            actors_log[m]=dict(values=values,normalized_residual=residual,latent_mean=latent)
        original=api.M.candidates(context,anchor,actors_log['bc']['values'],actors_log['ppo']['values'],bundle['train_rows'])
        banks={}
        for method,items in (('consistent',original),('arm_clearance',copy.deepcopy(original))):
            predictions=predictor.predict_inputs([api.M.inputs(context['bucket_xy_m'],c['values']) for c in items])
            ranking=selector.rank(contract['scene'],calibration,context,items,predictions,
                gravity=contract['native_reference']['allocated_options']['native']['gravity'])
            banks[method]=dict(candidates=items,predictions=predictions,ranking=ranking)
        resolution=dict(read(engine.directory/metadata['context_resolution_file']),file=metadata['context_resolution_file'],
                        sha256=metadata['context_resolution_sha256'],cache_hit=True)
        resolver=engine.adapter.resolver;api.verify_resolution(resolution,engine.sources,resolver.home)
        selections={}
        for method,bank in banks.items():
            items=bank['candidates'];order=bank['ranking']['order']
            if sorted(order)!=list(range(16)):raise ValueError('Complete16-candidate order required')
            checks=[];chosen=None;first_geometry=None
            for index in order[:4]:
                connection=resolver.connect(resolution,items[index]['values'])
                api.verify_connection(connection,resolution,resolver.home,items[index]['values'])
                check=dict(candidate_index=index,candidate_id=items[index]['id'],passed=bool(connection['passed']),
                    reason=connection.get('reason'),connection_file=connection['file'],connection_sha256=connection['sha256'])
                if connection['passed']:
                    if first_geometry is None:first_geometry=index
                    if method=='arm_clearance':
                        check['clearance']=clearance.check(connection['q10'],connection['q14'],context['bucket_xy_m'],
                            items[index]['values'],bank['ranking']['diagnostics'][index],
                            gravity=contract['native_reference']['allocated_options']['native']['gravity'])
                    if method=='consistent' or check['clearance']['passed']:chosen=index
                checks.append(check)
                if chosen is not None:break
            abstain=method=='arm_clearance' and chosen is None and first_geometry is not None
            if chosen is None:chosen=first_geometry if abstain else order[0]
            selections[method]=dict(checks=checks,selected_index=chosen,selected_values=items[chosen]['values'],
                all_geometry_checks_rejected=first_geometry is None,policy_abstain=abstain)

        log=dict(context_id=context['id'],bucket_xy_m=context['bucket_xy_m'],anchor=anchor,context_metadata=metadata,
            observation=obs,actor_values=actors_log,banks=banks,selection=selections,native_queries=0)
        rankings.append(log)
        for method in METHODS:
            chosen=selections[method]['selected_index'];bank=banks[method];items=bank['candidates']
            bymethod[method].append(request(context,method,items[chosen]['values'],metadata,obs,
                dict(candidate_index=chosen,candidate_id=items[chosen]['id'],ranking_sha256=digest(log),model_candidates=16,
                     geometry_checks=len(selections[method]['checks']),native_planning_queries=0,
                     policy_abstain=selections[method]['policy_abstain'],
                     clearance_rule_sha256=G.fingerprint(contract['clearance_rule']) if method=='arm_clearance' else None,
                     model_prediction=bank['predictions'][chosen],prediction_model_sha256=artifacts['ensemble']['sha256'])))
        report['planned_layouts']=len(rankings);report['elapsed_s']=time.monotonic()-started
        atomic(output/'report.json',report);print(json.dumps(dict(phase='planning',layouts=len(rankings))),flush=True)
    requests=pre+[r for m in METHODS for r in bymethod[m]]+post
    if len(requests)!=112 or len({r['id'] for r in requests})!=112:raise ValueError('Exactly112unique slots required')
    atomic(output/'rankings.json',rankings)
    packet=dict(version=VERSION,contract_fingerprint=contract['fingerprint'],requests=requests,
        rankings_sha256=sha(output/'rankings.json'),native_planning_queries=0,frozen_before_any_native_outcome=True,
        frozen_unix_s=time.time(),model_artifacts=artifacts)
    atomic(output/'prediction-plan.json',packet)
    if api.torch.cuda.is_initialized():raise RuntimeError('CPU preparation unexpectedly initialized CUDA')
    report.update(status='prepared',phase='frozen_before_gpu',prediction_plan_sha256=sha(output/'prediction-plan.json'),
        rankings_sha256=sha(output/'rankings.json'),planned_requests=112,planned_layouts=50)
    atomic(output/'report.json',report)
    return dict(output=output,engine=engine,report=report,contract=contract,requests=requests,rankings=rankings,
                preparation_started=started,prediction_sha256=sha(output/'prediction-plan.json')),api,S


def abstained(request, selection, native):
    """An explicit policy rejection, never an invented native observation."""
    if request.get('policy_abstain') is not True or request.get('method')!='arm_clearance':
        raise ValueError('Only explicit frozen clearance abstentions may skip native execution')
    selected=selection['selected_index']
    entry=next(c for c in selection['checks'] if c['candidate_index']==selected)
    if entry['passed'] is not True or entry.get('clearance',{}).get('passed') is not False:
        raise ValueError('Abstention requires an admitted endpoint and failed clearance proof')
    path=(Path(native)/entry['connection_file']).resolve()
    if not path.is_relative_to(Path(native).resolve()) or sha(path)!=entry['connection_sha256']:
        raise ValueError('Abstention exact endpoint evidence changed')
    connection=read(path)
    row=copy.deepcopy(request)
    row.update(native_release=False,invalid='policy_arm_clearance_rejected',invalid_physics=None,
        success=False,complete_success=False,route_success=False,route_quality=None,
        single_bounce_success=None,board_bounce=None,whole_ball_entry=None,
        max_retention_s=None,closest_opening_distance_m=None,external_contact_steps=None,
        board_contact_episodes=None,geometry_eligible=True,motion_admitted=True,
        endpoint_eligible=connection['endpoint_eligible'],geometry_reason=None,
        q10=connection['q10'],q14=connection['q14'],
        connection_file=entry['connection_file'],connection_sha256=entry['connection_sha256'],
        action_substituted=False,policy_rejection_version='sircus-arm-clearance-abstention-v1',
        observation_status='not_executed_policy_abstention')
    return row


def execute(state,api,S,args):
    output=state['output'];report=state['report'];rows=[];start=time.monotonic()
    def publish(phase):
        report.update(phase=phase,counts=S.counts(rows),acquired_elapsed_s=time.monotonic()-start)
        atomic(output/'report.json',report)
    def block(reqs,phase):
        result=[]
        for offset in range(0,len(reqs),32):
            if time.monotonic()-start>args.wall_time_limit-10:raise TimeoutError('Own one-hour native cap')
            if api.gpu.foreign_processes():raise RuntimeError('Foreign4090 work appeared; leaving it untouched')
            if sha(output/'prediction-plan.json')!=state['prediction_sha256']:raise ValueError('Frozen plan changed')
            for artifact in report['artifacts'].values():
                if sha(output/artifact['path'])!=artifact['sha256']:raise ValueError('Frozen checkpoint changed')
            if shutil.disk_usage(output).free<8*1024**3:raise RuntimeError('EightGiB reserve reached')
            requested=reqs[offset:offset+32]
            active=[r for r in requested if r.get('policy_abstain') is not True]
            if active:observed,timing=state['engine'].evaluate(active,phase)
            else:observed,timing=[],dict(requested=0,native_releases=0,wall_s=0.,scope='No native engine requests; all policy abstentions')
            byid={r['id']:r for r in observed};rejected=[]
            for request in requested:
                if request.get('policy_abstain') is True:
                    ledger=next(r for r in state['rankings'] if r['context_id']==request['context_id'])
                    row=abstained(request,ledger['selection']['arm_clearance'],output/'native')
                    rejected.append(row);byid[row['id']]=row
            chunk=[byid[r['id']] for r in requested]
            timing.update(policy_abstentions=len(rejected),requested_including_abstentions=len(requested))
            if len(chunk)!=len(reqs[offset:offset+32]):raise RuntimeError('Exact slot count changed')
            for row in chunk:row['learning_reward']=api.reward(row)
            if rejected:append(output/'policy-rejections.jsonl',rejected)
            append(output/'records.jsonl',chunk);append(output/'timing.jsonl',[dict(phase=phase,**timing)])
            rows.extend(chunk);result.extend(chunk);publish(phase)
        return result
    def gate(block):
        p=[r for r in block if r['role']=='positive_control'];n=[r for r in block if r['role']=='negative_control']
        ok=(len(block)==6 and len(p)==len(n)==3 and all(S.captured(r) for r in p)
            and all(r['native_release'] and not S.captured(r) and not r['whole_ball_entry'] for r in n)
            and all(not r.get('invalid') and r.get('invalid_physics') is False and r.get('scalar_score_audit_passed') is True for r in block))
        return dict(status='passed' if ok else 'failed',counts=S.counts(block))
    report.update(status='running',acquired_unix_s=time.time());publish('controls_pre')
    try:
        before=block(state['requests'][:6],'controls_pre');report['control_gates']['pre']=gate(before)
        if report['control_gates']['pre']['status']!='passed':raise RuntimeError('Positive/negative precontrols failed')
        for method in METHODS:
            group=block([r for r in state['requests'] if r['role']=='test' and r['method']==method],'test')
            report['comparisons'].append(dict(method=method,counts=S.counts(group)))
            for r in group:
                report['test_positions'].append(dict(context_id=r['context_id'],method=method,record_id=r['id'],
                    bucket_xy_m=r['bucket_xy_m'],values=r['values'],capture=S.captured(r),native_release=r['native_release'],
                    clean=S.captured(r) and r['route_quality']==2,whole_ball_entry=r['whole_ball_entry'],retention_s=r['max_retention_s'],
                    invalid_physics=r.get('invalid_physics'),clearance_rejected=r.get('policy_abstain') is True))
            publish('test')
        after=block(state['requests'][-6:],'controls_post');report['control_gates']['post']=gate(after)
        if report['control_gates']['post']['status']!='passed':raise RuntimeError('Positive/negative postcontrols failed')
        if len(rows)!=112:raise RuntimeError('Exactly112 slots required')
        report.update(status='completed',audit_pending=True)
        manifest={str(p.relative_to(output)):sha(p) for p in sorted(output.rglob('*')) if p.is_file() and p.name not in ('report.json','evidence-manifest.json')}
        atomic(output/'evidence-manifest.json',manifest);report['evidence_sha256']=manifest;publish('completed')
    except Exception as error:
        report.update(status='failed',error=type(error).__name__+': '+str(error));publish('failed');raise


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('parent-study','parent-audit','ensemble-study','ensemble-audit','consistent-study','consistent-audit','gap-study','gap-audit','native-source','assets','output','gpu-lock'):
        p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--queue-timeout',type=float,default=86400);p.add_argument('--wall-time-limit',type=float,default=3600)
    a=p.parse_args()
    if not 60<=a.wall_time_limit<=3600 or not 0<=a.queue_timeout<=86400:raise ValueError('Bounded limits required')
    if os.environ.get('CUDA_VISIBLE_DEVICES')!=GPU_UUID:raise RuntimeError('Only4090mayexecute')
    try:
        state,api,S=prepare(a)
        def waiting(q):
            state['report'].update(status='queued',phase='waiting_for_4090',queue=q);atomic(a.output/'report.json',state['report'])
        with api.gpu.gpu_slot(a.gpu_lock,timeout_s=a.queue_timeout,notify=waiting):
            def timeout(signum,frame):raise TimeoutError('Own native time limit')
            signal.signal(signal.SIGALRM,timeout);signal.setitimer(signal.ITIMER_REAL,a.wall_time_limit-5)
            try:execute(state,api,S,a)
            finally:signal.setitimer(signal.ITIMER_REAL,0)
    except Exception as e:
        if (a.output/'report.json').is_file():
            report=read(a.output/'report.json');report.update(status='failed',error=type(e).__name__+': '+str(e));atomic(a.output/'report.json',report)
        raise

if __name__=='__main__':main()
