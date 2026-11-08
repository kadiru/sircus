"""Frozen six-case development comparison of one clearance-distance correction.

CPU preparation is a separate process. Native execution uses the immutable
fixed64 package and exact already-admitted joint branches. No new training.
"""
from __future__ import annotations
import argparse
import copy
import importlib.util
import json
import os
from pathlib import Path
import shutil
import signal
import time
import numpy as np

VERSION='sircus-clearance-correction-v1'
CONTEXTS=('00','01','04','07','39','44')
METHODS=('original','corrected')
MAX_REQUESTS=48
REPEATS=3
GPU_UUID='GPU-ab1950b0-01dd-0fab-407f-2d17936b27f4'
PARENT_REPORT_SHA='fbc071332ad74302f9c3438196daaf9da036b72c3ee28059539ca234d9e09f38'
PARENT_AUDIT_SHA='d3f91a2243483c0aa5deede9e5bf1f5770646c3845fdcab7766730d13420d2bf'
SCOPE=('Six prespecified previously observed layouts; original clearance gate versus one '
    'tighter certified distance bound. Same candidate bank, ranking, top4 endpoint budget, '
    'frozen ensemble, horizon,2mm margin, sampled-flight allowance,64 physics and scoring. '
    'Three exact repeats plus12controls; abstentions count but never execute. No fitting, '
    'new holdout, noise,128, incoming-drop or driven-arm transit validation. Active selector unchanged.')


def helper():
    path=Path(__file__).with_name('policy64_diagnosis_native.py')
    spec=importlib.util.spec_from_file_location('_clearance_native_helpers',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module
N=helper()
read,sha,digest,atomic,vector=N.read,N.sha,N.digest,N.atomic,N.vector
contained,copy_proof=N.contained,N.copy_proof


def counts(rows):
    release=lambda r:r.get('native_release') is True
    return dict(requested=len(rows),native_releases=sum(release(r) for r in rows),captures=sum(N.captured(r) for r in rows),
        clean=sum(N.captured(r) and r.get('route_quality')==2 for r in rows),rejected=sum(not release(r) for r in rows),
        native_misses=sum(release(r) and not N.captured(r) and not r.get('invalid_physics') for r in rows),
        invalid=sum(release(r) and r.get('invalid_physics') is True for r in rows))


def control_gate(rows):
    out=N.control_gate(rows);out['counts']=counts(rows);return out


def abstained(request):
    row=copy.deepcopy(request)
    row.update(policy_abstain=True,native_release=False,route_success=False,success=False,complete_success=False,
        invalid='policy_arm_clearance_rejected',action_substituted=False,geometry_eligible=True,motion_admitted=True,
        observation_status='not_executed_policy_abstention')
    for key in ('invalid_physics','route_quality','single_bounce_success','board_bounce','whole_ball_entry',
                'max_retention_s','closest_opening_distance_m','external_contact_steps','board_contact_episodes',
                'quality_score','entry_class','rim_band_contact','bucket_contact_before_entry','exited_after_entry'):
        row[key]=None
    for key in list(row):
        if key.startswith('native_cache_') or key.startswith('native_initial_'):row.pop(key)
    return row


def select(ledger,method,gate,parent,gravity):
    bank=ledger['banks']['arm_clearance'];old=ledger['selection']['arm_clearance']
    lookup={r['candidate_index']:r for r in old['checks']};checks=[];chosen=None;first=None
    for index in bank['ranking']['order'][:4]:
        if index not in lookup:raise ValueError('Tighter bound must not need new geometry beyond original checked prefix')
        prior=lookup[index];connection=read(contained(parent/'native',prior['connection_file']))
        if sha(contained(parent/'native',prior['connection_file']))!=prior['connection_sha256']:raise ValueError('Endpoint proof changed')
        item=copy.deepcopy(prior)
        if prior['passed'] is True:
            if first is None:first=index
            bound=gate.check(connection['q10'],connection['q14'],ledger['bucket_xy_m'],bank['candidates'][index]['values'],
                bank['ranking']['diagnostics'][index],gravity=gravity)
            if method=='original' and bound!=prior['clearance']:raise ValueError('Original frozen clearance reconstruction differs')
            item['clearance']=bound
            if bound['passed'] is True:chosen=index
        checks.append(item)
        if chosen is not None:break
    if first is None:raise ValueError('Prespecified cases require some admitted endpoint')
    result=dict(checks=checks,selected_index=chosen if chosen is not None else first,policy_abstain=chosen is None,
        all_geometry_checks_rejected=False)
    if method=='original' and any(result[k]!=old[k] for k in ('selected_index','policy_abstain','all_geometry_checks_rejected','checks')):
        raise ValueError('Original rule/selection changed')
    return result


def freeze_requests(source_requests,source_rows,ledgers,selections,parent):
    source_by_id={r['id']:r for r in source_rows};plan=[]
    def add(source,source_row,variant,phase,repeat,selection=None,ledger=None):
        r=copy.deepcopy(source);r.update(id=f'clearance-fix-{phase}-{source["context_id"]}-{variant}-r{repeat}',
            role=source['role'] if phase.startswith('controls_') else 'diagnostic',phase=phase,method=variant,variant=variant,
            repeat=repeat,substeps=64,kind='exact_replay',selection_or_learning=False,
            parent_request_id=source['id'],parent_request_sha256=digest(source),parent_record_sha256=digest(source_row),
            parent_record_id=source_row['id'])
        if selection is None:
            r.update(policy_abstain=False,connection_file=source_row['connection_file'],connection_sha256=source_row['connection_sha256'])
            for k in ('q10','q14'):r[k]=vector(source_row[k],6)
        else:
            index=selection['selected_index'];bank=ledger['banks']['arm_clearance'];candidate=bank['candidates'][index]
            proof=next(c for c in selection['checks'] if c['candidate_index']==index)
            r.update(values=copy.deepcopy(candidate['values']),command_values=copy.deepcopy(candidate['values']),
                candidate_index=index,candidate_id=candidate['id'],policy_abstain=selection['policy_abstain'],
                model_prediction=copy.deepcopy(bank['predictions'][index]),geometry_checks=len(selection['checks']),
                connection_file=proof['connection_file'],connection_sha256=proof['connection_sha256'],
                clearance_rule_sha256=proof['clearance']['rule_sha256'],clearance_minimum_m=proof['clearance']['minimum_clearance_lower_bound_m'])
            connection=read(contained(parent/'native',proof['connection_file']))
            for k in ('q10','q14'):r[k]=vector(connection[k],6)
        plan.append(r)
    controls=[next(r for r in source_requests if r['phase']=='controls_pre' and r['role']==role and r.get('repeat')==0)
              for role in ('positive_control','negative_control')]
    def ctrl(phase):
        for source in controls:
            for repeat in range(REPEATS):add(source,source_by_id[source['id']],'control_exact',phase,repeat)
    ctrl('controls_pre')
    for code in CONTEXTS:
        context='arm-clearance-test-'+code
        source=next(r for r in source_requests if r.get('context_id')==context and r['method']=='arm_clearance')
        ledger=ledgers[context]
        for variant in METHODS:
            for repeat in range(REPEATS):add(source,source_by_id[source['id']],variant,'diagnostic',repeat,selections[context][variant],ledger)
    ctrl('controls_post')
    if len(plan)!=48 or len({r['id'] for r in plan})!=48:raise ValueError('Exact48-slot budget required')
    return plan


def prepare(args):
    from .prepare_arm_clearance_report import compact
    from .arm_clearance_selector import ArmClearanceGate
    from .arm_clearance_distance import PreciseArmClearanceGate
    parent=Path(args.parent_study);output=Path(args.output).resolve();repository=Path(__file__).resolve().parents[2]
    if sha(parent/'report.json')!=PARENT_REPORT_SHA or sha(args.parent_audit)!=PARENT_AUDIT_SHA:raise ValueError('Published original study changed')
    verified=compact(parent,args.parent_audit)
    if verified.get('status')!='verified':raise ValueError('Fully verified original evidence required')
    if output.exists() or any(output.is_relative_to(Path(p).resolve()) for p in (parent,args.native_source,args.assets,repository)):
        raise ValueError('New separate evidence namespace required')
    if shutil.disk_usage(output.parent).free<8*1024**3:raise RuntimeError('EightGiB reserve required')
    spec=read(parent/'contract.json');manifest=read(parent/'evidence-manifest.json');source_requests=read(parent/'prediction-plan.json')['requests']
    source_rows=[json.loads(x) for x in (parent/'records.jsonl').read_text().splitlines()]
    ledgers={r['context_id']:r for r in read(parent/'rankings.json')};gravity=spec['native_reference']['allocated_options']['native']['gravity']
    gates=dict(original=ArmClearanceGate(spec['scene'],args.assets),corrected=PreciseArmClearanceGate(spec['scene'],args.assets))
    selections={}
    for code in CONTEXTS:
        ctx='arm-clearance-test-'+code;selections[ctx]={m:select(ledgers[ctx],m,gates[m],parent,gravity) for m in METHODS}
    plan=freeze_requests(source_requests,source_rows,ledgers,selections,parent)
    output.mkdir();(output/'native').mkdir()
    for name in ('calibration.json','verification.json','setup-home.json','prior.json'):shutil.copyfile(parent/name,output/name)
    atomic(output/'selection-plan.json',selections)
    bindings=dict(parent_report_sha256=sha(parent/'report.json'),parent_contract_sha256=sha(parent/'contract.json'),
        parent_manifest_sha256=sha(parent/'evidence-manifest.json'),parent_audit_sha256=sha(args.parent_audit),
        parent_evidence_sha256=manifest,parent_plan_sha256=sha(parent/'prediction-plan.json'))
    names=('clearance_correction_study.py','clearance_correction_pipeline.py','clearance_correction_verify.py',
        'arm_clearance_selector.py','arm_clearance_distance.py','policy64_diagnosis_native.py','prepare_arm_clearance_report.py')
    names+=('gap_flight_correction_verify.py','arm_clearance_verify.py','clearance_horizon_diagnosis.py',
        'clearance_prediction_diagnosis.py','prepare_clearance_diagnosis_report.py')
    sources=copy.deepcopy(spec['source_sha256'])
    sources.update({'sircus/mjlab/'+n:sha(repository/'sircus/mjlab'/n) for n in names})
    protocol=dict(methods=list(METHODS),substeps=64,new_training_requests=0,maximum_requests=48)
    contract=dict(version=VERSION,scope=SCOPE,substeps=64,maximum_requests=48,maximum_acquired_s=3600,
        contexts=list(CONTEXTS),methods=list(METHODS),repeats=3,training=False,new_training_requests=0,noise=False,parent_bindings=bindings,protocol=protocol,
        scene=spec['scene'],identity=spec['identity'],native_source_sha256=spec['native_source_sha256'],packages=spec['packages'],
        diagnostic_source_sha256=sources,worker_sha256=sha(__file__),selection_plan_sha256=sha(output/'selection-plan.json'),
        created_unix_s=time.time())
    contract['fingerprint']=digest(contract);atomic(output/'contract.json',contract)
    atomic(output/'prediction-plan.json',dict(version=VERSION,contract_fingerprint=contract['fingerprint'],requests=plan,
        frozen_unix_s=time.time(),frozen_before_any_native_outcome=True,native_planning_queries=0))
    report=dict(version=VERSION,status='prepared',phase='frozen_before_gpu',scope=SCOPE,contract_fingerprint=contract['fingerprint'],
        plan_sha256=sha(output/'prediction-plan.json'),counts=counts([]),control_gates={},contexts=[],training=False,substeps=64,protocol=protocol,rows=[])
    atomic(output/'report.json',report);return report


def authenticate_prepared(args):
    output=Path(args.output);contract=read(output/'contract.json');report=read(output/'report.json')
    if sha(output/'contract.json')!=args.contract_sha256:raise ValueError('Frozen contract bytes differ from supervisor')
    unsigned=dict(contract);fp=unsigned.pop('fingerprint')
    if digest(unsigned)!=fp or contract['version']!=VERSION or report['contract_fingerprint']!=fp:raise ValueError('Frozen contract identity differs')
    parent=Path(args.parent_study);bindings=contract['parent_bindings']
    for name,key in (('report.json','parent_report_sha256'),('contract.json','parent_contract_sha256'),('evidence-manifest.json','parent_manifest_sha256'),('prediction-plan.json','parent_plan_sha256')):
        if sha(parent/name)!=bindings[key]:raise ValueError('Parent bytes changed/'+name)
    if sha(args.parent_audit)!=bindings['parent_audit_sha256']:raise ValueError('Parent audit changed')
    for name,expected in bindings['parent_evidence_sha256'].items():
        if sha(contained(parent,name))!=expected:raise ValueError('Original evidence changed/'+name)
    packet=read(output/'prediction-plan.json')
    if packet['version']!=VERSION or packet['contract_fingerprint']!=fp or len(packet['requests'])!=48 or sha(output/'prediction-plan.json')!=report['plan_sha256']:
        raise ValueError('Frozen48 requests changed')
    if sha(output/'selection-plan.json')!=contract['selection_plan_sha256']:raise ValueError('Frozen selection changed')
    repository=Path(__file__).resolve().parents[2]
    for name,expected in contract['diagnostic_source_sha256'].items():
        if sha(contained(repository,name))!=expected:raise ValueError('Frozen source changed/'+name)
    for name in ('calibration.json','verification.json','setup-home.json','prior.json'):
        if sha(output/name)!=bindings['parent_evidence_sha256'][name]:raise ValueError('Copied native input changed')
    return contract,report


def run(args,*,contract=None,report=None,runtime=None):
    started=time.monotonic();output=Path(args.output);native=output/'native'
    if contract is None:contract,report=authenticate_prepared(args)
    spec=read(Path(args.parent_study)/'contract.json');plan=read(output/'prediction-plan.json')['requests']
    native_spec=dict(spec,source_sha256=spec['native_source_sha256'])
    runtime=runtime or N.load_native(args.native_source,native_spec,args.assets)
    report.update(status='running',phase='starting');report.pop('queue',None)
    rows=[];env=None;packet=0;options_bytes=None
    def publish(phase):
        report.update(phase=phase,counts=counts(rows),reported_worker_elapsed_s=time.monotonic()-started,
            comparisons=[dict(method=m,counts=counts([r for r in rows if r['role']=='diagnostic' and r['method']==m])) for m in METHODS],
            rows=[dict(record_id=r['id'],context_id=r['context_id'],method=r['method'],repeat=r['repeat'],
                bucket_xy_m=r['bucket_xy_m'],values=r['values'],native_release=r.get('native_release') is True,
                capture=N.captured(r),clean=N.captured(r) and r.get('route_quality')==2,invalid=r.get('invalid_physics') is True,
                clearance_rejected=r.get('policy_abstain') is True) for r in rows if r['role']=='diagnostic'])
        atomic(output/'report.json',report)
        print(json.dumps(dict(phase=phase,**report['counts'])),flush=True)
    def limits():
        if len(rows)>48 or time.monotonic()-started>min(args.wall_time_limit,3600)-10:raise TimeoutError('Own finite diagnosis budget exceeded')
        if shutil.disk_usage(output).free<8*1024**3:raise RuntimeError('EightGiB evidence reserve reached')
        if runtime.gpu.foreign_processes():raise RuntimeError('Foreign4090 job appeared; leaving it untouched')
    publish('frozen')
    home=spec['identity']['setup']['home'];calibration=spec['identity']['calibration']['score_config']
    prior=read(output/'prior.json')['records']
    resolver=runtime.GeometryResolver(dict(scene=spec['scene']),args.assets,native,
        [runtime.source_geometry(r) for r in prior],home=home)
    def execute(requests,phase):
        nonlocal env,packet,options_bytes
        limits();prepared=[]
        for request in requests:
            row=copy.deepcopy(request)
            resolution=copy_proof(args.parent_study,native,row['context_resolution_file'],row['context_resolution_sha256'])
            runtime.verify_resolution(resolution,resolver.sources,resolver.home)
            connection=copy_proof(args.parent_study,native,row['connection_file'],row['connection_sha256'])
            runtime.verify_connection(connection,resolution,resolver.home,row['values'])
            if connection['passed'] is not True:raise ValueError('Selected endpoint must already pass strict admission')
            if row.get('policy_abstain') is True:
                row=abstained(row)
            else:
                row.update(geometry_eligible=True,motion_admitted=True,action_substituted=False)
            prepared.append(row)
        active=[i for i,r in enumerate(prepared) if r.get('policy_abstain') is not True]
        if active:
            if env is None:
                env=runtime.Environment(spec['scene'],args.assets,calibration,n=32,bounds=runtime.bounds,substeps=64)
                if env.compiled_geometry!=read(contained(Path(args.parent_study)/'native','native-compiled-geometry.json')):
                    raise RuntimeError('Compiled native collision geometry differs from parent')
                atomic(native/'native-compiled-geometry.json',env.compiled_geometry)
            actual=env.effective_options
            if actual.get('version')!='sircus-calibrated-native-warp-options-v1' or type(actual.get('substeps')) is not int or actual['substeps']!=64:
                raise RuntimeError('Exact effective64 native options required')
            if actual!=read(contained(Path(args.parent_study)/'native','effective-options.json')):
                raise RuntimeError('Allocated64 options differ from authenticated parent')
            raw=(json.dumps(actual,indent=2,sort_keys=True,allow_nan=False)+'\n').encode()
            path=native/'effective-options.json'
            if options_bytes is None:
                with path.open('xb') as stream:stream.write(raw)
                options_bytes=raw
            if raw!=options_bytes or path.read_bytes()!=options_bytes:raise RuntimeError('Effective options changed')
            bucket=np.asarray([prepared[i]['bucket_xy_m'] for i in active]);values=np.asarray([prepared[i]['values'] for i in active])
            q10,q14=(np.asarray([prepared[i][key] for i in active]) for key in ('q10','q14'))
            with runtime.fixed_endpoints(env,bucket,values,q10,q14,np.ones(len(active),dtype=bool)):
                measured,metadata=env.run(bucket,values,runtime.constraints(len(active)),record=True)
            if len(measured)!=len(active):raise RuntimeError('Native request count changed')
            arrays=env.last_arrays;filename=f'native-{packet:04d}.npz';packet+=1
            cache=native/filename
            if cache.exists():raise ValueError('Refusing to replace native evidence')
            np.savez_compressed(cache,**arrays,record_ids=np.asarray([prepared[i]['id'] for i in active]))
            fingerprint=sha(cache)
            for column,(index,observed) in enumerate(zip(active,measured)):
                row=prepared[index]
                if observed.get('native_release') is not True or observed.get('values')!=row['values'] or observed.get('bucket_xy_m')!=row['bucket_xy_m']:
                    raise RuntimeError('Native exact request association changed')
                for key in ('q10','q14'):
                    if not np.array_equal(observed[key],row[key]):raise RuntimeError('Native replay/connection joint branch changed')
                    initial='native_initial_'+key
                    expected=np.asarray(row[key],dtype=np.float32).astype(float)
                    if initial not in observed or not np.array_equal(observed[initial],expected):
                        raise RuntimeError('Actual time-zero native joint branch differs')
                if observed.get('contact_protocol',{}).get('substeps')!=64 or observed.get('scalar_score_audit_passed') is not True:
                    raise RuntimeError('Native64/scalar oracle evidence missing')
                expected_release=np.asarray([*row['values'][:2],row['values'][2]+row['values'][4]+spec['scene']['ball']['radius_m']])
                if (np.asarray(arrays['position_m']).shape[1:]!=(len(active),3)
                    or not np.allclose(arrays['position_m'][0,column],expected_release,atol=1e-6,rtol=0)):
                    raise RuntimeError('Actual initial ball release differs from requested geometry')
                trace={key:(value[:,column] if value.ndim>=2 and key!='time_s' else value) for key,value in arrays.items()}
                diagnostic=runtime.first_flight(spec['scene'],row['bucket_xy_m'],observed['board_normal'],calibration,trace,
                    valid=observed.get('invalid_physics') is False)
                row.update(copy.deepcopy(observed),native_cache_file=filename,native_cache_row=column,
                    native_cache_sha256=fingerprint,effective_options_file=path.name,effective_options_sha256=sha(path),
                    first_flight=diagnostic,native_metadata=copy.deepcopy(metadata),adapter_directory='native')
        for filename,subset in (('records.jsonl',prepared),('native/shots.jsonl',[r for r in prepared if r.get('policy_abstain') is not True]),
                                ('policy-rejections.jsonl',[r for r in prepared if r.get('policy_abstain') is True])):
            if subset:
                with (output/filename).open('a') as stream:
                    for row in subset:stream.write(json.dumps(row,allow_nan=False)+'\n')
        rows.extend(prepared);publish(phase)
        return prepared
    try:
        before=execute(plan[:6],'controls_pre');report['control_gates']['pre']=control_gate(before)
        if report['control_gates']['pre']['status']!='passed':raise RuntimeError('Initial matched native controls failed')
        for index,code in enumerate(CONTEXTS):
            cohort=execute(plan[6+6*index:12+6*index],'diagnostic')
            report['contexts'].append(dict(context_id='arm-clearance-test-'+code,
                variants={v:counts([r for r in cohort if r['variant']==v]) for v in dict.fromkeys(r['variant'] for r in cohort)}))
        after=execute(plan[-6:],'controls_post');report['control_gates']['post']=control_gate(after)
        if report['control_gates']['post']['status']!='passed':raise RuntimeError('Final matched native controls failed')
        if len(rows)!=48:raise RuntimeError('Complete48-request accounting required')
        limits();report.update(status='completed',gate=dict(status='passed',controls_passed=True,complete=True,
            scope='Diagnostic execution completion only; no new policy success-rate claim'))
        manifest={str(p.relative_to(output)):sha(p) for p in sorted(output.rglob('*'))
            if p.is_file() and p.name not in ('report.json','evidence-manifest.json')}
        atomic(output/'evidence-manifest.json',manifest);report['evidence_sha256']=manifest;publish('completed')
        return report
    except Exception as error:
        report.update(status='failed',error=type(error).__name__+': '+str(error));publish('failed');raise

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('parent-study','parent-audit','native-source','assets','output'):
        parser.add_argument('--'+key,type=Path,required=True)
    parser.add_argument('--prepare-only',action='store_true')
    parser.add_argument('--gpu-lock',type=Path)
    parser.add_argument('--contract-sha256')
    parser.add_argument('--wall-time-limit',type=float,default=3600)
    parser.add_argument('--queue-timeout',type=float,default=86400)
    args=parser.parse_args()
    if args.prepare_only:
        if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('CPU preparation requires GPUs hidden')
        prepare(args);return
    if not 60<=args.wall_time_limit<=3600 or not 0<=args.queue_timeout<=86400:raise ValueError('Bounded worker required')
    if os.environ.get('CUDA_VISIBLE_DEVICES')!=GPU_UUID:raise RuntimeError('Only authorized4090 may execute')
    contract,report=authenticate_prepared(args)
    native_spec=dict(read(Path(args.parent_study)/'contract.json'))
    native_spec['source_sha256']=native_spec['native_source_sha256']
    runtime=N.load_native(args.native_source,native_spec,args.assets)
    def queued(value):
        report.update(status='queued',phase='waiting_4090',queue=value);atomic(args.output/'report.json',report)
        print(json.dumps(value),flush=True)
    def deadline(signum,frame):raise TimeoutError('Own oneGPUhour deadline reached')
    try:
        with runtime.gpu.gpu_slot(args.gpu_lock,timeout_s=args.queue_timeout,notify=queued):
            signal.signal(signal.SIGALRM,deadline);signal.setitimer(signal.ITIMER_REAL,args.wall_time_limit-5)
            report['acquired_unix_s']=time.time()
            atomic(args.output.with_name(args.output.name+'-budget.json'),dict(acquired_unix_s=report['acquired_unix_s'],limit_s=args.wall_time_limit))
            try:run(args,contract=contract,report=report,runtime=runtime)
            finally:signal.setitimer(signal.ITIMER_REAL,0)
    except Exception as error:
        if read(args.output/'report.json').get('status')!='failed':
            report.update(status='failed',phase='failed',error=type(error).__name__+': '+str(error));atomic(args.output/'report.json',report)
        raise

if __name__=='__main__':main()
