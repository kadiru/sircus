"""Frozen case01 comparison of one longer conditional arm-clearance window.

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

VERSION='sircus-clearance-horizon-study-v1'
CONTEXTS=('01',)
CONTEXT_ID='arm-clearance-test-01'
CANDIDATE_INDEX=14
METHODS=('short_window','long_window')
MAX_REQUESTS=18
REPEATS=3
GPU_UUID='GPU-ab1950b0-01dd-0fab-407f-2d17936b27f4'
PARENT_REPORT_SHA='fbc071332ad74302f9c3438196daaf9da036b72c3ee28059539ca234d9e09f38'
PARENT_AUDIT_SHA='d3f91a2243483c0aa5deede9e5bf1f5770646c3845fdcab7766730d13420d2bf'
SCOPE=('One previously observed layout/candidate14; same exact values, admitted joints, '
    'five frozen ensemble paths, conservative support bound,2mm margin,64 physics and strict scoring. '
    'Only the conditional post-board clearance window extends from descending rim-band start '
    'to descending table-tangent height. Three repeats each plus12controls; no candidate replacement, '
    'new geometry queries, training, noise,128, policy promotion, incoming-drop or driven-arm transit proof. '
    'Projection after a bucket/table impact is not a prediction of its rebound.')


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
    """Certify exactly the same already admitted candidate; never try alternatives."""
    if method not in METHODS or ledger['context_id']!=CONTEXT_ID:
        raise ValueError('Exact case01/window method required')
    bank=ledger['banks']['arm_clearance'];old=ledger['selection']['arm_clearance']
    if old['selected_index']!=CANDIDATE_INDEX or old.get('policy_abstain') is not False:
        raise ValueError('Original case01 must have selected candidate14')
    checked=[r for r in old['checks'] if r['candidate_index']==CANDIDATE_INDEX]
    if len(checked)!=1 or checked[0]['passed'] is not True:
        raise ValueError('One original admitted candidate14 endpoint proof required')
    prior=checked[0];path=contained(parent/'native',prior['connection_file'])
    if sha(path)!=prior['connection_sha256']:raise ValueError('Endpoint proof changed')
    connection=read(path)
    if connection.get('passed') is not True:raise ValueError('Original endpoint was not admitted')
    proof=gate.check(connection['q10'],connection['q14'],ledger['bucket_xy_m'],
        bank['candidates'][CANDIDATE_INDEX]['values'],bank['ranking']['diagnostics'][CANDIDATE_INDEX],gravity=gravity)
    item=copy.deepcopy(prior);item['clearance']=proof
    return dict(checks=[item],selected_index=CANDIDATE_INDEX,policy_abstain=proof['passed'] is not True,
        all_geometry_checks_rejected=False)


def window_summary(method,certificate):
    horizons=[m['horizon_s'] for m in certificate['members']]
    return dict(method=method,passed=certificate['passed'],
        horizon_min_s=min(horizons) if horizons else None,horizon_max_s=max(horizons) if horizons else None,
        minimum_clearance_lower_bound_m=certificate['minimum_clearance_lower_bound_m'],
        required_clearance_m=certificate['required_margin_m'])

def freeze_requests(source_requests,source_rows,ledgers,selections,parent):
    source_by_id={r['id']:r for r in source_rows};plan=[]
    def add(source,source_row,variant,phase,repeat,selection=None,ledger=None):
        r=copy.deepcopy(source);r.update(id=f'horizon-window-{phase}-{source["context_id"]}-{variant}-r{repeat}',
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
    if len(plan)!=MAX_REQUESTS or len({r['id'] for r in plan})!=MAX_REQUESTS:raise ValueError('Exact18-slot budget required')
    return plan


def prepare(args):
    from .prepare_arm_clearance_report import compact
    from .clearance_horizon_selector import HorizonArmClearanceGate
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
    gates=dict(short_window=PreciseArmClearanceGate(spec['scene'],args.assets),long_window=HorizonArmClearanceGate(spec['scene'],args.assets))
    selections={}
    for code in CONTEXTS:
        ctx='arm-clearance-test-'+code;selections[ctx]={m:select(ledgers[ctx],m,gates[m],parent,gravity) for m in METHODS}
    plan=freeze_requests(source_requests,source_rows,ledgers,selections,parent)
    output.mkdir();(output/'native').mkdir()
    for name in ('calibration.json','verification.json','setup-home.json','prior.json'):shutil.copyfile(parent/name,output/name)
    atomic(output/'selection-plan.json',selections)
    ledger=ledgers[CONTEXT_ID];chosen=selections[CONTEXT_ID]
    selected=chosen[METHODS[0]]['checks'][0]
    certificates={m:chosen[m]['checks'][0]['clearance'] for m in METHODS}
    atomic(output/'window-certificates.json',dict(context_id=CONTEXT_ID,candidate_index=CANDIDATE_INDEX,
        connection_file=selected['connection_file'],connection_sha256=selected['connection_sha256'],
        diagnostic=ledger['banks']['arm_clearance']['ranking']['diagnostics'][CANDIDATE_INDEX],certificates=certificates))
    windows=[window_summary(m,certificates[m]) for m in METHODS]
    bindings=dict(parent_report_sha256=sha(parent/'report.json'),parent_contract_sha256=sha(parent/'contract.json'),
        parent_manifest_sha256=sha(parent/'evidence-manifest.json'),parent_audit_sha256=sha(args.parent_audit),
        parent_evidence_sha256=manifest,parent_plan_sha256=sha(parent/'prediction-plan.json'))
    # Pin all direct runtime Python modules before any native observation, including
    # this worker, the independent auditor and both gates. Native execution still
    # imports only the separately pinned immutable fixed64 package.
    sources=copy.deepcopy(spec['source_sha256'])
    for directory in ('sircus/mjlab','sircus/isaac'):
        for path in sorted((repository/directory).glob('*.py')):
            sources[str(path.relative_to(repository))]=sha(path)
    protocol=dict(methods=list(METHODS),substeps=64,new_training_requests=0,maximum_requests=MAX_REQUESTS)
    contract=dict(version=VERSION,scope=SCOPE,substeps=64,maximum_requests=MAX_REQUESTS,maximum_acquired_s=3600,
        context_id=CONTEXT_ID,candidate_index=CANDIDATE_INDEX,contexts=list(CONTEXTS),methods=list(METHODS),repeats=3,training=False,new_training_requests=0,noise=False,parent_bindings=bindings,protocol=protocol,
        scene=spec['scene'],identity=spec['identity'],native_source_sha256=spec['native_source_sha256'],packages=spec['packages'],
        diagnostic_source_sha256=sources,worker_sha256=sha(__file__),selection_plan_sha256=sha(output/'selection-plan.json'),
        window_certificates_sha256=sha(output/'window-certificates.json'),created_unix_s=time.time())
    contract['fingerprint']=digest(contract);atomic(output/'contract.json',contract)
    atomic(output/'prediction-plan.json',dict(version=VERSION,contract_fingerprint=contract['fingerprint'],requests=plan,
        frozen_unix_s=time.time(),frozen_before_any_native_outcome=True,native_planning_queries=0))
    report=dict(version=VERSION,status='prepared',phase='frozen_before_gpu',scope=SCOPE,contract_fingerprint=contract['fingerprint'],
        plan_sha256=sha(output/'prediction-plan.json'),counts=counts([]),control_gates={},contexts=[],training=False,substeps=64,protocol=protocol,windows=windows,rows=[])
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
    if packet['version']!=VERSION or packet['contract_fingerprint']!=fp or len(packet['requests'])!=MAX_REQUESTS or sha(output/'prediction-plan.json')!=report['plan_sha256']:
        raise ValueError('Frozen18 requests changed')
    if sha(output/'selection-plan.json')!=contract['selection_plan_sha256']:raise ValueError('Frozen selection changed')
    if sha(output/'window-certificates.json')!=contract['window_certificates_sha256']:raise ValueError('Frozen window certificates changed')
    certificates=read(output/'window-certificates.json')['certificates']
    if report['windows']!=[window_summary(m,certificates[m]) for m in METHODS]:raise ValueError('Frozen window summary changed')
    if report['status'] not in ('prepared','queued') or any((output/name).exists() for name in ('records.jsonl','evidence-manifest.json')):
        raise ValueError('Prepared evidence is single-use; never replay into existing results')
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
        if len(rows)>MAX_REQUESTS or time.monotonic()-started>min(args.wall_time_limit,3600)-10:raise TimeoutError('Own finite diagnosis budget exceeded')
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
        if len(rows)!=MAX_REQUESTS:raise RuntimeError('Complete18-request accounting required')
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
    if args.gpu_lock is None:raise ValueError('Existing shared4090 lock path required')
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
            # Queueing is not permission to execute changed source or evidence.
            contract,report=authenticate_prepared(args)
            report['acquired_unix_s']=time.time()
            atomic(args.output.with_name(args.output.name+'-budget.json'),dict(acquired_unix_s=report['acquired_unix_s'],limit_s=args.wall_time_limit))
            try:run(args,contract=contract,report=report,runtime=runtime)
            finally:signal.setitimer(signal.ITIMER_REAL,0)
    except Exception as error:
        if read(args.output/'report.json').get('status')!='failed':
            report.update(status='failed',phase='failed',error=type(error).__name__+': '+str(error));atomic(args.output/'report.json',report)
        raise

if __name__=='__main__':main()
