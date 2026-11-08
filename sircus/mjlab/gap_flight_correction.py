"""One prespecified +60mm-gap test on three previously diagnosed finger strikes.

CPU preparation freezes30 requests. Native execution reuses unchanged64 physics,
strict geometry, matched controls and exact original joint branches. No fitting.
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
import sys
import time
import numpy as np

VERSION='sircus-gap-flight-correction-v1'
CONTEXTS=('12','24','48')
GAP_DELTA_M=.060
MAX_REQUESTS=30
GPU_UUID='GPU-ab1950b0-01dd-0fab-407f-2d17936b27f4'
SCOPE=('Post-hoc development test on three original immediate robot-contact misses. '
       'Only release gap increases60mm; boardXY,height,tilt and bucket stay fixed. '
       'Three exact repeats per original/corrected shot plus12 matched controls; '
       'same64 physics and unchanged route scoring. No fitting, noise,128, generalization or driven-motion claim. '
       'Changing gap also changes impact energy; improvement is not assumed.')


def helper():
    path=Path(__file__).with_name('policy64_diagnosis_native.py')
    spec=importlib.util.spec_from_file_location('_gap_native_helpers',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module
N=helper()
read,sha,digest,atomic,vector=N.read,N.sha,N.digest,N.atomic,N.vector
contained,copy_proof,rejected,counts,control_gate=N.contained,N.copy_proof,N.rejected,N.counts,N.control_gate


def make_plan(rows):
    plan=[]
    def request(source,variant,phase,repeat):
        changed=variant=='gap_plus60'
        values=vector(source['values'],5)
        if changed:values[4]+=GAP_DELTA_M
        r=dict(id=f'gap-fix-{phase}-{source["context_id"]}-{variant}-r{repeat}',phase=phase,
            role=source['role'] if phase.startswith('controls_') else 'diagnostic',
            context_id=source['context_id'],variant=variant,kind='coordinate_swap' if changed else 'exact_replay',
            repeat=repeat,substeps=64,origin_method=source['method'],
            parent_record_id=source['id'],parent_record_sha256=digest(source),
            bucket_xy_m=vector(source['bucket_xy_m'],2),values=values,command_values=copy.deepcopy(values),
            original_native_release=source['native_release'],swapped_coordinates=[4] if changed else [],
            selection_or_learning=False,source_teacher_id=source['source_teacher_id'],observation=vector(source['observation'],9))
        for key in ('source_hint','effective_source_hint','context_resolution_file','context_resolution_sha256',
                    'context_home_sha256','context_layout_eligible','context_anchor_admitted','context_hard_reject'):
            r[key]=copy.deepcopy(source[key])
        if not changed:
            for key in ('q10','q14'):r[key]=vector(source[key],6)
            r.update(connection_file=source['connection_file'],connection_sha256=source['connection_sha256'],
                parent_native_cache_file=source['native_cache_file'],parent_native_cache_sha256=source['native_cache_sha256'])
        return r
    controls=[]
    for role in ('positive_control','negative_control'):
        source=[r for r in rows if r['phase']=='controls_pre' and r['role']==role and r.get('repeat')==0]
        if len(source)!=1 or not source[0].get('native_release') or N.captured(source[0])!=(role=='positive_control'):
            raise ValueError('Exact original admitted controls required')
        controls.append(source[0])
    def control_phase(phase):
        for source in controls:
            for repeat in range(3):plan.append(request(source,'control_exact',phase,repeat))
    control_phase('controls_pre')
    for code in CONTEXTS:
        found=[r for r in rows if r['method']=='consistent' and r['context_id']=='gap-coverage-test-'+code]
        if len(found)!=1 or not found[0].get('native_release') or N.captured(found[0]):
            raise ValueError('Exact three diagnosed original native misses required')
        for variant in ('original','gap_plus60'):
            for repeat in range(3):plan.append(request(found[0],variant,'diagnostic',repeat))
    control_phase('controls_post')
    if len(plan)!=30 or len({r['id'] for r in plan})!=30:raise RuntimeError('Literal30 unique requests required')
    return plan


def prepare(args):
    from .gap_flight_diagnosis import verify_inputs
    parent,report,spec,proof,manifest,packet,rows=verify_inputs(args.parent_study,args.parent_audit)
    output=Path(args.output).resolve();repository=Path(__file__).resolve().parents[2]
    if output.exists() or any(output.is_relative_to(Path(p).resolve()) for p in (parent,args.native_source,args.assets,repository)):
        raise ValueError('Separate new immutable namespace required')
    output.parent.mkdir(parents=True,exist_ok=True)
    if shutil.disk_usage(output.parent).free<8*1024**3:raise RuntimeError('EightGiB reserve required')
    analysis=read(args.analysis)
    if analysis.get('status')!='completed':raise ValueError('Completed source diagnosis required')
    plan=make_plan(rows);output.mkdir();(output/'native').mkdir()
    for name in ('calibration.json','verification.json','setup-home.json','prior.json'):
        shutil.copyfile(parent/name,output/name)
    shutil.copyfile(args.analysis,output/'analysis.json')
    bindings=dict(parent_contract_fingerprint=spec['fingerprint'],parent_report_sha256=sha(parent/'report.json'),
        parent_contract_sha256=sha(parent/'contract.json'),parent_manifest_sha256=sha(parent/'evidence-manifest.json'),
        parent_audit_sha256=sha(args.parent_audit),parent_rows_sha256=manifest['records.jsonl'],
        parent_evidence_sha256=manifest,analysis_sha256=sha(args.analysis),
        parent_records_sha256={r['parent_record_id']:r['parent_record_sha256'] for r in plan})
    atomic(output/'source-bindings.json',bindings)
    names=('gap_flight_correction.py','gap_flight_correction_verify.py','policy64_diagnosis_native.py','gap_flight_diagnosis.py','gap_flight_pipeline.py','prepare_gap_flight_report.py')
    sources={'sircus/mjlab/'+name:sha(repository/'sircus/mjlab'/name) for name in names}
    contract=dict(version=VERSION,scope=SCOPE,substeps=64,maximum_requests=30,maximum_acquired_s=3600,
        contexts=list(CONTEXTS),gap_delta_m=GAP_DELTA_M,repeats=3,plan=plan,training=False,noise=False,
        analysis_sha256=sha(args.analysis),parent_bindings=bindings,scene=spec['scene'],identity=spec['identity'],
        native_source_sha256=spec['native_source_sha256'],packages=spec['packages'],
        diagnostic_source_sha256=sources,worker_sha256=sha(__file__),created_unix_s=time.time())
    contract['fingerprint']=digest(contract);atomic(output/'contract.json',contract)
    packet=dict(version=VERSION,contract_fingerprint=contract['fingerprint'],requests=plan,
        frozen_unix_s=time.time(),frozen_before_any_native_outcome=True)
    atomic(output/'prediction-plan.json',packet)
    report=dict(version=VERSION,status='prepared',phase='frozen_before_gpu',scope=SCOPE,
        contract_fingerprint=contract['fingerprint'],analysis_sha256=sha(args.analysis),
        plan_sha256=sha(output/'prediction-plan.json'),counts=counts([]),control_gates={},contexts=[],
        training=False,substeps=64)
    atomic(output/'report.json',report);return report


def authenticate_prepared(args):
    output=Path(args.output);contract=read(output/'contract.json');report=read(output/'report.json')
    if sha(output/'contract.json')!=args.contract_sha256:raise ValueError('Frozen contract bytes differ from supervisor')
    unsigned=dict(contract);fp=unsigned.pop('fingerprint')
    if digest(unsigned)!=fp or contract['version']!=VERSION or report['contract_fingerprint']!=fp:
        raise ValueError('Frozen preparation fingerprint differs')
    bindings=contract['parent_bindings'];parent=Path(args.parent_study)
    for name,key in (('report.json','parent_report_sha256'),('contract.json','parent_contract_sha256'),('evidence-manifest.json','parent_manifest_sha256')):
        if sha(parent/name)!=bindings[key]:raise ValueError('Parent bytes changed: '+name)
    if sha(args.parent_audit)!=bindings['parent_audit_sha256']:raise ValueError('Original audit changed')
    for name,expected in bindings['parent_evidence_sha256'].items():
        if sha(contained(parent,name))!=expected:raise ValueError('Original evidence changed: '+name)
    if sha(output/'analysis.json')!=contract['analysis_sha256']:raise ValueError('Frozen diagnosis changed')
    rows=[json.loads(line) for line in (parent/'records.jsonl').read_text().splitlines()]
    if make_plan(rows)!=contract['plan']:raise ValueError('Literal prespecified correction plan differs')
    packet=read(output/'prediction-plan.json')
    if packet['requests']!=contract['plan'] or sha(output/'prediction-plan.json')!=report['plan_sha256']:
        raise ValueError('Frozen request packet changed')
    repository=Path(__file__).resolve().parents[2]
    for name,expected in contract['diagnostic_source_sha256'].items():
        if sha(contained(repository,name))!=expected:raise ValueError('Frozen correction source changed')
    for name in ('calibration.json','verification.json','setup-home.json','prior.json'):
        if sha(output/name)!=bindings['parent_evidence_sha256'][name]:raise ValueError('Copied native input differs')
    return contract,report


def run(args,*,contract=None,report=None,runtime=None):
    started=time.monotonic();output=Path(args.output);native=output/'native'
    if contract is None:contract,report=authenticate_prepared(args)
    spec=read(Path(args.parent_study)/'contract.json');plan=contract['plan']
    native_spec=dict(spec,source_sha256=spec['native_source_sha256'])
    runtime=runtime or N.load_native(args.native_source,native_spec,args.assets)
    report.update(status='running',phase='starting');report.pop('queue',None)
    rows=[];env=None;packet=0;options_bytes=None
    def publish(phase):
        report.update(phase=phase,counts=counts(rows),reported_worker_elapsed_s=time.monotonic()-started)
        atomic(output/'report.json',report)
        print(json.dumps(dict(phase=phase,**report['counts'])),flush=True)
    def limits():
        if len(rows)>30 or time.monotonic()-started>min(args.wall_time_limit,3600)-10:raise TimeoutError('Own finite diagnosis budget exceeded')
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
            if row['kind']=='historical_rejection':
                row=rejected(row,row['historical_rejection_reason'])
            elif row['kind']=='coordinate_swap':
                connection=resolver.connect(resolution,row['values'])
                runtime.verify_connection(connection,resolution,resolver.home,row['values'])
                row.update(connection_file=connection['file'],connection_sha256=connection['sha256'],
                    geometry_eligible=bool(connection['passed']),endpoint_eligible=connection['endpoint_eligible'],
                    motion_admitted=bool(connection['passed']),action_substituted=False,geometry_reason=connection.get('reason'))
                if connection['passed']:
                    row.update(q10=vector(connection['q10'],6),q14=vector(connection['q14'],6))
                else:row=rejected(row,connection.get('reason') or 'requested_setup_not_admitted')
            else:
                copy_proof(args.parent_study,native,row['connection_file'],row['connection_sha256'])
                row.update(geometry_eligible=True,motion_admitted=True,action_substituted=False)
            prepared.append(row)
        active=[i for i,r in enumerate(prepared) if r.get('geometry_eligible')]
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
        for row in prepared:
            if not row.get('native_release'):
                row['first_flight']=runtime.first_flight(spec['scene'],row['bucket_xy_m'],[0.,0.,1.],calibration,{})
            if row['kind']=='historical_rejection' and any(k in row for k in ('q10','q14','native_cache_file')):
                raise RuntimeError('Historical rejected slot cannot carry a native branch/cache')
        for filename in ('records.jsonl','native/shots.jsonl'):
            with (output/filename).open('a') as stream:
                for row in prepared:stream.write(json.dumps(row,allow_nan=False)+'\n')
        rows.extend(prepared);publish(phase)
        return prepared
    try:
        before=execute(plan[:6],'controls_pre');report['control_gates']['pre']=control_gate(before)
        if report['control_gates']['pre']['status']!='passed':raise RuntimeError('Initial matched native controls failed')
        for index,code in enumerate(CONTEXTS):
            cohort=execute(plan[6+6*index:12+6*index],'diagnostic')
            report['contexts'].append(dict(context_id='gap-coverage-test-'+code,
                variants={v:counts([r for r in cohort if r['variant']==v]) for v in dict.fromkeys(r['variant'] for r in cohort)}))
        after=execute(plan[-6:],'controls_post');report['control_gates']['post']=control_gate(after)
        if report['control_gates']['post']['status']!='passed':raise RuntimeError('Final matched native controls failed')
        if len(rows)!=30:raise RuntimeError('Complete30-request accounting required')
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
    parser.add_argument('--analysis',type=Path)
    parser.add_argument('--prepare-only',action='store_true')
    parser.add_argument('--gpu-lock',type=Path)
    parser.add_argument('--contract-sha256')
    parser.add_argument('--wall-time-limit',type=float,default=3600)
    parser.add_argument('--queue-timeout',type=float,default=86400)
    args=parser.parse_args()
    if args.prepare_only:
        if not args.analysis:raise ValueError('Completed diagnosis required')
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
