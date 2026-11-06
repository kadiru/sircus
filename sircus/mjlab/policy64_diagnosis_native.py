"""Standalone bounded paired diagnosis; no training or historical evidence edits.

Run this file by absolute path with the pinned Python. All ``sircus`` imports
are deliberately delayed until --native-source selects the immutable original
native package. Importing this worker as a current-package module is for mocks
only; a real run rejects mixed package roots.
"""
from __future__ import annotations

import argparse
from collections import Counter
import copy
import datetime
import hashlib
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import signal
import sys
import time
from types import SimpleNamespace

import numpy as np

VERSION='sircus-policy64-paired-native-diagnosis-v1'
PARENT_VERSION='sircus-policy64-bc-ppo-v1'
PARENT_AUDIT='sircus-policy64-independent-audit-v1'
ANALYSIS_VERSION='sircus-policy64-diagnosis-v1'
CONTEXTS=('04','07','13','17','21','26','43')
SWAPS=(('swap_x',(0,)),('swap_y',(1,)),('swap_height',(2,)),
       ('swap_tilt',(3,)),('swap_gap',(4,)),('swap_height_gap',(2,4)))
GPU_UUID='GPU-ab1950b0-01dd-0fab-407f-2d17936b27f4'
SCOPE=('Post-hoc diagnostic on seven historical BC-only TEST captures, not new '
       'held-out evaluation or training. Exact admitted original joint branches '
       'replayed at64 only; original rejects retained; six declared swaps use '
       'unchanged fresh finite strict admission. No jitter,128 or physical claim.')


def read(path):return json.loads(Path(path).read_text())


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda:stream.read(1024*1024),b''):h.update(part)
    return h.hexdigest()


def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,allow_nan=False).encode()).hexdigest()


def contained(root,name):
    root=Path(root).resolve()
    if not isinstance(name,str) or not name or Path(name).is_absolute() or '..' in Path(name).parts:
        raise ValueError('Relative evidence path required')
    path=(root/name).resolve()
    if not path.is_relative_to(root) or not path.is_file():raise ValueError('Evidence path missing or escaped')
    return path


def atomic(path,value):
    path=Path(path);temporary=path.with_name(path.name+'.tmp')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');temporary.replace(path)


def vector(value,n):
    a=np.asarray(value,dtype=float)
    if a.shape!=(n,) or not np.isfinite(a).all():raise ValueError('Exact finite vector required')
    return a.tolist()


def captured(row):
    return bool(row.get('native_release') and row.get('route_success')
                and not row.get('invalid') and row.get('invalid_physics') is False)


def counts(rows):
    return dict(attempts=len(rows),native_releases=sum(bool(r.get('native_release')) for r in rows),
        captures=sum(captured(r) for r in rows),clean=sum(captured(r) and r.get('route_quality')==2 for r in rows),
        rejected=sum(not r.get('native_release') for r in rows),
        invalid_physics=sum(bool(r.get('invalid_physics')) for r in rows))


def authenticate_parent(study,audit,published):
    """Check exact passed public ancestry and EVERY immutable manifest member."""
    study,audit=Path(study),Path(audit);public=read(published)['provenance']
    for path,key in ((study/'report.json','report_sha256'),(study/'contract.json','contract_sha256'),
                     (study/'evidence-manifest.json','evidence_manifest_sha256'),(audit,'independent_audit_sha256')):
        if sha(path)!=public[key]:raise ValueError('Published parent byte binding differs: '+path.name)
    spec,report,proof,manifest=(read(path) for path in (study/'contract.json',study/'report.json',audit,study/'evidence-manifest.json'))
    unsigned=dict(spec);fp=unsigned.pop('fingerprint',None)
    if digest(unsigned)!=fp or fp!=public['contract_fingerprint']:raise ValueError('Parent contract fingerprint differs')
    checks=proof.get('checks')
    if (proof.get('version')!=PARENT_AUDIT or proof.get('status')!='passed' or proof.get('failures')
        or proof.get('checks_count')!=346485 or not isinstance(checks,list) or len(checks)!=346485
        or any(c.get('passed') is not True for c in checks)):
        raise ValueError('Exact complete passed independent parent audit required')
    if (proof.get('report_sha256')!=sha(study/'report.json')
        or proof.get('evidence_manifest_sha256')!=sha(study/'evidence-manifest.json')
        or proof.get('contract_fingerprint')!=fp):raise ValueError('Parent independent proof binding differs')
    if (spec.get('version')!=PARENT_VERSION or report.get('version')!=PARENT_VERSION
        or report.get('status')!='completed' or report.get('phase')!='completed'
        or report.get('contract_fingerprint')!=fp or report.get('evidence_sha256')!=manifest):
        raise ValueError('Completed parent report/manifest required')
    if not isinstance(manifest,dict) or not manifest:raise ValueError('Nonempty parent manifest required')
    for name,expected in manifest.items():
        if sha(contained(study,name))!=expected:raise ValueError('Parent artifact bytes differ: '+name)
    actual={str(p.relative_to(study)) for p in study.rglob('*') if p.is_file()
            and p.name not in ('report.json','evidence-manifest.json') and not p.name.startswith('audit')}
    if actual!=set(manifest):raise ValueError('Parent manifest is incomplete')
    rows=[json.loads(line) for line in contained(study,'rows.jsonl').read_text().splitlines()]
    if len(rows)!=1284 or len({r['id'] for r in rows})!=1284:raise ValueError('Complete unique parent trial ledger required')
    # Every referenced proof/cache must be manifest-bound, including rejected
    # requests' setup evidence. Relative paths are scoped to original native/.
    for row in rows:
        for f,h in (('context_resolution_file','context_resolution_sha256'),('connection_file','connection_sha256')):
            name='native/'+row[f]
            if manifest.get(name)!=row[h]:raise ValueError('Parent row proof binding differs')
        if row.get('native_release'):
            name='native/'+row['native_cache_file']
            if manifest.get(name)!=row['native_cache_sha256']:raise ValueError('Parent native cache binding differs')
    bindings=dict(parent_contract_fingerprint=fp,parent_report_sha256=sha(study/'report.json'),
        parent_contract_sha256=sha(study/'contract.json'),parent_manifest_sha256=sha(study/'evidence-manifest.json'),
        parent_audit_sha256=sha(audit),parent_checks=346485,parent_rows_sha256=manifest['rows.jsonl'],
        parent_evidence_sha256=manifest)
    return dict(spec=spec,report=report,rows=rows,bindings=bindings)


def make_plan(rows):
    """Literal96, frozen before any new native label or admission result."""
    plan=[]
    def request(source,context,variant,phase,repeat,*,values=None,kind=None,swapped=()):
        native=source.get('native_release') is True
        if kind is None:kind='exact_replay' if native else 'historical_rejection'
        role=source['role'] if phase.startswith('controls_') else 'diagnostic'
        r=dict(id=f'diag64-{phase}-{context}-{variant}-r{repeat}',phase=phase,role=role,
            context_id=context,variant=variant,kind=kind,repeat=repeat,substeps=64,
            origin_method=source['method'],parent_record_id=source['id'],parent_record_sha256=digest(source),
            bucket_xy_m=vector(source['bucket_xy_m'],2),values=vector(source['values'] if values is None else values,5),
            original_native_release=native,swapped_coordinates=list(swapped),selection_or_learning=False,
            source_teacher_id=source['source_teacher_id'],observation=vector(source['observation'],9))
        r['command_values']=copy.deepcopy(r['values'])
        for k in ('source_hint','effective_source_hint','context_resolution_file','context_resolution_sha256',
                  'context_home_sha256','context_layout_eligible','context_anchor_admitted','context_hard_reject'):
            r[k]=copy.deepcopy(source[k])
        if kind=='exact_replay':
            for k in ('q10','q14'):r[k]=vector(source[k],6)
            r.update(connection_file=source['connection_file'],connection_sha256=source['connection_sha256'],
                     parent_native_cache_file=source['native_cache_file'],parent_native_cache_sha256=source['native_cache_sha256'])
        if kind=='historical_rejection':r['historical_rejection_reason']=source['invalid']
        return r
    controls=[]
    for role in ('positive_control','negative_control'):
        found=[r for r in rows if r['phase']=='controls_pre' and r['role']==role and r.get('repeat')==0]
        if len(found)!=1 or not found[0].get('native_release') or bool(captured(found[0]))!=(role=='positive_control'):
            raise ValueError('Exact native parent control source required')
        controls.append(found[0])
    def control_phase(phase):
        for source in controls:
            for repeat in range(3):plan.append(request(source,source['role'],'control_exact',phase,repeat))
    control_phase('controls_pre')
    for code in CONTEXTS:
        context='policy64-test-'+code;pair={}
        for method in ('bc','ppo'):
            found=[r for r in rows if r['role']=='test' and r['method']==method and r['context_id']==context]
            if len(found)!=1:raise ValueError('One exact parent BC/PPO request per diagnostic context required')
            pair[method]=found[0]
        bc,ppo=pair['bc'],pair['ppo']
        if not captured(bc) or captured(ppo) or bc['bucket_xy_m']!=ppo['bucket_xy_m']:
            raise ValueError('Prespecified historical BC-only capture pairing changed')
        for key in ('source_hint','effective_source_hint','context_resolution_file','context_resolution_sha256','context_home_sha256'):
            if bc[key]!=ppo[key]:raise ValueError('Paired methods lost common geometry-only anchor')
        for method in ('bc','ppo'):
            for repeat in range(3):plan.append(request(pair[method],context,method+'_exact','diagnostic',repeat))
        for variant,indices in SWAPS:
            values=vector(ppo['values'],5)
            for index in indices:values[index]=bc['values'][index]
            plan.append(request(ppo,context,variant,'diagnostic',0,values=values,kind='coordinate_swap',swapped=indices))
    control_phase('controls_post')
    if len(plan)!=96 or len({r['id'] for r in plan})!=96:raise RuntimeError('Literal96 unique requests required')
    return plan


def load_native(source,spec,assets):
    """Authenticate old snapshot first; only then import its unchanged modules."""
    source,assets=Path(source).resolve(),Path(assets)
    for name,expected in spec['source_sha256'].items():
        if sha(contained(source,name))!=expected:raise ValueError('Frozen native package differs: '+name)
    for name,expected in spec['identity']['asset_sha256'].items():
        if sha(contained(assets.parent,name))!=expected:raise ValueError('Frozen asset differs: '+name)
    for name,expected in spec['packages'].items():
        if importlib.metadata.version(name)!=expected:raise ValueError('Pinned native dependency differs: '+name)
    for name,module in tuple(sys.modules.items()):
        if name=='sircus' or name.startswith('sircus.'):
            filename=getattr(module,'__file__',None)
            if filename and not Path(filename).resolve().is_relative_to(source):
                raise RuntimeError('Run standalone: a different sircus package is already imported')
    sys.path.insert(0,str(source))
    admission=importlib.import_module('sircus.mjlab.calibrated_admission')
    env=importlib.import_module('sircus.mjlab.calibrated_env')
    bank=importlib.import_module('sircus.mjlab.bank_geometry')
    metric=importlib.import_module('sircus.mjlab.xy650_metrics')
    lc=importlib.import_module('sircus.mjlab.learning_contract')
    options=importlib.import_module('sircus.mjlab.expand_geometry')
    gpu=importlib.import_module('sircus.mjlab.gpu_slot')
    return SimpleNamespace(GeometryResolver=admission.GeometryResolver,verify_resolution=admission.verify_resolution,
        verify_connection=admission.verify_connection,source_geometry=admission.source_geometry,
        Environment=env.CalibratedEnvironment,fixed_endpoints=bank.fixed_endpoints,first_flight=metric.first_flight,
        constraints=lc.constraints,bounds=options.WIDE_BOUNDS,gpu=gpu)


def copy_proof(parent,native,name,expected):
    original=contained(Path(parent)/'native',name)
    if sha(original)!=expected:raise ValueError('Exact parent proof changed before copy')
    destination=Path(native)/name;destination.parent.mkdir(parents=True,exist_ok=True)
    if destination.exists():
        if sha(destination)!=expected:raise ValueError('Refusing to replace different evidence')
    else:
        with destination.open('xb') as stream:stream.write(original.read_bytes())
    return dict(read(destination),file=name,sha256=expected,cache_hit=True)


def rejected(request,reason):
    row=copy.deepcopy(request)
    row.pop('q10',None);row.pop('q14',None)
    row.update(native_release=False,invalid=reason,invalid_physics=False,route_success=False,route_quality=0,
        success=False,complete_success=False,board_bounce=False,whole_ball_entry=False,
        max_retention_s=0.,closest_opening_distance_m=2.,external_contact_steps=0,
        single_bounce_success=False,action_substituted=False,geometry_eligible=False)
    row['motion_admitted']=False
    return row


def control_gate(rows):
    positive=[r for r in rows if r['role']=='positive_control'];negative=[r for r in rows if r['role']=='negative_control']
    good=(len(rows)==6 and len(positive)==len(negative)==3 and all(captured(r) for r in positive)
          and all(r.get('native_release') and not captured(r) and not r.get('whole_ball_entry') for r in negative)
          and all(not r.get('invalid') and not r.get('invalid_physics') and r.get('scalar_score_audit_passed') is True for r in rows))
    return dict(status='passed' if good else 'failed',counts=counts(rows))


def run(args,*,parent=None,runtime=None):
    """Native execution entry; injectable runtime is solely for CPU mocks."""
    started=time.monotonic();repository=Path(__file__).resolve().parents[2]
    parent=parent or authenticate_parent(args.parent_study,args.parent_audit,
        repository/'baselines/mjlab-policy64-20261105/summary.json')
    spec=parent['spec'];plan=make_plan(parent['rows'])
    analysis_path=Path(args.analysis);analysis=read(analysis_path)
    if analysis.get('version')!=ANALYSIS_VERSION or analysis.get('status')!='completed':
        raise ValueError('Completed frozen post-hoc analysis required')
    analysis_source=analysis.get('source',{})
    for key,expected in dict(contract_fingerprint=parent['bindings']['parent_contract_fingerprint'],
            report_sha256=parent['bindings']['parent_report_sha256'],
            manifest_sha256=parent['bindings']['parent_manifest_sha256'],
            original_audit_sha256=parent['bindings']['parent_audit_sha256'],
            rows_sha256=parent['bindings']['parent_rows_sha256'],substeps=64).items():
        if analysis_source.get(key)!=expected:raise ValueError('CPU analysis parent binding differs: '+key)
    analysis_sha=sha(analysis_path)
    runtime=runtime or load_native(args.native_source,spec,args.assets)
    output=Path(args.output);output.mkdir(parents=True,exist_ok=False);native=output/'native';native.mkdir()
    with (output/'analysis.json').open('xb') as stream:stream.write(analysis_path.read_bytes())
    if sha(output/'analysis.json')!=analysis_sha:raise ValueError('Frozen analysis bytes changed during copy')
    for name in ('calibration.json','verification.json','setup-home.json','prior.json'):
        shutil.copyfile(contained(args.parent_study,name),output/name)
    bindings=copy.deepcopy(parent['bindings']);bindings['analysis_sha256']=analysis_sha
    bindings['parent_records_sha256']={r['parent_record_id']:r['parent_record_sha256'] for r in plan}
    atomic(output/'source-bindings.json',bindings)
    contract=dict(version=VERSION,scope=SCOPE,substeps=64,maximum_requests=96,maximum_acquired_s=3600,
        contexts=list(CONTEXTS),swaps=[dict(variant=v,coordinates=list(i)) for v,i in SWAPS],
        analysis_sha256=analysis_sha,parent_bindings=bindings,plan=plan,scene=spec['scene'],identity=spec['identity'],
        native_source_sha256=spec['source_sha256'],worker_sha256=sha(__file__),packages=spec['packages'],
        diagnostic_source_sha256={name:sha(repository/name) for name in (
            'sircus/mjlab/policy64_diagnosis.py','sircus/mjlab/policy64_diagnosis_native.py',
            'sircus/mjlab/policy64_diagnosis_verify.py')},
        created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),training=False,noise=False)
    contract['fingerprint']=digest(contract);atomic(output/'contract.json',contract)
    report=dict(version=VERSION,status='running',phase='frozen',scope=SCOPE,contract_fingerprint=contract['fingerprint'],
        analysis_sha256=analysis_sha,counts=counts([]),control_gates={},contexts=[],training=False,substeps=64)
    rows=[];env=None;packet=0;options_bytes=None
    def publish(phase):
        report.update(phase=phase,counts=counts(rows),reported_worker_elapsed_s=time.monotonic()-started)
        atomic(output/'report.json',report)
        print(json.dumps(dict(phase=phase,**report['counts'])),flush=True)
    def limits():
        if len(rows)>96 or time.monotonic()-started>min(args.wall_time_limit,3600)-10:raise TimeoutError('Own finite diagnosis budget exceeded')
        if shutil.disk_usage(output).free<4*1024**3:raise RuntimeError('FourGiB evidence reserve reached')
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
            cohort=execute(plan[6+12*index:18+12*index],'diagnostic')
            report['contexts'].append(dict(context_id='policy64-test-'+code,
                variants={v:counts([r for r in cohort if r['variant']==v]) for v in dict.fromkeys(r['variant'] for r in cohort)}))
        after=execute(plan[-6:],'controls_post');report['control_gates']['post']=control_gate(after)
        if report['control_gates']['post']['status']!='passed':raise RuntimeError('Final matched native controls failed')
        if len(rows)!=96:raise RuntimeError('Complete96-request accounting required')
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
    for key in ('parent-study','parent-audit','native-source','assets','analysis','output','gpu-lock'):
        parser.add_argument('--'+key,type=Path,required=True)
    parser.add_argument('--wall-time-limit',type=float,default=3600)
    parser.add_argument('--queue-timeout',type=float,default=1800)
    args=parser.parse_args()
    if not 60<=args.wall_time_limit<=3600 or not 0<=args.queue_timeout<=1800:raise ValueError('Bounded4090-only diagnosis required')
    if os.environ.get('CUDA_VISIBLE_DEVICES')!=GPU_UUID:raise RuntimeError('Only authorized4090 UUID may execute')
    # Import only the immutable lock helper before acquiring a slot. Parent
    # authentication/native import isolation is repeated by run after the lock.
    source=Path(args.native_source).resolve();sys.path.insert(0,str(source))
    repository=Path(__file__).resolve().parents[2]
    published=read(repository/'baselines/mjlab-policy64-20261105/summary.json')['provenance']
    expected=published['executed_source_sha256']['sircus/mjlab/gpu_slot.py']
    if sha(source/'sircus/mjlab/gpu_slot.py')!=expected:raise ValueError('Frozen native scheduler bytes differ')
    gpu=importlib.import_module('sircus.mjlab.gpu_slot')
    if Path(gpu.__file__).resolve()!=source/'sircus/mjlab/gpu_slot.py':raise RuntimeError('Wrong native scheduler source')
    if gpu.GPU_UUID!=GPU_UUID:raise RuntimeError('Native scheduler GPU identity differs')
    def deadline(signum,frame):raise TimeoutError('Own acquired diagnostic deadline reached')
    with gpu.gpu_slot(args.gpu_lock,timeout_s=args.queue_timeout,notify=lambda r:print(json.dumps(r),flush=True)):
        args.output.parent.mkdir(parents=True,exist_ok=True)
        atomic(args.output.with_name(args.output.name+'-budget.json'),dict(acquired_unix_s=time.time(),limit_s=args.wall_time_limit))
        signal.signal(signal.SIGALRM,deadline);signal.setitimer(signal.ITIMER_REAL,args.wall_time_limit-5)
        try:run(args)
        finally:signal.setitimer(signal.ITIMER_REAL,0)


if __name__=='__main__':main()
