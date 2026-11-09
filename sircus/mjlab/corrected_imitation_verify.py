"""Independent CPU audit of corrected-recipe imitation and frozen paired TEST.

The producer's selection, split, fit and summary functions are not imported.
Dense native traces and exact setup proofs use the existing independent oracle.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import time
import numpy as np
from . import benchmark_verify as common
from . import calibrated_benchmark_verify as calibrated
from . import coverage_training_verify as geometry
from . import coverage_recovery_verify as previous
from . import policy64_verify as models
from . import replay650_verify as replay
from . import table_training_verify as native
from . import table_assisted_score_oracle as oracle
from .benchmark_verify import EvidenceError, contained, digest, read, sha

VERSION='sircus-corrected-imitation-audit-v1'
STUDY_VERSION='sircus-corrected-imitation-v1'
METHODS=('corrected_bc','current_bc')
ROLES=('direct_control','table_control','negative_control')
RESIDUAL=np.array([.04,.04,.05,8.,.12])
META=previous.META
AuditChecks=replay.AuditChecks
PINS={
 'parent':('6a255223820656059fe16a5de2944ba2593f7ac3603b287f4955979cd4677504','e2ba653792b37d302425ba4af6cb22460c3d5b426274b3c1b3cbce270a857a9b','f54c341eba5258b427e4e1eb05a1b2e93bf76518d9f94a2fe8cf6bad744f6082',283251,970),
 'height':('a6f2d560e9bbd0e721f4b3aaaab651a11de967a6bdb221906208876865582931','81d45d57ea47c0ad558102a0cd8377fbd08caff0ce1952ae8b0ae1d57641f50a','74d500fea9c941f4101ad47176b8daaa3cabaabba12e45bd6514e34fd42e1778',22266,63),
 'gap':('dc18617048d35a60665974d9733e9efb3d66e5682a55c61ed3139bcbda64fd35','8c11bc5b12371c128df092725fe1f1e0c62c5ad42defadbb80d699e5cbd1c252','103355e9de55b139eca5797804b015e53baad5256567330657951caec45b6ea6',16072,36),
 'ik':('8199b13d944f507e78664a8c78734e92ab56e3db14025d7a93eefd4730a6ecd2','b5b65c8292147458cc79650638c42d17c63a71b302b228bff57b998234068d60','f80dc7fa8f0b535bf5d03040076e9bc0199f4781b2be3dd3ce7c8333f913c8d2',25329,66)}
SELECTION=(('height','recovery-test-r3-c1-n0','corrected'),('gap','recovery-test-r0-c5-n1','corrected'),('gap','recovery-test-r0-c6-n1','corrected'),('ik','recovery-test-r1-c1-n1','parent_xy'),('ik','recovery-test-r2-c4-n0','parent_xy'))
CURRENT_SHA='688acc7981522c3281dff47edd19edaf46f084f113455e50f14d7e5cabc3b9e9'
CURRENT_WEIGHTS='af319655875b021635e8e1778b1e10f4ebaba24291160504fb88a4c6a101b10c'


def authenticate(directory,audit,kind,checks):
    directory,audit=Path(directory),Path(audit);rs,aps,fp,count,nrows=PINS[kind]
    checks.equal((sha(directory/'report.json'),sha(audit)),(rs,aps),kind+'/exact reviewed report and audit bytes')
    report,spec,proof,manifest=map(read,(directory/'report.json',directory/'contract.json',audit,directory/'evidence-manifest.json'))
    unsigned=dict(spec);saved_fp=unsigned.pop('fingerprint')
    checks.equal((digest(unsigned),saved_fp,report['contract_fingerprint'],proof['contract_fingerprint']),(fp,)*4,kind+'/exact experiment identity')
    checks.equal((report['status'],proof['status'],proof['full_checks_verified'],proof['native_checks_verified'],proof['failures']),('completed','passed',True,True,[]),kind+'/complete independent native audit')
    checks.equal((proof['checks_count'],len(proof['checks'])),(count,count),kind+'/all audit checks present')
    checks.require(all(c.get('passed') is True for c in proof['checks']),kind+'/all audit checks passed')
    checks.equal((proof['report_sha256'],proof['evidence_manifest_sha256']),(rs,sha(directory/'evidence-manifest.json')),kind+'/proof binds full evidence')
    checks.equal(report['evidence_sha256'],manifest,kind+'/report binds manifest')
    consumed={}
    def bound(name):
        path=contained(directory,name);value=sha(path);checks.equal(value,manifest[name],kind+'/consumed evidence/'+name);consumed[name]=value;return path
    rows=common.lines(bound('rows.jsonl'));checks.equal(len(rows),nrows,kind+'/complete historical denominator')
    checks.equal(len({r['id'] for r in rows}),nrows,kind+'/unique historical IDs')
    bound('contract.json')
    return dict(directory=directory,report=report,spec=spec,manifest=manifest,rows=rows,bound=bound,consumed=consumed,report_sha256=rs,audit_sha256=aps,manifest_sha256=sha(directory/'evidence-manifest.json'))


def authenticate_row(source,row,checks):
    checks.require(row in source['rows'],'Selected teacher comes from authenticated historical ledger')
    namespace=row.get('adapter_directory','native')
    checks.require(namespace in ('native','native-controls'),'Known proof namespace')
    for prefix in ('connection','context_resolution'):
        checks.equal(sha(source['bound'](namespace+'/'+row[prefix+'_file'])),row[prefix+'_sha256'],row['id']+'/exact '+prefix+' evidence')
    if row.get('native_release'):
        checks.equal(sha(source['bound'](namespace+'/'+row['native_cache_file'])),row['native_cache_sha256'],row['id']+'/exact native dense trajectory')


def selected_rows(sources,checks):
    """One literal recipe per corrected context, with all three confirming rows."""
    selected=[]
    for kind,context,variant in SELECTION:
        source=sources[kind]
        group=[r for r in source['rows'] if r.get('phase')=='diagnostic' and r.get('context_id')==context and r.get('variant')==variant]
        checks.equal([r.get('repeat') for r in group],[0,1,2],context+'/exact three nominal confirmations')
        checks.require(all(native.captured(r) for r in group),context+'/all three corrected drops captured')
        first=group[0]
        original=[r for r in sources['parent']['rows'] if r.get('phase')=='test' and r.get('method')=='recovery_bc' and r.get('context_id')==context]
        checks.equal(len(original),1,context+'/one matched current-policy source');original=original[0]
        checks.require(not native.captured(original),context+'/correction of an inspected current-policy loss')
        for row in group:
            authenticate_row(source,row,checks)
            for field in ('values','bucket_xy_m','q10','q14','connection_sha256','context_resolution_sha256'):
                checks.equal(row[field],first[field],row['id']+'/unchanged nominal repeat/'+field)
            checks.equal(row['source_record_id'],original['id'],row['id']+'/matched current-policy lineage')
            checks.equal(row['source_record_sha256'],digest(original),row['id']+'/unchanged parent row hash')
            for field in META:checks.equal(row[field],original[field],row['id']+'/original geometry-only anchor/'+field)
        residual=(np.asarray(first['values'],float)-np.asarray(original['effective_source_hint']['values'],float))/RESIDUAL
        checks.require(residual.shape==(5,) and np.isfinite(residual).all() and (abs(residual)<1).all(),context+'/exact correction lies strictly inside actor support')
        checks.numeric(original['observation'],calibrated.expected_observation(first['bucket_xy_m'],original['effective_source_hint']),context+'/independent corrected-teacher observation',0)
        selected.append((kind,first,original,residual,group))
    checks.equal(len({x[1]['context_id'] for x in selected}),5,'Five distinct corrected contexts without repeat weighting')
    return selected


def regenerated_test(exclusions):
    used=np.array(sorted(set(tuple(map(float,p)) for p in exclusions)),float)
    if used.ndim!=2 or used.shape[1]!=2 or not len(used) or not np.isfinite(used).all():raise EvidenceError('Finite inventoried coordinates required')
    rng=np.random.default_rng(2026110941);result=[]
    for iy in range(5):
        for ix in range(7):
            for n in range(2):
                for _ in range(10000):
                    xy=np.array([.1,-.65])+(np.array([1.24,.25])-np.array([.1,-.65]))/[7,5]*([ix,iy]+rng.uniform(.05,.95,2))
                    if np.linalg.norm(used-xy,axis=1).min()>=.005:break
                else:raise EvidenceError('Fresh unfiltered coordinate generation exhausted')
                result.append(dict(id=f'corrected-imitation-test-r{iy}-c{ix}-n{n}',split='test',bucket_xy_m=xy.tolist(),grid_cell=[ix,iy]))
                used=np.vstack((used,xy))
    return result


def protocol_checks(value,checks):
    expected=dict(substeps=64,methods=list(METHODS),test_seed=2026110941,control_requests=18,test_layouts=70,test_requests=140,
        maximum_requests=158,actor_steps=800,critic_steps=0,actor_lr=.001,base_teacher_count=92,correction_teacher_count=5,
        teacher_count=97,noise=False,residual_scale=RESIDUAL.tolist(),worker_limit_s=3600,queue_limit_s=900,
        global_limit_s=6300,audit_reserve_s=1800,storage_reserve_bytes=8*1024**3,minimum_split_distance_m=.005)
    for key,value_expected in expected.items():checks.equal(value[key],value_expected,'Literal corrected-imitation protocol/'+key)


def reconstructed_teachers(sources,checks):
    parent=sources['parent'];old=read(parent['bound']('bootstrap/teachers.json'))
    checks.equal((len(old),len({r['context_id'] for r in old})),(92,92),'All92 distinct prior teachers preserved')
    checks.require(all(native.captured(r) and r['role']=='train' for r in old),'Only original captured training teachers')
    corrected=[];lineage=[]
    for kind,first,original,residual,group in selected_rows(sources,checks):
        teacher=copy.deepcopy(first)
        teacher.update(id='corrected-teacher-'+first['context_id'],role='train',original_role=original['role'],source_correction_role=first['role'],development_reclassified=True,
            normalized_residual=residual.tolist(),observation=copy.deepcopy(original['observation']),learning_reward=oracle.reward(first),
            source_correction_record_id=first['id'],source_correction_record_sha256=digest(first),
            correction_repeat_ids=[r['id'] for r in group],correction_repeat_sha256={r['id']:digest(r) for r in group})
        corrected.append(teacher)
        lineage.append(dict(context_id=first['context_id'],study=kind,variant=first['variant'],repeat=0,
            teacher_id=teacher['id'],teacher_sha256=digest(teacher),source_record_id=original['id'],source_record_sha256=digest(original),
            correction_record_id=first['id'],correction_record_sha256=digest(first),confirmation_record_ids=[r['id'] for r in group],
            confirmation_record_sha256={r['id']:digest(r) for r in group}))
    union=sorted(copy.deepcopy(old)+corrected,key=lambda r:r['context_id'])
    checks.equal((len(union),len({r['context_id'] for r in union}),len({r['id'] for r in union})),(97,97,97),'Exactly97 distinct teachers, no duplicated nominal repeats')
    return old,corrected,union,lineage


def audit_inputs(study,sources,spec,checks):
    parent=sources['parent'];data={}
    for key,name in (('prior','prior.json'),('old_prior','old-prior.json'),('controls','controls.json'),('control_replays','control-replays.json'),('calibration','calibration.json'),('home','setup-home.json')):
        data[key]=read(parent['bound'](name));checks.equal(read(study/name),data[key],'Exact current policy input/'+name)
    checks.equal(sha(study/'verification.json'),sha(parent['bound']('verification.json')),'Original calibration verification unchanged')
    checks.equal((len(data['prior']['records']),len(data['old_prior']['records'])),(19,17),'Same19 policy and17 control priors')
    checks.equal((spec['prior_sha256'],spec['old_prior_sha256']),(digest(data['prior']),digest(data['old_prior'])),'Prior identity frozen before evaluation')
    old,corrected,teachers,lineage=reconstructed_teachers(sources,checks)
    checks.equal(read(study/'bootstrap/teachers.json'),teachers,'Exact92 original plus5 independently reconstructed corrected teachers')
    checks.equal(read(study/'bootstrap/correction-teachers.json'),corrected,'Only the five literal correction teachers')
    checks.equal(read(study/'bootstrap/original-teachers.json'),old,'All92 original teacher records unchanged')
    boot=read(study/'bootstrap/source-bindings.json')
    for key,value in dict(parent_teacher_ids=[r['id'] for r in old],parent_teacher_sha256={r['id']:digest(r) for r in old},
        new_teacher_ids=[r['id'] for r in corrected],teacher_file_sha256=sha(study/'bootstrap/teachers.json'),teacher_lineage=lineage).items():
        checks.equal(boot[key],value,'Independent teacher outcome provenance/'+key)
    checks.equal(spec['teacher_file_sha256'],sha(study/'bootstrap/teachers.json'),'Exact97 teacher file bound before fitting')
    for kind,source in sources.items():
        if kind!='parent':checks.equal(source['spec']['identity'],parent['spec']['identity'],kind+'/same original simulation identity')
    bindings=read(study/'source-bindings.json');checks.equal(spec['source_bindings'],bindings,'Contract binds source identities and teachers')
    checks.equal(bindings['teacher_lineage'],lineage,'Every teacher bound to its three captured historical trials')
    checks.equal(bindings['old_teacher_file_sha256'],sha(parent['bound']('bootstrap/teachers.json')),'Original teacher file unmodified')
    checks.equal(bindings['old_teacher_sha256'],{r['id']:digest(r) for r in old},'All92 original teacher row bytes retained')
    for kind,source in sources.items():
        b=bindings['parent'] if kind=='parent' else bindings['corrections'][kind]
        rs,aps,fp,count,_=PINS[kind]
        for key,value in dict(report_sha256=rs,audit_sha256=aps,contract_fingerprint=fp,evidence_manifest_sha256=source['manifest_sha256'],checks_count=count).items():
            checks.equal(b[key],value,kind+'/literal source binding/'+key)
        for name,expected in b['files_sha256'].items():checks.equal(sha(source['bound'](name)),expected,kind+'/every claimed source file authenticated/'+name)
    names={'effective-options.json','native-compiled-geometry.json'}
    for row in data['control_replays']:
        for field in ('context_resolution','connection'):
            name=row[field+'_file'];names.add(name);checks.equal(sha(parent['bound']('control-parent/native/'+name)),row[field+'_sha256'],'Exact original17 control proof')
    mirror={name:sha(parent['bound']('control-parent/native/'+name)) for name in sorted(names)}
    checks.equal(read(study/'control-parent/mirror.json'),mirror,'Exact historical control mirror inventory')
    for name,expected in mirror.items():checks.equal(sha(contained(study/'control-parent/native',name)),expected,'Historical control mirror bytes/'+name)
    return data,old,corrected,teachers


def audit_inventory(study,source,history_root,spec,teachers,checks):
    inventory=read(study/'historical-coordinate-bindings.json');coordinates=set();names={'bucket_xy_m','bucket_xy','exclusions','excluded','coordinates','excluded_coordinates'}
    def collect(item,enabled=False):
        if enabled and isinstance(item,list) and len(item)==2 and all(type(x) in (int,float) and math.isfinite(x) for x in item):coordinates.add(tuple(map(float,item)));return
        if isinstance(item,dict):
            for key,value in item.items():collect(value,key in names)
        elif isinstance(item,list):
            for value in item:collect(value,enabled)
    checks.require(bool(inventory['files_sha256']),'Frozen historical coordinate inventory required')
    for name,expected in inventory['files_sha256'].items():
        prefix,relative=name.split('/',1);checks.require(prefix in ('output','baselines'),'Known historical coordinate root')
        path=contained(history_root if prefix=='output' else source/'baselines',relative)
        checks.require(not path.resolve().is_relative_to(study.resolve()),'New study excluded from own history')
        checks.equal(sha(path),expected,'Exact inventoried coordinate source/'+name)
        if path.suffix=='.jsonl':
            for row in common.lines(path):collect(row)
        else:collect(read(path),'exclusion' in path.name)
    expected_coordinates=[list(p) for p in sorted(coordinates)]
    checks.equal(inventory['coordinates'],expected_coordinates,'All observed coordinates retained without success filtering')
    checks.require(all(tuple(map(float,t['bucket_xy_m'])) in coordinates for t in teachers),'All97 training coordinates present in historical exclusion set')
    test=regenerated_test(expected_coordinates)
    checks.equal(read(study/'splits.json'),dict(train=[dict(id=t['context_id'],bucket_xy_m=t['bucket_xy_m'],split='train') for t in teachers],test=test),'Literal97 training and stratified70 fresh unfiltered bucket positions')
    checks.equal(spec['historical_coordinates_sha256'],sha(study/'historical-coordinate-bindings.json'),'Historical inventory bound before outcomes')
    checks.equal(spec['splits'],read(study/'splits.json'),'Split bound before outcomes')
    return dict(files=len(inventory['files_sha256']),historical_coordinates=len(coordinates),fresh_test_layouts=70)


def audit_models(study,report,spec,parent,teachers,checks):
    artifacts=report['artifacts'];checks.equal(set(artifacts),set(METHODS),'Exactly corrected and current frozen imitation models')
    current=artifacts['current_bc'];original=parent['report']['artifacts']['recovery_bc']
    checks.equal(current['sha256'],CURRENT_SHA,'Actual current37/70 recovery checkpoint')
    checks.equal(current['weight_sha256'],CURRENT_WEIGHTS,'Actual current recovery weights')
    checks.equal(sha(parent['bound'](original['path'])),CURRENT_SHA,'Current source checkpoint bytes unchanged')
    for key in ('sha256','weight_sha256','samples','seed'):checks.equal(current.get(key),original.get(key),'Exact current artifact/'+key)
    checks.equal(spec['old_artifacts']['current_bc'],current,'Current model bound before fitting')
    states={}
    for method,artifact in artifacts.items():
        checks.equal(artifact['path'],'checkpoints/'+method+'.pt','Literal frozen model path/'+method)
        path=contained(study,artifact['path']);checks.equal(sha(path),artifact['sha256'],'Frozen file bytes/'+method)
        states[method]=common.load_checkpoint(path,checks,method)
        for key in ('weight_sha256','samples','seed'):checks.equal(states[method].get(key),artifact.get(key),'Artifact content/'+method+'/'+key)
    checks.equal((artifacts['current_bc']['samples'],artifacts['corrected_bc']['samples']),(92,97),'True old and expanded training sample counts')
    fit=read(study/'fit.json');checks.equal(report['fit'],fit,'Reported fit is exact saved optimizer evidence')
    checks.equal((fit['parent_teacher_count'],fit['new_teacher_count']),(92,5),'Only five additional corrected teachers')
    checks.equal(fit['status'],'actor_updated','Fixed actor-only refit happened')
    checks.equal(states['corrected_bc']['contract_fingerprint'],spec['fingerprint'],'New weights bind frozen protocol')
    before,after=states['current_bc']['policy'],states['corrected_bc']['policy']
    result=previous.actor_finetune_replay(before,after,teachers,fit,checks)
    checks.equal((fit['before'],fit['after']),(common.weights_sha256(before),common.weights_sha256(after)),'True optimizer weight identities')
    critic=lambda state:common.weights_sha256({k.removeprefix('critic.'):v for k,v in state.items() if k.startswith('critic.')})
    checks.equal((fit['critic_before_sha256'],fit['critic_after_sha256']),(critic(before),critic(before)),'Critic unchanged')
    checks.equal(fit['critic_row_count'],97,'Descriptive unchanged critic loss rows')
    checks.numeric(fit['initial_critic_loss'],fit['final_critic_loss'],'No critic optimization',0)
    return states,result


def audit_test(study,rows,report,spec,states,checks):
    packets=[];comparison=[];positions=[]
    for method in METHODS:
        packet=read(study/f'predictions/test-{method}.json');packets.append(packet);artifact=report['artifacts'][method]
        group=[r for r in rows if r['phase']=='test' and r['method']==method]
        checks.equal((packet['method'],packet['role'],packet['artifact']),(method,'test',artifact),'Frozen method packet/'+method)
        checks.equal((packet['one_prediction_per_layout'],packet['selection_or_learning']),(True,False),'No adaptive TEST selection/'+method)
        checks.equal((len(packet['rows']),len(group)),(70,70),'All70 requested outcomes per frozen policy/'+method)
        for prediction,row,context in zip(packet['rows'],group,spec['splits']['test']):
            label=row['id'];checks.equal(label,f'corrected-imitation-test-{method}-{context["id"]}',label+'/literal test record ID')
            checks.equal((prediction['context_id'],prediction['bucket_xy_m']),(context['id'],context['bucket_xy_m']),label+'/unchanged held-out layout')
            for key,value in prediction.items():checks.equal(row.get(key),value,label+'/unmodified frozen prediction/'+key)
            checks.equal((prediction['artifact_sha256'],row['frozen_artifact_sha256']),(artifact['sha256'],)*2,label+'/frozen checkpoint binding')
            checks.equal((prediction['source_training_ids'],prediction['ball_flight_search_at_inference'],prediction['geometry_search_at_inference'],prediction['selection_or_learning']),([],False,True,False),label+'/geometry-only anchor resolution and one actor inference')
            checks.numeric(prediction['observation'],calibrated.expected_observation(context['bucket_xy_m'],prediction['effective_source_hint']),label+'/independent geometry observation',0)
            previous.audit_actor_transform(prediction,states[method],checks,label)
            checks.equal((row['values'],row['perturbation_mixed_units']),(row['command_values'],[0.]*5),label+'/exact unmodified noiseless command')
            checks.equal(row['test_plan_sha256'],sha(study/'test-plan.json'),label+'/all140 predictions bound before TEST')
            positions.append(dict(context_id=row['context_id'],bucket_xy_m=row['bucket_xy_m'],method=method,capture=native.captured(row),clean=native.captured(row) and row.get('route_quality')==2,native_release=row['native_release'],invalid=row.get('invalid'),invalid_physics=bool(row.get('invalid_physics')),direct_capture=row.get('direct_capture',False),table_assisted_capture=row.get('table_assisted_capture',False),forbidden_contact_steps=row.get('forbidden_contact_steps',0),values=row['values'],record_id=row['id']))
        comparison.append(dict(method=method,counts=native.statistics(group),artifact_sha256=artifact['sha256']))
    checks.equal(read(study/'test-plan.json'),dict(methods=packets,frozen_before_outcomes=True),'Both full inference packets frozen together')
    checks.equal(report['test_comparison'],comparison,'Independent paired denominator comparison')
    checks.equal(report['test_positions'],positions,'Every held-out position result reproduced')
    by={r['context_id']:{m:next(p for p in positions if p['context_id']==r['context_id'] and p['method']==m) for m in METHODS} for r in positions}
    gains=sum(v['corrected_bc']['capture'] and not v['current_bc']['capture'] for v in by.values())
    losses=sum(v['current_bc']['capture'] and not v['corrected_bc']['capture'] for v in by.values())
    paired=dict(paired_positions=70,both_capture=sum(all(p['capture'] for p in v.values()) for v in by.values()),
        corrected_only=gains,current_only=losses,both_fail=sum(not any(p['capture'] for p in v.values()) for v in by.values()),net_captures=gains-losses)
    checks.equal(report['paired_comparison'],paired,'Every paired gain and regression independently counted')
    for i in range(70):
        for key in META:checks.equal(packets[0]['rows'][i][key],packets[1]['rows'][i][key],str(i)+'/shared geometry-only setup/'+key)
    return comparison


def audit_controls(rows,sources,originals,checks):
    gates={}
    for phase,key in (('controls_pre','pre'),('controls_post','post')):
        group=[r for r in rows if r['phase']==phase];checks.equal(len(group),9,phase+'/nine prescribed controls')
        for i,(source,original,role) in enumerate(zip(sources,originals,ROLES)):
            for repeat in range(3):
                row=group[3*i+repeat];label=row['id']
                checks.equal(label,f'corrected-imitation-{phase}-{role}-r{repeat}',label+'/literal control identity')
                checks.equal((row['role'],row['method'],row['repeat']),(role,'control',repeat),label+'/matched control metadata')
                checks.equal((row['control_source_id'],row['control_source_sha256']),(source['id'],digest(source)),label+'/original control provenance')
                checks.equal((row['source_record_id'],row['source_record_sha256'],row['variant']),(original['id'],digest(original),'control_exact'),label+'/exact original17 admitted branch')
                checks.equal((row['values'],row['bucket_xy_m']),(source['values'],source['bucket_xy_m']),label+'/unchanged control action and position')
                for field in (*META,'q10','q14','connection_file','connection_sha256'):
                    checks.equal(row[field],original[field],label+'/original branch evidence/'+field)
                for field in ('q10','q14'):checks.equal(row['native_initial_'+field],np.asarray(original[field],np.float32).astype(float).tolist(),label+'/actual native branch reset/'+field)
        gates[key]=native.control_gate(group);checks.equal(gates[key]['status'],'passed',phase+'/direct/table/negative controls passed')
    return gates


def audit_sequence(study,rows,report,checks):
    expected=[('controls_pre',role,'control') for role in ROLES for _ in range(3)]
    expected += [('test','test',m) for m in METHODS for _ in range(70)]
    expected += [('controls_post',role,'control') for role in ROLES for _ in range(3)]
    checks.equal([(r['phase'],r['role'],r['method']) for r in rows],expected,'Exactly158 requests with no search, retries or omitted rejection')
    checks.equal(len({r['id'] for r in rows}),158,'Unique request IDs')
    checks.require(0<=report['elapsed_s']<3600,'One-hour acquired worker bound')
    boundaries=common.lines(study/'phase-boundaries.jsonl')
    checks.equal([(b['phase'],b['rows_count']) for b in boundaries],[('inputs-frozen',0),('controls_pre-complete',9),('fit-complete',9),('test-plan-frozen',9),('test-complete',149),('controls_post-complete',158)],'Fit and both prediction freezes precede any TEST outcome')
    raw=(study/'rows.jsonl').read_bytes().splitlines(keepends=True);last=0
    for b in boundaries:
        checks.equal(b['rows_sha256'],hashlib.sha256(b''.join(raw[:b['rows_count']])).hexdigest() if b['rows_count'] else None,b['phase']+'/immutable observed prefix')
        checks.require(last<=b['acquired_elapsed_s']<3600,'Ordered bounded phase times');last=b['acquired_elapsed_s']
    timings=common.lines(study/'timing.jsonl');checks.equal(len(timings),8,'Eight exact bounded execution batches')
    checks.require(boundaries[3]['acquired_elapsed_s']<=next(t['acquired_start_s'] for t in timings if t['phase']=='test'),'Both full70 prediction packets precede first TEST simulation')
    checks.require(all(t['acquired_end_s']<3600 for t in timings),'Every native batch within one-hour limit')
    return dict(total_requests=158,test_requests=140,control_requests=18,maximum_requests=158)


def audit_budget(study,report,checks):
    path=study.with_name(study.name+'-budget.json');budget=read(path);pipeline=read(study.with_suffix('.pipeline.json'))
    checks.equal((budget['limit_s'],budget['audit_reserve_s']),(3600,1800),'One-hour4090worker with half-hour independent audit reserve')
    checks.equal(budget['gpu_uuid'],'GPU-ab1950b0-01dd-0fab-407f-2d17936b27f4','Only authorized4090')
    checks.require(0<pipeline['global_limit_s']<=6300,'Pipeline bounded at6300seconds')
    checks.numeric(pipeline['global_deadline_unix_s']-pipeline['started_unix_s'],pipeline['global_limit_s'],'Absolute pipeline interval',.05)
    checks.equal(budget['global_deadline_unix_s'],pipeline['global_deadline_unix_s'],'One deadline shared by supervisor and worker')
    checks.require(pipeline['started_unix_s']<=budget['acquired_unix_s'],'Own lease acquired after supervisor starts')
    checks.numeric(budget['effective_worker_limit_s'],min(3600,budget['global_deadline_unix_s']-budget['acquired_unix_s']-1800),'Actual worker budget after queue',.1)
    checks.require(60<budget['effective_worker_limit_s']<=3600,'Positive bounded worker limit')
    for field in ('effective_worker_limit_s','global_deadline_unix_s'):checks.equal(report['deadlines'][field],budget[field],'Accurate report deadline/'+field)
    checks.equal(report['deadlines']['worker_limit_s'],3600,'Requested maximum worker budget')
    checks.require(report['elapsed_s']<budget['effective_worker_limit_s'],'All completed work within acquired allowance')
    return dict(worker_receipt_sha256=sha(path),effective_worker_limit_s=budget['effective_worker_limit_s'],global_limit_s=pipeline['global_limit_s'])


def audit(study,source,assets,*,parent_study,parent_audit,height_study,height_audit,gap_study,gap_audit,ik_study,ik_audit,history_root,reference):
    started=time.monotonic();checks=AuditChecks();study,source,assets,history_root=map(Path,(study,source,assets,history_root))
    manifest=replay.check_manifest(study,checks);spec=read(study/'contract.json');report=read(study/'report.json')
    unsigned=dict(spec);fp=unsigned.pop('fingerprint');checks.equal((digest(unsigned),report['contract_fingerprint']),(fp,fp),'Immutable full experiment fingerprint')
    checks.equal((spec['version'],report['version'],report['status'],report['phase']),(STUDY_VERSION,STUDY_VERSION,'completed','completed'),'Full versioned corrected-imitation comparison completed')
    checks.equal(report['evidence_sha256'],manifest,'Report binds all raw evidence');checks.equal(report['protocol'],spec['protocol'],'Protocol fixed before outcomes')
    protocol_checks(spec['protocol'],checks)
    replay.check_source_files(source,spec['source_sha256'],checks,'Frozen executed producer source')
    for module in (Path(__file__),Path(oracle.__file__)):
        checks.equal(sha(module),spec['source_sha256']['sircus/mjlab/'+module.name],'Actual independent auditor bound/'+module.name)
    sources={kind:authenticate(directory,proof,kind,checks) for kind,directory,proof in (
        ('parent',parent_study,parent_audit),('height',height_study,height_audit),('gap',gap_study,gap_audit),('ik',ik_study,ik_audit))}
    data,old,corrected,teachers=audit_inputs(study,sources,spec,checks)
    models.audit_runtime_reference(source,reference,spec,sources['parent']['spec'],assets,checks)
    inventory=audit_inventory(study,source,history_root,spec,teachers,checks)
    calibration=replay.check_calibration(study,spec['identity']['calibration'],checks)
    rows=replay.read_records(study/'rows.jsonl');native.check_rows(rows,checks)
    setup,native_count=geometry.audit_native_namespaces(study,rows,data['prior'],data['old_prior'],data['home'],spec,calibration,assets,checks)
    accounting=audit_sequence(study,rows,report,checks);requests=geometry.audit_request_packets(study,rows,checks);budget=audit_budget(study,report,checks)
    states,fit=audit_models(study,report,spec,sources['parent'],teachers,checks)
    comparison=audit_test(study,rows,report,spec,states,checks);gates=audit_controls(rows,data['controls'],data['control_replays'],checks)
    checks.equal(report['control_gates'],gates,'Both independently verified controls')
    checks.equal(report['counts'],native.statistics(rows),'Every requested outcome remains counted')
    checks.equal(report['stage_counts'],{p:native.statistics([r for r in rows if r['phase']==p]) for p in ('controls_pre','test','controls_post')},'Exact per-phase accounting')
    checks.equal((report['base_teachers'],report['new_teachers'],report['teacher_count'],report['expected_requests']),(92,5,97,158),'True preserved and added teacher counts and full requested denominator')
    correction_summary=[]
    for (kind,context,_),row in zip(SELECTION,corrected):
        correction_summary.append(dict(context_id=context,bucket_xy_m=row['bucket_xy_m'],values=row['values'],source_study=kind,
            source_record_id=row['source_correction_record_id'],source_record_sha256=row['source_correction_record_sha256'],
            source_report_sha256=PINS[kind][0],confirmation_captures=3,confirmation_requests=3))
    checks.equal(report['teacher_corrections'],correction_summary,'Reported correction recipes agree with original authenticated trials')
    checks.equal(report['gate'],dict(status='passed',substeps=64,controls_passed=True,complete_test=True,no_policy_promotion=True),'No automatic policy promotion')
    return dict(version=VERSION,status='passed',full_checks_verified=True,native_checks_verified=True,checks_count=checks.count,checks=checks.entries,failures=[],
        contract_fingerprint=fp,report_sha256=sha(study/'report.json'),evidence_manifest_sha256=sha(study/'evidence-manifest.json'),
        auditor_sha256=sha(Path(__file__)),producer_source_sha256=spec['source_sha256'],
        consumed_source_files_sha256={k:s['consumed'] for k,s in sources.items()},requested_trials=len(rows),native_releases=native_count,
        teacher_count=len(teachers),correction_teacher_count=len(corrected),actor_fit=fit,test_comparison=comparison,
        control_gates=gates,setup_evidence=setup,coordinate_inventory=inventory,request_accounting=accounting,request_packets=requests,budget=budget,
        elapsed_s=time.monotonic()-started,maximum_numeric_error=checks.numeric_error,
        limitations=['Five inspected prior TEST positions are now development data; old37/70 remains unchanged.',
            'Fresh70 positions are disjoint from inventoried records; unrecorded history is not certified.',
            'One frozen prediction per model per position, fixed64 held endpoints; no noise,128, executed transit or hardware validation.',
            'Candidate has five extra demonstrations and additional fitting; this is not an equal-training-budget algorithm comparison.'])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('study','source','assets','parent-study','parent-audit','height-study','height-audit','gap-study','gap-audit','ik-study','ik-audit','history-root','reference','output'):parser.add_argument('--'+name,type=Path,required=True)
    a=parser.parse_args()
    if a.output.exists() or a.output.resolve().is_relative_to(a.study.resolve()):raise ValueError('Fresh independent proof outside immutable study required')
    try:
        args=vars(a).copy();output=args.pop('output');result=audit(**args)
    except Exception as error:
        result=dict(version=VERSION,status='failed',failures=[type(error).__name__+': '+str(error)])
        a.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');raise
    a.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:result[k] for k in ('status','checks_count','requested_trials','native_releases','teacher_count','test_comparison')}))

if __name__=='__main__':main()
