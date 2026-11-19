"""Independent CPU evidence audit of bounded demonstration recovery.

Historical TEST is explicitly development data here. New TEST coordinates and
all four prediction packets must be frozen before their outcomes. This verifier
does not run physics, mutate source evidence, or import producer search logic.
"""
from __future__ import annotations

import argparse
from collections import Counter
import copy
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np

from . import benchmark_verify as common
from . import calibrated_benchmark_verify as calibrated
from . import coverage_training_verify as parent_audit
from . import policy64_verify as model_audit
from . import replay650_verify as replay
from . import table_training_verify as native
from . import table_assisted_score_oracle as oracle
from .benchmark_verify import EvidenceError, contained, digest, read, sha

VERSION = 'sircus-coverage-recovery-audit-v1'
STUDY_VERSION = 'sircus-coverage-recovery-v1'
PARENT_REPORT_SHA = '1d9f91dce8aa264120eaefdb532b3dc9b4fc4122b27939de387d8d1e723b0c21'
PARENT_AUDIT_SHA = 'ac54eeede0a1670e802dac028ef6b0a7a9d6b34f4561e558cc53698a8ec5594a'
PARENT_FP = 'ea0e2cab154c6725b477dc84683f9c8eedee3f0dbfb1cd1ffa7ab166785c1eb7'
METHODS = ('recovery_bc', 'parent_bc', 'parent_ppo', 'previous_ppo')
CONTROL_ROLES = ('direct_control', 'table_control', 'negative_control')
RESIDUAL = np.asarray([.04, .04, .05, 8., .12])
AuditChecks = replay.AuditChecks
captured = native.captured
statistics = native.statistics


def authenticate_parent(directory, audit, checks):
    """Reuse the exact full passed proof, and authenticate every consumed byte."""
    directory,audit=Path(directory),Path(audit)
    checks.equal(sha(directory/'report.json'),PARENT_REPORT_SHA,'Exact published parent report')
    checks.equal(sha(audit),PARENT_AUDIT_SHA,'Exact corrected passed parent audit')
    report,spec,proof,manifest=map(read,(directory/'report.json',directory/'contract.json',audit,directory/'evidence-manifest.json'))
    checks.equal((report['version'],report['status'],proof['version'],proof['status'],proof['full_checks_verified'],proof['failures']),
        ('sircus-coverage-training7h-v2','completed','sircus-coverage-training7h-audit-v2','passed',True,[]),'Complete audited parent experiment')
    checks.equal((proof['checks_count'],len(proof['checks'])),(857507,857507),'Every parent audit check retained')
    checks.require(all(c['passed'] is True for c in proof['checks']),'Every parent audit check passed')
    unsigned=dict(spec);fp=unsigned.pop('fingerprint')
    checks.equal((digest(unsigned),fp,report['contract_fingerprint'],proof['contract_fingerprint']),
                 (PARENT_FP,)*4,'Exact parent contract and experiment identity')
    checks.equal(proof['report_sha256'],PARENT_REPORT_SHA,'Parent audit binds report bytes')
    checks.equal(proof['evidence_manifest_sha256'],sha(directory/'evidence-manifest.json'),'Parent audit binds manifest')
    checks.equal(report['evidence_sha256'],manifest,'Parent report binds entire evidence manifest')
    consumed={}
    def bound(name):
        path=contained(directory,name);actual=sha(path)
        checks.equal(actual,manifest[name],'Consumed authenticated parent file/'+name)
        consumed[name]=actual
        return path
    bound('contract.json');rows=common.lines(bound('rows.jsonl'))
    checks.equal(len(rows),2757,'Full parent attempted-shot denominator')
    return dict(spec=spec,report=report,manifest=manifest,rows=rows,bound=bound,consumed=consumed,manifest_sha256=sha(directory/'evidence-manifest.json'))


def development_cases(rows,checks):
    """Independent literal failure selection: skip only fixed-base blockers."""
    test=[r for r in rows if r.get('phase')=='test' and r.get('method')=='selected_ppo']
    checks.equal((len(test),len({r['context_id'] for r in test})),(70,70),'All70 original selected-PPO layouts')
    failures=[r for r in test if not captured(r)]
    checks.equal(len(failures),42,'Original22 declined plus20 released failures')
    skipped=[r for r in failures if r.get('invalid')=='fixed_base_clearance_rejected']
    selected=[r for r in failures if r.get('invalid')!='fixed_base_clearance_rejected']
    checks.equal((len(selected),len(skipped)),(39,3),'Prescribed39 recovery cases and3 explicit fixed-base exclusions')
    checks.equal(Counter(r.get('invalid') for r in failures),Counter({None:20,'finite_ik_unresolved':13,
        'strict_endpoint_clearance_or_pose_rejected':4,'action_bounds_rejected':2,'fixed_base_clearance_rejected':3}),
        'Exact original failure reason accounting')
    return selected,skipped


def bind_parent_rows(parent,rows,checks):
    """Original proof covers their scoring; hashes prevent altered demonstrations."""
    by_id={r['id']:r for r in parent['rows']}
    for row in rows:
        ident=row['id'];checks.equal(row,by_id[ident],ident+'/original raw parent row unchanged')
        namespace=row.get('adapter_directory','native')
        checks.require(namespace in ('native','native-controls'),ident+'/known parent namespace')
        for key in ('context_resolution','connection'):
            relative=namespace+'/'+row[key+'_file']
            checks.equal(sha(parent['bound'](relative)),row[key+'_sha256'],ident+'/authenticated parent '+key)
        if row.get('native_release'):
            relative=namespace+'/'+row['native_cache_file']
            checks.equal(sha(parent['bound'](relative)),row['native_cache_sha256'],ident+'/authenticated parent dense trace')


def actor_finetune_replay(before,after,teachers,fit,checks):
    """Reproduce actor-only learning; no new critic fit or PPO is permitted."""
    import torch
    torch.set_num_threads(2)
    checks.require(bool(teachers),'Nonempty authenticated teacher union')
    checks.require(all(captured(r) and r.get('role')=='train' for r in teachers),'Only captured TRAIN/development teachers')
    x=torch.as_tensor(np.asarray([r['observation'] for r in teachers]),dtype=torch.float32)
    y=torch.as_tensor(np.asarray([r['normalized_residual'] for r in teachers]),dtype=torch.float32)
    checks.require(x.shape==(len(teachers),9) and y.shape==(len(teachers),5)
        and bool(torch.isfinite(x).all()) and bool(torch.isfinite(y).all()) and bool((y.abs()<=1).all()),'Finite bounded actor teacher tensors')
    model=model_audit.independent_policy();model.load_state_dict(before)
    optimizer=torch.optim.Adam(model.actor.parameters(),lr=.001)
    lossfn=lambda:((model.actor(x).tanh()-y)**2).mean()
    initial=float(lossfn().detach())
    for _ in range(800):
        loss=lossfn();optimizer.zero_grad();loss.backward();optimizer.step()
    final=float(lossfn().detach())
    for name,value in model.state_dict().items():
        tolerance=2e-6 if name.startswith('actor.') else 0
        checks.numeric(common.numpy_state(after)[name],value.detach().numpy(),'Actor-only learned state/'+name,tolerance)
        if not name.startswith('actor.'):
            checks.numeric(common.numpy_state(after)[name],common.numpy_state(before)[name],'Non-actor state unchanged/'+name,0)
    checks.equal((fit['actor_steps'],fit['critic_steps']),(800,0),'Exactly800 actor steps and no critic/PPO training')
    checks.numeric(fit['initial_actor_loss'],initial,'Independent initial actor loss',2e-6)
    checks.numeric(fit['final_actor_loss'],final,'Independent final actor loss',2e-6)
    checks.equal(fit['actor_teacher_ids'],[r['id'] for r in teachers],'Exact literal teacher order')
    return dict(initial_actor_loss=initial,final_actor_loss=final,actor_steps=800,critic_steps=0)


def audit_actor_transform(prediction,state,checks,label):
    latent=common.network_forward(state['policy'],prediction['observation'])
    residual=np.tanh(latent.astype(float));saved=np.tanh(np.asarray(prediction['latent_mean'],float))
    anchor=np.asarray(prediction['effective_source_hint']['values'],float)
    checks.numeric(prediction['latent_mean'],latent,label+'/independent actor output',2e-6)
    checks.numeric(prediction['normalized_residual'],residual,label+'/independent residual',2e-6)
    checks.numeric(prediction['normalized_residual'],saved,label+'/exact recorded tanh transform',1e-12)
    checks.numeric(prediction['command_values'],anchor+saved*RESIDUAL,label+'/exact recorded command transform',1e-12)
    checks.numeric((np.asarray(prediction['command_values'])-anchor)/RESIDUAL,residual,label+'/normalized actor request',2e-6)


META = ('source_hint','effective_source_hint','source_teacher_id','context_resolution_file',
        'context_resolution_sha256','context_home_sha256','context_layout_eligible',
        'context_anchor_admitted','context_hard_reject','observation')


def protocol_checks(value,checks):
    expected=dict(substeps=64,methods=list(METHODS),test_seed=2026110923,development_failures=42,
        fixed_base_skips=3,search_layouts=39,proposals_per_layout=16,search_requests=624,
        repeat_per_selected=2,repeat_requests_max=78,control_requests=18,test_layouts=70,
        test_requests=280,maximum_requests=1000,actor_steps=800,critic_steps=0,actor_lr=.001,
        base_teacher_count=68,noise=False,residual_scale=RESIDUAL.tolist(),search_limit_s=3600,
        repeat_cut_s=4500,worker_limit_s=7200,global_limit_s=10800,
        storage_reserve_bytes=8*1024**3,minimum_split_distance_m=.005)
    for key,expected_value in expected.items():checks.equal(value[key],expected_value,'Literal approved recovery protocol/'+key)


def rank(row):return (-int(captured(row) and row.get('route_quality')==2),-float(oracle.reward(row)),row['id'])


def independent_proposal(source,index,alternatives):
    """Sixteen literal support probes, independently reconstructed."""
    if type(index) is not int or not 0<=index<16:raise EvidenceError('Literal proposal index must be0..15')
    residual=np.array(source['normalized_residual'],float);ids=[source['id']];kind='original_failed_policy'
    if residual.shape!=(5,) or not np.isfinite(residual).all() or not (abs(residual)<1).all():
        raise EvidenceError('Finite original residual strictly inside support required')
    if index==1:
        choices=[r for r in alternatives if r['context_id']==source['context_id'] and captured(r)]
        if choices:
            best=sorted(choices,key=rank)[0]
            if any(best[k]!=source[k] for k in ('source_hint','effective_source_hint','observation')):
                raise EvidenceError('Alternative action cannot silently change geometry anchor')
            residual=np.array(best['normalized_residual'],float);ids=[best['id']];kind='observed_other_policy_capture'
        else:residual=np.zeros(5);kind='nominal_geometry_anchor'
    elif index>1:
        if not source['native_release']:
            kind='setup_support_probe'
            if index<11:
                residual[2]=(-.8,0.,.8)[(index-2)//3];residual[3]=(-.75,0.,.75)[(index-2)%3]
            elif index<13:residual[4]={11:.5,12:.9}[index]
            elif index==13:residual[0]=0.
            elif index==14:residual[1]=0.
            else:residual[2],residual[3]=.5,-.5
        else:
            kind='flight_support_probe'
            if index<8:residual[4]=(-.9,-.6,-.3,.3,.6,.9)[index-2]
            elif index<12:residual[3]=(-.75,-.375,.375,.75)[index-8]
            elif index<14:residual[2]=(-.75,.75)[index-12]
            elif index==14:residual[0]=residual[1]=0.
            else:residual[2],residual[3],residual[4]=0.,-.5,.75
    if not np.isfinite(residual).all() or not (abs(residual)<1).all():raise EvidenceError('Unclipped finite action support required')
    return residual,dict(index=index,kind=kind,parent_source_ids=ids)


def independent_search_plan(cases,parent_rows):
    alternatives=[r for r in parent_rows if r.get('phase')=='test'];plan=[]
    for index in range(16):
        for source in cases:
            residual,derivation=independent_proposal(source,index,alternatives)
            values=(np.asarray(source['effective_source_hint']['values'])+RESIDUAL*residual).tolist()
            plan.append(dict({k:copy.deepcopy(source[k]) for k in META},
                id=f'recovery-search-r{index:02d}-{source["context_id"]}',phase='search',role='train',method='recovery_search',
                context_id=source['context_id'],bucket_xy_m=source['bucket_xy_m'],values=values,command_values=values,
                normalized_residual=residual.tolist(),proposal_index=index,derivation=derivation,
                parent_failure_id=source['id'],parent_failure_sha256=digest(source),original_role='test',
                development_reclassified=True,perturbation_mixed_units=[0.]*5))
    return plan


def independent_repeat_plan(search,cases):
    selected=[];plan=[]
    for case in cases:
        captures=[r for r in search if r['context_id']==case['context_id'] and captured(r)]
        if not captures:continue
        source=sorted(captures,key=rank)[0]
        selected.append(dict(context_id=source['context_id'],discovery_id=source['id'],discovery_sha256=digest(source)))
        for repeat in range(2):
            fields=(*META,'context_id','bucket_xy_m','values','command_values','normalized_residual',
                    'parent_failure_id','parent_failure_sha256','original_role','development_reclassified')
            plan.append(dict({k:copy.deepcopy(source[k]) for k in fields},
                id=f'recovery-repeat-{source["context_id"]}-r{repeat}',phase='repeat',role='train',method='recovery_repeat',
                repeat=repeat,source_discovery_id=source['id'],source_discovery_sha256=digest(source),perturbation_mixed_units=[0.]*5))
    return selected,plan


def confirmed_teachers(search,repeats,selected,checks):
    source={r['id']:r for r in search};teachers=[]
    for item in selected:
        group=[r for r in repeats if r['source_discovery_id']==item['discovery_id']]
        checks.require(len(group) in (0,2),'Confirmation only complete nominal pairs')
        if group:
            checks.equal([r['repeat'] for r in group],[0,1],'Two unique independent repeat identities')
            for row in group:
                for key in ('q10','q14','connection_file','connection_sha256','values','bucket_xy_m'):
                    checks.equal(row.get(key),source[item['discovery_id']].get(key),row['id']+'/exact nominal branch/'+key)
        if len(group)==2 and all(captured(r) for r in group):teachers.append(source[item['discovery_id']])
    return sorted(teachers,key=lambda r:r['context_id'])


def regenerated_test(exclusions):
    used=np.asarray(sorted(set(tuple(map(float,xy)) for xy in exclusions)),float)
    if used.ndim!=2 or used.shape[1]!=2 or len(used)==0 or not np.isfinite(used).all():raise EvidenceError('Finite inventory required')
    result=[];rng=np.random.default_rng(2026110923)
    for iy in range(5):
        for ix in range(7):
            for n in range(2):
                for _ in range(10000):
                    xy=np.array([.1,-.65])+(np.array([1.24,.25])-np.array([.1,-.65]))/[7,5]*([ix,iy]+rng.uniform(.05,.95,2))
                    if np.linalg.norm(used-xy,axis=1).min()>=.005:break
                else:raise EvidenceError('Fresh coordinate generation exhausted')
                result.append(dict(id=f'recovery-test-r{iy}-c{ix}-n{n}',split='test',bucket_xy_m=xy.tolist(),grid_cell=[ix,iy]))
                used=np.vstack((used,xy))
    return result


def audit_inventory(study,source,history_root,spec,cases,checks):
    inventory=read(study/'historical-coordinate-bindings.json');coordinates=set()
    names={'bucket_xy_m','bucket_xy','exclusions','excluded','coordinates','excluded_coordinates'}
    def collect(item,enabled=False):
        if enabled and isinstance(item,list) and len(item)==2 and all(type(x) in (int,float) and math.isfinite(x) for x in item):
            coordinates.add(tuple(map(float,item)));return
        if isinstance(item,dict):
            for key,value in item.items():collect(value,key in names)
        elif isinstance(item,list):
            for value in item:collect(value,enabled)
    checks.require(bool(inventory['files_sha256']),'Nonempty frozen historical inventory')
    for name,expected in inventory['files_sha256'].items():
        prefix,relative=name.split('/',1);checks.require(prefix in ('output','baselines'),'Known coordinate inventory root')
        path=contained(history_root if prefix=='output' else source/'baselines',relative)
        checks.require(not path.resolve().is_relative_to(study.resolve()),'New study excluded from its own history')
        checks.equal(sha(path),expected,'Literal historical coordinate source/'+name)
        if path.suffix=='.jsonl':
            for item in common.lines(path):collect(item)
        else:collect(read(path),'exclusion' in path.name)
    expected_coordinates=[list(x) for x in sorted(coordinates)]
    checks.equal(inventory['coordinates'],expected_coordinates,'Complete inventoried coordinates without outcome filtering')
    expected=dict(development=[dict(id=r['context_id'],bucket_xy_m=r['bucket_xy_m'],split='train',original_role='test') for r in cases],
                  test=regenerated_test(expected_coordinates))
    checks.equal(read(study/'splits.json'),expected,'Unfiltered seeded70 fresh held-out positions and39 development cases')
    checks.equal(spec['splits'],expected,'Split bound before any outcome')
    return dict(files=len(inventory['files_sha256']),historical_coordinates=len(coordinates),fresh_test_layouts=70)


def audit_search_and_teachers(study,rows,cases,parent_rows,parent_teachers,report,spec,checks):
    plan=independent_search_plan(cases,parent_rows)
    checks.equal(read(study/'search-plan.json'),dict(rows=plan,frozen_before_outcomes=True),'Entire independently reconstructed624 proposal plan')
    checks.equal(spec['plans']['search_sha256'],sha(study/'search-plan.json'),'Search plan bound in pre-outcome contract')
    search=[r for r in rows if r['phase']=='search'];repeats=[r for r in rows if r['phase']=='repeat']
    checks.require(0<len(search)<=624 and (len(search)==624 or len(search)%32==0),'Only complete bounded search batches')
    checks.require(len(repeats)<=78 and len(repeats)%2==0,'Only full nominal confirmation pairs')
    for row,request in zip(search,plan):
        checks.equal(row['search_plan_sha256'],sha(study/'search-plan.json'),row['id']+'/frozen proposal plan')
        for key,value in request.items():checks.equal(row.get(key),value,row['id']+'/literal prescribed proposal/'+key)
    selected,repeat_plan=independent_repeat_plan(search,cases)
    checks.equal(read(study/'repeat-plan.json'),dict(selected=selected,rows=repeat_plan,frozen_before_outcomes=True),'Best per-case discovery and two repeats fixed before outcomes')
    checks.equal(report['selected'],selected,'Report selects exactly independent discovery ranking')
    checks.require(len(repeats)<=len(repeat_plan),'No unplanned repeat requests')
    for row,request in zip(repeats,repeat_plan):
        checks.equal(row['repeat_plan_sha256'],sha(study/'repeat-plan.json'),row['id']+'/pre-outcome repeat plan')
        for key,value in request.items():checks.equal(row.get(key),value,row['id']+'/exact nominal repeat request/'+key)
    confirmed=confirmed_teachers(search,repeats,selected,checks)
    union=sorted(copy.deepcopy(parent_teachers)+confirmed,key=lambda r:r['context_id'])
    checks.require(len({r['context_id'] for r in union})==len(union),'One teacher per distinct context; no duplicated positive weighting')
    checks.equal(read(study/'bootstrap/recovery-teachers.json'),confirmed,'Only2/2 confirmed new discoveries admitted')
    checks.equal(read(study/'bootstrap/teachers.json'),union,'Exact original68 plus confirmed teacher union')
    checks.equal(report['confirmed_teachers'],len(confirmed),'True confirmed new teacher count')
    bindings=read(study/'bootstrap/source-bindings.json')
    for key,value in dict(parent_teacher_ids=[r['id'] for r in parent_teachers],parent_teacher_sha256={r['id']:digest(r) for r in parent_teachers},
        new_teacher_ids=[r['id'] for r in confirmed],teacher_file_sha256=sha(study/'bootstrap/teachers.json'),repeat_plan_sha256=sha(study/'repeat-plan.json')).items():
        checks.equal(bindings[key],value,'Teacher outcome provenance/'+key)
    checks.equal(report['search_coverage'],[dict(context_id=c['context_id'],bucket_xy_m=c['bucket_xy_m'],
        counts=statistics([r for r in search if r['context_id']==c['context_id']]),confirmed=any(r['context_id']==c['context_id'] for r in confirmed)) for c in cases],
        'Every recovery context retains its complete requested denominator')
    return union,confirmed,len(repeat_plan)


def audit_parent_inputs(study,parent,checks,spec):
    data={}
    for key,name in (('prior','prior.json'),('old_prior','old-prior.json'),('controls','controls.json'),
        ('control_replays','control-replays.json'),('calibration','calibration.json'),('home','setup-home.json'),('teachers','bootstrap/teachers.json')):
        data[key]=read(parent['bound'](name))
        if key!='teachers':checks.equal(read(study/name),data[key],'Exact original input/'+name)
    checks.equal(sha(study/'verification.json'),sha(parent['bound']('verification.json')),'Original calibration verification unchanged')
    checks.equal((len(data['prior']['records']),len(data['old_prior']['records']),len(data['teachers'])),(19,17,68),'Shared19 geometry,17 controls and68 original teachers')
    checks.require(all(r['role']=='train' and r['phase']=='search' and captured(r) for r in data['teachers']),'Only original captured TRAIN teachers')
    cases,blocked=development_cases(parent['rows'],checks);ids={r['context_id'] for r in cases}
    hydrated=[r for r in parent['rows'] if r.get('phase')=='test' and r['context_id'] in ids and (r['method']=='selected_ppo' or captured(r))]
    bind_parent_rows(parent,hydrated+data['teachers'],checks)
    checks.equal(read(study/'blocked.json'),[dict(id=r['id'],context_id=r['context_id'],bucket_xy_m=r['bucket_xy_m'],reason=r['invalid'],parent_row_sha256=digest(r)) for r in blocked],
                 'Three fixed-base failures explicitly retained outside recoverable cohort')
    original_controls=data['control_replays'];names={'effective-options.json','native-compiled-geometry.json'}
    for row in original_controls:
        for field in ('context_resolution','connection'):
            name=row[field+'_file'];names.add(name)
            checks.equal(sha(parent['bound']('native-controls/'+name)),row[field+'_sha256'],'Original17 admitted control proof/'+name)
    mirror={name:sha(parent['bound']('native-controls/'+name)) for name in sorted(names)}
    checks.equal(read(study/'control-parent/mirror.json'),mirror,'Read-only matched-control proof mirror inventory')
    for name,expected in mirror.items():checks.equal(sha(contained(study/'control-parent/native',name)),expected,'Exact mirrored control evidence/'+name)
    checks.equal({str(p.relative_to(study/'control-parent/native')) for p in (study/'control-parent/native').rglob('*') if p.is_file()},set(mirror),'No unbound mirrored control files')
    bindings=read(study/'source-bindings.json');checks.equal(spec['source_bindings'],bindings,'Contract binds consumed parent evidence')
    for key,value in dict(report_sha256=PARENT_REPORT_SHA,audit_sha256=PARENT_AUDIT_SHA,
        contract_sha256=parent['manifest']['contract.json']).items():
        checks.equal(bindings[key],value,'Exact independently pinned parent binding/'+key)
    # Manifest hash is authenticated by the pinned corrected parent audit.
    checks.equal(bindings['manifest_sha256'],parent['manifest_sha256'],'Exact parent manifest byte binding')
    for name,expected in bindings['files_sha256'].items():checks.equal(sha(parent['bound'](name)),expected,'Each claimed parent input authenticated/'+name)
    checks.equal(bindings['case_rows_sha256'],{r['id']:digest(r) for r in parent['rows'] if r.get('phase')=='test' and r.get('method')=='selected_ppo' and not captured(r)},'All42 original failure rows byte bound')
    for key,bank in (('prior_sha256',data['prior']),('old_prior_sha256',data['old_prior'])):checks.equal(spec[key],digest(bank),'Exact shared geometry prior/'+key)
    return data,cases,blocked


def audit_models(study,report,spec,parent,teachers,confirmed,checks):
    artifacts=report['artifacts'];states={}
    checks.equal(set(artifacts),set(METHODS),'Exactly four frozen methods')
    for method,parent_key in (('parent_bc','bc'),('parent_ppo','selected'),('previous_ppo','current_ppo')):
        original=parent['report']['artifacts'][parent_key];record=artifacts[method]
        checks.equal(spec['old_artifacts'][method],record,'Pre-outcome original artifact/'+method)
        checks.equal(record['path'],'checkpoints/'+method+'.pt','Literal model path/'+method)
        for key in ('sha256','weight_sha256','samples','seed'):checks.equal(record.get(key),original.get(key),'Unchanged original model/'+method+'/'+key)
        checks.equal(sha(parent['bound'](original['path'])),record['sha256'],'Original checkpoint bytes authenticated/'+method)
    checks.equal(artifacts['previous_ppo']['sha256'],'9d3a6f549af8247b84fe39c6d9a2b2785020198145ffb4c205e8644994b8826e','Strongest prior PPO included')
    for method,record in artifacts.items():
        path=contained(study,record['path']);checks.equal(sha(path),record['sha256'],'Frozen checkpoint bytes/'+method)
        state=common.load_checkpoint(path,checks,method);states[method]=state
        for key in ('weight_sha256','samples'):checks.equal(state.get(key),record.get(key),'Checkpoint content identity/'+method+'/'+key)
        # Historical current_ppo's report omitted seed; the complete checkpoint
        # bytes are pinned above, so preserve that omission rather than invent it.
        if record.get('seed') is not None:checks.equal(state.get('seed'),record['seed'],'Declared checkpoint seed/'+method)
    fit=read(study/'fit.json');checks.equal(report['fit'],fit,'Report retains exact actor fit evidence')
    checks.equal((fit['parent_teacher_count'],fit['new_teacher_count']),(68,len(confirmed)),'Actual teacher counts in fit')
    before,after=states['parent_bc']['policy'],states['recovery_bc']['policy']
    critic=lambda state:common.weights_sha256({name.removeprefix('critic.'):value for name,value in state.items() if name.startswith('critic.')})
    checks.equal((fit['critic_before_sha256'],fit['critic_after_sha256']),(critic(before),critic(before)),'Critic unchanged and correctly hashed')
    if confirmed:
        checks.equal(fit['status'],'actor_updated','Confirmed demonstrations enable actor-only fitting')
        checks.equal(states['recovery_bc']['method'],'actor_only_recovery_bc','New checkpoint labels actor-only imitation')
        checks.equal(states['recovery_bc']['contract_fingerprint'],spec['fingerprint'],'Learned checkpoint binds new contract')
        checks.equal(artifacts['recovery_bc']['samples'],len(teachers),'Training sample count equals exact teacher union')
        result=actor_finetune_replay(before,after,teachers,fit,checks)
        checks.equal((fit['before'],fit['after']),(common.weights_sha256(before),common.weights_sha256(after)),'Exact before/after optimizer state hashes')
        checks.equal(fit['critic_row_count'],len(teachers),'Recorded unchanged critic dataset count')
        xx=np.asarray([r['observation'] for r in teachers]);yy=np.asarray([r['learning_reward'] for r in teachers])
        # Critic loss is descriptive only; its weights are checked exactly above.
        initial_loss=float(np.mean((np.array([common.network_forward(before,x,prefix='critic')[0] for x in xx])-yy)**2))
        checks.numeric(fit['initial_critic_loss'],initial_loss,'Unchanged critic descriptive loss',2e-5)
        checks.numeric(fit['final_critic_loss'],fit['initial_critic_loss'],'No critic optimizer step',0)
    else:
        checks.equal((fit['status'],fit['actor_steps'],fit['critic_steps'],fit['byte_identical_parent']),('no_confirmed_new_teacher',0,0,True),'No invented fit without confirmed discoveries')
        checks.equal(artifacts['recovery_bc']['sha256'],artifacts['parent_bc']['sha256'],'No new teachers means byte-identical parent model')
        checks.equal((fit['before'],fit['after']),(common.weights_sha256(before),)*2,'No-change fit hashes original weights')
        result=dict(actor_steps=0,critic_steps=0,byte_identical_parent=True)
    return states,result


def audit_test(study,rows,report,spec,states,checks):
    packets=[];comparisons=[];positions=[];contexts=spec['splits']['test']
    for method in METHODS:
        packet=read(study/f'predictions/test-{method}.json');packets.append(packet)
        record=report['artifacts'][method];group=[r for r in rows if r['phase']=='test' and r['method']==method]
        checks.equal((packet['method'],packet['role'],packet['artifact']),(method,'test',record),'Frozen inference packet/'+method)
        checks.equal((packet['one_prediction_per_layout'],packet['selection_or_learning']),(True,False),'One nonadaptive inference per context/'+method)
        checks.equal((len(packet['rows']),len(group)),(70,70),'Complete paired70 denominator/'+method)
        for prediction,row,context in zip(packet['rows'],group,contexts):
            label=row['id'];checks.equal(label,f'recovery-test-{method}-{context["id"]}',label+'/literal prediction identity')
            checks.equal((row['role'],row['phase'],row['method']),('test','test',method),label+'/unchanged held-out role')
            checks.equal((prediction['context_id'],prediction['bucket_xy_m']),(context['id'],context['bucket_xy_m']),label+'/unfiltered preselected bucket')
            for key,value in prediction.items():checks.equal(row.get(key),value,label+'/immutable prediction/'+key)
            checks.equal((prediction['artifact_sha256'],row['frozen_artifact_sha256']),(record['sha256'],)*2,label+'/frozen artifact binding')
            checks.equal((prediction['source_training_ids'],prediction['ball_flight_search_at_inference'],prediction['geometry_search_at_inference'],prediction['selection_or_learning']),
                         ([],False,True,False),label+'/geometry-only resolution without flight search or learning')
            checks.numeric(prediction['observation'],calibrated.expected_observation(context['bucket_xy_m'],prediction['effective_source_hint']),label+'/geometry-only nine-feature input',0)
            audit_actor_transform(prediction,states[method],checks,label)
            checks.equal((row['values'],row['perturbation_mixed_units']),(row['command_values'],[0.]*5),label+'/actual unmodified noiseless action')
            checks.equal(row['test_plan_sha256'],sha(study/'test-plan.json'),label+'/all-four-model pre-outcome plan')
            positions.append(dict(context_id=row['context_id'],bucket_xy_m=row['bucket_xy_m'],method=method,
                capture=captured(row),clean=captured(row) and row.get('route_quality')==2,native_release=row['native_release'],invalid=row.get('invalid'),
                forbidden_contact_steps=row.get('forbidden_contact_steps',0),values=row['values'],record_id=row['id']))
        comparisons.append(dict(method=method,counts=statistics(group),artifact_sha256=record['sha256']))
    checks.equal(read(study/'test-plan.json'),dict(methods=packets,frozen_before_outcomes=True),'All four packet objects frozen as one pre-outcome plan')
    checks.equal(report['test_comparison'],comparisons,'Independent full-denominator comparison')
    checks.equal(report['test_positions'],positions,'Every concrete outcome reproduced')
    # Every model is offered the same geometry-only anchor, including rejections.
    for index in range(70):
        first=packets[0]['rows'][index]
        for packet in packets[1:]:
            for key in META:checks.equal(packet['rows'][index][key],first[key],'Matched shared geometry-only reference/'+str(index)+'/'+key)
    return comparisons


def audit_controls(study,rows,sources,originals,checks):
    gates={}
    for phase,key in (('controls_pre','pre'),('controls_post','post')):
        group=[r for r in rows if r['phase']==phase];checks.equal(len(group),9,phase+'/nine matched controls')
        for i,(source,original,role) in enumerate(zip(sources,originals,CONTROL_ROLES)):
            for repeat in range(3):
                row=group[3*i+repeat];label=row['id']
                checks.equal(label,f'recovery-{phase}-{role}-r{repeat}',label+'/literal control ID')
                checks.equal((row['role'],row['method'],row['repeat']),(role,'control',repeat),label+'/control identity')
                checks.equal((row['control_source_id'],row['control_source_sha256']),(source['id'],digest(source)),label+'/original control provenance')
                checks.equal((row['source_record_id'],row['source_record_sha256'],row['variant']),(original['id'],digest(original),'control_exact'),label+'/exact original17 replay')
                checks.equal((row['values'],row['bucket_xy_m']),(source['values'],source['bucket_xy_m']),label+'/matched requested control')
                for name in (*META,'q10','q14','connection_file','connection_sha256'):
                    checks.equal(row[name],original[name],label+'/unchanged control branch/'+name)
                for name in ('q10','q14'):
                    checks.equal(row['native_initial_'+name],np.asarray(original[name],np.float32).astype(float).tolist(),label+'/actual matched reset/'+name)
        gates[key]=native.control_gate(group);checks.equal(gates[key]['status'],'passed',phase+'/unchanged direct, table-assisted and negative controls')
    return gates


def audit_sequence(rows,report,repeat_planned,checks):
    ns=sum(r['phase']=='search' for r in rows);nr=sum(r['phase']=='repeat' for r in rows)
    expected=[('controls_pre',r,'control') for r in CONTROL_ROLES for _ in range(3)]
    expected += [('search','train','recovery_search')]*ns+[('repeat','train','recovery_repeat')]*nr
    expected += [('test','test',m) for m in METHODS for _ in range(70)]
    expected += [('controls_post',r,'control') for r in CONTROL_ROLES for _ in range(3)]
    checks.equal([(r['phase'],r['role'],r['method']) for r in rows],expected,'No hidden attempts, retries, training or omitted held-out outcomes')
    checks.require(len(rows)<=1000,'Strict1000 total-request cap includes refusals')
    checks.equal(len({r['id'] for r in rows}),len(rows),'Unique attempted-shot IDs')
    checks.equal((report['search_requests'],report['search_completed_rounds'],report['search_partial_round_requests'],report['repeat_requests']),
        (ns,ns//39,ns%39,nr),'Exact complete and partial search/repeat counts')
    checks.equal((report['search_budget_complete'],report['repeat_budget_complete']),(ns==624,nr==repeat_planned),'Partial budgets never claimed as complete')
    checks.equal([r['stage'] for r in report['budget_reasons']],(['search'] if ns<624 else [])+(['repeat'] if nr<repeat_planned else []),'Only prescribed deadline truncations')
    for reason in report['budget_reasons']:
        if reason['stage']=='search':
            checks.equal((reason['reason'],reason['completed']),('search_deadline',ns),'Actual search deadline prefix')
            checks.require(reason['elapsed_s']>=3600,'Search acquired-slot deadline reached before truncation')
            checks.numeric(reason['search_elapsed_s'],reason['elapsed_s']-report['search_started_elapsed_s'],'Search duration measured independently',.05)
        else:
            checks.equal((reason['reason'],reason['completed'],reason['planned']),('repeat_deadline',nr,repeat_planned),'Actual repeat deadline prefix')
            checks.require(reason['elapsed_s']>=4500,'Repeat deadline reached before truncation')
    for key,value in dict(worker_limit_s=7200,search_limit_s=3600,repeat_cut_s=4500).items():checks.equal(report['deadlines'][key],value,'Exact bounded deadline/'+key)
    checks.require(0<=report['search_started_elapsed_s']<=report['elapsed_s']<7200,'Measured completed worker within2hours')
    return dict(search_requests=ns,repeat_requests=nr,test_requests=280,control_requests=18,total_requests=len(rows),maximum_requests=1000)


def audit_boundaries(study,rows,checks):
    boundaries=common.lines(study/'phase-boundaries.jsonl');ns=sum(r['phase']=='search' for r in rows);nr=sum(r['phase']=='repeat' for r in rows)
    expected=[('controls_pre-complete',9),('search-complete',9+ns),('repeat-complete',9+ns+nr),
              ('fit-complete',9+ns+nr),('test-plan-frozen',9+ns+nr),('controls_post-complete',len(rows))]
    checks.equal([(b['phase'],b['rows_count']) for b in boundaries],expected,'Search,confirmation,fit and all prediction freeze precede TEST')
    raw=(study/'rows.jsonl').read_bytes().splitlines(keepends=True);checks.equal(len(raw),len(rows),'Exact serialized request count')
    previous=0
    for boundary in boundaries:
        checks.equal(boundary['rows_sha256'],hashlib.sha256(b''.join(raw[:boundary['rows_count']])).hexdigest(),boundary['phase']+'/immutable observed prefix')
        checks.require(previous<=boundary['acquired_elapsed_s']<7200,'Monotone bounded phase times');previous=boundary['acquired_elapsed_s']
    timings=common.lines(study/'timing.jsonl')
    checks.require(boundaries[4]['acquired_elapsed_s']<=next(t['acquired_start_s'] for t in timings if t['phase']=='test'),'All280 predictions frozen before any TEST outcome')
    checks.require(all(t['acquired_end_s']<7200 for t in timings),'All actual batches within acquired-slot budget')
    search=[t for t in timings if t['phase']=='search'];repeats=[t for t in timings if t['phase']=='repeat']
    checks.require(all(t['acquired_start_s']<3600 for t in search),'No search batch starts after absolute search cutoff')
    checks.require(all(t['acquired_start_s']<4500 for t in repeats),'No repeat pair starts after repeat cutoff')


def audit_budget(study,report,checks):
    path=study.with_name(study.name+'-budget.json');budget=read(path);pipeline=read(study.with_suffix('.pipeline.json'))
    checks.equal((budget['limit_s'],budget['audit_reserve_s']),(7200,1800),'Two-hour worker with half-hour CPU audit reserve')
    checks.require(0<pipeline['global_limit_s']<=10800,'Pipeline deadline at most3hours')
    checks.numeric(pipeline['global_deadline_unix_s']-pipeline['started_unix_s'],pipeline['global_limit_s'],'Absolute supervisor interval',.05)
    checks.equal(budget['global_deadline_unix_s'],pipeline['global_deadline_unix_s'],'Worker and supervisor share absolute deadline')
    checks.require(pipeline['started_unix_s']<=budget['acquired_unix_s'],'Lease acquired after own pipeline starts')
    checks.numeric(budget['effective_worker_limit_s'],min(7200,budget['global_deadline_unix_s']-budget['acquired_unix_s']-1800),'Actual postqueue worker limit',.1)
    checks.require(60<budget['effective_worker_limit_s']<=7200,'Finite positive worker budget')
    checks.equal(report['deadlines']['effective_worker_limit_s'],budget['effective_worker_limit_s'],'Report actual worker cap')
    checks.equal(report['deadlines']['global_deadline_unix_s'],budget['global_deadline_unix_s'],'Report absolute pipeline cap')
    checks.require(report['elapsed_s']<budget['effective_worker_limit_s'],'Worker completed within actual limit')
    return dict(worker_receipt_sha256=sha(path),effective_worker_limit_s=budget['effective_worker_limit_s'],global_limit_s=pipeline['global_limit_s'])


def audit(study,source,assets,*,parent_study,parent_audit,history_root,reference):
    started=time.monotonic();checks=AuditChecks();study,source,assets,history_root=map(Path,(study,source,assets,history_root))
    manifest=replay.check_manifest(study,checks);spec=read(study/'contract.json');report=read(study/'report.json')
    unsigned=dict(spec);fp=unsigned.pop('fingerprint')
    checks.equal((digest(unsigned),report['contract_fingerprint']),(fp,fp),'Complete immutable experiment fingerprint')
    checks.equal((spec['version'],report['version'],report['status'],report['phase']),(STUDY_VERSION,STUDY_VERSION,'completed','completed'),'Completed versioned recovery comparison')
    checks.equal(report['evidence_sha256'],manifest,'Report binds all immutable raw evidence')
    checks.equal(report['protocol'],spec['protocol'],'Protocol fixed before outcomes');protocol_checks(spec['protocol'],checks)
    replay.check_source_files(source,spec['source_sha256'],checks,'Frozen executed producer source')
    for module in (Path(__file__),Path(oracle.__file__)):
        checks.equal(sha(module),spec['source_sha256']['sircus/mjlab/'+module.name],'Actual auditor implementation matches frozen source/'+module.name)
    parent=authenticate_parent(parent_study,parent_audit,checks)
    data,cases,blocked=audit_parent_inputs(study,parent,checks,spec)
    model_audit.audit_runtime_reference(source,reference,spec,parent['spec'],assets,checks)
    inventory=audit_inventory(study,source,history_root,spec,cases,checks)
    calibration=replay.check_calibration(study,spec['identity']['calibration'],checks)
    rows=replay.read_records(study/'rows.jsonl');native.check_rows(rows,checks)
    setup,native_count=parent_audit_module().audit_native_namespaces(study,rows,data['prior'],data['old_prior'],data['home'],spec,calibration,assets,checks)
    teachers,confirmed,nplanned=audit_search_and_teachers(study,rows,cases,parent['rows'],data['teachers'],report,spec,checks)
    accounting=audit_sequence(rows,report,nplanned,checks);requests=parent_audit_module().audit_request_packets(study,rows,checks)
    audit_boundaries(study,rows,checks);budget=audit_budget(study,report,checks)
    states,fit=audit_models(study,report,spec,parent,teachers,confirmed,checks)
    comparison=audit_test(study,rows,report,spec,states,checks)
    gates=audit_controls(study,rows,data['controls'],data['control_replays'],checks)
    checks.equal(report['control_gates'],gates,'Complete independently verified before/after controls')
    checks.equal(report['counts'],statistics(rows),'All attempted outcomes retained in denominator')
    checks.equal(report['stage_counts'],{p:statistics([r for r in rows if r['phase']==p]) for p in ('controls_pre','search','repeat','test','controls_post')},'Complete independent phase accounting')
    checks.equal(report['base_teachers'],68,'Original68 parent teachers retained')
    checks.equal(report['blocked'],read(study/'blocked.json'),'Fixed-base cases disclosed in report')
    checks.equal(report['gate'],dict(status='passed',substeps=64,controls_passed=True,complete_test=True,no_policy_promotion=True),'No automatic model deployment or promotion')
    return dict(version=VERSION,status='passed',full_checks_verified=True,native_checks_verified=True,checks_count=checks.count,
        checks=checks.entries,failures=[],contract_fingerprint=fp,report_sha256=sha(study/'report.json'),
        evidence_manifest_sha256=sha(study/'evidence-manifest.json'),auditor_sha256=sha(Path(__file__)),
        producer_source_sha256=spec['source_sha256'],parent_audit_sha256=PARENT_AUDIT_SHA,parent_report_sha256=PARENT_REPORT_SHA,
        consumed_parent_files_sha256=parent['consumed'],requested_trials=len(rows),native_releases=native_count,
        confirmed_teachers=len(confirmed),teacher_count=len(teachers),actor_fit=fit,test_comparison=comparison,
        control_gates=gates,setup_evidence=setup,coordinate_inventory=inventory,request_accounting=accounting,request_packets=requests,budget=budget,
        elapsed_s=time.monotonic()-started,maximum_numeric_error=checks.numeric_error,
        limitations=['FormerTEST failures are development data; only70 new inventoried-disjoint positions are held out.',
            'Fixed64 held endpoints, finite IK and unchanged19-template geometry resolver; no driven transit or real hardware.',
            'One frozen prediction per method per new layout, not repeatability or universal reachability.',
            'New teacher2/2 nominal confirmations contain no parameter noise or timestep-convergence claim.'])


def parent_audit_module():
    # Avoid colliding with the CLI path named parent_audit in audit().
    return parent_audit


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('study','source','assets','parent-study','parent-audit','history-root','reference','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    a=parser.parse_args()
    if a.output.exists() or a.output.resolve().is_relative_to(a.study.resolve()):raise ValueError('New proof must be outside immutable evidence')
    try:result=audit(a.study,a.source,a.assets,parent_study=a.parent_study,parent_audit=a.parent_audit,history_root=a.history_root,reference=a.reference)
    except Exception as error:
        result=dict(version=VERSION,status='failed',failures=[type(error).__name__+': '+str(error)])
        a.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');raise
    a.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({key:result[key] for key in ('status','checks_count','requested_trials','native_releases','confirmed_teachers','test_comparison')}))

if __name__=='__main__':main()
