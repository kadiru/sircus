"""Independent CPU audit of saved bouncer-branch alternatives; no fitting or fresh IK."""
from __future__ import annotations
import argparse
import copy
import json
import math
from pathlib import Path
import time
import numpy as np
from . import fixed35_six_verify as previous
from . import fixed35_recovery_verify as recovery
from . import fixed35_imitation_verify as fixed
from . import fixed35_bouncer_schedule_verify as schedule
from . import benchmark_verify as common
from . import calibrated_benchmark_verify as calibrated
from . import coverage_training_verify as geometry
from . import policy64_verify as models
from . import replay650_verify as replay
from . import table_training_verify as native
from . import table_assisted_score_oracle as oracle
from . import gripper_yaw_verify as yaw
from .benchmark_verify import EvidenceError,contained,digest,read,sha

VERSION='sircus-fixed35-bouncer-audit-v1'
STUDY_VERSION='sircus-fixed35-bouncer-v1'
PARENT_REPORT_SHA='629360baed3cd23261c125679202b8d129ffb28b9cec0c1c8709d0ef19eeeb32'
PARENT_AUDIT_SHA='ef56098a64bbd640a0c2e72aeaea1d10e7da00f853f1b9bb4b373d06c85c28e7'
PARENT_FP='adc1f18e13bdcaab2f7fb86bf63974bd762c9248b8a77c471f301d8184938c07'
BOUNCER_VERSION='sircus-fixed35-saved-bouncer-candidate-v1'
CASE_IDS=('fixed35-dev-grid-03-04','fixed35-dev-grid-03-05','fixed35-dev-grid-04-06')
POSES={'fixed35-dev-grid-03-04':[0,1,5,65], 'fixed35-dev-grid-03-05':[0,5,35,65], 'fixed35-dev-grid-04-06':[2,3,4,78]}
ROLES=previous.ROLES
META=previous.META
AuditChecks=replay.AuditChecks
captured=native.captured
statistics=native.statistics


def authenticate_parent(directory,audit,checks):
    directory,audit=Path(directory),Path(audit)
    checks.equal(sha(directory/'report.json'),PARENT_REPORT_SHA,'Exact published recovery report')
    checks.equal(sha(audit),PARENT_AUDIT_SHA,'Exact complete recovery independent audit')
    report,spec,proof,manifest=map(read,(directory/'report.json',directory/'contract.json',audit,directory/'evidence-manifest.json'))
    checks.equal((report['version'],report['status'],proof['version'],proof['status'],proof['full_checks_verified'],proof['native_checks_verified'],proof['failures']),
        ('sircus-fixed35-six-v1','completed','sircus-fixed35-six-audit-v1','passed',True,True,[]),'Complete recovery source')
    checks.equal((proof['checks_count'],len(proof['checks'])),(170755,170755),'All170755 recovery checks retained')
    checks.require(all(c['passed'] is True for c in proof['checks']),'Every recovery check passed')
    unsigned=dict(spec);fp=unsigned.pop('fingerprint')
    checks.equal((digest(unsigned),fp,report['contract_fingerprint'],proof['contract_fingerprint']),(PARENT_FP,)*4,'Original recovery identity')
    checks.equal(proof['report_sha256'],PARENT_REPORT_SHA,'Recovery audit report binding')
    checks.equal(proof['evidence_manifest_sha256'],sha(directory/'evidence-manifest.json'),'Recovery manifest binding')
    checks.equal(report['evidence_sha256'],manifest,'Recovery full evidence binding')
    consumed={}
    def bound(name):
        path=contained(directory,name);actual=sha(path)
        checks.equal(actual,manifest[name],'Authenticated recovery file/'+name)
        consumed[name]=actual;return path
    bound('contract.json');rows=common.lines(bound('rows.jsonl'))
    checks.equal(len(rows),641,'All641 recovery attempts retained')
    return dict(directory=directory,spec=spec,report=report,rows=rows,bound=bound,consumed=consumed,
        report_sha256=PARENT_REPORT_SHA,audit_sha256=PARENT_AUDIT_SHA,manifest_sha256=sha(directory/'evidence-manifest.json'))

def audit_exact_branch(row,parent,checks,label):
    for key in (*META,'values','bucket_xy_m','q10','q14','connection_file','connection_sha256','native_release'):
        checks.equal(row.get(key),parent.get(key),label+'/preserved exact admitted branch/'+key)
    if row.get('native_release'):
        for field in ('q10','q14'):
            checks.equal(row['native_initial_'+field],np.asarray(parent[field],np.float32).astype(float).tolist(),label+'/observed original native reset/'+field)


def wrapped_difference(first,second):
    return np.abs((np.asarray(first)-np.asarray(second)+np.pi)%(2*np.pi)-np.pi)


def select_candidates(parent,pool,checks):
    """Select from exact archived endpoints, never invent or re-solve joints."""
    eligible=[]
    for index,candidate in enumerate(pool['endpoint_attempts']['candidates']):
        if candidate['q10']!=parent['q10']:continue
        if np.max(wrapped_difference(candidate['q14'],parent['q14']))<.05:continue
        if any(np.max(wrapped_difference(candidate['q14'],old['candidate']['q14']))<.05 for old in eligible):continue
        eligible.append(dict(index=index,candidate=copy.deepcopy(candidate),proximal=bool(np.max(wrapped_difference(candidate['q14'][:3],parent['q14'][:3]))>=.05)))
    selected=[]
    proximal=[c for c in eligible if c['proximal']]
    if proximal:selected.append(proximal[0])
    elif eligible:selected.append(eligible[0])
    remaining=[c for c in eligible if c not in selected]
    wrist=[c for c in remaining if not c['proximal']]
    if remaining:selected.append(wrist[0] if wrist else remaining[0])
    checks.require(len(selected)<=2,'At most two saved q14 alternatives per numeric pose')
    return selected,eligible


def cohort(original,older,parent,checks):
    old=previous.cohort(original,older,checks)
    parent_by={r['id']:r for r in parent['rows']}
    checks.equal(len(parent_by),641,'Unique complete641 parent outcomes')
    replays=[r for r in parent['rows'] if r['phase']=='preserved']
    checks.equal(len(replays),29,'All29 old recipes replayed in parent study')
    for row,source in zip(replays,old['preserved']):
        checks.equal((row['parent_row_id'],row['parent_row_sha256']),(source['id'],digest(source)),row['id']+'/old recipe provenance')
        audit_exact_branch(row,source,checks,row['id'])
        checks.require(captured(row),row['id']+'/old recipe recaptured')
    additions=[]
    for case in parent['report']['cases']:
        source=case.get('confirmed_recipe')
        if not source:continue
        checks.equal(source,parent_by[source['id']],source['id']+'/unmodified parent discovery')
        checks.require(captured(source),source['id']+'/actual captured parent recipe')
        repeats=sorted([r for r in parent['rows'] if r['phase']=='repeat' and r['selected_from_id']==source['id']],key=lambda r:r['repeat'])
        checks.equal([r['repeat'] for r in repeats],[0,1,2],source['id']+'/three parent confirmations')
        for row in repeats:
            checks.require(captured(row),row['id']+'/parent confirmation retained')
            checks.equal(row['selected_from_sha256'],digest(source),row['id']+'/exact selected parent recipe')
            for key in ('values','bucket_xy_m','q10','q14','connection_sha256'):
                checks.equal(row[key],source[key],row['id']+'/confirmed parent branch/'+key)
        additions.append(source)
    checks.equal([r['context_id'] for r in additions],['fixed35-dev-grid-01-05'],'Exactly one newly recovered parent location')
    preserved=sorted(old['preserved']+additions,key=lambda r:r['context_id'])
    checks.equal((len(preserved),len({r['context_id'] for r in preserved})),(30,30),'Exactly30 preserved locations')
    cases=[r for r in old['cases'] if r['context_id'] in CASE_IDS]
    checks.equal(tuple(r['context_id'] for r in cases),CASE_IDS,'Only three prescribed residual cases')
    matched=[]
    for context in CASE_IDS:
        released=[r for r in parent['rows'] if r['phase']=='search' and r['context_id']==context and r['native_release']]
        checks.equal(sorted({r['numeric_proposal_index'] for r in released}),POSES[context],context+'/literal four previously released poses')
        for index in POSES[context]:
            group=sorted([r for r in released if r['numeric_proposal_index']==index],key=lambda r:r['ik_branch_slot'])
            checks.equal([r['ik_branch_slot'] for r in group],[0,1],context+'/two historical branch controls')
            checks.equal(group[0]['values'],group[1]['values'],context+'/matched historical numeric pose')
            checks.equal(group[0]['branch_pool_sha256'],group[1]['branch_pool_sha256'],context+'/shared source pool')
            matched.append(group[0])
    return dict(current=old['current'],refit=old['refit'],preserved=preserved,cases=cases,failures=cases+old['blocked'],
        blocked=old['blocked'],numeric_sources=matched,recovered=old['recovered']+additions,
        six_additions=additions,old_preserved=old['preserved'])


def independent_plan(data,checks):
    queues={key:[] for key in CASE_IDS};inventory=[]
    for parent in data['numeric_sources']:
        pool=data['candidate_pools'][parent['id']]
        checks.equal((pool['values'],pool['bucket_xy_m']),(parent['values'],parent['bucket_xy_m']),parent['id']+'/source pool action/layout')
        chosen,_=select_candidates(parent,pool,checks);selected=[]
        for alternative,item in enumerate(chosen):
            c=item['candidate'];checks.equal((c['values'],c['bucket_xy_m']),(parent['values'],parent['bucket_xy_m']),parent['id']+'/exact saved numeric pose')
            selected.append(dict(candidate_index=item['index'],candidate_sha256=digest(c),proximal_changed=item['proximal'],
                maximum_wrapped_q14_delta_rad=float(wrapped_difference(c['q14'],parent['q14']).max())))
            row=recovery.request(parent,f'fixed35-bouncer-search-p{parent["numeric_proposal_index"]:02d}-a{alternative}-{parent["context_id"]}',
                'search','bouncer_search',exact_parent_action=False,numeric_proposal_index=parent['numeric_proposal_index'],
                candidate_index=item['index'],candidate_pool_file=parent['branch_pool_file'],candidate_pool_sha256=parent['branch_pool_sha256'],
                candidate_sha256=digest(c),selected_q10=copy.deepcopy(c['q10']),selected_q14=copy.deepcopy(c['q14']),
                proximal_changed=item['proximal'],alternative_index=alternative)
            queues[parent['context_id']].append(row)
        inventory.append(dict(source_id=parent['id'],source_sha256=digest(parent),context_id=parent['context_id'],
            numeric_proposal_index=parent['numeric_proposal_index'],values=parent['values'],bucket_xy_m=parent['bucket_xy_m'],
            branch_pool_file=parent['branch_pool_file'],branch_pool_sha256=parent['branch_pool_sha256'],selected_count=len(chosen),selected_candidates=selected))
    plan=[]
    for index in range(max(map(len,queues.values()),default=0)):
        for context in CASE_IDS:
            if index<len(queues[context]):plan.append(dict(copy.deepcopy(queues[context][index]),proposal_index=index))
    checks.require(len(plan)<=24 and len({r['id'] for r in plan})==len(plan),'At most24 unique saved bouncer requests')
    return plan,inventory


def independent_preserved(data):
    return [recovery.request(r,'fixed35-bouncer-preserved-'+r['context_id'],'preserved','preserved_recipe',exact_parent_action=True) for r in data['preserved']]


def independent_repeat(discovery,group):
    if not captured(discovery) or group not in range(3):raise EvidenceError('Only captured discoveries select confirmations')
    result=[]
    for repeat in range(3):
        fields={k:copy.deepcopy(discovery[k]) for k in ('proposal_index','numeric_proposal_index','candidate_index','candidate_pool_file','candidate_pool_sha256',
            'candidate_sha256','selected_q10','selected_q14','proximal_changed','alternative_index')}
        row=recovery.request(discovery,f'fixed35-bouncer-repeat-c{group}-r{repeat}-{discovery["context_id"]}','repeat','bouncer_repeat',
            confirmation_index=group,repeat=repeat,selected_from_id=discovery['id'],selected_from_sha256=digest(discovery),
            selected_connection_sha256=discovery['connection_sha256'],**fields)
        row.update(parent_row_id=discovery['parent_row_id'],parent_row_sha256=discovery['parent_row_sha256']);result.append(row)
    return result


def protocol_checks(p,checks):
    literal=dict(substeps=64,grid_positions=35,historical_policy_captures=24,historical_recipe_coverage=30,search_layouts=3,
        numeric_source_poses=12,alternatives_per_pose=2,search_requests_max=24,repeat_per_selected=3,candidates_per_layout=3,
        repeat_requests_max=27,preserved_requests=30,control_requests=18,maximum_requests=99,worker_limit_s=3600,search_limit_s=2400,
        queue_limit_s=600,audit_reserve_s=900,global_limit_s=5400,storage_reserve_bytes=8*1024**3,training=False,actor_steps=0,
        critic_steps=0,noise=False,bucket_perturbations=False,ui_changes=False,new_ik=False,q10_preserved_exactly=True,
        wrapped_q14_tolerance_rad=.05,proximal_joint_indices=[0,1,2],
        evaluation_scope='Posthoc known-grid recipe development; unchanged24/35 policy score, not held-out inference')
    for key,wanted in literal.items():checks.equal(p[key],wanted,'Literal bouncer-only protocol/'+key)


def audit_inputs(study,original,older,parent,spec,checks):
    data=cohort(original,older,parent,checks)
    for source,rows in ((original,data['current']+data['refit']),(older,[r for r in data['old_preserved'] if r['phase']=='search']),
                        (parent,data['numeric_sources']+data['six_additions'])):
        recovery.bind_parent_rows(source,rows,checks)
    pools={}
    for row in data['numeric_sources']+data['six_additions']:
        path=parent['bound']('native-yaw/'+row['branch_pool_file'])
        checks.equal(sha(path),row['branch_pool_sha256'],row['id']+'/published candidate pool')
        pools[row['id']]=read(path)
    data['candidate_pools']=pools
    for name,key in (('parent-current','current'),('parent-refit','refit'),('preserved','preserved'),('cases','cases'),('blocked','blocked'),('numeric-sources','numeric_sources')):
        checks.equal(read(study/f'bootstrap/{name}.json'),data[key],'Exact source cohort/'+name)
    checks.equal(read(study/'candidate-pools.json'),pools,'Only authenticated saved endpoint candidates')
    checks.equal(spec['grid'],fixed.fixed_grid(),'Original35 grid unchanged')
    checks.equal(read(study/'grid.json'),fixed.fixed_grid(),'Saved35 grid unchanged')
    bindings=read(study/'source-bindings.json');checks.equal(spec['source_bindings'],bindings,'All source bindings frozen')
    sources=(('parent',parent,170755,PARENT_FP),('historical.parent',older,141041,previous.RECOVERY_FP),
             ('historical.original',original,40692,recovery.PARENT_FP))
    for key,source,count,fp in sources:
        declared=bindings
        for part in key.split('.'):declared=declared[part]
        wanted=dict(report_sha256=source['report_sha256'],audit_sha256=source['audit_sha256'],contract_fingerprint=fp,
            evidence_manifest_sha256=source['manifest_sha256'],checks_count=count)
        for field,value in wanted.items():checks.equal(declared[field],value,key+'/exact published binding/'+field)
        for name,value in declared['files_sha256'].items():checks.equal(sha(source['bound'](name)),value,key+'/declared source file/'+name)
    checks.equal(bindings['preserved_row_sha256'],{r['id']:digest(r) for r in data['preserved']},'All30 exact preserved source rows')
    checks.equal(bindings['numeric_source_sha256'],{r['id']:digest(r) for r in data['numeric_sources']},'All12 exact matched source rows')
    for name,key in (('prior.json','prior'),('old-prior.json','old_prior'),('controls.json','controls'),('control-replays.json','control_replays'),
        ('calibration.json','calibration'),('setup-home.json','home')):
        value=read(parent['bound'](name));data[key]=value
        checks.equal(read(study/name),value,'Immutable runtime input/'+name)
        checks.equal(value,read(older['bound'](name)),'Runtime input retained from prior recovery/'+name)
    checks.equal((spec['prior_sha256'],spec['old_prior_sha256']),(digest(data['prior']),digest(data['old_prior'])),'Original priors bound')
    checks.equal(read(study/'verification.json'),read(parent['bound']('verification.json')),'Calibrated scoring verification unchanged')
    for r in data['control_replays']:
        for field in ('context_resolution','connection'):
            name='control-parent/native/'+r[field+'_file'];checks.equal(sha(parent['bound'](name)),r[field+'_sha256'],r['id']+'/authentic control proof')
            checks.equal(sha(contained(study,name)),r[field+'_sha256'],r['id']+'/copied control proof unchanged')
    checks.equal(spec['parent_artifacts'],original['report']['artifacts'],'No model promotion or fitting')
    for value in spec['parent_artifacts'].values():checks.equal(sha(original['bound'](value['path'])),value['sha256'],'Original model bytes/'+value['path'])
    checks.require(not list(study.rglob('*.pt')),'No fitted model file')
    for name,key in (('search-plan.json','search_plan_sha256'),('preserved-plan.json','preserved_plan_sha256'),('rule.json','rule_sha256')):
        checks.equal(sha(study/name),spec[key],'Prespecified saved-candidate plan/'+name)
    return data

def audit_controls(rows,data,checks):
    gates={}
    for phase,key in (('controls_pre','pre'),('controls_post','post')):
        group=[r for r in rows if r['phase']==phase];checks.equal(len(group),9,phase+'/nine prescribed controls')
        for i,(source,original,role) in enumerate(zip(data['controls'],data['control_replays'],ROLES)):
            for repeat in range(3):
                row=group[3*i+repeat];label=row['id']
                checks.equal(label,f'fixed35-bouncer-{phase}-{role}-r{repeat}',label+'/literal control identity')
                checks.equal((row['role'],row['method'],row['repeat']),(role,'control',repeat),label+'/matched control metadata')
                checks.equal((row['control_source_id'],row['control_source_sha256']),(source['id'],digest(source)),label+'/original control provenance')
                checks.equal((row['source_record_id'],row['source_record_sha256'],row['variant']),(original['id'],digest(original),'control_exact'),label+'/original control joint branch')
                checks.equal((row['values'],row['bucket_xy_m']),(source['values'],source['bucket_xy_m']),label+'/unchanged control action and position')
                audit_exact_branch(row,original,checks,label)
        gates[key]=native.control_gate(group);checks.equal(gates[key]['status'],'passed',phase+'/unchanged direct/table/negative controls pass')
    return gates


def audit_preserved(rows,data,checks):
    group=[r for r in rows if r['phase']=='preserved'];planned=independent_preserved(data)
    checks.equal(len(group),30,'Exactly one fresh replay of each of30 successful parent recipes')
    for row,packet,parent in zip(group,planned,data['preserved']):
        for key,value in packet.items():checks.equal(row[key],value,row['id']+'/frozen preserved request/'+key)
        audit_exact_branch(row,parent,checks,row['id'])
        checks.require(row['native_cache_sha256']!=parent['native_cache_sha256'],row['id']+'/new native packet, not copied historical outcome')
    return statistics(group)


def audit_budget(study,report,checks):
    path=study.with_name(study.name+'-budget.json');budget=read(path);pipeline=read(study.with_suffix('.pipeline.json'))
    checks.equal((budget['limit_s'],budget['audit_reserve_s']),(3600,900),'One-hour4090worker and15-minute CPU audit reserve')
    checks.equal(budget['gpu_uuid'],'GPU-ab1950b0-01dd-0fab-407f-2d17936b27f4','Only authorized4090')
    checks.require(0<pipeline['global_limit_s']<=5400,'Finite pipeline limit')
    checks.numeric(pipeline['global_deadline_unix_s']-pipeline['started_unix_s'],pipeline['global_limit_s'],'One absolute pipeline interval',.05)
    checks.equal(budget['global_deadline_unix_s'],pipeline['global_deadline_unix_s'],'Shared original deadline')
    checks.require(pipeline['started_unix_s']<=budget['acquired_unix_s'],'Own lease acquired after supervisor start')
    checks.numeric(budget['effective_worker_limit_s'],min(3600,budget['global_deadline_unix_s']-budget['acquired_unix_s']-900),'Acquired worker budget reserves audit',.1)
    checks.require(60<budget['effective_worker_limit_s']<=3600,'Bounded positive worker time')
    for field in ('effective_worker_limit_s','global_deadline_unix_s'):checks.equal(report['deadlines'][field],budget[field],'Reported original deadline/'+field)
    checks.equal(report['deadlines']['worker_limit_s'],3600,'Approved worker cap')
    checks.require(report['elapsed_s']<budget['effective_worker_limit_s'],'Every completed phase within own deadline')
    return dict(worker_receipt_sha256=sha(path),effective_worker_limit_s=budget['effective_worker_limit_s'],global_limit_s=pipeline['global_limit_s'])


def audit_summary(study,rows,report,spec,data,preserved,gates,checks,plan):
    cases=schedule.independent_cases(rows,data,plan,finished=True,budget_truncated=report['budget_truncated'])
    checks.equal(report['cases'],cases,'All five cases and fresh confirmations independently summarized')
    confirmed=[r for r in cases if r['confirmed_recipe'] is not None];kept=preserved['captures']
    preservation=dict(requests=30,captures=kept,expected=30,all_replayed=True,all_recaptured=kept==30)
    coverage=dict(historical_recipe_coverage=30,historical_policy_captures=24,new_confirmed_layouts=len(confirmed),
        fresh_preserved_captures=kept,fresh_total=kept+len(confirmed),positions=35,not_a_policy_score=True)
    checks.equal(report['preservation'],preservation,'All30 fresh preserved outcomes honestly counted')
    checks.equal(report['demonstrated_recipe_coverage'],coverage,'Recipe coverage distinguished from frozen policy24/35')
    checks.equal(report['recovered_layouts'],len(confirmed),'Only three-repeat confirmations add a location')
    checks.equal(report['counts'],statistics(rows),'Controls/rejections/discoveries/repeats all counted')
    phases=('controls_pre','preserved','search','repeat','controls_post')
    checks.equal(report['stage_counts'],{p:statistics([r for r in rows if r['phase']==p]) for p in phases},'Separate phase denominators')
    checks.equal(report['control_gates'],gates,'Pre/post controls reproduced')
    checks.equal((report['expected_requests'],report['maximum_requests'],report['actual_requests']),(99,99,len(rows)),'99 is cap, not executed count')
    checks.equal((report['training'],report['posthoc_development'],report['parent_captures'],report['historical_policy_captures'],report['parent_positions'],report['preserved_recipes']),
        (False,True,30,24,35,30),'Historical policy and recipe counts not conflated')
    checks.equal(report['artifacts'],spec['parent_artifacts'],'Original model checkpoints unchanged')
    checks.equal(report['evaluation_scope'],spec['protocol']['evaluation_scope'],'Development scope unchanged')
    gate=dict(status='passed',substeps=64,controls_passed=True,required_replays_complete=True,
        search_cap_completed=not report['budget_truncated'],no_policy_promotion=True)
    checks.equal(report['gate'],gate,'Finite completion and nonpromotion gate')
    recipes=dict(preserved_parent=[dict(id=r['id'],sha256=digest(r),values=r['values'],q10=r['q10'],q14=r['q14'],bucket_xy_m=r['bucket_xy_m']) for r in data['preserved']],
        new_confirmed=[dict(r['confirmed_recipe'],confirmation_groups=r['confirmation_groups']) for r in confirmed],policy_unchanged=True,preservation=preservation)
    checks.equal(read(study/'recipes.json'),recipes,'Only authenticated retained recipes and confirmed additions')
    return cases,coverage


def audit_candidate_identity(namespace,row,connection,data,checks):
    """Authenticate a recorded alternative against an immutable historical pool."""
    label=row['id'];source=next(r for r in data['numeric_sources'] if r['id']==row['parent_row_id'])
    pool=data['candidate_pools'][source['id']]
    checks.equal(row['parent_row_sha256'],digest(source),label+'/exact matched historical row')
    checks.equal((row['values'],row['bucket_xy_m']),(source['values'],source['bucket_xy_m']),label+'/only bouncer joints may change')
    index=row['candidate_index'];checks.require(type(index) is int and 0<=index<len(pool['endpoint_attempts']['candidates']),label+'/saved candidate index')
    candidate=pool['endpoint_attempts']['candidates'][index]
    chosen,_=select_candidates(source,pool,checks)
    checks.require(index in [r['index'] for r in chosen],label+'/prespecified proximal/wrist alternative')
    checks.equal((candidate['values'],candidate['bucket_xy_m'],candidate['q10']),
        (source['values'],source['bucket_xy_m'],source['q10']),label+'/exact action and unchanged dropper joints')
    checks.require(float(wrapped_difference(candidate['q14'],source['q14']).max())>=.05,label+'/meaningfully different bouncer configuration')
    values=dict(candidate_index=index,candidate_sha256=digest(candidate),candidate_pool_file=source['branch_pool_file'],
        candidate_pool_sha256=source['branch_pool_sha256'],parent_row_id=source['id'],parent_row_sha256=digest(source),
        selected_q10=candidate['q10'],selected_q14=candidate['q14'])
    for key,value in values.items():
        checks.equal(row[key],value,label+'/frozen request identity/'+key)
        checks.equal(connection[key],value,label+'/connection source identity/'+key)
    for name,expected in ((source['branch_pool_file'],source['branch_pool_sha256']),
                          (source['connection_file'],source['connection_sha256'])):
        checks.equal(sha(contained(namespace,name)),expected,label+'/unaltered archived proof/'+name)
    checks.equal(read(contained(namespace,source['branch_pool_file'])),pool,label+'/full source candidate pool unchanged')
    for key,value in dict(parent_connection_file=source['connection_file'],parent_connection_sha256=source['connection_sha256'],
        resolution_file=source['context_resolution_file'],resolution_sha256=source['context_resolution_sha256'],
        bouncer_rule_version=BOUNCER_VERSION,ik_solve_performed=False).items():checks.equal(connection[key],value,label+'/saved-source binding/'+key)
    key=digest(dict(version=BOUNCER_VERSION,resolution_sha256=source['context_resolution_sha256'],
        parent_connection_sha256=source['connection_sha256'],candidate_index=index,candidate_sha256=digest(candidate),
        candidate_pool_sha256=source['branch_pool_sha256'],values=source['values'],bucket_xy_m=source['bucket_xy_m']))
    checks.equal((connection['bouncer_key'],row['bouncer_key'],row['connection_file']),
        (key,key,'fixed-yaw-connections/'+key+'.json'),label+'/complete saved branch cache identity')
    checks.equal((row['bouncer_rule_version'],row['ik_solve_performed']),(BOUNCER_VERSION,False),label+'/no hidden IK re-solve')
    original=read(contained(namespace,source['connection_file']))
    checks.equal((connection['release_behavior'],connection['behavior_key']),(original['release_behavior'],original['behavior_key']),label+'/unchanged bucket-facing release')
    endpoint=connection['endpoint_attempts']
    checks.require(endpoint is not None,label+'/fresh endpoint check required for archived valid layout')
    checks.equal((endpoint['source_candidate'],endpoint['source_candidate_sha256']),(candidate,digest(candidate)),label+'/original candidate not relabelled')
    checks.equal((endpoint['ik_solve_performed'],endpoint['selection_uses_flight_labels']),(False,False),label+'/recheck only without outcome-guided selection')
    checks.equal(set(endpoint),{'passed','reason','source_candidate','source_candidate_sha256','targets','candidates',
        'fresh_endpoint','native_pose_errors','ik_solve_performed','selection_uses_flight_labels'},label+'/literal endpoint recheck schema')
    expected_candidates=[dict(copy.deepcopy(candidate),endpoint=endpoint['fresh_endpoint'])] if endpoint['passed'] else []
    checks.equal(endpoint['candidates'],expected_candidates,label+'/fresh endpoint linked to same saved joints')
    checks.equal(connection['endpoint_eligible'],endpoint['passed'],label+'/eligibility follows recheck')
    if connection['passed']:
        checks.require(endpoint['passed'],label+'/no path admission without valid endpoint')
        for field in ('q10','q14'):checks.equal(connection[field],candidate[field],label+'/admitted saved joints/'+field)
        checks.equal(connection['endpoint'],endpoint['fresh_endpoint'],label+'/admitted fresh endpoint')
        checks.equal(connection['proof'],connection['path_attempts'][-1],label+'/admitted sampled path is last evaluated attempt')
    checks.require(len(connection['path_attempts'])<=2,label+'/only anchor suffix and bounded home fallback')
    if row.get('native_release'):
        checks.require(connection['passed'],label+'/release only after successful setup certification')
        for field in ('q10','q14'):checks.equal(row[field],candidate[field],label+'/native command unchanged/'+field)
    return candidate,endpoint


def audit_geometry(study,rows,scene,assets,data,checks):
    from .learning_ik import read_chain
    from .corrected_geometry import StrictEndpointGate
    from .admission import native_pose_error,base14
    old=[r for r in rows if r['phase'] not in ('search','repeat')]
    previous.audit_geometry(study,old,scene,assets,checks)
    chains=(read_chain(assets/'dropper.urdf'),read_chain(assets/'board_arm.urdf'))
    gate=StrictEndpointGate(scene,assets,.005);seen=set();measured=[]
    for row in rows:
        if row['phase'] not in ('search','repeat'):continue
        label=row['id'];namespace=study/'native-yaw'
        checks.equal(row['adapter_directory'],'native-yaw',label+'/oriented native namespace')
        path=contained(namespace,row['connection_file']);checks.equal(sha(path),row['connection_sha256'],label+'/immutable saved candidate proof')
        connection=read(path);targets,_=fixed.audit_behavior(connection,scene,checks,label)
        candidate,endpoint=audit_candidate_identity(namespace,row,connection,data,checks)
        if row['connection_sha256'] not in seen:
            for field in ('dropper_flange_world','board_flange_local','release_center_m'):
                checks.numeric(endpoint['targets'][field],targets[field],label+'/independent exact target/'+field,1e-12)
            checked=gate.check(candidate['q10'],candidate['q14'],row['bucket_xy_m'],row['values'],targets)
            checks.equal(checked,endpoint['fresh_endpoint'],label+'/independent complete5mm endpoint recheck')
            q=np.r_[candidate['q10'],candidate['q14']]
            # Native model FK corroborates the independent URDF endpoint reconstruction.
            errors=[native_pose_error(gate.strict,role,q,target) for role,target in (
                (10,targets['dropper_flange_world']),(14,base14(scene)@targets['board_flange_local']))]
            for got,wanted in zip(endpoint['native_pose_errors'],errors):
                for field,value in wanted.items():checks.numeric(got[field],value,label+'/native FK recheck/'+field,1e-10)
            accepted=not checked['reason'] and all(e['position_m']<1e-4 and .25*e['rotation_rad']<1e-4 for e in errors)
            checks.equal(endpoint['passed'],accepted,label+'/exact full-pose and strict clearance acceptance')
            checks.equal(endpoint['reason'],None if accepted else 'saved_candidate_endpoint_recheck_rejected',label+'/honest endpoint rejection')
            if accepted:
                checks.require(checked['minimum_clearance_lower_bound_m']>=.005-1e-12,label+'/original5mm endpoint clearance')
                for chain,field,target in zip(chains,('q10','q14'),(targets['dropper_flange_world'],targets['board_flange_local'])):
                    pose=yaw.forward_chain(chain,candidate[field])
                    checks.require(np.linalg.norm(pose[:3,3]-target[:3,3])<1e-4,label+'/saved full FK position/'+field)
                    checks.require(yaw.rotation_error(pose[:3,:3],target[:3,:3])<4e-4,label+'/saved full FK orientation/'+field)
            seen.add(row['connection_sha256'])
        evidence=row['yaw_evidence']
        checks.equal(row['release_behavior_version'],'sircus-continuous-bucket-facing-release-v1',label+'/fixed bucket-facing behavior')
        checks.numeric(evidence['requested_yaw_rad'],targets['yaw_rad'],label+'/display requested yaw',1e-12)
        checks.numeric(evidence['requested_rotation_matrix'],targets['dropper_flange_world'][:3,:3],label+'/display requested rotation',1e-12)
        if not row.get('native_release'):
            checks.require(not row.get('native_initial_q10') and not row.get('native_initial_q14') and not row.get('native_cache_file'),label+'/declined setup has no fabricated flight')
            continue
        checks.require(fixed.target_domain(row['values'],row['bucket_xy_m']),label+'/unchanged numeric bounds')
        poses=[]
        for chain,field,target in zip(chains,('q10','q14'),(targets['dropper_flange_world'],targets['board_flange_local'])):
            checks.equal(row['native_initial_'+field],np.asarray(candidate[field],np.float32).astype(float).tolist(),label+'/actual saved float32 joint reset/'+field)
            pose=yaw.forward_chain(chain,row['native_initial_'+field]);poses.append(pose)
            checks.require(np.linalg.norm(pose[:3,3]-target[:3,3])<1e-4,label+'/actual endpoint position/'+field)
            checks.require(yaw.rotation_error(pose[:3,:3],target[:3,:3])<4e-4,label+'/actual full endpoint orientation/'+field)
        checks.numeric(row['release_ball_center_m'],targets['release_center_m'],label+'/literal release center',1e-12)
        checks.numeric(row['board_center_m'],row['values'][:3],label+'/settled board center',1e-4)
        checks.numeric(row['board_normal'],targets['board_normal'],label+'/settled board normal',4e-4)
        drop,board=poses
        checks.equal(evidence['basis'],'model FK of observed native initial float32 joints; held endpoint',label+'/measured pose provenance')
        displayed=dict(measured_initial_rotation_matrix=drop[:3,:3],measured_initial_yaw_rad=math.atan2(drop[1,0],drop[0,0]),
            rotation_error_rad=yaw.rotation_error(drop[:3,:3],targets['dropper_flange_world'][:3,:3]),
            position_error_m=np.linalg.norm(drop[:3,3]-targets['dropper_flange_world'][:3,3]),
            board_rotation_error_rad=yaw.rotation_error(board[:3,:3],targets['board_flange_local'][:3,:3]),
            board_position_error_m=np.linalg.norm(board[:3,3]-targets['board_flange_local'][:3,3]),
            measured_initial_flange_position_m=drop[:3,3],requested_release_center_m=targets['release_center_m'])
        for key,wanted in displayed.items():checks.numeric(evidence[key],wanted,label+'/independent displayed pose/'+key,1e-9)
        measured.append(dict(record_id=label,yaw_rad=targets['yaw_rad']))
    return dict(new_proofs=len(seen),measured_yaw_releases=len(measured),measurements=measured)


def audit_native(study,rows,data,spec,calibration,assets,checks,parent):
    setup={};count=0
    for namespace,key,parent_namespace in (('native-yaw','prior','native-yaw'),('native-controls','old_prior','control-parent/native')):
        members=[r for r in rows if r['adapter_directory']==namespace];path=study/namespace
        checks.require(bool(members),namespace+'/entire requested cohort retained')
        setup[namespace]=calibrated.check_setup_evidence(geometry._NativeNamespace(path),members,data[key],data['home'],checks)
        admitted=[r for r in members if r.get('native_release')]
        if admitted:
            models.audit_native_options(path,spec['scene'],calibration,assets,checks)
            for row in admitted:checks.equal((row['effective_options_file'],row['effective_options_sha256']),('effective-options.json',sha(path/'effective-options.json')),row['id']+'/same native64 options')
            for name in ('effective-options.json','native-compiled-geometry.json'):
                checks.equal(read(path/name),read(parent['bound'](parent_namespace+'/'+name)),namespace+'/unchanged parent physics/'+name)
        else:checks.require(not list(path.rglob('*.npz')),namespace+'/no fabricated trajectories')
        count+=len(native.rescore_rows(path,members,spec['scene'],calibration,spec['identity']['trace_schema'],assets,checks))
    return setup,count,audit_geometry(study,rows,spec['scene'],assets,data,checks)

def audit(study,source,assets,*,parent_study,parent_audit,original_study,original_audit,recovery_study,recovery_audit,reference):
    started=time.monotonic();checks=AuditChecks();study,source,assets=map(Path,(study,source,assets))
    manifest=replay.check_manifest(study,checks);spec=read(study/'contract.json');report=read(study/'report.json')
    unsigned=dict(spec);fp=unsigned.pop('fingerprint')
    checks.equal((digest(unsigned),report['contract_fingerprint']),(fp,fp),'Frozen complete experiment identity')
    checks.equal((spec['version'],report['version'],report['status'],report['phase']),(STUDY_VERSION,STUDY_VERSION,'completed','completed'),'Completed finite saved-bouncer experiment')
    checks.equal(report['evidence_sha256'],manifest,'Report binds every saved artifact')
    checks.equal(report['protocol'],spec['protocol'],'Protocol frozen');protocol_checks(spec['protocol'],checks)
    checks.equal(spec['release_behavior'],spec['protocol']['rotation'],'Bucket-facing release fixed')
    checks.equal(report['scope'],spec['scope'],'Experiment scope unchanged')
    replay.check_source_files(source,spec['source_sha256'],checks,'Frozen source')
    for module in (Path(__file__),Path(oracle.__file__),Path(fixed.__file__),Path(schedule.__file__),Path(previous.__file__)):
        checks.equal(sha(module),spec['source_sha256']['sircus/mjlab/'+module.name],'Actual audit module frozen/'+module.name)
    rule=read(study/'rule.json');checks.equal(rule,read(source/'sircus/mjlab/fixed35_bouncer_rule.json'),'Checked-in prespecified rule')
    checks.equal((rule['version'],rule['protocol']),(STUDY_VERSION,spec['protocol']),'Rule identity and literal protocol')
    original=recovery.authenticate_parent(original_study,original_audit,checks)
    older=previous.authenticate_recovery(recovery_study,recovery_audit,checks)
    parent=authenticate_parent(parent_study,parent_audit,checks)
    data=audit_inputs(study,original,older,parent,spec,checks)
    models.audit_runtime_reference(source,reference,spec,parent['spec'],assets,checks)
    checks.equal(sha(study/'geometry-audit.json'),sha(parent['bound']('geometry-audit.json')),'Static geometry certificate unchanged from audited parent')
    static=previous.audit_static_geometry(study,spec,assets,checks)
    diagnoses=[r for r in read(parent['bound']('diagnoses.json')) if r['context_id'] in {c['context_id'] for c in data['failures']}]
    checks.equal(read(study/'diagnoses.json'),diagnoses,'Historical diagnoses retained unchanged, not current outcome labels')
    plan,inventory=independent_plan(data,checks);preserved_plan=independent_preserved(data)
    checks.equal(read(study/'search-plan.json'),dict(rows=plan,frozen_before_outcomes=True,maximum_requests=24),'Literal prespecified saved-candidate plan')
    checks.equal(read(study/'preserved-plan.json'),dict(rows=preserved_plan,frozen_before_outcomes=True),'Exact30 source recipe plan')
    checks.equal(read(study/'candidate-inventory.json'),inventory,'Independently selected saved alternative inventory')
    for name,key in (('candidate-inventory.json','candidate_inventory_sha256'),('candidate-pools.json','candidate_pools_sha256')):
        checks.equal(sha(study/name),spec[key],'Frozen saved-pool lineage/'+name)
    sources=data['preserved']+data['numeric_sources']
    receipts=[dict(source_id=r['id'],source_row_sha256=digest(r),resolution_file=r['context_resolution_file'],resolution_sha256=r['context_resolution_sha256'],
        connection_file=r['connection_file'],connection_sha256=r['connection_sha256']) for r in sources]
    checks.equal(read(study/'native-source-receipts.json'),receipts,'All42 exact source proof hydration receipts')
    calibration=replay.check_calibration(study,spec['identity']['calibration'],checks)
    rows=replay.read_records(study/'rows.jsonl');native.check_rows(rows,checks)
    checks.require(not any(r.get('context_id') in {b['context_id'] for b in data['blocked']} for r in rows),'No flight or search in two fixed-base blocked cells')
    setup,native_count,yaw_geometry=audit_native(study,rows,data,spec,calibration,assets,checks,parent)
    preserved=audit_preserved(rows,data,checks);gates=audit_controls(rows,data,checks)
    accounting=schedule.audit_schedule(study,rows,report,data,plan,preserved_plan,independent_repeat,checks)
    packets=geometry.audit_request_packets(study,rows,checks);budget=audit_budget(study,report,checks)
    cases,coverage=audit_summary(study,rows,report,spec,data,preserved,gates,checks,plan)
    return dict(version=VERSION,status='passed',full_checks_verified=True,native_checks_verified=True,
        checks_count=checks.count,checks=checks.entries,failures=[],contract_fingerprint=fp,
        report_sha256=sha(study/'report.json'),evidence_manifest_sha256=sha(study/'evidence-manifest.json'),
        auditor_sha256=sha(Path(__file__)),producer_source_sha256=spec['source_sha256'],
        consumed_source_files_sha256=dict(original=original['consumed'],recovery=older['consumed'],six=parent['consumed']),
        requested_trials=len(rows),native_releases=native_count,cases=cases,control_gates=gates,
        preservation=preserved,demonstrated_recipe_coverage=coverage,static_geometry=static,
        historical_diagnoses=diagnoses,setup_evidence=setup,yaw_geometry=yaw_geometry,
        request_accounting=accounting,request_packets=packets,budget=budget,
        elapsed_s=time.monotonic()-started,maximum_numeric_error=checks.numeric_error,
        limitations=['Historical35 fixed positions: finite saved-branch development, not a learned policy or held-out evaluation.',
            'Two actual fixed-base/bucket intersections remain blocked under unchanged geometry.',
            '30 previous recipes replay once; each discovered addition needs three fresh nominal confirmations.',
            'At most two saved bouncer alternatives per12 previously released poses; no new IK or exhaustive feasibility claim.',
            'Dropper joints, shot coordinates, bucket-facing behavior and native64/v3 scoring remain unchanged.',
            'Held endpoints with sampled setup-path checks; no training, noise,128 qualification, hardware, deployment or UI publication.'])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('study','source','assets','parent-study','parent-audit','original-study','original-audit','recovery-study','recovery-audit','reference','output'):
        parser.add_argument('--'+key,type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists() or args.output.resolve().is_relative_to(args.study.resolve()):
        raise ValueError('Fresh independent proof outside immutable experiment required')
    try:
        kwargs=vars(args).copy();kwargs.pop('output');result=audit(**kwargs)
    except Exception as error:
        args.output.write_text(json.dumps(dict(version=VERSION,status='failed',failures=[type(error).__name__+': '+str(error)]),indent=2,allow_nan=False)+'\n')
        raise
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:result[k] for k in ('status','checks_count','requested_trials','native_releases','demonstrated_recipe_coverage')}))


if __name__=='__main__':main()
