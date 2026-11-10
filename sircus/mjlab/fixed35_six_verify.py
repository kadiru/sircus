"""Independent CPU evidence audit for four finite IK/shot searches and two static blockers.

No producer planner/selector/scorer import; old model outcomes remain historical.
"""
from __future__ import annotations
import argparse
import copy
import json
import math
from pathlib import Path
import time
import numpy as np
from . import fixed35_recovery_verify as previous
from . import fixed35_imitation_verify as fixed
from . import fixed35_six_schedule_verify as schedule
from . import fixed35_six_geometry as static_geometry
from . import benchmark_verify as common
from . import calibrated_benchmark_verify as calibrated
from . import coverage_training_verify as geometry
from . import policy64_verify as models
from . import replay650_verify as replay
from . import table_training_verify as native
from . import table_assisted_score_oracle as oracle
from . import gripper_yaw_verify as yaw
from .benchmark_verify import EvidenceError,contained,digest,read,sha

VERSION='sircus-fixed35-six-audit-v1'
STUDY_VERSION='sircus-fixed35-six-v1'
RECOVERY_REPORT_SHA='2d62f6b43b54210b83c7323b5e46f0ccdf0817f36b67fbe43e1aa7618fb099d9'
RECOVERY_AUDIT_SHA='460e0e5496b54e9426fb170d64b9c0687813eb28e3b7310d3e8f212eda211569'
RECOVERY_FP='a7b58fc50ce6d7b28b0d2a2ab045f4031bc77d3c70145c01a670ec134a003bf5'
RECOVERED_CELLS=((0,0),(1,6),(2,3),(2,6),(4,4))
CASE_CELLS=((1,5),(3,4),(3,5),(4,6))
BLOCKED_CELLS=((3,0),(3,6))
ROLES=previous.ROLES
META=previous.META
AuditChecks=replay.AuditChecks
captured=native.captured
statistics=native.statistics
BRANCH_VERSION='sircus-fixed35-distinct-ik-branches-v1'
BRANCH_RULE=dict(version=BRANCH_VERSION,seeds=[20261110,20261111],starts_per_seed=32,iterations=360,
    normal_only=False,preserve_known=False,maximum_unique_pairs=64,maximum_endpoint_candidates=12,
    maximum_path_candidates=12,maximum_admitted_branches=2,distinct_joint_tolerance_rad=.05,
    periodic_equivalence=True,selection_uses_flight_labels=False)


def authenticate_recovery(directory,audit,checks):
    directory,audit=Path(directory),Path(audit)
    checks.equal(sha(directory/'report.json'),RECOVERY_REPORT_SHA,'Exact published recovery report')
    checks.equal(sha(audit),RECOVERY_AUDIT_SHA,'Exact complete recovery independent audit')
    report,spec,proof,manifest=map(read,(directory/'report.json',directory/'contract.json',audit,directory/'evidence-manifest.json'))
    checks.equal((report['version'],report['status'],proof['version'],proof['status'],proof['full_checks_verified'],proof['native_checks_verified'],proof['failures']),
        ('sircus-fixed35-recovery-v1','completed','sircus-fixed35-recovery-audit-v1','passed',True,True,[]),'Complete recovery source')
    checks.equal((proof['checks_count'],len(proof['checks'])),(141041,141041),'All141041 recovery checks retained')
    checks.require(all(c['passed'] is True for c in proof['checks']),'Every recovery check passed')
    unsigned=dict(spec);fp=unsigned.pop('fingerprint')
    checks.equal((digest(unsigned),fp,report['contract_fingerprint'],proof['contract_fingerprint']),(RECOVERY_FP,)*4,'Original recovery identity')
    checks.equal(proof['report_sha256'],RECOVERY_REPORT_SHA,'Recovery audit report binding')
    checks.equal(proof['evidence_manifest_sha256'],sha(directory/'evidence-manifest.json'),'Recovery manifest binding')
    checks.equal(report['evidence_sha256'],manifest,'Recovery full evidence binding')
    consumed={}
    def bound(name):
        path=contained(directory,name);actual=sha(path)
        checks.equal(actual,manifest[name],'Authenticated recovery file/'+name)
        consumed[name]=actual;return path
    bound('contract.json');rows=common.lines(bound('rows.jsonl'))
    checks.equal(len(rows),441,'All441 recovery attempts retained')
    return dict(directory=directory,spec=spec,report=report,rows=rows,bound=bound,consumed=consumed,
        report_sha256=RECOVERY_REPORT_SHA,audit_sha256=RECOVERY_AUDIT_SHA,manifest_sha256=sha(directory/'evidence-manifest.json'))


def cohort(original,recovery,checks):
    old=previous.partition_parent(original['rows'],checks)
    cases=tuple(f'fixed35-dev-grid-{y:02d}-{x:02d}' for y,x in CASE_CELLS)
    recovered_ids={f'fixed35-dev-grid-{y:02d}-{x:02d}' for y,x in RECOVERED_CELLS}
    by_id={r['id']:r for r in recovery['rows']};checks.equal(len(by_id),441,'Unique authenticated recovery IDs')
    preserved_replays=[r for r in recovery['rows'] if r.get('phase')=='preserved']
    checks.equal(len(preserved_replays),24,'Recovery retained all original24')
    for row,parent in zip(preserved_replays,old['preserved']):
        checks.require(captured(row),row['id']+'/historical preserved replay captured')
        checks.equal((row['parent_row_id'],row['parent_row_sha256']),(parent['id'],digest(parent)),row['id']+'/original source')
        previous.audit_exact_branch(row,parent,checks,row['id'])
    additions=[]
    for c in recovery['report']['cases']:
        source=c.get('confirmed_recipe')
        if source is None:continue
        checks.equal(source,by_id[source['id']],source['id']+'/unmodified authenticated discovery')
        checks.require(captured(source) and source['phase']=='search',source['id']+'/actual captured discovery')
        group=sorted((r for r in recovery['rows'] if r.get('phase')=='repeat' and r.get('selected_from_id')==source['id']),key=lambda r:r['repeat'])
        checks.equal([r['repeat'] for r in group],[0,1,2],source['id']+'/three fresh prior confirmations')
        for row in group:
            checks.require(captured(row),row['id']+'/valid retained prior capture')
            checks.equal(row['selected_from_sha256'],digest(source),row['id']+'/original discovery SHA')
            for key in ('values','bucket_xy_m','q10','q14','connection_sha256'):
                checks.equal(row[key],source[key],row['id']+'/exact prior confirmation/'+key)
        additions.append(source)
    checks.equal({r['context_id'] for r in additions},recovered_ids,'Exact five prior recovered locations')
    preserved=sorted(old['preserved']+additions,key=lambda r:r['context_id'])
    checks.equal((len(preserved),len({r['context_id'] for r in preserved})),(29,29),'29 distinct preserved recipes')
    failed=[r for r in old['current'] if r['context_id'] not in {s['context_id'] for s in preserved}]
    searchable=[r for r in failed if r['context_id'] in cases]
    checks.equal(tuple(r['context_id'] for r in searchable),cases,'Only four prescribed searched locations')
    checks.equal(len(failed),6,'Exactly six remaining locations')
    return dict(current=old['current'],refit=old['refit'],preserved=preserved,recovered=additions,
        failures=failed,cases=searchable,blocked=old['blocked'])


def protocol_checks(value,checks):
    literal=dict(substeps=64,methods=['preserved_recipe','six_search','six_repeat'],grid_positions=35,
        preserved_recipes=29,historical_policy_captures=24,development_failures=6,fixed_base_audits=2,
        search_layouts=4,numeric_proposals_per_layout=96,branch_slots=[0,1],requests_per_layout=192,
        search_requests=768,repeat_per_selected=3,candidates_per_layout=3,repeat_requests_max=36,
        preserved_requests=29,control_requests=18,maximum_requests=851,noise=False,bucket_perturbations=False,
        training=False,actor_steps=0,critic_steps=0,ui_changes=False,worker_limit_s=7200,search_limit_s=5400,
        queue_limit_s=900,global_limit_s=9900,audit_reserve_s=1800,storage_reserve_bytes=8*1024**3,
        rotation='Rz(atan2(bucket_y-board_y,bucket_x-board_x)) @ diag(1,-1,-1)',
        bounds=dict(low=[.30,-.50,.12,8.,.05],high=[1.04,.35,.60,60.,.65]),
        evaluation_scope='Historical fixed-grid recipe development, not held out or a new policy score')
    for key,value_expected in literal.items():checks.equal(value[key],value_expected,'Literal six-location protocol/'+key)


def audit_inputs(study,original,parent,spec,checks):
    data=cohort(original,parent,checks)
    previous.bind_parent_rows(original,data['current']+data['refit'],checks)
    previous.bind_parent_rows(parent,data['recovered'],checks)
    for name,key in (('parent-current','current'),('parent-refit','refit'),('preserved','preserved'),('cases','cases'),('blocked','blocked')):
        checks.equal(read(study/f'bootstrap/{name}.json'),data[key],'Unchanged authenticated cohort/'+name)
    checks.equal((spec['grid'],read(study/'grid.json')),(fixed.fixed_grid(),)*2,'Only canonical35 without jitter')
    bindings=read(study/'source-bindings.json');checks.equal(spec['source_bindings'],bindings,'Frozen lineage')
    for key,source,count,fp in (('parent',parent,141041,RECOVERY_FP),('original',original,40692,previous.PARENT_FP)):
        declared=bindings[key]
        expected=dict(report_sha256=source['report_sha256'],audit_sha256=source['audit_sha256'],contract_fingerprint=fp,
            evidence_manifest_sha256=source['manifest_sha256'],checks_count=count)
        for field,value in expected.items():checks.equal(declared[field],value,key+'/authenticated lineage/'+field)
        for name,value in declared['files_sha256'].items():checks.equal(sha(source['bound'](name)),value,key+'/declared source file/'+name)
    checks.equal(bindings['recipe_source'],{r['id']:dict(study='recovery' if r in data['recovered'] else 'original',row_sha256=digest(r)) for r in data['preserved']+data['failures']},'Every recipe has exact immutable source')
    for name,key in (('prior.json','prior'),('old-prior.json','old_prior'),('controls.json','controls'),
        ('control-replays.json','control_replays'),('calibration.json','calibration'),('setup-home.json','home')):
        value=read(parent['bound'](name));checks.equal(value,read(original['bound'](name)),'Shared old/recovery runtime input/'+name)
        checks.equal(read(study/name),value,'Current immutable runtime input/'+name);data[key]=value
    checks.equal((len(data['prior']['records']),len(data['old_prior']['records'])),(19,17),'Geometry anchors unchanged')
    checks.equal((spec['prior_sha256'],spec['old_prior_sha256']),(digest(data['prior']),digest(data['old_prior'])),'Both geometry priors bound')
    checks.equal(read(study/'verification.json'),read(parent['bound']('verification.json')),'Original native calibration receipt')
    for r in data['control_replays']:
        for field in ('context_resolution','connection'):
            name='control-parent/native/'+r[field+'_file']
            checks.equal(sha(parent['bound'](name)),r[field+'_sha256'],r['id']+'/original control proof')
            checks.equal(sha(contained(study,name)),r[field+'_sha256'],r['id']+'/copied control proof')
    artifacts=original['report']['artifacts'];checks.equal(spec['parent_artifacts'],artifacts,'No model replacement')
    for value in artifacts.values():checks.equal(sha(original['bound'](value['path'])),value['sha256'],'Original model file/'+value['path'])
    checks.require(not list(study.rglob('*.pt')),'No newly fitted model')
    for name,key in (('search-plan.json','search_plan_sha256'),('preserved-plan.json','preserved_plan_sha256'),('rule.json','rule_sha256')):
        checks.equal(sha(study/name),spec[key],'Frozen pre-outcome plan/'+name)
    return data


def audit_branch_pool(root,row,connection,scene,checks,label,*,home):
    """Bind selected slot to the complete pre-flight geometric branch pool."""
    from .calibrated_admission import verify_path
    filename=connection['branch_pool_file'];path=contained(root,filename)
    checks.equal(sha(path),connection['branch_pool_sha256'],label+'/immutable two-slot pool bytes')
    pool=read(path)
    checks.equal(pool['branch_rule'],BRANCH_RULE,label+'/literal finite branch rule')
    checks.equal((pool['version'],pool['selection_uses_flight_labels']),(BRANCH_VERSION,False),label+'/geometry-only pool')
    for key in ('behavior_key','release_behavior','bucket_xy_m','values','resolution_file','resolution_sha256','home_sha256'):
        checks.equal(pool[key],connection[key],label+'/pool input/'+key)
    poolkey=digest(dict(version=BRANCH_VERSION,branch_rule=BRANCH_RULE,behavior_key=pool['behavior_key'],resolution_sha256=pool['resolution_sha256']))
    checks.equal((pool['pool_key'],filename,connection['branch_pool_key']),(poolkey,'branch-pools/'+poolkey+'.json',poolkey),label+'/pool content-addressed identity')
    slot=row['ik_branch_slot'];checks.require(type(slot) is int and slot in (0,1),label+'/frozen branch slot')
    key=digest(dict(version=BRANCH_VERSION,pool_key=poolkey,branch_pool_sha256=sha(path),ik_branch_slot=slot))
    checks.equal((connection['branch_key'],connection['ik_branch_slot'],row['branch_key'],row['connection_file']),
        (key,slot,key,'fixed-yaw-connections/'+key+'.json'),label+'/selected slot proof key')
    for key in ('branch_pool_file','branch_pool_sha256','branch_pool_key','branch_rule_version'):
        checks.equal(row[key],connection[key],label+'/row branch provenance/'+key)
    checks.equal(connection['branch_rule_version'],BRANCH_VERSION,label+'/branch version')
    checks.equal(connection['endpoint_attempts'],pool['endpoint_attempts'],label+'/complete pre-flight endpoint evidence retained')
    checks.equal(connection['path_attempts'],pool['path_attempts'],label+'/complete sampled-path evidence retained')
    admitted=pool['admitted'];checks.require(len(admitted)<=2,label+'/at most two admitted geometric branches')
    if pool['endpoint_attempts'] is not None:
        checks.require(len(pool['endpoint_attempts']['candidates'])<=12,label+'/at most12 endpoint/path candidates')
    checks.require(len(pool['path_attempts'])<=24,label+'/at most12 candidates with one suffix/fallback each')
    checks.equal([a['candidate_index'] for a in admitted],sorted({a['candidate_index'] for a in admitted}),label+'/admitted candidate order is monotone and unique')
    checks.equal([a['path_attempt_index'] for a in admitted],[i for i,p in enumerate(pool['path_attempts']) if p.get('passed')],label+'/all passed paths retained in first-valid order')
    if len(admitted)==2:
        first=np.r_[admitted[0]['q10'],admitted[0]['q14']];second=np.r_[admitted[1]['q10'],admitted[1]['q14']]
        wrapped=(first-second+np.pi)%(2*np.pi)-np.pi
        checks.require(np.max(np.abs(wrapped))>=.05,label+'/branches differ beyond IK numerical noise and2pi wraps')
    for i,branch in enumerate(admitted):
        checks.equal(branch['slot'],i,label+'/deterministic admitted slot order')
        candidate=pool['endpoint_attempts']['candidates'][branch['candidate_index']]
        proof=pool['path_attempts'][branch['path_attempt_index']]
        checks.require(proof['passed'],label+'/admitted branch has a passed unchanged path')
        checks.require(verify_path(proof,np.r_[home['q10'],home['q14']],np.r_[branch['q10'],branch['q14']]),label+'/every admitted slot has full forward/reverse sampled proof')
        for field in ('q10','q14','endpoint'):checks.equal(branch[field],candidate[field],label+'/candidate-slot joint binding/'+field)
        checks.numeric(proof['waypoints'][-1],branch['q10']+branch['q14'],label+'/path endpoint equals exact branch',0)
    checks.equal(bool(connection['passed']),slot<len(admitted),label+'/no substitute for unavailable requested branch')
    if slot<len(admitted):
        branch=admitted[slot]
        for field in ('q10','q14','endpoint'):checks.equal(connection[field],branch[field],label+'/selected branch exact/'+field)
        checks.equal(connection['proof'],pool['path_attempts'][branch['path_attempt_index']],label+'/selected branch exact sampled proof')
    else:checks.require(not row['native_release'],label+'/missing branch never releases')
    return pool


def audit_branch_endpoint(endpoint,checks,label):
    checks.equal(endpoint['branch_rule'],BRANCH_RULE,label+'/complete literal branch solve rule')
    checks.require(len(endpoint['pair_attempts'])<=64,label+'/finite unique-pair solve cap')
    if 'targets' not in endpoint:
        checks.equal(endpoint['candidates'],[],label+'/early rejection has no posed candidate')
        return
    checks.equal(endpoint['seeds'],[20261110,20261111],label+'/two prescribed deterministic seeds')
    pools=endpoint['ik_pools'];checks.equal(len(pools),2,label+'/both arms recorded')
    expected=[dict(seed=seed,start_index=i) for seed in (20261110,20261111) for i in range(33)]
    for i,pool in enumerate(pools):
        checks.equal((len(pool['q']),len(pool['valid']),len(pool['residual'])),(66,66,66),label+'/full finite solve evidence')
        checks.equal(pool['provenance'],expected,label+'/start identity preserved')
        q=np.asarray(pool['q']);checks.require(q.shape==(66,6) and np.isfinite(q).all(),label+'/finite six-joint solutions')
        checks.require(all(type(v) is bool for v in pool['valid']),label+'/actual boolean validity flags')
        checks.equal(endpoint['valid_ik_branches'][i],sum(pool['valid']),label+'/full valid-branch count')
    seen=[]
    for attempt in endpoint['pair_attempts']:
        i,j=attempt['branch10'],attempt['branch14']
        checks.require(type(i) is int and type(j) is int and 0<=i<66 and 0<=j<66,label+'/pair indices within frozen pool')
        checks.require(pools[0]['valid'][i] and pools[1]['valid'][j],label+'/pair was marked full-pose IK valid')
        checks.equal((attempt['seed_start10'],attempt['seed_start14']),(expected[i],expected[j]),label+'/pair seed provenance')
        q=np.r_[pools[0]['q'][i],pools[1]['q'][j]]
        checks.require(all(np.max(np.abs((q-p+np.pi)%(2*np.pi)-np.pi))>=.05 for p in seen),label+'/no numerical clone consumes branch cap')
        seen.append(q)
    for candidate in endpoint['candidates']:
        i,j=candidate['branch10'],candidate['branch14']
        checks.equal((candidate['q10'],candidate['q14']),(pools[0]['q'][i],pools[1]['q'][j]),label+'/candidate exactly one recorded pool pair')
        matching=[a for a in endpoint['pair_attempts'] if (a['branch10'],a['branch14'])==(i,j)]
        checks.equal(len(matching),1,label+'/candidate has one checked pair')
        checks.equal(candidate['endpoint'],matching[0]['checked'],label+'/candidate endpoint proof retained')
        checks.equal(candidate['endpoint']['reason'],0,label+'/candidate passed strict gate')


def audit_native(study,rows,data,spec,calibration,assets,checks,parent):
    setup={};count=0
    for namespace,key,parent_namespace in (('native-yaw','prior','native-yaw'),('native-controls','old_prior','control-parent/native')):
        members=[r for r in rows if r['adapter_directory']==namespace];path=study/namespace
        checks.require(bool(members),namespace+'/entire requested cohort retained')
        setup[namespace]=calibrated.check_setup_evidence(geometry._NativeNamespace(path),members,data[key],data['home'],checks)
        admitted=[r for r in members if r.get('native_release')]
        if admitted:
            models.audit_native_options(path,spec['scene'],calibration,assets,checks)
            for row in admitted:
                checks.equal((row['effective_options_file'],row['effective_options_sha256']),('effective-options.json',sha(path/'effective-options.json')),row['id']+'/same native64 options')
            for name in ('effective-options.json','native-compiled-geometry.json'):
                checks.equal(read(path/name),read(parent['bound'](parent_namespace+'/'+name)),namespace+'/same physics as parent/'+name)
        else:checks.require(not list(path.rglob('*.npz')),namespace+'/no fabricated native trajectory')
        count+=len(native.rescore_rows(path,members,spec['scene'],calibration,spec['identity']['trace_schema'],assets,checks))
    return setup,count,audit_geometry(study,rows,spec['scene'],assets,checks)


def audit_exact_branch(row,parent,checks,label):
    for key in (*META,'values','bucket_xy_m','q10','q14','connection_file','connection_sha256','native_release'):
        checks.equal(row.get(key),parent.get(key),label+'/preserved exact admitted branch/'+key)
    if row.get('native_release'):
        for field in ('q10','q14'):
            checks.equal(row['native_initial_'+field],np.asarray(parent[field],np.float32).astype(float).tolist(),label+'/observed original native reset/'+field)

def audit_controls(rows,data,checks):
    gates={}
    for phase,key in (('controls_pre','pre'),('controls_post','post')):
        group=[r for r in rows if r['phase']==phase];checks.equal(len(group),9,phase+'/nine prescribed controls')
        for i,(source,original,role) in enumerate(zip(data['controls'],data['control_replays'],ROLES)):
            for repeat in range(3):
                row=group[3*i+repeat];label=row['id']
                checks.equal(label,f'fixed35-six-{phase}-{role}-r{repeat}',label+'/literal control identity')
                checks.equal((row['role'],row['method'],row['repeat']),(role,'control',repeat),label+'/matched control metadata')
                checks.equal((row['control_source_id'],row['control_source_sha256']),(source['id'],digest(source)),label+'/original control provenance')
                checks.equal((row['source_record_id'],row['source_record_sha256'],row['variant']),(original['id'],digest(original),'control_exact'),label+'/original control joint branch')
                checks.equal((row['values'],row['bucket_xy_m']),(source['values'],source['bucket_xy_m']),label+'/unchanged control action and position')
                audit_exact_branch(row,original,checks,label)
        gates[key]=native.control_gate(group);checks.equal(gates[key]['status'],'passed',phase+'/unchanged direct/table/negative controls pass')
    return gates

def audit_preserved(rows,data,checks):
    group=[r for r in rows if r['phase']=='preserved'];planned=schedule.independent_preserved(data)
    checks.equal(len(group),29,'Exactly one fresh replay of each of29 successful parent recipes')
    for row,packet,parent in zip(group,planned,data['preserved']):
        for key,value in packet.items():checks.equal(row[key],value,row['id']+'/frozen preserved request/'+key)
        audit_exact_branch(row,parent,checks,row['id'])
        checks.require(row['native_cache_sha256']!=parent['native_cache_sha256'],row['id']+'/new native packet, not copied historical outcome')
    return statistics(group)

def audit_budget(study,report,checks):
    path=study.with_name(study.name+'-budget.json');budget=read(path);pipeline=read(study.with_suffix('.pipeline.json'))
    checks.equal((budget['limit_s'],budget['audit_reserve_s']),(7200,1800),'Two-hour4090worker and half-hour CPU audit reserve')
    checks.equal(budget['gpu_uuid'],'GPU-ab1950b0-01dd-0fab-407f-2d17936b27f4','Only authorized4090')
    checks.require(0<pipeline['global_limit_s']<=9900,'Finite pipeline limit')
    checks.numeric(pipeline['global_deadline_unix_s']-pipeline['started_unix_s'],pipeline['global_limit_s'],'One absolute pipeline interval',.05)
    checks.equal(budget['global_deadline_unix_s'],pipeline['global_deadline_unix_s'],'Shared original deadline')
    checks.require(pipeline['started_unix_s']<=budget['acquired_unix_s'],'Own lease acquired after supervisor start')
    checks.numeric(budget['effective_worker_limit_s'],min(7200,budget['global_deadline_unix_s']-budget['acquired_unix_s']-1800),'Acquired worker budget reserves audit',.1)
    checks.require(60<budget['effective_worker_limit_s']<=7200,'Bounded positive worker time')
    for field in ('effective_worker_limit_s','global_deadline_unix_s'):checks.equal(report['deadlines'][field],budget[field],'Reported original deadline/'+field)
    checks.equal(report['deadlines']['worker_limit_s'],7200,'Approved worker cap')
    checks.require(report['elapsed_s']<budget['effective_worker_limit_s'],'Every completed phase within own deadline')
    return dict(worker_receipt_sha256=sha(path),effective_worker_limit_s=budget['effective_worker_limit_s'],global_limit_s=pipeline['global_limit_s'])

def audit_diagnoses(study,parent,data,checks):
    expected=[];cache={}
    for row in data['failures']:
        item=dict(context_id=row['context_id'],bucket_xy_m=row['bucket_xy_m'],values=row['values'],parent_row_id=row['id'],parent_row_sha256=digest(row),native_release=row['native_release'],invalid=row.get('invalid'))
        if row['native_release']:
            name='native-yaw/'+row['native_cache_file']
            if name not in cache:
                with np.load(parent['bound'](name),allow_pickle=False) as z:cache[name]={key:z[key] for key in ('position_m','forces_N','time_s')}
            z=cache[name];idx=row['native_cache_row'];p=z['position_m'][:,idx];f=z['forces_N'][:,idx];t=z['time_s']
            contacts={}
            for i,name in enumerate(('board','bucket','table','floor','robot','unattributed')):
                hits=np.flatnonzero(f[:,i]>.01);contacts[name]=float(t[hits[0]+1]) if len(hits) else None
            j=int(np.argmin(np.linalg.norm(p[:,:2]-row['bucket_xy_m'],axis=1)))
            robot=contacts['robot'] is not None and (contacts['bucket'] is None or contacts['robot']<contacts['bucket'])
            kind='robot_before_bucket' if robot else 'bucket_contact_without_entry' if contacts['bucket'] is not None else 'no_bucket_contact'
            item.update(category=kind,first_contacts_s=contacts,
                closest_xy=dict(time_s=float(t[j]),distance_m=float(np.linalg.norm(p[j,:2]-row['bucket_xy_m'])),ball_center_z_m=float(p[j,2])),
                whole_ball_entry=bool(row.get('whole_ball_entry')),max_retention_s=row.get('max_retention_s'),rim_band_contact=bool(row.get('rim_band_contact')),native_cache_sha256=row['native_cache_sha256'])
        else:item['category']='fixed_base_blocked' if row.get('invalid')=='fixed_base_clearance_rejected' else 'finite_setup_unresolved'
        expected.append(item)
    actual=read(study/'diagnoses.json');checks.equal(len(actual),6,'All6 remaining failures explained')
    for got,wanted in zip(actual,expected):
        for key,value in wanted.items():checks.equal(got[key],value,wanted['context_id']+'/independent historical diagnosis/'+key)
        if wanted['native_release']:
            checks.equal(got['contact_force_threshold_N'],.01,wanted['context_id']+'/scorer contact threshold')
            checks.require('scoring termination' in got['contact_scope'].lower() and 'full' in got['contact_scope'].lower(),wanted['context_id']+'/late contact timing scope disclosed')
    return actual


def audit_geometry(study,rows,scene,assets,checks):
    from .learning_ik import read_chain
    from .corrected_geometry import StrictEndpointGate
    from .calibrated_admission import source_geometry
    chains=(read_chain(assets/'dropper.urdf'),read_chain(assets/'board_arm.urdf'))
    gate=StrictEndpointGate(scene,assets,.005);seen=set();measured=[]
    for row in rows:
        label=row['id'];is_test=row['adapter_directory']=='native-yaw';namespace=row['adapter_directory']
        checks.equal(namespace,'native-yaw' if is_test else 'native-controls',label+'/explicit oriented policy/control namespace')
        targets=fixed.expected_targets(scene,row['values'],row['bucket_xy_m'],facing=is_test)
        if is_test:
            checks.equal(row['release_behavior_version'],'sircus-continuous-bucket-facing-release-v1',label+'/both policies use same fixed behavior')
            path=contained(study/namespace,row['connection_file']);checks.equal(sha(path),row['connection_sha256'],label+'/immutable yaw proof')
            connection=read(path);_,key=fixed.audit_behavior(connection,scene,checks,label)
            is_branch=connection.get('branch_rule_version') is not None
            if is_branch:audit_branch_pool(study/namespace,row,connection,scene,checks,label,home=read(study/'setup-home.json'))
            else:checks.equal(row['connection_file'],'fixed-yaw-connections/'+key+'.json',label+'/original exact-yaw cache namespace')
            if row['connection_sha256'] not in seen:
                endpoint=connection.get('endpoint_attempts')
                if endpoint is not None:
                    if is_branch:audit_branch_endpoint(endpoint,checks,label)
                    checks.require(len(endpoint['candidates'])<=(12 if is_branch else 6),label+'/bounded finite candidate pairs')
                    if endpoint.get('targets') is not None:
                        for field,wanted in dict(starts=32 if is_branch else 16,iterations=360 if is_branch else 180,normal_only=False,selection_uses_flight_labels=False).items():
                            checks.equal(endpoint[field],wanted,label+'/finite exact-pose IK/'+field)
                        for field in ('dropper_flange_world','board_flange_local','release_center_m'):
                            checks.numeric(endpoint['targets'][field],targets[field],label+'/independent IK target/'+field,1e-12)
                        resolution=read(contained(study/namespace,row['context_resolution_file']))
                        hint=source_geometry(resolution['effective_source_hint'])
                        if resolution.get('resolved'):hint.update({k:resolution['resolved'][k] for k in ('q10','q14','values','bucket_xy_m')})
                        checks.equal(endpoint['geometry_hint_sha256'],digest(hint),label+'/IK seed is geometry-only pre-policy anchor')
                        checks.require(len(endpoint['valid_ik_branches'])==2 and all(type(x) is int and 0<=x<=(66 if is_branch else 17) for x in endpoint['valid_ik_branches']),label+'/bounded recorded IK branches')
                    for candidate in endpoint['candidates']:
                        checks.equal((candidate['bucket_xy_m'],candidate['values']),(row['bucket_xy_m'],row['values']),label+'/candidate solves exact actor coordinates')
                        checked=gate.check(candidate['q10'],candidate['q14'],row['bucket_xy_m'],row['values'],targets)
                        checks.equal(checked['reason'],0,label+'/candidate passes independently recomputed endpoint clearance')
                        checks.require(checked['minimum_clearance_lower_bound_m']>=.005-1e-12,label+'/unchanged5mm strict clearance')
                        for chain,field,target in zip(chains,('q10','q14'),(targets['dropper_flange_world'],targets['board_flange_local'])):
                            actual=yaw.forward_chain(chain,candidate[field])
                            checks.require(np.linalg.norm(actual[:3,3]-target[:3,3])<1e-4,label+'/candidate position/'+field)
                            checks.require(yaw.rotation_error(actual[:3,:3],target[:3,:3])<4e-4,label+'/candidate full rotation/'+field)
                seen.add(row['connection_sha256'])
            evidence=row['yaw_evidence']
            checks.numeric(evidence['requested_yaw_rad'],targets['yaw_rad'],label+'/display requested yaw',1e-12)
            checks.numeric(evidence['requested_rotation_matrix'],targets['dropper_flange_world'][:3,:3],label+'/display requested rotation',1e-12)
        if not row.get('native_release'):
            checks.require(not row.get('native_initial_q10') and not row.get('native_initial_q14') and not row.get('native_cache_file'),label+'/no fabricated reset or flight for refused setup')
            continue
        if is_test:checks.require(fixed.target_domain(row['values'],row['bucket_xy_m']),label+'/released action within original bounds and distance greater than150mm')
        observed=[]
        for chain,field,target in zip(chains,('q10','q14'),(targets['dropper_flange_world'],targets['board_flange_local'])):
            checks.equal(row['native_initial_'+field],np.asarray(row[field],np.float32).astype(float).tolist(),label+'/actual observed float32 joints/'+field)
            pose=yaw.forward_chain(chain,row['native_initial_'+field]);observed.append(pose)
            checks.require(np.linalg.norm(pose[:3,3]-target[:3,3])<1e-4,label+'/actual endpoint position/'+field)
            checks.require(yaw.rotation_error(pose[:3,:3],target[:3,:3])<4e-4,label+'/actual full endpoint orientation/'+field)
        checks.numeric(row['release_ball_center_m'],targets['release_center_m'],label+'/literal release center',1e-12)
        checks.numeric(row['board_center_m'],row['values'][:3],label+'/settled board center',1e-4)
        checks.numeric(row['board_normal'],targets['board_normal'],label+'/settled board normal',4e-4)
        if is_test:
            drop,board=observed
            checks.equal(evidence['basis'],'model FK of observed native initial float32 joints; held endpoint',label+'/honest measured-pose basis')
            values=dict(measured_initial_rotation_matrix=drop[:3,:3],measured_initial_yaw_rad=math.atan2(drop[1,0],drop[0,0]),
                rotation_error_rad=yaw.rotation_error(drop[:3,:3],targets['dropper_flange_world'][:3,:3]),
                position_error_m=np.linalg.norm(drop[:3,3]-targets['dropper_flange_world'][:3,3]),
                board_rotation_error_rad=yaw.rotation_error(board[:3,:3],targets['board_flange_local'][:3,:3]),
                board_position_error_m=np.linalg.norm(board[:3,3]-targets['board_flange_local'][:3,3]),
                measured_initial_flange_position_m=drop[:3,3],requested_release_center_m=targets['release_center_m'])
            for key,wanted in values.items():checks.numeric(evidence[key],wanted,label+'/independent displayed pose/'+key,1e-9)
            measured.append(dict(record_id=label,yaw_rad=targets['yaw_rad'],rotation_error_rad=values['rotation_error_rad']))
    return dict(new_proofs=len(seen),measured_yaw_releases=len(measured),measurements=measured)


def audit_summary(study,rows,report,spec,data,preserved,gates,checks):
    cases=schedule.independent_cases(rows,data,finished=True,budget_truncated=report['budget_truncated'])
    checks.equal(report['cases'],cases,'All six cases and fresh confirmations independently summarized')
    confirmed=[r for r in cases if r['confirmed_recipe'] is not None];kept=preserved['captures']
    preservation=dict(requests=29,captures=kept,expected=29,all_replayed=True,all_recaptured=kept==29)
    coverage=dict(historical_recipe_coverage=29,historical_policy_captures=24,new_confirmed_layouts=len(confirmed),
        fresh_preserved_captures=kept,fresh_total=kept+len(confirmed),positions=35,not_a_policy_score=True)
    checks.equal(report['preservation'],preservation,'All29 fresh preserved outcomes honestly counted')
    checks.equal(report['demonstrated_recipe_coverage'],coverage,'Recipe coverage distinguished from frozen policy24/35')
    checks.equal(report['recovered_layouts'],len(confirmed),'Only three-repeat confirmations add a location')
    checks.equal(report['counts'],statistics(rows),'Controls/rejections/discoveries/repeats all counted')
    phases=('controls_pre','preserved','search','repeat','controls_post')
    checks.equal(report['stage_counts'],{p:statistics([r for r in rows if r['phase']==p]) for p in phases},'Separate phase denominators')
    checks.equal(report['control_gates'],gates,'Pre/post controls reproduced')
    checks.equal((report['expected_requests'],report['maximum_requests'],report['actual_requests']),(851,851,len(rows)),'851 is cap, not executed count')
    checks.equal((report['training'],report['posthoc_development'],report['parent_captures'],report['historical_policy_captures'],report['parent_positions'],report['preserved_recipes']),
        (False,True,29,24,35,29),'Historical policy and recipe counts not conflated')
    checks.equal(report['artifacts'],spec['parent_artifacts'],'Original model checkpoints unchanged')
    checks.equal(report['evaluation_scope'],spec['protocol']['evaluation_scope'],'Development scope unchanged')
    gate=dict(status='passed',substeps=64,controls_passed=True,required_replays_complete=True,
        search_cap_completed=not report['budget_truncated'],no_policy_promotion=True)
    checks.equal(report['gate'],gate,'Finite completion and nonpromotion gate')
    recipes=dict(preserved_parent=[dict(id=r['id'],sha256=digest(r),values=r['values'],q10=r['q10'],q14=r['q14'],bucket_xy_m=r['bucket_xy_m']) for r in data['preserved']],
        new_confirmed=[dict(r['confirmed_recipe'],confirmation_groups=r['confirmation_groups']) for r in confirmed],policy_unchanged=True,preservation=preservation)
    checks.equal(read(study/'recipes.json'),recipes,'Only authenticated retained recipes and confirmed additions')
    return cases,coverage


def audit_static_geometry(study,spec,assets,checks):
    report=read(study/'geometry-audit.json')
    checks.equal(sha(study/'geometry-audit.json'),spec['geometry_audit_sha256'],'Static geometry frozen before flight')
    measured=static_geometry.audit_geometry(spec['scene'],assets)
    checks.equal(report,measured,'Independent CPU recomputation of actual fixed-collider geometry')
    checks.equal([r['context_id'] for r in report['cases']],[c[0] for c in static_geometry.DEFAULT_CASES],'Two prescribed blockers and one clear geometry control')
    checks.equal([r['classification'] for r in report['cases']],['actual_fixed_base_overlap','actual_fixed_base_overlap','proven_clearance'],'Actual intersections distinguished from conservative AABB artifacts')
    for row in report['cases'][:2]:
        checks.require(row['certified_overlap_pairs']>0,row['context_id']+'/strict common-interior witness exists')
        checks.require(row['minimum_validated_signed_distance_m']<0,row['context_id']+'/native signed penetration corroborates independent intersection')
    checks.equal((report['guard_changed'],report['physics_steps'],report['gpu_used']),(False,0,False),'Static geometry is diagnostic only')
    return dict(sha256=sha(study/'geometry-audit.json'),version=report['version'],
        cases=[{k:r[k] for k in ('context_id','bucket_xy_m','classification','certified_overlap_pairs','minimum_validated_signed_distance_m')} for r in report['cases']])


def audit(study,source,assets,*,parent_study,parent_audit,original_study,original_audit,reference):
    started=time.monotonic();checks=AuditChecks();study,source,assets=map(Path,(study,source,assets))
    manifest=replay.check_manifest(study,checks);spec=read(study/'contract.json');report=read(study/'report.json')
    unsigned=dict(spec);fp=unsigned.pop('fingerprint')
    checks.equal((digest(unsigned),report['contract_fingerprint']),(fp,fp),'Frozen complete experiment identity')
    checks.equal((spec['version'],report['version'],report['status'],report['phase']),(STUDY_VERSION,STUDY_VERSION,'completed','completed'),'Completed finite six-location experiment')
    checks.equal(report['evidence_sha256'],manifest,'Report binds every saved artifact')
    checks.equal(report['protocol'],spec['protocol'],'Protocol frozen');protocol_checks(spec['protocol'],checks)
    checks.equal(spec['release_behavior'],spec['protocol']['rotation'],'Bucket-facing release fixed')
    checks.equal(report['scope'],spec['scope'],'Experiment scope unchanged')
    replay.check_source_files(source,spec['source_sha256'],checks,'Frozen source')
    for module in (Path(__file__),Path(oracle.__file__),Path(fixed.__file__),Path(schedule.__file__),Path(static_geometry.__file__)):
        checks.equal(sha(module),spec['source_sha256']['sircus/mjlab/'+module.name],'Actual audit module frozen/'+module.name)
    rule=read(study/'rule.json');checks.equal(rule,read(source/'sircus/mjlab/fixed35_six_rule.json'),'Checked-in prespecified rule')
    checks.equal((rule['version'],rule['protocol']),(STUDY_VERSION,spec['protocol']),'Rule identity and literal protocol')
    original=previous.authenticate_parent(original_study,original_audit,checks)
    parent=authenticate_recovery(parent_study,parent_audit,checks)
    data=audit_inputs(study,original,parent,spec,checks)
    models.audit_runtime_reference(source,reference,spec,parent['spec'],assets,checks)
    diagnoses=audit_diagnoses(study,original,data,checks)
    static=audit_static_geometry(study,spec,assets,checks)
    plan=schedule.audit_plan(study,data,checks)
    calibration=replay.check_calibration(study,spec['identity']['calibration'],checks)
    rows=replay.read_records(study/'rows.jsonl');native.check_rows(rows,checks)
    checks.require(not any(r.get('context_id') in {b['context_id'] for b in data['blocked']} for r in rows),'No flight or search in two blocked cells')
    setup,native_count,yaw_geometry=audit_native(study,rows,data,spec,calibration,assets,checks,parent)
    preserved=audit_preserved(rows,data,checks);gates=audit_controls(rows,data,checks)
    accounting=schedule.audit_schedule(study,rows,report,data,plan,checks)
    packets=geometry.audit_request_packets(study,rows,checks);budget=audit_budget(study,report,checks)
    cases,coverage=audit_summary(study,rows,report,spec,data,preserved,gates,checks)
    return dict(version=VERSION,status='passed',full_checks_verified=True,native_checks_verified=True,
        checks_count=checks.count,checks=checks.entries,failures=[],contract_fingerprint=fp,
        report_sha256=sha(study/'report.json'),evidence_manifest_sha256=sha(study/'evidence-manifest.json'),
        auditor_sha256=sha(Path(__file__)),producer_source_sha256=spec['source_sha256'],
        consumed_source_files_sha256=dict(original=original['consumed'],recovery=parent['consumed']),
        requested_trials=len(rows),native_releases=native_count,cases=cases,control_gates=gates,
        preservation=preserved,demonstrated_recipe_coverage=coverage,static_geometry=static,
        historical_diagnoses=diagnoses,setup_evidence=setup,yaw_geometry=yaw_geometry,
        request_accounting=accounting,request_packets=packets,budget=budget,
        elapsed_s=time.monotonic()-started,maximum_numeric_error=checks.numeric_error,
        limitations=['Historical35 fixed positions: finite development search, not a learned policy or held-out evaluation.',
            'Two actual fixed-base/bucket intersections remain blocked under unchanged geometry.',
            '29 previous recipes replay once; each discovered addition needs three fresh nominal confirmations.',
            '96 numeric proposals per searched cell with at most two distinct finite IK branches; no exhaustive feasibility claim.',
            'Unchanged native64/v3 allows table/rim assistance and excludes forbidden routes/invalid physics.',
            'Held endpoints with sampled setup-path checks; no new training, noise,128 qualification, hardware, deployment or UI publication.'])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('study','source','assets','parent-study','parent-audit','original-study','original-audit','reference','output'):
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
