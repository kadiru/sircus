"""Independent CPU audit of finite recovery on the original 35 grid cells.

Discovery and confirmation are development experiments, not policy inference.
This verifier does not import the producer's planner, scorer or selection rule.
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
from . import fixed35_imitation_verify as fixed
from . import calibrated_benchmark_verify as calibrated
from . import coverage_training_verify as geometry
from . import policy64_verify as models
from . import replay650_verify as replay
from . import table_training_verify as native
from . import table_assisted_score_oracle as oracle
from .benchmark_verify import EvidenceError, contained, digest, read, sha

VERSION='sircus-fixed35-recovery-audit-v1'
STUDY_VERSION='sircus-fixed35-recovery-v1'
PARENT_REPORT_SHA='91640ad07e7b8e396c58d937dcf8a5652a07c3eb5f7ae2ea0bce27f40425a389'
PARENT_AUDIT_SHA='374eb3ff78b302168622c091aca645283f48d1bee33355a9ac4eec3794e116d6'
PARENT_FP='9a5de51b320a36d3e498301c8f3b2c228c11ed999032bc7b1d9ef8777f213f6f'
FAILURE_CELLS=((0,0),(1,5),(1,6),(2,3),(2,6),(3,0),(3,4),(3,5),(3,6),(4,4),(4,6))
BLOCKED_CELLS=((3,0),(3,6))
ROLES=('direct_control','table_control','negative_control')
META=fixed.META
AuditChecks=replay.AuditChecks
captured=native.captured
statistics=native.statistics


def authenticate_parent(directory,audit,checks):
    directory,audit=Path(directory),Path(audit)
    checks.equal(sha(directory/'report.json'),PARENT_REPORT_SHA,'Exact published 35-position parent report')
    checks.equal(sha(audit),PARENT_AUDIT_SHA,'Exact complete independent parent audit')
    report,spec,proof,manifest=map(read,(directory/'report.json',directory/'contract.json',audit,directory/'evidence-manifest.json'))
    checks.equal((report['version'],report['status'],proof['version'],proof['status'],proof['full_checks_verified'],proof['native_checks_verified'],proof['failures']),
        ('sircus-fixed35-imitation-v1','completed','sircus-fixed35-imitation-audit-v1','passed',True,True,[]),'Complete independently audited parent')
    checks.equal((proof['checks_count'],len(proof['checks'])),(40692,40692),'Full parent audit retained')
    checks.require(all(c['passed'] is True for c in proof['checks']),'Every parent check passed')
    unsigned=dict(spec);fp=unsigned.pop('fingerprint')
    checks.equal((digest(unsigned),fp,report['contract_fingerprint'],proof['contract_fingerprint']),(PARENT_FP,)*4,'Frozen parent experiment identity')
    checks.equal(proof['report_sha256'],PARENT_REPORT_SHA,'Parent proof binds report')
    checks.equal(proof['evidence_manifest_sha256'],sha(directory/'evidence-manifest.json'),'Parent proof binds all evidence')
    checks.equal(report['evidence_sha256'],manifest,'Parent report binds full manifest')
    consumed={}
    def bound(name):
        path=contained(directory,name);actual=sha(path)
        checks.equal(actual,manifest[name],'Authenticated parent file/'+name)
        consumed[name]=actual;return path
    bound('contract.json');rows=common.lines(bound('rows.jsonl'))
    checks.equal(len(rows),88,'All original parent requests retained')
    return dict(directory=directory,spec=spec,report=report,rows=rows,bound=bound,consumed=consumed,
        report_sha256=PARENT_REPORT_SHA,audit_sha256=PARENT_AUDIT_SHA,manifest_sha256=sha(directory/'evidence-manifest.json'))


def partition_parent(rows,checks):
    """Literal current-policy grid and failure cells, not a producer-selected subset."""
    current=[r for r in rows if r.get('phase')=='test' and r.get('method')=='current_bc']
    refit=[r for r in rows if r.get('phase')=='test' and r.get('method')=='fixed35_bc']
    grid=fixed.fixed_grid()
    for label,group in (('current',current),('refit',refit)):
        checks.equal(len(group),35,label+'/all35 exact canonical outcomes')
        for r,g in zip(group,grid):
            checks.equal((r['context_id'],r['bucket_xy_m']),(g['id'],g['bucket_xy_m']),label+'/original grid without jitter/'+g['id'])
    preserved=[r for r in current if captured(r)];failures=[r for r in current if not captured(r)]
    checks.equal((len(preserved),len(failures)),(24,11),'Original 24 captures and11 failures')
    checks.equal([r['context_id'] for r in failures],[f'fixed35-dev-grid-{y:02d}-{x:02d}' for y,x in FAILURE_CELLS],'Exact11 failed grid cells')
    blocked=[r for r in failures if r.get('invalid')=='fixed_base_clearance_rejected']
    searchable=[r for r in failures if r not in blocked]
    checks.equal([r['context_id'] for r in blocked],[f'fixed35-dev-grid-{y:02d}-{x:02d}' for y,x in BLOCKED_CELLS],'Only two immutable base-clearance blockers skipped')
    checks.equal(Counter(r.get('invalid') for r in searchable),Counter({None:8,'finite_ik_unresolved':1}),'Eight flight misses and one finite-IK failure remain searchable')
    return dict(current=current,refit=refit,preserved=preserved,failures=failures,blocked=blocked,cases=searchable)


def bind_parent_rows(parent,rows,checks):
    by_id={r['id']:r for r in parent['rows']}
    for row in rows:
        label=row['id'];checks.equal(row,by_id[label],label+'/unaltered historical record')
        namespace=row['adapter_directory'];checks.require(namespace in ('native-yaw','native-controls'),label+'/known historical native namespace')
        for field in ('context_resolution','connection'):
            checks.equal(sha(parent['bound'](namespace+'/'+row[field+'_file'])),row[field+'_sha256'],label+'/authenticated '+field)
        if row.get('native_release'):
            checks.equal(sha(parent['bound'](namespace+'/'+row['native_cache_file'])),row['native_cache_sha256'],label+'/authenticated native trajectory')


def audit_blockers(parent,blocked,checks):
    """Unmoving base/bucket clearance cannot be repaired by arm posture changes."""
    results=[]
    for row,arm in zip(blocked,('Arm10','Arm14')):
        resolution=read(parent['bound'](row['adapter_directory']+'/'+row['context_resolution_file']))
        home=resolution['home_geometry'];blockers=home['fixed_base_blockers']
        checks.equal((home['hard_reject'],row['native_release']),(True,False),row['id']+'/fixed-layout rejection precedes release')
        checks.require(bool(blockers),row['id']+'/explicit fixed base obstruction evidence')
        for b in blockers:
            checks.require(b['pair'][0].startswith(arm+'/link_base/') and b['pair'][1].startswith('bucket/'),row['id']+'/only immovable base/bucket pair')
            checks.require(0<=b['lower_bound_m']<.005,row['id']+'/conservative clearance below unchanged5mm')
        results.append(dict(context_id=row['context_id'],bucket_xy_m=row['bucket_xy_m'],arm=arm,
            fixed_base_pairs=len(blockers),minimum_clearance_lower_bound_m=min(b['lower_bound_m'] for b in blockers)))
    return results


def protocol_checks(value,checks):
    literal=dict(substeps=64,methods=['preserved_current','recovery_search','recovery_repeat'],grid_positions=35,
        parent_captures=24,development_failures=11,fixed_base_skips=2,search_layouts=9,proposals_per_layout=80,
        search_requests=720,repeat_per_selected=3,candidates_per_layout=3,repeat_requests_max=81,
        preserved_requests=24,control_requests=18,maximum_requests=843,noise=False,bucket_perturbations=False,
        training=False,actor_steps=0,critic_steps=0,worker_limit_s=7200,search_limit_s=5400,queue_limit_s=900,
        global_limit_s=9900,audit_reserve_s=1800,storage_reserve_bytes=8*1024**3,
        rotation='Rz(atan2(bucket_y-board_y,bucket_x-board_x)) @ diag(1,-1,-1)',
        bounds=dict(low=[.30,-.50,.12,8.,.05],high=[1.04,.35,.60,60.,.65]),
        evaluation_scope='historical fixed-grid recipe development; not held out and not a new policy score')
    for key,wanted in literal.items():checks.equal(value[key],wanted,'Literal bounded recovery protocol/'+key)


def request(source,identity,phase,method,values=None,**extra):
    values=copy.deepcopy(source['values'] if values is None else values)
    return dict({k:copy.deepcopy(source[k]) for k in META},id=identity,phase=phase,role='development',method=method,
        context_id=source['context_id'],bucket_xy_m=copy.deepcopy(source['bucket_xy_m']),values=values,command_values=copy.deepcopy(values),
        parent_row_id=source['id'],parent_row_sha256=digest(source),substeps=64,original_role=source.get('role'),
        development_reclassified=True,selection_or_learning=False,perturbation_mixed_units=[0.]*5,**extra)


def independent_proposals(parent,refit,preserved):
    """Reconstruct80 literal predeclared actions; no trajectory-dependent tuning."""
    base=list(map(float,parent['values']));raw=[]
    def add(values,kind,ids=None):raw.append((list(map(float,values)),kind,ids or [parent['id']]))
    add(base,'original_failed_current')
    other=next(r for r in refit if r['context_id']==parent['context_id'])
    add(other['values'],'exact_historical_refit_capture' if captured(other) else 'exact_historical_refit_action',[other['id']])
    neighbors=sorted(preserved,key=lambda r:(float(np.linalg.norm(np.asarray(r['bucket_xy_m'])-parent['bucket_xy_m'])),r['id']))[:4]
    for row in neighbors:add(row['values'],'neighbor_success_action_at_fixed_bucket',[row['id']])
    for index,steps,label in ((4,[.02,.04,.06,.10,.16,.24],'gap_axis'),(3,[2.,4.,6.,10.,14.,18.],'tilt_axis'),(2,[.02,.05,.08,.12,.18],'height_axis')):
        for step in steps:
            for sign in (-1,1):
                v=base.copy();v[index]+=sign*step;add(v,label)
    for tilt in [-8.,-4.,4.,8.]:
        for gap in [-.08,-.04,.04,.08]:
            v=base.copy();v[3]+=tilt;v[4]+=gap;add(v,'coupled_tilt_gap')
    shifts=[(x,y) for x in [-.06,0.,.06] for y in [-.06,0.,.06] if x!=0 or y!=0]
    shifts.extend([(-.12,0.),(.12,0.),(0.,-.12),(0.,.12)])
    for x,y in shifts:
        v=base.copy();v[0]+=x;v[1]+=y;add(v,'bounce_plane_xy')
    for height in [.20,.32,.44]:
        for tilt in [20.,32.]:
            for gap in [.16,.32]:add(base[:2]+[height,tilt,gap],'coarse_height_tilt_gap')
    result=[]
    for i,(v,kind,ids) in enumerate(raw):
        bounded=[min(hi,max(lo,x)) for x,lo,hi in zip(v,fixed.ACTION_LOW,fixed.ACTION_HIGH)]
        result.append(dict(values=bounded,raw_values=v,clipped=v!=bounded,kind=kind,parent_source_ids=ids,proposal_index=i))
    if len(result)!=80:raise EvidenceError('Independent80-proposal specification is incomplete')
    return result


def independent_plan(data):
    pools={r['context_id']:independent_proposals(r,data['refit'],data['preserved']) for r in data['cases']}
    plan=[]
    for i in range(80):
        for r in data['cases']:
            p=pools[r['context_id']][i]
            plan.append(request(r,f'fixed35-recovery-search-p{i:02d}-{r["context_id"]}','search','recovery_search',p['values'],
                proposal_index=i,derivation=p,exact_parent_action=p['values']==r['values']))
    return plan


def independent_preserved(data):
    return [request(r,'fixed35-recovery-preserved-'+r['context_id'],'preserved','preserved_current',exact_parent_action=True) for r in data['preserved']]


def independent_repeat(discovery,group):
    if not captured(discovery) or group not in range(3):raise EvidenceError('Only successful discovery can select fresh confirmation')
    result=[]
    for i in range(3):
        r=request(discovery,f'fixed35-recovery-repeat-c{group}-r{i}-{discovery["context_id"]}','repeat','recovery_repeat',
            confirmation_index=group,repeat=i,selected_from_id=discovery['id'],selected_from_sha256=digest(discovery),
            proposal_index=discovery['proposal_index'],selected_connection_sha256=discovery['connection_sha256'],
            selected_q10=discovery['q10'],selected_q14=discovery['q14'])
        r.update(parent_row_id=discovery['parent_row_id'],parent_row_sha256=discovery['parent_row_sha256']);result.append(r)
    return result


def audit_plan(study,data,checks):
    packet=read(study/'search-plan.json');planned=packet['rows'];expected=independent_plan(data)
    checks.equal(packet,dict(rows=expected,frozen_before_outcomes=True,maximum_requests=720),'All720 proposals independently reconstructed before any outcome')
    checks.equal(read(study/'preserved-plan.json'),dict(rows=independent_preserved(data),frozen_before_outcomes=True),'All24 preserved requests frozen before outcomes')
    checks.equal(len({r['id'] for r in planned}),720,'No hidden omitted or duplicate slot identifiers')
    for r in planned:
        checks.equal(r['bucket_xy_m'],next(c['bucket_xy_m'] for c in data['cases'] if c['context_id']==r['context_id']),r['id']+'/bucket remains exact canonical location')
        checks.equal(r['values'],r['command_values'],r['id']+'/explicit bounded numeric command, no hidden substitution')
    return planned


def independent_groups(rows,context):
    groups=[]
    for i in range(3):
        group=[r for r in rows if r.get('phase')=='repeat' and r['context_id']==context and r['confirmation_index']==i]
        if not group:continue
        first=group[0];n=len(group);success=sum(captured(r) for r in group)
        groups.append(dict(confirmation_index=i,selected_from_id=first['selected_from_id'],selected_from_sha256=first['selected_from_sha256'],
            values=first['values'],bucket_xy_m=first['bucket_xy_m'],requests=n,captures=success,record_ids=[r['id'] for r in group],
            status='confirmed' if n==3 and success==3 else 'failed' if n==3 else 'incomplete'))
    return groups


def independent_cases(rows,data,*,finished=False,budget_truncated=False):
    result=[]
    for parent in data['failures']:
        search=[r for r in rows if r['phase']=='search' and r['context_id']==parent['context_id']]
        groups=independent_groups(rows,parent['context_id']);confirmed=next((g for g in groups if g['status']=='confirmed'),None)
        status=('fixed_base_blocked' if parent in data['blocked'] else 'confirmed' if confirmed else 'confirmation_limit' if len(groups)==3 else
                'search_exhausted' if len(search)==80 else 'budget_truncated' if finished and budget_truncated else 'in_progress')
        recipe=next((r for r in search if confirmed and r['id']==confirmed['selected_from_id']),None)
        result.append(dict(context_id=parent['context_id'],bucket_xy_m=parent['bucket_xy_m'],parent_row_id=parent['id'],parent_row_sha256=digest(parent),
            status=status,search_requests=len(search),search_captures=sum(captured(r) for r in search),search_counts=statistics(search),
            confirmation_groups=groups,confirmed_recipe=copy.deepcopy(recipe)))
    return result


def audit_native(study,rows,data,spec,calibration,assets,checks,parent):
    setup={};count=0
    for namespace,key,parent_namespace in (('native-yaw','prior','native-yaw'),('native-controls','old_prior','control-parent/native')):
        members=[r for r in rows if r['adapter_directory']==namespace];path=study/namespace
        checks.require(bool(members),namespace+'/requested cohort preserved')
        setup[namespace]=calibrated.check_setup_evidence(geometry._NativeNamespace(path),members,data[key],data['home'],checks)
        admitted=[r for r in members if r.get('native_release')]
        if admitted:
            models.audit_native_options(path,spec['scene'],calibration,assets,checks)
            for r in admitted:
                checks.equal((r['effective_options_file'],r['effective_options_sha256']),('effective-options.json',sha(path/'effective-options.json')),r['id']+'/unchanged native64 options')
            for name in ('effective-options.json','native-compiled-geometry.json'):
                checks.equal(read(path/name),read(parent['bound'](parent_namespace+'/'+name)),namespace+'/physics matches parent/'+name)
        else:checks.require(not list(path.rglob('*.npz')),namespace+'/no fabricated release trace')
        count+=len(native.rescore_rows(path,members,spec['scene'],calibration,spec['identity']['trace_schema'],assets,checks))
    # The mature independent FK/clearance audit keys fixed-yaw behavior by phase.
    # Remap only that display categorization, never the native ledger or its hashes.
    classified=[dict(r,phase='test') if r['adapter_directory']=='native-yaw' else r for r in rows]
    geometry_result=fixed.audit_geometry(study,classified,spec['scene'],assets,checks)
    return setup,count,geometry_result


def audit_exact_branch(row,parent,checks,label):
    for key in (*META,'values','bucket_xy_m','q10','q14','connection_file','connection_sha256','native_release'):
        checks.equal(row.get(key),parent.get(key),label+'/preserved exact admitted branch/'+key)
    if row.get('native_release'):
        for field in ('q10','q14'):
            checks.equal(row['native_initial_'+field],np.asarray(parent[field],np.float32).astype(float).tolist(),label+'/observed original native reset/'+field)


def audit_preserved(rows,data,checks):
    group=[r for r in rows if r['phase']=='preserved'];planned=independent_preserved(data)
    checks.equal(len(group),24,'Exactly one fresh replay of each of24 successful parent recipes')
    for row,packet,parent in zip(group,planned,data['preserved']):
        for key,value in packet.items():checks.equal(row[key],value,row['id']+'/frozen preserved request/'+key)
        audit_exact_branch(row,parent,checks,row['id'])
        checks.require(row['native_cache_sha256']!=parent['native_cache_sha256'],row['id']+'/new native packet, not copied historical outcome')
    return statistics(group)


def audit_controls(rows,data,checks):
    gates={}
    for phase,key in (('controls_pre','pre'),('controls_post','post')):
        group=[r for r in rows if r['phase']==phase];checks.equal(len(group),9,phase+'/nine prescribed controls')
        for i,(source,original,role) in enumerate(zip(data['controls'],data['control_replays'],ROLES)):
            for repeat in range(3):
                row=group[3*i+repeat];label=row['id']
                checks.equal(label,f'fixed35-recovery-{phase}-{role}-r{repeat}',label+'/literal control identity')
                checks.equal((row['role'],row['method'],row['repeat']),(role,'control',repeat),label+'/matched control metadata')
                checks.equal((row['control_source_id'],row['control_source_sha256']),(source['id'],digest(source)),label+'/original control provenance')
                checks.equal((row['source_record_id'],row['source_record_sha256'],row['variant']),(original['id'],digest(original),'control_exact'),label+'/original control joint branch')
                checks.equal((row['values'],row['bucket_xy_m']),(source['values'],source['bucket_xy_m']),label+'/unchanged control action and position')
                audit_exact_branch(row,original,checks,label)
        gates[key]=native.control_gate(group);checks.equal(gates[key]['status'],'passed',phase+'/unchanged direct/table/negative controls pass')
    return gates


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


def audit_inputs(study,parent,spec,checks):
    data=partition_parent(parent['rows'],checks)
    bind_parent_rows(parent,data['current']+data['refit'],checks)
    for name,key in (('parent-current','current'),('parent-refit','refit'),('preserved','preserved'),('cases','cases'),('blocked','blocked')):
        checks.equal(read(study/f'bootstrap/{name}.json'),data[key],'Exact historical cohort/'+name)
    checks.equal((spec['grid'],read(study/'grid.json')),(fixed.fixed_grid(),)*2,'Only original35 canonical bucket positions')
    bindings=read(study/'source-bindings.json');checks.equal(spec['source_bindings'],bindings,'Frozen parent bindings unchanged')
    declared=bindings['parent']
    expected=dict(report_sha256=PARENT_REPORT_SHA,audit_sha256=PARENT_AUDIT_SHA,contract_fingerprint=PARENT_FP,
        evidence_manifest_sha256=parent['manifest_sha256'],checks_count=40692)
    for key,value in expected.items():checks.equal(declared[key],value,'Exact source provenance/'+key)
    for name,value in declared['files_sha256'].items():checks.equal(sha(parent['bound'](name)),value,'Every declared parent file independently authenticated/'+name)
    checks.equal(bindings['current_row_sha256'],{r['id']:digest(r) for r in data['current']},'All35 original model outcomes bound')
    for name,key in (('prior.json','prior'),('old-prior.json','old_prior'),('controls.json','controls'),
        ('control-replays.json','control_replays'),('calibration.json','calibration'),('setup-home.json','home')):
        value=read(parent['bound'](name));checks.equal(read(study/name),value,'Original immutable runtime input/'+name);data[key]=value
    checks.equal((len(data['prior']['records']),len(data['old_prior']['records'])),(19,17),'Unchanged geometry anchors and original control prior')
    checks.equal((spec['prior_sha256'],spec['old_prior_sha256']),(digest(data['prior']),digest(data['old_prior'])),'Both geometry priors fingerprinted')
    checks.equal(read(study/'verification.json'),read(parent['bound']('verification.json')),'Original calibrated native verification')
    for r in data['control_replays']:
        for field in ('context_resolution','connection'):
            relative='control-parent/native/'+r[field+'_file'];checks.equal(sha(parent['bound'](relative)),r[field+'_sha256'],r['id']+'/authentic original control '+field)
            checks.equal(sha(contained(study,relative)),r[field+'_sha256'],r['id']+'/unaltered copied control proof')
    artifacts=parent['report']['artifacts']
    checks.equal(spec['parent_artifacts'],artifacts,'All original trained artifacts preserved by identity')
    for artifact in artifacts.values():checks.equal(sha(parent['bound'](artifact['path'])),artifact['sha256'],'Original checkpoint bytes preserved/'+artifact['path'])
    checks.require(not list(study.rglob('*.pt')),'Recovery fits or replaces no checkpoint')
    for name,key in (('search-plan.json','search_plan_sha256'),('preserved-plan.json','preserved_plan_sha256'),('rule.json','rule_sha256')):
        checks.equal(sha(study/name),spec[key],'Frozen plan or rule/'+name)
    return data,audit_blockers(parent,data['blocked'],checks)


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
    actual=read(study/'diagnoses.json');checks.equal(len(actual),11,'All11 original failures explained')
    for got,wanted in zip(actual,expected):
        for key,value in wanted.items():checks.equal(got[key],value,wanted['context_id']+'/independent historical diagnosis/'+key)
        if wanted['native_release']:
            checks.equal(got['contact_force_threshold_N'],.01,wanted['context_id']+'/scorer contact threshold')
            checks.require('scoring termination' in got['contact_scope'].lower() and 'full' in got['contact_scope'].lower(),wanted['context_id']+'/late contact timing scope disclosed')
    return actual


def audit_schedule(study,rows,report,data,plan,checks):
    """Reconstruct every executed round and adaptive stop from preceding outcomes."""
    boundaries=common.lines(study/'phase-boundaries.jsonl');raw=(study/'rows.jsonl').read_bytes().splitlines(keepends=True)
    timings=common.lines(study/'timing.jsonl');bi=0;ti=0;position=0;previous_time=0.
    def prefix(n):return hashlib.sha256(b''.join(raw[:n])).hexdigest() if n else None
    def boundary(phase,**extra):
        nonlocal bi,previous_time
        checks.require(bi<len(boundaries),phase+'/required boundary exists');b=boundaries[bi];bi+=1
        checks.equal((b['phase'],b['rows_count'],b['rows_sha256']),(phase,position,prefix(position)),phase+'/original immutable ledger prefix')
        checks.require(previous_time<=b['acquired_elapsed_s']<7200,phase+'/ordered bounded boundary time');previous_time=b['acquired_elapsed_s']
        for key,value in extra.items():checks.equal(b[key],value,phase+'/boundary '+key)
        return b
    def batch(phase,expected):
        nonlocal position,ti,previous_time
        count=len(expected);group=rows[position:position+count]
        checks.equal(len(group),count,phase+'/entire requested batch is present')
        checks.require(ti<len(timings),phase+'/timing exists');timing=timings[ti];ti+=1
        checks.equal(timing['phase'],phase,phase+'/execution timing phase')
        checks.require(previous_time<=timing['acquired_start_s']<=timing['acquired_end_s']<7200,phase+'/batch follows its freeze within deadline')
        for row,request_packet in zip(group,expected):
            for key,value in request_packet.items():checks.equal(row.get(key),value,row['id']+'/exact frozen request/'+key)
        previous_time=timing['acquired_end_s'];position+=count;return group,timing
    boundary('inputs-frozen')
    batch('controls_pre',[dict(phase='controls_pre')]*9);boundary('controls_pre-complete')
    batch('preserved',independent_preserved(data));boundary('preserved-complete')
    pools={r['context_id']:[p for p in plan if p['context_id']==r['context_id']] for r in data['cases']}
    stopped={};groups={key:0 for key in pools};round_count=0;repeat_files=[]
    for index in range(80):
        active=[c for c in pools if c not in stopped]
        if not active or rows[position]['phase']=='controls_post':break
        b=boundary('search-round-frozen',proposal_index=index,contexts=active)
        checks.require(b['acquired_elapsed_s']<5400,'No new search round beyond90-minute search limit')
        expected=[dict(pools[c][index],search_plan_sha256=sha(study/'search-plan.json')) for c in active]
        discoveries,_=batch('search',expected);selected=[r for r in discoveries if captured(r)]
        for row in discoveries:
            if row.get('exact_parent_action'):audit_exact_branch(row,next(r for r in data['cases'] if r['context_id']==row['context_id']),checks,row['id'])
        if selected:
            name=f'repeat-plans/round-{index:02d}.json';wanted=[r for d in selected for r in independent_repeat(d,groups[d['context_id']])]
            envelope=dict(proposal_index=index,rows=wanted,selected=[dict(id=r['id'],sha256=digest(r)) for r in selected],
                rows_count_before=position,rows_sha256_before=prefix(position),frozen_before_repeat_outcomes=True)
            checks.equal(read(study/name),envelope,name+'/every captured discovery selects three exact fresh repeats')
            boundary('repeat-plan-frozen',proposal_index=index,path=name,sha256=sha(study/name));repeat_files.append(name)
            observed,_=batch('repeat',[dict(r,repeat_plan_file=name,repeat_plan_sha256=sha(study/name)) for r in wanted])
            for discovery in selected:
                context=discovery['context_id'];group=[r for r in observed if r['selected_from_id']==discovery['id']]
                checks.equal([r['repeat'] for r in group],[0,1,2],context+'/three fresh complete confirmation attempts')
                for row in group:
                    for key in (*META,'values','bucket_xy_m','q10','q14','connection_file','connection_sha256'):
                        checks.equal(row[key],discovery[key],row['id']+'/unchanged discovery branch/'+key)
                    checks.require(row['native_cache_sha256']!=discovery['native_cache_sha256'],row['id']+'/fresh confirmation trajectory')
                groups[context]+=1
                if all(captured(r) for r in group):stopped[context]='confirmed'
                elif groups[context]==3:stopped[context]='confirmation_limit'
        boundary('search-round-complete',proposal_index=index,stopped=stopped);round_count=index+1
    truncated=report['budget_truncated'];checks.require(type(truncated) is bool,'Explicit boolean budget truncation')
    if truncated:
        checks.require(round_count<80 and len(stopped)<9,'Truncated search has unfinished candidate capacity')
        reason='search_time_limit'
    else:
        checks.require(round_count==80 or len(stopped)==9,'No unexplained early omission of active search cases')
        reason='all_searchable_cases_confirmed' if len(stopped)==9 and set(stopped.values())=={'confirmed'} else 'finite_candidate_or_confirmation_caps'
    checks.equal(report['completion_reason'],reason,'Honest finite-stop reason')
    finish=boundary('search-complete',completion_reason=reason,budget_truncated=truncated)
    if truncated:checks.require(finish['acquired_elapsed_s']>=5400,'Budget stop follows actual90-minute cutoff')
    batch('controls_post',[dict(phase='controls_post')]*9);boundary('controls_post-complete')
    checks.equal((position,bi,ti),(len(rows),len(boundaries),len(timings)),'No omitted or extra native batches or boundaries')
    checks.equal(sorted(str(p.relative_to(study)) for p in (study/'repeat-plans').glob('*.json')),repeat_files,'Every saved confirmation plan accounted')
    checks.require(len(rows)<=843 and len({r['id'] for r in rows})==len(rows),'Unique requests within843 global cap')
    return dict(total_requests=len(rows),preserved_requests=24,control_requests=18,search_rounds=round_count,
        search_requests=sum(r['phase']=='search' for r in rows),repeat_requests=sum(r['phase']=='repeat' for r in rows),
        maximum_requests=843,budget_truncated=truncated,stopped=stopped)


def audit_summary(study,rows,report,spec,data,preserved,gates,checks):
    cases=independent_cases(rows,data,finished=True,budget_truncated=report['budget_truncated'])
    checks.equal(report['cases'],cases,'Every original failure and fresh confirmation independently summarized')
    confirmed=[r for r in cases if r['confirmed_recipe'] is not None];kept=preserved['captures']
    preservation=dict(requests=24,captures=kept,expected=24,all_replayed=True,all_recaptured=kept==24)
    coverage=dict(original_historical_captures=24,new_confirmed_layouts=len(confirmed),fresh_preserved_captures=kept,
        fresh_total=kept+len(confirmed),positions=35,not_a_policy_score=True)
    checks.equal(report['preservation'],preservation,'Fresh preserved outcomes honestly counted')
    checks.equal(report['demonstrated_recipe_coverage'],coverage,'Recipe coverage is explicitly not policy evaluation')
    checks.equal(report['recovered_layouts'],len(confirmed),'Confirmed layout count excludes mere discovery captures')
    checks.equal(report['counts'],statistics(rows),'All attempts including controls and rejections counted')
    phases=('controls_pre','preserved','search','repeat','controls_post')
    checks.equal(report['stage_counts'],{p:statistics([r for r in rows if r['phase']==p]) for p in phases},'Unmixed per-stage denominators')
    checks.equal(report['control_gates'],gates,'Pre and post control results independently checked')
    checks.equal((report['expected_requests'],report['maximum_requests'],report['actual_requests']),(843,843,len(rows)),'843 is cap, not fabricated executed denominator')
    checks.equal((report['training'],report['posthoc_development'],report['parent_captures'],report['parent_positions'],report['preserved_recipes']),
        (False,True,24,35,24),'No training or rewritten historical policy score')
    checks.equal(report['artifacts'],spec['parent_artifacts'],'Only original policy checkpoint identities reported')
    checks.equal(report['evaluation_scope'],spec['protocol']['evaluation_scope'],'Development coverage scope unchanged')
    expected_gate=dict(status='passed',substeps=64,controls_passed=True,required_replays_complete=True,
        search_cap_completed=not report['budget_truncated'],no_policy_promotion=True)
    checks.equal(report['gate'],expected_gate,'Completion and nonpromotion gate honest')
    expected_recipes=dict(preserved_parent=[dict(id=r['id'],sha256=digest(r),values=r['values'],q10=r['q10'],q14=r['q14'],bucket_xy_m=r['bucket_xy_m']) for r in data['preserved']],
        new_confirmed=[dict(r['confirmed_recipe'],confirmation_groups=r['confirmation_groups']) for r in confirmed],policy_unchanged=True,preservation=preservation)
    checks.equal(read(study/'recipes.json'),expected_recipes,'All historical recipes intact and only three-repeat confirmations added')
    return cases,coverage


def audit(study,source,assets,*,parent_study,parent_audit,reference):
    started=time.monotonic();checks=AuditChecks();study,source,assets=map(Path,(study,source,assets))
    manifest=replay.check_manifest(study,checks);spec=read(study/'contract.json');report=read(study/'report.json')
    unsigned=dict(spec);fp=unsigned.pop('fingerprint')
    checks.equal((digest(unsigned),report['contract_fingerprint']),(fp,fp),'Entire experiment fingerprint unchanged')
    checks.equal((spec['version'],report['version'],report['status'],report['phase']),(STUDY_VERSION,STUDY_VERSION,'completed','completed'),'Completed bounded recovery experiment')
    checks.equal(report['evidence_sha256'],manifest,'Report binds every saved artifact')
    checks.equal(report['protocol'],spec['protocol'],'Frozen protocol unchanged');protocol_checks(spec['protocol'],checks)
    checks.equal(spec['release_behavior'],spec['protocol']['rotation'],'Bucket-facing drop behavior fixed')
    checks.equal(report['scope'],spec['scope'],'No rewritten experiment scope')
    replay.check_source_files(source,spec['source_sha256'],checks,'Frozen executed source')
    for module in (Path(__file__),Path(oracle.__file__),Path(fixed.__file__)):
        checks.equal(sha(module),spec['source_sha256']['sircus/mjlab/'+module.name],'Actual independent audit module frozen/'+module.name)
    rule=read(study/'rule.json');checks.equal(rule,read(source/'sircus/mjlab/fixed35_recovery_rule.json'),'Checked-in prespecified recovery rule unchanged')
    checks.equal((rule['version'],rule['protocol']),(STUDY_VERSION,spec['protocol']),'Rule binds literal protocol')
    parent=authenticate_parent(parent_study,parent_audit,checks);data,blockers=audit_inputs(study,parent,spec,checks)
    models.audit_runtime_reference(source,reference,spec,parent['spec'],assets,checks)
    diagnoses=audit_diagnoses(study,parent,data,checks);plan=audit_plan(study,data,checks)
    calibration=replay.check_calibration(study,spec['identity']['calibration'],checks)
    rows=replay.read_records(study/'rows.jsonl');native.check_rows(rows,checks)
    setup,native_count,geometry_result=audit_native(study,rows,data,spec,calibration,assets,checks,parent)
    preserved=audit_preserved(rows,data,checks);gates=audit_controls(rows,data,checks)
    accounting=audit_schedule(study,rows,report,data,plan,checks)
    packets=geometry.audit_request_packets(study,rows,checks);budget=audit_budget(study,report,checks)
    cases,coverage=audit_summary(study,rows,report,spec,data,preserved,gates,checks)
    return dict(version=VERSION,status='passed',full_checks_verified=True,native_checks_verified=True,
        checks_count=checks.count,checks=checks.entries,failures=[],contract_fingerprint=fp,
        report_sha256=sha(study/'report.json'),evidence_manifest_sha256=sha(study/'evidence-manifest.json'),
        auditor_sha256=sha(Path(__file__)),producer_source_sha256=spec['source_sha256'],
        consumed_source_files_sha256=parent['consumed'],requested_trials=len(rows),native_releases=native_count,
        cases=cases,control_gates=gates,preservation=preserved,demonstrated_recipe_coverage=coverage,
        fixed_base_blockers=blockers,historical_diagnoses=diagnoses,setup_evidence=setup,yaw_geometry=geometry_result,
        request_accounting=accounting,request_packets=packets,budget=budget,elapsed_s=time.monotonic()-started,
        maximum_numeric_error=checks.numeric_error,limitations=[
            'Original35 historical grid; finite posthoc recipe discovery, not a new frozen-policy score or held-out generalization.',
            'Two positions fail conservative fixed-base clearance under unchanged geometry; finite IK failure alone does not establish impossibility.',
            '24 exact parent branches replayed once; new recipes require three fresh nominal repeats of the same admitted branch.',
            'Bounded80 candidates per searchable case, with explicit numeric clipping and round-robin early stopping; no exhaustive coverage claim.',
            'Unchanged native64/v3 includes table and rim assistance but rejects robot/floor/unattributed routes and invalid physics.',
            'Held articulated endpoints; no jitter, noise,128 qualification, learned transit, new model, hardware or policy promotion.'])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('study','source','assets','parent-study','parent-audit','reference','output'):parser.add_argument('--'+name,type=Path,required=True)
    a=parser.parse_args()
    if a.output.exists() or a.output.resolve().is_relative_to(a.study.resolve()):raise ValueError('Fresh independent proof outside immutable study required')
    try:
        kwargs=vars(a).copy();kwargs.pop('output');result=audit(**kwargs)
    except Exception as error:
        result=dict(version=VERSION,status='failed',failures=[type(error).__name__+': '+str(error)])
        a.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');raise
    a.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:result[k] for k in ('status','checks_count','requested_trials','native_releases','demonstrated_recipe_coverage')}))


if __name__=='__main__':main()
