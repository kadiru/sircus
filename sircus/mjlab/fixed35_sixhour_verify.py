"""Independent audit of the bounded six-hour fixed-grid extension.

The frozen broad producer/auditor are reused as evidence and geometry primitives;
this module never edits their source or artifacts, runs IK, or advances physics.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import time

import numpy as np

from . import fixed35_broad_verify as broad
from . import fixed35_recovery_verify as recovery
from . import fixed35_six_verify as six
from . import fixed35_bouncer_verify as history
from . import benchmark_verify as common
from . import replay650_verify as replay
from . import table_training_verify as native
from .benchmark_verify import EvidenceError,contained,digest,read,sha

VERSION='sircus-fixed35-sixhour-audit-v1'
STUDY_VERSION='sircus-fixed35-sixhour-v1'
AuditChecks=replay.AuditChecks


def authenticate_broad(directory,audit,checks,*,expected_binding=None):
    """Accept only complete independently audited parent evidence, not progress."""
    directory,audit=Path(directory),Path(audit)
    spec=read(directory/'contract.json');report=read(directory/'report.json')
    proof=read(audit);manifest=read(directory/'evidence-manifest.json')
    unsigned=dict(spec);fp=unsigned.pop('fingerprint')
    checks.equal((digest(unsigned),report['contract_fingerprint'],proof['contract_fingerprint']),(fp,fp,fp),'Broad parent full contract identity')
    checks.equal((spec['version'],report['version'],proof['version']),('sircus-fixed35-broad1000-v1','sircus-fixed35-broad1000-v1','sircus-fixed35-broad1000-audit-v1'),'Known native64 broad parent versions')
    checks.equal((report['status'],report['phase']),('completed','completed'),'Fully completed broad parent before extension')
    checks.equal(report['fresh_search_releases'],broad.QUOTAS,'All1000 native parent discovery quotas complete')
    checks.equal(report['gate']['search_quota_completed'],True,'Parent quota gate passed')
    checks.equal((proof['status'],proof['full_checks_verified'],proof['native_checks_verified'],proof['failures']),('passed',True,True,[]),'Full independent broad audit required')
    checks.require(type(proof['checks_count']) is int and proof['checks_count']>0,'Actual positive independent checks')
    checks.equal(proof['checks_count'],len(proof['checks']),'All independent checks retained')
    checks.require(all(c.get('passed') is True for c in proof['checks']),'Every parent audit check passed')
    checks.equal(proof['report_sha256'],sha(directory/'report.json'),'Parent audit binds exact report')
    checks.equal(proof['evidence_manifest_sha256'],sha(directory/'evidence-manifest.json'),'Parent audit binds complete saved evidence')
    checks.equal(report['evidence_sha256'],manifest,'Parent report binds full evidence manifest')
    checks.equal(proof['producer_source_sha256'],spec['source_sha256'],'Audited parent producer source identity')
    checks.equal((report['training'],report['historical_policy_captures'],report['parent_positions']),(False,24,35),'Parent remains recipe development, not trained policy')
    checks.equal(report['gate']['controls_passed'],True,'Parent post-controls completed')
    checks.equal(report['gate']['preservation_passed'],True,'Parent original30 preserved')
    binding=dict(report_sha256=sha(directory/'report.json'),audit_sha256=sha(audit),contract_fingerprint=fp,
        evidence_manifest_sha256=sha(directory/'evidence-manifest.json'),checks_count=proof['checks_count'])
    if expected_binding is not None:
        for key,value in binding.items():checks.equal(expected_binding[key],value,'Frozen parent source binding/'+key)
    consumed={}
    def bound(name):
        path=contained(directory,name);actual=sha(path)
        checks.equal(actual,manifest[name],'Authenticated broad parent artifact/'+name);consumed[name]=actual;return path
    bound('contract.json');rows=common.lines(bound('rows.jsonl'))
    checks.equal(len(rows),report['actual_requests'],'Complete parent request denominator')
    checks.equal(len({r['id'] for r in rows}),len(rows),'Unique parent native record IDs')
    checks.equal(len(rows),proof['requested_trials'],'Parent audit examined every reported row')
    return dict(directory=directory,spec=spec,report=report,proof=proof,rows=rows,bound=bound,consumed=consumed,**binding)


def confirmed_broad_recipes(parent,checks):
    """Recover exact captured source rows with three matching fresh repeats."""
    by_id={r['id']:r for r in parent['rows']};result=[]
    for case in parent['report']['cases']:
        for source in case['confirmed_recipes']:
            checks.equal(source,by_id[source['id']],source['id']+'/unaltered parent discovery')
            checks.equal((source['phase'],source['context_id']),('search',case['context_id']),source['id']+'/actual discovery identity')
            checks.require(native.captured(source),source['id']+'/independently audited parent capture')
            repeats=sorted((r for r in parent['rows'] if r.get('phase')=='repeat' and r.get('selected_from_id')==source['id']),key=lambda r:r['repeat'])
            checks.equal([r['repeat'] for r in repeats],[0,1,2],source['id']+'/three fresh confirmations')
            for row in repeats:
                checks.require(native.captured(row),row['id']+'/parent confirmation captured')
                checks.equal(row['selected_from_sha256'],digest(source),row['id']+'/exact selected source binding')
                for key in ('values','bucket_xy_m','q10','q14','connection_sha256'):
                    checks.equal(row[key],source[key],row['id']+'/matching confirmed branch/'+key)
                checks.require(row['native_cache_sha256']!=source['native_cache_sha256'],row['id']+'/fresh confirmation archive')
            for key in ('context_resolution','connection'):
                path=parent['bound']('native-yaw/'+source[key+'_file'])
                checks.equal(sha(path),source[key+'_sha256'],source['id']+'/authenticated prepared source/'+key)
            path=parent['bound']('native-yaw/'+source['broad_pool_file'])
            checks.equal(sha(path),source['broad_pool_sha256'],source['id']+'/authenticated broad candidate pool')
            result.append(source)
    checks.equal(len({r['id'] for r in result}),len(result),'Confirmed recipe identities never duplicated')
    checks.equal(len({(r['context_id'],tuple(r['values']),tuple(r['q10']),tuple(r['q14'])) for r in result}),len(result),'Distinct exact confirmed commands')
    return result


SIXHOUR_VERSION='sircus-fixed35-sixhour-bouncer-branches-v1'
SIXHOUR_RULE=dict(broad.BRANCH_RULE,version=SIXHOUR_VERSION,maximum_admitted_branches=8)
BROAD_VERSION=SIXHOUR_VERSION
BRANCH_RULE=SIXHOUR_RULE
BROAD_FIELDS=broad.BROAD_FIELDS
fixed=broad.fixed
previous=history
check_tree=broad.check_tree


def audit_endpoint(endpoint,anchor,scene,assets,chains,gate,checks,label):
    """Validate the new rule, reuse only unchanged finite endpoint mechanics.

    Endpoint generation ends before the admitted-path branch cap is applied.
    The isolated metadata view normalizes only the two independently checked
    rule fields for the frozen legacy endpoint checker. Evidence is untouched.
    """
    checks.equal(endpoint['branch_rule'],SIXHOUR_RULE,label+'/new eight-branch literal rule')
    changes={k:(broad.BRANCH_RULE[k],SIXHOUR_RULE[k]) for k in SIXHOUR_RULE if broad.BRANCH_RULE[k]!=SIXHOUR_RULE[k]}
    checks.equal(changes,dict(version=(broad.BROAD_VERSION,SIXHOUR_VERSION),maximum_admitted_branches=(2,8)),label+'/only post-endpoint branch budget changed')
    view=copy.deepcopy(endpoint);view['branch_rule']=copy.deepcopy(broad.BRANCH_RULE)
    return broad.audit_endpoint(view,anchor,scene,assets,chains,gate,checks,label)


def audit_pool(namespace,record,proposal,scene,assets,chains,gate,home,prior,checks):
    from .calibrated_admission import verify_resolution,verify_path
    label=proposal['id'];path=contained(namespace,record['broad_pool_file']);checks.equal(sha(path),record['broad_pool_sha256'],label+'/immutable preparation pool')
    pool=read(path);checks.equal(pool['branch_rule'],BRANCH_RULE,label+'/frozen native branch rule')
    checks.equal((pool['version'],pool['selection_uses_flight_labels']),(BROAD_VERSION,False),label+'/geometry-only preparation')
    checks.equal((pool['values'],pool['bucket_xy_m']),(proposal['values'],proposal['bucket_xy_m']),label+'/no action substitution')
    anchorpath=contained(namespace,pool['resolution_file']);checks.equal(sha(anchorpath),pool['resolution_sha256'],label+'/authenticated context resolution')
    anchor=read(anchorpath);anchor.update(file=pool['resolution_file'],sha256=pool['resolution_sha256'])
    checks.require(verify_resolution(anchor,prior['records'],home),label+'/geometry-only anchor')
    checks.equal(pool['home_sha256'],home['sha256'],label+'/unchanged canonical home')
    from .calibrated_admission import CONNECTION_POLICY_VERSION
    _,behavior_key=fixed.audit_behavior(dict(pool,connection_policy=CONNECTION_POLICY_VERSION,action_substituted=False),scene,checks,label)
    key=digest(dict(version=BROAD_VERSION,branch_rule=BRANCH_RULE,behavior_key=behavior_key,resolution_sha256=pool['resolution_sha256']))
    checks.equal((pool['pool_key'],record['broad_pool_key'],record['broad_pool_file']),(key,key,'broad-pools/'+key+'.json'),label+'/complete pool cache key')
    endpoint=pool['endpoint_attempts']
    if anchor['home_passed']:
        checks.require(endpoint is not None,label+'/eligible layout has complete finite IK evidence')
        audit_endpoint(dict(endpoint,_values=pool['values'],_bucket=pool['bucket_xy_m']),anchor,scene,assets,chains,gate,checks,label)
    else:
        checks.equal((endpoint,pool['path_attempts'],pool['admitted']),(None,[],[]),label+'/blocked context cannot prepare a flight')
        checks.equal(pool['reason'],'fixed_base_clearance_rejected' if anchor.get('hard_reject') else 'canonical_home_unresolved_in_layout',label+'/context rejection retained')
    admitted=pool['admitted'];paths=pool['path_attempts'];checks.require(len(admitted)<=8 and len(paths)<=32,label+'/finite branch/path caps')
    checks.equal([a['slot'] for a in admitted],list(range(len(admitted))),label+'/contiguous admitted slots')
    checks.equal([a['candidate_index'] for a in admitted],sorted({a['candidate_index'] for a in admitted}),label+'/first valid unique candidates')
    checks.equal([a['path_attempt_index'] for a in admitted],[i for i,p in enumerate(paths) if p.get('passed')],label+'/all passing paths retained')
    for i,branch in enumerate(admitted):
        for other in admitted[:i]:checks.require(np.max(previous.wrapped_difference(branch['q14'],other['q14']))>=.05,label+'/every slot changes bouncer, not only dropper wrist')
    for slot,branch in enumerate(admitted):
        candidate=endpoint['candidates'][branch['candidate_index']];proof=paths[branch['path_attempt_index']]
        for field in ('q10','q14','endpoint'):checks.equal(branch[field],candidate[field],label+'/admitted exact endpoint/'+field)
        checks.require(verify_path(proof,np.r_[home['q10'],home['q14']],np.r_[branch['q10'],branch['q14']]),label+'/each admitted slot strict forward/reverse sampled path')
    if endpoint is not None:checks.equal(pool['reason'],endpoint['reason'] if not endpoint['passed'] else ('finite_requested_setup_path_unresolved' if not admitted else None),label+'/honest preparation reason')
    return pool,anchor




def expected_connection(pool,anchor,home,pool_file,pool_sha,slot):
    from .calibrated_admission import VERSION as connection_version,CONNECTION_POLICY_VERSION
    selected=pool['admitted'][slot]
    key=digest(dict(version=BROAD_VERSION,pool_key=pool['pool_key'],broad_pool_sha256=pool_sha,broad_branch_slot=slot))
    return dict(version=connection_version,bucket_xy_m=pool['bucket_xy_m'],values=pool['values'],resolution_file=anchor['file'],resolution_sha256=anchor['sha256'],
        home_sha256=home['sha256'],passed=True,geometry_eligible=True,endpoint_eligible=True,reason=None,q10=selected['q10'],q14=selected['q14'],
        proof=pool['path_attempts'][selected['path_attempt_index']],endpoint=selected['endpoint'],endpoint_attempts=pool['endpoint_attempts'],path_attempts=pool['path_attempts'],timing=pool['timing'],
        action_substituted=False,connection_policy=CONNECTION_POLICY_VERSION,anchor_values=anchor['anchor_values'],anchor_derivation_sha256=digest(anchor['anchor_derivation']),
        release_behavior=pool['release_behavior'],behavior_key=pool['behavior_key'],broad_rule_version=BROAD_VERSION,broad_key=key,broad_branch_slot=slot,
        broad_pool_file=pool_file,broad_pool_sha256=pool_sha,broad_pool_key=pool['pool_key'])




def audit_preparations(study,plan,data,spec,assets,checks):
    from .learning_ik import read_chain
    from .corrected_geometry import StrictEndpointGate
    namespace=study/'native-yaw';chains=tuple(read_chain(assets/name) for name in ('dropper.urdf','board_arm.urdf'))
    gate=StrictEndpointGate(spec['scene'],assets,.005);proposals={r['id']:r for r in plan['rows']};prepared={};seen=set()
    records=common.lines(study/'preparations.jsonl')
    for record in records:
        label=record['proposal_id'];checks.require(label not in seen,label+'/one preparation per frozen numeric proposal');seen.add(label)
        proposal=proposals[label];checks.equal(record['proposal_sha256'],digest(proposal),label+'/exact pre-flight frozen proposal')
        for key in ('context_id','values','bucket_xy_m'):checks.equal(record[key],proposal[key],label+'/unchanged proposal/'+key)
        pool,anchor=audit_pool(namespace,record,proposal,spec['scene'],assets,chains,gate,data['home'],data['prior'],checks)
        eligible=[]
        for slot in range(len(pool['admitted'])):
            connection=expected_connection(pool,anchor,data['home'],record['broad_pool_file'],record['broad_pool_sha256'],slot)
            filename='fixed-yaw-connections/'+connection['broad_key']+'.json';path=contained(namespace,filename)
            checks.equal(read(path),connection,label+'/full independently reconstructed immutable connection')
            packet=copy.deepcopy(proposal);packet.update(id=label+'-b'+str(slot),proposal_id=label,command_values=proposal['values'],native_prepared=True,
                prepared_connection_file=filename,prepared_connection_sha256=sha(path),selected_q10=connection['q10'],selected_q14=connection['q14'],
                source_hint=anchor['source_hint'],effective_source_hint=anchor['effective_source_hint'],source_teacher_id=anchor['source_hint']['id'],
                context_resolution_file=anchor['file'],context_resolution_sha256=anchor['sha256'],context_home_sha256=data['home']['sha256'],
                context_layout_eligible=anchor['layout_eligible'],context_anchor_admitted=anchor['anchor_admitted'],context_hard_reject=anchor['hard_reject'])
            packet.update({key:connection[key] for key in BROAD_FIELDS});eligible.append(packet);prepared[packet['id']]=packet
        checks.equal(record['admitted'],len(eligible),label+'/all admitted alternatives counted')
        checks.equal(record['admitted_request_ids'],[r['id'] for r in eligible],label+'/all prepared alternatives retained')
        checks.equal(record['admitted_request_sha256'],[digest(r) for r in eligible],label+'/full prepared request content-addressed')
        checks.equal((record['native_releases'],record['preparation_only']),(0,True),label+'/preparation is not a simulated drop')
        checks.equal(record['reason'],None if eligible else pool['reason'],label+'/exact preparation decline')
        checks.equal(record['timing'],pool['timing'],label+'/preparation timing provenance')
    return prepared,dict(numeric_preparations=len(records),admitted_branches=len(prepared),declines=sum(not r['admitted'] for r in records))




def audit_preserved_geometry(study,rows,data,scene,assets,checks):
    """Authenticate exact historical commands before reusing observed-FK checks.

    Older29+1 proof formats retain their original checker. V1 broad commands are
    checked as an isolated geometry-only replay view; no record or outcome is
    changed, and source connection SHA is authenticated separately.
    """
    sources={r['id']:r for r in data['preserved']}
    old=[];views=[];prepared={}
    for row in rows:
        if row['phase']=='preserved':
            parent=sources[row['parent_row_id']]
            history.audit_exact_branch(row,parent,checks,row['id'])
            if parent.get('broad_rule_version')==broad.BROAD_VERSION:
                connection=read(contained(study/'native-yaw',row['connection_file']))
                for field in broad.BROAD_FIELDS:checks.equal(row[field],parent[field],row['id']+'/exact historical broad field/'+field)
                checks.equal(sha(contained(study/'native-yaw',row['connection_file'])),parent['connection_sha256'],row['id']+'/original historical proof bytes')
                packet=dict(values=row['values'],bucket_xy_m=row['bucket_xy_m'],selected_q10=row['q10'],selected_q14=row['q14'],
                    prepared_connection_file=row['connection_file'],prepared_connection_sha256=row['connection_sha256'],proposal_id=row['id'])
                packet.update({k:connection[k] for k in broad.BROAD_FIELDS})
                view=copy.deepcopy(row);view.update(phase='repeat',prepared_request=packet)
                views.append(view);prepared[row['id']+'-b'+str(row['broad_branch_slot'])]=packet
                continue
        old.append(row)
    if old:six.audit_geometry(study,old,scene,assets,checks)
    if views:broad.audit_flights(study,views,prepared,scene,assets,checks)
    return dict(legacy_rows=len(old),historical_broad_preserved=len(views),geometry_only_transient_adapter=True)


def audit_budget(study,report,spec,parent,checks):
    budgetpath=study.with_name(study.name+'-budget.json')
    queuepath=study.with_name(study.name+'-queue.json')
    launchpath=study.with_name(study.name+'-launch.json')
    budget,queue,launch=read(budgetpath),read(queuepath),read(launchpath)
    pipeline=read(study.with_suffix('.pipeline.json'))
    checks.equal((budget['limit_s'],budget['audit_reserve_s'],budget['acquired_total_limit_s']),(18000,3600,21600),'Five-hour worker plus one-hour audit in six acquired hours')
    checks.equal(budget['gpu_uuid'],'GPU-ab1950b0-01dd-0fab-407f-2d17936b27f4','Only authorized4090')
    checks.equal(launch['queue_receipt_sha256'],sha(queuepath),'Exclusive launch binds exclusive initial queue')
    checks.equal(queue['pipeline_source_sha256'],spec['source_sha256']['sircus/mjlab/fixed35_sixhour_pipeline.py'],'Actual queued supervisor source frozen')
    checks.equal(launch['acquired_total_limit_s'],21600,'Exactly six additional acquired hours authorized')
    checks.require(0<queue['queue_limit_s']<=18000,'Separate finite parent/GPU queue')
    checks.numeric(queue['queue_deadline_unix_s']-queue['started_unix_s'],queue['queue_limit_s'],'Original queue deadline',.01)
    checks.numeric(queue['global_deadline_unix_s'],queue['queue_deadline_unix_s']+21600,'Outer bound includes one acquired interval',.01)
    checks.equal(pipeline['global_deadline_unix_s'],queue['global_deadline_unix_s'],'Supervisor never extends original deadline')
    checks.equal(budget['global_deadline_unix_s'],queue['global_deadline_unix_s'],'Worker shares immutable outer deadline')
    checks.require(queue['started_unix_s']<=launch['requested_unix_s']<=budget['acquired_unix_s']<=queue['queue_deadline_unix_s']+.1,'Parent launch then acquired lease within queue')
    deadline=min(budget['acquired_unix_s']+21600,queue['global_deadline_unix_s'])
    checks.numeric(budget['acquired_deadline_unix_s'],deadline,'One nonrenewable acquired deadline',.01)
    checks.numeric(budget['effective_worker_limit_s'],min(18000,deadline-budget['acquired_unix_s']-3600),'Independent audit reserve cannot be consumed by worker',.1)
    checks.require(60<budget['effective_worker_limit_s']<=18000,'Finite actual worker duration')
    checks.require(report['elapsed_s']<budget['effective_worker_limit_s'],'Worker completed before acquired deadline')
    for key in ('effective_worker_limit_s','global_deadline_unix_s','acquired_deadline_unix_s'):
        checks.equal(report['deadlines'][key],budget[key],'Reported original deadline/'+key)
    checks.equal((report['deadlines']['worker_limit_s'],report['deadlines']['search_limit_s']),(18000,16200),'Search cap leaves controls and finalization time')
    binding=launch['parent']
    for key in ('report_sha256','audit_sha256','contract_fingerprint','evidence_manifest_sha256'):
        checks.equal(binding[key],parent[key],'Launch bound completed parent/'+key)
    checks.equal(binding['contract_sha256'],sha(parent['directory']/'contract.json'),'Parent contract pinned before queue')
    checks.equal(binding['producer_source_manifest_sha256'],digest(parent['spec']['source_sha256']),'Parent producer manifest identity')
    checks.equal(binding['completed_search_releases'],broad.QUOTAS,'Parent completed all1000 before launch')
    parentpipeline=parent['directory'].with_suffix('.pipeline.json')
    checks.equal(sha(parentpipeline),binding['pipeline_sha256'],'Completed parent supervisor immutable')
    state=read(parentpipeline)
    checks.equal((state['status'],state['report_sha256'],state['audit_sha256']),('completed',parent['report_sha256'],parent['audit_sha256']),'Actual completed parent supervisor bound report and audit')
    checks.require(state['finished_unix_s']<=launch['requested_unix_s'],'No native extension launched before parent independent audit finished')
    return dict(worker_receipt_sha256=sha(budgetpath),queue_receipt_sha256=sha(queuepath),launch_receipt_sha256=sha(launchpath),
        effective_worker_limit_s=budget['effective_worker_limit_s'],acquired_total_limit_s=21600,
        acquired_deadline_unix_s=deadline,parent_pipeline_sha256=sha(parentpipeline))


CASE_IDS=broad.CASE_IDS
TARGET=CASE_IDS[1]
SEED=2026111201
LOW,HIGH=broad.LOW,broad.HIGH
FAMILIES=('axis','local','table_route','lateral')


def protocol_checks(protocol,checks):
    literal=dict(version=STUDY_VERSION,substeps=64,primary_context=TARGET,maximum_numeric_proposals=6000,
        search_native_max=1200,maximum_branches_per_numeric=8,repeat_per_selected=3,selected_successes_per_cell=3,
        repeat_max=27,controls=18,preserved='original30 plus every independently confirmed broad-parent command',
        acquired_total_limit_s=21600,worker_limit_s=18000,search_limit_s=16200,primary_limit_s=10800,
        queue_limit_s=18000,audit_reserve_s=3600,global_limit_s=39600,storage_reserve_bytes=8*1024**3,
        bounds=dict(low=LOW.tolist(),high=HIGH.tolist()),seed=SEED,families=list(FAMILIES),
        endpoint_and_path_clearance_m=.005,training=False,noise=False,policy_updated=False,ui_changes=False,
        scoring='sircus-table-assisted-native-capture-v3')
    for key,value in literal.items():checks.equal(protocol[key],value,'Independent six-hour protocol/'+key)


def independent_plan(sources,confirmed,scene,assets,*,stage='primary',limit=6000,interval_fn=None,reach_fn=None):
    """Independent literal multiscale generator, never calls the producer plan."""
    import math
    from collections import Counter
    from .learning_ik import read_chain
    if not 0<limit<=6000 or stage not in ('primary','window'):raise EvidenceError('Finite known independent stage')
    if interval_fn is None or reach_fn is None:
        chains=tuple(read_chain(Path(assets)/name) for name in ('dropper.urdf','board_arm.urdf'))
        if interval_fn is None:interval_fn=lambda assets,scene,bucket,xyz,tilt,**kw:broad.independent_gap_interval(chains,scene,bucket,xyz,tilt,**kw)
        if reach_fn is None:reach_fn=lambda assets,scene,bucket,values:broad.independent_reach_record(chains,assets,scene,bucket,values)
    seeds=([r for r in sources if r['context_id']==TARGET]+list(confirmed)) if stage=='primary' else list(confirmed)
    contexts=(TARGET,) if stage=='primary' else tuple(c for c in CASE_IDS if any(r['context_id']==c for r in seeds))
    if not contexts:raise EvidenceError('No authenticated window source')
    anchors={c:sorted((s for s in sources if s['context_id']==c),key=lambda r:r['id']) for c in contexts}
    choices={c:sorted((seeds if stage=='primary' else [s for s in seeds if s['context_id']==c]),key=lambda r:r['id']) for c in contexts}
    if not all(anchors.values()) or not all(choices.values()):raise EvidenceError('Every target requires original lineage and explicit seed')
    rng=np.random.default_rng(SEED+(stage=='window'));seen=set();result=[];reasons=Counter();families=Counter();seed_counts=Counter()
    for raw in range(limit*40):
        context=contexts[raw%len(contexts)];local=raw//len(contexts);pool=choices[context]
        seed=pool[local%len(pool)];cycle=local//len(pool);source=anchors[context][local%len(anchors[context])]
        family=FAMILIES[cycle%4] if stage=='primary' else ('axis' if cycle%2==0 else 'local')
        value=np.array(seed['values'],float);bucket=np.array(source['bucket_xy_m'],float);u=rng.random(6)
        scale=(.01,.03,.07,.14)[(cycle//4)%4]
        if family=='axis':
            axis=(cycle//4)%5;magnitude=((1.,3.,7.) if axis==3 else (.01,.03,.07))[(cycle//40)%3]
            value[axis]+=(1 if (cycle//20)%2==0 else -1)*magnitude
        elif family=='local':value+=np.array([scale,scale,scale,scale*100,scale])*(2*u[:5]-1)
        elif family=='table_route':
            value[:2]+=.08*(2*u[:2]-1);value[2]=.25+.35*u[2];value[3]=25.+35.*u[3];value[4]+=.12*(2*u[4]-1)
        else:
            angle=math.atan2(value[1]-bucket[1],value[0]-bucket[0])+(2*u[0]-1)*math.pi/2
            radius=.25+.50*u[1];value[:2]=bucket+radius*np.array([math.cos(angle),math.sin(angle)])
            value[2]+=.14*(2*u[2]-1);value[3]+=14*(2*u[3]-1);value[4]+=.18*(2*u[4]-1)
        if (value<LOW).any() or (value>HIGH).any():reasons['numeric_bounds']+=1;continue
        if np.linalg.norm(bucket-value[:2])<=.15:reasons['board_bucket_separation']+=1;continue
        interval=interval_fn(assets,scene,bucket,value[:3],float(value[3]),gap_bounds=(float(LOW[4]),float(HIGH[4])))
        if interval is None:reasons['wrist_reach_no_gap']+=1;continue
        lo,hi=map(float,interval)
        if not LOW[4]-1e-12<=lo<=hi<=HIGH[4]+1e-12:raise EvidenceError('Conservative gap interval cannot expand declared bounds')
        if not lo<=value[4]<=hi:reasons['gap_outside_reach_interval']+=1;continue
        reach=reach_fn(assets,scene,bucket,value)
        if not reach['passed']:reasons['wrist_reach']+=1;continue
        identity=(context,*value.tolist())
        if identity in seen:reasons['duplicate_numeric']+=1;continue
        seen.add(identity);index=len(result)
        row=recovery.request(source,f'fixed35-sixhour-{stage}-p{index:04d}-{context}','search','sixhour_search',value.tolist(),
            exact_parent_action=False,proposal_index=index,raw_proposal_index=raw,proposal_family=family,proposal_stage=stage,
            seed_source_id=seed['id'],seed_source_sha256=digest(seed),seed_context_id=seed['context_id'],seed_values=copy.deepcopy(seed['values']),
            conditional_gap_interval_m=[lo,hi],reach_filter=copy.deepcopy(reach))
        result.append(row);families[family]+=1;seed_counts[seed['id']]+=1
        if len(result)==limit:break
    return dict(version=STUDY_VERSION,stage=stage,seed=SEED+(stage=='window'),frozen_before_native_outcomes=True,
        limit=limit,raw_considered=raw+1,rows=result,geometric_declines=dict(reasons),families=dict(families),seed_counts=dict(seed_counts),
        seed_sha256={s['id']:digest(s) for s in seeds},source_sha256={s['id']:digest(s) for c in contexts for s in anchors[c]})



def audit_plan(actual,expected,checks,label):
    """Exact commands/order/provenance; roundoff only in geometric diagnostics."""
    checks.equal({k:v for k,v in actual.items() if k!='rows'},{k:v for k,v in expected.items() if k!='rows'},label+'/exact plan metadata')
    checks.equal(len(actual['rows']),len(expected['rows']),label+'/exact proposal count')
    diagnostic={'reach_filter','conditional_gap_interval_m'}
    for got,wanted in zip(actual['rows'],expected['rows']):
        checks.equal({k:v for k,v in got.items() if k not in diagnostic},
                     {k:v for k,v in wanted.items() if k not in diagnostic},label+'/'+wanted['id']+'/exact command and source identity')
        for key in diagnostic:check_tree(got[key],wanted[key],checks,label+'/'+wanted['id']+'/'+key,1e-12)


def independent_preserved(data):
    return [recovery.request(r,f'fixed35-sixhour-preserved-{i:02d}-'+digest(r)[:12],'preserved','preserved_recipe',exact_parent_action=True)
            for i,r in enumerate(data['preserved'])]


def independent_repeat(discovery,group):
    rows=broad.independent_repeat(discovery,group)
    for row in rows:row.update(id=row['id'].replace('fixed35-broad-','fixed35-sixhour-',1),method='sixhour_repeat')
    return rows


def command_key(row):
    return digest(dict(context_id=row['context_id'],values=row['values'],
        q10=row.get('selected_q10',row.get('q10')),q14=row.get('selected_q14',row.get('q14'))))


def audit_inputs(study,original,older,parent,broad_parent,spec,checks):
    old=broad.audit_inputs(broad_parent['directory'],original,older,parent,broad_parent['spec'],checks)
    additions=confirmed_broad_recipes(broad_parent,checks)
    data=dict(old,preserved=old['preserved']+additions,broad_sources=additions,broad_rows=broad_parent['rows'],old_preserved_sources=old['preserved'])
    bindings=dict(historical=broad_parent['spec']['source_bindings'],
        broad={k:broad_parent[k] for k in ('report_sha256','audit_sha256','contract_fingerprint','evidence_manifest_sha256','checks_count')},
        confirmed_row_sha256={r['id']:digest(r) for r in additions})
    checks.equal(spec['source_bindings'],bindings,'Completed parent plus original history fully authenticated')
    checks.equal(read(study/'source-bindings.json'),bindings,'Saved source binding matches frozen contract')
    for name,key in (('parent-current','current'),('parent-refit','refit'),('preserved','preserved'),('cases','cases'),('blocked','blocked'),('numeric-sources','numeric_sources'),('broad-confirmed','broad_sources')):
        checks.equal(read(study/f'bootstrap/{name}.json'),data[key],'All exact authenticated bootstrap rows/'+name)
    commands=[dict(id=r['id'],command_key=command_key(r)) for r in broad_parent['rows'] if r.get('native_release')]
    checks.equal(read(study/'bootstrap/broad-parent-commands.json'),commands,'Complete already-flown parent command exclusion set')
    checks.equal(spec['grid'],fixed.fixed_grid(),'Exact35 canonical development grid')
    checks.equal(read(study/'grid.json'),fixed.fixed_grid(),'Saved canonical grid retained')
    for name,key in (('prior.json','prior'),('old-prior.json','old_prior'),('controls.json','controls'),('control-replays.json','control_replays'),('calibration.json','calibration'),('setup-home.json','home')):
        checks.equal(read(study/name),data[key],'Unchanged native runtime input/'+name)
    checks.equal((spec['prior_sha256'],spec['old_prior_sha256']),(digest(data['prior']),digest(data['old_prior'])),'Both original priors bound')
    checks.equal(read(study/'verification.json'),read(parent['bound']('verification.json')),'Original scoring calibration verification')
    checks.equal(spec['parent_artifacts'],original['report']['artifacts'],'Original policy bytes retained; no model fitting')
    checks.require(not list(study.rglob('*.pt')),'No newly fitted checkpoint')
    for name,key in (('search-plan.json','search_plan_sha256'),('preserved-plan.json','preserved_plan_sha256'),('rule.json','rule_sha256')):
        checks.equal(sha(study/name),spec[key],'Frozen pre-outcome plan/'+name)
    sources=old['preserved']+old['numeric_sources']
    receipts=[dict(source_id=r['id'],source_row_sha256=digest(r),resolution_file=r['context_resolution_file'],resolution_sha256=r['context_resolution_sha256'],connection_file=r['connection_file'],connection_sha256=r['connection_sha256']) for r in sources]
    for row in additions:
        receipts.append(dict(source_id=row['id'],source_row_sha256=digest(row),resolution_file=row['context_resolution_file'],resolution_sha256=row['context_resolution_sha256'],
            connection_file=row['connection_file'],connection_sha256=row['connection_sha256'],broad_pool_file=row['broad_pool_file'],
            broad_pool_sha256=row['broad_pool_sha256'],broad_rule_version=broad.BROAD_VERSION,exact_historical_broad_recipe=True))
        for prefix in ('context_resolution','connection','broad_pool'):
            filename=row[prefix+'_file'];expected=row[prefix+'_sha256']
            checks.equal(sha(broad_parent['bound']('native-yaw/'+filename)),expected,row['id']+'/authenticated original broad proof/'+prefix)
            checks.equal(sha(contained(study/'native-yaw',filename)),expected,row['id']+'/unaltered copied broad proof/'+prefix)
    checks.equal(read(study/'native-source-receipts.json'),receipts,'All old42 plus confirmed broad recipe hydration receipts')
    return data


def audit_preserved(rows,data,checks):
    group=[r for r in rows if r['phase']=='preserved'];plan=independent_preserved(data)
    checks.equal(len(group),len(plan),'One fresh replay of every old and newly confirmed recipe')
    for row,packet,parent in zip(group,plan,data['preserved']):
        for key,value in packet.items():checks.equal(row[key],value,row['id']+'/exact frozen preservation/'+key)
        history.audit_exact_branch(row,parent,checks,row['id'])
        checks.require(row['native_cache_sha256']!=parent['native_cache_sha256'],row['id']+'/fresh native archive')
    return native.statistics(group)


def audit_controls(rows,data,checks):
    # Control identity prefix is the only change. Isolate this presentation field;
    # strict joint/action/proof equality and both nine-control gates stay intact.
    view=copy.deepcopy(rows)
    for row in view:
        if row['phase'] in ('controls_pre','controls_post'):
            expected=f"fixed35-sixhour-{row['phase']}-{row['role']}-r{row['repeat']}"
            checks.equal(row['id'],expected,row['id']+'/new explicit control identity')
            row['id']=row['id'].replace('fixed35-sixhour-','fixed35-broad-',1)
    return broad.audit_controls(view,data,checks)


def audit_summary(study,rows,report,data,preserved,gates,cases,checks):
    checks.equal(report['cases'],cases,'Every searched cell and fresh confirmation group reported')
    checks.equal(report['counts'],native.statistics(rows),'All request/native denominators')
    checks.equal(report['actual_requests'],len(rows),'Every actual request counted')
    checks.equal(report['actual_native_releases'],sum(r.get('native_release') is True for r in rows),'Actual native release denominator')
    checks.equal(report['stage_counts'],{p:native.statistics([r for r in rows if r['phase']==p]) for p in ('controls_pre','preserved','search','repeat','controls_post')},'Preservation and controls excluded from discovery claims')
    checks.equal(report['control_gates'],gates,'Pre/post original control gates')
    count=len(data['preserved']);kept=dict(requests=count,captures=preserved['captures'],expected=count,all_recaptured=preserved['captures']==count)
    checks.equal(report['preservation'],kept,'All retained recipe variants replayed');checks.require(kept['all_recaptured'],'All prior successes preserved')
    checks.equal((report['historical_policy_captures'],report['parent_captures'],report['parent_positions'],report['training'],report['posthoc_development']),
        (24,len({r['context_id'] for r in data['preserved']}),35,False,True),'Unique coverage, recipe count, and unchanged policy are distinct')
    checks.equal((report['maximum_native_search'],report['maximum_native_releases']),(1200,1200+27+18+count),'Finite discovery/repeat/preservation/control caps')
    checks.equal(report['gate'],dict(status='passed',controls_passed=True,preservation_passed=True,bounded_search_completed=True,no_policy_promotion=True),'Completed bounded investigation is not coverage or policy success')
    recipes=dict(preserved_parent=[dict(id=r['id'],sha256=digest(r),values=r['values'],q10=r['q10'],q14=r['q14'],bucket_xy_m=r['bucket_xy_m']) for r in data['preserved']],
        new_confirmed=[r for c in cases for r in c['confirmed_recipes']],policy_unchanged=True)
    checks.equal(read(study/'recipes.json'),recipes,'Only authenticated preserved or three-repeat new recipes retained')
    return cases


def audit(study,source,assets,*,parent_study,parent_audit,original_study,original_audit,recovery_study,recovery_audit,reference,broad_study,broad_audit):
    from . import fixed35_sixhour_schedule_verify as schedule
    started=time.monotonic();checks=AuditChecks();study,source,assets=map(Path,(study,source,assets))
    manifest=replay.check_manifest(study,checks);spec=read(study/'contract.json');report=read(study/'report.json')
    unsigned=dict(spec);fp=unsigned.pop('fingerprint')
    checks.equal((digest(unsigned),report['contract_fingerprint']),(fp,fp),'Complete immutable extension identity')
    checks.equal((spec['version'],report['version']),(STUDY_VERSION,STUDY_VERSION),'Separate six-hour study identity')
    checks.equal((report['status'],report['phase']),('completed','completed'),'Finished bounded study, never partial progress')
    checks.equal(report['evidence_sha256'],manifest,'Report binds complete saved evidence')
    checks.equal(report['protocol'],spec['protocol'],'Frozen protocol');protocol_checks(spec['protocol'],checks)
    checks.equal(spec['release_behavior'],'Rz(atan2(bucket_y-board_y,bucket_x-board_x)) @ diag(1,-1,-1)','Fixed bucket-facing drop behavior')
    checks.equal(report['scope'],spec['scope'],'Development scope retained')
    replay.check_source_files(source,spec['source_sha256'],checks,'Frozen extension source')
    for module in (Path(__file__),Path(schedule.__file__),Path(broad.__file__),Path(broad.oracle.__file__)):
        checks.equal(sha(module),spec['source_sha256']['sircus/mjlab/'+module.name],'Actual independent audit source/'+module.name)
    checks.equal(read(study/'rule.json'),dict(version=STUDY_VERSION,protocol=spec['protocol'],native_branch_rule=SIXHOUR_RULE),'Only explicit eight-branch native rule')
    original=recovery.authenticate_parent(original_study,original_audit,checks)
    older=six.authenticate_recovery(recovery_study,recovery_audit,checks)
    parent=history.authenticate_parent(parent_study,parent_audit,checks)
    bparent=authenticate_broad(broad_study,broad_audit,checks,expected_binding=spec['source_bindings']['broad'])
    data=audit_inputs(study,original,older,parent,bparent,spec,checks)
    broad.models.audit_runtime_reference(source,reference,spec,parent['spec'],assets,checks)
    checks.equal(sha(study/'geometry-audit.json'),sha(parent['bound']('geometry-audit.json')),'Two unchanged fixed-base blockers remain excluded')
    static=six.audit_static_geometry(study,spec,assets,checks)
    plan=read(study/'search-plan.json')
    expected=independent_plan(data['numeric_sources'],data['broad_sources'],spec['scene'],assets)
    audit_plan(plan,expected,checks,'Independent frozen primary multiscale plan')
    keptplan=independent_preserved(data)
    checks.equal(read(study/'preserved-plan.json'),dict(rows=keptplan,frozen_before_outcomes=True),'Every authenticated recipe variant has its own exact preservation ID')
    calibration=replay.check_calibration(study,spec['identity']['calibration'],checks)
    rows=replay.read_records(study/'rows.jsonl');native.check_rows(rows,checks)
    checks.require(not any(r.get('context_id') in {b['context_id'] for b in data['blocked']} for r in rows),'Fixed-base blocked cells never flown')
    allplan=copy.deepcopy(plan);window_files=sorted((study/'window-plans').glob('*.json'))
    checks.require(len(window_files)<=1,'At most one explicit posthoc window transition')
    for path in window_files:
        checks.equal(path.name,'plan.json','Unique frozen window plan name')
        window=read(path);count=window['selected_after_rows_count']
        checks.require(type(count) is int and 0<count<=len(rows),'Window selected from actual preceding evidence prefix')
        prior_cases=schedule.independent_cases(rows[:count]);target=next(c for c in prior_cases if c['context_id']==TARGET)
        checks.require(bool(target['confirmed_recipes']),'Window can open only after primary three-repeat capture')
        additions=[r for c in prior_cases for r in c['confirmed_recipes']]
        expected_window=independent_plan(data['numeric_sources'],data['broad_sources']+additions,spec['scene'],assets,stage='window',limit=window['limit'])
        metadata={k:window[k] for k in ('selected_after_rows_count','selected_after_rows_sha256','acquired_elapsed_s','posthoc_development')}
        checks.equal(metadata['posthoc_development'],True,'Outcome-selected windows explicitly labeled development')
        audit_plan(window,dict(expected_window,**metadata),checks,'Independent conditional window numeric plan')
        allplan['rows'].extend(window['rows'])
    prepared,preparation=audit_preparations(study,allplan,data,spec,assets,checks)
    setup,native_count=broad.audit_native(study,rows,data,spec,calibration,assets,checks,parent)
    newrows=[r for r in rows if r['phase'] in ('search','repeat')]
    flights=broad.audit_flights(study,newrows,prepared,spec['scene'],assets,checks)
    preserved_geometry=audit_preserved_geometry(study,[r for r in rows if r['phase'] not in ('search','repeat')],data,spec['scene'],assets,checks)
    kept=audit_preserved(rows,data,checks);gates=audit_controls(rows,data,checks)
    accounting=schedule.audit_schedule(study,rows,report,dict(data,prepared_requests=prepared),plan,keptplan,independent_repeat,checks)
    packets=broad.geometry.audit_request_packets(study,rows,checks)
    budget=audit_budget(study,report,spec,bparent,checks)
    cases=audit_summary(study,rows,report,data,kept,gates,schedule.independent_cases(rows),checks)
    return dict(version=VERSION,status='passed',full_checks_verified=True,native_checks_verified=True,checks_count=checks.count,checks=checks.entries,failures=[],
        contract_fingerprint=fp,report_sha256=sha(study/'report.json'),evidence_manifest_sha256=sha(study/'evidence-manifest.json'),
        auditor_sha256=sha(Path(__file__)),producer_source_sha256=spec['source_sha256'],
        consumed_source_files_sha256=dict(original=original['consumed'],recovery=older['consumed'],six=parent['consumed'],broad=bparent['consumed']),
        parent_broad_binding={k:bparent[k] for k in ('report_sha256','audit_sha256','contract_fingerprint','evidence_manifest_sha256','checks_count')},
        requested_trials=len(rows),native_releases=native_count,cases=cases,control_gates=gates,preservation=kept,
        preserved_geometry=preserved_geometry,static_geometry=static,setup_evidence=setup,preparation=preparation,flights=flights,
        request_accounting=accounting,request_packets=packets,budget=budget,elapsed_s=time.monotonic()-started,maximum_numeric_error=checks.numeric_error,
        limitations=['Known fixed-grid recipe development, not policy training or held-out accuracy.',
            'Up to8 distinct bouncer branches per numeric setting; strict original finite IK/path/5mm gates remain unchanged.',
            'Native64/v3 with held endpoints, no noise or128-step qualification; no hardware validation.',
            'Primary search precedes outcome-selected local windows; windows are explicitly posthoc development.',
            'Six acquired hours includes CPU audit; completed means bounded protocol completed, not every bucket recovered.'])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('study','source','assets','parent-study','parent-audit','original-study','original-audit','recovery-study','recovery-audit','reference','broad-study','broad-audit','output'):
        parser.add_argument('--'+key,type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists() or args.output.resolve().is_relative_to(args.study.resolve()):raise ValueError('Fresh external independent audit output required')
    try:
        kwargs=vars(args).copy();kwargs.pop('output');result=audit(**kwargs)
    except Exception as error:
        args.output.write_text(json.dumps(dict(version=VERSION,status='failed',failures=[type(error).__name__+': '+str(error)]),indent=2,allow_nan=False)+'\n');raise
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:result[k] for k in ('status','checks_count','requested_trials','native_releases')}))


if __name__=='__main__':main()
