"""Independent CPU audit of the reach-conditioned 1000-drop development study.

The auditor reconstructs the frozen proposal generator without importing the
producer selector, proves geometry from pinned URDFs, and scores saved native
traces. It never runs a new IK optimization or physics trajectory.
"""
from __future__ import annotations

import argparse
import copy
from collections import Counter
import json
import math
from pathlib import Path
import time

import numpy as np

from . import fixed35_bouncer_verify as previous
from . import fixed35_six_verify as six
from . import fixed35_recovery_verify as recovery
from . import fixed35_imitation_verify as fixed
from . import benchmark_verify as common
from . import calibrated_benchmark_verify as calibrated
from . import coverage_training_verify as geometry
from . import policy64_verify as models
from . import replay650_verify as replay
from . import table_training_verify as native
from . import table_assisted_score_oracle as oracle
from . import gripper_yaw_verify as yaw
from .benchmark_verify import EvidenceError, contained, digest, read, sha

VERSION = 'sircus-fixed35-broad1000-audit-v1'
STUDY_VERSION = 'sircus-fixed35-broad1000-v1'
CASE_IDS = tuple('fixed35-dev-grid-'+x for x in ('03-04', '03-05', '04-06'))
QUOTAS = dict(zip(CASE_IDS, (334, 333, 333)))
LOW = np.array([.30, -.50, .12, 8., .05])
HIGH = np.array([1.04, .35, .60, 60., .65])
SEED = 2026111101
FAMILIES = ('cartesian', 'lower_farther', 'bucket_polar', 'local_admitted')
AuditChecks = replay.AuditChecks
captured = native.captured
statistics = native.statistics


def chain_reach(chain, target):
    """Necessary wrist reach bound; satisfying it is not an IK certificate."""
    origins, axes, lower, upper = chain
    target = np.asarray(target, float)
    if target.shape != (4, 4) or not np.isfinite(target).all():
        raise EvidenceError('Finite complete flange target required')
    if not np.array_equal(axes[5], [0., 0., 1.]):
        raise EvidenceError('Pinned terminal revolute axis must be local Z')
    # Every upstream rotation preserves each link-translation length. The
    # triangle inequality remains conservative without joint-limit assumptions.
    shoulder = origins[0, :3, 3]
    radius = float(sum(np.linalg.norm(origin[:3, 3]) for origin in origins[1:5]))
    terminal = origins[5, :3, :3].T @ origins[5, :3, 3]
    axial = float(terminal[2])
    transverse = float(np.linalg.norm(terminal[:2]))
    allowance = 1e-4 + 2*abs(axial)*math.sin(4e-4/2)
    center = target[:3, 3] - target[:3, 2]*axial
    distance = float(np.linalg.norm(center-shoulder))
    return dict(shoulder_m=shoulder.tolist(), wrist_m=center.tolist(),
                radius_m=radius, axial_m=axial, transverse_m=transverse,
                tolerance_allowance_m=allowance, distance_m=distance,
                passed=distance <= radius+transverse+allowance)


def independent_reach(chains, scene, bucket, values):
    targets = fixed.expected_targets(scene, values, bucket)
    arms = [chain_reach(chain, targets[key]) for chain, key in zip(
        chains, ('dropper_flange_world', 'board_flange_local'))]
    return dict(passed=all(a['passed'] for a in arms), arms=arms)


def independent_gap_interval(chains, scene, bucket, xyz, tilt, gap_bounds=(.05, .65)):
    """Solve the downward-dropper sphere inequality before choosing a gap."""
    xyz = np.asarray(xyz, float)
    targets = fixed.expected_targets(scene, [*xyz, tilt, 0.], bucket)
    if not chain_reach(chains[1], targets['board_flange_local'])['passed']:
        return None
    info = chain_reach(chains[0], targets['dropper_flange_world'])
    center = np.asarray(info['wrist_m'])-np.asarray(info['shoulder_m'])
    radius = info['radius_m']+info['transverse_m']+info['tolerance_allowance_m']
    radial = radius*radius-float(np.dot(center[:2], center[:2]))
    if radial < 0:
        return None
    vertical = math.sqrt(radial)
    low = max(float(gap_bounds[0]), -vertical-float(center[2]))
    high = min(float(gap_bounds[1]), vertical-float(center[2]))
    return None if high < low else (low, high)


def check_tree(actual, expected, checks, label, tolerance=1e-12):
    """Exact discrete structure; bounded roundoff for independent geometry only."""
    if isinstance(expected, dict):
        checks.equal(set(actual), set(expected), label+'/fields')
        for key, value in expected.items():
            check_tree(actual[key], value, checks, label+'/'+key, tolerance)
    elif isinstance(expected, (list, tuple)):
        checks.equal(len(actual), len(expected), label+'/length')
        for index, (got, wanted) in enumerate(zip(actual, expected)):
            check_tree(got, wanted, checks, label+'/'+str(index), tolerance)
    elif isinstance(expected, bool):
        checks.require(type(actual) is bool and actual == expected, label)
    elif isinstance(expected, (float, np.floating)):
        checks.numeric(actual, float(expected), label, tolerance)
    else:
        checks.equal(actual, expected, label)


def independent_reach_record(chains, assets, scene, bucket, values):
    targets = fixed.expected_targets(scene, values, bucket)
    arms = {}
    for role, file, chain, key in zip(('dropper', 'bouncer'),
            ('dropper.urdf', 'board_arm.urdf'), chains,
            ('dropper_flange_world', 'board_flange_local')):
        v = chain_reach(chain, targets[key])
        radius = v['radius_m']+v['transverse_m']+v['tolerance_allowance_m']
        arms[role] = dict(urdf_file=file, urdf_sha256=sha(Path(assets)/file),
            shoulder_m=v['shoulder_m'], upstream_length_upper_m=v['radius_m'],
            terminal_axial_m=v['axial_m'], terminal_transverse_bound_m=v['transverse_m'],
            accepted_pose_allowance_m=v['tolerance_allowance_m'], outer_radius_m=radius,
            nominal_wrist_m=v['wrist_m'], distance_from_shoulder_m=v['distance_m'],
            outside_by_m=v['distance_m']-radius, passed=v['passed'])
    return dict(version='sircus-urdf-conservative-full-pose-wrist-bound-v1',
        passed=all(v['passed'] for v in arms.values()), arms=arms,
        scope='Necessary full-pose outer bound only; passing does not certify IK, limits, collisions or a path')


def independent_plan(sources, scene, assets, *, cap_fn=None, reach_fn=None,
                     per_cell=4000, raw_limit=40000):
    """Literal independent PCG64 proposal reconstruction, no selector import."""
    from .learning_ik import read_chain
    if cap_fn is None or reach_fn is None:
        chains = tuple(read_chain(Path(assets)/name) for name in ('dropper.urdf', 'board_arm.urdf'))
    if cap_fn is None:
        def cap_fn(assets, scene, bucket, xyz, tilt, gap_bounds):
            interval = independent_gap_interval(chains, scene, bucket, xyz, tilt, gap_bounds)
            return None if interval is None else interval[1]
    if reach_fn is None:
        def reach_fn(assets, scene, bucket, values):
            return independent_reach_record(chains, assets, scene, bucket, values)
    rows, summaries = [], []
    for number, context in enumerate(CASE_IDS):
        hints = sorted((r for r in sources if r['context_id'] == context), key=lambda r:r['id'])
        if not hints:
            raise EvidenceError('Every fixed context requires authenticated geometry hints')
        bucket = np.asarray(hints[0]['bucket_xy_m'], float)
        if any(r['bucket_xy_m'] != bucket.tolist() for r in hints):
            raise EvidenceError('Historical hint changed fixed bucket position')
        rng = np.random.default_rng(SEED+number)
        seen, accepted, reasons, families = set(), [], Counter(), Counter()
        for raw_index in range(raw_limit):
            family = FAMILIES[raw_index % 4]
            u = rng.random(6)
            source = hints[raw_index % len(hints)]
            if family == 'cartesian':
                values = LOW+(HIGH-LOW)*u[:5]
            elif family == 'lower_farther':
                values = np.array([.55+.49*u[0], -.5+.85*u[1], .12+.24*u[2], 8.+52.*u[3], .05])
            elif family == 'bucket_polar':
                angle = -math.pi+2*math.pi*u[0]
                radius = .16+.94*u[1]
                xy = bucket+radius*np.array([math.cos(angle), math.sin(angle)])
                values = np.r_[xy, .12+.48*u[2], 8.+52.*u[3], .05]
            else:
                values = np.asarray(source['values'], float)+np.array([.22,.22,.20,22.,0.])*(2*u[:5]-1)
            values[4] = .05
            if np.any(values < LOW) or np.any(values > HIGH):
                reasons['numeric_bounds'] += 1
                continue
            if np.linalg.norm(bucket-values[:2]) <= .15:
                reasons['board_bucket_separation'] += 1
                continue
            cap = cap_fn(assets, scene, bucket, values[:3], float(values[3]), gap_bounds=(.05,.65))
            if cap is None or not np.isfinite(cap) or cap < .05:
                reasons['wrist_reach_no_gap'] += 1
                continue
            if cap > .65+1e-12:
                raise EvidenceError('Independent gap bound expanded declared action range')
            fraction = u[4]**2 if (raw_index//4) % 2 else u[4]
            values[4] = float(.05+fraction*(cap-.05))
            reach = reach_fn(assets, scene, bucket, values)
            if not reach['passed']:
                reasons['wrist_reach'] += 1
                continue
            identity = tuple(values.tolist())
            if identity in seen:
                reasons['duplicate_numeric'] += 1
                continue
            seen.add(identity)
            index = len(accepted)
            accepted.append(recovery.request(source, f'fixed35-broad-p{index:04d}-{context}',
                'search', 'broad_search', values=values.tolist(), exact_parent_action=False,
                proposal_index=index, raw_proposal_index=raw_index, proposal_family=family,
                conditional_gap_cap_m=float(cap), reach_filter=copy.deepcopy(reach)))
            families[family] += 1
            if len(accepted) == per_cell:
                break
        rows.extend(accepted)
        summaries.append(dict(context_id=context, accepted_numeric=len(accepted),
            raw_considered=raw_index+1, geometric_declines=dict(reasons),
            families=dict(families), proposal_limit_reached=len(accepted)==per_cell))
    return dict(version=STUDY_VERSION, seed=SEED, frozen_before_native_outcomes=True,
        requested_per_cell=per_cell, raw_limit_per_cell=raw_limit, rows=rows, cells=summaries)


def protocol_checks(p, checks):
    literal = dict(version=STUDY_VERSION, substeps=64,
        search_native_quotas=QUOTAS, search_native_target=1000,
        maximum_native_releases=1075, proposals_per_cell=4000,
        raw_proposals_per_cell=40000, proposal_seed=SEED,
        families=list(FAMILIES), maximum_branches_per_numeric=2,
        repeat_per_selected=3, selected_successes_per_cell=3, repeat_max=27,
        preserved=30, controls=18, worker_limit_s=43200,
        search_limit_s=36000, queue_limit_s=900, audit_reserve_s=7200,
        global_limit_s=51300, storage_reserve_bytes=8*1024**3,
        bounds=dict(low=LOW.tolist(), high=HIGH.tolist()),
        endpoint_and_path_clearance_m=.005, training=False, noise=False,
        policy_updated=False, ui_changes=False, restart_in_place=False,
        scoring='sircus-table-assisted-native-capture-v3')
    for key, value in literal.items():
        checks.equal(p[key], value, 'Literal broad-development protocol/'+key)


def independent_preserved(data):
    return [recovery.request(r, 'fixed35-broad-preserved-'+r['context_id'],
            'preserved', 'preserved_recipe', exact_parent_action=True)
            for r in data['preserved']]


def independent_repeat(discovery, group):
    if not captured(discovery) or type(group) is not int or not 0 <= group < 3:
        raise EvidenceError('Only captured discoveries select at most3 confirmation groups')
    result = []
    for number in range(3):
        row = copy.deepcopy(discovery['prepared_request'])
        row.update(id=f'fixed35-broad-repeat-g{group}-r{number}-{discovery["context_id"]}',
            phase='repeat', method='broad_repeat', confirmation_index=group,
            repeat=number, selected_from_id=discovery['id'],
            selected_from_sha256=digest(discovery),
            selected_connection_sha256=discovery['connection_sha256'],
            selected_q10=copy.deepcopy(discovery['q10']),
            selected_q14=copy.deepcopy(discovery['q14']))
        result.append(row)
    return result


from . import fixed35_broad_schedule_verify as schedule
from .fixed35_bouncer_audit_v3 import compare_certificates
ROLES=previous.ROLES
audit_exact_branch=previous.audit_exact_branch

def audit_inputs(study,original,older,parent,spec,checks):
    data=previous.cohort(original,older,parent,checks)
    for source,rows in ((original,data['current']+data['refit']),(older,[r for r in data['old_preserved'] if r['phase']=='search']),
                        (parent,data['numeric_sources']+data['six_additions'])):
        recovery.bind_parent_rows(source,rows,checks)
    for name,key in (('parent-current','current'),('parent-refit','refit'),('preserved','preserved'),('cases','cases'),('blocked','blocked'),('numeric-sources','numeric_sources')):
        checks.equal(read(study/f'bootstrap/{name}.json'),data[key],'Exact source cohort/'+name)
    checks.equal(spec['grid'],fixed.fixed_grid(),'Original35 grid unchanged')
    checks.equal(read(study/'grid.json'),fixed.fixed_grid(),'Saved35 grid unchanged')
    bindings=read(study/'source-bindings.json');checks.equal(spec['source_bindings'],bindings,'All source bindings frozen')
    sources=(('parent',parent,170755,previous.PARENT_FP),('historical.parent',older,141041,six.RECOVERY_FP),
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
                checks.equal(label,f'fixed35-broad-{phase}-{role}-r{repeat}',label+'/literal control identity')
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



BROAD_VERSION='sircus-fixed35-broad-bouncer-branches-v1'
BRANCH_RULE=dict(version=BROAD_VERSION,starts_per_arm=32,iterations=360,seed_base=20261112,
    seed_rule='(base+int(sha256(sorted JSON of bucket/values)[:8],16)) %2147483647',
    normal_only=False,preserve_known=False,maximum_unique_pairs=64,maximum_endpoint_candidates=16,
    maximum_path_candidates=16,maximum_path_attempts_per_candidate=2,maximum_saved_path_attempts=32,maximum_admitted_branches=2,distinct_bouncer_joint_rad=.05,
    branch_priority='Farthest proximal q14 from immutable anchor first, then max-min proximal/full q14 diversity',
    pair_priority='One q10 rank across every distinct q14 before the next q10 rank',
    selection_uses_flight_labels=False,position_tolerance_m=1e-4,rotation_tolerance_rad=4e-4)
BROAD_FIELDS=('broad_rule_version','broad_branch_slot','broad_key','broad_pool_file','broad_pool_sha256','broad_pool_key')


def distinct_indices(pool):
    retained=[]
    for index in sorted(range(len(pool['q'])),key=lambda i:(pool['residual'][i],i)):
        if not pool['valid'][index]:continue
        if all(np.max(previous.wrapped_difference(pool['q'][index],pool['q'][old]))>=.05 for old in retained):retained.append(index)
    return retained


def branch_orders(pools,hint):
    first,second=pools
    drop=sorted(distinct_indices(first),key=lambda i:(float(previous.wrapped_difference(first['q'][i],hint['q10']).max()),first['residual'][i],i))
    pending=distinct_indices(second);board=[]
    while pending:
        refs=[second['q'][i] for i in board] or [hint['q14']]
        scores={i:(min(float(previous.wrapped_difference(second['q'][i][:3],q[:3]).max()) for q in refs),
                   min(float(previous.wrapped_difference(second['q'][i],q).max()) for q in refs),-second['residual'][i],-i) for i in pending}
        choice=max(pending,key=lambda i:scores[i]);board.append(choice);pending.remove(choice)
    return drop,board


def audit_endpoint(endpoint,anchor,scene,assets,chains,gate,checks,label):
    from .calibrated_admission import source_geometry
    from .admission import native_pose_error,base14
    checks.equal(endpoint['branch_rule'],BRANCH_RULE,label+'/literal branch rule')
    values,bucket=endpoint['_values'],endpoint['_bucket']
    reach=independent_reach_record(chains,assets,scene,bucket,values)
    check_tree(endpoint['reach_bound'],reach,checks,label+'/independent URDF reach')
    checks.require(reach['passed'],label+'/only reach-admissible frozen proposals enter IK')
    targets=fixed.expected_targets(scene,values,bucket)
    for key,wanted in endpoint['targets'].items():checks.numeric(wanted,targets[key],label+'/independent target/'+key,1e-12)
    savedtargets={k:np.asarray(v) for k,v in endpoint['targets'].items()}
    hint=source_geometry(anchor['effective_source_hint'])
    if anchor.get('resolved'):hint.update({k:anchor['resolved'][k] for k in ('q10','q14','values','bucket_xy_m')})
    checks.equal(endpoint['geometry_hint_sha256'],digest(hint),label+'/immutable geometry-only continuation')
    checks.equal(endpoint['seed'],(20261112+int(digest(dict(bucket_xy_m=bucket,values=values))[:8],16))%2147483647,label+'/action-derived deterministic seed')
    checks.equal(endpoint['selection_uses_flight_labels'],False,label+'/no flight labels in branch selection')
    pools=endpoint['ik_pools'];checks.equal(len(pools),2,label+'/both finite IK pools')
    for arm,(pool,chain,key) in enumerate(zip(pools,chains,('dropper_flange_world','board_flange_local'))):
        q=np.asarray(pool['q']);res=np.asarray(pool['residual'])
        checks.require(q.shape==(33,6) and res.shape==(33,) and np.isfinite(q).all() and np.isfinite(res).all(),label+'/32 canonical plus one continuation solutions')
        checks.require(len(pool['valid'])==33 and all(type(v) is bool for v in pool['valid']),label+'/33 boolean validity decisions')
        checks.equal(endpoint['valid_ik_branches'][arm],sum(pool['valid']),label+'/complete valid count')
        checks.require((q>=chain[2]-1e-10).all() and (q<=chain[3]+1e-10).all(),label+'/all solver joints within original URDF limits')
        for index,joints in enumerate(q):
            pose=yaw.forward_chain(chain,joints);p=float(np.linalg.norm(pose[:3,3]-savedtargets[key][:3,3]));r=yaw.rotation_error(pose[:3,:3],savedtargets[key][:3,:3])
            # The independent acos rotation diagnostic loses precision near zero;
            # use a tiny residual tolerance, never change valid acceptance gates.
            checks.numeric(res[index],math.hypot(p,.25*r),label+f'/arm{arm}/start{index}/independent residual',1e-8)
            if pool['valid'][index]:
                checks.require(p<1e-4 and r<4e-4,label+f'/arm{arm}/start{index}/full pose valid')
    order10,order14=branch_orders(pools,hint)
    checks.equal((endpoint['dropper_branch_order'],endpoint['bouncer_branch_order']),(order10,order14),label+'/independent proximal-first branch ordering')
    checks.equal(endpoint['distinct_ik_branches'],[len(order10),len(order14)],label+'/wrapped branch counts')
    attempts=endpoint['pair_attempts'];checks.require(len(attempts)<=64,label+'/64 checked pair cap')
    cursor=0;accepted=[];bad10=set();bad14=set()
    for i in order10:
        for j in order14:
            if i in bad10 or j in bad14:continue
            if cursor>=64 or len(accepted)>=16:break
            checks.require(cursor<len(attempts),label+'/no omitted finite pair check')
            attempt=attempts[cursor];cursor+=1
            checks.equal((attempt['branch10'],attempt['branch14']),(i,j),label+'/literal q10-outer q14-inner pair order')
            candidate=dict(values=values,bucket_xy_m=bucket,q10=pools[0]['q'][i],q14=pools[1]['q'][j],geometry_eligible=True,branch10=i,branch14=j,endpoint=attempt['checked'])
            independent=gate.check(candidate['q10'],candidate['q14'],bucket,values,targets)
            compare_certificates(gate,candidate,dict(id=label,bucket_xy_m=bucket,values=values),targets,independent,
                                 dict(targets=endpoint['targets'],fresh_endpoint=attempt['checked']),checks)
            checked=attempt['checked']
            if checked['reason']:
                for role in checked.get('independent_reject_arms',[]):(bad10 if role==10 else bad14).add(i if role==10 else j)
                checks.require('native_pose_errors' not in attempt,label+'/rejected pair has no fabricated native FK')
                continue
            q=np.r_[candidate['q10'],candidate['q14']]
            errors=[native_pose_error(gate.strict,role,q,target) for role,target in ((10,savedtargets['dropper_flange_world']),(14,base14(scene)@savedtargets['board_flange_local']))]
            check_tree(attempt['native_pose_errors'],errors,checks,label+'/native model full FK',1e-10)
            rejected=any(e['position_m']>=1e-4 or .25*e['rotation_rad']>=1e-4 for e in errors)
            checks.equal(attempt.get('pose_rejected',False),rejected,label+'/unchanged strict pose acceptance')
            if not rejected:accepted.append(candidate)
        if cursor>=64 or len(accepted)>=16:break
    checks.equal(cursor,len(attempts),label+'/no extra unplanned pair checks')
    checks.equal(endpoint['candidates'],accepted,label+'/every admitted endpoint is independently reproduced')
    checks.equal(endpoint['passed'],bool(accepted),label+'/honest finite IK endpoint result')
    reason=None if accepted else ('finite_ik_unresolved' if not any(pools[0]['valid']) or not any(pools[1]['valid']) else 'strict_endpoint_clearance_or_pose_rejected')
    checks.equal(endpoint['reason'],reason,label+'/bounded failure is not impossible-layout claim')
    return targets


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
    admitted=pool['admitted'];paths=pool['path_attempts'];checks.require(len(admitted)<=2 and len(paths)<=32,label+'/finite branch/path caps')
    checks.equal([a['slot'] for a in admitted],list(range(len(admitted))),label+'/contiguous admitted slots')
    checks.equal([a['candidate_index'] for a in admitted],sorted({a['candidate_index'] for a in admitted}),label+'/first valid unique candidates')
    checks.equal([a['path_attempt_index'] for a in admitted],[i for i,p in enumerate(paths) if p.get('passed')],label+'/all passing paths retained')
    if len(admitted)==2:checks.require(np.max(previous.wrapped_difference(admitted[0]['q14'],admitted[1]['q14']))>=.05,label+'/second slot changes bouncer, not only dropper wrist')
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


def audit_flights(study,rows,prepared,scene,assets,checks):
    from .learning_ik import read_chain
    chains=tuple(read_chain(assets/name) for name in ('dropper.urdf','board_arm.urdf'))
    old=[r for r in rows if r['phase'] not in ('search','repeat')]
    six.audit_geometry(study,old,scene,assets,checks)
    seen=set();count=0
    for row in rows:
        if row['phase'] not in ('search','repeat'):continue
        label=row['id'];packet=row['prepared_request'];original=prepared[packet['proposal_id']+'-b'+str(packet['broad_branch_slot'])]
        for key in ('values','bucket_xy_m','selected_q10','selected_q14','prepared_connection_file','prepared_connection_sha256',*BROAD_FIELDS):
            checks.equal(packet[key],original[key],label+'/no post-preparation command change/'+key)
        checks.equal(row['adapter_directory'],'native-yaw',label+'/fixed-yaw namespace')
        checks.equal((row['connection_file'],row['connection_sha256']),(original['prepared_connection_file'],original['prepared_connection_sha256']),label+'/exact prepared geometry proof')
        checks.require(row['native_release'] is True,label+'/each counted search/repeat is an actual fresh drop')
        connection=read(contained(study/'native-yaw',row['connection_file']))
        targets,_=fixed.audit_behavior(connection,scene,checks,label)
        for key in BROAD_FIELDS:checks.equal(row[key],connection[key],label+'/native immutable branch identity/'+key)
        q=[]
        for chain,field,target in zip(chains,('q10','q14'),(targets['dropper_flange_world'],targets['board_flange_local'])):
            checks.equal(row[field],original['selected_'+field],label+'/native exact prepared joints/'+field)
            checks.equal(row['native_initial_'+field],np.asarray(row[field],np.float32).astype(float).tolist(),label+'/observed float32 joint reset/'+field)
            pose=yaw.forward_chain(chain,row['native_initial_'+field]);q.append(pose)
            checks.require(np.linalg.norm(pose[:3,3]-target[:3,3])<1e-4,label+'/observed position remains within0.1mm/'+field)
            checks.require(yaw.rotation_error(pose[:3,:3],target[:3,:3])<4e-4,label+'/observed full orientation/'+field)
        checks.numeric(row['release_ball_center_m'],targets['release_center_m'],label+'/exact release center',1e-12)
        checks.numeric(row['board_center_m'],row['values'][:3],label+'/settled board center',1e-4)
        checks.numeric(row['board_normal'],targets['board_normal'],label+'/settled board normal',4e-4)
        evidence=row['yaw_evidence'];drop,board=q
        checks.equal(row['release_behavior_version'],'sircus-continuous-bucket-facing-release-v1',label+'/fixed bucket-facing behavior')
        checks.equal(evidence['basis'],'model FK of observed native initial float32 joints; held endpoint',label+'/measured orientation provenance')
        shown=dict(requested_yaw_rad=targets['yaw_rad'],requested_rotation_matrix=targets['dropper_flange_world'][:3,:3],
            measured_initial_rotation_matrix=drop[:3,:3],measured_initial_yaw_rad=math.atan2(drop[1,0],drop[0,0]),
            rotation_error_rad=yaw.rotation_error(drop[:3,:3],targets['dropper_flange_world'][:3,:3]),
            position_error_m=np.linalg.norm(drop[:3,3]-targets['dropper_flange_world'][:3,3]),
            board_rotation_error_rad=yaw.rotation_error(board[:3,:3],targets['board_flange_local'][:3,:3]),
            board_position_error_m=np.linalg.norm(board[:3,3]-targets['board_flange_local'][:3,3]),
            measured_initial_flange_position_m=drop[:3,3],requested_release_center_m=targets['release_center_m'])
        for key,value in shown.items():checks.numeric(evidence[key],value,label+'/independent displayed geometry/'+key,1e-9)
        if row['phase']=='search':
            identity=(tuple(row['values']),tuple(row['q10']),tuple(row['q14']),tuple(row['bucket_xy_m']))
            checks.require(identity not in seen,label+'/discovery quota never filled by duplicate exact joint/action replay');seen.add(identity)
        count+=1
    return dict(new_native_releases=count,unique_discovery_commands=len(seen))


def audit_native(study,rows,data,spec,calibration,assets,checks,parent):
    setup={};count=0
    for namespace,key,parent_namespace in (('native-yaw','prior','native-yaw'),('native-controls','old_prior','control-parent/native')):
        members=[r for r in rows if r['adapter_directory']==namespace];path=study/namespace
        checks.require(bool(members),namespace+'/complete retained native cohort')
        setup[namespace]=calibrated.check_setup_evidence(geometry._NativeNamespace(path),members,data[key],data['home'],checks)
        models.audit_native_options(path,spec['scene'],calibration,assets,checks)
        for row in members:checks.equal((row['effective_options_file'],row['effective_options_sha256']),('effective-options.json',sha(path/'effective-options.json')),row['id']+'/same native64 options')
        for name in ('effective-options.json','native-compiled-geometry.json'):
            checks.equal(read(path/name),read(parent['bound'](parent_namespace+'/'+name)),namespace+'/unchanged pinned physics/'+name)
        count+=len(native.rescore_rows(path,members,spec['scene'],calibration,spec['identity']['trace_schema'],assets,checks))
    return setup,count


def audit_budget(study,report,checks):
    path=study.with_name(study.name+'-budget.json');budget=read(path);pipeline=read(study.with_suffix('.pipeline.json'))
    checks.equal((budget['limit_s'],budget['audit_reserve_s']),(43200,7200),'Twelve-hour worker and two-hour audit reserve')
    checks.equal(budget['gpu_uuid'],'GPU-ab1950b0-01dd-0fab-407f-2d17936b27f4','Only authorized4090')
    checks.require(0<pipeline['global_limit_s']<=51300,'Finite global deadline')
    checks.numeric(pipeline['global_deadline_unix_s']-pipeline['started_unix_s'],pipeline['global_limit_s'],'One absolute global interval',.05)
    checks.equal(budget['global_deadline_unix_s'],pipeline['global_deadline_unix_s'],'Shared original deadline')
    checks.require(pipeline['started_unix_s']<=budget['acquired_unix_s'],'Lease acquired after supervisor start')
    checks.numeric(budget['effective_worker_limit_s'],min(43200,budget['global_deadline_unix_s']-budget['acquired_unix_s']-7200),'Worker reserves independent audit',.1)
    checks.require(60<budget['effective_worker_limit_s']<=43200,'Positive bounded worker interval')
    for key in ('effective_worker_limit_s','global_deadline_unix_s'):checks.equal(report['deadlines'][key],budget[key],'Reported original deadline/'+key)
    checks.equal((report['deadlines']['worker_limit_s'],report['deadlines']['search_limit_s']),(43200,36000),'Approved worker/search caps')
    checks.require(report['elapsed_s']<budget['effective_worker_limit_s'],'Worker finished within acquired deadline')
    return dict(worker_receipt_sha256=sha(path),effective_worker_limit_s=budget['effective_worker_limit_s'],global_limit_s=pipeline['global_limit_s'])


def audit_summary(study,rows,report,spec,data,preserved,gates,checks):
    cases=schedule.independent_cases(rows)
    checks.equal(report['cases'],cases,'Complete per-cell native discovery and fresh confirmation counts')
    checks.equal(report['counts'],statistics(rows),'All native releases included')
    checks.equal(report['stage_counts'],{p:statistics([r for r in rows if r['phase']==p]) for p in ('controls_pre','preserved','search','repeat','controls_post')},'Separate controls/preservation/discovery/repeat denominators')
    checks.equal(report['control_gates'],gates,'Unchanged controls pass both before and after search')
    kept=dict(requests=30,captures=preserved['captures'],expected=30,all_recaptured=preserved['captures']==30)
    checks.equal(report['preservation'],kept,'Every prior recipe freshly replayed');checks.require(kept['all_recaptured'],'All30 successes retained')
    checks.equal((report['historical_policy_captures'],report['parent_captures'],report['parent_positions'],report['training'],report['posthoc_development']),(24,30,35,False,True),'Recipe search is not new policy accuracy')
    checks.equal((report['expected_native_search'],report['maximum_native_releases']),(1000,1075),'1000 actual new drops,1075 total native cap')
    done=report['fresh_search_releases']==QUOTAS
    checks.equal(report['gate'],dict(status='passed' if done else 'incomplete',controls_passed=True,preservation_passed=True,search_quota_completed=done,no_policy_promotion=True),'Honest complete or incomplete gate')
    recipes=dict(preserved_parent=[dict(id=r['id'],sha256=digest(r),values=r['values'],q10=r['q10'],q14=r['q14'],bucket_xy_m=r['bucket_xy_m']) for r in data['preserved']],
        new_confirmed=[r for case in cases for r in case['confirmed_recipes']],policy_unchanged=True)
    checks.equal(read(study/'recipes.json'),recipes,'Only authenticated old recipes and three-repeat new recipes retained')
    return cases


def audit(study,source,assets,*,parent_study,parent_audit,original_study,original_audit,recovery_study,recovery_audit,reference):
    started=time.monotonic();checks=AuditChecks();study,source,assets=map(Path,(study,source,assets))
    manifest=replay.check_manifest(study,checks);spec=read(study/'contract.json');report=read(study/'report.json')
    unsigned=dict(spec);fp=unsigned.pop('fingerprint')
    checks.equal((digest(unsigned),report['contract_fingerprint']),(fp,fp),'Complete frozen experiment identity')
    checks.equal((spec['version'],report['version']),(STUDY_VERSION,STUDY_VERSION),'New broad study identity')
    checks.require(report['status'] in ('completed','incomplete') and report['phase']==report['status'],'Finished finite study, not partial running evidence')
    checks.equal(report['evidence_sha256'],manifest,'Report binds every immutable artifact')
    checks.equal(report['protocol'],spec['protocol'],'Frozen protocol');protocol_checks(spec['protocol'],checks)
    checks.equal(spec['release_behavior'],"Rz(atan2(bucket_y-board_y,bucket_x-board_x)) @ diag(1,-1,-1)", 'Fixed behavior description')
    checks.equal(report['scope'],spec['scope'],'Same development scope')
    replay.check_source_files(source,spec['source_sha256'],checks,'Frozen source')
    from . import fixed35_bouncer_audit_v3 as certificate
    for module in (Path(__file__),Path(oracle.__file__),Path(schedule.__file__),Path(previous.__file__),Path(certificate.__file__)):
        checks.equal(sha(module),spec['source_sha256']['sircus/mjlab/'+module.name],'Actual independent auditor source frozen/'+module.name)
    checks.equal(read(study/'rule.json'),dict(version=STUDY_VERSION,protocol=spec['protocol'],native_branch_rule=BRANCH_RULE),'Literal frozen rule and branch budgets')
    original=recovery.authenticate_parent(original_study,original_audit,checks);older=six.authenticate_recovery(recovery_study,recovery_audit,checks);parent=previous.authenticate_parent(parent_study,parent_audit,checks)
    data=audit_inputs(study,original,older,parent,spec,checks)
    models.audit_runtime_reference(source,reference,spec,parent['spec'],assets,checks)
    checks.equal(sha(study/'geometry-audit.json'),sha(parent['bound']('geometry-audit.json')),'Old collision geometry certificate preserved')
    static=six.audit_static_geometry(study,spec,assets,checks)
    plan=read(study/'search-plan.json');expected=independent_plan(data['numeric_sources'],spec['scene'],assets)
    check_tree(plan,expected,checks,'Independent frozen broad numeric plan',1e-12)
    preserved_plan=independent_preserved(data);checks.equal(read(study/'preserved-plan.json'),dict(rows=preserved_plan,frozen_before_outcomes=True),'Exactly30 original recipe requests')
    sources=data['preserved']+data['numeric_sources'];receipts=[dict(source_id=r['id'],source_row_sha256=digest(r),resolution_file=r['context_resolution_file'],resolution_sha256=r['context_resolution_sha256'],connection_file=r['connection_file'],connection_sha256=r['connection_sha256']) for r in sources]
    checks.equal(read(study/'native-source-receipts.json'),receipts,'Exact42 immutable source-proof hydration receipts')
    calibration=replay.check_calibration(study,spec['identity']['calibration'],checks)
    rows=replay.read_records(study/'rows.jsonl');native.check_rows(rows,checks)
    checks.require(not any(r.get('context_id') in {b['context_id'] for b in data['blocked']} for r in rows),'Neither fixed-base blocked cell searched')
    prepared,preparation=audit_preparations(study,plan,data,spec,assets,checks)
    setup,native_count=audit_native(study,rows,data,spec,calibration,assets,checks,parent)
    flights=audit_flights(study,rows,prepared,spec['scene'],assets,checks)
    preserved=audit_preserved(rows,data,checks);gates=audit_controls(rows,data,checks)
    accounting=schedule.audit_schedule(study,rows,report,data,plan,preserved_plan,independent_repeat,checks)
    packets=geometry.audit_request_packets(study,rows,checks);budget=audit_budget(study,report,checks)
    cases=audit_summary(study,rows,report,spec,data,preserved,gates,checks)
    return dict(version=VERSION,status='passed',full_checks_verified=True,native_checks_verified=True,checks_count=checks.count,checks=checks.entries,failures=[],
        contract_fingerprint=fp,report_sha256=sha(study/'report.json'),evidence_manifest_sha256=sha(study/'evidence-manifest.json'),auditor_sha256=sha(Path(__file__)),producer_source_sha256=spec['source_sha256'],
        consumed_source_files_sha256=dict(original=original['consumed'],recovery=older['consumed'],six=parent['consumed']),requested_trials=len(rows),native_releases=native_count,
        cases=cases,control_gates=gates,preservation=preserved,static_geometry=static,setup_evidence=setup,preparation=preparation,flights=flights,request_accounting=accounting,
        request_packets=packets,budget=budget,elapsed_s=time.monotonic()-started,maximum_numeric_error=checks.numeric_error,
        limitations=['Known fixed-grid development search, not frozen-policy accuracy or held-out generalization.',
          'Geometric outer bounds only reject impossible requested endpoints; passing does not prove feasible joint motion.',
          'Finite CPU IK and sampled setup paths; native flights hold admitted endpoints at64 substeps with original v3 scoring.',
          'Preparation refusals are never counted toward the1000 discovery-drop target; incomplete quotas remain incomplete.',
          'Original30 recipes are retained; new recipes require3 fresh nominal captures. No training, perturbations,128-step qualification or hardware.'])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('study','source','assets','parent-study','parent-audit','original-study','original-audit','recovery-study','recovery-audit','reference','output'):parser.add_argument('--'+key,type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists() or args.output.resolve().is_relative_to(args.study.resolve()):raise ValueError('Fresh independent output outside immutable producer evidence required')
    try:
        kwargs=vars(args).copy();kwargs.pop('output');result=audit(**kwargs)
    except Exception as error:
        args.output.write_text(json.dumps(dict(version=VERSION,status='failed',failures=[type(error).__name__+': '+str(error)]),indent=2,allow_nan=False)+'\n');raise
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:result[k] for k in ('status','checks_count','requested_trials','native_releases')}))


if __name__=='__main__':main()
