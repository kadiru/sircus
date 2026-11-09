"""Independent CPU proof of the literal36-request gripper yaw experiment.

No producer plan builder, summary function, or scorer is imported. Original
proofs, all requested outcomes, strict setup geometry, native reset and dense
v3 scoring are checked independently; a failed audit cannot verify the report.
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
from . import replay650_verify as replay
from . import table_training_verify as native
from . import coverage_training_verify as geometry
from . import table_assisted_score_oracle as oracle
from .benchmark_verify import EvidenceError, contained, digest, read, sha

VERSION='sircus-gripper-yaw-audit-v1'
STUDY_VERSION='sircus-gripper-yaw-v1'
ALL_PARENT_LOSSES=tuple('corrected-imitation-test-'+x for x in ('r0-c0-n0','r0-c1-n0','r0-c6-n1','r1-c1-n0','r1-c2-n0','r3-c0-n1'))
CASES=(ALL_PARENT_LOSSES[2],ALL_PARENT_LOSSES[4])
IK_CASES=()
VARIANTS=('current_exact','corrected_exact','bucket_facing')
MODE='bucket_facing_gripper_yaw'
DIAGNOSIS_SHA256='e16f551802c6c378e7911a0bca2470e4302babb075f8ef570df189839a7bd428'
ROLES=('direct_control','table_control','negative_control')
META=('source_hint','effective_source_hint','source_teacher_id','context_resolution_file',
    'context_resolution_sha256','context_home_sha256','context_layout_eligible','context_anchor_admitted','context_hard_reject','observation')
PARENT_REPORT='1fde3f7de5b5c2912a133c858105388a704a80ab0e2cb39e8d79c0c7a189bf14'
PARENT_AUDIT='ea4992828178d9e62ba41c63579ecc3876cb5bffaacf6b590c828d45876a7f63'
PARENT_FP='297b367fa2702d688084bc2e74dd30a83c6cf4312ba078938def5f9b223adce4'
PARENT_ROWS='f9d48dd328e203b76d7f6be41c01e436ea938701821e902bbbd7945df7248843'
PARENT_MANIFEST='e65451629a36e3c486c3c7dbedd5f2de67de97fac46f6d4631eb3c38c68f3957'



def expected_yaw(values, bucket, radius):
    """Independent target construction; never imports the yaw producer."""
    values=np.asarray(values,float);bucket=np.asarray(bucket,float)
    if values.shape!=(5,) or bucket.shape!=(2,) or not np.isfinite(values).all() or not np.isfinite(bucket).all():
        raise EvidenceError('Finite five shot coordinates and two bucket coordinates required')
    delta=bucket-values[:2]
    if np.linalg.norm(delta)<=.15 or not np.isfinite(radius) or radius<=0:
        raise EvidenceError('Original separated shot geometry and positive radius required')
    theta=math.atan2(float(delta[1]),float(delta[0]));c,s=math.cos(theta),math.sin(theta)
    rotation=np.asarray([[c,s,0.],[s,-c,0.],[0.,0.,-1.]])
    target=np.eye(4);target[:3,:3]=rotation
    target[:3,3]=values[:3]+[0.,0.,float(radius)+values[4]+.095]
    return dict(yaw_rad=theta,rotation_matrix=rotation.tolist(),dropper_flange_world=target.tolist())


def forward_chain(chain,q):
    """Independent scalar URDF FK of a finite six-joint vector."""
    from scipy.spatial.transform import Rotation
    origins,axes,low,high=chain;q=np.asarray(q,float)
    if q.shape!=(6,) or not np.isfinite(q).all() or (q<low-1e-6).any() or (q>high+1e-6).any():
        raise EvidenceError('Finite joint-limit-valid FK state required')
    pose=np.eye(4)
    for origin,axis,angle in zip(origins,axes,q):
        spin=np.eye(4);spin[:3,:3]=Rotation.from_rotvec(axis*angle).as_matrix();pose=pose@origin@spin
    return pose


def rotation_error(actual,expected):
    from scipy.spatial.transform import Rotation
    a,b=np.asarray(actual,float),np.asarray(expected,float)
    if a.shape!=b.shape or a.shape!=(3,3) or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise EvidenceError('Finite full rotation matrices required')
    for matrix in (a,b):
        if not np.allclose(matrix@matrix.T,np.eye(3),atol=2e-6,rtol=0) or abs(np.linalg.det(matrix)-1)>2e-6:
            raise EvidenceError('Proper orthonormal rotations required')
    return float(Rotation.from_matrix(b@a.T).magnitude())

def statistics(rows):
    result=native.statistics(rows)
    result['robot_contact_shots']=sum(r.get('diagnostic_robot_contact_steps',0)>0 for r in rows)
    result['cached_refusal_requests']=sum(r.get('admission_mode')=='cached_parent_refusal' for r in rows)
    return result


def audit_plan(plan,sources,controls,rule,checks):
    checks.equal(set(rule),{'mode','contexts','diagnosis_sha256','description','rotation','preserve_board_joints'},'Literal yaw rule schema')
    for key,value in dict(mode=MODE,contexts=list(CASES),diagnosis_sha256=DIAGNOSIS_SHA256,preserve_board_joints=True,
            rotation='Rz(atan2(bucket_y-board_y,bucket_x-board_x)) @ diag(1,-1,-1)').items():
        checks.equal(rule[key],value,'Single frozen yaw correction/'+key)
    checks.require(isinstance(rule['description'],str) and bool(rule['description'].strip()),'Evidence-based yaw rationale')
    checks.equal(len(sources),4,'Both original methods at two colliding contexts')
    checks.equal([r['role'] for r in controls],list(ROLES),'Three original17 direct/table/miss controls')
    checks.require(all(r.get('native_release') is True for r in controls),'Original admitted controls')
    expected=[(r,'control_exact','controls_pre',i) for r in controls for i in range(3)]
    for context in CASES:
        pair={}
        for method in ('current_bc','corrected_bc'):
            group=[r for r in sources if r['context_id']==context and r['method']==method and r['phase']=='test']
            checks.equal(len(group),1,context+'/one exact historical '+method);pair[method]=group[0]
        current,corrected=pair['current_bc'],pair['corrected_bc']
        checks.require(native.captured(current) and not native.captured(corrected),context+'/original capture regression')
        checks.require(all(r.get('native_release') is True and not r.get('invalid_physics') and not r.get('invalid') for r in pair.values()),context+'/both historical recipes genuinely released')
        checks.require(corrected.get('forbidden_contact_steps',0)>0,context+'/actual forbidden-contact failure')
        checks.equal(current['bucket_xy_m'],corrected['bucket_xy_m'],context+'/matched original bucket')
        for key in META:checks.equal(current[key],corrected[key],context+'/same original source resolution/'+key)
        for variant in VARIANTS:
            source=current if variant=='current_exact' else corrected
            expected.extend((source,variant,'diagnostic',i) for i in range(3))
    expected.extend((r,'control_exact','controls_post',i) for r in controls for i in range(3))
    checks.equal(len(plan),36,'Exactly36 slots with three nominal requests per variant')
    checks.equal(len({r['id'] for r in plan}),36,'No duplicated request identity')
    for request,(source,variant,phase,repeat) in zip(plan,expected):
        values=copy.deepcopy(source['values']);changed=variant=='bucket_facing'
        wanted_yaw=math.atan2(source['bucket_xy_m'][1]-values[1],source['bucket_xy_m'][0]-values[0]) if changed else 0.
        checks.numeric(request.get('requested_gripper_yaw_rad'),wanted_yaw,request['id']+'/independent target-facing angle',1e-12)
        namespace='native-controls' if phase.startswith('controls_') else ('native-yaw' if changed else 'native')
        wanted={k:copy.deepcopy(source[k]) for k in META}
        wanted.update(id=f'gripper-yaw-{phase}-{source["context_id"]}-{variant}-r{repeat}',phase=phase,
            role=source['role'] if phase.startswith('controls_') else 'diagnostic',
            method='control' if phase.startswith('controls_') else variant,variant=variant,repeat=repeat,
            context_id=source['context_id'],bucket_xy_m=source['bucket_xy_m'],values=values,command_values=values,
            source_record_id=source['id'],source_record_sha256=digest(source),donor_record_id=None,donor_record_sha256=None,
            realization_reference_record_id=source['id'],realization_reference_record_sha256=digest(source),
            parameter_indices=[],interpolation_alpha=None,requested_gripper_yaw_rad=request['requested_gripper_yaw_rad'],
            preserved_q14=source['q14'],adapter_directory=namespace,substeps=64,perturbation_mixed_units=[0.]*5,
            selection_or_learning=False,original_role=source['role'],posthoc_development=True,
            admission_mode='fixed_board_yaw_strict_proof' if changed else 'exact_parent_branch')
        checks.equal(request,wanted,request['id']+'/independently reconstructed yaw-only request')


def realized_comparison(row,source,checks):
    """Report IK realization changes independently; never force another branch."""
    if not row['native_release']:
        return None
    values={}
    for key in ('q10','q14'):
        a,b=(np.asarray(item[key],dtype=float) for item in (row,source))
        checks.require(a.shape==b.shape==(6,) and np.isfinite(a).all() and np.isfinite(b).all(),row['id']+'/finite '+key+' branch comparison')
        values[key+'_equal_to_source']=bool(np.array_equal(a,b))
        values[key+'_max_abs_delta_rad']=float(np.max(np.abs(a-b)))
    a,b=(np.asarray(item['board_center_m'],dtype=float) for item in (row,source))
    checks.require(a.shape==b.shape==(3,) and np.isfinite(a).all() and np.isfinite(b).all(),row['id']+'/finite realized board centers')
    values['board_center_delta_m']=(a-b).tolist()
    a,b=(np.asarray(item['board_normal'],dtype=float) for item in (row,source))
    checks.require(a.shape==b.shape==(3,) and np.isfinite(a).all() and np.isfinite(b).all() and np.linalg.norm(a)>0 and np.linalg.norm(b)>0,row['id']+'/finite nonzero realized board normals')
    values['board_normal_angle_deg']=float(np.degrees(np.arccos(np.clip(np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b)),-1.,1.))))
    return dict(source_record_id=source['id'],**values)


def audit_observations(rows,plan,sources,controls,checks):
    checks.equal(len(rows),36,'Complete36-request denominator including any setup declines')
    lookup={r['id']:r for r in sources+controls}
    for row,request in zip(rows,plan):
        for key,value in request.items():checks.equal(row.get(key),value,row['id']+'/frozen request/'+key)
        original=lookup[request['source_record_id']]
        checks.equal(request['values'],original['values'],row['id']+'/all five original shot coordinates unchanged')
        checks.equal(request['bucket_xy_m'],original['bucket_xy_m'],row['id']+'/original bucket unchanged')
        checks.equal(row.get('realization_comparison'),realized_comparison(row,original,checks),row['id']+'/actual pose comparison uses its own unchanged source')
        if request['variant']!='bucket_facing':
            for key in ('connection_file','connection_sha256','native_release','q10','q14'):
                checks.equal(row.get(key),original.get(key),row['id']+'/exact original admission/branch/'+key)
        else:
            checks.require(bool(row.get('connection_sha256')) and row['connection_sha256']!=original['connection_sha256'],row['id']+'/yaw-bound fresh proof cannot alias five-value original proof')
            checks.equal(row['adapter_directory'],'native-yaw',row['id']+'/separate yaw proof and native namespace')
            if row.get('native_release'):
                checks.equal(row['q14'],original['q14'],row['id']+'/original bouncer joint branch exactly fixed')
                checks.equal(row['native_initial_q14'],np.asarray(original['q14'],np.float32).astype(float).tolist(),row['id']+'/measured bouncer reset unchanged')
        if row.get('native_release'):
            for arm in ('q10','q14'):
                checks.equal(row['native_initial_'+arm],np.asarray(row[arm],np.float32).astype(float).tolist(),row['id']+'/actual float32 held reset/'+arm)
        else:
            checks.require(not row.get('native_cache_file') and not row.get('native_initial_q10') and not row.get('native_initial_q14'),row['id']+'/decline has no invented native flight')
    for context in CASES:
        changed=[r for r in rows if r['context_id']==context and r['variant']=='bucket_facing']
        checks.equal(len(changed),3,context+'/three frozen yaw requests')
        checks.equal(len({r['connection_sha256'] for r in changed}),1,context+'/one immutable yaw proof across nominal repeats')
        checks.require(all(r.get('q10')==changed[0].get('q10') and r.get('q14')==changed[0].get('q14') for r in changed),context+'/same admitted posture across repeats')


def audit_setup_summaries(study,rows,checks):
    """Reconstruct display admission evidence from hash-bound proof records."""
    for row in rows:
        root=Path(study)/row['adapter_directory']
        path=contained(root,row['connection_file']);resolution_path=contained(root,row['context_resolution_file'])
        checks.equal(sha(path),row['connection_sha256'],row['id']+'/setup summary connection bytes')
        checks.equal(sha(resolution_path),row['context_resolution_sha256'],row['id']+'/setup summary resolution bytes')
        connection=read(path);resolution=read(resolution_path)
        endpoints=connection.get('endpoint_attempts') or {};examples=[]
        for pair in endpoints.get('pair_attempts',[]):
            checked=pair.get('checked') or {};strict=checked.get('strict_geometry') or {}
            examples.append(dict(branch10=pair.get('branch10'),branch14=pair.get('branch14'),
                reason=checked.get('reason'),board_position_error_m=checked.get('board_position_error_m'),
                dropper_position_error_m=checked.get('dropper_position_error_m'),board_normal_error_deg=checked.get('board_normal_error_deg'),
                collision_pairs=checked.get('pairs',[]),independent_reject_arms=checked.get('independent_reject_arms',[]),
                pose_rejected=pair.get('pose_rejected',False),native_pose_errors=pair.get('native_pose_errors'),
                minimum_clearance_m=strict.get('minimum_clearance_m'),closest_pair=strict.get('closest_pair'),
                margin_contacts=strict.get('margin_contacts',[])))
        branches=endpoints.get('valid_ik_branches');group=connection.get('reason')
        if group=='finite_ik_unresolved' and isinstance(branches,list) and len(branches)==2:
            group='finite_ik_'+('both_arms' if branches==[0,0] else 'dropper' if branches[0]==0 else 'board' if branches[1]==0 else 'unknown')
        home=resolution.get('home_geometry') or {};proof=connection.get('proof') or {}
        wanted=dict(reason=connection.get('reason'),group=group or 'admitted',passed=connection['passed'],
            connection_sha256=row['connection_sha256'],resolution_sha256=row['context_resolution_sha256'],
            action_substituted=connection['action_substituted'],valid_ik_branches=branches,
            endpoint_examples=examples,endpoint_pair_attempts=len(examples),
            admitted_branches=[dict(branch10=c.get('branch10'),branch14=c.get('branch14')) for c in endpoints.get('candidates',[]) if c.get('q10')==row.get('q10') and c.get('q14')==row.get('q14')],
            layout_eligible=resolution.get('layout_eligible'),anchor_admitted=resolution.get('anchor_admitted'),
            proof_minimum_clearance_m=proof.get('minimum_clearance_m'),home_minimum_clearance_m=home.get('minimum_clearance_m'),
            home_closest_pair=home.get('closest_pair'),fixed_base_blockers=home.get('fixed_base_blockers',[]),required_clearance_m=.005,
            scope='Recorded finite admission proof; rejection does not prove unreachability, and a conservative zero bound does not prove penetration.')
        checks.equal(row.get('setup_proof'),wanted,row['id']+'/independent full admission summary')
        checks.equal(wanted['action_substituted'],False,row['id']+'/no silent target substitution')
        if row['native_release']:
            checks.equal(wanted['passed'],True,row['id']+'/actual release has admitted setup')
        else:
            checks.equal(wanted['passed'],False,row['id']+'/no flight fabricated for rejected setup')


def audit_parent(study,parent_study,parent_audit,spec,checks):
    checks.equal(sha(parent_study/'report.json'),PARENT_REPORT,'Exact original corrected-imitation report')
    checks.equal(sha(parent_audit),PARENT_AUDIT,'Exact complete independent corrected-imitation audit')
    report,proof,parent,manifest=(read(p) for p in (parent_study/'report.json',parent_audit,parent_study/'contract.json',parent_study/'evidence-manifest.json'))
    unsigned=dict(parent);fp=unsigned.pop('fingerprint')
    checks.equal((digest(unsigned),fp,report['contract_fingerprint'],proof['contract_fingerprint']),(PARENT_FP,)*4,'Immutable original corrected-imitation contract')
    checks.equal((proof['status'],proof['full_checks_verified'],proof['native_checks_verified'],proof['failures']),('passed',True,True,[]),'Full native parent proof passed')
    checks.equal((proof['checks_count'],len(proof['checks'])),(55633,55633),'Every original audit check retained')
    checks.require(all(c['passed'] is True for c in proof['checks']),'All original audit checks passed')
    checks.equal((report['status'],report['phase']),('completed','completed'),'Original study finished')
    checks.equal(report['evidence_sha256'],manifest,'Parent report binds original manifest')
    checks.equal(proof['report_sha256'],PARENT_REPORT,'Original audit authenticates original report')
    checks.equal(proof['evidence_manifest_sha256'],sha(parent_study/'evidence-manifest.json'),'Original manifest authenticated')
    bindings=spec['parent_bindings'];files=bindings['files_sha256']
    checks.equal({k:v for k,v in bindings.items() if k!='files_sha256'},dict(report_sha256=PARENT_REPORT,audit_sha256=PARENT_AUDIT,
        contract_fingerprint=PARENT_FP,manifest_sha256=sha(parent_study/'evidence-manifest.json')),'Published ancestry bindings')
    for name,expected in files.items():
        checks.equal(expected,manifest[name],'Consumed parent file in original manifest/'+name)
        checks.equal(sha(contained(parent_study,name)),expected,'Consumed parent bytes unchanged/'+name)
    checks.equal(sha(parent_study/'rows.jsonl'),PARENT_ROWS,'Literal original158-row ledger bytes')
    checks.equal(sha(parent_study/'evidence-manifest.json'),PARENT_MANIFEST,'Literal original evidence manifest bytes')
    rows=replay.read_records(parent_study/'rows.jsonl');checks.equal((len(rows),len({r['id'] for r in rows})),(158,158),'Complete original158 ledger')
    sources=read(study/'parent-rows.json');controls=read(study/'control-replays.json')
    paired={}
    for row in rows:
        if row.get('phase')=='test':paired.setdefault(row['context_id'],{})[row['method']]=row
    checks.equal(len(paired),70,'All70 original paired positions retained')
    checks.require(all(set(p)=={'current_bc','corrected_bc'} for p in paired.values()),'Exactly both original methods per position')
    losses=sorted(context for context,p in paired.items() if native.captured(p['current_bc']) and not native.captured(p['corrected_bc']))
    checks.equal(losses,list(ALL_PARENT_LOSSES),'All six historical losses unchanged; two robot-contact cases prespecified')
    expected=[]
    for context in CASES:
        for method in ('current_bc','corrected_bc'):
            group=[r for r in rows if r.get('phase')=='test' and r.get('context_id')==context and r.get('method')==method]
            checks.equal(len(group),1,'Original unique source/'+context+'/'+method);expected.append(group[0])
    checks.equal(sources,expected,'Full original source rows copied without relabeling')
    checks.equal(controls,read(parent_study/'control-replays.json'),'Original17 controls copied exactly')
    required={'rows.jsonl','control-replays.json','contract.json','prior.json','old-prior.json','calibration.json','verification.json','setup-home.json'}
    for namespace,group in (('native',sources),('control-parent/native',controls)):
        required.update(namespace+'/'+name for name in ('effective-options.json','native-compiled-geometry.json'))
        for row in group:
            for key,hkey in (('context_resolution_file','context_resolution_sha256'),('connection_file','connection_sha256')):
                name=namespace+'/'+row[key];required.add(name);checks.equal(files.get(name),row[hkey],row['id']+'/exact original proof binding/'+key)
    for row in sources:
        if row.get('native_release'):
            name='native/'+row['native_cache_file'];required.add(name)
            checks.equal(files.get(name),row['native_cache_sha256'],row['id']+'/original dense trace hash bound')
    checks.equal(set(files),required,'Every consumed original proof and dense trace accounted once')
    for name in ('prior.json','old-prior.json','calibration.json','verification.json','setup-home.json'):
        checks.equal(sha(study/name),sha(parent_study/name),'Inherited configuration byte-identical/'+name)
    for key in ('scene','identity','packages','runtime_reference'):checks.equal(spec[key],parent[key],'Unchanged native identity/'+key)
    return sources,controls,parent


def audit_diagnosis(study,rule,spec,checks):
    diagnosis=read(study/'diagnosis.json');unsigned=dict(diagnosis);fp=unsigned.pop('diagnosis_fingerprint')
    checks.equal((digest(unsigned),fp),('eb7a03972c4b6ef731a9eb3a477f670ccdcfe022b5c80cc4a20d7323b9258d59',)*2,'Complete reviewed CPU diagnosis fingerprint')
    checks.equal(diagnosis['version'],'sircus-imitation-regression-diagnosis-v1','Named saved-evidence diagnosis')
    checks.equal((sha(study/'diagnosis.json'),spec['diagnosis_sha256'],rule['diagnosis_sha256']),(DIAGNOSIS_SHA256,)*3,'Reviewed diagnosis frozen before experiment')
    for key,value in dict(report_sha256=PARENT_REPORT,audit_sha256=PARENT_AUDIT,contract_fingerprint=PARENT_FP,
        rows_sha256=spec['parent_bindings']['files_sha256']['rows.jsonl'],full_checks_verified=True,native_checks_verified=True,checks_count=55633).items():
        checks.equal(diagnosis['source'][key],value,'CPU diagnosis original evidence binding/'+key)
    checks.equal([r['context_id'] for r in diagnosis['losses']],list(ALL_PARENT_LOSSES),'All six original regressions retained without relabeling')


def audit_timing(study,rows,report,checks):
    boundaries=replay.read_records(study/'phase-boundaries.jsonl')
    checks.equal([(r['phase'],r['rows_count']) for r in boundaries],[('plan-frozen',0),('controls_pre-complete',9),('diagnostic-complete',27),('controls_post-complete',36)],'One frozen plan and complete bracketing phases')
    raw=(study/'rows.jsonl').read_bytes().splitlines(keepends=True);previous=0
    for boundary in boundaries:
        n=boundary['rows_count'];expected=hashlib.sha256(b''.join(raw[:n])).hexdigest() if n else None
        checks.equal(boundary['rows_sha256'],expected,boundary['phase']+'/immutable observed prefix')
        checks.equal(boundary['plan_sha256'],sha(study/'plan.json'),boundary['phase']+'/same pre-outcome plan')
        checks.require(previous<=boundary['acquired_elapsed_s']<3600,'Monotone bounded own worker times');previous=boundary['acquired_elapsed_s']
    packets=geometry.audit_request_packets(study,rows,checks)
    timing=replay.read_records(study/'timing.jsonl')
    checks.equal([t['phase'] for t in timing],['controls_pre']+['diagnostic']*4+['controls_post'],'Exactly six namespace-separated prescribed batches, no retries')
    checks.require(boundaries[0]['acquired_elapsed_s']<=timing[0]['acquired_start_s'],'All36 decisions frozen before first outcome')
    checks.require(all(0<=t['acquired_start_s']<=t['acquired_end_s']<3600 for t in timing),'All native batches within one-hour limit')
    budget=read(study.with_name(study.name+'-budget.json'));pipeline=read(study.with_suffix('.pipeline.json'))
    checks.equal(budget['gpu_uuid'],'GPU-ab1950b0-01dd-0fab-407f-2d17936b27f4','Only owned authorized4090 UUID')
    checks.equal(budget['audit_reserve_s'],1800,'Half-hour CPU audit reserve')
    checks.require(60<=budget['limit_s']<=3600 and 0<pipeline['global_limit_s']<=6300,'Finite worker and total pipeline caps')
    checks.numeric(pipeline['global_deadline_unix_s']-pipeline['started_unix_s'],pipeline['global_limit_s'],'Original absolute deadline',.05)
    checks.equal(budget['global_deadline_unix_s'],pipeline['global_deadline_unix_s'],'Worker shares original pipeline deadline')
    checks.require(pipeline['started_unix_s']<=budget['acquired_unix_s'],'Own lease follows supervisor start')
    checks.numeric(budget['effective_worker_limit_s'],min(budget['limit_s'],budget['global_deadline_unix_s']-budget['acquired_unix_s']-1800),'Postqueue effective worker budget',.1)
    checks.equal(report['deadlines'],dict(worker_limit_s=budget['limit_s'],effective_worker_limit_s=budget['effective_worker_limit_s'],global_deadline_unix_s=budget['global_deadline_unix_s']),'Report actual immutable budget')
    checks.require(0<report['elapsed_s']<budget['effective_worker_limit_s']<=3600,'Worker completes within acquired allowance')
    return packets,dict(worker_receipt_sha256=sha(study.with_name(study.name+'-budget.json')),effective_worker_limit_s=budget['effective_worker_limit_s'],global_limit_s=pipeline['global_limit_s'])


def independent_cases(plan,rows):
    result=[]
    for context in CASES:
        planned=[r for r in plan if r['context_id']==context];group=[r for r in rows if r['context_id']==context];outcomes=[]
        for row in group:
            item={k:copy.deepcopy(row.get(k)) for k in ('id','variant','repeat','values','native_release','invalid',
                'invalid_physics','connection_sha256','native_cache_file','native_cache_row','native_cache_sha256','admission_mode',
                'board_center_m','board_normal','pose_gate','realization_comparison','setup_proof','yaw_evidence','adapter_directory')}
            item.update(capture=native.captured(row),table_contact_steps=row.get('table_contact_steps',0),robot_contact_steps=row.get('diagnostic_robot_contact_steps',0));outcomes.append(item)
        values={v:next(r['values'] for r in planned if r['variant']==v) for v in VARIANTS}
        result.append(dict(context_id=context,bucket_xy_m=planned[0]['bucket_xy_m'],current_values=values['current_exact'],
            corrected_values=values['corrected_exact'],bucket_facing_values=values['bucket_facing'],original_yaw_rad=0.,
            target_yaw_rad=next(r['requested_gripper_yaw_rad'] for r in planned if r['variant']=='bucket_facing'),variant_values=values,
            variants={v:statistics([r for r in group if r['variant']==v]) for v in VARIANTS},outcomes=outcomes))
    return result



def audit_intervention(connection,source,radius,checks,label):
    """Bind full yaw geometry into a distinct proof key, independently of producer."""
    from .calibrated_admission import CONNECTION_POLICY_VERSION
    target=expected_yaw(source['values'],source['bucket_xy_m'],radius)
    ext=connection['yaw_intervention']
    static=dict(version='sircus-bucket-facing-fixed-board-yaw-v1',source_record_id=source['id'],
        source_record_sha256=digest(source),fixed_q14=source['q14'],source_q10=source['q10'],
        direction='local +X passage points toward bucket XY; no modulo180 substitution',
        board_branch_fixed=True,source_values=source['values'])
    checks.equal(set(ext),set(static)|{'requested_yaw_rad','requested_rotation_matrix','requested_dropper_flange_world','release_center_m'},label+'/literal full-pose extension schema')
    for key,value in static.items():checks.equal(ext[key],value,label+'/original binding/'+key)
    for key,wanted in (('requested_yaw_rad',target['yaw_rad']),('requested_rotation_matrix',target['rotation_matrix']),
            ('requested_dropper_flange_world',target['dropper_flange_world'])):
        checks.numeric(ext[key],wanted,label+'/independent full-pose target/'+key,1e-12)
    release=np.asarray(source['values'][:3])+[0,0,radius+source['values'][4]]
    checks.numeric(ext['release_center_m'],release,label+'/same release ball center',1e-12)
    checks.numeric(ext['release_center_m'],source['release_ball_center_m'],label+'/original measured release center',1e-12)
    checks.equal(connection['values'],source['values'],label+'/all five source values exact')
    checks.equal(connection['bucket_xy_m'],source['bucket_xy_m'],label+'/source bucket exact')
    checks.equal(connection['q14'],source['q14'],label+'/even rejected proof keeps original board joints')
    checks.equal(connection['connection_policy'],CONNECTION_POLICY_VERSION,label+'/unchanged setup path policy')
    key=digest(dict(version=static['version'],connection_policy=CONNECTION_POLICY_VERSION,
        resolution_sha256=connection['resolution_sha256'],values=source['values'],
        bucket_xy_m=source['bucket_xy_m'],yaw_intervention=ext))
    checks.equal(connection['yaw_key'],key,label+'/key covers source joint branch and full target matrix')
    return target,key


def audit_yaw_geometry(study,rows,sources,scene,assets,checks):
    """Recompute endpoint orientation using URDF FK, not claimed yaw summaries."""
    from .learning_ik import read_chain,shot_targets
    from .corrected_geometry import StrictEndpointGate
    chain=read_chain(Path(assets)/'dropper.urdf');board_chain=read_chain(Path(assets)/'board_arm.urdf')
    gate=StrictEndpointGate(scene,assets,.005)
    lookup={s['id']:s for s in sources};seen=set();summary=[]
    radius=scene['ball']['radius_m']
    for row in rows:
        source=lookup[row['source_record_id']];label=row['id'];changed=row['variant']=='bucket_facing'
        target=expected_yaw(row['values'],row['bucket_xy_m'],radius)
        requested=np.asarray(target['rotation_matrix']) if changed else np.diag([1.,-1.,-1.])
        targets={k:v[0] for k,v in shot_targets(scene,np.asarray([row['bucket_xy_m']]),np.asarray([row['values'][:3]]),[row['values'][3]],[row['values'][4]]).items()}
        targets['dropper_flange_world'][:3,:3]=requested
        if changed and row['connection_sha256'] not in seen:
            path=contained(study/'native-yaw',row['connection_file']);connection=read(path)
            checks.equal(sha(path),row['connection_sha256'],label+'/fresh full-pose proof bytes')
            _,key=audit_intervention(connection,source,radius,checks,label)
            checks.equal(row['connection_file'],'yaw-connections/'+key+'.json',label+'/explicit yaw namespace, not five-value cache key')
            endpoints=connection.get('endpoint_attempts')
            if endpoints is not None:
                for name,value in dict(starts=16,iterations=180,normal_only=False,fixed_q14=source['q14']).items():
                    checks.equal(endpoints[name],value,label+'/fresh dropper-only finite solver/'+name)
                checks.numeric(endpoints['targets']['dropper_flange_world'],targets['dropper_flange_world'],label+'/solver full orientation target',1e-12)
                checks.equal(endpoints['valid_ik_branches'][1],1,label+'/one fixed board branch')
                checks.require(len(endpoints['candidates'])<=6,label+'/at most six eligible endpoints, no flight search')
                checks.equal(len(set(endpoints['branch_order'])),len(endpoints['branch_order']),label+'/unique deterministic branch IDs')
                previous=None
                for n,pair in enumerate(endpoints['pair_attempts']):
                    plabel=label+'/candidate'+str(n)
                    checks.equal(pair['q14'],source['q14'],plabel+'/exact board joints')
                    checks.equal(pair['branch14'],'preserved_corrected',plabel+'/board branch identity')
                    checks.require(pair['branch10'] in endpoints['branch_order'],plabel+'/finite solver branch')
                    distance=float(np.linalg.norm(np.asarray(pair['q10'])-source['q10']))
                    checks.numeric(pair['source_joint_distance_rad'],distance,plabel+'/actual source-joint distance',1e-10)
                    rank=(distance,float(pair['residual']),pair['branch10'])
                    checks.require(all(np.isfinite(rank)) and (previous is None or previous<=rank),plabel+'/branch ranking independent of flight rewards');previous=rank
                    actual=gate.check(pair['q10'],pair['q14'],row['bucket_xy_m'],row['values'],targets)
                    checks.equal(actual['reason'],pair['checked']['reason'],plabel+'/fresh strict endpoint reason')
                    if actual['reason']==0:
                        checks.require(actual['minimum_clearance_lower_bound_m']>=.005-1e-12,plabel+'/fresh strict 5mm clearance')
                        checks.numeric(pair['checked']['minimum_clearance_lower_bound_m'],actual['minimum_clearance_lower_bound_m'],plabel+'/fresh clearance value',1e-10)
                for candidate in endpoints['candidates']:
                    checks.equal(candidate['q14'],source['q14'],label+'/eligible branch holds original board joints')
                    matches=[p for p in endpoints['pair_attempts'] if p['branch10']==candidate['branch10'] and p['q10']==candidate['q10'] and p['q14']==candidate['q14']]
                    checks.equal(len(matches),1,label+'/eligible endpoint has checked branch evidence')
                    checks.require(matches[0]['checked']['reason']==0 and not matches[0].get('pose_rejected'),label+'/eligible branch passed both pose and clearance')
                    pose=forward_chain(chain,candidate['q10'])
                    checks.require(np.linalg.norm(pose[:3,3]-targets['dropper_flange_world'][:3,3])<1e-4,label+'/eligible FK position tolerance')
                    checks.require(rotation_error(pose[:3,:3],requested)<4e-4,label+'/eligible FK full orientation tolerance')
                    board=forward_chain(board_chain,candidate['q14'])
                    checks.require(np.linalg.norm(board[:3,3]-targets['board_flange_local'][:3,3])<1e-4,label+'/preserved board FK position tolerance')
                    checks.require(rotation_error(board[:3,:3],targets['board_flange_local'][:3,:3])<4e-4,label+'/preserved board full rotation tolerance')
            seen.add(row['connection_sha256'])
        evidence=row['yaw_evidence']
        checks.equal(evidence['basis'],'URDF/native geometry FK of observed native_initial_q10 (float32); held endpoint',label+'/honest FK measurement provenance')
        checks.numeric(evidence['requested_yaw_rad'],target['yaw_rad'] if changed else 0.,label+'/declared yaw',1e-12)
        checks.numeric(evidence['requested_rotation_matrix'],requested,label+'/declared full rotation',1e-12)
        checks.equal(evidence['nominal_source_q14'],source['q14'],label+'/source board evidence')
        checks.equal(row['q14'],source['q14'],label+'/board branch unchanged')
        checks.equal(evidence['board_q14_unchanged'],True,label+'/board equality claim')
        if not row.get('native_release'):
            checks.equal(set(evidence),{'requested_yaw_rad','requested_rotation_matrix','nominal_source_q14','basis','board_q14_unchanged','release_unchanged'},label+'/no fabricated measured orientation after refusal')
            checks.equal(evidence['release_unchanged'],None,label+'/no release claim after refusal')
            continue
        pose=forward_chain(chain,row['native_initial_q10']);err=rotation_error(pose[:3,:3],requested)
        checks.require(err<4e-4,label+'/measured native-reset FK has requested full rotation')
        checks.require(np.linalg.norm(pose[:3,3]-targets['dropper_flange_world'][:3,3])<1e-4,label+'/actual flange position unchanged')
        checks.numeric(evidence['measured_initial_rotation_matrix'],pose[:3,:3],label+'/independent actual full orientation',1e-9)
        checks.numeric(evidence['measured_initial_flange_position_m'],pose[:3,3],label+'/independent actual flange position',1e-9)
        checks.numeric(evidence['measured_initial_yaw_rad'],math.atan2(pose[1,0],pose[0,0]),label+'/independent actual yaw',1e-9)
        checks.numeric(evidence['rotation_error_rad'],err,label+'/independent full rotation error',1e-9)
        checks.numeric(row['release_ball_center_m'],source['release_ball_center_m'],label+'/unchanged released ball center',1e-12)
        checks.equal(evidence['release_unchanged'],True,label+'/release center claim')
        checks.equal(evidence['board_native_initial_q14_unchanged'],True,label+'/native board equality claim')
        checks.numeric(row['board_center_m'],source['board_center_m'],label+'/actual board center unchanged',1e-8)
        checks.numeric(row['board_normal'],source['board_normal'],label+'/actual board normal unchanged',1e-8)
        summary.append(dict(id=row['id'],variant=row['variant'],requested_yaw_rad=target['yaw_rad'] if changed else 0.,
            measured_yaw_rad=math.atan2(pose[1,0],pose[0,0]),rotation_error_rad=err,board_branch_unchanged=True))
    checks.equal(len(seen),2,'Exactly two independent fixed-board yaw setup proofs')
    return dict(proof_count=len(seen),released_pose_checks=len(summary),measurements=summary,
        basis='Independent URDF FK of observed float32 reset joints; not a separate GPU orientation sensor.')


def audit_native_namespaces(study,rows,prior,old_prior,home,spec,calibration,assets,checks):
    """Audit each isolated namespace without weakening original native checks."""
    setup={};native_count=0
    for namespace,bank in (('native',prior),('native-yaw',prior),('native-controls',old_prior)):
        cohort=[r for r in rows if r['adapter_directory']==namespace]
        checks.require(bool(cohort),namespace+'/nonempty intended request cohort')
        path=study/namespace
        setup[namespace]=geometry.calibrated.check_setup_evidence(geometry._NativeNamespace(path),cohort,bank,home,checks)
        admitted=[r for r in cohort if r.get('native_release')]
        if admitted:
            geometry.old.audit_native_options(path,spec['scene'],calibration,assets,checks)
            for row in admitted:
                checks.equal((row['effective_options_file'],row['effective_options_sha256']),('effective-options.json',sha(path/'effective-options.json')),row['id']+'/same64 physics options')
        else:
            checks.require(not list(path.rglob('*.npz')),namespace+'/no fake physics cache for wholly refused namespace')
        traces=native.rescore_rows(path,cohort,spec['scene'],calibration,spec['identity']['trace_schema'],assets,checks)
        native_count+=len(traces)
    for name in ('effective-options.json','native-compiled-geometry.json'):
        checks.equal(read(study/'native'/name),read(study/'native-controls'/name),'Controls and originals use identical physics/'+name)
        if any(r.get('native_release') for r in rows if r['variant']=='bucket_facing'):
            checks.equal(read(study/'native'/name),read(study/'native-yaw'/name),'Yaw changes only arm posture, never native physics/'+name)
    return setup,native_count

def audit(study,source,assets,*,parent_study,parent_audit):
    started=time.monotonic();checks=replay.AuditChecks()
    study,source,assets,parent_study,parent_audit=map(Path,(study,source,assets,parent_study,parent_audit))
    manifest=replay.check_manifest(study,checks);spec=read(study/'contract.json');report=read(study/'report.json')
    unsigned=dict(spec);fp=unsigned.pop('fingerprint')
    checks.equal((digest(unsigned),report['contract_fingerprint']),(fp,fp),'Complete experiment contract fingerprint')
    checks.equal((spec['version'],report['version'],report['status'],report['phase']),(STUDY_VERSION,STUDY_VERSION,'completed','completed'),'Completed literal regression experiment')
    checks.equal(report['evidence_sha256'],manifest,'Final report binds every original artifact')
    checks.equal(spec['protocol'],report['protocol'],'Protocol unchanged after outcomes')
    for key,value in dict(substeps=64,maximum_requests=36,repeats=3,diagnostic_requests=18,control_requests=18,
        maximum_acquired_s=3600,maximum_queue_s=900,maximum_pipeline_s=6300,audit_reserve_s=1800,storage_reserve_bytes=8*1024**3,
        training=False,noise=False,posthoc_development=True,variants=list(VARIANTS),contexts=list(CASES),mode=MODE,
        changed_parameter='dropper_flange_world_yaw',parameter_indices=[],
        rotation='Rz(atan2(bucket_y-board_y,bucket_x-board_x)) @ diag(1,-1,-1)',
        preserve=['bucket_xy_m','five_shot_values','release_ball_center_m','board_q14','geometry','physics','scoring']).items():
        checks.equal(spec['protocol'][key],value,'Literal frozen design/'+key)
    replay.check_source_files(source,spec['source_sha256'],checks,'Executed frozen source snapshot')
    for module in (Path(__file__),Path(oracle.__file__)):
        checks.equal(sha(module),spec['source_sha256']['sircus/mjlab/'+module.name],'Actual independent checker frozen/'+module.name)
    replay.audit_packages(spec['packages'],checks)
    sources,controls,parent=audit_parent(study,parent_study,parent_audit,spec,checks)
    for name,expected in spec['identity']['asset_sha256'].items():checks.equal(sha(contained(assets.parent,name)),expected,'Original robot asset/'+name)
    for name,key in (('plan.json','plan_sha256'),('rule.json','rule_sha256'),('parent-rows.json','parent_rows_sha256'),('control-replays.json','control_replays_sha256')):
        checks.equal(sha(study/name),spec[key],'Frozen experiment input/'+name)
    plan=read(study/'plan.json');rule=read(study/'rule.json');checks.equal(rule,spec['rule'],'Reviewed rule unchanged')
    checks.equal(report['rule'],rule,'Report discloses actual intervention')
    audit_diagnosis(study,rule,spec,checks);audit_plan(plan,sources,controls,rule,checks)
    rows=replay.read_records(study/'rows.jsonl');audit_observations(rows,plan,sources,controls,checks);audit_setup_summaries(study,rows,checks);native.check_rows(rows,checks)
    calibration=replay.check_calibration(study,spec['identity']['calibration'],checks)
    setup,native_count=audit_native_namespaces(study,rows,read(study/'prior.json'),read(study/'old-prior.json'),read(study/'setup-home.json'),spec,calibration,assets,checks)
    yaw_geometry=audit_yaw_geometry(study,rows,sources+controls,spec['scene'],assets,checks)
    for namespace,parent_namespace in (('native','native'),('native-yaw','native'),('native-controls','control-parent/native')):
        if namespace=='native-yaw' and not any(r.get('native_release') for r in rows if r['adapter_directory']==namespace):continue
        for name in ('effective-options.json','native-compiled-geometry.json'):
            checks.equal(read(study/namespace/name),read(parent_study/parent_namespace/name),'Actual compiled geometry/options match parent/'+namespace+'/'+name)
        files={}
        for row in [r for r in rows if r['adapter_directory']==namespace]:
            if row['native_release']:files.setdefault(row['native_cache_file'],[]).append(row)
            else:checks.equal(row['diagnostic_robot_contact_steps'],0,row['id']+'/no flight means no measured contact')
        for name,members in files.items():
            with np.load(contained(study/namespace,name),allow_pickle=False) as data:
                for row in members:
                    n=int(np.count_nonzero(data['forces_N'][:int(row['physics_steps']),int(row['native_cache_row']),4]>.01))
                    checks.equal(row['diagnostic_robot_contact_steps'],n,row['id']+'/independent dense robot-force count')
    packets,budget=audit_timing(study,rows,report,checks)
    gates={}
    for phase,key in (('controls_pre','pre'),('controls_post','post')):
        gates[key]=native.control_gate([r for r in rows if r['phase']==phase]);checks.equal(gates[key]['status'],'passed',phase+'/only formal direct/table/miss hard gate')
    checks.equal(report['control_gates'],gates,'All18 original17 control outcomes')
    counts={v:statistics([r for r in rows if r['variant']==v]) for v in VARIANTS};cases=independent_cases(plan,rows)
    checks.equal(report['counts'],statistics(rows),'Full36 denominator includes all refusals')
    checks.equal(report['counts_by_variant'],counts,'All six requested outcomes per diagnostic variant')
    checks.equal(report['cases'],cases,'Exact matched cases and every outcome retained')
    checks.equal((report['expected_requests'],report['training'],report['posthoc_development']),(36,False,True),'No training or new held-out evaluation claim')
    checks.equal(report['gate'],dict(status='passed',substeps=64,controls_passed=True,complete_plan=True,no_policy_promotion=True),'Complete execution without deployment')
    return dict(version=VERSION,status='passed',full_checks_verified=True,native_checks_verified=True,checks_count=checks.count,checks=checks.entries,failures=[],
        contract_fingerprint=fp,report_sha256=sha(study/'report.json'),evidence_manifest_sha256=sha(study/'evidence-manifest.json'),auditor_sha256=sha(Path(__file__)),
        requested_trials=len(rows),native_releases=native_count,counts_by_variant=counts,cases=cases,control_gates=gates,
        setup_evidence=setup,yaw_geometry=yaw_geometry,request_packets=packets,budget=budget,elapsed_s=time.monotonic()-started,maximum_numeric_error=checks.numeric_error,
        limitations=['Two inspected formerTEST robot-contact regressions are development data; historical38/70 versus40/70 unchanged, no policy/generalization/promotion.',
            'Only requested dropper yaw changes; all five shot values and bouncer joints remain fixed. Different feasible dropper joint branches may change the rest of its arm posture.',
            'Fresh yaw proofs are reused across nominal repeats; finite IK refusals do not establish unreachability. No placeholder joints released.',
            'Three nominal requests per variant at64, sharing one admission proof per action; no repeated independent IK searches, noise,128 qualification, driven transit or hardware.'])


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('study','source','assets','parent-study','parent-audit','output'):p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args()
    if a.output.exists() or a.output.resolve().is_relative_to(a.study.resolve()):raise ValueError('Separate fresh audit output required')
    try:result=audit(a.study,a.source,a.assets,parent_study=a.parent_study,parent_audit=a.parent_audit)
    except Exception as error:
        a.output.write_text(json.dumps(dict(version=VERSION,status='failed',failures=[type(error).__name__+': '+str(error)]),indent=2)+'\n');raise
    a.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:result[k] for k in ('status','checks_count','requested_trials','native_releases','counts_by_variant')}))

if __name__=='__main__':main()
