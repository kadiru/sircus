"""Independent CPU audit of fixed-grid imitation and paired coverage.

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

VERSION='sircus-fixed35-imitation-audit-v1'
STUDY_VERSION='sircus-fixed35-imitation-v1'
METHODS=('fixed35_bc','current_bc')
ROLES=('direct_control','table_control','negative_control')
RESIDUAL=np.array([.04,.04,.05,8.,.12])
META=previous.META
AuditChecks=replay.AuditChecks
from . import corrected_imitation_verify as old
from . import gripper_yaw_verify as yaw
from . import yaw_tilt_gap_verify as targets_audit

PINS={
 'corrected':('1fde3f7de5b5c2912a133c858105388a704a80ab0e2cb39e8d79c0c7a189bf14','ea4992828178d9e62ba41c63579ecc3876cb5bffaacf6b590c828d45876a7f63','297b367fa2702d688084bc2e74dd30a83c6cf4312ba078938def5f9b223adce4',55633,158),
 'yaw':('39847d03be006b64af49da3ab1a6a8cc319883c7b1a8d81f37660e1d67f66a92','918f9d6d5cfa5cad93ba6baec2b5e4f46051c253c8459e7c09cf7396be732746','f0b63420c08265bb17e2a3e5cd40342ca0c492598f55d6e1772bf786e586737e',50615,106)}
CURRENT_SHA=old.CURRENT_SHA
CURRENT_WEIGHTS=old.CURRENT_WEIGHTS
FAR='corrected-imitation-test-r0-c6-n1'
NEAR='corrected-imitation-test-r1-c2-n0'
ACTION_LOW=np.asarray([.30,-.50,.12,8.,.05])
ACTION_HIGH=np.asarray([1.04,.35,.60,60.,.65])


def target_domain(values,bucket):
    values=np.asarray(values,float);bucket=np.asarray(bucket,float)
    return bool(np.linalg.norm(bucket-values[:2])>.15 and (values>=ACTION_LOW-1e-9).all()
                and (values<=ACTION_HIGH+1e-9).all())


def fixed_grid():
    """The original historical development grid, without new random sampling."""
    return [dict(id=f'fixed35-dev-grid-{iy:02d}-{ix:02d}',canonical_id=f'dev-grid-{iy:02d}-{ix:02d}',
                 split='test',evaluation_scope='historical_grid_coverage',bucket_xy_m=[float(x),float(y)],
                 grid_cell=[ix,iy],row=iy,column=ix)
            for iy,y in enumerate(np.linspace(-.65,.25,5))
            for ix,x in enumerate(np.linspace(.1,1.24,7))]


def protocol_checks(value,checks):
    expected=dict(substeps=64,methods=list(METHODS),test_layouts=35,test_requests=70,
        control_requests=18,maximum_requests=88,base_teacher_count=97,correction_teacher_count=2,
        teacher_count=99,actor_steps=800,critic_steps=0,actor_lr=.001,noise=False,
        residual_scale=RESIDUAL.tolist(),worker_limit_s=3600,queue_limit_s=900,global_limit_s=6300,
        audit_reserve_s=1800,storage_reserve_bytes=8*1024**3)
    for key,wanted in expected.items():checks.equal(value[key],wanted,'Literal fixed35 protocol/'+key)
    for key,wanted in dict(bucket_perturbations=False,grid_source='corrected_contract.grid_layouts',
            evaluation_scope='historical fixed-grid coverage/regression; not held out',
            rotation='Rz(atan2(bucket_y-board_y,bucket_x-board_x)) @ diag(1,-1,-1)').items():
        checks.equal(value[key],wanted,'Honest fixed-grid and controller semantics/'+key)


def authenticate_row(source,row,checks):
    checks.require(row in source['rows'],'Selected teacher is in the authenticated original ledger')
    namespace=row.get('adapter_directory','native')
    checks.require(namespace in ('native','native-yaw','native-controls'),'Known historical namespace')
    for prefix in ('connection','context_resolution'):
        checks.equal(sha(source['bound'](namespace+'/'+row[prefix+'_file'])),row[prefix+'_sha256'],row['id']+'/source '+prefix)
    checks.require(row.get('native_release') is True and native.captured(row),'New teacher is a genuine historical capture')
    checks.equal(sha(source['bound'](namespace+'/'+row['native_cache_file'])),row['native_cache_sha256'],row['id']+'/source dense trace')


def selected_yaw_groups(source,checks):
    selected=[]
    for context,phase,variant,index in ((FAR,'selected_repeats','selected',22),
                                       (NEAR,'references','near_success_reference',None)):
        group=sorted([r for r in source['rows'] if r.get('context_id')==context
            and r.get('phase')==phase and r.get('variant')==variant
            and (index is None or r.get('grid_index')==index)],key=lambda r:r['repeat'])
        checks.equal([r['repeat'] for r in group],[0,1,2],context+'/three prescribed yaw confirmations')
        checks.require(all(native.captured(r) for r in group),context+'/every selected nominal repeat captured')
        first=group[0]
        for row in group:
            authenticate_row(source,row,checks)
            checks.require(not row.get('invalid_physics') and not row.get('diagnostic_robot_contact_steps'),row['id']+'/valid contact-free confirmed teacher')
            checks.equal(row['adapter_directory'],'native-yaw',row['id']+'/confirmed bucket-facing source namespace')
            for field in (*META,'values','bucket_xy_m','q10','q14','connection_sha256'):
                checks.equal(row[field],first[field],row['id']+'/same nominal repeated setting/'+field)
            checks.numeric(row['requested_gripper_yaw_rad'],math.atan2(row['bucket_xy_m'][1]-row['values'][1],
                row['bucket_xy_m'][0]-row['values'][0]),row['id']+'/confirmed source faces bucket',1e-12)
        selected.append(group)
    checks.equal(selected[0][0]['values'][3:],[30.33068546648235,.09564961424222355],
                 'Prespecified minimal far gap correction, not all ten successful variants')
    return selected


def reconstruct_teachers(corrected,yaw_source,checks):
    original=read(corrected['bound']('bootstrap/teachers.json'))
    checks.equal((len(original),len({r['context_id'] for r in original})),(97,97),'Retain all97 distinct historical action targets')
    checks.require(all(native.captured(r) and r['role']=='train' for r in original),'Original audited captured historical supervision')
    additions=[]
    for name,group in zip(('far','near'),selected_yaw_groups(yaw_source,checks)):
        first=group[0];teacher=copy.deepcopy(first)
        obs=calibrated.expected_observation(first['bucket_xy_m'],first['effective_source_hint'])
        checks.numeric(first['observation'],obs,name+'/exact original geometry observation',0)
        residual=(np.asarray(first['values'])-np.asarray(first['effective_source_hint']['values']))/RESIDUAL
        checks.require(residual.shape==(5,) and np.isfinite(residual).all() and (np.abs(residual)<1).all(),name+'/teacher strictly inside actor support; never clipped')
        teacher.update(id='fixed35-yaw-teacher-'+name,role='train',original_role=first['role'],
            source_correction_role=first['role'],development_reclassified=True,normalized_residual=residual.tolist(),
            observation=np.asarray(obs).tolist(),learning_reward=oracle.reward(first),source_correction_record_id=first['id'],
            source_correction_record_sha256=digest(first),correction_repeat_ids=[r['id'] for r in group],
            correction_repeat_sha256={r['id']:digest(r) for r in group},supervision_provenance='confirmed bucket-facing native64 recipe')
        additions.append(teacher)
    teachers=sorted(copy.deepcopy(original)+additions,key=lambda r:r['context_id'])
    checks.equal((len(teachers),len({r['id'] for r in teachers}),len({r['context_id'] for r in teachers})),(99,99,99),
                 'Exactly99 unique layouts: each recipe contributes once, not once per repeat')
    checks.equal(len({tuple(r['bucket_xy_m']) for r in teachers}),99,'Equal contribution from99 distinct bucket coordinates, not merely renamed IDs')
    lineage=[dict(context_id=r['context_id'],teacher_id=r['id'],teacher_sha256=digest(r),
        correction_record_id=r['source_correction_record_id'],correction_record_sha256=r['source_correction_record_sha256'],
        confirmation_record_ids=r['correction_repeat_ids'],confirmation_record_sha256=r['correction_repeat_sha256']) for r in additions]
    return original,additions,teachers,lineage


def audit_grid(study,spec,teachers,checks):
    test=fixed_grid();splits=read(study/'splits.json')
    checks.equal(len(test),35,'Exactly35 canonical positions')
    checks.equal(len({tuple(r['bucket_xy_m']) for r in test}),35,'All35 canonical positions distinct')
    checks.equal(splits['test'],test,'Original7x5 fixed coordinates, no jitter or sampled replacement')
    checks.equal(splits['train'],[dict(id=t['context_id'],bucket_xy_m=t['bucket_xy_m'],split='train') for t in teachers],
                 'Exact teacher layout inventory')
    checks.equal(spec['splits'],splits,'Grid and training data frozen before outcomes')
    return dict(fixed_positions=35,columns=7,rows=5,perturbations=False,
                interpretation='Historical development grid coverage; not a fresh held-out or all-table generalization claim.')

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


def audit_inputs(study,sources,spec,checks):
    parent=sources['corrected'];data={}
    for key,name in (('prior','prior.json'),('old_prior','old-prior.json'),('controls','controls.json'),
            ('control_replays','control-replays.json'),('calibration','calibration.json'),('home','setup-home.json')):
        data[key]=read(parent['bound'](name));checks.equal(read(study/name),data[key],'Exact preserved input/'+name)
    checks.equal(sha(study/'verification.json'),sha(parent['bound']('verification.json')),'Original calibration verification unchanged')
    checks.equal((len(data['prior']['records']),len(data['old_prior']['records'])),(19,17),'Same19 geometry anchors and original17 control prior')
    checks.equal((spec['prior_sha256'],spec['old_prior_sha256']),(digest(data['prior']),digest(data['old_prior'])),'Immutable geometry-prior identities')
    original,additions,teachers,lineage=reconstruct_teachers(parent,sources['yaw'],checks)
    for name,wanted in (('bootstrap/teachers.json',teachers),('bootstrap/original-teachers.json',original),
                        ('bootstrap/correction-teachers.json',additions)):
        checks.equal(read(study/name),wanted,'Independent teacher reconstruction/'+name)
    checks.equal(spec['teacher_file_sha256'],sha(study/'bootstrap/teachers.json'),'All99 supervised rows frozen before fitting')
    boot=read(study/'bootstrap/source-bindings.json')
    for key,wanted in dict(parent_teacher_ids=[r['id'] for r in original],parent_teacher_sha256={r['id']:digest(r) for r in original},
            new_teacher_ids=[r['id'] for r in additions],teacher_file_sha256=sha(study/'bootstrap/teachers.json'),teacher_lineage=lineage).items():
        checks.equal(boot[key],wanted,'Exact teacher lineage/'+key)
    binding=read(study/'source-bindings.json');checks.equal(spec['source_bindings'],binding,'All source bindings frozen in contract')
    checks.equal(binding['teacher_lineage'],lineage,'Two literal new recipes with complete three-repeat evidence')
    checks.equal(binding['old_teacher_file_sha256'],sha(parent['bound']('bootstrap/teachers.json')),'Historical97-row file retained')
    checks.equal(binding['old_teacher_sha256'],{r['id']:digest(r) for r in original},'Historical97-row records retained byte-for-byte')
    checks.equal(binding['teacher_semantics'],spec['protocol']['teacher_semantics'],'Historical yaw labels do not become confirmed rotated captures')
    preserved=[dict(id=r['id'],sha256=digest(r),values=r['values'],bucket_xy_m=r['bucket_xy_m'])
               for r in sources['yaw']['rows'] if r.get('phase')=='selected_repeats' and r.get('repeat')==0]
    checks.equal(binding['preserved_yaw_recipes'],preserved,'All10 successful far settings retained; only one weights the far layout')
    checks.equal(len(preserved),10,'All10 original far recipes remain available')
    for kind,source in sources.items():
        saved=binding[kind];rs,aps,fp,count,_=PINS[kind]
        for key,wanted in dict(report_sha256=rs,audit_sha256=aps,contract_fingerprint=fp,
                              evidence_manifest_sha256=source['manifest_sha256'],checks_count=count).items():
            checks.equal(saved[key],wanted,kind+'/literal consumed source/'+key)
        for name,expected in saved['files_sha256'].items():checks.equal(sha(source['bound'](name)),expected,kind+'/every claimed consumed source/'+name)
        for field in ('scene_sha256','asset_sha256'):
            checks.equal(source['spec']['identity'][field],spec['identity'][field],kind+'/same native scene/'+field)
    names={'effective-options.json','native-compiled-geometry.json'}
    for row in data['control_replays']:
        for field in ('context_resolution','connection'):
            name=row[field+'_file'];names.add(name)
            checks.equal(sha(parent['bound']('control-parent/native/'+name)),row[field+'_sha256'],'Original control proof')
    mirror={name:sha(parent['bound']('control-parent/native/'+name)) for name in sorted(names)}
    checks.equal(read(study/'control-parent/mirror.json'),mirror,'Exact original control mirror inventory')
    for name,expected in mirror.items():checks.equal(sha(contained(study/'control-parent/native',name)),expected,'Untouched control mirror/'+name)
    return data,original,additions,teachers


def audit_models(study,report,spec,parent,teachers,checks):
    artifacts=report['artifacts'];checks.equal(set(artifacts),set(METHODS),'Exactly two frozen continuous imitation models')
    current=artifacts['current_bc'];original=parent['report']['artifacts']['current_bc']
    checks.equal(current['sha256'],CURRENT_SHA,'Exact preserved current policy checkpoint')
    checks.equal(current['weight_sha256'],CURRENT_WEIGHTS,'Exact preserved current policy weights')
    checks.equal(sha(parent['bound'](original['path'])),CURRENT_SHA,'Original file bytes retained')
    for key in ('sha256','weight_sha256','samples','seed'):
        checks.equal(current.get(key),original.get(key),'Original checkpoint identity/'+key)
    checks.equal(spec['old_artifacts']['current_bc'],current,'Current model bound before fitting')
    states={}
    for method,artifact in artifacts.items():
        checks.equal(artifact['path'],'checkpoints/'+method+'.pt','Literal model artifact path/'+method)
        path=contained(study,artifact['path']);checks.equal(sha(path),artifact['sha256'],'Frozen checkpoint bytes/'+method)
        states[method]=common.load_checkpoint(path,checks,method)
        for key in ('weight_sha256','samples','seed'):checks.equal(states[method].get(key),artifact.get(key),'Actual saved model metadata/'+method+'/'+key)
    checks.equal((artifacts['current_bc']['samples'],artifacts['fixed35_bc']['samples']),(92,99),'True historical and new training sample counts')
    fit=read(study/'fit.json');checks.equal(report['fit'],fit,'Report matches saved fitting evidence')
    checks.equal((fit['parent_teacher_count'],fit['new_teacher_count']),(97,2),'Exactly97 historical action targets and two confirmed yaw targets')
    checks.equal(fit['status'],'actor_updated','Actor fitting actually ran')
    checks.equal(states['fixed35_bc']['contract_fingerprint'],spec['fingerprint'],'Candidate weights bind the frozen protocol')
    before,after=states['current_bc']['policy'],states['fixed35_bc']['policy']
    result=previous.actor_finetune_replay(before,after,teachers,fit,checks)
    checks.equal((fit['before'],fit['after']),(common.weights_sha256(before),common.weights_sha256(after)),'Actual optimizer input and output hashes')
    critic=lambda state:common.weights_sha256({k.removeprefix('critic.'):v for k,v in state.items() if k.startswith('critic.')})
    checks.equal((fit['critic_before_sha256'],fit['critic_after_sha256']),(critic(before),critic(before)),'Critic unchanged')
    checks.equal(fit['critic_row_count'],99,'Unchanged critic diagnostic uses99 rows')
    checks.numeric(fit['initial_critic_loss'],fit['final_critic_loss'],'No critic optimization or PPO',0)
    return states,result


def expected_targets(scene,values,bucket,*,facing=True):
    """Independent scalar construction, including refused too-close proposals."""
    v=np.asarray(values,float);b=np.asarray(bucket,float)
    if v.shape!=(5,) or b.shape!=(2,) or not np.isfinite(v).all() or not np.isfinite(b).all():
        raise EvidenceError('Finite five-coordinate action and bucket required')
    angle=math.atan2(b[1]-v[1],b[0]-v[0]);tilt=math.radians(v[3]);mount=scene['board']['mount_yaw_rad']
    def rz(a):
        c,s=math.cos(a),math.sin(a);return np.array([[c,-s,0.],[s,c,0.],[0.,0.,1.]])
    c,s=math.cos(tilt),math.sin(tilt);rx=np.array([[1.,0.,0.],[0.,c,-s],[0.,s,c]])
    board=np.eye(4);board[:3,:3]=rz(angle+math.pi/2)@rx@rz(mount);board[:3,3]=v[:3]
    mounting=np.eye(4);mounting[:3,:3]=rz(mount);mounting[2,3]=scene['board']['tool_offset_m']
    base=np.diag([-1.,-1.,1.,1.]);base[0,3]=scene['arms']['base_spacing_m']
    release=v[:3]+[0.,0.,scene['ball']['radius_m']+v[4]]
    drop=np.eye(4);drop[:3,:3]=(rz(angle) if facing else np.eye(3))@np.diag([1.,-1.,-1.])
    drop[:3,3]=release+[0.,0.,.095]
    return dict(yaw_rad=angle if facing else 0.,dropper_flange_world=drop,
        board_flange_local=np.linalg.inv(base)@board@np.linalg.inv(mounting),
        board_world=board,board_normal=board[:3,2],board_normal_world=board[:3,2],release_center_m=release)


def audit_behavior(connection,scene,checks,label):
    from .calibrated_admission import CONNECTION_POLICY_VERSION
    targets=expected_targets(scene,connection['values'],connection['bucket_xy_m'])
    behavior=connection['release_behavior'];version='sircus-continuous-bucket-facing-release-v1'
    checks.equal(set(behavior),{'version','yaw_rad','rotation_matrix','target_dropper_flange_world','geometry_hint_only'},label+'/literal fixed-yaw behavior schema')
    checks.equal((behavior['version'],behavior['geometry_hint_only']),(version,True),label+'/deterministic geometry-only behavior')
    checks.numeric(behavior['yaw_rad'],targets['yaw_rad'],label+'/independent bucket-facing angle',1e-12)
    checks.numeric(behavior['rotation_matrix'],targets['dropper_flange_world'][:3,:3],label+'/independent full rotation',1e-12)
    eligible=target_domain(connection['values'],connection['bucket_xy_m'])
    if eligible:
        checks.numeric(behavior['target_dropper_flange_world'],targets['dropper_flange_world'],label+'/independent dropper flange target',1e-12)
    else:
        checks.equal(behavior['target_dropper_flange_world'],None,label+'/inadmissible action has no posed target')
        checks.equal(connection['passed'],False,label+'/inadmissible action cannot admit a drop')
    checks.equal(connection['connection_policy'],CONNECTION_POLICY_VERSION,label+'/original sampled setup policy')
    key=digest(dict(version=version,connection_policy=CONNECTION_POLICY_VERSION,
        resolution_sha256=connection['resolution_sha256'],values=connection['values'],
        bucket_xy_m=connection['bucket_xy_m'],release_behavior=behavior))
    checks.equal(connection['behavior_key'],key,label+'/cache key binds complete rotated target')
    checks.equal(connection['action_substituted'],False,label+'/five actor coordinates unchanged')
    return targets,key


def audit_geometry(study,rows,scene,assets,checks):
    from .learning_ik import read_chain
    from .corrected_geometry import StrictEndpointGate
    from .calibrated_admission import source_geometry
    chains=(read_chain(assets/'dropper.urdf'),read_chain(assets/'board_arm.urdf'))
    gate=StrictEndpointGate(scene,assets,.005);seen=set();measured=[]
    for row in rows:
        label=row['id'];is_test=row['phase']=='test';namespace=row['adapter_directory']
        checks.equal(namespace,'native-yaw' if is_test else 'native-controls',label+'/explicit oriented policy/control namespace')
        targets=expected_targets(scene,row['values'],row['bucket_xy_m'],facing=is_test)
        if is_test:
            checks.equal(row['release_behavior_version'],'sircus-continuous-bucket-facing-release-v1',label+'/both policies use same fixed behavior')
            path=contained(study/namespace,row['connection_file']);checks.equal(sha(path),row['connection_sha256'],label+'/immutable yaw proof')
            connection=read(path);_,key=audit_behavior(connection,scene,checks,label)
            checks.equal(row['connection_file'],'fixed-yaw-connections/'+key+'.json',label+'/fresh full-yaw cache namespace')
            if row['connection_sha256'] not in seen:
                endpoint=connection.get('endpoint_attempts')
                if endpoint is not None:
                    checks.require(len(endpoint['candidates'])<=6,label+'/bounded finite candidate pairs')
                    if endpoint.get('targets') is not None:
                        for field,wanted in dict(starts=16,iterations=180,normal_only=False,selection_uses_flight_labels=False).items():
                            checks.equal(endpoint[field],wanted,label+'/finite exact-pose IK/'+field)
                        for field in ('dropper_flange_world','board_flange_local','release_center_m'):
                            checks.numeric(endpoint['targets'][field],targets[field],label+'/independent IK target/'+field,1e-12)
                        resolution=read(contained(study/namespace,row['context_resolution_file']))
                        hint=source_geometry(resolution['effective_source_hint'])
                        if resolution.get('resolved'):hint.update({k:resolution['resolved'][k] for k in ('q10','q14','values','bucket_xy_m')})
                        checks.equal(endpoint['geometry_hint_sha256'],digest(hint),label+'/IK seed is geometry-only pre-policy anchor')
                        checks.require(len(endpoint['valid_ik_branches'])==2 and all(type(x) is int and 0<=x<=17 for x in endpoint['valid_ik_branches']),label+'/bounded recorded IK branches')
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
        if is_test:checks.require(target_domain(row['values'],row['bucket_xy_m']),label+'/released action within original bounds and distance greater than150mm')
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


def audit_native(study,rows,data,spec,calibration,assets,checks,parent):
    setup={};count=0
    for namespace,prior_key,parent_namespace in (('native-yaw','prior','native'),('native-controls','old_prior','control-parent/native')):
        members=[r for r in rows if r['adapter_directory']==namespace];path=study/namespace
        checks.require(bool(members),namespace+'/complete requested cohort')
        setup[namespace]=calibrated.check_setup_evidence(geometry._NativeNamespace(path),members,data[prior_key],data['home'],checks)
        admitted=[r for r in members if r.get('native_release')]
        if admitted:
            models.audit_native_options(path,spec['scene'],calibration,assets,checks)
            for row in admitted:
                checks.equal((row['effective_options_file'],row['effective_options_sha256']),('effective-options.json',sha(path/'effective-options.json')),row['id']+'/exact native64 options')
            for name in ('effective-options.json','native-compiled-geometry.json'):
                checks.equal(read(path/name),read(parent['bound'](parent_namespace+'/'+name)),namespace+'/unchanged physics/'+name)
        else:checks.require(not list(path.rglob('*.npz')),namespace+'/no invented native cache')
        count+=len(native.rescore_rows(path,members,spec['scene'],calibration,spec['identity']['trace_schema'],assets,checks))
    return setup,count




def audit_test(study,rows,report,spec,states,checks):
    packets=[];comparison=[];positions=[]
    for method in METHODS:
        packet=read(study/f'predictions/test-{method}.json');packets.append(packet);artifact=report['artifacts'][method]
        group=[r for r in rows if r['phase']=='test' and r['method']==method]
        checks.equal((packet['method'],packet['role'],packet['artifact']),(method,'test',artifact),'Frozen method packet/'+method)
        checks.equal((packet['one_prediction_per_layout'],packet['selection_or_learning']),(True,False),'No adaptive TEST selection/'+method)
        checks.equal((len(packet['rows']),len(group)),(35,35),'All35 requested outcomes per frozen policy/'+method)
        for prediction,row,context in zip(packet['rows'],group,spec['splits']['test']):
            label=row['id'];checks.equal(label,f'fixed35-imitation-test-{method}-{context["id"]}',label+'/literal test record ID')
            checks.equal((prediction['context_id'],prediction['bucket_xy_m']),(context['id'],context['bucket_xy_m']),label+'/unchanged fixed-grid layout')
            for key,value in prediction.items():checks.equal(row.get(key),value,label+'/unmodified frozen prediction/'+key)
            checks.equal((prediction['artifact_sha256'],row['frozen_artifact_sha256']),(artifact['sha256'],)*2,label+'/frozen checkpoint binding')
            checks.equal((prediction['source_training_ids'],prediction['ball_flight_search_at_inference'],prediction['geometry_search_at_inference'],prediction['selection_or_learning']),([],False,True,False),label+'/geometry-only anchor resolution and one actor inference')
            checks.numeric(prediction['observation'],calibrated.expected_observation(context['bucket_xy_m'],prediction['effective_source_hint']),label+'/independent geometry observation',0)
            previous.audit_actor_transform(prediction,states[method],checks,label)
            checks.equal((row['values'],row['perturbation_mixed_units']),(row['command_values'],[0.]*5),label+'/exact unmodified noiseless command')
            checks.equal(row['test_plan_sha256'],sha(study/'test-plan.json'),label+'/all70 predictions bound before TEST')
            positions.append(dict(context_id=row['context_id'],bucket_xy_m=row['bucket_xy_m'],method=method,capture=native.captured(row),clean=native.captured(row) and row.get('route_quality')==2,native_release=row['native_release'],invalid=row.get('invalid'),invalid_physics=bool(row.get('invalid_physics')),direct_capture=row.get('direct_capture',False),table_assisted_capture=row.get('table_assisted_capture',False),forbidden_contact_steps=row.get('forbidden_contact_steps',0),values=row['values'],record_id=row['id']))
        comparison.append(dict(method=method,counts=native.statistics(group),artifact_sha256=artifact['sha256']))
    checks.equal(read(study/'test-plan.json'),dict(methods=packets,frozen_before_outcomes=True),'Both full inference packets frozen together')
    checks.equal(report['test_comparison'],comparison,'Independent paired denominator comparison')
    checks.equal(report['test_positions'],positions,'Every fixed-grid position result reproduced')
    by={r['context_id']:{m:next(p for p in positions if p['context_id']==r['context_id'] and p['method']==m) for m in METHODS} for r in positions}
    gains=sum(v['fixed35_bc']['capture'] and not v['current_bc']['capture'] for v in by.values())
    losses=sum(v['current_bc']['capture'] and not v['fixed35_bc']['capture'] for v in by.values())
    paired=dict(paired_positions=35,both_capture=sum(all(p['capture'] for p in v.values()) for v in by.values()),
        fixed35_only=gains,current_only=losses,both_fail=sum(not any(p['capture'] for p in v.values()) for v in by.values()),net_captures=gains-losses)
    checks.equal(report['paired_comparison'],paired,'Every paired gain and regression independently counted')
    for i in range(35):
        for key in META:checks.equal(packets[0]['rows'][i][key],packets[1]['rows'][i][key],str(i)+'/shared geometry-only setup/'+key)
    return comparison




def audit_controls(rows,sources,originals,checks):
    gates={}
    for phase,key in (('controls_pre','pre'),('controls_post','post')):
        group=[r for r in rows if r['phase']==phase];checks.equal(len(group),9,phase+'/nine prescribed controls')
        for i,(source,original,role) in enumerate(zip(sources,originals,ROLES)):
            for repeat in range(3):
                row=group[3*i+repeat];label=row['id']
                checks.equal(label,f'fixed35-imitation-{phase}-{role}-r{repeat}',label+'/literal control identity')
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
    expected += [('test','test',m) for m in METHODS for _ in range(35)]
    expected += [('controls_post',role,'control') for role in ROLES for _ in range(3)]
    checks.equal([(r['phase'],r['role'],r['method']) for r in rows],expected,'Exactly88 requests with no search, retries or omitted rejection')
    checks.equal(len({r['id'] for r in rows}),88,'Unique request IDs')
    checks.require(0<=report['elapsed_s']<3600,'One-hour acquired worker bound')
    boundaries=common.lines(study/'phase-boundaries.jsonl')
    checks.equal([(b['phase'],b['rows_count']) for b in boundaries],[('inputs-frozen',0),('controls_pre-complete',9),('fit-complete',9),('test-plan-frozen',9),('test-complete',79),('controls_post-complete',88)],'Fit and both prediction freezes precede any TEST outcome')
    raw=(study/'rows.jsonl').read_bytes().splitlines(keepends=True);last=0
    for b in boundaries:
        checks.equal(b['rows_sha256'],hashlib.sha256(b''.join(raw[:b['rows_count']])).hexdigest() if b['rows_count'] else None,b['phase']+'/immutable observed prefix')
        checks.require(last<=b['acquired_elapsed_s']<3600,'Ordered bounded phase times');last=b['acquired_elapsed_s']
    timings=common.lines(study/'timing.jsonl');checks.equal(len(timings),6,'Six exact bounded execution batches')
    checks.require(boundaries[3]['acquired_elapsed_s']<=next(t['acquired_start_s'] for t in timings if t['phase']=='test'),'Both full35 prediction packets precede first TEST simulation')
    checks.require(all(t['acquired_end_s']<3600 for t in timings),'Every native batch within one-hour limit')
    return dict(total_requests=88,test_requests=70,control_requests=18,maximum_requests=88)




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


def audit(study,source,assets,*,corrected_study,corrected_audit,yaw_study,yaw_audit,reference):
    started=time.monotonic();checks=AuditChecks();study,source,assets=map(Path,(study,source,assets))
    manifest=replay.check_manifest(study,checks);spec=read(study/'contract.json');report=read(study/'report.json')
    unsigned=dict(spec);fp=unsigned.pop('fingerprint')
    checks.equal((digest(unsigned),report['contract_fingerprint']),(fp,fp),'Immutable full experiment fingerprint')
    checks.equal((spec['version'],report['version'],report['status'],report['phase']),
                 (STUDY_VERSION,STUDY_VERSION,'completed','completed'),'Full fixed35 comparison completed')
    checks.equal(report['evidence_sha256'],manifest,'Report binds every saved artifact')
    checks.equal(report['protocol'],spec['protocol'],'Prespecified protocol unchanged')
    protocol_checks(spec['protocol'],checks)
    checks.equal(spec['release_behavior'],spec['protocol']['rotation'],'Fixed yaw behavior bound before fitting')
    checks.equal((spec['evaluation_scope'],report['evaluation_scope']),
        ('historical fixed-grid coverage/regression; not held out',)*2,'No fresh held-out claim for historical grid')
    replay.check_source_files(source,spec['source_sha256'],checks,'Frozen executed source')
    for module in (Path(__file__),Path(oracle.__file__)):
        checks.equal(sha(module),spec['source_sha256']['sircus/mjlab/'+module.name],'Actual independent verifier frozen/'+module.name)
    sources={kind:authenticate(directory,proof,kind,checks) for kind,directory,proof in
        (('corrected',corrected_study,corrected_audit),('yaw',yaw_study,yaw_audit))}
    data,original,additions,teachers=audit_inputs(study,sources,spec,checks)
    models.audit_runtime_reference(source,reference,spec,sources['corrected']['spec'],assets,checks)
    inventory=audit_grid(study,spec,teachers,checks)
    calibration=replay.check_calibration(study,spec['identity']['calibration'],checks)
    rows=replay.read_records(study/'rows.jsonl');native.check_rows(rows,checks)
    setup,native_count=audit_native(study,rows,data,spec,calibration,assets,checks,sources['corrected'])
    geometry_result=audit_geometry(study,rows,spec['scene'],assets,checks)
    accounting=audit_sequence(study,rows,report,checks)
    packets=geometry.audit_request_packets(study,rows,checks);budget=audit_budget(study,report,checks)
    states,fit=audit_models(study,report,spec,sources['corrected'],teachers,checks)
    comparison=audit_test(study,rows,report,spec,states,checks)
    gates=audit_controls(rows,data['controls'],data['control_replays'],checks)
    checks.equal(report['control_gates'],gates,'Both independent control gates passed')
    checks.equal(report['counts'],native.statistics(rows),'Every requested outcome counted, including refusals')
    checks.equal(report['stage_counts'],{p:native.statistics([r for r in rows if r['phase']==p])
        for p in ('controls_pre','test','controls_post')},'Exact per-phase request accounting')
    checks.equal((report['base_teachers'],report['new_teachers'],report['teacher_count'],report['expected_requests']),
        (97,2,99,88),'True teacher and request counts')
    corrections=[dict(context_id=r['context_id'],bucket_xy_m=r['bucket_xy_m'],values=r['values'],source_study='yaw',
        source_record_id=r['source_correction_record_id'],source_record_sha256=r['source_correction_record_sha256'],
        source_report_sha256=PINS['yaw'][0],confirmation_captures=3,confirmation_requests=3) for r in additions]
    checks.equal(report['teacher_corrections'],corrections,'Two added yaw recipes with measured success provenance')
    checks.equal(report['gate'],dict(status='passed',substeps=64,controls_passed=True,complete_test=True,no_policy_promotion=True),
        'Complete finite comparison; no automatic policy activation')
    return dict(version=VERSION,status='passed',full_checks_verified=True,native_checks_verified=True,
        checks_count=checks.count,checks=checks.entries,failures=[],contract_fingerprint=fp,
        report_sha256=sha(study/'report.json'),evidence_manifest_sha256=sha(study/'evidence-manifest.json'),
        auditor_sha256=sha(Path(__file__)),producer_source_sha256=spec['source_sha256'],
        consumed_source_files_sha256={k:s['consumed'] for k,s in sources.items()},requested_trials=len(rows),
        native_releases=native_count,teacher_count=99,correction_teacher_count=2,actor_fit=fit,
        test_comparison=comparison,control_gates=gates,setup_evidence=setup,yaw_geometry=geometry_result,
        coordinate_inventory=inventory,request_accounting=accounting,request_packets=packets,budget=budget,
        elapsed_s=time.monotonic()-started,maximum_numeric_error=checks.numeric_error,
        limitations=['Original35 development grid, not fresh held-out positions; no bucket perturbations.',
            'Historical97 teacher actions retain original yaw outcome provenance; only two new yaw targets confirmed3/3.',
            'Continuous residual actor conditioned on19 geometry anchors, not a bank-free controller or discrete lookup action selector.',
            'Both policies use fixed bucket-facing behavior; this is not an orientation A/B or the old40/70 benchmark.',
            'Held endpoints at64 substeps, one prediction per policy/layout; no noise,128, learned transit, hardware or policy promotion.'])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('study','source','assets','corrected-study','corrected-audit','yaw-study','yaw-audit','reference','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    a=parser.parse_args()
    if a.output.exists() or a.output.resolve().is_relative_to(a.study.resolve()):
        raise ValueError('Fresh independent proof outside immutable study required')
    try:
        args=vars(a).copy();args.pop('output');result=audit(**args)
    except Exception as error:
        result=dict(version=VERSION,status='failed',failures=[type(error).__name__+': '+str(error)])
        a.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');raise
    a.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:result[k] for k in ('status','checks_count','requested_trials','native_releases','teacher_count','test_comparison')}))


if __name__=='__main__':main()
