"""Independent CPU audit of the fixed-yaw far-shot tilt/drop-gap search.

No producer planning, candidate selection, summary or scoring code is imported.
The original native trace and setup oracles retain their existing strict checks.
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
from . import gripper_yaw_verify as yaw
from . import replay650_verify as replay
from . import table_training_verify as native
from . import coverage_training_verify as geometry
from . import table_assisted_score_oracle as oracle
from .benchmark_verify import EvidenceError, contained, digest, read, sha

VERSION='sircus-yaw-tilt-gap-audit-v1'
STUDY_VERSION='sircus-yaw-tilt-gap-v1'
FAR='corrected-imitation-test-r0-c6-n1'
NEAR='corrected-imitation-test-r1-c2-n0'
TILTS=(24.,26.,28.,30.33068546648235,32.,34.,36.)
GAP_OFFSETS=(0.,.02,.04,.06,.09,.12,.16)
SOURCE_GAP=.07564961424222355
YAW_REPORT='2cd3644a19653efc157fe697557ebc861e40ab6b97ceeb3faeb48aaf04b923fa'
YAW_AUDIT='4f044f4c3b4dc3ffd79dcb65f42214ceb3f597d129afb0c79728697137b9ec21'
YAW_FP='e31c8db6e8b9a2b16f7213b3440396cccf83004e64988575f694b2b007cbf683'
META=yaw.META
ROLES=yaw.ROLES


def statistics(rows):
    result=native.statistics(rows)
    result['robot_contact_shots']=sum(r.get('diagnostic_robot_contact_steps',0)>0 for r in rows)
    result['cached_refusal_requests']=sum(r.get('admission_mode')=='cached_parent_refusal' for r in rows)
    return result


def expected_targets(scene,values,bucket,*,bucket_facing=True):
    """Independent matrix construction of both flange and release targets."""
    values=np.asarray(values,float);bucket=np.asarray(bucket,float)
    target=yaw.expected_yaw(values,bucket,scene['ball']['radius_m'])
    angle=target['yaw_rad'];tilt=math.radians(values[3]);mount=scene['board']['mount_yaw_rad']
    def rz(a):
        c,s=math.cos(a),math.sin(a);return np.asarray([[c,-s,0.],[s,c,0.],[0.,0.,1.]])
    c,s=math.cos(tilt),math.sin(tilt);rx=np.asarray([[1.,0.,0.],[0.,c,-s],[0.,s,c]])
    board=np.eye(4);board[:3,:3]=rz(angle+math.pi/2)@rx@rz(mount);board[:3,3]=values[:3]
    mounting=np.eye(4);mounting[:3,:3]=rz(mount);mounting[2,3]=scene['board']['tool_offset_m']
    base=np.diag([-1.,-1.,1.,1.]);base[0,3]=scene['arms']['base_spacing_m']
    drop=np.asarray(target['dropper_flange_world']);rotation=drop[:3,:3].copy()
    if not bucket_facing:drop[:3,:3]=np.diag([1.,-1.,-1.]);rotation=drop[:3,:3].copy()
    release=values[:3]+[0.,0.,scene['ball']['radius_m']+values[4]]
    return dict(yaw_rad=angle if bucket_facing else 0.,rotation_matrix=rotation,
        dropper_flange_world=drop,board_flange_local=np.linalg.inv(base)@board@np.linalg.inv(mounting),
        board_world=board,board_normal=board[:3,2],board_normal_world=board[:3,2],release_center_m=release)


def authenticate_yaw(study,audit,checks):
    study,audit=Path(study),Path(audit)
    checks.equal(sha(study/'report.json'),YAW_REPORT,'Exact audited yaw study report')
    checks.equal(sha(audit),YAW_AUDIT,'Exact audited yaw proof')
    report,proof,spec,manifest=(read(p) for p in (study/'report.json',audit,study/'contract.json',study/'evidence-manifest.json'))
    unsigned=dict(spec);fp=unsigned.pop('fingerprint')
    checks.equal((digest(unsigned),fp,report['contract_fingerprint'],proof['contract_fingerprint']),(YAW_FP,)*4,'Complete immutable yaw contract')
    checks.equal((report['status'],report['phase'],proof['status'],proof['full_checks_verified'],proof['native_checks_verified'],proof['failures']),
                 ('completed','completed','passed',True,True,[]),'Yaw source fully audited and complete')
    checks.equal((proof['checks_count'],len(proof['checks'])),(18330,18330),'All original yaw checks retained')
    checks.require(all(c['passed'] is True for c in proof['checks']),'Every original yaw check passed')
    checks.equal(proof['report_sha256'],YAW_REPORT,'Yaw audit binds report')
    checks.equal(proof['evidence_manifest_sha256'],sha(study/'evidence-manifest.json'),'Yaw audit binds evidence manifest')
    checks.equal(report['evidence_sha256'],manifest,'Yaw report/manifest agree')
    rows=replay.read_records(study/'rows.jsonl')
    checks.equal((len(rows),len({r['id'] for r in rows})),(36,36),'Original yaw complete request ledger')
    selected=[]
    for context in (FAR,NEAR):
        group=[r for r in rows if r['context_id']==context and r['variant']=='bucket_facing' and r['repeat']==0]
        checks.equal(len(group),1,'Unique original first yaw repeat/'+context)
        selected.append(group[0])
    checks.require(not native.captured(selected[0]) and native.captured(selected[1]),'Preserved far miss and near capture source')
    checks.require(all(r['native_release'] and r['diagnostic_robot_contact_steps']==0 and not r.get('invalid_physics') for r in selected),
                   'Both yaw source shots clear the robot and have valid physics')
    files={}
    for name in ('contract.json','rows.jsonl'):
        checks.equal(sha(study/name),manifest[name],'Original yaw file/'+name);files[name]=manifest[name]
    for row in selected:
        for key,hkey in (('connection_file','connection_sha256'),('context_resolution_file','context_resolution_sha256'),('native_cache_file','native_cache_sha256')):
            name=row['adapter_directory']+'/'+row[key]
            checks.equal(sha(contained(study,name)),manifest[name],'Original yaw source bytes/'+name)
            checks.equal(manifest[name],row[hkey],'Original yaw row binding/'+name);files[name]=manifest[name]
    for name in ('effective-options.json','native-compiled-geometry.json'):
        path='native-yaw/'+name;checks.equal(sha(study/path),manifest[path],'Original yaw runtime/'+name);files[path]=manifest[path]
    return selected,spec,dict(report_sha256=YAW_REPORT,audit_sha256=YAW_AUDIT,contract_fingerprint=YAW_FP,
        manifest_sha256=sha(study/'evidence-manifest.json'),files_sha256=files)


def selected_captures(rows):
    return sorted([r for r in rows if r.get('phase')=='discovery' and r.get('context_id')==FAR
        and r.get('native_release') is True and r.get('invalid_physics') is False and native.captured(r)],
        key=lambda r:int(r['grid_index']))


def audit_actual_poses(row,scene,chain10,chain14,checks):
    """Validate measured reset joints against independent world targets."""
    label=row['id'];facing=row['adapter_directory']!='native-controls' and row['variant']!='far_success_reference'
    targets=expected_targets(scene,row['values'],row['bucket_xy_m'],bucket_facing=facing)
    checks.numeric(row['requested_gripper_yaw_rad'],targets['yaw_rad'],label+'/literal bucket direction retained',1e-12)
    if not row.get('native_release'):
        checks.require(not row.get('native_initial_q10') and not row.get('native_initial_q14') and not row.get('native_cache_file'),label+'/decline has no fabricated flight or reset')
        return None
    for arm in ('q10','q14'):
        checks.equal(row['native_initial_'+arm],np.asarray(row[arm],np.float32).astype(float).tolist(),label+'/observed float32 reset/'+arm)
    drop=yaw.forward_chain(chain10,row['native_initial_q10']);board=yaw.forward_chain(chain14,row['native_initial_q14'])
    for arm,actual,expected in (('dropper',drop,targets['dropper_flange_world']),('board',board,targets['board_flange_local'])):
        error=np.linalg.norm(actual[:3,3]-expected[:3,3]);rot=yaw.rotation_error(actual[:3,:3],expected[:3,:3])
        checks.require(error<1e-4,label+'/actual '+arm+' position within original 0.1mm tolerance')
        checks.require(rot<4e-4,label+'/actual '+arm+' full orientation within original tolerance')
    checks.numeric(row['release_ball_center_m'],targets['release_center_m'],label+'/ball release center matches fixed board and selected gap',1e-12)
    checks.numeric(row['board_center_m'],row['values'][:3],label+'/actual fixed board center',1e-4)
    checks.numeric(row['board_normal'],targets['board_normal'],label+'/actual requested tilt normal',4e-4)
    return dict(id=label,variant=row['variant'],requested_yaw_rad=targets['yaw_rad'],
        measured_yaw_rad=math.atan2(drop[1,0],drop[0,0]),rotation_error_rad=yaw.rotation_error(drop[:3,:3],targets['rotation_matrix']))


def literal_rule():
    return dict(version=STUDY_VERSION,context=FAR,tilts_deg=list(TILTS),gap_offsets_m=list(GAP_OFFSETS),
        rotation='Rz(atan2(bucket_y-board_y,bucket_x-board_x)) @ diag(1,-1,-1)',
        fixed=['bucket_xy_m','board_x_m','board_y_m','board_height_m','gripper_yaw'],
        changed=['tilt_deg','drop_gap_m'],selection='all valid discovery captures in grid-index order, then three fresh nominal repeats',
        substeps=64,noise=False,training=False)


def protocol():
    return dict(**literal_rule(),maximum_requests=223,candidate_requests=49,fixed_reference_requests=9,control_requests=18,
        maximum_selected_repeat_requests=147,repeats=3,maximum_acquired_s=3600,maximum_queue_s=900,maximum_pipeline_s=6300,
        audit_reserve_s=1800,storage_reserve_bytes=8*1024**3,posthoc_development=True,outcome_adaptive_grid=False,
        selection_frozen_before_repeats=True,branch_selection='geometry only; nearest original yaw joints then residual and branch IDs',
        orientation_measurement='model FK of observed initial float32 native joints; held endpoints')


def independent_plan(sources,controls,yaw_rows):
    original={(r['context_id'],r['method']):r for r in sources};rotated={r['context_id']:r for r in yaw_rows}
    far=original[FAR,'corrected_bc'];near=original[NEAR,'corrected_bc'];current=original[FAR,'current_bc'];expected=[]
    def add(source,variant,phase,repeat,values=None,index=None):
        values=copy.deepcopy(source['values'] if values is None else values)
        facing=variant not in ('far_success_reference','control_exact');previous=rotated[source['context_id']] if facing else None
        request={k:copy.deepcopy(source[k]) for k in META}
        request.update(id=f'yaw-tilt-gap-{phase}-{variant}-'+(f'g{index:02d}' if index is not None else source['context_id'])+f'-r{repeat}',
            phase=phase,role=source['role'] if variant=='control_exact' else 'diagnostic',method=variant,variant=variant,repeat=repeat,
            context_id=source['context_id'],bucket_xy_m=source['bucket_xy_m'],values=values,command_values=copy.deepcopy(values),
            source_record_id=source['id'],source_record_sha256=digest(source),
            adapter_directory='native-controls' if variant=='control_exact' else 'native-yaw' if facing else 'native',
            requested_gripper_yaw_rad=math.atan2(source['bucket_xy_m'][1]-values[1],source['bucket_xy_m'][0]-values[0]) if facing else 0.,
            frozen_yaw_record_id=previous['id'] if facing else None,frozen_yaw_record_sha256=digest(previous) if facing else None,
            parameter_indices=[3,4] if variant=='grid' else [],grid_index=index,substeps=64,perturbation_mixed_units=[0.]*5,
            selection_or_learning=False,posthoc_development=True,admission_mode='yaw_tilt_gap_strict_proof' if variant=='grid' else 'exact_parent_branch')
        expected.append(request)
    for row in controls:
        for repeat in range(3):add(row,'control_exact','controls_pre',repeat)
    for source,variant in ((far,'far_rotated'),(current,'far_success_reference'),(near,'near_success_reference')):
        for repeat in range(3):add(source,variant,'references',repeat)
    for index,(tilt,offset) in enumerate((t,g) for t in TILTS for g in GAP_OFFSETS):
        add(far,'grid','discovery',0,far['values'][:3]+[tilt,far['values'][4]+offset],index)
    for row in controls:
        for repeat in range(3):add(row,'control_exact','controls_post',repeat)
    return expected


def audit_plan(plan,sources,controls,yaw_rows,rule,checks):
    checks.equal(rule,literal_rule(),'Literal49-candidate grid with fixed yaw')
    checks.equal([r['role'] for r in controls],list(ROLES),'Original ordered direct/table/miss controls')
    far=next(r for r in sources if r['context_id']==FAR and r['method']=='corrected_bc')
    checks.equal(far['values'][4],SOURCE_GAP,'Exact parent drop-gap origin')
    checks.equal(far['values'][3],TILTS[3],'Exact original tilt included in grid')
    wanted=independent_plan(sources,controls,yaw_rows)
    checks.equal((len(plan),len({r['id'] for r in plan})),(76,76),'Complete unique76 base requests')
    for row,expected in zip(plan,wanted):
        checks.numeric(row['requested_gripper_yaw_rad'],expected['requested_gripper_yaw_rad'],row['id']+'/independent bucket-facing angle',1e-12)
        expected['requested_gripper_yaw_rad']=row['requested_gripper_yaw_rad']
        checks.equal(row,expected,row['id']+'/independent complete frozen request')


def expected_selection(rows):
    result=[]
    for row in selected_captures(rows):
        result.append(dict(candidate_id=row['id'],grid_index=row['grid_index'],row_sha256=digest(row),
            **{k:copy.deepcopy(row[k]) for k in ('values','bucket_xy_m','connection_file','connection_sha256','q10','q14')}))
    return result


def expected_repeat_plan(plan,selection):
    lookup={r['id']:r for r in plan};result=[]
    for selected in selection:
        for repeat in range(3):
            row=copy.deepcopy(lookup[selected['candidate_id']]);row.update(id=f'yaw-tilt-gap-selected-g{selected["grid_index"]:02d}-r{repeat}',
                phase='selected_repeats',variant='selected',method='selected',repeat=repeat,
                selected_from_id=selected['candidate_id'],selected_from_sha256=selected['row_sha256'])
            result.append(row)
    return result


def audit_observations(rows,plan,sources,controls,yaw_rows,selection,selected_plan,checks):
    wanted_selection=expected_selection(rows)
    checks.equal(selection,wanted_selection,'Every valid discovery capture selected, no cherry-picked subset')
    checks.equal(selected_plan,expected_repeat_plan(plan,selection),'All selected candidates get exactly three fresh nominal requests')
    execution=[r for r in plan if r['phase']!='controls_post']+selected_plan+[r for r in plan if r['phase']=='controls_post']
    checks.equal((len(rows),len({r['id'] for r in rows})),(76+3*len(selection),76+3*len(selection)),'Complete requested denominator includes declines')
    checks.require(len(rows)<=223,'Frozen223 maximum respected')
    originals={r['id']:r for r in sources+controls};yaw_sources={r['id']:r for r in yaw_rows};discovered={r['id']:r for r in rows if r['phase']=='discovery'}
    for row,request in zip(rows,execution):
        label=row['id'];source=originals[request['source_record_id']]
        for key,value in request.items():checks.equal(row.get(key),value,label+'/frozen request field/'+key)
        checks.equal(row['values'][:3],source['values'][:3],label+'/board center request preserved')
        checks.equal(row['bucket_xy_m'],source['bucket_xy_m'],label+'/bucket preserved')
        if row['variant']=='selected':
            previous=discovered[row['selected_from_id']]
            for key in ('connection_file','connection_sha256','native_release','q10','q14'):
                checks.equal(row.get(key),previous.get(key),label+'/selected repeat retains exact admitted branch/'+key)
        elif row['phase']!='discovery':
            previous=yaw_sources[row['frozen_yaw_record_id']] if row['frozen_yaw_record_id'] else source
            for key in ('connection_file','connection_sha256','native_release','q10','q14'):
                checks.equal(row.get(key),previous.get(key),label+'/reference recipe/branch unchanged/'+key)
        elif row['grid_index']==21:
            previous=yaw_sources[row['frozen_yaw_record_id']]
            checks.equal(row['values'],previous['values'],label+'/exact zero-offset grid point')
            for key in ('connection_file','connection_sha256','native_release','q10','q14'):
                checks.equal(row.get(key),previous.get(key),label+'/zero offset preserves proven yaw branch/'+key)
        else:
            checks.require(row['connection_file'].startswith('yaw-tilt-gap-connections/'),label+'/new full-target proof namespace')
            checks.require(row['connection_sha256']!=source['connection_sha256'],label+'/no original five-value proof reuse')
        if row.get('realization_comparison') is not None:
            checks.equal(row['realization_comparison'],yaw.realized_comparison(row,source,checks),label+'/reported actual branch differences')


def audit_new_connection(connection,source,yaw_source,scene,checks,label):
    from .calibrated_admission import CONNECTION_POLICY_VERSION
    values=connection['values'];targets=expected_targets(scene,values,source['bucket_xy_m'])
    extension=connection['yaw_intervention']
    wanted=dict(version='sircus-fixed-yaw-tilt-gap-geometry-v1',source_record_id=source['id'],source_record_sha256=digest(source),
        frozen_yaw_record_id=yaw_source['id'],frozen_yaw_record_sha256=digest(yaw_source),source_q10=yaw_source['q10'],source_q14=yaw_source['q14'],
        requested_yaw_rad=targets['yaw_rad'],requested_rotation_matrix=targets['rotation_matrix'].tolist(),
        requested_dropper_flange_world=targets['dropper_flange_world'].tolist(),requested_board_flange_local=targets['board_flange_local'].tolist(),
        release_center_m=targets['release_center_m'].tolist(),board_branch_fixed=False,fixed_board_center_m=source['values'][:3],
        source_values=source['values'],requested_values=values,changed_parameter_indices=[3,4],branch_selection_uses_flight_outcomes=False)
    checks.equal(set(extension),set(wanted),label+'/literal full-target extension schema')
    numerical={'requested_yaw_rad','requested_rotation_matrix','requested_dropper_flange_world','requested_board_flange_local','release_center_m'}
    for key,value in wanted.items():
        if key in numerical:checks.numeric(extension[key],value,label+'/independent extension/'+key,1e-12)
        else:checks.equal(extension[key],value,label+'/immutable extension/'+key)
    checks.equal(values[:3],source['values'][:3],label+'/fixed requested board center')
    checks.equal(connection['bucket_xy_m'],source['bucket_xy_m'],label+'/fixed bucket')
    checks.equal(connection['connection_policy'],CONNECTION_POLICY_VERSION,label+'/original setup path policy')
    key=digest(dict(version=wanted['version'],connection_policy=CONNECTION_POLICY_VERSION,resolution_sha256=connection['resolution_sha256'],
        values=values,bucket_xy_m=source['bucket_xy_m'],yaw_intervention=extension))
    checks.equal(connection['yaw_key'],key,label+'/proof key binds full orientation/source/values')
    return targets,key


def audit_geometry(study,rows,sources,controls,yaw_rows,scene,assets,checks):
    from .learning_ik import read_chain
    from .corrected_geometry import StrictEndpointGate
    chain10=read_chain(Path(assets)/'dropper.urdf');chain14=read_chain(Path(assets)/'board_arm.urdf');gate=StrictEndpointGate(scene,assets,.005)
    originals={r['id']:r for r in sources+controls};previous={r['id']:r for r in yaw_rows};seen=set();measurements=[]
    for row in rows:
        label=row['id'];source=originals[row['source_record_id']]
        measured=audit_actual_poses(row,scene,chain10,chain14,checks)
        if measured is not None:measurements.append(measured)
        extension_name=row['connection_file'];path=contained(study/row['adapter_directory'],extension_name)
        if extension_name.startswith('yaw-tilt-gap-connections/') and row['connection_sha256'] not in seen:
            connection=read(path);checks.equal(sha(path),row['connection_sha256'],label+'/new connection immutable bytes')
            old=previous[row['frozen_yaw_record_id']];targets,key=audit_new_connection(connection,source,old,scene,checks,label)
            checks.equal(extension_name,'yaw-tilt-gap-connections/'+key+'.json',label+'/new full-target proof namespace')
            endpoint=connection.get('endpoint_attempts')
            if endpoint is not None:
                for key,value in dict(starts=16,iterations=180,normal_only=False,flight_labels_used=False).items():
                    checks.equal(endpoint[key],value,label+'/fixed finite independent IK/'+key)
                for name in ('dropper_flange_world','board_flange_local','release_center_m'):
                    checks.numeric(endpoint['targets'][name],targets[name],label+'/solver literal target/'+name,1e-12)
                checks.require(len(endpoint['candidates'])<=6,label+'/bounded finite eligible pairs')
                if endpoint.get('valid_ik_branches') is not None:
                    checks.require(all(type(x) is int and 0<=x<=17 for x in endpoint['valid_ik_branches']),label+'/finite branch cardinality')
                    order=[tuple(x) for x in endpoint['branch_order']]
                    checks.equal(len(order),len(set(order)),label+'/unique finite branch pair IDs')
                previous_rank=None
                for index,pair in enumerate(endpoint['pair_attempts']):
                    plabel=label+'/pair'+str(index);identity=(pair['branch10'],pair['branch14'])
                    checks.require(identity in [tuple(x) for x in endpoint['branch_order']],plabel+'/declared candidate pair')
                    distance=float(np.linalg.norm(np.asarray(pair['q10'])-old['q10'])+np.linalg.norm(np.asarray(pair['q14'])-old['q14']))
                    checks.numeric(pair['source_joint_distance_rad'],distance,plabel+'/joint-distance geometry ranking',1e-10)
                    rank=(distance,float(pair['residual']),*identity)
                    checks.require(all(np.isfinite(rank)) and (previous_rank is None or previous_rank<=rank),plabel+'/ranking independent of ball outcomes');previous_rank=rank
                    checked=gate.check(pair['q10'],pair['q14'],row['bucket_xy_m'],row['values'],targets)
                    checks.equal(checked['reason'],pair['checked']['reason'],plabel+'/strict endpoint reason recomputed')
                    if checked['reason']==0:
                        checks.require(checked['minimum_clearance_lower_bound_m']>=.005-1e-12,plabel+'/original strict5mm clearance')
                        checks.numeric(pair['checked']['minimum_clearance_lower_bound_m'],checked['minimum_clearance_lower_bound_m'],plabel+'/clearance bound recomputed',1e-10)
                for candidate in endpoint['candidates']:
                    matches=[p for p in endpoint['pair_attempts'] if p['branch10']==candidate['branch10'] and p['branch14']==candidate['branch14'] and p['q10']==candidate['q10'] and p['q14']==candidate['q14']]
                    checks.equal(len(matches),1,label+'/eligible pair has actual checked evidence')
                    checks.require(matches[0]['checked']['reason']==0 and not matches[0].get('pose_rejected'),label+'/eligible pair passed unchanged gates')
                    for chain,joint,target in ((chain10,'q10',targets['dropper_flange_world']),(chain14,'q14',targets['board_flange_local'])):
                        pose=yaw.forward_chain(chain,candidate[joint])
                        checks.require(np.linalg.norm(pose[:3,3]-target[:3,3])<1e-4,label+'/eligible pair position/'+joint)
                        checks.require(yaw.rotation_error(pose[:3,:3],target[:3,:3])<4e-4,label+'/eligible pair full rotation/'+joint)
            seen.add(row['connection_sha256'])
        # Display orientation is itself evidence, not trusted as a measurement.
        evidence=row['yaw_evidence'];targets=expected_targets(scene,row['values'],row['bucket_xy_m'],bucket_facing=row['adapter_directory']=='native-yaw')
        checks.equal(evidence['basis'],'model FK of observed native initial float32 joints; held endpoint',label+'/honest FK basis')
        checks.numeric(evidence['requested_yaw_rad'],targets['yaw_rad'],label+'/display yaw',1e-12)
        checks.numeric(evidence['requested_rotation_matrix'],targets['rotation_matrix'],label+'/display orientation',1e-12)
        if row.get('native_release'):
            pose=yaw.forward_chain(chain10,row['native_initial_q10']);board=yaw.forward_chain(chain14,row['native_initial_q14'])
            for name,value in dict(measured_initial_rotation_matrix=pose[:3,:3],measured_initial_yaw_rad=math.atan2(pose[1,0],pose[0,0]),
                rotation_error_rad=yaw.rotation_error(pose[:3,:3],targets['rotation_matrix']),position_error_m=np.linalg.norm(pose[:3,3]-targets['dropper_flange_world'][:3,3]),
                board_rotation_error_rad=yaw.rotation_error(board[:3,:3],targets['board_flange_local'][:3,:3]),
                board_position_error_m=np.linalg.norm(board[:3,3]-targets['board_flange_local'][:3,3]),
                measured_initial_flange_position_m=pose[:3,3],requested_release_center_m=targets['release_center_m']).items():
                checks.numeric(evidence[name],value,label+'/independent display '+name,1e-9)
        else:checks.equal(set(evidence),{'basis','requested_yaw_rad','requested_rotation_matrix'},label+'/refusal does not fabricate measured pose')
    checks.equal(len(seen),48,'Exactly48 new action proofs; zero-offset proof remains original')
    return dict(proof_count=len(seen),released_pose_checks=len(measurements),measurements=measurements,
        basis='Independent URDF FK of observed float32 reset joints; no driven transit or independent camera measurement.')


def audit_native(study,rows,spec,calibration,assets,checks,parent_study,yaw_study):
    setup={};count=0
    for namespace,prior_name,parent_root,parent_namespace in (('native','prior.json',parent_study,'native'),
            ('native-yaw','prior.json',yaw_study,'native-yaw'),('native-controls','old-prior.json',parent_study,'control-parent/native')):
        members=[r for r in rows if r['adapter_directory']==namespace];path=study/namespace
        checks.require(bool(members),namespace+'/expected nonempty evidence namespace')
        setup[namespace]=geometry.calibrated.check_setup_evidence(geometry._NativeNamespace(path),members,read(study/prior_name),read(study/'setup-home.json'),checks)
        admitted=[r for r in members if r.get('native_release')]
        if admitted:
            geometry.old.audit_native_options(path,spec['scene'],calibration,assets,checks)
            for row in admitted:
                checks.equal((row['effective_options_file'],row['effective_options_sha256']),('effective-options.json',sha(path/'effective-options.json')),row['id']+'/unchanged native64 effective options')
            for name in ('effective-options.json','native-compiled-geometry.json'):
                checks.equal(read(path/name),read(Path(parent_root)/parent_namespace/name),namespace+'/physics unchanged from parent/'+name)
        else:checks.require(not list(path.rglob('*.npz')),namespace+'/no fabricated native cache for rejected setups')
        count+=len(native.rescore_rows(path,members,spec['scene'],calibration,spec['identity']['trace_schema'],assets,checks))
        byfile={}
        for row in members:
            if row.get('native_release'):byfile.setdefault(row['native_cache_file'],[]).append(row)
            else:checks.equal(row.get('diagnostic_robot_contact_steps'),0,row['id']+'/no contact without flight')
        for name,group in byfile.items():
            with np.load(contained(path,name),allow_pickle=False) as data:
                for row in group:
                    contacts=int(np.count_nonzero(data['forces_N'][:int(row['physics_steps']),int(row['native_cache_row']),4]>.01))
                    checks.equal(row['diagnostic_robot_contact_steps'],contacts,row['id']+'/dense robot contact count')
    return setup,count


def independent_cases(plan,rows,selection):
    source=next(r for r in plan if r['variant']=='far_rotated');lookup={r['id']:r for r in rows};chosen={r['candidate_id'] for r in selection};cells=[]
    for candidate in [r for r in plan if r['phase']=='discovery']:
        row=lookup[candidate['id']]
        admission=dict(passed=bool(row['native_release']),reason=row.get('invalid'),setup_proof=row.get('setup_proof'),
            connection_file=row['connection_file'],connection_sha256=row['connection_sha256'])
        flight=dict(capture=native.captured(row),native_release=row['native_release'],invalid_physics=row.get('invalid_physics'),
            robot_contact_steps=row.get('diagnostic_robot_contact_steps',0),whole_ball_entry=row.get('whole_ball_entry'),
            max_retention_s=row.get('max_retention_s'),table_contact_steps=row.get('table_contact_steps',0),route_quality=row.get('route_quality'),record_sha256=digest(row))
        cells.append(dict(candidate_id=candidate['id'],grid_index=candidate['grid_index'],values=candidate['values'],bucket_xy_m=candidate['bucket_xy_m'],
            admission=admission,discovery=flight,selected=candidate['id'] in chosen,repeats=statistics([r for r in rows if r.get('selected_from_id')==candidate['id']])))
    return [dict(context_id=FAR,bucket_xy_m=source['bucket_xy_m'],baseline_values=source['values'],target_yaw_rad=source['requested_gripper_yaw_rad'],
        grid=cells,selection=selection,candidate_repeats=[dict(**s,summary=statistics([r for r in rows if r.get('selected_from_id')==s['candidate_id']])) for s in selection],
        variants={v:statistics([r for r in rows if r['variant']==v]) for v in ('far_rotated','far_success_reference','near_success_reference','grid','selected')})]


def audit_timing(study,rows,report,selection,checks):
    boundaries=replay.read_records(study/'phase-boundaries.jsonl');n=3*len(selection)
    expected=[('plan-frozen',0),('controls_pre-complete',9),('references-complete',18),('discovery-complete',67),
        ('selection-frozen',67),('selected_repeats-complete',67+n),('controls_post-complete',76+n)]
    checks.equal([(r['phase'],r['rows_count']) for r in boundaries],expected,'All phases and complete fixed-request prefixes')
    raw=(study/'rows.jsonl').read_bytes().splitlines(keepends=True);previous=0
    for boundary in boundaries:
        count=boundary['rows_count'];hashed=hashlib.sha256(b''.join(raw[:count])).hexdigest() if count else None
        checks.equal(boundary['rows_sha256'],hashed,boundary['phase']+'/immutable observed prefix')
        checks.equal(boundary['plan_sha256'],sha(study/'plan.json'),boundary['phase']+'/same pre-outcome49-grid plan')
        checks.require(previous<=boundary['acquired_elapsed_s']<3600,'Monotone bounded own worker times');previous=boundary['acquired_elapsed_s']
    packets=geometry.audit_request_packets(study,rows,checks);timing=replay.read_records(study/'timing.jsonl')
    expected_phases=['controls_pre']+['references']*3+['discovery']*7+['selected_repeats']*math.ceil(n/32)+['controls_post']
    checks.equal([t['phase'] for t in timing],expected_phases,'Exact namespace-separated batches and no hidden retries')
    checks.require(boundaries[0]['acquired_elapsed_s']<=timing[0]['acquired_start_s'],'Base grid and repeat-selection rule freeze before first outcome')
    for item in timing:
        checks.require(0<=item['acquired_start_s']<=item['acquired_end_s']<3600,'Every batch within bounded acquired time')
        if item['phase']=='selected_repeats':checks.require(item['acquired_start_s']>=boundaries[4]['acquired_elapsed_s'],'Repeat plan freezes before confirmation outcomes')
    budget=read(study.with_name(study.name+'-budget.json'));pipeline=read(study.with_suffix('.pipeline.json'))
    checks.equal(budget['gpu_uuid'],'GPU-ab1950b0-01dd-0fab-407f-2d17936b27f4','Only authorized4090 UUID')
    checks.equal(budget['audit_reserve_s'],1800,'Independent CPU audit reserve')
    checks.require(60<=budget['limit_s']<=3600 and 0<pipeline['global_limit_s']<=6300,'Finite worker/pipeline limits')
    checks.numeric(pipeline['global_deadline_unix_s']-pipeline['started_unix_s'],pipeline['global_limit_s'],'Original absolute deadline',.05)
    checks.equal(budget['global_deadline_unix_s'],pipeline['global_deadline_unix_s'],'Worker uses original pipeline deadline')
    checks.require(pipeline['started_unix_s']<=budget['acquired_unix_s'],'Lease follows pipeline startup')
    checks.numeric(budget['effective_worker_limit_s'],min(budget['limit_s'],budget['global_deadline_unix_s']-budget['acquired_unix_s']-1800),'Bounded worker window after queue',.1)
    checks.equal(report['deadlines'],dict(worker_limit_s=budget['limit_s'],effective_worker_limit_s=budget['effective_worker_limit_s'],global_deadline_unix_s=budget['global_deadline_unix_s']),'Report immutable executed budget')
    checks.require(0<report['elapsed_s']<budget['effective_worker_limit_s']<=3600,'Worker completed within acquired allowance')
    return packets,dict(worker_receipt_sha256=sha(study.with_name(study.name+'-budget.json')),effective_worker_limit_s=budget['effective_worker_limit_s'],global_limit_s=pipeline['global_limit_s'])


def audit(study,source,assets,*,parent_study,parent_audit,yaw_study,yaw_audit):
    started=time.monotonic();checks=replay.AuditChecks()
    study,source,assets,parent_study,parent_audit,yaw_study,yaw_audit=map(Path,(study,source,assets,parent_study,parent_audit,yaw_study,yaw_audit))
    manifest=replay.check_manifest(study,checks);spec=read(study/'contract.json');report=read(study/'report.json')
    unsigned=dict(spec);fp=unsigned.pop('fingerprint')
    checks.equal((digest(unsigned),report['contract_fingerprint']),(fp,fp),'Complete experiment contract fingerprint')
    checks.equal((spec['version'],report['version'],report['status'],report['phase']),(STUDY_VERSION,STUDY_VERSION,'completed','completed'),'Completed fixed-yaw tilt/drop-gap experiment')
    checks.equal(report['evidence_sha256'],manifest,'Report binds every saved artifact')
    checks.equal(spec['protocol'],protocol(),'Literal bounded protocol frozen before outcomes')
    checks.equal(report['protocol'],protocol(),'Report retains original protocol')
    replay.check_source_files(source,spec['source_sha256'],checks,'Executed frozen source snapshot')
    for module in (Path(__file__),Path(oracle.__file__)):
        checks.equal(sha(module),spec['source_sha256']['sircus/mjlab/'+module.name],'Actual independent checker frozen/'+module.name)
    replay.audit_packages(spec['packages'],checks)
    sources,controls,parent=yaw.audit_parent(study,parent_study,parent_audit,spec,checks)
    yaw_rows,yaw_spec,yaw_bindings=authenticate_yaw(yaw_study,yaw_audit,checks)
    checks.equal(read(study/'yaw-rows.json'),yaw_rows,'Original first yaw repetitions preserved exactly')
    checks.equal(spec['yaw_bindings'],yaw_bindings,'Complete consumed yaw provenance')
    for key in ('scene','identity','packages','runtime_reference'):
        checks.equal(spec[key],yaw_spec[key],'Yaw and corrected-imitation runtime identity agree/'+key)
    for name,expected in spec['identity']['asset_sha256'].items():checks.equal(sha(contained(assets.parent,name)),expected,'Original robot asset/'+name)
    for name,key in (('plan.json','plan_sha256'),('rule.json','rule_sha256'),('parent-rows.json','parent_rows_sha256'),
        ('control-replays.json','control_replays_sha256'),('yaw-rows.json','yaw_rows_sha256')):
        checks.equal(sha(study/name),spec[key],'Frozen input bytes/'+name)
    rule=read(study/'rule.json');plan=read(study/'plan.json');selection=read(study/'selection.json');selected_plan=read(study/'selected-plan.json')
    checks.equal((spec['rule'],report['rule']),(rule,rule),'Frozen original intervention disclosed')
    audit_plan(plan,sources,controls,yaw_rows,rule,checks)
    rows=replay.read_records(study/'rows.jsonl');audit_observations(rows,plan,sources,controls,yaw_rows,selection,selected_plan,checks)
    yaw.audit_setup_summaries(study,rows,checks);native.check_rows(rows,checks)
    calibration=replay.check_calibration(study,spec['identity']['calibration'],checks)
    setup,native_count=audit_native(study,rows,spec,calibration,assets,checks,parent_study,yaw_study)
    geometry_result=audit_geometry(study,rows,sources,controls,yaw_rows,spec['scene'],assets,checks)
    packets,budget=audit_timing(study,rows,report,selection,checks)
    gates={}
    for phase,key in (('controls_pre','pre'),('controls_post','post')):
        gates[key]=native.control_gate([r for r in rows if r['phase']==phase]);checks.equal(gates[key]['status'],'passed',phase+'/unchanged direct/table/miss hard gate')
    checks.equal(report['control_gates'],gates,'All18 original controls')
    variants=('far_rotated','far_success_reference','near_success_reference','grid','selected')
    counts={v:statistics([r for r in rows if r['variant']==v]) for v in variants};cases=independent_cases(plan,rows,selection)
    checks.equal(report['counts'],statistics(rows),'Complete requested denominator including setup declines')
    checks.equal(report['counts_by_variant'],counts,'Every diagnostic variant counted')
    checks.equal(report['cases'],cases,'All49 cells and all confirmation repeats disclosed')
    checks.equal(report['selection'],selection,'Every discovery capture selected by fixed rule')
    checks.equal((report['expected_requests'],report['maximum_requests'],report['training'],report['noise'],report['posthoc_development']),
        (len(rows),223,False,False,True),'No training, noise or generalization claim')
    checks.equal(report['gate'],dict(status='passed',substeps=64,controls_passed=True,complete_plan=True,no_policy_promotion=True),'Complete bounded execution without promotion')
    return dict(version=VERSION,status='passed',full_checks_verified=True,native_checks_verified=True,checks_count=checks.count,checks=checks.entries,failures=[],
        contract_fingerprint=fp,report_sha256=sha(study/'report.json'),evidence_manifest_sha256=sha(study/'evidence-manifest.json'),auditor_sha256=sha(Path(__file__)),
        requested_trials=len(rows),native_releases=native_count,counts_by_variant=counts,cases=cases,control_gates=gates,selection=selection,
        setup_evidence=setup,yaw_geometry=geometry_result,request_packets=packets,budget=budget,elapsed_s=time.monotonic()-started,maximum_numeric_error=checks.numeric_error,
        limitations=['One inspected development layout; no policy fitting, promotion, fresh held-out evaluation or all-table success claim.',
            'Only requested tilt/drop gap change in the grid; fixed board center, bucket and full dropper yaw. Necessary arm joints may change.',
            'Three fresh nominal confirmations of every discovery capture; no noise,128 qualification, learned transit or physical release validation.',
            'Finite setup declines count as unsuccessful requests, but do not prove physical unreachability.'])


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('study','source','assets','parent-study','parent-audit','yaw-study','yaw-audit','output'):p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args()
    if a.output.exists() or a.output.resolve().is_relative_to(a.study.resolve()):raise ValueError('Fresh separate audit output required')
    try:result=audit(a.study,a.source,a.assets,parent_study=a.parent_study,parent_audit=a.parent_audit,yaw_study=a.yaw_study,yaw_audit=a.yaw_audit)
    except Exception as error:
        a.output.write_text(json.dumps(dict(version=VERSION,status='failed',failures=[type(error).__name__+': '+str(error)]),indent=2)+'\n');raise
    a.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:result[k] for k in ('status','checks_count','requested_trials','native_releases','counts_by_variant')}))

if __name__=='__main__':main()
