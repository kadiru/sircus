"""Bounded far-shot tilt/gap discovery, with fixed bucket-facing wrist rotation."""
from __future__ import annotations
import copy
import json
from pathlib import Path
from .benchmark_verify import read,sha,digest,contained
from .gripper_yaw_contract import authenticate_parent
from .table_correction_native import META
from .table_training_contract import captured

VERSION='sircus-yaw-tilt-gap-v1'
FAR='corrected-imitation-test-r0-c6-n1'
NEAR='corrected-imitation-test-r1-c2-n0'
TILTS=(24.,26.,28.,30.33068546648235,32.,34.,36.)
GAP_OFFSETS=(0.,.02,.04,.06,.09,.12,.16)
TOTAL_CANDIDATES=49
MAXIMUM_REQUESTS=223
INPUT_KEYS=('parent-study','parent-audit','yaw-study','yaw-audit','rule','reference','assets')
YAW_REPORT='2cd3644a19653efc157fe697557ebc861e40ab6b97ceeb3faeb48aaf04b923fa'
YAW_AUDIT='4f044f4c3b4dc3ffd79dcb65f42214ceb3f597d129afb0c79728697137b9ec21'
YAW_FP='e31c8db6e8b9a2b16f7213b3440396cccf83004e64988575f694b2b007cbf683'
ROLES=('direct_control','table_control','negative_control')
VARIANTS=('far_rotated','far_success_reference','near_success_reference','grid','selected')


def literal_rule():
    return dict(version=VERSION,context=FAR,tilts_deg=list(TILTS),gap_offsets_m=list(GAP_OFFSETS),
        rotation='Rz(atan2(bucket_y-board_y,bucket_x-board_x)) @ diag(1,-1,-1)',
        fixed=['bucket_xy_m','board_x_m','board_y_m','board_height_m','gripper_yaw'],
        changed=['tilt_deg','drop_gap_m'],selection='all valid discovery captures in grid-index order, then three fresh nominal repeats',
        substeps=64,noise=False,training=False)


def validate_rule(rule):
    if rule!=literal_rule():raise ValueError('Exact frozen49-candidate far-shot rule required')
    return copy.deepcopy(rule)


def protocol():
    return dict(**literal_rule(),maximum_requests=MAXIMUM_REQUESTS,candidate_requests=49,
        fixed_reference_requests=9,control_requests=18,maximum_selected_repeat_requests=147,
        repeats=3,maximum_acquired_s=3600,maximum_queue_s=900,maximum_pipeline_s=6300,
        audit_reserve_s=1800,storage_reserve_bytes=8*1024**3,posthoc_development=True,
        outcome_adaptive_grid=False,selection_frozen_before_repeats=True,
        branch_selection='geometry only; nearest original yaw joints then residual and branch IDs',
        orientation_measurement='model FK of observed initial float32 native joints; held endpoints')


def authenticate_inputs(parent,parent_audit,yaw_study,yaw_audit):
    loaded=authenticate_parent(parent,parent_audit);root=Path(yaw_study)
    if sha(root/'report.json')!=YAW_REPORT or sha(yaw_audit)!=YAW_AUDIT:raise ValueError('Exact audited yaw parent required')
    report,spec,proof,manifest=(read(p) for p in (root/'report.json',root/'contract.json',yaw_audit,root/'evidence-manifest.json'))
    unsigned=dict(spec);fp=unsigned.pop('fingerprint')
    if (digest(unsigned)!=fp or fp!=YAW_FP or report['contract_fingerprint']!=fp or proof['contract_fingerprint']!=fp
        or report['status']!='completed' or proof['status']!='passed' or proof['full_checks_verified'] is not True
        or proof['native_checks_verified'] is not True or proof['failures'] or proof['checks_count']!=18330
        or len(proof['checks'])!=18330 or not all(c.get('passed') is True for c in proof['checks'])
        or proof['report_sha256']!=YAW_REPORT or report['evidence_sha256']!=manifest
        or proof['evidence_manifest_sha256']!=sha(root/'evidence-manifest.json')):raise ValueError('Bound yaw proof required')
    files={}
    def verify(name):
        path=contained(root,name)
        if sha(path)!=manifest[name]:raise ValueError('Yaw artifact changed: '+name)
        files[name]=manifest[name];return path
    rows=[json.loads(x) for x in verify('rows.jsonl').read_text().splitlines()]
    if len(rows)!=36:raise ValueError('Complete yaw36 ledger required')
    selected=[]
    for context in (FAR,NEAR):
        found=[r for r in rows if r['context_id']==context and r['variant']=='bucket_facing' and r['repeat']==0]
        if len(found)!=1 or found[0]['native_release'] is not True or found[0]['invalid_physics'] or found[0]['diagnostic_robot_contact_steps']:
            raise ValueError('Exact clear native yaw reference required')
        if captured(found[0])!=(context==NEAR):raise ValueError('Exact far-miss/near-capture source pair required')
        selected.append(found[0])
    for row in selected:
        for key,hkey in (('context_resolution_file','context_resolution_sha256'),('connection_file','connection_sha256'),('native_cache_file','native_cache_sha256')):
            name='native-yaw/'+row[key];verify(name)
            if files[name]!=row[hkey]:raise ValueError('Yaw row/evidence binding mismatch')
    for name in ('contract.json','native-yaw/effective-options.json','native-yaw/native-compiled-geometry.json'):verify(name)
    loaded.update(yaw_rows=selected,yaw_bindings=dict(report_sha256=YAW_REPORT,audit_sha256=YAW_AUDIT,
        contract_fingerprint=fp,manifest_sha256=sha(root/'evidence-manifest.json'),files_sha256=files))
    return loaded


def make_plan(loaded):
    sources=loaded['sources'];controls=loaded['controls'];yaw={r['context_id']:r for r in loaded['yaw_rows']}
    source=lambda context,method:next(r for r in sources if r['context_id']==context and r['method']==method)
    far=source(FAR,'corrected_bc');near=source(NEAR,'corrected_bc');current=source(FAR,'current_bc')
    plan=[]
    def add(s,variant,phase,repeat,values=None,grid_index=None):
        values=list(s['values'] if values is None else values);rotated=variant not in ('far_success_reference','control_exact')
        namespace='native-controls' if variant=='control_exact' else ('native-yaw' if rotated else 'native')
        old=yaw.get(s['context_id']) if rotated else None
        r={k:copy.deepcopy(s[k]) for k in META}
        r.update(id=f'yaw-tilt-gap-{phase}-{variant}-'+(f'g{grid_index:02d}' if grid_index is not None else s['context_id'])+f'-r{repeat}',
            phase=phase,role=s['role'] if variant=='control_exact' else 'diagnostic',method=variant,
            variant=variant,repeat=repeat,context_id=s['context_id'],bucket_xy_m=copy.deepcopy(s['bucket_xy_m']),values=values,command_values=copy.deepcopy(values),
            source_record_id=s['id'],source_record_sha256=digest(s),adapter_directory=namespace,
            requested_gripper_yaw_rad=old['requested_gripper_yaw_rad'] if rotated else 0.,
            frozen_yaw_record_id=old['id'] if rotated else None,frozen_yaw_record_sha256=digest(old) if rotated else None,
            parameter_indices=[3,4] if variant=='grid' else [],grid_index=grid_index,substeps=64,
            perturbation_mixed_units=[0.]*5,selection_or_learning=False,posthoc_development=True,
            admission_mode='yaw_tilt_gap_strict_proof' if variant=='grid' else 'exact_parent_branch')
        plan.append(r)
    for s in controls:
        for i in range(3):add(s,'control_exact','controls_pre',i)
    for s,v in ((far,'far_rotated'),(current,'far_success_reference'),(near,'near_success_reference')):
        for i in range(3):add(s,v,'references',i)
    for tilt in TILTS:
        for offset in GAP_OFFSETS:
            v=list(far['values']);v[3]=tilt;v[4]+=offset
            add(far,'grid','discovery',0,v,TILTS.index(tilt)*len(GAP_OFFSETS)+GAP_OFFSETS.index(offset))
    for s in controls:
        for i in range(3):add(s,'control_exact','controls_post',i)
    if len(plan)!=76 or len({r['id'] for r in plan})!=76:raise ValueError('Frozen76 base slots required')
    return plan


def select_captures(rows):
    selected=[]
    for row in sorted((r for r in rows if r['phase']=='discovery'),key=lambda r:r['grid_index']):
        if row.get('native_release') is True and row.get('invalid_physics') is False and not row.get('invalid') and captured(row):
            selected.append(dict(candidate_id=row['id'],grid_index=row['grid_index'],row_sha256=digest(row),
                **{k:copy.deepcopy(row[k]) for k in ('values','bucket_xy_m','connection_file','connection_sha256','q10','q14')}))
    return selected


def selected_requests(plan,selection):
    lookup={r['id']:r for r in plan};result=[]
    for s in selection:
        original=lookup[s['candidate_id']]
        for i in range(3):
            row=copy.deepcopy(original);row.update(id=f'yaw-tilt-gap-selected-g{s["grid_index"]:02d}-r{i}',phase='selected_repeats',variant='selected',method='selected',repeat=i,
                selected_from_id=s['candidate_id'],selected_from_sha256=s['row_sha256'])
            result.append(row)
    return result
