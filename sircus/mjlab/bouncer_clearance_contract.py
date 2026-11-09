"""Frozen XY exploration around three recorded bouncer-contact misses.

This keeps every failed trial as evidence. It is development search, not a
trained-policy evaluation or a claim that geometric admission predicts capture.
"""
from __future__ import annotations
import copy
from .benchmark_verify import digest
from .table_correction_native import META, CONTROL_ROLES
from .table_training_contract import captured

VERSION='sircus-bouncer-clearance-study-v1'
CONTEXTS=tuple(f'table-training-test-{i:02d}' for i in (0,18,26))
SOURCE_IDS=('height-tilt-table-training-test-00-g23',
            'height-tilt-table-training-test-18-g23',
            'height-tilt-table-training-test-26-g27')
OFFSETS=(-.12,-.06,0.,.06,.12)
TOTAL_CANDIDATES=75
MAXIMUM_REQUESTED=111


def protocol():
    return dict(contexts=list(CONTEXTS),source_record_ids=list(SOURCE_IDS),xy_offsets_m=list(OFFSETS),
        fixed_parameters=['bucket_xy_m','board_height_m','tilt_deg','drop_gap_m'],
        changed_parameters=['board_x_m','board_y_m'],heading='unchanged bucket-facing orientation convention',
        candidates_per_case=25,total_candidates=75,substeps=64,nominal_repeats=3,
        selection_order=['capture_desc','route_quality_desc','v3_reward_desc','grid_index_asc'],
        selection_requires_capture=False,selection_requires_board_bounce=True,
        selection_requires_zero_robot_contact_before_scoring_terminal=True,
        noise=False,training=False,posthoc_development=True,
        alternative_same_pose_joint_branches=False)


def budget():
    return dict(maximum_requested=111,candidate_proposals=75,original_requests=9,
        control_requests=18,maximum_selected_repeat_requests=9,cpu_admission_limit_s=3600,
        native_acquired_limit_s=3600,maximum_queue_s=7200,storage_reserve_bytes=8*1024**3)


def sources_from_parent(rows):
    sources=[]
    for role in CONTROL_ROLES:
        found=[r for r in rows if r.get('phase')=='controls_pre' and r.get('role')==role and r.get('repeat')==0]
        if len(found)!=1 or not found[0].get('native_release'):raise ValueError('One original admitted control required')
        sources.append(found[0])
    for identifier,context in zip(SOURCE_IDS,CONTEXTS):
        found=[r for r in rows if r.get('id')==identifier]
        if (len(found)!=1 or found[0].get('context_id')!=context or not found[0].get('native_release')
            or found[0].get('route_success') or found[0].get('diagnostic_robot_contact_steps',0)<=0):
            raise ValueError('One exact recorded bouncer-contact miss required')
        sources.append(found[0])
    return copy.deepcopy(sources)


def request(source,*,identifier,phase,variant,repeat,values,grid_index=None,case_grid_index=None):
    out={key:copy.deepcopy(source[key]) for key in META}
    out.update(id=identifier,phase=phase,role=source['role'] if variant=='control_exact' else 'diagnostic',
        method='control' if variant=='control_exact' else variant,variant=variant,repeat=repeat,
        context_id=source['context_id'],bucket_xy_m=copy.deepcopy(source['bucket_xy_m']),values=list(values),
        command_values=list(values),source_record_id=source['id'],source_record_sha256=digest(source),
        substeps=64,perturbation_mixed_units=[0.]*5,selection_or_learning=False,posthoc_development=True,
        grid_index=grid_index,case_grid_index=case_grid_index)
    return out


def grid_plan(sources):
    result=[];by_context={r['context_id']:r for r in sources}
    for context in CONTEXTS:
        source=by_context[context]
        for xi,dx in enumerate(OFFSETS):
            for yi,dy in enumerate(OFFSETS):
                case_index=xi*len(OFFSETS)+yi;values=copy.deepcopy(source['values'])
                values[0]+=dx;values[1]+=dy
                result.append(request(source,identifier=f'bouncer-clearance-{context}-g{case_index:02d}',phase='discovery',
                    variant='grid',repeat=0,values=values,grid_index=len(result),case_grid_index=case_index))
    if len(result)!=TOTAL_CANDIDATES:raise RuntimeError('Literal grid accounting changed')
    return result


def fixed_requests(sources,phase):
    if phase not in ('controls_pre','baseline','controls_post'):raise ValueError('Unknown fixed phase')
    result=[]
    ordered=([next(r for r in sources if r['role']==role) for role in CONTROL_ROLES]
             if phase.startswith('controls_') else [next(r for r in sources if r['context_id']==c) for c in CONTEXTS])
    variant='control_exact' if phase.startswith('controls_') else 'baseline'
    for source in ordered:
        for repeat in range(3):
            result.append(request(source,identifier=f'bouncer-clearance-{phase}-{source["context_id"]}-r{repeat}',
                phase=phase,variant=variant,repeat=repeat,values=source['values']))
    return result


def qualifies(row):
    return (row.get('native_release') is True and row.get('invalid_physics') is False
        and not row.get('invalid') and row.get('board_bounce') is True
        and row.get('diagnostic_robot_contact_steps')==0)


def select_clear(rows):
    selected=[]
    for context in CONTEXTS:
        group=[r for r in rows if r['phase']=='discovery' and r['context_id']==context and qualifies(r)]
        if not group:selected.append(dict(context_id=context,selected=None));continue
        row=min(group,key=lambda r:(-int(captured(r)),-r['route_quality'],-r['learning_reward'],r['grid_index']))
        selected.append(dict(context_id=context,selected=dict(candidate_id=row['id'],grid_index=row['grid_index'],
            case_grid_index=row['case_grid_index'],row_sha256=digest(row),values=row['values'],bucket_xy_m=row['bucket_xy_m'],
            connection_file=row['connection_file'],connection_sha256=row['connection_sha256'],q10=row['q10'],q14=row['q14'],
            route_quality=row['route_quality'],learning_reward=row['learning_reward'],capture=captured(row),
            diagnostic_robot_contact_steps=row['diagnostic_robot_contact_steps'])))
    return selected


def selected_requests(plan,selection):
    lookup={r['id']:r for r in plan};result=[]
    for record in selection:
        selected=record['selected']
        if selected is None:continue
        source=lookup[selected['candidate_id']]
        for repeat in range(3):
            row=copy.deepcopy(source);row.update(id=f'bouncer-clearance-selected-{record["context_id"]}-r{repeat}',
                phase='selected_repeats',variant='selected',method='selected',repeat=repeat,
                selected_from_id=selected['candidate_id'],selected_from_sha256=selected['row_sha256'])
            result.append(row)
    return result


def empty_matrix(plan):
    return [dict(candidate_id=r['id'],context_id=r['context_id'],grid_index=r['grid_index'],case_grid_index=r['case_grid_index'],
        values=r['values'],bucket_xy_m=r['bucket_xy_m'],status='not_attempted',passed=None,reason='pending',
        valid_ik_branches=None,connection_file=None,connection_sha256=None,
        resolution_file=r['context_resolution_file'],resolution_sha256=r['context_resolution_sha256'],q10=None,q14=None)
        for r in plan]
