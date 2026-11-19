"""Frozen tilt/gap exploration around the recorded contact-free Case 00 miss.

All outcomes remain evidence. Every valid discovery capture receives three
fresh nominal repeats; no discovery outcome changes the candidate grid.
"""
from __future__ import annotations
import copy
from .benchmark_verify import digest
from .table_correction_native import META, CONTROL_ROLES
from .table_training_contract import captured

VERSION='sircus-case00-tilt-gap-study-v1'
CONTEXTS=('table-training-test-00',)
SOURCE_IDS=('bouncer-clearance-table-training-test-00-g14',)
TILTS=(20.,22.,24.,26.,28.,30.,32.)
GAP_OFFSETS=(-.06,-.04,-.02,-.01,0.,.01,.02,.04,.06,.08)
TOTAL_CANDIDATES=70
MAXIMUM_REQUESTED=301


def protocol():
    return dict(contexts=list(CONTEXTS),source_record_ids=list(SOURCE_IDS),tilts_deg=list(TILTS),
        gap_offsets_m=list(GAP_OFFSETS),fixed_parameters=['bucket_xy_m','board_x_m','board_y_m','board_height_m'],
        changed_parameters=['tilt_deg','drop_gap_m'],heading='unchanged bucket-facing orientation convention',
        candidates_per_case=70,total_candidates=70,substeps=64,nominal_repeats=3,
        selection_rule='all valid native discovery captures, in frozen grid-index order',
        selection_requires_capture=True,selection_maximum=70,
        selection_frozen_before_repeats=True,noise=False,training=False,posthoc_development=True,
        alternative_same_pose_joint_branches=False)


def budget():
    return dict(maximum_requested=301,candidate_proposals=70,original_requests=3,
        control_requests=18,maximum_selected_repeat_requests=210,cpu_admission_limit_s=3600,
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
            or found[0].get('route_success') or found[0].get('invalid_physics') is not False
            or found[0].get('board_bounce') is not True or found[0].get('diagnostic_robot_contact_steps')!=0):
            raise ValueError('Exact recorded contact-free Case 00 miss required')
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
    result=[];source=next(r for r in sources if r['id']==SOURCE_IDS[0])
    for tilt in TILTS:
        for offset in GAP_OFFSETS:
            index=len(result);values=copy.deepcopy(source['values']);values[3]=tilt;values[4]+=offset
            result.append(request(source,identifier=f'case00-tilt-gap-{CONTEXTS[0]}-g{index:02d}',phase='discovery',
                variant='grid',repeat=0,values=values,grid_index=index,case_grid_index=index))
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
            result.append(request(source,identifier=f'case00-tilt-gap-{phase}-{source["context_id"]}-r{repeat}',
                phase=phase,variant=variant,repeat=repeat,values=source['values']))
    return result


def select_captures(rows):
    """Select every valid discovery capture, never rerank based on later repeats."""
    chosen=[r for r in rows if r['phase']=='discovery' and r['context_id'] in CONTEXTS
            and r.get('native_release') is True and r.get('invalid_physics') is False
            and not r.get('invalid') and captured(r)]
    selected=[]
    for row in sorted(chosen,key=lambda r:r['grid_index']):
        selected.append(dict(context_id=row['context_id'],selected=dict(candidate_id=row['id'],grid_index=row['grid_index'],
            case_grid_index=row['case_grid_index'],row_sha256=digest(row),values=row['values'],bucket_xy_m=row['bucket_xy_m'],
            connection_file=row['connection_file'],connection_sha256=row['connection_sha256'],q10=row['q10'],q14=row['q14'],
            route_quality=row['route_quality'],learning_reward=row['learning_reward'],capture=captured(row),
            diagnostic_robot_contact_steps=row['diagnostic_robot_contact_steps'])))
    return selected


def selected_requests(plan,selection):
    lookup={r['id']:r for r in plan};result=[]
    for record in selection:
        selected=record['selected'];source=lookup[selected['candidate_id']]
        for repeat in range(3):
            row=copy.deepcopy(source);row.update(id=f'case00-tilt-gap-selected-g{selected["grid_index"]:02d}-r{repeat}',
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
