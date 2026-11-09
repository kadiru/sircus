"""Frozen four-case height/tilt grid; diagnostic search, not policy training."""
from __future__ import annotations
import copy
from .benchmark_verify import digest
from .table_correction_native import META, CONTROL_ROLES
from .table_training_contract import captured

VERSION='sircus-table-height-tilt-study-v1'
CONTEXTS=tuple(f'table-training-test-{i:02d}' for i in (0,1,18,26))
HEIGHTS=(.32,.36,.40,.44,.46,.48)
TILTS=(16.,20.,24.,28.,32.)


def protocol():
    return dict(contexts=list(CONTEXTS),heights_m=list(HEIGHTS),tilts_deg=list(TILTS),
        fixed_parameters=['bucket_xy_m','board_x_m','board_y_m','drop_gap_m'],
        candidates_per_case=30,total_candidates=120,substeps=64,nominal_repeats=3,
        selection_order=['route_quality_desc','v3_reward_desc','grid_index_asc'],
        selection_requires_capture=True,noise=False,training=False,posthoc_development=True)


def budget():
    return dict(maximum_requested=162,candidate_proposals=120,original_requests=12,
        control_requests=18,maximum_selected_repeat_requests=12,cpu_admission_limit_s=3600,
        native_acquired_limit_s=3600,maximum_queue_s=7200,storage_reserve_bytes=8*1024**3)


def sources_from_parent(rows):
    sources=[]
    for role in CONTROL_ROLES:
        found=[r for r in rows if r.get('phase')=='controls_pre' and r.get('role')==role and r.get('repeat')==0]
        if len(found)!=1 or not found[0].get('native_release'):raise ValueError('One original admitted control required')
        sources.append(found[0])
    for context in CONTEXTS:
        found=[r for r in rows if r.get('phase')=='test' and r.get('method')=='ppo' and r.get('context_id')==context]
        if len(found)!=1 or not found[0].get('native_release') or found[0].get('route_success'):
            raise ValueError('One original admitted diagnosed miss required')
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
        for hi,height in enumerate(HEIGHTS):
            for ti,tilt in enumerate(TILTS):
                case_index=hi*len(TILTS)+ti;values=copy.deepcopy(source['values']);values[2:4]=[height,tilt]
                result.append(request(source,identifier=f'height-tilt-{context}-g{case_index:02d}',phase='discovery',
                    variant='grid',repeat=0,values=values,grid_index=len(result),case_grid_index=case_index))
    if len(result)!=120:raise RuntimeError('Literal grid accounting changed')
    return result


def fixed_requests(sources,phase):
    if phase not in ('controls_pre','baseline','controls_post'):raise ValueError('Unknown fixed phase')
    result=[]
    ordered=([next(r for r in sources if r['role']==role) for role in CONTROL_ROLES]
             if phase.startswith('controls_') else [next(r for r in sources if r['context_id']==c) for c in CONTEXTS])
    variant='control_exact' if phase.startswith('controls_') else 'baseline'
    for source in ordered:
        for repeat in range(3):
            result.append(request(source,identifier=f'height-tilt-{phase}-{source["context_id"]}-r{repeat}',
                phase=phase,variant=variant,repeat=repeat,values=source['values']))
    return result


def select_captures(rows):
    selected=[]
    for context in CONTEXTS:
        group=[r for r in rows if r['phase']=='discovery' and r['context_id']==context and captured(r)]
        if not group:selected.append(dict(context_id=context,selected=None));continue
        row=min(group,key=lambda r:(-r['route_quality'],-r['learning_reward'],r['grid_index']))
        selected.append(dict(context_id=context,selected=dict(candidate_id=row['id'],grid_index=row['grid_index'],
            case_grid_index=row['case_grid_index'],row_sha256=digest(row),values=row['values'],bucket_xy_m=row['bucket_xy_m'],
            connection_file=row['connection_file'],connection_sha256=row['connection_sha256'],q10=row['q10'],q14=row['q14'],
            route_quality=row['route_quality'],learning_reward=row['learning_reward'])))
    return selected


def selected_requests(plan,selection):
    lookup={r['id']:r for r in plan};result=[]
    for record in selection:
        selected=record['selected']
        if selected is None:continue
        source=lookup[selected['candidate_id']]
        for repeat in range(3):
            row=copy.deepcopy(source);row.update(id=f'height-tilt-selected-{record["context_id"]}-r{repeat}',
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
