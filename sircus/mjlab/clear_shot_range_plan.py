"""Fixed 135-degree board-pose range test after an observed arm-free miss."""
from __future__ import annotations
import copy
import math
from .benchmark_verify import digest,read,sha,contained
from .nearby_transfer_plan import audited_source,rows_at,physical_match,captured

VERSION='sircus-clear-shot-range-v1'
CASE_IDS=('parallel-board-miss-09',)
SOURCE_ID='parallel-board-miss-09-clearance-0016-b0'
GAP_DELTAS_M=(0.,.02,.04,.06,.08,.10,.13,.16,.20,.24)
TILT_DELTAS_DEG=(0.,-3.,3.,-6.,6.)
SEARCH_RULE=dict(version='sircus-clear-shot-range-menu-v1',gap_deltas_m=list(GAP_DELTAS_M),
    tilt_deltas_deg=list(TILT_DELTAS_DEG),proposal_order='Tilt outer, gap inner; unchanged pair is original only',
    proposals=49,one_per_setting=True,board_branch_tolerance_rad=1e-4,
    cpu_case_limit_s=600,proposal_limit_s=30,maximum_native_candidates_per_case=49,
    maximum_total_native_releases=226,repeats_per_success=3,preservation_releases=9,
    selection='Every eligible fixed-grid command; no flight ranking or post-drop selection',
    scope='One known arm-free miss; gap first and modest tilt adjustments; no training or promotion')


def source_inputs(a):
    report,contract,binding=audited_source(a.source_study,a.source_audit,'sircus-link-clearance-study-v1',
        ('preparation.json','rows.jsonl','source-bank.json','preserved-slots.json'))
    plan=read(a.source_study/'preparation.json');rows=rows_at(a.source_study)
    old=next(c for c in plan['cases'] if c['id']==CASE_IDS[0])
    original=next(r for r in old['candidates'] if r['id']==SOURCE_ID)
    observed=[r for r in rows if r.get('variant')=='search' and r.get('candidate_id')==SOURCE_ID]
    if len(observed)!=1:raise ValueError('Exactly one original clear-source release required')
    row=observed[0];physical_match(original,row)
    if (not row['native_release'] or row['invalid_physics'] or captured(row)
        or row['experimental_route']['robot_contact_steps']!=0
        or original['board_twist_deg']!=135. or abs(original['values'][4]-.1094386406066069)>1e-10):
        raise ValueError('Expected exact valid 135-degree arm-free short shot')
    trace=contained(a.source_study/row['adapter_directory'],row['native_cache_file'])
    if sha(trace)!=row['native_cache_sha256']:raise ValueError('Original trace changed')
    bank=[dict(id=SOURCE_ID,source_request=copy.deepcopy(original),case_id=CASE_IDS[0],
        source_directory=row['adapter_directory'],source_row_id=row['id'],source_row_sha256=digest(row),
        source_trace_file=row['native_cache_file'],source_trace_sha256=row['native_cache_sha256'],
        strength='observed_arm_free_non_capture',source_binding=binding)]
    case=dict(id=CASE_IDS[0],original=copy.deepcopy(original),source_directory=row['adapter_directory'],historical=row,sources=copy.deepcopy(bank))
    preserved=[]
    for parent in plan['preserved_cases']:
        command=parent['original']
        found=[r for r in rows if r.get('variant')=='preserved' and r.get('context_id')==parent['id']]
        if len(found)!=3 or not all(captured(r) for r in found):raise ValueError('All prior preservation shots must have captured')
        for r in found:
            physical_match(command,r)
            if sha(contained(a.source_study/r['adapter_directory'],r['native_cache_file']))!=r['native_cache_sha256']:
                raise ValueError('Preserved source trace changed')
        preserved.append(dict(id=parent['id'],original=copy.deepcopy(command),source_directory=parent['directory'],
            directory=parent['directory'],source_row_ids=[r['id'] for r in found],source_row_sha256=[digest(r) for r in found]))
    if len(preserved)!=3:raise ValueError('Three recovered positions must be preserved')
    return [case],bank,preserved,contract,binding


def source_summary(case):
    r=case['original']
    keys=('id','values','bucket_xy_m','board_heading_offset_deg','board_twist_deg','selected_q10','selected_q14')
    return dict({k:copy.deepcopy(r[k]) for k in keys},prior_no_robot_contact=True)


def proposal_menu(case):
    old=case['original'];rows=[]
    for tilt in TILT_DELTAS_DEG:
        for gap in GAP_DELTAS_M:
            if tilt==0. and gap==0.:continue
            values=list(old['values']);values[3]+=tilt;values[4]+=gap
            rows.append(dict(id=f"{case['id']}-range-{len(rows):02d}",context_id=case['id'],
                values=values,bucket_xy_m=list(old['bucket_xy_m']),
                board_heading_offset_deg=old['board_heading_offset_deg'],board_twist_deg=old['board_twist_deg'],
                seed_q10=list(old['selected_q10']),seed_q14=list(old['selected_q14']),
                phase='search',role='diagnostic',method='clear_shot_range',intended_route=old['intended_route'],
                source_request_id=old['id'],proposal_family='gap_only' if tilt==0. else 'tilt_and_gap',
                gap_delta_m=gap,tilt_delta_deg=tilt))
    return rows


def board_joint_change(request,original):
    return max(abs(math.remainder(x-y,2*math.pi)) for x,y in zip(request['selected_q14'],original['selected_q14'],strict=True))


def admit_board_branch(original,proposal,admitted,records):
    """Keep same-board tilt on its observed joint branch; preserve rejected proofs."""
    if len(records)!=1 or len(admitted)>1:raise ValueError('Exactly one prepared setting per grid cell')
    rejected=[];keep=[]
    for request in admitted:
        change=board_joint_change(request,original)
        if proposal['tilt_delta_deg']==0. and change>SEARCH_RULE['board_branch_tolerance_rad']:
            rejected.append(dict(request=copy.deepcopy(request),board_joint_change_rad=change,reason='fixed_board_joint_branch_changed'))
        else:keep.append(request)
    logs=copy.deepcopy(records)
    if rejected:
        logs[0].update(admitted=0,admitted_request_ids=[],reason='fixed_board_joint_branch_changed',
            rejected_request_ids=[x['request']['id'] for x in rejected],board_joint_change_rad=rejected[0]['board_joint_change_rad'])
    return keep,logs,rejected


def fixed_board_solver(implementation):
    """Keep an already admitted unchanged board endpoint; not nested reference IK."""
    def solve(targets,initial,*,starts,iterations,**kwargs):
        if 'preserve_known' in kwargs:
            return implementation(targets,initial,starts=starts,iterations=iterations,**kwargs)
        return implementation(targets,initial,starts=starts,iterations=iterations,
            preserve_known=[True]*len(targets),**kwargs)
    return solve


def prepare_grid_command(engine,proposal):
    """Prefer the saved branch, preserving board joints for gap-only queries."""
    if proposal['tilt_delta_deg']!=0.:
        return engine.prepare([proposal],one_per_setting=True)
    arm=engine.adapter.resolver.solver.ik.board;implementation=arm.solve_candidates
    arm.solve_candidates=fixed_board_solver(implementation)
    try:return engine.prepare([proposal],one_per_setting=True)
    finally:arm.solve_candidates=implementation
