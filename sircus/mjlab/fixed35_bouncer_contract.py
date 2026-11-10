"""Bouncer-only replay of already-solved endpoint alternatives; no new IK or fitting."""
from __future__ import annotations
import copy
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from .benchmark_verify import digest, read, sha
from .corrected_imitation_contract import authenticate_study, require
from .fixed35_recovery_contract import request
from . import fixed35_six_contract as previous
from .table_training_contract import captured, clean, statistics
from .table_assisted_score import reward

VERSION='sircus-fixed35-bouncer-v1'
AUDIT_VERSION='sircus-fixed35-bouncer-audit-v1'
MAXIMUM_REQUESTS=99
SEARCH_REQUESTS=24
ROTATION=previous.ROTATION
INPUT_KEYS=('parent-study','parent-audit','original-study','original-audit','recovery-study','recovery-audit','reference','assets')
CASE_IDS=('fixed35-dev-grid-03-04','fixed35-dev-grid-03-05','fixed35-dev-grid-04-06')
PARENT_REPORT_SHA='629360baed3cd23261c125679202b8d129ffb28b9cec0c1c8709d0ef19eeeb32'
PARENT_AUDIT_SHA='ef56098a64bbd640a0c2e72aeaea1d10e7da00f853f1b9bb4b373d06c85c28e7'
SCOPE=('Bouncer-only development correction on three unresolved fixed-grid cells. Select at most two saved endpoint-valid q14 alternatives '
       'per twelve admitted numeric poses; keep exact bucket, action and q10, and recheck strict setup admission without new IK. '
       'Replay30 successful recipes unchanged; new discoveries need3 fresh captures. Native64/v3 and bucket-facing yaw unchanged. '
       'No model fitting, new scenes, noise,128 qualification, guard relaxation, hardware, UI or policy promotion.')


def protocol():
    return dict(substeps=64,grid_positions=35,historical_policy_captures=24,historical_recipe_coverage=30,
        search_layouts=3,numeric_source_poses=12,alternatives_per_pose=2,search_requests_max=24,
        repeat_per_selected=3,candidates_per_layout=3,repeat_requests_max=27,preserved_requests=30,
        control_requests=18,maximum_requests=99,worker_limit_s=3600,search_limit_s=2400,
        queue_limit_s=600,audit_reserve_s=900,global_limit_s=5400,storage_reserve_bytes=8*1024**3,
        training=False,actor_steps=0,critic_steps=0,noise=False,bucket_perturbations=False,ui_changes=False,new_ik=False,
        rotation=ROTATION,q10_preserved_exactly=True,wrapped_q14_tolerance_rad=.05,proximal_joint_indices=[0,1,2],
        selection='Per admitted numeric pose choose lowest branch slot, then id; filter exact-parent-q10 saved endpoint candidates; deduplicate wrapped q14; proximal change first, then wrist-only else next proximal; maximum2.',
        ordering='Frozen per-case candidate queues, round-robin one per active case; stop after3/3 or3 failed confirmation groups.',
        confirmation='Exact selected candidate and admitted connection/joints replayed3 fresh times.',
        evaluation_scope='Posthoc known-grid recipe development; unchanged24/35 policy score, not held-out inference',scope=SCOPE)


def difference(first,second):
    return np.abs((np.asarray(first)-np.asarray(second)+np.pi)%(2*np.pi)-np.pi)


def select_alternatives(source,pool):
    require(pool['values']==source['values'] and pool['bucket_xy_m']==source['bucket_xy_m'], 'Source numeric pool differs')
    unique=[]
    candidates=(pool.get('endpoint_attempts') or {}).get('candidates',[])
    for index,candidate in enumerate(candidates):
        require(candidate['values']==source['values'] and candidate['bucket_xy_m']==source['bucket_xy_m'], 'Saved endpoint action differs')
        if candidate['q10']!=source['q10']:continue
        delta=difference(candidate['q14'],source['q14'])
        if float(delta.max())<.05:continue
        if any(float(difference(candidate['q14'],other['candidate']['q14']).max())<.05 for other in unique):continue
        unique.append(dict(candidate_index=index,candidate=copy.deepcopy(candidate),candidate_sha256=digest(candidate),
                           proximal_changed=bool(float(delta[:3].max())>=.05),maximum_wrapped_q14_delta_rad=float(delta.max())))
    proximal=[x for x in unique if x['proximal_changed']];wrist=[x for x in unique if not x['proximal_changed']]
    chosen=[]
    if proximal:chosen.append(proximal[0])
    elif wrist:chosen.append(wrist[0])
    remaining_wrist=[x for x in wrist if x not in chosen]
    remaining_proximal=[x for x in proximal if x not in chosen]
    if remaining_wrist:chosen.append(remaining_wrist[0])
    elif remaining_proximal:chosen.append(remaining_proximal[0])
    return chosen


def numeric_sources(rows):
    grouped={}
    for row in rows:
        if row.get('phase')=='search' and row.get('context_id') in CASE_IDS and row.get('native_release'):
            key=(row['context_id'],tuple(row['values']))
            grouped.setdefault(key,[]).append(row)
    result=[min(group,key=lambda r:(r['ik_branch_slot'],r['id'])) for group in grouped.values()]
    result.sort(key=lambda r:(r['context_id'],r['numeric_proposal_index'],r['id']))
    require(len(result)==12 and {r['context_id'] for r in result}==set(CASE_IDS), 'Exactly12 admitted numeric poses across3 cells required')
    return result


def load_inputs(a,root):
    original_args=SimpleNamespace(**vars(a));original_args.parent_study=a.recovery_study;original_args.parent_audit=a.recovery_audit
    out=previous.load_inputs(original_args,root)
    parent=authenticate_study(a.parent_study,a.parent_audit,
        dict(report_sha256=PARENT_REPORT_SHA,proof_sha256=PARENT_AUDIT_SHA),checks_count=170755)
    require(len(parent['rows'])==641, 'Complete six-case641 evidence required')
    known={r['id']:r for r in out['preserved']}
    preservation=[r for r in parent['rows'] if r.get('phase')=='preserved']
    require(len(preservation)==29 and all(captured(r) for r in preservation), 'All29 prior preserved recipes captured')
    for row in preservation:
        source=known[row['parent_row_id']]
        require(row['parent_row_sha256']==digest(source) and all(row[k]==source[k] for k in ('values','bucket_xy_m','q10','q14','connection_sha256')), 'Prior preservation changed')
    confirmed=[c['confirmed_recipe'] for c in parent['report']['cases'] if c.get('confirmed_recipe') is not None]
    require(len(confirmed)==1 and confirmed[0]['context_id']=='fixed35-dev-grid-01-05','One known six-case recovery required')
    selected=confirmed[0];source=next(r for r in parent['rows'] if r['id']==selected['id'])
    require(source==selected and captured(source),'New preserved recipe must equal actual discovery')
    repeats=sorted([r for r in parent['rows'] if r.get('phase')=='repeat' and r['selected_from_id']==source['id']],key=lambda r:r['repeat'])
    require(len(repeats)==3 and [r['repeat'] for r in repeats]==[0,1,2] and all(captured(r) for r in repeats),'Three fresh confirmations of six-case recovery required')
    for row in repeats:
        require(row['selected_from_sha256']==digest(source) and all(row[k]==source[k] for k in ('values','bucket_xy_m','q10','q14','connection_sha256')), 'Six-case confirmation differs')
    preserved=sorted(out['preserved']+[source],key=lambda r:r['context_id'])
    require(len({r['context_id'] for r in preserved})==30,'Exactly30 unique preserved recipes required')
    numeric=numeric_sources(parent['rows']);pools={}
    for row in numeric+[source]:
        for key in ('context_resolution','connection','branch_pool'):
            require(sha(parent['bound']('native-yaw/'+row[key+'_file']))==row[key+'_sha256'], 'Six-case native proof differs')
        if row.get('native_release'):
            require(sha(parent['bound']('native-yaw/'+row['native_cache_file']))==row['native_cache_sha256'],'Six-case native trace differs')
        pools[row['id']]=read(parent['bound']('native-yaw/'+row['branch_pool_file']))
    cases=[r for r in out['cases'] if r['context_id'] in CASE_IDS]
    source_roots=dict(out['source_roots'])
    for row in numeric+[source]:source_roots[row['id']]=Path(a.parent_study)/'native-yaw'
    sources=preserved+numeric
    source_roots={r['id']:source_roots[r['id']] for r in sources}
    for name in ('prior.json','old-prior.json','controls.json','control-replays.json','calibration.json','setup-home.json','verification.json',
                 'native-yaw/effective-options.json','native-yaw/native-compiled-geometry.json',
                 'control-parent/native/effective-options.json','control-parent/native/native-compiled-geometry.json'):
        require(read(parent['bound'](name))==read(Path(a.recovery_study)/name),'Pinned input differs: '+name)
    require(all(parent['spec']['identity'][k]==out['parent']['identity'][k] for k in ('scene_sha256','asset_sha256')),'Six-case scene/assets differ')
    parent['bound']('geometry-audit.json')
    old_bindings=out['bindings']
    out.update(parent=parent['spec'],parent_report=parent['report'],preserved=preserved,cases=cases,
        failures=cases+out['blocked'],diagnoses=[d for d in out['diagnoses'] if d['context_id'] in CASE_IDS+previous.BLOCKED_IDS],numeric_sources=numeric,candidate_pools=pools,sources=sources,source_roots=source_roots,
        bindings=dict(parent=parent['bindings'],historical=old_bindings,
            preserved_row_sha256={r['id']:digest(r) for r in preserved},numeric_source_sha256={r['id']:digest(r) for r in numeric}))
    return out


def make_plan(numeric,pools):
    queues={c:[] for c in CASE_IDS};inventory=[]
    for source in numeric:
        chosen=select_alternatives(source,pools[source['id']])
        inventory.append(dict(source_id=source['id'],source_sha256=digest(source),context_id=source['context_id'],
            numeric_proposal_index=source['numeric_proposal_index'],values=source['values'],bucket_xy_m=source['bucket_xy_m'],
            branch_pool_file=source['branch_pool_file'],branch_pool_sha256=source['branch_pool_sha256'],selected_count=len(chosen),
            selected_candidates=[{k:v for k,v in x.items() if k!='candidate'} for x in chosen]))
        for index,item in enumerate(chosen):
            candidate=item['candidate']
            row=request(source,identity=f'fixed35-bouncer-search-p{source["numeric_proposal_index"]:02d}-a{index}-{source["context_id"]}',
                phase='search',method='bouncer_search',exact_parent_action=False,numeric_proposal_index=source['numeric_proposal_index'],
                candidate_index=item['candidate_index'],candidate_pool_file=source['branch_pool_file'],candidate_pool_sha256=source['branch_pool_sha256'],
                candidate_sha256=item['candidate_sha256'],selected_q10=copy.deepcopy(candidate['q10']),selected_q14=copy.deepcopy(candidate['q14']),
                proximal_changed=item['proximal_changed'],alternative_index=index)
            queues[source['context_id']].append(row)
    plan=[]
    for index in range(max(map(len,queues.values()),default=0)):
        for context in CASE_IDS:
            if index<len(queues[context]):
                row=copy.deepcopy(queues[context][index]);row['proposal_index']=index;plan.append(row)
    require(len(plan)<=24 and len({r['id'] for r in plan})==len(plan),'At most24 distinct saved-alternative requests')
    return plan,inventory


def preserved_plan(successes):
    require(len(successes)==30 and len({r['context_id'] for r in successes})==30 and all(captured(r) for r in successes),'Exactly30 successful recipes')
    return [request(r,identity='fixed35-bouncer-preserved-'+r['context_id'],phase='preserved',method='preserved_recipe',exact_parent_action=True) for r in successes]


def repeat_plan(discovery,confirmation_index):
    require(captured(discovery) and confirmation_index in range(3),'Captured discovery and bounded confirmation required')
    result=[]
    for repeat in range(3):
        fields={k:copy.deepcopy(discovery[k]) for k in ('proposal_index','numeric_proposal_index','candidate_index','candidate_pool_file','candidate_pool_sha256',
            'candidate_sha256','selected_q10','selected_q14','proximal_changed','alternative_index')}
        row=request(discovery,identity=f'fixed35-bouncer-repeat-c{confirmation_index}-r{repeat}-{discovery["context_id"]}',phase='repeat',method='bouncer_repeat',
            confirmation_index=confirmation_index,repeat=repeat,selected_from_id=discovery['id'],selected_from_sha256=digest(discovery),
            selected_connection_sha256=discovery['connection_sha256'],**fields)
        row.update(parent_row_id=discovery['parent_row_id'],parent_row_sha256=discovery['parent_row_sha256'])
        result.append(row)
    return result


def confirmation_groups(rows,context_id):
    groups=[]
    for index in range(3):
        members=[r for r in rows if r.get('phase')=='repeat' and r['context_id']==context_id and r['confirmation_index']==index]
        if not members:continue
        first=members[0]
        exact=len(members)==3 and sorted(r['repeat'] for r in members)==[0,1,2] and len({r['id'] for r in members})==3
        exact=exact and all(all(r.get(k)==first.get(k) for k in ('selected_from_id','selected_from_sha256','values','bucket_xy_m','q10','q14','connection_sha256','candidate_sha256')) for r in members)
        groups.append(dict(confirmation_index=index,selected_from_id=first['selected_from_id'],selected_from_sha256=first['selected_from_sha256'],
            values=first['values'],bucket_xy_m=first['bucket_xy_m'],candidate_sha256=first['candidate_sha256'],requests=len(members),captures=sum(captured(r) for r in members),
            record_ids=[r['id'] for r in members],status='confirmed' if exact and all(captured(r) for r in members) else 'failed' if len(members)==3 else 'incomplete'))
    return groups


def case_results(rows,cases,blocked,plan,*,finished=False,budget_truncated=False):
    result=[]
    for parent in sorted(cases+blocked,key=lambda r:r['context_id']):
        search=[r for r in rows if r.get('phase')=='search' and r['context_id']==parent['context_id']]
        capacity=sum(r['context_id']==parent['context_id'] for r in plan)
        groups=confirmation_groups(rows,parent['context_id']);confirmed=next((g for g in groups if g['status']=='confirmed'),None)
        status=('fixed_base_clearance_blocked' if parent in blocked else 'confirmed' if confirmed else 'confirmation_limit' if len(groups)==3 else
                'search_exhausted' if len(search)==capacity else 'budget_truncated' if finished and budget_truncated else 'in_progress')
        recipe=next((r for r in search if confirmed and r['id']==confirmed['selected_from_id']),None)
        result.append(dict(context_id=parent['context_id'],bucket_xy_m=parent['bucket_xy_m'],parent_row_id=parent['id'],parent_row_sha256=digest(parent),
            status=status,planned_search_requests=capacity,search_requests=len(search),search_captures=sum(captured(r) for r in search),search_counts=statistics(search),
            confirmation_groups=groups,confirmed_recipe=copy.deepcopy(recipe)))
    return result
