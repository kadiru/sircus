"""Frozen reach-conditioned exploration at the three unresolved fixed cells."""
from __future__ import annotations
import copy
import math
from collections import Counter
import numpy as np
from . import fixed35_bouncer_contract as parent
from .benchmark_verify import digest
from .fixed35_recovery_contract import LOW, HIGH, request
from .table_training_contract import captured, statistics
from .table_assisted_score import reward

VERSION = 'sircus-fixed35-broad1000-v1'
INPUT_KEYS = parent.INPUT_KEYS
CASE_IDS = parent.CASE_IDS
QUOTAS = dict(zip(CASE_IDS, (334, 333, 333)))
SEED = 2026111101
PROPOSALS_PER_CELL = 4000
RAW_PER_CELL = 40000
MAXIMUM_RELEASES = 1075
FAMILIES = ('cartesian', 'lower_farther', 'bucket_polar', 'local_admitted')
SCOPE = ('Reach-conditioned search at three existing bucket locations, with diverse bouncer joint branches, '
         'lower board/release heights and 1000 new native64 drops. No policy fitting, new bucket layouts, '
         'noise, UI update, guard relaxation, hardware, or Isaac changes.')
require = parent.require
load_inputs = parent.load_inputs


def protocol():
    return dict(version=VERSION, substeps=64, search_native_quotas=QUOTAS,
        search_native_target=1000, maximum_native_releases=MAXIMUM_RELEASES,
        proposals_per_cell=PROPOSALS_PER_CELL, raw_proposals_per_cell=RAW_PER_CELL,
        proposal_seed=SEED, families=list(FAMILIES), maximum_branches_per_numeric=2,
        repeat_per_selected=3, selected_successes_per_cell=3, repeat_max=27,
        preserved=30, controls=18, worker_limit_s=43200, search_limit_s=36000,
        queue_limit_s=900, audit_reserve_s=7200, global_limit_s=51300,
        storage_reserve_bytes=8*1024**3, bounds=dict(low=LOW.tolist(), high=HIGH.tolist()),
        search_order='Round-robin cells, increasing frozen numeric proposal index, then admitted bouncer branch slot.',
        confirmation='First three distinct captured numeric settings per cell, each repeated three fresh times; discovery continues after success.',
        gap_selection='Conditional geometric wrist-reach cap; alternating uniform and low-biased samples above the unchanged50mm minimum.',
        endpoint_and_path_clearance_m=.005, training=False, noise=False,
        policy_updated=False, ui_changes=False, restart_in_place=False,
        scoring='sircus-table-assisted-native-capture-v3', scope=SCOPE)


def make_plan(sources, scene, assets, *, cap_fn, reach_fn,
              per_cell=PROPOSALS_PER_CELL, raw_limit=RAW_PER_CELL):
    """Freeze all numeric candidates before flight; no outcome is consulted.

    Sources provide historical geometry hints only. Every fifth coordinate is
    selected from the dropper's remaining geometric reach interval. Full IK,
    collisions and sampled paths are still required by the native preparer.
    """
    require(0 < per_cell <= PROPOSALS_PER_CELL and per_cell <= raw_limit <= RAW_PER_CELL,
            'Finite candidate limits required')
    rows=[];summaries=[]
    for ci, context in enumerate(CASE_IDS):
        hints=sorted((r for r in sources if r['context_id']==context),key=lambda r:r['id'])
        require(hints,'Each fixed location needs authenticated geometry hints')
        bucket=np.asarray(hints[0]['bucket_xy_m'],float)
        require(all(r['bucket_xy_m']==bucket.tolist() for r in hints),'Fixed bucket mismatch')
        rng=np.random.default_rng(SEED+ci);seen=set();accepted=[];reasons=Counter();families=Counter()
        for raw_index in range(raw_limit):
            family=FAMILIES[raw_index%4];u=rng.random(6);source=hints[raw_index%len(hints)]
            if family=='cartesian':
                v=LOW+(HIGH-LOW)*u[:5]
            elif family=='lower_farther':
                # Larger X is farther from the dropper base and toward the board arm.
                v=np.array([.55+.49*u[0], -.5+.85*u[1], .12+.24*u[2], 8.+52.*u[3], .05])
            elif family=='bucket_polar':
                bearing=-math.pi+2*math.pi*u[0];radius=.16+.94*u[1]
                xy=bucket+radius*np.array([math.cos(bearing),math.sin(bearing)])
                v=np.r_[xy,.12+.48*u[2],8.+52.*u[3],.05]
            else:
                v=np.asarray(source['values'],float)+np.array([.22,.22,.20,22.,0.])*(2*u[:5]-1)
                v[4]=.05
            v[4]=LOW[4]
            if np.any(v<LOW) or np.any(v>HIGH):reasons['numeric_bounds']+=1;continue
            if np.linalg.norm(bucket-v[:2])<=.15:reasons['board_bucket_separation']+=1;continue
            cap=cap_fn(assets,scene,bucket,v[:3],float(v[3]),gap_bounds=(float(LOW[4]),float(HIGH[4])))
            if cap is None or not np.isfinite(cap) or cap<LOW[4]:reasons['wrist_reach_no_gap']+=1;continue
            require(cap<=HIGH[4]+1e-12,'Gap cap cannot expand historical bounds')
            fraction=u[4]**2 if (raw_index//4)%2 else u[4]
            v[4]=float(LOW[4]+fraction*(cap-LOW[4]))
            reach=reach_fn(assets,scene,bucket,v)
            if not reach['passed']:reasons['wrist_reach']+=1;continue
            identity=tuple(v.tolist())
            if identity in seen:reasons['duplicate_numeric']+=1;continue
            seen.add(identity);index=len(accepted)
            row=request(source,identity=f'fixed35-broad-p{index:04d}-{context}',phase='search',
                method='broad_search',values=v.tolist(),exact_parent_action=False,
                proposal_index=index,raw_proposal_index=raw_index,proposal_family=family,
                conditional_gap_cap_m=float(cap),reach_filter=copy.deepcopy(reach))
            accepted.append(row);families[family]+=1
            if len(accepted)==per_cell:break
        rows.extend(accepted)
        summaries.append(dict(context_id=context,accepted_numeric=len(accepted),raw_considered=raw_index+1,
            geometric_declines=dict(reasons),families=dict(families),proposal_limit_reached=len(accepted)==per_cell))
    return dict(version=VERSION,seed=SEED,frozen_before_native_outcomes=True,
                requested_per_cell=per_cell,raw_limit_per_cell=raw_limit,rows=rows,cells=summaries)


def preserved_plan(sources):
    return [dict(r,id=r['id'].replace('fixed35-bouncer-','fixed35-broad-',1))
            for r in parent.preserved_plan(sources)]


def repeat_plan(discovery, group):
    require(captured(discovery) and 0<=group<3,'Captured discovery and bounded group required')
    result=[]
    for index in range(3):
        # Start with the exact prepared packet, not a new IK request.
        row=copy.deepcopy(discovery['prepared_request'])
        row.update(id=f'fixed35-broad-repeat-g{group}-r{index}-{discovery["context_id"]}',
            phase='repeat',method='broad_repeat',confirmation_index=group,repeat=index,
            selected_from_id=discovery['id'],selected_from_sha256=digest(discovery),
            selected_connection_sha256=discovery['connection_sha256'],
            selected_q10=copy.deepcopy(discovery['q10']),selected_q14=copy.deepcopy(discovery['q14']))
        result.append(row)
    return result


def confirmation_groups(rows,context):
    result=[]
    for group in range(3):
        members=[r for r in rows if r.get('phase')=='repeat' and r['context_id']==context and r['confirmation_index']==group]
        if not members:continue
        first=members[0]
        exact=len(members)==3 and sorted(r['repeat'] for r in members)==[0,1,2]
        exact=exact and all(all(r.get(k)==first.get(k) for k in ('selected_from_id','selected_from_sha256','values','bucket_xy_m','q10','q14','connection_sha256')) for r in members)
        result.append(dict(group=group,selected_from_id=first['selected_from_id'],requests=len(members),
            captures=sum(captured(r) for r in members),status='confirmed' if exact and all(captured(r) for r in members) else 'failed' if len(members)==3 else 'incomplete'))
    return result


def case_results(rows):
    result=[]
    for context in CASE_IDS:
        group=[r for r in rows if r.get('phase')=='search' and r['context_id']==context]
        native=[r for r in group if r.get('native_release')]
        confirms=confirmation_groups(rows,context)
        confirmed_ids={g['selected_from_id'] for g in confirms if g['status']=='confirmed'}
        result.append(dict(context_id=context,quota=QUOTAS[context],search_counts=statistics(group),
            fresh_search_releases=len(native),unique_numeric_settings=len({tuple(r['values']) for r in native}),
            exact_unique_joint_pairs=len({tuple(r['q10']+r['q14']) for r in native}),
            distinct_q14_quantized_50mrad=len({tuple(np.rint(((np.asarray(r['q14'])+np.pi)%(2*np.pi)-np.pi)/.05).astype(int).tolist()) for r in native}),
            confirmed_recipes=[copy.deepcopy(r) for r in group if r['id'] in confirmed_ids],
            confirmation_groups=confirms))
    return result
