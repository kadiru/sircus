"""Frozen six-hour development search; unchanged native64 physics and scoring."""
from __future__ import annotations
import copy
import math
from collections import Counter
from pathlib import Path
import numpy as np
from . import fixed35_broad_contract as old
from .benchmark_verify import digest
from .fixed35_recovery_contract import LOW,HIGH,request
from .table_training_contract import captured,statistics
from .table_assisted_score import reward

VERSION='sircus-fixed35-sixhour-v1'
INPUT_KEYS=old.INPUT_KEYS+('broad-study','broad-audit')
CASE_IDS=old.CASE_IDS
TARGET=CASE_IDS[1]
SEED=2026111201
NUMERIC_LIMIT=6000
SEARCH_RELEASE_LIMIT=1200
FAMILIES=('axis','local','table_route','lateral')
SCOPE=('Six acquired hours including audit: investigate bouncer-link clearance at fixed bucket03-05, '
       'then map local success windows if it is confirmed. Preserve all previously confirmed commands. '
       'Native64/v3, bucket-facing yaw and geometry unchanged. Development search only; no policy fitting, '
       'noise, UI, hardware, Isaac or4070 changes.')
parent=old.parent
require=old.require
confirmation_groups=old.confirmation_groups


def protocol():
    return dict(version=VERSION,substeps=64,primary_context=TARGET,maximum_numeric_proposals=NUMERIC_LIMIT,
        search_native_max=SEARCH_RELEASE_LIMIT,maximum_branches_per_numeric=8,
        repeat_per_selected=3,selected_successes_per_cell=3,repeat_max=27,controls=18,
        preserved='original30 plus every independently confirmed broad-parent command',
        acquired_total_limit_s=21600,worker_limit_s=18000,search_limit_s=16200,primary_limit_s=10800,
        queue_limit_s=18000,audit_reserve_s=3600,global_limit_s=39600,storage_reserve_bytes=8*1024**3,
        bounds=dict(low=LOW.tolist(),high=HIGH.tolist()),seed=SEED,families=list(FAMILIES),
        endpoint_and_path_clearance_m=.005,training=False,noise=False,policy_updated=False,ui_changes=False,
        confirmation='First3 distinct captured numeric settings per cell, each repeated3 fresh times.',
        ordering='Primary frozen numeric order then branch slot; after3h, switch to frozen multi-cell local windows only if primary has3/3 confirmation.',
        duplicates='Exclude exact cell/action/joint commands already flown by broad parent or this extension; repeats and controls are separate.',
        scoring='sircus-table-assisted-native-capture-v3',scope=SCOPE)


def load_inputs(a,root):
    from .fixed35_sixhour_verify import authenticate_broad,confirmed_broad_recipes,AuditChecks
    out=parent.load_inputs(a,root);checks=AuditChecks();b=authenticate_broad(a.broad_study,a.broad_audit,checks)
    require(b['report']['status']=='completed' and b['report']['fresh_search_releases']==old.QUOTAS,'Completed audited1000-drop parent required')
    require(b['spec']['source_bindings']==out['bindings'],'Broad parent original input bindings differ')
    for key in ('scene_sha256','asset_sha256'):
        require(b['spec']['identity'][key]==out['parent']['identity'][key],'Broad parent scene/assets differ')
    confirmed=confirmed_broad_recipes(b,checks)
    broad_binding={k:b[k] for k in ('report_sha256','audit_sha256','contract_fingerprint','evidence_manifest_sha256','checks_count')}
    if 'parent_acceptance' in b:broad_binding['parent_acceptance']=b['parent_acceptance']
    prior_preserved=list(out['preserved']);out['preserved']=prior_preserved+confirmed
    out.update(broad_sources=confirmed,broad_source_roots={r['id']:Path(a.broad_study)/'native-yaw' for r in confirmed},
        broad_rows=b['rows'],broad_parent=b,broad_authentication_checks=checks.entries,
        bindings=dict(historical=out['bindings'],broad=broad_binding,
            confirmed_row_sha256={r['id']:digest(r) for r in confirmed}))
    return out


def command_key(row):
    """Exact commands only; tiny changes are deliberately different experiments."""
    return digest(dict(context_id=row['context_id'],values=row['values'],q10=row.get('selected_q10',row.get('q10')),q14=row.get('selected_q14',row.get('q14'))))


def preserved_plan(sources):
    require(len(sources)>=30 and all(captured(r) for r in sources),'All confirmed sources required')
    return [request(r,identity=f'fixed35-sixhour-preserved-{i:02d}-'+digest(r)[:12],phase='preserved',method='preserved_recipe',exact_parent_action=True)
            for i,r in enumerate(sources)]


def repeat_plan(discovery,group):
    rows=old.repeat_plan(discovery,group)
    for r in rows:r.update(id=r['id'].replace('fixed35-broad-','fixed35-sixhour-',1),method='sixhour_repeat')
    return rows


def _plan(sources,seeds,scene,assets,*,interval_fn,reach_fn,limit,stage):
    require(0<limit<=NUMERIC_LIMIT,'Finite numeric limit required')
    # Independent seed and family counters prevent modulo coupling from dropping3 of4 anchors.
    targets=(TARGET,) if stage=='primary' else tuple(c for c in CASE_IDS if any(s['context_id']==c for s in seeds))
    require(targets,'At least one window source required')
    anchors={c:sorted((r for r in sources if r['context_id']==c),key=lambda r:r['id']) for c in targets}
    require(all(anchors.values()),'Every target needs its own authenticated lineage')
    choices={c:sorted((seeds if stage=='primary' else [s for s in seeds if s['context_id']==c]),key=lambda r:r['id']) for c in targets}
    require(all(choices.values()),'Missing numeric seeds')
    rng=np.random.default_rng(SEED+(0 if stage=='primary' else 1));seen=set();rows=[];reasons=Counter();families=Counter();seed_counts=Counter()
    max_raw=limit*40
    for raw in range(max_raw):
        ci=raw%len(targets);context=targets[ci];local=raw//len(targets)
        pool=choices[context];seed=pool[local%len(pool)];cycle=local//len(pool)
        family='axis' if stage=='window' and cycle%2==0 else FAMILIES[cycle%len(FAMILIES)] if stage=='primary' else 'local'
        source=anchors[context][local%len(anchors[context])];bucket=np.asarray(source['bucket_xy_m'],float)
        v=np.asarray(seed['values'],float).copy();u=rng.random(6)
        scale_index=(cycle//len(FAMILIES))%4;scale=(.01,.03,.07,.14)[scale_index]
        if family=='axis':
            axis=(cycle//4)%5;sign=1 if (cycle//20)%2==0 else -1
            delta=(.01,.03,.07)[(cycle//40)%3] if axis!=3 else (1.,3.,7.)[(cycle//40)%3]
            v[axis]+=sign*delta
        elif family=='local':v+=np.array([scale,scale,scale,scale*100,scale])*(2*u[:5]-1)
        elif family=='table_route':
            v[:2]+=.08*(2*u[:2]-1);v[2]=.25+.35*u[2];v[3]=25.+35.*u[3];v[4]+=.12*(2*u[4]-1)
        else:
            bearing=math.atan2(v[1]-bucket[1],v[0]-bucket[0])+(2*u[0]-1)*math.pi/2
            radius=.25+.50*u[1];v[:2]=bucket+radius*np.array([math.cos(bearing),math.sin(bearing)])
            v[2]+=.14*(2*u[2]-1);v[3]+=14*(2*u[3]-1);v[4]+=.18*(2*u[4]-1)
        if np.any(v<LOW) or np.any(v>HIGH):reasons['numeric_bounds']+=1;continue
        if np.linalg.norm(bucket-v[:2])<=.15:reasons['board_bucket_separation']+=1;continue
        interval=interval_fn(assets,scene,bucket,v[:3],float(v[3]),gap_bounds=(float(LOW[4]),float(HIGH[4])))
        if interval is None:reasons['wrist_reach_no_gap']+=1;continue
        lo,hi=map(float,interval);require(LOW[4]-1e-12<=lo<=hi<=HIGH[4]+1e-12,'Reach interval cannot expand bounds')
        # Do not replace an axis/local proposal with a different, silently clipped gap.
        if not lo<=v[4]<=hi:reasons['gap_outside_reach_interval']+=1;continue
        reach=reach_fn(assets,scene,bucket,v)
        if not reach['passed']:reasons['wrist_reach']+=1;continue
        key=(context,*v.tolist())
        if key in seen:reasons['duplicate_numeric']+=1;continue
        seen.add(key);index=len(rows)
        row=request(source,identity=f'fixed35-sixhour-{stage}-p{index:04d}-{context}',phase='search',method='sixhour_search',values=v.tolist(),
            exact_parent_action=False,proposal_index=index,raw_proposal_index=raw,proposal_family=family,proposal_stage=stage,
            seed_source_id=seed['id'],seed_source_sha256=digest(seed),seed_context_id=seed['context_id'],seed_values=copy.deepcopy(seed['values']),
            conditional_gap_interval_m=[lo,hi],reach_filter=copy.deepcopy(reach))
        rows.append(row);families[family]+=1;seed_counts[seed['id']]+=1
        if len(rows)==limit:break
    return dict(version=VERSION,stage=stage,seed=SEED+(stage=='window'),frozen_before_native_outcomes=True,
        limit=limit,raw_considered=raw+1,rows=rows,geometric_declines=dict(reasons),families=dict(families),seed_counts=dict(seed_counts),
        seed_sha256={s['id']:digest(s) for s in seeds},source_sha256={s['id']:digest(s) for c in targets for s in anchors[c]})


def make_plan(sources,confirmed,scene,assets,*,interval_fn,reach_fn,limit=NUMERIC_LIMIT):
    seeds=[r for r in sources if r['context_id']==TARGET]+list(confirmed)
    return _plan(sources,seeds,scene,assets,interval_fn=interval_fn,reach_fn=reach_fn,limit=limit,stage='primary')


def make_window_plan(sources,confirmed,scene,assets,*,interval_fn,reach_fn,limit):
    return _plan(sources,confirmed,scene,assets,interval_fn=interval_fn,reach_fn=reach_fn,limit=limit,stage='window')


def case_results(rows):
    result=[]
    for context in CASE_IDS:
        group=[r for r in rows if r.get('phase')=='search' and r['context_id']==context];groups=confirmation_groups(rows,context)
        ids={g['selected_from_id'] for g in groups if g['status']=='confirmed'}
        result.append(dict(context_id=context,search_counts=statistics(group),fresh_search_releases=sum(r.get('native_release') is True for r in group),
            unique_numeric_settings=len({tuple(r['values']) for r in group}),confirmed_recipes=[copy.deepcopy(r) for r in group if r['id'] in ids],confirmation_groups=groups))
    return result
