"""Prespecified translated-XY650 development search; old scorers stay frozen."""
import copy
import itertools
import math
from pathlib import Path
from types import SimpleNamespace

from . import clean650_contract as prior
from .corrected_contract import sha
from .learning_contract import digest

VERSION='sircus-xy650-shift-study-v1'
RULE='translated-board-and-bucket-fixed650-bearing-v1'
DX=(-.01,0.,.01)
DY=(.04,.06,.08)
HEIGHTS=(.36,.38,.40)
TILTS=(26.,28.,30.)
GAPS=(.39,.41,.43)
OFFSETS=((- .005,0.,.005),(-.5,0.,.5),(-.005,0.,.005))
STEPS=(64,128)
MAX_REQUESTS=388
SCOPE=('Translated board+bucket XY DEVELOPMENT search at fixed650mm distance and bearing, '
       'with calibrated simulated contacts, privileged geometry and held endpoints. '
       'No PPO update, noise study, held-out generalization, grasp/transit, actuator, '
       'perception or physical validation. Historical layouts/results stay unchanged.')
read=prior.read
captured=prior.captured
clean=prior.clean
statistics=prior.statistics
control_gate=prior.control_gate
qualification=prior.qualification


def load_inputs(a):
    # Reuse the exact audited original/coarse input chain, never a current policy.
    args=SimpleNamespace(**vars(a));args.profile='fine-v2'
    scene,sources,calibration,bindings=prior.load_inputs(args)
    root=Path(a.raised_study);spec=read(root/'contract.json');report=read(root/'report.json');audit=read(root/'audit-clean650.json')
    unsigned=dict(spec);fp=unsigned.pop('fingerprint',None);checks=audit.get('checks',[])
    if not (digest(unsigned)==fp=='fe28a1277b7d7e32428dd736346d9336ca274d7a0fe9c0dc11ac61bab6970036'
        and spec.get('version')==report.get('version')=='sircus-clean650-height-tilt-gap-raised-v3'
        and report.get('status')==report.get('phase')=='completed' and report.get('contract_fingerprint')==fp
        and audit.get('version')=='sircus-clean650-independent-audit-v3' and audit.get('status')=='passed'
        and audit.get('contract_fingerprint')==fp and not audit.get('failures') and len(checks)>0
        and audit.get('checks_count')==len(checks) and all(c.get('passed') is True for c in checks)
        and audit.get('report_sha256')==sha(root/'report.json') and audit.get('evidence_manifest_sha256')==sha(root/'evidence-manifest.json')
        and spec['scene']==scene and spec['calibration']==calibration):raise ValueError('Exact completed independently audited raised650 parent required')
    prior.previous.verify_manifest(root,'evidence-manifest.json')
    bindings['raised']={n:sha(root/n) for n in ('contract.json','report.json','audit-clean650.json','records.jsonl','evidence-manifest.json')}
    return scene,sources,calibration,bindings


def scene_fingerprint(scene,calibration,assets):
    return digest(dict(scene=scene,calibration=calibration,asset_sha256=assets,capture_version='sircus-calibrated-native-capture-v2'))


def query_payload(group,scene_fp):
    return dict(scene_fingerprint=scene_fp,values=group['values'],bucket_xy_m=group['bucket_xy_m'],
        distance_rule='historical-control-unchanged' if group['role'].endswith('_control') else RULE,
        source_record_id=group['source_record_id'],source_record_sha256=group['source_record_sha256'])


def bind(groups,scene_fp):
    for group in groups:group['query_fingerprint']=digest(query_payload(group,scene_fp))
    return groups


def controls(sources):
    return [dict(copy.deepcopy(g),phase='controls',query_kind='historical_control',translation_m=[0.,0.])
            for g in sources if g['role'].endswith('_control')]


def coarse(seed):
    result=[]
    for index,(dx,dy,h,t,g) in enumerate(itertools.product(DX,DY,HEIGHTS,TILTS,GAPS)):
        result.append(dict(id=f'xy-grid-{index:04d}',role='discovery',phase='coarse',query_kind='translated_grid',translation_m=[dx,dy],
            values=[seed['values'][0]+dx,seed['values'][1]+dy,h,t,g],bucket_xy_m=[seed['bucket_xy_m'][0]+dx,seed['bucket_xy_m'][1]+dy],
            source_record_id=seed['source_record_id'],source_record_sha256=seed['source_record_sha256']))
    for index,(dx,dy) in enumerate(itertools.product(DX,DY)):
        values=copy.deepcopy(seed['values']);values[:2]=[values[0]+dx,values[1]+dy]
        result.append(dict(id=f'xy-reference-{index:02d}',role='discovery',phase='coarse',query_kind='translated_reference',translation_m=[dx,dy],
            values=values,bucket_xy_m=[seed['bucket_xy_m'][0]+dx,seed['bucket_xy_m'][1]+dy],
            source_record_id=seed['source_record_id'],source_record_sha256=seed['source_record_sha256']))
    result.append(dict(copy.deepcopy(seed),id='reference-650',role='discovery',phase='coarse',query_kind='original_reference',translation_m=[0.,0.]))
    return result


def ranked(rows,count):
    def key(r):
        f=r.get('first_flight') or {};margin=f.get('mouth_clearance_m');distance=f.get('closest_opening_distance_m')
        crossing=bool(f.get('available') and f.get('first_crossing') is not None and margin is not None)
        return (-int(clean(r)),-int(captured(r)),-int(crossing),
            -(float(margin) if margin is not None else -1e9),float(distance) if distance is not None else 1e6,r['id'])
    return sorted((r for r in rows if r.get('native_release') and not r.get('invalid') and r.get('invalid_physics') is False),key=key)[:count]


def select_unique(rows,count):
    selected=[];seen=set()
    for row in ranked(rows,len(rows)):
        key=tuple(round(float(v),9) for v in row['values']+row['bucket_xy_m'])
        if key in seen:continue
        selected.append(row);seen.add(key)
        if len(selected)==count:break
    return selected


def refinement(seeds):
    groups=[]
    for s,seed in enumerate(seeds):
        for i,j,k in itertools.product(range(3),repeat=3):
            v=copy.deepcopy(seed['values']);v[2]+=OFFSETS[0][i];v[3]+=OFFSETS[1][j];v[4]+=OFFSETS[2][k]
            groups.append(dict(id=f'refine-s{s:02d}-h{i}t{j}g{k}',role='discovery',phase='refinement',query_kind='local_refinement',
                translation_m=copy.deepcopy(seed['translation_m']),values=v,bucket_xy_m=copy.deepcopy(seed['bucket_xy_m']),
                source_record_id=seed['id'],source_record_sha256=digest(seed),source_query_fingerprint=seed['query_fingerprint']))
    return groups


def verification(seeds):
    result=prior.verification(seeds)
    for g,r in zip(result,seeds):g.update(query_kind='nominal_verification',translation_m=copy.deepcopy(r['translation_m']),source_query_fingerprint=r['query_fingerprint'])
    return result


def plan(groups,phase,steps=(64,),repeats=1):
    rows=prior.plan(groups,phase,steps,repeats);lookup={g['id']:g for g in groups}
    for row in rows:
        g=lookup[row['group_id']]
        row.update({k:copy.deepcopy(g[k]) for k in ('query_fingerprint','query_kind','translation_m','source_record_id','source_record_sha256')})
        if 'source_query_fingerprint' in g:row['source_query_fingerprint']=g['source_query_fingerprint']
    return rows


def execution_fingerprint(row):
    return digest({k:row[k] for k in ('id','query_fingerprint','substeps','repeat_index','q10','q14','native_initial_q10','native_initial_q14')})


def settings():
    return dict(translations=dict(dx_m=list(DX),dy_m=list(DY),rule=RULE),
        grid=dict(heights_m=list(HEIGHTS),tilts_deg=list(TILTS),gaps_m=list(GAPS),reference=[.38,28.,.3868]),
        coarse_requests=253,translated_grid_requests=243,translated_reference_requests=9,unshifted_reference_requests=1,
        refinement_seeds=3,refinement_max_requests=81,refinement_offsets=dict(zip(('height_m','tilt_deg','gap_m'),map(list,OFFSETS))),
        verification_sources=5,verification_dedup_decimals=9,refinement_dedup_decimals=9,repeats_per_resolution=3,substeps=list(STEPS),max_requests=MAX_REQUESTS,
        minimum_clearance_m=.005,max_drift_350ms_m=.005,verification_scope='Nominal exact configurations only; no perturbation/noise study',
        selection='Unique native-valid targets; clean capture, capture, first-flight descending center-rim crossing, aperture clearance, pre-contact distance, stable ID',
        first_flight_version='sircus-xy650-first-flight-aperture-v1',scope=SCOPE)
