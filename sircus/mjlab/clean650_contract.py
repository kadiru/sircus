"""Prespecified calibrated fixed650 sweep; no noise, new PPO or path claims."""
import copy
import itertools
import math
from pathlib import Path

from . import replay650_contract as previous
from .corrected_contract import sha
from .learning_contract import digest

VERSION='sircus-clean650-height-tilt-gap-v1'
HEIGHTS=(.22,.26,.30,.34,.38,.42)
TILTS=(16.,20.,24.,28.,32.,36.)
GAPS=(.20,.24,.28,.32,.36,.40,.44,.48,.52,.56)
STEPS=(64,128)
MAX_REQUESTS=496
FINE_VERSION='sircus-clean650-height-tilt-gap-fine-v2'
RAISED_VERSION='sircus-clean650-height-tilt-gap-raised-v3'
PROFILES={
    'coarse-v1':dict(version=VERSION,heights=HEIGHTS,tilts=TILTS,gaps=GAPS,
        offsets=((- .02,0.,.02),(-2.,0.,2.),(-.02,0.,.02))),
    'fine-v2':dict(version=FINE_VERSION,heights=(.375,.38,.385),tilts=(27.,27.5,28.,28.5,29.),
        gaps=(.38,.3868,.39,.395,.40),offsets=((- .005,0.,.005),(-.25,0.,.25),(-.0025,0.,.0025))),
    'raised-v3':dict(version=RAISED_VERSION,heights=(.40,.42,.44),tilts=(26.,27.,28.),
        gaps=(.3268,.3468,.3668),offsets=((- .005,0.,.005),(-.5,0.,.5),(-.0025,0.,.0025))),
}
SCOPE=('Fixed650 DEVELOPMENT height/tilt/gap sweep with calibrated simulated contacts and held endpoints. '
       'No new PPO, noise study, held-out generalization, grasp/transit, actuator, perception or physical validation. '
       'Historical labels and the prior timestep-sensitive650 result remain unchanged.')


def read(path):return previous.read(path)


def load_inputs(a):
    scene,sources,calibration,bindings=previous.load_inputs(a.old_study,a.positive_study,a.positive_audit,a.calibration_report,a.verification_report)
    root=Path(a.replay_study);spec=read(root/'contract.json');report=read(root/'report.json');audit=read(root/'audit-replay650.json')
    payload=dict(spec);fp=payload.pop('fingerprint',None);checks=audit.get('checks',[])
    if not (fp==digest(payload)=='8ebbd58394180bbf3540228fd9725608fa0f6fdff2db799036f4f3ce875f348b'
        and report.get('status')==report.get('phase')=='completed' and report.get('contract_fingerprint')==fp
        and audit.get('version')=='sircus-calibrated-650-independent-audit-v2' and audit.get('status')=='passed'
        and audit.get('contract_fingerprint')==fp and not audit.get('failures') and len(checks)>0
        and audit.get('checks_count')==len(checks) and all(c.get('passed') is True for c in checks)
        and audit.get('report_sha256')==sha(root/'report.json') and audit.get('evidence_manifest_sha256')==sha(root/'evidence-manifest.json')
        and spec['scene']==scene and spec['calibration']==calibration):raise ValueError('Exact completed independently audited650 replay required')
    previous.verify_manifest(root,'evidence-manifest.json')
    bindings['replay']={n:sha(root/n) for n in ('contract.json','report.json','audit-replay650.json','records.jsonl','evidence-manifest.json')}
    if getattr(a,'profile','coarse-v1')!='coarse-v1':
        root=Path(a.coarse_study);spec=read(root/'contract.json');report=read(root/'report.json');audit=read(root/'audit-clean650.json')
        unsigned=dict(spec);fp=unsigned.pop('fingerprint',None);checks=audit.get('checks',[])
        if not (digest(unsigned)==fp=='28baa4e28639d0e7581561bf546a42de2a738ef4fd6a1521230d69887424f28b'
            and spec.get('version')==VERSION and report.get('version')==VERSION
            and report.get('status')==report.get('phase')=='completed' and report.get('contract_fingerprint')==fp
            and audit.get('version')=='sircus-clean650-independent-audit-v1' and audit.get('status')=='passed'
            and audit.get('contract_fingerprint')==fp and not audit.get('failures') and len(checks)>0
            and audit.get('checks_count')==len(checks) and all(c.get('passed') is True for c in checks)
            and audit.get('report_sha256')==sha(root/'report.json') and audit.get('evidence_manifest_sha256')==sha(root/'evidence-manifest.json')
            and spec['scene']==scene and spec['calibration']==calibration):raise ValueError('Exact completed independently audited coarse650 study required')
        previous.verify_manifest(root,'evidence-manifest.json')
        bindings['coarse']={n:sha(root/n) for n in ('contract.json','report.json','audit-clean650.json','records.jsonl','evidence-manifest.json')}
    return scene,sources,calibration,bindings


def coarse(seed,profile='coarse-v1'):
    config=PROFILES[profile]
    result=[]
    for i,(height,tilt,gap) in enumerate(itertools.product(config['heights'],config['tilts'],config['gaps'])):
        result.append(dict(id=f'coarse-{i:04}',role='discovery',phase='coarse',values=[*seed['values'][:2],height,tilt,gap],
            bucket_xy_m=copy.deepcopy(seed['bucket_xy_m']),source_record_id=seed['source_record_id'],source_record_sha256=seed['source_record_sha256']))
    result.append(dict(copy.deepcopy(seed),id='reference-650',role='discovery',phase='coarse'))
    return result


def captured(r):return bool(r.get('native_release') and r.get('route_success') and not r.get('invalid') and not r.get('invalid_physics'))
def clean(r):return captured(r) and r.get('route_quality')==2


def ranked(rows,count):
    rows=[r for r in rows if r.get('native_release') and not r.get('invalid') and not r.get('invalid_physics')]
    def key(r):
        margin=r.get('minimum_mouth_clearance_mm');distance=r.get('closest_opening_distance_m')
        return (-int(clean(r)),-int(captured(r)),-int(bool(r.get('whole_ball_entry')) and not r.get('external_contact_steps')),
            -(float(margin) if margin is not None else -1e9),-float(r.get('max_retention_s',0.)),
            float(distance) if distance is not None else 1e6,r['id'])
    return sorted(rows,key=key)[:count]


def refinement(seeds,profile='coarse-v1'):
    offsets=PROFILES[profile]['offsets']
    groups=[]
    for s,seed in enumerate(seeds):
        for i,j,k in itertools.product(range(3),repeat=3):
            values=copy.deepcopy(seed['values']);values[2]+=offsets[0][i];values[3]+=offsets[1][j];values[4]+=offsets[2][k]
            groups.append(dict(id=f'refine-s{s:02}-h{i}t{j}g{k}',role='discovery',phase='refinement',values=values,
                bucket_xy_m=copy.deepcopy(seed['bucket_xy_m']),source_record_id=seed['id'],source_record_sha256=digest(seed)))
    return groups


def verification_selection(rows,count):
    selected=[];seen=set()
    for row in ranked(rows,len(rows)):
        key=tuple(round(float(v),9) for v in row['values']+row['bucket_xy_m'])
        if key in seen:continue
        seen.add(key);selected.append(row)
        if len(selected)==count:break
    return selected


def verification(seeds):
    return [dict(id=f'verify-{i:02}',role='verification',phase='verification',freeze_joints=True,
        values=copy.deepcopy(r['values']),bucket_xy_m=copy.deepcopy(r['bucket_xy_m']),q10=copy.deepcopy(r['q10']),q14=copy.deepcopy(r['q14']),
        source_record_id=r['id'],source_record_sha256=digest(r)) for i,r in enumerate(seeds)]


def plan(groups,phase,steps=(64,),repeats=1):
    return [dict(id=f'{phase}-{step}-{g["id"]}-r{i}',pair_id=f'{phase}-{g["id"]}-r{i}',group_id=g['id'],role=g['role'],
        phase=phase,substeps=step,repeat_index=i,values=copy.deepcopy(g['values']),bucket_xy_m=copy.deepcopy(g['bucket_xy_m']),
        q10=copy.deepcopy(g['q10']),q14=copy.deepcopy(g['q14']),eligible=g['eligible'],geometry_row=g['geometry_row'])
        for step in steps for g in groups for i in range(repeats)]


def statistics(rows):
    result=previous.statistics(rows)
    result['rim_assisted']=sum(captured(r) and r.get('rim_band_contact') is True for r in rows)
    result['geometry_rejections']=sum(not bool(r.get('native_release')) for r in rows)
    return result


def control_gate(rows):
    complete=len(rows)==12
    grouped=lambda role,s:[r for r in rows if r['role']==role and r['substeps']==s]
    positive=complete and all(len(grouped('positive_control',s))==3 and all(clean(r) for r in grouped('positive_control',s)) for s in STEPS)
    negative=complete and all(len(grouped('negative_control',s))==3 and all(r.get('native_release') and not captured(r) and not r.get('invalid') and not r.get('invalid_physics') for r in grouped('negative_control',s)) for s in STEPS)
    physics=complete and all(r.get('native_release') and not r.get('invalid') and r.get('invalid_physics') is False for r in rows)
    oracle=complete and all(r.get('scalar_score_audit_passed') is True for r in rows)
    return dict(status='passed' if positive and negative and physics and oracle else 'failed' if complete else 'pending',
        positive_control=positive,negative_control=negative,physics_valid=physics,oracle_agreement=oracle)


def qualification(rows,positions,*,controls_passed,source_clean):
    complete=len(rows)==6
    clean_both=complete and all(len([r for r in rows if r['substeps']==s])==3 and all(clean(r) for r in rows if r['substeps']==s) for s in STEPS)
    physics=complete and all(r.get('native_release') and not r.get('invalid') and r.get('invalid_physics') is False for r in rows)
    oracle=complete and all(r.get('scalar_score_audit_passed') is True for r in rows)
    drift=previous.sensitivity(rows,positions)['nominal_position_delta_350ms_max_m']
    good=bool(controls_passed and source_clean and clean_both and physics and oracle and drift is not None and math.isfinite(drift) and drift<=.005)
    return dict(status='qualified' if good else 'failed',clean_both=clean_both,physics_valid=physics,oracle_agreement=oracle,max_drift_m=drift,qualified=good)


def settings(profile='coarse-v1'):
    config=PROFILES[profile];coarse_count=len(config['heights'])*len(config['tilts'])*len(config['gaps'])+1
    return dict(grid=dict(heights_m=list(config['heights']),tilts_deg=list(config['tilts']),gaps_m=list(config['gaps']),reference=[.38,28.,.3868]),
        coarse_requests=coarse_count,refinement_seeds=3,refinement_offsets=dict(zip(('height_m','tilt_deg','gap_m'),map(list,config['offsets']))),
        refinement_max_requests=81,verification_sources=5,verification_dedup_decimals=9,repeats_per_resolution=3,substeps=list(STEPS),max_requests=coarse_count+135,
        verification_scope='Nominal exact configurations only; no perturbation/noise study',minimum_clearance_m=.005,max_drift_350ms_m=.005,
        selection='Native valid only; clean capture, capture, whole entry without external contact, mouth clearance, retention, opening distance, stable ID',scope=SCOPE)
