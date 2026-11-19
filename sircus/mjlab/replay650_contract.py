"""Frozen matched engineering replay; no policy training or historical relabeling."""
import copy
from collections import Counter
import json
from pathlib import Path

import numpy as np

from .corrected_contract import sha
from .learning_contract import digest

VERSION='sircus-calibrated-650-replay-v1'
OLD_FINGERPRINT='c4e59119e71f724b1e1941ae080e40c8aa001d954356715188895803fcd41a84'
SUBSTEPS=(64,128)
GAPS=(.3768,.3808,.3848,.3868,.3888)
MAX_REQUESTS=48
REPLAY_EXTRA_PACKAGES={'scipy':'1.15.3'}
SCOPE=('Fresh matched calibrated-scoring engineering replay and conditional gap window. '
    'Privileged geometry, prescribed held endpoints, strict fresh5mm endpoint checks; '
    'no learned grasp/transit/actuator motion, new RL, held-out evaluation or physical validation. '
    'Old shallow-study misses remain under their original scoring definition.')


def read(path):return json.loads(Path(path).read_text())


def validate_replay_packages(versions):
    from .corrected_contract import PINNED_PACKAGES,validate_packages
    validate_packages(versions)
    if set(versions)!=set(PINNED_PACKAGES)|set(REPLAY_EXTRA_PACKAGES) or any(versions.get(k)!=v for k,v in REPLAY_EXTRA_PACKAGES.items()):
        raise ValueError('Exact historical replay dependency set required, including SciPy1.15.3')
    return dict(versions)


def verify_manifest(root,name):
    root=Path(root).resolve();manifest=read(root/name)
    for key,expected in manifest.items():
        path=(root/key).resolve()
        if not path.is_relative_to(root) or not path.is_file() or sha(path)!=expected:
            raise ValueError('Changed source evidence: '+key)
    return manifest


def load_inputs(old_study,positive_study,positive_audit,calibration_report,verification_report):
    from .calibrated_benchmark_contract import checked_calibration
    old_study,positive_study=Path(old_study),Path(positive_study)
    old,report,audit=(read(old_study/n) for n in ('contract.json','report.json','audit-shallow.json'))
    payload=dict(old);fp=payload.pop('fingerprint',None);checks=audit.get('checks',[])
    if not (digest(payload)==fp==OLD_FINGERPRINT==report.get('contract_fingerprint')==audit.get('contract_fingerprint')
            and report.get('status')==report.get('phase')=='completed' and audit.get('status')=='passed'
            and audit.get('checks_count')==len(checks)>0 and all(c.get('passed') is True for c in checks) and not audit.get('failures')):
        raise ValueError('Exact completed independently audited shallow parent required')
    verify_manifest(old_study,'evidence-manifest.json')
    rows=[json.loads(line) for line in (old_study/'records.jsonl').read_text().splitlines() if line]
    sources=[]
    for ident,role,gid in [('shallow-shot-000024','candidate','candidate-650'),('shallow-shot-000026','negative_control','negative-control')]:
        r=next(r for r in rows if r['id']==ident)
        if r['values'][2:]!=[.38,28.,.3868] or not r['native_release'] or r.get('invalid'):
            raise ValueError('Requested historical shot identity changed')
        sources.append(dict(id=gid,role=role,source_record_id=ident,source_record_sha256=digest(r),
            **{k:copy.deepcopy(r[k]) for k in ('values','bucket_xy_m','q10','q14')}))
    preflight,proof=read(positive_study/'report.json'),read(positive_audit)
    if not (preflight.get('status')=='passed' and preflight.get('synthetic_fixture') is False
            and preflight.get('controls_evaluated')==preflight.get('controls_captured')==17
            and proof.get('status')=='passed' and not proof.get('failures')
            and type(proof.get('checks')) is int and proof['checks']>0
            and proof.get('identity')==preflight.get('identity')
            and proof.get('report_sha256')==sha(positive_study/'report.json')
            and proof.get('study_manifest_sha256')==sha(positive_study/'manifest.json')):
        raise ValueError('Verified native positive-control study required')
    verify_manifest(positive_study,'manifest.json')
    positive=read(positive_study/'controls.json')['rows']
    if len(positive)!=17 or not all(r.get('native_release') and r.get('route_success') and not r.get('invalid') for r in positive):
        raise ValueError('All exact historical calibrated positive controls must have captured')
    r=positive[0]  # Declared original ordering, fixed before this replay's outcomes.
    sources.insert(0,dict(id='positive-control',role='positive_control',source_record_id=r['id'],source_record_sha256=digest(r),
        **{k:copy.deepcopy(r[k]) for k in ('values','bucket_xy_m','q10','q14')}))
    calibration=checked_calibration(calibration_report,verification_report)
    identity=preflight['identity']
    reference_scene=copy.deepcopy(old['scene'])
    reference_scene['bucket']['center_xy_m']=[.67,-.42]
    if identity.get('calibration')!=calibration or identity.get('scene_sha256')!=digest(reference_scene):
        raise ValueError('Positive controls used another calibration or scene outside the exact default-bucket metadata exception')
    if identity.get('asset_sha256')!=old['asset_sha256']:
        raise ValueError('Positive control assets differ from historical shot assets')
    files=dict(old={n:sha(old_study/n) for n in ('contract.json','report.json','audit-shallow.json','records.jsonl','evidence-manifest.json')},
        positive={n:sha(positive_study/n) for n in ('report.json','controls.json','manifest.json')},positive_audit_sha256=sha(positive_audit),
        scene_translation_metadata=dict(historical_scene_sha256=digest(old['scene']),control_reference_scene_sha256=digest(reference_scene),
            historical_default_bucket_xy_m=old['scene']['bucket']['center_xy_m'],control_reference_default_bucket_xy_m=[.67,-.42],
            scope='Only the unused default bucket translation differs; each actual request preserves its explicit bucket XY and native collider guard. No dimensions, materials or other scene fields differ.'))
    return copy.deepcopy(old['scene']),sources,calibration,files


def metadata(group,scene):
    values=group['values'];return dict(**{k:copy.deepcopy(group[k]) for k in ('id','role','source_record_id','source_record_sha256')},
        values=copy.deepcopy(values),gap_m=values[4],board_height_m=values[2],tilt_deg=values[3],
        release_center_height_m=values[2]+values[4]+scene['ball']['radius_m'],
        board_xy_m=values[:2],bucket_xy_m=copy.deepcopy(group['bucket_xy_m']),
        distance_m=float(np.linalg.norm(np.asarray(group['bucket_xy_m'])-values[:2])))


def plan(groups,phase):
    return [dict(id=f'replay650-{step}-{g["id"]}-r{i}',pair_id=f'{g["id"]}-r{i}',group_id=g['id'],role=g['role'],
        substeps=step,repeat_index=i,phase=phase,values=copy.deepcopy(g['values']),bucket_xy_m=copy.deepcopy(g['bucket_xy_m']),
        q10=copy.deepcopy(g['q10']),q14=copy.deepcopy(g['q14']),eligible=g['eligible'],geometry_row=g['geometry_row'])
        for step in SUBSTEPS for g in groups for i in range(3)]


def statistics(rows):
    hits=[r for r in rows if r.get('native_release') and r.get('route_success') and not r.get('invalid') and not r.get('invalid_physics')]
    return dict(attempts=len(rows),native_releases=sum(bool(r.get('native_release')) for r in rows),captures=len(hits),
        invalid=sum(bool(r.get('invalid') or r.get('invalid_physics')) for r in rows),clean=sum(r.get('route_quality')==2 for r in hits),
        whole_entries=sum(bool(r.get('native_release') and r.get('whole_ball_entry')) for r in rows),
        max_retention_s=max((r.get('max_retention_s',0.) for r in rows),default=0.))


def sensitivity(rows,positions):
    paired={}
    for row in rows:paired.setdefault(row['pair_id'],{})[row['substeps']]=row
    full=[p for p in paired.values() if set(p)==set(SUBSTEPS)]
    deltas=[]
    for p in full:
        if all(r['id'] in positions for r in p.values()):
            deltas.append(float(np.linalg.norm(np.asarray(positions[p[64]['id']][round(.350*240*64)],dtype=float)-
                np.asarray(positions[p[128]['id']][round(.350*240*128)],dtype=float))))
    return dict(paired_attempts=len(full),capture_agreement=sum(bool(p[64]['route_success'])==bool(p[128]['route_success']) for p in full),
        nominal_position_delta_350ms_max_m=max(deltas) if len(deltas)==len(full) and deltas else None)


def replay_gate(rows,positions):
    initial=[r for r in rows if r['phase']=='replay'];complete=len(initial)==18
    def capture_count(role,step):return statistics([r for r in initial if r['role']==role and r['substeps']==step])
    checks={role:complete and all(capture_count(role,s)['attempts']==3 and capture_count(role,s)['native_releases']==3
        and capture_count(role,s)['captures']==wanted and capture_count(role,s)['invalid']==0 for s in SUBSTEPS)
        for role,wanted in [('positive_control',3),('negative_control',0),('candidate',3)]}
    physics=complete and all(r.get('native_release') and not r.get('invalid') and r.get('invalid_physics') is False for r in initial)
    oracle=complete and all(r.get('scalar_score_audit_passed') is True for r in initial)
    delta=sensitivity([r for r in initial if r['role']=='candidate'],positions)['nominal_position_delta_350ms_max_m']
    passed=complete and all(checks.values()) and physics and oracle and delta is not None and delta<=.005
    return dict(status='passed' if passed else 'failed' if complete else 'pending',**checks,physics_valid=physics,
        oracle_agreement=oracle,max_drift_m=delta,window_opened=passed)


def settings():
    return dict(candidate=dict(board_height_m=.38,tilt_deg=28.,gap_m=.3868,release_center_height_m=.8043,distance_m=.65),
        substeps=list(SUBSTEPS),gaps_m=list(GAPS),repeats_per_group=3,max_requests=MAX_REQUESTS,
        gate=dict(positive_captures=3,negative_captures=0,candidate_captures=3,each_resolution=True,
            required_clearance_m=.005,max_drift_350ms_m=.005,valid_physics=True,oracle_agreement=True),
        scope=SCOPE,window_scope='Five fixed gaps, three nominal trials per gap and resolution; no fresh perturbation robustness or learning claim')
