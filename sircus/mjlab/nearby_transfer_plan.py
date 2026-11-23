"""Frozen nearby-shot sources and deterministic, bounded transfer menus.

Successful source shots are evidence, not a guarantee at a different bucket.
Only warm-start joint vectors transfer; every new endpoint and path is checked.
"""
from __future__ import annotations

import copy
import json
import math
from pathlib import Path

from .benchmark_verify import contained, digest, read, sha

VERSION = 'sircus-nearby-transfer-study-v1'
CASE_IDS = tuple('parallel-board-miss-' + x for x in ('03', '09', '13', '15', '19'))
SEARCH_RULE = dict(
    version='sircus-nearby-transfer-menu-v1', transforms={'retain': 0., 'blend': .5, 'translate': 1.},
    height_delta_m=[-.04, .04], tilt_delta_deg=[-6., 6.], gap_delta_m=[-.08, .08],
    heading_delta_deg=[-8., 8.], base_seed_reserve=8, confirmed_seed_reserve=3,
    maximum_full_checks=96, multistart_retry_limit=8, multistart_starts_per_arm=16,
    multistart_iterations=180, cpu_case_limit_s=600, cpu_workers=5,
    maximum_native_candidates_per_case=20, repeats_per_success=3,
    minimum_bucket_board_distance_m=.15, maximum_total_native_releases=433,
    selection='Nearest reserved unchanged transfers, then source/family round robin; full flight prediction ranks diverse admitted commands',
    scope='Known development misses; no model training or live promotion',
)
# Zero adjustment is first; symmetric changes and small paired changes follow.
DELTAS = ((0.,0.,0.,0.), (-.04,0.,0.,0.), (.04,0.,0.,0.),
          (0.,-6.,0.,0.), (0.,6.,0.,0.), (0.,0.,-.08,0.), (0.,0.,.08,0.),
          (0.,0.,0.,-8.), (0.,0.,0.,8.),
          (0.,-3.,-.04,0.), (0.,-3.,.04,0.), (0.,3.,-.04,0.), (0.,3.,.04,0.),
          (-.02,0.,.04,0.), (.02,0.,-.04,0.))


def captured(row):
    """Actual direct/table route, irrespective of the intended source route."""
    return bool(row.get('native_release') and not row.get('invalid_physics')
                and row.get('route_success') and row.get('shot_route') in ('direct','table'))


def audited_source(study, audit_path, version, files):
    """Accept older study contracts with no label_version without weakening hashes."""
    study, audit_path = Path(study), Path(audit_path)
    report, manifest, audit, contract = (read(study/'report.json'), read(study/'evidence-manifest.json'),
                                       read(audit_path), read(study/'contract.json'))
    names = ('report.json','contract.json','evidence-manifest.json',*files)
    binding = {name:sha(study/name) for name in names}; binding['audit'] = sha(audit_path)
    if (contract['version'] != version or contract['physics_substeps'] != 64
        or report['status'] != 'completed' or report['phase'] != 'completed'
        or audit['status'] != 'passed' or audit['checks_count'] <= 0
        or audit['report_sha256'] != binding['report.json']
        or audit['evidence_manifest_sha256'] != binding['evidence-manifest.json']
        or report['evidence_manifest_sha256'] != binding['evidence-manifest.json']
        or digest({k:v for k,v in contract.items() if k!='fingerprint'}) != contract['fingerprint']
        or any(manifest[name] != binding[name] for name in ('contract.json',*files))):
        raise ValueError('Source is not identical to its completed independent audit: '+str(study))
    return report, contract, binding


def rows_at(study):
    rows = [json.loads(line) for line in (Path(study)/'rows.jsonl').read_text().splitlines() if line]
    if len({r['id'] for r in rows}) != len(rows): raise ValueError('Source duplicate release IDs')
    return rows


def physical_match(request, row):
    if (request['values'] != row['values'] or request['bucket_xy_m'] != row['bucket_xy_m']
        or request['selected_q10'] != row['q10'] or request['selected_q14'] != row['q14']):
        raise ValueError('Source action/actual joint vectors differ')
    for name in ('board_heading_offset_deg','board_twist_deg'):
        if request.get(name,0.) != row.get(name,0.): raise ValueError('Source orientation changed')


def source_entry(study_key, study, request, rows, strength, binding):
    if not rows or not all(captured(r) for r in rows): raise ValueError('Only successful direct/table sources qualify')
    directory = rows[0]['adapter_directory']
    for row in rows:
        physical_match(request,row)
        if row['adapter_directory'] != directory: raise ValueError('Source proof channel changed')
        if sha(contained(Path(study)/directory,row['native_cache_file'])) != row['native_cache_sha256']:
            raise ValueError('Source native trace changed')
    proofs = {}
    for name in ('context_resolution','prepared_connection'):
        relative,expected = request[name+'_file'],request[name+'_sha256']
        if sha(contained(Path(study)/directory,relative)) != expected: raise ValueError('Source prepared proof changed')
        proofs[relative] = expected
    return dict(id=study_key+':'+request['id'],study_key=study_key,source_request=copy.deepcopy(request),
        source_directory=directory,source_row_ids=[r['id'] for r in rows],source_row_sha256=[digest(r) for r in rows],
        source_proofs=proofs,repeat_captures=len(rows),repeat_completed=len(rows),strength=strength,
        bucket_xy_m=request['bucket_xy_m'],source_binding=binding,
        observed_routes=[r['shot_route'] for r in rows])


def source_inputs(a):
    """Extract five originals, seven confirmed recovery shots, a control and 15 singles."""
    _,target_contract,target_binding = audited_source(a.target_study,a.target_audit,
        'sircus-parallel-board-study-v1',('choices.json','layouts.json','rows.jsonl','training-prior.json','calibration.json','setup-home.json'))
    _,recovery_contract,recovery_binding = audited_source(a.recovery_study,a.recovery_audit,
        'sircus-board-pose-recovery-v1',('preparation.json','rows.jsonl'))
    _,guarded_contract,guarded_binding = audited_source(a.guarded_study,a.guarded_audit,
        'sircus-guarded-board-fallback-study-v1',('choices.json','rows.jsonl'))
    for other in (recovery_contract,guarded_contract):
        if (other['identity'],other['scene']) != (target_contract['identity'],target_contract['scene']):
            raise ValueError('Sources use different calibrated native scenes')
    target_rows=rows_at(a.target_study); recovery_rows=rows_at(a.recovery_study); guarded_rows=rows_at(a.guarded_study)
    choices=read(a.target_study/'choices.json')
    requests={s['context_id']:s['request'] for s in choices['slots'] if s['method']=='baseline'}
    layouts=read(a.target_study/'layouts.json')['layouts']
    cases=[]; bank=[]
    for layout in layouts:
        source=requests[layout['id']]
        ids={r['id'] for r in choices['requests']['baseline'] if r['context_id']==layout['id']}
        observed=[r for r in target_rows if r['id'] in ids]
        if len(observed)!=3: raise ValueError('Expected three matched source releases')
        for row in observed: physical_match(source,row)
        if layout['role']=='previous_miss':
            if any(captured(r) or r['invalid_physics'] or not r['native_release'] for r in observed):
                raise ValueError('Expected five valid source noncaptures')
            cases.append(dict(id=layout['id'],original=copy.deepcopy(source),historical=observed[0],
                historical_repeats=observed,source_directory=observed[0]['adapter_directory']))
        elif layout['role']=='known_regression':
            bank.append(source_entry('parallel',a.target_study,source,observed,'confirmed_3_of_3',target_binding))
    if tuple(c['id'] for c in cases)!=CASE_IDS: raise ValueError('Expected exact five selected development targets')
    for case in read(a.recovery_study/'preparation.json')['cases']:
        for request in case['candidates']:
            initial=[r for r in recovery_rows if r.get('candidate_id')==request['id'] and r.get('variant')=='search']
            repeats=[r for r in recovery_rows if r.get('candidate_id')==request['id'] and r.get('variant')=='repeat']
            if len(initial)==1 and captured(initial[0]) and len(repeats)==3 and all(captured(r) for r in repeats):
                entry=source_entry('recovery',a.recovery_study,request,repeats,'confirmed_3_of_3',recovery_binding)
                entry['initial_search_row_id']=initial[0]['id']; bank.append(entry)
    guarded_by_id={r['id']:r for r in guarded_rows}
    for slot in read(a.guarded_study/'choices.json')['slots']:
        request=slot['request']
        if slot['method']=='baseline' and request is not None and captured(guarded_by_id[request['id']]):
            bank.append(source_entry('guarded',a.guarded_study,request,[guarded_by_id[request['id']]],'single_capture',guarded_binding))
    bank.sort(key=lambda r:r['id'])
    if (sum(r['study_key']=='recovery' for r in bank),sum(r['study_key']=='parallel' for r in bank),
        sum(r['study_key']=='guarded' for r in bank)) != (7,1,15):
        raise ValueError('Expected seven recovered, one control, and fifteen single-capture source settings')
    bindings=dict(target=target_binding,recovery=recovery_binding,guarded=guarded_binding)
    return cases,bank,target_contract,bindings


def nearest_sources(case,bank):
    xy=case['original']['bucket_xy_m']
    return sorted(bank,key=lambda s:(math.dist(xy,s['bucket_xy_m']),s['id']))


def reserved_sources(case,bank):
    ordered=nearest_sources(case,bank)
    confirmed=[s for s in ordered if s['strength']=='confirmed_3_of_3'][:SEARCH_RULE['confirmed_seed_reserve']]
    ids={s['id'] for s in confirmed}
    return sorted((confirmed+[s for s in ordered if s['id'] not in ids])[:SEARCH_RULE['base_seed_reserve']],
                  key=lambda s:(math.dist(case['original']['bucket_xy_m'],s['bucket_xy_m']),s['id']))


def proposal_menu(case,bank,low,high):
    """No proof or selected joints are copied into a proposed new command."""
    target=case['original']['bucket_xy_m']; result=[]
    for seed in nearest_sources(case,bank):
        source=seed['source_request']; old=source['values']; delta=[target[i]-source['bucket_xy_m'][i] for i in (0,1)]
        for mode,fraction in SEARCH_RULE['transforms'].items():
            for delta_index,(height,tilt,gap,heading) in enumerate(DELTAS):
                values=[old[0]+fraction*delta[0],old[1]+fraction*delta[1],old[2]+height,old[3]+tilt,old[4]+gap]
                angle=source.get('board_heading_offset_deg',0.)+heading
                if any(v<lo or v>hi for v,lo,hi in zip(values,low,high,strict=True)) or abs(angle)>180:
                    continue
                if math.dist(values[:2],target)<=SEARCH_RULE['minimum_bucket_board_distance_m']: continue
                result.append(dict(id=f"{case['id']}-transfer-{len(result):04d}",context_id=case['id'],
                    values=values,bucket_xy_m=list(target),board_heading_offset_deg=angle,
                    board_twist_deg=source.get('board_twist_deg',0.),phase='search',role='diagnostic',method='nearby_transfer',
                    intended_route=source.get('intended_route','direct'),group=mode,
                    seed_q10=list(source['selected_q10']),seed_q14=list(source['selected_q14']),
                    source_request_id=source['id'],source_bank_id=seed['id'],source_strength=seed['strength'],
                    source_distance_m=math.dist(target,source['bucket_xy_m']),transfer_mode=mode,transfer_fraction=fraction,
                    parameter_delta=dict(height_m=height,tilt_deg=tilt,gap_m=gap,heading_deg=heading),delta_index=delta_index))
    return result


def preparation_selection(case,bank,proposals,nominal,wrist):
    """Reserve exact teacher transforms, then balance seed and transform families."""
    from .board_pose_recovery_study import score_key
    available=[p for p in proposals if wrist[p['id']]['passed']]
    reserve={s['id'] for s in reserved_sources(case,bank)}
    chosen=[p for p in available if p['source_bank_id'] in reserve and p['delta_index']==0]
    seen={p['id'] for p in chosen}
    seeds=nearest_sources(case,bank)
    groups=[sorted([p for p in available if p['source_bank_id']==s['id'] and p['transfer_mode']==mode and p['id'] not in seen],
                   key=lambda p:score_key(p,nominal[p['id']])) for s in seeds for mode in SEARCH_RULE['transforms']]
    for i in range(max(map(len,groups),default=0)):
        chosen.extend(g[i] for g in groups if i<len(g))
    return chosen[:SEARCH_RULE['maximum_full_checks']]


def candidate_selection(rows,predictions,limit=20):
    from .board_pose_recovery_study import diversified, geometry_key, score_key
    # Keep up to eight unadjusted transferred teachers, including three with
    # confirmed source repeats. A noisy flight surrogate must not discard all
    # of the empirically successful starting configurations before testing.
    zero=[r for r in rows if r['delta_index']==0]
    by_seed={}
    for row in sorted(zero,key=lambda r:score_key(r,predictions[r['id']])):
        by_seed.setdefault(row['source_bank_id'],row)
    nearest=sorted(by_seed.values(),key=lambda r:(r['source_distance_m'],r['source_bank_id']))
    confirmed=[r for r in nearest if r['source_strength']=='confirmed_3_of_3'][:3]
    reserved_ids={r['source_bank_id'] for r in confirmed}
    reserve=(confirmed+[r for r in nearest if r['source_bank_id'] not in reserved_ids])[:8]
    # Also retain each transfer family when admitted, preferring confirmed
    # source evidence, then the frozen prediction score.
    for mode in SEARCH_RULE['transforms']:
        family=[r for r in zero if r['transfer_mode']==mode]
        if family:reserve.append(min(family,key=lambda r:(r['source_strength']!='confirmed_3_of_3',score_key(r,predictions[r['id']]))))
    selected=[];seen=set()
    for row in reserve+diversified(rows,predictions,len(rows)):
        geometry=geometry_key(row)
        if geometry not in seen:
            selected.append(row);seen.add(geometry)
        if len(selected)>=limit:break
    return selected


def bank_summary(bank):
    return dict(settings=len(bank),confirmed_settings=sum(s['strength']=='confirmed_3_of_3' for s in bank),
        single_capture_settings=sum(s['strength']=='single_capture' for s in bank),
        sources=[dict(id=s['id'],bucket_xy_m=s['bucket_xy_m'],study_key=s['study_key'],strength=s['strength'],
            repeat_captures=s['repeat_captures'],repeat_completed=s['repeat_completed'],
            report_sha256=s['source_binding']['report.json'],audit_sha256=s['source_binding']['audit'],
            values=s['source_request']['values'],board_heading_offset_deg=s['source_request'].get('board_heading_offset_deg',0.),
            board_twist_deg=s['source_request'].get('board_twist_deg',0.)) for s in bank])
