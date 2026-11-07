"""Post-hoc CPU inspection of authenticated frozen ensemble flights.

No simulator, CUDA, model fitting, model activation or contact rescoring. The
original independent audit owns success labels; an independent scalar scan
reconstructs the contact-limited first-flight diagnostics from dense evidence.
"""
from collections import Counter
import argparse
import copy
import json
from pathlib import Path

import numpy as np

from .curate_physics_ensemble import (verified_inputs, read, sha, digest,
    relative_file, REPORT_SHA, AUDIT_SHA, SOURCE_FP)
from .xy650_verify import first_flight

VERSION = 'sircus-ensemble-flight-diagnosis-v1'
AUDIT_VERSION = 'sircus-ensemble-flight-diagnosis-checks-v1'
KEY = 'ensemble_flight_diagnosis'
SCOPE = ('Post-hoc diagnosis of original fixed64 outcomes; no new training or '
    'native requests. All predictions precede original outcomes. These TEST '
    'layouts are now development evidence, not new held-out evaluation. '
    'Scores and member spread are uncalibrated. Conditional ballistic curves '
    'use the observed rebound origin and gravity, with no contacts; they are '
    'not full learned rollouts or capture predictions. Old results unchanged.')


def ballistic(origin, velocity, times, gravity):
    """A conditional free-flight diagnostic, never a simulator trajectory."""
    p, v, t, g = map(lambda a: np.asarray(a, float), (origin, velocity, times, gravity))
    if p.shape != (3,) or v.shape != (3,) or g.shape != (3,) or t.ndim != 1 or len(t) < 2:
        raise ValueError('Finite 3D references and at least two times required')
    if not all(np.isfinite(a).all() for a in (p, v, t, g)) or t[0] < 0 or np.any(np.diff(t) <= 0):
        raise ValueError('Finite increasing nonnegative times required')
    return (p + t[:, None]*v + .5*t[:, None]**2*g).tolist()


def bins(rows):
    """All requested slots count, including setup rejections."""
    result = []
    for low, high in ((0., .5), (.5, .9), (.9, .99), (.99, 1.00000001)):
        group = [r for r in rows if low <= r['prediction']['capture_score'] < high]
        result.append(dict(low=low, high=min(1., high), requested=len(group),
            captures=sum(r['observed']['capture'] for r in group),
            native_misses=sum(r['observed']['native_release'] and not r['observed']['capture'] for r in group),
            rejected=sum(not r['observed']['native_release'] for r in group),
            mean_score=float(np.mean([r['prediction']['capture_score'] for r in group])) if group else None))
    return result


def selected_ids(misses):
    top = sorted(misses, key=lambda r: (-r['prediction']['capture_score'], r['id']))[:3]
    ids = [r['id'] for r in top]
    for suffix in ('36', '35', '40'):
        row = next((r for r in misses if r['context_id'] == 'physics-ensemble-test-'+suffix), None)
        if row and row['id'] not in ids:
            ids.append(row['id'])
    return ids


def classification(observed):
    if not observed['native_release']:
        return 'Setup rejected; no native ball flight'
    if observed['capture']:
        return 'Whole-ball entry and one-second capture observed'
    if observed['whole_entry']:
        return 'Entry observed but capture failed'
    channels = observed.get('cutoff_contact_channels') or []
    crossing = observed.get('rim_crossing')
    if crossing is not None and observed.get('mouth_clearance_m') is not None:
        return 'No whole entry; pre-contact center-at-rim clearance %.1f mm' % (1000*observed['mouth_clearance_m'])
    if channels:
        return 'No whole entry; first free flight ends at '+', '.join(channels)+' contact before an observed descending center-at-rim crossing'
    return 'No whole entry; no contact-limited descending rim crossing observed'


def prediction(raw):
    return dict(capture_score=float(raw['capture_probability']),
        disagreement=float(raw['uncertainty']), aperture_distance_m=float(raw['aperture_distance_m']),
        rebound_velocity_m_s=list(raw['rebound_velocity_m_s']),
        ranking_score=float(raw['ranking_score']),
        member_capture_scores=list(raw['member_capture_probability']))


def support(row, train):
    # Exactly the model's fixed normalization; this is proximity, not a density
    # estimate, calibrated uncertainty or proof of interpolation/extrapolation.
    x = np.r_[row['values'], row['bucket_xy_m']]
    x = (x-np.array([.67,-.2,.35,30.,.3,.67,-.2]))/np.array([.57,.45,.3,30.,.3,.57,.45])
    features = np.array([t['x'] for t in train]); distances = np.linalg.norm(features-x, axis=1)
    index = int(np.argmin(distances)); nearest = train[index]
    capture_indices = [i for i,t in enumerate(train) if t['mask'][3] and t['y'][3] == 1]
    capture_index = min(capture_indices, key=lambda i: (distances[i], train[i]['id']))
    return dict(nearest_normalized_distance=float(distances[index]), nearest_train_id=nearest['id'],
        nearest_train_capture=bool(nearest['mask'][3] and nearest['y'][3]),
        nearest_capture_distance=float(distances[capture_index]), nearest_capture_train_id=train[capture_index]['id'],
        outside_training_feature_range=(np.flatnonzero((x<features.min(0))|(x>features.max(0)))).tolist(),
        scope='Seven-dimensional Euclidean proximity in fixed model normalization; not a reliability bound or causal explanation')


def make_row(row, train):
    obs = dict(native_release=row['native_release'], capture=bool(row['route_success']),
        whole_entry=bool(row['whole_ball_entry']), retention_s=float(row['max_retention_s']),
        invalid_physics=row['invalid_physics'], board_bounce=bool(row['board_bounce']),
        pose_gate=copy.deepcopy(row.get('pose_gate')), max_board_drift_mm=row.get('max_board_drift_mm'),
        bucket_collider_error_m=row.get('bucket_collider_error_m'))
    return dict(id=row['id'], context_id=row['context_id'], record_id=row['id'], record_sha256=digest(row),
        method=row['method'], values=list(row['values']), bucket_xy_m=list(row['bucket_xy_m']),
        board_bucket_distance_m=float(np.linalg.norm(np.array(row['values'][:2])-row['bucket_xy_m'])),
        prediction=prediction(row['model_prediction']), observed=obs, training_support=support(row, train))


def build(study, audit, output):
    output = Path(output).resolve(); study = Path(study).resolve()
    if output.is_relative_to(study) or output.exists():
        raise ValueError('New derived output outside immutable evidence required')
    study, report, spec, proof, manifest, packet, rows = verified_inputs(study, audit)
    targets = {t['id']:t for t in read(study/'evaluation-targets.json')}
    train = read(study/'training-dataset.json')
    if len(train) != 1024 or any(t['role'] != 'train' for t in train):
        raise ValueError('Original TRAIN-only proximity source required')
    test = [r for r in rows if r['role'] == 'test']
    items = {r['id']:make_row(r, train) for r in test}
    checks = []
    def check(ok, label):
        checks.append(dict(label=label, passed=bool(ok)))
        if not ok: raise ValueError(label)
    ensemble = [items[r['id']] for r in test if r['method']=='ensemble']
    misses = [r for r in ensemble if r['observed']['native_release'] and not r['observed']['capture']]
    check(len(ensemble)==60 and len(misses)==25, 'Exact original 60 ensemble requests and25 native misses')
    case_ids = selected_ids(misses); case_contexts={items[k]['context_id'] for k in case_ids}
    selected = [r for r in test if r['method']=='ensemble' and r['native_release'] or
                r['context_id'] in case_contexts and r['native_release']]
    groups = {}
    for r in selected: groups.setdefault(r['native_cache_file'], []).append(r)
    cache_hashes = {}; case_traces={}
    calibration=read(study/'calibration.json')['score_config']
    gravity=spec['native_reference']['public']['effective_native_options']['gravity']
    for name, group in groups.items():
        path = relative_file(study, 'native/'+name); actual_sha=sha(path)
        check(actual_sha==manifest['native/'+name], 'Dense cache is bound: '+name)
        cache_hashes[name]=actual_sha
        with np.load(path,allow_pickle=False) as z:
            arrays={k:z[k] for k in ('position_m','velocity_m_s','forces_N','time_s','dt_s','contact_distance_m')}
            ids=z['record_ids'].tolist()
            for row in group:
                col=row['native_cache_row'];ident=row['id'];item=items[ident];obs=item['observed']
                check(str(ids[col])==ident and actual_sha==row['native_cache_sha256'], ident+'/cache column')
                trace={k:v[:,col] if v.ndim>=2 and k!='time_s' else v for k,v in arrays.items()}
                metric=first_flight(spec['scene'],row['bucket_xy_m'],row['board_normal'],calibration,trace,valid=True)
                expected=targets[ident]['first_flight']
                check(metric['available']==expected['available'], ident+'/first flight availability')
                for key in ('start_index','end_index','cutoff_contact_channels'):
                    check(metric[key]==expected[key], ident+'/independent '+key)
                for key in ('closest_opening_distance_m','mouth_clearance_m'):
                    a,b=metric[key],expected[key]
                    check((a is None and b is None) or a is not None and b is not None and abs(a-b)<1e-9, ident+'/independent '+key)
                check(metric['available'],ident+'/measured rebound exists')
                start,end=metric['start_index'],metric['end_index']
                pos,vel,times=trace['position_m'],trace['velocity_m_s'],trace['time_s']
                direction=np.asarray(row['bucket_xy_m'])-row['values'][:2];direction/=np.linalg.norm(direction)
                lateral=np.array([-direction[1],direction[0]])
                world_velocity=np.asarray(vel[start],float)
                measured=[float(world_velocity[:2]@direction),float(world_velocity[:2]@lateral),float(world_velocity[2])]
                check(np.allclose(measured,targets[ident]['y'][6:9],rtol=0,atol=1e-10),ident+'/bound measured velocity')
                predicted=np.asarray(item['prediction']['rebound_velocity_m_s']);pred_world=np.r_[predicted[0]*direction+predicted[1]*lateral,predicted[2]]
                obs.update(rebound_velocity_m_s=measured,velocity_error_m_s=(predicted-measured).tolist(),
                    aperture_distance_m=metric['closest_opening_distance_m'],rim_crossing=metric['first_crossing'],
                    mouth_clearance_m=metric['mouth_clearance_m'],cutoff_contact_channels=metric['cutoff_contact_channels'],
                    rebound_time_s=metric['start_time_s'],cutoff_time_s=metric['end_time_s'],
                    cutoff_position_m=pos[end].astype(float).tolist(),first_flight_samples=metric['sample_count'])
                observed_center=np.asarray(row['board_center_m']); requested_center=np.asarray(row['values'][:3])
                center_error=float(np.linalg.norm(observed_center-requested_center)*1000)
                # Original FK target arithmetic is float32; allow0.1 micrometre
                # when comparing it with the requested float64 center norm.
                check(abs(center_error-row['pose_gate']['board_position_error_mm'])<1e-4, ident+'/held board center error')
                item['diagnosis']=classification(obs)
                if row['context_id'] in case_contexts:
                    indices=np.unique(np.linspace(start,end,min(241,end-start+1),dtype=int))
                    relative_times=times[indices]-times[start]
                    trace_out=dict(time_s=relative_times.tolist(),position_m=pos[indices].astype(float).tolist(),
                        record_id=ident,record_sha256=digest(row),native_cache_sha256=actual_sha,
                        scope='Downsampled recorded first rebound free flight; stops before the first subsequent contact interval response')
                    forecasts=[dict(label='Model velocity + gravity from observed rebound origin',
                        basis='observed_rebound_origin_model_velocity_gravity_no_contacts',time_s=relative_times.tolist(),
                        position_m=ballistic(pos[start],pred_world,relative_times,gravity)),
                        dict(label='Observed velocity + gravity from observed rebound origin',
                        basis='observed_rebound_origin_observed_velocity_gravity_no_contacts',time_s=relative_times.tolist(),
                        position_m=ballistic(pos[start],world_velocity,relative_times,gravity))]
                    case_traces[ident]=dict(trace=trace_out,forecasts=forecasts)
                    obs['velocity_head_endpoint_error_mm']=float(np.linalg.norm(np.array(forecasts[0]['position_m'][-1])-pos[end])*1000)
                    obs['observed_velocity_ballistic_error_mm']=float(np.linalg.norm(np.array(forecasts[1]['position_m'][-1])-pos[end])*1000)
    check(all(not r['observed']['whole_entry'] and r['observed']['retention_s']==0 for r in misses), 'All25 native misses failed entry, not retention')
    high=[r for r in ensemble if r['prediction']['capture_score']>=.9]
    high99=[r for r in ensemble if r['prediction']['capture_score']>=.99]
    check((len(high),sum(r['observed']['capture'] for r in high),len(high99),sum(r['observed']['capture'] for r in high99))
          == (36,18,13,4), 'Exact original high-score requested denominators and outcomes')
    cases=[]
    for ident in case_ids:
        case=copy.deepcopy(items[ident]);case.update(case_traces[ident])
        case['references']=[copy.deepcopy(items[r['id']]) for r in test if r['context_id']==case['context_id'] and r['method']!='ensemble']
        cases.append(case)
    paired36=next(c for c in cases if c['context_id'].endswith('-36'))
    bc36=next(r for r in paired36['references'] if r['method']=='bc')
    check(paired36['values'][:4]==bc36['values'][:4] and abs(paired36['values'][4]-bc36['values'][4]-.036)<1e-12
        and bc36['observed']['capture'] and not paired36['observed']['capture'], 'Case36 exact one-coordinate36mm gap contrast')
    for case in cases:
        case['bucket_rim_z_m']=float(calibration['compiled_rim_m'])
    result=dict(version=VERSION,status='completed',scope=SCOPE,
        source=dict(report_sha256=REPORT_SHA,audit_sha256=AUDIT_SHA,contract_fingerprint=SOURCE_FP,
            verified=True,original_native_audit_checks=proof['checks_count'],native_requests_added=0,
            records_sha256=manifest['records.jsonl'],targets_sha256=manifest['evaluation-targets.json']),
        summary=dict(requested=60,native_releases=49,captures=24,rejected=11,native_misses=25,
            high_score_threshold=.9,high_score_requests=len(high),high_score_captures=sum(r['observed']['capture'] for r in high),
            high_score_native_misses=sum(r['observed']['native_release'] and not r['observed']['capture'] for r in high),
            score99_requests=len(high99),score99_captures=sum(r['observed']['capture'] for r in high99),
            score99_mean=float(np.mean([r['prediction']['capture_score'] for r in high99])),
            all_native_misses_no_entry=True,cutoff_contact_counts=dict(Counter('|'.join(r['observed']['cutoff_contact_channels']) for r in misses))),
        calibration_bins=bins(ensemble),cases=cases,misses=copy.deepcopy(misses),
        findings=[dict(status='observed',statement='All25 ensemble native misses had zero whole-ball entry and zero retention.',evidence='Original independently audited labels; not a new scoring definition'),
            dict(status='observed',statement=f"Only {sum(r['observed']['capture'] for r in high99)} of {len(high99)} ensemble requests scored at least99% actually captured.",evidence='All60 requested ensemble outcomes; rejected setups remain in denominator'),
            dict(status='observed',statement='The first post-board free flight ended at bucket contact in16 misses, robot contact in3, and table contact in6.',evidence='Independent dense force-channel scan; contact cutoff is not by itself proof of rim contact or final capture status'),
            dict(status='paired_observation',statement='Case36 kept board XY/height/tilt identical to successful BC, raised drop gap36mm, and missed.',evidence='One original nominal shot per action; does not prove a repeatable parameter window'),
            dict(status='implementation',statement='Capture, opening-distance and rebound-velocity heads are independently supervised; ranking does not enforce ballistic consistency.',evidence='Frozen model loss and ranking source; not proof this alone caused a particular miss'),
            dict(status='limitation',statement='Small member disagreement is not calibrated confidence; shared training support can create correlated errors.',evidence='Five shared-architecture bootstrap members; nearestTRAIN proximity is descriptive, not proof of cause')],
        next_steps=['Inspect rebound velocity and aperture ranking on these development cases before another fit.',
            'Compare a physically constrained flight score with the independent capture head; keep any resulting tuning separate from this frozen evaluation.',
            'Any changed selector needs new frozen held-out bucket positions. Noise and128-substep testing remain outside this task.'],
        geometry=dict(ball_radius_m=spec['scene']['ball']['radius_m'],inner_top_diameter_m=spec['scene']['bucket']['inner_top_diameter_m'],
            rim_z_m=calibration['compiled_rim_m'],board_size_m=spec['scene']['board']['size_m'],
            bucket_height_status=spec['scene']['bucket']['height_interpretation']),
        processing=dict(cuda_used=False,training=False,new_native_requests=0,source_sha256=sha(__file__),consumed_cache_sha256=cache_hashes))
    output.mkdir(parents=True)
    proof_out=dict(version=AUDIT_VERSION,status='passed',checks_count=len(checks),checks=checks,
        source_report_sha256=REPORT_SHA,source_audit_sha256=AUDIT_SHA,diagnosis_source_sha256=sha(__file__))
    (output/'checks.json').write_text(json.dumps(proof_out,indent=2,allow_nan=False)+'\n')
    result['audit']=dict(version=AUDIT_VERSION,status='passed',full_checks_verified=True,checks_count=len(checks),source_sha256=sha(output/'checks.json'),
        scope='Derived diagnostic reconstruction checks; original native labels retain their separate897304-check full audit')
    (output/'diagnosis.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    payload=copy.deepcopy(result)
    for c in payload['cases']:
        c.pop('trace');c.pop('forecasts')
    (output/'summary.json').write_text(json.dumps(payload,indent=2,allow_nan=False)+'\n')
    (output/'checksums.json').write_text(json.dumps({name:sha(output/name) for name in ('diagnosis.json','summary.json','checks.json')},indent=2)+'\n')
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('study','audit','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();result=build(a.study,a.audit,a.output)
    print(json.dumps(dict(status=result['status'],summary=result['summary'],checks=result['audit']['checks_count'])))


if __name__=='__main__':main()
