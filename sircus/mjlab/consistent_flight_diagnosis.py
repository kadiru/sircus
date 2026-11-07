"""CPU diagnosis of the completed frozen-selector comparison; no new shots.

Original audit labels are authoritative. Consumed artifacts are byte-bound before
measured contact-limited flights are reconstructed. The former TEST positions
become development examples; conditional ballistic curves are not new rollouts.
"""
from __future__ import annotations
from collections import Counter
import argparse
import copy
import json
from pathlib import Path

import numpy as np

from .curate_physics_ensemble import statistics, captured, control_gate
from .curate_xy650 import read, sha, digest, relative_file, hash_map, require
from .consistent_flight_selector import geometry, nominal_origin, predict_candidate
from .ensemble_flight_diagnosis import ballistic, classification as legacy_classification, prediction
from .xy650_verify import first_flight

VERSION = 'sircus-consistent-flight-diagnosis-v1'
AUDIT_VERSION = 'sircus-consistent-flight-diagnosis-checks-v1'
KEY = 'consistent_flight_diagnosis'
METHODS = ('bc', 'ppo', 'ensemble', 'consistent')
SOURCE_HASHES = dict(report='62e0f18d2fcaaa4ffb69aae4c583651bce4b54bd984e628c943ccd8dadb39b38',
    manifest='3ad88c8ccaf304e04fdc6c068d761f6bbd4d9702a47ee76e541077e8bec1a1f3',
    audit='43a5c6e2dc0e204e3b646b9da7a74ca09e53a420e3869834d12da255e2029952')
SCOPE = ('CPU diagnosis of original fixed64 requests, not a new native test or fit. '
    'These former TEST positions are now development data. Original audited '
    'capture labels remain unchanged. Predictions and nominal origins are separate '
    'from observed rebounds; conditional ballistic curves stop at first contact. '
    'No 128-substep, noise, perception, driven-transit or physical validation.')


class Checks:
    def __init__(self): self.entries = []
    def __call__(self, passed, label):
        self.entries.append(dict(label=label, passed=bool(passed)))
        require(bool(passed), label)


def verify_inputs(study, audit_path, checks=None):
    """Bind an already completed full audit; do not relabel this as a rescore."""
    check = checks or Checks()
    study, audit_path = Path(study).resolve(), Path(audit_path).resolve()
    report, spec, proof = (read(p) for p in (study/'report.json', study/'contract.json', audit_path))
    for key,path in (('report',study/'report.json'),('manifest',study/'evidence-manifest.json'),('audit',audit_path)):
        check(sha(path)==SOURCE_HASHES[key],'Exact completed comparison bytes/'+key)
    entries = proof.get('checks')
    check(proof.get('version') == 'sircus-consistent-flight-audit-v1' and proof.get('status') == 'passed'
          and proof.get('failures') == [] and isinstance(entries, list) and bool(entries)
          and type(proof.get('checks_count')) is int and proof['checks_count'] == len(entries)
          and all(isinstance(c, dict) and c.get('passed') is True and isinstance(c.get('label'), str) for c in entries),
          'Complete strictly passed original native audit')
    check(proof.get('report_sha256') == sha(study/'report.json')
          and proof.get('evidence_manifest_sha256') == sha(study/'evidence-manifest.json'), 'Audit binds exact report and manifest')
    check(report.get('version') == spec.get('version') == 'sircus-consistent-flight-v1'
          and report.get('status') == report.get('phase') == 'completed', 'Completed fixed-selector source required')
    unsigned = dict(spec); fp = unsigned.pop('fingerprint', None)
    check(fp == digest(unsigned) == report.get('contract_fingerprint') == proof.get('contract_fingerprint'), 'Contract binding')
    expected = dict(substeps=64, methods=list(METHODS), test_layouts=30, maximum_fresh_requests=132,
                    control_requests=12, new_train_requests=0, noise=False, no_test_flight_search=True,
                    model_weights_unchanged=True, candidates=16, maximum_geometry_checks=4)
    check(report.get('protocol') == spec.get('protocol') and all(spec['protocol'].get(k) == v for k, v in expected.items()), 'Exact frozen comparison protocol')
    check(proof.get('requested_trials') == 132 and proof.get('training_requests') == 0, 'Original 132 requests and zero new fitting')
    check(proof.get('producer_source_sha256') == report.get('source_sha256') == spec.get('source_sha256')
          and proof.get('native_source_sha256') == spec.get('native_source_sha256'), 'Audited producer and native source identities')
    manifest = hash_map(read(study/'evidence-manifest.json'))
    check(report.get('evidence_sha256') == manifest and manifest.get('contract.json') == sha(study/'contract.json'), 'Report and manifest agree')
    actual = {str(p.relative_to(study)) for p in study.rglob('*') if p.is_file()
              and str(p.relative_to(study)) not in ('report.json', 'evidence-manifest.json')}
    check(set(manifest) == actual, 'Complete frozen artifact inventory')
    for name, expected_sha in manifest.items():
        check(sha(relative_file(study, name)) == expected_sha, 'Bound artifact/'+name)
    rows = [json.loads(line) for line in (study/'records.jsonl').read_text().splitlines() if line]
    packet = read(study/'prediction-plan.json')
    check(len(rows) == len({r['id'] for r in rows}) == 132 and [r['id'] for r in rows] == [r['id'] for r in packet['requests']], 'Every ordered requested slot retained')
    check(packet.get('frozen_before_any_native_outcome') is True and type(packet.get('native_planning_queries')) is int
          and packet['native_planning_queries'] == 0 and packet['frozen_unix_s'] <= report['acquired_unix_s']
          and sha(study/'prediction-plan.json') == report['prediction_plan_sha256'], 'Original plan frozen before native outcomes')
    for row, req in zip(rows, packet['requests']):
        check(all(row.get(k) == v for k, v in req.items()) and row.get('action_substituted') is False, row['id']+'/original requested action unchanged')
    check(Counter(r['role'] for r in rows) == Counter(test=120, positive_control=6, negative_control=6), 'Requested roles')
    test = [r for r in rows if r['role'] == 'test']
    check(Counter(r['method'] for r in test) == Counter({m:30 for m in METHODS}), 'Thirty per method')
    layouts = {r['context_id']:r['bucket_xy_m'] for r in test if r['method'] == 'consistent'}
    for method in METHODS:
        check({r['context_id']:r['bucket_xy_m'] for r in test if r['method'] == method} == layouts, method+'/paired layouts')
    comparisons = [dict(method=m, counts=statistics([r for r in test if r['method'] == m])) for m in METHODS]
    check(comparisons == report['comparisons'] == proof['comparisons'], 'Original comparison accounting')
    check(statistics(rows) == report['counts'] and proof['native_releases'] == report['counts']['native_releases'], 'Original all-phase accounting')
    for phase, key in (('controls_pre', 'pre'), ('controls_post', 'post')):
        gate = control_gate([r for r in rows if r['phase'] == phase])
        check(gate['status'] == 'passed' and report['control_gates'][key] == gate, key+'/original positive and negative controls')
    return study, report, spec, proof, manifest, packet, rows


verified_inputs = verify_inputs


def classification(observed):
    if observed.get('native_release') and not observed.get('capture') and observed.get('bucket_capture'):
        return ('Whole-ball entry and one-second bucket retention observed, but complete board-route scoring failed; '
                'external table/floor/robot/unattributed contacts are disallowed by the unchanged route contract')
    return legacy_classification(observed)


def make_item(row, log, scene, calibration, gravity):
    observed = dict(native_release=row['native_release'], capture=captured(row), bucket_capture=bool(row['success']),
                    external_contact_steps=row['external_contact_steps'], whole_entry=bool(row['whole_ball_entry']),
                    retention_s=float(row['max_retention_s']), board_bounce=bool(row['board_bounce']),
                    invalid_physics=row['invalid_physics'], pose_gate=copy.deepcopy(row.get('pose_gate')),
                    max_board_drift_mm=row.get('max_board_drift_mm'), bucket_collider_error_m=row.get('bucket_collider_error_m'),
                    first_flight_available=False,
                    missing_reason='not_reconstructed' if row['native_release'] else 'setup_rejected_no_native_flight')
    pred = prediction(row['model_prediction'])
    proxy = predict_candidate(scene, calibration, row['bucket_xy_m'], row['values'], row['model_prediction'], gravity=gravity)
    pred['proxy_aperture_margin_m'] = proxy['worst_clearance_m']
    item = dict(id=row['id'], record_id=row['id'], record_sha256=digest(row), context_id=row['context_id'], method=row['method'],
                values=copy.deepcopy(row['values']), bucket_xy_m=copy.deepcopy(row['bucket_xy_m']),
                board_bucket_distance_m=float(np.linalg.norm(np.array(row['values'][:2])-row['bucket_xy_m'])),
                prediction=pred, observed=observed, nominal=proxy['launch'], diagnosis=classification(observed))
    diagnostics = log['new_ranking']['diagnostics']
    margins = [r['worst_clearance_m'] for r in diagnostics if r['worst_clearance_m'] is not None]
    anchor = log['anchor']; anchor_values = anchor.get('values') if isinstance(anchor, dict) else anchor
    gap_values = sorted(set(c['values'][4] for c in log['candidates']))
    item['candidate_coverage'] = dict(candidates=len(log['candidates']),
        available_aperture_margins=len(margins), best_proxy_margin_m=max(margins) if margins else None,
        all_candidate_margins_available=len(margins)==len(log['candidates']),
        nonnegative_proxy_candidates=sum(v >= 0 for v in margins),
        all_available_margins_negative=bool(margins) and all(v < 0 for v in margins),
        gap_min_m=min(gap_values), gap_max_m=max(gap_values), gap_values_m=gap_values,
        anchor_gap_m=anchor_values[4] if anchor_values is not None else None,
        scope='Frozen model-only bank coverage, not native feasibility or counterfactual success')
    selected = log.get('selection', {}).get(row['method'])
    if selected:
        ranking = log['new_ranking' if row['method'] == 'consistent' else 'old_ranking']
        chosen = selected['selected_index']
        item['selection'] = dict(selected_candidate_id=log['candidates'][chosen]['id'], selected_candidate_index=chosen,
            rank=ranking['order'].index(chosen)+1, geometry_checks=len(selected['checks']),
            all_geometry_checks_rejected=selected['all_geometry_checks_rejected'],
            geometry_reason=row.get('geometry_reason'), context_reason=None,
            checks=[{k:e.get(k) for k in ('candidate_index','candidate_id','passed','reason')} for e in selected['checks']],
            native_inference_queries=0)
    else:
        item['selection'] = dict(selected_candidate_id=None, rank=None, geometry_checks=None,
                                 geometry_reason=row.get('geometry_reason'), context_reason=None, checks=[])
    return item


def analyze_trace(row, scene, calibration, arrays, gravity):
    """Return observed metric plus explicitly conditioned curves, never labels."""
    metric = first_flight(scene, row['bucket_xy_m'], row['board_normal'], calibration, arrays,
                          valid=bool(row['native_release'] and not row['invalid_physics'] and not row.get('invalid')))
    if not metric['available']:
        return dict(metric=metric, observed=dict(first_flight_available=False, missing_reason=metric['reason']), trace=None, forecasts=[])
    start, end = metric['start_index'], metric['end_index']
    pos, vel, times = (np.asarray(arrays[k], float) for k in ('position_m','velocity_m_s','time_s'))
    direction = np.asarray(row['bucket_xy_m'])-row['values'][:2]; direction /= np.linalg.norm(direction)
    lateral = np.array([-direction[1], direction[0]])
    measured = [float(vel[start,:2]@direction), float(vel[start,:2]@lateral), float(vel[start,2])]
    predicted = np.asarray(row['model_prediction']['rebound_velocity_m_s'], float)
    pred_world = np.r_[predicted[0]*direction+predicted[1]*lateral,predicted[2]]
    launch = nominal_origin(row['values'], row['bucket_xy_m'], geometry(scene,calibration,gravity=gravity))
    observed = dict(first_flight_available=True, rebound_velocity_m_s=measured,
        missing_reason=None,
        predicted_velocity_world_m_s=pred_world.tolist(), observed_velocity_world_m_s=vel[start].tolist(),
        observed_rebound_position_m=pos[start].tolist(), nominal_origin_m=launch['origin_m'],
        nominal_origin_minus_observed_rebound_mm=((np.asarray(launch['origin_m'])-pos[start])*1000).tolist(),
        velocity_error_m_s=(predicted-measured).tolist(), aperture_distance_m=metric['closest_opening_distance_m'],
        rim_crossing=metric['first_crossing'], mouth_clearance_m=metric['mouth_clearance_m'],
        cutoff_contact_channels=metric['cutoff_contact_channels'], cutoff_reason=metric['cutoff_reason'],
        rebound_time_s=metric['start_time_s'], cutoff_time_s=metric['end_time_s'],
        cutoff_position_m=pos[end].tolist(), first_flight_samples=metric['sample_count'])
    indices = np.unique(np.linspace(start,end,min(181,end-start+1),dtype=int))
    elapsed = times[indices]-times[start]
    trace = dict(time_s=elapsed.tolist(),position_m=pos[indices].tolist(),record_id=row['id'],record_sha256=digest(row),
        native_cache_sha256=row['native_cache_sha256'],scope='Recorded first rebound free flight; ends before first subsequent contact response')
    if len(indices) < 2:
        observed.update(velocity_head_endpoint_error_mm=None, observed_velocity_ballistic_error_mm=None, nominal_forecast_endpoint_error_mm=None)
        return dict(metric=metric, observed=observed, trace=trace, forecasts=[])
    models = [('Model velocity from observed rebound origin','observed_origin_model_velocity',pos[start],pred_world),
              ('Observed velocity from observed rebound origin','observed_origin_observed_velocity',pos[start],vel[start]),
              ('Model velocity from nominal board tangent','nominal_origin_model_velocity',launch['origin_m'],pred_world)]
    forecasts = [dict(label=label,basis=basis+'_gravity_no_contacts',time_s=elapsed.tolist(),position_m=ballistic(origin,velocity,elapsed,gravity))
                 for label,basis,origin,velocity in models]
    for key, forecast in zip(('velocity_head_endpoint_error_mm','observed_velocity_ballistic_error_mm','nominal_forecast_endpoint_error_mm'),forecasts):
        observed[key]=float(np.linalg.norm(np.array(forecast['position_m'][-1])-pos[end])*1000)
    return dict(metric=metric, observed=observed, trace=trace, forecasts=forecasts)


def paired_results(items):
    own = {r['context_id']:r for r in items if r['method']=='consistent'}
    result={}
    for method in ('bc','ppo','ensemble'):
        refs={r['context_id']:r for r in items if r['method']==method}
        require(set(refs)==set(own),'Paired request set differs')
        wins=[k for k in own if own[k]['observed']['capture'] and not refs[k]['observed']['capture']]
        losses=[k for k in own if not own[k]['observed']['capture'] and refs[k]['observed']['capture']]
        result[method]=dict(wins=wins,losses=losses,both_capture=[k for k in own if own[k]['observed']['capture'] and refs[k]['observed']['capture']],
                            both_fail=[k for k in own if not own[k]['observed']['capture'] and not refs[k]['observed']['capture']])
    return result


def choose_cases(own, paired):
    """Deterministic diagnostic selection, not test/model selection."""
    misses=sorted((r for r in own if r['observed']['native_release'] and not r['observed']['capture']),
                  key=lambda r:(-r['prediction']['capture_score'],r['id']))
    ids=[r['id'] for r in misses[:3]]
    # Preserve unusual route failures even if their old capture-head confidence
    # is lower. Entry/retention and the complete board-only route are distinct.
    retained=next((r for r in misses if r['observed'].get('bucket_capture')),None)
    if retained and retained['id'] not in ids:ids.append(retained['id'])
    for context in sorted(set(k for p in paired.values() for k in p['losses'])):
        r=next(r for r in own if r['context_id']==context)
        if r['id'] not in ids:ids.append(r['id'])
        if len(ids)>=5:break
    reject=next((r for r in own if not r['observed']['native_release']),None)
    if reject and reject['id'] not in ids:ids.append(reject['id'])
    return ids[:6]


def build(study,audit,output):
    study,output=Path(study).resolve(),Path(output).resolve()
    require(not output.exists() and not output.is_relative_to(study),'New derived directory outside immutable source required')
    checks=Checks();study,report,spec,proof,manifest,packet,rows=verify_inputs(study,audit,checks)
    calibration=read(study/'calibration.json')['score_config']
    gravity=spec['native_reference']['allocated_options']['native']['gravity']
    logs={r['context_id']:r for r in read(study/'rankings.json')}
    tests=[r for r in rows if r['role']=='test']
    items={r['id']:make_item(r,logs[r['context_id']],spec['scene'],calibration,gravity) for r in tests}
    consumed={};groups={}
    for row in tests:
        item=items[row['id']]
        for field,hashfield in (('context_resolution_file','context_resolution_sha256'),('connection_file','connection_sha256')):
            if row.get(field):
                path=relative_file(study,'native/'+row[field]);actual=sha(path)
                checks(actual==row[hashfield]==manifest['native/'+row[field]],row['id']+'/bound '+field)
                obj=read(path)
                if field=='context_resolution_file':item['selection']['context_reason']=obj.get('reason')
                else:item['selection']['connection_reason']=obj.get('reason')
        if row['native_release']:groups.setdefault(row['native_cache_file'],[]).append(row)
    for name,group in groups.items():
        path=relative_file(study,'native/'+name);actual=sha(path);consumed[name]=actual
        checks(actual==manifest['native/'+name],'Dense cache bound/'+name)
        with np.load(path,allow_pickle=False) as z:
            arrays={k:z[k] for k in ('position_m','velocity_m_s','forces_N','time_s','dt_s','contact_distance_m')}
            ids=z['record_ids'].tolist()
            for row in group:
                col=row['native_cache_row'];ident=row['id']
                checks(str(ids[col])==ident and actual==row['native_cache_sha256'],ident+'/exact cache column')
                trace={k:v[:,col] if v.ndim>=2 and k!='time_s' else v for k,v in arrays.items()}
                derived=analyze_trace(row,spec['scene'],calibration,trace,gravity)
                items[ident]['observed'].update(derived['observed'])
                items[ident]['trace']=derived['trace'];items[ident]['forecasts']=derived['forecasts']
                items[ident]['diagnosis']=classification(items[ident]['observed'])
                checks(type(derived['metric']['available']) is bool and
                       (derived['metric']['available'] or isinstance(derived['metric'].get('reason'),str)),
                       ident+'/explicit flight availability or unknown reason')
    all_items=list(items.values());own=[r for r in all_items if r['method']=='consistent'];paired=paired_results(all_items)
    misses=[r for r in own if r['observed']['native_release'] and not r['observed']['capture']]
    rejects=[r for r in own if not r['observed']['native_release']]
    cases=[]
    for ident in choose_cases(own,paired):
        c=copy.deepcopy(items[ident]);c['references']=[copy.deepcopy(r) for r in all_items if r['context_id']==c['context_id'] and r['method']!='consistent']
        c['bucket_rim_z_m']=calibration.get('compiled_rim_m',spec['scene']['bucket']['assumed_bottom_thickness_m']+spec['scene']['bucket']['assumed_interior_depth_m'])
        for ref in c['references']:ref.pop('trace',None);ref.pop('forecasts',None)
        cases.append(c)
    def scalar(item):
        r=copy.deepcopy(item);r.pop('trace',None);r.pop('forecasts',None);return r
    counts=statistics([r for r in tests if r['method']=='consistent'])
    summary=dict(**counts,all_native_misses_no_entry=all(not r['observed']['whole_entry'] for r in misses),
        native_first_flight_unknown=sum(r['observed']['native_release'] and not r['observed']['first_flight_available'] for r in own),
        no_entry_failures=sum(not r['observed']['whole_entry'] for r in misses),
        bucket_retention_but_route_failures=sum(r['observed']['bucket_capture'] for r in misses),
        all_bank_margins_negative_in_misses=sum(r['candidate_coverage']['all_candidate_margins_available'] and r['candidate_coverage']['all_available_margins_negative'] for r in misses),
        cutoff_contact_counts=dict(Counter('|'.join(r['observed'].get('cutoff_contact_channels',[])) or 'unknown' for r in misses)),
        rejection_reason_counts=dict(Counter(str(r['selection'].get('geometry_reason') or r['selection'].get('context_reason') or 'unknown') for r in rejects)),
        native_miss_contexts=[r['context_id'] for r in misses],rejected_contexts=[r['context_id'] for r in rejects])
    findings=[dict(status='observed',statement=f"New selector captured {counts['captures']}/{counts['requested']}; {counts['native_misses']} native misses and {counts['rejected']} setup rejects remain.",evidence='Original frozen comparison and native audit; no changed success definition'),
        dict(status='observed',statement='Contact-limited first-flight cutoff channels: '+json.dumps(summary['cutoff_contact_counts'],sort_keys=True),evidence='Byte-bound dense force samples; a bucket contact is not automatically a rim hit'),
        dict(status='limitation',statement='Nominal board tangent and measured post-contact rebound origin differ. Their diagnostic difference is not itself proof of cause.',evidence='Requested geometry and independently sampled dense state remain separate'),
        dict(status='limitation',statement='One nominal request per method and layout; paired differences do not establish repeatability or a reliable method winner.',evidence='Original30-position protocol; no added repetitions')]
    result=dict(version=VERSION,status='completed',scope=SCOPE,
        source=dict(verified=True,report_sha256=sha(study/'report.json'),audit_sha256=sha(audit),
            manifest_sha256=sha(study/'evidence-manifest.json'),contract_fingerprint=spec['fingerprint'],
            original_total_requests=report['counts']['requested'],original_native_releases=report['counts']['native_releases'],
            original_native_audit_checks=proof['checks_count'],native_requests_added=0,records_sha256=manifest['records.jsonl'],rankings_sha256=manifest['rankings.json']),
        summary=summary,comparisons=copy.deepcopy(report['comparisons']),paired=paired,requests=[scalar(r) for r in own],
        misses=[scalar(r) for r in misses],rejects=[scalar(r) for r in rejects],cases=cases,findings=findings,
        next_steps=['Use these former TEST examples only for diagnosis and targeted development.',
                    'Freeze one correction and its bounded matched diagnostic tests before new native outcomes.',
                    'Evaluate a changed selector on new held-out bucket positions before claiming generalization.'],
        geometry=dict(ball_radius_m=spec['scene']['ball']['radius_m'],rim_z_m=cases[0]['bucket_rim_z_m'] if cases else calibration.get('compiled_rim_m'),
                      board_size_m=spec['scene']['board']['size_m'],bucket_height_status=spec['scene']['bucket']['height_interpretation']),
        processing=dict(cuda_used=False,training=False,new_native_requests=0,source_sha256=sha(__file__),consumed_cache_sha256=consumed))
    output.mkdir(parents=True)
    derived_proof=dict(version=AUDIT_VERSION,status='passed',checks_count=len(checks.entries),checks=checks.entries,
        source_report_sha256=result['source']['report_sha256'],source_audit_sha256=result['source']['audit_sha256'],diagnosis_source_sha256=sha(__file__))
    (output/'checks.json').write_text(json.dumps(derived_proof,indent=2,allow_nan=False)+'\n')
    result['audit']=dict(version=AUDIT_VERSION,status='passed',checks_count=len(checks.entries),full_checks_verified=True,
        source_sha256=sha(output/'checks.json'),source_report_sha256=result['source']['report_sha256'],
        source_audit_sha256=result['source']['audit_sha256'],scope='Derived evidence-binding/reconstruction checks; original full native audit remains separate')
    (output/'diagnosis.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    reduced=copy.deepcopy(result)
    for c in reduced['cases']:c.pop('trace',None);c.pop('forecasts',None)
    (output/'summary.json').write_text(json.dumps(reduced,indent=2,allow_nan=False)+'\n')
    (output/'checksums.json').write_text(json.dumps({n:sha(output/n) for n in ('diagnosis.json','summary.json','checks.json')},indent=2)+'\n')
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('study','audit','output'):p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();r=build(a.study,a.audit,a.output)
    print(json.dumps(dict(status=r['status'],summary=r['summary'],checks=r['audit']['checks_count'])))

if __name__=='__main__':main()
