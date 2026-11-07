"""Byte-bound CPU diagnosis of the original 50-position gap comparison.

No fit, native rollout, relabeling or source-directory writes. Recorded first
flights end at the first subsequent contact; unknown crossings stay unknown.
"""
from __future__ import annotations

import argparse
from collections import Counter
import copy
import json
from pathlib import Path

import numpy as np

from . import consistent_flight_diagnosis as prior
from .curate_physics_ensemble import statistics, control_gate
from .curate_xy650 import read, sha, digest, relative_file, hash_map, require

VERSION = 'sircus-gap-flight-diagnosis-v1'
AUDIT_VERSION = 'sircus-gap-flight-diagnosis-checks-v1'
KEY = 'gap_flight_diagnosis'
METHODS = ('consistent', 'gap_coverage')
SOURCE_HASHES = dict(
    report='110b8b7f48f9f0cc153e0dc0c9341ed78bb04d861090abef592bfc98f2f5bf1b',
    manifest='7af5b7087d3ae24b47a2949e5a5cf2cd6c9a783411387186b0926fd2ae9e091e',
    audit='5c6bb8965ef264f9744c0f72c6da5c5bb4904d724938519eb21f45952d1c925a')
SCOPE = ('CPU diagnosis of the original fixed64 50-position paired comparison; '
    'no new native tests or training. Original complete-shot labels remain authoritative. '
    'Former TEST layouts are now development examples. Conditional ballistic curves '
    'stop at the first subsequent contact. No noise, 128-substep, driven-transit, perception '
    'or physical validation.')
EARLY_ROBOT_CONTACT_MS = 50.0  # Descriptive display threshold, never a scoring rule.
Checks = prior.Checks


def verify_inputs(study, audit_path, checks=None):
    check = checks or Checks()
    study, audit_path = Path(study).resolve(), Path(audit_path).resolve()
    report, spec, proof = (read(p) for p in (study/'report.json', study/'contract.json', audit_path))
    for key, path in (('report', study/'report.json'), ('manifest', study/'evidence-manifest.json'), ('audit', audit_path)):
        check(sha(path) == SOURCE_HASHES[key], 'Exact completed comparison bytes/'+key)
    entries = proof.get('checks')
    check(proof.get('version') == 'sircus-gap-coverage-audit-v2' and proof.get('status') == 'passed'
          and proof.get('failures') == [] and isinstance(entries, list) and bool(entries)
          and type(proof.get('checks_count')) is int and proof['checks_count'] == len(entries)
          and all(isinstance(c, dict) and c.get('passed') is True and isinstance(c.get('label'), str) for c in entries),
          'Complete strictly passed original audit v2')
    check(proof.get('report_sha256') == sha(study/'report.json')
          and proof.get('evidence_manifest_sha256') == sha(study/'evidence-manifest.json'), 'Audit binds exact report and manifest')
    check(report.get('version') == spec.get('version') == 'sircus-gap-coverage-study-v1'
          and report.get('status') == report.get('phase') == 'completed', 'Completed gap comparison required')
    unsigned = dict(spec); fp = unsigned.pop('fingerprint', None)
    check(fp == digest(unsigned) == report.get('contract_fingerprint') == proof.get('contract_fingerprint'), 'Contract binding')
    expected = dict(substeps=64, methods=list(METHODS), test_layouts=50, maximum_fresh_requests=112,
        control_requests=12, new_train_requests=0, noise=False, no_test_flight_search=True,
        model_weights_unchanged=True, candidates=16, maximum_geometry_checks=4,
        changed_candidate_indices=[8, 9], gap_residual_endpoints=[-1, 1], unchanged_candidate_count=14,
        no_hard_aperture_veto=True, post_bounce_robot_clearance_added=False)
    check(report.get('protocol') == spec.get('protocol') and all(spec['protocol'].get(k) == v for k, v in expected.items()), 'Exact frozen comparison protocol')
    check(proof.get('requested_trials') == 112 and proof.get('training_requests') == 0, 'Original 112 requests and zero new fitting')
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
    check(len(rows) == len({r['id'] for r in rows}) == 112 and [r['id'] for r in rows] == [r['id'] for r in packet['requests']], 'Every ordered requested slot retained')
    check(packet.get('frozen_before_any_native_outcome') is True and type(packet.get('native_planning_queries')) is int
          and packet['native_planning_queries'] == 0 and packet['frozen_unix_s'] <= report['acquired_unix_s']
          and sha(study/'prediction-plan.json') == report['prediction_plan_sha256'], 'Original plan frozen before outcomes')
    for row, req in zip(rows, packet['requests']):
        check(all(row.get(k) == v for k, v in req.items()) and row.get('action_substituted') is False, row['id']+'/requested action unchanged')
    check(Counter(r['role'] for r in rows) == Counter(test=100, positive_control=6, negative_control=6), 'Requested roles')
    tests = [r for r in rows if r['role'] == 'test']
    check(Counter(r['method'] for r in tests) == Counter({m: 50 for m in METHODS}), 'Fifty per method')
    layouts = {r['context_id']: r['bucket_xy_m'] for r in tests if r['method'] == 'consistent'}
    check(len(layouts) == 50, 'Fifty distinct context identifiers')
    for method in METHODS:
        check({r['context_id']: r['bucket_xy_m'] for r in tests if r['method'] == method} == layouts, method+'/paired layouts')
    comparisons = [dict(method=m, counts=statistics([r for r in tests if r['method'] == m])) for m in METHODS]
    check(comparisons == report['comparisons'] == proof['comparisons'], 'Original comparison accounting')
    check(statistics(rows) == report['counts'] and proof['native_releases'] == report['counts']['native_releases'], 'Original all-phase accounting')
    for phase, key in (('controls_pre', 'pre'), ('controls_post', 'post')):
        gate = control_gate([r for r in rows if r['phase'] == phase])
        check(gate['status'] == 'passed' and report['control_gates'][key] == gate, key+'/original controls')
    return study, report, spec, proof, manifest, packet, rows


def make_item(row, log, scene, calibration, gravity):
    """Adapt the two immutable candidate banks to the existing CPU presenter."""
    bank = log['banks'][row['method']]
    adapted = dict(anchor=log['anchor'], candidates=bank['candidates'], selection=log['selection'],
                   new_ranking=bank['ranking'], old_ranking=bank['ranking'])
    item = prior.make_item(row, adapted, scene, calibration, gravity)
    item['candidate_coverage']['bank'] = row['method']
    item['observed'].update(first_flight_contact=None, rebound_to_contact_ms=None,
        early_robot_contact=None, rim_forward_error_mm=None, rim_lateral_error_mm=None,
        robot_contact_link=None, robot_contact_link_status='not_recorded_by_dense_logger')
    return item


def setup_proof(row, resolution, connection):
    """Summarize actual saved attempts. Rejection is not proof of unreachability."""
    attempts = connection.get('endpoint_attempts') or {}
    pairs = attempts.get('pair_attempts', [])
    details = []
    for attempt in pairs:
        checked = attempt.get('checked') or {}; strict = checked.get('strict_geometry') or {}
        details.append(dict(branch10=attempt.get('branch10'), branch14=attempt.get('branch14'),
            board_position_error_m=checked.get('board_position_error_m'),
            dropper_position_error_m=checked.get('dropper_position_error_m'),
            board_normal_error_deg=checked.get('board_normal_error_deg'),
            minimum_clearance_m=strict.get('minimum_clearance_m'), closest_pair=strict.get('closest_pair'),
            independent_reject_arms=checked.get('independent_reject_arms'),
            margin_contacts=copy.deepcopy(strict.get('margin_contacts', []))))
    home = resolution.get('home_geometry') or {}
    return dict(reason=connection.get('reason'), connection_file=row.get('connection_file'),
        connection_sha256=row.get('connection_sha256'), resolution_file=row.get('context_resolution_file'),
        resolution_sha256=row.get('context_resolution_sha256'),
        valid_ik_branches=attempts.get('valid_ik_branches'),
        endpoint_pair_attempts=len(pairs), endpoint_examples=details,
        home_minimum_clearance_m=home.get('minimum_clearance_m'), home_closest_pair=home.get('closest_pair'),
        fixed_base_blockers=copy.deepcopy(home.get('fixed_base_blockers', [])),
        analytic_hint=copy.deepcopy(attempts.get('analytic_hint')),
        scope='Recorded finite setup search only; unavailable branch, clearance or pose values remain null. No native flight for rejected setups.')


def enrich_flight(item):
    """Separate measurements from optional contact labels and absent crossings."""
    o = item['observed']
    if not o.get('first_flight_available'):
        return
    channels = o.get('cutoff_contact_channels', [])
    contact = channels if channels else None
    o['first_flight_contact'] = contact
    o['rebound_to_contact_ms'] = ((o['cutoff_time_s']-o['rebound_time_s'])*1000 if contact else None)
    o['early_robot_contact'] = ('robot' in channels and o['rebound_to_contact_ms'] <= EARLY_ROBOT_CONTACT_MS) if contact else None
    crossing = o.get('rim_crossing')
    if crossing is not None:
        direction = np.asarray(item['bucket_xy_m'])-np.asarray(item['values'][:2]); direction /= np.linalg.norm(direction)
        lateral = np.array([-direction[1], direction[0]])
        error = np.asarray(crossing['xy_m'])-np.asarray(item['bucket_xy_m'])
        o['rim_forward_error_mm'] = float(error@direction*1000)
        o['rim_lateral_error_mm'] = float(error@lateral*1000)


def scalar(item):
    value = copy.deepcopy(item); value.pop('trace', None); value.pop('forecasts', None)
    return value


def paired_results(items):
    own = {r['context_id']: r for r in items if r['method'] == 'consistent'}
    refs = {r['context_id']: r for r in items if r['method'] == 'gap_coverage'}
    require(len(own) == len(refs) == 50 and set(own) == set(refs), 'Every paired context required')
    categories = dict(wins=[], losses=[], both_capture=[], both_fail=[])
    pairs = []
    for context, original in own.items():
        wider = refs[context]; a, b = original['observed']['capture'], wider['observed']['capture']
        # Wins/losses retain original-selector perspective used by the older UI.
        category = 'both_capture' if a and b else 'wins' if a else 'losses' if b else 'both_fail'
        categories[category].append(context)
        pairs.append(dict(context_id=context, bucket_xy_m=copy.deepcopy(original['bucket_xy_m']),
                          original=scalar(original), wider=scalar(wider)))
    return dict(gap_coverage=categories), pairs


def summary_for(own, source_rows):
    misses = [r for r in own if r['observed']['native_release'] and not r['observed']['capture']]
    rejects = [r for r in own if not r['observed']['native_release']]
    crossing = [r for r in misses if r['observed'].get('rim_forward_error_mm') is not None]
    return dict(**statistics(source_rows),
        native_first_flight_unknown=sum(r['observed']['native_release'] and not r['observed']['first_flight_available'] for r in own),
        no_entry_failures=sum(not r['observed']['whole_entry'] for r in misses),
        bucket_retention_but_route_failures=sum(r['observed']['bucket_capture'] for r in misses),
        cutoff_contact_counts=dict(Counter('|'.join(r['observed'].get('cutoff_contact_channels', [])) or 'unknown' for r in misses)),
        early_robot_contact_ms_threshold=EARLY_ROBOT_CONTACT_MS,
        early_robot_contact_count=sum(r['observed'].get('early_robot_contact') is True for r in misses),
        observed_precontact_rim_crossings=len(crossing),
        observed_precontact_short_crossings=sum(r['observed']['rim_forward_error_mm'] < 0 for r in crossing),
        observed_precontact_long_crossings=sum(r['observed']['rim_forward_error_mm'] > 0 for r in crossing),
        no_precontact_rim_crossing=len(misses)-len(crossing),
        rejection_reason_counts=dict(Counter(r.get('setup_proof', {}).get('reason') or 'unknown' for r in rejects)),
        native_miss_contexts=[r['context_id'] for r in misses], rejected_contexts=[r['context_id'] for r in rejects])


def build(study, audit, output):
    study, output = Path(study).resolve(), Path(output).resolve()
    require(not output.exists() and not output.is_relative_to(study), 'New derived directory outside immutable source required')
    checks = Checks(); study, report, spec, proof, manifest, packet, rows = verify_inputs(study, audit, checks)
    # Pin the reused algorithm to the exact one present in the audited producer.
    helper = Path(prior.__file__)
    expected_sources = {**spec['source_sha256'], **spec['native_source_sha256']}
    helper_sources = {}
    for name in ('consistent_flight_diagnosis', 'consistent_flight_selector', 'ensemble_flight_diagnosis',
                 'curate_physics_ensemble', 'curate_xy650', 'xy650_verify', 'calibrated_score_oracle'):
        relative = 'sircus/mjlab/'+name+'.py'
        helper_sources[relative] = sha(helper.with_name(name+'.py'))
        checks(helper_sources[relative] == expected_sources[relative], 'Reused CPU helper matches audited producer/'+name)
    calibration = read(study/'calibration.json')['score_config']
    gravity = spec['native_reference']['allocated_options']['native']['gravity']
    logs = {r['context_id']: r for r in read(study/'rankings.json')}
    tests = [r for r in rows if r['role'] == 'test']
    items = {r['id']: make_item(r, logs[r['context_id']], spec['scene'], calibration, gravity) for r in tests}
    groups = {}; consumed = {}
    for row in tests:
        objects = {}
        for field, hashfield in (('context_resolution_file', 'context_resolution_sha256'), ('connection_file', 'connection_sha256')):
            if row.get(field):
                path = relative_file(study, 'native/'+row[field]); actual = sha(path)
                checks(actual == row[hashfield] == manifest['native/'+row[field]], row['id']+'/bound '+field)
                objects[field] = read(path)
        item = items[row['id']]
        item['selection']['context_reason'] = objects.get('context_resolution_file', {}).get('reason')
        item['selection']['connection_reason'] = objects.get('connection_file', {}).get('reason')
        item['setup_proof'] = setup_proof(row, objects.get('context_resolution_file', {}), objects.get('connection_file', {}))
        if row['native_release']: groups.setdefault(row['native_cache_file'], []).append(row)
    for name, group in groups.items():
        path = relative_file(study, 'native/'+name); actual = sha(path); consumed[name] = actual
        checks(actual == manifest['native/'+name], 'Dense cache bound/'+name)
        with np.load(path, allow_pickle=False) as z:
            arrays = {k: z[k] for k in ('position_m', 'velocity_m_s', 'forces_N', 'time_s', 'dt_s', 'contact_distance_m')}
            ids = z['record_ids'].tolist()
            for row in group:
                col = row['native_cache_row']; ident = row['id']
                checks(str(ids[col]) == ident and actual == row['native_cache_sha256'], ident+'/exact cache column')
                trace = {k: v[:, col] if v.ndim >= 2 and k != 'time_s' else v for k, v in arrays.items()}
                derived = prior.analyze_trace(row, spec['scene'], calibration, trace, gravity)
                item = items[ident]; item['observed'].update(derived['observed'])
                item['trace'] = derived['trace']; item['forecasts'] = derived['forecasts']
                item['diagnosis'] = prior.classification(item['observed']); enrich_flight(item)
                checks(type(derived['metric']['available']) is bool and
                    (derived['metric']['available'] or isinstance(derived['metric'].get('reason'), str)), ident+'/explicit flight availability')
    all_items = list(items.values()); own = [r for r in all_items if r['method'] == 'consistent']
    paired, pair_table = paired_results(all_items)
    misses = [r for r in own if r['observed']['native_release'] and not r['observed']['capture']]
    rejects = [r for r in own if not r['observed']['native_release']]
    summary = summary_for(own, [r for r in tests if r['method'] == 'consistent'])
    checks(len(own) == 50 and len(misses) == 18 and len(rejects) == 8, 'Original 50 denominator and 18/8 diagnostic scope retained')
    example_contexts = ['gap-coverage-test-'+n for n in ('48', '25', '07', '27', '16', '01', '06', '41')]
    cases = []
    for item in own:
        c = copy.deepcopy(item); c['featured'] = c['context_id'] in example_contexts
        c['references'] = [scalar(r) for r in all_items if r['context_id'] == c['context_id'] and r['method'] != 'consistent']
        c['bucket_rim_z_m'] = calibration.get('compiled_rim_m')
        cases.append(c)
    findings = [
        dict(status='observed', statement='Original and wider-gap selectors each captured 24/50; each had 18 native misses and 8 setup rejections.', evidence='Original 112-request comparison, including 12 controls; no new rollout or labels'),
        dict(status='observed', statement='First-flight cutoff contacts among the 18 misses: '+json.dumps(summary['cutoff_contact_counts'], sort_keys=True), evidence='Recorded aggregate force channels; exact robot link is not logged and bucket contact does not establish a rim hit'),
        dict(status='observed', statement=str(summary['bucket_retention_but_route_failures'])+' misses entered and retained the ball for 1 s but failed the complete route contract.', evidence='Preserved bucket-retention and complete-shot labels are separate'),
        dict(status='observed', statement=str(summary['early_robot_contact_count'])+' misses met robot contact within 50 ms after rebound.', evidence='Descriptive threshold only; no new success rule'),
        dict(status='limitation', statement='Finite IK rejection does not prove a target is unreachable. Endpoint self-clearance and fixed-base obstruction are distinct failures.', evidence='Byte-bound saved branch searches and clearance witnesses'),
        dict(status='limitation', statement='Absent pre-contact rim crossings remain unknown. Model confidence and ballistic curves do not replace native capture evidence.', evidence='Contact-limited reconstruction; one nominal request per method/layout, no repeats')]
    result = dict(version=VERSION, status='completed', scope=SCOPE,
        source=dict(verified=True, report_sha256=sha(study/'report.json'), audit_sha256=sha(audit),
            manifest_sha256=sha(study/'evidence-manifest.json'), contract_fingerprint=spec['fingerprint'],
            original_total_requests=report['counts']['requested'], original_native_releases=report['counts']['native_releases'],
            original_native_audit_checks=proof['checks_count'], native_requests_added=0,
            records_sha256=manifest['records.jsonl'], rankings_sha256=manifest['rankings.json']),
        summary=summary, comparisons=copy.deepcopy(report['comparisons']), paired=paired, paired_table=pair_table,
        requests=[scalar(r) for r in own], misses=[scalar(r) for r in misses], rejects=[scalar(r) for r in rejects],
        cases=cases, example_contexts=example_contexts, findings=findings,
        next_steps=['Freeze one targeted clearance correction and matched diagnostic control tests before outcomes.',
                    'Retain all 50 requested contexts, including setup rejects, in every comparison.',
                    'Use fresh held-out positions before claiming improvement or generalization.'],
        geometry=dict(ball_radius_m=spec['scene']['ball']['radius_m'], rim_z_m=calibration.get('compiled_rim_m'),
            board_size_m=spec['scene']['board']['size_m'], bucket_height_status=spec['scene']['bucket']['height_interpretation']),
        processing=dict(cuda_used=False, training=False, new_native_requests=0, source_sha256=sha(__file__),
            helper_sha256=sha(helper), helper_sources_sha256=helper_sources, consumed_cache_sha256=consumed))
    output.mkdir(parents=True)
    derived = dict(version=AUDIT_VERSION, status='passed', checks_count=len(checks.entries), checks=checks.entries,
        source_report_sha256=result['source']['report_sha256'], source_audit_sha256=result['source']['audit_sha256'],
        diagnosis_source_sha256=sha(__file__))
    (output/'checks.json').write_text(json.dumps(derived, indent=2, allow_nan=False)+'\n')
    result['audit'] = dict(version=AUDIT_VERSION, status='passed', checks_count=len(checks.entries), full_checks_verified=True,
        source_sha256=sha(output/'checks.json'), source_report_sha256=result['source']['report_sha256'],
        source_audit_sha256=result['source']['audit_sha256'], scope='Derived binding/reconstruction checks; original native audit remains separate')
    (output/'diagnosis.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    reduced = copy.deepcopy(result); reduced['cases'] = [scalar(r) for r in result['cases']]
    (output/'summary.json').write_text(json.dumps(reduced, indent=2, allow_nan=False)+'\n')
    (output/'checksums.json').write_text(json.dumps({n: sha(output/n) for n in ('diagnosis.json', 'summary.json', 'checks.json')}, indent=2)+'\n')
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('study', 'audit', 'output'): p.add_argument('--'+key, type=Path, required=True)
    a = p.parse_args(); result = build(a.study, a.audit, a.output)
    print(json.dumps(dict(status=result['status'], summary=result['summary'], checks=result['audit']['checks_count'])))


if __name__ == '__main__': main()
