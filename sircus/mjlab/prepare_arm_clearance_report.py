"""Publish a separate read-only comparison only after complete native proof.

Progress is display-only. Original reports, traces, checkpoints and old UI keys
are never changed, and no native runtime or hardware client is imported.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path

KEY = 'arm_clearance_comparison'
VERSION = 'sircus-arm-clearance-study-v1'
AUDIT_VERSION = 'sircus-arm-clearance-audit-v1'
METHODS = ('consistent', 'arm_clearance')
COUNT_FIELDS = ('requested', 'native_releases', 'captures', 'clean', 'rejected', 'native_misses', 'invalid')
SEGMENTS = ('before_terminal', 'post_rebound', 'after_terminal')


def read(path): return json.loads(Path(path).read_text())
def sha(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''): result.update(block)
    return result.hexdigest()
def digest(value): return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()
def require(value, description):
    if not value: raise ValueError(description)
def integer(value): return type(value) is int and value >= 0
def finite(value): return type(value) in (int, float) and math.isfinite(value)
def vector(value, size): return isinstance(value, list) and len(value) == size and all(finite(v) for v in value)
def valid_counts(value):
    return (isinstance(value, dict) and all(integer(value.get(k)) for k in COUNT_FIELDS)
        and value['requested'] == value['native_releases']+value['rejected']
        and value['native_releases'] == value['captures']+value['native_misses']+value['invalid']
        and value['clean'] <= value['captures'])


def verify_manifest(study, raw):
    manifest = read(study/'evidence-manifest.json')
    actual = {str(p.relative_to(study)) for p in study.rglob('*')
              if p.is_file() and p.name not in ('report.json', 'evidence-manifest.json')}
    require(isinstance(manifest, dict) and set(manifest) == actual and raw.get('evidence_sha256') == manifest,
            'Complete unchanged native manifest required')
    for name, expected in manifest.items():
        path = (study/name).resolve()
        require(path.is_relative_to(study.resolve()) and sha(path) == expected, 'Native artifact changed: '+name)
    return sha(study/'evidence-manifest.json')


def validate_outcomes(raw, proof):
    require(raw.get('status') == raw.get('phase') == 'completed', 'Completed native report required')
    totals, groups, gates = raw.get('counts'), raw.get('comparisons'), raw.get('control_gates')
    require(valid_counts(totals) and totals['requested'] == 112, 'All112 requested slots required')
    require(isinstance(groups, list) and [r.get('method') for r in groups] == list(METHODS)
            and all(valid_counts(r.get('counts')) and r['counts']['requested'] == 50 for r in groups), 'Both50-position denominators required')
    require(isinstance(gates, dict) and set(gates) == {'pre', 'post'} and proof.get('control_gates') == gates,
            'Matched native control gates required')
    counts = [r['counts'] for r in groups]
    for phase in ('pre', 'post'):
        gate = gates[phase]; c = gate.get('counts', {})
        require(gate.get('status') == 'passed' and valid_counts(c)
                and all(c[k] == v for k, v in dict(requested=6, native_releases=6, captures=3, rejected=0, native_misses=3, invalid=0).items()),
                'All six matched controls must behave as expected: '+phase)
        counts.append(c)
    require(all(totals[k] == sum(c[k] for c in counts) for k in COUNT_FIELDS), 'Controls and both methods must sum to overall counts')
    require(proof.get('comparisons') == groups and proof.get('native_releases') == totals['native_releases'], 'Independent measured totals must match')


def verified_positions(raw, proof):
    base, enriched = raw.get('test_positions'), proof.get('test_positions')
    require(isinstance(base, list) and isinstance(enriched, list) and len(base) == len(enriched) == 100,
            'All100 paired native outcomes required')
    originals = {row['record_id']: row for row in base}
    require(len(originals) == len({r['record_id'] for r in enriched}) == 100, 'Unique exact native record identities required')
    values = []
    for source in enriched:
        old = originals.get(source['record_id'])
        require(old is not None and all(source.get(k) == value for k, value in old.items()), 'Audited contact fields cannot replace original outcomes')
        require(source.get('method') in METHODS and isinstance(source.get('context_id'), str)
                and bool(source['context_id']) and vector(source.get('bucket_xy_m'), 2) and vector(source.get('values'), 5)
                and all(type(source.get(k)) is bool for k in ('capture', 'native_release', 'clean', 'invalid', 'clearance_rejected')),
                'Exact finite settings and explicit outcome flags required')
        require((not source['capture'] or source['native_release'] and not source['invalid'])
                and (not source['clean'] or source['capture']) and (not source['clearance_rejected'] or not source['native_release']),
                'Capture, invalid and abstention flags must agree')
        for segment in SEGMENTS:
            observed = source.get('robot_contact_'+segment)
            require(observed is None or type(observed) is bool, 'Missing contact observations must remain unknown')
            require(source['native_release'] or observed is None, 'Rejected requests have no measured robot contact')
        # Scalar-only allowlist. No raw trajectories, local paths or calibration.
        allowed = ('context_id', 'method', 'record_id', 'bucket_xy_m', 'values', 'capture', 'native_release', 'clean', 'invalid',
                   'whole_ball_entry', 'retention_s', 'clearance_rejected', 'predicted_clearance_min_m',
                   'predicted_clearance_closest_geom', 'first_post_bounce_contact_channels', 'first_contact_delay_ms',
                   'robot_contact_before_terminal', 'robot_contact_post_rebound', 'robot_contact_after_terminal')
        values.append({key: copy.deepcopy(source.get(key)) for key in allowed})
    before = {}
    for method in METHODS:
        group = [r for r in values if r['method'] == method]
        require(len(group) == len({r['context_id'] for r in group}) == 50, 'Each method needs the same50 unique positions')
        expected = next(r['counts'] for r in raw['comparisons'] if r['method'] == method)
        observed = dict(requested=len(group), native_releases=sum(r['native_release'] for r in group), captures=sum(r['capture'] for r in group),
                        clean=sum(r['clean'] for r in group), rejected=sum(not r['native_release'] for r in group),
                        native_misses=sum(r['native_release'] and not r['capture'] and not r['invalid'] for r in group),
                        invalid=sum(r['native_release'] and r['invalid'] for r in group))
        require(observed == expected, 'Every displayed outcome belongs to the audited counts')
        if method == METHODS[0]: before = {r['context_id']: r for r in group}
        else: require(all(r['context_id'] in before and r['bucket_xy_m'] == before[r['context_id']]['bucket_xy_m'] for r in group), 'Exactly paired bucket coordinates required')
    return values


def verified_contacts(positions, proof):
    groups = proof.get('contact_comparisons')
    require(isinstance(groups, list) and [g.get('method') for g in groups] == list(METHODS), 'Both audited native contact cohorts required')
    result = []
    for group in groups:
        items = [r for r in positions if r['method'] == group['method']]
        native = [r for r in items if r['native_release']]
        known = [r for r in native if all(type(r['robot_contact_'+s]) is bool for s in SEGMENTS)]
        expected = dict(method=group['method'], requested=50, native_releases=len(native), observed_native=len(known), missing_native=len(native)-len(known), force_threshold_N=.01)
        expected.update({'robot_contact_trials_'+s: sum(r['robot_contact_'+s] for r in known) for s in SEGMENTS})
        require(all(group.get(k) == v for k, v in expected.items()), 'Measured contact totals must match exact recorded trials')
        result.append(expected)
    return result


def compact(study, audit_path=None):
    # Imported here so the publisher remains usable without importing native or GPU code.
    from .arm_clearance_study import protocol
    from .arm_clearance_selector import RULE, fingerprint as rule_fingerprint
    study = Path(study).resolve(); raw, spec = read(study/'report.json'), read(study/'contract.json')
    unsigned = dict(spec); fingerprint = unsigned.pop('fingerprint', None)
    require(spec.get('version') == raw.get('version') == VERSION and digest(unsigned) == fingerprint
            and raw.get('contract_fingerprint') == fingerprint and raw.get('protocol') == spec.get('protocol') == protocol()
            and spec.get('maximum_requests') == 112 and spec.get('new_training_requests') == 0 and spec.get('clearance_rule') == RULE,
            'Exact frozen50/112/fixed64/no-fit contract required')
    plan = study/'prediction-plan.json'
    plan_hash = sha(plan) if plan.is_file() else None
    if raw.get('prediction_plan_sha256'): require(raw['prediction_plan_sha256'] == plan_hash, 'Frozen plan changed')
    status = {'prepared': 'pending', 'completed': 'awaiting_audit'}.get(raw.get('status'), raw.get('status', 'unknown'))
    value = dict(version=VERSION, status=status, phase=raw.get('phase'), scope=raw.get('scope'), protocol=copy.deepcopy(raw['protocol']),
                 rule_description=('All five frozen predicted post-board flights must clear both arms at their exact admitted held joints, '
                    'with 2 mm extra ball margin. Samples are at most 2 ms apart with a swept-distance bound between them. '
                    'The check stops at the descending rim-band boundary; absent or over-1.5-second predictions are rejected. '
                    'Only the intended impact board is excluded. Incoming drop, arm transit and retention are not checked by this rule.'),
                 clearance_rule=copy.deepcopy(RULE), rule_sha256=rule_fingerprint(RULE),
                 planned_layouts=raw.get('planned_layouts', 0), progress_requests=raw.get('counts', {}).get('requested'),
                 report_sha256=sha(study/'report.json'), plan_sha256=plan_hash, contract_fingerprint=fingerprint,
                 audit=dict(passed=None), comparisons=[], test_positions=[], contact_comparisons=[])
    if raw.get('error'): value['error'] = raw['error']
    if raw.get('status') == 'queued': value['queue'] = dict(reason='Waiting for the existing RTX4090 workload and shared lock; both left untouched.', wait_s=raw.get('queue', {}).get('wait_s'))
    if not audit_path or not Path(audit_path).is_file(): return value
    proof = read(audit_path)
    if proof.get('status') == 'failed':
        value.update(status='audit_failed', audit=dict(passed=False)); return value
    checks = proof.get('checks'); manifest_hash = verify_manifest(study, raw)
    require(proof.get('version') == AUDIT_VERSION and proof.get('status') == 'passed' and proof.get('failures') == []
            and isinstance(checks, list) and checks and type(proof.get('checks_count')) is int and proof['checks_count'] == len(checks)
            and all(isinstance(c, dict) and c.get('passed') is True and isinstance(c.get('label'), str) and c['label'].strip() for c in checks)
            and proof.get('auditor_sha256') == sha(Path(__file__).with_name('arm_clearance_verify.py'))
            and proof.get('report_sha256') == value['report_sha256'] and proof.get('contract_fingerprint') == fingerprint
            and proof.get('evidence_manifest_sha256') == manifest_hash
            and proof.get('prediction_plan_sha256', proof.get('plan_sha256')) == plan_hash and plan_hash is not None
            and proof.get('requested_trials') == 112 and proof.get('training_requests') == 0
            and proof.get('producer_source_sha256') == spec['source_sha256']
            and proof.get('native_source_sha256') == spec['native_source_sha256'], 'Complete matching independent native proof required')
    validate_outcomes(raw, proof)
    positions = verified_positions(raw, proof); contacts = verified_contacts(positions, proof)
    value.update(status='verified', counts=copy.deepcopy(raw['counts']), comparisons=copy.deepcopy(raw['comparisons']),
                 control_gates=copy.deepcopy(raw['control_gates']), test_positions=positions, contact_comparisons=contacts,
                 manifest_sha256=manifest_hash, audit=dict(version=AUDIT_VERSION, passed=True, full_checks_verified=True,
                    checks_count=len(checks), source_sha256=sha(audit_path), auditor_sha256=proof['auditor_sha256'],
                    report_sha256=value['report_sha256'], plan_sha256=plan_hash, manifest_sha256=manifest_hash, contract_fingerprint=fingerprint))
    return value


def publish(display, study, audit_path=None, pipeline=None):
    display, study = Path(display).resolve(), Path(study).resolve()
    require(not display.is_relative_to(study), 'Display must stay outside immutable study')
    result = compact(study, audit_path)
    if pipeline and pipeline.get('status') in ('failed', 'audit_failed'):
        result.update(status='failed', error=pipeline.get('error') or 'Comparison pipeline failed', audit=dict(passed=False),
                      comparisons=[], test_positions=[], contact_comparisons=[])
    for _ in range(3):
        previous = display.read_bytes(); value = json.loads(previous)
        require(isinstance(value, dict), 'Existing object display required')
        value[KEY] = result
        if display.read_bytes() != previous: continue
        temp = display.with_suffix(display.suffix+'.arm-clearance.tmp')
        temp.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
        if display.read_bytes() != previous: temp.unlink(); continue
        temp.replace(display); return result
    raise RuntimeError('Concurrent display writer preserved')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('display', 'study'): parser.add_argument('--'+key, type=Path, required=True)
    parser.add_argument('--audit', type=Path); args = parser.parse_args()
    publish(args.display, args.study, args.audit)


if __name__ == '__main__': main()
