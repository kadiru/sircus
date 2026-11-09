"""Publish audit-bound failure evidence; no simulation, inference or service control."""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
from pathlib import Path
import re
from .prepare_table_training import atomic_bytes

VERSION = 'sircus-table-training-diagnosis-v1'
AUDIT = 'sircus-table-training-diagnosis-audit-v1'
CORRECTION = 'sircus-table-training-correction-v1'
CORRECTION_AUDIT = 'sircus-table-training-correction-audit-v1'
KEY = 'table_training_diagnosis'
START = '<!--TABLE_TRAINING_DIAGNOSIS_START-->'
END = '<!--TABLE_TRAINING_DIAGNOSIS_END-->'
SOURCE = {
    'report_sha256': '4605bcb801351ae91ad16b71a033ceb3866027db621f9d84227f23b1ef257e95',
    'audit_sha256': '6e50d87ddb7b2d4b37debbc2a5ac96507ff40bc236223277121acd385b8ec91b',
    'rows_sha256': 'ddb1ecd67b71951fe5d1152a496c1bbf5b01e4174b38a7f7b8af7b813f4817e3',
    'contract_fingerprint': '20f198caec28ff0402e8f7d2f7d2d1de65434b294f964a3ea35cde36bc3a1114',
}
COUNTS = ('attempts', 'native_releases', 'captures', 'setup_rejections', 'robot_contact_shots', 'forbidden_contact_shots', 'native_misses', 'invalid_physics')
CASE_KEYS = ('context_id', 'record_id', 'record_sha256', 'bucket_xy_m', 'values', 'native_release', 'capture',
             'reason', 'group', 'table_contact', 'whole_entry', 'retention_s', 'closest_approach_m',
             'first_flight_forward_error_m', 'first_flight_lateral_error_m', 'board_bounce_s', 'board_center_m', 'board_normal', 'board_bucket_distance_m', 'forbidden_contact', 'whole_ball_entry', 'max_retention_s')

def read(path): return json.loads(Path(path).read_text())
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def take(raw, keys): return {key: copy.deepcopy(raw[key]) for key in keys if isinstance(raw, dict) and key in raw}
def report_path(value):
    path = Path(value)
    return path / 'report.json' if path.is_dir() else path

def default_audit(path):
    return path.parent.with_name(path.parent.name + '-audit.json') if path.name == 'report.json' else path.with_name(path.stem + '-audit.json')

def proof(path, audit_path, version, raw):
    result = {'status': 'pending', 'full_checks_verified': False}
    if not audit_path.is_file(): return result
    evidence = read(audit_path)
    checks = evidence.get('checks')
    passed = (raw.get('status') == 'completed' and re.fullmatch('[a-f0-9]{64}', raw.get('contract_fingerprint', '')) is not None
              and evidence.get('version') == version and evidence.get('status') == 'passed'
              and evidence.get('full_checks_verified') is True
              and evidence.get('contract_fingerprint') == raw.get('contract_fingerprint')
              and evidence.get('report_sha256') == sha(path) and not evidence.get('failures')
              and isinstance(checks, list) and bool(checks) and evidence.get('checks_count') == len(checks)
              and all(isinstance(check, dict) and check.get('passed') is True for check in checks))
    return dict(status='passed' if passed else 'unverified', full_checks_verified=passed,
                version=version, checks_count=evidence.get('checks_count'), report_sha256=sha(path),
                contract_fingerprint=raw.get('contract_fingerprint'), proof_sha256=sha(audit_path))

def compact_case(raw):
    out = take(raw, CASE_KEYS)
    out['events'] = [take(event, ('label', 'time_s')) for event in raw.get('events', [])]
    out['first_post_board_contact'] = take(raw.get('first_post_board_contact'), ('label', 'category', 'time_s'))
    if isinstance(raw.get('setup_proof'), dict):
        keys = ('connection_sha256', 'resolution_sha256', 'reason', 'connection_policy', 'action_substituted', 'layout_eligible', 'anchor_admitted', 'requested_values', 'anchor_values', 'valid_ik_branches', 'required_clearance_m', 'interpretation')
        out['setup_proof'] = take(raw['setup_proof'], keys)
        out['setup_proof']['home_geometry'] = take(raw['setup_proof'].get('home_geometry'), ('passed', 'reason', 'minimum_clearance_m', 'closest_pair', 'inconsistent_distance_queries', 'unresolved_pair_count', 'fixed_base_blocker_count'))
    if isinstance(raw.get('trace'), dict):
        trace = take(raw['trace'], ('record_id', 'record_sha256', 'native_cache_sha256', 'time_s', 'position_m', 'sample_basis', 'scope'))
        if len(trace.get('time_s', [])) > 1201: raise ValueError('Display trace must be an explicitly sampled subset, at most 1201 points')
        out['trace'] = trace
    return out

def compact(diagnosis, audit_path=None, correction=None, correction_audit=None):
    path = report_path(diagnosis)
    raw = read(path)
    if raw.get('version') != VERSION: raise ValueError('Explicit table-training diagnosis required')
    if not all(raw.get('source', {}).get(key) == value for key, value in SOURCE.items()):
        raise ValueError('Diagnosis must bind the completed 50-position training evidence')
    out = take(raw, ('version', 'status', 'contract_fingerprint', 'scope', 'interpretation'))
    out['source'] = dict(SOURCE, **take(raw.get('source'), ('evidence_manifest_sha256', 'calibration_sha256', 'contract_sha256')))
    out['summary'] = take(raw.get('summary'), ('requested', 'captures', 'flight_misses', 'setup_rejections', 'forbidden_contact_misses', 'no_forbidden_contact_misses'))
    out['groups'] = [take(group, ('label', 'count', 'description')) for group in raw.get('groups', [])]
    out['cases'] = [compact_case(case) for case in raw.get('cases', [])]
    out['comparators'] = [compact_case(case) for case in raw.get('comparators', [])]
    out['scene'] = take(raw.get('scene'), ('board_size_m', 'bucket_base_z_m', 'bucket_height_m', 'bucket_inner_radius_m', 'bucket_outer_radius_m', 'ball_radius_m', 'geometry_basis'))
    out['audit'] = proof(path, Path(audit_path) if audit_path else default_audit(path), AUDIT, raw)
    out['correction'] = {'version': CORRECTION, 'status': 'not_started', 'audit': {'status': 'pending', 'full_checks_verified': False}}
    if correction:
        cpath = report_path(correction)
        if cpath.is_file():
            c = read(cpath)
            if c.get('version') != CORRECTION: raise ValueError('Explicit table-training correction required')
            v = take(c, ('version', 'status', 'contract_fingerprint', 'description', 'changed_parameter', 'unchanged', 'expected_requests', 'elapsed_s', 'target_value'))
            variant_rows = c.get('counts_by_variant', [])
            if isinstance(variant_rows, dict): variant_rows = [dict(counts, variant=variant) for variant, counts in variant_rows.items()]
            v['counts_by_variant'] = [take(row, ('variant',) + COUNTS) for row in variant_rows]
            case_rows = []
            for case in c.get('cases', []):
                if isinstance(case.get('variants'), dict):
                    case_rows.extend(dict(counts, context_id=case.get('context_id'), variant=variant, bucket_xy_m=case.get('bucket_xy_m'), values=case.get('baseline_values' if variant == 'baseline' else 'corrected_values')) for variant, counts in case['variants'].items())
                else: case_rows.append(case)
            v['cases'] = [take(row, ('context_id', 'variant', 'values', 'bucket_xy_m') + COUNTS) for row in case_rows]
            v['control_gates'] = {key: dict(take(value, ('status', 'direct_captures', 'table_assisted_captures', 'negative_captures')), counts=take(value.get('counts'), COUNTS)) for key, value in c.get('control_gates', {}).items() if key in ('pre', 'post')}
            v['audit'] = proof(cpath, Path(correction_audit) if correction_audit else default_audit(cpath), CORRECTION_AUDIT, c)
            out['correction'] = v
    json.dumps(out, allow_nan=False)
    return out

def publish(diagnosis, display, **kwargs):
    path = Path(display)
    if not path.is_file(): raise ValueError('Existing display required')
    snapshot = compact(diagnosis, **kwargs)
    for _ in range(3):
        original = path.read_bytes(); value = json.loads(original); value[KEY] = snapshot
        try:
            atomic_bytes(path, (json.dumps(value, allow_nan=False, indent=2) + '\n').encode(), original)
            return snapshot
        except RuntimeError: continue
    raise RuntimeError('Concurrent display writer preserved')

def embed_html(original, snapshot):
    if original.count(START) != original.count(END) or original.count(START) > 1: raise ValueError('Ambiguous diagnosis markers')
    web = Path(__file__).with_name('web')
    html = (web / 'table-training-diagnosis.html').read_text()
    css = (web / 'table-training-diagnosis.css').read_text()
    js = (web / 'table-training-diagnosis.js').read_text()
    js = re.sub(r'^export (?=(?:const|function|class)\b)', '', js, flags=re.M)
    if re.search(r'^\s*(?:import|export)\b', js, re.M) or '</script' in js.lower() or '</style' in css.lower(): raise ValueError('Self-contained inline assets required')
    data = json.dumps(snapshot, allow_nan=False).replace('<', '\\u003c').replace('>', '\\u003e').replace('&', '\\u0026')
    block = f'{START}\n<style>{css}</style>\n{html}\n<script type="application/json" id="table-training-diagnosis-snapshot">{data}</script>\n<script>(function(){{\n{js}\n}})();</script>\n{END}'
    if START in original:
        a, b = original.index(START), original.index(END) + len(END)
        if b < a: raise ValueError('Diagnosis markers out of order')
        return original[:a] + block + original[b:]
    closing = list(re.finditer(r'</body\s*>', original, re.I))
    if not closing: raise ValueError('Existing document body required')
    at = closing[-1].start()
    return original[:at] + block + '\n' + original[at:]

def install(target, snapshot, expected_sha256):
    target = Path(target); original = target.read_bytes()
    if hashlib.sha256(original).hexdigest() != expected_sha256: raise ValueError('Target HTML changed; existing work preserved')
    atomic_bytes(target, embed_html(original.decode(), snapshot).encode(), original)

def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('diagnosis', 'audit', 'correction', 'correction-audit', 'display', 'target-html', 'snapshot'):
        p.add_argument('--' + name, type=Path, required=name == 'diagnosis')
    p.add_argument('--expected-html-sha256'); a = p.parse_args()
    options = dict(audit_path=a.audit, correction=a.correction, correction_audit=a.correction_audit)
    value = publish(a.diagnosis, a.display, **options) if a.display else compact(a.diagnosis, **options)
    if a.target_html: install(a.target_html, value, a.expected_html_sha256)
    if a.snapshot:
        for source in (a.diagnosis, a.correction):
            if source and a.snapshot.resolve().is_relative_to(report_path(source).parent.resolve()):
                raise ValueError('Keep display snapshot outside immutable evidence')
        atomic_bytes(a.snapshot, (json.dumps(value, allow_nan=False, indent=2) + '\n').encode())
    print(json.dumps(dict(status=value.get('status'), audit=value.get('audit'), correction=value['correction'].get('status'))))
if __name__ == '__main__': main()
