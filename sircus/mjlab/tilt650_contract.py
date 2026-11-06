"""Frozen five-recipe 30-degree verification; no adaptive search or new policy."""
import copy
import math
from pathlib import Path

from . import xy650_contract as xy
from .corrected_contract import sha
from .learning_contract import digest
from .replay650_contract import sensitivity, verify_manifest

VERSION = 'sircus-tilt30-650-verification-v1'
AUDIT_VERSION = 'sircus-tilt30-650-independent-audit-v1'
PARENT_FINGERPRINT = '7db7440c12005c39808ee4dfbf831d8b81000cda048a8d4eddc896d9b1d31137'
PARENT_AUDIT_SHA = '9a605b7b8fbb1cb99db966ae92f6488aab0895a5db4bc9116b05e5a7fdc0b439'
SOURCE_IDS = tuple('coarse-64-xy-grid-%04d-r0' % i for i in (43, 70, 96, 151, 177))
STEPS = (64, 128)
SCOPE = ('Five exact previously unverified 30-degree recipes from the completed XY650 development study; '
         'three nominal repeats per resolution and matched historical controls. Privileged geometry and '
         'held articulated endpoints only. No new search, PPO, noise, held-out generalization, grasp/transit, '
         'actuator tracking, perception or physical validation. Parent rim captures remain rim captures.')
read = xy.read
statistics = xy.statistics
control_gate = xy.control_gate
captured = xy.captured
clean = xy.clean


def settings():
    return dict(source_count=5, source_ids=list(SOURCE_IDS), repeats_per_resolution=3,
                substeps=list(STEPS), control_requests=24, verification_requests=30,
                max_requests=54, minimum_clearance_m=.005, max_drift_350ms_m=.005,
                source_selection='Exact five native-valid coarse 30-degree captures not previously verified; literal source IDs',
                qualification='Fresh repeated capture and clean gates are separate; no inherited clean-source claim',
                scope=SCOPE)


def load_parent(parent, parent_audit):
    parent, parent_audit = Path(parent), Path(parent_audit)
    spec, report, proof = (read(p) for p in (parent/'contract.json', parent/'report.json', parent_audit))
    unsigned = copy.deepcopy(spec)
    fingerprint = unsigned.pop('fingerprint', None)
    checks = proof.get('checks', [])
    if not (digest(unsigned) == fingerprint == PARENT_FINGERPRINT
            and sha(parent_audit) == PARENT_AUDIT_SHA
            and spec.get('version') == report.get('version') == xy.VERSION
            and report.get('status') == report.get('phase') == 'completed'
            and report.get('contract_fingerprint') == fingerprint
            and proof.get('version') == 'sircus-xy650-independent-audit-v2'
            and proof.get('status') == 'passed' and not proof.get('failures')
            and proof.get('contract_fingerprint') == fingerprint
            and checks and proof.get('checks_count') == len(checks)
            and all(c.get('passed') is True for c in checks)
            and proof.get('report_sha256') == sha(parent/'report.json')
            and proof.get('evidence_manifest_sha256') == sha(parent/'evidence-manifest.json')):
        raise ValueError('Exact completed parent and byte-bound external audit-v2 proof required')
    verify_manifest(parent, 'evidence-manifest.json')
    rows = [__import__('json').loads(line) for line in (parent/'records.jsonl').read_text().splitlines() if line]
    eligible = [r for r in rows if r['phase'] == 'coarse' and r['substeps'] == 64
                and r['values'][3] == 30. and captured(r)
                and not r.get('invalid') and r.get('invalid_physics') is False]
    eligible.sort(key=lambda r: r['id'])
    if tuple(r['id'] for r in eligible) != SOURCE_IDS:
        raise ValueError('Literal five-source selection changed')
    previous = [g for g in read(parent/'groups.json') if g['role'] == 'verification']
    target = lambda r: tuple(round(float(v), 9) for v in r['values'] + r['bucket_xy_m'])
    if len({target(r) for r in eligible}) != 5 or any(target(r) == target(g) for r in eligible for g in previous):
        raise ValueError('Sources must be distinct and previously unverified')
    expected = [(-.01,.06,.41),(-.01,.08,.41),(0.,.04,.39),(0.,.08,.41),(.01,.04,.39)]
    for row, (dx, dy, gap) in zip(eligible, expected):
        if not (row['translation_m'] == [dx, dy] and row['values'][2:] == [.38, 30., gap]
                and row.get('route_quality') == 1 and row.get('rim_band_contact') is True
                and abs(math.dist(row['values'][:2], row['bucket_xy_m']) - .65) < 1e-12):
            raise ValueError('Exact source geometry or historical rim classification changed')
    controls = [copy.deepcopy(g) for g in read(parent/'initial-groups.json') if g['role'].endswith('_control')]
    if [g['role'] for g in controls] != ['positive_control', 'negative_control']:
        raise ValueError('Exact historical control pair required')
    bindings = dict(parent={n: sha(parent/n) for n in ('contract.json','report.json','evidence-manifest.json',
                         'records.jsonl','groups.json','initial-groups.json')}, parent_audit_sha256=sha(parent_audit))
    return spec, eligible, controls, bindings


def groups(sources, controls, scene_fp):
    result = copy.deepcopy(controls)
    for i, (group, source) in enumerate(zip(xy.verification(sources), sources)):
        group.update(id='tilt30-%02d' % i, parent_route_quality=source['route_quality'])
        result.append(group)
    return xy.bind(result, scene_fp)


def qualification(rows, positions, *, controls_passed, parent_source_capture):
    present = len(rows) == 6 and all(sum(r['substeps'] == s for r in rows) == 3 for s in STEPS)
    capture_both = present and all(captured(r) for r in rows)
    clean_both = present and all(clean(r) for r in rows)
    physics = present and all(r.get('native_release') and not r.get('invalid') and r.get('invalid_physics') is False for r in rows)
    oracle = present and all(r.get('scalar_score_audit_passed') is True for r in rows)
    drift = sensitivity(rows, positions)['nominal_position_delta_350ms_max_m'] if present else None
    common = bool(controls_passed and parent_source_capture and physics and oracle
                  and drift is not None and math.isfinite(drift) and drift <= .005)
    clean_ok, capture_ok = bool(common and clean_both), bool(common and capture_both)
    return dict(status='pending' if not present else 'qualified' if clean_ok else 'capture_only' if capture_ok else 'failed',
                qualified=clean_ok, capture_qualified=capture_ok, clean_qualified=clean_ok,
                clean_both=bool(clean_both), capture_both=bool(capture_both), physics_valid=bool(physics),
                oracle_agreement=bool(oracle), max_drift_m=drift, parent_source_capture=bool(parent_source_capture))
