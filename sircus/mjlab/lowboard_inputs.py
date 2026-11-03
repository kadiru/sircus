"""Read-only, fail-closed ancestry checks for the lower-board development study."""
import copy
import json
from pathlib import Path

from . import lowboard_contract as protocol
from .corrected_contract import sha
from .learning_contract import digest

PARENT_FINGERPRINT = '6d94edc505db7201f603456116303cd14acd9dde26e364ea295b1739c78f8499'
# This sole non-physics publisher changed after the immutable parent execution.
# Every other parent source hash, especially geometry/IK/scoring/physics, is exact.
PUBLICATION_OVERRIDE = {'sircus/mjlab/prepare_distance_report.py': (
    '7fb5088df5103de9155af365ecc85021c45d9e1aacdffa725fbbdad64ac29578',
    '9058f1873ed6aeabc348da0af2c6c290d403da27bb9d546fd4aded729cfa27a9')}


def read(path):
    return json.loads(Path(path).read_text())


def source_compatibility(current, previous):
    differences = {}
    for name, old in previous.items():
        new = current.get(name)
        if new == old:
            continue
        if PUBLICATION_OVERRIDE.get(name) != (old, new):
            raise ValueError('Parent source changed outside the exact publication-only exception: '+name)
        differences[name] = dict(parent_sha256=old, current_sha256=new,
            purpose='Read-only display adapter published after parent execution; not imported by native physics')
    return differences


def prepare_inputs(parent):
    parent = Path(parent).resolve()
    spec, report, audit = (read(parent/name) for name in ('contract.json','report.json','audit-distance.json'))
    payload = dict(spec); fp = payload.pop('fingerprint',None)
    checks = audit.get('checks',[])
    if (digest(payload) != fp or fp != PARENT_FINGERPRINT or fp != report.get('contract_fingerprint')
            or fp != audit.get('contract_fingerprint') or spec.get('version') != 'sircus-bucket-distance-development-v1'
            or report.get('status') != 'completed' or report.get('phase') != 'completed'
            or audit.get('status') != 'passed' or audit.get('failures') or not checks
            or type(audit.get('checks_count')) is not int or audit['checks_count'] != len(checks)
            or any(c.get('passed') is not True for c in checks)):
        raise ValueError('Exact completed independently audited distance parent required')
    manifest = read(parent/'evidence-manifest.json')
    if manifest != report['evidence_sha256']:
        raise ValueError('Parent manifest differs from completed report')
    for name, expected in manifest.items():
        path = (parent/name).resolve()
        if not path.is_relative_to(parent) or not path.is_file() or sha(path) != expected:
            raise ValueError('Parent evidence changed: '+name)
    seeds = copy.deepcopy(spec['seeds'])
    if tuple(s['id'] for s in seeds) != protocol.SEED_IDS:
        raise ValueError('Eight exact original shared-bucket seeds required')
    rows = [json.loads(line) for line in (parent/'records.jsonl').read_text().splitlines() if line.strip()]
    provenance = []
    for seed in seeds:
        controls = [r for r in rows if r['role']=='control' and r['source_anchor_id']==seed['id']]
        if len(controls)!=3 or not all(r['native_release'] and r['route_success'] and not r.get('invalid') and
                all(r[k]==seed[k] for k in ('values','bucket_xy_m','q10','q14')) for r in controls):
            raise ValueError('Parent exact control identity/outcome changed: '+seed['id'])
        provenance.append(dict(id=seed['id'], parent_control_record_ids=[r['id'] for r in controls],
            geometry_sha256=digest(seed)))
    if len([r for r in rows if r['role']=='control']) != 24:
        raise ValueError('Exactly 24 source controls required')
    _, targets = protocol.targets(spec['scene'],seeds)
    frozen = {t['id']:t for t in spec['settings']['targets']}
    if any(frozen[t['id']] != t for t in targets):
        raise ValueError('Far target geometry differs from parent')
    names=('contract.json','report.json','audit-distance.json','evidence-manifest.json','records.jsonl')
    metadata=dict(contract_fingerprint=fp, files_sha256={n:sha(parent/n) for n in names})
    return copy.deepcopy(spec['scene']),seeds,provenance,metadata,spec
