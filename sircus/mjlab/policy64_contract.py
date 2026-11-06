"""Fixed-64 TRAIN-only success bootstrap and fresh terminal-shot PPO protocol."""
import copy
import json
import math
from pathlib import Path
import numpy as np

from .corrected_contract import sha
from .learning_contract import digest
from .calibrated_benchmark_contract import reward, statistics

VERSION = 'sircus-policy64-bc-ppo-v1'
AUDIT_VERSION = 'sircus-policy64-independent-audit-v1'
SEED = 20261511
MILESTONES = (0, 256, 512, 1024)
METHODS = ('prior', 'nearest', 'bc', 'ppo')
SCOPE = ('Fixed 64-substep privileged-geometry, held-endpoint shot selection. '
         'TRAIN-only historical success imitation followed by fresh terminal-shot PPO; '
         'one frozen decision per new test bucket, no flight search or retries at inference. '
         'Geometry-only source/IK/setup search is separate. No noise, 128-step qualification, '
         'grasping, driven transit, perception, physical transfer or all-table guarantee. '
         'Historical 128 failures remain unchanged.')

def read(path):
    return json.loads(Path(path).read_text())

def captured(r):
    return bool(r.get('native_release') and r.get('route_success') and not r.get('invalid') and not r.get('invalid_physics'))

def clean(r):
    return captured(r) and r.get('route_quality') == 2

def protocol():
    return dict(substeps=64, seed=SEED, train_layouts=35, validation_layouts=12, test_layouts=50,
        train_requests=1024, batch_size=32, updates=32, milestones=list(MILESTONES),
        test_methods=list(METHODS), test_repeats=1, validation_requests=48, test_requests=200,
        control_requests=12, maximum_fresh_requests=1284, noise=False,
        actor_steps=800, critic_steps=400, bootstrap_lr=.001, ppo_lr=.0003,
        initial_logstd=-3., ppo_epochs=6, clip=.2, entropy=.001, gradient_norm=1.,
        selection='Validation captures, then mean calibrated reward, then earliest milestone; freeze before TEST',
        source_selection='One captured TRAIN teacher per context: clean, calibrated reward, stable ID',
        nearest='Nearest successful TRAIN bucket, then context ID and teacher ID; transfer its normalized residual without flight search',
        scope=SCOPE)

def make_splits(parent_splits, historical):
    train = [dict(id=c['id'], split='train', bucket_xy_m=copy.deepcopy(c['bucket_xy_m'])) for c in parent_splits['train']]
    excluded = sorted({tuple(map(float, c['bucket_xy_m'])) for groups in (parent_splits, historical)
                       for role in ('train','validation','test') for c in groups[role]})
    used = np.asarray(excluded, float)
    result = dict(train=train)
    for role, count, seed in (('validation',12,20261503),('test',50,20261504)):
        rng, rows = np.random.default_rng(seed), []
        for _ in range(100000):
            if len(rows) == count: break
            xy = rng.uniform([.1,-.65],[1.24,.25])
            if np.linalg.norm(used-xy,axis=1).min() < .005: continue
            rows.append(dict(id=f'policy64-{role}-{len(rows):02d}',split=role,bucket_xy_m=xy.tolist()))
            used = np.vstack([used,xy])
        if len(rows) != count: raise ValueError('Could not freeze disjoint bucket positions')
        result[role] = rows
    return result, [list(x) for x in excluded]

def schedule():
    rng, result = np.random.default_rng(SEED), []
    while len(result) < 1024: result.extend(rng.permutation(35).tolist())
    return result[:1024]

def select_teachers(rows):
    groups = {}
    for r in rows:
        if r['role'] != 'train': raise ValueError('Held-out labels cannot enter bootstrap')
        if not captured(r): continue
        groups.setdefault(r['context_id'], []).append(r)
    return [min(groups[k], key=lambda r:(-int(clean(r)),-reward(r),r['id'])) for k in sorted(groups)]

def load_parent(parent, audit, repository):
    """Bind immutable audited labels; do not consume historical held-out labels."""
    parent, repository, audit = Path(parent), Path(repository), Path(audit)
    public = read(repository/'baselines/mjlab-calibrated-benchmark-20261104/provenance.json')
    for path,key in ((parent/'report.json','report_sha256'),(parent/'manifest.json','source_manifest_sha256'),(audit,'audit_sha256')):
        if sha(path) != public[key]: raise ValueError('Historical calibrated evidence bytes differ: '+path.name)
    proof, spec, manifest = read(audit), read(parent/'contract.json'), read(parent/'manifest.json')
    if proof.get('status') != 'passed' or proof.get('failures') or proof.get('checks') != 1204473:
        raise ValueError('Exact passed historical calibrated audit required')
    unsigned = dict(spec); fp = unsigned.pop('fingerprint')
    if digest(unsigned) != fp or fp != public['contract_fingerprint']:
        raise ValueError('Historical source contract differs')
    if sha(parent/'contract.json') != manifest['contract.json']: raise ValueError('Parent contract manifest binding differs')
    prior=read(parent/'prior.json')
    if sha(parent/'prior.json')!=manifest['prior.json'] or digest(prior)!=spec['prior_sha256']:
        raise ValueError('Frozen historical geometric prior differs')
    for name in ('calibration.json','verification.json','setup-home.json'):
        if sha(parent/name)!=manifest[name]: raise ValueError('Historical configuration bytes differ: '+name)
    train = {c['id']:c['bucket_xy_m'] for c in spec['splits']['train']}
    rows, bindings = [], {}
    for method in ('ppo','sweep'):
        for seed in (20261411,20261412,20261413):
            name = f'runs/{method}-seed{seed}/rows.jsonl'; path = parent/name
            if sha(path) != manifest[name]: raise ValueError('Historical row bytes differ')
            bindings[name] = sha(path)
            selected = []
            for line in path.open():
                raw = json.loads(line)
                if raw.get('role') != 'train': continue
                if (raw.get('method') != method or raw.get('seed') != seed or raw.get('context_id') not in train
                    or raw['bucket_xy_m'] != train[raw['context_id']] or raw.get('perturbation_mixed_units') != [0.]*5):
                    raise ValueError('Historical TRAIN identity/coordinate differs')
                if raw.get('native_release') and (raw.get('contact_protocol',{}).get('substeps') != 64
                    or raw.get('scoring_version') != 'sircus-calibrated-native-capture-v2'
                    or raw.get('scalar_score_audit_passed') is not True):
                    raise ValueError('Native source is not calibrated64 with oracle agreement')
                selected.append(raw)
            if len(selected)!=1024 or sorted(r['train_index'] for r in selected)!=list(range(1024)):
                raise ValueError('Complete 1024-request historical TRAIN prefix required')
            rows.extend(sorted(selected,key=lambda r:r['train_index']))
    if len(rows)!=6144 or len({r['id'] for r in rows})!=6144: raise ValueError('Complete unique historical TRAIN dataset required')
    teachers = select_teachers(rows)
    dataset = [dict(id=r['id'],role='train',context_id=r['context_id'],bucket_xy_m=r['bucket_xy_m'],
        observation=r['observation'],normalized_residual=r['normalized_residual'],values=r['values'],
        effective_source_hint=r['effective_source_hint'],native_release=bool(r['native_release']),
        route_success=captured(r),route_quality=2 if clean(r) else 1 if captured(r) else 0,
        invalid=r.get('invalid'),invalid_physics=bool(r.get('invalid_physics')),learning_reward=reward(r),
        source_record_sha256=digest(r)) for r in rows]
    byid = {r['id']:r for r in dataset}
    control_positive = teachers[0]
    negatives = [r for r in rows if r.get('native_release') and not captured(r) and not r.get('invalid')
                 and not r.get('invalid_physics') and not r.get('whole_ball_entry')]
    control_negative = min(negatives,key=lambda r:r['id'])
    summary = dict(requests=len(rows),native_releases=sum(bool(r['native_release']) for r in rows),
        captures=sum(captured(r) for r in rows),clean=sum(clean(r) for r in rows),layouts=len(train),
        successful_layouts=len(teachers),teacher_count=len(teachers),
        labels_scope='Historical TRAIN only; positive BC subset, all TRAIN outcomes for critic; acquisition cost is not free')
    source = dict(parent_contract_fingerprint=fp,report_sha256=public['report_sha256'],
        parent_audit_sha256=public['audit_sha256'],parent_manifest_sha256=public['source_manifest_sha256'],
        train_files_sha256=bindings,teacher_ids=[r['id'] for r in teachers])
    return spec,prior,dataset,[byid[r['id']] for r in teachers],summary,source,[control_positive,control_negative]

def nearest_teacher(context,teachers):
    return min(teachers,key=lambda r:(math.dist(context['bucket_xy_m'],r['bucket_xy_m']),r['context_id'],r['id']))

def load_runtime_reference(reference,assets,repository):
    """New explicit bridge from historical geometry to the audited current runtime.

    The old sweep_worker.load_reference source guard stays unchanged. Its older
    native code differs after later reset/clock fixes, so it cannot certify this
    run. Bind historical scene/assets separately, and require every pre-existing
    native module to match the newer, independently audited impact-study source.
    Historical labels remain bootstrap information, not new rollout results.
    """
    reference,assets,repository=Path(reference),Path(assets),Path(repository)
    historical=read(repository/'baselines/mjlab-corrected-grid-20261102/native-results.json')
    for name in ('report.json','contract.json','confirmed-catalogue.json'):
        if sha(reference/name)!=historical['artifacts_sha256'][name]:raise ValueError('Historical geometry reference bytes differ')
    spec=read(reference/'contract.json');unsigned=dict(spec);fp=unsigned.pop('fingerprint')
    if digest(unsigned)!=fp or fp!=historical['contract_fingerprint']:raise ValueError('Historical geometry reference fingerprint differs')
    latest=read(repository/'baselines/mjlab-impact650-20261105/summary.json')['provenance']
    for name,expected in latest['executed_source_sha256'].items():
        if sha(repository/name)!=expected:raise ValueError('Current native source differs from audited impact runtime: '+name)
    if spec['asset_sha256']!=latest['asset_sha256']:raise ValueError('Historical/current native assets differ')
    for name,expected in spec['asset_sha256'].items():
        if sha(assets.parent/name)!=expected:raise ValueError('Current asset bytes differ: '+name)
    import importlib.metadata
    if importlib.metadata.version('scipy')!='1.15.3':raise ValueError('Pinned SciPy1.15.3 required')
    return dict(root=reference,scene=spec['scene'],contract=spec,report=read(reference/'report.json'),
        public=historical,catalogue=read(reference/'confirmed-catalogue.json'),asset_hashes=spec['asset_sha256'],runtime_reference=dict(
        version='sircus-policy64-current-native-reference-v1',historical_geometry_fingerprint=fp,
        current_native_proof_sha256=latest['independent_audit_sha256'],
        current_native_contract_fingerprint=latest['contract_fingerprint'],
        source_sha256=latest['executed_source_sha256'],asset_sha256=latest['asset_sha256'],
        scope='Historical geometry/assets plus current audited64/128 runtime source; bootstrap labels are historical, all new outcomes freshly measured64'))
