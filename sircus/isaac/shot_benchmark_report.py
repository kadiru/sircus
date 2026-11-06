"""Publish a read-only sweep/PPO comparison from native aggregate receipts.

No physics, policy inference, optimizer or network connection runs here. Missing
runs stay pending with null observations; counts never become synthetic results.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil
import statistics

SCHEMA = 'sircus-shot-benchmark-report-v1'
EXPERIMENT_SCHEMA = 'sircus-shot-benchmark-experiment-v1'
PROGRESS_SCHEMA = 'sircus-shot-benchmark-progress-v1'
METHODS = ('sweep', 'ppo')
MILESTONES = (0, 1000, 2500, 5000, 10000)
SOURCE_FIELDS = ('commit', 'archive_sha256', 'geometry_fingerprint', 'camera_sha256', 'scoring_version')


def require(value, message):
    if not value:
        raise ValueError(message)


_SHA_CACHE = {}


def sha(path):
    """Reuse only a fully hashed, stat-identical file in this publisher process.

    Both ctime and mtime, inode, device and size are checked. Mutable/atomic
    changes invalidate the cache; a file changing during hashing is rejected.
    There is no persistent cache to trust across processes or machines.
    """
    path = Path(path)
    def signature():
        s = path.stat()
        return (s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns)
    key = str(path.absolute())
    before = signature()
    cached = _SHA_CACHE.get(key)
    if cached is not None and cached[0] == before:
        return cached[1]
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            value.update(block)
    require(signature() == before, 'Evidence changed during hashing; retry a stable snapshot')
    result = value.hexdigest()
    _SHA_CACHE[key] = (before,result)
    return result


def read(path):
    def reject(value):
        raise ValueError('Non-finite JSON: ' + value)
    return json.loads(Path(path).read_text(), parse_constant=reject)


def read_snapshot(path):
    """Bind parsed mutable progress to the exact bytes, even across atomic updates."""
    data = Path(path).read_bytes()
    def reject(value):
        raise ValueError('Non-finite JSON: '+value)
    return json.loads(data,parse_constant=reject),hashlib.sha256(data).hexdigest()


def safe_path(root, name, *, exists=True):
    require(isinstance(name, str) and name and not Path(name).is_absolute()
            and '..' not in Path(name).parts, 'Unsafe evidence path')
    root = Path(root).resolve()
    path = root / name
    require(path.resolve().is_relative_to(root) and not path.is_symlink(), 'Evidence escapes root')
    if exists:
        require(path.is_file(), 'Missing evidence: ' + name)
    return path


def digest(value, label):
    require(isinstance(value, str) and len(value) == 64
            and all(c in '0123456789abcdef' for c in value), label + ' must be SHA256')
    return value


def integer(value, label):
    require(type(value) is int and value >= 0, label + ' must be a nonnegative integer')
    return value


def finite(value, label):
    require(type(value) in (int, float) and math.isfinite(value), label + ' must be finite')
    return value


def counts(raw):
    require(isinstance(raw, dict), 'Counts must be an object')
    result = {k: integer(raw[k], k) for k in ('attempts', 'captures', 'clean', 'rejected')}
    n, c = result['attempts'], result['captures']
    require(result['clean'] <= c <= n and result['rejected'] <= n-c,
            'Capture/rejection counts violate the attempt denominator')
    failures = raw.get('failures')
    if failures is not None:
        require(isinstance(failures, dict), 'Failure counts must be an object')
        failures = {str(k): integer(v, 'Failure count') for k, v in failures.items()}
        require(sum(failures.values()) <= n, 'Failure counts exceed attempted shots')
    result.update(released=n-result['rejected'], capture_rate=c/n if n else None,
                  clean_rate=result['clean']/n if n else None, failures=failures)
    if raw.get('mean_reward') is not None:
        result['mean_reward'] = finite(raw['mean_reward'], 'Mean reward')
    else:
        result['mean_reward'] = None
    return result


def source_identity(raw):
    require(isinstance(raw, dict), 'Source identity required')
    value = {k: raw[k] for k in SOURCE_FIELDS}
    require(isinstance(value['commit'], str) and len(value['commit']) == 40
            and all(c in '0123456789abcdef' for c in value['commit']), 'Full native source commit required')
    for k in ('archive_sha256', 'geometry_fingerprint', 'camera_sha256'):
        digest(value[k], k)
    require(isinstance(value['scoring_version'], str) and value['scoring_version'], 'Scoring identity required')
    for name in ('source_sha256','module_sha256','source_members_sha256','native_source_sha256'):
        if name in raw:
            require(isinstance(raw[name],dict), 'Source module receipt must be a mapping')
            value[name] = {str(k):digest(v,'Source module digest') for k,v in raw[name].items()}
    return value


def resources(raw):
    """Unknown measurements remain null; headless and rendering never combine."""
    raw = raw or {}
    result = {}
    for phase in ('headless', 'render'):
        item = raw.get(phase)
        if item is None:
            result[phase] = None
            continue
        require(isinstance(item, dict), 'Resource phase must be an object')
        v = {k: None if item.get(k) is None else finite(item[k], k)
             for k in ('wall_s', 'cpu_s', 'peak_rss_mib', 'peak_gpu_mib')}
        require(all(x is None or x >= 0 for x in v.values()), 'Negative resource measurement')
        v['scope'] = str(item.get('scope', 'Scope not supplied; do not assume aggregate worker memory'))
        result[phase] = v
    return result


def predictions(raw, expected_count):
    if raw is None:
        return None
    require(isinstance(raw, list) and len(raw) == expected_count, 'Incomplete frozen layout predictions')
    ids, xy, result = set(), set(), []
    for row in raw:
        name = row.get('layout_id', row.get('id'))
        require(isinstance(name, str) and name not in ids, 'Missing or duplicate layout ID')
        p = row.get('bucket_xy_m')
        require(isinstance(p, list) and len(p) == 2, 'Layout bucket XY required')
        p = tuple(finite(x, 'Bucket coordinate') for x in p)
        require(p not in xy, 'Duplicate frozen bucket coordinate')
        ids.add(name); xy.add(p)
        outcome = row.get('outcome')
        require(isinstance(outcome, dict) and all(type(outcome.get(k)) is bool
                for k in ('captured', 'clean', 'rejected')), 'Explicit frozen outcome required')
        require(not outcome['clean'] or outcome['captured'], 'Clean outcome without capture')
        require(not outcome['rejected'] or not outcome['captured'], 'Rejected shot cannot capture')
        result.append(dict(layout_id=name, bucket_xy_m=list(p),
            requested=row.get('requested'), measured=row.get('measured'),
            outcome={k: outcome.get(k) for k in ('captured', 'clean', 'rejected', 'reason')},
            provenance=str(row.get('provenance', 'Not supplied'))))
    return result


def assessment(raw, expected_count):
    if raw is None:
        return None
    value = counts(raw)
    require(value['attempts'] == expected_count, 'Frozen assessment denominator differs from declared layouts')
    layouts = predictions(raw.get('layouts'), expected_count)
    require(layouts is not None, 'Frozen assessment requires complete per-layout outcomes')
    if layouts is not None:
        for k, key in (('captures', 'captured'), ('clean', 'clean'), ('rejected', 'rejected')):
            require(sum(r['outcome'][key] for r in layouts) == value[k], 'Layout outcomes differ from aggregate counts')
    value.update(layouts=layouts, scope='Frozen direct inference; no held-out physics search or optimizer updates')
    return value


def timing_batches(raw):
    if not raw:
        return None
    require(isinstance(raw, list), 'Batch timings must be an array')
    total = {'batches': len(raw), 'filmed_batches': 0, 'setup_wall_s': 0., 'score_wall_s': 0.}
    known = {'setup_wall_s': True, 'score_wall_s': True}
    for row in raw:
        require(type(row.get('filming')) is bool, 'Batch filming flag required')
        total['filmed_batches'] += row['filming']
        for key in known:
            if row.get(key) is None:
                known[key] = False
            else:
                total[key] += finite(row[key], key)
    for key in known:
        if not known[key]: total[key] = None
    total['scope'] = 'Observed batch intervals, including filmed batches. Not a substitute for isolated headless resource accounting.'
    return total


def load_run(root, plan, experiment):
    path = safe_path(root, plan['path']+'/progress.json', exists=False)
    common = dict(method=plan['method'], seed=plan['seed'], path=plan['path'])
    if not path.exists():
        return dict(common, status='pending', train=None, learning_curve=[], checkpoints=[],
                    resources=resources(None), training_wall_s=None, algorithm_overhead=None, batch_timing=None, videos=[], evidence={}, error=None)
    raw, progress_sha = read_snapshot(path)
    require(raw.get('schema') == PROGRESS_SCHEMA and raw.get('experiment_id') == experiment['experiment_id'], 'Progress belongs to another experiment')
    require(raw.get('method') == plan['method'] and raw.get('seed') == plan['seed'], 'Run method/seed changed')
    require(source_identity(raw['source']) == experiment['source'], 'Run source/camera/scoring differs')
    require(raw.get('split_manifest_sha256') == experiment['split_manifest_sha256'], 'Run uses different layout split')
    status = raw.get('status')
    require(status in ('pending', 'queued', 'running', 'training', 'evaluating', 'in_progress', 'completed', 'failed', 'cancelled'), 'Unknown native run status')
    train = counts(raw['train']) if raw.get('train') is not None else None
    if train:
        require(train['attempts'] <= 10000, 'Run exceeds authorized TRAIN budget')
    curve, previous = [], None
    for row in raw.get('learning_curve', []):
        point = counts(row)
        if previous:
            require(point['attempts'] > previous['attempts'] and all(point[k] >= previous[k] for k in ('captures', 'clean', 'rejected')), 'Cumulative learning curve regressed or duplicated')
        require(train is not None and point['attempts'] <= train['attempts'], 'Curve exceeds observed TRAIN attempts')
        point['wall_s'] = None if row.get('wall_s') is None else finite(row['wall_s'], 'Curve wall seconds')
        curve.append(point); previous = point
    if train and curve and curve[-1]['attempts'] == train['attempts']:
        require(all(curve[-1][k] == train[k] for k in ('captures', 'clean', 'rejected')), 'Latest curve differs from TRAIN counts')
    evidence = {'progress.json': progress_sha}
    runroot = path.parent
    for name, expected in raw.get('source_trial_sha256', {}).items():
        file = safe_path(runroot, name)
        require(sha(file) == digest(expected, 'Trial digest'), 'Aggregate native trial bytes changed')
        evidence[name] = expected
    checkpoints, seen = [], set()
    for row in raw.get('checkpoints', []):
        budget = integer(row['training_attempts'], 'Checkpoint TRAIN attempts')
        require(budget in MILESTONES and budget not in seen and train is not None and budget <= train['attempts'], 'Invalid or duplicate checkpoint budget')
        seen.add(budget)
        cp, proof = digest(row['checkpoint_sha256'], 'Checkpoint digest'), digest(row['proof_sha256'], 'Checkpoint proof digest')
        if status == 'completed':
            require(row.get('checkpoint_file') and row.get('proof_file'), 'Completed checkpoint requires actual model and proof file bindings')
        for key, expected in (('checkpoint_file', cp), ('proof_file', proof)):
            if row.get(key):
                require(sha(safe_path(runroot, row[key])) == expected, 'Checkpoint/proof bytes changed')
                evidence[row[key]] = expected
        require(row.get('test') is None or budget == 10000, 'Test results may only be opened at final 10k checkpoint')
        checks = dict(validation=assessment(row.get('validation'), 35), test=assessment(row.get('test'), 70))
        for split, check in checks.items():
            if check and check['layouts'] is not None:
                observed = {p['layout_id']:tuple(p['bucket_xy_m']) for p in check['layouts']}
                require(observed == experiment['_split_positions'][split], 'Frozen predictions changed declared split positions')
        checkpoints.append(dict(training_attempts=budget, checkpoint_sha256=cp, proof_sha256=proof, **checks))
    checkpoints.sort(key=lambda p:p['training_attempts'])
    if status == 'completed':
        require(train is not None and train['attempts'] == 10000 and seen == set(MILESTONES), 'Completed run missing TRAIN budget/checkpoints')
        require(all(p['validation'] is not None for p in checkpoints) and checkpoints[-1]['test'] is not None, 'Completed run missing frozen validation or final test')
    videos = []
    video_rows = raw.get('videos', [])
    if (runroot/'video-index.json').is_file():
        video_index, index_sha = read_snapshot(safe_path(runroot,'video-index.json'))
        indexed = video_index['videos']
        require(indexed[:len(video_rows)] == video_rows, 'Video index replaced or dropped a progress chapter')
        video_rows = indexed
        evidence['video-index.json'] = index_sha
    for i, row in enumerate(video_rows):
        require(row.get('selection') == 'prescheduled' and row.get('camera_sha256') == experiment['source']['camera_sha256'], 'Video is not prescheduled at the fixed selected camera')
        require(row.get('method', plan['method']) == plan['method'] and row.get('seed', plan['seed']) == plan['seed'], 'Video method/seed differs')
        if row.get('experiment_sha256') is not None:
            require(row['experiment_id'] == experiment['experiment_id'] and row['experiment_sha256'] == experiment['_experiment_sha256'] and row['source_commit'] == experiment['source']['commit'], 'Video belongs to another experiment/source')
        if row.get('decode_verified') is not None:
            require(row['decode_verified'] is True, 'Video has no successful decode receipt')
        file = safe_path(runroot, row['file'])
        require(file.suffix.lower() in ('.mp4', '.webm') and sha(file) == digest(row['sha256'], 'Video digest'), 'Missing or changed encoded native video')
        videos.append(dict(file=row['file'], sha256=row['sha256'], selection='prescheduled', camera_sha256=row['camera_sha256'],
            training_attempts=row.get('training_attempts'), frames=row.get('frames'), duration_s=row.get('duration_s'), fps=row.get('fps'), decode_verified=row.get('decode_verified'), label=str(row.get('label', 'Native prescheduled trials')),
            provenance=str(row.get('provenance', 'Actual simulation; scripted transit and initialized release remain separate from learned shot parameters'))))
    training_wall = None if raw.get('training_wall_s') is None else finite(raw['training_wall_s'],'Full TRAIN wall seconds')
    require(training_wall is None or training_wall >= 0, 'Negative TRAIN wall time')
    overhead = raw.get('algorithm_overhead')
    if overhead is not None:
        overhead = {k: None if overhead.get(k) is None else finite(overhead[k],k) for k in ('proposal_wall_s','update_wall_s','checkpoint_wall_s')} | {'scope':str(raw['algorithm_overhead'].get('scope','Checkpoint overhead may include all stages, not only TRAIN'))}
    return dict(common, status=status, train=train, learning_curve=curve, checkpoints=checkpoints,
        training_wall_s=training_wall, algorithm_overhead=overhead,
        resources=resources(raw.get('resources')), batch_timing=timing_batches(raw.get('batches')),
        videos=videos, evidence=evidence, error=str(raw['error']) if raw.get('error') else None)


def method_summaries(runs):
    """Descriptive seed means/SD, never a confidence bound over repeated sites."""
    result = []
    for method in METHODS:
        group = [r for r in runs if r['method'] == method]
        for budget in MILESTONES:
            for split in ('validation', 'test') if budget == 10000 else ('validation',):
                points = [c[split]['capture_rate'] for r in group for c in r['checkpoints']
                          if c['training_attempts'] == budget and c[split] is not None]
                result.append(dict(method=method, training_attempts=budget, split=split,
                    measured_seeds=len(points), expected_seeds=3,
                    mean_capture_rate=statistics.mean(points) if points else None,
                    seed_sd=statistics.stdev(points) if len(points)>1 else None,
                    minimum=min(points) if points else None, maximum=max(points) if points else None,
                    scope='Descriptive mean/sample SD across available training seeds, not a confidence interval or independent new-layout count.'))
    return result


def summarize(root):
    root = Path(root)
    experiment, experiment_sha = read_snapshot(root/'experiment.json')
    require(experiment.get('schema') == EXPERIMENT_SCHEMA, 'Unknown benchmark experiment')
    require(isinstance(experiment.get('experiment_id'), str) and experiment['experiment_id'], 'Experiment ID required')
    require(experiment.get('methods') == list(METHODS), 'Expected matched sweep/PPO experiment')
    seeds = experiment['seeds']
    require(isinstance(seeds, list) and len(seeds) == len(set(seeds)) == 3 and all(type(v) is int for v in seeds), 'Three distinct training seeds required')
    require(experiment['train_budget_per_run'] == 10000 and experiment['milestones'] == list(MILESTONES), 'Authorized budget/checkpoints changed')
    require(experiment['validation_layout_count'] == 35 and experiment['test_layout_count'] == 70, 'Frozen layout counts changed')
    experiment = dict(experiment, source=source_identity(experiment['source']), _experiment_sha256=experiment_sha)
    digest(experiment['split_manifest_sha256'], 'Split manifest digest')
    evidence = {'experiment.json': experiment_sha}
    require(experiment.get('split_manifest_file'), 'Actual split manifest file required')
    f = safe_path(root, experiment['split_manifest_file'])
    require(sha(f) == experiment['split_manifest_sha256'], 'Split manifest changed')
    split = read(f)
    if split.get('schema') == 'sircus-fresh-shot-layouts-v1':
        from shot_benchmark_contract import validate_splits
        validate_splits(split)
    positions, identifiers, by_split = set(), set(), {}
    for label, count in (('train',35), ('validation',35), ('test',70)):
        rows = split[label]; require(len(rows) == count, 'Split layout count changed')
        by_split[label] = {}
        for row in rows:
            name = row.get('id', row.get('layout_id'))
            require(isinstance(name,str) and name not in identifiers, 'Duplicate split layout ID')
            require(isinstance(row.get('bucket_xy_m'),list) and len(row['bucket_xy_m']) == 2, 'Split bucket XY required')
            xy = tuple(finite(v,'Split coordinate') for v in row['bucket_xy_m'])
            require(xy not in positions, 'TRAIN/VAL/TEST coordinate leakage')
            require(row.get('split',label) == label, 'Layout split changed')
            positions.add(xy); identifiers.add(name); by_split[label][name] = xy
    experiment['_split_positions'] = by_split
    evidence[experiment['split_manifest_file']] = sha(f)
    plans = experiment.get('runs', [])
    expected = {(m,s) for m in METHODS for s in seeds}
    require(len(plans) == 6 and {(p['method'],p['seed']) for p in plans} == expected, 'Six explicit method/seed run paths required')
    require(len({p['path'] for p in plans}) == 6, 'Run paths must be distinct')
    runs = [load_run(root,p,experiment) for p in plans]
    return dict(schema=SCHEMA, experiment_id=experiment['experiment_id'], status='completed' if all(r['status']=='completed' for r in runs) else 'attention_required' if any(r['status'] in ('failed','cancelled') for r in runs) else 'in_progress',
        source=experiment['source'], protocol=experiment.get('protocol'), reward=experiment.get('reward'), split_manifest_sha256=experiment['split_manifest_sha256'],
        seeds=seeds, methods=list(METHODS), train_budget_per_run=10000, authorized_train_attempts=60000,
        observed_train_attempts=sum(r['train']['attempts'] for r in runs if r['train'] is not None),
        milestones=list(MILESTONES), train_layout_count=35, validation_layout_count=35, test_layout_count=70,
        runs=runs, method_summary=method_summaries(runs), source_sha256=evidence, preprocessing=experiment.get('preprocessing'),
        scope='Three seeds per method,10k TRAIN requests each. Rejected proposals count as attempts. Frozen VAL at declared checkpoints; unseen70-position TEST only at final checkpoint. No per-held-out physics search.',
        resource_scope='TRAIN physics budget excludes frozen evaluation and geometry preparation. Full TRAIN method wall includes learner overhead and excludes evaluation. Headless/render episode intervals exclude outside learner overhead and are not total method compute; phases remain separate. Unknown values remain null. Process peak memory is not aggregate worker memory unless its scope says so.',
        scientific_status='Native receipt consistency checked by this publisher; independent scoring/checkpoint audits remain separate evidence. No model promotion implied.',
        geometry_uncertainty='Bucket reported height175mm remains unresolved overall height versus interior depth; contact properties and rigid board attachment remain provisional.')


def publish(root, output, *, media=False):
    root, output = Path(root), Path(output)
    require(not output.resolve().is_relative_to(root.resolve()) and not root.resolve().is_relative_to(output.resolve()), 'Publication must be outside native evidence and its ancestors')
    value = summarize(root)
    output.mkdir(parents=True, exist_ok=True)
    if (output/'report.json').exists():
        old = read(output/'report.json')
        require(old.get('schema') == SCHEMA and old.get('experiment_id') == value['experiment_id'] and old.get('source') == value['source'], 'Refusing to replace another study publication')
    for run in value['runs']:
        for i, video in enumerate(run['videos']):
            target = f"media/{run['method']}-seed{run['seed']}-{i}{Path(video['file']).suffix.lower()}"
            destination = safe_path(output, target, exists=False)
            if media:
                destination.parent.mkdir(parents=True, exist_ok=True)
                if destination.exists(): require(sha(destination) == video['sha256'], 'Published media collision')
                else: shutil.copyfile(safe_path(root,run['path']+'/'+video['file']), destination)
            available = destination.is_file() and sha(destination) == video['sha256']
            video['file'] = target if available else None
            video['available'] = available
    temporary = output/'report.json.next'
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    temporary.replace(output/'report.json')
    return value


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--media', action='store_true', help='Copy only bound encoded native videos, never JPEGs/raw data')
    args = parser.parse_args()
    report = publish(args.root,args.output,media=args.media)
    print(json.dumps({k:report[k] for k in ('status','experiment_id','observed_train_attempts','authorized_train_attempts')}))
