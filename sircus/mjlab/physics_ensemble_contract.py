"""Frozen offline forward-model study; no simulator or hardware side effects."""
import copy
import hashlib
import json
from pathlib import Path
import numpy as np

VERSION='sircus-physics-ensemble-v1'
AUDIT_VERSION='sircus-physics-ensemble-audit-v1'
MODEL_VERSION='sircus-physics-ensemble-model-v1'
METHODS=('bc','ppo','ensemble')
SEEDS=(20261611,20261612,20261613,20261614,20261615)
MAX_REQUESTS=264
SCOPE=('Offline BC/PPO-seeded state-based physics ensemble with model-only finite candidate ranking and '
       'at most four geometry-only admission checks versus unchanged saved BC/PPO. '
       'Fresh nominal held-out layouts;64 substeps only. Extra candidate/geometry inference '
       'and unequal historical exposure are disclosed, not an equal-budget architecture ablation. '
       'No test flight search, noise,128 convergence, perception,WAM,grasp,driven transit or physical claim.')

def read(path):return json.loads(Path(path).read_text())
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def digest(v):return hashlib.sha256(json.dumps(v,sort_keys=True,allow_nan=False).encode()).hexdigest()
def contained(root,name):
    p=Path(root).resolve();r=Path(name)
    if r.is_absolute() or '..' in r.parts or not (p/r).resolve().is_relative_to(p) or not (p/r).is_file():
        raise ValueError('Bound relative artifact required')
    return p/r

def authenticate(parent_study,parent_audit,native_source,published):
    p,a,n=Path(parent_study),Path(parent_audit),Path(native_source)
    public=read(published)['provenance']
    for f,k in ((p/'report.json','report_sha256'),(p/'contract.json','contract_sha256'),
                (p/'evidence-manifest.json','evidence_manifest_sha256'),(a,'independent_audit_sha256')):
        if sha(f)!=public[k]:raise ValueError('Published audited parent binding differs: '+f.name)
    spec,report,proof,manifest=map(read,(p/'contract.json',p/'report.json',a,p/'evidence-manifest.json'))
    unsigned=dict(spec);fp=unsigned.pop('fingerprint')
    checks=proof.get('checks')
    if (digest(unsigned)!=fp or fp!=public['contract_fingerprint'] or report['status']!='completed'
        or report['contract_fingerprint']!=fp or report['evidence_sha256']!=manifest
        or proof.get('version')!='sircus-policy64-independent-audit-v1' or proof.get('status')!='passed'
        or proof.get('failures')!=[] or not isinstance(checks,list) or len(checks)!=346485
        or proof.get('checks_count')!=len(checks) or any(c.get('passed') is not True for c in checks)
        or proof['report_sha256']!=sha(p/'report.json') or proof['evidence_manifest_sha256']!=sha(p/'evidence-manifest.json')):
        raise ValueError('Complete exact independent parent proof required')
    for name,expected in manifest.items():
        if sha(contained(p,name))!=expected:raise ValueError('Parent evidence changed: '+name)
    for name,expected in spec['source_sha256'].items():
        if sha(contained(n,name))!=expected:raise ValueError('Native source changed: '+name)
    rows=[json.loads(line) for line in (p/'rows.jsonl').read_text().splitlines()]
    train=[r for r in rows if r.get('role')=='train']
    if len(rows)!=1284 or len(train)!=1024 or sorted(r['train_index'] for r in train)!=list(range(1024)):
        raise ValueError('Complete1024 current TRAIN rows required')
    train=sorted(train,key=lambda r:r['train_index'])
    for r in train:
        if r.get('method')!='ppo' or r.get('perturbation_mixed_units')!=[0.]*5:raise ValueError('Current TRAIN only')
    controls=[next(r for r in rows if r['phase']=='controls_pre' and r['role']==role and r['repeat']==0)
              for role in ('positive_control','negative_control')]
    artifacts={k:copy.deepcopy(report['artifacts'][key]) for k,key in (('bc','bc'),('ppo','selected'))}
    for v in artifacts.values():
        if sha(contained(p,v['path']))!=v['sha256']:raise ValueError('Saved baseline artifact changed')
    return dict(spec=spec,report=report,rows=rows,train_rows=train,prior=read(p/'prior.json'),
        calibration=read(p/'calibration.json'),home=read(p/'setup-home.json'),controls=controls,
        artifacts=artifacts,bindings=dict(parent_contract_fingerprint=fp,parent_report_sha256=sha(p/'report.json'),
        parent_manifest_sha256=sha(p/'evidence-manifest.json'),parent_audit_sha256=sha(a),
        parent_rows_sha256=manifest['rows.jsonl'],train_rows_sha256=digest(train),
        train_record_sha256={r['id']:digest(r) for r in train},baseline_artifacts=artifacts))

def make_splits(parent_spec):
    excluded={tuple(map(float,xy)) for xy in parent_spec['exclusions']}
    for role in ('train','validation','test'):
        excluded.update(tuple(map(float,c['bucket_xy_m'])) for c in parent_spec['splits'][role])
    excluded=sorted(excluded);used=np.asarray(excluded,dtype=float);splits={}
    for role,count,seed in (('validation',24,20261603),('test',60,20261604)):
        rng=np.random.default_rng(seed);items=[]
        for _ in range(100000):
            if len(items)==count:break
            xy=rng.uniform([.1,-.65],[1.24,.25])
            if len(used) and np.linalg.norm(used-xy,axis=1).min()<.01:continue
            items.append(dict(id=f'physics-ensemble-{role}-{len(items):02d}',split=role,bucket_xy_m=xy.tolist()))
            used=np.vstack((used,xy))
        if len(items)!=count:raise ValueError('Unable to freeze disjoint fresh layouts')
        splits[role]=items
    return dict(splits=splits,excluded=[list(x) for x in excluded])

def protocol():
    return dict(substeps=64,methods=list(METHODS),historical_current_train_requests=1024,new_train_requests=0,
        validation_layouts=24,test_layouts=60,validation_requests=72,test_requests=180,control_requests=12,
        maximum_fresh_requests=264,noise=False,no_test_flight_search=True,fixed_model_no_validation_selection=True,minimum_split_distance_m=.01,
        validation_seed=20261603,test_seed=20261604,model_seeds=list(SEEDS),fit_steps=600,
        fit_lr=.001,fit_device='cpu',ensemble_members=5,candidates=16,maximum_geometry_checks=4,
        selection='Fixed final ensemble; no validation tuning; all252 evaluation predictions frozen before any evaluation outcomes',
        score='mean(admission_probability*capture_probability) -0.25*std(admission_probability*capture_probability) -0.05*mean(min(aperture_distance_m,2))',
        training_scope='Only1024 authenticated current parent TRAIN rows; all invalid physical targets masked; no historical6144 labels or held-out/control/diagnostic labels',scope=SCOPE)

def captured(r):return bool(r.get('native_release') and r.get('route_success') and not r.get('invalid') and r.get('invalid_physics') is False)
def counts(rows):
    cap=sum(captured(r) for r in rows);native=sum(r.get('native_release') is True for r in rows)
    invalid=sum(r.get('native_release') is True and (bool(r.get('invalid')) or bool(r.get('invalid_physics'))) for r in rows)
    return dict(requested=len(rows),native_releases=native,captures=cap,clean=sum(captured(r) and r.get('route_quality')==2 for r in rows),
                rejected=len(rows)-native,native_misses=native-cap-invalid,invalid=invalid)
