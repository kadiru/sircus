"""Literal midpoint protocol and authentication, without native execution."""
from __future__ import annotations
import copy
import json
from pathlib import Path
from . import imitation_regression_native as N
from .benchmark_verify import contained, read, sha, digest

VERSION=N.VERSION
PARENT_REPORT='1fde3f7de5b5c2912a133c858105388a704a80ab0e2cb39e8d79c0c7a189bf14'
PARENT_AUDIT='ea4992828178d9e62ba41c63579ecc3876cb5bffaacf6b590c828d45876a7f63'
PARENT_FP='297b367fa2702d688084bc2e74dd30a83c6cf4312ba078938def5f9b223adce4'
INPUT_KEYS=('parent-study','parent-audit','diagnosis','rule','reference','assets')


def protocol():
    return dict(substeps=64,maximum_requests=72,repeats=3,diagnostic_requests=54,control_requests=18,
        maximum_acquired_s=3600,maximum_queue_s=900,maximum_pipeline_s=6300,audit_reserve_s=1800,
        storage_reserve_bytes=8*1024**3,training=False,noise=False,posthoc_development=True,
        variants=list(N.VARIANTS),contexts=list(N.CASES),mode=N.MODE,
        interpolation_alpha=.5,parameter_indices=list(range(5)),
        old_refusals='Nine original corrected-policy refusal slots reuse authenticated decisions; not independent fresh IK searches.',
        correction='One strict new midpoint-action proof per case, reused across three nominal requests; no solver or guard changes.',
        derived_release='All five task coordinates blend together; target-facing yaw and absolute release Z may change. No single-coordinate causal isolation.',
        selection='All72 slots frozen before outcomes; every variant repeats three times, including declines. No search, retries, fitting or promotion.',
        realization='Compare admitted diagnostic poses with matched successful current recipe, never with an unflown placeholder.',
        controls='Original17 direct/table/miss before and after; current captures are diagnostic outcomes, not a hard gate.')


def authenticate_parent(parent,audit):
    parent,audit=Path(parent),Path(audit)
    if sha(parent/'report.json')!=PARENT_REPORT or sha(audit)!=PARENT_AUDIT:raise ValueError('Exact audited corrected-imitation parent required')
    report,spec,proof,manifest=(read(p) for p in (parent/'report.json',parent/'contract.json',audit,parent/'evidence-manifest.json'))
    unsigned=dict(spec);fp=unsigned.pop('fingerprint')
    if (digest(unsigned)!=fp or fp!=PARENT_FP or report['contract_fingerprint']!=fp or proof['contract_fingerprint']!=fp
            or report['status']!='completed' or report['phase']!='completed' or report['evidence_sha256']!=manifest
            or proof['status']!='passed' or proof['full_checks_verified'] is not True or proof['native_checks_verified'] is not True
            or proof['failures'] or proof['checks_count']!=55633 or len(proof['checks'])!=55633
            or not all(c.get('passed') is True for c in proof['checks']) or proof['report_sha256']!=PARENT_REPORT
            or proof['evidence_manifest_sha256']!=sha(parent/'evidence-manifest.json')):
        raise ValueError('Complete bound corrected-imitation proof required')
    files={}
    def verify(name):
        path=contained(parent,name)
        if sha(path)!=manifest[name]:raise ValueError('Parent artifact changed: '+name)
        files[name]=manifest[name];return path
    rows=[json.loads(x) for x in verify('rows.jsonl').read_text().splitlines()]
    if len(rows)!=158 or len({r['id'] for r in rows})!=158:raise ValueError('Complete158-row parent ledger required')
    pairs=N.source_pairs(rows);sources=[pair[m] for pair in pairs for m in ('current_bc','corrected_bc')]
    controls=read(verify('control-replays.json'))
    config={name:read(verify(name)) for name in ('contract.json','prior.json','old-prior.json','calibration.json','verification.json','setup-home.json')}
    if len(config['prior.json']['records'])!=19 or len(config['old-prior.json']['records'])!=17:raise ValueError('Separate19/17 geometry priors required')
    for namespace,members in (('native',sources),('control-parent/native',controls)):
        for name in ('effective-options.json','native-compiled-geometry.json'):verify(namespace+'/'+name)
        for row in members:
            for key,hkey in (('context_resolution_file','context_resolution_sha256'),('connection_file','connection_sha256')):
                name=namespace+'/'+row[key];verify(name)
                if files[name]!=row[hkey]:raise ValueError('Parent row proof differs: '+name)
    for row in sources:
        if row.get('native_release'):
            name='native/'+row['native_cache_file'];verify(name)
            if files[name]!=row['native_cache_sha256']:raise ValueError('Parent dense trajectory differs')
    return dict(spec=spec,rows=rows,sources=sources,controls=controls,config=config,
        bindings=dict(report_sha256=PARENT_REPORT,audit_sha256=PARENT_AUDIT,contract_fingerprint=fp,
            manifest_sha256=sha(parent/'evidence-manifest.json'),files_sha256=files))


def authenticate_diagnosis(path,rule):
    if sha(path)!=rule['diagnosis_sha256']:raise ValueError('Reviewed diagnosis bytes differ')
    diagnosis=read(path);unsigned=dict(diagnosis);fp=unsigned.pop('diagnosis_fingerprint')
    if digest(unsigned)!=fp or diagnosis['version']!='sircus-imitation-regression-diagnosis-v1':raise ValueError('Diagnosis fingerprint/version differs')
    source=diagnosis['source']
    for key,value in dict(report_sha256=PARENT_REPORT,audit_sha256=PARENT_AUDIT,contract_fingerprint=PARENT_FP,
                          full_checks_verified=True,native_checks_verified=True,checks_count=55633).items():
        if source.get(key)!=value:raise ValueError('Diagnosis parent binding differs: '+key)
    if [x['context_id'] for x in diagnosis['losses']]!=list(N.CASES):raise ValueError('Literal six-regression diagnosis order differs')
    return diagnosis
