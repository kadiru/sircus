"""Publish independently verified case01 horizon evidence as scalar UI data.

Read-only evidence input; preserves all older display keys. No simulator imports,
model activation, score changes, hardware commands or native execution.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path

KEY='clearance_horizon_study'
VERSION='sircus-clearance-horizon-display-v1'
STUDY_VERSION='sircus-clearance-horizon-study-v1'
AUDIT_VERSION='sircus-clearance-horizon-audit-v1'
METHODS=('short_window','long_window')
CONTEXT='arm-clearance-test-01'
LABELS=dict(short_window='Shorter rim window',long_window='Longer table window')
COUNT_FIELDS=('requested','native_releases','captures','clean','rejected','native_misses','invalid')
OBSERVATIONS=('robot_contact_before_terminal','robot_contact_after_terminal',
              'table_contact_before_terminal','table_contact_after_terminal','bucket_retained')


def read(path):return json.loads(Path(path).read_text())
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    return h.hexdigest()
def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,allow_nan=False).encode()).hexdigest()
def require(value,label):
    if not value:raise ValueError(label)
def finite(value):return type(value) in (int,float) and math.isfinite(value)
def integer(value):return type(value) is int and value>=0
def vector(value,size):return isinstance(value,list) and len(value)==size and all(finite(v) for v in value)
def hash_value(value):return isinstance(value,str) and len(value)==64 and all(c in '0123456789abcdef' for c in value)
def source_map(value):return isinstance(value,dict) and bool(value) and all(isinstance(k,str) and k and hash_value(v) for k,v in value.items())
def valid_counts(value):
    return (isinstance(value,dict) and all(integer(value.get(k)) for k in COUNT_FIELDS)
        and value['requested']==value['native_releases']+value['rejected']
        and value['native_releases']==value['captures']+value['native_misses']+value['invalid']
        and value['clean']<=value['captures'])


def complete_checks(proof):
    checks=proof.get('checks')
    require(proof.get('version')==AUDIT_VERSION and proof.get('status')=='passed' and proof.get('failures')==[]
        and isinstance(checks,list) and checks and integer(proof.get('checks_count')) and proof['checks_count']==len(checks)
        and all(isinstance(c,dict) and c.get('passed') is True and isinstance(c.get('label'),str) and c['label'].strip() for c in checks),
        'Complete independent proof required')
    return len(checks)


def verify_manifest(study,raw):
    manifest=read(study/'evidence-manifest.json')
    actual={str(p.relative_to(study)) for p in study.rglob('*') if p.is_file() and p.name not in ('report.json','evidence-manifest.json')}
    require(isinstance(manifest,dict) and set(manifest)==actual and raw.get('evidence_sha256')==manifest,'Complete unchanged native manifest required')
    for name,expected in manifest.items():
        path=(study/name).resolve();require(path.is_relative_to(study.resolve()) and hash_value(expected) and sha(path)==expected,'Changed native artifact: '+name)
    return sha(study/'evidence-manifest.json')


def validate_native(raw,proof):
    require(raw.get('status')==raw.get('phase')=='completed','Completed native replay required')
    total,groups,gates=raw.get('counts'),raw.get('comparisons'),raw.get('control_gates')
    require(valid_counts(total) and total['requested']==18 and isinstance(groups,list) and [g.get('method') for g in groups]==list(METHODS)
        and all(valid_counts(g.get('counts')) and g['counts']['requested']==3 for g in groups),'Exactly six paired trials and twelve controls required')
    require(gates==proof.get('control_gates') and isinstance(gates,dict) and set(gates)=={'pre','post'},'Matched control proof required')
    sums=[g['counts'] for g in groups]
    for phase in ('pre','post'):
        gate=gates[phase];c=gate.get('counts');require(gate.get('status')=='passed' and valid_counts(c)
            and all(c[k]==v for k,v in dict(requested=6,native_releases=6,captures=3,clean=3,rejected=0,native_misses=3,invalid=0).items()),'Six matched controls must pass/'+phase);sums.append(c)
    require(all(total[k]==sum(c[k] for c in sums) for k in COUNT_FIELDS) and proof.get('comparisons')==groups and proof.get('counts')==total and proof.get('native_releases')==total['native_releases'],'Independent totals must agree')
    original,enriched=raw.get('rows'),proof.get('rows')
    require(isinstance(original,list) and isinstance(enriched,list) and len(original)==len(enriched)==6,'All six audited paired rows required')
    old={r['record_id']:r for r in original};require(len(old)==len({r['record_id'] for r in enriched})==6,'Unique original and audited identities required')
    allowed=('record_id','context_id','repeat','method','bucket_xy_m','values','native_release','capture','clean','invalid','clearance_rejected',*OBSERVATIONS)
    rows=[]
    for source in enriched:
        previous=old.get(source['record_id']);require(previous is not None and all(source.get(k)==v for k,v in previous.items()),'Native audit cannot replace observed outcomes')
        require(source.get('method') in METHODS and source.get('context_id')==CONTEXT and integer(source.get('repeat')) and source['repeat']<3 and vector(source.get('bucket_xy_m'),2) and vector(source.get('values'),5),'Exact matched case/repeat/settings required')
        require(all(type(source.get(k)) is bool for k in ('native_release','capture','clean','invalid','clearance_rejected'))
            and (not source['capture'] or source['native_release'] and not source['invalid']) and (not source['clean'] or source['capture'])
            and (not source['invalid'] or source['native_release']) and (not source['clearance_rejected'] or not source['native_release']),'Consistent typed outcome flags required')
        for key in OBSERVATIONS:
            require(source.get(key) is None or type(source[key]) is bool,'Unknown contact/retention must remain null')
            require(source['native_release'] or source.get(key) is None,'Nonexecuted shots have no native contact/retention observations')
        rows.append({k:copy.deepcopy(source.get(k)) for k in allowed})
    reference={}
    for method in METHODS:
        group=[r for r in rows if r['method']==method];require(len(group)==len({r['repeat'] for r in group})==3,'Three exact repeats per method required')
        counts=dict(requested=3,native_releases=sum(r['native_release'] for r in group),captures=sum(r['capture'] for r in group),clean=sum(r['clean'] for r in group),rejected=sum(not r['native_release'] for r in group),native_misses=sum(r['native_release'] and not r['capture'] and not r['invalid'] for r in group),invalid=sum(r['native_release'] and r['invalid'] for r in group))
        require(counts==next(g['counts'] for g in groups if g['method']==method),'Displayed outcomes must match cohort counts')
        if method==METHODS[0]:reference={r['repeat']:r for r in group}
        else:require(all(r['bucket_xy_m']==reference[r['repeat']]['bucket_xy_m'] and r['values']==reference[r['repeat']]['values'] for r in group),'Paired shot and bucket must remain fixed')
    return rows


def compact(study,audit_path=None):
    base=dict(version=VERSION,status='pending',audit=dict(passed=None),comparisons=[],rows=[],windows=[],
        recommendation='The existing selector remains active. No training or promotion is triggered by this replay.',
        scope='One previously evaluated layout is now development evidence. Synthetic fixed64 physics, held endpoints and privileged geometry. Conditional predictions are not physical or full-path validation. Strict scoring is unchanged.')
    if study is None or not (Path(study)/'report.json').is_file():return base
    study=Path(study).resolve();raw,spec=read(study/'report.json'),read(study/'contract.json')
    unsigned=dict(spec);fp=unsigned.pop('fingerprint',None)
    require(spec.get('version')==raw.get('version')==STUDY_VERSION and digest(unsigned)==fp and raw.get('contract_fingerprint')==fp,'Exact bounded horizon contract required')
    require(spec.get('substeps')==raw.get('substeps')==64 and spec.get('protocol')==raw.get('protocol') and spec.get('protocol',{}).get('new_training_requests')==0 and spec.get('training') is raw.get('training') is False and spec.get('noise') is False and spec.get('methods')==list(METHODS) and spec.get('context_id')==CONTEXT and spec.get('repeats')==3,'Fixed64 single-case two-method no-training protocol required')
    require(spec.get('maximum_requests')==spec['protocol'].get('maximum_requests')==18 and spec['protocol'].get('substeps')==64 and spec['protocol'].get('methods')==list(METHODS),'Exactly eighteen requested slots required')
    protocol=dict(methods=list(METHODS),context_id=CONTEXT,substeps=64,new_training_requests=0,requests_per_method=3,maximum_requests=18,evaluation_scope='development_replay')
    plan_hash=sha(study/'prediction-plan.json')
    require(raw.get('plan_sha256')==plan_hash and spec.get('selection_plan_sha256')==sha(study/'selection-plan.json'),'Frozen replay or selection plan changed')
    base.update(status={'completed':'awaiting_audit','prepared':'pending'}.get(raw.get('status'),raw.get('status','unknown')),
        description=raw.get('correction_description',spec.get('correction_description','Only the predicted clearance horizon extends; same shot, five paths, held geometry, +2 mm margin, physics and strict scoring.')),
        protocol=protocol,progress_requests=raw.get('counts',{}).get('requested'),report_sha256=sha(study/'report.json'),manifest_sha256=None,plan_sha256=plan_hash,contract_fingerprint=fp)
    if not audit_path or not Path(audit_path).is_file():return base
    proof=read(audit_path)
    if proof.get('status')=='failed':base.update(status='audit_failed',audit=dict(passed=False));return base
    checks=complete_checks(proof);mh=verify_manifest(study,raw)
    require(proof.get('report_sha256')==base['report_sha256'] and proof.get('evidence_manifest_sha256')==mh
        and proof.get('prediction_plan_sha256',proof.get('plan_sha256'))==plan_hash and proof.get('contract_fingerprint')==fp
        and proof.get('requested_trials')==18 and proof.get('training_requests')==0
        and proof.get('producer_source_sha256')==spec.get('diagnostic_source_sha256') and source_map(spec.get('diagnostic_source_sha256'))
        and proof.get('native_source_sha256')==spec.get('native_source_sha256') and source_map(spec.get('native_source_sha256'))
        and proof.get('auditor_sha256')==sha(Path(__file__).with_name('horizon_window_verify.py')),'Complete matching native proof required')
    rows=validate_native(raw,proof)
    windows=proof.get('windows')
    require(isinstance(windows,list) and windows==raw.get('windows') and [w.get('method') for w in windows]==list(METHODS),'Audited windows for both methods must match the saved report')
    for w in windows:
        require(type(w.get('passed')) is bool and w.get('required_clearance_m')==.002
            and all(w.get(k) is None or finite(w[k]) for k in ('horizon_min_s','horizon_max_s','minimum_clearance_lower_bound_m'))
            and (w.get('horizon_min_s') is None or w['horizon_min_s']>0)
            and (w.get('horizon_max_s') is None or w.get('horizon_min_s') is not None and w['horizon_min_s']<=w['horizon_max_s']<=1.5),'Finite ordered audited windows required')
        require(all(r['clearance_rejected']==(not w['passed']) for r in rows if r['method']==w['method']),'Exact same-shot admission decision must match all repeats')
    base.update(status='verified',counts=copy.deepcopy(raw['counts']),control_gates=copy.deepcopy(raw['control_gates']),comparisons=[dict(method=g['method'],label=LABELS[g['method']],counts=copy.deepcopy(g['counts'])) for g in raw['comparisons']],rows=rows,
        windows=[{k:copy.deepcopy(w.get(k)) for k in ('method','passed','horizon_min_s','horizon_max_s','minimum_clearance_lower_bound_m','required_clearance_m')} for w in windows],manifest_sha256=mh,
        audit=dict(passed=True,full_checks_verified=True,checks_count=checks,source_sha256=sha(audit_path),report_sha256=base['report_sha256'],manifest_sha256=mh,plan_sha256=plan_hash,contract_fingerprint=fp))
    before,after=(g['counts'] for g in base['comparisons']);gain=after['captures']-before['captures']
    base['recommendation']=('The longer window recovered '+str(gain)+' strict captures at this known layout. This is development evidence, not generalization or permission to activate a selector.' if gain>0 else 'The longer window did not improve strict capture at this known layout. Withholding a shot is not a capture or an observed collision-free flight. Keep the existing selector unchanged.')
    return base


def publish(display,study,audit_path=None,pipeline=None):
    display=Path(display).resolve()
    if audit_path is not None:require(display!=Path(audit_path).resolve(),'Display must not replace evidence')
    if study is not None:require(not display.is_relative_to(Path(study).resolve()),'Display must stay outside immutable native evidence')
    result=compact(study,audit_path)
    if pipeline and pipeline.get('status') in ('failed','audit_failed'):result.update(status=pipeline['status'],comparisons=[],rows=[],windows=[],audit=dict(passed=False))
    for _ in range(3):
        previous=display.read_bytes();doc=json.loads(previous);require(isinstance(doc,dict),'Existing object display required');doc[KEY]=result
        if display.read_bytes()!=previous:continue
        tmp=display.with_suffix(display.suffix+'.clearance-horizon.tmp');tmp.write_text(json.dumps(doc,indent=2,allow_nan=False)+'\n')
        if display.read_bytes()!=previous:tmp.unlink();continue
        tmp.replace(display);return result
    raise RuntimeError('Concurrent display writer preserved')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--display',type=Path,required=True);p.add_argument('--study',type=Path);p.add_argument('--audit',type=Path);a=p.parse_args();publish(a.display,a.study,a.audit)
if __name__=='__main__':main()
