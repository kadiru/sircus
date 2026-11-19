"""Publish authenticated clearance diagnosis and one bounded native correction.

Saved scalar evidence only. Does not import a simulator, activate a selector or
modify original reports. Diagnosis may be shown while the native replay waits.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path

KEY = 'clearance_prediction_diagnosis'
VERSION = 'sircus-clearance-diagnosis-display-v1'
DIAGNOSIS_VERSION = 'sircus-clearance-prediction-diagnosis-v1'
DIAGNOSIS_AUDIT_VERSION = 'sircus-clearance-prediction-diagnosis-audit-v1'
STUDY_VERSION = 'sircus-clearance-correction-v1'
AUDIT_VERSION = 'sircus-clearance-correction-audit-v1'
PARENT_AUDIT_SHA256 = 'd3f91a2243483c0aa5deede9e5bf1f5770646c3845fdcab7766730d13420d2bf'
METHODS = ('original', 'corrected')
CONTEXTS = ('00', '01', '04', '07', '39', '44')
COUNT_FIELDS = ('requested', 'native_releases', 'captures', 'clean', 'rejected', 'native_misses', 'invalid')
LABELS = dict(original='Frozen clearance gate', corrected='Corrected distance bound')


def read(path): return json.loads(Path(path).read_text())
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
def valid_counts(value):
    return (isinstance(value,dict) and all(integer(value.get(k)) for k in COUNT_FIELDS)
        and value['requested']==value['native_releases']+value['rejected']
        and value['native_releases']==value['captures']+value['native_misses']+value['invalid']
        and value['clean']<=value['captures'])

def complete_checks(proof,version):
    checks=proof.get('checks')
    require(proof.get('version')==version and proof.get('status')=='passed' and proof.get('failures')==[]
        and isinstance(checks,list) and checks and integer(proof.get('checks_count')) and proof['checks_count']==len(checks)
        and all(isinstance(c,dict) and c.get('passed') is True and isinstance(c.get('label'),str) and c['label'].strip() for c in checks),
        'Complete independent proof required')
    return len(checks)


def diagnosis_compact(report_path,audit_path=None):
    if report_path is None or not Path(report_path).is_file():return dict(status='pending',audit=dict(passed=None),finger_cases=[],horizon_cases=[])
    raw=read(report_path);result=dict(status='awaiting_audit',report_sha256=sha(report_path),source_audit_sha256=raw.get('source',{}).get('audit_sha256'),audit=dict(passed=None),finger_cases=[],horizon_cases=[])
    require(raw.get('version')==DIAGNOSIS_VERSION,'Known diagnosis version required')
    if not audit_path or not Path(audit_path).is_file():return result
    proof=read(audit_path)
    if proof.get('status')=='failed':result.update(status='audit_failed',audit=dict(passed=False));return result
    checks=complete_checks(proof,DIAGNOSIS_AUDIT_VERSION)
    require(raw.get('status')=='completed' and proof.get('report_sha256')==result['report_sha256']
        and result['source_audit_sha256']==proof.get('source_audit_sha256')==PARENT_AUDIT_SHA256,
        'Diagnosis must bind this exact independently audited source and derivative')
    require(proof.get('auditor_source_sha256')==raw.get('source',{}).get('diagnostic_source_sha256')==sha(Path(__file__).with_name('clearance_prediction_diagnosis.py')), 'Exact diagnosis implementation required')
    fingers,horizons=raw.get('finger_cases'),raw.get('horizon_cases')
    require(isinstance(fingers,list) and fingers and isinstance(horizons,list) and horizons,'Concrete finger and horizon evidence required')
    horizons=[dict(r,guard_end_min_s=r.get('predicted_horizon_min_s'),guard_end_max_s=r.get('predicted_horizon_max_s'),first_contact_delay_s=r.get('observed_first_contact_s'),robot_contact_delay_s=r.get('observed_robot_contact_s')) for r in horizons]
    finger_keys=('context_id','bucket_xy_m','native_capture','native_robot_contact','old_lower_bound_m','corrected_lower_bound_m','closest_geom','observed_first_flight_sample_clearance_m','observed_distance_scope','diagnosis')
    horizon_keys=('context_id','guard_end_min_s','guard_end_max_s','first_contact_delay_s','first_contact_channels','robot_contact_delay_s','diagnosis')
    for row in fingers:
        require(isinstance(row.get('context_id'),str) and row['context_id'] and all(row.get(k) is None or type(row[k]) is bool for k in ('native_capture','native_robot_contact'))
            and all(row.get(k) is None or finite(row[k]) for k in ('old_lower_bound_m','corrected_lower_bound_m','observed_first_flight_sample_clearance_m')),'Typed finger observations required')
    for row in horizons:
        require(isinstance(row.get('context_id'),str) and row['context_id'] and all(row.get(k) is None or finite(row[k]) and row[k]>=0 for k in ('guard_end_min_s','guard_end_max_s','first_contact_delay_s','robot_contact_delay_s')),'Finite relative horizon times required')
        require(row.get('guard_end_min_s') is None or row.get('guard_end_max_s') is None or row['guard_end_min_s']<=row['guard_end_max_s'],'Ordered horizon interval required')
        require(row.get('first_contact_channels') is None or isinstance(row['first_contact_channels'],list) and all(isinstance(v,str) for v in row['first_contact_channels']),'Known contact channels or unknown required')
    result.update(status='verified',finger_cases=[{k:copy.deepcopy(r.get(k)) for k in finger_keys} for r in fingers],
        horizon_cases=[{k:copy.deepcopy(r.get(k)) for k in horizon_keys} for r in horizons],
        finger_summary='Predicted bounds subtract the ball radius and between-sample motion allowance; a bound of at least +2 mm is required to pass. The saved-path column reconstructs geometric clearance at saved ball centers and held joints, without that allowance. It is not measured continuous clearance. Native outcomes in this table come from the prior study.',horizon_summary='Times are measured from rebound. The original guard ends at predicted rim entry; later contacts are outside that window. This replay changes only the geometric bound, not the horizon.',scope=raw.get('scope'),
        audit=dict(passed=True,full_checks_verified=True,checks_count=checks,source_sha256=sha(audit_path),
            report_sha256=result['report_sha256'],source_audit_sha256=result['source_audit_sha256']))
    return result


def verify_manifest(study,raw):
    manifest=read(study/'evidence-manifest.json');actual={str(p.relative_to(study)) for p in study.rglob('*') if p.is_file() and p.name not in ('report.json','evidence-manifest.json')}
    require(isinstance(manifest,dict) and set(manifest)==actual and raw.get('evidence_sha256')==manifest,'Complete unchanged native manifest required')
    for name,expected in manifest.items():
        p=(study/name).resolve();require(p.is_relative_to(study.resolve()) and sha(p)==expected,'Changed native artifact: '+name)
    return sha(study/'evidence-manifest.json')


def validate_native(raw,proof):
    require(raw.get('status')==raw.get('phase')=='completed','Completed native replay required')
    total,groups,gates=raw.get('counts'),raw.get('comparisons'),raw.get('control_gates')
    require(valid_counts(total) and total['requested']==48 and isinstance(groups,list) and [g.get('method') for g in groups]==list(METHODS)
        and all(valid_counts(g.get('counts')) and g['counts']['requested']==18 for g in groups),'Exactly 36 paired trials and12controls required')
    require(gates==proof.get('control_gates') and isinstance(gates,dict) and set(gates)=={'pre','post'},'Matched control proof required')
    sums=[g['counts'] for g in groups]
    for phase in ('pre','post'):
        gate=gates[phase];c=gate.get('counts');require(gate.get('status')=='passed' and valid_counts(c)
            and all(c[k]==v for k,v in dict(requested=6,native_releases=6,captures=3,clean=3,rejected=0,native_misses=3,invalid=0).items()),'Six matched controls must pass/'+phase);sums.append(c)
    require(all(total[k]==sum(c[k] for c in sums) for k in COUNT_FIELDS) and proof.get('comparisons')==groups and proof.get('counts')==total and proof.get('native_releases')==total['native_releases'],'Independent totals must agree')
    original=raw.get('rows',raw.get('test_positions'));enriched=proof.get('rows',proof.get('test_positions'))
    require(isinstance(original,list) and isinstance(enriched,list) and len(original)==len(enriched)==36,'All36 audited paired rows required')
    old={r['record_id']:r for r in original};require(len(old)==len({r['record_id'] for r in enriched})==36,'Unique original and audited identities required')
    allowed=('record_id','context_id','repeat','method','bucket_xy_m','values','native_release','capture','clean','invalid','clearance_rejected','robot_contact_before_terminal','robot_contact_after_terminal')
    rows=[]
    for source in enriched:
        previous=old.get(source['record_id']);require(previous is not None and all(source.get(k)==v for k,v in previous.items()),'Native audit cannot replace observed outcomes')
        require(source.get('method') in METHODS and isinstance(source.get('context_id'),str) and source['context_id'] and integer(source.get('repeat')) and 0<=source['repeat']<3 and vector(source.get('bucket_xy_m'),2) and vector(source.get('values'),5),'Exact matched case/repeat/settings required')
        require(all(type(source.get(k)) is bool for k in ('native_release','capture','clean','invalid','clearance_rejected'))
            and (not source['capture'] or source['native_release'] and not source['invalid']) and (not source['clean'] or source['capture'])
            and (not source['clearance_rejected'] or not source['native_release']),'Consistent typed native outcome flags required')
        for k in ('robot_contact_before_terminal','robot_contact_after_terminal'):
            require(source.get(k) is None or type(source[k]) is bool,'Unknown contact must remain null')
            require(source['native_release'] or source.get(k) is None,'Nonexecuted shots have no native contacts')
        rows.append({k:copy.deepcopy(source.get(k)) for k in allowed})
    reference={}
    for method in METHODS:
        group=[r for r in rows if r['method']==method];require(len(group)==len({(r['context_id'],r['repeat']) for r in group})==18 and len({r['context_id'] for r in group})==6,'Every case repeated three times per method required')
        counts=dict(requested=18,native_releases=sum(r['native_release'] for r in group),captures=sum(r['capture'] for r in group),clean=sum(r['clean'] for r in group),rejected=sum(not r['native_release'] for r in group),native_misses=sum(r['native_release'] and not r['capture'] and not r['invalid'] for r in group),invalid=sum(r['native_release'] and r['invalid'] for r in group))
        require(counts==next(g['counts'] for g in groups if g['method']==method),'Displayed outcomes must match cohort counts')
        if method==METHODS[0]:reference={(r['context_id'],r['repeat']):r for r in group}
        else:require(all((r['context_id'],r['repeat']) in reference and r['bucket_xy_m']==reference[(r['context_id'],r['repeat'])]['bucket_xy_m'] for r in group),'Paired bucket coordinates must match')
    return rows


def correction_compact(study,audit_path=None):
    if study is None or not (Path(study)/'report.json').is_file():return dict(status='pending',audit=dict(passed=None),comparisons=[],rows=[])
    study=Path(study).resolve();raw,spec=read(study/'report.json'),read(study/'contract.json')
    unsigned=dict(spec);fp=unsigned.pop('fingerprint',None)
    require(spec.get('version')==raw.get('version')==STUDY_VERSION and digest(unsigned)==fp and raw.get('contract_fingerprint')==fp,'Exact bounded correction contract required')
    require(spec.get('substeps')==raw.get('substeps')==64 and spec.get('protocol')==raw.get('protocol') and spec.get('protocol',{}).get('new_training_requests')==0 and spec.get('training') is raw.get('training') is False and spec.get('noise') is False and spec.get('methods')==list(METHODS) and spec.get('contexts')==list(CONTEXTS) and spec.get('repeats')==3,'Fixed64 two-method no-training protocol required')
    require(spec.get('maximum_requests')==spec['protocol'].get('maximum_requests')==48 and spec['protocol'].get('substeps')==64 and spec['protocol'].get('methods')==list(METHODS),'Exactly48 request ceiling required')
    protocol=dict(methods=list(METHODS),substeps=64,new_training_requests=0,requests_per_method=18,maximum_requests=48,evaluation_scope='development_replay')
    plan=study/'prediction-plan.json';plan_hash=sha(plan) if plan.is_file() else None
    require(raw.get('plan_sha256')==plan_hash and plan_hash is not None and spec.get('selection_plan_sha256')==sha(study/'selection-plan.json'),'Frozen replay or selection plan changed')
    result=dict(status={'completed':'awaiting_audit','prepared':'pending'}.get(raw.get('status'),raw.get('status','unknown')),description=raw.get('correction_description',spec.get('correction_description','Replace only the geometric distance bound; retain the original flight horizon and all other settings.')),
        protocol=protocol,progress_requests=raw.get('counts',{}).get('requested'),comparisons=[],rows=[],audit=dict(passed=None),
        report_sha256=sha(study/'report.json'),manifest_sha256=None,plan_sha256=plan_hash,contract_fingerprint=fp)
    if not audit_path or not Path(audit_path).is_file():return result
    proof=read(audit_path)
    if proof.get('status')=='failed':result.update(status='audit_failed',audit=dict(passed=False));return result
    checks=complete_checks(proof,AUDIT_VERSION);mh=verify_manifest(study,raw)
    require(proof.get('report_sha256')==result['report_sha256'] and proof.get('evidence_manifest_sha256')==mh
        and proof.get('prediction_plan_sha256',proof.get('plan_sha256'))==plan_hash and plan_hash is not None and proof.get('contract_fingerprint')==fp
        and proof.get('requested_trials')==48 and proof.get('training_requests')==0
        and proof.get('producer_source_sha256')==spec.get('diagnostic_source_sha256') and isinstance(spec.get('diagnostic_source_sha256'),dict) and spec['diagnostic_source_sha256']
        and proof.get('native_source_sha256')==spec.get('native_source_sha256') and isinstance(spec.get('native_source_sha256'),dict) and spec['native_source_sha256']
        and proof.get('source_parent',{}).get('audit_sha256')==PARENT_AUDIT_SHA256
        and proof.get('auditor_sha256')==sha(Path(__file__).with_name('clearance_correction_verify.py')),'Complete matching native proof required')
    rows=validate_native(raw,proof)
    result.update(status='verified',counts=copy.deepcopy(raw['counts']),control_gates=copy.deepcopy(raw['control_gates']),comparisons=[dict(method=g['method'],label=LABELS[g['method']],counts=copy.deepcopy(g['counts'])) for g in raw['comparisons']],rows=rows,manifest_sha256=mh,
        audit=dict(passed=True,full_checks_verified=True,checks_count=checks,source_sha256=sha(audit_path),report_sha256=result['report_sha256'],manifest_sha256=mh,plan_sha256=plan_hash,contract_fingerprint=fp))
    return result


def compact(diagnosis_report,diagnosis_audit=None,study=None,audit_path=None):
    diagnosis=diagnosis_compact(diagnosis_report,diagnosis_audit);correction=correction_compact(study,audit_path)
    require(correction.get('status')!='verified' or diagnosis.get('status')=='verified','Verified diagnosis required for corrected native interpretation')
    recommendation='The existing selector remains active. No new training or promotion is triggered by this diagnostic replay.'
    if correction.get('status')=='verified':
        before,after=(g['counts'] for g in correction['comparisons']);delta=after['captures']-before['captures']
        recommendation=('The candidate recovered '+str(delta)+' captures in this targeted development replay. A new frozen held-out comparison is still required before promotion.' if delta>0 else 'The candidate did not improve capture in this targeted replay. Keep the existing selector; investigate the remaining prediction error before more training.')
    return dict(version=VERSION,status='verified' if diagnosis.get('status')==correction.get('status')=='verified' else 'pending',diagnosis=diagnosis,correction=correction,recommendation=recommendation,
        scope='Previously observed layouts are development evidence. Synthetic fixed64 physics, held endpoints and privileged geometry; predictions are conditional bounds, not physical or full-path validation. No new model training or active selector change.')


def publish(display,diagnosis_report,diagnosis_audit=None,study=None,audit_path=None,pipeline=None):
    display=Path(display).resolve()
    for source in (diagnosis_report,diagnosis_audit,audit_path):
        if source is not None:require(display!=Path(source).resolve(),'Display must not replace evidence')
    if study is not None:require(not display.is_relative_to(Path(study).resolve()),'Display must stay outside immutable native evidence')
    result=compact(diagnosis_report,diagnosis_audit,study,audit_path)
    if pipeline and pipeline.get('status') in ('failed','audit_failed'):
        result['status']='failed';result['correction'].update(status=pipeline['status'],comparisons=[],rows=[],audit=dict(passed=False))
    for _ in range(3):
        previous=display.read_bytes();doc=json.loads(previous);require(isinstance(doc,dict),'Existing object display required');doc[KEY]=result
        if display.read_bytes()!=previous:continue
        tmp=display.with_suffix(display.suffix+'.clearance-diagnosis.tmp');tmp.write_text(json.dumps(doc,indent=2,allow_nan=False)+'\n')
        if display.read_bytes()!=previous:tmp.unlink();continue
        tmp.replace(display);return result
    raise RuntimeError('Concurrent display writer preserved')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--display',type=Path,required=True)
    p.add_argument('--diagnosis-report',type=Path,required=True);p.add_argument('--diagnosis-audit',type=Path)
    p.add_argument('--study',type=Path);p.add_argument('--audit',type=Path);a=p.parse_args()
    publish(a.display,a.diagnosis_report,a.diagnosis_audit,a.study,a.audit)
if __name__=='__main__':main()
