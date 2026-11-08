"""Strict read-only publication of diagnosis and a separately verified correction."""
import argparse
import copy
import json
from pathlib import Path
from .gap_flight_correction import VERSION, SCOPE, atomic, digest, read, sha
from .gap_flight_diagnosis import KEY, SOURCE_HASHES, VERSION as DIAGNOSIS_VERSION
from .physics_ensemble_study import counts

CORRECTION_KEY='gap_flight_correction'
AUDIT_VERSION='sircus-gap-flight-correction-checks-v1'


def diagnosis(folder):
    folder=Path(folder);value=read(folder/'diagnosis.json');proof=read(folder/'checks.json');hashes=read(folder/'checksums.json')
    for name in ('diagnosis.json','summary.json','checks.json'):
        if hashes.get(name)!=sha(folder/name):raise ValueError('Derived diagnosis bytes changed')
    checks=proof.get('checks')
    if (value.get('version')!=DIAGNOSIS_VERSION or value.get('status')!='completed'
        or value.get('processing',{}).get('source_sha256')!=sha(Path(__file__).with_name('gap_flight_diagnosis.py'))
        or proof.get('status')!='passed' or not isinstance(checks,list) or not checks
        or proof.get('checks_count')!=len(checks) or any(c.get('passed') is not True for c in checks)
        or value.get('audit',{}).get('source_sha256')!=hashes['checks.json']
        or any(value.get('source',{}).get(k+'_sha256')!=v for k,v in SOURCE_HASHES.items())
        or len(value.get('requests',[]))!=50 or len(value.get('misses',[]))!=18 or len(value.get('rejects',[]))!=8):
        raise ValueError('Complete byte-bound 50-position diagnosis required')
    return value


def correction(study,audit_path=None):
    study=Path(study);raw=read(study/'report.json');spec=read(study/'contract.json')
    unsigned=dict(spec);fp=unsigned.pop('fingerprint',None)
    if (spec.get('version')!=VERSION or raw.get('version')!=VERSION or digest(unsigned)!=fp
        or raw.get('contract_fingerprint')!=fp or spec.get('maximum_requests')!=30
        or spec.get('contexts')!=['12','24','48'] or spec.get('gap_delta_m')!=.060
        or spec.get('substeps')!=64 or spec.get('repeats')!=3 or spec.get('training') is not False):
        raise ValueError('Exact targeted correction protocol required')
    plan_sha=sha(study/'prediction-plan.json')
    if raw.get('plan_sha256')!=plan_sha:raise ValueError('Frozen plan binding differs')
    status=raw.get('status')
    value=dict(version=VERSION,status={'prepared':'pending','completed':'awaiting_audit'}.get(status,status),scope=SCOPE,
        protocol=dict(contexts=3,context_ids=spec['contexts'],repeats=3,maximum_requests=30,substeps=64,gap_delta_mm=60),
        plan_sha256=plan_sha,report_sha256=sha(study/'report.json'),audit=dict(passed=None),comparisons=[])
    if status=='queued':value['queue']=dict(reason='Waiting for existing4090 work and shared lock; left untouched',wait_s=raw.get('queue',{}).get('wait_s'))
    if raw.get('error'):value['error']=raw['error']
    if not audit_path or not Path(audit_path).is_file():return value
    proof=read(audit_path)
    if proof.get('status')=='failed':value.update(status='failed',audit=dict(passed=False));return value
    manifest=read(study/'evidence-manifest.json');checks=proof.get('checks')
    actual={str(p.relative_to(study)) for p in study.rglob('*') if p.is_file() and p.name not in ('report.json','evidence-manifest.json')}
    if set(manifest)!=actual or raw.get('evidence_sha256')!=manifest:raise ValueError('Complete correction manifest required')
    for name,expected in manifest.items():
        path=(study/name).resolve()
        if not path.is_relative_to(study.resolve()) or sha(path)!=expected:raise ValueError('Correction artifact changed')
    if (status!='completed' or raw.get('phase')!='completed' or proof.get('version')!='sircus-gap-flight-correction-audit-v1'
        or proof.get('status')!='passed' or proof.get('failures')!=[]
        or not isinstance(checks,list) or not checks or proof.get('checks_count')!=len(checks)
        or any(c.get('passed') is not True for c in checks)
        or proof.get('auditor_sha256')!=sha(Path(__file__).with_name('gap_flight_correction_verify.py'))
        or proof.get('contract_fingerprint')!=fp or proof.get('report_sha256')!=value['report_sha256']
        or proof.get('plan_sha256')!=plan_sha or proof.get('evidence_manifest_sha256')!=sha(study/'evidence-manifest.json')
        or proof.get('requested_trials')!=30):raise ValueError('Correction independent proof incomplete')
    rows=[json.loads(s) for s in (study/'records.jsonl').read_text().splitlines()]
    if len(rows)!=30 or proof.get('control_gates')!=raw.get('control_gates') or any(raw['control_gates'].get(k,{}).get('status')!='passed' for k in ('pre','post')):
        raise ValueError('Complete requests and matched controls required')
    comparisons=[dict(method=name,counts=counts([r for r in rows if r['variant']==variant])) for name,variant in (('original','original'),('corrected','gap_plus60'))]
    if any(c['counts']['requested']!=9 for c in comparisons):raise ValueError('All18 diagnostic slots required')
    manifest_sha=sha(study/'evidence-manifest.json')
    value.update(status='passed',manifest_sha256=manifest_sha,comparisons=comparisons,
        audit=dict(version=AUDIT_VERSION,passed=True,full_checks_verified=True,checks_count=len(checks),
            plan_sha256=plan_sha,report_sha256=value['report_sha256'],manifest_sha256=manifest_sha))
    return value


def publish(display,diagnosis_dir,study=None,audit_path=None,pipeline=None):
    display=Path(display);historical=diagnosis(diagnosis_dir)
    changed=correction(study,audit_path) if study and (Path(study)/'report.json').is_file() else None
    if pipeline and pipeline.get('status') in ('failed','audit_failed'):
        changed=changed or dict(version=VERSION,scope=SCOPE,protocol=dict(contexts=3,repeats=3,maximum_requests=30,substeps=64))
        changed.update(status='failed',error=pipeline.get('error') or 'Own pipeline failed; evidence preserved',audit=dict(passed=False),comparisons=[])
    for _ in range(3):
        before=display.read_bytes();data=json.loads(before)
        if not isinstance(data,dict):raise ValueError('Existing viewer report required')
        data[KEY]=copy.deepcopy(historical)
        if changed is not None:data[CORRECTION_KEY]=changed
        if display.read_bytes()==before:atomic(display,data);return data
    raise RuntimeError('Concurrent display writer preserved')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('display','diagnosis'):p.add_argument('--'+key,type=Path,required=True)
    for key in ('study','audit'):p.add_argument('--'+key,type=Path)
    a=p.parse_args();publish(a.display,a.diagnosis,a.study,a.audit)

if __name__=='__main__':main()
