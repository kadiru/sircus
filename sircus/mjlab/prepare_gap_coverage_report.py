"""Additive read-only gap ablation status; verified only with exact native proof."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
from .gap_coverage_study import VERSION, METHODS, protocol, digest

KEY='gap_coverage_study'
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


COUNT_FIELDS=('requested','native_releases','captures','clean','rejected','native_misses','invalid')


def complete_counts(value):
    """Strict finite requested/native accounting; missing fields stay unknown."""
    return (isinstance(value,dict) and set(value)==set(COUNT_FIELDS)
        and all(type(value[k]) is int and value[k]>=0 for k in COUNT_FIELDS)
        and value['requested']==value['native_releases']+value['rejected']
        and value['native_releases']==value['captures']+value['native_misses']+value['invalid']
        and value['clean']<=value['captures'])


def completed_evidence(raw,proof):
    """Reject internally incomplete proof/report pairs before VERIFIED display."""
    if raw.get('status')!='completed' or raw.get('phase')!='completed':return False
    counts=raw.get('counts');comparisons=raw.get('comparisons');gates=raw.get('control_gates')
    if (not complete_counts(counts) or counts['requested']!=112
        or type(proof.get('native_releases')) is not int or proof['native_releases']!=counts['native_releases']
        or not isinstance(comparisons,list) or len(comparisons)!=2
        or [c.get('method') if isinstance(c,dict) else None for c in comparisons]!=list(METHODS)
        or any(not complete_counts(c.get('counts')) or c['counts']['requested']!=50 for c in comparisons)
        or not isinstance(gates,dict) or set(gates)!={'pre','post'}
        or proof.get('control_gates')!=gates):return False
    groups=[c['counts'] for c in comparisons]
    for key in ('pre','post'):
        gate=gates[key]
        if not isinstance(gate,dict) or gate.get('status')!='passed' or not complete_counts(gate.get('counts')):return False
        control=gate['counts']
        if any(control[k]!=v for k,v in dict(requested=6,native_releases=6,captures=3,rejected=0,native_misses=3,invalid=0).items()):return False
        groups.append(control)
    return all(counts[k]==sum(c[k] for c in groups) for k in COUNT_FIELDS)


def compact(study,audit_path=None):
    study=Path(study);raw=read(study/'report.json');spec=read(study/'contract.json')
    unsigned=dict(spec);fp=unsigned.pop('fingerprint',None)
    if (raw.get('version')!=VERSION or spec.get('version')!=VERSION or raw.get('protocol')!=protocol()
        or spec.get('protocol')!=protocol() or digest(unsigned)!=fp or raw.get('contract_fingerprint')!=fp
        or spec.get('maximum_requests')!=112 or spec.get('new_training_requests')!=0):
        raise ValueError('Exact frozen50/112/fixed64/no-fit contract required')
    value=dict(version=VERSION,status=raw['status'],phase=raw['phase'],scope=raw['scope'],
        protocol=copy.deepcopy(raw['protocol']),counts=copy.deepcopy(raw.get('counts',{})),
        comparisons=copy.deepcopy(raw.get('comparisons',[])),planned_layouts=raw.get('planned_layouts',0),
        report_sha256=sha(study/'report.json'),audit=dict(passed=None),
        provenance=dict(source_report_sha256=raw['source_report_sha256'],selector_sha256=raw['selector_sha256'],
            candidate_sha256=spec['source_sha256']['sircus/mjlab/gap_coverage_candidates.py'],
            model_sha256=raw['artifacts']['ensemble']['sha256'],plan_sha256=raw.get('prediction_plan_sha256')),
        findings=['Only gap proposal slots08/09 change; other14 candidates and the16/top4 budgets are fixed.',
            'Same frozen velocity ensemble, flight ranking, native64 scene and calibrated route scoring.',
            '50 fresh layouts exclude all earlier evaluation/development layouts. No new model fitting.',
            'Broader gap coverage can trade away useful intermediate settings; improvement is not assumed.'])
    if raw.get('status')=='queued':
        value['queue']=dict(gpu='RTX4090',reason='Waiting for existing4090 work and shared lock; left untouched',wait_s=raw.get('queue',{}).get('wait_s'))
    elif raw.get('status')=='completed':
        value['queue']=dict(gpu='RTX4090',reason='Native work finished; released slot; verification separate')
    if raw.get('error'):value['error']=raw['error']
    if audit_path and Path(audit_path).is_file():
        proof=read(audit_path);checks=proof.get('checks')
        if proof.get('status')=='failed':value['audit']=dict(passed=False)
        elif (completed_evidence(raw,proof) and proof.get('version')=='sircus-gap-coverage-audit-v1'
            and proof.get('status')=='passed' and proof.get('failures')==[]
            and isinstance(checks,list) and bool(checks) and type(proof.get('checks_count')) is int
            and proof['checks_count']==len(checks) and all(isinstance(c,dict) and c.get('passed') is True and isinstance(c.get('label'),str) and bool(c['label'].strip()) for c in checks)
            and proof.get('report_sha256')==value['report_sha256']
            and proof.get('evidence_manifest_sha256')==sha(study/'evidence-manifest.json')
            and proof.get('contract_fingerprint')==fp and proof.get('requested_trials')==112
            and proof.get('training_requests')==0 and proof.get('comparisons')==raw['comparisons']
            and proof.get('producer_source_sha256')==spec['source_sha256']
            and proof.get('native_source_sha256')==spec['native_source_sha256']):
            value.update(status='verified',audit=dict(passed=True,report_sha256=value['report_sha256'],checks_count=len(checks)))
        else:raise ValueError('Incomplete/mismatched native proof; no verified publication')
    return value


def prepare(base,study,output,*,audit_path=None):
    base,study,output=map(Path,(base,study,output))
    if output.resolve()==base.resolve() or output.resolve().is_relative_to(study.resolve()):
        raise ValueError('Separate display outside evidence required')
    value=read(base)
    if not isinstance(value,dict):raise ValueError('Existing object display required')
    value[KEY]=compact(study,audit_path)
    tmp=output.with_suffix(output.suffix+'.tmp');tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');tmp.replace(output)
    return value


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('base','study','output'):p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--audit',type=Path);a=p.parse_args();prepare(a.base,a.study,a.output,audit_path=a.audit)

if __name__=='__main__':main()
