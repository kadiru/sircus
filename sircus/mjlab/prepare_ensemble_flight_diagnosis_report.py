"""Byte-bound additive publication of completed recorded-flight diagnostics."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import re

KEY='ensemble_flight_diagnosis'
VERSION='sircus-ensemble-flight-diagnosis-v1'
REPORT_SHA='9d76ed2225f5e9a659d5b8ef18839207553df87e14980cf0a290707de437b5a6'
AUDIT_SHA='72e5cc9fd4c424896a0c41588b3eec0bd85317ce931cd76e90bd850474f87ebb'
SOURCE_FP='90a7d5b7e3d1fb1e738a36a7cf30930116516da0d6d6435c06b98eb97be8c376'


def read(path):return json.loads(Path(path).read_text())
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def is_hash(v):return isinstance(v,str) and re.fullmatch('[a-f0-9]{64}',v) is not None


def prepare(base,diagnosis,output):
    base,diagnosis,output=map(Path,(base,diagnosis,output))
    if output.resolve()==base.resolve() or output.resolve().is_relative_to(diagnosis.resolve()):
        raise ValueError('Separate derived display outside existing evidence required')
    result=read(base);raw=read(diagnosis/'diagnosis.json');proof=read(diagnosis/'checks.json');manifest=read(diagnosis/'checksums.json')
    if not isinstance(result,dict) or set(manifest)!={'diagnosis.json','summary.json','checks.json'}:
        raise ValueError('Complete bound diagnostic publication required')
    for name,expected in manifest.items():
        if sha(diagnosis/name)!=expected:raise ValueError('Diagnostic bytes changed: '+name)
    source=raw.get('source',{});audit=raw.get('audit',{});checks=proof.get('checks')
    processing=raw.get('processing',{})
    if (raw.get('version')!=VERSION or raw.get('status')!='completed' or source.get('verified') is not True
        or source.get('report_sha256')!=REPORT_SHA or source.get('audit_sha256')!=AUDIT_SHA or source.get('contract_fingerprint')!=SOURCE_FP
        or type(source.get('native_requests_added')) is not int or source['native_requests_added']!=0 or source.get('original_native_audit_checks')!=897304
        or proof.get('version')!='sircus-ensemble-flight-diagnosis-checks-v1' or proof.get('status')!='passed'
        or not isinstance(checks,list) or not checks or type(proof.get('checks_count')) is not int
        or proof['checks_count']!=len(checks) or any(not isinstance(c,dict) or c.get('passed') is not True for c in checks)
        or proof.get('source_report_sha256')!=REPORT_SHA or proof.get('source_audit_sha256')!=AUDIT_SHA
        or not is_hash(processing.get('source_sha256')) or proof.get('diagnosis_source_sha256')!=processing['source_sha256']
        or processing.get('cuda_used') is not False or processing.get('training') is not False
        or type(processing.get('new_native_requests')) is not int or processing['new_native_requests']!=0
        or audit.get('version')!='sircus-ensemble-flight-diagnosis-checks-v1'
        or audit.get('source_sha256')!=sha(diagnosis/'checks.json') or audit.get('checks_count')!=len(checks)
        or audit.get('full_checks_verified') is not True or audit.get('status')!='passed'):
        raise ValueError('Exact completed source and strictly passed derived checks required')
    scalar=copy.deepcopy(raw)
    for case in scalar['cases']:
        case.pop('trace');case.pop('forecasts')
    if scalar!=read(diagnosis/'summary.json'):
        raise ValueError('Sanitized summary differs from published diagnostic evidence')
    result=copy.deepcopy(result);result[KEY]=raw
    output.parent.mkdir(parents=True,exist_ok=True)
    temp=output.with_suffix(output.suffix+'.tmp');temp.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');temp.replace(output)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('base','diagnosis','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();prepare(a.base,a.diagnosis,a.output)


if __name__=='__main__':main()
