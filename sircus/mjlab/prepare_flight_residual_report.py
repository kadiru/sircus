"""Publish only a byte-bound CPU diagnosis, preserving the existing viewer data."""
import argparse
import json
from pathlib import Path
from .consistent_flight_diagnosis import KEY, VERSION, AUDIT_VERSION, SOURCE_HASHES
from .consistent_flight_study import sha, atomic


def compact(directory):
    directory=Path(directory)
    read=lambda n:json.loads((directory/n).read_text())
    manifest=read('checksums.json')
    if set(manifest)!={'diagnosis.json','summary.json','checks.json'} or any(sha(directory/n)!=h for n,h in manifest.items()):
        raise ValueError('Exact derived diagnosis bytes required')
    result=read('diagnosis.json');proof=read('checks.json');source=result.get('source',{});audit=result.get('audit',{})
    checks=proof.get('checks')
    if (result.get('version')!=VERSION or result.get('status')!='completed' or source.get('verified') is not True
        or source.get('report_sha256')!=SOURCE_HASHES['report'] or source.get('audit_sha256')!=SOURCE_HASHES['audit']
        or source.get('manifest_sha256')!=SOURCE_HASHES['manifest'] or source.get('native_requests_added')!=0
        or proof.get('version')!=AUDIT_VERSION or proof.get('status')!='passed' or not isinstance(checks,list) or not checks
        or type(proof.get('checks_count')) is not int or proof['checks_count']!=len(checks)
        or not all(isinstance(c,dict) and c.get('passed') is True for c in checks)
        or proof.get('source_report_sha256')!=source['report_sha256'] or proof.get('source_audit_sha256')!=source['audit_sha256']
        or proof.get('diagnosis_source_sha256')!=result.get('processing',{}).get('source_sha256')
        or audit.get('source_sha256')!=manifest['checks.json'] or audit.get('status')!='passed'
        or audit.get('checks_count')!=len(checks) or audit.get('full_checks_verified') is not True):
        raise ValueError('Complete original/derived evidence bindings required')
    return result


def prepare(base,directory,output):
    base,directory,output=map(Path,(base,directory,output))
    if output.resolve()==base.resolve() or output.resolve().is_relative_to(directory.resolve()):
        raise ValueError('Separate display outside evidence required')
    result=json.loads(base.read_text())
    if not isinstance(result,dict):raise ValueError('Existing object display required')
    result[KEY]=compact(directory);atomic(output,result);return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('base','diagnosis','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();prepare(a.base,a.diagnosis,a.output)

if __name__=='__main__':main()
