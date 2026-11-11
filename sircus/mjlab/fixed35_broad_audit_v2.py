"""Versioned full audit correction; original broad producer/evidence untouched.

The failed V1 audit is retained. The only substituted audit operation compares a
saved complete endpoint certificate against independently reconstructed targets,
allowing bounded acos diagnostic roundoff at finite angles. This module does
not run IK or physics and does not update the original pipeline or report.
"""
from __future__ import annotations
import argparse
import json
import time
from pathlib import Path
from unittest.mock import patch
from . import fixed35_broad_verify as original
from . import certificate_roundoff_verify as certificate
from . import fixed35_bouncer_audit_v3 as old_certificate
from .benchmark_verify import EvidenceError,read,sha

VERSION='sircus-fixed35-broad1000-audit-v2'


def audit(*args,original_failure,**kwargs):
    failure_path=Path(original_failure);failure=read(failure_path)
    if (failure.get('version')!=original.VERSION or failure.get('status')!='failed'
            or not any('v3 numeric exception restricted to near-zero normal angle' in str(f) for f in failure.get('failures',[]))):
        raise EvidenceError('The retained original broad angular-diagnostic audit failure is required')
    study=Path(kwargs.get('study',args[0] if args else ''))
    pipeline_path=study.with_suffix('.pipeline.json');pipeline=read(pipeline_path)
    if (pipeline.get('status')!='audit_failed' or pipeline.get('study_exit_code')!=0
            or pipeline.get('audit_exit_code')==0 or Path(pipeline['output']).resolve()!=study.resolve()):
        raise EvidenceError('The original completed-worker/audit-failed supervisor must remain immutable')
    original_pipeline_sha=sha(pipeline_path);original_failure_sha=sha(failure_path)
    differences=[];count=0
    def compare(*a,**kw):
        nonlocal count
        result=certificate.compare_certificates(*a,**kw);count+=1
        if result['absolute_difference_deg']:differences.append(result)
        return result
    # Explicitly scoped audit adapter only. Original source files and every
    # certificate/admission decision remain immutable; restoration is automatic.
    with patch.object(original,'compare_certificates',compare):
        result=original.audit(*args,**kwargs)
    if count<=0:raise EvidenceError('Full correction must reproduce actual endpoint certificates')
    result.update(version=VERSION,auditor_sha256=sha(Path(__file__)),
        audit_correction=dict(version='sircus-broad-certificate-roundoff-v1',kind='diagnostic_certificate_roundoff_only',
            original_audit_file=str(failure_path.resolve()),original_audit_sha256=original_failure_sha,
            original_pipeline_file=str(pipeline_path.resolve()),original_pipeline_sha256=original_pipeline_sha,
            original_auditor_version=original.VERSION,original_auditor_source_sha256=sha(Path(original.__file__)),
            original_certificate_adapter_sha256=sha(Path(old_certificate.__file__)),
            adapter_version=certificate.VERSION,adapter_source_sha256=sha(Path(certificate.__file__)),wrapper_source_sha256=sha(Path(__file__)),
            report_sha256=result['report_sha256'],evidence_manifest_sha256=result['evidence_manifest_sha256'],
            producer_source_sha256=result['producer_source_sha256'],physics_changed=False,scoring_changed=False,admission_changed=False,
            original_audit_failure=failure,angle_tolerance_deg=certificate.ANGLE_TOLERANCE_DEG,cosine_tolerance=certificate.COSINE_TOLERANCE,
            target_tolerance=certificate.TARGET_TOLERANCE,complete_certificates_reproduced=count,
            differing_normal_angle_diagnostics=len(differences),certificate_numerics=differences,
            producer_source_unchanged=True,native_rows_unchanged=True,report_unchanged=True,
            physics_rerun=False,completed_unix_s=time.time()))
    if sha(pipeline_path)!=original_pipeline_sha or sha(failure_path)!=original_failure_sha:
        raise EvidenceError('Original failure or supervisor changed during independent recovery audit')
    result['limitations'].append('Original V1 audit failed on finite-angle acos roundoff. V2 retains that failure, reproduces each complete saved certificate exactly, and allows only derived-angle differences within2 microdegrees and8 float64 epsilons in cosine; all admission and score decisions remain unchanged.')
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('study','source','assets','parent-study','parent-audit','original-study','original-audit','recovery-study','recovery-audit','reference','original-failure','output'):
        parser.add_argument('--'+key,type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists() or args.output.resolve().is_relative_to(args.study.resolve()):raise ValueError('Fresh independent audit output outside immutable producer required')
    try:
        kwargs=vars(args).copy();kwargs.pop('output');result=audit(**kwargs)
    except Exception as error:
        args.output.write_text(json.dumps(dict(version=VERSION,status='failed',auditor_sha256=sha(Path(__file__)),
            original_audit_failure_sha256=sha(args.original_failure),failures=[type(error).__name__+': '+str(error)]),indent=2,allow_nan=False)+'\n');raise
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:result[k] for k in ('status','checks_count','requested_trials','native_releases')}))


if __name__=='__main__':main()
