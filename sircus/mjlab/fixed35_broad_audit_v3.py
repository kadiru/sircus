"""Full broad audit with explicit certificate and authenticated worker-clock adapters.

V1/V2 failed audits, frozen producer and native evidence remain immutable. V2's
certificate correction is reused exactly. The additional adapter retains every
packet check but binds its clock cap to this study's approved original budget,
not the older coverage study's hardcoded 19,800 seconds. No physics is executed.
"""
from __future__ import annotations
import argparse
import json
import time
from pathlib import Path
from unittest.mock import patch
from . import fixed35_broad_audit_v2 as prior
from . import fixed35_broad_verify as original
from . import packet_timing_verify as packets
from .benchmark_verify import EvidenceError,read,sha

VERSION='sircus-fixed35-broad1000-audit-v3'
CORRECTION_VERSION='sircus-broad-authenticated-worker-clock-v1'


def audit(*args,original_failure,previous_failure,**kwargs):
    failed_path=Path(previous_failure);failed=read(failed_path)
    if (failed.get('version')!=prior.VERSION or failed.get('status')!='failed'
            or failed.get('auditor_sha256')!=sha(Path(prior.__file__))
            or failed.get('original_audit_failure_sha256')!=sha(Path(original_failure))
            or not any('Monotone native batch timings within wall budget' in str(x) for x in failed.get('failures',[]))):
        raise EvidenceError('Retained exact V2 inherited worker-clock audit failure required')
    study=Path(kwargs.get('study',args[0] if args else '')).resolve()
    failed_sha=sha(failed_path);cap_receipt=None;calls=0
    def audit_packets(directory,rows,checks):
        nonlocal cap_receipt,calls
        checks.equal(Path(directory).resolve(),study,'Clock adapter applies only to the authenticated current broad study')
        report=read(study/'report.json');spec=read(study/'contract.json')
        checks.equal((spec['protocol']['worker_limit_s'],spec['protocol']['search_limit_s'],spec['protocol']['audit_reserve_s']),
                     (43200,36000,7200),'Frozen broad contract explicitly approves worker/search/audit clock caps')
        checked=original.audit_budget(study,report,checks)
        cap_receipt=dict(**checked,reported_elapsed_s=report['elapsed_s'],report_sha256=sha(study/'report.json'),
            contract_sha256=sha(study/'contract.json'),budget_file=str(study.with_name(study.name+'-budget.json')),
            original_packet_auditor_sha256=sha(Path(original.geometry.__file__)),obsolete_inherited_cap_s=19800)
        calls+=1
        return packets.audit_request_packets(study,rows,checks,effective_worker_limit_s=checked['effective_worker_limit_s'],reported_elapsed_s=report['elapsed_s'])
    with patch.object(original.geometry,'audit_request_packets',audit_packets):
        result=prior.audit(*args,original_failure=original_failure,**kwargs)
    if calls!=1 or cap_receipt is None:raise EvidenceError('Complete broad audit must invoke the corrected packet audit exactly once')
    if sha(failed_path)!=failed_sha:raise EvidenceError('Retained V2 failure changed during audit')
    result.update(version=VERSION,auditor_sha256=sha(Path(__file__)),
        timing_audit_correction=dict(version=CORRECTION_VERSION,kind='authenticated_original_worker_budget_only',
            previous_audit_file=str(failed_path.resolve()),previous_audit_sha256=failed_sha,previous_audit_failure=failed,
            previous_auditor_version=prior.VERSION,previous_auditor_source_sha256=sha(Path(prior.__file__)),
            adapter_version=packets.VERSION,adapter_source_sha256=sha(Path(packets.__file__)),wrapper_source_sha256=sha(Path(__file__)),
            original_budget=cap_receipt,report_sha256=result['report_sha256'],evidence_manifest_sha256=result['evidence_manifest_sha256'],
            producer_source_sha256=result['producer_source_sha256'],request_packets=result['request_packets'],
            physics_changed=False,scoring_changed=False,admission_changed=False,request_bytes_changed=False,
            timestamps_changed=False,physics_rerun=False,completed_unix_s=time.time()))
    result['limitations'].append('V2 passed certificate/native checks before rejecting packet timings against an inherited19,800-second cap. V3 retains V1/V2 failures and checks every original packet against the independently authenticated original43,200-second maximum/effective acquired budget; all packet identity/order/phase/hash checks remain, with additional finite-clock and report-end checks. No native or scoring changes.')
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('study','source','assets','parent-study','parent-audit','original-study','original-audit','recovery-study','recovery-audit','reference','original-failure','previous-failure','output'):
        parser.add_argument('--'+key,type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists() or args.output.resolve().is_relative_to(args.study.resolve()):raise ValueError('Fresh independent audit output outside immutable producer required')
    try:
        kwargs=vars(args).copy();kwargs.pop('output');result=audit(**kwargs)
    except Exception as error:
        args.output.write_text(json.dumps(dict(version=VERSION,status='failed',auditor_sha256=sha(Path(__file__)),
            original_audit_failure_sha256=sha(args.original_failure),previous_audit_failure_sha256=sha(args.previous_failure),
            failures=[type(error).__name__+': '+str(error)]),indent=2,allow_nan=False)+'\n');raise
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:result[k] for k in ('status','checks_count','requested_trials','native_releases')}))


if __name__=='__main__':main()
