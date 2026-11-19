"""Explicit future evaluation entry and a separate CPU-only audit route.

Importing this module is inert. Production startup requires the fixed native
flags and watchdog session cookie; the host supervisor separately binds operator
approval. Tests replace every process, model and native component with mocks.
"""

import argparse
import copy
import os
from pathlib import Path
import re
import subprocess
import sys
import time

from candidate_continuation_application import read,sha,require
from candidate_continuation_journal import write_new_json
from candidate_continuation_ledger import canonical_hash

AUDIT_SCHEMA='sircus-v4-test-combined-audit-v1'
REQUEST_SCHEMA='sircus-v4-test-audit-request-v1'


def audit_application(output,inputs,session,*,expected_request_sha256,checks=None):
    """Independently reread exact frozen models and every new native TEST trace."""
    require(os.environ.get('CUDA_VISIBLE_DEVICES')=='','Independent evaluation auditor must hide CUDA')
    output=Path(output);request_path=output/'audit-request.json'
    require(request_path.is_file() and not request_path.is_symlink() and sha(request_path)==expected_request_sha256,
        'Externally bound evaluation audit request changed')
    request=read(request_path)
    require(request.get('schema')==REQUEST_SCHEMA and request.get('session')==session
        and request.get('fingerprint')==canonical_hash({k:v for k,v in request.items() if k!='fingerprint'})
        and request.get('plan_fingerprint')==inputs['plan']['fingerprint']
        and request.get('source_sha256')==inputs['execution']['source_sha256']
        and request.get('manifest_sha256')==inputs['evaluation_manifest_sha256'],
        'Audit request session/source/plan identity changed')
    roles=dict(evaluation_result='evaluation-result.json',native_binding='native-binding.json',
        model_receipt='model-check.json',startup_receipt='native-setup/startup/receipt.json',plan='evaluation-plan.json')
    before={}
    for role,name in roles.items():
        path=output/name
        require(path.is_file() and not path.is_symlink() and sha(path)==request.get(role+'_sha256'),
            'Audit input bytes changed: '+role)
        before[role]=sha(path)
    require(read(output/'evaluation-plan.json')==inputs['plan'],'Saved TEST evaluation plan changed')
    summary=read(output/'evaluation-result.json');binding=read(output/'native-binding.json')
    require(summary.get('synthetic') is inputs['plan']['synthetic'] and summary.get('original_assessment_completed') is False,
        'Audit provenance differs or old assessment was promoted')
    if checks is None:
        from candidate_v4_model_check import check_models,validate_model_receipt
        from candidate_v4_native_audit import audit_native_evaluation
        checks=dict(model=check_models,model_validate=validate_model_receipt,native=audit_native_evaluation)
    else:
        require(inputs['plan']['synthetic'] is True,'Real audit cannot replace independent readers')
    from candidate_v4_inputs import model_request
    began=time.monotonic();cpu=time.process_time()
    initial=checks['model_validate'](read(output/'model-check.json'),inputs['plan'])
    # A new-only CPU receipt is produced independently after the App closes.
    # This performs no action selection and reads no assessment outcomes.
    reconstructed=checks['model'](model_request(inputs,output/'model-check-audit.json'))
    checks['model_validate'](reconstructed,inputs['plan'])
    signature=lambda value:{k:v for k,v in value.items() if k not in ('request_sha256','fingerprint')}
    require(signature(initial)==signature(reconstructed),'Frozen model reconstruction changed during native evaluation')
    require((output/'model-check-audit.json').is_file() and read(output/'model-check-audit.json')==reconstructed,
        'Reconstructed model proof differs from durable CPU receipt')
    p=inputs['paths'];e=inputs['evaluation_paths']
    native=checks['native'](output,plan=inputs['plan'],summary=summary,
        expected_head_sha256=summary['journal_head_sha256'],expected_result_sha256=before['evaluation_result'],
        model_receipt_path=output/'model-check.json',expected_model_receipt_sha256=before['model_receipt'],
        native_binding=binding,scene=inputs['values']['scene'],original_cells=inputs['train']['cells'],
        assets_paths={'Arm10':p['dropper_urdf'],'Arm14':p['board_urdf']},poses_path=p['poses'],camera_path=p['camera'],
        parent_checkpoint_path=e['parent_checkpoint'],final_checkpoint_path=e['final_checkpoint'],
        startup_directory=output/'native-setup',startup_cells=inputs['values']['source_contract']['cells'],
        synthetic=inputs['plan']['synthetic'])
    expected='passed_synthetic' if inputs['plan']['synthetic'] else 'passed'
    require(native.get('status')==expected and native.get('all_components_passed') is True
        and native.get('synthetic') is inputs['plan']['synthetic']
        and native.get('plan_fingerprint')==inputs['plan']['fingerprint']
        and native.get('journal_head_sha256')==summary['journal_head_sha256']
        and native.get('fingerprint')==canonical_hash({k:v for k,v in native.items() if k!='fingerprint'}),
        'Independent TEST model/startup/physics audit did not pass')
    require(before=={role:sha(output/name) for role,name in roles.items()},'Audit modified native source evidence')
    result=dict(schema=AUDIT_SCHEMA,status=expected,synthetic=inputs['plan']['synthetic'],all_components_passed=True,
        full_native_audit=not inputs['plan']['synthetic'],plan_fingerprint=inputs['plan']['fingerprint'],
        journal_head_sha256=summary['journal_head_sha256'],session=session,
        audit_request_sha256=expected_request_sha256,source_sha256=inputs['execution']['source_sha256'],
        native=native,model_recheck_sha256=sha(output/'model-check-audit.json'),
        original_assessment_completed=False,model_promoted=False,execution_authorized=False,release_authorized=False,
        optimizer_steps=0,training_requests=0,
        cost=dict(wall_s=time.monotonic()-began,process_cpu_s=time.process_time()-cpu,device='cpu',cuda_visible_devices='',
            scope='Independent CPU reader only; contained in application audit phase, not additive to it. Model-reader child CPU is reported separately by the application aggregate.'),
        scope='Separate 210-request untouched TEST comparison. Original partial assessment stays incomplete. Synthetic results certify software only.')
    result['fingerprint']=canonical_hash(result)
    write_new_json(output/'evaluation-audit.json',result)
    return result


def audit_callback(inputs,session,*,run=subprocess.run):
    """Create a fixed bounded CPU child invocation; construction performs no work."""
    def audit(output,plan,summary):
        output=Path(output)
        require(plan==inputs['plan'],'Audit callback plan changed')
        require(sha(output/'model-check.json')==inputs.get('model_receipt_sha256'),
            'Durable model proof changed after its pre-App freeze')
        request=dict(schema=REQUEST_SCHEMA,session=session,plan_fingerprint=plan['fingerprint'],
            source_sha256=inputs['execution']['source_sha256'],manifest_sha256=inputs['evaluation_manifest_sha256'],
            **{role+'_sha256':sha(output/name) for role,name in dict(evaluation_result='evaluation-result.json',
                native_binding='native-binding.json',model_receipt='model-check.json',
                startup_receipt='native-setup/startup/receipt.json',plan='evaluation-plan.json').items()})
        request['fingerprint']=canonical_hash(request)
        path=output/'audit-request.json';write_new_json(path,request)
        command=[sys.executable,str(Path(__file__).resolve()),'--audit-only',
            '--manifest',str(inputs['evaluation_manifest_path']),'--expected-manifest-sha256',inputs['evaluation_manifest_sha256'],
            '--output',str(output),'--session',session,'--expected-audit-request-sha256',sha(path)]
        run(command,check=True,timeout=600,env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1'))
        return read(output/'evaluation-audit.json')
    return audit


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--expected-manifest-sha256',required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--session',required=True)
    p.add_argument('--execute-approved',action='store_true')
    p.add_argument('--audit-only',action='store_true')
    p.add_argument('--expected-audit-request-sha256')
    p.add_argument('--headless',action='store_true')
    p.add_argument('--enable_cameras',action='store_true')
    p.add_argument('--device',default='cuda:0')
    return p


def main(argv=None):
    args=parser().parse_args(argv)
    require(re.fullmatch(r'[0-9a-f]{32}',args.session or '') is not None,'Exact watchdog session required')
    require(args.execute_approved != args.audit_only,'Choose explicit future execution or CPU-only audit')
    require(args.output.is_absolute() and args.output.is_dir() and not args.output.is_symlink(),
        'Supervisor-owned new output required')
    if args.audit_only:
        require(os.environ.get('CUDA_VISIBLE_DEVICES')=='','Independent audit must hide CUDA')
        require(args.expected_audit_request_sha256 is not None,'External audit-request SHA required')
    else:
        from candidate_v4_watchdog import COOKIE
        require(args.headless and args.enable_cameras and args.device=='cuda:0','Fixed headless 4090 runtime required')
        require(os.environ.get(COOKIE)==args.session,'Watchdog-owned session cookie required')
    from candidate_v4_inputs import production_inputs,frozen_verifier,model_request
    inputs=production_inputs(args.manifest,args.expected_manifest_sha256)
    if args.audit_only:
        return audit_application(args.output,inputs,args.session,expected_request_sha256=args.expected_audit_request_sha256)
    require(inputs['plan']['synthetic'] is False,'Production entry cannot disguise synthetic input as native')
    from candidate_v4_native import ProductionComponents
    from candidate_v4_application import run_application
    from candidate_v4_watchdog import ProgressWriter
    inputs['verify_frozen']=frozen_verifier(inputs)
    inputs['model_request']=model_request(inputs,args.output/'model-check.json')
    inputs['audit']=audit_callback(inputs,args.session)
    return run_application(args.output,inputs['plan'],
        ProductionComponents(inputs,args.output,args.session,approved=True,synthetic=False),
        ProgressWriter(args.output,args.session),approved=True)


if __name__=='__main__':main()
