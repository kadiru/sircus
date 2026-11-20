"""Explicit CPU-only entry for the separately approved saved-data audit.

No simulation, trainer, optimizer step, native retry or original receipt write.
The immutable producer helpers are imported from their pinned source directory;
this entry and the recovery modules live outside that directory.
"""
import argparse
import hashlib
import importlib.abc
import json
import os
from pathlib import Path
import sys
import time


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024**2), b''):
            result.update(block)
    return result.hexdigest()


def read(path):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, 'Duplicate control key')
            result[key] = value
        return result
    raw = Path(path).read_bytes()
    require(len(raw) <= 64*1024**2, 'Oversized recovery control')
    return json.loads(raw, object_pairs_hook=pairs,
        parse_constant=lambda _: (_ for _ in ()).throw(ValueError('Nonfinite recovery control')))


def canonical(path):
    path = Path(path)
    require(path.is_absolute() and str(path) == str(path.resolve())
        and not any(p.is_symlink() for p in (path, *path.parents)), 'Noncanonical recovery path')
    return path


def load_control(path, expected_sha256, producer_source):
    path, producer_source = canonical(path), canonical(producer_source)
    require(sha(path) == expected_sha256, 'Recovery request bytes changed')
    request = read(path)
    unsigned = {k:v for k,v in request.items() if k != 'fingerprint'}
    fingerprint = hashlib.sha256(json.dumps(unsigned, sort_keys=True,
        separators=(',', ':'), allow_nan=False).encode()).hexdigest()
    require(request.get('schema') == 'sircus-location-comparison-recovery-request-v1'
        and request.get('fingerprint') == fingerprint, 'Recovery request fingerprint changed')
    require(request.get('producer_source') == str(producer_source), 'Wrong immutable producer source')
    require(request.get('execute_approved') is True and request.get('native_application_starts') == 0
        and request.get('optimizer_steps') == 0 and request.get('new_shots') == 0,
        'Saved-data-only approval required')
    require(request.get('recovery_session_id') != request.get('native_session_id'),
        'Recovery requires a distinct process session')
    output = canonical(request['audit_output'])
    own_source = Path(__file__).resolve().parent
    for protected in (request['native_output'], str(producer_source), str(path.parent), str(own_source)):
        protected = canonical(protected)
        require(output != protected and not output.is_relative_to(protected)
            and not protected.is_relative_to(output), 'Audit output overlaps protected evidence')
    require(own_source != producer_source, 'Recovery code must be separate from producer source')
    required = {'location_comparison_recovery.py', 'location_comparison_recovery_verify.py',
                'location_comparison_recovery_watchdog.py'}
    sources = request.get('recovery_source_sha256', {})
    require(set(sources) == required, 'Incomplete recovery source binding')
    for name, checksum in sources.items():
        require(sha(canonical(own_source/name)) == checksum, 'Recovery source bytes changed: '+name)
    manifest_path = canonical(request['manifest_path'])
    require(sha(manifest_path) == request['manifest_sha256'], 'Original manifest changed')
    manifest = read(manifest_path)
    require(manifest.get('fingerprint') == request['manifest_fingerprint']
        and manifest.get('output') == request['native_output']
        and manifest.get('source_commit') == request['source_commit'], 'Original manifest identity changed')
    for field in ('inventory', 'retirement_proof'):
        require(sha(canonical(request[field+'_path'])) == request[field+'_sha256'],
                'Changed sealed '+field)
    require(not Path(manifest['audit_output']).exists(), 'Original audit destination unexpectedly exists')
    return request, manifest


def verify_producer_source(manifest, source):
    """Use only the standard library before any original helper can execute."""
    source = canonical(source)
    declared = manifest.get('source_sha256', {})
    require(type(declared) is dict and bool(declared)
        and {p.name for p in source.glob('*.py')} == set(declared), 'Original source membership changed')
    for name, expected in declared.items():
        require(sha(canonical(source/name)) == expected, 'Original source changed: '+name)


def verify_recovery_membership(request):
    require({p.name for p in Path(__file__).resolve().parent.glob('*.py')}
        == set(request['recovery_source_sha256']), 'Unexpected helper in isolated recovery source')


class NoSimulator(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'omni', 'isaacsim', 'isaaclab'}:
            raise RuntimeError('Simulator import forbidden in saved-data audit: '+fullname)
        return None


def install_cpu_boundary():
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == ''
        and os.environ.get('NVIDIA_VISIBLE_DEVICES') == 'void'
        and not list(Path('/dev').glob('nvidia*')), 'CPU-only environment and devices required')
    require(not any(k.split('.')[0] in {'omni','isaacsim','isaaclab'} for k in sys.modules),
            'Simulator module already loaded')
    sys.meta_path.insert(0, NoSimulator())


def verify_runtime(manifest):
    import numpy
    import scipy
    import torch
    require(dict(python=sys.version.split()[0], numpy=numpy.__version__, scipy=scipy.__version__,
                 torch=torch.__version__) == manifest['runtime'], 'Original numerical runtime changed')
    require(not torch.cuda.is_initialized() and torch.cuda.device_count() == 0, 'Unexpected CUDA state')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request', type=Path, required=True)
    parser.add_argument('--expected-request-sha256', required=True)
    parser.add_argument('--producer-source', type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--preflight-only', action='store_true')
    mode.add_argument('--execute-approved', action='store_true')
    mode.add_argument('--audit-child', action='store_true')
    args = parser.parse_args(argv)
    request, manifest = load_control(args.request, args.expected_request_sha256, args.producer_source)
    verify_recovery_membership(request)
    verify_producer_source(manifest, args.producer_source)
    install_cpu_boundary()
    sys.path.insert(0, str(args.producer_source))
    from candidate_continuation_journal import write_new_json
    from location_comparison_verify import read as original_read, verify_inputs
    from location_comparison_recovery_verify import (
        audit_recovery, verify_recovery_exit, verify_scientific_body, verify_inventory, verify_journal)
    from location_comparison_recovery_watchdog import supervise_recovery, validate_child_owner, LinuxRecoveryOps, LIMITS
    plan = original_read(manifest['files']['comparison']['path'])
    require(plan['fingerprint'] == request['comparison_fingerprint'], 'Original comparison changed')
    output = Path(request['audit_output'])
    last_integrity = -float('inf')
    def integrity(force=False):
        nonlocal last_integrity
        load_control(args.request, args.expected_request_sha256, args.producer_source)
        now = time.monotonic()
        if force or now-last_integrity >= 30:
            for item in manifest['files'].values():
                require(sha(canonical(item['path'])) == item['sha256'], 'Original input bytes changed')
            verify_producer_source(manifest, args.producer_source)
            last_integrity = now
        return True
    integrity(force=True)
    if args.preflight_only:
        require(not output.exists(), 'Audit attempt already exists')
        ops = LinuxRecoveryOps()
        require(ops.free_bytes(Path(request['storage_root'])) >= LIMITS['min_free_bytes'],
                'Recovery storage floor reached')
        ops.prepare()
        verify_runtime(manifest)
        verify_inputs(manifest, plan)
        verify_scientific_body()
        trace = original_read(Path(request['native_output'])/'comparison-trace.json')
        proof, inventory = verify_recovery_exit(Path(request['native_output']), manifest, plan, trace, request)
        sealed = verify_inventory(Path(request['native_output']), inventory)
        journal = verify_journal(Path(request['native_output']), trace)
        integrity(force=True)
        print(json.dumps(dict(status='passed', scope='CPU input/runtime preflight only; no scientific audit',
            request_sha256=args.expected_request_sha256, simulator_starts=0, optimizer_steps=0, new_shots=0,
            sealed_inventory=sealed, journal=journal, resource_controls=ops.resources(),
            original_audit_passed=False, clean_shutdown=False)))
        return 0
    if args.audit_child:
        validate_child_owner(output, request, request['recovery_session_id'])
        try:
            verify_runtime(manifest)
            trace = original_read(Path(request['native_output'])/'comparison-trace.json')
            report = audit_recovery(Path(request['native_output']), manifest, plan, trace, request)
            integrity(force=True)
            validate_child_owner(output, request, request['recovery_session_id'])
            write_new_json(output/'recovery-audit.json', report)
            return 0
        except BaseException as error:
            write_new_json(output/'recovery-audit-failure.json', dict(status='audit_failed_no_retry',
                request_fingerprint=request['fingerprint'], error_type=type(error).__name__,
                reason=str(error)[:4000], original_audit_passed=False, clean_shutdown=False))
            raise
    require(not output.exists(), 'Audit attempt already exists; no retry')
    command = [sys.executable, str(Path(__file__).resolve()), '--request', str(args.request),
        '--expected-request-sha256', args.expected_request_sha256,
        '--producer-source', str(args.producer_source), '--audit-child']
    def verify_report(destination, *unused):
        integrity(force=True)
        report = read(Path(destination)/'recovery-audit.json')
        require(report.get('recovery_request_fingerprint') == request['fingerprint']
            and report.get('original_audit_passed') is False and report.get('clean_shutdown') is False,
            'Recovery report omitted original failure')
        sealed = verify_inventory(Path(request['native_output']), read(request['inventory_path']))
        require(report.get('recovery_proof', {}).get('inventory') == sealed,
                'Post-exit sealed evidence differs from recovery report')
        return True
    result = supervise_recovery(command, output=output, request=request,
        session_id=request['recovery_session_id'], integrity_check=integrity,
        verify_report=verify_report, storage_root=Path(request['storage_root']), execute_approved=True)
    print(json.dumps(result, sort_keys=True), flush=True)
    return 0 if result.get('independent_report_verified') is True else 1


if __name__ == '__main__':
    raise SystemExit(main())
