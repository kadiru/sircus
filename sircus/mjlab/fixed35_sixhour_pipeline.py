"""Queue one follow-up after the fully audited parent; cap acquired work at6h."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

from .coverage_recovery_pipeline import supervise
from .gpu_slot import GPU_UUID
from .sweep_worker import atomic

INPUT_KEYS = ('parent-study', 'parent-audit', 'original-study', 'original-audit',
              'recovery-study', 'recovery-audit', 'reference', 'assets', 'broad-study', 'broad-audit')
WORKER_LIMIT_S = 18000
SEARCH_LIMIT_S = 16200
AUDIT_LIMIT_S = 3600
ACQUIRED_LIMIT_S = 21600
QUEUE_LIMIT_S = 18000
PARENT_VERSION = 'sircus-fixed35-broad1000-v1'
PARENT_AUDIT_VERSION = 'sircus-fixed35-broad1000-audit-v1'
CORRECTED_PARENT_AUDIT_VERSION = 'sircus-fixed35-broad1000-audit-v2'
CORRECTION_VERSION = 'sircus-broad-certificate-roundoff-v1'
TIMING_PARENT_AUDIT_VERSION = 'sircus-fixed35-broad1000-audit-v3'
CORRECTED_PARENT_VERSIONS = (CORRECTED_PARENT_AUDIT_VERSION, TIMING_PARENT_AUDIT_VERSION)
PARENT_QUOTAS = dict(zip(('fixed35-dev-grid-03-04', 'fixed35-dev-grid-03-05', 'fixed35-dev-grid-04-06'), (334, 333, 333)))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def exclusive_json(path, value):
    """An existing claim always blocks retry, including after an interrupted run."""
    with Path(path).open('x') as handle:
        handle.write(json.dumps(value, sort_keys=True, indent=2, allow_nan=False)+'\n')
        handle.flush()
        os.fsync(handle.fileno())


def corrected_parent_binding(study, audit_path, spec, audit, pipeline):
    """Explicit audit-only recovery; the failed supervisor is never rewritten."""
    correction = audit.get('audit_correction', {})
    pipeline_path = study.with_suffix('.pipeline.json')
    original_path = study.with_name(study.name+'-audit.json')
    source = Path(__file__).parent
    identities = {
        'original_auditor_source_sha256': source/'fixed35_broad_verify.py',
        'adapter_source_sha256': source/'certificate_roundoff_verify.py',
        'wrapper_source_sha256': source/'fixed35_broad_audit_v2.py',
    }
    if (audit.get('version') not in CORRECTED_PARENT_VERSIONS
            or correction.get('version') != CORRECTION_VERSION
            or correction.get('kind') != 'diagnostic_certificate_roundoff_only'
            or correction.get('original_auditor_version') != PARENT_AUDIT_VERSION
            or any(correction.get(k) is not False for k in ('physics_changed', 'scoring_changed', 'admission_changed'))
            or (pipeline.get('status'), pipeline.get('study_exit_code'), pipeline.get('audit_exit_code')) != ('audit_failed', 0, 1)
            or Path(correction.get('original_pipeline_file', '')).resolve() != pipeline_path.resolve()
            or Path(correction.get('original_audit_file', '')).resolve() != original_path.resolve()
            or audit_path.resolve() == original_path.resolve()
            or correction.get('original_pipeline_sha256') != sha(pipeline_path)
            or correction.get('original_audit_sha256') != sha(original_path)):
        raise ValueError('Corrected parent audit must preserve and bind the original failed audit and pipeline')
    if (type(audit.get('checks_count')) is not int or audit['checks_count'] <= 0
            or audit['checks_count'] != len(audit.get('checks', []))
            or not all(c.get('passed') is True for c in audit['checks'])):
        raise ValueError('Corrected parent audit must retain every full independent check')
    failure = read(original_path)
    if (failure.get('version') != PARENT_AUDIT_VERSION or failure.get('status') != 'failed'
            or not isinstance(failure.get('failures'), list) or not failure['failures']
            or not any('v3 numeric exception restricted to near-zero normal angle' in str(f) for f in failure['failures'])
            or correction.get('original_audit_failure') != failure):
        raise ValueError('Original parent audit is not the recorded V1 failure')
    if (correction.get('report_sha256') != sha(study/'report.json')
            or correction.get('evidence_manifest_sha256') != sha(study/'evidence-manifest.json')
            or correction.get('producer_source_sha256') != spec.get('source_sha256')
            or correction.get('original_auditor_source_sha256') != spec['source_sha256'].get('sircus/mjlab/fixed35_broad_verify.py')
            or any(correction.get(k) != sha(path) for k, path in identities.items())):
        raise ValueError('Corrected parent audit must bind unchanged evidence and exact known auditor sources')
    completed = correction.get('completed_unix_s')
    if (not isinstance(completed, (int, float)) or not math.isfinite(completed)
            or not isinstance(pipeline.get('finished_unix_s'), (int, float))
            or not 0 < pipeline['finished_unix_s'] <= completed <= time.time()):
        raise ValueError('Corrected full audit must finish after the failed pipeline and before acceptance')
    receipt = dict(basis='independent_corrected_parent_audit', correction_version=CORRECTION_VERSION,
        corrected_audit_version=audit['version'],
        correction_sha256=digest(correction), original_audit_sha256=sha(original_path),
        original_pipeline_sha256=sha(pipeline_path), completed_unix_s=completed,
        auditor_sources_sha256={k:sha(path) for k,path in identities.items()})
    if audit['version'] == TIMING_PARENT_AUDIT_VERSION:
        receipt['timing_correction'] = corrected_timing_binding(study, spec, audit, pipeline)
        receipt['completed_unix_s'] = receipt['timing_correction']['completed_unix_s']
    return receipt


def corrected_timing_binding(study, spec, audit, pipeline):
    """The second correction authenticates this study's original clock only."""
    c = audit['timing_audit_correction'];source = Path(__file__).parent
    previous = study.with_name(study.name+'-audit-v2.json');failure = read(previous)
    budgetpath = study.with_name(study.name+'-budget.json');budget = read(budgetpath)
    report = read(study/'report.json')
    identities = {'previous_auditor_source_sha256':source/'fixed35_broad_audit_v2.py',
        'adapter_source_sha256':source/'packet_timing_verify.py','wrapper_source_sha256':source/'fixed35_broad_audit_v3.py'}
    if (c.get('version') != 'sircus-broad-authenticated-worker-clock-v1'
            or c.get('kind') != 'authenticated_original_worker_budget_only'
            or c.get('previous_auditor_version') != CORRECTED_PARENT_AUDIT_VERSION
            or c.get('adapter_version') != 'sircus-authenticated-packet-worker-clock-audit-v1'
            or Path(c.get('previous_audit_file','')).resolve() != previous.resolve()
            or c.get('previous_audit_sha256') != sha(previous) or c.get('previous_audit_failure') != failure
            or failure.get('version') != CORRECTED_PARENT_AUDIT_VERSION or failure.get('status') != 'failed'
            or failure.get('auditor_sha256') != sha(identities['previous_auditor_source_sha256'])
            or failure.get('original_audit_failure_sha256') != audit['audit_correction']['original_audit_sha256']
            or not any('Monotone native batch timings within wall budget' in str(x) for x in failure.get('failures',[]))
            or any(c.get(k) != sha(path) for k,path in identities.items())
            or any(c.get(k) is not False for k in ('physics_changed','scoring_changed','admission_changed','request_bytes_changed','timestamps_changed','physics_rerun'))):
        raise ValueError('Timing correction must bind exact retained V2 clock failure and known sources')
    expected = dict(worker_receipt_sha256=sha(budgetpath),effective_worker_limit_s=budget['effective_worker_limit_s'],
        global_limit_s=pipeline['global_limit_s'],reported_elapsed_s=report['elapsed_s'],
        report_sha256=sha(study/'report.json'),contract_sha256=sha(study/'contract.json'),budget_file=str(budgetpath.resolve()),
        original_packet_auditor_sha256=sha(source/'coverage_training_verify.py'),obsolete_inherited_cap_s=19800)
    if (c.get('original_budget') != expected
            or c.get('report_sha256') != sha(study/'report.json')
            or c.get('evidence_manifest_sha256') != sha(study/'evidence-manifest.json')
            or c.get('producer_source_sha256') != spec['source_sha256']
            or c.get('request_packets') != audit['request_packets']
            or tuple(spec['protocol'].get(k) for k in ('worker_limit_s','search_limit_s','audit_reserve_s')) != (43200,36000,7200)
            or (budget['limit_s'],budget['audit_reserve_s']) != (43200,7200)
            or budget['gpu_uuid'] != GPU_UUID
            or not 0 < pipeline['global_limit_s'] <= 51300
            or abs(pipeline['global_deadline_unix_s']-pipeline['started_unix_s']-pipeline['global_limit_s']) > .05
            or budget['global_deadline_unix_s'] != pipeline['global_deadline_unix_s']
            or pipeline['started_unix_s'] > budget['acquired_unix_s']
            or abs(budget['effective_worker_limit_s']-min(43200,budget['global_deadline_unix_s']-budget['acquired_unix_s']-7200)) > .1
            or not 60 < budget['effective_worker_limit_s'] <= 43200
            or not 0 <= report['elapsed_s'] < budget['effective_worker_limit_s']
            or any(not isinstance(v,(int,float)) or not math.isfinite(v) for v in (report['elapsed_s'],budget['effective_worker_limit_s'],
                budget['global_deadline_unix_s'],budget['acquired_unix_s'],pipeline['global_limit_s'],
                pipeline['global_deadline_unix_s'],pipeline['started_unix_s']))
            or any(report['deadlines'].get(k) != budget[k] for k in ('effective_worker_limit_s','global_deadline_unix_s'))
            or (report['deadlines']['worker_limit_s'],report['deadlines']['search_limit_s']) != (43200,36000)):
        raise ValueError('Timing correction must retain original authenticated worker budget and report')
    completed = c.get('completed_unix_s')
    if (not isinstance(completed,(int,float)) or not math.isfinite(completed)
            or not audit['audit_correction']['completed_unix_s'] <= completed <= time.time()):
        raise ValueError('Complete corrected packet audit must precede acceptance')
    return dict(version=c['version'],correction_sha256=digest(c),previous_audit_sha256=sha(previous),
        worker_receipt_sha256=sha(budgetpath),completed_unix_s=completed,
        auditor_sources_sha256={k:sha(path) for k,path in identities.items()})


def parent_ready(study, audit_path, expected_contract_sha):
    """None means still running; a final failed/incomplete parent refuses launch."""
    study, audit_path = Path(study), Path(audit_path)
    contract_path = study/'contract.json'
    if sha(contract_path) != expected_contract_sha:
        raise ValueError('Parent immutable contract changed while queued')
    spec = read(contract_path)
    unsigned = dict(spec)
    fingerprint = unsigned.pop('fingerprint')
    if spec['version'] != PARENT_VERSION or digest(unsigned) != fingerprint:
        raise ValueError('Parent is not the expected intact broad-study contract')
    pipeline_path = study.with_suffix('.pipeline.json')
    if not pipeline_path.exists():
        return None
    pipeline = read(pipeline_path)
    status = pipeline.get('status')
    corrected = status == 'audit_failed' and audit_path.exists() and read(audit_path).get('version') in CORRECTED_PARENT_VERSIONS
    if status in ('failed', 'audit_failed', 'incomplete_audited') and not corrected:
        raise ValueError('Parent pipeline ended without fully audited1000-drop completion: '+status)
    if status != 'completed' and not corrected:
        return None
    report_path, manifest_path = study/'report.json', study/'evidence-manifest.json'
    report, audit, manifest = read(report_path), read(audit_path), read(manifest_path)
    report_sha, audit_sha = sha(report_path), sha(audit_path)
    if (report.get('version') != PARENT_VERSION or report.get('status') != 'completed'
            or report.get('phase') != 'completed' or report.get('fresh_search_releases') != PARENT_QUOTAS
            or report.get('gate', {}).get('search_quota_completed') is not True):
        raise ValueError('Parent report has not completed all1000 fresh search drops')
    expected_audit_version = audit.get('version') if corrected else PARENT_AUDIT_VERSION
    if (audit.get('version') != expected_audit_version or audit.get('status') != 'passed'
            or audit.get('failures') != [] or audit.get('full_checks_verified') is not True
            or audit.get('native_checks_verified') is not True
            or audit.get('report_sha256') != report_sha
            or audit.get('contract_fingerprint') != fingerprint
            or report.get('contract_fingerprint') != fingerprint
            or audit.get('evidence_manifest_sha256') != sha(manifest_path)
            or report.get('evidence_sha256') != manifest
            or audit.get('producer_source_sha256') != spec.get('source_sha256')):
        raise ValueError('Parent full audit does not bind final report, source and evidence')
    acceptance = corrected_parent_binding(study, audit_path, spec, audit, pipeline) if corrected else None
    if ((not corrected and (pipeline.get('report_sha256') != report_sha or pipeline.get('audit_sha256') != audit_sha))
            or Path(pipeline['output']).resolve() != study.resolve()
            or not isinstance(pipeline.get('finished_unix_s'), (int, float))):
        raise ValueError('Parent completed pipeline does not bind these final files')
    return dict(study=str(study.resolve()), audit_path=str(audit_path.resolve()),
        contract_sha256=expected_contract_sha, contract_fingerprint=fingerprint,
        report_sha256=report_sha, audit_sha256=audit_sha, evidence_manifest_sha256=sha(manifest_path),
        producer_source_manifest_sha256=digest(spec['source_sha256']),
        pipeline_file=str(pipeline_path.resolve()), pipeline_sha256=sha(pipeline_path),
        completed_search_releases=PARENT_QUOTAS, **({'parent_acceptance':acceptance} if corrected else {}))


def wait_for_parent(study, audit, expected_contract, deadline, state, statepath):
    while True:
        if time.monotonic() >= deadline:
            raise TimeoutError('Parent/GPU queue deadline reached before launch')
        receipt = parent_ready(study, audit, expected_contract)
        if receipt is not None:
            return receipt
        state.update(status='waiting_for_parent', parent_wait_elapsed_s=time.monotonic()-state['started_monotonic_s'])
        atomic(statepath, state)
        time.sleep(min(10., max(0., deadline-time.monotonic())))


def terminal_status(output, audit_path):
    report_path = output/'report.json'
    report, audit = read(report_path), read(audit_path)
    if (audit.get('status') != 'passed' or audit.get('failures') != []
            or audit.get('report_sha256') != sha(report_path)):
        raise ValueError('Follow-up audit does not bind the finished report')
    if report.get('status') not in ('completed', 'incomplete'):
        raise ValueError('Follow-up worker is not honestly finished')
    return ('completed' if report['status'] == 'completed' else 'incomplete_audited'), report, audit


def acquired_audit_time(budgetpath, *, outer_deadline, now):
    budget = read(budgetpath)
    acquired = budget['acquired_unix_s']
    deadline = budget['acquired_deadline_unix_s']
    expected = min(acquired+ACQUIRED_LIMIT_S, outer_deadline)
    if (budget.get('acquired_total_limit_s') != ACQUIRED_LIMIT_S
            or budget.get('gpu_uuid') != GPU_UUID or abs(deadline-expected) > .001
            or not 0 < acquired <= now
            or acquired > outer_deadline-ACQUIRED_LIMIT_S+.1
            or not 0 < budget['effective_worker_limit_s'] <= WORKER_LIMIT_S
            or budget.get('global_deadline_unix_s') != outer_deadline):
        raise ValueError('Worker receipt does not bind the six-hour acquired budget')
    remaining = min(AUDIT_LIMIT_S, deadline-now-5)
    if remaining <= 0:
        raise TimeoutError('Six-hour acquired deadline leaves no CPU audit time')
    return remaining, budget


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in (*INPUT_KEYS, 'output', 'gpu-lock'):
        parser.add_argument('--'+key, type=Path, required=True)
    parser.add_argument('--wall-time-limit', type=float, default=WORKER_LIMIT_S)
    parser.add_argument('--search-limit', type=float, default=SEARCH_LIMIT_S)
    parser.add_argument('--queue-timeout', type=float, default=QUEUE_LIMIT_S)
    args = parser.parse_args()
    if not (60 < args.wall_time_limit <= WORKER_LIMIT_S and 0 < args.search_limit <= SEARCH_LIMIT_S
            and args.search_limit < args.wall_time_limit and 0 < args.queue_timeout <= QUEUE_LIMIT_S):
        raise ValueError('Finite six-hour worker, search and separate queue caps required')
    source = Path(__file__).resolve().parents[2]
    for module in ('fixed35_sixhour_study.py', 'fixed35_sixhour_verify.py'):
        if not (source/'sircus/mjlab'/module).is_file():
            raise ValueError('Frozen worker and independent auditor must exist before queueing')
    output = args.output
    statepath, logpath = output.with_suffix('.pipeline.json'), output.with_suffix('.pipeline.log')
    budgetpath, auditpath = (output.with_name(output.name+suffix) for suffix in ('-budget.json', '-audit.json'))
    queuepath, launchpath = (output.with_name(output.name+suffix) for suffix in ('-queue.json', '-launch.json'))
    if output.exists() or any(p.exists() for p in (statepath, logpath, budgetpath, auditpath, queuepath, launchpath)):
        raise ValueError('Fresh exclusive queue/output required; no duplicate launch or in-place resume')
    contract_sha = sha(args.broad_study/'contract.json')
    started, unix = time.monotonic(), time.time()
    queue_deadline, queue_unix = started+args.queue_timeout, unix+args.queue_timeout
    outer_deadline = queue_unix+ACQUIRED_LIMIT_S
    output.parent.mkdir(parents=True, exist_ok=True)
    state = dict(status='queued', source=str(source), output=str(output), pipeline_pid=os.getpid(),
        started_monotonic_s=started, started_unix_s=unix, queue_limit_s=args.queue_timeout,
        queue_deadline_unix_s=queue_unix, global_deadline_unix_s=outer_deadline,
        global_limit_s=args.queue_timeout+ACQUIRED_LIMIT_S, acquired_total_limit_s=ACQUIRED_LIMIT_S,
        worker_limit_s=args.wall_time_limit, search_limit_s=args.search_limit, audit_limit_s=AUDIT_LIMIT_S,
        parent_contract_sha256=contract_sha, ui_publication=False, training=False, resume=False)
    # This is the atomic duplicate-launch claim, written before any waiting/child.
    exclusive_json(queuepath, dict(state, arguments={k:str(getattr(args, k.replace('-', '_'))) for k in INPUT_KEYS},
                                   pipeline_source_sha256=sha(Path(__file__))))
    atomic(statepath, state)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=GPU_UUID, PYTHONDONTWRITEBYTECODE='1',
               OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='2')
    try:
        with logpath.open('x') as log:
            parent = wait_for_parent(args.broad_study, args.broad_audit, contract_sha, queue_deadline, state, statepath)
            remaining_queue = queue_deadline-time.monotonic()
            if remaining_queue <= 0:
                raise TimeoutError('Parent completed after separate queue deadline')
            # Re-read immediately before the exclusive launch receipt.
            if parent_ready(args.broad_study, args.broad_audit, contract_sha) != parent:
                raise ValueError('Parent changed between acceptance and launch')
            command = [sys.executable, '-u', '-m', 'sircus.mjlab.fixed35_sixhour_study',
                '--global-deadline-unix', str(outer_deadline), '--queue-timeout', str(remaining_queue)]
            for key in (*INPUT_KEYS, 'output', 'gpu-lock', 'wall-time-limit', 'search-limit'):
                command += ['--'+key, str(getattr(args, key.replace('-', '_')))]
            exclusive_json(launchpath, dict(parent=parent, queue_receipt_sha256=sha(queuepath),
                source=str(source), command=command, requested_unix_s=time.time(),
                remaining_queue_s=remaining_queue, acquired_total_limit_s=ACQUIRED_LIMIT_S))
            state.update(status='waiting_for_gpu', parent=parent, launch_receipt_sha256=sha(launchpath))
            atomic(statepath, state)
            code = supervise(command, source=source, env=env, log=log, state=state, statepath=statepath,
                output=output, absolute_deadline=started+args.queue_timeout+ACQUIRED_LIMIT_S, worker_limit=args.wall_time_limit)
            state['study_exit_code'] = code
            if code:
                state.update(status='failed', error=state.get('error', 'Owned worker failed; saved evidence retained'))
            else:
                try:
                    timeout, budget = acquired_audit_time(budgetpath, outer_deadline=outer_deadline, now=time.time())
                    state.update(status='cpu_audit', acquired_deadline_unix_s=budget['acquired_deadline_unix_s'])
                    atomic(statepath, state)
                    audit_command = [sys.executable, '-u', '-m', 'sircus.mjlab.fixed35_sixhour_verify',
                        '--study', str(output), '--source', str(source), '--output', str(auditpath)]
                    for key in INPUT_KEYS:
                        audit_command += ['--'+key, str(getattr(args, key.replace('-', '_')))]
                    result = subprocess.run(audit_command, cwd=source, env=dict(env, CUDA_VISIBLE_DEVICES=''),
                        stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, timeout=timeout)
                    state['audit_exit_code'] = result.returncode
                    if result.returncode:
                        raise ValueError('Independent follow-up audit failed')
                    status, report, audit = terminal_status(output, auditpath)
                    if parent_ready(args.broad_study, args.broad_audit, contract_sha) != parent:
                        raise ValueError('Parent evidence changed during the follow-up')
                    state.update(status=status, study_status=report['status'], report_sha256=sha(output/'report.json'),
                                 audit_sha256=sha(auditpath), audit_checks=audit.get('checks_count'))
                except (OSError, ValueError, KeyError, TypeError, TimeoutError, subprocess.TimeoutExpired) as error:
                    state.update(status='audit_failed', error=type(error).__name__+': '+str(error))
    except Exception as error:
        state.update(status='failed', error=type(error).__name__+': '+str(error))
        raise
    finally:
        state.update(finished_unix_s=time.time(), elapsed_s=time.monotonic()-started)
        atomic(statepath, state)
    print(json.dumps(state), flush=True)
    if state['status'] not in ('completed', 'incomplete_audited'):
        raise RuntimeError('Six-hour follow-up not audited: '+state['status'])
    return state


if __name__ == '__main__': main()
