"""Mocked parent queue, single launch, GPU isolation and acquired audit budgets."""
import copy
import json
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from sircus.mjlab import fixed35_sixhour_pipeline as P


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def parent_fixture(root, *, status='completed'):
    study, auditpath = root/'broad-study', root/'broad-audit'
    spec = dict(version=P.PARENT_VERSION, source_sha256={'source.py':'f'*64})
    spec['fingerprint'] = P.digest(spec)
    write(study/'contract.json', spec)
    manifest = {'rows.jsonl':'e'*64}
    write(study/'evidence-manifest.json', manifest)
    report = dict(version=P.PARENT_VERSION, status='completed', phase='completed',
        fresh_search_releases=P.PARENT_QUOTAS, gate=dict(search_quota_completed=True),
        contract_fingerprint=spec['fingerprint'], evidence_sha256=manifest)
    write(study/'report.json', report)
    audit = dict(version=P.PARENT_AUDIT_VERSION, status='passed', failures=[],
        full_checks_verified=True, native_checks_verified=True, report_sha256=P.sha(study/'report.json'),
        contract_fingerprint=spec['fingerprint'], evidence_manifest_sha256=P.sha(study/'evidence-manifest.json'),
        producer_source_sha256=spec['source_sha256'])
    write(auditpath, audit)
    write(study.with_suffix('.pipeline.json'), dict(status=status, output=str(study),
        report_sha256=P.sha(study/'report.json'), audit_sha256=P.sha(auditpath), finished_unix_s=900.))
    return study, auditpath, P.sha(study/'contract.json')


class ParentTests(unittest.TestCase):
    def test_ready_requires_completed_pipeline_and_full_audit(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            study, audit, fingerprint = parent_fixture(root, status='cpu_audit')
            self.assertIsNone(P.parent_ready(study, audit, fingerprint))
            value = P.read(study.with_suffix('.pipeline.json')); value['status'] = 'completed'
            write(study.with_suffix('.pipeline.json'), value)
            receipt = P.parent_ready(study, audit, fingerprint)
            self.assertEqual(receipt['completed_search_releases'], P.PARENT_QUOTAS)
            self.assertEqual(receipt['report_sha256'], P.sha(study/'report.json'))

    def test_failed_incomplete_or_changed_parent_refuses(self):
        for mutation in ('failed', 'audit_failed', 'incomplete_audited', 'contract', 'report', 'audit', 'source', 'pipeline'):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as tmp:
                study, audit, fingerprint = parent_fixture(Path(tmp))
                if mutation in ('failed', 'audit_failed', 'incomplete_audited'):
                    value = P.read(study.with_suffix('.pipeline.json')); value['status'] = mutation
                    write(study.with_suffix('.pipeline.json'), value)
                elif mutation == 'contract':
                    (study/'contract.json').write_text('{}')
                elif mutation == 'report':
                    value = P.read(study/'report.json'); value['fresh_search_releases'][next(iter(P.PARENT_QUOTAS))] -= 1
                    write(study/'report.json', value)
                elif mutation == 'audit':
                    value = P.read(audit); value['full_checks_verified'] = False; write(audit, value)
                elif mutation == 'source':
                    value = P.read(audit); value['producer_source_sha256'] = {}; write(audit, value)
                else:
                    value = P.read(study.with_suffix('.pipeline.json')); value['report_sha256'] = 'wrong'
                    write(study.with_suffix('.pipeline.json'), value)
                with self.assertRaises(ValueError): P.parent_ready(study, audit, fingerprint)

    def test_bounded_queue_waits_for_audited_parent_not_merely_worker_completion(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            state = dict(started_monotonic_s=0.)
            with patch.object(P, 'parent_ready', side_effect=[None, None, {'ready':True}]), \
                    patch.object(P.time, 'monotonic', return_value=5.), patch.object(P.time, 'sleep') as sleep:
                result = P.wait_for_parent(root/'study', root/'audit', 'sha', 20., state, root/'state.json')
            self.assertEqual(result, {'ready':True})
            self.assertEqual(sleep.call_count, 2)
            with patch.object(P.time, 'monotonic', return_value=20.), patch.object(P, 'parent_ready') as ready:
                with self.assertRaises(TimeoutError): P.wait_for_parent(root/'study', root/'audit', 'sha', 20., state, root/'state.json')
                ready.assert_not_called()


class PipelineTests(unittest.TestCase):
    def argv(self, root, extra=()):
        return ['pipeline']+[value for key in (*P.INPUT_KEYS, 'output', 'gpu-lock')
            for value in ('--'+key, str(root/('study' if key == 'output' else key)))]+list(extra)

    def invoke(self, root, *, worker_code=0, audit_code=0, report_status='completed',
               audit_timeout=False, acquired=1000., parent_status='completed', mutate_parent=False, clock=None):
        parent_fixture(root, status=parent_status)
        calls = []
        def worker(command, **kwargs):
            calls.append(('worker', command, kwargs))
            output = root/'study'; output.mkdir()
            write(output/'report.json', dict(status=report_status))
            outer = float(command[command.index('--global-deadline-unix')+1])
            write(root/'study-budget.json', dict(acquired_unix_s=acquired,
                acquired_total_limit_s=21600, acquired_deadline_unix_s=min(acquired+21600, outer),
                global_deadline_unix_s=outer, effective_worker_limit_s=18000, gpu_uuid=P.GPU_UUID))
            (output/'evidence').write_text('keep')
            return worker_code
        def audit(command, **kwargs):
            calls.append(('audit', command, kwargs))
            if audit_timeout: raise subprocess.TimeoutExpired(command, kwargs['timeout'])
            write(root/'study-audit.json', dict(status='passed', failures=[], checks_count=99,
                report_sha256=P.sha(root/'study/report.json')))
            if mutate_parent:
                (root/'broad-study/report.json').write_text('{}')
            return SimpleNamespace(returncode=audit_code)
        with patch.object(P.sys, 'argv', self.argv(root)), patch.object(P.Path, 'is_file', return_value=True), \
                patch.object(P, 'supervise', side_effect=worker), patch.object(P.subprocess, 'run', side_effect=audit), \
                patch.object(P.time, 'monotonic', side_effect=clock, return_value=100.), patch.object(P.time, 'time', return_value=1000.), \
                patch('builtins.print'):
            try:
                result, error = P.main(), None
            except (RuntimeError, ValueError, TimeoutError) as exc:
                result, error = None, exc
        return calls, P.read(root/'study.pipeline.json'), result, error

    def test_launch_is_once_after_parent_and_audit_is_cpu_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            calls, state, result, error = self.invoke(root)
            self.assertIsNone(error)
            self.assertEqual(state['status'], 'completed')
            self.assertEqual([x[0] for x in calls], ['worker', 'audit'])
            worker, audit = calls
            self.assertEqual(worker[2]['env']['CUDA_VISIBLE_DEVICES'], P.GPU_UUID)
            self.assertEqual(audit[2]['env']['CUDA_VISIBLE_DEVICES'], '')
            self.assertEqual(worker[2]['worker_limit'], 18000)
            self.assertEqual(audit[2]['timeout'], 3600)
            self.assertEqual(worker[1][worker[1].index('--queue-timeout')+1], '18000.0')
            self.assertEqual(worker[1][worker[1].index('--global-deadline-unix')+1], '40600.0')
            receipt = P.read(root/'study-launch.json')
            self.assertEqual(receipt['queue_receipt_sha256'], P.sha(root/'study-queue.json'))
            self.assertEqual(receipt['parent'], state['parent'])
            self.assertEqual(receipt['acquired_total_limit_s'], 21600)
            self.assertFalse(state['ui_publication'])
            self.assertFalse(state['training'])

    def test_failed_parent_never_launches(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            calls, state, _, error = self.invoke(root, parent_status='failed')
            self.assertIsNotNone(error)
            self.assertEqual(calls, [])
            self.assertEqual(state['status'], 'failed')
            self.assertFalse((root/'study-launch.json').exists())

    def test_parent_wait_is_subtracted_from_remaining_gpu_queue(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            calls, state, _, error = self.invoke(root, clock=[100., 600., 600., 601.])
            self.assertIsNone(error)
            command = calls[0][1]
            self.assertEqual(command[command.index('--queue-timeout')+1], '17500.0')
            self.assertEqual(P.read(root/'study-launch.json')['remaining_queue_s'], 17500.)
            self.assertEqual(state['queue_deadline_unix_s'], 19000.)

    def test_duplicate_queue_or_launch_receipt_refuses_without_overwriting(self):
        for name in ('study-queue.json', 'study-launch.json', 'study.pipeline.json', 'study.pipeline.log', 'study-budget.json', 'study-audit.json', 'study'):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp); path = root/name; path.write_text('keep')
                with patch.object(P.sys, 'argv', self.argv(root)), patch.object(P.Path, 'is_file', return_value=True), \
                        patch.object(P, 'supervise') as worker:
                    with self.assertRaisesRegex(ValueError, 'Fresh exclusive'): P.main()
                    worker.assert_not_called()
                self.assertEqual(path.read_text(), 'keep')

    def test_worker_failed_preserves_partial_and_never_audits(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            calls, state, _, error = self.invoke(root, worker_code=1)
            self.assertIsNotNone(error)
            self.assertEqual([c[0] for c in calls], ['worker'])
            self.assertEqual(state['status'], 'failed')
            self.assertEqual((root/'study/evidence').read_text(), 'keep')

    def test_incomplete_audited_is_not_promoted_to_completed(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, state, _, error = self.invoke(Path(tmp), report_status='incomplete')
            self.assertIsNone(error)
            self.assertEqual(state['status'], 'incomplete_audited')

    def test_audit_failure_timeout_or_mutated_parent_cannot_pass(self):
        for case in (dict(audit_code=1), dict(audit_timeout=True), dict(mutate_parent=True)):
            with self.subTest(case=case), tempfile.TemporaryDirectory() as tmp:
                _, state, _, error = self.invoke(Path(tmp), **case)
                self.assertIsNotNone(error)
                self.assertEqual(state['status'], 'audit_failed')

    def test_acquired_deadline_not_unused_queue_time_bounds_audit(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'budget.json'
            budget = dict(acquired_unix_s=1000., acquired_total_limit_s=21600,
                acquired_deadline_unix_s=22600., global_deadline_unix_s=40600.,
                effective_worker_limit_s=18000, gpu_uuid=P.GPU_UUID)
            write(path, budget)
            seconds, _ = P.acquired_audit_time(path, outer_deadline=40600., now=22000.)
            self.assertEqual(seconds, 595.)
            with self.assertRaises(TimeoutError): P.acquired_audit_time(path, outer_deadline=40600., now=22600.)
            budget['acquired_deadline_unix_s'] = 40600.
            write(path, budget)
            with self.assertRaises(ValueError): P.acquired_audit_time(path, outer_deadline=40600., now=22000.)

    def test_caps_and_missing_auditor_refuse_before_queue_claim(self):
        for extra in (('--wall-time-limit', '18001'), ('--search-limit', '16201'), ('--queue-timeout', '18001'), ('--search-limit', 'nan')):
            with self.subTest(extra=extra), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                with patch.object(P.sys, 'argv', self.argv(root, extra)), patch.object(P, 'supervise') as worker:
                    with self.assertRaises(ValueError): P.main()
                    worker.assert_not_called()
                self.assertEqual(list(root.iterdir()), [])
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with patch.object(P.sys, 'argv', self.argv(root)), patch.object(P.Path, 'is_file', return_value=False):
                with self.assertRaisesRegex(ValueError, 'auditor'): P.main()
            self.assertEqual(list(root.iterdir()), [])


if __name__ == '__main__': unittest.main()
