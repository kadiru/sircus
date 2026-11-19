"""No GPU: completion, partial evidence, audit binding and finite supervision."""
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from sircus.mjlab import fixed35_broad_pipeline as P


class BroadPipelineTests(unittest.TestCase):
    def argv(self, root, extra=()):
        return ['pipeline'] + [v for key in (*P.INPUT_KEYS, 'output', 'gpu-lock')
            for v in ('--' + key, str(root / ('study' if key == 'output' else key)))] + list(extra)

    def invoke(self, root, *, worker_code=0, report_status='completed', audit_code=0,
               audit_status='passed', audit_timeout=False, bind_audit=True,
               write_audit=True, clock=None, extra=()):
        calls = []
        def worker(command, **kwargs):
            calls.append(('worker', command, kwargs))
            output = root/'study'
            output.mkdir()
            (output/'report.json').write_text(json.dumps(dict(status=report_status)))
            (output/'packet-receipt.json').write_text('preserve saved packet')
            return worker_code
        def audit(command, **kwargs):
            calls.append(('audit', command, kwargs))
            if audit_timeout:
                raise subprocess.TimeoutExpired(command, kwargs['timeout'])
            if write_audit:
                report_hash = hashlib.sha256((root/'study/report.json').read_bytes()).hexdigest()
                Path(command[command.index('--output')+1]).write_text(json.dumps(dict(
                    status=audit_status, failures=[] if audit_status == 'passed' else ['failed'],
                    report_sha256=report_hash if bind_audit else '0'*64, checks_count=17)))
            return SimpleNamespace(returncode=audit_code)
        with patch.object(P.sys, 'argv', self.argv(root, extra)), \
             patch.object(P, 'supervise', side_effect=worker), \
             patch.object(P.subprocess, 'run', side_effect=audit), \
             patch.object(P.Path, 'is_file', return_value=True), \
             patch.object(P.time, 'monotonic', side_effect=clock if clock else None, return_value=100.), \
             patch.object(P.time, 'time', return_value=1000.), patch('builtins.print'):
            try:
                result = P.main()
                error = None
            except RuntimeError as caught:
                result, error = None, caught
        return calls, json.loads((root/'study.pipeline.json').read_text()), result, error

    def test_completed_requires_bound_audit_and_completed_worker(self):
        with tempfile.TemporaryDirectory() as tmp:
            calls, state, result, error = self.invoke(Path(tmp))
            self.assertIsNone(error)
            self.assertEqual(state['status'], 'completed')
            self.assertEqual(result, state)
            self.assertEqual([c[0] for c in calls], ['worker', 'audit'])
            worker, audit = calls
            self.assertEqual(worker[2]['env']['CUDA_VISIBLE_DEVICES'], P.GPU_UUID)
            self.assertEqual(audit[2]['env']['CUDA_VISIBLE_DEVICES'], '')
            self.assertEqual(worker[2]['absolute_deadline'], 51400.)
            self.assertEqual(worker[2]['worker_limit'], 43200)
            self.assertEqual(audit[2]['timeout'], 7200)
            for key in P.INPUT_KEYS:
                self.assertIn('--'+key, worker[1])
                self.assertIn('--'+key, audit[1])
            self.assertEqual(worker[1][worker[1].index('--search-limit')+1], '36000')
            self.assertEqual(worker[1][worker[1].index('--global-deadline-unix')+1], '52300.0')
            self.assertFalse(state['training'])
            self.assertFalse(state['ui_publication'])
            self.assertFalse(state['resume'])
            self.assertEqual(state['target_search_releases'], 1000)

    def test_incomplete_worker_is_still_audited_without_false_completion(self):
        with tempfile.TemporaryDirectory() as tmp:
            calls, state, result, error = self.invoke(Path(tmp), report_status='incomplete')
            self.assertIsNone(error)
            self.assertEqual([c[0] for c in calls], ['worker', 'audit'])
            self.assertEqual(state['status'], 'incomplete_audited')
            self.assertEqual(result['study_status'], 'incomplete')
            self.assertEqual((Path(tmp)/'study/packet-receipt.json').read_text(), 'preserve saved packet')

    def test_worker_failure_skips_audit_and_preserves_partial_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            calls, state, _, error = self.invoke(Path(tmp), worker_code=1, report_status='failed')
            self.assertIsNotNone(error)
            self.assertEqual([c[0] for c in calls], ['worker'])
            self.assertEqual(state['status'], 'failed')
            self.assertTrue((Path(tmp)/'study/packet-receipt.json').is_file())

    def test_audit_nonzero_timeout_missing_wrong_hash_or_failed_status_cannot_complete(self):
        cases = [dict(audit_code=1), dict(audit_timeout=True), dict(write_audit=False),
                 dict(bind_audit=False), dict(audit_status='failed'), dict(report_status='failed')]
        for case in cases:
            with self.subTest(case=case), tempfile.TemporaryDirectory() as tmp:
                _, state, _, error = self.invoke(Path(tmp), **case)
                self.assertIsNotNone(error)
                self.assertEqual(state['status'], 'audit_failed')
                self.assertEqual((Path(tmp)/'study/packet-receipt.json').read_text(), 'preserve saved packet')

    def test_expired_global_deadline_never_grants_fresh_audit_budget(self):
        with tempfile.TemporaryDirectory() as tmp:
            calls, state, _, error = self.invoke(Path(tmp), clock=[100., 51400., 51401.])
            self.assertIsNotNone(error)
            self.assertEqual([c[0] for c in calls], ['worker'])
            self.assertEqual(state['status'], 'audit_failed')
            self.assertEqual(state['elapsed_s'], 51301.)

    def test_cpu_audit_is_bounded_by_remaining_original_time(self):
        with tempfile.TemporaryDirectory() as tmp:
            calls, state, _, error = self.invoke(Path(tmp), clock=[100., 50600., 50610.])
            self.assertIsNone(error)
            self.assertEqual(calls[1][2]['timeout'], 795.)
            self.assertEqual(state['global_deadline_unix_s'], 52300.)

    def test_all_old_namespaces_are_refused_without_overwriting(self):
        for name in ('study', 'study.pipeline.json', 'study.pipeline.log', 'study-budget.json', 'study-audit.json'):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                existing = root/name
                existing.write_text('keep')
                with patch.object(P.sys, 'argv', self.argv(root)), patch.object(P, 'supervise') as worker:
                    with self.assertRaisesRegex(ValueError, 'Fresh owned'):
                        P.main()
                    worker.assert_not_called()
                self.assertEqual(existing.read_text(), 'keep')

    def test_missing_auditor_refuses_before_any_worker_or_namespace(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with patch.object(P.sys, 'argv', self.argv(root)), patch.object(P.Path, 'is_file', return_value=False), \
                 patch.object(P, 'supervise') as worker:
                with self.assertRaisesRegex(ValueError, 'auditor must exist'):
                    P.main()
                worker.assert_not_called()
            self.assertEqual(list(root.iterdir()), [])

    def test_invalid_caps_refused_before_execution(self):
        cases = [('--wall-time-limit', '43201'), ('--search-limit', '36001'), ('--queue-timeout', '901'),
                 ('--queue-timeout', '0'), ('--global-limit', '51301'), ('--global-limit', '43200'),
                 ('--wall-time-limit', '60'), ('--search-limit', 'nan'),
                 ('--wall-time-limit', '60', '--search-limit', '30')]
        for extra in cases:
            with self.subTest(extra=extra), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                with patch.object(P.sys, 'argv', self.argv(root, extra)), patch.object(P, 'supervise') as worker:
                    with self.assertRaisesRegex(ValueError, 'Finite ordered'):
                        P.main()
                    worker.assert_not_called()
                self.assertEqual(list(root.iterdir()), [])

    def test_production_target_cannot_be_overridden(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(P.sys, 'argv', self.argv(Path(tmp), ('--search-target', '10'))), \
                 patch.object(P.sys, 'stderr'), patch.object(P, 'supervise') as worker:
                with self.assertRaises(SystemExit):
                    P.main()
                worker.assert_not_called()


if __name__ == '__main__':
    unittest.main()
