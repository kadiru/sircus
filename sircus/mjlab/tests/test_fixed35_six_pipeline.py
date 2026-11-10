"""Finite new-work isolation and audit failures stay explicit."""
import json
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from sircus.mjlab import fixed35_six_pipeline as P


class SixPipelineTests(unittest.TestCase):
    def invoke(self, root, worker=0, audit=0, timeout=False):
        argv = ['pipeline'] + [v for key in (*P.INPUT_KEYS, 'output', 'gpu-lock')
                              for v in ('--' + key, str(root / ('study' if key == 'output' else key)))]
        calls = []
        def supervise(command, **kwargs):
            calls.append(('worker', command, kwargs))
            return worker
        def run(command, **kwargs):
            calls.append(('audit', command, kwargs))
            if timeout:
                raise subprocess.TimeoutExpired(command, kwargs['timeout'])
            return SimpleNamespace(returncode=audit)
        with patch.object(P.sys, 'argv', argv), patch.object(P, 'supervise', side_effect=supervise), \
             patch.object(P.subprocess, 'run', side_effect=run), \
             patch.object(P.time, 'monotonic', return_value=100.), \
             patch.object(P.time, 'time', return_value=1000.), patch('builtins.print'):
            if worker or audit or timeout:
                with self.assertRaisesRegex(RuntimeError, 'incomplete'):
                    P.main()
            else:
                P.main()
        return calls, json.loads((root / 'study.pipeline.json').read_text())

    def test_4090_only_worker_cpu_audit_finite_deadline_no_publication(self):
        with tempfile.TemporaryDirectory() as tmp:
            calls, state = self.invoke(Path(tmp))
            self.assertEqual([c[0] for c in calls], ['worker', 'audit'])
            self.assertEqual(calls[0][2]['env']['CUDA_VISIBLE_DEVICES'], P.GPU_UUID)
            self.assertEqual(calls[0][2]['absolute_deadline'], 10000.)
            self.assertEqual(calls[0][2]['worker_limit'], 7200)
            self.assertEqual(calls[1][2]['env']['CUDA_VISIBLE_DEVICES'], '')
            self.assertEqual(calls[1][2]['timeout'], 1800)
            self.assertEqual(state['status'], 'completed')
            self.assertFalse(state['training'])
            self.assertFalse(state['ui_publication'])
            self.assertTrue(all('--' + key in calls[1][1] for key in P.INPUT_KEYS))

    def test_no_false_completion_when_worker_or_audit_fails(self):
        for worker, audit, timeout in ((1, 0, False), (0, 1, False), (0, 0, True)):
            with self.subTest(worker=worker, audit=audit, timeout=timeout), tempfile.TemporaryDirectory() as tmp:
                calls, state = self.invoke(Path(tmp), worker, audit, timeout)
                self.assertEqual(len(calls), 1 if worker else 2)
                self.assertIn(state['status'], ('failed', 'audit_failed'))

    def test_never_overwrite_prior_pipeline_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / 'study.pipeline.log'
            path.write_text('keep')
            with self.assertRaisesRegex(ValueError, 'Fresh owned'):
                self.invoke(root)
            self.assertEqual(path.read_text(), 'keep')


if __name__ == '__main__':
    unittest.main()
