"""Finite worker, isolated GPU assignment and fail-closed independent CPU audit."""
import json
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from sircus.mjlab import corrected_imitation_pipeline as P


class PipelineTests(unittest.TestCase):
    def invoke(self,root,*,worker=0,audit=0,timeout=False,extra=()):
        argv=['pipeline']+[v for key in (*P.INPUT_KEYS,'output','gpu-lock')
            for v in ('--'+key,str(root/('study' if key=='output' else key)))]+list(extra)
        calls=[]
        def supervise(command,**kwargs):calls.append(('worker',command,kwargs));return worker
        def run(command,**kwargs):
            calls.append(('audit',command,kwargs))
            if timeout:raise subprocess.TimeoutExpired(command,kwargs['timeout'])
            return SimpleNamespace(returncode=audit)
        with patch.object(P.sys,'argv',argv),patch.object(P,'supervise',side_effect=supervise),\
             patch.object(P.subprocess,'run',side_effect=run),patch.object(P.time,'monotonic',return_value=100.),\
             patch.object(P.time,'time',return_value=1000.),patch('builtins.print'):
            if worker or audit or timeout:
                with self.assertRaisesRegex(RuntimeError,'incomplete'):P.main()
            else:P.main()
        return calls,json.loads((root/'study.pipeline.json').read_text())

    def test_worker4090_only_cpu_audit_and_absolute_caps(self):
        with tempfile.TemporaryDirectory() as tmp:
            calls,state=self.invoke(Path(tmp))
            self.assertEqual([c[0] for c in calls],['worker','audit'])
            command,kwargs=calls[0][1:]
            self.assertEqual(kwargs['env']['CUDA_VISIBLE_DEVICES'],P.GPU_UUID)
            self.assertEqual(kwargs['worker_limit'],3600);self.assertEqual(kwargs['absolute_deadline'],6400.)
            self.assertEqual(command[command.index('--global-deadline-unix')+1],'7300.0')
            command,kwargs=calls[1][1:]
            self.assertEqual(kwargs['env']['CUDA_VISIBLE_DEVICES'],'');self.assertEqual(kwargs['timeout'],1800)
            self.assertTrue(all('--'+key in command for key in P.INPUT_KEYS))
            self.assertEqual(state['status'],'completed');self.assertEqual(state['global_limit_s'],6300.)

    def test_original_deadline_can_only_shorten_work(self):
        with tempfile.TemporaryDirectory() as tmp:
            calls,state=self.invoke(Path(tmp),extra=('--absolute-deadline-unix','6000'))
            self.assertEqual(state['global_deadline_unix_s'],6000.)
            self.assertEqual(calls[0][2]['absolute_deadline'],5100.)
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError,'CPU audit reserve'):self.invoke(Path(tmp),extra=('--absolute-deadline-unix','2800'))

    def test_failure_or_timeout_never_reports_completed(self):
        for worker,audit,timeout in ((1,0,False),(0,1,False),(0,0,True)):
            with self.subTest(worker=worker,audit=audit,timeout=timeout),tempfile.TemporaryDirectory() as tmp:
                calls,state=self.invoke(Path(tmp),worker=worker,audit=audit,timeout=timeout)
                self.assertEqual(len(calls),1 if worker else 2)
                self.assertIn(state['status'],('failed','audit_failed'))

    def test_existing_namespace_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);path=root/'study.pipeline.log';path.write_text('preserve')
            with self.assertRaisesRegex(ValueError,'Fresh owned'):self.invoke(root)
            self.assertEqual(path.read_text(),'preserve')

if __name__=='__main__':unittest.main()
