"""Own-process deadline and CPU-only postprocessing orchestration; no GPU/services."""
import io
import json
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
from sircus.mjlab import coverage_training_pipeline as P


class PipelineTests(unittest.TestCase):
    def argv(self,root):
        return ['pipeline']+[v for key in (*P.INPUT_KEYS,'output','gpu-lock')
            for v in ('--'+key,str(root/('study' if key=='output' else key)))]

    def invoke(self,root,*,worker_code=0,audit_code=0,video_code=0,timeout_video=False):
        calls=[]
        def supervise(command,**kwargs):
            calls.append(('worker',command,kwargs));return worker_code
        def run(command,**kwargs):
            stage='video' if 'sircus.mjlab.coverage_training_video' in command else 'audit'
            calls.append((stage,command,kwargs))
            if stage=='video':
                live=json.loads((root/'study.pipeline.json').read_text())
                self.assertEqual(live['status'],'cpu_video');self.assertNotIn('finished_unix_s',live)
            if stage=='video' and timeout_video:raise subprocess.TimeoutExpired(command,kwargs['timeout'])
            return SimpleNamespace(returncode=video_code if stage=='video' else audit_code)
        with patch.object(P.sys,'argv',self.argv(root)),patch.object(P,'supervise',side_effect=supervise),\
             patch.object(P.subprocess,'run',side_effect=run),patch.object(P.time,'monotonic',return_value=100.),\
             patch.object(P.time,'time',return_value=1000.),patch('builtins.print'):
            if worker_code or audit_code:
                with self.assertRaisesRegex(RuntimeError,'incomplete'):P.main()
            else:P.main()
        state=json.loads((root/'study.pipeline.json').read_text())
        return calls,state

    def test_full_pipeline_hides_gpu_from_audit_and_video_and_freezes_global_deadline(self):
        with tempfile.TemporaryDirectory() as tmp:
            calls,state=self.invoke(Path(tmp))
            self.assertEqual([c[0] for c in calls],['worker','audit','video'])
            worker=calls[0];self.assertEqual(worker[2]['env']['CUDA_VISIBLE_DEVICES'],P.GPU_UUID)
            self.assertEqual(worker[1][worker[1].index('--global-deadline-unix')+1],'26200.0')
            self.assertEqual(worker[2]['absolute_deadline'],25300.)
            self.assertEqual(worker[2]['worker_limit'],19800)
            for stage,command,kwargs in calls[1:]:
                self.assertEqual(kwargs['env']['CUDA_VISIBLE_DEVICES'],'')
                self.assertLessEqual(kwargs['timeout'],3600 if stage=='audit' else 900)
            self.assertEqual(calls[-1][2]['env']['LIBGL_ALWAYS_SOFTWARE'],'1')
            self.assertEqual(state['status'],'completed');self.assertEqual(state['video_status'],'completed')
            self.assertEqual(state['started_unix_s'],1000.);self.assertEqual(state['global_deadline_unix_s'],26200.)

    def test_worker_or_audit_failure_skips_later_phases(self):
        for worker_code,audit_code,phases in ((1,0,['worker']),(0,1,['worker','audit'])):
            with self.subTest(worker=worker_code,audit=audit_code),tempfile.TemporaryDirectory() as tmp:
                calls,state=self.invoke(Path(tmp),worker_code=worker_code,audit_code=audit_code)
                self.assertEqual([c[0] for c in calls],phases)
                self.assertIn(state['status'],('failed','audit_failed'))

    def test_optional_video_failure_preserves_passed_numerical_audit(self):
        for timed_out in (False,True):
            with self.subTest(timed_out=timed_out),tempfile.TemporaryDirectory() as tmp:
                calls,state=self.invoke(Path(tmp),video_code=1,timeout_video=timed_out)
                self.assertEqual(state['status'],'completed');self.assertEqual(state['audit_exit_code'],0)
                self.assertEqual(state['video_status'],'failed');self.assertIn('valid',state['video_error'])

    def test_supervisor_terminates_only_its_child_at_global_deadline(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);child=Mock(pid=123,returncode=-15);child.poll.return_value=None
            with patch.object(P.subprocess,'Popen',return_value=child) as launch,\
                 patch.object(P.time,'monotonic',return_value=101.),patch.object(P,'stop_owned') as stop:
                code=P.supervise(['own-worker'],source=root,env={},log=io.StringIO(),state={},
                    statepath=root/'state.json',output=root/'study',absolute_deadline=100.,worker_limit=19800)
            self.assertEqual(code,-15);stop.assert_called_once_with(child);launch.assert_called_once()

    def test_supervisor_uses_effective_acquired_cap_and_no_unrelated_process(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'study-budget.json').write_text(json.dumps(dict(acquired_unix_s=100.,effective_worker_limit_s=60.)))
            child=Mock(pid=987,returncode=-15);child.poll.return_value=None
            with patch.object(P.subprocess,'Popen',return_value=child),patch.object(P.time,'time',return_value=161.),\
                 patch.object(P.time,'monotonic',return_value=1.),patch.object(P,'stop_owned') as stop:
                P.supervise(['own-worker'],source=root,env={},log=io.StringIO(),state={},statepath=root/'state.json',
                    output=root/'study',absolute_deadline=1000.,worker_limit=19800)
            stop.assert_called_once_with(child)

    def test_owned_termination_escalates_only_when_that_child_ignores_term(self):
        child=Mock();child.wait.side_effect=[subprocess.TimeoutExpired('own',5),0]
        P.stop_owned(child)
        child.terminate.assert_called_once();child.kill.assert_called_once();self.assertEqual(child.wait.call_count,2)


if __name__=='__main__':unittest.main()
