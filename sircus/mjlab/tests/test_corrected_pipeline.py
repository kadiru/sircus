from pathlib import Path
import json
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

from sircus.mjlab.corrected_pipeline import commands,run
from sircus.mjlab.gpu_slot import GPU_UUID


class PipelineTests(unittest.TestCase):
    def args(self, root):
        return SimpleNamespace(parent_study=root/'parent',bank=root/'bank.json',contact_reference=root/'ref.json',assets=root/'assets',
            output=root/'study',gpu_lock=root/'scene.lock',source_study=[root/'one',root/'two'],historical_study=[root/'history'],queue_timeout=2)
    def test_exact_shared_inputs_and_audit_uses_actual_source(self):
        a=self.args(Path('/tmp/example'));grid,audit=commands(a,'python',Path('/tmp/frozen-source'))
        self.assertEqual(grid.count('--source-study'),2);self.assertEqual(audit.count('--source-study'),2)
        self.assertIn('/tmp/frozen-source',audit);self.assertNotIn('--gpu-lock',audit)
        self.assertEqual(grid[3],'sircus.mjlab.corrected_grid');self.assertEqual(audit[3],'sircus.mjlab.corrected_verify')
    def test_cpu_audit_only_after_success_and_cuda_hidden(self):
        with tempfile.TemporaryDirectory() as td,mock.patch.dict('os.environ',{'CUDA_VISIBLE_DEVICES':GPU_UUID}),mock.patch('sircus.mjlab.corrected_pipeline.subprocess.run',return_value=SimpleNamespace(returncode=0)) as child:
            a=self.args(Path(td));run(a)
            self.assertEqual(child.call_count,2)
            self.assertEqual(child.call_args_list[1].kwargs['env']['CUDA_VISIBLE_DEVICES'],'')
            self.assertLessEqual(child.call_args_list[0].kwargs['timeout'],28800)
    def test_failed_grid_never_launches_audit_and_output_is_not_reused(self):
        with tempfile.TemporaryDirectory() as td,mock.patch.dict('os.environ',{'CUDA_VISIBLE_DEVICES':GPU_UUID}),mock.patch('sircus.mjlab.corrected_pipeline.subprocess.run',return_value=SimpleNamespace(returncode=1)) as child:
            a=self.args(Path(td))
            with self.assertRaises(RuntimeError):run(a)
            self.assertEqual(child.call_count,1)
            with self.assertRaises(ValueError):run(a)
    def test_audit_failure_does_not_rewrite_completed_native_report(self):
        with tempfile.TemporaryDirectory() as td,mock.patch.dict('os.environ',{'CUDA_VISIBLE_DEVICES':GPU_UUID}):
            a=self.args(Path(td));original={'status':'completed','phase':'completed','counts':{'native_attempts':5}}
            def child(command,**kwargs):
                if 'sircus.mjlab.corrected_grid' in command:
                    a.output.mkdir();(a.output/'report.json').write_text(json.dumps(original))
                    return SimpleNamespace(returncode=0)
                return SimpleNamespace(returncode=1)
            with mock.patch('sircus.mjlab.corrected_pipeline.subprocess.run',side_effect=child):
                with self.assertRaises(RuntimeError):run(a)
            self.assertEqual(json.loads((a.output/'report.json').read_text()),original)
            self.assertEqual(json.loads(a.output.with_name('study-pipeline.json').read_text())['audit_exit_code'],1)

if __name__=='__main__':unittest.main()
