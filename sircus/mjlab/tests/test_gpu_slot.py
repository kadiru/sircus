import os,tempfile,unittest
from unittest.mock import patch
from sircus.mjlab.gpu_slot import gpu_slot,foreign_processes,GPU_UUID


class SlotTests(unittest.TestCase):
    def test_4070_does_not_block(self):
        with patch.dict(os.environ,CUDA_VISIBLE_DEVICES=GPU_UUID),patch('subprocess.check_output',return_value='OTHER, 77\n'):
            self.assertEqual(foreign_processes(),[])

    def test_foreign_4090_rejected_before_context(self):
        with tempfile.NamedTemporaryFile() as f,patch('sircus.mjlab.gpu_slot.foreign_processes',return_value=[77]):
            with self.assertRaises(TimeoutError):
                with gpu_slot(f.name,timeout_s=0):self.fail('must not acquire')

    def test_idle_slot_enters_and_releases(self):
        with tempfile.NamedTemporaryFile() as f,patch('sircus.mjlab.gpu_slot.foreign_processes',return_value=[]):
            with gpu_slot(f.name,timeout_s=0):pass
            with gpu_slot(f.name,timeout_s=0):pass

    def test_wrong_gpu_selection_fails(self):
        with patch.dict(os.environ,CUDA_VISIBLE_DEVICES='0,1'):
            with self.assertRaises(RuntimeError):foreign_processes()
