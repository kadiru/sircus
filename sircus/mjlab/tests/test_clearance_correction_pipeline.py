from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from sircus.mjlab import clearance_correction_pipeline as P

class Protocol(unittest.TestCase):
    def args(self):
        a={k:Path('/proof')/k for k in (*P.INPUT_KEYS,'diagnosis_report','diagnosis_audit','gpu_lock','display')}
        return SimpleNamespace(**a,queue_timeout=86400,wall_time_limit=3600)
    def test_cpu_prepare_and_standalone_native_paths_preserve_frozen_arguments(self):
        a=self.args();root=Path('/source');prepare=P.prepare_command(a,root);worker=P.worker_command(a,root,'a'*64)
        self.assertIn('-m',prepare);self.assertNotIn('-m',worker);self.assertIn('--prepare-only',prepare)
        self.assertEqual(worker[2],str(root/'sircus/mjlab/clearance_correction_study.py'))
        for command in (prepare,worker):
            for key in P.INPUT_KEYS:self.assertEqual(command[command.index('--'+key.replace('_','-'))+1],str(getattr(a,key)))
        self.assertEqual(worker[worker.index('--contract-sha256')+1],'a'*64)
        self.assertEqual(worker[worker.index('--gpu-lock')+1],str(a.gpu_lock))
        audit=P.audit_command(a,root)
        for flag,value in (('--parent',a.parent_study),('--parent-audit',a.parent_audit),('--study',a.output),('--native-source',a.native_source)):
            self.assertEqual(audit[audit.index(flag)+1],str(value))
    def test_environment_overrides_inherited_gpu_and_bounds_cpu_threads(self):
        with patch.dict('os.environ',{'CUDA_VISIBLE_DEVICES':'4070','OMP_NUM_THREADS':'32'}):
            self.assertEqual(P.environment('')['CUDA_VISIBLE_DEVICES'],'')
            self.assertEqual(P.environment(P.GPU_UUID)['CUDA_VISIBLE_DEVICES'],P.GPU_UUID)
            self.assertEqual(P.environment('')['OMP_NUM_THREADS'],'2')
    def test_parser_requires_diagnosis_and_proof(self):
        a=self.args();values=[v for k in (*P.INPUT_KEYS,'diagnosis_report','diagnosis_audit','gpu_lock','display') for v in ('--'+k.replace('_','-'),str(getattr(a,k)))]
        got=P.parser().parse_args(values);self.assertEqual(got.diagnosis_audit,a.diagnosis_audit)

if __name__=='__main__':unittest.main()
