"""Bounded preparation orchestration tests; no real solver, model or native job."""

from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile
import unittest
from unittest.mock import Mock,patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import candidate_fresh_prepare as prepare


class PrepareTests(unittest.TestCase):
    def test_existing_or_frozen_output_rejected_before_any_model_or_solver(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary); frozen=root/'frozen';frozen.mkdir()
            with patch.object(prepare,'load_geometry') as geometry:
                for dest in (frozen, frozen/'untouched'):
                    with self.assertRaisesRegex(ValueError,'untouched path'):
                        prepare.build_preparation(root,root/'parent.pt',dest,frozen_experiment_root=frozen)
                geometry.assert_not_called()

    def test_wrong_original_coordinates_or_scene_fail_before_qualification(self):
        with patch.object(prepare,'digest',return_value='f'*64):
            with self.assertRaisesRegex(ValueError,'coordinate manifest'): prepare.load_geometry('/unused')
        with patch.object(prepare,'digest',return_value=prepare.ORIGINAL_SPLIT_SHA256),patch.object(prepare,'read',return_value={}):
            with self.assertRaisesRegex(ValueError,'scene changed'): prepare.load_geometry('/unused')

    def test_known_original_poses_are_required(self):
        expected={'original-splits.json':prepare.ORIGINAL_SPLIT_SHA256,
                  'dropper.urdf':prepare.ASSET_SHA256['Arm10'],'board_arm.urdf':prepare.ASSET_SHA256['Arm14'],
                  'poses.json':'f'*64}
        with patch.object(prepare,'digest',side_effect=lambda p:expected[Path(p).name]), \
             patch.object(prepare,'read',return_value={}),patch.object(prepare,'canonical_hash',return_value=prepare.SCENE_SHA256), \
             patch.object(prepare.ET,'parse'),patch.object(prepare,'chain'):
            with self.assertRaisesRegex(ValueError,'home poses changed'): prepare.load_geometry('/unused')

    def fake_context(self, reply=None):
        process=Mock(pid=99999999)
        alive=[True]
        process.is_alive.side_effect=lambda:alive[0]
        process.terminate.side_effect=lambda:alive.__setitem__(0,False)
        context=Mock();context.Process.return_value=process
        if reply is not None:context.Queue.return_value.get.return_value=reply
        return context,process

    def test_live_cpu_cap_terminates_only_owned_auditor(self):
        context,process=self.fake_context()
        with patch.object(prepare.mp,'get_context',return_value=context), \
             patch.object(prepare.shutil,'disk_usage',return_value=SimpleNamespace(free=100*1024**3)), \
             patch.object(prepare.time,'process_time',side_effect=[0.,10.]):
            with self.assertRaisesRegex(TimeoutError,'CPU cap'):
                prepare.bounded_audit(Path('/unused'),Path('/unused'),wall_limit_s=60,cpu_limit_s=1)
        process.start.assert_called_once();process.terminate.assert_called_once()
        context.Queue.return_value.close.assert_called_once()

    def test_audit_receipt_cost_is_measured_and_new_json_cannot_overwrite(self):
        context,process=self.fake_context(({'status':'passed'},.01,None))
        with patch.object(prepare.mp,'get_context',return_value=context), \
             patch.object(prepare.shutil,'disk_usage',return_value=SimpleNamespace(free=100*1024**3)):
            audit,cost=prepare.bounded_audit(Path('/unused'),Path('/unused'),wall_limit_s=60,cpu_limit_s=60)
        self.assertEqual(audit,{'status':'passed'})
        self.assertGreaterEqual(cost['aggregate_cpu_s'],0.)
        self.assertEqual(cost['worker_compute_cpu_s'],.01)
        process.terminate.assert_called_once()
        with tempfile.TemporaryDirectory() as temporary:
            path=Path(temporary)/'new.json';prepare.write_new(path,{'preserve':True});raw=path.read_bytes()
            with self.assertRaises(FileExistsError):prepare.write_new(path,{'preserve':False})
            self.assertEqual(path.read_bytes(),raw)


if __name__=='__main__':unittest.main()
