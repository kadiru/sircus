"""Corrected-coordinate profile isolation; no hardware, cameras or cloud."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

from sircus.operations.service import GuidedOperations
from sircus.operations.training import TrainingJobs
from sircus.studies.profiles import PROFILE_ID

ROOT = Path(__file__).resolve().parents[3]

class ProfileIsolation(unittest.TestCase):
    def test_front_coordinates_preserve_measured_assets_and_uncertainty(self):
        with tempfile.TemporaryDirectory() as folder:
            service = GuidedOperations(data_root=folder, profile='front')
            self.addCleanup(service.close)
            state = service.status(); scene = state['scene']
            self.assertEqual(state['profile_id'], PROFILE_ID)
            self.assertLess(scene['bin']['center_xy_mm'][1], scene['board']['center_mm'][1])
            self.assertLess(scene['board']['rpy_deg'][1], 0)
            self.assertEqual(scene['board']['size_mm'], [200,300,5])
            self.assertEqual(scene['bin']['reported_height_mm'], 175)
            self.assertEqual(scene['bin']['height_interpretation'], 'unresolved')
            self.assertFalse(state['hardware_enabled'])
            self.assertIn('not ball-center', state['profile_metadata']['screenshot_status'])

    def test_switching_profile_drops_incompatible_plan_scan_and_loading(self):
        with tempfile.TemporaryDirectory() as folder:
            service = GuidedOperations(data_root=folder)
            self.addCleanup(service.close)
            legacy = copy.deepcopy(service.status()['scene'])
            service._state.update(plan={'old':True}, bucket_observation={'old':True}, loaded=True)
            service.action({'action':'select_profile','profile':'front'})
            state = service.status()
            self.assertIsNone(state['plan']); self.assertIsNone(state['bucket_observation'])
            self.assertFalse(state['loaded']); self.assertEqual(state['phase'],'position_board')
            old_checkpoint = json.loads((ROOT/'baselines/scene-conditioned-20261030/checkpoint.json').read_text())
            with self.assertRaisesRegex(ValueError,'match'):
                service.action({'action':'select_checkpoint','checkpoint':old_checkpoint})
            service.action({'action':'select_profile','profile':'historical'})
            self.assertEqual(service.status()['scene'],legacy)
            self.assertEqual(service.status()['checkpoint_id'],old_checkpoint['checkpoint_id'])

    def test_unknown_profile_rejected_without_changing_state(self):
        service = GuidedOperations(); self.addCleanup(service.close)
        before = service.status()
        with self.assertRaises(ValueError):service.action({'action':'select_profile','profile':'hardware'})
        self.assertEqual(service.status(),before)
        with self.assertRaises(ValueError):TrainingJobs(profile='hardware')

    def test_invalid_default_checkpoint_leaves_profile_and_evidence_unchanged(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            legacy = root/'baselines/scene-conditioned-20261030'
            legacy.mkdir(parents=True)
            for name in ('preview-scene.json', 'checkpoint.json'):
                (legacy/name).write_bytes((ROOT/'baselines/scene-conditioned-20261030'/name).read_bytes())
            target = root/'baselines/front-bucket-20261030/checkpoint.json'
            target.parent.mkdir(parents=True)
            service = GuidedOperations(root=root, data_root=root/'trials')
            self.addCleanup(service.close)
            service._state.update(plan={'old':True}, bucket_observation={'old':True}, loaded=True)
            before = service.status()
            for invalid in ('{', '[]', '{"schema":"historical"}'):
                target.write_text(invalid)
                with self.assertRaises(ValueError):
                    service.action({'action':'select_profile','profile':'front'})
                self.assertEqual(service.status(), before)

    def test_trainers_have_explicit_separate_profiles(self):
        old = TrainingJobs(); front = TrainingJobs(profile='front')
        self.assertEqual(old.trainer.name,'train_scene_study.py')
        self.assertEqual(front.trainer.name,'run_front_study.py')
        self.assertEqual(front.status()['profile_id'],PROFILE_ID)
        self.assertFalse(front.status()['limits']['hardware_enabled'])

if __name__ == '__main__':unittest.main()
