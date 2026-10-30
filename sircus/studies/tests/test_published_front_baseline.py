"""Portable evidence regression; never creates or connects to a runtime."""
import hashlib
import json
from pathlib import Path
import sys
import unittest
import numpy as np

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from sircus.studies import front_study as study
BASE=ROOT/'baselines/front-bucket-20261030'


@unittest.skipUnless((BASE/'report.json').exists(),'Front baseline not yet published')
class PublishedFrontBaselineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report=json.loads((BASE/'report.json').read_text())
        cls.manifest=json.loads((BASE/'manifest.json').read_text())
        cls.checkpoint=json.loads((BASE/'checkpoint.json').read_text())

    def test_artifact_checksums_and_source_contract(self):
        listed=set()
        for line in (BASE/'SHA256SUMS').read_text().splitlines():
            digest,name=line.split('  ',1);path=BASE/name
            self.assertTrue(path.resolve().is_relative_to(BASE.resolve()))
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(),digest,name);listed.add(name)
        actual={str(p.relative_to(BASE)) for p in BASE.rglob('*') if p.is_file() and p.name!='SHA256SUMS'}
        self.assertEqual(listed,actual)
        study.validate_checkpoint(self.checkpoint)
        for path in (BASE/'checkpoints').glob('*.json'):study.validate_checkpoint(json.loads(path.read_text()))
        study.validate_checkpoint(json.loads((BASE/'initial-24-6-8/checkpoint.json').read_text()))
        self.assertLess((BASE/'report.json').stat().st_size,32*1024*1024)

    def test_final_summaries_and_extensions_are_consistent(self):
        self.assertEqual(json.loads((BASE/'summary.json').read_text()),self.report['summary'])
        self.assertEqual(json.loads((BASE/'challenge-summary.json').read_text()),self.report['challenges'])
        schemas={e['schema'] for e in self.report['extensions']}
        self.assertTrue({'sircus-contact-intervention-v1','sircus-post-study-diagnostics-v1'}.issubset(schemas))
        for row in self.report['replays']:self.assertIn('in_distribution',row['scope'])

    def test_default_is_prespecified_not_best_test(self):
        first_seed=self.manifest['seeds'][0];n=max(self.manifest['nested_sizes'])
        expected=json.loads((BASE/'checkpoints'/f'seed-{first_seed}-checkpoint-n{n}.json').read_text())
        self.assertEqual(expected,self.checkpoint)
        self.assertEqual(self.report['summary']['default_checkpoint_id'],expected['checkpoint_id'])
        self.assertFalse(self.report['physically_verified']);self.assertFalse(self.report['native_paths_checked'])
        self.assertEqual(self.checkpoint['base_scene']['bin']['height_interpretation'],'unresolved')

    def test_saved_actions_equal_direct_frozen_predictions(self):
        episodes={e['scene_id']:e for e in self.manifest['shared']['test']}
        curve={(r['seed'],r['train_scenes']):r for r in self.report['learning_curve']}
        for group in json.loads((BASE/'test-scorecard.json').read_text()):
            seed,n=group['seed'],group['train_scenes']
            checkpoint=json.loads((BASE/'checkpoints'/f'seed-{seed}-checkpoint-n{n}.json').read_text())
            self.assertEqual(set(r['scene_id'] for r in group['rows']),set(episodes))
            for row in group['rows']:
                predicted=study.predict(checkpoint,episodes[row['scene_id']]['scene'])['action']
                np.testing.assert_allclose([predicted[k] for k in study.ACTION_NAMES],[row['action'][k] for k in study.ACTION_NAMES],rtol=0,atol=1e-10)
                self.assertEqual(row['checkpoint_id'],checkpoint['checkpoint_id'])
                self.assertEqual(row['role'],'frozen_policy')
            self.assertEqual(sum(r['result']['success'] for r in group['rows']),curve[(seed,n)]['captures'])
            self.assertEqual(len(group['rows']),curve[(seed,n)]['episodes'])

    def test_saved_numerical_replay_reproduces(self):
        # First saved core held-out replay, not an intervention or search oracle.
        saved=next(r for r in self.report['replays'] if r['split']=='test' and r['role']=='frozen_policy')
        episode=next(e for e in self.manifest['shared']['test'] if e['scene_id']==saved['scene_id'])
        predicted=study.preview(self.checkpoint,episode['scene'],episode['rollout_seed'])
        self.assertEqual(saved['result']['outcome'],predicted['result']['outcome'])
        self.assertEqual(saved['result']['success'],predicted['result']['success'])
        old=saved['result']['frames'];fresh=predicted['result']['frames'];self.assertEqual(len(old),len(fresh))
        for a,b in zip(old,fresh):
            self.assertAlmostEqual(a['t'],b['t'],places=8)
            np.testing.assert_allclose(a['ball_xyz_mm'],b['ball_xyz_mm'],atol=.0001,rtol=0)
        for replay in self.report['replays']:
            if replay['role']=='forced_contact_intervention':
                for key in ('roll_deg','pitch_deg','gap_mm'):
                    self.assertEqual(replay['action'][key],replay['predicted_action'][key])


if __name__=='__main__':unittest.main()
