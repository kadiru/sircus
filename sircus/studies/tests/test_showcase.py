"""Showcase evidence and reproducibility; numerical mocks only, no runtime."""
import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from sircus.studies import build_showcase as demo
from sircus.operations.numerical_worker import evaluate


class ShowcaseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.folder = ROOT / 'sircus/simulation/web/media/demo-reel'
        cls.bundle = json.loads((cls.folder / 'showcase.json').read_text())
        cls.clips = (cls.bundle['fixed']['clips'] + cls.bundle['learning']['same_layout_clips']
                     + [cls.bundle['learning']['success_example']])

    def test_fixed_four_use_same_frozen_action_and_all_predeclared_seeds(self):
        fixed = self.bundle['fixed']
        self.assertEqual([c['ball'] for c in fixed['clips']], list(demo.BALLS))
        self.assertEqual([c['seed'] for c in fixed['clips']], list(demo.FIXED_SEEDS))
        self.assertEqual(fixed['preset']['selection']['new_search_budget'], 0)
        self.assertTrue(fixed['preset']['selection']['frozen_before_releases'])
        self.assertEqual(fixed['trials'], 4)
        for clip in fixed['clips']:
            self.assertEqual(clip['action'], fixed['preset']['action'])
            self.assertEqual(clip['release_action'], fixed['preset']['release_action'])
            self.assertEqual(clip['scene'], fixed['clips'][0]['scene'])
            fresh = evaluate(clip['scene'], clip['release_action'], clip['seed'])['result']
            self.assertEqual(clip['result'], fresh)
            self.assertEqual(clip['role'], 'fixed_preset')
            self.assertEqual(clip['split'], 'demonstration')
            self.assertFalse(clip['multiball_physics'])
        self.assertEqual(fixed['captures'], sum(c['result']['success'] for c in fixed['clips']))

    def test_heldout_learning_preserves_original_results_and_no_selection_disguise(self):
        original = json.loads((demo.BASE / 'report.json').read_text())
        by_id = {r['id']: r for r in original['replays']}
        learning = self.bundle['learning']
        first = json.loads((demo.BASE / 'manifest.json').read_text())['shared']['test'][0]
        self.assertEqual([r['train_scenes'] for r in learning['same_layout_clips']], list(demo.SIZES))
        for clip in learning['same_layout_clips'] + [learning['success_example']]:
            saved = by_id[clip['provenance']['source_replay_id']]
            self.assertEqual(clip['result'], saved['result'])
            self.assertEqual(clip['action'], saved['action'])
            self.assertEqual(clip['checkpoint_id'], saved['checkpoint_id'])
            self.assertEqual(clip['role'], 'frozen_policy')
            self.assertEqual(clip['split'], 'test')
            if clip['paired_layout']:
                self.assertEqual(clip['scene_id'], first['scene_id'])
                self.assertEqual(clip['seed'], first['rollout_seed'])
        self.assertTrue(all(not c['result']['success'] for c in learning['same_layout_clips']))
        self.assertTrue(learning['success_example']['result']['success'])
        self.assertFalse(learning['success_example']['paired_layout'])
        self.assertIn('selected for successful illustration', learning['success_example']['provenance']['selection'])
        self.assertEqual(learning['learning_curve'], original['learning_curve'])
        self.assertEqual(learning['learning_curve_aggregate'], original['learning_curve_aggregate'])
        self.assertEqual(learning['summary'], original['summary'])
        self.assertEqual(learning['training_protocol']['shared_test_layouts'], 48)

    def test_contact_events_ids_and_clocks_are_actual_model_data(self):
        ids = []
        for clip in self.clips:
            release = clip['choreography']['release_time_s']
            times = [e['t'] for e in clip['events']]
            self.assertEqual(times, sorted(times))
            contacts = [e for e in clip['events'] if e['type'] == 'contact']
            self.assertEqual(len(contacts), len(clip['result']['contacts']))
            for event, source in zip(contacts, clip['result']['contacts']):
                self.assertEqual(event['source_contact'], source)
                self.assertEqual(event['contact_object'], source['object'])
                self.assertAlmostEqual(event['t'], release + source['t'])
            self.assertEqual([e['t'] for e in clip['events'] if e['type'] == 'carry_stop'], [release])
            for event in clip['events']:
                ids.append(event['event_id'])
                self.assertEqual(event['trial_id'], clip['id'])
                self.assertEqual(event['evidence'], 'synthetic')
                self.assertFalse(event['physically_verified'])
            outcome = clip['events'][-1]
            self.assertEqual(outcome['type'], 'outcome')
            self.assertEqual(outcome['outcome'], clip['result']['outcome'])
            self.assertEqual(outcome['success'], clip['result']['success'])
            self.assertAlmostEqual(outcome['t'], clip['duration_s'])
        self.assertEqual(len(ids), len(set(ids)))
        fake = copy.deepcopy(self.clips[0]['result'])
        fake.update(outcome='model_limit', success=False)
        self.assertFalse(demo.timeline_events('unknown-example', fake, 4.35)[-1]['audible'])

    def test_geometry_and_choreography_do_not_claim_calibration_or_native_paths(self):
        for clip in self.clips:
            self.assertEqual(clip['scene']['board']['size_mm'], [200., 300., 5.])
            self.assertEqual(clip['scene']['bin']['reported_height_mm'], 175.)
            self.assertEqual(clip['scene']['bin']['height_interpretation'], 'unresolved')
            self.assertFalse(clip['board_pose']['calibrated'])
            self.assertFalse(clip['native_paths_checked'])
            self.assertFalse(clip['physically_verified'])
            sequence = clip['choreography']
            self.assertEqual(sequence['schema'], 'sircus-numerical-shot-choreography-v1')
            self.assertTrue(sequence['synthetic'])
            self.assertFalse(sequence['native_paths_checked'])
            settle = next(s for s in sequence['segments'] if s['stage'] == 'settle')
            self.assertAlmostEqual(settle['end_s'] - settle['start_s'], 2.)
            numerical = [f for f in sequence['ball_frames'] if f['source'] == 'versioned_numerical_ball_physics']
            self.assertEqual(len(numerical), len(clip['result']['frames']))
            for frame, raw in zip(numerical, clip['result']['frames']):
                self.assertAlmostEqual(frame['t'], sequence['release_time_s'] + raw['t'])
                self.assertEqual(frame['ball_xyz_mm'], raw['ball_xyz_mm'])
            self.assertEqual(len(clip['board_pose']['joints_deg']), 6)
            self.assertEqual(len(clip['arm10_release_joints_deg']), 6)
            self.assertTrue(clip['arm10_release_boxes'])
            self.assertLess(clip['board_pose']['reconstruction_position_error_mm'], 1.)

    def test_manifest_covers_sources_clips_and_file(self):
        for source in self.bundle['manifest']['sources']:
            path = ROOT / source['path']
            self.assertTrue(path.resolve().is_relative_to(ROOT.resolve()))
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), source['sha256'])
        expected = {r['id']: r['sha256'] for r in self.bundle['manifest']['clips']}
        for clip in self.clips:
            value = copy.deepcopy(clip)
            digest = value.pop('artifact_sha256')
            self.assertEqual(demo.digest(value), digest)
            self.assertEqual(expected[clip['id']], digest)
        sha, name = (self.folder / 'SHA256SUMS').read_text().strip().split('  ', 1)
        self.assertEqual(name, 'showcase.json')
        self.assertEqual(sha, hashlib.sha256((self.folder / name).read_bytes()).hexdigest())
        demo.verify_baseline()

    def test_rebuild_identical_without_training_search_or_changing_baselines(self):
        with patch.object(demo.front_study, 'search_episode', side_effect=AssertionError('No teacher search')), \
             patch.object(demo.front_study, 'fit_model', side_effect=AssertionError('No model fit')):
            self.assertEqual(demo.build_bundle(), self.bundle)
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaisesRegex(ValueError, 'never overwritten'):
                demo.write_bundle(folder)


if __name__ == '__main__':
    unittest.main()
