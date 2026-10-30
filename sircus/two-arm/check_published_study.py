"""Verify the portable synthetic checkpoint and one real numerical replay.

No Viam client, service, physical robot or native-path replay is used. The
trajectory comparison allows 1e-4 mm for cross-platform floating-point rounding.
"""
import hashlib
import json
from pathlib import Path
import unittest

import numpy as np
import scene_study as study

BASELINE = Path(__file__).resolve().parents[2]/'baselines/scene-conditioned-20261030'


def read(name):
    return json.loads((BASELINE/name).read_text())


class PublishedStudyTests(unittest.TestCase):
    def test_checksums_cover_every_published_json(self):
        declared = {}
        for line in (BASELINE/'SHA256SUMS').read_text().splitlines():
            digest, name = line.split('  ', 1)
            self.assertEqual(Path(name).name, name)
            self.assertNotIn(name, declared)
            path = BASELINE/name
            self.assertFalse(path.is_symlink())
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), digest, name)
            declared[name] = digest
        self.assertEqual(set(declared), {p.name for p in BASELINE.glob('*.json')})
        self.assertTrue({'checkpoint.json', 'episodes.json', 'experiment.json',
            'summary.json', 'test-scorecard.json', 'preview-scene.json', 'preview.json'}.issubset(declared))

    def test_checkpoint_manifest_provenance_and_train_only_statistics(self):
        value = study.validate_checkpoint(read('checkpoint.json'))
        manifest = study.validate_manifest(read('episodes.json'))
        summary = read('summary.json')
        self.assertEqual(value['manifest_sha256'], manifest['manifest_sha256'])
        self.assertEqual(summary['manifest_sha256'], manifest['manifest_sha256'])
        self.assertEqual(value['checkpoint_id'], summary['checkpoint_id'])
        self.assertEqual(value['contract'], manifest['contract'])
        self.assertEqual(hashlib.sha256((BASELINE/'checkpoint.json').read_bytes()).hexdigest(),
                         summary['checkpoint_file_sha256'])
        for artifact in (value, manifest, summary, read('test-scorecard.json'), read('preview.json')):
            self.assertEqual(artifact['data_origin'], 'synthetic')
            self.assertIs(artifact['physically_verified'], False)
            self.assertIs(artifact.get('native_paths_checked', False), False)
        for split, key in [('train', 'train_scenes'), ('validation', 'validation_scenes'), ('test', 'test_scenes')]:
            self.assertEqual(len(manifest['splits'][split]), summary[key])
            self.assertEqual(value['split_layouts'][split],
                [{k: e[k] for k in ('scene_id', 'layout_sha256')} for e in manifest['splits'][split]])
        labels = set(value['training_scene_ids'])
        training = [e['observation'] for e in manifest['splits']['train'] if e['scene_id'] in labels]
        self.assertEqual(len(training), summary['teacher_labels'])
        self.assertEqual(summary['teacher_labels']+summary['teacher_failed_scenes'], summary['train_scenes'])
        np.testing.assert_allclose(value['model']['observation_mean'], np.mean(training, axis=0), atol=1e-10, rtol=0)
        expected_scale = np.std(training, axis=0)
        expected_scale = np.where(expected_scale < 1e-9, 1., expected_scale)
        np.testing.assert_allclose(value['model']['observation_scale'], expected_scale, atol=1e-10, rtol=0)
        self.assertIs(value['validation_selection']['test_used_for_selection'], False)
        self.assertEqual(value['validation_selection']['scene_ids'],
                         [e['scene_id'] for e in manifest['splits']['validation']])
        self.assertIs(summary['checkpoint_unchanged_after_test'], True)
        self.assertIs(summary['split_manifest_unchanged_after_test'], True)
        self.assertEqual(summary['scoring_version'], study.SCORING_VERSION)
        self.assertEqual(summary['physics_version'], study.PHYSICS_VERSION)

    def test_scorecard_totals_same_layouts_seeds_and_capture_evidence(self):
        value = read('checkpoint.json'); manifest = read('episodes.json')
        card = read('test-scorecard.json'); summary = read('summary.json')
        episodes = {e['scene_id']: e for e in manifest['splits']['test']}
        self.assertEqual(card['checkpoint_id'], value['checkpoint_id'])
        self.assertEqual(set(card['methods']), {'frozen_policy', 'fixed_baseline', 'held_out_search_oracle'})
        roles = {'frozen_policy': 'frozen_ridge_prediction', 'fixed_baseline': 'fixed_action_baseline',
                 'held_out_search_oracle': 'held_out_oracle'}
        for method, rows in card['methods'].items():
            self.assertEqual(len(rows), len(episodes))
            self.assertEqual({row['scene_id'] for row in rows}, set(episodes))
            outcomes = {}
            for row in rows:
                episode = episodes[row['scene_id']]
                self.assertEqual(row['layout_sha256'], episode['layout_sha256'])
                self.assertEqual(row['seed'], episode['rollout_seed'])
                self.assertEqual(row['role'], roles[method])
                self.assertEqual(row['scoring_version'], study.SCORING_VERSION)
                self.assertEqual(row['physics_version'], study.PHYSICS_VERSION)
                self.assertEqual(row['data_origin'], 'synthetic')
                self.assertIs(row['physically_verified'], False)
                self.assertIs(row['native_paths_checked'], False)
                if method == 'frozen_policy':
                    predicted = study.predict(value, episode['scene'])
                    np.testing.assert_allclose([predicted['action'][k] for k in study.ACTION_NAMES],
                        [row['action'][k] for k in study.ACTION_NAMES], atol=1e-10, rtol=0)
                elif method == 'fixed_baseline':
                    self.assertEqual(row['action'], study.FIXED_ACTION)
                _, release, coordinates = study.action_scene(episode['scene'], row['action'])
                self.assertEqual(release, row['release_action'])
                np.testing.assert_allclose(coordinates['commanded_release_ball_center_mm'],
                    row['release_coordinates']['commanded_release_ball_center_mm'], atol=1e-8, rtol=0)
                result = row['result']; outcomes[result['outcome']] = outcomes.get(result['outcome'], 0)+1
                if result['success']:
                    self.assertEqual(result['outcome'], 'captured')
                    self.assertTrue(result['valid_board_bounce'])
                    self.assertTrue(result['whole_ball_entry'])
                    self.assertTrue(result['bucket_retained_1s'])
                    self.assertGreaterEqual(result['current_containment_s'], 1.-1e-9)
                    self.assertGreaterEqual(result['max_continuous_containment_s'], 1.-1e-9)
                    self.assertIsNone(result['model_limit'])
                    self.assertIn(result['capture_class'], ('clean', 'rim_assisted'))
            counts = summary['test'][method]
            self.assertEqual(counts['episodes'], len(rows))
            self.assertEqual(counts['captures'], sum(r['result']['success'] is True for r in rows))
            self.assertAlmostEqual(counts['success_fraction'], counts['captures']/len(rows))
            self.assertEqual(counts['outcomes'], outcomes)
            self.assertEqual(counts['invalid_actions'], sum(r['result']['outcome'] == 'invalid_action' for r in rows))
            self.assertEqual(counts['model_limits'], sum(r['result']['outcome'] == 'model_limit' for r in rows))
            self.assertEqual(counts['endpoint_feasible'], sum(bool(r['endpoint_feasibility'] and
                r['endpoint_feasibility']['arm_reachable'] and r['endpoint_feasibility']['board_arm_reachable']) for r in rows))
            self.assertEqual(counts['capture_classes'], {kind: sum(r['result']['capture_class'] == kind for r in rows)
                                                       for kind in ('clean', 'rim_assisted')})

    def test_actual_frozen_preview_reproduces_first_test(self):
        value = read('checkpoint.json'); manifest = read('episodes.json')
        episode = read('preview-scene.json'); saved = read('preview.json')
        first = manifest['splits']['test'][0]
        self.assertEqual(episode['scene'], first['scene'])
        self.assertEqual(episode['seed'], first['rollout_seed'])
        self.assertEqual(episode['scene_id'], first['scene_id'])
        self.assertEqual(episode['layout_sha256'], first['layout_sha256'])
        score = next(r for r in read('test-scorecard.json')['methods']['frozen_policy'] if r['scene_id'] == first['scene_id'])
        current = study.preview(value, episode['scene'], episode['seed'])
        for expected in (saved['action'], score['action']):
            np.testing.assert_allclose([current['action'][k] for k in study.ACTION_NAMES],
                                      [expected[k] for k in study.ACTION_NAMES], atol=1e-10, rtol=0)
        self.assertEqual(current['mode'], 'numerical_prediction')
        self.assertIn('blocked', current['native_replay_status'])
        self.assertIs(current['native_paths_checked'], False)
        self.assertIs(current['physically_verified'], False)
        for key in ('success', 'outcome', 'capture_class', 'valid_board_bounce', 'whole_ball_entry', 'bucket_retained_1s'):
            self.assertEqual(current['result'][key], saved['result'][key], key)
            self.assertEqual(current['result'][key], score['result'][key], key)
        self.assertEqual(len(current['result']['frames']), len(saved['result']['frames']))
        for actual, expected in zip(current['result']['frames'], saved['result']['frames']):
            self.assertAlmostEqual(actual['t'], expected['t'], places=8)
            np.testing.assert_allclose(actual['ball_xyz_mm'], expected['ball_xyz_mm'], atol=1e-4, rtol=0)
            np.testing.assert_allclose(actual['board_center_mm'], expected['board_center_mm'], atol=1e-4, rtol=0)
        self.assertTrue(current['feasibility']['arm_reachable'])
        self.assertTrue(current['feasibility']['board_arm_reachable'])


if __name__ == '__main__':
    unittest.main()
