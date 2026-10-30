"""Determinism, split leakage, frozen inference and compatibility regressions."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import scene_study as study
import train_scene_study as trainer
from shot_rollout import rollout

BASE = json.loads((Path(__file__).parent/'scene.json').read_text())


def make_checkpoint():
    manifest = study.sample_splits(BASE, 72, 6, 2, 2)
    x = [e['observation'] for e in manifest['splits']['train']]
    y = [[(low+high)/2 + .05*(row[0]-600) for low, high in study.ACTION_BOUNDS] for row in x]
    model = study.fit_ridge(x, y, 1.)
    value = study.checkpoint(BASE, manifest, model,
        training_scene_ids=[e['scene_id'] for e in manifest['splits']['train']],
        validation_selection={'test_used_for_selection': False})
    return manifest, value


class SceneStudyTests(unittest.TestCase):
    def test_seeded_splits_are_repeatable_disjoint_and_observations_only_geometry(self):
        first = study.sample_splits(BASE, 42, 6, 3, 4)
        self.assertEqual(first, study.sample_splits(BASE, 42, 6, 3, 4))
        episodes = [e for rows in first['splits'].values() for e in rows]
        self.assertEqual(len({e['layout_sha256'] for e in episodes}), 13)
        self.assertEqual(len({e['scene_id'] for e in episodes}), 13)
        self.assertTrue(all(e['observation'] == study.observation(e['scene']) for e in episodes))
        self.assertEqual(BASE['board']['center_mm'], [600., 0., 350.])
        for e in episodes:
            self.assertNotEqual(e['action_rng_seed'], e['rollout_seed'])
            self.assertLessEqual(e['rollout_seed'], 2**32-1)

    def test_real_rollout_accepts_derived_uint32_seed(self):
        episode = study.sample_splits(BASE, 42, 4, 1, 1)['splits']['test'][0]
        scene, action, _ = study.action_scene(episode['scene'], study.FIXED_ACTION)
        result = rollout(scene, action, seed=episode['rollout_seed'])
        self.assertTrue(result['executed'], result)
        self.assertNotEqual(result['outcome'], 'invalid_action')
        self.assertEqual(result['scoring_version'], study.SCORING_VERSION)

    def test_manifest_rejects_overlap_even_with_recomputed_checksum(self):
        value = study.sample_splits(BASE, 42, 4, 1, 1)
        replacement = copy.deepcopy(value['splits']['train'][0])
        replacement['split'] = 'test'
        value['splits']['test'][0] = replacement
        value.pop('manifest_sha256'); value['manifest_sha256'] = study.digest(value)
        with self.assertRaisesRegex(ValueError, 'disjoint'):
            study.validate_manifest(value)

    def test_five_action_dimensions_preserve_layout_and_explicit_coordinates(self):
        action = {name: (low+high)/2 for name, (low, high) in zip(study.ACTION_NAMES, study.ACTION_BOUNDS)}
        selected, release, coords = study.action_scene(BASE, action)
        self.assertEqual(selected['board']['center_mm'], BASE['board']['center_mm'])
        self.assertEqual(selected['board']['rpy_deg'][2], BASE['board']['rpy_deg'][2])
        self.assertEqual(selected['bin'], BASE['bin'])
        self.assertEqual(set(release), {'u_mm', 'v_mm', 'gap_mm'})
        self.assertEqual(coords['gap_mm'], action['gap_mm'])
        self.assertGreater(coords['h_m'], 0)
        for key in study.ACTION_NAMES:
            bad = copy.deepcopy(action); bad[key] = float('nan')
            with self.assertRaises(ValueError): study.action_scene(BASE, bad)

    def test_fit_is_actual_scene_conditioned_regression(self):
        layouts = study.sample_splits(BASE, 53, 8, 1, 1)['splits']['train']
        x = np.array([e['observation'] for e in layouts])
        y = np.array([[0., 30.+(row[0]-600)/10, row[0]-600, -20., 150.+row[2]-350] for row in x])
        model = study.fit_ridge(x, y, .01)
        self.assertTrue(np.allclose(model['observation_mean'], x.mean(axis=0)))
        self.assertGreater(np.linalg.norm(np.asarray(model['weights'])[1:]), 0)
        a = copy.deepcopy(BASE); b = copy.deepcopy(BASE); b['board']['center_mm'][0] += 5
        self.assertNotEqual(study.infer_model(model, a), study.infer_model(model, b))

    def test_predict_has_no_search_rollout_or_ik_and_roundtrip_matches(self):
        manifest, value = make_checkpoint()
        scene = manifest['splits']['test'][0]['scene']
        with patch.object(study, 'search_episode', side_effect=AssertionError('optimizer in inference')), \
                patch.object(study, 'rollout', side_effect=AssertionError('rollout in inference')), \
                patch.object(study, 'feasibility', side_effect=AssertionError('IK in inference')):
            a = study.predict(value, scene)
            b = study.predict(json.loads(json.dumps(value)), json.loads(json.dumps(scene)))
        np.testing.assert_allclose(list(a['action'].values()), list(b['action'].values()), atol=1e-12, rtol=0)
        self.assertEqual(a['checkpoint_id'], b['checkpoint_id'])

    def test_unseen_in_domain_layout_allowed_assets_and_bounds_rejected(self):
        _, value = make_checkpoint()
        scene = copy.deepcopy(BASE); scene['board']['center_mm'][0] += 1.234
        study.predict(value, scene)
        bad = copy.deepcopy(scene); bad['ball']['mass_kg'] *= 1.1
        with self.assertRaisesRegex(ValueError, 'asset'): study.predict(value, bad)
        bad = copy.deepcopy(scene); bad['board']['center_mm'][0] = 800.
        with self.assertRaisesRegex(ValueError, 'distribution'): study.predict(value, bad)
        bad = copy.deepcopy(scene); bad['bin']['rpy_deg'] = [0, 0, 0]
        with self.assertRaises(ValueError): study.predict(value, bad)

    def test_checkpoint_corruption_source_and_split_guards(self):
        _, value = make_checkpoint()
        bad = copy.deepcopy(value); bad['model']['weights'][0][0] += .1
        with self.assertRaisesRegex(ValueError, 'checksum'): study.validate_checkpoint(bad)
        with patch.object(study, 'study_contract', return_value={'changed': True}):
            with self.assertRaisesRegex(ValueError, 'source'): study.validate_checkpoint(value)
        bad = copy.deepcopy(value)
        bad['training_scene_ids'].append(bad['split_layouts']['test'][0]['scene_id'])
        bad.pop('checkpoint_id'); bad['checkpoint_id'] = study.digest(bad)
        with self.assertRaisesRegex(ValueError, 'Held-out'): study.validate_checkpoint(bad)

    def test_teacher_rejects_held_out_layouts(self):
        manifest, _ = make_checkpoint()
        for split in ('validation', 'test'):
            with self.assertRaisesRegex(ValueError, 'held-out'):
                study.search_episode(manifest['splits'][split][0], 1)

    def test_invalid_episode_stays_in_denominator(self):
        row = {'result': study.failure_result('invalid_action', 'test'), 'feasibility': None}
        summary = study.summarize([row, row])
        self.assertEqual(summary['episodes'], 2)
        self.assertEqual(summary['invalid_actions'], 2)
        self.assertEqual(summary['success_fraction'], 0.)

    def test_training_freezes_before_test_and_fits_train_only(self):
        seen_test = []
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)/'study'
            def fake(scene, action, seed=0, frames=False):
                if scene['scene_id'].startswith('test-'):
                    self.assertTrue((output/'checkpoint.json').exists())
                    seen_test.append((output/'checkpoint.json').read_bytes())
                # Synthetic test double: does not stand in for measured/numerical success.
                result = {'executed': True, 'success': False, 'outcome': 'no_entry',
                    'shaping_reward': -abs(action['pitch_deg']-32.), 'capture_class': None,
                    'signed_miss_xy_mm': [10., 20.], 'scoring_version': study.SCORING_VERSION,
                    'physics_version': study.PHYSICS_VERSION}
                return {'action': copy.deepcopy(action), 'selected_scene': copy.deepcopy(scene),
                    'result': result, 'feasibility': {'arm_reachable': True, 'board_arm_reachable': True}}
            with patch.object(study, 'evaluate_action', side_effect=fake), patch.object(trainer, 'progress'):
                value, summary = trainer.run(BASE, output, seed=31, train_scenes=4,
                    validation_scenes=2, test_scenes=2, teacher_budget=2, oracle_budget=1, alphas=(1.,))
            manifest = json.loads((output/'episodes.json').read_text())
            x = np.array([e['observation'] for e in manifest['splits']['train']])
            np.testing.assert_allclose(value['model']['observation_mean'], x.mean(axis=0))
            self.assertTrue(seen_test)
            self.assertTrue(all(x == (output/'checkpoint.json').read_bytes() for x in seen_test))
            self.assertEqual(summary['test']['frozen_policy']['episodes'], 2)
            self.assertEqual(summary['test']['held_out_search_oracle']['episodes'], 2)
            self.assertEqual(summary['teacher_labels'], 4)


if __name__ == '__main__':
    unittest.main()
