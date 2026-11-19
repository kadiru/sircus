"""Synthetic logic controls; no native rollout, material fit or historical edit."""
import copy
import hashlib
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np

from sircus.mjlab import calibrated_score as old
from sircus.mjlab import table_assisted_score as primary
from sircus.mjlab import table_assisted_score_oracle as oracle
from sircus.mjlab.tests.test_calibrated_score import config, trace, truncate


class TableAssistedControls(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = Path(__file__).resolve().parents[3]
        cls.scene = json.loads((cls.root / 'sircus/isaac/scene.json').read_text())
        cls.cfg = config(cls.scene)
        cls.xy = cls.scene['bucket']['center_xy_m']
        cls.normal = [0., 0., 1.]

    def score(self, arrays):
        inputs = copy.deepcopy(arrays)
        a = primary.score_trace(self.scene, self.xy, self.normal, self.cfg, arrays)
        b = oracle.score_trace(self.scene, self.xy, self.normal, self.cfg, arrays)
        self.assertEqual(set(a), set(b))
        for key in a:
            if isinstance(a[key], float):
                self.assertAlmostEqual(a[key], b[key], delta=1e-9, msg=key)
            else:
                self.assertEqual(a[key], b[key], key)
        self.assertAlmostEqual(a['reward'], primary.reward(a), delta=1e-12)
        self.assertAlmostEqual(a['reward'], oracle.reward(a), delta=1e-12)
        for key in arrays:
            np.testing.assert_array_equal(arrays[key], inputs[key], err_msg=key)
        return a

    def test_direct_capture_is_unchanged_apart_from_versions_and_new_diagnostics(self):
        arrays = trace(self.scene)
        before = old.score_trace(self.scene, self.xy, self.normal, self.cfg, arrays)
        after = self.score(arrays)
        for key, value in before.items():
            if key not in ('scoring_version', 'reward_version', 'route_version'):
                self.assertEqual(after[key], value, key)
        self.assertTrue(after['direct_capture'])
        self.assertFalse(after['table_assisted_capture'])
        self.assertEqual(after['route_class'], 'direct')
        self.assertEqual(after['forbidden_contact_steps'], 0)

    def test_table_contact_before_or_after_board_counts_without_table_penalty(self):
        # These are synthetic event controls, not physically simulated bounces.
        for tick in (25, 100, 500):
            with self.subTest(tick=tick):
                arrays = trace(self.scene)
                arrays['forces_N'][tick, 2] = 1.
                before = old.score_trace(self.scene, self.xy, self.normal, self.cfg, arrays)
                after = self.score(arrays)
                self.assertFalse(before['route_success'])
                self.assertTrue(after['route_success'])
                self.assertTrue(after['complete_success'])
                self.assertTrue(after['table_assisted_capture'])
                self.assertFalse(after['direct_capture'])
                self.assertEqual(after['route_class'], 'table_assisted')
                self.assertEqual(after['external_contact_steps'], 1)
                self.assertEqual(after['table_contact_steps'], 1)
                self.assertEqual(after['forbidden_contact_steps'], 0)
                self.assertAlmostEqual(before['reward'], .15)
                self.assertAlmostEqual(after['reward'], 4.55)
                self.assertEqual(after['capture_time_s'], before['capture_time_s'])

    def test_every_forbidden_channel_blocks_bonus_even_with_table_retention(self):
        for channel in (3, 4, 5):
            with self.subTest(channel=channel):
                arrays = trace(self.scene)
                arrays['forces_N'][100, 2] = 1.
                arrays['forces_N'][500, channel] = 1.
                out = self.score(arrays)
                self.assertTrue(out['success'])
                self.assertFalse(out['route_success'])
                self.assertFalse(out['complete_success'])
                self.assertEqual(out['route_class'], 'not_captured')
                self.assertFalse(out['direct_capture'])
                self.assertFalse(out['table_assisted_capture'])
                self.assertEqual(out['forbidden_contact_steps'], 1)
                self.assertEqual(out['external_contact_steps'], 2)
                self.assertEqual(out['route_quality'], 0)
                self.assertAlmostEqual(out['reward'], .15)

    def test_contact_counts_are_interval_unions_and_threshold_is_strict(self):
        arrays = trace(self.scene)
        arrays['forces_N'][500, 2:] = 1.
        arrays['forces_N'][501, 2:] = .01
        arrays['forces_N'][502, 2] = np.nextafter(.01, 1.)
        out = self.score(arrays)
        self.assertEqual(out['forbidden_contact_steps'], 1)
        self.assertEqual(out['external_contact_steps'], 2)
        self.assertEqual(out['table_contact_steps'], 2)

    def test_partial_exit_resets_full_continuous_second(self):
        arrays = trace(self.scene, end=2.2)
        arrays['forces_N'][100, 2] = 1.
        bucket = self.scene['bucket']
        rim = bucket['assumed_bottom_thickness_m'] + bucket['assumed_interior_depth_m']
        radius = self.scene['ball']['radius_m']
        arrays['position_m'][(arrays['time_s'] >= .9) & (arrays['time_s'] < .95), 2] = rim - radius + .001
        out = self.score(arrays)
        self.assertTrue(out['route_success'])
        self.assertTrue(out['table_assisted_capture'])
        self.assertAlmostEqual(out['capture_time_s'], 1.95)
        self.assertGreaterEqual(out['retention_resets'], 1)
        self.assertFalse(self.score(truncate(arrays, 1.949))['route_success'])

    def test_table_contact_does_not_relax_one_second_requirement(self):
        arrays = trace(self.scene)
        arrays['forces_N'][100, 2] = 1.
        self.assertFalse(self.score(truncate(arrays, 1.199))['route_success'])
        self.assertTrue(self.score(truncate(arrays, 1.2))['route_success'])

    def test_post_terminal_contacts_and_bad_physics_do_not_revoke_capture(self):
        arrays = trace(self.scene)
        arrays['forces_N'][100, 2] = 1.
        arrays['forces_N'][1200:, 2:] = 10.
        arrays['position_m'][arrays['time_s'] >= 1.3, 2] = 1.
        arrays['velocity_m_s'][arrays['time_s'] >= 1.3] = 100.
        out = self.score(arrays)
        self.assertTrue(out['route_success'])
        self.assertEqual(out['physics_steps'], 1200)
        self.assertFalse(out['invalid_physics'])
        self.assertEqual(out['forbidden_contact_steps'], 0)
        self.assertEqual(out['table_contact_steps'], 1)
        arrays['forces_N'][1199, 4] = 1.
        self.assertFalse(self.score(arrays)['route_success'])

    def test_table_route_and_bucket_mouth_quality_are_independent(self):
        arrays = trace(self.scene)
        arrays['forces_N'][100, 2] = 1.
        clean = self.score(arrays)
        self.assertEqual(clean['route_quality'], 2)
        self.assertEqual(clean['entry_class'], 'clean_capture')
        t = arrays['time_s']
        arrays['position_m'][:, 0] += np.where(t <= .2, .05, np.maximum(0, .05 - (t - .2) * .25))
        arrays['forces_N'][199, 1] = 2.
        assisted = self.score(arrays)
        self.assertTrue(assisted['table_assisted_capture'])
        self.assertEqual(assisted['route_quality'], 1)
        self.assertTrue(assisted['rim_band_contact'])

    def test_single_board_episode_is_not_single_total_physical_bounce(self):
        arrays = trace(self.scene)
        arrays['forces_N'][100, 2] = 1.
        out = self.score(arrays)
        self.assertTrue(out['single_board_bounce_success'])
        self.assertTrue(out['single_bounce_success'])
        self.assertIn('not one total physical bounce', out['single_bounce_scope'])
        arrays['forces_N'][120, 0] = 1.
        out = self.score(arrays)
        self.assertTrue(out['route_success'])
        self.assertFalse(out['single_board_bounce_success'])
        self.assertFalse(out['single_bounce_success'])
        self.assertEqual(out['board_contact_episodes'], 2)

    def test_invalid_physics_and_missing_board_rebound_still_reject(self):
        arrays = trace(self.scene)
        arrays['forces_N'][100, 2] = 1.
        arrays['velocity_m_s'][1200, 0] = 6.3
        invalid = self.score(arrays)
        self.assertTrue(invalid['invalid_physics'])
        self.assertFalse(invalid['route_success'])
        self.assertEqual(invalid['reward'], -1.)
        arrays = trace(self.scene)
        arrays['forces_N'][:, 0] = 0.
        arrays['forces_N'][100, 2] = 1.
        out = self.score(arrays)
        self.assertFalse(out['board_bounce'])
        self.assertFalse(out['whole_ball_entry'])
        self.assertFalse(out['route_success'])

    def test_reward_requires_explicit_opt_in_but_rejections_need_no_fake_trace(self):
        historical = old.score_trace(self.scene, self.xy, self.normal, self.cfg, trace(self.scene))
        for implementation in (primary, oracle):
            with self.assertRaises(ValueError):
                implementation.reward(historical)
            self.assertEqual(implementation.reward({'invalid': 'setup_rejected'}), -1.)

    def test_oracle_does_not_call_production_route_or_containment(self):
        arrays = trace(self.scene)
        arrays['forces_N'][100, 2] = 1.
        with patch.object(primary, 'score_trace', side_effect=AssertionError('production route called')), \
             patch.object(old, 'score_trace', side_effect=AssertionError('production containment called')):
            result = oracle.score_trace(self.scene, self.xy, self.normal, self.cfg, arrays)
        self.assertTrue(result['table_assisted_capture'])

    def test_historical_v2_source_bytes_remain_untouched(self):
        expected = {
            'calibrated_score.py': 'b90fb4886f2297534b81b399620c647952eb2e486217289c65548713977e3539',
            'calibrated_score_oracle.py': '74ff1b43d342a28e7b4af6d3ec757f577aec076ac286535f4c16e053e31a00df',
        }
        for name, value in expected.items():
            self.assertEqual(hashlib.sha256((self.root / 'sircus/mjlab' / name).read_bytes()).hexdigest(), value)


if __name__ == '__main__':
    unittest.main()
