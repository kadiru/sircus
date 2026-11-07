"""Analytical controls for requested-origin flight ranking; no engine or GPU."""
import copy
import json
import math
from pathlib import Path
import subprocess
import sys
import unittest

import numpy as np

from sircus.mjlab import consistent_flight_selector as flight
from sircus.mjlab.tests.test_calibrated_score import config, resign


ROOT = Path(__file__).resolve().parents[3]


def prediction(velocity=(1., 0., .5), score=.99):
    return dict(member_raw_output=[[9., 9., 9., 9., 9., 0., *velocity] for _ in range(5)],
        ranking_score=score, capture_probability=.999999)


class ConsistentFlightTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.scene = json.loads((ROOT/'sircus/isaac/scene.json').read_text())
        cls.calibration = config(cls.scene)
        cls.geom = flight.geometry(cls.scene, cls.calibration)
        cls.radius = cls.scene['ball']['radius_m']
        cls.rim = .183

    def test_rim_uses_unmodified_facets_and_preserves_provisional_measurements(self):
        g = self.geom
        self.assertAlmostEqual(g['rim_m'], self.rim)
        self.assertEqual(g['calibration_fingerprint'], self.calibration['fingerprint'])
        self.assertEqual(g['bucket_reported_height_m'], .175)
        self.assertIn('unresolved', g['bucket_height_interpretation'])
        planes = np.asarray(g['inward_rim_halfplanes'])
        np.testing.assert_allclose(np.linalg.norm(planes[:, :2], axis=1), 1.)
        np.testing.assert_allclose(planes[:, 2], .177/2*math.cos(math.pi/64), atol=1e-12)

    def test_geometry_rejects_tampered_compiled_facets_or_nonvertical_gravity(self):
        changed = copy.deepcopy(self.calibration)
        changed['compiled_planes'][0][3] += .01
        for cfg in (changed, resign(changed)):
            with self.assertRaises(ValueError): flight.geometry(self.scene, cfg)
        for gravity in ([.001, 0, -9.81], [0, 0, 0], [0, 0, 9.81], [0, -9.81]):
            with self.subTest(gravity=gravity), self.assertRaises(ValueError):
                flight.geometry(self.scene, self.calibration, gravity=gravity)

    def test_tangent_origin_stays_on_vertical_drop_ray_and_includes_thickness(self):
        for tilt in (0., 15., 45., 65.):
            action = [.55, -.2, .30, tilt, .25]
            out = flight.nominal_origin(action, [.9, -.2], self.geom)
            self.assertEqual(out['origin_m'][:2], action[:2])
            self.assertAlmostEqual(out['origin_m'][2], .3+(.0375+.0025)/math.cos(math.radians(tilt)))
            normal = np.asarray(out['board_normal'])
            self.assertAlmostEqual(np.dot(np.asarray(out['origin_m'])-action[:3], normal), .04)
            self.assertAlmostEqual(np.dot(np.asarray(out['contact_point_m'])-action[:3], normal), .0025)
            self.assertAlmostEqual(out['board_local_contact_m'][2], .0025)
            self.assertEqual(out['provenance'], 'requested_geometry_nominal_sphere_plane_tangent_not_observed')
            self.assertNotIn('observed', out)

    def test_low_release_and_finite_board_edge_are_flagged_not_native_rejections(self):
        low = flight.nominal_origin([0., 0., .2, 45., .001], [.5, 0.], self.geom)
        self.assertFalse(low['release_above_tangent'])
        self.assertFalse(low['supported_planar_contact'])
        edge = flight.nominal_origin([0., 0., .2, 80., .5], [.5, 0.], self.geom)
        self.assertLess(edge['contact_edge_margin_m'], 0)
        self.assertFalse(edge['supported_planar_contact'])
        self.assertNotIn('native_rejected', edge)

    def test_zero_horizontal_velocity_center_passage_matches_polygon_apothem(self):
        out = flight.member_flight([0., 0., .35], [0., 0., .6], [0., 0.], self.geom)
        expected = .0885*math.cos(math.pi/64)-.0375
        self.assertTrue(out['full_downward_rim_band_available'])
        self.assertTrue(out['nominal_clear_aperture'])
        self.assertAlmostEqual(out['conservative_rim_band_clearance_m'], expected)
        self.assertAlmostEqual(out['first_descending_center_rim']['clearance_m'], expected)
        self.assertLess(out['rim_band_start_s'], out['first_descending_center_rim']['time_s'])
        self.assertLess(out['first_descending_center_rim']['time_s'], out['rim_band_end_s'])
        self.assertFalse(out['capture_predicted'])

    def test_below_rim_launch_uses_later_downward_crossing_after_ascent(self):
        p = [0., 0., .1]; v = [0., 0., 2.]
        out = flight.member_flight(p, v, [0., 0.], self.geom)
        self.assertTrue(out['starts_below_rim'])
        self.assertTrue(out['ascending_before_descent'])
        self.assertTrue(out['full_downward_rim_band_available'])
        t = out['first_descending_center_rim']['time_s']
        self.assertGreater(t, out['apex_time_s'])
        self.assertAlmostEqual(p[2]+2*t-.5*9.81*t*t, self.rim)
        self.assertLess(v[2]-9.81*t, 0)
        self.assertFalse(out['contacts_modeled'])

    def test_insufficient_apex_has_unknown_crossing_not_a_fake_capture_or_clearance(self):
        out = flight.member_flight([0., 0., .10], [1., 0., .1], [.4, 0.], self.geom)
        self.assertIsNone(out['first_descending_center_rim'])
        self.assertIsNone(out['conservative_rim_band_clearance_m'])
        self.assertIsNone(out['rim_band_start_s'])
        self.assertIsNone(out['rim_band_end_s'])
        self.assertFalse(out['full_downward_rim_band_available'])
        self.assertFalse(out['capture_predicted'])
        self.assertNotIn('capture', out)
        json.dumps(out, allow_nan=False)

    def test_center_crossing_alone_is_not_complete_sphere_clearance(self):
        out = flight.member_flight([0., 0., .16], [0., 0., .8], [0., 0.], self.geom)
        self.assertIsNotNone(out['first_descending_center_rim'])
        self.assertLess(out['apex_height_m'], self.rim+self.radius)
        self.assertFalse(out['full_downward_rim_band_available'])
        self.assertIsNone(out['conservative_rim_band_clearance_m'])

    def test_band_margin_catches_fast_horizontal_pass_that_center_point_cannot(self):
        base = flight.member_flight([-.4, 0., .25], [0., 0., 0.], [0., 0.], self.geom)
        t = base['first_descending_center_rim']['time_s']
        out = flight.member_flight([-.4, 0., .25], [.4/t, 0., 0.], [0., 0.], self.geom)
        self.assertGreater(out['first_descending_center_rim']['clearance_m'], 0)
        self.assertLess(out['conservative_rim_band_clearance_m'], 0)
        self.assertFalse(out['nominal_clear_aperture'])

    def test_endpoint_margin_equals_dense_minimum_without_timestep_dependence(self):
        p = [-.2, .03, .3]; v = [1.2, -.1, .5]
        out = flight.member_flight(p, v, [.2, 0.], self.geom)
        times = np.linspace(out['rim_band_start_s'], out['rim_band_end_s'], 10001)
        xyz = np.asarray(flight.trajectory_for_member(p, v, times))
        planes = np.asarray(self.geom['inward_rim_halfplanes'])
        values = (xyz[:, :2]-[.2, 0.]) @ planes[:, :2].T+planes[:, 2]-self.radius
        self.assertAlmostEqual(out['conservative_rim_band_clearance_m'], values.min(), places=12)

    def test_forward_lateral_velocity_is_rotated_using_each_requested_bucket_heading(self):
        raw = prediction((1., .2, .5))
        x = flight.predict_candidate(self.scene, self.calibration, [.8, -.2], [.4, -.2, .3, 30., .2], raw)
        y = flight.predict_candidate(self.scene, self.calibration, [.4, .2], [.4, -.2, .3, 30., .2], raw)
        np.testing.assert_allclose(x['members'][0]['velocity_world_m_s'], [1., .2, .5])
        np.testing.assert_allclose(y['members'][0]['velocity_world_m_s'], [-.2, 1., .5])

    def test_predictions_cannot_leak_observed_origin_or_outcomes(self):
        raw = prediction(); action = [.5, -.2, .3, 30., .2]
        first = flight.predict_candidate(self.scene, self.calibration, [.9, -.2], action, raw)
        raw.update(observed_origin_m=[8., 8., 8.], whole_ball_entry=True, route_success=True,
            rebound_velocity_m_s=[999., 999., 999.], aperture_distance_m=0.)
        second = flight.predict_candidate(self.scene, self.calibration, [.9, -.2], action, raw)
        self.assertEqual(first, second)
        self.assertFalse(first['uses_observed_rebound'])

    def test_ranking_ignores_saturated_classifier_when_ballistic_margin_is_better(self):
        context = dict(bucket_xy_m=[.9, -.2])
        values = [.5, -.2, .3, 30., .2]
        launch = flight.nominal_origin(values, context['bucket_xy_m'], self.geom)
        preliminary = flight.member_flight(launch['origin_m'], [0., 0., .5], context['bucket_xy_m'], self.geom)
        good_speed = .4/preliminary['first_descending_center_rim']['time_s']
        items = [dict(id=f'candidate-{i:02d}', values=values.copy()) for i in range(16)]
        preds = [prediction((.1, 0., .5), .999) for _ in items]
        preds[7] = prediction((good_speed, 0., .5), .5)
        before = copy.deepcopy((items, preds, context))
        out = flight.rank(self.scene, self.calibration, context, items, preds)
        self.assertEqual(out['order'][0], 7)
        self.assertEqual(set(out['order']), set(range(16)))
        self.assertEqual((items, preds, context), before)
        self.assertFalse(out['ball_flight_search_at_inference'])
        self.assertFalse(out['capture_probability_calibrated'])
        self.assertEqual(out['candidate_count'], 16)
        json.dumps(out, allow_nan=False)

    def test_negative_margin_candidate_is_retained_for_possible_rim_assisted_capture(self):
        items = [dict(id=f'candidate-{i:02d}', values=[.5, -.2, .3, 30., .2]) for i in range(16)]
        out = flight.rank(self.scene, self.calibration, dict(bucket_xy_m=[1.2, -.2]), items,
            [prediction((.1, 0., .2)) for _ in items])
        self.assertTrue(all(d['worst_clearance_m'] < 0 for d in out['diagnostics']))
        self.assertEqual(out['order'], list(range(16)))
        self.assertEqual(len(out['diagnostics']), 16)
        self.assertIn('Negative margins do not exclude rim-assisted capture', out['scope'])

    def test_missing_member_bands_sort_without_nonfinite_json_or_fake_values(self):
        items = [dict(id=f'candidate-{i:02d}', values=[.5, -.2, .08, 0., .1]) for i in range(16)]
        out = flight.rank(self.scene, self.calibration, dict(bucket_xy_m=[.9, -.2]), items,
            [prediction((.1, 0., .05)) for _ in items])
        self.assertTrue(all(d['worst_clearance_m'] is None for d in out['diagnostics']))
        self.assertEqual(out['order'], list(range(16)))
        json.dumps(out, allow_nan=False)

    def test_corrupt_members_shapes_ids_and_nonfinite_actions_are_rejected(self):
        for bad in ([.5, -.2, .3, 90., .2], [.5, -.2, .3, -1., .2],
                    [.5, -.2, .3, 30., 0.], [.5, -.2, float('nan'), 30., .2]):
            with self.subTest(values=bad), self.assertRaises(ValueError):
                flight.nominal_origin(bad, [.9, -.2], self.geom)
        for raw in ([], [[0.]*9]*4, [[0.]*8]*5, [[float('inf')]*9]*5):
            with self.assertRaises(ValueError):
                flight.predict_candidate(self.scene, self.calibration, [.9, -.2], [.5, -.2, .3, 30., .2],
                    dict(member_raw_output=raw, ranking_score=.5))
        items = [dict(id='duplicate', values=[.5, -.2, .3, 30., .2]) for _ in range(16)]
        with self.assertRaises(ValueError):
            flight.rank(self.scene, self.calibration, dict(bucket_xy_m=[.9, -.2]), items, [prediction()]*16)

    def test_finite_unsupported_actions_remain_without_fabricated_flights(self):
        examples = [
            ([.5, -.2, .3, 30., -.01], [.9, -.2], 'nonpositive_drop_gap'),
            ([.5, -.2, .3, 30., 0.], [.9, -.2], 'nonpositive_drop_gap'),
            ([.5, -.2, .3, -1., .2], [.9, -.2], 'tilt_outside_nominal_0_to_90_degrees'),
            ([.5, -.2, .3, 90., .2], [.9, -.2], 'tilt_outside_nominal_0_to_90_degrees'),
            ([.5, -.2, .3, 30., .2], [.5, -.2], 'board_bucket_heading_undefined')]
        for values, bucket, reason in examples:
            before = copy.deepcopy((values, bucket))
            with self.subTest(reason=reason, values=values):
                out = flight.predict_candidate(self.scene, self.calibration, bucket, values, prediction())
                self.assertEqual(out['members'], [])
                self.assertEqual(out['complete_rim_band_members'], 0)
                self.assertEqual(out['nominal_clear_aperture_members'], 0)
                self.assertFalse(out['launch']['supported_planar_contact'])
                self.assertIsNone(out['launch']['origin_m'])
                self.assertEqual(out['missing_reason'], reason)
                self.assertEqual(out['launch']['missing_reason'], reason)
                self.assertTrue(all(out[k] is None for k in (
                    'worst_clearance_m', 'mean_clearance_m', 'minimum_apex_margin_m')))
                self.assertEqual((values, bucket), before)
                self.assertNotIn('native_rejected', out)
                json.dumps(out, allow_nan=False)

    def test_rank_keeps_unsupported_candidate_identity_at_bottom_without_clipping(self):
        items = [dict(id=f'candidate-{i:02d}', values=[.5, -.2, .3, 30., .2]) for i in range(16)]
        items[0]['values'][4] = -.01
        items[1]['values'][3] = 90.
        preds = [prediction() for _ in items]
        preds[0]['ranking_score'] = 2.
        preds[1]['ranking_score'] = 1.
        before = copy.deepcopy((items, preds))
        out = flight.rank(self.scene, self.calibration, dict(bucket_xy_m=[.9, -.2]), items, preds)
        self.assertEqual(out['order'][-2:], [0, 1])
        self.assertEqual(set(out['order']), set(range(16)))
        self.assertEqual(out['diagnostics'][0]['candidate_id'], 'candidate-00')
        self.assertEqual(out['diagnostics'][0]['missing_reason'], 'nonpositive_drop_gap')
        self.assertEqual(out['diagnostics'][1]['candidate_id'], 'candidate-01')
        self.assertEqual((items, preds), before)

    def test_original_literal_candidate_with_minus_point_984_mm_gap_is_preserved(self):
        # Frozen old proposal, not an observed flight and not a modified action.
        values = [0.6343106041379774, -0.041399632707767114, 0.25271312949501057,
                  34.56318103656823, -0.000983542560295661]
        before = values.copy()
        out = flight.predict_candidate(self.scene, self.calibration,
            [values[0]+.29519209695938314, values[1]], values, prediction())
        self.assertEqual(out['missing_reason'], 'nonpositive_drop_gap')
        self.assertEqual(out['members'], [])
        self.assertEqual(out['complete_rim_band_members'], 0)
        self.assertIsNone(out['worst_clearance_m'])
        self.assertEqual(values, before)

    def test_unsupported_geometry_does_not_hide_malformed_evidence(self):
        invalid = [.5, -.2, .3, 30., -.01]
        for raw in (dict(member_raw_output=[]), prediction(score=float('nan'))):
            with self.assertRaises(ValueError):
                flight.predict_candidate(self.scene, self.calibration, [.9, -.2], invalid, raw)
        for values in ([.5, -.2, .3, 30.], [.5, -.2, float('nan'), 30., -.01]):
            with self.assertRaises(ValueError):
                flight.predict_candidate(self.scene, self.calibration, [.9, -.2], values, prediction())
        with self.assertRaises(ValueError):
            flight.predict_candidate(self.scene, self.calibration, [float('nan'), -.2], invalid, prediction())

    def test_import_does_not_load_torch_mujoco_warp_or_initialize_cuda(self):
        code = ('import sys; from sircus.mjlab import consistent_flight_selector; '
                'assert not any(x in sys.modules for x in ("torch","warp","mujoco","mjlab"))')
        result = subprocess.run([sys.executable, '-c', code], cwd=ROOT, text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == '__main__':
    unittest.main()
