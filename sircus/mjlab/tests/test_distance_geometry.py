"""CPU/mock protocol tests; these establish no native flight or capture rate."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np

from sircus.mjlab.distance_geometry import (DistanceGeometry, EDGE_MARGIN_M,
    LOCAL_VERSION, VERSION, _derive_rotated_anchor, audit_distance_variations,
    ray_limits, same_bearing, table_footprint)
from sircus.mjlab.learning_contract import digest
from sircus.mjlab.learning_ik import shot_targets
from sircus.mjlab.tests.test_neighborhood_geometry import FakeGeometry


ROOT = Path(__file__).resolve().parents[3]
SCENE = ROOT / 'baselines/mjlab-height-tilt-20261102/completed/scene.json'
BOARD_XY = [.4708757793012148, -.2778934014509746]
BUCKET_XY = [.16275255472120964, -.6365356466995716]


class DistanceFakeGeometry(FakeGeometry):
    substitute = False

    def prepare_contexts(self, contexts):
        rows = super().prepare_contexts(contexts)
        for row in rows:
            row['bucket_xy_m'] = copy.deepcopy(self.anchor['bucket_xy_m'])
        return rows

    def audit(self, ids, gaps):
        rows = super().audit(ids, gaps)
        for row in rows:
            if self.substitute:
                row['q14'][0] += .01
        return rows


class DistanceGeometryTests(unittest.TestCase):
    def setUp(self):
        self.scene = json.loads(SCENE.read_text())
        self.anchors = [dict(id='recipe-00', values=BOARD_XY + [.3973878458850598, 36.154791124080255, .32121875],
            bucket_xy_m=BUCKET_XY, q10=[.1] * 6, q14=[.2] * 6)]
        DistanceFakeGeometry.instances = []
        DistanceFakeGeometry.substitute = False
        DistanceFakeGeometry.substitute_board = DistanceFakeGeometry.substitute_dropper = False
        DistanceFakeGeometry.jump10 = DistanceFakeGeometry.jump14 = 0.
        self.derived = []

    def context(self, name='one', distance=.55, ray=None):
        if ray is None:
            ray = np.asarray(BUCKET_XY) - BOARD_XY
            ray /= np.linalg.norm(ray)
        return dict(id=name, source_anchor_id='recipe-00', distance_id='d-' + name,
            ray_id='original' if same_bearing(np.asarray(BOARD_XY), BUCKET_XY,
                np.asarray(BOARD_XY) + ray) else 'corner', distance_m=distance,
            bucket_xy_m=(np.asarray(BOARD_XY) + distance * ray).tolist(),
            board_height_m=self.anchors[0]['values'][2], tilt_deg=self.anchors[0]['values'][3])

    def derived_builder(self, pool, anchor, context):
        self.derived.append(copy.deepcopy(context))
        derived = copy.deepcopy(anchor)
        derived['bucket_xy_m'] = copy.deepcopy(context['bucket_xy_m'])
        derived['q14'] = (np.asarray(anchor['q14']) + .03).tolist()
        return derived, dict(admitted=True, preserve_known=False,
            old_joints_role='numerical_continuation_only', q14=derived['q14'])

    def geometry(self):
        return DistanceGeometry(self.scene, '/no-native-assets', self.anchors,
            geometry_factory=DistanceFakeGeometry, rotated_anchor_builder=self.derived_builder)

    def corner_ray(self):
        ray = np.asarray([-.125, -.77]) - BOARD_XY
        return ray / np.linalg.norm(ray)

    def test_measured_outer_footprint_and_original_exact_table_limit(self):
        point = table_footprint(self.scene, BUCKET_XY)
        self.assertTrue(point['passed']); self.assertEqual(point['outer_radius_m'], .1)
        self.assertEqual(point['required_edge_margin_m'], EDGE_MARGIN_M)
        self.assertAlmostEqual(point['minimum_edge_clearance_m'], .1384643533004284)
        ray = (np.asarray(BUCKET_XY) - BOARD_XY)
        ray /= np.linalg.norm(ray)
        limit = ray_limits(self.scene, BOARD_XY, ray)
        self.assertAlmostEqual(limit['maximum_distance_m'], .6487820950343407)
        self.assertTrue(table_footprint(self.scene, np.asarray(BOARD_XY) + limit['maximum_distance_m'] * ray)['passed'])
        self.assertTrue(table_footprint(self.scene, self.context(distance=.64)['bucket_xy_m'])['passed'])
        self.assertFalse(table_footprint(self.scene, self.context(distance=.65)['bucket_xy_m'])['passed'])

    def test_corner_distances_retain_off_table_requests_without_ik(self):
        geom = self.geometry()
        contexts = [self.context('ok', .77, self.corner_ray()), self.context('off', .8, self.corner_ray()),
            self.context('far', .95, self.corner_ray())]
        rows = geom.prepare_contexts(contexts)
        self.assertTrue(rows[0]['bucket_footprint']['passed'])
        for row in rows[1:]:
            self.assertFalse(row['board_admitted']); self.assertFalse(row['bucket_footprint']['passed'])
            self.assertEqual(row['reason'], 'bucket_outer_footprint_off_table')
        self.assertEqual(len(self.derived), 1)
        rejected = geom.audit(['far', 'off'], [.2, .3])
        self.assertEqual([r['bucket_xy_m'] for r in rejected], [contexts[2]['bucket_xy_m'], contexts[1]['bucket_xy_m']])
        self.assertTrue(all(not r['eligible'] and not r['native_release'] for r in rejected))
        self.assertEqual(sum(len(p.calls) for p in DistanceFakeGeometry.instances), 0)

    def test_original_bearing_reuses_original_exact_board_and_dropper(self):
        geom = self.geometry(); context = self.context()
        prepared = geom.prepare_contexts([context])[0]
        self.assertTrue(prepared['same_source_bearing']); self.assertEqual(self.derived, [])
        self.assertEqual(prepared['q14'], self.anchors[0]['q14'])
        gaps = [.28, self.anchors[0]['values'][4], .38]
        rows = geom.audit(['one'] * 3, gaps)
        self.assertTrue(all(r['q14'] == self.anchors[0]['q14'] for r in rows))
        self.assertEqual(rows[1]['q10'], self.anchors[0]['q10'])
        self.assertTrue(all(r['source_anchor_geometry_sha256'] == digest(self.anchors[0])
            and r['wrapper_version'] == VERSION for r in rows))
        self.assertEqual(rows[1]['bucket_xy_m'], context['bucket_xy_m'])

    def test_changed_heading_uses_only_derived_board_as_anchor(self):
        geom = self.geometry(); context = self.context('rotated', .7, self.corner_ray())
        prepared = geom.prepare_contexts([context])[0]
        self.assertFalse(prepared['same_source_bearing']); self.assertEqual(len(self.derived), 1)
        self.assertNotEqual(prepared['q14'], self.anchors[0]['q14'])
        self.assertFalse(prepared['rotated_board_derivation']['preserve_known'])
        row = geom.audit(['rotated'], [self.anchors[0]['values'][4]])[0]
        self.assertEqual(row['q14'], prepared['q14']); self.assertEqual(row['q10'], self.anchors[0]['q10'])
        self.assertEqual(self.anchors[0]['bucket_xy_m'], BUCKET_XY)
        self.assertEqual(self.anchors[0]['q14'], [.2] * 6)

    def test_failed_rotated_derivation_is_retained_without_gap_ik(self):
        geometry = DistanceGeometry(self.scene, '/none', self.anchors,
            geometry_factory=DistanceFakeGeometry, rotated_anchor_builder=lambda *args: (None, dict(admitted=False)))
        c = self.context('bad', .7, self.corner_ray())
        row = geometry.prepare_contexts([c])[0]
        self.assertEqual(row['reason'], 'rotated_board_ik_unresolved')
        self.assertEqual(geometry.audit(['bad'], [.3])[0]['reason'], row['reason'])
        self.assertFalse(any(p.calls for p in DistanceFakeGeometry.instances))

    def test_workspace_unreachable_ray_does_not_allocate_ik(self):
        ray = np.asarray([np.cos(np.deg2rad(-15)), np.sin(np.deg2rad(-15))])
        context = self.context('inward', .95, ray)
        geom = self.geometry(); row = geom.prepare_contexts([context])[0]
        self.assertTrue(row['bucket_footprint']['passed'])
        self.assertEqual(row['reason'], 'workspace_prior_reject')
        self.assertEqual(DistanceFakeGeometry.instances, []); self.assertEqual(self.derived, [])

    def test_contexts_and_exact_queries_are_order_independent_and_cached(self):
        contexts = [self.context('a', .45), self.context('b', .6), self.context('c', .7, self.corner_ray())]
        first, second = self.geometry(), self.geometry()
        a, b = first.prepare_contexts(contexts), second.prepare_contexts(contexts[::-1])
        self.assertEqual(a, b[::-1])
        names, gaps = ['b', 'a', 'c', 'b'], [.3, .28, .31, .3]
        x, y = first.audit(names, gaps), second.audit(names[::-1], gaps[::-1])
        self.assertEqual(x, y[::-1]); self.assertEqual(x[0], x[-1])
        before = sum(len(p.calls) for p in DistanceFakeGeometry.instances)
        x[0]['q14'][0] = 123.
        self.assertNotEqual(first.audit(['b'], [.3])[0]['q14'][0], 123.)
        self.assertEqual(before, sum(len(p.calls) for p in DistanceFakeGeometry.instances))

    def test_conflicts_and_outcomes_fail_before_any_child_mutation(self):
        geom = self.geometry(); geom.prepare_contexts([self.context('a')]); before = geom.contexts
        for changed in (dict(self.context('a'), tilt_deg=37.), dict(self.context('b'), route_success=True),
                dict(self.context('b'), source_anchor_id='unknown'), dict(self.context('b'), distance_m=.1),
                dict(self.context('b'), board_xy_m=[.471, -.278])):
            with self.assertRaises(ValueError): geom.prepare_contexts([self.context('new'), changed])
        self.assertEqual(geom.contexts, before)
        with self.assertRaises(ValueError): geom.prepare_contexts([self.context('dup'), self.context('dup')])
        with self.assertRaises(ValueError): geom.audit(['missing'], [.3])
        with self.assertRaises(ValueError): geom.audit(['a'], [float('nan')])
        for changed in (dict(self.anchors[0], reward=3), dict(self.anchors[0], q14=[float('nan')] * 6)):
            with self.assertRaises(ValueError): DistanceGeometry(self.scene, '/none', [changed])

    def test_source_or_frozen_branch_substitution_fails_closed(self):
        DistanceFakeGeometry.substitute_board = True
        with self.assertRaises(RuntimeError): self.geometry().prepare_contexts([self.context()])
        DistanceFakeGeometry.substitute_board = False
        geom = self.geometry(); geom.prepare_contexts([self.context()])
        DistanceFakeGeometry.substitute = True
        with self.assertRaises(RuntimeError): geom.audit(['one'], [.3])
        DistanceFakeGeometry.substitute = False; DistanceFakeGeometry.substitute_dropper = True
        with self.assertRaises(RuntimeError): geom.audit(['one'], [self.anchors[0]['values'][4]])

    def test_repeat_helper_uses_actual_bucket_and_preserves_existing_joint_delta_gate(self):
        context = self.context('nominal', .6)
        geom = self.geometry(); geom.prepare_contexts([context])
        nominal = geom.audit(['nominal'], [self.anchors[0]['values'][4]])[0]
        requests = [dict(context_id='noise', board_height_m=context['board_height_m'] + .001,
            tilt_deg=context['tilt_deg'] + .25, gap_m=self.anchors[0]['values'][4] + .002)]
        output = audit_distance_variations(self.scene, '/none', nominal, requests,
            geometry_factory=DistanceFakeGeometry)
        self.assertTrue(output[0]['eligible'])
        self.assertEqual(output[0]['bucket_xy_m'], context['bucket_xy_m'])
        self.assertEqual(output[0]['distance_wrapper_version'], LOCAL_VERSION)
        self.assertEqual(output[0]['local_variation_gate']['max_joint_delta_rad'], .1)
        DistanceFakeGeometry.jump10 = .11
        rejected = audit_distance_variations(self.scene, '/none', nominal, requests,
            geometry_factory=DistanceFakeGeometry)[0]
        self.assertFalse(rejected['eligible']); self.assertEqual(rejected['reason'], 'branch_discontinuity')

    def test_ray_and_footprint_inputs_are_not_silently_projected(self):
        for ray in ([1., 1.], [0., 0.], [float('nan'), 0.]):
            with self.assertRaises(ValueError): ray_limits(self.scene, BOARD_XY, ray)
        with self.assertRaises(ValueError): table_footprint(self.scene, [float('nan'), 0.])
        with self.assertRaises(ValueError): table_footprint(self.scene, BUCKET_XY, -.01)
        self.assertFalse(same_bearing(np.asarray(BOARD_XY), BUCKET_XY,
            2 * np.asarray(BOARD_XY) - BUCKET_XY))


class RotatedDerivationTests(unittest.TestCase):
    def test_real_builder_disables_known_preservation_and_selects_strict_full_pose_branch(self):
        scene = json.loads(SCENE.read_text())
        anchor = dict(id='source', values=BOARD_XY + [.3973878458850598, 36.154791124080255, .32121875],
            bucket_xy_m=BUCKET_XY, q10=[.1] * 6, q14=[.2] * 6)
        context = dict(board_height_m=anchor['values'][2], tilt_deg=anchor['values'][3], bucket_xy_m=[-.05, -.70])
        seen = {}
        def solve(target, initial, **kwargs):
            seen.update(kwargs); seen['target'] = target
            return dict(q=np.asarray([[[.2] * 6, [.21] * 6, [.22] * 6]]),
                valid=np.asarray([[True, True, True]]), residual=np.asarray([[0., 1e-6, 2e-6]]), query_sha256=['query'])
        def check(q14, board_world):
            seen.setdefault('checked', []).append(q14.tolist())
            # The old branch can have a small IK residual while failing the
            # independent native full pose. It must never become the new control.
            return dict(passed=q14[0] > .205, reason='native_board_full_pose_mismatch' if q14[0] <= .205 else None)
        geometry = SimpleNamespace(scene=scene, ik=SimpleNamespace(board=SimpleNamespace(solve_candidates=solve)),
            initials=dict(board_joint_rad=[0.] * 6), _check_board=check)
        derived, proof = _derive_rotated_anchor(geometry, anchor, context)
        self.assertEqual(seen['preserve_known'].tolist(), [False])
        self.assertEqual(derived['q14'], [.21] * 6)
        self.assertEqual(derived['q10'], anchor['q10'])
        self.assertEqual(derived['bucket_xy_m'], context['bucket_xy_m'])
        self.assertEqual(proof['derived_anchor_geometry_sha256'], digest(derived))
        self.assertTrue(proof['admitted']); self.assertEqual(len(proof['branch_checks']), 2)
        expected = shot_targets(scene, np.asarray([context['bucket_xy_m']]), np.asarray([anchor['values'][:3]]),
            [anchor['values'][3]], [anchor['values'][4]])
        np.testing.assert_array_equal(seen['target'], expected['board_flange_local'])


if __name__ == '__main__':
    unittest.main()
