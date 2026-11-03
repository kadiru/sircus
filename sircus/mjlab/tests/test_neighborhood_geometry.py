"""Protocol tests only: fake endpoints provide no native dynamics evidence."""
import copy
import unittest

import numpy as np

from sircus.mjlab.learning_contract import digest
from sircus.mjlab.neighborhood_geometry import (NeighborhoodGeometry, VERSION,
    audit_local_variations, nearest_anchor_id)


class FakeGeometry:
    instances = []
    substitute_board = False
    substitute_dropper = False
    jump10 = 0.
    jump14 = 0.

    def __init__(self, scene, assets, anchor_hint, bounds=None):
        self.anchor = copy.deepcopy(anchor_hint)
        self.bounds = copy.deepcopy(bounds)
        self.prepared, self.calls = {}, []
        type(self).instances.append(self)

    def prepare_contexts(self, contexts):
        result = []
        for c in contexts:
            row = copy.deepcopy(c)
            if c['id'] not in self.prepared:
                delta = ((c['board_height_m'] - self.anchor['values'][2]) * .2
                         + (c['tilt_deg'] - self.anchor['values'][3]) * .001)
                q14 = (np.asarray(self.anchor['q14']) + delta + self.jump14).tolist()
                if self.substitute_board:
                    q14[0] += .01
                row.update(prepared=True, board_admitted=True, q14=q14,
                    board_branch_sha256=digest(q14), reason=None)
                self.prepared[c['id']] = row
            result.append(copy.deepcopy(self.prepared[c['id']]))
        return result

    def audit(self, names, gaps):
        self.calls.append((list(names), list(gaps)))
        result = []
        for name, gap in zip(names, gaps):
            c = self.prepared[name]
            values = self.anchor['values'][:2] + [c['board_height_m'], c['tilt_deg'], gap]
            q10 = (np.asarray(self.anchor['q10']) + (gap - self.anchor['values'][4]) * .2 + self.jump10).tolist()
            if self.substitute_dropper:
                q10[0] += .01
            valid = self.bounds['low'][4] <= gap <= self.bounds['high'][4]
            result.append(dict(context_id=name, values=values, bucket_xy_m=self.anchor['bucket_xy_m'],
                q10=q10 if valid else [0.] * 6, q14=c['q14'], eligible=valid,
                reason=None if valid else 'action_bounds_or_invalid', native_release=False,
                board_branch_sha256=c['board_branch_sha256'], minimum_clearance_lower_bound_m=.006 if valid else None))
        return result


class NeighborhoodGeometryTests(unittest.TestCase):
    def setUp(self):
        FakeGeometry.instances = []
        FakeGeometry.substitute_board = FakeGeometry.substitute_dropper = False
        FakeGeometry.jump10 = FakeGeometry.jump14 = 0.
        self.anchors = [self.anchor('a', .4, 36., .30), self.anchor('b', .5, 36., .24),
                        self.anchor('c', .44, 32., .285)]

    @staticmethod
    def anchor(name, h, tilt, gap):
        index = {'a': 1, 'b': 2, 'c': 3}[name]
        return dict(id=name, values=[.47, -.28, h, tilt, gap], bucket_xy_m=[.16, -.63],
                    q10=[.1 * index] * 6, q14=[.2 * index] * 6)

    @staticmethod
    def context(name, h=.4, tilt=36.):
        return dict(id=name, board_height_m=h, tilt_deg=tilt)

    def geometry(self, anchors=None):
        return NeighborhoodGeometry({}, '/not-native-assets', anchors or self.anchors,
                                    geometry_factory=FakeGeometry)

    def test_nearest_assignment_uses_height_tilt_only_and_id_ties(self):
        self.assertEqual(nearest_anchor_id(.44, 32., self.anchors), 'c')
        self.assertEqual(nearest_anchor_id(.49, 36., self.anchors), 'b')
        self.assertEqual(nearest_anchor_id(.45, 36., self.anchors[:2][::-1]), 'a')
        geom = self.geometry(); out = geom.prepare_contexts([self.context('x', .49, 36.)])
        self.assertEqual(out[0]['source_anchor_id'], 'b')
        self.assertEqual(out[0]['wrapper_version'], VERSION)
        self.assertEqual(out[0]['source_anchor_geometry_sha256'], digest(self.anchors[1]))

    def test_context_and_query_order_do_not_change_branches(self):
        contexts = [self.context('x', .49), self.context('y', .405), self.context('z', .44, 32.)]
        a, b = self.geometry(), self.geometry(self.anchors[::-1])
        self.assertEqual(a.prepare_contexts(contexts), b.prepare_contexts(contexts[::-1])[::-1])
        names, gaps = ['x', 'y', 'z', 'x'], [.19, .30, .32, .25]
        first = a.audit(names, gaps); second = b.audit(names[::-1], gaps[::-1])
        self.assertEqual(first, second[::-1])
        by_id = {r['id']:r for r in a.contexts}
        for row in first:
            self.assertEqual(row['q14'], by_id[row['context_id']]['q14'])

    def test_exact_anchor_preserves_both_arrays_and_duplicate_query_resolves_once(self):
        geom = self.geometry(); contexts = [self.context(a['id'], a['values'][2], a['values'][3]) for a in self.anchors]
        geom.prepare_contexts(contexts)
        out = geom.audit(['a', 'b', 'c', 'a'], [.30, .24, .285, .30])
        self.assertEqual(out[0], out[-1])
        for row, anchor in zip(out, self.anchors):
            self.assertEqual(row['q10'], anchor['q10']); self.assertEqual(row['q14'], anchor['q14'])
        self.assertEqual(sum(len(c[0]) for p in FakeGeometry.instances for c in p.calls), 3)
        first_pool_ids = list(map(id, FakeGeometry.instances))
        out[0]['q14'][0] = 99.; geom.prepare_contexts(contexts)
        again = geom.audit(['a'], [.30])[0]
        self.assertEqual(again['q14'], self.anchors[0]['q14'])
        self.assertEqual(first_pool_ids, list(map(id, FakeGeometry.instances)))

    def test_anchors_reject_outcomes_and_mixed_xy_or_duplicate_identity(self):
        for mutate in (
                lambda a: a[0].update(route_success=True),
                lambda a: a[0].update(q10=[float('nan')] * 6),
                lambda a: a[1]['values'].__setitem__(0, .471),
                lambda a: a[1]['bucket_xy_m'].__setitem__(0, .161),
                lambda a: a[1].update(id='a'),
                lambda a: a[1]['values'].__setitem__(4, 2.)):
            anchors = copy.deepcopy(self.anchors); mutate(anchors)
            with self.assertRaises(ValueError): self.geometry(anchors)

    def test_context_validation_fails_before_mutating_children(self):
        geom = self.geometry(); geom.prepare_contexts([self.context('a')])
        before = copy.deepcopy(geom.contexts)
        with self.assertRaises(ValueError): geom.prepare_contexts([self.context('new'), self.context('a', .41)])
        with self.assertRaises(ValueError): geom.prepare_contexts([dict(self.context('other'), reward=1.)])
        with self.assertRaises(ValueError): geom.prepare_contexts([self.context('dup'), self.context('dup')])
        self.assertEqual(geom.contexts, before)
        with self.assertRaises(ValueError): geom.audit(['unknown'], [.3])
        with self.assertRaises(ValueError): geom.audit(['a'], [float('nan')])

    def test_exact_anchor_substitution_fails_closed(self):
        FakeGeometry.substitute_board = True
        with self.assertRaises(RuntimeError): self.geometry().prepare_contexts([self.context('anchor')])
        FakeGeometry.substitute_board = False; geom = self.geometry(); geom.prepare_contexts([self.context('anchor')])
        FakeGeometry.substitute_dropper = True
        with self.assertRaises(RuntimeError): geom.audit(['anchor'], [.30])

    def nominal(self):
        return dict(copy.deepcopy(self.anchors[0]), eligible=True, route_success=True, learning_reward=4.)

    def variations(self):
        return [dict(context_id='n', values=[.47, -.28, .4, 36., .3]),
                dict(context_id='up', board_height_m=.401, tilt_deg=36.1, gap_m=.302),
                dict(context_id='n', values=[.47, -.28, .4, 36., .3])]

    def local(self, rows=None):
        return audit_local_variations({}, '/not-native-assets', self.nominal(),
            self.variations() if rows is None else rows, geometry_factory=FakeGeometry)

    def test_local_variations_preserve_nominal_branch_and_request_order(self):
        out = self.local()
        self.assertEqual(out[0], out[2]); self.assertEqual([r['context_id'] for r in out], ['n', 'up', 'n'])
        self.assertEqual(out[0]['q10'], self.anchors[0]['q10'])
        self.assertEqual(out[0]['q14'], self.anchors[0]['q14'])
        self.assertTrue(all(r['eligible'] and r['local_variation_gate']['passed'] for r in out))
        self.assertNotIn('route_success', FakeGeometry.instances[-1].anchor)
        self.assertEqual(len(FakeGeometry.instances[-1].calls[0][0]), 2)

    def test_local_joint_jumps_are_rejected_on_either_arm_without_wrapping(self):
        for attribute in ('jump10', 'jump14'):
            setattr(FakeGeometry, attribute, 2 * np.pi)
            out = self.local([self.variations()[1]])[0]
            self.assertFalse(out['eligible']); self.assertEqual(out['reason'], 'branch_discontinuity')
            self.assertFalse(out['native_release']); self.assertTrue(out['endpoint_eligible_before_continuity_gate'])
            setattr(FakeGeometry, attribute, 0.)

    def test_local_nominal_substitution_and_bounds_rejection_remain_visible(self):
        FakeGeometry.substitute_dropper = True
        out = self.local([self.variations()[0]])[0]
        self.assertFalse(out['eligible']); self.assertEqual(out['reason'], 'nominal_branch_substituted')
        FakeGeometry.substitute_dropper = False
        out = self.local([dict(context_id='low', board_height_m=.4, tilt_deg=36., gap_m=.001)])[0]
        self.assertFalse(out['eligible']); self.assertEqual(out['reason'], 'action_bounds_or_invalid')
        self.assertIsNone(out['local_variation_gate']['arm10_max_abs_joint_delta_rad'])

    def test_local_request_identity_and_conflicting_contexts_are_rejected(self):
        for request in (dict(context_id='x', values=[.5, -.28, .4, 36., .3]),
                        dict(context_id='x', values=[.47, -.28, .4, 36., .3], gap_m=.31),
                        dict(context_id='x', board_height_m=.4, tilt_deg=36., gap_m=.3, route_success=True),
                        dict(context_id='', board_height_m=.4, tilt_deg=36., gap_m=.3)):
            with self.assertRaises(ValueError): self.local([request])
        with self.assertRaises(ValueError): self.local([self.variations()[0], dict(context_id='n', board_height_m=.41, tilt_deg=36., gap_m=.3)])
        self.assertEqual(self.local([]), [])


if __name__ == '__main__': unittest.main()
