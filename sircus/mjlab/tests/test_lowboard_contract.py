"""CPU protocol tests with synthetic outcomes, never native capture evidence."""
from collections import Counter
import copy
import math
import unittest

import numpy as np

from sircus.mjlab import distance_contract as parent
from sircus.mjlab import lowboard_contract as protocol
from sircus.mjlab.expand_geometry import maximum_gap, outer_geometry_mask
from sircus.mjlab.tests.test_distance_contract import outcome, paired_rows, parent_inputs


class LowBoardContractTests(unittest.TestCase):
    def setUp(self):
        self.scene, self.seeds = parent_inputs()
        self.rays, self.targets = protocol.targets(self.scene, self.seeds)
        self.contexts = protocol.contexts(self.seeds, self.targets)
        self.dev = [c for c in self.contexts if c['stage'] == 'lowboard']

    def test_analytic_ceiling_matches_existing_prior_and_is_not_native_admission(self):
        xy = self.seeds[0]['values'][:2]
        expected = .364 + math.sqrt(.842**2 - xy[0]**2 - xy[1]**2) - .095 - .09
        limit = protocol.reach_limit(self.scene, self.seeds)
        self.assertAlmostEqual(expected, .8193243380486807, places=15)
        self.assertAlmostEqual(limit['ball_center_height_m'], expected, places=15)
        self.assertAlmostEqual(limit['board_height_plus_gap_m'], expected-self.scene['ball']['radius_m'], places=15)
        self.assertIn('cheap', limit['scope']); self.assertIn('native IK', limit['scope'])
        for h in protocol.HEIGHTS:
            cap = maximum_gap(np.asarray([[*xy, h]]), self.scene)[0]
            self.assertAlmostEqual(h + self.scene['ball']['radius_m'] + cap, expected, places=15)
        # This board pose is inside the unchanged board workspace prior. Only
        # the dropper's analytic reach changes across the derived ceiling.
        target = np.asarray([self.targets[0]['bucket_xy_m']])
        values = np.asarray([[[*xy, .36, 40., expected-.36-self.scene['ball']['radius_m']+delta]
            for delta in (-1e-6, 1e-6)]])
        within, reach = outer_geometry_mask(self.scene, target, values, bounds=protocol.BOUNDS)
        self.assertTrue(within.all()); self.assertEqual(reach.tolist(), [[True, False]])
        changed = copy.deepcopy(self.seeds); changed[1]['values'][0] += .001
        with self.assertRaises(ValueError): protocol.reach_limit(self.scene, changed)
        changed = copy.deepcopy(self.seeds)
        for seed in changed: seed['values'][:2] = [1., 1.]
        with self.assertRaises(ValueError): protocol.reach_limit(self.scene, changed)

    def test_four_exact_parent_targets_and_control_target_are_kept_separate(self):
        _, original = parent.targets(self.scene, self.seeds)
        expected = [t for t in original if t['id'] in ('corner-d01', 'corner-d02', 'corner-d03', 'corner-d04')]
        self.assertEqual(self.targets, expected)
        self.assertEqual([t['distance_m'] for t in self.targets], [.65, .70, .75, .77])
        self.assertEqual(len(self.rays), 1); self.assertEqual(self.rays[0]['id'], 'corner')
        self.assertTrue(all(t['bucket_on_table'] for t in self.targets))
        controls = protocol.control_target(self.seeds)
        self.assertNotIn(controls['id'], {t['id'] for t in self.targets})
        self.assertEqual(controls['bucket_xy_m'], self.seeds[0]['bucket_xy_m'])
        self.assertEqual(controls['ray_id'], 'original_control')

    def test_source_assignment_is_geometry_only_with_stable_ties_and_no_label_leak(self):
        original = copy.deepcopy(self.seeds)
        annotated = copy.deepcopy(self.seeds)
        for i, seed in enumerate(annotated):
            seed.update(reward=1e9*i, route_success=bool(i % 2), captures=999-i)
        for h in protocol.HEIGHTS:
            for tilt in protocol.TILTS:
                expected = min(original, key=lambda s: (
                    ((h-s['values'][2])/.040)**2 + ((tilt-s['values'][3])/4.)**2, s['id']))['id']
                self.assertEqual(protocol.source_anchor(annotated, h, tilt)['id'], expected)
                self.assertEqual(protocol.source_anchor(annotated[::-1], h, tilt)['id'], expected)
        self.assertEqual(protocol.contexts(annotated, self.targets), self.contexts)
        tie = [dict(id='z', values=[0., 0., .30, 36., .2]),
               dict(id='a', values=[0., 0., .30, 36., .2])]
        self.assertEqual(protocol.source_anchor(tie, .30, 36.)['id'], 'a')
        self.assertEqual(self.seeds, original)

    def test_complete_grid_and_exact_controls_have_136_contexts_and_664_initial_requests(self):
        self.assertEqual(protocol.HEIGHTS, (.22, .24, .26, .28, .30, .32, .34, .36))
        self.assertEqual(protocol.TILTS, (28., 32., 36., 40.))
        self.assertEqual(protocol.HEADROOM, (.005, .015, .030, .050, .080))
        self.assertEqual(len(self.contexts), len({c['id'] for c in self.contexts}))
        self.assertEqual(len(self.contexts), 136); self.assertEqual(len(self.dev), 128)
        self.assertEqual(Counter(c['distance_id'] for c in self.dev), {t['id']:32 for t in self.targets})
        expected = {(h, t) for h in protocol.HEIGHTS for t in protocol.TILTS}
        for target in self.targets:
            self.assertEqual({(c['board_height_m'],c['tilt_deg']) for c in self.dev if c['distance_id']==target['id']}, expected)
        controls = protocol.control_plan(self.seeds, self.targets)
        self.assertEqual(len(controls), 24)
        indexed = {c['id']:c for c in self.contexts}
        for seed in self.seeds:
            group = [r for r in controls if r['source_anchor_id']==seed['id']]
            self.assertEqual([r['repeat_index'] for r in group], [0,1,2])
            for request in group:
                c = indexed[request['context_id']]
                self.assertEqual(c['stage'], 'control')
                self.assertEqual(c['bucket_xy_m'], seed['bucket_xy_m'])
                self.assertEqual([c['board_height_m'],c['tilt_deg'],request['gap_m']], seed['values'][2:])
                self.assertEqual(request['role'], 'control')
        plan = protocol.coarse_plan(self.contexts, self.scene, self.seeds)
        self.assertEqual(len(plan),640); self.assertEqual(len(plan)+len(controls),664)
        self.assertEqual(set(Counter(r['context_id'] for r in plan).values()),{5})
        self.assertEqual({r['role'] for r in plan},{'development'})
        self.assertEqual({r['phase'] for r in plan},{'coarse'})

    def test_gaps_derive_from_absolute_release_levels_without_clipping_or_mutation(self):
        before = copy.deepcopy(self.contexts)
        plan = protocol.coarse_plan(self.contexts, self.scene, self.seeds)
        indexed = {c['id']:c for c in self.contexts}
        expected_cap = .364 + math.sqrt(.842**2-sum(v*v for v in self.seeds[0]['values'][:2]))-.095-.09
        for request in plan:
            c = indexed[request['context_id']]
            expected = expected_cap-c['board_height_m']-self.scene['ball']['radius_m']-request['release_headroom_m']
            self.assertAlmostEqual(request['gap_m'], expected, places=15)
            self.assertAlmostEqual(c['board_height_m']+self.scene['ball']['radius_m']+request['gap_m'], request['requested_ball_center_height_m'],places=15)
            self.assertEqual(c['board_xy_m'], self.seeds[0]['values'][:2])
        self.assertAlmostEqual(min(r['gap_m'] for r in plan), .3418243380486807, places=15)
        self.assertAlmostEqual(max(r['gap_m'] for r in plan), .5568243380486807, places=15)
        self.assertEqual(self.contexts,before)
        with self.assertRaises(ValueError): protocol.coarse_plan(self.contexts)
        outside = copy.deepcopy(self.dev[0]);outside['board_height_m']=1.
        rejected = protocol.coarse_plan([outside], self.scene, self.seeds)
        self.assertTrue(all(r['gap_m']<0 for r in rejected))

    def synthetic_search(self, hit=True):
        return [outcome(c, request['gap_m'], hit, suffix=str(i))
            for c in self.dev for i,request in enumerate(protocol.coarse_plan([c],self.scene,self.seeds))]

    def test_adaptive_budget_minmax_and_actual_sample_recipe_selection(self):
        missed = self.synthetic_search(False)
        self.assertEqual(protocol.refinement_plan(self.contexts,missed),[])
        self.assertEqual(protocol.select_recipes(missed,self.targets),[])
        self.assertEqual(protocol.repeat_plan([]),[])
        rows = self.synthetic_search(); before=copy.deepcopy(rows)
        refinement = protocol.refinement_plan(self.contexts,rows)
        self.assertEqual(len(refinement),128*4)
        self.assertEqual(set(Counter(r['context_id'] for r in refinement).values()),{4})
        recipes = protocol.select_recipes(rows,self.targets)
        self.assertEqual(len(recipes),8)
        self.assertEqual(Counter(r['distance_id'] for r in recipes),{t['id']:2 for t in self.targets})
        self.assertEqual(len({r['source']['context_id'] for r in recipes}),8)
        originals={r['id']:r for r in rows}
        for recipe in recipes:
            self.assertEqual(recipe['source'],originals[recipe['source']['id']])
            self.assertTrue(recipe['source']['route_success'])
            self.assertEqual(recipe['gap_m'],recipe['source']['gap_m'])
        self.assertEqual(rows,before)
        self.assertEqual(len(protocol.repeat_plan(recipes))*3,792)
        settings=protocol.settings(self.scene,self.seeds)
        self.assertEqual(protocol.MAX_REQUESTS,24+640+512+792)
        self.assertEqual(sum(settings['budgets'][k] for k in ('controls','coarse','refinement_max','repeats_max')),1968)
        self.assertEqual(settings['budgets']['maximum_requests'],protocol.MAX_REQUESTS)
        for role in ('test','validation','control'):
            with self.assertRaises(ValueError): protocol.select_recipes([dict(rows[0],role=role)],self.targets)
            with self.assertRaises(ValueError): protocol.refinement_plan(self.contexts,[dict(rows[0],role=role)])

    def test_repeat_noise_is_joint_seeded_unclipped_and_three_nominals_are_duplicates(self):
        source=outcome(self.dev[0],protocol.BOUNDS['low'][4]);source['values'][2:4]=protocol.BOUNDS['low'][2:4]
        recipe=dict(id='recipe-00',source=source,**{k:self.dev[0][k] for k in ('distance_id','ray_id','distance_m','board_xy_m','bucket_xy_m')})
        before=copy.deepcopy(recipe);plan=protocol.repeat_plan([recipe])
        self.assertEqual(plan,protocol.repeat_plan([recipe]));self.assertEqual(recipe,before)
        self.assertEqual(Counter(r['phase'] for r in plan),{'nominal':3,'fresh':30})
        self.assertEqual(len({r['repeat_id'] for r in plan}),33)
        self.assertEqual(len({tuple(r['values']) for r in plan}),31)
        self.assertTrue(all(r['values']==source['values'] for r in plan[:3]))
        self.assertTrue(any(r['gap_m']<protocol.BOUNDS['low'][4] for r in plan[3:]))
        for request in plan:
            np.testing.assert_allclose(np.asarray(request['values'])[2:]-source['values'][2:],request['noise'],rtol=0,atol=1e-14)
            self.assertEqual(request['values'][:2],source['values'][:2]);self.assertEqual(request['bucket_xy_m'],recipe['bucket_xy_m'])
            self.assertNotIn('substeps',request)
        self.assertEqual(protocol.SUBSTEPS,(32,64,128))
        settings=protocol.settings(self.scene,self.seeds)['repeatability']
        self.assertEqual((settings['nominal'],settings['fresh']),(3,30))
        self.assertFalse(settings['clipping']);self.assertFalse(settings['retry']);self.assertTrue(settings['same_requests_and_joints'])

    def test_fresh_thresholds_and_all_resolution_gate_remain_distinct(self):
        rows,positions=paired_rows()
        self.assertTrue(protocol.sensitivity(rows,positions)['all_resolution_repeat_gate'])
        for r in rows:
            if r['substeps']==32 and r['repeat_index'] in (3,4,5):r['route_success']=False
        self.assertTrue(protocol.sensitivity(rows,positions)['all_resolution_repeat_gate'])
        next(r for r in rows if r['substeps']==32 and r['repeat_index']==6)['route_success']=False
        result=protocol.sensitivity(rows,positions)
        self.assertTrue(result['paired_engineering_gate']);self.assertFalse(result['all_resolution_repeat_gate'])
        rows,positions=paired_rows()
        for r in rows:
            if r['substeps']==64 and r['repeat_index'] in (3,4,5):r['route_success']=False
        result=protocol.sensitivity(rows,positions)
        self.assertEqual(result['capture_agreement_64_128'],30);self.assertTrue(result['paired_engineering_gate'])
        next(r for r in rows if r['substeps']==64 and r['repeat_index']==6)['route_success']=False
        self.assertFalse(protocol.sensitivity(rows,positions)['paired_engineering_gate'])

    def test_gate_requires_nominal_capture_pairing_agreement_and_5mm_early_difference(self):
        rows,positions=paired_rows();positions['0-128'][83,0]=.005
        self.assertTrue(protocol.sensitivity(rows,positions)['paired_engineering_gate'])
        positions['0-128'][83,0]+=.000001
        self.assertFalse(protocol.sensitivity(rows,positions)['paired_engineering_gate'])
        rows,positions=paired_rows()
        for r in rows:
            if (r['substeps'],r['repeat_index']) in ((64,3),(64,4),(128,5),(128,6)):r['route_success']=False
        result=protocol.sensitivity(rows,positions)
        self.assertEqual(result['capture_agreement_64_128'],29);self.assertFalse(result['paired_engineering_gate'])
        for step in (32,64,128):
            rows,positions=paired_rows()
            next(r for r in rows if r['substeps']==step and r['repeat_index']==0)['route_success']=False
            result=protocol.sensitivity(rows,positions)
            self.assertFalse(result['all_resolution_repeat_gate'])
            if step!=32:self.assertFalse(result['paired_engineering_gate'])
        rows,positions=paired_rows()
        self.assertFalse(protocol.sensitivity([r for r in rows if not(r['substeps']==64 and r['repeat_index']==32)],positions)['paired_engineering_gate'])


if __name__=='__main__':
    unittest.main()
