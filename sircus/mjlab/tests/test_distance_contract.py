"""CPU protocol checks; synthetic outcomes below are not simulated captures."""
from collections import Counter
import copy
import json
from pathlib import Path
import unittest

import numpy as np

from sircus.mjlab import distance_contract as protocol


def parent_inputs():
    root = Path(__file__).resolve().parents[3] / 'baselines/mjlab-height-tilt-20261102/completed'
    contract = json.loads((root / 'contract.json').read_text())
    comparison = json.loads((root / 'recipe-comparison.json').read_text())
    indexed = {r['recipe_id']: r for r in comparison['recipes']}
    board, bucket = contract['settings']['board_xy_m'], contract['settings']['bucket_xy_m']
    seeds = [dict(id=name, values=[*board, indexed[name]['board_height_m'], indexed[name]['tilt_deg'], indexed[name]['drop_gap_m']],
        bucket_xy_m=copy.deepcopy(bucket)) for name in protocol.SEED_IDS]
    return contract['scene'], seeds


def outcome(context, gap, hit=True, suffix='0', phase='coarse'):
    """Complete lightweight scorer input with explicitly synthetic outcomes."""
    return dict(id=f"{context['id']}:{suffix}", context_id=context['id'], distance_id=context['distance_id'],
        ray_id=context['ray_id'], distance_m=context['distance_m'], source_anchor_id=context['source_anchor_id'],
        role='development', phase=phase, gap_m=gap, native_release=True, route_success=hit, invalid='',
        route_quality=1 if hit else 0, board_bounce=True, closest_opening_distance_m=.01,
        max_retention_s=1. if hit else 0., external_contact_steps=0, single_bounce_success=hit,
        values=[*context['board_xy_m'], context['board_height_m'], context['tilt_deg'], gap])


def paired_rows():
    rows, positions = [], {}
    for i in range(33):
        for step in protocol.SUBSTEPS:
            row = dict(id=f'{i}-{step}', repeat_id=f'repeat-{i}', repeat_index=i, substeps=step,
                phase='nominal' if i < 3 else 'fresh', route_success=True, native_release=True)
            rows.append(row)
            positions[row['id']] = np.zeros((960, 3))
    return rows, positions


class DistanceContractTests(unittest.TestCase):
    def setUp(self):
        self.scene, self.seeds = parent_inputs()
        self.rays, self.targets = protocol.targets(self.scene, self.seeds)
        self.contexts = protocol.contexts(self.seeds, self.targets)

    def test_two_rays_fourteen_targets_and_three_explicit_unsupported_footprints(self):
        self.assertEqual([r['id'] for r in self.rays], ['original', 'corner'])
        self.assertEqual(len(self.targets), 14)
        self.assertEqual(len({t['id'] for t in self.targets}), 14)
        self.assertEqual(Counter(t['ray_id'] for t in self.targets), {'original': 7, 'corner': 7})
        rejected = {(t['ray_id'], round(t['distance_m'] * 1000)) for t in self.targets if not t['bucket_on_table']}
        self.assertEqual(rejected, {('original', 650), ('corner', 800), ('corner', 950)})
        bounds = protocol.table_bounds(self.scene)
        radius = self.scene['bucket']['outer_top_diameter_m'] / 2
        for target in self.targets:
            xy = np.asarray(target['bucket_xy_m'])
            margin = min(np.r_[xy - bounds[:, 0] - radius, bounds[:, 1] - xy - radius])
            self.assertAlmostEqual(target['table_margin_m'], margin)
            self.assertEqual(target['bucket_on_table'], margin >= .005 - 1e-12)
            self.assertAlmostEqual(np.linalg.norm(xy - target['board_xy_m']), target['distance_m'])
        reference = next(t for t in self.targets if t['id'] == 'original-d02')
        self.assertEqual(reference['bucket_xy_m'], self.seeds[0]['bucket_xy_m'])
        bad = copy.deepcopy(self.seeds)
        bad.reverse()
        with self.assertRaises(ValueError):
            protocol.targets(self.scene, bad)
        bad = copy.deepcopy(self.seeds)
        bad[1]['values'][0] += .001
        with self.assertRaises(ValueError):
            protocol.targets(self.scene, bad)

    def test_coarse_and_exact_controls_keep_finite_requested_denominators(self):
        plan = protocol.coarse_plan(self.contexts)
        self.assertEqual(len(self.contexts), 112)
        self.assertEqual(len(plan), 1904)
        self.assertEqual(set(Counter(p['context_id'] for p in plan).values()), {17})
        self.assertEqual({p['role'] for p in plan}, {'development'})
        self.assertEqual({p['phase'] for p in plan}, {'coarse'})
        for context in self.contexts:
            gaps = [p['gap_m'] for p in plan if p['context_id'] == context['id']]
            np.testing.assert_allclose(gaps, context['proposed_anchor_gap_m'] + np.linspace(-.08, .24, 17), rtol=0, atol=0)
        # Off-table queries remain requested failures; their planned gaps are not silently discarded.
        rejected_ids = {t['id'] for t in self.targets if not t['bucket_on_table']}
        self.assertEqual(sum(p['distance_id'] in rejected_ids for p in plan), 3 * 8 * 17)
        controls = protocol.control_plan(self.seeds, self.targets)
        self.assertEqual(len(controls), 24)
        for seed in self.seeds:
            rows = [p for p in controls if p['source_anchor_id'] == seed['id']]
            self.assertEqual([r['repeat_index'] for r in rows], [0, 1, 2])
            self.assertTrue(all(r['distance_id'] == 'original-d02' and r['gap_m'] == seed['values'][4] and r['role'] == 'control' for r in rows))
        settings = protocol.settings(self.scene, self.seeds)
        self.assertEqual(protocol.MAX_REQUESTS, 24 + 1904 + 1344 + 832 + 1485)
        self.assertEqual(settings['budgets']['maximum_requests'], protocol.MAX_REQUESTS)
        self.assertEqual(settings['search']['maximum_contexts'], 112 + 4 * 24)

    def test_only_unresolved_supported_far_targets_receive_twentyfour_extension_contexts(self):
        coarse = [dict(p, route_success=False) for p in protocol.coarse_plan(self.contexts)]
        extension = protocol.extension_contexts(self.seeds, self.targets, coarse)
        far = {t['id'] for t in self.targets if t['bucket_on_table'] and t['distance_m'] >= .65}
        self.assertEqual(len(far), 4)
        self.assertEqual(Counter(c['distance_id'] for c in extension), {name: 24 for name in far})
        self.assertEqual(len({c['id'] for c in extension}), 96)
        extra = protocol.extension_plan(extension)
        self.assertEqual(len(extra), 1344)
        self.assertEqual(set(Counter(p['context_id'] for p in extra).values()), {14})
        self.assertEqual({p['role'] for p in extra}, {'development'})
        self.assertTrue(all(p['phase'] == 'far_extension' for p in extra))
        for context in extension:
            np.testing.assert_allclose([p['gap_m'] for p in extra if p['context_id'] == context['id']], np.linspace(.12, .64, 14), rtol=0, atol=0)
        selected = next(iter(sorted(far)))
        matches = [r for r in coarse if r['distance_id'] == selected]
        matches[0]['route_success'] = matches[1]['route_success'] = True
        # Two successful gaps in one context do not count as two independently captured contexts.
        self.assertEqual(len(protocol.extension_contexts(self.seeds, self.targets, coarse)), 96)
        next(r for r in matches if r['context_id'] != matches[0]['context_id'])['route_success'] = True
        reduced = protocol.extension_contexts(self.seeds, self.targets, coarse)
        self.assertEqual(len(reduced), 72)
        self.assertNotIn(selected, {c['distance_id'] for c in reduced})
        for changed in (dict(coarse[0], role='test'), dict(coarse[0], phase='far_extension')):
            with self.assertRaises(ValueError):
                protocol.extension_contexts(self.seeds, self.targets, [changed])

    def test_refinement_has_at_most_four_new_midpoints_only_for_captured_development_contexts(self):
        contexts = self.contexts[:3]
        records = [outcome(c, .1 + .02*i, hit=(j != 1 and i in (2, 3)), suffix=str(i),
            phase='coarse' if j < 2 else 'far_extension') for j, c in enumerate(contexts) for i in range(6)]
        before = copy.deepcopy(records)
        plan = protocol.refinement_plan(contexts, records)
        self.assertEqual(len(plan), 8)
        self.assertEqual(Counter(r['context_id'] for r in plan), {contexts[0]['id']: 4, contexts[2]['id']: 4})
        self.assertEqual(records, before)
        for request in plan:
            source = [r for r in records if r['context_id'] == request['context_id']]
            midpoints = {(a['gap_m'] + b['gap_m']) / 2 for a, b in zip(source, source[1:])}
            self.assertIn(request['gap_m'], midpoints)
            self.assertNotIn(request['gap_m'], {r['gap_m'] for r in source})
        for changed in (dict(records[0], role='test'), dict(records[0], role='validation'), dict(records[0], phase='refinement')):
            with self.assertRaises(ValueError):
                protocol.refinement_plan(contexts, [changed])

    def test_selection_uses_actual_valid_capture_samples_and_distinct_contexts_with_fifteen_recipe_cap(self):
        supported = {t['id'] for t in self.targets if t['bucket_on_table']}
        selected_contexts = [c for c in self.contexts if c['distance_id'] in supported and c['source_anchor_id'] in protocol.SEED_IDS[:3]]
        records = [outcome(c, gap, hit, suffix=str(i)) for c in selected_contexts
            for i, (gap, hit) in enumerate(((.2, True), (.3, True), (.4, False)))]
        invalid = dict(records[0], id='invalid-high-quality', invalid='strict_geometry_rejected', route_quality=2)
        records.append(invalid)
        before = copy.deepcopy(records)
        recipes = protocol.select_recipes(records, self.targets)
        self.assertEqual(len(recipes), 15)
        self.assertEqual(len({r['source']['context_id'] for r in recipes}), 15)
        sources = {r['id']: r for r in records}
        for recipe in recipes:
            source = recipe['source']
            self.assertTrue(source['native_release'] and source['route_success'] and not source['invalid'])
            self.assertEqual(source, sources[source['id']])
            self.assertIn(recipe['gap_m'], (.2, .3))
            self.assertNotEqual(recipe['gap_m'], .25)
            self.assertIn(recipe['distance_id'], supported)
        self.assertEqual(records, before)
        self.assertEqual(protocol.select_recipes([outcome(self.contexts[0], .2, False)], self.targets), [])
        with self.assertRaises(ValueError):
            protocol.select_recipes([dict(records[0], role='test')], self.targets)

    def test_repeat_plan_is_seeded_paired_and_unclipped_with_thirtythree_unique_queries(self):
        context = self.contexts[0]
        source = outcome(context, protocol.BOUNDS['low'][4])
        source['values'][2:4] = protocol.BOUNDS['low'][2:4]
        recipe = dict(id='recipe-00', source=source, distance_id=context['distance_id'], ray_id=context['ray_id'],
            distance_m=context['distance_m'], bucket_xy_m=context['bucket_xy_m'], board_xy_m=context['board_xy_m'])
        before = copy.deepcopy(recipe)
        plan = protocol.repeat_plan([recipe])
        self.assertEqual(len(plan), 33)
        self.assertEqual(Counter(r['phase'] for r in plan), {'nominal': 3, 'fresh': 30})
        self.assertEqual(len({r['repeat_id'] for r in plan}), 33)
        self.assertEqual([r['repeat_index'] for r in plan], list(range(33)))
        self.assertEqual(plan, protocol.repeat_plan([recipe]))
        self.assertEqual(recipe, before)
        self.assertTrue(all(r['noise'] == [0, 0, 0] for r in plan[:3]))
        self.assertTrue(any(r['gap_m'] < protocol.BOUNDS['low'][4] for r in plan[3:]))
        self.assertTrue(any(r['board_height_m'] < protocol.BOUNDS['low'][2] for r in plan[3:]))
        self.assertTrue(any(r['tilt_deg'] < protocol.BOUNDS['low'][3] for r in plan[3:]))
        for query in plan:
            np.testing.assert_allclose(np.asarray(query['values'])[2:] - source['values'][2:], query['noise'], atol=1e-14, rtol=0)
            self.assertEqual(query['values'][:2], source['values'][:2])
            self.assertEqual(query['bucket_xy_m'], recipe['bucket_xy_m'])
            self.assertNotIn('substeps', query)  # One frozen query set is reused by every timestep allocation.
        self.assertEqual(protocol.SUBSTEPS, (32, 64, 128))
        settings = protocol.settings(self.scene, self.seeds)['repeatability']
        self.assertEqual((settings['nominal'], settings['fresh']), (3, 30))
        self.assertTrue(settings['same_requests_and_joints'])
        self.assertFalse(settings['clipping'])
        self.assertFalse(settings['retry'])

    def test_timestep_gates_separate_fine_pair_from_coarse_capture_threshold(self):
        rows, positions = paired_rows()
        self.assertTrue(protocol.sensitivity(rows, positions)['all_resolution_repeat_gate'])
        for row in rows:
            if row['substeps'] == 32 and row['repeat_index'] in (3, 4, 5):
                row['route_success'] = False
        result = protocol.sensitivity(rows, positions)
        self.assertTrue(result['paired_engineering_gate'])
        self.assertTrue(result['all_resolution_repeat_gate'])  # Exactly 27/30 at32 passes.
        next(r for r in rows if r['substeps'] == 32 and r['repeat_index'] == 6)['route_success'] = False
        result = protocol.sensitivity(rows, positions)
        self.assertTrue(result['paired_engineering_gate'])
        self.assertFalse(result['all_resolution_repeat_gate'])
        rows, positions = paired_rows()
        next(r for r in rows if r['substeps'] == 32 and r['repeat_index'] == 0)['route_success'] = False
        self.assertFalse(protocol.sensitivity(rows, positions)['all_resolution_repeat_gate'])

    def test_fine_pair_requires_reliability_agreement_all_pairs_and_nominal_drift(self):
        rows, positions = paired_rows()
        for row in rows:
            if row['substeps'] == 64 and row['repeat_index'] in (3, 4, 5):
                row['route_success'] = False
        result = protocol.sensitivity(rows, positions)
        self.assertEqual(result['capture_agreement_64_128'], 30)
        self.assertTrue(result['paired_engineering_gate'])  # Exactly27/30 and30/33 pass.
        next(r for r in rows if r['substeps'] == 64 and r['repeat_index'] == 6)['route_success'] = False
        self.assertFalse(protocol.sensitivity(rows, positions)['paired_engineering_gate'])
        rows, positions = paired_rows()
        for row in rows:
            if (row['substeps'], row['repeat_index']) in ((64, 3), (64, 4), (128, 5), (128, 6)):
                row['route_success'] = False
        result = protocol.sensitivity(rows, positions)
        self.assertEqual(result['capture_agreement_64_128'], 29)
        self.assertFalse(result['paired_engineering_gate'])  # Bothfresh28/30, insufficientagreement.
        rows, positions = paired_rows()
        positions['0-128'][83, 0] = .005
        self.assertTrue(protocol.sensitivity(rows, positions)['paired_engineering_gate'])
        positions['0-128'][83, 0] += 1e-6
        self.assertFalse(protocol.sensitivity(rows, positions)['paired_engineering_gate'])
        rows, positions = paired_rows()
        result = protocol.sensitivity(rows[:-1], positions)
        self.assertEqual(result['paired_attempts'], 32)
        self.assertFalse(result['paired_engineering_gate'])
        for step in (64, 128):
            rows, positions = paired_rows()
            next(r for r in rows if r['substeps'] == step and r['repeat_index'] == 0)['route_success'] = False
            self.assertFalse(protocol.sensitivity(rows, positions)['paired_engineering_gate'])
        rows, positions = paired_rows()
        next(r for r in rows if r['substeps'] == 128 and r['repeat_index'] == 0)['native_release'] = False
        self.assertFalse(protocol.sensitivity(rows, positions)['paired_engineering_gate'])


if __name__ == '__main__':
    unittest.main()
