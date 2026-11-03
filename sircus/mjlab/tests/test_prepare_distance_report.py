import json
import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from sircus.mjlab.prepare_distance_report import KEY, compact, far_miss_examples, prepare, watch


class DistanceDisplayTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.study = self.root / 'study'
        self.study.mkdir()
        self.base, self.output = self.root / 'base.json', self.root / 'display.json'
        self.old = dict(active_policy='unchanged', neighborhood_evaluation={'contract_fingerprint': 'old'},
            window_study_evaluation={'test': {'sac': {'captures': 0}}})
        self.base.write_text(json.dumps(self.old))

    def tearDown(self):
        self.tmp.cleanup()

    def test_optional_and_queued_reports_preserve_historical_evidence(self):
        result = prepare(self.base, self.study, self.output)
        self.assertIsNone(result[KEY])
        self.assertEqual({k: result[k] for k in self.old}, self.old)
        self.assertEqual(list(self.study.iterdir()), [])
        raw = dict(status='waiting_for_4090', phase='device_queue', queue={'reason': 'shared lock held', 'wait_s': 42})
        (self.study / 'report.json').write_text(json.dumps(raw))
        result = prepare(self.base, self.study, self.output)
        self.assertEqual(result[KEY]['queue'], raw['queue'])
        self.assertEqual({k: result[k] for k in self.old}, self.old)

    def test_distance_geometry_and_requested_native_denominators_survive_without_joint_arrays(self):
        raw = dict(settings={'board_xy_m': [.5, -.3], 'table_bounds_xy_m': [[-.2, 1.4], [-.9, .7]],
            'bucket_outer_radius_m': .1, 'rays': [{'id': 'far', 'label': 'Inward far ray'}], 'private': 'omit'},
            counts={'requests': 100, 'native_releases': 70, 'captures': 40},
            targets=[dict(id='far-850', ray_id='far', distance_m=.85, bucket_xy_m=[1.3, -.52],
                requested_contexts=8, counts={'requests': 100, 'native_releases': 70}, private='omit',
                footprint={'passed': False, 'minimum_edge_clearance_m': -.01, 'private': 'omit'})],
            contexts=[dict(id='ctx', distance_id='far-850', ray_id='far', distance_m=.85,
                bucket_xy_m=[1.3, -.52], board_height_m=.4, tilt_deg=32, board_admitted=True,
                geometry_requests=17, eligible=10, search_attempts=17, search_native_releases=10, search_captures=8, q14=[1]*6)],
            selected_recipes=[dict(id='recipe', context_id='ctx', distance_id='far-850', ray_id='far', distance_m=.85,
                bucket_xy_m=[1.3, -.52], board_height_m=.4, tilt_deg=32, gap_m=.3, release_center_height_m=.7375,
                by_substeps={'32': {'attempts': 23, 'captures': 18, 'native_releases': 20}, '128': {'attempts': 23, 'captures': 20}},
                sensitivity={'paired_engineering_gate': False, 'all_resolution_repeat_gate': False}, q10=[1]*6)])
        path = self.study / 'report.json'
        path.write_text(json.dumps(raw))
        before = path.read_bytes(), self.base.read_bytes()
        value = prepare(self.base, self.study, self.output)[KEY]
        self.assertEqual(value['settings']['table_bounds_xy_m'], raw['settings']['table_bounds_xy_m'])
        self.assertEqual(value['distances'][0]['counts']['native_releases'], 70)
        self.assertFalse(value['distances'][0]['footprint']['passed'])
        self.assertNotIn('private', value['distances'][0]['footprint'])
        self.assertEqual(value['contexts'][0]['bucket_xy_m'], [1.3, -.52])
        self.assertEqual(value['contexts'][0]['search_attempts'], 17)
        self.assertEqual(value['selected_recipes'][0]['distance_id'], 'far-850')
        self.assertEqual(value['selected_recipes'][0]['by_substeps']['32']['native_releases'], 20)
        self.assertNotIn('native_releases', value['selected_recipes'][0]['by_substeps']['128'])
        self.assertNotIn('q10', value['selected_recipes'][0])
        self.assertFalse(value['selected_recipes'][0]['sensitivity']['all_resolution_repeat_gate'])
        self.assertNotIn('q14', value['contexts'][0])
        for item in (value['settings'], value['distances'][0]):
            self.assertNotIn('private', item)
        self.assertEqual((path.read_bytes(), self.base.read_bytes()), before)

    def test_native_traces_keep_their_own_bucket_association_and_exact_observations(self):
        a, b = [[0, 0, .7], [.35, 0, .1]], [[0, 0, .8], [.8, -.2, .1]]
        value = compact(dict(trace_examples=[dict(record_id='a', distance_id='near', bucket_xy_m=[.35, 0], position_m=a),
            dict(record_id='b', distance_id='far', bucket_xy_m=[.8, -.2], position_m=b),
            dict(position_m=[[0, 0, float('nan')], [0, 0, 0]]), dict(position_m=[[True, 0, 0], [0, 0, 0]])]))
        self.assertEqual([r['position_m'] for r in value['trace_examples']], [a, b])
        self.assertEqual([r['bucket_xy_m'] for r in value['trace_examples']], [[.35, 0], [.8, -.2]])
        self.assertEqual([r['distance_id'] for r in value['trace_examples']], ['near', 'far'])

    def test_all_thirty_final_recipe_traces_include_the_last_far_distance(self):
        traces = [dict(record_id=f'recipe-{i // 2:02d}-{64 if i % 2 == 0 else 128}',
            distance_id='corner-d04' if i >= 28 else f'target-{i // 2:02d}',
            ray_id='corner' if i >= 28 else 'original', bucket_xy_m=[-.12, -.77] if i >= 28 else [.2, -.6],
            position_m=[[.47, -.28, .8], [.2, -.6, .1]]) for i in range(30)]
        value = compact(dict(trace_examples=traces + [dict(traces[0], record_id='over-budget')]))
        self.assertEqual(len(value['trace_examples']), 30)
        last = value['trace_examples'][-1]
        self.assertEqual(last['record_id'], 'recipe-14-128')
        self.assertEqual(last['distance_id'], 'corner-d04')
        self.assertEqual(last['bucket_xy_m'], [-.12, -.77])
        self.assertEqual(last['position_m'], traces[-1]['position_m'])
        self.assertNotIn('over-budget', {r['record_id'] for r in value['trace_examples']})

    def test_audit_requires_a_nonempty_matching_contract(self):
        path = self.study / 'report.json'
        audit = self.study / 'audit-distance.json'
        for fingerprint, audited in [(None, None), ('', ''), ('a', 'b')]:
            path.write_text(json.dumps(dict(status='completed', contract_fingerprint=fingerprint)))
            audit.write_text(json.dumps(dict(status='passed', contract_fingerprint=audited)))
            self.assertEqual(prepare(self.base, self.study, self.output)[KEY]['audit']['status'], 'failed')
        path.write_text(json.dumps(dict(status='completed', contract_fingerprint='a')))
        audit.write_text(json.dumps(dict(status='passed', contract_fingerprint='a', checks_count=100)))
        self.assertEqual(prepare(self.base, self.study, self.output)[KEY]['audit']['status'], 'passed')

    def test_write_guards_and_nonfinite_counts_leave_previous_display_intact(self):
        for target in (self.base, self.study / 'display.json'):
            with self.assertRaises(ValueError):
                prepare(self.base, self.study, target)
        prepare(self.base, self.study, self.output)
        before = self.output.read_bytes()
        (self.study / 'report.json').write_text(json.dumps(dict(counts={'requests': float('nan')})))
        with self.assertRaises(ValueError):
            prepare(self.base, self.study, self.output)
        self.assertEqual(self.output.read_bytes(), before)

    def test_watch_is_bounded_and_terminates_on_failed_or_completed_study(self):
        for seconds in (-1, float('inf'), 30001):
            with self.assertRaises(ValueError):
                watch(self.base, self.study, self.output, seconds)
        for status in ('failed', 'completed'):
            with mock.patch('sircus.mjlab.prepare_distance_report.prepare', side_effect=[{KEY: {'status': 'running'}}, {KEY: {'status': status}}]), \
                mock.patch('sircus.mjlab.prepare_distance_report.time.monotonic', side_effect=[0, 0]), \
                mock.patch('sircus.mjlab.prepare_distance_report.time.sleep') as sleep:
                self.assertEqual(watch(self.base, self.study, self.output, 10)[KEY]['status'], status)
                sleep.assert_called_once_with(5)


class FarMissDisplayTests(unittest.TestCase):
    """Synthetic cache fixtures validate read-only evidence handling, not physics."""
    def setUp(self):
        import numpy as np
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.study = self.root / 'study'
        self.study.mkdir()
        self.base, self.output = self.root / 'base.json', self.root / 'display.json'
        self.base.write_text(json.dumps({'neighborhood_evaluation': {'old': 'preserved'}}))
        self.targets = [dict(id='corner-650', ray_id='corner', distance_m=.65, bucket_xy_m=[.65, 0], board_xy_m=[0, 0], bucket_on_table=True),
            dict(id='corner-700', ray_id='corner', distance_m=.7, bucket_xy_m=[0, .7], board_xy_m=[0, 0], bucket_on_table=True),
            dict(id='outside', ray_id='corner', distance_m=.8, bucket_xy_m=[.8, 0], board_xy_m=[0, 0], bucket_on_table=False),
            dict(id='near', ray_id='original', distance_m=.6, bucket_xy_m=[.6, 0], board_xy_m=[0, 0], bucket_on_table=True)]
        definitions = [('no-entry', 0, False, 2., .001, False), ('entry-short', 0, True, .2, .001, False),
            ('entry-far', 0, True, .5, .03, False), ('entry-z', 0, True, .5, .02, False),
            ('entry-a', 0, True, .5, .02, False), ('capture-ignore', 0, True, 1., 0., True),
            ('b-miss', 1, True, .25, .01, False), ('outside-ignore', 2, True, 3., 0., False), ('near-ignore', 3, True, 3., 0., False)]
        self.rows = []
        for i, (name, target, entry, retention, closest, captured) in enumerate(definitions):
            t = self.targets[target]
            self.rows.append(dict(id=name, context_id='ctx-'+t['id'], distance_id=t['id'], ray_id=t['ray_id'], distance_m=t['distance_m'],
                bucket_xy_m=t['bucket_xy_m'], values=[0, 0, .4, 32, .3], gap_m=.3, native_release=True, invalid='',
                route_success=captured, whole_ball_entry=entry, max_retention_s=retention, closest_opening_distance_m=closest,
                external_contact_steps=10, native_cache_file='native.npz', native_cache_row=i, substeps=64, role='development', phase='coarse'))
        self.points = np.zeros((960, len(self.rows), 3))
        for i in range(len(self.rows)):
            self.points[:, i, 0] = np.linspace(0, .5, 960) + i*.001
            self.points[:, i, 2] = np.linspace(.8, .1, 960)
        np.savez_compressed(self.study / 'native.npz', position_m=self.points, record_ids=np.asarray([r['id'] for r in self.rows]))
        self.raw = dict(version='sircus-bucket-distance-development-v1', status='completed', phase='completed', contract_fingerprint='same', targets=self.targets,
            counts={'requests': 99, 'captures': 2}, selected_recipes=[dict(id='original-recipe', sensitivity={'paired_engineering_gate': False, 'all_resolution_repeat_gate': False})],
            trace_examples=[dict(record_id='original', distance_id='near', position_m=[[0, 0, .8], [.6, 0, .1]])])
        self.audit = dict(status='passed', contract_fingerprint='same', checks_count=1, checks=[{'passed': True}], failures=[])
        self.refresh_manifest()
        (self.study / 'audit-distance.json').write_text(json.dumps(self.audit))

    def tearDown(self):
        self.tmp.cleanup()

    def refresh_manifest(self):
        (self.study / 'records.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in self.rows))
        manifest = {name: hashlib.sha256((self.study / name).read_bytes()).hexdigest() for name in ('records.jsonl', 'native.npz')}
        self.raw['evidence_sha256'] = manifest
        (self.study / 'evidence-manifest.json').write_text(json.dumps(manifest))
        (self.study / 'report.json').write_text(json.dumps(self.raw))

    def test_opt_in_ranks_one_miss_per_supported_far_target_preserving_geometry_and_scores(self):
        import numpy as np
        snapshot = {p.name: p.read_bytes() for p in self.study.iterdir()}
        with mock.patch('numpy.load') as loader:
            default = prepare(self.base, self.study, self.output)[KEY]
            loader.assert_not_called()
        self.assertEqual([r['record_id'] for r in default['trace_examples']], ['original'])
        self.assertNotIn('far_miss_illustrations', default)
        value = prepare(self.base, self.study, self.output, include_far_misses=True)[KEY]
        examples = value['trace_examples'][1:]
        self.assertEqual([r['record_id'] for r in examples], ['entry-a', 'b-miss'])
        self.assertEqual([r['distance_id'] for r in examples], ['corner-650', 'corner-700'])
        self.assertEqual([r['bucket_xy_m'] for r in examples], [[.65, 0], [0, .7]])
        np.testing.assert_array_equal(examples[0]['position_m'], self.points[::3, 4])
        self.assertEqual(examples[0]['board_xy_m'], [0, 0])
        self.assertEqual(examples[0]['context_id'], 'ctx-corner-650')
        self.assertEqual(examples[0]['evidence_provenance']['cache_row'], 4)
        self.assertFalse(examples[0]['evidence_provenance']['allow_pickle'])
        self.assertIn('Post-hoc illustrative miss', examples[0]['selection_scope'])
        self.assertIn('external-contact steps 10', examples[0]['selection_scope'])
        self.assertEqual(value['counts'], default['counts'])
        self.assertEqual(value['selected_recipes'], default['selected_recipes'])
        self.assertEqual(value['far_miss_illustrations']['count'], 2)
        self.assertEqual(value['far_miss_illustrations']['warnings'], [])
        self.assertEqual({p.name: p.read_bytes() for p in self.study.iterdir()}, snapshot)
        again, _ = far_miss_examples(self.study, self.raw, self.audit)
        self.assertEqual(again, examples)

    def test_running_or_stale_or_internally_failed_audit_never_reads_native_arrays(self):
        for change in ('running', 'stale', 'failed', 'failed-check'):
            with self.subTest(change=change):
                raw, audit = dict(self.raw), dict(self.audit)
                if change == 'running': raw['status'] = 'running'
                if change == 'stale': audit['contract_fingerprint'] = 'old'
                if change == 'failed': audit['status'] = 'failed'
                if change == 'failed-check': audit['checks'] = [{'passed': False}]
                with mock.patch('numpy.load') as loader:
                    examples, warnings = far_miss_examples(self.study, raw, audit)
                    loader.assert_not_called()
                self.assertEqual(examples, [])
                self.assertTrue(warnings)

    def test_completed_phase_and_nonempty_exact_strictly_passed_audit_checks_are_required(self):
        changes = [lambda raw, audit: raw.pop('phase'), lambda raw, audit: raw.update(phase='fresh-128'),
            lambda raw, audit: audit.update(checks=[]), lambda raw, audit: audit.update(checks=[], checks_count=0),
            lambda raw, audit: audit.pop('checks_count'), lambda raw, audit: audit.update(checks_count=False),
            lambda raw, audit: audit.update(checks_count=True), lambda raw, audit: audit.update(checks_count=1.0),
            lambda raw, audit: audit.update(checks_count=0), lambda raw, audit: audit.update(checks_count=2),
            lambda raw, audit: audit.update(checks=[{'passed': 1}]), lambda raw, audit: audit.update(checks=[{'passed': 'yes'}]),
            lambda raw, audit: audit.update(checks=[{}])]
        for i, change in enumerate(changes):
            with self.subTest(case=i):
                raw, audit = dict(self.raw), dict(self.audit)
                change(raw, audit)
                with mock.patch('numpy.load') as loader:
                    examples, warnings = far_miss_examples(self.study, raw, audit)
                    loader.assert_not_called()
                self.assertEqual(examples, [])
                self.assertTrue(warnings)

    def test_tampered_records_or_cache_and_missing_cache_fail_closed(self):
        cache = self.study / 'native.npz'
        original_cache = cache.read_bytes()
        records = self.study / 'records.jsonl'
        original_records = records.read_bytes()
        for kind in ('cache', 'missing-cache', 'records'):
            with self.subTest(kind=kind):
                cache.write_bytes(original_cache)
                records.write_bytes(original_records)
                if kind == 'cache': cache.write_bytes(original_cache+b'tamper')
                if kind == 'missing-cache': cache.unlink()
                if kind == 'records': records.write_bytes(original_records+b'\n')
                examples, warnings = far_miss_examples(self.study, self.raw, self.audit)
                self.assertEqual(examples, [])
                self.assertTrue(warnings)

    def test_exact_column_and_target_association_are_checked_even_when_hashes_match(self):
        import numpy as np
        for kind in ('column', 'bucket', 'board', 'ids', 'shape', 'object-ids'):
            with self.subTest(kind=kind):
                # Restore an independently inspectable, complete cache fixture.
                self.rows[4].update(native_cache_row=4, bucket_xy_m=[.65, 0], values=[0, 0, .4, 32, .3])
                cache_ids = np.asarray([r['id'] for r in self.rows])
                points = self.points
                if kind == 'column': self.rows[4]['native_cache_row'] = 3
                if kind == 'bucket': self.rows[4]['bucket_xy_m'] = [0, .7]
                if kind == 'board': self.rows[4]['values'][0] = .01
                if kind == 'ids': cache_ids[[3, 4]] = cache_ids[[4, 3]]
                if kind == 'shape': points = points[:-1]
                if kind == 'object-ids': cache_ids = cache_ids.astype(object)
                np.savez_compressed(self.study / 'native.npz', position_m=points, record_ids=cache_ids)
                self.refresh_manifest()
                examples, warnings = far_miss_examples(self.study, self.raw, self.audit)
                self.assertNotIn('entry-a', {r['record_id'] for r in examples})
                self.assertTrue(warnings)
                if kind not in ('shape', 'object-ids'): self.assertEqual([r['record_id'] for r in examples], ['b-miss'])
                else: self.assertEqual(examples, [])

    def test_manifest_identity_mismatch_and_paths_outside_evidence_are_rejected(self):
        raw = dict(self.raw, evidence_sha256={'records.jsonl': 'different'})
        with mock.patch('numpy.load') as loader:
            examples, warnings = far_miss_examples(self.study, raw, self.audit)
            loader.assert_not_called()
        self.assertEqual(examples, [])
        self.assertTrue(warnings)
        self.rows[4]['native_cache_file'] = '../native.npz'
        (self.root / 'native.npz').write_bytes((self.study / 'native.npz').read_bytes())
        self.refresh_manifest()
        self.raw['evidence_sha256']['../native.npz'] = hashlib.sha256((self.root / 'native.npz').read_bytes()).hexdigest()
        (self.study / 'evidence-manifest.json').write_text(json.dumps(self.raw['evidence_sha256']))
        examples, warnings = far_miss_examples(self.study, self.raw, self.audit)
        self.assertEqual([r['record_id'] for r in examples], ['b-miss'])
        self.assertTrue(warnings)


if __name__ == '__main__':
    unittest.main()
