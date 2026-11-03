import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from sircus.mjlab.prepare_distance_report import KEY, compact, prepare, watch


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


if __name__ == '__main__':
    unittest.main()
