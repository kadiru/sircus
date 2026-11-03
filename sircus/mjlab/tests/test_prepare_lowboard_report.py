import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from sircus.mjlab.prepare_lowboard_report import KEY, VERSION, compact, full_audit_verified, prepare, watch


class LowBoardDisplayTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.study = self.root / 'study'
        self.study.mkdir()
        self.base, self.output = self.root / 'base.json', self.root / 'display.json'
        self.old = dict(active_policy='unchanged', distance_evaluation={'captures': 2, 'contract_fingerprint': 'old-distance'},
            neighborhood_evaluation={'status': 'completed'}, window_study_evaluation={'test': {'sac': {'captures': 0}}})
        self.base.write_text(json.dumps(self.old))
        self.raw = dict(version=VERSION, status='completed', phase='completed', contract_fingerprint='new-contract',
            settings={'grid': {'board_heights_m': [.22, .24], 'tilts_deg': [28, 32], 'release_headroom_m': [.005, .015], 'private': 'omit'},
                'reach_limit': {'ball_center_height_m': .8, 'board_height_plus_gap_m': .7625,
                    'formula': 'prior ceiling minus radius minus board H minus requested headroom', 'scope': 'cheap spherical prior only', 'q10': [1]*6},
                'control_target': {'id': 'control', 'bucket_xy_m': [.2, -.6], 'q14': [1]*6}},
            counts={'requests': 20, 'native_releases': 10, 'captures': 2}, targets=[dict(id='corner-650', ray_id='corner', distance_m=.65, bucket_xy_m=[0, -.7])],
            contexts=[dict(id='context', distance_id='corner-650', stage='development', board_height_m=.22, tilt_deg=28,
                board_admitted=True, geometry_requests=5, eligible=3, search_attempts=5, search_native_releases=3, search_captures=1)],
            selected_recipes=[dict(id='recipe', context_id='context', distance_id='corner-650', board_height_m=.22, tilt_deg=28, gap_m=.5375, release_center_height_m=.795,
                by_substeps={'64': {'attempts': 33, 'captures': 30, 'native_releases': 32}}, sensitivity={'paired_engineering_gate': True, 'all_resolution_repeat_gate': False})])
        self.audit = dict(status='passed', contract_fingerprint='new-contract', checks_count=2, checks=[{'passed': True}, {'passed': True}], failures=[])

    def tearDown(self):
        self.tmp.cleanup()

    def write(self):
        (self.study / 'report.json').write_text(json.dumps(self.raw))
        (self.study / 'audit-lowboard.json').write_text(json.dumps(self.audit))

    def test_missing_and_partial_studies_preserve_every_parent_and_solver_value(self):
        result = prepare(self.base, self.study, self.output)
        self.assertIsNone(result[KEY])
        self.assertEqual({k: result[k] for k in self.old}, self.old)
        self.raw.update(status='waiting_for_4090', phase='device_queue', queue={'wait_s': 30, 'reason': 'shared GPU lock'})
        self.write()
        result = prepare(self.base, self.study, self.output)
        self.assertFalse(result[KEY]['audit']['full_checks_verified'])
        self.assertEqual(result[KEY]['queue']['reason'], 'shared GPU lock')
        self.assertEqual({k: result[k] for k in self.old}, self.old)

    def test_compact_geometry_ceiling_provenance_and_denominators_do_not_infer_releases(self):
        self.write()
        before = {p.name: p.read_bytes() for p in self.study.iterdir()}
        value = prepare(self.base, self.study, self.output)[KEY]
        self.assertEqual(value['settings']['grid']['board_heights_m'], [.22, .24])
        self.assertNotIn('private', value['settings']['grid'])
        self.assertNotIn('q10', value['settings']['reach_limit'])
        self.assertNotIn('q14', value['settings']['control_target'])
        self.assertEqual(value['settings']['reach_limit']['ball_center_height_m'], .8)
        self.assertIn('cheap spherical prior', value['settings']['reach_limit']['scope'])
        self.assertEqual(value['contexts'][0]['stage'], 'development')
        self.assertEqual(value['contexts'][0]['search_native_releases'], 3)
        self.assertEqual(value['selected_recipes'][0]['by_substeps']['64']['native_releases'], 32)
        self.assertFalse(value['selected_recipes'][0]['sensitivity']['all_resolution_repeat_gate'])
        self.assertTrue(value['audit']['full_checks_verified'])
        self.assertEqual(value['audit']['source_sha256'], hashlib.sha256(before['audit-lowboard.json']).hexdigest())
        self.assertNotIn('checks', value['audit'])
        self.assertEqual({p.name: p.read_bytes() for p in self.study.iterdir()}, before)
        self.assertEqual(compact(dict(selected_recipes=[{'id': 'unknown', 'by_substeps': {'32': {'attempts': 33, 'captures': 1}}}]))['selected_recipes'][0]['by_substeps']['32'].get('native_releases'), None)

    def test_full_audit_requires_exact_version_completed_phase_nonempty_strict_checks_and_identity(self):
        self.assertTrue(full_audit_verified(self.raw, self.audit))
        cases = [lambda r, a: r.update(version='other'), lambda r, a: r.update(status='running'), lambda r, a: r.update(phase='fresh-128'),
            lambda r, a: r.update(contract_fingerprint=''), lambda r, a: a.update(contract_fingerprint='stale'), lambda r, a: a.update(status='failed'),
            lambda r, a: a.update(checks=[]), lambda r, a: a.update(checks_count=0), lambda r, a: a.update(checks_count=True),
            lambda r, a: a.update(checks_count=2.0), lambda r, a: a.update(checks_count=1), lambda r, a: a.update(checks=[{'passed': 1}, {'passed': True}]),
            lambda r, a: a.update(checks=[{'passed': 'yes'}, {'passed': True}]), lambda r, a: a.update(failures=[{'reason': 'bad'}])]
        for i, corrupt in enumerate(cases):
            with self.subTest(case=i):
                raw, audit = dict(self.raw), dict(self.audit)
                corrupt(raw, audit)
                self.assertFalse(full_audit_verified(raw, audit))

    def test_forged_summary_or_embedded_audit_is_not_a_full_audit(self):
        self.raw['audit'] = dict(self.audit, full_checks_verified=True)
        (self.study / 'report.json').write_text(json.dumps(self.raw))
        self.assertFalse(prepare(self.base, self.study, self.output)[KEY]['audit']['full_checks_verified'])
        self.audit.update(full_checks_verified=True, checks=[], checks_count=0)
        self.write()
        self.assertFalse(prepare(self.base, self.study, self.output)[KEY]['audit']['full_checks_verified'])

    def test_native_trace_coordinates_and_target_identity_remain_exact(self):
        points = [[.5, -.3, .8], [0, -.7, .1]]
        self.raw['trace_examples'] = [dict(record_id='actual', context_id='context', source_context_id='source-context', distance_id='corner-650', distance_m=.65,
            board_xy_m=[.5, -.3], bucket_xy_m=[0, -.7], substeps=128, route_success=False, position_m=points, scope='Actual recorded development illustration')]
        value = compact(self.raw)['trace_examples'][0]
        self.assertEqual(value['position_m'], points)
        self.assertEqual(value['bucket_xy_m'], [0, -.7])
        self.assertEqual(value['distance_id'], 'corner-650')
        self.assertEqual(value['source_context_id'], 'source-context')
        self.assertEqual(value['selection_scope'], 'Actual recorded development illustration')
        self.assertFalse(value['route_success'])

    def test_write_guards_nonfinite_values_and_bounded_watch(self):
        for target in (self.base, self.study / 'display.json'):
            with self.assertRaises(ValueError):
                prepare(self.base, self.study, target)
        self.write()
        prepare(self.base, self.study, self.output)
        before = self.output.read_bytes()
        self.raw['counts']['requests'] = float('nan')
        self.write()
        with self.assertRaises(ValueError):
            prepare(self.base, self.study, self.output)
        self.assertEqual(self.output.read_bytes(), before)
        for seconds in (-1, float('nan'), 30001):
            with self.assertRaises(ValueError):
                watch(self.base, self.study, self.output, seconds)
        with mock.patch('sircus.mjlab.prepare_lowboard_report.prepare', return_value={KEY: {'status': 'failed'}}):
            self.assertEqual(watch(self.base, self.study, self.output, 10)[KEY]['status'], 'failed')


if __name__ == '__main__':
    unittest.main()
