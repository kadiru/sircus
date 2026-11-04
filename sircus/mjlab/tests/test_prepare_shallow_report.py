import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from sircus.mjlab.prepare_shallow_report import KEY, VERSION, compact, full_audit_verified, prepare, watch


class ShallowDisplayTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.study = self.root / 'study'
        self.study.mkdir()
        self.base, self.output = self.root / 'base.json', self.root / 'display.json'
        self.old = dict(active_policy='existing', lowboard_evaluation={'captures': 24, 'contract_fingerprint': 'lower-parent'},
            distance_evaluation={'captures': 589}, neighborhood_evaluation={'status': 'completed'})
        self.base.write_text(json.dumps(self.old))
        self.raw = dict(version=VERSION, status='completed', phase='completed', contract_fingerprint='shallow',
            settings={'candidate': {'board_height_m': .38, 'tilt_deg': 28, 'gap_m': .3868, 'release_center_height_m': .8043,
                'ball_radius_m': .0375, 'release_headroom_m': .0150243380486807, 'q10': [1]*6}},
            counts={'requests': 28, 'native_releases': 26, 'captures': 24, 'clean': 0, 'phase_counts': {'controls': 24, 'coarse': 4}},
            targets=[{'id': 'corner-700', 'distance_m': .7, 'bucket_xy_m': [.7, 0]}],
            contexts=[dict(id='discovery', stage='shallow', distance_id='corner-700', board_height_m=.38, tilt_deg=28,
                board_admitted=True, geometry_requests=1, eligible=1, search_attempts=1, search_native_releases=1, search_captures=0, clean=0)],
            selected_recipes=[], trace_examples=[dict(record_id='native-miss', context_id='discovery', distance_id='corner-700',
                board_xy_m=[0, 0], bucket_xy_m=[.7, 0], distance_m=.7, substeps=64, route_success=False, position_m=[[0, 0, .8043], [.7, 0, .1]])])
        self.audit = dict(status='passed', contract_fingerprint='shallow', checks_count=1, checks=[{'passed': True}], failures=[])

    def tearDown(self):
        self.tmp.cleanup()

    def write(self):
        (self.study / 'report.json').write_text(json.dumps(self.raw))
        (self.study / 'audit-shallow.json').write_text(json.dumps(self.audit))

    def test_new_key_preserves_all_parents_policy_and_exact_candidate(self):
        self.write()
        before = {p.name: p.read_bytes() for p in self.study.iterdir()}
        result = prepare(self.base, self.study, self.output)
        self.assertEqual({k: result[k] for k in self.old}, self.old)
        value = result[KEY]
        self.assertEqual(value['settings']['candidate']['gap_m'], .3868)
        self.assertEqual(value['settings']['candidate']['release_center_height_m'], .8043)
        self.assertNotIn('q10', value['settings']['candidate'])
        self.assertEqual(value['contexts'][0]['stage'], 'shallow')
        self.assertEqual(value['counts']['phase_counts'], {'controls': 24, 'coarse': 4})
        self.assertEqual(value['trace_examples'][0]['bucket_xy_m'], [.7, 0])
        self.assertEqual(value['trace_examples'][0]['position_m'], self.raw['trace_examples'][0]['position_m'])
        self.assertTrue(value['audit']['full_checks_verified'])
        self.assertEqual(value['audit']['source_sha256'], hashlib.sha256(before['audit-shallow.json']).hexdigest())
        self.assertEqual({p.name: p.read_bytes() for p in self.study.iterdir()}, before)

    def test_only_own_complete_matching_full_audit_qualifies(self):
        self.assertTrue(full_audit_verified(self.raw, self.audit))
        for corrupt in [lambda r,a:r.update(version='sircus-low-board-far-development-v1'),lambda r,a:r.update(status='running'),
            lambda r,a:r.update(phase='coarse-64'),lambda r,a:a.update(contract_fingerprint='parent'),lambda r,a:a.update(checks=[]),
            lambda r,a:a.update(checks_count=False),lambda r,a:a.update(checks_count=1.0),lambda r,a:a.update(checks_count=2),
            lambda r,a:a.update(checks=[{'passed': 1}]),lambda r,a:a.update(failures=[{'passed': False}])]:
            raw,audit=copy.deepcopy(self.raw),copy.deepcopy(self.audit)
            corrupt(raw,audit)
            self.assertFalse(full_audit_verified(raw,audit))
        self.write()
        (self.study/'audit-shallow.json').unlink()
        self.raw['audit']=dict(self.audit,full_checks_verified=True)
        (self.study/'report.json').write_text(json.dumps(self.raw))
        self.assertFalse(prepare(self.base,self.study,self.output)[KEY]['audit']['full_checks_verified'])

    def test_unknown_native_and_quality_counts_are_not_filled(self):
        value=compact(dict(version=VERSION,selected_recipes=[dict(id='recipe',by_substeps={'32': {'attempts':33,'captures':1}})]))
        stats=value['selected_recipes'][0]['by_substeps']['32']
        self.assertNotIn('native_releases',stats)
        self.assertNotIn('clean',stats)

    def test_missing_queued_failed_reports_and_separate_atomic_output(self):
        self.assertIsNone(prepare(self.base,self.study,self.output)[KEY])
        self.raw.update(status='waiting_for_4090',phase='device_queue',queue={'reason':'shared lock','wait_s':20})
        self.write()
        value=prepare(self.base,self.study,self.output)[KEY]
        self.assertEqual(value['queue']['wait_s'],20)
        self.assertFalse(value['audit']['full_checks_verified'])
        for output in (self.base,self.study/'display.json'):
            with self.assertRaises(ValueError):prepare(self.base,self.study,output)
        before=self.output.read_bytes();self.raw['counts']['requests']=float('nan');self.write()
        with self.assertRaises(ValueError):prepare(self.base,self.study,self.output)
        self.assertEqual(self.output.read_bytes(),before)
        for seconds in (-1,float('inf'),30001):
            with self.assertRaises(ValueError):watch(self.base,self.study,self.output,seconds)
        with mock.patch('sircus.mjlab.prepare_shallow_report.prepare',return_value={KEY:{'status':'failed'}}):
            self.assertEqual(watch(self.base,self.study,self.output,10)[KEY]['status'],'failed')


if __name__=='__main__':unittest.main()
