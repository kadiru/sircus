import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from sircus.mjlab.prepare_window_report import KEY, compact, prepare, statistics, watch


class WindowDisplayTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
        self.base=self.root/'base.json';self.study=self.root/'study';self.study.mkdir();self.output=self.root/'display.json'
        self.original={'status':'completed','frozen_evaluation':{'captures':3},'corrected_grid_evaluation':{'old':'preserved'}}
        self.base.write_text(json.dumps(self.original))

    def tearDown(self):self.tmp.cleanup()

    def test_missing_study_preserves_old_sections_without_creating_evidence(self):
        value=prepare(self.base,self.study,self.output)
        self.assertIsNone(value[KEY])
        for key,row in self.original.items():self.assertEqual(value[key],row)
        self.assertFalse((self.study/'report.json').exists())

    def test_native_denominator_is_never_inferred_from_requests(self):
        value=statistics({'attempts':20,'captures':15,'invalid':2,'native_releases':18})
        self.assertEqual(value['native_releases'],18);self.assertEqual(value['attempts'],20)
        self.assertEqual(value['capture_rate'],.75)
        lo,hi=value['descriptive_95pct_wilson'];self.assertLess(lo,.75);self.assertGreater(hi,.75)
        self.assertNotIn('native_releases',statistics({'attempts':20,'captures':15}))
        self.assertIsNone(statistics({'attempts':0,'captures':0})['capture_rate'])

    def test_compact_native_traces_and_sampled_runs_do_not_change_source(self):
        raw=dict(version='study',status='running',contract_fingerprint='exact',settings={'bucket_xy_m':[.65,-.4],'private':'omit'},
            prepared_contexts=[{'id':'a','board_height_m':.3,'q14':[1]*6,'private':'omit'}],
            selected_checkpoints={'sac':{'milestone':504,'training_record_ids':['private'],'path':'checkpoints/sac.pt'}},
            learning_curve=[dict(method='sac',milestone=504,validation={'attempts':18,'native_releases':14,'captures':4})],
            test={'sac':{'attempts':184,'native_releases':100,'captures':30}},
            observed_windows=[dict(context_id='a',points=[{'gap_m':.2,'attempts':1,'captures':1,'native_releases':1}],
                observed_hit_runs=[{'low_observed_gap_m':.2,'high_observed_gap_m':.2,'points':1}])],
            trace_examples=[dict(context_id='a',gap_m=.2,position_m=[[0,0,.7],[0,0,.4],[.2,0,.2]],route_success=False),
                dict(context_id='bad',position_m=[[0,0,float('nan')],[0,0,0]])])
        path=self.study/'report.json';path.write_text(json.dumps(raw));before=path.read_bytes()
        value=prepare(self.base,self.study,self.output)[KEY]
        self.assertEqual(path.read_bytes(),before);self.assertEqual(len(value['trace_examples']),1)
        self.assertEqual(value['trace_examples'][0]['position_m'],raw['trace_examples'][0]['position_m'])
        self.assertEqual(value['test']['sac']['attempts'],184)
        self.assertEqual(value['learning_curve'][0]['validation']['native_releases'],14)
        self.assertEqual(value['observed_windows'][0]['observed_hit_runs'][0]['points'],1)
        for target in (value['settings'],value['prepared_contexts'][0]):self.assertNotIn('private',target)
        self.assertNotIn('q14',value['prepared_contexts'][0]);self.assertNotIn('training_record_ids',value['selected_checkpoints']['sac'])

    def test_audit_requires_matching_study_identity(self):
        (self.study/'report.json').write_text(json.dumps(dict(status='completed',contract_fingerprint='a')))
        audit=self.study/'audit-window.json';audit.write_text(json.dumps(dict(status='passed',contract_fingerprint='b',checks_count=12)))
        self.assertEqual(prepare(self.base,self.study,self.output)[KEY]['audit']['status'],'failed')
        audit.write_text(json.dumps(dict(status='passed',contract_fingerprint='a',checks_count=12)))
        self.assertEqual(prepare(self.base,self.study,self.output)[KEY]['audit']['status'],'passed')

    def test_write_guards_and_nonfinite_evidence_preserve_last_display(self):
        for output in (self.base,self.study/'derived/report.json'):
            with self.assertRaises(ValueError):prepare(self.base,self.study,output)
        prepare(self.base,self.study,self.output);before=self.output.read_bytes()
        (self.study/'report.json').write_text(json.dumps({'counts':{'requests':float('nan')}}))
        with self.assertRaises(ValueError):prepare(self.base,self.study,self.output)
        self.assertEqual(self.output.read_bytes(),before)

    def test_watcher_is_bounded_and_does_not_start_training(self):
        for seconds in (-1,float('nan'),30001):
            with self.assertRaises(ValueError):watch(self.base,self.study,self.output,seconds)
        with mock.patch('sircus.mjlab.prepare_window_report.prepare',side_effect=[{KEY:{'status':'running'}},{KEY:{'status':'completed'}}]),\
             mock.patch('sircus.mjlab.prepare_window_report.time.monotonic',side_effect=[0,0]),\
             mock.patch('sircus.mjlab.prepare_window_report.time.sleep') as sleep:
            value=watch(self.base,self.study,self.output,10)
        self.assertEqual(value[KEY]['status'],'completed');sleep.assert_called_once_with(5)


if __name__=='__main__':unittest.main()
