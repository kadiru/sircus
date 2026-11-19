import json
import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest import mock
import numpy as np

from sircus.mjlab.prepare_window_report import KEY, compact, confirmations, frozen_test_examples, prepare, statistics, watch


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
            prepared_contexts=[{'id':'a','board_height_m':.3,'q14':[1]*6,'private':'omit','prepared':True,'board_admitted':False}],
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
        self.assertIs(value['prepared_contexts'][0]['prepared'],True);self.assertIs(value['prepared_contexts'][0]['board_admitted'],False)

    def test_audit_requires_matching_study_identity(self):
        (self.study/'report.json').write_text(json.dumps(dict(status='completed',contract_fingerprint='a')))
        audit=self.study/'audit-window.json';audit.write_text(json.dumps(dict(status='passed',contract_fingerprint='b',checks_count=12)))
        self.assertEqual(prepare(self.base,self.study,self.output)[KEY]['audit']['status'],'failed')
        audit.write_text(json.dumps(dict(status='passed',contract_fingerprint='a',checks_count=12)))
        self.assertEqual(prepare(self.base,self.study,self.output)[KEY]['audit']['status'],'passed')

    def test_confirmation_parent_identity_and_whitelist_keep_counts_separate(self):
        external=dict(parent_contract_fingerprint='a',version='repeat',status='completed',private='omit',recipes=[dict(
            id='recipe',context_id='train',board_height_m=.3,tilt_deg=24,gap_m=.2,release_center_height_m=.5375,distance_m=.5,
            q14=[1]*6,nominal={'attempts':3,'captures':3,'native_releases':3},
            fresh={'attempts':20,'captures':18,'native_releases':19},total={'attempts':23,'captures':21,'native_releases':22,'clean':16})],
            audit={'status':'passed','checks_count':8,'private':'omit'})
        value=confirmations(external,'a');self.assertEqual(value['recipes'][0]['total']['attempts'],23)
        self.assertEqual(value['recipes'][0]['total']['native_releases'],22);self.assertEqual(value['recipes'][0]['total']['clean'],16)
        self.assertNotIn('private',value);self.assertNotIn('q14',value['recipes'][0]);self.assertNotIn('private',value['audit'])
        for parent in ('b',None,''):
            value=confirmations(external,parent);self.assertEqual(value['status'],'failed');self.assertEqual(value['recipes'],[])
        raw=dict(contract_fingerprint='a',test={'sac':{'attempts':184,'captures':13}},window_confirmations=external)
        value=compact(raw);self.assertEqual(value['test']['sac']['attempts'],184);self.assertEqual(value['window_confirmations']['recipes'][0]['total']['attempts'],23)

    def test_external_confirmation_is_read_only_and_separate_from_original_manifest(self):
        (self.study/'report.json').write_text(json.dumps({'status':'completed','contract_fingerprint':'a'}))
        (self.study/'evidence-manifest.json').write_text('{"frozen":"identity"}')
        external=self.root/'confirmation';external.mkdir()
        (external/'report.json').write_text(json.dumps(dict(parent_contract_fingerprint='a',status='completed',recipes=[])))
        (external/'audit-window-confirmation.json').write_text(json.dumps(dict(status='passed',checks_count=12)))
        before={str(p):p.read_bytes() for directory in (self.study,external) for p in directory.iterdir()}
        value=prepare(self.base,self.study,self.output,confirmation_study=external)[KEY]['window_confirmations']
        self.assertEqual(value['status'],'completed');self.assertEqual(value['audit']['checks_count'],12)
        self.assertEqual(before,{str(p):p.read_bytes() for directory in (self.study,external) for p in directory.iterdir()})
        for bad in (self.study,self.study/'nested',self.root):
            with self.assertRaises(ValueError):prepare(self.base,self.study,self.output,confirmation_study=bad)
        with self.assertRaises(ValueError):prepare(self.base,self.study,external/'view.json',confirmation_study=external)
        (external/'report.json').write_text(json.dumps(dict(parent_contract_fingerprint='other',status='completed',recipes=[{'id':'wrong'}])))
        value=prepare(self.base,self.study,self.output,confirmation_study=external)[KEY]['window_confirmations']
        self.assertEqual(value['status'],'failed');self.assertEqual(value['recipes'],[])

    def test_frozen_native_examples_require_manifest_and_record_association(self):
        rows=[dict(id=method,method=method,role='test',phase='nominal',repeat_index=0,native_release=True,invalid=None,
            context_id='test-04',gap_m=.3,route_success=method=='adaptive_surrogate',native_cache_file='native.npz',native_cache_row=i)
            for i,method in enumerate(('adaptive_surrogate','sac','uniform_bounded'))]
        records=self.study/'records.jsonl';records.write_text(''.join(json.dumps(r)+'\n' for r in rows))
        native=self.study/'native.npz';points=np.zeros((960,3,3));points[:,:,2]=np.arange(960)[:,None]/1000
        np.savez_compressed(native,position_m=points,record_ids=np.array([r['id'] for r in rows]))
        raw=dict(status='completed',contract_fingerprint='a',evidence_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (records,native)})
        audit=dict(status='passed',contract_fingerprint='a');before={p.name:p.read_bytes() for p in (records,native)}
        examples,warnings=frozen_test_examples(self.study,raw,audit)
        self.assertEqual([r['method'] for r in examples],['adaptive_surrogate','sac','uniform_bounded']);self.assertEqual(warnings,[])
        self.assertEqual(len(examples[0]['position_m']),320);self.assertEqual(examples[0]['sample_dt_s'],3/240)
        self.assertEqual(examples[0]['position_m'][1],points[3,0].tolist());self.assertIn('Illustrative outcome-selected',examples[0]['selection_scope'])
        self.assertEqual(before,{p.name:p.read_bytes() for p in (records,native)})
        raw['evidence_sha256']['native.npz']='wrong';self.assertEqual(frozen_test_examples(self.study,raw,audit)[0],[])
        self.assertIn('hash',frozen_test_examples(self.study,raw,audit)[1][0])
        np.savez_compressed(native,position_m=points,record_ids=np.array(['wrong','sac','uniform_bounded']))
        raw['evidence_sha256']['native.npz']=hashlib.sha256(native.read_bytes()).hexdigest()
        self.assertEqual(frozen_test_examples(self.study,raw,audit)[0],[])
        self.assertIn('association',frozen_test_examples(self.study,raw,audit)[1][0])
        self.assertEqual(frozen_test_examples(self.study,raw,dict(status='passed',contract_fingerprint='other'))[0],[])

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
