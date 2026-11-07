"""Synthetic read-only publication checks, not physics or model validation."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from sircus.mjlab import prepare_physics_ensemble_report as p


class PhysicsEnsemblePublisherTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name)
        self.study=self.root/'study';self.study.mkdir();self.base=self.root/'base.json';self.output=self.root/'display.json';self.audit=self.root/'proof.json'
        self.old={'policy64_study':{'original':True},'policy64_diagnosis':{'original':True},'old_array':[1,2]};self.base.write_text(json.dumps(self.old))
        self.raw=dict(version=p.VERSION,status='completed',phase='completed',contract_fingerprint='a'*64,scope='State-based learned physics, native64 held endpoints.',
            protocol=dict(validation_layouts=24,test_layouts=60,methods=['bc','ppo','ensemble'],maximum_fresh_requests=264,substeps=64,noise=False,no_test_flight_search=True,fixed_model_no_validation_selection=True),
            dataset=dict(train_requests=1024,train_native_releases=800,train_labels={'native_execution':1024,'rebound':800,'secret':'PRIVATE'},train_layouts=35,split_hashes={'train':'b'*64},host_path='/home/PRIVATE'),
            training=dict(status='completed',ensemble_members=5,fit_device='cpu',fit_steps=600,input_features=['geometry'],target_features=['rebound'],curves=[dict(member=0,step=0,train_loss=2.,full_train_loss=1.8),dict(member=0,step=600,train_loss=.1,full_train_loss=.2)],artifact=dict(path='models/ensemble.pt',sha256='c'*64,token='PRIVATE'),q10=['PRIVATE']),
            counts=dict(attempts=264,native_releases=200,captures=100),phase_counts={'validation':{'attempts':72}},
            prediction_metrics=[dict(split='validation',target='capture',support=72,brier=.1,private='PRIVATE')],
            test_comparison=[dict(method='ensemble',counts={'attempts':60,'captures':25,'native_releases':40,'clean':3,'rejected':20,'native_misses':15,'invalid':0},capture_ci95=[.3,.5],ci95_method='Wilson')],
            paired_differences=[dict(a='ensemble',b='bc',a_only=8,b_only=4,shared=17,neither=31,delta_rate=4/60,ci95=[-.02,.15],interval_method='paired bootstrap',q14=['PRIVATE'])],
            test_positions=[dict(context_id='fresh-0',method='ensemble',bucket_xy_m=[.65,0],values=[0,0,.38,28,.4],model_prediction=dict(admission_probability=.9,capture_probability=.7,conditional_capture_probability=.8,uncertainty=.03,aperture_distance_m=.04,retention_fraction=.8,ranking_score=.6,rebound_velocity_m_s=[1,0,1],rebound_basis='board_to_bucket_forward_lateral_vertical',q10=['PRIVATE']),actual=dict(native_release=True,capture=True,clean=False,invalid=None,invalid_physics=False,retention_s=1.1),q14=['PRIVATE'])],
            controls=dict(pre=dict(status='passed',counts={'attempts':6}),post=dict(status='passed',counts={'attempts':6})),gate=dict(status='passed',complete=True,controls_passed=True,no_test_flight_search=True,predictions_frozen_before_any_evaluation=True),
            trace_examples=[dict(id='native-0',method='ensemble',values=[0,0,.38,28,.4],bucket_xy_m=[.65,0],position_m=[[0,0,.82],[.65,0,.1]],predicted_position_m=[[0,0,.82],[.64,0,.11]],native_cache_sha256='d'*64,q10=['PRIVATE'])],credentials={'token':'PRIVATE'})
        self.proof=dict(version=p.AUDIT_VERSION,status='passed',contract_fingerprint='a'*64,checks_count=1,checks=[dict(passed=True,label='Synthetic fixture')],failures=[])

    def write(self):
        (self.study/'report.json').write_text(json.dumps(self.raw));(self.study/'evidence-manifest.json').write_text(json.dumps({'models/ensemble.pt':'c'*64}))
        self.proof.update(report_sha256=p.sha((self.study/'report.json').read_bytes()),evidence_manifest_sha256=p.sha((self.study/'evidence-manifest.json').read_bytes()));self.audit.write_text(json.dumps(self.proof))

    def prepare(self):return p.prepare(self.base,self.study,self.output,self.audit)[p.KEY]

    def test_additive_sanitized_snapshot_keeps_predictions_and_observations_separate(self):
        self.write();before={f:f.read_bytes() for f in (self.base,self.study/'report.json',self.study/'evidence-manifest.json',self.audit)}
        value=self.prepare();result=json.loads(self.output.read_text())
        self.assertEqual({k:result[k] for k in self.old},self.old);self.assertTrue(value['audit']['full_checks_verified']);self.assertTrue(value['audit']['bound_files_verified'])
        self.assertEqual(value['test_positions'][0]['model_prediction']['capture_probability'],.7);self.assertTrue(value['test_positions'][0]['actual']['capture'])
        self.assertEqual(value['test_positions'][0]['model_prediction']['aperture_distance_m'],.04);self.assertEqual(value['test_positions'][0]['model_prediction']['admission_probability'],.9)
        self.assertEqual(value['training']['curves'][0]['member'],0);self.assertEqual(value['training']['curves'][0]['full_train_loss'],1.8);self.assertEqual(value['dataset']['train_labels'],{'native_execution':1024,'rebound':800})
        self.assertEqual(len(value['trace_examples'][0]['position_m']),2);self.assertEqual(value['trace_examples'][0]['predicted_position_m'][1][0],.64)
        self.assertEqual(before,{f:f.read_bytes() for f in before});self.assertNotIn('PRIVATE',json.dumps(value));self.assertNotIn('q10',json.dumps(value));self.assertNotIn('q14',json.dumps(value))

    def test_queued_cpu_model_does_not_claim_native_results_or_audit_completion(self):
        self.raw.update(status='waiting_for_4090',phase='waiting_for_4090',fit_status='model_fit_completed');self.raw['counts']={'attempts':0};self.raw['test_comparison']=[];self.raw['test_positions']=[];self.write()
        value=self.prepare();self.assertEqual(value['fit_status'],'model_fit_completed');self.assertEqual(value['training']['status'],'completed');self.assertEqual(value['counts'],{'attempts':0});self.assertFalse(value['audit']['full_checks_verified'])
        self.audit.unlink();self.assertFalse(self.prepare()['audit']['full_checks_verified'])

    def test_exact_complete_literal_v1_checklist_and_hashes_are_required(self):
        original=copy.deepcopy(self.proof)
        for mutation in (dict(version='old'),dict(checks_count=0,checks=[]),dict(checks_count=True),dict(checks_count=2),dict(checks=[{'passed':1}]),dict(failures=['bad']),dict(status='failed'),dict(contract_fingerprint='b'*64)):
            self.proof=dict(original,**mutation);self.write();self.assertFalse(self.prepare()['audit']['full_checks_verified'])
        self.proof=original;self.write()
        for file in (self.study/'report.json',self.study/'evidence-manifest.json'):
            old=file.read_bytes();file.write_bytes(old+b'\n');self.assertFalse(self.prepare()['audit']['full_checks_verified']);file.write_bytes(old)

    def test_missing_values_are_not_inferred_and_nonfinite_traces_are_omitted(self):
        self.raw['trace_examples'][0]['position_m'][0][0]=float('nan');self.raw['test_positions'][0]['model_prediction']={};self.raw['test_positions'][0]['actual']={};self.write();value=self.prepare()
        self.assertEqual(value['trace_examples'],[]);self.assertEqual(value['test_positions'][0]['model_prediction'],{});self.assertEqual(value['test_positions'][0]['actual'],{})

    def test_outputs_cannot_replace_inputs_or_live_inside_source_evidence(self):
        self.write()
        for out in (self.base,self.audit,self.study/'derived.json',self.study/'report.json'):
            with self.assertRaises(ValueError):p.prepare(self.base,self.study,out,self.audit)
        self.base.write_text('[]')
        with self.assertRaises(ValueError):self.prepare()

    def test_posthoc_intervals_require_bound_complete_consistent_cohorts(self):
        self.raw['test_positions']=[];self.raw['test_comparison']=[]
        for method,captures in (('bc',20),('ppo',22),('ensemble',21)):
            self.raw['test_comparison'].append(dict(method=method,counts=dict(requested=60,native_releases=60,captures=captures,
                clean=0,rejected=0,native_misses=60-captures,invalid=0)))
            for i in range(60):
                self.raw['test_positions'].append(dict(context_id=f'fresh-{i:02d}',method=method,bucket_xy_m=[.1+i*.01,-.3],
                    values=[.5,-.2,.3,30.,.3],actual=dict(native_release=True,capture=i<captures,clean=False,invalid=None,invalid_physics=False)))
        self.write();source=(self.study/'report.json').read_bytes();value=self.prepare()
        self.assertEqual(value['derived_statistics']['status'],'computed');self.assertEqual(len(value['paired_differences']),3)
        ci=value['test_comparison'][0]['capture_ci95'];self.assertLess(ci[0],20/60);self.assertGreater(ci[1],20/60)
        self.assertEqual(value['derived_statistics']['source_report_sha256'],p.sha(source));self.assertEqual((self.study/'report.json').read_bytes(),source)
        self.raw['test_comparison'][0]['counts']['captures']+=1;self.write()
        self.assertEqual(self.prepare()['derived_statistics']['status'],'unavailable')
        self.raw['status']='queued';self.write();self.assertNotIn('derived_statistics',self.prepare())

    def test_optional_statistics_dependency_does_not_hide_audited_results(self):
        self.write()
        with patch.dict('sys.modules',{'sircus.mjlab.physics_ensemble_statistics':None}):
            value=self.prepare()
        self.assertTrue(value['audit']['full_checks_verified'])
        self.assertEqual(value['derived_statistics']['status'],'unavailable')
        self.assertEqual(value['test_comparison'][0]['counts']['captures'],25)


if __name__=='__main__':unittest.main()
