"""Synthetic fixed64 publication tests; no simulator, training or native proof."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from sircus.mjlab.prepare_policy64_report import AUDIT_VERSION, KEY, VERSION, compact, full_audit_verified, prepare, watch


class Policy64PublisherTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name);self.study=self.root/'study';self.study.mkdir()
        self.base,self.output=self.root/'base.json',self.root/'display.json'
        self.old={'impact650_study':{'original':True},'tilt650_study':{'original':True},'policy':{'unchanged':True}}
        self.base.write_text(json.dumps(self.old))
        a=dict(path='models/selected.npz',sha256='b'*64,weight_sha256='c'*64,samples=512,token='SECRET')
        self.raw=dict(version=VERSION,status='completed',phase='completed',contract_fingerprint='policy-fp',protocol=dict(train_requests=1024,updates=32,validation_layouts=12,test_layouts=50,maximum_fresh_requests=1284,milestones=[0,256,512,1024],substeps=64,noise=False,private_path='/home/SECRET'),bootstrap_summary=dict(requests=6144,native_releases=3000,captures=1000,clean=12,layouts=35,successful_layouts=20,teacher_count=35,labels_scope='Historical TRAIN labels only',credentials={'token':'SECRET'}),train_requests=1024,updates=32,counts=dict(attempts=1284,captures=100,clean=2,invalid=4),curves=[dict(samples=512,validation=dict(attempts=12,native_releases=12,captures=7,clean=2,invalid=0,mean_reward=.8,q10=[1]*6),artifact=a)],selected=a,artifacts=dict(initial=a,bc=a,selected=a,private=a),test_comparison=[dict(method='ppo',counts=dict(attempts=50,native_releases=45,captures=25,clean=10,invalid=5,mean_reward=.5),samples=512,artifact_sha256='b'*64,q14=[1]*6)],test_positions=[dict(context_id='test-00',method='ppo',bucket_xy_m=[.65,0],values=[0,0,.38,28,.4],board_bucket_distance_m=.65,capture=True,clean=False,native_release=True,invalid=None,q10=[1]*6)],control_gates=dict(pre=dict(status='passed'),post=dict(status='passed')),gate=dict(status='passed',private_path='/home/SECRET'),trace_examples=[dict(record_id='ppo-test-00',context_id='test-00',method='ppo',substeps=64,board_xy_m=[0,0],bucket_xy_m=[.65,0],distance_m=.65,board_height_m=.38,values=[0,0,.38,28,.4],position_m=[[0,0,.8175],[.65,0,.1]],route_success=True,route_quality=1,whole_ball_entry=True,max_retention_s=1,invalid=None,invalid_physics=False,scalar_score_audit_passed=True,q10=[1]*6,private_path='/home/SECRET')],source_sha256={'sircus/mjlab/policy64_study.py':'d'*64,'/home/SECRET':'a'*64})
        self.audit=dict(version=AUDIT_VERSION,status='passed',contract_fingerprint='policy-fp',checks_count=1,checks=[{'passed':True}],failures=[])
        self.audit_path=self.root/'study-audit.json'

    def tearDown(self):self.tmp.cleanup()

    def write(self):
        (self.study/'report.json').write_text(json.dumps(self.raw));(self.study/'evidence-manifest.json').write_text(json.dumps({'models/selected.npz':'b'*64}))
        self.audit.update(report_sha256=hashlib.sha256((self.study/'report.json').read_bytes()).hexdigest(),evidence_manifest_sha256=hashlib.sha256((self.study/'evidence-manifest.json').read_bytes()).hexdigest())
        self.audit_path.write_text(json.dumps(self.audit))

    def test_additive_exact_schema_preserves_progress_artifact_hashes_and_old_panels(self):
        self.write();before={p.name:p.read_bytes() for p in self.study.iterdir()};result=prepare(self.base,self.study,self.output);v=result[KEY]
        self.assertEqual({k:result[k] for k in self.old},self.old);self.assertTrue(v['audit']['full_checks_verified'])
        self.assertEqual(v['protocol']['substeps'],64);self.assertEqual(v['protocol']['maximum_fresh_requests'],1284)
        self.assertEqual(v['bootstrap_summary']['requests'],6144);self.assertEqual(v['train_requests'],1024);self.assertEqual(v['updates'],32)
        self.assertEqual(v['curves'][0]['validation']['captures'],7);self.assertEqual(v['selected']['weight_sha256'],'c'*64)
        self.assertEqual(v['test_comparison'][0]['artifact_sha256'],'b'*64);self.assertEqual(v['test_positions'][0]['bucket_xy_m'],[.65,0])
        self.assertEqual(len(v['trace_examples'][0]['position_m']),2);self.assertEqual({p.name:p.read_bytes() for p in self.study.iterdir()},before)
        encoded=json.dumps(v);self.assertNotIn('SECRET',encoded);self.assertNotIn('q10',encoded);self.assertNotIn('q14',encoded);self.assertNotIn('private_path',encoded)

    def test_only_completed_matching_exact_full_audit_can_verify(self):
        self.assertTrue(full_audit_verified(self.raw,self.audit))
        for edit in (lambda r,a:r.update(status='running'),lambda r,a:r.update(phase='test'),lambda r,a:r.update(version='sircus-impact650-study-v1'),lambda r,a:a.update(version='old'),lambda r,a:a.update(contract_fingerprint='other'),lambda r,a:a.update(checks=[]),lambda r,a:a.update(checks_count=0),lambda r,a:a.update(checks_count=True),lambda r,a:a.update(checks_count=2),lambda r,a:a.update(checks=[{'passed':1}]),lambda r,a:a.update(failures=['failure'])):
            r,a=copy.deepcopy(self.raw),copy.deepcopy(self.audit);edit(r,a);self.assertFalse(full_audit_verified(r,a))

    def test_external_proof_and_byte_bindings_fail_closed_after_tampering(self):
        self.write();explicit=self.root/'explicit-proof.json';explicit.write_bytes(self.audit_path.read_bytes());self.audit_path.unlink()
        self.assertFalse(prepare(self.base,self.study,self.output)[KEY]['audit']['full_checks_verified'])
        self.assertTrue(prepare(self.base,self.study,self.output,audit_path=explicit)[KEY]['audit']['full_checks_verified'])
        for name in ('report.json','evidence-manifest.json'):
            p=self.study/name;before=p.read_bytes();p.write_bytes(before+b'\n');self.assertFalse(prepare(self.base,self.study,self.output,audit_path=explicit)[KEY]['audit']['full_checks_verified']);p.write_bytes(before)
        with self.assertRaises(ValueError):prepare(self.base,self.study,explicit,audit_path=explicit)

    def test_missing_counts_and_nonfinite_trace_do_not_create_results(self):
        raw=dict(version=VERSION,status='running',curves=[dict(samples=0,validation={'attempts':0})],trace_examples=[{'position_m':[[0,0,float('nan')],[0,0,0]]}]);v=compact(raw)
        self.assertNotIn('captures',v['curves'][0]['validation']);self.assertNotIn('train_requests',v);self.assertEqual(v['trace_examples'],[])
        self.assertEqual(v['test_comparison'],[]);self.assertEqual(v['test_positions'],[])
        raw=copy.deepcopy(self.raw);raw['selected']['path']='/home/SECRET/model.npz';self.assertNotIn('path',compact(raw)['selected'])
        raw['selected']['path']='../escape';self.assertNotIn('path',compact(raw)['selected'])
        raw['selected']['path']='https://private:token@example.test/model';self.assertNotIn('path',compact(raw)['selected'])

    def test_actual_producer_trajectory_alias_keeps_observed_board_and_cache_identity(self):
        raw=copy.deepcopy(self.raw);trace=raw['trace_examples'][0]
        trace.update(id=trace.pop('record_id'),trajectory_m=trace.pop('position_m'),trajectory_dt_s=.01,board_center_m=[.0001,0,.3802],native_cache_file='native/test-ppo.npz',native_cache_row=0,native_cache_sha256='e'*64)
        raw['selected']=None;value=compact(raw);published=value['trace_examples'][0]
        self.assertEqual(value['selected'],{});self.assertEqual(published['sample_dt_s'],.01);self.assertEqual(published['board_center_m'],[.0001,0,.3802])
        self.assertEqual(published['native_cache_sha256'],'e'*64);self.assertEqual(published['id'],'ppo-test-00');self.assertEqual(len(published['position_m']),2)
        self.assertNotIn('trajectory_m',published);self.assertNotIn('q10',published)

    def test_output_guards_nan_and_bounded_watch(self):
        self.assertIsNone(prepare(self.base,self.study,self.output)[KEY]);self.write()
        for out in (self.base,self.study/'display.json'):
            with self.assertRaises(ValueError):prepare(self.base,self.study,out)
        self.raw['counts']['attempts']=float('nan');self.write()
        with self.assertRaises(ValueError):prepare(self.base,self.study,self.output)
        for seconds in (-1,float('nan'),30001):
            with self.assertRaises(ValueError):watch(self.base,self.study,self.output,seconds)


if __name__=='__main__':unittest.main()
