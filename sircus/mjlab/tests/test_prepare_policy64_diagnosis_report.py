"""Synthetic publication binding checks; not a native or independent proof."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from sircus.mjlab import prepare_policy64_diagnosis_report as p


class DiagnosisPublisherTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
        self.original=self.root/'parent';self.original.mkdir();self.native=self.root/'native';self.native.mkdir()
        self.base=self.root/'base.json';self.output=self.root/'display.json';self.report=self.root/'combined.json';self.audit=self.root/'proof.json';self.manifest=self.native/'evidence-manifest.json'
        self.parent_report=self.original/'report.json';self.parent_manifest=self.original/'evidence-manifest.json';self.parent_audit=self.root/'parent-audit.json'
        self.old={'policy64_study':{'original':True},'impact650_study':{'original':True}}
        self.base.write_text(json.dumps(self.old));self.parent_report.write_text(json.dumps(dict(version=p.PARENT_VERSION,status='completed',phase='completed',contract_fingerprint=p.PARENT_FP)));self.parent_manifest.write_text('{}')
        parent=dict(version=p.PARENT_AUDIT_VERSION,status='passed',contract_fingerprint=p.PARENT_FP,checks_count=1,checks=[dict(passed=True,label='Synthetic publisher fixture')],failures=[],report_sha256=p.sha(self.parent_report.read_bytes()),evidence_manifest_sha256=p.sha(self.parent_manifest.read_bytes()))
        self.parent_audit.write_text(json.dumps(parent))
        self.patch=patch.multiple(p,PARENT_REPORT_SHA=p.sha(self.parent_report.read_bytes()),PARENT_MANIFEST_SHA=p.sha(self.parent_manifest.read_bytes()),PARENT_AUDIT_SHA=p.sha(self.parent_audit.read_bytes()));self.patch.start();self.addCleanup(self.patch.stop);self.addCleanup(self.tmp.cleanup)
        source=dict(study_version=p.PARENT_VERSION,contract_fingerprint=p.PARENT_FP,report_sha256=p.PARENT_REPORT_SHA,manifest_sha256=p.PARENT_MANIFEST_SHA,original_audit_sha256=p.PARENT_AUDIT_SHA,substeps=64,selected=dict(samples=1024,sha256='a'*64,weight_sha256='b'*64),host_path='/home/SECRET')
        side=dict(record_id='bc-test-00',record_sha256='c'*64,values=[.5,-.2,.3,30.,.4],native_release=True,capture=True,clean=False,outcome='capture',geometry_evidence=dict(reason=None,best_checked_clearance_m=.01,minimum_required_clearance_m=.005,q10=['SECRET']),first_flight=dict(available=True,first_crossing=None,cutoff_reason='contact',forward_error_m=None,lateral_error_m=None,rebound_velocity_m_s=[1.,2.,3.],q14=['SECRET']),q10=['SECRET'])
        trace=dict(record_id='bc-test-00',record_sha256='c'*64,method='bc',context_id='test-00',values=side['values'],bucket_xy_m=[.2,0],position_m=[[.5,-.2,.7],[.2,0,.1]],sample_dt_s=1/60,native_cache_sha256='d'*64,q10=['SECRET'])
        self.raw=dict(version=p.VERSION,status='completed',phase='completed',contract_fingerprint='new-native',source=source,paired_counts=dict(layouts=50,both_capture=13,bc_only=7,ppo_only=5,neither=25,bc=dict(requested=50,native=43,captures=20,clean=3,rejected=7,native_misses=23),ppo=dict(requested=50,native=39,captures=18,clean=4,rejected=11,native_misses=21),transitions=[dict(bc_outcome='capture',ppo_outcome='rejected',count=5)]),pairs=[dict(context_id='test-00',bucket_xy_m=[.2,0],bc=side,ppo=dict(side,record_id='ppo-test-00',capture=False,native_release=False,outcome='rejected'),delta_values=[.001,0,-.01,1.,.02],delta_release_height_m=.01,shared_anchor=dict(same_id=True,same_values=True,same_observation=True,private_path='/home/SECRET'))],examples=[dict(context_id='test-00',basis='original_TEST_now_diagnosis',bc_trace=trace,ppo_trace=None,projection_reference=dict(origin_xy_m=[.2,0],direction_from_xy_m=[.5,-.2]))],counterfactuals=[dict(id='swap',context_id='test-00',basis='posthoc_geometry_only',status='not_executed',geometry_passed=None,native_release=None,capture=None,new_native_requests=0,intervention=dict(copied_parameters=['gap'],from_method='bc',to_method='ppo'),q14=['SECRET'])],native_study=dict(version=p.NATIVE_VERSION,status='completed',phase='completed',counts=dict(attempts=96,native_releases=70,captures=25),control_gates=dict(pre=dict(status='passed',counts={'attempts':6}),post=dict(status='passed',counts={'attempts':6})),contexts=[dict(context_id='test-00',variants={'bc_exact':dict(attempts=3,captures=3,native_releases=3)})],q14=['SECRET']),optimization_history=dict(mean_post_update_kl=.3022,mean_post_update_ratio_clip_fraction=.8164,bc_retention_penalty=False,updates=[]),model_options=dict(status='proposed_not_trained',recommendation='No model trained here',sources=[dict(title='Paper',url='https://arxiv.org/abs/1805.12114'),dict(title='Private',url='https://user:SECRET@example.test/')],private_path='/home/SECRET'),credentials={'token':'SECRET'})
        self.proof=dict(version=p.AUDIT_VERSION,status='passed',contract_fingerprint='new-native',checks_count=1,checks=[dict(passed=True,label='Synthetic fixture only')],failures=[],auditor_sha256='e'*64,native_report_sha256='f'*64,frozen_analysis_sha256='a'*64)

    def write(self):
        self.report.write_text(json.dumps(self.raw));self.manifest.write_text(json.dumps({'analysis.json':'a'*64}))
        self.proof.update(report_sha256=p.sha(self.report.read_bytes()),evidence_manifest_sha256=p.sha(self.manifest.read_bytes()));self.audit.write_text(json.dumps(self.proof))

    def run_prepare(self):return p.prepare(self.base,self.report,self.manifest,self.audit,self.parent_report,self.parent_manifest,self.parent_audit,self.output)

    def test_additive_both_proofs_and_exact_observations_preserved_without_private_fields(self):
        self.write();before={f:f.read_bytes() for f in (self.report,self.manifest,self.audit,self.parent_report,self.parent_manifest,self.parent_audit)};result=self.run_prepare();v=result[p.KEY]
        self.assertEqual({k:result[k] for k in self.old},self.old);self.assertTrue(v['source']['audit']['full_checks_verified']);self.assertTrue(v['audit']['full_checks_verified']);self.assertTrue(v['native_study']['audit']['full_checks_verified'])
        self.assertEqual(v['paired_counts']['bc_only'],7);self.assertEqual(v['native_study']['variants'][0]['variant'],'bc_exact');self.assertEqual(v['native_study']['controls']['pre']['counts']['attempts'],6)
        self.assertIsNone(v['pairs'][0]['bc']['first_flight']['first_crossing']);self.assertEqual(v['pairs'][0]['bc']['first_flight']['rebound_velocity_m_s'],[1.,2.,3.]);self.assertEqual(v['examples'][0]['projection_reference']['direction_from_xy_m'],[.5,-.2])
        self.assertEqual(v['optimization_history']['mean_post_update_kl'],.3022);self.assertIsNone(v['counterfactuals'][0]['capture']);self.assertEqual(len(v['model_options']['sources']),1)
        text=json.dumps(v);self.assertNotIn('SECRET',text);self.assertNotIn('q10',text);self.assertNotIn('q14',text);self.assertEqual(before,{f:f.read_bytes() for f in before})

    def test_original_source_stays_verified_while_new_native_or_proof_is_pending(self):
        self.raw['native_study'].update(status='running',phase='diagnostic');self.raw['native_study']['counts']['attempts']=12;self.write();self.audit.unlink()
        v=self.run_prepare()[p.KEY];self.assertTrue(v['source']['audit']['full_checks_verified']);self.assertFalse(v['audit']['full_checks_verified']);self.assertFalse(v['native_study']['audit']['full_checks_verified'])

    def test_exact_full_new_proof_and_all_three_new_bindings_required(self):
        old=copy.deepcopy(self.proof)
        for edit in (dict(version='old'),dict(checks=[],checks_count=0),dict(checks_count=True),dict(checks_count=2),dict(checks=[{'passed':1}]),dict(failures=['failed']),dict(native_report_sha256=None),dict(frozen_analysis_sha256='bad')):
            self.proof=dict(old,**edit);self.write();self.assertFalse(self.run_prepare()[p.KEY]['audit']['full_checks_verified'])
        self.proof=old;self.write()
        for file in (self.report,self.manifest):
            before=file.read_bytes();file.write_bytes(before+b'\n');self.assertFalse(self.run_prepare()[p.KEY]['audit']['full_checks_verified']);file.write_bytes(before)

    def test_parent_proof_or_source_identity_tampering_cannot_be_promoted(self):
        self.write();self.parent_audit.write_bytes(self.parent_audit.read_bytes()+b'\n');v=self.run_prepare()[p.KEY];self.assertFalse(v['source']['audit']['full_checks_verified']);self.assertFalse(v['audit']['full_checks_verified'])
        self.raw['source']['contract_fingerprint']='wrong';self.write();self.assertFalse(self.run_prepare()[p.KEY]['source']['audit']['full_checks_verified'])

    def test_missing_observations_remain_unknown_and_output_guards_hold(self):
        self.raw['pairs'][0]['bc']['first_flight']=None;self.raw['examples'][0]['bc_trace']['position_m'][0][0]=float('nan');self.write();v=self.run_prepare()[p.KEY]
        self.assertNotIn('first_flight',v['pairs'][0]['bc']);self.assertNotIn('bc_trace',v['examples'][0])
        for out in (self.base,self.report,self.audit,self.native/'derived.json',self.original/'derived.json'):
            with self.assertRaises(ValueError):p.prepare(self.base,self.report,self.manifest,self.audit,self.parent_report,self.parent_manifest,self.parent_audit,out)


if __name__=='__main__':unittest.main()
