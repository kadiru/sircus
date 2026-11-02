import json
from pathlib import Path
import tempfile
import unittest

from sircus.mjlab.prepare_contact_report import prepare, compact_transit


class ContactDisplayTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.root = Path(self.temp.name)
        self.base = self.root/'base.json'; self.study = self.root/'study'; self.out = self.root/'display.json'
        self.study.mkdir()
        self.original = dict(status='completed', frozen_evaluation={'captures':65},
            bank_evaluation={'status':'completed'}, stable_ik_evaluation={'cells':[{'id':'old','captures':13}]})
        self.base.write_text(json.dumps(self.original))

    def tearDown(self): self.temp.cleanup()

    def write(self, path, value): path.write_text(json.dumps(value))

    def test_optional_missing_report_preserves_every_old_field(self):
        result=prepare(self.base,self.study,self.out)
        self.assertIsNone(result['contact_repeatability_evaluation'])
        for key,value in self.original.items(): self.assertEqual(result[key],value)
        self.assertEqual(json.loads(self.base.read_text()),self.original)

    def test_exact_poses_actual_radius_and_denominators_with_no_runtime_paths(self):
        source=dict(id='source',values=[.47,-.278,.397,36.15,.303],bucket_xy_m=[.16,-.63],q10=[1]*6,q14=[2]*6)
        self.write(self.study/'contract.json',{'scene':{'ball':{'radius_m':.04}}})
        self.write(self.study/'report.json',dict(status='completed',source=source,records=112,
            groups=[dict(id='articulated-16',gravity_mode='passive_only',substeps=16,
                family_results=[dict(family='nominal',attempts=32,captures=32,invalid=0)])],
            host_path='/private/credentials'))
        self.write(self.study/'plan.json',[{**source,'family':'nominal'},
            {**source,'id':'jitter','values':[.471,-.278,.397,36.15,.303],'family':'fresh'}])
        files={p:p.read_bytes() for p in self.study.iterdir()}
        result=prepare(self.base,self.study,self.out)['contact_repeatability_evaluation']
        self.assertEqual(result['total_attempts'],112)
        self.assertEqual(result['groups'][0]['captures'],32)
        self.assertEqual(len(result['configurations']),2)
        self.assertAlmostEqual(result['configurations'][0]['release_center_m'][2],.74)
        self.assertEqual(result['verification']['status'],'pending')
        self.assertNotIn('host_path',result)
        self.assertTrue(result['no_training'])
        for path,raw in files.items(): self.assertEqual(path.read_bytes(),raw)

    def test_protected_outputs_and_missing_contract_fail(self):
        for output in [self.base,self.study/'nested/view.json']:
            with self.assertRaises(ValueError): prepare(self.base,self.study,output)
        self.write(self.study/'report.json',{'status':'completed'})
        with self.assertRaises(ValueError): prepare(self.base,self.study,self.out)

    def test_fresh_results_keep_their_stage_and_both_audits_are_required(self):
        source=dict(id='source',values=[.47,-.278,.397,36.15,.303],bucket_xy_m=[.16,-.63],q10=[1]*6,q14=[2]*6)
        final=self.root/'confirmation'; final.mkdir()
        for directory,version in [(self.study,'sircus-contact-repeatability-study-v1'),(final,'sircus-contact-fresh-confirmation-v1')]:
            self.write(directory/'contract.json',{'scene':{'ball':{'radius_m':.0375}}})
            self.write(directory/'report.json',dict(version=version,contract_fingerprint=version,status='completed',source=source,records=200,
                groups=[dict(id='corrected-32',gravity_mode='passive_only',substeps=32,
                    family_results=[dict(family='fresh_perturbation',attempts=200,independent_requests=100,captures=200,invalid=0)])]))
            self.write(directory/'plan.json',[source])
        audit=self.study/'audit-final.json'
        self.write(audit,dict(status='passed',contract_fingerprint='sircus-contact-repeatability-study-v1',failures=[],checks=[{'passed':True},{'passed':True}]))
        result=prepare(self.base,self.study,self.out,retest=final,audit=audit)['contact_repeatability_evaluation']
        self.assertEqual(result['verification']['status'],'pending')
        self.write(final/'audit-confirm.json',dict(status='passed',contract_fingerprint='sircus-contact-fresh-confirmation-v1',failures=[],checks=[{'passed':True}]))
        result=prepare(self.base,self.study,self.out,retest=final,audit=audit)['contact_repeatability_evaluation']
        self.assertEqual(result['verification']['checks'],3)
        self.assertEqual(result['verification']['status'],'passed')
        self.assertEqual(result['groups'][0]['stage'],'diagnostic')
        self.assertEqual(result['groups'][1]['stage'],'confirmation')
        self.assertIn('100 × 2 resets',result['groups'][1]['label'])
        self.assertEqual(len(result['configurations']),1)
        self.assertEqual(result['total_attempts'],400)
        self.write(final/'audit-confirm.json',dict(status='passed',contract_fingerprint='wrong-study',failures=[],checks=[{'passed':True}]))
        result=prepare(self.base,self.study,self.out,retest=final,audit=audit)['contact_repeatability_evaluation']
        self.assertEqual(result['verification']['status'],'failed')
        self.assertFalse(result['retest']['verification']['identity_matched'])

    def test_rejected_path_keeps_reason_and_never_gets_preview_frames(self):
        self.write(self.study/'report.json',dict(scope='sampled geometry',trials=[
            dict(source_id='bad',passed=False,initial={'reason':'clearance'},phases=[]),
            dict(source_id='ok',passed=True,minimum_clearance_m=.0051,checked_joint_configurations=230)]))
        frame=dict(board_center_m=[1,2,3],board_quaternion_wxyz=[1,0,0,0],link_positions10=[[0,0,0]],link_positions14=[[1,0,0]])
        self.write(self.study/'preview.json',dict(paths=[{'source_id':s,'frames':[frame,frame]} for s in ['bad','ok']]))
        result=compact_transit(self.study)
        self.assertEqual(result['routes'][0]['frames'],[])
        self.assertEqual(result['routes'][0]['reason'],'clearance')
        self.assertEqual(len(result['routes'][1]['frames']),2)
        self.assertEqual(result['routes'][1]['min_clearance_mm'],5.1000000000000005)
        self.assertIn('no actuator timing claim',result['routes'][1]['preview_time_scope'])


if __name__=='__main__':unittest.main()
