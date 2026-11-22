import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace

from sircus.mjlab import hybrid_miss_publish as mod


class PublishTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.study = self.root/'study'; self.study.mkdir()
        self.diag_path = self.root/'diagnosis.json'
        self.audit_path = self.root/'audit.json'
        self.media = self.root/'media'; self.media.mkdir()
        self.values = [.6, -.1, .38, 28., .3868]
        self.plan = dict(version=mod.VERSION, repeats=3, cases=[])
        self.report = dict(version=mod.VERSION, status='completed', phase='completed', physics_substeps=64,
                           progress=dict(completed=18,planned=18), cases=[], controls={k:dict(status='passed') for k in ('controls_pre','controls_post')})
        self.diag = dict(version='sircus-hybrid-miss-diagnosis-v1',status='completed',cases=[])
        for i,key in enumerate(mod.CASES):
            request = dict(values=list(self.values), bucket_xy_m=[.9+i*.1,-.1], selected_q10=[0]*6, selected_q14=[0]*6,
                           board_heading_offset_deg=0,board_twist_deg=0,id=key+'-hybrid')
            corrected = {**request,'id':key+'-alternate','selected_q14':[1]*6}
            self.plan['cases'].append(dict(id=key,original=request,corrected=corrected,route_for_ranking='direct',
                original_prediction={'direct':{'arm_clearance_m':-.02}},predictions={corrected['id']:{'direct':{'arm_clearance_m':.02}}}))
            counts=dict(requested=3,completed=3,released=3,captures=0,direct=0,table=0,robot_contacts=3,outcomes=['flight_miss']*3,rows=[])
            self.report['cases'].append(dict(id=key,results=dict(original=dict(counts),corrected=dict(counts))))
            self.diag['cases'].append(dict(id=key+'-hybrid',bucket_xy_m=request['bucket_xy_m'],values=self.values,
                observed_events=[dict(channel='board',start_s=.2),dict(channel='robot',start_s=.4)],
                reconstructed_first_robot_contacts=[{'geometries':['ball/geom','Arm14/link3/collision/0']}],
                conditional_no_robot_flight={'direct':{'conservative_rim_band_clearance_m':.03}},
                original_prediction={'predicted_route':'direct'},trace={'sha256':'a'*64}))
        self.save()

    def write(self,path,value): path.write_text(json.dumps(value))

    def save(self):
        self.write(self.study/'preparation.json',self.plan);self.write(self.study/'report.json',self.report);self.write(self.diag_path,self.diag)
        self.write(self.audit_path,dict(status='passed',report_sha256=mod.sha(self.study/'report.json'),plan_sha256=mod.sha(self.study/'preparation.json')))

    def test_compose_exact_settings_diagnosis_and_hash_bound_audit(self):
        section,*_=mod.compose(self.study,self.diag_path,self.audit_path)
        self.assertEqual(section['audit_status'],'passed')
        self.assertEqual(section['controls'],dict(status='passed',completed=18,planned=18))
        self.assertEqual(section['cases'][0]['settings']['original']['drop_gap_m'],.3868)
        self.assertIn('Arm14/link3/collision/0 at 0.400 s',section['cases'][0]['diagnosis']['summary'])
        self.assertIn('a prediction, not another simulated drop',section['cases'][0]['diagnosis']['summary'])
        self.assertNotIn('rows',section['cases'][0]['results']['original'])

    def test_no_audit_is_pending_and_changed_report_rejects_prior_audit(self):
        self.assertEqual(mod.compose(self.study,self.diag_path)[0]['audit_status'],'pending')
        self.report['status']='running';self.write(self.study/'report.json',self.report)
        with self.assertRaisesRegex(ValueError,'Audit does not bind current report'):mod.compose(self.study,self.diag_path,self.audit_path)

    def test_changes_outside_correction_and_mismatched_diagnosis_fail(self):
        self.plan['cases'][0]['corrected']['values']=[.61,-.1,.38,28,.3868];self.save()
        with self.assertRaisesRegex(ValueError,'More than board-arm'):mod.compose(self.study,self.diag_path)
        self.plan['cases'][0]['corrected']['values']=list(self.values)
        self.diag['cases'][0]['bucket_xy_m']=[5,5];self.save()
        with self.assertRaisesRegex(ValueError,'Original geometry differs'):mod.compose(self.study,self.diag_path)

    def test_known_setup_decline_becomes_failed_request_never_release(self):
        self.plan['cases'][1]['corrected']=None
        self.report['cases'][1]['results']['corrected'].update(completed=0,released=0,robot_contacts=0,setup_rejected=3,outcomes=[])
        self.save();section,*_=mod.compose(self.study,self.diag_path,self.audit_path)
        row=section['cases'][1]['results']['corrected']
        self.assertEqual(row['completed'],3);self.assertEqual(row['released'],0)
        self.assertEqual(row['outcomes'],['setup_rejected']*3)
        self.report['status']='running';self.save()
        self.assertEqual(mod.compose(self.study,self.diag_path)[0]['cases'][1]['results']['corrected']['completed'],0)

    def clip(self):
        movie=self.media/'case03-original.mp4';movie.write_bytes(b'example-test-movie')
        receipt=dict(status='completed',native_steps_added=0,held_arm_poses=True,output_sha256=mod.sha(movie),
            context_id=mod.CASES[0],record_id=mod.CASES[0]+'-hybrid',trace_sha256='a'*64,label='Saved original miss')
        self.write(movie.with_suffix('.json'),receipt)
        return movie,receipt

    def test_media_identity_bound_and_data_not_in_report(self):
        movie,receipt=self.clip();section,plan,report,diag=mod.compose(self.study,self.diag_path)
        assets=mod.media_assets(section,plan,report,diag,self.media)
        self.assertTrue(assets['case03-original']['url'].startswith('data:video/mp4;base64,'))
        self.assertEqual(section['cases'][0]['videos']['original']['asset_id'],'case03-original')
        self.assertNotIn('base64',json.dumps(section))
        receipt['trace_sha256']='b'*64;self.write(movie.with_suffix('.json'),receipt)
        with self.assertRaisesRegex(ValueError,'Video native trace differs'):mod.media_assets(section,plan,report,diag,self.media)

    def test_embed_changes_only_owned_placeholder(self):
        html='<html><script id="old">{"url":"data:old"}</script><script type="application/json" id="hybridMissVideoAssets">{}</script><p>existing</p></html>'
        updated=mod.embed_html(html,{'clip':{'url':'data:video/mp4;base64,AA==','trace_sha256':'a'*64}})
        self.assertIn('<script id="old">{"url":"data:old"}</script>',updated)
        self.assertIn('<p>existing</p>',updated)
        self.assertEqual(mod.ASSET_PATTERN.sub('',html),mod.ASSET_PATTERN.sub('',updated))
        with self.assertRaisesRegex(ValueError,'exactly one'):mod.embed_html('<html></html>',{})
        with self.assertRaisesRegex(ValueError,'exactly one'):mod.embed_html(html+html,{})

    def test_publish_preserves_historical_keys_and_raw_bytes(self):
        display=self.root/'display.json';self.write(display,{'historical':{'capture':17}})
        index=self.root/'index.html';index.write_text('<script id="hybridMissVideoAssets" type="application/json">{}</script>')
        before={p:mod.sha(p) for p in (self.study/'preparation.json',self.study/'report.json',self.diag_path,self.audit_path)}
        mod.publish(SimpleNamespace(study=self.study,diagnosis=self.diag_path,audit=self.audit_path,display_report=display,runtime_index=index,media_dir=None))
        self.assertEqual(mod.read(display)['historical'],{'capture':17})
        self.assertEqual(before,{p:mod.sha(p) for p in before})
        with self.assertRaisesRegex(ValueError,'separate from evidence'):
            mod.publish(SimpleNamespace(study=self.study,diagnosis=self.diag_path,audit=self.audit_path,display_report=self.study/'report.json',runtime_index=None,media_dir=None))


if __name__=='__main__': unittest.main()
