import copy
import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from sircus.mjlab import board_pose_recovery_publish as mod


class PublishTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name); self.study = self.root/'study'; self.study.mkdir()
        self.audit = self.root/'audit.json'; self.media = self.root/'media'; self.media.mkdir()
        self.plan = dict(version=mod.VERSION, cases=[])
        self.report = dict(version=mod.VERSION, status='completed', phase='completed', physics_substeps=64,
            progress=dict(completed=21, planned=21), controls={k:dict(status='passed') for k in ('controls_pre','controls_post')}, cases=[])
        for i, key in enumerate(mod.CASES):
            original = dict(id=key+'-original', values=[.6,-.1,.4,28,.3], bucket_xy_m=[.9+i*.1,-.1],
                selected_q10=[0]*6, selected_q14=[0]*6, board_heading_offset_deg=0, board_twist_deg=0)
            selected = copy.deepcopy(original); selected.update(id=key+'-candidate',values=[.65,-.12,.4,30,.32],board_heading_offset_deg=10)
            self.plan['cases'].append(dict(id=key, original=original, proposals=[{},{}], records=[{},{}], admitted=[selected], candidates=[selected]))
            def group(variant, outcomes):
                rows=[dict(id=key+'-'+variant+'-'+str(n),context_id=key,candidate_id=selected['id'],repeat=n,
                    native_release=True,invalid_physics=False,native_cache_sha256='a'*64) for n in range(len(outcomes))]
                return dict(requested=len(outcomes),completed=len(outcomes),released=len(outcomes),
                    captures=sum(x in ('direct','table') for x in outcomes),robot_contacts=outcomes.count('flight_miss'),outcomes=outcomes,rows=rows)
            self.report['cases'].append(dict(id=key,best_candidate=selected,
                confirmed_settings=[dict(candidate_id=selected['id'],request=selected,repeat_captures=3,repeat_completed=3)],
                results=dict(original=group('original',['flight_miss']*3),search=group('search',['direct']),repeat=group('repeat',['direct']*3))))
        self.save()

    def write(self,path,value):path.write_text(json.dumps(value))
    def save(self):
        self.write(self.study/'preparation.json',self.plan);self.write(self.study/'report.json',self.report)
        self.write(self.study/'evidence-manifest.json',{'identity':'example'})
        self.write(self.audit,dict(status='passed',report_sha256=mod.sha(self.study/'report.json'),
            plan_sha256=mod.sha(self.study/'preparation.json'),evidence_manifest_sha256=mod.sha(self.study/'evidence-manifest.json')))

    def test_exact_settings_separate_search_and_repeats(self):
        section,*_=mod.compose(self.study,self.audit);row=section['cases'][0]
        self.assertEqual(row['preparation'],dict(proposed=2,checked=2,admitted=1,selected=1))
        self.assertEqual(row['results']['search']['captures'],1);self.assertEqual(row['results']['repeat']['captures'],3)
        self.assertEqual(row['settings']['selected']['heading_offset_deg'],10)
        self.assertNotIn('rows',row['results']['search']);self.assertEqual(section['controls']['completed'],18)

    def test_audit_binds_all_evidence_and_no_audit_remains_pending(self):
        self.assertEqual(mod.compose(self.study)[0]['audit_status'],'pending')
        for filename in ('report.json','preparation.json','evidence-manifest.json'):
            self.save();p=self.study/filename;p.write_text(p.read_text()+' ')
            with self.assertRaisesRegex(ValueError,'Audit does not bind'):mod.compose(self.study,self.audit)

    def test_selected_setting_must_be_frozen(self):
        self.report['cases'][0]['best_candidate']=copy.deepcopy(self.report['cases'][0]['best_candidate'])
        self.report['cases'][0]['best_candidate']['values'][0]=.9;self.save()
        with self.assertRaisesRegex(ValueError,'not a frozen candidate'):mod.compose(self.study)

    def test_video_plan_uses_first_fresh_repeat_and_identity_checked_movie(self):
        section,_,report=mod.compose(self.study,self.audit);tasks=mod.media_plan(section,report)
        self.assertEqual(len(tasks),6);self.assertIn('-repeat-0',tasks[1]['record_id'])
        task=tasks[1];movie=self.media/(task['stem']+'.mp4');movie.write_bytes(b'test-film')
        receipt=dict(status='completed',native_steps_added=0,held_arm_poses=True,output_sha256=mod.sha(movie),
            context_id=task['case_id'],record_id=task['record_id'],trace_sha256=task['trace_sha256'],label=task['label'])
        self.write(movie.with_suffix('.json'),receipt)
        assets=mod.media_assets(section,report,self.media);self.assertIn(task['stem'],assets)
        self.assertNotIn('base64',json.dumps(section))
        receipt['record_id']='another';self.write(movie.with_suffix('.json'),receipt)
        with self.assertRaisesRegex(ValueError,'Wrong recorded shot'):mod.media_assets(section,report,self.media)

    def test_publication_preserves_other_studies_and_assets(self):
        display=self.root/'display.json';self.write(display,dict(old_study={'captures':17}))
        index=self.root/'index.html';index.write_text('<script id="hybridMissVideoAssets">{"keep":1}</script><script id="boardPoseVideoAssets" type="application/json">{}</script>')
        args=SimpleNamespace(study=self.study,audit=self.audit,display_report=display,runtime_index=index,media_dir=None,video_plan=self.root/'video-plan.json')
        before=mod.sha(self.study/'report.json');mod.publish(args)
        self.assertEqual(mod.read(display)['old_study'],{'captures':17});self.assertEqual(mod.sha(self.study/'report.json'),before)
        self.assertIn('<script id="hybridMissVideoAssets">{"keep":1}</script>',index.read_text())
        args.display_report=self.study/'report.json'
        with self.assertRaisesRegex(ValueError,'separate from evidence'):mod.publish(args)


if __name__=='__main__':unittest.main()
