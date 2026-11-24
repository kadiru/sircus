"""A stale, partial or mismatched approach test cannot enter the live display."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

from sircus.mjlab import bucket_approach_publish as pub
from sircus.mjlab.benchmark_verify import sha


class PublishTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.study = self.root/'study'; self.study.mkdir()
        self.audit = self.root/'audit.json'; self.display = self.root/'display.json'
        self.rows = []; cases = []; prepared = []; preserved = []; preserved_plan = []
        def forecast():
            return dict(version='sircus-full-bucket-approach-v1', predicted_route='table',
                routes={'table':dict(status='clear', minimum_clearance_lower_bound_m=.01,
                                    original_rim_margin_m=.02, first_predicted_contact=None)})
        def group(identity, bucket, captures):
            request = dict(id=identity, bucket_xy_m=bucket, values=[.5,0,.4,28.,.2],
                selected_q10=[0.]*6, selected_q14=[0.]*6, board_heading_offset_deg=0., board_twist_deg=0.)
            ids = []
            for i in range(3):
                name = identity+f'-repeat{i}'
                self.rows.append(dict(request, id=name, native_release=True, board_first_scalar_audit_passed=True,
                    release_ball_center_m=[.5,0,.6375], native_initial_q10=[0.]*6, native_initial_q14=[0.]*6,
                    board_first_score=dict(scoring_version='sircus-board-first-native-capture-v4', route_success=i<captures, invalid_physics=False)))
                ids.append(name)
            result = dict(requested=3, completed=3, released=3, captures=captures, setup_rejected=0,
                          invalid_physics=0, robot_contacts=0, bucket_contacts=3, rows=ids)
            return request, result
        for i, identity in enumerate(pub.CASES):
            bucket = [.6+i/10, -.1]
            original, original_result = group(identity+'-original',bucket,0)
            corrected, corrected_result = group(identity+'-corrected',bucket,3)
            prepared.append(dict(id=identity,original=original,corrected=corrected,
                                 original_approach=forecast(),approaches={corrected['id']:forecast()}))
            cases.append(dict(id=identity,bucket_xy_m=bucket,results=dict(original=original_result,corrected=corrected_result)))
        for i, identity in enumerate(('04-current','10-expanded','19-current','47-expanded')):
            bucket=[.6+i/10,.1]
            request,result=group(identity,bucket,3)
            preserved_plan.append(dict(id=identity,original=request,original_approach=forecast()))
            preserved.append(dict(result,id=identity,bucket_xy_m=bucket))
        diagnostic_request, diagnostic_result=group('47-current',[1.17,.21],0)
        self.rows.extend(dict(id=f'control{i}') for i in range(18))
        self.preparation = dict(cases=prepared,preserved_cases=preserved_plan,
            diagnostic_cases=[dict(id='47-current',original=diagnostic_request,original_approach=forecast())])
        self.report = dict(version=pub.STUDY_VERSION,status='completed',phase='completed',physics_substeps=64,
            cases=cases,preservation=preserved,native_releases=51,
            diagnostic_controls=[dict(diagnostic_result,id='47-current',bucket_xy_m=[1.17,.21])],
            controls={phase:dict(status='passed',board_first_passed=True) for phase in ('controls_pre','controls_post')})
        self.write('preparation.json',self.preparation)
        self.write('contract.json',dict(source_paths={}))
        self.write('slots.json',dict(frozen='fixture'))
        self.write('native-started.json',dict(frozen='fixture'))
        (self.study/'rows.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in self.rows))
        self.write('report.json',self.report)
        self.bind()

    def tearDown(self):
        self.temp.cleanup()

    def write(self,name,value):
        (self.study/name).write_text(json.dumps(value))

    def bind(self):
        self.write('evidence-manifest.json',{p.name:sha(p) for p in self.study.iterdir()
                   if p.name not in ('report.json','evidence-manifest.json')})
        receipt=dict(version=pub.AUDIT_VERSION,status='passed',checks_count=500,frozen_before_native=True,
            comparison={k:self.report[k] for k in ('cases','preservation','diagnostic_controls')})
        for key,filename in {'report_sha256':'report.json','evidence_manifest_sha256':'evidence-manifest.json',
                'contract_sha256':'contract.json','preparation_sha256':'preparation.json',
                'choices_sha256':'slots.json','native_started_sha256':'native-started.json'}.items():
            receipt[key]=sha(self.study/filename)
        self.audit.write_text(json.dumps(receipt))

    def test_compact_publish_preserves_previous_panels_and_media(self):
        old={'live_planner':'hybrid','expanded_selector_replays':{'cases':['unchanged']},'media':list(range(17))}
        self.display.write_text(json.dumps(old))
        self.assertEqual(pub.publish(self.study,self.audit,self.display)['cases'],3)
        result=json.loads(self.display.read_text())
        self.assertEqual({k:result[k] for k in old},old)
        section=result['bucket_approach_study']
        self.assertEqual(section['native_releases'],51)
        self.assertEqual(section['cases'][0]['methods']['corrected']['captures'],3)
        self.assertEqual(len(section['preservation']),4)
        self.assertNotIn(str(self.study),json.dumps(section))
        self.assertNotIn('rows',section['cases'][0]['methods']['original'])

    def test_stale_native_evidence_or_preparation_cannot_publish(self):
        for name in ('rows.jsonl','preparation.json','slots.json','native-started.json'):
            p=self.study/name;old=p.read_bytes();p.write_bytes(old+b' ')
            with self.assertRaises(ValueError):pub.compose(self.study,self.audit)
            p.write_bytes(old)

    def test_pending_audit_controls_and_wrong_counts_fail_closed(self):
        old=json.loads(self.audit.read_text())
        for change in ({'status':'pending'},{'frozen_before_native':False},{'checks_count':0}):
            self.audit.write_text(json.dumps(dict(old,**change)))
            with self.assertRaises(ValueError):pub.compose(self.study,self.audit)
        self.audit.write_text(json.dumps(old))
        self.report['controls']['controls_post']['status']='failed';self.write('report.json',self.report);self.bind()
        with self.assertRaises(ValueError):pub.compose(self.study,self.audit)
        self.report['controls']['controls_post']['status']='passed';self.report['cases'][0]['results']['original']['captures']=1
        self.write('report.json',self.report);self.bind()
        with self.assertRaises(ValueError):pub.compose(self.study,self.audit)

    def test_cannot_overwrite_original_evidence_or_audit(self):
        for destination in (self.study/'derived.json',self.audit):
            with self.assertRaises(ValueError):pub.publish(self.study,self.audit,destination)

    def test_geometry_warning_is_prediction_not_capture_ban(self):
        approach=copy.deepcopy(self.preparation['cases'][0]['original_approach'])
        approach['routes']['table'].update(status='sampled_overlap',minimum_clearance_lower_bound_m=-.01,
            first_predicted_contact=dict(time_s=.7,center_m=[.6,-.1,.1],geom='bucket/wall05',rising=True))
        result=pub.forecast_view(approach)
        self.assertTrue(result['collision']);self.assertIn('rising',result['summary']);self.assertEqual(result['first_collision_s'],.7)
        self.assertIn('does not forbid',result['limits'])


    def test_declined_correction_stays_in_attempt_denominator(self):
        result=dict(requested=3,completed=0,released=0,captures=0,setup_rejected=3,invalid_physics=0,
                    robot_contacts=0,bucket_contacts=0)
        view=pub.count_view(result)
        self.assertEqual(view['completed'],3)
        self.assertEqual(view['setup_rejections'],3)
        self.assertEqual(view['captures'],0)

    def test_native_setting_mismatch_rejected(self):
        request=self.preparation['cases'][0]['original']
        row=copy.deepcopy(self.rows[0]);row['values'][3]=25
        with self.assertRaises(ValueError):pub.settings_view(request,[row])


if __name__ == '__main__':
    unittest.main()
