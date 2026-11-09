import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from sircus.mjlab import prepare_bouncer_clearance as P


def write(path, value):
    path.write_text(json.dumps(value))


class PublisherTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.study = self.root / 'study'
        self.study.mkdir()
        self.report = self.study / 'report.json'
        self.audit = self.root / 'audit.json'
        grid = []
        for index, (admitted, capture, contact) in enumerate(((False,False,False),(True,False,True),(True,False,False),(True,True,False))):
            grid.append(dict(candidate_id='cell-'+str(index), grid_index=index, case_grid_index=index,
                values=[.3,-.4,.46,28,.25], bucket_xy_m=[.8,-.4], selected=index==3,
                admission=dict(passed=admitted, status='completed', reason=None if admitted else 'IK failed'),
                discovery=None if not admitted else dict(native_release=True, captured=capture, robot_contact_steps=int(contact))))
        self.raw = dict(version=P.VERSION, status='completed', contract_fingerprint='a'*64,
            grid_protocol=dict(substeps=64, selection_requires_zero_robot_contact_before_scoring_terminal=True),
            admission_summary=dict(proposals=4,admitted=3,declined=1),
            discovery_summary=dict(native_releases=3,robot_contact_shots=1,captures=1),
            cases=[dict(context_id='case-00',baseline_values=[.3,-.4,.46,28,.25],bucket_xy_m=[.8,-.4],grid=grid,
                selection=dict(candidate_id='cell-3',capture=True,diagnostic_robot_contact_steps=0),
                repeats=dict(baseline=dict(attempts=3,native_releases=3,captures=0,robot_contact_shots=3,native_misses=3),
                             selected=dict(attempts=3,native_releases=3,captures=0,robot_contact_shots=0,native_misses=3)))],
            private_path='/home/private/evidence')
        self.save_report()
        self.movie = self.root / 'movie.mp4'
        # Container-header fixture only: these tests never render physics or verify a codec.
        self.movie.write_bytes(b'\x00\x00\x00\x18ftypmp42' + b'fixture')
        frames = [dict(source_tick=i) for i in range(30)]
        self.media = dict(version=P.VIDEO,status='completed',
            source=dict(report_sha256=P.sha(self.report),audit_sha256=P.sha(self.audit),contract_fingerprint='a'*64),
            output_sha256=P.sha(self.movie),video_bytes=self.movie.stat().st_size,
            frames=frames,frames_sha256=hashlib.sha256(json.dumps(frames,sort_keys=True).encode()).hexdigest(),
            fps=30,duration_s=1,native_steps_added=0,held_arm_poses=True,sampled_ball_centers_only=True,
            clips=[dict(record_id='cell-3',label='Clear rebound / still missed',context_id='case-00',
                values=[.3,-.4,.46,28,.25],capture=False,start_s=0,end_s=1,
                events=dict(board=.2,bucket=.6,table=.7,robot=.4),private_path='/home/private/row')])
        self.manifest = self.root / 'movie.json'
        self.save_movie()

    def save_report(self):
        write(self.report,self.raw)
        write(self.audit,dict(version=P.AUDIT,status='passed',full_checks_verified=True,native_checks_verified=True,
            contract_fingerprint='a'*64,report_sha256=P.sha(self.report),checks_count=1,
            checks=[dict(passed=True)],failures=[],contact_diagnostics=[dict(record_id='cell-3',physics_steps=100,
                scoring_terminal_s=.8,capture=False,robot_contact_steps=0,
                robot_contact_steps_before_first_bucket_or_table=0,robot_contact_steps_from_first_bucket_or_table=0,
                first_board_contact_s=.2,first_bucket_contact_s=None,first_table_contact_s=.7,first_robot_contact_s=None)]))

    def save_movie(self):
        write(self.manifest,self.media)

    def snapshot(self, movie=False):
        value=P.compact(self.study,self.audit)
        if movie:value['video']=P.movie_summary(self.manifest,self.movie,value)
        return value

    def test_declines_contact_misses_clear_misses_and_success_remain_distinct(self):
        result=self.snapshot()
        self.assertEqual(len(result['cases'][0]['grid']),4)
        self.assertTrue(result['cases'][0]['selection']['capture'])
        self.assertNotIn('/home/private',json.dumps(result))
        panel=P.render_panel(result)
        for text in ('Setup declined','Miss','Capture','Failed trials are expected learning data',
                     'targeted exploration','A clear candidate can still miss','0/3','Measured contact timing'):
            self.assertIn(text,panel)
        self.assertEqual(result['cases'][0]['repeats']['selected']['captures'],0)

    def test_source_binding_and_native_audit_are_required(self):
        original=self.report.read_bytes()
        self.report.write_bytes(original+b' ')
        self.assertEqual(self.snapshot()['audit']['status'],'unverified')
        self.report.write_bytes(original)
        audit=json.loads(self.audit.read_text());audit['native_checks_verified']=False
        write(self.audit,audit)
        result=self.snapshot()
        self.assertEqual(result['audit']['status'],'unverified')
        self.assertEqual(result['contact_diagnostics'],[])
        with self.assertRaisesRegex(ValueError,'Verified study'):
            self.snapshot(True)

    def test_media_compaction_keeps_provenance_not_raw_frames_or_paths(self):
        result=self.snapshot(True)['video']
        self.assertEqual(result['frame_count'],30)
        self.assertNotIn('frames',result)
        self.assertNotIn('/home/private',json.dumps(result))
        self.assertEqual([e['label'] for e in result['clips'][0]['events']],['board','robot','bucket','table'])

    def test_movie_and_frame_map_tampering_is_rejected(self):
        self.media['frames'][0]['source_tick']=100
        self.save_movie()
        with self.assertRaisesRegex(ValueError,'Frame-map checksum'):
            self.snapshot(True)
        self.movie.write_bytes(self.movie.read_bytes()+b'changed')
        with self.assertRaisesRegex(ValueError,'checksum'):
            self.snapshot(True)

    def test_mutated_source_or_new_physics_is_rejected(self):
        original=copy.deepcopy(self.media)
        for key,value in [('native_steps_added',1),('held_arm_poses',False),('sampled_ball_centers_only',False)]:
            with self.subTest(key=key):
                self.media=copy.deepcopy(original);self.media[key]=value;self.save_movie()
                with self.assertRaisesRegex(ValueError,'held-endpoint'):
                    self.snapshot(True)
        self.media=copy.deepcopy(original);self.media['source']['report_sha256']='b'*64;self.save_movie()
        with self.assertRaisesRegex(ValueError,'different study'):
            self.snapshot(True)

    def test_bad_movie_timeline_and_nonfinite_values_are_rejected(self):
        original=copy.deepcopy(self.media)
        for key,value in [('start_s',-.1),('end_s',2),('values',[0,0,float('nan'),28,.2]),('capture',None)]:
            with self.subTest(key=key):
                self.media=copy.deepcopy(original);self.media['clips'][0][key]=value;self.save_movie()
                with self.assertRaises(ValueError):self.snapshot(True)

    def test_additive_idempotent_escaped_embedding_with_manual_playback(self):
        self.media['clips'][0]['label']='Clear </script><script>bad()</script>'
        self.save_movie();snapshot=self.snapshot(True)
        original='<html><body><!--OLD_PANEL-->existing study<!--END_OLD--></body></html>'
        result=P.embed_html(original,snapshot,self.movie.read_bytes())
        self.assertTrue(result.startswith(original.split('</body>')[0]))
        self.assertTrue(result.endswith('</body></html>'))
        self.assertEqual(P.embed_html(result,snapshot,self.movie.read_bytes()),result)
        self.assertIn('controls playsinline preload="metadata"',result)
        self.assertIn('data:video/mp4;base64,',result)
        self.assertNotIn('autoplay',result)
        self.assertNotIn('<script>bad()',result)

    def test_compare_and_swap_and_display_keep_unrelated_work(self):
        target=self.root/'index.html';original='<html><body>old</body></html>';target.write_text(original)
        snapshot=self.snapshot()
        with self.assertRaisesRegex(ValueError,'changed'):
            P.install(target,snapshot,'0'*64)
        self.assertEqual(target.read_text(),original)
        P.install(target,snapshot,P.sha(target))
        self.assertIn('old',target.read_text())
        display=self.root/'display.json';write(display,dict(other={'old':'evidence'}))
        P.publish_display(display,snapshot)
        self.assertEqual(json.loads(display.read_text())['other'],{'old':'evidence'})


if __name__=='__main__':
    unittest.main()
