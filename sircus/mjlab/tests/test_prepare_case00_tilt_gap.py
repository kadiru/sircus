import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from sircus.mjlab import prepare_case00_tilt_gap as P
from sircus.mjlab.case00_tilt_gap_contract import protocol


def write(path,value):
    path.write_text(json.dumps(value))


class PublisherTests(unittest.TestCase):
    def setUp(self):
        temp=tempfile.TemporaryDirectory();self.addCleanup(temp.cleanup)
        self.root=Path(temp.name);self.study=self.root/'study';self.study.mkdir()
        self.report=self.study/'report.json';self.audit=self.root/'audit.json'
        grid=[];base=[.3,-.4,.46,28.,.251818832]
        for tilt in protocol()['tilts_deg']:
            for offset in protocol()['gap_offsets_m']:
                index=len(grid)
                d=None if index>=6 or index==0 else dict(native_release=index!=5,captured=index==3,
                    robot_contact_steps=int(index==1),invalid_physics=index==4,invalid=None,
                    direct_capture=index==3,table_assisted_capture=False)
                grid.append(dict(candidate_id='cell-'+str(index),grid_index=index,case_grid_index=index,
                    values=[*base[:3],tilt,base[4]+offset],bucket_xy_m=[.8,-.4],selected=index==3,
                    admission=dict(passed=False if index==0 else True,status='completed',reason='IK failed' if index==0 else None),discovery=d))
        selection=[dict(candidate_id='cell-3',grid_index=3,values=grid[3]['values'],capture=True,diagnostic_robot_contact_steps=0)]
        self.raw=dict(version=P.VERSION,status='completed',contract_fingerprint='a'*64,grid_protocol=protocol(),
            admission_summary=dict(proposals=70,admitted=69,declined=1),discovery_summary=dict(native_releases=4,captures=1,robot_contact_shots=1),
            cases=[dict(context_id='case-00',baseline_values=base,bucket_xy_m=[.8,-.4],grid=grid,selection=selection,
                candidate_repeats=[dict(candidate_id='cell-3',grid_index=3,values=grid[3]['values'],discovery_record_sha256='b'*64,
                    summary=dict(attempts=3,captures=2,native_releases=3,robot_contact_shots=0))],
                repeats=dict(baseline=dict(attempts=3,captures=0,native_releases=3),selected=dict(attempts=3,captures=2,native_releases=3)))],
            private_path='/home/private/evidence')
        self.save_report()
        self.movie=self.root/'movie.mp4'
        # Header fixture only: no media codec, simulator or GPU is exercised.
        self.movie.write_bytes(b'\x00\x00\x00\x18ftypmp42'+b'fixture')
        frames=[dict(source_tick=i) for i in range(30)]
        self.media=dict(version=P.VIDEO,status='completed',
            source=dict(report_sha256=P.sha(self.report),audit_sha256=P.sha(self.audit),contract_fingerprint='a'*64),
            output_sha256=P.sha(self.movie),video_bytes=self.movie.stat().st_size,
            frames=frames,frames_sha256=hashlib.sha256(json.dumps(frames,sort_keys=True).encode()).hexdigest(),
            fps=30,duration_s=1,native_steps_added=0,held_arm_poses=True,sampled_ball_centers_only=True,
            clips=[dict(record_id='cell-3',label='Discovery capture',context_id='case-00',values=grid[3]['values'],
                capture=True,start_s=0,end_s=1,events=dict(board=.2,bucket=.4,whole_ball_entry=.5),private_path='/home/private')])
        self.manifest=self.root/'movie.json';self.save_movie()

    def save_report(self):
        write(self.report,self.raw)
        write(self.audit,dict(version=P.AUDIT,status='passed',full_checks_verified=True,native_checks_verified=True,
            contract_fingerprint='a'*64,report_sha256=P.sha(self.report),checks_count=1,checks=[dict(passed=True)],failures=[]))

    def save_movie(self):write(self.manifest,self.media)

    def snapshot(self,movie=False):
        value=P.compact(self.study,self.audit)
        if movie:value['video']=P.movie_summary(self.manifest,self.movie,value)
        return value

    def test_all_matrix_cells_remain_and_outcomes_are_distinct(self):
        snap=self.snapshot();case=snap['cases'][0]
        self.assertEqual(len(case['grid']),70)
        self.assertEqual([P.cell_status(cell)[0] for cell in case['grid'][:7]],['D','R','M','C','X','D','P'])
        panel=P.render_panel(snap)
        self.assertEqual(panel.count('aria-label="Candidate '),70)
        self.assertIn('fresh 2/3',panel)
        self.assertIn('1/1 discovery',panel)
        self.assertIn('Three successes in three unchanged trials',panel)
        self.assertNotIn('3/4',panel)
        self.assertNotIn('/home/private',json.dumps(snap))
        self.assertIn('Invalid native trial',panel)
        self.assertIn('Setup declined',panel)

    def test_all_capture_settings_get_their_own_repeat_summary(self):
        case=self.raw['cases'][0]
        second=copy.deepcopy(case['selection'][0]);second.update(candidate_id='cell-9',grid_index=9,values=case['grid'][9]['values'])
        case['selection'].append(second)
        repeated=copy.deepcopy(case['candidate_repeats'][0]);repeated.update(candidate_id='cell-9',grid_index=9,values=second['values'],summary=dict(attempts=3,captures=0,native_releases=3))
        case['candidate_repeats'].append(repeated);self.save_report()
        snap=self.snapshot()
        self.assertEqual(len(snap['cases'][0]['selection']),2)
        self.assertEqual([r['summary']['captures'] for r in snap['cases'][0]['candidate_repeats']],[2,0])
        self.assertIn('2 discovery captures selected',P.render_panel(snap))

    def test_duplicate_or_unmapped_settings_are_rejected(self):
        snap=self.snapshot();case=snap['cases'][0]
        case['grid'].append(copy.deepcopy(case['grid'][0]))
        with self.assertRaisesRegex(ValueError,'Duplicate tilt/gap'):P.matrix(case,snap['grid_protocol'])
        case['grid'].pop();case['grid'][0]['values'][3]=99
        with self.assertRaisesRegex(ValueError,'Every proposal'):P.matrix(case,snap['grid_protocol'])

    def test_source_binding_native_audit_and_provisional_selection(self):
        old=self.report.read_bytes();self.report.write_bytes(old+b' ')
        self.assertEqual(self.snapshot()['audit']['status'],'unverified')
        self.report.write_bytes(old)
        audit=json.loads(self.audit.read_text());audit['native_checks_verified']=False;write(self.audit,audit)
        with self.assertRaisesRegex(ValueError,'Verified study'):self.snapshot(True)
        self.raw['status']='running';self.raw['cases'][0]['selection']=[];self.save_report()
        self.assertIn('Discovery and repeat selection are pending',P.render_panel(self.snapshot()))

    def test_media_authentication_and_additive_escaped_html(self):
        self.media['clips'][0]['label']='Example </script><script>bad()</script>';self.save_movie()
        snap=self.snapshot(True);original='<html><body><section id="old">kept</section></body></html>'
        result=P.embed_html(original,snap,self.movie.read_bytes())
        self.assertTrue(result.startswith(original.split('</body>')[0]))
        self.assertEqual(P.embed_html(result,snap,self.movie.read_bytes()),result)
        self.assertNotIn('<script>bad()',result)
        self.assertNotIn('autoplay',result)
        self.assertIn('controls playsinline',result)
        self.assertIn('details.open=true',result)
        self.assertNotIn('frames',snap['video'])
        self.media['frames'][0]['source_tick']=900;self.save_movie()
        with self.assertRaisesRegex(ValueError,'Frame-map checksum'):self.snapshot(True)

    def test_wrong_media_source_new_physics_and_bad_timeline_rejected(self):
        original=copy.deepcopy(self.media)
        for key,value in [('native_steps_added',1),('held_arm_poses',False),('sampled_ball_centers_only',False)]:
            self.media=copy.deepcopy(original);self.media[key]=value;self.save_movie()
            with self.subTest(key=key),self.assertRaisesRegex(ValueError,'held-endpoint'):self.snapshot(True)
        self.media=copy.deepcopy(original);self.media['source']['report_sha256']='b'*64;self.save_movie()
        with self.assertRaisesRegex(ValueError,'different study'):self.snapshot(True)
        for key,value in [('start_s',-.1),('end_s',2),('values',[0,0,float('nan'),28,.2]),('capture',None)]:
            self.media=copy.deepcopy(original);self.media['clips'][0][key]=value;self.save_movie()
            with self.subTest(key=key),self.assertRaises(ValueError):self.snapshot(True)

    def test_publication_preserves_existing_work(self):
        snap=self.snapshot();target=self.root/'index.html';original='<html><body>old</body></html>';target.write_text(original)
        with self.assertRaisesRegex(ValueError,'changed'):P.install(target,snap,'0'*64)
        self.assertEqual(target.read_text(),original)
        P.install(target,snap,P.sha(target));self.assertIn('old',target.read_text())
        display=self.root/'display.json';write(display,dict(other={'old':'evidence'}));P.publish_display(display,snap)
        self.assertEqual(json.loads(display.read_text())['other'],{'old':'evidence'})


if __name__=='__main__':unittest.main()
