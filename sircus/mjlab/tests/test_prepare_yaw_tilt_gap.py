import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from sircus.mjlab import prepare_yaw_tilt_gap as P


def write(p,x):p.write_text(json.dumps(x,allow_nan=False))

class PublisherTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name);self.study=self.root/'study';self.study.mkdir()
        self.audit=self.root/'audit.json';self.html=self.root/'index.html';self.css=self.root/'style.css';self.snapshot=self.root/'snapshot.json'
        self.html.write_text('<html><body><video src="old.mp4"></video><!--OLD-->history</body></html>');self.css.write_text('body{color:white}/*OLD*/')
        self.contract={'protocol':{'substeps':64,'maximum_requests':223}};self.contract['fingerprint']=P.digest(self.contract);write(self.study/'contract.json',self.contract)
        self.rows=[dict(id=f'r{i}',context_id=P.FAR,phase='discovery' if i<49 else 'references' if i<58 else 'controls_pre',repeat=0,variant='grid',native_release=False,route_success=False,invalid='finite_ik_unresolved',values=[.65,-.2,.45,28,.1],bucket_xy_m=[1.1,-.53],requested_gripper_yaw_rad=-.63) for i in range(76)]
        (self.study/'rows.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in self.rows))
        self.manifest={n:P.sha(self.study/n) for n in ('rows.jsonl','contract.json')};write(self.study/'evidence-manifest.json',self.manifest)
        self.report=dict(version=P.STUDY,status='completed',phase='completed',expected_requests=76,maximum_requests=223,contract_fingerprint=self.contract['fingerprint'],evidence_sha256=self.manifest,counts=dict(attempts=76,captures=0,native_releases=0,setup_rejections=76),control_gates={k:dict(status='passed') for k in ('pre','post')})
        write(self.study/'report.json',self.report);self.proof()
        self.patch=patch.object(P,'recordings',return_value=[]);self.patch.start();self.addCleanup(self.patch.stop)

    def proof(self):
        write(self.audit,dict(version=P.AUDIT,status='passed',full_checks_verified=True,native_checks_verified=True,failures=[],checks=[dict(passed=True)],checks_count=1,report_sha256=P.sha(self.study/'report.json'),evidence_manifest_sha256=P.sha(self.study/'evidence-manifest.json'),contract_fingerprint=self.contract['fingerprint']))

    def compact(self):return P.compact(self.study,audit_path=self.audit,now=123)

    def test_complete_audit_bound_request_ledger(self):
        out=self.compact();self.assertEqual(out['study']['audit']['status'],'passed');self.assertEqual(len(out['study']['requests']),58)
        self.assertEqual(P.baseline_summary(out)['counts']['attempts'],76)
        self.assertTrue(all(r['capture'] is False and r['native_release'] is False for r in out['study']['requests']))

    def test_unverified_audit_hides_outcomes(self):
        a=P.read(self.audit);a['native_checks_verified']=False;write(self.audit,a)
        out=self.compact();self.assertEqual(out['study']['requests'],[]);self.assertNotIn('captures',out['study']['counts']);self.assertEqual(out['study']['audit']['status'],'unverified')
        with self.assertRaises(ValueError):P.baseline_summary(out)

    def test_changed_rows_and_incomplete_denominator_rejected(self):
        (self.study/'rows.jsonl').write_text('{}\n')
        with self.assertRaises(ValueError):self.compact()

    def test_wrong_clock_contract_rejected(self):
        self.contract['protocol']['substeps']=128;self.contract.pop('fingerprint');self.contract['fingerprint']=P.digest(self.contract);write(self.study/'contract.json',self.contract)
        self.report['contract_fingerprint']=self.contract['fingerprint'];write(self.study/'report.json',self.report)
        with self.assertRaises(ValueError):self.compact()

    def test_additive_install_preserves_old_bytes_and_detects_conflict(self):
        oldh=self.html.read_text();oldc=self.css.read_text();h,c=P.sha(self.html),P.sha(self.css)
        P.refresh(self.study,snapshot_path=self.snapshot,target_html=self.html,target_css=self.css,audit_path=self.audit,expected_html_sha256=h,expected_css_sha256=c)
        s=self.html.read_text();before,rest=s.split(P.START);_,after=rest.split(P.END);self.assertEqual(before+after,oldh)
        self.assertEqual(self.css.read_text().split(P.CSS_START)[0],oldc)
        self.assertIn('id="yawTiltGap"',s)
        with self.assertRaises(ValueError):P.refresh(self.study,snapshot_path=self.snapshot,target_html=self.html,target_css=self.css,audit_path=self.audit,expected_html_sha256=h)

    def test_encoded_snapshot_cannot_escape_script_or_comment(self):
        s=P.encoded(dict(version=P.VERSION,value='</script>*/<&>'))
        self.assertNotIn('</script>',s);self.assertNotIn('*/',s);self.assertEqual(json.loads(s)['value'],'</script>*/<&>')

    def test_queue_failure_is_not_zero_percent_success(self):
        other=self.root/'unstarted';state=self.root/'state.json';write(state,dict(status='failed',worker_started=False,finished_unix_s=123))
        out=P.compact(other,pipeline_path=state);self.assertEqual(out['study']['status'],'not_run');self.assertEqual(out['study']['counts'],dict(attempts=0))

    def movie_fixture(self):
        row=dict(id='first',record_sha256='c'*64,values=[.65,-.2,.45,28,.1],bucket_xy_m=[1.1,-.53],capture=False,native_release=True,trace=dict(native_cache_sha256='d'*64,native_cache_row=0))
        snapshot=self.compact();snapshot['study']['recordings']=[row]
        movie=self.root/'movie.mp4';movie.write_bytes(b'\x00\x00\x00\x20ftypisom'+b'0'*50)
        frames=[dict(frame=i,video_time_s=i/30,clip_index=0,record_id='first',source_tick=i*512,source_time_s=i/30) for i in range(30)]
        clip=dict(record_id='first',record_sha256='c'*64,values=row['values'],bucket_xy_m=row['bucket_xy_m'],capture=False,native_release=True,no_flight=False,native_cache_sha256='d'*64,native_cache_row=0,start_s=0.,end_s=1.,label='First repeat',context_id=P.FAR,variant='selected')
        audit=snapshot['study']['audit'];meta=self.root/'movie.json'
        value=dict(version=P.VIDEO,status='completed',native_steps_added=0,held_arm_poses=True,sampled_ball_centers_only=True,interpolation=False,backend=dict(software_only=True,cuda_visible_devices=''),source=dict(report_sha256=snapshot['study']['report_sha256'],audit_sha256=audit['proof_sha256'],contract_fingerprint=snapshot['study']['contract_fingerprint']),video_bytes=movie.stat().st_size,output_sha256=P.sha(movie),fps=30,duration_s=1.,frames=frames,frames_sha256=P.digest(frames),clips=[clip])
        write(meta,value);return snapshot,meta,movie,value

    def test_movie_requires_bound_samples_no_invented_motion(self):
        snapshot,meta,movie,value=self.movie_fixture();self.assertEqual(P.movie_summary(meta,movie,snapshot)['duration_s'],1.)
        for edit in (lambda v:v.update(interpolation=True),lambda v:v['backend'].update(software_only=False),lambda v:v['clips'][0].update(record_id='later-win'),lambda v:v['source'].update(report_sha256='e'*64)):
            v=copy.deepcopy(value);edit(v);write(meta,v)
            with self.assertRaises(ValueError):P.movie_summary(meta,movie,snapshot)

    def test_movie_install_and_refresh_keep_prior_panels(self):
        snapshot,meta,movie,value=self.movie_fixture();self.html.write_text(P.embed_html(self.html.read_text(),snapshot));write(self.snapshot,snapshot)
        before=self.html.read_text();P.publish_movie(self.snapshot,self.html,meta,movie,expected_html_sha256=P.sha(self.html));after=self.html.read_text()
        placeholder='<div data-yaw-tilt-gap="movies"></div>';a,b=after.split(P.VIDEO_START);_,c=b.split(P.VIDEO_END)
        self.assertEqual(a+placeholder+c,before);self.assertIn('data:video/mp4;base64,',after)
        refreshed=P.embed_html(after,snapshot);self.assertIn(P.VIDEO_START,refreshed);self.assertEqual(refreshed.count('<video'),2)

    def test_no_publish_inside_immutable_evidence(self):
        with self.assertRaises(ValueError):P.refresh(self.study,snapshot_path=self.study/'new.json',target_html=self.html,target_css=self.css,audit_path=self.audit)

if __name__=='__main__':unittest.main()
