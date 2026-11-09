import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from sircus.mjlab import prepare_height_tilt_replay as P
from sircus.mjlab import prepare_table_height_tilt as S


def write(path, value):
    path.write_text(json.dumps(value))


class ReplayPublisherTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        study = self.root / 'study'
        study.mkdir()
        self.report = study / 'report.json'
        write(self.report, dict(version=S.VERSION, status='completed', contract_fingerprint='a'*64,
                                grid_protocol=dict(substeps=64), cases=[]))
        self.audit = self.root / 'study-audit.json'
        write(self.audit, dict(version=S.AUDIT, status='passed', full_checks_verified=True,
                               contract_fingerprint='a'*64, report_sha256=P.sha(self.report),
                               checks_count=1, checks=[dict(passed=True)], failures=[]))
        self.movie = self.root / 'movie.mp4'
        # Header fixture only. No simulator, video codec, or GPU is used by these tests.
        self.movie.write_bytes(b'\x00\x00\x00\x18ftypmp42' + b'fixture recorded media')
        frames = [dict(source_tick=tick) for tick in range(30)]
        self.raw = dict(version=P.VERSION, status='completed',
                        source=dict(report_sha256=P.sha(self.report), audit_sha256=P.sha(self.audit),
                                    contract_fingerprint='a'*64),
                        output_sha256=P.sha(self.movie), video_bytes=self.movie.stat().st_size,
                        duration_s=1, fps=30, frames=frames,
                        frames_sha256=hashlib.sha256(json.dumps(frames, sort_keys=True).encode()).hexdigest(),
                        held_arm_poses=True, native_steps_added=0, sampled_ball_centers_only=True,
                        private_path='/home/private/trace.npz',
                        clips=[dict(record_id='height-tilt-case-00-g23', label='Case 00',
                                    context_id='case-00', start_s=0, end_s=1, values=[.1,.2,.46,28,.25],
                                    capture=False, events=[dict(label='board',time_s=.2),dict(label='robot',time_s=.4)],
                                    held_joints_rad=[0]*12, private_path='/home/private/row.json')])
        self.manifest = self.root / 'manifest.json'
        self.save()

    def save(self):
        write(self.manifest, self.raw)

    def compact(self):
        return P.compact(self.manifest, self.movie, self.report, self.audit)

    def test_verified_snapshot_keeps_identity_but_not_dense_frames_or_host_paths(self):
        result = self.compact()
        self.assertEqual(result['frame_count'],30)
        self.assertEqual(result['clips'][0]['record_id'],'height-tilt-case-00-g23')
        self.assertEqual(result['clips'][0]['events'][1]['time_s'],.4)
        self.assertEqual(result['manifest_sha256'],P.sha(self.manifest))
        self.assertNotIn('/home/private',json.dumps(result))
        self.assertNotIn('held_joints_rad',json.dumps(result))
        self.assertNotIn('frames',result)

    def test_changed_movie_or_frame_map_fails_before_publication(self):
        self.movie.write_bytes(self.movie.read_bytes()+b'changed')
        with self.assertRaises(ValueError):
            self.compact()

    def test_category_mapping_is_rendered_in_actual_time_order(self):
        self.raw['clips'][0]['events']={'board':.2,'bucket':.6,'table':.7,'robot':.4,'floor':None}
        self.save()
        self.assertEqual([event['label'] for event in self.compact()['clips'][0]['events']],
                         ['board','robot','bucket','table'])
        self.raw.update(video_bytes=self.movie.stat().st_size,output_sha256=P.sha(self.movie))
        self.raw['frames'][0]['source_tick']=999
        self.save()
        with self.assertRaisesRegex(ValueError,'frame-map checksum'):
            self.compact()

    def test_stale_audit_or_different_report_binding_is_rejected(self):
        original=self.report.read_bytes()
        self.report.write_bytes(original+b' ')
        with self.assertRaisesRegex(ValueError,'audited'):
            self.compact()
        self.report.write_bytes(original)
        self.raw['source']['report_sha256']='b'*64
        self.save()
        with self.assertRaisesRegex(ValueError,'source differs'):
            self.compact()

    def test_new_physics_or_nonheld_arm_claims_are_rejected(self):
        original=copy.deepcopy(self.raw)
        for key,value in [('native_steps_added',1),('held_arm_poses',False),('sampled_ball_centers_only',False)]:
            with self.subTest(key=key):
                self.raw=copy.deepcopy(original)
                self.raw[key]=value
                self.save()
                with self.assertRaisesRegex(ValueError,'held-endpoint'):
                    self.compact()

    def test_malformed_timeline_and_nonfinite_settings_are_rejected(self):
        original=copy.deepcopy(self.raw)
        for key,value in [('start_s',-.1),('end_s',2),('values',[0,0,float('nan'),28,.2]),('capture',None)]:
            with self.subTest(key=key):
                self.raw=copy.deepcopy(original)
                self.raw['clips'][0][key]=value
                self.save()
                with self.assertRaises(ValueError):
                    self.compact()

    def test_embed_is_additive_escaped_and_has_user_playback_controls(self):
        self.raw['clips'][0]['label']='Case </script><script>bad()</script>'
        self.save()
        snapshot=self.compact()
        original='<html><body>before<!--TABLE_HEIGHT_TILT_START-->existing study'+S.END+'after</body></html>'
        result=P.embed_html(original,snapshot,self.movie.read_bytes())
        self.assertIn('controls playsinline preload="metadata"',result)
        self.assertNotIn('autoplay',result)
        self.assertIn('data:video/mp4;base64,',result)
        self.assertIn(P.TITLE,result)
        self.assertNotIn('<script>bad()',result)
        self.assertIn('existing study'+S.END,result)
        self.assertEqual(result.split(P.START)[0],original.split(S.END)[0]+S.END+'\n')
        self.assertEqual(result.split(P.END)[1],'after</body></html>')
        self.assertEqual(P.embed_html(result,snapshot,self.movie.read_bytes()),result)

    def test_html_compare_and_swap_preserves_concurrent_writer(self):
        target=self.root/'index.html'
        original='<html><body>'+S.END+'</body></html>'
        target.write_text(original)
        with self.assertRaisesRegex(ValueError,'changed'):
            P.install(target,self.compact(),self.movie,'0'*64)
        self.assertEqual(target.read_text(),original)
        P.install(target,self.compact(),self.movie,P.sha(target))
        self.assertEqual(target.read_text().count(P.START),1)

    def test_display_addition_keeps_every_existing_key(self):
        display=self.root/'display.json'
        before=dict(table_height_tilt_study=dict(status='completed'),other=['evidence'])
        write(display,before)
        snapshot=self.compact()
        P.publish_display(display,snapshot)
        after=json.loads(display.read_text())
        self.assertEqual(after.pop(P.KEY),snapshot)
        self.assertEqual(after,before)

    def test_missing_or_ambiguous_markers_do_not_change_document(self):
        snapshot=self.compact()
        for original in ('<html></html>',P.START+S.END,P.END+S.END+P.START):
            with self.subTest(original=original),self.assertRaises(ValueError):
                P.embed_html(original,snapshot,self.movie.read_bytes())


if __name__=='__main__':
    unittest.main()
