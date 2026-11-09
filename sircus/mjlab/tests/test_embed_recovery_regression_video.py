import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from sircus.mjlab import embed_recovery_regression_video as V
from sircus.mjlab import prepare_recovery_regression as P


class SavedMovie(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name)
        self.video=self.root/'replay.mp4';self.video.write_bytes(b'unit-fixture-not-real-video');self.manifest=self.root/'replay.json';self.diagnosis=self.root/'diagnosis.json'
        def row(i,capture=False):return dict(record_id=f'r{i}',record_sha256=str(i)*64,capture=capture,values=[.5,0,.4,25,.2],bucket_xy_m=[.7,0],native_release=True,forbidden_contact_steps=0,trace=dict(native_cache_sha256='c'*64))
        self.source=dict(source=dict(report_sha256='a'*64,audit_sha256='b'*64,contract_fingerprint='c'*64),losses=[dict(parent_bc=row(1,True),recovery_bc=row(2)),dict(parent_bc=row(3,True),recovery_bc=row(4))],flight_misses=[row(5)])
        self.source['flight_misses'][0]['forbidden_contact_steps']=1
        clips=[copy.deepcopy(r) for loss in self.source['losses'] for r in (loss['parent_bc'],loss['recovery_bc'])]+[copy.deepcopy(self.source['flight_misses'][0])]
        for c in clips:c['native_cache_sha256']=c.pop('trace')['native_cache_sha256']
        self.raw=dict(version=V.VERSION,status='completed',source=self.source['source'],native_steps_added=0,held_arm_poses=True,sampled_ball_centers_only=True,interpolation=False,historical_labels_unchanged=True,native_dt_s=1/15360,backend=dict(software_only=True,cuda_visible_devices=''),output_sha256=V.sha(self.video),video_bytes=self.video.stat().st_size,clips=clips,duration_s=75)
        self.save();self.mock=patch.object(V,'checked_diagnosis',return_value=self.source);self.mock.start();self.addCleanup(self.mock.stop)

    def save(self):self.manifest.write_text(json.dumps(self.raw))

    def test_exact_native_sources_and_cpu_only_render_required(self):
        self.assertEqual(V.verified_video(self.video,self.manifest,self.diagnosis)['duration_s'],75)
        for edit in (lambda r:r.update(native_steps_added=1),lambda r:r.update(interpolation=True),lambda r:r['backend'].update(cuda_visible_devices='0'),lambda r:r['clips'][0].update(record_sha256='x'),lambda r:r.update(output_sha256='x')):
            original=copy.deepcopy(self.raw);edit(self.raw);self.save()
            with self.assertRaises(ValueError):V.verified_video(self.video,self.manifest,self.diagnosis)
            self.raw=original

    def test_additive_inline_media_survives_later_snapshot_refresh(self):
        seed=dict(version=P.VERSION);original=P.embed_html('<body><video id="historical"></video></body>',seed)
        result=V.embed_video(original,b'media');self.assertIn('id="historical"',result);self.assertIn('bWVkaWE=',result)
        self.assertEqual(V.embed_video(result,b'media'),result)
        refreshed=P.embed_html(result,{**seed,'later':True})
        self.assertEqual(result.split(P.VIDEO_START)[1].split(P.VIDEO_END)[0],refreshed.split(P.VIDEO_START)[1].split(P.VIDEO_END)[0])
        self.assertEqual(refreshed.count(P.VIDEO_START),1)
        self.assertNotIn('bWVkaWE=',P.embed_css('body{}',seed))

    def test_wrong_surface_and_outside_panel_markers_are_rejected(self):
        target=self.root/'index.html';original=P.embed_html('<body></body>',dict(version=P.VERSION));target.write_text(original)
        with self.assertRaisesRegex(ValueError,'HTML changed'):V.publish(target,self.video,self.manifest,self.diagnosis,'0'*64)
        self.assertEqual(target.read_text(),original)
        with self.assertRaisesRegex(ValueError,'within'):V.embed_video(P.VIDEO_START+P.VIDEO_END+original,b'media')
        with self.assertRaisesRegex(ValueError,'own panel'):P.embed_html(P.VIDEO_START+P.VIDEO_END+original,dict(version=P.VERSION))
        receipt=V.publish(target,self.video,self.manifest,self.diagnosis,V.sha(target));self.assertEqual(receipt['services_restarted'],0);self.assertTrue(receipt['css_unchanged'])

if __name__=='__main__':unittest.main()
