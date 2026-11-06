"""Synthetic metadata plus mocked encoder/decode; no ffmpeg, GPU or services."""
import copy
import gzip
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import shot_benchmark_video as video


class VideoTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.run=Path(self.temp.name)/'run';self.run.mkdir();(self.run/'frames').mkdir();(self.run/'trials').mkdir()
        self.camera=dict(eye_m=[1.,-2.,2.5],target_m=[0.,0.,.3],resolution=[1920,1080],up_axis='Z')
        self.write(self.run/'selected-camera.json',self.camera)
        self.source=dict(commit='a'*40,camera_sha256=video.sha(self.run/'selected-camera.json'))
        self.experiment=dict(experiment_id='synthetic-encoder-test-only',source=self.source,split_manifest_sha256='b'*64,methods=['sweep','ppo'],seeds=[20261121,20261122,20261123])
        self.exp=Path(self.temp.name)/'experiment.json';self.write(self.exp,self.experiment)
        self.owner=dict(experiment_id=self.experiment['experiment_id'],source=self.source,split_manifest_sha256='b'*64,method='ppo',seed=20261121)
        self.write(self.run/'progress.json',self.owner);self.progress_bytes=(self.run/'progress.json').read_bytes()
        self.write(self.run/'experiment-binding.json',dict(experiment_sha256=video.sha(self.exp),source=self.source))
        self.font=Path(self.temp.name)/'font.ttf';self.font.write_bytes(b'synthetic-font-test-only')
        self.clip=self.make_clip(0)
        self.write(self.run/'clips.json',dict(clips=[self.clip]))
    def write(self,path,data):path.write_text(json.dumps(data))
    def make_clip(self,start):
        # Sufficient JPEG SOF for header validation; not a claimed decodable image.
        jpeg=b'\xff\xd8\xff\xc0\x00\x11\x08\x04\x38\x07\x80'+b'\x03'+b'\x01\x11\x00'*3+b'\xff\xd9'
        frames=[]
        for i,phase in enumerate(['setup','release']):
            frame=start+i;name=self.run/f'frames/{frame:05d}.jpg';name.write_bytes(jpeg)
            frames.append(dict(frame=frame,phase=phase,physics_step=frame*8,eye_m=self.camera['eye_m'],target_m=self.camera['target_m'],jpeg_sha256=video.sha(name),resolution=[1920,1080],jpeg_quality=82,all_joints_rad={a:[[0.]*6 for _ in range(35)] for a in ('Arm10','Arm14')},all_ball_world_m=[[0.,0.,0.] for _ in range(35)]))
        trial=dict(mode='train',batch=start,release_step=(start+1)*8,source=self.source,split_manifest_sha256='b'*64,rows=[],actions=[])
        path=self.run/f'trials/train-{start}.json.gz'
        with gzip.open(path,'wt') as stream:json.dump(trial,stream)
        return dict(mode='train',batch=start,training_attempts=start*35,method='ppo',seed=20261121,trial_file=f'trials/train-{start}.json.gz',trial_sha256=video.sha(path),start_frame=start,end_frame=start+2,frames=frames,fps=30,release_step=trial['release_step'],camera_sha256=self.source['camera_sha256'],rows=[],actions=[])
    def encode(self):return video.encode_run(self.run,self.exp,ffmpeg='mock-ffmpeg',ffprobe='mock-ffprobe')
    def mocks(self):
        def encode(command,**kwargs):
            self.assertIn('libx264',command);self.assertNotIn('cuda',' '.join(command));self.assertEqual(command[command.index('-frames:v')+1],'2');Path(command[-1]).write_bytes(b'synthetic-video-not-real')
        def inspect(path,*args,**kwargs):return dict(sha256=video.sha(path),frames=2,duration_s=2/30,fps=30,decode_verified=True)
        return patch.object(video,'font_path',return_value=self.font),patch.object(video.subprocess,'run',side_effect=encode),patch.object(video,'inspect_movie',side_effect=inspect)
    def test_header_camera_hash_and_trial_are_verified_before_encoding(self):
        self.assertEqual(video.jpeg_dimensions((self.run/'frames/00000.jpg').read_bytes()),[1920,1080])
        good=video.validate_clip(self.run,self.clip,self.experiment,self.camera,self.owner);self.assertEqual(good['release_frames'],1)
        for mutate,pattern in [(lambda c:c['frames'][0].update(eye_m=[10,10,10]),'Camera'),(lambda c:c.update(trial_sha256='0'*64),'Trial bytes'),(lambda c:c['frames'][1].update(phase='setup'),'complete chapter'),(lambda c:c.update(end_frame=4),'boundaries')]:
            c=copy.deepcopy(self.clip);mutate(c)
            with self.assertRaisesRegex(ValueError,pattern):video.validate_clip(self.run,c,self.experiment,self.camera,self.owner)
        (self.run/'frames/00000.jpg').write_bytes(b'corrupted')
        with self.assertRaisesRegex(ValueError,'JPEG bytes'):video.validate_clip(self.run,self.clip,self.experiment,self.camera,self.owner)
    def test_successful_single_pass_is_incremental_all_frames_and_no_progress_mutation(self):
        a,b,c=self.mocks()
        with a,b as encoder,c as inspector:
            first=self.encode();second=self.encode();self.assertEqual(encoder.call_count,1);self.assertEqual(inspector.call_count,1)
        self.assertEqual(first,second);self.assertEqual(first['videos'][0]['frames'],2);self.assertTrue(first['videos'][0]['decode_verified'])
        self.assertEqual(first['videos'][0]['selection'],'prescheduled');self.assertEqual((self.run/'progress.json').read_bytes(),self.progress_bytes)
        self.assertTrue((self.run/'frames/00000.jpg').exists());self.assertEqual(video.read(self.run/'video-index.json')['videos'],first['videos'])
        self.assertFalse((self.run/'media/.encoder.lock').exists())
    def test_later_failure_preserves_successful_index_and_never_publishes_bad_chapter(self):
        nextclip=self.make_clip(2);nextclip['frames'][0]['jpeg_sha256']='0'*64;self.write(self.run/'clips.json',dict(clips=[self.clip,nextclip]))
        a,b,c=self.mocks()
        with a,b,c:
            with self.assertRaisesRegex(ValueError,'JPEG bytes'):self.encode()
        self.assertEqual(len(video.read(self.run/'video-index.json')['videos']),1);self.assertEqual((self.run/'progress.json').read_bytes(),self.progress_bytes)
    def test_existing_media_or_caption_corruption_and_manifest_regression_rejected(self):
        a,b,c=self.mocks()
        with a,b,c:self.encode()
        original=video.read(self.run/'video-index.json')['videos'][0];movie=self.run/original['file'];old=movie.read_bytes();movie.write_bytes(b'changed')
        with patch.object(video,'font_path',return_value=self.font):
            with self.assertRaisesRegex(ValueError,'movie changed'):self.encode()
        movie.write_bytes(old);self.write(self.run/'clips.json',dict(clips=[]))
        with patch.object(video,'font_path',return_value=self.font):
            with self.assertRaisesRegex(ValueError,'regressed'):self.encode()
        self.assertEqual(len(video.read(self.run/'video-index.json')['videos']),1)
    def test_owned_lock_refuses_overlap_and_gzip_trial_corruption(self):
        media=self.run/'media';media.mkdir();(media/'.encoder.lock').write_text('other-owner')
        with patch.object(video,'font_path',return_value=self.font):
            with self.assertRaisesRegex(ValueError,'Another encoder'):self.encode()
        self.assertEqual((media/'.encoder.lock').read_text(),'other-owner')
        path=self.run/self.clip['trial_file'];trial=video.trial_read(path);trial['source']={};
        with gzip.open(path,'wt') as stream:json.dump(trial,stream)
        self.clip['trial_sha256']=video.sha(path)
        with self.assertRaisesRegex(ValueError,'source/split'):video.validate_clip(self.run,self.clip,self.experiment,self.camera,self.owner)
    def test_actual_decode_check_requires_exact_frames_format_and_normal_time(self):
        p=self.run/'fake.mp4';p.write_bytes(b'mock-only')
        probe=dict(streams=[dict(codec_type='video',codec_name='h264',width=1920,height=1080,avg_frame_rate='30/1',nb_read_frames='2',duration=2/30)],format={'duration':2/30})
        with patch.object(video.subprocess,'check_output',return_value=json.dumps(probe).encode()),patch.object(video.subprocess,'run') as decoder:
            result=video.inspect_movie(p,'ffmpeg','ffprobe');self.assertTrue(result['decode_verified']);self.assertIn('-xerror',decoder.call_args.args[0])
        probe['streams'][0]['avg_frame_rate']='60/1'
        with patch.object(video.subprocess,'check_output',return_value=json.dumps(probe).encode()):
            with self.assertRaisesRegex(ValueError,'format changed'):video.inspect_movie(p,'ffmpeg','ffprobe')

if __name__=='__main__':unittest.main()
