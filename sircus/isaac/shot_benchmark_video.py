"""Single-pass CPU encoding of completed, prescheduled native shot chapters.

All native frames survive at30fps. No optimizer, simulator, robot, GPU encoder,
progress mutation, outcome selection or source-frame deletion occurs here.
"""
import argparse
from contextlib import contextmanager
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import uuid

from shot_benchmark_report import digest, read, require, safe_path, sha

RECEIPT_SCHEMA = 'sircus-shot-benchmark-video-receipt-v1'


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,allow_nan=False,separators=(',',':')).encode()).hexdigest()


def jpeg_dimensions(data):
    """Read actual JPEG SOF dimensions without a per-frame subprocess."""
    require(data[:2]==b'\xff\xd8','Not a JPEG')
    i=2
    while i<len(data):
        require(data[i]==255,'Invalid JPEG marker');i+=1
        while i<len(data) and data[i]==255:i+=1
        require(i<len(data),'Truncated JPEG marker');marker=data[i];i+=1
        if marker in (0xd8,0xd9) or 0xd0<=marker<=0xd7:continue
        require(i+2<=len(data),'Truncated JPEG length')
        length=int.from_bytes(data[i:i+2],'big');require(length>=2 and i+length<=len(data),'Truncated JPEG segment')
        if marker in (0xc0,0xc1,0xc2,0xc3,0xc5,0xc6,0xc7,0xc9,0xca,0xcb,0xcd,0xce,0xcf):
            require(length>=8,'Invalid JPEG SOF')
            return [int.from_bytes(data[i+5:i+7],'big'),int.from_bytes(data[i+3:i+5],'big')]
        require(marker!=0xda,'JPEG scan before dimensions');i+=length
    raise ValueError('Missing JPEG dimensions')


def trial_read(path):
    require(path.name.endswith('.json.gz'),'Native compressed JSON trial required')
    with gzip.open(path,'rt') as stream:
        return json.load(stream)


def validate_clip(run, clip, experiment, camera, expected_owner):
    run=Path(run)
    require(clip['method']==expected_owner['method'] and clip['seed']==expected_owner['seed'],'Clip method/seed differs from native run')
    require(clip['method'] in experiment['methods'] and clip['seed'] in experiment['seeds'],'Undeclared method/seed')
    require(type(clip['training_attempts']) is int and 0<=clip['training_attempts']<=10000,'Invalid TRAIN prefix')
    require(clip['fps']==30 and clip['camera_sha256']==experiment['source']['camera_sha256'],'Camera/fps changed')
    trial_path=safe_path(run,clip['trial_file']);require(sha(trial_path)==digest(clip['trial_sha256'],'Trial SHA'),'Trial bytes changed')
    trial=trial_read(trial_path)
    require(trial['source']==experiment['source'] and trial['split_manifest_sha256']==experiment['split_manifest_sha256'],'Trial source/split changed')
    for key in ('mode','batch','release_step'):
        require(trial[key]==clip[key],'Clip differs from native trial '+key)
    for key in ('rows','actions'):
        if key in clip:require(clip[key]==trial[key],'Clip differs from native '+key)
    frames=clip['frames'];require(isinstance(frames,list) and frames,'No completed native frames')
    ids=[f['frame'] for f in frames]
    require(all(type(i) is int and i>=0 for i in ids) and ids==list(range(ids[0],ids[-1]+1)),'Native frame sequence has gaps/reordering')
    require(clip['start_frame']==ids[0] and clip['end_frame']==ids[-1]+1,'Clip boundaries changed')
    release=0;setup=0;previous=-1;maximum_camera_error=0
    for frame in frames:
        require(type(frame['physics_step']) is int and frame['physics_step']>previous,'Physics steps reordered');previous=frame['physics_step']
        require(frame['phase'] in ('setup','move','release'),'Unknown native phase')
        release+=frame['phase']=='release';setup+=frame['phase'] in ('setup','move')
        require(frame['resolution']==[1920,1080] and frame['jpeg_quality']==82,'Native image settings changed')
        require(len(frame['eye_m'])==3 and all(type(v) in (int,float) and math.isfinite(v) for v in frame['eye_m']),'Invalid camera readback')
        maximum_camera_error=max(maximum_camera_error,math.dist(frame['eye_m'],camera['eye_m']))
        require(maximum_camera_error<=.0002 and frame['target_m']==camera['target_m'],'Camera followed/moved away from fixed selected pose')
        joints=frame['all_joints_rad'];require(set(joints)=={'Arm10','Arm14'} and all(len(v)==35 and all(len(q)==6 and all(type(x) in (int,float) and math.isfinite(x) for x in q) for q in v) for v in joints.values()),'Missing six-axis native joint readbacks')
        path=safe_path(run,f"frames/{frame['frame']:05d}.jpg");data=path.read_bytes()
        require(hashlib.sha256(data).hexdigest()==digest(frame['jpeg_sha256'],'JPEG SHA'),'Native JPEG bytes changed')
        require(jpeg_dimensions(data)==[1920,1080],'Actual JPEG resolution differs')
    require(release>0 and setup>0,'A complete chapter requires visible setup and release frames')
    return dict(source_clip_sha256=canonical_sha(clip),source_trial_sha256=sha(trial_path),frames=len(frames),
        release_frames=release,setup_frames=setup,first_frame=ids[0],last_frame=ids[-1],
        maximum_camera_readback_error_m=maximum_camera_error,
        source_frame_hashes_sha256=canonical_sha([[f['frame'],f['jpeg_sha256']] for f in frames]))


def inspect_movie(path, ffmpeg, ffprobe, *, threads=4):
    raw=json.loads(subprocess.check_output([ffprobe,'-v','error','-count_frames','-show_streams','-show_format','-of','json',str(path)]))
    streams=raw['streams'];require(len(streams)==1 and streams[0]['codec_type']=='video','Exactly one video stream and no audio required')
    s=streams[0];num,den=map(int,s['avg_frame_rate'].split('/'));fps=num/den
    require(s['codec_name']=='h264' and [s['width'],s['height']]==[1920,1080] and fps==30,'Encoded native format changed')
    frames=int(s['nb_read_frames']);duration=float(s.get('duration',raw['format']['duration']))
    require(abs(duration-frames/30)<.001,'Encoded footage was stretched')
    subprocess.run([ffmpeg,'-v','error','-xerror','-threads',str(threads),'-i',str(path),'-map','0:v:0','-f','null','-'],check=True,stdout=subprocess.DEVNULL)
    return dict(sha256=sha(path),frames=frames,duration_s=duration,fps=fps,decode_verified=True)


def caption_for(clip):
    return (f"{clip['method'].upper()} | seed {clip['seed']} | {clip['mode'].upper()} | "
            f"TRAIN attempts before chapter {clip['training_attempts']}\n"
            'Native simulation; scripted joint travel; initialized ball release. No improvement claim.')


def font_path():
    choices=('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf','/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf','/System/Library/Fonts/Supplemental/Arial.ttf')
    return next((Path(p) for p in choices if Path(p).is_file()),None)


def filter_path(path):
    return str(Path(path).resolve()).replace('\\','\\\\').replace("'","\\'").replace(':','\\:')


def atomic(path, value):
    temporary=path.with_name(path.name+'.next-'+uuid.uuid4().hex)
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');temporary.replace(path)


@contextmanager
def owned_lock(media):
    path=media/'.encoder.lock';cookie=uuid.uuid4().hex
    try:
        with path.open('x') as file:file.write(cookie)
    except FileExistsError as error:
        raise ValueError('Another encoder owns this media directory, or its stale lock requires explicit inspection') from error
    try:yield
    finally:
        if path.is_file() and path.read_text()==cookie:path.unlink()


def encode_run(run, experiment_file, *, ffmpeg=None, ffprobe=None, threads=4):
    run,experiment_file=Path(run),Path(experiment_file)
    require(type(threads) is int and 1<=threads<=8,'CPU threads must be1..8')
    experiment=read(experiment_file);owner=read(run/'progress.json')
    require(owner['experiment_id']==experiment['experiment_id'] and owner['source']==experiment['source'],'Native run belongs to another source/experiment')
    require(owner['split_manifest_sha256']==experiment['split_manifest_sha256'],'Native split changed')
    binding=read(run/'experiment-binding.json')
    require(binding['experiment_sha256']==sha(experiment_file) and binding['source']==experiment['source'],'Experiment binding changed')
    camera_file=run/'selected-camera.json';require(sha(camera_file)==experiment['source']['camera_sha256'],'Selected camera bytes changed')
    camera=read(camera_file);require(camera['resolution']==[1920,1080] and camera['up_axis']=='Z','Selected camera format changed')
    if not (run/'clips.json').is_file():return dict(status='waiting_for_completed_clips',videos=[])
    manifest=read(run/'clips.json');clips=manifest if isinstance(manifest,list) else manifest['clips']
    ffmpeg=ffmpeg or shutil.which('ffmpeg');ffprobe=ffprobe or shutil.which('ffprobe')
    require(ffmpeg and ffprobe,'CPU ffmpeg and ffprobe executables required')
    font=font_path();require(font is not None,'Installed caption font required')
    media=run/'media';require(not media.is_symlink(),'Media directory cannot be a symlink');media.mkdir(exist_ok=True)
    with owned_lock(media):
        index=run/'video-index.json';prior=read(index)['videos'] if index.is_file() else []
        videos=[];previous=-1
        for i,clip in enumerate(clips):
            validated=validate_clip(run,clip,experiment,camera,owner)
            require(validated['first_frame']>previous,'Chapter frame order changed');previous=validated['last_frame']
            source_binding=dict(experiment_sha256=sha(experiment_file),camera_sha256=sha(camera_file),encoder_sha256=sha(__file__),**validated)
            stem=f"{clip['method']}-seed{clip['seed']}-{clip['mode']}-batch{clip['batch']:05d}-{validated['source_clip_sha256'][:12]}"
            require('/' not in stem and '\\' not in stem and '..' not in stem,'Unsafe chapter identity')
            movie=media/(stem+'.mp4');receipt=media/(stem+'.receipt.json');caption=media/(stem+'.caption.txt')
            require(not receipt.is_symlink() and not movie.is_symlink() and not caption.is_symlink(), 'Derived chapter files cannot be symlinks')
            if receipt.is_file():
                saved=read(receipt)
                require(saved['schema']==RECEIPT_SCHEMA and saved['source_binding']==source_binding,'Existing chapter receipt belongs to different source')
                require(movie.is_file() and not movie.is_symlink() and saved['video']['sha256']==sha(movie),'Existing encoded movie changed')
                require(caption.is_file() and sha(caption)==saved['caption_sha256'] and caption.read_text()==caption_for(clip),'Existing caption receipt changed')
                require(saved['video']['decode_verified'] is True and saved['video']['frames']==validated['frames'],'Incomplete existing decode receipt')
                record=saved['video']
            else:
                require(not movie.exists(),'Encoded movie without its receipt; inspect interrupted export before retrying')
                if caption.exists():require(caption.read_text()==caption_for(clip),'Interrupted caption differs')
                else:caption.write_text(caption_for(clip))
                temporary=media/(stem+'.part-'+uuid.uuid4().hex+'.mp4')
                vf=("drawbox=x=18:y=16:w=1760:h=90:color=0x17202bcc:t=fill,"
                    f"drawtext=fontfile='{filter_path(font)}':textfile='{filter_path(caption)}':"
                    'x=32:y=26:fontsize=24:line_spacing=10:fontcolor=0xeff5fa')
                subprocess.run([str(ffmpeg),'-v','error','-n','-framerate','30','-start_number',str(validated['first_frame']),'-i',str(run/'frames/%05d.jpg'),'-frames:v',str(validated['frames']),'-vf',vf,'-an','-c:v','libx264','-pix_fmt','yuv420p','-threads',str(threads),'-preset','fast','-crf','18','-movflags','+faststart',str(temporary)],check=True)
                details=inspect_movie(temporary,str(ffmpeg),str(ffprobe),threads=threads)
                require(details['frames']==validated['frames'],'Encoded movie lost or added native frames')
                require(not movie.exists(),'Concurrent encoded destination appeared')
                temporary.replace(movie)
                record=dict(experiment_id=experiment['experiment_id'],experiment_sha256=sha(experiment_file),source_commit=experiment['source']['commit'],file='media/'+movie.name,selection='prescheduled',camera_sha256=clip['camera_sha256'],method=clip['method'],seed=clip['seed'],training_attempts=clip['training_attempts'],label=f"{clip['mode']} batch {clip['batch']} · full native chapter",provenance='Actual prescheduled native simulation; all source frames at normal30fps, no outcome selection/speedup. Scripted joint travel and initialized release are not learned motor control.',**details)
                atomic(receipt,dict(schema=RECEIPT_SCHEMA,source_binding=source_binding,caption_sha256=sha(caption),video=record))
            if i<len(prior):require(record==prior[i],'Existing video-index chapter changed')
            videos.append(record)
            # Every successful chapter is independently publishable, even if a later chapter fails.
            atomic(index,dict(videos=videos+prior[len(videos):]))
        require(len(videos)>=len(prior),'Native clip manifest regressed; refusing to remove published chapters')
        return dict(status='completed_single_pass',videos=videos)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True);parser.add_argument('--experiment',type=Path,required=True)
    parser.add_argument('--ffmpeg');parser.add_argument('--ffprobe');parser.add_argument('--threads',type=int,default=4)
    a=parser.parse_args();value=encode_run(a.run,a.experiment,ffmpeg=a.ffmpeg,ffprobe=a.ffprobe,threads=a.threads)
    print(json.dumps(dict(status=value['status'],chapters=len(value['videos']),frames=sum(v['frames'] for v in value['videos']))))
