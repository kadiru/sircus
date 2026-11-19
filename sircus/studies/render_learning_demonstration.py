"""Render saved learning evidence, with separately labeled FK positioning.

Two stages permit independent numerical and media virtual environments:
  python -m sircus.studies.render_learning_demonstration prepare --bundle showcase.json --output geometry.json
  python -m sircus.studies.render_learning_demonstration render --bundle showcase.json --geometry geometry.json --output film.mp4

Prepare needs the existing numpy/scipy environment; render needs Pillow and
imageio-ffmpeg. Neither stage trains, optimizes, starts a server or uses hardware.
"""
from __future__ import annotations
import argparse
from array import array
import bisect
import copy
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import wave

ROOT=Path(__file__).resolve().parents[2]
W,H,FPS=1280,720,20
INTRO_S,CHART_S,OUTRO_S,HOLD_S=6.,8.,5.,1.5

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def selected_clips(bundle):
    if bundle.get('data_origin')!='synthetic' or bundle.get('physically_verified') is not False:
        raise ValueError('Only explicitly synthetic unverified evidence may be rendered')
    rows=sorted(bundle['learning']['same_layout_clips'],key=lambda r:r['train_scenes'])
    if [r['train_scenes'] for r in rows]!=[8,24,64,128] or len({r['scene_id'] for r in rows})!=1:
        raise ValueError('Four prespecified paired layouts are required, without selective omission')
    rows=rows+[bundle['learning']['success_example']]
    if len({r['id'] for r in rows})!=5:raise ValueError('Clip identities must be distinct')
    for r in rows:
        if r['role']!='frozen_policy' or r.get('native_paths_checked') is not False or not r['result']['frames']:
            raise ValueError('Frozen saved trajectories only; no search or physical-path claims')
    return rows

def smooth(f):return max(0.,min(1.,f))**2*(3-2*max(0.,min(1.,f)))
def joint_at(segment,t,prefix):
    f=smooth((t-segment['start_s'])/(segment['end_s']-segment['start_s']))
    return [a+(b-a)*f for a,b in zip(segment[prefix+'_from_joints_deg'],segment[prefix+'_to_joints_deg'])]

def prepare(bundle,source_sha):
    sys.path.insert(0,str(ROOT/'sircus/two-arm'))
    import numpy as np
    from two_arm_kinematics import forward_world,board_mount_rotation
    output={'schema':'sircus-learning-video-geometry-v1','bundle_sha256':source_sha,
        'provenance':'Illustrative joint interpolation through existing FK; no learned joint policy or collision-validated path',
        'native_paths_checked':False,'physically_verified':False,'clips':{}}
    for row in selected_clips(bundle):
        scene=row['scene'];sequence=row['choreography'];segments={s['stage']:s for s in sequence['segments']}
        stop=sequence['release_time_s'];times=[i/FPS for i in range(math.ceil(stop*FPS))]+[stop]
        frames=[]
        for t in times:
            a=forward_world(scene,10,joint_at(segments['move_dropper'],t,'arm10'))
            b=forward_world(scene,14,joint_at(segments['orient_board'],t,'arm14'))
            rotation=np.asarray(b['rotation_matrix'])@board_mount_rotation(scene)
            local=np.asarray(scene['board']['support_offset_mm'])+[0,0,-scene['board']['size_mm'][2]/2]
            center=np.asarray(b['tcp_xyz_mm'])-rotation@local
            frames.append({'t':t,'arm10_boxes':a['boxes'],'arm14_boxes':b['boxes'],
                'board_center_mm':center.tolist(),'board_rotation':rotation.tolist(),'ball_xyz_mm':a['tcp_xyz_mm']})
        output['clips'][row['id']]=frames
    return output

def frame_at(frames,t):
    if not frames:raise ValueError('Missing saved frames')
    return frames[max(0,bisect.bisect_right([f['t'] for f in frames],t)-1)]

def validate_geometry(bundle,geometry,source_sha):
    if geometry.get('bundle_sha256')!=source_sha:raise ValueError('Geometry does not match this evidence bundle')
    for row in selected_clips(bundle):
        frames=geometry['clips'][row['id']]
        if not frames or frames[0]['t']!=0 or abs(frames[-1]['t']-row['choreography']['release_time_s'])>1e-8:
            raise ValueError('Incomplete illustration timeline')

def soundtrack_events(rows,chapters):
    events=[]
    for row,chapter in zip(rows,[c for c in chapters if c['kind'] in ('paired_checkpoint','separate_success')]):
        start=chapter['start_s'];release=row['choreography']['release_time_s']
        for i in range(3):events.append({'t':start+1.35+i,'asset':'drum_roll.wav','gain':.30,'source':'illustrated carry and two-second settle','trial_id':row['trial_id']})
        previous=-math.inf
        for index,contact in enumerate(row['result'].get('contacts',[])):
            if contact['t']-previous<.08:continue
            previous=contact['t']
            events.append({'t':start+release+contact['t'],'asset':'cartoon_boing.wav' if contact['object']=='board' else 'cartoon_plop.wav','gain':.38,
                'source':'saved simulated contact; soundtrack cue spacing at least80ms','contact_index':index,'contact_object':contact['object'],'trial_id':row['trial_id']})
        if row['result']['outcome'] in ('captured','no_entry'):
            events.append({'t':start+release+row['result']['frames'][-1]['t'],'asset':'capture.wav' if row['result']['success'] else 'miss.wav','gain':.42,
                'source':'known saved simulation outcome, not physical sensing','trial_id':row['trial_id']})
    return sorted(events,key=lambda e:e['t'])

def mux_soundtrack(output,ffmpeg,rows,chapters,duration):
    events=soundtrack_events(rows,chapters);rate=48000;samples=array('f',[0.])*math.ceil(duration*rate);assets={}
    for event in events:
        name=event['asset'];path=ROOT/'sircus/audio/booth/assets'/name
        if name not in assets:
            with wave.open(str(path),'rb') as wav:
                if (wav.getnchannels(),wav.getsampwidth(),wav.getframerate())!=(1,2,rate):raise ValueError('Expected bundled mono48kHz PCM16 effect')
                assets[name]=array('h',wav.readframes(wav.getnframes()))
        offset=round(event['t']*rate)
        for i,value in enumerate(assets[name]):
            if offset+i>=len(samples):break
            samples[offset+i]+=value*event['gain']
    pcm=array('h',(round(max(-31000,min(31000,value))) for value in samples))
    with tempfile.TemporaryDirectory(prefix='sircus-video-audio-') as directory:
        sound=Path(directory)/'synthetic-soundtrack.wav';movie=Path(directory)/'muxed.mp4'
        with wave.open(str(sound),'wb') as wav:wav.setnchannels(1);wav.setsampwidth(2);wav.setframerate(rate);wav.writeframes(pcm.tobytes())
        subprocess.run([ffmpeg,'-y','-loglevel','error','-i',str(output),'-i',str(sound),'-c:v','copy','-c:a','aac','-b:a','128k','-shortest','-movflags','+faststart',str(movie)],check=True)
        output.write_bytes(movie.read_bytes())
    return {'kind':'synthetic soundtrack from original bundled effects; not microphone, BLE or physical contact data','events':events,
        'assets':{name:sha(ROOT/'sircus/audio/booth/assets'/name) for name in sorted(assets)},'peak_pcm16':max(abs(x) for x in pcm)}

def render(bundle,geometry,source_sha,output):
    from PIL import Image,ImageDraw
    import imageio_ffmpeg
    from sircus.studies import render_video as rv
    validate_geometry(bundle,geometry,source_sha)
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    rows=selected_clips(bundle);aggregate=bundle['learning']['learning_curve_aggregate']
    ink,cream,teal,coral,soft=rv.INK,rv.CREAM,rv.TEAL,rv.CORAL,rv.SOFT
    def text(d,xy,value,size=18,fill=ink):rv.text(d,xy,value,size,fill)
    def wrap(d,xy,value,width,size=17,fill=soft,line=25):return rv.wrapped(d,xy,value,width,size,fill,line)
    def base(title,kicker='SIRCUS  /  SIMPLE CIRCUS'):
        im=Image.new('RGB',(W,H),cream);d=ImageDraw.Draw(im,'RGBA')
        text(d,(36,22),kicker,17,teal);text(d,(36,54),title,34)
        text(d,(1008,26),'SIMULATION ONLY',15,coral)
        text(d,(36,680),'Synthetic motion, contacts and soundtrack  •  provisional dynamics  •  no physical/native-path validation',14,soft)
        return im,d
    def card(kind):
        if kind=='intro':
            im,d=base('Teach the bounce. Watch the plan change.')
            text(d,(52,161),'BOARD ANGLE',17,teal);text(d,(52,193),'Arm 14 orients the board.',31)
            text(d,(52,270),'RELEASE HEIGHT',17,teal);text(d,(52,302),'Arm 10 raises the ball.',31)
            text(d,(52,382),'FROZEN PREDICTION',17,teal);text(d,(52,414),'One scene. One learned action.',31)
            wrap(d,(52,495),'The model learns scene-to-action targets: roll, pitch, board-local U/V and release gap. It does not learn a joint-motion policy.',790,22,soft,32)
            d.rounded_rectangle((944,164,1224,584),20,fill='#e4eee5')
            for i,n in enumerate([8,24,64,128]):text(d,(992,195+i*91),f'N = {n}',38)
            text(d,(974,553),'training layouts',17,soft)
            text(d,(52,618),'Next: the same held-out layout at four training checkpoints.',23,coral)
        elif kind=='chart':
            im,d=base('Learning across the held-out population')
            text(d,(42,111),'Mean over 3 training seeds  ·  48 shared test layouts  ·  frozen inference without search',19,soft)
            for i,row in enumerate(aggregate):
                x=80+i*235;y=520-row['mean_success_fraction']*370
                d.rounded_rectangle((x,y,x+150,520),9,fill=teal)
                text(d,(x-3,520-row['layout_ci95'][1]*370-48),f"{row['mean_success_fraction']:.1%}",36)
                text(d,(x+28,541),f"N = {row['train_scenes']}",22)
                low,high=row['layout_ci95'];cx=x+75
                d.line([(cx,520-low*370),(cx,520-high*370)],fill='#edba68',width=3)
                for val in [low,high]:d.line([(cx-10,520-val*370),(cx+10,520-val*370)],fill='#edba68',width=3)
            wrap(d,(1035,210),'Default N128 checkpoint',185,20,teal)
            text(d,(1035,284),'33 / 48',32);text(d,(1035,330),'captured',18,soft)
            text(d,(1035,390),'Fixed: 25 / 48',19,soft)
            text(d,(1035,424),'Search: 37 / 48',18,soft)
            wrap(d,(1035,463),'Search is a separate per-scene comparator.',185,16,soft,24)
            text(d,(42,604),'Gold intervals: 95% layout uncertainty. Seed variability is reported separately.',18,soft)
            text(d,(42,638),'More training improves the average here; individual layouts can still fail.',20,coral)
        else:
            im,d=base('A useful policy. A clear boundary.')
            text(d,(54,170),'33 / 48',74,teal);text(d,(54,262),'captures with the predeclared N128 model',25)
            text(d,(54,331),'54.7–80.1%  ·  95% interval',22,soft)
            wrap(d,(54,397),'Capture means a board bounce, whole-ball entry and one continuous second inside the bucket.',650,26,ink,38)
            wrap(d,(825,188),'BEFORE REAL ROBOTS',340,18,coral)
            wrap(d,(825,241),'Measure the setup. Calibrate the camera and mount. Validate collision-safe paths. Test the real bounce.',340,27,ink,42)
            text(d,(54,602),'Positioning was illustrated. Ball trajectories and outcomes came from saved simulation.',22,soft)
        return im

    def scene_frame(row,t,projection):
        release_t=row['choreography']['release_time_s'];pre=t<release_t
        illustrated=frame_at(geometry['clips'][row['id']],min(t,release_t))
        recorded=frame_at(row['result']['frames'],max(0,t-release_t))
        frame=illustrated if pre else recorded
        display=copy.copy(row);display['arm14_boxes']=illustrated['arm14_boxes'];display['arm10_release_boxes']=illustrated['arm10_boxes']
        title=f"N = {row['train_scenes']}  /  same held-out layout" if row.get('paired_layout') else 'A different held-out layout  /  N = 128'
        im,d=base(title)
        phase='ILLUSTRATED ROBOT POSITIONING' if pre else 'RECORDED NUMERICAL BALL TRAJECTORY'
        text(d,(38,101),phase,15,coral if pre else teal)
        d.rounded_rectangle((26,137,936,650),16,fill='#103237')
        polys,rings=rv.scene_mesh(display,frame)
        polys.sort(key=lambda item:sum(projection.xyz(p)[2] for p in item[0])/len(item[0]),reverse=True)
        for points,color in polys:d.polygon([projection.xyz(p)[:2] for p in points],fill=color,outline=(25,67,70,100))
        for ring in rings[1:3]:d.line([projection.xyz(p)[:2] for p in ring+[ring[0]]],fill=(255,217,141,230),width=2)
        center=frame['board_center_mm'];rotation=frame['board_rotation'];normal=[rotation[i][2] for i in range(3)]
        tip=[center[i]+normal[i]*130 for i in range(3)]
        d.line([projection.xyz(center)[:2],projection.xyz(tip)[:2]],fill='#a4ead5',width=3)
        point=projection.xyz(tip);text(d,(point[0]+6,point[1]-12),'normal',12,'#b2e5d6')
        ball=projection.xyz(frame['ball_xyz_mm']);radius=max(5,projection.scale*row['scene']['ball']['diameter_mm']/2/ball[2])
        if not pre:
            history=[projection.xyz(f['ball_xyz_mm'])[:2] for f in row['result']['frames'] if f['t']<=recorded['t']]
            if len(history)>1:d.line(history,fill='#efad79',width=3)
        d.ellipse((ball[0]-radius,ball[1]-radius,ball[0]+radius,ball[1]+radius),fill='#ec8865',outline='#ffd1b5',width=2)
        if pre and t>=2.35:
            a=row['release_coordinates']['nominal_contact_ball_center_mm'];b=row['release_coordinates']['commanded_release_ball_center_mm']
            d.line([projection.xyz(a)[:2],projection.xyz(b)[:2]],fill='#f2c365',width=3)
        label='Board arm adjusts' if t<1.35 else 'Dropper raises the ball' if t<2.35 else 'Two-second hold' if pre else f"Physics time  {recorded['t']:.2f} s"
        text(d,(45,618),label,18,'#c9e5d9')
        a=row['action'];normal=row['board_pose']['normal'];heading=math.degrees(math.atan2(normal[1],normal[0]))
        text(d,(970,145),'PREDICTED TARGETS',15,teal)
        text(d,(970,180),f"{a['gap_mm']:.0f} mm",41);text(d,(970,233),'vertical release gap',16,soft)
        text(d,(970,280),f"{row['board_tilt_deg']:.1f}°",34);text(d,(970,322),'board tilt',16,soft)
        text(d,(970,363),f"Heading {heading:.1f}°",20)
        text(d,(970,398),f"Roll {a['roll_deg']:.1f}°",18,soft);text(d,(970,427),f"Pitch {a['pitch_deg']:.1f}°",18,soft)
        text(d,(970,464),f"U {a['u_mm']:.0f} / V {a['v_mm']:.0f} mm",18,soft)
        finished=t>=release_t+row['result']['frames'][-1]['t']
        if finished:
            text(d,(970,522),'CAPTURED' if row['result']['success'] else 'MISSED',30,teal if row['result']['success'] else coral)
            wrap(d,(970,566),'Separate success example' if not row.get('paired_layout') else 'All four checkpoints miss this same layout.',270,17,soft,24)
        else:wrap(d,(970,524),'Frozen checkpoint; no search at inference.',270,18,soft,25)
        text(d,(38,656),'Joint motion is an illustration, not a learned or collision-validated path. Gap is not flange height.',13,soft)
        return im

    # Keep one camera across the four paired clips. Include every swept FK box
    # plus all true ball positions so motion/failure exits are not cropped away.
    def projection_for(candidates):
        proxy=copy.deepcopy(candidates[0]);proxy['arm14_boxes']=[];proxy['arm10_release_boxes']=[];proxy['result']['frames']=[]
        for r in candidates:
            for f in geometry['clips'][r['id']][::5]:
                proxy['arm14_boxes']+=f['arm14_boxes'];proxy['arm10_release_boxes']+=f['arm10_boxes']
            proxy['result']['frames']+=r['result']['frames']
        p=rv.Projection(proxy)
        old=p.xyz
        p.xyz=lambda point:(lambda q:(q[0],q[1]+35,q[2]))(old(point))
        return p
    pair_projection=projection_for(rows[:4]);success_projection=projection_for(rows[4:])
    cover=card('intro');poster=output.with_name(output.stem+'-poster.jpg');cover.save(poster,quality=94)
    ffmpeg=imageio_ffmpeg.get_ffmpeg_exe();command=[ffmpeg,'-y','-f','rawvideo','-pixel_format','rgb24','-video_size',f'{W}x{H}','-framerate',str(FPS),'-i','pipe:0','-an','-c:v','libx264','-preset','fast','-crf','21','-pix_fmt','yuv420p','-movflags','+faststart',str(output)]
    chapters=[];count=0;qa=ROOT/'.runtime/learning-video-qa';qa.mkdir(parents=True,exist_ok=True)
    with subprocess.Popen(command,stdin=subprocess.PIPE,stderr=subprocess.PIPE) as process:
        def send(im):
            nonlocal count
            process.stdin.write(im.tobytes());count+=1
        def still(kind,duration):
            chapters.append({'start_s':count/FPS,'kind':kind});im=card(kind);im.save(qa/(kind+'.jpg'),quality=92)
            for _ in range(round(duration*FPS)):send(im)
        still('intro',INTRO_S);still('chart',CHART_S)
        for index,row in enumerate(rows):
            chapters.append({'start_s':count/FPS,'kind':'paired_checkpoint' if index<4 else 'separate_success','id':row['id'],'checkpoint_id':row['checkpoint_id'],'train_scenes':row['train_scenes'],'outcome':row['result']['outcome']})
            duration=row['choreography']['release_time_s']+row['result']['frames'][-1]['t']+HOLD_S
            print('Rendering',row['id'],row['result']['outcome'],flush=True)
            for i in range(math.ceil(duration*FPS)):
                frame=scene_frame(row,i/FPS,pair_projection if index<4 else success_projection);send(frame)
                if i in [0,20,45,87,120,math.ceil(duration*FPS)-1]:frame.save(qa/f'clip-{index+1}-{i:03d}.jpg',quality=92)
        still('outro',OUTRO_S)
        process.stdin.close();stderr=process.stderr.read();code=process.wait()
        if code:raise RuntimeError('ffmpeg failed: '+stderr.decode(errors='replace')[-2000:])
    soundtrack=mux_soundtrack(output,ffmpeg,rows,chapters,count/FPS)
    manifest={'schema':'sircus-learning-demonstration-video-v1','bundle_sha256':source_sha,'renderer_sha256':sha(__file__),
        'data_origin':'synthetic','physically_verified':False,'native_paths_checked':False,
        'frames':count,'fps':FPS,'width':W,'height':H,'duration_seconds':count/FPS,'codec':'H.264','audio':soundtrack,
        'chapters':chapters,'selection':'All N8/24/64/128 clips on the predeclared first test layout; separately labeled saved success example',
        'positioning':'Smooth joint interpolation rendered through the existing FK model; illustration only',
        'trajectory':'Original saved ball frames selected by timestamp; no new optimizer, model fit or ball interpolation',
        'policy':'Scene-conditioned roll/pitch/U/V/gap targets; not a learned joint policy',
        'files':{output.name:sha(output),poster.name:sha(poster)}}
    target=output.with_name(output.stem+'-manifest.json');target.write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({'video':str(output),'duration_seconds':count/FPS,'bytes':output.stat().st_size}),flush=True)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=['prepare','render']);p.add_argument('--bundle',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--geometry',type=Path)
    a=p.parse_args();bundle=json.loads(a.bundle.read_text());source_sha=sha(a.bundle)
    if a.stage=='prepare':
        a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(prepare(bundle,source_sha),separators=(',',':'))+'\n')
    else:
        if a.geometry is None:p.error('--geometry is required for render')
        render(bundle,json.loads(a.geometry.read_text()),source_sha,a.output)

if __name__=='__main__':main()
