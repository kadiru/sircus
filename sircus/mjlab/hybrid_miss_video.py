"""CPU-only videos of audited held-endpoint native64 shots.

Frames use exact recorded ball samples and original arm joints. No dynamics,
interpolated trajectory, learned transit, or synthetic contact is introduced.
"""
from __future__ import annotations
import argparse
import importlib.metadata
import json
import math
import os
from pathlib import Path

import numpy as np
from .saved_shot_video import (SOURCE_MODULES, contained, digest, nearest_tick,
    read, require, sha, software_backend, vector, verify_assets)
from .fixed35_trial_video import validate_render_environment
from .height_tilt_video import CONTACTS, first_events

VERSION = 'sircus-hybrid-miss-recorded-video-v1'
DT = 1 / 15360


def verify_envelope(study, audit_path):
    study = Path(study).resolve()
    report, contract, manifest, audit = [read(p) for p in
        (study/'report.json', study/'contract.json', study/'evidence-manifest.json', audit_path)]
    require(report.get('status') == 'completed' and audit.get('status') == 'passed',
            'Completed independently audited evidence required')
    require(audit.get('checks_count', 0) > 0, 'Nonempty independent audit required')
    require(audit.get('report_sha256') == sha(study/'report.json') and
            audit.get('evidence_manifest_sha256') == sha(study/'evidence-manifest.json'),
            'Audit must bind report and evidence manifest bytes')
    require(report.get('evidence_manifest_sha256') == sha(study/'evidence-manifest.json') or
            report.get('evidence_sha256') == manifest, 'Report must bind its evidence manifest')
    for name in ('rows.jsonl', 'contract.json'):
        require(manifest.get(name) == sha(study/name), 'Manifest differs: '+name)
    unsigned = dict(contract); fp = unsigned.pop('fingerprint')
    require(fp == digest(unsigned), 'Contract fingerprint differs')
    require(contract['identity']['trace_schema']['native_dt_s'] == DT and
            contract['identity']['physics']['substeps'] == 64, 'Fixed64 native clock required')
    require(digest(contract['scene']) == contract['identity']['scene_sha256'], 'Scene differs')
    return report, contract, manifest, audit


def load_recording(a):
    study = Path(a.study).resolve()
    report, contract, manifest, audit = verify_envelope(study, a.audit)
    assets = verify_assets(a.assets, contract['identity']['asset_sha256'])
    root = Path(__file__).resolve().parents[2]
    sources = {name: sha(root/name) for name in SOURCE_MODULES}
    require(all(contract['identity']['source_sha256'].get(k) == v for k,v in sources.items()),
            'Frozen geometry source differs')
    rows = [json.loads(l) for l in (study/'rows.jsonl').read_text().splitlines() if l.strip()]
    require(len({r['id'] for r in rows}) == len(rows), 'Duplicate record identity')
    matches = [r for r in rows if r['id'] == a.record]
    require(len(matches) == 1, 'Exactly one requested recording required')
    row = matches[0]
    require(row.get('native_release') is True and row.get('invalid_physics') is False and
            row.get('contact_protocol', {}).get('substeps') == 64, 'Valid native64 flight required')
    relative = row['adapter_directory']+'/'+row['native_cache_file']
    cache = contained(study, relative)
    require(sha(cache) == row['native_cache_sha256'] == manifest.get(relative), 'Native archive differs')
    column = row['native_cache_row']
    require(type(column) is int and column >= 0, 'Literal native column required')
    with np.load(cache, allow_pickle=False) as z:
        require(column < len(z['record_ids']) and str(z['record_ids'][column]) == row['id'], 'Native record ID differs')
        times = np.asarray(z['time_s'], float)
        positions = np.asarray(z['position_m'][:, column], float)
        forces = np.asarray(z['forces_N'][:, column], float)
    require(times.ndim == 1 and len(times)>1 and np.isfinite(times).all() and
            np.allclose(times, np.arange(len(times))*DT, atol=1e-12, rtol=0), 'Native clock differs')
    require(positions.shape == (len(times),3) and np.isfinite(positions).all() and
            np.isfinite(forces).all(), 'Finite recorded samples required')
    require(np.array_equal(positions[0], row['native_initial_ball_center_m']), 'Release center differs')
    require(type(row['physics_steps']) is int and 0 < row['physics_steps'] < len(times), 'Scored interval missing')
    for k in ('q10','q14'): vector(row['native_initial_'+k],6,k)
    events = first_events(times,forces,row)
    return dict(row=row, contract=contract, scene=contract['scene'], time_s=times,
        positions=positions, events=events, scored_until_s=float(times[row['physics_steps']]),
        input_sha256={**{n:sha(study/n) for n in ('report.json','contract.json','rows.jsonl','evidence-manifest.json')},
                      'audit':sha(a.audit),'native_archive':sha(cache)},
        record_sha256=digest(row), assets_sha256=assets, model_source_sha256=sources)


def build_scene(bundle, assets):
    import mujoco
    from .model import build_scene as xml_scene
    physics = bundle['contract']['identity']['physics']
    xml = xml_scene(bundle['scene'], assets, solref=tuple(physics['solref']), contact_friction=physics['contact_friction'])
    model = mujoco.MjModel.from_xml_string(xml); data = mujoco.MjData(model); row=bundle['row']
    for role,key in (('Arm10','q10'),('Arm14','q14')):
        for j,value in enumerate(row['native_initial_'+key],1):
            data.qpos[model.jnt_qposadr[model.joint(f'{role}/joint{j}').id]]=value
    for side in ('left','right'):
        data.qpos[model.jnt_qposadr[model.joint(f'Arm10/finger_{side}_joint').id]]=float(np.float32(.025))
    ball_q=int(model.jnt_qposadr[model.joint('ball/free').id])
    data.qpos[ball_q:ball_q+3]=bundle['positions'][0]
    data.qpos[ball_q+3:ball_q+7]=[1,0,0,0]
    bucket=model.body('bucket').id
    data.mocap_pos[model.body_mocapid[bucket]]=row['native_initial_bucket_center_m']
    data.mocap_quat[model.body_mocapid[bucket]]=[1,0,0,0]
    mujoco.mj_forward(model,data)
    board=model.body('Arm14/impact_board').id
    require(np.allclose(data.xpos[board],row['board_center_m'],atol=2e-6,rtol=0), 'Rendered board center differs')
    require(np.allclose(data.xmat[board].reshape(3,3)[:,2],row['board_normal'],atol=2e-6,rtol=0), 'Rendered board normal differs')
    require(np.allclose(data.xpos[bucket],[*row['bucket_xy_m'],0],atol=1e-6,rtol=0), 'Rendered bucket differs')
    return model,data,ball_q,digest(xml)


def cameras(bundle):
    row=bundle['row'];board=np.asarray(row['board_center_m']);bucket=np.r_[row['bucket_xy_m'],.12]
    release=np.asarray(row['native_initial_ball_center_m'])
    direction=bucket[:2]-board[:2];side=np.asarray([direction[1],-direction[0]])
    angle=math.degrees(math.atan2(side[1],side[0]))
    target=np.mean([board,bucket,release],axis=0).tolist()
    return dict(overview=dict(lookat=[.69,-.13,.37],distance=2.25,azimuth=angle+18,elevation=-27),
                close=dict(lookat=target,distance=1.5,azimuth=angle,elevation=-19))


def frame_plan(bundle,fps=30):
    require(type(fps) is int and 1 <= fps <= 60,'Bounded integer FPS required')
    times=bundle['time_s'];end=bundle['scored_until_s'];events=bundle['events'];frames=[]
    require(0 < end <= times[-1],'Scored interval outside recording')
    def add(segment,view,speed,wanted,count=1):
        tick=nearest_tick(times,wanted)
        for _ in range(count):
            frames.append(dict(frame=len(frames),video_time_s=len(frames)/fps,segment=segment,
                view=view,playback_speed=speed,source_tick=tick,source_time_s=float(times[tick])))
    add('title','overview',0,0,round(.6*fps))
    for i in range(math.ceil(end*fps)):add('normal','overview',1,min(end,i/fps))
    frozen=False
    for i in range(math.ceil(end/.25*fps)):
        wanted=min(end,i/fps*.25)
        if not frozen and 'robot' in events and wanted>=events['robot']:
            add('robot_contact_hold','close',0,events['robot'],round(.8*fps));frozen=True
        add('slow','close',.25,wanted)
    add('outcome','close',0,end,round(1.*fps))
    return frames


def stage(item,bundle):
    t=item['source_time_s'];events=bundle['events'];row=bundle['row']
    if item['segment']=='robot_contact_hold':return 'Recorded ball / robot contact'
    if item['segment']=='outcome':
        return 'CAPTURE / one continuous second inside' if row['route_success'] else 'MISS / no accepted capture'
    if 'capture' in events and t>=events['capture']:return 'CAPTURE / retained one second'
    if 'whole_ball_entry' in events and t>=events['whole_ball_entry']:return 'Whole ball inside / retention clock running'
    occurred=[(v,k) for k,v in events.items() if k in CONTACTS and v<=t]
    return 'Latest recorded contact: '+max(occurred)[1] if occurred else 'Released / falling toward board'


def render(a):
    environment=validate_render_environment(os.environ)
    for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','LP_NUM_THREADS'):os.environ[key]='2'
    require(a.width%2==a.height%2==0 and 960<=a.width<=1920 and 540<=a.height<=1080,'Even960x540..1920x1080 output required')
    output=Path(a.output).resolve();require(output.suffix=='.mp4','MP4 output required')
    manifest=output.with_suffix('.preview.json' if a.preview_only else '.json')
    for p in (a.study,a.assets):require(not output.is_relative_to(Path(p).resolve()),'Keep videos outside original evidence/assets')
    require(not output.exists() and not manifest.exists(),'Refusing to overwrite media evidence')
    bundle=load_recording(a);model,data,ball_q,model_hash=build_scene(bundle,a.assets)
    frames=frame_plan(bundle);views=cameras(bundle)
    if a.cameras:views.update(read(a.cameras))
    for v in views.values():
        vector(v['lookat'],3,'camera target')
        require(all(math.isfinite(v[k]) for k in ('distance','azimuth','elevation')) and v['distance']>0,'Finite camera required')
    import mujoco
    from OpenGL import GL
    from PIL import Image,ImageDraw,ImageFont
    import imageio.v2 as imageio
    output.parent.mkdir(parents=True,exist_ok=True)
    font_path='/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
    fonts=[ImageFont.truetype(font_path,n) for n in (27,21,18)]
    header,footer=112,72;viewport=a.height-header-footer
    model.vis.global_.offwidth=max(a.width,model.vis.global_.offwidth)
    model.vis.global_.offheight=max(viewport,model.vis.global_.offheight)
    camera=mujoco.MjvCamera();camera.type=mujoco.mjtCamera.mjCAMERA_FREE
    row=bundle['row'];previews=[];writer=None
    with mujoco.Renderer(model,height=viewport,width=a.width) as renderer:
        backend=software_backend((GL.glGetString(GL.GL_RENDERER) or b'').decode(),
            (GL.glGetString(GL.GL_VENDOR) or b'').decode(),os.environ.get('CUDA_VISIBLE_DEVICES'))
        def draw_frame(item):
            data.qpos[ball_q:ball_q+3]=bundle['positions'][item['source_tick']]
            data.time=item['source_time_s'];mujoco.mj_forward(model,data)
            settings=views[item['view']];camera.lookat[:]=settings['lookat']
            for k in ('distance','azimuth','elevation'):setattr(camera,k,settings[k])
            renderer.update_scene(data,camera=camera)
            picture=Image.new('RGB',(a.width,a.height),'#101722')
            picture.paste(Image.fromarray(renderer.render()),(0,header));draw=ImageDraw.Draw(picture)
            xy=row['bucket_xy_m'];values=row['values']
            draw.text((20,8),a.label,font=fonts[0],fill='white')
            speed='HOLD' if item['playback_speed']==0 else str(item['playback_speed'])+'x'
            draw.text((20,44),f"{speed} | t={item['source_time_s']:.4f}s | {stage(item,bundle)}",font=fonts[1],fill='#85dec2')
            draw.text((20,77),f"Bucket ({xy[0]*1000:.1f}, {xy[1]*1000:.1f}) mm | board {values[2]*1000:.1f} mm | tilt {values[3]:.2f} deg | gap {values[4]*1000:.1f} mm",font=fonts[2],fill='#c7d2df')
            draw.text((20,a.height-64),'Saved MuJoCo/mjlab native64 trace / held arm endpoints / no new physics',font=fonts[2],fill='#c7d2df')
            draw.text((20,a.height-39),a.caption or 'Contact labels come from recorded aggregate forces; positions are actual native samples.',font=fonts[2],fill='#c7d2df')
            draw.text((20,a.height-20),row['id'],font=fonts[2],fill='#8392a8')
            return picture
        if a.preview_only:
            directory=output.with_name(output.stem+'-previews');directory.mkdir(exist_ok=False)
            for view in views:
                for name,t in {'release':0.,**bundle['events'],'end':bundle['scored_until_s']}.items():
                    item=dict(view=view,source_tick=nearest_tick(bundle['time_s'],t),source_time_s=t,
                        playback_speed=0,segment='robot_contact_hold' if name=='robot' else 'preview')
                    path=directory/f'{view}-{name}.jpg';draw_frame(item).save(path,quality=92)
                    previews.append(dict(file=str(path),sha256=sha(path),event=name,view=view))
        else:
            try:
                writer=imageio.get_writer(output,fps=30,codec='libx264',quality=8,macro_block_size=1,
                    ffmpeg_params=['-threads','2','-movflags','+faststart'])
                for item in frames:writer.append_data(np.asarray(draw_frame(item)))
            finally:
                if writer:writer.close()
    receipt=dict(version=VERSION,status='preview_complete' if a.preview_only else 'completed',
        record_id=row['id'],context_id=row['context_id'],label=a.label,caption=a.caption,
        record_sha256=bundle['record_sha256'],trace_sha256=row['native_cache_sha256'],native_cache_row=row['native_cache_row'],
        input_sha256=bundle['input_sha256'],asset_sha256=bundle['assets_sha256'],
        model_source_sha256=bundle['model_source_sha256'],renderer_source_sha256=sha(__file__),
        full_model_xml_sha256=model_hash,backend=backend,backend_request=environment,
        packages={n:importlib.metadata.version(n) for n in ('mujoco','numpy','Pillow','imageio')},
        width=a.width,height=a.height,fps=30,duration_s=len(frames)/30,events=bundle['events'],
        source_interval_s=[0,bundle['scored_until_s']],capture=row['route_success'],whole_ball_entry=row['whole_ball_entry'],
        cameras=views,frames=[] if a.preview_only else frames,native_steps_added=0,held_arm_poses=True,
        interpolation=False,sampled_ball_centers_only=True,ball_spin_recorded=False,historical_labels_unchanged=True,
        output_sha256=None if a.preview_only else sha(output),video_bytes=None if a.preview_only else output.stat().st_size,previews=previews)
    manifest.write_text(json.dumps(receipt,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(status=receipt['status'],output=str(output),manifest=str(manifest),duration_s=receipt['duration_s'])),flush=True)
    return receipt


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('study','audit','assets','output'):p.add_argument('--'+name,type=Path,required=True)
    for name in ('record','label'):p.add_argument('--'+name,required=True)
    p.add_argument('--caption',default='');p.add_argument('--cameras',type=Path)
    p.add_argument('--preview-only',action='store_true')
    p.add_argument('--width',type=int,default=1280);p.add_argument('--height',type=int,default=720)
    return render(p.parse_args())

if __name__=='__main__':main()
