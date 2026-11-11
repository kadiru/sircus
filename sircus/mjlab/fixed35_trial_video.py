"""CPU-only films of every authenticated native trial at three fixed-grid cells.

No simulation is advanced. Recorded ball-center samples and observed initial
joint positions are rendered through mj_forward. Setup refusals are indexed,
never animated as invented robot motion. All native trials, including invalid
physics and repeated failures, remain visible in cohort/ledger order.
"""
from __future__ import annotations
import argparse
from collections import Counter
import importlib.metadata
import json
import math
import os
from pathlib import Path
import subprocess
import time

import numpy as np
from .saved_shot_video import (SOURCE_MODULES, build_held_scene, contained, digest,
    nearest_tick, read, require, sha, software_backend, vector, verify_assets)

VERSION='sircus-fixed35-all-trials-video-v1'
CONTEXTS=tuple('fixed35-dev-grid-'+c for c in ('03-04','03-05','04-06'))
COHORTS=('fixed35-imitation-v1','fixed35-recovery-v1','fixed35-six-v1','fixed35-bouncer-v1')
COHORT_LABELS=dict(zip(COHORTS,('Model test','Earlier parameter search','Latest parameter search','Alternate bouncer joints')))
CONTACTS=('board','bucket','table','floor','robot','unattributed')
COLORS={'board':'#61c9ff','bucket':'#60e5aa','table':'#d1b786','floor':'#8b94aa','robot':'#ff6565','unattributed':'#d78bff'}
SCORER='sircus-table-assisted-native-capture-v3'
DT=1/15360
MESA_VENDOR='/usr/share/glvnd/egl_vendor.d/50_mesa.json'


def jsonl(path):
    return [(i,json.loads(line)) for i,line in enumerate(Path(path).read_text().splitlines(),1) if line.strip()]


def cell_xy(context):
    require(context in CONTEXTS,'Only the three requested canonical cells are allowed')
    row,col=map(int,context[-5:].split('-'))
    return [float(np.linspace(.1,1.24,7)[col]),float(np.linspace(-.65,.25,5)[row])]


def cohort_inputs(descriptor,assets):
    study=Path(descriptor['path']).resolve();audit_path=Path(descriptor['audit_path']).resolve()
    paths=dict(report=study/'report.json',contract=study/'contract.json',manifest=study/'evidence-manifest.json',rows=study/'rows.jsonl',audit=audit_path)
    for key in ('report','contract','manifest','audit'):
        require(sha(paths[key])==descriptor[key+'_sha256'],'Inventory source binding differs: '+key)
    report,contract,manifest,audit=(read(paths[k]) for k in ('report','contract','manifest','audit'))
    require(report['status']=='completed' and audit['status']=='passed','Completed independently audited cohort required')
    require(audit.get('full_checks_verified') is True and audit.get('native_checks_verified') is True and audit.get('failures')==[], 'Full native audit required')
    require(audit['checks_count']==len(audit['checks'])>0 and all(c['passed'] is True for c in audit['checks']),'All independent audit checks must pass')
    require(audit['report_sha256']==sha(paths['report']) and audit['evidence_manifest_sha256']==sha(paths['manifest']),'Audit source binding differs')
    require(report['evidence_sha256']==manifest,'Report manifest differs')
    for key in ('contract','rows'):require(manifest[paths[key].name]==sha(paths[key]),'Manifest source binding differs: '+key)
    unsigned=dict(contract);fp=unsigned.pop('fingerprint')
    require(fp==digest(unsigned)==report['contract_fingerprint']==audit['contract_fingerprint'],'Immutable contract differs')
    require(contract['protocol']['substeps']==64 and contract['identity']['trace_schema']['native_dt_s']==DT,'Fixed64 trace contract required')
    scene=contract['scene'];require(digest(scene)==contract['identity']['scene_sha256'],'Original scene differs')
    hashes=verify_assets(assets,contract['identity']['asset_sha256'])
    root=Path(__file__).resolve().parents[2]
    for module in SOURCE_MODULES:
        require(sha(root/module)==contract['identity']['source_sha256'][module],'Model source differs: '+module)
    rows=jsonl(paths['rows']);require(len({r['id'] for _,r in rows})==len(rows),'Duplicate ledger IDs')
    return dict(study=study,contract=contract,scene=scene,manifest=manifest,rows=rows,
        inputs={k:sha(v) for k,v in paths.items()},assets=hashes)


def load_inventory(path,assets):
    inventory=read(path)
    require(set(inventory['cohorts'])==set(COHORTS),'All four historical cohorts are required')
    source={name:cohort_inputs(inventory['cohorts'][name],assets) for name in COHORTS}
    require(len({digest(s['scene']) for s in source.values()})==1,'Cohort scene mismatch')
    for field in ('physics','calibration'):
        require(len({digest(s['contract']['identity'].get(field)) for s in source.values()})==1,'Cohort '+field+' mismatch')
    expected=[]
    for name in COHORTS:
        for line,row in source[name]['rows']:
            if row.get('context_id') not in CONTEXTS:continue
            require(row['bucket_xy_m']==cell_xy(row['context_id']),'Canonical bucket coordinates changed')
            expected.append((name,line,row))
    indexed={(item['cohort'],item['row_id']):item for item in inventory['records']}
    require(len(indexed)==len(inventory['records'])==len(expected),'Inventory omitted or duplicated requests')
    records=[]
    for name,line,row in expected:
        item=indexed.get((name,row['id']));require(item is not None,'Inventory omitted ledger request')
        require(item['row_line']==line and item['row_sha256']==digest(row) and item['context_id']==row['context_id'],'Inventory row identity differs')
        if row.get('native_release'):
            require(row.get('contact_protocol',{}).get('substeps')==64 and row.get('scoring_version')==SCORER,'Native64/v3 required')
            native=item['native'];relative=row['adapter_directory']+'/'+row['native_cache_file']
            require(native['relative_path']==relative and native['column']==row['native_cache_row'] and native['sha256']==row['native_cache_sha256'],'Native archive identity differs')
            require(source[name]['manifest'][relative]==row['native_cache_sha256'],'Archive not in audited manifest')
        else:
            require(item.get('native') is None and not row.get('native_cache_file') and not row.get('route_success'),'Declined setup cannot have invented flight')
        records.append(dict(cohort=name,row=row,row_line=line,row_sha256=digest(row),native=item.get('native')))
    return dict(inventory=inventory,inventory_sha256=sha(path),sources=source,records=records)


def contact_intervals(times,forces):
    """Force interval k is [time[k], time[k+1]]; event stamps use its end."""
    require(forces.shape==(len(times)-1,6) and np.isfinite(forces).all(),'Dense six-channel interval forces required')
    result={}
    for j,name in enumerate(CONTACTS):
        mask=forces[:,j]>.01;changes=np.diff(np.r_[False,mask,False].astype(np.int8))
        starts=np.flatnonzero(changes==1);ends=np.flatnonzero(changes==-1)
        result[name]=[dict(start_tick=int(a),end_tick=int(b),start_s=float(times[a]),
            first_event_s=float(times[a+1]),end_s=float(times[b]),peak_N=float(np.max(forces[a:b,j]))) for a,b in zip(starts,ends)]
    return result


def active_contacts(intervals,previous_s,current_s):
    """Include microsecond contacts between video frames, not just sampled force."""
    return [name for name in CONTACTS if any(x['end_s']>previous_s and x['start_s']<current_s for x in intervals[name])]


def load_trace(record,source,verified_cache):
    row=record['row'];path=contained(source['study'],record['native']['relative_path'])
    if path not in verified_cache:
        require(sha(path)==row['native_cache_sha256'],'Immutable raw archive changed');verified_cache.add(path)
    column=row['native_cache_row']
    require(type(column) is int and column>=0,'Literal raw archive column required')
    with np.load(path,allow_pickle=False) as z:
        require(column<len(z['record_ids']) and str(z['record_ids'][column])==row['id'],'Raw archive ID/column mismatch')
        times=np.asarray(z['time_s'],float);position=np.asarray(z['position_m'][:,column],float);forces=np.asarray(z['forces_N'][:,column],float)
    require(times.ndim==1 and len(times)>1 and np.isfinite(times).all() and np.allclose(times,np.arange(len(times))*DT,atol=1e-12,rtol=0),'Recorded native clock changed')
    require(position.shape==(len(times),3) and np.isfinite(position).all(),'Finite recorded positions required')
    require(np.array_equal(position[0],row['native_initial_ball_center_m']),'Recorded release center differs')
    require(type(row['physics_steps']) is int and 0<row['physics_steps']<len(times),'Original score terminal outside trace')
    for key in ('q10','q14'):vector(row['native_initial_'+key],6,key)
    return dict(row=row,contract=source['contract'],scene=source['scene'],positions=position,time_s=times,
        contacts=contact_intervals(times,forces),scored_until_s=float(times[row['physics_steps']]),
        cohort=record['cohort'],row_sha256=record['row_sha256'],cache_sha256=row['native_cache_sha256'])


def frame_ticks(times,fps=24):
    require(type(fps) is int and 1<=fps<=60,'Bounded integer frame rate required')
    require(len(times)>1 and times[0]==0 and np.all(np.diff(times)>0),'Monotonic actual sample clock required')
    # Final exact source sample is included; the added endpoint frame is explicit.
    wanted=np.arange(math.ceil(float(times[-1])*fps)+1)/fps
    return [nearest_tick(times,min(float(t),float(times[-1]))) for t in wanted]


def camera_spec(bundle,ball=None):
    row=bundle['row'];board=np.asarray(row['board_center_m']);bucket=np.r_[row['bucket_xy_m'],.10]
    release=np.asarray(row['native_initial_ball_center_m']);direction=bucket[:2]-board[:2]
    side=np.array([direction[1],-direction[0]])
    target=np.mean(np.stack((board,bucket,release)),axis=0)
    distance=max(1.65,2.8*float(np.linalg.norm(np.ptp(np.stack((board,bucket,release)),axis=0))))
    if ball is not None:
        # Expand only when the actual recorded ball leaves the work area; no invented motion.
        delta=np.asarray(ball)-target;length=float(np.linalg.norm(delta))
        if length>.8:
            target=target+.5*(1-.8/length)*delta
            distance=max(distance,2.6*(length/2+.4))
    return dict(lookat=target.tolist(),distance=distance,azimuth=math.degrees(math.atan2(side[1],side[0]))+12.,elevation=-25.)


def outcome(row):
    if row.get('invalid_physics'):return 'INVALID PHYSICS / excluded from capture evidence'
    if row.get('route_success') and not row.get('invalid'):return 'CAPTURE / one second fully inside'
    return 'NO CAPTURE'


def compact_record(record):
    row=record['row']
    keys=('id','context_id','phase','method','values','bucket_xy_m','native_release','invalid','invalid_physics','invalid_physics_reasons',
          'route_success','entry_class','board_bounce','whole_ball_entry','max_retention_s','physics_steps',
          'native_cache_file','native_cache_row','native_cache_sha256','adapter_directory','ik_branch_slot','candidate_index')
    return dict(cohort=record['cohort'],row_line=record['row_line'],row_sha256=record['row_sha256'],
                row={k:row[k] for k in keys if k in row})


def validate_render_environment(env):
    """Reject ambiguous EGL vendor selection before importing any GL module."""
    require(env.get('CUDA_VISIBLE_DEVICES')=='','Hide both GPUs before importing MuJoCo renderer')
    backend=env.get('MUJOCO_GL')
    if backend=='osmesa':
        require(env.get('PYOPENGL_PLATFORM') in (None,'','osmesa'),'OSMesa platform must not select another GL backend')
        return dict(requested_backend='osmesa',cuda_visible_devices='')
    require(backend=='egl','Explicit OSMesa or Mesa EGL CPU backend required')
    expected=dict(PYOPENGL_PLATFORM='egl',LIBGL_ALWAYS_SOFTWARE='1',MESA_LOADER_DRIVER_OVERRIDE='llvmpipe',
                  __EGL_VENDOR_LIBRARY_FILENAMES=MESA_VENDOR)
    for key,value in expected.items():require(env.get(key)==value,'Explicit CPU Mesa EGL setting required: '+key)
    vendor=read(MESA_VENDOR)
    require(vendor.get('ICD',{}).get('library_path')=='libEGL_mesa.so.0','EGL vendor must resolve the Mesa ICD, never NVIDIA')
    return dict(requested_backend='egl',cuda_visible_devices='',environment=expected,
                vendor_file=MESA_VENDOR,vendor_file_sha256=sha(MESA_VENDOR),vendor_library='libEGL_mesa.so.0')


def render(a):
    backend_request=validate_render_environment(os.environ)
    for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','LP_NUM_THREADS'):os.environ[key]='2'
    require(a.width>=960 and a.height>=540 and a.width<=1920 and a.height<=1080 and a.width%2==a.height%2==0,'Even960x540..1920x1080 output required')
    require(a.fps in (24,30),'24 or30 FPS required')
    context='fixed35-dev-grid-'+a.case if len(a.case)==5 else a.case
    require(context in CONTEXTS,'One literal requested case required')
    grid_row,grid_col=map(int,context[-5:].split('-'));human_cell=f'row {grid_row+1}, column {grid_col+1}'
    output=Path(a.output).resolve();require(output.suffix=='.mp4','MP4 output required')
    manifest=output.with_suffix('.preview.json' if a.preview_only else '.json')
    require(not manifest.exists() and (a.preview_only or not output.exists()),'Fresh video/receipt required')
    data=load_inventory(a.inventory,a.assets)
    for source in data['sources'].values():require(not output.is_relative_to(source['study']),'Output cannot mutate evidence')
    require(not output.is_relative_to(Path(a.assets).resolve()),'Output cannot mutate assets')
    records=[r for r in data['records'] if r['row']['context_id']==context]
    native=[r for r in records if r['row']['native_release']];total_native=len(native)
    if a.sample_only:native=native[:1]
    declined=[r for r in records if not r['row']['native_release']]
    require(native,'Recorded native trials required')
    import mujoco
    from OpenGL import GL
    from PIL import Image,ImageDraw,ImageFont
    import imageio.v2 as imageio
    font_path='/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
    fonts=[ImageFont.truetype(font_path,s) if Path(font_path).exists() else ImageFont.load_default() for s in (28,21,17,14)]
    width,height=a.width,a.height;header=112;footer=128;viewport_height=height-header-footer
    require(viewport_height>=300,'Video viewport too small for readable geometry')
    output.parent.mkdir(parents=True,exist_ok=True)
    previews=output.with_name(output.stem+'-previews')
    if a.preview_only:previews.mkdir(exist_ok=False)
    writer=None;renderer=None;backend=None;clips=[];preview_rows=[];verified=set();frames=0;start=time.monotonic()
    scope='Saved native64/v3 samples; held arm joints; no physics stepping, interpolation or invented setup motion'
    def card(title,lines):
        im=Image.new('RGB',(width,height),'#101722');draw=ImageDraw.Draw(im)
        draw.text((35,35),title,font=fonts[0],fill='white')
        for i,line in enumerate(lines):draw.text((35,95+35*i),line,font=fonts[1],fill='#c7d2df')
        draw.text((35,height-40),scope,font=fonts[2],fill='#9aaabc')
        return np.asarray(im)
    def append_card(image,seconds):
        nonlocal frames
        for _ in range(round(a.fps*seconds)):writer.append_data(image);frames+=1
    try:
        if not a.preview_only:
            writer=imageio.get_writer(output,fps=a.fps,codec='libx264',quality=8,pixelformat='yuv420p',macro_block_size=1,ffmpeg_params=['-threads','2','-movflags','+faststart'])
            append_card(card(('Sircus / SAMPLE ONLY / ' if a.sample_only else 'Sircus / all trials / ')+human_cell,[f'Bucket XY: {cell_xy(context)[0]*1000:.0f}, {cell_xy(context)[1]*1000:.0f} mm',
                f'{len(native)} recorded drops / {len(declined)} setup declines',
                'All drops in historical cohort and request order; complete scored horizon.',
                '1x playback; contact categories from recorded forces > 0.01 N.',
                'Arms were held at their saved poses during these experiments.']),3.)
        for index,record in enumerate(native):
            bundle=load_trace(record,data['sources'][record['cohort']],verified);row=bundle['row'];archive_times=bundle['time_s'];times=archive_times[:row['physics_steps']+1];ticks=frame_ticks(times,a.fps)
            model,state,ball_q,model_sha=build_held_scene(bundle,a.assets)
            model.vis.global_.offwidth=max(width,model.vis.global_.offwidth);model.vis.global_.offheight=max(viewport_height,model.vis.global_.offheight)
            renderer=mujoco.Renderer(model,height=viewport_height,width=width)
            current_backend=software_backend((GL.glGetString(GL.GL_RENDERER) or b'').decode(),(GL.glGetString(GL.GL_VENDOR) or b'').decode(),os.environ['CUDA_VISIBLE_DEVICES'])
            if backend is None:backend=current_backend
            else:require(backend==current_backend,'Rendering backend changed')
            camera=mujoco.MjvCamera();camera.type=mujoco.mjtCamera.mjCAMERA_FREE
            def frame(tick,previous):
                when=float(times[tick]);ball=bundle['positions'][tick];state.qpos[ball_q:ball_q+3]=ball;state.time=when
                mujoco.mj_forward(model,state);spec=camera_spec(bundle,ball)
                camera.lookat[:]=spec['lookat']
                for key in ('distance','azimuth','elevation'):setattr(camera,key,spec[key])
                renderer.update_scene(state,camera=camera)
                im=Image.new('RGB',(width,height),'#101722');im.paste(Image.fromarray(renderer.render()),(0,header));draw=ImageDraw.Draw(im)
                label=f'{human_cell} / drop {index+1:02d} of {total_native} / {COHORT_LABELS[record["cohort"]]}'
                draw.text((18,7),label,font=fonts[0],fill='white')
                values=row['values'];distance=np.linalg.norm(np.asarray(row['bucket_xy_m'])-np.asarray(values[:2]))
                draw.text((18,42),f'Bucket ({row["bucket_xy_m"][0]*1000:.0f}, {row["bucket_xy_m"][1]*1000:.0f}) mm / Board {values[2]*1000:.1f} mm / tilt {values[3]:.2f} deg / gap {values[4]*1000:.1f} mm / distance {distance*1000:.1f} mm',font=fonts[2],fill='#c7d2df')
                active=active_contacts(bundle['contacts'],previous,when)
                status=' | '.join('Robot contact' if name=='robot' else name for name in active) if active else 'free flight / no contact in this frame interval'
                post=' / after original score terminal' if when>bundle['scored_until_s'] else ''
                draw.text((18,68),f'1x / native t={when:.4f}s{post} / {status}',font=fonts[2],fill='#85dec2' if 'robot' not in active else '#ff7777')
                draw.text((18,91),outcome(row),font=fonts[3],fill='#ffb17d' if row.get('invalid_physics') else '#c7d2df')
                y0=height-footer+8;x0=116;x1=width-20;length=float(times[-1]);scale=(x1-x0)/length
                for j,name in enumerate(CONTACTS):
                    y=y0+j*12;draw.text((15,y-3),name,font=fonts[3],fill=COLORS[name]);draw.line((x0,y+5,x1,y+5),fill='#293748',width=1)
                    for event in bundle['contacts'][name]:
                        if event['start_s']>=length:continue
                        left=x0+scale*event['start_s'];right=x0+scale*min(length,event['end_s'])
                        draw.rectangle((left,y+2,max(left+2,right),y+8),fill=COLORS[name])
                cursor=x0+scale*when;terminal=x0+scale*bundle['scored_until_s']
                draw.line((terminal,y0-3,terminal,y0+75),fill='#9a9a9a',width=1);draw.line((cursor,y0-3,cursor,y0+75),fill='white',width=2)
                draw.text((18,height-43),f'Contact timeline 0–{length:.2f}s / gray marker: score terminal {bundle["scored_until_s"]:.4f}s / white: playback',font=fonts[3],fill='#aabbcc')
                draw.text((18,height-23),'Recorded ball centers + held joints; CPU render, no new physics. '+row['id'],font=fonts[3],fill='#9aaabc')
                return np.asarray(im)
            clip=dict(cohort=record['cohort'],record_id=row['id'],record_sha256=digest(row),cache_file=record['native']['relative_path'],
                cache_sha256=row['native_cache_sha256'],cache_column=row['native_cache_row'],frame_start=frames,
                model_xml_sha256=model_sha,original_outcome=outcome(row),values=row['values'],bucket_xy_m=row['bucket_xy_m'],
                native_initial_q10=row['native_initial_q10'],native_initial_q14=row['native_initial_q14'],
                archive_trace_end_s=float(archive_times[-1]),rendered_start_tick=0,rendered_end_tick=row['physics_steps'],full_scored_horizon=True,
                full_trace_end_s=float(times[-1]),score_terminal_s=bundle['scored_until_s'],contacts=bundle['contacts'],
                camera=camera_spec(bundle),camera_follow_rule='Widen only beyond0.8m from initial shot center',
                source_ticks=ticks,source_ticks_sha256=digest(ticks),source_time_rule='source_tick/15360',playback_speed=1.,endpoint_frame_added=True)
            if a.preview_only:
                # Preview first shot from each cohort plus every bouncer alternative, not a favorable-result selection.
                selected=index==0 or native[index-1]['cohort']!=record['cohort'] or record['cohort']=='fixed35-bouncer-v1'
                if selected:
                    events={'release':0.,'board':next((e['first_event_s'] for e in bundle['contacts']['board']),0.),'robot':next((e['first_event_s'] for e in bundle['contacts']['robot']),0.),'last':float(times[-1])}
                    for name,when in events.items():
                        if when>times[-1]:continue
                        tick=nearest_tick(times,when);file=previews/f'{index:03d}-{name}.jpg';Image.fromarray(frame(tick,max(0.,when-1/a.fps))).save(file,quality=94)
                        preview_rows.append(dict(file=file.name,sha256=sha(file),record_id=row['id'],source_tick=tick,source_time_s=float(times[tick])))
            else:
                previous=-1.
                for tick in ticks:writer.append_data(frame(tick,previous));frames+=1;previous=float(times[tick])
            clip['frame_end_exclusive']=frames;clips.append(clip)
            renderer.close();renderer=None
            print(json.dumps(dict(progress=index+1,total=len(native),frames=frames,elapsed_s=round(time.monotonic()-start,2))),flush=True)
        if writer is not None:
            reasons=Counter(str(r['row'].get('invalid') or r['row'].get('geometry_reason') or 'setup declined') for r in declined)
            summary=[f'{len(native)} / {total_native} native trials shown; {len(declined)} requests had no release.',
                'Setup declines are indexed in the companion JSON; no trajectory was invented.']
            summary += [f'{count}: {reason[:115]}' for reason,count in reasons.most_common(6)]
            append_card(card('Complete trial ledger / setup-decline appendix',summary),5.)
            writer.close();writer=None
    finally:
        if renderer is not None:renderer.close()
        if writer is not None:writer.close()
    result=dict(version=VERSION,status='preview_complete' if a.preview_only else ('sample_complete' if a.sample_only else 'completed'),case=context,
        inventory_sha256=data['inventory_sha256'],cohorts={name:s['inputs'] for name,s in data['sources'].items()},
        renderer_source_sha256=sha(__file__),helper_source_sha256={n:sha(Path(__file__).resolve().parents[2]/n) for n in (*SOURCE_MODULES,'sircus/mjlab/saved_shot_video.py')},
        fps=a.fps,width=width,height=height,frames=frames,duration_s=frames/a.fps,clips=clips,
        request_count=len(records),native_count=len(native),setup_declines=len(declined),decline_index=[compact_record(r) for r in declined],
        all_native_trials_included=not(a.sample_only or a.preview_only),total_native_count=total_native,rendered_horizon='Recorded release through original row.physics_steps; batch remainder excluded',native_steps_added=0,held_arm_poses=True,ball_spin_recorded=False,interpolation=False,
        original_scores_unchanged=True,contact_force_threshold_N=.01,contact_interval_timestamp='first_event=time_s[k+1], physical interval [time_s[k],time_s[k+1]]',
        backend=backend,backend_request=backend_request,previews=preview_rows,scope=scope,wall_seconds=time.monotonic()-start,
        output_sha256=None if a.preview_only else sha(output),video_bytes=None if a.preview_only else output.stat().st_size)
    manifest.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:result[k] for k in ('status','case','frames','duration_s','wall_seconds')}));return result


def combine(manifests,output):
    paths=[Path(p).resolve() for p in manifests];require(len(paths)==3,'Three complete case movies required')
    reports=[read(p) for p in paths]
    require([r['case'] for r in reports]==list(CONTEXTS),'Canonical case order required')
    require(all(r['version']==VERSION and r['status']=='completed' and r['all_native_trials_included'] for r in reports),'Complete movie receipts required')
    require(len({r['inventory_sha256'] for r in reports})==1,'Same inventory required')
    require(len({(r['fps'],r['width'],r['height']) for r in reports})==1,'Matching video format required')
    output=Path(output).resolve();require(output.suffix=='.mp4' and not output.exists() and not output.with_suffix('.json').exists(),'Fresh combined MP4 required')
    movies=[p.with_suffix('.mp4') for p in paths]
    for p,r in zip(movies,reports):require(sha(p)==r['output_sha256'],'Completed video bytes changed')
    output.parent.mkdir(parents=True,exist_ok=True);listing=output.with_suffix('.concat.txt')
    require(not listing.exists(),'Fresh concat list required')
    # ffmpeg concat demuxer escaping; no shell evaluation.
    listing.write_text(''.join("file '"+str(p).replace("'","'\\''")+"'\n" for p in movies))
    subprocess.run(['ffmpeg','-nostdin','-v','error','-f','concat','-safe','0','-i',str(listing),'-c','copy','-movflags','+faststart',str(output)],check=True)
    result=dict(version=VERSION,status='completed',kind='all_three_cases',inventory_sha256=reports[0]['inventory_sha256'],
        case_movie_manifests=[dict(path=str(p),sha256=sha(p),movie_sha256=r['output_sha256']) for p,r in zip(paths,reports)],
        native_count=sum(r['native_count'] for r in reports),setup_declines=sum(r['setup_declines'] for r in reports),
        frames=sum(r['frames'] for r in reports),fps=reports[0]['fps'],duration_s=sum(r['duration_s'] for r in reports),
        output_sha256=sha(output),video_bytes=output.stat().st_size,native_steps_added=0,transcode=False)
    output.with_suffix('.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');return result


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='mode',required=True)
    r=sub.add_parser('render');r.add_argument('--inventory',type=Path,required=True);r.add_argument('--assets',type=Path,required=True)
    r.add_argument('--case',choices=('03-04','03-05','04-06'),required=True);r.add_argument('--output',type=Path,required=True)
    r.add_argument('--width',type=int,default=1280);r.add_argument('--height',type=int,default=720);r.add_argument('--fps',type=int,default=24);r.add_argument('--preview-only',action='store_true');r.add_argument('--sample-only',action='store_true')
    c=sub.add_parser('combine');c.add_argument('--manifests',type=Path,nargs=3,required=True);c.add_argument('--output',type=Path,required=True)
    args=p.parse_args();render(args) if args.mode=='render' else print(json.dumps(combine(args.manifests,args.output)))


if __name__=='__main__':main()
