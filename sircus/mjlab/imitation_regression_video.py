"""Replay audited corrected-imitation regression observations using CPU software rendering only.

Actual saved ball centers and held joint poses, with no physics stepping or
trajectory interpolation. Selected examples do not replace the full study.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import os
from pathlib import Path

import numpy as np

from .saved_shot_video import (SOURCE_MODULES, build_held_scene, contained, digest,
                              nearest_tick, read, require, sha, software_backend,
                              vector, verify_assets)
from .height_tilt_video import CONTACTS, first_events, cameras
from .bouncer_clearance_video import frame_plan as native_frame_plan, stage as native_stage
STUDY_VERSION='sircus-imitation-regression-v1'
AUDIT='sircus-imitation-regression-audit-v1'
VERSION='sircus-imitation-regression-recorded-video-v1'

FPS = 30


def load_recordings(a):
    study=Path(a.study).resolve()
    paths={name:study/filename for name,filename in (
        ('report','report.json'),('rows','rows.jsonl'),('contract','contract.json'),
        ('evidence_manifest','evidence-manifest.json'))}
    paths['audit']=Path(a.audit)
    report,audit,contract,manifest=(read(paths[k]) for k in ('report','audit','contract','evidence_manifest'))
    require(report.get('version')==STUDY_VERSION and report.get('status')=='completed', 'Completed imitation regression study required')
    require(audit.get('version')==AUDIT and audit.get('status')=='passed'
            and audit.get('full_checks_verified') is True and audit.get('native_checks_verified') is True
            and audit.get('failures')==[], 'Passed independent native audit required')
    require(isinstance(audit.get('checks'),list) and audit['checks']
            and audit.get('checks_count')==len(audit['checks'])
            and all(c.get('passed') is True for c in audit['checks']), 'All independent checks must pass')
    require(audit['report_sha256']==sha(paths['report'])
            and audit['evidence_manifest_sha256']==sha(paths['evidence_manifest']), 'Audit bytes differ')
    require(report['evidence_sha256']==manifest, 'Report manifest differs')
    for key in ('rows','contract'):
        require(manifest[paths[key].name]==sha(paths[key]), 'Evidence bytes differ: '+key)
    unsigned=dict(contract);fp=unsigned.pop('fingerprint')
    require(fp==digest(unsigned)==audit['contract_fingerprint']==report['contract_fingerprint'], 'Contract fingerprint differs')
    require(contract['protocol']['substeps']==64, 'Fixed64 protocol required')
    scene=contract['scene']
    require(digest(scene)==contract['identity']['scene_sha256'], 'Original scene differs')
    assets=verify_assets(a.assets,contract['identity']['asset_sha256'])
    root=Path(__file__).resolve().parents[2]
    sources={name:sha(root/name) for name in SOURCE_MODULES}
    for name,value in sources.items():
        require(contract.get('source_sha256',{}).get(name)==value,'Current frozen model differs: '+name)
        require(contract['identity']['source_sha256'].get(name)==value,'Historical model differs: '+name)
    selected=read(a.records)
    require(isinstance(selected,list) and len(selected)==3, 'Exactly three matched first-repeat records required')
    require(all(isinstance(item,dict) and isinstance(item.get('id'),str)
                and isinstance(item.get('label'),str) and item['label'] for item in selected), 'Explicit clip identity and label required')
    require(len({item['id'] for item in selected})==len(selected), 'Duplicate clip selection')
    rows=[json.loads(line) for line in paths['rows'].read_text().splitlines() if line.strip()]
    require(len({row['id'] for row in rows})==len(rows), 'Duplicate source record IDs')
    require(selected==choose_records(rows,a.context),'Explicit selection differs from deterministic first-repeat comparison')
    lookup={row['id']:row for row in rows};bundles=[]
    dt=contract['identity']['trace_schema']['native_dt_s']
    require(dt==1/15360,'Canonical native64 clock required')
    for item in selected:
        require(item['id'] in lookup,'Selected record missing: '+item['id'])
        row=lookup[item['id']]
        if row.get('native_release') is False:
            require(not row.get('route_success') and not row.get('native_cache_file'), 'Refusal must not claim capture or native recording')
            bundles.append(dict(contract=contract,scene=scene,row=row,label=item['label'],record_sha256=digest(row),events={},no_flight=True))
            continue
        require(row.get('native_release') is True and row.get('invalid_physics') is False
                and row.get('contact_protocol',{}).get('substeps')==64,'Valid native64 release required')
        cache=contained(study/'native',row['native_cache_file']);cache_hash=sha(cache)
        require(cache_hash==row['native_cache_sha256']==manifest['native/'+row['native_cache_file']], 'Recorded archive changed')
        column=row['native_cache_row'];require(type(column) is int and column>=0,'Exact native archive column required')
        with np.load(cache,allow_pickle=False) as archive:
            require(column<len(archive['record_ids']) and str(archive['record_ids'][column])==item['id'],'Archive identity differs')
            times=np.asarray(archive['time_s'],dtype=float)
            positions=np.asarray(archive['position_m'][:,column],dtype=float)
            forces=np.asarray(archive['forces_N'][:,column],dtype=float)
        require(times.ndim==1 and len(times)>1 and np.isfinite(times).all()
                and np.allclose(times,np.arange(len(times))*dt,atol=1e-12,rtol=0),'Recorded clock differs')
        require(positions.shape==(len(times),3) and np.isfinite(positions).all() and np.isfinite(forces).all(),'Finite samples required')
        require(np.array_equal(positions[0],np.asarray(row['native_initial_ball_center_m'])),'Release center differs')
        for key in ('q10','q14'):vector(row['native_initial_'+key],6,key)
        events=first_events(times,forces,row)
        require(type(row.get('route_success')) is bool and type(row.get('whole_ball_entry')) is bool,'Recorded outcome required')
        bundles.append(dict(contract=contract,scene=scene,row=row,label=item['label'],time_s=times,
            positions=positions,events=events,record_sha256=digest(row),cache_sha256=cache_hash,
            native_dt_s=dt,scored_until_s=float(times[int(row['physics_steps'])])))
    shared=dict(source=dict(report_sha256=sha(paths['report']),audit_sha256=sha(paths['audit']),contract_fingerprint=fp),
        input_sha256={**{key:sha(path) for key,path in paths.items()},'clip_selection':sha(a.records)},
        asset_sha256=assets,model_source_sha256=sources,scene_sha256=digest(scene))
    return bundles,shared


def stage(item,bundle):
    row=bundle['row']
    if not row.get('route_success') and row.get('max_retention_s',0)>=1:
        if item['segment']=='outcome_hold' or ('capture' in bundle['events'] and item['source_time_s']>=bundle['events']['capture']):
            return 'Retained one second, but route rejected / forbidden contact'
    return native_stage(item,bundle)


def render(a):
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "", "Hide both GPUs before importing the renderer")
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "LP_NUM_THREADS"):
        os.environ[name] = "2"
    require(a.width >= 640 and a.height >= 360 and a.width <= 1920 and a.height <= 1080
            and a.width % 2 == a.height % 2 == 0, "Even dimensions between640x360 and1920x1080 required")
    output = Path(a.output).resolve()
    require(output.suffix == ".mp4", "MP4 output required")
    for path in (a.study, a.assets):
        require(not output.is_relative_to(Path(path).resolve()), "Keep rendered outputs outside original evidence/assets")
    manifest_path = output.with_suffix(".preview.json" if a.preview_only else ".json")
    preview_dir = output.with_name(output.stem + "-previews")
    require(not manifest_path.exists() and (a.preview_only or not output.exists()), "Refusing to overwrite rendered evidence")
    bundles, shared = load_recordings(a)
    plan, clips = frame_plan(bundles)
    overrides = {} if a.cameras is None else read(a.cameras)
    output.parent.mkdir(parents=True, exist_ok=True)
    if a.preview_only:
        preview_dir.mkdir(exist_ok=False)
    import mujoco
    from OpenGL import GL
    from PIL import Image, ImageDraw, ImageFont
    import imageio.v2 as imageio
    fonts = []
    for size in (27, 21, 18):
        paths = ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "/System/Library/Fonts/Supplemental/Arial.ttf")
        fonts.append(next((ImageFont.truetype(p, size) for p in paths if Path(p).is_file()), ImageFont.load_default()))
    previews, model_hashes, view_settings = [], {}, {}
    writer, renderer, backend = None, None, None
    try:
        if not a.preview_only:
            writer = imageio.get_writer(output, fps=FPS, codec="libx264", quality=8, macro_block_size=1,
                                       ffmpeg_params=["-threads", "2", "-movflags", "+faststart"])
        for index, bundle in enumerate(bundles):
            if bundle.get('no_flight'):
                image = Image.new('RGB', (a.width,a.height), '#101722')
                draw = ImageDraw.Draw(image)
                draw.text((28,35), bundle['label'], font=fonts[0], fill='white')
                draw.text((28,110), 'NO FLIGHT / setup declined', font=fonts[0], fill='#f0c680')
                draw.text((28,155), str(bundle['row'].get('invalid') or 'setup rejected'), font=fonts[1], fill='#c7d2df')
                draw.text((28,210), 'No ball was released; no pose or trajectory is invented.', font=fonts[2], fill='#c7d2df')
                draw.text((28,250), 'A finite IK rejection does not prove the pose impossible.', font=fonts[2], fill='#c7d2df')
                if a.preview_only:
                    path=preview_dir/f'{index}-no-flight.jpg';image.save(path,quality=94)
                    previews.append(dict(name=path.name,sha256=sha(path),record_id=bundle['row']['id'],no_flight=True))
                else:
                    for item in plan:
                        if item['clip_index']==index:writer.append_data(np.asarray(image))
                continue
            model, data, ball_q, model_hash = build_held_scene(bundle, a.assets)
            model_hashes[bundle["row"]["id"]] = model_hash
            model.vis.global_.offwidth = max(a.width, model.vis.global_.offwidth)
            model.vis.global_.offheight = max(a.height, model.vis.global_.offheight)
            renderer = mujoco.Renderer(model, height=a.height, width=a.width)
            backend = software_backend((GL.glGetString(GL.GL_RENDERER) or b"").decode(),
                                       (GL.glGetString(GL.GL_VENDOR) or b"").decode(), os.environ.get("CUDA_VISIBLE_DEVICES"))
            views = cameras(bundle)
            views.update(overrides.get(bundle["row"]["id"], {}))
            for view in views.values():
                vector(view["lookat"], 3, "camera target")
                require(all(math.isfinite(view[k]) for k in ("distance", "azimuth", "elevation"))
                        and view["distance"] > 0, "Finite camera settings required")
            view_settings[bundle["row"]["id"]] = views
            camera = mujoco.MjvCamera()
            camera.type = mujoco.mjtCamera.mjCAMERA_FREE

            def frame(item):
                data.qpos[ball_q:ball_q + 3] = bundle["positions"][item["source_tick"]]
                data.time = item["source_time_s"]
                mujoco.mj_forward(model, data)
                settings = views[item["view"]]
                camera.lookat[:] = settings["lookat"]
                for key in ("distance", "azimuth", "elevation"):
                    setattr(camera, key, settings[key])
                renderer.update_scene(data, camera=camera)
                image = Image.fromarray(renderer.render())
                draw = ImageDraw.Draw(image)
                draw.rectangle((0, 0, a.width, 106), fill="#101722")
                draw.text((20, 9), bundle["label"], font=fonts[0], fill="white")
                speed = "HOLD" if item["playback_speed"] == 0 else f'{item["playback_speed"]:g}x'
                draw.text((20, 45), f'{speed} | actual time {item["source_time_s"]:.4f}s | {stage(item, bundle)}',
                          font=fonts[1], fill="#85dec2")
                values = bundle["row"]["values"]
                draw.text((20, 76), f'Board center {values[2]*1000:.1f} mm | tilt {values[3]:.2f} deg | drop gap {values[4]*1000:.1f} mm',
                          font=fonts[2], fill="#c7d2df")
                draw.rectangle((0, a.height - 54, a.width, a.height), fill="#101722")
                draw.text((20, a.height - 46), "Saved MuJoCo/mjlab simulation / held arm poses / no new physics or synthesized motion", font=fonts[2], fill="#c7d2df")
                draw.text((20, a.height - 24), f'Native64 | robot contact is an aggregate category | {bundle["row"]["id"]}', font=fonts[2], fill="#c7d2df")
                return np.asarray(image)

            if a.preview_only:
                events = {"release": 0., **{key: value for key, value in bundle["events"].items()
                                          if key in ("board", "robot", "table", "bucket", "capture")}}
                for view in views:
                    for name, when in events.items():
                        tick = nearest_tick(bundle["time_s"], when)
                        item = dict(view=view, source_tick=tick, source_time_s=float(bundle["time_s"][tick]),
                                    playback_speed=0., segment="robot_contact_hold" if name == "robot" else "preview")
                        path = preview_dir / f"{index}-{view}-{name}.jpg"
                        Image.fromarray(frame(item)).save(path, quality=94)
                        previews.append(dict(name=path.name, sha256=sha(path), record_id=bundle["row"]["id"], **item))
            else:
                for item in plan:
                    if item["clip_index"] == index:
                        writer.append_data(frame(item))
            renderer.close()
            renderer = None
        if writer is not None:
            writer.close()
            writer = None
    finally:
        if renderer is not None:
            renderer.close()
        if writer is not None:
            writer.close()
    result = dict(version=VERSION, status="preview_complete" if a.preview_only else "completed", **shared,
                  renderer_source_sha256=sha(__file__),
                  helper_source_sha256={name: sha(Path(__file__).with_name(Path(name).name)) for name in ("sircus/mjlab/saved_shot_video.py", "sircus/mjlab/height_tilt_video.py", "sircus/mjlab/bouncer_clearance_video.py")},
                  full_model_xml_sha256=model_hashes, backend=backend,
                  packages={name: importlib.metadata.version(name) for name in ("mujoco", "numpy", "Pillow", "imageio")},
                  width=a.width, height=a.height, fps=FPS, duration_s=len(plan) / FPS, clips=clips,
                  cameras=view_settings, camera_overrides_sha256=None if a.cameras is None else sha(a.cameras),
                  native_dt_s=1 / 15360, native_steps_added=0, held_arm_poses=True,
                  sampled_ball_centers_only=True, ball_spin_recorded=False, interpolation=False,
                  historical_labels_unchanged=True, contact_threshold_N=.01, contact_force_order=list(CONTACTS),
                  frames=[] if a.preview_only else plan, frames_sha256=None if a.preview_only else digest(plan),
                  output_sha256=None if a.preview_only else sha(output), video_bytes=None if a.preview_only else output.stat().st_size,
                  previews=previews, scope="Matched first repeat of current, corrected and targeted intervention at one inspected development position. Actual held native64 replay; no new physics, learned transit or outcome-selected examples. Original 70-position policy scores stay unchanged.")
    manifest_path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(dict(status=result["status"], manifest=str(manifest_path), duration_s=result["duration_s"], previews=len(previews))))
    return result



def parser():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('study','audit','assets','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--records',type=Path,help='Explicit matched repeat-zero clips; default chosen by context')
    p.add_argument('--context',required=True,help='Exact diagnostic context ID')
    p.add_argument('--cameras',type=Path,help='Optional record-ID to camera-name settings JSON; framing only')
    p.add_argument('--preview-only',action='store_true')
    p.add_argument('--width',type=int,default=1280);p.add_argument('--height',type=int,default=720)
    return p


def choose_records(rows, context):
    """Choose representative repeat zero, never a successful later repeat."""
    variants=('current_exact','corrected_exact','blend50')
    selected=[r for r in rows if r.get('phase')=='diagnostic' and r.get('context_id')==context and r.get('repeat')==0]
    require(len(selected)==3 and {r.get('variant') for r in selected}==set(variants), 'Exact current/corrected/blend50 first repeats required')
    labels={'current_exact':'Current policy / unchanged recipe','corrected_exact':'Corrected policy / unchanged recipe','blend50':'Halfway correction / first repeat'}
    lookup={r['variant']:r for r in selected}
    return [dict(id=lookup[v]['id'],label=labels[v]+' / '+context.replace('corrected-imitation-test-','')) for v in variants]


def frame_plan(bundles, *, fps=FPS):
    frames,clips=[],[]
    for index,bundle in enumerate(bundles):
        start=len(frames)/fps
        if bundle.get('no_flight'):
            row=bundle['row']
            for _ in range(3*fps):
                frames.append(dict(frame=len(frames),video_time_s=len(frames)/fps,clip_index=index,record_id=row['id'],segment='no_flight',view=None,playback_speed=0.,source_tick=None,source_time_s=None))
            clips.append(dict(record_id=row['id'],context_id=row['context_id'],label=bundle['label'],variant=row['variant'],start_s=start,end_s=len(frames)/fps,values=row['values'],bucket_xy_m=row['bucket_xy_m'],capture=False,native_release=False,no_flight=True,reason=row.get('invalid'),events={},record_sha256=bundle['record_sha256']))
        else:
            part,meta=native_frame_plan([bundle],fps=fps)
            offset=len(frames)
            for item in part:
                frames.append(dict(item,frame=item['frame']+offset,video_time_s=item['video_time_s']+start,clip_index=index))
            clips.append(dict(meta[0],start_s=start,end_s=len(frames)/fps,native_release=True,no_flight=False,variant=bundle['row']['variant']))
    require(len(frames)<=120*fps,'Matched movie exceeds two-minute bound')
    return frames,clips


def main():
    a=parser().parse_args()
    if a.records is None:
        out=Path(a.output).resolve();require(not out.is_relative_to(Path(a.study).resolve()),'Keep media outside source evidence')
        out.parent.mkdir(parents=True,exist_ok=True);a.records=out.with_name(out.stem+'-clips.json')
        require(not a.records.exists(),'Preserve existing clip selection')
        rows=[json.loads(line) for line in (a.study/'rows.jsonl').read_text().splitlines() if line.strip()]
        a.records.write_text(json.dumps(choose_records(rows,a.context),indent=2)+'\n')
    return render(a)


if __name__=="__main__":
    main()
