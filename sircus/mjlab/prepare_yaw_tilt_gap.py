"""Add audit-bound tilt/drop-gap results without simulator calls or service restart.

The existing static viewer receives an independently marked HTML/CSS addition.
Raw trajectories, calibration, movies and runtime configuration remain outside Git.
"""
from __future__ import annotations
import argparse
import base64
import html
import json
import math
from pathlib import Path
import re
import time
from .prepare_coverage_recovery import checked_file, digest, read, require, sha, summarize_flight
from .prepare_table_training import atomic_bytes, take
from .prepare_gripper_yaw import compact_row, replace_block, guard_targets
from .yaw_tilt_gap_video import choose_records

VERSION='sircus-yaw-tilt-gap-display-v1'
STUDY='sircus-yaw-tilt-gap-v1'
AUDIT='sircus-yaw-tilt-gap-audit-v1'
VIDEO='sircus-yaw-tilt-gap-recorded-video-v1'
START='<!--YAW_TILT_GAP_START-->'
END='<!--YAW_TILT_GAP_END-->'
VIDEO_START='<!--YAW_TILT_GAP_VIDEOS_START-->'
VIDEO_END='<!--YAW_TILT_GAP_VIDEOS_END-->'
CSS_START='/*SIRCUS_YAW_TILT_GAP_SNAPSHOT_START\n'
CSS_END='\nSIRCUS_YAW_TILT_GAP_SNAPSHOT_END*/'
MAX_SNAPSHOT_BYTES=512*1024
FAR='corrected-imitation-test-r0-c6-n1'
STATS=('attempts','captures','native_releases','setup_rejections','native_misses','invalid_physics','robot_contact_shots','direct_captures','table_assisted_captures','cached_refusal_requests')


def audit_proof(path,raw,audit_path):
    if audit_path is None or not Path(audit_path).is_file():return dict(status='pending',full_checks_verified=False,native_checks_verified=False)
    a=read(audit_path);checks=a.get('checks',[]);mp=Path(path).parent/'evidence-manifest.json'
    manifest=read(mp) if mp.is_file() else None
    ok=(manifest is not None and raw.get('evidence_sha256')==manifest and a.get('evidence_manifest_sha256')==sha(mp)
        and raw.get('status')=='completed' and a.get('version')==AUDIT and a.get('status')=='passed'
        and a.get('full_checks_verified') is True and a.get('native_checks_verified') is True and a.get('failures')==[]
        and bool(checks) and a.get('checks_count')==len(checks) and all(c.get('passed') is True for c in checks)
        and a.get('report_sha256')==sha(path) and a.get('contract_fingerprint')==raw.get('contract_fingerprint'))
    return dict(status='passed' if ok else 'unverified',full_checks_verified=ok,native_checks_verified=ok,
        report_sha256=sha(path),contract_fingerprint=raw.get('contract_fingerprint'),proof_sha256=sha(audit_path),checks_count=a.get('checks_count'))


def request_row(row):
    out=compact_row(row)
    out.update(take(row,('phase','candidate_id','grid_index','case_grid_index','requested_gripper_yaw_rad','adapter_directory','command_values')))
    if row.get('phase')=='discovery':out['candidate_id']=row['id']
    elif row.get('phase')=='selected_repeats':out['candidate_id']=row['selected_from_id']
    out['capture']=row.get('route_success') is True
    out['record_sha256']=digest(row)
    out['robot_contact_steps']=row.get('diagnostic_robot_contact_steps',row.get('robot_contact_steps',0))
    return out


def recordings(study,raw,rows):
    manifest=raw['evidence_sha256'];selected=choose_records(rows,FAR)
    lookup={r['id']:r for r in rows};out=[];groups={}
    for choice in selected:
        row=lookup[choice['id']];item=request_row(row);item['label']=choice['label'];out.append(item)
        if row.get('native_release') is True:
            adapter=row.get('adapter_directory');require(adapter in ('native','native-yaw'),'Exact diagnostic native namespace required')
            groups.setdefault((adapter,row['native_cache_file']),[]).append((row,item))
    import numpy as np
    for (adapter,name),members in groups.items():
        path=checked_file(study,adapter+'/'+name,manifest[adapter+'/'+name])
        with np.load(path,allow_pickle=False) as archive:
            times=archive['time_s'];positions=archive['position_m'];forces=archive['forces_N'];ids=archive['record_ids']
            for row,item in members:
                col=row['native_cache_row'];require(type(col) is int and 0<=col<len(ids),'Native column required')
                require(str(ids[col])==row['id'] and row['native_cache_sha256']==manifest[adapter+'/'+name],'Native record binding differs')
                evidence=summarize_flight(row,times,positions[:,col],forces[:,col])
                evidence['trace'].update(native_cache_sha256=manifest[adapter+'/'+name],native_cache_row=col,record_id=row['id'])
                item.update(compact_row(dict(row,**evidence)))
    return out


def compact(study,*,audit_path=None,pipeline_path=None,now=None):
    study=Path(study);path=study/'report.json';state=read(pipeline_path) if pipeline_path and Path(pipeline_path).is_file() else {}
    result=dict(status=state.get('status','pending'),phase='Awaiting frozen search',counts={},audit=dict(status='pending'),requests=[],recordings=[])
    if path.is_file():
        raw=read(path);require(raw.get('version')==STUDY,'Matching fixed-yaw tilt/gap study required')
        contract=read(study/'contract.json');unsigned=dict(contract);fp=unsigned.pop('fingerprint')
        require(digest(unsigned)==fp==raw.get('contract_fingerprint'),'Contract fingerprint differs')
        require(contract.get('protocol',{}).get('substeps')==64,'Fixed64 protocol required')
        result.update(take(raw,('version','status','phase','contract_fingerprint','expected_requests','maximum_requests','elapsed_s','training','posthoc_development')))
        result['protocol']=contract['protocol'];result['report_sha256']=sha(path)
        result['counts']=take(raw.get('counts'),STATS);result['audit']=audit_proof(path,raw,audit_path)
        result['control_gates']={k:take(v,('status','attempts','direct_captures','table_assisted_captures','negative_captures')) for k,v in raw.get('control_gates',{}).items() if k in ('pre','post')}
        if result['audit']['status']=='passed':
            manifest=raw['evidence_sha256'];rows_path=checked_file(study,'rows.jsonl',manifest['rows.jsonl'])
            rows=[json.loads(line) for line in rows_path.read_text().splitlines() if line.strip()]
            require(len({r['id'] for r in rows})==len(rows),'Duplicate row identity')
            require(len(rows)==result['counts'].get('attempts')==result.get('expected_requests') and 76<=len(rows)<=223,'Bounded complete request ledger required')
            result['requests']=[request_row(r) for r in rows if not r.get('phase','').startswith('controls_')]
            result['counts_by_variant']={k:take(v,STATS) for k,v in raw.get('counts_by_variant',{}).items()}
            result['recordings']=recordings(study,raw,rows)
        else:
            result['counts']=take(result['counts'],('attempts',))
    if state.get('status') in ('failed','audit_failed','audit_deferred','timeout','timed_out','stopped'):
        result['status']=state['status']
        if not path.is_file() and state.get('worker_started') is False:result.update(status='not_run',phase='4090 slot not acquired; no new shots ran',counts=dict(attempts=0))
    terminal=state.get('status') in ('completed','failed','audit_failed','audit_deferred','timeout','timed_out','stopped') and isinstance(state.get('finished_unix_s'),(int,float))
    if result.get('status')=='completed' and result['audit'].get('status')=='passed':terminal=True
    out=dict(version=VERSION,study=result,publication=dict(updated_unix_s=time.time() if now is None else now,terminal=terminal,pipeline_status=state.get('status')))
    require(len(json.dumps(out,allow_nan=False,separators=(',',':')).encode())<=MAX_SNAPSHOT_BYTES,'Compact panel exceeds512KiB')
    return out


def baseline_summary(snapshot):
    r=snapshot['study'];a=r['audit']
    require(snapshot.get('version')==VERSION and r.get('status')=='completed' and a.get('status')=='passed' and a.get('native_checks_verified') is True and a.get('full_checks_verified') is True and a.get('report_sha256')==r.get('report_sha256') and a.get('contract_fingerprint')==r.get('contract_fingerprint') and all(r.get('control_gates',{}).get(k,{}).get('status')=='passed' for k in ('pre','post')),'Verified completed study required')
    out=take(r,('version','status','contract_fingerprint','report_sha256','expected_requests','maximum_requests','elapsed_s','training','posthoc_development','protocol','counts','counts_by_variant','control_gates','audit'))
    out['requests']=[take(x,('id','phase','candidate_id','grid_index','context_id','variant','repeat','values','bucket_xy_m','native_release','capture','invalid','invalid_physics','max_retention_s','whole_ball_entry','robot_contact_steps','requested_gripper_yaw_rad')) for x in r['requests']]
    out['scope']='One inspected development layout; fixed board center and bucket, bucket-facing gripper, 64 substeps. Finite tilt/drop-gap search and nominal repeats, not a new policy score or noise qualification.'
    return out


def encoded(snapshot):
    s=json.dumps(snapshot,allow_nan=False,separators=(',',':'));require(len(s.encode())<=MAX_SNAPSHOT_BYTES,'Snapshot too large')
    return s.replace('<','\\u003c').replace('>','\\u003e').replace('&','\\u0026').replace('*','\\u002a')


def embed_css(original,snapshot):return replace_block(original,CSS_START,CSS_END,CSS_START+encoded(snapshot)+CSS_END,append=True)


def embed_html(original,snapshot):
    web=Path(__file__).with_name('web');panel=(web/'yaw-tilt-gap.html').read_text();css=(web/'yaw-tilt-gap.css').read_text();js=(web/'yaw-tilt-gap.js').read_text()
    js=re.sub(r'^export (?=(?:const|function|class)\b)','',js,flags=re.M)
    require(not re.search(r'^\s*(?:import|export)\b',js,re.M) and '</script' not in js.lower() and '</style' not in css.lower(),'Self-contained panel required')
    require(original.count(VIDEO_START)==original.count(VIDEO_END) and original.count(VIDEO_START)<=1,'Ambiguous movie markers')
    if VIDEO_START in original:
        require(START in original and original.index(START)<original.index(VIDEO_START)<original.index(VIDEO_END)<original.index(END),'Movies must belong to own panel')
        video=original[original.index(VIDEO_START):original.index(VIDEO_END)+len(VIDEO_END)]
        panel=panel.replace('<div data-yaw-tilt-gap="movies"></div>',video)
    block=f'{START}\n<style>{css}</style>\n{panel}\n<script type="application/json" id="yaw-tilt-gap-snapshot">{encoded(snapshot)}</script>\n<script>(function(){{\n{js}\n}})();</script>\n{END}'
    return replace_block(original,START,END,block)


def refresh(study,*,snapshot_path,target_html,target_css,audit_path=None,pipeline_path=None,expected_html_sha256=None,expected_css_sha256=None,now=None):
    guard_targets(study,(audit_path,pipeline_path),(snapshot_path,target_html,target_css))
    hp,cp,sp=map(Path,(target_html,target_css,snapshot_path));oldh=hp.read_bytes();oldc=cp.read_bytes();olds=sp.read_bytes() if sp.exists() else None
    require(expected_html_sha256 is None or sha(hp)==expected_html_sha256,'HTML changed; preserve concurrent work')
    require(expected_css_sha256 is None or sha(cp)==expected_css_sha256,'CSS changed; preserve concurrent work')
    require(olds is None or json.loads(olds).get('version')==VERSION,'Snapshot belongs to another task')
    out=compact(study,audit_path=audit_path,pipeline_path=pipeline_path,now=now)
    newh=embed_html(oldh.decode(),out).encode();newc=embed_css(oldc.decode(),out).encode()
    atomic_bytes(sp,(encoded(out)+'\n').encode(),olds);atomic_bytes(cp,newc,oldc);atomic_bytes(hp,newh,oldh)
    return out


def movie_summary(manifest_path,movie_path,snapshot):
    r=snapshot['study'];audit=r['audit'];require(r.get('status')=='completed' and audit.get('status')=='passed' and audit.get('native_checks_verified') is True,'Audited study required')
    raw=read(manifest_path);require(raw.get('version')==VIDEO and raw.get('status')=='completed','Completed native replay required')
    require(raw.get('native_steps_added')==0 and raw.get('held_arm_poses') is True and raw.get('sampled_ball_centers_only') is True and raw.get('interpolation') is False,'Recorded held endpoint replay only')
    require(raw.get('backend',{}).get('software_only') is True and raw['backend'].get('cuda_visible_devices')=='','CPU software rendering required')
    source=dict(report_sha256=r['report_sha256'],audit_sha256=audit['proof_sha256'],contract_fingerprint=r['contract_fingerprint'])
    require(all(raw.get('source',{}).get(k)==v for k,v in source.items()),'Movie source differs')
    movie=Path(movie_path);size=movie.stat().st_size
    require(0<size<=48*1024*1024 and raw.get('video_bytes')==size and sha(movie)==raw.get('output_sha256'),'Movie checksum differs')
    with movie.open('rb') as f:require(f.read(12)[4:8]==b'ftyp','MP4 required')
    fps=raw.get('fps');frames=raw.get('frames');duration=raw.get('duration_s')
    require(type(fps) is int and 1<=fps<=60 and isinstance(duration,(int,float)) and math.isfinite(duration) and 0<duration<=120 and frames and abs(len(frames)/fps-duration)<1e-9 and digest(frames)==raw.get('frames_sha256'),'Bounded native frame map required')
    records=r['recordings'];clips=raw.get('clips',[]);require([c.get('record_id') for c in clips]==[q['id'] for q in records],'Exact frozen comparison order required')
    end=0.
    for index,(c,q) in enumerate(zip(clips,records)):
        require(c.get('record_sha256')==q.get('record_sha256') and c.get('values')==q.get('values') and c.get('bucket_xy_m')==q.get('bucket_xy_m') and c.get('capture')==q.get('capture') and c.get('native_release')==q.get('native_release'),'Clip differs from audited record')
        require(c.get('start_s')==end and isinstance(c.get('end_s'),(int,float)) and end<c['end_s']<=duration,'Contiguous clips required');end=c['end_s']
        part=[f for f in frames if f.get('clip_index')==index];require(part and all(f.get('record_id')==q['id'] for f in part),'Frame identity differs')
        if q.get('native_release') is False:
            require(c.get('no_flight') is True and not c.get('native_cache_sha256') and all(f.get('source_tick') is None and f.get('source_time_s') is None and f.get('segment')=='no_flight' for f in part),'No invented flight for refusal')
        else:
            require(c.get('native_cache_sha256')==q.get('trace',{}).get('native_cache_sha256') and c.get('native_cache_row')==q['trace'].get('native_cache_row'),'Exact native archive required')
            require(all(type(f.get('source_tick')) is int and f['source_tick']>=0 and isinstance(f.get('source_time_s'),(int,float)) and abs(f['source_time_s']-f['source_tick']/15360)<1e-10 for f in part),'Actual native64 ticks required')
    require(end==duration and all(f.get('frame')==i and abs(f.get('video_time_s',-1)-i/fps)<1e-9 for i,f in enumerate(frames)),'Complete frame sequence required')
    return dict(**take(raw,('version','output_sha256','video_bytes','fps','duration_s','width','height','frames_sha256')),manifest_sha256=sha(manifest_path),source=source,clips=[take(c,('record_id','context_id','variant','label','start_s','end_s','capture','native_release','no_flight','record_sha256','native_cache_sha256','events')) for c in clips])


def publish_movie(snapshot_path,target_html,manifest,movie,*,expected_html_sha256):
    snapshot=read(snapshot_path);require(snapshot.get('version')==VERSION,'Matching snapshot required');summary=movie_summary(manifest,movie,snapshot)
    hp=Path(target_html);old=hp.read_bytes();require(sha(hp)==expected_html_sha256,'HTML changed; preserve concurrent work')
    esc=html.escape;data=base64.b64encode(Path(movie).read_bytes()).decode('ascii')
    chapters=''.join(f'<li>{esc(c["label"])}: {c["start_s"]:.1f}–{c["end_s"]:.1f}s · {"No flight" if c["no_flight"] else "Capture" if c["capture"] else "Miss / rejected route"}</li>' for c in summary['clips'])
    block=VIDEO_START+f'\n<article class="ytg-movie"><video controls playsinline preload="none" aria-label="Far shot tilt and drop-gap recorded comparison"><source src="data:video/mp4;base64,{data}" type="video/mp4"></video><ol>{chapters}</ol><p class="ytg-small">Actual saved native samples, held endpoint joints, no new physics. Reference and baseline repeat zero, then the first up to three grid-ordered selected recipes at fresh repeat zero, regardless of repeat outcome. MP4 SHA-256 {esc(summary["output_sha256"])}</p></article>\n'+VIDEO_END
    original=old.decode();placeholder='<div data-yaw-tilt-gap="movies"></div>'
    if VIDEO_START in original:
        require(original.index(START)<original.index(VIDEO_START)<original.index(VIDEO_END)<original.index(END),'Own movie markers required');new=replace_block(original,VIDEO_START,VIDEO_END,block)
    else:
        require(original.count(placeholder)==1 and original.index(START)<original.index(placeholder)<original.index(END),'Own unique movie placeholder required');new=original.replace(placeholder,block)
    atomic_bytes(hp,new.encode(),old)
    return dict(version='sircus-yaw-tilt-gap-movie-publication-v1',video=summary,html_sha256=sha(hp),historical_content_preserved=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('study','audit','pipeline-state','snapshot','target-html','target-css'):p.add_argument('--'+name,type=Path,required=name in ('study','snapshot','target-html','target-css'))
    p.add_argument('--expected-html-sha256',required=True);p.add_argument('--expected-css-sha256',required=True)
    a=p.parse_args();out=refresh(a.study,snapshot_path=a.snapshot,target_html=a.target_html,target_css=a.target_css,audit_path=a.audit,pipeline_path=a.pipeline_state,expected_html_sha256=a.expected_html_sha256,expected_css_sha256=a.expected_css_sha256)
    print(json.dumps(dict(status=out['study']['status'],audit=out['study']['audit']['status'],snapshot_bytes=len(encoded(out).encode()),html_sha256=sha(a.target_html),css_sha256=sha(a.target_css))))

if __name__=='__main__':main()
