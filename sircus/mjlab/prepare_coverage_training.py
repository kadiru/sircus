"""Read-only snapshots for the finite coverage study; never starts simulation.

The installer adds one panel to existing HTML. A bounded watcher updates only
that marked block, a small JSON snapshot and one display key using atomic
compare-and-swap. The running viewer has no small JSON route; its unchanged
/index.html route supplies browser refreshes. No service restart is required.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import html
import json
import math
from pathlib import Path
import re
import time

from .prepare_table_training import atomic_bytes, take, sha
from .prepare_height_tilt_replay import event_list, MAX_VIDEO_BYTES

VERSION = 'sircus-coverage-training7h-v2'
AUDIT = 'sircus-coverage-training7h-audit-v2'
KEY = 'coverage_training7h'
START = '<!--COVERAGE_TRAINING7H_START-->'
END = '<!--COVERAGE_TRAINING7H_END-->'
METHODS = ('current_bc', 'current_ppo', 'bc', 'selected_ppo')
STATS = ('attempts','native_releases','captures','direct_captures','table_assisted_captures',
         'setup_rejections','native_misses','forbidden_contact_shots','robot_contact_shots',
         'clean','mean_reward','invalid','invalid_physics')
LAYOUT = ('context_id','bucket_xy_m','grid_cell','origin','split')
MAX_SNAPSHOT_BYTES = 512*1024
VIDEO = 'sircus-coverage-training7h-recorded-video-v2'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def finite(value):
    return isinstance(value,(int,float)) and not isinstance(value,bool) and math.isfinite(value)


def failure_summary(raw,pipeline):
    """Explain declared failure gates without forwarding potentially private exceptions."""
    gates=raw.get('control_gates',{})
    for key,label in (('pre','Startup'),('post','Final')):
        gate=gates.get(key,{})
        if gate.get('status')=='failed':
            def observed(name):
                value=gate.get(name)
                return str(value) if type(value) is int and 0<=value<=3 else 'unknown'
            note=(f"{label} control gate failed: direct captures {observed('direct_captures')}/3; "
                  f"table-assisted captures {observed('table_assisted_captures')}/3; "
                  f"negative-control captures {observed('negative_captures')}/3 (expected 0).")
            if raw.get('search_requests')==0 and raw.get('train_requests')==0:
                note+=' TRAIN search and PPO did not start.'
            note+=' The saved control evidence is retained; this is not a policy score.'
            return dict(code=key+'_control_gate_failed',message=note)
    if pipeline.get('status') in ('audit_failed','audit_deferred'):
        return dict(code='independent_audit_incomplete',message='Independent audit did not complete successfully; reported results remain unverified and saved.')
    if str(raw.get('error','')).startswith('TimeoutError:') or pipeline.get('status') in ('timeout','timed_out'):
        return dict(code='bounded_deadline',message='The bounded execution deadline stopped the study; partial evidence is saved and no completed comparison is claimed.')
    return dict(code='worker_stopped',message='The worker stopped before a verified comparison; partial evidence is saved. Its detailed error remains in the local log.')


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,allow_nan=False).encode()).hexdigest()


def layouts(study,raw):
    contract=read(study/'contract.json');unsigned=dict(contract);fingerprint=unsigned.pop('fingerprint')
    require(digest(unsigned)==fingerprint==raw['contract_fingerprint'],'Frozen layout contract differs')
    require(raw['protocol']==contract['protocol'],'Report protocol differs from frozen contract')
    splits=read(study/'splits.json')
    require(splits==contract['splits'],'Frozen split file differs')
    low,high=contract['protocol']['bucket_bounds_m'];result={}
    for split in ('train','validation','test'):
        result[split]=[]
        for row in splits[split]:
            item=take(row,('bucket_xy_m','grid_cell','origin','split'));item['context_id']=row['id']
            if 'grid_cell' not in item:
                xy=row['bucket_xy_m']
                if all(low[i]-1e-9<=xy[i]<=high[i]+1e-9 for i in range(2)):
                    item['grid_cell']=[max(0,min(limit-1,int((xy[i]-low[i])/(high[i]-low[i])*limit))) for i,limit in enumerate((7,5))]
            result[split].append(item)
    return result


def video_paths(study):
    directory=Path(study).with_name(Path(study).name+'-video')
    return directory/'replay.json',directory/'replay.mp4'


def video_summary(manifest,movie,snapshot):
    raw=read(manifest);audit=snapshot['audit']
    require(audit.get('status')=='passed' and raw.get('version')==VIDEO and raw.get('status')=='completed','Audited completed coverage video required')
    require(raw.get('native_steps_added')==0 and raw.get('held_arm_poses') is True and raw.get('sampled_ball_centers_only') is True,'Saved held-endpoint video required')
    expected=dict(report_sha256=snapshot['report_sha256'],audit_sha256=audit['proof_sha256'],contract_fingerprint=snapshot['contract_fingerprint'])
    require(all(raw.get('source',{}).get(k)==v for k,v in expected.items()),'Movie source differs from audited study')
    size=Path(movie).stat().st_size
    require(0<size<=MAX_VIDEO_BYTES and raw.get('video_bytes')==size and raw.get('output_sha256')==sha(movie),'Movie checksum or size differs')
    require(Path(movie).read_bytes()[4:8]==b'ftyp','MP4 movie required')
    frames=raw.get('frames');fps=raw.get('fps');duration=raw.get('duration_s')
    require(isinstance(frames,list) and frames and digest(frames)==raw.get('frames_sha256')
        and finite(fps) and 0<fps<=60 and finite(duration) and 0<duration<=180
        and abs(len(frames)/fps-duration)<1/fps,'Finite authenticated movie timeline required')
    clips=raw.get('clips')
    require(isinstance(clips,list) and 1<=len(clips)<=5,'One to five actual recorded clips required')
    clean=[];previous=0
    for clip in clips:
        require(isinstance(clip.get('record_id'),str) and isinstance(clip.get('label'),str) and type(clip.get('capture')) is bool,'Explicit recorded clip outcome required')
        require(finite(clip.get('start_s')) and finite(clip.get('end_s')) and previous<=clip['start_s']<clip['end_s']<=duration,'Ordered recorded clips required')
        row=take(clip,('record_id','context_id','label','start_s','end_s','capture','values','bucket_xy_m','record_sha256','native_cache_sha256'))
        row['events']=event_list(clip.get('events',{}));clean.append(row);previous=clip['end_s']
    return dict(version=VIDEO,status='ready',source=expected,manifest_sha256=sha(manifest),
        output_sha256=raw['output_sha256'],video_bytes=size,duration_s=duration,fps=fps,
        frame_count=len(frames),frames_sha256=raw['frames_sha256'],clips=clean)


def model(raw):
    result = take(raw,('method','seed','samples','train_requests','updates','step','sha256','artifact_sha256','checkpoint_sha256','selection_rule'))
    result['validation'] = take(raw.get('validation'),STATS) if isinstance(raw,dict) else {}
    return result


def proof(report, raw, audit_path):
    result = dict(status='pending',full_checks_verified=False,native_checks_verified=False)
    if not audit_path.is_file():
        return result
    audit = read(audit_path);checks = audit.get('checks')
    passed = (raw.get('status')=='completed' and raw.get('phase')=='completed' and audit.get('version')==AUDIT
        and audit.get('status')=='passed' and audit.get('full_checks_verified') is True
        and audit.get('native_checks_verified') is True
        and re.fullmatch('[a-f0-9]{64}',raw.get('contract_fingerprint','')) is not None
        and audit.get('contract_fingerprint')==raw.get('contract_fingerprint')
        and audit.get('report_sha256')==sha(report) and not audit.get('failures')
        and isinstance(checks,list) and bool(checks) and audit.get('checks_count')==len(checks)
        and all(isinstance(c,dict) and c.get('passed') is True for c in checks))
    return dict(version=AUDIT,status='passed' if passed else 'unverified',full_checks_verified=passed,
        native_checks_verified=passed,checks_count=audit.get('checks_count'),report_sha256=sha(report),
        proof_sha256=sha(audit_path),contract_fingerprint=raw.get('contract_fingerprint'))


def compact(study, audit_path=None, pipeline=None):
    """Whitelist only display facts. Native arrays, local paths and errors stay private."""
    study=Path(study);path=study/'report.json';pipeline=pipeline or {}
    if not path.is_file():
        pending=dict(version=VERSION,status=pipeline.get('status','preparing'),phase='Waiting for input verification / GPU slot',
                    counts={},stage_counts=[],layouts={},curves=[],test_comparison=[],test_positions=[],
                    clock=dict(started_unix_s=pipeline.get('started_unix_s'),deadline_unix_s=pipeline.get('global_deadline_unix_s'),limit_s=pipeline.get('global_limit_s')),
                    video_status='waiting_for_audit',
                    audit=dict(status='pending',full_checks_verified=False,native_checks_verified=False))
        if pipeline.get('status') in ('failed','audit_failed','audit_deferred','timeout','timed_out'):
            pending.update(status='stopped',failure=failure_summary({},pipeline))
        return pending
    raw=read(path)
    require(raw.get('version')==VERSION,'Explicit coverage training study required')
    out=take(raw,('version','status','phase','scope','contract_fingerprint','train_requests','updates','elapsed_s',
        'search_requests','search_completed_rounds','search_partial_round_requests','training_budget_complete','inference_scope'))
    out['report_sha256']=sha(path)
    out['counts']=take(raw.get('counts'),STATS)
    out['clock']=dict(started_unix_s=pipeline.get('started_unix_s'),deadline_unix_s=pipeline.get('global_deadline_unix_s'),
                      limit_s=pipeline.get('global_limit_s'))
    budget_path=study.with_name(study.name+'-budget.json')
    if budget_path.is_file():
        budget=read(budget_path)
        effective=budget.get('effective_worker_limit_s')
        out['clock'].update(worker_acquired_unix_s=budget.get('acquired_unix_s'),
            worker_limit_s=effective if finite(effective) else budget.get('limit_s'),
            worker_configured_limit_s=budget.get('limit_s'))
    out['protocol']=take(raw.get('protocol'),('substeps','train_layouts','historical_train_layouts','new_train_layouts',
        'validation_layouts','test_layouts','grid_rows','grid_columns','bucket_bounds_m','maximum_fresh_requests',
        'global_limit_s','seeds','bc_seed','noise','search_requests','train_requests','train_requests_per_seed',
        'validation_requests','test_requests','control_requests','updates_per_seed','milestones','common_prior'))
    maximum={'controls_pre':9,'search':out['protocol'].get('search_requests'),'train':out['protocol'].get('train_requests'),
             'validation':out['protocol'].get('validation_requests'),'test':out['protocol'].get('test_requests'),'controls_post':9}
    out['stage_counts']=[dict(stage=stage,maximum_requests=maximum[stage],counts=take(raw.get('stage_counts',{}).get(stage),STATS))
                         for stage in maximum]
    out['layouts']=layouts(study,raw)
    out['bootstrap_summary']=take(raw.get('bootstrap_summary'),('historical_requests','new_requests','requests','native_releases',
        'captures','teacher_count','table_assisted_teachers','successful_layouts','historical_successful_layouts',
        'new_successful_layouts','searched_layouts','setup_rejections','native_misses','layouts_requested'))
    out['bootstrap_coverage']=[dict(take(r,LAYOUT+('has_teacher',)),counts=take(r.get('counts'),STATS)) for r in raw.get('search_coverage',[])]
    out['curves']=[dict(take(r,('method','seed','samples','updates')),validation=take(r.get('validation'),STATS),artifact=model(r.get('artifact',{}))) for r in raw.get('curves',[])]
    out['per_seed']={seed:take(value,('samples','updates')) for seed,value in raw.get('per_seed',{}).items() if str(seed) in [str(s) for s in out['protocol'].get('seeds',[])]}
    out['selected']=None if raw.get('selected') is None else model(raw['selected'])
    if out['selected'] is not None:
        matching=[r for r in raw.get('curves',[]) if r.get('artifact',{}).get('sha256')==raw['selected'].get('sha256')]
        require(len(matching)==1,'Selected checkpoint needs its unique validation milestone')
        out['selected'].update(method=matching[0]['method'],validation=take(matching[0]['validation'],STATS))
    out['budget_reasons']=[take(r,('stage','reason','elapsed_s','samples_per_seed','requests')) for r in raw.get('budget_reasons',[])]
    out['gate']=take(raw.get('gate'),('status','controls_passed','complete_test','full_training_budget','substeps'))
    out['test_comparison']=[dict(method=r.get('method'),counts=take(r.get('counts'),STATS),
                                 artifact_sha256=r.get('artifact_sha256')) for r in raw.get('test_comparison',[]) if r.get('method') in METHODS]
    out['test_positions']=[take(r,LAYOUT+('method','capture','table_assisted_capture','direct_capture','native_release',
        'robot_contact','robot_contact_steps','invalid_physics','values','record_id','clean'))
        for r in raw.get('test_positions',[]) if r.get('method') in METHODS]
    out['control_gates']={k:dict(take(v,('status','positive_captures','negative_captures','direct_captures','table_assisted_captures')),
                                counts=take(v.get('counts'),STATS))
                          for k,v in raw.get('control_gates',{}).items() if k in ('pre','post')}
    audit_path=Path(audit_path) if audit_path else study.with_name(study.name+'-audit.json')
    out['audit']=proof(path,raw,audit_path)
    manifest,movie=video_paths(study)
    out['video_status']='waiting_for_audit' if out['audit']['status']!='passed' else 'waiting_for_recorded_video'
    if out['audit']['status']=='passed' and manifest.is_file() and movie.is_file():
        out['video']=video_summary(manifest,movie,out);out['video_status']='ready'
    if raw.get('status') in ('failed','timeout','timed_out','stopped') or pipeline.get('status') in ('failed','audit_failed','audit_deferred','timeout','timed_out'):
        out['status']='stopped'
        out['failure']=failure_summary(raw,pipeline)
    require(len(json.dumps(out,allow_nan=False).encode())<=MAX_SNAPSHOT_BYTES,'Compact snapshot exceeds bounded sidecar size')
    return out


def guarded_targets(study, targets, inputs=()):
    root=Path(study).resolve();sources={Path(p).resolve() for p in inputs if p is not None}
    destinations=[Path(p).resolve() for p in targets if p is not None]
    require(len(destinations)==len(set(destinations)),'Publication destinations must be distinct')
    for target in destinations:
        require(not target.is_relative_to(root) and target not in sources,'Publication must stay outside immutable study/input evidence')


def publish_display(target,snapshot):
    path=Path(target)
    require(path.is_file(),'Existing display report required')
    for _ in range(3):
        old=path.read_bytes();raw=json.loads(old);raw[KEY]=snapshot
        try:
            atomic_bytes(path,(json.dumps(raw,indent=2,allow_nan=False)+'\n').encode(),old)
            return
        except RuntimeError:
            continue
    raise RuntimeError('Concurrent display writer preserved')


def publish_sidecar(target,snapshot):
    path=Path(target)
    old=path.read_bytes() if path.exists() else None
    require(old is None or json.loads(old).get('version')==VERSION,'Existing sidecar belongs to another task')
    data=(json.dumps(snapshot,indent=2,allow_nan=False)+'\n').encode()
    require(len(data)<=MAX_SNAPSHOT_BYTES,'Compact snapshot exceeds bounded sidecar size')
    atomic_bytes(path,data,old)


def embed_html(original,snapshot,movie_bytes=None):
    require(original.count(START)==original.count(END) and original.count(START)<=1,'Ambiguous own publication markers')
    web=Path(__file__).with_name('web')
    template=(web/'coverage-training.html').read_text();css=(web/'coverage-training.css').read_text();js=(web/'coverage-training.js').read_text()
    js=re.sub(r'^export (?=(?:const|function|class)\b)','',js,flags=re.M)
    require(not re.search(r'^\s*(?:import|export)\b',js,re.M) and '</script' not in js.lower() and '</style' not in css.lower(),'Self-contained panel assets required')
    data=json.dumps(snapshot,allow_nan=False).replace('<','\\u003c').replace('>','\\u003e').replace('&','\\u0026')
    if snapshot.get('video'):
        require(movie_bytes is not None and hashlib.sha256(movie_bytes).hexdigest()==snapshot['video']['output_sha256'],'Verified movie bytes required for embedding')
        encoded=base64.b64encode(movie_bytes).decode('ascii')
        media=f'<div data-coverage-training="video"><video data-coverage-video-sha="{html.escape(snapshot["video"]["output_sha256"])}" controls playsinline preload="metadata"><source src="data:video/mp4;base64,{encoded}" type="video/mp4"></video></div>'
        template=template.replace('<div data-coverage-training="video"></div>',media)
    block=f'{START}\n<style>{css}</style>\n{template}\n<script type="application/json" id="coverage-training-snapshot">{data}</script>\n<script>(function(){{\n{js}\n}})();</script>\n{END}'
    if START in original:
        a,b=original.index(START),original.index(END)+len(END)
        require(b>a,'Own markers out of order')
        return original[:a]+block+original[b:]
    closing=list(re.finditer(r'</body\s*>',original,re.I))
    require(bool(closing),'Existing document body required')
    at=closing[-1].start()
    return original[:at]+block+'\n'+original[at:]


def install(target,snapshot,expected_sha256,movie_bytes=None):
    path=Path(target);old=path.read_bytes()
    require(hashlib.sha256(old).hexdigest()==expected_sha256,'Target HTML changed; preserving existing work')
    atomic_bytes(path,embed_html(old.decode(),snapshot,movie_bytes).encode(),old)


def refresh(study, *, snapshot_path, target_html, display=None, audit_path=None, pipeline_path=None, now=None):
    guarded_targets(study,(snapshot_path,target_html,display),(audit_path,pipeline_path))
    state=read(pipeline_path) if pipeline_path and Path(pipeline_path).is_file() else {'status':'starting'}
    value=compact(study,audit_path,pipeline=state)
    timestamp=time.time() if now is None else now
    require(finite(timestamp),'Finite publication time required')
    value['publication']=dict(updated_unix_s=timestamp,transport='embedded_html',refresh_seconds=30,
        pipeline_status=state.get('status'),pipeline_finished_unix_s=state.get('finished_unix_s'))
    publish_sidecar(snapshot_path,value)
    if display:publish_display(display,value)
    movie_bytes=video_paths(study)[1].read_bytes() if value.get('video') else None
    install(target_html,value,sha(target_html),movie_bytes)
    return value,state


def watch(study, *, snapshot_path, target_html, receipt, display=None, audit_path=None,
          pipeline_path=None, max_seconds=28800, monotonic=time.monotonic, sleep=time.sleep, now=time.time):
    """Finite observer only. Worker owns and enforces its independent seven-hour limit."""
    require(finite(max_seconds) and 1<=max_seconds<=28800,'Finite publication duration at most eight hours required')
    guarded_targets(study,(snapshot_path,target_html,display,receipt),(audit_path,pipeline_path))
    started=monotonic();updates=0;errors=0;status='watching'
    while True:
        try:
            value,state=refresh(study,snapshot_path=snapshot_path,target_html=target_html,display=display,
                                audit_path=audit_path,pipeline_path=pipeline_path,now=now())
            updates+=1
            terminal=state.get('status') in ('completed','failed','audit_failed','audit_deferred','timeout','timed_out','stopped')
            if (not pipeline_path and value.get('audit',{}).get('status')=='passed') or (terminal and finite(state.get('finished_unix_s'))):
                status='finished'
        except (OSError,ValueError,RuntimeError,json.JSONDecodeError):
            errors+=1;state={};value={}
        elapsed=monotonic()-started
        if status=='watching' and elapsed>=max_seconds:status='publication_timeout'
        record=dict(version='sircus-coverage-training7h-publication-watch-v2',status=status,updates=updates,read_or_publication_errors=errors,elapsed_s=elapsed,
                    pipeline_status=state.get('status'),audit_status=value.get('audit',{}).get('status'),
                    source_sha256=sha(__file__))
        atomic_bytes(receipt,(json.dumps(record,indent=2,allow_nan=False)+'\n').encode())
        if status!='watching':return record
        sleep(min(30.,max_seconds-elapsed))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('study','audit','pipeline-state','snapshot','display','target-html','receipt'):
        p.add_argument('--'+name,type=Path,required=name in ('study','snapshot','target-html'))
    p.add_argument('--expected-html-sha256',required=True)
    p.add_argument('--watch',action='store_true');p.add_argument('--max-seconds',type=float,default=28800)
    a=p.parse_args()
    guarded_targets(a.study,(a.snapshot,a.display,a.target_html,a.receipt),(a.audit,a.pipeline_state))
    require(not a.watch or a.receipt is not None,'Bounded watcher receipt required')
    require(sha(a.target_html)==a.expected_html_sha256,'Target HTML changed; preserving existing work')
    value,_=refresh(a.study,snapshot_path=a.snapshot,target_html=a.target_html,display=a.display,
                    audit_path=a.audit,pipeline_path=a.pipeline_state)
    if a.watch:
        result=watch(a.study,snapshot_path=a.snapshot,target_html=a.target_html,receipt=a.receipt,display=a.display,
                     audit_path=a.audit,pipeline_path=a.pipeline_state,max_seconds=a.max_seconds)
    else:result={key:value.get(key) for key in ('status','phase','audit')}
    print(json.dumps(result))


if __name__=='__main__':main()
