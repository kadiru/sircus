"""Publish bounded saved regression evidence without solver calls or a restart.

The existing viewer has a fixed static allowlist. A separately marked CSS comment
carries the compact JSON update; this temporary data channel adds no robot/API
endpoint. Everything outside this task's HTML/CSS markers is preserved byte for
byte. The original study, audit and diagnostic inputs are read-only.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import re
import time

from .prepare_coverage_recovery import checked_file, digest, read, require, sha, summarize_flight
from .prepare_table_training import atomic_bytes, take

VERSION='sircus-regression-ik-display-v1'
DIAGNOSIS='sircus-coverage-regression-diagnosis-v1'
STUDY='sircus-regression-ik-feasibility-v1'
AUDIT='sircus-regression-ik-feasibility-audit-v1'
START='<!--REGRESSION_IK_START-->'
END='<!--REGRESSION_IK_END-->'
CSS_START='/*SIRCUS_REGRESSION_IK_SNAPSHOT_START\n'
CSS_END='\nSIRCUS_REGRESSION_IK_SNAPSHOT_END*/'
MAX_SNAPSHOT_BYTES=256*1024
STATS=('attempts','captures','native_releases','setup_rejections','native_misses','invalid_physics','robot_contact_shots','direct_captures','table_assisted_captures','cached_refusal_requests')
ROW_FIELDS=('record_id','id','record_sha256','context_id','bucket_xy_m','values','native_release','capture','route_success','invalid','reason','invalid_physics','whole_ball_entry','max_retention_s','forbidden_contact','forbidden_contact_steps','table_contact','table_contact_steps','robot_contact_steps','board_bounce','variant','repeat','admission_mode','source_record_id')


def compact_row(row):
    out=take(row,ROW_FIELDS)
    if row.get('realization_comparison'):
        out['realization_comparison']=take(row['realization_comparison'],('basis','reference_record_id','source_record_id','q10_equal_to_source','q14_equal_to_source','q10_max_abs_delta_rad','q14_max_abs_delta_rad','board_center_delta_m','board_normal_angle_deg'))
    # Already-authenticated observations only; never synthesize a missing trace.
    first_events={}
    for event in row.get('events',[]):
        first_events.setdefault(event['category'],event)
    out['events']=[take(e,('category','time_s','ball_center_m')) for e in first_events.values()]
    if row.get('first_flight'):
        out['first_flight']=take(row['first_flight'],('available','forward_error_m','lateral_error_m','cutoff_contact_channels','rebound_position_m','rebound_velocity_m_s','end_position_m'))
    trace=row.get('trace')
    if trace:
        require(row.get('native_release') is True,'Unflown setup cannot carry a flight replay')
        times=trace.get('time_s',[]);positions=trace.get('position_m',[])
        require(len(times)==len(positions) and 1<len(times)<=512,'Bounded matching native trace required')
        require(all(isinstance(t,(int,float)) and (i==0 or t>times[i-1]) for i,t in enumerate(times)),'Ordered native trace required')
        require(all(isinstance(p,list) and len(p)==3 for p in positions),'Three-coordinate native samples required')
        indices={round(i*(len(times)-1)/min(40,len(times)-1)) for i in range(min(41,len(times)))}
        for event in out['events']:
            if isinstance(event.get('time_s'),(int,float)):
                near=min(range(len(times)),key=lambda i:abs(times[i]-event['time_s']))
                indices.update((max(0,near-1),near,min(len(times)-1,near+1)))
        for event in out['events']:
            if event.get('time_s') in times:event['ball_center_m']=positions[times.index(event['time_s'])]
        indices=sorted(indices)
        out['trace']=take(trace,('native_cache_sha256','record_id'))
        out['trace'].update(time_s=[times[i] for i in indices],position_m=[positions[i] for i in indices],
            sample_basis='Subset of authenticated native samples, including endpoints and first-event neighbors; no interpolation')
    setup=row.get('setup_proof',row.get('setup'))
    if setup:
        out['setup_proof']=take(setup,('valid_ik_branches','maximum_tested_minimum_clearance_m','required_clearance_m','connection_sha256','resolution_sha256','reason','layout_eligible','anchor_admitted','endpoint_pair_attempts','admitted_branches','home_minimum_clearance_m','home_closest_pair','proof_minimum_clearance_m'))
        out['setup_proof']['endpoint_examples']=[take(p,('reason','minimum_clearance_m','closest_pair','board_position_error_m','dropper_position_error_m','board_normal_error_deg','pose_rejected')) for p in (setup.get('endpoint_examples',[])[:4] if row.get('native_release') is False else [])]
    return out


CASES=('recovery-test-r1-c1-n1','recovery-test-r2-c4-n0')
VARIANTS=('parent_exact','recovery_exact','parent_xy','parent_height_gap','parent_tilt','blend25','blend50','blend75')


def checked_diagnosis(path):
    from .prepare_recovery_regression import checked_diagnosis as original
    d=original(path)
    d['losses']=[row for row in d['losses'] if row['context_id'] in CASES]
    require(tuple(row['context_id'] for row in d['losses'])==CASES,'Exact two selected IK losses required')
    require(all(row['recovery_bc'].get('native_release') is False for row in d['losses']),'Both recovery losses must be unflown setup declines')
    d.pop('setup_rejections',None);d.pop('flight_misses',None)
    return d


def audit_proof(path,raw,audit_path):
    out=dict(status='pending',full_checks_verified=False,native_checks_verified=False)
    if audit_path is None or not Path(audit_path).is_file():return out
    a=read(audit_path);checks=a.get('checks',[])
    manifest_path=Path(path).parent/'evidence-manifest.json'
    manifest=read(manifest_path) if manifest_path.is_file() else None
    passed=(manifest is not None and raw.get('evidence_sha256')==manifest and a.get('evidence_manifest_sha256')==sha(manifest_path)
        and raw.get('status')=='completed' and a.get('version')==AUDIT and a.get('status')=='passed'
        and a.get('full_checks_verified') is True and a.get('native_checks_verified') is True and a.get('failures')==[]
        and bool(checks) and a.get('checks_count')==len(checks) and all(c.get('passed') is True for c in checks)
        and a.get('report_sha256')==sha(path) and a.get('contract_fingerprint')==raw.get('contract_fingerprint'))
    return dict(status='passed' if passed else 'unverified',full_checks_verified=passed,native_checks_verified=passed,
        report_sha256=sha(path),contract_fingerprint=raw.get('contract_fingerprint'),proof_sha256=sha(audit_path),checks_count=a.get('checks_count'))


_RECORDINGS_CACHE={}


def comparison_recordings(study,raw):
    """First saved repeat of each variant, only after the bound audit passed."""
    identity=(str(Path(study).resolve()),sha(Path(study)/'report.json'))
    if identity in _RECORDINGS_CACHE:return _RECORDINGS_CACHE[identity]
    manifest=raw['evidence_sha256'];rows_path=checked_file(study,'rows.jsonl',manifest['rows.jsonl'])
    rows=[json.loads(line) for line in rows_path.read_text().splitlines() if line.strip()]
    selected=[r for r in rows if r.get('phase')=='diagnostic' and r.get('repeat')==0]
    require(len(selected)==16 and {(r['context_id'],r['variant']) for r in selected}=={(c,v) for c in CASES for v in VARIANTS},'Exactly sixteen first-repeat IK comparison records required')
    groups={};out=[]
    for row in selected:
        item=compact_row(row);item['capture']=row.get('route_success') is True
        if row.get('native_release') is True:
            require(row.get('adapter_directory')=='native','Exact diagnostic native namespace required')
            groups.setdefault(row['native_cache_file'],[]).append((row,item))
        out.append(item)
    import numpy as np
    for name,members in groups.items():
        path=checked_file(study,'native/'+name,manifest['native/'+name])
        with np.load(path,allow_pickle=False) as archive:
            times=np.asarray(archive['time_s'],dtype=float);positions=archive['position_m'];forces=archive['forces_N'];ids=archive['record_ids']
            for row,item in members:
                column=row['native_cache_row']
                require(str(ids[column])==row['id'] and row['native_cache_sha256']==manifest['native/'+name],'Exact correction recording identity required')
                trace=summarize_flight(row,times,positions[:,column],forces[:,column])
                item.update(compact_row(dict(row,**trace)))
    _RECORDINGS_CACHE.clear();_RECORDINGS_CACHE[identity]=out
    return out


def compact(study,diagnosis_path,*,audit_path=None,pipeline_path=None,now=None):
    diagnosis=checked_diagnosis(diagnosis_path);study=Path(study);path=study/'report.json'
    state=read(pipeline_path) if pipeline_path and Path(pipeline_path).is_file() else {}
    correction=dict(status='pending',phase='awaiting frozen rule',counts={},cases=[],audit=dict(status='pending'))
    if path.is_file():
        raw=read(path);require(raw.get('version')==STUDY,'Explicit correction study required')
        contract=read(study/'contract.json');unsigned=dict(contract);fp=unsigned.pop('fingerprint')
        require(digest(unsigned)==fp==raw.get('contract_fingerprint'),'Correction contract differs')
        require(contract.get('protocol',{}).get('substeps')==64,'Fixed64 comparison required')
        require(raw.get('expected_requests')==66,'Exact bounded 66-slot protocol required')
        rule=raw.get('rule',{});require(rule.get('diagnosis_sha256')==sha(diagnosis_path),'Correction must bind this diagnosis')
        require(rule.get('mode')=='matched_parent_coordinate_blocks_and_interpolation' and tuple(rule.get('contexts',[]))==CASES and tuple(contract.get('protocol',{}).get('variants',[]))==VARIANTS,'Exact two-case eight-variant IK plan required')
        correction=take(raw,('version','status','phase','contract_fingerprint','expected_requests','elapsed_s','training','posthoc_development'))
        correction['rule']=take(rule,('mode','description','contexts','coordinate_groups','blend_fractions','diagnosis_sha256'))
        correction['report_sha256']=sha(path);correction['counts']=take(raw.get('counts'),STATS)
        correction['counts_by_variant']={k:take(v,STATS) for k,v in raw.get('counts_by_variant',{}).items() if k in VARIANTS}
        correction['cases']=[dict(**take(c,('context_id','bucket_xy_m','parent_values','recovery_values','variant_values')),
            variants={k:take(v,STATS) for k,v in c.get('variants',{}).items() if k in VARIANTS}) for c in raw.get('cases',[])]
        correction['control_gates']={k:take(v,('status','attempts','direct_captures','table_assisted_captures','negative_captures')) for k,v in raw.get('control_gates',{}).items() if k in ('pre','post')}
        correction['audit']=audit_proof(path,raw,audit_path)
        if correction['audit']['status']=='passed':correction['recordings']=comparison_recordings(study,raw)
    if state.get('status') in ('failed','audit_failed','audit_deferred','timeout','timed_out','stopped'):correction['status']=state['status']
    terminal=state.get('status') in ('completed','failed','audit_failed','audit_deferred','timeout','timed_out','stopped') and isinstance(state.get('finished_unix_s'),(int,float))
    out=dict(version=VERSION,diagnosis=diagnosis,correction=correction,publication=dict(updated_unix_s=time.time() if now is None else now,terminal=terminal,pipeline_status=state.get('status')))
    require(len(json.dumps(out,allow_nan=False,separators=(',',':')).encode())<=MAX_SNAPSHOT_BYTES,'Compact regression snapshot exceeds 256 KiB')
    return out


def baseline_summary(snapshot):
    """Compact audit-bound numbers for Git; omit raw samples and runtime paths."""
    require(snapshot.get('version')==VERSION,'Matching snapshot required')
    r=snapshot['correction'];proof=r.get('audit',{})
    require(r.get('status')=='completed' and proof.get('status')=='passed' and proof.get('full_checks_verified') is True and proof.get('native_checks_verified') is True,'Completed independently verified comparison required')
    require(r.get('expected_requests')==66 and r.get('counts',{}).get('attempts')==66,'All 66 requested slots required')
    require(proof.get('report_sha256')==r.get('report_sha256') and proof.get('contract_fingerprint')==r.get('contract_fingerprint') and all(r.get('control_gates',{}).get(x,{}).get('status')=='passed' for x in ('pre','post')),'Bound native audit and both controls required')
    out=take(r,('version','status','contract_fingerprint','report_sha256','expected_requests','elapsed_s','training','posthoc_development','rule','counts','counts_by_variant','cases','control_gates','audit'))
    out['scope']='Two selected development positions, fixed 64 substeps, eight frozen variants and three nominal requests each; no new policy score, noise qualification or policy activation.'
    out['first_repeat_realizations']=[take(row,('id','context_id','variant','repeat','native_release','capture','realization_comparison')) for row in r.get('recordings',[])]
    out['first_repeat_admissions']=[dict(take(row,('id','context_id','variant','repeat','native_release','invalid','reason','admission_mode')),setup_proof=take(row.get('setup_proof'),('valid_ik_branches','reason','endpoint_pair_attempts','maximum_tested_minimum_clearance_m','required_clearance_m'))) for row in r.get('recordings',[])]
    out['historical_source']=take(snapshot['diagnosis']['source'],('report_sha256','audit_sha256','rows_sha256','contract_fingerprint'))
    return out


def replace_block(original,start,end,block,*,append=False):
    require(original.count(start)==original.count(end) and original.count(start)<=1,'Ambiguous regression publication markers')
    if start in original:
        a,b=original.index(start),original.index(end)+len(end);require(b>a,'Reversed publication markers')
        return original[:a]+block+original[b:]
    if append:return original+block
    closing=list(re.finditer(r'</body\s*>',original,re.I));require(closing,'Existing HTML body required')
    at=closing[-1].start();return original[:at]+block+original[at:]


def encoded(snapshot):
    text=json.dumps(snapshot,allow_nan=False,separators=(',',':'))
    require(len(text.encode())<=MAX_SNAPSHOT_BYTES,'Compact regression snapshot exceeds 256 KiB')
    return text.replace('<','\\u003c').replace('>','\\u003e').replace('&','\\u0026').replace('*','\\u002a')


def embed_css(original,snapshot):
    return replace_block(original,CSS_START,CSS_END,CSS_START+encoded(snapshot)+CSS_END,append=True)


def embed_html(original,snapshot):
    web=Path(__file__).with_name('web');html=(web/'regression-ik.html').read_text();css=(web/'regression-ik.css').read_text();js=(web/'regression-ik.js').read_text()
    js=re.sub(r'^export (?=(?:const|function|class)\b)','',js,flags=re.M)
    require(not re.search(r'^\s*(?:import|export)\b',js,re.M) and '</script' not in js.lower() and '</style' not in css.lower(),'Self-contained panel required')
    block=f'{START}\n<style>{css}</style>\n{html}\n<script type="application/json" id="regression-ik-snapshot">{encoded(snapshot)}</script>\n<script>(function(){{\n{js}\n}})();</script>\n{END}'
    return replace_block(original,START,END,block)


def guard_targets(study,inputs,targets):
    roots=[Path(study).resolve()];sources={Path(p).resolve() for p in inputs if p is not None};dest=[Path(p).resolve() for p in targets if p is not None]
    require(len(dest)==len(set(dest)),'Publication targets must be distinct')
    require(all(p not in sources and not any(p.is_relative_to(r) for r in roots) for p in dest),'Keep publication outside immutable evidence')


def refresh(study,diagnosis_path,*,snapshot_path,target_html,target_css,audit_path=None,pipeline_path=None,expected_html_sha256=None,expected_css_sha256=None,now=None):
    guard_targets(study,(diagnosis_path,audit_path,pipeline_path),(snapshot_path,target_html,target_css))
    html=Path(target_html);css=Path(target_css);snapshot=Path(snapshot_path)
    html_bytes=html.read_bytes();css_bytes=css.read_bytes()
    if expected_html_sha256 is not None:require(sha(html)==expected_html_sha256,'Target HTML changed; preserving existing work')
    if expected_css_sha256 is not None:require(sha(css)==expected_css_sha256,'Target CSS changed; preserving existing work')
    prior=snapshot.read_bytes() if snapshot.exists() else None
    require(prior is None or json.loads(prior).get('version')==VERSION,'Snapshot belongs to another study')
    out=compact(study,diagnosis_path,audit_path=audit_path,pipeline_path=pipeline_path,now=now)
    new_html=embed_html(html_bytes.decode(),out).encode();new_css=embed_css(css_bytes.decode(),out).encode()
    # Each mutation compares current bytes immediately before replacement.
    atomic_bytes(snapshot,(encoded(out)+'\n').encode(),prior)
    atomic_bytes(css,new_css,css_bytes)
    atomic_bytes(html,new_html,html_bytes)
    return out


def watch(study,diagnosis_path,*,receipt,max_seconds=3600,monotonic=time.monotonic,sleep=time.sleep,**options):
    require(0<max_seconds<=7200,'Finite publication watch at most two hours required')
    guard_targets(study,(diagnosis_path,options.get('audit_path'),options.get('pipeline_path')),
        (receipt,options['snapshot_path'],options['target_html'],options['target_css']))
    start=monotonic();updates=0;errors=0;status='watching';value={}
    while True:
        try:
            value=refresh(study,diagnosis_path,**options);updates+=1
            # Initial explicit hashes apply only to the first mutation; every
            # subsequent refresh still uses immediate byte-compare replacement.
            options.pop('expected_html_sha256',None);options.pop('expected_css_sha256',None)
            if value['publication']['terminal'] and (value['correction']['audit']['status']=='passed'
                or value['correction']['status'] in ('failed','audit_failed','audit_deferred','timeout','timed_out','stopped')):status='finished'
        except (OSError,ValueError,RuntimeError,json.JSONDecodeError):
            errors+=1
            if updates==0:raise
        elapsed=monotonic()-start
        if status=='watching' and elapsed>=max_seconds:status='publication_timeout'
        record=dict(version='sircus-regression-ik-publication-v1',status=status,updates=updates,errors=errors,elapsed_s=elapsed,
            correction_status=value.get('correction',{}).get('status'),audit_status=value.get('correction',{}).get('audit',{}).get('status'))
        atomic_bytes(receipt,(json.dumps(record,indent=2)+'\n').encode())
        if status!='watching':return record
        sleep(min(30,max_seconds-elapsed))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('study','diagnosis','audit','pipeline-state','snapshot','target-html','target-css','receipt'):
        p.add_argument('--'+name,type=Path,required=name in ('study','diagnosis','snapshot','target-html','target-css'))
    p.add_argument('--expected-html-sha256',required=True);p.add_argument('--expected-css-sha256',required=True)
    p.add_argument('--watch',action='store_true');p.add_argument('--max-seconds',type=float,default=3600)
    a=p.parse_args();options=dict(snapshot_path=a.snapshot,target_html=a.target_html,target_css=a.target_css,audit_path=a.audit,pipeline_path=a.pipeline_state,expected_html_sha256=a.expected_html_sha256,expected_css_sha256=a.expected_css_sha256)
    require(not a.watch or a.receipt is not None,'Watcher receipt required')
    if a.watch:result=watch(a.study,a.diagnosis,receipt=a.receipt,max_seconds=a.max_seconds,**options)
    else:
        out=refresh(a.study,a.diagnosis,**options)
        result=dict(status=out['correction']['status'],audit=out['correction']['audit']['status'],snapshot_bytes=len(encoded(out).encode()),html_sha256=sha(a.target_html),css_sha256=sha(a.target_css))
    print(json.dumps(result))

if __name__=='__main__':main()
