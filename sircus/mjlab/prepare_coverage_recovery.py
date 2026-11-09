"""Read-only recovery diagnostics/publication; never launches physics or controls services."""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
import time

from .prepare_table_training import atomic_bytes, take

VERSION='sircus-coverage-recovery-v1'
AUDIT='sircus-coverage-recovery-audit-v1'
PARENT='sircus-coverage-training7h-v2'
PARENT_AUDIT='sircus-coverage-training7h-audit-v2'
DIAGNOSIS='sircus-coverage-recovery-parent-diagnosis-v1'
KEY='coverage_recovery'
START='<!--COVERAGE_RECOVERY_START-->'
END='<!--COVERAGE_RECOVERY_END-->'
CONTACTS=('board','bucket','table','floor','robot','unattributed')
STATS=('attempts','native_releases','captures','direct_captures','table_assisted_captures','setup_rejections','native_misses','forbidden_contact_shots','invalid_physics','clean','mean_reward')
MAX_SNAPSHOT_BYTES=768*1024


def require(condition,message):
    if not condition:raise ValueError(message)


def read(path):return json.loads(Path(path).read_text())
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    return h.hexdigest()
def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,allow_nan=False).encode()).hexdigest()


def checked_file(root,relative,expected):
    require(isinstance(relative,str) and relative and not Path(relative).is_absolute() and '..' not in Path(relative).parts,'Relative evidence file required')
    path=(Path(root)/relative).resolve()
    require(path.is_relative_to(Path(root).resolve()) and path.is_file(),'Missing or escaped evidence file')
    require(sha(path)==expected,'Evidence file hash differs: '+relative)
    return path


def checked_parent(study,audit_path):
    study=Path(study);report=read(study/'report.json');audit=read(audit_path)
    require(report.get('version')==PARENT and report.get('status')=='completed','Completed parent study required')
    require(audit.get('version')==PARENT_AUDIT and audit.get('status')=='passed' and audit.get('full_checks_verified') is True and audit.get('native_checks_verified') is True and not audit.get('failures'),'Passed parent native audit required')
    checks=audit.get('checks',[])
    require(checks and audit.get('checks_count')==len(checks) and all(c.get('passed') is True for c in checks),'Complete passed parent checks required')
    require(audit.get('report_sha256')==sha(study/'report.json'),'Parent report hash differs')
    manifest=read(study/'evidence-manifest.json')
    require(audit.get('evidence_manifest_sha256')==sha(study/'evidence-manifest.json') and report.get('evidence_sha256')==manifest,'Parent evidence manifest differs')
    for name in ('rows.jsonl','contract.json','splits.json'):checked_file(study,name,manifest[name])
    contract=read(study/'contract.json');unsigned=dict(contract);fp=unsigned.pop('fingerprint')
    require(digest(unsigned)==fp==audit['contract_fingerprint']==report['contract_fingerprint'],'Parent contract differs')
    require(contract['protocol']['substeps']==64,'Parent native64 protocol required')
    rows=[json.loads(line) for line in (study/'rows.jsonl').read_text().splitlines() if line.strip()]
    require(len({r['id'] for r in rows})==len(rows),'Duplicate parent record IDs')
    return report,audit,contract,manifest,rows


def summarize_setup(study,row,manifest):
    connection_file='native/'+row['connection_file'];resolution_file='native/'+row['context_resolution_file']
    require(manifest[connection_file]==row['connection_sha256'] and manifest[resolution_file]==row['context_resolution_sha256'],'Setup row/manifest binding differs')
    connection=read(checked_file(study,connection_file,row['connection_sha256']))
    resolution=read(checked_file(study,resolution_file,row['context_resolution_sha256']))
    require(connection['bucket_xy_m']==row['bucket_xy_m'] and connection['values']==row['values'] and connection['action_substituted'] is False and connection['passed'] is False,'Exact rejected setup proof required')
    require(connection['resolution_sha256']==row['context_resolution_sha256'],'Setup resolution binding differs')
    attempts=connection.get('endpoint_attempts') or {};pairs=attempts.get('pair_attempts',[])
    checked=[r.get('checked',{}) for r in pairs]
    lower=[r['minimum_clearance_lower_bound_m'] for r in checked if isinstance(r.get('minimum_clearance_lower_bound_m'),(int,float))]
    closest=[r.get('strict_geometry',{}).get('closest_pair') for r in checked]
    home=resolution.get('home_geometry') or {}
    result=dict(reason=row.get('invalid'),connection_sha256=row['connection_sha256'],resolution_sha256=row['context_resolution_sha256'],
        valid_ik_branches=attempts.get('valid_ik_branches'),pair_attempts=len(pairs),
        maximum_tested_minimum_clearance_m=max(lower) if lower else None,required_clearance_m=.005,
        closest_pair=next((p for p in closest if p),home.get('closest_pair')),
        layout_eligible=resolution.get('layout_eligible'),anchor_admitted=resolution.get('anchor_admitted'),
        home_minimum_clearance_m=home.get('minimum_clearance_m'),fixed_base_blockers=len(home.get('fixed_base_blockers',[])),
        interpretation='Finite IK failure does not prove unreachable geometry; a conservative clearance bound below 5 mm does not by itself prove penetration. Rejected setups had no ball flight.')
    return result


def summarize_flight(row,times,positions,forces):
    import numpy as np
    stop=int(row['physics_steps'])
    require(0<stop<len(times) and stop<=len(forces),'Scored terminal outside saved native trace')
    require(positions.shape==(len(times),3) and forces.shape[1]==6 and np.isfinite(positions).all() and np.isfinite(forces).all(),'Finite native trace with six contact channels required')
    require(np.array_equal(positions[0],np.asarray(row['native_initial_ball_center_m'])),'Native initial center differs')
    require(np.allclose(times,np.arange(len(times))/15360,atol=1e-12,rtol=0),'Native64 trace clock differs')
    events=[]
    for channel,name in enumerate(CONTACTS):
        hits=np.flatnonzero(forces[:stop,channel]>.01)
        if len(hits):
            index=int(hits[0]);events.append(dict(category=name,time_s=float(times[index]),ball_center_m=positions[index].astype(float).tolist()))
    events.sort(key=lambda e:(e['time_s'],e['category']))
    board=next((e for e in events if e['category']=='board'),None)
    subsequent=None
    if board:
        start=int(np.flatnonzero(forces[:stop,0]>.01)[0]);following=[]
        for channel,name in enumerate(CONTACTS[1:],1):
            hits=np.flatnonzero(forces[start:stop,channel]>.01)
            if len(hits):
                index=start+int(hits[0]);following.append(dict(category=name,time_s=float(times[index]),ball_center_m=positions[index].astype(float).tolist()))
        subsequent=min(following,key=lambda e:(e['time_s'],e['category'])) if following else None
    for field,name in [('whole_ball_entry_time_s','whole_ball_entry'),('capture_time_s','retention_completed')]:
        if row.get(field) is not None:events.append(dict(category=name,time_s=row[field]))
    events.sort(key=lambda e:(e['time_s'],e['category']))
    forbidden=any((forces[:stop,i]>.01).any() for i in (3,4,5))
    require(forbidden==bool(row.get('forbidden_contact_steps')),'Stored forbidden-contact classification differs')
    require(bool((forces[:stop,2]>.01).any())==bool(row.get('table_contact_steps')),'Stored table-contact classification differs')
    indices=np.unique(np.r_[np.linspace(0,stop,min(241,stop+1),dtype=int),0,stop])
    return dict(events=events,first_post_board_contact=subsequent,
        trace=dict(time_s=times[indices].astype(float).tolist(),position_m=positions[indices].astype(float).tolist(),
                   sample_basis='At most 241 actual native samples including endpoints; no interpolation',scope='Display trajectory only; dense native evidence determines scoring'),
        native_cache_sha256=row['native_cache_sha256'])


def diagnose_parent(study,audit_path):
    """Derive compact observations from byte-bound parent evidence; do not rescore."""
    study=Path(study);report,audit,contract,manifest,rows=checked_parent(study,audit_path)
    tests=[r for r in rows if r['role']=='test' and r['method']=='selected_ppo']
    require(len(tests)==70 and len({r['context_id'] for r in tests})==70,'Exactly 70 distinct parent selected-PPO requests required')
    cases=[];groups=defaultdict(list)
    for row in tests:
        if row['route_success']:continue
        item=dict(record_id=row['id'],record_sha256=digest(row),context_id=row['context_id'],bucket_xy_m=row['bucket_xy_m'],
            values=row['values'],native_release=row['native_release'],capture=False,reason=row.get('invalid'),
            whole_ball_entry=bool(row.get('whole_ball_entry')),max_retention_s=row.get('max_retention_s',0),
            forbidden_contact=bool(row.get('forbidden_contact_steps')),table_contact=bool(row.get('table_contact_steps')),
            board_bounce=bool(row.get('board_bounce')),invalid_physics=bool(row.get('invalid_physics')),events=[])
        if row['native_release']:groups[row['native_cache_file']].append((row,item))
        else:item['setup']=summarize_setup(study,row,manifest)
        cases.append(item)
    import numpy as np
    for name,members in groups.items():
        path=checked_file(study,'native/'+name,manifest['native/'+name])
        require(all(row['native_cache_sha256']==manifest['native/'+name] for row,_ in members),'Native row/manifest binding differs')
        with np.load(path,allow_pickle=False) as archive:
            times=np.asarray(archive['time_s'],dtype=float);positions=archive['position_m'];forces=archive['forces_N'];record_ids=archive['record_ids']
            for row,item in members:
                column=row['native_cache_row'];require(str(record_ids[column])==row['id'],'Native archive column differs')
                item.update(summarize_flight(row,times,positions[:,column],forces[:,column]))
    rejects=[r for r in cases if not r['native_release']];misses=[r for r in cases if r['native_release']]
    result=dict(version=DIAGNOSIS,status='completed',source=dict(report_sha256=sha(study/'report.json'),audit_sha256=sha(audit_path),
        rows_sha256=sha(study/'rows.jsonl'),evidence_manifest_sha256=sha(study/'evidence-manifest.json'),contract_fingerprint=contract['fingerprint'],checks_count=audit['checks_count']),
        summary=dict(requests=70,captures=sum(bool(r['route_success']) for r in tests),setup_rejections=len(rejects),flight_misses=len(misses),
            no_entry=sum(not r['whole_ball_entry'] for r in misses),entered_retained_but_forbidden=sum(r['whole_ball_entry'] and r['max_retention_s']>=1 and r['forbidden_contact'] for r in misses),
            forbidden_contact_misses=sum(r['forbidden_contact'] for r in misses),table_contact_misses=sum(r['table_contact'] for r in misses),
            uncovered_train_layouts=sum(not r['has_teacher'] for r in report['search_coverage'])),
        reject_reasons=dict(sorted(Counter(r['reason'] for r in rejects).items())),
        first_contact_groups=dict(sorted(Counter((r.get('first_post_board_contact') or {}).get('category','none') for r in misses).items())),cases=cases,
        scope='Authenticated saved outcomes, not new shots or a causal explanation. These former TEST positions are now DEVELOPMENT for recovery. Table rebounds are permitted; robot/floor/unattributed contacts remain disqualifying.')
    result['diagnosis_fingerprint']=digest(result)
    return result


METHODS=('recovery_bc','parent_bc','parent_ppo','previous_ppo')
_PARENT_CACHE={}

def parent_snapshot(study,audit_path):
    identity=(str(Path(study).resolve()),sha(Path(study)/'report.json'),sha(audit_path))
    if identity not in _PARENT_CACHE:
        _PARENT_CACHE.clear();_PARENT_CACHE[identity]=diagnose_parent(study,audit_path)
    return _PARENT_CACHE[identity]


def checked_diagnosis(value):
    out=dict(value);fp=out.pop('diagnosis_fingerprint',None)
    require(value.get('version')==DIAGNOSIS and value.get('status')=='completed' and digest(out)==fp,'Matching parent diagnosis fingerprint required')
    require(value['summary']['requests']==70 and len(value['cases'])==value['summary']['setup_rejections']+value['summary']['flight_misses'],'Parent diagnosis accounting differs')
    return value


def proof(report_path,raw,audit_path):
    result=dict(status='pending',full_checks_verified=False,native_checks_verified=False)
    if not audit_path or not Path(audit_path).is_file():return result
    audit=read(audit_path);checks=audit.get('checks',[])
    passed=(raw.get('status')=='completed' and audit.get('version')==AUDIT and audit.get('status')=='passed'
        and audit.get('full_checks_verified') is True and audit.get('native_checks_verified') is True
        and not audit.get('failures') and bool(checks) and audit.get('checks_count')==len(checks)
        and all(c.get('passed') is True for c in checks) and audit.get('report_sha256')==sha(report_path)
        and audit.get('contract_fingerprint')==raw.get('contract_fingerprint'))
    return dict(status='passed' if passed else 'unverified',full_checks_verified=passed,native_checks_verified=passed,
        proof_sha256=sha(audit_path),report_sha256=sha(report_path),contract_fingerprint=raw.get('contract_fingerprint'),checks_count=audit.get('checks_count'))


def stats(rows):
    captures=sum(bool(r.get('route_success')) for r in rows);releases=sum(bool(r.get('native_release')) for r in rows)
    return dict(attempts=len(rows),captures=captures,native_releases=releases,setup_rejections=len(rows)-releases,native_misses=releases-captures)


def compact(study,diagnosis,*,audit_path=None,pipeline=None):
    diagnosis=checked_diagnosis(diagnosis);study=Path(study);path=study/'report.json';pipeline=pipeline or {}
    defaults=dict(eligible_cases=39,blocked_cases=3,search_requests=624,confirmation_requests=78,maximum_fresh_requests=1000)
    out=dict(version=VERSION,status='pending',phase='preparing',parent_diagnosis=diagnosis,protocol=defaults,
        counts={},stage_counts=[],recovery_cases=[],test_comparison=[],test_positions=[],test_layouts=[],
        audit=dict(status='pending',full_checks_verified=False,native_checks_verified=False),
        clock=dict(started_unix_s=pipeline.get('started_unix_s'),deadline_unix_s=pipeline.get('global_deadline_unix_s')))
    if not path.is_file():
        if pipeline.get('status') in ('failed','audit_failed','audit_deferred','timeout','timed_out'):out['status']='stopped'
        return out
    raw=read(path);require(raw.get('version')==VERSION,'Explicit recovery study required')
    contract=read(study/'contract.json');unsigned=dict(contract);fp=unsigned.pop('fingerprint')
    require(fp==digest(unsigned)==raw.get('contract_fingerprint'),'Recovery contract fingerprint differs')
    require(raw['protocol']==contract['protocol'],'Recovery report protocol differs from frozen contract')
    protocol=raw['protocol']
    out.update(take(raw,('status','phase','elapsed_s','search_requests','search_completed_rounds','repeat_requests','budget_reasons','scope','contract_fingerprint')))
    out['report_sha256']=sha(path);out['counts']=take(raw.get('counts'),STATS)
    out['protocol']=dict(defaults,eligible_cases=protocol['search_layouts'],blocked_cases=protocol['fixed_base_skips'],
        search_requests=protocol['search_requests'],confirmation_requests=protocol['repeat_requests_max'],
        maximum_fresh_requests=protocol['maximum_requests'])
    out['stage_counts']=[dict(stage=k,counts=take(v,STATS)) for k,v in raw.get('stage_counts',{}).items()]
    out['control_gates']={key:take(value,('status','direct_captures','table_assisted_captures','negative_captures')) for key,value in raw.get('control_gates',{}).items() if key in ('pre','post')}
    out['test_comparison']=[dict(method=r['method'],counts=take(r['counts'],STATS),artifact_sha256=r.get('artifact_sha256')) for r in raw.get('test_comparison',[]) if r.get('method') in METHODS]
    out['test_positions']=[take(r,('context_id','bucket_xy_m','method','values','capture','native_release','invalid','invalid_physics','record_id')) for r in raw.get('test_positions',[]) if r.get('method') in METHODS]
    # Producer stores the frozen TEST coordinates before any outcomes.
    splits=read(study/'splits.json')
    require(splits==contract['splits'],'Recovery split file differs from frozen contract')
    layouts=splits['test']
    out['test_layouts']=[take(r,('id','context_id','bucket_xy_m','grid_cell')) for r in layouts]
    confirmed=raw.get('confirmed_teachers',[])
    out['confirmed_teachers']=len(confirmed) if isinstance(confirmed,list) else confirmed
    fit=raw.get('fit') or {}
    out['training_summary']=dict(status=fit.get('status'),trained=True if fit.get('status')=='actor_updated' else False if fit.get('status')=='no_confirmed_new_teacher' else None,
        old_teachers=fit.get('parent_teacher_count'),new_teachers=fit.get('new_teacher_count'),actor_steps=fit.get('actor_steps'))
    selected=raw.get('selected',[])
    if isinstance(selected,dict):selected=list(selected.values())
    selected_ids={r.get('context_id') for r in selected if isinstance(r,dict)}
    coverage={r['context_id']:r for r in raw.get('search_coverage',[])}
    confirmed_ids={context for context,value in coverage.items() if value.get('confirmed') is True}
    rows=[]
    if (study/'rows.jsonl').is_file():rows=[json.loads(line) for line in (study/'rows.jsonl').read_text().splitlines() if line.strip()]
    for case in diagnosis['cases']:
        if case['reason']=='fixed_base_clearance_rejected':continue
        context=case['context_id'];own=[r for r in rows if r.get('context_id')==context]
        search=[r for r in own if r.get('phase')=='search'];repeat=[r for r in own if r.get('phase')=='repeat']
        accepted=context in confirmed_ids
        finalized=raw.get('phase') in ('fit','test','controls_post','completed') or raw['status']=='completed'
        rejected=(len(repeat)==2 and stats(repeat)['captures']<2) or finalized
        out['recovery_cases'].append(dict(context_id=context,bucket_xy_m=case['bucket_xy_m'],search=stats(search),confirmation=stats(repeat),
            selected_for_confirmation=context in selected_ids,teacher_accepted=True if accepted else False if rejected else None))
    out['audit']=proof(path,raw,audit_path)
    if pipeline.get('status') in ('failed','audit_failed','audit_deferred','timeout','timed_out'):
        out['status']='stopped'
    require(len(json.dumps(out,allow_nan=False).encode())<=MAX_SNAPSHOT_BYTES,'Recovery snapshot exceeds bounded display size')
    return out


def guarded_targets(study,targets,inputs=()):
    root=Path(study).resolve();sources={Path(p).resolve() for p in inputs if p is not None};dest=[Path(p).resolve() for p in targets if p is not None]
    require(len(set(dest))==len(dest),'Publication targets must be distinct')
    require(all(not p.is_relative_to(root) and p not in sources for p in dest),'Publication must remain outside immutable study/input evidence')


def embed_html(original,snapshot):
    require(original.count(START)==original.count(END) and original.count(START)<=1,'Ambiguous recovery markers')
    web=Path(__file__).with_name('web');html=(web/'coverage-recovery.html').read_text();css=(web/'coverage-recovery.css').read_text();js=(web/'coverage-recovery.js').read_text()
    js=re.sub(r'^export (?=(?:const|function|class)\b)','',js,flags=re.M)
    require(not re.search(r'^\s*(?:import|export)\b',js,re.M) and '</script' not in js.lower() and '</style' not in css.lower(),'Self-contained recovery panel required')
    data=json.dumps(snapshot,allow_nan=False).replace('<','\\u003c').replace('>','\\u003e').replace('&','\\u0026')
    block=f'{START}\n<style>{css}</style>\n{html}\n<script type="application/json" id="coverage-recovery-snapshot">{data}</script>\n<script>(function(){{\n{js}\n}})();</script>\n{END}'
    if START in original:
        a,b=original.index(START),original.index(END)+len(END);require(b>a,'Recovery markers reversed');return original[:a]+block+original[b:]
    closing=list(re.finditer(r'</body\s*>',original,re.I));require(closing,'Existing document body required');at=closing[-1].start();return original[:at]+block+'\n'+original[at:]


def refresh(study,diagnosis,*,snapshot_path,target_html,display=None,audit_path=None,pipeline_path=None,now=None):
    guarded_targets(study,(snapshot_path,target_html,display),(audit_path,pipeline_path))
    state=read(pipeline_path) if pipeline_path and Path(pipeline_path).is_file() else {}
    out=compact(study,diagnosis,audit_path=audit_path,pipeline=state)
    out['publication']=dict(updated_unix_s=time.time() if now is None else now,pipeline_status=state.get('status'),pipeline_finished_unix_s=state.get('finished_unix_s'))
    snapshot=Path(snapshot_path);old=snapshot.read_bytes() if snapshot.exists() else None
    require(old is None or json.loads(old).get('version')==VERSION,'Existing snapshot belongs to another study')
    atomic_bytes(snapshot,(json.dumps(out,indent=2,allow_nan=False)+'\n').encode(),old)
    if display:
        path=Path(display);require(path.is_file(),'Existing display report required')
        for attempt in range(3):
            old=path.read_bytes();value=json.loads(old);value[KEY]=out
            try:atomic_bytes(path,(json.dumps(value,indent=2,allow_nan=False)+'\n').encode(),old);break
            except RuntimeError:
                if attempt==2:raise
    path=Path(target_html);old=path.read_bytes();atomic_bytes(path,embed_html(old.decode(),out).encode(),old)
    return out,state


def watch(study,diagnosis,*,snapshot_path,target_html,receipt,display=None,audit_path=None,pipeline_path=None,max_seconds=10830,monotonic=time.monotonic,sleep=time.sleep):
    require(0<max_seconds<=14400,'Finite watcher duration at most four hours required')
    guarded_targets(study,(snapshot_path,target_html,display,receipt),(audit_path,pipeline_path))
    start=monotonic();updates=0;errors=0;status='watching'
    while True:
        try:
            value,state=refresh(study,diagnosis,snapshot_path=snapshot_path,target_html=target_html,display=display,audit_path=audit_path,pipeline_path=pipeline_path)
            updates+=1
            if state.get('status') in ('completed','failed','audit_failed','audit_deferred','timeout','timed_out') and isinstance(state.get('finished_unix_s'),(int,float)):status='finished'
        except (OSError,ValueError,RuntimeError,json.JSONDecodeError):errors+=1;state={};value={}
        elapsed=monotonic()-start
        if status=='watching' and elapsed>=max_seconds:status='publication_timeout'
        record=dict(version='sircus-coverage-recovery-publication-v1',status=status,updates=updates,read_or_publication_errors=errors,elapsed_s=elapsed,pipeline_status=state.get('status'),audit_status=value.get('audit',{}).get('status'))
        atomic_bytes(receipt,(json.dumps(record,indent=2)+'\n').encode())
        if status!='watching':return record
        sleep(min(30,max_seconds-elapsed))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('study','parent-study','parent-audit','audit','pipeline-state','snapshot','target-html','display','receipt','diagnosis-output'):
        p.add_argument('--'+name,type=Path,required=name in ('study','parent-study','parent-audit','snapshot','target-html'))
    p.add_argument('--expected-html-sha256',required=True);p.add_argument('--watch',action='store_true');p.add_argument('--max-seconds',type=float,default=10830)
    a=p.parse_args();require(sha(a.target_html)==a.expected_html_sha256,'Target HTML changed; preserving existing work')
    guarded_targets(a.study,(a.snapshot,a.target_html,a.display,a.receipt,a.diagnosis_output),(a.parent_audit,a.audit,a.pipeline_state))
    for target in (a.snapshot,a.target_html,a.display,a.receipt,a.diagnosis_output):
        if target:require(not target.resolve().is_relative_to(a.parent_study.resolve()),'Do not write into parent evidence')
    require(not a.watch or a.receipt is not None,'Watcher receipt required')
    diagnosis=parent_snapshot(a.parent_study,a.parent_audit)
    if a.diagnosis_output:
        require(not a.diagnosis_output.exists(),'Preserve existing diagnosis evidence');atomic_bytes(a.diagnosis_output,(json.dumps(diagnosis,indent=2)+'\n').encode())
    options=dict(snapshot_path=a.snapshot,target_html=a.target_html,display=a.display,audit_path=a.audit,pipeline_path=a.pipeline_state)
    value,_=refresh(a.study,diagnosis,**options)
    result=watch(a.study,diagnosis,receipt=a.receipt,max_seconds=a.max_seconds,**options) if a.watch else {k:value.get(k) for k in ('status','phase','audit')}
    print(json.dumps(result))


if __name__=='__main__':main()
