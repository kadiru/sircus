"""Publish compact audited recovery evidence without running physics or services."""
from __future__ import annotations
import argparse
from collections import defaultdict
import json
from pathlib import Path
import time
from .prepare_coverage_recovery import checked_file, digest, read, require, sha
from .prepare_table_training import atomic_bytes, take
from .prepare_fixed35_imitation import replace_block, guard_targets

VERSION='sircus-fixed35-recovery-v1'
AUDIT='sircus-fixed35-recovery-audit-v1'
START='<!--FIXED35_RECOVERY_START-->'
END='<!--FIXED35_RECOVERY_END-->'
CSS_START='/*SIRCUS_FIXED35_RECOVERY_SNAPSHOT_START\n'
CSS_END='\nSIRCUS_FIXED35_RECOVERY_SNAPSHOT_END*/'
MAX_SNAPSHOT_BYTES=768*1024
MAXIMUM_REQUESTS=843
FIELDS=('id','context_id','phase','method','bucket_xy_m','values','native_release','invalid','invalid_physics','repeat','confirmation_index','selected_from_id','selected_from_sha256','forbidden_contact_steps','table_contact_steps','robot_contact_steps','whole_ball_entry','max_retention_s','board_bounce','native_initial_ball_center_m')


def captured(row):
    return row.get('route_success',row.get('capture')) is True and row.get('native_release') is True and not row.get('invalid_physics') and row.get('invalid')!='native_physics_outside_calibrated_domain'


def compact_row(row):
    out=take(row,FIELDS);out['capture']=captured(row);out['record_sha256']=digest(row)
    out['invalid_physics']=row.get('invalid_physics') is True or row.get('invalid')=='native_physics_outside_calibrated_domain'
    return out


def statistics(rows):
    n=len(rows);releases=sum(r.get('native_release') is True for r in rows);captures=sum(captured(r) for r in rows)
    require(all(type(r.get('native_release')) is bool for r in rows),'Every slot must be released or declined')
    return dict(attempts=n,native_releases=releases,captures=captures,setup_rejections=n-releases,native_misses=releases-captures,
                invalid_physics=sum(r.get('invalid_physics') is True or r.get('invalid')=='native_physics_outside_calibrated_domain' for r in rows))


def audit_proof(path,raw,audit_path):
    if audit_path is None or not Path(audit_path).is_file():return dict(status='pending',full_checks_verified=False,native_checks_verified=False)
    audit=read(audit_path);checks=audit.get('checks',[]);mp=Path(path).parent/'evidence-manifest.json';manifest=read(mp) if mp.is_file() else None
    passed=(raw.get('status')=='completed' and audit.get('version')==AUDIT and audit.get('status')=='passed'
        and audit.get('full_checks_verified') is True and audit.get('native_checks_verified') is True
        and audit.get('failures')==[] and bool(checks) and audit.get('checks_count')==len(checks)
        and all(c.get('passed') is True for c in checks) and audit.get('report_sha256')==sha(path)
        and audit.get('contract_fingerprint')==raw.get('contract_fingerprint') and manifest is not None
        and raw.get('evidence_sha256')==manifest and audit.get('evidence_manifest_sha256')==sha(mp))
    return dict(status='passed' if passed else 'unverified',full_checks_verified=passed,native_checks_verified=passed,
        report_sha256=sha(path),contract_fingerprint=raw.get('contract_fingerprint'),proof_sha256=sha(audit_path),checks_count=audit.get('checks_count'))


def build_cases(parent_cases,preserved,blocked,rows):
    require(len(parent_cases)==11 and len(preserved)==24,'Exact11 failed and24 successful parent layouts required')
    require(len({r['context_id'] for r in parent_cases+preserved})==35,'Distinct complete35-grid parent inventory required')
    require(all(not captured(r) for r in parent_cases) and all(captured(r) for r in preserved),'Parent outcomes differ from recovery task')
    blocked_ids={r['context_id'] for r in blocked};require(len(blocked_ids)==len(blocked)==2,'Exactlytwo fixed-base clearance blocked locations required')
    require(all(r.get('invalid')=='fixed_base_clearance_rejected' and r.get('native_release') is False for r in blocked),'Fixed-base clearance refusal required')
    lookup={r['id']:r for r in rows};require(len(lookup)==len(rows),'Duplicate new request IDs')
    cases=[]
    for parent in parent_cases:
        context=parent['context_id'];search=[r for r in rows if r['context_id']==context and r.get('phase')=='search']
        repeats=[r for r in rows if r['context_id']==context and r.get('phase')=='repeat'];groups=defaultdict(list)
        require(all(r['bucket_xy_m']==parent['bucket_xy_m'] for r in search+repeats),'No bucket displacement in recovery')
        if context in blocked_ids:require(not search and not repeats,'Blocked layouts cannot be silently searched')
        for row in repeats:groups[row['selected_from_id']].append(row)
        require(len(groups)<=3,'Atmostthree confirmation groups per context')
        confirmations=[]
        for selected_id,group in sorted(groups.items(),key=lambda item:(min(r.get('confirmation_index',0) for r in item[1]),item[0])):
            selected=lookup.get(selected_id)
            require(selected is not None and selected in search and captured(selected),'Confirmation must select an actual captured discovery request')
            require(len(group)==3 and sorted(r['repeat'] for r in group)==[0,1,2],'Three fresh repeats, all outcomes retained')
            require(all(r['values']==selected['values'] and r['bucket_xy_m']==selected['bucket_xy_m'] and r['selected_from_sha256']==digest(selected) for r in group),'Exact selected recipe/source binding required')
            confirmations.append(dict(selected=compact_row(selected),repeats=[compact_row(r) for r in sorted(group,key=lambda r:r['repeat'])]))
        good=[g for g in confirmations if all(r['capture'] for r in g['repeats'])]
        require(len(good)<=1,'Search stops after the first confirmed recipe')
        cases.append(dict(context_id=context,bucket_xy_m=parent['bucket_xy_m'],parent=compact_row(parent),blocked=context in blocked_ids,
            search=statistics(search),confirmation_groups=confirmations,confirmed_recipe=good[0]['selected'] if good else None))
    fresh=[r for r in rows if r.get('phase')=='preserved'];require(len(fresh)==24 and len({r['context_id'] for r in fresh})==24,'All24 unchanged parent replays required')
    originals={r['context_id']:r for r in preserved}
    for row in fresh:
        old=originals.get(row['context_id']);require(old is not None and row['values']==old['values'] and row['bucket_xy_m']==old['bucket_xy_m'],'Fresh preservation must use exact original recipe')
    require(all(r['phase'].startswith('controls_') or r['context_id'] in {p['context_id'] for p in parent_cases+preserved} for r in rows),'No added bucket layouts')
    return cases,[compact_row(r) for r in fresh]


def compact(study,*,audit_path=None,pipeline_path=None,now=None):
    study=Path(study);path=study/'report.json';state=read(pipeline_path) if pipeline_path and Path(pipeline_path).is_file() else {}
    terminal=state.get('status') in ('completed','failed','audit_failed','stopped','timeout','timed_out') and isinstance(state.get('finished_unix_s'),(int,float))
    display_protocol=dict(substeps=64,failure_layouts=11,total_layouts=35,preserve_layouts=24,confirmation_repeats=3,maximum_requests=MAXIMUM_REQUESTS,bucket_perturbations=False,training=False)
    out=dict(version=VERSION,status=state.get('status') if state.get('status') in ('starting','queued','running','cpu_audit') else 'pending',phase='Awaiting worker evidence',
        protocol=display_protocol,counts={},cases=[],preserved_recipes=[],preserved_replays=[],requests=[],audit=dict(status='pending'),control_gates={},
        publication=dict(terminal=terminal,pipeline_status=state.get('status'),updated_unix_s=time.time() if now is None else now))
    if path.is_file():
        raw=read(path);require(raw.get('version')==VERSION,'Matching fixed35 recovery study required')
        spec=read(study/'contract.json');unsigned=dict(spec);fp=unsigned.pop('fingerprint')
        require(digest(unsigned)==fp==raw.get('contract_fingerprint'),'Frozen recovery contract differs')
        protocol=spec['protocol'];require(raw.get('protocol')==protocol and protocol.get('training') is False and protocol.get('substeps')==64 and protocol.get('maximum_requests')==MAXIMUM_REQUESTS and protocol.get('noise') is False and protocol.get('bucket_perturbations') is False,'Exact native64 unperturbed bounded recovery required')
        for key,value in dict(grid_positions=35,parent_captures=24,development_failures=11,fixed_base_skips=2,search_layouts=9,repeat_per_selected=3,candidates_per_layout=3,preserved_requests=24,control_requests=18).items():
            require(protocol.get(key)==value,'Literal recovery protocol differs: '+key)
        out.update(take(raw,('status','phase','elapsed_s','scope','contract_fingerprint','completion_reason','budget_truncated')))
        out['request_cap']=MAXIMUM_REQUESTS
        out.update(raw_protocol=protocol,report_sha256=sha(path),audit=audit_proof(path,raw,audit_path),counts=take(raw.get('counts'),('attempts',)))
        out['control_gates']={k:take(v,('status','direct_captures','table_assisted_captures','negative_captures')) for k,v in raw.get('control_gates',{}).items() if k in ('pre','post')}
        if out['audit']['status']=='passed':
            manifest=raw['evidence_sha256']
            def bound(name):return checked_file(study,name,manifest[name])
            rows=[json.loads(line) for line in bound('rows.jsonl').read_text().splitlines() if line.strip()]
            require(42<=len(rows)<=MAXIMUM_REQUESTS and len(rows)==raw['counts']['attempts'] and raw['expected_requests']==MAXIMUM_REQUESTS,'Complete bounded request ledger and cap required')
            out['expected_requests']=len(rows)
            require(all(out['control_gates'].get(k,{}).get('status')=='passed' for k in ('pre','post')),'Both formal control gates required')
            for phase in ('controls_pre','controls_post'):require(sum(r['phase']==phase for r in rows)==9,'Nine original controls per gate')
            searchable,preserved,blocked=(read(bound('bootstrap/'+name+'.json')) for name in ('cases','preserved','blocked'))
            require(len(searchable)==9,'Nine searchable parent layouts required')
            parent_cases=sorted(searchable+blocked,key=lambda r:r['context_id'])
            out['cases'],out['preserved_replays']=build_cases(parent_cases,preserved,blocked,rows)
            diagnoses=read(bound('diagnoses.json'))
            require(len(diagnoses)==11 and {d['context_id'] for d in diagnoses}=={c['context_id'] for c in out['cases']},'All11 bound parent diagnoses required')
            summaries={c['context_id']:c for c in raw.get('cases',[])}
            for case in out['cases']:
                diagnosis=next(d for d in diagnoses if d['context_id']==case['context_id'])
                case['diagnosis']=take(diagnosis,('category','first_contacts_s','closest_xy','whole_ball_entry','max_retention_s','rim_band_contact','parent_row_sha256'))
                case['search_status']=summaries.get(case['context_id'],{}).get('status','completed')
            out['preserved_recipes']=[compact_row(r) for r in preserved]
            out['counts']=statistics(rows);require(all(raw['counts'].get(k)==v for k,v in out['counts'].items()),'Aggregate counts differ from ledger')
            out['requests']=[compact_row(r) for r in rows]
            out['confirmed_layouts']=sum(c['confirmed_recipe'] is not None for c in out['cases'])
            out['preserved_replay_counts']=statistics([r for r in rows if r['phase']=='preserved'])
            out['publication']['terminal']=True
        elif out['status']=='completed':out['status']='awaiting_audit'
    if state.get('status') in ('failed','audit_failed','timeout','timed_out','stopped'):out.update(status='stopped',phase=state['status'])
    require(len(json.dumps(out,allow_nan=False).encode())<=MAX_SNAPSHOT_BYTES,'Bounded768KiB snapshot required')
    return out


def baseline_summary(snapshot):
    a=snapshot.get('audit',{})
    require(snapshot.get('version')==VERSION and snapshot.get('status')=='completed' and a.get('status')=='passed' and a.get('full_checks_verified') is True and a.get('native_checks_verified') is True
        and a.get('report_sha256')==snapshot.get('report_sha256') and a.get('contract_fingerprint')==snapshot.get('contract_fingerprint') and len(snapshot.get('cases',[]))==11 and len(snapshot.get('preserved_recipes',[]))==24
        and snapshot.get('counts',{}).get('attempts')==snapshot.get('expected_requests'),'Completed audited recovery required')
    return {**take(snapshot,('version','status','contract_fingerprint','report_sha256','elapsed_s','scope','protocol','raw_protocol','counts','expected_requests','request_cap','completion_reason','budget_truncated','control_gates','audit','cases','preserved_recipes','preserved_replays','preserved_replay_counts','confirmed_layouts')),
        'interpretation':'Local search on11 historical fixed-grid failures;24 original successful recipes retained and replayed separately. Confirmed recipes require3fresh captures. Not a newly trained policy score or generalization test.'}


def encoded(snapshot):
    s=json.dumps(snapshot,allow_nan=False,separators=(',',':'));require(len(s.encode())<=MAX_SNAPSHOT_BYTES,'Bounded recovery snapshot required')
    return s.replace('<','\\u003c').replace('>','\\u003e').replace('&','\\u0026').replace('*','\\u002a')


def embed_css(original,snapshot):return replace_block(original,CSS_START,CSS_END,CSS_START+encoded(snapshot)+CSS_END,append=True)


def embed_html(original,snapshot):
    import re
    web=Path(__file__).with_name('web');html=(web/'fixed35-recovery.html').read_text();css=(web/'fixed35-recovery.css').read_text();js=(web/'fixed35-recovery.js').read_text()
    js=re.sub(r'^export (?=(?:const|function|class)\b)','',js,flags=re.M)
    require(not re.search(r'^\s*(?:import|export)\b',js,re.M) and '</script' not in js.lower() and '</style' not in css.lower(),'Self-contained safe panel required')
    block=f'{START}\n<style>{css}</style>\n{html}\n<script type="application/json" id="fixed35-recovery-snapshot">{encoded(snapshot)}</script>\n<script>(function(){{\n{js}\n}})();</script>\n{END}'
    return replace_block(original,START,END,block)


def refresh(study,*,snapshot_path,target_html,target_css,audit_path=None,pipeline_path=None,expected_html_sha256=None,expected_css_sha256=None,now=None):
    guard_targets(study,(audit_path,pipeline_path),(snapshot_path,target_html,target_css))
    hp,cp,sp=map(Path,(target_html,target_css,snapshot_path));oldh,oldc=hp.read_bytes(),cp.read_bytes();olds=sp.read_bytes() if sp.is_file() else None
    require(expected_html_sha256 is None or sha(hp)==expected_html_sha256,'HTML changed; preserve concurrent work')
    require(expected_css_sha256 is None or sha(cp)==expected_css_sha256,'CSS changed; preserve concurrent work')
    require(olds is None or json.loads(olds).get('version')==VERSION,'Snapshot belongs to another study')
    out=compact(study,audit_path=audit_path,pipeline_path=pipeline_path,now=now)
    newh,newc=embed_html(oldh.decode(),out).encode(),embed_css(oldc.decode(),out).encode()
    atomic_bytes(sp,(encoded(out)+'\n').encode(),olds);atomic_bytes(cp,newc,oldc);atomic_bytes(hp,newh,oldh)
    return out


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('study','audit','pipeline-state','snapshot','target-html','target-css'):p.add_argument('--'+name,type=Path,required=name in ('study','snapshot','target-html','target-css'))
    p.add_argument('--expected-html-sha256',required=True);p.add_argument('--expected-css-sha256',required=True);a=p.parse_args()
    out=refresh(a.study,snapshot_path=a.snapshot,target_html=a.target_html,target_css=a.target_css,audit_path=a.audit,pipeline_path=a.pipeline_state,expected_html_sha256=a.expected_html_sha256,expected_css_sha256=a.expected_css_sha256)
    print(json.dumps(dict(status=out['status'],audit=out['audit']['status'],snapshot_bytes=len(encoded(out).encode()))))

if __name__=='__main__':main()
