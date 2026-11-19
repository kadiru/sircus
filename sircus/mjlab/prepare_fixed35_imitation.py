"""Audit-gated, additive saved results; no fitting, solver or service commands."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import re
import time

from .prepare_coverage_recovery import checked_file, digest, read, require, sha
from .prepare_table_training import atomic_bytes, take

VERSION='sircus-fixed35-imitation-v1'
AUDIT='sircus-fixed35-imitation-audit-v1'
METHODS=('fixed35_bc','current_bc')
START='<!--FIXED35_IMITATION_START-->'
END='<!--FIXED35_IMITATION_END-->'
CSS_START='/*SIRCUS_FIXED35_IMITATION_SNAPSHOT_START\n'
CSS_END='\nSIRCUS_FIXED35_IMITATION_SNAPSHOT_END*/'
MAX_SNAPSHOT_BYTES=256*1024
STATS=('attempts','native_releases','captures','setup_rejections','native_misses','invalid_physics','direct_captures','table_assisted_captures','forbidden_contact_shots','robot_contact_shots')
PROTOCOL_FIELDS=('substeps','methods','test_layouts','test_requests','control_requests','maximum_requests','base_teacher_count','correction_teacher_count','new_teacher_count','teacher_count','actor_steps','critic_steps','actor_lr','noise','bucket_perturbations','evaluation_scope','rotation','teacher_semantics','geometry_prior')
POSITION_FIELDS=('context_id','bucket_xy_m','method','values','capture','native_release','invalid','invalid_physics','record_id','forbidden_contact_steps','table_contact_steps')


def audit_proof(path,raw,audit_path):
    if audit_path is None or not Path(audit_path).is_file():return dict(status='pending',full_checks_verified=False,native_checks_verified=False)
    audit=read(audit_path);checks=audit.get('checks',[]);manifest_path=Path(path).parent/'evidence-manifest.json'
    manifest=read(manifest_path) if manifest_path.is_file() else None
    passed=(raw.get('status')=='completed' and audit.get('version')==AUDIT and audit.get('status')=='passed'
        and audit.get('full_checks_verified') is True and audit.get('native_checks_verified') is True
        and audit.get('failures')==[] and bool(checks) and audit.get('checks_count')==len(checks)
        and all(c.get('passed') is True for c in checks) and audit.get('report_sha256')==sha(path)
        and audit.get('contract_fingerprint')==raw.get('contract_fingerprint') and manifest is not None
        and manifest==raw.get('evidence_sha256') and audit.get('evidence_manifest_sha256')==sha(manifest_path))
    return dict(status='passed' if passed else 'unverified',full_checks_verified=passed,native_checks_verified=passed,
        proof_sha256=sha(audit_path),report_sha256=sha(path),contract_fingerprint=raw.get('contract_fingerprint'),checks_count=audit.get('checks_count'))


def counts_valid(c):
    keys=('attempts','native_releases','captures','setup_rejections','native_misses')
    return isinstance(c,dict) and all(type(c.get(k)) is int and c[k]>=0 for k in keys) and c['attempts']==c['native_releases']+c['setup_rejections'] and c['native_releases']==c['captures']+c['native_misses']


def paired_positions(layouts,positions,comparison):
    require(len(layouts)==35 and len({r.get('id',r.get('context_id')) for r in layouts})==35,'Exact35 distinct fixed layouts required')
    from .fixed35_imitation_contract import fixed_grid
    require(layouts==[take(r,('id','context_id','bucket_xy_m','grid_cell','canonical_id','row','column')) for r in fixed_grid()], 'Exact canonical35 coordinates required; no perturbations')
    expected={r.get('id',r.get('context_id')):r['bucket_xy_m'] for r in layouts}
    require(len(comparison)==2 and {r['method'] for r in comparison}==set(METHODS),'Both frozen policy comparisons required')
    maps={}
    for method in METHODS:
        rows=[r for r in positions if r['method']==method];counts=next(r['counts'] for r in comparison if r['method']==method)
        require(counts_valid(counts) and counts['attempts']==35,'All35 requests including declines required')
        require(len(rows)==35 and len({r['context_id'] for r in rows})==35 and {r['context_id'] for r in rows}==set(expected),'Complete paired35-position outcomes required')
        require(all(r['bucket_xy_m']==expected[r['context_id']] and type(r.get('capture')) is bool and type(r.get('native_release')) is bool for r in rows),'Exact matched bucket coordinates and outcome booleans required')
        require(all(not r['capture'] or (r['native_release'] and not r.get('invalid_physics') and r.get('invalid')!='native_physics_outside_calibrated_domain') for r in rows),'Declined or invalid request cannot capture')
        require(sum(r['capture'] for r in rows)==counts['captures'] and sum(r['native_release'] for r in rows)==counts['native_releases'],'Paired outcomes differ from model counts')
        maps[method]={r['context_id']:r for r in rows}
    require(len(positions)==70,'Exactly70 grid outcomes required')
    gains=losses=shared=neither=0
    for key in expected:
        new,old=(maps[m][key]['capture'] for m in METHODS)
        gains+=new and not old;losses+=old and not new;shared+=new and old;neither+=not new and not old
    return dict(gains=gains,losses=losses,shared_captures=shared,shared_failures=neither,net_captures=gains-losses,positions=35)


def compact(study,*,audit_path=None,pipeline_path=None,now=None):
    study=Path(study);path=study/'report.json';state=read(pipeline_path) if pipeline_path and Path(pipeline_path).is_file() else {}
    terminal=state.get('status') in ('completed','failed','audit_failed','timeout','timed_out','stopped') and isinstance(state.get('finished_unix_s'),(int,float))
    out=dict(version=VERSION,status=state.get('status') if state.get('status') in ('starting','queued','running','cpu_audit') else 'pending',phase='Awaiting worker evidence',counts={},control_gates={},test_layouts=[],test_comparison=[],test_positions=[],teacher_corrections=[],artifacts={},fit={},audit=dict(status='pending'),
        publication=dict(updated_unix_s=time.time() if now is None else now,pipeline_status=state.get('status'),terminal=terminal),
        protocol=dict(substeps=64,test_layouts=35,test_requests=70,control_requests=18,maximum_requests=88,base_teacher_count=97,correction_teacher_count=2,new_teacher_count=2,teacher_count=99,actor_steps=800,noise=False,bucket_perturbations=False,methods=list(METHODS)),paired=None)
    if path.is_file():
        raw=read(path);require(raw.get('version')==VERSION,'Matching fixed35-imitation study required')
        spec=read(study/'contract.json');unsigned=dict(spec);fp=unsigned.pop('fingerprint')
        require(digest(unsigned)==fp==raw.get('contract_fingerprint'),'Frozen comparison contract differs')
        protocol=spec['protocol'];require(raw.get('protocol')==protocol and protocol['substeps']==64 and protocol['test_layouts']==35 and protocol['test_requests']==70 and protocol['maximum_requests']==88 and protocol.get('control_requests')==18 and protocol.get('methods')==list(METHODS),'Exact native64/35-position frozen protocol required')
        require(protocol.get('base_teacher_count')==97 and protocol.get('correction_teacher_count')==2 and protocol.get('teacher_count')==99 and protocol.get('noise') is False and protocol.get('bucket_perturbations') is False and protocol.get('actor_steps')==800 and protocol.get('critic_steps')==0,'Fixed97+2 actor-only refit required')
        out.update(take(raw,('status','phase','elapsed_s','scope','contract_fingerprint')))
        out['report_sha256']=sha(path);out['counts']=take(raw.get('counts'),STATS)
        out['protocol']=take(protocol,PROTOCOL_FIELDS)
        out['control_gates']={k:take(v,('status','direct_captures','table_assisted_captures','negative_captures')) for k,v in raw.get('control_gates',{}).items() if k in ('pre','post')}
        out['audit']=audit_proof(path,raw,audit_path)
        # No unpublished/provisional TEST outcomes or weights become a visible
        # completed comparison. The bound independent audit owns that gate.
        if out['audit']['status']=='passed':
            manifest=raw['evidence_sha256'];splits=read(checked_file(study,'splits.json',manifest['splits.json']))
            require(splits==spec['splits'],'Frozen split file differs')
            out['test_layouts']=[take(r,('id','context_id','bucket_xy_m','grid_cell','canonical_id','row','column')) for r in splits['test']]
            out['test_comparison']=[dict(method=r['method'],counts=take(r['counts'],STATS),artifact_sha256=r['artifact_sha256']) for r in raw['test_comparison']]
            out['test_positions']=[take(r,POSITION_FIELDS) for r in raw['test_positions']]
            out['paired']=paired_positions(out['test_layouts'],out['test_positions'],out['test_comparison'])
            require(out['counts'].get('attempts')==88 and all(out['control_gates'].get(k,{}).get('status')=='passed' for k in ('pre','post')),'Complete request ledger and both controls required')
            out['fit']=take(raw.get('fit'),('status','actor_steps','critic_steps','parent_teacher_count','new_teacher_count','teacher_count','before','after','initial_actor_loss','final_actor_loss','critic_before_sha256','critic_after_sha256'))
            out['artifacts']={k:take(v,('sha256','weight_sha256','samples','seed')) for k,v in raw.get('artifacts',{}).items() if k in METHODS}
            require(raw.get('fit',{}).get('parent_teacher_count')==97 and raw.get('fit',{}).get('new_teacher_count')==2 and raw.get('fit',{}).get('actor_steps')==800 and raw.get('fit',{}).get('critic_steps')==0,'Audited fit must match fixed teacher/step counts')
            require(len(raw.get('teacher_corrections',[]))==2 and len({r['context_id'] for r in raw['teacher_corrections']})==2,'Exactly two distinct corrected teacher contexts required')
            require(all(r['confirmation_captures']==r['confirmation_requests']==3 for r in raw['teacher_corrections']),'Every correction requires its three verified captures')
            require(set(out['artifacts'])==set(METHODS) and all(out['artifacts'][r['method']]['sha256']==r['artifact_sha256'] for r in out['test_comparison']),'Frozen artifact identity differs from tested checkpoint')
            out['teacher_corrections']=[take(r,('context_id','bucket_xy_m','values','source_study','source_record_id','source_record_sha256','source_report_sha256','confirmation_captures','confirmation_requests')) for r in raw.get('teacher_corrections',[])]
        else:
            out['counts']=take(out['counts'],('attempts',))
            if out['status']=='completed':out['status']='awaiting_audit'
    if state.get('status') in ('failed','audit_failed','timeout','timed_out','stopped'):
        out.update(status='stopped',phase=state.get('status'))
    if out['status']=='completed' and out['audit'].get('status')=='passed':out['publication']['terminal']=True
    require(len(json.dumps(out,allow_nan=False).encode())<=MAX_SNAPSHOT_BYTES,'Bounded256KiB snapshot required')
    return out


def baseline_summary(snapshot):
    audit=snapshot.get('audit',{})
    require(snapshot.get('version')==VERSION and snapshot.get('status')=='completed'
        and audit.get('status')=='passed' and audit.get('full_checks_verified') is True
        and audit.get('native_checks_verified') is True
        and audit.get('report_sha256')==snapshot.get('report_sha256')
        and audit.get('contract_fingerprint')==snapshot.get('contract_fingerprint')
        and snapshot.get('counts',{}).get('attempts')==88
        and all(snapshot.get('control_gates',{}).get(k,{}).get('status')=='passed' for k in ('pre','post')),
        'Completed audited comparison required')
    paired=paired_positions(snapshot['test_layouts'],snapshot['test_positions'],snapshot['test_comparison'])
    require(snapshot.get('paired')==paired,'Paired summary differs')
    protocol=snapshot.get('protocol',{})
    require(protocol.get('substeps')==64 and protocol.get('test_layouts')==35
        and protocol.get('test_requests')==70 and protocol.get('bucket_perturbations') is False,
        'Exact unperturbed fixed-grid protocol required')
    return {**take(snapshot,('version','status','contract_fingerprint','report_sha256','elapsed_s','scope','protocol','counts','control_gates','audit','fit','artifacts','teacher_corrections','test_layouts','test_comparison','test_positions','paired')),
        'interpretation':'Frozen paired comparison on the 35 historical fixed grid positions; both actors use bucket-facing yaw. This is coverage/regression, not held-out generalization. One request per policy per position, all declines count. No perturbations or automatic promotion.'}


def replace_block(original,start,end,block,*,append=False):
    require(original.count(start)==original.count(end) and original.count(start)<=1,'Ambiguous publication markers')
    if start in original:
        a,b=original.index(start),original.index(end)+len(end);require(b>a,'Reversed publication markers');return original[:a]+block+original[b:]
    if append:return original+block
    closing=list(re.finditer(r'</body\s*>',original,re.I));require(closing,'Existing HTML body required');at=closing[-1].start();return original[:at]+block+original[at:]


def encoded(snapshot):
    text=json.dumps(snapshot,allow_nan=False,separators=(',',':'));require(len(text.encode())<=MAX_SNAPSHOT_BYTES,'Bounded256KiB snapshot required')
    return text.replace('<','\\u003c').replace('>','\\u003e').replace('&','\\u0026').replace('*','\\u002a')


def embed_css(original,snapshot):return replace_block(original,CSS_START,CSS_END,CSS_START+encoded(snapshot)+CSS_END,append=True)


def embed_html(original,snapshot):
    web=Path(__file__).with_name('web');html=(web/'fixed35-imitation.html').read_text();css=(web/'fixed35-imitation.css').read_text();js=(web/'fixed35-imitation.js').read_text()
    js=re.sub(r'^export (?=(?:const|function|class)\b)','',js,flags=re.M)
    require(not re.search(r'^\s*(?:import|export)\b',js,re.M) and '</script' not in js.lower() and '</style' not in css.lower(),'Self-contained panel required')
    block=f'{START}\n<style>{css}</style>\n{html}\n<script type="application/json" id="fixed35-imitation-snapshot">{encoded(snapshot)}</script>\n<script>(function(){{\n{js}\n}})();</script>\n{END}'
    return replace_block(original,START,END,block)


def guard_targets(study,inputs,targets):
    root=Path(study).resolve();sources={Path(p).resolve() for p in inputs if p is not None};dest=[Path(p).resolve() for p in targets if p is not None]
    require(len(dest)==len(set(dest)) and all(p not in sources and not p.is_relative_to(root) for p in dest),'Keep distinct publication outputs outside immutable evidence')


def refresh(study,*,snapshot_path,target_html,target_css,audit_path=None,pipeline_path=None,expected_html_sha256=None,expected_css_sha256=None,now=None):
    guard_targets(study,(audit_path,pipeline_path),(snapshot_path,target_html,target_css))
    html,css,snapshot=map(Path,(target_html,target_css,snapshot_path));old_html,old_css=html.read_bytes(),css.read_bytes()
    if expected_html_sha256 is not None:require(sha(html)==expected_html_sha256,'HTML changed; preserve concurrent work')
    if expected_css_sha256 is not None:require(sha(css)==expected_css_sha256,'CSS changed; preserve concurrent work')
    previous=snapshot.read_bytes() if snapshot.exists() else None
    require(previous is None or json.loads(previous).get('version')==VERSION,'Snapshot belongs to another study')
    out=compact(study,audit_path=audit_path,pipeline_path=pipeline_path,now=now)
    new_html,new_css=embed_html(old_html.decode(),out).encode(),embed_css(old_css.decode(),out).encode()
    atomic_bytes(snapshot,(encoded(out)+'\n').encode(),previous);atomic_bytes(css,new_css,old_css);atomic_bytes(html,new_html,old_html)
    return out


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('study','audit','pipeline-state','snapshot','target-html','target-css'):p.add_argument('--'+name,type=Path,required=name in ('study','snapshot','target-html','target-css'))
    p.add_argument('--expected-html-sha256',required=True);p.add_argument('--expected-css-sha256',required=True);a=p.parse_args()
    out=refresh(a.study,snapshot_path=a.snapshot,target_html=a.target_html,target_css=a.target_css,audit_path=a.audit,pipeline_path=a.pipeline_state,expected_html_sha256=a.expected_html_sha256,expected_css_sha256=a.expected_css_sha256)
    print(json.dumps(dict(status=out['status'],audit=out['audit']['status'],bytes=len(encoded(out)))))


if __name__=='__main__':main()
