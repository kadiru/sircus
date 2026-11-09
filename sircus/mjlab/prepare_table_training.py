"""Additive read-only publication; no simulation, policy promotion or restart."""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile

VERSION='sircus-table-assisted-training-v1'
AUDIT='sircus-table-assisted-training-audit-v1'
KEY='table_training_study'
START='<!--TABLE_TRAINING_START-->'
END='<!--TABLE_TRAINING_END-->'
STATS=('attempts','native_releases','captures','direct_captures','table_assisted_captures','setup_rejections','native_misses','forbidden_contact_shots','clean','mean_reward','invalid')

def read(path):return json.loads(Path(path).read_text())
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def take(raw,keys):return {k:copy.deepcopy(raw[k]) for k in keys if isinstance(raw,dict) and k in raw}
def atomic_bytes(path,raw,expected=None):
    path=Path(path);tmp=None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent,prefix='.table-training-',delete=False) as stream:
            tmp=Path(stream.name);stream.write(raw);stream.flush();os.fsync(stream.fileno())
        if path.exists():os.chmod(tmp,path.stat().st_mode&0o777)
        if expected is not None and path.read_bytes()!=expected:raise RuntimeError('Concurrent publication preserved')
        os.replace(tmp,path);tmp=None
    finally:
        if tmp is not None:tmp.unlink(missing_ok=True)

def compact(study,audit_path=None,pipeline=None):
    study=Path(study);path=study/'report.json'
    if not path.is_file():
        return dict(version=VERSION,status=(pipeline or {}).get('status','preparing'),phase='input verification / waiting for 4090',train_requests=0,updates=0)
    raw=read(path)
    if raw.get('version')!=VERSION:raise ValueError('Explicit v3 training report required')
    out=take(raw,('version','status','phase','scope','contract_fingerprint','train_requests','updates','elapsed_s'))
    # Only finite scalar counts and declared identifiers reach this display.
    out['counts']=take(raw.get('counts'),STATS)
    out['bootstrap_summary']=take(raw.get('bootstrap_summary'),('requests','native_releases','captures','teacher_count','table_assisted_teachers','newly_credited_captures','changed_teacher_count','successful_layouts'))
    out['protocol']=take(raw.get('protocol'),('substeps','train_requests','test_layouts','maximum_fresh_requests','noise'))
    out['curves']=[dict(samples=r.get('samples'),validation=take(r.get('validation'),STATS)) for r in raw.get('curves',[])]
    out['test_comparison']=[dict(method=r.get('method'),counts=take(r.get('counts'),STATS),artifact_sha256=r.get('artifact_sha256')) for r in raw.get('test_comparison',[]) if r.get('method') in ('old_bc','old_ppo','bc','ppo')]
    out['test_positions']=[take(r,('context_id','bucket_xy_m','method','capture','table_assisted_capture','direct_capture','native_release')) for r in raw.get('test_positions',[])]
    out['control_gates']={key:take(value,('status','positive_captures','negative_captures','direct_captures','table_assisted_captures')) for key,value in raw.get('control_gates',{}).items() if key in ('pre','post')}
    out['audit']=dict(status='pending')
    proofpath=Path(audit_path) if audit_path else study.with_name(study.name+'-audit.json')
    if proofpath.is_file():
        proof=read(proofpath);checks=proof.get('checks',[])
        passed=(raw.get('status')=='completed' and proof.get('version')==AUDIT and proof.get('status')=='passed'
                and proof.get('full_checks_verified') is True and proof.get('contract_fingerprint')==raw.get('contract_fingerprint')
                and proof.get('report_sha256')==sha(path) and not proof.get('failures') and isinstance(checks,list) and bool(checks)
                and proof.get('checks_count')==len(checks) and all(c.get('passed') is True for c in checks))
        out['audit']=dict(status='passed' if passed else 'unverified',full_checks_verified=passed,
                          contract_fingerprint=proof.get('contract_fingerprint'),proof_sha256=sha(proofpath))
    if raw.get('status')=='failed':out['error']='Worker stopped; its detailed error and partial evidence are saved in the study log.'
    if pipeline and pipeline.get('status') in ('failed','audit_failed'):
        out.update(status='failed',error='Pipeline stopped; evidence retained. See the saved worker/audit log.')
    json.dumps(out,allow_nan=False)
    return out

def publish(study,display,pipeline=None):
    display=Path(display)
    if not display.is_file():raise ValueError('Existing read-only display report required')
    for _ in range(3):
        original=display.read_bytes();value=json.loads(original);value[KEY]=compact(study,pipeline=pipeline)
        try:
            atomic_bytes(display,(json.dumps(value,allow_nan=False,indent=2)+'\n').encode(),original)
            return value[KEY]
        except RuntimeError:continue
    raise RuntimeError('Concurrent display writer preserved')

def embed_html(original,snapshot):
    if original.count(START)!=original.count(END) or original.count(START)>1:raise ValueError('Ambiguous own UI markers')
    web=Path(__file__).with_name('web');html=(web/'table-training.html').read_text();css=(web/'table-training.css').read_text();js=(web/'table-training.js').read_text()
    js=re.sub(r'^export (?=(?:const|function|class)\b)','',js,flags=re.M)
    if re.search(r'^\s*(?:import|export)\b',js,re.M) or '</script' in js.lower() or '</style' in css.lower():raise ValueError('Self-contained inline assets required')
    data=json.dumps(snapshot,allow_nan=False).replace('<','\\u003c').replace('>','\\u003e').replace('&','\\u0026')
    block=f'{START}\n<style>{css}</style>\n{html}\n<script type="application/json" id="table-training-snapshot">{data}</script>\n<script>(function(){{\n{js}\n}})();</script>\n{END}'
    if START in original:
        a,b=original.index(START),original.index(END)+len(END)
        if b<a:raise ValueError('Own markers out of order')
        return original[:a]+block+original[b:]
    closing=list(re.finditer(r'</body\s*>',original,re.I))
    if not closing:raise ValueError('Existing document body required')
    at=closing[-1].start();return original[:at]+block+'\n'+original[at:]

def install(target,snapshot,expected_sha256):
    target=Path(target);original=target.read_bytes()
    if hashlib.sha256(original).hexdigest()!=expected_sha256:raise ValueError('Target HTML changed; existing work preserved')
    atomic_bytes(target,embed_html(original.decode(),snapshot).encode(),original)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--study',type=Path,required=True);p.add_argument('--display',type=Path);p.add_argument('--target-html',type=Path);p.add_argument('--expected-html-sha256');p.add_argument('--snapshot',type=Path);a=p.parse_args()
    value=publish(a.study,a.display) if a.display else compact(a.study)
    if a.target_html:install(a.target_html,value,a.expected_html_sha256)
    if a.snapshot:
        if a.snapshot.resolve().is_relative_to(a.study.resolve()):raise ValueError('Keep display snapshot outside immutable evidence')
        atomic_bytes(a.snapshot,(json.dumps(value,allow_nan=False,indent=2)+'\n').encode())
    print(json.dumps(dict(status=value.get('status'),phase=value.get('phase'),audit=value.get('audit'))))
if __name__=='__main__':main()
