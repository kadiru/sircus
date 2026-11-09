"""Validate a saved-shot rescore and embed its read-only scalar panel.

Does not execute simulation, import either scorer, alter source evidence or
restart a service. Only the explicitly named existing HTML file is replaced.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile

VERSION='sircus-table-assisted-scoring-display-v1'
SOURCE_VERSION='sircus-table-assisted-saved-rescore-v1'
AUDIT_VERSION='sircus-table-assisted-saved-rescore-audit-v1'
START='<!--TABLE_ASSISTED_SCORING_START-->'
END='<!--TABLE_ASSISTED_SCORING_END-->'
RAW_COUNTS=('requested','scorable_native','rejected','unscorable','evidence_errors',
    'invalid_native','old_captures','new_captures','new_table_assisted_captures','losses',
    'unchanged_misses','old_reward_sum','new_reward_sum','reward_count')


def require(value,label):
    if not value:raise ValueError(label)
def read(path):return json.loads(Path(path).read_text())
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    return h.hexdigest()
def finite(value):return type(value) in (int,float) and math.isfinite(value)
def integer(value):return type(value) is int and value>=0
def hash_value(value):return isinstance(value,str) and re.fullmatch('[a-f0-9]{64}',value) is not None
def label(value):return isinstance(value,str) and 0<len(value)<300 and not value.startswith('/') and '..' not in value.split('/') and '/Users/' not in value and '/home/' not in value
def check_counts(c):
    require(isinstance(c,dict) and all(integer(c.get(k)) for k in RAW_COUNTS if not k.endswith('_reward_sum')) and all(finite(c.get(k)) for k in ('old_reward_sum','new_reward_sum')),'Typed complete count/reward fields required')
    require(c['requested']==c['scorable_native']+c['rejected']+c['unscorable']+c['evidence_errors'],'Requested denominator changed')
    require(c['losses']==0 and c['new_captures']-c['old_captures']==c['new_table_assisted_captures'] and c['scorable_native']==c['new_captures']+c['unchanged_misses']+c['invalid_native'] and c['reward_count']==c['scorable_native'],'Capture, invalid-physics and reward accounting changed')


def display_counts(c):
    return dict(unique_requests=c['requested'],scorable_native=c['scorable_native'],retained_rejections=c['rejected'],
        unscorable=c['unscorable'],proof_errors=c['evidence_errors'],invalid_native=c['invalid_native'],
        old_captures=c['old_captures'],new_captures=c['new_captures'],new_table_assisted_captures=c['new_table_assisted_captures'],unchanged_misses=c['unchanged_misses'])


def compact(study):
    study=Path(study).resolve()
    summary,proof,manifest,contract=(read(study/name) for name in ('summary.json','audit.json','manifest.json','contract.json'))
    require(summary.get('version')==contract.get('version')==SOURCE_VERSION and summary.get('status')=='verified','Completed known rescore version required')
    checks=proof.get('checks')
    require(proof.get('version')==AUDIT_VERSION and proof.get('status')=='passed' and proof.get('full_checks_verified') is True and proof.get('failures')==[]
        and isinstance(checks,list) and checks and integer(proof.get('checks_count')) and proof['checks_count']==len(checks)
        and all(isinstance(c,dict) and c.get('passed') is True and isinstance(c.get('label'),str) and c['label'].strip() for c in checks),'Complete row/input verification required')
    pins=dict(summary_sha256=sha(study/'summary.json'),manifest_sha256=sha(study/'manifest.json'),contract_sha256=sha(study/'contract.json'))
    require(all(proof.get(k)==v for k,v in pins.items()) and summary.get('manifest_sha256')==pins['manifest_sha256'] and summary.get('contract_sha256')==pins['contract_sha256'],'Audit must bind exact summary, manifest and contract bytes')
    require(contract.get('no_new_native_steps') is True and contract.get('no_training') is True and proof.get('new_trials')==0 and proof.get('new_training_requests')==0,'Saved-shot only contract required')
    sources=contract.get('source_sha256');require(isinstance(sources,dict) and all(hash_value(v) for v in sources.values()) and hash_value(sources.get('table_assisted_score.py')),'Versioned rule source hash required')
    rule_hash=sources['table_assisted_score.py']
    require(isinstance(manifest.get('inputs'),dict) and manifest['inputs'] and all(label(k) and hash_value(v) for k,v in manifest['inputs'].items()),'Complete bound input manifest required')
    require(isinstance(manifest.get('artifacts'),dict) and set(manifest['artifacts'])=={'records.jsonl','contract.json'},'Exact derivative artifacts required')
    for name,value in manifest['artifacts'].items():require(hash_value(value) and sha(study/name)==value,'Derivative artifact changed: '+name)
    totals=summary.get('totals');check_counts(totals)
    require(totals['evidence_errors']==0 and proof.get('totals')==totals,'Contradictory evidence cannot publish verified results')
    metadata=summary.get('metadata');require(isinstance(metadata,dict) and all(integer(metadata.get(k)) for k in ('rows','duplicate_rows','unique_requests')) and metadata['rows']==metadata['unique_requests']+metadata['duplicate_rows'] and metadata['unique_requests']==totals['requested'],'Indexed and unique denominators differ')
    groups=summary.get('groups');require(isinstance(groups,list) and groups and groups==proof.get('groups'),'Proof must bind every study/method/phase/timestep group')
    seen=set();out_groups=[]
    for g in groups:
        require(all(label(g.get(k)) for k in ('source_study','method','phase')) and (g.get('substeps') in (None,'unknown') or integer(g['substeps']) and g['substeps']>0),'Safe group identity and explicit timestep required')
        key=tuple(g.get(k) for k in ('source_study','method','phase','substeps'));require(key not in seen,'Duplicate scoring group');seen.add(key);check_counts(g.get('counts'))
        c=g['counts'];out_groups.append(dict(source_study=g['source_study'],method=g['method'],phase=g['phase'],substeps=None if g.get('substeps')=='unknown' else g.get('substeps'),counts=display_counts(c),old_reward_sum=c['old_reward_sum'],new_reward_sum=c['new_reward_sum'],reward_count=c['reward_count']))
    for k in RAW_COUNTS:require(math.isclose(sum(g['counts'][k] for g in groups),totals[k],rel_tol=1e-10,abs_tol=1e-7),'Groups must partition totals: '+k)
    native_records={};record_count=0
    with (study/'records.jsonl').open() as stream:
        for line in stream:
            if not line.strip():continue
            row=json.loads(line);record_count+=1
            if row.get('status')=='scored':native_records[(row['source_study'],row['id'])]=row
    require(record_count==totals['requested'] and len(native_records)==totals['scorable_native'],'Complete unique derivative records required')
    examples=[];identities=set()
    for e in summary.get('examples',[]):
        require(all(label(e.get(k)) for k in ('id','source_study','method','phase')),'Safe example identity required')
        identity=(e['source_study'],e['id']);require(identity not in identities,'Duplicate example');identities.add(identity)
        row=native_records.get(identity);require(row is not None,'Example must have a bound native record')
        require(e.get('old_capture') is False and e.get('new_capture') is True and row['old_score'].get('route_success') is False and row['new_score'].get('route_success') is True,'Only newly credited captures are examples')
        require(all(e.get(k)==row.get(k) for k in ('method','phase','events','native_sha256','native_row')) and e.get('old_reward')==row['old_score'].get('reward') and e.get('new_reward')==row['new_score'].get('reward'),'Example cannot differ from native derivative')
        require(hash_value(e['native_sha256']) and integer(e['native_row']) and finite(e['old_reward']) and finite(e['new_reward']),'Bound example reward/archive identity required')
        events=e['events'];times={dst:events.get(src) for dst,src in dict(board_s='first_board_contact_s',table_contact_s='first_table_contact_s',first_entry_s='whole_ball_entry_s',capture_s='capture_s').items()}
        require(all(v is None or finite(v) and v>=0 for v in times.values()),'Finite observed timestamps required')
        examples.append(dict(record_id=e['id'],source_study=e['source_study'],method=e['method'],phase=e['phase'],old_capture=False,new_capture=True,old_reward=e['old_reward'],new_reward=e['new_reward'],native_archive_sha256=e['native_sha256'],native_row=e['native_row'],**times))
    require(len(examples)<=totals['new_table_assisted_captures'],'Example denominator exceeds newly credited count')
    t=display_counts(totals);t.update(metadata_rows=metadata['rows'],duplicate_rows=metadata['duplicate_rows'])
    exclusions=[]
    for reason,n in summary.get('exclusions',{}).items():
        require(integer(n),'Counted exclusions required');name=reason if isinstance(reason,str) and re.fullmatch(r'[A-Za-z0-9 _:.+-]{1,120}',reason) else 'other unsupported evidence'
        exclusions.append(f'{name}: {n}')
    return dict(version=VERSION,status='verified',totals=t,groups=out_groups,examples=examples,
        exclusion_summary='; '.join(exclusions) or 'No unsupported or contradictory evidence in this selected source set.',
        old_rule_version=contract['old_scoring_version'],new_rule_version=contract['new_scoring_version'],new_trials=0,new_training_requests=0,
        summary_sha256=pins['summary_sha256'],manifest_sha256=pins['manifest_sha256'],rule_sha256=rule_hash,
        audit=dict(passed=True,full_checks_verified=True,checks_count=len(checks),proof_sha256=sha(study/'audit.json'),
            summary_sha256=pins['summary_sha256'],manifest_sha256=pins['manifest_sha256'],rule_sha256=rule_hash))


def render_block(snapshot):
    assets=Path(__file__).with_name('web')
    html=(assets/'table-assisted-scoring.html').read_text();css=(assets/'table-assisted-scoring.css').read_text();js=(assets/'table-assisted-scoring.js').read_text()
    require('</style' not in css.lower() and '</script' not in js.lower(),'Unexpected inline closing tag')
    # This self-contained panel must not queue behind the host page's deferred
    # module dependency graph. Strip only our explicit top-level declarations;
    # retain the source module's exports for direct unit testing.
    classic=re.sub(r'^export (?=(?:const|function|class)\b)','',js,flags=re.M)
    require(re.search(r'^\s*(?:export|import)\b',classic,re.M) is None,'Embedded panel must be dependency-free')
    script="(function(){'use strict';\n"+classic+"\n})();"
    encoded=json.dumps(snapshot,allow_nan=False,separators=(',',':')).replace('<','\\u003c').replace('>','\\u003e').replace('&','\\u0026')
    return f'{START}\n<style>{css}</style>\n{html}\n<script type="application/json" id="table-assisted-scoring-snapshot">{encoded}</script>\n<script>{script}</script>\n{END}'


def embed_html(original,snapshot):
    require(original.count(START)==original.count(END) and original.count(START)<=1,'Own panel markers are ambiguous')
    block=render_block(snapshot)
    if START in original:
        begin=original.index(START);end=original.index(END)+len(END);require(end>begin,'Own panel markers are out of order');return original[:begin]+block+original[end:]
    matches=list(re.finditer(r'</body\s*>',original,re.I));require(matches,'Existing HTML body required')
    index=matches[-1].start();return original[:index]+block+'\n'+original[index:]


def install(target,study,expected_html_sha256):
    target,study=Path(target).resolve(),Path(study).resolve()
    require(target.is_file() and not target.is_relative_to(study),'Explicit existing HTML outside immutable evidence required')
    require(hash_value(expected_html_sha256) and sha(target)==expected_html_sha256,'Target changed before install')
    snapshot=compact(study);previous=target.read_bytes();require(hashlib.sha256(previous).hexdigest()==expected_html_sha256,'Concurrent target edit preserved')
    result=embed_html(previous.decode('utf-8'),snapshot).encode('utf-8');temporary=None
    try:
        with tempfile.NamedTemporaryFile(dir=target.parent,prefix='.table-assisted-scoring-',suffix='.tmp',delete=False) as stream:
            temporary=Path(stream.name);stream.write(result);stream.flush();os.fsync(stream.fileno())
        os.chmod(temporary,target.stat().st_mode&0o777)
        require(target.read_bytes()==previous,'Concurrent target edit preserved')
        os.replace(temporary,target);temporary=None
    finally:
        if temporary is not None:temporary.unlink(missing_ok=True)
    return snapshot


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--study',type=Path,required=True);p.add_argument('--target-html',type=Path);p.add_argument('--expected-html-sha256');p.add_argument('--snapshot',type=Path);a=p.parse_args()
    snapshot=install(a.target_html,a.study,a.expected_html_sha256) if a.target_html else compact(a.study)
    if a.snapshot:
        require(not a.snapshot.resolve().is_relative_to(a.study.resolve()),'Snapshot must stay outside evidence');a.snapshot.write_text(json.dumps(snapshot,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(status=snapshot['status'],totals=snapshot['totals'])))
if __name__=='__main__':main()
