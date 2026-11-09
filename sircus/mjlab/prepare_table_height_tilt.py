"""Publish a separate audit-bound height/tilt study without changing existing panels."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import re
from .prepare_table_training_diagnosis import take, proof, report_path, default_audit, sha
from .prepare_table_training import atomic_bytes

VERSION='sircus-table-height-tilt-study-v1'
AUDIT='sircus-table-height-tilt-audit-v1'
KEY='table_height_tilt_study'
START='<!--TABLE_HEIGHT_TILT_START-->'
END='<!--TABLE_HEIGHT_TILT_END-->'
COUNTS=('attempts','native_releases','captures','setup_rejections','native_misses','robot_contact_shots','forbidden_contact_shots','invalid_physics','direct_captures','table_assisted_captures')
BINDINGS=('report_sha256','audit_sha256','manifest_sha256','rows_sha256','contract_sha256','contract_fingerprint')

def compact_cell(raw):
    result=take(raw,('candidate_id','grid_index','case_grid_index','values','bucket_xy_m','selected'))
    result['admission']=take(raw.get('admission'),('status','passed','reason','valid_ik_branches','connection_sha256','resolution_sha256'))
    result['discovery']=None if raw.get('discovery') is None else take(raw['discovery'],('record_id','record_sha256','native_release','captured','reason','route_quality','direct_capture','table_assisted_capture','invalid_physics','robot_contact_steps','forbidden_contact_steps','reward'))
    return result

def compact(study,audit_path=None):
    path=report_path(study);raw=json.loads(path.read_text())
    if raw.get('version')!=VERSION:raise ValueError('Explicit height/tilt study required')
    out=take(raw,('version','status','phase','contract_fingerprint','description','interpretation','elapsed_s','cpu_elapsed_s','native_elapsed_s','native_requested','requested_configurations','error','training','noise','posthoc_development','admission_matrix_sha256'))
    out['parent_bindings']=take(raw.get('parent_bindings'),BINDINGS)
    out['grid_protocol']=take(raw.get('grid_protocol'),('heights_m','tilts_deg','contexts','fixed_parameters','candidates_per_case','total_candidates','substeps','nominal_repeats'))
    out['budget']=take(raw.get('budget'),('maximum_requested','candidate_proposals','original_requests','control_requests','maximum_selected_repeat_requests','cpu_admission_limit_s','native_acquired_limit_s','maximum_queue_s','actual_requested','native_discovery_requests','selected_repeat_requests'))
    out['admission_summary']=take(raw.get('admission_summary'),('proposals','completed','admitted','declined','unattempted'))
    out['discovery_summary']=take(raw.get('discovery_summary'),('proposals','admitted')+COUNTS)
    out['counts']=take(raw.get('counts'),COUNTS)
    out['cases']=[]
    for case in raw.get('cases',[]):
        c=take(case,('context_id','source_record_id','source_record_sha256','bucket_xy_m','baseline_values'))
        c['grid']=[compact_cell(cell) for cell in case.get('grid',[])]
        c['selection']=None if case.get('selection') is None else take(case['selection'],('candidate_id','grid_index','case_grid_index','values','row_sha256','connection_sha256','route_quality','learning_reward'))
        c['repeats']={key:take(value,COUNTS) for key,value in case.get('repeats',{}).items() if key in ('baseline','selected')}
        out['cases'].append(c)
    out['repeats']={key:take(value,COUNTS) for key,value in raw.get('repeats',{}).items() if key in ('baseline','selected')}
    out['control_gates']={key:dict(take(value,('status','direct_captures','table_assisted_captures','negative_captures')),counts=take(value.get('counts'),COUNTS)) for key,value in raw.get('control_gates',{}).items() if key in ('pre','post')}
    out['audit']=proof(path,Path(audit_path) if audit_path else default_audit(path),AUDIT,raw)
    json.dumps(out,allow_nan=False)
    return out

def publish(study,display,**options):
    path=Path(display)
    if not path.is_file():raise ValueError('Existing derived display required')
    if path.resolve().is_relative_to(report_path(study).parent.resolve()):raise ValueError('Display must stay outside immutable study evidence')
    snapshot=compact(study,**options)
    for _ in range(3):
        original=path.read_bytes();value=json.loads(original);value[KEY]=snapshot
        try:atomic_bytes(path,(json.dumps(value,allow_nan=False,indent=2)+'\n').encode(),original);return snapshot
        except RuntimeError:continue
    raise RuntimeError('Concurrent display writer preserved')

def embed_html(original,snapshot):
    if original.count(START)!=original.count(END) or original.count(START)>1:raise ValueError('Ambiguous height/tilt markers')
    web=Path(__file__).with_name('web');html=(web/'table-height-tilt.html').read_text();css=(web/'table-height-tilt.css').read_text();js=(web/'table-height-tilt.js').read_text()
    js=re.sub(r'^export (?=(?:const|function|class)\b)','',js,flags=re.M)
    if re.search(r'^\s*(?:import|export)\b',js,re.M) or '</script' in js.lower() or '</style' in css.lower():raise ValueError('Self-contained assets required')
    data=json.dumps(snapshot,allow_nan=False).replace('<','\\u003c').replace('>','\\u003e').replace('&','\\u0026')
    block=f'{START}\n<style>{css}</style>\n{html}\n<script type="application/json" id="table-height-tilt-snapshot">{data}</script>\n<script>(function(){{\n{js}\n}})();</script>\n{END}'
    if START in original:
        a,b=original.index(START),original.index(END)+len(END)
        if b<a:raise ValueError('Height/tilt markers out of order')
        return original[:a]+block+original[b:]
    closing=list(re.finditer(r'</body\s*>',original,re.I))
    if not closing:raise ValueError('Existing document body required')
    at=closing[-1].start();return original[:at]+block+'\n'+original[at:]

def install(target,snapshot,expected_sha256):
    path=Path(target);original=path.read_bytes()
    if hashlib.sha256(original).hexdigest()!=expected_sha256:raise ValueError('Target HTML changed; existing work preserved')
    atomic_bytes(path,embed_html(original.decode(),snapshot).encode(),original)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('study','audit','display','target-html','snapshot'):parser.add_argument('--'+name,type=Path,required=name=='study')
    parser.add_argument('--expected-html-sha256');a=parser.parse_args()
    for target in (a.display,a.target_html,a.snapshot):
        if target and target.resolve().is_relative_to(report_path(a.study).parent.resolve()):raise ValueError('Publication must stay outside immutable study evidence')
    snapshot=publish(a.study,a.display,audit_path=a.audit) if a.display else compact(a.study,audit_path=a.audit)
    if a.target_html:install(a.target_html,snapshot,a.expected_html_sha256)
    if a.snapshot:atomic_bytes(a.snapshot,(json.dumps(snapshot,allow_nan=False,indent=2)+'\n').encode())
    print(json.dumps(dict(status=snapshot.get('status'),audit=snapshot['audit']['status'])))
if __name__=='__main__':main()
