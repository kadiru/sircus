"""Publish an additive read-only physics-ensemble evidence snapshot.

No training, inference, native simulation, existing report or service is changed.
Partial observations remain provisional until the complete independent proof binds
the exact finished report and evidence-manifest bytes.
"""
import argparse
import copy
import datetime
import hashlib
import json
import math
from pathlib import Path
import re

KEY='physics_ensemble_study'
VERSION='sircus-physics-ensemble-v1'
AUDIT_VERSION='sircus-physics-ensemble-audit-v1'
METHODS=('bc','ppo','ensemble')
STATS=('attempts','requested','native_releases','captures','clean','rejected','native_misses','invalid','invalid_physics','mean_reward')


def select(raw,keys):
    return {k:copy.deepcopy(raw[k]) for k in keys if isinstance(raw,dict) and k in raw}


def optional_json(path):
    try:return json.loads(Path(path).read_text())
    except (OSError,ValueError):return None


def sha(data):return hashlib.sha256(data).hexdigest()


def is_sha(value):return isinstance(value,str) and bool(re.fullmatch(r'[a-f0-9]{64}',value))


def hashes(raw):
    return {k:v for k,v in raw.items() if isinstance(k,str) and not Path(k).is_absolute() and '..' not in Path(k).parts and is_sha(v)} if isinstance(raw,dict) else {}


def artifact(raw):
    value={k:v for k,v in select(raw,('sha256','weight_sha256')).items() if is_sha(v)}
    name=raw.get('path') if isinstance(raw,dict) else None
    if isinstance(name,str) and name and not Path(name).is_absolute() and '..' not in Path(name).parts and ':' not in name and '\\' not in name:value['path']=name
    return value


def vectors(raw,width,minimum=2,maximum=5000):
    return isinstance(raw,list) and minimum<=len(raw)<=maximum and all(isinstance(p,list) and len(p)==width and all(type(v) in (int,float) and math.isfinite(v) for v in p) for p in raw)


def compact(raw):
    if not isinstance(raw,dict):return None
    value=select(raw,('version','status','phase','scope','contract_fingerprint','reason','error','elapsed_s','fit_status'))
    value['protocol']=select(raw.get('protocol'),('validation_layouts','test_layouts','methods','maximum_fresh_requests','substeps','noise','no_test_flight_search','predictions_frozen_before_any_evaluation','ensemble_candidates','geometry_screen_top_k','fixed_model_no_validation_selection'))
    dataset=raw.get('dataset',{})
    value['dataset']=select(dataset,('train_requests','train_native_releases','train_labels','train_layouts','source_version','source_contract_fingerprint','source_report_sha256','source_audit_sha256','excluded_prior_layouts','labels_scope','train_split_scope'))
    if isinstance(dataset.get('train_labels'),dict):
        value['dataset']['train_labels']={k:v for k,v in dataset['train_labels'].items() if isinstance(k,str) and type(v) is int and v>=0}
    value['dataset']['split_hashes']=hashes(dataset.get('split_hashes'))
    if isinstance(dataset.get('target_support'),list) and all(type(v) is int and v>=0 for v in dataset['target_support']):value['dataset']['target_support']=copy.deepcopy(dataset['target_support'])
    training=raw.get('training',{})
    value['training']=select(training,('status','ensemble_members','fit_device','fit_steps','input_features','target_features','selection_basis'))
    value['training']['artifact']=artifact(training.get('artifact'))
    value['training']['curves']=[select(r,('member','step','train_loss','full_train_loss')) for r in training.get('curves',[]) if isinstance(r,dict)]
    value['counts']=select(raw.get('counts'),STATS)
    value['phase_counts']={k:select(v,STATS) for k,v in raw.get('phase_counts',{}).items() if k in ('controls_pre','validation','test','controls_post') and isinstance(v,dict)}
    value['prediction_metrics']=[select(r,('split','target','support','mae','rmse','brier','accuracy','unit','scope')) for r in raw.get('prediction_metrics',[]) if isinstance(r,dict)]
    value['test_comparison']=[dict(method=r['method'],counts=select(r.get('counts'),STATS),**select(r,('capture_ci95','ci95_method','artifact_sha256'))) for r in raw.get('test_comparison',[]) if isinstance(r,dict) and r.get('method') in METHODS]
    value['paired_differences']=[select(r,('a','b','a_only','b_only','shared','neither','delta_rate','ci95','interval_method')) for r in raw.get('paired_differences',[]) if isinstance(r,dict) and r.get('a') in METHODS and r.get('b') in METHODS]
    value['test_positions']=[]
    for r in raw.get('test_positions',[]):
        if not isinstance(r,dict) or r.get('method') not in METHODS:continue
        value['test_positions'].append(dict(**select(r,('context_id','method','bucket_xy_m','values','board_bucket_distance_m','admission','artifact_sha256','record_id')),
            model_prediction=select(r.get('model_prediction'),('admission_probability','capture_probability','conditional_capture_probability','uncertainty','aperture_distance_m','retention_fraction','ranking_score','rebound_velocity_m_s','rebound_basis','selection_scope')),
            actual=select(r.get('actual'),('native_release','capture','clean','invalid','invalid_physics','whole_entry','retention_s','entry_class'))))
    value['controls']={k:dict(**select(r,('status','reason')),counts=select(r.get('counts'),STATS)) for k,r in raw.get('controls',raw.get('control_gates',{})).items() if k in ('pre','post') and isinstance(r,dict)}
    value['gate']=select(raw.get('gate'),('status','complete','controls_passed','no_test_flight_search','predictions_frozen_before_any_evaluation','reason','scope'))
    value['source_sha256']=hashes(raw.get('source_sha256'));value['evidence_sha256']=hashes(raw.get('evidence_sha256'))
    queue=raw.get('queue',{})
    value['queue']=select(queue,('status','wait_s'))
    if isinstance(queue.get('reason'),list):value['queue']['reason']=[v for v in queue['reason'] if type(v) is int and v>0 or v=='shared-lock']
    value['trace_examples']=[]
    for r in raw.get('trace_examples',[])[:24]:
        if not isinstance(r,dict) or not vectors(r.get('position_m'),3):continue
        trace=select(r,('id','record_id','context_id','method','bucket_xy_m','values','board_center_m','board_normal','sample_dt_s','capture_time_s','capture','clean','whole_ball_entry','max_retention_s','invalid','invalid_physics','native_cache_sha256','native_cache_row','selection_scope'))
        trace['position_m']=copy.deepcopy(r['position_m'])
        if vectors(r.get('predicted_position_m'),3):trace['predicted_position_m']=copy.deepcopy(r['predicted_position_m'])
        value['trace_examples'].append(trace)
    return value


def full_audit_verified(raw,audit):
    checks=audit.get('checks');count=audit.get('checks_count')
    return bool(raw.get('version')==VERSION and raw.get('status')==raw.get('phase')=='completed' and is_sha(raw.get('contract_fingerprint'))
        and audit.get('version')==AUDIT_VERSION and audit.get('status')=='passed' and audit.get('failures')==[]
        and audit.get('contract_fingerprint')==raw['contract_fingerprint'] and isinstance(checks,list) and checks
        and type(count) is int and count>0 and count==len(checks) and all(isinstance(c,dict) and c.get('passed') is True for c in checks))


def prepare(base_display,study,output,audit_path=None):
    base_display,study,output=map(Path,(base_display,study,output))
    audit_path=Path(audit_path) if audit_path is not None else study.with_name(study.name+'-audit.json')
    if output.resolve() in (base_display.resolve(),audit_path.resolve()) or output.resolve().is_relative_to(study.resolve()):raise ValueError('Separate derived report outside immutable evidence required')
    base=optional_json(base_display)
    if not isinstance(base,dict):raise ValueError('Existing display must be a JSON object')
    try:report_bytes=(study/'report.json').read_bytes();raw=json.loads(report_bytes)
    except (OSError,ValueError):report_bytes,raw=None,None
    value=compact(raw)
    if value is not None:
        try:audit_bytes=audit_path.read_bytes();audit=json.loads(audit_bytes)
        except (OSError,ValueError):audit_bytes,audit=None,{}
        if not isinstance(audit,dict):audit={}
        try:manifest_bytes=(study/'evidence-manifest.json').read_bytes()
        except OSError:manifest_bytes=None
        value['audit']=select(audit,('version','status','contract_fingerprint','checks_count','report_sha256','evidence_manifest_sha256','reason','limitations'))
        files=dict(report_sha256=report_bytes,evidence_manifest_sha256=manifest_bytes)
        bound=all(data is not None and is_sha(audit.get(key)) and sha(data)==audit[key] for key,data in files.items())
        value['audit'].update(bound_files_verified=bound,full_checks_verified=bool(isinstance(raw,dict) and full_audit_verified(raw,audit) and bound))
        for key,data in files.items():
            if data is not None:value['audit']['observed_'+key]=sha(data)
        if audit_bytes is not None:value['audit']['source_sha256']=sha(audit_bytes)
        if audit.get('status')=='passed' and not value['audit']['full_checks_verified']:
            value['audit']['verification_reason']='Exact completed evidence, full strictly passed v1 checklist and matching report/manifest bytes are required.'
        value['display_updated_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
    result=copy.deepcopy(base);result[KEY]=value
    payload=json.dumps(result,indent=2,allow_nan=False)+'\n';output.parent.mkdir(parents=True,exist_ok=True)
    temporary=output.with_suffix(output.suffix+'.tmp');temporary.write_text(payload);temporary.replace(output)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('base-display','study','output'):parser.add_argument('--'+key,type=Path,required=True)
    parser.add_argument('--audit',type=Path)
    a=parser.parse_args();prepare(a.base_display,a.study,a.output,a.audit)


if __name__=='__main__':main()
