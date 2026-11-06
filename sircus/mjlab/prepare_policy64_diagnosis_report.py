"""Publish additive, read-only paired diagnosis with two independently bound proofs."""
import argparse
import copy
import datetime
import hashlib
import json
import math
from pathlib import Path
import re

from .prepare_policy64_report import hashes, safe_path, select

KEY='policy64_diagnosis'
VERSION='sircus-policy64-diagnosis-v1'
AUDIT_VERSION='sircus-policy64-diagnosis-audit-v1'
NATIVE_VERSION='sircus-policy64-paired-native-diagnosis-v1'
PARENT_VERSION='sircus-policy64-bc-ppo-v1'
PARENT_AUDIT_VERSION='sircus-policy64-independent-audit-v1'
PARENT_FP='d30ad91111cb8beb26438f2271357a103daf50d0399ac3646f3b7f163d87f02e'
PARENT_REPORT_SHA='637731856cba291b57a1119b32b56fe97a3c4cbc000d55db4bfb1793c6757cc3'
PARENT_MANIFEST_SHA='632ce6cd692ea68d862243deb07c71cb605994fd017f7f15a603aa7995bd3514'
PARENT_AUDIT_SHA='4dc317e0697e426a4708c3c997f8671d97b1f49de6bd4450c56e6487a6e93ac3'
COUNTS=('layouts','attempts','requested','native','native_releases','capture','captures','clean','rejected','setup_rejections','native_misses','invalid_flights','invalid','invalid_physics','mean_reward','count')
IDENTITY=('id','record_id','record_sha256','source_record_id','source_record_sha256','context_id','method','variant','variant_id','group_id','basis','status','phase','role','selection_rule','label','reason')
GEOMETRY=('values','bucket_xy_m','board_xy_m','board_height_m','board_center_m','board_normal','gap_m','tilt_deg','release_center_height_m','board_bucket_distance_m','distance_m','substeps','native_release','capture','clean','route_success','route_quality','invalid','invalid_physics','whole_ball_entry','max_retention_s','scalar_score_audit_passed','rim_band_contact','native_cache_sha256','native_cache_row')
GEOMETRY_EVIDENCE=('status','stage','reason','passed','starts','iterations','branches','minimum_clearance_m','min_clearance_m','clearance_threshold_m','board_position_error_mm','board_normal_error_deg','dropper_position_error_mm','dropper_error_mm','endpoint_admitted','bounds_passed','fixed_base_passed','collision_pair_labels','minimum_required_clearance_m','checked_branch_pairs','best_checked_clearance_m','limiting_pairs','proof_minimum_clearance_m','endpoint_reason','interpretation')


def read_bytes(path):
    try:
        data=Path(path).read_bytes();value=json.loads(data)
        return data,value if isinstance(value,dict) else None
    except (OSError,ValueError):return None,None


def sha(data):return hashlib.sha256(data).hexdigest() if data is not None else None


def valid_hash(value):return isinstance(value,str) and re.fullmatch(r'[a-f0-9]{64}',value) is not None


def full_checks(proof,version):
    checks,count=proof.get('checks'),proof.get('checks_count')
    return bool(proof.get('version')==version and proof.get('status')=='passed' and proof.get('failures')==[] and isinstance(checks,list) and checks and type(count) is int and count==len(checks) and all(isinstance(c,dict) and c.get('passed') is True for c in checks))


def proof_summary(proof,data):
    result=select(proof,('version','status','contract_fingerprint','checks_count','report_sha256','evidence_manifest_sha256','native_report_sha256','frozen_analysis_sha256','auditor_sha256','reason'))
    if data is not None:result['source_sha256']=sha(data)
    dependencies=proof.get('auditor_identity',{}).get('dependencies_sha256')
    if isinstance(dependencies,dict):result['dependencies_sha256']=hashes(dependencies)
    return result


def flight_metadata(raw):
    if not isinstance(raw,dict):return None
    value=select(raw,('version','available','reason','cutoff_reason','forward_error_m','lateral_error_m','mouth_clearance_m','closest_opening_distance_m','sample_count'))
    if raw.get('first_crossing') is None and 'first_crossing' in raw:value['first_crossing']=None
    elif isinstance(raw.get('first_crossing'),dict):value['first_crossing']=select(raw['first_crossing'],('time_s','interval_index','fraction'))
    for name in ('rebound_position_m','rebound_velocity_m_s','first_contact_position_m','first_contact_velocity_before_m_s'):
        v=raw.get(name)
        if name in raw and v is None:value[name]=None
        elif isinstance(v,list) and len(v)==3 and all(type(x) in (float,int) and math.isfinite(x) for x in v):value[name]=copy.deepcopy(v)
    return value


def side(raw):
    value=select(raw,IDENTITY+GEOMETRY+('outcome','setup_stage','source_hint_id','source_hint_sha256','normalized_residual','learning_reward','board_bounce','closest_opening_distance_m','external_contact_steps','board_contact_episodes','connection_sha256','source_teacher_id','bucket_contact_before_entry','entry_class','minimum_mouth_clearance_mm','whole_ball_entry_time_s','capture_time_s','retention_resets','first_mouth_passage_time_s'))
    value['geometry_evidence']=select(raw.get('geometry_evidence',{}),GEOMETRY_EVIDENCE) if isinstance(raw,dict) else {}
    if isinstance(raw,dict) and isinstance(raw.get('first_flight'),dict):value['first_flight']=flight_metadata(raw['first_flight'])
    return value


def trace(raw):
    if not isinstance(raw,dict):return None
    points=raw.get('position_m',raw.get('trajectory_m'))
    if not isinstance(points,list) or not 2<=len(points)<=5000 or not all(isinstance(p,list) and len(p)==3 and all(type(v) in (int,float) and math.isfinite(v) for v in p) for p in points):return None
    value=select(raw,IDENTITY+GEOMETRY+('sample_dt_s','trajectory_dt_s','selection_scope','provenance'))
    if isinstance(raw.get('first_flight'),dict):value['first_flight']=flight_metadata(raw['first_flight'])
    value['position_m']=copy.deepcopy(points)
    if 'sample_dt_s' not in value and 'trajectory_dt_s' in value:value['sample_dt_s']=value.pop('trajectory_dt_s')
    cache=safe_path(raw.get('native_cache_file'))
    if cache is not None:value['native_cache_file']=cache
    return value


def example(raw):
    value=select(raw,IDENTITY+('context_id','bucket_xy_m','interpretation','new_native_requests'))
    for method in ('bc','ppo'):
        t=trace(raw.get(method+'_trace'))
        if t is not None:value[method+'_trace']=t
    t=trace(raw)
    if t is not None:value['trace']=t
    value['projection_reference']=select(raw.get('projection_reference',{}),('origin_xy_m','axis_xy','direction_from_xy_m','reference_method','reference_record_id'))
    return value


def compact(raw):
    if not isinstance(raw,dict):return None
    result=select(raw,('version','status','phase','scope','error','reason','contract_fingerprint','analysis_code_sha256'))
    result['source']=select(raw.get('source',{}),('study_version','contract_fingerprint','report_sha256','manifest_sha256','original_audit_sha256','substeps'))
    result['source']['selected']=select(raw.get('source',{}).get('selected',{}),('samples','sha256','weight_sha256'))
    pc=raw.get('paired_counts',{})
    result['paired_counts']=select(pc,('layouts','both_capture','bc_only','ppo_only','neither','admission_cohorts'))
    result['paired_counts']['jointly_admitted']=select(pc.get('jointly_admitted',{}),('layouts','bc_captures','ppo_captures','caveat'))
    for method in ('bc','ppo'):result['paired_counts'][method]=select(pc.get(method,{}),COUNTS)
    result['paired_counts']['transitions']=[select(r,('bc_outcome','ppo_outcome','count')) for r in pc.get('transitions',[]) if isinstance(r,dict)]
    result['delta_summary']=[select(r,('group','layouts','parameters','mean_delta','mean_abs_delta','max_abs_delta','min_delta','max_delta')) for r in raw.get('delta_summary',[]) if isinstance(r,dict)]
    result['actor_shift']=select(raw.get('actor_shift',{}),('board_height_lower_count','release_gap_higher_count'))
    result['rejection_reasons']=[select(r,('method','reason','count','context_ids','interpretation')) for r in raw.get('rejection_reasons',[]) if isinstance(r,dict)]
    result['pairs']=[]
    for r in raw.get('pairs',[]):
        if not isinstance(r,dict):continue
        value=select(r,('context_id','bucket_xy_m','category','delta_values','delta_release_height_m'))
        value.update(bc=side(r.get('bc',{})),ppo=side(r.get('ppo',{})),shared_anchor=select(r.get('shared_anchor',{}),('same_id','same_values','same_observation','bc_id','ppo_id','bc_sha256','ppo_sha256')))
        result['pairs'].append(value)
    result['examples']=[example(r) for r in raw.get('examples',[]) if isinstance(r,dict)]
    result['counterfactuals']=[]
    for r in raw.get('counterfactuals',[]):
        if not isinstance(r,dict):continue
        value=select(r,IDENTITY+GEOMETRY+('context_id','geometry_passed','source_record_ids','new_native_requests','interpretation'))
        value['intervention']=select(r.get('intervention',{}),('copied_parameters','from_method','to_method','description'))
        result['counterfactuals'].append(value)
    native=raw.get('native_study')
    if isinstance(native,dict):
        n=select(native,('version','status','phase','plan_requests','observed_requests','scope','error','reason','contract_fingerprint','analysis_sha256','substeps','training','reported_worker_elapsed_s'))
        n['counts']=select(native.get('counts',{}),COUNTS)
        controls=native.get('controls',native.get('control_gates',{}))
        n['controls']={k:dict(select(v,COUNTS+('status',)),counts=select(v.get('counts',{}),COUNTS)) for k,v in controls.items() if k in ('pre','post') and isinstance(v,dict)} if isinstance(controls,dict) else {}
        n['comparison']=[dict(side(r),counts=select(r.get('counts',r.get('stats',{})),COUNTS)) for r in native.get('comparison',[]) if isinstance(r,dict)]
        n['variants']=[dict(side(r),counts=select(r.get('counts',r.get('stats',{})),COUNTS)) for r in native.get('variants',[]) if isinstance(r,dict)]
        for context in native.get('contexts',[]):
            if isinstance(context,dict) and isinstance(context.get('variants'),dict):
                n['variants'] += [dict(context_id=context.get('context_id'),variant=key,counts=select(v,COUNTS)) for key,v in context['variants'].items() if isinstance(v,dict)]
        n['examples']=[example(r) for r in native.get('examples',[]) if isinstance(r,dict)]
        result['native_study']=n
    model=raw.get('model_options',{})
    if isinstance(model,dict):
        result['model_options']=select(model,('title','kind','scope','note','status','current','recommendation','feasibility','inputs','targets','evaluation','wam'))
        result['model_options']['options']=[select(v,('id','label','kind','description','scope')) for v in model.get('options',[]) if isinstance(v,dict)]
        result['model_options']['sources']=[select(v,('title','url')) for v in model.get('sources',[]) if isinstance(v,dict) and isinstance(v.get('url'),str) and re.match(r'^https://[A-Za-z0-9.-]+/',v['url'])]
    history=raw.get('optimization_history',{})
    result['optimization_history']=select(history,('bc_retention_penalty','caveat','mean_post_update_kl','mean_post_update_ratio_clip_fraction','metric_definition'))
    result['optimization_history']['updates']=[select(v,('index','samples','analytical_kl_mean','analytical_kl_max','sampled_ratio_clip_fraction','critic_mse_before','critic_mse_after','standard_deviation_before','standard_deviation_after','gradient_norm_max','before_sha256','after_sha256')) for v in history.get('updates',[]) if isinstance(v,dict)]
    result['hypotheses']=[select(v,('statement','evidence','status')) for v in raw.get('hypotheses',[]) if isinstance(v,dict)]
    return result


def prepare(base_display,report,manifest,audit,parent_report,parent_manifest,parent_audit,output):
    paths=[Path(p) for p in (base_display,report,manifest,audit,parent_report,parent_manifest,parent_audit,output)]
    base_display,report,manifest,audit,parent_report,parent_manifest,parent_audit,output=paths
    if output.resolve() in {p.resolve() for p in paths[:-1]} or output.resolve().is_relative_to(manifest.parent.resolve()) or output.resolve().is_relative_to(parent_report.parent.resolve()):raise ValueError('Separate derived output outside native/original evidence required')
    _,base=read_bytes(base_display)
    if not isinstance(base,dict):raise ValueError('Valid existing display required')
    report_bytes,raw=read_bytes(report);manifest_bytes,_=read_bytes(manifest);audit_bytes,proof=read_bytes(audit);proof=proof or {}
    pbytes,parent=read_bytes(parent_report);mbytes,_=read_bytes(parent_manifest);abytes,pproof=read_bytes(parent_audit);pproof=pproof or {}
    value=compact(raw)
    if value is not None:
        source=raw.get('source',{})
        parent_bound=sha(pbytes)==PARENT_REPORT_SHA==pproof.get('report_sha256') and sha(mbytes)==PARENT_MANIFEST_SHA==pproof.get('evidence_manifest_sha256') and sha(abytes)==PARENT_AUDIT_SHA
        parent_verified=bool(parent_bound and isinstance(parent,dict) and parent.get('version')==PARENT_VERSION and parent.get('status')==parent.get('phase')=='completed' and parent.get('contract_fingerprint')==PARENT_FP==pproof.get('contract_fingerprint') and full_checks(pproof,PARENT_AUDIT_VERSION) and source.get('study_version')==PARENT_VERSION and source.get('contract_fingerprint')==PARENT_FP and source.get('report_sha256')==PARENT_REPORT_SHA and source.get('manifest_sha256')==PARENT_MANIFEST_SHA and source.get('original_audit_sha256')==PARENT_AUDIT_SHA and source.get('substeps')==64)
        value['source']['audit']=proof_summary(pproof,abytes)
        value['source']['audit'].update(full_checks_verified=parent_verified,bound_files_verified=parent_bound)
        bound=bool(report_bytes is not None and manifest_bytes is not None and proof.get('report_sha256')==sha(report_bytes) and proof.get('evidence_manifest_sha256')==sha(manifest_bytes))
        verified=bool(parent_verified and bound and raw.get('version')==VERSION and raw.get('status')==raw.get('phase')=='completed' and raw.get('contract_fingerprint')==proof.get('contract_fingerprint') and full_checks(proof,AUDIT_VERSION) and valid_hash(proof.get('auditor_sha256')) and valid_hash(proof.get('native_report_sha256')) and valid_hash(proof.get('frozen_analysis_sha256')))
        value['audit']=proof_summary(proof,audit_bytes)
        value['audit'].update(full_checks_verified=verified,bound_files_verified=bound,parent_source_verified=parent_verified)
        if 'native_study' in value:
            n=value['native_study'];observed=n.get('observed_requests',n.get('counts',{}).get('attempts'))
            native_verified=bool(verified and n.get('version')==NATIVE_VERSION and n.get('status')==n.get('phase')=='completed' and observed==96)
            n['audit']=dict(value['audit'],full_checks_verified=native_verified)
        value['display_updated_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
    result=copy.deepcopy(base);result[KEY]=value
    payload=json.dumps(result,indent=2,allow_nan=False)+'\n';output.parent.mkdir(parents=True,exist_ok=True)
    temporary=output.with_suffix(output.suffix+'.tmp');temporary.write_text(payload);temporary.replace(output)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('base-display','report','manifest','audit','parent-report','parent-manifest','parent-audit','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();prepare(**{k:getattr(a,k) for k in vars(a)})
