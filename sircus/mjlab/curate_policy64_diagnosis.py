"""Export sanitized, byte-bound post-hoc BC/PPO diagnostic evidence.

Standard library only: this reads a completed independent proof; it does not
execute physics, rescore contacts, load checkpoints, or change source evidence.
"""
from collections import Counter
import argparse
import copy
import csv
import json
import math
from pathlib import Path
from urllib.parse import urlsplit

from .curate_xy650 import boolean, digest, fingerprint, hash_map, identifier, number, read, relative_file, require, sha, write_json

VERSION = 'sircus-policy64-diagnosis-curation-v1'
STUDY_VERSION = 'sircus-policy64-diagnosis-v1'
NATIVE_VERSION = 'sircus-policy64-paired-native-diagnosis-v1'
AUDIT_VERSION = 'sircus-policy64-diagnosis-audit-v1'
PARENT_VERSION = 'sircus-policy64-bc-ppo-v1'
PARENT_FP = 'd30ad91111cb8beb26438f2271357a103daf50d0399ac3646f3b7f163d87f02e'
PARENT_REPORT_SHA = '637731856cba291b57a1119b32b56fe97a3c4cbc000d55db4bfb1793c6757cc3'
PARENT_MANIFEST_SHA = '632ce6cd692ea68d862243deb07c71cb605994fd017f7f15a603aa7995bd3514'
PARENT_AUDIT_SHA = '4dc317e0697e426a4708c3c997f8671d97b1f49de6bd4450c56e6487a6e93ac3'
CONTEXTS = tuple('policy64-test-'+v for v in ('04','07','13','17','21','26','43'))
VARIANTS = ('bc_exact','ppo_exact','swap_x','swap_y','swap_height','swap_tilt','swap_gap','swap_height_gap')
PARAMETERS = ('board_x_m','board_y_m','board_height_m','board_tilt_deg','nominal_plane_gap_m')
STAT_KEYS = ('attempts','native_releases','captures','clean','rejected','invalid_physics')
SCOPE = ('Post-hoc development diagnosis of the original paired50 TEST layouts and seven selected BC-only losses. '
    'The96 nominal64 requests include84 diagnostic slots and12 matched controls; controls are not candidate shots. '
    'Coordinate swaps are interventions, not new training or a fresh generalization test. Historical policies, '
    'scores and128-substep failures remain unchanged. No noise robustness, driven transit, perception or physical validation.')


def vector(value, size):
    require(isinstance(value,list) and len(value)==size,'Complete finite vector required')
    return [number(v) for v in value]


def captured(row):
    native=boolean(row.get('native_release'));boolean(row.get('invalid_physics'))
    success=boolean(row.get('route_success'))
    require(type(row.get('route_quality')) is int and row['route_quality'] in (0,1,2),'Capture tier missing')
    require(not success or native and not row.get('invalid') and not row['invalid_physics'],'Invalid capture claim')
    return bool(native and success and not row.get('invalid') and not row['invalid_physics'])


def counts(rows):
    return dict(attempts=len(rows),native_releases=sum(boolean(r['native_release']) for r in rows),
        captures=sum(captured(r) for r in rows),clean=sum(captured(r) and r['route_quality']==2 for r in rows),
        rejected=sum(not r['native_release'] for r in rows),invalid_physics=sum(boolean(r['invalid_physics']) for r in rows))


def control_gate(rows):
    pos=[r for r in rows if r['role']=='positive_control'];neg=[r for r in rows if r['role']=='negative_control']
    good=(len(rows)==6 and len(pos)==len(neg)==3 and all(captured(r) for r in pos)
        and all(r['native_release'] and not captured(r) and not r.get('whole_ball_entry') for r in neg)
        and all(not r.get('invalid') and not r['invalid_physics'] and r.get('scalar_score_audit_passed') is True for r in rows))
    return dict(status='passed' if good else 'failed',counts=counts(rows))


def strict_audit(audit):
    checks=audit.get('checks')
    require(audit.get('version')==AUDIT_VERSION and audit.get('status')=='passed' and audit.get('failures')==[]
        and isinstance(checks,list) and bool(checks) and type(audit.get('checks_count')) is int
        and audit['checks_count']==len(checks)
        and all(isinstance(c,dict) and c.get('passed') is True and isinstance(c.get('label'),str) for c in checks),
        'Complete exact-version v1 independent audit required')


def verified_inputs(report_path,audit_path,diagnosis):
    report_path,audit_path,diagnosis=map(lambda p:Path(p).resolve(),(report_path,audit_path,diagnosis))
    combined,audit,spec,native,frozen=(read(p) for p in
        (report_path,audit_path,diagnosis/'contract.json',diagnosis/'report.json',diagnosis/'analysis.json'))
    strict_audit(audit)
    for field,path in (('report_sha256',report_path),('analysis_report_sha256',report_path),
        ('native_report_sha256',diagnosis/'report.json'),('frozen_analysis_sha256',diagnosis/'analysis.json'),
        ('evidence_manifest_sha256',diagnosis/'evidence-manifest.json')):
        require(audit.get(field)==sha(path),'Audit byte binding differs: '+field)
    require(combined.get('version')==frozen.get('version')==STUDY_VERSION,'Unexpected analysis version')
    require(spec.get('version')==native.get('version')==NATIVE_VERSION,'Unexpected native version')
    require(combined.get('status')==combined.get('phase')==native.get('status')==native.get('phase')=='completed','Completed96-slot diagnosis required')
    unsigned=dict(spec);fp=unsigned.pop('fingerprint',None)
    require(fingerprint(fp)==digest(unsigned)==combined.get('contract_fingerprint')==native.get('contract_fingerprint')==audit.get('contract_fingerprint'),'Contract fingerprint differs')
    require(spec.get('analysis_sha256')==native.get('analysis_sha256')==sha(diagnosis/'analysis.json'),'Frozen analysis binding differs')
    manifest=hash_map(read(diagnosis/'evidence-manifest.json'))
    require(native.get('evidence_sha256')==manifest and manifest.get('contract.json')==sha(diagnosis/'contract.json'),'Native manifest binding differs')
    for name,value in manifest.items():require(sha(relative_file(diagnosis,name))==value,'Changed evidence bytes: '+name)
    actual={str(p.relative_to(diagnosis)) for p in diagnosis.rglob('*') if p.is_file() and str(p.relative_to(diagnosis)) not in ('report.json','evidence-manifest.json')}
    require(set(manifest)==actual,'Manifest must cover every frozen artifact')
    expected=copy.deepcopy(frozen);expected.update(contract_fingerprint=fp,phase='completed',status='completed',native_study=native,evidence_sha256=manifest)
    require(combined==expected,'Combined report is not the exact immutable analysis/native merge')
    parent=dict(contract_fingerprint=PARENT_FP,report_sha256=PARENT_REPORT_SHA,manifest_sha256=PARENT_MANIFEST_SHA,original_audit_sha256=PARENT_AUDIT_SHA)
    source=frozen.get('source',{})
    require(source.get('study_version')==PARENT_VERSION and source.get('substeps')==64
        and all(source.get(k)==v for k,v in parent.items()) and audit.get('source_parent')==parent
        and audit.get('source')==source,'Exact original parent identity differs')
    bindings=spec.get('parent_bindings',{})
    for k,v in (('parent_contract_fingerprint',PARENT_FP),('parent_report_sha256',PARENT_REPORT_SHA),
        ('parent_manifest_sha256',PARENT_MANIFEST_SHA),('parent_audit_sha256',PARENT_AUDIT_SHA),('analysis_sha256',sha(diagnosis/'analysis.json'))):
        require(bindings.get(k)==v,'Native parent binding differs: '+k)
    require(audit.get('producer_source_sha256')==hash_map(spec.get('diagnostic_source_sha256'))
        and audit.get('native_source_sha256')==hash_map(spec.get('native_source_sha256')),'Audited code identities differ')
    fingerprint(audit.get('auditor_sha256'))
    for key,value in dict(substeps=64,maximum_requests=96,maximum_acquired_s=3600,training=False,noise=False,contexts=[c.removeprefix('policy64-test-') for c in CONTEXTS]).items():
        require(spec.get(key)==value,'Literal finite diagnostic protocol differs: '+key)
    require(native.get('training') is False and native.get('substeps')==64,'Unexpected training or timestep')
    rows=[json.loads(line) for line in (diagnosis/'records.jsonl').read_text().splitlines() if line]
    require(rows==[json.loads(line) for line in (diagnosis/'native/shots.jsonl').read_text().splitlines() if line],'Native request logs differ')
    require(len(rows)==96 and type(audit.get('requested_trials')) is int and audit['requested_trials']==96
        and len({r['id'] for r in rows})==96,'Exactly96 unique requested slots required')
    plan=spec.get('plan');require(isinstance(plan,list) and len(plan)==96 and [r['id'] for r in rows]==[r['id'] for r in plan],'Exact planned request order differs')
    for row,request in zip(rows,plan):
        identifier(row['id']);vector(row['values'],5);vector(row['bucket_xy_m'],2)
        require(all(row.get(k)==v for k,v in request.items()),'Requested plan changed')
        require(row.get('substeps')==64 and row.get('selection_or_learning') is False,'Learning or timestep substitution appeared')
    require(native.get('counts')==audit.get('counts')==counts(rows) and audit.get('native_releases')==counts(rows)['native_releases'],'Native accounting differs')
    elapsed=number(native.get('reported_worker_elapsed_s'));require(0<=elapsed<=3600,'Worker ceiling exceeded')
    require(native.get('gate')==dict(status='passed',controls_passed=True,complete=True,
        scope='Diagnostic execution completion only; no new policy success-rate claim'),'Execution-completion gate differs')
    return diagnosis,combined,audit,spec,native,manifest,rows


def paired_checks(report,audit):
    pairs=report.get('pairs');require(isinstance(pairs,list) and len(pairs)==50 and len({r['context_id'] for r in pairs})==50,'Exactly50 unique paired layouts required')
    record_ids=set();transitions=Counter();admission=Counter();reasons=Counter()
    method_stats={}
    for pair in pairs:
        identifier(pair['context_id']);vector(pair['bucket_xy_m'],2)
        for method in ('bc','ppo'):
            r=pair[method];identifier(r['record_id']);fingerprint(r['record_sha256'])
            require(r['record_id'] not in record_ids,'Duplicate source record');record_ids.add(r['record_id'])
            vector(r['values'],5);native=boolean(r['native_release']);capture=boolean(r['capture']);clean=boolean(r['clean']);boolean(r['invalid_physics'])
            expected='rejected' if not native else 'invalid' if r.get('invalid') or r['invalid_physics'] else 'capture' if capture else 'miss'
            require(r['outcome']==expected and (not clean or capture) and (not capture or native and not r.get('invalid') and not r['invalid_physics']),'Paired outcome inconsistency')
            if not native:reasons[method,identifier(r['invalid'])]+=1
        delta=[p-b for b,p in zip(pair['bc']['values'],pair['ppo']['values'])]
        require(all(abs(a-b)<=1e-12 for a,b in zip(vector(pair['delta_values'],5),delta))
            and abs(number(pair['delta_release_height_m'])-delta[2]-delta[4])<=1e-12,'Paired action delta differs')
        b,p=pair['bc'],pair['ppo'];transitions[b['outcome'],p['outcome']]+=1
        admission['both_native' if b['native_release'] and p['native_release'] else 'bc_only_native' if b['native_release'] else 'ppo_only_native' if p['native_release'] else 'neither_native']+=1
    for method in ('bc','ppo'):
        cohort=[r[method] for r in pairs]
        method_stats[method]=dict(requested=50,native=sum(r['native_release'] for r in cohort),captures=sum(r['capture'] for r in cohort),clean=sum(r['clean'] for r in cohort),
            rejected=sum(not r['native_release'] for r in cohort),native_misses=sum(r['outcome']=='miss' for r in cohort),invalid=sum(r['outcome']=='invalid' for r in cohort),
            mean_reward=math.fsum(number(r['learning_reward']) for r in cohort)/50)
    declared=report['paired_counts'];require(declared==audit.get('paired_counts'),'Independent paired accounting differs')
    for method,stats in method_stats.items():
        for key,value in stats.items():require(abs(number(declared[method].get(key))-value)<=1e-10,'Paired count/reward differs: '+method+'/'+key)
    expected=dict(layouts=50,both_capture=sum(r['bc']['capture'] and r['ppo']['capture'] for r in pairs),bc_only=sum(r['bc']['capture'] and not r['ppo']['capture'] for r in pairs),
        ppo_only=sum(not r['bc']['capture'] and r['ppo']['capture'] for r in pairs),neither=sum(not r['bc']['capture'] and not r['ppo']['capture'] for r in pairs),
        transitions=[dict(bc_outcome=b,ppo_outcome=p,count=n) for (b,p),n in sorted(transitions.items())],admission_cohorts=dict(admission))
    require(all(declared.get(k)==v for k,v in expected.items()),'Paired transition accounting differs')
    require({(r['method'],r['reason']):r['count'] for r in report['rejection_reasons']}==dict(reasons),'Rejection accounting differs')
    return pairs


def native_checks(native,rows):
    require(Counter(r['phase'] for r in rows)==Counter(diagnostic=84,controls_pre=6,controls_post=6),'84 diagnostics and12 separate controls required')
    gates={k:control_gate([r for r in rows if r['phase']==phase]) for k,phase in (('pre','controls_pre'),('post','controls_post'))}
    require(native.get('control_gates')==gates and all(g['status']=='passed' for g in gates.values()),'Matched control gates differ')
    contexts=[];variants=[]
    for context in CONTEXTS:
        group={}
        for variant in VARIANTS:
            cohort=[r for r in rows if r['phase']=='diagnostic' and r['context_id']==context and r['variant']==variant]
            size=3 if variant.endswith('_exact') else 1
            require(len(cohort)==size and {r['repeat'] for r in cohort}==set(range(size)) and all(r['role']=='diagnostic' for r in cohort),'Prescribed context/variant repeats differ')
            group[variant]=counts(cohort)
            variants.append(dict(context_id=context,variant=variant,**group[variant],historical_rejection_slots=sum(r['kind']=='historical_rejection' for r in cohort)))
        contexts.append(dict(context_id=context,variants=group))
    require(native.get('contexts')==contexts,'Native context/variant accounting differs')
    require(sum(v['attempts'] for v in variants)==84,'Diagnostic variant denominator differs')
    return gates,variants


def safe_text(value):
    require(isinstance(value,str) and not any(v in value for v in ('/Users/','/home/','file://','Bearer ')),'Host path or credential-like text is not publishable')
    return value


def model_options(raw):
    require(raw.get('status')=='proposed_not_trained','No new model was trained here')
    result={k:safe_text(raw[k]) for k in ('status','current','recommendation','feasibility','evaluation','wam') if k in raw}
    for key in ('inputs','targets'):result[key]=[safe_text(v) for v in raw.get(key,[])]
    sources=[]
    for row in raw.get('sources',[]):
        url=urlsplit(row.get('url',''));require(url.scheme=='https' and bool(url.netloc) and not url.username and not url.password,'Public HTTPS model source required')
        sources.append(dict(title=safe_text(row['title']),url=row['url']))
    result['sources']=sources;return result


def optimizer_summary(raw):
    updates=raw.get('updates');require(isinstance(updates,list) and len(updates)==32 and [r['index'] for r in updates]==list(range(32)),'Exactly32 scalar optimizer updates required')
    output=[]
    for row in updates:
        out={k:number(row[k]) for k in ('index','samples','analytical_kl_mean','analytical_kl_max','sampled_ratio_clip_fraction','critic_mse_before','critic_mse_after','gradient_norm_max')}
        for key in ('before_sha256','after_sha256'):out[key]=fingerprint(row[key])
        output.append(out)
    mean_kl=math.fsum(r['analytical_kl_mean'] for r in output)/32
    mean_clip=math.fsum(r['sampled_ratio_clip_fraction'] for r in output)/32
    for key,value in (('mean_post_update_kl',mean_kl),('mean_post_update_ratio_clip_fraction',mean_clip)):
        if key in raw:require(abs(number(raw[key])-value)<=1e-10,'Optimizer mean differs from32 saved updates')
    return dict(updates=output,bc_retention_penalty=boolean(raw['bc_retention_penalty']),
        mean_post_update_kl=mean_kl,mean_post_update_ratio_clip_fraction=mean_clip,
        metric_definition=safe_text(raw['metric_definition']),caveat=safe_text(raw['caveat']))


def paired_summary(raw):
    result={k:number(raw[k]) for k in ('layouts','both_capture','bc_only','ppo_only','neither')}
    for method in ('bc','ppo'):
        result[method]={k:number(raw[method][k]) for k in ('requested','native','captures','clean','rejected','native_misses','invalid','mean_reward')}
    result['transitions']=[dict(bc_outcome=identifier(r['bc_outcome']),ppo_outcome=identifier(r['ppo_outcome']),count=number(r['count'])) for r in raw['transitions']]
    result['admission_cohorts']={k:number(v) for k,v in raw['admission_cohorts'].items()}
    joint=raw.get('jointly_admitted',{})
    result['jointly_admitted']={k:number(joint[k]) for k in ('layouts','bc_captures','ppo_captures')}
    result['jointly_admitted']['caveat']='Conditional descriptive cohort; rejected requests remain in the primary50 denominator.'
    return result


def pair_csv(pair):
    out=dict(context_id=identifier(pair['context_id']),bucket_x_m=number(pair['bucket_xy_m'][0]),bucket_y_m=number(pair['bucket_xy_m'][1]))
    for method in ('bc','ppo'):
        raw=pair[method];geometry=raw.get('geometry_evidence') or {};flight=raw.get('first_flight') or {}
        out.update({method+'_'+k:number(v) for k,v in zip(PARAMETERS,raw['values'])})
        for key in ('record_id','source_teacher_id'):
            out[method+'_'+key]=identifier(raw[key]) if raw.get(key) is not None else None
        out[method+'_record_sha256']=fingerprint(raw['record_sha256'])
        for key in ('outcome','invalid','setup_stage','entry_class'):
            out[method+'_'+key]=identifier(raw[key]) if raw.get(key) is not None else None
        for key in ('native_release','capture','clean','invalid_physics','board_bounce','whole_ball_entry','rim_band_contact','bucket_contact_before_entry'):
            out[method+'_'+key]=boolean(raw[key]) if raw.get(key) is not None else None
        for key in ('learning_reward','max_retention_s','minimum_mouth_clearance_mm','whole_ball_entry_time_s','capture_time_s','retention_resets','first_mouth_passage_time_s'):
            out[method+'_'+key]=number(raw.get(key),optional=True)
        for key in ('minimum_required_clearance_m','checked_branch_pairs','best_checked_clearance_m','proof_minimum_clearance_m'):
            out[method+'_'+key]=number(geometry.get(key),optional=True)
        for key in ('forward_error_m','lateral_error_m','mouth_clearance_m','closest_opening_distance_m'):
            out[method+'_first_flight_'+key]=number(flight.get(key),optional=True)
    out.update({'delta_'+k:number(v) for k,v in zip(PARAMETERS,pair['delta_values'])})
    out['delta_release_height_m']=number(pair['delta_release_height_m']);return out


def write_csv(path,rows):
    with path.open('x',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]),lineterminator='\n');writer.writeheader();writer.writerows(rows)


def curate(report_path,audit_path,diagnosis,output):
    diagnosis,report,audit,spec,native,manifest,rows=verified_inputs(report_path,audit_path,diagnosis)
    output=Path(output).resolve()
    require(not output.exists() and not output.is_relative_to(diagnosis) and not diagnosis.is_relative_to(output),'New output directory outside frozen evidence required')
    require(not any(Path(p).resolve().is_relative_to(output) for p in (report_path,audit_path)),'Output cannot contain input reports')
    pairs=paired_checks(report,audit);gates,variants=native_checks(native,rows)
    require(audit.get('control_gates')==gates,'Independent control accounting differs')
    delta=[]
    for row in report['delta_summary']:
        require(row.get('parameters')==list(PARAMETERS),'Delta parameter order differs')
        delta.append(dict(group=identifier(row['group']),layouts=number(row['layouts']),parameters=list(PARAMETERS),
            **{k:vector(row[k],5) for k in ('mean_delta','mean_abs_delta','max_abs_delta','min_delta','max_delta')}))
    reasons=[dict(method=identifier(r['method']),reason=identifier(r['reason']),count=number(r['count']),context_ids=[identifier(v) for v in r['context_ids']]) for r in report['rejection_reasons']]
    source=report['source'];source_summary={k:source[k] for k in ('study_version','contract_fingerprint','report_sha256','manifest_sha256','original_audit_sha256','substeps')}
    for key in ('selected','bc'):
        source_summary[key]={k:source[key][k] for k in ('samples','sha256','weight_sha256') if k in source.get(key,{})}
    summary=dict(version=VERSION,study_version=STUDY_VERSION,native_version=NATIVE_VERSION,status='completed',scope=SCOPE,
        paired_counts=paired_summary(report['paired_counts']),delta_summary=delta,rejection_reasons=reasons,
        actor_shift={k:number(report['actor_shift'][k]) for k in ('board_height_lower_count','release_gap_higher_count')},
        optimization_history=optimizer_summary(report['optimization_history']),model_options=model_options(report['model_options']),source=source_summary,
        native_study=dict(counts=counts(rows),diagnostic_counts=counts([r for r in rows if r['phase']=='diagnostic']),controls=gates,
            contexts=[dict(context_id=c,variants={v:counts([r for r in rows if r['phase']=='diagnostic' and r['context_id']==c and r['variant']==v]) for v in VARIANTS}) for c in CONTEXTS],
            nominal_repeats=3,substeps=64,training=False,reported_worker_elapsed_s=number(native['reported_worker_elapsed_s'])),
        provenance=dict(contract_fingerprint=spec['fingerprint'],combined_report_sha256=sha(report_path),native_report_sha256=sha(diagnosis/'report.json'),
            frozen_analysis_sha256=sha(diagnosis/'analysis.json'),evidence_manifest_sha256=sha(diagnosis/'evidence-manifest.json'),independent_audit_sha256=sha(audit_path),
            audit_version=AUDIT_VERSION,audit_checks=audit['checks_count'],auditor_sha256=audit['auditor_sha256'],
            producer_source_sha256=hash_map(spec['diagnostic_source_sha256']),native_source_sha256=hash_map(spec['native_source_sha256'])))
    # Validate all row exports before making any destination.
    csv_pairs=[pair_csv(p) for p in pairs]
    output.mkdir(parents=True,exist_ok=False);write_json(output/'summary.json',summary)
    write_csv(output/'pairs.csv',csv_pairs);write_csv(output/'native-variants.csv',variants)
    checksums={p.name:sha(p) for p in sorted(output.iterdir()) if p.is_file()}
    write_json(output/'checksums.json',checksums)
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('report','audit','diagnosis','output'):parser.add_argument('--'+key,type=Path,required=True)
    args=parser.parse_args();result=curate(args.report,args.audit,args.diagnosis,args.output)
    print(json.dumps(dict(version=VERSION,status=result['status'],pairs=result['paired_counts']['layouts'],native=result['native_study']['counts'])))


if __name__=='__main__':main()
