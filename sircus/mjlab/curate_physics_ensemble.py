"""Export sanitized results of the exact audited physics-ensemble study.

Standard library only. No Torch/pickle loading, fitting, inference, contact
rescoring or evidence mutation occurs here. The external independent native
audit remains the authority for physics and checkpoint contents.
"""
from collections import Counter
import argparse
import csv
import json
import math
from pathlib import Path

from .curate_xy650 import boolean, digest, fingerprint, hash_map, identifier, number, read, relative_file, require, sha, write_json

VERSION='sircus-physics-ensemble-curation-v1'
STUDY_VERSION='sircus-physics-ensemble-v1'
AUDIT_VERSION='sircus-physics-ensemble-audit-v1'
MODEL_VERSION='sircus-physics-ensemble-model-v1'
SOURCE_FP='90a7d5b7e3d1fb1e738a36a7cf30930116516da0d6d6435c06b98eb97be8c376'
REPORT_SHA='9d76ed2225f5e9a659d5b8ef18839207553df87e14980cf0a290707de437b5a6'
MANIFEST_SHA='7b4483aff3d99691b0c0712b8805f7718d1738ac13b15c50c9e2d0cdb373752c'
AUDIT_SHA='72e5cc9fd4c424896a0c41588b3eec0bd85317ce931cd76e90bd850474f87ebb'
ARTIFACT_SHA=dict(ensemble='0f8d00e59d7b9c07056a74b9f05f8a850ec46603b67a6f96489c90e31ebd40b4',
    bc='1b201dc48101078ed641b614b89115440aa9725cea567fb2327ba71b9679b51d',
    ppo='d42f683e0fd79dd0ee6e01c042abc9df9d49273622a7d50a1026733dea64dcd0')
PARENT=dict(contract_fingerprint='d30ad91111cb8beb26438f2271357a103daf50d0399ac3646f3b7f163d87f02e',
    report_sha256='637731856cba291b57a1119b32b56fe97a3c4cbc000d55db4bfb1793c6757cc3',
    manifest_sha256='632ce6cd692ea68d862243deb07c71cb605994fd017f7f15a603aa7995bd3514',
    original_audit_sha256='4dc317e0697e426a4708c3c997f8671d97b1f49de6bd4450c56e6487a6e93ac3')
METHODS=('bc','ppo','ensemble')
PHASES=('controls_pre','validation','test','controls_post')
PARAMETERS=('board_x_m','board_y_m','board_height_m','board_tilt_deg','nominal_plane_gap_m')
INPUTS=('board_x_m','board_y_m','board_height_m','tilt_deg','gap_m','bucket_x_m','bucket_y_m')
TARGETS=('native_execution','board_bounce','whole_ball_entry','capture','retention_fraction',
    'log_first_flight_opening_distance','rebound_forward_m_s','rebound_lateral_m_s','rebound_vertical_m_s')
PREDICTIONS=('admission_probability','board_bounce_probability','whole_entry_probability','capture_probability',
    'conditional_capture_probability','uncertainty','retention_fraction','aperture_distance_m','ranking_score')
SCOPE=('Fixed64 state-based physics ensemble fitted on1024 existing parent TRAIN requests, with no new TRAIN acquisition. '
    'All252 predictions were frozen before evaluation:24VAL and60TEST layouts, three methods each, plus12 separate controls. '
    'Five bootstrapped networks rank16 model proposals and screen at most4 geometry proofs; saved BC/PPO predict directly. '
    'This unequal-training/inference-budget system comparison does not establish an architecture winner. Model scores are '
    'uncalibrated and disagreement is not a reliability guarantee. One nominal shot per method/layout; no native flight search, '
    'noise robustness,128-substep qualification, point-cloud/video WAM, driven arm transit, grasping or physical validation. '
    'Earlier evidence and policies are unchanged; this export does not activate a model.')


def vector(raw,size):
    require(isinstance(raw,list) and len(raw)==size,'Complete finite nominal vector required')
    return [number(v) for v in raw]


def captured(row):
    native=boolean(row.get('native_release'));boolean(row.get('invalid_physics'));success=boolean(row.get('route_success'))
    require(type(row.get('route_quality')) is int and row['route_quality'] in (0,1,2),'Explicit capture tier required')
    require(not success or native and not row.get('invalid') and not row['invalid_physics'],'Capture claims rejected/invalid execution')
    return bool(native and success and not row.get('invalid') and not row['invalid_physics'])


def statistics(rows):
    native=sum(boolean(r.get('native_release')) for r in rows);caps=sum(captured(r) for r in rows)
    invalid=sum(r['native_release'] and (bool(r.get('invalid')) or r['invalid_physics']) for r in rows)
    return dict(requested=len(rows),native_releases=native,captures=caps,clean=sum(captured(r) and r['route_quality']==2 for r in rows),
        rejected=len(rows)-native,native_misses=native-caps-invalid,invalid=invalid)


def control_gate(rows):
    pos=[r for r in rows if r['role']=='positive_control'];neg=[r for r in rows if r['role']=='negative_control']
    good=(len(rows)==6 and len(pos)==len(neg)==3 and all(captured(r) for r in pos)
        and all(r['native_release'] and not captured(r) and not r.get('whole_ball_entry') for r in neg)
        and all(not r.get('invalid') and not r['invalid_physics'] and r.get('scalar_score_audit_passed') is True for r in rows))
    return dict(status='passed' if good else 'failed',counts=statistics(rows))


def strict_audit(proof):
    checks=proof.get('checks')
    require(proof.get('version')==AUDIT_VERSION and proof.get('status')=='passed' and proof.get('failures')==[]
        and isinstance(checks,list) and bool(checks) and type(proof.get('checks_count')) is int
        and proof['checks_count']==len(checks) and all(isinstance(c,dict) and c.get('passed') is True and isinstance(c.get('label'),str) for c in checks),
        'Complete nonempty strictly passed v1 independent audit required')


def verified_inputs(study,audit_path):
    study,audit_path=Path(study).resolve(),Path(audit_path).resolve()
    report,spec,proof=map(read,(study/'report.json',study/'contract.json',audit_path));strict_audit(proof)
    for path,expected in ((study/'report.json',REPORT_SHA),(study/'evidence-manifest.json',MANIFEST_SHA),(audit_path,AUDIT_SHA)):
        require(sha(path)==expected,'Exact frozen publication bytes differ: '+path.name)
    require(proof.get('report_sha256')==REPORT_SHA and proof.get('evidence_manifest_sha256')==MANIFEST_SHA,'Proof byte bindings differ')
    require(spec.get('version')==report.get('version')==STUDY_VERSION and report.get('status')==report.get('phase')=='completed','Completed exact-version study required')
    unsigned=dict(spec);fp=unsigned.pop('fingerprint',None)
    require(fp==SOURCE_FP==digest(unsigned)==report.get('contract_fingerprint')==proof.get('contract_fingerprint'),'Exact frozen contract fingerprint differs')
    manifest=hash_map(read(study/'evidence-manifest.json'))
    require(report.get('evidence_sha256')==manifest and manifest.get('contract.json')==sha(study/'contract.json'),'Report/full-manifest contract binding differs')
    for name,value in manifest.items():require(sha(relative_file(study,name))==value,'Changed evidence artifact: '+name)
    actual={str(p.relative_to(study)) for p in study.rglob('*') if p.is_file() and str(p.relative_to(study)) not in ('report.json','evidence-manifest.json')}
    require(set(manifest)==actual,'Manifest must cover every frozen evidence artifact')
    require(proof.get('source_parent')==PARENT and spec.get('parent_contract_fingerprint')==PARENT['contract_fingerprint'],'Exact audited parent differs')
    bindings=spec['parent_bindings']
    for key,value in (('parent_contract_fingerprint',PARENT['contract_fingerprint']),('parent_report_sha256',PARENT['report_sha256']),
        ('parent_manifest_sha256',PARENT['manifest_sha256']),('parent_audit_sha256',PARENT['original_audit_sha256'])):
        require(bindings.get(key)==value,'Parent source byte identity differs: '+key)
    require(read(study/'source-bindings.json')==bindings,'Copied source bindings differ')
    require(proof.get('producer_source_sha256')==hash_map(spec.get('source_sha256')) and report.get('source_sha256')==spec['source_sha256']
        and proof.get('native_source_sha256')==hash_map(spec.get('native_source_sha256')),'Audited source modules differ')
    fingerprint(proof.get('auditor_sha256'))
    protocol=spec['protocol'];require(report.get('protocol')==protocol,'Executed protocol differs')
    fixed=dict(substeps=64,methods=list(METHODS),historical_current_train_requests=1024,new_train_requests=0,
        validation_layouts=24,test_layouts=60,validation_requests=72,test_requests=180,control_requests=12,
        maximum_fresh_requests=264,noise=False,no_test_flight_search=True,fixed_model_no_validation_selection=True,
        fit_steps=600,fit_device='cpu',ensemble_members=5,candidates=16,maximum_geometry_checks=4)
    require(all(protocol.get(k)==v for k,v in fixed.items()),'Literal offline fitting/evaluation protocol differs')
    require(spec.get('training_native_requests')==0 and spec.get('validation_selects_model') is False
        and proof.get('training_requests')==1024 and proof.get('new_training_requests')==0,'Training/validation boundary changed')
    packet=read(study/'prediction-plan.json')
    require(packet.get('frozen_before_any_native_outcome') is True and packet.get('validation_selects_model') is False
        and packet.get('native_planning_queries')==0 and report.get('prediction_plan_sha256')==sha(study/'prediction-plan.json'),'Preoutcome frozen inference boundary differs')
    rows=[json.loads(line) for line in (study/'records.jsonl').read_text().splitlines() if line]
    require(len(rows)==264 and len({r['id'] for r in rows})==264 and type(proof.get('requested_trials')) is int and proof['requested_trials']==264,'Exactly264 unique requested slots required')
    plan=packet.get('requests');require(isinstance(plan,list) and len(plan)==264 and [r['id'] for r in rows]==[r['id'] for r in plan],'Complete preoutcome request order differs')
    for row,request in zip(rows,plan):
        identifier(row['id']);vector(row['values'],5);vector(row['bucket_xy_m'],2)
        require(all(row.get(k)==v for k,v in request.items()),'Frozen requested decision was replaced')
        require(row.get('selection_or_learning') is False and row.get('ball_flight_search_at_inference') is False
            and row.get('action_substituted') is False and row.get('perturbation_mixed_units')==[0.]*5,'Undeclared learning, substitution or native search')
        if row['native_release']:require(row.get('contact_protocol',{}).get('substeps')==64 and row.get('scalar_score_audit_passed') is True,'Native64/oracle identity missing')
    require(report.get('counts')==statistics(rows) and proof.get('native_releases')==statistics(rows)['native_releases'],'All-phase native accounting differs')
    return study,report,spec,proof,manifest,packet,rows


def artifacts(study,report,packet):
    result={}
    require(set(report['artifacts'])==set(METHODS),'All three frozen model identities required')
    for method in METHODS:
        raw=report['artifacts'][method];name='model/'+method+'.pt'
        require(raw.get('path')==name and raw.get('sha256')==ARTIFACT_SHA[method]==sha(relative_file(study,name)),'Exact model checkpoint bytes differ: '+method)
        result[method]=dict(sha256=ARTIFACT_SHA[method])
        if method!='ensemble':result[method]['samples']=number(raw.get('samples'))
    model=report['artifacts']['ensemble']
    require(model.get('version')==MODEL_VERSION and model.get('ensemble_members')==5 and model.get('fit_device')=='cpu' and model.get('fit_steps')==600,'Prespecified CPU ensemble differs')
    require(packet.get('ensemble_artifact')==model and report['training'].get('artifact')==model,'Fitted/frozen/executed ensemble artifact differs')
    return result


def checked_splits(study,spec,rows):
    splits=read(study/'splits.json');require(splits==spec['splits'],'Frozen split manifest differs')
    used=set();result={}
    for role,size in (('validation',24),('test',60)):
        contexts=splits.get(role);require(isinstance(contexts,list) and len(contexts)==size,'Fresh layout denominator differs')
        byid={}
        for context in contexts:
            ident=identifier(context['id']);xy=vector(context['bucket_xy_m'],2)
            require(ident not in used and context.get('split')==role,'Repeated/mislabeled split identity');used.add(ident);byid[ident]=xy
        require(len({tuple(p) for p in byid.values()})==size,'Repeated bucket coordinates')
        result[role]=byid
        for method in METHODS:
            cohort=[r for r in rows if r['role']==role and r['method']==method]
            require(len(cohort)==size and {r['context_id'] for r in cohort}==set(byid),'Same layouts and one request per method required')
            require(all(r['bucket_xy_m']==byid[r['context_id']] for r in cohort),'Paired method bucket coordinates differ')
    require(not {tuple(v) for v in result['validation'].values()}&{tuple(v) for v in result['test'].values()},'VAL/TEST coordinates overlap')
    return result


def check_accounting(report,proof,rows):
    require(Counter(r['role'] for r in rows)==Counter(validation=72,test=180,positive_control=6,negative_control=6),'72VAL180TEST12 controls required')
    require(Counter(r['phase'] for r in rows)==Counter(controls_pre=6,validation=72,test=180,controls_post=6),'Finite four-phase budget differs')
    phases={p:statistics([r for r in rows if r['phase']==p]) for p in PHASES};require(report.get('phase_counts')==phases,'Per-phase accounting differs')
    gates={key:control_gate([r for r in rows if r['phase']==phase]) for key,phase in (('pre','controls_pre'),('post','controls_post'))}
    require(report.get('control_gates')==proof.get('control_gates')==gates and all(g['status']=='passed' for g in gates.values()),'Matched control gates differ')
    comparisons=[]
    for role in ('validation','test'):
        for method in METHODS:
            cohort=[r for r in rows if r['role']==role and r['method']==method]
            comparisons.append(dict(role=role,method=method,counts=statistics(cohort),artifact_sha256=ARTIFACT_SHA[method],mean_reward=math.fsum(number(r['learning_reward']) for r in cohort)/len(cohort)))
    for declared in (report.get('comparisons'),proof.get('comparisons')):
        require(isinstance(declared,list) and len(declared)==6,'All six split/method cohorts required')
        for actual,expected in zip(declared,comparisons):
            require(all(actual.get(k)==v for k,v in expected.items() if k!='mean_reward') and abs(number(actual.get('mean_reward'))-expected['mean_reward'])<=1e-8,'Method/split comparison differs')
    require(report.get('test_comparison')==[r for r in report['comparisons'] if r['role']=='test'],'Validation mixed into TEST comparison')
    g=report.get('gate',{});require(g.get('status')=='passed' and all(g.get(k) is True for k in ('complete','controls_passed','predictions_frozen_before_any_evaluation','no_test_flight_search')),'Completed frozen inference gate required')
    return phases,gates,comparisons


def test_rows(report,ledger,splits):
    positions=report.get('test_positions');require(isinstance(positions,list) and len(positions)==180,'All180 TEST position records required')
    byid={r['id']:r for r in ledger if r['role']=='test'};seen=set();output=[]
    for raw in positions:
        ident=identifier(raw['record_id']);require(ident in byid and ident not in seen,'Unknown/repeated TEST source record');seen.add(ident);row=byid[ident]
        require(raw['context_id']==row['context_id'] and raw['method']==row['method'] and raw['bucket_xy_m']==row['bucket_xy_m'] and raw['values']==row['values'],'Nominal TEST settings differ from spent request')
        require(raw.get('artifact_sha256')==row.get('frozen_artifact_sha256')==ARTIFACT_SHA[row['method']] and raw.get('model_prediction')==row.get('model_prediction'),'Forecast/model identity differs from preoutcome request')
        actual=dict(native_release=row['native_release'],capture=captured(row),clean=captured(row) and row['route_quality']==2,
            invalid=row.get('invalid'),invalid_physics=boolean(row['invalid_physics']),whole_entry=bool(row.get('whole_ball_entry')),
            retention_s=float(row.get('max_retention_s',0.)),entry_class='clean_capture' if captured(row) and row['route_quality']==2 else 'captured_nonclean' if captured(row) else 'no_capture')
        require(raw.get('actual')==actual,'Published actual outcome differs from native request')
        values=vector(row['values'],5);bucket=vector(row['bucket_xy_m'],2);distance=math.hypot(bucket[0]-values[0],bucket[1]-values[1])
        require(abs(number(raw.get('board_bucket_distance_m'))-distance)<=1e-10,'Nominal board-to-bucket distance differs')
        out=dict(record_id=ident,record_sha256=digest(row),context_id=identifier(row['context_id']),method=identifier(row['method']),
            bucket_x_m=bucket[0],bucket_y_m=bucket[1],**dict(zip(PARAMETERS,values)),board_bucket_distance_m=distance,
            artifact_sha256=ARTIFACT_SHA[row['method']])
        prediction=raw['model_prediction']
        for key in PREDICTIONS:out['predicted_'+key]=number(prediction.get(key))
        require(prediction.get('rebound_basis')=='board_to_bucket_forward_lateral_vertical','Prediction rebound coordinate basis differs')
        velocity=vector(prediction.get('rebound_velocity_m_s'),3)
        out.update(dict(zip(('predicted_rebound_forward_m_s','predicted_rebound_lateral_m_s','predicted_rebound_vertical_m_s'),velocity)))
        out.update({'actual_'+k:v for k,v in actual.items() if k!='invalid'})
        out['actual_invalid']=identifier(actual['invalid']) if actual['invalid'] is not None else None
        output.append(out)
    require(seen==set(byid),'TEST records omitted')
    return output


def training_summary(raw,proof):
    dataset=raw['dataset'];training=raw['training']
    require(dataset.get('train_requests')==dataset.get('train_labels')==1024 and dataset.get('source_contract_fingerprint')==PARENT['contract_fingerprint']
        and dataset.get('source_report_sha256')==PARENT['report_sha256'] and dataset.get('source_audit_sha256')==PARENT['original_audit_sha256'],'Current parent TRAIN-only dataset binding differs')
    support=dataset.get('target_support');require(isinstance(support,list) and len(support)==9 and all(type(v) is int and 0<=v<=1024 for v in support)
        and support==proof.get('training_masks'),'Observed TRAIN target support differs')
    require(training.get('status')=='completed' and training.get('ensemble_members')==5 and training.get('fit_device')=='cpu' and training.get('fit_steps')==600
        and training.get('input_features')==list(INPUTS) and training.get('target_features')==list(TARGETS),'Model fitting contract differs')
    curves=[]
    for row in training.get('curves',[]):curves.append({k:number(row[k]) for k in ('member','step','train_loss','full_train_loss')})
    require([(r['member'],r['step']) for r in curves]==[(m,s) for m in range(5) for s in range(0,601,100)],'All35 actual member/step curve samples required')
    return dict(train_requests=1024,new_train_requests=0,train_native_releases=number(dataset['train_native_releases']),
        train_layouts=number(dataset['train_layouts']),excluded_prior_layouts=number(dataset['excluded_prior_layouts']),
        target_support=dict(zip(TARGETS,support)),input_features=list(INPUTS),target_features=list(TARGETS),
        ensemble_members=5,fit_device='cpu',fit_steps=600,curves=curves,
        split_sha256=hash_map(dataset['split_hashes']))


def prediction_metrics(raw):
    result=[]
    for row in raw:
        require(row.get('split') in ('validation','test') and row.get('target') in (*TARGETS,'requested_capture_uncalibrated_score'),'Unknown supported-target prediction metric')
        n=row.get('support');require(type(n) is int and 0<=n<=(72 if row['split']=='validation' else 180),'Observed metric support differs')
        item=dict(split=row['split'],target=row['target'],support=n)
        if 'unit' in row:require(row['unit'] in ('m','m/s','fraction'),'Unknown target unit');item['unit']=row['unit']
        for key in ('mae','rmse','brier','accuracy'):
            if key in row:item[key]=number(row[key])
        result.append(item)
    return result


def curate(study,audit_path,output):
    study,report,spec,proof,manifest,packet,rows=verified_inputs(study,audit_path)
    output=Path(output).resolve();require(not output.exists() and not output.is_relative_to(study) and not study.is_relative_to(output)
        and not Path(audit_path).resolve().is_relative_to(output),'New publication directory outside source evidence required')
    models=artifacts(study,report,packet);splits=checked_splits(study,spec,rows);phases,gates,comparisons=check_accounting(report,proof,rows)
    csv_rows=test_rows(report,rows,splits);training=training_summary(report,proof)
    metrics=prediction_metrics(report.get('prediction_metrics',[]))
    summary=dict(version=VERSION,study_version=STUDY_VERSION,status='completed',scope=SCOPE,counts=statistics(rows),phase_counts=phases,
        test_comparison=[r for r in comparisons if r['role']=='test'],validation_comparison=[r for r in comparisons if r['role']=='validation'],
        controls=gates,training=training,prediction_metrics=metrics,artifacts=models,
        budgets=dict(validation_layouts=24,test_layouts=60,methods=list(METHODS),fresh_requests=264,new_train_requests=0,
            model_candidates=16,maximum_geometry_checks=4,actual_geometry_planning_checks=number(report['geometry_planning_checks'])),
        timing_s=dict(cpu_preparation=number(report['cpu_preparation_elapsed_s']),acquired_native=number(report['acquired_elapsed_s'])),
        provenance=dict(contract_fingerprint=SOURCE_FP,report_sha256=REPORT_SHA,evidence_manifest_sha256=MANIFEST_SHA,
            independent_audit_sha256=AUDIT_SHA,audit_version=AUDIT_VERSION,audit_checks=proof['checks_count'],auditor_sha256=fingerprint(proof['auditor_sha256']),
            source_parent=PARENT,producer_source_sha256=hash_map(spec['source_sha256']),native_source_sha256=hash_map(spec['native_source_sha256']),
            prediction_plan_sha256=sha(study/'prediction-plan.json'),records_sha256=sha(study/'records.jsonl')))
    output.mkdir(parents=True,exist_ok=False);write_json(output/'results-summary.json',summary)
    with (output/'test-settings.csv').open('x',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(csv_rows[0]),lineterminator='\n');writer.writeheader();writer.writerows(csv_rows)
    write_json(output/'checksums.json',{p.name:sha(p) for p in sorted(output.iterdir()) if p.is_file()})
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('study','audit','output'):parser.add_argument('--'+key,type=Path,required=True)
    args=parser.parse_args();result=curate(args.study,args.audit,args.output)
    print(json.dumps(dict(version=VERSION,status=result['status'],test_rows=180,counts=result['counts'])))


if __name__=='__main__':main()
