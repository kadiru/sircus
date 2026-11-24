"""Read-only preparation and saved-native audit for bucket-approach correction."""
from __future__ import annotations
import argparse
from collections import defaultdict
import copy
import json
import math
from pathlib import Path
import numpy as np
from .benchmark_verify import Checks,contained,digest,read,sha
from .sweep_worker import atomic

VERSION='sircus-bucket-approach-independent-audit-v1'
PREPARATION_VERSION='sircus-bucket-approach-preparation-audit-v1'


def expected_slots(plan):
    """Independently reconstruct matching repeats and all preservation slots."""
    result=[]
    for kind,cases in (('preserved',plan['preserved_cases']),('diagnostic_control',plan['diagnostic_cases']),('diagnostic',plan['cases'])):
        for case in cases:
            for repeat in range(3):
                variants=[kind] if kind in ('preserved','diagnostic_control') else (['original','corrected'] if repeat%2==0 else ['corrected','original'])
                for variant in variants:
                    source=case['original'] if kind in ('preserved','diagnostic_control') else case[variant]
                    request=None
                    if source is not None:
                        request=copy.deepcopy(source)
                        request.update(id=f"{case['id']}-{variant}-{repeat}",context_id=case['id'],variant=variant,
                            repeat=repeat,method=variant,phase='preservation' if kind=='preserved' else 'diagnostic',source_request_id=source['id'])
                    result.append(dict(case_id=case['id'],variant=variant,repeat=repeat,directory=case['directory'],request=request))
    return result


def independent_summary(plan,rows):
    def summary(source,found):
        scores=[row['board_first_score'] for row in found]
        outcomes=[]
        for score in scores:
            outcomes.append(score['route_class'] if score['route_success'] else 'invalid_physics' if score['invalid_physics'] else 'miss')
        return dict(requested=3,completed=len(found),released=len(found),setup_rejected=3 if source is None else 0,
            captures=sum(bool(score['route_success']) for score in scores),invalid_physics=sum(bool(score['invalid_physics']) for score in scores),
            robot_contacts=sum(score['robot_contact_steps']>0 for score in scores),bucket_contacts=sum(bool(score['bucket_contact_before_entry']) for score in scores),
            outcomes=outcomes,rows=found)
    cases=[];preservation=[];diagnostic_controls=[]
    for case in plan['cases']:
        groups={}
        for variant in ('original','corrected'):
            found=[row for row in rows if row.get('context_id')==case['id'] and row.get('variant')==variant]
            groups[variant]=summary(case[variant],found)
        cases.append(dict(id=case['id'],bucket_xy_m=case['original']['bucket_xy_m'],results=groups))
    for case in plan['preserved_cases']:
        found=[row for row in rows if row.get('context_id')==case['id'] and row.get('variant')=='preserved']
        preservation.append(dict(id=case['id'],bucket_xy_m=case['original']['bucket_xy_m'],**summary(case['original'],found)))
    for case in plan['diagnostic_cases']:
        found=[row for row in rows if row.get('context_id')==case['id'] and row.get('variant')=='diagnostic_control']
        diagnostic_controls.append(dict(id=case['id'],bucket_xy_m=case['original']['bucket_xy_m'],**summary(case['original'],found)))
    return dict(cases=cases,preservation=preservation,diagnostic_controls=diagnostic_controls)


def verify_preparation(a,checks):
    from .bucket_approach_study import (VERSION as STUDY_VERSION,SEARCH_RULE,source_inputs,verify_frozen,
        proposal_menu,ordered_proposals)
    from .board_pose_recovery_study import nominal_prediction,wrist_screen
    from .bucket_approach import BucketApproach
    from .board_first_selector_plan import BoardFirstFlightRanker
    from .direct_path_first_audit import _proof,audit_preparation
    from .fixed35_broad_native import reach_models
    from .table_assisted_env import _same
    study=a.output;spec=verify_frozen(a);plan=read(study/'preparation.json')
    cases,preserved,diagnostic,source_contract,binding=source_inputs(a)
    checks.equal((spec['version'],plan['version']),(STUDY_VERSION,)*2,'Declared study versions')
    checks.equal((spec['physics_substeps'],spec['repeats'],spec['controls'],spec['preservation_requests'],spec['maximum_native_releases']),
                 (64,3,18,12,51),'Fixed protocol')
    checks.equal((spec['source'],plan['source']),(binding,binding),'Unchanged parent evidence')
    checks.equal(spec['identity'],source_contract['identity'],'Original scene identity')
    checks.equal((spec['search_rule'],plan['search_rule']),(SEARCH_RULE,SEARCH_RULE),'Frozen finite search rule')
    checks.equal([c['id'] for c in plan['cases']],[c['id'] for c in cases],'Exactly three rising-wall targets')
    checks.equal([c['id'] for c in plan['preserved_cases']],[c['id'] for c in preserved],'All four captured parent lanes')
    checks.equal([c['id'] for c in plan['diagnostic_cases']],[c['id'] for c in diagnostic],'One unchanged robot-contact comparison')
    prior,home=read(study/'training-prior.json'),read(study/'setup-home.json')
    calibration=read(study/'calibration.json')['score_config']
    ranker=BoardFirstFlightRanker(spec['scene'],calibration,a.assets,source_contract['flight_ranking']['rule'])
    checker=BucketApproach(spec['scene']);models=reach_models(str(a.assets.resolve()))
    for case,original_case in zip(plan['cases']+plan['preserved_cases']+plan['diagnostic_cases'],cases+preserved+diagnostic,strict=True):
        checks.equal({key:case[key] for key in original_case},original_case,case['id']+'/exact historical case')
        directory=study/case['directory'];resolutions={}
        layout=dict(id=case['id'],bucket_xy_m=case['original']['bucket_xy_m'])
        _proof(case['original'],layout,directory,prior,home,resolutions,checks)
        prediction=ranker.predict(case['original']);approach=checker.assess(case['original'],ranker,prediction=prediction)
        _same(case['original_prediction'],prediction,case['id']+'/original prediction'); checks.count+=1
        _same(case['original_approach'],approach,case['id']+'/original full approach'); checks.count+=1
        if case in plan['preserved_cases']+plan['diagnostic_cases']:continue
        expected=proposal_menu(case,read(study/'candidate-sources.json'),read(study/'shot-library-rows.json'),
            read(study/'route-models.json'),case['bounds']['low'],case['bounds']['high'])
        checks.equal(case['proposals'],expected,case['id']+'/deterministic finite proposal menu')
        nominal={p['id']:nominal_prediction(ranker,spec['scene'],p) for p in expected}
        wrist={p['id']:wrist_screen(spec['scene'],p,models) for p in expected}
        _same(case['nominal_predictions'],nominal,'Recomputed initial rim screen');checks.count+=1
        checks.equal(case['wrist_reach'],wrist,'Recomputed necessary wrist reach')
        ordered=ordered_proposals(expected,nominal,wrist)
        checks.equal(case['preparation_order'],[p['id'] for p in ordered],'Fixed bounded check order')
        checks.require(len(case['checked'])<=40,'At most forty checked candidates')
        checks.equal(case['checked'],ordered[:len(case['checked'])],'Only complete ordered prefix')
        artifact=dict(proposals=case['checked'],admitted=case['candidates'],records=case['records'])
        audit_preparation(artifact,layout,'diagnostic',directory,prior,home,resolutions,checks)
        predictions={};approaches={}
        for request in case['candidates']:
            predictions[request['id']]=ranker.predict(request)
            approaches[request['id']]=checker.assess(request,ranker,prediction=predictions[request['id']])
        _same(case['predictions'],predictions,'Every admitted prediction recomputed');checks.count+=len(predictions)
        _same(case['approaches'],approaches,'Every admitted approach recomputed');checks.count+=len(approaches)
        def score(value):
            result=value.get('predicted_score_m')
            return result if type(result) in (int,float) and math.isfinite(result) else -math.inf
        ranked=sorted(case['candidates'],key=lambda r:(-score(approaches[r['id']]),r['id']))
        selected=ranked[0] if ranked and score(approaches[ranked[0]['id']])>score(approach)+1e-9 else None
        checks.equal(case['corrected'],selected,'Choice fixed by full approach score, not observed drop')
        def oldscore(value):
            v=value.get('score_m');return float(v) if type(v) in (int,float) and math.isfinite(v) else -math.inf
        without_bucket=max([case['original']]+case['candidates'],key=lambda r:(oldscore(prediction if r is case['original'] else predictions[r['id']]),r['id']))
        checks.equal(case['selected_without_bucket_check'],without_bucket,'Same admitted menu ranking without new check')
    slots=expected_slots(plan)
    checks.equal(read(study/'slots.json'),slots,'Independent matched schedule')
    checks.equal(len(slots),33,'Thirty-three development/preservation slots including declines')
    return spec,plan,slots


def preparation(a):
    checks=Checks();spec,plan,slots=verify_preparation(a,checks)
    checks.require(not (a.output/'native-started.json').exists(),'No native execution before preflight')
    return dict(version=PREPARATION_VERSION,status='passed',checks_count=checks.count,native_execution_performed=False,
        contract_sha256=sha(a.output/'contract.json'),preparation_sha256=sha(a.output/'preparation.json'),
        slots_sha256=sha(a.output/'slots.json'),planned_slots=len(slots),planned_releases=sum(s['request'] is not None for s in slots)+18)


def audit(a):
    from . import board_first_score_oracle as new_oracle,table_assisted_score_oracle as old_oracle
    from .bucket_approach_study import lines
    from .cell1050_rebound_study import control_requests,parent_inputs
    from .direct_path_first_audit import verify_heading
    from .expanded_selector_study import board_first_controls
    from .expanded_selector_audit import verify_initial_trace
    from .policy64_verify import audit_native_options
    from .table_assisted_env import _same
    checks=Checks();study=a.output
    spec,plan,slots=verify_preparation(a,checks)
    report=read(study/'report.json');manifest=read(study/'evidence-manifest.json')
    checks.equal((report['status'],report['phase']),('completed','completed'),'Completed native study')
    checks.equal(report['evidence_manifest_sha256'],sha(study/'evidence-manifest.json'),'Report binds complete evidence')
    inventory={str(p.relative_to(study)) for p in study.rglob('*') if p.is_file() and p.name not in ('report.json','evidence-manifest.json')}
    checks.equal(set(manifest),inventory,'Complete evidence inventory')
    for name,value in manifest.items():checks.equal(sha(contained(study,name)),value,'Evidence/'+name)
    started=read(study/'native-started.json')
    for key,name in (('contract_sha256','contract.json'),('slots_sha256','slots.json'),('preparation_sha256','preparation.json')):
        checks.equal(started[key],sha(study/name),'Native start binds '+name)
    if a.preflight_audit:
        checks.equal(started['preflight_audit_sha256'],sha(a.preflight_audit),'Native started after exact independent preflight')
    parent,parent_binding,_=parent_inputs(a.parent_study,a.parent_audit)
    checks.equal((spec['identity'],spec['scene'],spec['parent']),(parent['identity'],parent['scene'],parent_binding),'Original controlled native identity')
    sources=read(a.parent_study/'control-replays.json')
    controls={phase:control_requests(sources,phase) for phase in ('controls_pre','controls_post')}
    expected=[*controls['controls_pre'],*[slot['request'] for slot in slots if slot['request'] is not None],*controls['controls_post']]
    rows=lines(study/'rows.jsonl');checks.equal([r['id'] for r in rows],[r['id'] for r in expected],'Exact ordered native releases')
    checks.equal(len(set(r['id'] for r in rows)),len(rows),'Every release ID unique')
    by_id={r['id']:r for r in expected}; packets=sorted((study/'requests').glob('*.json'))
    seen=[]
    for index,path in enumerate(packets):
        checks.equal(path.name,f'batch-{index:04d}.json','Contiguous request packets')
        packet=read(path)
        for request in packet['rows']:
            checks.equal(request,by_id[request['id']],'Exact frozen packet command')
            seen.append(request['id'])
    checks.equal(seen,[r['id'] for r in expected],'Every packet accounted for')
    calibration=read(study/'calibration.json')['score_config'];prior=read(study/'training-prior.json');home=read(study/'setup-home.json')
    groups=defaultdict(list);resolutions=defaultdict(dict)
    for row in rows:
        request=by_id[row['id']];directory=study/row['adapter_directory'];packet=read(study/row['request_packet'])
        checks.equal(row['request_packet_sha256'],sha(study/row['request_packet']),'Exact packet bytes')
        checks.equal(row['phase'],packet['phase'],'Executed phase')
        checks.equal(row['adapter_directory'],packet['adapter_directory'],'Executed native namespace')
        for key in ('id','values','bucket_xy_m'):checks.equal(row[key],request[key],row['id']+'/command '+key)
        checks.require(row['native_release'] is True and row['action_substituted'] is False,'Exact native release')
        if row['adapter_directory']!='native-controls':
            verify_heading(row,request,directory,spec['scene'],home,prior,resolutions[str(directory)],checks)
            checks.equal((row['variant'],row['repeat'],row['context_id']),(request['variant'],request['repeat'],request['context_id']),'Frozen variant/repeat/context')
        groups[row['adapter_directory'],row['native_cache_file']].append(row)
    for (directory,filename),group in groups.items():
        path=contained(study/directory,filename)
        with np.load(path,allow_pickle=False) as archive:arrays={key:archive[key] for key in archive.files}
        for row in group:
            col=row['native_cache_row'];checks.equal((row['native_cache_sha256'],row['id']),(sha(path),str(arrays['record_ids'][col])),'Saved dense native column')
            verify_initial_trace(row,arrays,col,checks)
            trace={key:arrays[key][:,col] for key in ('position_m','velocity_m_s','forces_N','contact_distance_m')}
            for key in ('time_s','dt_s'):
                if key in arrays:trace[key]=arrays[key]
            arguments=spec['scene'],row['bucket_xy_m'],row['board_normal'],calibration,trace
            old,new=old_oracle.score_trace(*arguments),new_oracle.score_trace(*arguments)
            for key,value in old.items():_same(row[key],value,'Historical scalar score/'+key);checks.count+=1
            _same(row['board_first_score'],new,'Independent board-first score');checks.count+=len(new)
    for directory in {r['adapter_directory'] for r in rows}:
        audit_native_options(study/directory,spec['scene'],calibration,a.assets,checks)
        checks.equal({name for folder,name in groups if folder==directory},
            {p.relative_to(study/directory).as_posix() for p in (study/directory).rglob('*.npz')},'No uncounted native arrays')
    for phase,requests in controls.items():
        native=[r for r in rows if r['id'] in {q['id'] for q in requests}]
        gate=board_first_controls(native)
        checks.equal(gate['status'],'passed','Matched controls pass')
        checks.equal(report['controls'][phase],gate,'Control accounting')
    comparison=independent_summary(plan,rows)
    for key,value in comparison.items():checks.equal(report[key],value,'Independent '+key+' counts')
    checks.equal(report['native_releases'],len(rows),'All releases retained in denominator')
    checks.require(0<report['gpu_lease_s']<=3600,'At most one GPU hour')
    return dict(version=VERSION,status='passed',checks_count=checks.count,report_sha256=sha(study/'report.json'),
        evidence_manifest_sha256=sha(study/'evidence-manifest.json'),contract_sha256=sha(study/'contract.json'),
        preparation_sha256=sha(study/'preparation.json'),slots_sha256=sha(study/'slots.json'),choices_sha256=sha(study/'slots.json'),
        native_started_sha256=sha(study/'native-started.json'),frozen_before_native=True,comparison=comparison,
        scope='Read-only full saved trajectory rescoring and deterministic choice/proof reconstruction; no physics rerun')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('source-study','source-audit','diagnosis','parent-study','parent-audit','assets','output','audit-output'):
        parser.add_argument('--'+key,type=Path,required=True)
    parser.add_argument('--preflight-audit',type=Path)
    parser.add_argument('--stage',choices=('preparation','full'),required=True)
    args=parser.parse_args()
    if args.audit_output.exists() or args.audit_output.resolve().is_relative_to(args.output.resolve()):parser.error('Fresh audit output outside study required')
    try:result=preparation(args) if args.stage=='preparation' else audit(args)
    except Exception as error:result=dict(version=PREPARATION_VERSION if args.stage=='preparation' else VERSION,status='failed',error=f'{type(error).__name__}: {error}')
    atomic(args.audit_output,result);print(json.dumps({k:v for k,v in result.items() if k!='comparison'}),flush=True)
    if result['status']!='passed':raise SystemExit(1)


if __name__=='__main__':main()
