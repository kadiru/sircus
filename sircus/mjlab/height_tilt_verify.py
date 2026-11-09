"""Independent CPU audit of the finite, fixed-XY height/tilt development grid.

The producer's proposal, selection and accounting functions are not imported.
Native evidence is checked with the unchanged independent v3/dense checker.
"""
from __future__ import annotations
import argparse
import copy
import json
from pathlib import Path
import time
import numpy as np
from . import replay650_verify as replay
from . import table_training_verify as v3
from . import calibrated_benchmark_verify as calibrated
from . import policy64_verify as old
from .benchmark_verify import EvidenceError, contained, digest, read, sha

VERSION='sircus-table-height-tilt-audit-v1'
STUDY_VERSION='sircus-table-height-tilt-study-v1'
CASES=tuple(f'table-training-test-{i:02d}' for i in (0,1,18,26))
HEIGHTS=(.32,.36,.40,.44,.46,.48)
TILTS=(16.,20.,24.,28.,32.)
ROLES=('direct_control','table_control','negative_control')
META=('source_hint','effective_source_hint','source_teacher_id','context_resolution_file',
      'context_resolution_sha256','context_home_sha256','context_layout_eligible',
      'context_anchor_admitted','context_hard_reject','observation')
PARENT_REPORT='4605bcb801351ae91ad16b71a033ceb3866027db621f9d84227f23b1ef257e95'
PARENT_AUDIT='6e50d87ddb7b2d4b37debbc2a5ac96507ff40bc236223277121acd385b8ec91b'


def source_cases(sources,checks):
    result=[]
    for context in CASES:
        group=[r for r in sources if r.get('phase')=='test' and r.get('method')=='ppo' and r.get('context_id')==context]
        checks.equal(len(group),1,'One exact parent case/'+context)
        checks.require(group[0].get('native_release') is True and not group[0].get('route_success') and not group[0].get('invalid_physics'),context+'/admitted original miss')
        result.append(group[0])
    return result


def audit_grid(plan,sources,checks):
    """Reconstruct all120 fixed proposals independently, without feasibility filtering."""
    cases=source_cases(sources,checks)
    checks.equal(len(plan),120,'All120 candidates retained before admission')
    checks.equal(len({r['id'] for r in plan}),120,'Unique grid candidate IDs')
    for ci,source in enumerate(cases):
        for hi,height in enumerate(HEIGHTS):
            for ti,tilt in enumerate(TILTS):
                local=hi*5+ti;index=ci*30+local;row=plan[index];label=f'{source["context_id"]}/grid{local:02d}'
                values=copy.deepcopy(source['values']);values[2]=height;values[3]=tilt
                exact=request(source,f'height-tilt-{source["context_id"]}-g{local:02d}','discovery','grid',0,values,index,local)
                checks.equal(row,exact,label+'/independently reconstructed whole proposal')
                checks.equal(row['id'],f'height-tilt-{source["context_id"]}-g{local:02d}',label+'/literal identity')
                checks.equal((row['context_id'],row['grid_index'],row['case_grid_index']),(source['context_id'],index,local),label+'/fixed grid ordering')
                checks.equal((row['phase'],row['variant']),('discovery','grid'),label+'/discovery only')
                checks.equal((row['values'],row['command_values'],row['bucket_xy_m']),(values,values,source['bucket_xy_m']),label+'/only prescribed height and tilt change')
                checks.equal((row['source_record_id'],row['source_record_sha256']),(source['id'],digest(source)),label+'/complete parent input identity')
                checks.equal(row['substeps'],64,label+'/fixed64')
                checks.equal(row['perturbation_mixed_units'],[0.]*5,label+'/no noise')
                for key in META:checks.equal(row[key],source[key],label+'/unchanged source metadata/'+key)


def selected_captures(rows):
    """One valid discovery capture per case; failures never become repeat candidates."""
    result={}
    for context in CASES:
        group=[r for r in rows if r.get('phase')=='discovery' and r.get('context_id')==context and v3.captured(r)]
        result[context]=min(group,key=lambda r:(-int(r['route_quality']),-float(r['learning_reward']),int(r['grid_index']))) if group else None
    return result


def request(source,identifier,phase,variant,repeat,values,grid_index=None,case_grid_index=None):
    item={key:copy.deepcopy(source[key]) for key in META}
    item.update(id=identifier,phase=phase,role=source['role'] if variant=='control_exact' else 'diagnostic',
        method='control' if variant=='control_exact' else variant,variant=variant,repeat=repeat,
        context_id=source['context_id'],bucket_xy_m=copy.deepcopy(source['bucket_xy_m']),values=copy.deepcopy(values),
        command_values=copy.deepcopy(values),source_record_id=source['id'],source_record_sha256=digest(source),
        substeps=64,perturbation_mixed_units=[0.]*5,selection_or_learning=False,posthoc_development=True,
        grid_index=grid_index,case_grid_index=case_grid_index)
    return item


def fixed_requests(sources,phase,checks):
    controls=phase.startswith('controls_')
    if controls:
        ordered=[]
        for role in ROLES:
            found=[r for r in sources if r.get('phase')=='controls_pre' and r.get('role')==role and r.get('repeat')==0]
            checks.equal(len(found),1,'One original admitted control/'+role)
            checks.require(found[0].get('native_release') is True,'Original control was released/'+role)
            ordered.append(found[0])
    else:ordered=source_cases(sources,checks)
    return [request(source,f'height-tilt-{phase}-{source["context_id"]}-r{i}',phase,
                    'control_exact' if controls else 'baseline',i,source['values']) for source in ordered for i in range(3)]


def audit_matrix(study,plan,matrix,prior,home,checks):
    from .calibrated_admission import verify_resolution,verify_connection
    checks.equal(len(matrix),120,'Matrix preserves all120 proposed candidates')
    resolutions={};completed=[];unattempted=False
    for proposal,cell in zip(plan,matrix):
        label=proposal['id']
        wanted=dict(candidate_id=proposal['id'],context_id=proposal['context_id'],grid_index=proposal['grid_index'],
                    case_grid_index=proposal['case_grid_index'],values=proposal['values'],bucket_xy_m=proposal['bucket_xy_m'],
                    resolution_file=proposal['context_resolution_file'],resolution_sha256=proposal['context_resolution_sha256'])
        for key,value in wanted.items():checks.equal(cell[key],value,label+'/matrix identity/'+key)
        checks.require(cell['status'] in ('completed','not_attempted'),label+'/explicit CPU admission status')
        if cell['status']=='not_attempted':
            unattempted=True
            for key in ('passed','valid_ik_branches','connection_file','connection_sha256','q10','q14'):
                checks.equal(cell[key],None,label+'/unattempted is not rejected/'+key)
            continue
        checks.require(not unattempted,label+'/admission follows fixed grid prefix without skips')
        checks.require(type(cell['passed']) is bool,label+'/completed admission result is boolean')
        checks.require(type(cell['elapsed_s']) in (int,float) and np.isfinite(cell['elapsed_s']) and cell['elapsed_s']>=0,label+'/measured CPU admission elapsed')
        name=cell['resolution_file'];path=contained(study/'native',name)
        checks.equal(sha(path),cell['resolution_sha256'],label+'/original resolution proof bytes')
        if name not in resolutions:
            value=read(path);value.update(file=name,sha256=sha(path))
            checks.require(verify_resolution(value,prior['records'],home),label+'/geometry-only fixed resolution')
            resolutions[name]=value
        resolution=resolutions[name]
        checks.equal(resolution['bucket_xy_m'],cell['bucket_xy_m'],label+'/resolution bucket fixed')
        path=contained(study/'native',cell['connection_file'])
        checks.equal(sha(path),cell['connection_sha256'],label+'/exact-action connection bytes')
        connection=read(path)
        checks.require(verify_connection(connection,resolution,home,cell['values']),label+'/all original admission guards retained')
        checks.equal(connection['bucket_xy_m'],cell['bucket_xy_m'],label+'/connection bucket fixed')
        checks.equal((connection['passed'],connection['reason']),(cell['passed'],cell['reason']),label+'/admission label comes from exact proof')
        checks.equal((connection.get('endpoint_attempts') or {}).get('valid_ik_branches'),cell['valid_ik_branches'],label+'/finite IK counts from proof')
        for joint in ('q10','q14'):
            checks.equal(cell[joint],connection[joint] if cell['passed'] else None,label+'/only admitted joints exposed/'+joint)
        completed.append(cell)
    return dict(total=120,attempted=len(completed),admitted=sum(c['passed'] for c in completed),
                rejected=sum(not c['passed'] for c in completed),not_attempted=120-len(completed))


def authenticate_parent(parent,audit,checks):
    parent,audit=Path(parent),Path(audit)
    checks.equal(sha(parent/'report.json'),PARENT_REPORT,'Exact completed parent report')
    checks.equal(sha(audit),PARENT_AUDIT,'Exact independent parent audit')
    proof,spec=read(audit),read(parent/'contract.json')
    checks.equal((proof['status'],proof['full_checks_verified'],proof['failures']),('passed',True,[]),'Parent audit completed without failures')
    checks.equal((proof['checks_count'],len(proof['checks'])),(823442,823442),'Full parent check accounting')
    checks.require(all(c['passed'] is True for c in proof['checks']),'All parent audit checks passed')
    checks.equal(proof['report_sha256'],PARENT_REPORT,'Parent proof binds report')
    unsigned=dict(spec);fp=unsigned.pop('fingerprint')
    checks.equal(digest(unsigned),fp,'Whole parent contract fingerprint')
    checks.equal(proof['contract_fingerprint'],fp,'Parent audit binds contract')
    manifest=read(parent/'evidence-manifest.json')
    checks.equal(sha(parent/'evidence-manifest.json'),proof['evidence_manifest_sha256'],'Parent evidence manifest identity')
    checks.equal(manifest,read(parent/'report.json')['evidence_sha256'],'Parent manifest equals report evidence')
    for name in ('rows.jsonl','contract.json','calibration.json','verification.json','setup-home.json','prior.json','native/effective-options.json','native/native-compiled-geometry.json'):
        checks.equal(sha(parent/name),manifest[name],'Parent bound artifact/'+name)
    bindings=dict(report_sha256=PARENT_REPORT,audit_sha256=PARENT_AUDIT,manifest_sha256=sha(parent/'evidence-manifest.json'),
      rows_sha256=sha(parent/'rows.jsonl'),contract_sha256=sha(parent/'contract.json'),contract_fingerprint=fp)
    return spec,replay.read_records(parent/'rows.jsonl'),bindings


def protocol():
    return dict(contexts=list(CASES),heights_m=list(HEIGHTS),tilts_deg=list(TILTS),
        fixed_parameters=['bucket_xy_m','board_x_m','board_y_m','drop_gap_m'],candidates_per_case=30,
        total_candidates=120,substeps=64,nominal_repeats=3,
        selection_order=['route_quality_desc','v3_reward_desc','grid_index_asc'],selection_requires_capture=True,
        noise=False,training=False,posthoc_development=True)


def budget():
    return dict(maximum_requested=162,candidate_proposals=120,original_requests=12,control_requests=18,
        maximum_selected_repeat_requests=12,cpu_admission_limit_s=3600,native_acquired_limit_s=3600,
        maximum_queue_s=7200,storage_reserve_bytes=8*1024**3)


def statistics(rows):
    result=v3.statistics(rows)
    result['robot_contact_shots']=sum(r.get('diagnostic_robot_contact_steps',0)>0 for r in rows)
    return result


def expected_selection(rows):
    result=[]
    for context,row in selected_captures(rows).items():
        selected=None if row is None else dict(candidate_id=row['id'],grid_index=row['grid_index'],case_grid_index=row['case_grid_index'],
            row_sha256=digest(row),values=row['values'],bucket_xy_m=row['bucket_xy_m'],connection_file=row['connection_file'],
            connection_sha256=row['connection_sha256'],q10=row['q10'],q14=row['q14'],route_quality=row['route_quality'],learning_reward=row['learning_reward'])
        result.append(dict(context_id=context,selected=selected))
    return result


def repeat_requests(plan,selection):
    lookup={r['id']:r for r in plan};result=[]
    for record in selection:
        selected=record['selected']
        if selected is None:continue
        for repeat in range(3):
            row=copy.deepcopy(lookup[selected['candidate_id']])
            row.update(id=f'height-tilt-selected-{record["context_id"]}-r{repeat}',phase='selected_repeats',variant='selected',method='selected',repeat=repeat,
                       selected_from_id=selected['candidate_id'],selected_from_sha256=selected['row_sha256'])
            result.append(row)
    return result


def audit_native_requests(rows,expected,sources,matrix,checks):
    checks.equal(len(rows),len(expected),'Every prescribed native request retained')
    checks.equal(len({r['id'] for r in rows}),len(rows),'Unique observed request identities')
    source_lookup={r['id']:r for r in sources};cells={r['candidate_id']:r for r in matrix}
    for row,request_value in zip(rows,expected):
        label=row['id']
        for key,value in request_value.items():checks.equal(row.get(key),value,label+'/whole exact request/'+key)
        checks.require(row.get('native_release') is True,label+'/only fully admitted native releases')
        if row['variant'] in ('baseline','control_exact'):bound=source_lookup[row['source_record_id']]
        else:bound=cells[row['id'] if row['variant']=='grid' else row['selected_from_id']]
        for key in ('connection_file','connection_sha256','q10','q14'):
            checks.equal(row[key],bound[key],label+'/original or pre-native matrix branch/'+key)


def audit_report(report,spec,sources,matrix,rows,selection,checks):
    admission=dict(proposals=120,completed=sum(c['status']=='completed' for c in matrix),admitted=sum(c['passed'] is True for c in matrix),
                   declined=sum(c['passed'] is False for c in matrix),unattempted=sum(c['status']!='completed' for c in matrix))
    discovery=[r for r in rows if r['phase']=='discovery']
    for key,wanted in dict(parent_bindings=spec['parent_bindings'],grid_protocol=protocol(),budget=budget(),
            admission_matrix_sha256=spec['admission_matrix_sha256'],admission_summary=admission,
            discovery_summary=dict(proposals=120,admitted=admission['admitted'],**statistics(discovery)),
            counts=statistics(rows),native_requested=len(rows),requested_configurations=admission['completed']+sum(r['phase']!='discovery' for r in rows),
            selection=selection,repeats={v:statistics([r for r in rows if r['variant']==v]) for v in ('baseline','selected')},
            training=False,noise=False,posthoc_development=True).items():checks.equal(report[key],wanted,'Independently reconstructed report/'+key)
    observations={r['id']:r for r in rows};by_context={r['context_id']:r for r in sources};selected={s['context_id']:s['selected'] for s in selection}
    expected=[]
    for context in CASES:
        grid=[]
        for cell in matrix:
            if cell['context_id']!=context:continue
            row=observations.get(cell['candidate_id']);observed=None
            if row is not None:
                observed=dict(record_id=row['id'],record_sha256=digest(row),native_release=row['native_release'],captured=v3.captured(row),route_quality=row['route_quality'],
                    direct_capture=row['direct_capture'],table_assisted_capture=row['table_assisted_capture'],invalid_physics=row.get('invalid_physics',False),invalid=row.get('invalid'),
                    robot_contact_steps=row['diagnostic_robot_contact_steps'],forbidden_contact_steps=row['forbidden_contact_steps'],reward=row['learning_reward'])
            details={k:copy.deepcopy(v) for k,v in cell.items() if k not in ('values','bucket_xy_m','candidate_id','context_id','grid_index','case_grid_index','q10','q14')}
            grid.append(dict(candidate_id=cell['candidate_id'],grid_index=cell['grid_index'],case_grid_index=cell['case_grid_index'],values=cell['values'],bucket_xy_m=cell['bucket_xy_m'],
                admission=details,discovery=observed,selected=bool(selected.get(context) and selected[context]['candidate_id']==cell['candidate_id'])))
        expected.append(dict(context_id=context,source_record_id=by_context[context]['id'],source_record_sha256=digest(by_context[context]),baseline_values=by_context[context]['values'],bucket_xy_m=by_context[context]['bucket_xy_m'],
            grid=grid,selection=selected.get(context),repeats={v:statistics([r for r in rows if r['context_id']==context and r['variant']==v]) for v in ('baseline','selected')}))
    checks.equal(report['cases'],expected,'All120 cells and case summaries reconstructed without filtering')


def audit(study,source,assets,*,parent_study,parent_audit,admission_only=False):
    started=time.monotonic();checks=replay.AuditChecks()
    study,source,assets,parent_study,parent_audit=map(Path,(study,source,assets,parent_study,parent_audit))
    manifest=replay.check_manifest(study,checks)
    spec,base,report=read(study/'contract.json'),read(study/'request-contract.json'),read(study/'report.json')
    for name,value in [('contract',spec),('request-contract',base)]:
        unsigned=dict(value);fp=unsigned.pop('fingerprint');checks.equal(digest(unsigned),fp,name+'/whole fingerprint')
    checks.equal((spec['version'],base['version'],report['version']),(STUDY_VERSION,)*3,'Literal new study namespace')
    checks.equal(report['contract_fingerprint'],spec['fingerprint'],'Report binds final contract')
    checks.equal(report['evidence_sha256'],manifest,'Complete saved evidence identity')
    for key,value in dict(grid_protocol=protocol(),budget=budget(),training=False,noise=False,posthoc_development=True).items():
        checks.equal(base[key],value,'Frozen literal protocol/'+key)
    replay.check_source_files(source,spec['source_sha256'],checks,'Executed source snapshot')
    checks.equal(sha(Path(__file__)),spec['source_sha256']['sircus/mjlab/height_tilt_verify.py'],'Independent auditor frozen before execution')
    replay.audit_packages(spec['packages'],checks)
    parent,original,bindings=authenticate_parent(parent_study,parent_audit,checks)
    checks.equal(spec['parent_bindings'],bindings,'Authenticated original parent provenance')
    for key in ('scene','identity','packages'):checks.equal(spec[key],parent[key],'Unchanged original identity/'+key)
    for name,expected in spec['identity']['asset_sha256'].items():checks.equal(sha(assets.parent/name),expected,'Unchanged robot asset/'+name)
    for name in ('calibration.json','verification.json','setup-home.json','prior.json'):
        checks.equal(sha(study/name),sha(parent_study/name),'Unchanged inherited configuration/'+name)
    for name,key in [('plan.json','plan_sha256'),('parent-rows.json','parent_rows_sha256'),('request-contract.json','request_contract_sha256'),
                     ('admission-matrix.json','admission_matrix_sha256'),('admission-lock.json','admission_lock_sha256'),('pre-native-plan.json','pre_native_plan_sha256')]:
        checks.equal(sha(study/name),spec[key],'Frozen exact input/'+name)
    expected_spec=dict(base,request_contract_sha256=sha(study/'request-contract.json'),admission_matrix_sha256=sha(study/'admission-matrix.json'),
        admission_lock_sha256=sha(study/'admission-lock.json'),admission_complete=spec['admission_complete'],pre_native_plan_sha256=sha(study/'pre-native-plan.json'))
    expected_spec.pop('fingerprint');expected_spec['fingerprint']=digest(expected_spec)
    checks.equal(spec,expected_spec,'Final contract extends exact pre-admission contract only')
    sources=read(study/'parent-rows.json');checks.equal(len(sources),7,'Only three controls and four source cases')
    ids={r['id'] for r in sources};lookup={r['id']:r for r in original}
    checks.equal(sources,[lookup[r['id']] for r in sources],'Parent rows copied exactly')
    checks.equal(len(ids),7,'Distinct parent row identities')
    plan,matrix=read(study/'plan.json'),read(study/'admission-matrix.json');audit_grid(plan,sources,checks)
    prior=read(study/'prior.json');home=spec['identity']['setup']['home']
    calibration=replay.check_calibration(study,spec['identity']['calibration'],checks)
    admission=audit_matrix(study,plan,matrix,prior,home,checks)
    lock=read(study/'admission-lock.json')
    checks.equal((lock['version'],lock['admission_matrix_sha256'],lock['request_contract_sha256']),
                 (STUDY_VERSION,sha(study/'admission-matrix.json'),sha(study/'request-contract.json')),'Matrix lock exact identities')
    checks.equal(lock['completed'],admission['attempted'],'Lock counts all completed cells')
    checks.require(type(lock['complete']) is bool,'Admission lock complete flag is boolean')
    checks.equal(lock['complete'],spec['admission_complete'],'Contract preserves lock completion state')
    checks.require(not lock['complete'] or admission['not_attempted']==0,'Complete admission requires all120 attempts')
    checks.require(0<=lock['cpu_elapsed_s']<=3600,'Measured CPU admission within one-hour cap')
    checks.numeric(report['cpu_elapsed_s'],lock['cpu_elapsed_s'],'Report preserves CPU elapsed',0)
    checks.require(lock['locked_unix_s']>=base['created_unix_s'],'Matrix locked after immutable request contract')
    pre=fixed_requests(sources,'controls_pre',checks)+fixed_requests(sources,'baseline',checks)
    admitted={r['candidate_id'] for r in matrix if r['passed'] is True}
    pre += [r for r in plan if r['id'] in admitted]
    checks.equal(read(study/'pre-native-plan.json'),pre,'All and only admitted candidates plus fixed originals/controls')
    rows=[];selection=[];gates={};traces={};setup={}
    if admission_only:
        checks.require(report['status'] in ('admission_complete','admission_stopped'),'Admission-only report phase')
        checks.require(not (study/'rows.jsonl').exists(),'Admission-only proof cannot include native evidence')
    else:
        checks.equal((report['status'],report['phase']),('completed','completed'),'Full native study completed')
        checks.equal(admission['not_attempted'],0,'All120 CPU candidates attempted before native')
        checks.require(0<=report['native_elapsed_s']<=3600,'Native acquired worker within one-hour cap')
        receipt=read(study.with_name(study.name+'-native-budget.json'))
        checks.require(lock['locked_unix_s']<=receipt['acquired_unix_s'],'Exact matrix frozen before native lease')
        checks.require(60<=receipt['limit_s']<=3600,'Native lease uses finite authorized cap')
        rows=replay.read_records(study/'rows.jsonl');selection=expected_selection(rows)
        checks.equal(read(study/'selection.json'),selection,'Independent capture-only quality/reward/index selection')
        post=repeat_requests(plan,selection)+fixed_requests(sources,'controls_post',checks)
        checks.equal(read(study/'post-selection-plan.json'),post,'Only frozen selected captures repeated, plus final controls')
        checks.require(120+sum(r['phase']!='discovery' for r in rows)<=162,'All candidates and repeats fit162 budget including declines')
        audit_native_requests(rows,pre+post,sources,matrix,checks);v3.check_rows(rows,checks)
        setup=calibrated.check_setup_evidence(study,rows,prior,home,checks)
        old.audit_native_options(study/'native',spec['scene'],calibration,assets,checks)
        for name in ('effective-options.json','native-compiled-geometry.json'):
            checks.equal(read(study/'native'/name),read(parent_study/'native'/name),'Actual native physics/geometry unchanged/'+name)
        traces=v3.rescore_rows(study/'native',rows,spec['scene'],calibration,spec['identity']['trace_schema'],assets,checks)
        files={}
        for row in rows:
            checks.equal((row['effective_options_file'],row['effective_options_sha256']),('effective-options.json',sha(study/'native/effective-options.json')),row['id']+'/actual native options identity')
            files.setdefault(row['native_cache_file'],[]).append(row)
        for name,members in files.items():
            with np.load(contained(study/'native',name),allow_pickle=False) as data:forces=data['forces_N']
            for row in members:
                count=int(np.count_nonzero(forces[:int(row['physics_steps']),int(row['native_cache_row']),4]>.01))
                checks.equal(row['diagnostic_robot_contact_steps'],count,row['id']+'/measured robot contacts before score terminal')
        for phase,key in [('controls_pre','pre'),('controls_post','post')]:
            gates[key]=v3.control_gate([r for r in rows if r['phase']==phase])
            checks.equal(gates[key]['status'],'passed',phase+'/unchanged direct/table/miss controls')
            actual=copy.deepcopy(report['control_gates'][key]);actual['counts'].setdefault('invalid_physics',0)
            checks.equal(actual,gates[key],phase+'/all control counts independently checked')
    audit_report(report,spec,sources,matrix,rows,selection,checks)
    return dict(version=VERSION,status='passed',full_checks_verified=True,native_checks_verified=not admission_only,
        checks_count=checks.count,checks=checks.entries,failures=[],contract_fingerprint=spec['fingerprint'],
        report_sha256=sha(study/'report.json'),evidence_manifest_sha256=sha(study/'evidence-manifest.json'),
        admission_matrix_sha256=sha(study/'admission-matrix.json'),auditor_sha256=sha(Path(__file__)),
        admission_summary=admission,requested_configurations=report['requested_configurations'],native_requests=len(rows),native_releases=len(traces),
        selection=selection,repeats=report['repeats'],control_gates=gates,setup_evidence=setup,elapsed_s=time.monotonic()-started,
        maximum_numeric_error=checks.numeric_error,scope='Independent CPU matrix/provenance audit only; no native validity claim' if admission_only else 'Independent exact-grid, admission, dense native64 and v3 scoring audit; no training/generalization claim')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('study','source','assets','parent-study','parent-audit','output'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--admission-only',action='store_true');a=p.parse_args()
    if a.output.exists() or a.output.resolve().is_relative_to(a.study.resolve()):raise ValueError('Fresh independent audit outside immutable evidence required')
    try:result=audit(a.study,a.source,a.assets,parent_study=a.parent_study,parent_audit=a.parent_audit,admission_only=a.admission_only)
    except Exception as error:
        a.output.write_text(json.dumps(dict(version=VERSION,status='failed',failures=[type(error).__name__+': '+str(error)]),indent=2)+'\n');raise
    a.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:result[k] for k in ('status','checks_count','native_checks_verified','admission_summary','requested_configurations','native_requests','repeats')}))


if __name__=='__main__':main()
