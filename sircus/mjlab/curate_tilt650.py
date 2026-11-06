"""Export audited fixed 30-degree replays without raw poses or inherited success.

The shared exporter helpers use only the standard library. No producer, solver,
capture scorer or independent-auditor implementation is imported here.
"""

from collections import Counter
import argparse
import csv
import json
from pathlib import Path

from .curate_xy650 import (
    boolean, captured, checked_gate, checked_statistics, digest, fingerprint,
    flight, geometry, hash_map, identifier, number, read, relative_file,
    require, sha, statistics, write_json,
)


VERSION = "sircus-tilt30-650-curation-v1"
STUDY_VERSION = "sircus-tilt30-650-verification-v1"
AUDIT_VERSION = "sircus-tilt30-650-independent-audit-v1"
PARENT_FP = "7db7440c12005c39808ee4dfbf831d8b81000cda048a8d4eddc896d9b1d31137"
PARENT_AUDIT_SHA = "9a605b7b8fbb1cb99db966ae92f6488aab0895a5db4bc9116b05e5a7fdc0b439"
SOURCE_IDS = tuple("coarse-64-xy-grid-%04d-r0" % i for i in (43,70,96,151,177))
RECIPES = (([-.01,.06],.41),([-.01,.08],.41),([0.,.04],.39),([0.,.08],.41),([.01,.04],.39))
STEPS = (64,128)
PHASES = ("controls_pre","verification","controls_post")
SCOPE = ("Five prescribed, previously unverified 30-degree recipes; three fresh nominal "
         "repeats per resolution with matched controls. Capture repeatability and clean "
         "capture are separate gates; historical parent rim captures remain rim captures. "
         "No new search, PPO, noise, held-out generalization, grasp/transit, actuator, "
         "perception or physical validation.")


def settings(raw):
    fixed = dict(source_count=5,source_ids=list(SOURCE_IDS),repeats_per_resolution=3,
                 substeps=list(STEPS),control_requests=24,verification_requests=30,max_requests=54,
                 minimum_clearance_m=.005,max_drift_350ms_m=.005)
    require(all(raw.get(k)==v for k,v in fixed.items()),"Unexpected fixed 54-request protocol")
    return dict(**fixed,nominal_only=True,new_search=False,new_ppo=False,noise_study=False)


def verified_inputs(study,audit_path):
    study,audit_path=Path(study).resolve(),Path(audit_path).resolve()
    spec,report,audit=(read(p) for p in (study/'contract.json',study/'report.json',audit_path))
    require(spec.get('version')==report.get('version')==STUDY_VERSION,"Unexpected study version")
    require(report.get('status')==report.get('phase')=='completed',"Study is not completed")
    require(spec.get('evidence_kind')=='native_calibrated_tilt30_verification',"Unexpected evidence kind")
    checks=audit.get('checks')
    require(audit.get('version')==AUDIT_VERSION and audit.get('status')=='passed'
            and isinstance(checks,list) and checks and type(audit.get('checks_count')) is int
            and audit['checks_count']==len(checks) and all(isinstance(c,dict) and c.get('passed') is True for c in checks)
            and audit.get('failures')==[],"Full exact-version independent audit must pass")
    require(audit.get('report_sha256')==sha(study/'report.json'),"Audit/report bytes differ")
    require(audit.get('evidence_manifest_sha256')==sha(study/'evidence-manifest.json'),"Audit/manifest bytes differ")
    unsigned=dict(spec);fp=unsigned.pop('fingerprint',None)
    require(fp==digest(unsigned)==report.get('contract_fingerprint')==audit.get('contract_fingerprint'),"Contract fingerprint binding differs")
    scene_fp=digest(dict(scene=spec['scene'],calibration=spec['calibration'],asset_sha256=spec['asset_sha256'],capture_version='sircus-calibrated-native-capture-v2'))
    require(scene_fp==spec.get('scene_fingerprint')==report.get('scene_fingerprint'),"Static scene fingerprint differs")
    if 'scene_fingerprint' in audit:require(audit['scene_fingerprint']==scene_fp,"Audit static scene fingerprint differs")
    require(spec.get('parent_contract_fingerprint')==report.get('parent_contract_fingerprint')==PARENT_FP,"Historical parent fingerprint differs")
    require(spec['input_bindings'].get('parent_audit_sha256')==PARENT_AUDIT_SHA,"Historical external proof binding differs")
    bindings=hash_map(spec['input_bindings']['parent'])
    require(set(bindings)=={'contract.json','report.json','evidence-manifest.json','records.jsonl','groups.json','initial-groups.json'},"Complete exact parent input bindings required")
    manifest=hash_map(read(study/'evidence-manifest.json'))
    require(report.get('evidence_sha256')==manifest and manifest.get('contract.json')==sha(study/'contract.json'),"Report/contract manifest differs")
    for name,h in manifest.items():require(sha(relative_file(study,name))==h,"Changed evidence bytes: "+name)
    actual={str(p.relative_to(study)) for p in study.rglob('*') if p.is_file() and p.name not in ('report.json','evidence-manifest.json') and not p.name.startswith('audit')}
    require(set(manifest)==actual,"Manifest does not cover every evidence artifact")
    require(report['settings']==spec['settings'],"Executed and reported settings differ")
    require(report.get('gate')==audit.get('gate'),"Independent gate differs")
    rows=[json.loads(s) for s in (study/'records.jsonl').read_text().splitlines() if s]
    require(len(rows)==report['counts']['attempts']==audit.get('requested_trials'),"Requested trial accounting differs")
    require(sum(bool(r.get('native_release')) for r in rows)==audit.get('native_releases'),"Native release accounting differs")
    return study,spec,report,audit,manifest,rows


def sources_and_plan(study,spec,report):
    sources=read(study/'source-selection.json')
    require(isinstance(sources,list) and tuple(r['id'] for r in sources)==SOURCE_IDS,"Literal five-source selection differs")
    raw_groups=read(study/'groups.json');groups={g['id']:g for g in raw_groups}
    require(len(raw_groups)==len(groups)==7,"Exactly two controls and five verification targets required")
    published={g['id']:g for g in report['groups']}
    require(set(groups)==set(published),"Published group identities differ")
    for ident,g in groups.items():
        require(all(published[ident].get(k)==g.get(k) for k in ('values','bucket_xy_m','role','phase','eligible','query_kind','translation_m','query_fingerprint','source_query_fingerprint','source_record_id','source_record_sha256')),"Published geometry differs from frozen group")
    configurations={k:geometry(g,spec['scene_fingerprint']) for k,g in published.items()}
    radius=number(spec['scene']['ball']['radius_m'])
    require(radius>0 and all(abs(c['release_center_height_m']-c['board_height_m']-c['gap_m']-radius)<=1e-12 for c in configurations.values()),"Configured release height differs")
    controls=[g for g in raw_groups if g['role'].endswith('_control')]
    require([g['role'] for g in controls]==['positive_control','negative_control'],"Historical control pair differs")
    for g in controls:
        require(g['query_kind']=='historical_control' and g['translation_m']==[0.,0.],"Historical control was translated")
    parent=[]
    for i,(source,(translation,gap)) in enumerate(zip(sources,RECIPES)):
        require(source['phase']=='coarse' and source['substeps']==64 and captured(source)
                and source.get('route_quality')==1 and source.get('rim_band_contact') is True
                and source.get('invalid_physics') is False,"Historical source rim capture was relabelled")
        require(source['translation_m']==translation and source['values'][2:]==[.38,30.,gap],"Literal source geometry differs")
        group=groups['tilt30-%02d'%i];cfg=configurations[group['id']]
        require(group.get('freeze_joints') is True and group['q10']==source['q10'] and group['q14']==source['q14'],"Source joint branch was not frozen")
        require(group['values']==source['values'] and group['bucket_xy_m']==source['bucket_xy_m']
                and cfg['source_record_id']==source['id'] and cfg['source_record_sha256']==digest(source)
                and cfg['source_query_fingerprint']==source['query_fingerprint'],"Parent source provenance differs")
        require(cfg['translation_m']==translation and abs(cfg['distance_m']-.65)<=1e-12
                and cfg['board_height_m']==.38 and cfg['tilt_deg']==30. and cfg['gap_m']==gap,"Fixed candidate650 geometry differs")
        parent.append(dict(record_id=identifier(source['id']),record_sha256=digest(source),
            query_fingerprint=fingerprint(source['query_fingerprint']),execution_fingerprint=fingerprint(source['execution_fingerprint']),
            parent_route_quality=1,parent_capture=True,parent_rim_band_contact=True,configuration=cfg))
    require(len({tuple(g['values']+g['bucket_xy_m']) for g in raw_groups if g['role']=='verification'})==5,"Verification targets are not distinct")
    plan=read(study/'initial-plan.json')
    require(set(plan)==set(PHASES) and [len(plan[p]) for p in PHASES]==[12,30,12],"Literal54-request plan differs")
    planned={}
    for phase in PHASES:
        expected_groups={g['id'] for g in raw_groups if (g['role']=='verification')==(phase=='verification')}
        require(Counter((r['group_id'],r['substeps']) for r in plan[phase])==Counter({(g,s):3 for g in expected_groups for s in STEPS}),"Planned matched cohort differs")
        require(Counter((r['group_id'],r['substeps'],r['repeat_index']) for r in plan[phase])==Counter({(g,s,i):1 for g in expected_groups for s in STEPS for i in range(3)}),"Planned repeat identities differ")
        for row in plan[phase]:
            require(row['id'] not in planned and row['phase']==phase and type(row['repeat_index']) is int and row['repeat_index'] in (0,1,2),"Planned trial identity differs")
            g=groups[row['group_id']]
            require(all(row[k]==g[k] for k in ('values','bucket_xy_m','q10','q14','eligible','query_fingerprint','source_record_id','source_record_sha256')),"Plan changed a frozen target")
            planned[row['id']]=row
    require(len(planned)==54,"Fixed plan denominator differs")
    return configurations,parent,planned


def sanitize_records(records,configurations,planned):
    require(len({r['id'] for r in records})==len(records),"Duplicate requested trial ID")
    rows=[];observations={};seen=set()
    for r in records:
        cfg=configurations[r['group_id']];p=planned[r['id']]
        require(all(r.get(k)==p.get(k) for k in ('phase','group_id','role','substeps','repeat_index','values','bucket_xy_m','q10','q14','eligible','query_fingerprint','source_record_id','source_record_sha256')),"Executed request differs from frozen plan")
        native=boolean(r['native_release']);execution=r['execution_fingerprint']
        if native:
            fingerprint(execution);require(execution not in seen,"Duplicate actual execution fingerprint");seen.add(execution)
        else:require(execution is None,"Rejected request invents execution provenance")
        observed=flight(r['first_flight'],native=native);observations[r['id']]=observed;crossing=(observed or {}).get('first_crossing') or {}
        invalid=r.get('invalid');require(invalid is None or isinstance(invalid,str),"Invalid reason must be text or unknown")
        if invalid is not None:identifier(invalid)
        retention=number(r['max_retention_s']);require(0<=retention<=1+1e-10,"Invalid retention duration")
        rim=r.get('rim_band_contact');require(rim is None or type(rim) is bool,"Unexpected rim evidence")
        row=dict(id=identifier(r['id']),phase=r['phase'],group_id=identifier(r['group_id']),role=r['role'],substeps=r['substeps'],repeat_index=r['repeat_index'],
            source_record_id=cfg['source_record_id'],source_record_sha256=cfg['source_record_sha256'],source_query_fingerprint=cfg['source_query_fingerprint'],
            query_fingerprint=cfg['query_fingerprint'],execution_fingerprint=execution,translation_x_m=cfg['translation_m'][0],translation_y_m=cfg['translation_m'][1],
            board_x_m=cfg['board_xy_m'][0],board_y_m=cfg['board_xy_m'][1],board_height_m=cfg['board_height_m'],tilt_deg=cfg['tilt_deg'],release_gap_m=cfg['gap_m'],
            release_ball_center_height_m=cfg['release_center_height_m'],bucket_x_m=cfg['bucket_xy_m'][0],bucket_y_m=cfg['bucket_xy_m'][1],distance_m=cfg['distance_m'],
            parent_route_quality=1 if r['phase']=='verification' else None,eligible=boolean(r['eligible']),native_release=native,capture=captured(r),
            clean=captured(r) and r.get('route_quality')==2,rim_band_contact=rim,whole_ball_entry=boolean(r['whole_ball_entry']),invalid=invalid,
            invalid_physics=boolean(r['invalid_physics']),max_retention_s=retention,minimum_mouth_clearance_mm=number(r.get('minimum_mouth_clearance_mm'),optional=True),
            first_flight_version=(observed or {}).get('version'),first_flight_available=(observed or {}).get('available'),first_flight_reason=(observed or {}).get('reason'),
            first_flight_start_index=(observed or {}).get('start_index'),first_flight_end_index=(observed or {}).get('end_index'),
            first_flight_start_time_s=(observed or {}).get('start_time_s'),first_flight_end_time_s=(observed or {}).get('end_time_s'),
            first_flight_cutoff_reason=(observed or {}).get('cutoff_reason'),first_flight_cutoff_channels='|'.join((observed or {}).get('cutoff_contact_channels',[])) if observed else None,
            first_flight_crossing_time_s=crossing.get('time_s'),first_flight_crossing_interval_index=crossing.get('interval_index'),first_flight_crossing_fraction=crossing.get('fraction'),
            first_flight_mouth_clearance_m=(observed or {}).get('mouth_clearance_m'),first_flight_closest_opening_distance_m=(observed or {}).get('closest_opening_distance_m'),
            first_flight_sample_count=(observed or {}).get('sample_count'))
        rows.append(row)
    return rows,observations


def curate(study,audit_path,output):
    study,spec,report,audit,manifest,records=verified_inputs(study,audit_path)
    output=Path(output).resolve();require(not output.exists() and not output.is_relative_to(study),"New publication directory outside evidence required")
    declared=settings(spec['settings']);require(12<=len(records)<=54,"Fixed attempt budget exceeded")
    configurations,parent,planned=sources_and_plan(study,spec,report)
    rows,observations=sanitize_records(records,configurations,planned)
    counts=checked_statistics(report['counts'],records)
    phases=dict(Counter(r['phase'] for r in records));steps=dict(Counter(str(r['substeps']) for r in records))
    require(report['counts']['phase_counts']==phases and report['counts']['substep_counts']==steps,"Phase/resolution accounting differs")
    gates={k:checked_gate(report['control_gates'][k]) for k in ('controls_pre','controls_post')}
    if gates['controls_pre']['status']!='passed':
        require(phases=={'controls_pre':12} and gates['controls_post']['status']=='not_run',"Verification ran despite failed initial controls")
    else:require(phases==dict(controls_pre=12,verification=30,controls_post=12) and set(r['id'] for r in records)==set(planned),"Fixed54 execution accounting differs")
    controls=[]
    for c in report['controls']:
        require(c['phase'] in ('controls_pre','controls_post') and c['substeps'] in STEPS,"Unexpected control cohort")
        selected=[r for r in records if r['phase']==c['phase'] and r['group_id']==c['group_id'] and r['substeps']==c['substeps']]
        controls.append(dict(phase=c['phase'],group_id=identifier(c['group_id']),role=identifier(c['role']),substeps=c['substeps'],stats=checked_statistics(c['stats'],selected)))
    require(len(controls)==8,"All control cohorts must remain visible")
    verified=[]
    require(len(report['verification'])==5,"All five prescribed targets must remain visible")
    for i,result in enumerate(report['verification']):
        cfg=configurations[result['id']];require(cfg['id']=='tilt30-%02d'%i and result.get('parent_route_quality')==1,"Historical source order or rim label differs")
        selected=[r for r in records if r['group_id']==cfg['id']]
        by_step={str(s):checked_statistics(result['by_substeps'][str(s)],[r for r in selected if r['substeps']==s]) for s in STEPS}
        raw=result['qualification'];q={k:boolean(raw[k]) for k in ('qualified','capture_qualified','clean_qualified','clean_both','capture_both','physics_valid','oracle_agreement','parent_source_capture')}
        q.update(status=identifier(raw['status']),max_drift_m=number(raw.get('max_drift_m'),optional=True))
        present=len(selected)==6 and all(sum(r['substeps']==s for r in selected)==3 for s in STEPS)
        capture=present and all(captured(r) for r in selected);clean=present and all(captured(r) and r.get('route_quality')==2 for r in selected)
        physics=present and all(r.get('native_release') and not r.get('invalid') and r.get('invalid_physics') is False for r in selected)
        oracle=present and all(r.get('scalar_score_audit_passed') is True for r in selected)
        require(q['parent_source_capture'] is True and (q['capture_both'],q['clean_both'],q['physics_valid'],q['oracle_agreement'])==(capture,clean,physics,oracle),"Fresh repeated outcome qualification differs")
        common=all(g['status']=='passed' for g in gates.values()) and physics and oracle and q['max_drift_m'] is not None and 0<=q['max_drift_m']<=.005
        capture_ok=bool(common and capture);clean_ok=bool(common and clean)
        require((q['capture_qualified'],q['clean_qualified'],q['qualified'])==(capture_ok,clean_ok,clean_ok),"Capture and clean gates were conflated")
        expected='pending' if not present else 'qualified' if clean_ok else 'capture_only' if capture_ok else 'failed'
        require(q['status']==expected,"Fresh qualification status differs")
        sensitivity=result['sensitivity'];paired=sensitivity['paired_attempts'];agreement=sensitivity['capture_agreement']
        require(type(paired) is int and type(agreement) is int and 0<=agreement<=paired<=3,"Paired denominator differs")
        drift=number(sensitivity.get('nominal_position_delta_350ms_max_m'),optional=True)
        require(drift==q['max_drift_m'] and (drift is None or drift>=0),"Paired drift differs")
        verified.append(dict(configuration=cfg,parent_route_quality=1,by_substeps=by_step,qualification=q,
            sensitivity=dict(paired_attempts=paired,capture_agreement=agreement,nominal_position_delta_350ms_max_m=drift),
            executions=[dict(id=identifier(r['id']),substeps=r['substeps'],repeat_index=r['repeat_index'],execution_fingerprint=r['execution_fingerprint'],first_flight=observations[r['id']]) for r in selected]))
    gate=report['gate'];clean_count=sum(v['qualification']['clean_qualified'] for v in verified);capture_count=sum(v['qualification']['capture_qualified'] for v in verified)
    pre=gates['controls_pre']['status']=='passed';post=gates['controls_post']['status']=='passed'
    expected='blocked_controls' if not pre or not post else 'passed' if clean_count else 'capture_only' if capture_count else 'failed'
    require(gate==dict(status=expected,controls_pre_passed=pre,controls_post_passed=post,qualified_settings=clean_count,capture_qualified_settings=capture_count,more_ppo_started=False),"Final capture/clean gate differs")
    candidates=[r for r in records if r['phase']=='verification'];control_records=[r for r in records if r['phase']!='verification']
    summary=dict(version=VERSION,study_version=STUDY_VERSION,study_id=identifier(study.name),status='completed_independently_audited',scope=SCOPE,settings=declared,
        counts=dict(**counts,phase_counts=phases,substep_counts=steps),candidate_counts=statistics(candidates),control_counts=statistics(control_records),gate=gate,
        control_gates=gates,controls=controls,parent_sources=parent,verification=verified,
        provenance=dict(contract_fingerprint=spec['fingerprint'],scene_fingerprint=spec['scene_fingerprint'],parent_contract_fingerprint=PARENT_FP,
            parent_external_audit_sha256=PARENT_AUDIT_SHA,parent_input_sha256=hash_map(spec['input_bindings']['parent']),
            contract_sha256=sha(study/'contract.json'),report_sha256=sha(study/'report.json'),independent_audit_sha256=sha(audit_path),
            independent_audit_version=AUDIT_VERSION,independent_auditor_sha256=fingerprint(audit['auditor_sha256']),independent_audit_checks=audit['checks_count'],
            evidence_manifest_sha256=sha(study/'evidence-manifest.json'),evidence_sha256=manifest,executed_source_sha256=hash_map(spec['source_sha256']),asset_sha256=hash_map(spec['asset_sha256'])),
        privacy='Configured geometry, historical labels, fresh outcomes, diagnostic scalars and hashes only; no measured crossing XY, sampled poses, joints, trajectories, credentials or host paths.')
    dependencies=audit.get('auditor_identity',{}).get('dependencies_sha256')
    if dependencies:summary['provenance']['auditor_dependencies_sha256']=hash_map(dependencies)
    output.mkdir(parents=True);write_json(output/'summary.json',summary)
    with (output/'requested-shots.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]),lineterminator='\n');writer.writeheader();writer.writerows(rows)
    write_json(output/'checksums.json',{p.name:sha(p) for p in sorted(output.iterdir()) if p.is_file()})
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study',type=Path,required=True);parser.add_argument('--audit',type=Path);parser.add_argument('--output',type=Path,required=True)
    a=parser.parse_args();result=curate(a.study,a.audit or a.study/'audit-tilt650.json',a.output)
    print(json.dumps(dict(status=result['status'],candidate_counts=result['candidate_counts'],gate=result['gate'])))


if __name__=='__main__':main()
