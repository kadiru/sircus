"""Publish audited impact-point parameters and scalar outcomes, never raw sensing.

Only standard-library publication helpers are reused. This is not a native
contact scorer or an independent audit; both must already have completed.
"""
from collections import Counter
import argparse
import csv
import itertools
import json
import math
from pathlib import Path

from .curate_xy650 import (
    boolean, captured, checked_gate, checked_statistics, digest, fingerprint,
    flight, hash_map, identifier, number, read, relative_file, require, sha,
    statistics, write_json,
)

VERSION = 'sircus-impact650-curation-v1'
STUDY_VERSION = 'sircus-impact650-study-v1'
AUDIT_VERSION = 'sircus-impact650-independent-audit-v1'
RULE = 'sircus-board-center-plane-forward-lateral-impact-v1'
SOURCE_IDS = ('refinement-64-refine-s02-h1t0g2-r0', 'coarse-64-xy-grid-0043-r0', 'coarse-64-xy-grid-0096-r0')
FORWARD = (-.04,-.03,-.02,-.01,0.,.01,.02,.03,.04)
LATERAL = (-.02,-.01,0.,.01,.02)
STEPS = (64,128)
PHASES = ('controls_pre','coarse','verification','controls_post')
PARENT_FP = '7db7440c12005c39808ee4dfbf831d8b81000cda048a8d4eddc896d9b1d31137'
PARENT_AUDIT_SHA = '9a605b7b8fbb1cb99db966ae92f6488aab0895a5db4bc9116b05e5a7fdc0b439'
SCOPE = ('Bounded impact-point DEVELOPMENT search; three fixed families, 135 coarse requests, '
         'at most 30 fresh matched nominal verification requests and 24 controls. Board-center '
         'distance is 650 mm; nominal center-plane drop-ray distance varies. No new PPO, noise, '
         'held-out generalization, grasp/transit, actuator, perception or physical validation.')


def settings(raw):
    expected = dict(source_ids=list(SOURCE_IDS),families=3,forward_m=list(FORWARD),lateral_m=list(LATERAL),
        coarse_requests=135,control_requests=24,verification_sources=5,repeats_per_resolution=3,
        verification_requests=30,max_requests=189,user_request_cap=200,worker_wall_limit_s=3600,
        substeps=list(STEPS),minimum_clearance_m=.005,max_drift_350ms_m=.005,impact_rule=RULE)
    require(all(raw.get(k)==v for k,v in expected.items()),'Unexpected frozen impact189 protocol')
    return dict(**expected,nominal_only=True,new_ppo=False,noise_study=False,
        gap_convention='Vertical ball-bottom clearance above nominal offset point on the board CENTER plane; not physical top face.',
        distance_convention='Board-center distance stays 650 mm; nominal drop-ray horizontal distance varies.')


def verified_inputs(study,audit_path):
    study,audit_path = Path(study).resolve(),Path(audit_path).resolve()
    spec,report,audit = (read(p) for p in (study/'contract.json',study/'report.json',audit_path))
    require(spec.get('version')==report.get('version')==STUDY_VERSION,'Unexpected study version')
    require(report.get('status')==report.get('phase')=='completed','Study is not completed')
    require(spec.get('evidence_kind')=='native_calibrated_impact650_development','Unexpected evidence kind')
    checks = audit.get('checks')
    require(audit.get('version')==AUDIT_VERSION and audit.get('status')=='passed' and audit.get('failures')==[]
        and isinstance(checks,list) and checks and type(audit.get('checks_count')) is int
        and audit['checks_count']==len(checks) and all(isinstance(c,dict) and c.get('passed') is True for c in checks),
        'Full exact-version independent audit must pass')
    require(audit.get('report_sha256')==sha(study/'report.json'),'Audit/report bytes differ')
    require(audit.get('evidence_manifest_sha256')==sha(study/'evidence-manifest.json'),'Audit/manifest bytes differ')
    unsigned = dict(spec);fp = unsigned.pop('fingerprint',None)
    require(fp==digest(unsigned)==report.get('contract_fingerprint')==audit.get('contract_fingerprint'),'Contract fingerprint binding differs')
    physical = digest(dict(scene=spec['scene'],calibration=spec['calibration'],asset_sha256=spec['asset_sha256'],capture_version='sircus-calibrated-native-capture-v2'))
    scene = digest(dict(physical_scene_fingerprint=physical,impact_rule=RULE))
    require(spec.get('physical_scene_fingerprint')==report.get('physical_scene_fingerprint')==physical,'Physical scene fingerprint differs')
    require(spec.get('scene_fingerprint')==report.get('scene_fingerprint')==scene,'Action-bound scene fingerprint differs')
    for key,expected in (('scene_fingerprint',scene),('physical_scene_fingerprint',physical)):
        if key in audit: require(audit[key]==expected,'Independent '+key+' differs')
    require(spec.get('parent_contract_fingerprint')==report.get('parent_contract_fingerprint')==PARENT_FP,'Historical parent identity differs')
    require(spec['input_bindings'].get('parent_audit_sha256')==PARENT_AUDIT_SHA,'Historical external proof differs')
    parent = hash_map(spec['input_bindings']['parent'])
    require(set(parent)=={'contract.json','report.json','evidence-manifest.json','records.jsonl','groups.json','initial-groups.json'},'Complete historical bindings required')
    manifest = hash_map(read(study/'evidence-manifest.json'))
    require(report.get('evidence_sha256')==manifest and manifest.get('contract.json')==sha(study/'contract.json'),'Report/contract manifest differs')
    for name,h in manifest.items(): require(sha(relative_file(study,name))==h,'Changed evidence bytes: '+name)
    actual = {str(p.relative_to(study)) for p in study.rglob('*') if p.is_file() and p.name not in ('report.json','evidence-manifest.json') and not p.name.startswith('audit')}
    require(set(manifest)==actual,'Manifest does not cover every evidence artifact')
    require(report['settings']==spec['settings'],'Executed and reported settings differ')
    require(report.get('gate')==audit.get('gate'),'Independent gate differs')
    rows = [json.loads(line) for line in (study/'records.jsonl').read_text().splitlines() if line]
    require(len(rows)==report['counts']['attempts']==audit.get('requested_trials'),'Requested trial accounting differs')
    require(sum(bool(r.get('native_release')) for r in rows)==audit.get('native_releases'),'Native release accounting differs')
    return study,spec,report,audit,manifest,rows


def vector(value,size):
    require(isinstance(value,list) and len(value)==size,'Incomplete nominal geometry vector')
    return [number(v) for v in value]


def near(a,b):
    return len(a)==len(b) and all(abs(x-y)<=1e-12 for x,y in zip(a,b))


def geometry(g,scene_fp,radius):
    action,bucket,offset = vector(g['values'],5),vector(g['bucket_xy_m'],2),vector(g['impact_offset_m'],2)
    require(g['board_xy_m']==action[:2] and g['board_height_m']==action[2] and g['tilt_deg']==action[3] and g['gap_m']==action[4],'Configured action fields disagree')
    distance = math.dist(bucket,action[:2]);require(distance>.15 and abs(number(g['distance_m'])-distance)<=1e-12,'Board-center distance differs')
    d = [(bucket[i]-action[i])/distance for i in range(2)];angle = math.radians(action[3])
    forward = [math.cos(angle)*d[0],math.cos(angle)*d[1],-math.sin(angle)]
    lateral = [-d[1],d[0],0.]
    point = [action[i]+offset[0]*forward[i]+offset[1]*lateral[i] for i in range(3)]
    release = point[:];release[2] += action[4]+radius
    impact_distance = math.dist(bucket,point[:2])
    require(near(vector(g['impact_point_world_m'],3),point),'Nominal center-plane point differs')
    require(near(vector(g['release_center_m'],3),release) and abs(number(g['release_center_height_m'])-release[2])<=1e-12,'Nominal release point differs')
    require(abs(number(g['impact_to_bucket_horizontal_m'])-impact_distance)<=1e-12,'Nominal drop-ray distance differs')
    role,kind = identifier(g['role']),identifier(g['query_kind'])
    require(kind in ('historical_control','impact_grid','nominal_verification'),'Unexpected query kind')
    query = dict(scene_fingerprint=scene_fp,values=action,bucket_xy_m=bucket,impact_offset_m=offset,impact_rule=RULE,
        source_record_id=g['source_record_id'],source_record_sha256=g['source_record_sha256'])
    require(fingerprint(g['query_fingerprint'])==digest(query),'Configured query fingerprint differs')
    source_query = g.get('source_query_fingerprint')
    if source_query is not None: fingerprint(source_query)
    return dict(id=identifier(g['id']),phase=identifier(g['phase']),role=role,query_kind=kind,
        family_id=identifier(g['family_id']),seed_id=identifier(g['seed_id']),eligible=boolean(g['eligible']),
        source_record_id=identifier(g['source_record_id']),source_record_sha256=fingerprint(g['source_record_sha256']),
        source_query_fingerprint=source_query,query_fingerprint=g['query_fingerprint'],
        translation_m=vector(g['translation_m'],2),board_xy_m=action[:2],bucket_xy_m=bucket,
        board_height_m=action[2],tilt_deg=action[3],gap_m=action[4],distance_m=distance,
        impact_offset_m=offset,nominal_center_plane_point_m=point,nominal_release_center_m=release,
        release_center_height_m=release[2],nominal_point_to_bucket_horizontal_m=impact_distance)


def layouts_and_plan(study,spec,report):
    sources = read(study/'source-selection.json')
    require(isinstance(sources,list) and tuple(r['id'] for r in sources)==SOURCE_IDS,'Literal three source families differ')
    raw = read(study/'groups.json');groups = {g['id']:g for g in raw}
    published = {g['id']:g for g in report['groups']}
    require(len(raw)==len(groups)==len(published) and set(groups)==set(published),'Unique published group identities required')
    radius = number(spec['scene']['ball']['radius_m']);require(radius>0,'Positive ball radius required')
    configurations = {key:geometry(g,spec['scene_fingerprint'],radius) for key,g in published.items()}
    for key,g in groups.items():
        require(all(published[key].get(k)==g.get(k) for k in ('phase','role','values','bucket_xy_m','eligible','query_kind','family_id','seed_id','impact_offset_m','impact_point_world_m','release_center_m','query_fingerprint','source_record_id','source_record_sha256')),'Published frozen geometry differs')
    initial = read(study/'initial-groups.json');require(len(initial)==137,'Exactly137 initial groups required')
    require([g['role'] for g in initial[:2]]==['positive_control','negative_control'],'Historical control pair differs')
    coarse = [g for g in raw if g['phase']=='coarse']
    require(len(coarse)==135 and {tuple([g['family_id']]+g['impact_offset_m']) for g in coarse}==set(itertools.product(('family-00','family-01','family-02'),FORWARD,LATERAL)),'Declared135-point impact grid differs')
    recipes = ([.38,27.5,.3918],[.38,30.,.41],[.38,30.,.39])
    parent_sources = []
    for i,source in enumerate(sources):
        require(captured(source) and source.get('route_quality')==1 and source.get('rim_band_contact') is True and source.get('invalid_physics') is False,'Historical source rim capture was relabelled')
        require(near(vector(source['values'],5)[2:],recipes[i]),'Literal source recipe differs')
        members = [g for g in coarse if g['family_id']==f'family-{i:02d}']
        for g in members:
            cfg = configurations[g['id']]
            require(g['values']==source['values'] and g['bucket_xy_m']==source['bucket_xy_m'] and g['q14']==source['q14'],'Fixed source board branch or geometry differs')
            require(cfg['seed_id']==source['id'] and cfg['source_record_id']==source['id'] and cfg['source_record_sha256']==digest(source) and cfg['source_query_fingerprint']==source['query_fingerprint'],'Source provenance differs')
            require(abs(cfg['distance_m']-.65)<=1e-12,'Candidate board-center distance is not650')
            if g['impact_offset_m']==[0.,0.]:require(g['q10']==source['q10'],'Zero-offset source dropper branch differs')
        parent_sources.append(dict(record_id=identifier(source['id']),record_sha256=digest(source),family_id=f'family-{i:02d}',historical_route_quality=1,historical_capture=True,query_fingerprint=fingerprint(source['query_fingerprint'])))
    selection = read(study/'verification-selection.json')
    targets = [g for g in raw if g['role']=='verification']
    require(len(selection)==len(targets)==len(report['verification'])<=5 and len(raw)==137+len(targets),'Verification selection denominator differs')
    require(len({tuple(r['values']+r['bucket_xy_m']+r['impact_offset_m']) for r in selection})==len(selection),'Verification targets are not distinct')
    for i,(g,source) in enumerate(zip(targets,selection)):
        cfg = configurations[g['id']]
        require(g['id']==f'impact-verify-{i:02d}' and source['phase']=='coarse' and source.get('native_release') and not source.get('invalid') and source.get('invalid_physics') is False,'Verification source is not native-valid coarse evidence')
        require(all(g[k]==source[k] for k in ('values','bucket_xy_m','impact_offset_m','q10','q14','family_id','seed_id')),'Frozen verification target differs from source')
        require(cfg['source_record_id']==source['id'] and cfg['source_record_sha256']==digest(source) and cfg['source_query_fingerprint']==source['query_fingerprint'],'Fresh source provenance differs')
    plan = read(study/'initial-plan.json');require(set(plan)=={'controls_pre','coarse','controls_post'} and [len(plan[k]) for k in ('controls_pre','coarse','controls_post')]==[12,135,12],'Literal initial159 request plan differs')
    plan['verification'] = read(study/'verification-plan.json')
    require(isinstance(plan['verification'],list) and len(plan['verification'])==6*len(targets),'Matched verification plan differs')
    planned = {}
    for phase in PHASES:
        for row in plan[phase]:
            require(row['id'] not in planned and row['phase']==phase and row['group_id'] in groups,'Planned trial identity differs')
            g = groups[row['group_id']]
            require(all(row[k]==g[k] for k in ('values','bucket_xy_m','impact_offset_m','q10','q14','eligible','query_fingerprint','source_record_id','source_record_sha256')),'Plan changed a frozen target')
            planned[row['id']] = row
        expected_groups = [g for g in raw if g['role'].endswith('_control')] if phase.startswith('controls_') else coarse if phase=='coarse' else targets
        expected = Counter({(g['id'],s,j):1 for g in expected_groups for s in ((64,) if phase=='coarse' else STEPS) for j in range(1 if phase=='coarse' else 3)})
        require(Counter((r['group_id'],r['substeps'],r['repeat_index']) for r in plan[phase])==expected,'Planned phase/resolution cohort differs')
    return configurations,parent_sources,planned,selection


def sanitize_records(records,configurations,planned):
    require(len({r['id'] for r in records})==len(records),'Duplicate requested trial ID')
    exported=[];diagnostics={};execution_ids=set()
    for r in records:
        require(r['id'] in planned and r['group_id'] in configurations,'Unplanned request')
        p,cfg = planned[r['id']],configurations[r['group_id']]
        require(all(r.get(k)==p.get(k) for k in ('phase','group_id','role','substeps','repeat_index','values','bucket_xy_m','impact_offset_m','q10','q14','eligible','query_fingerprint','source_record_id','source_record_sha256')),'Executed request differs from frozen plan')
        native = boolean(r['native_release']);execution = r.get('execution_fingerprint')
        if native:
            fingerprint(execution);require(execution not in execution_ids,'Duplicate execution fingerprint');execution_ids.add(execution)
        else: require(execution is None,'Rejected request invents execution provenance')
        diagnostic = flight(r.get('first_flight'),native=native);diagnostics[r['id']] = diagnostic
        invalid = r.get('invalid');require(invalid is None or isinstance(invalid,str),'Invalid reason must be text or unknown')
        if invalid is not None:identifier(invalid)
        retention = number(r['max_retention_s']);require(0<=retention<=1+1e-10,'Invalid retention duration')
        rim = r.get('rim_band_contact');require(rim is None or type(rim) is bool,'Unexpected rim evidence')
        point,release = cfg['nominal_center_plane_point_m'],cfg['nominal_release_center_m']
        exported.append(dict(id=identifier(r['id']),phase=identifier(r['phase']),group_id=identifier(r['group_id']),role=identifier(r['role']),substeps=r['substeps'],repeat_index=r['repeat_index'],
            family_id=cfg['family_id'],seed_id=cfg['seed_id'],source_record_id=cfg['source_record_id'],source_record_sha256=cfg['source_record_sha256'],source_query_fingerprint=cfg['source_query_fingerprint'],query_fingerprint=cfg['query_fingerprint'],execution_fingerprint=execution,
            board_x_m=cfg['board_xy_m'][0],board_y_m=cfg['board_xy_m'][1],board_height_m=cfg['board_height_m'],tilt_deg=cfg['tilt_deg'],nominal_plane_gap_m=cfg['gap_m'],
            forward_offset_m=cfg['impact_offset_m'][0],lateral_offset_m=cfg['impact_offset_m'][1],
            nominal_center_plane_point_x_m=point[0],nominal_center_plane_point_y_m=point[1],nominal_center_plane_point_z_m=point[2],
            nominal_release_center_x_m=release[0],nominal_release_center_y_m=release[1],nominal_release_center_z_m=release[2],
            bucket_x_m=cfg['bucket_xy_m'][0],bucket_y_m=cfg['bucket_xy_m'][1],board_center_distance_m=cfg['distance_m'],nominal_point_to_bucket_horizontal_m=cfg['nominal_point_to_bucket_horizontal_m'],
            eligible=boolean(r['eligible']),native_release=native,capture=captured(r),clean=captured(r) and r.get('route_quality')==2,rim_band_contact=rim,
            whole_ball_entry=boolean(r['whole_ball_entry']),invalid=invalid,invalid_physics=boolean(r['invalid_physics']),max_retention_s=retention,
            minimum_mouth_clearance_mm=number(r.get('minimum_mouth_clearance_mm'),optional=True),
            first_flight_available=(diagnostic or {}).get('available'),first_flight_mouth_clearance_m=(diagnostic or {}).get('mouth_clearance_m'),first_flight_closest_opening_distance_m=(diagnostic or {}).get('closest_opening_distance_m')))
    return exported,diagnostics


def curate(study,audit_path,output):
    study,spec,report,audit,manifest,records = verified_inputs(study,audit_path)
    output = Path(output).resolve();require(not output.exists() and not output.is_relative_to(study),'New publication directory outside evidence required')
    declared = settings(spec['settings']);require(12<=len(records)<=189,'Finite189 attempt budget exceeded')
    elapsed = number(report['elapsed_s']);require(0<=elapsed<=declared['worker_wall_limit_s'],'Worker-hour limit exceeded')
    cfg,parent_sources,planned,selection = layouts_and_plan(study,spec,report)
    rows,diagnostics = sanitize_records(records,cfg,planned)
    counts = checked_statistics(report['counts'],records)
    phases = dict(Counter(r['phase'] for r in records));steps = dict(Counter(str(r['substeps']) for r in records))
    require(report['counts']['phase_counts']==phases and report['counts']['substep_counts']==steps,'Phase/resolution accounting differs')
    gates = {k:checked_gate(report['control_gates'][k]) for k in ('controls_pre','controls_post')}
    if gates['controls_pre']['status']!='passed':
        require(phases=={'controls_pre':12} and gates['controls_post']['status']=='not_run' and not selection,'Candidates ran despite failed initial controls')
    else:
        expected = dict(controls_pre=12,coarse=135,controls_post=12)
        if selection:expected['verification']=6*len(selection)
        require(phases==expected and set(r['id'] for r in records)==set(planned),'Completed bounded execution accounting differs')
    controls = []
    for c in report['controls']:
        require(c['phase'] in ('controls_pre','controls_post') and c['substeps'] in STEPS,'Unexpected control cohort')
        cohort = [r for r in records if r['phase']==c['phase'] and r['group_id']==c['group_id'] and r['substeps']==c['substeps']]
        controls.append(dict(phase=c['phase'],group_id=identifier(c['group_id']),role=identifier(c['role']),substeps=c['substeps'],stats=checked_statistics(c['stats'],cohort)))
    require(len(controls)==8 and len({(r['phase'],r['group_id'],r['substeps']) for r in controls})==8,'Every control cohort must remain visible')
    lookup = {r['id']:r for r in records};verified=[]
    for result,source in zip(report['verification'],selection):
        config = cfg[result['id']];cohort = [r for r in records if r['group_id']==config['id']]
        require(source['id'] in lookup and source==lookup[source['id']] and result['source_route_quality']==source['route_quality'],'Selected source is not exact coarse evidence')
        by_step = {str(s):checked_statistics(result['by_substeps'][str(s)],[r for r in cohort if r['substeps']==s]) for s in STEPS}
        raw = result['qualification'];q = {k:boolean(raw[k]) for k in ('qualified','capture_qualified','clean_qualified','capture_both','clean_both','source_capture','physics_valid','oracle_agreement')}
        q.update(status=identifier(raw['status']),max_drift_m=number(raw.get('max_drift_m'),optional=True))
        present = len(cohort)==6 and all(sum(r['substeps']==s for r in cohort)==3 for s in STEPS)
        capture = present and all(captured(r) for r in cohort);clean = present and all(captured(r) and r.get('route_quality')==2 for r in cohort)
        physics = present and all(r.get('native_release') and not r.get('invalid') and r.get('invalid_physics') is False for r in cohort)
        oracle = present and all(r.get('scalar_score_audit_passed') is True for r in cohort)
        require((q['source_capture'],q['capture_both'],q['clean_both'],q['physics_valid'],q['oracle_agreement'])==(captured(source),capture,clean,physics,oracle),'Fresh outcome qualification differs')
        common = all(v['status']=='passed' for v in gates.values()) and captured(source) and physics and oracle and q['max_drift_m'] is not None and 0<=q['max_drift_m']<=.005
        capture_ok,clean_ok = bool(common and capture),bool(common and clean)
        require((q['capture_qualified'],q['clean_qualified'],q['qualified'])==(capture_ok,clean_ok,clean_ok),'Capture and clean gates were conflated')
        require(q['status']==('pending' if not present else 'qualified' if clean_ok else 'capture_only' if capture_ok else 'failed'),'Qualification status differs')
        s = result['sensitivity'];paired,agreement = s['paired_attempts'],s['capture_agreement']
        require(type(paired) is int and type(agreement) is int and 0<=agreement<=paired<=3,'Paired sensitivity counts differ')
        drift = number(s.get('nominal_position_delta_350ms_max_m'),optional=True)
        require(drift==q['max_drift_m'] and (drift is None or drift>=0),'Paired drift differs')
        verified.append(dict(configuration=config,source_route_quality=source['route_quality'],by_substeps=by_step,qualification=q,
            sensitivity=dict(paired_attempts=paired,capture_agreement=agreement,nominal_position_delta_350ms_max_m=drift),
            executions=[dict(id=identifier(r['id']),substeps=r['substeps'],repeat_index=r['repeat_index'],execution_fingerprint=r.get('execution_fingerprint'),first_flight=diagnostics[r['id']]) for r in cohort]))
    nclean = sum(v['qualification']['clean_qualified'] for v in verified);ncap = sum(v['qualification']['capture_qualified'] for v in verified)
    pre,post = gates['controls_pre']['status']=='passed',gates['controls_post']['status']=='passed'
    expected_gate = dict(status='blocked_controls' if not pre or not post else 'passed' if nclean else 'capture_only' if ncap else 'failed',controls_pre_passed=pre,controls_post_passed=post,qualified_settings=nclean,capture_qualified_settings=ncap,more_ppo_started=False)
    require(report['gate']==expected_gate,'Final capture/clean gate differs')
    top=[]
    for result in report['top_settings']:
        record_id = result.get('record_id',result.get('id'));require(record_id in lookup,'Top setting lacks actual record')
        r=lookup[record_id];require(r['phase']=='coarse' and result['group_id']==r['group_id'],'Top setting is not coarse evidence')
        top.append(dict(record_id=identifier(record_id),configuration=cfg[r['group_id']],substeps=r['substeps'],stats=checked_statistics(result['stats'],[r]),execution_fingerprint=r.get('execution_fingerprint'),first_flight=diagnostics[record_id]))
    coarse = [r for r in records if r['phase']=='coarse'];verification = [r for r in records if r['phase']=='verification']
    candidates = coarse+verification;control_records=[r for r in records if r['phase'].startswith('controls_')]
    summary = dict(version=VERSION,study_version=STUDY_VERSION,study_id=identifier(study.name),status='completed_independently_audited',scope=SCOPE,settings=declared,worker_elapsed_s=elapsed,
        counts=dict(**counts,phase_counts=phases,substep_counts=steps),candidate_counts=statistics(candidates),discovery_counts=statistics(coarse),verification_counts=statistics(verification),control_counts=statistics(control_records),
        gate=expected_gate,control_gates=gates,controls=controls,parent_sources=parent_sources,top_settings=top,verification=verified,
        provenance=dict(contract_fingerprint=spec['fingerprint'],physical_scene_fingerprint=spec['physical_scene_fingerprint'],scene_fingerprint=spec['scene_fingerprint'],parent_contract_fingerprint=PARENT_FP,
            parent_external_audit_sha256=PARENT_AUDIT_SHA,parent_input_sha256=hash_map(spec['input_bindings']['parent']),
            contract_sha256=sha(study/'contract.json'),report_sha256=sha(study/'report.json'),independent_audit_sha256=sha(audit_path),independent_audit_version=AUDIT_VERSION,
            independent_auditor_sha256=fingerprint(audit['auditor_sha256']),independent_audit_checks=audit['checks_count'],evidence_manifest_sha256=sha(study/'evidence-manifest.json'),evidence_sha256=manifest,
            executed_source_sha256=hash_map(spec['source_sha256']),asset_sha256=hash_map(spec['asset_sha256'])),
        privacy='Configured nominal geometry, diagnostic scalars, outcomes and hashes only; no measured crossing XY, raw contacts, sampled poses, joints, point trajectories, credentials or host paths.')
    dependencies = audit.get('auditor_identity',{}).get('dependencies_sha256')
    if dependencies:summary['provenance']['auditor_dependencies_sha256']=hash_map(dependencies)
    output.mkdir(parents=True);write_json(output/'summary.json',summary)
    with (output/'requested-shots.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]),lineterminator='\n');writer.writeheader();writer.writerows(rows)
    write_json(output/'checksums.json',{p.name:sha(p) for p in sorted(output.iterdir()) if p.is_file()})
    return summary


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--study',type=Path,required=True);p.add_argument('--audit',type=Path);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();result=curate(a.study,a.audit or a.study/'audit-impact650.json',a.output)
    print(json.dumps(dict(status=result['status'],candidate_counts=result['candidate_counts'],gate=result['gate'])))


if __name__=='__main__':main()
