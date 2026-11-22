"""Read-only CPU verification of the three-miss, single-correction experiment.

Reconstructs the matched schedule and counts independently, authenticates pose
proofs, and rescores every native flight from saved contact/position arrays.
No native rollout, model training, GPU allocation or source-evidence edits.
"""
from __future__ import annotations
import argparse
import copy
import json
import math
from pathlib import Path

import numpy as np

from .benchmark_verify import Checks, contained, digest, fingerprint, read, sha
from .sweep_worker import atomic

VERSION = 'sircus-hybrid-miss-independent-audit-v1'
STUDY_VERSION = 'sircus-hybrid-miss-correction-v1'
CASES = tuple('stalled-reference-test-' + x for x in ('03', '09', '19'))
SOURCE_REPORT = 'e8cd7009362af9c80f7fe6a54b70937b6cbef8f4182eaf04d22e6b7e8dbac2e1'


def expected_slots(plan):
    """Rebuild all requests, including unadmitted treatment denominators."""
    slots=[]
    for case in plan['cases']:
        for repeat in range(3):
            order=('original','corrected') if repeat != 1 else ('corrected','original')
            for variant in order:
                source=case[variant]
                request=None if source is None else dict(copy.deepcopy(source),
                    id=f"{case['id']}-{variant}-{repeat}",context_id=case['id'],variant=variant,
                    repeat=repeat,method=variant,phase='diagnostic',source_request_id=source['id'])
                slots.append(dict(case_id=case['id'],variant=variant,repeat=repeat,
                                  directory=case['directory'],request=request))
    return slots


def arm_score(prediction,route):
    value=(prediction.get(route) or {}).get('arm_clearance_m')
    return float(value) if type(value) in (int,float) and math.isfinite(value) else -math.inf


def unchanged_geometry(original,corrected,old_proof,new_proof,scene,checks,label):
    for key in ('values','bucket_xy_m','board_heading_offset_deg','board_twist_deg','selected_q10'):
        checks.equal(corrected[key],original[key],label+'/fixed '+key)
    twist=math.remainder(corrected['board_twist_deg']-original['board_twist_deg'],360.)
    checks.require(abs(twist)<1e-10,label+'/joint-only test preserves full board orientation')
    old=np.asarray(old_proof['targets']['board_world'],float)
    new=np.asarray(new_proof['targets']['board_world'],float)
    checks.numeric(old[:3,3],new[:3,3],label+'/same board center',1e-10)
    checks.numeric(old[:3,2],new[:3,2],label+'/same board normal',1e-10)
    checks.numeric(old_proof['targets']['dropper_flange_world'],new_proof['targets']['dropper_flange_world'],
                   label+'/same exact release flange',1e-10)
    if abs(twist)<1e-10:
        checks.equal(old_proof['targets'],new_proof['targets'],label+'/same full Cartesian targets')
    else:
        # A rectangle rotated180° in its plane occupies exactly the same volume.
        # Checking R_new=R_old*diag(-1,-1,1) proves this for any board dimensions.
        checks.numeric(new[:3,:3],old[:3,:3]@np.diag([-1.,-1.,1.]),label+'/same rectangle volume',1e-10)
    delta=max(abs(math.remainder(a-b,math.tau)) for a,b in zip(
        corrected['selected_q14'],original['selected_q14'],strict=True))
    checks.require(delta>=.05-1e-12,label+'/materially distinct bouncer joints')


def classify(row):
    if row['invalid_physics']:return 'invalid_physics'
    if row['route_success']:
        return row['shot_route'] or 'unclassified_capture'
    if row['experimental_route']['robot_assisted_capture']:
        return 'arm_assisted_capture'
    return 'flight_miss'


def independent_summary(plan,rows):
    result=[]
    for case in plan['cases']:
        groups={}
        for variant in ('original','corrected'):
            found=[r for r in rows if r.get('context_id')==case['id'] and r.get('variant')==variant]
            outcomes=[classify(r) for r in found]
            groups[variant]=dict(requested=3,completed=len(found),released=len(found),
                setup_rejected=3 if case[variant] is None else 0,
                captures=sum(x in ('direct','table','unclassified_capture') for x in outcomes),
                direct=outcomes.count('direct'),table=outcomes.count('table'),
                robot_contacts=sum(r['experimental_route']['robot_contact_steps']>0 for r in found),
                outcomes=outcomes,rows=found)
        result.append(dict(id=case['id'],bucket_xy_m=case['original']['bucket_xy_m'],results=groups))
    return result


def audit(a):
    from . import replay650_verify as replay
    from .cell1050_rebound_study import parent_inputs,control_requests
    from .direct_path_first_audit import _proof,verify_heading
    from .intergrid_route_refit_inputs import _audited_source
    from .intergrid_route_labels import annotate_routes
    from .parallel_capture_grid_audit import observed_rows,rescore_packet,verify_release
    from .policy64_verify import audit_native_options
    from .table_training_verify import control_gate,rescore_rows
    from .intergrid_flight_ranker import FlightRanker

    study,checks=a.output,Checks()
    spec,report,plan=(read(study/name) for name in ('contract.json','report.json','preparation.json'))
    fingerprint(spec,checks,'Frozen correction contract')
    checks.equal((spec['version'],report['version'],report['status'],report['phase']),
                 (STUDY_VERSION,STUDY_VERSION,'completed','completed'),'Completed declared correction')
    checks.equal((spec['physics_substeps'],spec['repeats'],spec['controls'],spec['planned_requests']),
                 (64,3,18,36),'Fixed64, three matched repeats,18 controls,36 slots')
    checks.require(0<report['gpu_lease_s']<=spec['gpu_cap_s']<=3600,'Bounded acquired GPU time')
    manifest=replay.check_manifest(study,checks)
    checks.equal(report['evidence_manifest_sha256'],sha(study/'evidence-manifest.json'),'Report binds immutable evidence')
    checks.equal(spec['plan_sha256'],manifest['preparation.json'],'Pre-release plan hash')
    for name,value in spec['frozen_files'].items():
        checks.equal(manifest[name],value,'Every frozen preparation file unchanged/'+name)
    source=a.source_root or Path(__file__).resolve().parents[2]
    for name,value in spec['source_sha256'].items():
        checks.equal(sha(contained(source,name)),value,'Exact frozen producer source/'+name)
    parent_spec,parent_binding,_=parent_inputs(a.parent_study,a.parent_audit)
    checks.equal((spec['identity'],spec['scene'],spec['parent']),
                 (parent_spec['identity'],parent_spec['scene'],parent_binding),'Original calibrated native scene')
    _,source_manifest,_,bindings,source_spec=_audited_source(a.source_study,a.source_audit,
        'sircus-stalled-reference-study-v1',
        ('choices.json','rows.jsonl','training-prior.json','calibration.json','setup-home.json'))
    checks.equal(bindings['report.json'],SOURCE_REPORT,'Exact source three-miss study')
    checks.equal(plan['source'],bindings,'Pre-release source binding')
    checks.equal(plan['identity'],source_spec['identity'],'Same source native identity')
    source_choices={s['context_id']:s['request'] for s in read(a.source_study/'choices.json')['slots']
                    if s['method']=='hybrid' and s['request'] is not None}
    source_rows={r['id']:r for r in observed_rows(a.source_study/'rows.jsonl')}
    checks.equal([c['id'] for c in plan['cases']],list(CASES),'Exactly three selected development misses')
    checks.equal(plan['repeats'],3,'Three predeclared repeats')
    checks.equal(plan['seeds'],[2026112203,2026112209,2026112219],'Deterministic finite CPU search seeds')
    prior,home=read(study/'training-prior.json'),read(study/'setup-home.json')
    calibration=replay.check_calibration(study,spec['identity']['calibration'],checks)
    ranker=FlightRanker(spec['scene'],calibration,a.assets,source_spec['flight_ranking']['rule'])
    resolutions={}
    for case in plan['cases']:
        label=case['id'];original=source_choices[label];directory=study/case['directory']
        checks.equal(case['original'],original,label+'/unchanged original request')
        checks.equal(case['historical'],source_rows[original['id']],label+'/exact archived observation')
        checks.require(case['historical']['native_release'] and not case['historical']['invalid_physics']
                       and not case['historical']['route_success'],label+'/actual valid source miss')
        checks.equal(case['directory'],'native-'+label[-2:],label+'/separate case proof namespace')
        resolutions[case['directory']]={}
        old_proof=_proof(original,dict(id=label,bucket_xy_m=original['bucket_xy_m']),directory,prior,home,
                         resolutions[case['directory']],checks)
        prediction=ranker.predict(original)
        checks.equal(prediction,case['original_prediction'],label+'/recomputed original prediction')
        route=prediction['predicted_route'] or original['intended_route']
        checks.equal(case['route_for_ranking'],route,label+'/fixed original route for clearance ranking')
        predictions={}
        checks.require(len(case['alternatives'])<=12,label+'/bounded CPU alternatives')
        for candidate in case['alternatives']:
            proof=_proof(candidate,dict(id=label,bucket_xy_m=original['bucket_xy_m']),directory,prior,home,
                         resolutions[case['directory']],checks)
            unchanged_geometry(original,candidate,old_proof,proof,spec['scene'],checks,candidate['id'])
            predictions[candidate['id']]=ranker.predict(candidate)
        checks.equal(predictions,case['predictions'],label+'/all candidate predictions rebuilt without flight outcomes')
        ordered=sorted(case['alternatives'],key=lambda r:(-arm_score(predictions[r['id']],route),r['id']))
        checks.equal(case['corrected'],ordered[0] if ordered else None,label+'/pre-release maximum clearance choice')
    slots=expected_slots(plan)
    checks.equal(read(study/'slots.json'),slots,'Independent reconstruction of18 paired slots')
    checks.equal(read(study/'native-started.json')['slots_sha256'],sha(study/'slots.json'),'Native start binds frozen schedule')
    expected={c['directory']:[s['request'] for s in slots if s['case_id']==c['id'] and s['request'] is not None]
              for c in plan['cases']}
    ledger,timings=observed_rows(study/'rows.jsonl'),observed_rows(study/'timing.jsonl')
    sources=read(a.parent_study/'control-replays.json')
    used={d:set() for d in expected};seen=set();phases=[];observed=[];controls={}
    for index,path in enumerate(sorted((study/'requests').glob('*.json'))):
        relative=f'requests/batch-{index:04d}.json'
        checks.equal(path.relative_to(study).as_posix(),relative,'Contiguous packet names')
        packet=read(path);phase=packet['phase'];requests=packet['rows'];directory=packet['adapter_directory']
        control=phase in ('controls_pre','controls_post')
        checks.equal(packet['control'],control,relative+'/declared control status')
        checks.require(control or phase=='diagnostic',relative+'/known phase')
        phases.append(0 if phase=='controls_pre' else 2 if phase=='controls_post' else 1)
        checks.require(0<len(requests)<=32,relative+'/bounded batch')
        timing=next(timings,None)
        checks.require(timing is not None,relative+'/saved timing')
        checks.equal((timing['packet'],timing['phase']),(relative,phase),relative+'/matching timing')
        if control:
            checks.equal(directory,'native-controls',relative+'/control namespace')
            checks.require(phase not in controls,relative+'/one control block per side')
            checks.equal(requests,control_requests(sources,phase),relative+'/exact matched control commands')
        else:
            checks.require(directory in expected,relative+'/known diagnostic directory')
            checks.equal(requests,expected[directory],relative+'/exact paired frozen requests')
        rows=[]
        for request in requests:
            row=next(ledger,None);checks.require(row is not None,request['id']+'/observation exists')
            verify_release(row,request,relative,manifest[relative],phase,checks)
            checks.require(row['id'] not in seen,row['id']+'/one native release only');seen.add(row['id'])
            checks.equal(row['adapter_directory'],directory,row['id']+'/bound native namespace')
            if not control:
                checks.equal((row['variant'],row['repeat']), (request['variant'],request['repeat']),row['id']+'/matched repeat')
                verify_heading(row,request,study/directory,spec['scene'],home,prior,resolutions[directory],checks)
            rows.append(row)
        if control: controls[phase]=rows
        else:
            alias={'native-heading/'+name[len(directory)+1:]:v for name,v in manifest.items() if name.startswith(directory+'/')}
            list(rescore_packet(study/directory,rows,spec['scene'],calibration,alias,used[directory],checks))
            relabeled=annotate_routes(copy.deepcopy(rows),study/directory)
            for row,relabel in zip(rows,relabeled,strict=True):
                checks.equal((row['shot_route'],row['table_rebounds_before_entry']),
                             (relabel['shot_route'],relabel['table_rebounds_before_entry']),row['id']+'/observed route label')
            observed.extend(rows)
    checks.equal(phases,[0,1,1,1,2],'Controls bracket exactly one batch per diagnosed case')
    checks.require(next(ledger,None) is None and next(timings,None) is None,'No unaccounted observations or timing rows')
    checks.equal([r['id'] for r in observed],[s['request']['id'] for s in slots if s['request'] is not None],
                 'Every and only frozen admitted request executed')
    for directory in expected:
        audit_native_options(study/directory,spec['scene'],calibration,a.assets,checks)
        checks.equal(used[directory],{p.relative_to(study/directory).as_posix() for p in (study/directory).rglob('*.npz')},
                     directory+'/complete native archive inventory')
    control_rows=[r for rows in controls.values() for r in rows]
    audit_native_options(study/'native-controls',spec['scene'],calibration,a.assets,checks)
    rescore_rows(study/'native-controls',control_rows,spec['scene'],calibration,spec['identity']['trace_schema'],a.assets,checks)
    checks.equal({r['native_cache_file'] for r in control_rows},
                 {p.relative_to(study/'native-controls').as_posix() for p in (study/'native-controls').rglob('*.npz')},
                 'Complete control archive inventory')
    checks.equal(set(controls),{'controls_pre','controls_post'},'Both original control blocks present')
    for phase,rows in controls.items():
        gate=control_gate(rows);checks.equal(gate['status'],'passed',phase+'/native controls passed')
        checks.equal(report['controls'][phase],gate,phase+'/independent control counts')
    summary=independent_summary(plan,observed)
    checks.equal(report['cases'],summary,'Independent per-case repeated outcomes and contact counts')
    checks.equal(report['progress'],dict(completed=len(observed),planned=18),'All diagnostic requests retained in denominator')
    return dict(version=VERSION,status='passed',checks_count=checks.count,report_sha256=sha(study/'report.json'),
        plan_sha256=sha(study/'preparation.json'),evidence_manifest_sha256=sha(study/'evidence-manifest.json'),
        verifier_source_sha256=sha(Path(__file__)),frozen_source_root=str(source),
        diagnostic_requests=18,diagnostic_releases=len(observed),control_releases=len(control_rows),
        results=[dict(id=c['id'],bucket_xy_m=c['bucket_xy_m'],results={v:{k:x for k,x in g.items() if k!='rows'}
                 for v,g in c['results'].items()}) for c in summary])


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('source-study','source-audit','parent-study','parent-audit','assets','output','audit'):
        p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--source-root',type=Path)
    a=p.parse_args()
    if a.audit.exists() or a.audit.resolve().is_relative_to(a.output.resolve()):
        p.error('Write a new audit receipt outside frozen evidence')
    try: result=audit(a)
    except Exception as e: result=dict(version=VERSION,status='failed',error=f'{type(e).__name__}: {e}')
    atomic(a.audit,result);print(json.dumps(result),flush=True)
    if result['status']!='passed':raise SystemExit(1)


if __name__=='__main__':main()
