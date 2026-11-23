"""Independent read-only admission and native scoring audit for two link-clearance targets."""
from __future__ import annotations
import argparse
import copy
import json
import math
import os
from pathlib import Path
import numpy as np
from .benchmark_verify import Checks, contained, digest, fingerprint, read, sha
from .board_pose_recovery_audit import (exact_repeat, independent_summary, recompute_screen,
    verify_confirmations, verify_request_keys)
from .nearby_transfer_audit import load_source, check_source_trace, qualified
from .sweep_worker import atomic
VERSION='sircus-link-clearance-independent-audit-v1'
PREPARATION_VERSION='sircus-link-clearance-preparation-audit-v1'
STUDY_VERSION='sircus-link-clearance-study-v1'
CASES=('parallel-board-miss-09','parallel-board-miss-19')


def verify_sources(a,plan,bank,spec,checks):
    source=load_source(a.source_study,a.source_audit,'sircus-nearby-transfer-study-v1',
        ('preparation.json','rows.jsonl','source-bank.json'),checks)
    checks.equal((spec['source_bindings'],plan['source']),(source['binding'],)*2,'Exact audited parent source')
    checks.equal((source['spec']['identity'],source['spec']['scene']),(spec['identity'],spec['scene']),'Unchanged calibrated scene')
    for name in ('calibration.json','training-prior.json','setup-home.json'):
        checks.equal((source['manifest'][name],sha(source['study']/name)),(sha(Path(a.output)/name),)*2,'Frozen '+name)
    expected_cases=[];expected_bank=[];preserved=[]
    for old in read(source['study']/'preparation.json')['cases']:
        if old['id'] in CASES:
            sources=[]
            for command in old['candidates']:
                rows=[r for r in source['rows'] if r.get('variant')=='search' and r.get('candidate_id')==command['id']]
                checks.equal(len(rows),1,'One prior initial trial per source')
                row=rows[0]
                checks.require(row['native_release'] and not row['invalid_physics'] and not qualified(row)
                    and row['experimental_route']['robot_contact_steps']>0,'Prior candidate is valid robot-contact miss')
                check_source_trace(source,command,row,checks)
                item=dict(id=command['id'],source_request=command,case_id=old['id'],source_directory=row['adapter_directory'],
                    source_row_id=row['id'],source_row_sha256=digest(row),source_trace_file=row['native_cache_file'],
                    source_trace_sha256=row['native_cache_sha256'],strength='prior_robot_contact',source_binding=source['binding'],
                    prediction=old['predictions'][command['id']])
                sources.append(item);expected_bank.append(item)
            expected_cases.append(dict(id=old['id'],original=old['original'],source_directory=old['directory'],sources=sources,historical=old['historical']))
        else:
            choices=[]
            for command in old['candidates']:
                rows=[r for r in source['rows'] if r.get('variant')=='repeat' and r.get('candidate_id')==command['id']]
                if len(rows)==3 and all(qualified(r) for r in rows):
                    checks.equal(sorted(r['repeat'] for r in rows),[1,2,3],'Prior success has three separate confirmations')
                    for row in rows:check_source_trace(source,command,row,checks)
                    choices.append((command,rows))
            checks.require(bool(choices),'Preserved position has a confirmed success')
            command,rows=choices[0]
            preserved.append(dict(id=old['id'],original=command,source_directory=old['directory'],directory='preserved-'+old['id'][-2:],
                source_row_ids=[r['id'] for r in rows],source_row_sha256=[digest(r) for r in rows]))
    checks.equal([c['id'] for c in plan['cases']],list(CASES),'Exactly two unresolved positions')
    checks.equal([c['id'] for c in preserved],['parallel-board-miss-'+i for i in ('03','13','15')],'All recovered positions preserved')
    checks.equal(plan['preserved_cases'],preserved,'Independent first confirmed settings for preservation')
    checks.equal(bank,expected_bank,'All40 failed source commands with native proof provenance')
    checks.equal(len(bank),40,'Twenty source attempts per unresolved target')
    for got,expected in zip(plan['cases'],expected_cases,strict=True):
        checks.equal({k:got[k] for k in expected},expected,got['id']+'/source case unchanged')
    return source


def negative_score(prediction):
    value=prediction.get('score_m')
    return -float(value) if type(value) in (int,float) and math.isfinite(value) else math.inf


def independent_seeds(case):
    ordered=sorted(case['sources'],key=lambda s:(negative_score(s['prediction']),s['id']))
    selected=[];used=set()
    for row in ordered:
        key=tuple(round(value/width) for value,width in zip(row['source_request']['values'],(.05,.05,.04,5.,.06),strict=True))
        if key not in used:selected.append(row);used.add(key)
        if len(selected)==6:return selected
    identities={s['id'] for s in selected}
    return (selected+[s for s in ordered if s['id'] not in identities])[:6]


def verify_proposals(case,checks):
    expected=[]
    options=[(0.,0.,0.,'fixed_cartesian',k) for k in range(2)]
    options += [(v,0.,0.,'radial_shift',0) for v in (-.12,-.08,-.04,.04,.08,.12)]
    options += [(0.,v,0.,'lateral_shift',0) for v in (-.15,-.10,-.05,.05,.10,.15)]
    options += [(0.,0.,v,'board_twist',0) for v in (-20.,20.,-45.,45.,180.)]
    for index,source in enumerate(independent_seeds(case)):
        command=source['source_request'];old=command['values'];bucket=command['bucket_xy_m']
        delta=[bucket[i]-old[i] for i in (0,1)];length=math.hypot(*delta)
        forward=[v/length for v in delta];side=[-forward[1],forward[0]]
        for radial,lateral,twist,family,random_index in options:
            values=list(old);values[:2]=[old[i]+radial*forward[i]+lateral*side[i] for i in (0,1)]
            if any(v<lo or v>hi for v,lo,hi in zip(values,case['bounds']['low'],case['bounds']['high'],strict=True)) or math.dist(values[:2],bucket)<=.15:continue
            expected.append(dict(id=f"{case['id']}-clearance-{len(expected):04d}",context_id=case['id'],values=values,bucket_xy_m=list(bucket),
                board_heading_offset_deg=command.get('board_heading_offset_deg',0.),board_twist_deg=math.remainder(command.get('board_twist_deg',0.)+twist,360.),
                seed_q10=list(command['selected_q10']),seed_q14=list(command['selected_q14']),phase='search',role='diagnostic',method='link_clearance',
                intended_route=command.get('intended_route','direct'),group=family,proposal_family=family,source_request_id=command['id'],source_bank_id=source['id'],
                source_strength=source['strength'],source_distance_m=0.,seed_index=index,radial_offset_m=radial,lateral_offset_m=lateral,twist_delta_deg=twist,
                branch_seed=2026112307+int(case['id'][-2:])*100+index*10+random_index,fixed_cartesian=family=='fixed_cartesian'))
    checks.equal(case['proposals'],expected,'Independent bounded translation/twist/branch menu')


def independent_order(case):
    valid=[p for p in case['proposals'] if case['wrist_reach'][p['id']]['passed']]
    groups=[sorted([p for p in valid if p['seed_index']==seed and p['proposal_family']==family],
                   key=lambda p:(negative_score(case['nominal_predictions'][p['id']]),p['id']))
            for seed in sorted({p['seed_index'] for p in valid}) for family in ('fixed_cartesian','radial_shift','lateral_shift','board_twist')]
    return [p['id'] for p in [g[i] for i in range(max(map(len,groups),default=0)) for g in groups if i<len(g)][:96]]


def verify_preparation_records(case,order,checks):
    records=case['records']
    checks.equal([r['proposal_id'] for r in records],order[:len(records)],'Unskipped preparation prefix')
    terminal=[i for i,r in enumerate(records) if r['reason']=='case_preparation_time_limit']
    checks.equal(bool(terminal),case['preparation_timed_out'],'Explicit case timeout')
    checks.require(not terminal or terminal==[len(records)-1],'One case timeout terminates preparation')
    for record in records:
        if record['reason'] in ('case_preparation_time_limit','proposal_preparation_time_limit'):
            checks.equal((record['admitted'],record['admitted_request_ids']),(0,[]),'Interrupted checks cannot admit a partial proof')
    checks.require(case['preparation_timed_out'] or len(records)==len(order),'No unexplained dropped preparation')
    checks.require(len(records)<=96,'At most96full checks per case')


def branch_change(request,source):
    delta={role:[abs(math.remainder(x-y,2*math.pi)) for x,y in zip(request['selected_'+role],source['selected_'+role],strict=True)] for role in ('q10','q14')}
    return dict(proximal_rad=max(delta['q10'][:3]+delta['q14'][:3]),maximum_rad=max(delta['q10']+delta['q14']),per_arm_rad=delta)


def independent_candidates(case):
    sources={s['id']:s['source_request'] for s in case['sources']};valid=[]
    for row in case['admitted']:
        change=branch_change(row,sources[row['source_request_id']])
        if row['fixed_cartesian'] and change['maximum_rad']<.05:continue
        valid.append(dict(row,joint_change=change))
    rank=lambda r:(negative_score(case['predictions'][r['id']]),-r['joint_change']['proximal_rad'],r['id'])
    fixed=sorted((r for r in valid if r['fixed_cartesian']),key=lambda r:(-max(r['joint_change']['per_arm_rad']['q14'][:3]),-max(r['joint_change']['per_arm_rad']['q10'][:3]),*rank(r)))
    reserve=fixed[:8]
    for source in sorted({r['source_bank_id'] for r in valid}):
        for family in ('radial_shift','lateral_shift','board_twist'):
            group=[r for r in valid if r['source_bank_id']==source and r['proposal_family']==family]
            if group:reserve.append(min(group,key=rank))
    result=[];seen=set()
    for row in reserve+sorted(valid,key=rank):
        key=digest([row['bucket_xy_m'],row['values'],row['board_heading_offset_deg'],row['board_twist_deg'],row['selected_q10'],row['selected_q14']])
        if key in seen or any(equivalent_commands(row,old) for old in result):continue
        seen.add(key);result.append(row)
        if len(result)==32:break
    return result


def equivalent_commands(a,b):
    if a['bucket_xy_m']!=b['bucket_xy_m']:return False
    if any(abs(x-y)>1e-8 for x,y in zip(a['values'],b['values'],strict=True)):return False
    if any(abs(math.remainder(a[k]-b[k],360.))>1e-8 for k in ('board_heading_offset_deg','board_twist_deg')):return False
    return all(abs(math.remainder(x-y,2*math.pi))<.01 for role in ('q10','q14')
               for x,y in zip(a['selected_'+role],b['selected_'+role],strict=True))


def verify_admissions(case,directory,prior,home,resolutions,checks):
    from .direct_path_first_audit import _proof,verify_seed
    proposals={p['id']:p for p in case['proposals']};admitted={r['id']:r for r in case['admitted']};seen=[]
    checks.equal(len(admitted),len(case['admitted']),'Unique admitted records')
    for record in case['records']:
        proposal=proposals[record['proposal_id']];ids=record['admitted_request_ids']
        checks.equal(record['proposal_sha256'],digest(proposal),'Exact checked proposal')
        checks.equal(record['admitted'],len(ids),'All admitted branches accounted')
        checks.require(len(ids)<=4,'At most four checked branches per geometry')
        for index,ident in enumerate(ids):
            row=admitted[ident]
            checks.equal(ident,proposal['id']+f'-b{index}','Consecutive branch identity')
            checks.equal({k:row[k] for k in proposal if k!='id'},{k:v for k,v in proposal.items() if k!='id'},'No command substitution')
            checks.equal((row['proposal_id'],row['native_prepared'],row['command_values']),(proposal['id'],True,proposal['values']),'Prepared exact command')
            proof=_proof(row,dict(id=case['id'],bucket_xy_m=case['original']['bucket_xy_m']),directory,prior,home,resolutions,checks)
            verify_seed(proof,proposal,checks);seen.append(ident)
    checks.equal(seen,list(admitted),'Every admitted proof accounted')


def verify_preservation_slots(plan,slots,checks):
    expected={(c['id'],i):c for c in plan['preserved_cases'] for i in range(3)}
    seen=set();ids=set()
    for slot in slots:
        request=slot['request'];key=(request['context_id'],request['repeat'])
        checks.require(key not in seen and request['id'] not in ids,'Unique preservation release');seen.add(key);ids.add(request['id'])
        checks.require(key in expected,'Declared preservation target and repeat')
        case=expected[key]
        checks.equal((request['variant'],request['phase'],slot['directory'],request['candidate_id']),('preserved','preservation',case['directory'],case['original']['id']),'Preservation identity and phase')
        exact_repeat(case['original'],request,checks,request['id'])
    checks.equal(seen,set(expected),'Three unchanged releases at each successful target')


def release_accounting(initial,repeats,preserved):
    """Preservation is a real release, but outside the two-target progress bar."""
    if any(type(x) is not int or x<0 for x in (initial,repeats,preserved)) or preserved!=9:
        raise ValueError('Require nonnegative release counts and all nine preservation drops')
    return dict(progress=dict(completed=initial+repeats,planned=initial+repeats),
                native_releases=initial+repeats+preserved+18)


def verify_preparation(a,checks):
    from . import replay650_verify as replay
    from .cell1050_rebound_study import parent_inputs
    from .direct_path_first_audit import _proof
    from .fixed35_broad_native import reach_models
    from .intergrid_flight_ranker import FlightRanker
    from .intergrid_route_labels import LABEL_VERSION
    from .link_clearance_plan import SEARCH_RULE
    from .link_clearance_diagnosis import diagnose
    study=Path(a.output);spec,plan,bank=(read(study/name) for name in ('contract.json','preparation.json','source-bank.json'))
    fingerprint(spec,checks,'Frozen two-target contract')
    checks.equal((spec['version'],plan['version'],spec['physics_substeps'],spec['repeats'],spec['controls'],spec['preservation_requests'],spec['gpu_cap_s']),
        (STUDY_VERSION,STUDY_VERSION,64,3,18,9,3600),'Bounded two-target64 protocol')
    checks.equal((plan['seed'],plan['repeats']),(2026112307,3),'Frozen seed and repeat count')
    checks.equal((spec['label_version'],plan['label_version']),(LABEL_VERSION,)*2,'Unchanged route scoring')
    checks.equal((spec['search_rule'],plan['search_rule']),(SEARCH_RULE,)*2,'Fixed finite search rule')
    checks.equal((SEARCH_RULE['version'],SEARCH_RULE['proposal_limit_s'],SEARCH_RULE['cpu_case_limit_s']),
                 ('sircus-link-clearance-menu-v2',30,900),'Explicit per-proposal and total CPU budget')
    checks.equal(spec['plan_sha256'],sha(study/'preparation.json'),'Frozen preparation bytes')
    checks.equal(plan['source_bank_sha256'],sha(study/'source-bank.json'),'Frozen source bank')
    paths={key:str(Path(getattr(a,key)).resolve()) for key in ('source_study','source_audit','parent_study','parent_audit','diagnosis')}
    checks.equal((spec['source_paths'],plan['source_paths']),(paths,paths),'Exact source paths')
    root=Path(getattr(a,'source_root',None) or Path(__file__).resolve().parents[2]).resolve()
    required={'sircus/mjlab/link_clearance_plan.py','sircus/mjlab/link_clearance_study.py','sircus/mjlab/link_clearance_audit.py','sircus/mjlab/link_clearance_diagnosis.py'}
    checks.require(required<=set(spec['source_sha256']),'Producer and diagnosis and independent verifier frozen')
    for name,value in spec['source_sha256'].items():checks.equal(sha(contained(root,name)),value,'Frozen source '+name)
    for name,value in spec['frozen_files'].items():checks.equal(sha(contained(study,name)),value,'Frozen input '+name)
    checks.require({'preparation.json','slots.json','preserved-slots.json','source-bank.json','diagnosis.json','calibration.json','training-prior.json','setup-home.json'}<=set(spec['frozen_files']),'All pre-native inputs bound')
    parent,binding,_=parent_inputs(Path(a.parent_study),Path(a.parent_audit))
    checks.equal((spec['identity'],spec['scene'],spec['parent'],plan['parent'],plan['identity'],plan['scene']),
        (parent['identity'],parent['scene'],binding,binding,parent['identity'],parent['scene']),'Original calibration parent preserved')
    source=verify_sources(a,plan,bank,spec,checks)
    checks.equal((spec['diagnosis_sha256'],plan['diagnosis_sha256'],sha(a.diagnosis)),(sha(study/'diagnosis.json'),)*3,'Exact source diagnosis')
    checks.equal(read(study/'diagnosis.json'),diagnose(a.source_study,a.source_audit,a.assets),'Independently rerun force/link reconstruction')
    prior,home=read(study/'training-prior.json'),read(study/'setup-home.json')
    calibration=replay.check_calibration(study,spec['identity']['calibration'],checks)
    target_contract=read(Path(source['spec']['source_paths']['target_study'])/'contract.json')
    checks.equal(sha(Path(source['spec']['source_paths']['target_study'])/'contract.json'),source['spec']['source_bindings']['target']['contract.json'],'Frozen inherited flight-ranker rule')
    ranker=FlightRanker(spec['scene'],calibration,a.assets,target_contract['flight_ranking']['rule'])
    models=reach_models(str(Path(a.assets).resolve()));resolutions={}
    for case in plan['cases']+plan['preserved_cases']:
        directory=study/case['directory'];resolutions[case['directory']]={}
        _proof(case['original'],dict(id=case['id'],bucket_xy_m=case['original']['bucket_xy_m']),directory,prior,home,resolutions[case['directory']],checks)
    for case in plan['cases']:
        label=case['id'];directory=study/case['directory']
        checks.equal(case['directory'],'native-'+label[-2:],'Isolated target directory')
        checks.equal(read(study/f'preparation-{label[-2:]}.json'),case,'Complete final target receipt')
        checks.equal(case['status'],'prepared','Preparation completed')
        checks.equal(case['original_prediction'],ranker.predict(case['original']),'Recomputed original prediction')
        verify_proposals(case,checks)
        proposals={p['id']:p for p in case['proposals']}
        checks.equal(set(proposals),set(case['nominal_predictions']),'Complete nominal predictions')
        checks.equal(set(proposals),set(case['wrist_reach']),'Complete reach screen')
        for proposal in proposals.values():
            reach,nominal=recompute_screen(ranker,spec['scene'],proposal,models)
            checks.equal((case['wrist_reach'][proposal['id']],case['nominal_predictions'][proposal['id']]),(reach,nominal),'Independent nominal/reach screening')
        order=independent_order(case)
        checks.equal(case['preparation_order'],order,'Independent fixed-first source-balanced order')
        verify_preparation_records(case,order,checks)
        verify_admissions(case,directory,prior,home,resolutions[case['directory']],checks)
        checks.equal(case['predictions'],{r['id']:ranker.predict(r) for r in case['admitted']},'Recomputed admitted flight/arm predictions')
        checks.equal(case['candidates'],independent_candidates(case),'Independent branch-aware native selection')
        checks.equal(case['selection_order'],[r['id'] for r in case['candidates']],'Frozen native ordering')
        counts={}
        for row in case['candidates']:counts[row['source_bank_id']]=counts.get(row['source_bank_id'],0)+1
        checks.equal(case['source_counts'],counts,'Native selection source counts')
    count=sum(len(c['candidates']) for c in plan['cases'])
    checks.require(count<=64,'At most32selected poses per target')
    checks.equal((spec['initial_requests'],spec['maximum_confirmation_requests'],spec['maximum_total_requests']),
                 (6+count,3*count,33+4*count),'Originals plus candidates plus confirmations plus preservation plus controls')
    return spec,plan,root,prior,home,resolutions


def preflight(a):
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('Hide GPUs for audit')
    checks=Checks();verify_preparation(a,checks);study=Path(a.output);plan=read(study/'preparation.json')
    checks.require(not (study/'native-started.json').exists(),'No native release before independent preflight')
    verify_request_keys(plan,read(study/'slots.json'),checks)
    verify_preservation_slots(plan,read(study/'preserved-slots.json'),checks)
    return dict(version=PREPARATION_VERSION,status='passed',checks_count=checks.count,native_execution_performed=False,
        contract_sha256=sha(study/'contract.json'),preparation_sha256=sha(study/'preparation.json'),plan_sha256=sha(study/'preparation.json'),
        slots_sha256=sha(study/'slots.json'),preserved_slots_sha256=sha(study/'preserved-slots.json'),
        source_bank_sha256=sha(study/'source-bank.json'),verifier_source_sha256=sha(__file__),diagnosis_sha256=sha(study/'diagnosis.json'))

def verify_launch_receipt(study, checks):
    marker=read(study/'native-started.json')
    receipt_path=Path(marker['preflight_audit'])
    checks.equal(sha(receipt_path),marker['preflight_audit_sha256'],'Launch bound the unchanged independent preflight')
    receipt=read(receipt_path)
    checks.equal((receipt['version'],receipt['status'],receipt['native_execution_performed']),
                 (PREPARATION_VERSION,'passed',False),'CPU-only preflight passed before native launch')
    checks.require(type(receipt['checks_count']) is int and receipt['checks_count']>0,'Preflight performed actual checks')
    for field,name in (('contract_sha256','contract.json'),('preparation_sha256','preparation.json'),
                       ('slots_sha256','slots.json'),('source_bank_sha256','source-bank.json'),('preserved_slots_sha256','preserved-slots.json')):
        checks.equal(receipt[field],sha(study/name),'Preflight binds final frozen '+name)
    checks.equal(receipt['verifier_source_sha256'],sha(Path(__file__)),'Same independent verifier before and after native')
    checks.equal(marker['slots_sha256'],sha(study/'slots.json'),'Launch binds exact initial command schedule')


def audit(a):
    from . import replay650_verify as replay
    from .cell1050_rebound_study import control_requests
    from .direct_path_first_audit import verify_heading
    from .intergrid_route_labels import annotate_routes
    from .parallel_capture_grid_audit import observed_rows, rescore_packet, verify_release
    from .policy64_verify import audit_native_options
    from .table_training_verify import control_gate, rescore_rows

    study, checks = Path(a.output), Checks()
    spec,plan,source,prior,home,resolutions = verify_preparation(a,checks)
    report = read(study/'report.json')
    checks.equal((report['version'],report['status'],report['phase'],report['scope'],report['physics_substeps']),
                 (STUDY_VERSION,'completed','completed','development',64),'Completed native64 development experiment')
    checks.require(0 < report['gpu_lease_s'] <= 3600, 'Bounded acquired4090 time')
    manifest = replay.check_manifest(study,checks)
    checks.equal(report['evidence_manifest_sha256'],sha(study/'evidence-manifest.json'),'Final report binds all saved evidence')
    for name,value in spec['frozen_files'].items():
        checks.equal(manifest[name],value,'Final manifest preserves frozen preparation/'+name)
    calibration = replay.check_calibration(study,spec['identity']['calibration'],checks)
    verify_launch_receipt(study,checks)
    slots = read(study/'slots.json'); repeat_slots = read(study/'repeat-slots.json')
    verify_request_keys(plan, slots, checks)
    preserved_slots=read(study/'preserved-slots.json'); verify_preservation_slots(plan,preserved_slots,checks)
    checks.equal(read(study/'native-started.json')['slots_sha256'], sha(study/'slots.json'), 'Native launch binds frozen initial commands')
    initial_count = 6+sum(len(c['candidates']) for c in plan['cases'])
    checks.equal((spec['initial_requests'], spec['maximum_confirmation_requests'], spec['maximum_total_requests']),
                 (initial_count, (initial_count-6)*3, initial_count+(initial_count-6)*3+27), 'Explicit maximum native budget')
    ledger, timings = observed_rows(study/'rows.jsonl'), observed_rows(study/'timing.jsonl')
    sources = read(a.parent_study/'control-replays.json')
    used = {c['directory']: set() for c in plan['cases']+plan['preserved_cases']}; seen = set(); phases = []; observed = []; controls = {}
    expected_commands = {s['request']['id']: s for s in preserved_slots+slots+repeat_slots}
    checks.equal(len(expected_commands), len(preserved_slots)+len(slots)+len(repeat_slots), 'Globally unique diagnostic commands')
    for index, path in enumerate(sorted((study/'requests').glob('*.json'))):
        relative = f'requests/batch-{index:04d}.json'
        checks.equal(path.relative_to(study).as_posix(), relative, 'Contiguous native packet ledger')
        packet = read(path); phase = packet['phase']; requests = packet['rows']; directory = packet['adapter_directory']
        control = phase in ('controls_pre', 'controls_post')
        checks.equal(packet['control'], control, relative+'/control declaration')
        checks.require(phase in ('controls_pre', 'preservation', 'diagnostic', 'confirmation', 'controls_post'), relative+'/known phase')
        phases.append(('controls_pre', 'preservation', 'diagnostic', 'confirmation', 'controls_post').index(phase))
        checks.require(0 < len(requests) <= 16, relative+'/bounded native batch')
        timing = next(timings, None); checks.require(timing is not None, relative+'/timing exists')
        checks.equal((timing['packet'], timing['phase']), (relative, phase), relative+'/matching timing identity')
        if control:
            checks.equal(directory, 'native-controls', relative+'/control directory')
            checks.require(phase not in controls, relative+'/one control block per side')
            checks.equal(requests, control_requests(sources, phase), relative+'/exact control requests')
        else:
            checks.require(directory in used, relative+'/known diagnostic directory')
        rows = []
        for request in requests:
            row = next(ledger, None); checks.require(row is not None, request['id']+'/native observation exists')
            verify_release(row, request, relative, manifest[relative], phase, checks)
            if row.get('route_success'):
                checks.require(row.get('shot_route') in ('direct', 'table') or control,
                               row['id']+'/standard success has explicit direct/table route')
            checks.require(row['id'] not in seen, row['id']+'/exactly one native release'); seen.add(row['id'])
            checks.equal(row['adapter_directory'], directory, row['id']+'/native archive namespace')
            if not control:
                slot = expected_commands[request['id']]
                checks.equal((slot['request'], slot['directory']), (request, directory), row['id']+'/frozen exact command')
                for field in ('variant', 'repeat', 'context_id', 'candidate_id'):
                    checks.equal(row[field], request[field], row['id']+'/'+field)
                verify_heading(row, request, study/directory, spec['scene'], home, prior, resolutions[directory], checks)
            rows.append(row)
        if control:
            controls[phase] = rows
        else:
            alias = {'native-heading/'+name[len(directory)+1:]: v for name, v in manifest.items() if name.startswith(directory+'/')}
            list(rescore_packet(study/directory, rows, spec['scene'], calibration, alias, used[directory], checks))
            relabeled = annotate_routes(copy.deepcopy(rows), study/directory)
            for row, other in zip(rows, relabeled, strict=True):
                checks.equal((row['shot_route'], row['table_rebounds_before_entry']),
                             (other['shot_route'], other['table_rebounds_before_entry']), row['id']+'/independent route label')
            observed.extend(rows)
    checks.equal(phases, sorted(phases), 'Controls bracket initial search then confirmations')
    checks.require(next(ledger, None) is None and next(timings, None) is None, 'No unaccounted rows or timings')
    checks.equal([r['id'] for r in observed], [s['request']['id'] for s in preserved_slots+slots+repeat_slots], 'Every scheduled diagnostic executed once in order')
    eligible = verify_confirmations(plan, observed, checks)
    verify_request_keys(plan, repeat_slots, checks, confirmations=True, eligible=eligible)
    for directory, archives in used.items():
        audit_native_options(study/directory, spec['scene'], calibration, a.assets, checks)
        checks.equal(archives, {p.relative_to(study/directory).as_posix() for p in (study/directory).rglob('*.npz')},
                     directory+'/complete native archive inventory')
    control_rows = [r for group in controls.values() for r in group]
    audit_native_options(study/'native-controls', spec['scene'], calibration, a.assets, checks)
    rescore_rows(study/'native-controls', control_rows, spec['scene'], calibration, spec['identity']['trace_schema'], a.assets, checks)
    checks.equal({r['native_cache_file'] for r in control_rows},
                 {p.relative_to(study/'native-controls').as_posix() for p in (study/'native-controls').rglob('*.npz')}, 'Complete native control archives')
    checks.equal(set(controls), {'controls_pre', 'controls_post'}, 'Both matched control blocks present')
    for phase, records in controls.items():
        gate = control_gate(records); checks.equal(gate['status'], 'passed', phase+'/native control behavior')
        checks.equal(report['controls'][phase], gate, phase+'/independent control counts')
    preserved=[r for r in observed if r.get('variant')=='preserved']
    checks.equal(report['preserved_shots'],dict(requested=9,completed=len(preserved),captures=sum(qualified(r) for r in preserved),passed=len(preserved)==9 and all(qualified(r) for r in preserved),rows=preserved),'All nine unchanged successful recipes preserved')
    checks.require(len(preserved)==9 and all(qualified(r) for r in preserved),'Three successful settings still capture3/3')
    summary = independent_summary(plan, observed)
    for item, prepared in zip(summary, plan['cases'], strict=True):
        item.update(nearest_seed_distance_m=None,source_counts=prepared['source_counts'])
    checks.require(len(observed)+18 <= 289, 'Authorized total native release cap')
    checks.equal(report['cases'], summary, 'Independent proposal, original, search and confirmation counts')
    accounting=release_accounting(len(slots),len(repeat_slots),len(preserved))
    checks.equal(len(observed),len(slots)+len(repeat_slots)+len(preserved),'Complete diagnostic and preservation denominator')
    checks.equal(report['progress'], accounting['progress'], 'Complete diagnostic denominator')
    checks.equal((report['native_releases'], report['initial_requests'], report['confirmation_requests']),
                 (accounting['native_releases'], len(slots), len(repeat_slots)), 'All native releases including controls accounted')
    return dict(version=VERSION, status='passed', checks_count=checks.count, report_sha256=sha(study/'report.json'),
        plan_sha256=sha(study/'preparation.json'), source_bank_sha256=sha(study/'source-bank.json'),
        contract_sha256=sha(study/'contract.json'), evidence_manifest_sha256=sha(study/'evidence-manifest.json'),
        verifier_source_sha256=sha(Path(__file__)), frozen_source_root=str(source),
        initial_requests=len(slots), confirmation_requests=len(repeat_slots), preservation_requests=len(preserved), control_releases=len(control_rows), preserved_shots={k:v for k,v in report['preserved_shots'].items() if k!='rows'}, diagnosis_sha256=sha(study/'diagnosis.json'),
        results=[dict(id=c['id'], bucket_xy_m=c['bucket_xy_m'], proposed=c['proposed'], checked=c['checked'],
            admitted=c['admitted'], selected=c['selected'], results={v: {k: x for k, x in g.items() if k != 'rows'}
                for v, g in c['results'].items()}, confirmed_settings=c['confirmed_settings']) for c in summary])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('source-study','source-audit','diagnosis','parent-study','parent-audit','assets','output','audit'):
        parser.add_argument('--'+key,type=Path,required=True)
    parser.add_argument('--source-root',type=Path)
    parser.add_argument('--stage',choices=('preflight','final'),default='final')
    args=parser.parse_args()
    roots=[args.output,args.source_study,args.parent_study,args.source_audit,args.parent_audit,args.diagnosis]
    targets=(args.audit,args.audit.with_suffix(args.audit.suffix+'.tmp'))
    if args.audit.exists() or any(p.resolve()==r.resolve() or p.resolve().is_relative_to(r.resolve()) for p in targets for r in roots):
        parser.error('Write a fresh audit outside all immutable source evidence')
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':parser.error('Hide GPUs for this CPU-only audit')
    try:result=preflight(args) if args.stage=='preflight' else audit(args)
    except Exception as error:result=dict(version=PREPARATION_VERSION if args.stage=='preflight' else VERSION,status='failed',error=f'{type(error).__name__}: {error}')
    atomic(args.audit,result)
    print(json.dumps(result),flush=True)
    if result['status']!='passed':raise SystemExit(1)


if __name__=='__main__':main()
