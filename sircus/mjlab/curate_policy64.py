"""Export byte-bound fixed64 policy evidence and two audited checkpoints.

The exporter uses only the standard library. It does not load pickle/Torch,
replay a policy, rescore contacts, retrain or modify the original evidence.
The completed independent audit must already validate checkpoint contents.
"""
from collections import Counter
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import re

from .curate_xy650 import boolean, digest, fingerprint, hash_map, identifier, number, read, relative_file, require, sha, write_json

VERSION = 'sircus-policy64-curation-v1'
STUDY_VERSION = 'sircus-policy64-bc-ppo-v1'
AUDIT_VERSION = 'sircus-policy64-independent-audit-v1'
METHODS = ('prior','nearest','bc','ppo')
MILESTONES = (0,256,512,1024)
SOURCE_FP = 'f96e6c9e0878564ef88c626cc2cbc46cc48a83f803a16300b3d1945d3fd62f11'
SOURCE_AUDIT_SHA = '8df7b8f3b48cfc9c1c41f33f02ea0b68ca828909e2c8774d3cbabc0f1a45f530'
SOURCE_REPORT_SHA = '8d59cf3d44d6657aa8cddd5a22270e39f044357c51ebf690ae0ba940ccf44ba7'
SOURCE_MANIFEST_SHA = '970f2b079c5d00491e95b3f58bfd1180e58fd366d116e0ab4e44731073f9ccb4'
SCOPE = ('Fixed64 warm-start ablation: 6,144 historical TRAIN requests and 22 successful-context '
         'teachers bootstrap BC; PPO adds 1,024 fresh TRAIN requests. Twelve fresh VAL layouts '
         'select a frozen milestone; four methods get one nominal shot at each of 50 fresh TEST '
         'positions. Privileged geometry, prescribed held endpoints and geometry-only setup '
         'resolution, without ball-flight search at TEST. Not equal-acquisition-budget algorithm '
         'superiority, noise robustness, 128-substep qualification, all-table coverage, learned '
         'grasp/transit, perception or physical validation. Historical 128 failures are unchanged.')


def protocol(raw):
    fixed = dict(substeps=64,seed=20261511,train_layouts=35,validation_layouts=12,test_layouts=50,
        train_requests=1024,batch_size=32,updates=32,milestones=list(MILESTONES),test_methods=list(METHODS),
        test_repeats=1,validation_requests=48,test_requests=200,control_requests=12,maximum_fresh_requests=1284,
        noise=False,actor_steps=800,critic_steps=400,bootstrap_lr=.001,ppo_lr=.0003,initial_logstd=-3.,
        ppo_epochs=6,clip=.2,entropy=.001,gradient_norm=1.)
    require(all(raw.get(k)==v for k,v in fixed.items()),'Unexpected fixed64 learning protocol')
    return dict(**fixed,selection='Fresh VAL captures, then mean calibrated reward, then earliest milestone; frozen before TEST')


def finite_vector(value,size):
    require(isinstance(value,list) and len(value)==size,'Expected complete nominal geometry')
    return [number(v) for v in value]


def capture(row):
    native,success,invalid_physics = (boolean(row[k]) for k in ('native_release','route_success','invalid_physics'))
    require(type(row.get('route_quality')) is int and row['route_quality'] in (0,1,2),'Invalid capture quality')
    require(not success or native and not row.get('invalid') and not invalid_physics,'Capture claims invalid physics or a rejected release')
    return bool(native and success and not row.get('invalid') and not invalid_physics)


def statistics(rows):
    captures = [capture(r) for r in rows]
    return dict(attempts=len(rows),native_releases=sum(boolean(r['native_release']) for r in rows),
        captures=sum(captures),clean=sum(ok and r['route_quality']==2 for r,ok in zip(rows,captures)),
        invalid=sum(bool(r.get('invalid')) for r in rows),
        mean_reward=math.fsum(number(r['learning_reward']) for r in rows)/len(rows) if rows else 0.)


def checked_statistics(actual,rows):
    expected = statistics(rows)
    for key,value in expected.items():
        if key=='mean_reward':require(abs(number(actual.get(key))-value)<=1e-8,'Mean reward accounting differs')
        else:require(type(actual.get(key)) is int and actual[key]==value,'Outcome accounting differs: '+key)
    return expected


def artifact(study,raw,*,samples=None):
    require(isinstance(raw,dict),'Missing frozen artifact')
    count = raw.get('samples');require(type(count) is int and count in MILESTONES,'Unexpected checkpoint sample prefix')
    if samples is not None:require(count==samples,'Wrong checkpoint sample prefix')
    require(raw.get('path')==f'checkpoints/milestone-{count:04d}.pt','Unexpected checkpoint publication source')
    path = relative_file(study,raw['path']);expected = fingerprint(raw.get('sha256'))
    require(sha(path)==expected,'Checkpoint bytes differ')
    return dict(samples=count,sha256=expected,weight_sha256=fingerprint(raw.get('weight_sha256')))


def verified_inputs(study,audit_path):
    study,audit_path = Path(study).resolve(),Path(audit_path).resolve()
    report,spec,audit = (read(p) for p in (study/'report.json',study/'contract.json',audit_path))
    require(report.get('version')==spec.get('version')==STUDY_VERSION,'Unexpected fixed64 study version')
    require(report.get('status')==report.get('phase')=='completed','Fixed64 study is not completed')
    require(spec.get('evidence_kind')=='native_fixed64_success_bootstrap_ppo','Unexpected evidence kind')
    checks = audit.get('checks')
    require(audit.get('version')==AUDIT_VERSION and audit.get('status')=='passed' and audit.get('failures')==[]
        and isinstance(checks,list) and checks and type(audit.get('checks_count')) is int and audit['checks_count']==len(checks)
        and all(isinstance(c,dict) and c.get('passed') is True for c in checks),'Full exact-version independent audit required')
    require(audit.get('report_sha256')==sha(study/'report.json'),'Audit/report bytes differ')
    require(audit.get('evidence_manifest_sha256')==sha(study/'evidence-manifest.json'),'Audit/manifest bytes differ')
    unsigned=dict(spec);fp=unsigned.pop('fingerprint',None)
    require(fingerprint(fp)==digest(unsigned)==report.get('contract_fingerprint')==audit.get('contract_fingerprint'),'Contract fingerprint binding differs')
    manifest=hash_map(read(study/'evidence-manifest.json'))
    require(report.get('evidence_sha256')==manifest and manifest.get('contract.json')==sha(study/'contract.json'),'Report/contract manifest binding differs')
    for name,value in manifest.items():require(sha(relative_file(study,name))==value,'Changed evidence bytes: '+name)
    actual={str(p.relative_to(study)) for p in study.rglob('*') if p.is_file() and p.name not in ('report.json','evidence-manifest.json') and not p.name.startswith('audit')}
    require(set(manifest)==actual,'Manifest does not cover every evidence artifact')
    require(report['protocol']==spec['protocol'],'Executed protocol differs')
    require(audit.get('producer_source_sha256')==spec['source_sha256'],'Audited source identities differ')
    require(audit.get('source_contract_fingerprint')==SOURCE_FP,'Historical source contract differs')
    bindings=spec['source_bindings']
    for name,value in (('parent_contract_fingerprint',SOURCE_FP),('report_sha256',SOURCE_REPORT_SHA),('parent_audit_sha256',SOURCE_AUDIT_SHA),('parent_manifest_sha256',SOURCE_MANIFEST_SHA)):
        require(bindings.get(name)==value,'Historical source binding differs: '+name)
    files=hash_map(bindings['train_files_sha256'])
    require(set(files)=={f'runs/{m}-seed{s}/rows.jsonl' for m in ('ppo','sweep') for s in (20261411,20261412,20261413)},'Six historical TRAIN sources required')
    rows=[json.loads(line) for line in (study/'rows.jsonl').read_text().splitlines() if line]
    require(len(rows)==1284==audit.get('requested_trials') and len({r['id'] for r in rows})==1284,'Exactly1284 unique requested decisions required')
    require(sum(boolean(r['native_release']) for r in rows)==audit.get('native_releases'),'Native release accounting differs')
    return study,spec,report,audit,manifest,rows


def checked_splits(study,spec):
    splits=spec['splits'];require(read(study/'splits.json')==splits,'Saved split manifest differs')
    result={};used=set()
    for role,size in (('train',35),('validation',12),('test',50)):
        group=splits[role];require(isinstance(group,list) and len(group)==size,'35/12/50 split denominator differs')
        byid={}
        for r in group:
            ident=identifier(r['id']);xy=finite_vector(r['bucket_xy_m'],2)
            require(r['split']==role and ident not in byid and ident not in used,'Duplicate or mislabeled split identity')
            byid[ident]=xy;used.add(ident)
        result[role]=byid
    points=[tuple(xy) for group in result.values() for xy in group.values()]
    require(len(set(points))==97,'TRAIN/VAL/TEST coordinates overlap')
    return result


def checked_bootstrap(study,spec,report,audit,splits):
    raw=spec['bootstrap_summary'];require(report['bootstrap_summary']==raw,'Reported bootstrap evidence differs')
    fixed=dict(requests=6144,layouts=35,successful_layouts=22,teacher_count=22)
    require(all(raw.get(k)==v for k,v in fixed.items()),'Expected6144 historical TRAIN requests and22 teachers')
    dataset=read(study/'bootstrap/dataset.json');teachers=read(study/'bootstrap/teachers.json')
    require(len(dataset)==6144 and len({r['id'] for r in dataset})==6144 and len(teachers)==22,'Complete bootstrap denominator differs')
    require(all(r['role']=='train' and r['context_id'] in splits['train'] and r['bucket_xy_m']==splits['train'][r['context_id']] for r in dataset),'Held-out labels entered bootstrap')
    lookup={r['id']:r for r in dataset}
    require(len({r['context_id'] for r in teachers})==22 and all(r['id'] in lookup and r==lookup[r['id']] and capture(r) for r in teachers),'Teachers are not22 distinct successful TRAIN contexts')
    require(spec['source_bindings']['teacher_ids']==[r['id'] for r in teachers],'Teacher source IDs differ')
    for name,key in (('bootstrap/dataset.json','bootstrap_dataset_sha256'),('bootstrap/teachers.json','teachers_sha256')):
        require(sha(study/name)==spec[key],'Frozen bootstrap bytes differ')
    counts=statistics(dataset)
    selected={**fixed,**{k:counts[k] for k in ('native_releases','captures','clean')}}
    require(all(raw.get(k)==v and audit['bootstrap_summary'].get(k)==v for k,v in selected.items()),'Historical bootstrap acquisition accounting differs')
    selected['labels_scope']='Historical TRAIN only; 22 successful-context actor teachers, all 6,144 outcomes for critic. Not fresh TEST results.'
    return selected


def check_rows(rows,splits,selected_samples):
    expected=[('controls_pre','positive_control')]*3+[('controls_pre','negative_control')]*3+[('validation','validation')]*12
    for offset in range(0,1024,32):
        expected += [('train','train')]*32
        if offset+32 in MILESTONES:expected += [('validation','validation')]*12
    expected += [('test','test')]*200+[('controls_post','positive_control')]*3+[('controls_post','negative_control')]*3
    require([(r['phase'],r['role']) for r in rows]==expected,'Controls/VAL/TRAIN/TEST execution order differs')
    require([r['train_index'] for r in rows if r['role']=='train']==list(range(1024)),'Fresh TRAIN prefix differs')
    for r in rows:
        identifier(r['id']);capture(r);number(r['learning_reward'])
        require(r.get('action_substituted') is False and r.get('perturbation_mixed_units')==[0.]*5,'Action substitution or noise appeared')
        if r['role'] in splits:
            require(r['context_id'] in splits[r['role']] and r['bucket_xy_m']==splits[r['role']][r['context_id']],'Split context coordinates differ')
        if r['role'] in ('validation','test'):
            require(r.get('ball_flight_search_at_inference') is False and r.get('selection_or_learning') is False,'Flight search or learning at evaluation')
            require(finite_vector(r['values'],5)==finite_vector(r['command_values'],5),'Predicted action was replaced')
    for samples in MILESTONES:
        cohort=[r for r in rows if r['role']=='validation' and r['samples']==samples]
        require(len(cohort)==12 and {r['context_id'] for r in cohort}==set(splits['validation']) and all(r['method']=='ppo' for r in cohort),'Fresh VAL cohort differs')
    tests=[r for r in rows if r['role']=='test']
    require([r['method'] for r in tests]==[m for m in METHODS for _ in range(50)],'Four50-shot method cohorts differ')
    for method in METHODS:
        cohort=[r for r in tests if r['method']==method]
        require(len(cohort)==50 and {r['context_id'] for r in cohort}==set(splits['test']),'One requested shot per TEST layout required')
        require(all(r['samples']==(selected_samples if method=='ppo' else 0) for r in cohort),'Wrong frozen checkpoint tested')


def portable_contract(radius):
    return dict(version='sircus-policy64-portable-residual-actor-v1',study_version=STUDY_VERSION,substeps=64,
        inputs=dict(dtype='float32',size=9,order=['target_bucket_x_norm','target_bucket_y_norm','anchor_bucket_x_norm','anchor_bucket_y_norm','anchor_board_x_norm','anchor_board_y_norm','anchor_board_height_norm','anchor_tilt_norm','anchor_gap_norm'],
            bucket_center_m=[.67,-.20],bucket_scale_m=[.57,.45],anchor_action_low=[.30,-.50,.12,8.,.05],anchor_action_high=[1.04,.35,.60,60.,.65],
            formula='concat((target_bucket-center)/scale,(anchor_bucket-center)/scale,2*(anchor_values-low)/(high-low)-1)'),
        actor=dict(layers=[9,64,64,5],hidden_activations=['tanh','tanh'],state_prefix='actor.',deterministic_output='tanh(actor(observation))',
            checkpoint_policy_key='policy',actor_state_keys=['actor.0.weight','actor.0.bias','actor.2.weight','actor.2.bias','actor.4.weight','actor.4.bias'],
            loader='PyTorch torch.load(path,map_location="cpu",weights_only=True); from checkpoint["policy"] select actor.* entries, strip the actor. prefix, and load into Sequential(Linear(9,64),Tanh(),Linear(64,64),Tanh(),Linear(64,5)). Evaluate in eval/no_grad mode; apply tanh to the five output means.'),
        full_training_state=dict(critic_layers=[9,64,64,1],critic_hidden_activations=['tanh','tanh'],logstd_size=5,
            checkpoint_keys=['policy','optimizer','samples','seed','method','contract_fingerprint','weight_sha256','actor_sha256']),
        action=dict(size=5,order=['board_center_x_m','board_center_y_m','board_center_height_m','board_tilt_deg_from_horizontal','vertical_ball_bottom_gap_m'],residual_scale=[.04,.04,.05,8.,.12],
            formula='anchor_values+tanh(actor(observation))*residual_scale',bounds_low=[.30,-.50,.12,8.,.05],bounds_high=[1.04,.35,.60,60.,.65],
            board_heading='Derived by the geometry provider to face the bucket; not a sixth learned output',
            ball_radius_m=radius,release_center='[board_x,board_y,board_height+gap+ball_radius]; vertical gap is above board center plane',
            invalid_action='Reject inadmissible bounds/IK/collision/setup actions; never substitute another shot or search flights at TEST'),
        required_external_geometry='Caller supplies target bucket XY and geometry-only admitted anchor bucket XY/five anchor action values. No raw prior joint bank is exported; checkpoints alone are not a complete arm setup resolver.',
        capture_contract=dict(version='sircus-calibrated-native-capture-v2',success='Board bounce, whole-ball entry and one continuous second retained. Rim-assisted captures count; clean is separate.'),
        training_only='Critic, optimizer and logstd are retained in original audited checkpoints; deterministic evaluation uses only the actor mean followed by tanh.',scope=SCOPE)


def curate(study,audit_path,output):
    study,spec,report,audit,manifest,rows=verified_inputs(study,audit_path)
    output=Path(output).resolve();require(not output.exists() and not output.is_relative_to(study),'New publication directory outside evidence required')
    declared=protocol(spec['protocol']);splits=checked_splits(study,spec);bootstrap=checked_bootstrap(study,spec,report,audit,splits)
    require(report.get('train_requests')==1024 and report.get('updates')==32,'PPO did not complete1024 requests and32 updates')
    updates=[json.loads(line) for line in (study/'updates.jsonl').read_text().splitlines() if line]
    require(len(updates)==32 and [r['index'] for r in updates]==list(range(32)),'Exact32 PPO update records required')
    train=[r for r in rows if r['role']=='train']
    require(all(r['train_ids']==[x['id'] for x in train[32*i:32*i+32]] for i,r in enumerate(updates)),'PPO update TRAIN identity differs')
    require([p['samples'] for p in report['curves']]==list(MILESTONES),'Literal validation milestones differ')
    curves=[]
    for point in report['curves']:
        cohort=[r for r in rows if r['role']=='validation' and r['samples']==point['samples']]
        curves.append(dict(samples=point['samples'],validation=checked_statistics(point['validation'],cohort),artifact=artifact(study,point['artifact'],samples=point['samples'])))
    best=max(curves,key=lambda p:(p['validation']['captures'],p['validation']['mean_reward'],-p['samples']))
    selected=artifact(study,report['selected']);bc=artifact(study,report['artifacts']['bc'],samples=0)
    require(selected==best['artifact'] and report['selected']==report['artifacts']['selected']==audit.get('selected'),'Selection was not frozen from VAL only')
    require(report['selected']==read(study/'selected.json')['selected']==read(study/'test-plan.json')['selected'],'Frozen selection records differ')
    require(read(study/'test-plan.json').get('frozen_before_outcomes') is True,'Whole TEST plan was not frozen before outcomes')
    check_rows(rows,splits,selected['samples'])
    counts=checked_statistics(report['counts'],rows)
    role_counts={role:statistics([r for r in rows if r['role']==role]) for role in ('train','validation','test','positive_control','negative_control')}
    for role,cohort in role_counts.items():checked_statistics(audit['role_counts'][role],[r for r in rows if r['role']==role])
    gates={}
    for phase,key in (('controls_pre','pre'),('controls_post','post')):
        cohort=[r for r in rows if r['phase']==phase];positive=[r for r in cohort if r['role']=='positive_control'];negative=[r for r in cohort if r['role']=='negative_control']
        require(len(cohort)==6 and len(positive)==len(negative)==3 and all(capture(r) for r in positive) and all(boolean(r['native_release']) and not capture(r) and not r.get('whole_ball_entry') for r in negative),'Matching64 control cohort failed')
        require(all(not r.get('invalid') and r.get('invalid_physics') is False and r.get('scalar_score_audit_passed') is True for r in cohort),'Control physics/oracle failed')
        actual=report['control_gates'][key];require(actual==audit['control_gates'][key] and actual.get('status')=='passed' and actual.get('positive_captures')==3 and actual.get('negative_captures')==0,'Independent control gate differs')
        gates[key]=dict(status='passed',counts=checked_statistics(actual['counts'],cohort),positive_captures=3,negative_captures=0)
    require(report['gate']==dict(status='passed',controls_passed=True,complete=True,substeps=64,scope='Protocol/controls completion, not a128convergence or minimum success-rate gate'),'Completion gate differs')
    require(report['test_comparison']==audit.get('test_comparison') and [r['method'] for r in report['test_comparison']]==list(METHODS),'Independent TEST comparison differs')
    tests=[r for r in rows if r['role']=='test'];comparison=[];positions=[]
    expected_positions=[]
    for result in report['test_comparison']:
        method=result['method'];cohort=[r for r in tests if r['method']==method]
        source=report['selected'] if method=='ppo' else report['artifacts']['bc'] if method=='bc' else report['artifacts'][method]
        require(sha(relative_file(study,source['path']))==fingerprint(source['sha256'])==result['artifact_sha256'],'TEST artifact bytes differ')
        require(all(r.get('artifact_sha256')==source['sha256']==r.get('frozen_artifact_sha256') for r in cohort),'TEST predictions used another artifact')
        comparison.append(dict(method=method,counts=checked_statistics(result['counts'],cohort),samples=result['samples'],artifact_sha256=source['sha256']))
        for r in cohort:
            values,bucket=finite_vector(r['values'],5),finite_vector(r['bucket_xy_m'],2);distance=math.dist(values[:2],bucket);ok=capture(r)
            invalid=r.get('invalid');require(invalid is None or isinstance(invalid,str),'Invalid reason must be text or unknown')
            if invalid is not None:identifier(invalid)
            expected_positions.append(dict(context_id=r['context_id'],bucket_xy_m=r['bucket_xy_m'],method=method,values=r['values'],board_bucket_distance_m=distance,capture=ok,clean=ok and r['route_quality']==2,native_release=r['native_release'],invalid=invalid,invalid_physics=r['invalid_physics']))
            positions.append(dict(id=identifier(r['id']),context_id=identifier(r['context_id']),method=method,samples=result['samples'],artifact_sha256=source['sha256'],
                bucket_x_m=bucket[0],bucket_y_m=bucket[1],board_x_m=values[0],board_y_m=values[1],board_height_m=values[2],board_tilt_deg=values[3],nominal_plane_gap_m=values[4],board_bucket_distance_m=distance,
                native_release=r['native_release'],capture=ok,clean=ok and r['route_quality']==2,invalid=invalid,invalid_physics=r['invalid_physics'],learning_reward=number(r['learning_reward'])))
    actual=report['test_positions'];require(len(actual)==len(expected_positions)==200,'All200 TEST positions required')
    for a,b in zip(actual,expected_positions):
        require(set(a)==set(b) and all(abs(number(a[k])-b[k])<=1e-12 if k=='board_bucket_distance_m' else a[k]==b[k] for k in b),'Reported TEST position/action/outcome differs')
    radius=number(spec['scene']['ball']['radius_m']);require(0<radius<1,'Finite positive ball radius required')
    checkpoints={}
    for key,name,raw in (('bc','bc-policy.pt',report['artifacts']['bc']),('selected','selected-policy.pt',report['selected'])):
        payload=relative_file(study,raw['path']).read_bytes();require(hashlib.sha256(payload).hexdigest()==raw['sha256'],'Checkpoint changed before copying')
        checkpoints[key]=dict(file=name,sha256=raw['sha256'],weight_sha256=raw['weight_sha256'],samples=raw['samples'],payload=payload)
    elapsed=number(report['elapsed_s']);require(elapsed>=0,'Negative reported worker runtime')
    packages={}
    for key,value in spec['packages'].items():
        identifier(key);require(isinstance(value,str) and re.fullmatch(r'[A-Za-z0-9_.+!,=-]+',value),'Unsafe package-version value')
        packages[key]=value
    summary=dict(version=VERSION,study_version=STUDY_VERSION,study_id=identifier(study.name),status='completed_independently_audited',scope=SCOPE,
        protocol=declared,packages=packages,bootstrap_summary=bootstrap,train_requests=1024,updates=32,reported_worker_elapsed_s=elapsed,counts=counts,
        counts_by_role=role_counts,control_gates=gates,gate=dict(status='completed',scope='Protocol/controls completion only; no minimum capture or timestep-convergence gate'),
        curves=curves,selected=selected,test_comparison=comparison,test_layouts=[dict(context_id=key,bucket_xy_m=xy) for key,xy in splits['test'].items()],
        checkpoints={key:{k:v for k,v in row.items() if k!='payload'} for key,row in checkpoints.items()},portable_policy_contract=portable_contract(radius),
        provenance=dict(contract_fingerprint=spec['fingerprint'],contract_sha256=sha(study/'contract.json'),report_sha256=sha(study/'report.json'),evidence_manifest_sha256=sha(study/'evidence-manifest.json'),evidence_sha256=manifest,
            independent_audit_version=AUDIT_VERSION,independent_audit_sha256=sha(audit_path),independent_auditor_sha256=fingerprint(audit['auditor_sha256']),independent_audit_checks=audit['checks_count'],
            source_contract_fingerprint=SOURCE_FP,source_report_sha256=SOURCE_REPORT_SHA,source_audit_sha256=SOURCE_AUDIT_SHA,source_manifest_sha256=SOURCE_MANIFEST_SHA,
            source_train_files_sha256=hash_map(spec['source_bindings']['train_files_sha256']),executed_source_sha256=hash_map(spec['source_sha256']),asset_sha256=hash_map(spec['identity']['asset_sha256']),
            scene_sha256=fingerprint(spec['identity']['scene_sha256']),bootstrap_dataset_sha256=spec['bootstrap_dataset_sha256'],teachers_sha256=spec['teachers_sha256'],splits_sha256=sha(study/'splits.json')),
        privacy='Nominal TEST geometry, scalar outcomes, model checkpoints and hashes only. No raw prior joints, poses, contact arrays, trajectories, host paths or credentials. Checkpoints copied unchanged only after their exact bytes passed the independent audit; the exporter never unpickles them.')
    dependencies=audit.get('auditor_identity',{}).get('dependencies_sha256')
    if dependencies:summary['provenance']['auditor_dependencies_sha256']=hash_map(dependencies)
    output.mkdir(parents=True)
    for row in checkpoints.values():(output/row['file']).write_bytes(row['payload'])
    write_json(output/'summary.json',summary)
    with (output/'test-positions.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(positions[0]),lineterminator='\n');writer.writeheader();writer.writerows(positions)
    write_json(output/'checksums.json',{p.name:sha(p) for p in sorted(output.iterdir()) if p.is_file()})
    return summary


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--study',type=Path,required=True);p.add_argument('--audit',type=Path);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();proof=a.audit or a.study.with_name(a.study.name+'-audit.json');result=curate(a.study,proof,a.output)
    print(json.dumps(dict(status=result['status'],selected=result['selected'],test_comparison=result['test_comparison'])))


if __name__=='__main__':main()
