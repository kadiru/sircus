"""Frozen seven-hour, spatially stratified search/imitation/PPO protocol.

Only geometry from the two previously inspected Case00 captures augments the
shared prior. Those examples are development information, never new TEST data.
The new actor/critic fit uses fresh TRAIN search outcomes only.
"""
from __future__ import annotations
import copy
import hashlib
import json
import json
from pathlib import Path
import numpy as np
from . import table_training_contract as old
from .benchmark_contract import RESIDUAL_SCALE
from .learning_contract import digest
from .corrected_contract import sha
from .table_assisted_score import reward, VERSION as SCORING_VERSION, ROUTE_VERSION, REWARD_VERSION

VERSION='sircus-coverage-training7h-v2'
AUDIT_VERSION='sircus-coverage-training7h-audit-v2'
BC_SEED=2026110901
SEEDS=(2026110902,2026110903)
TRAIN_SEED,VALIDATION_SEED,TEST_SEED=2026110911,2026110912,2026110913
MILESTONES=(0,256,512)
METHODS=('current_bc','current_ppo','bc','selected_ppo')
SCOPE=('Privileged-geometry held-endpoint shot selection at fixed64 with unchanged v3 scoring. '
       'Shared explicit19-recipe geometry bank:17 original plus2 audited Case00 development poses. '
       'Fresh TRAIN search fits imitation and critic; two PPO seeds start from the same fitted model. '
       'One frozen decision per untouched TEST layout and method, all setup declines counted. '
       'No learned grasp/transit, no perception/noise/128 qualification, no policy promotion.')
captured,clean,statistics,select_teachers=old.captured,old.clean,old.statistics,old.select_teachers
collect_exclusions=old.collect_exclusions


def protocol():
    return dict(control_prior='Exact authenticated original17 control branches; expanded19 shared only by search and policies',substeps=64,bc_seed=BC_SEED,seeds=list(SEEDS),train_seed=TRAIN_SEED,
        validation_seed=VALIDATION_SEED,test_seed=TEST_SEED,train_layouts=105,
        historical_train_layouts=35,new_train_layouts=70,validation_layouts=35,test_layouts=70,
        grid_columns=7,grid_rows=5,bucket_bounds_m=[[.1,-.65],[1.24,.25]],
        minimum_split_distance_m=.005,search_rounds=12,search_requests=1260,
        train_requests_per_seed=512,train_requests=1024,batch_size=32,updates_per_seed=16,
        milestones=list(MILESTONES),validation_requests=175,test_methods=list(METHODS),
        test_requests=280,test_repeats=1,control_requests=18,maximum_fresh_requests=2757,
        global_limit_s=25200,acquired_slot_limit_s=19800,search_limit_s=7200,train_cut_s=16200,
        storage_reserve_bytes=8*1024**3,noise=False,actor_steps=800,critic_steps=400,
        bootstrap_lr=.001,ppo_lr=.0003,initial_logstd=-3.,ppo_epochs=6,clip=.2,
        entropy=.001,gradient_norm=1.,residual_scale=RESIDUAL_SCALE.tolist(),
        scoring_version=SCORING_VERSION,route_version=ROUTE_VERSION,reward_version=REWARD_VERSION,
        selection='Validation captures, mean v3 reward, earliest samples, then seed; BC at step0 eligible; freeze before TEST',
        search_order='Literal105 TRAIN order each round; all failures consume requests; no resampling',
        bootstrap='Fresh TRAIN search only: one best captured teacher/context; all fresh search rewards fit critic',
        time_truncation='Stop search at a completed batch after7200s; stop interleaved PPO at completed seed-pair boundaries after16200s; choose only validated milestones; full280 TEST and18 controls required for completion',
        common_prior='All four methods share expanded19 geometric sources; old checkpoint weights are unchanged, but this is not their historical prior/evaluation',scope=SCOPE)


def require(value,message):
    if not value: raise ValueError(message)


def read(path):return json.loads(Path(path).read_text())


def contained(root,name):
    return old.contained(root,name)


def verified_study(study,audit,public,*,kind):
    study,audit=Path(study),Path(audit)
    if kind=='current':
        expected=public['provenance'];report_hash=expected['report_sha256'];audit_hash=expected['independent_audit_sha256']
        version='sircus-table-assisted-training-v1';audit_version='sircus-table-assisted-training-audit-v1';count=823442
    else:
        expected=public['audit'];report_hash=expected['report_sha256'];audit_hash=expected['proof_sha256']
        version='sircus-case00-tilt-gap-study-v1';audit_version='sircus-case00-tilt-gap-audit-v1';count=27547
    require(sha(study/'report.json')==report_hash and sha(audit)==audit_hash,'Published '+kind+' evidence binding differs')
    report,proof,spec,manifest=(read(p) for p in (study/'report.json',audit,study/'contract.json',study/'evidence-manifest.json'))
    unsigned=dict(spec);fp=unsigned.pop('fingerprint')
    require(digest(unsigned)==fp==report['contract_fingerprint']==proof['contract_fingerprint'],'Parent contract differs')
    checks=proof.get('checks',[])
    require(report['version']==version and report['status']=='completed' and proof['version']==audit_version
        and proof['status']=='passed' and proof.get('full_checks_verified') is True and proof['failures']==[]
        and len(checks)==proof['checks_count']==count and all(c.get('passed') is True for c in checks),'Completed parent audit required')
    require(report['evidence_sha256']==manifest and proof['evidence_manifest_sha256']==sha(study/'evidence-manifest.json'),'Parent manifest differs')
    if kind=='current':
        require(sha(study/'contract.json')==expected['contract_file_sha256'] and sha(study/'evidence-manifest.json')==expected['evidence_manifest_sha256'],'Current published manifest/contract differ')
    for name in ('contract.json',):
        if name in manifest:require(sha(study/name)==manifest[name],'Manifest '+name+' differs')
    return spec,report,manifest,dict(report_sha256=report_hash,audit_sha256=audit_hash,
        contract_sha256=sha(study/'contract.json'),manifest_sha256=sha(study/'evidence-manifest.json'),contract_fingerprint=fp)


def load_inputs(a,root):
    root=Path(root);current=Path(a.current_study);case=Path(a.case00_study)
    parent,report,manifest,current_binding=verified_study(current,a.current_audit,
        read(root/'baselines/mjlab-table-assisted-training-20261108/summary.json'),kind='current')
    case_spec,case_report,case_manifest,case_binding=verified_study(case,a.case00_audit,
        read(root/'baselines/mjlab-case00-tilt-gap-20261109/summary.json'),kind='case00')
    def current_read(name):
        require(sha(contained(current,name))==manifest[name],'Current source changed: '+name)
        return read(current/name)
    old_prior=current_read('prior.json');controls=current_read('controls.json')
    require(len(old_prior['records'])==17 and len(controls)==3,'Expected17 frozen geometric sources and3 controls')
    current_read('calibration.json');current_read('setup-home.json');current_read('verification.json')
    require(sha(current/'rows.jsonl')==manifest['rows.jsonl'],'Current original control rows changed')
    original_rows=[json.loads(line) for line in (current/'rows.jsonl').read_text().splitlines() if line.strip()]
    control_replays=[]
    for role in ('direct_control','table_control','negative_control'):
        matched=[r for r in original_rows if r.get('phase')=='controls_pre' and r.get('role')==role and r.get('repeat')==0]
        require(len(matched)==1 and matched[0].get('native_release') is True,'Exact admitted original control required')
        r=matched[0]
        for key in ('context_resolution','connection'):
            name='native/'+r[key+'_file'];require(sha(contained(current,name))==r[key+'_sha256']==manifest[name],'Original control proof changed')
        cache='native/'+r['native_cache_file'];require(sha(contained(current,cache))==r['native_cache_sha256']==manifest[cache],'Original control native data changed')
        control_replays.append(copy.deepcopy(r))
    calibration=read(current/'calibration.json');home=read(current/'setup-home.json')
    require(parent['identity']['scene_sha256']==case_spec['identity']['scene_sha256']
        and parent['identity']['asset_sha256']==case_spec['identity']['asset_sha256'],'Case00 geometry differs from current training scene')
    old_artifacts={}
    for method,key in (('current_bc','bc'),('current_ppo','selected')):
        info=copy.deepcopy(report['artifacts'][key]);relative=info['path'];p=contained(current,relative)
        require(sha(p)==info['sha256']==manifest[relative],'Current frozen model changed')
        info.update(path=str(p),relative_path=relative,historical_relative_path=relative)
        old_artifacts[method]=info
    require(sha(case/'rows.jsonl')==case_manifest['rows.jsonl'],'Case00 rows changed')
    wanted=('case00-tilt-gap-table-training-test-00-g34','case00-tilt-gap-table-training-test-00-g64')
    rows={r['id']:r for r in map(json.loads,(case/'rows.jsonl').read_text().splitlines())}
    sources=[]
    for name in wanted:
        r=rows[name]
        require(captured(r) and not r.get('robot_contact_steps') and r['contact_protocol']['substeps']==64,'Audited Case00 capture required')
        cache='native/'+r['native_cache_file']
        require(sha(contained(case,cache))==r['native_cache_sha256']==case_manifest[cache],'Case00 native evidence changed')
        sources.append(dict(id='coverage-development-'+name,source_id=name,source_namespace='case00-tilt-gap-v1-development',
            source_study='case00-tilt-gap-v1',source_record_sha256=digest(r),values=copy.deepcopy(r['values']),
            bucket_xy_m=copy.deepcopy(r['bucket_xy_m']),q10=copy.deepcopy(r['native_initial_q10']),q14=copy.deepcopy(r['native_initial_q14'])))
    prior=copy.deepcopy(old_prior);prior['records']+=sources
    prior.update(records_sha256=digest(prior['records']),recipes=19,version='sircus-coverage-expanded-geometric-prior-v1',
        development_source_ids=list(wanted),scope='17 original sources plus2 explicitly reclassified Case00 development sources; shared geometry only, no new TEST outcomes')
    bindings=dict(current=current_binding,case00=case_binding,current_rows_sha256=manifest['rows.jsonl'],control_replays_sha256={r['role']:digest(r) for r in control_replays},current_artifacts=copy.deepcopy(old_artifacts),
        prior_sha256=digest(prior),development_source_sha256={r['source_id']:r['source_record_sha256'] for r in sources},
        current_files_sha256={name:manifest[name] for name in ('prior.json','controls.json','calibration.json','setup-home.json','verification.json')})
    return dict(parent=parent,prior=prior,old_prior=old_prior,old_artifacts=old_artifacts,
        controls=controls,control_replays=control_replays,calibration=calibration,home=home,bindings=bindings,train_splits=parent['splits'])


def make_splits(train_splits,excluded):
    original=train_splits['train'] if isinstance(train_splits,dict) else train_splits
    require(len(original)==35 and len({r['id'] for r in original})==35,'Original35 TRAIN rows required')
    train=[dict(id=r['id'],split='train',bucket_xy_m=list(r['bucket_xy_m']),origin='historical_train') for r in original]
    points={tuple(map(float,xy)) for xy in excluded};points.update(tuple(r['bucket_xy_m']) for r in train)
    exclusions=[list(p) for p in sorted(points)];require(exclusions,'Finite nonempty coordinate exclusions required')
    used=np.asarray(exclusions,float);require(used.ndim==2 and used.shape[1]==2 and np.isfinite(used).all(),'Finite exclusion coordinates')
    result=dict(train=train);low=np.asarray([.1,-.65]);size=(np.asarray([1.24,.25])-low)/[7,5]
    for role,n,seed in (('train',2,TRAIN_SEED),('validation',1,VALIDATION_SEED),('test',2,TEST_SEED)):
        rng=np.random.default_rng(seed);new=[]
        for iy in range(5):
            for ix in range(7):
                for sample in range(n):
                    for attempt in range(10000):
                        xy=low+size*([ix,iy]+rng.uniform(.05,.95,2))
                        if np.linalg.norm(used-xy,axis=1).min()>=.005:break
                    else:raise ValueError('Cannot sample disjoint grid cell')
                    new.append(dict(id=f'coverage-{role}-r{iy}-c{ix}-n{sample}',split=role,
                        bucket_xy_m=xy.tolist(),grid_cell=[ix,iy],origin='fresh_stratified'))
                    used=np.vstack((used,xy))
        if role=='train':result['train']+=new
        else:result[role]=new
    return result,exclusions


def schedule(seed):
    require(seed in SEEDS,'Declared PPO seed required');rng=np.random.default_rng(seed);indices=[]
    while len(indices)<512:indices.extend(rng.permutation(105).tolist())
    return indices[:512]


def search_residual(context,round_index,best=None):
    require(type(round_index) is int and 0<=round_index<12,'Search round0..11 required')
    require(context.get('split')=='train','Only TRAIN can be searched')
    changes=((0,0,0,0,0),(0,0,0,-2,0),(0,0,0,2,0),(0,0,0,0,-.01),(0,0,0,0,.01),
             (0,0,-.025,0,0),(0,0,.025,0,0),(-.02,0,0,0,0),(.02,0,0,0,0),(0,.04,0,0,0))
    sources=[]
    if round_index<10:
        r=np.asarray(changes[round_index],float)/RESIDUAL_SCALE;strategy='prescribed_axis_probe'
    else:
        if best is not None:
            require(best.get('role')=='train' and best['context_id']==context['id'],'Refinement may use only earlier same-context TRAIN evidence')
            center=np.asarray(best['normalized_residual'],float);sources=[best['id']]
        else:center=np.zeros(5)
        require(center.shape==(5,) and np.isfinite(center).all() and (np.abs(center)<=1).all(),'Finite bounded center')
        seed=int(hashlib.sha256(f'{BC_SEED}:{context["id"]}:{round_index}'.encode()).hexdigest()[:16],16)
        rng=np.random.default_rng(seed)
        r=np.tanh(np.arctanh(np.clip(center,-.999999,.999999))+rng.normal(0,.2,5));strategy='train_best_joint_refinement'
    return r,dict(strategy=strategy,proposal_index=round_index,source_training_ids=sources)
