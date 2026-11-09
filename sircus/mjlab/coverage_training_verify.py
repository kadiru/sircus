"""Independent CPU-only evidence audit for the bounded coverage training study.

No producer proposal/selection functions are used to reconstruct training.
Dense native scoring and geometry proof checks reuse the earlier independent
oracles; historical datasets and scoring labels are never modified.
"""
from __future__ import annotations
import argparse
from collections import Counter
import copy
import hashlib
import json
import math
from pathlib import Path
import time
import numpy as np
from . import benchmark_verify as common
from . import calibrated_benchmark_verify as calibrated
from . import policy64_verify as old
from . import replay650_verify as replay
from . import table_training_verify as native
from . import table_assisted_score_oracle as oracle
from .benchmark_verify import EvidenceError, contained, digest, read, sha

VERSION = 'sircus-coverage-training7h-audit-v1'
STUDY_VERSION = 'sircus-coverage-training7h-v1'
BC_SEED = 2026110901
SEEDS = (2026110902, 2026110903)
METHODS = ('current_bc', 'current_ppo', 'bc', 'selected_ppo')
CONTROL_ROLES = ('direct_control', 'table_control', 'negative_control')
RESIDUAL = np.asarray([.04, .04, .05, 8., .12])
AuditChecks = replay.AuditChecks
captured = native.captured
statistics = native.statistics


def protocol_checks(protocol, checks):
    expected=dict(substeps=64,bc_seed=BC_SEED,seeds=list(SEEDS),train_seed=2026110911,
        validation_seed=2026110912,test_seed=2026110913,train_layouts=105,
        historical_train_layouts=35,new_train_layouts=70,validation_layouts=35,test_layouts=70,
        grid_columns=7,grid_rows=5,bucket_bounds_m=[[.1,-.65],[1.24,.25]],
        minimum_split_distance_m=.005,search_rounds=12,search_requests=1260,
        train_requests_per_seed=512,train_requests=1024,batch_size=32,updates_per_seed=16,
        milestones=[0,256,512],validation_requests=175,test_methods=list(METHODS),
        test_requests=280,test_repeats=1,control_requests=18,maximum_fresh_requests=2757,
        global_limit_s=25200,acquired_slot_limit_s=19800,search_limit_s=7200,train_cut_s=16200,
        storage_reserve_bytes=8*1024**3,noise=False,actor_steps=800,critic_steps=400,
        bootstrap_lr=.001,ppo_lr=.0003,initial_logstd=-3.,ppo_epochs=6,clip=.2,
        entropy=.001,gradient_norm=1.,residual_scale=RESIDUAL.tolist(),
        scoring_version=oracle.VERSION,route_version=oracle.ROUTE_VERSION,reward_version=oracle.REWARD_VERSION)
    for key,value in expected.items():
        checks.equal(protocol.get(key),value,'Literal bounded training protocol/'+key)


def regenerated_splits(parent_splits, exclusions):
    original=parent_splits['train']
    if len(original)!=35 or len({r['id'] for r in original})!=35:
        raise EvidenceError('Exact historical35 TRAIN contexts required')
    train=[dict(id=c['id'],split='train',bucket_xy_m=list(c['bucket_xy_m']),origin='historical_train') for c in original]
    used=np.asarray(sorted(set(tuple(map(float,xy)) for xy in exclusions)),float)
    if used.ndim!=2 or used.shape[1]!=2 or not np.isfinite(used).all():
        raise EvidenceError('Finite historical exclusion coordinates required')
    result=dict(train=train)
    for role,per_cell,seed in (('train',2,2026110911),('validation',1,2026110912),('test',2,2026110913)):
        rng=np.random.default_rng(seed);fresh=[]
        for iy in range(5):
            for ix in range(7):
                for sample in range(per_cell):
                    for _ in range(10000):
                        xy=np.asarray([.1,-.65])+(np.asarray([1.24,.25])-np.asarray([.1,-.65]))/[7,5]*([ix,iy]+rng.uniform(.05,.95,2))
                        if np.linalg.norm(used-xy,axis=1).min()>=.005:break
                    else:raise EvidenceError('Unable to reproduce disjoint stratified coordinate')
                    fresh.append(dict(id=f'coverage-{role}-r{iy}-c{ix}-n{sample}',split=role,bucket_xy_m=xy.tolist(),
                                      grid_cell=[ix,iy],origin='fresh_stratified'))
                    used=np.vstack((used,xy))
        if role=='train':result['train'].extend(fresh)
        else:result[role]=fresh
    return result


def audit_inventory(directory,source,history_root,parent_splits,spec,checks):
    inventory=read(directory/'historical-coordinate-bindings.json')
    checks.equal(sha(directory/'historical-coordinate-bindings.json'),spec['historical_coordinate_bindings_sha256'],'Frozen full historical coordinate inventory')
    coordinates=set();names={'bucket_xy_m','bucket_xy','exclusions','excluded','coordinates','excluded_coordinates'}
    def collect(item,enabled=False):
        if enabled and isinstance(item,(list,tuple)) and len(item)==2 and all(type(v) in (int,float) and math.isfinite(v) for v in item):
            coordinates.add(tuple(map(float,item)));return
        if isinstance(item,dict):
            for key,value in item.items():collect(value,key in names)
        elif isinstance(item,list):
            for value in item:collect(value,enabled)
    for name,expected in inventory['files_sha256'].items():
        prefix,relative=name.split('/',1)
        checks.require(prefix in ('output','baselines'),'Known inventory root')
        path=contained(history_root if prefix=='output' else source/'baselines',relative)
        checks.require(not path.resolve().is_relative_to(directory.resolve()),'New study cannot enter historical inventory')
        checks.equal(sha(path),expected,'Historical coordinate evidence unchanged/'+name)
        if path.suffix=='.jsonl':
            for item in common.lines(path):collect(item)
        else:collect(read(path),'exclusion' in path.name)
    checks.equal(inventory['coordinates'],[list(x) for x in sorted(coordinates)],'All inventoried coordinates, irrespective of outcome')
    coordinates.update(tuple(c['bucket_xy_m']) for c in parent_splits['train'])
    excluded=[list(x) for x in sorted(coordinates)]
    checks.equal(read(directory/'exclusions.json'),excluded,'Complete pre-split exclusion set')
    checks.equal(spec['exclusions'],excluded,'Contract binds historical exclusions')
    expected=regenerated_splits(parent_splits,excluded)
    checks.equal(read(directory/'splits.json'),expected,'Seeded105/35/70 stratified split regenerated')
    checks.equal(spec['splits'],expected,'Whole split frozen before outcomes')
    return dict(files=len(inventory['files_sha256']),historical_coordinates=len(excluded),new_train=70,validation=35,test=70)


def authenticate_parent(directory,audit,public,checks,kind):
    directory,audit=Path(directory),Path(audit)
    if kind=='current':
        p=public['provenance'];report_sha=p['report_sha256'];proof_sha=p['independent_audit_sha256']
        version='sircus-table-assisted-training-v1';proof_version='sircus-table-assisted-training-audit-v1';count=823442
    else:
        p=public['audit'];report_sha=p['report_sha256'];proof_sha=p['proof_sha256']
        version='sircus-case00-tilt-gap-study-v1';proof_version='sircus-case00-tilt-gap-audit-v1';count=27547
    checks.equal(sha(directory/'report.json'),report_sha,kind+'/published report bytes')
    checks.equal(sha(audit),proof_sha,kind+'/published full audit bytes')
    report,spec,proof,manifest=map(read,(directory/'report.json',directory/'contract.json',audit,directory/'evidence-manifest.json'))
    checks.equal((report['version'],report['status'],proof['version'],proof['status'],proof['full_checks_verified'],proof['failures']),
                 (version,'completed',proof_version,'passed',True,[]),kind+'/completed parent')
    checks.equal((proof['checks_count'],len(proof['checks'])),(count,count),kind+'/entire audit retained')
    checks.require(all(r['passed'] is True for r in proof['checks']),kind+'/all independent checks passed')
    unsigned=dict(spec);fp=unsigned.pop('fingerprint')
    checks.equal((digest(unsigned),report['contract_fingerprint'],proof['contract_fingerprint']),(fp,fp,fp),kind+'/exact contract')
    checks.equal(proof['report_sha256'],report_sha,kind+'/audit binds report')
    checks.equal(proof['evidence_manifest_sha256'],sha(directory/'evidence-manifest.json'),kind+'/audit binds manifest')
    checks.equal(report['evidence_sha256'],manifest,kind+'/report binds manifest')
    if kind=='current':
        checks.equal(sha(directory/'contract.json'),p['contract_file_sha256'],kind+'/published contract')
        checks.equal(sha(directory/'evidence-manifest.json'),p['evidence_manifest_sha256'],kind+'/published manifest')
    def bound(name):
        path=contained(directory,name);checks.equal(sha(path),manifest[name],kind+'/original artifact/'+name);return path
    bound('contract.json')
    binding=dict(report_sha256=report_sha,audit_sha256=proof_sha,contract_sha256=sha(directory/'contract.json'),
                 manifest_sha256=sha(directory/'evidence-manifest.json'),contract_fingerprint=fp)
    return spec,report,manifest,binding,bound


def audit_inputs(study,source,current,current_audit,case,case_audit,spec,checks):
    parent,previous,manifest,current_binding,bound=authenticate_parent(current,current_audit,
        read(source/'baselines/mjlab-table-assisted-training-20261108/summary.json'),checks,'current')
    case_spec,_,case_manifest,case_binding,case_bound=authenticate_parent(case,case_audit,
        read(source/'baselines/mjlab-case00-tilt-gap-20261109/summary.json'),checks,'case00')
    original=read(bound('prior.json'));controls=read(bound('controls.json'))
    checks.equal(len(original['records']),17,'Original17 geometry recipes unchanged')
    checks.equal(len(controls),3,'Original direct/table/miss controls')
    for name in ('calibration.json','setup-home.json','verification.json'):
        checks.equal(sha(study/name),sha(bound(name)),'Unchanged current calibration/home/'+name)
    checks.equal(read(study/'controls.json'),controls,'Same preselected historical controls')
    checks.equal(sha(study/'controls.json'),spec['controls_sha256'],'Bound controls file')
    for key in ('scene_sha256','asset_sha256'):
        checks.equal(parent['identity'][key],case_spec['identity'][key],'Development geometry matches current/'+key)
    rows=common.lines(case_bound('rows.jsonl'));by_id={r['id']:r for r in rows}
    wanted=('case00-tilt-gap-table-training-test-00-g34','case00-tilt-gap-table-training-test-00-g64')
    additions=[]
    for ident in wanted:
        r=by_id[ident]
        checks.require(captured(r) and not r.get('robot_contact_steps') and r['contact_protocol']['substeps']==64,ident+'/audited development capture')
        checks.equal(sha(case_bound('native/'+r['native_cache_file'])),r['native_cache_sha256'],ident+'/native development evidence')
        additions.append(dict(id='coverage-development-'+ident,source_id=ident,source_namespace='case00-tilt-gap-v1-development',
            source_study='case00-tilt-gap-v1',source_record_sha256=digest(r),values=copy.deepcopy(r['values']),bucket_xy_m=copy.deepcopy(r['bucket_xy_m']),
            q10=copy.deepcopy(r['native_initial_q10']),q14=copy.deepcopy(r['native_initial_q14'])))
    prior=copy.deepcopy(original);prior['records']+=additions
    prior.update(records_sha256=digest(prior['records']),recipes=19,version='sircus-coverage-expanded-geometric-prior-v1',
        development_source_ids=list(wanted),scope='17 original sources plus2 explicitly reclassified Case00 development sources; shared geometry only, no new TEST outcomes')
    checks.equal(read(study/'prior.json'),prior,'Exact common19 geometric sources for every method')
    checks.equal(read(study/'old-prior.json'),original,'Preserved historical17-source prior')
    checks.equal((spec['prior_sha256'],spec['old_prior_sha256']),(digest(prior),digest(original)),'Both prior identities separately bound')
    bindings=spec['source_bindings'];checks.equal(read(study/'source-bindings.json'),bindings,'Frozen source binding sidecar')
    checks.equal(bindings['current'],current_binding,'Exact current lineage')
    checks.equal(bindings['case00'],case_binding,'Exact inspected development lineage')
    checks.equal(bindings['prior_sha256'],digest(prior),'Common prior source binding')
    checks.equal(bindings['development_source_sha256'],{r['source_id']:r['source_record_sha256'] for r in additions},'Exact development rows')
    checks.equal(bindings['current_files_sha256'],{n:manifest[n] for n in ('prior.json','controls.json','calibration.json','setup-home.json','verification.json')},'Immutable current source files')
    for method,key in (('current_bc','bc'),('current_ppo','selected')):
        previous_artifact=previous['artifacts'][key];record=spec['old_artifacts'][method]
        checks.equal(record['path'],'checkpoints/'+method+'.pt',method+'/frozen copy path')
        checks.equal(record['historical_relative_path'],previous_artifact['path'],method+'/original checkpoint path')
        checks.equal(sha(bound(previous_artifact['path'])),record['sha256'],method+'/immutable source weights')
        checks.equal(sha(contained(study,record['path'])),record['sha256'],method+'/exact copy of source weights')
        for field in ('sha256','weight_sha256','samples'):
            checks.equal(record[field],previous_artifact[field],method+'/source artifact identity/'+field)
        checks.require(record['evaluated_with_shared_expanded_prior'] is True,method+'/common prior explicitly disclosed')
        expected_binding=copy.deepcopy(previous_artifact)
        expected_binding.update(path=str(contained(current,previous_artifact['path'])),relative_path=previous_artifact['path'],
                                historical_relative_path=previous_artifact['path'])
        checks.equal(bindings['current_artifacts'][method],expected_binding,method+'/full historical source binding')
    checks.equal(read(study/'old-artifacts.json'),spec['old_artifacts'],'Current model declarations')
    return parent,prior,controls


def search_proposal(context,round_index,best=None):
    deltas=((0,0,0,0,0),(0,0,0,-2,0),(0,0,0,2,0),(0,0,0,0,-.01),(0,0,0,0,.01),
            (0,0,-.025,0,0),(0,0,.025,0,0),(-.02,0,0,0,0),(.02,0,0,0,0),(0,.04,0,0,0))
    if round_index<10:
        return np.asarray(deltas[round_index],float)/RESIDUAL,dict(strategy='prescribed_axis_probe',proposal_index=round_index,source_training_ids=[])
    if round_index not in (10,11):raise EvidenceError('Only12 frozen search rounds')
    center=np.zeros(5) if best is None else np.asarray(best['normalized_residual'],float)
    seed=int(hashlib.sha256(f'{BC_SEED}:{context["id"]}:{round_index}'.encode()).hexdigest()[:16],16)
    r=np.tanh(np.arctanh(np.clip(center,-.999999,.999999))+np.random.default_rng(seed).normal(0,.2,5))
    return r,dict(strategy='train_best_joint_refinement',proposal_index=round_index,source_training_ids=[] if best is None else [best['id']])


def audit_search(rows,contexts,checks):
    checks.require(0<len(rows)<=1260,'Nonempty bounded fresh TRAIN search')
    best={}
    for index,row in enumerate(rows):
        context=contexts[index%105];round_index=index//105;label=row['id']
        checks.equal(label,f'coverage-search-{index:05d}',label+'/literal search identity')
        checks.equal((row['role'],row['phase'],row['method'],row['train_index'],row['round_index']),
            ('train','search','guided_search',index,round_index),label+'/unbroken ordered search prefix')
        checks.equal((row['context_id'],row['bucket_xy_m']),(context['id'],context['bucket_xy_m']),label+'/TRAIN-only scheduled coordinate')
        residual,meta=search_proposal(context,round_index,best.get(context['id']))
        checks.numeric(row['normalized_residual'],residual,label+'/independent prescribed or prior-TRAIN refinement',0)
        checks.equal(row['search_metadata'],meta,label+'/search derivation provenance')
        checks.equal(row['source_training_ids'],meta['source_training_ids'],label+'/same-context earlier labels only')
        checks.numeric(row['values'],np.asarray(row['effective_source_hint']['values'])+residual*RESIDUAL,label+'/exact residual action',1e-12)
        rank=lambda r:(-int(captured(r)),-int(captured(r) and r.get('route_quality')==2),-r['learning_reward'],r['id'])
        if context['id'] not in best or rank(row)<rank(best[context['id']]):best[context['id']]=row


def teachers_from_search(rows):
    """Only fresh TRAIN search labels may enter this new bootstrap."""
    for row in rows:
        if row.get('role') != 'train' or row.get('phase') != 'search':
            raise EvidenceError('Only fresh TRAIN search outcomes may bootstrap')
    return native.best_teachers(rows)


def checkpoint_choice(curves):
    if not curves:
        raise EvidenceError('At least the complete BC validation is required')
    return min(curves, key=lambda p: (-p['validation']['captures'],
        -p['validation']['mean_reward'], p['samples'], p['seed']))


def seeded_schedule(seed):
    rng, result = np.random.default_rng(seed), []
    while len(result) < 512:
        result.extend(rng.permutation(105).tolist())
    return result[:512]


def bootstrap_replay(initial, trained, dataset, teachers, fit, checks, seed):
    """Reproduce all actor/critic gradient steps, with an audit-owned model."""
    import torch
    if not dataset or not teachers or any(r.get('phase') != 'search' or r.get('role') != 'train'
                                          for r in dataset + teachers):
        raise EvidenceError('Nonempty fresh TRAIN search data required')
    model = old.independent_policy(seed)
    checks.equal(set(initial), set(model.state_dict()), 'Literal actor/critic architecture')
    for name, value in model.state_dict().items():
        checks.numeric(common.numpy_state(initial)[name], value.detach().numpy(), 'Seeded initializer/' + name, 0)
    model.load_state_dict(initial)
    x = torch.as_tensor(np.asarray([r['observation'] for r in teachers]), dtype=torch.float32)
    y = torch.as_tensor(np.asarray([r['normalized_residual'] for r in teachers]), dtype=torch.float32)
    xx = torch.as_tensor(np.asarray([r['observation'] for r in dataset]), dtype=torch.float32)
    yy = torch.as_tensor([r['learning_reward'] for r in dataset], dtype=torch.float32)
    checks.require(x.shape == (len(teachers), 9) and y.shape == (len(teachers), 5)
        and xx.shape == (len(dataset), 9) and yy.shape == (len(dataset),)
        and all(torch.isfinite(v).all().item() for v in (x,y,xx,yy)) and bool((y.abs() <= 1).all()),
        'Finite bounded TRAIN-only bootstrap tensors')
    actor, critic = torch.optim.Adam(model.actor.parameters(), lr=.001), torch.optim.Adam(model.critic.parameters(), lr=.001)
    aloss = lambda: ((model.actor(x).tanh() - y)**2).mean()
    closs = lambda: ((model.critic(xx).squeeze(-1) - yy)**2).mean()
    losses = dict(initial_actor_loss=float(aloss().detach()), initial_critic_loss=float(closs().detach()))
    for _ in range(800):
        loss=aloss();actor.zero_grad();loss.backward();actor.step()
    for _ in range(400):
        loss=closs();critic.zero_grad();loss.backward();critic.step()
    losses.update(final_actor_loss=float(aloss().detach()), final_critic_loss=float(closs().detach()))
    for key, expected in losses.items():
        checks.numeric(fit[key], expected, 'Full bootstrap replay/' + key, 2e-6)
    for name, value in model.state_dict().items():
        checks.numeric(common.numpy_state(trained)[name], value.detach().numpy(), 'Bootstrap learned parameter/' + name, 2e-6)
    checks.equal((fit['before'],fit['after']), (common.weights_sha256(initial),common.weights_sha256(trained)), 'Exact bootstrap weight hashes')
    checks.equal((fit['actor_steps'],fit['critic_steps']), (800,400), 'Literal bootstrap update budget')
    checks.equal(fit['actor_teacher_ids'], [r['id'] for r in teachers], 'One captured teacher per TRAIN context')
    checks.equal(fit['critic_row_count'], len(dataset), 'Every fresh search failure/capture trains critic')
    checks.numeric(common.numpy_state(trained)['logstd'], [-3.]*5, 'Bootstrap exploration scale fixed', 0)
    return losses


def artifact(directory, record, fingerprint, checks, label, *, seed, samples=None):
    path = contained(directory, record['path'])
    checks.equal(sha(path), record['sha256'], label + '/checkpoint bytes')
    state = common.load_checkpoint(path, checks, label)
    checks.equal(state['contract_fingerprint'], fingerprint, label + '/same experiment')
    checks.equal((state['seed'],state['method']), (seed,'ppo'), label + '/explicit learner seed and architecture')
    checks.equal(state['samples'], record['samples'], label + '/checkpoint prefix')
    if samples is not None:
        checks.equal(state['samples'], samples, label + '/literal prefix size')
    checks.equal(state['weight_sha256'], record['weight_sha256'], label + '/published weight hash')
    return state


def audit_packet(packet, rows, contexts, record, state, checks):
    role,method,samples,seed = (packet[k] for k in ('role','method','samples','seed'))
    checks.require(method in METHODS + ('ppo',), 'Known frozen model method')
    checks.equal(packet['artifact'], record, 'Prediction packet binds one immutable model')
    checks.require(packet.get('selection_or_learning') is False and packet.get('one_prediction_per_layout') is True,
                   'Direct inference without held-out flight search')
    checks.equal(len(packet['rows']), len(contexts), 'Every held-out layout predicted once')
    checks.equal(len(rows), len(contexts), 'Every frozen prediction attempted once')
    for prediction,row,context in zip(packet['rows'],rows,contexts):
        label=row['id']
        checks.equal(label,f'coverage-{role}-{method}-seed{seed}-{samples:04d}-{context["id"]}',label+'/literal inference ID')
        checks.equal((row['role'],row['phase'],row['method'],row['samples'],row['seed']),
                     (role,role,method,samples,seed), label + '/exact phase and frozen model')
        checks.equal(prediction['context_id'],context['id'],label+'/preselected context')
        checks.equal(prediction['bucket_xy_m'],context['bucket_xy_m'],label+'/preselected coordinate')
        for key,value in prediction.items():
            checks.equal(row.get(key),value,label+'/prediction survives execution/'+key)
        checks.equal(prediction['artifact_sha256'],record['sha256'],label+'/artifact binding')
        checks.equal(row['frozen_artifact_sha256'],record['sha256'],label+'/actual inference artifact')
        checks.require(prediction.get('ball_flight_search_at_inference') is False and prediction.get('selection_or_learning') is False
            and prediction.get('geometry_search_at_inference') is True,label+'/only geometry resolution at inference')
        checks.equal(prediction['source_training_ids'],[],label+'/no demonstration lookup at inference')
        latent=common.network_forward(state['policy'],prediction['observation'])
        residual=np.tanh(latent.astype(float))
        checks.numeric(prediction['latent_mean'],latent,label+'/independent actor output',2e-6)
        checks.numeric(prediction['normalized_residual'],residual,label+'/independent residual',2e-6)
        checks.numeric(prediction['command_values'],np.asarray(prediction['effective_source_hint']['values'])+residual*RESIDUAL,
                       label+'/actual actor request',2e-6)
        checks.equal(row['values'],row['command_values'],label+'/no action substitution')


def audit_training(directory,rows,report,spec,checks):
    import torch
    torch.set_num_threads(2)
    search=[r for r in rows if r['phase']=='search']
    audit_search(search,spec['splits']['train'],checks)
    teachers=teachers_from_search(search)
    checks.require(bool(teachers),'At least one fresh captured TRAIN teacher')
    checks.equal(read(directory/'bootstrap/dataset.json'),search,'Bootstrap critic uses every actual fresh search row only')
    checks.equal(read(directory/'bootstrap/teachers.json'),teachers,'Independent one-capture teacher selection per context')
    summary=dict(requests=len(search),captures=sum(captured(r) for r in search),native_releases=sum(bool(r.get('native_release')) for r in search),
        teacher_count=len(teachers),successful_layouts=len(teachers),layouts_requested=len({r['context_id'] for r in search}),
        source_scope='Fresh TRAIN search outcomes only; no historical or held-out outcomes in new actor/critic fitting')
    checks.equal(report['bootstrap_summary'],summary,'Actual bootstrap exposure and success counts')
    checks.equal(report['search_coverage'],[dict(context_id=c['id'],bucket_xy_m=c['bucket_xy_m'],
        counts=statistics([r for r in search if r['context_id']==c['id']]),
        has_teacher=any(captured(r) for r in search if r['context_id']==c['id'])) for c in spec['splits']['train']],
        'Every TRAIN context remains visible, including unattempted or unsuccessful positions')
    checks.equal(read(directory/'bootstrap/source-bindings.json'),dict(dataset_sha256=sha(directory/'bootstrap/dataset.json'),
        teachers_sha256=sha(directory/'bootstrap/teachers.json'),train_ids=[r['id'] for r in search],summary=summary),'Immutable fresh bootstrap records')
    initial=artifact(directory,report['artifacts']['initial'],spec['fingerprint'],checks,'Initial',seed=BC_SEED,samples=0)
    bc=artifact(directory,report['artifacts']['bc'],spec['fingerprint'],checks,'BC',seed=BC_SEED,samples=0)
    checks.equal(report['artifacts']['initial']['path'],'checkpoints/initial.pt','Seeded initializer path')
    checks.equal(report['artifacts']['bc']['path'],'checkpoints/bc.pt','One shared imitation model')
    checks.require(not initial.get('optimizer') and not bc.get('optimizer'),'No historical PPO optimizer imported')
    fit=bootstrap_replay(initial['policy'],bc['policy'],search,teachers,read(directory/'bootstrap/fit.json'),checks,BC_SEED)
    train=[r for r in rows if r['phase']=='train']
    updates=common.lines(directory/'updates.jsonl') if (directory/'updates.jsonl').exists() else []
    checks.require(len(updates)<=32 and len(updates)%2==0,'Completed interleaved seed-pair update prefix')
    checks.equal(len(train),len(updates)*32,'Every requested PPO shot, including declines, belongs to a completed update')
    previous={seed:bc for seed in SEEDS};rngs={}
    milestones={(BC_SEED,0):dict(record=report['artifacts']['bc'],state=bc,method='bc')}
    for seed in SEEDS:
        checks.equal(read(directory/f'schedule-seed{seed}.json'),seeded_schedule(seed),f'Seed{seed}/balanced deterministic TRAIN schedule')
        torch.manual_seed(seed);rngs[seed]=torch.get_rng_state().clone()
    for position,update in enumerate(updates):
        index=position//2;seed=SEEDS[position%2];offset=index*32;label=f'PPO seed{seed} update{index:02d}'
        cohort=train[position*32:(position+1)*32]
        checks.equal((update['seed'],update['index']),(seed,index),label+'/fixed interleaving without winner-dependent allocation')
        checks.equal(update['train_ids'],[r['id'] for r in cohort],label+'/only its current32 TRAIN outcomes')
        before=artifact(directory,update['behavior'],spec['fingerprint'],checks,label+' before',seed=seed,samples=offset)
        after=artifact(directory,update['after_artifact'],spec['fingerprint'],checks,label+' after',seed=seed,samples=offset+32)
        checks.equal(update['behavior']['path'],f'behavior/seed{seed}-update-{index:04d}.pt',label+'/behavior path')
        checks.equal(update['after_artifact']['path'],f'after/seed{seed}-update-{index:04d}.pt',label+'/after path')
        checks.equal(before['weight_sha256'],previous[seed]['weight_sha256'],label+'/same starting BC or prior update')
        if index==0:checks.require(not before['optimizer']['state'],label+'/fresh Adam moments for each seed')
        else:checks.equal(common.tensor_object_digest(before['optimizer']),common.tensor_object_digest(previous[seed]['optimizer']),label+'/continuous own Adam state')
        checks.equal((update['before'],update['after']),(before['weight_sha256'],after['weight_sha256']),label+'/exact weight chain')
        checks.equal(update['buffer'],f'buffers/seed{seed}-update-{index:04d}.npz',label+'/literal TRAIN buffer')
        buffer_path=contained(directory,update['buffer']);checks.equal(sha(buffer_path),update['buffer_sha256'],label+'/immutable update tensors')
        with np.load(buffer_path,allow_pickle=False) as archive:
            checks.equal(set(archive.files),{'observation','latent','old_logp','reward'},label+'/only actual current TRAIN inputs')
            buffer={key:archive[key].copy() for key in archive.files}
        schedule=seeded_schedule(seed)
        for local,row in enumerate(cohort):
            context=spec['splits']['train'][schedule[offset+local]];ident=f'coverage-train-seed{seed}-{offset+local:05d}'
            checks.equal(row['id'],ident,label+'/literal request ID')
            checks.equal((row['method'],row['role'],row['phase'],row['seed'],row['train_index'],row['samples']),
                ('ppo','train','train',seed,offset+local,offset+local+1),ident+'/actual learner prefix')
            checks.equal((row['context_id'],row['bucket_xy_m']),(context['id'],context['bucket_xy_m']),ident+'/scheduled TRAIN-only context')
            checks.equal(row['source_training_ids'],[],ident+'/no action substitution from teacher')
            checks.equal((row['behavior_checkpoint'],row['behavior_checkpoint_sha256']),(update['behavior']['path'],update['behavior']['sha256']),ident+'/sampling actor binding')
        for key,rowkey in (('observation','observation'),('latent','latent'),('old_logp','old_logp'),('reward','learning_reward')):
            checks.numeric(buffer[key],np.asarray([r[rowkey] for r in cohort],dtype=np.float32),label+'/exact current TRAIN tensor/'+key,0)
        common.check_behavior_likelihood(before['policy'],buffer['observation'],buffer['latent'],buffer['old_logp'],checks,label)
        model=old.independent_policy();model.load_state_dict(before['policy'])
        torch.set_rng_state(rngs[seed])
        with torch.no_grad():
            distribution=torch.distributions.Normal(model.actor(torch.tensor(buffer['observation'])),model.logstd.exp())
            expected_sample=distribution.sample();expected_logp=distribution.log_prob(expected_sample).sum(-1)
        rngs[seed]=torch.get_rng_state().clone()
        checks.numeric(buffer['latent'],expected_sample.numpy(),label+'/exact seeded Gaussian samples',0)
        checks.numeric(buffer['old_logp'],expected_logp.numpy(),label+'/exact sampled old likelihood',0)
        residual=np.tanh(np.asarray(buffer['latent'],float))
        checks.numeric([r['normalized_residual'] for r in cohort],residual,label+'/bounded sampled residual',1e-12)
        for local,row in enumerate(cohort):
            checks.numeric(row['values'],np.asarray(row['effective_source_hint']['values'])+residual[local]*RESIDUAL,row['id']+'/exact sampled native request',1e-12)
        replayed=common.replay_ppo_update(before['policy'],before['optimizer'],after['policy'],
            dict(obs=buffer['observation'],latent=buffer['latent'],old_logp=buffer['old_logp'],reward=buffer['reward']),checks,label)
        checks.equal(len(update['epochs']),6,label+'/six PPO epochs')
        checks.numeric([epoch['loss'] for epoch in update['epochs']],replayed['losses'],label+'/independent PPO losses',2e-6)
        checks.require(before['weight_sha256']!=after['weight_sha256'],label+'/actual learned change')
        for state in after['optimizer']['state'].values():checks.numeric(float(state['step']),6*(index+1),label+'/Adam accumulated steps',0)
        checks.numeric(common.numpy_state(after['policy'])['logstd'],np.clip(common.numpy_state(after['policy'])['logstd'],-3.,-.7),label+'/bounded exploration scale',0)
        previous[seed]=after
        if offset+32 in (256,512):
            record=report['per_seed'][str(seed)]['artifacts'][str(offset+32)]
            checks.equal(record['path'],f'checkpoints/seed{seed}-milestone-{offset+32:04d}.pt',label+'/literal validated checkpoint')
            state=artifact(directory,record,spec['fingerprint'],checks,label+' milestone',seed=seed,samples=offset+32)
            checks.equal(state['weight_sha256'],after['weight_sha256'],label+'/same frozen prefix weights')
            checks.equal(common.tensor_object_digest(state['optimizer']),common.tensor_object_digest(after['optimizer']),label+'/same frozen prefix optimizer')
            milestones[(seed,offset+32)]=dict(record=record,state=state,method='ppo')
    each=len(train)//2
    for seed in SEEDS:
        per=report['per_seed'][str(seed)]
        checks.equal((per['samples'],per['updates']),(each,each//32),f'Seed{seed}/actual completed exposure')
        checks.equal(set(per['artifacts']),{str(n) for n in (256,512) if n<=each},f'Seed{seed}/only completed milestones')
    checks.equal((report['train_requests'],report['updates']),(len(train),len(updates)),'Actual PPO request/update accounting')
    return milestones,fit,summary


def audit_predictions(directory,rows,report,spec,milestones,checks):
    curves=[]
    for (seed,samples),member in milestones.items():
        method=member['method'];packet=read(directory/f'predictions/validation-{method}-seed{seed}-{samples:04d}.json')
        cohort=[r for r in rows if r['role']=='validation' and r['seed']==seed and r['samples']==samples]
        checks.equal((packet['role'],packet['method'],packet['seed'],packet['samples']),('validation',method,seed,samples),'Only prescheduled validation milestones')
        audit_packet(packet,cohort,spec['splits']['validation'],member['record'],member['state'],checks)
        curves.append(dict(method=method,seed=seed,samples=samples,validation=statistics(cohort),artifact=member['record']))
    checks.equal(report['curves'],curves,'Every validation outcome retained in chronological order')
    best=checkpoint_choice(curves);selected=best['artifact']
    checks.equal((report['selected'],report['artifacts']['selected']),(selected,selected),'Selection uses validation only; BC can win')
    checks.equal(read(directory/'selected.json'),dict(selected=selected,method=best['method'],validation=best['validation'],
        rule='Completed validation captures, mean reward, earliest samples, seed; freeze before TEST'),'Pretest selection receipt')
    plan=read(directory/'test-plan.json')
    checks.equal(plan['selected'],selected,'Entire TEST plan binds selected artifact')
    checks.require(plan['frozen_before_outcomes'] is True,'All four TEST prediction sets fixed before first outcome')
    checks.equal([p['method'] for p in plan['methods']],list(METHODS),'Fixed four-method paired test order')
    comparisons=[];positions=[]
    for method,packet in zip(METHODS,plan['methods']):
        if method.startswith('current_'):
            record=spec['old_artifacts'][method];checks.equal(report['artifacts'][method],record,method+'/same source checkpoint declaration')
            state=common.load_checkpoint(contained(directory,record['path']),checks,method)
            checks.equal((state['weight_sha256'],state['samples']),(record['weight_sha256'],record['samples']),method+'/unchanged historical weights and prefix')
        else:
            member=milestones[(BC_SEED,0) if method=='bc' else (best['seed'],best['samples'])];record,state=member['record'],member['state']
        checks.equal((packet['role'],packet['samples'],packet['seed']),('test',record['samples'],record.get('seed')),method+'/prescribed frozen prefix')
        checks.equal(read(directory/f'predictions/test-{method}-seed{record.get("seed")}-{record["samples"]:04d}.json'),packet,method+'/same frozen packet')
        cohort=[r for r in rows if r['role']=='test' and r['method']==method]
        audit_packet(packet,cohort,spec['splits']['test'],record,state,checks)
        for row in cohort:checks.equal(row['test_plan_sha256'],sha(directory/'test-plan.json'),row['id']+'/full TEST plan bound before this native request')
        comparisons.append(dict(method=method,counts=statistics(cohort),samples=record['samples'],seed=record.get('seed'),artifact_sha256=record['sha256']))
        for row in cohort:
            positions.append(dict(context_id=row['context_id'],bucket_xy_m=row['bucket_xy_m'],method=method,values=row['values'],
                board_bucket_distance_m=float(np.linalg.norm(np.asarray(row['values'][:2])-row['bucket_xy_m'])),capture=captured(row),
                clean=captured(row) and row.get('route_quality')==2,direct_capture=bool(row.get('direct_capture')),
                table_assisted_capture=bool(row.get('table_assisted_capture')),native_release=row['native_release'],
                invalid=row.get('invalid'),invalid_physics=row.get('invalid_physics'),record_id=row['id']))
    checks.equal(report['test_comparison'],comparisons,'Complete paired70-position metrics')
    checks.equal(report['test_positions'],positions,'Authentic per-position UI evidence')
    return selected,comparisons


def audit_sequence(rows,report,checks):
    search=[r for r in rows if r['phase']=='search'];train=[r for r in rows if r['phase']=='train']
    checks.require(0<len(search)<=1260 and (len(search)%105) in (0,32,64,96),'Search stops only at a completed prescribed batch')
    checks.require(len(train)<=1024 and len(train)%64==0,'PPO stops only at a completed seed pair')
    each=len(train)//2
    expected=[('controls_pre',role,None,None) for role in CONTROL_ROLES for _ in range(3)]
    expected += [('search','train',None,None)]*len(search)
    expected += [('validation','validation',BC_SEED,0)]*35
    for offset in range(0,each,32):
        for seed in SEEDS:
            expected += [('train','train',seed,None)]*32
            if offset+32 in (256,512):expected += [('validation','validation',seed,offset+32)]*35
    expected += [('test','test',None,None)]*280
    expected += [('controls_post',role,None,None) for role in CONTROL_ROLES for _ in range(3)]
    actual=[(r['phase'],r['role'],r.get('seed') if r['phase'] in ('validation','train') else None,
             r.get('samples') if r['phase']=='validation' else None) for r in rows]
    checks.equal(actual,expected,'Complete ordered search/TRAIN/validation/TEST chronology')
    checks.require(len(rows)<=2757,'All requested native and rejected actions within cap')
    checks.equal(len({r['id'] for r in rows}),len(rows),'Unique request identities')
    checks.equal([r['method'] for r in rows if r['phase']=='test'],[m for m in METHODS for _ in range(70)],'All four paired70 tests complete')
    checks.equal((report['search_requests'],report['search_completed_rounds'],report['search_partial_round_requests']),
                 (len(search),len(search)//105,len(search)%105),'True completed search rounds and partial prefix')
    full=len(search)==1260 and each==512
    checks.equal(report['training_budget_complete'],full,'No partial training reported as full budget')
    reasons=report['budget_reasons']
    checks.equal([r['stage'] for r in reasons],(['search'] if len(search)<1260 else [])+(['train'] if each<512 else []),'Only prespecified deadline truncations')
    for name,value in dict(worker_limit_s=19800,training_cut_s=16200,search_limit_s=7200).items():
        checks.equal(report['deadlines'][name],value,'Exact frozen wall-time limit/'+name)
    for reason in reasons:
        if reason['stage']=='search':
            checks.equal(reason['reason'],'search_or_training_deadline','Search truncation has literal reason')
            checks.equal((reason['round_index'],reason['completed_in_round']),(len(search)//105,len(search)%105),'Search stop records its actual next boundary')
            checks.require(reason['search_elapsed_s']>=7200 or reason['elapsed_s']>=16200,'Search cap actually reached before truncation')
            checks.numeric(reason['search_elapsed_s'],reason['elapsed_s']-report['search_started_elapsed_s'],'Search cap uses measured elapsed time',.05)
        else:
            checks.equal(reason['reason'],'training_deadline_reserve_test','PPO truncation has literal reason')
            checks.equal(reason['samples_per_seed'],each,'PPO stop at actual symmetric prefix')
            checks.require(reason['elapsed_s']>=16200,'PPO cap actually reached before truncation')
    checks.require(0<=report['search_started_elapsed_s']<=report['elapsed_s']<19800,'Measured complete worker wall time within cap')
    return dict(search_requests=len(search),ppo_requests=len(train),ppo_requests_per_seed=each,
                validation_requests=sum(r['phase']=='validation' for r in rows),test_requests=280,control_requests=18,
                total_requests=len(rows),maximum_requests=2757,full_training_budget=full)


def audit_budget(directory,report,checks):
    budget_path=directory.with_name(directory.name+'-budget.json')
    pipeline_path=directory.with_suffix('.pipeline.json')
    budget,pipeline=read(budget_path),read(pipeline_path)
    for key,value in dict(limit_s=19800,training_cut_s=16200,search_limit_s=7200,audit_reserve_s=3600).items():
        checks.equal(budget[key],value,'Launch budget cap/'+key)
    checks.equal(pipeline['global_limit_s'],25200,'Absolute seven-hour pipeline cap')
    checks.numeric(pipeline['global_deadline_unix_s']-pipeline['started_unix_s'],25200,'Absolute supervisor deadline interval',.05)
    checks.equal(budget['global_deadline_unix_s'],pipeline['global_deadline_unix_s'],'Worker binds supervisor absolute deadline')
    checks.require(pipeline['started_unix_s']<=budget['acquired_unix_s'],'GPU acquired after own pipeline starts')
    remaining=budget['global_deadline_unix_s']-budget['acquired_unix_s']-3600
    checks.numeric(budget['effective_worker_limit_s'],min(19800,remaining),'Worker cap reserves actual CPU audit time after queue',.1)
    checks.require(60<budget['effective_worker_limit_s']<=19800,'Positive acquired worker budget bounded')
    checks.equal(report['deadlines']['effective_worker_limit_s'],budget['effective_worker_limit_s'],'Reported actual worker cap')
    checks.equal(report['deadlines']['global_deadline_unix_s'],budget['global_deadline_unix_s'],'Reported absolute global cap')
    checks.require(report['elapsed_s']<budget['effective_worker_limit_s'],'Completed worker stayed within effective cap')
    return dict(worker_receipt_sha256=sha(budget_path),effective_worker_limit_s=budget['effective_worker_limit_s'],
                global_deadline_unix_s=budget['global_deadline_unix_s'],global_limit_s=25200,audit_reserve_s=3600)


def audit_request_packets(directory,rows,checks):
    packets=[];seen=set();current=None;group=[]
    for row in rows:
        name=row['request_packet']
        if name!=current:
            if group:packets.append((current,group))
            checks.require(name not in seen,'Every request packet occurs once in chronological sequence')
            seen.add(name);current=name;group=[]
        group.append(row)
    if group:packets.append((current,group))
    timings=common.lines(directory/'timing.jsonl')
    checks.equal(len(timings),len(packets),'Every frozen batch has one execution timing')
    end=0.
    for index,((name,group),timing) in enumerate(zip(packets,timings)):
        checks.equal(name,f'requests/batch-{index:05d}.json','Ordered immutable request batches')
        path=contained(directory,name);packet=read(path)
        checks.require(packet['pre_outcome'] is True and 0<len(group)<=32,'Bounded pre-outcome request batch')
        checks.equal(packet['phase'],group[0]['phase'],'Batch phase binding')
        checks.equal(len(packet['rows']),len(group),'Every issued request accounted, including native declines')
        for request,row in zip(packet['rows'],group):
            checks.equal(row['request_packet_sha256'],sha(path),row['id']+'/pre-outcome request hash')
            for key,value in request.items():checks.equal(row.get(key),value,row['id']+'/request field unchanged/'+key)
        checks.equal(timing['phase'],packet['phase'],'Timing phase matches request batch')
        checks.equal(timing['request_packet'],name,'Timing binds actual request batch')
        checks.equal(timing['request_packet_sha256'],sha(path),'Timing binds exact request bytes')
        checks.require(end<=timing['acquired_start_s']<=timing['acquired_end_s']<19800,'Monotone native batch timings within wall budget')
        end=timing['acquired_end_s']
    checks.equal({str(p.relative_to(directory)) for p in (directory/'requests').glob('*.json')},seen,'No missing or extra request packet')
    return dict(batches=len(packets),last_native_end_s=end)


def audit_boundaries(directory,rows,checks):
    """A pre-TEST prefix receipt cannot include any held-out outcome."""
    boundaries=common.lines(directory/'phase-boundaries.jsonl')
    search=sum(r['phase']=='search' for r in rows)
    before_test=next(i for i,r in enumerate(rows) if r['phase']=='test')
    expected=[('controls_pre-complete',9),('search-complete',9+search),('bootstrap-complete',9+search),
              ('test-plan-frozen',before_test),('controls_post-complete',len(rows))]
    checks.equal([(b['phase'],b['rows_count']) for b in boundaries],expected,'Pretraining and pre-TEST evidence boundaries')
    raw=(directory/'rows.jsonl').read_bytes().splitlines(keepends=True)
    checks.equal(len(raw),len(rows),'Exact serialized request accounting')
    last=0.
    for boundary in boundaries:
        checks.equal(boundary['rows_sha256'],hashlib.sha256(b''.join(raw[:boundary['rows_count']])).hexdigest(),
                     boundary['phase']+'/immutable observed prefix at boundary')
        checks.require(last<=boundary['acquired_elapsed_s']<19800,'Monotone phase receipts within worker cap')
        last=boundary['acquired_elapsed_s']
    timings=common.lines(directory/'timing.jsonl')
    first_test=next(t for t in timings if t['phase']=='test')
    checks.require(boundaries[3]['acquired_elapsed_s']<=first_test['acquired_start_s'],'All four prediction packets frozen before first TEST simulation')


def audit_controls(rows,sources,checks):
    result={}
    for phase,key in (('controls_pre','pre'),('controls_post','post')):
        group=[r for r in rows if r['phase']==phase]
        checks.equal(len(group),9,phase+'/exactly nine controls')
        for index,(source,role) in enumerate(zip(sources,CONTROL_ROLES)):
            for repeat in range(3):
                row=group[index*3+repeat];label=f'coverage-{phase}-{role}-r{repeat}'
                checks.equal(row['id'],label,label+'/literal identity')
                checks.equal((row['role'],row['method'],row['repeat']),(role,'control',repeat),label+'/control role')
                checks.equal((row['control_source_id'],row['control_source_sha256']),(source['id'],digest(source)),label+'/original control provenance')
                checks.equal((row['values'],row['bucket_xy_m']),(source['values'],source['bucket_xy_m']),label+'/unchanged control action and layout')
        result[key]=native.control_gate(group)
        checks.equal(result[key]['status'],'passed',phase+'/direct, table-assisted and negative controls all pass')
    return result


def audit(study,source,assets,*,current_study,current_audit,case00_study,case00_audit,reference,history_root=None):
    started=time.monotonic();checks=AuditChecks()
    study,source,assets,current_study,current_audit,case00_study,case00_audit,reference=map(Path,
        (study,source,assets,current_study,current_audit,case00_study,case00_audit,reference))
    history_root=Path(history_root) if history_root else current_study.parent
    manifest=replay.check_manifest(study,checks)
    spec,report=read(study/'contract.json'),read(study/'report.json')
    unsigned=dict(spec);fp=unsigned.pop('fingerprint')
    checks.equal(digest(unsigned),fp,'Full frozen study fingerprint')
    checks.equal((spec['version'],report['version']),(STUDY_VERSION,STUDY_VERSION),'Explicit new study version')
    checks.equal((report['status'],report['phase']),('completed','completed'),'Full TEST and controls completed')
    checks.equal(report['contract_fingerprint'],fp,'Report binds exact experiment')
    checks.equal(spec['evidence_kind'],'native_fixed64_expanded_prior_fresh_train_search_imitation_ppo','Native fixed64 expanded coverage evidence')
    checks.equal(report['evidence_sha256'],manifest,'Report binds every raw artifact')
    checks.equal(report['protocol'],spec['protocol'],'Frozen protocol unchanged')
    protocol_checks(spec['protocol'],checks)
    replay.check_source_files(source,spec['source_sha256'],checks,'Frozen actual producer source')
    checks.equal(sha(Path(__file__)),spec['source_sha256']['sircus/mjlab/coverage_training_verify.py'],'Same frozen independent audit implementation')
    checks.equal(sha(Path(oracle.__file__)),spec['source_sha256']['sircus/mjlab/table_assisted_score_oracle.py'],'Same independent v3 oracle implementation')
    parent,prior,controls=audit_inputs(study,source,current_study,current_audit,case00_study,case00_audit,spec,checks)
    old.audit_runtime_reference(source,reference,spec,parent,assets,checks)
    inventory=audit_inventory(study,source,history_root,parent['splits'],spec,checks)
    calibration=replay.check_calibration(study,spec['identity']['calibration'],checks)
    rows=replay.read_records(study/'rows.jsonl')
    native.check_rows(rows,checks)
    accounting=audit_sequence(rows,report,checks)
    budget=audit_budget(study,report,checks)
    requests=audit_request_packets(study,rows,checks)
    audit_boundaries(study,rows,checks)
    setup=calibrated.check_setup_evidence(study,rows,prior,spec['identity']['setup']['home'],checks)
    old.audit_native_options(study/'native',spec['scene'],calibration,assets,checks)
    for row in rows:
        if row.get('native_release'):
            checks.equal((row['effective_options_file'],row['effective_options_sha256']),
                ('effective-options.json',sha(study/'native/effective-options.json')),row['id']+'/fixed64 native option identity')
    traces=native.rescore_rows(study/'native',rows,spec['scene'],calibration,spec['identity']['trace_schema'],assets,checks)
    milestones,fit,bootstrap_summary=audit_training(study,rows,report,spec,checks)
    selected,comparisons=audit_predictions(study,rows,report,spec,milestones,checks)
    control_gates=audit_controls(rows,controls,checks)
    checks.equal(report['control_gates'],control_gates,'Complete before/after control gates')
    checks.equal(report['counts'],statistics(rows),'Full denominator includes rejected setup requests')
    checks.equal(report['stage_counts'],{p:statistics([r for r in rows if r['phase']==p]) for p in
        ('controls_pre','search','train','validation','test','controls_post')},'All phase metrics reconstructed')
    checks.equal(report['gate'],dict(status='passed',controls_passed=True,complete_test=True,full_training_budget=accounting['full_training_budget'],
        substeps=64,scope='Complete frozen comparison; not128 convergence, guaranteed capture rate or automatic policy promotion'),'Completion is not universal capture qualification')
    return dict(version=VERSION,status='passed',full_checks_verified=True,native_checks_verified=True,checks_count=checks.count,
        checks=checks.entries,failures=[],contract_fingerprint=fp,report_sha256=sha(study/'report.json'),
        evidence_manifest_sha256=sha(study/'evidence-manifest.json'),auditor_sha256=sha(Path(__file__)),
        producer_source_sha256=spec['source_sha256'],requested_trials=len(rows),native_releases=len(traces),
        bootstrap_summary=bootstrap_summary,bootstrap_replay=fit,selected=selected,test_comparison=comparisons,
        control_gates=control_gates,setup_evidence=setup,coordinate_inventory=inventory,request_accounting=accounting,
        request_packets=requests,budget=budget,elapsed_s=time.monotonic()-started,maximum_numeric_error=checks.numeric_error,
        limitations=['Fixed64 held-endpoint simulator evidence only; no driven transit, grasping, perception, noise or physical validation.',
            'One frozen decision per method at70 novel bucket positions; not an all-table or repeatability guarantee.',
            'Current weights are evaluated with a common expanded prior, not their historical geometric bank.',
            'Two inspected Case00 shots are explicitly development geometry; they are not new held-out successes.',
            'Finite IK declines one request, not every possible shot at a bucket; every decline remains in the denominator.'])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('study','source','assets','current-study','current-audit','case00-study','case00-audit','reference','history-root','output'):
        parser.add_argument('--'+key,type=Path,required=True)
    a=parser.parse_args()
    if a.output.exists() or a.output.resolve().is_relative_to(a.study.resolve()):
        raise ValueError('Audit output must be new and outside immutable study')
    try:
        result=audit(a.study,a.source,a.assets,current_study=a.current_study,current_audit=a.current_audit,
            case00_study=a.case00_study,case00_audit=a.case00_audit,reference=a.reference,history_root=a.history_root)
    except Exception as error:
        result=dict(version=VERSION,status='failed',failures=[type(error).__name__+': '+str(error)])
        a.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');raise
    a.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:result[k] for k in ('status','checks_count','requested_trials','native_releases','selected')}))


if __name__=='__main__':main()
