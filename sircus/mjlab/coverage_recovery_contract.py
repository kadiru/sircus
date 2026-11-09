"""Bounded, post-hoc recovery search and actor-only imitation contract."""
from __future__ import annotations
import copy
import json
from pathlib import Path
import numpy as np
from .benchmark_contract import RESIDUAL_SCALE, action_from_residual
from .corrected_contract import sha
from .learning_contract import digest
from .table_training_contract import captured, clean, statistics, collect_exclusions, contained
from .table_assisted_score import reward
from .table_correction_native import META

VERSION='sircus-coverage-recovery-v1'
AUDIT_VERSION='sircus-coverage-recovery-audit-v1'
METHODS=('recovery_bc','parent_bc','parent_ppo','previous_ppo')
TEST_SEED=2026110923
SCOPE=('Former coverage TEST failures are explicitly development data. Bounded five-coordinate residual search, '
       'two fresh confirmations per selected capture, then actor-only imitation from parent BC. '
       'Four frozen models share unchanged19 geometry prior on70 new bucket positions at native64/v3. '
       'Fixed-base blocked layouts are declared skipped, not learned feasible. No PPO, noise,128, learned transit or policy activation.')

def read(path):return json.loads(Path(path).read_text())
def require(value,message):
    if not value:raise ValueError(message)
def rank(row):return (-int(clean(row)),-float(reward(row)),row['id'])

def protocol():
    return dict(substeps=64,methods=list(METHODS),test_seed=TEST_SEED,development_failures=42,fixed_base_skips=3,
        search_layouts=39,proposals_per_layout=16,search_requests=624,repeat_per_selected=2,repeat_requests_max=78,
        control_requests=18,test_layouts=70,test_requests=280,maximum_requests=1000,
        actor_steps=800,critic_steps=0,actor_lr=.001,base_teacher_count=68,noise=False,
        residual_scale=RESIDUAL_SCALE.tolist(),search_limit_s=3600,repeat_cut_s=4500,worker_limit_s=7200,
        global_limit_s=10800,storage_reserve_bytes=8*1024**3,minimum_split_distance_m=.005,
        search_order='Round-robin literal39 failure contexts,16 frozen proposals; bounded32-request batches',
        timing='Stop search at completed batch when acquired elapsed>=3600s, repeats at completed pair when>=4500s; worker hard7200s; full280TEST required',
        confirmation='Best captured discovery per context by clean then reward then ID; two fresh repeats; both must capture to supply a teacher',
        fitting='Parent BC actor-only full-batch800 Adam steps at.001 on68 original plus confirmed recovery teachers; critic/logstd unchanged; zero new teachers copies parent BC bytes',
        test='All280 predictions frozen before first outcome; no ball-flight search at inference',
        control_prior='Exact original17 control joints/proofs; all search and model comparisons use common19',scope=SCOPE)

def load_inputs(a,root):
    directory=Path(a.parent_study);public=read(Path(root)/'baselines/mjlab-coverage-training-20261109/summary.json')
    require(sha(directory/'report.json')==public['report_sha256'],'Published parent report differs')
    require(sha(a.parent_audit)==public['audit']['proof_sha256'],'Corrected parent audit differs')
    report,proof,spec,manifest=(read(p) for p in (directory/'report.json',a.parent_audit,directory/'contract.json',directory/'evidence-manifest.json'))
    require(report['status']=='completed' and proof['status']=='passed' and proof['failures']==[]
        and proof['checks_count']==857507 and len(proof['checks'])==857507
        and all(x.get('passed') is True for x in proof['checks']),'Full corrected parent audit required')
    unsigned=dict(spec);fp=unsigned.pop('fingerprint')
    require(digest(unsigned)==fp==report['contract_fingerprint']==proof['contract_fingerprint'],'Parent contract differs')
    require(report['evidence_sha256']==manifest and proof['evidence_manifest_sha256']==sha(directory/'evidence-manifest.json'),'Parent manifest differs')
    checked={}
    def verified(name):
        path=contained(directory,name)
        if name not in checked:
            require(sha(path)==manifest[name],'Parent evidence changed: '+name);checked[name]=manifest[name]
        return path
    rows=[json.loads(x) for x in verified('rows.jsonl').read_text().splitlines()]
    files={name:manifest[name] for name in ('rows.jsonl','contract.json')}
    out={}
    for key,name in (('prior','prior.json'),('old_prior','old-prior.json'),('controls','controls.json'),
        ('control_replays','control-replays.json'),('calibration','calibration.json'),('home','setup-home.json'),('teachers','bootstrap/teachers.json')):
        out[key]=read(verified(name));files[name]=manifest[name]
    verified('verification.json');files['verification.json']=manifest['verification.json']
    require(len(out['prior']['records'])==19 and len(out['old_prior']['records'])==17 and len(out['teachers'])==68,'Frozen prior/teacher counts differ')
    require(all(r['role']=='train' and r['phase']=='search' and captured(r) for r in out['teachers']),'Original captured TRAIN teachers required')
    current=[r for r in rows if r.get('phase')=='test' and r.get('method')=='selected_ppo']
    require(len(current)==70,'Exact70 parent TEST results required')
    failures=[r for r in current if not captured(r)];blocked=[r for r in failures if r.get('invalid')=='fixed_base_clearance_rejected']
    cases=[r for r in failures if r not in blocked]
    require((len(failures),len(blocked),len(cases))==(42,3,39),'Frozen failure cohort differs')
    for row in [*cases,*out['teachers'],*out['control_replays']]:
        namespace='native-controls' if row in out['control_replays'] else 'native'
        for field,hashfield in (('context_resolution_file','context_resolution_sha256'),('connection_file','connection_sha256')):
            name=namespace+'/'+row[field];verified(name);require(manifest[name]==row[hashfield],'Parent proof field differs')
        if row.get('native_release') and namespace=='native':
            name=namespace+'/'+row['native_cache_file'];verified(name);require(manifest[name]==row['native_cache_sha256'],'Parent native cache differs')
    for name in ('native-controls/effective-options.json','native-controls/native-compiled-geometry.json'):
        verified(name);files[name]=manifest[name]
    artifacts={}
    for method,key in (('parent_bc','bc'),('parent_ppo','selected'),('previous_ppo','current_ppo')):
        info=copy.deepcopy(report['artifacts'][key]);path=verified(info['path']);require(sha(path)==info['sha256'],'Parent checkpoint differs')
        info['source_path']=str(path);artifacts[method]=info;files[info['path']]=info['sha256']
    out.update(parent=spec,parent_report=report,rows=rows,cases=cases,blocked=blocked,artifacts=artifacts,
        bindings=dict(report_sha256=sha(directory/'report.json'),audit_sha256=sha(a.parent_audit),
            contract_sha256=sha(directory/'contract.json'),manifest_sha256=sha(directory/'evidence-manifest.json'),
            files_sha256=dict(files,**checked),case_rows_sha256={r['id']:digest(r) for r in failures},
            original_test_reclassified='All42 inspected parent selected-PPO failures are development; no new TEST labels used'))
    return out

def proposal(source,index,alternates):
    require(type(index) is int and 0<=index<16,'Proposal0..15 required')
    r=np.asarray(source['normalized_residual'],float).copy();lineage=[source['id']]
    require(r.shape==(5,) and np.isfinite(r).all() and (np.abs(r)<1).all(),'Parent residual support differs')
    kind='original_failed_policy'
    if index==1:
        successes=[x for x in alternates if x['context_id']==source['context_id'] and captured(x)]
        if successes:
            best=min(successes,key=rank)
            require(best['effective_source_hint']==source['effective_source_hint'] and best['source_hint']==source['source_hint']
                and best['observation']==source['observation'],'Alternative residual must share exact original anchor and observation')
            r=np.asarray(best['normalized_residual'],float).copy();lineage=[best['id']];kind='observed_other_policy_capture'
        else:r=np.zeros(5);kind='nominal_geometry_anchor'
    elif index>=2 and not source['native_release']:
        kind='setup_support_probe'
        if index<=10:r[2],r[3]=[(h,t) for h in (-.8,0.,.8) for t in (-.75,0.,.75)][index-2]
        elif index in (11,12):r[4]=(.5,.9)[index-11]
        elif index==13:r[0]=0.
        elif index==14:r[1]=0.
        else:r[2],r[3]=.5,-.5
    elif index>=2:
        kind='flight_support_probe'
        if index<=7:r[4]=(-.9,-.6,-.3,.3,.6,.9)[index-2]
        elif index<=11:r[3]=(-.75,-.375,.375,.75)[index-8]
        elif index<=13:r[2]=(-.75,.75)[index-12]
        elif index==14:r[:2]=0.
        else:r[2],r[3],r[4]=0.,-.5,.75
    require(np.isfinite(r).all() and (np.abs(r)<1).all(),'Search must remain strictly inside policy support')
    return r,dict(index=index,kind=kind,parent_source_ids=lineage)

def make_plan(cases,parent_rows):
    alternatives=[r for r in parent_rows if r.get('phase')=='test']
    requests=[]
    for index in range(16):
        for source in cases:
            residual,derivation=proposal(source,index,alternatives)
            values=action_from_residual(source['effective_source_hint']['values'],residual).tolist()
            meta={k:copy.deepcopy(source[k]) for k in META}
            requests.append(dict(meta,id=f'recovery-search-r{index:02d}-{source["context_id"]}',phase='search',role='train',method='recovery_search',
                context_id=source['context_id'],bucket_xy_m=source['bucket_xy_m'],values=values,command_values=values,
                normalized_residual=residual.tolist(),proposal_index=index,derivation=derivation,
                parent_failure_id=source['id'],parent_failure_sha256=digest(source),original_role='test',
                development_reclassified=True,perturbation_mixed_units=[0.]*5))
    require(len(requests)==624 and len({r['id'] for r in requests})==624,'Exact bounded624 proposal plan required')
    return requests

def repeat_plan(rows,cases):
    selected=[];requests=[]
    for case in cases:
        choices=[r for r in rows if r['context_id']==case['context_id'] and captured(r)]
        if not choices:continue
        source=min(choices,key=rank);selected.append(dict(context_id=source['context_id'],discovery_id=source['id'],discovery_sha256=digest(source)))
        for repeat in range(2):
            row=copy.deepcopy(source)
            # Request evidence contains only pre-outcome decision fields.
            row={k:copy.deepcopy(source[k]) for k in (*META,'context_id','bucket_xy_m','values','command_values','normalized_residual','parent_failure_id','parent_failure_sha256','original_role','development_reclassified')}
            row.update(id=f'recovery-repeat-{source["context_id"]}-r{repeat}',phase='repeat',role='train',method='recovery_repeat',
                repeat=repeat,source_discovery_id=source['id'],source_discovery_sha256=digest(source),perturbation_mixed_units=[0.]*5)
            requests.append(row)
    return selected,requests

def confirmed_teachers(search,repeats,selected):
    byid={r['id']:r for r in search};teachers=[]
    for item in selected:
        group=[r for r in repeats if r['source_discovery_id']==item['discovery_id']]
        if len(group)==2 and all(captured(r) for r in group):teachers.append(copy.deepcopy(byid[item['discovery_id']]))
    return sorted(teachers,key=lambda r:r['context_id'])

def test_split(excluded):
    points=np.asarray(sorted(set(tuple(map(float,p)) for p in excluded)),float)
    require(points.ndim==2 and points.shape[1]==2 and len(points),'Historical coordinate exclusions required')
    rng=np.random.default_rng(TEST_SEED);low=np.asarray([.1,-.65]);size=(np.asarray([1.24,.25])-low)/[7,5];result=[]
    for iy in range(5):
        for ix in range(7):
            for n in range(2):
                for _ in range(10000):
                    xy=low+size*([ix,iy]+rng.uniform(.05,.95,2))
                    if np.linalg.norm(points-xy,axis=1).min()>=.005:break
                else:raise ValueError('Disjoint table cell sampling exhausted')
                result.append(dict(id=f'recovery-test-r{iy}-c{ix}-n{n}',split='test',bucket_xy_m=xy.tolist(),grid_cell=[ix,iy]))
                points=np.vstack((points,xy))
    return result
