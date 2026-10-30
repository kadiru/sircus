"""Train and benchmark the separate front-bucket profile; no robot or service I/O."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import copy
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import numpy as np
from sircus.studies import front_study as study
from sircus.studies.profiles import front_scene,profile_metadata,PROFILE_ID

MODEL_CANDIDATES=((1,.1),(2,1.),(2,100.))
LIMITS=['Synthetic reduced-order dynamics and provisional geometry/contact parameters',
        'Narrow front-bucket training distribution; challenges explicitly label extrapolation',
        'Endpoint IK is not native trajectory or collision-path validation',
        'Limited-budget oracle is search, not learned inference or an optimality bound',
        'No physical, camera, gripper or board-suction validation',
        'Training seeds share held-out layouts; uncertainty and seed variability are reported separately']


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n')


def progress(phase,**kwargs):print(json.dumps({'phase':phase,**kwargs},allow_nan=False),flush=True)


def compact(row):
    value=copy.deepcopy(row);value.pop('selected_scene',None)
    if value.get('result'):value['result'].pop('frames',None)
    return value


def _teacher_task(args):
    episode,budget,role=args
    rows,best=study.search_episode(episode,budget,role=role)
    return [compact(r) for r in rows],compact(best) if best else None


def _evaluate_task(args):
    episode,action,noise,allow_ood=args
    row=study.evaluate_action(episode['scene'],action,episode['rollout_seed'],noise=noise,allow_ood=allow_ood)
    row.update(scene_id=episode['scene_id'],layout_sha256=episode['layout_sha256'],split=episode['split'])
    return compact(row)


def map_tasks(function,items,workers=2):
    if workers==1:
        yield from map(function,items)
    else:
        # Spawn avoids inheriting a dashboard's threads or sockets. CPU BLAS limits
        # are set by the launcher before imports; never exceed two study workers.
        with ProcessPoolExecutor(max_workers=min(2,workers),mp_context=multiprocessing.get_context('spawn')) as pool:
            yield from pool.map(function,items,chunksize=1)


def evaluate_model(model,episodes,workers=2,*,noise=None,allow_ood=False):
    tasks=[(e,study.infer_model(model,e['scene']),noise,allow_ood) for e in episodes]
    return list(map_tasks(_evaluate_task,tasks,workers))


def selection(rows):
    return (sum(r['result']['success'] is True for r in rows),sum(float(r['result']['shaping_reward']) for r in rows))


def select_model(labels,validation,workers):
    x=[e['observation'] for e in labels];y=[[e['selected']['action'][n] for n in study.ACTION_NAMES] for e in labels]
    candidates=[]
    for degree,alpha in MODEL_CANDIDATES:
        model=study.fit_model(x,y,alpha,degree);rows=evaluate_model(model,validation,workers)
        candidates.append({'model':model,'rows':rows,'summary':study.summarize(rows),'degree':degree,'regularization':alpha})
    return max(candidates,key=lambda c:selection(c['rows'])),candidates


def teachers(episodes,budget,output,workers,*,context=None):
    labels=[]
    with (output/'training-candidates.jsonl').open('w') as stream:
        for index,(candidates,best) in enumerate(map_tasks(_teacher_task,[(e,budget,'training_teacher') for e in episodes],workers)):
            for row in candidates:stream.write(json.dumps(row,sort_keys=True,allow_nan=False)+'\n')
            stream.flush();episode=episodes[index]
            labels.append({'scene_id':episode['scene_id'],'layout_sha256':episode['layout_sha256'],
                'observation':episode['observation'],'selected':best,'label_available':best is not None,
                'teacher_success':bool(best and best['result']['success'])})
            progress('teacher',completed=index+1,total=len(episodes),label_available=best is not None,
                     captured=bool(best and best['result']['success']),**(context or {}))
    write(output/'training-labels.json',labels)
    return labels


def frozen_evaluation(checkpoint,episodes,workers):
    # Validate and freeze before iterating any held-out episode. No optimization here.
    study.validate_checkpoint(checkpoint)
    before=study.digest(checkpoint)
    rows=evaluate_model(checkpoint['model'],episodes,workers)
    for row in rows:row.update(role='frozen_policy',checkpoint_id=checkpoint['checkpoint_id'])
    if study.digest(checkpoint)!=before:raise RuntimeError('Checkpoint changed during held-out inference')
    return rows


def fixed_evaluation(episodes,workers):
    rows=list(map_tasks(_evaluate_task,[(e,study.FIXED_ACTION,None,False) for e in episodes],workers))
    for row in rows:row['role']='fixed_baseline'
    return rows


def oracle_evaluation(episodes,budget,workers):
    rows=[];candidates=[]
    for proposed,best in map_tasks(_teacher_task,[(e,budget,'held_out_oracle') for e in episodes],workers):
        candidates.extend(proposed);rows.append(best if best is not None else proposed[0])
    return rows,candidates


def _checkpoint(model,labels,manifest,selected):
    return study.checkpoint(model,[r['scene_id'] for r in labels],manifest,
        {'selection':'capture count then total shaping on validation only',
         'degree':selected['degree'],'regularization':selected['regularization'],
         'candidates':[{'degree':d,'regularization':a} for d,a in MODEL_CANDIDATES],
         'validation_summary':selected['summary'],'test_used_for_selection':False})


def run(output,*,seed=20261030,train_scenes=24,validation_scenes=6,test_scenes=8,
        teacher_budget=12,oracle_budget=4,workers=2):
    for n,low,high in [(train_scenes,4,256),(validation_scenes,1,100),(test_scenes,1,200),(teacher_budget,1,128),(oracle_budget,1,128)]:study.integer(n,low,high)
    study.integer(workers,1,2);start=time.monotonic();output=Path(output);output.mkdir(parents=True,exist_ok=False)
    used=set();manifest={key:study.sample_layouts(seed,key,count,used=used) for key,count in [('train',train_scenes),('validation',validation_scenes),('test',test_scenes)]}
    write(output/'episodes.json',manifest);frozen_manifest=(output/'episodes.json').read_bytes()
    write(output/'experiment.json',{'schema':study.STUDY_SCHEMA,'profile_id':PROFILE_ID,'seed':seed,
        'teacher_budget':teacher_budget,'oracle_budget':oracle_budget,'model_candidates':MODEL_CANDIDATES,
        'fixed_baseline':study.FIXED_ACTION,'manifest_sha256':study.digest(manifest),'contract':study.study_contract(),
        'data_origin':'synthetic','physically_verified':False,'limits':LIMITS})
    all_labels=teachers(manifest['train'],teacher_budget,output,workers)
    labels=[r for r in all_labels if r['label_available']]
    if len(labels)<4:raise ValueError('Insufficient valid training labels; attempts preserved')
    chosen,candidates=select_model(labels,manifest['validation'],workers);write(output/'validation.json',candidates)
    value=_checkpoint(chosen['model'],labels,manifest,chosen);write(output/'checkpoint.json',value)
    checkpoint_bytes=(output/'checkpoint.json').read_bytes();loaded=json.loads(checkpoint_bytes)
    progress('validation',degree=chosen['degree'],regularization=chosen['regularization'],**chosen['summary'])
    predicted=frozen_evaluation(loaded,manifest['test'],workers)
    fixed=fixed_evaluation(manifest['test'],workers)
    oracle,oracle_candidates=oracle_evaluation(manifest['test'],oracle_budget,workers)
    write(output/'test.json',{'frozen_policy':predicted,'fixed_baseline':fixed,'held_out_search_oracle':oracle,'oracle_candidates':oracle_candidates})
    first=manifest['test'][0];write(output/'preview-scene.json',first)
    write(output/'preview.json',study.preview(loaded,first['scene'],first['rollout_seed']))
    if checkpoint_bytes!=(output/'checkpoint.json').read_bytes() or frozen_manifest!=(output/'episodes.json').read_bytes():raise RuntimeError('Frozen evidence changed')
    summary={'schema':study.STUDY_SCHEMA,'profile_id':PROFILE_ID,'status':'complete',
        'checkpoint_id':value['checkpoint_id'],'checkpoint_file_sha256':hashlib.sha256(checkpoint_bytes).hexdigest(),
        'manifest_sha256':study.digest(manifest),'scoring_version':study.SCORING_VERSION,'physics_version':study.PHYSICS_VERSION,
        'train_scenes':train_scenes,'validation_scenes':validation_scenes,'test_scenes':test_scenes,
        'teacher_budget':teacher_budget,'oracle_budget_per_scene':oracle_budget,'teacher_labels':len(labels),
        'teacher_failed_scenes':train_scenes-len(labels),'teacher_captures':sum(r['teacher_success'] for r in labels),
        'regularization':chosen['regularization'],'model_degree':chosen['degree'],'validation':chosen['summary'],
        'test':{'frozen_policy':study.summarize(predicted),'fixed_baseline':study.summarize(fixed),'held_out_search_oracle':study.summarize(oracle)},
        'checkpoint_unchanged_after_test':True,'split_manifest_unchanged_after_test':True,'elapsed_seconds':time.monotonic()-start,
        'preview_seed':first['rollout_seed'],'limits':LIMITS,'data_origin':'synthetic','physically_verified':False,'native_paths_checked':False}
    write(output/'summary.json',summary);progress('complete',**summary)
    return value,summary


def challenge_manifest(seed=20261030,layouts=24):
    rng=np.random.default_rng(study.stream_seed(seed,'challenges'))
    groups=[]
    specifications=[('distance',d) for d in (60.,100.,160.,220.,280.,340.,400.,450.)]
    specifications += [('offset',x) for x in (-200.,-120.,-60.,0.,60.,120.,200.)]
    for kind,value in specifications:
        rows=[]
        for index in range(layouts):
            scene=front_scene();scene['board']['center_mm']=[float(rng.uniform(580,620)),-150.,float(rng.uniform(335,370))]
            distance=value if kind=='distance' else 250.
            lateral=float(rng.uniform(-25,25)) if kind=='distance' else value
            scene['bin']['center_xy_mm']=[scene['board']['center_mm'][0]+lateral,-150.-distance]
            scene_id=f'challenge-{kind}-{value:g}-{index:03d}';scene['scene_id']=scene_id
            rows.append({'scene_id':scene_id,'scene':scene,'split':'challenge','layout_sha256':study.layout_hash(scene),
                'rollout_seed':study.stream_seed(seed,scene_id),'observation':study.observation(scene),
                'distance_mm':distance,'lateral_offset_mm':lateral})
        groups.append({'kind':kind,'value':value,'units':'mm','label':f'{kind} {value:g} mm','episodes':rows})
    noise_base=study.sample_layouts(seed,'challenge-noise-layout',layouts)
    for xy,gap in ((0.,0.),(2.,3.),(5.,8.),(10.,15.),(20.,30.),(30.,45.)):
        episodes=[]
        for row in noise_base:
            for replicate in range(3):
                episode=copy.deepcopy(row);episode['scene_id']=row['scene_id']+f'-r{replicate}';episode['split']='challenge'
                episode['replicate']=replicate;episode['rollout_seed']=study.stream_seed(seed,row['scene_id']+'/noise',replicate)
                episodes.append(episode)
        groups.append({'kind':'release_noise','value':xy,'units':'mm XY','noise_xy_mm':xy,'noise_gap_mm':gap,
            'noise':{'xy_mm':xy,'gap_mm':gap},'label':f'Release noise ±{xy:g} XY / ±{gap:g} gap mm','episodes':episodes})
    return groups


def layout_interval(means,seed):
    """Conservative layout-level interval; repeated noise/seed trials are clustered.

    Enveloping a layout bootstrap with a Wilson effective-layout bound avoids
    claiming certainty when a finite sample contains all successes or failures.
    Fractional layout means never increase the independent sample count.
    """
    means=np.asarray(means,float)
    if len(means)==0:return [0.,1.]
    rng=np.random.default_rng(seed)
    samples=means[rng.integers(0,len(means),size=(3000,len(means)))].mean(axis=1)
    bootstrap=np.quantile(samples,[.025,.975]);finite_sample=study.wilson(float(means.sum()),len(means))
    return [min(float(bootstrap[0]),finite_sample[0]),max(float(bootstrap[1]),finite_sample[1])]


def cluster_summary(rows,seed):
    result=study.summarize(rows);groups={}
    for row in rows:groups.setdefault(row['layout_sha256'],[]).append(float(row['result']['success']))
    means=np.asarray([np.mean(x) for x in groups.values()])
    result['ci95']=layout_interval(means,seed)
    result['layouts']=len(means);result['ci95_method']='layout bootstrap envelope with Wilson effective-layout bound; noise repeats stay clustered'
    return result


def paired_gain(rows,fixed,seed):
    values=np.asarray([float(a['result']['success'])-float(b['result']['success']) for a,b in zip(rows,fixed)])
    rng=np.random.default_rng(seed);samples=values[rng.integers(0,len(values),size=(3000,len(values)))].mean(axis=1)
    return {'mean_difference':float(values.mean()),'ci95':[float(x) for x in np.quantile(samples,[.025,.975])],
            'method':'paired layout bootstrap; same layouts and release seeds'}


def replay(checkpoint,episode,label,role='frozen_policy',noise=None):
    row=study.preview(checkpoint,episode['scene'],episode['rollout_seed'],allow_ood=episode['split']=='challenge',noise=noise)
    result={'id':episode['scene_id'],'label':label,'role':role,'split':episode['split'],'scene_id':episode['scene_id'],
        'profile_id':PROFILE_ID,'scene':row['selected_scene'],'action':row['action'],'result':row['result'],
        'checkpoint_id':checkpoint['checkpoint_id'],'inference':checkpoint['inference'],
        'provenance':'synthetic frozen checkpoint; no per-scene optimization; endpoint IK only',
        'data_origin':'synthetic','physically_verified':False,'native_paths_checked':False}
    if row['feasibility'] and row['feasibility']['arm_reachable'] and row['feasibility']['board_arm_reachable']:
        from two_arm_kinematics import forward_world
        result['arm14_boxes']=forward_world(row['selected_scene'],14,row['feasibility']['board_check']['joints_deg'])['boxes']
        arm=next(r for r in row['feasibility']['arm_checks'] if r['step']=='release')
        result['arm10_release_boxes']=forward_world(row['selected_scene'],10,arm['joints_deg'])['boxes']
    result['result'].setdefault('frames',[])
    return result


def curves(output,*,seeds=(20261030,20261031,20261032),sizes=(8,24,64,128),validation_count=16,
           test_count=48,teacher_budget=16,oracle_budget=16,workers=2,challenge_layouts=24):
    if not 1<=len(seeds)<=3 or len(set(seeds))!=len(seeds):raise ValueError('One to three unique training seeds required')
    for seed in seeds:study.integer(seed,0,2**31-1)
    if not sizes or tuple(sorted(set(sizes)))!=tuple(sizes):raise ValueError('Increasing unique nested sizes required')
    for n in sizes:study.integer(n,4,256)
    for n,low,high in [(validation_count,1,64),(test_count,1,128),(teacher_budget,1,64),(oracle_budget,1,64),(challenge_layouts,4,32),(workers,1,2)]:study.integer(n,low,high)
    start=time.monotonic();output=Path(output);output.mkdir(parents=True,exist_ok=False)
    used=set();shared={k:study.sample_layouts(20261030,'shared-'+k,n,used=used) for k,n in [('validation',validation_count),('test',test_count)]}
    for key,rows in shared.items():
        for row in rows:row['split']=key
    pools={str(seed):study.sample_layouts(seed,'train',max(sizes),used=used) for seed in seeds}
    challenges=challenge_manifest(layouts=challenge_layouts)
    all_layouts=[r['layout_sha256'] for rows in shared.values() for r in rows]+[r['layout_sha256'] for rows in pools.values() for r in rows]
    if len(all_layouts)!=len(set(all_layouts)):raise RuntimeError('Nested training/held-out layout leakage')
    manifest={'schema':study.STUDY_SCHEMA,'contract':study.study_contract(),'seeds':list(seeds),'nested_sizes':list(sizes),
              'shared':shared,'training_pools':pools,'challenges':challenges,'fixed_baseline':study.FIXED_ACTION,
              'model_candidates':MODEL_CANDIDATES,'teacher_budget':teacher_budget,'oracle_budget':oracle_budget,
              'selection_policy':'All model/regularization choices from fixed validation; default checkpoint prespecified final N and first seed',
              'data_origin':'synthetic','physically_verified':False}
    write(output/'manifest.json',manifest);frozen_manifest=(output/'manifest.json').read_bytes()
    fixed=fixed_evaluation(shared['test'],workers);oracle,oracle_candidates=oracle_evaluation(shared['test'],oracle_budget,workers)
    write(output/'comparators.json',{'fixed_baseline':fixed,'held_out_search_oracle':oracle,'oracle_candidates':oracle_candidates})
    curve_rows=[];all_results=[];checkpoints={};validations=[]
    for seed in seeds:
        seed_output=output/f'seed-{seed}';seed_output.mkdir()
        labels=teachers(pools[str(seed)],teacher_budget,seed_output,workers,context={'seed':seed})
        for n in sizes:
            selected_labels=[r for r in labels[:n] if r['label_available']]
            if len(selected_labels)<4:raise ValueError('Insufficient labels; preserved failures, no fabricated checkpoint')
            chosen,candidates=select_model(selected_labels,shared['validation'],workers)
            split={'train':pools[str(seed)][:n],**shared}
            checkpoint=_checkpoint(chosen['model'],selected_labels,split,chosen)
            checkpoint_path=seed_output/f'checkpoint-n{n}.json';write(checkpoint_path,checkpoint);frozen_bytes=checkpoint_path.read_bytes()
            rows=frozen_evaluation(json.loads(frozen_bytes),shared['test'],workers)
            if checkpoint_path.read_bytes()!=frozen_bytes:raise RuntimeError('Checkpoint mutated during test')
            checkpoints[(seed,n)]=checkpoint
            summary={**study.summarize(rows),'train_scenes':n,'seed':seed,'method':'frozen_policy',
                'checkpoint_id':checkpoint['checkpoint_id'],'model_degree':chosen['degree'],'regularization':chosen['regularization'],
                'teacher_labels':len(selected_labels),'teacher_failed_scenes':n-len(selected_labels),
                'teacher_captures':sum(r['teacher_success'] for r in selected_labels),
                'paired_vs_fixed':paired_gain(rows,fixed,study.stream_seed(seed,'paired',n))}
            curve_rows.append(summary);all_results.append({'seed':seed,'train_scenes':n,'rows':rows})
            validations.append({'seed':seed,'train_scenes':n,'candidates':candidates})
            write(output/'curve-results.json',all_results);write(output/'learning-curve.json',curve_rows)
            progress('learning_curve',**summary)
    write(output/'validation.json',validations)
    aggregates=[]
    for n in sizes:
        subset=[r for r in all_results if r['train_scenes']==n]
        per_seed=np.asarray([[float(x['result']['success']) for x in r['rows']] for r in subset])
        layout_means=per_seed.mean(axis=0)
        aggregates.append({'train_scenes':n,'seeds':list(seeds),'layouts':len(layout_means),'mean_success_fraction':float(per_seed.mean()),
            'layout_ci95':layout_interval(layout_means,study.stream_seed(20261030,'aggregate',n)),
            'seed_success_fractions':per_seed.mean(axis=1).tolist(),'seed_range':[float(per_seed.mean(axis=1).min()),float(per_seed.mean(axis=1).max())],
            'ci_method':'layout bootstrap envelope with Wilson effective-layout bound; seed variability separately reported'})
    challenge_rows=[];challenge_results=[]
    for seed in seeds:
        checkpoint=checkpoints[(seed,max(sizes))]
        for group in challenges:
            rows=evaluate_model(checkpoint['model'],group['episodes'],workers,noise=group.get('noise'),allow_ood=True)
            for row in rows:row.update(role='frozen_policy_challenge',checkpoint_id=checkpoint['checkpoint_id'])
            meta={k:v for k,v in group.items() if k!='episodes'}
            meta.update(distance_mm=group['value'] if group['kind']=='distance' else 250. if group['kind']=='offset' else None,
                        lateral_offset_mm=group['value'] if group['kind']=='offset' else None)
            summary={**meta,**cluster_summary(rows,study.stream_seed(seed,group['label'])),'seed':seed,'train_scenes':max(sizes),
                     'checkpoint_id':checkpoint['checkpoint_id'],'method':'frozen_policy',
                     'out_of_distribution_layouts':len({r['layout_sha256'] for r in rows if not r['scope']['in_distribution']})}
            challenge_rows.append(summary);challenge_results.append({'seed':seed,'label':group['label'],'rows':rows})
            write(output/'challenge-summary.json',challenge_rows);write(output/'challenge-results.json',challenge_results)
            progress('challenge',label=group['label'],seed=seed,**study.summarize(rows))
    default=checkpoints[(seeds[0],max(sizes))];write(output/'checkpoint.json',default)
    gallery=[]
    first_rows=next(r['rows'] for r in all_results if r['seed']==seeds[0] and r['train_scenes']==max(sizes))
    for desired,label in [(True,'Held-out capture'),(False,'Held-out failure')]:
        match=next((i for i,r in enumerate(first_rows) if r['result']['success'] is desired),None)
        if match is not None:gallery.append(replay(default,shared['test'][match],label))
    for kind,value in [('distance',60.),('distance',450.),('offset',-200.),('offset',200.),('release_noise',30.)]:
        group=next(g for g in challenges if g['kind']==kind and g['value']==value)
        gallery.append(replay(default,group['episodes'][0],group['label'],noise=group.get('noise')))
    write(output/'replay-gallery.json',gallery)
    if (output/'manifest.json').read_bytes()!=frozen_manifest:raise RuntimeError('Frozen manifest changed')
    report={'schema':'sircus-learning-challenge-report-v1','profile_id':PROFILE_ID,'study_id':study.digest(manifest)[:20],
        'status':'complete','data_origin':'synthetic','geometry_status':'provisional front-side placements and contact dynamics',
        'learning_curve':curve_rows,'learning_curve_aggregate':aggregates,'challenges':challenge_rows,'replays':gallery,
        'summary':{'fixed_baseline':study.summarize(fixed),'limited_search_oracle':study.summarize(oracle),
            'default_checkpoint_id':default['checkpoint_id'],'default_selection':'prespecified maximum N and first training seed, never best test',
            'default_seed':seeds[0],'default_train_scenes':max(sizes),'test_layouts':test_count,'validation_layouts':validation_count,
            'elapsed_seconds':time.monotonic()-start,'manifest_sha256':hashlib.sha256(frozen_manifest).hexdigest()},
        'profile':profile_metadata(),'limits':LIMITS,'physically_verified':False,'native_paths_checked':False}
    write(output/'report.json',report);write(output/'summary.json',report['summary']);progress('complete_curve',study_id=report['study_id'],elapsed_seconds=report['summary']['elapsed_seconds'])
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--seed',type=int,default=20261030);p.add_argument('--train-scenes',type=int,default=24)
    p.add_argument('--validation-scenes',type=int,default=6);p.add_argument('--test-scenes',type=int,default=8)
    p.add_argument('--teacher-budget',type=int,default=12);p.add_argument('--oracle-budget',type=int,default=4)
    p.add_argument('--workers',type=int,choices=(1,2),default=2);p.add_argument('--curve',action='store_true')
    p.add_argument('--seeds',type=int,nargs='+',default=[20261030,20261031,20261032])
    p.add_argument('--sizes',type=int,nargs='+',default=[8,24,64,128]);p.add_argument('--challenge-layouts',type=int,default=24)
    args=p.parse_args()
    if args.curve:
        curves(args.output,seeds=tuple(args.seeds),sizes=tuple(args.sizes),validation_count=args.validation_scenes,
               test_count=args.test_scenes,teacher_budget=args.teacher_budget,oracle_budget=args.oracle_budget,
               workers=args.workers,challenge_layouts=args.challenge_layouts)
    else:
        run(args.output,seed=args.seed,train_scenes=args.train_scenes,validation_scenes=args.validation_scenes,
            test_scenes=args.test_scenes,teacher_budget=args.teacher_budget,oracle_budget=args.oracle_budget,workers=args.workers)


if __name__=='__main__':main()
