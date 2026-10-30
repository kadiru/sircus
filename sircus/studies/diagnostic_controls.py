"""Post-study finite-search controls on declared challenge conditions; no fitting."""
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import random
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from sircus.studies import front_study as study
from sircus.studies.run_front_study import write,progress,map_tasks,_evaluate_task,compact,cluster_summary

CONDITIONS=(('distance',220.),('distance',340.),('distance',450.),('offset',-200.),('offset',200.))
BUDGET=16


def search_task(episode):
    """Explicit OOD diagnostic only; same bounded candidate family as teacher."""
    rng=random.Random(episode['action_seed']);aimed=study.aimed_action(episode['scene']);best=None;rows=[]
    for i in range(BUDGET):
        if i==0:action=copy.deepcopy(study.FIXED_ACTION)
        elif i==1:action=copy.deepcopy(aimed)
        elif i<6:
            action=copy.deepcopy(aimed);action['pitch_deg']=[-18.,-24.,-36.,-44.][i-2]
            dx,dy=study.observation(episode['scene'])[3:]
            action['roll_deg']=max(-35.,min(35.,math.degrees(math.atan2(dx*abs(math.sin(math.radians(action['pitch_deg']))),max(1.,-dy)))))
        else:
            center=best['action'] if best else aimed;scale=.22 if i<BUDGET//2 else .09
            action={name:max(low,min(high,center[name]+rng.gauss(0,(high-low)*scale))) for name,(low,high) in zip(study.ACTION_NAMES,study.ACTION_BOUNDS)}
        row=study.evaluate_action(episode['scene'],action,episode['rollout_seed'],allow_ood=True)
        row.update(scene_id=episode['scene_id'],layout_sha256=episode['layout_sha256'],split='post_study_diagnostic',
                   candidate=i,role='post_study_search_diagnostic')
        rows.append(compact(row))
        if study.valid(row) and (best is None or study.selection_key(row['result'])>study.selection_key(best['result'])):best=row
    return rows,compact(best) if best else rows[0]


def run(study_dir,*,workers=2,seed=20261030):
    study.integer(workers,1,2);study.integer(seed,0,2**31-1);study_dir=Path(study_dir)
    report_path=study_dir/'report.json';before=report_path.read_bytes();report=json.loads(before)
    if report['status']!='complete' or not report.get('extensions'):raise ValueError('Core and contact extension must complete first')
    if any(e.get('schema')=='sircus-post-study-diagnostics-v1' for e in report['extensions']):raise ValueError('Diagnostics already present')
    manifest=json.loads((study_dir/'manifest.json').read_text());groups=[]
    for kind,value in CONDITIONS:
        group=copy.deepcopy(next(g for g in manifest['challenges'] if g['kind']==kind and g['value']==value))
        for row in group['episodes']:row['action_seed']=study.stream_seed(seed,'post-study-diagnostic/'+row['scene_id'])
        groups.append(group)
    cp_path=study_dir/'checkpoint.json';cp_before=cp_path.read_bytes();checkpoint=json.loads(cp_before);study.validate_checkpoint(checkpoint)
    output=study_dir/'post-study-diagnostics';output.mkdir(exist_ok=False);start=time.monotonic()
    declaration={'schema':'sircus-post-study-diagnostics-v1','conditions':CONDITIONS,'search_budget_per_layout':BUDGET,
        'groups':groups,'action_seed_master':seed,'compares_default_checkpoint_id':checkpoint['checkpoint_id'],
        'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'contract':study.study_contract(),
        'previous_report_sha256':hashlib.sha256(before).hexdigest(),'checkpoint_file_sha256':hashlib.sha256(cp_before).hexdigest(),
        'interpretation':'Post-study diagnostic controls on existing challenge layouts; no refit, no checkpoint selection; finite search failure does not prove impossibility',
        'data_origin':'synthetic','physically_verified':False,'native_paths_checked':False}
    write(output/'manifest.json',declaration)
    summaries=[];results=[]
    for group in groups:
        fixed=list(map_tasks(_evaluate_task,[(e,study.FIXED_ACTION,None,True) for e in group['episodes']],workers))
        for row in fixed:row['role']='post_study_fixed_diagnostic'
        search=[]
        with (output/(group['label'].replace(' ','-')+'-candidates.jsonl')).open('w') as stream:
            for rows,best in map_tasks(search_task,group['episodes'],workers):
                for row in rows:stream.write(json.dumps(row,sort_keys=True,allow_nan=False)+'\n')
                stream.flush();search.append(best)
        for method,role,rows,short in [('held_out_search_oracle','post_study_search_diagnostic',search,'search'),('fixed_baseline','post_study_fixed_diagnostic',fixed,'fixed')]:
            summary={**cluster_summary(rows,study.stream_seed(seed,group['label']+'/'+method)),
                'method':method,'role':role,'label':'Diagnostic '+short+' · '+group['label'],'kind':group['kind'],
                'value':group['value'],'units':group['units'],'seed':seed,
                'distance_mm':group['value'] if group['kind']=='distance' else 250.,
                'lateral_offset_mm':group['value'] if group['kind']=='offset' else None,
                'out_of_distribution_layouts':len({r['layout_sha256'] for r in rows if not r['scope']['in_distribution']}),
                'search_budget_per_layout':BUDGET if short=='search' else 0,
                'provenance':declaration['interpretation']}
            summaries.append(summary);results.append({'label':group['label'],'method':method,'rows':rows})
            write(output/'summary.json',summaries);write(output/'results.json',results)
            progress('post_study_diagnostic',**summary)
    if cp_path.read_bytes()!=cp_before or report_path.read_bytes()!=before:raise RuntimeError('Frozen checkpoint or report changed during diagnostics')
    (study_dir/'report-pre-diagnostic.json').write_bytes(before)
    extended=copy.deepcopy(report);extended['challenges']+=summaries
    extended['extensions'].append({k:v for k,v in declaration.items() if k not in ('groups','contract')})
    extended['extensions'][-1]['elapsed_seconds']=time.monotonic()-start
    extended['summary']['protocol_counts']['diagnostic_search_proposals']=sum(len(g['episodes']) for g in groups)*BUDGET
    extended['summary']['protocol_counts']['diagnostic_fixed_attempts']=sum(len(g['episodes']) for g in groups)
    extended['limits'].append('Post-study finite-search/fixed controls are diagnostics, not policy inference; zero successes do not establish impossibility')
    noise_group=next(g for g in manifest['challenges'] if g['kind']=='release_noise')
    noise_layouts=list({e['layout_sha256']:e for e in noise_group['episodes']}.values())
    cohorts={}
    for name,episodes in [('shared_test',manifest['shared']['test']),('release_noise',noise_layouts)]:
        distances=[e['scene']['board']['center_mm'][1]-e['scene']['bin']['center_xy_mm'][1] for e in episodes]
        cohorts[name]={'unique_layouts':len(episodes),'mean_forward_distance_mm':sum(distances)/len(distances),
                       'layouts_beyond_280mm':sum(d>280 for d in distances)}
    extended['summary']['cohort_composition']=cohorts
    extended['limits'].append('Noise cohort is closer: mean forward distance 207.1 mm and 2/24 beyond 280 mm, versus shared test 241.2 mm and 16/48; its high capture rate is not overall held-out performance. Manifest was not resampled.')
    write(report_path,extended);progress('complete_diagnostics',elapsed_seconds=time.monotonic()-start)
    return extended


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--study-dir',type=Path,required=True)
    p.add_argument('--workers',type=int,choices=(1,2),default=2);a=p.parse_args();run(a.study_dir,workers=a.workers)
