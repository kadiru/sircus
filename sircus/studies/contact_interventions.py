"""Declared U/V interventions after frozen inference, not policy performance."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import numpy as np
from sircus.studies import front_study as study
from sircus.studies.run_front_study import write,progress,map_tasks,_evaluate_task,cluster_summary,paired_gain,replay as numerical_replay
from two_arm_kinematics import forward_world

CONTACTS=((0.,0.),(-65.,0.),(65.,0.),(0.,-95.),(0.,95.),(-65.,-95.),(-65.,95.),(65.,-95.),(65.,95.))
METHOD='forced_contact_intervention'


def intervened_action(checkpoint,scene,u,v):
    prediction=study.predict(checkpoint,scene)
    action={**prediction['action'],'u_mm':float(u),'v_mm':float(v)}
    study.action_scene(scene,action)
    return prediction['action'],action


def gallery_row(checkpoint,episode,u,v,label):
    predicted,action=intervened_action(checkpoint,episode['scene'],u,v)
    row=study.evaluate_action(episode['scene'],action,episode['rollout_seed'],frames=True)
    result={'id':episode['scene_id']+f'-contact-{u:g}-{v:g}','scene_id':episode['scene_id'],
        'label':label,'role':METHOD,'split':'test_intervention','profile_id':study.PROFILE_ID,
        'scene':row['selected_scene'],'action':action,'result':row['result'],'predicted_action':predicted,'scope':row['scope'],
        'checkpoint_id':checkpoint['checkpoint_id'],'intervention':{'u_mm':u,'v_mm':v},
        'provenance':'synthetic declared U/V intervention after frozen prediction; predicted roll/pitch/gap unchanged; not policy inference performance',
        'data_origin':'synthetic','physically_verified':False,'native_paths_checked':False}
    check=row['feasibility']
    if check and check['arm_reachable'] and check['board_arm_reachable']:
        result['arm14_boxes']=forward_world(row['selected_scene'],14,check['board_check']['joints_deg'])['boxes']
        release=next(r for r in check['arm_checks'] if r['step']=='release')
        result['arm10_release_boxes']=forward_world(row['selected_scene'],10,release['joints_deg'])['boxes']
    result['result'].setdefault('frames',[])
    return result


def run(study_dir,*,workers=2):
    study.integer(workers,1,2);study_dir=Path(study_dir)
    core_path=study_dir/'report.json';raw_core=core_path.read_bytes();core=json.loads(raw_core)
    if core.get('status')!='complete' or core.get('profile_id')!=study.PROFILE_ID:raise ValueError('Completed front study required')
    if core.get('extensions'):raise ValueError('Core report already extended; do not overwrite evidence')
    manifest=json.loads((study_dir/'manifest.json').read_text());episodes=manifest['shared']['test']
    output=study_dir/'contact-intervention';output.mkdir(exist_ok=False);start=time.monotonic()
    declarations={'schema':'sircus-contact-intervention-v1','method':METHOD,'contacts_uv_mm':CONTACTS,
        'layout_source':'frozen shared test manifest','scene_ids':[e['scene_id'] for e in episodes],
        'checkpoint_selection':'prespecified maximum N for every original training seed',
        'unchanged_predicted_fields':['roll_deg','pitch_deg','gap_mm'],'release_noise':study.DEFAULT_NOISE,
        'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'core_report_sha256':hashlib.sha256(raw_core).hexdigest(),'data_origin':'synthetic',
        'interpretation':'Controlled release-location intervention; not frozen policy inference. Changing U/V changes impact location, launch trajectory, endpoint reachability, and compliant response, so effects cannot be attributed solely to suspension.'}
    write(output/'manifest.json',declarations)
    summaries=[];raw=[];selected_checkpoint=None
    for seed in manifest['seeds']:
        path=study_dir/f'seed-{seed}'/f"checkpoint-n{max(manifest['nested_sizes'])}.json"
        checkpoint_bytes=path.read_bytes();cp=json.loads(checkpoint_bytes);study.validate_checkpoint(cp)
        if selected_checkpoint is None:selected_checkpoint=cp
        center=None
        for u,v in CONTACTS:
            predicted_actions=[];actions=[]
            for episode in episodes:
                predicted,action=intervened_action(cp,episode['scene'],u,v);predicted_actions.append(predicted);actions.append(action)
            rows=list(map_tasks(_evaluate_task,[(e,a,None,False) for e,a in zip(episodes,actions)],workers))
            for row,predicted in zip(rows,predicted_actions):
                row.update(role=METHOD,predicted_action=predicted,intervention={'u_mm':u,'v_mm':v},checkpoint_id=cp['checkpoint_id'])
            if center is None:center=rows
            impacts=[r['result']['impact_uv_mm'] for r in rows if r['result'].get('impact_uv_mm') is not None]
            tilts=[r['result']['max_support_tilt_deg'] for r in rows if r['result'].get('max_support_tilt_deg') is not None]
            label=f'Forced board contact U={u:g}, V={v:g} mm'
            summary={**cluster_summary(rows,study.stream_seed(seed,label)),'kind':'contact_intervention','label':label,
                'method':METHOD,'u_mm':u,'v_mm':v,'seed':seed,'checkpoint_id':cp['checkpoint_id'],
                'out_of_distribution_layouts':len({r['layout_sha256'] for r in rows if not r['scope']['in_distribution']}),
                'observed_impact_samples':len(impacts),'support_tilt_samples':len(tilts),
                'mean_observed_impact_uv_mm':np.mean(impacts,axis=0).tolist() if impacts else None,
                'mean_peak_support_tilt_deg':float(np.mean(tilts)) if tilts else None,
                'paired_vs_forced_center':paired_gain(rows,center,study.stream_seed(seed,label+'/pair')),
                'interpretation':declarations['interpretation']}
            summaries.append(summary);raw.append({'seed':seed,'u_mm':u,'v_mm':v,'rows':rows})
            write(output/'results.json',raw);write(output/'summary.json',summaries)
            progress('contact_intervention',seed=seed,u_mm=u,v_mm=v,**study.summarize(rows))
        if path.read_bytes()!=checkpoint_bytes:raise RuntimeError('Frozen checkpoint changed')
    gallery=[gallery_row(selected_checkpoint,episodes[0],u,v,label) for u,v,label in
             ((0.,0.,'Forced center contact'),(65.,0.,'Forced near-edge contact'),(65.,95.,'Forced corner contact'))]
    for n in manifest['nested_sizes']:
        cp=json.loads((study_dir/f"seed-{manifest['seeds'][0]}"/f'checkpoint-n{n}.json').read_text())
        row=numerical_replay(cp,episodes[0],f'N={n}: same predeclared held-out layout')
        row.update(id=episodes[0]['scene_id']+f'-n{n}',train_scenes=n,paired_layout=episodes[0]['scene_id'],
                   scope=study.validate_layout(episodes[0]['scene']),
                   provenance='Synthetic frozen prediction on the same predeclared first test layout; not selected for favorable improvement')
        gallery.append(row)
    write(output/'replays.json',gallery)
    if core_path.read_bytes()!=raw_core:raise RuntimeError('Core report changed during extension')
    (study_dir/'report-core.json').write_bytes(raw_core)
    extended=copy.deepcopy(core)
    episode_by_id={e['scene_id']:e for rows in manifest['shared'].values() for e in rows}
    for group in manifest['challenges']:
        episode_by_id.update({e['scene_id']:e for e in group['episodes']})
    for replay in extended['replays']:
        original=episode_by_id[replay['scene_id']]
        replay['scope']=study.validate_layout(original['scene'],allow_ood=True)
    extended['challenges']+=summaries;extended['replays']+=gallery
    training_hashes={e['layout_sha256'] for pool in manifest['training_pools'].values() for e in pool}
    held_hashes={e['layout_sha256'] for pool in manifest['shared'].values() for e in pool}
    challenge_hashes={e['layout_sha256'] for group in manifest['challenges'] for e in group['episodes']}
    extended['summary']['protocol_counts']={
        'unique_training_layouts':len(training_hashes),'unique_shared_validation_test_layouts':len(held_hashes),
        'unique_challenge_layouts':len(challenge_hashes),'unique_core_layouts':len(training_hashes|held_hashes|challenge_hashes),
        'teacher_proposals':sum(len(pool) for pool in manifest['training_pools'].values())*manifest['teacher_budget'],
        'validation_attempts':len(manifest['seeds'])*len(manifest['nested_sizes'])*len(manifest['model_candidates'])*len(manifest['shared']['validation']),
        'frozen_test_attempts':len(manifest['seeds'])*len(manifest['nested_sizes'])*len(episodes),
        'fixed_baseline_attempts':len(episodes),'search_oracle_proposals':len(episodes)*manifest['oracle_budget'],
        'frozen_challenge_attempts':len(manifest['seeds'])*sum(len(g['episodes']) for g in manifest['challenges']),
        'forced_contact_attempts':sum(len(g['rows']) for g in raw),
        'scope':'Protocol attempts excluding separately rendered gallery replays and regression inspections; repeated layouts are not additional independent samples'}
    extended['limits']+=['Center/edge U/V intervention changes trajectory, endpoint reachability and compliant response together; it does not isolate suspension effects',
        'Distance challenge groups independently sample small board-pose variations; descriptive sampled comparison, not a paired causal distance experiment',
        'Raw stdout challenge progress uses generic per-attempt summaries: its repeated-noise intervals are not inference-quality; published clustered challenge-summary/report intervals are authoritative']
    extended['extensions']=[{**declarations,'elapsed_seconds':time.monotonic()-start,'summary_rows':len(summaries),
        'paired_difference_ci_limitation':'Percentile paired bootstrap can degenerate for all-equal differences; absolute rates retain finite-sample bounds'}]
    write(core_path,extended);progress('complete_contact_interventions',elapsed_seconds=time.monotonic()-start)
    return extended


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--study-dir',type=Path,required=True)
    parser.add_argument('--workers',type=int,choices=(1,2),default=2);args=parser.parse_args();run(args.study_dir,workers=args.workers)
