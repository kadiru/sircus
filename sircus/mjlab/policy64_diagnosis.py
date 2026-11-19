"""Read-only post-hoc explanation of the audited fixed64 BC/PPO study.

No fitting, action selection, physics execution or evidence mutation occurs here.
Published TEST positions become development/diagnosis data after this inspection.
"""
import argparse
import collections
import copy
import hashlib
import json
import math
from pathlib import Path
import statistics

VERSION = 'sircus-policy64-diagnosis-v1'
PARENT_FP = 'd30ad91111cb8beb26438f2271357a103daf50d0399ac3646f3b7f163d87f02e'
PARENT_AUDIT = '4dc317e0697e426a4708c3c997f8671d97b1f49de6bd4450c56e6487a6e93ac3'
PARAMETERS = ['board_x_m','board_y_m','board_height_m','board_tilt_deg','nominal_plane_gap_m']
LOSS_IDS = ['policy64-test-'+n for n in ('04','07','13','17','21','26','43')]
SCOPE = ('Post-hoc diagnosis of one fixed64 BC/PPO experiment, not another training run. '
         'The original50 TEST positions are now development/diagnosis data. Historical '
         'scores and policies stay immutable; any later tuned model needs fresh holdouts. '
         'Geometry/held-endpoint simulation only; no noise,128 qualification, learned '
         'grasp/transit, perception or physical validation.')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,allow_nan=False).encode()).hexdigest()


def captured(row):
    return bool(row.get('native_release') and row.get('route_success') and not row.get('invalid') and not row.get('invalid_physics'))


def outcome(row):
    if row.get('native_release') is False:return 'rejected'
    if row.get('native_release') is not True:return 'unknown'
    if row.get('invalid') or row.get('invalid_physics'):return 'invalid'
    return 'capture' if captured(row) else 'miss'


def stats(rows):
    return dict(requested=len(rows),native=sum(r.get('native_release') is True for r in rows),
        captures=sum(captured(r) for r in rows),clean=sum(captured(r) and r.get('route_quality')==2 for r in rows),
        rejected=sum(r.get('native_release') is False for r in rows),
        native_misses=sum(outcome(r)=='miss' for r in rows),invalid=sum(outcome(r)=='invalid' for r in rows),
        mean_reward=statistics.fmean(r['learning_reward'] for r in rows) if rows else None)


def geometry_evidence(connection):
    """Summarize measured proof fields without exposing joint vectors."""
    endpoint=connection.get('endpoint_attempts') or {}
    checks=[p.get('checked',{}) for p in endpoint.get('pair_attempts',[]) if isinstance(p,dict)]
    strict=[c.get('strict_geometry') for c in checks if isinstance(c.get('strict_geometry'),dict)]
    margins=[g.get('minimum_clearance_m') for g in strict if type(g.get('minimum_clearance_m')) in (int,float) and math.isfinite(g['minimum_clearance_m'])]
    pairs=sorted({tuple(g['closest_pair']) for g in strict if isinstance(g.get('closest_pair'),list) and len(g['closest_pair'])==2})
    return dict(reason=connection.get('reason'),passed=connection.get('passed'),
        minimum_required_clearance_m=.005,checked_branch_pairs=len(checks),
        valid_ik_branches=endpoint.get('valid_ik_branches'),
        best_checked_clearance_m=max(margins) if margins else None,
        limiting_pairs=[list(p) for p in pairs],
        proof_minimum_clearance_m=(connection.get('proof') or {}).get('minimum_clearance_m'),
        endpoint_reason=endpoint.get('reason'),
        interpretation=('No admitted solution found within the fixed solver budget; not a proof of unreachability.'
            if connection.get('reason')=='finite_ik_unresolved' else
            'Endpoint or sampled setup proof; does not establish driven actuator transit.'))


def compact_row(row,connection=None):
    keys=('id','values','native_release','invalid','invalid_physics','learning_reward','board_bounce',
          'whole_ball_entry','max_retention_s','closest_opening_distance_m','external_contact_steps',
          'board_contact_episodes','artifact_sha256','connection_sha256','source_teacher_id',
          'rim_band_contact','bucket_contact_before_entry','entry_class','minimum_mouth_clearance_mm',
          'whole_ball_entry_time_s','capture_time_s','retention_resets','first_mouth_passage_time_s')
    result={k:copy.deepcopy(row.get(k)) for k in keys}
    result.update(record_id=row['id'],record_sha256=digest(row),capture=captured(row),
        clean=captured(row) and row.get('route_quality')==2,outcome=outcome(row),
        setup_stage=row.get('geometry_reason'),geometry_evidence=geometry_evidence(connection or {}))
    return result


def paired_analysis(rows,connections=None):
    selected=[r for r in rows if r.get('role')=='test' and r.get('method') in ('bc','ppo')]
    by={(r['method'],r['context_id']):r for r in selected}
    ids=sorted({r['context_id'] for r in selected})
    if len(selected)!=100 or len(by)!=100 or len(ids)!=50 or any((m,i) not in by for m in ('bc','ppo') for i in ids):
        raise ValueError('Exactly50 paired BC/PPO requested layouts required')
    connections=connections or {};pairs=[];transitions=collections.Counter();admission=collections.Counter()
    for ident in ids:
        b,p=by['bc',ident],by['ppo',ident]
        if b['bucket_xy_m']!=p['bucket_xy_m']:raise ValueError('Paired bucket coordinates differ')
        delta=[float(y)-float(x) for x,y in zip(b['values'],p['values'])]
        if len(delta)!=5 or not all(math.isfinite(v) for v in delta):raise ValueError('Finite five-dimensional paired actions required')
        transitions[outcome(b),outcome(p)]+=1
        admission['both_native' if b['native_release'] and p['native_release'] else
                  'bc_only_native' if b['native_release'] else 'ppo_only_native' if p['native_release'] else 'neither_native']+=1
        shared=dict(same_id=b.get('source_teacher_id')==p.get('source_teacher_id'),
            same_values=b.get('effective_source_hint',{}).get('values')==p.get('effective_source_hint',{}).get('values'),
            same_observation=b.get('observation')==p.get('observation'),bc_id=b.get('source_teacher_id'),ppo_id=p.get('source_teacher_id'))
        pairs.append(dict(context_id=ident,bucket_xy_m=b['bucket_xy_m'],bc=compact_row(b,connections.get(b['id'])),
            ppo=compact_row(p,connections.get(p['id'])),delta_values=delta,
            delta_release_height_m=delta[2]+delta[4],shared_anchor=shared))
    counts=dict(layouts=50,bc=stats([by['bc',i] for i in ids]),ppo=stats([by['ppo',i] for i in ids]),
        both_capture=sum(captured(by['bc',i]) and captured(by['ppo',i]) for i in ids),
        bc_only=sum(captured(by['bc',i]) and not captured(by['ppo',i]) for i in ids),
        ppo_only=sum(not captured(by['bc',i]) and captured(by['ppo',i]) for i in ids),
        neither=sum(not captured(by['bc',i]) and not captured(by['ppo',i]) for i in ids),
        transitions=[dict(bc_outcome=b,ppo_outcome=p,count=n) for (b,p),n in sorted(transitions.items())],
        admission_cohorts=dict(admission))
    jointly=[i for i in ids if by['bc',i]['native_release'] and by['ppo',i]['native_release']]
    counts['jointly_admitted']=dict(layouts=len(jointly),bc_captures=sum(captured(by['bc',i]) for i in jointly),
        ppo_captures=sum(captured(by['ppo',i]) for i in jointly),
        caveat='Conditioning on joint admission is descriptive and does not remove rejected requests from the primary denominator.')
    groups={'all':pairs,'bc_only':[r for r in pairs if r['bc']['capture'] and not r['ppo']['capture']],
            'ppo_only':[r for r in pairs if r['ppo']['capture'] and not r['bc']['capture']]}
    delta_summary=[]
    for name,group in groups.items():
        if not group:continue
        columns=list(zip(*(r['delta_values'] for r in group)))
        delta_summary.append(dict(group=name,layouts=len(group),parameters=PARAMETERS,
            mean_delta=[statistics.fmean(v) for v in columns],mean_abs_delta=[statistics.fmean(abs(x) for x in v) for v in columns],
            max_abs_delta=[max(abs(x) for x in v) for v in columns],min_delta=[min(v) for v in columns],max_delta=[max(v) for v in columns]))
    reasons=[]
    for method in ('bc','ppo'):
        for reason in sorted({by[method,i].get('invalid') for i in ids if not by[method,i]['native_release']}):
            cases=[i for i in ids if not by[method,i]['native_release'] and by[method,i].get('invalid')==reason]
            reasons.append(dict(method=method,reason=reason,count=len(cases),context_ids=cases))
    return dict(paired_counts=counts,pairs=pairs,delta_summary=delta_summary,rejection_reasons=reasons,
        actor_shift=dict(board_height_lower_count=sum(r['delta_values'][2]<0 for r in pairs),
            release_gap_higher_count=sum(r['delta_values'][4]>0 for r in pairs)))


def original_flights(study,rows,scene,calibration):
    import numpy as np
    from .xy650_metrics import first_flight
    grouped=collections.defaultdict(list);result={}
    for r in rows:
        if r.get('role')=='test' and r.get('method') in ('bc','ppo') and r.get('native_release'):
            grouped[r['native_cache_file']].append(r)
    for name,group in sorted(grouped.items()):
        path=study/'native'/name;cache_sha=sha(path)
        with np.load(path,allow_pickle=False) as z:
            names=('position_m','velocity_m_s','forces_N','time_s','dt_s','contact_distance_m')
            arrays={k:z[k] for k in names if k in z.files};ids=z['record_ids'].tolist()
            for r in group:
                column=r['native_cache_row']
                if str(ids[column])!=r['id'] or cache_sha!=r['native_cache_sha256']:raise ValueError('Original native trace identity changed')
                trace={k:v[:,column] if v.ndim>=2 and k!='time_s' else v for k,v in arrays.items()}
                metric=first_flight(scene,r['bucket_xy_m'],r['board_normal'],calibration,trace,valid=not r.get('invalid_physics'))
                if metric.get('available'):
                    start,end=metric['start_index'],metric['end_index']
                    metric.update(rebound_position_m=trace['position_m'][start].tolist(),
                        rebound_velocity_m_s=trace['velocity_m_s'][start].tolist(),
                        first_contact_position_m=trace['position_m'][end].tolist() if metric['cutoff_reason']=='contact' else None,
                        first_contact_velocity_before_m_s=trace['velocity_m_s'][end].tolist() if metric['cutoff_reason']=='contact' else None)
                direction=np.asarray(r['bucket_xy_m'])-np.asarray(r['values'][:2]);direction/=np.linalg.norm(direction)
                if metric.get('first_crossing') is not None:
                    offset=np.asarray(metric['first_crossing']['xy_m'])-r['bucket_xy_m']
                    metric['forward_error_m']=float(offset@direction)
                    metric['lateral_error_m']=float(offset@np.array([-direction[1],direction[0]]))
                else:metric.update(forward_error_m=None,lateral_error_m=None)
                result[r['id']]=dict(record_id=r['id'],record_sha256=digest(r),context_id=r['context_id'],method=r['method'],
                    bucket_xy_m=r['bucket_xy_m'],values=r['values'],board_center_m=r['board_center_m'],
                    board_normal=r['board_normal'],substeps=64,position_m=r['trajectory_m'],sample_dt_s=r['trajectory_dt_s'],
                    capture=captured(r),clean=captured(r) and r.get('route_quality')==2,
                    whole_ball_entry=r['whole_ball_entry'],max_retention_s=r['max_retention_s'],
                    native_cache_sha256=r['native_cache_sha256'],native_cache_row=column,first_flight=metric,
                    provenance='Observed original native trace; no generated trajectory or new rollout.')
    return result


def optimizer_analysis(study):
    import numpy as np
    import torch
    from .benchmark_policy import ResidualPolicy
    torch.set_num_threads(2);updates=[json.loads(x) for x in (study/'updates.jsonl').read_text().splitlines()]
    result=[]
    for row in updates:
        before,after=ResidualPolicy(),ResidualPolicy()
        for model,key in ((before,'behavior'),(after,'after_artifact')):
            state=torch.load(study/row[key]['path'],map_location='cpu',weights_only=True);model.load_state_dict(state['policy']);model.eval()
        with np.load(study/row['buffer'],allow_pickle=False) as z:
            obs=torch.tensor(z['observation'],dtype=torch.float32);latent=torch.tensor(z['latent'],dtype=torch.float32)
            old=torch.tensor(z['old_logp'],dtype=torch.float32);reward=torch.tensor(z['reward'],dtype=torch.float32)
        with torch.no_grad():
            b,a=before.distribution(obs),after.distribution(obs);ratio=(a.log_prob(latent).sum(-1)-old).exp()
            kl=torch.distributions.kl_divergence(b,a).sum(-1)
            result.append(dict(index=row['index'],samples=(row['index']+1)*32,
                analytical_kl_mean=float(kl.mean()),analytical_kl_max=float(kl.max()),
                sampled_ratio_clip_fraction=float(((ratio<.8)|(ratio>1.2)).float().mean()),
                critic_mse_before=float(((before.critic(obs).squeeze(-1)-reward)**2).mean()),
                critic_mse_after=float(((after.critic(obs).squeeze(-1)-reward)**2).mean()),
                standard_deviation_before=b.scale[0].tolist(),standard_deviation_after=a.scale[0].tolist(),
                gradient_norm_max=max(e['gradient_norm'] for e in row['epochs']),
                before_sha256=row['behavior']['sha256'],after_sha256=row['after_artifact']['sha256']))
    return dict(updates=result,bc_retention_penalty=False,
        mean_post_update_kl=statistics.fmean(r['analytical_kl_mean'] for r in result),
        mean_post_update_ratio_clip_fraction=statistics.fmean(r['sampled_ratio_clip_fraction'] for r in result),
        metric_definition='KL and likelihood-ratio fractions compare each saved behavior model with its final model after six epochs on that update buffer. Ratio outside[0.8,1.2] is not an invalid physical action.',
        caveat='PPO clipping compares each update with its own behavior policy, not with the original BC policy. These diagnostics do not establish causality.')


def model_options():
    return dict(status='proposed_not_trained',current='Current BC/PPO already use a neural network: geometry in, five shot parameters out. It does not explicitly predict physics.',
        recommendation='Try a small uncertainty-aware ensemble forward model before a video WAM: geometry plus shot settings predicts rebound velocity, first-flight aperture error and capture/retention.',
        feasibility='Keep exact IK/clearance checks separate. A dynamics model cannot make a rejected arm configuration feasible.',
        inputs=['bucket-relative board position','board height and normal','ball release pose/velocity','ball/board/bucket geometry and contact parameters'],
        targets=['post-impact ball velocity','signed forward/lateral aperture error','whole entry','retention duration','capture'],
        evaluation='Fit only TRAIN/development trajectories; split by bucket layout and shot family. Validate uncertainty and native captures on newly frozen future holdouts. Do not count predictions as simulator successes.',
        wam='A world-action model predicts future observations/states and actions jointly. Rich visual interaction data would be needed for a useful task-specific comparison; pretrained video priors are not verified bounce physics.',
        sources=[dict(title='PETS: uncertainty-aware dynamics ensemble',url='https://arxiv.org/abs/1805.12114'),
                 dict(title='MBPO: short model rollouts and model bias',url='https://arxiv.org/abs/1906.08253'),
                 dict(title='NVIDIA: world-action models',url='https://developer.nvidia.com/blog/beyond-vlas-how-world-action-models-reshape-robot-manipulation')])


def analyze(study,parent_audit,output):
    from .curate_policy64 import verified_inputs
    study,spec,report,audit,manifest,rows=verified_inputs(study,parent_audit)
    if spec['fingerprint']!=PARENT_FP or sha(parent_audit)!=PARENT_AUDIT:raise ValueError('Unexpected exact parent study/proof')
    connections={r['id']:read(study/'native'/r['connection_file']) for r in rows if r['role']=='test' and r['method'] in ('bc','ppo')}
    result=paired_analysis(rows,connections)
    lost=[r['context_id'] for r in result['pairs'] if r['bc']['capture'] and not r['ppo']['capture']]
    if lost!=LOSS_IDS:raise ValueError('Prespecified seven paired losses differ')
    flights=original_flights(study,rows,spec['scene'],read(study/'calibration.json')['score_config'])
    chosen=[r for r in result['pairs'] if r['bc']['capture']!=r['ppo']['capture']]
    result['examples']=[dict(context_id=r['context_id'],selection_rule='All7 BC-only losses and all5 PPO-only gains, selected post-hoc from the original paired50.',
        basis='original_TEST_now_diagnosis',projection_reference=dict(origin_xy_m=r['bucket_xy_m'],direction_from_xy_m=r['bc']['values'][:2]),
        bc_trace=flights.get(r['bc']['record_id']),ppo_trace=flights.get(r['ppo']['record_id'])) for r in chosen]
    for pair in result['pairs']:
        for method in ('bc','ppo'):
            trace=flights.get(pair[method]['record_id']);pair[method]['first_flight']=trace.get('first_flight') if trace else None
    result.update(version=VERSION,status='completed',phase='original_analysis',scope=SCOPE,
        source=dict(study_version=report['version'],contract_fingerprint=spec['fingerprint'],report_sha256=sha(study/'report.json'),
            manifest_sha256=sha(study/'evidence-manifest.json'),original_audit_sha256=sha(parent_audit),
            original_audit_checks=audit['checks_count'],rows_sha256=sha(study/'rows.jsonl'),substeps=64,
            selected=report['selected'],bc=report['artifacts']['bc']),
        parameter_order=PARAMETERS,original_comparison=report['test_comparison'],
        validation=report['curves'],optimization_history=optimizer_analysis(study),model_options=model_options(),
        native_study=dict(status='pending',plan_requests=96,observed_requests=0),
        hypotheses=[dict(statement='PPO changes pushed several safe BC endpoints across finite IK/clearance admission boundaries.',evidence='Five of seven lost captures became rejected setups; exact proofs and parameter swaps test sensitivity.',status='observed_association_counterfactuals_pending'),
            dict(statement='Global actor drift may trade local feasible recipes for higher TRAIN reward.',evidence='Every TEST board height decreased; no explicit retention term anchors PPO to BC.',status='hypothesis_not_proven'),
            dict(statement='Shaped reward gaming explains lower captures.',evidence='PPO TEST mean reward also fell; no improvement in shaped TEST reward was observed.',status='not_supported_by_this_comparison')])
    path=Path(output)
    if path.exists() or path.resolve().is_relative_to(study):raise ValueError('New external analysis path required')
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(result,indent=2,sort_keys=True,allow_nan=False)+'\n')
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('study','parent-audit','output'):p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();r=analyze(a.study,a.parent_audit,a.output)
    print(json.dumps(dict(version=r['version'],status=r['status'],paired_counts=r['paired_counts'],examples=len(r['examples']))))


if __name__=='__main__':main()
