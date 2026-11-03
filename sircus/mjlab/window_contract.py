"""Prespecified fixed-bucket release-gap windows and terminal-learning comparison."""
from collections import defaultdict
import numpy as np

from .admission import REFERENCE_BUCKET, REFERENCE_VALUES
from .learning_contract import reward, statistics

VERSION = "sircus-fixed-bucket-gap-study-v1"
SEED = 20261203
BOUNDS = dict(version="sircus-window-low-board-actions-v1",
              low=[.30,-.50,.04,8.,.02],high=[1.04,.35,.56,64.,.65])
GAP_BOUNDS = (.02,.65)
CONTEXT_BOUNDS = ((.04,.56),(8.,64.))
MILESTONES = (504,672,840,1008)
NOISE_GAP_M = .002


def contexts():
    heights = [.06,.12,.20,.30,REFERENCE_VALUES[2],.50]
    tilts = [12.,24.,REFERENCE_VALUES[3],52.]
    def row(name,h,t):return dict(id=name,board_height_m=float(h),tilt_deg=float(t))
    groups = dict(train=[row(f"train-h{i:02d}-t{j:02d}",h,t) for i,h in enumerate(heights) for j,t in enumerate(tilts)],
        validation=[row(f"validation-{i:02d}",h,t) for i,(h,t) in enumerate(zip([.09,.16,.25,.35,.45,.54],[18.,30.,42.,16.,48.,34.]))],
        test=[row(f"test-{i:02d}",h,t) for i,(h,t) in enumerate(zip([.08,.14,.18,.24,.34,.38,.44,.52],[20.,40.,16.,46.,28.,44.,32.,24.]))])
    all_rows=[r for values in groups.values() for r in values]
    assert len(all_rows)==len({r['id'] for r in all_rows})==len({(r['board_height_m'],r['tilt_deg']) for r in all_rows})==38
    return groups


def seed_gaps():
    gaps=np.linspace(*GAP_BOUNDS,21)
    gaps[np.argmin(abs(gaps-REFERENCE_VALUES[4]))]=REFERENCE_VALUES[4]
    return sorted(gaps.tolist())


def settings():
    return dict(version=VERSION,seed=SEED,bucket_xy_m=REFERENCE_BUCKET,board_xy_m=REFERENCE_VALUES[:2],
        action_bounds=BOUNDS,context_bounds=CONTEXT_BOUNDS,gap_bounds=GAP_BOUNDS,contexts=contexts(),seed_gaps_m=seed_gaps(),
        anchor_selection="Known robust development geometry; historical outcomes are not reused as new labels",
        learning="Context=[boardH,tilt]; continuous ACTION=release gap only; each shot terminal; no next-state bootstrap",
        methods=["adaptive_surrogate","sac","uniform_bounded"],
        objectives=dict(adaptive_surrogate="Supervised binary capture probability, uncalibrated; adaptive acquisition also uses shaped reward",sac="Terminal shaped reward plus normalized-action entropy",evaluation="Both ranked by held-out capture counts; no claim of identical training objectives"),
        uniform="One frozen independent bounded-uniform base gap per test context (8 actions); repeat each3 nominal +20 paired gap-noise trials; no feasible-action oracle or resampling",
        budgets=dict(shared_seed_requests=504,adaptive_additional_requests=504,sac_additional_requests=504,
            per_learner_training_requests=1008,unique_training_requests=1512,milestones=list(MILESTONES),
            validation_repeats_per_context=3,test_nominal_repeats=3,test_fresh_repeats=20,
            shared_data_charged_to_each_learner=True,uniform_training_requests=0),
        sac=dict(alpha=.02,warmup_updates=400,updates_per_round=32,batch_size=64,rounds=21,device="cpu"),
        adaptive=dict(rounds=3,queries_per_context_per_round=7,method="Refine observed transitions plus broad uncertain intervals; preserve disconnected sampled hit runs"),
        selection="Max validation captures, then mean shaped reward, then earliest milestone; test never selects/retrains",
        evaluation="Frozen mean SAC action or frozen129-gap surrogate prediction grid; no native outcome search, projection, retry or resampling at inference",
        noise=dict(gap_std_m=NOISE_GAP_M,scope="Gap-only perturbations; exact board context/joint branch fixed; out-of-bounds requests fail"),
        gate_scope="All requests use frozen board branch and strict endpoint checks; common-start paths checked separately for selected control/recipes; no actuator-tracking claim",
        reward="Unchanged sircus.mjlab.learning_contract.reward; success is board rebound + whole-ball entry + continuous1s retention",
        scope="Fixed bucket and board XY; held-posture simulation. Tests generalize across held-out boardH/tilt contexts, not bucket positions, perception or arm trajectories.")


def adaptive_gaps(rows,count=7):
    """Outcome-adaptive proposals on TRAIN only; no monotonic success assumption."""
    if any(r.get('role')!='train' for r in rows):raise ValueError('Adaptive search requires TRAIN rows only')
    grouped=defaultdict(list)
    for r in rows:grouped[round(float(r['gap_m']),12)].append(r)
    ordered=sorted(grouped)
    candidates=[]
    for lo,hi in zip(ordered,ordered[1:]):
        if hi-lo<1e-6:continue
        left,right=grouped[lo],grouped[hi]
        pl=np.mean([bool(r['route_success']) for r in left]);pr=np.mean([bool(r['route_success']) for r in right])
        score=max(np.mean([r['learning_reward'] for r in left]),np.mean([r['learning_reward'] for r in right]))
        candidates.append(((pl!=pr,score,(hi-lo)),(lo+hi)/2))
    # Half transition/promising refinement, half largest remaining intervals.
    ranked=[v for _,v in sorted(candidates,reverse=True)]
    broad=[v for _,v in sorted(candidates,key=lambda x:x[0][2],reverse=True)]
    chosen=[]
    for value in ranked[:(count+1)//2]+broad+ranked:
        if value not in chosen:chosen.append(value)
        if len(chosen)==count:return chosen
    raise ValueError('Insufficient untested gap intervals')


def summary(rows):
    return dict(**statistics(rows),native_releases=sum(bool(r.get('native_release')) for r in rows),
                mean_learning_reward=float(np.mean([reward(r) for r in rows])) if rows else 0.)


def windows(rows):
    """Observed adjacent sample runs, not certified continuous success intervals."""
    by_context=defaultdict(list)
    for row in rows:by_context[row['context_id']].append(row)
    result=[]
    for context_id,records in by_context.items():
        grouped=defaultdict(list)
        for row in records:grouped[float(row['gap_m'])].append(row)
        points=[]
        for gap,values in sorted(grouped.items()):
            points.append(dict(gap_m=gap,attempts=len(values),captures=sum(bool(v['route_success']) for v in values),
                native_releases=sum(bool(v.get('native_release')) for v in values)))
        runs=[];start=None
        for i,p in enumerate(points+[dict(captures=0,attempts=1)]):
            hit=p['captures']==p['attempts'] and p['captures']>0
            if hit and start is None:start=i
            if not hit and start is not None:
                runs.append(dict(low_observed_gap_m=points[start]['gap_m'],high_observed_gap_m=points[i-1]['gap_m'],
                    points=i-start,lower_tested_nonhit_gap_m=points[start-1]['gap_m'] if start else None,
                    upper_tested_nonhit_gap_m=points[i]['gap_m'] if i<len(points) else None));start=None
        result.append(dict(context_id=context_id,points=points,observed_hit_runs=runs,
            scope="All observed requests captured at each listed sample; gaps between samples and boundary probabilities are unvalidated"))
    return result


def outcome_diagnostic(row,positions=None,velocities=None,forces=None,scene=None):
    if row.get('invalid'):return dict(category='geometry_rejected')
    if row.get('route_success'):category='clean_capture' if row['route_quality']==2 else 'rim_assisted_capture'
    elif not row.get('board_bounce'):category='no_board_bounce'
    elif row.get('external_contact_steps'):category='external_contact_failure'
    elif row.get('rim_band_contact') or row.get('bucket_contact_before_entry'):category='bucket_contact_failure'
    else:category='flight_miss'
    result=dict(category=category,scope='240Hz diagnostic observations; original scorer alone determines capture')
    if positions is None:return result
    p=np.asarray(positions);v=np.asarray(velocities);f=np.asarray(forces)
    impact=np.flatnonzero(f[:,0]>1e-3)
    if not len(impact):return result
    first=int(impact[0]);result.update(first_board_contact_sample=first,preimpact_center_z_m=float(p[max(0,first-1),2]),
        preimpact_vertical_velocity_m_s=float(v[max(0,first-1),2]))
    rim=scene['bucket']['assumed_interior_depth_m']+scene['bucket']['assumed_bottom_thickness_m']
    plane=rim+scene['ball']['radius_m'];result['approach_plane_height_m']=plane
    end=np.flatnonzero((f[:,1:]>1e-3).any(1)&(np.arange(len(f))>first))
    stop=int(end[0])+1 if len(end) else len(p)
    after=p[first:stop];result['uninterrupted_peak_center_z_m']=float(after[:,2].max())
    crossing=np.flatnonzero((after[:-1,2]>=plane)&(after[1:,2]<plane))
    if not len(crossing):
        result['approach']='below_rim_approach_plane' if after[:,2].max()<plane else 'no_uninterrupted_descending_crossing';return result
    i=int(crossing[0]);a,b=after[i:i+2];fraction=(a[2]-plane)/(a[2]-b[2]);xy=a[:2]+fraction*(b[:2]-a[:2])
    board=np.asarray(row['values'][:2]);bucket=np.asarray(row['bucket_xy_m']);axis=(bucket-board)/np.linalg.norm(bucket-board)
    along=float((xy-bucket)@axis);lateral=float((xy-bucket)@np.array([-axis[1],axis[0]]))
    usable=scene['bucket']['inner_top_diameter_m']/2-scene['ball']['radius_m']
    result.update(approach_xy_m=xy.tolist(),along_target_error_m=along,lateral_target_error_m=lateral,
        approach='undershoot' if along < -usable else 'overshoot' if along > usable else 'lateral_miss' if abs(lateral)>usable else 'mouth_approach')
    return result
