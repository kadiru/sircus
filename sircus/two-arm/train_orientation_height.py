"""Offline two-arm endpoint-IK + compliant-contact search. No robot client.

Optimizes world Euler roll/pitch and vertical gap; yaw, board center, bucket,
release XY and dropper orientation are fixed. This is fixed-scene optimization,
not a generalized neural policy or collision-planned trajectory learner.
"""
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import random
import time
from datetime import datetime, timezone
import numpy as np
from physics import rotation, release_point, simulate, validate_config
from two_arm_kinematics import feasibility
from trial_store import fingerprint

ROOT = Path(__file__).resolve().parent
SCHEMA = 'sircus-orientation-height-v1'


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def contract_fingerprint(scene):
    raw = fingerprint(scene) + hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    return hashlib.sha256(raw.encode()).hexdigest()


def candidate(base, roll, pitch, gap, release_xy):
    scene = copy.deepcopy(base)
    scene['board']['rpy_deg'][:2] = [float(roll), float(pitch)]
    validate_config(scene)
    r = np.array(rotation(scene['board']['rpy_deg']))
    center = np.array(scene['board']['center_mm'])
    normal = r[:, 2]
    if normal[2] <= .25:
        raise ValueError('Board must face upward with sufficient vertical normal')
    # Intersect the vertical release line with the sphere-center contact plane.
    # This compensates U/V as the board rotates, preserving world release X/Y.
    offset = (scene['board']['size_mm'][2]+scene['ball']['diameter_mm'])/2
    dx, dy = np.array(release_xy)-center[:2]
    z = center[2]+(offset-normal[0]*dx-normal[1]*dy)/normal[2]
    local = r.T @ (np.array([*release_xy,z])-center)
    action = {'u_mm':float(local[0]),'v_mm':float(local[1]),'gap_mm':float(gap)}
    actual = release_point(scene, action)  # validates board-local point and gap
    if not np.allclose(actual[:2], release_xy, atol=1e-7):
        raise ValueError('Release XY invariant failed')
    return scene, action


def verify_policy(policy, base):
    if policy.get('schema') != SCHEMA or policy.get('contract_fingerprint') != contract_fingerprint(base):
        raise ValueError('Policy scene/source fingerprint mismatch; retrain')
    p=policy['parameters'];scene, action=candidate(base,p['roll_deg'],p['pitch_deg'],p['gap_mm'],policy['release_xy_mm'])
    if scene != policy['selected_scene'] or action != policy['release_action']:
        raise ValueError('Policy candidate does not match its saved scene/action')
    return scene, action


def train(base, output, budget=32, tests=10, seed=42, roll_bounds=(-20,20), pitch_bounds=(10,50), gap_bounds=(60,380), release_xy=None):
    validate_config(base)
    if not 4 <= budget <= 500 or not 1 <= tests <= 100:
        raise ValueError('Budget must be4..500; tests1..100')
    if not 0 <= seed < 2**30:
        raise ValueError('Seed must be0..2^30-1')
    bounds=[roll_bounds,pitch_bounds,gap_bounds]
    if any(len(b)!=2 or not all(math.isfinite(v) for v in b) or b[0]>=b[1] for b in bounds):
        raise ValueError('Bounds must be finite ascending pairs')
    if gap_bounds[0]<20 or gap_bounds[1]>1000:
        raise ValueError('Gap bounds must stay within20..1000mm')
    # Known baseline impact location, used only to establish a fixed release XY.
    baseline_action={'u_mm':-16.408,'v_mm':-44.821,'gap_mm':161.715}
    release_xy=list(release_xy) if release_xy is not None else release_point(base,baseline_action)[:2]
    if len(release_xy)!=2 or not all(math.isfinite(v) for v in release_xy):
        raise ValueError('Release XY must be two finite numbers')
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    rng=random.Random(seed);best=None;rows=[]
    write(output/'scene.json',base)
    write(output/'experiment.json',{'schema':SCHEMA,'seed':seed,'budget':budget,'test_count':tests,
        'contract_fingerprint':contract_fingerprint(base),'release_xy_mm':release_xy,'bounds':bounds,
        'data_origin':'synthetic','physically_verified':False,'collision_paths_checked':False,
        'scope':'fixed board center/bucket/release XY/yaw; joint endpoint IK plus reduced-order contact',
        'held_out':'Independent noise seeds; +/-2mm release XY and +/-3mm gap. No cross-scene generalization.'})

    def evaluate(params, trial_seed, phase, number, testing=False):
        scene,action=candidate(base,*params,release_xy)
        check=feasibility(scene,action)
        if testing:
            scene['simulation'].update(noise_xy_mm=2.,noise_gap_mm=3.)
        reachable=check['arm_reachable'] and check['board_arm_reachable']
        result=simulate(scene,action,seed=trial_seed,frames=False) if reachable else {'reward':-1000.,'retained':False,'outcome':'unreachable_arm_target'}
        valid=reachable and not result.get('model_limit',False)
        row={'phase':phase,'trial':number,'seed':trial_seed,'parameters':dict(zip(('roll_deg','pitch_deg','gap_mm'),params)),
             'release_action':action,'reward':float(result['reward']),'retained':bool(result['retained']),
             'valid_for_selection':valid,'outcome':result['outcome'],'feasibility':check,
             'scene':scene,'result':result,'physically_verified':False,'native_paths_checked':False}
        write(output/f'{phase}-{number:04d}.json',row)
        return row

    for i in range(budget):
        if i==0:values=[*base['board']['rpy_deg'][:2],161.715]
        elif best is None or i<budget//2:values=[rng.uniform(*b) for b in bounds]
        else:
            scale=max(.08,1-i/budget)
            values=[best['parameters'][key]+rng.gauss(0,(b[1]-b[0])*.3*scale)
                    for key,b in zip(('roll_deg','pitch_deg','gap_mm'),bounds)]
        values=[max(b[0],min(b[1],v)) for v,b in zip(values,bounds)]
        try: row=evaluate(values,seed,'train',i+1)
        except ValueError as exc:
            row={'phase':'train','trial':i+1,'parameters':values,'valid_for_selection':False,'error':str(exc)}
            write(output/f'train-{i+1:04d}.json',row)
        rows.append(row)
        if row['valid_for_selection'] and (best is None or row['reward']>best['reward']):best=row
        print(json.dumps({'trial':i+1,'budget':budget,'best_reward':None if best is None else best['reward']}),flush=True)
    if best is None:raise RuntimeError('No valid reachable candidate; inspect trials and adjust bounds/scene')
    selected,action=candidate(base,*[best['parameters'][k] for k in ('roll_deg','pitch_deg','gap_mm')],release_xy)
    policy={'schema':SCHEMA,'contract_fingerprint':contract_fingerprint(base),'parameters':best['parameters'],
            'release_xy_mm':release_xy,'release_action':action,'selected_scene':selected,'seed':seed,
            'frozen_utc':datetime.now(timezone.utc).isoformat(),'physically_verified':False,'native_paths_checked':False}
    write(output/'policy.json',policy)  # Freeze BEFORE fresh evaluation; never select on test results.
    write(output/'selected-scene.json',selected)
    evaluated=[evaluate(list(best['parameters'].values()),seed+104729*(j+1),'test',j+1,True) for j in range(tests)]
    summary={'train_trials':budget,'valid_train_trials':sum(r['valid_for_selection'] for r in rows),
             'test_trials':tests,'test_retained':sum(r['retained'] for r in evaluated),'best_parameters':best['parameters'],
             'scope':'synthetic fixed-scene search; endpoint IK only; no hardware or generalization claim'}
    write(output/'summary.json',summary);print(json.dumps(summary),flush=True)
    return policy,summary


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--scene',type=Path,default=ROOT/'scene.json');p.add_argument('--output',type=Path)
    p.add_argument('--budget',type=int,default=32);p.add_argument('--tests',type=int,default=10);p.add_argument('--seed',type=int,default=42)
    p.add_argument('--roll',type=float,nargs=2,default=(-20,20));p.add_argument('--pitch',type=float,nargs=2,default=(10,50));p.add_argument('--gap',type=float,nargs=2,default=(60,380))
    p.add_argument('--release-xy',type=float,nargs=2)
    a=p.parse_args();out=a.output or ROOT/'experiments'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    train(json.loads(a.scene.read_text()),out,a.budget,a.tests,a.seed,a.roll,a.pitch,a.gap,a.release_xy)
if __name__=='__main__':main()
