"""Front-bucket scene-conditioned regression; numerical-only and version guarded.

Prediction is a frozen feature expansion and matrix product. Teacher/oracle search
is a separately named operation and never invoked by prediction or preview.
"""
import copy
import hashlib
import json
import math
from pathlib import Path
import random
import sys

import numpy as np
import scipy

ROOT = Path(__file__).resolve().parents[2]
TWO_ARM = ROOT/'sircus/two-arm'
sys.path.insert(0, str(TWO_ARM))
from shot_rollout import (canonical, contract_manifest, prepare_scene, release_coordinates,
                          rollout, selection_key, failure_result)
from capture_scoring import PHYSICS_VERSION, SCORING_VERSION
from two_arm_kinematics import feasibility
from sircus.studies.profiles import PROFILE_ID, DISTRIBUTION, front_scene, profile_metadata

CHECKPOINT_SCHEMA = 'sircus-front-scene-ridge-v1'
STUDY_SCHEMA = 'sircus-front-learning-study-v1'
ACTION_NAMES = ('roll_deg', 'pitch_deg', 'u_mm', 'v_mm', 'gap_mm')
ACTION_BOUNDS = ((-35.,35.), (-55.,-8.), (-70.,70.), (-100.,100.), (60.,280.))
OBSERVATION_NAMES = ('board_x_mm','board_y_mm','board_z_mm','bucket_dx_mm','bucket_dy_mm')
FIXED_ACTION = dict(zip(ACTION_NAMES, (0.,-30.,0.,0.,220.)))
DEFAULT_NOISE = {'xy_mm':2., 'gap_mm':3.}


def digest(value): return hashlib.sha256(canonical(value)).hexdigest()


def finite(value):
    if type(value) not in (int,float) or not math.isfinite(value):
        raise ValueError('Finite numeric value required')
    return float(value)


def integer(value, low, high):
    if type(value) is not int or not low <= value <= high:
        raise ValueError(f'Integer in {low}..{high} required')
    return value


def stream_seed(seed, namespace, index=0):
    integer(seed,0,2**31-1)
    return int(hashlib.sha256(f'{seed}/{namespace}/{index}'.encode()).hexdigest()[:8],16)


def observation(scene):
    c=scene['board']['center_mm']; b=scene['bin']['center_xy_mm']
    return [finite(x) for x in [*c,b[0]-c[0],b[1]-c[1]]]


def layout_hash(scene): return digest(observation(scene))


def asset_template(scene):
    value=copy.deepcopy(scene);value.pop('scene_id',None)
    value['board']['center_mm']=None;value['board']['rpy_deg'][:2]=[None,None]
    value['bin']['center_xy_mm']=None
    return value


def validate_layout(scene, *, allow_ood=False):
    prepare_scene(scene)
    if asset_template(scene)!=asset_template(front_scene()):
        raise ValueError('Front profile asset/tool/contact contract mismatch; historical geometry is separate')
    bounds=DISTRIBUTION['board_center_mm']+DISTRIBUTION['bucket_relative_xy_mm']
    outside=[name for name,x,limits in zip(OBSERVATION_NAMES,observation(scene),bounds) if not limits[0]<=x<=limits[1]]
    if outside and not allow_ood:raise ValueError('Outside front trained distribution: '+', '.join(outside))
    return {'in_distribution':not outside,'out_of_distribution_fields':outside}


def study_contract():
    scene=front_scene(); inherited=contract_manifest(scene)
    sources=dict(inherited['source_sha256'])
    for name in ('profiles.py','front_study.py','run_front_study.py'):
        sources['sircus/studies/'+name]=hashlib.sha256((ROOT/'sircus/studies'/name).read_bytes()).hexdigest()
    return {'schema':STUDY_SCHEMA,'profile_id':PROFILE_ID,'scoring_version':SCORING_VERSION,
        'physics_version':PHYSICS_VERSION,'asset_sha256':digest(asset_template(scene)),
        'source_sha256':sources,'dependency_pins':inherited['dependency_pins'],
        'actual_packages':{'numpy':np.__version__,'scipy':scipy.__version__},
        'action_names':list(ACTION_NAMES),'action_bounds':[list(x) for x in ACTION_BOUNDS],
        'observation_names':list(OBSERVATION_NAMES),'distribution':copy.deepcopy(DISTRIBUTION),
        'coordinates':inherited['coordinates'],'settings':inherited['settings'],
        'release_noise':DEFAULT_NOISE,'profile':profile_metadata()}


def sample_layouts(seed, namespace, count, *, used=None):
    integer(count,1,2000);rng=random.Random(stream_seed(seed,namespace));used=set() if used is None else used
    rows=[]
    for index in range(count):
        for _ in range(100):
            s=front_scene();s['board']['center_mm']=[rng.uniform(*b) for b in DISTRIBUTION['board_center_mm']]
            delta=[rng.uniform(*b) for b in DISTRIBUTION['bucket_relative_xy_mm']]
            s['bin']['center_xy_mm']=[s['board']['center_mm'][i]+delta[i] for i in (0,1)]
            h=layout_hash(s)
            if h not in used:break
        else:raise ValueError('Cannot sample unique scene')
        used.add(h);scene_id=f'{namespace}-{index:04d}-{h[:12]}';s['scene_id']=scene_id
        rows.append({'scene_id':scene_id,'layout_sha256':h,'scene':s,'split':namespace,
            'rollout_seed':stream_seed(seed,namespace+'/noise',index),
            'action_seed':stream_seed(seed,namespace+'/action',index),'observation':observation(s)})
    return rows


def action_scene(scene,action):
    if not isinstance(action,dict) or set(action)!=set(ACTION_NAMES):raise ValueError('All five explicit action fields required')
    action={name:finite(action[name]) for name in ACTION_NAMES}
    for name,(low,high) in zip(ACTION_NAMES,ACTION_BOUNDS):
        if not low<=action[name]<=high:raise ValueError('Action outside front bounds: '+name)
    selected=copy.deepcopy(scene);selected['board']['rpy_deg'][:2]=[action['roll_deg'],action['pitch_deg']]
    release={key:action[key] for key in ('u_mm','v_mm','gap_mm')}
    return selected,release,release_coordinates(selected,release)


def evaluate_action(scene,action,seed=0,*,frames=False,noise=None,allow_ood=False):
    selected=release=coords=check=None
    try:
        scope=validate_layout(scene,allow_ood=allow_ood)
        selected,release,coords=action_scene(scene,action)
        noise=DEFAULT_NOISE if noise is None else noise
        if set(noise)!={'xy_mm','gap_mm'}:raise ValueError('Explicit XY/gap noise fields required')
        xy,gap=finite(noise['xy_mm']),finite(noise['gap_mm'])
        if not 0<=xy<=30 or not 0<=gap<=50:raise ValueError('Noise outside supported envelope')
        selected['simulation'].update(noise_xy_mm=xy,noise_gap_mm=gap)
        check=feasibility(selected,release)
        if check['arm_reachable'] and check['board_arm_reachable']:
            result=rollout(selected,release,seed=seed,frames=frames)
        else:result=failure_result('invalid_action','one or more endpoint IK targets unreachable')
    except (ValueError,KeyError,TypeError) as exc:
        result=failure_result('invalid_action',str(exc));scope={'in_distribution':False}
    return {'action':copy.deepcopy(action),'selected_scene':selected,'release_action':release,
        'release_coordinates':coords,'feasibility':check,'result':result,'seed':seed,
        'profile_id':PROFILE_ID,'scope':scope,'noise':copy.deepcopy(noise or DEFAULT_NOISE),
        'scoring_version':SCORING_VERSION,'physics_version':PHYSICS_VERSION,
        'data_origin':'synthetic','physically_verified':False,'native_paths_checked':False}


def valid(row):
    return bool(row['result'].get('executed')) and row['result']['outcome'] not in ('invalid_action','invalid_contact','model_limit')


def aimed_action(scene):
    """Predeclared geometric search seed, never a learned inference fallback."""
    dx,dy=observation(scene)[3:];distance=math.hypot(dx,dy)
    pitch=-30.;roll=math.degrees(math.atan2(dx*.5,max(1.,-dy)))
    return {'roll_deg':max(-35.,min(35.,roll)),'pitch_deg':pitch,'u_mm':0.,'v_mm':0.,
            'gap_mm':max(60.,min(260.,220.*(distance/250.)**1.35))}


def search_episode(episode,budget=16,*,role='training_teacher'):
    integer(budget,1,128)
    if role=='training_teacher' and episode['split']!='train':raise ValueError('Teacher cannot use held-out scenes')
    if role not in ('training_teacher','held_out_oracle'):raise ValueError('Explicit search role required')
    rng=random.Random(episode['action_seed']);best=None;rows=[];aimed=aimed_action(episode['scene'])
    for i in range(budget):
        if i==0:action=copy.deepcopy(FIXED_ACTION)
        elif i==1:action=copy.deepcopy(aimed)
        elif i<min(6,budget):
            action=copy.deepcopy(aimed)
            action['pitch_deg']=[-18.,-24.,-36.,-44.][i-2]
            action['roll_deg']=max(-35.,min(35.,math.degrees(math.atan2(observation(episode['scene'])[3]*abs(math.sin(math.radians(action['pitch_deg']))),max(1.,-observation(episode['scene'])[4])))))
        else:
            center=best['action'] if best else aimed
            scale=.22 if i<budget//2 else .09
            action={name:max(low,min(high,center[name]+rng.gauss(0,(high-low)*scale))) for name,(low,high) in zip(ACTION_NAMES,ACTION_BOUNDS)}
        row=evaluate_action(episode['scene'],action,episode['rollout_seed'])
        row.update(scene_id=episode['scene_id'],layout_sha256=episode['layout_sha256'],split=episode['split'],candidate=i,role=role)
        rows.append(row)
        if valid(row) and (best is None or selection_key(row['result'])>selection_key(best['result'])):best=row
    return rows,best


def features(z,degree):
    z=np.atleast_2d(np.asarray(z,float));columns=[np.ones(len(z)),*[z[:,i] for i in range(5)]]
    if degree==2:columns.extend(z[:,i]*z[:,j] for i in range(5) for j in range(i,5))
    elif degree!=1:raise ValueError('Linear or quadratic feature degree required')
    return np.column_stack(columns)


def fit_model(x,y,alpha,degree):
    x,y=np.asarray(x,float),np.asarray(y,float)
    if x.ndim!=2 or x.shape[1]!=5 or y.shape!=(len(x),5) or len(x)<4 or not np.isfinite(x).all() or not np.isfinite(y).all():raise ValueError('At least four finite train-only labels required')
    alpha=finite(alpha)
    if alpha<=0:raise ValueError('Positive ridge regularization required')
    mean=x.mean(axis=0);scale=np.maximum(x.std(axis=0),1e-9);design=features((x-mean)/scale,degree)
    bounds=np.asarray(ACTION_BOUNDS);normalized=2*(y-bounds[:,0])/(bounds[:,1]-bounds[:,0])-1
    penalty=np.diag([0.,*([alpha]*(design.shape[1]-1))])
    weights=np.linalg.solve(design.T@design+penalty,design.T@normalized)
    return {'kind':'polynomial_ridge','degree':degree,'alpha':alpha,'mean':mean.tolist(),'scale':scale.tolist(),'weights':weights.tolist()}


def infer_model(model,scene):
    x=np.asarray(observation(scene));mean=np.asarray(model['mean']);scale=np.asarray(model['scale']);weights=np.asarray(model['weights'])
    values=features((x-mean)/scale,model['degree'])[0]@weights
    bounds=np.asarray(ACTION_BOUNDS);actions=bounds[:,0]+(np.clip(values,-1,1)+1)/2*(bounds[:,1]-bounds[:,0])
    return dict(zip(ACTION_NAMES,actions.tolist()))


def checkpoint(model,training_ids,split_manifest,selection):
    value={'schema':CHECKPOINT_SCHEMA,'profile_id':PROFILE_ID,'base_scene':front_scene(),'contract':study_contract(),
        'model':copy.deepcopy(model),'training_scene_ids':list(training_ids),
        'split_manifest_sha256':digest(split_manifest),
        'split_layouts':{key:[{'scene_id':r['scene_id'],'layout_sha256':r['layout_sha256']} for r in rows] for key,rows in split_manifest.items()},
        'validation_selection':selection,'inference':'frozen polynomial ridge matrix prediction; no per-scene optimizer',
        'data_origin':'synthetic','physically_verified':False,'native_paths_checked':False}
    value['checkpoint_id']=digest(value);validate_checkpoint(value);return value


def validate_checkpoint(value):
    if not isinstance(value,dict) or value.get('schema')!=CHECKPOINT_SCHEMA or value.get('profile_id')!=PROFILE_ID:raise ValueError('Front-negative-Y checkpoint required; historical checkpoint is separate')
    clean=copy.deepcopy(value);declared=clean.pop('checkpoint_id',None)
    if digest(clean)!=declared or clean.get('contract')!=study_contract():raise ValueError('Front checkpoint checksum/source/assets/dependencies mismatch')
    if clean.get('data_origin')!='synthetic' or clean.get('physically_verified') is not False:raise ValueError('Synthetic evidence contract required')
    model=clean['model'];degree=model.get('degree');weights=np.asarray(model['weights']);mean=np.asarray(model['mean']);scale=np.asarray(model['scale'])
    if model.get('kind')!='polynomial_ridge' or type(degree) is not int or degree not in (1,2) or finite(model.get('alpha'))<=0:raise ValueError('Invalid frozen model kind or regularization')
    if mean.shape!=(5,) or scale.shape!=(5,) or weights.shape!=(6 if degree==1 else 21,5):raise ValueError('Invalid frozen model shapes')
    if not all(np.isfinite(x).all() for x in (weights,mean,scale)) or np.any(scale<=0):raise ValueError('Invalid frozen model values')
    if set(clean['split_layouts'])!={'train','validation','test'}:raise ValueError('Three split identities required')
    ids=set();layouts=set()
    for rows in clean['split_layouts'].values():
        for row in rows:
            if row['scene_id'] in ids or row['layout_sha256'] in layouts:raise ValueError('Checkpoint split leakage')
            ids.add(row['scene_id']);layouts.add(row['layout_sha256'])
    if not set(clean['training_scene_ids']).issubset({r['scene_id'] for r in clean['split_layouts']['train']}):raise ValueError('Held-out labels cannot train model')
    return value


def predict(value,scene,*,allow_ood=False):
    validate_checkpoint(value);scope=validate_layout(scene,allow_ood=allow_ood)
    action=infer_model(value['model'],scene);selected,release,coords=action_scene(scene,action)
    return {'action':action,'selected_scene':selected,'release_action':release,'release_coordinates':coords,
            'checkpoint_id':value['checkpoint_id'],'inference':value['inference'],'profile_id':PROFILE_ID,'scope':scope}


def preview(value,scene,seed=0,*,allow_ood=False,noise=None):
    proposal=predict(value,scene,allow_ood=allow_ood)
    evaluated=evaluate_action(scene,proposal['action'],seed,frames=True,noise=noise,allow_ood=allow_ood)
    return {**proposal,**evaluated,'preview_schema':'sircus-front-scene-preview-v1','mode':'numerical_prediction',
            'native_replay_status':'blocked: matching v2 native adapter and calibrated collision paths missing'}


def wilson(successes,n):
    if n==0:return [0.,1.]
    z=1.959963984540054;p=successes/n;den=1+z*z/n;center=(p+z*z/(2*n))/den
    margin=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return [max(0.,center-margin),min(1.,center+margin)]


def summarize(rows):
    captures=sum(r['result']['success'] is True for r in rows);outcomes={}
    for row in rows:outcomes[row['result']['outcome']]=outcomes.get(row['result']['outcome'],0)+1
    return {'captures':captures,'episodes':len(rows),'success_fraction':captures/len(rows) if rows else 0.,
        'ci95':wilson(captures,len(rows)),'ci95_method':'Wilson interval over independent sampled layouts',
        'outcomes':outcomes,'endpoint_infeasible':sum(not(r.get('feasibility') and r['feasibility']['arm_reachable'] and r['feasibility']['board_arm_reachable']) for r in rows),
        'native_paths_checked':False,'physically_verified':False}
