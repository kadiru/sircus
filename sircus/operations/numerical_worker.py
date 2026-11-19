"""Bounded one-request numerical worker; no Viam, audio or hardware imports."""
import copy
import json
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[1] / 'two-arm'
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parents[1]))
import numpy as np
import scene_study as study
from sircus.studies import front_study
from sircus.studies.profiles import PROFILE_ID
from sircus.operations.bucket_facing import (search_orientation,constrain_prediction,tilt_bounds,CONTRACT_VERSION)
from shot_rollout import prepare_scene, release_coordinates, rollout, selection_key
from two_arm_kinematics import (board_target, board_mount_rotation, forward_world,
                                inverse_world, feasibility, DOWN)


def board_pose(scene):
    """Reconstruct the impact plane from simulated joints and provisional mount."""
    cup, orientation = board_target(scene)
    ik = inverse_world(scene, 14, cup, orientation)
    if not ik['arm_reachable']:
        raise ValueError('Board pose is outside the simulated arm-14 endpoint workspace')
    fk = forward_world(scene, 14, ik['joints_deg'])
    board_rotation = np.asarray(fk['rotation_matrix']) @ board_mount_rotation(scene)
    local_cup = np.asarray(scene['board']['support_offset_mm']) + [0, 0, -scene['board']['size_mm'][2]/2]
    center = np.asarray(fk['tcp_xyz_mm']) - board_rotation @ local_cup
    surface = center + board_rotation @ [0, 0, scene['board']['size_mm'][2]/2]
    return {'source': 'simulated_joint_kinematics', 'calibrated': False,
        'frame': 'world_sim', 'units': 'mm, degrees', 'joints_deg': ik['joints_deg'],
        'center_mm': center.tolist(), 'rotation_matrix': board_rotation.tolist(),
        'normal': board_rotation[:, 2].tolist(), 'impact_surface_center_mm': surface.tolist(),
        'arm14_boxes': fk['boxes'], 'endpoint_check': ik,
        'target_center_mm': scene['board']['center_mm'],
        'target_rpy_deg': scene['board']['rpy_deg'],
        'reconstruction_position_error_mm': float(np.linalg.norm(center-scene['board']['center_mm'])),
        'scope': 'Simulated endpoint FK check; provisional mount; not measured calibration or a collision-safe path'}


def loading_boxes(scene):
    # Numerical ball-at-TCP proxy only; screenshot loading TCP is not a measured grasp offset.
    pose = inverse_world(scene, 10, scene['pickup_mm'], DOWN)
    if not pose['arm_reachable']:
        return []
    return forward_world(scene, 10, pose['joints_deg'])['boxes']


def evaluate(scene, action, seed, frames=True):
    selected = prepare_scene(scene)
    if scene.get('measurement_status', {}).get('coordinate_profile') == PROFILE_ID:
        selected['simulation'].update(noise_xy_mm=front_study.DEFAULT_NOISE['xy_mm'],
                                      noise_gap_mm=front_study.DEFAULT_NOISE['gap_mm'])
    coords = release_coordinates(selected, action)
    check = feasibility(selected, action)
    if not check['arm_reachable'] or not check['board_arm_reachable']:
        raise ValueError('One or more simulated arm endpoints are unreachable; no drop was executed')
    result = rollout(selected, action, seed=seed, frames=frames)
    return {'selected_scene': selected, 'release_action': copy.deepcopy(action),
        'action': {'roll_deg': scene['board']['rpy_deg'][0], 'pitch_deg': scene['board']['rpy_deg'][1], **action},
        'release_coordinates': coords, 'feasibility': check, 'result': result,
        'seed': seed, 'data_origin': 'synthetic', 'physically_verified': False,
        'native_paths_checked': False, 'scoring_version': study.SCORING_VERSION,
        'physics_version': study.PHYSICS_VERSION}


def annotate(row):
    row['board_pose'] = board_pose(row['selected_scene'])
    release = row['release_coordinates']['commanded_release_ball_center_mm']
    arm = inverse_world(row['selected_scene'], 10, release, DOWN)
    row['arm10_release_boxes'] = forward_world(row['selected_scene'], 10, arm['joints_deg'])['boxes']
    row['arm10_release_joints_deg'] = arm['joints_deg']
    row['native_replay_status'] = 'blocked: matching v2 native adapter and calibrated collision paths are missing'
    return row


def facing_plan(scene,payload):
    mode=payload['mode'];seed=payload['seed'];bounds=tilt_bounds(payload.get('tilt_bounds_deg'))
    if mode=='facing_learned':
        checkpoint=payload['checkpoint']
        engine=front_study if checkpoint.get('schema')==front_study.CHECKPOINT_SCHEMA else study
        # Original asset/layout/source guards apply BEFORE the derived yaw action.
        proposal=engine.predict(checkpoint,payload['scene'])
        selected,action,info=constrain_prediction(proposal,bounds)
        coords=release_coordinates(selected,action)
        error=float(np.linalg.norm(np.asarray(coords['commanded_release_ball_center_mm'])-info['expected_rotated_release_mm']))
        if error>1e-6:raise ValueError('Derived yaw/release coordinate consistency check failed')
        row=evaluate(selected,action,seed)
        if not row['result'].get('executed') or row['result']['outcome'] in ('invalid_action','invalid_contact','model_limit'):
            raise ValueError('Heading-constrained prediction has no valid numerical plan; no search fallback')
        info['release_transform_error_mm']=error
        row.update(checkpoint_id=proposal['checkpoint_id'],raw_prediction=proposal,
            derived_policy_schema='sircus-frozen-heading-constrained-v1',search_budget=0,
            inference='Frozen matrix inference once, then explicit geometric yaw constraint; no per-scene search or retraining')
    else:
        rng=random.Random(seed);budget=payload['budget'];best=None;candidates=[]
        front=scene.get('measurement_status',{}).get('coordinate_profile')==PROFILE_ID
        max_gap=front_study.ACTION_BOUNDS[-1][1] if front else 380.
        action_bounds=[('tilt_deg',*bounds),('u_mm',-70.,70.),('v_mm',-100.,100.),('gap_mm',60.,max_gap)]
        for index in range(budget):
            if index==0:
                proposal={'tilt_deg':max(bounds[0],min(bounds[1],30.)),'u_mm':0.,'v_mm':0.,'gap_mm':min(160.,max_gap)}
            elif best is None or index<max(2,budget//2):
                proposal={key:rng.uniform(low,high) for key,low,high in action_bounds}
            else:
                proposal={key:max(low,min(high,best['candidate_action'][key]+rng.gauss(0,(high-low)*.12))) for key,low,high in action_bounds}
            selected,info=search_orientation(scene,proposal['tilt_deg'],bounds)
            action={key:proposal[key] for key in ('u_mm','v_mm','gap_mm')}
            try:
                candidate=evaluate(selected,action,seed,frames=False);outcome=candidate['result']
                valid=outcome.get('executed') and outcome['outcome'] not in ('invalid_action','invalid_contact','model_limit')
                candidates.append({'action':proposal,'outcome':outcome['outcome'],'success':outcome['success'],'endpoint_feasible':True})
                if valid and (best is None or selection_key(outcome)>selection_key(best['result'])):
                    best={**candidate,'candidate_action':proposal}
            except ValueError as exc:
                candidates.append({'action':proposal,'outcome':'invalid_action','success':False,'reason':str(exc),'endpoint_feasible':False})
        if best is None:raise ValueError('No valid bucket-facing release in bounded search; adjust setup/bounds')
        row=evaluate(best['selected_scene'],best['release_action'],seed)
        _,info=search_orientation(scene,best['candidate_action']['tilt_deg'],bounds)
        row.update(search_budget=budget,candidates=candidates,
            inference='Explicit per-scene tilt/U/V/height numerical search; not frozen learned inference')
    row.update(mode=mode,board_facing=info,facing=copy.deepcopy(info),facing_contract=CONTRACT_VERSION,tilt_bounds_deg=bounds)
    return annotate(row)


def choreography(row,initial_scene):
    """Display-only endpoint sequence followed by actual numerical ball frames."""
    initial_board=board_pose(initial_scene)
    loading=inverse_world(initial_scene,10,initial_scene['pickup_mm'],DOWN)
    if not loading['arm_reachable']:raise ValueError('Mock loading endpoint is unreachable')
    load_boxes=forward_world(initial_scene,10,loading['joints_deg'])['boxes']
    release_t=4.35
    segments=[{'stage':'grip','start_s':0.,'end_s':.35,'gripper_state':'closing'},
        {'stage':'orient_board','start_s':.35,'end_s':1.35,
         'arm14_from_joints_deg':initial_board['joints_deg'],'arm14_to_joints_deg':row['board_pose']['joints_deg'],
         'from_boxes':initial_board['arm14_boxes'],'to_boxes':row['board_pose']['arm14_boxes']},
        {'stage':'move_dropper','start_s':1.35,'end_s':2.35,
         'arm10_from_joints_deg':loading['joints_deg'],'arm10_to_joints_deg':row['arm10_release_joints_deg'],
         'from_boxes':load_boxes,'to_boxes':row['arm10_release_boxes']},
        {'stage':'settle','start_s':2.35,'end_s':release_t,'seconds':2.},
        {'stage':'release','start_s':release_t,'end_s':release_t,'gripper_state':'open'},
        {'stage':'ball_flight','start_s':release_t,'end_s':release_t+row['result'].get('duration_s',0.)}]
    release=row['release_coordinates']['commanded_release_ball_center_mm']
    held=[{'t':t,'ball_xyz_mm':copy.deepcopy(point),'source':'synthetic_endpoint_choreography'}
          for t,point in [(0.,initial_scene['pickup_mm']),(.35,initial_scene['pickup_mm']),
                          (1.35,initial_scene['pickup_mm']),(2.35,release),(release_t,release)]]
    frames=held+[{**copy.deepcopy(frame),'t':release_t+frame['t'],'source':'versioned_numerical_ball_physics'}
                 for frame in row['result'].get('frames',[])]
    return {'schema':'sircus-numerical-shot-choreography-v1','synthetic':True,'native_paths_checked':False,
        'physically_verified':False,'release_time_s':release_t,'segments':segments,'ball_frames':frames,
        'initial_board_pose':initial_board,'initial_arm10_boxes':load_boxes,
        'scope':'Illustrative endpoint interpolation; no collision-validated arm path, grip dynamics or physical timing',
        'physics_time_basis':'Ball physics frames retain original t in result.frames; choreography frames add release_time_s'}


def handle(payload):
    op = payload['op']
    scene = prepare_scene(payload['scene'])
    if op == 'scan':
        bucket = scene['bin']
        return {'board_pose': board_pose(scene), 'arm10_loading_boxes': loading_boxes(scene), 'bucket_observation': {
            'source': 'synthetic_scene_ground_truth', 'frame': 'world_sim', 'calibrated': False,
            'opening_center_mm': [*bucket['center_xy_mm'], bucket['floor_top_mm']+bucket['interior_depth_mm']],
            'normal': [0, 0, 1], 'inner_diameter_mm': bucket['inner_top_diameter_mm'],
            'capture_method': 'simulation fixture; no image, depth, camera movement or physical detection',
            'height_interpretation': bucket.get('height_interpretation', 'unresolved')}}
    if op == 'execute':
        row = evaluate(scene, payload['release_action'], payload['seed'])
        row['mode'] = 'fresh_numerical_trial'
        row=annotate(row)
        if payload.get('choreography') is True:
            row['choreography']=choreography(row,payload['initial_scene'])
        return row
    if op != 'plan':
        raise ValueError('Unknown numerical operation')
    seed, mode = payload['seed'], payload['mode']
    if mode in ('facing_search','facing_learned'):return facing_plan(scene,payload)
    if mode == 'learned_board':
        # Prediction is frozen matrix inference; evaluation does not alter weights/actions.
        checkpoint = payload['checkpoint']
        engine = front_study if checkpoint.get('schema') == front_study.CHECKPOINT_SCHEMA else study
        row = engine.preview(checkpoint, payload['scene'], seed=seed)
        if not row['result'].get('executed') or row['result']['outcome'] in ('invalid_action', 'invalid_contact', 'model_limit'):
            raise ValueError('Frozen prediction has no valid numerical plan: '+row['result']['outcome'])
        row['mode'] = 'learned_board'
        return annotate(row)
    if mode != 'fixed_board':
        raise ValueError('Unknown planning mode')
    front = scene.get('measurement_status', {}).get('coordinate_profile') == PROFILE_ID
    fixed_action = front_study.FIXED_ACTION if front else study.FIXED_ACTION
    max_gap = front_study.ACTION_BOUNDS[-1][1] if front else 380
    rng = random.Random(seed)
    budget = payload['budget']
    best = None
    candidates = []
    for index in range(budget):
        if index == 0:
            action = {k: fixed_action[k] for k in ('u_mm', 'v_mm', 'gap_mm')}
        elif best is None or index < max(2, budget//2):
            action = {'u_mm': rng.uniform(-70, 70), 'v_mm': rng.uniform(-100, 100), 'gap_mm': rng.uniform(60, max_gap)}
        else:
            action = {key: max(low, min(high, best['release_action'][key]+rng.gauss(0, (high-low)*.12)))
                      for key, low, high in [('u_mm', -70, 70), ('v_mm', -100, 100), ('gap_mm', 60, max_gap)]}
        try:
            row = evaluate(scene, action, seed, frames=False)
            result = row['result']
            valid = result.get('executed') and result['outcome'] not in ('invalid_action', 'invalid_contact', 'model_limit')
            candidates.append({'action': action, 'outcome': result['outcome'], 'success': result['success']})
            if valid and (best is None or selection_key(result) > selection_key(best['result'])):
                best = row
        except ValueError as exc:
            candidates.append({'action': action, 'outcome': 'invalid_action', 'reason': str(exc), 'success': False})
    if best is None:
        raise ValueError('No valid fixed-board release found; adjust board/bucket placement')
    row = evaluate(scene, best['release_action'], seed, frames=True)
    row.update(mode='fixed_board', inference='explicit per-scene U/V/height search; not frozen learned inference',
               search_budget=budget, candidates=candidates)
    return annotate(row)


def main():
    try:
        raw = sys.stdin.buffer.read(1_000_001)
        if len(raw) > 1_000_000:
            raise ValueError('Numerical request too large')
        value = json.loads(raw, parse_constant=lambda _: (_ for _ in ()).throw(ValueError('Finite JSON required')))
        result = handle(value)
        print(json.dumps(result, allow_nan=False))
        return 0
    except (ValueError, KeyError, TypeError) as exc:
        print(json.dumps({'error': str(exc)[:400]}))
        return 2
    except Exception:
        print(json.dumps({'error': 'Numerical rehearsal failed; no hardware runtime was used'}))
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
