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
        return annotate(row)
    if op != 'plan':
        raise ValueError('Unknown numerical operation')
    seed, mode = payload['seed'], payload['mode']
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
