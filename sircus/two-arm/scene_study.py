"""Seeded contextual one-shot study; pure numerical inference, never robot I/O.

The small ridge predictor imitates training-scene search labels. It is not RL,
online adaptation, or a guarantee of capture. Native v2 replay stays separate.
"""
import copy
import hashlib
import json
import math
from pathlib import Path
import random

import numpy as np
import scipy

from capture_scoring import PHYSICS_VERSION, SCORING_VERSION
from physics import validate_config
from shot_rollout import (canonical, contract_manifest, failure_result, prepare_scene,
                          release_coordinates, rollout, selection_key)
from two_arm_kinematics import feasibility

ROOT = Path(__file__).resolve().parent
STUDY_SCHEMA = 'sircus-randomized-layout-study-v1'
CHECKPOINT_SCHEMA = 'sircus-scene-conditioned-ridge-v1'
ACTION_NAMES = ('roll_deg', 'pitch_deg', 'u_mm', 'v_mm', 'gap_mm')
OBSERVATION_NAMES = ('board_x_mm', 'board_y_mm', 'board_z_mm',
                     'bucket_dx_mm', 'bucket_dy_mm')
ACTION_BOUNDS = ((-20., 20.), (10., 50.), (-70., 70.), (-100., 100.), (60., 380.))
FIXED_ACTION = dict(zip(ACTION_NAMES, (0., 30., -16.408, -44.821, 161.715)))
DISTRIBUTION = {
    'schema': 'sircus-upright-layout-distribution-v1',
    'board_center_mm': [[570., 625.], [-25., 25.], [330., 375.]],
    'bucket_relative_xy_mm': [[40., 110.], [200., 275.]],
    'variable_fields': ['board.center_mm', 'bin.center_xy_mm'],
    'fixed_fields': 'all assets/contact parameters, bucket floor/geometry, board yaw, arm bases and tools',
    'scope': 'narrow synthetic upright-bucket layouts; no physical or out-of-distribution claim',
}
NOISE = {'release_xy_mm': 2., 'release_gap_mm': 3.}
SELECTION = 'lexicographic versioned capture success, then continuous shaping'
BASELINE_PROVENANCE = 'predeclared historical nominal action; unchanged across every layout'


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def _finite(value, name):
    if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value):
        raise ValueError(f'{name} must be finite numeric')
    return float(value)


def _integer(value, name, low, high):
    if type(value) is not int or not low <= value <= high:
        raise ValueError(f'{name} must be an integer in {low}..{high}')
    return value


def stream_seed(seed, namespace, index=0):
    _integer(seed, 'seed', 0, 2**31-1)
    # physics.simulate accepts uint32 seeds; separate namespaces retain
    # independence without creating out-of-range episode seeds.
    return int(hashlib.sha256(f'{seed}/{namespace}/{index}'.encode()).hexdigest()[:8], 16)


def asset_template(scene):
    """Mask only layout/action fields; measured/provisional assets still hash."""
    scene = copy.deepcopy(scene)
    scene.pop('scene_id', None)
    scene['board']['center_mm'] = None
    scene['board']['rpy_deg'][:2] = [None, None]
    scene['bin']['center_xy_mm'] = None
    return scene


def observation(scene):
    board = scene['board']['center_mm']
    bucket = scene['bin']['center_xy_mm']
    values = [*board, bucket[0]-board[0], bucket[1]-board[1]]
    return [_finite(value, name) for name, value in zip(OBSERVATION_NAMES, values)]


def layout_hash(scene):
    return digest(observation(scene))


def validate_layout(scene, base_scene):
    prepare_scene(scene)  # Includes unsupported bucket orientation rejection.
    if asset_template(scene) != asset_template(base_scene):
        raise ValueError('Study asset/contact/tool/geometry mismatch; only declared layout fields may vary')
    bounds = DISTRIBUTION['board_center_mm'] + DISTRIBUTION['bucket_relative_xy_mm']
    for value, limits, name in zip(observation(scene), bounds, OBSERVATION_NAMES):
        if not limits[0] <= value <= limits[1]:
            raise ValueError(f'Layout outside trained distribution: {name}')
    return scene


def study_contract(base_scene):
    base = contract_manifest(base_scene)
    sources = dict(base['source_sha256'])
    for name in ('scene_study.py', 'train_scene_study.py'):
        sources[name] = hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
    return {
        'schema': STUDY_SCHEMA, 'checkpoint_schema': CHECKPOINT_SCHEMA,
        'scoring_version': SCORING_VERSION, 'physics_version': PHYSICS_VERSION,
        'distribution': copy.deepcopy(DISTRIBUTION),
        'observation_names': list(OBSERVATION_NAMES), 'observation_units': ['mm']*5,
        'action_names': list(ACTION_NAMES), 'action_bounds': [list(x) for x in ACTION_BOUNDS],
        'action_units': ['degrees', 'degrees', 'mm', 'mm', 'mm'],
        'action_semantics': 'Rz(fixed yaw) Ry(pitch) Rx(roll); board-local U/V; gap is world-Z clearance above nominal sphere/top-face contact plane',
        'height_semantics': 'release_coordinates separately records finite-board first-contact h_m; gap need not equal h_m near edges',
        'asset_sha256': digest(asset_template(base_scene)),
        'physics_settings': base['settings'], 'coordinates': base['coordinates'],
        'source_sha256': sources, 'dependency_pins': base['dependency_pins'],
        'actual_packages': {'numpy': np.__version__, 'scipy': scipy.__version__},
        'release_noise': copy.deepcopy(NOISE),
        'domain_randomization': 'layout and release noise only; contact parameters are fixed provisional assumptions',
    }


def sample_splits(base_scene, seed=20261030, train_count=24, validation_count=6, test_count=8):
    """Serialize layouts before searches; split RNGs and geometry hashes disjoint."""
    _integer(seed, 'seed', 0, 2**31-1)
    for name, count in [('train', train_count), ('validation', validation_count), ('test', test_count)]:
        _integer(count, name+' count', 1, 10000)
    validate_layout(base_scene, base_scene)
    used = set()
    splits = {}
    for name, count in [('train', train_count), ('validation', validation_count), ('test', test_count)]:
        scene_seed = stream_seed(seed, name+'/layout')
        rng = random.Random(scene_seed)
        episodes = []
        for index in range(count):
            for _ in range(100):
                scene = copy.deepcopy(base_scene)
                center = [rng.uniform(*bounds) for bounds in DISTRIBUTION['board_center_mm']]
                delta = [rng.uniform(*bounds) for bounds in DISTRIBUTION['bucket_relative_xy_mm']]
                scene['board']['center_mm'] = center
                scene['bin']['center_xy_mm'] = [center[i]+delta[i] for i in (0, 1)]
                geometry = layout_hash(scene)
                if geometry not in used:
                    validate_layout(scene, base_scene)
                    break
            else:
                raise ValueError('Could not sample a unique valid layout')
            used.add(geometry)
            scene_id = f'{name}-{index:04d}-{geometry[:12]}'
            scene['scene_id'] = scene_id
            episodes.append({'scene_id': scene_id, 'layout_sha256': geometry, 'split': name,
                'layout_rng_seed': scene_seed, 'action_rng_seed': stream_seed(seed, name+'/action', index),
                'rollout_seed': stream_seed(seed, name+'/rollout', index),
                'scene': scene, 'observation': observation(scene)})
        splits[name] = episodes
    manifest = {'schema': STUDY_SCHEMA, 'seed': seed, 'contract': study_contract(base_scene),
                'splits': splits, 'data_origin': 'synthetic', 'physically_verified': False,
                'layout_feasibility': 'layouts are geometric-valid; each proposed action gets independent endpoint IK'}
    manifest['manifest_sha256'] = digest(manifest)
    validate_manifest(manifest)
    return manifest


def validate_manifest(manifest):
    value = copy.deepcopy(manifest)
    declared = value.pop('manifest_sha256', None)
    if value.get('schema') != STUDY_SCHEMA or digest(value) != declared:
        raise ValueError('Study manifest schema/checksum mismatch')
    seen_ids, seen_geometry = set(), set()
    if set(value['splits']) != {'train', 'validation', 'test'}:
        raise ValueError('Require train/validation/test splits')
    base = value['splits']['train'][0]['scene'] if value['splits']['train'] else None
    if base is None or value['contract'] != study_contract(base):
        raise ValueError('Manifest source/assets/distribution contract mismatch')
    for split, episodes in value['splits'].items():
        if not episodes:
            raise ValueError('Empty study split')
        for episode in episodes:
            validate_layout(episode['scene'], base)
            geometry = layout_hash(episode['scene'])
            if episode['split'] != split or episode['layout_sha256'] != geometry:
                raise ValueError('Episode split/geometry mismatch')
            if episode['scene_id'] in seen_ids or geometry in seen_geometry:
                raise ValueError('Train/validation/test layouts must be disjoint')
            if episode['observation'] != observation(episode['scene']):
                raise ValueError('Episode observation must come from scene geometry only')
            if episode['scene']['scene_id'] != episode['scene_id']:
                raise ValueError('Episode scene identity mismatch')
            for name in ('layout_rng_seed', 'action_rng_seed', 'rollout_seed'):
                _integer(episode[name], name, 0, 2**32-1)
            seen_ids.add(episode['scene_id']); seen_geometry.add(geometry)
    return manifest


def action_scene(scene, action):
    if not isinstance(action, dict) or set(action) != set(ACTION_NAMES):
        raise ValueError('Require all five explicit action fields, no hidden dimensions')
    action = {name: _finite(action[name], name) for name in ACTION_NAMES}
    for name, (low, high) in zip(ACTION_NAMES, ACTION_BOUNDS):
        if not low <= action[name] <= high:
            raise ValueError('Action outside bounds: '+name)
    selected = copy.deepcopy(scene)
    selected['board']['rpy_deg'][:2] = [action['roll_deg'], action['pitch_deg']]
    validate_config(selected)
    release = {name: action[name] for name in ('u_mm', 'v_mm', 'gap_mm')}
    coords = release_coordinates(selected, release)
    return selected, release, coords


def evaluate_action(scene, action, seed=0, *, frames=False):
    """Same numerical evaluator for teachers, model, fixed baseline and oracle."""
    selected = release = coords = check = None
    try:
        selected, release, coords = action_scene(scene, action)
        selected['simulation'].update(noise_xy_mm=NOISE['release_xy_mm'], noise_gap_mm=NOISE['release_gap_mm'])
        check = feasibility(selected, release)
        if check['arm_reachable'] and check['board_arm_reachable']:
            result = rollout(selected, release, seed=seed, frames=frames)
        else:
            result = failure_result('invalid_action', 'unreachable arm endpoint')
        selection_key(result)
    except (ValueError, KeyError, TypeError) as exc:
        result = failure_result('invalid_action', str(exc))
    return {'action': copy.deepcopy(action), 'selected_scene': selected,
            'release_action': release, 'release_coordinates': coords, 'seed': seed,
            'feasibility': check, 'result': result,
            'scoring_version': SCORING_VERSION, 'physics_version': PHYSICS_VERSION,
            'data_origin': 'synthetic', 'physically_verified': False,
            'native_paths_checked': False}


def valid_teacher(row):
    result = row['result']
    return bool(result.get('executed')) and result['outcome'] not in ('invalid_action', 'invalid_contact', 'model_limit')


def search_episode(episode, budget=12, *, role='training_teacher'):
    """Explicit optimizer baseline/teacher; NEVER called by predict or preview."""
    _integer(budget, 'search budget', 1, 1000)
    if role == 'training_teacher' and episode['split'] != 'train':
        raise ValueError('Training teacher cannot use held-out layouts')
    if role not in ('training_teacher', 'held_out_oracle'):
        raise ValueError('Explicit search role required')
    rng = random.Random(episode['action_rng_seed'])
    rows, best = [], None
    for index in range(budget):
        if index == 0:
            action = copy.deepcopy(FIXED_ACTION)
        elif best is None or index < max(2, budget//2):
            action = {name: rng.uniform(*bounds) for name, bounds in zip(ACTION_NAMES, ACTION_BOUNDS)}
        else:
            scale = max(.08, .3*(1-index/budget))
            action = {name: max(low, min(high, best['action'][name]+rng.gauss(0, (high-low)*scale)))
                      for name, (low, high) in zip(ACTION_NAMES, ACTION_BOUNDS)}
        row = evaluate_action(episode['scene'], action, episode['rollout_seed'])
        row.update(scene_id=episode['scene_id'], split=episode['split'],
                   candidate=index, role=role, layout_sha256=episode['layout_sha256'])
        rows.append(row)
        if valid_teacher(row) and (best is None or selection_key(row['result']) > selection_key(best['result'])):
            best = row
    return rows, best


def fit_ridge(observations, actions, regularization):
    """Fit actual scene-conditioned weights; statistics only use supplied train rows."""
    x = np.asarray(observations, float); y = np.asarray(actions, float)
    if x.ndim != 2 or x.shape[1] != 5 or y.shape != (len(x), 5) or len(x) < 4:
        raise ValueError('At least four training labels with five observations/actions required')
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError('Finite training rows required')
    alpha = _finite(regularization, 'regularization')
    if alpha <= 0:
        raise ValueError('Positive ridge regularization required')
    mean = x.mean(axis=0); scale = x.std(axis=0)
    scale = np.where(scale < 1e-9, 1., scale)
    design = np.column_stack([np.ones(len(x)), (x-mean)/scale])
    bounds = np.array(ACTION_BOUNDS); low, high = bounds[:, 0], bounds[:, 1]
    normalized = 2*(y-low)/(high-low)-1
    penalty = np.diag([0., *([alpha]*5)])
    weights = np.linalg.solve(design.T@design+penalty, design.T@normalized)
    return {'kind': 'linear_ridge_action_regression', 'regularization': alpha,
            'observation_mean': mean.tolist(), 'observation_scale': scale.tolist(),
            'weights': weights.tolist(), 'feature_order': ['bias', *OBSERVATION_NAMES],
            'output_order': list(ACTION_NAMES), 'output_transform': 'clip normalized [-1,1], affine to action bounds'}


def infer_model(model, scene):
    """One matrix product only. No optimizer, simulator, IK, or test feedback."""
    x = np.asarray(observation(scene), float)
    mean = np.asarray(model['observation_mean'], float)
    scale = np.asarray(model['observation_scale'], float)
    weights = np.asarray(model['weights'], float)
    normalized = np.r_[1., (x-mean)/scale] @ weights
    clipped = np.clip(normalized, -1., 1.)
    bounds = np.array(ACTION_BOUNDS); low, high = bounds[:, 0], bounds[:, 1]
    values = low + (clipped+1)*(high-low)/2
    return dict(zip(ACTION_NAMES, values.tolist()))


def checkpoint(base_scene, manifest, model, *, training_scene_ids, validation_selection):
    value = {'schema': CHECKPOINT_SCHEMA, 'base_scene': copy.deepcopy(base_scene),
        'contract': study_contract(base_scene), 'manifest_sha256': manifest['manifest_sha256'],
        'split_layouts': {split: [{'scene_id': x['scene_id'], 'layout_sha256': x['layout_sha256']}
                                 for x in episodes] for split, episodes in manifest['splits'].items()},
        'training_scene_ids': list(training_scene_ids), 'model': copy.deepcopy(model),
        'validation_selection': copy.deepcopy(validation_selection),
        'inference': 'frozen direct ridge prediction; no per-scene search or test adaptation',
        'data_origin': 'synthetic', 'physically_verified': False, 'native_paths_checked': False}
    value['weights_sha256'] = digest(value['model'])
    value['checkpoint_id'] = digest(value)
    validate_checkpoint(value)
    return value


def validate_checkpoint(value):
    if not isinstance(value, dict) or value.get('schema') != CHECKPOINT_SCHEMA:
        raise ValueError('Unsupported scene-conditioned checkpoint schema')
    saved = copy.deepcopy(value)
    identity = saved.pop('checkpoint_id', None)
    if digest(saved) != identity or value.get('weights_sha256') != digest(value['model']):
        raise ValueError('Checkpoint/weights checksum mismatch')
    base = value['base_scene']
    validate_layout(base, base)
    if value['contract'] != study_contract(base):
        raise ValueError('Checkpoint asset/action/scoring/source/dependency/distribution mismatch')
    model = value['model']
    if (model.get('kind') != 'linear_ridge_action_regression' or
            model.get('feature_order') != ['bias', *OBSERVATION_NAMES] or
            model.get('output_order') != list(ACTION_NAMES) or
            model.get('output_transform') != 'clip normalized [-1,1], affine to action bounds' or
            _finite(model.get('regularization'), 'regularization') <= 0):
        raise ValueError('Unsupported inference model/feature ordering')
    for name, shape in [('weights', (6, 5)), ('observation_mean', (5,)), ('observation_scale', (5,))]:
        arr = np.asarray(model[name], float)
        if arr.shape != shape or not np.isfinite(arr).all():
            raise ValueError('Invalid model array: '+name)
    if any(v <= 0 for v in model['observation_scale']):
        raise ValueError('Normalization scales must be positive')
    ids, geometries = set(), set()
    if set(value['split_layouts']) != {'train', 'validation', 'test'}:
        raise ValueError('Checkpoint must identify three disjoint splits')
    for rows in value['split_layouts'].values():
        for row in rows:
            if row['scene_id'] in ids or row['layout_sha256'] in geometries:
                raise ValueError('Checkpoint splits overlap')
            ids.add(row['scene_id']); geometries.add(row['layout_sha256'])
    train_ids = {x['scene_id'] for x in value['split_layouts']['train']}
    if not set(value['training_scene_ids']).issubset(train_ids):
        raise ValueError('Held-out scene used as a training label')
    return value


def predict(value, scene):
    validate_checkpoint(value)
    validate_layout(scene, value['base_scene'])
    action = infer_model(value['model'], scene)
    selected, release, coords = action_scene(scene, action)
    return {'action': action, 'selected_scene': selected, 'release_action': release,
            'release_coordinates': coords, 'checkpoint_id': value['checkpoint_id'],
            'inference': value['inference']}


def preview(value, scene, seed=0):
    """Frozen inference and numerical trajectory; no native/physical connection."""
    proposal = predict(value, scene)
    evaluated = evaluate_action(scene, proposal['action'], seed, frames=True)
    return {**proposal, **evaluated, 'preview_schema': 'sircus-scene-policy-preview-v1',
            'mode': 'numerical_prediction', 'native_replay_status':
            'blocked: matching v2 native runtime adapter not implemented'}


def summarize(rows):
    results = [r['result'] for r in rows]
    outcomes = sorted({r['outcome'] for r in results})
    valid = [r for r in rows if valid_teacher(r)]
    errors = [math.hypot(*r['signed_miss_xy_mm']) for r in results if r.get('signed_miss_xy_mm') is not None]
    return {'episodes': len(rows), 'captures': sum(r['success'] is True for r in results),
            'success_fraction': sum(r['success'] is True for r in results)/len(rows) if rows else None,
            'endpoint_feasible': sum(bool(r['feasibility'] and r['feasibility']['arm_reachable'] and r['feasibility']['board_arm_reachable']) for r in rows),
            'invalid_actions': sum(r['outcome'] == 'invalid_action' for r in results),
            'model_limits': sum(r['outcome'] == 'model_limit' for r in results),
            'outcomes': {name: sum(r['outcome'] == name for r in results) for name in outcomes},
            'capture_classes': {name: sum(r.get('capture_class') == name for r in results) for name in ('clean', 'rim_assisted')},
            'mean_shaping_valid': float(np.mean([r['result']['shaping_reward'] for r in valid])) if valid else None,
            'opening_radial_error_mm': {'observations': len(errors), 'mean': float(np.mean(errors)) if errors else None,
                'median': float(np.median(errors)) if errors else None, 'max': max(errors) if errors else None},
            'native_paths_checked': False, 'physically_verified': False}
