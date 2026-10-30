"""Headless one-shot contract for orientation/height v2; no robot connections."""
import copy
import hashlib
import json
import math
from pathlib import Path

from scipy.optimize import brentq, minimize_scalar
from capture_scoring import PHYSICS_VERSION, SCORING_VERSION, SUCCESS_DEFINITION
from physics import BucketContactError, release_point, rotation, simulate, transposevec, validate_config
from two_arm_kinematics import DOWN

ROOT = Path(__file__).resolve().parent
RESULT_SCHEMA = 'sircus-shot-result-v1'
SETTINGS = {'dt_s':.00025, 'entry_deadline_s':4., 'retention_s':1.,
            'duration_s':5.00025, 'bucket_restitution':.35,
            'bucket_projection_limit_mm':2., 'containment_roundoff_mm':1e-6,
            'launch_velocity_m_s':[0.,0.,0.]}
COORDINATES = {
    'frame':'world; Z upward', 'position_units':'mm', 'angle_units':'degrees',
    'mass_units':'kg', 'time_units':'s', 'height_units':'m',
    'board_rotation':'Rz(yaw) Ry(pitch) Rx(roll)',
    'uv_definition':'board-local top-face U/V in mm; not world XY',
    'gap_definition':'world-Z mm above R*[U,V,thickness/2+radius]+board_center',
    'height_definition':'h_m = (release ball-center world Z - nominal first finite-board-contact ball-center world Z)/1000',
    'ball_to_tool':{'translation_mm':[0.,0.,0.], 'rotation_matrix':[[1,0,0],[0,1,0],[0,0,1]],
                    'status':'provisional ball center at simulated gripper TCP; uncalibrated'},
    'tcp_world_rotation_matrix':DOWN.tolist(),
}


def canonical(value):
    """Numeric normalization only for this NEW contract; old hashes stay intact."""
    def normalize(v):
        if isinstance(v, bool) or v is None or isinstance(v, str): return v
        if isinstance(v, (int, float)):
            if not math.isfinite(v): raise ValueError('Non-finite contract number')
            return 0. if v == 0 else float(v)
        if isinstance(v, dict): return {k:normalize(x) for k,x in v.items()}
        if isinstance(v, (list, tuple)): return [normalize(x) for x in v]
        raise ValueError('Unsupported contract value')
    return json.dumps(normalize(value), sort_keys=True, separators=(',',':'), allow_nan=False).encode()


def contract_manifest(scene):
    sources = ('physics.py','capture_scoring.py','shot_rollout.py','train_orientation_height.py',
               'bucket_geometry.py','kinematics.py','two_arm_kinematics.py',
               'models/xarm850_kinematics.json','models/gripper-proxy.json','models/suction-proxy.json',
               '../simulation/requirements-viam.txt')
    return {'fingerprint_version':'sircus-shot-contract-v1', 'result_schema':RESULT_SCHEMA,
            'scoring_version':SCORING_VERSION, 'physics_version':PHYSICS_VERSION,
            'success_definition':SUCCESS_DEFINITION, 'settings':copy.deepcopy(SETTINGS),
            'coordinates':copy.deepcopy(COORDINATES),
            'scene_sha256':hashlib.sha256(canonical(scene)).hexdigest(),
            'source_sha256':{name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in sources},
            'dependency_pins':(ROOT/'../simulation/requirements-viam.txt').read_text().splitlines()}


def fingerprint(scene):
    return hashlib.sha256(canonical(contract_manifest(scene))).hexdigest()


def first_contact(config, xy):
    """Highest vertical ray intersection with finite OBB dilated by sphere radius."""
    board = config['board']; radius = config['ball']['diameter_mm']/2
    rot = rotation(board['rpy_deg']); center = board['center_mm']; half = [v/2 for v in board['size_mm']]
    def distance(z):
        local = transposevec(rot, [xy[0]-center[0],xy[1]-center[1],z-center[2]])
        return math.sqrt(sum(max(abs(x)-h,0)**2 for x,h in zip(local,half)))-radius
    extent = sum(abs(rot[2][i])*half[i] for i in range(3))+radius+1
    minimum = minimize_scalar(distance, bounds=(center[2]-extent,center[2]+extent), method='bounded')
    if minimum.fun > 1e-7: return None
    z = brentq(distance,minimum.x,center[2]+extent) if minimum.fun < 0 else minimum.x
    return [*xy,float(z)]


def release_coordinates(scene, action):
    release = release_point(scene, action)
    contact = first_contact(scene, release[:2])
    if contact is None: raise ValueError('Release line has no finite-board contact reference')
    return {'commanded_release_ball_center_mm':release, 'nominal_contact_ball_center_mm':contact,
            'h_m':(release[2]-contact[2])/1000, 'gap_mm':float(action['gap_mm']),
            'u_mm':float(action['u_mm']), 'v_mm':float(action['v_mm']),
            'coordinate_contract':copy.deepcopy(COORDINATES)}


def prepare_scene(scene):
    config = copy.deepcopy(scene)
    if config.get('bin',{}).get('shape') != 'tapered_bucket':
        raise ValueError('Capture v1 supports only an upright tapered bucket')
    if any(k in config['bin'] for k in ('rpy_deg','rotation','orientation')):
        raise ValueError('Bucket orientation fields are not supported by this version')
    config['simulation'].update(dt_s=SETTINGS['dt_s'], duration_s=SETTINGS['duration_s'])
    config['capture_physics'] = copy.deepcopy(SETTINGS)
    return validate_config(config)


def failure_result(outcome, reason, *, executed=False):
    return {'schema':RESULT_SCHEMA, 'scoring_version':SCORING_VERSION, 'physics_version':PHYSICS_VERSION,
            'episode_kind':'complete_shot', 'execution_scope':'headless_endpoint_ik',
            'success':False, 'outcome':outcome, 'reason':str(reason), 'executed':executed,
            'truncated':False, 'whole_ball_entry':False, 'capture_class':None,
            'actual_h_m':None, 'opening_clearance_mm':None, 'signed_miss_xy_mm':None,
            'shaping_reward':-1000., 'physically_verified':False,
            'success_definition':SUCCESS_DEFINITION}


def rollout(scene, action, seed=0, frames=False):
    try:
        config = prepare_scene(scene)
        coords = release_coordinates(config, action)
        result = simulate(config, action, seed=seed, frames=frames, capture_mode=SCORING_VERSION)
    except BucketContactError as exc:
        return failure_result('model_limit', exc, executed=True)
    except (ValueError, KeyError, TypeError) as exc:
        return failure_result('invalid_action', exc)
    result.update(schema=RESULT_SCHEMA, executed=True, execution_scope='headless_endpoint_ik',
                  effective_physics=copy.deepcopy(SETTINGS), **coords)
    return result


def selection_key(result):
    if (result.get('scoring_version') != SCORING_VERSION or
            result.get('physics_version') != PHYSICS_VERSION or 'success' not in result):
        raise ValueError('Selection requires the versioned capture result, never legacy retained')
    return (result['success'] is True, float(result['shaping_reward']))
