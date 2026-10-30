"""Local 850 kinematic preview, using the pinned Viam model.

Interpolated joints are for visualization only. Viam's motion service separately
plans collision-aware paths. This module has no hardware or network client.
"""
import json
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

MODEL = json.loads((Path(__file__).parent / 'models/xarm850_kinematics.json').read_text())
JOINTS = MODEL['joints']
LINKS = MODEL['links']
LOW = np.radians([j['min'] for j in JOINTS])
HIGH = np.radians([j['max'] for j in JOINTS])
TOOL_MM = 100.0  # provisional flange-to-ball-center, not a hardware calibration
DOWN = np.diag([-1., 1., -1.])
HOME = np.radians([0, -35, -70, 0, 15, 0])


def vector(d):
    return np.array([d.get(k, 0.) for k in ('x', 'y', 'z')], dtype=float)


def fixed_frame(item):
    t = np.eye(4)
    t[:3, 3] = vector(item.get('translation', {}))
    orientation = item.get('orientation', {})
    if orientation:
        if orientation['type'] != 'euler_angles':
            raise ValueError('Unsupported model orientation')
        a = orientation['value']
        t[:3, :3] = Rotation.from_euler('xyz', [a[k] for k in ('roll', 'pitch', 'yaw')]).as_matrix()
    return t


def forward(q, include_boxes=True):
    frames = {'world': np.eye(4)}
    frames['link_base'] = fixed_frame(LINKS[0])
    boxes = []
    # The RDK link geometry is expressed in the link's parent frame.
    if include_boxes:
        boxes.append(box(LINKS[0], frames['world']))
    for i, joint in enumerate(JOINTS):
        turn = np.eye(4)
        turn[:3, :3] = Rotation.from_rotvec(vector(joint['axis']) * q[i]).as_matrix()
        frames[joint['id']] = frames[joint['parent']] @ turn
        link = LINKS[i + 1]
        parent = frames[link['parent']]
        frames[link['id']] = parent @ fixed_frame(link)
        if include_boxes:
            boxes.append(box(link, parent))
    tip = frames['link_6'].copy()
    flange = tip.copy()
    tip[:3, 3] += tip[:3, :3] @ np.array([0, 0, TOOL_MM])
    if include_boxes:
        g = flange[:3, 3] + flange[:3, :3] @ [0, 0, 55]
        boxes.append({'name': 'gripper-10-provisional', 'center_mm': g.tolist(),
                      'size_mm': [60, 60, 80], 'rotation_matrix': flange[:3, :3].tolist()})
    return {'boxes': boxes, 'joints_deg': np.degrees(q).tolist(),
            'tcp_xyz_mm': tip[:3, 3].tolist(), 'rotation_matrix': tip[:3, :3].tolist(),
            'model': 'UFACTORY 850', 'tool_offset_mm': TOOL_MM,
            'path_validation': 'kinematic interpolation; collision planning is in Viam'}


def box(link, parent):
    g = link['geometry']
    local = fixed_frame(g)
    pose = parent @ local
    return {'name': link['id'], 'center_mm': pose[:3, 3].tolist(),
            'size_mm': [g[k] for k in ('x', 'y', 'z')],
            'rotation_matrix': pose[:3, :3].tolist()}


def inverse(xyz, seed=None):
    target = np.asarray(xyz, dtype=float)
    def residual(q):
        f = forward(q, False)
        r = np.array(f['rotation_matrix'])
        return np.r_[np.array(f['tcp_xyz_mm']) - target,
                     120 * Rotation.from_matrix(DOWN.T @ r).as_rotvec()]
    guesses = [HOME if seed is None else seed, HOME,
               np.radians([0, 20, -120, 0, -20, 0]),
               np.radians([40, -60, -80, 20, 30, 10])]
    best = None
    for initial in guesses:
        fit = least_squares(residual, np.clip(initial, LOW + 1e-5, HIGH - 1e-5),
                            bounds=(LOW, HIGH), max_nfev=160, ftol=1e-8, xtol=1e-8, gtol=1e-8)
        error = np.linalg.norm(residual(fit.x)[:3])
        orientation = np.linalg.norm(residual(fit.x)[3:]) / 120
        if best is None or error < best[1]:
            best = (fit.x, error, orientation)
        if error < .5 and orientation < .01:
            return fit.x, error, orientation
    return best


@lru_cache(maxsize=128)
def _sequence(pick_tuple, release_tuple):
    pick, release = np.array(pick_tuple), np.array(release_tuple)
    approach = pick + [0, 0, 150]
    lift = pick + [0, 0, max(200, release[2] - pick[2] + 60)]
    above = release + [0, 0, 60]
    targets = [('approach', approach, True), ('grasp', pick, True),
               ('lift', lift, False), ('transfer', above, False), ('release', release, False)]
    q = HOME.copy()
    frames, checks, t = [], [], 0.
    carried = False
    for phase, target, opened in targets:
        nxt, error, rotation_error = inverse(target, q)
        checks.append({'phase': phase, 'target_mm': target.tolist(), 'position_error_mm': error,
                       'orientation_error_deg': float(np.degrees(rotation_error))})
        if error > 2 or rotation_error > .02:
            return {'arm_reachable': False, 'arm_sequence': [], 'arm_checks': checks,
                    'arm_note': 'An arm target is unreachable; physics result is independent of reachability.'}
        for fraction in np.linspace(0, 1, 13):
            current = forward(q + (nxt - q) * fraction)
            current.update(t=t, phase=phase, gripper_open=opened,
                           ball_xyz_mm=current['tcp_xyz_mm'] if carried else pick.tolist())
            frames.append(current)
            t += .065
        q = nxt
        if phase == 'grasp':
            carried = True
    frames[-1]['gripper_open'] = True
    return {'arm_reachable': True, 'arm_sequence': frames, 'arm_checks': checks,
            'arm_note': 'Official 850 kinematics; modeled grasp; local path is not collision checked.'}


def sequence(config, release):
    table = config['table']
    radius = config['ball']['diameter_mm'] / 2
    pick = config.get('pickup_mm', [350., -220., table['center_mm'][2] + table['size_mm'][2] / 2 + radius])
    return _sequence(tuple(pick), tuple(release))


def home():
    return forward(HOME)
