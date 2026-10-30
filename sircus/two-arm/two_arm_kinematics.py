"""World-frame IK and collision boxes for two official 850 fake-arm models."""
import copy
import math
from functools import lru_cache
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from kinematics import forward, LOW, HIGH, HOME, DOWN
from physics import release_point


def base_transform(config, arm_id):
    if arm_id == 10:
        return np.eye(3), np.zeros(3)
    robot = config['robot14']
    return Rotation.from_euler('z',robot['base_yaw_deg'],degrees=True).as_matrix(), np.array(robot['base_translation_mm'],float)


def board_mount_rotation(config):
    """Board frame relative to suction TCP; yaw rotates in the board plane."""
    angle=config['robot14'].get('board_mount_yaw_deg',0)
    if isinstance(angle,bool) or not math.isfinite(float(angle)):
        raise ValueError('Board mount yaw must be finite degrees')
    return Rotation.from_euler('z',float(angle),degrees=True).as_matrix()


def board_target(config):
    board=config['board']
    r=Rotation.from_euler('xyz',board['rpy_deg'],degrees=True).as_matrix()
    # support_offset is pivot offset from board center; cup touches underside.
    local=np.array(board['support_offset_mm'],float)+[0,0,-board['size_mm'][2]/2]
    return np.array(board['center_mm'],float)+r@local,r@board_mount_rotation(config).T


def forward_world(config, arm_id, joints_deg):
    q=np.radians(joints_deg)
    f=forward(q)
    br,bt=base_transform(config,arm_id)
    if arm_id == 14:
        offset=config['robot14']['suction_offset_mm']
        f['boxes'][-1]['name']='suction-14-provisional'
        f['boxes'][-1]['size_mm']=[80,80,80]
        f['tcp_xyz_mm']=(np.array(f['tcp_xyz_mm'])+np.array(f['rotation_matrix'])@[0,0,offset-100]).tolist()
    for box in f['boxes']:
        box['name']=f'arm-{arm_id}/'+box['name']
        box['center_mm']=(bt+br@box['center_mm']).tolist()
        box['rotation_matrix']=(br@box['rotation_matrix']).tolist()
    f['tcp_xyz_mm']=(bt+br@f['tcp_xyz_mm']).tolist()
    f['rotation_matrix']=(br@f['rotation_matrix']).tolist()
    return f


@lru_cache(maxsize=1024)
def _inverse(target_tuple, rotation_tuple, tool):
    target=np.array(target_tuple)
    rotation=np.array(rotation_tuple).reshape(3,3)
    def residual(q):
        f=forward(q,False)
        r=np.array(f['rotation_matrix'])
        p=np.array(f['tcp_xyz_mm'])+r@[0,0,tool-100]
        return np.r_[p-target,120*Rotation.from_matrix(rotation.T@r).as_rotvec()]
    guesses=[np.radians([188,88,-213,35,-120,190]),HOME,np.radians([0,20,-120,0,-20,0]),np.radians([0,-50,-60,0,100,0]),
             np.radians([30,-50,-60,-30,100,10]),np.radians([-30,-20,-100,30,100,-10])]
    best=None
    for initial in guesses:
        fit=least_squares(residual,np.clip(initial,LOW+1e-5,HIGH-1e-5),bounds=(LOW,HIGH),max_nfev=220)
        values=residual(fit.x); error=float(np.linalg.norm(values[:3])); orient=float(np.linalg.norm(values[3:])/120)
        candidate=(fit.x,error,orient)
        if best is None or error+120*orient<best[1]+120*best[2]: best=candidate
        if error<.5 and orient<.01: break
    return best


def inverse_world(config,arm_id,xyz,rotation):
    br,bt=base_transform(config,arm_id)
    local=br.T@(np.array(xyz)-bt)
    local_r=br.T@rotation
    tool=100. if arm_id==10 else config['robot14']['suction_offset_mm']
    q,error,orientation=_inverse(tuple(local),tuple(local_r.flatten()),tool)
    return {'arm_reachable':error<=2 and orientation<=.02,'joints_deg':np.degrees(q).tolist(),
            'position_error_mm':error,'orientation_error_deg':math.degrees(orientation)}


def pickup(config):
    return config['pickup_mm']


def release_sequence(config, action):
    release=release_point(config,action)
    pick=pickup(config)
    radius=config['ball']['diameter_mm']/2
    hover=max(pick[2],release[2])+radius+60
    return [('approach',[pick[0],pick[1],hover]),('descend',pick),('lift',[pick[0],pick[1],hover]),
            ('transfer',[release[0],release[1],hover]),('release',list(release)),
            ('retreat',[release[0],release[1],hover]),('reset',[pick[0],pick[1],hover])]


def feasibility(config,action):
    point,orientation=board_target(config)
    board=inverse_world(config,14,point,orientation)
    checks=[{'step':name,**inverse_world(config,10,xyz,DOWN)} for name,xyz in release_sequence(config,action)]
    arm=all(x['arm_reachable'] for x in checks)
    return {'arm_reachable':arm,'board_arm_reachable':board['arm_reachable'],'arm_checks':checks,'board_check':board,
            'reason':'' if arm and board['arm_reachable'] else 'One or more tool targets are outside the two-arm kinematic workspace',
            'scope':'kinematic target checks; native replay separately validates motion paths'}
