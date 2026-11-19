"""Explicit bucket-facing geometry; independent of frozen learning contracts.

A heading constraint changes a policy action. It is not the benchmarked frozen
policy, a contact guarantee, a native arm path, or physical calibration.
"""
import copy
import math
import random

CONTRACT_VERSION='sircus-bucket-facing-v1'
DEFAULT_TILT_BOUNDS=[5.,75.]


def tilt_bounds(value=None):
    value=DEFAULT_TILT_BOUNDS if value is None else value
    if (not isinstance(value,list) or len(value)!=2 or any(type(v) not in (int,float) or not math.isfinite(v) for v in value)
            or not 0<value[0]<=value[1]<90):
        raise ValueError('Tilt bounds require 0 < low <= high < 90 degrees')
    return list(map(float,value))


def wrap(angle):return (angle+180.)%360.-180.


def normal(rpy):
    roll,pitch,yaw=map(math.radians,rpy)
    x=math.sin(pitch)*math.cos(roll);y=-math.sin(roll)
    return [math.cos(yaw)*x-math.sin(yaw)*y,math.sin(yaw)*x+math.cos(yaw)*y,math.cos(pitch)*math.cos(roll)]


def heading(scene):
    bx,by=scene['board']['center_mm'][:2];tx,ty=scene['bin']['center_xy_mm']
    dx,dy=tx-bx,ty-by
    if math.hypot(dx,dy)<1.:raise ValueError('Bucket needs at least 1 mm horizontal separation from board center')
    return math.degrees(math.atan2(dy,dx))


def describe(scene,source,bounds=None):
    bounds=tilt_bounds(bounds);n=normal(scene['board']['rpy_deg'])
    tilt=math.degrees(math.acos(max(-1.,min(1.,n[2]))));target=heading(scene)
    if not bounds[0]-1e-9<=tilt<=bounds[1]+1e-9:raise ValueError('Board tilt is outside requested nonvertical facing bounds')
    length=math.hypot(n[0],n[1]);toward=[math.cos(math.radians(target)),math.sin(math.radians(target))]
    dot=(n[0]*toward[0]+n[1]*toward[1])/length
    if dot<1.-1e-9:raise ValueError('Horizontal board normal does not point toward the bucket')
    return {'contract_version':CONTRACT_VERSION,'heading_deg':target,'tilt_deg':tilt,'normal':n,
            'heading_dot':dot,'tilt_bounds_deg':bounds,'tilt_definition':'0 horizontal; 90 vertical; upward impact normal',
            'source':source,'native_paths_checked':False,'physically_verified':False}


def search_orientation(scene,tilt,bounds=None):
    bounds=tilt_bounds(bounds)
    if type(tilt) not in (int,float) or not math.isfinite(tilt) or not bounds[0]<=tilt<=bounds[1]:raise ValueError('Search tilt outside bounds')
    selected=copy.deepcopy(scene);azimuth=heading(scene)
    negative=scene['board']['rpy_deg'][1]<0
    selected['board']['rpy_deg']=[0.,-tilt if negative else tilt,wrap(azimuth+(180. if negative else 0.))]
    return selected,describe(selected,'bounded_tilt_uv_height_search',bounds)


def constrain_prediction(proposal,bounds=None):
    """Rigidly rotate predicted board/release about world Z, retaining local U/V.

    Caller MUST validate/infer with the original checkpoint first and recompute
    release coordinates and both endpoint checks after applying this transform.
    """
    selected=copy.deepcopy(proposal['selected_scene']);raw=copy.deepcopy(selected['board']['rpy_deg'])
    n=normal(raw);tilt=math.degrees(math.acos(max(-1.,min(1.,n[2]))))
    bounds=tilt_bounds(bounds)
    if not bounds[0]<=tilt<=bounds[1]:raise ValueError('Frozen predicted tilt is outside requested facing bounds; no clamp/search was applied')
    delta=wrap(heading(selected)-math.degrees(math.atan2(n[1],n[0])))
    selected['board']['rpy_deg'][2]=wrap(raw[2]+delta)
    center=selected['board']['center_mm'];old=proposal['release_coordinates']['commanded_release_ball_center_mm']
    a=math.radians(delta);dx,dy=old[0]-center[0],old[1]-center[1]
    expected=[center[0]+math.cos(a)*dx-math.sin(a)*dy,center[1]+math.sin(a)*dx+math.cos(a)*dy,old[2]]
    info=describe(selected,'frozen_prediction_plus_geometric_heading_constraint',bounds)
    info.update(yaw_correction_deg=delta,raw_predicted_rpy_deg=raw,
        expected_rotated_release_mm=expected,release_transform='Rz(yaw_correction) around board center; board-local U/V and world-Z gap retained',
        evaluation_scope='New constrained policy variant; previous frozen-policy benchmark rates do not apply',
        frozen_model_updated=False,per_scene_search=False)
    return selected,copy.deepcopy(proposal['release_action']),info


def random_center(bounds,seed):
    if type(seed) is not int or not 0<=seed<2**32:raise ValueError('Use a uint32 randomization seed')
    if not isinstance(bounds,dict) or set(bounds)!={'min','max'}:raise ValueError('Board bounds need min/max three-vectors in mm')
    lo,hi=bounds['min'],bounds['max']
    if any(not isinstance(v,list) or len(v)!=3 or any(type(x) not in (int,float) or not math.isfinite(x) for x in v) for v in (lo,hi)):
        raise ValueError('Finite numeric board bounds required')
    if any(not -2000<=a<=b<=2000 or b-a>500 for a,b in zip(lo,hi)) or not 50<=lo[2]<=hi[2]<=900:
        raise ValueError('Board box must fit +/-2000 mm, Z50..900 mm, each span <=500 mm')
    rng=random.Random(seed)
    return [rng.uniform(a,b) for a,b in zip(lo,hi)]
