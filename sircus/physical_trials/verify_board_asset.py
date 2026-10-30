"""Read-only installed-board frame/geometry verification. No motion/tool calls."""
import asyncio
import hashlib
import json
import math
from pathlib import Path
from viam.proto.common import Pose, PoseInFrame
from viam_adapter import ViamAdapter, utc, proto, write_json

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'evidence/board-asset/runtime.json'


def scrub(value):
    if isinstance(value,dict):return {k:scrub(v) for k,v in value.items() if k not in ('original_file','bytes')}
    if isinstance(value,list):return [scrub(x) for x in value]
    return value


def dimensions(value):
    found=[]
    if isinstance(value,dict):
        if 'dims_mm' in value:found.append([value['dims_mm'].get(k,0) for k in ('x','y','z')])
        if value.get('type')=='box' and all(k in value for k in ('x','y','z')):found.append([value[k] for k in ('x','y','z')])
        for v in value.values():found.extend(dimensions(v))
    elif isinstance(value,list):
        for v in value:found.extend(dimensions(v))
    return found


async def main():
    asset_path=ROOT/'assets/board.json'
    asset=json.loads(asset_path.read_text());name=asset['asset_id']
    a=ViamAdapter(ROOT.parent/'perception/.private/sfhack10.json')
    result={'started_at':utc(),'asset_id':name,'asset_sha256':hashlib.sha256(asset_path.read_bytes()).hexdigest(),
        'physical_commands':[],'configuration_changes':[],'pose_provenance':'Kinematic configured-frame transform, not observed or calibrated board pose',
        'mount_status':asset['mount']['pose_status'],'world_pose_calibrated':False}
    try:
        await a.connect()
        result['before']=await a.snapshot()
        frames=await asyncio.wait_for(a.robot.get_frame_system_config(),15)
        named=[proto(x) for x in frames if x.frame.reference_frame==name]
        result['matching_frame_count']=len(named)
        result['installed_frame']=scrub(named[0]) if named else None
        result['board_resource_present']=any(x.name==name for x in a.robot.resource_names)
        if not named:raise RuntimeError('Installed board frame not found')
        frame=named[0]['frame'];poseframe=frame['pose_in_observer_frame'];p=poseframe.get('pose',{})
        expected=asset['mount']['translation_mm']
        result['checks']={'parent_matches':poseframe['reference_frame']==asset['mount']['parent_frame'],
            'mount_translation_matches':all(abs(p.get(k,0)-v)<1e-6 for k,v in zip(('x','y','z'),expected)),
            'mount_yaw_matches':abs(p.get('theta',0)-asset['mount']['yaw_deg'])<1e-6 and abs(p.get('o_z',0)-1)<1e-6}
        result['published_box_dimensions_mm']=dimensions(named[0])
        result['checks']['dimensions_match']=any(all(abs(x-y)<1e-6 for x,y in zip(d,asset['dimensions']['size_mm'])) for d in result['published_box_dimensions_mm'])
        for label,z in [('center',0),('impact_surface_center',2.5),('normal_endpoint',102.5)]:
            q=PoseInFrame(reference_frame=name,pose=Pose(z=z,o_z=1))
            world=await asyncio.wait_for(a.robot.transform_pose(q,'world'),10)
            result[label+'_world']=proto(world)
        c=result['impact_surface_center_world']['pose'];tip=result['normal_endpoint_world']['pose']
        n=[tip.get(k,0)-c.get(k,0) for k in ('x','y','z')];norm=math.sqrt(sum(x*x for x in n))
        result['nominal_world_normal']=[x/norm for x in n]
        result['after']=await a.snapshot()
        result['joint_delta_degrees']={name:[b-a for a,b in zip(result['before']['arms'][name]['joints'],result['after']['arms'][name]['joints'])] for name in ('arm','arm-board')}
        result['both_stationary_at_readbacks']=all(not sample['arms'][n]['is_moving'] for sample in [result['before'],result['after']] for n in ('arm','arm-board'))
        result['success']=all(result['checks'].values())
    except Exception as exc:
        result['success']=False;result['error']=a._safe_error(exc)
    finally:
        result['finished_at']=utc();OUT.parent.mkdir(parents=True,exist_ok=True);write_json(OUT,result)
        if a.robot:
            try:await asyncio.wait_for(a.close(),8)
            except Exception as exc:result['close_error']=a._safe_error(exc)
        write_json(OUT,result)
    print(json.dumps({k:result.get(k) for k in ('success','checks','published_box_dimensions_mm','joint_delta_degrees','both_stationary_at_readbacks','error')},indent=2),flush=True)

if __name__=='__main__':asyncio.run(main())
