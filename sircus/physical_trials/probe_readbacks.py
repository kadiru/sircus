"""Diagnostic per-read timing. This module has no actuator method calls."""
import asyncio
import json
from pathlib import Path
import time
from viam_adapter import ViamAdapter, utc, write_json, proto

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'evidence/connection-repair/readbacks'

async def main():
    OUT.mkdir(parents=True,exist_ok=True)
    a=ViamAdapter(ROOT.parent/'perception/.private/sfhack10.json')
    a.rpc_timeout=15
    result={'started_at':utc(),'physical_commands':[],'reads':{}}
    async def read(label,fn):
        start=time.monotonic()
        try:
            value=await asyncio.wait_for(fn(),17)
            row={'ok':True,'elapsed_s':time.monotonic()-start}
            if label=='camera':
                images,metadata=value
                row['metadata']=proto(metadata)
                row['images']=[]
                for image in images:
                    mime=getattr(image.mime_type,'value',str(image.mime_type))
                    row['images'].append({'name':image.name,'mime':mime,'bytes':len(image.data)})
                    if mime=='image/jpeg':(OUT/'camera.jpg').write_bytes(image.data)
            elif isinstance(value,bool):row['value']=value
            else:row['value']=proto(value)
        except Exception as exc:row={'ok':False,'elapsed_s':time.monotonic()-start,'error':a._safe_error(exc)}
        result['reads'][label]=row
        write_json(OUT/'readbacks.json',result)
    try:
        result['capabilities']=await a.connect()
        write_json(OUT/'readbacks.json',result)
        calls=[read('camera',lambda:a.camera.get_images(timeout=15))]
        for name in ['arm','arm-board']:
            arm=a.arms[name]
            calls += [read(name+':joints',lambda arm=arm:arm.get_joint_positions(timeout=15)),
                      read(name+':moving',lambda arm=arm:arm.is_moving(timeout=15)),
                      read(name+':pose',lambda name=name:a.motion.get_pose(name,'world',timeout=15))]
        await asyncio.gather(*calls)
    finally:
        result['finished_at']=utc();write_json(OUT/'readbacks.json',result)
        if a.robot:await asyncio.wait_for(a.close(),8)
    print(json.dumps({k:{kk:vv for kk,vv in v.items() if kk not in ('value','metadata','images')} for k,v in result['reads'].items()},indent=2))

if __name__=='__main__':asyncio.run(main())
