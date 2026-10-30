"""Bounded read-only adapter probe. No actuator/configuration command is called.

Use the existing Viam environment. Parent process can enforce an outer timeout
because SDK transport shutdown can outlast an asyncio RPC timeout.
"""
import argparse
import asyncio
import hashlib
import json
from pathlib import Path
import time
from viam_adapter import ViamAdapter, utc, write_json


async def probe(args):
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    adapter = ViamAdapter(args.credentials, {"transport":args.transport})
    result = {"started_at":utc(),"transport":args.transport,"physical_commands":[],
              "configuration_changes":[],"readbacks":[],"camera_frames":[]}
    began = time.monotonic()
    try:
        result['capabilities'] = await adapter.connect()
        result['connect_elapsed_s'] = time.monotonic()-began
        write_json(out/'probe.json',result)
        for index in range(args.frames):
            state = await adapter.snapshot()
            raw,mime,meta = await adapter.camera_frame()
            name=f'camera-{index:02d}'+('.jpg' if mime=='image/jpeg' else '.png')
            (out/name).write_bytes(raw)
            result['readbacks'].append(state)
            result['camera_frames'].append({'file':name,'mime_type':mime,'bytes':len(raw),
                'sha256':hashlib.sha256(raw).hexdigest(),'metadata':meta})
            write_json(out/'probe.json',result)
            if index+1<args.frames: await asyncio.sleep(.3)
        result['success']=True
    except BaseException as exc:
        result['success']=False
        result['error']=adapter._safe_error(exc)
    finally:
        result['finished_at']=utc()
        result['elapsed_s']=time.monotonic()-began
        write_json(out/'probe.json',result)
        if adapter.robot:
            try: await asyncio.wait_for(adapter.close(),8)
            except BaseException as exc: result['close_error']=adapter._safe_error(exc)
        write_json(out/'probe.json',result)
    print(json.dumps({'success':result['success'],'transport':args.transport,
        'connect_elapsed_s':result.get('connect_elapsed_s'),'readbacks':len(result['readbacks']),
        'camera_frames':len(result['camera_frames']),'error':result.get('error'),
        'output':str(out)},indent=2),flush=True)
    return 0 if result['success'] else 1


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--credentials',type=Path,default=Path(__file__).resolve().parent.parent/'perception/.private/sfhack10.json')
    p.add_argument('--transport',choices=('auto','direct'),default='auto')
    p.add_argument('--frames',type=int,default=3)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if not 1<=args.frames<=5:p.error('frames must be 1..5')
    raise SystemExit(asyncio.run(probe(args)))
