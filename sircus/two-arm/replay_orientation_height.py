"""Replay a frozen orientation/height candidate only on the fake local8083 runtime."""
import argparse,asyncio,json
from pathlib import Path
from local_client import connect,wait_idle
from train_orientation_height import verify_policy,write

async def run(args):
    policy=json.loads(args.policy.read_text())
    base=json.loads((args.policy.parent/'scene.json').read_text())
    scene,action=verify_policy(policy,base)
    robot,controller=await connect()
    try:
        state=await controller.do_command({'status':True})
        if state['status'] in ('running','training'):raise RuntimeError('Simulator already busy; do not interrupt it')
        await controller.do_command({'set_scene':scene},timeout=30)
        await wait_idle(controller)
        await controller.do_command({'run':True,'action':action,'seed':policy['seed'], 'playback_speed':4,'delay_s':0},timeout=30)
        result=await wait_idle(controller)
        target=args.output or args.policy.parent/'native-replay.json'
        write(target,result)
        print('Saved fake-arm native motion replay:',target)
        print('This does not validate hardware or a generalized policy.')
    finally:await robot.close()
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('policy',type=Path);p.add_argument('--output',type=Path)
    asyncio.run(run(p.parse_args()))
