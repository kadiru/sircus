import argparse,asyncio,json,os,sys,time
from uuid import UUID
from pathlib import Path
from viam.robot.client import RobotClient
from viam.rpc.dial import DialOptions,Credentials
from viam.components.generic import Generic
ROOT=Path(__file__).resolve().parent
async def main(args):
    if args.local:
        robot=await RobotClient.at_address('127.0.0.1:8083',RobotClient.Options(dial_options=DialOptions(insecure=True,disable_webrtc=True)))
    else:
        cache=json.loads((Path.home()/'.viam/cached_cloud_config_28a499f1-b207-4111-90bb-105fefa5a840.json').read_text())
        # See source below for authoritative cached API-key fields.
        keys=[]
        for handler in cache.get('auth',{}).get('handlers',[]):
            if handler.get('type')!='api-key':continue
            for candidate,secret in handler.get('config',{}).items():
                try:UUID(candidate)
                except (ValueError,TypeError):continue
                if isinstance(secret,str) and secret:keys.append((candidate,secret))
        if len(keys)!=1:raise RuntimeError('Expected one cached machine API credential')
        key_id,key=keys[0]
        robot=await RobotClient.at_address('127.0.0.1:8082',RobotClient.Options(dial_options=DialOptions(disable_webrtc=True,auth_entity=key_id,credentials=Credentials(type='api-key',payload=key))))
    try:
        c=Generic.from_robot(robot,'sircus'); probe=await c.do_command({'probe':True});print('PROBE',probe['config_guard'],list(probe['arms']),flush=True)
        cmd={'run':True,'action':{'u_mm':0,'v_mm':30,'gap_mm':80},'seed':42,'playback_speed':4,'delay_s':0}
        if args.train:cmd={'train':True,'budget':8,'seed':42}
        if args.position:cmd={'set_board':{'center_mm':probe['config']['board']['center_mm'],'rpy_deg':probe['config']['board']['rpy_deg']}}
        if not(args.run or args.position or args.train):return
        print(await c.do_command(cmd),flush=True)
        before=None
        for i in range(300):
            await asyncio.sleep(.5); status=await c.do_command({'status':True})
            label=(status['status'],status.get('step'),status.get('error'))
            if label!=before:print(label,flush=True);before=label
            if status['status'] not in ('running','training'):break
        (ROOT/'data'/('local-proof.json' if args.local else 'cloud-proof.json')).write_text(json.dumps(status,indent=2)+'\n')
        if status['status'] not in ('complete','idle'):raise RuntimeError(status.get('error'))
        print('PASSED moves',len([e for e in status.get('events',[]) if e.get('motion_ok')]),flush=True)
    finally:await robot.close()
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--local',action='store_true');p.add_argument('--run',action='store_true');p.add_argument('--position',action='store_true');p.add_argument('--train',action='store_true');asyncio.run(main(p.parse_args()))
