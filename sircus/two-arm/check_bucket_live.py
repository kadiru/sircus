"""Probe the pinned fake-only machine; optional bounded training/native replay."""
import argparse, asyncio, json, time
from pathlib import Path
from uuid import UUID
from viam.robot.client import RobotClient
from viam.rpc.dial import DialOptions, Credentials
from viam.components.generic import Generic
from cloud_guard import check_simulation_config
ROOT=Path(__file__).resolve().parent

async def main(args):
    check_simulation_config()
    cache=json.loads((Path.home()/'.viam/cached_cloud_config_28a499f1-b207-4111-90bb-105fefa5a840.json').read_text())
    keys=[]
    for handler in cache.get('auth',{}).get('handlers',[]):
        if handler.get('type')!='api-key':continue
        for key,secret in handler.get('config',{}).items():
            try:UUID(key)
            except (TypeError,ValueError):continue
            if isinstance(secret,str) and secret:keys.append((key,secret))
    if len(keys)!=1:raise RuntimeError('Expected one cached machine credential')
    key,secret=keys[0]
    robot=await RobotClient.at_address('127.0.0.1:8082',RobotClient.Options(dial_options=DialOptions(disable_webrtc=True,auth_entity=key,credentials=Credentials(type='api-key',payload=secret))))
    try:
        for attempt in range(15):
            await robot.refresh()
            try:
                controller=Generic.from_robot(robot,'sircus');probe=await controller.do_command({'probe':True});break
            except Exception:
                if attempt==14:raise
                await asyncio.sleep(1)
        assert probe['config']['bin']['shape']=='tapered_bucket','Old scene still active'
        assert probe['config']['robot14']['board_mount_yaw_deg']==90,'Mount rotation missing'
        (ROOT/'data/bucket-mount-probe.json').write_text(json.dumps(probe,indent=2)+'\n')
        print('PROBE',probe['config_guard'],probe['status'],'bucket',probe['config']['bin']['inner_top_diameter_mm'],'mount90','saved_policy',bool(probe.get('saved_training')),flush=True)
        if not(args.train or args.run_best):return
        command={'train':True,'budget':24,'seed':42}
        if args.run_best:
            saved=probe.get('saved_training')
            if not saved:raise RuntimeError('No matching bucket policy to replay')
            command={'run':True,'action':saved['best_action'],'seed':42,'delay_s':0,'playback_speed':4}
        print('START',await controller.do_command(command),flush=True)
        before=None
        for attempt in range(360):
            await asyncio.sleep(.5)
            state=await controller.do_command({'status':True})
            detail=state.get('training',{}) if args.train else {}
            marker=(state['status'],state.get('step'),detail.get('phase'),detail.get('trial'))
            if marker!=before:print(marker,flush=True);before=marker
            if state['status'] not in ('running','training'):break
        name='bucket-training-live.json' if args.train else 'bucket-native-live.json'
        (ROOT/'data'/name).write_text(json.dumps(state,indent=2)+'\n')
        if state['status']!='complete':raise RuntimeError(state.get('error','Task did not complete'))
        print('SESSION',state.get('data_collection',{}).get('session_id'),flush=True)
        if args.train:
            result=state['training'];print('RESULT',result['best_action'],'heldout',result['test']['retained_count'],'/',result['test']['trials'],flush=True)
        else:print('NATIVE',len(state['result']['native_motion_checks']),'moves','retained',state['result']['retained'],flush=True)
    finally:await robot.close()

if __name__=='__main__':
    parser=argparse.ArgumentParser();group=parser.add_mutually_exclusive_group();group.add_argument('--train',action='store_true');group.add_argument('--run-best',action='store_true');asyncio.run(main(parser.parse_args()))
