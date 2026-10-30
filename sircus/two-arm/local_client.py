"""Credential-free client pinned to THIS package's loopback fake-only server."""
import asyncio
import os
from viam.robot.client import RobotClient
from viam.rpc.dial import DialOptions
from viam.components.generic import Generic
from cloud_guard import check_simulation_config

async def connect():
    os.environ['SIRCUS_LOCAL_TEST']='1'
    check_simulation_config()
    robot=await RobotClient.at_address('127.0.0.1:8083',RobotClient.Options(
        dial_options=DialOptions(insecure=True,disable_webrtc=True)))
    try:
        controller=Generic.from_robot(robot,'sircus')
        result=await controller.do_command({'probe':True},timeout=30)
        if result.get('config_guard')!='passed' or result.get('kinematics')!='two UFACTORY850 fake arms':
            raise RuntimeError('Local simulator probe failed')
        return robot,controller
    except BaseException:
        await robot.close();raise

async def wait_idle(controller, timeout=240):
    async with asyncio.timeout(timeout):
        while True:
            status=await controller.do_command({'status':True},timeout=30)
            if status['status'] not in ('running','training'):
                if status['status'] not in ('idle','complete'):
                    raise RuntimeError(str(status.get('error',status['status'])))
                return status
            await asyncio.sleep(.3)
