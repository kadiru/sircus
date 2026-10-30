"""Build cloud or isolated-test configuration with two fake 850 models."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parent

def frame(parent,xyz=(0,0,0),yaw=0):
 return {'parent':parent,'translation':dict(zip(('x','y','z'),xyz)), 'orientation':{'type':'ov_degrees','value':{'x':0,'y':0,'z':1,'th':yaw}}}

def expected_config(test=False):
 scene=json.loads((ROOT/'scene.json').read_text())
 return {'network':{'bind_address':'127.0.0.1:'+('8083' if test else '8082')},'components':[
 {'name':'arm-10','api':'rdk:component:arm','model':'rdk:builtin:fake','attributes':{'model-path':str(ROOT/'models/xarm850_kinematics.json')},'frame':frame('world')},
 {'name':'arm-14','api':'rdk:component:arm','model':'rdk:builtin:fake','attributes':{'model-path':str(ROOT/'models/xarm850_kinematics.json')},'frame':frame('world',scene['robot14']['base_translation_mm'],scene['robot14']['base_yaw_deg'])},
 {'name':'gripper-10','api':'rdk:component:gripper','model':'rdk:builtin:fake','attributes':{'model-path':str(ROOT/'models/gripper-proxy.json')},'frame':frame('arm-10',(0,0,100))},
 {'name':'suction-14','api':'rdk:component:gripper','model':'rdk:builtin:fake','attributes':{'model-path':str(ROOT/'models/suction-proxy.json')},'frame':frame('arm-14',(0,0,scene['robot14']['suction_offset_mm']))},
 {'name':'sircus','api':'rdk:component:generic','model':'local:sircus:controller','attributes':{}}],
 'services':[
 {'name':'sircus-scene','api':'rdk:service:world_state_store','model':'local:sircus:scene','attributes':{}},
 {'name':'sircus-data','api':'rdk:service:data_manager','model':'rdk:builtin:builtin','attributes':{
 'additional_sync_paths':[str(ROOT/'data/sync-outbox')], 'capture_dir':str(ROOT/'data/capture'),
 'sync_interval_mins':0.1,'tags':['sircus','simulation','two-arm'],'sync_disabled':test,'file_last_modified_millis':10000}}],
 'modules':[{'name':'sircus-two-arm','type':'local','executable_path':str(ROOT/('run-local-test.sh' if test else 'run-module.sh'))}]}

if __name__=='__main__':
 for test,name in ((False,'viam-config.cloud.json'),(True,'viam-config.test.json')):
  (ROOT/name).write_text(json.dumps(expected_config(test),indent=2)+'\n')
 print('Built two-arm fake-only cloud and local test configurations.')
