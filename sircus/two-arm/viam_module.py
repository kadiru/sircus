"""Two native fake 850 arms with attached compliant board and Viam-managed training."""
import asyncio, copy, json, math, os, time, uuid
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from google.protobuf.field_mask_pb2 import FieldMask
from viam.components.arm import Arm
from viam.components.generic import Generic
from viam.components.gripper import Gripper
from viam.module.module import Module
from viam.proto.common import GeometriesInFrame, Geometry, Pose, PoseInFrame, RectangularPrism, Sphere, Transform, Vector3, WorldState
from viam.proto.service.worldstatestore import StreamTransformChangesResponse
from viam.resource.easy_resource import EasyResource
from viam.services.motion import MotionClient
from viam.services.worldstatestore import WorldStateStore
from viam.utils import dict_to_struct
from physics import simulate, validate_config, release_point
from bucket_geometry import clearance_boxes
from cloud_guard import check_simulation_config
from two_arm_kinematics import board_target, board_mount_rotation, forward_world, feasibility, release_sequence, DOWN
from training_loop import train
from trial_store import TrialStore, fingerprint
from policy_cache import load_saved_training
ROOT=Path(__file__).resolve().parent
SCENE_PATH=ROOT/'scene.json'
MODULE=None
VISUALS=None

def load_scene():
    c=json.loads(SCENE_PATH.read_text());validate_scene(c);return c

def validate_scene(c):
    validate_config(c)
    robot=c['robot14']
    if robot['base_translation_mm'] != [1337.31,0,0] or robot['base_yaw_deg'] != 180 or robot['suction_offset_mm'] != 100:
        raise ValueError('Arm base/tool frames are fixed in the Viam configuration; update configuration deliberately')
    if len(c['pickup_mm'])!=3 or not all(math.isfinite(float(v)) for v in c['pickup_mm']): raise ValueError('Invalid pickup')

def pose(xyz=(0, 0, 0), rotation=None):
    """Matrix -> Viam orientation vector (ZYZ, with Viam's pole convention)."""
    matrix = np.eye(3) if rotation is None else np.asarray(rotation, dtype=float)
    normal = matrix[:, 2]
    lon = math.atan2(normal[1], normal[0])
    lat = math.acos(float(np.clip(normal[2], -1, 1)))
    base = Rotation.from_euler("ZY", [lon, lat]).as_matrix()
    twist = base.T @ matrix
    theta = math.atan2(twist[1, 0], twist[0, 0])
    if 1 - abs(normal[2]) <= 0.0001:
        theta += lon if normal[2] > 0 else -lon
    theta = (math.degrees(theta) + 180) % 360 - 180
    return Pose(x=xyz[0], y=xyz[1], z=xyz[2], o_x=normal[0], o_y=normal[1], o_z=normal[2], theta=theta)


def orientation_matrix(p):
    normal=np.array([p.o_x,p.o_y,p.o_z],float);normal/=np.linalg.norm(normal)
    lon=math.atan2(normal[1],normal[0]);lat=math.acos(float(np.clip(normal[2],-1,1)))
    theta=math.radians(p.theta)
    if 1-abs(normal[2])<=.0001:theta-=lon if normal[2]>0 else -lon
    return Rotation.from_euler('ZYZ',[lon,lat,theta]).as_matrix()

def board_rotation(config):
    return Rotation.from_euler("xyz", config["board"]["rpy_deg"], degrees=True).as_matrix()


def pickup(config):
    table = config["table"]
    center, size = table["center_mm"], table["size_mm"]
    return list(config.get("pickup_mm", [350, -220,
            center[2] + size[2]/2 + config["ball"]["diameter_mm"]/2]))


def solids(config, frame=None):
    """Board and conservative outer-container clearance proxies for native Viam."""
    board = config["board"]
    rotation = board_rotation(config) if frame is None else frame["board_rotation"]
    center = board["center_mm"] if frame is None else frame["board_center_mm"]
    result = [("board-14", center, board["size_mm"], rotation, (49, 202, 186))]
    for solid in clearance_boxes(config):
        color = (210, 215, 221) if solid["name"] == "table" else (236, 174, 57)
        result.append((solid["name"], solid["center_mm"], solid["size_mm"], solid.get("rotation_matrix"), color))
    return result


def geometry(name, xyz=(0, 0, 0), size=None, rotation=None, radius=None):
    center = pose(xyz, rotation)
    if radius is not None:
        return Geometry(label=name, center=center, sphere=Sphere(radius_mm=radius))
    return Geometry(label=name, center=center, box=RectangularPrism(dims_mm=Vector3(x=size[0], y=size[1], z=size[2])))


def transform(name, xyz, size=None, rotation=None, parent="world", radius=None, color=(230, 59, 111)):
    return Transform(reference_frame=name,
        uuid=uuid.uuid5(uuid.NAMESPACE_URL, "sircus-simulation/"+name).bytes,
        pose_in_observer_frame=PoseInFrame(reference_frame=parent, pose=pose(xyz, rotation)),
        physical_object=geometry(name, size=size, radius=radius),
        metadata=dict_to_struct({"color": {"r": color[0], "g": color[1], "b": color[2]}, "opacity": 1.0}))


class SircusScene(WorldStateStore, EasyResource):
    MODEL = "local:sircus:scene"

    @classmethod
    def validate_config(cls, config):
        return [], []

    @classmethod
    def new(cls, config, dependencies):
        global VISUALS
        self = cls(config.name)
        self.transforms = {}
        self.subscribers = set()
        self.config = load_scene()
        self.sync(self.config)
        self.ball(pickup(self.config))
        VISUALS = self
        return self

    async def list_uuids(self, **kwargs):
        return list(self.transforms)

    async def get_transform(self, uuid, **kwargs):
        return self.transforms[uuid]

    async def stream_transform_changes(self, **kwargs):
        queue = asyncio.Queue(maxsize=256)
        self.subscribers.add(queue)
        try:
            for value in list(self.transforms.values()):
                yield StreamTransformChangesResponse(change_type=1, transform=value)
            while True:
                yield await queue.get()
        finally:
            self.subscribers.discard(queue)

    async def do_command(self, command, **kwargs):
        if command.get("status"):
            return {"objects": [t.reference_frame for t in self.transforms.values()], "config": self.config}
        raise ValueError("Use the sircus controller for scene changes and runs")

    def update(self, value):
        kind = 3 if value.uuid in self.transforms else 1
        self.transforms[value.uuid] = value
        change = StreamTransformChangesResponse(change_type=kind, transform=value,
            updated_fields=FieldMask(paths=["pose_in_observer_frame", "physical_object", "metadata"]))
        for queue in self.subscribers:
            if queue.full():
                queue.get_nowait()
            queue.put_nowait(change)

    def sync(self, config, frame=None):
        self.config = copy.deepcopy(config)
        nominal_xyz, nominal_rotation = board_target(config)
        active_solids=solids(config, frame)
        active_names={entry[0] for entry in active_solids}|{'oddball-75mm'}
        for identifier, previous in list(self.transforms.items()):
            if previous.reference_frame not in active_names:
                del self.transforms[identifier]
                change=StreamTransformChangesResponse(change_type=2,transform=previous)
                for queue in self.subscribers:
                    if queue.full():queue.get_nowait()
                    queue.put_nowait(change)
        for name, xyz, size, rotation, color in active_solids:
            if name == "board-14":
                # The board is a child of the actual arm14 suction TCP. During a
                # bounce, local deflection follows the contact model, not a rigid mount.
                xyz = (nominal_rotation.T @ (np.array(xyz)-nominal_xyz)).tolist()
                rotation = nominal_rotation.T @ np.array(rotation)
                self.update(transform(name, xyz, size, rotation, parent="suction-14", color=color))
            else:
                self.update(transform(name, xyz, size, rotation, color=color))

    def ball(self, xyz, parent="world"):
        self.update(transform("oddball-75mm", xyz, parent=parent,
                              radius=self.config["ball"]["diameter_mm"]/2))


class SircusController(Generic, EasyResource):
    MODEL='local:sircus:controller'
    @classmethod
    def validate_config(cls, config):
        return ['arm-10','arm-14','gripper-10','suction-14','sircus-scene','rdk:service:motion/builtin'],[]
    @classmethod
    def new(cls, config, dependencies):
        self=cls(config.name);self.config=load_scene();self.task=None;self.stop_event=asyncio.Event();self.store=None
        self.state={'status':'idle','step':'ready','mode':'simulation','physically_verified':False,
                    'drop_robot_id':10,'board_robot_id':14,'board_arm_mode':'native_850','board_positioned':False}
        self.events=[];self.arms={};self.saved_training=load_saved_training(self.config);return self
    def saved_policy(self):
        if self.saved_training and self.saved_training.get("config_fingerprint")==fingerprint(self.config):
            return self.saved_training
        return None
    def record(self,event,**values):
        self.events.append({'event':event,'time':time.time(),**values})
    def execution_origin(self):
        local=os.environ.get('SIRCUS_LOCAL_TEST')=='1'
        return {'viam_execution':'isolated_local_test' if local else 'cloud_connected_simulation','native_machine':None if local else 'sircus-sim','machine_id':None if local else '5db5a9df-2e56-4711-9d39-b42cc86e4d79','loopback_port':8083 if local else 8082}
    def collection(self):
        info=self.store.status() if self.store else {'pending_files':len(list((ROOT/'data/sync-outbox').glob('*.json')))}
        return {'enabled':True,'sync_configured':os.environ.get('SIRCUS_LOCAL_TEST')!='1','cloud_upload_verified':False,**info}
    async def capture_arms(self):
        if not MODULE or not MODULE.parent: return self.arms
        for arm_id in (10,14):
            arm=Arm.from_robot(MODULE.parent,f'arm-{arm_id}')
            q=await arm.get_joint_positions(timeout=5)
            self.arms[str(arm_id)]=forward_world(self.config,arm_id,list(q.values))
        f=self.arms['14'];r=np.array(f['rotation_matrix'])@board_mount_rotation(self.config)
        local=-np.array(self.config['board']['support_offset_mm'],float)+[0,0,self.config['board']['size_mm'][2]/2]
        self.state['board_pose']={'center_mm':(np.array(f['tcp_xyz_mm'])+r@local).tolist(),'rpy_deg':Rotation.from_matrix(r).as_euler('xyz',degrees=True).tolist(),'rotation':r.tolist()}
        return self.arms
    async def probe(self):
        check_simulation_config();robot=MODULE.parent;await robot.refresh()
        expected=json.loads((ROOT/'models/xarm850_kinematics.json').read_text())
        for arm_id in (10,14):
            kinematics=await Arm.from_robot(robot,f'arm-{arm_id}').get_kinematics(timeout=10)
            model=json.loads(kinematics[1])
            if model.get('links')!=expected['links'] or model.get('joints')!=expected['joints']:
                raise RuntimeError(f'arm-{arm_id} does not report pinned850model')
        await self.capture_arms()
        return {**self.state,'config_guard':'passed','kinematics':'two UFACTORY850 fake arms',
                'default_action':{'u_mm':0,'v_mm':30,'gap_mm':80},'saved_training':self.saved_policy(),'config':self.config,'arms':self.arms,'data_collection':self.collection(),'events':self.events}
    async def snapshot(self):
        await self.capture_arms()
        return {**self.state,'saved_training':self.saved_policy(),'config':self.config,'arms':self.arms,'events':self.events,'data_collection':self.collection()}
    async def do_command(self,command,**kwargs):
        if command.get('status'):return await self.snapshot()
        if command.get('probe'):return await self.probe()
        if command.get('stop'):
            self.stop_event.set()
            if self.task and not self.task.done():
                self.task.cancel()
                try:await self.task
                except asyncio.CancelledError:pass
            for arm_id in (10,14):await Arm.from_robot(MODULE.parent,f'arm-{arm_id}').stop(timeout=5)
            return await self.snapshot()
        if self.task and not self.task.done():raise RuntimeError('Simulation task already running')
        check_simulation_config()
        if 'set_scene' in command or 'set_board' in command:
            candidate=copy.deepcopy(command.get('set_scene',self.config))
            if 'set_board' in command:
                allowed={'center_mm','rpy_deg'}
                if set(command['set_board'])-allowed:raise ValueError('set_board accepts center_mm/rpy_deg')
                candidate['board'].update(command['set_board'])
            validate_scene(candidate)
            p,r=board_target(candidate)
            from two_arm_kinematics import inverse_world
            if not (await asyncio.to_thread(inverse_world,candidate,14,p,r))['arm_reachable']:raise ValueError('Board target is not reachable by arm14')
            self.pending_scene=candidate
            self.state.update(status='running',step='position_board',board_positioned=False)
            self.events=[];self.stop_event=asyncio.Event()
            self.task=asyncio.create_task(self.change_scene(candidate))
            return {'started':True,'status':'running','step':'position_board','config':candidate}
        if command.get('run') or command.get('train'):
            seed=command.get('seed',42)
            if isinstance(seed,bool) or not isinstance(seed,(int,float)) or seed!=int(seed) or not 0<=seed<2**30:raise ValueError('seed must be integer0..2^30-1')
            self.state.pop('error',None);self.state.pop('result',None)
            self.events=[];self.stop_event=asyncio.Event()
            if command.get('train'):
                budget=command.get('budget',24)
                if isinstance(budget,bool) or budget!=int(budget) or not 8<=budget<=80:raise ValueError('budget must be8..80')
                self.state.update(status='training',step='position_board',training={'phase':'starting','budget':int(budget)})
                self.task=asyncio.create_task(self.run_training(int(seed),int(budget)))
            else:
                action=command.get('action',{'u_mm':0,'v_mm':30,'gap_mm':80})
                release_point(self.config,action)
                feasible=await asyncio.to_thread(feasibility,self.config,action)
                if not feasible['arm_reachable'] or not feasible['board_arm_reachable']:raise ValueError(feasible['reason'])
                delay=float(command.get('delay_s',.1));speed=float(command.get('playback_speed',1))
                if not 0<=delay<=5 or not .1<=speed<=4:raise ValueError('Invalid delay/playback speed')
                self.state.update(status='running',step='position_board')
                self.task=asyncio.create_task(self.run_trial(copy.deepcopy(action),int(seed),delay,speed))
            return {'started':True,'mode':'simulation','status':self.state['status']}
        raise ValueError('Use probe/status/set_scene/set_board/run/train/stop')
    async def change_scene(self,candidate):
        old=copy.deepcopy(self.config)
        try:
            await self.position_board(candidate)
            self.config=candidate;VISUALS.sync(candidate);VISUALS.ball(pickup(candidate))
            temporary=SCENE_PATH.with_suffix('.tmp');temporary.write_text(json.dumps(candidate,indent=2,allow_nan=False)+'\n');temporary.replace(SCENE_PATH)
            self.state.pop('training',None);self.state.pop('result',None);self.state.pop('error',None)
            if not self.saved_policy():self.saved_training=load_saved_training(self.config)
            self.state.update(status='idle',step='board_positioned',board_positioned=True)
        except asyncio.CancelledError:self.state.update(status='cancelled',step='stopped');raise
        except Exception as e:
            self.config=old;self.state.update(status='aborted',error=f'{type(e).__name__}: {e}')
    async def position_board(self,config):
        check_simulation_config()
        motion=MotionClient.from_robot(MODULE.parent,'builtin')
        xyz,rotation=board_target(config)
        obstacles=[geometry(s['name'],s['center_mm'],s['size_mm'],s.get('rotation_matrix')) for s in clearance_boxes(config)]
        mount=board_mount_rotation(config)
        local=mount@(-np.array(config['board']['support_offset_mm'],float)+[0,0,config['board']['size_mm'][2]/2])
        attached=transform('board-14-planning',local.tolist(),config['board']['size_mm'],rotation=mount,parent='suction-14')
        self.state['step']='position_board'
        ok=await motion.move(component_name='suction-14',destination=PoseInFrame(reference_frame='world',pose=pose(xyz,rotation)),
            world_state=WorldState(obstacles=[GeometriesInFrame(reference_frame='world',geometries=obstacles)],transforms=[attached]),extra={'timeout':20.0},timeout=55)
        if not ok:raise RuntimeError('Viam rejected board placement')
        actual=(await motion.get_pose('suction-14','world',timeout=10)).pose
        error=math.dist(xyz,[actual.x,actual.y,actual.z])
        actual_r=orientation_matrix(actual)
        orientation_error=math.degrees(Rotation.from_matrix(rotation.T@actual_r).magnitude())
        if error>2 or orientation_error>1:raise RuntimeError('Board suction TCP readback mismatch')
        self.state['board_positioned']=True
        self.record('position_board',motion_ok=True,position_error_mm=error,orientation_error_deg=orientation_error,target_xyz_mm=xyz.tolist(),target_rpy_deg=Rotation.from_matrix(rotation).as_euler('xyz',degrees=True).tolist(),board_rpy_deg=config['board']['rpy_deg'],board_mount_yaw_deg=config['robot14'].get('board_mount_yaw_deg',0))
        await self.capture_arms()
        self.record('arm_snapshot',arms=copy.deepcopy(self.arms))
        VISUALS.sync(config)
    async def run_training(self,seed,budget):
        self.store=TrialStore(self.config,purpose='training',metadata={**self.execution_origin(),'proposal_motion':'kinematic endpoint validation; selected policy requires native replay'})
        try:
            await self.probe();await self.position_board(self.config)
            def progress(p):self.state['training']=copy.deepcopy(p);self.state['step']='training'
            result=await train(copy.deepcopy(self.config),seed=seed,budget=budget,feasibility=feasibility,on_progress=progress,stop_event=self.stop_event,store=self.store)
            self.saved_training=copy.deepcopy(result)
            self.state.update(status='complete',step='training_complete',training=result,result=result['best_result'])
        except asyncio.CancelledError:self.state.update(status='cancelled',step='stopped');raise
        except Exception as e:self.state.update(status='aborted',error=f'{type(e).__name__}: {e}')
        finally:
            if not (self.store.archive/'summary.json').exists():
                self.store.save_summary({'status':self.state['status'],'error':self.state.get('error'),'events':self.events})
    async def run_trial(self,action,seed,delay,speed):
        result=None;held=False
        self.store=TrialStore(self.config,purpose='manual',metadata={**self.execution_origin(),'execution':'native Viam motion plans with synthetic ball contact'})
        try:
            await self.probe();config=copy.deepcopy(self.config);await self.position_board(config)
            result=await asyncio.to_thread(simulate,config,action,seed=seed,frames=True)
            release=result['release_xyz_mm'];pick=pickup(config);radius=config['ball']['diameter_mm']/2
            motion=MotionClient.from_robot(MODULE.parent,'builtin');grip=Gripper.from_robot(MODULE.parent,'gripper-10')
            async def move(name,xyz):
                check_simulation_config();self.state['step']=name
                obstacles=[geometry(n,p,s,r) for n,p,s,r,_ in solids(config)]
                attachments=[transform('held-oddball',[0,0,0],parent='gripper-10',radius=radius)] if held else []
                target=pose(xyz,DOWN)
                ok=await motion.move(component_name='gripper-10',destination=PoseInFrame(reference_frame='world',pose=target),
                    world_state=WorldState(obstacles=[GeometriesInFrame(reference_frame='world',geometries=obstacles)],transforms=attachments),extra={'timeout':20.0},timeout=55)
                if not ok:raise RuntimeError('Viam rejected '+name)
                actual=(await motion.get_pose('gripper-10','world',timeout=10)).pose
                err=math.dist(xyz,[actual.x,actual.y,actual.z])
                if err>2 or actual.o_z>-.999:raise RuntimeError('TCP readback mismatch at '+name)
                await self.capture_arms();self.record(name,motion_ok=True,position_error_mm=err,target_xyz_mm=xyz,arms=copy.deepcopy(self.arms))
                await asyncio.sleep(delay)
            VISUALS.ball(pick)
            seq=release_sequence(config,action)
            for name,xyz in seq:
                if name=='descend':await grip.open()
                await move(name,xyz)
                if name=='descend':
                    raw=await grip.grab();held=True;VISUALS.ball([0,0,0],'gripper-10');self.record('modeled_grasp',raw_fake_grab=raw)
                if name=='release':
                    await grip.open();held=False;VISUALS.ball(release)
                    previous=0
                    for f in result['frames']:
                        await asyncio.sleep(max(0,f['t']-previous)/speed)
                        VISUALS.sync(config,f);VISUALS.ball(f['ball_xyz_mm']);previous=f['t']
            VISUALS.sync(config)
            result['native_motion_checks']=[e for e in self.events if e.get('motion_ok')]
            result['native_replay_verified']=True
            self.state.update(status='complete',step='trial_complete',result=result)
        except asyncio.CancelledError:self.state.update(status='cancelled',step='stopped');raise
        except Exception as e:self.state.update(status='aborted',error=f'{type(e).__name__}: {e}')
        finally:
            self.state['held_modeled']=held
            observed={**(result or {}),'valid_for_training':self.state['status']=='complete','native_replay_verified':self.state['status']=='complete'}
            if self.state['status']!='complete':
                observed={'reward':None,'retained':False,'outcome':'execution_failed','valid_for_training':False,'native_replay_verified':False,'predicted_physics_result':result}
            self.store.record_trial('native',1,action,seed,{**observed,'execution_state':self.state,'events':self.events},feasibility=await asyncio.to_thread(feasibility,self.config,action))
            self.store.save_summary({'status':self.state['status'],'error':self.state.get('error'),'events':self.events})
    async def close(self):
        if self.task and not self.task.done():
            self.task.cancel()
            try:await self.task
            except asyncio.CancelledError:pass

async def main():
    global MODULE
    check_simulation_config();MODULE=Module.from_args()
    MODULE.add_model_from_registry(SircusScene.API,SircusScene.MODEL)
    MODULE.add_model_from_registry(SircusController.API,SircusController.MODEL)
    await MODULE.start()
if __name__=='__main__':asyncio.run(main())
