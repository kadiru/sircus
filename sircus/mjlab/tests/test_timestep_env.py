import copy
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

import numpy as np

from sircus.mjlab import timestep_env as module


def options(substeps=64, worlds=1):
    native = NS(timestep=1 / (240 * substeps), tolerance=1e-8,
        gravity=np.asarray([0., 0., -9.81]), solver=2, integrator=3, cone=0, iterations=100)
    warp = NS(timestep=np.full(worlds, native.timestep, np.float32),
        tolerance=np.full(worlds, native.tolerance, np.float32),
        gravity=np.tile(native.gravity, (worlds, 1)).astype(np.float32),
        solver=2, integrator=3, cone=0, iterations=100)
    return native, warp


class FrozenOptions(NS):
    def __setattr__(self, key, value):
        if getattr(self, '_locked', False):
            raise AssertionError('Native options mutated after Warp allocation')
        return super().__setattr__(key, value)


class Model:
    def __init__(self):
        self.opt = FrozenOptions(timestep=1 / 240, tolerance=1e-6, gravity=np.array([0., 0., -9.81]),
            solver=1, integrator=0, cone=1, iterations=50)
        names = [f'Arm{arm}/joint{k}' for arm in (10, 14) for k in range(1, 7)]
        names += ['Arm10/finger_left_joint', 'Arm10/finger_right_joint', 'ball/free']
        self.joints = {name:i for i,name in enumerate(names)}
        self.bodies = {name:i for i,name in enumerate([f'Arm{arm}/link{k}' for arm in (10, 14) for k in range(1, 7)] + ['bucket', 'Arm14/impact_board'])}
        self.jnt_qposadr = np.arange(15); self.jnt_dofadr = np.arange(15)
        self.body_mocapid = np.full(14, -1); self.body_mocapid[self.bodies['bucket']] = 0
        self.actuator_trnid = np.asarray([[i, 0] for i in range(12)])
        self.ngeom = 3

    def joint(self, name): return NS(id=self.joints[name])
    def body(self, name): return NS(id=self.bodies[name])


def fake_dependencies():
    state = NS(simulations=[], sensors=[], models=[], make_calls=[], checked_bounds=[])
    class Parent:
        def __init__(self, *args, **kwargs): raise AssertionError('Do not allocate an old64 engine first')
        def run(self): return 'unchanged-parent-run'
        def _collision_gate(self): return 'unchanged-parent-collision'
    class Spec:
        def compile(self):
            m = Model(); state.models.append(m); return m
    def make_model(scene, assets, mode):
        state.make_calls.append((scene, Path(assets), mode)); return Spec(), Model()
    class Sensor:
        def initialize(self, *args): self.args=args; state.sensors.append(self)
    def sensor_config(spec, model): return [Sensor(), Sensor()]
    class Simulation:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs); native=model=kwargs['model']; cfg=kwargs['cfg'].mujoco
            native.opt.timestep=cfg.timestep; native.opt.tolerance=cfg.tolerance
            native.opt.gravity=np.asarray(cfg.gravity); native.opt.solver={'newton':2}[cfg.solver]
            native.opt.integrator={'implicitfast':3}[cfg.integrator]; native.opt.cone={'pyramidal':0}[cfg.cone]
            native.opt.iterations=cfg.iterations
            native.opt.gravity.setflags(write=False); native.opt._locked=True
            _, warp=options(round(1/(240*cfg.timestep)), worlds=2)
            self.mj_model=native; self.model=NS(opt=warp); self.data=object()
            state.simulations.append(self)
    deps = NS(ContactEnvironment=Parent, torch=NS(cuda=NS(device_count=lambda:1,
        get_device_name=lambda _: 'NVIDIA GeForce RTX 4090')), make_model=make_model,
        checked_bounds=lambda bounds:state.checked_bounds.append(bounds), OLD_BOUNDS={'low':[0], 'high':[1]},
        sensor_config=sensor_config, Simulation=Simulation, SimulationCfg=lambda **kwargs:NS(**kwargs),
        MujocoCfg=lambda **kwargs:NS(**kwargs), ShotIK=lambda *args, **kwargs:NS(args=args, kwargs=kwargs),
        mujoco=NS(MjData=lambda m:NS(model=m), mjtObj=NS(mjOBJ_GEOM=5),
            mj_id2name=lambda m, kind, i:('bucket/base', 'Arm10/link1', 'bucket/wall')[i]))
    return deps,state


class TimestepOptionsTests(unittest.TestCase):
    def test_distinct_native_contracts_keep_scoring_rate_and_materials(self):
        values=[module.physics_settings(s) for s in module.SUBSTEPS]
        self.assertEqual(len({x['version'] for x in values}),3)
        self.assertEqual([x['native_dt_s'] for x in values],[1/7680,1/15360,1/30720])
        for value in values:
            self.assertEqual(value['scoring_dt_s'],1/240); self.assertEqual(value['solref'],[.004,.15])
            self.assertEqual(value['contact_friction'],.35); self.assertEqual(value['board_mode'],'articulated')
            self.assertEqual(value['gravity_mode'],'passive_only')
        for value in (8,16,256,64.,True,None):
            with self.assertRaises(ValueError):module.physics_settings(value)

    def test_native_and_all_expanded_warp_values_are_recorded_without_mutation(self):
        for s in module.SUBSTEPS:
            native,warp=options(s,3); before=(copy.deepcopy(native),copy.deepcopy(warp))
            result=module.validate_options(native,warp,s)
            self.assertEqual(result['substeps'],s);self.assertEqual(result['timestep_s'],1/(240*s))
            self.assertEqual(result['warp']['timestep_s'],np.full(3,1/(240*s),np.float32).tolist())
            self.assertEqual(result['warp']['dtypes']['timestep'],'float32')
            self.assertTrue(np.array_equal(native.gravity,before[0].gravity));self.assertTrue(np.array_equal(warp.timestep,before[1].timestep))

    def test_native_option_mismatch_fails_closed(self):
        for field,value in (('timestep',1/7680),('solver',1),('integrator',0),('cone',1),
                ('iterations',99),('tolerance',2e-8),('gravity',[0,0,-9.8])):
            native,warp=options();setattr(native,field,value)
            with self.subTest(field=field),self.assertRaises(RuntimeError):module.validate_options(native,warp,64)

    def test_warp_later_world_or_precision_mismatch_fails_closed(self):
        for field in ('timestep','tolerance','gravity','solver','iterations','dtype'):
            native,warp=options(worlds=2)
            if field in ('timestep','tolerance'):
                target=getattr(warp,field);target[1]=np.nextafter(target[1],np.float32(np.inf))
            elif field=='gravity':warp.gravity[1,2]=-9.8
            elif field=='dtype':warp.timestep=warp.timestep.astype(np.float64)
            else:setattr(warp,field,3)
            with self.subTest(field=field),self.assertRaises(RuntimeError):module.validate_options(native,warp,64)

    def test_missing_or_nonfinite_warp_values_fail(self):
        for value in (np.asarray([],np.float32),np.array([np.nan],np.float32)):
            native,warp=options();warp.timestep=value
            with self.assertRaises(RuntimeError):module.validate_options(native,warp,64)


class TimestepConstructionTests(unittest.TestCase):
    def test_invalid_parameters_or_uuid_never_import_native_dependencies(self):
        with patch.object(module,'_dependencies') as deps:
            for kwargs in (dict(substeps=16),dict(n=0),dict(n=True),dict(n=129),dict(board_mode='kinematic'),dict(gravity_mode='legacy')):
                with self.subTest(kwargs=kwargs),self.assertRaises(ValueError):module.TimestepEnvironment({},'/missing',**kwargs)
            with patch.dict(os.environ,{'CUDA_VISIBLE_DEVICES':'1'}),self.assertRaises(RuntimeError):
                module.TimestepEnvironment({},'/missing')
            deps.assert_not_called()

    def test_fresh_allocation_receives_timestep_before_warp_creation_and_inherits_run(self):
        deps,state=fake_dependencies()
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory);(path/'poses.json').write_text(json.dumps(dict(dropper_joint_rad=[0]*6,board_joint_rad=[0]*6)))
            with patch.object(module,'_dependencies',return_value=deps),patch.dict(os.environ,{'CUDA_VISIBLE_DEVICES':module.GPU_UUID}):
                envs=[module.TimestepEnvironment({'id':'fixture'},path,n=64,bounds={'low':[1],'high':[2]},substeps=s) for s in module.SUBSTEPS]
        self.assertEqual(len(state.simulations),3);self.assertEqual(len({id(e.sim) for e in envs}),3)
        for s,e in zip(module.SUBSTEPS,envs):
            self.assertIsInstance(e,deps.ContactEnvironment);self.assertEqual(type(e).run,deps.ContactEnvironment.run)
            self.assertEqual(type(e)._collision_gate,deps.ContactEnvironment._collision_gate)
            self.assertEqual(e.run(),'unchanged-parent-run');self.assertEqual(e.substeps,s)
            self.assertEqual(e.sim.cfg.mujoco.timestep,1/(240*s));self.assertEqual(e.sim.cfg.nconmax,512)
            self.assertEqual(e.sim.cfg.njmax,2048);self.assertEqual(e.sim.num_envs,64)
            self.assertEqual(e.board_mode,'articulated');self.assertEqual(e.gravity_mode,'passive_only')
            self.assertEqual(e.q10,list(range(6)));self.assertEqual(e.q14,list(range(6,12)))
            self.assertEqual(e.arm_q,list(range(12)));self.assertEqual(e.fingers,[12,13])
            self.assertEqual(e.ball_q,14);self.assertEqual(e.ball_v,14);self.assertEqual(e.bucket_geoms,[0,2])
            self.assertEqual(e.actuator_q,list(range(12)));self.assertIsNone(e.last_arrays)
            self.assertEqual(e.effective_options['version'],e.physics_version)
            self.assertIs(e.cpu.model,e.m);self.assertIs(e.attached_data.model,e.attached_model)
        self.assertEqual(len(state.sensors),6)
        for e,sensors in zip(envs,[state.sensors[:2],state.sensors[2:4],state.sensors[4:]]):
            for sensor in sensors:self.assertEqual(sensor.args,(e.sim.mj_model,e.sim.model,e.sim.data,'cuda:0'))
        self.assertTrue(all(call[2]=='articulated' for call in state.make_calls))

    def test_wrong_actual_device_or_count_never_builds_model(self):
        for count,name in ((2,'NVIDIA GeForce RTX 4090'),(1,'NVIDIA GeForce RTX 4070')):
            deps,state=fake_dependencies();deps.torch.cuda=NS(device_count=lambda:count,get_device_name=lambda _:name)
            with patch.object(module,'_dependencies',return_value=deps),patch.dict(os.environ,{'CUDA_VISIBLE_DEVICES':module.GPU_UUID}),self.assertRaises(RuntimeError):
                module.TimestepEnvironment({},'/not-read')
            self.assertFalse(state.make_calls)


if __name__=='__main__':unittest.main()
