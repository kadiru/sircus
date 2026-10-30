"""Deterministic dry-run adapter. Imports no SDK and has no hardware connection."""
import asyncio
import copy
import json
import time
from .controller import utc, write_json
from .viam_adapter import linear_move_parameters, validate_linear_start, linear_move_evidence


class MockAdapter:
    def __init__(self):
        pose = dict(x=450., y=0., z=300., o_x=0., o_y=0., o_z=-1., theta=0.)
        self.state = {'arms': {name:{'pose':dict(pose),'joints':[0.,-25.,35.,0.,65.,0.],
            'is_moving':False, 'pose_frame':'world', 'pose_component':name} for name in ('arm','arm-board')}}
        self.state['arms']['arm-board']['pose']['x'] = 850.
        self.commands = []
        self.record = {'active':False,'ready':False,'frame_count':0,'error':None}
        self.record_task = None
        self.directory = None

    async def connect(self):
        return {'mock':True,'local_recording':True,'native_capture_available':False,'cloud_upload':'disabled in dry run'}

    async def snapshot(self):
        captured = utc()
        return copy.deepcopy({**self.state,'captured_at':captured,'request_started_at':captured,'provenance':'mock fixture, not hardware'})

    async def camera_frame(self):
        svg = '<svg xmlns="http://www.w3.org/2000/svg" width="960" height="540"><rect width="960" height="540" fill="#123238"/><text x="480" y="250" text-anchor="middle" fill="#f7f3e8" font-family="sans-serif" font-size="42">DRY-RUN CAMERA</text><text x="480" y="300" text-anchor="middle" fill="#6fb9b4" font-family="sans-serif" font-size="24">No physical camera or robot commands</text></svg>'
        return svg.encode(), 'image/svg+xml', {'mock':True,'received_at':utc()}

    async def open_gripper(self):
        self.commands.append('open_gripper')

    async def close_gripper(self):
        self.commands.append('close_gripper')
        return {'grabbed':True}

    async def move_pose(self, pose):
        self.commands.append({'move_pose':dict(pose)})
        await self._complete_move(pose)

    async def move_pose_linear(self, pose, expected_start, line_tolerance_mm=2.0, orientation_tolerance_deg=1.0):
        parameters = linear_move_parameters(pose, expected_start, line_tolerance_mm, orientation_tolerance_deg)
        started = time.monotonic()
        validate_linear_start(await self.snapshot(), parameters, time.monotonic() - started)
        self.commands.append({'move_pose_linear':dict(pose), **linear_move_evidence(parameters)})
        await self._complete_move(pose)
        return {**linear_move_evidence(parameters), 'mock':True, 'planner_reported_success':False}

    async def _complete_move(self, pose):
        self.state['arms']['arm']['is_moving'] = True
        try:
            await asyncio.sleep(0.5)
            self.state['arms']['arm']['pose'] = dict(pose)
        finally:
            self.state['arms']['arm']['is_moving'] = False

    async def stop(self):
        self.commands.append('stop_dropper_only')
        self.state['arms']['arm']['is_moving'] = False
        return {'planner_cancellation_verified':True,'gripper_or_vacuum_commanded':False}

    async def start_recording(self, trial_id, trial_dir, metadata):
        self.directory = trial_dir
        self.directory.mkdir(parents=True, exist_ok=True)
        self.record = {'active':True,'ready':False,'frame_count':0,'error':None,'trial_id':trial_id,'mock':True}
        self.record_task = asyncio.create_task(self._record())

    async def _record(self):
        while self.record['active']:
            data, _, _ = await self.camera_frame()
            name = f"mock-frame-{self.record['frame_count']:04d}.svg"
            (self.directory/name).write_bytes(data)
            with (self.directory/'observations.jsonl').open('a') as f:
                f.write(json.dumps({'mock':True,'timestamp':utc(),'image':name,'state':await self.snapshot()})+'\n')
            self.record.update(ready=True,frame_count=self.record['frame_count']+1,last_frame_monotonic=time.monotonic())
            await asyncio.sleep(0.25)

    async def recording_status(self):
        return dict(self.record)

    async def finish_recording(self):
        self.record.update(active=False,ready=False)
        if self.record_task:
            await self.record_task
        if self.directory:
            write_json(self.directory/'recording.json',self.record)
        return dict(self.record)

    async def save_trial(self, trial_dir, metadata):
        write_json(trial_dir/'trial-metadata.json',metadata)
        return {'state':'mock_saved','cloud_verified':False,'mock':True}

    async def close(self):
        await self.finish_recording()
