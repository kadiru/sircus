"""Hardware-free adapter contract checks; no credentials or network needed.

Run: sircus/simulation/.venv/bin/python -m unittest discover -s sircus/physical_trials -p test_viam_adapter.py
"""
import asyncio
import copy
from contextlib import ExitStack
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from viam_adapter import ViamAdapter, PART_ID
from viam.proto.common import Pose, PoseInFrame, ResponseMetadata
from viam.proto.component.arm import JointPositions
from viam.proto.robot import Operation
from google.protobuf.struct_pb2 import Struct


class Arm:
    def __init__(self, name, log):
        self.name, self.log = name, log

    async def get_joint_positions(self, **kw):
        return JointPositions(values=[0] * 6)

    async def is_moving(self, **kw):
        return False

    async def stop(self, **kw):
        self.log.append("stop:" + self.name)


class Camera:
    async def get_images(self, **kw):
        return [SimpleNamespace(name="color", mime_type="image/jpeg", data=b"test-jpeg",
                                width=2, height=2)], ResponseMetadata()


class Robot:
    def __init__(self, log):
        self.log, self.token = log, None

    async def get_operations(self):
        # Include an unrelated board-arm operation; Stop must never cancel it.
        other = Struct()
        other.update({"component_name": "arm-board", "extra": {"other_client": True}})
        ops = [Operation(id="other-board-client", method="/viam.service.motion.v1.MotionService/Move", arguments=other)]
        if self.token:
            args = Struct()
            args.update({"component_name": "arm", "extra": {"sircus_operator_operation": self.token}})
            ops.append(Operation(id="only-panel-move", method="/viam.service.motion.v1.MotionService/Move", arguments=args))
        return ops

    async def cancel_operation(self, operation_id):
        self.log.append("cancel:" + operation_id)
        self.token = None


class Motion:
    def __init__(self, robot):
        self.robot = robot

    async def get_pose(self, *args, **kw):
        return PoseInFrame(reference_frame="world", pose=Pose(x=100, z=200, o_z=-1))

    async def move(self, **kw):
        assert kw["component_name"] == "arm"
        assert kw["destination"].reference_frame == "world"
        self.robot.token = kw["extra"]["sircus_operator_operation"]
        await asyncio.Event().wait()


class OfflineAdapter(ViamAdapter):
    async def connect(self):
        return {"connected": True}

    def _credentials(self):
        return {"api_key_id": "mock-key-id", "api_key": "mock-key"}


class Data:
    def __init__(self):
        self.calls = []

    async def binary_data_capture_upload(self, **kw):
        self.calls.append(("binary", kw))
        return "binary-id"

    async def tabular_data_capture_upload(self, **kw):
        self.calls.append(("tabular", kw))
        return "tabular-id-" + kw["component_name"]

    async def file_upload(self, **kw):
        self.calls.append(("file", kw))
        return "file-id-" + kw["file_name"]

    async def create_sequence(self, **kw):
        self.calls.append(("sequence", kw))
        return "sequence-id"


class AdapterTests(unittest.IsolatedAsyncioTestCase):
    async def test_physical_cloud_connection_enables_webrtc_by_default(self):
        required = [('component','arm','arm'),('component','arm','arm-board'),
                    ('component','gripper','gripper'),('component','camera','cam'),
                    ('service','motion','builtin')]
        robot=SimpleNamespace(resource_names=[SimpleNamespace(type=t,subtype=s,name=n) for t,s,n in required])
        captured={}
        async def connect(address,options):
            captured.update(address=address,options=options)
            return robot
        adapter=ViamAdapter('/not-read')
        with ExitStack() as stack:
            stack.enter_context(patch.object(adapter,'_credentials',return_value={'api_key_id':'mock-id','api_key':'mock-key'}))
            stack.enter_context(patch('viam.robot.client.RobotClient.at_address',connect))
            for klass in ('viam.components.arm.Arm','viam.components.camera.Camera',
                          'viam.components.gripper.Gripper','viam.services.motion.MotionClient'):
                stack.enter_context(patch(klass+'.from_robot',return_value=object()))
            result=await adapter.connect()
        self.assertFalse(captured['options'].dial_options.disable_webrtc)
        self.assertFalse(captured['options'].dial_options.insecure)
        self.assertEqual(captured['address'],'sfhack10-main.ty1cywh7tq.viam.cloud')
        self.assertEqual(result['transport'],'SDK default; WebRTC enabled')

    def adapter(self):
        log = []
        adapter = OfflineAdapter("/not-read", {"upload_to_viam": False})
        adapter.robot = Robot(log)
        adapter.arms = {n: Arm(n, log) for n in ("arm", "arm-board")}
        adapter.motion = Motion(adapter.robot)
        adapter.camera = Camera()
        return adapter, log

    async def record(self, adapter, directory):
        await adapter.start_recording("trial-mock", directory)
        for _ in range(100):
            if (await adapter.recording_status())["ready"]:
                break
            await asyncio.sleep(.01)
        status = await adapter.recording_status()
        self.assertTrue(status["ready"])
        self.assertGreater(status["last_frame_monotonic"], 0)
        self.assertTrue(Path(directory, "observations.jsonl").exists())
        await adapter.finish_recording()
        self.assertFalse((await adapter.recording_status())["ready"])

    async def test_recorder_requires_saved_camera_and_both_arm_state(self):
        adapter, _ = self.adapter()
        with tempfile.TemporaryDirectory() as directory:
            await self.record(adapter, directory)
            row = json.loads(Path(directory, "observations.jsonl").read_text().splitlines()[0])
            self.assertEqual(set(row["state"]["arms"]), {"arm", "arm-board"})
            self.assertTrue(Path(directory, row["images"][0]["file"]).exists())
            adapter._last_frame_mono = 0
            adapter._record["active"] = True
            self.assertFalse((await adapter.recording_status())["ready"])

    async def test_stop_only_cancels_panel_dropper_and_never_board_or_tools(self):
        adapter, log = self.adapter()
        task = asyncio.create_task(adapter.move_pose(dict(x=100., y=0., z=200., o_x=0., o_y=0., o_z=-1., theta=0.)))
        await asyncio.sleep(.01)
        result = await adapter.stop()
        await asyncio.gather(task, return_exceptions=True)
        self.assertTrue(result["planner_cancellation_verified"])
        self.assertEqual(log.count("stop:arm"), 2)
        self.assertEqual(log.count("cancel:only-panel-move"), 1)
        self.assertEqual(len(log), 3)
        self.assertFalse(hasattr(adapter, "vacuum"))

    async def test_repeated_stop_preserves_unverified_planner_cancellation(self):
        adapter, log = self.adapter()
        adapter._move_active = True
        adapter._move_token = "undiscovered-panel-move"
        # Robot returns only an unrelated board operation: no matching planner
        # operation is discovered, even though an own move remains unverified.
        first = await adapter.stop()
        self.assertFalse(first["planner_cancellation_verified"])
        self.assertTrue(adapter._move_active)
        self.assertEqual(adapter._move_token, "undiscovered-panel-move")
        second = await adapter.stop()
        self.assertFalse(second["planner_cancellation_verified"])
        self.assertTrue(adapter._move_active)
        self.assertEqual(adapter._move_token, "undiscovered-panel-move")
        self.assertEqual(log, ["stop:arm"] * 4)

    def vertical_poses(self):
        start = dict(x=100., y=0., z=200., o_x=0., o_y=0., o_z=-1., theta=0.)
        return start, dict(start, z=100.)

    async def test_vertical_move_sends_typed_bounds_and_preserves_ownership(self):
        adapter, _ = self.adapter();calls=[]
        async def moved(**kw):
            calls.append(kw)
            return True
        adapter.motion.move = moved
        start, target = self.vertical_poses()
        result = await adapter.move_pose_linear(target, start)
        self.assertEqual(len(calls), 1)
        request = calls[0]
        self.assertEqual(request['component_name'], 'arm')
        self.assertEqual(request['destination'].reference_frame, 'world')
        self.assertEqual(request['destination'].pose.z, 100.)
        constraint = request['constraints'].linear_constraint[0]
        self.assertEqual(constraint.line_tolerance_mm, 2.)
        self.assertEqual(constraint.orientation_tolerance_degs, 1.)
        self.assertTrue(request['extra']['sircus_operator_operation'])
        self.assertTrue(result['planner_reported_success'])
        self.assertFalse(result['physical_path_verified'])
        self.assertIsNone(adapter._move_task)

    async def test_vertical_move_rejects_bad_endpoints_before_connecting(self):
        adapter, _ = self.adapter()
        async def forbidden():
            raise AssertionError('Invalid parameters must not connect')
        adapter.connect = forbidden
        start, target = self.vertical_poses()
        cases = [(dict(target, x=101), start, 2., 1.), (dict(target, theta=1.), start, 2., 1.),
                 (start, start, 2., 1.), (dict(target, z=float('nan')), start, 2., 1.),
                 (target, start, True, 1.), (target, start, 2., 0.),
                 (dict(target, o_z=0), dict(start, o_z=0), 2., 1.)]
        for end, beginning, line, orientation in cases:
            with self.subTest(end=end, line=line, orientation=orientation):
                with self.assertRaises(ValueError):
                    await adapter.move_pose_linear(end, beginning, line, orientation)

    async def test_vertical_move_refuses_stale_moving_or_wrong_start_without_actuation(self):
        adapter, log = self.adapter();calls=[]
        async def moved(**kw):calls.append(kw);return True
        adapter.motion.move = moved
        original = await adapter.snapshot()
        stale = (datetime.now(timezone.utc) - timedelta(seconds=4)).isoformat()
        cases=[]
        for arm in ('arm', 'arm-board'):
            state=copy.deepcopy(original);state['arms'][arm]['is_moving']=True;cases.append(state)
        for field, value in [('x',103.), ('theta',2.)]:
            state=copy.deepcopy(original);state['arms']['arm']['pose'][field]=value;cases.append(state)
        state=copy.deepcopy(original);state['request_started_at']=stale;cases.append(state)
        state=copy.deepcopy(original);state['arms']['arm']['pose_frame']='camera';cases.append(state)
        start, target = self.vertical_poses()
        for state in cases:
            async def snapshot():return state
            adapter.snapshot=snapshot
            with self.subTest(state=state):
                with self.assertRaises(RuntimeError):await adapter.move_pose_linear(target,start)
                self.assertIsNone(adapter._move_task)
        self.assertEqual(calls, []);self.assertEqual(log, [])

    async def test_rejected_linear_constraint_never_retries_unconstrained(self):
        adapter, log = self.adapter();calls=[]
        async def rejected(**kw):
            calls.append(kw)
            adapter.robot.token=kw['extra']['sircus_operator_operation']
            raise RuntimeError('linear constraints unsupported')
        adapter.motion.move=rejected
        start,target=self.vertical_poses()
        with self.assertRaisesRegex(RuntimeError,'unsupported'):
            await adapter.move_pose_linear(target,start)
        self.assertEqual(len(calls),1);self.assertIn('constraints',calls[0])
        self.assertEqual(log.count('cancel:only-panel-move'),1)
        self.assertFalse(any('board' in entry for entry in log))

    async def test_linear_start_owns_move_slot_and_rejects_slow_readback(self):
        adapter, log = self.adapter();entered=asyncio.Event();proceed=asyncio.Event();calls=[]
        state=await adapter.snapshot()
        async def snapshot():
            entered.set();await proceed.wait();return state
        async def moved(**kw):calls.append(kw);return True
        adapter.snapshot=snapshot;adapter.motion.move=moved
        start,target=self.vertical_poses()
        with patch('viam_adapter.LINEAR_START_FRESH_SECONDS', .005):
            task=asyncio.create_task(adapter.move_pose_linear(target,start))
            await entered.wait()
            with self.assertRaisesRegex(RuntimeError,'already running'):
                await adapter.move_pose(start)
            await asyncio.sleep(.01);proceed.set()
            with self.assertRaisesRegex(RuntimeError,'freshness'):
                await task
        self.assertEqual(calls,[]);self.assertEqual(log,[]);self.assertIsNone(adapter._move_task)

    async def test_stop_during_linear_start_readback_prevents_late_motion(self):
        adapter,log=self.adapter();entered=asyncio.Event();calls=[]
        async def snapshot():
            entered.set()
            await asyncio.Event().wait()
        async def moved(**kw):calls.append(kw);return True
        adapter.snapshot=snapshot;adapter.motion.move=moved
        start,target=self.vertical_poses()
        task=asyncio.create_task(adapter.move_pose_linear(target,start))
        await entered.wait()
        stopped=await adapter.stop()
        await asyncio.gather(task,return_exceptions=True)
        self.assertTrue(stopped['planner_cancellation_verified'])
        self.assertEqual(calls,[]);self.assertEqual(log,['stop:arm','stop:arm'])

    async def test_stop_during_linear_motion_cancels_only_owned_dropper_plan(self):
        adapter, log=self.adapter();start,target=self.vertical_poses()
        task=asyncio.create_task(adapter.move_pose_linear(target,start))
        for _ in range(100):
            if adapter.robot.token:break
            await asyncio.sleep(.001)
        self.assertTrue(adapter.robot.token)
        stopped=await adapter.stop()
        await asyncio.gather(task,return_exceptions=True)
        self.assertTrue(stopped['planner_cancellation_verified'])
        self.assertEqual(log.count('stop:arm'),2)
        self.assertEqual(log.count('cancel:only-panel-move'),1)
        self.assertEqual(len(log),3)

    async def test_dry_run_linear_move_records_requested_bounds_without_hardware_claim(self):
        from sircus.physical_trials.mock_adapter import MockAdapter
        adapter=MockAdapter();start=copy.deepcopy(adapter.state['arms']['arm']['pose'])
        target=dict(start,z=start['z']-100.)
        result=await adapter.move_pose_linear(target,start)
        self.assertTrue(result['mock']);self.assertFalse(result['physical_path_verified'])
        self.assertFalse(result['planner_reported_success'])
        self.assertEqual(adapter.state['arms']['arm']['pose'],target)
        self.assertEqual(len(adapter.commands),1)
        self.assertIn('move_pose_linear',adapter.commands[0])
        with self.assertRaises(RuntimeError):await adapter.move_pose_linear(target,start)

    async def test_timestamped_upload_and_sequence_are_checkpointed(self):
        adapter, _ = self.adapter()
        data = Data()

        async def close():
            pass

        async def create(*a, **kw):
            return SimpleNamespace(data_client=data, close=close)

        with tempfile.TemporaryDirectory() as directory:
            await self.record(adapter, directory)
            adapter.config["upload_to_viam"] = True
            with patch("viam.app.viam_client.ViamClient.create_from_dial_options", create):
                result = await adapter.save_trial(directory, {"trial_id": "trial-mock", "outcome": "captured"})
                before = len(data.calls)
                again = await adapter.save_trial(directory, {"trial_id": "trial-mock", "outcome": "captured"})
            self.assertEqual(result["state"], "uploaded")
            self.assertEqual(again["sequence_id"], "sequence-id")
            self.assertEqual(len(data.calls), before)
            self.assertFalse(result["cloud_verified"])
            for kind, call in data.calls:
                self.assertEqual(call["part_id"], PART_ID)
                if kind == "sequence":
                    self.assertEqual(call["sequence_tags"], ["sircus-physical", "trial:trial-mock", "outcome:captured"])
                    self.assertEqual({x.resource_name for x in call["resources"]}, {"cam", "arm", "arm-board"})


if __name__ == "__main__":
    unittest.main()
