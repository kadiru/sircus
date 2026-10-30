import asyncio
from dataclasses import replace
import json
from pathlib import Path
import time
import unittest
from sircus.deployment.adapters import MockAdapter, MockBucketDetector, UnknownBucketDetector, SYNTHETIC
from sircus.deployment.contracts import camera_to_world, Evidence, RobotState
from sircus.deployment.controller import DeploymentController


async def ready(controller):
    for action in ("confirm_board", "confirm_bucket_placed", "scan_bucket", "review_bucket", "go_loading", "confirm_loaded"):
        await controller.command({"action": action, "confirmed": True})
    return controller.status()


class DeploymentTests(unittest.IsolatedAsyncioTestCase):
    async def test_blocked_default_rejects_all_actuation(self):
        c = DeploymentController()
        self.assertFalse(c.status()["hardware_enabled"])
        for action in ("grip_drop", "position_board", "scan_bucket"):
            with self.assertRaises(ValueError):
                await c.command({"action": action})
        with self.assertRaises(ValueError):
            DeploymentController("hardware")
        with self.assertRaises(ValueError):
            DeploymentController("dry_run", adapter=object())
        self.assertEqual(c.status()["phase"], "blocked")

    async def test_complete_dry_run_order_ids_unknown_outcome(self):
        adapter = MockAdapter()
        c = DeploymentController("dry_run", adapter=adapter)
        self.assertEqual((await ready(c))["phase"], "ready_drop")
        s = await c.command({"action": "grip_drop"})
        self.assertEqual(s["phase"], "complete")
        self.assertEqual(s["trial"]["outcome"], "unknown")
        calls = [v[0] for v in adapter.commands]
        self.assertEqual(calls, ["move", "move", "open_gripper", "record_start", "close_gripper", "move", "open_gripper", "record_finish"])
        trial_events = [e for e in s["events"] if e["kind"] == "release_commanded"]
        self.assertEqual(trial_events[0]["trial_id"], s["trial"]["trial_id"])
        self.assertEqual(s["trial"]["measured_release_pose"]["evidence"]["source"], "synthetic")
        json.dumps(s, allow_nan=False)

    async def test_unknown_detection_never_becomes_world_estimate(self):
        c = DeploymentController("dry_run", detector=UnknownBucketDetector())
        for action in ("confirm_board", "confirm_bucket_placed", "scan_bucket"):
            s = await c.command({"action": action})
        self.assertEqual(s["phase"], "blocked_trial")
        self.assertIsNone(s["bucket"])
        self.assertIn("unknown", s["error"])

    async def test_missing_calibration_and_mixed_provenance(self):
        a = MockAdapter()
        observation = await a.capture_rgbd()
        detection = await MockBucketDetector().detect(observation)
        transform = await a.camera_world_transform(observation)
        self.assertEqual(camera_to_world(detection, observation, transform).frame, "world")
        for changed in (replace(transform, evidence=Evidence("kinematic", "unverified")),
                        replace(transform, camera_serial="wrong"),
                        replace(transform, observation_id="old"),
                        replace(transform, rotation=((-1.,0.,0.),(0.,1.,0.),(0.,0.,1.))),
                        replace(transform, evidence=Evidence("kinematic", "physical", True))):
            with self.assertRaises(ValueError):
                camera_to_world(detection, observation, changed)
        with self.assertRaises(ValueError):
            camera_to_world(detection, replace(observation, depth_scale_verified=False), transform)

    async def test_board_change_stale_scan_and_unvalidated_plan(self):
        for change in ("board", "stale", "plan"):
            a = MockAdapter()
            c = DeploymentController("dry_run", adapter=a)
            await ready(c)
            before = len(a.commands)
            if change == "board": a.board_joints = (1.,0.,0.,0.,0.,0.)
            if change == "stale": c.observation = replace(c.observation, timestamp_monotonic=time.monotonic()-121)
            if change == "plan": c.plan = replace(c.plan, trajectory_checked=False)
            s = await c.command({"action": "grip_drop"})
            self.assertEqual(s["phase"], "blocked_trial")
            self.assertEqual(len(a.commands), before)

    async def test_recording_and_grasp_failure_prevent_release(self):
        for failure in ("recording_ready", "grasp_confirmed", "settled"):
            a = MockAdapter()
            c = DeploymentController("dry_run", adapter=a)
            await ready(c)
            setattr(a, failure, False)
            before = len(a.commands)
            s = await c.command({"action": "grip_drop"})
            self.assertEqual(s["phase"], "blocked_trial")
            self.assertNotIn("open_gripper", [v[0] for v in a.commands[before:]])

    async def test_board_motion_cancel_stops_only_owned_without_tool_changes(self):
        a = MockAdapter()
        a.delay_s = 1
        c = DeploymentController("dry_run", adapter=a)
        task = asyncio.create_task(c.command({"action": "position_board", "xyz_mm": [600,-150,300],
                                            "orientation_vector_deg": [0,0,1,0]}))
        await asyncio.sleep(.01)
        with self.assertRaises(ValueError):
            await c.command({"action": "confirm_board"})
        s = await c.stop()
        await asyncio.gather(task, return_exceptions=True)
        self.assertEqual(s["phase"], "cancelled")
        self.assertEqual([v[0] for v in a.commands], ["move", "stop"])
        self.assertEqual(a.commands[-1][1], "arm-board")
        self.assertEqual(a.commands[0][2], a.commands[1][2])

    async def test_release_motion_cancel_keeps_gripper_closed(self):
        a = MockAdapter()
        c = DeploymentController("dry_run", adapter=a)
        await ready(c)
        before = len(a.commands)
        a.delay_s = 1
        task = asyncio.create_task(c.command({"action": "grip_drop"}))
        await asyncio.sleep(.01)
        await c.stop()
        await asyncio.gather(task, return_exceptions=True)
        calls = [v[0] for v in a.commands[before:]]
        self.assertIn("close_gripper", calls)
        self.assertIn("stop", calls)
        self.assertNotIn("open_gripper", calls)
        self.assertEqual(c.trial["state"], "cancelled")

    async def test_timeout_stops_motion_and_reset_clears_evidence(self):
        a = MockAdapter()
        a.delay_s = .1
        c = DeploymentController("dry_run", adapter=a, timeout_s=.01)
        s = await c.command({"action": "position_board", "xyz_mm": [600,-150,300],
                             "orientation_vector_deg": [0,0,1,0]})
        self.assertEqual(s["phase"], "blocked_trial")
        self.assertEqual([x[0] for x in a.commands], ["move", "stop"])
        s = await c.command({"action": "reset"})
        self.assertEqual(s["phase"], "configure_board")
        self.assertIsNone(s["plan"])

    async def test_capture_joint_motion_and_unverified_detection_are_blocked(self):
        class MovingCapture(MockAdapter):
            async def capture_rgbd(self):
                obs = await super().capture_rgbd()
                return replace(obs, state_after=replace(obs.state_after, dropper_joints_deg=(1.,0.,0.,0.,0.,0.)))
        c = DeploymentController("dry_run", adapter=MovingCapture())
        for action in ("confirm_board", "confirm_bucket_placed", "scan_bucket"):
            result = await c.command({"action": action})
        self.assertEqual(result["phase"], "blocked_trial")
        self.assertIsNone(result["bucket"])
        a = MockAdapter()
        observation = await a.capture_rgbd()
        detection = await MockBucketDetector().detect(observation)
        with self.assertRaises(ValueError):
            camera_to_world(replace(detection, evidence=replace(SYNTHETIC, verified=False)),
                            observation, await a.camera_world_transform(observation))

    async def test_failed_stop_retains_ownership_and_blocks_reset(self):
        class FailingStop(MockAdapter):
            async def stop_motion(self, arm, operation_id):
                raise RuntimeError("mock stop failure")
        a = FailingStop(); a.delay_s = 1
        c = DeploymentController("dry_run", adapter=a)
        task = asyncio.create_task(c.command({"action": "position_board", "xyz_mm": [600,-150,300],
                                             "orientation_vector_deg": [0,0,1,0]}))
        await asyncio.sleep(.01)
        result = await c.stop()
        await asyncio.gather(task, return_exceptions=True)
        self.assertEqual(result["phase"], "stop_failed")
        self.assertEqual(result["owned_motion_resources"], ["arm-board"])
        self.assertNotIn("reset", result["available_actions"])

    async def test_unknown_stationarity_is_rejected(self):
        with self.assertRaises(ValueError):
            RobotState(time.monotonic(), (0.,)*6, (0.,)*6, None, False, SYNTHETIC)


class ViamModuleTests(unittest.IsolatedAsyncioTestCase):
    async def test_module_construction_has_no_connections(self):
        from viam.proto.app.robot import ComponentConfig
        from google.protobuf.struct_pb2 import Struct
        from sircus.deployment.viam_module import OperationCoordinator, BucketVisionStub, validate_coordinator, register_models
        register_models()  # Registration only, no socket/server start.
        config = ComponentConfig(name="sircus-operation-coordinator")
        coordinator = OperationCoordinator.new(config, {})
        result = await coordinator.do_command({"action": "capabilities"})
        self.assertFalse(result["hardware_enabled"])
        self.assertEqual(result["mode"], "blocked")
        vision = BucketVisionStub("sircus-bucket-vision")
        props = await vision.get_properties()
        self.assertFalse(props.detections_supported)
        self.assertFalse(props.detections_3d_supported)
        vision_status = await vision.do_command({"action":"status"})
        self.assertIsNone(vision_status["estimate"])
        self.assertIs(vision_status["estimate_available"], False)
        self.assertEqual(vision_status["status"], "unknown")
        self.assertIs(vision_status["camera_read"], False)
        with self.assertRaises(NotImplementedError):
            await vision.get_detections_from_camera("cam")
        bad = Struct(); bad.update({"hardware_enabled": True})
        with self.assertRaises(ValueError):
            validate_coordinator(ComponentConfig(name="bad", attributes=bad))
        await coordinator.close()

    async def test_fragment_default_is_blocked_no_dependencies(self):
        path = Path(__file__).parents[1]/"viam-fragment.template.json"
        config = json.loads(path.read_text())
        self.assertEqual(config["services"][0]["attributes"]["mode"], "blocked")
        self.assertFalse(config["services"][0]["attributes"]["hardware_enabled"])
        self.assertTrue(all("depends_on" not in item for item in config["services"]))


if __name__ == "__main__":
    unittest.main()
