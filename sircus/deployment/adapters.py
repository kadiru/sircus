"""No physical adapter is provided. Mock fixtures are explicitly synthetic."""
import asyncio
import time
from uuid import uuid4
from .contracts import (Evidence, Pose, RobotState, BoardEstimate, RGBDObservation,
                        BucketEstimate, VerifiedTransform, ReleasePlan)

SYNTHETIC = Evidence("synthetic", "deterministic deployment rehearsal fixture", True,
                     ("Not a camera measurement, calibration, physics prediction, or hardware validation.",))
IDENTITY = ((1.,0.,0.),(0.,1.,0.),(0.,0.,1.))


class NotCommissioned(RuntimeError):
    pass


class UncommissionedAdapter:
    """Typed port remains blocked; no resource lookup, connection, or robot I/O."""
    async def snapshot(self): raise NotCommissioned("Physical execution adapter is not implemented/commissioned")
    async def board_from_kinematics(self, state): raise NotCommissioned("Board mount and base transforms unverified")
    async def move_to_pose(self, arm, pose, operation_id): raise NotCommissioned("Physical motion is disabled")
    async def stop_motion(self, arm, operation_id): raise NotCommissioned("No physical motion ownership exists")
    async def capture_rgbd(self): raise NotCommissioned("No connected camera adapter")
    async def camera_world_transform(self, observation): raise NotCommissioned("Camera-to-world calibration missing")
    async def plan_release(self, board, bucket, revision): raise NotCommissioned("No commissioned release planner")
    async def start_recording(self, trial_id, metadata): raise NotCommissioned("No connected recorder")
    async def finish_recording(self, trial_id): return None
    async def close_gripper(self): raise NotCommissioned("Physical tool actuation is disabled")
    async def open_gripper(self): raise NotCommissioned("Physical tool actuation is disabled")
    async def at_pose_and_settled(self, pose): return False
    async def measured_release_pose(self): raise NotCommissioned("No pose readback")
    async def recording_status(self): return False


class UnknownBucketDetector:
    async def detect(self, observation):
        return None  # Unknown, not a failed capture and not a fabricated detection.


class MockBucketDetector:
    async def detect(self, observation):
        return BucketEstimate(observation.observation_id, observation.frame, (25.,50.,600.),
                              (0.,0.,-1.), 88.5, 2., SYNTHETIC)


class MockAdapter:
    def __init__(self):
        self.commands = []
        self.board_joints = (0.,)*6
        self.dropper_joints = (0.,)*6
        self.delay_s = 0.
        self.grasp_confirmed = True
        self.recording_ready = True
        self.settled = True
        self.last_pose = None
        self.board_pose = Pose("world", (600.,-150.,300.), (0.,0.,1.,0.), SYNTHETIC)
        self.recording = False
        self.loading_pose = Pose("world", (150.,-600.,200.), (-.01,-.01,-1.,-85.26), SYNTHETIC)
        self.scan_pose = Pose("world", (100.,-350.,500.), (0.,0.,-1.,0.), SYNTHETIC)

    async def snapshot(self):
        return RobotState(time.monotonic(), self.dropper_joints, self.board_joints,
                          False, False, SYNTHETIC)

    async def board_from_kinematics(self, state):
        # This adapter is a contract fixture, not the numerical simulator's FK.
        # Real implementation must compose arm14 FK + calibrated suction/board mount.
        return BoardEstimate(self.board_pose.xyz_mm, self.board_pose.orientation_vector_deg[:3], state.board_joints_deg,
                             "world", "synthetic-mount", SYNTHETIC)

    async def move_to_pose(self, arm, pose, operation_id):
        self.commands.append(("move", arm, operation_id))
        await asyncio.sleep(self.delay_s)
        if arm == "arm-board":
            self.board_pose = pose
        else:
            self.last_pose = pose

    async def stop_motion(self, arm, operation_id):
        self.commands.append(("stop", arm, operation_id))

    async def capture_rgbd(self):
        before = await self.snapshot()
        return RGBDObservation(str(uuid4()), "cam", "cam", time.monotonic(), "SYNTHETIC",
                               "synthetic:rgb", "synthetic:depth", 1., True, True, True,
                               before, await self.snapshot(), SYNTHETIC,
                               ("Synthetic fixture; no real exposure synchronization tested.",))

    async def camera_world_transform(self, observation):
        return VerifiedTransform("cam", "world", IDENTITY, (0.,0.,0.), observation.observation_id,
                                 observation.camera_serial, "synthetic-camera-world", SYNTHETIC)

    async def plan_release(self, board, bucket, revision):
        return ReleasePlan(str(uuid4()), revision, bucket.observation_id, board.joints_deg,
                           Pose("world", (600.,-150.,600.), (-.01,-.01,-1.,-85.26), SYNTHETIC),
                           True, True, True, SYNTHETIC)

    async def start_recording(self, trial_id, metadata):
        self.commands.append(("record_start", trial_id, metadata))
        self.recording = self.recording_ready
        return self.recording_ready

    async def finish_recording(self, trial_id):
        self.commands.append(("record_finish", trial_id))
        self.recording = False

    async def close_gripper(self):
        self.commands.append(("close_gripper",))
        return self.grasp_confirmed

    async def open_gripper(self):
        self.commands.append(("open_gripper",))

    async def at_pose_and_settled(self, pose):
        return self.settled and self.last_pose == pose

    async def measured_release_pose(self):
        return self.last_pose

    async def recording_status(self):
        return self.recording and self.recording_ready
