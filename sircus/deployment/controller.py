"""Operator-driven rehearsal and fail-closed commissioning interface.

No hardware mode exists. A future physical backend needs an independent review;
changing a JSON flag cannot unlock it. Stop never opens a tool or releases suction.
"""
from __future__ import annotations
import asyncio
from collections import deque
from datetime import datetime, timezone
import time
from uuid import uuid4
from .adapters import MockAdapter, MockBucketDetector, UncommissionedAdapter, UnknownBucketDetector, SYNTHETIC
from .contracts import Pose, camera_to_world, serializable, vector

RESOURCES = {"dropper": "arm", "board_arm": "arm-board", "camera": "cam", "gripper": "gripper",
             "board_vacuum": "vacuum-board", "capture_control": "capture-control", "motion": "builtin"}
BLOCKERS = [
    "Physical adapter is intentionally unimplemented; no hardware execution mode exists.",
    "Verify a single board-arm command owner; historical sfhack14 may reference the same controller.",
    "Calibrate arm10 tool-bracket camera extrinsics against a known world reference; pin serial 112322070710.",
    "Verify depth scale, RGB/depth alignment, intrinsics and timestamp-to-joint association.",
    "Implement and validate bucket detector on physical RGB-D; unknown is not a detection.",
    "Verify arm base transform and arm14-to-suction-to-board impact surface mount; current asset Z122 mm is provisional.",
    "Teach and validate scan/loading poses, release envelope, held-ball offset, and collision-free paths.",
    "Verify motion ownership/cancellation for both arms and recording readiness; commission capture-control/Sequences.",
    "Confirm bucket 175 mm means overall height or interior depth; current placement is provisional.",
]


class DeploymentController:
    def __init__(self, mode="blocked", *, adapter=None, detector=None, timeout_s=20., max_scan_age_s=120.):
        if mode not in ("blocked", "dry_run"):
            raise ValueError("Only blocked and dry_run modes exist; hardware is unavailable")
        self.mode = mode
        if mode == "blocked" and (adapter is not None or detector is not None):
            raise ValueError("Blocked mode does not accept execution dependencies")
        if mode == "dry_run" and adapter is not None and not isinstance(adapter, MockAdapter):
            raise ValueError("Dry-run requires MockAdapter; physical adapter injection is prohibited")
        self.adapter = adapter or (MockAdapter() if mode == "dry_run" else UncommissionedAdapter())
        self.detector = detector or (MockBucketDetector() if mode == "dry_run" else UnknownBucketDetector())
        self.timeout_s, self.max_scan_age_s = timeout_s, max_scan_age_s
        self._gate = asyncio.Lock()
        self._task = None
        self._stopping = False
        self._owned = {}  # arm -> owned operation UUID, not blanket robot stop
        self._events = deque(maxlen=200)
        self._revision = 0
        self._clear()

    def _clear(self):
        self.phase = "configure_board" if self.mode == "dry_run" else "blocked"
        self.board = self.observation = self.bucket = self.plan = self.trial = None
        self.error = None

    def _event(self, kind, **payload):
        self._events.append({"event_id": str(uuid4()), "kind": kind, "at_utc": datetime.now(timezone.utc).isoformat(),
                             "at_monotonic": time.monotonic(), "revision": self._revision,
                             "trial_id": self.trial["trial_id"] if self.trial else None,
                             "provenance": "synthetic rehearsal" if self.mode == "dry_run" else "blocked scaffold",
                             **payload})

    def status(self):
        busy = self._task is not None and not self._task.done()
        actions = ["status", "stop", "reset"]
        by_phase = {"configure_board": ["position_board", "confirm_board"], "place_bucket": ["confirm_bucket_placed"],
                    "ready_scan": ["scan_bucket"], "review_bucket": ["review_bucket"],
                    "ready_loading": ["go_loading"], "insert_ball": ["confirm_loaded"],
                    "ready_drop": ["grip_drop"]}
        if not busy and not self._stopping and self.mode == "dry_run":
            actions += by_phase.get(self.phase, [])
        if busy or self._stopping or self._owned:
            actions = ["status", "stop"]
        return {"schema": "sircus.deployment.v1", "mode": self.mode, "hardware_enabled": False,
                "machine_target": "sfhack10", "resources": RESOURCES, "phase": self.phase,
                "busy": busy or self._stopping, "revision": self._revision, "available_actions": actions,
                "blockers": BLOCKERS, "board": serializable(self.board), "bucket": serializable(self.bucket),
                "observation": serializable(self.observation), "plan": serializable(self.plan), "trial": self.trial,
                "error": self.error, "owned_motion_resources": sorted(self._owned), "events": list(self._events),
                "evidence_scope": "contract rehearsal only; not numerical physics, real vision, native replay, or calibration"}

    async def command(self, payload):
        action = payload.get("action", "status")
        if action in ("status", "capabilities"):
            return self.status()
        if action == "stop":
            return await self.stop()
        async with self._gate:
            if action not in self.status()["available_actions"]:
                raise ValueError(f"Action {action!r} unavailable in phase {self.phase!r}")
            if action == "reset":
                self._revision += 1
                self._clear()
                self._event("reset")
                return self.status()
            self.error = None
            self._task = asyncio.create_task(self._bounded(action, payload))
            task = self._task
        try:
            await task
        except asyncio.CancelledError:
            # RPC/client cancellation must not strand an owned action.
            await self.stop()
            raise
        return self.status()

    async def _bounded(self, action, payload):
        try:
            await asyncio.wait_for(self._perform(action, payload), self.timeout_s)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self.error = f"{type(exc).__name__}: {str(exc)[:400]}"
            self.phase = "blocked_trial"
            self._event("error", action=action, message=self.error)
            await self._stop_owned()
            if self.trial:
                self.trial["state"] = "failed"
                await self._finish_recording()

    async def _move(self, arm, pose):
        if pose.frame != "world" or not pose.evidence.verified:
            raise ValueError("Motion pose must have verified world-frame evidence")
        operation = str(uuid4())
        self._owned[arm] = operation
        self._event("motion_requested", arm=arm, operation_id=operation, pose=serializable(pose))
        await self.adapter.move_to_pose(arm, pose, operation)
        self._owned.pop(arm, None)
        self._event("motion_complete", arm=arm, operation_id=operation)

    def _fresh_scan(self):
        if self.observation is None or time.monotonic()-self.observation.timestamp_monotonic > self.max_scan_age_s:
            raise ValueError("Bucket observation missing or stale; scan again")

    async def _stationary(self):
        state = await self.adapter.snapshot()
        if state.dropper_moving or state.board_moving or time.monotonic()-state.timestamp_monotonic > 2:
            raise ValueError("Both arms must have fresh stationary readbacks")
        return state

    async def _board_unchanged(self):
        state = await self._stationary()
        a, b = state.board_joints_deg, self.board.joints_deg
        if len(a) != len(b) or not a or any(abs(x-y) > .5 for x,y in zip(a,b)):
            raise ValueError("Board arm changed; invalidate scene and reconfirm board")
        return state

    async def _perform(self, action, payload):
        self._event("action", action=action)
        if action == "position_board":
            pose = Pose("world", vector(payload["xyz_mm"]), vector(payload["orientation_vector_deg"], 4), SYNTHETIC)
            await self._move("arm-board", pose)
            self._revision += 1
        elif action == "confirm_board":
            self.board = await self.adapter.board_from_kinematics(await self._stationary())
            if not self.board.evidence.verified or not self.board.mount_calibration_id or self.board.frame != "world":
                raise ValueError("Board kinematic/mount evidence unverified")
            self._revision += 1
            self.phase = "place_bucket"
        elif action == "confirm_bucket_placed":
            self._revision += 1
            self.phase = "ready_scan"
        elif action == "scan_bucket":
            await self._board_unchanged()
            self.phase = "scanning"
            await self._move("arm", self.adapter.scan_pose)
            await self._board_unchanged()
            if not await self.adapter.at_pose_and_settled(self.adapter.scan_pose):
                raise ValueError("Dropper not settled at scan pose")
            observation = await self.adapter.capture_rgbd()
            for state in (observation.state_before, observation.state_after):
                if state.dropper_moving or state.board_moving:
                    raise ValueError("Camera observation was not stationary")
            before, after = observation.state_before, observation.state_after
            if (any(abs(a-b) > .1 for a,b in zip(before.dropper_joints_deg, after.dropper_joints_deg))
                    or any(abs(a-b) > .1 for a,b in zip(before.board_joints_deg, after.board_joints_deg))
                    or any(abs(a-b) > .5 for a,b in zip(after.board_joints_deg, self.board.joints_deg))):
                raise ValueError("Joint motion or changed board during RGB-D observation")
            if not (before.timestamp_monotonic <= observation.timestamp_monotonic <= after.timestamp_monotonic):
                raise ValueError("Observation is not bracketed by joint readbacks")
            detection = await self.detector.detect(observation)
            if detection is None:
                raise ValueError("Bucket estimate unknown; detector needs physical commissioning")
            transform = await self.adapter.camera_world_transform(observation)
            self.bucket = camera_to_world(detection, observation, transform)
            self.observation = observation
            self.phase = "review_bucket"
        elif action == "review_bucket":
            if payload.get("confirmed") is not True:
                raise ValueError("Explicit bucket observation review confirmation required")
            self._fresh_scan()
            await self._board_unchanged()
            self.plan = await self.adapter.plan_release(self.board, self.bucket, self._revision)
            self._validate_plan()
            self.phase = "ready_loading"
        elif action == "go_loading":
            self._fresh_scan()
            await self._board_unchanged()
            await self._move("arm", self.adapter.loading_pose)
            if not await self.adapter.at_pose_and_settled(self.adapter.loading_pose):
                raise ValueError("Dropper not settled at loading pose")
            # Preparing an empty gripper is operator-authorized by this action.
            # No vacuum operation exists anywhere in the execution port.
            await self.adapter.open_gripper()
            self._event("loading_gripper_open")
            self.phase = "insert_ball"
        elif action == "confirm_loaded":
            if payload.get("confirmed") is not True:
                raise ValueError("Operator must confirm ball inserted and hands clear")
            self._fresh_scan()
            await self._board_unchanged()
            self.phase = "ready_drop"
        elif action == "grip_drop":
            self._fresh_scan()
            self._validate_plan()
            await self._board_unchanged()
            if not await self.adapter.at_pose_and_settled(self.adapter.loading_pose):
                raise ValueError("Dropper left the verified loading pose")
            self.trial = {"trial_id": str(uuid4()), "state": "recording", "outcome": "unknown",
                          "plan_id": self.plan.plan_id, "revision": self._revision,
                          "bucket_observation_id": self.bucket.observation_id,
                          "requested_release_pose": serializable(self.plan.release_pose),
                          "measured_release_pose": None, "provenance": "synthetic rehearsal"}
            ready = await self.adapter.start_recording(self.trial["trial_id"], dict(self.trial))
            if ready is not True:
                raise ValueError("Recording is not ready")
            self._event("recording_ready")
            self.phase = "gripping"
            if await self.adapter.close_gripper() is not True:
                raise ValueError("Held ball not confirmed; refusing release motion")
            self._event("gripper_closed")
            await self._board_unchanged()
            self.phase = "moving_to_release"
            await self._move("arm", self.plan.release_pose)
            await self._board_unchanged()
            if not await self.adapter.at_pose_and_settled(self.plan.release_pose):
                raise ValueError("Dropper not settled at intended release pose")
            self._fresh_scan()
            # Mock adapter supplies synthetic pose evidence, never a physical readback.
            if await self.adapter.recording_status() is not True:
                raise ValueError("Recording lost readiness before release")
            measured = await self.adapter.measured_release_pose()
            if measured is None or not measured.evidence.verified or measured.frame != "world":
                raise ValueError("Measured release pose missing or unverified")
            if any(abs(a-b) > 1. for a,b in zip(measured.xyz_mm, self.plan.release_pose.xyz_mm)):
                raise ValueError("Measured release position mismatches intended pose")
            self.trial["measured_release_pose"] = serializable(measured)
            self._event("release_commanded", measured_release_pose=self.trial["measured_release_pose"])
            await self.adapter.open_gripper()
            self.trial["state"] = "released"
            self._event("release_complete")
            await self._finish_recording()
            self.phase = "complete"  # Outcome remains unknown without capture evidence.

    def _validate_plan(self):
        p = self.plan
        if (p is None or p.revision != self._revision or p.bucket_observation_id != self.bucket.observation_id
                or p.board_joints_deg != self.board.joints_deg
                or not all((p.collision_checked, p.limits_checked, p.trajectory_checked, p.evidence.verified))):
            raise ValueError("Release plan missing, incompatible, or not validated")

    async def _finish_recording(self):
        try:
            await asyncio.wait_for(self.adapter.finish_recording(self.trial["trial_id"]), 3.)
        except Exception:
            self._event("recording_finish_failed")

    async def _stop_owned(self, motions=None):
        motions = dict(self._owned) if motions is None else motions
        failed = {}
        for arm, operation in motions.items():
            try:
                await asyncio.wait_for(self.adapter.stop_motion(arm, operation), 3.)
                self._event("owned_motion_stopped", arm=arm, operation_id=operation)
            except Exception:
                failed[arm] = operation
                self._event("owned_motion_stop_failed", arm=arm, operation_id=operation)
        self._owned = failed
        if failed:
            self.error = "Owned motion stop unconfirmed; retry Stop and retain ownership"

    async def stop(self):
        async with self._gate:
            if self._stopping:
                return self.status()
            self._stopping = True
            task, motions = self._task, dict(self._owned)
            if task is not None and not task.done():
                task.cancel()
        try:
            if task is not None and task is not asyncio.current_task():
                try:
                    await asyncio.wait_for(asyncio.shield(task), 2.)
                except (asyncio.CancelledError, asyncio.TimeoutError):
                    pass
            await self._stop_owned(motions)
            if self.trial:
                if self.trial["state"] not in ("released", "failed"):
                    self.trial["state"] = "cancelled"
                await self._finish_recording()
            self._revision += 1
            self.plan = None
            self.phase = "stop_failed" if self._owned else ("cancelled" if self.mode == "dry_run" else "blocked")
            self._event("stop", tools_changed=False)
        finally:
            self._stopping = False
        return self.status()

    async def close(self):
        await self.stop()
