"""Narrow sfhack10 adapter. Importing/connecting never moves an actuator.

All commands are called by the operator controller; this adapter has no startup
motions, board-arm command handle, vacuum handle, or arbitrary DoCommand endpoint.
Viam credentials remain in a separate owner-only file, never in browser state.
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
import time
from uuid import uuid4

ADDRESS = "sfhack10-main.ty1cywh7tq.viam.cloud"
MACHINE_ID = "3103375b-9bab-4222-825d-b3fb9b2897fa"
PART_ID = "29154d15-3b8c-4097-a3c6-2bc9d04a6e12"
POSE_FIELDS = ("x", "y", "z", "o_x", "o_y", "o_z", "theta")


def utc():
    return datetime.now(timezone.utc).isoformat()


def proto(value):
    from google.protobuf.json_format import MessageToDict
    return MessageToDict(value, preserving_proto_field_name=True)


def write_json(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w") as out:
        json.dump(value, out, indent=2, allow_nan=False)
        out.write("\n")
        out.flush()
        os.fsync(out.fileno())
    temp.replace(path)


class ViamAdapter:
    def __init__(self, credentials_path, config=None):
        self.credentials_path = Path(credentials_path)
        self.config = dict(config or {})
        self.transport = self.config.get("transport", "auto")
        if self.transport not in ("auto", "direct"):
            raise ValueError("Physical transport must be auto or direct")
        self.robot = None
        self.motion = None
        self.gripper = None
        self.camera = None
        self.arms = {}
        self.capture_control = None
        self.capabilities = {}
        self._credential = None
        self._connect_lock = asyncio.Lock()
        self._camera_lock = asyncio.Lock()
        self._stop_lock = asyncio.Lock()
        self._actuation_epoch = 0
        self._rpc_tasks = set()
        self._actuation_tasks = set()
        self._move_task = None
        self._move_token = None
        self._move_active = False
        self._move_operations = set()
        self._move_monitor = None
        self._record_task = None
        self._record_cleanup_lock = asyncio.Lock()
        self._record_stop = asyncio.Event()
        self._record = {"active": False, "ready": False, "frame_count": 0, "error": None}
        self._last_frame_mono = None
        self._last_rgb = None
        self._last_rgb_meta = None
        self._trial_dir = None
        self._native_capture_owned = False
        self._native_capture_start_unknown = False
        self.rpc_timeout = 5.0

    def _credentials(self):
        p = self.credentials_path
        info = p.stat()
        if p.is_symlink() or info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) != 0o600:
            raise RuntimeError("sfhack10 credentials must be owner-only mode0600 and not a symlink")
        c = json.loads(p.read_text())
        if c.get("address") != ADDRESS or not c.get("api_key_id") or not c.get("api_key"):
            raise RuntimeError("Physical adapter credentials must target the pinned sfhack10 address")
        self._credential = c
        return c

    def _safe_error(self, exc):
        message = str(exc)
        for key in ("api_key", "api_key_id"):
            secret = (self._credential or {}).get(key)
            if secret:
                message = message.replace(secret, "[REDACTED]")
        return f"{type(exc).__name__}: {message[:700]}"

    async def _rpc(self, awaitable, timeout=None, *, actuation=False):
        # asyncio.wait_for waits for cancellation acknowledgement and can exceed
        # its deadline. Late RPC results are owned/observed but never returned.
        task=asyncio.ensure_future(awaitable);self._rpc_tasks.add(task)
        if actuation:self._actuation_tasks.add(task)
        def finished(value):
            self._rpc_tasks.discard(value);self._actuation_tasks.discard(value)
            if not value.cancelled():value.exception()
        task.add_done_callback(finished)
        try:
            done,_=await asyncio.wait({task},timeout=self.rpc_timeout+1 if timeout is None else timeout)
            if not done:task.cancel();raise asyncio.TimeoutError('RPC deadline expired')
            return task.result()
        except asyncio.CancelledError:
            task.cancel();raise
        except Exception as exc:
            raise RuntimeError(self._safe_error(exc)) from None

    def _actuation_current(self,epoch):
        task=asyncio.current_task()
        if epoch!=self._actuation_epoch or (task is not None and task.cancelling()):
            raise asyncio.CancelledError()
        if self._stop_lock.locked():raise RuntimeError('A Stop is pending; actuator request rejected')
        if any(value is not task and not value.done() for value in self._actuation_tasks):
            raise RuntimeError('An owned actuator request is still pending; verify Stop before another command')

    async def connect(self):
        async with self._connect_lock:
            if self.robot is not None:
                return dict(self.capabilities)
            from viam.components.arm import Arm
            from viam.components.camera import Camera
            from viam.components.gripper import Gripper
            from viam.components.sensor import Sensor
            from viam.robot.client import RobotClient
            from viam.rpc.dial import Credentials, DialOptions
            from viam.services.motion import MotionClient
            c = self._credentials()
            options = RobotClient.Options(dial_options=DialOptions(
                # A cloud-address robot behind the hackathon network needs
                # Viam's negotiated WebRTC/TLS transport. Direct-only dial was
                # copied from the loopback simulator and timed out on sfhack10.
                disable_webrtc=self.transport == "direct", auth_entity=c["api_key_id"],
                credentials=Credentials(type="api-key", payload=c["api_key"])))
            robot = await self._rpc(RobotClient.at_address(ADDRESS, options), 20)
            resources = {(x.type, x.subtype, x.name) for x in robot.resource_names}
            required = {("component", "arm", "arm"), ("component", "arm", "arm-board"),
                        ("component", "gripper", "gripper"), ("component", "camera", "cam"),
                        ("service", "motion", "builtin")}
            if not required.issubset(resources):
                await self._rpc(robot.close())
                raise RuntimeError("sfhack10 is missing an expected arm, camera, gripper, or motion resource")
            self.robot = robot
            self.arms = {name: Arm.from_robot(robot, name) for name in ("arm", "arm-board")}
            self.camera = Camera.from_robot(robot, "cam")
            self.gripper = Gripper.from_robot(robot, "gripper")
            self.motion = MotionClient.from_robot(robot, "builtin")
            capture_name = self.config.get("capture_control_name")
            data_manager_names = sorted(n for t, s, n in resources if t == "service" and s == "data_manager")
            if capture_name and ("component", "sensor", capture_name) in resources and data_manager_names:
                self.capture_control = Sensor.from_robot(robot, capture_name)
            self.capabilities = {
                "connected": True, "machine": "sfhack10", "machine_id": MACHINE_ID, "part_id": PART_ID,
                "transport": "SDK default; WebRTC enabled" if self.transport == "auto" else "direct TLS only",
                "pose_frame": "world", "pose_component": "arm", "pose_source": "builtin Motion.GetPose",
                "camera": "cam", "arms": ["arm", "arm-board"], "commanded_arm": "arm",
                "commanded_gripper": "gripper", "board_commands": False,
                "resources": [{"type": t, "subtype": s, "name": n} for t, s, n in sorted(resources)],
                "data_manager_names": data_manager_names,
                "capture_control_name": capture_name if self.capture_control else None,
                "native_capture_available": bool(self.capture_control),
                "native_capture_verified": False,
                "local_recording": True,
                "cloud_upload": "Viam DataClient on Save; availability not assumed until upload succeeds",
                "sequence": "CreateSequence after timestamped image/joint uploads; native capture-control optional",
            }
            return dict(self.capabilities)

    async def snapshot(self):
        await self.connect()
        started = utc()

        async def one(name):
            arm = self.arms[name]
            joints, moving, pose = await asyncio.gather(
                self._rpc(arm.get_joint_positions(timeout=self.rpc_timeout)),
                self._rpc(arm.is_moving(timeout=self.rpc_timeout)),
                self._rpc(self.motion.get_pose(name, "world", timeout=self.rpc_timeout)))
            if pose.reference_frame != "world":
                raise RuntimeError("Expected world-frame readback from Viam motion service")
            return name, {"pose": {k: float(getattr(pose.pose, k)) for k in POSE_FIELDS},
                          "joints": list(joints.values), "is_moving": moving,
                          "pose_frame": "world", "pose_component": name,
                          "provenance": "kinematic readback from configured Viam frame system; not camera calibration"}

        arms = dict(await asyncio.gather(one("arm"), one("arm-board")))
        return {"arms": arms, "captured_at": utc(), "request_started_at": started,
                "camera": "cam", "synchronization": "Separate RPCs; not hardware-synchronized"}

    async def _images(self):
        await self.connect()
        async with self._camera_lock:
            requested = utc()
            images, metadata = await self._rpc(self.camera.get_images(timeout=self.rpc_timeout))
            received = utc()
            data = []
            for image in images:
                mime = getattr(image.mime_type, "value", str(image.mime_type))
                data.append({"name": image.name, "mime_type": mime, "bytes": bytes(image.data),
                             "width": image.width, "height": image.height})
            return data, {"requested_at": requested, "received_at": received,
                          "response_metadata": proto(metadata)}

    async def camera_frame(self):
        if self._last_rgb is not None and self._last_frame_mono is not None and time.monotonic() - self._last_frame_mono < 2:
            return self._last_rgb[0], self._last_rgb[1], dict(self._last_rgb_meta or {})
        images, meta = await self._images()
        rgb = next((x for x in images if x["mime_type"] in ("image/jpeg", "image/png")), None)
        if not rgb:
            raise RuntimeError("Camera did not return a JPEG/PNG color image")
        return rgb["bytes"], rgb["mime_type"], meta

    async def open_gripper(self):
        epoch=self._actuation_epoch
        await self.connect();self._actuation_current(epoch)
        async def command():
            self._actuation_current(epoch)
            return await self.gripper.open(timeout=self.rpc_timeout)
        await self._rpc(command(),actuation=True)
        return {"command":"gripper.open","completed_at":utc()}

    async def close_gripper(self):
        epoch=self._actuation_epoch
        await self.connect();self._actuation_current(epoch)
        async def command():
            self._actuation_current(epoch)
            return await self.gripper.grab(timeout=self.rpc_timeout)
        grabbed=await self._rpc(command(),actuation=True)
        return {"command":"gripper.grab","completed_at":utc(),"grabbed":grabbed is True,
                "note":"Driver report; not visual verification of a held ball"}

    async def _discover_move_operations(self):
        if not self._move_token:
            return
        operations = await self._rpc(self.robot.get_operations())
        for op in operations:
            args = proto(op.arguments)
            # The unique extra marker identifies this panel's Move. Never cancel
            # every motion operation, or another client's board-arm operation.
            extra = args.get("extra", {})
            component = args.get("component_name", args.get("componentName"))
            if ("MotionService/Move" in op.method and component == "arm"
                    and extra.get("sircus_operator_operation") == self._move_token):
                self._move_operations.add(op.id)

    async def _monitor_operations(self):
        while self._move_token:
            try:
                await self._discover_move_operations()
            except Exception:
                pass
            await asyncio.sleep(0.1)

    async def move_pose(self, pose):
        epoch=self._actuation_epoch
        await self.connect();self._actuation_current(epoch)
        if self._move_task and not self._move_task.done():
            raise RuntimeError("A physical move is already running")
        if (not isinstance(pose,dict) or set(pose)!=set(POSE_FIELDS) or
                any(type(pose[k]) not in (int,float) or not math.isfinite(pose[k]) for k in POSE_FIELDS)):
            raise ValueError("A finite full world-frame arm pose is required")
        from viam.proto.common import Pose, PoseInFrame
        self._move_token = uuid4().hex
        self._move_active = True
        self._move_operations = set()
        self._move_task = asyncio.current_task()
        self._move_monitor = asyncio.create_task(self._monitor_operations())
        try:
            async def command():
                self._actuation_current(epoch)
                return await self.motion.move(
                    component_name="arm",destination=PoseInFrame(reference_frame="world",pose=Pose(**pose)),
                    extra={"sircus_operator_operation":self._move_token},timeout=45)
            ok=await self._rpc(command(),47,actuation=True)
            if not ok:
                raise RuntimeError("Viam motion service did not complete the planned move")
            self._move_active = False
            self._move_token = None
            self._move_operations.clear()
            return {"command": "builtin.Move", "component": "arm", "completed_at": utc()}
        except asyncio.CancelledError:
            # The controller concurrently calls stop(). Retain token/operation
            # evidence until that explicit stop cancels the remote operation.
            raise
        except Exception:
            await self.stop()
            raise
        finally:
            if self._move_monitor:
                self._move_monitor.cancel()
            self._move_task = None

    async def stop(self):
        self._actuation_epoch += 1
        await self.connect()
        async with self._stop_lock:
            # Do not delay the immediate actuator stop behind operation-list RPCs.
            first_stop = asyncio.create_task(self._rpc(self.arms["arm"].stop(timeout=self.rpc_timeout)))
            first_stop.add_done_callback(lambda task: None if task.cancelled() else task.exception())
            for owned in list(self._actuation_tasks):
                if not owned.done():owned.cancel()
            was_active = self._move_active
            discovered = False
            try:
                await self._discover_move_operations()
                discovered = True
            except Exception:
                pass
            errors = []
            cancelled = []
            # First cancel the known planner RPC so it cannot send another
            # trajectory segment after stopping the physical dropper arm.
            for operation_id in sorted(self._move_operations):
                try:
                    await self._rpc(self.robot.cancel_operation(operation_id))
                    cancelled.append(operation_id)
                except Exception as exc:
                    errors.append(self._safe_error(exc))
            task = self._move_task
            if task and task is not asyncio.current_task() and not task.done():
                task.cancel()
            try:await first_stop
            except Exception as exc:errors.append(self._safe_error(exc))
            # Reassert after cancelling the planner, which may have been sending
            # a segment concurrently with the first Stop call.
            try:await self._rpc(self.arms["arm"].stop(timeout=self.rpc_timeout))
            except Exception as exc:errors.append(self._safe_error(exc))
            remaining=[]
            try:
                operations=await self._rpc(self.robot.get_operations())
                for operation in operations:
                    args=proto(operation.arguments);extra=args.get('extra',{})
                    if operation.id in cancelled or (self._move_token and extra.get('sircus_operator_operation')==self._move_token):
                        remaining.append(operation.id)
            except Exception as exc:
                if was_active:errors.append(self._safe_error(exc))
            pending_actuation=sum(not task.done() for task in self._actuation_tasks)
            verified = not errors and not remaining and not pending_actuation and (not was_active or bool(cancelled))
            # Do not erase an unverified planner operation: another Stop must
            # not turn missing cancellation evidence into a false success.
            if verified:
                self._move_token = None
                self._move_active = False
                self._move_operations.clear()
            if self._move_monitor:
                self._move_monitor.cancel()
            return {"command": "arm.stop", "completed_at": utc(), "cancelled_operations": cancelled,
                    "operation_discovery_ok": discovered, "cancel_errors": errors, "remaining_owned_operations":remaining,
                    "planner_cancellation_verified": verified,"pending_owned_actuator_requests":pending_actuation,
                    "gripper_or_vacuum_commanded": False}

    async def start_recording(self, trial_id, trial_dir, metadata=None):
        await self.connect()
        if self._record_task and not self._record_task.done():
            raise RuntimeError("A trial recording is already active")
        if self._native_capture_owned:
            raise RuntimeError("Previous native recording stop is not confirmed; finish it before another trial")
        if self._native_capture_start_unknown:
            raise RuntimeError("Previous native capture start is unconfirmed; operator inspection is required")
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,100}", trial_id):
            raise ValueError("Invalid trial ID")
        self._trial_dir = Path(trial_dir)
        self._trial_dir.mkdir(parents=True, exist_ok=True)
        if (self._trial_dir / "observations.jsonl").exists():
            raise RuntimeError("A recording with observations already exists in this trial folder")
        self._record_stop = asyncio.Event()
        self._last_frame_mono = None
        self._record = {"active": True, "ready": False, "frame_count": 0, "error": None,
                        "trial_id": trial_id, "started_at": utc(), "backend": "local Mac RGB-D and state",
                        "freshness_basis": "Age since the earliest local image/state request began, including collection and persistence delay; not sensor exposure time",
                        "native_capture": {"enabled": False, "verified": False},
                        "synchronization_limits": [
                            "Camera GetImages and each arm readback are separate RPCs.",
                            "Local request/response windows bracket samples; exposure/joint clocks are not synchronized.",
                            "Raw depth is saved in device counts; no metric scale or camera-to-world calibration asserted."]}
        self._native_capture_owned = False
        try:
            if self.capture_control:
                try:
                    readings = await self._rpc(self.capture_control.get_readings(timeout=self.rpc_timeout))
                    if readings.get("sequences") or any(float(x.get("capture_frequency_hz", 0)) > 0 for x in readings.get("overrides", [])):
                        raise RuntimeError("capture-control is already recording; refusing to take over another recording")
                    self._native_capture_start_unknown = True
                    await self._rpc(self.capture_control.do_command(
                        {"start_capture": True, "frequency_hz": 2, "tags": ["sircus-physical", "trial:" + trial_id]},
                        timeout=self.rpc_timeout))
                    self._native_capture_owned = True
                    self._native_capture_start_unknown = False
                    readings = await self._rpc(self.capture_control.get_readings(timeout=self.rpc_timeout))
                    self._record["native_capture"] = {"enabled": True, "sensor_readings": dict(readings),
                                                       "verified": False,
                                                       "note": "Sensor acknowledged; data-manager files/cloud sequence not yet verified"}
                except Exception as exc:
                    self._record["native_capture"] = {"enabled": self._native_capture_owned, "verified": False,
                                                       "error": self._safe_error(exc)}
                    if self._native_capture_start_unknown:
                        raise RuntimeError("Native capture start was not acknowledged; operator inspection is required") from None
            await asyncio.to_thread(write_json, self._trial_dir / "recording.json", self._record)
            self._record_task = asyncio.create_task(self._record_loop())
        except BaseException as exc:
            # Native capture may already be active even though no local worker
            # exists yet. Preserve ownership on failed cleanup so Stop can retry.
            self._record.update(active=False, ready=False, finished_at=utc())
            self._record["error"] = self._safe_error(exc)
            if self._native_capture_start_unknown:
                self._record["native_capture"].update(
                    state="start_unconfirmed", enabled=None, ownership="unknown",
                    note="The start request may have reached the server; no ownership-safe stop is available. Operator inspection required.")
            try:
                await self._stop_owned_native_capture()
            except Exception:
                pass
            try:
                await asyncio.to_thread(write_json, self._trial_dir / "recording.json", self._record)
            except Exception:
                pass
            raise
        return await self.recording_status()

    async def _stop_owned_native_capture(self):
        async with self._record_cleanup_lock:
            if self._native_capture_start_unknown:
                raise RuntimeError("Native capture start is unconfirmed; operator inspection is required before ownership-safe cleanup")
            if not self._native_capture_owned:
                return
            try:
                await self._rpc(self.capture_control.do_command({"stop_capture": True}, timeout=self.rpc_timeout))
            except Exception as exc:
                self._record["native_capture"]["stop_error"] = self._safe_error(exc)
                raise
            self._native_capture_owned = False
            self._record["native_capture"].pop("stop_error", None)
            self._record["native_capture"]["stop_acknowledged"] = True

    def _persist_observation(self, images, meta, state, index, trial_dir):
        # Worker-thread I/O can finish after asyncio cancellation. Its original
        # destination must not follow a subsequent trial's mutable adapter state.
        trial_dir = Path(trial_dir)
        entries = []
        for image in images:
            name = re.sub(r"[^A-Za-z0-9_-]", "_", image["name"]) or "image"
            suffix = {"image/jpeg": ".jpg", "image/png": ".png", "image/vnd.viam.dep": ".dep.gz"}.get(image["mime_type"], ".bin")
            filename = f"frame_{index:04d}_{name}{suffix}"
            raw = image["bytes"]
            (trial_dir / filename).write_bytes(gzip.compress(raw, mtime=0) if suffix.endswith(".gz") else raw)
            entries.append({"name": image["name"], "mime_type": image["mime_type"], "file": filename,
                            "width": image["width"], "height": image["height"], "sha256_raw": hashlib.sha256(raw).hexdigest()})
        record = {"index": index, "camera": meta, "images": entries, "state": state}
        with (trial_dir / "observations.jsonl").open("a") as f:
            f.write(json.dumps(record, allow_nan=False) + "\n")
            f.flush()
            os.fsync(f.fileno())
        return record

    async def _record_loop(self):
        start = time.monotonic()
        try:
            while not self._record_stop.is_set() and time.monotonic() - start < 120:
                tick = time.monotonic()
                try:
                    (images, meta), state = await asyncio.gather(self._images(), self.snapshot())
                    meta = {**meta, "observation_request_started_monotonic": tick,
                            "observation_rpc_completed_monotonic": time.monotonic(),
                            "freshness_basis": "earliest local image/state request start; not sensor exposure time"}
                    rgb = next((x for x in images if x["mime_type"] in ("image/jpeg", "image/png")), None)
                    if rgb is None or len(state["arms"]) != 2:
                        raise RuntimeError("Recording requires a color frame and readbacks from both arms")
                    await asyncio.to_thread(self._persist_observation, images, meta, state, self._record["frame_count"], self._trial_dir)
                    # Persistence establishes availability, not sample recency.
                    # A delayed RPC or disk write must never make old evidence
                    # newly fresh enough to authorize a release.
                    self._last_frame_mono = tick
                    self._last_rgb = (rgb["bytes"], rgb["mime_type"])
                    self._last_rgb_meta = meta
                    self._record.update(frame_count=self._record["frame_count"] + 1,
                                        last_saved_at=utc(), error=None)
                except Exception as exc:
                    self._record["error"] = self._safe_error(exc)
                await asyncio.to_thread(write_json, self._trial_dir / "recording.json", self._record)
                remaining = max(0.01, 0.5 - (time.monotonic() - tick))
                try:
                    await asyncio.wait_for(self._record_stop.wait(), remaining)
                except asyncio.TimeoutError:
                    pass
            if not self._record_stop.is_set():
                self._record["error"] = "120-second recording limit reached; prepare a new trial"
        finally:
            self._record.update(active=False, ready=False, finished_at=utc())
            try:
                await self._stop_owned_native_capture()
            except Exception:
                pass  # Ownership/error retained; finish_recording can retry.
            await asyncio.to_thread(write_json, self._trial_dir / "recording.json", self._record)

    async def recording_status(self):
        age = None if self._last_frame_mono is None else time.monotonic() - self._last_frame_mono
        result = dict(self._record)
        result["last_frame_age_s"] = age
        result["last_frame_monotonic"] = self._last_frame_mono or 0
        result["ready"] = bool(result["active"] and result["frame_count"] > 0 and not result["error"] and age is not None and age <= 3)
        return result

    async def finish_recording(self):
        self._record_stop.set()
        if self._record_task and not self._record_task.done():
            await self._record_task
        await self._stop_owned_native_capture()
        return await self.recording_status()

    async def save_trial(self, trial_dir, metadata):
        """Upload explicit completed trial on operator Save; retain all local files.

        Uses Viam DataSync APIs directly because the physical data-manager runs
        remotely and cannot see this Mac's local path. Never points remote sync
        paths at the Mac. Individual successful uploads have checkpoint IDs.
        """
        folder = Path(trial_dir)
        await asyncio.to_thread(write_json, folder / "trial-metadata.json", metadata)
        if not self.config.get("upload_to_viam", True):
            return {"state": "local_only", "cloud_verified": False, "reason": "Viam upload disabled in physical config"}
        await self.connect()
        from viam.app.viam_client import ViamClient
        from viam.proto.app.data import SequenceResourceFilter
        from viam.rpc.dial import Credentials, DialOptions
        c = self._credentials()
        checkpoint_file = folder / "viam-upload.json"
        checkpoint = json.loads(checkpoint_file.read_text()) if checkpoint_file.exists() else {
            "state": "pending", "part_id": PART_ID, "uploads": {}, "cloud_verified": False}
        client = None
        try:
            client = await self._rpc(ViamClient.create_from_dial_options(DialOptions(
                auth_entity=c["api_key_id"], credentials=Credentials(type="api-key", payload=c["api_key"]))), 20)
            data = client.data_client
            trial_id = str(metadata.get("trial_id", folder.name))
            outcome = metadata.get("outcome")
            if isinstance(outcome, dict):
                outcome = outcome.get("outcome", outcome.get("result", "unsure"))
            tags = ["sircus-physical", "trial:" + trial_id, "outcome:" + str(outcome or "unsure")]
            records = [json.loads(x) for x in (folder / "observations.jsonl").read_text().splitlines() if x]
            for record in records:
                requested = datetime.fromisoformat(record["camera"]["requested_at"])
                received = datetime.fromisoformat(record["camera"]["received_at"])
                for entry in record["images"]:
                    if entry["mime_type"] not in ("image/jpeg", "image/png"):
                        continue
                    key = entry["file"]
                    if key in checkpoint["uploads"]:
                        continue
                    upload_id = await self._rpc(data.binary_data_capture_upload(
                        binary_data=(folder / key).read_bytes(), part_id=PART_ID,
                        component_type="rdk:component:camera", component_name="cam", method_name="GetImages",
                        file_extension=Path(key).suffix, tags=tags, data_request_times=(requested, received),
                        mime_type=entry["mime_type"]), 30)
                    checkpoint["uploads"][key] = upload_id
                    await asyncio.to_thread(write_json, checkpoint_file, checkpoint)
            for name in ("arm", "arm-board"):
                key = "joint-readings:" + name
                if key not in checkpoint["uploads"] and records:
                    upload_id = await self._rpc(data.tabular_data_capture_upload(
                        part_id=PART_ID, component_type="rdk:component:arm", component_name=name,
                        method_name="JointPositions", tags=tags,
                        tabular_data=[{"positions": {"values": r["state"]["arms"][name]["joints"]},
                                       "world_pose": r["state"]["arms"][name]["pose"], "trial_id": trial_id} for r in records],
                        data_request_times=[(datetime.fromisoformat(r["state"]["request_started_at"]),
                                             datetime.fromisoformat(r["state"]["captured_at"])) for r in records]), 30)
                    checkpoint["uploads"][key] = upload_id
                    await asyncio.to_thread(write_json, checkpoint_file, checkpoint)
            # Metadata outcome and depth are preserved in Files as well. JPEG/PNG
            # and tabular joints are the only entries used by Viam Sequences.
            for path in sorted(folder.iterdir()):
                if not path.is_file() or path.name == checkpoint_file.name or path.suffix == ".tmp":
                    continue
                key = "file:" + path.name
                if key in checkpoint["uploads"] or path.suffix in (".jpg", ".png"):
                    continue
                upload_id = await self._rpc(data.file_upload(part_id=PART_ID, data=path.read_bytes(),
                    component_type="sircus:physical-trial", component_name="sircus-physical-trial",
                    method_name="operator_trial", file_name=path.name, file_extension="".join(path.suffixes), tags=tags), 45)
                checkpoint["uploads"][key] = upload_id
                await asyncio.to_thread(write_json, checkpoint_file, checkpoint)
            if records and not checkpoint.get("sequence_id"):
                start = min(datetime.fromisoformat(r["camera"]["requested_at"]) for r in records)
                end = max(datetime.fromisoformat(r["state"]["captured_at"]) for r in records)
                checkpoint["sequence_id"] = await self._rpc(data.create_sequence(
                    part_id=PART_ID,
                    resources=[SequenceResourceFilter(resource_name="cam", method_name="GetImages"),
                               SequenceResourceFilter(resource_name="arm", method_name="JointPositions"),
                               SequenceResourceFilter(resource_name="arm-board", method_name="JointPositions")],
                    start_time=start, end_time=end, sequence_tags=tags, timeout=20), 22)
            checkpoint.update(state="uploaded", cloud_verified=False, upload_acknowledged=True,
                              note="API upload IDs acknowledged; review Viam DATA/SEQUENCES for independent cloud verification",
                              completed_at=utc())
        except Exception as exc:
            checkpoint.update(state="pending_retry", error=self._safe_error(exc), cloud_verified=False)
        finally:
            if client is not None:
                await client.close()
            await asyncio.to_thread(write_json, checkpoint_file, checkpoint)
        return checkpoint

    async def close(self):
        """Explicit shutdown only; normal Stop intentionally keeps session alive."""
        await self.finish_recording()
        if self.robot:
            await self._rpc(self.robot.close())
            self.robot = None
