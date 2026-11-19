"""Recorder adversarial checks: fake RPCs and temporary files, never hardware.

Freshness is deliberately measured against real monotonic time so a slow RPC
or writer cannot manufacture a fresh release gate by resetting a timestamp.
"""
import asyncio
import json
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

from viam_adapter import ViamAdapter, utc


class RecorderOnlyAdapter(ViamAdapter):
    def __init__(self, collection_delay=0, write_delay=0, missing_arm=False):
        super().__init__("/never-read", {"upload_to_viam": False})
        self.collection_delay = collection_delay
        self.write_delay = write_delay
        self.missing_arm = missing_arm
        self.image_calls = 0

    async def connect(self):
        return {"connected": True, "mock": True}

    async def _images(self):
        self.image_calls += 1
        if self.image_calls > 1:
            # Hold the second collection so the first saved sample can be
            # inspected without a subsequent sample masking its actual age.
            await asyncio.Event().wait()
        requested = utc()
        await asyncio.sleep(self.collection_delay)
        return [{"name": "color", "mime_type": "image/jpeg", "bytes": b"fake-jpeg",
                 "width": 2, "height": 2}], {"requested_at": requested, "received_at": utc()}

    async def snapshot(self):
        names = ("arm",) if self.missing_arm else ("arm", "arm-board")
        return {"arms": {name: {"joints": [0.] * 6, "is_moving": False,
                    "pose": dict(x=0., y=0., z=200., o_x=0., o_y=0., o_z=-1., theta=0.)}
                    for name in names}, "request_started_at": utc(), "captured_at": utc()}

    def _persist_observation(self, *args):
        time.sleep(self.write_delay)
        return super()._persist_observation(*args)


class RecorderAdversarialTests(unittest.IsolatedAsyncioTestCase):
    async def sample(self, adapter, directory):
        await adapter.start_recording("adversarial-trial", directory)
        deadline = time.monotonic() + 6
        while time.monotonic() < deadline:
            status = await adapter.recording_status()
            if status["frame_count"] or status["error"]:
                return status
            await asyncio.sleep(.01)
        self.fail("Mock recording did not produce a result")

    async def cleanup_adapter(self, adapter):
        if adapter._record_task:
            adapter._record_task.cancel()
            await asyncio.wait_for(asyncio.gather(adapter._record_task, return_exceptions=True), 2)

    async def check_delay(self, **kwargs):
        adapter = RecorderOnlyAdapter(**kwargs)
        with tempfile.TemporaryDirectory() as directory:
            try:
                status = await self.sample(adapter, directory)
                self.assertTrue(status["active"])
                self.assertEqual(status["frame_count"], 1)
                self.assertIsNone(status["error"])
                self.assertGreaterEqual(status["last_frame_age_s"], 3.)
                self.assertFalse(status["ready"], "Old observations must never authorize release")
                record = json.loads(Path(directory, "observations.jsonl").read_text().splitlines()[0])
                self.assertEqual(status["last_frame_monotonic"], record["camera"]["observation_request_started_monotonic"])
                self.assertLessEqual(record["camera"]["observation_request_started_monotonic"],
                                     record["camera"]["observation_rpc_completed_monotonic"])
                self.assertIn("not sensor exposure time", status["freshness_basis"])
                self.assertIn("last_saved_at", status)
            finally:
                await self.cleanup_adapter(adapter)

    async def test_initial_disk_failure_stops_owned_native_capture(self):
        adapter = RecorderOnlyAdapter()
        sensor = FakeCaptureControl()
        adapter.capture_control = sensor
        with tempfile.TemporaryDirectory() as directory:
            with patch("viam_adapter.write_json", side_effect=OSError("mock disk full")):
                with self.assertRaises(OSError):
                    await adapter.start_recording("failed-start", directory)
            self.assertEqual(sensor.calls, ["start", "stop"])
            self.assertFalse(adapter._native_capture_owned)
            self.assertFalse((await adapter.recording_status())["active"])
            await adapter.finish_recording()
            self.assertEqual(sensor.calls, ["start", "stop"])

    async def test_failed_native_cleanup_keeps_ownership_for_retry(self):
        adapter = RecorderOnlyAdapter()
        sensor = FakeCaptureControl(stop_failures=1)
        adapter.capture_control = sensor
        with tempfile.TemporaryDirectory() as directory:
            with patch("viam_adapter.write_json", side_effect=OSError("mock disk full")):
                with self.assertRaises(OSError):
                    await adapter.start_recording("failed-start", directory)
            self.assertTrue(adapter._native_capture_owned)
            self.assertIn("stop_error", (await adapter.recording_status())["native_capture"])
            with self.assertRaisesRegex(RuntimeError, "Previous native recording"):
                await adapter.start_recording("next-trial", Path(directory, "next"))
            await adapter.finish_recording()
            self.assertFalse(adapter._native_capture_owned)
            self.assertEqual(sensor.calls, ["start", "stop", "stop"])

    async def test_canceled_native_setup_cleans_capture_before_worker_exists(self):
        adapter = RecorderOnlyAdapter()
        sensor = FakeCaptureControl(block_confirmation=True)
        adapter.capture_control = sensor
        with tempfile.TemporaryDirectory() as directory:
            task = asyncio.create_task(adapter.start_recording("canceled-start", directory))
            await asyncio.wait_for(sensor.confirmation_started.wait(), 1)
            self.assertIsNone(adapter._record_task)
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
            self.assertEqual(sensor.calls, ["start", "stop"])
            self.assertFalse(adapter._native_capture_owned)

    async def test_existing_foreign_native_capture_is_never_stopped(self):
        adapter = RecorderOnlyAdapter()
        sensor = FakeCaptureControl(foreign_active=True)
        adapter.capture_control = sensor
        with tempfile.TemporaryDirectory() as directory:
            try:
                status = await self.sample(adapter, directory)
                self.assertTrue(status["ready"])
                self.assertIn("refusing", status["native_capture"]["error"])
            finally:
                await self.cleanup_adapter(adapter)
            self.assertEqual(sensor.calls, [])

    async def test_unacknowledged_native_start_is_latched_unknown_without_foreign_stop(self):
        adapter = RecorderOnlyAdapter()
        sensor = FakeCaptureControl(unacknowledged_start=True)
        adapter.capture_control = sensor
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(RuntimeError, "not acknowledged"):
                await adapter.start_recording("unknown-start", directory)
            self.assertTrue(adapter._native_capture_start_unknown)
            self.assertFalse(adapter._native_capture_owned)
            self.assertEqual(sensor.calls, ["start"])
            saved = json.loads(Path(directory, "recording.json").read_text())
            self.assertEqual(saved["native_capture"]["state"], "start_unconfirmed")
            self.assertEqual(saved["native_capture"]["ownership"], "unknown")
            self.assertIsNone(saved["native_capture"]["enabled"])
            self.assertFalse(saved["active"])
            with self.assertRaisesRegex(RuntimeError, "operator inspection"):
                await adapter.start_recording("another-trial", Path(directory, "another"))
            with self.assertRaisesRegex(RuntimeError, "operator inspection"):
                await adapter.finish_recording()
            self.assertEqual(sensor.calls, ["start"])

    async def test_canceled_unacknowledged_start_is_not_claimed_cleaned_up(self):
        adapter = RecorderOnlyAdapter()
        sensor = FakeCaptureControl(block_start_ack=True)
        adapter.capture_control = sensor
        with tempfile.TemporaryDirectory() as directory:
            task = asyncio.create_task(adapter.start_recording("unknown-canceled-start", directory))
            await asyncio.wait_for(sensor.start_applied.wait(), 1)
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
            self.assertTrue(adapter._native_capture_start_unknown)
            self.assertEqual(sensor.calls, ["start"])
            self.assertEqual((await adapter.recording_status())["native_capture"]["state"], "start_unconfirmed")

    async def test_canceled_writer_cannot_put_old_observations_in_new_trial(self):
        started, release, done = threading.Event(), threading.Event(), threading.Event()
        class BlockedWriter(RecorderOnlyAdapter):
            def _persist_observation(self, *args):
                started.set()
                if not release.wait(4):
                    raise RuntimeError("Mock writer gate timed out")
                try:
                    return super()._persist_observation(*args)
                finally:
                    done.set()
        adapter = BlockedWriter()
        with tempfile.TemporaryDirectory() as directory:
            old, new = Path(directory, "old"), Path(directory, "new")
            try:
                await adapter.start_recording("old-trial", old)
                self.assertTrue(await asyncio.to_thread(started.wait, 1))
                await self.cleanup_adapter(adapter)
                # The Python worker still owns the old write, while the new
                # recorder is allowed to prepare its independent destination.
                await adapter.start_recording("new-trial", new)
                release.set()
                self.assertTrue(await asyncio.to_thread(done.wait, 1))
                self.assertTrue((old / "observations.jsonl").exists())
                self.assertFalse((new / "observations.jsonl").exists())
                self.assertFalse((await adapter.recording_status())["ready"])
            finally:
                release.set()
                await asyncio.to_thread(done.wait, 1)
                await self.cleanup_adapter(adapter)

    async def test_slow_collection_cannot_refresh_old_sample(self):
        await self.check_delay(collection_delay=3.05)

    async def test_slow_persistence_cannot_refresh_old_sample(self):
        await self.check_delay(write_delay=3.05)

    async def test_fresh_saved_color_and_both_arms_are_ready(self):
        adapter = RecorderOnlyAdapter()
        with tempfile.TemporaryDirectory() as directory:
            try:
                status = await self.sample(adapter, directory)
                self.assertTrue(status["ready"])
                self.assertLess(status["last_frame_age_s"], 3.)
                record = json.loads(Path(directory, "observations.jsonl").read_text().splitlines()[0])
                self.assertEqual(set(record["state"]["arms"]), {"arm", "arm-board"})
                self.assertTrue(Path(directory, record["images"][0]["file"]).is_file())
            finally:
                await self.cleanup_adapter(adapter)

    async def test_missing_board_state_never_establishes_recording_readiness(self):
        adapter = RecorderOnlyAdapter(missing_arm=True)
        with tempfile.TemporaryDirectory() as directory:
            try:
                status = await self.sample(adapter, directory)
                self.assertFalse(status["ready"])
                self.assertEqual(status["frame_count"], 0)
                self.assertIn("both arms", status["error"])
            finally:
                await self.cleanup_adapter(adapter)


class FakeCaptureControl:
    def __init__(self, stop_failures=0, block_confirmation=False, foreign_active=False,
                 unacknowledged_start=False, block_start_ack=False):
        self.calls = []
        self.stop_failures = stop_failures
        self.block_confirmation = block_confirmation
        self.foreign_active = foreign_active
        self.confirmation_started = asyncio.Event()
        self.start_applied = asyncio.Event()
        self.unacknowledged_start = unacknowledged_start
        self.block_start_ack = block_start_ack

    async def get_readings(self, **kwargs):
        if self.calls and self.block_confirmation:
            self.confirmation_started.set()
            await asyncio.Event().wait()
        return {"sequences": ["some-other-client"]} if self.foreign_active else {}

    async def do_command(self, command, **kwargs):
        if command.get("start_capture"):
            self.calls.append("start")
            self.start_applied.set()
            if self.unacknowledged_start:
                raise TimeoutError("mock ACK lost after server applied start")
            if self.block_start_ack:
                await asyncio.Event().wait()
        elif command.get("stop_capture"):
            self.calls.append("stop")
            if self.stop_failures:
                self.stop_failures -= 1
                raise RuntimeError("mock stop unavailable")
        return {}


if __name__ == "__main__":
    unittest.main()
