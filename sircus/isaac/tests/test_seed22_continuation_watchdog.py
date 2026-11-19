"""Independent clock/counter/process tests; no GPU or native simulator work."""

import copy
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import seed22_continuation_watchdog as subject


class FakeOps:
    def __init__(self, root, session, fingerprint):
        self.root, self.session, self.fingerprint = root, session, fingerprint
        self.t, self.free = 10., subject.MIN_FREE_BYTES + 1000
        self.started, self.signalled, self.prepared = [], [], False
        self.running, self.exit_code, self.on_sleep, self.on_start = False, None, None, None
        self.records = {777: dict(pid=777, start_token="boot:77", cookie=session,
                                 ppid=1, sid=777, state="R")}

    def monotonic(self): return self.t
    def free_bytes(self, path): return self.free
    def own_pid(self): return 777
    def prepare(self): self.prepared = True
    def identity(self, pid): return copy.deepcopy(self.records.get(pid))

    def write(self, requested=0, **fields):
        value = subject.signed(dict(session_id=self.session,
            runtime_fingerprint=self.fingerprint, train_requested=requested, **fields))
        temporary = self.root / "out/.new-progress"
        temporary.write_text(json.dumps(value))
        temporary.replace(self.root / "out/progress.json")

    def start(self, command, environment):
        self.started.append((command, environment))
        self.running = True
        self.records[888] = dict(pid=888, start_token="boot:88", cookie=self.session,
                                ppid=777, sid=888, state="S")
        self.write()
        if self.on_start:
            self.on_start(self)
        return 888

    def poll(self, pid):
        if self.exit_code is not None:
            self.running = False
            self.records.pop(888, None)
        return self.exit_code

    def owned(self, pid, session):
        return [copy.deepcopy(self.records[888])] if self.running and 888 in self.records else []

    def signal_owned(self, record, signum, session):
        self.signalled.append((copy.deepcopy(record), signum, session))
        self.running = False
        self.records.pop(888, None)
        return "signalled"

    def sleep(self, seconds):
        self.t += seconds
        if self.on_sleep:
            self.on_sleep(self)


class WatchdogTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.session, self.fingerprint = "a" * 32, "b" * 64
        (self.root / "scene.lock").write_text("preserved shared lock metadata")
        self.ops = FakeOps(self.root, self.session, self.fingerprint)

    def run_watchdog(self, **overrides):
        args = dict(command=["/mock/python", "inert.py"], output=self.root / "out",
                    lock=self.root / "scene.lock", session_id=self.session,
                    runtime_fingerprint=self.fingerprint, ops=self.ops, wall_s=2)
        args.update(overrides)
        return subject.supervise(**args)

    def clean_finish(self, requested=20):
        def finish(ops):
            ops.write(requested, train_completed=requested, status="completed")
            ops.exit_code = 0
        self.ops.on_sleep = finish

    def test_one_start_and_bound_exit_receipts(self):
        self.clean_finish()
        result = self.run_watchdog()
        self.assertEqual(result["status"], "native_exit_zero_independent_audit_required")
        self.assertEqual(result["train_requested"], 20)
        self.assertEqual(result["train_completed"], 20)
        self.assertEqual(result["native_application_starts"], 1)
        self.assertEqual(len(self.ops.started), 1)
        self.assertFalse(self.ops.signalled)
        self.assertFalse(result["independent_audit_passed"])
        self.assertFalse(result["model_promoted"])
        self.assertTrue(result["lock_release_safe"])
        self.assertEqual(json.loads((self.root / "out/watchdog-exit.json").read_text()), result)
        environment = self.ops.started[0][1]
        self.assertEqual(environment["CUDA_VISIBLE_DEVICES"], subject.GPU_UUID)
        self.assertEqual(environment[subject.COOKIE], self.session)
        self.assertEqual(float(environment["SIRCUS_SEED22_DEADLINE_MONOTONIC"]), 12.)
        self.assertEqual(float(environment["SIRCUS_CONTINUATION_DEADLINE_MONOTONIC"]), 12.)
        self.assertEqual(float(environment["SIRCUS_CONTINUATION_STARTED_MONOTONIC"]), 10.)

    def test_existing_output_cannot_start_again(self):
        self.clean_finish()
        self.run_watchdog()
        original = (self.root / "out/watchdog-exit.json").read_bytes()
        with self.assertRaisesRegex(ValueError, "existing output"):
            self.run_watchdog()
        self.assertEqual(len(self.ops.started), 1)
        self.assertEqual((self.root / "out/watchdog-exit.json").read_bytes(), original)

    def test_deadline_stops_blocked_child_independently(self):
        result = self.run_watchdog()
        self.assertEqual(result["status"], "stopped_no_retry")
        self.assertIn("eight-hour wall", result["reason"])
        self.assertEqual(result["wall_s"], 2.)
        self.assertTrue(self.ops.signalled)
        self.assertEqual(self.ops.signalled[0][0]["pid"], 888)
        self.assertEqual(self.ops.signalled[0][1], signal.SIGTERM)
        self.assertTrue(result["lock_release_safe"])

    def test_request_cap_allows_clean_final_assessment_without_more_train(self):
        self.clean_finish(30000)
        result = self.run_watchdog()
        self.assertEqual(result["train_requested"], 30000)
        self.assertIsNone(result["reason"])

    def test_one_request_over_limit_stops_owned_child(self):
        self.ops.on_sleep = lambda ops: ops.write(30001)
        result = self.run_watchdog()
        self.assertIn("exceeded cap", result["reason"])
        self.assertTrue(self.ops.signalled)

    def test_reserved_requests_cannot_be_refunded(self):
        def change(ops):
            ops.write(10 if ops.t < 11 else 9)
        self.ops.on_sleep = change
        result = self.run_watchdog()
        self.assertIn("moved backwards", result["reason"])
        self.assertEqual(result["train_requested"], 10)

    def test_completion_cannot_exceed_reserved_requests(self):
        self.ops.on_sleep = lambda ops: ops.write(10, train_completed=11)
        result = self.run_watchdog()
        self.assertIn("completion counter", result["reason"])

    def test_storage_checked_before_start(self):
        self.ops.free = subject.MIN_FREE_BYTES - 1
        result = self.run_watchdog()
        self.assertEqual(result["native_application_starts"], 0)
        self.assertFalse(self.ops.started)
        self.assertIn("before start", result["reason"])

    def test_storage_drop_during_child_stops_work(self):
        self.ops.on_sleep = lambda ops: setattr(ops, "free", subject.MIN_FREE_BYTES - 1)
        result = self.run_watchdog()
        self.assertIn("storage floor", result["reason"])
        self.assertTrue(self.ops.signalled)

    def test_initialization_counts_towards_wall_limit(self):
        self.ops.prepare = lambda: setattr(self.ops, "t", 12.)
        result = self.run_watchdog()
        self.assertIn("before child start", result["reason"])
        self.assertFalse(self.ops.started)

    def test_cookie_needed_before_child_start(self):
        self.ops.records[777]["cookie"] = "wrong"
        result = self.run_watchdog()
        self.assertIn("session cookie", result["reason"])
        self.assertFalse(self.ops.started)

    def test_reused_leader_pid_is_never_signalled(self):
        self.ops.on_sleep = lambda ops: ops.records[888].update(start_token="foreign:1", cookie="foreign")
        original_signal = self.ops.signal_owned
        def guarded(record, signum, session):
            if record["start_token"] != "boot:88":
                return "ownership_mismatch"
            return original_signal(record, signum, session)
        self.ops.signal_owned = guarded
        result = self.run_watchdog()
        self.assertIn("ownership changed", result["reason"])
        self.assertFalse(self.ops.signalled)
        self.assertFalse(result["lock_release_safe"])
        # Release only the mock test's retained lease, never a production lease.
        stream = subject._RETAINED_LOCKS.pop()
        stream.close()

    def test_progress_changed_session_or_runtime_fails_closed(self):
        for field in ("session_id", "runtime_fingerprint"):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "progress.json"
                value = dict(session_id=self.session, runtime_fingerprint=self.fingerprint,
                             train_requested=1)
                value[field] = "wrong"
                path.write_text(json.dumps(subject.signed(value)))
                with self.assertRaisesRegex(ValueError, "binding changed"):
                    subject.read_progress(path, session_id=self.session, runtime_fingerprint=self.fingerprint)

    def test_fingerprint_and_duplicate_keys_are_checked(self):
        path = self.root / "progress.json"
        value = subject.signed(dict(session_id=self.session, runtime_fingerprint=self.fingerprint,
                                    train_requested=0))
        value["train_requested"] = 2
        path.write_text(json.dumps(value))
        with self.assertRaisesRegex(ValueError, "fingerprint"):
            subject.read_progress(path, session_id=self.session, runtime_fingerprint=self.fingerprint)
        path.write_text('{"train_requested":1,"train_requested":2}')
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            subject.read_progress(path, session_id=self.session, runtime_fingerprint=self.fingerprint)

    def test_oversized_or_linked_progress_refused(self):
        path = self.root / "progress.json"
        path.write_bytes(b" " * (subject.MAX_PROGRESS_BYTES + 1))
        with self.assertRaisesRegex(ValueError, "64 KiB"):
            subject.read_progress(path, session_id=self.session, runtime_fingerprint=self.fingerprint)
        link = self.root / "linked.json"
        link.symlink_to(path)
        with self.assertRaisesRegex(ValueError, "Linked"):
            subject.read_progress(link, session_id=self.session, runtime_fingerprint=self.fingerprint)

    def test_boolean_counter_is_not_an_integer_request(self):
        path = self.root / "progress.json"
        path.write_text(json.dumps(subject.signed(dict(session_id=self.session,
            runtime_fingerprint=self.fingerprint, train_requested=True))))
        with self.assertRaisesRegex(ValueError, "requests exceeded"):
            subject.read_progress(path, session_id=self.session, runtime_fingerprint=self.fingerprint)

    def test_no_progress_cannot_claim_zero_exit_success(self):
        def no_progress(ops):
            (ops.root / "out/progress.json").unlink()
            ops.exit_code = 0
        self.ops.on_start = no_progress
        result = self.run_watchdog()
        self.assertIn("without bound progress", result["reason"])

    def test_published_progress_cannot_disappear(self):
        self.ops.on_sleep = lambda ops: (ops.root / "out/progress.json").unlink()
        result = self.run_watchdog()
        self.assertIn("progress disappeared", result["reason"])
        self.assertTrue(self.ops.signalled)

    def test_busy_shared_lock_starts_nothing(self):
        import fcntl
        with (self.root / "scene.lock").open("w") as stream:
            stream.write("existing lock metadata")
            stream.flush()
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            with self.assertRaises(BlockingIOError):
                self.run_watchdog()
        self.assertFalse((self.root / "out").exists())
        self.assertFalse(self.ops.started)
        self.assertEqual((self.root / "scene.lock").read_text(), "existing lock metadata")

    def test_missing_shared_lock_cannot_create_a_different_lock(self):
        (self.root / "scene.lock").unlink()
        with self.assertRaisesRegex(ValueError, "Existing shared scene lock"):
            self.run_watchdog()
        self.assertFalse(self.ops.started)
        self.assertFalse((self.root / "scene.lock").exists())

    def test_limits_cannot_be_weakened(self):
        for arguments in (dict(wall_s=28801), dict(max_requests=30001),
                          dict(storage_floor_bytes=subject.MIN_FREE_BYTES - 1),
                          dict(poll_s=2)):
            with self.subTest(arguments=arguments), self.assertRaises(ValueError):
                self.run_watchdog(**arguments)
        self.assertFalse(self.ops.started)

    def test_module_import_is_inert_and_stdlib_only(self):
        source = str(Path(__file__).resolve().parents[1])
        code = ("import sys; sys.path.insert(0,sys.argv[1]); "
                "import seed22_continuation_watchdog; "
                "assert not {'numpy','scipy','torch','isaaclab','isaacsim'} & sys.modules.keys()")
        subprocess.run([sys.executable, "-B", "-S", "-c", code, source],
                       check=True, capture_output=True, timeout=10)

    def test_cli_requires_explicit_execution_flag(self):
        with self.assertRaisesRegex(ValueError, "approved execution"):
            subject.main(["--output", str(self.root / "out"), "--lock", str(self.root / "scene.lock"),
                "--session-id", self.session, "--runtime-fingerprint", self.fingerprint,
                "--", "/mock/python"])


if __name__ == "__main__":
    unittest.main()
