"""Fake process/clock evidence only: no simulator, real process, or signal."""

import copy
import json
from pathlib import Path
import signal
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import location_comparison_watchdog as subject
from location_comparison_bounds import SCHEMA as BOUNDS_SCHEMA

SESSION, RUNTIME, COMPARISON = "a" * 32, "b" * 64, "c" * 64


def plan():
    return dict(fingerprint=COMPARISON, protocol=dict(
        proposed_train_requests_per_arm=8960, proposed_total_train_requests=17920,
        proposed_wall_seconds_per_training_arm=28800,
        proposed_shared_assessment_wall_seconds=3600,
        minimum_free_storage_bytes=subject.MIN_FREE_BYTES, total_assessment_requests=600))


def snapshot(now=10., *, phase=0, requested=0, completed=0, assessment=0):
    active = ("shared", "original_usable", "broader", "shared")[phase]
    elapsed = dict(shared=0., original_usable=0., broader=0.)
    elapsed[active] = now - 10.
    train = dict(original_usable=requested, broader=0)
    done = dict(original_usable=completed, broader=0)
    if phase == 3:
        train = done = dict(original_usable=8960, broader=8960)
    return dict(schema=BOUNDS_SCHEMA, comparison_fingerprint=COMPARISON,
        started_monotonic=10., observed_monotonic=now, active_phase=active, phase_index=phase,
        phase_elapsed_s=elapsed, train_requested=copy.deepcopy(train), train_completed=copy.deepcopy(done),
        assessment_requested=assessment, assessment_completed=assessment,
        pending=None, reservations=0, stopped=None, free_storage_bytes=subject.MIN_FREE_BYTES + 1000)


class FakeOps:
    def __init__(self, root, *, audit=False):
        self.root, self.audit = root, audit
        self.t, self.free = 10., subject.MIN_FREE_BYTES + 1000
        self.started, self.signalled, self.prepared = [], [], False
        self.running, self.exit_code, self.on_sleep, self.on_start = False, None, None, None
        self.records = {777: dict(pid=777, start_token="boot:77", cookie=SESSION,
                                 ppid=1, sid=777, state="R")}
        self.writer = None

    def monotonic(self): return self.t
    def free_bytes(self, path): return self.free
    def own_pid(self): return 777
    def prepare(self): self.prepared = True
    def identity(self, pid): return copy.deepcopy(self.records.get(pid))

    def write(self, value=None, *, running=True, status="running"):
        return self.writer.publish(value or snapshot(self.t), running=running, status=status)

    def start(self, command, environment):
        self.started.append((command, environment))
        self.running = True
        self.records[888] = dict(pid=888, start_token="boot:88", cookie=SESSION,
                                ppid=777, sid=888, state="S")
        if not self.audit:
            self.writer = subject.ProgressWriter(self.root / "out", SESSION, RUNTIME, COMPARISON)
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
        if record["start_token"] != "boot:88" or record["cookie"] != SESSION:
            return "ownership_mismatch"
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
        (self.root / "scene.lock").write_text("existing lock metadata")
        self.ops = FakeOps(self.root)
        self.addCleanup(self.release_mock_retained_locks)
        self.real_start = patch("subprocess.Popen", side_effect=AssertionError("Real process forbidden"))
        self.real_start.start()
        self.addCleanup(self.real_start.stop)
        self.real_signal = patch("os.kill", side_effect=AssertionError("Real signal forbidden"))
        self.real_signal.start()
        self.addCleanup(self.real_signal.stop)

    def release_mock_retained_locks(self):
        while subject._RETAINED_LOCKS:
            subject._RETAINED_LOCKS.pop().close()

    def run_watchdog(self, **overrides):
        args = dict(command=["/mock/python", "inert.py"], plan=plan(), output=self.root / "out",
            lock=self.root / "scene.lock", session_id=SESSION, runtime_fingerprint=RUNTIME,
            ops=self.ops, integrity_check=lambda: True, execute_approved=True)
        args.update(overrides)
        return subject.supervise(**args)

    def finish(self):
        def finish(ops):
            ops.write(snapshot(ops.t, phase=3, assessment=600), running=False,
                      status="runner_complete_pending_audit")
            ops.exit_code = 0
        self.ops.on_sleep = finish

    def test_one_start_exact_counts_still_requires_independent_audit(self):
        self.finish()
        result = self.run_watchdog()
        self.assertEqual(result["status"], "native_exit_zero_independent_audit_required")
        self.assertEqual(result["final_snapshot"]["train_requested"], dict(original_usable=8960, broader=8960))
        self.assertEqual(result["native_application_starts"], 1)
        self.assertEqual(result["native_child_start_attempts"], 1)
        self.assertEqual(result["native_child_process_starts"], 1)
        self.assertIn("does not prove simulator", result["start_counter_scope"])
        self.assertFalse(result["independent_audit_passed"])
        self.assertFalse(result["model_promoted"])
        self.assertTrue(result["lock_release_safe"])
        self.assertFalse(self.ops.signalled)
        owner = json.loads((self.root / "out/comparison-watchdog-owner.json").read_text())
        self.assertEqual(owner["owner"]["pid"], 777)
        self.assertEqual(owner["native_owner"]["pid"], 888)
        env = self.ops.started[0][1]
        self.assertEqual(env["CUDA_VISIBLE_DEVICES"], subject.GPU_UUID)
        self.assertEqual(env[subject.COOKIE], SESSION)
        self.assertEqual(env["SIRCUS_ISAAC_SOURCE"], str(Path(subject.__file__).resolve().parent))
        self.assertEqual(float(env["SIRCUS_COMPARISON_STARTED_MONOTONIC"]), 10.)
        self.assertEqual(json.loads((self.root / "out/comparison-watchdog-exit.json").read_text()), result)

    def test_no_start_without_explicit_authorization_or_integrity(self):
        with self.assertRaisesRegex(ValueError, "approved execution"):
            self.run_watchdog(execute_approved=False)
        with self.assertRaisesRegex(ValueError, "integrity check required"):
            self.run_watchdog(integrity_check=None)
        result = self.run_watchdog(integrity_check=lambda: False)
        self.assertEqual(result["native_application_starts"], 0)
        self.assertFalse(self.ops.started)

    def test_failed_process_spawn_preserves_start_intent_without_claiming_process(self):
        def fail(command, environment):
            raise OSError("fake spawn failed")
        self.ops.start = fail
        result = self.run_watchdog()
        self.assertEqual(result["native_child_start_attempts"], 1)
        self.assertEqual(result["native_child_process_starts"], 0)
        self.assertEqual(result["termination"], "unknown_child_identity")
        self.assertFalse(result["lock_release_safe"])

    def test_snapshot_stall_still_expires_initial_shared_clock(self):
        self.ops.on_sleep = lambda ops: setattr(ops, "t", 3610.)
        result = self.run_watchdog()
        self.assertIn("wall deadline", result["reason"])
        self.assertEqual(result["termination"], "stopped")
        self.assertEqual(self.ops.signalled[0][1], signal.SIGTERM)

    def test_expired_deadline_skips_grace_and_kills_term_resistant_owned_child(self):
        def signal_owned(record, signum, session):
            self.ops.signalled.append((copy.deepcopy(record), signum, session))
            if signum == signal.SIGKILL:
                self.ops.running = False
                self.ops.records.pop(888, None)
            return "signalled"
        self.ops.signal_owned = signal_owned
        self.ops.on_sleep = lambda ops: setattr(ops, "t", 3610.)
        result = self.run_watchdog()
        self.assertEqual([row[1] for row in self.ops.signalled], [signal.SIGTERM, signal.SIGKILL])
        self.assertEqual(result["wall_s"], 3600.)
        self.assertEqual(result["termination"], "stopped")
        self.assertTrue(result["lock_release_safe"])

    def test_blocked_train_phase_uses_independent_eight_hour_limit(self):
        def begin(ops):
            ops.write(snapshot(ops.t, phase=1, requested=35))
        self.ops.on_start = begin
        self.ops.on_sleep = lambda ops: setattr(ops, "t", 28810.)
        result = self.run_watchdog()
        self.assertIn("wall deadline", result["reason"])
        self.assertEqual(result["final_snapshot"]["train_requested"]["original_usable"], 35)

    def test_initialization_counts_against_shared_hour(self):
        self.ops.prepare = lambda: setattr(self.ops, "t", 3610.)
        result = self.run_watchdog()
        self.assertIn("deadline", result["reason"])
        self.assertFalse(self.ops.started)

    def test_cleanup_after_final_snapshot_still_uses_shared_deadline(self):
        self.ops.on_start = lambda ops: ops.write(snapshot(ops.t, phase=3, assessment=600),
            running=False, status="runner_complete_pending_audit")
        self.ops.on_sleep = lambda ops: setattr(ops, "t", 3610.)
        result = self.run_watchdog()
        self.assertIn("wall deadline", result["reason"])
        self.assertTrue(self.ops.signalled)

    def test_unchanged_progress_cannot_extend_deadline(self):
        def stale(ops):
            ops.write(snapshot(10.))
            ops.t = 3610.
        self.ops.on_sleep = stale
        result = self.run_watchdog()
        self.assertIn("wall deadline", result["reason"])

    def test_counter_overflow_or_rollback_stops(self):
        self.ops.on_start = lambda ops: ops.write(snapshot(ops.t, phase=1, requested=35))
        self.ops.on_sleep = lambda ops: ops.write(snapshot(ops.t, phase=1, requested=34))
        result = self.run_watchdog()
        self.assertIn("moved backwards", result["reason"])
        self.assertTrue(self.ops.signalled)

    def test_future_phase_cannot_absorb_elapsed_time(self):
        def changed(ops):
            value = snapshot(ops.t)
            value["phase_elapsed_s"] = dict(shared=0., original_usable=ops.t-10., broader=0.)
            ops.write(value)
        self.ops.on_sleep = changed
        result = self.run_watchdog()
        self.assertIn("Future phase", result["reason"])

    def test_inactive_completed_phase_cannot_absorb_train_time(self):
        self.ops.on_start = lambda ops: ops.write(snapshot(ops.t, phase=1))
        def changed(ops):
            value = snapshot(ops.t, phase=1)
            value["phase_elapsed_s"] = dict(shared=ops.t-10., original_usable=0., broader=0.)
            ops.write(value)
        self.ops.on_sleep = changed
        result = self.run_watchdog()
        self.assertIn("Inactive phase", result["reason"])

    def test_cap_cannot_be_widened_by_plan(self):
        changed = plan()
        changed["protocol"]["proposed_train_requests_per_arm"] = 8961
        with self.assertRaisesRegex(ValueError, "Unsupported"):
            self.run_watchdog(plan=changed)
        self.assertFalse(self.ops.started)

    def test_storage_checked_before_and_during_child(self):
        self.ops.on_sleep = lambda ops: setattr(ops, "free", subject.MIN_FREE_BYTES - 1)
        result = self.run_watchdog()
        self.assertIn("storage", result["reason"])
        self.assertTrue(self.ops.signalled)

    def test_input_mutation_fails_closed_during_child(self):
        self.ops.on_sleep = lambda ops: setattr(ops, "changed", True)
        result = self.run_watchdog(integrity_check=lambda: not getattr(self.ops, "changed", False))
        self.assertIn("integrity", result["reason"])
        self.assertTrue(self.ops.signalled)

    def test_reused_pid_is_not_signalled(self):
        self.ops.on_sleep = lambda ops: self.ops.records[888].update(start_token="foreign:1", cookie="foreign")
        result = self.run_watchdog()
        self.assertIn("ownership changed", result["reason"])
        self.assertFalse(self.ops.signalled)
        self.assertFalse(result["lock_release_safe"])

    def test_real_adapter_pidfd_pin_rechecks_before_signalling(self):
        record = self.ops.identity(777)
        adapter = subject.LinuxProcessOps()
        adapter.identity = lambda pid: copy.deepcopy(record)
        adapter.owned = lambda pid, session: [copy.deepcopy(record)]
        calls = []
        with patch("candidate_continuation_watchdog.pidfd_open", return_value=41), \
             patch("candidate_continuation_watchdog.pidfd_send_signal", side_effect=lambda *a: calls.append(a)), \
             patch("candidate_continuation_watchdog.os.close") as close:
            self.assertEqual(adapter.signal_owned(record, signal.SIGTERM, SESSION), "signalled")
            self.assertEqual(calls, [(41, signal.SIGTERM, None, 0)])
            close.assert_called_once_with(41)
        copies = [copy.deepcopy(record), dict(record, start_token="reused:1")]
        adapter.identity = lambda pid: copies.pop(0)
        with patch("candidate_continuation_watchdog.pidfd_open", return_value=42), \
             patch("candidate_continuation_watchdog.pidfd_send_signal") as send, \
             patch("candidate_continuation_watchdog.os.close") as close:
            self.assertEqual(adapter.signal_owned(record, signal.SIGTERM, SESSION), "ownership_mismatch")
            send.assert_not_called()
            close.assert_called_once_with(42)

    def test_process_exits_zero_without_complete_evidence_fails(self):
        self.ops.on_sleep = lambda ops: setattr(ops, "exit_code", 0)
        result = self.run_watchdog()
        self.assertIn("final progress", result["reason"])
        self.assertEqual(result["status"], "stopped_no_retry")

    def transient_child_read(self, ops, read_number, *, absent=False):
        original = ops.identity
        count = [0]
        def identity(pid):
            value = original(pid)
            if pid == 888:
                count[0] += 1
                if count[0] == read_number:
                    return None if absent else dict(value, cookie="")
            return value
        ops.identity = identity

    def finish_after_identity_retry(self, ops):
        if ops.t < 10.1:
            return
        ops.write(snapshot(ops.t, phase=3, assessment=600), running=False,
                  status="runner_complete_pending_audit")
        ops.exit_code = 0

    def test_initial_empty_cookie_is_reread_before_owner_receipt(self):
        self.transient_child_read(self.ops, 1)
        self.ops.on_sleep = self.finish_after_identity_retry
        result = self.run_watchdog()
        self.assertEqual(result["status"], "native_exit_zero_independent_audit_required")
        self.assertEqual(result["native_owner"]["cookie"], SESSION)
        self.assertEqual(len(self.ops.started), 1)
        self.assertFalse(self.ops.signalled)

    def test_initial_persistent_empty_cookie_retains_unconfirmed_cleanup_and_lock(self):
        original = self.ops.identity
        self.ops.identity = lambda pid:dict(original(pid),cookie="") if pid == 888 else original(pid)
        result = self.run_watchdog()
        self.assertEqual(result["status"], "stopped_no_retry")
        self.assertIsNone(result["native_owner"])
        self.assertEqual(result["termination"], "unknown_child_identity")
        self.assertFalse(result["lock_release_safe"])
        self.assertFalse(self.ops.signalled)
        self.assertEqual(len(self.ops.started), 1)
        self.assertFalse((self.root/"out/comparison-watchdog-owner.json").exists())

    def test_initial_empty_cookie_cannot_adopt_reused_start_token(self):
        original, reads = self.ops.identity, [0]
        def identity(pid):
            value = original(pid)
            if pid == 888:
                reads[0] += 1
                return dict(value,cookie="") if reads[0] == 1 else dict(value,start_token="reused:999")
            return value
        self.ops.identity = identity
        result = self.run_watchdog()
        self.assertIn("ownership changed", result["reason"])
        self.assertEqual(result["termination"], "unknown_child_identity")
        self.assertFalse(result["lock_release_safe"])
        self.assertFalse(self.ops.signalled)

    def test_persistent_permission_denial_does_not_weaken_cleanup_ownership(self):
        original, reads = self.ops.identity, [0]
        def identity(pid):
            if pid == 888:
                reads[0] += 1
                if reads[0] > 1:
                    raise PermissionError("synthetic unreadable environ")
            return original(pid)
        self.ops.identity = identity
        result = self.run_watchdog()
        self.assertEqual(result["status"], "stopped_no_retry")
        self.assertEqual(result["native_owner"]["cookie"], SESSION)
        self.assertIn("identity", result["reason"])
        self.assertEqual(result["termination"], "termination_unconfirmed:PermissionError")
        self.assertFalse(result["lock_release_safe"])
        self.assertFalse(self.ops.signalled)
        self.assertEqual(len(self.ops.started), 1)

    def test_live_missing_cookie_recovers_without_signal_or_second_start(self):
        self.transient_child_read(self.ops, 2)
        self.ops.on_sleep = self.finish_after_identity_retry
        result = self.run_watchdog()
        self.assertEqual(result["status"], "native_exit_zero_independent_audit_required")
        self.assertEqual(len(self.ops.started), 1)
        self.assertFalse(self.ops.signalled)

    def test_exit_between_poll_and_identity_reads_fresh_final_progress(self):
        original_poll = self.ops.poll
        polls = [0]
        def poll(pid):
            polls[0] += 1
            if polls[0] == 2:
                # Initial owner is already pinned. The child finishes after
                # progress was read and the first poll reported it still live.
                self.ops.write(snapshot(self.ops.t, phase=3, assessment=600), running=False,
                               status="runner_complete_pending_audit")
                self.ops.exit_code = 0
                self.ops.records[888].update(state="Z", cookie="")
                return None
            return original_poll(pid)
        self.ops.poll = poll
        result = self.run_watchdog()
        self.assertEqual(result["status"], "native_exit_zero_independent_audit_required")
        self.assertEqual(result["final_snapshot"]["assessment_completed"], 600)
        self.assertFalse(self.ops.signalled)

    def test_progress_published_inside_successful_poll_is_not_stale_at_terminal_check(self):
        original_poll = self.ops.poll
        polls = [0]
        def poll(pid):
            polls[0] += 1
            if polls[0] == 2:
                self.ops.write(snapshot(self.ops.t, phase=3, assessment=600), running=False,
                               status="runner_complete_pending_audit")
                self.ops.exit_code = 0
            return original_poll(pid)
        self.ops.poll = poll
        self.assertEqual(self.run_watchdog()["status"], "native_exit_zero_independent_audit_required")

    def test_storage_loss_during_identity_reread_stops_without_continuing(self):
        self.transient_child_read(self.ops, 2)
        self.ops.on_sleep = lambda ops: setattr(ops, "free", subject.MIN_FREE_BYTES - 1)
        result = self.run_watchdog()
        self.assertIn("storage", result["reason"])
        self.assertEqual(result["status"], "stopped_no_retry")

    def test_integrity_loss_during_identity_reread_stops_without_continuing(self):
        self.transient_child_read(self.ops, 2)
        self.ops.on_sleep = lambda ops: setattr(ops, "changed", True)
        result = self.run_watchdog(integrity_check=lambda:not getattr(self.ops,"changed",False))
        self.assertIn("integrity", result["reason"])
        self.assertEqual(result["status"], "stopped_no_retry")

    def test_phase_deadline_during_identity_reread_is_not_extended(self):
        self.transient_child_read(self.ops, 2)
        self.ops.on_sleep = lambda ops: setattr(ops, "t", 3610.)
        result = self.run_watchdog()
        self.assertIn("deadline", result["reason"])
        self.assertEqual(result["status"], "stopped_no_retry")

    def test_existing_output_preserved_and_cannot_retry(self):
        self.finish()
        self.run_watchdog()
        old = (self.root / "out/comparison-watchdog-exit.json").read_bytes()
        with self.assertRaisesRegex(ValueError, "existing output"):
            self.run_watchdog()
        self.assertEqual(old, (self.root / "out/comparison-watchdog-exit.json").read_bytes())
        self.assertEqual(len(self.ops.started), 1)

    def test_busy_or_missing_shared_lock_cannot_start(self):
        import fcntl
        with (self.root / "scene.lock").open("r+") as stream:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with self.assertRaises(BlockingIOError):
                self.run_watchdog()
        self.assertFalse((self.root / "out").exists())
        (self.root / "scene.lock").unlink()
        with self.assertRaisesRegex(ValueError, "Existing shared scene lock"):
            self.run_watchdog()
        self.assertFalse(self.ops.started)

    def test_changed_lock_inode_stops_owned_child(self):
        def replace(ops):
            (self.root / "scene.lock").unlink()
            (self.root / "scene.lock").write_text("foreign lock")
        self.ops.on_sleep = replace
        result = self.run_watchdog()
        self.assertIn("lock identity", result["reason"])
        self.assertTrue(self.ops.signalled)

    def test_atomic_writer_and_reader_reject_rewrite_and_reopen(self):
        out = self.root / "progress-only"
        out.mkdir()
        writer = subject.ProgressWriter(out, SESSION, RUNTIME, COMPARISON)
        first = writer.publish(snapshot())
        read = lambda previous=None: subject.read_progress(out, session_id=SESSION,
            runtime_fingerprint=RUNTIME, comparison_fingerprint=COMPARISON, previous=previous)
        self.assertEqual(read(), first)
        changed = dict(first, status="failed_no_retry", running=False)
        changed = subject.signed({k: v for k, v in changed.items() if k != "fingerprint"})
        (out / "comparison-progress.json").write_text(json.dumps(changed))
        with self.assertRaisesRegex(ValueError, "rewritten"):
            read(first)
        with self.assertRaisesRegex(ValueError, "already exists"):
            subject.ProgressWriter(out, SESSION, RUNTIME, COMPARISON)

    def test_progress_disappearance_and_wrong_binding_stop(self):
        self.ops.on_sleep = lambda ops: (self.root / "out/comparison-progress.json").unlink()
        result = self.run_watchdog()
        self.assertIn("disappeared", result["reason"])
        self.assertTrue(self.ops.signalled)

    def test_progress_symlink_or_duplicate_json_rejected(self):
        out = self.root / "progress-only"
        out.mkdir()
        path = out / "comparison-progress.json"
        path.write_text('{"sequence":0,"sequence":1}')
        args = dict(session_id=SESSION, runtime_fingerprint=RUNTIME, comparison_fingerprint=COMPARISON)
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            subject.read_progress(out, **args)
        path.unlink()
        path.symlink_to(self.root / "scene.lock")
        with self.assertRaisesRegex(ValueError, "Linked"):
            subject.read_progress(out, **args)

    def test_terminating_writer_cannot_publish_again(self):
        out = self.root / "progress-only"
        out.mkdir()
        writer = subject.ProgressWriter(out, SESSION, RUNTIME, COMPARISON)
        writer.publish(snapshot(), running=False, status="failed_no_retry")
        with self.assertRaisesRegex(ValueError, "already closed"):
            writer.publish(snapshot())

    def audit(self, native_exit, ops, **overrides):
        args = dict(command=["/mock/python", "audit-only.py"], output=self.root / "audit",
            native_exit=native_exit, session_id=SESSION, runtime_fingerprint=RUNTIME,
            comparison_fingerprint=COMPARISON, integrity_check=lambda: True,
            verify_report=lambda output, receipt: receipt == native_exit["fingerprint"],
            execute_approved=True, ops=ops)
        args.update(overrides)
        return subject.supervise_audit(**args)

    def finished_native(self):
        self.finish()
        return self.run_watchdog()

    def test_cpu_audit_is_separate_and_report_bound(self):
        native = self.finished_native()
        ops = FakeOps(self.root, audit=True)
        ops.on_sleep = lambda x: setattr(x, "exit_code", 0)
        audit = self.audit(native, ops)
        self.assertEqual(audit["status"], "audit_exit_zero_report_verified")
        self.assertEqual(audit["native_exit_fingerprint"], native["fingerprint"])
        self.assertEqual(audit["native_application_starts"], 0)
        self.assertEqual(audit["audit_process_starts"], 1)
        self.assertEqual(ops.started[0][1]["CUDA_VISIBLE_DEVICES"], "")
        self.assertTrue(audit["independent_report_verified"])
        self.assertFalse(audit["model_promoted"])

    def test_audit_two_hour_deadline_preempts_blocked_reader(self):
        native = self.finished_native()
        ops = FakeOps(self.root, audit=True)
        ops.on_sleep = lambda x: setattr(x, "t", 7210.)
        audit = self.audit(native, ops)
        self.assertIn("deadline", audit["reason"])
        self.assertTrue(ops.signalled)
        self.assertFalse(audit["independent_report_verified"])

    def test_audit_nonfinite_clock_records_failure_without_nonfinite_json(self):
        native = self.finished_native()
        ops = FakeOps(self.root, audit=True)
        ops.on_sleep = lambda x: setattr(x, "t", float("nan"))
        audit = self.audit(native, ops)
        self.assertEqual(audit["status"], "audit_stopped_no_retry")
        self.assertIsNone(audit["wall_s"])
        self.assertIn("clock failed", audit["reason"])
        self.assertTrue((self.root / "audit/comparison-audit-exit.json").is_file())

    def test_audit_exit_zero_without_verified_report_fails(self):
        native = self.finished_native()
        ops = FakeOps(self.root, audit=True)
        ops.on_sleep = lambda x: setattr(x, "exit_code", 0)
        audit = self.audit(native, ops, verify_report=lambda output, receipt: False)
        self.assertIn("report did not verify", audit["reason"])
        self.assertFalse(audit["independent_report_verified"])

    def test_audit_cannot_follow_unverified_or_mutated_native_receipt(self):
        native = self.finished_native()
        native["status"] = "stopped_no_retry"
        ops = FakeOps(self.root, audit=True)
        with self.assertRaisesRegex(ValueError, "exact successful native exit"):
            self.audit(native, ops)
        self.assertFalse(ops.started)

    def test_audit_reused_pid_does_not_signal_foreign_process(self):
        native = self.finished_native()
        ops = FakeOps(self.root, audit=True)
        ops.on_sleep = lambda x: x.records[888].update(start_token="foreign:1", cookie="foreign")
        audit = self.audit(native, ops)
        self.assertIn("ownership changed", audit["reason"])
        self.assertFalse(ops.signalled)
        self.assertFalse(audit["cleanup_safe"])

    def test_audit_initial_empty_cookie_reread_keeps_report_gate(self):
        native = self.finished_native()
        ops = FakeOps(self.root, audit=True)
        self.transient_child_read(ops, 1)
        ops.on_sleep = lambda x: setattr(x, "exit_code", 0) if x.t >= 10.1 else None
        result = self.audit(native, ops, verify_report=lambda *args:False)
        self.assertIn("report did not verify", result["reason"])
        self.assertEqual(result["native_owner"]["cookie"], SESSION)
        self.assertEqual(len(ops.started), 1)

    def test_audit_exit_during_identity_read_uses_actual_exit_status(self):
        native = self.finished_native()
        ops = FakeOps(self.root, audit=True)
        original_identity = ops.identity
        reads = [0]
        def identity(pid):
            if pid == 888:
                reads[0] += 1
                if reads[0] == 2:
                    ops.exit_code = 9
                    ops.records[888].update(state="Z", cookie="")
            return original_identity(pid)
        ops.identity = identity
        result = self.audit(native, ops)
        self.assertEqual(result["exit_code"], 9)
        self.assertIn("exited unsuccessfully", result["reason"])
        self.assertFalse(ops.signalled)


class ChildObservationTests(unittest.TestCase):
    """Only synthetic process snapshots; no process start or signaling."""

    def setUp(self):
        self.ops = FakeOps(Path("/unused"))
        self.owner = dict(pid=888, start_token="boot:88", cookie=SESSION, ppid=777, sid=888, state="R")
        self.ops.records[888] = copy.deepcopy(self.owner)
        self.ops.running = True
        self.checks = 0

    def check(self):
        self.checks += 1

    def observe(self, owner=None, check=None):
        return subject._observe_child(self.ops,888,SESSION,owner=owner,check=check or self.check,label="Test child")

    def test_missing_initial_snapshot_then_full_identity(self):
        values = [None, copy.deepcopy(self.owner)]
        self.ops.identity = lambda pid: values.pop(0)
        current, code = self.observe()
        self.assertEqual(current, self.owner)
        self.assertIsNone(code)
        self.assertEqual(self.checks, 3)

    def test_unreadable_identity_then_confirmed_exit_uses_exact_child_status(self):
        for error in (PermissionError, FileNotFoundError, ProcessLookupError):
            with self.subTest(error=error.__name__):
                self.ops.exit_code = None
                def identity(pid):
                    self.assertEqual(pid, 888)
                    self.ops.exit_code = 7
                    raise error("synthetic process read race")
                self.ops.identity = identity
                current, code = self.observe(owner=self.owner)
                self.assertIsNone(current)
                self.assertEqual(code, 7)
                self.assertFalse(self.ops.signalled)

    def test_unreadable_identity_recovers_only_with_strict_full_identity(self):
        for error in (PermissionError, FileNotFoundError, ProcessLookupError):
            for initial in (True, False):
                with self.subTest(error=error.__name__, initial=initial):
                    reads = [0]
                    def identity(pid):
                        reads[0] += 1
                        if reads[0] == 1:
                            raise error("synthetic process read race")
                        return copy.deepcopy(self.owner)
                    self.ops.identity = identity
                    current, code = self.observe(owner=None if initial else self.owner)
                    self.assertEqual(current, self.owner)
                    self.assertIsNone(code)

    def test_persistent_permission_denial_is_bounded_and_never_accepted(self):
        def identity(pid):
            raise PermissionError("synthetic unreadable environ")
        self.ops.identity = identity
        with self.assertRaisesRegex(ValueError,"identity (unavailable|read deadline)"):
            self.observe(owner=self.owner)
        self.assertLessEqual(self.checks, subject.IDENTITY_READ_MAX_ATTEMPTS)
        self.assertLessEqual(self.ops.t-10., subject.IDENTITY_READ_MAX_S + 1e-9)
        self.assertFalse(self.ops.signalled)

    def test_identity_change_after_permission_denial_is_never_adopted(self):
        for field, value in (("pid",889), ("start_token","reused:2"), ("ppid",666), ("sid",889), ("cookie","foreign")):
            with self.subTest(field=field):
                reads = [0]
                def identity(pid):
                    reads[0] += 1
                    if reads[0] == 1:
                        raise PermissionError("synthetic unreadable environ")
                    return dict(self.owner, **{field:value})
                self.ops.identity = identity
                with self.assertRaisesRegex(ValueError,"ownership changed"):
                    self.observe(owner=self.owner)
                self.assertFalse(self.ops.signalled)

    def test_other_identity_errors_are_not_swallowed_or_retried(self):
        def identity(pid):
            raise OSError("unrelated I/O failure")
        self.ops.identity = identity
        self.ops.sleep = lambda _:self.fail("Unrelated error must not be retried")
        with self.assertRaisesRegex(OSError,"unrelated I/O failure"):
            self.observe(owner=self.owner)

    def test_first_empty_cookie_pins_start_and_ancestry_before_recovery(self):
        for field, value in (("pid",889), ("start_token","reused:2"), ("ppid",666), ("sid",889), ("cookie","foreign")):
            with self.subTest(field=field):
                values = [dict(self.owner,cookie=""), dict(self.owner, **{field:value})]
                self.ops.identity = lambda pid: values.pop(0)
                with self.assertRaisesRegex(ValueError,"ownership changed"):
                    self.observe()

    def test_nonempty_wrong_cookie_never_waits_or_passes(self):
        self.ops.records[888]["cookie"] = "foreign"
        self.ops.sleep = lambda _:self.fail("Wrong cookie must not be retried")
        with self.assertRaisesRegex(ValueError,"ownership changed"):
            self.observe(owner=self.owner)

    def test_persistent_missing_or_empty_identity_is_bounded(self):
        for value in (None, dict(self.owner,cookie="")):
            with self.subTest(value=value):
                self.ops.t, self.checks = 10., 0
                self.ops.identity = lambda pid:copy.deepcopy(value)
                with self.assertRaisesRegex(ValueError,"identity (unavailable|read deadline)"):
                    self.observe(owner=self.owner)
                self.assertLessEqual(self.checks, subject.IDENTITY_READ_MAX_ATTEMPTS)
                self.assertLessEqual(self.ops.t-10., subject.IDENTITY_READ_MAX_S + 1e-9)

    def test_frozen_clock_still_has_finite_read_bound(self):
        self.ops.identity = lambda pid:None
        self.ops.sleep = lambda _:None
        with self.assertRaisesRegex(ValueError,"bounded reread"):
            self.observe(owner=self.owner)
        self.assertEqual(self.checks, subject.IDENTITY_READ_MAX_ATTEMPTS)

    def test_slow_full_identity_check_does_not_trigger_transient_retry_limit(self):
        def check():
            self.ops.t += 1.
        current, code = self.observe(owner=self.owner, check=check)
        self.assertEqual(current, self.owner)
        self.assertIsNone(code)

    def test_recovered_identity_read_cannot_cross_retry_deadline(self):
        reads = [0]
        def identity(pid):
            reads[0] += 1
            if reads[0] == 1:
                return dict(self.owner,cookie="")
            self.ops.t += .2
            return copy.deepcopy(self.owner)
        self.ops.identity = identity
        with self.assertRaisesRegex(ValueError,"identity read deadline"):
            self.observe(owner=self.owner)

    def test_recovery_rechecks_bounds_after_the_identity_read(self):
        reads = [0]
        def identity(pid):
            reads[0] += 1
            if reads[0] == 1:
                return dict(self.owner,cookie="")
            self.ops.free = subject.MIN_FREE_BYTES - 1
            return copy.deepcopy(self.owner)
        self.ops.identity = identity
        def check():
            if self.ops.free < subject.MIN_FREE_BYTES:
                raise ValueError("storage floor")
        with self.assertRaisesRegex(ValueError,"storage floor"):
            self.observe(owner=self.owner,check=check)


if __name__ == "__main__":
    unittest.main()
