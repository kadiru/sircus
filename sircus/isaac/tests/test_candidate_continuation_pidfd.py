"""Mock-only pidfd ABI/capability checks; no real pidfd or signal operation."""

import ctypes
import errno
from pathlib import Path
import signal
import sys
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import candidate_continuation_pidfd as subject


class PidfdTests(unittest.TestCase):
    def setUp(self):
        self.platform = patch.object(subject.sys, "platform", "linux")
        self.machine = patch.object(subject.platform, "machine", return_value="x86_64")
        self.platform.start(); self.machine.start()
        self.addCleanup(self.platform.stop); self.addCleanup(self.machine.stop)
        # Every possible operational boundary is mocked for every test.
        self.opened = Mock(return_value=44)
        self.sent = Mock(return_value=None)
        self.libc = Mock()
        self.libc.syscall.side_effect = [44, 0]
        for target, name, value in ((subject.os, "pidfd_open", self.opened),
                (subject.signal, "pidfd_send_signal", self.sent)):
            guard = patch.object(target, name, value, create=True)
            guard.start(); self.addCleanup(guard.stop)
        self.library_guard = patch.object(ctypes, "CDLL", return_value=self.libc)
        self.library = self.library_guard.start(); self.addCleanup(self.library_guard.stop)
        self.close_guard = patch.object(subject.os, "close")
        self.closed = self.close_guard.start(); self.addCleanup(self.close_guard.stop)
        self.kill_guard = patch.object(subject.os, "kill")
        self.killed = self.kill_guard.start(); self.addCleanup(self.kill_guard.stop)
        self.group_guard = patch.object(subject.os, "killpg", create=True)
        self.grouped = self.group_guard.start(); self.addCleanup(self.group_guard.stop)

    def tearDown(self):
        self.killed.assert_not_called(); self.grouped.assert_not_called()

    def fallback(self):
        with patch.object(subject.os, "pidfd_open", None), \
                patch.object(subject.signal, "pidfd_send_signal", None):
            return subject.PidfdAdapter()

    def test_construction_and_backend_selection_do_not_open_signal_or_load_libc(self):
        adapter = subject.PidfdAdapter()
        self.assertEqual(adapter.backends, dict(pidfd_open="cpython", pidfd_send_signal="cpython"))
        self.opened.assert_not_called(); self.sent.assert_not_called(); self.library.assert_not_called()
        adapter = self.fallback()
        self.assertEqual(adapter.backends, dict(pidfd_open="linux-x86_64-syscall",
            pidfd_send_signal="linux-x86_64-syscall"))
        self.library.assert_not_called()

    def test_cpython_apis_are_preferred_with_exact_original_arguments(self):
        adapter = subject.PidfdAdapter()
        self.assertEqual(adapter.open(123, 0), 44)
        self.assertIsNone(adapter.send_signal(44, signal.SIGTERM, None, 0))
        self.opened.assert_called_once_with(123, 0)
        self.sent.assert_called_once_with(44, signal.SIGTERM, None, 0)
        self.library.assert_not_called(); self.closed.assert_not_called()

    def test_fallback_passes_reviewed_x86_64_syscall_numbers_and_typed_null_arguments(self):
        adapter = self.fallback()
        self.assertEqual(adapter.open(123, 0), 44)
        self.assertIsNone(adapter.send_signal(44, signal.SIGKILL, None, 0))
        self.library.assert_called_once_with(None, use_errno=True)
        self.assertIs(self.libc.syscall.restype, ctypes.c_long)
        opened, sent = [row.args for row in self.libc.syscall.call_args_list]
        self.assertEqual([arg.value for arg in opened], [434, 123, 0])
        self.assertEqual([type(arg) for arg in opened], [ctypes.c_long, ctypes.c_int, ctypes.c_uint])
        self.assertEqual([arg.value for arg in sent], [424, 44, signal.SIGKILL, None, 0])
        self.assertEqual([type(arg) for arg in sent],
            [ctypes.c_long, ctypes.c_int, ctypes.c_int, ctypes.c_void_p, ctypes.c_uint])
        self.opened.assert_not_called(); self.sent.assert_not_called()

    def test_each_existing_cpython_wrapper_is_used_in_a_mixed_build(self):
        with patch.object(subject.signal, "pidfd_send_signal", None):
            adapter = subject.PidfdAdapter()
        self.libc.syscall.side_effect = [0]
        self.assertEqual(adapter.open(123), 44)
        adapter.send_signal(44, signal.SIGTERM)
        self.opened.assert_called_once_with(123, 0); self.sent.assert_not_called()
        self.assertEqual(adapter.backends, dict(pidfd_open="cpython", pidfd_send_signal="linux-x86_64-syscall"))
        self.library.reset_mock(); self.opened.reset_mock()
        with patch.object(subject.os, "pidfd_open", None):
            adapter = subject.PidfdAdapter()
        self.libc.syscall.side_effect = [45]
        self.assertEqual(adapter.open(123), 45)
        adapter.send_signal(45, signal.SIGTERM)
        self.opened.assert_not_called(); self.sent.assert_called_once_with(45, signal.SIGTERM, None, 0)
        self.assertEqual(adapter.backends, dict(pidfd_open="linux-x86_64-syscall", pidfd_send_signal="cpython"))

    def test_other_platforms_and_architectures_fail_before_any_operation(self):
        for platform, machine in (("darwin", "x86_64"), ("linux", "aarch64"),
                ("linux", "AMD64"), ("win32", "x86_64")):
            with self.subTest(platform=platform, machine=machine), \
                    patch.object(subject.sys, "platform", platform), \
                    patch.object(subject.platform, "machine", return_value=machine):
                with self.assertRaisesRegex(ValueError, "Linux x86_64"):
                    subject.PidfdAdapter()
        self.opened.assert_not_called(); self.sent.assert_not_called(); self.library.assert_not_called()

    def test_32_bit_userspace_on_an_x86_64_kernel_is_not_the_reviewed_syscall_abi(self):
        with patch.object(subject.sys, "maxsize", 2 ** 31 - 1):
            with self.assertRaisesRegex(ValueError, "64-bit Linux x86_64"):
                subject.PidfdAdapter()
        self.opened.assert_not_called(); self.sent.assert_not_called(); self.library.assert_not_called()

    def test_native_errors_propagate_without_switching_to_syscalls(self):
        adapter = subject.PidfdAdapter()
        self.opened.side_effect = OSError(errno.ENOSYS, "unavailable")
        with self.assertRaises(OSError) as raised:
            adapter.open(123)
        self.assertEqual(raised.exception.errno, errno.ENOSYS)
        self.sent.side_effect = PermissionError(errno.EPERM, "denied")
        with self.assertRaises(PermissionError):
            adapter.send_signal(44, signal.SIGTERM)
        self.library.assert_not_called()

    def test_syscall_errno_preserves_process_lookup_permission_and_unsupported_failures(self):
        for code, exception in ((errno.ESRCH, ProcessLookupError), (errno.EPERM, PermissionError),
                (errno.ENOSYS, OSError), (errno.EINVAL, OSError)):
            with self.subTest(errno=code):
                adapter = self.fallback()
                self.libc.syscall.side_effect = [-1]
                with patch.object(ctypes, "get_errno", return_value=code), \
                        patch.object(ctypes, "set_errno") as cleared:
                    with self.assertRaises(exception) as raised:
                        adapter.open(123)
                self.assertEqual(raised.exception.errno, code)
                cleared.assert_called_once_with(0)

    def test_capability_probe_can_only_open_self_send_zero_then_close(self):
        with patch.object(subject.os, "getpid", return_value=567):
            result = subject.capability_preflight()
        self.opened.assert_called_once_with(567, 0)
        self.sent.assert_called_once_with(44, 0, None, 0)
        self.closed.assert_called_once_with(44)
        self.assertEqual(result["status"], "passed")
        self.assertEqual(result["signal"], 0)
        self.assertTrue(result["self_only"]); self.assertTrue(result["descriptor_closed"])
        self.assertEqual(result["process_starts"], 0); self.assertFalse(result["execution_authorized"])
        self.library.assert_not_called()

    def test_fallback_capability_probe_has_same_self_zero_close_scope(self):
        adapter = self.fallback()
        with patch.object(subject.os, "getpid", return_value=567):
            result = adapter.capability_preflight()
        opened, sent = [row.args for row in self.libc.syscall.call_args_list]
        self.assertEqual([arg.value for arg in opened], [434, 567, 0])
        self.assertEqual([arg.value for arg in sent], [424, 44, 0, None, 0])
        self.closed.assert_called_once_with(44)
        self.assertEqual(result["backends"]["pidfd_open"], "linux-x86_64-syscall")

    def test_probe_closes_on_native_or_kernel_signal_zero_failure_and_never_claims_pass(self):
        self.sent.side_effect = PermissionError(errno.EPERM, "blocked")
        with self.assertRaises(PermissionError):
            subject.capability_preflight()
        self.closed.assert_called_once_with(44)
        self.closed.reset_mock()
        adapter = self.fallback(); self.libc.syscall.side_effect = [44, -1]
        with patch.object(ctypes, "get_errno", return_value=errno.ENOSYS):
            with self.assertRaises(OSError) as raised:
                adapter.capability_preflight()
        self.assertEqual(raised.exception.errno, errno.ENOSYS)
        self.closed.assert_called_once_with(44)

    def test_open_failure_has_no_descriptor_to_close_and_no_signal(self):
        self.opened.side_effect = PermissionError(errno.EPERM, "blocked")
        with self.assertRaises(PermissionError):
            subject.capability_preflight()
        self.closed.assert_not_called(); self.sent.assert_not_called()

    def test_zero_flags_null_info_and_narrow_signals_cannot_be_overridden(self):
        adapter = self.fallback()
        for pid, flags in ((0, 0), (-1, 0), (True, 0), (2 ** 31, 0), (123, 1), (123, False)):
            with self.subTest(pid=pid, flags=flags), self.assertRaises(ValueError):
                adapter.open(pid, flags)
        for descriptor, signum, info, flags in ((-1, 0, None, 0), (True, 0, None, 0),
                (2 ** 31, 0, None, 0), (44, signal.SIGUSR1, None, 0), (44, True, None, 0),
                (44, 0, {}, 0), (44, 0, None, 1), (44, 0, None, False)):
            with self.subTest(descriptor=descriptor, signum=signum, info=info, flags=flags), \
                    self.assertRaises(ValueError):
                adapter.send_signal(descriptor, signum, info, flags)
        self.opened.assert_not_called(); self.sent.assert_not_called(); self.library.assert_not_called()


if __name__ == "__main__":
    unittest.main()
