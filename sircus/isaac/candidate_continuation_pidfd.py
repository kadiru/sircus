"""Inert Linux x86_64 pidfds for the bounded watchdog, without PID signalling.

Prefer CPython's pidfd APIs when present. Some bundled Python builds omit those
wrappers even when the Linux kernel implements them; only that case uses the
same kernel operations through libc.syscall. No process is opened or signalled
at import or adapter construction. There is no kill/process-group fallback.
"""

import os
import platform
import signal
import sys

# Primary local ABI source, verified on the Ubuntu simulation host:
# /usr/include/x86_64-linux-gnu/asm/unistd_64.h
#   #define __NR_pidfd_send_signal 424
#   #define __NR_pidfd_open 434
# These numbers are admitted only for Linux x86_64, never another architecture.
PIDFD_SEND_SIGNAL_SYSCALL = 424
PIDFD_OPEN_SYSCALL = 434
SCHEMA = "sircus-linux-x86_64-pidfd-capability-v1"


def _require(value, message):
    if not value:
        raise ValueError(message)


def _integer(value, minimum, message):
    _require(isinstance(value, int) and not isinstance(value, bool)
        and minimum <= value <= 2 ** 31 - 1, message)


class PidfdAdapter:
    """One API selection; lazy libc loading, pinned descriptors only."""

    def __init__(self):
        _require(sys.platform == "linux" and platform.machine() == "x86_64"
            and sys.maxsize == 2 ** 63 - 1,
            "Production pidfds require the reviewed 64-bit Linux x86_64 ABI")
        native_open = getattr(os, "pidfd_open", None)
        native_signal = getattr(signal, "pidfd_send_signal", None)
        self._native_open = native_open if callable(native_open) else None
        self._native_signal = native_signal if callable(native_signal) else None
        self._syscall = None

    @property
    def backends(self):
        return dict(pidfd_open="cpython" if self._native_open is not None else "linux-x86_64-syscall",
            pidfd_send_signal="cpython" if self._native_signal is not None else "linux-x86_64-syscall")

    def _call(self, number, *arguments):
        import ctypes
        if self._syscall is None:
            library = ctypes.CDLL(None, use_errno=True)
            self._syscall = library.syscall
            # syscall is variadic: every argument below has an explicit ABI type.
            self._syscall.restype = ctypes.c_long
        ctypes.set_errno(0)
        result = self._syscall(ctypes.c_long(number), *arguments)
        if result == -1:
            code = ctypes.get_errno()
            raise OSError(code, os.strerror(code))
        return result

    def open(self, pid, flags=0):
        _integer(pid, 1, "Expected a positive Linux PID")
        _require(type(flags) is int and flags == 0, "Only original pidfd flags=0 are admitted")
        if self._native_open is not None:
            return self._native_open(pid, flags)
        import ctypes
        return self._call(PIDFD_OPEN_SYSCALL, ctypes.c_int(pid), ctypes.c_uint(flags))

    def send_signal(self, descriptor, signum, info=None, flags=0):
        _integer(descriptor, 0, "Expected an open pidfd descriptor")
        _require(isinstance(signum, int) and not isinstance(signum, bool)
            and signum in (0, signal.SIGTERM, signal.SIGKILL),
            "Only self-probe zero or watchdog TERM/KILL signals are admitted")
        _require(info is None and type(flags) is int and flags == 0,
            "Only original NULL siginfo and pidfd flags=0 are admitted")
        if self._native_signal is not None:
            return self._native_signal(descriptor, signum, info, flags)
        import ctypes
        self._call(PIDFD_SEND_SIGNAL_SYSCALL, ctypes.c_int(descriptor), ctypes.c_int(signum),
            ctypes.c_void_p(), ctypes.c_uint(flags))
        return None

    def capability_preflight(self):
        """Open only this process, send only signal 0, and always close the fd.

        Explicit callers may use this before any application start. Signal zero
        checks the kernel operation without delivering a termination signal.
        Errors fail closed; this does not authorize starting or retrying a job.
        """
        descriptor = self.open(os.getpid(), 0)
        try:
            self.send_signal(descriptor, 0, None, 0)
        finally:
            os.close(descriptor)
        return dict(schema=SCHEMA, status="passed", platform="linux", machine="x86_64",
            backends=self.backends, self_only=True, signal=0, descriptor_closed=True,
            process_starts=0, execution_authorized=False)


def pidfd_open(pid, flags=0):
    return PidfdAdapter().open(pid, flags)


def pidfd_send_signal(descriptor, signum, info=None, flags=0):
    return PidfdAdapter().send_signal(descriptor, signum, info, flags)


def capability_preflight():
    return PidfdAdapter().capability_preflight()
