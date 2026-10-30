"""One cooperating sfhack10 dashboard per local user, across all worktrees.

This advisory OS lock does not detect older clients, other users, sfhack14, or
remote controllers. It is not evidence that the robot is stationary or safe.
No SDK, network, actuator or service is accessed here. The pathname is fixed;
ordinary launch flags cannot select another lease and bypass ownership.
"""
from __future__ import annotations

from datetime import datetime, timezone
import errno
import fcntl
import json
import os
from pathlib import Path
import stat


MACHINE = 'sfhack10'
LOCK_NAME = 'sfhack10-controller.lock'


def lease_path():
    return Path.home() / '.local' / 'state' / 'sircus' / LOCK_NAME


def _check_directory(fd, *, private=False):
    info = os.fstat(fd)
    mode = stat.S_IMODE(info.st_mode)
    if (not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid()
            or mode & 0o022 or (private and mode != 0o700)):
        raise RuntimeError('Unsafe sfhack10 lease directory ownership or permissions')


def _open_directory():
    """Traverse verified owner-controlled directories using no-follow handles."""
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
    fd = os.open(Path.home(), flags)
    try:
        _check_directory(fd)
        for name in ('.local', 'state', 'sircus'):
            try:
                os.mkdir(name, mode=0o700, dir_fd=fd)
            except FileExistsError:
                pass
            child = os.open(name, flags, dir_fd=fd)
            try:
                _check_directory(child, private=name == 'sircus')
            except BaseException:
                os.close(child)
                raise
            os.close(fd)
            fd = child
        return fd
    except BaseException:
        os.close(fd)
        raise


class HardwareLease:
    def __init__(self, fd):
        self._fd = fd

    def close(self):
        fd, self._fd = self._fd, None
        if fd is not None:
            try:
                fcntl.flock(fd, fcntl.LOCK_UN)
            finally:
                os.close(fd)
        # Never unlink: removing a locked inode lets another process lock a new
        # file at the same pathname while the original owner still exists.

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


def acquire_sfhack10_lease():
    directory = _open_directory()
    fd = None
    try:
        flags = os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW | os.O_CLOEXEC | os.O_NONBLOCK
        fd = os.open(LOCK_NAME, flags, 0o600, dir_fd=directory)
        info = os.fstat(fd)
        if (not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid()
                or stat.S_IMODE(info.st_mode) != 0o600 or info.st_nlink != 1):
            raise RuntimeError('sfhack10 lease must be an owner-only regular file (0600), without links')
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            if exc.errno in (errno.EACCES, errno.EAGAIN):
                raise RuntimeError('Another local sfhack10 dashboard owns the hardware lease; do not start a second controller') from None
            raise
        # Only the OS lock establishes ownership; stale PID text never does.
        value = {'schema': 'sircus-hardware-lease-v1', 'machine': MACHINE,
                 'pid': os.getpid(), 'acquired_at': datetime.now(timezone.utc).isoformat(),
                 'scope': 'cooperating dashboards on this host/user only; remote ownership unknown'}
        raw = (json.dumps(value, sort_keys=True) + '\n').encode()
        os.ftruncate(fd, 0)
        os.lseek(fd, 0, os.SEEK_SET)
        written = 0
        while written < len(raw):
            written += os.write(fd, raw[written:])
        os.fsync(fd)
        lease, fd = HardwareLease(fd), None
        return lease
    finally:
        os.close(directory)
        if fd is not None:
            os.close(fd)
