"""Filesystem-only hardware ownership tests, always inside temporary homes."""
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from sircus.physical_trials import ownership


class HardwareOwnershipTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.home = Path(self.temp.name)
        self.patch = patch.object(ownership.Path, 'home', return_value=self.home)
        self.patch.start()
        self.leases = []

    def tearDown(self):
        for lease in self.leases:
            lease.close()
        self.patch.stop()
        self.temp.cleanup()

    def acquire(self):
        lease = ownership.acquire_sfhack10_lease()
        self.leases.append(lease)
        return lease

    def test_shared_path_and_owner_only_file_survive_clean_release(self):
        first = self.acquire()
        path = ownership.lease_path()
        self.assertEqual(path, self.home/'.local/state/sircus/sfhack10-controller.lock')
        self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o600)
        self.assertEqual(stat.S_IMODE(path.parent.stat().st_mode), 0o700)
        metadata = json.loads(path.read_text())
        self.assertEqual(metadata['machine'], 'sfhack10')
        self.assertEqual(metadata['pid'], os.getpid())
        inode = path.stat().st_ino
        first.close();first.close()
        self.acquire()
        self.assertEqual(path.stat().st_ino, inode)

    def test_second_instance_cannot_truncate_or_steal_active_lease(self):
        self.acquire()
        before = ownership.lease_path().read_bytes()
        with self.assertRaisesRegex(RuntimeError, 'Another local sfhack10'):
            self.acquire()
        self.assertEqual(ownership.lease_path().read_bytes(), before)

    def test_another_process_is_blocked_then_stale_pid_is_not_an_owner(self):
        lease = self.acquire()
        script = '''from pathlib import Path
from unittest.mock import patch
from sircus.physical_trials.ownership import acquire_sfhack10_lease
import sys
with patch('sircus.physical_trials.ownership.Path.home', return_value=Path(sys.argv[1])):
 try:
  lease=acquire_sfhack10_lease()
 except RuntimeError:
  print('BUSY')
 else:
  print('ACQUIRED')
'''
        def child():
            return subprocess.check_output([sys.executable, '-c', script, str(self.home)],
                                           cwd=Path(__file__).resolve().parents[2], text=True).strip()
        self.assertEqual(child(), 'BUSY')
        lease.close()
        self.assertEqual(child(), 'ACQUIRED')
        # The child exited without unlinking or explicitly releasing its lock.
        self.acquire()

    def test_reject_symlink_directory_and_group_writable_parent(self):
        local = self.home/'.local'
        local.mkdir(mode=0o700)
        target = self.home/'target';target.mkdir(mode=0o700)
        (local/'state').symlink_to(target, target_is_directory=True)
        with self.assertRaises(OSError):
            self.acquire()
        (local/'state').unlink()
        local.chmod(0o777)
        with self.assertRaisesRegex(RuntimeError, 'directory ownership or permissions'):
            self.acquire()
        local.chmod(0o700)

    def test_reject_symlink_hardlink_nonregular_and_loose_file_permissions(self):
        lease = self.acquire();lease.close()
        path = ownership.lease_path();path.unlink()
        target = self.home/'target';target.write_text('do not modify');target.chmod(0o600)
        path.symlink_to(target)
        with self.assertRaises(OSError):self.acquire()
        self.assertEqual(target.read_text(), 'do not modify')
        path.unlink();os.link(target,path)
        with self.assertRaisesRegex(RuntimeError, 'owner-only regular file'):self.acquire()
        path.unlink();os.mkfifo(path,mode=0o600)
        with self.assertRaisesRegex(RuntimeError, 'owner-only regular file'):self.acquire()
        path.unlink();path.write_text('unchanged');path.chmod(0o644)
        with self.assertRaisesRegex(RuntimeError, 'owner-only regular file'):self.acquire()
        self.assertEqual(path.read_text(), 'unchanged')

    def test_wrong_owner_and_failed_write_do_not_leave_a_held_lock(self):
        with patch.object(ownership.os, 'getuid', return_value=os.getuid()+1):
            with self.assertRaisesRegex(RuntimeError, 'directory ownership or permissions'):
                self.acquire()
        with patch.object(ownership.os, 'fsync', side_effect=OSError('disk failure')):
            with self.assertRaisesRegex(OSError, 'disk failure'):
                self.acquire()
        self.acquire()


if __name__ == '__main__':
    unittest.main()
