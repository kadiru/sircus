"""Control ordering regressions with a silent in-memory backend; no server needed."""
import sys
import threading
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from bridge import Controller


class Backend:
    def __init__(self): self.commands = []
    def send(self, **command): self.commands.append(command)


class ControlOrder(unittest.TestCase):
    def setUp(self):
        self.backend = Backend()
        self.controller = Controller(self.backend)
        self.controller.ready = True

    def intent(self, sequence, enabled=None, client='panel'):
        body = {'control_client_id': client, 'control_sequence': sequence}
        if enabled is not None:
            body['enabled'] = enabled
            if enabled: body['control_epoch'] = self.controller.status()['control_epoch']
        return body

    def test_delayed_enable_cannot_undo_off_or_panic(self):
        for kind in ('fx', 'panic'):
            with self.subTest(kind=kind):
                self.setUp()
                old_on = self.intent(1, True)
                gate = threading.Event()
                result = []
                def deliver_old_on():
                    gate.wait()
                    result.append(self.controller.settings('fx', old_on))
                worker = threading.Thread(target=deliver_old_on)
                worker.start()
                try:
                    off = self.intent(2, False if kind == 'fx' else None)
                    self.assertFalse(self.controller.settings(kind, off)['fx_enabled'])
                finally:
                    gate.set()
                    worker.join(timeout=1)
                self.assertFalse(worker.is_alive())
                self.assertFalse(result[0]['accepted'])
                self.assertFalse(self.controller.enabled)
                self.assertNotIn({'cmd': 'enable', 'value': True}, self.backend.commands)

    def test_mute_from_another_client_invalidates_observed_enable_epoch(self):
        old_on = self.intent(1, True, client='dashboard')
        self.controller.settings('panic', self.intent(1, client='panel'))
        with self.assertRaisesRegex(ValueError, 'stale audio enable'):
            self.controller.settings('fx', old_on)
        self.assertFalse(self.controller.enabled)
        # Only a fresh, explicit command with current status can enable again.
        self.assertTrue(self.controller.settings('fx', self.intent(2, True, client='dashboard'))['fx_enabled'])

    def test_duplicate_control_never_reexecutes_and_reused_sequence_rejected(self):
        off = self.intent(2, False)
        self.controller.settings('fx', off)
        commands = list(self.backend.commands)
        epoch = self.controller.control_epoch
        self.assertFalse(self.controller.settings('fx', off)['accepted'])
        self.assertEqual(commands, self.backend.commands)
        self.assertEqual(epoch, self.controller.control_epoch)
        with self.assertRaisesRegex(ValueError, 'reused'):
            self.controller.settings('fx', self.intent(2, True))

    def test_legacy_enable_rejected_but_legacy_mute_always_works(self):
        with self.assertRaisesRegex(ValueError, 'ordered FX enable'):
            self.controller.settings('fx', {'enabled': True})
        for kind, body in [('fx', {'enabled': False}), ('panic', {})]:
            self.controller.enable(True)
            epoch = self.controller.control_epoch
            self.assertFalse(self.controller.settings(kind, body)['fx_enabled'])
            self.assertNotEqual(epoch, self.controller.control_epoch)

    def test_validation_rejects_invalid_control_ids_sequences_and_epochs(self):
        for overrides in ({'control_sequence': True}, {'control_sequence': 0},
                          {'control_sequence': 2**53}, {'control_client_id': 'bad id'},
                          {'control_client_id': 'x'*65}, {'control_epoch': {}},
                          {'control_epoch': 'x'*65}):
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                self.controller.settings('fx', {**self.intent(1, True), **overrides})
        self.assertFalse(self.controller.enabled)
        self.assertEqual(self.controller.control_clients, {})

    def test_capacity_preserves_history_rejects_enable_but_allows_mute(self):
        old_on = self.intent(1, True, client='first')
        self.controller.settings('fx', self.intent(2, False, client='first'))
        for index in range(127):
            self.controller.settings('fx', self.intent(1, False, client=f'panel-{index}'))
        self.assertEqual(len(self.controller.control_clients), 128)
        with self.assertRaisesRegex(ValueError, 'capacity'):
            self.controller.settings('fx', self.intent(1, True, client='new'))
        self.assertFalse(self.controller.settings('panic', self.intent(1, client='new'))['fx_enabled'])
        self.assertFalse(self.controller.settings('fx', old_on)['accepted'])
        self.assertEqual(len(self.controller.control_clients), 128)

    def test_native_mute_and_restart_invalidate_previously_observed_epoch(self):
        for message in ({'type': 'connection', 'connected': False},
                        {'type': 'connection', 'connected': True},
                        {'type': 'watchdog'}, {'type': 'error', 'message': 'mock'},
                        {'type': 'native_exit', 'message': 'mock'}):
            with self.subTest(message=message):
                self.setUp()
                old_on = self.intent(1, True)
                self.controller.native(message)
                with self.assertRaisesRegex(ValueError, 'stale audio enable'):
                    self.controller.settings('fx', old_on)
                self.assertFalse(self.controller.enabled)
        replacement = Controller(Backend())
        self.assertNotEqual(self.controller.control_epoch, replacement.control_epoch)


if __name__ == '__main__': unittest.main()
