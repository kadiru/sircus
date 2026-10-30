"""Audio HTTP contract checks without sockets, workers, sound, or robot clients."""
import io
import json
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent.parent))
import server
from sircus.audio.dashboard import AudioObserver


class AudioHTTPTests(unittest.TestCase):
    def setUp(self):
        self.sequence = 0
        self.audio = AudioObserver(environ={
            'SIRCUS_AUDIO_URL': 'http://192.168.1.50:8766',
            'SIRCUS_AUDIO_TOKEN': 'private-mini-credential'}, start_worker=False)
        self.native, self.physical = Mock(), Mock()
        self.patches = [patch.object(server, 'AUDIO', self.audio),
                        patch.object(server, 'BRIDGE', self.native),
                        patch.object(server, 'PHYSICAL', self.physical),
                        patch.object(server, 'OFFLINE', False)]
        for change in self.patches:
            change.start()

    def tearDown(self):
        self.assertEqual(self.native.mock_calls, [])
        self.assertEqual(self.physical.mock_calls, [])
        self.audio.close()
        for change in reversed(self.patches):
            change.stop()

    def request(self, *, method='POST', path='/api/audio/control', payload=None,
                host='127.0.0.1:18769', origin='http://127.0.0.1:18769',
                token=None, content_type='application/json', raw=None, length=None):
        data = {'action': 'fx', 'enabled': False} if payload is None else payload
        if isinstance(data, dict):
            self.sequence += 1
            data = {'client_id': 'http-test-session', 'sequence': self.sequence, **data}
        body = raw if raw is not None else json.dumps(data).encode()
        handler = server.Handler.__new__(server.Handler)
        handler.path = path
        handler.server = types.SimpleNamespace(server_address=('127.0.0.1', 18769))
        handler.headers = {'Host': host, 'Origin': origin, 'Content-Type': content_type,
                           'Content-Length': str(len(body) if length is None else length),
                           'X-Sircus-Audio-Token': self.audio.token if token is None else token}
        handler.rfile = io.BytesIO(body)
        handler.send = Mock()
        (handler.do_GET if method == 'GET' else handler.do_POST)()
        args = handler.send.call_args.args
        return args[0], args[1] if len(args) > 1 else 200

    def test_status_is_default_off_and_never_exposes_mini_credentials(self):
        result, status = self.request(method='GET', path='/api/audio/status')
        self.assertEqual(status, 200)
        self.assertFalse(result['enabled'])
        self.assertFalse(result['requested_enabled'])
        self.assertEqual(result['action_token'], self.audio.token)
        wire = json.dumps(result)
        for secret in ('private-mini-credential', '192.168.1.50', 'SIRCUS_AUDIO_URL', 'SIRCUS_AUDIO_TOKEN'):
            self.assertNotIn(secret, wire)
        self.assertNotEqual(result['action_token'], 'private-mini-credential')

    def test_audio_controls_do_not_wait_for_native_lock_or_touch_robots(self):
        server.LOCK.acquire()
        try:
            enabled, status = self.request(payload={'action': 'fx', 'enabled': True})
            self.assertEqual(status, 202)
            self.assertTrue(enabled['requested_enabled'])
            self.assertFalse(enabled['enabled'])  # Never assume acknowledgment.
            self.assertTrue(enabled['pending'])
            muted, status = self.request(payload={'action': 'fx', 'enabled': False})
            self.assertEqual(status, 202)
            self.assertFalse(muted['enabled'])
            self.assertFalse(muted['requested_enabled'])
            volume, status = self.request(payload={'action': 'volume', 'volume': .2})
            self.assertEqual(status, 202)
            self.assertEqual(volume['volume'], .2)
        finally:
            server.LOCK.release()
        self.assertIsNone(self.audio._process)

    def test_control_host_origin_token_size_and_endpoint_allowlist(self):
        invalid = [({'host': 'evil.example:18769'}, 403),
                   ({'host': '127.0.0.1:8765'}, 403),
                   ({'origin': 'http://evil.example'}, 403),
                   ({'token': ''}, 403), ({'token': 'private-mini-credential'}, 403),
                   ({'content_type': 'text/plain'}, 415),
                   ({'length': 4097}, 413), ({'length': -1}, 413),
                   ({'length': 'invalid'}, 413),
                   ({'path': '/api/audio/event'}, 404),
                   ({'path': '/api/audio/physical/action'}, 404)]
        for kwargs, expected in invalid:
            with self.subTest(kwargs=kwargs):
                self.assertEqual(self.request(**kwargs)[1], expected)
        self.assertEqual(len(self.audio._queue), 0)
        self.assertEqual(self.request(method='GET', path='/api/audio/status', host='example.com')[1], 403)
        self.assertEqual(self.request(method='GET', path='/api/audio/control')[1], 404)

    def test_only_fx_boolean_or_bounded_volume_are_accepted(self):
        invalid = [[], 'path/to/audio.py', {'action': 'stop', 'trial_id': 'trial'},
                   {'action': 'fx', 'enabled': 1},
                   {'action': 'fx', 'enabled': True, 'command': 'open_gripper'},
                   {'action': 'volume', 'volume': True},
                   {'action': 'volume', 'volume': -1},
                   {'action': 'volume', 'volume': 1.1},
                   {'action': 'volume', 'volume': float('nan')},
                   {'action': 'volume', 'volume': float('inf')}]
        for payload in invalid:
            with self.subTest(payload=payload):
                self.assertEqual(self.request(payload=payload)[1], 400)
        self.assertEqual(self.request(raw=b'{invalid json')[1], 400)
        self.assertEqual(len(self.audio._queue), 0)

    def test_controls_require_ordering_metadata_and_retire_delayed_on(self):
        self.assertEqual(self.request(raw=b'{"action":"fx","enabled":true}')[1], 400)
        for invalid in ({'client_id': ''}, {'client_id': 'x'*65}, {'client_id': 'bad client'},
                        {'sequence': 0}, {'sequence': True}, {'sequence': 2**53}):
            self.assertEqual(self.request(payload={'action': 'fx', 'enabled': True, **invalid})[1], 400)
        off = {'action': 'fx', 'enabled': False, 'client_id': 'out-of-order', 'sequence': 2}
        self.assertEqual(self.request(payload=off)[1], 202)
        queued = len(self.audio._queue)
        # The earlier ON packet arrives last and cannot revive playback.
        state, code = self.request(payload={**off, 'enabled': True, 'sequence': 1})
        self.assertEqual(code, 202)
        self.assertFalse(state['requested_enabled'])
        self.assertFalse(state['enabled'])
        self.assertEqual(len(self.audio._queue), queued)
        self.assertEqual(self.request(payload=off)[1], 202)
        self.assertEqual(len(self.audio._queue), queued)
        self.assertEqual(self.request(payload={**off, 'enabled': True})[1], 400)

    def test_audio_initialization_failure_does_not_escape_into_robot_startup(self):
        with patch('sircus.audio.dashboard.AudioObserver', side_effect=RuntimeError('private diagnostic')):
            self.assertIsNone(server.optional_audio_observer())

    def test_unavailable_backend_and_errors_are_generic(self):
        with patch.object(server, 'AUDIO', None):
            self.assertEqual(self.request()[1], 403)
            state, code = self.request(method='GET', path='/api/audio/status')
            self.assertEqual(code, 503)
            self.assertFalse(state['enabled'])
        with patch.object(self.audio, 'control', side_effect=RuntimeError('private-mini-credential')):
            error, code = self.request()
            self.assertEqual(code, 503)
            self.assertNotIn('private-mini-credential', json.dumps(error))
        self.audio._accept_status({'revision': 0, 'status': {'enabled': False, 'pending': False,
            'connection': 'error', 'error': 'private-mini-credential',
            'url': 'http://192.168.1.50:8766', 'token': 'private-mini-credential'}})
        state, code = self.request(method='GET', path='/api/audio/status')
        self.assertEqual(code, 200)
        self.assertNotIn('private-mini-credential', json.dumps(state))
        self.assertNotIn('192.168.1.50', json.dumps(state))


if __name__ == '__main__':
    unittest.main()
