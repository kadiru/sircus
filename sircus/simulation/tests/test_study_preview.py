"""No sockets/SDK connections: exercise the actual request handler in memory."""
import io
import json
from pathlib import Path
import subprocess
import sys
import types
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import server


class StudyPreviewHTTPTests(unittest.TestCase):
    def request(self, payload, *, path='/api/study/preview', host='127.0.0.1:18769',
                origin='http://127.0.0.1:18769', content_type='application/json', raw=None, length=None):
        body = raw if raw is not None else json.dumps(payload).encode()
        handler = server.Handler.__new__(server.Handler)
        handler.path = path
        handler.server = types.SimpleNamespace(server_address=('127.0.0.1', 18769))
        handler.headers = {'Host': host, 'Origin': origin, 'Content-Type': content_type,
                           'Content-Length': str(len(body) if length is None else length)}
        handler.rfile = io.BytesIO(body)
        handler.send = Mock()
        handler.do_POST()
        args = handler.send.call_args.args
        return args[0], args[1] if len(args) > 1 else 200

    def test_numerical_route_cannot_touch_native_or_physical_bridge(self):
        native, physical = Mock(), Mock()
        with patch.object(server, 'BRIDGE', native), patch.object(server, 'PHYSICAL', physical), \
             patch.object(server, 'OFFLINE', False), patch.object(server, 'study_preview', return_value={'synthetic': True}) as run:
            # Legacy simulation lock does not gate this independent numerical route.
            server.LOCK.acquire()
            try:
                result, code = self.request({'checkpoint': {}, 'scene': {}, 'seed': 0})
            finally:
                server.LOCK.release()
            self.assertEqual(code, 200)
            self.assertTrue(result['synthetic'])
            run.assert_called_once()
            self.assertEqual(native.mock_calls, [])
            self.assertEqual(physical.mock_calls, [])

    def test_host_origin_content_and_size_rejected_before_compute(self):
        with patch.object(server, 'study_preview') as run:
            for kwargs, code in [({'host': 'evil.example:18769'}, 403),
                                 ({'origin': 'http://evil.example'}, 403),
                                 ({'content_type': 'text/plain'}, 415),
                                 ({'length': 1_000_001}, 413),
                                 ({'path': '/api/study/native'}, 404)]:
                self.assertEqual(self.request({}, **kwargs)[1], code)
            run.assert_not_called()

    def test_nan_and_concurrent_request_rejected(self):
        with patch.object(server, 'study_preview') as run:
            self.assertEqual(self.request({}, raw=b'{"seed":NaN}')[1], 400)
            server.STUDY_LOCK.acquire()
            try:
                self.assertEqual(self.request({})[1], 409)
            finally:
                server.STUDY_LOCK.release()
            run.assert_not_called()

    def test_worker_path_fixed_and_json_values_only(self):
        payload = {'checkpoint': {'schema': 'test'}, 'scene': {}, 'seed': 2**32-1}
        response = types.SimpleNamespace(returncode=0, stdout='{"data_origin":"synthetic"}', stderr='')
        with patch.object(server.subprocess, 'run', return_value=response) as run:
            self.assertEqual(server.study_preview(payload)['data_origin'], 'synthetic')
            args, kwargs = run.call_args
            self.assertEqual(args[0], [sys.executable, str(ROOT.parent/'two-arm'/'study_preview.py')])
            self.assertEqual(json.loads(kwargs['input']), payload)
            self.assertEqual(kwargs['timeout'], 35)
            for bad in ({**payload, 'path': '/private/model.py'}, {**payload, 'checkpoint': '/tmp/model'},
                        {**payload, 'seed': True}, {**payload, 'seed': -1}, {**payload, 'seed': 2**32}):
                with self.assertRaises(ValueError):
                    server.study_preview(bad)
            self.assertEqual(run.call_count, 1)

    def test_corrupt_checkpoint_and_timeout_fail_without_fallback(self):
        payload = {'checkpoint': {}, 'scene': {}, 'seed': 1}
        response = types.SimpleNamespace(returncode=2, stdout='{"error":"Checkpoint fingerprint mismatch"}', stderr='')
        with patch.object(server.subprocess, 'run', return_value=response):
            value, code = self.request(payload)
            self.assertEqual(code, 400)
            self.assertIn('fingerprint mismatch', value['error'])
        with patch.object(server.subprocess, 'run', side_effect=subprocess.TimeoutExpired('worker', 35)):
            value, code = self.request(payload)
            self.assertEqual(code, 400)
            self.assertIn('no runtime action', value['error'])


if __name__ == '__main__':
    unittest.main()
