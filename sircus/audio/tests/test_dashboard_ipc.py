"""Python → Node supplied client → disposable loopback fake; no sound/hardware."""
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
import shutil
import threading
import time
import unittest

from sircus.audio.dashboard import AudioObserver


class DashboardIPCTests(unittest.TestCase):
    @unittest.skipUnless(shutil.which('node'), 'Node required for dashboard worker IPC')
    def test_real_worker_preserves_evidence_and_cancels_against_fake_service(self):
        calls, heard, retired, seen = [], [], set(), set()
        state = {'fx_enabled': False, 'control_epoch': 'TEST_ONLY_EPOCH'}

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_):
                pass

            def reply(self, data):
                body = json.dumps(data).encode()
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Content-Length', str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                # No MIDI hardware is connected; HTTP reachability still succeeds.
                self.reply({'connected': False, 'audio_ready': True, **state})

            def do_POST(self):
                if self.headers.get('Authorization') != 'Bearer TEST_ONLY_IPC':
                    self.send_error(401)
                    return
                data = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                calls.append((self.path, data))
                accepted = True
                if self.path == '/api/fx':
                    state['fx_enabled'] = data['enabled']
                elif self.path == '/api/stop':
                    retired.add(data['trial_id'])
                elif self.path == '/api/event':
                    key = (data['trial_id'], data['event_id'])
                    accepted = state['fx_enabled'] and key not in seen and data['trial_id'] not in retired
                    if accepted:
                        heard.append(data)
                        seen.add(key)
                self.reply({'accepted': accepted, 'reason': None if accepted else 'duplicate_or_retired'})

        fake = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        fake.daemon_threads = True
        thread = threading.Thread(target=fake.serve_forever, kwargs={'poll_interval': .01}, daemon=True)
        thread.start()
        observer = AudioObserver(environ={
            'PATH': os.environ.get('PATH', ''), 'SIRCUS_AUDIO_NODE': shutil.which('node'),
            'SIRCUS_AUDIO_URL': f'http://127.0.0.1:{fake.server_port}',
            'SIRCUS_AUDIO_TOKEN': 'TEST_ONLY_IPC'})

        def wait_for(predicate):
            until = time.monotonic()+5
            while time.monotonic() < until:
                if predicate():
                    return
                time.sleep(.01)
            self.fail('Mock IPC condition did not complete within five seconds')

        try:
            wait_for(lambda: observer.snapshot()['connection'] == 'connected')
            self.assertFalse(observer.snapshot()['enabled'])
            self.assertEqual(calls, [])  # Startup performs a read, never an implicit enable/mute.
            observer.control({'action': 'fx', 'enabled': True, 'client_id': 'ipc', 'sequence': 1})
            wait_for(lambda: observer.snapshot()['enabled'])
            event = dict(event_id='release-1', trial_id='trial-1', source='operator', event_type='release',
                         contact_object='unknown', timestamp=12.25, time_basis='host_monotonic_s',
                         clock_id='mock-capture-clock', evidence='operator_action',
                         provenance='Explicit fake command; not observed physical contact')
            before = time.monotonic()
            observer.emit(event); observer.emit(event)
            self.assertLess(time.monotonic()-before, .1)
            wait_for(lambda: sum(path == '/api/event' for path, _ in calls) == 2)
            self.assertEqual(len(heard), 1)
            self.assertEqual(heard[0]['timestamp'], 12.25)
            self.assertEqual(heard[0]['provenance'], event['provenance'])
            self.assertEqual(heard[0]['event_id'], 'release-1')
            self.assertFalse(heard[0]['confirmed'])
            observer.stop('trial-1')
            wait_for(lambda: 'trial-1' in retired)
            observer.emit({**event, 'event_id': 'late'})
            self.assertFalse(any(row.get('event', {}).get('event_id') == 'late' for row in observer._queue))
            # Mini-local OFF (or a restarted service) must clear pending dashboard
            # intent all the way through the real Node worker and Python observer.
            state['fx_enabled'] = False
            wait_for(lambda: not observer.snapshot()['requested_enabled'])
            self.assertFalse(observer.snapshot()['enabled'])
            self.assertFalse(observer.snapshot()['pending'])
            observer.control({'action': 'fx', 'enabled': False, 'client_id': 'ipc', 'sequence': 2})
            wait_for(lambda: observer.snapshot()['mute_confirmed'] is True)
            self.assertFalse(state['fx_enabled'])
            control_count = sum(path == '/api/fx' for path, _ in calls)
            state['fx_enabled'] = True  # Ghost Booth operator enabled locally.
            wait_for(lambda: observer.snapshot()['mute_confirmed'] is False)
            self.assertFalse(observer.snapshot()['requested_enabled'])
            self.assertFalse(observer.snapshot()['enabled'])
            self.assertIn('unconfirmed', observer.snapshot()['error'])
            self.assertEqual(sum(path == '/api/fx' for path, _ in calls), control_count)
            public = json.dumps(observer.snapshot())
            self.assertNotIn('TEST_ONLY_IPC', public)
            self.assertNotIn('127.0.0.1', public)
        finally:
            observer.close()
            if observer._process is not None:
                observer._process.wait(timeout=3)
                observer._process.stdin.close()
                observer._process.stdout.close()
            fake.shutdown()
            fake.server_close()
            thread.join(timeout=2)


if __name__ == '__main__':
    unittest.main()
