"""Loopback dashboard bridge to the isolated, cloud-bound Sircus Viam machine.

The default backend executes commands on Viam. --offline explicitly selects the
old local simulator; a Viam failure never falls back to local execution.
"""
import argparse
import asyncio
import json
import mimetypes
import os
import secrets
import sys
import threading
from concurrent.futures import TimeoutError as FutureTimeout
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit
from uuid import UUID

from physics import default_config, validate_config, simulate
from kinematics import home, sequence
from learning import train

ROOT = Path(__file__).resolve().parent
PART_ID = '28a499f1-b207-4111-90bb-105fefa5a840'
MACHINE_ID = '5db5a9df-2e56-4711-9d39-b42cc86e4d79'
MACHINE_URL = f'https://app.viam.com/machine/{MACHINE_ID}/3d-scene'
LOCK = threading.Lock()
STATE = {'phase': 'idle'}
OFFLINE = False
BRIDGE = None
PHYSICAL = None


class PhysicalBridge:
    """Own event loop and credentials, separate from the simulation bridge."""
    def __init__(self, mode, credentials, root):
        sys.path.insert(0, str(ROOT.parent))
        from physical_trials.controller import Controller
        if mode == 'hardware':
            from physical_trials.viam_adapter import ViamAdapter
            adapter = ViamAdapter(credentials)
        else:
            from physical_trials.mock_adapter import MockAdapter
            adapter = MockAdapter()
        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(target=self.loop.run_forever, daemon=True)
        self.thread.start()
        self.token = secrets.token_urlsafe(32)
        self.mode = mode
        self.controller = Controller(adapter, root, mode=mode)
        self.startup = asyncio.run_coroutine_threadsafe(self.controller.start(), self.loop)

    def call(self, kind, payload=None):
        if kind == 'status':
            # Read cached state outside the SDK loop: a stalled transport must
            # not hide disconnected state, calibration gaps or the Stop button.
            result = self.controller.status_snapshot()
            if not result['connected'] and not result.get('error'):
                result['error'] = 'sfhack10 connection/readback unavailable; movement and release disabled'
            return {**result, 'action_token': self.token}
        elif kind == 'camera':
            call, timeout = self.controller.adapter.camera_frame(), 8
        else:
            call, timeout = self.controller.action(payload), 30
        future = asyncio.run_coroutine_threadsafe(call, self.loop)
        try:
            result = future.result(timeout)
            return {**result, 'action_token': self.token} if isinstance(result, dict) else result
        except FutureTimeout:
            future.cancel()
            raise RuntimeError('Physical service timed out; no command retry was issued') from None


class ViamBridge:
    def __init__(self):
        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(target=self.loop.run_forever, daemon=True)
        self.thread.start()
        self.robot = None
        self.controller = None

    def credentials(self):
        cache = Path.home() / '.viam' / f'cached_cloud_config_{PART_ID}.json'
        info = cache.stat()
        if cache.is_symlink() or info.st_uid != os.getuid() or info.st_mode & 0o022:
            raise RuntimeError('Unexpected simulation configuration permissions')
        config = json.loads(cache.read_text())
        if config.get('cloud', {}).get('id') != PART_ID or config.get('remotes'):
            raise RuntimeError('Dashboard is bound only to the isolated Sircus simulation')
        resources = {item['name']: item for item in config.get('components', [])}
        for item in resources.values():
            if item.get('api') in ('rdk:component:arm', 'rdk:component:gripper') and item.get('model') != 'rdk:builtin:fake':
                raise RuntimeError('Dashboard refuses physical arm or gripper drivers')
        for name in ('arm-10', 'arm-14'):
            item = resources.get(name, {})
            if item.get('api') != 'rdk:component:arm' or item.get('model') != 'rdk:builtin:fake':
                raise RuntimeError('Two-arm Viam simulation is not ready: expected fake arm-10 and arm-14')
        if resources.get('sircus', {}).get('model') != 'local:sircus:controller':
            raise RuntimeError('Expected the Sircus simulation controller')
        keys = []
        for handler in config.get('auth', {}).get('handlers', []):
            if handler.get('type') != 'api-key':
                continue
            for key_id, secret in handler.get('config', {}).items():
                try:
                    UUID(key_id)
                except (ValueError, TypeError):
                    continue
                if isinstance(secret, str) and secret:
                    keys.append((key_id, secret))
        if len(keys) != 1:
            raise RuntimeError('Expected one existing simulation machine credential')
        return keys[0]

    async def invoke(self, command):
        # This check also runs after a connection has been established. A later
        # cloud configuration change cannot redirect dashboard commands to an arm.
        key_id, secret = self.credentials()
        if self.robot is None:
            from viam.components.generic import Generic
            from viam.robot.client import RobotClient
            from viam.rpc.dial import Credentials, DialOptions
            self.robot = await asyncio.wait_for(RobotClient.at_address('127.0.0.1:8082', RobotClient.Options(
                dial_options=DialOptions(disable_webrtc=True, auth_entity=key_id,
                    credentials=Credentials(type='api-key', payload=secret)))), timeout=12)
            self.controller = Generic.from_robot(self.robot, 'sircus')
        try:
            return await asyncio.wait_for(self.controller.do_command(command, timeout=25), timeout=28)
        except Exception:
            # Reconnect on the next request after deployment or a module restart.
            try:
                await asyncio.wait_for(self.robot.close(), timeout=2)
            except Exception:
                pass
            self.robot = self.controller = None
            raise

    def command(self, payload):
        future = asyncio.run_coroutine_threadsafe(self.invoke(payload), self.loop)
        try:
            return future.result(timeout=32)
        except FutureTimeout:
            future.cancel()
            raise RuntimeError('Viam did not answer in time; no local simulation was substituted') from None


def connection(connected=True):
    return {'backend': 'offline' if OFFLINE else 'viam', 'connected': connected,
            'machine_id': None if OFFLINE else MACHINE_ID,
            'machine_url': None if OFFLINE else MACHINE_URL,
            'scope': 'simulation', 'physically_verified': False}


def local_config():
    path = ROOT / 'scene.json'
    config = json.loads(path.read_text()) if path.exists() else default_config()
    validate_config(config)
    return config


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        if args and len(args) > 1 and str(args[1]) not in ('200', '304'):
            super().log_message(fmt, *args)

    def send(self, body, status=200, content_type='application/json'):
        if isinstance(body, (dict, list)):
            body = json.dumps(body, allow_nan=False).encode()
        elif isinstance(body, str):
            body = body.encode()
        self.send_response(status)
        self.send_header('Content-Type', content_type)
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Cache-Control', 'no-store')
        self.send_header('X-Content-Type-Options', 'nosniff')
        self.end_headers()
        try:
            self.wfile.write(body)
        except BrokenPipeError:
            pass

    def do_GET(self):
        path = urlsplit(self.path).path
        if path.startswith('/api/physical/'):
            if not self.physical_host_ok():
                return self.send({'error':'loopback host required'}, 403)
            if PHYSICAL is None:
                return self.send({'mode':'disabled','connected':False,'allowed_actions':[],
                    'calibration_missing':['Physical backend not enabled for this dashboard process'],
                    'error':'Start with --physical-mode hardware or dry_run'}, 503)
            try:
                if path == '/api/physical/status':
                    return self.send(PHYSICAL.call('status'))
                if path == '/api/physical/camera':
                    data, mime, _ = PHYSICAL.call('camera')
                    return self.send(data, content_type=mime)
                return self.send({'error':'unknown physical endpoint'},404)
            except Exception as exc:
                return self.send({'error':str(exc)[:500]},503)
        if path.startswith('/api/'):
            try:
                if OFFLINE:
                    if path == '/api/config': return self.send({'config': local_config(), 'connection': connection()})
                    if path == '/api/arm-model': return self.send(home())
                    if path == '/api/status': return self.send({**STATE.copy(), 'connection': connection()})
                elif path in ('/api/config', '/api/arm-model', '/api/probe', '/api/status'):
                    response = BRIDGE.command({'status': True} if path == '/api/status' else {'probe': True})
                    return self.send({**response, 'connection': connection()})
                return self.send({'error': 'unknown endpoint'}, 404)
            except Exception as exc:
                return self.send({'error': str(exc)[:1500], 'connection': connection(False)}, 503)
        base = ROOT / 'models' if path.startswith('/models/') else ROOT / 'web'
        relative = path.removeprefix('/models/') if path.startswith('/models/') else path.lstrip('/')
        requested = (base / (relative or 'index.html')).resolve()
        if not requested.is_relative_to(base.resolve()) or not requested.is_file():
            return self.send({'error': 'not found'}, 404)
        return self.send(requested.read_bytes(), content_type=mimetypes.guess_type(str(requested))[0] or 'application/octet-stream')

    def do_POST(self):
        origin = self.headers.get('Origin')
        if origin and origin != 'http://' + self.headers.get('Host', ''):
            return self.send({'error': 'same-origin requests required'}, 403)
        if not self.headers.get('Content-Type', '').startswith('application/json'):
            return self.send({'error': 'application/json required'}, 415)
        try:
            length = int(self.headers.get('Content-Length', '0'))
        except ValueError:
            return self.send({'error': 'invalid request size'}, 413)
        if length <= 0 or length > 200000:
            return self.send({'error': 'invalid request size'}, 413)
        if urlsplit(self.path).path.startswith('/api/physical/'):
            if not self.physical_host_ok() or PHYSICAL is None:
                return self.send({'error':'physical backend unavailable'},403)
            if not secrets.compare_digest(self.headers.get('X-Sircus-Token',''),PHYSICAL.token):
                return self.send({'error':'physical action token required'},403)
            if urlsplit(self.path).path != '/api/physical/action':
                return self.send({'error':'unknown physical endpoint'},404)
            try:
                payload = json.loads(self.rfile.read(length))
                if not isinstance(payload, dict):
                    raise ValueError('JSON object required')
                return self.send(PHYSICAL.call('action',payload))
            except ValueError as exc:
                return self.send({'error':str(exc)},409)
            except Exception as exc:
                return self.send({'error':str(exc)[:500]},503)
        if not LOCK.acquire(blocking=False):
            return self.send({'error': 'another dashboard command is being submitted'}, 409)
        try:
            payload = json.loads(self.rfile.read(length))
            if not isinstance(payload, dict): raise ValueError('JSON object required')
            path = urlsplit(self.path).path
            if not OFFLINE:
                if path == '/api/config':
                    command = {'set_scene': payload['config']}
                elif path == '/api/board':
                    command = {'set_board': payload['board']}
                elif path in ('/api/simulate', '/api/run'):
                    command = {'run': True, 'action': payload['action'], 'seed': payload.get('seed', 42)}
                elif path == '/api/train':
                    command = {'train': True, 'seed': payload.get('seed', 42), 'budget': payload.get('budget', 24)}
                elif path == '/api/stop':
                    command = {'stop': True}
                else:
                    return self.send({'error': 'unknown endpoint'}, 404)
                return self.send({**BRIDGE.command(command), 'connection': connection()})
            # Explicit --offline compatibility mode, visibly labeled in the UI.
            config = payload.get('config', local_config())
            validate_config(config)
            if path == '/api/config':
                temp = ROOT / 'scene.json.tmp'
                temp.write_text(json.dumps(config, indent=2, allow_nan=False))
                temp.replace(ROOT / 'scene.json')
                return self.send({'config': config, 'connection': connection()})
            if path == '/api/simulate':
                STATE.update(phase='simulating')
                result = simulate(config, payload['action'], seed=payload.get('seed', 42), frames=True)
                result.update(sequence(config, result['release_xyz_mm']))
                return self.send(result)
            if path == '/api/train':
                STATE.update(phase='training')
                def progress(n, budget, best, row):
                    STATE.update(trial=n, budget=budget, best_reward=best['reward'], last_outcome=row['outcome'])
                result = train(config, payload.get('seed', 42), payload.get('budget', 24), progress)
                result['best_result'].update(sequence(config, result['best_result']['release_xyz_mm']))
                return self.send(result)
            return self.send({'error': 'unknown endpoint'}, 404)
        except (ValueError, KeyError, TypeError) as exc:
            return self.send({'error': str(exc)}, 400)
        except Exception as exc:
            return self.send({'error': str(exc)[:1500], 'connection': connection(False)}, 503)
        finally:
            STATE.update(phase='idle')
            LOCK.release()

    def physical_host_ok(self):
        port = self.server.server_address[1]
        return self.headers.get('Host','') in (f'127.0.0.1:{port}',f'localhost:{port}')


def main():
    global OFFLINE, BRIDGE, PHYSICAL
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=8765)
    parser.add_argument('--offline', action='store_true', help='Explicitly use standalone physics; no Viam commands')
    parser.add_argument('--physical-mode', choices=('disabled','dry_run','hardware'), default='disabled')
    parser.add_argument('--physical-credentials', type=Path, default=ROOT.parent/'perception/.private/sfhack10.json')
    parser.add_argument('--physical-data-root', type=Path, default=None)
    args = parser.parse_args()
    OFFLINE = args.offline
    if not OFFLINE: BRIDGE = ViamBridge()
    if args.physical_mode != 'disabled':
        data_root = args.physical_data_root or ROOT.parent/'physical_trials/data'/args.physical_mode
        PHYSICAL = PhysicalBridge(args.physical_mode,args.physical_credentials,data_root)
    print(f'Sircus dashboard: http://127.0.0.1:{args.port} ({"OFFLINE simulation" if OFFLINE else "Viam two-arm simulation bridge"})', flush=True)
    ThreadingHTTPServer(('127.0.0.1', args.port), Handler).serve_forever()


if __name__ == '__main__':
    main()
