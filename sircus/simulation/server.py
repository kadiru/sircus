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
import signal
import subprocess
import sys
import threading
from concurrent.futures import TimeoutError as FutureTimeout
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit, parse_qs
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
STUDY_LOCK = threading.Lock()
STUDY_MAX_BYTES = 1_000_000
OPERATIONS = None
TRAINING_JOBS = None
WORKFLOW_TOKEN = secrets.token_urlsafe(32)


def study_preview(payload):
    """A separate numerical process avoids legacy/two-arm module-name collisions.

    No robot client, config save, policy migration or runtime fallback is used.
    The only executable/path is repository-owned; callers supply JSON values.
    """
    if not isinstance(payload, dict) or set(payload) != {'checkpoint', 'scene', 'seed'}:
        raise ValueError('Supply checkpoint, scene and seed JSON values only')
    if not isinstance(payload['checkpoint'], dict) or not isinstance(payload['scene'], dict):
        raise ValueError('Checkpoint and scene must be JSON objects')
    seed = payload['seed']
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError('Seed must be an integer from 0 to 2^32-1')
    raw = json.dumps(payload, allow_nan=False)
    worker = ROOT.parent / 'two-arm' / 'study_preview.py'
    try:
        completed = subprocess.run([sys.executable, str(worker)], input=raw,
            capture_output=True, text=True, timeout=35, check=False,
            env={**os.environ, 'PYTHONDONTWRITEBYTECODE': '1'})
    except subprocess.TimeoutExpired:
        raise ValueError('Numerical preview timed out; no runtime action was sent') from None
    if len(completed.stdout) > 5_000_000:
        raise ValueError('Numerical preview response exceeded its limit')
    try:
        value = json.loads(completed.stdout)
    except (ValueError, TypeError):
        raise ValueError('Numerical preview unavailable; no runtime action was sent') from None
    if completed.returncode or value.get('error'):
        raise ValueError(str(value.get('error', 'Numerical preview rejected'))[:500])
    return value


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
        if path.startswith(('/api/operations/', '/api/training/')):
            return self.workflow_get(path)
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
        path = urlsplit(self.path).path
        origin = self.headers.get('Origin')
        if origin and origin != 'http://' + self.headers.get('Host', ''):
            return self.send({'error': 'same-origin requests required'}, 403)
        if not self.headers.get('Content-Type', '').startswith('application/json'):
            return self.send({'error': 'application/json required'}, 415)
        try:
            length = int(self.headers.get('Content-Length', '0'))
        except ValueError:
            return self.send({'error': 'invalid request size'}, 413)
        if length <= 0 or length > (STUDY_MAX_BYTES if path == '/api/study/preview' or path.startswith('/api/operations/') else 200000):
            return self.send({'error': 'invalid request size'}, 413)
        if path.startswith(('/api/operations/', '/api/training/')):
            return self.workflow_post(path, length)
        if path.startswith('/api/study/'):
            if not self.physical_host_ok():
                return self.send({'error': 'loopback host required'}, 403)
            if path != '/api/study/preview':
                return self.send({'error': 'unknown study endpoint'}, 404)
            if not STUDY_LOCK.acquire(blocking=False):
                return self.send({'error': 'A numerical study preview is already running'}, 409)
            try:
                payload = json.loads(self.rfile.read(length),
                    parse_constant=lambda _: (_ for _ in ()).throw(ValueError('Finite JSON values required')))
                return self.send(study_preview(payload))
            except (ValueError, KeyError, TypeError) as exc:
                return self.send({'error': str(exc)[:500]}, 400)
            except Exception:
                return self.send({'error': 'Numerical preview unavailable; no runtime action was sent'}, 503)
            finally:
                STUDY_LOCK.release()
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

    def workflow_ready(self):
        return OFFLINE and PHYSICAL is None and OPERATIONS is not None and TRAINING_JOBS is not None

    def workflow_get(self, path):
        if not self.physical_host_ok():
            return self.send({'error': 'loopback host required'}, 403)
        if not self.workflow_ready():
            return self.send({'error': 'Start the separate operator simulation with ./start-operator-sim.sh',
                              'hardware_enabled': False}, 503)
        try:
            if path == '/api/operations/status':
                return self.send({**OPERATIONS.status(), 'action_token': WORKFLOW_TOKEN})
            if path == '/api/training/status':
                return self.send({**TRAINING_JOBS.status(), 'action_token': WORKFLOW_TOKEN})
            if path == '/api/training/artifact':
                params = parse_qs(urlsplit(self.path).query, keep_blank_values=True)
                if set(params) != {'run_id', 'name'} or any(len(v) != 1 for v in params.values()):
                    raise ValueError('Supply one run_id and artifact name')
                return self.send(TRAINING_JOBS.artifact(params['run_id'][0], params['name'][0]))
            return self.send({'error': 'unknown workflow endpoint'}, 404)
        except (ValueError, KeyError, TypeError, FileNotFoundError) as exc:
            return self.send({'error': str(exc)[:300]}, 400)
        except Exception:
            return self.send({'error': 'Simulation workspace unavailable'}, 503)

    def workflow_post(self, path, length):
        if not self.physical_host_ok():
            return self.send({'error': 'loopback host required'}, 403)
        if not self.workflow_ready():
            return self.send({'error': 'Operator simulation is disabled; no hardware fallback'}, 503)
        if not secrets.compare_digest(self.headers.get('X-Sircus-Workflow-Token', ''), WORKFLOW_TOKEN):
            return self.send({'error': 'workflow action token required'}, 403)
        if path not in ('/api/operations/action', '/api/training/start', '/api/training/cancel'):
            return self.send({'error': 'unknown workflow endpoint'}, 404)
        try:
            value = json.loads(self.rfile.read(length), parse_constant=lambda _: (_ for _ in ()).throw(ValueError('Finite JSON required')))
            if not isinstance(value, dict):
                raise ValueError('JSON object required')
            if path == '/api/operations/action':
                result = OPERATIONS.action(value)
            elif path == '/api/training/start':
                result = TRAINING_JOBS.start(value)
            else:
                if value:
                    raise ValueError('Cancellation requires an empty object')
                result = TRAINING_JOBS.cancel()
            return self.send({**result, 'action_token': WORKFLOW_TOKEN}, 202)
        except (ValueError, KeyError, TypeError) as exc:
            return self.send({'error': str(exc)[:300]}, 400)
        except RuntimeError as exc:
            return self.send({'error': str(exc)[:300]}, 409)
        except Exception:
            return self.send({'error': 'Simulation action failed; no hardware fallback'}, 503)

    def physical_host_ok(self):
        port = self.server.server_address[1]
        return self.headers.get('Host','') in (f'127.0.0.1:{port}',f'localhost:{port}')


def main():
    global OFFLINE, BRIDGE, PHYSICAL, OPERATIONS, TRAINING_JOBS
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=8765)
    parser.add_argument('--offline', action='store_true', help='Explicitly use standalone physics; no Viam commands')
    parser.add_argument('--physical-mode', choices=('disabled','dry_run','hardware'), default='disabled')
    parser.add_argument('--physical-credentials', type=Path, default=ROOT.parent/'perception/.private/sfhack10.json')
    parser.add_argument('--physical-data-root', type=Path, default=None)
    parser.add_argument('--operations', action='store_true', help='Enable guided simulation and local training; requires offline and physical disabled')
    parser.add_argument('--workflow-data-root', type=Path, default=ROOT.parent.parent/'.runtime'/'operator-lab')
    args = parser.parse_args()
    if args.operations and (not args.offline or args.physical_mode != 'disabled'):
        parser.error('--operations requires --offline --physical-mode disabled')
    OFFLINE = args.offline
    if not OFFLINE: BRIDGE = ViamBridge()
    if args.physical_mode != 'disabled':
        data_root = args.physical_data_root or ROOT.parent/'physical_trials/data'/args.physical_mode
        PHYSICAL = PhysicalBridge(args.physical_mode,args.physical_credentials,data_root)
    if args.operations:
        sys.path.insert(0, str(ROOT.parent.parent))
        from sircus.operations.service import GuidedOperations
        from sircus.operations.training import TrainingJobs
        OPERATIONS = GuidedOperations(data_root=args.workflow_data_root/'trials')
        TRAINING_JOBS = TrainingJobs(data_root=args.workflow_data_root/'training')
    print(f'Sircus dashboard: http://127.0.0.1:{args.port} ({"OFFLINE simulation" if OFFLINE else "Viam two-arm simulation bridge"})', flush=True)
    httpd = ThreadingHTTPServer(('127.0.0.1', args.port), Handler)
    try:
        if args.operations:
            # SIGTERM normally bypasses Python finally. Rehearsal/training owns
            # numerical children, so route its shutdown through the cleanup below.
            # Legacy and physical server signal behavior is deliberately unchanged.
            def terminate_operations(_signum, _frame):
                raise SystemExit(0)
            signal.signal(signal.SIGTERM, terminate_operations)
        httpd.serve_forever()
    finally:
        httpd.server_close()
        if OPERATIONS is not None: OPERATIONS.close()
        if TRAINING_JOBS is not None: TRAINING_JOBS.close()


if __name__ == '__main__':
    main()
