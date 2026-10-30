"""Optional, bounded observer for the Python dashboard; no robot dependencies.

Only this backend process receives the audio URL/token. Robot callbacks enqueue
immutable evidence without waiting for Node, a socket, or an audio acknowledgment.
"""
from collections import deque
from contextlib import contextmanager
import ipaddress
import json
import math
import os
from pathlib import Path
import secrets
import subprocess
import threading
import time
from urllib.parse import urlsplit

from .contract import AudioEvent


class AudioObserver:
    def __init__(self, *, environ=None, process_factory=subprocess.Popen,
                 max_queue=32, start_worker=True, clock=time.monotonic):
        env = dict(os.environ if environ is None else environ)
        self.show = None
        self.token = secrets.token_urlsafe(32)  # Local action token; never the Mini credential.
        self._clock = clock
        self._condition = threading.Condition()
        self._queue = deque()
        self._max_queue = max_queue
        self._retired = set()
        self._clients = {}  # Do not evict ordering history and replay an old browser request.
        self._revision = 0
        self._playback_epoch = 0
        self._closed = False
        self._process = None
        self._factory = process_factory
        self._env = env
        self._state = dict(configured=False, enabled=False, requested_enabled=False,
                           volume=.35, connection='not_configured', pending=False,
                           error=None, mute_confirmed=None, show_audio_capable=False,
                           engine_ready=None, mapping_matches=False, ball_inputs=[],
                           bound_ball_count=0, connected_ball_count=0)
        raw = env.get('SIRCUS_AUDIO_URL', '')
        if raw:
            try:
                url = urlsplit(raw)
                address = ipaddress.ip_address(url.hostname or '')
                private = any(address in ipaddress.ip_network(net) for net in
                              ('10.0.0.0/8', '172.16.0.0/12', '192.168.0.0/16'))
                if (url.scheme != 'http' or url.username or url.password or url.query or
                    url.fragment or url.path not in ('', '/') or address.version != 4 or
                    not (address.is_loopback or private) or not url.port or
                    (not address.is_loopback and not env.get('SIRCUS_AUDIO_TOKEN'))):
                    raise ValueError('unsupported audio configuration')
                self._state.update(configured=True, connection='connecting')
            except (ValueError, TypeError):
                self._state.update(connection='configuration_error', error='Invalid private audio configuration')
        if self._state['configured'] and start_worker:
            threading.Thread(target=self._run, name='sircus-audio-io', daemon=True).start()

    def snapshot(self):
        with self._condition:
            return {**self._state, 'ball_inputs':[dict(row) for row in self._state['ball_inputs']], 'action_token': self.token}

    def playback_readiness(self):
        """Snapshot an audio-only creation permit, never a robot readiness gate."""
        with self._condition:
            return {'epoch':self._playback_epoch,'ready':self._playback_ready()}

    def _playback_ready(self):
        return (not self._closed and self._state['enabled'] and self._state['show_audio_capable']
                and self._state['connection']!='error')

    @contextmanager
    def playback_creation(self,expected_epoch):
        # ShowAudio holds its lifecycle lock first. The body only assigns the new
        # Event/identity; no IO or callback may run under this short condition.
        with self._condition:
            yield self._playback_epoch==expected_epoch and self._playback_ready()

    def _error(self, text='Optional audio unavailable'):
        with self._condition:
            self._playback_epoch += 1
            self._state.update(connection='error', error=text, pending=False, enabled=False,
                               engine_ready=None, mapping_matches=False, show_audio_capable=False,
                               ball_inputs=[], bound_ball_count=0, connected_ball_count=0)
            if not self._state['requested_enabled']:
                self._state['mute_confirmed'] = False
        if self.show is not None:self.show.cancel_replay(blocking=False)

    def emit(self, event):
        """Never performs IO, starts a process, or waits for transport."""
        try:
            if self.show is not None:
                event = self.show.decorate(event)
                if event is None:return  # A configuration lock cannot delay a robot hook.
            with self._condition:
                if self._closed or not self._state['enabled']:
                    return
                # Validate and snapshot while preserving every producer evidence field.
                value = AudioEvent.from_dict(event).to_dict()
                if value.get('ball_id') and not self._state['show_audio_capable']:
                    self._state['error'] = 'Mini service has not verified four-ball audio protocol support'
                    return
                if value['trial_id'] in self._retired:
                    return
                if value['event_type'] == 'outcome' and (value['outcome'] == 'unknown' or
                   (value['source'] != 'simulation' and not value['confirmed'])):
                    return
                if len(self._queue) >= self._max_queue:
                    self._state['error'] = 'Audio queue full; cue dropped'
                    return
                self._queue.append({'op': 'emit', 'event': value, 'revision': self._revision,
                                    'queued_at': self._clock(), 'queued_unix_ms': time.time()*1000})
                self._condition.notify()
        except Exception:
            self._error('Invalid audio evidence; cue dropped')

    def stop(self, trial_id):
        """Retire locally and prioritize scoped cancellation; never await audio."""
        try:
            if not isinstance(trial_id, str) or not 1 <= len(trial_id) <= 256:
                return
            if self.show is not None:
                self.show.cancel_trial(trial_id)
            with self._condition:
                if self._closed:
                    return
                if len(self._retired) >= 4096 and trial_id not in self._retired:
                    self._state.update(enabled=False, requested_enabled=False,
                                       error='Audio trial capacity reached; FX disabled')
                    self._enqueue_control({'op': 'fx', 'enabled': False})
                    return
                self._retired.add(trial_id)
                if self._state['configured']:
                    self._enqueue_control({'op': 'stop', 'trial_id': trial_id})
        except Exception:
            self._error('Audio cancellation unavailable; use Mini mute')

    def show_status(self):
        return self.show.status() if self.show is not None else {'configured':False}

    def configure_show(self, config):
        if self.show is None:raise ValueError('Show audio coordinator is unavailable')
        return self.show.configure(config)

    def configure_show_remote(self, config):
        """Audio-only configuration changes mute first; never called by motion guards."""
        with self._condition:
            self._playback_epoch += 1
            self._state.update(enabled=False, requested_enabled=False, show_audio_capable=False,
                               mapping_matches=False, ball_inputs=[], bound_ball_count=0, connected_ball_count=0)
            if self._state['configured']:
                self._enqueue_control({'op':'fx','enabled':False})
                self._enqueue_control({'op':'show_config','config':config})
        if self.show is not None:self.show.cancel_replay(blocking=False)

    def _enqueue_control(self, command):
        # Called under the condition lock. Cancelling never waits behind stale cues.
        self._revision += 1
        if command['op']=='fx' and not command['enabled']:self._playback_epoch += 1
        command['revision'] = self._revision
        command['queued_at'] = self._clock()
        if command['op'] in ('stop', 'fx'):
            # A scoped trial stop must never discard an outstanding global mute.
            self._queue = deque(item for item in self._queue
                                if item['op'] != 'emit' and
                                not (command['op'] == 'fx' and item['op'] == 'fx'))
        if command['op'] == 'show_config':
            self._queue = deque(item for item in self._queue if item['op'] != 'show_config')
        if command['op'] == 'volume':
            self._queue = deque(item for item in self._queue if item['op'] != 'volume')
        if len(self._queue) >= self._max_queue:
            # Losing cancellation is worse than losing optional playback. Collapse
            # an overloaded control queue to an explicit global mute, never an enable.
            self._queue.clear()
            self._playback_epoch += 1
            command = {'op': 'fx', 'enabled': False, 'revision': self._revision,
                       'queued_at': self._clock()}
            self._state.update(enabled=False, requested_enabled=False, mute_confirmed=False,
                               error='Audio control capacity reached; FX disabled')
        if command['op'] == 'stop' or (command['op'] == 'fx' and not command['enabled']):
            self._queue.appendleft(command)
        else:
            self._queue.append(command)
        self._state['pending'] = True
        self._condition.notify()

    def control(self, payload):
        if not isinstance(payload, dict):
            raise ValueError('JSON object required')
        client_id, sequence = payload.get('client_id'), payload.get('sequence')
        if (not isinstance(client_id, str) or not 1 <= len(client_id) <= 64 or
                not all(c.isalnum() or c in '-_' for c in client_id) or
                type(sequence) is not int or not 1 <= sequence <= 2**53-1):
            raise ValueError('Valid client_id and positive sequence are required')
        body = {k: v for k, v in payload.items() if k not in ('client_id', 'sequence')}
        action = payload.get('action')
        if action == 'fx' and set(body) == {'action', 'enabled'} and type(payload['enabled']) is bool:
            command = {'op': 'fx', 'enabled': payload['enabled']}
        elif action == 'volume' and set(body) == {'action', 'volume'}:
            volume = payload['volume']
            if isinstance(volume, bool) or not isinstance(volume, (int, float)) or not math.isfinite(volume) or not 0 <= volume <= 1:
                raise ValueError('Volume must be between 0 and 1')
            command = {'op': 'volume', 'volume': float(volume)}
        else:
            raise ValueError('Unsupported audio control')
        with self._condition:
            if not self._state['configured'] or self._closed:
                raise ValueError('Audio backend is not configured')
            prior = self._clients.get(client_id)
            if prior and sequence <= prior[0]:
                if sequence == prior[0] and body != prior[1]:
                    raise ValueError('Audio sequence was reused with different content')
                return self.snapshot()
            if prior is None and len(self._clients) >= 128:
                raise ValueError('Audio control session capacity reached; restart required')
            self._clients[client_id] = (sequence, dict(body))
            if action == 'fx':
                enabled = payload['enabled']
                self._playback_epoch += 1
                self._state.update(requested_enabled=enabled, enabled=False,
                                   mute_confirmed=None if enabled else False)
            else:
                self._state['volume'] = float(volume)
            self._enqueue_control(command)
        if action=='fx' and self.show is not None:self.show.cancel_replay(blocking=False)
        return self.snapshot()

    def _take(self):
        with self._condition:
            while not self._queue and not self._closed:
                self._condition.wait()
            if self._closed:
                return None
            item = self._queue.popleft()
            if item['op'] == 'emit' and (self._clock()-item['queued_at'] > 1 or
                                       item['event']['trial_id'] in self._retired):
                self._state['error'] = 'Stale or cancelled audio cue dropped'
                return {}
            return {key: value for key, value in item.items() if key != 'queued_at'}

    def _accept_status(self, message):
        # Whitelist instead of relaying the service status (which includes host metadata).
        cancel_replay = False
        with self._condition:
            if not isinstance(message, dict) or message.get('revision', -1) < self._revision:
                return
            status = message.get('status', {})
            if not isinstance(status, dict):
                return
            was_enabled = self._state['enabled']
            for key in ('enabled', 'requested_enabled', 'pending', 'mute_confirmed', 'show_audio_capable', 'engine_ready', 'mapping_matches'):
                value = status.get(key)
                if type(value) is bool or (key in ('mute_confirmed','engine_ready') and value is None):
                    self._state[key] = value
            if 'ball_inputs' in status:
                rows=status['ball_inputs'];allowed={'ball-pink','ball-purple','ball-green','ball-gray'};clean=[]
                if isinstance(rows,list) and len(rows)<=4:
                    for row in rows:
                        if not isinstance(row,dict) or row.get('ball_id') not in allowed or type(row.get('bound')) is not bool or type(row.get('connected')) is not bool:continue
                        allowed.remove(row['ball_id'])
                        clean.append({'ball_id':row['ball_id'],'bound':row['bound'],
                                      'connected':bool(self._state['mapping_matches'] and row['bound'] and row['connected'])})
                self._state['ball_inputs']=clean
                self._state['bound_ball_count']=sum(row['bound'] for row in clean)
                self._state['connected_ball_count']=sum(row['connected'] for row in clean)
            if not self._state['requested_enabled']:
                self._state['enabled'] = False
            if status.get('connection') in ('connected', 'error', 'connecting'):
                self._state['connection'] = status['connection']
            # Never pass arbitrary exception strings, URLs, tokens, or service error payloads.
            errors = {
                'audio_disabled_remotely': 'FX was disabled on the audio host; enable it here to resume cues',
                'audio_mapping_unverified': 'Audio mappings changed or are unverified; reapply saved mappings before enabling FX',
                'audio_engine_unavailable': 'Native audio engine is unavailable; inspect the audio host',
                'remote_mute_unconfirmed': 'Remote audio may be on; mute is unconfirmed',
            }
            code = status.get('error')
            if not code:
                self._state['error'] = None
            elif isinstance(code, str) and code in errors:
                self._state['error'] = errors[code]
            else:
                self._state['error'] = 'Optional audio unavailable; remote mute may be unconfirmed'
            cancel_replay = (was_enabled and not self._state['enabled']) or self._state['connection']=='error'
            if cancel_replay:self._playback_epoch += 1
        # Never acquire ShowAudio's lock under the queue condition (emit takes the reverse order).
        if cancel_replay and self.show is not None:self.show.cancel_replay(blocking=False)

    def _read(self, process):
        try:
            for line in process.stdout:
                if len(line) <= 4096:
                    self._accept_status(json.loads(line))
        except Exception:
            pass
        if not self._closed:
            self._error('Audio worker disconnected; use Mini mute if needed')

    def _run(self):
        try:
            worker = Path(__file__).parent/'booth/integration/dashboard-worker.mjs'
            worker_env = {name: self._env[name] for name in
                          ('PATH', 'SIRCUS_AUDIO_URL', 'SIRCUS_AUDIO_TOKEN') if name in self._env}
            process = self._factory([self._env.get('SIRCUS_AUDIO_NODE', 'node'), str(worker)],
                                    stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                    stderr=subprocess.DEVNULL, text=True, bufsize=1, env=worker_env)
            with self._condition:
                self._process = process
                if self._closed:
                    process.terminate()
                    return
            threading.Thread(target=self._read, args=(process,), daemon=True, name='sircus-audio-status').start()
            while not self._closed:
                item = self._take()
                if item is None:
                    break
                if item:
                    process.stdin.write(json.dumps(item, allow_nan=False)+'\n')
                    process.stdin.flush()
        except Exception:
            self._error('Audio worker unavailable; check backend configuration')

    def close(self):
        # Shutdown itself never sends/awaits robot commands or restarts a service.
        with self._condition:
            self._closed = True
            self._playback_epoch += 1
            self._queue.clear()
            self._state.update(enabled=False, requested_enabled=False)
            self._condition.notify_all()
        if self.show is not None:self.show.close()
        if self._process is not None:
            try:
                self._process.terminate()
            except OSError:
                pass
