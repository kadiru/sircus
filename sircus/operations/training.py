"""Bounded, local numerical training jobs. This module never creates a robot client."""
from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import threading
import time
import uuid

ROOT = Path(__file__).resolve().parents[2]
TRAINER = ROOT / 'sircus' / 'two-arm' / 'train_scene_study.py'
DEFAULTS = dict(seed=20261030, train_scenes=4, validation_scenes=1,
                test_scenes=2, teacher_budget=4, oracle_budget=2)
BOUNDS = dict(seed=(0, 2**31-1), train_scenes=(4, 64), validation_scenes=(1, 16),
              test_scenes=(1, 32), teacher_budget=(1, 32), oracle_budget=(1, 16))
MAX_ROLLOUTS = 1200
MAX_SECONDS = 1800
MAX_OUTPUT_BYTES = 4_000_000
MAX_ARTIFACT_BYTES = 24_000_000
ARTIFACTS = ('checkpoint.json', 'summary.json', 'episodes.json', 'experiment.json',
             'preview-scene.json', 'preview.json', 'test.json', 'validation.json',
             'training-labels.json')
ACTIVE = ('starting', 'running', 'canceling')
RUN_ID = re.compile(r'^[0-9a-f]{32}$')


def _json(path, value):
    raw = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n'
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(raw)
    temporary.replace(path)


def configuration(payload):
    if not isinstance(payload, dict) or set(payload) - set(DEFAULTS):
        raise ValueError('Training accepts only the documented numeric budget and seed fields')
    values = {**DEFAULTS, **payload}
    for key, value in values.items():
        low, high = BOUNDS[key]
        if type(value) is not int or not low <= value <= high:
            raise ValueError(f'{key} must be an integer from {low} to {high}')
    work = (values['train_scenes'] * values['teacher_budget'] +
            3 * values['validation_scenes'] +
            values['test_scenes'] * (2 + values['oracle_budget']))
    if work > MAX_ROLLOUTS:
        raise ValueError(f'Combined rollout budget exceeds {MAX_ROLLOUTS}')
    return values


class TrainingJobs:
    """One owned subprocess at a time; status and cancel do not wait for training.

    data_root is an application-owned path, never an HTTP payload field. Each run
    gets a random fresh directory. Local artifacts are immutable after completion.
    """
    def __init__(self, data_root=None, *, _popen=subprocess.Popen,
                 _max_seconds=MAX_SECONDS, _clock=time.monotonic, profile='historical'):
        if profile not in ('front', 'historical'):
            raise ValueError('Choose front or historical training profile')
        self.profile = profile
        self.trainer = ROOT/'sircus/studies/run_front_study.py' if profile == 'front' else TRAINER
        # Child cwd is the trainer directory; explicit relative output roots must
        # be anchored before spawning or artifacts land in a different directory.
        self.root = (Path(data_root) if data_root is not None else ROOT / '.runtime' / 'training').resolve()
        self._popen, self._max_seconds, self._clock = _popen, _max_seconds, _clock
        self._lock = threading.RLock()
        self._process = None
        self._thread = None
        self._cancel = threading.Event()
        self._state = dict(mode='simulation_only', phase='idle', run_id=None, config=None,
                           progress=None, recent_progress=[], summary=None, error=None, artifacts=[])

    def status(self):
        with self._lock:
            state = copy.deepcopy(self._state)
        state['profile'] = self.profile
        state['profile_id'] = 'front-negative-y-v1' if self.profile == 'front' else 'historical-positive-y-v1'
        state['limits'] = dict(max_rollouts=MAX_ROLLOUTS, max_seconds=self._max_seconds,
                               data_origin='synthetic', hardware_enabled=False,
                               native_paths_checked=False,
                               distribution='front-negative-y core layouts' if self.profile == 'front' else 'historical positive-y narrow layouts',
                               action_names=['roll_deg', 'pitch_deg', 'u_mm', 'v_mm', 'gap_mm'],
                               height_semantics='gap is world-Z ball clearance above nominal contact, not TCP Z',
                               training='ridge regression from training-only teacher search',
                               evaluation='frozen prediction; separately labeled per-scene oracle')
        return state

    def _metadata_path(self, run_id):
        return self.root / f'{run_id}.job.json'

    def _persist(self):
        # Caller owns lock. Metadata contains no environment, credentials or stderr.
        if self._state['run_id']:
            _json(self._metadata_path(self._state['run_id']), self._state)

    def start(self, payload):
        config = configuration(payload)
        with self._lock:
            if self._state['phase'] in ACTIVE:
                raise ValueError('A training job is already active; cancel it or wait')
            self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
            if self.root.is_symlink():
                raise ValueError('Training storage must not be a symlink')
            run_id = uuid.uuid4().hex
            self._cancel = threading.Event()
            self._state = dict(mode='simulation_only', phase='starting', run_id=run_id,
                               config=config, progress=None, recent_progress=[], summary=None,
                               error=None, artifacts=[], started_unix_s=time.time())
            try:
                self._persist()
            except OSError:
                self._state.update(phase='failed', error='Unable to create local training metadata')
                raise ValueError('Unable to create local training metadata') from None
            self._thread = threading.Thread(target=self._run, args=(run_id, config, self._cancel),
                                            name='sircus-numerical-training', daemon=True)
            self._thread.start()
            return self.status()

    def cancel(self):
        with self._lock:
            if self._state['phase'] in ACTIVE:
                self._cancel.set()
                self._state['phase'] = 'canceling'
                try:
                    self._persist()
                except OSError:
                    self._state['error'] = 'Unable to persist cancellation; owned job is still being stopped'
            return self.status()

    def close(self):
        """Bounded application shutdown; terminate only our owned training child."""
        self.cancel()
        with self._lock:
            thread = self._thread
        if thread is None or thread is threading.current_thread():
            return self.status()
        deadline = time.monotonic() + 4
        thread.join(timeout=1)
        with self._lock:
            process = self._process
        if thread.is_alive() and process is not None and process.poll() is None:
            self._stop_process(process)
        thread.join(timeout=max(0, deadline-time.monotonic()))
        return self.status()

    @staticmethod
    def _stop_process(process):
        # start_new_session ensures this group belongs to this job, never a server.
        if process.poll() is not None:
            return
        try:
            if os.getpgid(process.pid) != process.pid:
                raise RuntimeError('Owned training process group changed')
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=1)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=2)
        except ProcessLookupError:
            pass

    def _consume(self, stream, run_id, is_stdout, output, exceeded):
        try:
            while True:
                line = stream.readline(65_537)
                if not line:
                    break
                raw_size = len(line.encode('utf8', errors='replace'))
                with self._lock:
                    output['size'] += raw_size
                    if output['size'] > MAX_OUTPUT_BYTES or len(line) > 65_536:
                        exceeded.set()
                        return
                    output['stdout' if is_stdout else 'stderr'].append(line)
                    if is_stdout:
                        try:
                            value = json.loads(line)
                        except (ValueError, TypeError):
                            continue
                        if isinstance(value, dict) and isinstance(value.get('phase'), str):
                            if self._state['run_id'] == run_id:
                                self._state['progress'] = value
                                self._state['recent_progress'].append(value)
                                del self._state['recent_progress'][:-64]
        finally:
            stream.close()

    def _run(self, run_id, config, canceled):
        output = {'size': 0, 'stdout': [], 'stderr': []}
        exceeded = threading.Event()
        process, readers = None, []
        phase, error, summary, artifacts = 'failed', 'Numerical trainer failed; inspect local evidence', None, []
        start = self._clock()
        try:
            command = [sys.executable, str(self.trainer), '--output', str(self.root / run_id)]
            for key, value in config.items():
                command += ['--' + key.replace('_', '-'), str(value)]
            # No inherited Viam/audio secrets, cloud identity, or shell expansion.
            environment = {'PATH': os.environ.get('PATH', '/usr/bin:/bin'),
                           'PYTHONDONTWRITEBYTECODE': '1', 'PYTHONUNBUFFERED': '1',
                           'OPENBLAS_NUM_THREADS': '1', 'OMP_NUM_THREADS': '1'}
            process = self._popen(command, cwd=str(self.trainer.parent), env=environment,
                                  stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                  stderr=subprocess.PIPE, text=True, encoding='utf8', errors='replace',
                                  start_new_session=True, bufsize=1)
            with self._lock:
                self._process = process
                if not canceled.is_set():
                    self._state['phase'] = 'running'
                self._persist()
            for stream, is_stdout in ((process.stdout, True), (process.stderr, False)):
                reader = threading.Thread(target=self._consume,
                    args=(stream, run_id, is_stdout, output, exceeded), daemon=True)
                readers.append(reader)
                reader.start()
            reason = None
            while process.poll() is None:
                if canceled.is_set():
                    reason = 'canceled'
                elif exceeded.is_set():
                    reason = 'output_limit'
                elif self._clock() - start > self._max_seconds:
                    reason = 'timeout'
                if reason:
                    self._stop_process(process)
                    break
                time.sleep(.04)
            process.wait(timeout=3)
            for reader in readers:
                reader.join(timeout=2)
            if canceled.is_set():
                phase, error = 'canceled', None
            elif exceeded.is_set():
                error = 'Trainer exceeded its bounded output limit'
            elif reason == 'timeout':
                error = 'Training time limit reached; partial evidence preserved'
            elif process.returncode != 0:
                error = 'Numerical trainer failed; partial evidence preserved'
            else:
                summary = self._load_file(run_id, 'summary.json')
                checkpoint_raw = self._artifact_path(run_id, 'checkpoint.json').read_bytes()
                checkpoint = json.loads(checkpoint_raw)
                if (summary.get('status') != 'complete' or
                    summary.get('checkpoint_file_sha256') != hashlib.sha256(checkpoint_raw).hexdigest() or
                    summary.get('checkpoint_id') != checkpoint.get('checkpoint_id') or
                    summary.get('checkpoint_unchanged_after_test') is not True or
                    summary.get('split_manifest_unchanged_after_test') is not True or
                    summary.get('physically_verified') is not False):
                    raise ValueError('Trainer completion artifact failed consistency checks')
                artifacts = self._list_artifacts(run_id)
                phase, error = 'complete', None
        except Exception:
            if canceled.is_set():
                phase, error = 'canceled', None
            else:
                error = 'Numerical training failed; no validated checkpoint is available'
        finally:
            if process is not None and process.poll() is None:
                self._stop_process(process)
            for reader in readers:
                reader.join(timeout=2)
            # Bounded local diagnostics only; never publish arbitrary stderr into status.
            storage_failed = False
            for kind in ('stdout', 'stderr'):
                try:
                    (self.root / f'{run_id}.{kind}.log').write_text(''.join(output[kind]))
                except OSError:
                    storage_failed = True
            with self._lock:
                # A cancellation during final artifact hashing must still win.
                if canceled.is_set():
                    phase, error = 'canceled', None
                elif storage_failed:
                    phase, error = 'failed', 'Unable to persist local training evidence'
                self._process = None
                self._state.update(phase=phase, error=error, summary=summary if phase == 'complete' else None,
                                   artifacts=artifacts if phase == 'complete' else [],
                                   finished_unix_s=time.time(), elapsed_seconds=self._clock()-start)
                try:
                    self._persist()
                except OSError:
                    # Never leave an apparent running job or expose a checkpoint whose
                    # completed manifest could not be saved (for example a full disk).
                    self._state.update(phase='canceled' if canceled.is_set() else 'failed',
                        error='Unable to persist local training metadata', summary=None, artifacts=[])

    def _artifact_path(self, run_id, name):
        if not isinstance(run_id, str) or not RUN_ID.fullmatch(run_id) or name not in ARTIFACTS:
            raise ValueError('Unknown training run or artifact')
        path = self.root / run_id / name
        if path.is_symlink() or path.parent.is_symlink() or not path.is_file():
            raise ValueError('Training artifact unavailable')
        if path.resolve().parent != (self.root / run_id).resolve() or path.stat().st_size > MAX_ARTIFACT_BYTES:
            raise ValueError('Training artifact exceeds its safe boundary')
        return path

    def _load_file(self, run_id, name):
        return json.loads(self._artifact_path(run_id, name).read_text())

    def _list_artifacts(self, run_id):
        values = []
        for name in ARTIFACTS:
            path = self.root / run_id / name
            if path.is_file():
                raw = self._artifact_path(run_id, name).read_bytes()
                values.append(dict(name=name, size_bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest()))
        return values

    def artifact(self, run_id, name):
        if not isinstance(run_id, str) or not RUN_ID.fullmatch(run_id) or name not in ARTIFACTS:
            raise ValueError('Unknown training run or artifact')
        metadata_path = self._metadata_path(run_id)
        if metadata_path.is_symlink() or not metadata_path.is_file():
            raise ValueError('Unknown training run')
        metadata = json.loads(metadata_path.read_text())
        if metadata.get('phase') != 'complete':
            raise ValueError('Artifacts from incomplete or canceled jobs are not selectable')
        entry = next((row for row in metadata.get('artifacts', []) if row.get('name') == name), None)
        if entry is None:
            raise ValueError('Training artifact unavailable')
        path = self._artifact_path(run_id, name)
        raw = path.read_bytes()
        if len(raw) != entry['size_bytes'] or hashlib.sha256(raw).hexdigest() != entry['sha256']:
            raise ValueError('Training artifact changed after completion')
        return json.loads(raw)
