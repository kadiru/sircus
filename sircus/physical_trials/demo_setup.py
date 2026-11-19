"""Offline preparation checks for the saved-pose sfhack10 demonstration.

Reads local files and installed package metadata only. Never imports the robot
SDK/controller, instantiates a service, contacts a device, executes a shell, or
changes calibration/trial records. Passing these checks is not live readiness.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import ipaddress
import json
import math
import os
from pathlib import Path
import re
import shlex
import shutil
import stat
from urllib.parse import urlsplit

SCHEMA = 'sircus-hardware-demo-profile-v1'
ADDRESS = 'sfhack10-main.ty1cywh7tq.viam.cloud'
POSE_FIELDS = {'x', 'y', 'z', 'o_x', 'o_y', 'o_z', 'theta'}
PROFILE_FIELDS = {'schema', 'machine', 'python', 'credentials_path', 'data_root',
                  'show_data_root', 'port', 'audio_mode', 'audio_env_file'}
AUDIO_KEYS = {'SIRCUS_AUDIO_URL', 'SIRCUS_AUDIO_TOKEN', 'SIRCUS_AUDIO_NODE'}
ROOT = Path(__file__).resolve().parents[2]


class SetupError(ValueError):
    """Safe fixed diagnostics: never include input values or file contents."""


def absolute_path(value, field):
    if not isinstance(value, str) or not value or '\0' in value or not Path(value).is_absolute():
        raise SetupError(field + ' must be an explicit absolute filesystem path')
    return Path(value)


def read_json(path, *, private=False):
    try:
        info = path.lstat()
        if not stat.S_ISREG(info.st_mode):
            raise SetupError('Expected a regular file; symlinks are not accepted')
        if private and (info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) != 0o600):
            raise SetupError('Private configuration must be owned by this user with mode0600')
        value = json.loads(path.read_text(), parse_constant=lambda _: (_ for _ in ()).throw(ValueError()))
    except SetupError:
        raise
    except (OSError, ValueError, UnicodeError):
        raise SetupError('Required JSON file is missing, unreadable, or invalid') from None
    if not isinstance(value, dict):
        raise SetupError('JSON configuration must contain an object')
    return value


def load_profile(path):
    value = read_json(Path(path))
    if set(value) != PROFILE_FIELDS or value.get('schema') != SCHEMA or value.get('machine') != 'sfhack10':
        raise SetupError('Expected the exact sfhack10 hardware-demo profile schema')
    for key in ('python', 'credentials_path', 'data_root', 'show_data_root'):
        absolute_path(value[key], key)
    if type(value['port']) is not int or not 1024 <= value['port'] <= 65535:
        raise SetupError('port must be an integer from1024 through65535')
    if value['audio_mode'] not in ('local', 'mini', 'disabled'):
        raise SetupError('audio_mode must be local, mini, or disabled')
    if value['audio_env_file'] is not None:
        absolute_path(value['audio_env_file'], 'audio_env_file')
    return value


def load_audio_environment(profile):
    """Return validated backend environment for an explicit future launcher.

    Callers must never log this return value. This function does not mutate the
    current process environment or contact/start the service.
    """
    if profile['audio_mode'] == 'disabled':
        return {}
    if profile['audio_env_file'] is None:
        raise SetupError('Audio backend environment JSON has not been supplied')
    value = read_json(absolute_path(profile['audio_env_file'], 'audio_env_file'), private=True)
    if not set(value).issubset(AUDIO_KEYS) or any(not isinstance(v, str) for v in value.values()):
        raise SetupError('Audio environment accepts only the documented backend string fields')
    try:
        url = urlsplit(value.get('SIRCUS_AUDIO_URL', ''))
        address = ipaddress.ip_address(url.hostname or '')
        private = address.version == 4 and any(address in ipaddress.ip_network(net)
            for net in ('10.0.0.0/8', '172.16.0.0/12', '192.168.0.0/16'))
        if (url.scheme != 'http' or url.username or url.password or url.query or url.fragment
                or url.path not in ('', '/') or not url.port or address.version != 4
                or not (address.is_loopback or private)):
            raise ValueError()
        if profile['audio_mode'] == 'local' and not address.is_loopback:
            raise ValueError()
        if profile['audio_mode'] == 'mini' and (not private or not value.get('SIRCUS_AUDIO_TOKEN')):
            raise ValueError()
    except (ValueError, TypeError):
        raise SetupError('Audio address must match its explicit loopback/private-LAN mode; Mini requires a token') from None
    node = value.get('SIRCUS_AUDIO_NODE')
    if node is not None and (not absolute_path(node, 'SIRCUS_AUDIO_NODE').is_file() or not os.access(node, os.X_OK)):
        raise SetupError('Configured backend Node executable is unavailable')
    return value


def inspect_dependencies(python):
    python = Path(python)
    if not python.is_file() or not os.access(python, os.X_OK):
        raise SetupError('Configured project Python executable is unavailable')
    # Inspect a venv without executing its interpreter or importing any package.
    venv = python.parent.parent
    try:
        config = (venv / 'pyvenv.cfg').read_text()
    except OSError:
        raise SetupError('Project Python must identify a virtual environment with readable metadata') from None
    match = re.search(r'^version\s*=\s*(\d+\.\d+(?:\.\d+)?)\s*$', config, re.M)
    if not match or not match[1].startswith('3.12.'):
        raise SetupError('The reviewed hardware runtime requires project Python3.12')
    paths = list((venv / 'lib').glob('python*/site-packages'))
    installed = {dist.metadata.get('Name', '').lower().replace('_', '-'): dist.version
                 for dist in importlib.metadata.distributions(path=[str(p) for p in paths])}
    required = {}
    for line in (ROOT / 'sircus/simulation/requirements-viam.txt').read_text().splitlines():
        if '==' in line and not line.lstrip().startswith('#'):
            name, version = line.strip().split('==', 1)
            required[name] = version
    if any(installed.get(name) != version for name, version in required.items()):
        raise SetupError('Project virtual-environment dependencies do not match the pinned requirements')
    return {'python_version': match[1], 'pinned_dependencies_match': True,
            'packages': required, 'sdk_imported': False, 'source': 'venv and distribution metadata only'}


def inspect_calibration(data_root):
    data_root = Path(data_root)
    if data_root.name != 'hardware' or any(p in {'dry_run', 'mock', 'mocks'} for p in data_root.parts):
        raise SetupError('Use an explicitly named hardware data directory, never mock/dry-run records')
    if not data_root.is_dir() or data_root.is_symlink():
        raise SetupError('Hardware data directory is missing or is a symlink')
    path = data_root / 'calibration.json'
    c = read_json(path)
    if c.get('mode', 'hardware') != 'hardware' or c.get('machine', 'sfhack10') != 'sfhack10':
        raise SetupError('Calibration belongs to a different mode or machine')
    if c.get('frame') != 'world':
        raise SetupError('Calibration must explicitly identify world-frame targets')
    for key in ('loading_pose', 'drop_pose'):
        pose = c.get(key)
        if not isinstance(pose, dict) or set(pose) != POSE_FIELDS or any(
                type(v) not in (int, float) or not math.isfinite(v) for v in pose.values()):
            raise SetupError('Both complete finite loading/drop TCP poses are required')
        if sum(pose[k] ** 2 for k in ('o_x', 'o_y', 'o_z')) < 1e-16:
            raise SetupError('Saved TCP orientation vector is invalid')
        if pose['z'] < 0 or any(abs(pose[k]) > 2000 for k in ('x', 'y', 'z')):
            raise SetupError('Saved TCP target lies outside the operator editor bounds')
    joints = c.get('board_joints')
    if not isinstance(joints, list) or len(joints) != 6 or any(
            type(v) not in (int, float) or not math.isfinite(v) for v in joints):
        raise SetupError('Six finite taught board-arm reference joints are required')
    pending = 0
    for journal in (data_root / 'trials').glob('*/trial.json'):
        trial = read_json(journal)
        if trial.get('mode') != 'hardware' or trial.get('machine') != 'sfhack10':
            raise SetupError('Hardware data directory contains foreign or mock trial records')
        if not trial.get('saved'):
            pending += 1
    return {'poses_present': True, 'board_reference_present': True, 'frame': 'world',
            'pose_reference': 'arm flange/TCP; not measured ball center',
            'calibration_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
            'pending_journals': pending, 'live_route_verified': False,
            'variable_height_ready': all(c.get(k) is not None for k in ('reference_z_mm', 'min_height_mm', 'max_height_mm')) and c.get('confirmed') is True,
            'height_limits_required_for_fixed_cycle': False}


def launch_argv(profile, source_root=ROOT):
    """Exact explicit command only; never invoked by this module."""
    return [profile['python'], str(Path(source_root) / 'sircus/simulation/server.py'),
            '--offline', '--show', '--physical-mode', 'hardware',
            '--physical-credentials', profile['credentials_path'],
            '--physical-data-root', profile['data_root'],
            '--show-data-root', profile['show_data_root'], '--port', str(profile['port'])]


def _empty_report():
    return {'schema': 'sircus-hardware-demo-preflight-v1', 'checked_at': datetime.now(timezone.utc).isoformat(),
              'machine': 'sfhack10', 'inspection': 'offline filesystem only', 'prepared': False,
              'robot_prepared': False, 'audio_prepared': False, 'live_hardware_verified': False,
              'checks': {}, 'errors': [], 'unknown_live_checks': [
                  'Exclusive physical control including any sfhack14 reference',
                  'Live camera and both-arm stationary readback',
                  'Fixed board/suction and bucket placement; verified loading/drop route',
                  'Fresh saved recording before release and actual gripper held-object behavior',
                  'Physical Stop and no unrequested repeat',
                  'Measured per-ball MIDI identity, audible impacts, FX mute and audio transport']}


def inspect_profile(profile_path, source_root=ROOT):
    """Read/validate a profile once for the standalone offline inspection CLI."""
    try:
        profile = load_profile(profile_path)
    except SetupError as exc:
        report = _empty_report()
        report['errors'].append(str(exc));return report
    return inspect_loaded_profile(profile, source_root)


def inspect_loaded_profile(profile, source_root=ROOT):
    """Inspect the exact in-memory snapshot already returned by load_profile.

    The operator launcher must build its command from this same snapshot, not a
    second read of a potentially edited profile. Credentials/audio values are
    never copied into the report. Host files still require live runtime checks.
    """
    report = _empty_report()
    report['audio_mode'] = profile['audio_mode']
    for name, inspect in (
        ('python', lambda: inspect_dependencies(profile['python'])),
        ('credentials', lambda: _inspect_credentials(profile['credentials_path'])),
        ('calibration', lambda: inspect_calibration(profile['data_root'])),
        ('show_data', lambda: _inspect_show_data(profile)),
    ):
        try:
            report['checks'][name] = {'status': 'prepared', **inspect()}
        except SetupError as exc:
            report['checks'][name] = {'status': 'missing_or_invalid', 'reason': str(exc)}
            report['errors'].append(name + ': ' + str(exc))
        except (OSError, ValueError, TypeError):
            report['checks'][name] = {'status': 'missing_or_invalid'}
            report['errors'].append(name + ' configuration is missing or invalid; inspect the local profile/file')
    report['robot_prepared'] = all(row['status'] == 'prepared' for row in report['checks'].values())
    try:
        env = load_audio_environment(profile)
        node_available = bool(env.get('SIRCUS_AUDIO_NODE') or shutil.which('node'))
        report['audio_prepared'] = profile['audio_mode'] == 'disabled' or node_available
        report['checks']['audio'] = {'status': 'disabled' if profile['audio_mode'] == 'disabled' else ('prepared' if node_available else 'node_missing'),
            'environment_valid': True, 'node_available': node_available, 'connection_verified': False,
            'node_version_verified': False,
            'fx_initially_enabled': False, 'ball_bindings_verified': False}
    except (SetupError, OSError, ValueError, TypeError):
        report['checks']['audio'] = {'status': 'missing_or_invalid', 'connection_verified': False,
                                    'fx_initially_enabled': False, 'ball_bindings_verified': False}
        report['errors'].append('Audio backend configuration is missing or invalid; values are intentionally redacted')
    report['prepared'] = report['robot_prepared'] and report['audio_prepared']
    report['sound_goal_commissioned'] = False
    if report['robot_prepared']:
        report['launch_argv'] = launch_argv(profile, source_root)
        report['launch_command'] = shlex.join(report['launch_argv'])
        report['launch_note'] = 'Operator deployment only. Load audio JSON through the approved launcher, never shell-source it. This command is not executed.'
    return report


def _inspect_credentials(path):
    value = read_json(Path(path), private=True)
    if (value.get('address') != ADDRESS or any(not isinstance(value.get(k), str) or not value[k]
            for k in ('api_key_id', 'api_key'))):
        raise SetupError('Credentials must identify the pinned sfhack10 endpoint with nonempty API fields')
    return {'file_valid': True, 'machine_pin_matches': True, 'live_authentication_verified': False}


def _inspect_show_data(profile):
    path = Path(profile['show_data_root'])
    # Compare resolved paths as well as rejecting a direct symlink: a symlinked
    # ancestor or '..' must not hide shared/nested hardware and show state.
    resolved = path.resolve()
    hardware = Path(profile['data_root']).resolve()
    if (resolved == hardware or resolved in hardware.parents or hardware in resolved.parents
            or path.is_symlink() or (path.exists() and not path.is_dir())):
        raise SetupError('Show configuration needs its own explicit directory')
    return {'explicit_directory': True, 'exists': path.is_dir(), 'ball_binding_configuration_present': (path / 'show-audio.json').is_file()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile', type=Path, required=True)
    args = parser.parse_args()
    report = inspect_profile(args.profile)
    print(json.dumps(report, indent=2, allow_nan=False))
    raise SystemExit(0 if report['prepared'] else 2)


if __name__ == '__main__':
    main()
