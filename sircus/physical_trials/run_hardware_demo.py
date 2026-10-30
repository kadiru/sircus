"""Explicit operator launcher. The default is an offline check, never deployment."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

from .demo_setup import (ROOT, SetupError, inspect_loaded_profile, inspect_profile,
                         launch_argv, load_audio_environment, load_profile)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile', type=Path, required=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--check', action='store_true', help='Inspect local files only (default)')
    mode.add_argument('--launch', action='store_true', help='Start the physical dashboard; operator deployment only')
    parser.add_argument('--disable-cameras', action='store_true', help='Disable dashboard physical camera requests and recording')
    args = parser.parse_args(argv)
    if not args.launch:
        report = inspect_profile(args.profile)
        print(json.dumps(report, indent=2, allow_nan=False))
        return 0 if report['prepared'] else 2
    try:
        profile = load_profile(args.profile)
    except SetupError:
        print('Hardware launch blocked: invalid profile. Run --check for the redacted report.', file=sys.stderr)
        return 2
    # Inspect and execute the same validated snapshot. A profile edit while the
    # offline checks run must not silently substitute different deployment paths.
    report = inspect_loaded_profile(profile)
    if not report['robot_prepared']:
        print('Hardware launch blocked: offline robot preparation failed. Run --check for the redacted report.', file=sys.stderr)
        return 2
    # Never inherit a previous Mini address/token or shell-source credential files.
    env = {key: value for key, value in os.environ.items() if not key.startswith('SIRCUS_AUDIO_')}
    try:
        if report['audio_prepared']:
            env.update(load_audio_environment(profile))
        else:
            print('Optional audio is not prepared; starting with FX unavailable.', file=sys.stderr)
    except SetupError:
        print('Optional audio configuration changed; starting with FX unavailable.', file=sys.stderr)
    command = list(launch_argv(profile, ROOT))
    if args.disable_cameras:command.append("--disable-physical-cameras")
    print('Starting the sfhack10 dashboard on loopback. This does not command motion or tools. FX starts off.', flush=True)
    try:
        os.execvpe(command[0], command, env)
    except OSError:
        print('The configured Python could not start. No replacement runtime was launched.', file=sys.stderr)
        return 2
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
