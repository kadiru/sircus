"""Repeat the real controller with MockAdapter only; no Viam import or hardware option.

Run from the repository root:
  python -m sircus.physical_trials.validate_deterministic --cycles 20 --output .runtime/reliability-001
The default two-second waits are deliberately retained. These are control-flow
checks, not simulated ball trajectories, capture scores or physical evidence.
"""
import argparse
import asyncio
import hashlib
import json
from pathlib import Path
import statistics
import time

from . import controller as module
from .controller import Controller, TrialError, utc, write_json
from .mock_adapter import MockAdapter

BALL_IDS = ('ball-pink', 'ball-purple', 'ball-green', 'ball-gray')


def require(condition, message):
    if not condition:
        raise AssertionError(message)


async def command(controller, action, **values):
    await controller.action({'action': action, **values})
    if controller.task:
        await asyncio.wait_for(asyncio.shield(controller.task), 60)


def verify_trial(controller, adapter, trial, previous_id):
    require(trial['id'] != previous_id, 'Every operator press must create a new trial ID')
    require(trial['mode'] == 'dry_run', 'Evidence must remain explicitly synthetic')
    require(trial['status'] == 'completed' and trial['saved'], 'Practice trial was not retained')
    require(trial.get('outcome') is None and trial['label_status'] == 'unlabeled',
            'Control completion must not be labelled capture')
    require(trial['target_pose'] == trial['measured_release_pose'], 'Mock released at wrong pose')
    require(trial['height_mm'] is None, 'Fixed routine must not imply a calibrated ball gap')
    events = trial['events']

    def stamp(command_name, event):
        return next(e['monotonic'] for e in events if e['command'] == command_name and e['event'] == event)

    load_wait = stamp('close_gripper', 'command_sent') - stamp('load_drop_cycle', 'loading_countdown_started')
    drop_wait = stamp('drop', 'command_sent') - stamp('load_drop_cycle', 'move_drop_returned')
    require(load_wait >= module.LOAD_WAIT_SECONDS, 'Loading countdown was shortened')
    require(drop_wait >= module.DROP_SETTLE_SECONDS, 'Release occurred before the settling interval')
    require(stamp('recording', 'ready') < stamp('drop', 'command_sent'), 'Recording started after release')
    state = controller.status_snapshot()
    require(not state['busy'] and not state['fault'], 'Controller did not return to a usable idle state')
    require('load_drop_cycle' in state['allowed_actions'], 'Next operator press is disabled')
    require(not adapter.record['active'], 'Recorder was left running after completion')
    directory = controller.root / 'trials' / trial['id']
    observations = [json.loads(line) for line in (directory / 'observations.jsonl').read_text().splitlines()]
    require(bool(observations), 'No saved mock observations')
    require(all(row['mock'] and set(row['state']['arms']) == {'arm', 'arm-board'} for row in observations),
            'Each observation needs both mock arm states')
    require(all((directory / row['image']).is_file() for row in observations), 'Missing saved mock image')
    require(json.loads((directory / 'trial.json').read_text())['id'] == trial['id'], 'Saved trial identity mismatch')
    return {'trial_id': trial['id'], 'ball_id': trial['ball_id'], 'status': trial['status'],
            'outcome': None, 'loading_wait_s': load_wait, 'drop_settle_s': drop_wait,
            'mock_frames': len(observations), 'loading_open_skipped': any(
                e['event'] == 'loading_open_skipped' for e in events)}


async def validate(output, cycles=20):
    if isinstance(cycles, bool) or not isinstance(cycles, int) or not 1 <= cycles <= 1000:
        raise ValueError('cycles must be an integer from 1 to 1000')
    output = Path(output)
    # Never reuse a previous run or the hardware calibration/data directory.
    output.mkdir(parents=True, exist_ok=False)
    source_dir = Path(__file__).parent
    report = {'schema': 'sircus.deterministic-control-validation.v1', 'status': 'running',
              'started_at': utc(), 'mode': 'dry_run', 'adapter': 'MockAdapter',
              'evidence': 'synthetic control flow; no ball physics, hardware, Viam or audio connection',
              'capture_rate': None, 'requested_cycles': cycles, 'cycles': [],
              'source_sha256': {name: hashlib.sha256((source_dir / name).read_bytes()).hexdigest()
                  for name in ('controller.py', 'mock_adapter.py', 'validate_deterministic.py')},
              'timing_seconds': {'loading': module.LOAD_WAIT_SECONDS, 'settle': module.DROP_SETTLE_SECONDS,
                                 'post_release': module.POST_RELEASE_SECONDS}}
    write_json(output / 'report.json', report)
    adapter = MockAdapter()
    controller = Controller(adapter, output / 'mock-data', mode='dry_run')
    started = time.monotonic()
    try:
        await controller.start()
        loading = dict(x=150., y=-600., z=200., o_x=0., o_y=0., o_z=-1., theta=0.)
        dropping = dict(loading, x=600., y=-150., z=600.)
        await command(controller, 'configure_poses', loading_pose=loading, drop_pose=dropping, confirmed=True)
        report['fixture'] = {'loading_pose': loading, 'drop_pose': dropping,
                             'provenance': 'Explicit mock TCP fixtures; not physical calibration'}
        previous_id = None
        for index in range(cycles):
            await command(controller, 'set_ball', ball_id=BALL_IDS[index % len(BALL_IDS)])
            before = len(adapter.commands)
            t0 = time.monotonic()
            await controller.action({'action': 'load_drop_cycle', 'fixed_setup_confirmed': True, 'practice_mode': True})
            # A second press during this accepted command must never queue a repeat.
            try:
                await controller.action({'action': 'load_drop_cycle', 'fixed_setup_confirmed': True, 'practice_mode': True})
            except TrialError:
                pass
            else:
                raise AssertionError('A duplicate in-flight press was accepted')
            await asyncio.wait_for(asyncio.shield(controller.task), 60)
            row = verify_trial(controller, adapter, controller.trial, previous_id)
            row['duration_s'] = time.monotonic() - t0
            commands = adapter.commands[before:]
            expected = [{'move_pose': loading}]
            if index == 0:
                expected.append('open_gripper')
            expected.extend(['close_gripper', {'move_pose': dropping}, 'open_gripper'])
            require(commands == expected, 'Unexpected actuator command order or redundant loading Open')
            previous_id = row['trial_id']
            report['cycles'].append(row)
            write_json(output / 'report.json', report)
            print(f"Mock cycle {index+1}/{cycles}: completed, {row['mock_frames']} saved frames; outcome unknown", flush=True)
        # There is no repeat timer: the controller must remain idle without a new press.
        before = list(adapter.commands)
        await asyncio.sleep(1)
        require(adapter.commands == before, 'Controller issued an unrequested automatic repeat')
        report.update(status='passed', completed_cycles=len(report['cycles']),
                      duplicate_presses_rejected=cycles, unrequested_repeat=False,
                      median_cycle_s=statistics.median(row['duration_s'] for row in report['cycles']))
    except BaseException as exc:
        report.update(status='failed', error_type=type(exc).__name__, error=str(exc), completed_cycles=len(report['cycles']))
        raise
    finally:
        try:
            await controller.close()
        finally:
            report.update(finished_at=utc(), duration_s=time.monotonic() - started)
            write_json(output / 'report.json', report)
            manifest = {str(path.relative_to(output)): hashlib.sha256(path.read_bytes()).hexdigest()
                        for path in sorted(output.rglob('*')) if path.is_file() and path.name != 'manifest.json'}
            write_json(output / 'manifest.json', {'algorithm': 'sha256', 'files': manifest})
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cycles', type=int, default=20)
    parser.add_argument('--output', type=Path, required=True, help='New empty run directory; never a hardware data root')
    args = parser.parse_args()
    asyncio.run(validate(args.output, args.cycles))


if __name__ == '__main__':
    main()
