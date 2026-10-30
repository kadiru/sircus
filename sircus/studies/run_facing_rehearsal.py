"""Small, prespecified bucket-facing smoke experiment; never connects to robots.

The search and geometrically constrained frozen predictor are separate methods.
Fresh rollout seeds evaluate each selected action without selecting/refitting on
those outcomes. This is diagnostic evidence, not a generalization benchmark.
"""
import argparse
import copy
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from sircus.operations.numerical_worker import handle, evaluate
from sircus.studies.profiles import front_scene

CASES = (
    ('front', [600., -150., 350.], [600., -400.]),
    ('left', [590., -170., 360.], [530., -430.]),
    ('right_farther', [620., -130., 345.], [700., -420.]),
)


def run(budget=32, trials=5):
    checkpoint = json.loads((ROOT/'baselines/front-bucket-20261030/checkpoint.json').read_text())
    rows = []
    for name, board, bucket in CASES:
        scene = front_scene()
        scene['board']['center_mm'] = copy.deepcopy(board)
        scene['bin']['center_xy_mm'] = copy.deepcopy(bucket)
        for mode in ('facing_search', 'facing_learned'):
            row = {'case': name, 'mode': mode, 'board_center_mm': board,
                   'bucket_center_xy_mm': bucket, 'planning_seed': 42,
                   'evaluation_seeds': list(range(1001, 1001+trials))}
            try:
                plan = handle({'op': 'plan', 'scene': scene, 'checkpoint': checkpoint,
                               'mode': mode, 'seed': 42, 'budget': budget})
                row.update(action=plan['action'], board_rpy_deg=plan['selected_scene']['board']['rpy_deg'],
                           board_facing=plan.get('board_facing'),
                           release_coordinates=plan['release_coordinates'],
                           inference=plan.get('inference'),
                           planning_success=plan['result']['success'],
                           planning_outcome=plan['result']['outcome'],
                           native_paths_checked=False, trials=[])
                for seed in row['evaluation_seeds']:
                    result = evaluate(plan['selected_scene'], plan['release_action'], seed, frames=False)['result']
                    row['trials'].append({'seed': seed, 'success': result['success'],
                                          'outcome': result['outcome']})
                row['captures'] = sum(item['success'] for item in row['trials'])
                row['trial_count'] = len(row['trials'])
            except ValueError as error:
                row.update(error=str(error), captures=None, trial_count=0)
            rows.append(row)
            print(json.dumps(row, allow_nan=False), flush=True)
    return {'schema': 'sircus-facing-rehearsal-smoke-v1',
            'created_at_utc': datetime.now(timezone.utc).isoformat(),
            'data_origin': 'synthetic', 'physically_verified': False,
            'native_paths_checked': False, 'profile_id': 'front-negative-y-v1',
            'checkpoint_id': checkpoint['checkpoint_id'], 'search_budget': budget,
            'scope': 'Three prespecified layouts; fresh noisy rollouts, no refitting or test-based action selection. '
                     'Small diagnostic; separate from all published learning baselines.',
            'tilt_convention': '0 degrees horizontal; 90 degrees vertical; horizontal normal aims at bucket center',
            'rows': rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--budget', type=int, default=32, choices=range(1, 33))
    parser.add_argument('--trials', type=int, default=5, choices=range(1, 21))
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Choose a new output path; historical results are never overwritten')
    report = run(args.budget, args.trials)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')


if __name__ == '__main__':
    main()
