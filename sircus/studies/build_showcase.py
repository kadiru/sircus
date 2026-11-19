"""Build a portable, truthful numerical demo reel; no server or hardware I/O.

The fixed preset is reused unchanged from the published nominal rehearsal. Four
predeclared release seeds are retained regardless of outcome. Learning clips are
copied from immutable held-out replays after guarded frozen predictions match.
"""
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from sircus.operations.numerical_worker import annotate, choreography, evaluate
from sircus.studies import front_study
from sircus.studies.profiles import front_scene, profile_metadata
from shot_rollout import release_coordinates

SCHEMA = 'sircus-demo-reel-v1'
BASE = ROOT / 'baselines/front-bucket-20261030'
PRESET_PATH = ROOT / 'baselines/facing-rehearsal-20261030/results.json'
BALLS = (
    {'id': 'ball-pink', 'label': 'Pink', 'color': '#f472b6'},
    {'id': 'ball-purple', 'label': 'Purple', 'color': '#a78bfa'},
    {'id': 'ball-green', 'label': 'Green', 'color': '#4ade80'},
    {'id': 'ball-gray', 'label': 'Gray', 'color': '#94a3b8'},
)
FIXED_SEEDS = (2026103001, 2026103002, 2026103003, 2026103004)
SIZES = (8, 24, 64, 128)
TRAIN_SEED = 20261030
# BLAS/CPU rounding only: one billionth of a millimetre or degree; no relative tolerance.
ACTION_NUMERICAL_ATOL = 1e-9


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def load(path):
    return json.loads(path.read_text())


def artifact(path):
    return {'path': str(path.relative_to(ROOT)),
            'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def verify_baseline():
    """Reject changed historical evidence rather than re-labeling it."""
    for line in (BASE / 'SHA256SUMS').read_text().splitlines():
        expected, name = line.split('  ', 1)
        path = BASE / name
        if not path.resolve().is_relative_to(BASE.resolve()):
            raise ValueError('Invalid published artifact path')
        if artifact(path)['sha256'] != expected:
            raise ValueError('Published baseline checksum mismatch: ' + name)


def timeline_events(clip_id, result, release_s):
    """Sonification events refer to actual model contacts, never guessed objects."""
    rows = [
        {'t': 1.35, 'type': 'carry_start', 'source': 'illustrative_endpoint_choreography'},
        {'t': release_s, 'type': 'carry_stop', 'source': 'illustrative_endpoint_choreography'},
        {'t': release_s, 'type': 'release', 'source': 'illustrative_endpoint_choreography'},
    ]
    for index, contact in enumerate(result['contacts']):
        rows.append({'t': release_s + contact['t'], 'physics_t': contact['t'],
                     'type': 'contact', 'contact_object': contact['object'],
                     'source': 'versioned_numerical_ball_physics', 'contact_index': index,
                     'source_contact': copy.deepcopy(contact)})
    # Keep all terminal outcomes, but only the versioned scorer can label capture.
    rows.append({'t': release_s + result['duration_s'], 'type': 'outcome',
                 'outcome': result['outcome'], 'success': result['success'],
                 'source': 'versioned_complete_shot_evaluator',
                 'scoring_version': result['scoring_version'],
                 'audible': result['outcome'] == 'captured' or result['outcome'] == 'no_entry'})
    rows.sort(key=lambda row: row['t'])
    for index, row in enumerate(rows):
        row.update(event_id=f'{clip_id}:event:{index:04d}', trial_id=clip_id,
                   evidence='synthetic', physically_verified=False)
    return rows


def make_clip(clip_id, label, row, initial_scene, ball, role, provenance, **extra):
    row = annotate(copy.deepcopy(row))
    sequence = choreography(row, initial_scene)
    normal = row['board_pose']['normal']
    tilt = math.degrees(math.acos(max(-1., min(1., normal[2]))))
    result = row['result']
    clip = {
        'id': clip_id, 'trial_id': clip_id, 'label': label, 'ball': copy.deepcopy(ball),
        'scene': row['selected_scene'], 'action': row['action'],
        'release_action': row['release_action'], 'release_coordinates': row['release_coordinates'],
        'board_pose': row['board_pose'], 'arm10_release_boxes': row['arm10_release_boxes'],
        'arm10_release_joints_deg': row['arm10_release_joints_deg'],
        'board_tilt_deg': tilt, 'drop_height_mm': row['release_action']['gap_mm'],
        'height_definition': 'World-Z clearance from nominal contact ball center to commanded release ball center; not calibrated TCP height',
        'initial_scene': copy.deepcopy(initial_scene), 'result': result,
        'choreography': sequence,
        'events': timeline_events(clip_id, result, sequence['release_time_s']),
        'duration_s': sequence['release_time_s'] + result['duration_s'],
        'role': role, 'provenance': provenance, 'data_origin': 'synthetic',
        'physically_verified': False, 'native_paths_checked': False,
        'multiball_physics': False, 'seed': result['seed'], **extra,
    }
    clip['artifact_sha256'] = digest(clip)
    return clip


def validate_predicted_action(predicted, saved):
    """Validate floating inference across CPUs while replaying original evidence.

    This tolerance never changes a checkpoint, recorded action, or saved result.
    Key/schema changes and nonfinite values are rejected independently.
    """
    keys = set(front_study.ACTION_NAMES)
    if set(predicted) != keys or set(saved) != keys:
        raise ValueError('Frozen prediction action keys differ from the saved action contract')
    for key in front_study.ACTION_NAMES:
        actual, expected = predicted[key], saved[key]
        if (type(actual) not in (int, float) or type(expected) not in (int, float)
                or not math.isfinite(actual) or not math.isfinite(expected)):
            raise ValueError(f'Frozen prediction action.{key}: finite numeric values required')
        delta = abs(actual - expected)
        if delta > ACTION_NUMERICAL_ATOL:
            raise ValueError(f'Frozen prediction action.{key}: computed={actual!r}, '
                             f'saved={expected!r}, absolute_difference={delta:.17g}, '
                             f'allowed={ACTION_NUMERICAL_ATOL:g} (mm or degrees; rtol=0)')


def published_clip(saved, episode, checkpoint, ball, *, paired=False):
    proposal = front_study.predict(checkpoint, episode['scene'])
    if proposal['checkpoint_id'] != saved['checkpoint_id']:
        raise ValueError('Published replay checkpoint_id differs from guarded frozen prediction')
    validate_predicted_action(proposal['action'], saved['action'])
    if saved['result']['seed'] != episode['rollout_seed']:
        raise ValueError('Published held-out rollout seed mismatch')
    # Recomputed inference is validation only. Never pair a slightly different
    # action/board orientation with the original trajectory. Preserve both exactly.
    selected = copy.deepcopy(saved['scene'])
    action = copy.deepcopy(saved['action'])
    release = {key: action[key] for key in ('u_mm', 'v_mm', 'gap_mm')}
    row = {**proposal, 'selected_scene': selected, 'action': action,
           'release_action': release, 'release_coordinates': release_coordinates(selected, release),
           'result': copy.deepcopy(saved['result'])}
    n = saved.get('train_scenes', 128)
    return make_clip('learning-' + saved['id'], saved['label'], row,
                     episode['scene'], ball, 'frozen_policy',
                     {'source': artifact(BASE / 'report.json'), 'source_replay_id': saved['id'],
                      'source_result_sha256': digest(saved['result']),
                      'prediction_validation': {'absolute_tolerance_mm_or_degrees': ACTION_NUMERICAL_ATOL,
                                                'relative_tolerance': 0., 'saved_action_preserved': True},
                      'inference': 'Guarded frozen prediction; no fitting, search or action selection during replay',
                      'selection': ('Prespecified first held-out layout, same release seed at every N; all misses retained'
                                    if paired else 'Separately labeled pre-existing held-out capture example; selected for successful illustration, not evidence of improvement')},
                     checkpoint_id=checkpoint['checkpoint_id'], train_scenes=n,
                     training_seed=TRAIN_SEED, scene_id=episode['scene_id'], split='test',
                     paired_layout=paired, in_distribution=proposal['scope']['in_distribution'])


def build_bundle():
    verify_baseline()
    report, manifest = load(BASE / 'report.json'), load(BASE / 'manifest.json')
    nominal = next(row for row in load(PRESET_PATH)['rows']
                   if row['case'] == 'front' and row['mode'] == 'facing_search')
    scene = front_scene()
    scene['board']['center_mm'] = copy.deepcopy(nominal['board_center_mm'])
    scene['board']['rpy_deg'] = copy.deepcopy(nominal['board_rpy_deg'])
    scene['bin']['center_xy_mm'] = copy.deepcopy(nominal['bucket_center_xy_mm'])
    action = {key: nominal['action'][key] for key in ('u_mm', 'v_mm', 'gap_mm')}
    selection = {
        'method': 'reuse_published_nominal_preset', 'source': artifact(PRESET_PATH),
        'source_case': 'front', 'source_mode': 'facing_search',
        'original_search_budget': 32, 'original_search_seed': 42,
        'new_search_budget': 0, 'frozen_before_releases': True,
        'seed_selection': 'Four seeds declared in builder source before execution; all retained without retries',
        'scope': 'Single handpicked nominal presentation layout, not a held-out policy benchmark',
    }
    fixed = [make_clip(f'fixed-{ball["id"]}-{seed}', f'{ball["label"]}: fixed preset',
                       evaluate(scene, action, seed), scene, ball, 'fixed_preset',
                       copy.deepcopy(selection), split='demonstration')
             for ball, seed in zip(BALLS, FIXED_SEEDS)]
    episodes = {row['scene_id']: row for row in manifest['shared']['test']}
    first = manifest['shared']['test'][0]
    paired = []
    for size, ball in zip(SIZES, BALLS):
        saved = next(row for row in report['replays']
                     if row.get('train_scenes') == size and row.get('paired_layout') == first['scene_id'])
        if saved['scene_id'] != first['scene_id']:
            raise ValueError('Paired clips must use the first predeclared held-out layout')
        checkpoint = load(BASE / f'checkpoints/seed-{TRAIN_SEED}-checkpoint-n{size}.json')
        paired.append(published_clip(saved, first, checkpoint, ball, paired=True))
    saved_success = next(row for row in report['replays']
                         if row['role'] == 'frozen_policy' and row['split'] == 'test'
                         and row['result']['success'] is True and 'train_scenes' not in row)
    success = published_clip(saved_success, episodes[saved_success['scene_id']],
                             load(BASE / 'checkpoint.json'), BALLS[2])
    source_paths = [Path(__file__), PRESET_PATH, BASE / 'report.json', BASE / 'manifest.json',
                    BASE / 'checkpoint.json', ROOT / 'sircus/operations/numerical_worker.py',
                    ROOT / 'sircus/studies/front_study.py', ROOT / 'sircus/studies/profiles.py',
                    ROOT / 'sircus/two-arm/shot_rollout.py', ROOT / 'sircus/two-arm/capture_scoring.py',
                    ROOT / 'sircus/two-arm/two_arm_kinematics.py', ROOT / 'sircus/two-arm/scene.json']
    source_paths += [BASE / f'checkpoints/seed-{TRAIN_SEED}-checkpoint-n{n}.json' for n in SIZES]
    return {
        'schema': SCHEMA, 'data_origin': 'synthetic', 'physically_verified': False,
        'native_paths_checked': False, 'hardware_enabled': False, 'profile': profile_metadata(),
        'fixed': {'preset': {'scene': scene, 'action': copy.deepcopy(nominal['action']),
                             'release_action': action, 'selection': selection},
                  'clips': fixed, 'captures': sum(c['result']['success'] for c in fixed), 'trials': 4,
                  'scope': 'Four independent single-ball numerical releases, each starting with an empty bucket; earlier balls do not persist or interact. Colors label identical ball mechanics. No accumulated-ball or inter-ball collision model, and no physical identity binding.'},
        'learning': {'same_layout_clips': paired, 'success_example': success,
                     'learning_curve': copy.deepcopy(report['learning_curve']),
                     'learning_curve_aggregate': copy.deepcopy(report['learning_curve_aggregate']),
                     'summary': copy.deepcopy(report['summary']),
                     'paired_scene_id': first['scene_id'],
                     'training_protocol': {'seeds': manifest['seeds'], 'nested_sizes': manifest['nested_sizes'],
                                           'shared_test_layouts': len(manifest['shared']['test']),
                                           'shared_validation_layouts': len(manifest['shared']['validation']),
                                           'teacher_budget': manifest['teacher_budget'],
                                           'selection_policy': manifest['selection_policy']},
                     'scope': 'Actual saved frozen checkpoints and held-out replays. Paired clips all miss; success example is separately selected. Aggregate learning is not strictly monotonic; no physical transfer claim.'},
        'provenance': {'physics_version': fixed[0]['result']['physics_version'],
                       'scoring_version': fixed[0]['result']['scoring_version'],
                       'success_definition': fixed[0]['result']['success_definition'],
                       'geometry': 'Measured dimensions retained; placements, mounts, contact dynamics and bucket height interpretation remain provisional',
                       'animation': 'Actual numerical ball frames plus illustrative arm endpoint interpolation; no validated arm paths',
                       'audio': 'Synthetic contact sonification only; no BLE input, external audio service, or physical outcome detection'},
        'manifest': {'sources': [artifact(path) for path in source_paths],
                     'clips': [{'id': clip['id'], 'sha256': clip['artifact_sha256']}
                               for clip in fixed + paired + [success]],
                     'fixed_release_seeds': list(FIXED_SEEDS),
                     'paired_layout_selection': 'manifest.shared.test[0]'},
    }


def write_bundle(destination):
    destination = Path(destination)
    if destination.exists():
        raise ValueError('Choose a new output directory; existing artifacts are never overwritten')
    bundle = build_bundle()
    destination.mkdir(parents=True)
    path = destination / 'showcase.json'
    path.write_bytes(canonical(bundle) + b'\n')
    (destination / 'SHA256SUMS').write_text(hashlib.sha256(path.read_bytes()).hexdigest() + '  showcase.json\n')
    return bundle


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    bundle = write_bundle(args.output)
    print(json.dumps({'output': str(args.output), 'fixed_captures': bundle['fixed']['captures'],
                      'fixed_trials': 4, 'paired_outcomes': [c['result']['outcome'] for c in bundle['learning']['same_layout_clips']],
                      'success_example': bundle['learning']['success_example']['result']['outcome']}))


if __name__ == '__main__':
    main()
