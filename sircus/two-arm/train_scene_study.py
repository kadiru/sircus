"""Train a small scene-conditioned predictor and benchmark frozen held-out inference.

No robot clients, Viam connection, deployment, cloud identity or audio. Teacher
search and the separately reported held-out oracle are never policy inference.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import time

import scene_study as study


def write(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+'\n')


def progress(phase, **values):
    print(json.dumps({'phase': phase, **values}, allow_nan=False), flush=True)


def model_score(rows):
    # Invalid episodes stay in the denominator and shaping score. Never select
    # a model using test rows or a success-only subset.
    return (sum(row['result']['success'] is True for row in rows),
            sum(float(row['result']['shaping_reward']) for row in rows))


def model_rows(model, episodes):
    rows = []
    for episode in episodes:
        action = study.infer_model(model, episode['scene'])
        row = study.evaluate_action(episode['scene'], action, episode['rollout_seed'])
        row.update(scene_id=episode['scene_id'], split=episode['split'],
                   layout_sha256=episode['layout_sha256'], role='frozen_ridge_prediction')
        rows.append(row)
    return rows


def run(base_scene, output, *, seed=20261030, train_scenes=24, validation_scenes=6,
        test_scenes=8, teacher_budget=12, oracle_budget=4, alphas=(.01, 1., 100.)):
    study._integer(train_scenes, 'train scenes', 4, 10000)
    study._integer(teacher_budget, 'teacher budget', 1, 1000)
    study._integer(oracle_budget, 'oracle budget', 1, 1000)
    if not alphas or len(alphas) > 20:
        raise ValueError('Require 1..20 regularization candidates')
    for alpha in alphas:
        if study._finite(alpha, 'regularization') <= 0:
            raise ValueError('Ridge regularization must be positive')
    start = time.monotonic()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    manifest = study.sample_splits(base_scene, seed, train_scenes, validation_scenes, test_scenes)
    write(output/'episodes.json', manifest)  # Frozen splits BEFORE labels/optimization.
    manifest_bytes = (output/'episodes.json').read_bytes()
    write(output/'experiment.json', {'schema': study.STUDY_SCHEMA, 'seed': seed,
        'teacher_budget': teacher_budget, 'held_out_oracle_budget': oracle_budget,
        'alphas': list(alphas), 'manifest_sha256': manifest['manifest_sha256'],
        'selection': study.SELECTION, 'data_origin': 'synthetic', 'physically_verified': False,
        'fixed_baseline': study.FIXED_ACTION, 'fixed_baseline_provenance': study.BASELINE_PROVENANCE,
        'training': 'supervised action regression from training-scene search; failures retained',
        'evaluation': 'frozen single matrix prediction; no optimization or adaptation in inference'})
    teachers, labels = [], []
    with (output/'training-candidates.jsonl').open('w') as stream:
        for index, episode in enumerate(manifest['splits']['train']):
            candidates, best = study.search_episode(episode, teacher_budget)
            for row in candidates:
                stream.write(json.dumps(row, sort_keys=True, allow_nan=False)+'\n')
            stream.flush()
            teacher = {'scene_id': episode['scene_id'], 'layout_sha256': episode['layout_sha256'],
                'observation': episode['observation'], 'candidates': len(candidates),
                'label_available': best is not None, 'selected': best}
            teachers.append(teacher)
            if best is not None:
                labels.append(teacher)
            progress('teacher', completed=index+1, total=train_scenes,
                     label_available=best is not None,
                     captured=False if best is None else best['result']['success'])
    write(output/'training-labels.json', teachers)
    if len(labels) < 4:
        write(output/'summary.json', {'status': 'insufficient_training_labels',
            'valid_label_scenes': len(labels), 'required': 4,
            'all_invalid_scenes': train_scenes-len(labels), 'physically_verified': False})
        raise ValueError('Fewer than four valid training labels; preserved attempts, no fabricated checkpoint')
    x = [row['observation'] for row in labels]
    y = [[row['selected']['action'][name] for name in study.ACTION_NAMES] for row in labels]
    validations = []
    for alpha in alphas:
        model = study.fit_ridge(x, y, alpha)
        rows = model_rows(model, manifest['splits']['validation'])
        validations.append({'regularization': alpha, 'model': model, 'rows': rows,
                            'summary': study.summarize(rows)})
        progress('validation', regularization=alpha, **study.summarize(rows))
    selected = max(validations, key=lambda row: model_score(row['rows']))
    write(output/'validation.json', validations)
    value = study.checkpoint(base_scene, manifest, selected['model'],
        training_scene_ids=[row['scene_id'] for row in labels],
        validation_selection={'criterion': study.SELECTION, 'regularization': selected['regularization'],
            'candidate_regularizations': list(alphas),
            'scene_ids': [row['scene_id'] for row in manifest['splits']['validation']],
            'validation_summary': selected['summary'], 'test_used_for_selection': False,
            'training_label_scenes': len(labels), 'teacher_failed_scenes': train_scenes-len(labels),
            'training_teacher_captures': sum(row['selected']['result']['success'] for row in labels)})
    checkpoint_path = output/'checkpoint.json'
    write(checkpoint_path, value)  # Weights and normalizer freeze BEFORE any test evaluation.
    frozen_bytes = checkpoint_path.read_bytes()
    frozen_sha = hashlib.sha256(frozen_bytes).hexdigest()
    loaded = json.loads(frozen_bytes)
    study.validate_checkpoint(loaded)
    predicted, fixed, oracle, oracle_candidates = [], [], [], []
    for index, episode in enumerate(manifest['splits']['test']):
        proposal = study.predict(loaded, episode['scene'])
        row = study.evaluate_action(episode['scene'], proposal['action'], episode['rollout_seed'])
        row.update(scene_id=episode['scene_id'], layout_sha256=episode['layout_sha256'],
                   split='test', role='frozen_ridge_prediction', checkpoint_id=value['checkpoint_id'])
        predicted.append(row)
        baseline = study.evaluate_action(episode['scene'], study.FIXED_ACTION, episode['rollout_seed'])
        baseline.update(scene_id=episode['scene_id'], layout_sha256=episode['layout_sha256'],
                        split='test', role='fixed_action_baseline')
        fixed.append(baseline)
        candidates, best = study.search_episode(episode, oracle_budget, role='held_out_oracle')
        oracle_candidates.extend(candidates)
        # Every held-out layout stays in the denominator, even if no valid search result exists.
        oracle.append(best if best is not None else candidates[0])
        progress('test', completed=index+1, total=test_scenes,
                 frozen_capture=row['result']['success'], fixed_capture=baseline['result']['success'],
                 oracle_capture=oracle[-1]['result']['success'])
    if checkpoint_path.read_bytes() != frozen_bytes or (output/'episodes.json').read_bytes() != manifest_bytes:
        raise RuntimeError('Frozen checkpoint or split manifest changed during evaluation')
    write(output/'test.json', {'frozen_policy': predicted, 'fixed_baseline': fixed,
                             'held_out_search_oracle': oracle, 'oracle_candidates': oracle_candidates})
    first = manifest['splits']['test'][0]
    write(output/'preview-scene.json', {'scene': first['scene'], 'scene_id': first['scene_id'],
        'split': 'test', 'seed': first['rollout_seed'], 'layout_sha256': first['layout_sha256']})
    numerical_preview = study.preview(loaded, first['scene'], first['rollout_seed'])
    write(output/'preview.json', numerical_preview)
    summary = {'schema': study.STUDY_SCHEMA, 'status': 'complete',
        'checkpoint_id': value['checkpoint_id'], 'checkpoint_file_sha256': frozen_sha,
        'manifest_sha256': manifest['manifest_sha256'],
        'scoring_version': study.SCORING_VERSION, 'physics_version': study.PHYSICS_VERSION,
        'train_scenes': train_scenes, 'validation_scenes': validation_scenes, 'test_scenes': test_scenes,
        'teacher_budget': teacher_budget, 'teacher_labels': len(labels),
        'teacher_failed_scenes': train_scenes-len(labels),
        'teacher_captures': sum(row['selected']['result']['success'] for row in labels),
        'regularization': selected['regularization'], 'validation': selected['summary'],
        'test': {'frozen_policy': study.summarize(predicted), 'fixed_baseline': study.summarize(fixed),
                 'held_out_search_oracle': study.summarize(oracle)},
        'oracle_budget_per_scene': oracle_budget,
        'checkpoint_unchanged_after_test': True, 'split_manifest_unchanged_after_test': True,
        'elapsed_seconds': time.monotonic()-start,
        'preview_seed': first['rollout_seed'],
        'limits': ['narrow synthetic layout distribution; fixed provisional dynamics',
            'ridge imitation learns teacher actions, not guaranteed successful labels',
            'held-out oracle searches each scene and is reported separately; it is not policy inference',
            'endpoint IK only; native v2 replay adapter missing; no physical validation'],
        'data_origin': 'synthetic', 'physically_verified': False, 'native_paths_checked': False}
    write(output/'summary.json', summary)
    progress('complete', **summary)
    return value, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scene', type=Path, default=study.ROOT/'scene.json')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=20261030)
    parser.add_argument('--train-scenes', type=int, default=24)
    parser.add_argument('--validation-scenes', type=int, default=6)
    parser.add_argument('--test-scenes', type=int, default=8)
    parser.add_argument('--teacher-budget', type=int, default=12)
    parser.add_argument('--oracle-budget', type=int, default=4)
    parser.add_argument('--alphas', type=float, nargs='+', default=[.01, 1., 100.])
    args = parser.parse_args()
    run(json.loads(args.scene.read_text()), args.output, seed=args.seed,
        train_scenes=args.train_scenes, validation_scenes=args.validation_scenes,
        test_scenes=args.test_scenes, teacher_budget=args.teacher_budget,
        oracle_budget=args.oracle_budget, alphas=args.alphas)


if __name__ == '__main__':
    main()
