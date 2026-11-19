"""Frozen lower-board/far-bucket DEVELOPMENT search; no policy training."""
import copy
import numpy as np

from . import distance_contract as parent

VERSION = 'sircus-low-board-far-development-v1'
SEED = 20261207
SEED_IDS = parent.SEED_IDS
SUBSTEPS = parent.SUBSTEPS
BOUNDS = copy.deepcopy(parent.BOUNDS)
HEIGHTS = tuple(i / 1000 for i in range(220, 361, 20))
TILTS = (28., 32., 36., 40.)
HEADROOM = (.005, .015, .030, .050, .080)
MAX_REQUESTS = 1968
SCOPE = ('Lower-board/moderate-tilt DEVELOPMENT search at four previously studied far bucket positions; '
    'privileged geometry and prescribed endpoints. No new RL, held-out layout performance, '
    'continuous success interval, arm transit, actuator or physical validation.')


def reach_limit(scene, seeds):
    xy = np.asarray(seeds[0]['values'][:2], float)
    if tuple(s['id'] for s in seeds) != SEED_IDS or any(s['values'][:2] != xy.tolist() for s in seeds):
        raise ValueError('Eight exact shared-XY parent geometry seeds required')
    radial = .842 ** 2 - float(xy @ xy)
    if radial <= 0 or not np.isfinite(radial):
        raise ValueError('Board XY lies outside the unchanged dropper outer prior')
    ball = float(.364 + np.sqrt(radial) - .095 - .09)
    return dict(ball_center_height_m=ball, board_height_plus_gap_m=ball-scene['ball']['radius_m'],
        wrist_radius_m=.842, shoulder_height_m=.364, flange_above_ball_m=.095,
        prior_wrist_offset_m=.09,
        formula='Zball_max = 0.364 + sqrt(0.842^2 - X^2 - Y^2) - 0.095 - 0.09',
        scope='Unchanged cheap spherical workspace prior only; native IK and strict clearance are separate checks')


def targets(scene, seeds):
    rays, rows = parent.targets(scene, seeds)
    chosen = [copy.deepcopy(t) for t in rows if t['id'] in ('corner-d01', 'corner-d02', 'corner-d03', 'corner-d04')]
    if len(chosen) != 4 or not all(t['bucket_on_table'] for t in chosen):
        raise ValueError('Four exact supported parent corner targets required')
    ray = copy.deepcopy(next(r for r in rays if r['id'] == 'corner'))
    ray['distances_m'] = [r['distance_m'] for r in chosen]
    return [ray], chosen


def control_target(seeds):
    board, bucket = seeds[0]['values'][:2], seeds[0]['bucket_xy_m']
    return dict(id='parent-control', distance_id='parent-control', ray_id='original_control',
        distance_m=float(np.linalg.norm(np.asarray(bucket)-board)), board_xy_m=copy.deepcopy(board),
        bucket_xy_m=copy.deepcopy(bucket), scope='Exact source controls; excluded from far-target search tables')


def source_anchor(seeds, height, tilt):
    # Geometry-only assignment is frozen before new shot outcomes.
    return min(seeds, key=lambda s: (((s['values'][2]-height)/.04)**2 +
        ((s['values'][3]-tilt)/4.)**2, s['id']))


def contexts(seeds, target_rows):
    result = []
    control = control_target(seeds)
    for seed in seeds:
        result.append(dict(id='control:'+seed['id'], distance_id=control['id'], ray_id=control['ray_id'],
            distance_m=control['distance_m'], board_xy_m=control['board_xy_m'], bucket_xy_m=control['bucket_xy_m'],
            source_anchor_id=seed['id'], source_family=seed['id'], board_height_m=seed['values'][2],
            tilt_deg=seed['values'][3], proposed_anchor_gap_m=seed['values'][4], stage='control'))
    for target in target_rows:
        for hi, height in enumerate(HEIGHTS):
            for ti, tilt in enumerate(TILTS):
                seed = source_anchor(seeds, height, tilt)
                result.append(dict(id=f"{target['id']}:h{hi:02d}-t{ti:02d}",
                    **{k:copy.deepcopy(target[k]) for k in ('distance_id','ray_id','distance_m','board_xy_m','bucket_xy_m')},
                    source_anchor_id=seed['id'], source_family=seed['id'], board_height_m=height,
                    tilt_deg=tilt, proposed_anchor_gap_m=seed['values'][4], stage='lowboard'))
    return result


def control_plan(seeds, target_rows):
    target = control_target(seeds)
    return [dict(role='control', phase='controls', source_anchor_id=s['id'], context_id='control:'+s['id'],
        distance_id=target['id'], ray_id=target['ray_id'], distance_m=target['distance_m'],
        gap_m=s['values'][4], repeat_index=i) for s in seeds for i in range(3)]


def coarse_plan(context_rows, scene=None, seeds=None):
    if scene is None or seeds is None:
        raise ValueError('Exact scene and source seeds required to derive the reach ceiling')
    limit = reach_limit(scene, seeds)['board_height_plus_gap_m']
    return [dict(context_id=c['id'], distance_id=c['distance_id'], ray_id=c['ray_id'],
        distance_m=c['distance_m'], source_anchor_id=c['source_anchor_id'], role='development', phase='coarse',
        gap_m=float(limit-c['board_height_m']-margin), release_headroom_m=margin,
        requested_ball_center_height_m=float(limit+scene['ball']['radius_m']-margin))
        for c in context_rows if c['stage']=='lowboard' for margin in HEADROOM]


def refinement_plan(context_rows, records):
    return parent.refinement_plan(context_rows, records)


def select_recipes(records, target_rows):
    result = parent.select_recipes(records, target_rows)
    if len(result) > 8:
        raise ValueError('At most two actually captured recipes per far target')
    return result


def repeat_plan(recipes):
    rng = np.random.default_rng(SEED)
    result = []
    for recipe in recipes:
        for i, noise in enumerate(np.r_[np.zeros((3,3)), rng.normal(0, parent.NOISE, (30,3))]):
            values = np.asarray(recipe['source']['values']).copy(); values[2:] += noise
            result.append(dict(repeat_id=f"{recipe['id']}-repeat-{i:02d}", recipe_id=recipe['id'],
                context_id=f"{recipe['id']}-local-{i:02d}", role='engineering_repeat',
                phase='nominal' if i<3 else 'fresh', repeat_index=i, noise=noise.tolist(), values=values.tolist(),
                board_height_m=float(values[2]), tilt_deg=float(values[3]), gap_m=float(values[4]),
                **{k:copy.deepcopy(recipe[k]) for k in ('distance_id','ray_id','distance_m','bucket_xy_m','board_xy_m')}))
    return result


summary = parent.summary
sensitivity = parent.sensitivity


def settings(scene, seeds):
    rays, rows = targets(scene, seeds)
    return dict(version=VERSION, seed=SEED, board_xy_m=seeds[0]['values'][:2], rays=rays, targets=rows,
        control_target=control_target(seeds), table_bounds_xy_m=parent.table_bounds(scene).tolist(),
        table_bucket_margin_m=.005, bucket_outer_radius_m=scene['bucket']['outer_top_diameter_m']/2,
        bounds=BOUNDS, substeps=list(SUBSTEPS), reach_limit=reach_limit(scene,seeds),
        grid=dict(board_heights_m=list(HEIGHTS), tilts_deg=list(TILTS), release_headroom_m=list(HEADROOM),
            source_anchor_selection='Nearest normalized H/40mm and tilt/4deg, then stable source ID; no outcome labels'),
        height_noise_m=.001, tilt_noise_deg=.25, gap_noise_m=.002,
        repeatability=dict(nominal=3, fresh=30, noise_std=parent.NOISE.tolist(), same_requests_and_joints=True,
            clipping=False, retry=False, max_local_joint_delta_rad=.10),
        search=dict(control_contexts=8, coarse_contexts=128, coarse_requests=640,
            refinement_per_captured_context=4, maximum_recipes=8),
        gates=parent.settings(scene,seeds)['gates'],
        budgets=dict(controls=24, coarse=640, refinement_max=512, repeats_max=792, maximum_requests=MAX_REQUESTS),
        scope=SCOPE)
