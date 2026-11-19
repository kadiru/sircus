"""One exact requested shallow-board shot, four far targets; no RL training."""
import copy

import numpy as np

from . import lowboard_contract as prior

VERSION = 'sircus-shallow-board-candidate-v1'
RECORD_PREFIX = 'shallow-shot-'
SEED = 20261208
SEED_IDS = prior.SEED_IDS
SUBSTEPS = prior.SUBSTEPS
BOUNDS = copy.deepcopy(prior.BOUNDS)
HEIGHT = .380
TILT = 28.
GAP = .3868  # Exact user-requested rounded value, not the earlier CPU-only .386824338… proposal.
MAX_REQUESTS = 424  # 24 controls +4 discovery +4*(3 nominal +30 fresh)*3 resolutions.
SCOPE = ('Exact 380 mm /28 degree /386.8 mm gap DEVELOPMENT experiment at four previously studied '
    'far bucket positions; privileged geometry and prescribed held endpoints. No new RL, '
    'held-out layout performance, arm transit, actuator, perception or physical validation.')

targets = prior.targets
control_target = prior.control_target
control_plan = prior.control_plan
source_anchor = prior.source_anchor
reach_limit = prior.reach_limit
summary = prior.summary
sensitivity = prior.sensitivity


def contexts(seeds, target_rows):
    controls = [c for c in prior.contexts(seeds, []) if c['stage'] == 'control']
    seed = source_anchor(seeds, HEIGHT, TILT)
    return controls + [dict(id=t['id']+':exact-candidate',
        **{k:copy.deepcopy(t[k]) for k in ('distance_id','ray_id','distance_m','board_xy_m','bucket_xy_m')},
        source_anchor_id=seed['id'], source_family=seed['id'], board_height_m=HEIGHT,
        tilt_deg=TILT, proposed_anchor_gap_m=GAP, stage='shallow') for t in target_rows]


def coarse_plan(context_rows, scene=None, seeds=None):
    if scene is None or seeds is None:
        raise ValueError('Exact parent scene and seeds required')
    release = HEIGHT + GAP + scene['ball']['radius_m']
    headroom = reach_limit(scene, seeds)['ball_center_height_m'] - release
    return [dict(context_id=c['id'], distance_id=c['distance_id'], ray_id=c['ray_id'],
        distance_m=c['distance_m'], source_anchor_id=c['source_anchor_id'], role='development',
        phase='coarse', gap_m=GAP, release_headroom_m=headroom,
        requested_ball_center_height_m=release) for c in context_rows if c['stage']=='shallow']


def refinement_plan(context_rows, records):
    if any(r['role'] != 'development' or r['phase'] != 'coarse' for r in records):
        raise ValueError('Only the four exact discovery requests may precede repeats')
    return []  # No adaptive changes to the requested candidate in this experiment.


def select_recipes(records, target_rows):
    recipes = prior.select_recipes(records, target_rows)
    if len(recipes) > 4 or len({r['distance_id'] for r in recipes}) != len(recipes):
        raise ValueError('At most one actually captured exact candidate per target')
    for recipe in recipes:
        if recipe['source']['values'][2:] != [HEIGHT,TILT,GAP]:
            raise ValueError('Only exact requested candidate can qualify for repeats')
        recipe['selection'] = 'Actual captured exact requested candidate; no search or parameter refinement'
    return recipes


def repeat_plan(recipes):
    rng = np.random.default_rng(SEED)
    result = []
    for recipe in recipes:
        for i, noise in enumerate(np.r_[np.zeros((3,3)),rng.normal(0,[.001,.25,.002],(30,3))]):
            values=np.asarray(recipe['source']['values']).copy(); values[2:]+=noise
            result.append(dict(repeat_id=f"{recipe['id']}-repeat-{i:02d}",recipe_id=recipe['id'],
                context_id=f"{recipe['id']}-local-{i:02d}",role='engineering_repeat',
                phase='nominal' if i<3 else 'fresh',repeat_index=i,noise=noise.tolist(),values=values.tolist(),
                board_height_m=float(values[2]),tilt_deg=float(values[3]),gap_m=float(values[4]),
                **{k:copy.deepcopy(recipe[k]) for k in ('distance_id','ray_id','distance_m','bucket_xy_m','board_xy_m')}))
    return result


def settings(scene, seeds):
    result=prior.settings(scene,seeds)
    release=HEIGHT+GAP+scene['ball']['radius_m']
    result.update(version=VERSION,seed=SEED,scope=SCOPE,
        candidate=dict(board_height_m=HEIGHT,tilt_deg=TILT,gap_m=GAP,release_center_height_m=release,
            ball_radius_m=scene['ball']['radius_m'],release_headroom_m=reach_limit(scene,seeds)['ball_center_height_m']-release,
            gap_definition='Vertical clearance from released ball bottom to board-center height',
            input_precision='Exact 386.8 mm user input; not the earlier 386.824338 mm CPU proposal'),
        grid=dict(board_heights_m=[HEIGHT],tilts_deg=[TILT],
            release_headroom_m=[reach_limit(scene,seeds)['ball_center_height_m']-release],
            source_anchor_selection='Nearest normalized H/40mm and tilt/4deg, then stable source ID; no outcome labels'),
        search=dict(control_contexts=8,coarse_contexts=4,coarse_requests=4,
            refinement_per_captured_context=0,maximum_recipes=4),
        budgets=dict(controls=24,coarse=4,refinement_max=0,repeats_max=396,maximum_requests=MAX_REQUESTS))
    return result
