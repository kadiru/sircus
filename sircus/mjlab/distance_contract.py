"""Prespecified seeded distance development search; no policy training."""
from collections import defaultdict
import copy
import numpy as np

from .learning_contract import reward, digest, statistics
from .window_contract import BOUNDS
from .neighborhood_contract import _best_context_row

VERSION='sircus-bucket-distance-development-v1'
SEED=20261206
SUBSTEPS=(32,64,128)
SEED_IDS=('recipe-00','recipe-02','recipe-03','recipe-04','recipe-08','recipe-09','recipe-10','recipe-11')
EDGE_MARGIN_M=.005
NOISE=np.asarray([.001,.25,.002])
MAX_REQUESTS=5589
SCOPE=('Seeded DEVELOPMENT search along two separately labeled bucket rays; privileged geometry and prescribed held endpoints. '
       'No new RL, unseen-layout policy evaluation, continuous success interval, transit/actuator or physical claim.')


def table_bounds(scene):
    center=np.asarray(scene['table']['center_m'][:2],float);size=np.asarray(scene['table']['size_m'][:2],float)
    return np.c_[center-size/2,center+size/2]


def targets(scene,seeds):
    if tuple(s['id'] for s in seeds)!=SEED_IDS:raise ValueError('Eight frozen verified seeds in declared order required')
    board=np.asarray(seeds[0]['values'][:2],float);reference=np.asarray(seeds[0]['bucket_xy_m'],float)
    if any(s['values'][:2]!=board.tolist() or s['bucket_xy_m']!=reference.tolist() for s in seeds):raise ValueError('Shared seed XY required')
    radius=scene['bucket']['outer_top_diameter_m']/2;bounds=table_bounds(scene)
    original=reference-board;dref=float(np.linalg.norm(original));original/=dref
    corner=bounds[:,0]+radius+EDGE_MARGIN_M-board;corner/=np.linalg.norm(corner)
    definitions=[('original','Original bucket direction',original,[.35,.45,dref,.55,.60,.64,.65]),
                 ('corner','Diagonal toward supported table corner',corner,[.55,.65,.70,.75,.77,.80,.95])]
    rays=[];rows=[]
    for ray_id,label,unit,distances in definitions:
        rays.append(dict(id=ray_id,label=label,unit_xy=unit.tolist(),heading_deg=float(np.rad2deg(np.arctan2(unit[1],unit[0]))),distances_m=distances))
        for index,distance in enumerate(distances):
            xy=reference.copy() if ray_id=='original' and index==2 else board+unit*distance
            actual=float(min(*(xy-bounds[:,0]-radius),*(bounds[:,1]-xy-radius)))
            rows.append(dict(id=f'{ray_id}-d{index:02d}',distance_id=f'{ray_id}-d{index:02d}',ray_id=ray_id,
                distance_m=float(distance),board_xy_m=board.tolist(),bucket_xy_m=xy.tolist(),
                bucket_on_table=bool(actual>=EDGE_MARGIN_M-1e-12),table_margin_m=actual,
                reason=None if actual>=EDGE_MARGIN_M-1e-12 else 'bucket_footprint_outside_table'))
    if len(rows)!=14 or sum(t['bucket_on_table'] and t['distance_m']>=.65 for t in rows)!=4:
        raise ValueError('This version fixes the current provisional table and four supported far targets')
    return rays,rows


def contexts(seeds,target_rows):
    return [dict(id=f"{t['id']}:{s['id']}",distance_id=t['id'],ray_id=t['ray_id'],distance_m=t['distance_m'],
        board_xy_m=t['board_xy_m'],bucket_xy_m=t['bucket_xy_m'],source_anchor_id=s['id'],source_family=s['id'],
        board_height_m=s['values'][2],tilt_deg=s['values'][3],proposed_anchor_gap_m=s['values'][4],stage='coarse')
        for t in target_rows for s in seeds]


def coarse_plan(context_rows):
    return [dict(context_id=c['id'],distance_id=c['distance_id'],ray_id=c['ray_id'],distance_m=c['distance_m'],
        source_anchor_id=c['source_anchor_id'],role='development',phase='coarse',gap_m=float(c['proposed_anchor_gap_m']+dg))
        for c in context_rows for dg in np.linspace(-.08,.24,17)]


def control_plan(seeds,target_rows):
    target=next(t for t in target_rows if t['id']=='original-d02')
    return [dict(role='control',phase='controls',source_anchor_id=s['id'],context_id=f"{target['id']}:{s['id']}",
        distance_id=target['id'],ray_id=target['ray_id'],distance_m=target['distance_m'],gap_m=s['values'][4],repeat_index=i)
        for s in seeds for i in range(3)]


def extension_contexts(seeds,target_rows,coarse_rows):
    if any(r['phase']!='coarse' or r['role']!='development' for r in coarse_rows):raise ValueError('Coarse DEV outcomes only')
    result=[]
    for t in target_rows:
        captured={r['context_id'] for r in coarse_rows if r['distance_id']==t['id'] and r['route_success']}
        if not t['bucket_on_table'] or t['distance_m']<.65 or len(captured)>=2:continue
        for s in seeds:
            for i,(dh,dt) in enumerate(((-.08,-12.),(0.,-12.),(.04,-8.))):
                result.append(dict(id=f"{t['id']}:{s['id']}:extend{i}",distance_id=t['id'],ray_id=t['ray_id'],distance_m=t['distance_m'],
                    board_xy_m=t['board_xy_m'],bucket_xy_m=t['bucket_xy_m'],source_anchor_id=s['id'],source_family=s['id'],
                    board_height_m=float(s['values'][2]+dh),tilt_deg=float(s['values'][3]+dt),
                    proposed_anchor_gap_m=s['values'][4],stage='far_extension'))
    return result


def extension_plan(context_rows):
    return [dict(context_id=c['id'],distance_id=c['distance_id'],ray_id=c['ray_id'],distance_m=c['distance_m'],
        source_anchor_id=c['source_anchor_id'],role='development',phase='far_extension',gap_m=float(gap))
        for c in context_rows for gap in np.linspace(.12,.64,14)]


def refinement_plan(context_rows,records):
    if any(r['role']!='development' or r['phase'] not in ('coarse','far_extension') for r in records):raise ValueError('Initial DEV outcomes only')
    result=[]
    for c in context_rows:
        rows=sorted([r for r in records if r['context_id']==c['id']],key=lambda r:r['gap_m'])
        if not any(r['route_success'] for r in rows):continue
        candidates=[]
        for lo,hi in zip(rows,rows[1:]):
            priority=(bool(lo['route_success'])!=bool(hi['route_success']),max(reward(lo),reward(hi)),hi['gap_m']-lo['gap_m'],-lo['gap_m'])
            candidates.append((priority,(hi['gap_m']+lo['gap_m'])/2))
        for _,gap in sorted(candidates,reverse=True)[:4]:
            result.append(dict(context_id=c['id'],distance_id=c['distance_id'],ray_id=c['ray_id'],distance_m=c['distance_m'],
                source_anchor_id=c['source_anchor_id'],role='development',phase='refinement',gap_m=float(gap)))
    return result


def select_recipes(records,target_rows):
    if any(r['role']!='development' for r in records):raise ValueError('Only search DEV rows may select recipes')
    result=[]
    for target in target_rows:
        grouped=defaultdict(list)
        for row in records:
            if row['distance_id']==target['id']:grouped[row['context_id']].append(row)
        choices=[v for values in grouped.values() if (v:=_best_context_row(values))]
        selected=[]
        for _ in range(2 if target['distance_m']>=.65 else 1):
            if not choices:break
            def rank(item):
                r=item['source'];point=np.asarray(r['values'][2:4])/[.04,4.]
                diversity=min(float(np.linalg.norm(point-np.asarray(x['source']['values'][2:4])/[.04,4.])) for x in selected) if selected else 0.
                return (diversity,item['sampled_run_width_m'],r['route_quality'],reward(r),r['id'])
            chosen=max(choices,key=rank);choices.remove(chosen);selected.append(chosen)
        for item in selected:
            row=item['source'];item.update(id=f'recipe-{len(result):02d}',distance_id=target['id'],ray_id=target['ray_id'],
                distance_m=target['distance_m'],bucket_xy_m=target['bucket_xy_m'],board_xy_m=target['board_xy_m'],
                context_id=row['context_id'],board_height_m=row['values'][2],tilt_deg=row['values'][3],gap_m=row['gap_m'],
                selection='Actual captured sample near widest sampled run center; optional far-context diversity')
            result.append(item)
    if len(result)>15:raise ValueError('Finite selected recipe budget exceeded')
    return result


def repeat_plan(recipes):
    rng=np.random.default_rng(SEED);result=[]
    for recipe in recipes:
        for i,noise in enumerate(np.r_[np.zeros((3,3)),rng.normal(0,NOISE,(30,3))]):
            values=np.asarray(recipe['source']['values']).copy();values[2:]+=noise
            result.append(dict(repeat_id=f"{recipe['id']}-repeat-{i:02d}",recipe_id=recipe['id'],context_id=f"{recipe['id']}-local-{i:02d}",
                role='engineering_repeat',phase='nominal' if i<3 else 'fresh',repeat_index=i,noise=noise.tolist(),
                values=values.tolist(),board_height_m=float(values[2]),tilt_deg=float(values[3]),gap_m=float(values[4]),
                distance_id=recipe['distance_id'],ray_id=recipe['ray_id'],distance_m=recipe['distance_m'],
                bucket_xy_m=recipe['bucket_xy_m'],board_xy_m=recipe['board_xy_m']))
    return result


def sensitivity(rows,positions):
    groups=defaultdict(dict)
    for r in rows:groups[r['repeat_id']][r['substeps']]=r
    pairs=[v for v in groups.values() if 64 in v and 128 in v]
    agree=sum(bool(v[64]['route_success'])==bool(v[128]['route_success']) for v in pairs)
    delta=[float(np.linalg.norm(positions[v[64]['id']][83]-positions[v[128]['id']][83]))
        for v in pairs if v[64]['phase']=='nominal' and v[64]['native_release'] and v[128]['native_release']]
    def counts_ok(s):
        nominal=[r for r in rows if r['substeps']==s and r['phase']=='nominal'];fresh=[r for r in rows if r['substeps']==s and r['phase']=='fresh']
        return len(nominal)==3 and sum(bool(r['route_success']) for r in nominal)==3 and len(fresh)==30 and sum(bool(r['route_success']) for r in fresh)>=27
    fine=(len(pairs)==33 and agree>=30 and len(delta)==3 and max(delta)<=.005 and counts_ok(64) and counts_ok(128))
    return dict(paired_attempts=len(pairs),capture_agreement_64_128=agree,nominal_position_delta_350ms_max_m=max(delta) if delta else None,
        paired_engineering_gate=fine,all_resolution_repeat_gate=bool(fine and counts_ok(32)),
        scope='Prespecified all-resolution capture repeats plus fine-pair nominal350ms check; not trajectory convergence or physical calibration')


def summary(rows):return dict(**statistics(rows),native_releases=sum(bool(r.get('native_release')) for r in rows))


def settings(scene,seeds):
    rays,rows=targets(scene,seeds)
    return dict(version=VERSION,seed=SEED,board_xy_m=seeds[0]['values'][:2],rays=rays,targets=rows,
        table_bounds_xy_m=table_bounds(scene).tolist(),table_bucket_margin_m=EDGE_MARGIN_M,
        bucket_outer_radius_m=scene['bucket']['outer_top_diameter_m']/2,bounds=BOUNDS,
        substeps=list(SUBSTEPS),height_noise_m=.001,tilt_noise_deg=.25,gap_noise_m=.002,
        search=dict(coarse_contexts=112,coarse_gaps=17,coarse_offsets_m=[-.08,.24],coarse_requests=1904,
            far_extension_rule='Supported target>=650mm with fewer than2 coarse-captured contexts',
            extension_deltas_m_deg=[[-.08,-12.],[0.,-12.],[.04,-8.]],extension_gaps_m=np.linspace(.12,.64,14).tolist(),
            refinement_per_captured_context=4,maximum_contexts=208,maximum_recipes=15),
        repeatability=dict(nominal=3,fresh=30,noise_std=NOISE.tolist(),same_requests_and_joints=True,clipping=False,retry=False,max_local_joint_delta_rad=.10),
        gates=dict(fine_pair='3/3nominal and>=27/30fresh at64/128,>=30/33capture agreement,<=5mm nominal350mspositiondelta',
            all_resolution='Fine-pair gate plus3/3nominal and>=27/30fresh at32; no32position-convergence claim'),
        budgets=dict(controls=24,coarse=1904,extension_max=1344,refinement_max=832,repeats_max=1485,maximum_requests=MAX_REQUESTS),scope=SCOPE)
