"""Frozen development-only local height/tilt/gap experiment; no policy training."""
from collections import defaultdict
import copy
import numpy as np

from .window_contract import BOUNDS
from .learning_contract import digest, reward, statistics
from .admission import REFERENCE_BUCKET, REFERENCE_VALUES

VERSION = 'sircus-local-height-tilt-sensitivity-v1'
SEED = 20261205
SUBSTEPS = (32,64,128)
NOISE = np.asarray([.001,.25,.002])  # height m, tilt deg, gap m
MAX_REQUESTS = 27 + 75*17 + 75*6 + 12*23*3
ANCHORS = (
    dict(id='anchor397', height=.3973878458850598,tilt=36.154791124080255,gap=.32121875,
         study='confirmation',record_id='confirmation-023'),
    dict(id='anchor500', height=.5,tilt=36.154791124080255,gap=.23853124999999997,
         study='confirmation',record_id='confirmation-069'),
    dict(id='anchor440', height=.44,tilt=32.,gap=.28578125000000004,
         study='comparison',record_id='window-shot-001797'))
SCOPE = ('Finite local DEVELOPMENT search at one fixed bucket and board XY; privileged geometry, '
         'bucket-facing board, prescribed endpoints. No new RL, unseen-layout evaluation, '
         'actuator tracking, continuous interval guarantee or physical calibration.')


def contexts():
    rows=[]
    for anchor in ANCHORS:
        for i,dh in enumerate((-.04,-.02,0.,.02,.04)):
            for j,dt in enumerate((-4.,-2.,0.,2.,4.)):
                rows.append(dict(id=f"{anchor['id']}-h{i:02d}-t{j:02d}",source_family=anchor['id'],
                    board_height_m=float(anchor['height']+dh),tilt_deg=float(anchor['tilt']+dt),
                    proposed_anchor_gap_m=anchor['gap']))
    assert len(rows)==len({(r['board_height_m'],r['tilt_deg']) for r in rows})==75
    return rows


def coarse_plan():
    return [dict(context_id=c['id'],source_family=c['source_family'],role='development',phase='coarse',
        gap_m=float(c['proposed_anchor_gap_m']+offset)) for c in contexts() for offset in np.linspace(-.08,.08,17)]


def settings():
    return dict(version=VERSION,seed=SEED,anchors=list(ANCHORS),contexts=contexts(),bounds=BOUNDS,
        bucket_xy_m=list(REFERENCE_BUCKET),board_xy_m=list(REFERENCE_VALUES[:2]),substeps=list(SUBSTEPS),
        gap_noise_m=.002,height_noise_m=.001,tilt_noise_deg=.25,
        coarse_gaps=dict(count=17,relative_offsets_m=[-.08,.08],spacing_m=.01),
        refinement=dict(queries_per_captured_context=6,maximum=450,source='coarse DEVELOPMENT only',
            rule='Three transition/promising midpoints then broadest remaining intervals; no monotonicity assumption'),
        selection=dict(maximum_recipes=12,mandatory_anchors=3,additional_distinct_contexts=9,
            rule='Known controls plus actual observed captured gap nearest widest sampled hit-run center; '
                 'farthest height/tilt context from selected set, then clean quality, run width, source ID'),
        repeatability=dict(nominal=3,fresh=20,noise_std=NOISE.tolist(),noise_order=['height_m','tilt_deg','gap_m'],
            paired_substeps=list(SUBSTEPS),same_exact_joint_arrays=True,max_local_joint_deviation_rad=.10,
            clipping=False,retry=False,unsupported_requests='Retained as failed requested attempts'),
        sensitivity=dict(observation_hz=240,substeps=list(SUBSTEPS),
            paired_gate='3/3 nominal and >=18/20 fresh captures at both64/128, >=21/23 capture-status agreements, '
                'maximum nominal position difference at350ms <=5mm. Finite engineering gate, not proof of convergence.'),
        budgets=dict(controls=27,coarse=1275,refinement_max=450,paired_repeat_max=828,maximum_requests=MAX_REQUESTS),
        scoring='Unchanged board rebound + whole sphere entry + one continuous second inside; rim assistance counts; external assistance fails',
        scope=SCOPE)


def refine_gaps(rows):
    if any(r.get('role')!='development' or r.get('phase')!='coarse' for r in rows):
        raise ValueError('Refinement may inspect only coarse development outcomes')
    if not any(r.get('native_release') and r['route_success'] for r in rows):return []
    by_gap={float(r['gap_m']):r for r in rows};values=sorted(by_gap)
    intervals=[]
    for lo,hi in zip(values,values[1:]):
        left,right=by_gap[lo],by_gap[hi]
        priority=(bool(left['route_success'])!=bool(right['route_success']),
            max(reward(left),reward(right)),hi-lo,-lo)
        intervals.append((priority,(lo+hi)/2))
    promising=[v for _,v in sorted(intervals,reverse=True)[:3]]
    broad=[v for _,v in sorted(intervals,key=lambda pair:(pair[0][2],pair[0][3]),reverse=True)]
    result=[]
    for gap in promising+broad:
        if gap not in result:result.append(gap)
        if len(result)==6:return result
    raise ValueError('Insufficient distinct refinement intervals')


def _best_context_row(records):
    # Choose an ACTUALLY captured sample, not an untested interval midpoint.
    ordered=sorted(records,key=lambda r:r['gap_m']);runs=[];current=[]
    for row in ordered+[dict(route_success=False)]:
        if row['route_success'] and not row.get('invalid'):current.append(row)
        elif current:runs.append(current);current=[]
    candidates=[]
    for run in runs:
        width=run[-1]['gap_m']-run[0]['gap_m'];center=(run[-1]['gap_m']+run[0]['gap_m'])/2
        chosen=min(run,key=lambda r:(abs(r['gap_m']-center),-r['route_quality'],-reward(r),r['id']))
        candidates.append((width,chosen))
    if not candidates:return None
    width,row=max(candidates,key=lambda pair:(pair[0],pair[1]['route_quality'],reward(pair[1]),pair[1]['id']))
    return dict(source=copy.deepcopy(row),sampled_run_width_m=float(width))


def select_recipes(rows):
    if any(r.get('role') not in ('development','control') for r in rows):raise ValueError('No held-out outcomes may select recipes')
    selected=[];used=set()
    for anchor in ANCHORS:
        row=next(r for r in rows if r['role']=='control' and r.get('substeps')==64 and r.get('anchor_id')==anchor['id'] and r['route_success'])
        selected.append(dict(source=copy.deepcopy(row),sampled_run_width_m=None,selection='prespecified_anchor'));used.add(row['context_id'])
    grouped=defaultdict(list)
    for row in rows:
        if row['role']=='development':grouped[row['context_id']].append(row)
    candidates=[v for name,records in grouped.items() if name not in used and (v:=_best_context_row(records))]
    def normalized(item):return np.asarray(item['source']['values'][2:4])/[.04,4.]
    while candidates and len(selected)<12:
        def rank(candidate):
            nearest=min(float(np.linalg.norm(normalized(candidate)-normalized(s))) for s in selected)
            r=candidate['source'];return (nearest,r['route_quality'],candidate['sampled_run_width_m'],r['id'])
        chosen=max(candidates,key=rank);chosen['selection']='diverse_observed_capture';selected.append(chosen);candidates.remove(chosen)
    for i,item in enumerate(selected):
        source=item['source'];item.update(id=f'recipe-{i:02d}',context_id=source['context_id'],
            board_height_m=source['values'][2],tilt_deg=source['values'][3],gap_m=source['gap_m'])
    return selected


def repeat_plan(recipes):
    rng=np.random.default_rng(SEED);result=[]
    for recipe in recipes:
        for index,noise in enumerate(np.r_[np.zeros((3,3)),rng.normal(0,NOISE,(20,3))]):
            values=np.asarray(recipe['source']['values'],float).copy();values[2:]+=noise
            result.append(dict(repeat_id=f"{recipe['id']}-repeat-{index:02d}",recipe_id=recipe['id'],
                context_id=f"{recipe['id']}-local-{index:02d}",role='engineering_repeat',
                phase='nominal' if index<3 else 'fresh',repeat_index=index,noise=noise.tolist(),
                values=values.tolist(),board_height_m=float(values[2]),tilt_deg=float(values[3]),gap_m=float(values[4])))
    return result


def summary(rows):
    return dict(**statistics(rows),native_releases=sum(bool(r.get('native_release')) for r in rows))


def sensitivity(rows,positions):
    grouped=defaultdict(dict)
    for row in rows:grouped[row['repeat_id']][int(row['substeps'])]=row
    pairs=[values for values in grouped.values() if 64 in values and 128 in values]
    agree=sum(bool(v[64]['route_success'])==bool(v[128]['route_success']) for v in pairs)
    delta=[float(np.linalg.norm(positions[v[64]['id']][83]-positions[v[128]['id']][83]))
        for v in pairs if v[64]['phase']=='nominal' and v[64]['native_release'] and v[128]['native_release']]
    fresh={step:[r for r in rows if r['substeps']==step and r['phase']=='fresh'] for step in (64,128)}
    nominal={step:[r for r in rows if r['substeps']==step and r['phase']=='nominal'] for step in (64,128)}
    passed=(len(pairs)==23 and agree>=21 and len(delta)==3 and max(delta)<=.005
        and all(len(nominal[s])==3 and sum(bool(r['route_success']) for r in nominal[s])==3
            and len(fresh[s])==20 and sum(bool(r['route_success']) for r in fresh[s])>=18 for s in (64,128)))
    return dict(paired_attempts=len(pairs),capture_agreement_64_128=agree,
        nominal_position_delta_350ms_max_m=max(delta) if delta else None,paired_engineering_gate=passed,
        scope='Paired identical requests/joints and finite nominal350ms check; not mathematical convergence or physical calibration')
