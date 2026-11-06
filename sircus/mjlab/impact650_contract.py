"""Prespecified impact-offset development protocol; physical scoring is unchanged."""
import copy
import itertools
import math
from pathlib import Path

from . import tilt650_contract as prior
from . import xy650_contract as xy
from .learning_contract import digest
from .replay650_contract import sensitivity

VERSION = 'sircus-impact650-study-v1'
AUDIT_VERSION = 'sircus-impact650-independent-audit-v1'
RULE = 'sircus-board-center-plane-forward-lateral-impact-v1'
SOURCE_IDS = ('refinement-64-refine-s02-h1t0g2-r0',
              'coarse-64-xy-grid-0043-r0', 'coarse-64-xy-grid-0096-r0')
FORWARD = (-.04,-.03,-.02,-.01,0.,.01,.02,.03,.04)
LATERAL = (-.02,-.01,0.,.01,.02)
STEPS = (64,128)
MAX_REQUESTS = 189
SCOPE = ('Bounded impact-offset DEVELOPMENT search with board centers 650 mm from their buckets. '
         'Actual nominal impact-to-bucket horizontal distance varies with offset and is reported separately. '
         'Offsets are surface-tangent displacements of the nominal drop-ray intersection with the board center plane, '
         'not measured finite-ball surface contacts. Gap remains vertical ball-bottom clearance above that offset center-plane point. '
         'Calibrated simulated contacts and held articulated endpoints only; no new PPO, noise, grasp/transit, '
         'actuator tracking, perception, physical validation or held-out generalization. Historical evidence is unchanged.')
read = prior.read
statistics = prior.statistics
captured = prior.captured
clean = prior.clean
control_gate = prior.control_gate


def settings():
    return dict(source_ids=list(SOURCE_IDS),families=3,forward_m=list(FORWARD),lateral_m=list(LATERAL),
                coarse_requests=135,control_requests=24,verification_sources=5,repeats_per_resolution=3,
                verification_requests=30,max_requests=MAX_REQUESTS,user_request_cap=200,worker_wall_limit_s=3600,
                substeps=list(STEPS),minimum_clearance_m=.005,max_drift_350ms_m=.005,impact_rule=RULE,
                selection='Best native-valid unique target from each literal family in family order; then globally ranked distinct targets until five. Empty families skipped; no replacement requests.',
                ranking='Clean, capture, first-flight center-plane crossing, larger aperture margin, smaller pre-contact distance, stable record ID',
                gap_convention='Vertical ball-bottom clearance above the nominal offset point on the board CENTER plane; zero offset exactly preserves old releases.',
                distance_convention='Board-center to bucket stays 650 mm; nominal offset-ray point distance varies and must not be labeled a 650 mm flight.',scope=SCOPE)


def load_inputs(parent_study,parent_audit):
    parent,_,controls,bindings = prior.load_parent(parent_study,parent_audit)
    rows=[__import__('json').loads(line) for line in (Path(parent_study)/'records.jsonl').read_text().splitlines() if line]
    lookup={r['id']:r for r in rows}
    selected=[copy.deepcopy(lookup[ident]) for ident in SOURCE_IDS]
    for source in selected:
        if not (captured(source) and not source.get('invalid') and source.get('invalid_physics') is False
                and source['route_quality']==1 and abs(math.dist(source['values'][:2],source['bucket_xy_m'])-.65)<1e-12):
            raise ValueError('Exact valid parent rim-capture family required')
    if selected[0]['values'][2:] != [.38,27.5,.3918] and any(abs(a-b)>1e-12 for a,b in zip(selected[0]['values'][2:],[.38,27.5,.3918])):
        raise ValueError('Literal 27.5-degree family changed')
    if selected[1]['values'][2:] != [.38,30.,.41] or selected[2]['values'][2:] != [.38,30.,.39]:
        raise ValueError('Literal 30-degree families changed')
    return parent,selected,controls,bindings


def scene_fingerprint(parent):
    return digest(dict(physical_scene_fingerprint=parent['scene_fingerprint'],impact_rule=RULE))


def query_payload(group,scene_fp):
    return dict(scene_fingerprint=scene_fp,values=group['values'],bucket_xy_m=group['bucket_xy_m'],
                impact_offset_m=group['impact_offset_m'],impact_rule=RULE,
                source_record_id=group['source_record_id'],source_record_sha256=group['source_record_sha256'])


def bind(groups,scene_fp):
    for group in groups:group['query_fingerprint']=digest(query_payload(group,scene_fp))
    return groups


def controls(sources):
    return [dict(copy.deepcopy(g),family_id='control',seed_id=g['source_record_id'],
                 impact_offset_m=[0.,0.],query_kind='historical_control') for g in sources]


def coarse(sources):
    result=[]
    for family,source in enumerate(sources):
        for i,j in itertools.product(range(len(FORWARD)),range(len(LATERAL))):
            result.append(dict(id=f'impact-f{family:02d}-u{i}v{j}',role='discovery',phase='coarse',
                family_id=f'family-{family:02d}',seed_id=source['id'],query_kind='impact_grid',
                impact_offset_m=[FORWARD[i],LATERAL[j]],values=copy.deepcopy(source['values']),
                bucket_xy_m=copy.deepcopy(source['bucket_xy_m']),translation_m=copy.deepcopy(source['translation_m']),
                q10=copy.deepcopy(source['q10']),q14=copy.deepcopy(source['q14']),
                source_record_id=source['id'],source_record_sha256=digest(source),source_query_fingerprint=source['query_fingerprint']))
    return result


def target_key(row):
    return tuple(round(float(v),9) for v in row['values']+row['bucket_xy_m']+row['impact_offset_m'])


def select(rows,count=5):
    ordered=xy.ranked(rows,len(rows));chosen=[];seen=set()
    for family in ('family-00','family-01','family-02'):
        item=next((r for r in ordered if r['family_id']==family and target_key(r) not in seen),None)
        if item is not None:chosen.append(item);seen.add(target_key(item))
    for row in ordered:
        if len(chosen)>=count:break
        if target_key(row) not in seen:chosen.append(row);seen.add(target_key(row))
    return chosen[:count]


def verification(sources):
    groups=[]
    for i,source in enumerate(sources):
        groups.append(dict(id=f'impact-verify-{i:02d}',role='verification',phase='verification',freeze_joints=True,
            query_kind='nominal_verification',family_id=source['family_id'],seed_id=source['seed_id'],
            source_record_id=source['id'],source_record_sha256=digest(source),source_query_fingerprint=source['query_fingerprint'],
            source_route_quality=source.get('route_quality',0),
            **{k:copy.deepcopy(source[k]) for k in ('values','bucket_xy_m','impact_offset_m','q10','q14','translation_m')}))
    return groups


def plan(groups,phase,steps=(64,),repeats=1):
    rows=xy.plan(groups,phase,steps,repeats);lookup={g['id']:g for g in groups}
    for row in rows:
        group=lookup[row['group_id']]
        row.update({k:copy.deepcopy(group[k]) for k in ('family_id','seed_id','impact_offset_m','impact_point_world_m',
                    'impact_to_bucket_horizontal_m','release_center_m','release_center_height_m')})
    return rows


def execution_fingerprint(row):
    return digest({k:row[k] for k in ('id','query_fingerprint','substeps','repeat_index','q10','q14',
                  'native_initial_q10','native_initial_q14','native_initial_ball_center_m')})


def qualification(rows,positions,*,controls_passed,source_capture):
    value=prior.qualification(rows,positions,controls_passed=controls_passed,parent_source_capture=source_capture)
    value['source_capture']=value.pop('parent_source_capture')
    return value
