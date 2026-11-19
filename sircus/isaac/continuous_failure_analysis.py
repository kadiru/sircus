"""Post-run CPU failure taxonomy from independently audited native observations.

All attempted shots stay in capture-rate denominators. Surface attribution uses
measured filtered contact magnitudes, never inferred ball sounds or a hand-drawn
trajectory. Signed contact offsets describe locations, not causal short/long
labels. No simulator or policy optimizer is launched.
"""

import argparse
import collections
import json
import math
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from continuous_native_audit import digest, finite, require, safe_path
from rl_contract import ShotScore

SCHEMA = "sircus-continuous-native-failure-analysis-v1"
FORCES = ("board", "bucket", "table", "floor", "robot", "unattributed")
EXTERNAL = ("table", "floor", "robot", "unattributed")
RADIUS_EDGES = (.28, .40, .60, .80, 1.0, 1.20)
BUCKET_EDGES = (0., .30, .50, .70, .90, 2.0)
THRESHOLD = .01


def statistics(rows):
    attempts=len(rows);released=sum(not r['invalid'] for r in rows)
    captures=sum(bool(r['route_success']) for r in rows)
    return dict(attempts=attempts,released=released,invalid=attempts-released,captures=captures,
                capture_rate=captures/attempts if attempts else None,
                capture_rate_given_release=captures/released if released else None,
                board_bounces=sum(bool(r['board_bounce']) for r in rows),
                whole_ball_entries=sum(bool(r['whole_ball_entry']) for r in rows),
                raw_retained_captures=sum(bool(r['success']) for r in rows),
                clean_captures=sum(r['route_quality']==2 for r in rows),
                invalid_reasons=dict(collections.Counter(r['invalid'] for r in rows if r['invalid'])))


def binned(rows,key,edges):
    result=[]
    for index,(lo,hi) in enumerate(zip(edges[:-1],edges[1:])):
        group=[r for r in rows if lo<=r[key] and (r[key]<=hi if index==len(edges)-2 else r[key]<hi)]
        result.append(dict(lower_m=lo,upper_m=hi,upper_inclusive=index==len(edges)-2,**statistics(group)))
    require(sum(r['attempts'] for r in result)==len(rows),'Distance bin lost requested attempts')
    return result



def relative_timing(sample,board_sample):
    if sample is None:return 'not_observed'
    if board_sample is None:return 'no_board_contact_observed'
    return 'before_first_board_contact' if sample<board_sample else 'same_sample_as_first_board_contact' if sample==board_sample else 'after_first_board_contact'

def surface_events(forces,positions,board_xy,bucket_xy,end):
    """Measured score-window first/peak external surfaces; simultaneous ties kept."""
    forces=np.asarray(forces)[:end];positions=np.asarray(positions)[:end]
    require(forces.ndim==2 and forces.shape[1]==6 and len(forces)==len(positions),'Invalid contact packet')
    require(np.isfinite(forces).all() and (forces>=0).all(),'Invalid contact force magnitudes')
    hit=forces[:,2:]>THRESHOLD
    first_samples={name:(int(indices[0]) if len(indices:=np.flatnonzero(forces[:,index]>THRESHOLD)) else None) for index,name in enumerate(FORCES)}
    timeline=dict(position_provenance='Native ball center at contact sample; contact-manifold point is not recorded',first_board_contact_sample=first_samples['board'],first_bucket_contact_sample=first_samples['bucket'],
                  first_robot_contact_sample=first_samples['robot'],
                  first_robot_relative_to_board=relative_timing(first_samples['robot'],first_samples['board']))
    indices=np.flatnonzero(hit.any(axis=1))
    peaks=forces[:,2:].max(axis=0) if len(forces) else np.zeros(4)
    if not len(indices):
        return dict(**timeline,first_external_relative_to_board='not_observed',first_surfaces=[],first_sample=None,first_local_position_m=None,
                    peak_surfaces=[],peak_external_force_N=float(peaks.max()),
                    peak_by_surface_N=dict(zip(EXTERNAL,map(float,peaks))),
                    bucket_contact_before_first_external=False,
                    goal_axis_offset_mm=None,goal_lateral_offset_mm=None)
    first=int(indices[0]);maximum=float(peaks.max())
    first_names=[EXTERNAL[j] for j in np.flatnonzero(hit[first])]
    peak_names=[EXTERNAL[j] for j in range(4) if peaks[j]>THRESHOLD and abs(float(peaks[j])-maximum)<=1e-9]
    vector=np.asarray(bucket_xy,dtype=float)-board_xy
    length=float(np.linalg.norm(vector));unit=vector/length if length>1e-12 else None
    delta=positions[first,:2]-bucket_xy
    along=float(delta@unit)*1000 if unit is not None else None
    lateral=float(abs(delta[0]*unit[1]-delta[1]*unit[0]))*1000 if unit is not None else None
    return dict(**timeline,first_external_relative_to_board=relative_timing(first,first_samples['board']),first_surfaces=first_names,first_sample=first,first_local_position_m=positions[first].tolist(),
                peak_surfaces=peak_names,peak_external_force_N=maximum,
                peak_by_surface_N=dict(zip(EXTERNAL,map(float,peaks))),
                bucket_contact_before_first_external=bool((forces[:first,1]>THRESHOLD).any()),
                goal_axis_offset_mm=along,goal_lateral_offset_mm=lateral)


def observed_pose(scene,cell,row,positions,boards,quats,origin):
    target=np.array([row['board_x_m'],row['board_y_m'],row['height_m']])
    board=boards[0]-origin
    direction=np.asarray(cell['bucket_xy_m'])-target[:2]
    heading=math.atan2(direction[1],direction[0]);tilt=math.radians(row['tilt_deg'])
    target_normal=np.array([math.cos(heading)*math.sin(tilt),math.sin(heading)*math.sin(tilt),math.cos(tilt)])
    normal=Rotation.from_quat(quats[0][[1,2,3,0]]).as_matrix()[:,2]
    requested_release=target+[0,0,scene['ball']['radius_m']+row['gap_m']]
    return dict(board_center_error_mm=float(np.linalg.norm(board-target)*1000),
                board_normal_error_deg=math.degrees(math.acos(np.clip(normal@target_normal,-1,1))),
                ball_first_sample_error_from_initialized_release_mm=float(np.linalg.norm(positions[0]-origin-requested_release)*1000),
                measured_board_center_m=board.tolist(),measured_board_normal=normal.tolist(),
                requested_ball_release_center_m=requested_release.tolist(),
                first_observed_ball_center_m=(positions[0]-origin).tolist(),
                scope='Board pose is native body readback. Ball is initialized at release, then first observed one physics step later; no grasp/release calibration claim.')


def summarize_contacts(rows):
    valid=[r for r in rows if not r['invalid']]
    misses=[r for r in valid if not r['route_success']]
    first=collections.Counter();peak=collections.Counter()
    for row in misses:
        event=row['external_contact']
        first['+'.join(event['first_surfaces']) if event['first_surfaces'] else 'none_observed_during_score']+=1
        peak['+'.join(event['peak_surfaces']) if event['peak_surfaces'] else 'none_observed_during_score']+=1
    retained_failures=[r for r in valid if r['success'] and not r['route_success']]
    retained_first=collections.Counter('+'.join(r['external_contact']['first_surfaces']) if r['external_contact']['first_surfaces'] else 'none_observed_during_score' for r in retained_failures)
    robot_timing=collections.Counter(r['external_contact']['first_robot_relative_to_board'] for r in retained_failures)
    poses=[r['observed_pose'] for r in valid]
    return dict(valid_attempts=len(valid),valid_misses=len(misses),first_external_surface=dict(first),
                raw_retained_route_failures=len(retained_failures),
                raw_retained_route_failure_first_external_surface=dict(retained_first),
                raw_retained_route_failure_robot_timing=dict(robot_timing),
                valid_miss_first_external_relative_to_board=dict(collections.Counter(r['external_contact']['first_external_relative_to_board'] for r in misses)),
                valid_miss_robot_timing=dict(collections.Counter(r['external_contact']['first_robot_relative_to_board'] for r in misses)),

                dominant_peak_external_surface=dict(peak),
                no_qualified_board_rebound=sum(not r['board_bounce'] for r in misses),
                qualified_board_rebound_but_missed=sum(bool(r['board_bounce']) for r in misses),
                bucket_contact_then_external=sum(r['external_contact']['bucket_contact_before_first_external'] for r in misses),
                valid_release_pose=dict(max_board_center_error_mm=max((p['board_center_error_mm'] for p in poses),default=None),
                    max_board_normal_error_deg=max((p['board_normal_error_deg'] for p in poses),default=None),
                    max_ball_first_sample_error_mm=max((p['ball_first_sample_error_from_initialized_release_mm'] for p in poses),default=None)))


def analyze(run,native_audit_path):
    run=Path(run);audit=json.loads(Path(native_audit_path).read_text())
    result=json.loads((run/'result.json').read_text());spec=json.loads((run/'contract.json').read_text())
    require(result.get('status')=='completed' and audit.get('status')=='passed' and audit.get('schema')=='sircus-continuous-native-outcome-audit-v1','Completed independently audited native run required')
    require(audit['source_result_sha256']==digest(run/'result.json') and audit['source_contract_sha256']==digest(run/'contract.json'),'Native audit belongs to another or changed run')
    origins=finite(result['origins_m'],(35,3),'origins');dt=spec['scene']['simulation']['physics_dt_s']
    records=[];trial_hashes={};packet_hashes={}
    summaries=result.get('coverage',[])+result['batches']+result['assessments']
    for summary in summaries:
        path=safe_path(run,summary['file'],'.json',subdir=True)
        sha=digest(path);require(sha==summary['sha256']==audit['source_trial_sha256'].get(summary['file']),'Native trial changed after its audit')
        trial_hashes[summary['file']]=sha;trial=json.loads(path.read_text())
        mode=trial['mode'] if trial['mode']!='eval' else 'initial_eval' if trial['checkpoint']['file']==result['initial']['file'] else 'final_eval'
        if trial['mode']=='eval':require(trial['checkpoint'] in (result['initial'],result['final']),'Unknown frozen checkpoint identity')
        data=None
        if trial.get('physics'):
            packet=trial['physics'];packet_path=safe_path(run,packet['file'],'.npz')
            require(digest(packet_path)==packet['sha256'],'Native packet changed after its audit')
            with np.load(packet_path,allow_pickle=False) as f:data={k:f[k] for k in ('position_m','velocity_m_s','forces_N','board_position_m','board_quaternion_wxyz')}
            packet_hashes[packet['file']]=packet['sha256']
        for i,(cell,row,enabled) in enumerate(zip(spec['cells'],trial['rows'],trial['enabled'])):
            if not enabled:continue
            item=dict(trial_file=summary['file'],batch=trial['batch'],mode=mode,split=row['split'],cell_id=cell['id'],
                      bucket_xy_m=cell['bucket_xy_m'],radius_m=row['radius_m'],
                      bucket_reference_distance_m=float(np.linalg.norm(np.asarray(cell['bucket_xy_m'])-[.67,-.10])),
                      **{k:row[k] for k in ('invalid','route_success','route_quality','success','board_bounce','whole_ball_entry','max_retention_s','closest_opening_distance_m','external_contact_steps')})
            if not row['invalid']:
                require(data is not None,'Missing valid native trace')
                bucket=dict(spec['scene']['bucket'],center_xy_m=cell['bucket_xy_m'])
                score=ShotScore(bucket,spec['scene']['ball']['radius_m'])
                normal=np.asarray(row.get('scoring_board_normal',row['settled_board_normal']))
                positions=data['position_m'][:,i]-origins[i]
                for j,position in enumerate(positions):
                    score.step(position,float(data['forces_N'][j,i,0]),dt,float(data['velocity_m_s'][j,i]@normal))
                    if score.success or position[2]<-.2:break
                end=j+1
                require(score.success==row['success'] and score.bounced==row['board_bounce'],'Native score changed during failure analysis')
                event=surface_events(data['forces_N'][:,i],positions,np.asarray(row['settled_board_center_m'])[:2],np.asarray(cell['bucket_xy_m']),end)
                event['first_board_physics_step']=trial['release_step']+1+event['first_board_contact_sample'] if event['first_board_contact_sample'] is not None else None
                event['first_robot_physics_step']=trial['release_step']+1+event['first_robot_contact_sample'] if event['first_robot_contact_sample'] is not None else None
                event['first_external_physics_step']=trial['release_step']+1+event['first_sample'] if event['first_sample'] is not None else None
                item.update(score_samples=end,external_contact=event,
                            observed_pose=observed_pose(spec['scene'],cell,row,data['position_m'][:,i],data['board_position_m'][:,i],data['board_quaternion_wxyz'][:,i],origins[i]))
            records.append(item)
    groups={}
    for mode in ('coverage','train','initial_eval','final_eval'):
        rows=[r for r in records if r['mode']==mode]
        groups[mode]=dict(**statistics(rows),**summarize_contacts(rows),
                         by_split={split:statistics([r for r in rows if r['split']==split]) for split in ('train','validation','test')},
                         radius_bins=binned(rows,'radius_m',RADIUS_EDGES),
                         bucket_reference_distance_bins=binned(rows,'bucket_reference_distance_m',BUCKET_EDGES))
    return dict(schema=SCHEMA,status='completed',source_result_sha256=digest(run/'result.json'),
                source_native_audit_sha256=digest(native_audit_path),source_code_sha256=digest(__file__),
                source_trial_sha256=trial_hashes,source_packet_sha256=packet_hashes,
                groups=groups,records=records,
                spatial_cells=[dict(cell_id=c['id'],bucket_xy_m=c['bucket_xy_m'],groups={m:statistics([r for r in records if r['mode']==m and r['cell_id']==c['id']]) for m in groups}) for c in spec['cells']],
                scope='All-attempt denominators retained. Measured external contacts are restricted to each cell scoring window; later contacts do not become failure causes. Signed contact offsets are descriptive positions, not short/long diagnoses. Robot forces group both arms/tools; link identity and causal gripper attribution are not recorded. No new experiments, inference search or physical validation.')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True);parser.add_argument('--native-audit',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():parser.error('Choose a new output; prior evidence is preserved')
    value=analyze(args.run,args.native_audit)
    with args.output.open('x') as f:json.dump(value,f,allow_nan=False)
    summary={mode:{key:group[key] for key in ('attempts','released','invalid','captures','capture_rate','board_bounces','first_external_surface','raw_retained_route_failures','raw_retained_route_failure_first_external_surface','raw_retained_route_failure_robot_timing','valid_release_pose')}
             for mode,group in value['groups'].items()}
    print(json.dumps(dict(schema=SCHEMA,status=value['status'],source_result_sha256=value['source_result_sha256'],groups=summary)),flush=True)


if __name__=='__main__':main()
