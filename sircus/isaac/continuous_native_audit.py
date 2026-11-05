"""Reconstruct continuous-pilot outcomes from every full-rate native observation.

CPU-only: no simulator, policy optimizer, hardware or service is launched.
Measured native scalar outcomes, release guards and recorded camera readbacks
are checked separately from policy likelihood/update verification.
"""

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from board_route import BoardRoute
from entry_quality import EntryQuality
from farm_contract import reward as policy_reward
from rl_contract import ShotScore
from settling_contract import accepted

SCHEMA = "sircus-continuous-native-outcome-audit-v1"
FORCES = ("board", "bucket", "table", "floor", "robot", "unattributed")
KEYS = {"time_step", "position_m", "velocity_m_s", "forces_N",
        "joint_position_rad", "joint_velocity_rad_s", "board_position_m", "board_quaternion_wxyz"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def signed(value, label):
    unsigned=dict(value);fingerprint=unsigned.pop("fingerprint",None)
    actual=hashlib.sha256(json.dumps(unsigned,sort_keys=True,allow_nan=False).encode()).hexdigest()
    require(actual==fingerprint,label+" fingerprint changed")
    return fingerprint


def same(actual, expected, tolerance=1e-7):
    if isinstance(expected,dict):
        return isinstance(actual,dict) and set(actual)==set(expected) and all(same(actual[k],v,tolerance) for k,v in expected.items())
    if isinstance(expected,list):
        return isinstance(actual,list) and len(actual)==len(expected) and all(same(a,b,tolerance) for a,b in zip(actual,expected))
    if isinstance(expected,(int,float)) and not isinstance(expected,bool):
        return isinstance(actual,(int,float)) and not isinstance(actual,bool) and math.isfinite(actual) and abs(actual-expected)<=tolerance
    return actual==expected


def fields_equal(row, expected):
    for key,value in expected.items():
        require(key in row and same(row[key],value),"Native rescored outcome differs: "+key)


def finite(value, shape, label):
    value=np.asarray(value)
    require(value.shape==shape and np.issubdtype(value.dtype,np.number) and np.isfinite(value).all(),"Invalid native packet "+label)
    return value


def safe_path(run,name,suffix,subdir=False):
    require(isinstance(name,str),"Unsafe evidence filename")
    path=Path(name)
    require(not path.is_absolute() and '..' not in path.parts and name.endswith(suffix),"Unsafe evidence filename")
    require((len(path.parts)==2 and path.parts[0]=='trials') if subdir else len(path.parts)==1,"Unsafe evidence filename")
    result=Path(run)/path
    require(result.is_file() and not result.is_symlink(),"Missing or linked evidence file")
    return result


def expected_empty(scene,cell,dt):
    bucket=dict(scene['bucket'],center_xy_m=cell['bucket_xy_m'])
    score=ShotScore(bucket,scene['ball']['radius_m'])
    quality=EntryQuality(bucket,scene['ball']['radius_m'])
    route=BoardRoute(dt)
    outcome=score.result();q=quality.result(outcome['success'])
    return dict(**outcome,**q,**route.result(outcome['success'],q['quality_score']))


def verify_trial(run,trial,scene,cells,origins,clip=None):
    """All actual valid releases use the original scorer and per-cell freeze rule."""
    rows=trial['rows'];count=len(cells)
    origins=finite(origins,(count,3),'origins')
    require(len(rows)==len(trial['actions'])==len(trial['enabled'])==count,"Native trial cell count changed")
    dt=float(scene['simulation']['physics_dt_s'])
    require(abs(dt-1/240)<1e-15,"Native audit requires the original 240Hz cadence")
    valid=[]
    for i,(row,cell,action) in enumerate(zip(rows,cells,trial['actions'])):
        require(row['cell_id']==cell['id'] and all(k in row and same(row[k],v) for k,v in action.items()),"Native action/workcell association changed")
        if not row['invalid']:
            require(trial['enabled'][i] and accepted(row['settled']) and row['settled']['dropper_position_error_mm']<=2,"Native measured release admission failed")
            require(row.get('qualification',{}).get('accepted',False) and not row.get('path_rejection'),"Native geometry/path admission failed")
            normal=finite(row.get('scoring_board_normal',row['settled_board_normal']),(3,),'release normal')
            require(abs(float(np.linalg.norm(normal))-1)<=1e-4,"Native release normal is not normalized")
            valid.append(i)
        else:
            require(not row['route_success'] and not row['success'],"Rejected release cannot capture")
            fields_equal(row,expected_empty(scene,cell,dt))
        require(np.isclose(float(trial['rewards'][i]),policy_reward(row),atol=1e-7,rtol=0),"Native PPO reward differs from original reward contract")
    packet=trial.get('physics')
    if packet is None:
        require(not valid and trial['score_end_step']==trial['release_step'],"Missing physics for an admitted native release")
        require(clip is None or not any(f['phase']=='release' for f in clip['frames']),"All-rejected trial cannot have release frames without observations")
        return dict(native_steps=0,rescored_releases=0,invalid_cells=count,captures=0,clean_captures=0,frame_readbacks=0)
    path=safe_path(run,packet['file'],'.npz')
    require(digest(path)==packet['sha256'],"Native physics checksum changed")
    with np.load(path,allow_pickle=False) as archive:
        require(set(archive.files)==KEYS,"Native packet columns changed")
        data={k:archive[k] for k in KEYS}
    steps=data['time_step']
    require(steps.ndim==1 and np.issubdtype(steps.dtype,np.integer),"Invalid native physics clock")
    require(np.array_equal(steps,np.arange(trial['release_step']+1,trial['score_end_step']+1)),"Missing full-rate native clock coverage")
    length=len(steps)
    require(length==packet['samples'] and 1<=length<=960,"Native scoring horizon/sample count changed")
    for key,tail in (('position_m',(3,)),('velocity_m_s',(3,)),('forces_N',(6,)),
                     ('joint_position_rad',(2,6)),('joint_velocity_rad_s',(2,6)),
                     ('board_position_m',(3,)),('board_quaternion_wxyz',(4,))):
        finite(data[key],(length,count,*tail),key)
    require((data['forces_N']>=0).all(),"Negative native contact force magnitude")
    require(np.allclose(np.linalg.norm(data['board_quaternion_wxyz'],axis=-1),1,atol=1e-3,rtol=0),"Native board quaternion is not normalized")
    board_drift=[]
    for i in valid:
        row=rows[i];bucket=dict(scene['bucket'],center_xy_m=cells[i]['bucket_xy_m'])
        score=ShotScore(bucket,scene['ball']['radius_m']);quality=EntryQuality(bucket,scene['ball']['radius_m']);route=BoardRoute(dt)
        normal=np.asarray(row.get('scoring_board_normal',row['settled_board_normal']))
        board_first=data['board_position_m'][0,i]-origins[i]
        require(np.linalg.norm(board_first-np.asarray(row['settled_board_center_m']))<=.001,"Native board release center differs from measured setup")
        first_quat=data['board_quaternion_wxyz'][0,i]
        first_normal=Rotation.from_quat(first_quat[[1,2,3,0]]).as_matrix()[:,2]
        require(np.rad2deg(np.arccos(np.clip(first_normal@normal,-1,1)))<=.5,"Native board release normal differs from measured setup")
        require(np.linalg.norm(data['position_m'][0,i]-origins[i]-row['release_ball_center_m'])<=.002,"Native ball release differs from initialization")
        consumed=0
        for j in range(length):
            position=data['position_m'][j,i]-origins[i]
            velocity=data['velocity_m_s'][j,i]
            forces=data['forces_N'][j,i]
            score.step(position,float(forces[0]),dt,float(velocity@normal))
            quality.step(position,float(forces[1]),score.entered)
            route.step(dict(zip(FORCES,map(float,forces))))
            consumed+=1
            if score.success or position[2]<-.2:
                break
        outcome=score.result();q=quality.result(outcome['success'])
        combined=dict(**outcome,**q,**route.result(outcome['success'],q['quality_score']))
        fields_equal(row,combined)
        require(consumed==960 or consumed<=length and (score.success or data['position_m'][consumed-1,i,2]-origins[i,2]<-.2),"Native per-cell score terminated before its freeze rule")
        board_drift.append(float(1000*np.max(np.linalg.norm(data['board_position_m'][:consumed,i]-data['board_position_m'][0,i],axis=1))))
    frames_checked=0
    if clip is not None:
        require(clip['rows']==rows and clip['actions']==trial['actions'] and clip['release_step']==trial['release_step'],"Recorded clip trial association changed")
        for frame in clip['frames']:
            if frame['phase']!='release':
                continue
            j=frame['physics_step']-trial['release_step']-1
            require(0<=j<length and np.array_equal(frame['all_ball_world_m'],data['position_m'][j]),"Camera ball readback differs from native packet")
            for arm,name in enumerate(('Arm10','Arm14')):
                require(np.array_equal(frame['all_joints_rad'][name],data['joint_position_rad'][j,:,arm]),"Camera joint readback differs from native packet")
            frames_checked+=1
    return dict(file=packet['file'],sha256=packet['sha256'],native_steps=length,
                rescored_releases=len(valid),invalid_cells=count-len(valid),
                captures=sum(bool(rows[i]['route_success']) for i in valid),
                clean_captures=sum(rows[i]['route_quality']==2 for i in valid),
                frame_readbacks=frames_checked,board_max_translation_through_score_mm=max(board_drift,default=0.0))



def verify_camera_frame(frame,camera):
    # RTX pose readback uses float32. Preserve the exact selected-camera file
    # hash while applying the producer's existing 0.2mm numerical pose gate.
    eye=finite(frame['eye_m'],(3,),'camera eye')
    target=finite(frame['target_m'],(3,),'camera target')
    require(np.linalg.norm(eye-np.asarray(camera['eye_m']))<=2e-4
            and np.allclose(target,camera['target_m'],atol=1e-9,rtol=0),
            'Recorded camera moved')

def audit(run):
    run=Path(run)
    result=json.loads((run/'result.json').read_text());study=json.loads((run/'study.json').read_text());spec=json.loads((run/'contract.json').read_text())
    schema=study.get('schema')
    require(schema in ('sircus-native-continuous-pilot-v1','sircus-native-feasibility-guided-pilot-v1')
            and result.get('schema')==schema and result.get('status')=='completed',"Completed continuous native pilot required")
    if schema=='sircus-native-feasibility-guided-pilot-v1':
        # Reuse the same physical scorer only after checking the separate,
        # geometry-conditioned policy/support protocol. Never relabel old runs.
        from feasibility_guided_audit import backend_for_run
        backend_for_run(run).validate_study(study,spec)
    study_hash=signed(study,'Study');geometry_hash=signed(spec,'Geometry')
    require(result['study_fingerprint']==study_hash and result['geometry_fingerprint']==study['geometry_fingerprint']==geometry_hash,"Native study/geometry binding changed")
    cells=spec['cells'];require(len(cells)==35 and result['workcells']==35 and result['articulations']==70,"Expected unchanged 35 workcells/70arms")
    origins=finite(result['origins_m'],(35,3),'origins')
    scoring_files=('rl_contract.py','board_route.py','entry_quality.py','farm_contract.py','settling_contract.py')
    scoring_source_sha256={name:digest(Path(__file__).with_name(name)) for name in scoring_files}
    require(all(result['source_sha256'].get(name)==sha for name,sha in scoring_source_sha256.items()),
            'Native scoring definitions differ from the recorded producer')
    clips_payload=json.loads((run/'clips.json').read_text())
    clips={clip['trial_file']:clip for clip in clips_payload['clips']}
    require(len(clips)==len(clips_payload['clips']),"Duplicate recorded trial clips")
    summaries=result.get('coverage',[])+result['batches']+result['assessments']
    require(len({s['file'] for s in summaries})==len(summaries),"Duplicate native trials")
    audits=[];trial_hashes={}
    for summary in summaries:
        path=safe_path(run,summary['file'],'.json',subdir=True)
        require(digest(path)==summary['sha256'],"Native trial checksum changed")
        trial=json.loads(path.read_text())
        require(trial['study_fingerprint']==study_hash and trial['mode']==summary['mode'],"Native trial identity changed")
        if trial['mode']=='coverage':require(trial['optimizer_samples']==0 and not trial['policy_action_used'],"Geometry coverage entered the policy optimizer")
        clip=clips.get(summary['file'])
        if clip is not None:
            require(clip['camera_sha256']==study['source']['camera_sha256']==digest(run/'selected-camera.json'),"Recorded camera source changed")
            for frame in clip['frames']:
                verify_camera_frame(frame,result['camera'])
        value=verify_trial(run,trial,spec['scene'],cells,origins,clip)
        value.update(trial_file=summary['file'],mode=trial['mode'])
        audits.append(value);trial_hashes[summary['file']]=summary['sha256']
    require(set(clips).issubset(trial_hashes),"Unassociated recorded clip")
    return dict(schema=SCHEMA,status='passed',audit_source_sha256=digest(__file__),
                native_study_schema=schema,
                scoring_source_sha256=scoring_source_sha256,source_result_sha256=digest(run/'result.json'),
                source_study_sha256=digest(run/'study.json'),source_contract_sha256=digest(run/'contract.json'),
                clips_sha256=digest(run/'clips.json'),study_fingerprint=study_hash,source_trial_sha256=trial_hashes,
                trials=audits,native_time_steps=sum(a['native_steps'] for a in audits),
                all_cell_observations=sum(a['native_steps']*35 for a in audits),
                rescored_releases=sum(a['rescored_releases'] for a in audits),
                frame_readbacks=sum(a['frame_readbacks'] for a in audits),
                scope='Every admitted release independently rescored from all240Hz native ball/force observations using original retention/route/entry definitions; camera joints/balls bound. Policy likelihoods and mesh/path safety remain separate audits.')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():parser.error('Choose a new audit output; previous evidence is preserved')
    value=audit(args.run)
    with args.output.open('x') as f:json.dump(value,f,allow_nan=False)
    print(json.dumps(value),flush=True)


if __name__=='__main__':main()
