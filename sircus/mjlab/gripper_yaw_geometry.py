"""Opt-in bucket-facing release yaw, fixed board branch and strict geometry.

No global IK target or collision rule is modified. Yaw has an explicit immutable
connection-key/target binding even though the five shot coordinates stay fixed.
"""
from __future__ import annotations
import copy
import math
import time
import numpy as np
from scipy.spatial.transform import Rotation
from .benchmark_verify import digest
from .calibrated_admission import (VERSION as BASE_VERSION, CONNECTION_POLICY_VERSION, verify_resolution,
    verify_connection, source_geometry, compose_anchor_prefix, anchor_prefix_identity)
from .admission import native_pose_error, body_transform, base14
from .learning_ik import shot_targets

VERSION='sircus-bucket-facing-fixed-board-yaw-v1'
DOWNWARD=np.diag([1.,-1.,-1.])


def yaw_target(scene,bucket,values):
    bucket=np.asarray(bucket,float);values=np.asarray(values,float)
    if bucket.shape!=(2,) or values.shape!=(5,) or not np.isfinite(np.r_[bucket,values]).all():
        raise ValueError('Finite bucketXY and five shot coordinates required')
    delta=bucket-values[:2]
    if np.linalg.norm(delta)<=.15:raise ValueError('Original board-bucket distance constraint required')
    yaw=float(math.atan2(delta[1],delta[0]))
    rotation=Rotation.from_euler('z',yaw).as_matrix()@DOWNWARD
    targets={k:v[0] for k,v in shot_targets(scene,bucket[None],values[None,:3],[values[3]],[values[4]]).items()}
    targets['dropper_flange_world'][:3,:3]=rotation
    return yaw,targets


def intervention(scene,source):
    yaw,targets=yaw_target(scene,source['bucket_xy_m'],source['values'])
    return dict(version=VERSION,source_record_id=source['id'],source_record_sha256=digest(source),
        fixed_q14=copy.deepcopy(source['q14']),source_q10=copy.deepcopy(source['q10']),
        requested_yaw_rad=yaw,requested_rotation_matrix=targets['dropper_flange_world'][:3,:3].tolist(),
        requested_dropper_flange_world=targets['dropper_flange_world'].tolist(),
        release_center_m=targets['release_center_m'].tolist(),
        direction='local +X passage points toward bucket XY; no modulo180 substitution',
        board_branch_fixed=True,source_values=copy.deepcopy(source['values']))


class YawResolver:
    """Wrap an authenticated resolver; only the changed orientation gets new proofs."""
    def __init__(self,base,sources):
        self.base=base;self.sources_by_query={digest(dict(bucket_xy_m=s['bucket_xy_m'],values=s['values'])):copy.deepcopy(s) for s in sources}
        if len(self.sources_by_query)!=len(sources):raise ValueError('Unique yaw source queries required')
        self._yaw_connected={}
    def __getattr__(self,name):return getattr(self.base,name)

    def connect(self,anchor,requested_values):
        values=list(map(float,requested_values));bucket=anchor['bucket_xy_m']
        verify_resolution(anchor,self.sources,self.home);self.base._check_resolution_bytes(anchor)
        source=self.sources_by_query.get(digest(dict(bucket_xy_m=bucket,values=values)))
        if source is None:raise ValueError('Yaw may only alter the two exact corrected recipes')
        ext=intervention(self.scene,source)
        key=digest(dict(version=VERSION,connection_policy=CONNECTION_POLICY_VERSION,
            resolution_sha256=anchor['sha256'],values=values,bucket_xy_m=bucket,yaw_intervention=ext))
        if key in self._yaw_connected:return dict(copy.deepcopy(self._yaw_connected[key]),cache_hit=True)
        begin=time.monotonic();timing=dict(home_s=0.,endpoint_s=0.,path_s=0.,wall_s=0.)
        record=dict(version=BASE_VERSION,bucket_xy_m=bucket,values=values,resolution_file=anchor['file'],
            resolution_sha256=anchor['sha256'],home_sha256=self.home['sha256'],passed=False,geometry_eligible=False,
            endpoint_eligible=False,reason=None,q10=[0.]*6,q14=copy.deepcopy(source['q14']),proof=None,
            endpoint_attempts=None,path_attempts=[],timing=timing,action_substituted=False,
            connection_policy=CONNECTION_POLICY_VERSION,anchor_values=anchor['anchor_values'],
            anchor_derivation_sha256=digest(anchor['anchor_derivation']),yaw_intervention=ext,yaw_key=key)
        if not anchor['home_passed']:
            record['reason']='canonical_home_unresolved_in_layout'
        else:
            started=time.monotonic();endpoint=self.solve(source);timing['endpoint_s']=time.monotonic()-started
            record.update(endpoint_attempts=endpoint,endpoint_eligible=endpoint['passed'],reason=endpoint['reason'])
            for row in endpoint['candidates']:
                target=np.r_[row['q10'],row['q14']];started=time.monotonic()
                if anchor['passed']:
                    aq=np.r_[anchor['resolved']['q10'],anchor['resolved']['q14']]
                    suffix=self.planner.plan(target,bucket,start=aq)
                    if suffix['passed']:proof=compose_anchor_prefix(anchor,self.home,target,suffix)
                    else:
                        rejected=dict(passed=False,reason='finite_anchor_suffix_unresolved',connection_route=dict(
                            kind='anchor_suffix_rejected',prefix=anchor_prefix_identity(anchor),target_q=target.tolist(),suffix_proof=suffix))
                        index=len(record['path_attempts']);record['path_attempts'].append(rejected)
                        proof=copy.deepcopy(self.planner.plan(target,bucket));proof['connection_route']=dict(
                            kind='full_home_after_anchor_suffix',prefix=anchor_prefix_identity(anchor),suffix_attempt_index=index)
                else:proof=self.planner.plan(target,bucket)
                timing['path_s']+=time.monotonic()-started;record['path_attempts'].append(proof)
                if proof['passed']:
                    record.update(passed=True,geometry_eligible=True,q10=row['q10'],q14=row['q14'],proof=proof,endpoint=row['endpoint'],reason=None);break
            if endpoint['passed'] and not record['passed']:record['reason']='finite_requested_setup_path_unresolved'
        timing['wall_s']=time.monotonic()-begin
        record.update(self.base._save('yaw-connections',key,record))
        verify_connection(record,anchor,self.home,values)
        self._yaw_connected[key]=copy.deepcopy(record)
        return dict(record,cache_hit=False)

    def solve(self,source):
        bucket=np.asarray(source['bucket_xy_m']);values=np.asarray(source['values']);q14=np.asarray(source['q14'])
        _,targets=yaw_target(self.scene,bucket,values);solver=self.base.solver
        low,high=solver.low,solver.high
        if (values<low-1e-9).any() or (values>high+1e-9).any():raise ValueError('Original bounds required')
        pool=solver.ik.dropper.solve_candidates(targets['dropper_flange_world'][None],
            solver.initial['dropper_joint_rad'],known=np.asarray(source['q10'])[None],starts=16,iterations=180,
            normal_only=False,preserve_known=np.asarray([False]))
        q=pool['q'][0].cpu().numpy();valid=pool['valid'][0].cpu().numpy();residual=pool['residual'][0].cpu().numpy()
        # Deterministic joint proximity to the preserved source, then residual/ID;
        # no rollout or reward is consulted for branch selection.
        ids=sorted(np.flatnonzero(valid),key=lambda i:(float(np.linalg.norm(q[i]-source['q10'])),float(residual[i]),int(i)))
        info=dict(passed=False,reason=None,candidates=[],pair_attempts=[],valid_ik_branches=[len(ids),1],
            dropper_query_sha256=pool['query_sha256'][0],branch_order=list(map(int,ids)),fixed_q14=q14.tolist(),
            targets={k:v.tolist() for k,v in targets.items()},starts=16,iterations=180,normal_only=False)
        seen=set()
        for i in ids:
            q10=q[i];identity=tuple(np.round(q10,5))
            if identity in seen:continue
            seen.add(identity);checked=solver.endpoint.check(q10,q14,bucket,values,targets)
            attempt=dict(branch10=int(i),branch14='preserved_corrected',q10=q10.tolist(),q14=q14.tolist(),residual=float(residual[i]),source_joint_distance_rad=float(np.linalg.norm(q10-source['q10'])),checked=checked);info['pair_attempts'].append(attempt)
            if checked['reason']:continue
            errors=[native_pose_error(solver.gate,role,np.r_[q10,q14],target) for role,target in (
                (10,targets['dropper_flange_world']),(14,base14(self.scene)@targets['board_flange_local']))]
            attempt['native_pose_errors']=errors
            if any(e['position_m']>=1e-4 or .25*e['rotation_rad']>=1e-4 for e in errors):
                attempt['pose_rejected']=True;continue
            info['candidates'].append(dict(bucket_xy_m=bucket.tolist(),values=values.tolist(),q10=q10.tolist(),q14=q14.tolist(),
                geometry_eligible=True,branch10=int(i),branch14='preserved_corrected',endpoint=checked))
            if len(info['candidates'])>=solver.max_pairs:break
        info['passed']=bool(info['candidates']);info['reason']=None if info['passed'] else ('finite_ik_unresolved' if not ids else 'strict_endpoint_clearance_or_pose_rejected')
        return info


def yaw_evidence(scene,gate,row,source):
    targeted=row['variant']=='bucket_facing';angle,targets=yaw_target(scene,row['bucket_xy_m'],row['values'])
    rotation=targets['dropper_flange_world'][:3,:3] if targeted else DOWNWARD
    result=dict(requested_yaw_rad=angle if targeted else 0.,requested_rotation_matrix=rotation.tolist(),
        nominal_source_q14=copy.deepcopy(source['q14']),basis='URDF/native geometry FK of observed native_initial_q10 (float32); held endpoint',
        board_q14_unchanged=bool(np.array_equal(row.get('q14'),source['q14'])),release_unchanged=None)
    if row.get('native_release'):
        observed=np.r_[row['native_initial_q10'],row['native_initial_q14']]
        pose=body_transform(gate,10,observed);actual=pose[:3,:3]
        error=float(Rotation.from_matrix(rotation.T@actual).magnitude())
        result.update(measured_initial_rotation_matrix=actual.tolist(),measured_initial_yaw_rad=float(math.atan2(actual[1,0],actual[0,0])),
            rotation_error_rad=error,measured_initial_flange_position_m=pose[:3,3].tolist(),
            release_unchanged=bool(np.allclose(row['release_ball_center_m'],source['release_ball_center_m'],atol=1e-12,rtol=0)),
            board_q14_unchanged=bool(np.array_equal(row['q14'],source['q14'])),
            board_native_initial_q14_unchanged=bool(np.array_equal(row['native_initial_q14'],np.asarray(source['q14'],np.float32).astype(float))))
        if error>=4e-4 or np.linalg.norm(pose[:3,3]-targets['dropper_flange_world'][:3,3])>=1e-4:
            raise RuntimeError('Observed native-reset dropper full pose differs from declared yaw')
        if not result['board_q14_unchanged'] or not result['board_native_initial_q14_unchanged'] or not result['release_unchanged']:
            raise RuntimeError('Yaw experiment altered board branch or release coordinates')
    return result
