"""Scoped fixed-yaw two-parameter IK resolver; original yaw behavior is unchanged."""
from __future__ import annotations
import copy
import math
import time
import numpy as np
from scipy.spatial.transform import Rotation
from .benchmark_verify import digest
from .calibrated_admission import (VERSION as BASE_VERSION, CONNECTION_POLICY_VERSION, verify_resolution,
    verify_connection, compose_anchor_prefix, anchor_prefix_identity)
from .admission import native_pose_error, body_transform, base14
from .gripper_yaw_geometry import yaw_target,DOWNWARD

VERSION='sircus-fixed-yaw-tilt-gap-geometry-v1'


def intervention(scene,source,yaw_source,values):
    yaw,targets=yaw_target(scene,source['bucket_xy_m'],values)
    return dict(version=VERSION,source_record_id=source['id'],source_record_sha256=digest(source),
        frozen_yaw_record_id=yaw_source['id'],frozen_yaw_record_sha256=digest(yaw_source),
        source_q10=copy.deepcopy(yaw_source['q10']),source_q14=copy.deepcopy(yaw_source['q14']),
        requested_yaw_rad=yaw,requested_rotation_matrix=targets['dropper_flange_world'][:3,:3].tolist(),
        requested_dropper_flange_world=targets['dropper_flange_world'].tolist(),
        requested_board_flange_local=targets['board_flange_local'].tolist(),
        release_center_m=targets['release_center_m'].tolist(),
        board_branch_fixed=False,fixed_board_center_m=copy.deepcopy(source['values'][:3]),
        source_values=copy.deepcopy(source['values']),requested_values=list(values),
        changed_parameter_indices=[3,4],branch_selection_uses_flight_outcomes=False)


class TiltGapResolver:
    def __init__(self,base,sources,yaw_rows,plan,exact):
        self.base=base;self._yaw_connected={};self.exact=exact
        byid={s['id']:s for s in sources};self.yaw_by_source={r['source_record_id']:r for r in yaw_rows}
        self.sources_by_query={digest(dict(bucket_xy_m=r['bucket_xy_m'],values=r['values'])):byid[r['source_record_id']]
            for r in plan if r['adapter_directory']=='native-yaw'}
    def __getattr__(self,name):return getattr(self.base,name)

    def connect(self,anchor,requested_values):
        values=list(map(float,requested_values));bucket=anchor['bucket_xy_m']
        verify_resolution(anchor,self.sources,self.home);self.base._check_resolution_bytes(anchor)
        query=digest(dict(bucket_xy_m=bucket,values=values))
        if query in self.exact:
            return dict(copy.deepcopy(self.exact[query]),cache_hit=True)
        source=self.sources_by_query.get(query)
        if source is None:raise ValueError('Only the frozen far grid and exact yaw references may rotate')
        ext=intervention(self.scene,source,self.yaw_by_source[source['id']],values)
        key=digest(dict(version=VERSION,connection_policy=CONNECTION_POLICY_VERSION,
            resolution_sha256=anchor['sha256'],values=values,bucket_xy_m=bucket,yaw_intervention=ext))
        if key in self._yaw_connected:return dict(copy.deepcopy(self._yaw_connected[key]),cache_hit=True)
        begin=time.monotonic();timing=dict(home_s=0.,endpoint_s=0.,path_s=0.,wall_s=0.)
        record=dict(version=BASE_VERSION,bucket_xy_m=bucket,values=values,resolution_file=anchor['file'],
            resolution_sha256=anchor['sha256'],home_sha256=self.home['sha256'],passed=False,geometry_eligible=False,
            endpoint_eligible=False,reason=None,q10=[0.]*6,q14=[0.]*6,proof=None,
            endpoint_attempts=None,path_attempts=[],timing=timing,action_substituted=False,
            connection_policy=CONNECTION_POLICY_VERSION,anchor_values=anchor['anchor_values'],
            anchor_derivation_sha256=digest(anchor['anchor_derivation']),yaw_intervention=ext,yaw_key=key)
        if not anchor['home_passed']:
            record['reason']='canonical_home_unresolved_in_layout'
        else:
            started=time.monotonic();endpoint=self.solve(source,values);timing['endpoint_s']=time.monotonic()-started
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
        record.update(self.base._save('yaw-tilt-gap-connections',key,record))
        verify_connection(record,anchor,self.home,values)
        self._yaw_connected[key]=copy.deepcopy(record)
        return dict(record,cache_hit=False)

    def solve(self,source,requested_values):
        bucket=np.asarray(source['bucket_xy_m']);values=np.asarray(requested_values);hint=self.yaw_by_source[source['id']]
        _,targets=yaw_target(self.scene,bucket,values);solver=self.base.solver
        info=dict(passed=False,reason=None,candidates=[],pair_attempts=[],starts=16,iterations=180,normal_only=False,
            targets={k:v.tolist() for k,v in targets.items()},flight_labels_used=False)
        if (values<solver.low-1e-9).any() or (values>solver.high+1e-9).any():
            info['reason']='action_bounds_rejected';return info
        pools=[]
        for arm,target,initial,known in ((solver.ik.dropper,targets['dropper_flange_world'],solver.initial['dropper_joint_rad'],hint['q10']),
                                       (solver.ik.board,targets['board_flange_local'],solver.initial['board_joint_rad'],hint['q14'])):
            p=arm.solve_candidates(target[None],initial,known=np.asarray(known)[None],starts=16,iterations=180,
                normal_only=False,preserve_known=np.asarray([False]))
            pools.append({k:p[k][0].cpu().numpy() for k in ('q','valid','residual')})
        first,second=pools
        ids10=list(np.flatnonzero(first['valid']));ids14=list(np.flatnonzero(second['valid']))
        pairs=sorted(((i,j) for i in ids10 for j in ids14),key=lambda ij:(
            float(np.linalg.norm(first['q'][ij[0]]-hint['q10'])+np.linalg.norm(second['q'][ij[1]]-hint['q14'])),
            float(first['residual'][ij[0]]+second['residual'][ij[1]]),int(ij[0]),int(ij[1])))
        info['valid_ik_branches']=[len(ids10),len(ids14)];info['branch_order']=[[int(i),int(j)] for i,j in pairs]
        seen=set()
        for i,j in pairs:
            q10,q14=first['q'][i],second['q'][j];identity=tuple(np.round(np.r_[q10,q14],5))
            if identity in seen:continue
            seen.add(identity);checked=solver.endpoint.check(q10,q14,bucket,values,targets)
            attempt=dict(branch10=int(i),branch14=int(j),q10=q10.tolist(),q14=q14.tolist(),checked=checked,
                source_joint_distance_rad=float(np.linalg.norm(q10-hint['q10'])+np.linalg.norm(q14-hint['q14'])),
                residual=float(first['residual'][i]+second['residual'][j]))
            info['pair_attempts'].append(attempt)
            if checked['reason']:continue
            errors=[native_pose_error(solver.gate,role,np.r_[q10,q14],target) for role,target in (
                (10,targets['dropper_flange_world']),(14,base14(self.scene)@targets['board_flange_local']))]
            attempt['native_pose_errors']=errors
            if any(e['position_m']>=1e-4 or .25*e['rotation_rad']>=1e-4 for e in errors):
                attempt['pose_rejected']=True;continue
            info['candidates'].append(dict(bucket_xy_m=bucket.tolist(),values=values.tolist(),q10=q10.tolist(),q14=q14.tolist(),
                geometry_eligible=True,branch10=int(i),branch14=int(j),endpoint=checked))
            if len(info['candidates'])>=solver.max_pairs:break
        info['passed']=bool(info['candidates']);info['reason']=None if info['passed'] else (
            'finite_ik_unresolved' if not ids10 or not ids14 else 'strict_endpoint_clearance_or_pose_rejected')
        return info


def rotation_evidence(scene,gate,row):
    targeted=row['adapter_directory']=='native-yaw';angle,targets=yaw_target(scene,row['bucket_xy_m'],row['values'])
    if not targeted:targets['dropper_flange_world'][:3,:3]=DOWNWARD
    result=dict(requested_yaw_rad=angle if targeted else 0.,
        requested_rotation_matrix=targets['dropper_flange_world'][:3,:3].tolist(),
        basis='model FK of observed native initial float32 joints; held endpoint')
    if row.get('native_release'):
        q=np.r_[row['native_initial_q10'],row['native_initial_q14']]
        actual=body_transform(gate,10,q);board=body_transform(gate,14,q)
        expected_board=base14(scene)@targets['board_flange_local']
        error=float(Rotation.from_matrix(targets['dropper_flange_world'][:3,:3].T@actual[:3,:3]).magnitude())
        board_error=float(Rotation.from_matrix(expected_board[:3,:3].T@board[:3,:3]).magnitude())
        poserr=float(np.linalg.norm(actual[:3,3]-targets['dropper_flange_world'][:3,3]));boardpos=float(np.linalg.norm(board[:3,3]-expected_board[:3,3]))
        result.update(measured_initial_rotation_matrix=actual[:3,:3].tolist(),measured_initial_yaw_rad=float(math.atan2(actual[1,0],actual[0,0])),
            rotation_error_rad=error,position_error_m=poserr,board_rotation_error_rad=board_error,board_position_error_m=boardpos,
            measured_initial_flange_position_m=actual[:3,3].tolist(),requested_release_center_m=targets['release_center_m'].tolist())
        if error>=4e-4 or board_error>=4e-4 or poserr>=1e-4 or boardpos>=1e-4:
            raise RuntimeError('Observed native reset FK differs from exact yaw/tilt/gap targets')
    return result
