"""Reach-filtered, bouncer-diverse preparation followed by exact native64 replay.

Preparation uses CPU full-pose IK and the unchanged endpoint/sample-path gates.
Only a separately frozen admitted connection can be released. The native engine,
model, v3 scorer, collision margins and original successful recipes are untouched.
"""
from __future__ import annotations
import copy
from collections import deque
from functools import lru_cache
import math
from pathlib import Path
import time

import numpy as np
from scipy.spatial.transform import Rotation
from .admission import native_pose_error,base14
from .benchmark_verify import contained,digest,read,sha
from .calibrated_admission import (VERSION,CONNECTION_POLICY_VERSION,verify_resolution,verify_connection,
    source_geometry,compose_anchor_prefix,anchor_prefix_identity,_vector)
from .fixed35_imitation_native import BEHAVIOR_VERSION,yaw_target
from .fixed35_recovery_native import RecoveryEngine
from .fixed35_bouncer_native import BouncerEngine,wrapped_difference
from .gripper_yaw_geometry import DOWNWARD
from .learning_ik import read_chain

BROAD_VERSION='sircus-fixed35-broad-bouncer-branches-v1'
BRANCH_RULE=dict(version=BROAD_VERSION,starts_per_arm=32,iterations=360,seed_base=20261112,
    seed_rule='(base+int(sha256(sorted JSON of bucket/values)[:8],16)) %2147483647',
    normal_only=False,preserve_known=False,maximum_unique_pairs=64,maximum_endpoint_candidates=16,
    maximum_path_candidates=16,maximum_path_attempts_per_candidate=2,maximum_saved_path_attempts=32,
    maximum_admitted_branches=2,distinct_bouncer_joint_rad=.05,
    branch_priority='Farthest proximal q14 from immutable anchor first, then max-min proximal/full q14 diversity',
    pair_priority='One q10 rank across every distinct q14 before the next q10 rank',
    selection_uses_flight_labels=False,position_tolerance_m=1e-4,rotation_tolerance_rad=4e-4)
BROAD_RULE=BRANCH_RULE
REACH_VERSION='sircus-urdf-conservative-full-pose-wrist-bound-v1'
BROAD_FIELDS=('broad_rule_version','broad_branch_slot','broad_key','broad_pool_file','broad_pool_sha256','broad_pool_key')
HISTORICAL_BRANCH_FIELDS=('ik_branch_slot','branch_key','branch_pool_file','branch_pool_sha256','branch_pool_key','branch_rule_version')


@lru_cache(maxsize=8)
def reach_models(assets):
    """Immutable asset-path cache. Caller authenticates assets before preparing a run."""
    assets=Path(assets).resolve();result={}
    for role,name in (('dropper','dropper.urdf'),('bouncer','board_arm.urdf')):
        path=assets/name;origins,axes,_,_=read_chain(path)
        if not np.allclose(axes[5],[0,0,1],atol=1e-12,rtol=0):raise ValueError('Wrist bound requires the pinned axial sixth revolute joint')
        local=origins[5,:3,:3].T@origins[5,:3,3]
        upstream=float(sum(np.linalg.norm(t) for t in origins[1:5,:3,3]))
        transverse=float(np.linalg.norm(local[:2]));axial=float(local[2])
        tolerance=1e-4+2*abs(axial)*math.sin(4e-4/2)
        result[role]=dict(urdf_file=name,urdf_sha256=sha(path),shoulder_m=origins[0,:3,3].tolist(),
            upstream_length_upper_m=upstream,terminal_axial_m=axial,terminal_transverse_bound_m=transverse,
            accepted_pose_allowance_m=tolerance,outer_radius_m=upstream+transverse+tolerance)
    return result


def wrist_bound(model,target):
    target=np.asarray(target,float)
    if target.shape!=(4,4) or not np.isfinite(target).all():raise ValueError('Finite full-pose target required')
    wrist=target[:3,3]-target[:3,2]*model['terminal_axial_m']
    distance=float(np.linalg.norm(wrist-np.asarray(model['shoulder_m'])))
    return dict(**copy.deepcopy(model),nominal_wrist_m=wrist.tolist(),distance_from_shoulder_m=distance,
        outside_by_m=distance-model['outer_radius_m'],passed=distance<=model['outer_radius_m'])


def reach_bound(assets,scene,bucket_xy,values):
    _,targets=yaw_target(scene,bucket_xy,values);models=reach_models(str(Path(assets).resolve()))
    arms=dict(dropper=wrist_bound(models['dropper'],targets['dropper_flange_world']),
              bouncer=wrist_bound(models['bouncer'],targets['board_flange_local']))
    return dict(version=REACH_VERSION,passed=all(v['passed'] for v in arms.values()),arms=arms,
        scope='Necessary full-pose outer bound only; passing does not certify IK, limits, collisions or a path')


def drop_gap_interval(assets,scene,bucket_xy,board_xyz,tilt_deg,gap_bounds=(.05,.65)):
    """Conditional outer-bound interval; sample inside it, never clip sampled actions."""
    lo,hi=map(float,gap_bounds)
    if not 0<lo<=hi or not np.isfinite([lo,hi]).all():raise ValueError('Positive bounded drop gap required')
    _,targets=yaw_target(scene,bucket_xy,[*board_xyz,tilt_deg,lo]);models=reach_models(str(Path(assets).resolve()))
    if not wrist_bound(models['bouncer'],targets['board_flange_local'])['passed']:return None
    model=models['dropper'];target=targets['dropper_flange_world']
    wrist=target[:3,3]-target[:3,2]*model['terminal_axial_m'];offset=wrist-np.asarray(model['shoulder_m'])
    square=model['outer_radius_m']**2-float(offset[:2]@offset[:2])
    if square<0:return None
    radius=math.sqrt(square);bottom=max(lo,lo-offset[2]-radius);top=min(hi,lo-offset[2]+radius)
    return None if top<bottom else (float(bottom),float(top))


def drop_gap_cap(assets,scene,bucket_xy,board_xyz,tilt_deg,gap_bounds=(.05,.65)):
    interval=drop_gap_interval(assets,scene,bucket_xy,board_xyz,tilt_deg,gap_bounds)
    return None if interval is None else interval[1]


def distinct_indices(pool):
    selected=[]
    for index in sorted(np.flatnonzero(pool['valid']),key=lambda i:(float(pool['residual'][i]),int(i))):
        if not any(np.max(wrapped_difference(pool['q'][index],pool['q'][old]))<.05 for old in selected):selected.append(int(index))
    return selected


def bouncer_order(pool,hint):
    remaining=distinct_indices(pool);selected=[];reference=np.asarray(hint)
    while remaining:
        def key(index):
            references=[pool['q'][old] for old in selected] or [reference]
            proximal=min(float(wrapped_difference(pool['q'][index][:3],r[:3]).max()) for r in references)
            full=min(float(wrapped_difference(pool['q'][index],r).max()) for r in references)
            return (proximal,full,-float(pool['residual'][index]),-index)
        chosen=max(remaining,key=key);remaining.remove(chosen);selected.append(chosen)
    return selected


def action_seed(bucket,values):
    return (BRANCH_RULE['seed_base']+int(digest(dict(bucket_xy_m=list(bucket),values=list(values)))[:8],16))%2147483647


class BroadResolver:
    def __init__(self,legacy,sources,connections,assets):
        self.legacy,self.base=legacy,legacy.base;self.parent_rows=sources;self.parent_connections=connections
        self.assets=Path(assets);self.pending=deque();self._pools={};self._connections={}
    def __getattr__(self,name):return getattr(self.legacy,name)

    def solve(self,bucket_xy,values,hint):
        bucket=_vector(bucket_xy,2,'bucket');values=_vector(values,5,'values');solver=self.base.solver
        info=dict(passed=False,reason=None,candidates=[],pair_attempts=[],branch_rule=copy.deepcopy(BRANCH_RULE))
        if (values<solver.low-1e-9).any() or (values>solver.high+1e-9).any():info['reason']='action_bounds_rejected';return info
        if np.linalg.norm(bucket-values[:2])<=.15:info['reason']='board_bucket_distance_constraint_rejected';return info
        reach=reach_bound(self.assets,self.scene,bucket,values);info['reach_bound']=reach
        if not reach['passed']:info['reason']='proven_full_pose_wrist_reach_rejected';return info
        hint=source_geometry(hint);_,targets=yaw_target(self.scene,bucket,values);seed=action_seed(bucket,values);pools=[]
        for arm,target,initial,known in ((solver.ik.dropper,targets['dropper_flange_world'],solver.initial['dropper_joint_rad'],hint['q10']),
                (solver.ik.board,targets['board_flange_local'],solver.initial['board_joint_rad'],hint['q14'])):
            result=arm.solve_candidates(target[None],initial,known=np.asarray(known)[None],starts=32,iterations=360,seed=seed,
                normal_only=False,preserve_known=np.asarray([False]))
            pools.append({k:result[k][0].cpu().numpy() for k in ('q','valid','residual')})
        first,second=pools;order14=bouncer_order(second,hint['q14'])
        order10=sorted(distinct_indices(first),key=lambda i:(float(wrapped_difference(first['q'][i],hint['q10']).max()),float(first['residual'][i]),i))
        info.update(targets={k:v.tolist() for k,v in targets.items()},seed=seed,geometry_hint_sha256=digest(hint),
            valid_ik_branches=[int(p['valid'].sum()) for p in pools],distinct_ik_branches=[len(order10),len(order14)],
            bouncer_branch_order=order14,dropper_branch_order=order10,ik_pools=[{k:v.tolist() for k,v in p.items()} for p in pools],
            selection_uses_flight_labels=False)
        bad10,bad14=set(),set()
        for i,j in ((i,j) for i in order10 for j in order14):
            if i in bad10 or j in bad14:continue
            if len(info['pair_attempts'])>=64:break
            q10,q14=first['q'][i],second['q'][j];q=np.r_[q10,q14]
            checked=solver.endpoint.check(q10,q14,bucket,values,targets)
            attempt=dict(branch10=i,branch14=j,checked=checked);info['pair_attempts'].append(attempt)
            if checked['reason']:
                for role in checked.get('independent_reject_arms',[]):(bad10 if role==10 else bad14).add(i if role==10 else j)
                continue
            errors=[native_pose_error(solver.gate,role,q,target) for role,target in ((10,targets['dropper_flange_world']),(14,base14(self.scene)@targets['board_flange_local']))]
            attempt['native_pose_errors']=errors
            if any(e['position_m']>=1e-4 or .25*e['rotation_rad']>=1e-4 for e in errors):attempt['pose_rejected']=True;continue
            info['candidates'].append(dict(values=values.tolist(),bucket_xy_m=bucket.tolist(),q10=q10.tolist(),q14=q14.tolist(),
                geometry_eligible=True,branch10=i,branch14=j,endpoint=checked))
            if len(info['candidates'])>=16:break
        info['passed']=bool(info['candidates']);info['reason']=None if info['passed'] else ('finite_ik_unresolved' if not first['valid'].any() or not second['valid'].any() else 'strict_endpoint_clearance_or_pose_rejected')
        return info

    def pool(self,anchor,values):
        verify_resolution(anchor,self.sources,self.home);self.base._check_resolution_bytes(anchor)
        bucket=anchor['bucket_xy_m'];delta=np.asarray(bucket)-np.asarray(values[:2])
        valid=np.linalg.norm(delta)>.15 and (np.asarray(values)>=self.solver.low-1e-9).all() and (np.asarray(values)<=self.solver.high+1e-9).all()
        yaw,targets=yaw_target(self.scene,bucket,values) if valid else (float(math.atan2(delta[1],delta[0])),None)
        behavior=dict(version=BEHAVIOR_VERSION,yaw_rad=yaw,rotation_matrix=(Rotation.from_euler('z',yaw).as_matrix()@DOWNWARD).tolist(),
            target_dropper_flange_world=None if targets is None else targets['dropper_flange_world'].tolist(),geometry_hint_only=True)
        behavior_key=digest(dict(version=BEHAVIOR_VERSION,connection_policy=CONNECTION_POLICY_VERSION,resolution_sha256=anchor['sha256'],values=values,bucket_xy_m=bucket,release_behavior=behavior))
        key=digest(dict(version=BROAD_VERSION,branch_rule=BRANCH_RULE,behavior_key=behavior_key,resolution_sha256=anchor['sha256']))
        if key in self._pools:return copy.deepcopy(self._pools[key])
        start=time.monotonic();timing=dict(home_s=0.,endpoint_s=0.,path_s=0.,wall_s=0.)
        pool=dict(version=BROAD_VERSION,pool_key=key,branch_rule=copy.deepcopy(BRANCH_RULE),behavior_key=behavior_key,release_behavior=behavior,
            bucket_xy_m=bucket,values=values,resolution_file=anchor['file'],resolution_sha256=anchor['sha256'],home_sha256=self.home['sha256'],
            endpoint_attempts=None,path_attempts=[],admitted=[],reason=None,timing=timing,selection_uses_flight_labels=False)
        if not anchor['home_passed']:pool['reason']='fixed_base_clearance_rejected' if anchor.get('hard_reject') else 'canonical_home_unresolved_in_layout'
        else:
            hint=source_geometry(anchor['effective_source_hint'])
            if anchor.get('resolved'):hint.update({k:anchor['resolved'][k] for k in ('q10','q14','values','bucket_xy_m')})
            then=time.monotonic();endpoint=self.solve(bucket,values,hint);timing['endpoint_s']=time.monotonic()-then
            pool['endpoint_attempts']=endpoint;pool['reason']=endpoint['reason']
            for index,row in enumerate(endpoint['candidates'][:16]):
                if any(np.max(wrapped_difference(row['q14'],old['q14']))<.05 for old in pool['admitted']):continue
                q=np.r_[row['q10'],row['q14']];then=time.monotonic()
                if anchor['passed']:
                    aq=np.r_[anchor['resolved']['q10'],anchor['resolved']['q14']]
                    if np.array_equal(aq,q):proof=compose_anchor_prefix(anchor,self.home,q)
                    else:
                        suffix=self.planner.plan(q,bucket,start=aq)
                        if suffix['passed']:proof=compose_anchor_prefix(anchor,self.home,q,suffix)
                        else:
                            failed=dict(passed=False,reason='finite_anchor_suffix_unresolved',connection_route=dict(kind='anchor_suffix_rejected',prefix=anchor_prefix_identity(anchor),target_q=q.tolist(),suffix_proof=suffix))
                            suffix_index=len(pool['path_attempts']);pool['path_attempts'].append(failed)
                            proof=copy.deepcopy(self.planner.plan(q,bucket));proof['connection_route']=dict(kind='full_home_after_anchor_suffix',prefix=anchor_prefix_identity(anchor),suffix_attempt_index=suffix_index)
                else:proof=self.planner.plan(q,bucket)
                timing['path_s']+=time.monotonic()-then;proof_index=len(pool['path_attempts']);pool['path_attempts'].append(proof)
                if proof['passed']:
                    pool['admitted'].append(dict(slot=len(pool['admitted']),candidate_index=index,path_attempt_index=proof_index,q10=row['q10'],q14=row['q14'],endpoint=row['endpoint']))
                    if len(pool['admitted'])>=2:break
            if endpoint['passed'] and not pool['admitted']:pool['reason']='finite_requested_setup_path_unresolved'
        timing['wall_s']=time.monotonic()-start;pool.update(self.base._save('broad-pools',key,pool));self._pools[key]=copy.deepcopy(pool);return pool

    def prepared_connection(self,anchor,pool,slot):
        selected=pool['admitted'][slot]
        key=digest(dict(version=BROAD_VERSION,pool_key=pool['pool_key'],broad_pool_sha256=pool['sha256'],broad_branch_slot=slot))
        if key in self._connections:return copy.deepcopy(self._connections[key])
        record=dict(version=VERSION,bucket_xy_m=pool['bucket_xy_m'],values=pool['values'],resolution_file=anchor['file'],resolution_sha256=anchor['sha256'],
            home_sha256=self.home['sha256'],passed=True,geometry_eligible=True,endpoint_eligible=True,reason=None,q10=selected['q10'],q14=selected['q14'],
            proof=copy.deepcopy(pool['path_attempts'][selected['path_attempt_index']]),endpoint=selected['endpoint'],
            endpoint_attempts=copy.deepcopy(pool['endpoint_attempts']),path_attempts=copy.deepcopy(pool['path_attempts']),timing=copy.deepcopy(pool['timing']),
            action_substituted=False,connection_policy=CONNECTION_POLICY_VERSION,anchor_values=anchor['anchor_values'],anchor_derivation_sha256=digest(anchor['anchor_derivation']),
            release_behavior=pool['release_behavior'],behavior_key=pool['behavior_key'],broad_rule_version=BROAD_VERSION,broad_key=key,broad_branch_slot=slot,
            broad_pool_file=pool['file'],broad_pool_sha256=pool['sha256'],broad_pool_key=pool['pool_key'])
        record.update(self.base._save('fixed-yaw-connections',key,record));verify_connection(record,anchor,self.home,pool['values'])
        self._connections[key]=copy.deepcopy(record);return record

    def connect(self,anchor,values):
        if not self.pending:raise ValueError('Explicit frozen broad request identity required')
        request=self.pending.popleft();source=self.parent_rows[request['parent_row_id']]
        if (request['parent_row_sha256']!=digest(source) or request['bucket_xy_m']!=anchor['bucket_xy_m']
                or request['values']!=list(values)):raise ValueError('Prepared request lineage or action changed')
        if request.get('exact_parent_action'):
            if source['values']!=list(values):raise ValueError('Exact preserved action changed')
            return dict(copy.deepcopy(self.parent_connections[source['id']]),cache_hit=True)
        if request.get('native_prepared') is not True:raise ValueError('No native flight without completed separate preparation')
        path=contained(self.base.directory,request['prepared_connection_file'])
        if sha(path)!=request['prepared_connection_sha256']:raise ValueError('Prepared immutable connection bytes changed')
        connection=read(path);connection.update(file=request['prepared_connection_file'],sha256=request['prepared_connection_sha256'])
        verify_connection(connection,anchor,self.home,values)
        if not connection['passed'] or connection['broad_rule_version']!=BROAD_VERSION:raise ValueError('Admitted broad connection required')
        for key in BROAD_FIELDS:
            if request.get(key)!=connection[key]:raise ValueError('Prepared broad identity differs: '+key)
        for key in ('q10','q14'):
            if request.get('selected_'+key)!=connection[key]:raise ValueError('Prepared joint branch changed')
        return dict(connection,cache_hit=True)


class BroadEngine(BouncerEngine):
    def __init__(self,ref,assets,directory,**kwargs):
        super().__init__(ref,assets,directory,**kwargs)
        old=self.adapter.resolver
        self.adapter.resolver=BroadResolver(old.legacy,self.parent_rows,old.parent_connections,assets)
        self._prepared_ids=set()

    def prepare(self,proposals):
        """CPU-only; one record per numeric proposal, zero, one or two native packets."""
        if self.adapter.resolver.pending:raise RuntimeError('Cannot prepare during native request consumption')
        initial_env=self.adapter.env;eligible=[];preparations=[]
        for proposal in proposals:
            identifier=proposal.get('id')
            if not isinstance(identifier,str) or not identifier or identifier in self._prepared_ids:raise ValueError('Unique preparation ID required')
            if proposal.get('exact_parent_action') or proposal.get('phase')!='search':raise ValueError('Only new search proposals require preparation')
            source=self.parent_rows[proposal['parent_row_id']]
            if proposal['parent_row_sha256']!=digest(source) or proposal['bucket_xy_m']!=source['bucket_xy_m']:raise ValueError('Preparation parent/layout binding changed')
            values=_vector(proposal['values'],5,'values').tolist();anchor,_=self.adapter.resolve_context(proposal)
            pool=self.adapter.resolver.pool(anchor,values);prepared=[]
            for slot in range(len(pool['admitted'])):
                connection=self.adapter.resolver.prepared_connection(anchor,pool,slot)
                packet=copy.deepcopy(proposal);packet.update(id=identifier+'-b'+str(slot),proposal_id=identifier,values=values,command_values=values,
                    native_prepared=True,prepared_connection_file=connection['file'],prepared_connection_sha256=connection['sha256'],
                    selected_q10=connection['q10'],selected_q14=connection['q14'],
                    source_hint=copy.deepcopy(anchor['source_hint']),effective_source_hint=copy.deepcopy(anchor['effective_source_hint']),source_teacher_id=anchor['source_hint']['id'],
                    context_resolution_file=anchor['file'],context_resolution_sha256=anchor['sha256'],context_home_sha256=self.adapter.resolver.home['sha256'],
                    context_layout_eligible=anchor['layout_eligible'],context_anchor_admitted=anchor['anchor_admitted'],context_hard_reject=anchor['hard_reject'])
                packet.update({key:connection[key] for key in BROAD_FIELDS});prepared.append(packet);eligible.append(packet)
            preparations.append(dict(proposal_id=identifier,proposal_sha256=digest(proposal),context_id=proposal['context_id'],values=values,bucket_xy_m=proposal['bucket_xy_m'],
                broad_pool_file=pool['file'],broad_pool_sha256=pool['sha256'],broad_pool_key=pool['pool_key'],
                admitted=len(prepared),admitted_request_ids=[r['id'] for r in prepared],admitted_request_sha256=[digest(r) for r in prepared],
                reason=None if prepared else pool['reason'],timing=pool['timing'],native_releases=0,preparation_only=True))
            self._prepared_ids.add(identifier)
        if self.adapter.env is not initial_env:raise RuntimeError('Preparation initialized native physics')
        return eligible,preparations

    def evaluate(self,requests,phase):
        resolver=self.adapter.resolver
        if resolver.pending:raise RuntimeError('Unfinished prepared request queue')
        resolver.pending=deque(copy.deepcopy(requests))
        try:
            rows,timing=RecoveryEngine.evaluate(self,requests,phase)
            if resolver.pending:raise RuntimeError('Not every frozen native request was consumed')
        finally:resolver.pending.clear()
        for row in rows:
            connection=read(self.directory/row['connection_file'])
            if row.get('exact_parent_action'):
                # Make retained historical branch metadata explicit; never invent another branch.
                for key in HISTORICAL_BRANCH_FIELDS:
                    if key in connection:
                        if key in row and row[key]!=connection[key]:raise RuntimeError('Preserved historical branch metadata conflict')
                        row[key]=copy.deepcopy(connection[key])
            else:
                if not row.get('native_release'):raise RuntimeError('A prepared admitted packet unexpectedly failed native admission')
                for key in BROAD_FIELDS:
                    if row.get(key)!=connection[key]:raise RuntimeError('Native broad proof metadata changed')
        return rows,timing
