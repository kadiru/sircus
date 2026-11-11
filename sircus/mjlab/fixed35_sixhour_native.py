"""Versioned eight-bouncer-branch extension; immutable broad native physics.

Only the number of distinct admitted q14 branches changes. Historical confirmed
broad recipes are hydrated with their exact original evidence and joints. CPU
preparation never advances physics; inherited evaluation consumes frozen proofs.
"""
from __future__ import annotations
import copy
from collections import deque
from pathlib import Path
import math
import time
import numpy as np
from scipy.spatial.transform import Rotation
from . import fixed35_broad_native as broad
from .fixed35_broad_native import (reach_bound,drop_gap_cap,drop_gap_interval,reach_models,
    wrist_bound,distinct_indices,bouncer_order,action_seed,BROAD_FIELDS,HISTORICAL_BRANCH_FIELDS)
from .admission import native_pose_error,base14
from .benchmark_verify import contained,digest,read,sha
from .calibrated_admission import (VERSION,CONNECTION_POLICY_VERSION,verify_resolution,verify_connection,
    source_geometry,compose_anchor_prefix,anchor_prefix_identity,_vector,anchor_search_spec)
from .fixed35_imitation_native import BEHAVIOR_VERSION,yaw_target
from .fixed35_bouncer_native import wrapped_difference
from .gripper_yaw_geometry import DOWNWARD
from .table_correction_native import copy_bound

SIXHOUR_VERSION='sircus-fixed35-sixhour-bouncer-branches-v1'
SIXHOUR_RULE=dict(broad.BRANCH_RULE,version=SIXHOUR_VERSION,maximum_admitted_branches=8)
BROAD_VERSION=SIXHOUR_VERSION
BRANCH_RULE=SIXHOUR_RULE
BROAD_RULE=SIXHOUR_RULE

class SixHourResolver:
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
                    if len(pool['admitted'])>=BRANCH_RULE['maximum_admitted_branches']:break
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


def hydrate_broad_sources(engine,sources,source_roots):
    """Authenticate original V1 broad proofs, never reinterpret them as new ones."""
    resolver=engine.adapter.resolver
    ids=[r['id'] for r in sources]
    if len(set(ids))!=len(ids) or set(source_roots)!=set(ids) or set(ids)&set(engine.parent_rows):
        raise ValueError('Unique additional source IDs and explicit evidence directories required')
    staged=[]
    for source in sources:
        if (source.get('native_release') is not True or source.get('route_success') is not True
                or source.get('invalid') or source.get('invalid_physics')):
            raise ValueError('Additional preserved broad recipe must be an authenticated native capture')
        origin=Path(source_roots[source['id']])
        resolution=copy_bound(origin,engine.directory,source['context_resolution_file'],source['context_resolution_sha256'])
        verify_resolution(resolution,resolver.sources,resolver.home)
        resolution_key=digest(dict(version=VERSION,bucket_xy_m=source['bucket_xy_m'],home_sha256=resolver.home['sha256'],
            sources=resolver.sources,bounds=resolver.bounds,anchor_search=anchor_search_spec()))
        if resolution['key']!=resolution_key or resolution['bucket_xy_m']!=source['bucket_xy_m']:
            raise ValueError('Original broad resolution changed')
        connection=copy_bound(origin,engine.directory,source['connection_file'],source['connection_sha256'])
        verify_connection(connection,resolution,resolver.home,source['values'])
        if not connection['passed'] or connection.get('broad_rule_version')!=broad.BROAD_VERSION:
            raise ValueError('Only original V1 broad admitted proofs can be retained')
        behavior=connection['release_behavior']
        behavior_key=digest(dict(version=BEHAVIOR_VERSION,connection_policy=CONNECTION_POLICY_VERSION,
            resolution_sha256=resolution['sha256'],values=source['values'],bucket_xy_m=source['bucket_xy_m'],release_behavior=behavior))
        if behavior['version']!=BEHAVIOR_VERSION or connection['behavior_key']!=behavior_key:
            raise ValueError('Original bucket-facing behavior changed')
        pool=copy_bound(origin,engine.directory,connection['broad_pool_file'],connection['broad_pool_sha256'])
        pool_key=digest(dict(version=broad.BROAD_VERSION,branch_rule=broad.BRANCH_RULE,behavior_key=behavior_key,resolution_sha256=resolution['sha256']))
        if (pool['version']!=broad.BROAD_VERSION or pool['branch_rule']!=broad.BRANCH_RULE
                or pool['behavior_key']!=behavior_key or pool['pool_key']!=pool_key
                or pool['file']!='broad-pools/'+pool_key+'.json'):
            raise ValueError('Original broad pool identity changed')
        expected_pool=dict(values=source['values'],bucket_xy_m=source['bucket_xy_m'],
            resolution_file=source['context_resolution_file'],resolution_sha256=source['context_resolution_sha256'],
            home_sha256=resolver.home['sha256'],release_behavior=behavior,selection_uses_flight_labels=False)
        if any(pool.get(k)!=v for k,v in expected_pool.items()):
            raise ValueError('Original broad pool geometry or provenance changed')
        slot=connection['broad_branch_slot']
        if type(slot) is not int or not 0<=slot<len(pool['admitted']):raise ValueError('Original broad branch slot changed')
        selected=pool['admitted'][slot]
        if selected['slot']!=slot:raise ValueError('Original broad pool ordering changed')
        key=digest(dict(version=broad.BROAD_VERSION,pool_key=pool_key,broad_pool_sha256=pool['sha256'],broad_branch_slot=slot))
        if connection['broad_key']!=key or source['connection_file']!='fixed-yaw-connections/'+key+'.json':
            raise ValueError('Original broad connection namespace changed')
        for field in BROAD_FIELDS:
            if source.get(field)!=connection[field]:raise ValueError('Original broad metadata changed: '+field)
        for field in ('q10','q14','endpoint'):
            if connection[field]!=selected[field]:raise ValueError('Original broad selected branch changed: '+field)
        for field in ('q10','q14'):
            if source[field]!=connection[field]:raise ValueError('Original broad native command changed: '+field)
        if connection['proof']!=pool['path_attempts'][selected['path_attempt_index']]:
            raise ValueError('Original broad selected path changed')
        if (connection['endpoint_attempts']!=pool['endpoint_attempts'] or connection['path_attempts']!=pool['path_attempts']
                or connection['broad_pool_key']!=pool_key):
            raise ValueError('Original broad complete endpoint/path evidence changed')
        staged.append((source,resolution,connection,pool,resolution_key))
    for source,resolution,connection,pool,key in staged:
        engine.parent_rows[source['id']]=copy.deepcopy(source)
        resolver.parent_connections[source['id']]=copy.deepcopy(connection)
        resolver.base._resolved[key]=copy.deepcopy(resolution)
        engine.source_receipts.append(dict(source_id=source['id'],source_row_sha256=digest(source),
            resolution_file=source['context_resolution_file'],resolution_sha256=source['context_resolution_sha256'],
            connection_file=source['connection_file'],connection_sha256=source['connection_sha256'],
            broad_pool_file=pool['file'],broad_pool_sha256=pool['sha256'],broad_rule_version=broad.BROAD_VERSION,
            exact_historical_broad_recipe=True))


class SixHourEngine(broad.BroadEngine):
    def __init__(self,ref,assets,directory,*,additional_sources=(),additional_source_roots=None,**kwargs):
        super().__init__(ref,assets,directory,**kwargs)
        old=self.adapter.resolver
        self.adapter.resolver=SixHourResolver(old.legacy,self.parent_rows,old.parent_connections,assets)
        hydrate_broad_sources(self,list(additional_sources),additional_source_roots or {})

    def evaluate(self,requests,phase):
        rows,timing=super().evaluate(requests,phase)
        for row in rows:
            if not row.get('exact_parent_action'):continue
            connection=read(self.directory/row['connection_file'])
            for key in BROAD_FIELDS:
                if key in connection:
                    if key in row and row[key]!=connection[key]:raise RuntimeError('Preserved broad metadata conflict')
                    row[key]=copy.deepcopy(connection[key])
        return rows,timing
