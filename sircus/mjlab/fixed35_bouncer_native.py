"""Replay saved alternative bouncer joints without invoking IK or changing shots.

Every candidate was already a strict endpoint candidate in the immutable six-case
study. This adapter rechecks that exact pair and its sampled setup path, leaving
the dropper joints, target geometry, yaw, native64 physics and v3 score unchanged.
"""
from __future__ import annotations
import copy
from collections import deque
from pathlib import Path
import time
import numpy as np

from .admission import native_pose_error, base14
from .benchmark_verify import digest, read
from .calibrated_admission import (VERSION, CONNECTION_POLICY_VERSION,
    anchor_search_spec, verify_resolution, verify_connection, compose_anchor_prefix,
    anchor_prefix_identity, _vector)
from .fixed35_imitation_native import FixedYawEngine, BEHAVIOR_VERSION, yaw_target
from .fixed35_recovery_native import RecoveryEngine
from .fixed35_six_native import BRANCH_VERSION, BRANCH_RULE
from .table_correction_native import copy_bound

BOUNCER_VERSION = 'sircus-fixed35-saved-bouncer-candidate-v1'


def wrapped_difference(first, second):
    return np.abs((np.asarray(first)-np.asarray(second)+np.pi) % (2*np.pi)-np.pi)


def bouncer_key(anchor, source, candidate_index, candidate_sha256, pool_sha256):
    return digest(dict(version=BOUNCER_VERSION,resolution_sha256=anchor['sha256'],
        parent_connection_sha256=source['connection_sha256'],candidate_index=candidate_index,
        candidate_sha256=candidate_sha256,candidate_pool_sha256=pool_sha256,
        values=source['values'],bucket_xy_m=source['bucket_xy_m']))


class CandidateResolver:
    """Bind each sequential adapter call to its frozen request, preserving order."""
    def __init__(self, legacy, sources, connections, pools):
        self.legacy,self.base = legacy,legacy.base
        self.parent_rows,self.parent_connections,self.parent_pools = sources,connections,pools
        self.pending = deque()
        self._candidates = {}

    def __getattr__(self,name):
        return getattr(self.legacy,name)

    def connect(self,anchor,requested_values):
        if not self.pending: raise ValueError('Saved-candidate request identity required')
        request = self.pending.popleft()
        source = self.parent_rows[request['parent_row_id']]
        verify_resolution(anchor,self.sources,self.home)
        self.base._check_resolution_bytes(anchor)
        values = _vector(requested_values,5,'requested values').tolist()
        if (request['values']!=values or source['values']!=values or anchor['bucket_xy_m']!=source['bucket_xy_m']
                or request['parent_row_sha256']!=digest(source)
                or anchor['sha256']!=source['context_resolution_sha256']):
            raise ValueError('Saved bouncer replay changed original action, layout or resolution')
        original = self.parent_connections[source['id']]
        if request.get('exact_parent_action'):
            return dict(copy.deepcopy(original),cache_hit=True)
        pool = self.parent_pools.get(source['id'])
        if pool is None: raise ValueError('New bouncer candidate must reference a saved six-case pool')
        index = request.get('candidate_index')
        if type(index) is not int or not 0<=index<len(pool['endpoint_attempts']['candidates']):
            raise ValueError('Saved endpoint candidate index required')
        candidate = copy.deepcopy(pool['endpoint_attempts']['candidates'][index])
        for key,expected in (('candidate_pool_file',pool['file']),('candidate_pool_sha256',pool['sha256']),
                ('candidate_sha256',digest(candidate)),('selected_q10',candidate['q10']),('selected_q14',candidate['q14'])):
            if request.get(key)!=expected: raise ValueError('Frozen candidate binding changed: '+key)
        if (candidate['values']!=values or candidate['bucket_xy_m']!=source['bucket_xy_m']
                or candidate['q10']!=source['q10']
                or np.max(wrapped_difference(candidate['q14'],source['q14']))<.05):
            raise ValueError('Only a distinct saved bouncer branch with identical dropper joints is allowed')
        key = bouncer_key(anchor,source,index,digest(candidate),pool['sha256'])
        if key in self._candidates: return dict(copy.deepcopy(self._candidates[key]),cache_hit=True)
        begin = time.monotonic();timing=dict(home_s=0.,endpoint_s=0.,path_s=0.,wall_s=0.)
        record = dict(version=VERSION,bucket_xy_m=source['bucket_xy_m'],values=values,
            resolution_file=anchor['file'],resolution_sha256=anchor['sha256'],home_sha256=self.home['sha256'],
            passed=False,geometry_eligible=False,endpoint_eligible=False,reason=None,q10=[0.]*6,q14=[0.]*6,
            proof=None,endpoint_attempts=None,path_attempts=[],timing=timing,action_substituted=False,
            connection_policy=CONNECTION_POLICY_VERSION,anchor_values=anchor['anchor_values'],
            anchor_derivation_sha256=digest(anchor['anchor_derivation']),
            release_behavior=copy.deepcopy(original['release_behavior']),behavior_key=original['behavior_key'],
            bouncer_rule_version=BOUNCER_VERSION,bouncer_key=key,candidate_index=index,
            candidate_sha256=digest(candidate),candidate_pool_file=pool['file'],candidate_pool_sha256=pool['sha256'],
            parent_row_id=source['id'],parent_row_sha256=digest(source),
            parent_connection_file=source['connection_file'],parent_connection_sha256=source['connection_sha256'],
            selected_q10=candidate['q10'],selected_q14=candidate['q14'],ik_solve_performed=False)
        if not anchor['home_passed']:
            record['reason']='fixed_base_clearance_rejected' if anchor.get('hard_reject') else 'canonical_home_unresolved_in_layout'
        else:
            _,targets = yaw_target(self.scene,source['bucket_xy_m'],values)
            q10,q14 = candidate['q10'],candidate['q14'];q=np.r_[q10,q14]
            start=time.monotonic()
            checked=self.solver.endpoint.check(q10,q14,source['bucket_xy_m'],values,targets)
            errors=[native_pose_error(self.solver.gate,role,q,target) for role,target in (
                (10,targets['dropper_flange_world']),(14,base14(self.scene)@targets['board_flange_local']))]
            accepted=not checked['reason'] and all(e['position_m']<1e-4 and .25*e['rotation_rad']<1e-4 for e in errors)
            fresh=dict(copy.deepcopy(candidate),endpoint=checked)
            endpoint=dict(passed=accepted,reason=None if accepted else 'saved_candidate_endpoint_recheck_rejected',
                source_candidate=copy.deepcopy(candidate),source_candidate_sha256=digest(candidate),
                targets={k:v.tolist() for k,v in targets.items()},candidates=[fresh] if accepted else [],
                fresh_endpoint=checked,native_pose_errors=errors,ik_solve_performed=False,
                selection_uses_flight_labels=False)
            record['endpoint_attempts']=endpoint;record['endpoint_eligible']=accepted
            record['reason']=endpoint['reason'];timing['endpoint_s']=time.monotonic()-start
            if accepted:
                start=time.monotonic()
                if anchor['passed']:
                    aq=np.r_[anchor['resolved']['q10'],anchor['resolved']['q14']]
                    if np.array_equal(aq,q):
                        proof=compose_anchor_prefix(anchor,self.home,q)
                    else:
                        suffix=self.planner.plan(q,source['bucket_xy_m'],start=aq)
                        if suffix['passed']:
                            proof=compose_anchor_prefix(anchor,self.home,q,suffix)
                        else:
                            record['path_attempts'].append(dict(passed=False,reason='finite_anchor_suffix_unresolved',
                                connection_route=dict(kind='anchor_suffix_rejected',prefix=anchor_prefix_identity(anchor),
                                    target_q=q.tolist(),suffix_proof=suffix)))
                            proof=copy.deepcopy(self.planner.plan(q,source['bucket_xy_m']))
                            proof['connection_route']=dict(kind='full_home_after_anchor_suffix',prefix=anchor_prefix_identity(anchor),suffix_attempt_index=0)
                else:
                    proof=self.planner.plan(q,source['bucket_xy_m'])
                timing['path_s']=time.monotonic()-start
                record['path_attempts'].append(proof)
                if proof['passed']:
                    record.update(passed=True,geometry_eligible=True,reason=None,q10=q10,q14=q14,proof=proof,endpoint=checked)
                else:record['reason']='finite_saved_bouncer_setup_path_unresolved'
        timing['wall_s']=time.monotonic()-begin
        record.update(self.base._save('fixed-yaw-connections',key,record))
        verify_connection(record,anchor,self.home,values)
        self._candidates[key]=copy.deepcopy(record)
        return dict(record,cache_hit=False)


class BouncerEngine(RecoveryEngine):
    """No IK calls: hydrate original evidence and certify an exact saved q pair."""
    def __init__(self,ref,assets,directory,*,parent,sources,source_roots,**kwargs):
        FixedYawEngine.__init__(self,ref,assets,directory,**kwargs)
        self.parent_rows={r['id']:copy.deepcopy(r) for r in sources}
        if len(self.parent_rows)!=len(sources) or set(source_roots)!=set(self.parent_rows):
            raise ValueError('Unique source IDs with explicit original evidence directories required')
        resolver=self.adapter.resolver;connections={};pools={};self.source_receipts=[]
        for source in sources:
            origin=Path(source_roots[source['id']])
            resolution=copy_bound(origin,directory,source['context_resolution_file'],source['context_resolution_sha256'])
            verify_resolution(resolution,resolver.sources,resolver.home)
            key=digest(dict(version=VERSION,bucket_xy_m=source['bucket_xy_m'],home_sha256=resolver.home['sha256'],
                sources=resolver.sources,bounds=resolver.bounds,anchor_search=anchor_search_spec()))
            if resolution['key']!=key or resolution['bucket_xy_m']!=source['bucket_xy_m']:
                raise ValueError('Parent geometry resolution changed')
            resolver.base._resolved[key]=copy.deepcopy(resolution)
            connection=copy_bound(origin,directory,source['connection_file'],source['connection_sha256'])
            verify_connection(connection,resolution,resolver.home,source['values'])
            behavior=connection['release_behavior']
            behavior_key=digest(dict(version=BEHAVIOR_VERSION,connection_policy=CONNECTION_POLICY_VERSION,
                resolution_sha256=resolution['sha256'],values=source['values'],bucket_xy_m=source['bucket_xy_m'],release_behavior=behavior))
            if behavior['version']!=BEHAVIOR_VERSION or connection['behavior_key']!=behavior_key:
                raise ValueError('Original bucket-facing behavior changed')
            key=behavior_key
            if connection.get('branch_rule_version') is not None:
                if connection['branch_rule_version']!=BRANCH_VERSION:raise ValueError('Unknown historical branch proof')
                pool=copy_bound(origin,directory,connection['branch_pool_file'],connection['branch_pool_sha256'])
                if pool['branch_rule']!=BRANCH_RULE or pool['behavior_key']!=behavior_key:
                    raise ValueError('Original finite branch pool changed')
                poolkey=digest(dict(version=BRANCH_VERSION,branch_rule=BRANCH_RULE,behavior_key=behavior_key,resolution_sha256=resolution['sha256']))
                if pool['pool_key']!=poolkey or pool['file']!='branch-pools/'+poolkey+'.json':raise ValueError('Original pool identity changed')
                slot=connection['ik_branch_slot']
                key=digest(dict(version=BRANCH_VERSION,pool_key=poolkey,branch_pool_sha256=pool['sha256'],ik_branch_slot=slot))
                if connection['branch_key']!=key:raise ValueError('Original branch key changed')
                if connection['passed']:
                    branch=pool['admitted'][slot]
                    if any(connection[k]!=branch[k] for k in ('q10','q14','endpoint')):
                        raise ValueError('Original admitted joint pair changed')
                pools[source['id']]=copy.deepcopy(pool)
            if source['connection_file']!='fixed-yaw-connections/'+key+'.json':
                raise ValueError('Original proof namespace changed')
            connections[source['id']]=copy.deepcopy(connection)
            self.source_receipts.append(dict(source_id=source['id'],source_row_sha256=digest(source),
                resolution_file=source['context_resolution_file'],resolution_sha256=source['context_resolution_sha256'],
                connection_file=source['connection_file'],connection_sha256=source['connection_sha256']))
        self.parent_options=read(Path(parent)/'native-yaw/effective-options.json')
        self.parent_geometry=read(Path(parent)/'native-yaw/native-compiled-geometry.json')
        self.adapter.resolver=CandidateResolver(resolver,self.parent_rows,connections,pools)

    def evaluate(self,requests,phase):
        if self.adapter.resolver.pending:raise RuntimeError('Unfinished saved-candidate request queue')
        self.adapter.resolver.pending=deque(copy.deepcopy(requests))
        try:
            rows,timing=super().evaluate(requests,phase)
            if self.adapter.resolver.pending:raise RuntimeError('Not every saved-candidate request was consumed')
        finally:
            self.adapter.resolver.pending.clear()
        for row in rows:
            if row.get('exact_parent_action'):continue
            connection=read(self.directory/row['connection_file'])
            if connection.get('bouncer_rule_version')!=BOUNCER_VERSION or connection['bouncer_key']!=Path(row['connection_file']).stem:
                raise RuntimeError('Saved bouncer branch proof lost')
            row.update(bouncer_rule_version=BOUNCER_VERSION,bouncer_key=connection['bouncer_key'],ik_solve_performed=False)
            if row.get('native_release') and any(row[k]!=row['selected_'+k] for k in ('q10','q14')):
                raise RuntimeError('Native release changed saved candidate joints')
        return rows,timing
