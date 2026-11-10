"""Finite alternative full-pose IK branches, with unchanged native64 physics.

Branch slots select distinct joint configurations for the same exact shot. They
never relax endpoint/path clearance, replace an action, or use flight outcomes.
Preservation replays keep the original evidence bytes and selected joints.
"""
from __future__ import annotations

import copy
import math
import time
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from .admission import native_pose_error, base14
from .benchmark_verify import digest, read
from .calibrated_admission import (VERSION, CONNECTION_POLICY_VERSION,
    anchor_search_spec, verify_resolution, verify_connection, source_geometry,
    compose_anchor_prefix, anchor_prefix_identity, _vector, outer_geometry_mask,
    ordered_pairs)
from .calibrated_adapter import timing_record
from .fixed35_imitation_native import FixedYawEngine, BEHAVIOR_VERSION, yaw_target
from .fixed35_recovery_native import RecoveryEngine
from .gripper_yaw_geometry import DOWNWARD
from .table_correction_native import copy_bound

BRANCH_VERSION = "sircus-fixed35-distinct-ik-branches-v1"
BRANCH_RULE = dict(version=BRANCH_VERSION, seeds=[20261110, 20261111],
    starts_per_seed=32, iterations=360, normal_only=False, preserve_known=False,
    maximum_unique_pairs=64, maximum_endpoint_candidates=12,
    maximum_path_candidates=12, maximum_admitted_branches=2,
    distinct_joint_tolerance_rad=0.05, periodic_equivalence=True,
    selection_uses_flight_labels=False)


def same_branch(first, second):
    """Treat 2π wrist wraps as one geometric branch, keeping exact saved joints."""
    difference = (np.asarray(first)-np.asarray(second)+np.pi) % (2*np.pi)-np.pi
    return bool(np.max(np.abs(difference)) < BRANCH_RULE['distinct_joint_tolerance_rad'])


class BranchResolver:
    """Legacy preservation cache plus an immutable two-slot branch pool."""
    def __init__(self, legacy):
        self.legacy, self.base = legacy, legacy.base
        self.active_slot = None
        self._pools, self._branches = {}, {}

    def __getattr__(self, name):
        return getattr(self.legacy, name)

    def solve(self, bucket_xy, values, hint):
        bucket, values = _vector(bucket_xy, 2, 'bucket'), _vector(values, 5, 'values')
        info = dict(passed=False, reason=None, candidates=[], pair_attempts=[],
                    branch_rule=copy.deepcopy(BRANCH_RULE))
        solver = self.base.solver
        within, reach = outer_geometry_mask(solver.scene, bucket[None], values[None, None], bounds=solver.bounds)
        info['analytic_hint'] = dict(old_within=bool(within[0, 0]), old_workspace=bool(reach[0, 0]), hard_veto=False)
        if (values < solver.low-1e-9).any() or (values > solver.high+1e-9).any():
            info['reason'] = 'action_bounds_rejected'; return info
        if np.linalg.norm(bucket-values[:2]) <= .15:
            info['reason'] = 'board_bucket_distance_constraint_rejected'; return info
        hint = source_geometry(hint)
        _, targets = yaw_target(self.scene, bucket, values)
        pools = []
        for arm, target, initial, known in (
            (solver.ik.dropper, targets['dropper_flange_world'], solver.initial['dropper_joint_rad'], hint['q10']),
            (solver.ik.board, targets['board_flange_local'], solver.initial['board_joint_rad'], hint['q14'])):
            pieces, provenance = [], []
            for seed in BRANCH_RULE['seeds']:
                result = arm.solve_candidates(target[None], initial, known=np.asarray(known)[None],
                    starts=BRANCH_RULE['starts_per_seed'], iterations=BRANCH_RULE['iterations'],
                    seed=seed, normal_only=False, preserve_known=np.asarray([False]))
                piece = {k:result[k][0].cpu().numpy() for k in ('q', 'valid', 'residual')}
                pieces.append(piece)
                provenance.extend(dict(seed=seed, start_index=i) for i in range(len(piece['q'])))
            pools.append(dict(**{k:np.concatenate([p[k] for p in pieces], axis=0) for k in ('q','valid','residual')}, provenance=provenance))
        first, second = pools
        info.update(targets={k:v.tolist() for k,v in targets.items()},
            starts=BRANCH_RULE['starts_per_seed'], seeds=list(BRANCH_RULE['seeds']),
            iterations=BRANCH_RULE['iterations'], normal_only=False,
            selection_uses_flight_labels=False, geometry_hint_sha256=digest(hint),
            valid_ik_branches=[int(first['valid'].sum()), int(second['valid'].sum())],
            ik_pools=[{k:(v.tolist() if isinstance(v,np.ndarray) else v) for k,v in pool.items()} for pool in pools])
        seen, bad10, bad14 = [], set(), set()
        for i,j in ordered_pairs(first, second, known10=np.asarray(hint['q10']), known14=np.asarray(hint['q14'])):
            if i in bad10 or j in bad14: continue
            q10, q14 = first['q'][i], second['q'][j]
            q = np.r_[q10,q14]
            if any(same_branch(q, previous) for previous in seen): continue
            if len(seen) >= BRANCH_RULE['maximum_unique_pairs']: break
            seen.append(q.copy())
            checked = solver.endpoint.check(q10, q14, bucket, values, targets)
            attempt = dict(branch10=int(i), branch14=int(j), checked=checked,
                seed_start10=first['provenance'][i], seed_start14=second['provenance'][j])
            info['pair_attempts'].append(attempt)
            if checked['reason']:
                for role in checked.get('independent_reject_arms', []):
                    (bad10 if role == 10 else bad14).add(i if role == 10 else j)
                continue
            errors = [native_pose_error(solver.gate, role, q, target) for role,target in (
                (10,targets['dropper_flange_world']), (14,base14(solver.scene) @ targets['board_flange_local']))]
            attempt['native_pose_errors'] = errors
            if any(e['position_m'] >= 1e-4 or .25*e['rotation_rad'] >= 1e-4 for e in errors):
                attempt['pose_rejected'] = True; continue
            info['candidates'].append(dict(bucket_xy_m=bucket.tolist(), values=values.tolist(),
                q10=q10.tolist(), q14=q14.tolist(), geometry_eligible=True,
                branch10=int(i), branch14=int(j), endpoint=checked))
            if len(info['candidates']) >= BRANCH_RULE['maximum_endpoint_candidates']: break
        info['passed'] = bool(info['candidates'])
        info['reason'] = None if info['passed'] else ('finite_ik_unresolved' if not first['valid'].any() or not second['valid'].any() else 'strict_endpoint_clearance_or_pose_rejected')
        return info

    def _pool(self, anchor, values, behavior, behavior_key):
        pool_key = digest(dict(version=BRANCH_VERSION, branch_rule=BRANCH_RULE,
            behavior_key=behavior_key, resolution_sha256=anchor['sha256']))
        if pool_key in self._pools:
            return copy.deepcopy(self._pools[pool_key]), True
        begin = time.monotonic()
        timing = dict(home_s=0., endpoint_s=0., path_s=0., wall_s=0.)
        pool = dict(version=BRANCH_VERSION, pool_key=pool_key, branch_rule=copy.deepcopy(BRANCH_RULE),
            behavior_key=behavior_key, release_behavior=behavior, bucket_xy_m=anchor['bucket_xy_m'],
            values=values, resolution_file=anchor['file'], resolution_sha256=anchor['sha256'],
            home_sha256=self.home['sha256'], endpoint_attempts=None, path_attempts=[],
            admitted=[], reason=None, timing=timing, selection_uses_flight_labels=False)
        if not anchor['home_passed']:
            pool['reason'] = 'fixed_base_clearance_rejected' if anchor.get('hard_reject') else 'canonical_home_unresolved_in_layout'
        else:
            hint = source_geometry(anchor['effective_source_hint'])
            if anchor.get('resolved'):
                hint.update({k:anchor['resolved'][k] for k in ('q10','q14','values','bucket_xy_m')})
            start = time.monotonic()
            endpoint = self.solve(anchor['bucket_xy_m'], values, hint)
            timing['endpoint_s'] = time.monotonic()-start
            pool['endpoint_attempts'], pool['reason'] = endpoint, endpoint['reason']
            for candidate_index,row in enumerate(endpoint['candidates'][:BRANCH_RULE['maximum_path_candidates']]):
                target = np.r_[row['q10'],row['q14']]
                start = time.monotonic()
                if anchor['passed']:
                    aq = np.r_[anchor['resolved']['q10'],anchor['resolved']['q14']]
                    if np.array_equal(aq,target):
                        proof = compose_anchor_prefix(anchor,self.home,target)
                    else:
                        suffix = self.planner.plan(target,anchor['bucket_xy_m'],start=aq)
                        if suffix['passed']:
                            proof = compose_anchor_prefix(anchor,self.home,target,suffix)
                        else:
                            failed = dict(passed=False,reason='finite_anchor_suffix_unresolved',
                                connection_route=dict(kind='anchor_suffix_rejected',prefix=anchor_prefix_identity(anchor),target_q=target.tolist(),suffix_proof=suffix))
                            suffix_index = len(pool['path_attempts'])
                            pool['path_attempts'].append(failed)
                            proof = copy.deepcopy(self.planner.plan(target,anchor['bucket_xy_m']))
                            proof['connection_route'] = dict(kind='full_home_after_anchor_suffix',prefix=anchor_prefix_identity(anchor),suffix_attempt_index=suffix_index)
                else:
                    proof = self.planner.plan(target,anchor['bucket_xy_m'])
                timing['path_s'] += time.monotonic()-start
                proof_index = len(pool['path_attempts'])
                pool['path_attempts'].append(proof)
                if proof['passed']:
                    pool['admitted'].append(dict(slot=len(pool['admitted']),candidate_index=candidate_index,
                        path_attempt_index=proof_index,q10=row['q10'],q14=row['q14'],endpoint=row['endpoint']))
                    if len(pool['admitted']) >= BRANCH_RULE['maximum_admitted_branches']: break
            if endpoint['passed'] and not pool['admitted']:
                pool['reason'] = 'finite_requested_setup_path_unresolved'
        timing['wall_s'] = time.monotonic()-begin
        pool.update(self.base._save('branch-pools',pool_key,pool))
        self._pools[pool_key] = copy.deepcopy(pool)
        return pool, False

    def connect(self, anchor, requested_values):
        if self.active_slot is None:
            return self.legacy.connect(anchor,requested_values)
        slot = self.active_slot
        if type(slot) is not int or slot not in (0,1): raise ValueError('Only branch slots 0 and 1 exist')
        verify_resolution(anchor,self.sources,self.home)
        self.base._check_resolution_bytes(anchor)
        values = _vector(requested_values,5,'requested values').tolist()
        bucket = _vector(anchor['bucket_xy_m'],2,'bucket').tolist()
        delta = np.asarray(bucket)-np.asarray(values[:2])
        valid = (np.linalg.norm(delta)>.15 and (np.asarray(values)>=self.solver.low-1e-9).all() and (np.asarray(values)<=self.solver.high+1e-9).all())
        yaw,targets = yaw_target(self.scene,bucket,values) if valid else (float(math.atan2(delta[1],delta[0])),None)
        behavior = dict(version=BEHAVIOR_VERSION,yaw_rad=yaw,rotation_matrix=(Rotation.from_euler('z',yaw).as_matrix() @ DOWNWARD).tolist(),
            target_dropper_flange_world=targets['dropper_flange_world'].tolist() if targets is not None else None,geometry_hint_only=True)
        behavior_key = digest(dict(version=BEHAVIOR_VERSION,connection_policy=CONNECTION_POLICY_VERSION,
            resolution_sha256=anchor['sha256'],values=values,bucket_xy_m=bucket,release_behavior=behavior))
        pool,pool_cache_hit = self._pool(anchor,values,behavior,behavior_key)
        key = digest(dict(version=BRANCH_VERSION,pool_key=pool['pool_key'],branch_pool_sha256=pool['sha256'],ik_branch_slot=slot))
        if key in self._branches: return dict(copy.deepcopy(self._branches[key]),cache_hit=True)
        endpoint = pool['endpoint_attempts']
        record = dict(version=VERSION,bucket_xy_m=bucket,values=values,resolution_file=anchor['file'],
            resolution_sha256=anchor['sha256'],home_sha256=self.home['sha256'],passed=False,
            geometry_eligible=False,endpoint_eligible=endpoint['passed'] if endpoint else None,
            reason=pool['reason'],q10=[0.]*6,q14=[0.]*6,proof=None,endpoint_attempts=copy.deepcopy(endpoint),
            path_attempts=copy.deepcopy(pool['path_attempts']),timing=copy.deepcopy(pool['timing']) if not pool_cache_hit else dict(home_s=0.,endpoint_s=0.,path_s=0.,wall_s=0.),
            action_substituted=False,connection_policy=CONNECTION_POLICY_VERSION,anchor_values=anchor['anchor_values'],
            anchor_derivation_sha256=digest(anchor['anchor_derivation']),release_behavior=behavior,behavior_key=behavior_key,
            branch_rule_version=BRANCH_VERSION,branch_key=key,ik_branch_slot=slot,branch_pool_file=pool['file'],
            branch_pool_sha256=pool['sha256'],branch_pool_key=pool['pool_key'],branch_pool_cache_hit=pool_cache_hit)
        if slot < len(pool['admitted']):
            selected = pool['admitted'][slot]
            record.update(passed=True,geometry_eligible=True,reason=None,q10=selected['q10'],q14=selected['q14'],
                proof=copy.deepcopy(pool['path_attempts'][selected['path_attempt_index']]),endpoint=selected['endpoint'])
        elif pool['admitted']:
            record['reason'] = 'requested_distinct_branch_unavailable'
        record.update(self.base._save('fixed-yaw-connections',key,record))
        verify_connection(record,anchor,self.home,values)
        self._branches[key] = copy.deepcopy(record)
        return dict(record,cache_hit=False)


class SixEngine(RecoveryEngine):
    """Original29 exact replay and new explicitly selected branch flights."""
    def __init__(self,ref,assets,directory,*,parent,sources,source_roots=None,**kwargs):
        FixedYawEngine.__init__(self,ref,assets,directory,**kwargs)
        self.parent_rows = {r['id']:copy.deepcopy(r) for r in sources}
        if len(self.parent_rows)!=len(sources): raise ValueError('Unique historical source rows required')
        roots = source_roots or {r['id']:Path(parent)/'native-yaw' for r in sources}
        if set(roots)!=set(self.parent_rows): raise ValueError('Every source needs one explicit evidence root')
        resolver = self.adapter.resolver
        self.source_receipts = []
        for source in sources:
            origin = Path(roots[source['id']])
            resolution = copy_bound(origin,directory,source['context_resolution_file'],source['context_resolution_sha256'])
            verify_resolution(resolution,resolver.sources,resolver.home)
            key = digest(dict(version=VERSION,bucket_xy_m=source['bucket_xy_m'],home_sha256=resolver.home['sha256'],
                sources=resolver.sources,bounds=resolver.bounds,anchor_search=anchor_search_spec()))
            if resolution['key']!=key or resolution['bucket_xy_m']!=source['bucket_xy_m']:
                raise ValueError('Parent geometry resolution key changed')
            resolver.base._resolved[key] = copy.deepcopy(resolution)
            connection = copy_bound(origin,directory,source['connection_file'],source['connection_sha256'])
            verify_connection(connection,resolution,resolver.home,source['values'])
            behavior = connection.get('release_behavior',{})
            key = digest(dict(version=BEHAVIOR_VERSION,connection_policy=CONNECTION_POLICY_VERSION,
                resolution_sha256=resolution['sha256'],values=source['values'],bucket_xy_m=source['bucket_xy_m'],release_behavior=behavior))
            if (behavior.get('version')!=BEHAVIOR_VERSION or connection.get('behavior_key')!=key
                    or Path(source['connection_file']).parent.as_posix()!='fixed-yaw-connections'
                    or Path(source['connection_file']).stem!=key):
                raise ValueError('Parent fixed-yaw behavior key changed')
            resolver._connected[key] = copy.deepcopy(connection)
            self.source_receipts.append(dict(source_id=source['id'],source_row_sha256=digest(source),
                resolution_file=source['context_resolution_file'],resolution_sha256=source['context_resolution_sha256'],
                connection_file=source['connection_file'],connection_sha256=source['connection_sha256']))
        self.parent_options = read(Path(parent)/'native-yaw/effective-options.json')
        self.parent_geometry = read(Path(parent)/'native-yaw/native-compiled-geometry.json')
        self.adapter.resolver = BranchResolver(resolver)

    def evaluate(self,requests,phase):
        ids = [r['id'] for r in requests]
        if len(ids)!=len(set(ids)): raise ValueError('Unique request IDs required')
        for request in requests:
            slot = request.get('ik_branch_slot')
            if request.get('exact_parent_action'):
                if slot is not None: raise ValueError('Exact preservation cannot select a new branch')
            elif type(slot) is not int or slot not in (0,1):
                raise ValueError('New actions and repeats require frozen branch slot 0 or 1')
        begin = time.monotonic(); rows = []; combined = timing_record()
        try:
            for slot in (None,0,1):
                group = [r for r in requests if r.get('ik_branch_slot')==slot]
                if not group: continue
                self.adapter.resolver.active_slot = slot
                measured,timing = super().evaluate(group,phase)
                for row in measured:
                    if slot is not None:
                        connection = read(self.directory/row['connection_file'])
                        if (connection.get('ik_branch_slot')!=slot or connection.get('branch_rule_version')!=BRANCH_VERSION
                                or connection.get('branch_key')!=Path(row['connection_file']).stem):
                            raise RuntimeError('Native row lost its requested branch proof')
                        row.update(branch_rule_version=BRANCH_VERSION,branch_pool_file=connection['branch_pool_file'],
                            branch_pool_sha256=connection['branch_pool_sha256'],branch_pool_key=connection['branch_pool_key'],
                            branch_key=connection['branch_key'])
                    rows.append(row)
                for key in combined:
                    if key!='scope': combined[key] += timing.get(key,0)
        finally:
            self.adapter.resolver.active_slot = None
        by_id = {r['id']:r for r in rows}
        if len(by_id)!=len(ids) or set(by_id)!=set(ids): raise RuntimeError('Native branch grouping changed request count')
        combined['wall_s'] = time.monotonic()-begin
        return [by_id[i] for i in ids],combined
