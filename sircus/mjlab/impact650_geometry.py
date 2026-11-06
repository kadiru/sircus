"""Nominal off-center drop rays with a frozen source board-arm branch.

The configured point lies on the board center plane. It is not an observed
finite-sphere contact point. Historical center-shot target/scoring code stays
unchanged; every new dropper endpoint needs its own native clearance check.
"""

import copy

import numpy as np

from .clean650_geometry import finite_vector
from .corrected_geometry import StrictEndpointGate
from .expanded_actions import WIDE_BOUNDS
from .learning_contract import digest
from .learning_ik import shot_targets
from .replay650_geometry import footprint
from .stable_ik import StableShotIK, target_digest


VERSION = "sircus-impact650-frozen-board-endpoints-v1"
BASIS_VERSION = "sircus-impact650-board-surface-forward-lateral-v1"
GAP_DEFINITION = "vertical_ball_bottom_clearance_above_offset_board_center_plane"
STARTS = 16
ITERATIONS = 180
MAX_BRANCHES = STARTS+1


def impact_targets(scene,bucket,values,offsets):
    """Return real release/flange targets for [forward,lateral] surface offsets.

    Forward is downhill along the board toward the bucket; lateral=normal×
    forward. Both distances are measured on the configured board center plane.
    The gap is vertical, so positive forward offsets also lower release Z.
    Zero offset reproduces the historical nominal release bit for bit.
    """
    bucket,values,offsets=(np.asarray(a,dtype=float) for a in (bucket,values,offsets))
    if values.ndim!=2:raise ValueError('Expected values[N,5]')
    n=len(values)
    if values.shape!=(n,5) or bucket.shape!=(n,2) or offsets.shape!=(n,2) or not n or not all(np.isfinite(a).all() for a in (bucket,values,offsets)):
        raise ValueError("Finite values[N,5],bucket[N,2],impact offsets[N,2] required")
    radius=float(scene['ball']['radius_m'])
    if not np.isfinite(radius) or radius<=0:raise ValueError('Finite positive ball radius required')
    targets=shot_targets(scene,bucket,values[:,:3],values[:,3],values[:,4])
    delta=bucket-values[:,:2];direction=delta/np.linalg.norm(delta,axis=1)[:,None]
    tilt=np.deg2rad(values[:,3]);c,s=np.cos(tilt),np.sin(tilt)
    forward=np.column_stack((c[:,None]*direction,-s))
    lateral=np.column_stack((-direction[:,1],direction[:,0],np.zeros(n)))
    point=values[:,:3]+offsets[:,0,None]*forward+offsets[:,1,None]*lateral
    release=point.copy();release[:,2]+=radius+values[:,4]
    flange=targets['dropper_flange_world'].copy();flange[:,:3,3]=release+[0,0,.095]
    targets.update(release_center_m=release,dropper_flange_world=flange,
        impact_point_world_m=point,impact_forward_world=forward,impact_lateral_world=lateral,
        impact_to_bucket_horizontal_m=np.linalg.norm(bucket-point[:,:2],axis=1),
        release_center_height_m=release[:,2].copy())
    return targets


class ImpactGeometry:
    """Solve only the dropper; retain each request's exact source board joints."""
    def __init__(self,scene,assets,*,ik=None,gate=None):
        self.scene,self.assets=copy.deepcopy(scene),assets
        self.ik=ik
        self.gate=gate if gate is not None else StrictEndpointGate(self.scene,assets)
        self.proofs=[];self.pools={}

    def _prepare(self,request):
        group=copy.deepcopy(request)
        action=finite_vector(group['values'],5,'action')
        bucket=finite_vector(group['bucket_xy_m'],2,'bucket XY')
        offset=finite_vector(group['impact_offset_m'],2,'impact offset')
        q10,q14=(finite_vector(group[k],6,'source '+k) for k in ('q10','q14'))
        targets={k:v[0] for k,v in impact_targets(self.scene,bucket[None],action[None],offset[None]).items()}
        group.update(values=action.tolist(),bucket_xy_m=bucket.tolist(),impact_offset_m=offset.tolist(),
            impact_point_world_m=targets['impact_point_world_m'].tolist(),
            release_center_m=targets['release_center_m'].tolist(),
            impact_to_bucket_horizontal_m=float(targets['impact_to_bucket_horizontal_m']),
            release_center_height_m=float(targets['release_center_height_m']),
            impact_basis_version=BASIS_VERSION,impact_gap_definition=GAP_DEFINITION)
        within=bool((action>=WIDE_BOUNDS['low']).all() and (action<=WIDE_BOUNDS['high']).all())
        table=footprint(self.scene,bucket);distance=float(np.linalg.norm(bucket-action[:2]))
        local=targets['board_world'][:3,:3].T@(targets['impact_point_world_m']-action[:3])
        margin=np.asarray(self.scene['board']['size_m'][:2])/2-np.abs(local[:2])
        point_inside=bool((margin>=.005-1e-12).all() and abs(local[2])<1e-9)
        zero=bool((offset==0).all());control=group['role'].endswith('_control')
        frozen=control or zero or group.get('freeze_joints') is True
        mode='exact_control' if control else 'exact_zero_offset' if zero else 'frozen_verification' if frozen else 'solved_dropper_cpu'
        reason=('action_bounds_or_invalid' if not within else 'bucket_table_footprint' if not table['passed']
                else 'board_bucket_distance' if distance<=.15 else 'impact_point_outside_board' if not point_inside
                else 'control_impact_offset' if control and not zero else None)
        proof=dict(version=VERSION,group_id=group['id'],values=action.tolist(),bucket_xy_m=bucket.tolist(),impact_offset_m=offset.tolist(),
            impact_point_world_m=group['impact_point_world_m'],impact_to_bucket_horizontal_m=group['impact_to_bucket_horizontal_m'],
            release_center_m=group['release_center_m'],
            release_center_height_m=group['release_center_height_m'],impact_basis_version=BASIS_VERSION,impact_gap_definition=GAP_DEFINITION,
            source_mode=mode,board_q14_frozen=True,source_q14=q14.tolist(),bounds_passed=within,bucket_footprint=table,
            board_point_footprint_passed=point_inside,board_point_edge_margin_m=float(margin.min()),distance_m=distance,
            ik_starts=STARTS,ik_iterations=ITERATIONS,pair_budget=MAX_BRANCHES,attempts=[],held_endpoints_only=True)
        return dict(group=group,action=action,bucket=bucket,offset=offset,q10=q10,q14=q14,targets=targets,
                    frozen=frozen,mode=mode,rejection=reason,proof=proof)

    def _pools(self,prepared):
        missing={}
        for p in prepared:
            if p['rejection'] is not None or p['frozen']:continue
            key=digest(dict(target=target_digest(p['targets']['dropper_flange_world'],role='dropper'),hint=p['q10'].tolist()))
            p['pool_key']=key
            if key not in self.pools:missing.setdefault(key,p)
        if not missing:return
        if self.ik is None:self.ik=StableShotIK(self.scene,self.assets,device='cpu').dropper
        keys=sorted(missing)
        for start in range(0,len(keys),128):
            selected=keys[start:start+128]
            targets=np.stack([missing[k]['targets']['dropper_flange_world'] for k in selected])
            hints=np.stack([missing[k]['q10'] for k in selected])
            pool=self.ik.solve_candidates(targets,initial=hints,known=hints,starts=STARTS,iterations=ITERATIONS,
                preserve_known=np.zeros(len(selected),dtype=bool))
            joints,valid=pool['q'].cpu().numpy(),pool['valid'].cpu().numpy()
            if joints.ndim!=3 or joints.shape[0]!=len(selected) or joints.shape[2]!=6 or not 1<=joints.shape[1]<=MAX_BRANCHES or valid.shape!=joints.shape[:2]:
                raise ValueError("Unexpected bounded dropper IK pool")
            for i,key in enumerate(selected):
                q=np.asarray(joints[i],float).copy();good=np.asarray(valid[i],bool)&np.isfinite(q).all(-1)
                q.setflags(write=False);good.setflags(write=False);self.pools[key]=dict(q=q,valid=good)

    def _check(self,p):
        group,proof,reason=p['group'],p['proof'],p['rejection'];chosen=None
        if reason is not None:pairs=[]
        elif p['frozen']:pairs=[(None,p['q10'])]
        else:
            pool=self.pools[p['pool_key']];proof['ik_query_sha256']=p['pool_key'];proof['valid_dropper_branches']=int(pool['valid'].sum())
            indices=sorted(np.flatnonzero(pool['valid']),key=lambda i:(not np.array_equal(pool['q'][i],p['q10']),int(i)))
            pairs=[(int(i),pool['q'][i]) for i in indices]
            if not pairs:reason='ik_unresolved'
        for branch,q10 in pairs:
            q10=q10.copy();q14=p['q14'].copy()
            checked=self.gate.check(q10,q14,p['bucket'],p['action'],p['targets'])
            proof['attempts'].append(dict(branch10=branch,q10=q10.tolist(),q14=q14.tolist(),checked=checked))
            if checked['reason']==0:
                clearance=float(checked.get('minimum_clearance_lower_bound_m',-1.))
                if not np.isfinite(clearance) or clearance<.005-1e-12:raise RuntimeError('Successful impact endpoint lacks 5 mm clearance proof')
                chosen=q10.tolist();proof.update(minimum_clearance_lower_bound_m=clearance,selected_branch10=branch);break
            if 14 in checked.get('independent_reject_arms',[]):break
        if chosen is None and reason is None:reason='strict_endpoint_rejected'
        proof.update(eligible=chosen is not None,reason=reason if chosen is None else None)
        group.update(eligible=proof['eligible'],q10=p['q10'].tolist() if chosen is None else chosen,q14=p['q14'].tolist(),
                     geometry_row=len(self.proofs),geometry_sha256=digest(proof))
        self.proofs.append(proof);return group

    def check_many(self,requests):
        prepared=[self._prepare(r) for r in requests];self._pools(prepared)
        return [self._check(p) for p in prepared]
