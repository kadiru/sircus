"""Strict CPU rechecks of frozen branches and board-fixed local gap variants."""
import copy

import numpy as np

from .corrected_geometry import StrictEndpointGate
from .expanded_actions import WIDE_BOUNDS
from .learning_contract import digest
from .learning_ik import shot_targets


def footprint(scene,bucket):
    center=np.asarray(scene['table']['center_m'][:2]);size=np.asarray(scene['table']['size_m'][:2]);point=np.asarray(bucket)
    radius=max(scene['bucket']['outer_top_diameter_m'],scene['bucket']['outer_bottom_diameter_m'])/2
    edges=np.r_[point-(center-size/2)-radius,(center+size/2)-point-radius]
    return dict(outer_radius_m=radius,edge_clearance_m=edges.tolist(),minimum_edge_clearance_m=float(min(edges)),
        passed=bool(min(edges)>=.005-1e-12))


class ReplayGeometry:
    def __init__(self,scene,assets):
        self.scene,self.assets=scene,assets;self.gate=StrictEndpointGate(scene,assets)
        self.proofs=[];self.ik=None

    def check(self,source,*,candidate=None):
        group=copy.deepcopy(source);v=np.asarray(group['values'],float);xy=np.asarray(group['bucket_xy_m'],float)
        target={k:val[0] for k,val in shot_targets(self.scene,xy[None],v[None,:3],[v[3]],[v[4]]).items()}
        fp=footprint(self.scene,xy);bounds=bool((v>=WIDE_BOUNDS['low']).all() and (v<=WIDE_BOUNDS['high']).all())
        proof=dict(group_id=group['id'],values=v.tolist(),bucket_xy_m=xy.tolist(),bucket_footprint=fp,bounds_passed=bounds,
            preserved_source=candidate is None,held_endpoints_only=True,attempts=[])
        pools=[(np.asarray(source['q10']),np.asarray(source['q14']))]
        if candidate is not None:
            from .stable_ik import StableShotIK
            if self.ik is None:self.ik=StableShotIK(self.scene,self.assets,device='cpu')
            if v[4]!=candidate['values'][4]:
                pool=self.ik.candidates(xy[None],v[None,:3],[v[3]],[v[4]],
                    known_q10=np.asarray(candidate['q10'])[None],known_q14=np.asarray(candidate['q14'])[None],
                    initial10=np.asarray(candidate['q10']),initial14=np.asarray(candidate['q14']),starts=16,iterations=180,preserve_known=[False])
                q=pool['dropper']['q'][0].cpu().numpy();valid=pool['dropper']['valid'][0].cpu().numpy()
                pools=[(row,np.asarray(candidate['q14'])) for row,ok in zip(q,valid) if ok
                    and np.max(np.abs(row-candidate['q10']))<=.10]
            else:pools=[(np.asarray(candidate['q10']),np.asarray(candidate['q14']))]
        chosen=None
        if fp['passed'] and bounds and np.linalg.norm(xy-v[:2])>.15:
            for q10,q14 in pools:
                checked=self.gate.check(q10,q14,xy,v,target)
                proof['attempts'].append(dict(q10=q10.tolist(),q14=q14.tolist(),checked=checked))
                if checked['reason']==0:
                    chosen=(q10.tolist(),q14.tolist());proof['minimum_clearance_lower_bound_m']=checked['minimum_clearance_lower_bound_m'];break
        proof['eligible']=chosen is not None;proof['reason']=None if chosen else 'strict_endpoint_or_local_branch_rejected'
        group.update(eligible=proof['eligible'],geometry_row=len(self.proofs),geometry_sha256=digest(proof))
        if chosen:group.update(q10=chosen[0],q14=chosen[1])
        # Rejected requests retain their original proposed source branch as data;
        # they are never executed or counted as admitted geometry.
        self.proofs.append(proof)
        return group
