"""Versioned strict endpoint admission for corrected-contact grid studies.

Reuses deterministic source-aware IK and the independent 5mm geometry model.
Endpoint eligibility is separate from flight outcome and selected-shot transit.
"""
from collections import Counter
import copy
import json
from pathlib import Path
import time

import numpy as np

from .bank_geometry import _bucket_array
from .expand_geometry import WIDE_BOUNDS, outer_geometry_mask
from .stable_geometry import known_array, known_request_match, ordered_pairs
from .stable_ik import StableShotIK, VERSION as IK_VERSION

VERSION="sircus-corrected-strict-endpoint-admission-v1"
REASONS={0:"endpoint_admitted",1:"action_bounds_or_invalid",2:"workspace_prior_reject",
    3:"ik_unresolved",4:"historical_native_collision",5:"native_pose_mismatch",
    6:"bucket_collider_transform",7:"strict_clearance_reject",8:"strict_clearance_unproven"}


def independent_rejections(pairs):
    """A same-arm/environment failure cannot be fixed by changing the other arm."""
    rejected=set()
    for pair in pairs:
        roles=set()
        for name in pair:
            if name.startswith(("Arm10/","ball/")):roles.add(10)
            elif name.startswith("Arm14/"):roles.add(14)
        if len(roles)==1:rejected.update(roles)
    return rejected


class StrictEndpointGate:
    def __init__(self,scene,assets,clearance_m=.005):
        from .transit_audit import GeometricGate
        self.strict=GeometricGate(scene,assets,scene["bucket"]["center_xy_m"],clearance_m)
        self.native=self.strict.native
        self.clearance_m=clearance_m

    def check(self,q10,q14,bucket_xy,values,targets):
        result=self.native.check(q10,q14,bucket_xy,values,targets)
        result["independent_reject_arms"]=[]
        if result["reason"]:
            result["independent_reject_arms"]=sorted(independent_rejections(result["pairs"]))
            return result
        self.strict.bucket_xy=np.asarray(bucket_xy,float)
        checked=self.strict.check(np.r_[q10,q14])
        result["strict_geometry"]=checked
        result["minimum_clearance_lower_bound_m"]=checked.get("minimum_clearance_m",-1.)
        if not checked["passed"]:
            result["reason"]=8 if checked["reason"]=="unproven_clearance" else 7
            pairs=[r["pair"] for r in checked.get("margin_contacts",[])]+[r["pair"] for r in checked.get("unresolved_clearance_pairs",[])]
            result["pairs"]=pairs
            result["independent_reject_arms"]=sorted(independent_rejections(pairs))
        return result


class CorrectedGeometry:
    """Reusable CPU model/IK instance; requests have stable independent identities."""
    def __init__(self,scene,assets,*,starts=16,iterations=180,ik=None,gate=None,initials=None):
        if starts!=16 or iterations!=180:
            raise ValueError("This admission version pins16starts/180iterations")
        self.scene,self.assets,self.starts,self.iterations=copy.deepcopy(scene),Path(assets),starts,iterations
        self.ik=ik if ik is not None else StableShotIK(self.scene,self.assets,device="cpu")
        self.gate=gate if gate is not None else StrictEndpointGate(self.scene,self.assets)
        self.initials=initials if initials is not None else json.loads((self.assets/"poses.json").read_text())

    def audit_values(self,bucket_xy,values,*,bounds=None,known_q10=None,known_q14=None,
        known_values=None,known_bucket_xy=None,chunk_size=128,progress=None):
        begin=time.monotonic();xy=_bucket_array(bucket_xy);values=np.asarray(values,float)
        bounds=WIDE_BOUNDS if bounds is None else bounds
        within,reach=outer_geometry_mask(self.scene,xy,values,bounds=bounds)
        if not 1<=chunk_size<=4096:raise ValueError("Chunk1..4096 required")
        shape=values.shape[:2];flat=values.reshape(-1,5)
        bucket=np.broadcast_to(xy[:,None],(*shape,2)).reshape(-1,2)
        hints10=known_array(known_q10,shape).reshape(-1,6);hints14=known_array(known_q14,shape).reshape(-1,6)
        nominal=known_request_match(xy,values,known_values,known_bucket_xy)
        reason=np.ones(shape,np.uint8);reason[within]=2;reason[within&reach]=3
        q10,q14=np.zeros((*shape,6)),np.zeros((*shape,6))
        branch10,branch14=np.full(shape,-1,np.int16),np.full(shape,-1,np.int16)
        preserved=np.zeros(shape,bool);refined=np.zeros(shape,bool)
        residual10,residual14=np.full(shape,-1.),np.full(shape,-1.)
        clearance=np.full(shape,-1.);pair_count=np.zeros(shape,np.int16);pruned=np.zeros(shape,np.int16)
        query10,query14=np.full(shape,"",dtype="<U64"),np.full(shape,"",dtype="<U64")
        diagnostics=[];selected=np.flatnonzero((within&reach).reshape(-1))
        for offset in range(0,len(selected),chunk_size):
            ids=selected[offset:offset+chunk_size];action=flat[ids]
            pool=self.ik.candidates(bucket[ids],action[:,:3],action[:,3],action[:,4],
                initial10=self.initials["dropper_joint_rad"],initial14=self.initials["board_joint_rad"],
                known_q10=hints10[ids],known_q14=hints14[ids],starts=self.starts,iterations=self.iterations,
                preserve_known=nominal.reshape(-1)[ids])
            converted=[]
            for role in ("dropper","board"):
                converted.append({k:pool[role][k].cpu().numpy() for k in ("q","valid","residual")})
            for j,index in enumerate(ids):
                first,second=({k:v[j] for k,v in p.items()} for p in converted)
                query10.reshape(-1)[index]=pool["dropper"]["query_sha256"][j]
                query14.reshape(-1)[index]=pool["board"]["query_sha256"][j]
                targets={k:v[j] for k,v in pool["targets"].items()}
                counts=Counter();first_failure=None;bad10,bad14=set(),set()
                for i10,i14 in ordered_pairs(first,second,known10=hints10[index],known14=hints14[index]):
                    if i10 in bad10 or i14 in bad14:
                        pruned.reshape(-1)[index]+=1;continue
                    checked=self.gate.check(first["q"][i10],second["q"][i14],bucket[index],flat[index],targets)
                    pair_count.reshape(-1)[index]+=1;counts[checked["reason"]]+=1
                    if checked["reason"]:
                        if first_failure is None:first_failure=checked
                        for role in checked.get("independent_reject_arms",[]):
                            (bad10 if role==10 else bad14).add(i10 if role==10 else i14)
                        continue
                    reason.reshape(-1)[index]=0
                    q10.reshape(-1,6)[index]=first["q"][i10];q14.reshape(-1,6)[index]=second["q"][i14]
                    branch10.reshape(-1)[index]=i10;branch14.reshape(-1)[index]=i14
                    residual10.reshape(-1)[index]=first["residual"][i10];residual14.reshape(-1)[index]=second["residual"][i14]
                    clearance.reshape(-1)[index]=checked.get("minimum_clearance_lower_bound_m",-1.)
                    same=np.array_equal(first["q"][i10],hints10[index]) and np.array_equal(second["q"][i14],hints14[index])
                    preserved.reshape(-1)[index]=same
                    refined.reshape(-1)[index]=i10==0 and i14==0 and np.isfinite(hints10[index]).all() and np.isfinite(hints14[index]).all() and not same
                    break
                if first_failure is not None:
                    if reason.reshape(-1)[index]!=0:reason.reshape(-1)[index]=first_failure["reason"]
                    diagnostics.append(dict(layout_index=int(index//shape[1]),candidate_index=int(index%shape[1]),
                        rejected_pair_reasons={str(k):int(v) for k,v in counts.items() if k},
                        first_rejection_pairs=first_failure.get("pairs",[]),
                        first_strict_rejection=first_failure.get("strict_geometry")))
            if progress:progress(dict(processed=min(offset+chunk_size,len(selected)),eligible=len(selected),endpoint_admitted=int((reason==0).sum())))
        mask=reason==0
        summary=dict(version=VERSION,ik_version=IK_VERSION,starts=self.starts,iterations=self.iterations,
            layouts=shape[0],candidates_per_layout=shape[1],total_requests=int(np.prod(shape)),
            outer_prior_eligible=len(selected),endpoint_admitted=int(mask.sum()),
            native_pairs_tested=int(pair_count.sum()),independent_branch_pairs_pruned=int(pruned.sum()),
            reason_counts={REASONS[int(k)]:int(v) for k,v in Counter(reason.reshape(-1)).items()},
            elapsed_s=time.monotonic()-begin,scope="Strict endpoint eligibility only; no ball flight or selected-shot transit guarantee")
        return mask,dict(version=VERSION,summary=summary,reasons=reason,q10=q10,q14=q14,branch10=branch10,branch14=branch14,
            known_request_match=nominal,known_pair_preserved=preserved,preserved_exact_nominal=preserved&nominal,
            refined_continuation=refined,residual10=residual10,residual14=residual14,
            native_pairs_tested=pair_count,independent_branch_pairs_pruned=pruned,
            minimum_clearance_lower_bound_m=clearance,query10_sha256=query10,query14_sha256=query14,diagnostics=diagnostics)


def audit_values(scene,assets,bucket_xy,values,**kwargs):
    """Convenience adapter; use CorrectedGeometry to reuse compiled models."""
    return CorrectedGeometry(scene,assets).audit_values(bucket_xy,values,**kwargs)
