"""Common-staging admission for selected simulated shots; CPU geometry only.

An own-endpoint round trip is not common-home connectivity. This module freezes
one exact global joint pair, checks it in every bucket layout, and certifies a
sampled serialized route to the exact selected shot branches and back.
"""
import copy
from pathlib import Path
import time

import numpy as np
from scipy.spatial.transform import Rotation

from .corrected_geometry import StrictEndpointGate
from .expand_geometry import WIDE_BOUNDS, outer_geometry_mask
from .learning_contract import digest
from .learning_ik import shot_targets
from .stable_geometry import ordered_pairs
from .stable_ik import StableShotIK
from .transit_audit import audit_segment, pose_samples, VERSION as SWEEP_VERSION

VERSION="sircus-fixed-common-home-transit-admission-v1"
HOME_VERSION="sircus-fixed-common-home-v1"
PROOF_SCOPE="fixed_common_home_to_exact_selected_shot_and_reverse_sampled_geometry"
REFERENCE_BUCKET=[.16275255472120964,-.6365356466995716]
REFERENCE_VALUES=[.4708757793012148,-.2778934014509746,.3973878458850598,36.154791124080255,.3032930420811218]
LOW_HEIGHTS=(.30,.35,.40)
APPROACH_GAPS=(.14,.08,.04,.02,0.)
HOME_BOARD_CENTERS=((.95,.15,.30),(.67,-.20,.34),(.67,-.22,.30),(.62,-.24,.30),(.72,-.20,.40),tuple(REFERENCE_VALUES[:3]))
REFERENCE_Q14=[3.380062669046468,1.5039574753021427,-3.1992193103493975,-.4047718442151326,-1.0624829296587495,2.301608077910527]
VERIFIED_PARK_Q10=[-.38051489244682624,.1917165050458849,-1.7468899918092256,9.64150028083706e-7,-1.9386064977117545,-.3805191251073095]


def base14(scene):
    out=np.eye(4);out[:3,:3]=Rotation.from_euler("z",scene["arms"]["board_base_yaw_rad"]).as_matrix()
    out[0,3]=scene["arms"]["base_spacing_m"]
    return out


def board_flange_world(scene,world_board):
    mount=np.eye(4);mount[:3,:3]=Rotation.from_euler("z",scene["board"]["mount_yaw_rad"]).as_matrix()
    mount[2,3]=scene["board"]["tool_offset_m"]
    return world_board@np.linalg.inv(mount)


def body_transform(gate,role,q):
    gate.state(q);body=gate.m.body(f"Arm{role}/link6").id
    out=np.eye(4);out[:3,:3]=gate.d.xmat[body].reshape(3,3);out[:3,3]=gate.d.xpos[body]
    return out


def native_pose_error(gate,role,q,target):
    actual=body_transform(gate,role,q)
    return dict(position_m=float(np.linalg.norm(actual[:3,3]-target[:3,3])),
        rotation_rad=float(Rotation.from_matrix(actual[:3,:3]@target[:3,:3].T).magnitude()))


def freeze_common_home(scene,assets,*,ik=None,gate=None):
    """Generate once, independent of candidates. Every request rechecks its world."""
    ik=ik if ik is not None else StableShotIK(scene,assets,device="cpu")
    gate=gate if gate is not None else StrictEndpointGate(scene,assets).strict
    # Generation must not inherit the last queried candidate's mutable bucket.
    gate.bucket_xy=np.asarray(scene["bucket"]["center_xy_m"],float).copy()
    ref=shot_targets(scene,np.asarray([REFERENCE_BUCKET]),np.asarray([REFERENCE_VALUES[:3]]),
        [REFERENCE_VALUES[3]],[REFERENCE_VALUES[4]])
    target10=np.eye(4);target10[:3,:3]=np.diag([1.,-1.,-1.]);target10[:3,3]=[.25,-.10,.95]
    first=ik.dropper.solve_candidates(target10[None],known=VERIFIED_PARK_Q10,preserve_known=np.asarray([True]),starts=16,iterations=180)
    first={k:first[k][0].cpu().numpy() for k in ("q","valid","residual")}
    boards=[];targets14=[]
    for center in HOME_BOARD_CENTERS:
        board=ref["board_world"][0].copy();board[:3,3]=center
        boards.append(board);targets14.append(board_flange_world(scene,board))
    second=ik.board.solve_candidates(np.asarray([np.linalg.inv(base14(scene))@t for t in targets14]),known=REFERENCE_Q14,
        preserve_known=np.asarray([tuple(c)==tuple(REFERENCE_VALUES[:3]) for c in HOME_BOARD_CENTERS]),starts=16,iterations=180)
    second={k:second[k].cpu().numpy() for k in ("q","valid","residual")}
    checked=[]
    for family_index,(board,target14) in enumerate(zip(boards,targets14)):
        local={k:v[family_index] for k,v in second.items()}
        for i,j in ordered_pairs(first,local):
            q=np.r_[first["q"][i],local["q"][j]]
            result=gate.check(q)
            if not result["passed"]:
                checked.append(dict(family_index=family_index,branch10=i,branch14=j,reason=result["reason"]));continue
            errors=[native_pose_error(gate,role,q,target) for role,target in ((10,target10),(14,target14))]
            if any(e["position_m"]>=1e-4 or .25*e["rotation_rad"]>=1e-4 for e in errors):
                checked.append(dict(family_index=family_index,branch10=i,branch14=j,reason="native_home_pose_mismatch"));continue
            record=dict(version=HOME_VERSION,passed=True,scene_sha256=digest(scene),
                q10=q[:6].tolist(),q14=q[6:].tolist(),branch10=int(i),branch14=int(j),
                generation_family=[list(c) for c in HOME_BOARD_CENTERS],selected_family_index=family_index,
                rejected_home_pairs=checked,
                valid_board_branches_by_family=second["valid"].sum(1).astype(int).tolist(),
                dropper_flange_world=target10.tolist(),board_world=board.tolist(),board_flange_world=target14.tolist(),
                reference_orientation_source=dict(bucket_xy_m=REFERENCE_BUCKET,values=REFERENCE_VALUES),
                reference_joint_hints=dict(q10=VERIFIED_PARK_Q10,q14=REFERENCE_Q14,
                    provenance="Synthetic stable-shot-0004711 board and previously audited high-park path; rechecked here"),
                generation_bucket_xy_m=gate.bucket_xy.tolist(),initial_geometry=result,
                validation_model_sha256=gate.validation_model_sha256,
                scope="One frozen global joint pair selected once from declared finite family; individually collision-checked in every layout")
            record["sha256"]=digest(record);return record
    result=dict(version=HOME_VERSION,passed=False,scene_sha256=digest(scene),reason="finite_common_home_ik_or_clearance_unresolved",
        generation_family=[list(c) for c in HOME_BOARD_CENTERS],valid_branches10=int(first["valid"].sum()),
        valid_branches14_by_family=second["valid"].sum(1).astype(int).tolist(),rejected_pairs=checked)
    result["sha256"]=digest(result);return result


def validate_home_record(home,scene):
    payload=dict(home);fingerprint=payload.pop("sha256",None)
    if home.get("version")!=HOME_VERSION or digest(payload)!=fingerprint or home.get("scene_sha256")!=digest(scene):
        raise ValueError("Common-home identity or scene changed")
    if home.get("passed"):
        q=np.r_[home["q10"],home["q14"]]
        if q.shape!=(12,) or not np.isfinite(q).all():raise ValueError("Common home requires finite exact12joints")
    return copy.deepcopy(home)


class TransitAdmission:
    def __init__(self,scene,assets,*,home=None,ik=None,gate=None):
        self.scene,self.assets=copy.deepcopy(scene),Path(assets)
        self.ik=ik if ik is not None else StableShotIK(self.scene,assets,device="cpu")
        self.endpoint=gate if gate is not None else StrictEndpointGate(self.scene,assets)
        self.gate=self.endpoint.strict
        self._home=validate_home_record(home,self.scene) if home is not None else freeze_common_home(self.scene,assets,ik=self.ik,gate=self.gate)
        self.cache={}

    @property
    def home(self):
        return copy.deepcopy(self._home)

    def check_home(self,bucket_xy):
        if not self.home["passed"]:return dict(passed=False,reason="common_home_unresolved")
        self.gate.bucket_xy=np.asarray(bucket_xy,float)
        q=np.r_[self.home["q10"],self.home["q14"]]
        result=self.gate.check(q)
        result["common_home_sha256"]=self.home["sha256"]
        return result

    def _move(self,q,role,target,phase,corridor,*,initial_target=None):
        arm=self.ik.dropper if role==10 else self.ik.board;offset=0 if role==10 else 6
        current_world=body_transform(self.gate,role,q)
        result=dict(name=phase,moving_arm=role,corridor=corridor,target_flange_world=target.tolist(),passed=False,segments=[])
        frames=[]
        for world in pose_samples(current_world,target):
            local=world if role==10 else np.linalg.inv(base14(self.scene))@world
            pool=arm.solve_candidates(local[None],known=q[offset:offset+6],initial=initial_target,starts=16,iterations=180)
            candidates=pool["q"][0].cpu().numpy();valid=pool["valid"][0].cpu().numpy()
            accepted=None;continuous=0;rejects=[]
            for branch in sorted(np.flatnonzero(valid),key=lambda i:np.linalg.norm(candidates[i]-q[offset:offset+6])):
                proposed=q.copy();proposed[offset:offset+6]=candidates[branch]
                if np.abs(proposed-q).max()>.5:continue
                continuous+=1;checked=audit_segment(self.gate,q,proposed,corridor=corridor)
                if not checked["passed"]:rejects.append(checked);continue
                error=native_pose_error(self.gate,role,proposed,world)
                if error["position_m"]>=1e-4 or .25*error["rotation_rad"]>=1e-4:
                    rejects.append(dict(passed=False,reason="native_pose_mismatch",error=error));continue
                accepted=proposed,checked,int(branch);break
            if accepted is None:
                result.update(reason="transit_ik_unresolved" if not valid.any() else ("transit_branch_discontinuity" if not continuous else "transit_sampled_clearance_or_corridor_reject"),
                    valid_ik_branches=int(valid.sum()),continuous_branches=continuous,rejections=rejects)
                return q,result,frames
            proposed,checked,branch=accepted
            result["segments"].append(dict(**checked,branch=branch,q10=proposed[:6].tolist(),q14=proposed[6:].tolist()))
            q=proposed;frames.append(self.gate.frame(q,phase))
        result["passed"]=True
        return q,result,frames

    def _admit(self,row):
        bucket=np.asarray(row["bucket_xy_m"],float);values=np.asarray(row["values"],float)
        first,second=np.asarray(row["q10"],float),np.asarray(row["q14"],float)
        if bucket.shape!=(2,) or values.shape!=(5,) or first.shape!=(6,) or second.shape!=(6,) or not all(np.isfinite(a).all() for a in (bucket,values,first,second)):
            raise ValueError("Finite bucket2/action5/q10six/q14six required")
        selected=np.r_[first,second]
        identity=dict(version=VERSION,scene_sha256=digest(self.scene),common_home_sha256=self.home["sha256"],
            bucket_xy_m=bucket.tolist(),values=values.tolist(),q10=selected[:6].tolist(),q14=selected[6:].tolist())
        fingerprint=digest(identity)
        if fingerprint in self.cache:return copy.deepcopy(self.cache[fingerprint])
        result=dict(version=VERSION,proof_scope=PROOF_SCOPE,scope=PROOF_SCOPE,passed=False,request_sha256=fingerprint,
            request=identity,common_home_sha256=self.home["sha256"],attempts=[],frames=[],
            protocol=dict(low_board_center_m=list(LOW_HEIGHTS),dropper_approach_gap_m=list(APPROACH_GAPS),
                ik_starts=16,ik_iterations=180,maximum_joint_sample_step_rad=float(np.deg2rad(.5)),
                minimum_modeled_clearance_m=.005,max_branch_step_rad=.5))
        within,reach=outer_geometry_mask(self.scene,bucket[None],values[None,None],bounds=WIDE_BOUNDS)
        if not bool((within&reach)[0,0]):
            result["reason"]="selected_request_bounds_or_workspace_rejected";return result
        begin=time.monotonic();targets=shot_targets(self.scene,bucket[None],values[None,:3],[values[3]],[values[4]])
        target={k:v[0] for k,v in targets.items()}
        endpoint=self.endpoint.check(selected[:6],selected[6:],bucket,values,target)
        result["endpoint"]=endpoint
        if endpoint["reason"]:
            result["reason"]="selected_endpoint_not_admitted";return result
        result["home_geometry"]=self.check_home(bucket)
        if not result["home_geometry"]["passed"]:
            result["reason"]="common_home_not_admitted_in_layout";return result
        homeq=np.r_[self.home["q10"],self.home["q14"]]
        board_target=board_flange_world(self.scene,target["board_world"])
        drop_target=target["dropper_flange_world"]
        for low_height in LOW_HEIGHTS:
            for approach in APPROACH_GAPS:
                q=homeq.copy();frames=[self.gate.frame(q,"common_home")]
                low_board=target["board_world"].copy();low_board[2,3]=low_height
                board_low=board_flange_world(self.scene,low_board)
                raised=drop_target.copy();raised[2,3]+=approach
                board_cap=max(float(target["board_world"][2,3]),low_height)+.17
                attempt=dict(low_board_center_m=low_height,dropper_approach_gap_m=approach,passed=False,phases=[])
                result["attempts"].append(attempt)
                phases=[("board_low_transfer_orient",14,board_low,dict(dropper_flange_min_m=.94,board_top_max_m=low_height+.17)),
                    ("board_set_shot_height",14,board_target,dict(dropper_flange_min_m=.94,board_top_max_m=board_cap)),
                    ("dropper_high_transfer",10,raised,dict(dropper_flange_min_m=float(drop_target[2,3])-.01)),
                    ("dropper_set_release",10,drop_target,dict(dropper_flange_min_m=float(drop_target[2,3])-.01))]
                failed=False
                for label,role,world,corridor in phases:
                    q,checked,newframes=self._move(q,role,world,label,corridor,
                        initial_target=selected[:6] if role==10 else selected[6:])
                    attempt["phases"].append(checked);frames.extend(newframes)
                    if not checked["passed"]:failed=True;break
                    if label=="board_set_shot_height":
                        exact=q.copy();exact[6:]=selected[6:]
                        checked=audit_segment(self.gate,q,exact,corridor=corridor)
                        attempt["exact_board_branch"]=checked
                        if not checked["passed"]:failed=True;break
                        q=exact;frames.append(self.gate.frame(q,"exact_board_branch"))
                if failed:continue
                exact=audit_segment(self.gate,q,selected)
                attempt["exact_selected_branch"]=exact
                if not exact["passed"]:continue
                q=selected.copy();frames.append(self.gate.frame(q,"exact_selected_shot"))
                # Reverse samples represent exactly the same static geometry.
                # Audit independently in reverse order, including exact branch joins.
                reverse=[selected.copy()]+[np.r_[f["q10"],f["q14"]] for f in frames[-2::-1]]
                reverse_checks=[audit_segment(self.gate,a,b) for a,b in zip(reverse,reverse[1:])]
                if not all(c["passed"] for c in reverse_checks):
                    attempt["reverse_rejected"]=True;continue
                attempt.update(passed=True,reverse_checks=reverse_checks)
                count=sum(s["total_samples"] for p in attempt["phases"] for s in p["segments"])+exact["total_samples"]+attempt["exact_board_branch"]["total_samples"]+sum(c["total_samples"] for c in reverse_checks)
                clear=[s["minimum_clearance_m"] for p in attempt["phases"] for s in p["segments"]]+[exact["minimum_clearance_m"],attempt["exact_board_branch"]["minimum_clearance_m"]]+[c["minimum_clearance_m"] for c in reverse_checks]
                indices=np.unique(np.linspace(0,len(frames)-1,min(100,len(frames))).astype(int))
                result.update(passed=True,reason=None,selected_attempt=len(result["attempts"])-1,
                    checked_joint_configurations=int(count),minimum_clearance_lower_bound_m=min(clear),
                    frames=[frames[i] for i in indices],returned_home_joint_max_error_rad=float(np.abs(reverse[-1]-homeq).max()),
                    exact_selected_joint_max_error_rad=0.,elapsed_s=time.monotonic()-begin,
                    limitations=["Finite sampled geometry, not continuous proof or actuator dynamics",
                        "Proxy geometry and assumed rigid carried ball; no grasp stability or real robot certification"])
                self.cache[fingerprint]=copy.deepcopy(result);return result
        result.update(reason="finite_common_home_connection_unresolved",elapsed_s=time.monotonic()-begin)
        self.cache[fingerprint]=copy.deepcopy(result);return result

    def admit(self,row):
        result=self._admit(row)
        return {**{k:result.get(k) for k in ("passed","scope","proof_scope","reason","common_home_sha256","request_sha256")},
            "report":{k:v for k,v in result.items() if k!="frames"},"frames":result.get("frames",[])}

    def connect_variation(self,source_row,jitter_row):
        """Compose common-home proof with serialized nominal→jitter local edges.

        The perturbed exact branches and request must pass fresh strict endpoint
        checks. A changed bucket layout requires its own common-home admission.
        """
        nominal=self.admit(source_row)
        result=dict(version=VERSION,passed=False,scope="fixed_common_home_via_nominal_to_exact_variation_and_reverse_sampled_geometry",
            proof_scope="fixed_common_home_via_nominal_to_exact_variation_and_reverse_sampled_geometry",
            common_home_sha256=self.home["sha256"],nominal_request_sha256=nominal["request_sha256"],frames=[])
        if not nominal["passed"]:
            result["reason"]="nominal_common_home_connection_not_admitted";return result
        xy=np.asarray(jitter_row["bucket_xy_m"],float)
        if not np.array_equal(xy,np.asarray(source_row["bucket_xy_m"],float)):
            result["reason"]="variation_layout_changed_requires_full_admission";return result
        v=np.asarray(jitter_row["values"],float);q=np.r_[jitter_row["q10"],jitter_row["q14"]]
        if v.shape!=(5,) or q.shape!=(12,) or not np.isfinite(np.r_[v,q]).all():raise ValueError("Finite variation action5 and12joints required")
        source=np.r_[source_row["q10"],source_row["q14"]]
        within,reach=outer_geometry_mask(self.scene,xy[None],v[None,None],bounds=WIDE_BOUNDS)
        if not bool((within&reach)[0,0]):
            result["reason"]="variation_request_bounds_or_workspace_rejected";return result
        targets=shot_targets(self.scene,xy[None],v[None,:3],[v[3]],[v[4]])
        target={k:a[0] for k,a in targets.items()}
        endpoint=self.endpoint.check(q[:6],q[6:],xy,v,target)
        identity=dict(bucket_xy_m=xy.tolist(),values=v.tolist(),q10=q[:6].tolist(),q14=q[6:].tolist())
        result.update(request=identity,request_sha256=digest(identity),endpoint=endpoint)
        if endpoint["reason"]:
            result["reason"]="variation_endpoint_not_admitted";return result
        self.gate.bucket_xy=xy
        middle=source.copy();middle[6:]=q[6:]
        pairs=[(source,middle),(middle,q),(q,middle),(middle,source)]
        checks=[audit_segment(self.gate,a,b) for a,b in pairs]
        result["local_edges"]=checks
        if not all(c["passed"] for c in checks):
            result["reason"]="variation_local_sampled_path_rejected";return result
        result.update(passed=True,reason=None,checked_joint_configurations=sum(c["total_samples"] for c in checks),
            minimum_clearance_lower_bound_m=min(c["minimum_clearance_m"] for c in checks),
            returned_nominal_joint_max_error_rad=0.)
        return result
