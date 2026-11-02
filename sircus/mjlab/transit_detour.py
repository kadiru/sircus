"""Bounded, seeded six-joint detours; CPU geometry only, never actuation."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from scipy.spatial.transform import Rotation

from .transit_audit import GeometricGate, audit_segment, VERSION as TRANSIT_VERSION

VERSION="sircus-bounded-joint-detour-v1"


def rrt_detour(first,goal,edge,lower,upper,*,seed=20261122,max_nodes=500,deadline=None):
    """Finite local-box RRT. Every extension and final join uses full edge audit."""
    first,goal,lower,upper=map(lambda x:np.asarray(x,float),(first,goal,lower,upper))
    if any(a.shape!=(6,) for a in (first,goal,lower,upper)) or not np.isfinite([first,goal,lower,upper]).all():
        raise ValueError("Finite six-joint arrays required")
    if not 1<=max_nodes<=500:raise ValueError("Node budget1..500 required")
    direct=edge(first,goal)
    if direct["passed"]:return [first,goal],dict(passed=True,nodes=2,seed=seed,direct=True,edge_checks=1)
    lo=np.maximum(np.minimum(first,goal)-.25,lower)
    hi=np.minimum(np.maximum(first,goal)+.25,upper)
    rng=np.random.default_rng(seed);nodes=[first];parents=[-1];checks=1
    for attempt in range(max_nodes-1):
        if deadline is not None and time.monotonic()>=deadline:
            return None,dict(passed=False,reason="wall_budget",nodes=len(nodes),seed=seed,edge_checks=checks)
        target=goal if rng.random()<.25 else rng.uniform(lo,hi)
        distance=np.linalg.norm(np.asarray(nodes)-target,axis=1)
        nearest=int(distance.argmin());delta=target-nodes[nearest]
        proposed=nodes[nearest]+delta*min(1.,.03/max(np.linalg.norm(delta),1e-12))
        checks+=1
        if not edge(nodes[nearest],proposed)["passed"]:continue
        nodes.append(proposed);parents.append(nearest)
        if np.linalg.norm(goal-proposed)<.08:
            checks+=1
            if edge(proposed,goal)["passed"]:
                path=[goal];index=len(nodes)-1
                while index>=0:path.append(nodes[index]);index=parents[index]
                return path[::-1],dict(passed=True,nodes=len(nodes),seed=seed,direct=False,edge_checks=checks)
    return None,dict(passed=False,reason="node_budget",nodes=len(nodes),seed=seed,edge_checks=checks)


def main():
    from .stable_ik import StableShotIK
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--study",type=Path,required=True);p.add_argument("--failed-routes",type=Path,required=True)
    p.add_argument("--assets",type=Path,required=True);p.add_argument("--output",type=Path,required=True)
    p.add_argument("--wall-budget",type=float,default=180)
    a=p.parse_args()
    if a.output.exists():raise ValueError("Fresh output required")
    if not 1<=a.wall_budget<=300:raise ValueError("Finite1..300s budget required")
    a.output.mkdir(parents=True)
    cfg=json.loads((a.study/"contract.json").read_text())["scene"]
    failed=json.loads(a.failed_routes.read_text())["trials"]
    ik=StableShotIK(cfg,a.assets);results=[];start=time.monotonic();deadline=start+a.wall_budget
    for trial in failed:
        if trial["source_id"] not in ("stable-shot-0007490","stable-shot-0007729","stable-shot-0009165"):continue
        gate=GeometricGate(cfg,a.assets,trial["bucket_xy_m"])
        case_deadline=min(deadline,time.monotonic()+a.wall_budget/3)
        q0=np.r_[trial["q10"],trial["q14"]]
        result=dict(source_id=trial["source_id"],bucket_xy_m=trial["bucket_xy_m"],values=trial["values"],passed=False,attempts=[])
        results.append(result)
        variants=trial.get("attempt_history",[])+[{k:v for k,v in trial.items() if k!="attempt_history"}]
        # If the requested very-low goal is unreachable, a declared near-source
        # ring tests whether a modest low reconfiguration is feasible at all.
        # No endpoint or clearance limit is relaxed.
        template=next((v for v in variants if v["lift_mode"]=="retract" and len(v["phases"])>=3),None)
        if template is not None:
            gate.state(q0)
            body=gate.m.body("Arm14/link6").id
            original=np.eye(4);original[:3,:3]=gate.d.xmat[body].reshape(3,3);original[:3,3]=gate.d.xpos[body]
            for dx,dy,dz in ((0,.05,-.03),(0,-.05,-.03),(.05,0,-.03),(-.05,0,-.03),
                (.03,.03,-.03),(-.03,-.03,-.03),(0,.05,.03),(0,-.05,.03)):
                variant=copy.deepcopy(template)
                goal_world=original.copy();goal_world[:3,3]+=[dx,dy,dz]
                variant["phases"][2]["target_flange_world"]=goal_world.tolist()
                variant["requested_low_board_center_m"]=trial["values"][2]+dz
                variant["board_retraction_xy_m"]=[dx,dy]
                variant["near_source_goal_offset_m"]=[dx,dy,dz]
                variants.append(variant)
        for vi,variant in enumerate(variants):
            if time.monotonic()>=case_deadline:
                result["budget_exhausted"]=True;break
            if variant["lift_mode"]!="retract":continue
            phases=variant["phases"]
            if len(phases)<3:continue
            current=q0.copy();prefix=[];prefix_ok=True
            for phase in phases[:2]:
                for seg in phase["segments"]:
                    target=np.r_[seg["q10"],seg["q14"]]
                    checked=audit_segment(gate,current,target,corridor=phase["corridor"])
                    if not checked["passed"]:prefix_ok=False;break
                    prefix.append(dict(first=current.tolist(),second=target.tolist(),audit=checked))
                    current=target
                if not prefix_ok:break
            attempt=dict(variant=vi,low_board_center_m=variant["requested_low_board_center_m"],
                board_shift_xy_m=variant["board_retraction_xy_m"],prefix_passed=prefix_ok,passed=False)
            if variant.get("near_source_goal_offset_m") is not None:
                attempt["near_source_goal_offset_m"]=variant["near_source_goal_offset_m"]
            result["attempts"].append(attempt)
            if not prefix_ok:continue
            world=np.asarray(phases[2]["target_flange_world"])
            base=np.eye(4);base[:3,:3]=Rotation.from_euler("z",cfg["arms"]["board_base_yaw_rad"]).as_matrix();base[0,3]=cfg["arms"]["base_spacing_m"]
            pool=ik.board.solve_candidates((np.linalg.inv(base)@world)[None],known=current[6:],starts=16,iterations=180)
            qs=pool["q"][0].cpu().numpy();valid=pool["valid"][0].cpu().numpy()
            attempt["valid_ik_branches"]=int(valid.sum());attempt["goals"]=[]
            cap=max(gate.check(current)["board_top_m"],variant["requested_low_board_center_m"]+.13)+.05
            corridor=dict(board_top_max_m=cap,dropper_flange_min_m=.94)
            for bi in sorted(np.flatnonzero(valid),key=lambda i:np.linalg.norm(qs[i]-current[6:])):
                if time.monotonic()>=case_deadline:break
                goal=current.copy();goal[6:]=qs[bi]
                endpoint=gate.check(goal,corridor=corridor)
                g=dict(branch=int(bi),endpoint=endpoint)
                attempt["goals"].append(g)
                if not endpoint["passed"]:continue
                def edge(first,second):
                    x,y=current.copy(),current.copy();x[6:]=first;y[6:]=second
                    return audit_segment(gate,x,y,corridor=corridor)
                seed=int.from_bytes(hashlib.sha256(f"{trial['source_id']}:{vi}:{bi}".encode()).digest()[:4],"little")
                path,meta=rrt_detour(current[6:],goal[6:],edge,gate.native.limits14[:,0],gate.native.limits14[:,1],seed=seed,deadline=case_deadline)
                g["rrt"]=meta
                if path is None:continue
                audits=[edge(x,y) for x,y in zip(path,path[1:])]
                if not all(e["passed"] for e in audits):raise RuntimeError("Independent path reaudit failed")
                result.update(passed=True,selected_variant=vi,selected_branch=int(bi),corridor=corridor,
                    board_path_joint_rad=[q.tolist() for q in path],path_audits=audits,
                    prefix=prefix,initial_q=q0.tolist(),parked_dropper_q=current[:6].tolist(),
                    minimum_clearance_lower_bound_m=min(e["minimum_clearance_m"] for e in audits),
                    scope="Prefix plus board detour to low configuration; reverse follows same audited samples. Other low transfers not certified")
                attempt["passed"]=True;break
            if result["passed"] or time.monotonic()>=case_deadline:break
        print(trial["source_id"],result["passed"],flush=True)
    report=dict(version=VERSION,transit_version=TRANSIT_VERSION,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        failed_routes_sha256=hashlib.sha256(a.failed_routes.read_bytes()).hexdigest(),
        budgets=dict(max_nodes=500,step_norm_rad=.03,joint_box_padding_rad=.25,edge_sample_max_rad=float(np.deg2rad(.5)),
            global_wall_s=a.wall_budget),elapsed_s=time.monotonic()-start,trials=results,
        scope="Finite CPU joint-space geometry experiment; no dynamics, training or actuation")
    (a.output/"report.json").write_text(json.dumps(report,indent=2,allow_nan=False))


if __name__=="__main__":main()
