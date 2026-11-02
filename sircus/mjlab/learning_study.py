"""Finite coverage exploration, imitation warm start and genuine terminal PPO.

All test layouts are frozen before collection. No test-scene optimization or
test reward enters training, warm start, checkpoint selection or inference.
"""

import argparse
import copy
import datetime
import fcntl
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import time

import numpy as np
import torch
from .learning_contract import VERSION,SEED,LOW,HIGH,NAMES,splits,constraints,observations,actions,proposals,reward,statistics,digest
from .learning_env import ShotEnvironment
from .learning_policy import Policy,update,warm_start,weight_hash
from .learning_reach import sample_proposals
from .run import atomic


def run(a):
    # The same lock is used by the existing native Isaac jobs. Open read-only:
    # lock file ownership need not be changed and Isaac setup stays intact.
    lock=open(a.gpu_lock,"r")
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    processes=subprocess.check_output(["nvidia-smi","--query-compute-apps=gpu_uuid,pid","--format=csv,noheader"],text=True)
    if any("GPU-ab1950b0" in line for line in processes.splitlines()):
        raise RuntimeError("4090 already has another compute process; no workload will be stopped")
    torch.set_num_threads(2)
    torch.manual_seed(SEED);np.random.seed(SEED)
    rng=np.random.default_rng(SEED)
    a.output.mkdir(parents=True,exist_ok=False)
    scene=json.loads(a.benchmark.read_text())["scene"]
    pools=splits()
    spec=dict(version=VERSION,seed=SEED,scene=scene,action_names=NAMES,low=LOW.tolist(),high=HIGH.tolist(),
              train_layouts=pools["train"],validation_layouts=pools["validation"],test_layouts=pools["test"],
              exploration_batches=a.explore,ppo_updates=a.updates,batch_size=a.batch,
              capture="unchanged ShotScore + EntryQuality + BoardRoute; one-second whole-ball retention after board rebound; no external contact",
              learning="return-weighted train-only imitation warmstart, then actual on-policy terminal PPO; gamma=1, no GAE/bootstrap",
              proposal_strategy="Both UF850 wrist-distance priors below .81m, 64 finite resamples; full IK/contact gates remain authoritative",
              checkpoint_selection="Validation mean shaped reward; test never selects or updates weights",
              constraints="exact boardXYZ/releaseXYZ positional locks; center-impact vertical-drop family; bucket-facing board yaw",
              physics=dict(engine="mjlab MuJoCo Warp",dt_s=1/1920,scoring_dt_s=1/240,solref=[.004,.15],friction=.35),
              scope="Simulated privileged-geometry endpoint configuration learning. Arm endpoints initialized/held; no grasp or transit policy, no perception, no physical validation.")
    spec["fingerprint"]=digest(spec)
    atomic(a.output/"contract.json",spec);atomic(a.output/"splits.json",pools)
    report=dict(status="running",phase="initializing",seed=SEED,contract_fingerprint=spec["fingerprint"],
                scope=spec["scope"],train_episodes=0,updates=0,learning_curve=[],coverage=dict(cells=[]),frozen_evaluation=None)
    start=time.monotonic()
    def publish(phase):
        report.update(phase=phase,elapsed_s=time.monotonic()-start,timestamp_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
        atomic(a.output/"report.json",report)
        print(json.dumps(dict(phase=phase,train_episodes=report["train_episodes"],updates=report["updates"],elapsed_s=round(report["elapsed_s"],1))),flush=True)
    publish("initializing")
    env=ShotEnvironment(scene,a.assets,a.batch)
    records=[]
    def trial(bucket,values,locks,phase,record=False):
        rows,timing=env.run(bucket,values,locks,record=record)
        offset=len(records)
        for i,row in enumerate(rows):
            row.update(id=f"shot-{offset+i:07d}",phase=phase,locks={k:None if not np.isfinite(v[i]).all() else v[i].tolist() for k,v in locks.items()})
        records.extend(rows)
        with (a.output/"shots.jsonl").open("a") as out:
            for row in rows: out.write(json.dumps(row,allow_nan=False)+"\n")
        return rows,timing
    train=np.asarray(pools["train"])
    exploration=[]
    for batch in range(a.explore):
        ids=np.arange(a.batch)+batch*a.batch
        xy=train[ids%len(train)]
        values,exhausted=sample_proposals(xy,rng,scene)
        rows,timing=trial(xy,values,constraints(a.batch),"train-exploration")
        exploration.extend(rows);report["train_episodes"]+=len(rows)
        report["exploration"]=statistics(exploration)
        report["last_batch"]=dict(**statistics(rows),**timing,proposal_filter_exhausted=int(exhausted.sum()))
        publish("exploration")
    elite=[]
    cells=[]
    for i,xy in enumerate(train):
        group=[r for r in exploration if np.allclose(r["bucket_xy_m"],xy,atol=1e-12)]
        valid=sorted([r for r in group if not r["invalid"]],key=reward,reverse=True)
        elite.extend(valid[:3])
        cells.append(dict(id=f"train-{i:02d}",bucket_xy_m=xy.tolist(),split="train",captures=sum(r["route_success"] for r in valid),trials=len(group),valid=len(valid),best=valid[0] if valid else None))
    report["coverage"]["cells"]=cells
    # Confirm top TRAIN candidates with small independent perturbations. This
    # improves the teacher catalogue; these are development, never test data.
    confirm_values,confirm_xy,confirm_refs=[],[],[]
    for ci,cell in enumerate(cells):
        group=sorted([r for r in exploration if not r["invalid"] and np.allclose(r["bucket_xy_m"],cell["bucket_xy_m"],atol=1e-12)],key=reward,reverse=True)[:3]
        for candidate in group:
            for rep in range(3):
                value=np.asarray(candidate["values"])+rng.normal(0,[.001,.001,.002,.35,.002])
                confirm_values.append(np.clip(value,LOW,HIGH));confirm_xy.append(cell["bucket_xy_m"]);confirm_refs.append((ci,candidate["id"]))
    confirmations=[]
    for offset in range(0,len(confirm_values),a.batch):
        rr,_=trial(np.asarray(confirm_xy[offset:offset+a.batch]),np.asarray(confirm_values[offset:offset+a.batch]),constraints(min(a.batch,len(confirm_values)-offset)),"train-confirmation")
        for row,ref in zip(rr,confirm_refs[offset:offset+len(rr)]): row["teacher_source_id"]=ref[1]
        confirmations.extend(rr);report["train_episodes"]+=len(rr)
        publish("catalogue-confirmation")
    elite=[]
    for cell in cells:
        group=[r for r in exploration if not r["invalid"] and np.allclose(r["bucket_xy_m"],cell["bucket_xy_m"],atol=1e-12)]
        if not group: continue
        def confirmed_rank(r):
            repeats=[x for x in confirmations if x.get("teacher_source_id")==r["id"]]
            return (sum(x["route_success"] and not x["invalid"] for x in repeats),reward(r))
        selected_teacher=max(group,key=confirmed_rank)
        cell["best"]=selected_teacher
        repeats=[x for x in confirmations if x.get("teacher_source_id")==selected_teacher["id"]]
        cell["confirmation"]=statistics(repeats)
        elite.append(selected_teacher)
    atomic(a.output/"catalogue.json",cells)
    if not elite: raise RuntimeError("No valid exploratory endpoints; learning not claimed")
    # Augment exact same successful train configurations with visible lock masks;
    # this is imitation, not independent constrained-scene success evidence.
    wx,wy,ww=[],[],[]
    for r in elite:
        for mode in ("free","board","dropper"):
            locks=constraints(1)
            if mode=="board": locks["board"][0]=r["values"][:3]
            if mode=="dropper": locks["dropper"][0]=r["release_ball_center_m"]
            wx.append(observations([r["bucket_xy_m"]],locks)[0]);wy.append(r["values"])
            ww.append(1. if r["route_success"] else .15)
    policy=Policy().to(env.device)
    initial=copy.deepcopy(policy.state_dict())
    torch.save(dict(state=initial,contract=spec),a.output/"initial.pt")
    report["imitation"]=dict(samples=len(wx),source="train exploration elites only",loss=warm_start(policy,np.asarray(wx),np.asarray(wy),ww))
    warm=copy.deepcopy(policy.state_dict())
    torch.save(dict(state=warm,contract=spec),a.output/"warm.pt")
    opt=torch.optim.Adam(policy.parameters(),lr=.0003)
    selected=copy.deepcopy(warm);best_validation=-float("inf");best_update=0
    def evaluate(label,xy,state,repeat=1,lock_mode=None,reference=None,record=False):
        policy.load_state_dict(state)
        allrows=[]
        for rep in range(repeat):
            for offset in range(0,len(xy),a.batch):
                bx=np.asarray(xy[offset:offset+a.batch]);locks=constraints(len(bx))
                if lock_mode and reference is not None:
                    for i,idx in enumerate(range(offset,offset+len(bx))):
                        ref=reference[idx]
                        if lock_mode=="board": locks["board"][i]=ref["values"][:3]
                        else: locks["dropper"][i]=ref["release_ball_center_m"]
                x=torch.tensor(observations(bx,locks),device=env.device)
                with torch.no_grad(): z=policy.actor(x).cpu().numpy()
                rs,_=trial(bx,actions(z),locks,label,record=record)
                allrows.extend(rs)
        return allrows
    # Validation participates in checkpoint selection; never use test here.
    val=evaluate("validation-warm",pools["validation"],warm)
    best_validation=statistics(val)["mean_reward"]
    report["warm_validation"]=statistics(val)
    publish("warm-start")
    for iteration in range(a.updates):
        policy.load_state_dict(warm if iteration==0 else current)
        ids=rng.integers(0,len(train),a.batch)
        xy=train[ids];locks=constraints(a.batch)
        # Real PPO constraints are anchored to train-only exploratory endpoints.
        for i in range(a.batch):
            ref=cells[ids[i]]["best"]
            mode=rng.choice(["free","board","dropper"],p=[.6,.2,.2])
            if ref and mode=="board": locks["board"][i]=ref["values"][:3]
            if ref and mode=="dropper": locks["dropper"][i]=ref["release_ball_center_m"]
        x=torch.tensor(observations(xy,locks),device=env.device)
        with torch.no_grad():
            dist=policy.distribution(x);z=dist.sample();logp=dist.log_prob(z).sum(-1)
        rows,timing=trial(xy,actions(z.cpu().numpy()),locks,"train-ppo")
        returns=torch.tensor([reward(r) for r in rows],device=env.device,dtype=torch.float32)
        log=update(policy,opt,x,z,logp,returns)
        current=copy.deepcopy(policy.state_dict())
        report["train_episodes"]+=len(rows);report["updates"]=iteration+1
        curve=dict(update=iteration+1,**statistics(rows),timing=timing,weight_change=log)
        report["learning_curve"].append(curve)
        if (iteration+1)%a.validate_every==0 or iteration+1==a.updates:
            val=evaluate(f"validation-{iteration+1}",pools["validation"],current)
            curve["validation"]=statistics(val)
            if curve["validation"]["mean_reward"]>best_validation:
                selected=copy.deepcopy(current);best_validation=curve["validation"]["mean_reward"];best_update=iteration+1
        torch.save(dict(state=current,contract=spec,updates=iteration+1),a.output/"latest.pt")
        publish("ppo")
    checkpoint=dict(state=selected,contract=spec,selected_update=best_update,weight_sha256=None)
    policy.load_state_dict(selected);checkpoint["weight_sha256"]=weight_hash(policy)
    torch.save(checkpoint,a.output/"selected.pt")
    selected_hash=checkpoint["weight_sha256"]
    report["selected_update"]=best_update
    report["checkpoint_sha256"]=hashlib.sha256((a.output/"selected.pt").read_bytes()).hexdigest()
    publish("frozen-evaluation")
    # Freeze before looking at any held-out test results. No search/refinement.
    evaluation={}
    free_reference=None
    for label,state in (("initial",initial),("warm",warm),("selected",selected)):
        rows=evaluate(f"test-{label}",pools["test"],state,repeat=a.test_repeats,record=(label=="selected"))
        evaluation[label]=dict(**statistics(rows),policy_weight_sha256=weight_hash(policy),layouts=len(pools["test"]),repeats=a.test_repeats)
        if label=="selected":
            free_reference=rows[:len(pools["test"])]
            report["test_rows"]=rows
    for mode in ("board","dropper"):
        rows=evaluate("test-lock-"+mode,pools["test"],selected,lock_mode=mode,reference=free_reference)
        evaluation[mode+"_locked"]=dict(**statistics(rows),layouts=len(pools["test"]),constraint_source="frozen selected free-policy endpoints at held-out XY; not arbitrary XYZ constraints")
    policy.load_state_dict(selected)
    if weight_hash(policy)!=selected_hash: raise RuntimeError("Checkpoint changed during evaluation")
    report["frozen_evaluation"]=dict(**evaluation["selected"],trials=evaluation["selected"]["attempts"],comparison=evaluation,search_at_inference=False)
    report["status"]="completed"
    report["versions"]={k:importlib.metadata.version(k) for k in ("mjlab","mujoco","mujoco-warp","torch","warp-lang")}
    report["source_sha256"]={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob("*.py")}
    report["records"]=len(records)
    publish("completed")


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--benchmark",type=Path,required=True);p.add_argument("--assets",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True);p.add_argument("--gpu-lock",type=Path,required=True)
    p.add_argument("--batch",type=int,default=128);p.add_argument("--explore",type=int,default=24)
    p.add_argument("--updates",type=int,default=24);p.add_argument("--validate-every",type=int,default=4)
    p.add_argument("--test-repeats",type=int,default=3)
    a=p.parse_args()
    if not 1<=a.batch<=256 or not 1<=a.explore<=256 or not 1<=a.updates<=256: p.error("Finite positive study budget required")
    run(a)
