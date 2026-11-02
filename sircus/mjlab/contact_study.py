"""Finite paired contact/step/reset diagnosis; no policy training or promotion."""

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from .learning_contract import digest

VERSION = "sircus-contact-repeatability-study-v1"
SOURCE_ID = "stable-shot-0004711"
SEED = 20261124
NOISE = np.asarray([.001, .001, .002, .35, .002])


def atomic(path,value):
    temp=path.with_suffix(".tmp")
    temp.write_text(json.dumps(value,indent=2,allow_nan=False)+"\n")
    temp.replace(path)


def shot_plan(source, final_rows, fresh=20, nominal=16):
    """Exact provenance and separate fresh noise; outcomes never select inputs."""
    rng = np.random.default_rng(SEED)
    plans = [dict(id=f"nominal-{i:02d}", family="nominal", values=source["values"],
                  q10=source["q10"], q14=source["q14"]) for i in range(nominal)]
    for row in final_rows:
        plans.append(dict(id=row["id"], family="original_final_development",
            values=row["values"], q10=row["q10"], q14=row["q14"],
            historical_success=row["route_success"]))
    for i in range(fresh):
        plans.append(dict(id=f"fresh-{i:02d}", family="fresh_perturbation",
                          values=(np.asarray(source["values"])+rng.normal(0, NOISE)).tolist()))
    return plans


def flight_metrics(arrays, row, index):
    f=arrays["forces_N"][:,index];p=arrays["position_m"][:,index];v=arrays["velocity_m_s"][:,index]
    board=np.flatnonzero(f[:,0]>.01)
    if not len(board):return dict(first_board_contact_s=None, outgoing_velocity_m_s=None)
    first=int(board[0]);end=first
    while end+1<len(f) and f[end+1,0]>.01:end+=1
    after=min(end+1,len(f)-1)
    normal=np.asarray(row["board_normal"])
    return dict(first_board_contact_s=(first+1)/240, first_board_contact_end_s=(end+1)/240,
        incoming_velocity_m_s=v[max(0,first-2)].tolist(),outgoing_velocity_m_s=v[after].tolist(),
        outgoing_normal_speed_m_s=float(v[after]@normal),
        contact_displacement_m=(p[end]-p[max(0,first-1)]).tolist(),
        board_peak_N=float(f[:,0].max()),bucket_peak_N=float(f[:,1].max()))


def summarize(records, arrays):
    result=[]
    for family in dict.fromkeys(r["family"] for r in records):
        rows=[r for r in records if r["family"]==family]
        result.append(dict(family=family,attempts=len(rows),released=sum(r["native_release"] for r in rows),
            captures=sum(r["route_success"] for r in rows),clean=sum(r["route_quality"]==2 for r in rows),
            board_bounces=sum(r["board_bounce"] for r in rows),
            invalid=sum(bool(r["invalid"]) for r in rows),
            max_integrated_joint_excursion_rad=max(r["max_integrated_joint_excursion_rad"] for r in rows),
            peak_joint_velocity_rad_s=max(r["peak_integrated_joint_velocity_rad_s"] for r in rows)))
    return result


def run(a):
    from .expanded_actions import WIDE_BOUNDS
    from .stable_geometry import audit_values
    prior=json.loads((a.prior_study/"contract.json").read_text())
    old=[json.loads(s) for s in (a.prior_study/"shots.jsonl").read_text().splitlines()]
    source=next(r for r in old if r["id"]==SOURCE_ID)
    final=[r for r in old if r["phase"]=="final_confirmation" and r.get("confirmation_source_id")==SOURCE_ID]
    if len(final)!=20:raise RuntimeError("Expected exact historical 20-shot final")
    plan=shot_plan(source,final,a.fresh,a.nominal)
    xy=np.tile(source["bucket_xy_m"],(len(plan),1));values=np.asarray([p["values"] for p in plan])
    known10=np.asarray([p.get("q10",source["q10"]) for p in plan]);known14=np.asarray([p.get("q14",source["q14"]) for p in plan])
    knownvalues=np.asarray([p["values"] if "q10" in p else source["values"] for p in plan])
    mask,detail=audit_values(prior["scene"],a.assets,xy,values[:,None],bounds=WIDE_BOUNDS,
        known_q10=known10,known_q14=known14,known_values=knownvalues,known_bucket_xy=xy,
        starts=16,iterations=180)
    eligible=mask[:,0];q10=detail["q10"][:,0];q14=detail["q14"][:,0]
    for i,p in enumerate(plan):
        p.update(bucket_xy_m=xy[i].tolist(),q10=q10[i].tolist(),q14=q14[i].tolist(),
            geometry_eligible=bool(eligible[i]),geometry_reason=int(detail["reasons"][i,0]))
    variants=[dict(id=f"{mode}-{gravity}-{steps}",board_mode=mode,gravity_mode=gravity,substeps=steps)
        for mode,gravity in (("articulated","legacy"),("articulated","passive_only"),("kinematic","passive_only"))
        for steps in a.substeps]
    spec=dict(version=VERSION,source_id=SOURCE_ID,source=source,seed=SEED,scene=prior["scene"],
        noise_std=NOISE.tolist(),prior_contract_sha256=hashlib.sha256((a.prior_study/"contract.json").read_bytes()).hexdigest(),
        prior_records_sha256=hashlib.sha256((a.prior_study/"shots.jsonl").read_bytes()).hexdigest(),
        variants=variants,reset_repeats=2,plan=plan,
        semantics="Original scorer/material/geometry retained.8/16/32native substeps per240Hz score. Legacy has extra arm bias force despite passive gravcomp1. Passive-only removes duplicate force; articulated board still responds within substep. Kinematic ideal-held proxy explicitly changes board effective inertia; no physical calibration claim.",
        scope="Finite paired development diagnosis, not training, teacher promotion or untouched validation")
    spec["fingerprint"]=digest(spec)
    a.output.mkdir(parents=True,exist_ok=False);atomic(a.output/"contract.json",spec)
    atomic(a.output/"plan.json",plan)
    report=dict(status="running",version=VERSION,source_id=SOURCE_ID,source=source,
        contract_fingerprint=spec["fingerprint"],groups=[],records=0,scope=spec["scope"])
    atomic(a.output/"report.json",report)
    if not eligible.all():
        report.update(status="blocked_geometry",geometry_rejects=int((~eligible).sum()))
        atomic(a.output/"report.json",report)
        raise RuntimeError("Paired contact diagnosis requires identical qualified inputs; rejected rows are recorded, never removed")
    from .gpu_slot import gpu_slot
    from .contact_env import ContactEnvironment
    from .bank_study import run_bank_rows
    start=time.monotonic()
    with gpu_slot(a.gpu_lock,timeout_s=1800,notify=lambda x:print(json.dumps(x),flush=True)):
        for variant in variants:
            env=ContactEnvironment(prior["scene"],a.assets,n=len(plan),bounds=WIDE_BOUNDS,
                substeps=variant["substeps"],board_mode=variant["board_mode"],gravity_mode=variant["gravity_mode"])
            result=[];first_arrays=None
            for reset in range(2):
                order=np.arange(len(plan)) if reset==0 else np.arange(len(plan)-1,-1,-1)
                rows=run_bank_rows(env,xy[order],values[order],q10[order],q14[order],eligible[order],record=True)
                raw=env.last_arrays
                arrays={k:v[:,np.argsort(order)] for k,v in raw.items()}
                if len(rows)!=len(plan):raise RuntimeError("Query count changed")
                rows=[rows[int(j)] for j in np.argsort(order)]
                np.savez_compressed(a.output/f'{variant["id"]}-reset-{reset}.npz',**arrays)
                for i,(p,r) in enumerate(zip(plan,rows)):
                    r.pop("trajectory_m",None)
                    r.update(id=f'{variant["id"]}-reset-{reset}-{p["id"]}',source_plan_id=p["id"],
                        family=p["family"],variant=variant["id"],reset=reset,
                        **flight_metrics(arrays,r,i))
                    result.append(r)
                if reset==0:first_arrays=arrays
                else:
                    same=np.max(np.abs(arrays["position_m"]-first_arrays["position_m"]),axis=(0,2))
                    reset_metrics=dict(max_position_difference_m=float(same.max()),
                        identical_trajectories=int((same==0).sum()),queries=len(same),
                        capture_disagreements=sum(x["route_success"]!=y["route_success"] for x,y in zip(result[:len(plan)],rows)))
            with (a.output/"results.jsonl").open("a") as f:
                for r in result:f.write(json.dumps(r,allow_nan=False)+"\n")
            group=dict(**variant,attempts=len(result),captures=sum(r["route_success"] for r in result),
                family_results=summarize(result,None),reset_order=reset_metrics)
            report["groups"].append(group);report["records"]+=len(result)
            report.update(elapsed_s=time.monotonic()-start)
            atomic(a.output/"report.json",report);print(json.dumps(group),flush=True)
            del env
            import gc,torch
            gc.collect();torch.cuda.empty_cache()
    report.update(status="completed",elapsed_s=time.monotonic()-start)
    atomic(a.output/"report.json",report)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("prior-study","assets","output","gpu-lock"):p.add_argument("--"+name,type=Path,required=True)
    p.add_argument("--substeps",type=int,nargs="+",default=[8,16,32]);p.add_argument("--fresh",type=int,default=20);p.add_argument("--nominal",type=int,default=16)
    a=p.parse_args()
    if a.fresh<1 or a.nominal<2 or a.fresh+a.nominal>108 or not set(a.substeps)<={8,16,32}:p.error("Finite controlled budgets required")
    run(a)


if __name__=="__main__":main()
