"""Fresh paired confirmation of the unchanged nominal shot after contact fixes.

No candidate selection, search, optimization, training or policy promotion.
"""

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from .contact_study import NOISE, SOURCE_ID, atomic, flight_metrics
from .learning_contract import digest

VERSION="sircus-contact-fresh-confirmation-v1"
SEED=20261125


def confirmation_plan(source, count=100, nominal=16):
    rng=np.random.default_rng(SEED)
    plan=[dict(id=f"nominal-{i:03d}",family="nominal",values=source["values"]) for i in range(nominal)]
    plan.extend(dict(id=f"fresh-{i:03d}",family="fresh_perturbation",
        values=(np.asarray(source["values"])+rng.normal(0,NOISE)).tolist()) for i in range(count))
    return plan


def paired_differences(first,second,rows1,rows2):
    """Separate pre-bucket ballistic differences from later chaotic motion."""
    full=np.abs(first["position_m"]-second["position_m"]).max(axis=(0,2))
    prefix=[];prebucket=[]
    for i,(r1,r2) in enumerate(zip(rows1,rows2)):
        stop=min(r1.get("physics_steps",960),r2.get("physics_steps",960))
        prefix.append(float(np.abs(first["position_m"][:stop,i]-second["position_m"][:stop,i]).max()))
        contacts=[np.flatnonzero(x["forces_N"][:,i,1]>.01) for x in (first,second)]
        limit=min([int(x[0]) for x in contacts if len(x)]+[stop])
        prebucket.append(float(np.abs(first["position_m"][:max(1,limit),i]-second["position_m"][:max(1,limit),i]).max()))
    return dict(max_full_horizon_difference_m=float(full.max()),
        max_common_scored_prefix_difference_m=max(prefix),
        max_before_first_bucket_contact_difference_m=max(prebucket),
        capture_disagreements=sum(r1["route_success"]!=r2["route_success"] for r1,r2 in zip(rows1,rows2)),
        exactly_identical_trajectories=int((full==0).sum()),queries=len(rows1),
        scope="Full4s includes postterminal motion; prebucket measures flight before either paired run first touches the bucket")


def run(a):
    from .expanded_actions import WIDE_BOUNDS
    from .stable_geometry import audit_values
    from .gpu_slot import gpu_slot
    from .contact_env import ContactEnvironment
    from .bank_study import run_bank_rows
    contract=json.loads((a.prior_study/"contract.json").read_text())
    old=[json.loads(x) for x in (a.prior_study/"shots.jsonl").read_text().splitlines()]
    source=next(r for r in old if r["id"]==SOURCE_ID)
    plan=confirmation_plan(source,a.fresh,a.nominal);n=len(plan)
    xy=np.tile(source["bucket_xy_m"],(n,1));values=np.asarray([p["values"] for p in plan])
    mask,detail=audit_values(contract["scene"],a.assets,xy,values[:,None],bounds=WIDE_BOUNDS,
        known_q10=np.tile(source["q10"],(n,1)),known_q14=np.tile(source["q14"],(n,1)),
        known_values=np.tile(source["values"],(n,1)),known_bucket_xy=xy,starts=16,iterations=180)
    eligible=mask[:,0];q10=detail["q10"][:,0];q14=detail["q14"][:,0]
    for i,p in enumerate(plan):p.update(q10=q10[i].tolist(),q14=q14[i].tolist(),geometry_eligible=bool(eligible[i]),geometry_reason=int(detail["reasons"][i,0]))
    variants=[dict(id=f"articulated-{gravity}-{steps}",board_mode="articulated",gravity_mode=gravity,substeps=steps)
        for gravity,steps in [("legacy",8)]+[("passive_only",k) for k in a.substeps]]
    spec=dict(version=VERSION,source_id=SOURCE_ID,source=source,scene=contract["scene"],seed=SEED,
        plan=plan,noise_std=NOISE.tolist(),variants=variants,reset_repeats=2,
        prior_records_sha256=hashlib.sha256((a.prior_study/"shots.jsonl").read_bytes()).hexdigest(),
        source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob("*.py")},
        selection="Nominal configuration unchanged from historical source; all fresh inputs fixed before native outcomes. Same audited branches per protocol. One reset and its reversed-slot rerun; repetitions are paired, not200 independent jitters.",
        scope="Simulation-only local contact confirmation; no new model training, all-table coverage, transit or hardware claim")
    spec["fingerprint"]=digest(spec)
    a.output.mkdir(parents=True,exist_ok=False);atomic(a.output/"contract.json",spec);atomic(a.output/"plan.json",plan)
    report=dict(status="running",version=VERSION,source=source,contract_fingerprint=spec["fingerprint"],
        groups=[],records=0,scope=spec["scope"])
    atomic(a.output/"report.json",report)
    if not eligible.all():
        report.update(status="blocked_geometry",geometry_rejects=int((~eligible).sum()))
        atomic(a.output/"report.json",report)
        raise RuntimeError("Paired contact confirmation requires qualified endpoints; rejects retained in plan, no false release")
    start=time.monotonic();arrays_by_variant={};rows_by_variant={}
    with gpu_slot(a.gpu_lock,timeout_s=1800,notify=lambda x:print(json.dumps(x),flush=True)):
        for variant in variants:
            env=ContactEnvironment(contract["scene"],a.assets,n=n,bounds=WIDE_BOUNDS,
                substeps=variant["substeps"],board_mode="articulated",gravity_mode=variant["gravity_mode"])
            records=[];reset_arrays=[];reset_rows=[]
            for reset in range(2):
                order=np.arange(n) if reset==0 else np.arange(n-1,-1,-1)
                rows=run_bank_rows(env,xy[order],values[order],q10[order],q14[order],eligible[order],record=True)
                arrays={k:v[:,np.argsort(order)] for k,v in env.last_arrays.items()}
                rows=[rows[int(j)] for j in np.argsort(order)]
                np.savez_compressed(a.output/f'{variant["id"]}-reset-{reset}.npz',**arrays,
                    q10=q10,q14=q14,eligible=eligible,requested_values=values)
                for i,(p,r) in enumerate(zip(plan,rows)):
                    r.pop("trajectory_m",None)
                    r.update(id=f'{variant["id"]}-reset-{reset}-{p["id"]}',source_plan_id=p["id"],
                        family=p["family"],variant=variant["id"],reset=reset,
                        ballistic_velocity_at_350ms_m_s=arrays["velocity_m_s"][83,i].tolist(),
                        **flight_metrics(arrays,r,i))
                records.extend(rows);reset_rows.append(rows);reset_arrays.append(arrays)
            with (a.output/"results.jsonl").open("a") as f:
                for r in records:f.write(json.dumps(r,allow_nan=False)+"\n")
            families=[]
            for family in ("nominal","fresh_perturbation"):
                rr=[r for r in records if r["family"]==family]
                families.append(dict(family=family,attempts=len(rr),independent_requests=len(rr)//2,
                    captures=sum(r["route_success"] for r in rr),clean=sum(r["route_quality"]==2 for r in rr),
                    invalid=sum(bool(r["invalid"]) for r in rr),board_bounces=sum(r["board_bounce"] for r in rr),
                    maximum_integrated_arm_hinge_excursion_rad=max(r["max_integrated_joint_excursion_rad"] for r in rr)))
            group=dict(**variant,attempts=len(records),captures=sum(r["route_success"] for r in records),
                family_results=families,reset_order=paired_differences(*reset_arrays,*reset_rows))
            arrays_by_variant[variant["id"]]=reset_arrays[0]
            rows_by_variant[variant["id"]]=reset_rows[0]
            report["groups"].append(group);report["records"]+=len(records)
            report.update(elapsed_s=time.monotonic()-start)
            atomic(a.output/"report.json",report);print(json.dumps(group),flush=True)
            del env
            import gc,torch
            gc.collect();torch.cuda.empty_cache()
        report["singletons"]=[]
        if not a.skip_singletons:
            # Exact same audited q/geometry for a nominal and first fresh input,
            # one world at a time, separated from the fresh denominator above.
            for variant in [v for v in variants if v["gravity_mode"]=="passive_only"]:
                env=ContactEnvironment(contract["scene"],a.assets,n=1,bounds=WIDE_BOUNDS,
                    substeps=variant["substeps"],board_mode="articulated",gravity_mode="passive_only")
                for index in (0,a.nominal):
                    row=run_bank_rows(env,xy[index:index+1],values[index:index+1],q10[index:index+1],q14[index:index+1],eligible[index:index+1],record=True)[0]
                    raw=env.last_arrays
                    np.savez_compressed(a.output/f'{variant["id"]}-singleton-{plan[index]["id"]}.npz',**raw)
                    prior_arrays=arrays_by_variant[variant["id"]]
                    batch_arrays={k:v[:,index:index+1] for k,v in prior_arrays.items()}
                    differences=paired_differences(batch_arrays,raw,[rows_by_variant[variant["id"]][index]],[row])
                    row.pop("trajectory_m",None)
                    row.update(id=f'{variant["id"]}-singleton-{plan[index]["id"]}',source_plan_id=plan[index]["id"],
                        family="singleton_diagnostic",variant=variant["id"],reset="singleton",
                        **flight_metrics(raw,row,0))
                    with (a.output/"results.jsonl").open("a") as f:f.write(json.dumps(row,allow_nan=False)+"\n")
                    report["singletons"].append(dict(variant=variant["id"],source_plan_id=plan[index]["id"],
                        capture=row["route_success"],batch_comparison=differences))
                    report["records"]+=1
                    atomic(a.output/"report.json",report)
                del env
                gc.collect();torch.cuda.empty_cache()
    corrected=[v for v in variants if v["gravity_mode"]=="passive_only"]
    comparisons=[]
    for first,second in zip(corrected,corrected[1:]):
        a1,a2=arrays_by_variant[first["id"]],arrays_by_variant[second["id"]]
        delta=np.linalg.norm(a1["velocity_m_s"][83]-a2["velocity_m_s"][83],axis=1)
        pos=np.linalg.norm(a1["position_m"][83]-a2["position_m"][83],axis=1)
        comparisons.append(dict(first=first["id"],second=second["id"],
            ballistic_velocity_difference_at_350ms_max_m_s=float(delta.max()),
            ballistic_velocity_difference_at_350ms_mean_m_s=float(delta.mean()),
            ballistic_position_difference_at_350ms_max_m=float(pos.max())))
    report.update(status="completed",elapsed_s=time.monotonic()-start,timestep_comparisons=comparisons)
    atomic(a.output/"report.json",report)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ("prior-study","assets","output","gpu-lock"):p.add_argument("--"+key,type=Path,required=True)
    p.add_argument("--substeps",type=int,nargs="+",default=[32,64]);p.add_argument("--fresh",type=int,default=100);p.add_argument("--nominal",type=int,default=16)
    p.add_argument("--skip-singletons",action="store_true")
    a=p.parse_args()
    if not set(a.substeps)<={16,32,64} or len(a.substeps)!=len(set(a.substeps)) or not 20<=a.fresh<=110 or not 2<=a.nominal<=16 or a.fresh+a.nominal>128:p.error("Finite paired budgets required")
    a.substeps=sorted(a.substeps)
    run(a)


if __name__=="__main__":main()
