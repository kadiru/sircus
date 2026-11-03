"""Finite corrected-contact35-cell development map; no PPO or policy deployment."""

import argparse
from collections import Counter
import datetime
import importlib.metadata
import json
from pathlib import Path
import shutil
import time

import numpy as np

from .corrected_contract import (VERSION,CATALOGUE_VERSION,PHYSICS,SEED,NOISE,sha,
    grid_layouts,split_coordinates,geometry_hint,diverse_selection,reject_row,qualification,load_fingerprinted,validate_packages)
from .contact_study import atomic
from .expanded_actions import WIDE_BOUNDS
from .learning_contract import digest

MIN_FREE_BYTES=512*1024**2
MAX_OUTPUT_BYTES=512*1024**2


def storage_check(output):
    free=shutil.disk_usage(output).free
    used=sum(p.stat().st_size for p in output.rglob("*") if p.is_file())
    if free<MIN_FREE_BYTES or used>MAX_OUTPUT_BYTES:
        raise RuntimeError(f"storage_budget: free={free}, study_bytes={used}; no other files deleted")
    return dict(free_bytes=free,study_bytes=used)


def source_manifest(repository):
    root=Path(repository)
    paths=list((root/"sircus/mjlab").glob("*.py"))
    paths.append(root/"sircus/mjlab/requirements.txt")
    paths += [root/"sircus/isaac"/name for name in ("geometry.py","rl_contract.py","entry_quality.py","board_route.py")]
    return {str(p.relative_to(root)):sha(p) for p in sorted(paths)}


def prepare_inputs(a):
    from .learning_solve import required_asset_paths
    parent=load_fingerprinted(a.parent_study/"contract.json")
    bank=load_fingerprinted(a.bank,"sha256")
    if len(bank.get("templates",[]))!=17:raise RuntimeError("Expected preserved17-template geometry bank")
    reference=json.loads(a.contact_reference.read_text())
    if reference.get("status")!="completed" or not any(g.get("id")=="articulated-passive_only-64" for g in reference.get("groups",[])):
        raise RuntimeError("Completed corrected64 contact reference required")
    reference_contract=load_fingerprinted(a.contact_reference.parent/"contract.json")
    if reference_contract["fingerprint"]!=reference.get("contract_fingerprint") or reference_contract["scene"]!=parent["scene"]:
        raise RuntimeError("Corrected reference belongs to another contract/scene")
    for name in ("contact_env.py","contact_model.py","model.py","learning_score.py","contract.py"):
        if reference_contract.get("source_sha256",{}).get(name)!=sha(Path(__file__).parent/name):
            raise RuntimeError("Corrected reference native/scoring source differs: "+name)
    hints=[];provenance=[];excluded=[]
    for study in a.source_study:
        path=study/"shots.jsonl"
        if not path.is_file():raise RuntimeError("Source study lacks recorded geometry: "+str(study.name))
        provenance.append(dict(namespace=study.name,records_sha256=sha(path)))
        for line in path.read_text().splitlines():
            if not line.strip():continue
            row=json.loads(line)
            xy=row.get("bucket_xy_m")
            if isinstance(xy,list) and len(xy)==2:excluded.append(xy)
            hint=geometry_hint(row,study.name)
            if hint is not None:hints.append(hint)
    templates=[]
    for template in bank["templates"]:
        candidates=[h for h in hints if h["source_id"]==template["source_trial_id"]]
        if template.get("source_record_sha256"):
            candidates=[h for h in candidates if h["source_record_sha256"]==template["source_record_sha256"]]
        templates.append(candidates[0] if candidates else None)
    # Disambiguated original teacher requests supply seeds only, never outcome labels.
    seed_hints=[x for x in templates if x is not None]
    seed_hints += [h for h in hints if h["source_id"]=="stable-shot-0004711"][:1]
    if not seed_hints:raise RuntimeError("No source joint geometry hints were found")
    for key in ("train_layouts","validation_layouts","test_layouts"):
        excluded.extend(parent.get(key,[]))
    historical_studies=[]
    for study in a.historical_study:
        files={}
        for name in ("contract.json","report.json","splits.json"):
            path=study/name
            if not path.is_file():continue
            files[name]=sha(path);data=json.loads(path.read_text())
            for key in ("train_layouts","validation_layouts","test_layouts","train","validation","test"):
                points=data.get(key)
                if isinstance(points,list) and all(isinstance(x,list) and len(x)==2 for x in points):excluded.extend(points)
        if "contract.json" not in files:raise RuntimeError("Historical split contract missing: "+study.name)
        historical_studies.append(dict(namespace=study.name,files_sha256=files))
    layouts=grid_layouts();excluded.extend(r["bucket_xy_m"] for r in layouts)
    excluded=sorted({tuple(map(float,x)) for x in excluded})
    assets={name:sha(a.assets.parent/name) for name in required_asset_paths(a.assets)}
    return dict(scene=parent["scene"],parent_fingerprint=parent["fingerprint"],bank=bank,
        bank_file_sha256=sha(a.bank),template_hints=templates,seed_hints=seed_hints,
        source_studies=provenance,historical_coordinates=[list(x) for x in excluded],
        historical_studies=historical_studies,
        asset_sha256=assets,reference_sha256=sha(a.contact_reference),
        reference_fingerprint=reference["contract_fingerprint"])


def proposal_plan(inputs,extra=64):
    from .bank_geometry import build_candidates
    from .expand_far import far_proposals
    scene,bank=inputs["scene"],inputs["bank"]
    layouts=grid_layouts();xy=np.asarray([r["bucket_xy_m"] for r in layouts])
    values,metadata=build_candidates(bank,xy,scene)
    plan=[]
    for cell_index,cell in enumerate(layouts):
        rng=np.random.default_rng(SEED+cell_index)
        additional,additional_meta=far_proposals(xy[cell_index],extra,rng,scene,seeds=()) if extra else (np.empty((0,5)),[])
        rows=np.r_[values[cell_index],additional]
        meta=[dict({k:v for k,v in m.items() if k!="qualification"},proposal_family="preserved-bank-geometry") for m in metadata]
        meta += [dict(candidate_id=f"new-{j:03d}",template_index=None,rotation_index=None,**m) for j,m in enumerate(additional_meta)]
        for j,(v,m) in enumerate(zip(rows,meta)):
            hint=inputs["template_hints"][m["template_index"]] if m["template_index"] is not None else None
            if hint is None:
                hint=min(inputs["seed_hints"],key=lambda h:float(np.linalg.norm((np.asarray(h["values"])-v)/[.74,.85,.48,52,.60])))
            plan.append(dict(id=f'{cell["id"]}-{m["candidate_id"]}',cell_id=cell["id"],cell_index=cell_index,
                candidate_id=m["candidate_id"],candidate_index=j,bucket_xy_m=cell["bucket_xy_m"],values=v.tolist(),
                proposal=m,geometry_hint=hint))
    return plan


def row_rank(row):
    return (bool(row.get("route_success")),int(row.get("route_quality",0)),
        float(row.get("minimum_mouth_clearance_mm") if row.get("minimum_mouth_clearance_mm") is not None else -1e9),
        -float(row.get("closest_opening_distance_m",2)),row["id"])


def home_layout_summary(checked):
    """Classify only actual negative bucket/base contacts as fixed penetration."""
    fixed=[]
    for contact in checked.get("margin_contacts",[]):
        pair=contact.get("pair",[])
        if len(pair)==2 and contact.get("distance_m",0)<0 and any("/link_base/" in n for n in pair) and any(n.startswith("bucket/") for n in pair):
            fixed.append(contact)
    category=("home_clear" if checked.get("passed") else "fixed_base_bucket_penetration" if fixed else
        "home_unresolved" if checked.get("reason")=="common_home_unresolved" else
        "staging_clearance_unproven" if checked.get("reason")=="unproven_clearance" else "staging_clearance_blocked")
    return dict(passed=bool(checked.get("passed")),reason=checked.get("reason"),category=category,
        minimum_clearance_m=checked.get("minimum_clearance_m"),closest_pair=checked.get("closest_pair"),
        fixed_base_penetrations=fixed,common_home_sha256=checked.get("common_home_sha256"))


def _run(a):
    from .corrected_geometry import CorrectedGeometry
    from .corrected_admission import TransitAdmission,VERSION as ADMISSION_VERSION
    from .corrected_geometry import VERSION as GEOMETRY_VERSION
    from .contract import ScoreStream,FORCE_NAMES
    inputs=prepare_inputs(a);scene=inputs["scene"];plan=proposal_plan(inputs,a.extra_proposals)
    packages=validate_packages({name:importlib.metadata.version(name) for name in
        ("mjlab","mujoco","mujoco-warp","torch","warp-lang","numpy","scipy","ml-dtypes")})
    admission=TransitAdmission(scene,a.assets)
    a.output.mkdir(parents=True,exist_ok=False)
    home_layouts=[]
    for cell in grid_layouts():
        checked=admission.check_home(cell["bucket_xy_m"])
        home_layouts.append(dict(id=cell["id"],bucket_xy_m=cell["bucket_xy_m"],
            home_geometry=home_layout_summary(checked),evidence=checked))
    atomic(a.output/"common-home-layouts.json",home_layouts)
    repository=Path(__file__).resolve().parents[2]
    spec=dict(version=VERSION,physics=PHYSICS,scene=scene,parent_fingerprint=inputs["parent_fingerprint"],
        bank_file_sha256=inputs["bank_file_sha256"],bank_geometry_fingerprint=inputs["bank"]["sha256"],
        contact_reference_sha256=inputs["reference_sha256"],contact_reference_fingerprint=inputs["reference_fingerprint"],
        source_studies=inputs["source_studies"],historical_studies=inputs["historical_studies"],
        common_home=admission.home,common_home_sha256=admission.home["sha256"],
        common_home_layouts_sha256=sha(a.output/"common-home-layouts.json"),
        geometry_version=GEOMETRY_VERSION,transit_version=ADMISSION_VERSION,
        grid_layouts=grid_layouts(),historical_coordinates=inputs["historical_coordinates"],
        future_splits=split_coordinates(inputs["historical_coordinates"]),
        future_split_scope="Frozen before current native outcomes and disjoint from supplied mjlab source/historical coordinates; grid is DEVELOPMENT then TRAIN; validation/test receive no outcome search here. No claim against every experiment ever run.",
        source_sha256=source_manifest(repository),asset_sha256=inputs["asset_sha256"],
        packages=packages,
        seed=SEED,noise_std=NOISE.tolist(),bounds=WIDE_BOUNDS,
        budgets=dict(bank_candidates_per_cell=272,extra_proposals_per_cell=a.extra_proposals,
            native_per_cell=a.native_per_cell,top=a.top,nominal_repeats=3,fresh_confirmations=20,required_fresh_captures=18,
            batch=a.batch,queue_timeout_s=a.queue_timeout,maximum_output_bytes=MAX_OUTPUT_BYTES,minimum_free_bytes=MIN_FREE_BYTES),
        strict_admission=dict(clearance_m=.005,self_collision="Nonadjacent enabled; only documented mechanical/grasp exclusions",carried_ball=True),
        scope="Finite35-cell spatial development search with corrected contact, strict endpoints and serialized geometric transit; no PPO, perception, hardware, actuator tracking or continuous table coverage claim")
    spec["fingerprint"]=digest(spec)
    atomic(a.output/"contract.json",spec)
    atomic(a.output/"plan.json",plan);atomic(a.output/"geometry-bank.json",inputs["bank"])
    atomic(a.output/"common-home.json",admission.home)
    report=dict(version=VERSION,status="running",phase="cpu_geometry",contract_fingerprint=spec["fingerprint"],no_training=True,
        physics=PHYSICS,records=0,geometry_total_requests=0,native_attempts=0,native_releases=0,cells=[dict(**c,geometry_requests=0,strict_eligible=0,
        geometry_rejected=0,skipped_eligible=0,search_attempts=0,search_captures=0,confirmed_sources=[]) for c in grid_layouts()],
        qualified_sources=[],transit_attempts=[],scope=spec["scope"])
    report["common_home_layouts_sha256"]=spec["common_home_layouts_sha256"]
    for cell,home in zip(report["cells"],home_layouts):cell["home_geometry"]=home["home_geometry"]
    started=time.monotonic();all_rows=[];packet=0;geometry_packet=0
    def publish(phase,**extra):
        report["counts"]=dict(grid_positions=35,geometry_requests=report["geometry_total_requests"],
            endpoint_eligible=sum(c["strict_eligible"] for c in report["cells"]),native_attempts=report["native_attempts"],
            captures=sum(bool(r.get("native_release") and r.get("route_success")) for r in all_rows),
            confirmation_attempts=sum(r["phase"]=="confirmation" for r in all_rows),
            common_home_connected=len({r["cell_id"] for r in report["transit_attempts"] if r["passed"]}),
            home_clear_positions=sum(c["home_geometry"]["passed"] for c in report["cells"]),
            qualified_positions=len({r["source"]["cell_id"] for r in report["qualified_sources"]}))
        for cell in report["cells"]:
            cell["geometry"]=dict(done=cell["geometry_requests"]>0,requests=cell["geometry_requests"],
                eligible=cell["strict_eligible"] if cell["geometry_requests"] else None,
                rejected=cell["geometry_rejected"] if cell["geometry_requests"] else None,not_selected=cell["skipped_eligible"])
            cell["search"]=dict(attempts=cell["search_attempts"],captures=cell["search_captures"],invalid=0)
            cell["confirmations"]=[dict(source_id=c["source"]["id"],nominal=dict(attempts=c["nominal_attempts"],captures=c["nominal_captures"]),
                fresh=dict(attempts=c["fresh_attempts"],captures=c["fresh_captures"],invalid=c.get("fresh_invalid",0)),
                motion=dict(scope=c["transit_scope"],file=c["transit_file"]),qualified=c["qualified"]) for c in cell["confirmed_sources"]]
            cell["status"]=("qualified" if any(x["qualified"] for x in cell["confirmed_sources"]) else
                "confirmation_failed" if cell["confirmed_sources"] else "native_capture" if cell["search_captures"] else
                "no_capture_in_budget" if cell["search_attempts"] else "ready_search" if cell["strict_eligible"] else
                "endpoint_unresolved" if cell["geometry_requests"] else "pending_geometry")
        report.update(phase=phase,elapsed_s=time.monotonic()-started,
            timestamp_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),**extra)
        atomic(a.output/"report.json",report)
        print(json.dumps(dict(phase=phase,records=report["records"],native_attempts=report["native_attempts"],**extra)),flush=True)
    def write_rows(rows):
        with (a.output/"shots.jsonl").open("a") as f:
            for row in rows:f.write(json.dumps(row,allow_nan=False)+"\n")
        all_rows.extend(rows);report["records"]+=len(rows)
    publish("cpu_geometry")
    gate=CorrectedGeometry(scene,a.assets,starts=16,iterations=180)
    def audit(requests,phase):
        nonlocal geometry_packet
        n=len(requests);bucket=np.asarray([p["bucket_xy_m"] for p in requests]);values=np.asarray([p["values"] for p in requests])
        hints=[p["geometry_hint"] for p in requests]
        mask,detail=gate.audit_values(bucket,values[:,None],bounds=WIDE_BOUNDS,
            known_q10=np.asarray([h["q10"] for h in hints]),known_q14=np.asarray([h["q14"] for h in hints]),
            known_values=np.asarray([h["values"] for h in hints]),known_bucket_xy=np.asarray([h["bucket_xy_m"] for h in hints]))
        cache=f"endpoint-batch-{geometry_packet:04d}.npz";geometry_packet+=1
        arrays={k:v for k,v in detail.items() if isinstance(v,np.ndarray)}
        np.savez_compressed(a.output/cache,bucket=bucket,values=values,mask=mask,record_ids=np.asarray([p["id"] for p in requests]),**arrays)
        result=[]
        for i,p in enumerate(requests):
            q10,q14=detail["q10"][i,0],detail["q14"][i,0]
            result.append(dict(**p,q10=q10.tolist(),q14=q14.tolist(),geometry_eligible=bool(mask[i,0]),
                geometry_reason=int(detail["reasons"][i,0]),geometry_cache_file=cache,geometry_cache_row=i,
                geometry_version=GEOMETRY_VERSION,phase=phase))
        storage_check(a.output)
        report["geometry_total_requests"]+=len(requests)
        return result
    candidates=[];native_requests=[]
    for cell_index,cell in enumerate(report["cells"]):
        requests=[p for p in plan if p["cell_index"]==cell_index]
        audited=[]
        for offset in range(0,len(requests),a.batch):audited.extend(audit(requests[offset:offset+a.batch],"search"))
        eligible=np.asarray([r["geometry_eligible"] for r in audited])
        chosen=set(diverse_selection([r["values"] for r in audited],eligible,[r["proposal"] for r in audited],a.native_per_cell,SEED+cell_index))
        candidates.extend(audited)
        cell.update(geometry_requests=len(audited),strict_eligible=int(eligible.sum()),geometry_rejected=int((~eligible).sum()),
            skipped_eligible=int(eligible.sum())-len(chosen),geometry_reasons=dict(Counter(str(r["geometry_reason"]) for r in audited)))
        for i,r in enumerate(audited):
            if i in chosen:native_requests.append(r)
            else:
                row=reject_row(r,"strict_endpoint_rejected" if not eligible[i] else None,phase="search",
                    geometry_cache_file=r["geometry_cache_file"],geometry_cache_row=r["geometry_cache_row"])
                row.update(status="geometry_rejected" if not eligible[i] else "not_selected_budget",geometry_reason=r["geometry_reason"],
                    q10=r["q10"],q14=r["q14"],geometry_eligible=bool(eligible[i]))
                write_rows([row])
        publish("cpu_geometry",completed_cells=cell_index+1)
    atomic(a.output/"audited-plan.json",candidates)
    atomic(a.output/"selected-search-plan.json",native_requests)
    if a.cpu_only:
        publish("cpu_prepared",status="cpu_prepared");return
    from .gpu_slot import gpu_slot
    from .contact_env import ContactEnvironment
    from .bank_study import run_bank_rows
    def waiting(state):publish("waiting_for_4090",status="waiting",queue=state)
    def execute(env,requests,phase,audited=False,motion_source=None):
        nonlocal packet
        result=[]
        for offset in range(0,len(requests),a.batch):
            storage_check(a.output)
            rr=requests[offset:offset+a.batch]
            if not audited:rr=audit(rr,phase)
            xy=np.asarray([r["bucket_xy_m"] for r in rr]);values=np.asarray([r["values"] for r in rr])
            eligible=np.asarray([r["geometry_eligible"] for r in rr]);q10=np.asarray([r["q10"] for r in rr]);q14=np.asarray([r["q14"] for r in rr])
            if motion_source is not None:
                for i,r in enumerate(rr):
                    if not eligible[i]:continue
                    proof=admission.connect_variation(motion_source,r)
                    path=f'variation-{r["id"]}.json';atomic(a.output/path,proof)
                    r.update(motion_admitted=bool(proof.get("passed")),motion_admission_file=path,motion_reason=proof.get("reason"))
                    eligible[i]&=bool(proof.get("passed"))
            rows=run_bank_rows(env,xy,values,q10,q14,eligible,record=True)
            positions=np.flatnonzero(eligible)
            if any(not rows[int(i)].get("native_release") for i in positions):
                raise RuntimeError("Native endpoint gate disagreed with prior strict CPU admission")
            path=f"native-batch-{packet:04d}.npz";packet+=1
            if len(positions):
                arrays=env.last_arrays
                np.savez_compressed(a.output/path,**arrays,record_ids=np.asarray([rr[i]["id"] for i in positions]))
                for j,i in enumerate(positions):
                    scalar=ScoreStream(scene,rr[i]["bucket_xy_m"],rows[i]["board_normal"])
                    for p,v,f in zip(arrays["position_m"][:,j],arrays["velocity_m_s"][:,j],arrays["forces_N"][:,j]):scalar.step(p,v,dict(zip(FORCE_NAMES,f)))
                    scored=scalar.result()
                    for key in ("success","route_success","route_quality","complete_success","board_bounce","whole_ball_entry","board_contact_episodes","external_contact_steps","single_bounce_success","physics_steps"):
                        if scored[key]!=rows[i][key]:raise RuntimeError("Corrected scalar scorer mismatch: "+rr[i]["id"]+"/"+key)
                    rows[i].update(native_cache_file=path,native_cache_row=j,scalar_score_audit_passed=True)
            for r,p in zip(rows,rr):
                r.pop("trajectory_m",None)
                r.update(id=p["id"],cell_id=p["cell_id"],candidate_id=p["candidate_id"],phase=phase,
                    values=p["values"],bucket_xy_m=p["bucket_xy_m"],q10=p["q10"],q14=p["q14"],
                    geometry_cache_file=p["geometry_cache_file"],geometry_cache_row=p["geometry_cache_row"],
                    geometry_eligible=p["geometry_eligible"],geometry_reason=p["geometry_reason"],
                    status="native_observed" if r.get("native_release") else "geometry_rejected",
                    physics_version=PHYSICS["version"])
                if "confirmation_source_id" in p:r["confirmation_source_id"]=p["confirmation_source_id"]
                for key in ("motion_admitted","motion_admission_file","motion_reason"):
                    if key in p:r[key]=p[key]
                if p.get("geometry_eligible") and p.get("motion_admitted") is False:
                    r.update(invalid="variation_common_home_path_rejected",status="motion_rejected")
            write_rows(rows);result.extend(rows);report["native_attempts"]+=len(rows)
            report["native_releases"]+=sum(bool(r.get("native_release")) for r in rows)
            publish(phase,status="running")
        return result
    try:
        with gpu_slot(a.gpu_lock,timeout_s=a.queue_timeout,notify=waiting):
            storage_check(a.output)
            env=ContactEnvironment(scene,a.assets,n=a.batch,bounds=WIDE_BOUNDS,substeps=64,board_mode="articulated",gravity_mode="passive_only")
            if abs(env.sim.mj_model.opt.timestep-PHYSICS["native_dt_s"])>1e-15:raise RuntimeError("Native timestep differs from corrected contract")
            import mujoco
            effective=env.sim.mj_model.opt
            if (effective.solver!=mujoco.mjtSolver.mjSOL_NEWTON or effective.cone!=mujoco.mjtCone.mjCONE_PYRAMIDAL or
                effective.integrator!=mujoco.mjtIntegrator.mjINT_IMPLICITFAST or effective.iterations!=100 or abs(effective.tolerance-1e-8)>1e-16):
                raise RuntimeError("Actual solver settings differ from corrected contract")
            if not np.allclose(effective.gravity,[0,0,-9.81],atol=1e-12,rtol=0):raise RuntimeError("Actual native gravity differs from corrected contract")
            report["effective_native_options"]=dict(timestep=float(effective.timestep),solver=int(effective.solver),cone=int(effective.cone),
                integrator=int(effective.integrator),iterations=int(effective.iterations),tolerance=float(effective.tolerance),gravity=np.asarray(effective.gravity).tolist())
            publish("native_search",status="running")
            searched=execute(env,native_requests,"search",audited=True)
            for cell in report["cells"]:
                cell_rows=[r for r in searched if r["cell_id"]==cell["id"]]
                cell.update(search_attempts=len(cell_rows),search_captures=sum(r["route_success"] for r in cell_rows))
                captured=sorted([r for r in cell_rows if r["route_success"]],key=row_rank,reverse=True)[:a.top]
                for rank,source in enumerate(captured):
                    transit=admission.admit(source)
                    transit_id=f'{cell["id"]}-source-{rank:02d}'
                    transit_file=f"transit-{transit_id}.json";atomic(a.output/transit_file,transit)
                    report["transit_attempts"].append(dict(source_id=source["id"],cell_id=cell["id"],passed=transit.get("passed",False),
                        scope=transit.get("scope"),reason=transit.get("reason"),file=transit_file))
                    if not transit.get("passed"):continue
                    hint=geometry_hint(source,"corrected_search")
                    def request(value,phase,index):return dict(id=f'{source["id"]}-{phase}-{index:02d}',cell_id=cell["id"],
                        candidate_id=source["candidate_id"],bucket_xy_m=cell["bucket_xy_m"],values=value,
                        geometry_hint=hint,confirmation_source_id=source["id"])
                    nominal_plan=[request(source["values"],"nominal",i) for i in range(3)]
                    nominal=execute(env,nominal_plan,"nominal",motion_source=source)
                    noise_seed=SEED+1000+35*cell["row"]+cell["column"]*2+rank
                    rng=np.random.default_rng(noise_seed)
                    fresh_plan=[request((np.asarray(source["values"])+rng.normal(0,NOISE)).tolist(),"confirmation",i) for i in range(20)]
                    atomic(a.output/f"confirmation-plan-{transit_id}.json",dict(source_id=source["id"],noise_seed=noise_seed,nominal=nominal_plan,fresh=fresh_plan))
                    fresh=execute(env,fresh_plan,"confirmation",motion_source=source)
                    qualified=qualification(nominal,fresh,transit)
                    entry=dict(source=source,transit_file=transit_file,transit_scope=transit.get("scope"),noise_seed=noise_seed,
                        nominal_ids=[r["id"] for r in nominal],fresh_ids=[r["id"] for r in fresh],
                        nominal_attempts=3,nominal_captures=sum(r["route_success"] for r in nominal),
                        fresh_attempts=20,fresh_captures=sum(r["route_success"] for r in fresh),qualified=qualified,
                        fresh_invalid=sum(bool(r["invalid"]) for r in fresh),
                        physics_version=PHYSICS["version"])
                    cell["confirmed_sources"].append(entry)
                    if qualified:report["qualified_sources"].append(entry)
                    publish("fresh_confirmation",status="running")
            catalogue=dict(version=CATALOGUE_VERSION,contract_fingerprint=spec["fingerprint"],physics=PHYSICS,
                sources=report["qualified_sources"],qualification="Strict source endpoint/common-home transit,3/3nominal and>=18/20fresh64; no historical success inherited")
            catalogue["fingerprint"]=digest(catalogue);atomic(a.output/"confirmed-catalogue.json",catalogue)
            report["catalogue_fingerprint"]=catalogue["fingerprint"]
            report["catalogue_sha256"]=sha(a.output/"confirmed-catalogue.json")
            report["cells_with_native_capture"]=sum(c["search_captures"]>0 for c in report["cells"])
            report["cells_with_qualified_source"]=sum(any(s["qualified"] for s in c["confirmed_sources"]) for c in report["cells"])
            report["evidence_sha256"]={p.name:sha(p) for p in a.output.iterdir() if p.is_file() and p.name!="report.json"}
            publish("completed",status="completed")
    except Exception as exc:
        publish("failed",status="failed",error=str(exc));raise


def run(a):
    try:return _run(a)
    except Exception as exc:
        path=a.output/"report.json"
        if path.is_file():
            report=json.loads(path.read_text());report.update(status="failed",phase="failed",error=str(exc))
            atomic(path,report)
        raise


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ("parent-study","bank","contact-reference","assets","output","gpu-lock"):p.add_argument("--"+key,type=Path,required=True)
    p.add_argument("--source-study",type=Path,action="append",required=True)
    p.add_argument("--historical-study",type=Path,action="append",default=[])
    p.add_argument("--batch",type=int,default=128);p.add_argument("--extra-proposals",type=int,default=64)
    p.add_argument("--native-per-cell",type=int,default=64);p.add_argument("--top",type=int,default=2)
    p.add_argument("--queue-timeout",type=float,default=21600);p.add_argument("--cpu-only",action="store_true")
    a=p.parse_args()
    if not 1<=a.batch<=128 or not 0<=a.extra_proposals<=256 or not 1<=a.native_per_cell<=128 or not 1<=a.top<=2 or not 1<=a.queue_timeout<=21600:p.error("Bounded study budget required")
    run(a)


if __name__=="__main__":main()
