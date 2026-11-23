"""Frozen alternative arm branches and translated shots for two known misses.

No successful parent request is edited. New poses require fresh endpoint/path
proofs; a clearance prediction is a ranking signal, never a capture label.
"""
from __future__ import annotations
import copy
import math
from pathlib import Path
from .benchmark_verify import digest, read, sha, contained
from .nearby_transfer_plan import audited_source, rows_at, physical_match, captured

VERSION = "sircus-link-clearance-study-v1"
CASE_IDS = ("parallel-board-miss-09", "parallel-board-miss-19")
SEARCH_RULE = dict(version="sircus-link-clearance-menu-v2", seeds_per_target=6,
    random_seeds_per_fixed_command=2, maximum_full_checks=96,
    starts_per_arm=16, iterations=180, cpu_workers=2, cpu_case_limit_s=900,
    proposal_limit_s=30, preparation_order="Round robin source/fixed/radial/lateral/twist; repeat cycles",
    timeout_recovery="Discard unfinished admission and reconstruct engine before next proposal",
    radial_offsets_m=[-.12,-.08,-.04,.04,.08,.12],
    lateral_offsets_m=[-.15,-.10,-.05,.05,.10,.15],
    twist_offsets_deg=[-20.,20.,-45.,45.,180.],
    minimum_joint_difference_rad=.05, joint_duplicate_tolerance_rad=.01, maximum_native_candidates_per_case=32,
    repeats_per_success=3, preservation_releases=9, maximum_total_native_releases=289,
    fixed_pose_priority="Distinct proximal joints 1-3 before wrist-only changes; both arms may change",
    selection="Reserve fixed-command branches and each seed/family, then precise arm/rim clearance rank",
    scope="Two known development positions; no training or live planner promotion")


def source_inputs(a):
    report,contract,binding=audited_source(a.source_study,a.source_audit,
        "sircus-nearby-transfer-study-v1",("preparation.json","rows.jsonl","source-bank.json"))
    ranking_file=Path(contract["source_paths"]["target_study"])/"contract.json"
    if sha(ranking_file)!=contract["source_bindings"]["target"]["contract.json"]:
        raise ValueError("Parent ranking-rule source changed")
    contract=dict(contract,flight_ranking=read(ranking_file)["flight_ranking"])
    plan=read(a.source_study/"preparation.json"); rows=rows_at(a.source_study)
    cases=[];bank=[];preserved=[]
    for parent in plan["cases"]:
        if parent["id"] in CASE_IDS:
            records=[]
            for request in parent["candidates"]:
                found=[r for r in rows if r.get("variant")=="search" and r.get("candidate_id")==request["id"]]
                if len(found)!=1 or not found[0]["native_release"] or found[0]["invalid_physics"] or captured(found[0]):
                    raise ValueError("Expected valid prior released noncaptures at both targets")
                if found[0]["experimental_route"]["robot_contact_steps"]<=0:raise ValueError("Expected measured robot contact")
                physical_match(request,found[0]); row=found[0]
                if sha(contained(a.source_study/row["adapter_directory"],row["native_cache_file"]))!=row["native_cache_sha256"]:
                    raise ValueError("Source trace bytes changed")
                entry=dict(id=request["id"],source_request=copy.deepcopy(request),case_id=parent["id"],
                    source_directory=row["adapter_directory"],source_row_id=row["id"],source_row_sha256=digest(row),
                    source_trace_file=row["native_cache_file"],source_trace_sha256=row["native_cache_sha256"],
                    strength="prior_robot_contact",source_binding=binding,prediction=parent["predictions"][request["id"]])
                records.append(entry);bank.append(entry)
            original=copy.deepcopy(parent["original"])
            cases.append(dict(id=parent["id"],original=original,source_directory=parent["directory"],
                sources=records,historical=parent["historical"]))
        else:
            choices=[]
            for request in parent["candidates"]:
                found=[r for r in rows if r.get("variant")=="repeat" and r.get("candidate_id")==request["id"]]
                if len(found)==3 and all(captured(r) for r in found):
                    for row in found:physical_match(request,row)
                    choices.append((request,found))
            if not choices:raise ValueError("Previously recovered position lost confirmed source")
            request,found=choices[0]
            preserved.append(dict(id=parent["id"],original=copy.deepcopy(request),source_directory=parent["directory"],
                directory="preserved-"+parent["id"][-2:],source_row_ids=[r["id"] for r in found],source_row_sha256=[digest(r) for r in found]))
    if tuple(c["id"] for c in cases)!=CASE_IDS or len(preserved)!=3 or len(bank)!=40:
        raise ValueError("Expected two unresolved targets, 40 failed sources and three preserved positions")
    return cases,bank,preserved,contract,binding


def _score(row):
    value=row.get("score_m")
    return -float(value) if type(value) in (int,float) and math.isfinite(value) else math.inf


def seed_selection(case):
    """Use diverse prior commands, not only tiny changes to one recipe."""
    ordered=sorted(case["sources"],key=lambda s:(_score(s["prediction"]),s["id"]))
    picked=[];cells=set()
    for source in ordered:
        r=source["source_request"];v=r["values"]
        cell=tuple(round(x/width) for x,width in zip(v,[.05,.05,.04,5.,.06]))
        if cell not in cells: picked.append(source);cells.add(cell)
        if len(picked)==SEARCH_RULE["seeds_per_target"]:return picked
    ids={s["id"] for s in picked}
    return (picked+[s for s in ordered if s["id"] not in ids])[:SEARCH_RULE["seeds_per_target"]]


def proposal_menu(case,low,high):
    result=[]
    for seed_index,seed in enumerate(seed_selection(case)):
        source=seed["source_request"];old=source["values"];target=source["bucket_xy_m"]
        delta=[target[i]-old[i] for i in (0,1)];dist=math.hypot(*delta)
        forward=[x/dist for x in delta];side=[-forward[1],forward[0]]
        def add(radial,lateral,twist,family,random_index=0):
            values=list(old);values[:2]=[old[i]+radial*forward[i]+lateral*side[i] for i in (0,1)]
            if any(v<lo or v>hi for v,lo,hi in zip(values,low,high,strict=True)) or math.dist(values[:2],target)<=.15:return
            angle=math.remainder(source.get("board_twist_deg",0.)+twist,360.)
            result.append(dict(id=f"{case['id']}-clearance-{len(result):04d}",context_id=case["id"],
                values=values,bucket_xy_m=list(target),board_heading_offset_deg=source.get("board_heading_offset_deg",0.),
                board_twist_deg=angle,seed_q10=list(source["selected_q10"]),seed_q14=list(source["selected_q14"]),
                phase="search",role="diagnostic",method="link_clearance",intended_route=source.get("intended_route","direct"),
                group=family,proposal_family=family,source_request_id=source["id"],source_bank_id=seed["id"],
                source_strength=seed["strength"],source_distance_m=0.,seed_index=seed_index,
                radial_offset_m=radial,lateral_offset_m=lateral,twist_delta_deg=twist,
                branch_seed=2026112307+int(case["id"][-2:])*100+seed_index*10+random_index,
                fixed_cartesian=family=="fixed_cartesian"))
        for k in range(2):add(0.,0.,0.,"fixed_cartesian",k)
        for value in SEARCH_RULE["radial_offsets_m"]:add(value,0.,0.,"radial_shift")
        for value in SEARCH_RULE["lateral_offsets_m"]:add(0.,value,0.,"lateral_shift")
        for value in SEARCH_RULE["twist_offsets_deg"]:add(0.,0.,value,"board_twist")
    return result


def preparation_selection(proposals,nominal,wrist):
    valid=[p for p in proposals if wrist[p["id"]]["passed"]]
    groups=[]
    for seed in sorted({p["seed_index"] for p in valid}):
        for family in ("fixed_cartesian","radial_shift","lateral_shift","board_twist"):
            groups.append(sorted((p for p in valid if p["seed_index"]==seed and p["proposal_family"]==family),
                key=lambda p:(_score(nominal[p["id"]]),p["id"])))
    chosen=[]
    for i in range(max(map(len,groups),default=0)):chosen.extend(g[i] for g in groups if i<len(g))
    return chosen[:SEARCH_RULE["maximum_full_checks"]]


def joint_change(request,source):
    differences={role:[abs(math.remainder(x-y,2*math.pi)) for x,y in zip(request["selected_"+role],source["selected_"+role],strict=True)] for role in ("q10","q14")}
    return dict(proximal_rad=max(differences["q10"][:3]+differences["q14"][:3]),
        maximum_rad=max(differences["q10"]+differences["q14"]),per_arm_rad=differences)


def equivalent_commands(a,b):
    if a["bucket_xy_m"]!=b["bucket_xy_m"] or any(abs(x-y)>1e-8 for x,y in zip(a["values"],b["values"],strict=True)):return False
    if any(abs(math.remainder(a[k]-b[k],360.))>1e-8 for k in ("board_heading_offset_deg","board_twist_deg")):return False
    return all(abs(math.remainder(x-y,2*math.pi))<SEARCH_RULE["joint_duplicate_tolerance_rad"]
        for role in ("q10","q14") for x,y in zip(a["selected_"+role],b["selected_"+role],strict=True))


def command_key(request):
    return digest([request["bucket_xy_m"],request["values"],request["board_heading_offset_deg"],request["board_twist_deg"],
                   request["selected_q10"],request["selected_q14"]])


def candidate_selection(rows,predictions,sources,limit=32):
    """Keep alternative joint solutions; do not deduplicate just Cartesian poses."""
    valid=[]
    for row in rows:
        change=joint_change(row,sources[row["source_request_id"]])
        if row["fixed_cartesian"] and change["maximum_rad"]<SEARCH_RULE["minimum_joint_difference_rad"]:continue
        valid.append(dict(row,joint_change=change))
    rank=lambda r:(_score(predictions[r["id"]]),-r["joint_change"]["proximal_rad"],r["id"])
    fixed=sorted((r for r in valid if r["fixed_cartesian"]),key=lambda r:(-max(r["joint_change"]["per_arm_rad"]["q14"][:3]),-max(r["joint_change"]["per_arm_rad"]["q10"][:3]),*rank(r)))
    reserve=fixed[:8]
    for seed in sorted({r["source_bank_id"] for r in valid}):
        for family in ("radial_shift","lateral_shift","board_twist"):
            group=[r for r in valid if r["source_bank_id"]==seed and r["proposal_family"]==family]
            if group:reserve.append(min(group,key=rank))
    selected=[];seen=set()
    for row in reserve+sorted(valid,key=rank):
        key=command_key(row)
        if key in seen or any(equivalent_commands(row,old) for old in selected):continue
        seen.add(key);selected.append(row)
        if len(selected)==limit:break
    return selected


def branch_solver(implementation,branch_seed):
    """Seed only the outer candidate IK; preserve nested reference-search rules."""
    def solve(targets,initial,*,starts,iterations,**kwargs):
        # GeometryResolver's reference search explicitly owns preserve_known
        # (and its own seed/budget). Do not override or duplicate its keywords.
        if "preserve_known" in kwargs:
            return implementation(targets,initial,starts=starts,iterations=iterations,**kwargs)
        return implementation(targets,initial,starts=SEARCH_RULE["starts_per_arm"],iterations=SEARCH_RULE["iterations"],
            seed=branch_seed,preserve_known=[True]*len(targets),**kwargs)
    return solve


def prepare_alternatives(engine,proposal):
    """Preserve the active resolver/path rules while exploring both joint branches."""
    import numpy as np
    from .fixed35_broad_native import distinct_indices
    from .fixed35_bouncer_native import wrapped_difference
    resolver=engine.adapter.resolver;previous=resolver._branches
    arms=(resolver.solver.ik.dropper,resolver.solver.ik.board)
    original_solve=[a.solve_candidates for a in arms]
    hint={"q10":np.array(proposal["seed_q10"]),"q14":np.array(proposal["seed_q14"])}
    def branches(first,second,hint_unused,**kwargs):
        pairs=[(i,j) for i in distinct_indices(first) for j in distinct_indices(second)]
        def key(pair):
            i,j=pair;d10=wrapped_difference(first["q"][i],hint["q10"]);d14=wrapped_difference(second["q"][j],hint["q14"])
            return (-float(d14[:3].max()),-float(d10[:3].max()),-float(max(d10.max(),d14.max())),
                    float(first["residual"][i]+second["residual"][j]),i,j)
        return sorted(pairs,key=key)
    try:
        resolver._branches=branches
        for arm,implementation in zip(arms,original_solve,strict=True):
            arm.solve_candidates=branch_solver(implementation,proposal["branch_seed"])
        return engine.prepare([proposal],one_per_setting=False)
    finally:
        resolver._branches=previous
        for arm,implementation in zip(arms,original_solve,strict=True):arm.solve_candidates=implementation
