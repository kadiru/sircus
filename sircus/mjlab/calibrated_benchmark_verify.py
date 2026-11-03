"""Independent CPU audit for the calibrated v2 study; no physics or hardware.

This auditor reconstructs layouts, geometry-resolution bindings, exact requested
commands, direct frozen predictions and CPU PPO updates, then independently
scores every dense native trace. It never rewrites the source study.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np

from . import benchmark_verify as common
from .calibrated_benchmark_contract import (
    VERSION as STUDY_VERSION, AUDIT_VERSION, SCORER_VERSION, REWARD_VERSION,
    CRITICAL, PRIOR_PATH, V1_BASELINE, SPLIT_SEED, SEEDS, BUDGET,
)
from .contract import FORCE_NAMES
from .corrected_contract import PHYSICS, NOISE, sha
from .expanded_actions import WIDE_BOUNDS
from .learning_contract import digest

VERSION = AUDIT_VERSION
Checks, EvidenceError = common.Checks, common.EvidenceError
read, lines, contained = common.read, common.lines, common.contained
fingerprint, check_manifest = common.fingerprint, common.check_manifest
RESIDUAL = np.asarray([.04, .04, .05, 8., .12])
MILESTONES = [0, 256, 512, 1024]


def row_reward(row):
    """Independent declaration, not the producer's reward implementation."""
    if row["invalid"]:
        return -1.0
    shaped = .15 * float(row["board_bounce"]) + .5 * math.exp(-row["closest_opening_distance_m"] / .12)
    if row["external_contact_steps"]:
        return float(shaped - .5)
    return float(shaped + .5 * min(row["max_retention_s"], 1.)
                 + 3 * row["route_success"] + .3 * (row["route_quality"] == 2)
                 + .1 * row["single_bounce_success"])


def counts(rows):
    return dict(attempts=len(rows),
                captures=sum(bool(r.get("native_release") and r.get("route_success") and not r.get("invalid")) for r in rows),
                native_releases=sum(bool(r.get("native_release")) for r in rows),
                invalid=sum(bool(r.get("invalid")) for r in rows),
                clean=sum(bool(r.get("native_release") and not r.get("invalid") and r.get("route_quality") == 2) for r in rows),
                mean_reward=float(np.mean([row_reward(r) for r in rows])) if rows else 0.)


def admission_counts(rows):
    rejected=[r for r in rows if not r.get("native_release")]
    keys={str(r.get("invalid") or "unspecified_setup_reject") for r in rejected}
    return dict(requested=len(rows),native_releases=len(rows)-len(rejected),rejected=len(rejected),
                reasons={key:sum(str(r.get("invalid") or "unspecified_setup_reject")==key for r in rejected) for key in keys},
                fixed_base_rejected=sum(bool(r.get("context_hard_reject")) for r in rows),
                anchor_unresolved=sum(not r.get("context_anchor_admitted") for r in rows),
                invalid_physics=sum(bool(r.get("invalid_physics")) for r in rows))


def stats_equal(actual, rows, checks, label):
    for key, expected in counts(rows).items():
        if isinstance(expected, float):
            checks.numeric(actual.get(key), expected, label + "/" + key, 1e-8)
        else:
            checks.equal(actual.get(key), expected, label + "/" + key)


def expected_schedule(seed, budget):
    rng, result = np.random.default_rng(seed), []
    while len(result) < budget:
        result.extend(rng.permutation(35).tolist())
    return result[:budget]


def expected_noise(context_id, repeat):
    if repeat == 0:
        return np.zeros(5)
    seed = int(hashlib.sha256(f"{SPLIT_SEED}:{context_id}:{repeat}".encode()).hexdigest()[:16], 16)
    return np.random.default_rng(seed).normal(0, NOISE)


def expected_observation(bucket, anchor):
    low, high = np.asarray(WIDE_BOUNDS["low"]), np.asarray(WIDE_BOUNDS["high"])
    return np.r_[(np.asarray(bucket)-[.67,-.20])/[.57,.45],
                 (np.asarray(anchor["bucket_xy_m"])-[.67,-.20])/[.57,.45],
                 2*(np.asarray(anchor["values"])-low)/(high-low)-1].astype(np.float32)


def expected_splits(repository):
    historical = read(Path(repository) / V1_BASELINE / "splits.json")
    excluded = [c["bucket_xy_m"] for role in ("train", "validation", "test") for c in historical[role]]
    if len(excluded) != 71 or len({tuple(c) for c in excluded}) != 71:
        raise EvidenceError("Published historical coordinate exclusion changed")
    used = np.asarray(excluded, float)
    groups = {"train": [dict(id=f"dev-grid-{r:02d}-{c:02d}", bucket_xy_m=[float(x),float(y)])
                        for r,y in enumerate(np.linspace(-.65,.25,5))
                        for c,x in enumerate(np.linspace(.1,1.24,7))]}
    for role, count, seed in (("validation",12,SPLIT_SEED),("test",24,SPLIT_SEED+1)):
        rng, rows = np.random.default_rng(seed), []
        for _ in range(100000):
            if len(rows) == count:
                break
            xy = rng.uniform([.1,-.65],[1.24,.25])
            if np.linalg.norm(used-xy,axis=1).min() < .005:
                continue
            rows.append(dict(id=f"calibrated-{role}-{len(rows):02d}", bucket_xy_m=xy.tolist()))
            used = np.vstack([used,xy])
        if len(rows) != count:
            raise EvidenceError("Fresh split reconstruction failed")
        groups[role] = rows
    return groups, excluded


def check_contract(study, contract, prior, ref, repository, checks):
    from .calibrated_admission import validate_home, anchor_search_spec, VERSION as RESOLVER_VERSION
    checks.equal(contract["version"], STUDY_VERSION, "Explicit calibrated study version")
    fingerprint(contract, checks, "Calibrated contract")
    p = contract["protocol"]
    synthetic = p.get("study_kind") == "synthetic_fixture"
    checks.require(p.get("study_kind") in {"synthetic_fixture", "native_current_ball_v2"}, "Declared evidence kind")
    if not synthetic:
        checks.equal(p["seeds"], SEEDS, "Fixed native seed set")
        checks.equal(p["sample_budget"], BUDGET, "Fixed native TRAIN request budget")
        checks.equal(p["milestones"], MILESTONES, "Fixed native validation milestones")
    checks.require(p["sample_budget"] >= 32 and p["sample_budget"] % 32 == 0, "Bounded whole-batch request budget")
    checks.equal(p["milestones"][0], 0, "Zero-budget validation retained")
    checks.equal(p["milestones"][-1], p["sample_budget"], "Last budget measured")
    checks.equal(p["batch_size"], 32, "Matched native batch size")
    checks.equal([m["id"] for m in p["methods"]], ["sweep","ppo"], "Both declared methods")
    checks.equal(p["evaluation"]["validation_layouts"],12,"Validation layout count")
    checks.equal(p["evaluation"]["test_layouts"],24,"Test layout count")
    checks.equal(p["evaluation"]["repeats"],3,"Paired repetitions")
    checks.numeric(p["residual_scale"],RESIDUAL,"Shared five-parameter residual bounds",0)
    checks.equal(p["action_bounds"],WIDE_BOUNDS,"Physical global bounds unchanged")
    checks.equal(p["reward_version"],REWARD_VERSION,"Versioned calibrated reward")
    checks.equal(p["scorer_version"],SCORER_VERSION,"Versioned calibrated scorer")
    checks.equal(p["ppo"],dict(network=[9,64,64,5],critic=[9,64,64,1],optimizer="CPU Adam",lr=.0003,
                 epochs=6,clip=.2,value_weight=.5,entropy_weight=.001,gradient_norm=1.,initial_logstd=-1.5,
                 logstd_bounds=[-3.,-.7],episode="One terminal shot decision; no GAE/bootstrap; actual native reward updates"),
                 "Frozen PPO architecture and optimizer settings")
    groups, excluded = expected_splits(repository)
    checks.equal(contract["historical_exclusion_count"],71,"Historical exclusion count")
    checks.equal(contract["historical_exclusion_sha256"],digest(excluded),"Historical exclusion identity")
    for role, expected in groups.items():
        actual = contract["splits"][role]
        checks.equal(len(actual),len(expected),"Frozen fresh split count/"+role)
        for a,b in zip(actual,expected):
            checks.equal(a["id"],b["id"],"Frozen fresh layout ID")
            checks.numeric(a["bucket_xy_m"],b["bucket_xy_m"],"Unconditional fresh coordinates",0)
    identity = contract["identity"]
    expected_sources = dict(ref["source_hashes"])
    for name in CRITICAL:
        relative = "sircus/mjlab/"+name
        expected_sources[relative] = sha(Path(repository)/relative)
    checks.equal(identity["source_sha256"],expected_sources,"All execution/calibration source bytes")
    checks.equal(identity["contract_fingerprint"],ref["contract"]["fingerprint"],"Reference engine contract")
    checks.equal(identity["scene_sha256"],digest(ref["scene"]),"Scene geometry")
    checks.equal(identity["asset_sha256"],ref["asset_hashes"],"Robot asset bytes")
    checks.equal(contract["packages"],ref["packages"],"Pinned runtime package versions")
    checks.equal(identity["reward_version"],REWARD_VERSION,"Identity reward version")
    checks.equal(identity["scorer_version"],SCORER_VERSION,"Identity scorer version")
    # Physics dynamics stay fixed; the scorer now uses every native interval.
    expected_physics=dict(copy.deepcopy(PHYSICS), native_engine_version=PHYSICS["version"],
                          version="sircus-calibrated-articulated-passive-contact-15360hz-v2",
                          scoring_dt_s=1/15360, scorer=SCORER_VERSION)
    checks.equal(identity["physics"],expected_physics,"Native dynamics fixed; calibrated dense scoring versioned")
    trace = identity["trace_schema"]
    expected_trace_version="sircus-synthetic-interval-trace-fixture-v2" if synthetic else "sircus-dense-native-interval-trace-v2"
    expected_dt=.001 if synthetic else 1/15360
    expected_horizon=1.6 if synthetic else 4.
    checks.equal(trace["version"],expected_trace_version,"Explicit evidence-kind trace schema")
    checks.numeric(trace["native_dt_s"],expected_dt,"Declared sampling time step",0)
    checks.numeric(trace["scoring_dt_s"],expected_dt,"Dense scoring time step",0)
    checks.equal(trace["force_names"],list(FORCE_NAMES),"Interval force attribution order")
    checks.numeric(trace["horizon_s"],expected_horizon,"Full declared horizon",0)
    calibration = identity["calibration"]
    for file,key in (("calibration.json","calibration_sha256"),("verification.json","verification_sha256")):
        checks.equal(sha(study/file),calibration[key],"Frozen control bytes/"+file)
        report = read(study/file)
        checks.equal(report["status"],"passed","Controls passed/"+file)
        checks.require(not report.get("failures"),"Control failures absent/"+file)
        count = report.get("checks",report.get("checks_count"))
        checks.require(type(count) is int and count > 0,"Positive independent control checks/"+file)
        checks.equal(bool(report.get("synthetic_fixture",False)),synthetic,"Control evidence kind/"+file)
    a,v = read(study/"calibration.json"),read(study/"verification.json")
    checks.equal(v["calibration_sha256"],sha(study/"calibration.json"),"Verification independently binds calibration")
    checks.equal(a["score_config"],calibration["score_config"],"Executed calibrated configuration")
    config = calibration["score_config"]
    checks.equal(digest(config),calibration["score_config_sha256"],"Calibrated score configuration digest")
    checks.equal(v["score_config_sha256"],digest(config),"Independent verification configuration digest")
    checks.numeric(config["retention_s"],1.,"One continuous second required",0)
    if "native_dt_s" in config:
        checks.numeric(config["native_dt_s"],expected_dt,"Calibrated control sampling matches study",0)
    setup = identity["setup"]
    checks.equal(sha(study/"setup-home.json"),setup["home_file_sha256"],"Canonical initial park bytes")
    checks.equal(read(study/"setup-home.json"),setup["home"],"Canonical initial park record")
    checks.equal(digest(setup["home"]),setup["home_record_sha256"],"Canonical initial park identity")
    validate_home(setup["home"],ref["scene"])
    checks.equal(setup["resolver"],RESOLVER_VERSION,"Frozen geometry resolver version")
    checks.equal(setup["anchor_search"],anchor_search_spec(),"Bounded geometry-only candidate rule")
    checks.equal(setup["geometry_candidate_budget"],64,"Frozen geometry endpoint/path candidate budget")
    checks.equal(setup["rrt_nodes"],192,"Frozen per-path RRT budget")
    checks.require(setup["no_flight_labels"] is True and setup["no_postprediction_action_fallback"] is True,"Geometry-only fixed request setup contract")
    checks.equal(contract["prior_sha256"],digest(prior),"Shared geometry-only prior")
    return synthetic


def check_rows(rows, checks):
    checks.equal(len(rows),len({r["id"] for r in rows}),"Unique requested trial IDs")
    low,high = np.asarray(WIDE_BOUNDS["low"]),np.asarray(WIDE_BOUNDS["high"])
    for row in rows:
        values=np.asarray(row["values"],float)
        checks.require(values.shape==(5,) and np.isfinite(values).all(),row["id"]+"/finite exact request")
        if row.get("native_release"):
            checks.require(((values>=low)&(values<=high)).all(),row["id"]+"/released request global bounds")
            checks.require(not row.get("invalid") or row.get("invalid_physics"),row["id"]+"/released invalid is explicit physics failure")
        else:
            checks.require(bool(row.get("invalid")) and not row.get("route_success"),row["id"]+"/rejection retains failed denominator")
        if row.get("invalid_physics"):
            checks.require(bool(row.get("invalid")) and not row.get("route_success"),row["id"]+"/invalid physics cannot capture")
        if row.get("route_success"):
            checks.require(row.get("native_release") and row.get("board_bounce") and row.get("whole_ball_entry")
                           and row.get("max_retention_s",0)>=1-1e-9 and not row.get("external_contact_steps"),row["id"]+"/complete board-route capture")
        checks.numeric(row.get("learning_reward",row_reward(row)),row_reward(row),row["id"]+"/calibrated terminal reward",1e-8)


def check_setup_evidence(directory, rows, prior, home, checks):
    from .calibrated_admission import verify_resolution, verify_connection
    sources={r["id"]:r for r in prior["records"]}
    resolved,connections={},{}
    raw=lines(directory/"native"/"shots.jsonl")
    checks.equal([r["id"] for r in raw],[r["id"] for r in rows],"Every raw native/admission request retained once in order")
    for source,row in zip(raw,rows):
        for key,value in source.items():
            if key=="invalid":
                checks.equal(bool(row.get(key)),bool(value),row["id"]+"/raw rejection retained")
            else:
                checks.equal(row.get(key),value,row["id"]+"/raw native field preserved/"+key)
    for row in rows:
        label=row["id"]
        name=row["context_resolution_file"]
        path=contained(directory/"native",name)
        checks.equal(sha(path),row["context_resolution_sha256"],label+"/preprediction resolution bytes")
        if name not in resolved:
            record=read(path)
            record.update(file=name,sha256=sha(path))
            checks.require(verify_resolution(record,prior["records"],home),label+"/deterministic geometry-only source resolution")
            resolved[name]=record
        resolution=resolved[name]
        hint=resolution["source_hint"]
        anchor=sources.get(hint.get("id"))
        checks.require(anchor is not None,label+"/shared source exists")
        for key in ("values","bucket_xy_m","source_record_sha256"):
            checks.equal(hint[key],anchor[key],label+"/source geometry only/"+key)
            checks.equal(row["source_hint"][key],anchor[key],label+"/request source unchanged/"+key)
        checks.equal(row["source_teacher_id"],anchor["id"],label+"/source identity before prediction")
        checks.equal(row["context_home_sha256"],home["sha256"],label+"/frozen canonical home")
        checks.equal(row["context_layout_eligible"],resolution["layout_eligible"],label+"/layout eligibility")
        checks.equal(row["context_anchor_admitted"],resolution["anchor_admitted"],label+"/nominal anchor separate from actual request")
        checks.equal(row["context_hard_reject"],resolution["hard_reject"],label+"/hard blocker identity")
        effective=resolution.get("effective_source_hint",hint)
        checks.equal(row["effective_source_hint"],effective,label+"/effective geometry anchor fixed before prediction")
        checks.numeric(row["observation"],expected_observation(row["bucket_xy_m"],effective),label+"/observation after fixed geometry resolution",0)
        connection_name=row.get("connection_file")
        if connection_name:
            connection_path=contained(directory/"native",connection_name)
            checks.equal(sha(connection_path),row["connection_sha256"],label+"/same-action setup proof bytes")
            connection=read(connection_path)
            checks.require(verify_connection(connection,resolution,home,row["values"]),label+"/same action endpoint and sampled setup path")
            connections[connection_name]=connection
            checks.equal(bool(row.get("motion_admitted")),bool(connection["passed"]),label+"/no path-result substitution")
            if row.get("native_release"):
                for joint in ("q10","q14"):
                    checks.numeric(row[joint],connection[joint],label+"/requested reset joint branch/"+joint,1e-12)
                    checks.numeric(row["native_initial_"+joint],connection[joint],label+"/observed native reset joint branch/"+joint,2e-6)
        else:
            checks.require(resolution["hard_reject"] or not resolution["layout_eligible"],label+"/only a hard layout blocker skips actual requested-action checks")
        if row.get("native_release"):
            checks.require(connection_name is not None and connections[connection_name]["passed"],label+"/native release has exact same-action path proof")
    return dict(resolutions=len(resolved),connections=len(connections))


def rescore_rows(directory, rows, scene, calibration, checks, label, trace_spec, *, assets=None):
    from .calibrated_score_oracle import score_trace
    by_file,traces={},{}
    tangency_model=None
    required=("route_success","route_quality","board_bounce","whole_ball_entry","max_retention_s",
              "closest_opening_distance_m","external_contact_steps","single_bounce_success","invalid_physics","physics_steps")
    dt=float(trace_spec["native_dt_s"])
    total_steps=int(round(float(trace_spec["horizon_s"])/dt))
    for row in rows:
        if row.get("native_release"):
            index=row["native_cache_row"]
            checks.require(type(index) is int and index>=0,label+"/native cache index")
            by_file.setdefault(row["native_cache_file"],[]).append((index,row))
    for filename,members in by_file.items():
        trace_path=contained(directory,filename)
        raw_sha=sha(trace_path)
        for _,row in members:
            checks.equal(row["native_cache_sha256"],raw_sha,label+"/frozen native trace bytes")
        with np.load(trace_path,allow_pickle=False) as archive:
            # NPZ indexing decompresses the whole member. Load each member once
            # per bounded packet, never once per trial column.
            arrays={key:archive[key] for key in archive.files}
            ids=arrays["record_ids"].tolist()
            checks.equal(len(ids),len(set(ids)),label+"/unique native IDs")
            checks.equal({i for i,_ in members},set(range(len(ids))),label+"/all dense trace columns assigned")
            checks.equal(len(members),len(ids),label+"/no repeated native columns")
            checks.equal(arrays["position_m"].shape,(total_steps+1,len(ids),3),label+"/dense position time-zero dimensions")
            checks.equal(arrays["velocity_m_s"].shape,(total_steps+1,len(ids),3),label+"/dense velocity time-zero dimensions")
            checks.equal(arrays["forces_N"].shape,(total_steps,len(ids),6),label+"/aligned native interval force dimensions")
            for key,shape in (("contact_distance_m",(total_steps,len(ids),2)),("contact_geom_ids",(total_steps,len(ids),2,2)),
                              ("contact_position_m",(total_steps,len(ids),2,3)),("contact_normal",(total_steps,len(ids),2,3))):
                checks.require(key in arrays,label+"/native interval contact evidence/"+key)
                checks.equal(arrays[key].shape,shape,label+"/native contact dimensions/"+key)
            contact=arrays["contact_distance_m"]
            checks.require((np.isfinite(contact)|np.isposinf(contact)).all(),label+"/contact distances finite or explicitly absent")
            present=np.isfinite(contact)
            geoms=arrays["contact_geom_ids"]
            checks.require(np.issubdtype(geoms.dtype,np.integer),label+"/native contact geom IDs integral")
            checks.require((geoms[present]>=0).all() and (geoms[~present]==-1).all(),label+"/present and absent native contact IDs")
            checks.require(np.isfinite(arrays["contact_position_m"]).all() and np.isfinite(arrays["contact_normal"]).all(),label+"/finite native contact frame observations")
            frame_norm=np.linalg.norm(arrays["contact_normal"],axis=-1)
            zero=present & ~arrays["contact_normal"].any(axis=-1)
            unit=np.abs(frame_norm-1.)<=1e-4
            # Exact CCD tangency can have an undefined all-zero direction. It is
            # never treated as a usable normal and must pass a separate CPU
            # distance query for the same geom pair and pre-integration state.
            tangency_epsilon_m=2e-6
            checks.require((~present|unit|(zero&(np.abs(contact)<=tangency_epsilon_m))).all(),label+"/native unit frames or precision-bounded exact tangencies")
            tangency_references={i:[] for i in range(len(ids))}
            geometry_file=directory/"native-compiled-geometry.json"
            synthetic=trace_spec["version"]=="sircus-synthetic-interval-trace-fixture-v2"
            if not synthetic:
                geometry=read(geometry_file)
                checks.numeric(geometry["compiled_planes"],calibration["compiled_planes"],label+"/runtime collision planes match calibration",calibration["geometry_epsilon_m"])
                tags=geometry["contact_geom_ids"]
                for category,allowed in ((0,tags["walls"]),(1,[tags["base"]] if isinstance(tags["base"],int) else tags["base"])):
                    pairs=geoms[:,:,category][present[:,:,category]]
                    if len(pairs):
                        a,b=pairs[:,0],pairs[:,1]
                        checks.require((((a==tags["ball"])&np.isin(b,allowed))|((b==tags["ball"])&np.isin(a,allowed))).all(),label+"/contact geom category attribution")
                for _,row in members:
                    checks.equal(row["calibrated_geometry_sha256"],geometry["geometry_sha256"],label+"/runtime compiled geometry identity")
            if zero.any():
                checks.require(not synthetic and assets is not None,label+"/zero frames require independently queryable native geometry")
                import mujoco
                from .verify_capture_calibration import compile_cpu_model
                if tangency_model is None:
                    model,_=compile_cpu_model(scene,assets,64)
                    tangency_model=(model,mujoco.MjData(model))
                model,data=tangency_model
                ball=int(model.geom("ball/geom").id)
                base=int(model.geom("bucket/base").id)
                q=int(model.jnt_qposadr[model.joint("ball/free").id])
                mocap=int(model.body_mocapid[model.body("bucket").id])
                by_index=dict(members)
                for tick,world,category in np.argwhere(zero):
                    pair=geoms[tick,world,category]
                    other=int(pair[1] if pair[0]==tags["ball"] else pair[0])
                    cpu_other=base if category==1 else int(model.geom(f"bucket/wall{tags['walls'].index(other):02d}").id)
                    row=by_index[int(world)]
                    data.mocap_pos[mocap]=[*row["bucket_xy_m"],0.]
                    data.qpos[q:q+3]=arrays["position_m"][tick,world]
                    data.qpos[q+3:q+7]=[1.,0.,0.,0.]
                    data.qvel[:]=0
                    mujoco.mj_forward(model,data)
                    cpu=float(mujoco.mj_geomDistance(model,data,ball,cpu_other,.05,None))
                    logged=float(contact[tick,world,category]);error=abs(cpu-logged)
                    checks.require(abs(cpu)<=tangency_epsilon_m and error<=tangency_epsilon_m,label+"/zero frame independently verified exact geometric tangency")
                    tangency_references[int(world)].append([int(tick),int(category),cpu,logged,error])
            ticks=np.arange(total_steps+1,dtype=np.float64)*dt
            checks.numeric(arrays["time_s"],ticks,label+"/explicit canonical native ticks",1e-12)
            checks.require("native_time_s" in arrays,label+"/observed native GPU clock retained")
            native_time=arrays["native_time_s"]
            checks.require(native_time.shape in ((total_steps+1,),(total_steps+1,len(ids))),label+"/native clock dimensions")
            checks.require(np.isfinite(native_time).all() and (np.diff(native_time,axis=0)>=0).all(),label+"/finite monotonic observed native clock")
            for index,row in members:
                name=label+"/"+row["id"]
                checks.equal(str(ids[index]),row["id"],name+"/dense trace request binding")
                trace={"position_m":arrays["position_m"][:,index],
                       "velocity_m_s":arrays["velocity_m_s"][:,index],
                       "forces_N":arrays["forces_N"][:,index],
                       "time_s":arrays["time_s"],
                       "contact_distance_m":contact[:,index],
                       "native_time_s":native_time if native_time.ndim==1 else native_time[:,index]}
                values=np.asarray(row["values"],float)
                release=values[:3].copy();release[2]+=values[4]+scene["ball"]["radius_m"]
                delta=np.asarray(row["bucket_xy_m"])-values[:2]
                distance=np.linalg.norm(delta)
                checks.require(distance>.15,name+"/strict board-to-bucket task distance")
                tilt=np.deg2rad(values[3])
                normal=np.r_[np.sin(tilt)*delta/distance,np.cos(tilt)]
                actual_center=np.asarray(row["board_center_m"],float)
                actual_normal=np.asarray(row["board_normal"],float)
                position_error=float(np.linalg.norm(actual_center-values[:3]))
                angle_error=float(np.rad2deg(np.arccos(np.clip(np.dot(actual_normal,normal),-1,1))))
                checks.require(position_error<.002,name+"/settled board center within unchanged2mm gate")
                checks.require(angle_error<.5,name+"/settled board normal within unchanged0.5degree gate")
                checks.numeric(row["release_ball_center_m"],release,name+"/release target bound to drop gap",2e-6)
                checks.numeric(row["native_initial_ball_center_m"],trace["position_m"][0],name+"/observed time-zero ball association",1e-9)
                checks.numeric(trace["position_m"][0],release,name+"/dense time-zero release pose",2e-6)
                checks.numeric(row["native_initial_bucket_center_m"],[*row["bucket_xy_m"],0.],name+"/native bucket matches requested layout",2e-6)
                pose=row.get("pose_gate",{})
                for key,bound in (("board_position_error_mm",2.),("board_normal_error_deg",.5),("dropper_error_mm",2.)):
                    value=pose.get(key,float("inf"))
                    checks.require(isinstance(value,(float,int)) and np.isfinite(value) and 0<=value<bound,name+"/settled native pose gate/"+key)
                checks.numeric(pose["board_position_error_mm"],1000*position_error,name+"/reported actual board center error",2e-3)
                checks.numeric(pose["board_normal_error_deg"],angle_error,name+"/reported actual board angle error",2e-3)
                checks.require(float(row.get("bucket_collider_error_m",float("inf")))<=1e-5,name+"/compiled bucket collider target")
                measured=score_trace(scene,row["bucket_xy_m"],row["board_normal"],calibration,trace)
                for key in required:
                    checks.require(key in row and key in measured,name+"/canonical score field/"+key)
                    if isinstance(measured[key],(float,np.floating)):
                        checks.numeric(row[key],measured[key],name+"/independent calibrated score/"+key,1e-8)
                    else:
                        checks.equal(row[key],measured[key],name+"/independent calibrated score/"+key)
                refs=tangency_references[index]
                traces[row["id"]]=dict(zero_frame_contacts=len(refs),references_sha256=digest(refs),
                    max_cpu_distance_error_m=max((r[4] for r in refs),default=0.),
                    max_cpu_tangency_distance_m=max((abs(r[2]) for r in refs),default=0.))
    checks.equal(set(by_file),{str(p.relative_to(directory)) for p in directory.rglob("*.npz")},label+"/no unassigned dense native arrays")
    return traces


def contact_geometry_summary(traces):
    return dict(zero_frame_contacts=sum(r["zero_frame_contacts"] for r in traces.values()),
                maximum_cpu_distance_error_m=max((r["max_cpu_distance_error_m"] for r in traces.values()),default=0.),
                maximum_cpu_tangency_distance_m=max((r["max_cpu_tangency_distance_m"] for r in traces.values()),default=0.),
                references_sha256=digest(traces),all_zero_frame_contacts_checked=True,
                treatment="No direction inferred; zero frames accepted only within fixed2um exact tangency and independently checked same-pair PREstate CPU geomDistance")


def checked_artifacts(directory,contract,method,seed,checks):
    result={}
    for samples in contract["protocol"]["milestones"]:
        relative=(f"checkpoints/milestone-{samples:04d}.pt" if method=="ppo" else f"sweep/milestone-{samples:04d}.json")
        path=contained(directory,relative)
        state=common.load_checkpoint(path,checks,f"{method}/{seed}/{samples}") if method=="ppo" else read(path)
        checks.equal(state["contract_fingerprint"],contract["fingerprint"],"Milestone exact contract")
        checks.equal(state["samples"],samples,"Milestone exact TRAIN prefix")
        checks.equal(state["method"],method,"Milestone method")
        checks.equal(state["seed"],seed,"Milestone seed")
        result[samples]=dict(path=relative,sha256=sha(path),state=state)
    return result


def audit_predictions(directory,contract,rows,frozen,sweep_states,selected,method,checks):
    prediction_files=set()
    for role, samples_set in (("validation",contract["protocol"]["milestones"]),("test",[selected])):
        contexts=contract["splits"][role]
        cohort=[r for r in rows if r["role"]==role]
        for samples in samples_set:
            relative=f"predictions/{role}-{samples:04d}.json"
            prediction_files.add(relative)
            prediction=read(directory/relative)
            artifact=frozen[samples]
            checks.equal(prediction["role"],role,"Prediction role")
            checks.equal(prediction["samples"],samples,"Prediction TRAIN prefix")
            checks.equal(prediction["artifact"]["sha256"],artifact["sha256"],"Frozen direct-prediction artifact")
            checks.require(prediction["selection_or_learning"] is False,"No heldout learning or flight search")
            checks.equal(len(prediction["rows"]),len(contexts),"Exactly one base prediction per layout")
            stage=[r for r in cohort if r["samples"]==samples]
            checks.equal(len(stage),3*len(contexts),"All paired evaluation requests retained")
            for context,predicted in zip(contexts,prediction["rows"]):
                matching=[r for r in stage if r["context_id"]==context["id"]]
                checks.equal(len(matching),3,"Exactly three paired requests")
                checks.equal({r["repeat"] for r in matching},{0,1,2},"Unique paired repetitions")
                base=matching[0]
                anchor=base["effective_source_hint"]
                checks.equal(predicted["context_id"],context["id"],"Direct prediction layout ID")
                checks.numeric(predicted["bucket_xy_m"],context["bucket_xy_m"],"Direct prediction coordinate",0)
                checks.equal(predicted["source_teacher_id"],anchor["id"],"Preprediction geometry-selected source")
                obs=expected_observation(context["bucket_xy_m"],anchor)
                checks.numeric(predicted["observation"],obs,"Direct prediction observation",0)
                if method=="ppo":
                    latent=common.network_forward(frozen[samples]["state"]["policy"],obs,"actor")
                    checks.numeric(predicted["latent_mean"],latent,"Independent frozen actor prediction",2e-6)
                    residual=np.tanh(np.asarray(latent,float))
                    checks.equal(predicted["source_training_ids"],[],"No heldout imitation labels")
                else:
                    values,metadata=common.independent_sweep_prediction(sweep_states[samples],context,anchor)
                    residual=metadata["normalized_residual"]
                    checks.equal(predicted["source_training_ids"],metadata["source_training_ids"],"TRAIN-only transferred library dependency")
                values=np.asarray(anchor["values"])+np.asarray(residual)*RESIDUAL
                checks.numeric(predicted["command_values"],values,"Frozen base action without outcome search",2e-6)
                for row in matching:
                    checks.equal(row["frozen_artifact_sha256"],artifact["sha256"],"Exact frozen test/validation artifact")
                    checks.numeric(row["command_values"],predicted["command_values"],"No action replacement after prediction",0)
                    noise=expected_noise(context["id"],row["repeat"])
                    checks.numeric(row["perturbation_mixed_units"],noise,"Paired placement perturbation",0)
                    checks.numeric(row["values"],np.asarray(row["command_values"])+noise,"Exact realized request, no projection",1e-12)
                    checks.equal(row["context_resolution_sha256"],predicted["context_resolution_sha256"],"Same geometry source before all repetitions")
            checks.numeric(prediction["pre_prediction_plus_policy_s"],prediction["query_s"]+prediction["context_resolution_timing"]["wall_s"],"Pre-prediction geometry plus frozen policy timing",1e-9)
            if role=="test":
                final=read(directory/"report.json")["final_test"]
                checks.equal(prediction["query_s"],final["query_s"],"Cohort prediction/preparation timing")
                checks.equal(prediction["context_resolution_timing"],final["context_resolution_timing"],"Cohort geometry timing retained")
                checks.equal(prediction["pre_prediction_plus_policy_s"],final["pre_prediction_plus_policy_s"],"Pre-prediction pipeline timing retained")
    checks.equal(prediction_files,{str(p.relative_to(directory)) for p in (directory/"predictions").glob("*.json")},"No extra test/validation query files")


def audit_run(directory,contract,prior,scene,checks,*,assets=None):
    directory=Path(directory)
    common.check_manifest(directory,checks)
    report=read(directory/"report.json")
    fingerprint(report,checks,"Worker report")
    checks.equal(report["status"],"completed","Worker completed")
    checks.equal(report["phase"],"completed","Worker final phase")
    checks.equal(report["version"],STUDY_VERSION,"Worker calibrated version")
    checks.equal(report["contract_fingerprint"],contract["fingerprint"],"Worker contract binding")
    method,seed,p=report["method"],report["seed"],contract["protocol"]
    checks.require(method in {"sweep","ppo"} and seed in p["seeds"],"Declared method/seed")
    checks.equal(directory.name,f"{method}-seed{seed}","Independent method/seed directory")
    rows=lines(directory/"rows.jsonl")
    check_rows(rows,checks)
    train=[r for r in rows if r["role"]=="train"]
    budget=p["sample_budget"]
    checks.equal(len(train),budget,"Paid requested TRAIN budget")
    checks.equal([r["train_index"] for r in train],list(range(budget)),"TRAIN prefix and ordering")
    schedule=expected_schedule(seed,budget)
    checks.equal(read(directory/"schedule.json"),schedule,"Independent balanced context schedule")
    for index,row in enumerate(train):
        context=contract["splits"]["train"][schedule[index]]
        checks.equal(row["context_id"],context["id"],"TRAIN-only scheduled context")
        checks.numeric(row["bucket_xy_m"],context["bucket_xy_m"],"TRAIN coordinate",0)
        checks.equal(row["samples"],index+1,"Exact TRAIN sample prefix")
        residual=np.tanh(row["latent"]) if method=="ppo" else np.asarray(row["normalized_residual"])
        checks.numeric(row["normalized_residual"],residual,"Bounded residual action",1e-12)
        expected=np.asarray(row["effective_source_hint"]["values"])+residual*RESIDUAL
        checks.numeric(row["command_values"],expected,"Requested policy action",1e-12)
        checks.numeric(row["values"],expected,"No training action substitution",1e-12)
    expected_count=budget+36*len(p["milestones"])+72
    checks.equal(len(rows),expected_count,"All TRAIN/validation/test requests retained")
    provenance=check_setup_evidence(directory,rows,prior,contract["identity"]["setup"]["home"],checks)
    traces=rescore_rows(directory/"native",rows,scene,contract["identity"]["calibration"]["score_config"],checks,directory.name,contract["identity"]["trace_schema"],assets=assets)
    frozen=checked_artifacts(directory,contract,method,seed,checks)
    sweep_states={}
    if method=="ppo":
        common.audit_ppo_training(directory,contract,train,frozen,checks)
    else:
        states={k:v["state"]["state"] for k,v in frozen.items()}
        common.reconstruct_sweep(seed,contract["splits"]["train"],train,checks,snapshots=states)
        sweep_states=states
    points=report["curves"]
    checks.equal([r["samples"] for r in points],p["milestones"],"Every prespecified curve milestone")
    for point in points:
        stage=[r for r in rows if r["role"]=="validation" and r["samples"]==point["samples"]]
        stats_equal(point["validation"],stage,checks,"Validation cohort counts")
        checks.equal(point["artifact_sha256"],frozen[point["samples"]]["sha256"],"Curve milestone artifact")
    chosen=max(points,key=lambda r:(r["validation"]["captures"],r["validation"]["mean_reward"],-r["samples"]))
    selection=read(directory/"selected.json")
    selected=chosen["samples"]
    checks.equal(selection["selected_samples"],selected,"Validation-only exact checkpoint selection")
    checks.require(selection["frozen_before_test"] is True,"Artifact selected before test")
    checks.equal(selection["artifact"]["sha256"],frozen[selected]["sha256"],"Frozen selection bytes")
    checks.equal(report["selected_samples"],selected,"Report selection")
    checks.equal(report["checkpoint_sha256"],frozen[selected]["sha256"],"Report frozen checkpoint")
    audit_predictions(directory,contract,rows,frozen,sweep_states,selected,method,checks)
    test=[r for r in rows if r["role"]=="test"]
    stats_equal(report["final_test"]["counts"],test,checks,"Final frozen test counts")
    stats_equal(report["train_counts"],train,checks,"TRAIN counts")
    checks.equal(report["admission"],admission_counts(train),"TRAIN-only setup status and failure reasons")
    checks.equal(report["train_requests"],budget,"Full paid TRAIN requests")
    checks.equal(report["train_native_releases"],counts(train)["native_releases"],"TRAIN native release accounting")
    checks.equal(report["final_test"]["frozen_artifact_sha256"],frozen[selected]["sha256"],"Final test exact frozen artifact")
    checks.require(report["final_test"]["ball_flight_search_at_inference"] is False,"No final test ball-flight optimization")
    checks.require(report["final_test"]["geometry_search_at_inference"] is True,"Geometry preprocessing cost retained")
    resource=common.check_resources(directory,report,checks)
    batches=lines(directory/"batches.jsonl")
    for point in report["curves"]:
        prefix=[b for b in batches if b["samples"]<=point["samples"]]
        for key in ("path_s","context_resolution_s","postprocess_s"):
            checks.numeric(point["resources"][key],sum(b["adapter_timing"].get(key,0.) for b in prefix),"Measured TRAIN prefix cost/"+key,1e-9)
        checks.numeric(point["resources"]["admission_wall_s"],sum(b["adapter_timing"].get("wall_s",0.) for b in prefix),"Measured TRAIN admission wall prefix",1e-9)
    return report,rows,dict(provenance=provenance,resources=resource,native_releases=len(traces),contact_geometry=contact_geometry_summary(traces))


def build_run_summary(method,seed,budget,rows,native_summary):
    if native_summary["native_releases"] != counts(rows)["native_releases"]:
        raise EvidenceError("Run summary native count differs")
    return dict(method=method,seed=seed,requested=len(rows),training_requests=budget,**native_summary)


def audit(study,reference,assets,*,repository):
    from .sweep_worker import load_reference
    begin,checks=time.monotonic(),Checks()
    study,repository=Path(study),Path(repository)
    check_manifest(study,checks,ignored=("controller.lock","active-worker.json"))
    contract,report,prior=(read(study/name) for name in ("contract.json","report.json","prior.json"))
    common.verify_prior_bundle(repository,prior,checks=checks)
    ref=load_reference(reference,assets,repository=repository)
    synthetic=check_contract(study,contract,prior,ref,repository,checks)
    fingerprint(report,checks,"Complete overall report")
    checks.equal(report["version"],STUDY_VERSION,"Overall calibrated version")
    checks.equal(report["status"],"completed","Overall report frozen before audit")
    checks.equal(report["phase"],"completed","All finite workers finished")
    checks.equal(report["contract_fingerprint"],contract["fingerprint"],"Overall contract binding")
    checks.equal(report["protocol"],contract["protocol"],"Overall protocol unchanged")
    checks.equal(read(study/"splits.json"),contract["splits"],"Copied split manifest")
    preflight=read(study/"preflight.json")
    checks.equal(sha(study/"preflight.json"),contract["preflight"]["report_sha256"],"Frozen preflight bytes")
    checks.equal(preflight["status"],"passed","Native integration smoke passed")
    checks.equal(preflight["identity"],contract["identity"],"Preflight executed same code/scorer/home")
    checks.equal(preflight["prior_sha256"],contract["prior_sha256"],"Preflight same shared prior")
    checks.equal(preflight["source_sha256"],sha(repository/"sircus/mjlab/calibrated_benchmark_smoke.py"),"Preflight driver identity")
    control_audit=read(study/"controls-audit.json")
    checks.equal(sha(study/"controls-audit.json"),contract["preflight"]["controls_audit_sha256"],"Passed independent controls audit bytes")
    checks.equal(control_audit.get("version"),VERSION,"Independent controls audit version")
    checks.equal(control_audit.get("status"),"passed","Independent controls audit passed")
    checks.require(type(control_audit.get("checks")) is int and control_audit["checks"]>0 and not control_audit.get("failures"),"Positive complete controls audit")
    checks.equal(control_audit["identity"],contract["identity"],"Control auditor exact execution identity")
    checks.equal(control_audit["report_sha256"],sha(study/"preflight.json"),"Control audit binds frozen smoke report")
    checks.equal(control_audit["study_manifest_sha256"],contract["preflight"]["manifest_sha256"],"Control audit binds frozen smoke manifest")
    checks.equal(control_audit["auditor_sha256"],sha(Path(__file__)),"Control audit uses current independent auditor")
    checks.equal(preflight["requests"],50,"Quarantined engineering request budget")
    checks.equal(preflight["controls_evaluated"],17,"All historical source controls retained")
    checks.require(1<=preflight["controls_captured"]<=17,"Positive source-smoke capture")
    checks.equal(preflight["discarded_ppo_requests"],32,"Discarded actual PPO batch")
    checks.require(preflight["actor_changed"] is True,"Native smoke changed actor")
    pairs={(m,seed) for m in ("sweep","ppo") for seed in contract["protocol"]["seeds"]}
    checks.equal({p.name for p in (study/"runs").iterdir() if p.is_dir()},{f"{m}-seed{s}" for m,s in pairs},"Exactly declared worker directories")
    all_rows,summaries,run_reports=[],[],{}
    for method,seed in sorted(pairs):
        rr,rows,summary=audit_run(study/"runs"/f"{method}-seed{seed}",contract,prior,ref["scene"],checks,assets=assets)
        run_reports[(method,seed)]=rr
        all_rows.extend(rows)
        summaries.append(build_run_summary(method,seed,contract["protocol"]["sample_budget"],rows,summary))
    checks.equal(len(all_rows),len({r["id"] for r in all_rows}),"Global unique trial IDs")
    checks.equal(len(report["runs"]),len(pairs),"Every complete worker summary")
    for rr in report["runs"]:
        checks.equal(rr,run_reports[(rr["method"],rr["seed"] )],"Overall keeps exact immutable worker report")
    expected_curves=[p for rr in report["runs"] for p in rr["curves"]]
    checks.equal(report["curves"],expected_curves,"Overall keeps exact validation curves")
    expected_final=[rr["final_test"] for rr in report["runs"]]
    checks.equal(report["final_test"],expected_final,"Overall keeps exact frozen test outcomes")
    expected=len(pairs)*(contract["protocol"]["sample_budget"]+36*len(contract["protocol"]["milestones"])+72)
    checks.equal(len(all_rows),expected,"Exact full requested-trial budget")
    return dict(version=VERSION,status="passed",checks=checks.count,failures=[],
                study_kind=contract["protocol"]["study_kind"],synthetic_fixture=synthetic,
                contract_fingerprint=contract["fingerprint"],report_sha256=sha(study/"report.json"),
                report_fingerprint=report["fingerprint"],study_manifest_sha256=sha(study/"manifest.json"),
                auditor_sha256=sha(Path(__file__)),
                auditor_dependencies={name:sha(repository/name) for name in ("sircus/mjlab/benchmark_verify.py","sircus/mjlab/verify_capture_calibration.py")},
                source_sha256=contract["identity"]["source_sha256"],
                requested_trials=len(all_rows),native_releases=counts(all_rows)["native_releases"],
                role_counts={role:counts([r for r in all_rows if r["role"]==role]) for role in ("train","validation","test")},
                admission_by_role={role:admission_counts([r for r in all_rows if r["role"]==role]) for role in ("train","validation","test")},
                runs=summaries,elapsed_s=time.monotonic()-begin,maximum_numeric_error=checks.numeric_error,
                scope="Independent CPU fresh-split/budget/source/calibration/home/proof checks, direct prediction and actual PPO/Adam replay, and dense native-interval oracle rescoring; no new rollout.",
                limitations=["Sampled geometry proofs are hash/semantic checked, not a new continuous collision proof.",
                             "Measured memory includes simulator/evaluation; telemetry is sampled; instrumented wall time is not kernel utilization.",
                             "Current-ball privileged-geometry held-endpoint shot selection, not physical grasp/transit/compliance validation."])


def audit_controls(study,reference,assets,*,repository):
    """Audit quarantined TRAIN-only smoke rows and its actual discarded update."""
    from .sweep_worker import load_reference
    begin,checks=time.monotonic(),Checks()
    study,repository=Path(study),Path(repository)
    check_manifest(study,checks)
    report=read(study/"report.json")
    prior=common.verify_prior_bundle(repository,checks=checks)
    # verify_prior_bundle includes derived audit metadata; the producer stores the
    # canonical original geometry-only load_prior object instead.
    from .benchmark_contract import load_prior
    canonical=load_prior(repository/PRIOR_PATH)
    ref=load_reference(reference,assets,repository=repository)
    checks.equal(report["status"],"passed","Quarantined native smoke passed")
    from .calibrated_benchmark_contract import checked_calibration, checked_setup_home, study_identity
    synthetic=bool(report.get("synthetic_fixture",False))
    calibration=checked_calibration(study/"calibration.json",study/"verification.json",synthetic=synthetic)
    setup=checked_setup_home(study/"setup-home.json")
    checks.equal(report["identity"],study_identity(ref,calibration,setup,repository=repository,synthetic=synthetic),"Smoke exact execution/calibration/home identity")
    checks.equal(report["source_sha256"],sha(repository/"sircus/mjlab/calibrated_benchmark_smoke.py"),"Smoke driver source identity")
    checks.equal(report["prior_sha256"],digest(canonical),"Control shared prior")
    identity=report["identity"]
    checks.equal(identity["calibration"]["calibration_sha256"],sha(study/"calibration.json"),"Control frozen calibration")
    checks.equal(identity["calibration"]["verification_sha256"],sha(study/"verification.json"),"Control frozen independent verification")
    controls=read(study/"controls.json")["rows"]
    smoke=read(study/"ppo-smoke.json")
    raw=[*controls]
    invalid=read(study/"deliberate-reject.json")["rows"]
    raw.extend(invalid)
    raw.extend(smoke["rows"])
    checks.equal(len(raw),50,"All quarantined engineering requests")
    checks.equal(len(controls),17,"Source control count")
    for original,actual in zip(canonical["records"],controls):
        checks.equal(actual["id"],"control-"+original["id"],"All historical control identities retained")
        checks.numeric(actual["values"],original["values"],"Historical control exact requested parameters",0)
        checks.numeric(actual["bucket_xy_m"],original["bucket_xy_m"],"Historical control exact layouts",0)
    checks.equal(len(smoke["rows"]),32,"Discarded native PPO requests")
    checks.equal(len(invalid),1,"Deliberate invalid request retained")
    checks.require(not invalid[0]["native_release"] and bool(invalid[0]["invalid"]),"Invalid request is not silently replaced")
    for row in raw:
        row.setdefault("learning_reward",row_reward(row))
    check_rows(raw,checks)
    # Smokes do not have a learned observation on every row; setup proofs and
    # oracle scores are still fully checked independently.
    for row in raw:
        row.setdefault("observation",expected_observation(row["bucket_xy_m"],row["effective_source_hint"]).tolist())
    provenance=check_setup_evidence(study,raw,canonical,identity["setup"]["home"],checks)
    traces=rescore_rows(study/"native",raw,ref["scene"],identity["calibration"]["score_config"],checks,"engineering smoke",identity["trace_schema"],assets=assets)
    before=common.load_checkpoint(study/"ppo-before.pt",checks,"smoke before")
    after=common.load_checkpoint(study/"ppo-after.pt",checks,"smoke after")
    with np.load(study/"ppo-buffer.npz",allow_pickle=False) as stored:
        buffer={k:stored[k] for k in stored.files}
    checks.numeric(buffer["reward"],[row_reward(r) for r in smoke["rows"]],"Smoke native calibrated rewards",1e-6)
    common.check_behavior_likelihood(before["policy"],buffer["observation"],buffer["latent"],buffer["old_logp"],checks,"discarded smoke")
    common.replay_ppo_update(before["policy"],before["optimizer"],after["policy"],dict(obs=buffer["observation"],**{k:buffer[k] for k in ("latent","old_logp","reward")}),checks,"discarded smoke")
    checks.require(common.actor_module_sha256(before["policy"])!=common.actor_module_sha256(after["policy"]),"Actual smoke actor update")
    checks.equal(report["controls_captured"],counts(controls)["captures"],"Source control labels measured")
    return dict(version=VERSION,status="passed",checks=checks.count,failures=[],identity=identity,
                report_sha256=sha(study/"report.json"),study_manifest_sha256=sha(study/"manifest.json"),
                auditor_sha256=sha(Path(__file__)),requested_trials=50,native_releases=len(traces),
                contact_geometry=contact_geometry_summary(traces),
                elapsed_s=time.monotonic()-begin,scope="Independent CPU dense oracle, setup-proof, source and discarded PPO update audit; no new rollout.")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("study","reference","assets","repository","output"):
        parser.add_argument("--"+name,type=Path,required=True)
    parser.add_argument("--controls-only",action="store_true")
    args=parser.parse_args()
    if args.output.resolve().is_relative_to(args.study.resolve()) or args.output.exists():
        parser.error("New audit output must be outside immutable source evidence")
    try:
        result=(audit_controls if args.controls_only else audit)(args.study,args.reference,args.assets,repository=args.repository)
    except Exception as exc:
        result=dict(version=VERSION,status="failed",checks=0,failures=[f"{type(exc).__name__}: {exc}"])
        for filename,key in (("report.json","report_sha256"),("contract.json","contract_sha256")):
            path=args.study/filename
            if path.is_file(): result[key]=sha(path)
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
    print(json.dumps({k:result[k] for k in ("status","checks","failures","requested_trials","native_releases") if k in result}),flush=True)
    if result["status"]!="passed": raise SystemExit(1)


if __name__=="__main__": main()
