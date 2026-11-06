"""Independent CPU audit for bounded650 mm surface-tangent impact exploration.

Nominal board-center-plane ray intersections are not measured sphere contacts.
The calibrated capture oracle is unchanged; actual saved native release states
must realize the new action rather than silently using a center-only reset.
"""
from __future__ import annotations

import argparse
from collections import Counter
import copy
import itertools
import json
import math
from pathlib import Path
import sys
import time

import numpy as np

from .benchmark_verify import EvidenceError, contained, digest, read, sha
from . import clean650_verify as clean_audit
from . import replay650_verify as replay
from . import xy650_verify as xy_audit
from . import tilt650_verify as tilt_audit
from .calibrated_benchmark_verify import check_native_clock

VERSION = "sircus-impact650-independent-audit-v1"
STUDY_VERSION = "sircus-impact650-study-v1"
SOURCE_IDS = ("refinement-64-refine-s02-h1t0g2-r0", "coarse-64-xy-grid-0043-r0", "coarse-64-xy-grid-0096-r0")
FAMILIES = ("family-00", "family-01", "family-02")
FORWARD = (-.04, -.03, -.02, -.01, 0., .01, .02, .03, .04)
LATERAL = (-.02, -.01, 0., .01, .02)
STEPS = (64, 128)
MAX_REQUESTS = 189
RULE = "sircus-board-center-plane-forward-lateral-impact-v1"
BASE_VERSION = "sircus-impact650-board-surface-forward-lateral-v1"
GAP_DEFINITION = "vertical_ball_bottom_clearance_above_offset_board_center_plane"
AuditChecks = replay.AuditChecks


def nominal_geometry(scene, bucket_xy, values, offset):
    values, xy, offset = np.asarray(values, dtype=float), np.asarray(bucket_xy, dtype=float), np.asarray(offset, dtype=float)
    if values.shape != (5,) or xy.shape != (2,) or offset.shape != (2,) or not all(np.isfinite(v).all() for v in (values, xy, offset)):
        raise ValueError("Finite five-value action, bucketXY and two tangent offsets required")
    delta = xy-values[:2];distance = float(np.linalg.norm(delta))
    if distance <= .15:
        raise ValueError("Separated board and bucket required")
    direction = delta/distance;angle = math.radians(float(values[3]))
    if not 0 <= angle < math.pi/2 or values[4] <= 0:
        raise ValueError("Valid tilt and positive vertical drop gap required")
    forward = np.r_[math.cos(angle)*direction, -math.sin(angle)]
    lateral = np.asarray([-direction[1], direction[0], 0.])
    normal = np.r_[math.sin(angle)*direction, math.cos(angle)]
    point = values[:3]+offset[0]*forward+offset[1]*lateral
    release = point+np.asarray([0., 0., values[4]+scene["ball"]["radius_m"]])
    return dict(impact_point_world_m=point.tolist(), release_center_m=release.tolist(), release_center_height_m=float(release[2]),
                impact_to_bucket_horizontal_m=float(np.linalg.norm(xy-point[:2])),
                board_normal_world=normal.tolist(), forward_world=forward.tolist(), lateral_world=lateral.tolist())


def endpoint_targets(scene, bucket_xy, values, offset):
    """Independently build offset release and unchanged board/tool transforms."""
    geometry = nominal_geometry(scene, bucket_xy, values, offset)
    values, xy = np.asarray(values, dtype=float), np.asarray(bucket_xy, dtype=float)
    heading = math.atan2(xy[1]-values[1], xy[0]-values[0])+math.pi/2
    angle, yaw = math.radians(values[3]), float(scene["board"]["mount_yaw_rad"])
    rz = lambda a: np.asarray([[math.cos(a), -math.sin(a), 0.], [math.sin(a), math.cos(a), 0.], [0., 0., 1.]])
    rx = np.asarray([[1., 0., 0.], [0., math.cos(angle), -math.sin(angle)], [0., math.sin(angle), math.cos(angle)]])
    world = np.eye(4);world[:3, :3] = rz(heading)@rx@rz(yaw);world[:3, 3] = values[:3]
    mount = np.eye(4);mount[:3, :3] = rz(yaw);mount[2, 3] = scene["board"]["tool_offset_m"]
    base = np.diag([-1., -1., 1., 1.]);base[0, 3] = scene["arms"]["base_spacing_m"]
    dropper = np.eye(4);dropper[:3, :3] = np.diag([1., -1., -1.])
    dropper[:3, 3] = np.asarray(geometry["release_center_m"])+[0., 0., .095]
    return dict(board_world=world, board_flange_local=np.linalg.inv(base)@world@np.linalg.inv(mount),
                dropper_flange_world=dropper, release_center_m=np.asarray(geometry["release_center_m"]),
                board_normal_world=np.asarray(geometry["board_normal_world"]))


def target_key(row):
    return tuple(round(float(value), 9) for value in [*row["values"], *row["bucket_xy_m"], *row["impact_offset_m"]])


def selected_verification(rows, count=5):
    valid = sorted((row for row in rows if xy_audit.native_valid(row)), key=xy_audit.order_key)
    chosen, seen = [], set()
    for family in FAMILIES:
        candidate = next((row for row in valid if row["family_id"] == family and target_key(row) not in seen), None)
        if candidate is not None:
            chosen.append(candidate);seen.add(target_key(candidate))
    for row in valid:
        if len(chosen) == count:
            break
        if target_key(row) not in seen:
            chosen.append(row);seen.add(target_key(row))
    return chosen[:count]


def qualification(rows, positions, *, controls_passed, source_capture):
    result = tilt_audit.qualification(rows, positions, controls_passed=controls_passed, parent_source_capture=source_capture)
    result["source_capture"] = result.pop("parent_source_capture")
    return result


def scene_fingerprint(parent_physical):
    return digest(dict(physical_scene_fingerprint=parent_physical, impact_rule=RULE))


def query_fingerprint(group, scene_fp):
    return digest(dict(scene_fingerprint=scene_fp, values=group["values"], bucket_xy_m=group["bucket_xy_m"],
                       impact_offset_m=group["impact_offset_m"], impact_rule=RULE,
                       source_record_id=group["source_record_id"], source_record_sha256=group["source_record_sha256"]))


def execution_fingerprint(row):
    return digest({key: row[key] for key in ("id", "query_fingerprint", "substeps", "repeat_index", "q10", "q14",
                                            "native_initial_q10", "native_initial_q14", "native_initial_ball_center_m")})


def make_plan(groups, phase, *, steps, repeats):
    rows = xy_audit.make_plan(groups, phase, steps=steps, repeats=repeats)
    lookup = {group["id"]: group for group in groups}
    for row in rows:
        for key in ("family_id", "seed_id", "impact_offset_m", "impact_point_world_m", "impact_to_bucket_horizontal_m", "release_center_m", "release_center_height_m"):
            row[key] = copy.deepcopy(lookup[row["group_id"]][key])
    return rows


def select_families(parent_rows, checks):
    lookup = {row["id"]: row for row in parent_rows}
    selected = [lookup[ident] for ident in SOURCE_IDS]
    for source, parameters in zip(selected, ((.38, 27.5, .3918), (.38, 30., .41), (.38, 30., .39))):
        checks.require(xy_audit.native_valid(source) and replay.captured(source), source["id"]+"/original valid rim capture")
        checks.equal(source["route_quality"], 1, source["id"]+"/historical source remains rim class")
        checks.numeric(source["values"][2:], parameters, source["id"]+"/literal fixed family", 1e-12)
        checks.numeric(np.linalg.norm(np.asarray(source["bucket_xy_m"])-source["values"][:2]), .65, source["id"]+"/650mm board-center distance", 1e-12)
    return selected


def audit_initial(groups, selected, controls, checks):
    checks.equal(len(groups), 137, "Two original controls plus135 literal impact offsets")
    for actual, source in zip(groups[:2], controls):
        for key in ("id", "role", "phase", "values", "bucket_xy_m", "q10", "q14", "source_record_id", "source_record_sha256", "translation_m"):
            checks.equal(actual[key], source[key], "Exact unchanged historical control/"+key)
        checks.equal(actual["family_id"], "control", "Controls remain outside candidate families")
        checks.equal(actual["seed_id"], source["source_record_id"], "Historical control provenance")
        checks.equal(actual["impact_offset_m"], [0., 0.], "No off-center historical control")
    for family, source in enumerate(selected):
        for index, (forward, lateral) in enumerate(itertools.product(FORWARD, LATERAL)):
            i, j = divmod(index, 5)
            group = groups[2+family*45+index];label = f"impact-f{family:02d}-u{i}v{j}"
            checks.equal(group["id"], label, "Literal family/forward/lateral grid order")
            for key, value in dict(role="discovery", phase="coarse", family_id=FAMILIES[family], seed_id=source["id"], query_kind="impact_grid",
                                   impact_offset_m=[forward, lateral], source_record_id=source["id"], source_record_sha256=digest(source),
                                   source_query_fingerprint=source["query_fingerprint"]).items():
                checks.equal(group[key], value, label+"/literal input/"+key)
            for key in ("values", "bucket_xy_m", "q14", "translation_m"):
                checks.equal(group[key], source[key], label+"/fixed source board/family/"+key)
            if forward == lateral == 0.:
                checks.equal(group["q10"], source["q10"], label+"/exact historical zero-offset dropper branch")


def audit_verification(groups, selected, checks):
    checks.equal(len(groups), len(selected), "Exactly one verification group per family-diverse selected target")
    checks.equal(len({target_key(row) for row in selected}), len(selected), "Unique selected offset targets")
    for index, (group, source) in enumerate(zip(groups, selected)):
        label = f"impact-verify-{index:02d}"
        for key, value in dict(id=label, role="verification", phase="verification", query_kind="nominal_verification",
                               freeze_joints=True, source_record_id=source["id"], source_record_sha256=digest(source),
                               source_query_fingerprint=source["query_fingerprint"], source_route_quality=source.get("route_quality", 0)).items():
            checks.equal(group[key], value, label+"/exact selected source/"+key)
        for key in ("family_id", "seed_id", "values", "bucket_xy_m", "impact_offset_m", "q10", "q14", "translation_m"):
            checks.equal(group[key], source[key], label+"/exact frozen target/"+key)


def audit_geometry(groups, proofs, scene, assets, checks):
    from .corrected_geometry import StrictEndpointGate
    from .expanded_actions import WIDE_BOUNDS
    gate = StrictEndpointGate(scene, assets, .005)
    checks.equal(len(groups), len(proofs), "One complete proof per requested impact query")
    checks.equal(len({group["id"] for group in groups}), len(groups), "Unique geometry request identities")
    table = scene["table"];center = np.asarray(table["center_m"][:2]);half = np.asarray(table["size_m"][:2])/2
    radius = max(scene["bucket"]["outer_top_diameter_m"], scene["bucket"]["outer_bottom_diameter_m"])/2
    for index, (group, proof) in enumerate(zip(groups, proofs)):
        label = group["id"];values = np.asarray(group["values"]);xy = np.asarray(group["bucket_xy_m"])
        targets = endpoint_targets(scene, xy, values, group["impact_offset_m"])
        nominal = nominal_geometry(scene, xy, values, group["impact_offset_m"])
        checks.equal(group["geometry_row"], index, label+"/ordered geometry proof")
        checks.equal(group["geometry_sha256"], digest(proof), label+"/immutable geometry proof bytes")
        checks.equal(proof["group_id"], label, label+"/proof exact identity")
        for key in ("values", "bucket_xy_m", "impact_offset_m", "eligible"):
            checks.equal(proof[key], group[key], label+"/proof exact input/"+key)
        for key in ("impact_point_world_m", "impact_to_bucket_horizontal_m", "release_center_m", "release_center_height_m"):
            checks.numeric(group[key], nominal[key], label+"/independent offset geometry/"+key, 1e-12)
            checks.numeric(proof[key], nominal[key], label+"/proof offset geometry/"+key, 1e-12)
        for key, expected in dict(impact_basis_version=BASE_VERSION, impact_gap_definition=GAP_DEFINITION,
                                   board_q14_frozen=True, ik_starts=16, ik_iterations=180, pair_budget=17, held_endpoints_only=True).items():
            checks.equal(proof[key], expected, label+"/literal endpoint contract/"+key)
        checks.equal(proof["source_q14"], group["q14"], label+"/never solve another board branch")
        bounded = bool((values >= WIDE_BOUNDS["low"]).all() and (values <= WIDE_BOUNDS["high"]).all())
        checks.equal(proof["bounds_passed"], bounded, label+"/original action bounds")
        checks.require(bounded or not group["eligible"], label+"/out-of-bounds target cannot release")
        distance = float(np.linalg.norm(xy-values[:2]))
        checks.numeric(proof["distance_m"], distance, label+"/board-center distance", 1e-12)
        checks.require(distance > .15, label+"/board and bucket separated")
        edges = np.r_[xy-(center-half)-radius, (center+half)-xy-radius]
        footprint = proof["bucket_footprint"]
        checks.numeric(footprint["outer_radius_m"], radius, label+"/whole outer bucket radius", 0)
        checks.numeric(footprint["edge_clearance_m"], edges, label+"/all table margins", 1e-12)
        checks.numeric(footprint["minimum_edge_clearance_m"], min(edges), label+"/minimum table margin", 1e-12)
        checks.equal(footprint["passed"], bool(min(edges) >= .005-1e-12), label+"/5mm table-footprint rule")
        checks.require(footprint["passed"] or not group["eligible"], label+"/off-table bucket cannot release")
        local = targets["board_world"][:3, :3].T@(np.asarray(nominal["impact_point_world_m"])-values[:3])
        margins = np.asarray(scene["board"]["size_m"][:2])/2-np.abs(local[:2])
        point_inside = bool((margins >= .005-1e-12).all() and abs(local[2]) < 1e-9)
        checks.equal(proof["board_point_footprint_passed"], point_inside, label+"/nominal center-plane point inside board")
        checks.numeric(proof["board_point_edge_margin_m"], float(min(margins)), label+"/nominal board-edge margin", 1e-12)
        checks.require(point_inside or not group["eligible"], label+"/off-board ray cannot release")
        zero = bool((np.asarray(group["impact_offset_m"]) == 0).all())
        mode = "exact_control" if group["role"].endswith("_control") else "exact_zero_offset" if zero else "frozen_verification" if group.get("freeze_joints") else "solved_dropper_cpu"
        checks.equal(proof["source_mode"], mode, label+"/only dropper may be solved")
        checks.require(len(proof["attempts"]) <= (1 if mode != "solved_dropper_cpu" else 17), label+"/finite dropper branch budget")
        successful = []
        for branch_index, attempt in enumerate(proof["attempts"]):
            checks.equal(attempt["q14"], group["q14"], label+"/every tested board branch frozen")
            actual = gate.check(attempt["q10"], attempt["q14"], xy, values, targets)
            checks.equal(actual["reason"], attempt["checked"]["reason"], label+"/fresh independently targeted endpoint/"+str(branch_index))
            if actual["reason"] == 0:
                successful.append(attempt)
                checks.require(actual["minimum_clearance_lower_bound_m"] >= .005-1e-12, label+"/fresh actual5mm endpoint clearance")
                checks.numeric(attempt["checked"]["minimum_clearance_lower_bound_m"], actual["minimum_clearance_lower_bound_m"], label+"/actual clearance identity", 1e-10)
        checks.equal(group["eligible"], bool(successful), label+"/admission has a fresh passing endpoint")
        if successful:
            checks.equal(proof["attempts"][-1], successful[0], label+"/first passing branch selected")
            checks.equal(successful[0]["q10"], group["q10"], label+"/actual selected dropper branch")
            checks.require(proof["minimum_clearance_lower_bound_m"] >= .005-1e-12, label+"/declared strict clearance")


def rescore_impact_rows(directory, rows, scene, calibration, checks, label, trace_spec, *, assets=None):
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
            checks.equal(native_time.shape,(total_steps+1,len(ids)),label+"/native clock dimensions")
            clock_drift=check_native_clock(native_time,ticks,checks,label,synthetic=synthetic)
            for index,row in members:
                name=label+"/"+row["id"]
                checks.equal(str(ids[index]),row["id"],name+"/dense trace request binding")
                checks.numeric(row["native_clock_drift_s"],float(clock_drift[index]),name+"/reported actual native clock drift",1e-12)
                trace={"position_m":arrays["position_m"][:,index],
                       "velocity_m_s":arrays["velocity_m_s"][:,index],
                       "forces_N":arrays["forces_N"][:,index],
                       "time_s":arrays["time_s"],
                       "contact_distance_m":contact[:,index],
                       "native_time_s":native_time if native_time.ndim==1 else native_time[:,index]}
                values=np.asarray(row["values"],float)
                release=np.asarray(nominal_geometry(scene,row["bucket_xy_m"],values,row["impact_offset_m"])["release_center_m"])
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
                    max_cpu_tangency_distance_m=max((abs(r[2]) for r in refs),default=0.),native_clock_drift_s=float(clock_drift[index]))
    checks.equal(set(by_file),{str(p.relative_to(directory)) for p in directory.rglob("*.npz")},label+"/no unassigned dense native arrays")
    return traces


def audit_native(study, rows, scene, calibration, assets, checks):
    """Audit actual offset reset, full dense contacts, calibrated labels and FK."""
    import mujoco
    from .model import build_scene
    from .calibrated_score_oracle import score_trace
    cpu = mujoco.MjModel.from_xml_string(build_scene(scene, assets, solref=(.004, .15), contact_friction=.35))
    data = mujoco.MjData(cpu)
    q10 = [int(cpu.jnt_qposadr[cpu.joint(f"Arm10/joint{i}").id]) for i in range(1, 7)]
    q14 = [int(cpu.jnt_qposadr[cpu.joint(f"Arm14/joint{i}").id]) for i in range(1, 7)]
    board, flange = int(cpu.body("Arm14/impact_board").id), int(cpu.body("Arm10/link6").id)
    positions, traces = {}, {}
    for step in STEPS:
        chosen = [row for row in rows if row["substeps"] == step]
        directory = study/"native"/str(step)
        traces.update(rescore_impact_rows(directory, chosen, scene, calibration, checks, "dense impact "+str(step), replay.native_trace_spec(step), assets=assets))
        by_file = {}
        for row in chosen:
            if row["native_release"]:
                by_file.setdefault(row["native_cache_file"], []).append(row)
        for name, members in by_file.items():
            with np.load(contained(directory, name), allow_pickle=False) as archive:
                arrays = {key: archive[key] for key in archive.files}
                for row in members:
                    label = row["id"];index = row["native_cache_row"]
                    trace = {key: (value[:, index] if value.ndim >= 2 else value) for key, value in arrays.items() if key != "record_ids"}
                    actual = score_trace(scene, row["bucket_xy_m"], row["board_normal"], calibration, trace)
                    for key, expected in actual.items():
                        if key == "invalid" and expected:
                            expected = "native_physics_outside_calibrated_domain"
                        checks.require(key in row, label+"/complete independent oracle field/"+key)
                        if isinstance(expected, float):
                            checks.numeric(row[key], expected, label+"/complete independently rescored contact/"+key, 1e-8)
                        else:
                            checks.equal(row[key], expected, label+"/complete independently rescored contact/"+key)
                    targets = endpoint_targets(scene, row["bucket_xy_m"], row["values"], row["impact_offset_m"])
                    release_error = float(np.linalg.norm(trace["position_m"][0]-targets["release_center_m"]))
                    checks.require(release_error <= 1e-6, label+"/actual native reset realizes the offset within1um")
                    checks.numeric(row["native_release_target_error_m"], release_error, label+"/observed actual offset release error", 1e-12)
                    for key, expected in (("impact_basis_version", BASE_VERSION), ("impact_gap_definition", GAP_DEFINITION)):
                        checks.equal(row[key], expected, label+"/explicit actual release semantics/"+key)
                    data.qpos[:] = cpu.qpos0
                    data.qpos[q10] = row["native_initial_q10"];data.qpos[q14] = row["native_initial_q14"]
                    data.qvel[:] = 0.
                    mujoco.mj_forward(cpu, data)
                    checks.numeric(row["board_center_m"], data.xpos[board], label+"/native actual board center independently associated with measured joints", 2e-6)
                    checks.numeric(row["board_normal"], data.xmat[board].reshape(3, 3)[:, 2], label+"/actual board normal independently associated with measured joints", 2e-6)
                    drop_error = float(np.linalg.norm(data.xpos[flange]-targets["dropper_flange_world"][:3, 3]))
                    checks.require(drop_error < .002, label+"/actual float32-initialized dropper reaches OFFSET release target")
                    checks.numeric(row["pose_gate"]["dropper_error_mm"], drop_error*1000, label+"/actual offset dropper FK error", 2e-3)
                    tick = round(.350*240*step)
                    checks.numeric(arrays["time_s"][tick], .350, label+"/canonical paired350ms tick", 1e-12)
                    positions[label] = arrays["position_m"][tick, index].copy()
    return positions, traces


def audit_metadata(actual, group, scene, checks, label):
    for key in ("id", "role", "phase", "values", "bucket_xy_m", "source_record_id", "source_record_sha256", "eligible",
                "family_id", "seed_id", "impact_offset_m", "query_kind", "query_fingerprint", "translation_m",
                "source_query_fingerprint", "source_route_quality"):
        if key in group:
            checks.equal(actual[key], group[key], label+"/exact displayed input/"+key)
    values = group["values"];nominal = nominal_geometry(scene, group["bucket_xy_m"], values, group["impact_offset_m"])
    for key, expected in dict(board_xy_m=values[:2], board_height_m=values[2], tilt_deg=values[3], gap_m=values[4],
                               distance_m=float(np.linalg.norm(np.asarray(group["bucket_xy_m"])-values[:2]))).items():
        checks.equal(actual[key], expected, label+"/displayed configured geometry/"+key)
    for key in ("impact_point_world_m", "impact_to_bucket_horizontal_m", "release_center_m", "release_center_height_m"):
        checks.numeric(actual[key], nominal[key], label+"/displayed actual offset semantics/"+key, 1e-12)


def audit_fingerprints(spec, report, groups, rows, parent, checks):
    physical = xy_audit.scene_fingerprint(spec["scene"], spec["calibration"], spec["asset_sha256"])
    checks.equal(physical, parent["scene_fingerprint"], "Original physical scene fingerprint unchanged")
    checks.equal(spec["physical_scene_fingerprint"], physical, "Separate physical scene identity")
    checks.equal(report["physical_scene_fingerprint"], physical, "Display preserves physical scene identity")
    checks.equal(spec["scene_fingerprint"], scene_fingerprint(physical), "New offset-action semantics explicitly versioned")
    checks.equal(report["scene_fingerprint"], spec["scene_fingerprint"], "Display new action scene identity")
    for group in groups:
        checks.equal(group["query_fingerprint"], query_fingerprint(group, spec["scene_fingerprint"]), group["id"]+"/offset/action/source query identity")
    executions = []
    for row in rows:
        if row["native_release"]:
            expected = execution_fingerprint(row)
            checks.equal(row["execution_fingerprint"], expected, row["id"]+"/actual ball and joints execution identity")
            executions.append(expected)
        else:
            checks.equal(row["execution_fingerprint"], None, row["id"]+"/reject has no execution identity")
    checks.equal(len(set(executions)), len(executions), "Unique actual release identities")


def audit_settings(settings, checks):
    expected = dict(source_ids=list(SOURCE_IDS), families=3, forward_m=list(FORWARD), lateral_m=list(LATERAL),
                    coarse_requests=135, control_requests=24, verification_sources=5, repeats_per_resolution=3,
                    verification_requests=30, max_requests=189, user_request_cap=200, worker_wall_limit_s=3600,
                    substeps=list(STEPS), minimum_clearance_m=.005, max_drift_350ms_m=.005, impact_rule=RULE)
    for key, value in expected.items():
        checks.equal(settings[key], value, "Literal bounded impact protocol/"+key)
    checks.require("CENTER" in settings["gap_convention"] and "zero offset" in settings["gap_convention"], "Center-plane gap convention preserves zero-offset history")
    checks.require("varies" in settings["distance_convention"], "Nominal offset flight distance differs from board-center650mm")


def audit_report(report, rows, groups, verified, sources, positions, pre, post, scene, checks):
    stat = clean_audit.audit_statistics
    stat(report["counts"], rows, checks, "All requested development/control/verification shots")
    for field, key in (("phase_counts", "phase"), ("substep_counts", "substeps")):
        checks.equal(report["counts"][field], dict(Counter(str(row[key]) for row in rows)), "Complete denominator/"+field)
    checks.equal(report["control_gates"], dict(controls_pre=pre, controls_post=post), "Both independent control gates")
    checks.equal(len(report["groups"]), len(groups), "All opened geometry groups visible")
    for actual, group in zip(report["groups"], groups):
        audit_metadata(actual, group, scene, checks, group["id"])
        for step in STEPS:
            stat(actual["by_substeps"][str(step)], [row for row in rows if row["group_id"] == group["id"] and row["substeps"] == step],
                 checks, group["id"]+"/actual group outcomes/"+str(step))
    control_keys = [(phase, group["id"], group["role"], step)
                    for phase in ("controls_pre", "controls_post") for group in groups[:2] for step in STEPS]
    checks.equal([(row["phase"], row["group_id"], row["role"], row["substeps"]) for row in report["controls"]], control_keys, "All controls and resolutions displayed")
    for actual in report["controls"]:
        cohort = [row for row in rows if all(row[key] == actual[key] for key in ("phase", "group_id", "substeps"))]
        stat(actual["stats"], cohort, checks, "Exact control cohort")
    top = xy_audit.select([row for row in rows if row["phase"] == "coarse"], 10)
    checks.equal([row["record_id"] for row in report["top_settings"]], [row["id"] for row in top], "Top10 prespecified first-flight ranking")
    lookup = {group["id"]: group for group in groups}
    for actual, source in zip(report["top_settings"], top):
        audit_metadata(actual, lookup[source["group_id"]], scene, checks, source["id"])
        checks.equal(actual["group_id"], source["group_id"], source["id"]+"/actual group")
        checks.equal(actual["substeps"], source["substeps"], source["id"]+"/actual development resolution")
        stat(actual["stats"], [source], checks, source["id"]+"/actual observed trial")
        for key in ("execution_fingerprint", "minimum_mouth_clearance_mm"):
            checks.equal(actual[key], source.get(key), source["id"]+"/actual scalar/"+key)
        xy_audit.same(actual["first_flight"], source["first_flight"], checks, source["id"]+"/actual first-flight diagnostic")
    checks.equal(len(report["verification"]), len(verified), "Every selected target reported regardless of outcome")
    clean_count, capture_count = 0, 0
    for actual, group, source in zip(report["verification"], verified, sources):
        audit_metadata(actual, group, scene, checks, group["id"])
        checks.equal(actual["source_route_quality"], source.get("route_quality", 0), group["id"]+"/actual discovery quality retained")
        cohort = [row for row in rows if row["group_id"] == group["id"]]
        for step in STEPS:
            stat(actual["by_substeps"][str(step)], [row for row in cohort if row["substeps"] == step], checks, group["id"]+"/fresh repeats/"+str(step))
        replay.check_pairing(cohort, checks)
        checks.equal(actual["sensitivity"], replay.paired_sensitivity(cohort, positions), group["id"]+"/matched controls and drift")
        result = qualification(cohort, positions, controls_passed=pre["status"] == post["status"] == "passed", source_capture=replay.captured(source))
        checks.equal(actual["qualification"], result, group["id"]+"/fresh separate capture and clean qualification")
        clean_count += int(result["clean_qualified"]);capture_count += int(result["capture_qualified"])
    status = "blocked_controls" if pre["status"] != "passed" or post["status"] != "passed" else "passed" if clean_count else "capture_only" if capture_count else "failed"
    gate = dict(status=status, controls_pre_passed=pre["status"] == "passed", controls_post_passed=post["status"] == "passed",
                qualified_settings=clean_count, capture_qualified_settings=capture_count, more_ppo_started=False)
    checks.equal(report["gate"], gate, "Final independent promotion gate")
    return gate


def audit_traces(study, report, rows, groups, scene, checks):
    expected = [row for row in rows if row["phase"] == "controls_pre" and row["repeat_index"] == 0 and row["native_release"]]
    expected += xy_audit.select([row for row in rows if row["phase"] == "coarse"], 3)
    expected += [row for row in rows if row["phase"] == "verification" and row["repeat_index"] == 0 and row["native_release"]]
    checks.equal([trace["record_id"] for trace in report["trace_examples"]], [row["id"] for row in expected], "Prespecified illustrative controls/development/verification only")
    checks.require(len(expected) <= 17, "At most17 actual illustrative trajectories")
    lookup = {group["id"]: group for group in groups}
    for trace, row in zip(report["trace_examples"], expected):
        audit_metadata(trace, lookup[row["group_id"]], scene, checks, row["id"])
        for key in ("group_id", "substeps", "whole_ball_entry", "max_retention_s", "external_contact_steps", "route_success", "route_quality",
                    "rim_band_contact", "invalid", "invalid_physics", "scalar_score_audit_passed", "execution_fingerprint"):
            checks.equal(trace[key], row[key], row["id"]+"/actual illustrated native field/"+key)
        xy_audit.same(trace["first_flight"], row["first_flight"], checks, row["id"]+"/actual illustrated first flight")
        with np.load(contained(study, row["native_directory"])/row["native_cache_file"], allow_pickle=False) as archive:
            points = archive["position_m"][::row["substeps"]*4, row["native_cache_row"]]
        checks.numeric(trace["position_m"], points, row["id"]+"/actual saved native points", 0)
        checks.numeric(trace["sample_dt_s"], 1/60, row["id"]+"/actual illustrative cadence", 0)
        checks.require(bool(trace.get("selection_scope")), "Illustration selection remains explicit")


def auditor_identity(producer_source, producer_hashes, checks):
    root = Path(__file__).resolve().parents[2];own = "sircus/mjlab/impact650_verify.py";dependencies = {}
    for module in tuple(sys.modules.values()):
        name, filename = getattr(module, "__name__", ""), getattr(module, "__file__", None)
        if not filename or not (name.startswith("sircus.mjlab.") or name.startswith("sircus.isaac.")):
            continue
        path = Path(filename).resolve()
        checks.require(path.is_relative_to(root), "One explicit auditor source snapshot/"+name)
        relative = str(path.relative_to(root))
        if len(Path(relative).parts) != 3 or path.suffix != ".py" or relative == own:
            continue
        if relative.startswith("sircus/isaac/") and relative not in producer_hashes:
            continue
        checks.require(relative in producer_hashes, "Actual imported dependency declared/"+relative)
        actual = sha(path);checks.equal(actual, producer_hashes[relative], "Unchanged imported audit dependency/"+relative)
        dependencies[relative] = actual
    checks.require("sircus/mjlab/calibrated_score_oracle.py" in dependencies, "Independent calibrated capture oracle retained")
    checks.equal(sha(contained(producer_source, own)), producer_hashes[own], "Original producer checker retained")
    return dict(version=VERSION, auditor_sha256=sha(Path(__file__)), original_producer_auditor_sha256=producer_hashes[own], dependencies_sha256=dependencies)


def audit(study, source, assets, *, parent_study, parent_audit):
    import hashlib
    started, checks = time.monotonic(), AuditChecks()
    study, source, assets, parent_study, parent_audit = map(Path, (study, source, assets, parent_study, parent_audit))
    manifest = replay.check_manifest(study, checks)
    spec, report = read(study/"contract.json"), read(study/"report.json")
    unsigned = dict(spec);fingerprint = unsigned.pop("fingerprint", None)
    checks.equal(digest(unsigned), fingerprint, "Complete frozen impact development contract")
    checks.equal(spec["version"], STUDY_VERSION, "Explicit new impact action protocol")
    checks.equal(report["version"], STUDY_VERSION, "Report belongs to new action semantics")
    checks.equal((report["status"], report["phase"]), ("completed", "completed"), "Finite acquired-slot experiment completed")
    checks.equal(report["contract_fingerprint"], fingerprint, "Report exact complete contract")
    checks.equal(spec["evidence_kind"], "native_calibrated_impact650_development", "Actual native offset evidence")
    checks.equal(report["evidence_sha256"], manifest, "Report binds complete raw evidence")
    checks.require(0 <= report["elapsed_s"] <= 3600, "Completed worker remains within one-hour acquired-slot limit")
    replay.check_source_files(source, spec["source_sha256"], checks, "Immutable actual native source")
    paths = list((source/"sircus/mjlab").glob("*.py"))+[source/"sircus/mjlab/requirements.txt"]
    paths += [source/"sircus/isaac"/name for name in ("geometry.py", "rl_contract.py", "entry_quality.py", "board_route.py")]
    checks.equal(set(spec["source_sha256"]), {str(path.relative_to(source)) for path in paths}, "All actual execution/audit modules declared")
    from .learning_solve import required_asset_paths
    actual_assets = {name: sha(contained(assets.parent, name)) for name in required_asset_paths(assets)}
    checks.equal(spec["asset_sha256"], actual_assets, "Actual collision assets unchanged")
    replay.audit_packages(spec["packages"], checks)
    calibration = replay.check_calibration(study, spec["calibration"], checks)
    parent, _, parent_rows, _ = tilt_audit.audit_parent(parent_study, parent_audit, spec["input_bindings"], checks)
    for key in ("scene", "calibration", "packages", "asset_sha256"):
        checks.equal(spec[key], parent[key], "Unchanged physical parent/"+key)
    checks.equal(spec["parent_contract_fingerprint"], tilt_audit.PARENT_FP, "Exact original XY parent")
    checks.equal(report["parent_contract_fingerprint"], tilt_audit.PARENT_FP, "Report exact original XY parent")
    selected = select_families(parent_rows, checks)
    checks.equal(read(study/"source-selection.json"), selected, "Literal complete source-family records frozen")
    frozen = {"initial-groups.json", "initial-geometry.json", "initial-plan.json", "source-selection.json", "calibration.json", "verification.json"}
    checks.equal(set(spec["frozen_files_sha256"]), frozen, "All six pre-outcome inputs frozen")
    for name, expected in spec["frozen_files_sha256"].items():
        checks.equal(sha(contained(study, name)), expected, "Frozen pre-outcome bytes/"+name)
    audit_settings(spec["settings"], checks)
    checks.equal(report["settings"], spec["settings"], "Reported exact frozen protocol")
    initial, initial_proofs = read(study/"initial-groups.json"), read(study/"initial-geometry.json")
    controls = [group for group in read(parent_study/"initial-groups.json") if group["role"].endswith("_control")]
    checks.equal([group["role"] for group in controls], ["positive_control", "negative_control"], "Original ordered control pair")
    audit_initial(initial, selected, controls, checks)
    before = make_plan(initial[:2], "controls_pre", steps=STEPS, repeats=3)
    coarse = make_plan(initial[2:], "coarse", steps=(64,), repeats=1)
    after = make_plan(initial[:2], "controls_post", steps=STEPS, repeats=3)
    checks.equal(read(study/"initial-plan.json"), dict(controls_pre=before, coarse=coarse, controls_post=after), "All initial/control/coarse requests frozen")
    groups, proofs = read(study/"groups.json"), read(study/"geometry.json")
    checks.equal(groups[:137], initial, "Initial queries never changed after outcomes")
    checks.equal(proofs[:137], initial_proofs, "Initial geometry proofs unchanged")
    audit_geometry(groups, proofs, spec["scene"], assets, checks)
    rows = replay.read_records(study/"records.jsonl")
    checks.require(12 <= len(rows) <= MAX_REQUESTS < 200, "All requests including rejects stay below200 cap")
    checks.equal([row["id"] for row in rows[:12]], [row["id"] for row in before], "Controls precede every impact query")
    pre = clean_audit.control_gate(rows[:12])
    checks.equal(read(study/"initial-control-gate.json"), pre, "Independent initial controls")
    candidates = [row for row in rows if row["phase"] == "coarse"]
    chosen = selected_verification(candidates) if pre["status"] == "passed" else []
    verified = read(study/"verification-groups.json")
    audit_verification(verified, chosen, checks)
    checks.equal(read(study/"verification-selection.json"), chosen, "Exact family-diverse observed records selected")
    verification = make_plan(verified, "verification", steps=STEPS, repeats=3)
    checks.equal(read(study/"verification-plan.json"), verification, "All selected repeats frozen before execution")
    checks.equal(groups, initial+verified, "No undeclared family, refinement or retry groups")
    if pre["status"] == "passed":
        stage = read(study/"verification-stage.json")
        lines = (study/"records.jsonl").read_bytes().splitlines(keepends=True)
        prefix = hashlib.sha256(b"".join(lines[:12+135])).hexdigest()
        checks.equal(stage["prior_records_sha256"], prefix, "Selection depends only on exact preceding coarse evidence")
        names = {"verification-selection.json", "verification-groups.json", "verification-plan.json"}
        checks.equal(set(stage["files_sha256"]), names, "All conditional verification files bound")
        for name, expected in stage["files_sha256"].items():
            checks.equal(sha(contained(study, name)), expected, "Frozen selected verification bytes/"+name)
        checks.equal(stage["geometry_sha256"], digest(proofs), "All final selected endpoint proofs frozen before repeats")
    else:
        checks.require(not (study/"verification-stage.json").exists(), "Failed initial controls open no conditional stage")
    expected = before+coarse+verification+after if pre["status"] == "passed" else before
    clean_audit.audit_native_rows(rows, expected, groups, checks)
    audit_fingerprints(spec, report, groups, rows, parent, checks)
    replay.audit_batches(study, rows, checks)
    post = clean_audit.control_gate([row for row in rows if row["phase"] == "controls_post"])
    if pre["status"] != "passed":post["status"] = "not_run"
    checks.equal(read(study/"final-control-gate.json"), post, "Independent final controls")
    actual_steps = {row["substeps"] for row in rows if row["native_release"]}
    checks.require(actual_steps in (set(), set(STEPS)), "Both resolutions allocated or wholly rejected controls")
    if actual_steps:
        options = read(study/"effective-options.json")
        replay.audit_options(options, checks)
        replay.audit_compiled(study, spec["scene"], assets, calibration, options, checks)
        positions, traces = audit_native(study, rows, spec["scene"], calibration, assets, checks)
    else:
        checks.require(not (study/"effective-options.json").exists(), "No invented native options")
        checks.require(not list((study/"native").rglob("*.npz")), "No invented native observations")
        positions, traces = {}, {}
    xy_audit.audit_first_flight(study, rows, spec["scene"], calibration, checks)
    gate = audit_report(report, rows, groups, verified, chosen, positions, pre, post, spec["scene"], checks)
    audit_traces(study, report, rows, groups, spec["scene"], checks)
    identity = auditor_identity(source, spec["source_sha256"], checks)
    return dict(version=VERSION, status="passed", contract_fingerprint=fingerprint, parent_contract_fingerprint=tilt_audit.PARENT_FP,
                scene_fingerprint=spec["scene_fingerprint"], physical_scene_fingerprint=spec["physical_scene_fingerprint"],
                checks_count=checks.count, checks=checks.entries, failures=[], auditor_sha256=sha(Path(__file__)), auditor_identity=identity,
                producer_source_sha256=spec["source_sha256"], report_sha256=sha(study/"report.json"),
                evidence_manifest_sha256=sha(study/"evidence-manifest.json"), requested_trials=len(rows), native_releases=len(traces),
                gate=gate, elapsed_s=time.monotonic()-started,
                limitations=["Three observed development families; no learned policy, held-out generalization or workspace coverage.",
                             "Offsets describe nominal center-plane drop-ray intersections, not measured finite-ball contacts; impact-to-bucket distance varies.",
                             "Calibrated simulated contacts and held endpoints only; no noise, grasp/transit, actuator, perception or physical validation."])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("study", "source", "assets", "parent-study", "parent-audit", "output"):
        parser.add_argument("--"+name, type=Path, required=True)
    args = parser.parse_args()
    if args.output.resolve().is_relative_to(args.study.resolve()):
        raise ValueError("Independent proof must be outside the immutable native namespace")
    try:
        result = audit(args.study, args.source, args.assets, parent_study=args.parent_study, parent_audit=args.parent_audit)
    except (EvidenceError, ValueError, KeyError, OSError) as error:
        result = dict(version=VERSION, status="failed", failures=[str(error)])
        args.output.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
        raise
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    print(json.dumps({key: result[key] for key in ("status", "checks_count", "requested_trials", "native_releases", "gate")}))


if __name__ == "__main__":
    main()


