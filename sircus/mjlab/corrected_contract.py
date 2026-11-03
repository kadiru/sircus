"""Explicit corrected-contact, strict-clearance spatial-development contract.

Old geometry, material and scoring definitions remain immutable inputs. Historical
success labels are neither qualifications nor targets in this protocol.
"""

import hashlib
import json
from pathlib import Path

import numpy as np

from .learning_contract import digest

VERSION="sircus-corrected64-spatial-development-v1"
CATALOGUE_VERSION="sircus-corrected64-confirmed-catalogue-v1"
SEED=20261127
NOISE=np.asarray([.001,.001,.002,.35,.002])
PINNED_PACKAGES={"mjlab":"1.6.0","mujoco":"3.11.0","mujoco-warp":"3.11.0",
    "torch":"2.7.0","warp-lang":"1.14.0","numpy":"1.26.4","ml-dtypes":"0.5.4"}
PHYSICS=dict(version="sircus-articulated-passive-contact-15360hz-v1",board_mode="articulated",
    gravity_mode="passive_only",substeps=64,native_dt_s=1/15360,scoring_dt_s=1/240,
    horizon_s=4.,solref=[.004,.15],contact_friction=.35,
    solver="newton",solver_iterations=100,solver_tolerance=1e-8,
    cone="pyramidal",integrator="implicitfast",
    arm_control="Prescribed joint endpoints reset before each native step; no learned grasp/transit/actuator dynamics",
    scorer="Unchanged board rebound + whole sphere entry + continuous1s retention; external assistance disqualifies route; bucket rim assistance counts")


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_packages(versions):
    for name,expected in PINNED_PACKAGES.items():
        actual=versions.get(name,"")
        # CUDA local build tag is recorded exactly, while torch's public version
        # follows the existing requirements pin.
        if (actual.split("+")[0] if name=="torch" else actual)!=expected:
            raise RuntimeError(f"Pinned native dependency differs: {name}")
    return dict(versions)


def load_fingerprinted(path,key="fingerprint"):
    value=json.loads(Path(path).read_text());payload=dict(value);expected=payload.pop(key,None)
    if expected is None or digest(payload)!=expected:raise RuntimeError(f"Changed artifact fingerprint: {Path(path).name}")
    return value


def grid_layouts():
    return [dict(id=f"dev-grid-{row:02d}-{col:02d}",row=row,column=col,role="development_then_train",
        bucket_xy_m=[float(x),float(y)]) for row,y in enumerate(np.linspace(-.65,.25,5))
        for col,x in enumerate(np.linspace(.1,1.24,7))]


def split_coordinates(excluded,validation_count=30,test_count=100):
    """Freeze future spatial splits before current grid outcomes; no test search."""
    forbidden=np.asarray(excluded,dtype=float).reshape(-1,2);groups={}
    for role,count,seed in (("validation",validation_count,SEED+101),("test",test_count,SEED+102)):
        rng=np.random.default_rng(seed);chosen=[]
        for _ in range(100000):
            if len(chosen)==count:break
            p=rng.uniform([.1,-.65],[1.24,.25])
            previous=np.r_[forbidden,np.asarray(chosen).reshape(-1,2)]
            if len(previous) and np.linalg.norm(previous-p,axis=1).min()<.005:continue
            chosen.append(p.tolist())
        if len(chosen)!=count:raise RuntimeError("Could not draw bounded separated future spatial split")
        groups[role]=chosen;forbidden=np.r_[forbidden,np.asarray(chosen)]
    groups["train"]=[r["bucket_xy_m"] for r in grid_layouts()]
    return groups


def geometry_hint(row,namespace):
    """Retain only geometry/provenance, deliberately discard all old outcomes."""
    values=np.asarray(row.get("values",[]),float)
    q10=np.asarray(row.get("q10",[]),float);q14=np.asarray(row.get("q14",[]),float)
    bucket=np.asarray(row.get("bucket_xy_m",[]),float)
    if values.shape!=(5,) or q10.shape!=(6,) or q14.shape!=(6,) or bucket.shape!=(2,) or not all(np.isfinite(a).all() for a in (values,q10,q14,bucket)):
        return None
    return dict(source_id=row["id"],source_namespace=namespace,source_record_sha256=digest(row),
        values=values.tolist(),q10=q10.tolist(),q14=q14.tolist(),bucket_xy_m=bucket.tolist(),
        use="Geometry continuation seed only; no historical outcome inherited")


def diverse_selection(values,eligible,metadata,budget=64,seed=SEED):
    """Farthest-point normalized geometry; stable seeded tie ordering, no outcomes."""
    from .expanded_actions import WIDE_BOUNDS
    values=np.asarray(values,float);eligible=np.asarray(eligible,bool)
    if values.shape!=(len(eligible),5) or len(metadata)!=len(values) or budget<1:raise ValueError("Invalid diverse selection inputs")
    ids=np.flatnonzero(eligible)
    if not len(ids):return []
    rng=np.random.default_rng(seed);ids=ids[rng.permutation(len(ids))]
    scaled=(values-np.asarray(WIDE_BOUNDS["low"]))/ (np.asarray(WIDE_BOUNDS["high"])-WIDE_BOUNDS["low"])
    chosen=[int(ids[0])];distance=np.linalg.norm(scaled[ids]-scaled[chosen[0]],axis=1)
    while len(chosen)<min(budget,len(ids)):
        distance[np.isin(ids,chosen)]=-1
        index=int(np.argmax(distance));selected=int(ids[index]);chosen.append(selected)
        distance=np.minimum(distance,np.linalg.norm(scaled[ids]-scaled[selected],axis=1))
    return chosen


def reject_row(request,reason,*,phase,geometry_cache_file=None,geometry_cache_row=None):
    return dict(id=request["id"],cell_id=request["cell_id"],candidate_id=request["candidate_id"],phase=phase,
        values=request["values"],bucket_xy_m=request["bucket_xy_m"],native_release=False,
        invalid=reason,route_success=False,route_quality=0,complete_success=False,success=False,
        board_bounce=False,whole_ball_entry=False,max_retention_s=0.,single_bounce_success=False,
        geometry_cache_file=geometry_cache_file,geometry_cache_row=geometry_cache_row)


def qualification(nominal,fresh,transit):
    """Full attempt denominators; geometric rejects remain failed trials."""
    return bool(transit.get("passed") and len(nominal)==3 and len(fresh)==20 and
        all(r.get("native_release") and r.get("route_success") for r in nominal) and
        sum(bool(r.get("native_release") and r.get("route_success")) for r in fresh)>=18)
