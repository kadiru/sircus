"""Read-only validation of every signed continuous feasibility survey request.

Rebuilds the seeded Sobol schedule, physical action geometry, conservative bounds,
FK residuals, provisional clearances and aggregate denominators. It does not
rerun numerical IK or convert unresolved endpoints to impossibility claims.
"""

import argparse
import hashlib
import json
import math
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation
from scipy.stats import qmc

from continuous_feasibility import (VERSION, LOW, HIGH, ACTION_NAMES,
    approximate_clearance, canonical_hash, geometry_check, split_for_index)
from expanded_pose_menu import endpoint_targets, outer_bounds
from prepare_assets import chain, fk

SCHEMA = "sircus-continuous-feasibility-audit-v1"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def close(actual, expected):
    if isinstance(expected, dict):
        return isinstance(actual, dict) and set(actual) == set(expected) and all(close(actual[k],v) for k,v in expected.items())
    if isinstance(expected, list):
        return isinstance(actual, list) and len(actual) == len(expected) and all(close(a,b) for a,b in zip(actual,expected))
    if isinstance(expected, (float,int)) and not isinstance(expected,bool):
        return isinstance(actual,(float,int)) and not isinstance(actual,bool) and math.isfinite(actual) and abs(actual-expected) <= 1e-8
    return actual == expected


def verify_fit(target, fit, links, normal_only):
    """Compute residuals from saved joints, not the producer's solver success flag."""
    q = np.asarray(fit["q"],dtype=float)
    low,high = np.array([l[2] for l in links]),np.array([l[3] for l in links])
    require(q.shape == (6,) and np.isfinite(q).all() and (q>=low).all() and (q<=high).all(),"Invalid saved feasibility joints")
    current=fk(q,links)
    angular = current[:3,2]-target[:3,2] if normal_only else Rotation.from_matrix(target[:3,:3]@current[:3,:3].T).as_rotvec()
    residual=np.r_[current[:3,3]-target[:3,3],.25*angular]
    norm=float(np.linalg.norm(residual))
    require(abs(norm-fit["residual"])<=1e-8,"Saved feasibility residual changed")
    require(bool(fit["solved"]) == (norm<1e-4),"Saved solved flag disagrees with independent FK")
    require(abs(float(np.linalg.norm(residual[:3]))*1000-fit["position_error_mm"])<=1e-5,"Saved feasibility position error changed")
    return norm


def audit(run, producer, assets):
    run,producer,assets = map(Path,(run,producer,assets))
    spec=json.loads((run/"contract.json").read_text())
    result=json.loads((run/"feasibility.json").read_text())
    provenance=json.loads((run/"run-provenance.json").read_text())
    for name,payload in (("contract",spec),("result",result)):
        require(canonical_hash({k:v for k,v in payload.items() if k!="fingerprint"}) == payload.get("fingerprint"), f"Changed {name} fingerprint")
    require(spec["version"]==VERSION and result["schema"]=="sircus-continuous-feasibility-result-v1","Unknown feasibility protocol")
    require(result["spec_fingerprint"]==spec["fingerprint"] and result["source_fingerprint"]==spec["source_fingerprint"],"Feasibility source identity changed")
    require(spec["action_low"]==LOW.tolist() and spec["action_high"]==HIGH.tolist() and spec["action_names"]==list(ACTION_NAMES),"Feasibility domain changed")
    require(provenance["cuda_visible_devices"]=="" and provenance["samples_per_cell"]==spec["samples_per_cell"],"CPU survey provenance changed")
    for name,sha in provenance["source_sha256"].items():
        require(Path(name).name==name and name.endswith(".py"),"Unsafe producer filename")
        require(digest(producer/name)==sha,f"Frozen producer source changed: {name}")
    for name,sha in result["asset_sha256"].items():
        require(Path(name).name==name and name in ("dropper.urdf","board_arm.urdf","poses.json"),"Unsafe asset filename")
        require(digest(assets/name)==sha and provenance["assets"][name]==sha,f"Survey asset changed: {name}")
    links={name:chain(ET.parse(assets/file).getroot()) for name,file in (("Arm10","dropper.urdf"),("Arm14","board_arm.urdf"))}
    require(len(spec["cells"])==35 and len({c["id"] for c in spec["cells"]})==35 and len({tuple(c["bucket_xy_m"]) for c in spec["cells"]})==35,"Changed distinct bucket cells")
    n=spec["samples_per_cell"]
    require(type(n) is int and n>0 and n&(n-1)==0,"Invalid Sobol sample count")
    require(len(result["records"])==35*n==spec["total_requests"]==result["requests"],"Missing feasibility requests")
    solved,accepted=0,0
    reasons=Counter()
    for i,cell in enumerate(spec["cells"]):
        expected_split=split_for_index(i)
        require(spec["split_assignment"][cell["id"]]==expected_split,"Spatial split changed")
        unit=qmc.Sobol(d=5,scramble=True,seed=spec["seed"]+i).random_base2(int(math.log2(n)))
        values=LOW+unit*(HIGH-LOW)
        group=result["records"][i*n:(i+1)*n]
        for j,(row,value) in enumerate(zip(group,values)):
            require((row["cell_id"],row["cell_index"],row["proposal"],row["split"])==(cell["id"],i,j,expected_split),"Changed/missing Sobol request order")
            action=row["action"]
            require(action["id"]==f"sobol-{i:02d}-{j:04d}" and np.array_equal([action[k] for k in ACTION_NAMES],value),"Changed sampled action; projection/resampling is forbidden")
            offset,radius,height,gap,tilt=map(float,value)
            reference=np.array([.67,-.10])-cell["bucket_xy_m"]
            bearing=math.atan2(reference[1],reference[0])+offset
            require(abs(action["board_x_m"]-(cell["bucket_xy_m"][0]+radius*math.cos(bearing)))<=1e-12 and abs(action["board_y_m"]-(cell["bucket_xy_m"][1]+radius*math.sin(bearing)))<=1e-12,"Changed board bearing/radius geometry")
            require(abs(action["tilt_deg"]-math.degrees(tilt))<=1e-12,"Changed tilt units")
            geometry=geometry_check(spec["scene"],cell,action)
            require(close(row["geometry"],geometry),"Changed geometric gate evidence")
            expected_reason=geometry["reasons"][0] if not geometry["accepted"] else None
            if expected_reason is None:
                targets=endpoint_targets(spec["scene"],cell,action)
                bound=outer_bounds(targets,links)
                require(close(row["outer_bounds"],bound),"Changed outer reach bound")
                if not bound["eligible"]:
                    expected_reason="outer_link_length_bound"
                else:
                    for name in ("Arm10","Arm14"):
                        require(name in row["ik"],"Missing attempted endpoint IK")
                        norm=verify_fit(targets[name],row["ik"][name],links[name],name=="Arm14")
                        solved+=int(norm<1e-4)
                        if norm>=1e-4:
                            expected_reason="endpoint_ik_unresolved_"+name
                            break
                    if expected_reason is None:
                        clearance=approximate_clearance(spec["scene"],links,row["ik"])
                        require(close(row["approximate_clearance"],clearance),"Changed endpoint clearance evidence")
                        expected_reason=clearance["reasons"][0] if not clearance["accepted"] else None
            require(row["reason"]==expected_reason and row["accepted"]==(expected_reason is None),"Changed admission/rejection label")
            accepted+=int(row["accepted"])
            reasons[expected_reason or "endpoint_proxy_qualified"]+=1
        cell_summary=result["cells"][i]
        good=[r for r in group if r["accepted"]]
        expected=dict(cell_id=cell["id"],cell_index=i,bucket_xy_m=cell["bucket_xy_m"],split=expected_split,
                      requests=n,qualified=len(good),reasons=dict(Counter(r["reason"] or "endpoint_proxy_qualified" for r in group)),
                      qualified_ranges={k:[min(r["action"][k] for r in good),max(r["action"][k] for r in good)] for k in ACTION_NAMES} if good else {})
        require(close(cell_summary,expected),"Changed per-cell denominator or range")
    require(accepted==result["qualified"],"Changed qualified total")
    return dict(schema=SCHEMA,status="passed",requests=35*n,qualified=accepted,independently_checked_solved_endpoints=solved,
                cells_with_qualified=sum(c["qualified"]>0 for c in result["cells"]),reasons=dict(reasons),
                spec_fingerprint=spec["fingerprint"],result_fingerprint=result["fingerprint"],
                contract_sha256=digest(run/"contract.json"),feasibility_sha256=digest(run/"feasibility.json"),
                producer_provenance_sha256=digest(run/"run-provenance.json"),producer_module_sha256=provenance["source_sha256"]["continuous_feasibility.py"],
                scope="All sampled requests, immutable producer/assets, decoded actions, FK residuals, geometric/proxy admission and denominators verified. No IK search rerun, native settling or collision-path certificate.")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run",type=Path,required=True)
    parser.add_argument("--producer",type=Path,required=True)
    parser.add_argument("--assets",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists(): parser.error("Choose a new audit output; preserve prior evidence")
    value=audit(args.run,args.producer,args.assets)
    with args.output.open("x") as f: json.dump(value,f,allow_nan=False)
    print(json.dumps(value),flush=True)


if __name__=="__main__": main()
