"""One frozen relative-bank choice, followed by one native verification throw."""

import argparse
import fcntl
import importlib.metadata
import json
from pathlib import Path
import numpy as np

from .interface import DEFAULT_SCENE, validate_request
from .learning_solve import (
    validate_artifacts, sha256, _verify_files, _assert_gpu_free,
    _bound_child_lifetime, _write,
)
from .learning_contract import digest


def validate_bank(study, parent_report, parent_checkpoint, repository=None, load_checkpoint=None):
    root=Path(repository) if repository else Path(__file__).resolve().parents[2]
    report=json.loads((study/"report.json").read_text())
    if report.get("status")!="completed" or not report.get("frozen_evaluation"):
        raise RuntimeError("Relative-bank study and frozen evaluation must complete first")
    if sha256(study/"selected.pt")!=report["checkpoint_sha256"]:
        raise RuntimeError("Relative-bank checkpoint checksum changed")
    files=("bank_policy.py","bank_geometry.py","bank_study.py")
    hashes=report.get("source_sha256",{})
    _verify_files(root/"sircus/mjlab",{k:hashes[k] for k in files if k in hashes},files)
    for key,value in report.get("versions",{}).items():
        if importlib.metadata.version(key)!=value:
            raise RuntimeError("Relative-bank runtime dependency changed: "+key)
    if load_checkpoint is None:
        import torch
        load_checkpoint=lambda path:torch.load(path,map_location="cpu",weights_only=True)
    cp=load_checkpoint(study/"selected.pt")
    spec=dict(cp["contract"]);fp=spec.pop("fingerprint")
    if digest(spec)!=fp or fp!=report["contract_fingerprint"]:
        raise RuntimeError("Relative-bank scene contract changed")
    if cp["contract"]!=json.loads((study/"contract.json").read_text()):
        raise RuntimeError("Bank checkpoint and saved contract differ")
    if spec["parent_contract_fingerprint"]!=parent_report["contract_fingerprint"] or spec["parent_checkpoint_sha256"]!=parent_report["checkpoint_sha256"]:
        raise RuntimeError("Relative-bank model uses another parent experiment")
    if spec["scene"]!=parent_checkpoint["contract"]["scene"]:
        raise RuntimeError("Relative-bank geometry differs from evaluated scene")
    bank=dict(cp["bank"]);bank_fp=bank.pop("sha256")
    if digest(bank)!=bank_fp or bank_fp!=spec["proposal_bank_sha256"]:
        raise RuntimeError("Relative candidate bank changed")
    if report.get("policy_weight_sha256")!=cp["weight_sha256"]:
        raise RuntimeError("Relative-bank frozen evaluation uses another policy")
    if cp.get("policy_class")!="relative_bank_categorical" or report["frozen_evaluation"].get("policy_weight_sha256")!=cp["weight_sha256"]:
        raise RuntimeError("Wrong bank policy class or frozen evaluation weights")
    return report,cp


def run(a):
    parent,pcp,_,_=validate_artifacts(a.parent_study,a.assets)
    report,cp=validate_bank(a.study,parent,pcp)
    request=validate_request(json.loads(a.request.read_text()),DEFAULT_SCENE)
    if request["mode"]!="policy" or any(v is not None for v in request["constraints"].values()):
        raise RuntimeError("Relative-bank learned policy supports free positions only; use continuous policy or explicit search for locks")
    import torch
    from .bank_geometry import build_candidates,audit_candidates
    from .bank_policy import Policy,predict,weight_hash
    torch.set_num_threads(2)
    xy=np.asarray([request["bucket_xy"]],dtype=float)
    values,metadata=build_candidates(cp["bank"],xy,cp["contract"]["scene"])
    geometry=audit_candidates(cp["contract"]["scene"],a.assets,xy,values)
    # audit_candidates returns mask + native endpoints; no ball is simulated.
    if isinstance(geometry,tuple):
        mask,detail=geometry
    else:
        mask,detail=geometry["mask"],geometry
    policy=Policy();policy.load_state_dict(cp["state"]);policy.eval()
    if weight_hash(policy)!=cp["weight_sha256"]:
        raise RuntimeError("Relative-bank tensor hash changed")
    with torch.no_grad():
        answer=predict(policy,torch.tensor(xy,dtype=torch.float32),torch.tensor(values,dtype=torch.float32),torch.tensor(mask,dtype=torch.bool),sample=False)
    if not bool(answer["eligible"][0]):
        return _write(a.output,dict(request_id=request["request_id"],bucket_xy_m=xy[0].tolist(),method="bank_policy",success=False,
            message="No endpoint-feasible candidate in this finite learned bank; explicit search may explore other settings",trajectory=[],
            metrics=dict(candidate_count=int(values.shape[1]),feasible_candidates=0),
            provenance=dict(search_at_inference=False,proposal_bank_sha256=cp["bank"]["sha256"],endpoint_configuration_only=True)))
    index=int(answer["indices"][0]);chosen=values[:,index,:]
    with open(a.gpu_lock,"r") as lock:
        try: fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError: raise RuntimeError("4090 is busy; no existing workload was interrupted") from None
        _assert_gpu_free()
        from .learning_env import ShotEnvironment
        from .bank_study import run_bank_rows
        env=ShotEnvironment(cp["contract"]["scene"],a.assets,n=1)
        rows=run_bank_rows(env,xy,chosen,np.asarray(detail["q10"])[:,index],np.asarray(detail["q14"])[:,index],np.array([True]),record=True)
        if isinstance(rows,tuple):rows=rows[0]
        row=rows[0]
        if not row.get("scalar_score_audit_passed"):
            raise RuntimeError("Native bank replay failed independent scorer audit")
        success=bool(row["route_success"] and not row["invalid"])
        configuration=dict(board_center=row["board_center_m"],board_normal=row["board_normal"],
             board_quaternion_wxyz=row["board_quaternion_wxyz"],release_center=row["release_ball_center_m"],
             dropper_center=row["release_ball_center_m"],q10=row["q10"],q14=row["q14"],
             link_positions10=row["link_positions10"],link_positions14=row["link_positions14"],tilt_deg=row["values"][3],gap_m=row["values"][4])
        _write(a.output,dict(request_id=request["request_id"],bucket_xy_m=xy[0].tolist(),method="bank_policy",success=success,
             message="Measured native capture from one frozen learned-bank choice" if success else "Frozen bank choice did not capture; it remains a miss, with no hidden retry",configuration=configuration,
             trajectory=[dict(time=(i+1)*row["trajectory_dt_s"],ball=p) for i,p in enumerate(row["trajectory_m"])],
             metrics=dict(board_bounces=row["board_contact_episodes"],retention_s=row["max_retention_s"],quality=row["route_quality"],
                          invalid=row["invalid"],attempts=1,feasible_candidates=int(np.asarray(mask).sum()),candidate_count=int(values.shape[1]),
                          eligible_candidates=int(np.asarray(mask).sum()),native_candidate_count=int(values.shape[1]),tilt_deg=row["values"][3],scalar_score_audit_passed=True),
             provenance=dict(engine="mjlab MuJoCo Warp",device="RTX4090",search_at_inference=False,endpoint_configuration_only=True,
                             policy_weight_sha256=cp["weight_sha256"],checkpoint_sha256=report["checkpoint_sha256"],
                             contract_fingerprint=report["contract_fingerprint"],proposal_bank_sha256=cp["bank"]["sha256"],
                             candidate_index=index,candidate_metadata=metadata[index],selection_probability=float(answer["probability"][0]),
                             source_scope="Learned choice among fixed train-derived relative proposals with a geometry-only IK mask")))


if __name__=="__main__":
    p=argparse.ArgumentParser()
    for key in ("study","parent-study","assets","gpu-lock","request","output"):p.add_argument("--"+key,type=Path,required=True)
    a=p.parse_args();_bound_child_lifetime();run(a)
