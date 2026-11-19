"""Independent, CPU-only reconstruction of the immutable V4 shot commands.

Importing this module loads no model or simulator. The explicit checker starts
a fresh CPU reader so the original producer modules and checkpoint bindings
cannot be replaced by the new evaluator's module cache. No optimizer is created
or resumed. Tests substitute the process runner and fabricated readouts only.
"""

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile

SCHEMA = "sircus-v4-independent-model-check-v1"
REQUEST_SCHEMA = "sircus-v4-model-check-request-v1"
V4_SOURCE = "a8c6955f25377d5e404e710c5b685b2fc82725e3b1379b86f285fb31f0d358c2"
RUNTIME = dict(python="3.11.13",torch="2.7.0+cu128",numpy="1.26.0",scipy="1.15.3")
REAL_FILES = {
    "original_manifest":"16c7c5edca352b192794505d5433643211a7bf16aca0cc631ff6409183340c47",
    "original_learning_result":"0ab677a89142be6c0cf423194f2720493f5c18939e7ba5fb9a3f93872e408a86",
    "original_native_binding":"2c9d8431feaf79a1ba37de743c3a556b7a1ba973fdc49b3698afd4f1dea5d8ca",
    "old_assessment_plan":"88f0b350a9de5a025237d12cad8ad5da5ea40b47718aa8d4aa4a407ccb35b94f",
    "parent_checkpoint":"57d7658e9ed2f72ad09cf4e8523ab497ccbe0708364f1b8ec4a4c5c1aef1fe9f",
    "final_checkpoint":"15020a948d295fe5e7258a732145b89aa61fda2c385a0d6966a9e4cc7e81bd41",
}
ROLES = frozenset((*REAL_FILES,"prepared_plan"))


def require(value,message):
    if not value:raise ValueError(message)


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _json(raw):
    def pairs(items):
        value={}
        for key,item in items:
            require(key not in value,"Duplicate model-check JSON key");value[key]=item
        return value
    return json.loads(raw,object_pairs_hook=pairs,
        parse_constant=lambda _:(_ for _ in ()).throw(ValueError("Nonfinite model-check JSON")))


def _sha(value):return type(value) is str and re.fullmatch(r"[0-9a-f]{64}",value) is not None


def _path(raw,*,file=False):
    require(type(raw) is str,"Expected model-check path")
    p=Path(raw)
    require(p.is_absolute() and str(p)==str(p.resolve()) and not any(x.is_symlink() for x in (p,*p.parents)),
        "Linked or noncanonical model-check path")
    if file:require(p.is_file(),"Missing model-check input")
    return p


def source_hash(directory):
    directory=Path(directory)
    require(directory.is_dir() and not directory.is_symlink(),"Missing model source directory")
    files=sorted(directory.glob("*.py"))
    require(files and all(p.is_file() and not p.is_symlink() for p in files),"Linked model source member")
    return canonical_hash({p.name:digest(p) for p in files})


def _verify_request(request):
    require(type(request) is dict and set(request)=={"schema","synthetic","files","original_source_dir",
        "original_source_sha256","evaluation_source_sha256","protected_roots","receipt_path","expected_runtime","fingerprint"},
        "Model-check request fields changed")
    require(request["schema"]==REQUEST_SCHEMA and type(request["synthetic"]) is bool
        and request["fingerprint"]==canonical_hash({k:v for k,v in request.items() if k!="fingerprint"}),
        "Model-check request fingerprint or provenance changed")
    require(type(request["files"]) is dict and set(request["files"])==ROLES,"Model-check input roles changed")
    paths={};before={}
    for role,record in request["files"].items():
        require(type(record) is dict and set(record)=={"path","sha256"} and _sha(record["sha256"]),"Malformed model-check input")
        paths[role]=_path(record["path"],file=True)
        before[role]=digest(paths[role])
        require(before[role]==record["sha256"],"Model-check input bytes changed: "+role)
    old=_path(request["original_source_dir"])
    require(_sha(request["original_source_sha256"]) and source_hash(old)==request["original_source_sha256"],
        "Original producer source changed")
    require(_sha(request["evaluation_source_sha256"]) and source_hash(Path(__file__).resolve().parent)==request["evaluation_source_sha256"],
        "Evaluation reader source changed")
    require(type(request["expected_runtime"]) is dict and set(request["expected_runtime"])==set(RUNTIME),"Pinned CPU runtime is missing")
    if not request["synthetic"]:
        require(request["original_source_sha256"]==V4_SOURCE and request["expected_runtime"]==RUNTIME
            and all(before[k]==v for k,v in REAL_FILES.items()),"Original V4 model/source/runtime identity changed")
    target=_path(request["receipt_path"])
    require(not target.exists() and target.parent.is_dir(),"Model receipt must use an untouched output")
    require(type(request["protected_roots"]) is list and request["protected_roots"],"Protected model/evidence roots are required")
    protected=[_path(x) for x in request["protected_roots"]]
    require(all(not target.is_relative_to(p) and not p.is_relative_to(target) for p in
        (*protected,old,Path(__file__).resolve().parent,*(p.parent for p in paths.values()))),
        "Model receipt overlaps protected model/evidence/source")
    require(target not in paths.values(),"Model receipt cannot replace an input")
    plan=_json(paths["prepared_plan"].read_bytes());original=_json(paths["old_assessment_plan"].read_bytes())
    require(plan.get("original_plan")==original and plan.get("fingerprint")==canonical_hash({k:v for k,v in plan.items() if k!="fingerprint"})
        and plan.get("synthetic") is request["synthetic"] and plan.get("requests")==210
        and plan.get("bindings",{}).get("evaluation_source_sha256")==request["evaluation_source_sha256"],
        "Prepared model plan differs from original evidence or evaluator identity")
    require(plan.get("final_checkpoint_sha256")==REAL_FILES["final_checkpoint"]
        and plan.get("parent_checkpoint_sha256")==REAL_FILES["parent_checkpoint"],"Wrong V4 model or parent")
    return paths,old,target,plan,original,before


def _verify_suffix(plan,reconstructed):
    """Independently preserve every shot field while changing outer IDs only."""
    require(plan["original_plan"]==reconstructed and len(reconstructed["choices"])==420
        and len(plan["choices"])==210 and len(plan["cohorts"])==6,"Frozen commands were not independently reconstructed")
    for index,(choice,old) in enumerate(zip(plan["choices"],reconstructed["choices"][210:])):
        expected=dict(copy.deepcopy(old),original_request_id=old["request_id"],original_request_index=old["request_index"],
            original_cohort=old["cohort"],request_id=f"test-evaluation-{index:06d}",request_index=index,cohort=index//35)
        require(choice==expected and choice["split"]=="test","A final TEST command changed after model freeze")


def validate_model_receipt(receipt,plan):
    expected="passed_synthetic" if plan.get("synthetic") is True else "passed"
    require(type(receipt) is dict and receipt.get("schema")==SCHEMA and receipt.get("status")==expected
        and receipt.get("synthetic") is plan.get("synthetic")
        and receipt.get("fingerprint")==canonical_hash({k:v for k,v in receipt.items() if k!="fingerprint"}),
        "Independent model receipt identity or provenance changed")
    require(receipt.get("plan_fingerprint")==plan["fingerprint"]
        and receipt.get("original_plan_fingerprint")==plan["original_plan"]["fingerprint"]
        and receipt.get("final_checkpoint_sha256")==plan["final_checkpoint_sha256"]
        and receipt.get("parent_checkpoint_sha256")==plan["parent_checkpoint_sha256"]
        and receipt.get("model_hashes")==plan["original_plan"]["model_hashes"]
        and receipt.get("evaluation_source_sha256")==plan["bindings"]["evaluation_source_sha256"]
        and receipt.get("original_source_sha256")==plan["bindings"]["producer_source_sha256"],
        "Independent model receipt does not bind both frozen models and sources")
    require(receipt.get("cpu_only") is True and receipt.get("cuda_visible_devices")==""
        and receipt.get("source_and_input_bytes_unchanged") is True
        and type(receipt.get("original_guard_source_sha256")) is dict and receipt["original_guard_source_sha256"]
        and all(type(k) is str and _sha(v) for k,v in receipt["original_guard_source_sha256"].items())
        and type(receipt.get("original_native_binding")) is dict
        and receipt["original_native_binding"].get("fingerprint")==canonical_hash({k:v for k,v in receipt["original_native_binding"].items() if k!="fingerprint"})
        and receipt["original_native_binding"].get("source_sha256")==receipt["original_source_sha256"]
        and receipt["original_native_binding"].get("guard_source_sha256")==receipt["original_guard_source_sha256"]
        and type(receipt.get("commands_reconstructed")) is int and receipt["commands_reconstructed"]==420
        and type(receipt.get("test_commands_verified")) is int and receipt["test_commands_verified"]==210
        and all(type(receipt.get(k)) is int and receipt[k]==0 for k in ("optimizer_steps","native_requests_executed"))
        and all(receipt.get(k) is False for k in ("model_promoted","execution_authorized","release_authorized","original_assessment_completed"))
        and _sha(receipt.get("checkpoint_bindings_sha256")) and _sha(receipt.get("forward_readouts_sha256")),
        "Independent model verification lacks CPU, unchanged-state or zero-work proof")
    if not plan["synthetic"]:
        require(receipt.get("runtime")==RUNTIME,"Independent model CPU runtime changed")
        require(type(receipt.get("input_sha256")) is dict and set(receipt["input_sha256"])==ROLES
            and all(_sha(v) for v in receipt["input_sha256"].values())
            and all(receipt["input_sha256"][k]==v for k,v in REAL_FILES.items()),
            "Independent real-model input identities changed")
    return copy.deepcopy(receipt)


def _write_new(path,value):
    raw=(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+"\n").encode()
    fd,name=tempfile.mkstemp(prefix=".model-check-",dir=path.parent)
    try:
        with os.fdopen(fd,"wb") as stream:stream.write(raw);stream.flush();os.fsync(stream.fileno())
        os.link(name,path)
        descriptor=os.open(path.parent,os.O_RDONLY)
        try:os.fsync(descriptor)
        finally:os.close(descriptor)
    finally:Path(name).unlink(missing_ok=True)


def check_models(request,*,run=subprocess.run):
    """Explicit fresh CPU reader. Unit tests provide a fabricated process runner."""
    paths,old,target,plan,original,before=_verify_request(request)
    raw=json.dumps(request,sort_keys=True,separators=(",",":"),allow_nan=False).encode()
    env=dict(os.environ,CUDA_VISIBLE_DEVICES="",PYTHONDONTWRITEBYTECODE="1")
    command=[sys.executable,str(Path(__file__).resolve()),"--cpu-helper","--expected-request-sha256",hashlib.sha256(raw).hexdigest()]
    run(command,input=raw,check=True,timeout=600,env=env)
    require(target.is_file() and not target.is_symlink(),"CPU model reader did not publish an independent receipt")
    receipt=validate_model_receipt(_json(target.read_bytes()),plan)
    require(receipt.get("request_sha256")==hashlib.sha256(raw).hexdigest()
        and receipt.get("input_sha256")==before,"CPU model receipt differs from exact request/input bytes")
    require({k:digest(v) for k,v in paths.items()}==before
        and source_hash(old)==request["original_source_sha256"]
        and source_hash(Path(__file__).resolve().parent)==request["evaluation_source_sha256"],"Inputs or sources changed during independent model check")
    return receipt


def cpu_helper(raw,expected_request_sha256):
    """Runs only in the fresh pinned producer-container CPU interpreter."""
    require(os.environ.get("CUDA_VISIBLE_DEVICES")=="","Model reconstruction must hide CUDA")
    require(hashlib.sha256(raw).hexdigest()==expected_request_sha256,"Model reader request bytes changed")
    request=_json(raw)
    paths,old,target,plan,original,before=_verify_request(request)
    require(request["synthetic"] is False,"Synthetic model checks use mocks, never a real child/model load")
    # No project module has been imported before this point. Remove the new
    # source directory from resolution, then use the separately byte-bound old
    # producer implementation to retain its model/input/proposal arithmetic.
    new_source=Path(__file__).resolve().parent
    sys.dont_write_bytecode=True
    sys.path=[str(old)]+[p for p in sys.path if Path(p or ".").resolve()!=new_source]
    from candidate_continuation_production import production_inputs,verify_dependencies,assessment_context
    from candidate_continuation_assessment_audit import _independent_plan
    from candidate_continuation_ledger import Ledger
    runtime=verify_dependencies(RUNTIME)
    inputs=production_inputs(paths["original_manifest"],REAL_FILES["original_manifest"])
    learned=_json(paths["original_learning_result"].read_bytes())
    binding=_json(paths["original_native_binding"].read_bytes())
    require(learned.get("status")=="learning_complete_audit_pending" and learned.get("synthetic") is False
        and learned.get("checkpoint_sha256")==REAL_FILES["final_checkpoint"]
        and learned.get("bindings")==inputs["bindings"],"Original completed TRAIN/model bindings changed")
    ledger=Ledger.from_dict(learned["ledger"])
    context=assessment_context(inputs,ledger,learned,paths["final_checkpoint"],binding)
    reconstructed=_independent_plan(ledger,REAL_FILES["final_checkpoint"],inputs["cells_by_split"],inputs["supports"],
        inputs["records"],context["parent_distributions"],context["continued_distributions"],context["model_hashes"],context["bindings"])
    _verify_suffix(plan,reconstructed)
    after={k:digest(v) for k,v in paths.items()}
    require(after==before and source_hash(old)==V4_SOURCE
        and source_hash(new_source)==request["evaluation_source_sha256"],"Independent model reader changed input/source bytes")
    receipt=dict(schema=SCHEMA,status="passed",synthetic=False,request_sha256=expected_request_sha256,input_sha256=before,
        plan_fingerprint=plan["fingerprint"],original_plan_fingerprint=original["fingerprint"],
        original_source_sha256=V4_SOURCE,evaluation_source_sha256=request["evaluation_source_sha256"],
        parent_checkpoint_sha256=REAL_FILES["parent_checkpoint"],final_checkpoint_sha256=REAL_FILES["final_checkpoint"],
        model_hashes=context["model_hashes"],checkpoint_bindings_sha256=canonical_hash(learned["bindings"]),
        original_guard_source_sha256=copy.deepcopy(binding["guard_source_sha256"]),
        original_native_binding=copy.deepcopy(binding),
        forward_readouts_sha256=canonical_hash(dict(parent=context["parent_distributions"],continued=context["continued_distributions"])),
        commands_reconstructed=420,test_commands_verified=210,cpu_only=True,cuda_visible_devices="",runtime=runtime,
        source_and_input_bytes_unchanged=True,optimizer_steps=0,native_requests_executed=0,
        model_promoted=False,execution_authorized=False,release_authorized=False,original_assessment_completed=False,
        scope="Fresh original-source CPU model-only loads and independent full-plan/TEST-suffix reconstruction; no optimizer restoration, new shot, model selection or old assessment completion.")
    receipt["fingerprint"]=canonical_hash(receipt);validate_model_receipt(receipt,plan);_write_new(target,receipt)
    return receipt


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpu-helper",action="store_true",required=True)
    parser.add_argument("--expected-request-sha256",required=True)
    args=parser.parse_args(argv)
    require(_sha(args.expected_request_sha256),"External model request digest required")
    return cpu_helper(sys.stdin.buffer.read(),args.expected_request_sha256)


if __name__=="__main__":main()
