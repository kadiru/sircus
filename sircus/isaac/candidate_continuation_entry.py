"""Inert-by-default production entry and separate CPU-only final audit.

No simulator, training or shots start on import or without --execute-approved.
That flag is a future caller's explicit authorization, never granted by a
manifest, successful mock tests, model checkpoint or an audit result.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import re
import time

from candidate_continuation_application import read,sha,require,run_pipeline
from candidate_continuation_journal import write_new_json
from candidate_continuation_ledger import Ledger,canonical_hash


def audit_application(output,inputs,session,*,expected_request_sha256,checks=None):
    """Join independently replayed learning, native TRAIN, startup and assessment."""
    require(os.environ.get("CUDA_VISIBLE_DEVICES")=="","Independent final auditor must hide CUDA")
    output=Path(output);request_path=output/"audit-request.json"
    require(sha(request_path)==expected_request_sha256,"External final audit request bytes changed")
    request=read(request_path)
    require(request.get("session")==session and request.get("source_sha256")==inputs["bindings"]["source_sha256"],
            "Final audit request source/session changed")
    began,cpu=time.monotonic(),time.process_time()
    i=inputs;p=i["paths"]
    if checks is None:
        from candidate_continuation_verify import verify_run
        from candidate_continuation_native_audit import audit_native_train
        from candidate_continuation_assessment_audit import audit_assessment
        from candidate_continuation_scene import audit_startup
        from candidate_continuation_production import assessment_context
        checks=dict(learning=verify_run,native=audit_native_train,assessment=audit_assessment,
                    startup=audit_startup,context=assessment_context)
    learned=read(output/"learning/result.json");assessed=read(output/"assessment/assessment-result.json")
    require(type(learned.get("synthetic")) is bool and assessed.get("synthetic") is learned["synthetic"],
            "Mixed or missing real/synthetic assessment provenance")
    require(sha(output/"learning/result.json")==request["learning_result_sha256"]
        and sha(output/"assessment/assessment-result.json")==request["assessment_result_sha256"]
        and sha(output/"native-binding.json")==request["native_binding_sha256"]
        and sha(output/"native-setup/startup/receipt.json")==request["startup_receipt_sha256"],
        "Audit input result/native/startup bytes changed")
    native_binding=read(output/"native-binding.json")
    ledger=Ledger.from_dict(learned["ledger"])
    learning=checks["learning"](output/"learning",expected_result_sha256=request["learning_result_sha256"],
        expected_bindings=i["bindings"],train_context_ids=ledger.train_context_ids,
        supports=i["train_supports"],demonstrations=i["train"]["demonstrations"])
    native=checks["native"](output/"learning",expected_head_sha256=learned["journal_head_sha256"],
        expected_bindings=i["bindings"],expected_native_binding=native_binding,scene=i["values"]["scene"],
        cells=i["train"]["cells"],supports=i["train_supports"],prepared_records=i["train_records"],
        assets_paths={"Arm10":p["dropper_urdf"],"Arm14":p["board_urdf"]},
        poses_path=p["poses"],camera_path=p["camera"],parent_checkpoint_path=p["parent_checkpoint"])
    context=checks["context"](i,ledger,learned,output/"learning/final.pt",native_binding)
    assessment=checks["assessment"](output/"assessment",expected_head_sha256=assessed["journal_head_sha256"],
        expected_bindings=context["bindings"],train_ledger=ledger,expected_native_binding=native_binding,
        scene=i["values"]["scene"],cells_by_split=i["cells_by_split"],supports=i["supports"],prepared_records=i["records"],
        parent_distributions=context["parent_distributions"],continued_distributions=context["continued_distributions"],
        model_hashes=context["model_hashes"],assets_paths={"Arm10":p["dropper_urdf"],"Arm14":p["board_urdf"]},
        poses_path=p["poses"],camera_path=p["camera"],parent_checkpoint_path=p["parent_checkpoint"],
        final_checkpoint_path=output/"learning/final.pt",startup_directory=output/"native-setup",
        expected_result_sha256=request["assessment_result_sha256"],historical_capture_ids=i["historical_capture_ids"])
    from camera_options import preview_origins
    startup=checks["startup"](output/"native-setup/startup",scene=i["values"]["scene"],
        cells=i["values"]["source_contract"]["cells"],origins=preview_origins())
    require(all(r.get("status")=="passed" for r in (learning,native,assessment,startup)),
            "Independent audit component failed; no promotion")
    require(all(r.get("synthetic") is learned["synthetic"] for r in (learning,native,assessment)),
            "Independent audit provenance differs from its source history")
    require(startup["startup_control_sha256"]==native_binding["startup_control_sha256"],
            "Observed startup controls differ from every native trial binding")
    result=dict(schema="sircus-candidate-continuation-combined-audit-v1",status="passed",session=session,
        all_components_passed=True,synthetic=learned["synthetic"],learning=learning,native_train=native,assessment=assessment,startup=startup,
        source_sha256=i["bindings"]["source_sha256"],audit_request_sha256=expected_request_sha256,
        final_checkpoint_sha256=learned["checkpoint_sha256"],model_promoted=False,
        audit_cost=dict(wall_s=time.monotonic()-began,process_cpu_s=time.process_time()-cpu,
            lifetime_peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
            device="cpu",cuda_visible_devices="",scope="Independent reader process only; also contained in parent audit-phase wall/aggregate CPU, not additive. RSS is reader lifetime high-water."),
        execution_authorized=False,release_authorized=False)
    result["fingerprint"]=canonical_hash(result)
    write_new_json(output/"combined-audit.json",result)
    return result


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest",type=Path,required=True)
    p.add_argument("--expected-manifest-sha256",required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--session",required=True)
    p.add_argument("--execute-approved",action="store_true")
    p.add_argument("--audit-only",action="store_true")
    p.add_argument("--expected-audit-request-sha256")
    p.add_argument("--headless",action="store_true")
    p.add_argument("--enable_cameras",action="store_true")
    p.add_argument("--device",default="cuda:0")
    return p


def main(argv=None):
    args=parser().parse_args(argv)
    require(re.fullmatch(r"[0-9a-f]{32}",args.session) is not None,"Expected a 128-bit session token")
    require(args.execute_approved != args.audit_only,"Choose explicit future execution or read-only CPU audit")
    require(args.output.is_dir() and not args.output.is_symlink(),"Supervisor-owned new output is required")
    if args.audit_only:
        require(os.environ.get("CUDA_VISIBLE_DEVICES")=="","Independent final auditor must hide CUDA")
        require(args.expected_audit_request_sha256 is not None,"External audit-request SHA required")
        from candidate_continuation_production import production_inputs
        return audit_application(args.output,production_inputs(args.manifest,args.expected_manifest_sha256),args.session,
                                 expected_request_sha256=args.expected_audit_request_sha256)
    require(args.headless and args.enable_cameras and args.device=="cuda:0", "Fixed headless 4090 camera runtime required")
    require(os.environ.get("SIRCUS_CONTINUATION_SESSION")==args.session,"Watchdog-owned session cookie required")
    from candidate_continuation_production import ProductionComponents
    from candidate_continuation_watchdog import ProgressWriter
    return run_pipeline(args.output,args.session,
        ProductionComponents(args.manifest,args.expected_manifest_sha256,args.output,args.session,device=args.device),
        ProgressWriter(args.output,args.session))


if __name__=="__main__":main()
