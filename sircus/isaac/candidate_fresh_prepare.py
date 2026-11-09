"""Bounded CPU-only fresh geometry preparation; never trains or runs a shot.

Reads only frozen coordinate/scene/assets inputs and one original checkpoint.
All output paths must be new and outside the completed benchmark. The manifest
is frozen before parent inference or endpoint qualification; no outcome selects
positions or replaces empty supports.
"""

import argparse
import hashlib
import json
import multiprocessing as mp
import os
import platform
from pathlib import Path
import queue
import resource
import shutil
import time
import xml.etree.ElementTree as ET

from candidate_fresh_splits import make_fresh_splits, audit_fresh_splits
from candidate_fresh_parent import export_parent
from candidate_fresh_bank import generate_fresh_candidates, qualify_fresh_candidates, PreparationLimitError
from continuous_feasibility import canonical_hash
from prepare_assets import chain

ORIGINAL_SPLIT_SHA256 = "f46c690d80bd2079b27c8d3c72267c39acf1cee61fb4625f37994bff92292785"
SCENE_SHA256 = "9a50295d9908be8634c2df1c0b460ec6011710cf8e14ce33cbc67a75749ff1c8"
ASSET_SHA256 = {"Arm10":"3f40e56e46b040fb3339c1235ea85872f46511c45db2d37f7661db94a470bae3",
                "Arm14":"9981178abd0d87db43f7cca9fe26b0a74618aaa75ea10709194bd23acf59a340"}
POSES_SHA256 = "5847c2793606860a82cc74030c3fe529767318f189c5d48e3d66c0dc7b9f776a"
FILES = {"Arm10":"dropper.urdf", "Arm14":"board_arm.urdf"}
FLOOR = 6*1024**3
WALL_LIMIT_S = 1800.
CPU_LIMIT_S = 7200.


def digest(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_bytes(), parse_constant=lambda _: (_ for _ in ()).throw(ValueError("Nonfinite preparation JSON")))


def write_new(path, value):
    with Path(path).open("x") as stream:
        json.dump(value,stream,indent=2,allow_nan=False);stream.write("\n")


def load_geometry(inputs):
    inputs = Path(inputs)
    if digest(inputs/"original-splits.json") != ORIGINAL_SPLIT_SHA256:
        raise ValueError("Original frozen coordinate manifest changed")
    scene = read(inputs/"scene.json")
    if canonical_hash(scene) != SCENE_SHA256: raise ValueError("Original physics/material/geometry scene changed")
    links = {}
    for name,filename in FILES.items():
        if digest(inputs/filename) != ASSET_SHA256[name]: raise ValueError("Frozen arm asset changed")
        links[name] = chain(ET.parse(inputs/filename).getroot())
    if digest(inputs/"poses.json") != POSES_SHA256: raise ValueError("Frozen canonical home poses changed")
    poses = read(inputs/"poses.json")
    starts = {"Arm10":poses["dropper_joint_rad"], "Arm14":poses["board_joint_rad"]}
    return scene,links,starts


def _audit_worker(output, inputs, outbox):
    """Fresh process, original read-only FK/path auditor, no IK optimization."""
    try:
        from candidate_fresh_bank_audit import audit_fresh_bank
        scene,links,starts = load_geometry(inputs)
        plan,result = read(output/"plan.json"),read(output/"qualification.json")
        splits,parent = read(output/"split-manifest.json"),read(output/"parent-predictions.json")
        begin = time.process_time()
        audit = audit_fresh_bank(plan,result,splits,scene,links,starts,
            parent_actions=parent["parent_actions"],parent_latent_means=parent["parent_latent_means"],
            parent_latent_scales=parent["parent_latent_scales"],bindings=plan["input_bindings"])
        outbox.put((audit,time.process_time()-begin,None))
    except BaseException as error:
        outbox.put((None,None,type(error).__name__+": "+str(error)))


def bounded_audit(output, inputs, *, wall_limit_s, cpu_limit_s):
    context = mp.get_context("spawn")
    outbox = context.Queue()
    process = context.Process(target=_audit_worker,args=(output,inputs,outbox))
    begin,parent_cpu = time.monotonic(),time.process_time()
    children_before = resource.getrusage(resource.RUSAGE_CHILDREN)
    cost = None
    process.start()

    def live_cpu():
        try:
            fields=Path(f"/proc/{process.pid}/stat").read_text().rsplit(")",1)[1].split()
            child=(int(fields[11])+int(fields[12]))/os.sysconf("SC_CLK_TCK")
        except (OSError,ValueError,IndexError): child=0.
        return time.process_time()-parent_cpu+child

    try:
        while time.monotonic()-begin < min(600.,wall_limit_s):
            if shutil.disk_usage(output).free < FLOOR: raise RuntimeError("Original 6 GiB storage floor reached")
            if live_cpu() >= cpu_limit_s: raise TimeoutError("Remaining fresh preparation CPU cap reached during audit")
            try: audit,cpu,error = outbox.get(timeout=.05)
            except queue.Empty:
                if not process.is_alive(): raise RuntimeError("Owned fresh-audit worker exited without a receipt")
                continue
            if error: raise RuntimeError(error)
            cost=dict(wall_s=time.monotonic()-begin,worker_compute_cpu_s=cpu)
            return audit,cost
        raise TimeoutError("Bounded fresh-bank audit wall limit reached")
    finally:
        process.join(timeout=1)
        if process.is_alive(): process.terminate();process.join(timeout=5)
        if process.is_alive(): process.kill();process.join(timeout=5)
        outbox.close()
        if cost is not None:
            children=resource.getrusage(resource.RUSAGE_CHILDREN)
            cost.update(wall_s=time.monotonic()-begin,aggregate_cpu_s=time.process_time()-parent_cpu
                +children.ru_utime+children.ru_stime-children_before.ru_utime-children_before.ru_stime)
            if cost["wall_s"] > min(600.,wall_limit_s) or cost["aggregate_cpu_s"] > cpu_limit_s:
                raise TimeoutError("Measured fresh audit exceeded its remaining wall/CPU allowance")


def build_preparation(inputs, checkpoint, output, *, frozen_experiment_root):
    """Explicit geometry preparation entry point, never a training launcher."""
    inputs,checkpoint,output = Path(inputs).resolve(),Path(checkpoint).resolve(),Path(output).resolve()
    frozen = Path(frozen_experiment_root).resolve()
    if output.exists() or output.is_relative_to(frozen): raise ValueError("Fresh preparation needs an untouched path outside frozen experiment")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "": raise ValueError("Preparation is CPU-only; hide CUDA")
    if shutil.disk_usage(output.parent).free < FLOOR: raise RuntimeError("Original 6 GiB storage floor reached")
    scene,links,starts = load_geometry(inputs)
    original,history = read(inputs/"original-splits.json"),read(inputs/"history-receipt.json")
    input_sha = {p.name:digest(p) for p in sorted(inputs.iterdir()) if p.is_file()}
    checkpoint_before = digest(checkpoint)
    output.mkdir()
    begin,cpu_start = time.monotonic(),time.process_time()
    child_start = resource.getrusage(resource.RUSAGE_CHILDREN)
    progress = []

    def elapsed_cpu():
        children = resource.getrusage(resource.RUSAGE_CHILDREN)
        return time.process_time()-cpu_start+children.ru_utime+children.ru_stime-child_start.ru_utime-child_start.ru_stime

    def guard():
        if time.monotonic()-begin >= WALL_LIMIT_S or elapsed_cpu() >= CPU_LIMIT_S:
            raise TimeoutError("Fresh preparation cumulative wall/CPU cap reached")
        if shutil.disk_usage(output).free < FLOOR: raise RuntimeError("Original 6 GiB storage floor reached")

    try:
        write_new(output/"input-bindings.json",dict(input_sha256=input_sha,checkpoint_sha256=checkpoint_before,
            source_sha256={p.name:digest(p) for p in sorted(Path(__file__).parent.glob("*.py"))},
            common_starts=starts,common_start_source="Declared frozen asset poses.json canonical simulation home; actual native readback must be rechecked"))
        splits = make_fresh_splits(history,original)
        split_audit = audit_fresh_splits(splits,history,original,replay=True)
        write_new(output/"split-manifest.json",splits)
        write_new(output/"split-audit.json",split_audit)
        guard()
        parent = export_parent(output/"split-manifest.json",checkpoint,
                               expected_split_sha256=digest(output/"split-manifest.json"))
        write_new(output/"parent-predictions.json",parent)
        bindings = dict(parent_checkpoint_sha256=checkpoint_before,
            parent_prediction_sha256=digest(output/"parent-predictions.json"),
            fresh_split_manifest_sha256=digest(output/"split-manifest.json"),
            history_receipt_sha256=digest(inputs/"history-receipt.json"),
            original_split_manifest_sha256=ORIGINAL_SPLIT_SHA256,
            scene_file_sha256=digest(inputs/"scene.json"),poses_sha256=digest(inputs/"poses.json"),
            arm10_asset_sha256=ASSET_SHA256["Arm10"],arm14_asset_sha256=ASSET_SHA256["Arm14"])
        plan = generate_fresh_candidates(splits,parent["parent_actions"],parent["parent_latent_means"],
                                         parent["parent_latent_scales"],bindings=bindings)
        write_new(output/"plan.json",plan)
        guard()

        def record_progress(value):
            guard();progress.append(value)
            with (output/"progress.jsonl").open("a") as stream: stream.write(json.dumps(value,allow_nan=False)+"\n")
            print(json.dumps(value),flush=True)

        qualified = qualify_fresh_candidates(plan,scene,links,starts,workers=4,progress=record_progress,
            wall_limit_s=WALL_LIMIT_S-(time.monotonic()-begin),cpu_limit_s=CPU_LIMIT_S-elapsed_cpu())
        write_new(output/"qualification.json",qualified)
        guard()
        audit,audit_cost = bounded_audit(output,inputs,wall_limit_s=WALL_LIMIT_S-(time.monotonic()-begin),
                                       cpu_limit_s=CPU_LIMIT_S-elapsed_cpu())
        write_new(output/"audit.json",audit)
        guard()
        if digest(checkpoint) != checkpoint_before or any(digest(inputs/name) != sha for name,sha in input_sha.items()):
            raise ValueError("Frozen preparation inputs or parent changed")
        summary = dict(schema="sircus-candidate-fresh-preparation-v1",status="passed",
            counts=dict(validation=35,test=70,raw_proposals=8505,unique_candidates=qualified["unique_candidates"],
                admitted=qualified["admitted"],supported_contexts=audit["supported_contexts"],
                empty_contexts=len(audit["empty_contexts"])),
            exclusion_counts=splits["exclusion_counts"],exclusion_reasons=audit["exclusion_reasons"],
            split_audit=split_audit,audit_fingerprint=audit["fingerprint"],
            artifact_sha256={name:digest(output/name) for name in ("input-bindings.json","split-manifest.json",
                "split-audit.json","parent-predictions.json","plan.json","qualification.json","audit.json")},
            parent_checkpoint_sha256_before=checkpoint_before,parent_checkpoint_sha256_after=digest(checkpoint),
            input_sha256=input_sha,canonical_starts_sha256=canonical_hash(starts),
            cost=dict(total_wall_s=time.monotonic()-begin,total_aggregate_cpu_s=elapsed_cpu(),
                      geometry=qualified["cost"],audit=audit_cost,
                      parent_lifetime_peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
                      child_max_lifetime_peak_rss_mib=resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss/1024,
                      memory_scope="Linux process high-water RSS; child value is the largest individual child across qualification/audit, not summed or isolated-phase memory; no GPU context."),
            runtime=dict(python=platform.python_version(),numpy=__import__("numpy").__version__,
                scipy=__import__("scipy").__version__,torch=parent["runtime"]["torch"],
                device="cpu",cuda_visible_devices=""),
            limits=dict(max_cpu_workers=4,max_wall_s=1800,max_aggregate_cpu_s=7200,min_storage_gib=6),
            optimizer_updates=0,native_shots=0,new_task_ball_releases=0,held_out_outcome_files_read=0,
            release_authorized=False,
            scope="Outcome-blind fresh geometry and finite candidate qualification only; empty contexts remain failures. No training, simulation shot or performance measurement.")
        summary["fingerprint"] = canonical_hash(summary)
        write_new(output/"summary.json",summary)
        return summary
    except BaseException as error:
        if isinstance(error,PreparationLimitError): write_new(output/"partial-qualification.json",error.partial)
        write_new(output/"incomplete.json",dict(status="incomplete",error=type(error).__name__+": "+str(error),
            wall_s=time.monotonic()-begin,aggregate_cpu_s=elapsed_cpu(),optimizer_updates=0,native_shots=0))
        raise


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for flag in ("inputs","checkpoint","output","frozen-experiment-root"): p.add_argument("--"+flag,type=Path,required=True)
    args=p.parse_args()
    result=build_preparation(args.inputs,args.checkpoint,args.output,frozen_experiment_root=args.frozen_experiment_root)
    print(json.dumps(dict(status=result["status"],counts=result["counts"],cost=result["cost"])),flush=True)


if __name__=="__main__": main()
