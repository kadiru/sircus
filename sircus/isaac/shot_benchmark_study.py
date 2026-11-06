"""Finite host-side six-run queue; the existing shared 4090 lock is mandatory.

No work starts on import. This coordinator never restarts a container or signals
another workload. A failed/partial run stops the queue rather than spending its
budget again under the same seed. Completed runs require a passed CPU audit.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from shot_benchmark_contract import validate_protocol
from shot_benchmark_supervisor import atomic, sha, verify_archive


def native_arguments(experiment, method, seed, *, smoke=False):
    values=["--appearance","silver","--continuous-source","{source_run}",
        "--continuous-camera-json","{camera}","--shot-benchmark-plan",str(experiment),
        "--shot-benchmark-method",method,"--shot-benchmark-seed",str(seed)]
    if smoke:values.append("--shot-benchmark-smoke")
    return values


def run(args):
    root=args.experiment.parent.resolve()
    experiment=json.loads(args.experiment.read_text())
    validate_protocol(experiment["protocol"])
    if experiment["source"]["commit"]!=args.source_commit or experiment["source"]["archive_sha256"]!=sha(args.archive):
        raise ValueError("Frozen experiment/source identity changed")
    verify_archive(args.archive,args.source,args.source_commit)
    helper=args.source/"sircus/isaac/shot_benchmark_supervisor.py"
    if helper.resolve()!=Path(__file__).with_name("shot_benchmark_supervisor.py").resolve():
        raise ValueError("Coordinator must execute its pinned source snapshot")
    controls=root/"control";controls.mkdir(exist_ok=True)
    started=time.monotonic();deadline=started+8*86400
    status=root/"coordinator.json"
    def publish(state,**extra):
        atomic(status,dict(status=state,pid=os.getpid(),source_commit=args.source_commit,
            updated_utc=datetime.now(timezone.utc).isoformat(),elapsed_s=time.monotonic()-started,
            total_authorized_train_attempts=60000,**extra))
    declared={(x["method"],x["seed"]):x for x in experiment["runs"]}
    audit_env=dict(os.environ,CUDA_VISIBLE_DEVICES="",OMP_NUM_THREADS="1",
                   OPENBLAS_NUM_THREADS="1",MKL_NUM_THREADS="1",PYTHONDONTWRITEBYTECODE="1")
    try:
        for ordered in experiment["protocol"]["run_order"]:
            method,seed=ordered["method"],ordered["seed"]
            item=declared[(method,seed)];dest=root/item["path"]
            done=dest.with_name(dest.name+".audit.json")
            if done.exists():
                if json.loads(done.read_text()).get("status")!="passed":raise ValueError("Existing audit did not pass")
                continue
            if dest.exists() or dest.with_name(dest.name+".launch.json").exists():
                raise ValueError("Partial/existing run requires explicit recovery; refusing duplicate paid trials: "+str(dest))
            dest.parent.mkdir(exist_ok=True)
            mapped="/workspace/"+str(args.experiment.resolve().relative_to(args.workspace.resolve()))
            af=controls/f"{method}-seed{seed}-args.json"
            atomic(af,native_arguments(mapped,method,seed))
            command=[sys.executable,str(helper),"--source-host",str(args.source),
                "--output-host",str(dest),"--camera-host",str(args.camera),
                "--source-run-host",str(args.source_run),"--archive-host",str(args.archive),
                "--native-args-json",str(af),"--source-commit",args.source_commit,
                "--workspace-host",str(args.workspace),"--helper-host",str(helper),"--batches","289"]
            publish("launching",method=method,seed=seed)
            subprocess.run(command,check=True,timeout=60)
            sp=dest.with_name(dest.name+".supervisor.json")
            while True:
                if time.monotonic()>deadline:raise TimeoutError("Finite study coordinator deadline reached; no next run launched")
                state=json.loads(sp.read_text()) if sp.exists() else {"status":"queued"}
                publish("running",method=method,seed=seed,native_status=state)
                if state["status"] in ("completed","failed"):
                    if state.get("returncode")!=0:raise RuntimeError("Native run failed; later runs were not launched")
                    break
                time.sleep(15)
            publish("auditing",method=method,seed=seed)
            subprocess.run([sys.executable,str(args.source/"sircus/isaac/shot_benchmark_audit.py"),
                "--run",str(dest),"--plan",str(args.experiment),"--output",str(done)],
                env=audit_env,check=True,timeout=7200)
            if json.loads(done.read_text()).get("status")!="passed":raise RuntimeError("Audit failed")
        publish("completed")
    except BaseException as exc:
        publish("failed",error=str(exc));raise


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for flag in ("experiment","source","archive","camera","source-run","workspace"):
        p.add_argument("--"+flag,type=Path,required=True)
    p.add_argument("--source-commit",required=True)
    a=p.parse_args();run(a)


if __name__=="__main__":main()
