"""Finite CPU-only media/report publication; never launches or gates training."""
import argparse
import json
import os
from pathlib import Path
import time

from shot_benchmark_report import publish, read, sha
from shot_benchmark_supervisor import atomic
from shot_benchmark_video import encode_run
import shot_benchmark_video as video_module


def watch(root, output, *, poll_s=300, duration_s=8*86400):
    root,output=Path(root),Path(output)
    if not 15<=poll_s<=1800 or not 60<=duration_s<=8*86400:
        raise ValueError("Unbounded publication interval/deadline")
    if os.environ.get("CUDA_VISIBLE_DEVICES")!="":
        raise ValueError("CPU publication requires an explicitly empty CUDA_VISIBLE_DEVICES")
    experiment=read(root/"experiment.json")
    expected=sha(root/"experiment.json");seen={};started=time.monotonic()
    status=root/"publication-watch.json"
    try:
        while time.monotonic()-started<duration_s:
            if sha(root/"experiment.json")!=expected:raise ValueError("Frozen experiment bytes changed")
            media_errors=[]
            for item in experiment["runs"]:
                run=root/item["path"];manifest=run/"clips.json"
                if not manifest.is_file() or not (run/"progress.json").is_file():continue
                current=sha(manifest)
                # Only a new source chapter manifest needs another encoder pass.
                # The encoder verifies all listed JPEGs/receipts on every pass;
                # the final independent audit verifies original raw media too.
                if seen.get(item["path"])!=current:
                    try:
                        encode_run(run,root/"experiment.json",threads=2)
                    except Exception as exc:
                        # Keep valid saved curves/earlier chapters available. The
                        # report publisher still verifies every referenced byte;
                        # a bad new video never bypasses its integrity checks.
                        media_errors.append(dict(run=item["path"],error=str(exc)))
                    else:
                        seen[item["path"]]=current
            value=publish(root,output,media=True)
            coordinator=value.get("coordinator")
            complete=(value.get("status")=="completed" and coordinator is not None
                and coordinator.get("status")=="completed" and not media_errors)
            atomic(status,dict(status="completed" if complete else "watching_with_media_errors" if media_errors else "watching",pid=os.getpid(),
                experiment_sha256=expected,elapsed_s=time.monotonic()-started,
                observed_train_attempts=value["observed_train_attempts"],report_sha256=sha(output/"report.json"),
                media_errors=media_errors,publisher_sha256=sha(__file__),encoder_sha256=sha(video_module.__file__),
                scope="CPU encoding/publication only; native training and its independent audit run separately"))
            if complete:return value
            time.sleep(poll_s)
        raise TimeoutError("Finite publication watcher expired; native jobs retain their own bounds")
    except BaseException as exc:
        atomic(status,dict(status="failed",pid=os.getpid(),error=str(exc),elapsed_s=time.monotonic()-started))
        raise


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("root",type=Path);p.add_argument("output",type=Path)
    p.add_argument("--poll-s",type=int,default=300);p.add_argument("--duration-s",type=int,default=8*86400)
    a=p.parse_args();watch(a.root,a.output,poll_s=a.poll_s,duration_s=a.duration_s)
