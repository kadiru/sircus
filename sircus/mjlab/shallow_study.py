"""Run one exact requested shallow-board shot at four distances on the4090."""
import argparse
import json
from pathlib import Path

from . import shallow_contract as protocol
from .contact_study import atomic
from .lowboard_study import _run, read


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('parent-study','assets','output','gpu-lock'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--batch',type=int,default=64,choices=(32,64,128))
    parser.add_argument('--queue-timeout',type=float,default=14400)
    parser.add_argument('--wall-time-limit',type=float,default=7200)
    a=parser.parse_args()
    from .gpu_slot import gpu_slot
    try:
        with gpu_slot(a.gpu_lock,timeout_s=a.queue_timeout,notify=lambda r:print(json.dumps(r),flush=True)):
            _run(a,protocol=protocol)
    except Exception as error:
        if a.output.is_dir():
            p=a.output/'report.json';report=read(p) if p.exists() else {}
            report.update(status='failed',error=type(error).__name__+': '+str(error));atomic(p,report)
        raise


if __name__=='__main__':main()
