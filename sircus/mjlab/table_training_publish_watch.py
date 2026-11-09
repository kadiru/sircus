"""Finite CPU-only HTML snapshot updater; keeps frozen producer bytes untouched."""
import argparse
import json
from pathlib import Path
import time
from .prepare_table_training import compact,install,sha,atomic_bytes

def refresh(study,target,pipeline):
    state=json.loads(pipeline.read_text()) if pipeline.is_file() else {'status':'starting'}
    value=compact(study,pipeline=state)
    install(target,value,sha(target))
    return state

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('study','target-html','pipeline-state','receipt'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--max-seconds',type=float,default=12000);a=p.parse_args()
    if not 1<=a.max_seconds<=14400:raise ValueError('Finite publication duration required')
    start=time.monotonic();count=0
    while True:
        state=refresh(a.study,a.target_html,a.pipeline_state);count+=1
        receipt=dict(status='watching',updates=count,elapsed_s=time.monotonic()-start,pipeline_status=state.get('status'),source_sha256=sha(Path(__file__)),html_sha256=sha(a.target_html))
        if state.get('status') in ('completed','failed','audit_failed'):
            receipt['status']='finished'
        elif receipt['elapsed_s']>=a.max_seconds:
            receipt['status']='publication_timeout'
        atomic_bytes(a.receipt,(json.dumps(receipt,indent=2)+'\n').encode())
        if receipt['status']!='watching':break
        time.sleep(15)
if __name__=='__main__':main()
