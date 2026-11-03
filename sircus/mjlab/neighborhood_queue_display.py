"""Finite CPU-only queued-study display; never acquires a GPU or writes evidence."""
import argparse
import json
from pathlib import Path
import time

from . import neighborhood_contract as protocol
from .contact_study import atomic


def pending_report(study,now=None):
    study=Path(study)
    if (study/'report.json').exists():return None
    now=time.time() if now is None else now
    state_path=study.with_suffix('.pipeline.json')
    state=json.loads(state_path.read_text()) if state_path.exists() else {}
    failed=state.get('status') in ('failed','audit_failed')
    settings=protocol.settings()
    anchors=[dict(id=a['id'],values=settings['board_xy_m']+[a['height'],a['tilt'],a['gap']]) for a in protocol.ANCHORS]
    return dict(version=protocol.VERSION,status='failed' if failed else 'waiting_for_4090',phase='device_queue',
        scope=protocol.SCOPE,settings=settings,anchors=anchors,max_requests=protocol.MAX_REQUESTS,
        counts=dict(requests=0,native_releases=0,captures=0,geometry_rejections=0),
        contexts=[dict(**c,board_admitted=None,reason='waiting_for_native_study',geometry_requests=0,eligible=0,
            search_attempts=0,search_native_releases=0,search_captures=0,clean=0,best_gap_m=None) for c in protocol.contexts()],
        selected_recipes=[],trace_examples=[],queue=dict(wait_s=max(0,now-state.get('started_unix_s',now)),
            reason='Existing shared4090 allocation is preserved; this study has not started physics'),
        error=state.get('error'))


def publish_pending(study,display):
    from .prepare_neighborhood_report import compact
    study,display=Path(study),Path(display);raw=pending_report(study)
    if raw is None:return False
    if display.resolve().is_relative_to(study.resolve()):raise ValueError('Display must remain outside evidence')
    for _ in range(3):
        original=display.read_bytes();value=json.loads(original);value['neighborhood_evaluation']=compact(raw)
        if (study/'report.json').exists():return False
        if display.read_bytes()!=original:continue
        atomic(display,value);return True
    raise RuntimeError('Concurrent display write preserved')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study',type=Path,required=True);parser.add_argument('--display',type=Path,required=True)
    parser.add_argument('--maximum-seconds',type=float,default=7320)
    a=parser.parse_args()
    if not 0<a.maximum_seconds<=7320:raise ValueError('Finite queue display duration required')
    deadline=time.monotonic()+a.maximum_seconds
    while time.monotonic()<deadline:
        if not publish_pending(a.study,a.display):return
        state=json.loads(a.study.with_suffix('.pipeline.json').read_text())
        if state.get('status') in ('failed','audit_failed','completed'):return
        time.sleep(min(10,max(0,deadline-time.monotonic())))


if __name__=='__main__':main()
