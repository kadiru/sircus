"""Finite4090 calibrated replay with matched controls and a gated five-gap window."""
import argparse
from collections import Counter
import copy
import datetime
import importlib.metadata
import json
from pathlib import Path
import shutil
import time

import numpy as np

from . import replay650_contract as protocol
from .benchmark_adapter import jsonable
from .corrected_contract import PINNED_PACKAGES,sha,validate_packages
from .corrected_grid import source_manifest
from .expanded_actions import WIDE_BOUNDS
from .learning_contract import digest
from .sweep_worker import atomic


def storage_check(output):
    free=shutil.disk_usage(output).free;used=sum(p.stat().st_size for p in output.rglob('*') if p.is_file())
    if free<2*1024**3 or used>4*1024**3:raise RuntimeError('Finite dense-evidence storage budget exceeded; no files deleted')
    return dict(free_bytes=free,study_bytes=used)


def run(a):
    from .replay650_geometry import ReplayGeometry
    from .learning_solve import required_asset_paths
    scene,sources,calibration,bindings=protocol.load_inputs(a.old_study,a.positive_study,a.positive_audit,a.calibration_report,a.verification_report)
    packages=protocol.validate_replay_packages({name:importlib.metadata.version(name) for name in set(PINNED_PACKAGES)|set(protocol.REPLAY_EXTRA_PACKAGES)})
    assets={n:sha(a.assets.parent/n) for n in required_asset_paths(a.assets)}
    old=protocol.read(a.old_study/'contract.json')
    if assets!=old['asset_sha256'] or packages!=old['packages']:raise ValueError('Pinned asset/package mismatch')
    a.output.mkdir(parents=True,exist_ok=False);storage_check(a.output)
    for src,name in [(a.calibration_report,'calibration.json'),(a.verification_report,'verification.json')]:
        shutil.copyfile(src,a.output/name)
    source=Path(__file__).resolve().parents[2];started=time.monotonic();geometry=ReplayGeometry(scene,a.assets)
    groups=[geometry.check(g) for g in sources]
    candidate=next(g for g in groups if g['role']=='candidate')
    for gap in protocol.GAPS:
        group=copy.deepcopy(candidate);group.update(id=f'window-gap-{round(gap*10000)}',role='window')
        group['values'][4]=gap;groups.append(geometry.check(group,candidate=candidate))
    atomic(a.output/'groups.json',groups);atomic(a.output/'geometry.json',jsonable(geometry.proofs))
    initial=protocol.plan(groups[:3],'replay');window=protocol.plan(groups[3:],'window');plans=initial+window
    atomic(a.output/'plan.json',plans)
    settings=protocol.settings();settings['controls']={g['role']:protocol.metadata(g,scene) for g in groups[:3] if g['role'].endswith('_control')}
    spec=dict(version=protocol.VERSION,scene=scene,settings=settings,calibration=calibration,input_bindings=bindings,
        source_sha256=source_manifest(source),asset_sha256=assets,packages=packages,
        frozen_files_sha256={n:sha(a.output/n) for n in ('groups.json','geometry.json','plan.json','calibration.json','verification.json')},
        created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),scope=protocol.SCOPE,
        evidence_kind='native_current_ball_calibrated_replay',native_environment_batch=a.batch)
    spec['fingerprint']=digest(spec);atomic(a.output/'contract.json',spec)
    report=dict(version=protocol.VERSION,status='running',phase='prepared',contract_fingerprint=spec['fingerprint'],settings=settings,
        calibration=dict(score_config_sha256=calibration['score_config_sha256'],calibration_sha256=calibration['calibration_sha256'],verification_sha256=calibration['verification_sha256']),
        groups=[],counts={},gate=dict(status='pending',window_opened=False),trace_examples=[],scope=protocol.SCOPE)
    rows=[];batches=[];positions={};environments={};options={};group_lookup={g['id']:g for g in groups}
    def publish(phase):
        report.update(phase=phase,elapsed_s=time.monotonic()-started)
        report['counts']=dict(**protocol.statistics(rows),phase_counts=dict(Counter(r['phase'] for r in rows)),
            role_counts=dict(Counter(r['role'] for r in rows)),substep_counts=dict(Counter(str(r['substeps']) for r in rows)))
        report['groups']=[]
        for group in groups:
            selected=[r for r in rows if r['group_id']==group['id']]
            report['groups'].append(dict(**protocol.metadata(group,scene),eligible=group['eligible'],
                by_substeps={str(s):protocol.statistics([r for r in selected if r['substeps']==s]) for s in protocol.SUBSTEPS},
                sensitivity=protocol.sensitivity(selected,positions)))
        report['gate']=protocol.replay_gate(rows,positions);report['storage']=storage_check(a.output)
        atomic(a.output/'report.json',report)
        print(json.dumps(dict(phase=phase,elapsed_s=round(report['elapsed_s'],1),counts=report['counts'],gate=report['gate'])),flush=True)
        if len(rows)>protocol.MAX_REQUESTS or report['elapsed_s']>a.wall_time_limit:raise RuntimeError('Finite study request/time budget exceeded')
    publish('prepared')
    def env(step):
        if step not in environments:
            from .calibrated_env import CalibratedEnvironment,verified_native_options
            environments[step]=CalibratedEnvironment(scene,a.assets,calibration['score_config'],n=a.batch,bounds=WIDE_BOUNDS,substeps=step)
            native=a.output/'native'/str(step);native.mkdir(parents=True,exist_ok=False)
            atomic(native/'native-compiled-geometry.json',environments[step].compiled_geometry)
            options[str(step)]=verified_native_options(environments[step].sim,step)
            atomic(native/'effective-options.json',options[str(step)]);atomic(a.output/'effective-options.json',options)
        return environments[step]
    def execute(requests,step):
        from .bank_geometry import fixed_endpoints
        from .learning_contract import constraints
        for offset in range(0,len(requests),a.batch):
            packet=len(batches);chunk=requests[offset:offset+a.batch];active=[r for r in chunk if r['eligible']]
            measured=[];arrays=None;filename=None;native=a.output/'native'/str(step)
            if active:
                current=env(step)
                bucket=np.asarray([r['bucket_xy_m'] for r in active]);values=np.asarray([r['values'] for r in active])
                q10,q14=(np.asarray([r[k] for r in active]) for k in ('q10','q14'))
                with fixed_endpoints(current,bucket,values,q10,q14,np.ones(len(active),bool)):
                    measured,timing=current.run(bucket,values,constraints(len(active)),record=True)
                arrays=current.last_arrays;filename=f'native-{packet:04d}.npz'
                np.savez_compressed(native/filename,**arrays,record_ids=np.asarray([r['id'] for r in active]))
                raw_hash=sha(native/filename)
            lookup={r['id']:(i,m) for i,(r,m) in enumerate(zip(active,measured))}
            for request in chunk:
                row=dict(request,invalid='strict_endpoint_or_local_branch_rejected',native_release=False,route_success=False,route_quality=0,
                    whole_ball_entry=False,board_bounce=False,max_retention_s=0.,invalid_physics=False,scalar_score_audit_passed=False)
                if request['id'] in lookup:
                    i,m=lookup[request['id']]
                    if m['values']!=request['values'] or m['bucket_xy_m']!=request['bucket_xy_m'] or any(m[k]!=request[k] for k in ('q10','q14')):
                        raise RuntimeError('Native replay changed an exact frozen request or joint branch')
                    row.update(m);row.pop('trajectory_m',None)
                    row.update(native_cache_file=filename,native_cache_row=i,native_cache_sha256=raw_hash,native_directory='native/'+str(step))
                    positions[row['id']]=arrays['position_m'][:,i].copy()
                rows.append(row)
                with (a.output/'records.jsonl').open('a') as f:f.write(json.dumps(row,allow_nan=False)+'\n')
            batches.append(dict(packet=packet,substeps=step,phase=chunk[0]['phase'],record_ids=[r['id'] for r in chunk],
                native_record_ids=[r['id'] for r in active],native_directory='native/'+str(step),cache_file=filename))
            atomic(a.output/'batches.json',batches);publish(chunk[0]['phase']+'-'+str(step))
    for step in protocol.SUBSTEPS:execute([r for r in initial if r['substeps']==step],step)
    gate=protocol.replay_gate(rows,positions);atomic(a.output/'replay-gate.json',gate)
    if gate['status']=='passed':
        for step in protocol.SUBSTEPS:execute([r for r in window if r['substeps']==step],step)
    report['trace_examples']=[]
    for group in groups:
        for step in protocol.SUBSTEPS:
            row=next((r for r in rows if r['group_id']==group['id'] and r['substeps']==step and r['repeat_index']==0 and r['native_release']),None)
            if row is None:continue
            report['trace_examples'].append(dict(record_id=row['id'],group_id=group['id'],role=group['role'],substeps=step,
                **{k:protocol.metadata(group,scene)[k] for k in ('board_xy_m','bucket_xy_m','board_height_m','tilt_deg','gap_m','release_center_height_m','distance_m')},
                position_m=positions[row['id']][::step*4].tolist(),sample_dt_s=1/60,
                **{k:row[k] for k in ('whole_ball_entry','max_retention_s','external_contact_steps','route_success','invalid','invalid_physics','scalar_score_audit_passed')},
                selection_scope='Declared first nominal for each executed group and resolution; not outcome-selected or aggregate performance'))
    report['status']='completed'
    manifest={str(p.relative_to(a.output)):sha(p) for p in sorted(a.output.rglob('*')) if p.is_file() and p.name not in ('report.json','evidence-manifest.json') and not p.name.startswith('audit')}
    report['evidence_sha256']=manifest;atomic(a.output/'evidence-manifest.json',manifest);publish('completed')
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('old-study','positive-study','positive-audit','calibration-report','verification-report','assets','output','gpu-lock'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--batch',type=int,default=16,choices=(16,32));parser.add_argument('--queue-timeout',type=float,default=14400)
    parser.add_argument('--wall-time-limit',type=float,default=7200);a=parser.parse_args()
    from .gpu_slot import gpu_slot
    try:
        with gpu_slot(a.gpu_lock,timeout_s=a.queue_timeout,notify=lambda r:print(json.dumps(r),flush=True)):run(a)
    except Exception as error:
        if a.output.exists():
            path=a.output/'report.json';r=protocol.read(path) if path.exists() else {}
            r.update(status='failed',error=type(error).__name__+': '+str(error));atomic(path,r)
        raise


if __name__=='__main__':main()
