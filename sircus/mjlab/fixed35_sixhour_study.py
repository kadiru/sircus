"""Bounded six-hour extension with independent preparation and flight ledgers."""
from __future__ import annotations
import argparse
import copy
import datetime
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import signal
import time
from . import fixed35_sixhour_contract as C
from .benchmark_verify import digest, sha, read
from .corrected_grid import source_manifest
from .coverage_training_study import append, share_native_environment
from .gpu_slot import GPU_UUID, gpu_slot, foreign_processes
from .policy64_contract import load_runtime_reference
from .table_correction_native import MatchedEngine, META
from .fixed35_sixhour_native import SixHourEngine, BROAD_RULE
from .fixed35_broad_native import reach_bound, drop_gap_interval
from .corrected_imitation_study import mirror_control_proofs
from .table_training_study import control_gate
from .sweep_worker import atomic

INPUT_KEYS=C.INPUT_KEYS


def native_search_counts(rows):
    return {c:sum(r.get('phase')=='search' and r['context_id']==c and r.get('native_release') is True for r in rows) for c in C.CASE_IDS}


def select_fresh(prepared, counts, seen):
    selected=[];unused=[];keys=set(seen)
    capacity=C.SEARCH_RELEASE_LIMIT-sum(counts.values())
    for row in prepared:
        key=C.command_key(row)
        reason='previously_flown_exact_command' if key in keys else 'native_release_limit' if len(selected)>=capacity else None
        if reason:unused.append(dict(id=row['id'],command_key=key,reason=reason))
        else:selected.append(row);keys.add(key)
    return selected,unused


def run(a):
    root=Path(__file__).resolve().parents[2];started=time.monotonic();acquired=getattr(a,'acquired_monotonic',started)
    elapsed=lambda:time.monotonic()-acquired
    loaded=C.load_inputs(a,root);parent=loaded['parent'];ref=load_runtime_reference(a.reference,a.assets,root)
    C.require(digest(ref['scene'])==parent['identity']['scene_sha256'] and ref['asset_hashes']==parent['identity']['asset_sha256'],'Parent scene/assets differ')
    C.require({n:importlib.metadata.version(n) for n in parent['packages']}==parent['packages'],'Pinned dependencies differ')
    output=Path(a.output).resolve()
    C.require(not output.exists() and not any(output.is_relative_to(Path(p).resolve()) for p in (root,a.parent_study,a.original_study,a.recovery_study,a.broad_study,a.assets)),'Fresh separate output required')
    output.mkdir(parents=True)
    for name in ('bootstrap','requests','repeat-plans','preparation-requests','prepared','window-plans'):(output/name).mkdir()
    for name in ('calibration.json','verification.json','setup-home.json'):shutil.copyfile(a.parent_study/name,output/name)
    for name,key in (('prior.json','prior'),('old-prior.json','old_prior'),('controls.json','controls'),('control-replays.json','control_replays')):atomic(output/name,loaded[key])
    for name,key in (('preserved','preserved'),('cases','cases'),('blocked','blocked'),('parent-current','current'),('parent-refit','refit'),('numeric-sources','numeric_sources')):
        atomic(output/f'bootstrap/{name}.json',loaded[key])
    atomic(output/'source-bindings.json',loaded['bindings']);atomic(output/'grid.json',loaded['grid'])
    atomic(output/'bootstrap/broad-confirmed.json',loaded['broad_sources'])
    atomic(output/'bootstrap/broad-parent-commands.json',[dict(id=r['id'],command_key=C.command_key(r)) for r in loaded['broad_rows'] if r.get('native_release')])
    shutil.copyfile(a.parent_study/'geometry-audit.json',output/'geometry-audit.json')
    # All numeric settings are frozen before controls or new drop outcomes.
    print(json.dumps(dict(phase='freezing_reach_conditioned_proposals',elapsed_s=elapsed())),flush=True)
    plan=C.make_plan(loaded['numeric_sources'],loaded['broad_sources'],ref['scene'],a.assets,interval_fn=drop_gap_interval,reach_fn=reach_bound)
    preserved=C.preserved_plan(loaded['preserved'])
    atomic(output/'search-plan.json',plan);atomic(output/'preserved-plan.json',dict(rows=preserved,frozen_before_outcomes=True))
    rule=dict(version=C.VERSION,protocol=C.protocol(),native_branch_rule=BROAD_RULE);atomic(output/'rule.json',rule)
    spec=dict(version=C.VERSION,scope=C.SCOPE,protocol=C.protocol(),identity=copy.deepcopy(parent['identity']),parent_identity=copy.deepcopy(parent['identity']),
        scene=ref['scene'],packages=parent['packages'],runtime_reference=ref['runtime_reference'],source_sha256=source_manifest(root),
        source_bindings=loaded['bindings'],prior_sha256=digest(loaded['prior']),old_prior_sha256=digest(loaded['old_prior']),
        geometry_audit_sha256=sha(output/'geometry-audit.json'),search_plan_sha256=sha(output/'search-plan.json'),
        preserved_plan_sha256=sha(output/'preserved-plan.json'),rule_sha256=sha(output/'rule.json'),
        grid=loaded['grid'],parent_artifacts=loaded['artifacts'],release_behavior=C.parent.ROTATION,
        created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    spec['identity']['source_sha256']=spec['source_sha256'];spec['fingerprint']=digest(spec);atomic(output/'contract.json',spec)
    rows=[];preparations=[];report=dict(version=C.VERSION,status='running',phase='setup',scope=C.SCOPE,
        contract_fingerprint=spec['fingerprint'],protocol=C.protocol(),training=False,posthoc_development=True,
        historical_policy_captures=24,parent_captures=len({r["context_id"] for r in loaded["preserved"]}),parent_positions=35,control_gates={},cases=[],
        completion_reason=None,budget_truncated=False,maximum_native_search=C.SEARCH_RELEASE_LIMIT,maximum_native_releases=C.SEARCH_RELEASE_LIMIT+27+18+len(preserved),
        deadlines=dict(worker_limit_s=a.wall_time_limit,effective_worker_limit_s=getattr(a,'effective_worker_limit',a.wall_time_limit),
            search_limit_s=a.search_limit,global_deadline_unix_s=getattr(a,'global_deadline_unix',None),
            acquired_deadline_unix_s=getattr(a,'acquired_deadline_unix_s',None)),gate=dict(status='pending'))
    def publish(phase):
        kept=[r for r in rows if r['phase']=='preserved']
        report.update(phase=phase,elapsed_s=elapsed(),counts=C.statistics(rows),actual_requests=len(rows),
            actual_native_releases=sum(r.get('native_release') is True for r in rows),
            fresh_search_releases=native_search_counts(rows),preparation_count=len(preparations),
            preparation_declines=sum(not r.get('admitted_request_ids') for r in preparations),
            stage_counts={p:C.statistics([r for r in rows if r['phase']==p]) for p in ('controls_pre','preserved','search','repeat','controls_post')},
            cases=C.case_results(rows),preservation=dict(requests=len(kept),captures=sum(C.captured(r) for r in kept),expected=len(preserved),
                all_recaptured=len(kept)==len(preserved) and all(C.captured(r) for r in kept)))
        atomic(output/'report.json',report)
        print(json.dumps(dict(phase=phase,status=report['status'],native_search=report['fresh_search_releases'],
            search_captures=report['stage_counts']['search'].get('captures'),prepared=len(preparations),elapsed_s=elapsed())),flush=True)
    def boundary(phase,**extra):
        append(output/'phase-boundaries.jsonl',[dict(phase=phase,acquired_elapsed_s=elapsed(),rows_count=len(rows),
            rows_sha256=sha(output/'rows.jsonl') if rows else None,**extra)])
    publish('setup');boundary('inputs-frozen')
    engine=SixHourEngine(ref,a.assets,output/'native-yaw',parent=a.parent_study,sources=loaded['sources'],source_roots=loaded['source_roots'],
        additional_sources=loaded['broad_sources'],additional_source_roots=loaded['broad_source_roots'],
        prior=loaded['prior'],calibration=loaded['calibration'],home=loaded['home'])
    atomic(output/'native-source-receipts.json',engine.source_receipts)
    expected_options=read(a.parent_study/'native-yaw/effective-options.json');expected_geometry=read(a.parent_study/'native-yaw/native-compiled-geometry.json')
    mirror=mirror_control_proofs(a.parent_study,output,loaded['control_replays'])
    controls=MatchedEngine(ref,a.assets,output/'native-controls',parent=mirror,sources=loaded['control_replays'],prior=loaded['old_prior'],calibration=loaded['calibration'],home=loaded['home'])
    share_native_environment(engine,controls)
    def limits(n=0):
        C.require(sum(r.get('native_release') is True for r in rows)+n<=report['maximum_native_releases'],'Bounded native-flight cap exceeded')
        if elapsed()>getattr(a,'effective_worker_limit',a.wall_time_limit)-60:raise TimeoutError('Own finite worker budget reached')
        C.require(shutil.disk_usage(output).free>=C.protocol()['storage_reserve_bytes'],'8GiB storage reserve reached')
        C.require(not foreign_processes(),'Another4090workload appeared; left untouched')
    number=0
    def execute(requests,phase,*,control=False):
        nonlocal number
        result=[]
        for offset in range(0,len(requests),32):
            chunk=requests[offset:offset+32];limits(len(chunk));packet=f'requests/batch-{number:05d}.json';number+=1
            atomic(output/packet,dict(phase=phase,rows=chunk,pre_outcome=True));start=elapsed()
            observed,timing=(controls if control else engine).evaluate(chunk,phase)
            C.require([r['id'] for r in observed]==[r['id'] for r in chunk],'Requested order changed')
            if engine.adapter.env is not None:C.require(engine.adapter.env.effective_options==expected_options and engine.adapter.env.compiled_geometry==expected_geometry,'Native physics/geometry changed')
            for request,row in zip(chunk,observed):
                row.update(learning_reward=C.reward(row),request_packet=packet,request_packet_sha256=sha(output/packet),adapter_directory='native-controls' if control else 'native-yaw')
                if phase in ('search','repeat'):row['prepared_request']=copy.deepcopy(request)
            append(output/'rows.jsonl',observed);rows.extend(observed);result.extend(observed)
            append(output/'timing.jsonl',[dict(phase=phase,**timing,acquired_start_s=start,acquired_end_s=elapsed(),request_packet=packet,request_packet_sha256=sha(output/packet))])
            caches={r['adapter_directory']+'/'+r['native_cache_file']:r['native_cache_sha256'] for r in observed if r.get('native_release')}
            for name,expected in caches.items():C.require(sha(output/name)==expected,'Flight archive changed before packet commit')
            append(output/'packet-commits.jsonl',[dict(packet=packet,packet_sha256=sha(output/packet),row_ids=[r['id'] for r in observed],
                observed_rows_sha256=digest(observed),rows_count=len(rows),rows_sha256=sha(output/'rows.jsonl'),native_files=caches)])
            publish(phase)
        return result
    def control_block(phase):
        requests=[]
        for original,source,role in zip(loaded['controls'],loaded['control_replays'],('direct_control','table_control','negative_control')):
            C.require(source['role']==role and source['values']==original['values'] and source['bucket_xy_m']==original['bucket_xy_m'],'Control inputs differ')
            for i in range(3):
                requests.append(dict({k:copy.deepcopy(source[k]) for k in META},id=f'fixed35-sixhour-{phase}-{role}-r{i}',phase=phase,
                    role=role,method='control',context_id=source['context_id'],bucket_xy_m=source['bucket_xy_m'],values=source['values'],command_values=source['values'],
                    source_record_id=source['id'],source_record_sha256=digest(source),variant='control_exact',adapter_directory='native-controls',
                    control_source_id=original['id'],control_source_sha256=digest(original),repeat=i,perturbation_mixed_units=[0.]*5))
        gate=control_gate(execute(requests,phase,control=True));report['control_gates']['pre' if phase=='controls_pre' else 'post']=gate
        C.require(gate['status']=='passed','Unchanged original control gate failed');boundary(phase+'-complete')
    control_block('controls_pre')
    saved=execute(preserved,'preserved');C.require(len(saved)==len(preserved) and all(C.captured(r) for r in saved),'All exact preserved commands must recapture before search')
    boundary('preserved-complete')
    active_plan=plan;plan_file='search-plan.json';cursor=0;stage='primary';window_frozen=False
    seen={C.command_key(r) for r in loaded['broad_rows'] if r.get('native_release')}
    confirms={c:[] for c in C.CASE_IDS};round_number=0
    while True:
        counts=native_search_counts(rows)
        if sum(counts.values())>=C.SEARCH_RELEASE_LIMIT:report['completion_reason']='search_native_limit';break
        if elapsed()>=a.search_limit:report.update(completion_reason='search_time_limit',budget_truncated=True);break
        if len(preparations)>=C.NUMERIC_LIMIT:report['completion_reason']='numeric_proposal_limit';break
        target=next(c for c in C.case_results(rows) if c['context_id']==C.TARGET)
        if not window_frozen and elapsed()>=C.protocol()['primary_limit_s'] and target['confirmed_recipes']:
            sources=loaded['broad_sources']+[r for c in C.case_results(rows) for r in c['confirmed_recipes']]
            active_plan=C.make_window_plan(loaded['numeric_sources'],sources,ref['scene'],a.assets,
                interval_fn=drop_gap_interval,reach_fn=reach_bound,limit=C.NUMERIC_LIMIT-len(preparations))
            active_plan.update(selected_after_rows_count=len(rows),selected_after_rows_sha256=sha(output/'rows.jsonl'),
                acquired_elapsed_s=elapsed(),posthoc_development=True)
            plan_file='window-plans/plan.json';atomic(output/plan_file,active_plan)
            boundary('window-plan-frozen',plan_file=plan_file,plan_sha256=sha(output/plan_file))
            cursor=0;stage='window';window_frozen=True
        if elapsed()>=a.search_limit:report.update(completion_reason='search_time_limit',budget_truncated=True);break
        proposal_batch=active_plan['rows'][cursor:cursor+min(12,C.NUMERIC_LIMIT-len(preparations))]
        cursor+=len(proposal_batch)
        if not proposal_batch:report['completion_reason']='finite_reachable_proposal_pool_exhausted';break
        limits();packet=f'preparation-requests/batch-{round_number:05d}.json'
        atomic(output/packet,dict(rows=proposal_batch,search_plan_file=plan_file,search_plan_sha256=sha(output/plan_file),pre_outcome=True))
        publish('preparing');begin=elapsed()
        admitted,prepared=engine.prepare(proposal_batch)
        selected,unused=select_fresh(admitted,counts,seen)
        result_file=f'prepared/batch-{round_number:05d}.json'
        atomic(output/result_file,dict(requests=selected,preparations=prepared,not_executed=unused,
            preparation_request_file=packet,preparation_request_sha256=sha(output/packet),frozen_before_native_outcomes=True))
        for r in prepared:r.update(preparation_request_file=packet,preparation_request_sha256=sha(output/packet),prepared_file=result_file,prepared_sha256=sha(output/result_file))
        append(output/'preparations.jsonl',prepared);preparations.extend(prepared)
        boundary('preparation-committed',round=round_number,request_packet=packet,prepared_file=result_file,prepared_sha256=sha(output/result_file),elapsed_preparation_s=elapsed()-begin)
        selected=[dict(r,prepared_batch_file=result_file,prepared_batch_sha256=sha(output/result_file)) for r in selected]
        observed=execute(selected,'search') if selected else []
        seen.update(C.command_key(r) for r in observed)
        discoveries=[]
        for row in observed:
            c=row['context_id'];numeric=tuple(row['values'])
            if C.captured(row) and len(confirms[c])<3 and numeric not in confirms[c]:
                group=len(confirms[c]);confirms[c].append(numeric);discoveries.append((row,group))
        if discoveries:
            repeats=[r for source,group in discoveries for r in C.repeat_plan(source,group)]
            repeat_file=f'repeat-plans/group-{round_number:05d}.json'
            atomic(output/repeat_file,dict(rows=repeats,selected=[dict(id=r['id'],sha256=digest(r)) for r,_ in discoveries],
                rows_count_before=len(rows),rows_sha256_before=sha(output/'rows.jsonl'),frozen_before_repeat_outcomes=True))
            execute([dict(r,repeat_plan_file=repeat_file,repeat_plan_sha256=sha(output/repeat_file)) for r in repeats],'repeat')
        round_number+=1;publish('search');boundary('search-round-complete',round=round_number,cursor=cursor,search_stage=stage)
    boundary('search-complete',reason=report['completion_reason'],cursor=cursor,search_stage=stage)
    control_block('controls_post')
    report.update(status='completed',gate=dict(status='passed',controls_passed=True,preservation_passed=True,
        bounded_search_completed=True,no_policy_promotion=True))
    publish('completed')
    recipes=dict(preserved_parent=[dict(id=r['id'],sha256=digest(r),values=r['values'],q10=r['q10'],q14=r['q14'],bucket_xy_m=r['bucket_xy_m']) for r in loaded['preserved']],
        new_confirmed=[r for c in report['cases'] for r in c['confirmed_recipes']],policy_unchanged=True)
    atomic(output/'recipes.json',recipes)
    report['evidence_sha256']={str(p.relative_to(output)):sha(p) for p in sorted(output.rglob('*')) if p.is_file() and p.name not in ('report.json','evidence-manifest.json')}
    atomic(output/'evidence-manifest.json',report['evidence_sha256']);publish(report['status']);return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in (*INPUT_KEYS,'output','gpu-lock'):parser.add_argument('--'+key,type=Path,required=True)
    parser.add_argument('--wall-time-limit',type=float,default=18000);parser.add_argument('--search-limit',type=float,default=16200)
    parser.add_argument('--queue-timeout',type=float,default=18000);parser.add_argument('--global-deadline-unix',type=float,required=True);a=parser.parse_args()
    C.require(60<a.wall_time_limit<=18000 and 0<a.search_limit<min(a.wall_time_limit,16201) and 0<a.queue_timeout<=18000 and not a.output.exists(),'Fresh output and finite budgets required')
    C.require(os.environ.get('CUDA_VISIBLE_DEVICES')==GPU_UUID,'Only authorized4090 UUID may execute')
    def deadline(signum,frame):raise TimeoutError('Own acquired-slot hard deadline')
    lease_timeout=min(a.queue_timeout,a.global_deadline_unix-21600-time.time())
    if lease_timeout<=0:raise TimeoutError('Separate parent/GPU queue deadline elapsed during startup')
    try:
        with gpu_slot(a.gpu_lock,timeout_s=lease_timeout,notify=lambda v:print(json.dumps(v),flush=True)):
            a.acquired_monotonic=time.monotonic();acquired_unix=time.time()
            acquired_deadline=min(acquired_unix+21600,a.global_deadline_unix);a.acquired_deadline_unix_s=acquired_deadline
            a.effective_worker_limit=min(a.wall_time_limit,acquired_deadline-time.time()-3600)
            if a.effective_worker_limit<=60:raise TimeoutError('Audit reserve leaves no worker time')
            budget=a.output.with_name(a.output.name+'-budget.json');C.require(not budget.exists(),'Fresh owned budget required')
            atomic(budget,dict(acquired_unix_s=acquired_unix,acquired_total_limit_s=21600,acquired_deadline_unix_s=acquired_deadline,limit_s=a.wall_time_limit,effective_worker_limit_s=a.effective_worker_limit,
                search_limit_s=a.search_limit,global_deadline_unix_s=a.global_deadline_unix,audit_reserve_s=3600,gpu_uuid=GPU_UUID))
            signal.signal(signal.SIGALRM,deadline);signal.setitimer(signal.ITIMER_REAL,a.effective_worker_limit-10)
            try:run(a)
            finally:signal.setitimer(signal.ITIMER_REAL,0)
    except Exception as error:
        if a.output.exists():
            p=a.output/'report.json';r=read(p) if p.exists() else {};r.update(status='failed',error=type(error).__name__+': '+str(error));atomic(p,r)
        raise


if __name__=='__main__':main()
