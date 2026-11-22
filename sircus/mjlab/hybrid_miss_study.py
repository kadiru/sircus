"""One causal correction: change board-arm joints, preserve Cartesian shots.

Three diagnosed development misses; frozen nominal repeats, not a policy test.
CPU preparation and native64 execution are separate, resumable stages.
"""
from __future__ import annotations
import argparse
import copy
import json
import math
import os
from pathlib import Path
import shutil
import time

from .benchmark_verify import contained, digest, read, sha
from .sweep_worker import atomic

VERSION = 'sircus-hybrid-miss-correction-v1'
CASES = tuple('stalled-reference-test-' + x for x in ('03', '09', '19'))
SEEDS = (2026112203, 2026112209, 2026112219)
REPEATS = 3
FIXED = ('values', 'bucket_xy_m', 'board_heading_offset_deg', 'board_twist_deg', 'selected_q10')
SOURCE_REPORT = 'e8cd7009362af9c80f7fe6a54b70937b6cbef8f4182eaf04d22e6b7e8dbac2e1'


def source_inputs(a):
    from .intergrid_route_refit_inputs import _audited_source
    from .stalled_reference_study import VERSION as SOURCE_VERSION
    report, manifest, audit, bindings, contract = _audited_source(
        a.source_study, a.source_audit, SOURCE_VERSION,
        ('choices.json', 'rows.jsonl', 'training-prior.json', 'calibration.json', 'setup-home.json'))
    if bindings['report.json'] != SOURCE_REPORT:
        raise ValueError('Use the exact audited three-miss comparison')
    requests = {s['context_id']: s['request'] for s in read(a.source_study/'choices.json')['slots']
                if s['method'] == 'hybrid' and s['request'] is not None}
    rows = {r['id']: r for r in map(json.loads, (a.source_study/'rows.jsonl').read_text().splitlines())}
    selected = []
    for key in CASES:
        request = requests[key]
        row = rows[request['id']]
        if not row['native_release'] or row['invalid_physics'] or row['route_success']:
            raise ValueError('Expected a valid released noncapture')
        for field in ('values', 'bucket_xy_m'):
            if row[field] != request[field]: raise ValueError('Source action mismatch')
        if row['q10'] != request['selected_q10'] or row['q14'] != request['selected_q14']:
            raise ValueError('Source joint branch mismatch')
        for key2 in ('context_resolution', 'prepared_connection'):
            path = contained(a.source_study/'native-hybrid', request[key2+'_file'])
            if sha(path) != request[key2+'_sha256']: raise ValueError('Source proof changed')
        trace = contained(a.source_study/row['adapter_directory'], row['native_cache_file'])
        if sha(trace) != row['native_cache_sha256']: raise ValueError('Source trace changed')
        selected.append(dict(id=key, original=request, historical=row))
    return selected, contract, bindings


def native_inputs(a):
    from .clearance_expansion_study import native_inputs as load
    return load(a)


def engine_for(a, request, directory, ref):
    from .cell1050_heading_native import HeadingEngine
    from .direct_path_first import DirectPathEngine, hydrate_direct_resolution
    from .parallel_capture_grid_workers import load_resolution
    from .stalled_reference_plan import request_mode
    mode = request_mode(request)
    cls = DirectPathEngine if mode == 'direct_first' else HeadingEngine
    engine = cls(ref, a.assets, directory, prior=read(a.output/'training-prior.json'),
                 calibration=read(a.output/'calibration.json'), home=read(a.output/'setup-home.json'))
    anchor = (hydrate_direct_resolution(engine, request['bucket_xy_m']) if mode == 'direct_first'
              else load_resolution(engine, request['bucket_xy_m']))
    if anchor is None or anchor['sha256'] != request['context_resolution_sha256']:
        raise ValueError('Prepared context unavailable; no native replanning allowed')
    return engine


def validate_change(original, corrected):
    if any(original[k] != corrected[k] for k in FIXED):
        raise ValueError('Only board-arm joint configuration may change')
    before, after = original['selected_q14'], corrected['selected_q14']
    if len(before) != 6 or len(after) != 6 or not all(math.isfinite(x) for x in (*before, *after)):
        raise ValueError('Six finite board-arm joints required')
    if max(abs(math.remainder(x-y, 2*math.pi)) for x,y in zip(before,after)) < .05:
        raise ValueError('Correction must be a distinct board-arm joint branch')


def predicted_arm_margin(prediction, route):
    row = prediction.get(route) or {}
    value = row.get('arm_clearance_m')
    return float(value) if isinstance(value, (int, float)) and math.isfinite(value) else -math.inf


def choose_branch(original, alternatives, predictions, route):
    for row in alternatives: validate_change(original, row)
    # Fixed ranking chosen before any new drops. Ties use stable request IDs.
    ordered = sorted(alternatives, key=lambda r: (-predicted_arm_margin(predictions[r['id']], route), r['id']))
    return ordered[0] if ordered else None


def publish(a, report):
    report['updated_at_unix'] = time.time()
    atomic(a.output/'report.json', report)
    print(json.dumps({k: report[k] for k in ('status', 'phase', 'progress', 'error') if k in report}), flush=True)


def prepare(a):
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '': raise ValueError('CPU preparation requires hidden GPUs')
    import torch
    torch.set_num_threads(1)
    from .clearance_expansion_plan import _prepare_branches
    from .cell1050_rebound_study import preparation_time_limit, SearchPreparationExpired
    from .interactive_stalled_planner import copy_prepared_proofs
    from .intergrid_flight_ranker import FlightRanker
    from .intergrid_system_study import require_storage
    selected, source_contract, binding = source_inputs(a)
    spec, parent_binding, source, ref = native_inputs(a)
    if spec['identity'] != source_contract['identity']: raise ValueError('Scene identity changed')
    a.output.mkdir(parents=True, exist_ok=False)
    require_storage(a.output)
    for name in ('training-prior.json', 'calibration.json', 'setup-home.json'):
        shutil.copyfile(a.source_study/name, a.output/name)
    shutil.copyfile(a.parent_study/'verification.json', a.output/'verification.json')
    ranker = FlightRanker(ref['scene'], read(a.output/'calibration.json')['score_config'],
                         a.assets, source_contract['flight_ranking']['rule'])
    plan = dict(version=VERSION, repeats=REPEATS, seeds=list(SEEDS), source=binding,
                parent=parent_binding, identity=spec['identity'], scene=spec['scene'],
                changed_parameter='bouncer_joint_configuration', cases=[])
    report = dict(version=VERSION, status='running', phase='preparing', physics_substeps=64,
                  audit_status='pending', progress=dict(completed=0, planned=18), cases=[])
    publish(a, report)
    for case in selected:
        original = case['original']
        directory = a.output/('native-'+case['id'][-2:])
        copy_prepared_proofs(a.source_study/'native-hybrid', directory, original)
        engine = engine_for(a, original, directory, ref)
        proposals = [dict(id=case['id']+'-alternate-'+str(seed), context_id=case['id'],
            bucket_xy_m=original['bucket_xy_m'], values=original['values'],
            board_heading_offset_deg=original['board_heading_offset_deg'], board_twist_deg=original['board_twist_deg'],
            phase='diagnostic', role='diagnostic', method='corrected', intended_route=original['intended_route'],
            seed_q10=original['selected_q10'], seed_q14=original['selected_q14'],
            source_request_id=original['id'], branch_seed=seed) for seed in SEEDS]
        alternatives, logs = [], []
        begin = time.monotonic()
        for proposal in proposals:
            try:
                with preparation_time_limit(180 - (time.monotonic()-begin)):
                    result = _prepare_branches(engine, [proposal], {original['id']: original}, a.assets)
                alternatives.extend(result['admitted']); logs.extend(result['records'])
            except SearchPreparationExpired:
                logs.append(dict(proposal_id=proposal['id'], reason='180_second_case_preparation_limit'))
                break
        unique = {}
        for row in alternatives:
            validate_change(original, row)
            unique.setdefault(digest([row['selected_q10'], row['selected_q14']]), row)
        alternatives = list(unique.values())
        predictions = {row['id']: ranker.predict(row) for row in alternatives}
        prediction = ranker.predict(original)
        route = prediction['predicted_route'] or original['intended_route']
        corrected = choose_branch(original, alternatives, predictions, route)
        if engine.adapter.env is not None or torch.cuda.is_initialized(): raise RuntimeError('CPU stage allocated CUDA')
        plan['cases'].append(dict(case, directory=directory.name, route_for_ranking=route,
            original_prediction=prediction, corrected=corrected, alternatives=alternatives,
            predictions=predictions, preparation=logs, preparation_s=time.monotonic()-begin))
        atomic(a.output/'preparation.json', plan)
        print(json.dumps(dict(case=case['id'], alternatives=len(alternatives), selected=corrected['id'] if corrected else None,
            original_arm_m=predicted_arm_margin(prediction, route),
            corrected_arm_m=predicted_arm_margin(predictions[corrected['id']], route) if corrected else None)), flush=True)
    report.update(phase='prepared', prepared_cases=len(plan['cases']))
    publish(a, report)


def schedule(plan):
    requests = []
    for case in plan['cases']:
        for repeat in range(REPEATS):
            # Alternate paired order; outcomes never feed back into selection.
            for variant in (('original', 'corrected') if repeat % 2 == 0 else ('corrected', 'original')):
                source = case[variant]
                requests.append(dict(case_id=case['id'], variant=variant, repeat=repeat,
                    directory=case['directory'], request=None if source is None else dict(copy.deepcopy(source),
                    id=f"{case['id']}-{variant}-{repeat}", context_id=case['id'], variant=variant,
                    repeat=repeat, method=variant, phase='diagnostic', source_request_id=source['id'])))
    return requests


def freeze(a):
    from .corrected_grid import source_manifest
    plan = read(a.output/'preparation.json')
    if len(plan['cases']) != 3 or (a.output/'contract.json').exists(): raise ValueError('Fresh complete preparation required')
    slots = schedule(plan)
    atomic(a.output/'slots.json', slots)
    files = {str(p.relative_to(a.output)): sha(p) for p in a.output.rglob('*') if p.is_file() and p.name != 'report.json'}
    spec = dict(version=VERSION, identity=plan['identity'], scene=plan['scene'], parent=plan['parent'],
        plan_sha256=sha(a.output/'preparation.json'), frozen_files=files,
        source_sha256=source_manifest(Path(__file__).resolve().parents[2]), physics_substeps=64,
        repeats=REPEATS, controls=18, planned_requests=36, gpu_cap_s=3600,
        correction='Only bouncer joints; identical full board and release poses, exact dropper joints',
        selection='Maximum predicted arm clearance on original predicted route, stable ID tie-break, before new drops',
        scope='Post-hoc development diagnosis; no training, planner promotion, noise or driven-arm native validation')
    spec['fingerprint'] = digest(spec)
    atomic(a.output/'contract.json', spec)
    report=read(a.output/'report.json'); report['phase']='frozen'; publish(a,report)


def verify_frozen(a):
    contract=read(a.output/'contract.json')
    if digest({k:v for k,v in contract.items() if k!='fingerprint'}) != contract['fingerprint']:
        raise ValueError('Study contract changed')
    root=Path(__file__).resolve().parents[2]
    for name, fingerprint in contract['source_sha256'].items():
        if sha(root/name) != fingerprint: raise ValueError('Frozen source changed: '+name)
    for name, fingerprint in contract['frozen_files'].items():
        if sha(a.output/name) != fingerprint: raise ValueError('Frozen input changed: '+name)
    return contract


def summarize(plan, rows):
    from .intergrid_route_selector_plan import outcome, STANDARD_CAPTURES
    result=[]
    for case in plan['cases']:
        groups={}
        for variant in ('original','corrected'):
            found=[r for r in rows if r.get('context_id')==case['id'] and r.get('variant')==variant]
            outcomes=[outcome(r) for r in found]
            groups[variant]=dict(requested=REPEATS, completed=len(found), released=len(found),
                setup_rejected=REPEATS if case[variant] is None else 0,
                captures=sum(x in STANDARD_CAPTURES for x in outcomes), direct=outcomes.count('direct'),
                table=outcomes.count('table'), robot_contacts=sum(r['experimental_route']['robot_contact_steps']>0 for r in found),
                outcomes=outcomes, rows=found)
        result.append(dict(id=case['id'],bucket_xy_m=case['original']['bucket_xy_m'],results=groups))
    return result


def run(a):
    from .cell1050_rebound_study import annotate, control_requests
    from .coverage_training_study import append
    from .gpu_slot import foreign_processes, gpu_slot
    from .intergrid_route_labels import annotate_routes
    from .intergrid_system_study import gpu_deadline, require_storage
    from .parallel_capture_grid_study import control_mirror
    from .table_correction_native import MatchedEngine
    from .table_training_study import control_gate
    verify_frozen(a)
    _, _, _, ref = native_inputs(a)
    plan, slots = read(a.output/'preparation.json'), read(a.output/'slots.json')
    if (a.output/'native-started.json').exists(): raise ValueError('Never repeat a started native study')
    report=read(a.output/'report.json')
    def waiting(event):
        report.update(phase='waiting_for_4090',gpu_wait=event); publish(a,report)
    with gpu_slot(a.gpu_lock,timeout_s=a.gpu_wait_s,poll_s=15,notify=waiting):
        begin=time.monotonic()
        atomic(a.output/'native-started.json',dict(slots_sha256=sha(a.output/'slots.json'), started_unix=time.time()))
        with gpu_deadline(3600):
            engines={c['id']:engine_for(a,c['original'],a.output/c['directory'],ref) for c in plan['cases']}
            sources=read(a.parent_study/'control-replays.json')
            mirror=control_mirror(a.parent_study,a.output,sources,False)
            controls=MatchedEngine(ref,a.assets,a.output/'native-controls',parent=mirror,sources=sources,
                prior=read(a.parent_study/'old-prior.json'),calibration=read(a.output/'calibration.json'),home=read(a.output/'setup-home.json'))
            adapters=[e.adapter for e in engines.values()]+[controls.engine.adapter]
            factory,allocated=adapters[0].environment_factory,[]
            def shared():
                if not allocated: allocated.append(factory())
                return allocated[0]
            for adapter in adapters: adapter.environment_factory=shared
            rows=[]; gates={}; packet=0
            expected_options=read(a.parent_study/'native-yaw/effective-options.json')
            expected_geometry=read(a.parent_study/'native-yaw/native-compiled-geometry.json')
            def execute(engine, batch, phase, directory, control=False):
                nonlocal packet
                if foreign_processes(): raise RuntimeError('Foreign workload appeared on leased4090')
                verify_frozen(a); require_storage(a.output)
                name=f'requests/batch-{packet:04d}.json'; packet+=1
                atomic(a.output/name,dict(phase=phase,control=control,adapter_directory=directory,rows=batch))
                measured,timing=engine.evaluate(batch,phase)
                if [r['id'] for r in measured]!=[r['id'] for r in batch] or any(not r['native_release'] for r in measured):
                    raise RuntimeError('Exact frozen request did not release')
                adapter=engine.engine.adapter if control else engine.adapter
                if adapter.env.effective_options!=expected_options or adapter.env.compiled_geometry!=expected_geometry:
                    raise RuntimeError('Pinned native64 geometry/options changed')
                if not control:
                    annotate(measured,a.output/directory); annotate_routes(measured,a.output/directory)
                for row,request in zip(measured,batch,strict=True):
                    row.update(adapter_directory=directory,request_packet=name,request_packet_sha256=sha(a.output/name))
                    if not control: row.update(variant=request['variant'],repeat=request['repeat'],context_id=request['context_id'])
                rows.extend(measured); append(a.output/'rows.jsonl',measured)
                append(a.output/'timing.jsonl',[dict(packet=name,phase=phase,timing=timing)])
                report.update(progress=dict(completed=sum(r.get('variant') in ('original','corrected') for r in rows),planned=18),
                    phase=phase,gpu_lease_s=time.monotonic()-begin,controls=gates,cases=summarize(plan,rows))
                publish(a,report)
                return measured
            for phase in ('controls_pre','diagnostic','controls_post'):
                if phase!='diagnostic':
                    gates[phase]=control_gate(execute(controls,control_requests(sources,phase),phase,'native-controls',True))
                    if gates[phase]['status']!='passed': raise RuntimeError('Matched controls failed')
                else:
                    for case in plan['cases']:
                        batch=[s['request'] for s in slots if s['case_id']==case['id'] and s['request'] is not None]
                        execute(engines[case['id']],batch,phase,case['directory'])
            report.update(status='completed',phase='completed',audit_status='pending',controls=gates,
                          cases=summarize(plan,rows),gpu_lease_s=time.monotonic()-begin)
            manifest={str(p.relative_to(a.output)):sha(p) for p in a.output.rglob('*') if p.is_file()
                      and p.name not in ('report.json','evidence-manifest.json')}
            atomic(a.output/'evidence-manifest.json',manifest)
            report['evidence_manifest_sha256']=sha(a.output/'evidence-manifest.json')
            publish(a,report)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('source-study','source-audit','parent-study','parent-audit','reference','assets','output','gpu-lock'):
        p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--gpu-wait-s',type=int,default=1800)
    p.add_argument('--stage',choices=('prepare','freeze','run'),required=True)
    a=p.parse_args()
    globals()[a.stage](a)


if __name__=='__main__': main()
