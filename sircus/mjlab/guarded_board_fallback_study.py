"""Frozen 20-position comparison of the current and board-geometry planners.

CPU selections finish before native64 execution. Each method gets one attempted
shot per position under the same decision budget. The local aiming diagnostic,
historical successful recipes, live planner and simulator contracts are separate.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import ExitStack
import json
from pathlib import Path
import random
import shutil
import time

from .benchmark_verify import digest, read, sha
from .corrected_grid import source_manifest
from .interactive_fast_check import coordinates
from .interactive_physics_check import sample_positions
from .sweep_worker import atomic

VERSION = "sircus-guarded-board-fallback-study-v1"
METHODS = ("baseline", "guarded")
POSITION_COUNT = 20
SEED = 2026112307
PROCESSING_SEED = 2026112308
PLANNING_BUDGET_S = 30.
GPU_CAP_S = 1800
HISTORY_NAMES = ("request.json", "preview.json", "plan.json", "positions.json", "result.json")


def make_layouts(provenance, seed=SEED):
    occupied = list(map(tuple, provenance["excluded_coordinates_m"]))
    layouts = []
    for offset in range(2):
        rows = sample_positions(provenance["geometry"], provenance["grid"], occupied,
                                provenance["bounds"], seed+offset)
        for row in rows:
            identity = f"guarded-board-test-{len(layouts):02d}"
            layouts.append(dict(row, id=identity, context_id=identity,
                                split="test", stratum=row["region"]))
        occupied.extend(tuple(row["bucket_xy_m"]) for row in rows)
    if len(layouts) != POSITION_COUNT:
        raise ValueError("Expected 20 fresh paired placements")
    return layouts


def publish(args, report, phase=None):
    if phase is not None:
        report["phase"] = phase
    report["updated_at_unix"] = time.time()
    atomic(args.output/"report.json", report)
    if args.display_report:
        display = read(args.display_report) if args.display_report.exists() else {}
        display["guarded_board_fallback_study"] = report
        atomic(args.display_report, display)
    print(json.dumps({k: report[k] for k in ("status", "phase", "progress", "gpu_wait", "error") if k in report}), flush=True)


def initialize(args):
    from .clearance_expansion_study import native_inputs
    from .fast_planning_study import COPIED_INPUTS
    from .intergrid_route_refit_inputs import _audited_source
    from .intergrid_route_labels import LABEL_VERSION
    from .intergrid_system_study import require_storage
    from .guarded_board_fallback_plan import RULE

    native, parent_binding, source, _ = native_inputs(args)
    files = (*COPIED_INPUTS, "layouts.json", "setup-home.json", "cpu-equivalence.json", "choices.json", "plans.json")
    _, _, _, binding, previous = _audited_source(args.source_study, args.source_audit,
                                                "sircus-board-fallback-study-v1", files)
    if (previous["identity"], previous["scene"]) != (native["identity"], native["scene"]):
        raise ValueError("Current planner and calibrated native scene differ")
    old_layouts = read(args.source_study/"layouts.json")
    known = set(map(tuple, old_layouts["provenance"]["excluded_coordinates_m"]))
    known.update(coordinates(old_layouts["layouts"]))
    if not args.evidence_root.is_dir():
        raise ValueError("Need current interactive evidence inventory")
    history = {}
    for name in HISTORY_NAMES:
        for path in sorted(args.evidence_root.rglob(name)):
            if path.resolve().is_relative_to(args.output.resolve()):
                raise ValueError("New study cannot contribute exclusion history")
            value = read(path)
            known.update(coordinates(value))
            history[str(path)] = sha(path)
    provenance = dict(excluded_coordinates_m=sorted(known), excluded_coordinates_sha256=digest(sorted(known)),
                      geometry=old_layouts["provenance"]["geometry"], grid=old_layouts["provenance"]["grid"],
                      bounds=old_layouts["provenance"]["bounds"], history_files=history,
                      source_layouts_sha256=sha(args.source_study/"layouts.json"))
    layouts = make_layouts(provenance)
    old_diagnostic = next(r for r in old_layouts['layouts'] if r['id'] == 'board-fallback-test-23')
    diagnostic = dict(old_diagnostic,id='guarded-board-known-regression',context_id='guarded-board-known-regression',
                      split='test',stratum='known regression; excluded from fresh comparison')
    args.output.mkdir(parents=True, exist_ok=False)
    require_storage(args.output)
    atomic(args.output/"layouts.json", dict(layouts=layouts, provenance=provenance))
    atomic(args.output/"diagnostic-layout.json",dict(layout=diagnostic,source_id=old_diagnostic['id'],repeats=3))
    for name in (*COPIED_INPUTS, "setup-home.json", "cpu-equivalence.json"):
        shutil.copyfile(args.source_study/name, args.output/name)
    order = [r["id"] for r in layouts]
    random.Random(PROCESSING_SEED).shuffle(order)
    spec = dict(version=VERSION, parent=parent_binding, source_parent=binding,
                identity=native["identity"], scene=native["scene"], flight_ranking=previous["flight_ranking"],
                methods=list(METHODS), test_positions=20, physics_substeps=64, label_version=LABEL_VERSION,
                planning_budget_s=PLANNING_BUDGET_S, reference_path_s=5., early_score_m=.005,
                attempts_per_method_per_position=1, seed=SEED, processing_seed=PROCESSING_SEED,
                processing_order=order, diagnostic_repeats=3, fallback_rule=RULE, gpu_cap_s=GPU_CAP_S,
                frozen_inputs={p.name:sha(p) for p in args.output.glob("*.json")},
                source_sha256=source_manifest(source),
                baseline="Unchanged StalledReferencePlanner with warm_early frozen model and same 30-second decision budget",
                scope="Privileged geometry, held-endpoint native64; no retraining or automatic live promotion")
    spec["fingerprint"] = digest(spec)
    atomic(args.output/"contract.json", spec)
    report = dict(version=VERSION, status="running", phase="initialized", audit_status="pending", physics_substeps=64,
                  budgets=dict(positions=20, planning_s=30.), controls={}, gpu_lease_s=0., gpu_cap_s=GPU_CAP_S,
                  progress=dict(prepared=0, planned=40, completed_releases=0, planned_releases=None),
                  next="Freeze both planners' choices before any test drops. Existing live planner is unchanged.")
    publish(args, report)


def frozen_check(args):
    spec = read(args.output/"contract.json")
    if digest({k:v for k,v in spec.items() if k != "fingerprint"}) != spec["fingerprint"]:
        raise ValueError("Frozen comparison contract changed")
    for name, expected in spec["frozen_inputs"].items():
        if sha(args.output/name) != expected:
            raise ValueError("Frozen study input changed: "+name)
    source = Path(__file__).resolve().parents[2]
    for name, expected in spec["source_sha256"].items():
        if sha(source/name) != expected:
            raise ValueError("Frozen computational source changed: "+name)
    return spec


def planner_inputs(args):
    from .clearance_expansion_study import native_inputs
    from .intergrid_library_plan import libraries
    spec = frozen_check(args)
    _, _, _, ref = native_inputs(args)
    raw = read(args.output/"shot-library-rows.json")
    libs = libraries(raw["original"], raw["additions"])
    model, history, neural = (read(args.output/name) for name in ("route-models.json", "route-history.json", "neural-checkpoint.json"))
    kwargs = dict(prior=read(args.output/"training-prior.json"), calibration=read(args.output/"calibration.json"),
                  home=read(args.output/"setup-home.json"))
    config = dict(scene=ref["scene"], calibration=kwargs["calibration"]["score_config"],
                  assets=str(args.assets), rule=spec["flight_ranking"]["rule"])
    return ref, libs, model, history, neural, kwargs, config


def repeated_requests(choice, method, repeats=3):
    """Repeat one frozen checked command; only release identity changes."""
    from .planning_budget_plan import _wrap_choice
    import copy
    slot = _wrap_choice(choice, method)
    if slot['request'] is None: return []
    return [dict(copy.deepcopy(slot['request']),id=slot['request']['id']+f'-repeat-{n}',
                 repeat=n,source_request_id=slot['request']['id']) for n in range(1,repeats+1)]


def prepare(args):
    from .stalled_reference_plan import StalledReferencePlanner
    from .guarded_board_fallback_plan import GuardedBoardFallbackPlanner
    from .planning_budget_plan import _wrap_choice
    from .intergrid_system_study import require_storage
    spec = frozen_check(args)
    if (args.output/'plans.json').exists(): raise ValueError('Preserve frozen completed choices')
    ref,libs,model,history,neural,kwargs,config=planner_inputs(args)
    report=read(args.output/'report.json')
    def prepare_group(directory, layouts, order, diagnostic=False):
        by_id={r['id']:r for r in layouts};sites={}
        with ExitStack() as stack:
            planners={method:stack.enter_context(cls('warm_early',libs,model,history,config,neural,
                reference=ref,assets=args.assets,directory=directory/('native-'+method),
                engine_kwargs=kwargs,budget_s=PLANNING_BUDGET_S))
                for method,cls in zip(METHODS,(StalledReferencePlanner,GuardedBoardFallbackPlanner),strict=True)}
            pool=stack.enter_context(ThreadPoolExecutor(max_workers=2))
            for identity in order:
                futures={pool.submit(planners[v].plan,by_id[identity]):v for v in METHODS}
                for future in as_completed(futures):
                    variant,site=futures[future],future.result()
                    if site['status'] not in {'completed','deadline'}: raise RuntimeError('Planner worker failed: '+str(site))
                    sites[identity,variant]=site
                    atomic(directory/'preparation'/variant/(identity+'.json'),site)
                    if diagnostic: report['diagnostic_progress']=dict(prepared=len(sites),planned=2)
                    else: report['progress']['prepared']=len(sites)
                    publish(args,report,'preparing_diagnostic' if diagnostic else 'preparing')
                    require_storage(args.output)
        wrappers=[dict(variant=v,site=sites[r['id'],v]) for r in layouts for v in METHODS]
        atomic(directory/'plans.json',wrappers)
        return wrappers
    layouts=read(args.output/'layouts.json')['layouts']
    wrappers=prepare_group(args.output,layouts,spec['processing_order'])
    diagnostic_layout=read(args.output/'diagnostic-layout.json')['layout']
    diagnostic=prepare_group(args.output/'diagnostic',[diagnostic_layout],[diagnostic_layout['id']],True)
    diagnostic_slots=[_wrap_choice(w['site']['choice'],w['variant']) for w in diagnostic]
    diagnostic_requests={w['variant']:repeated_requests(w['site']['choice'],w['variant']) for w in diagnostic}
    atomic(args.output/'diagnostic/choices.json',dict(slots=diagnostic_slots,requests=diagnostic_requests,
        bindings={name:sha(args.output/name) for name in ('contract.json','diagnostic-layout.json','diagnostic/plans.json')}))
    slots=[_wrap_choice(w['site']['choice'],w['variant']) for w in wrappers]
    bindings={name:sha(args.output/name) for name in (*spec['frozen_inputs'],'contract.json','plans.json','diagnostic/plans.json','diagnostic/choices.json')}
    atomic(args.output/'choices.json',dict(slots=slots,bindings=bindings))
    report['progress']['planned_releases']=sum(s['request'] is not None for s in slots)
    report['diagnostic_progress'].update(completed_releases=0,planned_releases=sum(map(len,diagnostic_requests.values())))
    publish(args,report,'prepared')


def run_native(args):
    from .clearance_expansion_study import native_inputs
    from .cell1050_heading_native import HeadingEngine
    from .direct_path_first import DirectPathEngine, hydrate_direct_resolution
    from .table_correction_native import MatchedEngine
    from .cell1050_rebound_study import annotate, control_requests
    from .coverage_training_study import append
    from .gpu_slot import foreign_processes, gpu_slot
    from .intergrid_route_labels import annotate_routes
    from .intergrid_system_study import gpu_deadline, require_storage
    from .parallel_capture_grid_study import NATIVE_BATCH, control_mirror
    from .parallel_capture_grid_workers import load_resolution
    from .stalled_reference_study import execute_mixed
    from .table_training_study import control_gate

    frozen_check(args)
    _, _, _, ref = native_inputs(args)
    output = args.output; report = read(output/"report.json"); choices = read(output/"choices.json")
    if report["status"] != "running" or report["phase"] != "prepared":
        raise ValueError("Complete and freeze all forty fresh and two diagnostic choices before native test")
    bindings = dict(choices["bindings"], **{"choices.json":sha(output/"choices.json")})
    if any(sha(output/name) != value for name,value in bindings.items()):
        raise ValueError("Pre-release frozen choices changed")
    planned = {v:[s["request"] for s in choices["slots"] if s["method"] == v and s["request"] is not None] for v in METHODS}
    diagnostic_requests=read(output/'diagnostic/choices.json')['requests']
    with (output/"native-started.json").open("x") as stream:
        json.dump(dict(bindings=bindings, requests_sha256=digest(planned), diagnostic_requests_sha256=digest(diagnostic_requests), timestamp_unix=time.time()), stream)
    kwargs = dict(prior=read(output/"training-prior.json"), calibration=read(output/"calibration.json"), home=read(output/"setup-home.json"))
    expected_options = read(args.parent_study/"native-yaw/effective-options.json")
    expected_geometry = read(args.parent_study/"native-yaw/native-compiled-geometry.json")
    packet_number = 0

    def execute(engine, batch, phase, directory, control=False):
        nonlocal packet_number
        if foreign_processes():
            raise RuntimeError("Foreign process appeared on leased 4090")
        require_storage(output)
        if any(sha(output/name) != value for name,value in bindings.items()):
            raise ValueError("Frozen choices changed during native execution")
        if not control:
            for xy in {tuple(r["bucket_xy_m"]) for r in batch}:
                (hydrate_direct_resolution if isinstance(engine,DirectPathEngine) else load_resolution)(engine,list(xy))
        packet = f"requests/batch-{packet_number:05d}.json"; packet_number += 1
        atomic(output/packet,dict(phase=phase,control=control,adapter_directory=directory,rows=batch))
        rows,timing = engine.evaluate(batch,phase)
        if [r["id"] for r in rows] != [r["id"] for r in batch] or any(r["native_release"] is not True for r in rows):
            raise RuntimeError("Frozen command did not release exactly once")
        adapter = engine.engine.adapter if control else engine.adapter
        if adapter.env.effective_options != expected_options or adapter.env.compiled_geometry != expected_geometry:
            raise ValueError("Native physics or geometry differs from calibrated controls")
        if not control:
            annotate(rows,output/directory); annotate_routes(rows,output/directory)
            for row,request in zip(rows,batch,strict=True): row["intended_route"] = request["intended_route"]
        for row in rows: row.update(request_packet=packet,request_packet_sha256=sha(output/packet),adapter_directory=directory)
        append(output/"rows.jsonl",rows); append(output/"timing.jsonl",[dict(packet=packet,phase=phase,timing=timing)])
        return rows

    def waiting(status):
        report["gpu_wait"] = status; publish(args,report,"waiting_for_4090")

    with gpu_slot(args.gpu_lock,timeout_s=args.gpu_wait_s,poll_s=30,notify=waiting):
        started = time.monotonic()
        with gpu_deadline(GPU_CAP_S):
            report.pop("gpu_wait",None)
            engines = {v:{"reference":HeadingEngine(ref,args.assets,output/("native-"+v),**kwargs),
                          "direct_first":DirectPathEngine(ref,args.assets,output/("native-"+v),**kwargs)} for v in METHODS}
            diagnostic_engines={v:{'reference':HeadingEngine(ref,args.assets,output/'diagnostic'/('native-'+v),**kwargs),
                'direct_first':DirectPathEngine(ref,args.assets,output/'diagnostic'/('native-'+v),**kwargs)} for v in METHODS}
            sources = read(args.parent_study/"control-replays.json")
            mirror = control_mirror(args.parent_study,output,sources,False)
            controls = MatchedEngine(ref,args.assets,output/"native-controls",parent=mirror,sources=sources,
                prior=read(args.parent_study/"old-prior.json"),calibration=kwargs["calibration"],home=kwargs["home"])
            adapters = [e.adapter for group in (engines,diagnostic_engines) for pair in group.values() for e in pair.values()]+[controls.engine.adapter]
            factory, allocated = adapters[0].environment_factory, []
            def shared_factory():
                if not allocated: allocated.append(factory())
                return allocated[0]
            for adapter in adapters: adapter.environment_factory = shared_factory
            for phase in ("controls_pre","diagnostic","test","controls_post"):
                if phase in ("controls_pre","controls_post"):
                    gate = control_gate(execute(controls,control_requests(sources,phase),phase,"native-controls",True))
                    report["controls"][phase] = gate
                    if gate["status"] != "passed": raise RuntimeError("Matched control gate failed: "+phase)
                    publish(args,report,phase); continue
                if phase=='diagnostic':
                    for method in METHODS:
                        batch=diagnostic_requests[method]
                        if batch:
                            rows=execute_mixed(diagnostic_engines[method],batch,'diagnostic_'+method,'diagnostic/native-'+method,execute)
                            report['diagnostic_progress']['completed_releases']+=len(rows)
                            publish(args,report,'diagnostic_'+method)
                    continue
                for offset in range(0,max(map(len,planned.values())),NATIVE_BATCH):
                    for method in METHODS:
                        batch=planned[method][offset:offset+NATIVE_BATCH]
                        if batch:
                            rows=execute_mixed(engines[method],batch,"test_"+method,"native-"+method,execute)
                            report["progress"]["completed_releases"] += len(rows)
                            report["gpu_lease_s"] = time.monotonic()-started
                            publish(args,report,"test_"+method)
        report["gpu_lease_s"] = time.monotonic()-started
    report["native_completed"] = True; publish(args,report,"native_completed")


def summarize(layouts, slots, observations, wrappers):
    from .direct_path_first_audit import summarize as summarize_independent
    return summarize_independent(layouts,slots,observations,wrappers,variants=METHODS)


def diagnostic_summary(layout, artifact, rows):
    from .hybrid_miss_audit import classify
    from .intergrid_route_selector_plan import STANDARD_CAPTURES
    result=dict(bucket_xy_m=layout['bucket_xy_m'],repeats=3,methods={})
    for slot in artifact['slots']:
        method=slot['method'];requests=artifact['requests'][method]
        outcomes=[classify(rows[r['id']]) for r in requests]
        result['methods'][method]=dict(requested=3,checked_shot=slot['request'] is not None,
            released=len(outcomes),captures=sum(v in STANDARD_CAPTURES for v in outcomes),outcomes=outcomes)
    return result


def fallback_counts(wrappers):
    rows = [w["site"].get("board_fallback",{}) for w in wrappers if w["variant"] == "guarded"]
    return dict(activated=sum(r.get("triggered") is True for r in rows),
                changed_choice=sum(bool(r.get("selected_fallback_candidate_id")) for r in rows))


def finalize(args):
    from .parallel_capture_grid_audit import observed_rows
    frozen_check(args); report=read(args.output/"report.json")
    if report.get("native_completed") is not True: raise ValueError("Finish all frozen drops and controls first")
    rows={r["id"]:r for r in observed_rows(args.output/"rows.jsonl") if r["adapter_directory"] != "native-controls"}
    wrappers=read(args.output/"plans.json")
    fresh={k:r for k,r in rows.items() if not r['adapter_directory'].startswith('diagnostic/')}
    report["comparison"] = summarize(read(args.output/"layouts.json")["layouts"],read(args.output/"choices.json")["slots"],fresh,wrappers)
    report['diagnostic']=diagnostic_summary(read(args.output/'diagnostic-layout.json')['layout'],
        read(args.output/'diagnostic/choices.json'),rows)
    report["fallback"] = fallback_counts(wrappers)
    atomic(args.output/"evidence-manifest.json",{str(p.relative_to(args.output)):sha(p) for p in sorted(args.output.rglob("*"))
        if p.is_file() and p.name not in {"report.json","evidence-manifest.json"}})
    report.update(status="completed",audit_status="pending",evidence_manifest_sha256=sha(args.output/"evidence-manifest.json"),
                  next="Review independently audited capture and timing results before any live planner change.")
    publish(args,report,"completed")


def argument_parser():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("parent-study","parent-audit","source-study","source-audit", "reference","assets","evidence-root","output","gpu-lock"):
        parser.add_argument("--"+name,type=Path,required=True)
    parser.add_argument("--display-report",type=Path); parser.add_argument("--audit-output",type=Path)
    parser.add_argument("--frozen-source-root",type=Path)
    parser.add_argument("--gpu-wait-s",type=int,default=12*3600)
    parser.add_argument("--stage",required=True,choices=("initialize","prepare","run","finalize","preflight","audit"))
    return parser


def main():
    args=argument_parser().parse_args()
    try:
        if args.stage == "initialize": initialize(args)
        elif args.stage == "prepare": prepare(args)
        elif args.stage == "run": run_native(args)
        elif args.stage == "finalize": finalize(args)
        else:
            from .guarded_board_fallback_audit import audit
            if args.audit_output is None or args.audit_output.exists() or args.audit_output.resolve().is_relative_to(args.output.resolve()):
                raise ValueError("Use a new independent receipt outside evidence")
            if args.stage=='preflight':
                from .guarded_board_fallback_audit import preflight
                result=preflight(args)
            else: result=audit(args)
            atomic(args.audit_output,result)
            print(json.dumps({k:v for k,v in result.items() if k != "comparison"}),flush=True)
    except Exception as error:
        if (args.output/"report.json").exists() and args.stage not in {"audit","preflight"}:
            report=read(args.output/"report.json"); report.update(status="failed",error=f"{type(error).__name__}: {error}")
            publish(args,report,"failed_"+args.stage)
        raise


if __name__ == "__main__": main()
