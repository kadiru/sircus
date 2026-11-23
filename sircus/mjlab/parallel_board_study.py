"""Six-target development comparison of concurrent board-pose planning.

All twelve CPU decisions freeze before native64. Each checked decision gets
three identical repeats. These six selected known targets are development
evidence, not unseen generalization. Historical recipes and live services stay separate.
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
from .sweep_worker import atomic

VERSION = "sircus-parallel-board-study-v1"
METHODS = ("baseline", "parallel")
NATIVE_CHANNELS = ("baseline", "parallel/original", "parallel/alternative")
POSITION_COUNT = 6
PROCESSING_SEED = 2026112309
PLANNING_BUDGET_S = 30.
GPU_CAP_S = 1800
SOURCE_MISSES = ("guarded-board-test-03", "guarded-board-test-09", "guarded-board-test-13", "guarded-board-test-15", "guarded-board-test-19")
REPEATS = 3
SOURCE_VERSION = "sircus-guarded-board-fallback-study-v1"


def make_layouts(source):
    """Select six declared development targets, never claim unseen evaluation."""
    previous = read(source / "layouts.json")["layouts"]
    outcomes = {r["id"]: r["outcomes"] for r in read(source / "report.json")["comparison"]["positions"]}
    layouts = []
    for identity in SOURCE_MISSES:
        if outcomes[identity] != dict(baseline="flight_miss", guarded="flight_miss"):
            raise ValueError("The declared source target is not a paired miss: " + identity)
        row = next(r for r in previous if r["id"] == identity)
        new_id = "parallel-board-miss-" + identity.rsplit("-", 1)[1]
        layouts.append(dict(row, id=new_id, context_id=new_id, split="test",
                            source_id=identity, role="previous_miss", stratum="selected development miss"))
    row = read(source / "diagnostic-layout.json")["layout"]
    diagnostic = read(source / "report.json")["diagnostic"]["methods"]
    if any(diagnostic[v]["captures"] != 3 or diagnostic[v]["released"] != 3 for v in ("baseline", "guarded")):
        raise ValueError("The source regression control must retain its three successful repeats")
    layouts.append(dict(row, id="parallel-board-known-regression", context_id="parallel-board-known-regression",
                        source_id=row["id"], role="known_regression", split="test", stratum="known development regression control"))
    return layouts


def publish(args, report, phase=None):
    if phase is not None:
        report["phase"] = phase
    report["updated_at_unix"] = time.time()
    atomic(args.output/"report.json", report)
    if args.display_report:
        display = read(args.display_report) if args.display_report.exists() else {}
        display["parallel_board_study"] = report
        atomic(args.display_report, display)
    print(json.dumps({k: report[k] for k in ("status", "phase", "progress", "gpu_wait", "error") if k in report}), flush=True)


def initialize(args):
    from .clearance_expansion_study import native_inputs
    from .fast_planning_study import COPIED_INPUTS
    from .intergrid_route_refit_inputs import _audited_source
    from .intergrid_route_labels import LABEL_VERSION
    from .intergrid_system_study import require_storage
    from .parallel_board_plan import RULE
    native, parent_binding, source, _ = native_inputs(args)
    files = (*COPIED_INPUTS, "layouts.json", "diagnostic-layout.json", "setup-home.json", "cpu-equivalence.json", "choices.json", "plans.json")
    _, _, _, binding, previous = _audited_source(args.source_study, args.source_audit, SOURCE_VERSION, files)
    if (previous["identity"], previous["scene"]) != (native["identity"], native["scene"]):
        raise ValueError("Current planner and calibrated native scene differ")
    layouts = make_layouts(args.source_study)
    args.output.mkdir(parents=True, exist_ok=False)
    require_storage(args.output)
    atomic(args.output / "layouts.json", dict(layouts=layouts, scope="development", source_ids=[r["source_id"] for r in layouts]))
    for name in (*COPIED_INPUTS, "setup-home.json", "cpu-equivalence.json"):
        shutil.copyfile(args.source_study / name, args.output / name)
    order = [r["id"] for r in layouts]; random.Random(PROCESSING_SEED).shuffle(order)
    spec = dict(version=VERSION, parent=parent_binding, source_parent=binding,
        source_paths=dict(study=str(args.source_study.resolve()),audit=str(args.source_audit.resolve()),parent_study=str(args.parent_study.resolve()),parent_audit=str(args.parent_audit.resolve())),
        identity=native["identity"], scene=native["scene"], flight_ranking=previous["flight_ranking"],
        methods=list(METHODS), test_positions=6, repeats=3, physics_substeps=64, label_version=LABEL_VERSION,
        planning_budget_s=30., reference_path_s=5., early_score_m=.005, scope="development",
        processing_seed=PROCESSING_SEED, processing_order=order, parallel_rule=RULE, gpu_cap_s=GPU_CAP_S,
        frozen_inputs={p.name:sha(p) for p in args.output.glob("*.json")}, source_sha256=source_manifest(source),
        baseline="Unchanged StalledReferencePlanner; equal30s decision budget", max_native_releases=54,
        limitations="Selected known development targets; privileged geometry; held-endpoint native64; no retraining or live promotion")
    spec["fingerprint"] = digest(spec); atomic(args.output / "contract.json", spec)
    publish(args, dict(version=VERSION, status="running", phase="initialized", audit_status="pending", scope="development",
        physics_substeps=64, budgets=dict(positions=6, repeats=3, planning_s=30.), controls={}, gpu_lease_s=0., gpu_cap_s=GPU_CAP_S,
        progress=dict(prepared=0, planned=12, completed_releases=0, planned_releases=None),
        next="Freeze all twelve decisions before any repeated native drop. Existing live planner unchanged."))


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


def native_channels(wrappers):
    """Keep each checked command in the proof directory that admitted it."""
    channels={v:[] for v in NATIVE_CHANNELS}
    for wrapper in wrappers:
        site,method=wrapper["site"],wrapper["variant"]
        lane="baseline" if method=="baseline" else "parallel/"+site["selected_adapter_directory"]
        if lane not in channels: raise ValueError("Unknown selected proof namespace")
        channels[lane].extend(repeated_requests(site["choice"],method))
    return channels


def prepare(args):
    import os
    from .stalled_reference_plan import StalledReferencePlanner
    from .parallel_board_plan import ParallelBoardPlanner
    from .planning_budget_plan import _wrap_choice
    from .intergrid_system_study import require_storage
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "": raise ValueError("CPU preparation requires hidden GPUs")
    spec = frozen_check(args)
    if (args.output / "plans.json").exists(): raise ValueError("Preserve frozen completed choices")
    ref,libs,model,history,neural,kwargs,config = planner_inputs(args)
    report=read(args.output / "report.json"); layouts=read(args.output / "layouts.json")["layouts"]
    by_id={r["id"]:r for r in layouts}; sites={}
    with ExitStack() as stack:
        planners={method:stack.enter_context(cls("warm_early",libs,model,history,config,neural,
            reference=ref,assets=args.assets,directory=args.output / ("native-"+method),
            engine_kwargs=kwargs,budget_s=30.))
            for method,cls in zip(METHODS,(StalledReferencePlanner,ParallelBoardPlanner),strict=True)}
        pool=stack.enter_context(ThreadPoolExecutor(max_workers=2))
        for identity in spec["processing_order"]:
            futures={pool.submit(planners[v].plan,by_id[identity]):v for v in METHODS}
            for future in as_completed(futures):
                method,site=futures[future],future.result()
                if site["status"] not in {"completed","deadline"}: raise RuntimeError("Planner worker failed: "+str(site))
                sites[identity,method]=site
                atomic(args.output / "preparation" / method / (identity+".json"),site)
                report["progress"]["prepared"]=len(sites);publish(args,report,"preparing");require_storage(args.output)
    wrappers=[dict(variant=v,site=sites[r["id"],v]) for r in layouts for v in METHODS]
    atomic(args.output / "plans.json",wrappers)
    slots=[_wrap_choice(w["site"]["choice"],w["variant"]) for w in wrappers]
    requests={v:[r for w in wrappers if w["variant"]==v for r in repeated_requests(w["site"]["choice"],v)] for v in METHODS}
    bindings={name:sha(args.output/name) for name in (*spec["frozen_inputs"],"contract.json","plans.json")}
    atomic(args.output / "choices.json",dict(slots=slots,requests=requests,channels=native_channels(wrappers),bindings=bindings))
    report["progress"]["planned_releases"]=sum(map(len,requests.values()))
    publish(args,report,"prepared")


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
        raise ValueError("Complete and freeze all twelve development choices before native test")
    bindings = dict(choices["bindings"], **{"choices.json":sha(output/"choices.json")})
    if any(sha(output/name) != value for name,value in bindings.items()):
        raise ValueError("Pre-release frozen choices changed")
    planned = choices["channels"]
    with (output/"native-started.json").open("x") as stream:
        json.dump(dict(bindings=bindings, requests_sha256=digest(planned), timestamp_unix=time.time()), stream)
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
                          "direct_first":DirectPathEngine(ref,args.assets,output/("native-"+v),**kwargs)} for v in NATIVE_CHANNELS}
            sources = read(args.parent_study/"control-replays.json")
            mirror = control_mirror(args.parent_study,output,sources,False)
            controls = MatchedEngine(ref,args.assets,output/"native-controls",parent=mirror,sources=sources,
                prior=read(args.parent_study/"old-prior.json"),calibration=kwargs["calibration"],home=kwargs["home"])
            adapters = [e.adapter for pair in engines.values() for e in pair.values()]+[controls.engine.adapter]
            factory, allocated = adapters[0].environment_factory, []
            def shared_factory():
                if not allocated: allocated.append(factory())
                return allocated[0]
            for adapter in adapters: adapter.environment_factory = shared_factory
            for phase in ("controls_pre","test","controls_post"):
                if phase in ("controls_pre","controls_post"):
                    gate = control_gate(execute(controls,control_requests(sources,phase),phase,"native-controls",True))
                    report["controls"][phase] = gate
                    if gate["status"] != "passed": raise RuntimeError("Matched control gate failed: "+phase)
                    publish(args,report,phase); continue
                for offset in range(0,max(map(len,planned.values())),NATIVE_BATCH):
                    for method in NATIVE_CHANNELS:
                        batch=planned[method][offset:offset+NATIVE_BATCH]
                        if batch:
                            rows=execute_mixed(engines[method],batch,"test_"+method,"native-"+method,execute)
                            report["progress"]["completed_releases"] += len(rows)
                            report["gpu_lease_s"] = time.monotonic()-started
                            publish(args,report,"test_"+method)
        report["gpu_lease_s"] = time.monotonic()-started
    report["native_completed"] = True; publish(args,report,"native_completed")


def summarize(layouts,choices,rows,wrappers):
    from .hybrid_miss_audit import classify
    from .intergrid_route_selector_plan import STANDARD_CAPTURES
    from .fast_planning_audit import latency as latency_summary
    sites={(w["site"]["layout"]["id"],w["variant"]):w["site"] for w in wrappers}
    result=dict(positions=[],methods={},latency={},decision_latency={})
    for layout in layouts:
        item=dict(id=layout["id"],source_id=layout["source_id"],bucket_xy_m=layout["bucket_xy_m"],role=layout["role"],methods={})
        for method in METHODS:
            site=sites[layout["id"],method]
            expected=repeated_requests(site["choice"],method)
            outcomes=[classify(rows[r["id"]]) for r in expected]
            item["methods"][method]=dict(requested=3,checked_shot=bool(expected),released=len(expected),
                captures=sum(o in STANDARD_CAPTURES for o in outcomes),outcomes=outcomes,
                request_wall_s=site["request_wall_s"],decision_latency_s=site["decision_latency_s"])
        state=sites[layout["id"],"parallel"]
        item["parallel_lane"]=lane_summary(state)
        result["positions"].append(item)
    for method in METHODS:
        values=[r["methods"][method] for r in result["positions"]]
        result["methods"][method]=dict(positions=len(layouts),requested=3*len(layouts),
            checked_positions=sum(v["checked_shot"] for v in values),releases=sum(v["released"] for v in values),
            captures=sum(v["captures"] for v in values),reliable_positions=sum(v["captures"]==3 for v in values))
        result["latency"][method]=latency_summary([v["request_wall_s"] for v in values])
        result["decision_latency"][method]=latency_summary([v["decision_latency_s"] for v in values])
    return result


def lane_summary(state):
    # Populated from the immutable planner receipt, never inferred from drops.
    alternative=(state.get("lanes") or {}).get("alternative") or {}
    evidence=alternative.get("board_alternative") or {}
    return dict(triggered=state.get("trigger") is not None,changed_choice=state.get("selected_lane")=="alternative",
        checked_candidates=len(evidence.get("records",[])),admitted_candidates=len(evidence.get("admitted",[])))


def finalize(args):
    from .parallel_capture_grid_audit import observed_rows
    frozen_check(args); report=read(args.output / "report.json")
    if report.get("native_completed") is not True: raise ValueError("Finish all frozen drops and controls first")
    rows={r["id"]:r for r in observed_rows(args.output / "rows.jsonl") if r["adapter_directory"]!="native-controls"}
    wrappers=read(args.output / "plans.json")
    report["comparison"]=summarize(read(args.output / "layouts.json")["layouts"],read(args.output / "choices.json"),rows,wrappers)
    lanes=[r["parallel_lane"] for r in report["comparison"]["positions"]]
    report["parallel"]=dict(activated=sum(s["triggered"] for s in lanes),changed_choice=sum(s["changed_choice"] for s in lanes))
    atomic(args.output / "evidence-manifest.json",{str(p.relative_to(args.output)):sha(p) for p in sorted(args.output.rglob("*"))
        if p.is_file() and p.name not in {"report.json","evidence-manifest.json"}})
    report.update(status="completed",audit_status="pending",evidence_manifest_sha256=sha(args.output / "evidence-manifest.json"),
        next="Development results only. Review the independent audit before any fresh comparison or live change.")
    publish(args,report,"completed")


def argument_parser():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("parent-study","parent-audit","source-study","source-audit", "reference","assets","output","gpu-lock"):
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
            from .parallel_board_audit import audit
            if args.audit_output is None or args.audit_output.exists() or args.audit_output.resolve().is_relative_to(args.output.resolve()):
                raise ValueError("Use a new independent receipt outside evidence")
            if args.stage=='preflight':
                from .parallel_board_audit import preflight
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
