"""Development recovery and fresh comparison of preserved checked shots.

Raw current/expanded preparation is unchanged. A small outcome-blind selection
layer chooses expanded's admission or retains current's exact checked command.
Native evaluation reuses the established engines, controls and native64 scorer.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
import copy
import json
from pathlib import Path
import random
import shutil
import time

from . import approach_comparison_study as shared
from .benchmark_verify import contained, digest, read, sha
from .preserved_shot_plan import METHODS, RAW_METHODS, description, select_pair, slots_for_pair
from .sweep_worker import atomic

VERSION = "sircus-preserved-shot-study-v1"
SEED = 2026112801
PROCESSING_SEED = 2026112808
PLANNING_S = 30.
NATIVE_BATCH_SIZE = 32
NATIVE_ADAPTERS = ("reference", "direct_first")
GPU_CAP_S = 3600
NATIVE_SCHEDULE_VERSION = "sircus-preserved-source-native-lanes-v1"
lines, publish, cpu_only = shared.lines, shared.publish, shared.cpu_only


def audited_binding(study, audit_path):
    study, audit_path = Path(study), Path(audit_path)
    report, audit = read(study/"report.json"), read(audit_path)
    if (report.get("status") != "completed" or audit.get("status") != "passed"
            or audit.get("report_sha256") != sha(study/"report.json")
            or audit.get("evidence_manifest_sha256") != sha(study/"evidence-manifest.json")
            or report.get("evidence_manifest_sha256") != sha(study/"evidence-manifest.json")):
        raise ValueError("Source study must have a passed audit bound to its completed evidence")
    manifest = read(study/"evidence-manifest.json")
    # Verify every immutable source file, not only the rows used in selection.
    for name, checksum in manifest.items():
        if sha(contained(study, name)) != checksum:
            raise ValueError("Audited source evidence changed: " + name)
    return dict(study=str(study.resolve()), audit=str(audit_path.resolve()),
        report_sha256=sha(study/"report.json"), audit_sha256=sha(audit_path),
        manifest_sha256=sha(study/"evidence-manifest.json"),
        contract_sha256=sha(study/"contract.json"))


def development_layouts(a):
    old = read(a.comparison_study/"layouts.json")["layouts"]
    slots = read(a.comparison_study/"choices.json")["slots"]
    refused = {s["context_id"] for s in slots if s["method"] == "expanded" and s["request"] is None}
    if len(refused) != 9:
        raise ValueError("Development source must contain the nine reported expanded setup declines")
    rows = []
    for row in old:
        if row["id"] not in refused:
            continue
        # Keep exact query identity: planner seed derivation can depend on IDs.
        # The cohort/provenance label (not a query mutation) marks these as known.
        rows.append(copy.deepcopy(row))
    return dict(layouts=rows, provenance=dict(kind="nine_known_setup_declines",
        source_layouts_sha256=sha(a.comparison_study/"layouts.json"),
        source_choices_sha256=sha(a.comparison_study/"choices.json"),
        excluded_from_fresh_evaluation=True))


def fresh_layouts(a, reference):
    from .intergrid_proposer_plan import make_splits, sample_splits
    original = read(a.neural_study/"layouts.json")
    previous = [row for split in original.values() for row in split["layouts"]]
    grid = read(a.parent_study/"grid.json")
    exclusions, inventory, geometry = set(), {}, None
    for root in a.history_root:
        _, provenance = make_splits(root, grid, scene=reference["scene"], assets=a.assets,
            previous_layouts=previous, seed=SEED, include_train=False)
        exclusions.update(map(tuple, provenance["excluded_coordinates_m"]))
        inventory[str(root.resolve())] = provenance["historical_files_sha256"]
        if geometry is not None and geometry != provenance["geometry"]:
            raise ValueError("Historical inventory changed placement geometry")
        geometry = provenance["geometry"]
    if geometry is None:
        raise ValueError("At least one history root is required")
    exclusions.update(tuple(xy) for xy in read(a.library/"all-positions.json"))
    for split in ("train", "validation"):
        exclusions.update(tuple(r["request"]["bucket_xy_m"]) for r in lines(a.library/(split+"-dataset.jsonl")))
    exclusions.update(tuple(r["bucket_xy_m"]) for r in read(a.library/"candidate-sources.json"))
    explicit = {}
    for path in (a.expanded_study/"layouts.json", a.comparison_study/"layouts.json", a.development_study/"layouts.json"):
        exclusions.update(tuple(r["bucket_xy_m"]) for r in read(path)["layouts"])
        explicit[str(path.resolve())] = sha(path)
    sampled = sample_splits(sorted(exclusions), geometry, grid, seed=SEED, include_train=False)["test"]
    rows = [dict(row, id=f"preserved-fresh-{i:02d}", context_id=f"preserved-fresh-{i:02d}", split="test")
            for i,row in enumerate(sampled)]
    if len(rows) != 50:
        raise ValueError("Expected 50 fresh positions")
    return dict(layouts=rows, provenance=dict(kind="fresh_disjoint_positions", seed=SEED,
        excluded_coordinates_m=sorted(exclusions), geometry=geometry, history_files=inventory,
        explicit_layout_files=explicit, minimum_separation_m=.005, layouts_sha256=digest(rows)))


def schedule(layouts):
    rng = random.Random(PROCESSING_SEED)
    order = [row["id"] for row in layouts]
    rng.shuffle(order)
    first = [RAW_METHODS[i % 2] for i in range(len(order))]
    rng.shuffle(first)
    return order, {identity:[method,next(m for m in RAW_METHODS if m != method)]
                   for identity,method in zip(order,first,strict=True)}


def freeze(a):
    from .corrected_grid import source_manifest
    cpu_only()
    _, _, reference, _, _ = shared.source_inputs(a)
    shared.library_inputs(a)
    shared.expanded_parent_binding(a)
    parent = audited_binding(a.comparison_study, a.comparison_audit)
    original = read(a.comparison_study/"contract.json")
    if original.get("version") != shared.VERSION:
        raise ValueError("The original approach comparison is required")
    development = None
    if a.cohort == "fresh":
        if a.development_study is None or a.development_audit is None:
            raise ValueError("Fresh evaluation follows the completed audited nine-position test")
        development = audited_binding(a.development_study,a.development_audit)
        dev = read(a.development_study/"contract.json")
        if dev.get("version") != VERSION or dev.get("cohort") != "development" or dev.get("preservation") != description():
            raise ValueError("Development must test this identical frozen preservation rule")
    archive = development_layouts(a) if a.cohort == "development" else fresh_layouts(a,reference)
    if a.output.exists():
        raise ValueError("Preserve prior evidence; choose a fresh output")
    a.output.mkdir(parents=True,exist_ok=False)
    report = dict(version=VERSION,status="running",phase="freezing",audit_status="pending",cohort=a.cohort,
        controls={},progress=dict(prepared=0,completed_releases=0,planned_releases=None),
        gpu_lease_s=0.,gpu_wait_s=0.,training=dict(performed=False), live_default_changed=False)
    publish(a,report)
    for name in original["frozen_inputs"]:
        if name == "layouts.json":
            continue
        if sha(a.comparison_study/name) != original["frozen_inputs"][name]:
            raise ValueError("Frozen comparison input changed: "+name)
        shutil.copyfile(a.comparison_study/name,a.output/name)
    atomic(a.output/"layouts.json",archive)
    order, method_order = schedule(archive["layouts"])
    repeats = 3 if a.cohort == "development" else 1
    contract = copy.deepcopy(original)
    contract.pop("fingerprint",None)
    contract.update(version=VERSION,cohort=a.cohort,methods=list(METHODS),raw_methods=list(RAW_METHODS),
        source_comparison=parent,development=development,preservation=description(),
        source_sha256=source_manifest(Path(__file__).resolve().parents[2]),
        seed=SEED,processing_seed=PROCESSING_SEED,processing_order=order,method_order=method_order,
        test_positions=len(archive["layouts"]),attempts_per_method_per_position=repeats,
        max_native_releases=2*len(archive["layouts"])*repeats+18,
        frozen_inputs={p.name:sha(p) for p in a.output.iterdir() if p.name != "report.json"},
        native_execution=dict(version=NATIVE_SCHEDULE_VERSION,batch_size=NATIVE_BATCH_SIZE,
            method_order=list(METHODS),source_order=list(RAW_METHODS),adapter_order=list(NATIVE_ADAPTERS),
            ordering="Round-robin result/source/adapter packets; frozen position/repeat order per lane"),
        scope="Two frozen CPU planning lanes; portfolio preserves exact current admission only on expanded decline. Held-endpoint native64 drops; no retraining, driven motion, perception, hardware or live promotion.")
    contract["fingerprint"] = digest(contract)
    atomic(a.output/"contract.json",contract)
    report["phase"] = "models_frozen"
    publish(a,report)


def frozen(a):
    contract = shared.frozen(a)
    if contract.get("version") != VERSION or contract.get("preservation") != description():
        raise ValueError("Unknown preserved-shot study or changed selection rule")
    return contract


def prepare(a):
    from .stalled_reference_plan import StalledReferencePlanner
    from .approach_comparison_plan import ApproachComparisonPlanner
    cpu_only()
    contract = frozen(a)
    reference,libs,model,history,checkpoints,kwargs,configs = shared.planner_inputs(a)
    shared.start_preparation(a.output,contract)
    layouts = read(a.output/"layouts.json")["layouts"]
    by_id = {row["id"]:row for row in layouts}
    results, selections = {}, {}
    report = read(a.output/"report.json")
    with ExitStack() as stack:
        planners = {m:stack.enter_context((StalledReferencePlanner if m=="current" else ApproachComparisonPlanner)(
            "warm_early",libs[m],model,history,configs[m],checkpoints[m],reference=reference,assets=a.assets,
            directory=a.output/("native-"+m),engine_kwargs=kwargs,budget_s=PLANNING_S)) for m in RAW_METHODS}
        pool = stack.enter_context(ThreadPoolExecutor(max_workers=2))
        def run(method,layout):
            start = time.perf_counter()
            site = planners[method].plan(layout)
            return site,dict(started_s=start,finished_s=time.perf_counter())
        for identity in contract["processing_order"]:
            timing = dict(pair_started_s=time.perf_counter(),lanes={})
            jobs = {m:pool.submit(run,m,by_id[identity]) for m in contract["method_order"][identity]}
            for method in RAW_METHODS:
                site,lane = jobs[method].result()
                results[identity,method] = site
                timing["lanes"][method] = lane
            timing["pair_finished_s"] = time.perf_counter()
            selections[identity] = select_pair(*(results[identity,m] for m in RAW_METHODS),timing)
            for method in RAW_METHODS:
                atomic(a.output/"preparation"/method/(identity+".json"),results[identity,method])
            atomic(a.output/"selection"/(identity+".json"),selections[identity])
            report.update(phase="preparing")
            report["progress"]["prepared"] = len(results)
            publish(a,report)
    plans = [dict(variant=m,site=results[l["id"],m]) for l in layouts for m in RAW_METHODS]
    selected = [selections[l["id"]] for l in layouts]
    atomic(a.output/"plans.json",plans)
    atomic(a.output/"selection.json",selected)
    slots = [slot for layout in layouts for slot in slots_for_pair(
        *(results[layout["id"],m] for m in RAW_METHODS),selections[layout["id"]],contract["attempts_per_method_per_position"])]
    proofs = {}
    for slot in slots:
        request = slot["request"]
        if request is None:continue
        for prefix in ("context_resolution","prepared_connection"):
            name = "native-"+slot["source_method"]+"/"+request[prefix+"_file"]
            if sha(contained(a.output,name)) != request[prefix+"_sha256"]:
                raise ValueError("Exact selected source proof unavailable")
            proofs[name] = request[prefix+"_sha256"]
    atomic(a.output/"admission-proof-bindings.json",proofs)
    atomic(a.output/"native-schedule.json",native_schedule(contract,slots))
    bindings = {p.relative_to(a.output).as_posix():sha(p) for p in a.output.rglob("*")
                if p.is_file() and p.name not in {"report.json","choices.json"}}
    atomic(a.output/"choices.json",dict(slots=slots,bindings=bindings))
    report.update(phase="prepared")
    report["progress"]["planned_releases"] = sum(s["request"] is not None for s in slots)
    publish(a,report)


def native_schedule(contract,slots):
    from .stalled_reference_plan import request_mode
    repeats = contract["attempts_per_method_per_position"]
    lookup = {(s["context_id"],s["method"],s["repeat_index"]):s for s in slots}
    expected = {(i,m,r) for i in contract["processing_order"] for m in METHODS for r in range(repeats)}
    if len(lookup) != len(slots) or set(lookup) != expected or len({s["id"] for s in slots}) != len(slots):
        raise ValueError("One unique frozen slot per method, position and repeat required")
    lanes = {(m,s,a):[] for m in METHODS for s in RAW_METHODS for a in NATIVE_ADAPTERS}
    for identity in contract["processing_order"]:
        for method in METHODS:
            for repeat in range(repeats):
                slot = lookup[identity,method,repeat]
                request = slot["request"]
                if request is None:continue
                if request["id"] != slot["id"] or request["method"] != method:
                    raise ValueError("Frozen slot and release identity differ")
                lanes[method,slot["source_method"],request_mode(request)].append(slot["id"])
    packets=[]
    for offset in range(0,max(map(len,lanes.values()),default=0),NATIVE_BATCH_SIZE):
        for (method,source,adapter),identities in lanes.items():
            ids=identities[offset:offset+NATIVE_BATCH_SIZE]
            if ids:packets.append(dict(method=method,source_method=source,adapter=adapter,request_ids=ids))
    return packets


def planned_requests(contract,slots):
    lookup={s["id"]:s["request"] for s in slots if s["request"] is not None}
    packets=native_schedule(contract,slots)
    return {method:[lookup[i] for p in packets if p["method"]==method for i in p["request_ids"]] for method in METHODS}


def require_preflight(a):
    from .preserved_shot_audit import require_preflight as verified
    return verified(a)


def run_native(a):
    from .cell1050_heading_native import HeadingEngine
    from .direct_path_first import DirectPathEngine,hydrate_direct_resolution
    from .cell1050_rebound_study import control_requests,annotate
    from .intergrid_route_labels import annotate_routes
    from .coverage_training_study import append
    from .gpu_slot import gpu_slot,foreign_processes
    from .intergrid_system_study import gpu_deadline,require_storage
    from .parallel_capture_grid_study import control_mirror,NATIVE_BATCH
    from .parallel_capture_grid_workers import load_resolution
    from .table_correction_native import MatchedEngine
    contract=frozen(a)
    require_preflight(a)
    _,_,reference,_,_=shared.source_inputs(a)
    report=read(a.output/"report.json")
    if report["phase"] not in {"prepared","waiting_for_4090"}:
        raise ValueError("Native starts only after complete frozen preparation")
    choices=read(a.output/"choices.json")
    bindings=dict(choices["bindings"],**{"choices.json":sha(a.output/"choices.json")})
    packets=native_schedule(contract,choices["slots"])
    if NATIVE_BATCH != NATIVE_BATCH_SIZE or packets!=read(a.output/"native-schedule.json"):
        raise ValueError("Frozen native batching changed")
    planned=planned_requests(contract,choices["slots"])
    started=dict(bindings=bindings,requests_sha256=digest(planned),native_schedule_sha256=sha(a.output/"native-schedule.json"),
                 preflight_sha256=sha(a.preflight_output))
    if (a.output/"native-started.json").exists():
        raise ValueError("No implicit retry of partial native execution")
    kwargs=dict(prior=read(a.output/"training-prior.json"),calibration=read(a.output/"calibration.json"),home=read(a.output/"setup-home.json"))
    options=read(a.parent_study/"native-yaw/effective-options.json")
    geometry=read(a.parent_study/"native-yaw/native-compiled-geometry.json")
    packet_number=0

    def execute(engine,batch,phase,directory,control=False,packet=None):
        nonlocal packet_number
        if foreign_processes():raise RuntimeError("Foreign workload appeared on leased4090")
        require_storage(a.output)
        if any(sha(a.output/name)!=value for name,value in bindings.items()):
            raise ValueError("Frozen evidence changed during native execution")
        if not control:
            for xy in {tuple(row["bucket_xy_m"]) for row in batch}:
                anchor=hydrate_direct_resolution(engine,list(xy)) if isinstance(engine,DirectPathEngine) else load_resolution(engine,list(xy))
                if anchor is None or {r["context_resolution_sha256"] for r in batch if tuple(r["bucket_xy_m"])==xy}!={anchor["sha256"]}:
                    raise ValueError("Exact selected-source resolution unavailable; replanning forbidden")
        name=f"requests/batch-{packet_number:05d}.json"
        packet_number+=1
        payload=dict(rows=batch,phase=phase,control=control,adapter_directory=directory,
            native_adapter_mode="matched_control" if control else packet["adapter"])
        if packet is not None:payload.update(method=packet["method"],source_method=packet["source_method"])
        atomic(a.output/name,payload)
        measured,timing=engine.evaluate(batch,phase)
        if [r["id"] for r in measured]!=[r["id"] for r in batch] or any(r["native_release"] is not True for r in measured):
            raise ValueError("Frozen admitted command did not release exactly once")
        adapter=engine.engine.adapter if control else engine.adapter
        if adapter.env.effective_options!=options or adapter.env.compiled_geometry!=geometry:
            raise ValueError("Native physics differs from matched controls")
        if not control:
            annotate(measured,a.output/directory)
            annotate_routes(measured,a.output/directory)
        shared.attach_native_scores(reference["scene"],kwargs["calibration"]["score_config"],measured,adapter.env.last_arrays)
        for row in measured:
            row.update(request_packet=name,request_packet_sha256=sha(a.output/name),adapter_directory=directory)
            if packet is not None:row.update(result_method=packet["method"],source_method=packet["source_method"])
        append(a.output/"rows.jsonl",measured)
        append(a.output/"timing.jsonl",[dict(packet=name,phase=phase,timing=timing)])
        return measured

    prior_wait=float(report.get("gpu_wait_s",0.))
    def waiting(value):
        report.pop("queue_expired",None);report.pop("error",None)
        report.update(phase="waiting_for_4090",gpu_wait=dict(status=value["status"],wait_s=value["wait_s"]),gpu_wait_s=prior_wait+value["wait_s"])
        publish(a,report)
    wait_start=time.monotonic()
    with gpu_slot(a.gpu_lock,timeout_s=a.gpu_wait_s,poll_s=30,notify=waiting):
        report["gpu_wait_s"]=prior_wait+time.monotonic()-wait_start
        if foreign_processes():raise RuntimeError("4090 has foreign work despite lease")
        frozen(a)
        if any(sha(a.output/name)!=value for name,value in bindings.items()):raise ValueError("Frozen inputs changed while queued")
        with (a.output/"native-started.json").open("x") as stream:json.dump(started,stream)
        begin=time.monotonic()
        with gpu_deadline(GPU_CAP_S):
            engines={source:dict(reference=HeadingEngine(reference,a.assets,a.output/("native-"+source),**kwargs),
                direct_first=DirectPathEngine(reference,a.assets,a.output/("native-"+source),**kwargs)) for source in RAW_METHODS}
            sources=read(a.parent_study/"control-replays.json")
            mirror=control_mirror(a.parent_study,a.output,sources,False)
            controls=MatchedEngine(reference,a.assets,a.output/"native-controls",parent=mirror,sources=sources,
                prior=read(a.parent_study/"old-prior.json"),calibration=kwargs["calibration"],home=kwargs["home"])
            adapters=[e.adapter for pair in engines.values() for e in pair.values()]+[controls.engine.adapter]
            factory,allocated=adapters[0].environment_factory,[]
            def shared_environment():
                if not allocated:allocated.append(factory())
                return allocated[0]
            for adapter in adapters:adapter.environment_factory=shared_environment
            lookup={r["id"]:r for group in planned.values() for r in group}
            for phase in ("controls_pre","test","controls_post"):
                if phase!="test":
                    gate=shared.board_first_controls(execute(controls,control_requests(sources,phase),phase,"native-controls",True))
                    report["controls"][phase]=gate
                    if gate["status"]!="passed":raise RuntimeError("Matched controls failed: "+phase)
                else:
                    for packet in packets:
                        engine=shared.native_engine(engines[packet["source_method"]],packet["adapter"])
                        batch=[lookup[i] for i in packet["request_ids"]]
                        execute(engine,batch,"test_"+packet["method"],"native-"+packet["source_method"],packet=packet)
                        report["progress"]["completed_releases"]+=len(batch)
                        report.update(phase="test_"+packet["method"],gpu_lease_s=time.monotonic()-begin)
                        publish(a,report)
                report["phase"]=phase;publish(a,report)
        report["gpu_lease_s"]=time.monotonic()-begin
    report.update(native_completed=True,phase="native_completed")
    report.pop("gpu_wait",None)
    publish(a,report)


def summarize(layouts,slots,rows,plans,selections):
    from statistics import median
    lookup={r["id"]:r for r in rows if r["adapter_directory"]!="native-controls"}
    expected={s["id"] for s in slots if s["request"] is not None}
    if set(lookup)!=expected or len(lookup)!=len([r for r in rows if r["adapter_directory"]!="native-controls"]):
        raise ValueError("One observed native drop per admitted slot required")
    plans_by={(p["site"]["layout"]["id"],p["variant"]):p["site"] for p in plans}
    selected_by={s["context_id"]:s for s in selections}
    methods,positions={},[]
    for method in METHODS:
        own=[s for s in slots if s["method"]==method]
        observed=[lookup[s["id"]] for s in own if s["request"] is not None]
        captures=sum(r["board_first_score"]["route_success"] for r in observed)
        invalid=sum(r["board_first_score"]["invalid_physics"] for r in observed)
        wall=[plans_by[l["id"],"current"]["request_wall_s"] if method=="current" else selected_by[l["id"]]["timing"]["pair_wall_s"] for l in layouts]
        methods[method]=dict(positions=len(layouts),attempt_slots=len(own),captures=captures,releases=len(observed),
            setup_rejections=len(own)-len(observed),invalid_physics=invalid,flight_misses=len(observed)-captures-invalid,
            capture_rate=captures/len(own),mean_request_s=sum(wall)/len(wall),median_request_s=median(wall),
            p90_request_s=shared.percentile90(wall),max_request_s=max(wall))
    for layout in layouts:
        row=dict(id=layout["id"],bucket_xy_m=layout["bucket_xy_m"],methods={},
            protected_source=selected_by[layout["id"]]["selected_source"],fallback_used=selected_by[layout["id"]]["fallback_used"])
        for method in METHODS:
            own=[s for s in slots if s["context_id"]==layout["id"] and s["method"]==method]
            successes=[bool(s["request"] is not None and lookup[s["id"]]["board_first_score"]["route_success"]) for s in own]
            row["methods"][method]=dict(captures=sum(successes),attempts=len(own),repeat_captures=successes,
                releases=sum(s["request"] is not None for s in own),all_repeats_captured=all(successes))
        positions.append(row)
    paired=dict(both=0,current_only=0,protected_only=0,neither=0)
    for row in positions:
        left,right=[row["methods"][m]["repeat_captures"][0] for m in METHODS]
        paired["both" if left and right else "current_only" if left else "protected_only" if right else "neither"]+=1
    return dict(methods=methods,positions=positions,paired_first_drop=paired,
        protected_fallback_positions=sum(s["fallback_used"] for s in selections),
        paired_timing=dict(mean_pair_wall_s=sum(s["timing"]["pair_wall_s"] for s in selections)/len(selections),
            mean_reserved_worker_window_s=sum(s["timing"]["reserved_worker_window_s"] for s in selections)/len(selections),
            measured_cpu_consumption=False,cpu_lanes=2,decision_budget_per_lane_s=PLANNING_S))


def finalize(a):
    frozen(a)
    report=read(a.output/"report.json")
    if report.get("native_completed") is not True:raise ValueError("Finish native drops and controls first")
    report["comparison"]=summarize(read(a.output/"layouts.json")["layouts"],read(a.output/"choices.json")["slots"],
        lines(a.output/"rows.jsonl"),read(a.output/"plans.json"),read(a.output/"selection.json"))
    atomic(a.output/"evidence-manifest.json",{p.relative_to(a.output).as_posix():sha(p) for p in sorted(a.output.rglob("*"))
        if p.is_file() and p.name not in {"report.json","evidence-manifest.json"}})
    report.update(status="completed",phase="completed",audit_status="pending",evidence_manifest_sha256=sha(a.output/"evidence-manifest.json"),
        next="Independent audit; retain current live planner until separately authorized promotion.")
    publish(a,report)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("neural-study","neural-audit","current-planner-study","current-planner-audit","library","expanded-study","expanded-audit",
                 "parent-study","parent-audit","reference","assets","comparison-study","comparison-audit","output","gpu-lock"):
        parser.add_argument("--"+name,type=Path,required=True)
    parser.add_argument("--history-root",action="append",type=Path,required=True)
    parser.add_argument("--cohort",choices=("development","fresh"),required=True)
    parser.add_argument("--development-study",type=Path)
    parser.add_argument("--development-audit",type=Path)
    parser.add_argument("--gpu-wait-s",type=int,default=43200)
    parser.add_argument("--stage",choices=("freeze","prepare","preflight","run","finalize","audit"),required=True)
    parser.add_argument("--preflight-output",type=Path)
    parser.add_argument("--audit-output",type=Path)
    args=parser.parse_args()
    if args.gpu_wait_s<=0:parser.error("Positive GPU wait limit required")
    if args.stage in {"preflight","audit"}:
        from .preserved_shot_audit import preflight,audit
        target=args.preflight_output if args.stage=="preflight" else args.audit_output
        if target is None or target.exists() or target.resolve().is_relative_to(args.output.resolve()):
            parser.error("Use a fresh audit receipt outside immutable evidence")
        result=(preflight if args.stage=="preflight" else audit)(args)
        atomic(target,result)
        print(json.dumps({k:v for k,v in result.items() if k!="comparison"}),flush=True)
        return
    existed=args.output.exists()
    try:dict(freeze=freeze,prepare=prepare,run=run_native,finalize=finalize)[args.stage](args)
    except Exception as exc:
        path=args.output/"report.json"
        if path.exists() and not(args.stage=="freeze" and existed):
            report=read(path)
            if args.stage=="run" and isinstance(exc,TimeoutError) and not(args.output/"native-started.json").exists():
                report.update(phase="waiting_for_4090",queue_expired=True,error="Queue expired before native start; explicit run retry permitted")
            else:report.update(status="failed",phase=args.stage+"_failed",error=type(exc).__name__+": "+str(exc))
            publish(args,report)
        raise


if __name__=="__main__":main()
