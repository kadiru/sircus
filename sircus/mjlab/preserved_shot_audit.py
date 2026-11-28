"""Independent, read-only audit of preserved checked commands and native outcomes.

The checker never selects from drop outcomes, re-plans a shot, or starts CUDA.
It reconstructs portfolio decisions from the original CPU planner evidence and
re-scores the authenticated dense traces with the separate scalar oracles.
"""
from __future__ import annotations

from collections import defaultdict
import copy
import math
from pathlib import Path
from statistics import median

import numpy as np

from .benchmark_verify import Checks, contained, digest, read, sha
from .approach_comparison_audit import (history_coordinates, inventory,
    verify_initial_trace, verify_historical_observation, verify_plans)

VERSION = "sircus-preserved-shot-independent-audit-v1"
RAW_METHODS = ("current", "expanded")
METHODS = ("current", "protected")
IDENTITY_FIELDS = frozenset(("id", "method"))


def physical_request(request):
    """Every request field except the two administrative release labels matters."""
    if request is None:
        return None
    return {key: value for key, value in request.items() if key not in IDENTITY_FIELDS}


def verify_pair(selection, current, expanded, checks):
    from .preserved_shot_plan import VERSION as rule_version
    sites = dict(current=current, expanded=expanded)
    label = current["layout"]["id"]
    checks.equal(expanded["layout"], current["layout"], label + "/same target in both CPU lanes")
    checks.equal(selection["version"], rule_version, label + "/frozen selection version")
    checks.equal(selection["context_id"], label, label + "/selection target")
    source = "expanded" if expanded["choice"]["request"] is not None else "current"
    checks.equal(selection["selected_source"], source, label + "/outcome-independent fallback rule")
    checks.equal(selection["fallback_used"], source == "current", label + "/fallback indicator")
    checks.equal(selection["source_choice_sha256"], {m: digest(s["choice"]) for m,s in sites.items()},
                 label + "/complete original choices preserved")
    checks.equal(selection["source_request_sha256"], {m: digest(s["choice"]["request"]) for m,s in sites.items()},
                 label + "/complete original requests preserved")
    checks.equal(selection["source_has_request"], {m:s["choice"]["request"] is not None for m,s in sites.items()},
                 label + "/admission flags include empty choices")
    timing = selection["timing"]
    start, end = timing["pair_started_s"], timing["pair_finished_s"]
    checks.require(all(type(v) in (float,int) and math.isfinite(v) for v in (start,end)) and end >= start,
                   label + "/finite ordered pair timing")
    checks.numeric(timing["pair_wall_s"], end-start, label + "/measured paired wall time")
    checks.equal(set(timing["lanes"]),set(RAW_METHODS),label + "/both CPU lanes measured")
    for method, site in sites.items():
        checks.require(site["status"] in {"completed", "deadline"}, label + "/no hidden worker error")
        checks.equal(site["choice"]["context_id"],label,label + "/source target " + method)
        checks.equal(site["choice"]["bucket_xy_m"],current["layout"]["bucket_xy_m"],label + "/source bucket " + method)
        interval = timing["lanes"][method]
        first, last = interval["started_s"],interval["finished_s"]
        checks.require(all(type(v) in (float,int) and math.isfinite(v) for v in (first,last)) and start <= first <= last <= end,
                       label + "/complete lane interval " + method)
        checks.require(last-first + 1e-6 >= site["request_wall_s"], label + "/pair includes full request time " + method)
    checks.numeric(timing["reserved_worker_window_s"],sum(s["decision_latency_s"] for s in sites.values()),
                   label + "/reported summed decision windows, not CPU utilization")
    return source


def verify_slots(slots, plans, selections, layouts, repeats, checks):
    """Independently reconstruct every slot, including full fallback metadata."""
    by_plan = {(p["site"]["layout"]["id"],p["variant"]):p["site"] for p in plans}
    checks.equal(len(by_plan),len(plans),"Unique raw planning evidence")
    checks.equal([r["context_id"] for r in selections],[l["id"] for l in layouts],
                 "Exactly one ordered selection receipt per target")
    by_selection = {s["context_id"]:s for s in selections}
    expected = []
    for layout in layouts:
        current,expanded = (by_plan[layout["id"],m] for m in RAW_METHODS)
        selection = by_selection[layout["id"]]
        protected_source = verify_pair(selection,current,expanded,checks)
        for method in METHODS:
            source = "current" if method == "current" else protected_source
            original = by_plan[layout["id"],source]["choice"]
            for repeat in range(repeats):
                identifier = f"{layout['id']}-{method}-r{repeat:02d}"
                slot = copy.deepcopy(original)
                slot.update(id=identifier,method=method,source_method=source,repeat_index=repeat,
                    source_choice_sha256=digest(original),source_request_sha256=digest(original["request"]),
                    fallback_used=method == "protected" and source == "current")
                if slot["request"] is not None:
                    slot["request"].update(id=identifier,method=method)
                expected.append(slot)
    checks.equal(slots,expected,"Exact source command and proof fields; only id/method relabeled")
    checks.equal(len({s["id"] for s in slots}),len(slots),"Unique release IDs across repeats")
    return by_plan


def expected_schedule(contract, slots):
    """Reconstruct homogeneous native batches without consulting observations."""
    from .stalled_reference_plan import request_mode
    modes = ("reference","direct_first")
    repeats = contract["attempts_per_method_per_position"]
    by_key = {(s["context_id"],s["method"],s["repeat_index"]):s for s in slots}
    if len(by_key) != len(slots):
        raise ValueError("Duplicate release slot")
    expected = {(identity,method,repeat) for identity in contract["processing_order"]
                for method in METHODS for repeat in range(repeats)}
    if set(by_key) != expected:
        raise ValueError("Missing or extra release slot")
    lanes = {(method,source,mode):[] for method in METHODS for source in RAW_METHODS for mode in modes}
    for identity in contract["processing_order"]:
        for method in METHODS:
            for repeat in range(repeats):
                slot = by_key[identity,method,repeat]
                if slot["request"] is not None:
                    lanes[method,slot["source_method"],request_mode(slot["request"])].append(slot["id"])
    result = []
    for offset in range(0,max(map(len,lanes.values()),default=0),32):
        for method in METHODS:
            for source in RAW_METHODS:
                for mode in modes:
                    ids = lanes[method,source,mode][offset:offset+32]
                    if ids:
                        result.append(dict(method=method,source_method=source,adapter=mode,request_ids=ids))
    return result


def verify_admission_proofs(study,slots,checks):
    """Keep original source namespaces even when a protected release is current."""
    proofs = {}
    for slot in slots:
        request = slot["request"]
        if request is None:
            continue
        for prefix in ("context_resolution","prepared_connection"):
            relative = "native-" + slot["source_method"] + "/" + request[prefix+"_file"]
            checks.equal(sha(contained(study,relative)),request[prefix+"_sha256"],slot["id"]+"/source proof "+prefix)
            proofs[relative] = request[prefix+"_sha256"]
    checks.equal(read(study/"admission-proof-bindings.json"),proofs,"Exact complete selected proof inventory")
    return proofs


def verify_native(a, checks, contract, report, choices):
    """Recompute all native outcomes and command associations from dense traces."""
    from . import board_first_score_oracle as revised
    from . import table_assisted_score_oracle as historical
    from .approach_comparison_study import board_first_controls, lines
    from .direct_path_first_audit import verify_heading
    from .cell1050_rebound_study import control_requests
    from .policy64_verify import audit_native_options
    from .table_assisted_env import _same
    study = a.output
    slots = choices["slots"]
    by_slot = {s["id"]:s for s in slots}
    packets = expected_schedule(contract,slots)
    selected = {s["id"]:s["request"] for s in slots if s["request"] is not None}
    planned = {method:[selected[identity] for packet in packets if packet["method"]==method
                       for identity in packet["request_ids"]] for method in METHODS}
    started = read(study/"native-started.json")
    expected_bindings = dict(choices["bindings"],**{"choices.json":sha(study/"choices.json")})
    checks.equal(started["bindings"],expected_bindings,"Native start binds all preflight evidence")
    for name,expected in started["bindings"].items():
        checks.equal(sha(contained(study,name)),expected,"Native-start binding/"+name)
    checks.equal(started["requests_sha256"],digest(planned),"Exact frozen native command ledger")
    checks.equal(started["native_schedule_sha256"],sha(study/"native-schedule.json"),"Native packet schedule frozen")
    rows = lines(study/"rows.jsonl")
    sources = read(a.parent_study/"control-replays.json")
    controls = {phase:control_requests(sources,phase) for phase in ("controls_pre","controls_post")}
    requests = dict(selected)
    requests.update({r["id"]:r for group in controls.values() for r in group})
    expected_order = ([r["id"] for r in controls["controls_pre"]]
        + [identity for packet in packets for identity in packet["request_ids"]]
        + [r["id"] for r in controls["controls_post"]])
    checks.equal([r["id"] for r in rows],expected_order,"Every native repeat/control in frozen packet order")
    checks.equal(len({r["id"] for r in rows}),len(rows),"Unique observed native identities")
    checks.equal(len(rows),len(requests),"All and only expected native drops")
    checks.require(len(rows)<=contract["max_native_releases"],"No extra native search releases")
    checks.equal(contract["max_native_releases"],len(slots)+18,"Evaluation cap includes every slot plus eighteen controls")
    packet_names = {row["request_packet"] for row in rows}
    checks.equal(packet_names,{p.relative_to(study).as_posix() for p in (study/"requests").glob("*.json")},
                 "No undeclared native request packets")
    observed_packets, seen = [],set()
    for row in rows:
        name = row["request_packet"]
        if name in seen:continue
        seen.add(name)
        packet = read(contained(study,name))
        measured = [r for r in rows if r["request_packet"]==name]
        checks.equal(packet["rows"],[requests[r["id"]] for r in measured],"Exact command bytes in packet/"+name)
        checks.require(all(r["request_packet_sha256"]==sha(study/name) and r["adapter_directory"]==packet["adapter_directory"]
                           for r in measured),"Authenticated packet and execution namespace/"+name)
        if row["id"] in by_slot:
            selected_slots = [by_slot[r["id"]] for r in measured]
            method, source = selected_slots[0]["method"], selected_slots[0]["source_method"]
            checks.require(all((s["method"],s["source_method"])==(method,source) for s in selected_slots),
                           "Homogeneous result and proof source lane/"+name)
            checks.equal(packet["adapter_directory"],"native-"+source,"Original proof namespace used/"+name)
            checks.equal((packet["method"],packet["source_method"]),(method,source),"Packet result and source labels/"+name)
            observed_packets.append(dict(method=method,source_method=source,adapter=packet["native_adapter_mode"],
                                         request_ids=[r["id"] for r in measured]))
        else:
            checks.equal((packet["adapter_directory"],packet["native_adapter_mode"]),
                         ("native-controls","matched_control"),"Matched control adapter/"+name)
    checks.equal(observed_packets,packets,"Actual batch boundaries, adapters and proof sources match frozen schedule")
    calibration = read(study/"calibration.json")["score_config"]
    prior, home = read(study/"training-prior.json"),read(study/"setup-home.json")
    groups,resolutions = defaultdict(list),defaultdict(dict)
    for row in rows:
        request = requests[row["id"]]
        for key in ("id","bucket_xy_m","values"):
            checks.equal(row[key],request[key],row["id"]+"/exact requested "+key)
        checks.require(row["native_release"] is True and row["action_substituted"] is False,
                       row["id"]+"/unsubstituted native release")
        if row["id"] in by_slot:
            slot=by_slot[row["id"]]
            checks.equal(row["adapter_directory"],"native-"+slot["source_method"],row["id"]+"/source proof directory")
            checks.equal((row["result_method"],row["source_method"]),(slot["method"],slot["source_method"]),
                         row["id"]+"/native result and source labels")
            directory=study/row["adapter_directory"]
            verify_heading(row,request,directory,contract["scene"],home,prior,resolutions[str(directory)],checks)
        else:
            checks.equal(row["adapter_directory"],"native-controls",row["id"]+"/control namespace")
        groups[row["adapter_directory"],row["native_cache_file"]].append(row)
    for (directory,filename),group in groups.items():
        path=contained(study/directory,filename)
        checksum=sha(path)
        with np.load(path,allow_pickle=False) as archive:
            arrays={name:archive[name] for name in archive.files}
        columns=[]
        for row in group:
            col=row["native_cache_row"]
            columns.append(col)
            checks.equal((checksum,str(arrays["record_ids"][col])),
                         (row["native_cache_sha256"],row["id"]),row["id"]+"/authenticated dense column")
            verify_initial_trace(row,arrays,col,checks)
            trace={key:arrays[key][:,col] for key in ("position_m","velocity_m_s","forces_N")}
            for key in ("time_s","dt_s"):
                if key in arrays:trace[key]=arrays[key]
            if "contact_distance_m" in arrays:trace["contact_distance_m"]=arrays["contact_distance_m"][:,col]
            args=contract["scene"],row["bucket_xy_m"],row["board_normal"],calibration,trace
            verify_historical_observation(row,historical.score_trace(*args),checks)
            new=revised.score_trace(*args)
            _same(row["board_first_score"],new,row["id"]+"/independent board-first retention score")
            checks.count+=len(new)
            checks.equal(row["board_first_scalar_audit_passed"],True,row["id"]+"/producer scalar crosscheck recorded")
        checks.equal(len(set(columns)),len(columns),directory+"/unique trace columns/"+filename)
        # Simulator batches can include inert padding worlds; every named
        # record is associated with one saved observation, not hidden drops.
        named={str(identity) for identity in arrays["record_ids"] if str(identity)}
        checks.equal(named,{r["id"] for r in group},directory+"/all recorded worlds accounted/"+filename)
    observed_dirs={row["adapter_directory"] for row in rows}
    for directory in observed_dirs:
        audit_native_options(study/directory,contract["scene"],calibration,a.assets,checks)
        checks.equal({name for namespace,name in groups if namespace==directory},
                     {p.relative_to(study/directory).as_posix() for p in (study/directory).rglob("*.npz")},
                     directory+"/no uncounted native traces")
    all_npz={p.relative_to(study).as_posix() for p in study.rglob("*.npz")}
    checks.equal(all_npz,{str(Path(directory)/filename) for directory,filename in groups},
                 "No native cache outside authenticated execution namespaces")
    for phase in controls:
        ids={row["id"] for row in controls[phase]}
        gate=board_first_controls([r for r in rows if r["id"] in ids])
        checks.equal(gate["status"],"passed",phase+"/direct, table and miss controls pass")
        checks.equal(report["controls"][phase],gate,phase+"/reported control gate")
    checks.require(type(report["gpu_wait_s"]) in (int,float) and math.isfinite(report["gpu_wait_s"]) and report["gpu_wait_s"]>=0,
                   "GPU waiting separated from planning time")
    checks.require(0<report["gpu_lease_s"]<=contract["gpu_cap_s"]<=3600,"Native GPU lease within one-hour cap")
    return rows


def verify_parent_binding(binding, checks, label):
    study,audit_path=Path(binding["study"]),Path(binding["audit"])
    report,receipt=read(study/"report.json"),read(audit_path)
    expected=dict(study=str(study.resolve()),audit=str(audit_path.resolve()),
        report_sha256=sha(study/"report.json"),audit_sha256=sha(audit_path),
        manifest_sha256=sha(study/"evidence-manifest.json"),contract_sha256=sha(study/"contract.json"))
    checks.equal(binding,expected,label+"/exact source binding")
    checks.equal((report["status"],receipt["status"]),("completed","passed"),label+"/completed and independently audited")
    checks.equal(receipt["report_sha256"],expected["report_sha256"],label+"/audit binds completed report")
    checks.equal((receipt["evidence_manifest_sha256"],report["evidence_manifest_sha256"]),
                 (expected["manifest_sha256"],)*2,label+"/complete evidence manifest binding")
    for name,checksum in read(study/"evidence-manifest.json").items():
        checks.equal(sha(contained(study,name)),checksum,label+"/immutable source/"+name)
    return study,read(study/"contract.json")


def verify_layouts(a,contract,archive,checks):
    from .intergrid_proposer_plan import sample_splits
    from .preserved_shot_study import SEED,PROCESSING_SEED
    import random
    layouts,provenance=archive["layouts"],archive["provenance"]
    parent=Path(contract["source_comparison"]["study"])
    prior_layouts=read(parent/"layouts.json")["layouts"]
    if contract["cohort"]=="development":
        declines={s["context_id"] for s in read(parent/"choices.json")["slots"]
                  if s["method"]=="expanded" and s["request"] is None}
        checks.equal(len(declines),9,"Exactly nine original expanded setup declines")
        expected=[l for l in prior_layouts if l["id"] in declines]
        checks.equal(layouts,expected,"Retest exact original targets and IDs; no new seed through renaming")
        checks.equal(provenance,dict(kind="nine_known_setup_declines",source_layouts_sha256=sha(parent/"layouts.json"),
            source_choices_sha256=sha(parent/"choices.json"),excluded_from_fresh_evaluation=True),
            "Known development cases labeled separately")
        checks.equal(contract["development"],None,"Development does not require its own later result")
    else:
        checks.equal(contract["cohort"],"fresh","Known cohort")
        checks.equal(provenance["kind"],"fresh_disjoint_positions","Fresh positions explicitly labeled")
        checks.equal(provenance["geometry"],read(parent/"layouts.json")["provenance"]["geometry"],
                     "Fresh placement geometry matches independently audited scene")
        checks.equal((provenance["seed"],provenance["minimum_separation_m"]),(SEED,.005),"Frozen novel-placement sampler")
        dev,dev_contract=verify_parent_binding(contract["development"],checks,"Completed development prerequisite")
        checks.equal((dev_contract["version"],dev_contract["cohort"],dev_contract["preservation"]),
                     (contract["version"],"development",contract["preservation"]),"Fresh test follows identical audited rule")
        checks.equal(read(Path(contract["development"]["audit"]))["version"],VERSION,"Development audited by this checker")
        exclusions=set(map(tuple,provenance["excluded_coordinates_m"]))
        explicit={tuple(l["bucket_xy_m"]) for l in prior_layouts}
        explicit.update(tuple(l["bucket_xy_m"]) for l in read(dev/"layouts.json")["layouts"])
        explicit.update(tuple(xy) for xy in read(a.output/"all-positions.json"))
        from .approach_comparison_study import lines
        for role in ("train","validation"):
            explicit.update(tuple(r["request"]["bucket_xy_m"]) for r in lines(a.output/(role+"-dataset.jsonl")))
        explicit.update(tuple(r["bucket_xy_m"]) for r in read(a.output/"candidate-sources.json"))
        checks.require(explicit<=exclusions,"Training, source, previous-test and development targets excluded")
        expected_files={str(path.resolve()):sha(path) for path in
            (a.expanded_study/"layouts.json",a.comparison_study/"layouts.json",a.development_study/"layouts.json")}
        checks.equal(provenance["explicit_layout_files"],expected_files,"All explicit prior layout files bound")
        for filename,checksum in provenance["explicit_layout_files"].items():
            path=Path(filename)
            checks.equal(sha(path),checksum,"Previous layout unchanged/"+filename)
            checks.require({tuple(l["bucket_xy_m"]) for l in read(path)["layouts"]}<=exclusions,
                           "Every previous bucket coordinate excluded/"+filename)
        checks.require(bool(provenance["history_files"]),"Historical inventory is nonempty")
        for root,bindings in provenance["history_files"].items():
            for name,checksum in bindings.items():
                path=(Path(__file__).resolve().parents[2]/name if name.startswith("baselines/")
                      else Path(root)/name.removeprefix("output/"))
                checks.equal(sha(path),checksum,"History metadata unchanged/"+name)
                checks.require(history_coordinates(path)<=exclusions,"All inventoried coordinates excluded/"+name)
        sampled=sample_splits(provenance["excluded_coordinates_m"],provenance["geometry"],read(a.parent_study/"grid.json"),
                              seed=SEED,include_train=False)["test"]
        expected=[dict(row,id=f"preserved-fresh-{i:02d}",context_id=f"preserved-fresh-{i:02d}",split="test")
                  for i,row in enumerate(sampled)]
        checks.equal(layouts,expected,"Fresh50 sampled before native outcomes")
        checks.equal(provenance["layouts_sha256"],digest(layouts),"Fresh layout content binding")
        for i,layout in enumerate(layouts):
            checks.require(all(math.dist(layout["bucket_xy_m"],xy)>=.005 for xy in exclusions),layout["id"]+"/history disjoint")
            checks.require(all(math.dist(layout["bucket_xy_m"],other["bucket_xy_m"])>=.005 for other in layouts[:i]),
                           layout["id"]+"/test disjoint")
    rng=random.Random(PROCESSING_SEED)
    order=[l["id"] for l in layouts];rng.shuffle(order)
    first=[RAW_METHODS[i%2] for i in range(len(order))];rng.shuffle(first)
    methods={identity:[method,next(m for m in RAW_METHODS if m!=method)] for identity,method in zip(order,first,strict=True)}
    checks.equal((contract["processing_order"],contract["method_order"]),(order,methods),"Balanced frozen CPU submission order")
    return layouts


def verify_preparation(a):
    from .preserved_shot_study import frozen,VERSION as STUDY_VERSION,SEED,PROCESSING_SEED,NATIVE_SCHEDULE_VERSION
    from .preserved_shot_plan import description
    from .board_first_score import VERSION as SCORE_VERSION
    from .approach_comparison_plan import description as approach_description
    checks,study=Checks(),a.output
    contract=frozen(a)
    checks.equal(contract["version"],STUDY_VERSION,"Preserved-shot study version")
    checks.equal(contract["cohort"],a.cohort,"Explicit selected cohort")
    parent,original=verify_parent_binding(contract["source_comparison"],checks,"Prior comparison")
    checks.equal(parent,a.comparison_study.resolve(),"Explicit prior-comparison argument")
    checks.equal(contract["preservation"],description(),"Exact two-lane admission-only fallback")
    checks.equal(contract["approach_planner"],approach_description(),"Unchanged expanded proposal and scoring rule")
    checks.equal((contract["methods"],contract["raw_methods"]),(list(METHODS),list(RAW_METHODS)),"Distinct result and proof-source methods")
    checks.equal((contract["seed"],contract["processing_seed"],contract["planning_budget_s"],contract["workers"],contract["physics_substeps"],
                  contract["training_performed"],contract["scoring_version"],contract["matched_controls"]),
                 (SEED,PROCESSING_SEED,30.,2,64,False,SCORE_VERSION,18),"Frozen CPU, native64 and unchanged scoring protocol")
    count,repeats=(9,3) if contract["cohort"]=="development" else (50,1)
    checks.equal((contract["test_positions"],contract["attempts_per_method_per_position"],contract["max_native_releases"]),
                 (count,repeats,2*count*repeats+18),"Separate development repeats and fresh one-drop test")
    native=contract["native_execution"]
    checks.equal((native["version"],native["batch_size"],native["method_order"],native["source_order"],native["adapter_order"]),
                 (NATIVE_SCHEDULE_VERSION,32,list(METHODS),list(RAW_METHODS),["reference","direct_first"]),"Exact homogeneous native execution lanes")
    for key in ("identity","scene","flight_ranking","current_planner_binding","approach_planner_version","approach_planner",
                "parent","neural_parent","expanded_parent","library_path","library_summary_sha256","library_manifest_sha256",
                "reference_path_s","early_score_m","gpu_cap_s"):
        checks.equal(contract[key],original[key],"Unchanged audited source configuration/"+key)
    for name,checksum in original["frozen_inputs"].items():
        if name=="layouts.json":continue
        checks.equal(sha(study/name),checksum,"Exact original frozen input/"+name)
    checks.equal(set(contract["frozen_inputs"]),set(original["frozen_inputs"]),"Same frozen inputs with newly declared layouts only")
    layouts=verify_layouts(a,contract,read(study/"layouts.json"),checks)
    checks.equal(len(layouts),count,"Exact cohort size")
    manifest=inventory(study)
    started=read(study/"preparation-started.json")
    checks.equal(started["contract_sha256"],sha(study/"contract.json"),"CPU preparation starts after contract freeze")
    checks.equal(started["layouts_sha256"],sha(study/"layouts.json"),"CPU preparation binds exact positions")
    checks.equal(started["frozen_inputs"],contract["frozen_inputs"],"CPU preparation binds exact models/library")
    plans=verify_plans(study,layouts,dict(contract,methods=list(RAW_METHODS)),a.assets,read(study/"training-prior.json"),read(study/"setup-home.json"),
                       manifest,checks,variants=RAW_METHODS)
    selections=read(study/"selection.json")
    checks.equal({p.stem for p in (study/"selection").glob("*.json")},{l["id"] for l in layouts},"Exact per-target selection receipt inventory")
    for row in selections:
        checks.equal(row,read(study/"selection"/(row["context_id"]+".json")),row["context_id"]+"/complete selection receipt")
    choices=read(study/"choices.json")
    verify_slots(choices["slots"],plans,selections,layouts,repeats,checks)
    verify_admission_proofs(study,choices["slots"],checks)
    checks.equal(read(study/"native-schedule.json"),expected_schedule(contract,choices["slots"]),
                 "Independent native schedule with exact fallback proof sources")
    required=set(contract["frozen_inputs"])|{"contract.json","plans.json","selection.json","preparation-started.json",
                                                "admission-proof-bindings.json","native-schedule.json"}
    checks.require(required<=set(choices["bindings"]),"Complete model, data, command and proof freeze")
    for name,checksum in choices["bindings"].items():
        checks.equal(sha(contained(study,name)),checksum,"Frozen preparation evidence/"+name)
    report=read(study/"report.json")
    checks.equal((report["cohort"],report["training"]["performed"],report["live_default_changed"]),
                 (contract["cohort"],False,False),"No fitting or live runtime promotion")
    return checks,contract,report,manifest,layouts,plans,choices,selections


def preflight(a):
    from .preserved_shot_study import VERSION as STUDY_VERSION
    checks,contract,report,manifest,layouts,plans,choices,selections=verify_preparation(a)
    checks.equal((report["status"],report["phase"]),("running","prepared"),"CPU preparation complete")
    for name in ("native-started.json","rows.jsonl","timing.jsonl","evidence-manifest.json"):
        checks.require(not(a.output/name).exists(),"No native observation before audited freeze/"+name)
    checks.require(not list(a.output.rglob("*.npz")),"No uncounted preflight drops")
    checks.equal(set(manifest)-{"choices.json"},set(choices["bindings"]),"Every preparation file frozen")
    return dict(version=VERSION,study_version=STUDY_VERSION,cohort=contract["cohort"],stage="preflight",status="passed",
        checks_count=checks.count,contract_sha256=sha(a.output/"contract.json"),choices_sha256=sha(a.output/"choices.json"),
        preparation_manifest_sha256=digest(manifest),frozen_before_test=True,
        planned_releases=sum(s["request"] is not None for s in choices["slots"]),
        checkpoint_sha256={m:sha(a.output/(m+"-checkpoint.json")) for m in RAW_METHODS})


def require_preflight(a):
    path=a.preflight_output
    if path is None or not path.is_file() or path.resolve().is_relative_to(a.output.resolve()):
        raise ValueError("Require a separate preflight receipt outside the immutable study")
    receipt=read(path)
    choices=read(a.output/"choices.json")
    bindings=dict(choices["bindings"],**{"choices.json":sha(a.output/"choices.json")})
    if (receipt.get("version")!=VERSION or receipt.get("stage")!="preflight" or receipt.get("status")!="passed"
            or receipt.get("contract_sha256")!=sha(a.output/"contract.json")
            or receipt.get("choices_sha256")!=sha(a.output/"choices.json")
            or receipt.get("preparation_manifest_sha256")!=digest(bindings)
            or receipt.get("checkpoint_sha256")!={m:sha(a.output/(m+"-checkpoint.json")) for m in RAW_METHODS}):
        raise ValueError("Preflight receipt does not bind this exact frozen preparation")
    for name,checksum in bindings.items():
        if sha(contained(a.output,name))!=checksum:
            raise ValueError("Preflight evidence changed: "+name)
    return receipt


def independent_summary(layouts,slots,rows,plans,selections):
    """Count captures without using producer totals or dropping declined setups."""
    observed={row["id"]:row for row in rows if row["adapter_directory"]!="native-controls"}
    expected={slot["id"] for slot in slots if slot["request"] is not None}
    if set(observed)!=expected or len(observed)!=sum(row["adapter_directory"]!="native-controls" for row in rows):
        raise ValueError("Independent accounting requires exactly the frozen releases")
    plan_by={(p["site"]["layout"]["id"],p["variant"]):p["site"] for p in plans}
    selection_by={row["context_id"]:row for row in selections}
    slot_by=defaultdict(list)
    for slot in slots:slot_by[slot["context_id"],slot["method"]].append(slot)
    positions=[]
    total={m:dict(positions=len(layouts),attempt_slots=0,captures=0,releases=0,setup_rejections=0,
                  invalid_physics=0,flight_misses=0) for m in METHODS}
    paired=dict(both=0,current_only=0,protected_only=0,neither=0)
    for layout in layouts:
        selection=selection_by[layout["id"]]
        position=dict(id=layout["id"],bucket_xy_m=layout["bucket_xy_m"],methods={},
                      protected_source=selection["selected_source"],fallback_used=selection["fallback_used"])
        for method in METHODS:
            own=slot_by[layout["id"],method]
            captures=[]
            for slot in own:
                total[method]["attempt_slots"]+=1
                if slot["request"] is None:
                    total[method]["setup_rejections"]+=1
                    captures.append(False)
                    continue
                total[method]["releases"]+=1
                score=observed[slot["id"]]["board_first_score"]
                success=bool(score["route_success"])
                captures.append(success)
                if success:total[method]["captures"]+=1
                elif score["invalid_physics"]:total[method]["invalid_physics"]+=1
                else:total[method]["flight_misses"]+=1
            position["methods"][method]=dict(captures=sum(captures),attempts=len(own),repeat_captures=captures,
                releases=sum(slot["request"] is not None for slot in own),all_repeats_captured=all(captures))
        left,right=(position["methods"][m]["repeat_captures"][0] for m in METHODS)
        paired["both" if left and right else "current_only" if left else "protected_only" if right else "neither"]+=1
        positions.append(position)
    for method in METHODS:
        times=[plan_by[l["id"],"current"]["request_wall_s"] if method=="current"
               else selection_by[l["id"]]["timing"]["pair_wall_s"] for l in layouts]
        ordered=sorted(times)
        counts=total[method]
        counts.update(capture_rate=counts["captures"]/counts["attempt_slots"],mean_request_s=sum(times)/len(times),
            median_request_s=median(times),p90_request_s=ordered[math.ceil(.9*len(times))-1],max_request_s=ordered[-1])
    return dict(methods=total,positions=positions,paired_first_drop=paired,
        protected_fallback_positions=sum(row["fallback_used"] for row in selections),
        paired_timing=dict(mean_pair_wall_s=sum(row["timing"]["pair_wall_s"] for row in selections)/len(selections),
            mean_reserved_worker_window_s=sum(row["timing"]["reserved_worker_window_s"] for row in selections)/len(selections),
            measured_cpu_consumption=False,cpu_lanes=2,decision_budget_per_lane_s=30.))


def audit(a):
    checks,contract,report,actual,layouts,plans,choices,selections=verify_preparation(a)
    study=a.output
    checks.equal((report["status"],report["phase"]),("completed","completed"),"Completed native cohort")
    manifest=read(study/"evidence-manifest.json")
    checks.equal(report["evidence_manifest_sha256"],sha(study/"evidence-manifest.json"),"Final report binds evidence inventory")
    checks.equal(manifest,actual,"All saved evidence authenticated")
    rows=verify_native(a,checks,contract,report,choices)
    require_preflight(a)
    checks.equal(read(study/"native-started.json")["preflight_sha256"],sha(a.preflight_output),
                 "Native execution followed exact passed preflight")
    comparison=independent_summary(layouts,choices["slots"],rows,plans,selections)
    checks.equal(report["comparison"],comparison,"Independent complete per-position and aggregate accounting")
    checks.equal(report["progress"]["completed_releases"],sum(s["request"] is not None for s in choices["slots"]),
                 "Reported completed evaluation drops exclude controls")
    return dict(version=VERSION,study_version=contract["version"],cohort=contract["cohort"],stage="audit",status="passed",
        checks_count=checks.count,frozen_before_test=True,report_sha256=sha(study/"report.json"),
        evidence_manifest_sha256=sha(study/"evidence-manifest.json"),contract_sha256=sha(study/"contract.json"),
        choices_sha256=sha(study/"choices.json"),selection_sha256=sha(study/"selection.json"),
        checkpoint_sha256={m:sha(study/(m+"-checkpoint.json")) for m in RAW_METHODS},
        preflight_sha256=sha(a.preflight_output),native_started_sha256=sha(study/"native-started.json"),comparison=comparison,
        scope="Independent raw planning/proof audit, exact fallback requests, timing and fresh-position isolation; every dense native trace rescored; no simulator rerun")
