"""Read-only audit of corrected labels, frozen choices and native captures."""
from __future__ import annotations

from collections import defaultdict
import copy
import math
import random
from pathlib import Path

import numpy as np

from .benchmark_verify import Checks, contained, digest, read, sha

VERSION = "sircus-approach-comparison-independent-audit-v1"


def verify_initial_trace(row, arrays, column, checks):
    checks.equal(arrays["position_m"][0,column].tolist(),row["native_initial_ball_center_m"],
                 row["id"]+"/exact observed time-zero ball association")
    checks.numeric(arrays["position_m"][0,column],row["release_ball_center_m"],
                   row["id"]+"/time-zero ball matches commanded release",tolerance=1e-6)
    # The frozen drop behavior releases from rest. No initial velocity is
    # supplied by these requests; audit the recorded vector rather than assume.
    checks.equal(arrays["velocity_m_s"][0,column].tolist(),[0.,0.,0.],
                 row["id"]+"/recorded release from rest")


def verify_plans(study, layouts, spec, assets, prior, home, manifest, checks, *, artifact_root=None, variants=("current", "expanded")):
    """Rebuild finite menus, checked prefixes, full proof bindings and choices."""
    from .clearance_expansion_plan import _empty, branch_proposals, needs_expansion
    from .intergrid_candidate_replay_plan import command_key
    from .intergrid_flight_ranker import FlightRanker
    from .approach_comparison_plan import ApproachFlightRanker, configure
    from .expanded_selector_plan import VERSION as MENU_VERSION
    from .direct_path_first_audit import audit_preparation, audit_branches, verify_workers
    from .fast_planning_audit import verify_deadline, verify_early_rule
    from .stalled_reference_audit import verify_switch
    from .intergrid_library_plan import libraries
    from .neural_shot_model import order
    from .neural_shot_plan import CHECK_BATCH, CHECK_LIMIT, _choice, menu

    artifact_root = study if artifact_root is None else artifact_root
    wrappers = read(artifact_root / "plans.json")
    expected = [(layout, method) for layout in layouts for method in variants]
    checks.equal([(item["site"]["layout"], item["variant"]) for item in wrappers], expected,
                 "Exact ordered preparation for every placement and both variants")
    checks.require(all(set(item) == {"variant", "site"} for item in wrappers), "Exact outer wrapper schema")
    checks.require(all(item["site"]["method"] == "warm_early" for item in wrappers),
                   "Original warm_early inner identity retained without rewriting receipts")
    raw = read(study / "shot-library-rows.json")
    libs = libraries(raw["original"], raw["additions"])
    source_checkpoint = read(study/"route-models.json")
    checkpoints = {m:read(study/(m+"-checkpoint.json")) for m in variants}
    libraries_by_method = dict(current=libs, expanded=dict(expanded_menu_version=MENU_VERSION,
        baseline_libraries=libs, expanded_sources=read(study/"candidate-sources.json")))
    from .board_first_selector_plan import VERSION as CONTACT_VERSION
    libraries_by_method["expanded"], _ = configure(libraries_by_method["expanded"],
        dict(contact_ranking_version=CONTACT_VERSION), checkpoints["expanded"])
    rankers = {m:cls(spec["scene"],read(study/"calibration.json")["score_config"],assets,spec["flight_ranking"]["rule"])
               for m,cls in (("current",FlightRanker),("expanded",ApproachFlightRanker))}
    for ranker in rankers.values():
        implementation, predictions = ranker.predict, {}
        def independent_prediction(request, implementation=implementation, predictions=predictions):
            key = command_key(request)
            if key not in predictions:predictions[key] = implementation(request)
            return predictions[key]
        ranker.predict = independent_prediction
    for method in variants:
        checks.equal({path.stem for path in (artifact_root / "preparation" / method).glob("*.json")},
                     {layout["id"] for layout in layouts}, method + "/exact preparation file inventory")
    for wrapper in wrappers:
        site = wrapper["site"]
        layout, method, plan = site["layout"], wrapper["variant"], site["plan"]
        identity = layout["id"] + "/" + method
        directory = artifact_root / ("native-" + method)
        checks.equal(site, read(artifact_root / "preparation" / method / (layout["id"] + ".json")),
                     identity + "/assembled plan retains exact per-job receipt")
        prefix = artifact_root.relative_to(study).as_posix()
        own_manifest = manifest if prefix == "." else {
            name[len(prefix) + 1:]: value for name, value in manifest.items() if name.startswith(prefix + "/")}
        snapshots = verify_deadline(site, directory, own_manifest, checks, spec["planning_budget_s"])
        ranker = rankers[method]
        proposals = menu(layout, libraries_by_method[method], source_checkpoint)
        ordered = order(checkpoints[method], proposals)
        verify_switch(site=site, snapshots=snapshots, directory=directory, manifest=own_manifest,
                      checks=checks, spec=spec, variant="hybrid", expected_menu=proposals, expected_order=ordered)
        checks.equal(set(plan), {"base", "fallback", "neural", "branches"}, identity + "/exact preparation stages")
        if not snapshots:
            verify_early_rule(site, snapshots, checks, spec["early_score_m"])
            continue
        resolutions = {}
        for stage in ("base", "fallback", "neural"):
            audit_preparation(plan[stage], layout, "test", directory, prior, home, resolutions, checks)
        # Check each accepted publication, not a command chosen after observing
        # a simulated outcome. Full endpoint proofs need only be checked once.
        for snapshot in snapshots:
            p, generated = snapshot["plan"], snapshot["generated"]
            for stage in p:
                # Every earlier publication is an immutable prefix of the final
                # accepted complete evidence; late candidate files do not enter it.
                for key in ("proposals", "admitted", "records"):
                    checks.equal(plan[stage][key][:len(p[stage][key])], p[stage][key],
                                 identity + "/immutable complete stage prefix " + stage + "/" + key)
            completed = [row["id"] for stage in ("base", "fallback", "neural", "branches")
                         for row in p[stage]["proposals"]]
            checks.equal(snapshot["completed_proposal_ids"], completed, identity + "/only complete candidates count")
            checks.require(set(generated) <= {"menu", "neural", "branches"}, identity + "/learned menu stages only")
            checks.equal(generated["menu"], proposals, identity + "/frozen pre-IK proposal menu")
            checks.equal(snapshot["ranked_proposal_ids"], [row["id"] for row in ordered],
                         identity + "/frozen checkpoint orders proposals before IK")
            checked = p["neural"]["proposals"]
            checks.require(len(checked) <= min(CHECK_LIMIT, len(ordered)), identity + "/bounded learned prefix")
            checks.equal(checked, ordered[:len(checked)], identity + "/no skipped candidate or hidden full-menu search")
            generated_neural = generated.get("neural", [])
            checks.equal(generated_neural, ordered[:len(generated_neural)], identity + "/generated checked prefix")
            checks.require(len(checked) <= len(generated_neural) <= min(CHECK_LIMIT, len(ordered))
                           and len(generated_neural) - len(checked) < CHECK_BATCH,
                           identity + "/only current bounded candidate batch may be incomplete")
            checks.require(not any(p[stage][key] for stage in ("base", "fallback") for key in p[stage]),
                           identity + "/learned planner does not borrow baseline admissions")
            for stop in range(CHECK_BATCH, len(checked), CHECK_BATCH):
                ids = {row["id"] for row in checked[:stop]}
                prefix = [row for row in p["neural"]["admitted"] if row["proposal_id"] in ids]
                checks.require(needs_expansion(_choice(layout, prefix, ranker, stop, snapshot["ranked_proposal_ids"])),
                               identity + "/more checks only after nonpositive predicted clearance")
            if "branches" in generated:
                checks.equal(len(checked), min(CHECK_LIMIT, len(ordered)),
                             identity + "/joint alternatives follow exhausted neural budget")
                initial = _choice(layout, p["neural"]["admitted"], ranker, len(checked), snapshot["ranked_proposal_ids"])
                checks.require(needs_expansion(initial) and bool(p["neural"]["admitted"]),
                               identity + "/alternative joints require an admitted poor-clearance shot")
                branch_plan = dict(base=p["neural"], fallback=_empty(), expansion=_empty(), branches=_empty())
                branches = branch_proposals(layout, branch_plan, ranker)
                checks.equal(generated["branches"], branches, identity + "/frozen alternative-arm menu")
                checks.equal(p["branches"]["proposals"], branches[:len(p["branches"]["proposals"])],
                             identity + "/only prefix of alternative-arm checks")
            else:
                checks.require(not any(p["branches"].values()), identity + "/no undeclared alternative arm checks")
            admitted = list({command_key(row): row for stage in ("neural", "branches")
                             for row in p[stage]["admitted"]}.values())
            choice = _choice(layout, admitted, ranker, len(checked), snapshot["ranked_proposal_ids"])
            choice = copy.deepcopy(choice)
            identifier = layout["id"] + "-" + site["method"]
            choice.update(id=identifier, method=site["method"])
            if choice["request"] is not None:
                choice["request"].update(id=identifier, method=site["method"])
            choice["predicted_flight_risk"] = needs_expansion(choice)
            checks.equal(snapshot["choice"], choice, identity + "/pre-deadline command independently reconstructed")
            if snapshot["active_stage"] == "finished":
                limit = min(CHECK_LIMIT, len(ordered))
                checks.require(len(checked) == limit or (len(checked) > 0 and len(checked) % CHECK_BATCH == 0),
                               identity + "/normal stop follows completed batch")
                initial = _choice(layout, p["neural"]["admitted"], ranker, len(checked), snapshot["ranked_proposal_ids"])
                checks.require(len(checked) == limit or not needs_expansion(initial),
                               identity + "/normal stop needs positive clearance or exhausted candidates")
                if needs_expansion(initial) and p["neural"]["admitted"]:
                    checks.require("branches" in generated, identity + "/required alternative joints considered")
                    checks.equal(p["branches"]["proposals"], generated["branches"],
                                 identity + "/normal completion finishes joint alternatives")
        verify_early_rule(site, snapshots, checks, spec["early_score_m"])
        audit_branches(plan["branches"], layout, {row["id"]: row for row in plan["neural"]["admitted"]},
                           directory, prior, home, resolutions, checks)
    worker_spec = spec
    if artifact_root != study:
        order = [layout["id"] for layout in layouts]
        random.Random(spec["processing_seed"]).shuffle(order)
        worker_spec = dict(spec, processing_order=order)
    verify_workers(wrappers, layouts, worker_spec, checks, variants=variants)
    return wrappers


def inventory(study):
    return {str(p.relative_to(study)):sha(p) for p in sorted(study.rglob("*"))
            if p.is_file() and p.name not in {"report.json", "evidence-manifest.json"}}


def history_coordinates(path):
    """Re-read every inventoried coordinate source, including packet-only requests."""
    from .table_training_contract import _coordinates
    import json
    points = set()
    with path.open() as stream:
        payloads = (json.loads(line) for line in stream if line.strip()) if path.suffix == ".jsonl" else [json.load(stream)]
        for payload in payloads:
            _coordinates(payload, points, coordinate_list="exclusion" in path.name)
            if isinstance(payload, dict):
                for splits in (payload, payload.get("splits"), payload.get("future_splits")):
                    if isinstance(splits, dict):
                        for role in ("train", "validation", "test"):
                            _coordinates(splits.get(role), points, coordinate_list=True)
    return points


def verify_preparation(a):
    from .expanded_selector_model import VERSION as label_version
    from .expanded_shot_library import verify_library
    from .approach_comparison_study import (METHODS, POSITIONS, VERSION as STUDY_VERSION,
        frozen, lines, source_inputs, current_planner_binding, expanded_parent_binding,
        SEED, PROCESSING_SEED, schedule, native_schedule, NATIVE_BATCH_SIZE, NATIVE_ADAPTERS, NATIVE_SCHEDULE_VERSION)
    from .intergrid_proposer_plan import sample_splits
    from .board_first_score import VERSION as score_version
    from .approach_comparison_plan import description
    checks, study = Checks(), a.output
    contract = frozen(a)
    source_inputs(a)
    checks.equal(contract["expanded_parent"], expanded_parent_binding(a), "Unchanged independently audited expanded parent")
    checks.equal(contract["training_performed"], False, "No fitting in this study")
    checks.equal(contract["approach_planner"], description(), "Exact bounded proposal and forecast protocol")
    native = contract["native_execution"]
    checks.equal((native["version"],native["batch_size"],native["method_order"],native["adapter_order"]),
                 (NATIVE_SCHEDULE_VERSION,NATIVE_BATCH_SIZE,list(METHODS),list(NATIVE_ADAPTERS)), "Explicit independent-world native batching")
    checks.equal(contract["scoring_version"], score_version, "Unchanged native version 4 scoring")
    checks.equal((contract["seed"], contract["processing_seed"], contract["planning_budget_s"],
                  contract["workers"], contract["max_native_releases"], contract["matched_controls"]),
                 (SEED, PROCESSING_SEED, 30., 2, 118, 18), "Exact frozen experimental limits")
    checks.equal(contract["current_planner_binding"],current_planner_binding(a),"Exact current live planner input binding")
    report = read(study / "report.json")
    manifest = inventory(study)
    checks.equal(contract["version"], STUDY_VERSION, "Comparison version")
    checks.equal((contract["physics_substeps"], contract["attempts_per_method_per_position"], contract["methods"], contract["test_positions"]),
                 (64, 1, list(METHODS), POSITIONS), "Fixed test protocol")
    library = verify_library(a.library)
    checks.equal(sha(a.library/"library-summary.json"),contract["library_summary_sha256"],"Audited library summary binding")
    checks.equal(sha(a.library/"library-manifest.json"),contract["library_manifest_sha256"],"Audited library manifest binding")
    for name in ("train-dataset.jsonl", "validation-dataset.jsonl", "candidate-sources.json", "library-summary.json", "library-manifest.json", "all-positions.json"):
        checks.equal(sha(study/name),sha(a.library/name),"Exact expanded library input/"+name)
    for name, expected in contract["expanded_parent"]["inputs"].items():
        if name != "layouts.json":
            checks.equal(sha(study/name), expected, "Exact copied parent input/"+name)
    checkpoints = {m:read(study/(m+"-checkpoint.json")) for m in METHODS}
    checks.equal(checkpoints["current"],read(a.neural_study/"neural-checkpoint.json"),"Current live original selector unchanged")
    from .neural_shot_model import _validate_checkpoint, _digest
    trained = checkpoints["expanded"]
    _validate_checkpoint(trained)
    checks.equal(trained["label_version"], label_version, "Explicit corrected training labels")
    checks.equal(trained["fingerprint"], _digest({k:v for k,v in trained.items() if k != "fingerprint"}), "Frozen weight fingerprint")
    checks.equal(trained["training_sha256"], _digest(lines(study/"train-dataset.jsonl")), "TRAIN provenance")
    checks.equal(trained["validation_sha256"], _digest(lines(study/"validation-dataset.jsonl")), "Validation provenance")
    fit = read(study/"fit.json")["validation"]
    checks.equal(fit["selected_epoch"], min(fit["history"],key=lambda r:(r["loss"],r["epoch"]))["epoch"], "Validation-only epoch selection")
    checks.equal(trained["epoch"], fit["selected_epoch"], "Selected checkpoint epoch")
    archive = read(study / "layouts.json")
    layouts, provenance = archive["layouts"], archive["provenance"]
    sampled = sample_splits(provenance["excluded_coordinates_m"], provenance["geometry"], read(a.parent_study/"grid.json"),
                            seed=contract["seed"], include_train=False)["test"]
    reconstructed = [dict(row, id=f"approach-comparison-test-{i:02d}", context_id=f"approach-comparison-test-{i:02d}", split="test") for i,row in enumerate(sampled)]
    checks.equal(layouts, reconstructed, "Fresh positions fixed independently of outcomes")
    checks.equal((contract["processing_order"], contract["method_order"]), schedule(layouts), "Balanced randomized frozen order")
    previous = Path(provenance["previous_comparison"])/"layouts.json"
    checks.equal(sha(previous),provenance["previous_layouts_sha256"],"Previous fifty test sites remain bound")
    explicit = {tuple(r["bucket_xy_m"]) for r in read(previous)["layouts"]}
    explicit.update(tuple(xy) for xy in read(study/"all-positions.json"))
    for split in ("train", "validation"):
        explicit.update(tuple(r["request"]["bucket_xy_m"]) for r in lines(study/(split+"-dataset.jsonl")))
    explicit.update(tuple(r["bucket_xy_m"]) for r in read(study/"candidate-sources.json"))
    checks.require(explicit <= set(map(tuple,provenance["excluded_coordinates_m"])),"Every dataset/source/prior-test location explicitly excluded")
    for i, layout in enumerate(layouts):
        checks.require(all(math.dist(layout["bucket_xy_m"], xy) >= .005 for xy in provenance["excluded_coordinates_m"]), layout["id"]+"/history disjoint")
        checks.require(all(math.dist(layout["bucket_xy_m"], other["bucket_xy_m"]) >= .005 for other in layouts[:i]), layout["id"]+"/test disjoint")
    for root, bindings in provenance["history_files"].items():
        for name, expected in bindings.items():
            path = (Path(__file__).resolve().parents[2]/name if name.startswith("baselines/")
                    else Path(root)/name.removeprefix("output/"))
            checks.equal(sha(path), expected, "Historical coordinates/"+name)
            checks.require(history_coordinates(path) <= set(map(tuple, provenance["excluded_coordinates_m"])),
                           "All inventoried bucket coordinates excluded/"+name)
    preparation_started = read(study/"preparation-started.json")
    checks.equal(preparation_started["contract_sha256"],sha(study/"contract.json"),"Preparation started after frozen contract")
    checks.equal(preparation_started["layouts_sha256"],sha(study/"layouts.json"),"Preparation binds frozen positions")
    checks.equal(preparation_started["frozen_inputs"],contract["frozen_inputs"],"Preparation binds exact models/library")
    plans = verify_plans(study, layouts, contract, a.assets, read(study/"training-prior.json"), read(study/"setup-home.json"), manifest, checks)
    choices = read(study/"choices.json")
    from .planning_budget_plan import _wrap_choice
    checks.equal(choices["slots"], [_wrap_choice(plan["site"]["choice"], plan["variant"]) for plan in plans],
                 "Frozen choices exactly wrap pre-deadline selected commands")
    packets = read(study/"native-schedule.json")
    checks.equal(packets, native_schedule(contract,choices["slots"]),
                 "Native packets frozen from commands before release, without outcomes")
    from .stalled_reference_plan import request_mode
    selected = {s["id"]:s for s in choices["slots"] if s["request"] is not None}
    released_ids = [identity for packet in packets for identity in packet["request_ids"]]
    checks.equal((len(released_ids),set(released_ids)),(len(selected),set(selected)),
                 "Exactly one scheduled native release for every admitted slot")
    indexed = {(s["context_id"],s["method"]):s for s in choices["slots"]}
    for packet in packets:
        checks.require(0 < len(packet["request_ids"]) <= NATIVE_BATCH_SIZE,
                       "Bounded nonempty native packet")
        checks.require(all(selected[identity]["method"] == packet["method"]
                           and request_mode(selected[identity]["request"]) == packet["adapter"]
                           for identity in packet["request_ids"]), "Every packet uses one exact method and adapter")
    for method in METHODS:
        for mode in NATIVE_ADAPTERS:
            expected = []
            for identity in contract["processing_order"]:
                slot = indexed[identity,method]
                if slot["request"] is not None and request_mode(slot["request"]) == mode:
                    expected.append(slot["id"])
            actual = [identity for packet in packets if (packet["method"],packet["adapter"]) == (method,mode)
                      for identity in packet["request_ids"]]
            checks.equal(actual,expected, "Frozen position order retained within native lane/"+method+"/"+mode)
    required = set(contract["frozen_inputs"]) | {"contract.json", "plans.json", "preparation-started.json", "admission-proof-bindings.json", "native-schedule.json"}
    checks.require(required <= set(choices["bindings"]), "All models, data, inputs and decisions frozen")
    for name, checksum in choices["bindings"].items():
        checks.equal(sha(contained(study,name)), checksum, "Frozen preparation input/"+name)
    checks.equal({name:value for name,value in manifest.items() if name != "choices.json" and name in choices["bindings"]},
                 choices["bindings"], "Choice receipt binds exact preparation evidence")
    for plan in plans:
        method = plan["variant"]
        checks.require(plan["site"]["status"] in {"completed", "deadline"}, "No planner process error used as a decline")
    return checks, contract, report, manifest, layouts, plans, choices


def preflight(a):
    from .approach_comparison_study import VERSION as STUDY_VERSION
    checks, contract, report, manifest, layouts, plans, choices = verify_preparation(a)
    checks.equal((report["status"],report["phase"]), ("running","prepared"), "CPU preparation complete")
    for name in ("native-started.json", "rows.jsonl", "timing.jsonl", "evidence-manifest.json"):
        checks.require(not (a.output/name).exists(), "No native observation before freeze/"+name)
    checks.require(not list(a.output.rglob("*.npz")), "No uncounted preflight drops")
    checks.equal(set(manifest)-{"choices.json"}, set(choices["bindings"]), "Every preparation artifact frozen before native")
    return dict(version=VERSION, study_version=STUDY_VERSION, stage="preflight", status="passed", checks_count=checks.count,
        contract_sha256=sha(a.output/"contract.json"), choices_sha256=sha(a.output/"choices.json"),
        preparation_manifest_sha256=digest(manifest), frozen_before_test=True,
        planned_releases=sum(slot["request"] is not None for slot in choices["slots"]),
        checkpoint_sha256={m:sha(a.output/(m+"-checkpoint.json")) for m in contract["methods"]})


def require_preflight(a):
    path = a.preflight_output
    if path is None or not path.is_file() or path.resolve().is_relative_to(a.output.resolve()):
        raise ValueError("Require an independent preflight receipt outside immutable study evidence")
    receipt = read(path)
    contract = read(a.output/"contract.json")
    choices = read(a.output/"choices.json")
    expected = dict(choices["bindings"], **{"choices.json":sha(a.output/"choices.json")})
    if (receipt.get("version") != VERSION or receipt.get("status") != "passed" or receipt.get("stage") != "preflight"
            or receipt.get("contract_sha256") != sha(a.output/"contract.json")
            or receipt.get("choices_sha256") != sha(a.output/"choices.json")
            or receipt.get("preparation_manifest_sha256") != digest(expected)
            or receipt.get("checkpoint_sha256") != {m:sha(a.output/(m+"-checkpoint.json")) for m in contract["methods"]}):
        raise ValueError("Passed preflight does not match frozen commands and checkpoints")
    for name, expected_sha in expected.items():
        if sha(contained(a.output,name)) != expected_sha:
            raise ValueError("Preflight evidence changed: "+name)
    return receipt


def audit(a):
    from . import board_first_score_oracle as revised
    from . import table_assisted_score_oracle as historical
    from .approach_comparison_study import METHODS, board_first_controls, lines, summarize, planned_requests, native_schedule
    from .direct_path_first_audit import verify_heading
    from .cell1050_rebound_study import control_requests
    from .policy64_verify import audit_native_options
    from .table_assisted_env import _same
    checks, contract, report, actual, layouts, plans, choices = verify_preparation(a)
    study = a.output
    checks.equal((report["status"], report["phase"]), ("completed", "completed"), "Completed comparison")
    manifest = read(study/"evidence-manifest.json")
    checks.equal(report["evidence_manifest_sha256"],sha(study/"evidence-manifest.json"),"Manifest binding")
    checks.equal(manifest, actual, "Complete authenticated evidence inventory")
    choices, started = read(study/"choices.json"), read(study/"native-started.json")
    slots = choices["slots"]
    from .planning_budget_plan import _wrap_choice
    checks.equal(slots, [_wrap_choice(plan["site"]["choice"],plan["variant"]) for plan in plans], "Frozen choice exactly matches prepared command")
    checks.equal([(s["context_id"],s["method"]) for s in slots], [(l["id"],m) for l in layouts for m in METHODS], "One slot per policy and position")
    for name, expected in started["bindings"].items():
        checks.equal(manifest[name], expected, "Inputs frozen before TEST/"+name)
    required = {"current-checkpoint.json", "expanded-checkpoint.json", "train-dataset.jsonl", "validation-dataset.jsonl", "choices.json", "plans.json", "preparation-started.json"}
    checks.require(required <= started["bindings"].keys(), "Weights, labels and choices frozen before test")
    planned = planned_requests(contract, slots)
    packets = native_schedule(contract,slots)
    expected_test_order = [identity for packet in packets for identity in packet["request_ids"]]
    checks.equal(started["native_schedule_sha256"], sha(study/"native-schedule.json"), "Native start binds batched schedule")
    checks.equal(started["requests_sha256"], digest(planned), "One-drop command ledger frozen")
    rows = lines(study/"rows.jsonl")
    checks.equal([r["id"] for r in rows if r["adapter_directory"] != "native-controls"],
                 expected_test_order, "Exact frozen batched native order")
    checks.require(len(rows) <= 118, "At most 100 evaluation drops and 18 controls")
    calibration = read(study/"calibration.json")["score_config"]
    prior, home = read(study/"training-prior.json"), read(study/"setup-home.json")
    requests = {r["id"]:r for group in planned.values() for r in group}
    sources = read(a.parent_study/"control-replays.json")
    controls = {phase:control_requests(sources,phase) for phase in ("controls_pre","controls_post")}
    requests.update({r["id"]:r for group in controls.values() for r in group})
    checks.equal(len(rows), len(requests), "All and only expected native drops")
    checks.equal(len({r["id"] for r in rows}), len(rows), "Unique native observation IDs")
    checks.equal([r["id"] for r in rows[:9]], [r["id"] for r in controls["controls_pre"]], "Controls precede test drops")
    checks.equal([r["id"] for r in rows[-9:]], [r["id"] for r in controls["controls_post"]], "Controls follow test drops")
    for method in METHODS:
        checks.equal([r["id"] for r in rows if r["adapter_directory"]=="native-"+method],
                     [r["id"] for r in planned[method]], method+"/exact frozen release order")
    groups, resolutions = defaultdict(list), defaultdict(dict)
    packet_names = {row["request_packet"] for row in rows}
    observed_packets = []
    seen_packets = set()
    for row in rows:
        name = row["request_packet"]
        if name in seen_packets:
            continue
        seen_packets.add(name)
        if row["adapter_directory"] != "native-controls":
            actual = read(study/name)
            observed_packets.append(dict(method=row["adapter_directory"].removeprefix("native-"),
                adapter=actual["native_adapter_mode"],request_ids=[r["id"] for r in actual["rows"]]))
    checks.equal(observed_packets, packets, "Actual native packet boundaries and adapter modes match frozen schedule")
    checks.equal(packet_names, {p.relative_to(study).as_posix() for p in (study/"requests").glob("*.json")},
                 "No undeclared request packets")
    for name in packet_names:
        path = contained(study, name)
        packet = read(path)
        measured = [row for row in rows if row["request_packet"] == name]
        checks.equal([requests[row["id"]] for row in measured], packet["rows"], "Packet exact frozen commands/"+name)
        checks.require(all(row["request_packet_sha256"] == sha(path) and row["adapter_directory"] == packet["adapter_directory"]
                           for row in measured), "Packet bytes and execution lane/"+name)
    for row in rows:
        request = requests[row["id"]]
        for key in ("id", "bucket_xy_m", "values"):
            checks.equal(row[key], request[key], row["id"]+"/exact requested "+key)
        checks.require(row["native_release"] is True and row["action_substituted"] is False, row["id"]+"/exact native release")
        directory = study/row["adapter_directory"]
        if row["adapter_directory"] != "native-controls":
            verify_heading(row, request, directory, contract["scene"], home, prior, resolutions[str(directory)], checks)
        groups[(row["adapter_directory"],row["native_cache_file"])].append(row)
    for (directory, filename), group in groups.items():
        path = contained(study/directory, filename)
        checksum = sha(path)
        with np.load(path, allow_pickle=False) as archive:
            arrays = {name:archive[name] for name in archive.files}
        for row in group:
            col = row["native_cache_row"]
            checks.equal((checksum, str(arrays["record_ids"][col])), (row["native_cache_sha256"], row["id"]), row["id"]+"/authenticated dense column")
            verify_initial_trace(row,arrays,col,checks)
            trace = {k:arrays[k][:,col] for k in ("position_m","velocity_m_s","forces_N")}
            for key in ("time_s","dt_s"):
                if key in arrays:trace[key]=arrays[key]
            if "contact_distance_m" in arrays:trace["contact_distance_m"]=arrays["contact_distance_m"][:,col]
            args = contract["scene"], row["bucket_xy_m"], row["board_normal"], calibration, trace
            old, new = historical.score_trace(*args), revised.score_trace(*args)
            for key, value in old.items():
                _same(row[key], value, row["id"]+"/historical/"+key)
                checks.count += 1
            _same(row["board_first_score"], new, row["id"]+"/new independent score")
            checks.count += len(new)
    for directory in {r["adapter_directory"] for r in rows}:
        audit_native_options(study/directory, contract["scene"], calibration, a.assets, checks)
        checks.equal({name for namespace,name in groups if namespace==directory},
                     {p.relative_to(study/directory).as_posix() for p in (study/directory).rglob("*.npz")}, directory+"/no uncounted native drops")
    for phase in controls:
        measured = [r for r in rows if r["id"] in {x["id"] for x in controls[phase]}]
        gate = board_first_controls(measured)
        checks.equal(gate["status"], "passed", phase+"/both scoring control gates")
        checks.equal(report["controls"][phase], gate, phase+"/reported control counts")
    require_preflight(a)
    checks.equal(started["preflight_sha256"], sha(a.preflight_output), "Native started only after this passed preflight")
    checks.require(type(report["gpu_wait_s"]) in (int,float) and math.isfinite(report["gpu_wait_s"]) and report["gpu_wait_s"] >= 0,
                   "GPU wait reported separately from decision and native times")
    comparison = summarize(layouts,slots,rows,plans)
    checks.equal(report["comparison"], comparison, "Independent comparison accounting")
    checks.require(0 < report["gpu_lease_s"] <= contract["gpu_cap_s"] <=3600, "At most one GPU hour")
    return dict(version=VERSION,status="passed",checks_count=checks.count,frozen_before_test=True,
                report_sha256=sha(study/"report.json"),evidence_manifest_sha256=sha(study/"evidence-manifest.json"),
                contract_sha256=sha(study/"contract.json"),choices_sha256=sha(study/"choices.json"),
                checkpoint_sha256={m:sha(study/(m+"-checkpoint.json")) for m in METHODS},
                preflight_sha256=sha(a.preflight_output),
                native_started_sha256=sha(study/"native-started.json"),comparison=comparison,
                scope="Saved command/proof and deadline receipts, split isolation, immutable weights, full native contact/retention rescoring; no simulator rerun")
