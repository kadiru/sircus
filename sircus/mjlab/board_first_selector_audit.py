"""Read-only audit of corrected labels, frozen choices and native captures."""
from __future__ import annotations

from collections import defaultdict
import copy
import math
from pathlib import Path

import numpy as np

from .benchmark_verify import Checks, contained, digest, read, sha

VERSION = "sircus-board-first-selector-independent-audit-v1"


def verify_plans(a, contract, manifest, checks):
    from .board_first_selector_plan import BoardFirstFlightRanker
    from .clearance_expansion_audit import audit_branches
    from .clearance_expansion_plan import needs_expansion
    from .intergrid_candidate_replay_plan import command_key
    from .intergrid_library_plan import libraries
    from .intergrid_proposer_audit import audit_preparation
    from .neural_shot_model import order
    from .neural_shot_plan import menu, _choice, CHECK_LIMIT
    from .planning_budget_audit import verify_deadline
    from .planning_budget_plan import _wrap_choice, CachedRanker

    study = a.output
    prior, home = read(study / "training-prior.json"), read(study / "setup-home.json")
    raw = read(study / "shot-library-rows.json")
    libs = libraries(raw["original"], raw["additions"])
    source = read(study / "route-models.json")
    ranker = CachedRanker(BoardFirstFlightRanker(contract["scene"], read(study / "calibration.json")["score_config"],
                           a.assets, contract["flight_ranking"]["rule"]))
    plans = read(study / "plans.json")
    checkpoints = {method: read(study / (method+"-checkpoint.json")) for method in contract["methods"]}
    for site in plans:
        variant = site["method"]
        native = study / ("native-"+variant)
        label = site["layout"]["id"]+"/"+variant
        checks.equal(site, read(study / "preparation" / variant / (site["layout"]["id"]+".json")), label+"/frozen job receipt")
        # The shared budget worker knows one selector method. Variant renaming
        # happens outside it and must preserve its exact accepted snapshot.
        original_site = copy.deepcopy(site)
        original_site["method"] = "neural_selector"
        original_site["choice"] = _wrap_choice(site["choice"], "neural_selector")
        original_site["choices"] = [original_site["choice"]]
        snapshots = verify_deadline(original_site, native, manifest, checks, contract["planning_budget_s"])
        plan, layout = site["plan"], site["layout"]
        resolutions = {}
        for stage in ("base", "fallback", "neural"):
            audit_preparation(plan[stage], layout, "test", native, prior, home, resolutions, checks)
        audit_branches(plan["branches"], layout, {r["id"]: r for r in plan["neural"]["admitted"]},
                       native, prior, home, resolutions, checks)
        proposals = menu(layout, libs, source)
        ranked = order(checkpoints[variant], proposals)
        for snapshot in snapshots:
            checks.equal(snapshot["generated"]["menu"], proposals, label+"/common frozen proposal menu")
            checks.equal(snapshot["ranked_proposal_ids"], [r["id"] for r in ranked], label+"/checkpoint ranking")
            p = snapshot["plan"]
            checked = p["neural"]["proposals"]
            checks.equal(checked, ranked[:len(checked)], label+"/only ranked prefix checked")
            checks.require(len(checked) <= CHECK_LIMIT, label+"/bounded checks")
            checks.require(not any(p[s][k] for s in ("base", "fallback") for k in p[s]), label+"/no borrowed baseline checks")
            admitted = list({command_key(r): r for s in ("neural", "branches") for r in p[s]["admitted"]}.values())
            choice = _choice(layout, admitted, ranker, len(checked), snapshot["ranked_proposal_ids"])
            choice["predicted_flight_risk"] = needs_expansion(choice)
            checks.equal(snapshot["choice"], choice, label+"/same contact-permitting ranking before release")
        checks.equal(site["choice"], _wrap_choice(original_site["choice"], variant), label+"/variant renaming only")
    return plans


def audit(a):
    from . import board_first_score_oracle as revised
    from . import table_assisted_score_oracle as historical
    from .board_first_selector_model import attach_scores
    from .board_first_selector_model import VERSION as label_version
    from .board_first_selector_study import (METHODS, POSITIONS, VERSION as STUDY_VERSION,
                    board_first_controls, frozen, ledger_by_id, lines, source_inputs, summarize)
    from .cell1050_rebound_audit import verify_heading
    from .cell1050_rebound_study import control_requests
    from .intergrid_proposer_plan import sample_splits
    from .policy64_verify import audit_native_options
    from .table_assisted_env import _same

    checks, study = Checks(), a.output
    contract = frozen(a)
    source_inputs(a)
    report, manifest = read(study / "report.json"), read(study / "evidence-manifest.json")
    checks.equal((contract["version"], report["status"], report["phase"]), (STUDY_VERSION, "completed", "completed"), "Completed comparison")
    checks.equal((contract["physics_substeps"], contract["attempts_per_method_per_position"], contract["methods"], contract["test_positions"]),
                 (64, 1, list(METHODS), POSITIONS), "Fixed test protocol")
    checks.equal(report["evidence_manifest_sha256"], sha(study / "evidence-manifest.json"), "Manifest binding")
    actual = {str(p.relative_to(study)) for p in study.rglob("*") if p.is_file() and p.name not in {"report.json", "evidence-manifest.json"}}
    checks.equal(set(manifest), actual, "Complete evidence inventory")
    for name, expected in manifest.items():
        checks.equal(sha(contained(study, name)), expected, "Evidence/"+name)
    ledger = ledger_by_id(a)
    for split, name in (("train", "training-dataset.jsonl"), ("validation", "validation-dataset.jsonl")):
        expected = attach_scores(lines(a.neural_study / name), ledger)
        checks.equal(lines(study / (split+"-dataset.jsonl")), expected, split+"/only original split with revised label sidecars")
    checkpoints = {m: read(study / (m+"-checkpoint.json")) for m in METHODS}
    checks.equal(checkpoints["original"], read(a.neural_study / "neural-checkpoint.json"), "Original checkpoint unchanged")
    for key in ("architecture", "seed", "learning_rate", "weight_decay", "checkpoint_epochs", "feature_names", "heads"):
        checks.equal(checkpoints["retrained"][key], checkpoints["original"][key], "Matched training setting/"+key)
    from .neural_shot_model import _validate_checkpoint, _digest
    trained = checkpoints["retrained"]
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
    reconstructed = [dict(row, id=f"board-first-test-{i:02d}", context_id=f"board-first-test-{i:02d}", split="test") for i,row in enumerate(sampled)]
    checks.equal(layouts, reconstructed, "Fresh positions fixed independently of outcomes")
    for i, layout in enumerate(layouts):
        checks.require(all(math.dist(layout["bucket_xy_m"], xy) >= .005 for xy in provenance["excluded_coordinates_m"]), layout["id"]+"/history disjoint")
        checks.require(all(math.dist(layout["bucket_xy_m"], other["bucket_xy_m"]) >= .005 for other in layouts[:i]), layout["id"]+"/test disjoint")
    for root, bindings in provenance["history_files"].items():
        for name, expected in bindings.items():
            path = (Path(__file__).resolve().parents[2]/name if name.startswith("baselines/")
                    else Path(root)/name.removeprefix("output/"))
            checks.equal(sha(path), expected, "Historical coordinates/"+name)
    plans = verify_plans(a, contract, manifest, checks)
    choices, started = read(study/"choices.json"), read(study/"native-started.json")
    slots = choices["slots"]
    checks.equal(slots, [plan["choice"] for plan in plans], "Frozen choice exactly matches prepared command")
    checks.equal([(s["context_id"],s["method"]) for s in slots], [(l["id"],m) for l in layouts for m in METHODS], "One slot per policy and position")
    for name, expected in started["bindings"].items():
        checks.equal(manifest[name], expected, "Inputs frozen before TEST/"+name)
    required = {"original-checkpoint.json", "retrained-checkpoint.json", "train-dataset.jsonl", "validation-dataset.jsonl", "choices.json", "plans.json"}
    checks.require(required <= started["bindings"].keys(), "Weights, labels and choices frozen before test")
    planned = {method:[s["request"] for s in slots if s["method"]==method and s["request"] is not None] for method in METHODS}
    checks.equal(started["requests_sha256"], digest(planned), "One-drop command ledger frozen")
    rows = lines(study/"rows.jsonl")
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
    comparison = summarize(layouts,slots,rows,plans)
    checks.equal(report["comparison"], comparison, "Independent comparison accounting")
    checks.require(0 < report["gpu_lease_s"] <= contract["gpu_cap_s"] <=3600, "At most one GPU hour")
    return dict(version=VERSION,status="passed",checks_count=checks.count,frozen_before_test=True,
                report_sha256=sha(study/"report.json"),evidence_manifest_sha256=sha(study/"evidence-manifest.json"),
                contract_sha256=sha(study/"contract.json"),choices_sha256=sha(study/"choices.json"),
                checkpoint_sha256={m:sha(study/(m+"-checkpoint.json")) for m in METHODS},
                native_started_sha256=sha(study/"native-started.json"),comparison=comparison,
                scope="Saved command/proof and deadline receipts, split isolation, immutable weights, full native contact/retention rescoring; no simulator rerun")
