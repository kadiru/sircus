"""CPU-only independent audit of the two-slot gap-coverage ablation.

Old held-out outcomes are development history, never new counterfactual labels.
This auditor verifies the original source audits, regenerated forecasts, exact
new requested actions, native setup proofs and dense capture scoring. It does
not fit a model or execute a native shot. Version 2 changes only actor-number
comparison units and verifies saved nonlinear/action derivations separately;
the original producer source, evidence and any failed v1 audit stay immutable.
"""
from __future__ import annotations

import argparse
import copy
from collections import Counter
import importlib.metadata
import json
import math
from pathlib import Path
import time

import numpy as np

from . import calibrated_benchmark_verify as calibrated
from . import consistent_flight_diagnosis as reference
from . import curate_physics_ensemble as original
from . import physics_ensemble_verify as old
from . import policy64_verify as policy
from . import replay650_verify as replay
from .benchmark_verify import EvidenceError, contained, digest, read, sha

VERSION = "sircus-gap-coverage-audit-v2"
STUDY_VERSION = "sircus-gap-coverage-study-v1"
METHODS = ("consistent", "gap_coverage")
MAX_REQUESTS = 112
SEED = 20261801
AuditChecks = replay.AuditChecks


ACTOR_COMPARISON_REVISION = dict(
    version="sircus-gap-coverage-actor-comparison-v2",
    supersedes_auditor_version="sircus-gap-coverage-audit-v1",
    action_scales=[.04, .04, .05, 8., .12],
    action_units=["m", "m", "m", "degrees", "m"],
    independent_latent_atol=2e-6, independent_residual_atol=2e-6,
    normalized_action_atol=2e-6, saved_tanh_atol=1e-12, saved_action_atol=1e-12,
    scope="Unit-consistent CPU actor audit only; models, commands, native outcomes and scoring unchanged")


def actor_proposal_checks(recorded, mean, anchor, checks, label):
    """Separate float32 forward rounding from the exact saved action derivation.

    NumPy and Torch may differ by a few float32 ulps. Their independent means
    and residuals keep the original 2e-6 tolerance. The action comparison is
    normalized by the declared five scales, avoiding an inconsistent absolute
    bound shared by meters and degrees. Saved nonlinear/scaling identities
    receive a separate strict 1e-12 check and cannot hide action substitution.
    """
    scale = np.asarray(ACTOR_COMPARISON_REVISION["action_scales"])
    center = np.asarray(anchor["values"], dtype=float)
    mean = np.asarray(mean, dtype=float)
    residual = np.tanh(mean)
    checks.numeric(recorded["latent_mean"], mean, label+"/independent actor forward", 2e-6)
    checks.numeric(recorded["normalized_residual"], residual, label+"/independent actor residual", 2e-6)
    saved_mean = np.asarray(recorded["latent_mean"], dtype=float)
    saved_residual = np.asarray(recorded["normalized_residual"], dtype=float)
    saved_action = np.asarray(recorded["values"], dtype=float)
    checks.numeric(saved_residual, np.tanh(saved_mean), label+"/saved tanh derivation", 1e-12)
    checks.numeric(saved_action, center+saved_residual*scale, label+"/saved action derivation", 1e-12)
    checks.numeric((saved_action-center)/scale, residual, label+"/normalized independent action", 2e-6)


def expected_splits(old_spec):
    """Regenerate all exclusions and fifty fresh positions without labels."""
    excluded = {tuple(map(float, xy)) for xy in old_spec["excluded"]}
    for contexts in old_spec["splits"].values():
        excluded.update(tuple(map(float, c["bucket_xy_m"])) for c in contexts)
    excluded = sorted(excluded)
    used = np.asarray(excluded, dtype=float).reshape(-1, 2)
    rng, items = np.random.default_rng(SEED), []
    for _ in range(100000):
        if len(items) == 50:
            break
        xy = rng.uniform([.1, -.65], [1.24, .25])
        if len(used) and np.linalg.norm(used - xy, axis=1).min() < .01:
            continue
        items.append(dict(id=f"gap-coverage-test-{len(items):02d}", split="test", bucket_xy_m=xy.tolist()))
        used = np.vstack((used, xy))
    if len(items) != 50:
        raise EvidenceError("Could not regenerate fifty disjoint fresh layouts")
    return dict(splits={"test": items}, excluded=[list(xy) for xy in excluded])


def budget_checks(rows, checks):
    checks.equal(len(rows), MAX_REQUESTS, "Exactly112 requested slots including rejects and controls")
    checks.equal(len({r["id"] for r in rows}), MAX_REQUESTS, "Every requested identity unique")
    checks.equal(Counter(r["role"] for r in rows), Counter(test=100, positive_control=6, negative_control=6), "50x2 TEST plus12 controls, no new training")
    checks.equal(Counter(r["phase"] for r in rows), Counter(test=100, controls_pre=6, controls_post=6), "Three prespecified execution phases")
    tests = [r for r in rows if r["role"] == "test"]
    checks.equal(Counter(r["method"] for r in tests), Counter({m: 50 for m in METHODS}), "Two methods on the same fifty requested layouts")
    checks.equal([r["method"] for r in tests], [m for m in METHODS for _ in range(50)], "Frozen method execution order")
    layouts = None
    for method in METHODS:
        cohort = [r for r in tests if r["method"] == method]
        found = {r["context_id"]: r["bucket_xy_m"] for r in cohort}
        checks.equal(len(found), 50, method+"/one shot per layout without retry")
        if layouts is None:
            layouts = found
        else:
            checks.equal(found, layouts, method+"/same paired bucket positions")
    expected_order = ["positive_control"]*3+["negative_control"]*3+["test"]*100+["positive_control"]*3+["negative_control"]*3
    checks.equal([r["role"] for r in rows], expected_order, "Initial controls then fixed TEST then final controls")


def freeze_checks(packet, report, spec, checks):
    checks.equal((packet["version"], packet["contract_fingerprint"]), (STUDY_VERSION, spec["fingerprint"]), "Whole prediction plan exact contract")
    checks.require(packet.get("frozen_before_any_native_outcome") is True, "Plan frozen before any outcome")
    checks.require(type(packet.get("native_planning_queries")) is int and packet["native_planning_queries"] == 0, "No native planning queries")
    frozen, acquired = packet.get("frozen_unix_s"), report.get("acquired_unix_s")
    checks.require(type(frozen) in (int, float) and type(acquired) in (int, float)
                   and math.isfinite(frozen) and math.isfinite(acquired) and 0 < frozen <= acquired,
                   "Frozen plan precedes acquired native execution timestamp")
    budget_checks(packet["requests"], checks)


def selection_check(selection, order, candidates, directory, resolution, home, checks, label):
    """A strict first-admitted top-four choice; no native outcomes are inputs."""
    from .calibrated_admission import verify_connection
    checks.equal(sorted(order), list(range(16)), label+"/complete candidate ordering")
    records = selection["checks"]
    checks.require(1 <= len(records) <= 4, label+"/one to four strict geometry proofs")
    first = None
    for offset, entry in enumerate(records):
        index = order[offset]
        checks.equal((entry["candidate_index"], entry["candidate_id"]), (index, candidates[index]["id"]), label+"/exact ranking prefix")
        path = contained(directory, entry["connection_file"])
        checks.equal(sha(path), entry["connection_sha256"], label+"/bound geometry proof")
        proof = read(path)
        proof.update(file=entry["connection_file"], sha256=entry["connection_sha256"], cache_hit=True)
        checks.require(verify_connection(proof, resolution, home, candidates[index]["values"]), label+"/unchanged strict setup verification")
        checks.require(type(entry.get("passed")) is bool and entry["passed"] == proof["passed"], label+"/explicit actual geometry verdict")
        checks.equal(entry.get("reason"), proof.get("reason"), label+"/actual rejection reason")
        if entry["passed"]:
            first = index
            checks.equal(offset, len(records)-1, label+"/stop immediately at first admitted candidate")
    if first is None:
        checks.equal(len(records), 4, label+"/spend all four geometry checks before preserving rejected top1")
    chosen = first if first is not None else order[0]
    checks.equal(selection["selected_index"], chosen, label+"/exact chosen candidate")
    return dict(index=chosen, admitted=first is not None, geometry_checks=len(records))


def _source_checks(source, native_source, spec, original_spec, checks):
    checks.equal(spec["native_source_sha256"], original_spec["native_source_sha256"], "Original native source unchanged")
    for name, expected in original_spec["source_sha256"].items():
        checks.equal(spec["source_sha256"].get(name), expected, "Borrowed original model/loader/checker source unchanged/"+name)
    for root, files, label in ((source, spec["source_sha256"], "New isolated selector source"),
                              (native_source, spec["native_source_sha256"], "Original native source")):
        checks.require(isinstance(files, dict) and bool(files), label+"/nonempty byte manifest")
        for name, expected in files.items():
            checks.equal(sha(contained(root, name)), expected, label+"/"+name)
    checks.require("sircus/mjlab/consistent_flight_selector.py" in spec["source_sha256"]
                   and "sircus/mjlab/consistent_flight_study.py" in spec["source_sha256"], "Selector and producer frozen as source bytes")
    checks.equal({name: importlib.metadata.version(name) for name in spec["packages"]}, spec["packages"], "Pinned actual native dependency versions")


def gap_candidates(original_items, anchor):
    """Independently reconstruct the two replacements without producer imports."""
    expected = copy.deepcopy(original_items)
    for index, sign, bound in ((8, -1., "lower"), (9, 1., "upper")):
        row = expected[index]
        row["values"][4] = float(anchor["values"][4] + sign*.12)
        row["normalized_residual"][4] = sign
        row["source_kind"] = "bc_pose_anchor_gap_"+bound+"_endpoint"
        row["gap_coverage"] = dict(version="sircus-gap-coverage-candidates-v1",
            original_candidate=copy.deepcopy(original_items[index]),
            anchor_gap_m=float(anchor["values"][4]), domain_half_width_m=.12,
            bound=bound, normalized_gap=sign,
            scope="Only gap changed; board XYZ/tilt and original slot identity retained")
    return expected


def candidate_banks_check(saved, expected_original, anchor, checks, label):
    """Exact independent two-slot ablation, including unsupported endpoints."""
    checks.equal(set(saved), set(METHODS), label+"/exactly two candidate banks")
    expected = dict(consistent=expected_original,
                    gap_coverage=gap_candidates(expected_original, anchor))
    for method in METHODS:
        old.same(saved[method]["candidates"], expected[method], checks,
                 label+"/independently regenerated sixteen candidates/"+method, 1e-12)
    return expected


def _forecasts(study, packet, spec, train_rows, checks):
    """Recompute both equal-size proposal banks and all saved frozen forwards."""
    from . import consistent_flight_selector as selector
    from .calibrated_admission import verify_resolution
    import torch
    checkpoint = torch.load(study/"model/ensemble.pt", map_location="cpu", weights_only=True)
    actors = {m: old.common.load_checkpoint(study/f"model/{m}.pt", checks, "Unchanged "+m)["policy"]
              for m in ("bc", "ppo")}
    logs = read(study/"rankings.json")
    contexts = spec["splits"]["test"]
    checks.equal([r["context_id"] for r in logs], [c["id"] for c in contexts],
                 "All fifty candidate ledgers in frozen split order")
    tests = [r for r in packet["requests"] if r["role"] == "test"]
    by_key = {(r["context_id"], r["method"]): r for r in tests}
    expected_keys = {(c["id"], m) for c in contexts for m in METHODS}
    checks.equal(set(by_key), expected_keys, "Every paired request has its frozen context/method")
    home, prior = read(study/"setup-home.json"), read(study/"prior.json")
    home = home.get("home", home)
    calibration = read(study/"calibration.json")["score_config"]
    gravity = spec["native_reference"]["allocated_options"]["native"]["gravity"]
    selected = {}
    context_fields = ("source_hint", "effective_source_hint", "context_resolution_file",
        "context_resolution_sha256", "context_home_sha256", "context_layout_eligible",
        "context_anchor_admitted", "context_hard_reject")
    for context, log in zip(contexts, logs):
        ident = context["id"]
        checks.equal(log["bucket_xy_m"], context["bucket_xy_m"], ident+"/candidate bucket")
        checks.require(type(log.get("native_queries")) is int and log["native_queries"] == 0,
                       ident+"/no candidate native-flight queries")
        base = by_key[(ident, "consistent")]
        anchor = base["effective_source_hint"]
        checks.equal(log["anchor"], anchor, ident+"/exact shared anchor")
        for field in context_fields:
            checks.equal(log["context_metadata"][field], base[field], ident+"/ledger shared context/"+field)
        observation = calibrated.expected_observation(context["bucket_xy_m"], anchor)
        checks.numeric(log["observation"], observation, ident+"/independent actor observation", 0)
        checks.equal(set(log["actor_values"]), {"bc", "ppo"}, ident+"/two unchanged proposal actors")
        actor_values = {}
        for method in ("bc", "ppo"):
            recorded = log["actor_values"][method]
            mean = old.common.network_forward(actors[method], observation)
            actor_proposal_checks(recorded, mean, anchor, checks, ident+"/"+method)
            # The independent float32 forward can differ by a few ulps; byte-bound
            # recorded actor values reproduce the producer's literal candidate bank.
            actor_values[method] = recorded["values"]
        expected_original = old.candidates(context, anchor, actor_values["bc"], actor_values["ppo"], train_rows)
        expected_banks = candidate_banks_check(log["banks"], expected_original, anchor, checks, ident)
        resolution_path = contained(study/"native", base["context_resolution_file"])
        checks.equal(sha(resolution_path), base["context_resolution_sha256"], ident+"/bound shared context proof")
        resolution = read(resolution_path)
        resolution.update(file=base["context_resolution_file"], sha256=base["context_resolution_sha256"], cache_hit=True)
        checks.require(verify_resolution(resolution, prior["records"], home), ident+"/original shared setup contract")
        checks.equal(set(log["selection"]), set(METHODS), ident+"/exactly two frozen selections")
        for method in METHODS:
            expected = expected_banks[method]
            bank = log["banks"][method]
            normalized = [((np.asarray([*c["values"], *context["bucket_xy_m"]])-old.INPUT_CENTER)/old.INPUT_SCALE).tolist()
                          for c in expected]
            inferred = old.ensemble_forward(checkpoint, normalized)
            saved = bank["predictions"]
            old.same(saved, inferred, checks, ident+"/independent five-member forward/"+method, 3e-5)
            decoded = [old.decode_raw(p["member_raw_output"]) for p in saved]
            old.same(saved, decoded, checks, ident+"/exact saved raw-output decoder/"+method, 1e-12)
            # The unchanged geometry formula consumes only byte-verified forecasts
            # and requested geometry, never measured flight origins or TEST outcomes.
            ranking = selector.rank(spec["scene"], calibration, context, expected, saved, gravity=gravity)
            old.same(bank["ranking"], ranking, checks, ident+"/unchanged geometric ranking/"+method, 1e-10)
            chosen = selection_check(log["selection"][method], ranking["order"], expected,
                                     study/"native", resolution, home, checks, ident+"/"+method)
            row = by_key[(ident, method)]
            checks.equal(row["values"], expected[chosen["index"]]["values"], ident+"/exact selected request/"+method)
            checks.equal(row["candidate_index"], chosen["index"], ident+"/selected candidate identity/"+method)
            checks.equal(row["candidate_id"], expected[chosen["index"]]["id"], ident+"/stable selected candidate ID/"+method)
            checks.equal(row["ranking_sha256"], digest(log), ident+"/exact selected full ranking ledger/"+method)
            checks.equal(row["geometry_checks"], chosen["geometry_checks"], ident+"/charged selection geometry checks/"+method)
            checks.equal(row["model_candidates"], 16, ident+"/literal equal candidate inference budget/"+method)
            checks.require(type(row.get("native_planning_queries")) is int and row["native_planning_queries"] == 0,
                           ident+"/zero native inference requests/"+method)
            checks.equal(log["selection"][method]["selected_values"], row["values"], ident+"/ledger selected values/"+method)
            checks.require(log["selection"][method]["all_geometry_checks_rejected"] is (not chosen["admitted"]),
                           ident+"/explicit rejection accounting/"+method)
            selected[(ident, method)] = chosen
            for field in context_fields:
                checks.equal(row[field], base[field], ident+"/common pre-policy context/"+method+"/"+field)
            checks.equal(row["id"], "gap-coverage-"+method+"-"+ident, ident+"/literal new request identity/"+method)
            checks.numeric(row["observation"], observation, ident+"/unchanged shared observation/"+method, 0)
            checks.equal(row["source_teacher_id"], base["source_hint"]["id"], ident+"/unchanged teacher provenance/"+method)
            checks.equal(row["bucket_xy_m"], context["bucket_xy_m"], ident+"/same paired bucket/"+method)
            checks.equal(row["command_values"], row["values"], ident+"/no command replacement/"+method)
            checks.require(row["selection_or_learning"] is False and row["ball_flight_search_at_inference"] is False,
                           ident+"/no learning or native inference search/"+method)
            checks.equal(row["perturbation_mixed_units"], [0.]*5, ident+"/nominal fixed64/"+method)
            checks.equal(row["frozen_artifact_sha256"], original.ARTIFACT_SHA["ensemble"], ident+"/original saved checkpoint/"+method)
            checks.equal(row["prediction_model_sha256"], original.ARTIFACT_SHA["ensemble"], ident+"/forecast checkpoint identity/"+method)
            normalized = ((np.asarray([*row["values"], *row["bucket_xy_m"]])-old.INPUT_CENTER)/old.INPUT_SCALE).tolist()
            old.same(row["model_prediction"], old.ensemble_forward(checkpoint, [normalized])[0], checks,
                     ident+"/independent requested forecast/"+method, 3e-5)
            old.same(row["model_prediction"], old.decode_raw(row["model_prediction"]["member_raw_output"]), checks,
                     ident+"/exact forecast decoder/"+method, 1e-12)
    return selected


def audit(study, source, native_source, assets, *, ensemble_study, ensemble_audit, parent_study, parent_audit, reference_study, reference_audit):
    started, checks = time.monotonic(), AuditChecks()
    study, source, native_source, assets = map(Path, (study, source, native_source, assets))
    _, original_report, original_spec, _, _, _, _ = original.verified_inputs(ensemble_study, ensemble_audit)
    checks.require(True, "Original897304-check ensemble proof and all frozen bytes authenticated")
    _, _, reference_spec, _, _, _, _ = reference.verified_inputs(reference_study, reference_audit)
    checks.require(True, "Completed485047-check consistent-flight proof and all frozen bytes authenticated")
    parent, _, train_rows, parent_manifest, parent_rows = old.parent_inputs(parent_study, parent_audit, checks)
    manifest = replay.check_manifest(study, checks)
    spec, report, packet = map(read, (study/"contract.json", study/"report.json", study/"prediction-plan.json"))
    unsigned = dict(spec); fp = unsigned.pop("fingerprint", None)
    checks.equal(digest(unsigned), fp, "Exact pre-outcome source and protocol fingerprint")
    checks.equal((spec["version"], report["version"]), (STUDY_VERSION, STUDY_VERSION), "New selector study identity")
    checks.equal(report["contract_fingerprint"], fp, "Report exact frozen protocol")
    checks.equal((report["status"], report["phase"]), ("completed", "completed"), "Complete finite comparison required")
    checks.equal(report["evidence_sha256"], manifest, "Report binds all recorded bytes")
    checks.equal(report["protocol"], spec["protocol"], "No post-outcome protocol change")
    checks.equal(report["source_sha256"], spec["source_sha256"], "Report exact pre-outcome source bytes")
    checks.equal(report["source_report_sha256"], reference.SOURCE_HASHES["report"], "Report exact development source")
    checks.equal(report["selector_sha256"], spec["source_sha256"]["sircus/mjlab/consistent_flight_selector.py"],
                 "Report unchanged frozen ranker bytes")
    checks.equal(spec["parent_contract_fingerprint"], parent["fingerprint"], "Exact authenticated TRAIN-parent contract")
    checks.equal(spec["parent_bindings"], original_spec["parent_bindings"], "Unchanged TRAIN-parent proof and checkpoint bindings")
    checks.equal((spec["substeps"], spec["maximum_requests"], report["new_training_requests"]),
                 (64, 112, 0), "Whole study fixed64, bounded112 and zero new fitting")
    checks.equal((spec["ensemble_report_sha256"], spec["ensemble_audit_sha256"]),
                 (original.REPORT_SHA, original.AUDIT_SHA), "Exact independently audited inherited ensemble provenance")
    for key, hash_key in (("consistent_report_sha256", "report"), ("consistent_audit_sha256", "audit"),
                          ("consistent_manifest_sha256", "manifest")):
        checks.equal(spec[key], reference.SOURCE_HASHES[hash_key], "Exact development comparison provenance/"+key)
    for key, value in dict(substeps=64, methods=list(METHODS), test_layouts=50, control_requests=12,
                           maximum_fresh_requests=112, new_train_requests=0, no_test_flight_search=True,
                           candidates=16, maximum_geometry_checks=4, test_seed=SEED,
                           minimum_split_distance_m=.01, noise=False, model_weights_unchanged=True,
                           no_hard_aperture_veto=True, post_bounce_robot_clearance_added=False,
                           changed_candidate_indices=[8, 9], gap_residual_endpoints=[-1, 1],
                           unchanged_candidate_count=14).items():
        checks.equal(spec["protocol"][key], value, "Literal bounded comparison/"+key)
    checks.equal(spec["new_training_requests"], 0, "No new fit or native training acquisition")
    checks.equal((spec["native_gpu"], spec["maximum_acquired_s"], spec["maximum_queue_s"]),
                 ("GPU-ab1950b0-01dd-0fab-407f-2d17936b27f4", 3600, 86400), "Only4090 with finite queue/acquired budget")
    for key in ("scene", "identity", "packages", "native_reference"):
        checks.equal(spec[key], original_spec[key], "Unchanged original simulation contract/"+key)
    _source_checks(source, native_source, spec, original_spec, checks)
    for name, expected_sha in reference_spec["source_sha256"].items():
        if name in spec["source_sha256"]:
            checks.equal(spec["source_sha256"][name], expected_sha, "Unchanged reference source/"+name)
    for name in ("gap_coverage_study", "gap_coverage_candidates", "gap_coverage_verify"):
        checks.require("sircus/mjlab/"+name+".py" in spec["source_sha256"], "New ablation source frozen/"+name)
    for name in ("calibration.json", "verification.json", "setup-home.json", "prior.json"):
        checks.equal(sha(study/name), parent_manifest[name], "Unchanged original geometry/calibration/"+name)
    expected = expected_splits(reference_spec)
    checks.equal((spec["splits"], spec["excluded"]), (expected["splits"], expected["excluded"]), "Independent fresh seed excludes all historical and used development positions")
    checks.equal(read(study/"splits.json"), spec["splits"], "Saved split coordinates preserve exact frozen contract")
    checks.equal(read(study/"exclusions.json"), spec["excluded"], "Saved exclusion coordinates preserve exact frozen contract")
    for method, expected_sha in original.ARTIFACT_SHA.items():
        checks.equal(sha(study/f"model/{method}.pt"), expected_sha, "No retrained or substituted model/"+method)
    checks.equal(set(spec["artifacts"]), set(METHODS) | {"bc", "ppo", "ensemble"}, "All frozen proposal and method checkpoint identities")
    checks.equal(report["artifacts"], spec["artifacts"], "Published artifacts preserve frozen method mapping")
    checks.equal(packet["model_artifacts"], spec["artifacts"], "Pre-outcome inference artifacts preserve frozen method mapping")
    for method in (*original.ARTIFACT_SHA, *METHODS):
        source_method = method if method in ("bc", "ppo") else "ensemble"
        checks.equal(spec["artifacts"][method], dict(path=f"model/{source_method}.pt", sha256=original.ARTIFACT_SHA[source_method]), "Exact immutable method checkpoint/"+method)
    checks.equal(report["prediction_plan_sha256"], sha(study/"prediction-plan.json"), "Report frozen plan bytes")
    checks.equal(packet["rankings_sha256"], sha(study/"rankings.json"), "Frozen all-candidate ranking bytes")
    checks.equal(report["rankings_sha256"], packet["rankings_sha256"], "Report same ranking bytes")
    freeze_checks(packet, report, spec, checks)
    selected = _forecasts(study, packet, spec, train_rows, checks)
    rows = replay.read_records(study/"records.jsonl")
    budget_checks(rows, checks)
    checks.equal([r["id"] for r in rows], [r["id"] for r in packet["requests"]], "All112 frozen slots executed once in exact order")
    for row, request in zip(rows, packet["requests"]):
        for key, value in request.items():
            checks.equal(row[key], value, row["id"]+"/unchanged request/"+key)
        checks.require(row.get("action_substituted") is False, row["id"]+"/no outcome-conditioned replacement")
        if row["role"] == "test" and row["method"] in METHODS:
            checks.equal(row["native_release"], selected[(row["context_id"], row["method"])]["admitted"], row["id"]+"/native admission matches frozen selected proof")
        if row["native_release"]:
            checks.equal(row["contact_protocol"], dict(substeps=64, board_mode="articulated", gravity_mode="passive_only"), row["id"]+"/same held-endpoint fixed64 native physics")
            checks.equal(row["scoring_version"], "sircus-calibrated-native-capture-v2", row["id"]+"/same versioned capture semantics")
            checks.require(row["scalar_score_audit_passed"] is True, row["id"]+"/online independent scorer agreed")
            checks.equal(row["effective_options_sha256"], sha(study/"native/effective-options.json"), row["id"]+"/actual allocated engine identity")
    controls = [next(r for r in parent_rows if r["phase"] == "controls_pre" and r["role"] == role and r.get("repeat") == 0)
                for role in ("positive_control", "negative_control")]
    gates = {}
    for phase, key in (("controls_pre", "pre"), ("controls_post", "post")):
        block = [r for r in rows if r["phase"] == phase]
        gates[key] = old.control_gate(block)
        checks.equal(gates[key]["status"], "passed", phase+"/native positive and native-miss controls pass")
        for row in block:
            src = controls[0 if row["role"] == "positive_control" else 1]
            checks.equal((row["values"], row["bucket_xy_m"]), (src["values"], src["bucket_xy_m"]), row["id"]+"/original matched control")
            checks.equal(row["id"], "gap-coverage-"+phase+"-"+row["role"]+"-r"+str(row["repeat"]), row["id"]+"/literal new control identity")
            checks.equal((row["control_source_id"], row["control_source_sha256"], row["frozen_artifact_sha256"]),
                         (src["id"], digest(src), digest(src)), row["id"]+"/full original control provenance")
        for role in ("positive_control", "negative_control"):
            checks.equal([r["repeat"] for r in block if r["role"] == role], [0, 1, 2], phase+"/three exact unchanged control repeats/"+role)
    checks.equal(report["control_gates"], gates, "Reported exact pre/post control gates")
    calibrated.check_rows(rows, checks)
    setup = calibrated.check_setup_evidence(study, rows, read(study/"prior.json"), spec["identity"]["setup"]["home"], checks)
    calibration = replay.check_calibration(study, spec["identity"]["calibration"], checks)
    from .learning_solve import required_asset_paths
    checks.equal({name: sha(contained(assets.parent, name)) for name in required_asset_paths(assets)}, spec["identity"]["asset_sha256"], "Actual collision assets unchanged")
    allocated = policy.audit_native_options(study/"native", spec["scene"], calibration, assets, checks)
    checks.equal(allocated, spec["native_reference"]["allocated_options"], "Actual allocated64 options exactly match original")
    traces = calibrated.rescore_rows(study/"native", rows, spec["scene"], calibration, checks,
                                    "New frozen two-slot gap-coverage comparison", spec["identity"]["trace_schema"], assets=assets)
    comparisons = [dict(method=m, counts=old.statistics([r for r in rows if r["role"] == "test" and r["method"] == m])) for m in METHODS]
    checks.equal(report["counts"], old.statistics(rows), "All requests, rejects, invalids and measured captures accounted")
    for actual, expected_row in zip(report["comparisons"], comparisons):
        checks.equal((actual["method"], actual["counts"]), (expected_row["method"], expected_row["counts"]), "Reported native comparison/"+expected_row["method"])
    checks.equal(len(report["comparisons"]), 2, "Both measured method comparisons")
    expected_positions = [dict(context_id=r["context_id"], method=r["method"], record_id=r["id"],
                               bucket_xy_m=r["bucket_xy_m"], values=r["values"], capture=policy.captured(r),
                               native_release=r["native_release"], clean=policy.captured(r) and r["route_quality"] == 2,
                               whole_ball_entry=r["whole_ball_entry"], retention_s=r["max_retention_s"])
                          for r in rows if r["role"] == "test"]
    checks.equal(report["test_positions"], expected_positions, "Every UI result is the same measured TEST request; no predicted captures")
    return dict(version=VERSION, status="passed", failures=[], checks=checks.entries, checks_count=checks.count,
                actor_comparison_revision=copy.deepcopy(ACTOR_COMPARISON_REVISION),
                contract_fingerprint=fp, report_sha256=sha(study/"report.json"), evidence_manifest_sha256=sha(study/"evidence-manifest.json"),
                auditor_sha256=sha(Path(__file__)), producer_source_sha256=spec["source_sha256"], native_source_sha256=spec["native_source_sha256"],
                original_ensemble_report_sha256=original.REPORT_SHA, original_ensemble_audit_sha256=original.AUDIT_SHA,
                consistent_report_sha256=reference.SOURCE_HASHES["report"], consistent_audit_sha256=reference.SOURCE_HASHES["audit"],
                requested_trials=len(rows), native_releases=len(traces), training_requests=0, comparisons=comparisons,
                control_gates=gates, setup_evidence=setup, elapsed_s=time.monotonic()-started,
                limitations=["Same frozen models and ranker; only two gap proposals changed. Fifty fresh paired bucket positions, one nominal request per method.",
                             "Ballistic aperture proxy does not simulate rim collisions or one-second retention; actual native scoring remains authoritative.",
                             "Equal sixteen-candidate and top-four geometry budgets; native wall time may differ. No retraining or full-table generalization claim.",
                             "Fixed64 only; earlier128 failures remain. No driven arm transit, grasp, noise or physical validation."])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("study", "source", "native-source", "assets", "ensemble-study", "ensemble-audit", "parent-study", "parent-audit", "output"):
        parser.add_argument("--"+key, type=Path, required=True)
    parser.add_argument("--reference-study", "--consistent-study", dest="reference_study", type=Path, required=True)
    parser.add_argument("--reference-audit", "--consistent-audit", dest="reference_audit", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or any(args.output.resolve().is_relative_to(p.resolve()) for p in (args.study, args.ensemble_study, args.parent_study, args.reference_study)):
        raise ValueError("New audit file outside immutable studies required")
    try:
        result = audit(args.study, args.source, args.native_source, args.assets,
                       ensemble_study=args.ensemble_study, ensemble_audit=args.ensemble_audit,
                       parent_study=args.parent_study, parent_audit=args.parent_audit,
                       reference_study=args.reference_study, reference_audit=args.reference_audit)
    except Exception as error:
        args.output.write_text(json.dumps(dict(version=VERSION, status="failed", failures=[type(error).__name__+": "+str(error)]), indent=2)+"\n")
        raise
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    print(json.dumps({k: result[k] for k in ("status", "checks_count", "requested_trials", "native_releases")}))


if __name__ == "__main__":
    main()
