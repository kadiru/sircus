"""Independent CPU verification of frozen physics-ensemble shot planning.

All native contacts are rescored from immutable dense observations. Predicted
success, model disagreement and geometry-only candidate search remain separate
from a measured capture. Existing BC/PPO models and historical results are not
rewritten or silently activated by this read-only auditor.
"""
from __future__ import annotations

import argparse
from collections import Counter
import copy
import importlib.metadata
import json
import math
from pathlib import Path
import sys
import time

import numpy as np

from . import benchmark_verify as common
from . import calibrated_benchmark_verify as calibrated
from . import policy64_verify as policy_audit
from . import replay650_verify as replay
from . import xy650_verify as xy_audit
from .benchmark_verify import EvidenceError, contained, digest, read, sha

VERSION = "sircus-physics-ensemble-audit-v1"
STUDY_VERSION = "sircus-physics-ensemble-v1"
PARENT_FP = "d30ad91111cb8beb26438f2271357a103daf50d0399ac3646f3b7f163d87f02e"
PARENT_AUDIT_SHA = "4dc317e0697e426a4708c3c997f8671d97b1f49de6bd4450c56e6487a6e93ac3"
PARENT_REPORT_SHA = "637731856cba291b57a1119b32b56fe97a3c4cbc000d55db4bfb1793c6757cc3"
PARENT_MANIFEST_SHA = "632ce6cd692ea68d862243deb07c71cb605994fd017f7f15a603aa7995bd3514"
METHODS = ("bc", "ppo", "ensemble")
MAX_REQUESTS = 264
MODEL_SEEDS = tuple(range(20261611, 20261616))
INPUT_CENTER = np.asarray([.67, -.2, .35, 30., .3, .67, -.2])
INPUT_SCALE = np.asarray([.57, .45, .3, 30., .3, .57, .45])
AuditChecks = replay.AuditChecks


def same(actual, expected, checks, label, tolerance=1e-9):
    if isinstance(expected, dict):
        checks.require(isinstance(actual, dict), label+"/object")
        checks.equal(set(actual), set(expected), label+"/complete fields")
        for key, value in expected.items():
            same(actual[key], value, checks, label+"/"+str(key), tolerance)
    elif isinstance(expected, (list, tuple)):
        checks.require(isinstance(actual, (list, tuple)), label+"/sequence")
        checks.equal(len(actual), len(expected), label+"/length")
        for index, value in enumerate(expected):
            same(actual[index], value, checks, label+"/"+str(index), tolerance)
    elif type(expected) is bool:
        checks.require(type(actual) is bool and actual is expected, label+"/explicit boolean")
    elif isinstance(expected, (float, np.floating)):
        checks.numeric(actual, expected, label, tolerance)
    else:
        checks.equal(actual, expected, label)


def parent_inputs(study, audit_path, checks):
    """Authenticate old models and only the1024 current native TRAIN labels."""
    study, audit_path = Path(study), Path(audit_path)
    for path, expected in ((audit_path, PARENT_AUDIT_SHA), (study/"report.json", PARENT_REPORT_SHA),
                            (study/"evidence-manifest.json", PARENT_MANIFEST_SHA)):
        checks.equal(sha(path), expected, "Exact complete parent artifact/"+path.name)
    spec, report, proof = read(study/"contract.json"), read(study/"report.json"), read(audit_path)
    unsigned = dict(spec);fp = unsigned.pop("fingerprint", None)
    checks.equal((fp, digest(unsigned)), (PARENT_FP, PARENT_FP), "Original fixed64 parent full contract")
    checks.equal(proof["version"], "sircus-policy64-independent-audit-v1", "Original complete independent policy proof")
    checks.require(proof.get("status") == "passed" and not proof.get("failures")
                   and proof["checks_count"] == len(proof["checks"]) == 346485
                   and all(item.get("passed") is True for item in proof["checks"]), "All346485 parent checks passed")
    checks.equal((proof["report_sha256"], proof["evidence_manifest_sha256"]), (PARENT_REPORT_SHA, PARENT_MANIFEST_SHA), "Proof binds original report and full manifest")
    checks.equal((report["contract_fingerprint"], proof["contract_fingerprint"]), (PARENT_FP, PARENT_FP), "Original report/proof exact model study")
    checks.equal((report["status"], report["phase"]), ("completed", "completed"), "Original study remains completed")
    manifest = read(study/"evidence-manifest.json")
    checks.equal(report["evidence_sha256"], manifest, "Original raw-evidence manifest identity")
    for name in ("contract.json", "rows.jsonl", "splits.json", "prior.json", "setup-home.json", "calibration.json", "verification.json", "selected.json"):
        checks.equal(sha(contained(study, name)), manifest[name], "Consumed parent bytes/"+name)
    raw = replay.read_records(study/"rows.jsonl")
    checks.equal((len(raw), sum(bool(row["native_release"]) for row in raw)), (1284, 1103), "Unchanged parent attempt/native counts")
    train = [row for row in raw if row.get("role") == "train"]
    checks.equal(len(train), 1024, "Exactly1024 current parent TRAIN requests, not historical6144 bootstrap labels")
    checks.equal([row["train_index"] for row in train], list(range(1024)), "Every original current TRAIN exposure retained")
    checks.equal([row["id"] for row in train], [f"policy64-train-{index:05d}" for index in range(1024)], "Literal unique TRAIN source IDs")
    contexts = {context["id"]: context for context in spec["splits"]["train"]}
    for row in train:
        label = row["id"]
        checks.equal(row["bucket_xy_m"], contexts[row["context_id"]]["bucket_xy_m"], label+"/TRAIN-only source bucket")
        checks.equal(row["perturbation_mixed_units"], [0.]*5, label+"/original nominal source request")
        if row["native_release"]:
            checks.equal(row["contact_protocol"]["substeps"], 64, label+"/canonical native64 label source")
            checks.equal(row["scoring_version"], "sircus-calibrated-native-capture-v2", label+"/versioned calibrated source label")
            checks.require(row.get("scalar_score_audit_passed") is True, label+"/source independent scorer agreed")
        for filename_key, hash_key in (("context_resolution_file", "context_resolution_sha256"), ("connection_file", "connection_sha256")):
            checks.equal(manifest["native/"+row[filename_key]], row[hash_key], label+"/source setup proof bound by passed parent audit")
        if row["native_release"]:
            checks.equal(manifest["native/"+row["native_cache_file"]], row["native_cache_sha256"], label+"/source dense trace bound by passed parent audit")
    checks.equal((sum(bool(row["native_release"]) for row in train), sum(bool(row.get("invalid_physics")) for row in train)), (887, 1), "Source execution and known physics-invalid row retained distinctly")
    checks.equal(report["selected"], proof["selected"], "Saved PPO selection remains unchanged")
    for method, value in (("bc", report["artifacts"]["bc"]), ("ppo", report["selected"])):
        checks.equal(sha(contained(study, value["path"])), value["sha256"], "Unmodified saved model bytes/"+method)
        checks.equal(manifest[value["path"]], value["sha256"], "Saved baseline checkpoint bound to original audit/"+method)
    return spec, report, train, manifest, raw


def literal_budget_checks(rows, checks):
    """No new TRAIN acquisition or uncounted native candidate flights."""
    checks.equal(len(rows), MAX_REQUESTS, "Exactly264 native/admission request slots")
    checks.equal(len({row["id"] for row in rows}), MAX_REQUESTS, "Unique requested-trial identities")
    checks.equal(Counter(row["role"] for row in rows), Counter(validation=72, test=180, positive_control=6, negative_control=6), "24VALx3 +60TESTx3 +12 matched controls, no new TRAIN")
    for role, layouts in (("validation", 24), ("test", 60)):
        cohort = [row for row in rows if row["role"] == role]
        checks.equal(Counter(row["method"] for row in cohort), Counter({method: layouts for method in METHODS}), role+"/three methods get one native request per layout")
        for method in METHODS:
            selected = [row for row in cohort if row["method"] == method]
            checks.equal(len({row["context_id"] for row in selected}), layouts, role+"/no retries or substituted positions/"+method)


def physical_target_valid(row):
    """Executed-but-physics-invalid is not a valid physical training target."""
    return bool(row.get("native_release") is True and row.get("invalid_physics") is False and not row.get("invalid"))


def audit_protocol(protocol, checks):
    literals = dict(substeps=64, methods=list(METHODS), historical_current_train_requests=1024, new_train_requests=0,
        validation_layouts=24, test_layouts=60, validation_requests=72, test_requests=180, control_requests=12,
        maximum_fresh_requests=264, noise=False, no_test_flight_search=True, minimum_split_distance_m=.01,
        validation_seed=20261603, test_seed=20261604, model_seeds=list(MODEL_SEEDS), fit_steps=600, fit_lr=.001,
        fit_device="cpu", ensemble_members=5, candidates=16, maximum_geometry_checks=4)
    for key, value in literals.items():
        checks.equal(protocol[key], value, "Literal offline forward-ensemble study/"+key)
    checks.require("no validation tuning" in protocol["selection"].lower() and "before any evaluation outcomes" in protocol["selection"].lower(),
                   "All252 predictions frozen without validation selection/tuning")
    checks.require("unequal historical exposure" in protocol["scope"].lower() and "geometry" in protocol["scope"].lower(),
                   "Model-ranking and different acquisition/inference costs explicitly scoped")


def expected_splits(parent):
    excluded = {tuple(float(value) for value in xy) for xy in parent["exclusions"]}
    for role in ("train", "validation", "test"):
        excluded.update(tuple(float(value) for value in context["bucket_xy_m"]) for context in parent["splits"][role])
    excluded = sorted(excluded);used = np.asarray(excluded, dtype=float);splits = {}
    for role, count, seed in (("validation", 24, 20261603), ("test", 60, 20261604)):
        generator, group = np.random.default_rng(seed), []
        for _ in range(100000):
            if len(group) == count:
                break
            xy = generator.uniform([.1, -.65], [1.24, .25])
            if len(used) and np.linalg.norm(used-xy, axis=1).min() < .01:
                continue
            group.append(dict(id=f"physics-ensemble-{role}-{len(group):02d}", split=role, bucket_xy_m=xy.tolist()))
            used = np.vstack((used, xy))
        if len(group) != count:
            raise EvidenceError("Could not regenerate disjoint24/60 fresh evaluation layouts")
        splits[role] = group
    return dict(splits=splits, excluded=[list(xy) for xy in excluded])


def target_row(row, first_flight, velocity):
    """Nine targets/masks; missing physical data is never an observed zero."""
    x = np.asarray([*row["values"], *row["bucket_xy_m"]], dtype=float)
    target, mask = np.zeros(9, dtype=float), np.zeros(9, dtype=float)
    target[0] = float(row["native_release"]);mask[0] = 1.
    valid = physical_target_valid(row)
    if valid:
        target[1:5] = [float(row["board_bounce"]), float(row["whole_ball_entry"]), float(row["route_success"]), min(float(row["max_retention_s"]), 1.)]
        mask[1:5] = 1.
        if first_flight is not None and first_flight.get("available") is True:
            if velocity is None or np.asarray(velocity).shape != (3,) or not np.isfinite(velocity).all():
                raise EvidenceError("Available first flight requires its measured finite rebound velocity")
            direction = np.asarray(row["bucket_xy_m"], dtype=float)-np.asarray(row["values"][:2], dtype=float)
            distance = float(np.linalg.norm(direction))
            if not distance > 0:
                raise EvidenceError("Nonzero board-to-bucket bearing required")
            direction /= distance
            lateral = np.asarray([-direction[1], direction[0]])
            target[5:] = [math.log1p(float(first_flight["closest_opening_distance_m"])/.1),
                          float(np.asarray(velocity)[:2]@direction), float(np.asarray(velocity)[:2]@lateral), float(velocity[2])]
            mask[5:] = 1.
    return x, target, mask


def reconstruct_dataset(study, rows, spec, manifest, checks, *, require_train=True):
    """Independently reconstruct all observed targets, masks and row provenance."""
    if require_train and any(row.get("role") != "train" for row in rows):
        raise EvidenceError("Only authenticated TRAIN labels may enter ensemble fit")
    grouped, targets = {}, {}
    for row in rows:
        if row["native_release"]:
            grouped.setdefault(row["native_cache_file"], []).append(row)
    for filename, cohort in grouped.items():
        path = contained(study/"native", filename)
        checks.equal(sha(path), manifest["native/"+filename], "Consumed TRAIN dense observation bytes/"+filename)
        with np.load(path, allow_pickle=False) as archive:
            arrays = {key: archive[key] for key in ("position_m", "velocity_m_s", "forces_N", "time_s", "dt_s", "contact_distance_m") if key in archive.files}
            ids = archive["record_ids"].tolist()
            for row in cohort:
                column = row["native_cache_row"];label = row["id"]
                checks.equal(ids[column], label, label+"/TRAIN trace column identity")
                checks.equal(row["native_cache_sha256"], manifest["native/"+filename], label+"/TRAIN trace authenticated by parent proof")
                metric = dict(available=False, reason="unavailable_or_invalid_trace");velocity = None
                if physical_target_valid(row):
                    trace = {key: value[:, column] if value.ndim >= 2 and key != "time_s" else value for key, value in arrays.items()}
                    metric = xy_audit.first_flight(spec["scene"], row["bucket_xy_m"], row["board_normal"], spec["identity"]["calibration"]["score_config"], trace, valid=True)
                    if metric["available"]:
                        velocity = trace["velocity_m_s"][metric["start_index"]]
                targets[label] = metric, velocity
    result = []
    for row in rows:
        metric, velocity = targets.get(row["id"], (dict(available=False, reason="unavailable_or_invalid_trace"), None))
        raw, y, mask = target_row(row, metric, velocity)
        x = ((raw-INPUT_CENTER)/INPUT_SCALE).tolist()
        result.append(dict(id=row["id"], role=row["role"], context_id=row["context_id"], bucket_xy_m=row["bucket_xy_m"], values=row["values"],
            x=x, y=y.tolist(), mask=mask.tolist(), record_sha256=digest(row), native_cache_sha256=row.get("native_cache_sha256"),
            native_cache_row=row.get("native_cache_row"), native_cache_file=row.get("native_cache_file"), first_flight=metric,
            physics_invalid=bool(row.get("invalid_physics")), native_release=row.get("native_release") is True))
    return result


def independent_network():
    import torch
    return torch.nn.Sequential(torch.nn.Linear(7, 64), torch.nn.SiLU(), torch.nn.Linear(64, 64), torch.nn.SiLU(), torch.nn.Linear(64, 9)).cpu()


def independent_loss(output, y, mask):
    import torch
    binary = torch.nn.functional.binary_cross_entropy_with_logits(output[:, :4], y[:, :4], reduction="none")
    predicted = torch.cat((output[:, 4:5].sigmoid(), output[:, 5:]), dim=1)
    losses = torch.cat((binary, (predicted-y[:, 4:])**2), dim=1)
    return ((losses*mask).sum(dim=0)/mask.sum(dim=0).clamp_min(1)).sum()


def replay_fit(dataset, checkpoint, contract_fp, checks, *, steps=600, seeds=MODEL_SEEDS):
    """Reproduce each seeded bootstrap/initialization/full-batch Adam update."""
    import torch
    torch.set_num_threads(2)
    checks.equal(checkpoint["version"], "sircus-physics-ensemble-model-v1", "Versioned offline forward model")
    checks.equal(checkpoint["contract_fingerprint"], contract_fp, "Ensemble trained under exact frozen study")
    checks.equal((checkpoint["fit_device"], checkpoint["fit_steps"], checkpoint["lr"]), ("cpu", steps, .001), "Literal CPU600-step Adam fit")
    checks.equal(checkpoint["member_seeds"], list(seeds), "Five fixed independent bootstrap seeds")
    checks.equal(checkpoint["dataset_sha256"], digest(dataset), "Exact TRAIN-only target/mask matrix")
    checks.equal((checkpoint["center"], checkpoint["scale"]), (INPUT_CENTER.tolist(), INPUT_SCALE.tolist()), "Fixed normalization constants")
    for key in ("members", "initial_members", "optimizers", "bootstrap_indices"):
        checks.equal(len(checkpoint[key]), len(seeds), "All ensemble members preserved/"+key)
    x, y, mask = (torch.as_tensor(np.asarray([row[key] for row in dataset]), dtype=torch.float32) for key in ("x", "y", "mask"))
    checks.require(x.shape == (len(dataset), 7) and y.shape == mask.shape == (len(dataset), 9)
                   and all(torch.isfinite(value).all().item() for value in (x, y, mask))
                   and ((mask == 0)|(mask == 1)).all().item(), "Finite observed-mask fit tensors")
    curves = []
    for member, seed in enumerate(seeds):
        torch.manual_seed(seed);net = independent_network()
        for key, value in net.state_dict().items():
            checks.numeric(common.numpy_state(checkpoint["initial_members"][member])[key], value.detach().numpy(), f"Member{member}/seeded initialization/"+key, 0)
        indices = np.random.default_rng(seed).integers(0, len(dataset), len(dataset)).tolist()
        checks.equal(checkpoint["bootstrap_indices"][member], indices, f"Member{member}/literal with-replacement TRAIN bootstrap")
        xx, yy, mm = x[indices], y[indices], mask[indices]
        optimizer = torch.optim.Adam(net.parameters(), lr=.001)
        for step in range(steps+1):
            loss = independent_loss(net(xx), yy, mm)
            if step%100 == 0:
                with torch.no_grad():
                    full = float(independent_loss(net(x), y, mask))
                curves.append(dict(member=member, step=step, train_loss=float(loss.detach()), full_train_loss=full))
            if step == steps:
                break
            optimizer.zero_grad();loss.backward();optimizer.step()
        for key, value in net.state_dict().items():
            checks.numeric(common.numpy_state(checkpoint["members"][member])[key], value.detach().numpy(), f"Member{member}/independently replayed final parameter/"+key, 1e-5)
        expected_state, actual_state = optimizer.state_dict(), checkpoint["optimizers"][member]
        checks.equal(expected_state["param_groups"], actual_state["param_groups"], f"Member{member}/complete Adam hyperparameters")
        checks.equal(set(expected_state["state"]), set(actual_state["state"]), f"Member{member}/all Adam parameter moments retained")
        for parameter, values in expected_state["state"].items():
            checks.equal(set(values), set(actual_state["state"][parameter]), f"Member{member}/Adam state keys")
            for key, value in values.items():
                checks.numeric(np.asarray(value), np.asarray(actual_state["state"][parameter][key]), f"Member{member}/independently replayed Adam/"+key, 1e-5)
    same(checkpoint["curves"], curves, checks, "Independent seeded full-batch learning curves", 1e-5)
    return curves


def ensemble_forward(checkpoint, normalized):
    """Independent NumPy SiLU forward and fixed nine-head decoder."""
    x = np.asarray(normalized, dtype=np.float32);outputs = []
    if x.ndim != 2 or x.shape[1] != 7 or not np.isfinite(x).all():
        raise EvidenceError("Finite normalized seven-input query matrix required")
    for member in checkpoint["members"]:
        state = common.numpy_state(member);value = x
        for layer in (0, 2, 4):
            value = value@state[f"{layer}.weight"].T+state[f"{layer}.bias"]
            if layer != 4:
                value = value/(1+np.exp(-value))
        outputs.append(value.astype(float))
    outputs = np.stack(outputs)
    result = []
    for index in range(len(x)):
        raw = outputs[:, index, :]
        probability = 1/(1+np.exp(-np.clip(raw[:, :5], -50, 50)))
        distance = .1*np.expm1(np.clip(raw[:, 5], 0., 6.));product = probability[:, 0]*probability[:, 3]
        result.append(dict(admission_probability=float(probability[:, 0].mean()), board_bounce_probability=float(probability[:, 1].mean()),
            whole_entry_probability=float(probability[:, 2].mean()), capture_probability=float(product.mean()), conditional_capture_probability=float(probability[:, 3].mean()),
            uncertainty=float(product.std(ddof=0)), retention_fraction=float(probability[:, 4].mean()), aperture_distance_m=float(distance.mean()),
            rebound_velocity_m_s=raw[:, 6:9].mean(axis=0).tolist(), rebound_basis="board_to_bucket_forward_lateral_vertical",
            member_raw_output=raw.tolist(), member_capture_probability=product.tolist(),
            ranking_score=float(product.mean()-.25*product.std(ddof=0)-.05*np.minimum(distance, 2.).mean())))
    return result


def candidates(context, anchor, bc_values, ppo_values, train_rows):
    if any(row.get("role") != "train" for row in train_rows):
        raise EvidenceError("Candidate source library must contain TRAIN only")
    scale = np.asarray([.04, .04, .05, 8., .12]);values = np.asarray(anchor["values"], dtype=float)
    bc, ppo = (np.asarray(action, dtype=float)-values for action in (bc_values, ppo_values));bc /= scale;ppo /= scale
    successful = [row for row in train_rows if policy_audit.captured(row)]
    if not successful:
        raise EvidenceError("Current TRAIN source has no captured candidate seed")
    source = min(successful, key=lambda row: (float(np.linalg.norm(np.asarray(row["bucket_xy_m"])-context["bucket_xy_m"])),
                  -int(row.get("route_quality") == 2), -row["learning_reward"], row["id"]))
    residuals = [bc, ppo, np.zeros(5), (np.asarray(source["values"])-source["effective_source_hint"]["values"])/scale]
    kinds = ["saved_bc", "saved_ppo", "shared_geometry_anchor", "nearest_current_train_capture"]
    for coordinate, delta in ((2, -.3), (2, .3), (3, -.35), (3, .35), (4, -.3), (4, .3), (0, -.3), (0, .3)):
        value = bc.copy();value[coordinate] += delta;residuals.append(value);kinds.append(f"bc_axis_{coordinate}_{delta}")
    seed = int(digest(dict(context_id=context["id"], seed=20261621))[:16], 16)
    generator = np.random.default_rng(seed)
    for _ in range(4):
        residuals.append(bc+generator.uniform(-.35, .35, 5));kinds.append("seeded_bc_joint_neighborhood")
    result = []
    for index, (residual, kind) in enumerate(zip(residuals, kinds)):
        residual = np.clip(residual, -1., 1.)
        result.append(dict(id=f"candidate-{index:02d}", values=(values+residual*scale).tolist(), normalized_residual=residual.tolist(),
                           source_kind=kind, source_training_ids=[source["id"]] if index == 3 else []))
    return result


def decode_raw(raw):
    """Reconstruct every rank scalar directly from saved per-member outputs."""
    raw = np.asarray(raw, dtype=float)
    if raw.shape != (5, 9) or not np.isfinite(raw).all():
        raise EvidenceError("All five finite nine-head member predictions required")
    probability = 1/(1+np.exp(-np.clip(raw[:, :5], -50, 50)))
    distance = .1*np.expm1(np.clip(raw[:, 5], 0., 6.));product = probability[:, 0]*probability[:, 3]
    return dict(admission_probability=float(probability[:, 0].mean()), board_bounce_probability=float(probability[:, 1].mean()),
        whole_entry_probability=float(probability[:, 2].mean()), capture_probability=float(product.mean()), conditional_capture_probability=float(probability[:, 3].mean()),
        uncertainty=float(product.std(ddof=0)), retention_fraction=float(probability[:, 4].mean()), aperture_distance_m=float(distance.mean()),
        rebound_velocity_m_s=raw[:, 6:9].mean(axis=0).tolist(), rebound_basis="board_to_bucket_forward_lateral_vertical",
        member_raw_output=raw.tolist(), member_capture_probability=product.tolist(),
        ranking_score=float(product.mean()-.25*product.std(ddof=0)-.05*np.minimum(distance, 2.).mean()))


def audit_rankings(directory, logs, contexts, baseline_predictions, dataset_rows, checkpoint, prior, home, checks):
    from .calibrated_admission import verify_resolution, verify_connection
    checks.equal(len(logs), 84, "Exactly one complete model-only candidate ledger per84 fresh layouts")
    checks.equal([entry["context_id"] for entry in logs], [context["id"] for context in contexts], "Model-ranking order follows literal split manifest")
    selected = {}
    for log, context in zip(logs, contexts):
        label = context["id"];base = baseline_predictions[label]
        checks.equal((log["role"], log["bucket_xy_m"]), (context["split"], context["bucket_xy_m"]), label+"/exact fresh query")
        same(log["context_metadata"], base["bc"]["metadata"], checks, label+"/same preprediction geometry as unchanged baselines", 0)
        same(log["observation"], base["bc"]["observation"], checks, label+"/same geometry-only observation", 0)
        same(log["anchor"], base["bc"]["anchor"], checks, label+"/same effective anchor", 0)
        expected_candidates = candidates(context, log["anchor"], base["bc"]["values"], base["ppo"]["values"], dataset_rows)
        same(log["candidates"], expected_candidates, checks, label+"/literal sixteen model-only candidates and TRAIN provenance", 1e-12)
        normalized = [((np.asarray([*item["values"], *context["bucket_xy_m"]])-INPUT_CENTER)/INPUT_SCALE).tolist() for item in expected_candidates]
        expected_predictions = ensemble_forward(checkpoint, normalized)
        checks.equal(len(log["ranking"]["predictions"]), 16, label+"/all member predictions retained")
        decoded = []
        for index, prediction in enumerate(log["ranking"]["predictions"]):
            # Independent network values permit usual CPU BLAS rounding; every
            # reported ranking scalar is then recomputed exactly from those
            # tightly checked saved raw member outputs.
            same(prediction, expected_predictions[index], checks, label+f"/independent model inference/{index}", 3e-5)
            reconstructed = decode_raw(prediction["member_raw_output"])
            same(prediction, reconstructed, checks, label+f"/literal decoder and uncertainty score/{index}", 1e-12)
            decoded.append(reconstructed)
        order = sorted(range(16), key=lambda index: (-decoded[index]["ranking_score"], expected_candidates[index]["id"]))
        checks.equal(log["ranking"]["order"], order, label+"/independent score/tie ordering")
        checks.equal((log["native_queries"], log["model_candidates"], log["maximum_geometry_checks"], log["ranking"]["candidate_count"], log["ranking"]["ball_flight_search_at_inference"]),
                     (0, 16, 4, 16, False), label+"/no native candidate search or hidden probes")
        resolution_name = log["context_metadata"]["context_resolution_file"]
        resolution_path = contained(directory/"native", resolution_name)
        checks.equal(sha(resolution_path), log["context_metadata"]["context_resolution_sha256"], label+"/frozen preprediction geometry resolution")
        resolution = read(resolution_path);resolution.update(file=resolution_name, sha256=sha(resolution_path), cache_hit=True)
        checks.require(verify_resolution(resolution, prior["records"], home), label+"/unchanged source/home strict geometry")
        checks.require(1 <= len(log["checks"]) <= 4, label+"/at most four geometry checks")
        first_admitted = None
        for index, record in enumerate(log["checks"]):
            candidate_index = order[index];item = expected_candidates[candidate_index]
            checks.equal((record["candidate_index"], record["candidate_id"]), (candidate_index, item["id"]), label+"/only model top4 screened in rank order")
            path = contained(directory/"native", record["connection_file"])
            checks.equal(sha(path), record["connection_sha256"], label+"/exact geometry candidate proof bytes")
            connection = read(path)
            checks.require(verify_connection(connection, resolution, home, item["values"]), label+"/same candidate unchanged admission constraints")
            checks.equal((record["passed"], record["reason"]), (bool(connection["passed"]), connection.get("reason")), label+"/actual strict candidate result")
            if record["passed"]:
                checks.require(first_admitted is None and index == len(log["checks"])-1, label+"/stop at first admitted, no later cherry-picking")
                first_admitted = candidate_index
        if first_admitted is None:
            checks.equal(len(log["checks"]), 4, label+"/all four prescribed geometry failures retained")
        chosen = order[0] if first_admitted is None else first_admitted
        checks.equal((log["selected_index"], log["selected_values"], log["geometry_checks"], log["all_geometry_checks_rejected"]),
                     (chosen, expected_candidates[chosen]["values"], len(log["checks"]), first_admitted is None), label+"/first-admitted or exact model-top1 rejected fallback")
        selected[label] = dict(values=expected_candidates[chosen]["values"], candidate_index=chosen, candidate_id=expected_candidates[chosen]["id"],
                               ranking_sha256=digest(log), geometry_checks=len(log["checks"]), admitted=first_admitted is not None)
    return selected


def audit_prediction_plan(study, packet, spec, parent_study, parent_report, train_rows, checkpoint, checks):
    """Reconstruct immutable actors, all16 model candidates and all252 forecasts."""
    checks.equal((packet["version"], packet["contract_fingerprint"]), (STUDY_VERSION, spec["fingerprint"]), "Prediction packet exact model study")
    checks.require(packet["frozen_before_any_native_outcome"] is True and packet["validation_selects_model"] is False
                   and packet["native_planning_queries"] == 0, "All252 forecasts fixed before any evaluation; no validation selection/native search")
    checks.equal(sha(study/"predictions/rankings.json"), packet["rankings_sha256"], "All model candidates and rankings frozen as bytes")
    requests = packet["requests"];literal_budget_checks(requests, checks)
    expected_roles = ["positive_control"]*3+["negative_control"]*3+["validation"]*72+["test"]*180+["positive_control"]*3+["negative_control"]*3
    checks.equal([row["role"] for row in requests], expected_roles, "Literal264 request execution order")
    actors, artifacts = {}, {}
    for method in ("bc", "ppo"):
        original = parent_report["artifacts"]["bc"] if method == "bc" else parent_report["selected"]
        path = study/"model"/(method+".pt")
        checks.equal(sha(path), original["sha256"], "Exact unchanged copied baseline model/"+method)
        state = common.load_checkpoint(path, checks, "Saved "+method)
        checks.equal(state["contract_fingerprint"], PARENT_FP, "Baseline remains original policy study/"+method)
        artifacts[method] = dict(path=f"model/{method}.pt", sha256=original["sha256"], samples=original["samples"])
        actors[method] = state["policy"]
    artifacts["ensemble"] = packet["ensemble_artifact"]
    checks.equal(artifacts["ensemble"]["path"], "model/ensemble.pt", "Literal final fixed ensemble artifact path")
    checks.equal(sha(study/"model/ensemble.pt"), artifacts["ensemble"]["sha256"], "Frozen final ensemble bytes before evaluation")
    contexts = [context for role in ("validation", "test") for context in spec["splits"][role]]
    metadata_fields = ("source_hint", "effective_source_hint", "context_resolution_file", "context_resolution_sha256", "context_home_sha256",
                       "context_layout_eligible", "context_anchor_admitted", "context_hard_reject")
    base = {}
    expected_ids = []
    for role in ("validation", "test"):
        for method in METHODS:
            expected_ids.extend(f"physics-ensemble-{role}-{method}-{context['id']}" for context in spec["splits"][role])
    evaluated = [row for row in requests if row["role"] in ("validation", "test")]
    checks.equal([row["id"] for row in evaluated], expected_ids, "One prespecified shot per fresh layout and method")
    by_id = {row["id"]: row for row in requests}
    for context in contexts:
        ident = context["id"];base[ident] = {}
        for method in ("bc", "ppo"):
            row = by_id[f"physics-ensemble-{context['split']}-{method}-{ident}"]
            label = row["id"];anchor = row["effective_source_hint"]
            obs = calibrated.expected_observation(context["bucket_xy_m"], anchor)
            checks.numeric(row["observation"], obs, label+"/independent geometry-only observation", 0)
            mean = common.network_forward(actors[method], obs)
            residual = np.tanh(mean.astype(float));action = np.asarray(anchor["values"])+residual*np.asarray([.04, .04, .05, 8., .12])
            checks.numeric(row["latent_mean"], mean, label+"/unmodified saved actor prediction", 2e-6)
            checks.numeric(row["normalized_residual"], residual, label+"/saved actor residual", 2e-6)
            checks.numeric(row["values"], action, label+"/direct actor one-shot decision", 2e-6)
            metadata = {key: copy.deepcopy(row[key]) for key in metadata_fields}
            base[ident][method] = dict(metadata=metadata, observation=row["observation"], anchor=anchor, values=row["values"])
            checks.equal(row["frozen_artifact_sha256"], artifacts[method]["sha256"], label+"/saved checkpoint provenance")
        same(base[ident]["bc"]["metadata"], base[ident]["ppo"]["metadata"], checks, ident+"/common geometry-only setup across direct baselines", 0)
    selected = audit_rankings(study, read(study/"predictions/rankings.json"), contexts, base, train_rows, checkpoint,
                              read(study/"prior.json"), spec["identity"]["setup"]["home"], checks)
    for context in contexts:
        ident = context["id"];row = by_id[f"physics-ensemble-{context['split']}-ensemble-{ident}"];chosen = selected[ident]
        checks.equal(row["values"], chosen["values"], ident+"/single selected model-ranked exact action")
        for key in ("candidate_index", "candidate_id", "ranking_sha256", "geometry_checks"):
            checks.equal(row[key], chosen[key], ident+"/selected candidate identity/"+key)
        checks.equal((row["model_candidates"], row["native_planning_queries"]), (16, 0), ident+"/inference search budget disclosed")
        checks.equal(row["frozen_artifact_sha256"], artifacts["ensemble"]["sha256"], ident+"/one frozen ensemble checkpoint")
    for row in evaluated:
        checks.require(row["selection_or_learning"] is False and row["ball_flight_search_at_inference"] is False
                       and row["geometry_search_at_inference"] is True, row["id"]+"/fixed preoutcome geometry-only selection")
        checks.equal(row["values"], row["command_values"], row["id"]+"/no postprediction action substitution")
        checks.equal(row["perturbation_mixed_units"], [0.]*5, row["id"]+"/nominal no-noise study")
        checks.equal(row["prediction_model_sha256"], artifacts["ensemble"]["sha256"], row["id"]+"/forecast from fixed ensemble checkpoint")
        normalized = ((np.asarray([*row["values"], *row["bucket_xy_m"]])-INPUT_CENTER)/INPUT_SCALE).tolist()
        same(row["model_prediction"], ensemble_forward(checkpoint, [normalized])[0], checks, row["id"]+"/independent preoutcome physical forecast", 3e-5)
        same(row["model_prediction"], decode_raw(row["model_prediction"]["member_raw_output"]), checks, row["id"]+"/exact raw-member forecast decoder", 1e-12)
    controls = [next(row for row in replay.read_records(parent_study/"rows.jsonl") if row["phase"] == "controls_pre" and row["role"] == role and row.get("repeat") == 0)
                for role in ("positive_control", "negative_control")]
    for phase, group in (("controls_pre", requests[:6]), ("controls_post", requests[-6:])):
        for source, role in zip(controls, ("positive_control", "negative_control")):
            cohort = [row for row in group if row["role"] == role]
            checks.equal(len(cohort), 3, phase+"/three unchanged positive/native-miss requests")
            for repeat, row in enumerate(cohort):
                checks.equal((row["id"], row["phase"], row["repeat"]), (f"physics-ensemble-{phase}-{role}-r{repeat}", phase, repeat), "Exact matched control identity")
                checks.equal((row["control_source_id"], row["control_source_sha256"], row["frozen_artifact_sha256"]), (source["id"], digest(source), digest(source)), "Exact control provenance")
                checks.equal((row["bucket_xy_m"], row["values"]), (source["bucket_xy_m"], source["values"]), "Unchanged control command and bucket")
    return artifacts, selected


def statistics(rows):
    captures = sum(policy_audit.captured(row) for row in rows);native = sum(row.get("native_release") is True for row in rows)
    invalid = sum(row.get("native_release") is True and (bool(row.get("invalid")) or bool(row.get("invalid_physics"))) for row in rows)
    return dict(requested=len(rows), native_releases=native, captures=captures,
        clean=sum(policy_audit.captured(row) and row.get("route_quality") == 2 for row in rows), rejected=len(rows)-native,
        native_misses=native-captures-invalid, invalid=invalid)


def control_gate(rows):
    positive = [row for row in rows if row["role"] == "positive_control"]
    negative = [row for row in rows if row["role"] == "negative_control"]
    good = (len(rows) == 6 and len(positive) == len(negative) == 3 and all(policy_audit.captured(row) for row in positive)
            and all(row.get("native_release") and not policy_audit.captured(row) and not row.get("whole_ball_entry") for row in negative)
            and all(not row.get("invalid") and not row.get("invalid_physics") and row.get("scalar_score_audit_passed") is True for row in rows))
    return dict(status="passed" if good else "failed", counts=statistics(rows))


def prediction_metrics(checkpoint, dataset, split):
    import torch
    normalized = torch.tensor(np.asarray([row["x"] for row in dataset]), dtype=torch.float32)
    outputs = []
    for weights in checkpoint["members"]:
        net = independent_network();net.load_state_dict(weights);net.eval()
        with torch.no_grad():
            outputs.append(net(normalized).numpy().astype(float))
    outputs = np.stack(outputs)
    predictions = [decode_raw(outputs[:, index, :]) for index in range(len(dataset))]
    targets = np.asarray([row["y"] for row in dataset]);masks = np.asarray([row["mask"] for row in dataset], dtype=bool)
    names = ("native_execution", "board_bounce", "whole_ball_entry", "capture", "retention_fraction", "log_first_flight_opening_distance", "rebound_forward_m_s", "rebound_lateral_m_s", "rebound_vertical_m_s")
    fields = ("admission_probability", "board_bounce_probability", "whole_entry_probability", "conditional_capture_probability", "retention_fraction", "aperture_distance_m")
    result = []
    for head, name in enumerate(names):
        support = masks[:, head];count = int(support.sum())
        if not count:
            result.append(dict(split=split, target=name, support=0));continue
        actual = targets[support, head]
        predicted = np.asarray([prediction[fields[head]] if head < 6 else prediction["rebound_velocity_m_s"][head-6] for prediction in predictions])[support]
        if head == 5:
            actual = .1*np.expm1(actual)
        error = predicted-actual
        row = dict(split=split, target=name, support=count, mae=float(np.abs(error).mean()), rmse=float(np.sqrt((error**2).mean())),
                   unit="m" if head == 5 else "m/s" if head >= 6 else "fraction")
        if head < 4:
            row.update(brier=float((error**2).mean()), accuracy=float(((predicted >= .5) == (actual >= .5)).mean()))
        result.append(row)
    use = np.asarray([not row.get("physics_invalid") for row in dataset], dtype=bool)
    if use.any():
        actual = targets[:, 0]*targets[:, 3];predicted = np.asarray([prediction["capture_probability"] for prediction in predictions])
        error = (predicted-actual)[use]
        result.append(dict(split=split, target="requested_capture_uncalibrated_score", support=int(use.sum()),
            brier=float((error**2).mean()), mae=float(np.abs(error).mean()), rmse=float(np.sqrt((error**2).mean())), unit="fraction"))
    return result


def audit_report(report, rows, packet, artifacts, selected, checkpoint, evaluation_targets, checks):
    same(report["counts"], statistics(rows), checks, "Complete264 request accounting")
    gates = {key: control_gate([row for row in rows if row["phase"] == phase]) for key, phase in (("pre", "controls_pre"), ("post", "controls_post"))}
    same(report["control_gates"], gates, checks, "Before/after unchanged calibrated controls")
    checks.require(all(gate["status"] == "passed" for gate in gates.values()), "Both matched native control blocks passed")
    same(report["artifacts"], artifacts, checks, "Published immutable models exactly match inference artifacts")
    comparisons, positions, metrics = [], [], []
    for role in ("validation", "test"):
        for method in METHODS:
            cohort = [row for row in rows if row["role"] == role and row["method"] == method]
            comparisons.append(dict(role=role, method=method, counts=statistics(cohort), artifact_sha256=artifacts[method]["sha256"],
                                    mean_reward=math.fsum(row["learning_reward"] for row in cohort)/len(cohort)))
            if role == "test":
                for row in cohort:
                    captured = policy_audit.captured(row)
                    positions.append(dict(context_id=row["context_id"], method=method, bucket_xy_m=row["bucket_xy_m"], values=row["values"], record_id=row["id"],
                        board_bucket_distance_m=float(np.linalg.norm(np.asarray(row["values"][:2])-row["bucket_xy_m"])),
                        admission=bool(row.get("geometry_eligible", row["native_release"])), artifact_sha256=row["frozen_artifact_sha256"], model_prediction=row["model_prediction"],
                        actual=dict(native_release=row["native_release"], capture=captured, clean=captured and row["route_quality"] == 2,
                        invalid=row.get("invalid"), invalid_physics=bool(row.get("invalid_physics")), whole_entry=bool(row.get("whole_ball_entry")),
                        retention_s=float(row.get("max_retention_s", 0.)), entry_class="clean_capture" if captured and row["route_quality"] == 2 else "captured_nonclean" if captured else "no_capture")))
        metrics.extend(prediction_metrics(checkpoint, [target for target in evaluation_targets if target["role"] == role], role))
    same(report["comparisons"], comparisons, checks, "All requested-shot comparisons, including admission failures")
    same(report["test_comparison"], [item for item in comparisons if item["role"] == "test"], checks, "TEST comparison never mixed with validation outcomes")
    same(report["test_positions"], positions, checks, "All180 nominal TEST forecasts/outcomes, no replacement")
    same(report["prediction_metrics"], metrics, checks, "Held-out supported-target forecast errors separate from measured capture", 1e-5)
    checks.equal(report["geometry_planning_checks"], sum(value["geometry_checks"] for value in selected.values()), "Extra model-ranking geometry cost disclosed")
    checks.equal(report["gate"], dict(status="passed", complete=True, controls_passed=True, predictions_frozen_before_any_evaluation=True, no_test_flight_search=True,
                 scope="Protocol completion only, not all-table or physical reliability"), "Completion cannot imply success-rate/convergence qualification")
    checks.equal((report["cpu_fit"], report["native_substeps"], report["native_requests"], report["train_requests"], report["validation_selects_model"]),
                 (True, 64, 264, 0, False), "No new TRAIN or postvalidation fit/selection")
    checks.require(math.isfinite(report["acquired_elapsed_s"]) and 0 <= report["acquired_elapsed_s"] <= 7200, "Bounded acquired worker time, no timing-winner claim")
    same(report["phase_counts"], {phase: statistics([row for row in rows if row["phase"] == phase]) for phase in ("controls_pre", "validation", "test", "controls_post")}, checks, "All phase budgets, controls separate")
    return comparisons, gates


def audit_trace_examples(study, report, rows, checks):
    expected = []
    for method in METHODS:
        for success in (True, False):
            cohort = [row for row in rows if row["role"] == "test" and row["method"] == method and row.get("native_release")
                      and policy_audit.captured(row) == success and row.get("trajectory_m")]
            if cohort:
                row = cohort[0]
                expected.append(dict(id=row["id"], record_id=row["id"], context_id=row["context_id"], method=method,
                    bucket_xy_m=row["bucket_xy_m"], values=row["values"], board_center_m=row["board_center_m"], board_normal=row["board_normal"],
                    position_m=row["trajectory_m"], sample_dt_s=row["trajectory_dt_s"], capture_time_s=row.get("capture_time_s"),
                    capture=policy_audit.captured(row), clean=policy_audit.captured(row) and row["route_quality"] == 2,
                    whole_ball_entry=row.get("whole_ball_entry"), max_retention_s=row.get("max_retention_s"), invalid=row.get("invalid"), invalid_physics=row.get("invalid_physics"),
                    native_cache_sha256=row["native_cache_sha256"], native_cache_row=row["native_cache_row"],
                    selection_scope="First actual TEST native flight for each method/outcome; measured trace, not model trajectory"))
                with np.load(contained(study/"native", row["native_cache_file"]), allow_pickle=False) as archive:
                    checks.numeric(row["trajectory_m"], archive["position_m"][256::256, row["native_cache_row"]], row["id"]+"/displayed measured native positions", 0)
                checks.numeric(row["trajectory_dt_s"], 1/60, row["id"]+"/displayed measured sample cadence", 0)
    same(report["trace_examples"], expected, checks, "Literal first measured method/outcome examples, never model trajectories", 0)


def source_identity(source, native_source, spec, checks):
    source, native_source = Path(source).resolve(), Path(native_source).resolve()
    expected = {"sircus/mjlab/physics_ensemble_"+name+".py" for name in ("contract", "model", "study", "verify")}
    checks.equal(set(spec["source_sha256"]), expected, "Complete new contract/model/worker/checker source snapshot")
    for root, hashes, label in ((source, spec["source_sha256"], "New ensemble source"), (native_source, spec["native_source_sha256"], "Unchanged original f00 native source")):
        for name, value in hashes.items():
            checks.equal(sha(contained(root, name)), value, label+"/"+name)
    dependencies = {}
    for module in tuple(sys.modules.values()):
        name, filename = getattr(module, "__name__", ""), getattr(module, "__file__", None)
        if not filename or not (name.startswith("sircus.mjlab.") or name.startswith("sircus.isaac.")):
            continue
        path = Path(filename).resolve();root = next((candidate for candidate in (source, native_source) if path.is_relative_to(candidate)), None)
        checks.require(root is not None, "Only explicit new-checker/original-native roots imported/"+name)
        relative = str(path.relative_to(root))
        if len(Path(relative).parts) != 3 or path.suffix != ".py":
            continue
        hashes = spec["source_sha256"] if relative in expected else spec["native_source_sha256"]
        if relative.startswith("sircus/isaac/") and relative not in hashes:
            continue
        checks.require(relative in hashes, "Imported actual core dependency declared/"+relative)
        checks.equal(sha(path), hashes[relative], "Actual independent core/oracle dependency/"+relative)
        dependencies[relative] = sha(path)
    checks.require("sircus/mjlab/calibrated_score_oracle.py" in dependencies, "Independent dense oracle actually imported")
    checks.equal({name: importlib.metadata.version(name) for name in spec["packages"]}, spec["packages"], "Exact CPU/native package set including SciPy")
    return dict(version=VERSION, auditor_sha256=sha(Path(__file__)), dependencies_sha256=dependencies,
                producer_source_sha256=spec["source_sha256"], native_source_sha256=spec["native_source_sha256"])


def audit(study, source, native_source, assets, *, parent_study, parent_audit):
    import torch
    started, checks = time.monotonic(), AuditChecks()
    study, source, native_source, assets, parent_study, parent_audit = map(Path, (study, source, native_source, assets, parent_study, parent_audit))
    parent, parent_report, train_rows, parent_manifest, parent_rows = parent_inputs(parent_study, parent_audit, checks)
    manifest = replay.check_manifest(study, checks)
    spec, report = read(study/"contract.json"), read(study/"report.json")
    unsigned = dict(spec);fp = unsigned.pop("fingerprint", None)
    checks.equal(digest(unsigned), fp, "Full prefit/preprediction ensemble protocol fingerprint")
    checks.equal((spec["version"], report["version"]), (STUDY_VERSION, STUDY_VERSION), "Explicit new offline physics-ensemble study")
    checks.equal((report["status"], report["phase"]), ("completed", "completed"), "Complete finite264-request native evaluation")
    checks.equal(report["contract_fingerprint"], fp, "Native results exact frozen model protocol")
    same(report["evidence_sha256"], manifest, checks, "Report binds every model, prediction and native observation")
    same(report["protocol"], spec["protocol"], checks, "Reported protocol never changed after evaluation")
    audit_protocol(spec["protocol"], checks)
    for key, value in dict(substeps=64, maximum_requests=264, cpu_fit=True, native_gpu="GPU-ab1950b0-01dd-0fab-407f-2d17936b27f4",
                           maximum_acquired_s=7200, maximum_queue_s=86400, validation_selects_model=False, training_native_requests=0).items():
        checks.equal(spec[key], value, "Literal CPU preparation/4090 native budget/"+key)
    same(spec["identity"], parent["identity"], checks, "Original calibration/home/scorer/physics identity unchanged")
    same(spec["scene"], parent["scene"], checks, "Original scene and explicit provisional dimensions preserved")
    same(spec["native_source_sha256"], parent["source_sha256"], checks, "Original executed native dependency source preserved")
    same(spec["packages"], parent["packages"], checks, "Original complete native/CPU package identity")
    checks.equal(spec["parent_contract_fingerprint"], PARENT_FP, "Only exact authenticated parent supplies labels/baselines")
    original_models = dict(bc=parent_report["artifacts"]["bc"], ppo=parent_report["selected"])
    bindings = dict(parent_contract_fingerprint=PARENT_FP, parent_report_sha256=PARENT_REPORT_SHA, parent_manifest_sha256=PARENT_MANIFEST_SHA,
        parent_audit_sha256=PARENT_AUDIT_SHA, parent_rows_sha256=parent_manifest["rows.jsonl"], train_rows_sha256=digest(train_rows),
        train_record_sha256={row["id"]: digest(row) for row in train_rows}, baseline_artifacts=original_models)
    same(spec["parent_bindings"], bindings, checks, "Complete exact1024 TRAIN-only source and unchanged baseline provenance")
    same(read(study/"source-bindings.json"), bindings, checks, "Frozen source binding artifact")
    checks.equal(spec["training_rows_sha256"], digest(train_rows), "No historical6144/held-out/control labels added to training")
    for name in ("calibration.json", "verification.json", "setup-home.json", "prior.json"):
        checks.equal(sha(study/name), parent_manifest[name], "Copied unchanged original native input/"+name)
    split_record = expected_splits(parent)
    same((spec["splits"], spec["excluded"]), (split_record["splits"], split_record["excluded"]), checks, "Independent seeded24/60 splits and declared old-layout exclusion", 0)
    same(read(study/"splits.json"), spec["splits"], checks, "Complete copied fresh split manifest", 0)
    same(read(study/"exclusions.json"), spec["excluded"], checks, "Complete copied historical exclusion manifest", 0)
    dataset = reconstruct_dataset(parent_study, train_rows, parent, parent_manifest, checks)
    same(read(study/"training-dataset.json"), dataset, checks, "Every1024 target/mask independently measured from TRAIN source", 1e-9)
    checks.equal(spec["dataset_sha256"], sha(study/"training-dataset.json"), "Frozen TRAIN target file bytes")
    checkpoint = torch.load(study/"model/ensemble.pt", map_location="cpu", weights_only=True)
    # Use exact saved JSON scalar serialization after the independently checked
    # target reconstruction, avoiding meaningless log1p last-bit re-encoding.
    saved_dataset = read(study/"training-dataset.json")
    curves = replay_fit(saved_dataset, checkpoint, fp, checks)
    inputs = ["board_x_m", "board_y_m", "board_height_m", "tilt_deg", "gap_m", "bucket_x_m", "bucket_y_m"]
    targets = ["native_execution", "board_bounce", "whole_ball_entry", "capture", "retention_fraction", "log_first_flight_opening_distance", "rebound_forward_m_s", "rebound_lateral_m_s", "rebound_vertical_m_s"]
    checks.equal((checkpoint["input_features"], checkpoint["target_features"]), (inputs, targets), "Explicit state-based input/target meanings")
    fit = read(study/"model/fit.json")
    checks.equal(fit["sha256"], sha(study/"model/ensemble.pt"), "Fitted model artifact bytes")
    checks.equal(fit["dataset_sha256"], digest(saved_dataset), "Fitted checkpoint only consumes frozen TRAIN dataset")
    same(fit["curves"], checkpoint["curves"], checks, "Published curves preserve actual model fitting")
    packet = read(study/"prediction-plan.json")
    expected_artifact = dict(fit, path="model/ensemble.pt", sha256=sha(study/"model/ensemble.pt"))
    same(packet["ensemble_artifact"], expected_artifact, checks, "Only fixed final ensemble selected, no validation winner")
    checks.equal(report["fit_status"], "model_fit_completed", "Real model fitting complete before queued/native phases")
    expected_training = dict(status="completed", ensemble_members=5, fit_device="cpu", fit_steps=600,
        selection_basis="Prespecified final fixed ensemble; no validation tuning", curves=checkpoint["curves"], artifact=expected_artifact,
        input_features=inputs, target_features=targets)
    same(report["training"], expected_training, checks, "Explicit real ensemble training record")
    expected_dataset = dict(train_requests=1024, train_native_releases=887, train_labels=1024, train_layouts=len({row["context_id"] for row in train_rows}),
        source_version=parent["version"], source_contract_fingerprint=PARENT_FP, source_report_sha256=PARENT_REPORT_SHA, source_audit_sha256=PARENT_AUDIT_SHA,
        excluded_prior_layouts=len(spec["excluded"]), split_hashes={name:sha(study/name) for name in ("splits.json", "exclusions.json")},
        target_support=np.asarray([row["mask"] for row in saved_dataset]).sum(axis=0).astype(int).tolist(),
        labels_scope="Only1024 current parent TRAIN requests; no historical6144, held-out, control or diagnosis labels fitted",
        train_split_scope="Existing35 TRAIN layouts; new24VAL60TEST excluded from all prior layouts")
    same(report["dataset"], expected_dataset, checks, "Actual TRAIN costs/masked target support and historical exclusion disclosed")
    checks.equal(report["prediction_plan_sha256"], sha(study/"prediction-plan.json"), "Preoutcome whole-plan byte identity")
    artifacts, selected = audit_prediction_plan(study, packet, spec, parent_study, parent_report, train_rows, checkpoint, checks)
    same(report["rankings_sha256"], packet["rankings_sha256"], checks, "Preoutcome complete rankings byte identity")
    rows = replay.read_records(study/"records.jsonl");literal_budget_checks(rows, checks)
    checks.equal([row["id"] for row in rows], [row["id"] for row in packet["requests"]], "Every preoutcome requested decision spent once in exact order")
    for row, request in zip(rows, packet["requests"]):
        for key, value in request.items():
            same(row[key], value, checks, row["id"]+"/request frozen before any outcome/"+key, 0)
        checks.equal(row["action_substituted"], False, row["id"]+"/no later shot replacement")
        checks.equal(row["adapter_directory"], "native", row["id"]+"/unchanged native64 engine")
        if row["method"] == "ensemble" and row["role"] in ("validation", "test"):
            checks.equal(row["native_release"], selected[row["context_id"]]["admitted"], row["id"]+"/selected exact candidate admission preserved")
        if row["native_release"]:
            checks.equal(row["contact_protocol"], dict(substeps=64, board_mode="articulated", gravity_mode="passive_only"), row["id"]+"/same allocated held-endpoint64 physics")
            checks.equal(row["scoring_version"], "sircus-calibrated-native-capture-v2", row["id"]+"/same capture/retention semantics")
            checks.require(row["scalar_score_audit_passed"] is True, row["id"]+"/online oracle agreement")
            checks.equal((row["effective_options_file"], row["effective_options_sha256"]), ("effective-options.json", sha(study/"native/effective-options.json")), row["id"]+"/observed allocated options bound to trial")
    calibrated.check_rows(rows, checks)
    setup = calibrated.check_setup_evidence(study, rows, read(study/"prior.json"), spec["identity"]["setup"]["home"], checks)
    calibration = replay.check_calibration(study, spec["identity"]["calibration"], checks)
    from .learning_solve import required_asset_paths
    actual_assets = {name: sha(contained(assets.parent, name)) for name in required_asset_paths(assets)}
    checks.equal(actual_assets, spec["identity"]["asset_sha256"], "Actual original collision assets/robot models")
    allocated = policy_audit.audit_native_options(study/"native", spec["scene"], calibration, assets, checks)
    reference = spec["native_reference"]
    checks.equal((reference["parent_options_file"], reference["parent_options_sha256"]), ("native/effective-options.json", parent_manifest["native/effective-options.json"]), "Original actual engine option source")
    checks.equal(sha(parent_study/"native/effective-options.json"), reference["parent_options_sha256"], "Parent allocated options bytes preserved")
    same(reference["allocated_options"], allocated, checks, "Same exact native/Warp options as authenticated parent")
    same(reference["public"], dict(effective_native_options={key: allocated["native"][key] for key in ("timestep", "solver", "cone", "integrator", "iterations", "tolerance", "gravity")}), checks, "Legacy native option guard supplied explicitly, not bypassed")
    traces = calibrated.rescore_rows(study/"native", rows, spec["scene"], calibration, checks, "Fresh ensemble-study native64", spec["identity"]["trace_schema"], assets=assets)
    evaluated = [row for row in rows if row["role"] in ("validation", "test")]
    evaluation_targets = reconstruct_dataset(study, evaluated, spec, manifest, checks, require_train=False)
    same(read(study/"evaluation-targets.json"), evaluation_targets, checks, "Held-out targets derived only after frozen evaluation; not training inputs", 1e-9)
    comparisons, gates = audit_report(report, rows, packet, artifacts, selected, checkpoint, evaluation_targets, checks)
    audit_trace_examples(study, report, rows, checks)
    identity = source_identity(source, native_source, spec, checks)
    return dict(version=VERSION, status="passed", contract_fingerprint=fp, checks_count=checks.count, checks=checks.entries, failures=[],
        report_sha256=sha(study/"report.json"), evidence_manifest_sha256=sha(study/"evidence-manifest.json"), auditor_sha256=sha(Path(__file__)), auditor_identity=identity,
        producer_source_sha256=spec["source_sha256"], native_source_sha256=spec["native_source_sha256"], requested_trials=len(rows), native_releases=len(traces),
        source_parent=dict(contract_fingerprint=PARENT_FP, report_sha256=PARENT_REPORT_SHA, manifest_sha256=PARENT_MANIFEST_SHA, original_audit_sha256=PARENT_AUDIT_SHA),
        training_requests=1024, new_training_requests=0, training_masks=np.asarray([row["mask"] for row in saved_dataset]).sum(axis=0).astype(int).tolist(),
        control_gates=gates, comparisons=comparisons, setup_evidence=setup, elapsed_s=time.monotonic()-started, maximum_numeric_error=checks.numeric_error,
        limitations=["Only1024 existing current TRAIN requests fit the model; saved BC/PPO used additional historical data, so acquisition budgets are unequal.",
                     "Ensemble ranks16 model candidates and checks at most4 geometry proofs; direct saved actors use less inference work. This is a system comparison, not pure architecture superiority.",
                     "All252 forecasts were frozen before validation/test; no validation tuning or native candidate-flight search.",
                     "One nominal shot per method at60 fresh TEST layouts; no repeatability/noise,128convergence, perception, grasp/transit, all-table or physical validation.",
                     "Model disagreement is a heuristic, not calibrated probability or an independent-seed policy reliability estimate. Missing physical targets stay masked."])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("study", "source", "native-source", "assets", "parent-study", "parent-audit", "output"):
        parser.add_argument("--"+name, type=Path, required=True)
    args = parser.parse_args()
    if any(args.output.resolve().is_relative_to(path.resolve()) for path in (args.study, args.parent_study)):
        raise ValueError("Write the independent proof outside both immutable studies")
    try:
        result = audit(args.study, args.source, args.native_source, args.assets, parent_study=args.parent_study, parent_audit=args.parent_audit)
    except Exception as error:
        args.output.write_text(json.dumps(dict(version=VERSION, status="failed", failures=[type(error).__name__+": "+str(error)]), indent=2, allow_nan=False)+"\n")
        raise
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    print(json.dumps({key: result[key] for key in ("status", "checks_count", "requested_trials", "native_releases")}))


if __name__ == "__main__":
    main()
