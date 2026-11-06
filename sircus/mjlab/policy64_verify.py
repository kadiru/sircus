"""Independent CPU audit of saved-data bootstrap and fixed64 policy evaluation.

This new contract is specific to64 substeps. Historical128 failures remain
historical evidence, not a gate silently removed from an old study. Training
and every frozen prediction are reconstructed without held-out flight labels.
"""
from __future__ import annotations

import argparse
from collections import Counter
import copy
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import numpy as np

from . import benchmark_verify as common
from . import calibrated_benchmark_verify as calibrated
from .benchmark_verify import EvidenceError, contained, digest, read, sha
from . import replay650_verify as replay

VERSION = "sircus-policy64-independent-audit-v1"
STUDY_VERSION = "sircus-policy64-bc-ppo-v1"
SOURCE_FP = "f96e6c9e0878564ef88c626cc2cbc46cc48a83f803a16300b3d1945d3fd62f11"
SOURCE_AUDIT_SHA = "8df7b8f3b48cfc9c1c41f33f02ea0b68ca828909e2c8774d3cbabc0f1a45f530"
SOURCE_REPORT_SHA = "8d59cf3d44d6657aa8cddd5a22270e39f044357c51ebf690ae0ba940ccf44ba7"
SOURCE_MANIFEST_SHA = "970f2b079c5d00491e95b3f58bfd1180e58fd366d116e0ab4e44731073f9ccb4"
SOURCE_SEEDS = (20261411, 20261412, 20261413)
SEED = 20261511
VAL_SEED, TEST_SEED = 20261503, 20261504
MILESTONES = (0, 256, 512, 1024)
RESIDUAL = np.asarray([.04, .04, .05, 8., .12])
MAX_REQUESTS = 1284
AuditChecks = replay.AuditChecks


def captured(row):
    return bool(row.get("native_release") and row.get("route_success") and not row.get("invalid") and row.get("invalid_physics") is False)


def best_teachers(rows):
    """One TRAIN capture per context: clean, terminal reward, stable raw ID."""
    result = {}
    for row in rows:
        if row["role"] != "train":
            raise EvidenceError("Bootstrap teacher input contains a held-out or control row")
        if not captured(row):
            continue
        key = row["context_id"]
        rank = lambda value: (-int(value.get("route_quality") == 2), -calibrated.row_reward(value), value["id"])
        if key not in result or rank(row) < rank(result[key]):
            result[key] = row
    return [result[key] for key in sorted(result)]


def source_train_rows(parent, parent_contract, checks):
    """Retain every original TRAIN attempt before filtering any positive label."""
    contexts = {context["id"]: context for context in parent_contract["splits"]["train"]}
    checks.equal(len(contexts), 35, "Exactly35 original TRAIN layouts")
    result, files = [], {}
    for method in ("ppo", "sweep"):
        for seed in SOURCE_SEEDS:
            relative = f"runs/{method}-seed{seed}/rows.jsonl"
            path = contained(parent, relative);files[relative] = sha(path)
            selected = [row for row in common.lines(path) if row["role"] == "train"]
            checks.equal(len(selected), 1024, relative+"/all1024 original requests retained")
            checks.equal([row["train_index"] for row in selected], list(range(1024)), relative+"/complete ordered TRAIN prefix")
            for index, row in enumerate(selected):
                label = row["id"]
                checks.equal(label, f"{method}-{seed}-train-{index:05d}", label+"/literal source request")
                checks.equal((row["method"], row["seed"]), (method, seed), label+"/source method and seed")
                checks.require(row["context_id"] in contexts, label+"/TRAIN context only")
                checks.numeric(row["bucket_xy_m"], contexts[row["context_id"]]["bucket_xy_m"], label+"/exact TRAIN bucket coordinate", 0)
                checks.numeric(row["perturbation_mixed_units"], [0.]*5, label+"/no hidden held-out perturbation", 0)
                anchor = row["effective_source_hint"]
                expected_observation = calibrated.expected_observation(row["bucket_xy_m"], anchor)
                checks.numeric(row["observation"], expected_observation, label+"/original geometry-only observation", 0)
                residual = np.asarray(row["normalized_residual"], dtype=float)
                checks.require(residual.shape == (5,) and np.isfinite(residual).all() and (np.abs(residual) <= 1.+1e-8).all(), label+"/bounded five-dimensional source residual")
                checks.numeric(row["values"], np.asarray(anchor["values"])+residual*RESIDUAL, label+"/actual requested source action", 2e-6)
                checks.numeric(row["command_values"], row["values"], label+"/source action never replaced", 1e-12)
                checks.numeric(row["learning_reward"], calibrated.row_reward(row), label+"/source calibrated terminal reward", 1e-8)
                if captured(row):
                    checks.equal(row["contact_protocol"]["substeps"], 64, label+"/source capture is canonical64")
                    checks.equal(row["scoring_version"], "sircus-calibrated-native-capture-v2", label+"/source capture uses current scorer")
                    checks.require(row.get("scalar_score_audit_passed") is True, label+"/source capture independently scored")
                    checks.require(row.get("board_bounce") and row.get("whole_ball_entry") and row.get("max_retention_s", 0) >= 1.-1e-9
                                   and not row.get("external_contact_steps"), label+"/source complete-shot capture")
            result.extend(selected)
    checks.equal(len(result), 6144, "Every saved6144 TRAIN exposure remains accounted")
    checks.equal(len({row["id"] for row in result}), 6144, "Unique source TRAIN identities")
    return result, files


def independent_policy(seed=None):
    """Auditor-owned fixed architecture; no learner model implementation import."""
    import torch
    if seed is not None:
        torch.manual_seed(seed)
    class Policy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            def network(output):
                return torch.nn.Sequential(torch.nn.Linear(9, 64), torch.nn.Tanh(), torch.nn.Linear(64, 64), torch.nn.Tanh(), torch.nn.Linear(64, output))
            self.actor, self.critic = network(5), network(1)
            self.logstd = torch.nn.Parameter(torch.full((5,), -3.))
            torch.nn.init.zeros_(self.actor[-1].weight);torch.nn.init.zeros_(self.actor[-1].bias)
    return Policy().cpu()


def reduced_dataset(rows):
    """Reconstruct the published training view from byte-bound raw TRAIN only."""
    if any(row.get("role") != "train" for row in rows):
        raise EvidenceError("Held-out labels cannot enter the bootstrap dataset")
    return [dict(id=row["id"], role="train", context_id=row["context_id"], bucket_xy_m=row["bucket_xy_m"],
                 observation=row["observation"], normalized_residual=row["normalized_residual"], values=row["values"],
                 effective_source_hint=row["effective_source_hint"], native_release=bool(row["native_release"]),
                 route_success=captured(row), route_quality=2 if captured(row) and row.get("route_quality") == 2 else 1 if captured(row) else 0,
                 invalid=row.get("invalid"), invalid_physics=bool(row.get("invalid_physics")),
                 learning_reward=calibrated.row_reward(row), source_record_sha256=digest(row)) for row in rows]


def regenerated_splits(parent_splits, historical):
    """Seeded fresh positions; no outcome from an excluded layout is inspected."""
    train = [dict(id=context["id"], split="train", bucket_xy_m=copy.deepcopy(context["bucket_xy_m"])) for context in parent_splits["train"]]
    excluded = sorted({tuple(float(value) for value in context["bucket_xy_m"])
                       for source in (parent_splits, historical) for role in ("train", "validation", "test") for context in source[role]})
    used = np.asarray(excluded, dtype=float)
    result = dict(train=train)
    for role, count, seed in (("validation", 12, VAL_SEED), ("test", 50, TEST_SEED)):
        rng, rows = np.random.default_rng(seed), []
        for _ in range(100000):
            if len(rows) == count:
                break
            xy = rng.uniform([.1, -.65], [1.24, .25])
            if np.linalg.norm(used-xy, axis=1).min() < .005:
                continue
            rows.append(dict(id=f"policy64-{role}-{len(rows):02d}", split=role, bucket_xy_m=xy.tolist()))
            used = np.vstack((used, xy))
        if len(rows) != count:
            raise EvidenceError("Could not reconstruct fresh disjoint validation/test layouts")
        result[role] = rows
    return result, [list(xy) for xy in excluded]


def protocol_checks(protocol, checks):
    literal = dict(substeps=64, seed=SEED, train_layouts=35, validation_layouts=12, test_layouts=50,
                   train_requests=1024, batch_size=32, updates=32, milestones=list(MILESTONES),
                   test_methods=["prior", "nearest", "bc", "ppo"], test_repeats=1, validation_requests=48,
                   test_requests=200, control_requests=12, maximum_fresh_requests=MAX_REQUESTS,
                   noise=False, actor_steps=800, critic_steps=400, bootstrap_lr=.001, ppo_lr=.0003,
                   initial_logstd=-3., ppo_epochs=6, clip=.2, entropy=.001, gradient_norm=1.)
    for key, expected in literal.items():
        checks.equal(protocol.get(key), expected, "Prespecified fixed64 protocol/"+key)
    scope = str(protocol.get("scope", "")).lower()
    checks.require(all(word in scope for word in ("64", "128", "historical", "geometry", "no noise")),
                   "New fixed64 scope preserves historical128 failures and geometry-only limitation")


def audit_parent(parent, parent_audit, bindings, checks):
    """Authenticate historical labels without reusing any old held-out outcomes."""
    parent, parent_audit = Path(parent), Path(parent_audit)
    for path, expected, label in ((parent/"report.json", SOURCE_REPORT_SHA, "report"),
                                  (parent/"manifest.json", SOURCE_MANIFEST_SHA, "manifest"),
                                  (parent_audit, SOURCE_AUDIT_SHA, "independent audit")):
        checks.equal(sha(path), expected, "Exact completed calibrated-v2 parent/"+label)
    spec, proof, manifest = read(parent/"contract.json"), read(parent_audit), read(parent/"manifest.json")
    unsigned = dict(spec);fp = unsigned.pop("fingerprint", None)
    checks.equal((digest(unsigned), fp), (SOURCE_FP, SOURCE_FP), "Exact complete historical contract fingerprint")
    checks.equal(proof.get("status"), "passed", "Completed source independent audit passed")
    checks.require(not proof.get("failures") and proof.get("checks") == 1204473, "All1204473 historical source checks passed")
    checks.equal(proof.get("contract_fingerprint"), SOURCE_FP, "Historical proof belongs to exact label source")
    checks.equal(proof.get("report_sha256"), SOURCE_REPORT_SHA, "Historical proof binds report bytes")
    checks.equal(proof.get("study_manifest_sha256"), SOURCE_MANIFEST_SHA, "Historical proof binds complete evidence")
    checks.equal(sha(parent/"contract.json"), manifest["contract.json"], "Historical contract bytes manifest-bound")
    rows, files = source_train_rows(parent, spec, checks)
    for name, value in files.items():
        checks.equal(value, manifest[name], "Historical TRAIN file bytes bound by passed audit/"+name)
    for key, expected in dict(parent_contract_fingerprint=SOURCE_FP, report_sha256=SOURCE_REPORT_SHA,
                              parent_audit_sha256=SOURCE_AUDIT_SHA, parent_manifest_sha256=SOURCE_MANIFEST_SHA,
                              train_files_sha256=files, teacher_ids=[row["id"] for row in best_teachers(rows)]).items():
        checks.equal(bindings[key], expected, "New bootstrap source binding/"+key)
    return spec, rows


def bootstrap_replay(initial, trained, dataset, teachers, fit, checks, *, actor_steps=800, critic_steps=400):
    """Recompute both full-batch Adam fits, not merely a loss-decreased assertion."""
    import torch
    if not dataset or not teachers or any(row.get("role") != "train" for row in dataset+teachers):
        raise EvidenceError("Only TRAIN samples may fit actor and critic")
    model = independent_policy(SEED)
    expected_initial = model.state_dict()
    checks.equal(set(initial), set(expected_initial), "Literal initial actor/critic architecture")
    for name, value in expected_initial.items():
        checks.numeric(common.numpy_state(initial)[name], value.detach().numpy(), "Seeded initialization/"+name, 0)
    model.load_state_dict(initial)
    x = torch.as_tensor(np.asarray([row["observation"] for row in teachers]), dtype=torch.float32)
    y = torch.as_tensor(np.asarray([row["normalized_residual"] for row in teachers]), dtype=torch.float32)
    xx = torch.as_tensor(np.asarray([row["observation"] for row in dataset]), dtype=torch.float32)
    yy = torch.as_tensor([row["learning_reward"] for row in dataset], dtype=torch.float32)
    checks.require(x.ndim == 2 and x.shape[1] == 9 and y.shape == (len(teachers), 5)
                   and xx.shape == (len(dataset), 9) and yy.shape == (len(dataset),)
                   and all(torch.isfinite(value).all().item() for value in (x, y, xx, yy)), "Finite fixed-dimensional TRAIN fitting data")
    actor_optimizer = torch.optim.Adam(model.actor.parameters(), lr=.001)
    critic_optimizer = torch.optim.Adam(model.critic.parameters(), lr=.001)
    def actor_loss():
        return ((model.actor(x).tanh()-y)**2).mean()
    def critic_loss():
        return ((model.critic(xx).squeeze(-1)-yy)**2).mean()
    losses = dict(initial_actor_loss=float(actor_loss().detach()), initial_critic_loss=float(critic_loss().detach()))
    for _ in range(actor_steps):
        loss = actor_loss();actor_optimizer.zero_grad();loss.backward();actor_optimizer.step()
    for _ in range(critic_steps):
        loss = critic_loss();critic_optimizer.zero_grad();loss.backward();critic_optimizer.step()
    losses.update(final_actor_loss=float(actor_loss().detach()), final_critic_loss=float(critic_loss().detach()))
    for key, expected in losses.items():
        checks.numeric(fit[key], expected, "Independent full-batch imitation/critic fit/"+key, 2e-6)
    for name, value in model.state_dict().items():
        checks.numeric(common.numpy_state(trained)[name], value.detach().numpy(), "Replayed bootstrap parameter/"+name, 2e-6)
    checks.equal(fit["before"], common.weights_sha256(initial), "Bootstrap exact before weights")
    checks.equal(fit["after"], common.weights_sha256(trained), "Bootstrap exact after weights")
    checks.equal((fit["actor_steps"], fit["critic_steps"]), (actor_steps, critic_steps), "Literal actor/critic update budgets")
    checks.equal(fit["actor_teacher_ids"], [row["id"] for row in teachers], "One uniformly weighted successful TRAIN context teacher")
    checks.equal(fit["critic_row_count"], len(dataset), "Every historical TRAIN failure/capture reaches critic fit")
    checks.numeric(common.numpy_state(trained)["logstd"], [-3.]*5, "Bootstrap keeps exact exploration scale", 0)
    return losses


def artifact(directory, record, contract_fp, checks, label, *, samples=None):
    path = contained(directory, record["path"])
    checks.equal(sha(path), record["sha256"], label+"/immutable checkpoint bytes")
    state = common.load_checkpoint(path, checks, label)
    checks.equal(state["contract_fingerprint"], contract_fp, label+"/exact study")
    checks.equal((state["seed"], state["method"]), (SEED, "ppo"), label+"/fixed learner seed/method")
    checks.equal(state["samples"], record["samples"], label+"/artifact training-prefix identity")
    if samples is not None:
        checks.equal(state["samples"], samples, label+"/literal training-prefix size")
    checks.equal(record["weight_sha256"], state["weight_sha256"], label+"/published exact weight identity")
    return state


def audit_training(directory, rows, report, spec, dataset, teachers, checks):
    """Audit source-only BC and all32 real PPO updates from fresh TRAIN rows."""
    import torch
    torch.set_num_threads(2)
    initial = artifact(directory, report["artifacts"]["initial"], spec["fingerprint"], checks, "Initial", samples=0)
    bc = artifact(directory, report["artifacts"]["bc"], spec["fingerprint"], checks, "TRAIN-imitation model", samples=0)
    checks.equal(report["artifacts"]["initial"]["path"], "checkpoints/initial.pt", "Fixed seeded initializer path")
    checks.equal(report["artifacts"]["bc"]["path"], "checkpoints/milestone-0000.pt", "BC is the zero-fresh-request checkpoint")
    checks.require(not initial["optimizer"]["state"] and not bc["optimizer"]["state"], "Fresh PPO Adam state after two separate imitation fits")
    fit = read(directory/"bootstrap/fit.json")
    fit_result = bootstrap_replay(initial["policy"], bc["policy"], dataset, teachers, fit, checks)
    train = [row for row in rows if row["role"] == "train"]
    checks.equal(len(train), 1024, "Exactly1024 fresh TRAIN requests including all rejections")
    checks.equal([row["train_index"] for row in train], list(range(1024)), "Complete ordered fresh TRAIN prefix")
    schedule = calibrated.expected_schedule(SEED, 1024)
    checks.equal(read(directory/"schedule.json"), schedule, "Seeded repeated balanced TRAIN-layout schedule")
    contexts = spec["splits"]["train"]
    updates = common.lines(directory/"updates.jsonl")
    checks.equal(len(updates), 32, "All32 PPO updates persisted")
    previous = bc
    milestones = {0: dict(record=report["artifacts"]["bc"], state=bc)}
    for index, update in enumerate(updates):
        offset = index*32;label = f"PPO update{index:02d}"
        cohort = train[offset:offset+32]
        checks.equal(update["index"], index, label+"/ordered update index")
        checks.equal(update["train_ids"], [row["id"] for row in cohort], label+"/only current32 TRAIN outcomes")
        before = artifact(directory, update["behavior"], spec["fingerprint"], checks, label+" behavior", samples=offset)
        after = artifact(directory, update["after_artifact"], spec["fingerprint"], checks, label+" after", samples=offset+32)
        checks.equal(update["behavior"]["path"], f"behavior/update-{index:04d}.pt", label+"/literal behavior path")
        checks.equal(update["after_artifact"]["path"], f"after/update-{index:04d}.pt", label+"/literal after path")
        checks.equal(before["weight_sha256"], previous["weight_sha256"], label+"/continuous model chain")
        checks.equal(common.tensor_object_digest(before["optimizer"]), common.tensor_object_digest(previous["optimizer"]), label+"/continuous Adam moments and steps")
        checks.equal((update["before"], update["after"]), (before["weight_sha256"], after["weight_sha256"]), label+"/actual changed-state hashes")
        checks.equal(update["buffer"], f"buffers/update-{index:04d}.npz", label+"/literal TRAIN buffer path")
        path = contained(directory, update["buffer"])
        checks.equal(sha(path), update["buffer_sha256"], label+"/exact frozen update buffer")
        with np.load(path, allow_pickle=False) as archive:
            checks.equal(set(archive.files), {"observation", "latent", "old_logp", "reward"}, label+"/no extra held-out buffer input")
            buffer = {key: archive[key].copy() for key in archive.files}
        for local, row in enumerate(cohort):
            ident = f"policy64-train-{offset+local:05d}"
            context = contexts[schedule[offset+local]]
            checks.equal(row["id"], ident, label+"/literal request ID")
            checks.equal((row["method"], row["seed"], row["phase"], row["samples"]), ("ppo", SEED, "train", offset+local+1), ident+"/fresh learner identity")
            checks.equal(row["context_id"], context["id"], ident+"/TRAIN-only scheduled context")
            checks.equal(row["bucket_xy_m"], context["bucket_xy_m"], ident+"/TRAIN-only bucket coordinate")
            checks.equal(row["source_training_ids"], [], ident+"/no label-derived teacher action injected into PPO sampling")
            checks.equal((row["behavior_checkpoint"], row["behavior_checkpoint_sha256"]),
                         (update["behavior"]["path"], update["behavior"]["sha256"]), ident+"/exact sampling actor before flight")
        for key, row_key in (("observation", "observation"), ("latent", "latent"), ("old_logp", "old_logp"), ("reward", "learning_reward")):
            expected = np.asarray([row[row_key] for row in cohort], dtype=np.float32)
            checks.numeric(buffer[key], expected, label+"/buffer contains only current TRAIN/"+key, 0)
        common.check_behavior_likelihood(before["policy"], buffer["observation"], buffer["latent"], buffer["old_logp"], checks, label)
        residual = np.tanh(np.asarray(buffer["latent"], dtype=float))
        checks.numeric([row["normalized_residual"] for row in cohort], residual, label+"/sampled bounded residual", 1e-12)
        for local, row in enumerate(cohort):
            expected = np.asarray(row["effective_source_hint"]["values"])+residual[local]*RESIDUAL
            checks.numeric(row["values"], expected, row["id"]+"/exact sampled continuous request", 1e-12)
        replayed = common.replay_ppo_update(before["policy"], before["optimizer"], after["policy"],
                    dict(obs=buffer["observation"], latent=buffer["latent"], old_logp=buffer["old_logp"], reward=buffer["reward"]), checks, label)
        checks.equal(len(update["epochs"]), 6, label+"/six actual PPO epochs")
        checks.numeric([epoch["loss"] for epoch in update["epochs"]], replayed["losses"], label+"/independent epoch losses", 2e-6)
        checks.require(after["weight_sha256"] != before["weight_sha256"], label+"/actual model update, not unchanged fake training")
        for state in after["optimizer"]["state"].values():
            checks.numeric(float(state["step"]), 6*(index+1), label+"/actual cumulative Adam step", 0)
        checks.numeric(common.numpy_state(after["policy"])["logstd"], np.clip(common.numpy_state(after["policy"])["logstd"], -3., -.7), label+"/bounded exploration scale", 0)
        previous = after
        samples = offset+32
        if samples in MILESTONES:
            relative = f"checkpoints/milestone-{samples:04d}.pt"
            state = common.load_checkpoint(directory/relative, checks, f"Milestone{samples}")
            record = dict(path=relative, sha256=sha(directory/relative), weight_sha256=state["weight_sha256"], samples=samples)
            loaded = artifact(directory, record, spec["fingerprint"], checks, f"Milestone{samples}", samples=samples)
            checks.equal(loaded["weight_sha256"], after["weight_sha256"], f"Milestone{samples}/frozen exact TRAIN prefix")
            checks.equal(common.tensor_object_digest(loaded["optimizer"]), common.tensor_object_digest(after["optimizer"]), f"Milestone{samples}/same optimizer checkpoint")
            milestones[samples] = dict(record=record, state=loaded)
    checks.equal((report["train_requests"], report["updates"]), (1024, 32), "Reported actual fresh-learning budget")
    return milestones, fit_result


def nearest_teacher(context, teachers):
    return min(teachers, key=lambda row: (math.dist(context["bucket_xy_m"], row["bucket_xy_m"]), row["context_id"], row["id"]))


def audit_prediction_packet(packet, rows, contexts, record, state, teachers, checks):
    role, method, samples = packet["role"], packet["method"], packet["samples"]
    label = f"{role}/{method}/{samples}"
    checks.equal(packet["artifact"], record, label+"/single immutable model/library")
    checks.require(packet.get("selection_or_learning") is False and packet.get("one_prediction_per_layout") is True, label+"/frozen single-shot inference")
    checks.equal(len(packet["rows"]), len(contexts), label+"/one prediction per requested layout")
    checks.equal(len(rows), len(contexts), label+"/every prediction spent once including setup failures")
    for prediction, row, context in zip(packet["rows"], rows, contexts):
        ident = f"{role}-{method}-{samples:04d}-{context['id']}"
        checks.equal(row["id"], ident, label+"/literal evaluation request")
        checks.equal((row["role"], row["phase"], row["method"], row["samples"]), (role, role, method, samples), ident+"/frozen evaluation cohort")
        checks.equal(prediction["context_id"], context["id"], ident+"/exact held-out context")
        checks.equal(prediction["bucket_xy_m"], context["bucket_xy_m"], ident+"/exact held-out coordinate")
        for key, value in prediction.items():
            checks.equal(row.get(key), value, ident+"/prediction preserved before outcome/"+key)
        checks.equal(prediction["artifact_sha256"], record["sha256"], ident+"/published prediction model/library identity")
        checks.equal(row["frozen_artifact_sha256"], record["sha256"], ident+"/exact frozen artifact before flight")
        checks.require(prediction.get("ball_flight_search_at_inference") is False and prediction.get("selection_or_learning") is False
                       and prediction.get("geometry_search_at_inference") is True, ident+"/geometry-only setup distinct from no flight search")
        latent = None;source_ids = []
        if method == "prior":
            residual = np.zeros(5)
        elif method == "nearest":
            teacher = nearest_teacher(context, teachers)
            residual = np.asarray(teacher["normalized_residual"], dtype=float);source_ids = [teacher["id"]]
        elif method in ("bc", "ppo"):
            checks.require(state is not None, ident+"/fixed actor state required")
            latent = common.network_forward(state["policy"], prediction["observation"])
            residual = np.tanh(latent.astype(float))
        else:
            raise EvidenceError("Unknown frozen prediction method")
        checks.equal(prediction["source_training_ids"], source_ids, ident+"/immutable TRAIN-only baseline library")
        checks.numeric(prediction["latent_mean"], None if latent is None else latent, ident+"/independent frozen actor mean", 2e-6)
        checks.numeric(prediction["normalized_residual"], residual, ident+"/independent frozen residual", 2e-6)
        checks.numeric(prediction["command_values"], np.asarray(prediction["effective_source_hint"]["values"])+residual*RESIDUAL, ident+"/single predicted shot without flight optimization", 2e-6)
        checks.numeric(row["values"], row["command_values"], ident+"/no postprediction action substitution", 0)


def control_gate(rows):
    positive = [row for row in rows if row["role"] == "positive_control"]
    negative = [row for row in rows if row["role"] == "negative_control"]
    good = (len(rows) == 6 and len(positive) == len(negative) == 3 and all(captured(row) for row in positive)
            and all(row.get("native_release") and not captured(row) and not row.get("whole_ball_entry") for row in negative)
            and all(not row.get("invalid") and not row.get("invalid_physics") and row.get("scalar_score_audit_passed") is True for row in rows))
    return dict(status="passed" if good else "failed", counts=calibrated.counts(rows),
                positive_captures=sum(captured(row) for row in positive), negative_captures=sum(captured(row) for row in negative))


def audit_native_options(directory, scene, calibration, assets, checks):
    """CPU geometry comparison and actual native/Warp64 option identities."""
    import mujoco
    from .model import build_scene
    from .verify_capture_calibration import compile_cpu_model, inspect_compiled_geometry
    options = read(directory/"effective-options.json")
    checks.equal((options["version"], options["substeps"]), ("sircus-calibrated-native-warp-options-v1", 64), "Only allocated canonical64 execution")
    expected = dict(timestep=1/15360, solver=2, iterations=100, tolerance=1e-8, cone=0, integrator=3,
                    ccd_iterations=50, ccd_tolerance=1e-6, disableflags=0, enableflags=0, gravity=[0., 0., -9.81])
    checks.equal(options["native"], expected, "Unchanged exact allocated native physics options")
    warp = {key: np.asarray(1e-6 if key == "tolerance" else value, dtype=np.float32).astype(float).tolist()
            if key in {"timestep", "tolerance", "ccd_tolerance", "gravity"} else value for key, value in expected.items()}
    checks.equal(options["warp"], warp, "Actual float32 Warp options with vendor tolerance floor")
    compiled = read(directory/"native-compiled-geometry.json")
    full = mujoco.MjModel.from_xml_string(build_scene(scene, assets, solref=(.004, .15), contact_friction=.35))
    names = [f"bucket/wall{index:02d}" for index in range(scene["bucket"]["segments"])]
    tags = dict(ball=int(full.geom("ball/geom").id), base=int(full.geom("bucket/base").id), walls=[int(full.geom(name).id) for name in names])
    checks.equal(compiled["contact_geom_ids"], tags, "Actual full native-model contact category IDs")
    checks.numeric(compiled["compiled_planes"], calibration["compiled_planes"], "Runtime bucket contact planes equal frozen calibration", calibration["geometry_epsilon_m"])
    checks.equal(compiled["effective_options"], expected, "Compiled geometry same native physics")
    checks.equal(compiled["geometry_sha256"], digest(dict(planes=compiled["compiled_planes"], floor=compiled["compiled_floor_m"],
                  rim=compiled["compiled_rim_m"], radius=compiled["ball_radius_m"])), "Independent compiled geometry identity")
    cpu, _ = compile_cpu_model(scene, assets, 64)
    isolated = copy.deepcopy(compiled)
    isolated["contact_geom_ids"] = dict(ball=int(cpu.geom("ball/geom").id), base=int(cpu.geom("bucket/base").id), walls=[int(cpu.geom(name).id) for name in names])
    class GeometryChecks:
        require = checks.require
        def close(self, actual, expected, label, *, atol=1e-12):
            checks.numeric(actual, expected, label, atol)
    inspect_compiled_geometry(cpu, scene, isolated, GeometryChecks())
    return options


def audit_controls(rows, sources, checks):
    result = {}
    for phase, key in (("controls_pre", "pre"), ("controls_post", "post")):
        group = [row for row in rows if row["phase"] == phase]
        checks.equal(len(group), 6, phase+"/two controls with three fresh nominal requests each")
        for index, (source, role) in enumerate(zip(sources, ("positive_control", "negative_control"))):
            for repeat in range(3):
                row = group[index*3+repeat];ident = f"{phase}-{role}-r{repeat}"
                checks.equal(row["id"], ident, phase+"/literal control identity")
                checks.equal((row["role"], row["method"], row["repeat"]), (role, "control", repeat), ident+"/control metadata")
                checks.equal((row["control_source_id"], row["control_source_sha256"]), (source["id"], digest(source)), ident+"/full historical control provenance")
                checks.equal(row["bucket_xy_m"], source["bucket_xy_m"], ident+"/same historical layout")
                checks.equal(row["values"], source["values"], ident+"/same exact action, newly checked endpoint")
        result[key] = control_gate(group)
        checks.equal(result[key]["status"], "passed", phase+"/both calibrated controls pass")
    return result


def audit_predictions(directory, rows, report, spec, milestones, teachers, checks):
    curves = []
    for samples in MILESTONES:
        record, state = milestones[samples]["record"], milestones[samples]["state"]
        packet = read(directory/f"predictions/validation-ppo-{samples:04d}.json")
        cohort = [row for row in rows if row["role"] == "validation" and row["samples"] == samples]
        checks.equal((packet["role"], packet["method"], packet["samples"]), ("validation", "ppo", samples), "Literal validation milestone")
        audit_prediction_packet(packet, cohort, spec["splits"]["validation"], record, state, teachers, checks)
        point = dict(samples=samples, validation=calibrated.counts(cohort), artifact=record)
        curves.append(point)
    checks.equal(report["curves"], curves, "Every fresh validation outcome remains visible")
    best = max(curves, key=lambda point: (point["validation"]["captures"], point["validation"]["mean_reward"], -point["samples"]))
    selected = milestones[best["samples"]]["record"]
    checks.equal(report["selected"], selected, "Validation-only frozen model selection")
    checks.equal(report["artifacts"]["selected"], selected, "Same frozen chosen artifact published")
    chosen = read(directory/"selected.json")
    checks.equal(chosen, dict(selected=selected, rule=spec["protocol"]["selection"], validation=best["validation"]), "Selection record uses validation only and is frozen before TEST")
    plan = read(directory/"test-plan.json")
    checks.equal(plan["selected"], selected, "Whole TEST plan binds frozen selected model")
    checks.require(plan["frozen_before_outcomes"] is True, "All200 predictions frozen as one plan before first TEST outcome")
    checks.equal([packet["method"] for packet in plan["methods"]], ["prior", "nearest", "bc", "ppo"], "Four prespecified paired methods in literal order")
    comparisons, positions = [], []
    for method, packet in zip(("prior", "nearest", "bc", "ppo"), plan["methods"]):
        if method in ("prior", "nearest"):
            record = report["artifacts"][method];state = None
            checks.equal((record["path"], record["samples"]), (f"checkpoints/{method}.json", 0), method+"/immutable source-only baseline")
            checks.equal(sha(contained(directory, record["path"])), record["sha256"], method+"/exact frozen library bytes")
            definition = read(contained(directory, record["path"]))
            checks.equal(definition["method"], method, method+"/library method")
            checks.equal(definition["contract_fingerprint"], spec["fingerprint"], method+"/same contract")
            checks.equal(definition["teachers_sha256"], spec["teachers_sha256"], method+"/only immutable successful TRAIN teachers")
            checks.equal(definition["rule"], spec["protocol"]["nearest"] if method == "nearest" else "zero residual", method+"/frozen baseline rule")
        else:
            member = milestones[best["samples"] if method == "ppo" else 0]
            record, state = member["record"], member["state"]
        checks.equal((packet["role"], packet["samples"]), ("test", record["samples"]), method+"/literal TEST checkpoint")
        checks.equal(read(directory/f"predictions/test-{method}-{record['samples']:04d}.json"), packet, method+"/complete preoutcome TEST prediction packet")
        cohort = [row for row in rows if row["role"] == "test" and row["method"] == method]
        audit_prediction_packet(packet, cohort, spec["splits"]["test"], record, state, teachers, checks)
        comparisons.append(dict(method=method, counts=calibrated.counts(cohort), samples=record["samples"], artifact_sha256=record["sha256"]))
        for row in cohort:
            positions.append(dict(context_id=row["context_id"], bucket_xy_m=row["bucket_xy_m"], method=method, values=row["values"],
                 board_bucket_distance_m=float(np.linalg.norm(np.asarray(row["values"][:2])-row["bucket_xy_m"])),
                 capture=captured(row), clean=captured(row) and row.get("route_quality") == 2, native_release=row["native_release"], invalid=row.get("invalid"), invalid_physics=row.get("invalid_physics")))
    checks.equal(report["test_comparison"], comparisons, "Reported paired TEST denominators and outcomes")
    checks.equal(report["test_positions"], positions, "Every one-shot TEST request, including failures, visible")
    return selected, comparisons


def audit_sequence(rows, selected_samples, checks):
    expected = [("controls_pre", "positive_control")]*3+[("controls_pre", "negative_control")]*3
    expected += [("validation", "validation")]*12
    for offset in range(0, 1024, 32):
        expected += [("train", "train")]*32
        if offset+32 in MILESTONES:
            expected += [("validation", "validation")]*12
    for _ in ("prior", "nearest", "bc", "ppo"):
        expected += [("test", "test")]*50
    expected += [("controls_post", "positive_control")]*3+[("controls_post", "negative_control")]*3
    checks.equal(len(rows), MAX_REQUESTS, "Exact1284 fresh requests spent without hidden retries")
    checks.equal([(row["phase"], row["role"]) for row in rows], expected, "Controls, validation milestones, TRAIN and TEST execution order")
    test = [row for row in rows if row["role"] == "test"]
    checks.equal([row["method"] for row in test], [method for method in ("prior", "nearest", "bc", "ppo") for _ in range(50)], "Paired TEST methods cannot interleave or replace failures")
    checks.equal([row["samples"] for row in test], [0]*150+[selected_samples]*50, "Only selected frozen PPO and immutable baselines tested")


def audit_trace_examples(directory, report, rows, checks):
    expected = []
    for method in ("bc", "ppo"):
        for success in (True, False):
            cohort = [row for row in rows if row["role"] == "test" and row["method"] == method and row.get("native_release") and captured(row) == success]
            if cohort:
                expected.append(cohort[0])
    checks.equal([trace["id"] for trace in report["trace_examples"]], [row["id"] for row in expected], "Literal first successful/failed native TEST examples, no cherry-picked invention")
    for trace, row in zip(report["trace_examples"], expected):
        for key in ("values", "bucket_xy_m", "board_center_m", "board_normal", "trajectory_m", "trajectory_dt_s", "whole_ball_entry", "max_retention_s", "invalid", "invalid_physics", "scalar_score_audit_passed", "native_cache_sha256", "native_cache_file", "native_cache_row"):
            checks.equal(trace[key], row[key], row["id"]+"/display retains actual measured/"+key)
        checks.equal((trace["phase"], trace["method"], trace["substeps"], trace["capture"], trace["clean"]),
                     ("test", row["method"], 64, captured(row), captured(row) and row["route_quality"] == 2), row["id"]+"/trace outcome provenance")
        with np.load(contained(directory/"native", row["native_cache_file"]), allow_pickle=False) as archive:
            actual = archive["position_m"][256::256, row["native_cache_row"]]
        checks.numeric(trace["trajectory_m"], actual, row["id"]+"/displayed actual dense native trace", 0)
        checks.numeric(trace["trajectory_dt_s"], 1/60, row["id"]+"/display actual sample cadence", 0)


def auditor_identity(producer, source_hashes, checks):
    root = Path(__file__).resolve().parents[2]
    own = "sircus/mjlab/policy64_verify.py"
    dependencies = {}
    for module in tuple(sys.modules.values()):
        name, filename = getattr(module, "__name__", ""), getattr(module, "__file__", None)
        if not filename or not (name.startswith("sircus.mjlab.") or name.startswith("sircus.isaac.")):
            continue
        path = Path(filename).resolve()
        checks.require(path.is_relative_to(root), "One explicit independent audit source snapshot/"+name)
        relative = str(path.relative_to(root))
        if len(Path(relative).parts) != 3 or path.suffix != ".py" or relative == own:
            continue
        if relative.startswith("sircus/isaac/") and relative not in source_hashes:
            continue
        checks.require(relative in source_hashes, "Imported dependency declared by native producer/"+relative)
        checks.equal(sha(path), source_hashes[relative], "Unchanged actual imported scorer/geometry dependency/"+relative)
        dependencies[relative] = sha(path)
    checks.require("sircus/mjlab/calibrated_score_oracle.py" in dependencies, "Actual independent calibrated native scorer retained")
    checks.equal(sha(contained(producer, own)), source_hashes[own], "Native source retains original checker bytes")
    return dict(version=VERSION, auditor_sha256=sha(Path(__file__)), original_producer_auditor_sha256=source_hashes[own], dependencies_sha256=dependencies)


def audit_runtime_reference(source, reference, spec, parent, assets, checks):
    """Old label ancestry and newer execution ancestry are distinct contracts."""
    import importlib.metadata
    source, reference, assets = Path(source), Path(reference), Path(assets)
    public = read(source/"baselines/mjlab-corrected-grid-20261102/native-results.json")
    checks.equal(public["contract_fingerprint"], "8edba8a3349aa8bcb4c06b97f3f61a302684856513b0ac972cf849afd3f4f037", "Historical scene/assets geometry reference")
    expected_files = {"report.json": "46b5e318863b9f750d170fbbb9aad32a17e5a4b92777563ca3e17fccba3324ee",
                      "contract.json": "f0b99c9084f5070dfebe0134e9a9194b4dd414f13b72eef3a15d2d3fa137f5f5",
                      "confirmed-catalogue.json": "cd4d1f351b83a631c70eefefa5e3e58f064b30934729250a80ec0ac6c1b52ade"}
    for name, value in expected_files.items():
        checks.equal(public["artifacts_sha256"][name], value, "Published original geometry bytes/"+name)
        checks.equal(sha(reference/name), value, "Actual geometry input bytes/"+name)
    geometry = read(reference/"contract.json");unsigned = dict(geometry);fp = unsigned.pop("fingerprint")
    checks.equal((digest(unsigned), fp), (public["contract_fingerprint"], public["contract_fingerprint"]), "Actual geometry reference full fingerprint")
    recent_path = source/"baselines/mjlab-impact650-20261105/summary.json"
    checks.equal(sha(recent_path), "de996c0e54860707a70650c76104ca944309a50b071c01a3f25532fcfb4b2364", "Exact recently audited current-runtime provenance")
    recent = read(recent_path)["provenance"];runtime = spec["runtime_reference"]
    checks.equal(recent["contract_fingerprint"], "1c59ce4342e6064b52f43ab64856a2245019de574ae563c88894f179840a0f31", "Current native parent contract identity")
    checks.equal(recent["independent_audit_sha256"], "ee7331edec9580fed7cdea0b9b28ab5a40cc6337034fa1f7e8b8d5cae17b4883", "Current native parent full independent proof")
    for key, value in dict(version="sircus-policy64-current-native-reference-v1", historical_geometry_fingerprint=fp,
            current_native_proof_sha256=recent["independent_audit_sha256"], current_native_contract_fingerprint=recent["contract_fingerprint"],
            source_sha256=recent["executed_source_sha256"], asset_sha256=recent["asset_sha256"]).items():
        checks.equal(runtime[key], value, "Explicit new current-runtime ancestry/"+key)
    for name, value in recent["executed_source_sha256"].items():
        checks.equal(sha(contained(source, name)), value, "Existing executed native module unchanged since audited impact study/"+name)
    checks.equal(spec["parent_identity"], parent["identity"], "Historical saved-label native identity remains separately preserved")
    expected = copy.deepcopy(parent["identity"]);expected["source_sha256"] = spec["source_sha256"]
    checks.equal(spec["identity"], expected, "Current execution identity explicitly replaces only source snapshot")
    checks.equal(spec["scene"], geometry["scene"], "Exact original scene and provisional bucket dimensions preserved")
    checks.equal(digest(spec["scene"]), parent["identity"]["scene_sha256"], "Saved label scene geometry compatible")
    checks.equal(geometry["asset_sha256"], recent["asset_sha256"], "Original scene/current runtime share exact assets")
    checks.equal(parent["identity"]["asset_sha256"], recent["asset_sha256"], "Bootstrap source shares exact robot collision geometry")
    from .learning_solve import required_asset_paths
    actual_assets = {name: sha(contained(assets.parent, name)) for name in required_asset_paths(assets)}
    checks.equal(actual_assets, recent["asset_sha256"], "Actual independently read asset bytes")
    packages = dict(parent["packages"], scipy="1.15.3")
    checks.equal(spec["packages"], packages, "Full current package set including CPU hull dependency")
    checks.equal({name: importlib.metadata.version(name) for name in packages}, packages, "CPU auditor actual package identities equal native source")


def audit(study, source, assets, *, parent_study, parent_audit, reference):
    started, checks = time.monotonic(), AuditChecks()
    study, source, assets, parent_study, parent_audit = map(Path, (study, source, assets, parent_study, parent_audit))
    manifest = replay.check_manifest(study, checks)
    spec, report = read(study/"contract.json"), read(study/"report.json")
    unsigned = dict(spec);fingerprint = unsigned.pop("fingerprint", None)
    checks.equal(digest(unsigned), fingerprint, "Complete new fixed64 protocol fingerprint")
    checks.equal((spec["version"], report["version"]), (STUDY_VERSION, STUDY_VERSION), "Explicit newly versioned64-only learning study")
    checks.equal((report["status"], report["phase"]), ("completed", "completed"), "All finite training/evaluation phases finished")
    checks.equal(report["contract_fingerprint"], fingerprint, "Published results bind exact pretraining contract")
    checks.equal(spec["evidence_kind"], "native_fixed64_success_bootstrap_ppo", "Fresh native simulated evidence, not synthetic fixture")
    checks.equal(report["evidence_sha256"], manifest, "Report binds every artifact and native packet")
    checks.equal(report["protocol"], spec["protocol"], "Reported literal protocol unchanged")
    protocol_checks(spec["protocol"], checks)
    replay.check_source_files(source, spec["source_sha256"], checks, "Actual complete producer source")
    paths = list((source/"sircus/mjlab").glob("*.py"))+[source/"sircus/mjlab/requirements.txt"]
    paths += [source/"sircus/isaac"/name for name in ("geometry.py", "rl_contract.py", "entry_quality.py", "board_route.py")]
    checks.equal(set(spec["source_sha256"]), {str(path.relative_to(source)) for path in paths}, "Complete source manifest includes new learner/checker and existing native paths")
    parent, raw_train = audit_parent(parent_study, parent_audit, spec["source_bindings"], checks)
    audit_runtime_reference(source, reference, spec, parent, assets, checks)
    dataset, raw_teachers = reduced_dataset(raw_train), best_teachers(raw_train)
    by_id = {row["id"]: row for row in dataset};teachers = [by_id[row["id"]] for row in raw_teachers]
    checks.equal(read(study/"bootstrap/dataset.json"), dataset, "Complete TRAIN-only reduced bootstrap data with full raw-row hashes")
    checks.equal(read(study/"bootstrap/teachers.json"), teachers, "One literal best captured TRAIN teacher per successful context")
    checks.equal(read(study/"bootstrap/source-bindings.json"), spec["source_bindings"], "Frozen historical dataset byte bindings")
    for name, key in (("bootstrap/dataset.json", "bootstrap_dataset_sha256"), ("bootstrap/teachers.json", "teachers_sha256"), ("controls.json", "controls_sha256")):
        checks.equal(sha(study/name), spec[key], "Pretraining frozen input bytes/"+name)
    summary = dict(requests=6144, native_releases=sum(bool(row["native_release"]) for row in raw_train),
           captures=sum(captured(row) for row in raw_train), clean=sum(captured(row) and row["route_quality"] == 2 for row in raw_train),
           layouts=35, successful_layouts=len(teachers), teacher_count=len(teachers))
    for key, value in summary.items():
        checks.equal(spec["bootstrap_summary"][key], value, "Historical TRAIN acquisition, not free warm start/"+key)
    checks.equal(report["bootstrap_summary"], spec["bootstrap_summary"], "Historical exposure cost displayed separately")
    negatives = [row for row in raw_train if row.get("native_release") and not captured(row) and not row.get("invalid")
                 and not row.get("invalid_physics") and not row.get("whole_ball_entry")]
    controls = [raw_teachers[0], min(negatives, key=lambda row: row["id"])]
    checks.equal(read(study/"controls.json"), controls, "Predeclared positive/negative control rows selected only from TRAIN")
    prior = read(study/"prior.json")
    checks.equal(sha(parent_study/"prior.json"), read(parent_study/"manifest.json")["prior.json"], "Old geometry-only prior bound by parent audit")
    checks.equal(prior, read(parent_study/"prior.json"), "Same17 geometry-only setup anchors, no success library injected into resolver")
    checks.equal(digest(prior), spec["prior_sha256"], "Frozen geometry-only prior identity")
    historical = read(source/"baselines/mjlab-sweep-rl-benchmark-20261103/splits.json")
    splits, excluded = regenerated_splits(parent["splits"], historical)
    checks.equal((spec["splits"], read(study/"splits.json")), (splits, splits), "Fresh seeded disjoint35/12/50 split manifests")
    checks.equal((spec["exclusions"], read(study/"exclusions.json")), (excluded, excluded), "Every historical TRAIN/validation/TEST coordinate excluded from new heldouts")
    calibration = replay.check_calibration(study, spec["identity"]["calibration"], checks)
    for name in ("calibration.json", "verification.json", "setup-home.json"):
        checks.equal(sha(study/name), sha(parent_study/name), "Original calibration and canonical home unchanged/"+name)
    home = spec["identity"]["setup"]["home"]
    checks.equal(read(study/"setup-home.json"), home, "Explicit frozen canonical-home record")
    rows = replay.read_records(study/"rows.jsonl")
    calibrated.check_rows(rows, checks)
    for row in rows:
        checks.equal(row["adapter_directory"], "native", row["id"]+"/single native64 adapter")
        checks.equal(row["command_values"], row["values"], row["id"]+"/exact requested action preserved")
        checks.equal(row["perturbation_mixed_units"], [0.]*5, row["id"]+"/noise expressly deferred")
        checks.require(row.get("action_substituted") is False, row["id"]+"/rejected actions never replaced")
        if row["native_release"]:
            checks.equal(row["contact_protocol"], dict(substeps=64, board_mode="articulated", gravity_mode="passive_only"), row["id"]+"/unchanged canonical held-endpoint contact protocol")
            checks.equal(row["scoring_version"], "sircus-calibrated-native-capture-v2", row["id"]+"/versioned calibrated outcome")
            checks.require(row.get("scalar_score_audit_passed") is True, row["id"]+"/online independent scalar scorer agreement")
            checks.equal((row["effective_options_file"], row["effective_options_sha256"]),
                         ("effective-options.json", sha(study/"native/effective-options.json")), row["id"]+"/observed allocated options bytes bound to release")
    setup = calibrated.check_setup_evidence(study, rows, prior, home, checks)
    options = audit_native_options(study/"native", spec["scene"], calibration, assets, checks)
    traces = calibrated.rescore_rows(study/"native", rows, spec["scene"], calibration, checks, "Fresh native64", spec["identity"]["trace_schema"], assets=assets)
    milestones, fit = audit_training(study, rows, report, spec, dataset, teachers, checks)
    selected, comparisons = audit_predictions(study, rows, report, spec, milestones, teachers, checks)
    audit_sequence(rows, selected["samples"], checks)
    gates = audit_controls(rows, controls, checks)
    checks.equal(report["control_gates"], gates, "Reported before/after fresh64 control gates")
    calibrated.stats_equal(report["counts"], rows, checks, "Every fresh attempt incl. controls and rejects")
    audit_trace_examples(study, report, rows, checks)
    checks.equal(report["gate"], dict(status="passed", controls_passed=True, complete=True, substeps=64,
                 scope="Protocol/controls completion, not a128convergence or minimum success-rate gate"), "Completion cannot be misread as timestep or success-rate qualification")
    identity = auditor_identity(source, spec["source_sha256"], checks)
    return dict(version=VERSION, status="passed", contract_fingerprint=fingerprint, checks_count=checks.count,
        checks=checks.entries, failures=[], auditor_sha256=sha(Path(__file__)), auditor_identity=identity,
        report_sha256=sha(study/"report.json"), evidence_manifest_sha256=sha(study/"evidence-manifest.json"),
        producer_source_sha256=spec["source_sha256"], requested_trials=len(rows), native_releases=len(traces),
        bootstrap_summary=summary, source_contract_fingerprint=SOURCE_FP, selected=selected,
        test_comparison=comparisons, control_gates=gates, setup_evidence=setup, bootstrap_replay=fit,
        role_counts={role: calibrated.counts([row for row in rows if row["role"] == role]) for role in ("train", "validation", "test", "positive_control", "negative_control")},
        elapsed_s=time.monotonic()-started, maximum_numeric_error=checks.numeric_error,
        limitations=["Fixed64 privileged geometry only; archived128 failures remain and no timestep-convergence claim is made.",
                     "One deterministic request per method at each of50 fresh TEST layouts, not50 repeated-reliability trials.",
                     "BC+PPO adds1024 fresh exposures to6144 historical TRAIN requests; comparisons are ablations, not equal-acquisition-budget algorithm superiority.",
                     "Geometry-only resolution and sampled setup paths remain separate from learned shot selection, driven arm transit and grasping.",
                     "No perception, noise robustness, physical transfer or all-table coverage guarantee."])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("study", "source", "assets", "parent-study", "parent-audit", "reference", "output"):
        parser.add_argument("--"+name, type=Path, required=True)
    args = parser.parse_args()
    if args.output.resolve().is_relative_to(args.study.resolve()):
        raise ValueError("Write the new independent proof outside original study evidence")
    try:
        result = audit(args.study, args.source, args.assets, parent_study=args.parent_study, parent_audit=args.parent_audit, reference=args.reference)
    except Exception as error:
        result = dict(version=VERSION, status="failed", failures=[type(error).__name__+": "+str(error)])
        args.output.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
        raise
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    print(json.dumps({key: result[key] for key in ("status", "checks_count", "requested_trials", "native_releases", "selected")}))


if __name__ == "__main__":
    main()
