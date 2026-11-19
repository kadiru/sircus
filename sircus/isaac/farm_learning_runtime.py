"""Native headless sweep, categorical PPO and frozen per-table evaluation.

Reuses the original full-rate PhysX contact/capture evaluator and measured
endpoint gates. Direct endpoint initialization is deliberate and is not a
driven-path or joint-control validation. No RGB frames are written.
"""

import copy
import hashlib
import json
import shutil
import time
from contextlib import contextmanager
from pathlib import Path
from types import MethodType

import numpy as np


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@contextmanager
def endpoint_mode(env):
    from feasibility_runtime import FeasibilityShots
    from parallel_runtime import ParallelShots

    methods = {
        k: getattr(env, k) for k in ("prepare", "approach_endpoints", "tick", "render")
    }
    previous = {
        k: getattr(env, k, None)
        for k in ("training_active", "preparation_enabled", "record_video")
    }
    env.prepare = MethodType(FeasibilityShots.prepare, env)
    env.approach_endpoints = MethodType(ParallelShots.approach_endpoints, env)
    env.tick = MethodType(ParallelShots.tick, env)
    env.render = lambda: None
    env.training_active = False
    env.record_video = False
    try:
        yield
    finally:
        for key, value in methods.items():
            setattr(env, key, value)
        for key, value in previous.items():
            setattr(env, key, value)


def load_offline_bank(env, spec, path, limit):
    from expanded_farm import validate_action
    from expanded_pose_menu import endpoint_targets
    from expanded_pose_native import canonical_hash, verified_fit
    from shot_sweep import check_fitted_geometry

    source = json.loads(Path(path).read_bytes())
    unsigned = copy.deepcopy(source)
    fingerprint = unsigned.pop("fingerprint")
    if (
        canonical_hash(unsigned) != fingerprint
        or source["qualification"] != "offline_endpoint_ik"
        or source["source_fingerprint"] != spec["fingerprint"]
        or canonical_hash(source["spec"]) != canonical_hash(spec)
        or len(source["menus"]) != 35
        or len(source["fits"]) != 35
        or not 2 <= limit <= 512
    ):
        raise ValueError("Offline learning bank identity, shape or budget mismatch")
    menus, fits = [], []
    workspace = source.get("proposal_protocol", {}).get("workspace", "common")
    for cell, options, solutions in zip(spec["cells"], source["menus"], source["fits"]):
        if not options or len(options) != len(solutions):
            raise ValueError("Each site needs matching candidate/IK records")
        options, solutions = options[:limit], solutions[:limit]
        for action, fitted in zip(options, solutions):
            # Validate the five physical parameters and exact bucket-relative
            # mapping. Historical common-XY proposal bounds are not reused:
            # this study's source explicitly declares its enlarged workspace.
            validate_action(cell, action)
            reason = check_fitted_geometry(
                spec["scene"],
                cell,
                action,
                fitted,
                {"Arm10": env.links, "Arm14": env.links},
                workspace,
            )
            if reason is not None:
                raise ValueError(
                    f"Offline candidate violates declared geometry: {reason}"
                )
            targets = endpoint_targets(spec["scene"], cell, action)
            for name in ("Arm10", "Arm14"):
                checked = verified_fit(
                    targets[name], fitted[name], env.links, name == "Arm14"
                )
                env.ik_cache[(name, tuple(targets[name].ravel()))] = checked
        menus.append(copy.deepcopy(options))
        fits.append(copy.deepcopy(solutions))
    return source, menus, fits


def native_bank(spec, menus, fits, checks, source_path):
    from farm_learning import digest

    result = dict(
        schema="sircus-study-native-candidate-bank-v1",
        source_fingerprint=spec["fingerprint"],
        offline_source_sha256=file_hash(source_path),
        qualification="Unchanged native endpoint measured settling and dropper-position gates during development sweep; no swept-path certificate.",
        selection="All endpoint-accepted candidates, independent of release outcome/reward.",
        menus=menus,
        fits=fits,
        checks=checks,
    )
    result["fingerprint"] = digest(result)
    return result


def audit_study(run):
    """CPU-only reload of categorical sampling likelihoods and update linkage.

    This checks saved policy/bank/data identity, not an independent optimizer
    reconstruction, native contact trace or an unseen-layout evaluation.
    """
    import torch
    from farm_learning import VERSION, candidate_tensor_data, digest, make_policy
    from farm_policy import weight_hash

    run = Path(run)
    result = json.loads((run / "farm-study-result.json").read_bytes())
    contract = json.loads((run / "farm-study-contract.json").read_bytes())
    bank = json.loads((run / "native-qualified-study-bank.json").read_bytes())
    scene = json.loads((run / "contract.json").read_bytes())
    unsigned = copy.deepcopy(contract)
    expected = unsigned.pop("fingerprint")
    if (
        contract["version"] != VERSION
        or digest(unsigned) != expected
        or result["study_fingerprint"] != expected
        or result["source_fingerprint"] != contract["source_fingerprint"]
        or scene["fingerprint"] != contract["source_fingerprint"]
        or result["native_bank_sha256"]
        != file_hash(run / "native-qualified-study-bank.json")
    ):
        raise ValueError("Stale or incompatible categorical study identity")
    unsigned = copy.deepcopy(bank)
    bank_fp = unsigned.pop("fingerprint")
    if digest(unsigned) != bank_fp:
        raise ValueError("Native candidate bank checksum mismatch")
    obs, features, mask = candidate_tensor_data(
        scene["cells"], bank["menus"], scene["action_low"], scene["action_high"]
    )
    features, mask = torch.tensor(features), torch.tensor(mask)
    current = result["initial_weight_hash"]
    model = make_policy().cpu()
    max_error, samples = 0.0, 0
    batches = result["stages"]["train"]
    if (
        len(batches) != contract["training_batches"]
        or len(result["updates"]) != len(batches) // 2
    ):
        raise ValueError("Incomplete categorical training budget")
    for i, entry in enumerate(batches):
        path = run / entry["file"]
        if file_hash(path) != entry["sha256"]:
            raise ValueError("Training batch checksum mismatch")
        batch = json.loads(path.read_bytes())
        saved_path = run / batch["sampling_checkpoint"]
        saved = torch.load(saved_path, weights_only=True, map_location="cpu")
        if (
            file_hash(saved_path) != batch["sampling_checkpoint_sha256"]
            or saved["study_contract"] != contract
            or saved["native_bank_sha256"] != result["native_bank_sha256"]
            or saved["weight_hash"] != batch["policy_hash"]
            or batch["policy_hash"] != current
            or not np.array_equal(np.asarray(batch["observation"]), obs)
        ):
            raise ValueError("Sampling policy or candidate bank mismatch")
        model.load_state_dict(saved["state_dict"], strict=True)
        if weight_hash(model) != current:
            raise ValueError("Saved categorical sampling weights disagree")
        indices = torch.tensor(batch["sampled_indices"], dtype=torch.long)
        expected_actions = [bank["menus"][j][int(k)] for j, k in enumerate(indices)]
        if batch["actions"] != expected_actions or not all(batch["enabled"]):
            raise ValueError(
                "Sampling indices do not identify the executed shot parameters"
            )
        with torch.no_grad():
            logp = model.distribution(features, mask).log_prob(indices).cpu().numpy()
        recorded = np.asarray(batch["log_prob"])
        error = float(np.abs(logp - recorded).max())
        if not np.allclose(logp, recorded, atol=2e-5, rtol=2e-5):
            raise ValueError("Categorical sampling likelihood mismatch")
        max_error, samples = max(max_error, error), samples + len(indices)
        if i % 2:
            update = result["updates"][i // 2]
            if update["before"] != current or update["after"] == current:
                raise ValueError("Categorical PPO update linkage mismatch")
            current = update["after"]
    final = torch.load(
        run / "candidate-policy-final.pt", weights_only=True, map_location="cpu"
    )
    model.load_state_dict(final["state_dict"], strict=True)
    if (
        current != result["final_weight_hash"]
        or final["weight_hash"] != current
        or weight_hash(model) != current
        or samples != contract["training_throws"]
    ):
        raise ValueError("Categorical final checkpoint mismatch")
    for name, checksum in result["checkpoints"].items():
        if file_hash(run / name) != checksum:
            raise ValueError("Categorical checkpoint artifact digest mismatch")
    return dict(
        schema="sircus-categorical-study-checkpoint-audit-v1",
        study_fingerprint=expected,
        source_result_sha256=file_hash(run / "farm-study-result.json"),
        native_bank_sha256=result["native_bank_sha256"],
        batches_checked=len(batches),
        samples_checked=samples,
        updates_checked=len(result["updates"]),
        log_probability_max_abs_error=max_error,
        scope="CPU sampling likelihood and logged update-chain audit; not an independent optimizer reconstruction, full native contact audit or unseen-layout test.",
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Audit a completed candidate PPO study on CPU"
    )
    parser.add_argument("run", type=Path)
    command = parser.parse_args()
    result = audit_study(command.run)
    (command.run / "candidate-checkpoint-audit.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    print(json.dumps(result, indent=2))


def run_study(env, spec, offline_menu_path, args):
    import torch
    import expanded_farm
    from farm_contract import reward as historical_reward
    from farm_learning import (
        VERSION,
        candidate_tensor_data,
        choose_actions,
        learning_reward,
        make_policy,
        per_cell,
        ppo_update,
        study_contract,
        summary,
    )
    from farm_policy import Policy, weight_hash
    from parallel_runtime import atomic
    from settling_contract import accepted

    spec = copy.deepcopy(spec)
    source, proposals, proposal_fits = load_offline_bank(
        env, spec, offline_menu_path, args.study_sweep_limit
    )
    contract = study_contract(
        spec,
        file_hash(offline_menu_path),
        args.study_train_batches,
        args.study_eval_repeats,
        args.study_seed,
    )
    atomic(env.dest / "farm-study-contract.json", contract)
    atomic(
        env.dest / "sweep-proposal-protocol.json",
        dict(
            offline_source_sha256=file_hash(offline_menu_path),
            source_fingerprint=spec["fingerprint"],
            search_proposal_version=source.get("search_proposal_version"),
            proposal_protocol=source.get("proposal_protocol"),
            requested_candidates_per_cell=list(map(len, proposals)),
            qualification="Offline geometry and strict endpoint IK only; native scoring follows independently.",
        ),
    )
    cells = spec["cells"]
    phase_rows, phase_batches, best_actions = {}, {}, [None] * env.n
    native_menus, native_fits, native_checks = ([[] for _ in cells] for _ in range(3))
    started = time.monotonic()
    changes, train_buffer = [], []
    checkpoint_paths = {}
    native_path = env.dest / "native-qualified-study-bank.json"

    def table_report():
        reports = []
        for i, cell in enumerate(cells):
            reports.append(
                dict(
                    cell_id=cell["id"],
                    bucket_xy_m=list(cell["bucket_xy_m"]),
                    native_candidates=len(native_menus[i]),
                    stages={
                        phase: summary([r for r in rows if r["cell_id"] == cell["id"]])
                        for phase, rows in phase_rows.items()
                    },
                    best_sweep_action=best_actions[i],
                )
            )
        return reports

    def progress(phase, **extra):
        atomic(
            env.dest / "progress.json",
            dict(
                schema=VERSION,
                phase=phase,
                study_fingerprint=contract["fingerprint"],
                source_fingerprint=spec["fingerprint"],
                elapsed_s=time.monotonic() - started,
                summary={k: summary(v) for k, v in phase_rows.items()},
                per_cell=table_report(),
                updates=len(changes),
                **extra,
            ),
        )

    def batch(phase, number, actions, enabled=None, **sampling):
        if shutil.disk_usage(env.dest).free < 3 * 1024**3:
            raise RuntimeError("Less than 3 GiB free during bounded learning study")
        enabled = (
            np.ones(env.n, dtype=bool)
            if enabled is None
            else np.asarray(enabled, dtype=bool)
        )
        env.preparation_enabled = enabled.copy()
        rows, raw = env.shot_batch(actions, record_video=False, enabled=enabled)
        for row in rows:
            row["learning_reward"] = learning_reward(row)
            row["historical_reward"] = historical_reward(row)
        requested = [row for row, use in zip(rows, enabled) if use]
        phase_rows.setdefault(phase, []).extend(copy.deepcopy(requested))
        target = env.dest / "stages" / phase
        target.mkdir(parents=True, exist_ok=True)
        artifact = target / f"batch-{number:03}.json"
        atomic(
            artifact,
            dict(
                schema="sircus-candidate-study-batch-v1",
                stage=phase,
                batch=number,
                study_fingerprint=contract["fingerprint"],
                source_fingerprint=spec["fingerprint"],
                actions=actions,
                enabled=enabled.tolist(),
                rows=rows,
                physics_dt_s=env.dt,
                wall_s=raw["wall_s"],
                scope="Complete 240 Hz contact/capture scoring; direct endpoint initialization, no RGB capture, no learned joint path.",
                **sampling,
            ),
        )
        phase_batches.setdefault(phase, []).append(
            dict(
                file=str(artifact.relative_to(env.dest)),
                sha256=file_hash(artifact),
                **summary(requested),
            )
        )
        progress(phase, completed_batches=number + 1)
        print(
            json.dumps(dict(phase=phase, batch=number, **summary(requested))),
            flush=True,
        )
        return rows

    def save_policy(policy, name, optimizer=None):
        path = env.dest / name
        saved = dict(
            state_dict=policy.state_dict(),
            weight_hash=weight_hash(policy),
            study_contract=contract,
            source_fingerprint=spec["fingerprint"],
            native_bank_sha256=file_hash(native_path),
            updates=len(changes),
            policy_kind="candidate-conditioned-categorical",
        )
        if optimizer is not None:
            saved["optimizer"] = optimizer.state_dict()
        torch.save(saved, path)
        checkpoint_paths[name] = file_hash(path)
        return path

    with endpoint_mode(env):
        # This sweep is development search, not PPO or held-out evaluation.
        for j in range(max(map(len, proposals))):
            enabled = np.array([j < len(m) for m in proposals])
            actions = [m[j] if j < len(m) else m[0] for m in proposals]
            rows = batch("sweep", j, actions, enabled)
            for i, row in enumerate(rows):
                if not enabled[i]:
                    continue
                passed = (
                    not row["invalid"]
                    and accepted(row["settled"])
                    and row["settled"]["dropper_position_error_mm"] <= 2
                )
                native_checks[i].append(
                    dict(
                        action_id=actions[i]["id"],
                        accepted=bool(passed),
                        settled=row["settled"],
                        invalid=row["invalid"],
                    )
                )
                if passed:
                    native_menus[i].append(copy.deepcopy(actions[i]))
                    native_fits[i].append(copy.deepcopy(proposal_fits[i][j]))
            atomic(
                native_path,
                native_bank(
                    spec, native_menus, native_fits, native_checks, offline_menu_path
                ),
            )
        if any(not options for options in native_menus):
            progress(
                "blocked_no_native_candidate",
                missing_sites=[i for i, m in enumerate(native_menus) if not m],
            )
            raise RuntimeError(
                "At least one site has no endpoint-qualified training option"
            )
        for i, cell in enumerate(cells):
            available = [
                r
                for r in phase_rows["sweep"]
                if r["cell_id"] == cell["id"] and not r["invalid"]
            ]
            best = max(
                available,
                key=lambda r: (
                    r["route_success"],
                    r["route_quality"],
                    r["learning_reward"],
                ),
            )
            best_actions[i] = next(
                copy.deepcopy(a) for a in native_menus[i] if a["id"] == best["id"]
            )
        for repeat in range(contract["best_setting_repeats"]):
            batch("best_repeat", repeat, best_actions)

        obs_array, feature_array, mask_array = candidate_tensor_data(
            cells, native_menus, expanded_farm.LOW, expanded_farm.HIGH
        )
        obs = torch.tensor(obs_array, device=env.sim.device)
        features = torch.tensor(feature_array, device=env.sim.device)
        masks = torch.tensor(mask_array, device=env.sim.device)
        torch.manual_seed(contract["seed"])
        policy = make_policy().to(env.sim.device)
        optimizer = torch.optim.Adam(policy.parameters(), lr=contract["lr"])
        initial_hash = weight_hash(policy)
        save_policy(policy, "candidate-policy-initial.pt")

        def evaluate(model, label):
            frozen_hash = weight_hash(model)
            for deterministic, repeats in (
                (True, 1),
                (False, contract["evaluation_repeats"]),
            ):
                stage = label + ("_deterministic" if deterministic else "_stochastic")
                # Reinitialize sampling RNG for paired initial/final evaluation.
                torch.manual_seed(contract["seed"] + 10000)
                for repeat in range(repeats):
                    with torch.no_grad():
                        distribution = model.distribution(features, masks)
                        indices = (
                            distribution.probs.argmax(-1)
                            if deterministic
                            else distribution.sample()
                        )
                        logp = distribution.log_prob(indices)
                    batch(
                        stage,
                        repeat,
                        choose_actions(native_menus, indices.cpu().numpy()),
                        policy_hash=frozen_hash,
                        sampled_indices=indices.cpu().tolist(),
                        log_prob=logp.cpu().tolist(),
                    )
                if weight_hash(model) != frozen_hash:
                    raise RuntimeError("Frozen evaluation changed policy weights")

        legacy_path = getattr(args, "study_legacy_checkpoint", None)
        if legacy_path:
            legacy_path = Path(legacy_path)
            legacy_menu = legacy_path.parent / "expanded-menu-qualified.json"
            from expanded_pose_menu import load_native_menu

            old_menus, old_audit = load_native_menu(legacy_menu, spec)
            saved = torch.load(
                legacy_path, weights_only=True, map_location=env.sim.device
            )
            if (
                saved["menu_sha256"] != old_audit["file_sha256"]
                or saved["contract"]["fingerprint"] != spec["fingerprint"]
            ):
                raise ValueError("Historical checkpoint/menu contract mismatch")
            old_model = Policy(
                expanded_farm.LOW,
                expanded_farm.HIGH,
                expanded_farm.WARM,
                expanded_farm.LOGSTD,
            ).to(env.sim.device)
            old_model.load_state_dict(saved["state_dict"], strict=True)
            # Its original qualified IK fits are checked independently too.
            old_source = json.loads(legacy_menu.read_bytes())
            from expanded_pose_menu import endpoint_targets
            from expanded_pose_native import verified_fit

            for cell, options, fits in zip(
                cells, old_source["menus"], old_source["fits"]
            ):
                for action, fitted in zip(options, fits):
                    targets = endpoint_targets(spec["scene"], cell, action)
                    for name in ("Arm10", "Arm14"):
                        env.ik_cache[(name, tuple(targets[name].ravel()))] = (
                            verified_fit(
                                targets[name], fitted[name], env.links, name == "Arm14"
                            )
                        )
            for deterministic, repeats in (
                (True, 1),
                (False, contract["evaluation_repeats"]),
            ):
                stage = "legacy_deterministic" if deterministic else "legacy_stochastic"
                torch.manual_seed(contract["seed"] + 10000)
                for repeat in range(repeats):
                    with torch.no_grad():
                        dist = old_model.distribution(obs)
                        latent = dist.mean if deterministic else dist.sample()
                    batch(
                        stage,
                        repeat,
                        expanded_farm.project_actions(
                            cells, latent.cpu().numpy(), old_menus
                        ),
                        policy_hash=weight_hash(old_model),
                        historical_checkpoint_sha256=file_hash(legacy_path),
                        historical_menu_sha256=file_hash(legacy_menu),
                    )
            if weight_hash(old_model) != saved["weight_hash"]:
                raise RuntimeError("Historical frozen evaluation altered weights")

        evaluate(policy, "candidate_initial")
        torch.manual_seed(contract["seed"] + 20000)
        for number in range(contract["training_batches"]):
            policy_hash = weight_hash(policy)
            checkpoint = (
                save_policy(policy, f"candidate-policy-update-{len(changes):03}.pt")
                if not train_buffer
                else checkpoint
            )
            with torch.no_grad():
                distribution = policy.distribution(features, masks)
                indices = distribution.sample()
                logp = distribution.log_prob(indices)
            actions = choose_actions(native_menus, indices.cpu().numpy())
            rows = batch(
                "train",
                number,
                actions,
                policy_hash=policy_hash,
                sampling_checkpoint=checkpoint.name,
                sampling_checkpoint_sha256=file_hash(checkpoint),
                sampled_indices=indices.cpu().tolist(),
                log_prob=logp.cpu().tolist(),
                observation=obs_array.tolist(),
            )
            returns = torch.tensor(
                [learning_reward(r) for r in rows],
                dtype=torch.float32,
                device=env.sim.device,
            )
            train_buffer.append((obs, features, masks, indices, logp, returns))
            if len(train_buffer) == contract["batches_per_update"]:
                changes.append(
                    ppo_update(
                        policy,
                        optimizer,
                        *[torch.cat(t) for t in zip(*train_buffer)],
                        contract,
                    )
                )
                train_buffer.clear()
                atomic(env.dest / "ppo-updates.json", changes)
        final_hash = weight_hash(policy)
        if (
            train_buffer
            or len(changes) != contract["training_batches"] // 2
            or final_hash == initial_hash
        ):
            raise RuntimeError(
                "Longer study did not complete its predeclared PPO updates"
            )
        save_policy(policy, "candidate-policy-final.pt", optimizer)
        evaluate(policy, "candidate_final")

    result = dict(
        version=VERSION,
        study_fingerprint=contract["fingerprint"],
        source_fingerprint=spec["fingerprint"],
        source_python_sha256={
            p.name: file_hash(p) for p in Path(__file__).parent.glob("*.py")
        },
        elapsed_s=time.monotonic() - started,
        training_throws=contract["training_throws"],
        ppo_updates=len(changes),
        updates=changes,
        initial_weight_hash=initial_hash,
        final_weight_hash=final_hash,
        checkpoints=checkpoint_paths,
        native_bank_sha256=file_hash(native_path),
        native_candidates_per_cell=list(map(len, native_menus)),
        summary={phase: summary(rows) for phase, rows in phase_rows.items()},
        stages=phase_batches,
        per_cell=table_report(),
        bucket_coordinates_unchanged=True,
        peak_bucket_drift_m=env.peak_bucket_drift,
        scope=contract["evaluation_scope"],
        setup_scope=contract["setup_scope"],
    )
    atomic(env.dest / "farm-study-result.json", result)
    atomic(env.dest / "result.json", result)
    progress("complete", result_file="farm-study-result.json")
    return result
