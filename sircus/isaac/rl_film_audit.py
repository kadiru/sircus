"""CPU-only likelihood, action and checkpoint-chain audit for the RL farm film.

Native outcome reconstruction is a separate exporter check. This module verifies
which policy sampled every attempted shot, including rejected releases, and that
assessment trials never enter the logged PPO update membership. It does not
reconstruct optimizer arithmetic or establish unseen-layout generalization.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

SCHEMA = "sircus-native-rl-film-checkpoint-audit-v1"
FILENAME = "checkpoint-audit.json"
LOGP_TOLERANCE = 2e-5
STAGES = [0, 32, 64, 128]
FILMED = list(range(8)) + list(range(60, 68)) + list(range(120, 128))
SETUP_PROTOCOL = "scripted-visual-transit-then-common-canonical-endpoint-init-v1"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def require(condition, message):
    if not condition:
        raise ValueError(message)


def canonical_hash(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, allow_nan=False).encode()
    ).hexdigest()


def signed(value, label):
    unsigned = dict(value)
    fingerprint = unsigned.pop("fingerprint", None)
    require(fingerprint == canonical_hash(unsigned), label + " fingerprint changed")
    return fingerprint


def finite(value, shape, label):
    try:
        result = np.asarray(value, dtype=np.float32)
    except (ValueError, TypeError) as error:
        raise ValueError("Invalid " + label) from error
    require(result.shape == shape and np.isfinite(result).all(), "Invalid " + label)
    return result


def safe_file(run, name, suffix):
    require(
        isinstance(name, str) and Path(name).name == name and name.endswith(suffix),
        "Unsafe evidence filename",
    )
    path = Path(run) / name
    require(path.is_file() and not path.is_symlink(), "Missing or linked evidence file")
    return path


def validate_study(study):
    """Do not infer a study budget or mix it with the historical ten batches."""
    fingerprint = signed(study, "RL study")
    expected = dict(
        schema="sircus-native-categorical-ppo-study-v1",
        training_batches=128,
        batches_per_update=2,
        assessment_stages=STAGES,
        assessment_batches=8,
        filmed_batches=FILMED,
        seed=20261110,
        assessment_seed_base=20361110,
        workcells=35,
        options_per_cell=24,
        ppo=dict(lr=0.003, epochs=4, clip=0.2, value_coef=0.5,
                 entropy_coef=0.01, grad_norm=1.0),
    )
    require(all(study.get(k) == v for k, v in expected.items()), "RL study plan changed")
    return fingerprint


def validate_indices(value, count=35, options=24):
    require(
        isinstance(value, list) and len(value) == count
        and all(type(i) is int and 0 <= i < options for i in value),
        "Invalid sampled menu categories",
    )
    return np.asarray(value, dtype=np.int64)


def diversity(trials, menus, cells):
    """Report executed menu coverage, not the actor's continuous/global limits."""
    from expanded_farm import ACTION_NAMES

    result = []
    for i, (cell, menu) in enumerate(zip(cells, menus)):
        indices = [validate_indices(t["selected_indices"])[i] for t in trials]
        counts = np.bincount(indices, minlength=len(menu))
        probabilities = counts[counts > 0] / len(indices)
        actions = [menu[int(index)] for index in indices]
        result.append(dict(
            cell_id=cell["id"], attempted_shots=len(indices),
            menu_counts=counts.tolist(), unique_selected_options=int((counts > 0).sum()),
            executed_category_entropy=float(-(probabilities * np.log(probabilities)).sum()),
            executed_ranges={k: [min(a[k] for a in actions), max(a[k] for a in actions)]
                             for k in ACTION_NAMES},
        ))
    return result


def verify_checkpoint_chain(run, source=None):
    """Verify all 128 training and 32 frozen-assessment batches on the CPU."""
    import torch
    from action_training import load_expanded_training_menu
    from expanded_farm import make_spec, validate_action
    from export_cylinder_film import matches_action
    from farm_contract import reward, statistics
    from rl_film_policy import (
        ARCHITECTURE, POLICY_VERSION, actor_weight_hash, actions_from_indices,
        build_inputs, load_checkpoint, weight_hash,
    )

    run = Path(run)
    torch.set_num_threads(2)
    spec, study, report = (read(run / name) for name in
                           ("contract.json", "study.json", "result.json"))
    geometry_hash, study_hash = signed(spec, "Original geometry"), validate_study(study)
    require(spec == make_spec(spec["scene"], 10), "Historical geometry contract changed")
    require(
        report.get("schema") == "sircus-native-rl-film-result-v1"
        and report.get("status") == "completed"
        and report.get("study_fingerprint") == study_hash
        and report.get("source_fingerprint") == study.get("geometry_fingerprint") == geometry_hash
        and report.get("workcells") == 35 and report.get("articulations") == 70
        and report.get("study") == study
        and report.get("source") == study.get("source")
        and report.get("training_attempts") == report.get("optimizer_samples") == 4480
        and report.get("setup_protocol") == study.get("setup_protocol") == SETUP_PROTOCOL
        and study.get("policy_version") == POLICY_VERSION
        and study.get("architecture") == ARCHITECTURE,
        "RL result/study/policy binding changed",
    )
    menu_path = run / "expanded-menu-qualified.json"
    menus, menu_audit = load_expanded_training_menu(menu_path, spec)
    menu_hash = digest(menu_path)
    require(
        len(menus) == 35 and all(len(menu) == 24 for menu in menus)
        and study.get("menu_sha256") == menu_hash
        and report.get("menu_sha256") == menu_hash
        and study.get("camera_sha256") == digest(run / "selected-camera.json"),
        "RL native menu or camera changed",
    )
    if source is not None:
        from presentation_recording import load_source
        from rl_film_study import make_study

        bundle = load_source(source, run / "selected-camera.json")
        require(study == make_study(bundle), "RL study does not match the frozen source")
    summaries, changes, assessments = (
        report["training_batches"], report["updates"], report["assessments"]
    )
    require(len(summaries) == 128 and len(changes) == 64 and len(assessments) == 4
            and report["filmed_batches"] == FILMED, "Incomplete RL study series")
    expected_obs, expected_candidates = build_inputs(spec["cells"], menus)
    obs = torch.from_numpy(expected_obs)
    candidates = torch.from_numpy(expected_candidates)
    checkpoint_hashes, trial_hashes, trials = {}, {}, []
    maximum_logp_error = maximum_entropy_error = 0.0
    stage_hashes, stage_actor_hashes = {}, {}
    expected_hash, expected_actor_hash = report["initial_hash"], report["initial_actor_hash"]
    initial_hash, initial_actor_hash = expected_hash, expected_actor_hash
    cache = {}

    def check_trial(trial, expected_mode, batch_count, expected_policy, expected_actor):
        nonlocal maximum_logp_error, maximum_entropy_error
        require(
            trial["schema"] == "sircus-native-rl-film-trial-v1"
            and trial["mode"] == expected_mode
            and trial["source_fingerprint"] == geometry_hash
            and trial["study_fingerprint"] == study_hash
            and trial["menu_sha256"] == menu_hash
            and trial["policy_version"] == POLICY_VERSION
            and trial["update_count_before"] == batch_count // 2
            and trial["policy_hash"] == expected_policy
            and trial["actor_hash"] == expected_actor
            and trial["setup_protocol"] == SETUP_PROTOCOL
            and trial["release_enabled"] == [True] * 35,
            "Trial policy, source, setup or release identity changed",
        )
        path = safe_file(run, trial["checkpoint_file"], ".pt")
        checkpoint_sha = digest(path)
        require(checkpoint_sha == trial["checkpoint_sha256"], "Checkpoint checksum changed")
        checkpoint_hashes[path.name] = checkpoint_sha
        if path.name not in cache:
            policy, saved = load_checkpoint(
                path, study_fingerprint=study_hash, geometry_fingerprint=geometry_hash,
                menu_sha256=menu_hash, device="cpu", expected_updates=batch_count // 2,
                expected_training_batches=batch_count,
            )
            cache[path.name] = policy, saved
        policy, saved = cache[path.name]
        require(saved["seed"] == study["seed"] and saved["updates"] == batch_count // 2
                and saved["training_batches"] == batch_count
                and weight_hash(policy) == expected_policy
                and actor_weight_hash(policy) == expected_actor,
                "Sampling checkpoint weight or count binding changed")
        actual_obs = finite(trial["observations"], (35, 5), "scene observations")
        actual_candidates = finite(trial["candidate_features"], (35, 24, 10), "candidate features")
        require(np.array_equal(actual_obs, expected_obs)
                and np.array_equal(actual_candidates, expected_candidates),
                "Policy inputs do not match fixed scene and native menu")
        indices = validate_indices(trial["selected_indices"])
        expected_actions = actions_from_indices(spec["cells"], menus, indices)
        actions, rows = trial["actions"], trial["rollout"]
        require(len(actions) == len(rows) == 35, "Incomplete trial outcomes")
        for cell, requested, action, row in zip(spec["cells"], expected_actions, actions, rows):
            require(action == requested and row["cell_id"] == cell["id"]
                    and row["id"] == requested["id"] and matches_action(row, requested)
                    and np.allclose(row["bucket_xy_m"], cell["bucket_xy_m"], atol=1e-7, rtol=0),
                    "Sampled category does not match executed native action")
            validate_action(cell, row)
        rewards = finite(trial["rewards"], (35,), "PPO rewards")
        expected_rewards = np.asarray([reward(row) for row in rows], dtype=np.float32)
        require(np.array_equal(rewards, expected_rewards), "PPO rewards differ from native outcome rule")
        logged = finite(trial["old_logp"], (35,), "stored category log probabilities")
        entropy = finite(trial["entropy"], (35,), "stored category entropy")
        with torch.no_grad():
            distribution = policy.distribution(obs, candidates)
            actual = distribution.log_prob(torch.from_numpy(indices)).numpy()
            actual_entropy = distribution.entropy().numpy()
        require(np.allclose(actual, logged, atol=LOGP_TOLERANCE, rtol=LOGP_TOLERANCE),
                "Stored category log probabilities do not match sampling checkpoint")
        require(np.allclose(actual_entropy, entropy, atol=LOGP_TOLERANCE, rtol=LOGP_TOLERANCE),
                "Stored categorical entropy does not match sampling checkpoint")
        maximum_logp_error = max(maximum_logp_error, float(np.max(np.abs(actual - logged))))
        maximum_entropy_error = max(maximum_entropy_error, float(np.max(np.abs(actual_entropy - entropy))))
        return policy, rows

    for index, summary in enumerate(summaries):
        filename = f"train-batch-{index:03d}.json"
        path = safe_file(run, filename, ".json")
        trial = read(path)
        require(summary["batch"] == trial["batch"] == index and summary["file"] == filename
                and summary["policy_hash"] == expected_hash
                and summary["file_sha256"] == digest(path)
                and summary["selected_indices"] == trial["selected_indices"],
                "Training batch identity or order changed")
        policy, rows = check_trial(trial, "train", index, expected_hash, expected_actor_hash)
        if index == 0:
            with torch.no_grad():
                probs = policy.distribution(obs, candidates).probs.numpy()
            require(np.allclose(probs, np.full((35, 24), 1 / 24), atol=1e-7, rtol=0),
                    "Initial actor must be uniform, not a reused learned policy")
            stage_hashes[0], stage_actor_hashes[0] = expected_hash, expected_actor_hash
        require(summary["optimizer_samples"] == 35, "Rejected attempts left optimizer denominator")
        require(trial["optimizer_samples"] == 35
                and trial["setup_visualization"] == ("driven" if index in FILMED else "not-recorded"),
                "Training setup visualization or optimizer membership changed")
        for key in ("attempts", "captures", "invalid"):
            require(summary[key] == statistics(rows)[key], "Training outcome counts changed")
        trial_hashes[filename] = digest(path)
        trials.append(trial)
        if index % 2:
            change = changes[index // 2]
            require(change["before"] == expected_hash
                    and change["actor_before"] == expected_actor_hash
                    and change["after"] != expected_hash
                    and change["optimizer_batches"] == [index - 1, index]
                    and change["optimizer_samples"] == change["samples"] == 70
                    and len(change["epochs"]) == study["ppo"]["epochs"],
                    "PPO update chain or optimizer membership changed")
            for epoch in change["epochs"]:
                require(all(type(value) in (int, float) and np.isfinite(value)
                            for value in epoch.values()), "Nonfinite PPO update diagnostic")
            expected_hash, expected_actor_hash = change["after"], change["actor_after"]
            if index + 1 in STAGES:
                stage_hashes[index + 1], stage_actor_hashes[index + 1] = expected_hash, expected_actor_hash
    require(expected_hash == report["final_hash"] and expected_actor_hash == report["final_actor_hash"]
            and expected_hash != initial_hash and expected_actor_hash != initial_actor_hash,
            "Final actor and complete policy must change through actual updates")
    assessment_rows = []
    for stage, summary in zip(STAGES, assessments):
        filenames = [f"assessment-{stage:03d}-{repeat:02d}.json" for repeat in range(8)]
        require(summary["after_train_batches"] == stage and summary["files"] == filenames
                and summary["update_count_before"] == stage // 2
                and summary["policy_hash"] == stage_hashes[stage],
                "Frozen assessment checkpoint or membership changed")
        rows_at_stage = []
        for repeat, filename in enumerate(filenames):
            path = safe_file(run, filename, ".json")
            trial = read(path)
            require(trial["after_train_batches"] == stage and trial["repeat"] == repeat
                    and trial["assessment_seed"] == study["assessment_seed_base"] + repeat
                    and trial["setup_visualization"] == "not-recorded",
                    "Assessment trial identity, common seed or setup changed")
            _, rows = check_trial(trial, "assessment", stage, stage_hashes[stage], stage_actor_hashes[stage])
            require(trial["optimizer_samples"] == 0,
                    "Assessment samples must never enter PPO")
            rows_at_stage.extend(rows)
            trial_hashes[filename] = digest(path)
        counts = statistics(rows_at_stage)
        require(all(summary[k] == counts[k] for k in ("attempts", "captures", "invalid"))
                and summary["attempts"] == 280, "Assessment sample budget or outcome counts changed")
        assessment_rows.append(dict(after_train_batches=stage, policy_hash=stage_hashes[stage],
                                    actor_hash=stage_actor_hashes[stage], **counts))
    final_path = safe_file(run, report["final_checkpoint_file"], ".pt")
    require(digest(final_path) == report["final_checkpoint_sha256"], "Final checkpoint checksum changed")
    final_policy, saved = load_checkpoint(
        final_path, study_fingerprint=study_hash, geometry_fingerprint=geometry_hash,
        menu_sha256=menu_hash, device="cpu", expected_updates=64, expected_training_batches=128,
    )
    require(weight_hash(final_policy) == expected_hash
            and actor_weight_hash(final_policy) == expected_actor_hash,
            "Final checkpoint does not complete PPO chain")
    checkpoint_hashes[final_path.name] = digest(final_path)
    return dict(
        schema=SCHEMA, status="passed", study_fingerprint=study_hash,
        source_fingerprint=geometry_hash, source_result_sha256=digest(run / "result.json"),
        source_study_sha256=digest(run / "study.json"),
        source_contract_sha256=digest(run / "contract.json"), menu_sha256=menu_hash,
        source_trial_sha256=trial_hashes, checkpoint_sha256=checkpoint_hashes,
        training_batches_checked=128, assessments_checked=4, assessment_batches_checked=32,
        training_actions_checked=4480, assessment_actions_checked=1120,
        optimizer_samples_checked=4480, updates_checked=64,
        final_policy_hash=expected_hash, final_actor_hash=expected_actor_hash,
        initial_policy_hash=initial_hash, initial_actor_hash=initial_actor_hash,
        category_logprob_max_abs_error=maximum_logp_error,
        categorical_entropy_max_abs_error=maximum_entropy_error,
        logprob_atol=LOGP_TOLERANCE, logprob_rtol=LOGP_TOLERANCE,
        assessments=assessment_rows,
        executed_training_diversity=diversity(trials, menus, spec["cells"]),
        checkpoint_loading="torch.load(weights_only=True, map_location='cpu')",
        scope="Checkpoint/category likelihood, action/reward and logged PPO membership/weight-chain audit. Native scoring/frames are independently audited by the exporter; optimizer arithmetic is not independently reconstructed. Assessments use the same 35 known layouts, not unseen-layout generalization.",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = verify_checkpoint_chain(args.run, args.source)
    destination = args.output or args.run / FILENAME
    with destination.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({k: result[k] for k in ("status", "training_actions_checked", "assessment_actions_checked", "updates_checked", "category_logprob_max_abs_error")}))
