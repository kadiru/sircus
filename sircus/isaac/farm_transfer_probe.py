"""Frozen learned-policy transfer to new candidate shots at seen bucket sites.

Qualification uses unchanged native endpoint/settling gates. Every accepted
candidate remains available regardless of its observed shot outcome. No policy
updates, reward optimization, rendering or physical robot access occur here.
"""

import copy
import json
import shutil
import time
from pathlib import Path

import numpy as np

VERSION = "sircus-frozen-candidate-transfer-v1"


def validate_checkpoint(saved, spec):
    from farm_learning import FEATURE_NAMES, VERSION as LEARNING_VERSION, digest

    contract = saved.get("study_contract", {})
    unsigned = copy.deepcopy(contract)
    fingerprint = unsigned.pop("fingerprint", None)
    if (
        digest(unsigned) != fingerprint
        or saved.get("policy_kind") != "candidate-conditioned-categorical"
        or saved.get("source_fingerprint") != spec["fingerprint"]
        or contract.get("source_fingerprint") != spec["fingerprint"]
        or contract.get("version") != LEARNING_VERSION
        or contract.get("candidate_feature_names") != list(FEATURE_NAMES)
        or contract.get("actor_input_dim") != 11
        or contract.get("observation_dim") != 5
        or contract.get("num_envs") != 35
        or not isinstance(saved.get("native_bank_sha256"), str)
        or len(saved["native_bank_sha256"]) != 64
        or not contract.get("training_batches")
        or saved.get("updates") != contract["training_batches"] // 2
    ):
        raise ValueError(
            "Transfer requires a completed matching categorical checkpoint"
        )
    return contract


def run_transfer_probe(env, spec, offlinepath, args):
    import torch
    import expanded_farm
    from farm_learning import (
        candidate_tensor_data,
        choose_actions,
        digest,
        learning_reward,
        make_policy,
        per_cell,
        summary,
    )
    from farm_contract import reward as historical_reward
    from farm_learning_runtime import (
        endpoint_mode,
        file_hash,
        load_offline_bank,
        native_bank,
    )
    from farm_policy import weight_hash
    from parallel_runtime import atomic
    from settling_contract import accepted

    spec = copy.deepcopy(spec)
    source_path = Path(args.transfer_probe_checkpoint)
    source = torch.load(source_path, map_location="cpu", weights_only=True)
    learned_contract = validate_checkpoint(source, spec)
    repeats = int(args.study_eval_repeats)
    if not 2 <= repeats <= 64:
        raise ValueError("Transfer evaluation repeats must be2..64")
    _, proposals, proposal_fits = load_offline_bank(
        env, spec, offlinepath, args.study_sweep_limit
    )
    contract = dict(
        version=VERSION,
        source_fingerprint=spec["fingerprint"],
        source_checkpoint_sha256=file_hash(source_path),
        source_policy_weight_hash=source["weight_hash"],
        source_learning_fingerprint=learned_contract["fingerprint"],
        source_native_bank_sha256=source["native_bank_sha256"],
        offline_probe_bank_sha256=file_hash(offlinepath),
        evaluation_repeats=repeats,
        seed=int(args.study_seed) + 30000,
        policy_updates=0,
        menu_selection="Every native endpoint-accepted proposal, without release-outcome/reward filtering.",
        scope="Frozen policy ranking on new candidate shot parameters at the same35seen bucket coordinates; not unseen-layout generalization. Qualification and direct initialized shots do not certify driven arm paths.",
    )
    contract["fingerprint"] = digest(contract)
    atomic(env.dest / "transfer-probe-contract.json", contract)
    shutil.copyfile(source_path, env.dest / "source-policy.pt")
    menus, fits, checks = ([[] for _ in range(35)] for _ in range(3))
    stage_rows, stages = {}, {}
    started = time.monotonic()
    bank_path = env.dest / "native-qualified-transfer-bank.json"

    def progress(phase):
        atomic(
            env.dest / "progress.json",
            dict(
                version=VERSION,
                phase=phase,
                fingerprint=contract["fingerprint"],
                source_fingerprint=spec["fingerprint"],
                elapsed_s=time.monotonic() - started,
                summary={k: summary(v) for k, v in stage_rows.items()},
                per_cell={k: per_cell(spec["cells"], v) for k, v in stage_rows.items()},
            ),
        )

    def batch(stage, number, actions, enabled=None, **sampling):
        if shutil.disk_usage(env.dest).free < 3 * 1024**3:
            raise RuntimeError("Less than3GiB free during frozen transfer probe")
        enabled = (
            np.ones(35, dtype=bool)
            if enabled is None
            else np.asarray(enabled, dtype=bool)
        )
        env.preparation_enabled = enabled.copy()
        rows, raw = env.shot_batch(actions, record_video=False, enabled=enabled)
        for row in rows:
            row.update(
                learning_reward=learning_reward(row),
                historical_reward=historical_reward(row),
            )
        requested = [row for row, use in zip(rows, enabled) if use]
        stage_rows.setdefault(stage, []).extend(copy.deepcopy(requested))
        folder = env.dest / "stages" / stage
        folder.mkdir(parents=True, exist_ok=True)
        target = folder / f"batch-{number:03}.json"
        atomic(
            target,
            dict(
                version=VERSION,
                stage=stage,
                batch=number,
                transfer_fingerprint=contract["fingerprint"],
                actions=actions,
                enabled=enabled.tolist(),
                rows=rows,
                physics_dt_s=env.dt,
                wall_s=raw["wall_s"],
                **sampling,
            ),
        )
        stages.setdefault(stage, []).append(
            dict(
                file=str(target.relative_to(env.dest)),
                sha256=file_hash(target),
                **summary(requested),
            )
        )
        progress(stage)
        print(
            json.dumps(dict(stage=stage, batch=number, **summary(requested))),
            flush=True,
        )
        return rows

    with endpoint_mode(env):
        for j in range(max(map(len, proposals))):
            enabled = np.array([j < len(m) for m in proposals])
            actions = [m[j] if j < len(m) else m[0] for m in proposals]
            rows = batch("probe_sweep", j, actions, enabled)
            for i, row in enumerate(rows):
                if not enabled[i]:
                    continue
                passed = (
                    not row["invalid"]
                    and accepted(row["settled"])
                    and row["settled"]["dropper_position_error_mm"] <= 2
                )
                checks[i].append(
                    dict(
                        action_id=actions[i]["id"],
                        accepted=bool(passed),
                        settled=row["settled"],
                        invalid=row["invalid"],
                    )
                )
                if passed:
                    menus[i].append(copy.deepcopy(actions[i]))
                    fits[i].append(copy.deepcopy(proposal_fits[i][j]))
            atomic(bank_path, native_bank(spec, menus, fits, checks, offlinepath))
        if any(not m for m in menus):
            raise RuntimeError(
                "Frozen transfer cannot run with an empty native candidate bank"
            )
        _, features, mask = candidate_tensor_data(
            spec["cells"], menus, expanded_farm.LOW, expanded_farm.HIGH
        )
        features = torch.tensor(features, device=env.sim.device)
        mask = torch.tensor(mask, device=env.sim.device)
        policy = make_policy().to(env.sim.device)
        policy.load_state_dict(source["state_dict"], strict=True)
        policy.eval()
        frozen = weight_hash(policy)
        if frozen != source["weight_hash"]:
            raise ValueError("Frozen transfer checkpoint weights disagree")
        for deterministic, count in ((True, 1), (False, repeats)):
            stage = "transfer_deterministic" if deterministic else "transfer_stochastic"
            torch.manual_seed(contract["seed"])
            for repeat in range(count):
                with torch.no_grad():
                    distribution = policy.distribution(features, mask)
                    indices = (
                        distribution.probs.argmax(-1)
                        if deterministic
                        else distribution.sample()
                    )
                    logp = distribution.log_prob(indices)
                batch(
                    stage,
                    repeat,
                    choose_actions(menus, indices.cpu().numpy()),
                    sampled_indices=indices.cpu().tolist(),
                    log_prob=logp.cpu().tolist(),
                    policy_hash=frozen,
                    source_checkpoint_sha256=contract["source_checkpoint_sha256"],
                    source_native_bank_sha256=source["native_bank_sha256"],
                    transfer_native_bank_sha256=file_hash(bank_path),
                )
                if weight_hash(policy) != frozen:
                    raise RuntimeError("Frozen transfer changed policy weights")
    result = dict(
        version=VERSION,
        fingerprint=contract["fingerprint"],
        parent_study_fingerprint=learned_contract["fingerprint"],
        source_learning_fingerprint=learned_contract["fingerprint"],
        source_fingerprint=spec["fingerprint"],
        source_checkpoint_sha256=contract["source_checkpoint_sha256"],
        source_policy_weight_hash=frozen,
        source_native_bank_sha256=source["native_bank_sha256"],
        transfer_native_bank_sha256=file_hash(bank_path),
        policy_updates=0,
        native_candidates_per_cell=list(map(len, menus)),
        summary={k: summary(v) for k, v in stage_rows.items()},
        per_cell={k: per_cell(spec["cells"], v) for k, v in stage_rows.items()},
        stages=stages,
        elapsed_s=time.monotonic() - started,
        source_python_sha256={
            p.name: file_hash(p) for p in Path(__file__).parent.glob("*.py")
        },
        peak_bucket_drift_m=env.peak_bucket_drift,
        scope=contract["scope"],
    )
    atomic(env.dest / "transfer-probe-result.json", result)
    progress("complete")
    return result


def audit_transfer_probe(run):
    import torch
    from farm_learning import (
        candidate_tensor_data,
        choose_actions,
        digest,
        make_policy,
        per_cell,
        summary,
    )
    from farm_learning_runtime import file_hash
    from farm_policy import weight_hash

    run = Path(run)
    result = json.loads((run / "transfer-probe-result.json").read_bytes())
    contract = json.loads((run / "transfer-probe-contract.json").read_bytes())
    spec = json.loads((run / "contract.json").read_bytes())
    bank = json.loads((run / "native-qualified-transfer-bank.json").read_bytes())
    unsigned = copy.deepcopy(contract)
    fingerprint = unsigned.pop("fingerprint")
    if (
        digest(unsigned) != fingerprint
        or fingerprint != result["fingerprint"]
        or contract["version"] != VERSION
        or result["policy_updates"] != 0
    ):
        raise ValueError("Frozen transfer identity or zero-update contract mismatch")
    for key in (
        "source_fingerprint",
        "source_checkpoint_sha256",
        "source_policy_weight_hash",
        "source_native_bank_sha256",
    ):
        if result.get(key) != contract.get(key):
            raise ValueError("Frozen transfer result/contract provenance mismatch")
    if result["source_learning_fingerprint"] != contract["source_learning_fingerprint"]:
        raise ValueError("Frozen transfer source learning identity mismatch")
    if (
        result["source_fingerprint"] != spec["fingerprint"]
        or result["source_fingerprint"] != contract["source_fingerprint"]
        or result["parent_study_fingerprint"] != contract["source_learning_fingerprint"]
    ):
        raise ValueError("Frozen transfer parent study/scene identity mismatch")
    if (
        file_hash(run / "source-policy.pt") != contract["source_checkpoint_sha256"]
        or file_hash(run / "native-qualified-transfer-bank.json")
        != result["transfer_native_bank_sha256"]
    ):
        raise ValueError("Frozen transfer checkpoint/bank digest mismatch")
    saved = torch.load(run / "source-policy.pt", map_location="cpu", weights_only=True)
    validate_checkpoint(saved, spec)
    if (
        saved["study_contract"]["fingerprint"] != result["parent_study_fingerprint"]
        or saved["native_bank_sha256"] != contract["source_native_bank_sha256"]
    ):
        raise ValueError("Frozen transfer original bank identity mismatch")
    model = make_policy().cpu()
    model.load_state_dict(saved["state_dict"], strict=True)
    if (
        weight_hash(model) != result["source_policy_weight_hash"]
        or weight_hash(model) != saved["weight_hash"]
    ):
        raise ValueError("Frozen transfer weights changed")
    _, features, mask = candidate_tensor_data(
        spec["cells"], bank["menus"], spec["action_low"], spec["action_high"]
    )
    features, mask = torch.tensor(features), torch.tensor(mask)
    unsigned = copy.deepcopy(bank)
    bankfp = unsigned.pop("fingerprint")
    if digest(unsigned) != bankfp or bank["source_fingerprint"] != spec["fingerprint"]:
        raise ValueError("Frozen transfer native bank identity mismatch")
    if (
        set(result["stages"])
        != {"probe_sweep", "transfer_deterministic", "transfer_stochastic"}
        or len(result["stages"]["transfer_deterministic"]) != 1
        or len(result["stages"]["transfer_stochastic"])
        != contract["evaluation_repeats"]
    ):
        raise ValueError("Frozen transfer stage budgets incomplete")
    count, error = 0, 0.0
    for stage, entries in result["stages"].items():
        rows = []
        for entry in entries:
            path = run / entry["file"]
            if file_hash(path) != entry["sha256"]:
                raise ValueError("Frozen transfer batch checksum mismatch")
            batch = json.loads(path.read_bytes())
            if batch["transfer_fingerprint"] != fingerprint:
                raise ValueError("Frozen transfer batch identity mismatch")
            requested = [r for r, use in zip(batch["rows"], batch["enabled"]) if use]
            rows.extend(requested)
            if summary(requested) != {k: entry[k] for k in summary(requested)}:
                raise ValueError("Frozen transfer batch counts disagree")
            if stage == "probe_sweep":
                continue
            if (
                batch["source_checkpoint_sha256"]
                != contract["source_checkpoint_sha256"]
                or batch["source_native_bank_sha256"]
                != contract["source_native_bank_sha256"]
                or batch["transfer_native_bank_sha256"]
                != result["transfer_native_bank_sha256"]
            ):
                raise ValueError("Frozen transfer batch policy/bank identity mismatch")
            indices = np.asarray(batch["sampled_indices"])
            if (
                batch["policy_hash"] != saved["weight_hash"]
                or choose_actions(bank["menus"], indices) != batch["actions"]
                or not all(batch["enabled"])
            ):
                raise ValueError("Frozen transfer sampled actions differ")
            with torch.no_grad():
                distribution = model.distribution(features, mask)
                logp = distribution.log_prob(torch.tensor(indices)).numpy()
            observed = np.asarray(batch["log_prob"])
            error = max(error, float(np.abs(logp - observed).max()))
            if not np.allclose(logp, observed, atol=2e-5, rtol=2e-5):
                raise ValueError("Frozen transfer likelihood mismatch")
            if stage == "transfer_deterministic" and not np.allclose(
                logp, distribution.logits.max(-1).values.numpy(), atol=2e-5, rtol=2e-5
            ):
                raise ValueError("Frozen transfer greedy choice mismatch")
            count += len(indices)
        if (
            summary(rows) != result["summary"][stage]
            or per_cell(spec["cells"], rows) != result["per_cell"][stage]
        ):
            raise ValueError("Frozen transfer stage/per-cell counts disagree")
    if count != 35 * (contract["evaluation_repeats"] + 1):
        raise ValueError("Frozen transfer evaluation budget incomplete")
    return dict(
        version="sircus-frozen-candidate-transfer-audit-v1",
        source_result_sha256=file_hash(run / "transfer-probe-result.json"),
        fingerprint=fingerprint,
        source_checkpoint_sha256=contract["source_checkpoint_sha256"],
        source_native_bank_sha256=contract["source_native_bank_sha256"],
        transfer_native_bank_sha256=result["transfer_native_bank_sha256"],
        frozen_evaluation_samples_checked=count,
        log_probability_max_abs_error=error,
        policy_updates=0,
        scope="CPU artifact, frozen-weight, categorical likelihood and recorded scoring-summary audit; no full contact replay or unseen-layout generalization.",
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Audit a frozen candidate transfer probe on CPU"
    )
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    audited = audit_transfer_probe(args.run)
    (args.run / "transfer-probe-audit.json").write_text(
        json.dumps(audited, indent=2) + "\n"
    )
    print(json.dumps(audited, indent=2))
