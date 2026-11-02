"""Native PPO selection over a frozen TRAIN-derived relative-shot proposal bank.

Fresh spatial splits are fixed before collecting any new rewards. Geometry-only
candidate audits do not optimize ball trajectories. The frozen test compares the
selected categorical policy, uniform-valid bank baseline and parent v2 policy.
"""

import argparse
import copy
import datetime
import fcntl
import hashlib
import importlib.metadata
import json
from pathlib import Path
import time

import numpy as np

from .learning_contract import (
    SEED,
    WARM,
    actions,
    constraints,
    digest,
    observations,
    reward,
    statistics,
)
from .learning_search import load_training_bank
from .learning_solve import _assert_gpu_free, validate_artifacts

VERSION = "sircus-mjlab-bank-study-v1"
TRAIN_SEED, VALIDATION_SEED, TEST_SEED = SEED + 303, SEED + 304, SEED + 305


def new_splits(grid):
    grid = np.asarray(grid, dtype=float)
    if grid.shape != (35, 2):
        raise ValueError("Expected the parent's frozen 35-layout training grid")
    train = np.random.default_rng(TRAIN_SEED).uniform(
        [0.1, -0.65], [1.24, 0.25], (64, 2)
    )
    validation = np.random.default_rng(VALIDATION_SEED).uniform(
        [0.1, -0.65], [1.24, 0.25], (20, 2)
    )
    test = np.random.default_rng(TEST_SEED).uniform([0.1, -0.65], [1.24, 0.25], (50, 2))
    pools = dict(
        train=np.concatenate([grid, train]).tolist(),
        validation=validation.tolist(),
        test=test.tolist(),
    )
    if len({tuple(xy) for group in pools.values() for xy in group}) != 169:
        raise RuntimeError("Spatial split overlap")
    return pools


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic(path, value):
    path = Path(path)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False))
    tmp.replace(path)


def mask_hash(mask):
    return hashlib.sha256(np.asarray(mask, dtype=np.bool_).tobytes()).hexdigest()


def unresolved_row(bucket):
    """A geometry rejection is counted, never converted into a fake release."""
    return dict(
        invalid="no_eligible_bank_endpoint",
        bucket_xy_m=list(bucket),
        board_bounce=False,
        whole_ball_entry=False,
        route_success=False,
        route_quality=0,
        success=False,
        complete_success=False,
        closest_opening_distance_m=2.0,
        max_retention_s=0.0,
        single_bounce_success=False,
        external_contact_steps=0,
        board_contact_episodes=0,
        native_release=False,
    )


def run_bank_rows(environment, bucket, values, q10, q14, eligible, *, record=False):
    """Replay exactly the audited joint branches; keep unresolved query slots.

    This shared helper performs one prescribed shot per eligible query. It does
    not generate candidates, search, change a policy, or substitute outcomes.
    """
    from .bank_geometry import fixed_endpoints

    bucket, values = np.asarray(bucket), np.asarray(values)
    q10, q14, eligible = (
        np.asarray(q10),
        np.asarray(q14),
        np.asarray(eligible, dtype=bool),
    )
    n = len(bucket)
    if (
        bucket.shape != (n, 2)
        or values.shape != (n, 5)
        or q10.shape != (n, 6)
        or q14.shape != (n, 6)
        or eligible.shape != (n,)
    ):
        raise ValueError("Inconsistent selected bank endpoint arrays")
    rows = [unresolved_row(xy) for xy in bucket]
    positions = np.flatnonzero(eligible)
    if len(positions):
        with fixed_endpoints(
            environment,
            bucket[positions],
            values[positions],
            q10[positions],
            q14[positions],
            np.ones(len(positions), dtype=bool),
        ):
            measured, _ = environment.run(
                bucket[positions],
                values[positions],
                constraints(len(positions)),
                record=record,
            )
        if len(measured) != len(positions):
            raise RuntimeError("Native bank replay changed the query count")
        for position, row in zip(positions, measured):
            row["native_release"] = not bool(row["invalid"])
            rows[position] = row
    return rows


def run(a):
    parent_report, parent_cp, parent_sha, parent_fp = validate_artifacts(
        a.parent_study, a.assets
    )
    bank = load_training_bank(a.parent_study, parent_fp)
    if not bank["templates"]:
        raise RuntimeError("No TRAIN-derived shot proposals are available")
    scene = parent_cp["contract"]["scene"]
    pools = new_splits(parent_cp["contract"]["train_layouts"])
    a.output.mkdir(parents=True, exist_ok=False)
    spec = dict(
        version=VERSION,
        seed=TRAIN_SEED,
        train_seed=TRAIN_SEED,
        validation_seed=VALIDATION_SEED,
        test_seed=TEST_SEED,
        parent_contract_fingerprint=parent_fp,
        parent_checkpoint_sha256=parent_sha,
        scene=scene,
        proposal_bank_sha256=bank["sha256"],
        train_layouts=pools["train"],
        validation_layouts=pools["validation"],
        test_layouts=pools["test"],
        batch_size=a.batch,
        ppo_updates=a.updates,
        validation_every=a.validate_every,
        test_repeats=a.test_repeats,
        actor="shared 9->64->64->1 candidate scorer; masked categorical",
        critic="query XY 2->64->64->1",
        lr=0.0003,
        epochs=6,
        clip=0.2,
        entropy_coef=0.001,
        value_coef=0.5,
        grad_norm=1.0,
        checkpoint_selection="Validation captures first, then mean shaped reward",
        initial_policy="Zero final actor layer: uniform over valid bank candidates",
        baseline_protocol="Initial uniform bank samples one candidate per test layout with fixed seed and repeats that action; selected policy and v2 use frozen deterministic predictions",
        learning="Actual on-policy terminal categorical PPO; no bootstrap/GAE; train contexts sampled uniformly from geometry-eligible subset of 99 layouts, all-invalid queries retained as unresolved in complete evaluation denominator",
        constraints="Free-position mode only. No claim for arbitrary arm position locks.",
        scope="Simulation-only privileged geometry selection among fixed relative proposals; solved/held joint endpoints; no grasp, transit, perception or physical validation",
    )
    spec["fingerprint"] = digest(spec)
    atomic(a.output / "contract.json", spec)
    atomic(a.output / "proposal-bank.json", bank)
    atomic(a.output / "splits.json", pools)
    report = dict(
        status="running",
        phase="geometry-audit",
        contract_fingerprint=spec["fingerprint"],
        proposal_bank_sha256=bank["sha256"],
        parent_checkpoint_sha256=parent_sha,
        train_episodes=0,
        native_train_releases=0,
        updates=0,
        learning_curve=[],
        frozen_evaluation=None,
        scope=spec["scope"],
        constraints=spec["constraints"],
    )
    started = time.monotonic()

    def publish(phase):
        report.update(
            phase=phase,
            elapsed_s=time.monotonic() - started,
            timestamp_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        )
        atomic(a.output / "report.json", report)
        print(
            json.dumps(
                dict(
                    phase=phase,
                    updates=report["updates"],
                    train_episodes=report["train_episodes"],
                    elapsed_s=round(report["elapsed_s"], 1),
                )
            ),
            flush=True,
        )

    from .bank_geometry import build_candidates, audit_candidates

    data, metadata = {}, None
    audit_summary = {}
    for label, layout in pools.items():
        xy = np.asarray(layout)
        values, meta = build_candidates(bank, xy, scene)
        if metadata is not None and meta != metadata:
            raise RuntimeError("Candidate IDs changed across spatial splits")
        metadata = meta
        arrays = dict(bucket=xy, values=values)
        masks, q10, q14, reasons = [], [], [], []
        summary = []
        for offset in range(0, len(xy), a.audit_chunk):
            stop = min(offset + a.audit_chunk, len(xy))
            mask, detail = audit_candidates(
                scene, a.assets, xy[offset:stop], values[offset:stop]
            )
            masks.append(mask)
            q10.append(detail["q10"])
            q14.append(detail["q14"])
            reasons.append(detail["reasons"])
            summary.append(detail["summary"])
            report["geometry_progress"] = dict(
                split=label,
                completed=stop,
                total=len(xy),
                candidates_per_query=values.shape[1],
            )
            publish("geometry-audit")
        arrays.update(
            mask=np.concatenate(masks),
            q10=np.concatenate(q10),
            q14=np.concatenate(q14),
            reasons=np.concatenate(reasons),
        )
        data[label] = arrays
        np.savez_compressed(a.output / ("geometry-" + label + ".npz"), **arrays)
        audit_summary[label] = dict(
            queries=len(xy),
            candidates_per_query=values.shape[1],
            eligible_candidates=int(arrays["mask"].sum()),
            queries_with_candidate=int(arrays["mask"].any(1).sum()),
            unresolved_queries=int((~arrays["mask"].any(1)).sum()),
            chunks=summary,
        )
        report["geometry"] = audit_summary
        atomic(a.output / "candidate-metadata.json", metadata)
        publish("geometry-audit")
    if not data["train"]["mask"].any():
        raise RuntimeError("No train query has an eligible bank endpoint")
    if a.prepare_only:
        report["status"] = "geometry_prepared"
        publish("geometry-prepared")
        return

    # CPU audits do not occupy or initialize either GPU. Native work begins only
    # after the shared lock and foreign-workload checks succeed.
    with open(a.gpu_lock, "r") as gpu_lock:
        fcntl.flock(gpu_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        _assert_gpu_free()
        import torch
        from .learning_env import ShotEnvironment
        from .learning_policy import (
            Policy as ParentPolicy,
            weight_hash as parent_weight_hash,
        )
        from .bank_policy import FEATURE_NAMES, Policy, predict, update, weight_hash

        torch.set_num_threads(2)
        torch.manual_seed(TRAIN_SEED)
        rng = np.random.default_rng(TRAIN_SEED)
        env = ShotEnvironment(scene, a.assets, a.batch)
        policy = Policy().to(env.device)
        initial = copy.deepcopy(policy.state_dict())
        current = copy.deepcopy(initial)
        optimizer = torch.optim.Adam(policy.parameters(), lr=0.0003)
        eligible_train_ids = np.flatnonzero(data["train"]["mask"].any(1))
        report["train_sampling"] = dict(
            rule="Uniform over geometry-eligible training contexts, with replacement",
            declared_contexts=len(data["train"]["bucket"]),
            eligible_contexts=len(eligible_train_ids),
            excluded_contexts=int(
                len(data["train"]["bucket"]) - len(eligible_train_ids)
            ),
        )
        records = []

        def tensors(group, ids):
            return (
                torch.tensor(
                    group["bucket"][ids], dtype=torch.float32, device=env.device
                ),
                torch.tensor(
                    group["values"][ids], dtype=torch.float32, device=env.device
                ),
                torch.tensor(group["mask"][ids], dtype=torch.bool, device=env.device),
            )

        def append_rows(rows, phase):
            offset = len(records)
            for i, row in enumerate(rows):
                row.update(id=f"bank-shot-{offset+i:07d}", phase=phase)
            records.extend(rows)
            with (a.output / "shots.jsonl").open("a") as output:
                for row in rows:
                    output.write(json.dumps(row, allow_nan=False) + "\n")
            return rows

        def bank_trial(
            group, ids, indices, phase, *, record=False, logp=None, policy_hash=None
        ):
            ids = np.asarray(ids, dtype=int)
            indices = np.asarray(indices, dtype=int)
            eligible = group["mask"][ids].any(1)
            if not group["mask"][ids[eligible], indices[eligible]].all():
                raise RuntimeError("Selected candidate is masked out")
            rows = run_bank_rows(
                env,
                group["bucket"][ids],
                group["values"][ids, indices],
                group["q10"][ids, indices],
                group["q14"][ids, indices],
                eligible,
                record=record,
            )
            for i, row in enumerate(rows):
                row.update(
                    query_index=int(ids[i]),
                    candidate_index=int(indices[i]),
                    candidate_metadata=(
                        metadata[int(indices[i])] if eligible[i] else None
                    ),
                    mask_sha256=mask_hash(group["mask"][ids[i]]),
                    eligible_bank_endpoint=bool(eligible[i]),
                    behavior_logp=(
                        float(logp[i]) if logp is not None and eligible[i] else None
                    ),
                    behavior_policy_weight_sha256=policy_hash,
                )
            return append_rows(rows, phase)

        def evaluate_bank(state, label, *, sample=False, repeats=1, record=False):
            policy.load_state_dict(state)
            group = data[label]
            picks, logps = [], []
            generator = torch.Generator(device=env.device).manual_seed(
                TEST_SEED + 17 if label == "test" else VALIDATION_SEED + 17
            )
            for offset in range(0, len(group["bucket"]), a.batch):
                ids = np.arange(offset, min(offset + a.batch, len(group["bucket"])))
                txy, tvalues, tmask = tensors(group, ids)
                prediction = predict(
                    policy, txy, tvalues, tmask, sample=sample, generator=generator
                )
                picks.extend(prediction["indices"].cpu().tolist())
                logps.extend(prediction["logp"].cpu().tolist())
            result = []
            tag = "uniform" if sample else "selected"
            for repeat in range(repeats):
                for offset in range(0, len(picks), a.batch):
                    ids = np.arange(offset, min(offset + a.batch, len(picks)))
                    rr = bank_trial(
                        group,
                        ids,
                        np.asarray(picks)[ids],
                        f"{label}-{tag}-{repeat}",
                        record=record,
                        logp=np.asarray(logps)[ids],
                        policy_hash=weight_hash(policy),
                    )
                    result.extend(rr)
            return result

        val = evaluate_bank(initial, "validation")
        selected, best_update = copy.deepcopy(initial), 0
        best_key = (statistics(val)["captures"], statistics(val)["mean_reward"])
        report["initial_validation"] = statistics(val)
        torch.save(
            dict(
                state=initial,
                contract=spec,
                policy_class="relative_bank_categorical",
                bank=bank,
                weight_sha256=weight_hash(policy),
            ),
            a.output / "update-000.pt",
        )
        publish("ppo")
        for iteration in range(a.updates):
            policy.load_state_dict(current)
            ids = rng.choice(eligible_train_ids, a.batch, replace=True)
            txy, tvalues, tmask = tensors(data["train"], ids)
            before_hash = weight_hash(policy)
            with torch.no_grad():
                distribution, eligible = policy.distribution(txy, tvalues, tmask)
                indices = distribution.sample()
                logp = distribution.log_prob(indices)
            rows = bank_trial(
                data["train"],
                ids,
                indices.cpu().numpy(),
                "train-ppo",
                logp=logp.cpu().numpy(),
                policy_hash=before_hash,
            )
            returns = torch.tensor(
                [reward(row) for row in rows], dtype=torch.float32, device=env.device
            )
            update_log = update(
                policy, optimizer, txy, tvalues, tmask, indices, logp, returns
            )
            current = copy.deepcopy(policy.state_dict())
            report["train_episodes"] += len(rows)
            report["native_train_releases"] += sum(
                row["native_release"] for row in rows
            )
            report["updates"] = iteration + 1
            curve = dict(
                update=iteration + 1, **statistics(rows), weight_change=update_log
            )
            report["learning_curve"].append(curve)
            if (iteration + 1) % a.validate_every == 0 or iteration + 1 == a.updates:
                val = evaluate_bank(current, "validation")
                curve["validation"] = statistics(val)
                key = (
                    curve["validation"]["captures"],
                    curve["validation"]["mean_reward"],
                )
                if key > best_key:
                    selected, best_update, best_key = (
                        copy.deepcopy(current),
                        iteration + 1,
                        key,
                    )
            policy.load_state_dict(current)
            torch.save(
                dict(
                    state=current,
                    contract=spec,
                    policy_class="relative_bank_categorical",
                    bank=bank,
                    update=iteration + 1,
                    weight_sha256=weight_hash(policy),
                ),
                a.output / f"update-{iteration+1:03d}.pt",
            )
            publish("ppo")
        policy.load_state_dict(selected)
        selected_hash = weight_hash(policy)
        checkpoint = dict(
            state=selected,
            contract=spec,
            policy_class="relative_bank_categorical",
            bank=bank,
            selected_update=best_update,
            weight_sha256=selected_hash,
            feature_names=list(FEATURE_NAMES),
        )
        torch.save(checkpoint, a.output / "selected.pt")
        report.update(
            selected_update=best_update,
            checkpoint_sha256=sha256(a.output / "selected.pt"),
            policy_weight_sha256=selected_hash,
        )
        publish("frozen-evaluation")
        evaluation = {}
        for label, state, sample in (
            ("uniform_valid_bank", initial, True),
            ("selected_bank", selected, False),
        ):
            rr = evaluate_bank(
                state,
                "test",
                sample=sample,
                repeats=a.test_repeats,
                record=(label == "selected_bank"),
            )
            evaluation[label] = dict(
                **statistics(rr),
                layouts=50,
                repeats=a.test_repeats,
                policy_weight_sha256=weight_hash(policy),
                action_protocol=(
                    "One frozen uniform-valid sample per layout"
                    if sample
                    else "Frozen deterministic argmax"
                ),
                search_at_inference=False,
            )
            if label == "selected_bank":
                report["test_rows"] = rr
            publish("frozen-evaluation")
        # Parent v2 is evaluated on these NEW test coordinates without changing
        # its weights. It uses its original endpoint solver, not the bank mask.
        parent = ParentPolicy().to(env.device)
        parent.load_state_dict(parent_cp["state"])
        parent.eval()
        if parent_weight_hash(parent) != parent_cp["weight_sha256"]:
            raise RuntimeError("Parent v2 policy tensor hash changed")
        parent_rows = []
        for repeat in range(a.test_repeats):
            for offset in range(0, len(pools["test"]), a.batch):
                xy = np.asarray(pools["test"][offset : offset + a.batch])
                locks = constraints(len(xy))
                with torch.no_grad():
                    z = (
                        parent.actor(
                            torch.tensor(observations(xy, locks), device=env.device)
                        )
                        .cpu()
                        .numpy()
                    )
                rr, _ = env.run(xy, actions(z), locks, record=False)
                for row in rr:
                    row["native_release"] = not bool(row["invalid"])
                parent_rows.extend(append_rows(rr, f"test-parent-v2-{repeat}"))
        evaluation["parent_v2"] = dict(
            **statistics(parent_rows),
            layouts=50,
            repeats=a.test_repeats,
            policy_weight_sha256=parent_cp["weight_sha256"],
            checkpoint_sha256=parent_sha,
            search_at_inference=False,
            action_protocol="Frozen deterministic continuous actor; original full endpoint solver/gates",
        )
        policy.load_state_dict(selected)
        if (
            weight_hash(policy) != selected_hash
            or sha256(a.parent_study / "selected.pt") != parent_sha
        ):
            raise RuntimeError("Frozen checkpoint changed during evaluation")
        report["frozen_evaluation"] = dict(
            **evaluation["selected_bank"],
            comparison=evaluation,
            trials=evaluation["selected_bank"]["attempts"],
            scope="New continuous tabletop positions, not arbitrary arm locks; nominal unperturbed repeats",
        )
        report["records"] = len(records)
        report["versions"] = {
            name: importlib.metadata.version(name)
            for name in ("mjlab", "mujoco", "mujoco-warp", "torch", "warp-lang")
        }
        report["source_sha256"] = {
            path.name: sha256(path) for path in Path(__file__).parent.glob("*.py")
        }
        report["artifacts_sha256"] = {
            path.name: sha256(path)
            for path in a.output.iterdir()
            if path.is_file() and path.name not in ("report.json",)
        }
        report["status"] = "completed"
        publish("completed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for name in ("parent-study", "assets", "output", "gpu-lock"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--updates", type=int, default=16)
    parser.add_argument("--validate-every", type=int, default=4)
    parser.add_argument("--test-repeats", type=int, default=3)
    parser.add_argument("--audit-chunk", type=int, default=4)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    if (
        not 1 <= args.batch <= 256
        or not 1 <= args.updates <= 128
        or not 1 <= args.validate_every <= args.updates
        or not 1 <= args.test_repeats <= 10
        or not 1 <= args.audit_chunk <= 16
    ):
        parser.error("Finite positive study/audit budget required")
    run(args)
