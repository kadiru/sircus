"""One isolated method/seed worker for the matched current-ball benchmark."""

from __future__ import annotations

import argparse
import copy
import datetime
import json
import os
from pathlib import Path
import shutil
import time

import numpy as np
import torch

from .benchmark_adapter import BenchmarkAdapter
from .benchmark_contract import (
    VERSION,
    SCOPE,
    anchor_for,
    observation,
    action_from_latent,
    action_from_residual,
    perturbation,
    schedule,
    statistics,
    study_identity,
)
from .benchmark_policy import ResidualPolicy, GuidedSweep, update, weight_hash
from .benchmark_resources import ResourceMeter
from .corrected_contract import sha
from .gpu_slot import GPU_UUID, gpu_slot, foreign_processes
from .learning_contract import digest, reward
from .sweep_worker import atomic, load_reference


def read(path):
    return json.loads(Path(path).read_text())


def signed(value):
    value = copy.deepcopy(value)
    value.pop("fingerprint", None)
    value["fingerprint"] = digest(value)
    return value


def append(path, rows):
    with Path(path).open("a") as stream:
        for row in rows:
            stream.write(json.dumps(row, allow_nan=False) + "\n")


def run(args):
    root = Path(args.study)
    contract, prior = read(root / "contract.json"), read(root / "prior.json")
    unsigned = dict(contract)
    fingerprint = unsigned.pop("fingerprint")
    if digest(unsigned) != fingerprint or digest(prior) != contract["prior_sha256"]:
        raise ValueError("Study contract/shared prior changed")
    ref = load_reference(args.reference, args.assets)
    if study_identity(ref) != contract["identity"]:
        raise ValueError("Worker differs from the frozen native source/scene")
    protocol, pools = contract["protocol"], contract["splits"]
    if args.method not in {"sweep", "ppo"} or args.seed not in protocol["seeds"]:
        raise ValueError("Method/seed is outside the frozen contract")
    directory = root / "runs" / f"{args.method}-seed{args.seed}"
    directory.mkdir(parents=True, exist_ok=False)
    for name in ("checkpoints", "behavior", "after", "buffers", "predictions", "sweep"):
        (directory / name).mkdir()
    report = dict(
        version=VERSION,
        method=args.method,
        seed=args.seed,
        status="queued",
        phase="waiting_for_4090",
        contract_fingerprint=fingerprint,
        train_requests=0,
        train_native_releases=0,
        updates=0,
        curves=[],
        resources=None,
        selected_samples=None,
        checkpoint_sha256=None,
        final_test=None,
        scope=SCOPE,
    )
    meter = None

    def publish(phase, **extra):
        report.update(phase=phase, **extra)
        report["updated_at"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        if meter is not None:
            report["resources"] = meter.snapshot()
        atomic(directory / "report.json", signed(report))
        print(
            json.dumps(
                dict(
                    method=args.method,
                    seed=args.seed,
                    phase=phase,
                    train_requests=report["train_requests"],
                    train_native_releases=report["train_native_releases"],
                )
            ),
            flush=True,
        )

    publish("waiting_for_4090")
    try:
        with gpu_slot(
            args.gpu_lock,
            timeout_s=args.queue_timeout,
            poll_s=2,
            notify=lambda value: publish("waiting_for_4090", queue=value),
        ):
            torch.set_num_threads(2)
            torch.manual_seed(args.seed)
            np.random.seed(args.seed)
            meter = ResourceMeter(
                gpu_uuid=GPU_UUID, log_path=directory / "resource-samples.jsonl"
            )
            report.update(status="running", queue=None)
            with meter.phase("setup"):
                adapter = BenchmarkAdapter(
                    ref, args.assets, directory / "native", batch=32
                )
                policy = ResidualPolicy() if args.method == "ppo" else None
                optimizer = (
                    torch.optim.Adam(policy.parameters(), lr=0.0003)
                    if policy is not None
                    else None
                )
                sweep = (
                    GuidedSweep(args.seed, pools["train"])
                    if args.method == "sweep"
                    else None
                )
                train_schedule = schedule(args.seed, protocol["sample_budget"])
                atomic(directory / "schedule.json", train_schedule)
            training_rows, updates, curves, artifacts = [], [], [], {}
            started = time.monotonic()

            def check_limits():
                if time.monotonic() - started > args.wall_time_limit:
                    raise TimeoutError(
                        "Own finite benchmark worker exceeded its declared wall-time limit"
                    )
                if shutil.disk_usage(directory).free < 1024**3:
                    raise RuntimeError(
                        "Benchmark reached its 1GiB output-filesystem reserve; no data deleted"
                    )
                if foreign_processes():
                    raise RuntimeError(
                        "Another 4090 process appeared; leaving it untouched"
                    )

            def snapshot(samples):
                if policy is not None:
                    relative = f"checkpoints/milestone-{samples:04d}.pt"
                    torch.save(
                        dict(
                            policy=policy.state_dict(),
                            optimizer=optimizer.state_dict(),
                            method=args.method,
                            seed=args.seed,
                            samples=samples,
                            contract_fingerprint=fingerprint,
                            weight_sha256=weight_hash(policy),
                            actor_sha256=weight_hash(policy.actor),
                        ),
                        directory / relative,
                    )
                else:
                    relative = f"sweep/milestone-{samples:04d}.json"
                    atomic(
                        directory / relative,
                        dict(
                            contract_fingerprint=fingerprint,
                            method=args.method,
                            seed=args.seed,
                            samples=samples,
                            state=sweep.state(),
                        ),
                    )
                artifact = dict(
                    path=relative, sha256=sha(directory / relative), samples=samples
                )
                artifacts[samples] = artifact
                return artifact

            def evaluate(role, samples, artifact):
                contexts = pools[role]
                path = directory / artifact["path"]
                if sha(path) != artifact["sha256"]:
                    raise ValueError("Frozen artifact changed before evaluation")
                if policy is not None:
                    state = torch.load(path, map_location="cpu", weights_only=True)
                    policy.load_state_dict(state["policy"])
                    before = weight_hash(policy)
                    frozen_sweep = None
                else:
                    state = read(path)
                    frozen_sweep = GuidedSweep.from_state(state["state"])
                    before = digest(frozen_sweep.state())
                predicted, requests = [], []
                query_begin = time.perf_counter()
                for context in contexts:
                    anchor = anchor_for(context["bucket_xy_m"], prior["records"])
                    obs = observation(context["bucket_xy_m"], anchor)
                    if policy is not None:
                        with torch.no_grad():
                            latent = policy.actor(torch.tensor(obs).unsqueeze(0))[
                                0
                            ].numpy()
                        values = action_from_latent(anchor["values"], latent)
                        metadata = dict(
                            latent_mean=latent.tolist(),
                            normalized_residual=np.tanh(latent.astype(float)).tolist(),
                            source_training_ids=[],
                        )
                    else:
                        values, metadata = frozen_sweep.predict(context, anchor)
                    predicted.append(
                        dict(
                            context_id=context["id"],
                            bucket_xy_m=context["bucket_xy_m"],
                            source_teacher_id=anchor["id"],
                            command_values=values.tolist(),
                            observation=obs.tolist(),
                            artifact_sha256=artifact["sha256"],
                            **metadata,
                        )
                    )
                    for repeat in range(3):
                        noise = perturbation(context["id"], repeat)
                        requests.append(
                            dict(
                                id=f"{args.method}-{args.seed}-{role}-{samples:04d}-{context['id']}-r{repeat}",
                                context_id=context["id"],
                                role=role,
                                method=args.method,
                                seed=args.seed,
                                samples=samples,
                                repeat=repeat,
                                bucket_xy_m=context["bucket_xy_m"],
                                source_hint=anchor,
                                source_teacher_id=anchor["id"],
                                values=(values + noise).tolist(),
                                command_values=values.tolist(),
                                perturbation_mixed_units=noise.tolist(),
                                **metadata,
                                frozen_artifact_sha256=artifact["sha256"],
                            )
                        )
                query_s = time.perf_counter() - query_begin
                atomic(
                    directory / f"predictions/{role}-{samples:04d}.json",
                    dict(
                        role=role,
                        samples=samples,
                        artifact=artifact,
                        rows=predicted,
                        query_s=query_s,
                        selection_or_learning=False,
                    ),
                )
                allrows = []
                for offset in range(0, len(requests), 32):
                    check_limits()
                    batch = requests[offset : offset + 32]
                    rows, timing = adapter.evaluate(batch, role)
                    meter.add_adapter(timing, training=False)
                    for row, request in zip(rows, batch):
                        row.update(
                            {
                                k: copy.deepcopy(v)
                                for k, v in request.items()
                                if k != "values"
                            }
                        )
                        row["learning_reward"] = reward(row)
                        row["adapter_directory"] = "native"
                    append(directory / "rows.jsonl", rows)
                    allrows.extend(rows)
                after = (
                    weight_hash(policy)
                    if policy is not None
                    else digest(frozen_sweep.state())
                )
                if before != after or sha(path) != artifact["sha256"]:
                    raise RuntimeError("Frozen inference changed its model/archive")
                return dict(
                    counts=statistics(allrows),
                    query_s=query_s,
                    frozen_artifact_sha256=artifact["sha256"],
                    samples=samples,
                    prediction_count=len(predicted),
                    search_at_inference=False,
                )

            best_key, selected = None, None
            artifact = snapshot(0)
            with meter.phase("evaluation"):
                validation = evaluate("validation", 0, artifact)
            point = dict(
                method=args.method,
                seed=args.seed,
                samples=0,
                validation=validation["counts"],
                resources=meter.snapshot(),
                artifact_sha256=artifact["sha256"],
            )
            curves.append(point)
            best_key = (
                validation["counts"]["captures"],
                validation["counts"]["mean_reward"],
                0,
            )
            selected = artifact
            report["curves"] = curves
            publish("training")
            for offset in range(0, protocol["sample_budget"], 32):
                check_limits()
                with meter.phase("train"):
                    contexts = [
                        pools["train"][i] for i in train_schedule[offset : offset + 32]
                    ]
                    anchors = [
                        anchor_for(c["bucket_xy_m"], prior["records"]) for c in contexts
                    ]
                    obs = np.asarray(
                        [
                            observation(c["bucket_xy_m"], a)
                            for c, a in zip(contexts, anchors)
                        ]
                    )
                    behavior = None
                    if policy is not None:
                        x = torch.tensor(obs)
                        with torch.no_grad():
                            distribution = policy.distribution(x)
                            z = distribution.sample()
                            logp = distribution.log_prob(z).sum(-1)
                        behavior_relative = f"behavior/update-{offset // 32:04d}.pt"
                        torch.save(
                            dict(
                                policy=policy.state_dict(),
                                optimizer=optimizer.state_dict(),
                                contract_fingerprint=fingerprint,
                                samples=offset,
                                weight_sha256=weight_hash(policy),
                                actor_sha256=weight_hash(policy.actor),
                            ),
                            directory / behavior_relative,
                        )
                        behavior = dict(
                            path=behavior_relative,
                            sha256=sha(directory / behavior_relative),
                        )
                        normalized = np.tanh(z.numpy().astype(float))
                        proposed = [
                            action_from_latent(a["values"], latent)
                            for a, latent in zip(anchors, z.numpy())
                        ]
                        metadata = [
                            dict(
                                latent=latent.tolist(),
                                old_logp=float(lp),
                                behavior_checkpoint_sha256=behavior["sha256"],
                                behavior_checkpoint=behavior["path"],
                                source_training_ids=[],
                            )
                            for latent, lp in zip(z.numpy(), logp.numpy())
                        ]
                    else:
                        proposal = [
                            sweep.propose(c, a) for c, a in zip(contexts, anchors)
                        ]
                        normalized = np.asarray([p[0] for p in proposal])
                        proposed = [
                            action_from_residual(a["values"], r)
                            for a, r in zip(anchors, normalized)
                        ]
                        metadata = [p[1] for p in proposal]
                    requests = []
                    for i, (context, anchor, values, residual, meta) in enumerate(
                        zip(contexts, anchors, proposed, normalized, metadata)
                    ):
                        requests.append(
                            dict(
                                id=f"{args.method}-{args.seed}-train-{offset + i:05d}",
                                role="train",
                                method=args.method,
                                seed=args.seed,
                                train_index=offset + i,
                                samples=offset + i + 1,
                                context_id=context["id"],
                                bucket_xy_m=context["bucket_xy_m"],
                                source_hint=anchor,
                                source_teacher_id=anchor["id"],
                                observation=obs[i].tolist(),
                                values=values.tolist(),
                                command_values=values.tolist(),
                                perturbation_mixed_units=[0.0] * 5,
                                normalized_residual=residual.tolist(),
                                **meta,
                            )
                        )
                    rows, timing = adapter.evaluate(requests, "train")
                    meter.add_adapter(timing, training=True)
                    for row, request in zip(rows, requests):
                        row.update(
                            {
                                k: copy.deepcopy(v)
                                for k, v in request.items()
                                if k != "values"
                            }
                        )
                        row["learning_reward"] = reward(row)
                        row["adapter_directory"] = "native"
                    if policy is not None:
                        returns = torch.tensor(
                            [r["learning_reward"] for r in rows], dtype=torch.float32
                        )
                        buffer_relative = f"buffers/update-{offset // 32:04d}.npz"
                        np.savez_compressed(
                            directory / buffer_relative,
                            observation=obs,
                            latent=z.numpy(),
                            old_logp=logp.numpy(),
                            reward=returns.numpy(),
                            ids=np.asarray([r["id"] for r in rows]),
                        )
                        actor_before = weight_hash(policy.actor)
                        log = update(policy, optimizer, x, z, logp, returns)
                        after_relative = f"after/update-{offset // 32:04d}.pt"
                        torch.save(
                            dict(
                                policy=policy.state_dict(),
                                optimizer=optimizer.state_dict(),
                                contract_fingerprint=fingerprint,
                                samples=offset + len(rows),
                                weight_sha256=weight_hash(policy),
                                actor_sha256=weight_hash(policy.actor),
                            ),
                            directory / after_relative,
                        )
                        entry = dict(
                            update=offset // 32,
                            samples=offset + len(rows),
                            behavior=behavior,
                            buffer=buffer_relative,
                            buffer_sha256=sha(directory / buffer_relative),
                            after=dict(
                                path=after_relative,
                                sha256=sha(directory / after_relative),
                            ),
                            actor_before=actor_before,
                            actor_after=weight_hash(policy.actor),
                            train_ids=[r["id"] for r in rows],
                            log=log,
                        )
                        append(directory / "updates.jsonl", [entry])
                        updates.append(entry)
                    else:
                        for row in rows:
                            sweep.observe(row)
                    append(directory / "rows.jsonl", rows)
                    training_rows.extend(rows)
                    report.update(
                        train_requests=len(training_rows),
                        train_native_releases=sum(
                            bool(r["native_release"]) for r in training_rows
                        ),
                        updates=len(updates),
                        train_counts=statistics(training_rows),
                    )
                    append(
                        directory / "batches.jsonl",
                        [
                            dict(
                                samples=len(training_rows),
                                counts=statistics(rows),
                                adapter_timing=timing,
                            )
                        ],
                    )
                samples = len(training_rows)
                if samples in protocol["milestones"]:
                    artifact = snapshot(samples)
                    with meter.phase("evaluation"):
                        validation = evaluate("validation", samples, artifact)
                    point = dict(
                        method=args.method,
                        seed=args.seed,
                        samples=samples,
                        validation=validation["counts"],
                        resources=meter.snapshot(),
                        artifact_sha256=artifact["sha256"],
                    )
                    curves.append(point)
                    key = (
                        validation["counts"]["captures"],
                        validation["counts"]["mean_reward"],
                        -samples,
                    )
                    if key > best_key:
                        best_key, selected = key, artifact
                    report["curves"] = curves
                publish("training")
            selection = dict(
                method=args.method,
                seed=args.seed,
                selected_samples=selected["samples"],
                artifact=selected,
                validation_key=list(best_key),
                contract_fingerprint=fingerprint,
                frozen_before_test=True,
            )
            atomic(directory / "selected.json", selection)
            report.update(
                selected_samples=selected["samples"],
                checkpoint_sha256=selected["sha256"],
            )
            publish("frozen_test")
            with meter.phase("evaluation"):
                final = evaluate("test", selected["samples"], selected)
            report.update(
                status="completed",
                final_test=dict(
                    method=args.method,
                    seed=args.seed,
                    counts=final["counts"],
                    frozen_artifact_sha256=selected["sha256"],
                    query_s=final["query_s"],
                    selected_samples=selected["samples"],
                    search_at_inference=False,
                ),
            )
            meter.close()
            publish("completed")
            meter = None
            manifest = {
                str(p.relative_to(directory)): sha(p)
                for p in directory.rglob("*")
                if p.is_file() and p.name != "manifest.json"
            }
            atomic(directory / "manifest.json", manifest)
    except Exception as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        publish("failed")
        raise
    finally:
        if meter is not None:
            meter.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("study", "reference", "assets", "gpu-lock"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--method", choices=("sweep", "ppo"), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--queue-timeout", type=float, default=900)
    parser.add_argument("--wall-time-limit", type=float, default=14400)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
