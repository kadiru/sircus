"""One isolated method/seed worker for the calibrated current-ball v2 benchmark."""

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

from .calibrated_adapter import CalibratedAdapter as BenchmarkAdapter
from .calibrated_benchmark_contract import (
    VERSION,
    SCOPE,
    observation,
    action_from_latent,
    action_from_residual,
    perturbation,
    schedule,
    statistics,
    study_identity,
    reward,
    admission_summary,
)
from .benchmark_policy import ResidualPolicy, GuidedSweep, update, weight_hash
from .benchmark_resources import ResourceMeter as _BaseResourceMeter
from .corrected_contract import sha
from .gpu_slot import GPU_UUID, gpu_slot, foreign_processes
from .learning_contract import digest
from .sweep_worker import atomic, load_reference


class ResourceMeter(_BaseResourceMeter):
    """V2 adds setup-path and preprocessing costs without changing v1 evidence."""
    def __init__(self, **kwargs):
        self.path_s=self.context_resolution_s=self.admission_wall_s=self.postprocess_s=0.
        super().__init__(**kwargs)

    def add_adapter(self,timing,*,training):
        super().add_adapter(timing,training=training)
        if training:
            for key in ("path_s","context_resolution_s","postprocess_s"):
                setattr(self,key,getattr(self,key)+float(timing.get(key,0.)))
            self.admission_wall_s+=float(timing.get("wall_s",0.))

    def snapshot(self):
        result=super().snapshot()
        result.update(path_s=self.path_s,context_resolution_s=self.context_resolution_s,
                      admission_wall_s=self.admission_wall_s,postprocess_s=self.postprocess_s)
        self._record("snapshot",result)
        return result


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
    if study_identity(ref, contract["identity"]["calibration"], contract["identity"]["setup"], synthetic=contract["protocol"].get("study_kind") == "synthetic_fixture") != contract["identity"]:
        raise ValueError("Worker differs from the frozen native source/scene")
    protocol, pools = contract["protocol"], contract["splits"]
    if protocol.get("study_kind") != "synthetic_fixture" and os.environ.get("CUDA_VISIBLE_DEVICES") != GPU_UUID:
        raise RuntimeError("Only the authorized4090 UUID may execute native benchmark workers")
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
        admission=admission_summary([]),
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
                    ref, args.assets, directory / "native", batch=32,
                    prior_records=prior["records"],
                    calibration=contract["identity"]["calibration"]["score_config"],
                    home=contract["identity"]["setup"]["home"]
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

            def resolve_context(context, role):
                resolution, timing = adapter.resolve_context(context)
                if not isinstance(resolution, dict) or not isinstance(timing, dict):
                    raise ValueError("Context resolver must return evidence plus measured timing")
                hint = resolution.get("source_hint", {})
                matches = [r for r in prior["records"] if r["id"] == hint.get("id")]
                if len(matches) != 1:
                    raise ValueError("Geometry resolver selected an unknown shared source")
                anchor = copy.deepcopy(matches[0])
                for name in ("values", "bucket_xy_m"):
                    if hint.get(name) != anchor[name]:
                        raise ValueError("Geometry resolver changed its source action family")
                if hint.get("source_record_sha256") != anchor["source_record_sha256"]:
                    raise ValueError("Geometry resolver source provenance changed")
                for name in ("file", "sha256", "home_sha256"):
                    if not isinstance(resolution.get(name), str) or not resolution[name]:
                        raise ValueError("Context resolution lacks immutable proof identity")
                if any(type(resolution.get(name)) is not bool for name in ("layout_eligible", "hard_reject", "anchor_admitted")):
                    raise ValueError("Context resolution eligibility must be explicit")
                if resolution["hard_reject"] and resolution["layout_eligible"]:
                    raise ValueError("Hard rejected context cannot be layout eligible")
                effective = copy.deepcopy(resolution.get("effective_source_hint", hint))
                if effective.get("id") != anchor["id"] or effective.get("bucket_xy_m") != anchor["bucket_xy_m"]:
                    raise ValueError("Effective anchor lost its original shared source identity")
                offset = np.asarray(effective.get("derivation_offset", [0.0] * 5), dtype=float)
                values = np.asarray(effective.get("values"), dtype=float)
                if offset.shape != (5,) or values.shape != (5,) or not np.isfinite(offset).all() or not np.isfinite(values).all():
                    raise ValueError("Effective geometry anchor requires finite exact derivation")
                if not np.allclose(values, np.asarray(anchor["values"]) + offset, atol=1e-12, rtol=0):
                    raise ValueError("Effective anchor parameters differ from frozen geometry derivation")
                metadata = dict(
                    source_hint=anchor,
                    effective_source_hint=effective,
                    context_resolution_file=resolution["file"],
                    context_resolution_sha256=resolution["sha256"],
                    context_home_sha256=resolution["home_sha256"],
                    context_layout_eligible=resolution["layout_eligible"],
                    context_anchor_admitted=resolution["anchor_admitted"],
                    context_hard_reject=resolution["hard_reject"],
                )
                append(directory / "context-resolutions.jsonl", [dict(
                    context_id=context["id"], role=role, bucket_xy_m=context["bucket_xy_m"],
                    source_teacher_id=anchor["id"], **metadata,
                    timing=timing, selection_or_flight_query=False,
                )])
                return effective, metadata, timing

            def merged_timing(timing, resolutions):
                result = copy.deepcopy(timing)
                for key in ("geometry_s", "path_s", "postprocess_s", "native_init_s", "wall_s"):
                    result[key] = float(result.get(key, 0.0)) + sum(float(t.get(key, 0.0)) for t in resolutions)
                result["context_resolution_s"] = sum(float(t.get("wall_s", 0.0)) for t in resolutions)
                result["resolver_candidates"] = sum(int(t.get("resolver_candidates", 0)) for t in resolutions)
                result["resolver_cache_hits"] = sum(int(t.get("resolver_cache_hits", 0)) for t in resolutions)
                return result

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
                resolved = [resolve_context(context, role) for context in contexts]
                resolution_timing = merged_timing({}, [r[2] for r in resolved])
                meter.add_adapter(resolution_timing, training=False)
                query_begin = time.perf_counter()
                for context, (anchor, context_metadata, _) in zip(contexts, resolved):
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
                    metadata.pop("search_at_inference", None)
                    metadata.update(ball_flight_search_at_inference=False, geometry_search_at_inference=True)
                    predicted.append(
                        dict(
                            context_id=context["id"],
                            bucket_xy_m=context["bucket_xy_m"],
                            source_teacher_id=anchor["id"],
                            command_values=values.tolist(),
                            observation=obs.tolist(),
                            artifact_sha256=artifact["sha256"],
                            **context_metadata,
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
                                observation=obs.tolist(),
                                bucket_xy_m=context["bucket_xy_m"],
                                source_teacher_id=anchor["id"],
                                **context_metadata,
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
                        context_resolution_timing=resolution_timing,
                        pre_prediction_plus_policy_s=query_s+resolution_timing["wall_s"],
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
                    ball_flight_search_at_inference=False,
                    geometry_search_at_inference=True,
                    context_resolution_timing=resolution_timing,
                    pre_prediction_plus_policy_s=query_s+resolution_timing["wall_s"],
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
                    resolved = [resolve_context(c, "train") for c in contexts]
                    anchors = [r[0] for r in resolved]
                    context_metadata = [r[1] for r in resolved]
                    context_timings = [r[2] for r in resolved]
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
                                source_teacher_id=anchor["id"],
                                **context_metadata[i],
                                observation=obs[i].tolist(),
                                values=values.tolist(),
                                command_values=values.tolist(),
                                perturbation_mixed_units=[0.0] * 5,
                                normalized_residual=residual.tolist(),
                                **meta,
                            )
                        )
                    rows, timing = adapter.evaluate(requests, "train")
                    timing = merged_timing(timing, context_timings)
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
                        admission=admission_summary(training_rows),
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
                    context_resolution_timing=final["context_resolution_timing"],
                    pre_prediction_plus_policy_s=final["pre_prediction_plus_policy_s"],
                    selected_samples=selected["samples"],
                    ball_flight_search_at_inference=False,
                    geometry_search_at_inference=True,
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
