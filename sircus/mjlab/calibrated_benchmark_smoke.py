"""TRAIN-only native controls and a discarded PPO update before benchmarking."""

import argparse
import json
import os
from pathlib import Path
import time

import numpy as np
import torch

from .calibrated_adapter import CalibratedAdapter as BenchmarkAdapter
from .calibrated_benchmark_contract import (
    load_prior,
    observation,
    action_from_latent,
    study_identity,
    checked_calibration,
    checked_setup_home,
    reward,
)
from .benchmark_policy import ResidualPolicy, update, weight_hash
from .corrected_contract import sha
from .gpu_slot import GPU_UUID, gpu_slot
from .learning_contract import digest
from .sweep_worker import atomic, load_reference


def run(args, *, synthetic=False):
    if not synthetic and os.environ.get("CUDA_VISIBLE_DEVICES") != GPU_UUID:
        raise RuntimeError("Only4090 may run the native smoke")
    ref = load_reference(args.reference, args.assets)
    prior = load_prior(args.prior_bundle)
    args.output.mkdir(parents=True, exist_ok=False)
    calibration = checked_calibration(args.calibration_report, args.verification_report, synthetic=synthetic)
    setup = checked_setup_home(args.setup_home)
    allrows, identity = [], study_identity(ref, calibration, setup, synthetic=synthetic)
    (args.output / "calibration.json").write_bytes(args.calibration_report.read_bytes())
    (args.output / "verification.json").write_bytes(args.verification_report.read_bytes())
    (args.output / "setup-home.json").write_bytes(args.setup_home.read_bytes())
    started = time.monotonic()
    with gpu_slot(args.gpu_lock, timeout_s=900):
        torch.set_num_threads(2)
        torch.manual_seed(20261401)
        adapter = BenchmarkAdapter(ref, args.assets, args.output / "native", batch=32,
                                   prior_records=prior["records"], calibration=calibration["score_config"], home=setup["home"])

        def resolve(bucket):
            context = dict(id="smoke-context-" + digest(bucket), bucket_xy_m=bucket)
            resolution, timing = adapter.resolve_context(context)
            hint = resolution["source_hint"]
            anchor = next((r for r in prior["records"] if r["id"] == hint.get("id")), None)
            if anchor is None or any(hint.get(k) != anchor[k] for k in ("values", "bucket_xy_m", "source_record_sha256")):
                raise ValueError("Smoke geometry resolver source changed")
            effective = resolution.get("effective_source_hint", hint)
            if effective.get("id") != anchor["id"] or effective.get("bucket_xy_m") != anchor["bucket_xy_m"]:
                raise ValueError("Smoke effective anchor lost original source identity")
            metadata = dict(source_hint=anchor, effective_source_hint=effective,
                            context_resolution_file=resolution["file"],
                            context_resolution_sha256=resolution["sha256"],
                            context_home_sha256=resolution["home_sha256"],
                            context_layout_eligible=resolution["layout_eligible"],
                            context_anchor_admitted=resolution["anchor_admitted"],
                            context_hard_reject=resolution["hard_reject"])
            return effective, metadata

        resolved_controls = [resolve(row["bucket_xy_m"]) for row in prior["records"]]
        controls = [
            dict(
                id=f"control-{row['id']}",
                bucket_xy_m=row["bucket_xy_m"],
                values=row["values"],
                source_teacher_id=resolved_controls[i][0]["id"],
                **resolved_controls[i][1],
            )
            for i, row in enumerate(prior["records"])
        ]
        rows, timing = adapter.evaluate(controls, "engineering_control")
        for row, request in zip(rows, controls):
            row.update({k: v for k, v in request.items() if k != "values"})
        allrows.extend(rows)
        controls_captured = sum(bool(r.get("native_release") and r.get("route_success") and not r.get("invalid")) for r in rows)
        atomic(args.output / "controls.json", dict(rows=rows, timing=timing))
        if len(rows) != 17 or controls_captured < 1:
            raise RuntimeError("Calibrated TRAIN-only smoke has no successful source shot; investigate before long run")
        source = prior["records"][0]
        bad = dict(
            id="deliberate-invalid-global-bounds",
            bucket_xy_m=source["bucket_xy_m"],
            values=[10.0, *source["values"][1:]],
            source_teacher_id=resolved_controls[0][0]["id"],
            **resolved_controls[0][1],
        )
        invalid, invalid_timing = adapter.evaluate([bad], "engineering_reject")
        invalid[0].update({k: v for k, v in bad.items() if k != "values"})
        allrows.extend(invalid)
        atomic(args.output / "deliberate-reject.json", dict(rows=invalid, timing=invalid_timing))
        if invalid[0].get("native_release") or not invalid[0].get("invalid"):
            raise RuntimeError("Deliberate invalid request was not charged as a reject")
        policy = ResidualPolicy()
        optimizer = torch.optim.Adam(policy.parameters(), lr=0.0003)
        buckets = [prior["records"][i % 17]["bucket_xy_m"] for i in range(32)]
        resolved_ppo = [resolve(bucket) for bucket in buckets]
        anchors = [a for a, _ in resolved_ppo]
        obs = np.asarray([observation(bucket, a) for bucket, a in zip(buckets, anchors)])
        x = torch.tensor(obs)
        with torch.no_grad():
            dist = policy.distribution(x)
            latent = dist.sample()
            logp = dist.log_prob(latent).sum(-1)
        torch.save(
            dict(policy=policy.state_dict(), optimizer=optimizer.state_dict(),
                 weight_sha256=weight_hash(policy), actor_sha256=weight_hash(policy.actor)),
            args.output / "ppo-before.pt",
        )
        requests = [
            dict(
                id=f"discarded-ppo-smoke-{i:02d}",
                bucket_xy_m=buckets[i],
                values=action_from_latent(a["values"], z).tolist(),
                source_teacher_id=a["id"],
                **resolved_ppo[i][1],
            )
            for i, (a, z) in enumerate(zip(anchors, latent.numpy()))
        ]
        rows, timing = adapter.evaluate(requests, "discarded_ppo_smoke")
        for row, request in zip(rows, requests):
            row.update({k: v for k, v in request.items() if k != "values"})
        allrows.extend(rows)
        returns = torch.tensor([reward(r) for r in rows], dtype=torch.float32)
        if (
            not any(r.get("native_release") for r in rows)
            or float(returns.max() - returns.min()) <= 1e-6
        ):
            raise RuntimeError("PPO smoke lacks native releases or nonconstant rewards")
        with torch.no_grad():
            likelihood_error = float(
                (policy.distribution(x).log_prob(latent).sum(-1) - logp).abs().max()
            )
        actor_before = weight_hash(policy.actor)
        log = update(policy, optimizer, x, latent, logp, returns)
        if actor_before == weight_hash(policy.actor) or likelihood_error > 1e-6:
            raise RuntimeError(
                "PPO actor did not update or behavior likelihood changed"
            )
        np.savez_compressed(
            args.output / "ppo-buffer.npz",
            observation=obs,
            latent=latent.numpy(),
            old_logp=logp.numpy(),
            reward=returns.numpy(),
        )
        torch.save(
            dict(policy=policy.state_dict(), optimizer=optimizer.state_dict(),
                 weight_sha256=weight_hash(policy), actor_sha256=weight_hash(policy.actor)),
            args.output / "ppo-after.pt",
        )
        atomic(
            args.output / "ppo-smoke.json",
            dict(
                rows=rows,
                timing=timing,
                update=log,
                actor_before=actor_before,
                actor_after=weight_hash(policy.actor),
                likelihood_error=likelihood_error,
            ),
        )
    report = dict(
        status="passed",
        synthetic_fixture=bool(synthetic),
        identity=identity,
        prior_sha256=digest(prior),
        requests=len(allrows),
        native_releases=sum(bool(r.get("native_release")) for r in allrows),
        controls_evaluated=17,
        controls_captured=controls_captured,
        deliberate_rejects=1,
        discarded_ppo_requests=32,
        actor_changed=True,
        behavior_logp_max_error=likelihood_error,
        elapsed_s=time.monotonic() - started,
        scope="Separate TRAIN-only engineering controls; no heldout validation/test outcomes and no model/data reused for benchmark training",
        source_sha256=sha(Path(__file__)),
    )
    atomic(args.output / "report.json", report)
    atomic(
        args.output / "manifest.json",
        {
            str(p.relative_to(args.output)): sha(p)
            for p in args.output.rglob("*")
            if p.is_file() and p.name != "manifest.json"
        },
    )
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("reference", "assets", "gpu-lock", "prior-bundle", "output", "calibration-report", "verification-report", "setup-home"):
        parser.add_argument("--" + name, type=Path, required=True)
    run(parser.parse_args())
