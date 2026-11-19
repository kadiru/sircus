"""TRAIN-only native controls and a discarded PPO update before benchmarking."""

import argparse
import json
import os
from pathlib import Path
import time

import numpy as np
import torch

from .benchmark_adapter import BenchmarkAdapter
from .benchmark_contract import (
    load_prior,
    observation,
    action_from_latent,
    study_identity,
)
from .benchmark_policy import ResidualPolicy, update, weight_hash
from .corrected_contract import sha
from .gpu_slot import GPU_UUID, gpu_slot
from .learning_contract import digest, reward
from .sweep_worker import atomic, load_reference


def run(args):
    if os.environ.get("CUDA_VISIBLE_DEVICES") != GPU_UUID:
        raise RuntimeError("Only4090 may run the native smoke")
    ref = load_reference(args.reference, args.assets)
    prior = load_prior(args.prior_bundle)
    args.output.mkdir(parents=True, exist_ok=False)
    allrows, identity = [], study_identity(ref)
    started = time.monotonic()
    with gpu_slot(args.gpu_lock, timeout_s=900):
        torch.set_num_threads(2)
        torch.manual_seed(20261301)
        adapter = BenchmarkAdapter(ref, args.assets, args.output / "native", batch=32)
        controls = [
            dict(
                id=f"control-{row['id']}",
                bucket_xy_m=row["bucket_xy_m"],
                values=row["values"],
                source_hint=row,
            )
            for row in prior["records"]
        ]
        rows, timing = adapter.evaluate(controls, "engineering_control")
        allrows.extend(rows)
        atomic(args.output / "controls.json", dict(rows=rows, timing=timing))
        if len(rows) != 17 or not all(
            r.get("native_release") and r.get("route_success") and not r.get("invalid")
            for r in rows
        ):
            raise RuntimeError(
                "One or more exact historical native controls failed; benchmark not started"
            )
        source = prior["records"][0]
        bad = dict(
            id="deliberate-invalid-global-bounds",
            bucket_xy_m=source["bucket_xy_m"],
            values=[10.0, *source["values"][1:]],
            source_hint=source,
        )
        invalid, _ = adapter.evaluate([bad], "engineering_reject")
        allrows.extend(invalid)
        if invalid[0].get("native_release") or not invalid[0].get("invalid"):
            raise RuntimeError("Deliberate invalid request was not charged as a reject")
        policy = ResidualPolicy()
        optimizer = torch.optim.Adam(policy.parameters(), lr=0.0003)
        anchors = [prior["records"][i % 17] for i in range(32)]
        obs = np.asarray([observation(a["bucket_xy_m"], a) for a in anchors])
        x = torch.tensor(obs)
        with torch.no_grad():
            dist = policy.distribution(x)
            latent = dist.sample()
            logp = dist.log_prob(latent).sum(-1)
        torch.save(
            dict(policy=policy.state_dict(), optimizer=optimizer.state_dict()),
            args.output / "ppo-before.pt",
        )
        requests = [
            dict(
                id=f"discarded-ppo-smoke-{i:02d}",
                bucket_xy_m=a["bucket_xy_m"],
                values=action_from_latent(a["values"], z).tolist(),
                source_hint=a,
            )
            for i, (a, z) in enumerate(zip(anchors, latent.numpy()))
        ]
        rows, timing = adapter.evaluate(requests, "discarded_ppo_smoke")
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
            dict(policy=policy.state_dict(), optimizer=optimizer.state_dict()),
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
        identity=identity,
        prior_sha256=digest(prior),
        requests=len(allrows),
        native_releases=sum(bool(r.get("native_release")) for r in allrows),
        controls_captured=17,
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
    for name in ("reference", "assets", "gpu-lock", "prior-bundle", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    run(parser.parse_args())
