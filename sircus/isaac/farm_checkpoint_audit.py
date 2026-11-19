"""CPU-only audit proving filmed actions came from the stored PPO policies."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from farm_policy import Policy, weight_hash
from farm_contract import actions, observation, splits


def audit(run):
    torch.set_num_threads(2)
    report = json.loads((run / "progress.json").read_text())
    if report["phase"] != "completed":
        raise ValueError("Training is incomplete")
    n = report["contract"]["num_envs"]
    policy = Policy()
    checked = 0
    for item in report["clips"]:
        meta = json.loads((run / item["path"] / "clip.json").read_text())
        saved = torch.load(
            run / f"rollout-policy-{meta['iteration']:03d}.pt",
            map_location="cpu",
            weights_only=True,
        )
        policy.load_state_dict(saved["state_dict"])
        if (
            saved["updates"] != meta["policy_updates_applied"]
            or weight_hash(policy) != meta["weight_hash"]
        ):
            raise ValueError("Filmed batch has wrong policy checkpoint")
        x = torch.tensor(
            [row["observation"] for row in meta["rows"]], dtype=torch.float32
        )
        z = torch.tensor([row["latent"] for row in meta["rows"]], dtype=torch.float32)
        with torch.no_grad():
            logs = policy.distribution(x).log_prob(z).sum(-1).numpy()
        np.testing.assert_allclose(
            logs, [row["old_logp"] for row in meta["rows"]], rtol=1e-5, atol=1e-5
        )
        checked += len(logs)
    frozen = {}
    for label, name in [
        ("initial", "initial.pt"),
        ("final", "latest.pt"),
        ("selected", "selected.pt"),
    ]:
        saved = torch.load(run / name, map_location="cpu", weights_only=True)
        if saved["contract"]["fingerprint"] != report["contract"]["fingerprint"]:
            raise ValueError("Checkpoint contract mismatch")
        policy.load_state_dict(saved["state_dict"])
        evaluation = json.loads((run / f"test-{label}.json").read_text())
        if weight_hash(policy) != evaluation["weight_hash"]:
            raise ValueError("Test used a different checkpoint")
        x = torch.tensor(np.array([observation(c) for c in splits(n)["test"]]))
        with torch.no_grad():
            predicted = actions(policy.actor(x).numpy())
        for a, b in zip(predicted, evaluation["rows"]):
            np.testing.assert_allclose(
                [a[k] for k in ("height_m", "gap_m", "tilt_deg")],
                [b[k] for k in ("height_m", "gap_m", "tilt_deg")],
                rtol=1e-5,
                atol=1e-6,
            )
        frozen[label] = dict(
            weight_hash=weight_hash(policy), test_actions=n, updates=saved["updates"]
        )
    result = dict(
        filmed_actions_checked=checked,
        old_logprobs_match=True,
        frozen_test_checkpoints=frozen,
        numeric_tolerance="CPU/CUDA float32 logprob rtol=atol=1e-5; action rtol1e-5/atol1e-6. Physics/pose acceptance unchanged.",
    )
    (run / "checkpoint-audit.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("run", type=Path)
    print(json.dumps(audit(p.parse_args().run)))
