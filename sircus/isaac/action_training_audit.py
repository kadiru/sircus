"""CPU-only verification of the policies that sampled a fresh training film.

This audits checkpoints and latent likelihoods, not held-out policy performance.
It never starts Isaac, accesses CUDA or changes the saved training records.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

SCHEMA = "sircus-action-training-checkpoint-audit-v1"
FILENAME = "checkpoint-audit.json"
LOGP_TOLERANCE = 2e-5


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def finite(value, shape, label):
    array = np.asarray(value, dtype=np.float32)
    require(array.shape == shape and np.isfinite(array).all(), f"Invalid {label}")
    return array


def verify_checkpoint_chain(run):
    """Return a bound audit without writing files; all tensors load on the CPU."""
    import torch
    from cylinder_farm import HIGH, LOGSTD, LOW, WARM, observation
    from farm_policy import Policy, weight_hash

    run = Path(run)
    torch.set_num_threads(2)
    contract, report = read(run / "contract.json"), read(run / "result.json")
    unsigned = dict(contract)
    fingerprint = unsigned.pop("fingerprint")
    require(
        hashlib.sha256(json.dumps(unsigned, sort_keys=True).encode()).hexdigest()
        == fingerprint
        == report["source_fingerprint"]
        == report["fingerprint"],
        "Checkpoint audit contract fingerprint mismatch",
    )
    require(
        report.get("schema") == "sircus-action-training-film-v1"
        and contract["training_batches"] == 10
        and contract["num_envs"] == report["workcells"] == 35
        and report["articulations"] == 70
        and contract["batches_per_update"] == 2
        and len(report["batches"]) == 10
        and len(report["updates"]) == 5,
        "Incomplete fixed-camera training checkpoint series",
    )
    policy = Policy(LOW, HIGH, WARM, LOGSTD).cpu()
    expected_obs = np.asarray(
        [observation(cell) for cell in contract["cells"]], dtype=np.float32
    )
    require(expected_obs.shape == (35, 5), "Invalid checkpoint scene observations")
    expected_hash = report["initial_hash"]
    menu_hash = report["menu_sha256"]
    maximum_error = 0.0
    clips, checkpoints, hashes = {}, {}, []
    for index, summary in enumerate(report["batches"]):
        clip_name, checkpoint_name = f"clip-{index:02}.json", f"policy-{index:02}.pt"
        clip = read(run / clip_name)
        require(
            summary["batch"] == clip["batch"] == index
            and summary["file"] == clip_name
            and clip["checkpoint_file"] == checkpoint_name
            and clip["source_fingerprint"] == fingerprint
            and clip["menu_sha256"] == menu_hash,
            "Checkpoint batch identity mismatch",
        )
        checkpoint_path = run / checkpoint_name
        checkpoint_sha = digest(checkpoint_path)
        require(
            checkpoint_sha == clip["checkpoint_sha256"],
            "Sampling checkpoint checksum mismatch",
        )
        saved = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        require(
            saved["source_fingerprint"] == fingerprint
            and saved["menu_sha256"] == menu_hash
            and saved["updates"] == index // 2,
            "Sampling checkpoint metadata mismatch",
        )
        policy.load_state_dict(saved["state_dict"], strict=True)
        actual_hash = weight_hash(policy)
        require(
            actual_hash
            == saved["weight_hash"]
            == clip["policy_hash"]
            == summary["policy_hash"]
            == expected_hash,
            "Sampling policy weight hash or update chain mismatch",
        )
        observations = finite(clip["observation"], (35, 5), "policy observations")
        require(
            np.array_equal(observations, expected_obs),
            "Policy observations do not match the contracted scene",
        )
        latents = finite(clip["latent"], (35, 4), "sampled latents")
        logged = finite(clip["old_logp"], (35,), "stored log probabilities")
        with torch.no_grad():
            actual = (
                policy.distribution(torch.from_numpy(observations))
                .log_prob(torch.from_numpy(latents))
                .sum(-1)
                .numpy()
            )
        require(
            np.isfinite(actual).all()
            and np.allclose(actual, logged, atol=LOGP_TOLERANCE, rtol=LOGP_TOLERANCE),
            "Stored latent log probabilities do not match the sampling policy",
        )
        maximum_error = max(maximum_error, float(np.max(np.abs(actual - logged))))
        clips[clip_name], checkpoints[checkpoint_name] = (
            digest(run / clip_name),
            checkpoint_sha,
        )
        hashes.append(actual_hash)
        if index % 2:
            change = report["updates"][index // 2]
            require(
                change["before"] == expected_hash
                and change["after"] != expected_hash
                and len(change["epochs"]) == contract["epochs"],
                "PPO update chain mismatch",
            )
            expected_hash = change["after"]
    final_path = run / "policy-final.pt"
    checkpoints[final_path.name] = digest(final_path)
    require(
        checkpoints[final_path.name] == report["final_checkpoint_sha256"],
        "Final checkpoint checksum mismatch",
    )
    saved = torch.load(final_path, map_location="cpu", weights_only=True)
    require(
        json.dumps(saved["contract"], sort_keys=True)
        == json.dumps(contract, sort_keys=True)
        and saved["menu_sha256"] == menu_hash,
        "Final checkpoint contract mismatch",
    )
    policy.load_state_dict(saved["state_dict"], strict=True)
    final_hash = weight_hash(policy)
    require(
        final_hash == saved["weight_hash"] == report["final_hash"] == expected_hash
        and final_hash != report["initial_hash"],
        "Final policy weights do not complete the update chain",
    )
    return dict(
        schema=SCHEMA,
        source_result_sha256=digest(run / "result.json"),
        source_contract_sha256=digest(run / "contract.json"),
        source_clip_sha256=clips,
        checkpoint_sha256=checkpoints,
        sampled_policy_hashes=hashes,
        final_policy_hash=final_hash,
        batches_checked=10,
        latent_actions_checked=350,
        updates_checked=5,
        latent_logprob_max_abs_error=maximum_error,
        logprob_atol=LOGP_TOLERANCE,
        logprob_rtol=LOGP_TOLERANCE,
        checkpoint_loading="torch.load(weights_only=True, map_location='cpu')",
        scope="Sampling checkpoint, latent likelihood and logged update-chain audit; not a held-out evaluation or independent optimizer reconstruction.",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = verify_checkpoint_chain(args.run)
    output = args.output or args.run / FILENAME
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
