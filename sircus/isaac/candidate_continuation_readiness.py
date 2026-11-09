"""CPU prediction-only verification of the seed-22 candidate adapter.

There is deliberately no training/native command, optimizer or checkpoint writer.
The output describes adapter readiness, not improved capture performance.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np

from candidate_continuation_inputs import (validate_inputs, read, digest, require,
    source_hashes, PARENT_SHA256, STUDY_SHA, GEOMETRY_SHA, DOMAIN_SHA)
from candidate_continuation_policy import CandidateContinuationPolicy, PROBABILITY_LAW, torch
from candidate_continuation_losses import context_balanced_demo_nll, exact_forward_kl
from candidate_selector import preference_probabilities
from continuous_policy import load_checkpoint, build_observations, weight_hash, actor_weight_hash
from continuous_feasibility import canonical_hash

SCHEMA = "sircus-candidate-adapter-readiness-v1"
LOG_PROB_TOLERANCE = 1e-4
PARENT_MEAN_TOLERANCE = 3e-6


def check_readiness(inputs_path, checkpoint_path, *, expected_inputs_sha256):
    require(torch is not None, "Readiness requires the existing configured CPU PyTorch runtime")
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "", "Readiness is CPU-only; CUDA_VISIBLE_DEVICES must be empty")
    require(digest(inputs_path) == expected_inputs_sha256, "Compact TRAIN export bytes changed")
    value = read(inputs_path)
    supports = validate_inputs(value)
    require(digest(checkpoint_path) == PARENT_SHA256, "Frozen seed-22 checkpoint changed")
    before_rng = torch.get_rng_state().clone()
    start = time.monotonic()
    parent, saved = load_checkpoint(checkpoint_path, study_fingerprint=STUDY_SHA,
        geometry_fingerprint=GEOMETRY_SHA, domain_fingerprint=DOMAIN_SHA,
        expected_updates=73, expected_attempted_shots=10000, device="cpu")
    require(saved["seed"] == 20261122, "Wrong parent training seed")
    before_weight, before_actor = weight_hash(parent), actor_weight_hash(parent)
    adapter = CandidateContinuationPolicy(parent)
    parent.eval()
    with torch.no_grad():
        observations = torch.tensor(build_observations(value["cells"]), device="cpu")
        distribution = adapter.distribution(observations, supports)
        parameters = parent.distribution(observations)
        chosen = distribution.deterministic().tolist()
        max_error, mean_error, probability_error = 0., 0., 0.
        selected = []
        for i, (cell, support, index) in enumerate(zip(value["cells"], supports, chosen)):
            frozen = value["parent_distributions"][cell["id"]]
            rows = [dict(candidate_id=identifier, action=action)
                    for identifier, action in zip(support.candidate_ids, support.actions)]
            reference = preference_probabilities(frozen["latent_mean"], frozen["latent_scale"], rows,
                                                 frozen["exact_action"])
            actual = distribution.log_probabilities[i].cpu().numpy()
            max_error = max(max_error, float(np.max(np.abs(actual-np.asarray(reference["log_probabilities"])))))
            mean_error = max(mean_error, float(np.max(np.abs(parameters.loc[i].cpu().numpy()
                                                            -np.asarray(frozen["latent_mean"])))))
            probability_error = max(probability_error, abs(float(distribution.probabilities[i].sum())-1.))
            identifier = support.candidate_ids[index]
            require(identifier == value["reference_choices"][i]
                    == support.candidate_ids[reference["selected_index"]],
                    "Frozen selector candidate/action ranking changed at "+cell["id"])
            selected.append(dict(context_id=cell["id"], candidate_id=identifier,
                                 support_fingerprint=support.support_fingerprint))
        require(max_error <= LOG_PROB_TOLERANCE and mean_error <= PARENT_MEAN_TOLERANCE
                and probability_error <= 1e-12, "Parent CPU/source prediction tolerance exceeded")
        lookup = {cell["id"]:i for i,cell in enumerate(value["cells"])}
        demos = value["demonstrations"]
        indices = [lookup[d["context_id"]] for d in demos]
        demo_distribution = adapter.distribution(observations[indices], tuple(supports[i] for i in indices))
        teacher_nll = context_balanced_demo_nll(demo_distribution, [d["index"] for d in demos],
                                               value["successes_by_context"])
        anchor = exact_forward_kl(distribution, distribution)
        require(float(anchor) == 0., "Identity categorical KL is not zero")
        require(all(parameter.grad is None for parameter in parent.parameters()),
                "Readiness unexpectedly created parameter gradients")
    require(weight_hash(parent) == before_weight and actor_weight_hash(parent) == before_actor,
            "Readiness changed in-memory parent weights")
    require(digest(checkpoint_path) == PARENT_SHA256, "Readiness changed saved parent bytes")
    require(torch.equal(before_rng, torch.get_rng_state()), "Readiness changed global CPU RNG")
    report = dict(schema=SCHEMA, status="passed", probability_law=PROBABILITY_LAW,
        input_sha256=expected_inputs_sha256, input_fingerprint=value["fingerprint"],
        source_sha256=source_hashes(), parent_checkpoint_sha256=PARENT_SHA256,
        parent_weight_hash_before=before_weight, parent_weight_hash_after=weight_hash(parent),
        parent_actor_hash_before=before_actor, parent_actor_hash_after=actor_weight_hash(parent),
        inherited=dict(seed=20261122, requested_train_shots=10000, optimizer_updates=73),
        contexts=35, qualified_candidates=sum(len(s.candidate_ids) for s in supports),
        demonstration_records=len(demos), success_contexts=len(value["successes_by_context"]),
        contexts_without_success=35-len(value["successes_by_context"]),
        clean_demonstrations=sum(d["clean"] for d in demos), choices_matching_frozen_selector=len(selected),
        selected=selected, max_log_probability_error=max_error,
        max_parent_mean_error=mean_error, max_normalization_error=probability_error,
        tolerances=dict(log_probability=LOG_PROB_TOLERANCE, parent_mean=PARENT_MEAN_TOLERANCE),
        teacher_weighted_nll_before_updates=float(teacher_nll), identity_forward_kl=float(anchor),
        optimizer_updates=0, optimizer_steps=0, backward_calls_on_parent=0,
        native_requests=0, task_ball_releases=0, held_out_shot_files_read=0,
        global_cpu_rng_unchanged=True, gradients_created=False, training_authorized=False,
        release_authorized=False, wall_s=time.monotonic()-start,
        runtime=dict(python=__import__("platform").python_version(), torch=torch.__version__,
                     numpy=np.__version__, device="cpu", cuda_visible_devices=""),
        scope="CPU forward-only compatibility on existing TRAIN inputs; no optimizer, backward, physics, new shot or capture measurement.")
    report["fingerprint"] = canonical_hash(report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--expected-inputs-sha256", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), "Refuse to replace a readiness receipt")
    result = check_readiness(args.inputs, args.checkpoint, expected_inputs_sha256=args.expected_inputs_sha256)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({key:result[key] for key in ("status", "contexts", "qualified_candidates",
        "demonstration_records", "choices_matching_frozen_selector", "optimizer_updates", "native_requests")}))


if __name__ == "__main__":
    main()
