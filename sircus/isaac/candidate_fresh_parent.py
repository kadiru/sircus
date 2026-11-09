"""CPU forward-only parent predictions for new geometry, without ball outcomes."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import time

from candidate_continuation_inputs import PARENT_SHA256, STUDY_SHA, GEOMETRY_SHA, DOMAIN_SHA
from candidate_fresh_splits import cells_from_splits
from continuous_feasibility import LOW, HIGH, decode_action, canonical_hash
from continuous_policy import load_checkpoint, build_observations, physical_actions, weight_hash, actor_weight_hash, torch


def export_parent(split_path, checkpoint_path, *, expected_split_sha256):
    if torch is None or os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ValueError("Fresh parent export requires configured CPU Torch with CUDA hidden")
    digest = lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    if digest(split_path) != expected_split_sha256 or digest(checkpoint_path) != PARENT_SHA256:
        raise ValueError("Frozen split or parent bytes changed")
    splits = json.loads(Path(split_path).read_bytes())
    cells = cells_from_splits(splits)
    rng = torch.get_rng_state().clone()
    begin = time.monotonic()
    parent, saved = load_checkpoint(checkpoint_path, study_fingerprint=STUDY_SHA,
        geometry_fingerprint=GEOMETRY_SHA, domain_fingerprint=DOMAIN_SHA,
        expected_updates=73, expected_attempted_shots=10000, device="cpu")
    if saved["seed"] != 20261122: raise ValueError("Wrong frozen parent seed")
    before, actor = weight_hash(parent), actor_weight_hash(parent)
    parent.eval()
    actions, means, scales = {}, {}, {}
    with torch.no_grad():
        normal = parent.distribution(torch.tensor(build_observations(cells), device="cpu"))
        normalized = normal.loc.tanh()
        physical = physical_actions(normalized, LOW, HIGH)
        for cell, values, mean, scale in zip(cells, physical, normal.loc, normal.scale):
            key = cell["id"]
            actions[key] = decode_action(cell, values, "parent-"+key)
            means[key] = mean.tolist()
            scales[key] = scale.tolist()
    if weight_hash(parent) != before or actor_weight_hash(parent) != actor \
            or digest(checkpoint_path) != PARENT_SHA256 or not torch.equal(rng, torch.get_rng_state()) \
            or any(p.grad is not None for p in parent.parameters()):
        raise ValueError("CPU parent export changed state, RNG or gradients")
    names = ("candidate_fresh_parent.py", "candidate_fresh_splits.py", "continuous_policy.py",
             "cylinder_farm.py", "continuous_feasibility.py")
    result = dict(schema="sircus-fresh-heldout-parent-export-v1", seed=20261122,
        checkpoint_sha256=PARENT_SHA256, split_manifest_sha256=expected_split_sha256,
        split_fingerprint=splits["fingerprint"], parent_actions=actions,
        parent_latent_means=means, parent_latent_scales=scales,
        contexts=len(cells), weight_hash_before=before, weight_hash_after=weight_hash(parent),
        actor_hash_before=actor, actor_hash_after=actor_weight_hash(parent),
        source_sha256={n:digest(Path(__file__).with_name(n)) for n in names},
        runtime=dict(torch=torch.__version__, device="cpu", cuda_visible_devices=""),
        cpu_only=True, global_rng_unchanged=True, gradients_created=False,
        optimizer_updates=0, native_shots=0, held_out_outcome_files_read=0,
        wall_s=time.monotonic()-begin,
        scope="Frozen direct parent inference on new bucket geometry only; no outcomes, flight search, training or release.")
    result["fingerprint"] = canonical_hash(result)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--splits", type=Path, required=True)
    p.add_argument("--expected-split-sha256", required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.output.exists(): raise ValueError("Refuse to replace fresh parent predictions")
    result = export_parent(args.splits, args.checkpoint, expected_split_sha256=args.expected_split_sha256)
    with args.output.open("x") as stream: json.dump(result, stream, indent=2, allow_nan=False); stream.write("\n")
    print(json.dumps(dict(status="passed", contexts=result["contexts"], optimizer_updates=0, native_shots=0)))


if __name__ == "__main__": main()
