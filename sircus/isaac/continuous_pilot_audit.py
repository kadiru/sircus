"""Independent CPU audit of continuous PPO inputs, likelihoods and membership.

Native telemetry/contact rescoring is a separate report check. This audit proves
which checkpoint generated each attempted shot and verifies exclusion of held-out
layouts from every logged PPO update; it does not replay optimizer arithmetic.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

SCHEMA = "sircus-continuous-pilot-checkpoint-audit-v1"
LOGP_TOLERANCE = 3e-5


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def signed(value, label):
    unsigned = dict(value)
    fingerprint = unsigned.pop("fingerprint", None)
    require(fingerprint == canonical_hash(unsigned), label + " fingerprint changed")
    return fingerprint


def finite(value, shape, label):
    try:
        array = np.asarray(value, dtype=np.float32)
    except (ValueError, TypeError) as error:
        raise ValueError("Invalid " + label) from error
    require(array.shape == shape and np.isfinite(array).all(), "Invalid " + label)
    return array


def evidence_file(run, name, *, trial=False, suffix=".json"):
    require(isinstance(name, str), "Invalid evidence filename")
    relative = Path(name)
    require(not relative.is_absolute() and relative.suffix == suffix
            and relative.parts == (("trials", relative.name) if trial else (relative.name,)),
            "Unsafe evidence filename")
    path = Path(run) / relative
    require(path.is_file() and not path.is_symlink()
            and not any(parent.is_symlink() for parent in path.parents if parent != Path(run).parent),
            "Missing or linked evidence file")
    return path


def validate_study(study, spec):
    from continuous_feasibility import LOW, HIGH, ACTION_NAMES, split_for_index
    from continuous_policy import ARCHITECTURE, POLICY_VERSION
    from expanded_farm import make_spec

    study_hash = signed(study, "Continuous study")
    geometry_hash = signed(spec, "Original geometry")
    domain_hash = signed(study["domain"], "Continuous domain")
    require(spec == make_spec(spec["scene"],10), "Historical source geometry changed")
    if "coverage_batches" in study:
        require(study["coverage_batches"] == 4 and isinstance(study.get("feasibility_sha256"),str),
                "Geometry coverage protocol changed")
    cells = study["cells"]
    require(len(cells) == len(spec["cells"]) == 35 and
            cells == [dict(c, split=split_for_index(i)) for i,c in enumerate(spec["cells"])],
            "Train/validation/test cell geometry or split changed")
    counts = {label: sum(c["split"] == label for c in cells) for label in ("train", "validation", "test")}
    require(counts == dict(train=25, validation=5, test=5), "Expected disjoint 25/5/5 layouts")
    require(len({tuple(c["bucket_xy_m"]) for c in cells}) == 35, "Spatial layout overlap")
    require(study.get("schema") == "sircus-native-continuous-pilot-v1"
            and study.get("geometry_fingerprint") == geometry_hash
            and study.get("training_batches") in (4,64)
            and study.get("batches_per_update") == 4
            and study.get("assessment_stages") == [0,study["training_batches"]]
            and study.get("assessment_batches") == (2 if study["training_batches"] == 4 else 8)
            and study.get("seed") == 20261112
            and study.get("architecture") == ARCHITECTURE
            and study.get("policy_version") == POLICY_VERSION,
            "Continuous study plan/policy changed")
    require(study["domain"].get("names") == list(ACTION_NAMES)
            and study["domain"].get("low") == LOW.tolist()
            and study["domain"].get("high") == HIGH.tolist()
            and study["domain"].get("reference_xy_m") == [.67,-.10],
            "Physical continuous action domain changed")
    require(study.get("ppo") == dict(lr=.0003, epochs=4, clip=.2, value_coef=.5,
                                     entropy_coef=.005, grad_norm=1.),
            "Continuous PPO hyperparameters changed")
    return study_hash, geometry_hash, domain_hash


def verify_checkpoint_chain(run, executed_source=None):
    """Check every continuous trial against its frozen CPU-loaded checkpoint."""
    import torch
    from continuous_feasibility import LOW, HIGH, decode_action, geometry_coverage
    from continuous_policy import (build_observations, physical_actions, load_checkpoint,
                                   weight_hash, actor_weight_hash)
    from farm_contract import reward, statistics

    run = Path(run)
    spec, study, report = (read(run/name) for name in ("contract.json", "study.json", "result.json"))
    study_hash, geometry_hash, domain_hash = validate_study(study,spec)
    require(report.get("schema") == study["schema"] and report.get("status") == "completed"
            and report.get("study_fingerprint") == study_hash
            and report.get("geometry_fingerprint") == geometry_hash
            and report.get("workcells") == 35 and report.get("articulations") == 70,
            "Continuous result identity changed")
    batches = study["training_batches"]
    repeats = study["assessment_batches"]
    require(len(report["batches"]) == batches and len(report["updates"]) == batches//4
            and len(report["assessments"]) == 2*repeats, "Incomplete continuous pilot")
    if executed_source is not None:
        executed_source = Path(executed_source)
        require(isinstance(report.get("source_sha256"),dict) and report["source_sha256"],
                "Missing native source hashes")
        for name, expected in report["source_sha256"].items():
            require(Path(name).name == name and name.endswith(".py"), "Unsafe native source filename")
            require(digest(executed_source/name) == expected, "Executed native source changed: " + name)
    torch.set_num_threads(2)
    expected_obs = build_observations(spec["cells"])
    observations = torch.from_numpy(expected_obs)
    train_indices = [i for i,c in enumerate(study["cells"]) if c["split"] == "train"]
    expected_train_ids = [study["cells"][i]["id"] for i in train_indices]
    checkpoint_hashes, trial_hashes, policies = {}, {}, {}
    max_logp_error = 0.0
    training_rows, evaluation_rows = [], {"initial":[],"final":[]}
    membership, coverage_rows = [], []
    coverage_batches = study.get("coverage_batches",0)
    require(len(report.get("coverage",[])) == coverage_batches, "Incomplete geometry coverage")
    coverage_plans = []
    if coverage_batches:
        path = evidence_file(run,"feasibility.json")
        require(digest(path) == study["feasibility_sha256"], "Geometry feasibility map changed")
        feasibility = read(path)
        signed(feasibility,"Geometry feasibility map")
        require(feasibility.get("source_fingerprint") == geometry_hash,
                "Geometry coverage uses a different source")
        coverage_plans = geometry_coverage(feasibility,spec["cells"],batches=coverage_batches)

    def checkpoint(reference, updates, attempts):
        path = evidence_file(run, reference["file"], suffix=".pt")
        require(digest(path) == reference["sha256"], "Checkpoint checksum changed")
        checkpoint_hashes[path.name] = digest(path)
        if path.name not in policies:
            policies[path.name] = load_checkpoint(path, study_fingerprint=study_hash,
                geometry_fingerprint=geometry_hash, domain_fingerprint=domain_hash,
                device="cpu", expected_updates=updates, expected_attempted_shots=attempts)
        policy, saved = policies[path.name]
        require(saved["seed"] == study["seed"] and saved["update_count"] == updates
                and saved["attempted_shots"] == attempts,
                "Sampling checkpoint counters changed")
        require(weight_hash(policy) == reference["policy_hash"]
                and actor_weight_hash(policy) == reference["actor_hash"],
                "Sampling checkpoint weights changed")
        return policy

    initial = checkpoint(report["initial"],0,0)
    final = checkpoint(report["final"],batches//4,batches*25)
    require(report["initial_hash"] == weight_hash(initial)
            and report["final_hash"] == weight_hash(final), "Initial/final policy identity changed")
    expected_hash, expected_actor = weight_hash(initial), actor_weight_hash(initial)

    def trial(summary, mode, batch, expected_reference, updates, attempts, coverage_plan=None):
        nonlocal max_logp_error
        path = evidence_file(run,summary["file"],trial=True)
        require(digest(path) == summary["sha256"], "Trial checksum changed")
        trial_hashes[summary["file"]] = digest(path)
        data = read(path)
        policy = checkpoint(data["checkpoint"],updates,attempts)
        require(data["checkpoint"] == expected_reference
                and data.get("schema") == study["schema"]
                and data.get("study_fingerprint") == study_hash
                and data.get("mode") == summary.get("mode") == mode
                and data.get("batch") == summary.get("batch") == batch
                and data.get("update_count") == summary.get("update_count") == updates
                and data.get("policy_hash") == weight_hash(policy),
                "Trial identity or policy binding changed")
        enabled = ([c["split"] == "train" for c in study["cells"]] if mode == "train"
                   else coverage_plan["enabled"] if mode == "coverage" else [True]*35)
        active = [i for i,value in enumerate(enabled) if value]
        active_ids = [study["cells"][i]["id"] for i in active]
        require(data.get("enabled") == enabled and data.get("active_cell_ids") == active_ids
                and data.get("optimizer_samples") == (25 if mode == "train" else 0)
                and data.get("deterministic") is (mode == "eval")
                and data.get("policy_action_used") is (mode != "coverage"),
                "Training membership or frozen-evaluation protocol changed")
        actual_obs = finite(data["observations"],(35,5),"policy observations")
        latent = finite(data["latent"],(35,5),"Gaussian action latent")
        normalized = finite(data["actions_normalized"],(35,5),"bounded action")
        logp = finite(data["logp"],(35,),"saved log probability")
        require(np.array_equal(actual_obs,expected_obs), "Policy observations differ from bucket geometry")
        with torch.no_grad():
            latent_t = torch.from_numpy(latent)
            actual_normalized = latent_t.tanh().numpy()
            actual_logp = policy.log_prob(observations,latent_t).numpy()
            if mode == "eval":
                require(np.allclose(latent,policy.distribution(observations).loc.numpy(),atol=2e-6,rtol=2e-6),
                        "Frozen deterministic assessment action changed")
        require(np.allclose(normalized,actual_normalized,atol=2e-6,rtol=2e-6),
                "Saved bounded action does not match sampled latent")
        require(np.allclose(logp,actual_logp,atol=LOGP_TOLERANCE,rtol=LOGP_TOLERANCE),
                "Saved log probability differs from frozen sampling checkpoint")
        max_logp_error = max(max_logp_error,float(np.abs(logp-actual_logp).max()))
        physical = physical_actions(normalized,LOW,HIGH)
        expected_actions = (coverage_plan["actions"] if mode == "coverage" else
                            [decode_action(c,a,f"{mode}-{batch}-{i}")
                             for i,(c,a) in enumerate(zip(spec["cells"],physical))])
        require(data.get("actions") == expected_actions and len(data["rows"]) == 35,
                "Executed continuous action differs from direct policy decoding")
        for i,(row,action) in enumerate(zip(data["rows"],expected_actions)):
            require(row["cell_id"] == study["cells"][i]["id"]
                    and row["split"] == study["cells"][i]["split"]
                    and all(row.get(k) == v for k,v in action.items()),
                    "Native outcome is not bound to its requested action/layout")
        rewards = finite(data["rewards"],(35,),"terminal reward")
        expected_rewards = np.asarray([reward(r) for r in data["rows"]],dtype=np.float32)
        require(np.array_equal(rewards,expected_rewards), "Saved rewards differ from native scalar reward rule")
        stats = statistics([data["rows"][i] for i in active])
        for key,value in stats.items():
            require(key in summary and np.isclose(summary[key],value,rtol=0,atol=1e-7),
                    "Trial summary counts changed: " + key)
        if mode == "train":
            membership.append(dict(batch=batch, cell_ids=active_ids,
                includes_invalid_attempts=True, samples=len(active),
                input_sha256=canonical_hash(dict(observations=actual_obs[active].tolist(),
                    latent=latent[active].tolist(), logp=logp[active].tolist(), rewards=rewards[active].tolist()))))
        return data,policy

    for batch,(summary,plan) in enumerate(zip(report.get("coverage",[]),coverage_plans)):
        data,_ = trial(summary,"coverage",batch,report["initial"],0,0,plan)
        coverage_rows.extend(row for row,use in zip(data["rows"],plan["enabled"]) if use)
    for batch,summary in enumerate(report["batches"]):
        reference = read(evidence_file(run,summary["file"],trial=True))["checkpoint"]
        require(reference["policy_hash"] == expected_hash and reference["actor_hash"] == expected_actor,
                "Training policy update chain changed")
        data,policy = trial(summary,"train",batch,reference,batch//4,batch*25)
        training_rows.extend(data["rows"][i] for i in train_indices)
        if batch%4 == 3:
            change = report["updates"][batch//4]
            require(change.get("update_count") == batch//4+1
                    and change.get("training_batches") == list(range(batch-3,batch+1))
                    and change.get("before") == expected_hash and change.get("actor_before") == expected_actor
                    and change.get("sample_count") == change.get("samples") == 100
                    and len(change.get("epochs",[])) == study["ppo"]["epochs"],
                    "PPO update membership or hash chain changed")
            for epoch in change["epochs"]:
                require(epoch.get("samples") == 100
                        and all(type(v) in (int,float) and np.isfinite(v) for v in epoch.values()),
                        "Nonfinite or incomplete PPO diagnostics")
            expected_hash,expected_actor = change["after"],change["actor_after"]
    require(expected_hash == weight_hash(final) and expected_actor == actor_weight_hash(final),
            "Final checkpoint does not follow logged PPO updates")
    for stage,label in enumerate(("initial","final")):
        reference = report[label]
        after_batches = stage*batches
        for repeat in range(repeats):
            summary = report["assessments"][stage*repeats+repeat]
            data,_ = trial(summary,"eval",repeat,reference,after_batches//4,after_batches*25)
            evaluation_rows[label].extend(data["rows"])
    require(len(training_rows) == 25*batches and len(membership) == batches
            and all(m["cell_ids"] == expected_train_ids and m["samples"] == 25 for m in membership),
            "Held-out layouts entered optimization or rejected attempts were filtered")
    return dict(schema=SCHEMA,status="passed",study_fingerprint=study_hash,
        geometry_fingerprint=geometry_hash,domain_fingerprint=domain_hash,
        source_result_sha256=digest(run/"result.json"), study_sha256=digest(run/"study.json"),
        checkpoints=checkpoint_hashes,trials=trial_hashes,updates=len(report["updates"]),
        optimizer_samples=len(training_rows), training_batches=batches,
        train_cell_ids=expected_train_ids,
        held_out_cell_ids={label:[c["id"] for c in study["cells"] if c["split"] == label]
                           for label in ("validation","test")},
        optimizer_membership_sha256=canonical_hash(membership),
        max_logp_error=max_logp_error,logp_tolerance=LOGP_TOLERANCE,
        actor_changed=actor_weight_hash(initial)!=actor_weight_hash(final),
        train_statistics=statistics(training_rows),
        geometry_coverage_attempts=len(coverage_rows),
        geometry_coverage_excluded_from_optimizer=True,
        assessment_statistics={label:{split:statistics([r for r in rows if r["split"] == split])
                                      for split in ("train","validation","test")}
                               for label,rows in evaluation_rows.items()},
        scope="CPU checkpoint/input/likelihood/reward/update-membership audit only; native telemetry rescoring and optimizer arithmetic are separate")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run",type=Path,required=True)
    parser.add_argument("--executed-source",type=Path)
    parser.add_argument("--output",type=Path,required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Audit output already exists")
    value = verify_checkpoint_chain(args.run,args.executed_source)
    with args.output.open("x") as output:
        json.dump(value,output,allow_nan=False)
    print(json.dumps(value,allow_nan=False))


if __name__ == "__main__":
    main()
