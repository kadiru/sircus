"""Independent CPU audit for the warm-started sweep versus residual-PPO study.

The checker reads saved requests, scores, likelihoods and native arrays. It never
starts a native rollout, acquires a GPU, or edits the original study evidence.
"""

from __future__ import annotations

from collections import Counter
import argparse
import copy
import hashlib
import json
from pathlib import Path
import time

import numpy as np


VERSION = "sircus-sweep-residual-ppo-independent-audit-v1"
PRIOR_PATH = "baselines/mjlab-grid35-expanded-20261103"
METHODS = ("sweep", "ppo")
WIDE_LOW = np.asarray([0.30, -0.50, 0.12, 8.0, 0.05])
WIDE_HIGH = np.asarray([1.04, 0.35, 0.60, 60.0, 0.65])
RESIDUAL_SCALE = np.asarray([0.04, 0.04, 0.05, 8.0, 0.12])
SPLIT_SEED = 20261303
PLACEMENT_NOISE = np.asarray([0.001, 0.001, 0.002, 0.35, 0.002])


class EvidenceError(ValueError):
    pass


class Checks:
    def __init__(self):
        self.count = 0
        self.numeric_error = {}

    def require(self, passed, label):
        self.count += 1
        if not passed:
            raise EvidenceError(label)

    def equal(self, actual, expected, label):
        self.require(actual == expected, label)

    def numeric(self, actual, expected, label, tolerance=1e-9):
        if actual is None or expected is None:
            self.equal(actual, expected, label)
            return
        a, b = np.asarray(actual, dtype=float), np.asarray(expected, dtype=float)
        self.require(a.shape == b.shape, label + "/shape")
        error = float(np.abs(a - b).max()) if a.size else 0.0
        self.require(
            np.isfinite(a).all() and np.isfinite(b).all() and error <= tolerance, label
        )
        self.numeric_error[label.rsplit("/", 1)[-1]] = max(
            self.numeric_error.get(label.rsplit("/", 1)[-1], 0.0), error
        )


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, allow_nan=False).encode()
    ).hexdigest()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def contained(directory, name):
    root = Path(directory).resolve()
    relative = Path(name)
    candidate = (root / relative).resolve()
    if relative.is_absolute() or not candidate.is_relative_to(root):
        raise EvidenceError("Artifact escapes its declared directory: " + str(name))
    return candidate


def fingerprint(value, checks, label):
    payload = dict(value)
    expected = payload.pop("fingerprint", None)
    checks.require(isinstance(expected, str), label + "/fingerprint present")
    checks.equal(digest(payload), expected, label + "/fingerprint")


def verify_prior_bundle(repository, expected_prior=None, *, checks=None):
    """Independently reconstruct all17 exact shared teachers from their audits."""
    checks = checks or Checks()
    root = Path(repository)
    if not (root / "manifest.sha256").is_file():
        root = root / PRIOR_PATH
    file_hashes = {}
    for line in (root / "manifest.sha256").read_text().splitlines():
        expected, name = line.split("  ", 1)
        path = contained(root, name)
        checks.equal(sha(path), expected, "Prior published byte hash/" + name)
        checks.require(name not in file_hashes, "Prior unique manifest member/" + name)
        file_hashes[name] = expected
    checks.require(bool(file_hashes), "Prior complete published manifest")
    actual = {
        str(p.relative_to(root))
        for p in root.rglob("*")
        if p.is_file() and p.name != "manifest.sha256"
    }
    checks.equal(set(file_hashes), actual, "Prior manifest covers actual bundle")
    reports, settings, metadata = {}, {}, {}
    for kind in ("parent", "followup"):
        directory = root / kind
        manifest = read(directory / "manifest.json")
        for name, expected in manifest.items():
            checks.equal(
                sha(contained(directory, name)),
                expected,
                kind + "/compact byte hash/" + name,
            )
        bundle = read(directory / "BUNDLE.json")
        report, contract, audit = (
            read(directory / name)
            for name in ("report.json", "contract.json", "audit.json")
        )
        for name, value in (("report", report), ("contract", contract)):
            fingerprint(value, checks, kind + "/" + name)
        checks.equal(report["status"], "completed", kind + "/prior study completed")
        checks.equal(audit["status"], "passed", kind + "/prior audit passed")
        checks.equal(audit["failures"], [], kind + "/prior audit has no failures")
        checks.require(audit["checks"] > 0, kind + "/prior audit has checks")
        checks.equal(
            audit["report_sha256"],
            sha(directory / "report.json"),
            kind + "/audit report binding",
        )
        checks.equal(
            audit["study_manifest_sha256"],
            sha(directory / "original-manifest.json"),
            kind + "/audit original manifest binding",
        )
        checks.equal(
            audit["frozen_selection_sha256"],
            sha(directory / "frozen-selection.json"),
            kind + "/audit frozen-selection binding",
        )
        checks.equal(
            audit["contract_fingerprint"],
            contract["fingerprint"],
            kind + "/audit contract binding",
        )
        checks.equal(
            bundle["audit_sha256"],
            sha(directory / "audit.json"),
            kind + "/compact audit binding",
        )
        checks.equal(
            bundle["contract_fingerprint"],
            contract["fingerprint"],
            kind + "/compact contract binding",
        )
        checks.equal(
            report["identity"], contract["identity"], kind + "/report identity"
        )
        reports[kind] = {c["id"]: c for c in report["cells"]}
        rows = read(directory / "selected-settings.json")
        checks.equal(
            len(rows), len({r["id"] for r in rows}), kind + "/unique selected settings"
        )
        settings[kind] = {r["id"]: r for r in rows}
        metadata[kind] = dict(
            report_fingerprint=report["fingerprint"],
            contract_fingerprint=contract["fingerprint"],
            audit_sha256=sha(directory / "audit.json"),
        )
    bank = read(root / "recipe-bank.json")
    checks.equal(
        bank["parent_fingerprint"],
        metadata["parent"]["report_fingerprint"],
        "Prior bank parent report",
    )
    checks.equal(
        bank["study_fingerprint"],
        metadata["followup"]["report_fingerprint"],
        "Prior bank follow-up report",
    )
    checks.equal(len(bank["rows"]), 35, "Prior all35 targets remain present")
    teachers = []
    for bank_row in bank["rows"]:
        selected = bank_row.get("selected")
        if not selected:
            continue
        identifier = bank_row["id"]
        kind = "followup" if selected["source"] == "Fresh study" else "parent"
        checks.require(
            selected["source"] in ("Fresh study", "Saved reference"),
            identifier + "/known teacher source",
        )
        source = reports[kind][identifier]
        record = settings["followup"][identifier]
        configuration = record["configuration"]
        expected_sha = selected["configurationSha"]
        checks.equal(
            digest(configuration),
            expected_sha,
            identifier + "/exact configuration hash",
        )
        checks.equal(
            record["selected"]["configuration_sha256"],
            expected_sha,
            identifier + "/endpoint configuration binding",
        )
        checks.equal(
            source["selected"]["configuration_sha256"],
            expected_sha,
            identifier + "/source configuration binding",
        )
        checks.equal(
            record["bucket_xy_m"], bank_row["xy"], identifier + "/fixed source bucket"
        )
        checks.equal(
            source["bucket_xy_m"],
            bank_row["xy"],
            identifier + "/source bucket identity",
        )
        frozen = source["frozen_validation"]
        checks.equal(
            frozen["status"], "completed", identifier + "/frozen cohort completed"
        )
        checks.equal(
            frozen["configuration_sha256"],
            expected_sha,
            identifier + "/frozen exact configuration",
        )
        checks.equal(
            frozen["nominal"]["attempts"], 3, identifier + "/nominal cohort size"
        )
        checks.equal(
            frozen["fresh"]["attempts"], 30, identifier + "/perturbed cohort size"
        )
        shown = bank_row["frozen"] if kind == "followup" else bank_row["previousFrozen"]
        checks.equal(
            shown["fresh"]["captures"],
            frozen["fresh"]["captures"],
            identifier + "/source-matched stored rate",
        )
        values = np.asarray(
            [
                *configuration["board_center"],
                configuration["tilt_deg"],
                configuration["gap_m"],
            ],
            dtype=float,
        )
        checks.require(
            values.shape == (5,) and np.isfinite(values).all(),
            identifier + "/finite five-action teacher",
        )
        checks.require(
            ((values >= WIDE_LOW) & (values <= WIDE_HIGH)).all(),
            identifier + "/teacher within shared global bounds",
        )
        for role in ("q10", "q14"):
            q = np.asarray(configuration[role], dtype=float)
            checks.require(
                q.shape == (6,) and np.isfinite(q).all(),
                identifier + "/finite source branch/" + role,
            )
        teachers.append(
            dict(
                id=identifier,
                bucket_xy_m=bank_row["xy"],
                values=values.tolist(),
                q10=configuration["q10"],
                q14=configuration["q14"],
                configuration_sha256=expected_sha,
                source_study=kind,
            )
        )
    checks.equal(len(teachers), 17, "Prior exactly17 shared teachers")
    checks.equal(len({r["id"] for r in teachers}), 17, "Prior unique teachers")
    records = []
    for teacher in sorted(teachers, key=lambda r: r["id"]):
        record = dict(teacher)
        record["source_record_sha256"] = digest(record)
        records.append(record)
    result = dict(
        records=records,
        records_sha256=digest(records),
        bundle_manifest_sha256=sha(root / "manifest.sha256"),
        recipes=17,
        latest_study_native_releases=4234,
        latest_study_requested_trials=4243,
        full_historical_cost_known=False,
        files_sha256=file_hashes,
        source_reports=metadata,
    )
    if expected_prior is not None:
        for key in (
            "records",
            "records_sha256",
            "bundle_manifest_sha256",
            "recipes",
            "latest_study_native_releases",
            "latest_study_requested_trials",
            "full_historical_cost_known",
        ):
            checks.equal(
                expected_prior.get(key),
                result[key],
                "Shared prior exact semantic binding/" + key,
            )
        checks.require(
            isinstance(expected_prior.get("scope"), str)
            and bool(expected_prior["scope"]),
            "Prior historical acquisition cost scope",
        )
    return result


def row_reward(row):
    """Reconstruct the fixed terminal reward without learner implementation."""
    if row.get("invalid"):
        return -1.0
    shaped = 0.15 * float(row["board_bounce"]) + 0.5 * np.exp(
        -row["closest_opening_distance_m"] / 0.12
    )
    if row["external_contact_steps"]:
        return float(shaped - 0.5)
    return float(
        shaped
        + 0.5 * min(row["max_retention_s"], 1.0)
        + 3 * row["route_success"]
        + 0.3 * (row["route_quality"] == 2)
        + 0.1 * row["single_bounce_success"]
    )


def counts(rows):
    return dict(
        attempts=len(rows),
        captures=sum(
            bool(
                r.get("native_release")
                and r.get("route_success")
                and not r.get("invalid")
            )
            for r in rows
        ),
        native_releases=sum(bool(r.get("native_release")) for r in rows),
        invalid=sum(bool(r.get("invalid")) for r in rows),
        clean=sum(
            bool(
                r.get("native_release")
                and not r.get("invalid")
                and r.get("route_quality") == 2
            )
            for r in rows
        ),
        mean_reward=float(np.mean([row_reward(r) for r in rows])) if rows else 0.0,
    )


def check_row_invariants(rows, checks):
    checks.equal(len(rows), len({r["id"] for r in rows}), "Unique requested trial IDs")
    for row in rows:
        label = row["id"]
        values = np.asarray(row.get("values", []), dtype=float)
        checks.require(
            values.shape == (5,) and np.isfinite(values).all(),
            label + "/finite five-action request",
        )
        bucket = np.asarray(row.get("bucket_xy_m", []), dtype=float)
        checks.require(
            bucket.shape == (2,) and np.isfinite(bucket).all(),
            label + "/finite bucket XY",
        )
        if row.get("native_release"):
            checks.require(not row.get("invalid"), label + "/released request admitted")
            checks.require(
                ((values >= WIDE_LOW) & (values <= WIDE_HIGH)).all(),
                label + "/released action inside bounds",
            )
        else:
            checks.require(
                bool(row.get("invalid")),
                label + "/nonreleased request retains rejection",
            )
            checks.require(
                not row.get("route_success"), label + "/rejection cannot capture"
            )
        if row.get("route_success"):
            checks.require(
                row.get("native_release")
                and row.get("board_bounce")
                and row.get("whole_ball_entry")
                and row.get("max_retention_s", 0) >= 1 - 1e-9
                and not row.get("external_contact_steps"),
                label + "/board-to-bucket capture invariants",
            )
        if "learning_reward" in row:
            checks.numeric(
                row["learning_reward"], row_reward(row), label + "/terminal reward"
            )


def expected_splits():
    """Rebuild the prespecified layout pools without the runner's generator."""
    train = [
        dict(id=f"dev-grid-{row:02d}-{column:02d}", bucket_xy_m=[float(x), float(y)])
        for row, y in enumerate(np.linspace(-0.65, 0.25, 5))
        for column, x in enumerate(np.linspace(0.10, 1.24, 7))
    ]
    used = np.asarray([r["bucket_xy_m"] for r in train])
    result = {"train": train}
    for role, count, seed in (
        ("validation", 12, SPLIT_SEED),
        ("test", 24, SPLIT_SEED + 1),
    ):
        generator, group = np.random.default_rng(seed), []
        for _ in range(100000):
            if len(group) == count:
                break
            xy = generator.uniform([0.1, -0.65], [1.24, 0.25])
            if np.linalg.norm(used - xy, axis=1).min() < 0.005:
                continue
            group.append(
                dict(id=f"bench-{role}-{len(group):02d}", bucket_xy_m=xy.tolist())
            )
            used = np.vstack([used, xy])
        if len(group) != count:
            raise EvidenceError("Failed to independently rebuild fixed layout pools")
        result[role] = group
    return result


def expected_schedule(seed, budget):
    generator, result = np.random.default_rng(seed), []
    while len(result) < budget:
        result.extend(generator.permutation(35).tolist())
    return result[:budget]


def source_anchor(bucket, records):
    return min(
        records,
        key=lambda r: (
            float(np.linalg.norm(np.asarray(r["bucket_xy_m"]) - bucket)),
            r["id"],
        ),
    )


def expected_observation(bucket, anchor):
    return np.r_[
        (np.asarray(bucket) - [0.67, -0.20]) / [0.57, 0.45],
        (np.asarray(anchor["bucket_xy_m"]) - [0.67, -0.20]) / [0.57, 0.45],
        2 * (np.asarray(anchor["values"]) - WIDE_LOW) / (WIDE_HIGH - WIDE_LOW) - 1,
    ].astype(np.float32)


def expected_perturbation(context_id, repeat):
    if repeat == 0:
        return np.zeros(5)
    code = f"{SPLIT_SEED}:{context_id}:{repeat}"
    seed = int(hashlib.sha256(code.encode()).hexdigest()[:16], 16)
    return np.random.default_rng(seed).normal(0, PLACEMENT_NOISE)


def check_contract_protocol(contract, prior, checks):
    fingerprint(contract, checks, "Execution contract")
    checks.equal(
        contract.get("version"),
        "sircus-sweep-rl-benchmark-v1",
        "Execution study version",
    )
    checks.equal(
        contract["prior_sha256"],
        digest(prior),
        "Shared prior payload bound before outcomes",
    )
    p = contract["protocol"]
    checks.equal(
        [r["id"] for r in p["methods"]], list(METHODS), "Matched declared methods"
    )
    seeds, milestones, budget = p["seeds"], p["milestones"], p["sample_budget"]
    checks.require(
        bool(seeds)
        and all(isinstance(s, int) and not isinstance(s, bool) for s in seeds)
        and len(seeds) == len(set(seeds)),
        "Unique independent declared seeds",
    )
    checks.require(
        isinstance(budget, int)
        and not isinstance(budget, bool)
        and 32 <= budget <= 4096
        and budget % 32 == 0,
        "Finite equal requested-decision budget",
    )
    checks.require(
        milestones == sorted(set(milestones))
        and milestones[0] == 0
        and milestones[-1] == budget
        and all(
            isinstance(m, int) and not isinstance(m, bool) and m % 32 == 0
            for m in milestones
        ),
        "Ordered declared equal-budget checkpoints",
    )
    checks.equal(p["batch_size"], 32, "Shared batch size")
    checks.numeric(
        p["action_bounds"]["low"], WIDE_LOW, "Shared global action lower bounds", 0
    )
    checks.numeric(
        p["action_bounds"]["high"], WIDE_HIGH, "Shared global action upper bounds", 0
    )
    checks.numeric(
        p["residual_scale"], RESIDUAL_SCALE, "Shared restricted residual scales", 0
    )
    evaluation = p["evaluation"]
    for key, value in (
        ("validation_layouts", 12),
        ("test_layouts", 24),
        ("repeats", 3),
    ):
        checks.equal(evaluation[key], value, "Fixed evaluation cohort/" + key)
    for key, value in (
        ("lr", 0.0003),
        ("epochs", 6),
        ("clip", 0.2),
        ("value_weight", 0.5),
        ("entropy_weight", 0.001),
        ("gradient_norm", 1.0),
        ("initial_logstd", -1.5),
    ):
        checks.equal(p["ppo"][key], value, "Prespecified PPO parameter/" + key)
    checks.equal(
        p["ppo"]["network"], [9, 64, 64, 5], "Continuous residual actor architecture"
    )
    checks.equal(p["ppo"]["critic"], [9, 64, 64, 1], "Terminal critic architecture")
    checks.equal(p["ppo"]["optimizer"], "CPU Adam", "Auditable CPU policy optimizer")
    checks.equal(
        p["ppo"]["logstd_bounds"], [-3.0, -0.7], "Prespecified exploration bounds"
    )
    for key in (
        "recipes",
        "records_sha256",
        "bundle_manifest_sha256",
        "latest_study_native_releases",
        "latest_study_requested_trials",
        "full_historical_cost_known",
    ):
        checks.equal(
            p["prior"][key], prior[key], "Protocol common historical information/" + key
        )
    expected, found = expected_splits(), {}
    coordinates, identifiers = [], []
    for role, expected_rows in expected.items():
        actual = contract["splits"][role]
        checks.equal(
            len(actual), len(expected_rows), "All declared layouts retained/" + role
        )
        for original, row in zip(expected_rows, actual):
            checks.equal(row["id"], original["id"], "Fixed layout ID/" + role)
            checks.equal(
                row["bucket_xy_m"],
                original["bucket_xy_m"],
                "Prespecified layout coordinate/" + original["id"],
            )
            checks.equal(row["split"], role, "Declared layout split/" + row["id"])
            found[row["id"]] = (role, row)
            coordinates.append(tuple(row["bucket_xy_m"]))
            identifiers.append(row["id"])
    checks.equal(
        len(coordinates),
        len(set(coordinates)),
        "Train validation and test coordinates disjoint",
    )
    checks.equal(
        len(identifiers), len(set(identifiers)), "Train validation and test IDs unique"
    )
    train_xy = {tuple(r["bucket_xy_m"]) for r in contract["splits"]["train"]}
    checks.require(
        all(tuple(r["bucket_xy_m"]) in train_xy for r in prior["records"]),
        "Historical teacher labels restricted to TRAIN layouts",
    )
    return found


def numpy_state(state):
    return {
        key: np.asarray(
            value.detach().cpu().numpy() if hasattr(value, "detach") else value
        )
        for key, value in state.items()
    }


def weights_sha256(state):
    result = hashlib.sha256()
    for key, value in sorted(numpy_state(state).items()):
        result.update(key.encode())
        result.update(value.tobytes())
    return result.hexdigest()


def actor_sha256(state):
    return weights_sha256(
        {
            key: value
            for key, value in state.items()
            if key.startswith("actor.") or key == "logstd"
        }
    )


def network_forward(state, observation, prefix="actor"):
    """Independent NumPy forward pass through the fixed two-layer tanh net."""
    state = numpy_state(state)
    value = np.asarray(observation, dtype=np.float32)
    for layer in (0, 2, 4):
        value = (
            value @ state[f"{prefix}.{layer}.weight"].T
            + state[f"{prefix}.{layer}.bias"]
        )
        if layer != 4:
            value = np.tanh(value)
    return value


def behavior_log_prob(state, observation, latent):
    mean = network_forward(state, observation)
    logstd = numpy_state(state)["logstd"]
    standard = np.exp(logstd)
    latent = np.asarray(latent, dtype=np.float32)
    return (
        -0.5 * ((latent - mean) / standard) ** 2 - logstd - 0.5 * np.log(2 * np.pi)
    ).sum(-1)


def check_behavior_likelihood(state, observation, latent, recorded, checks, label):
    observed = np.asarray(observation)
    samples = np.asarray(latent)
    checks.require(
        observed.ndim == 2 and observed.shape[1] == 9,
        label + "/nine observable features",
    )
    checks.equal(
        samples.shape, (len(observed), 5), label + "/five continuous behavior latents"
    )
    checks.numeric(
        recorded,
        behavior_log_prob(state, observed, samples),
        label + "/behavior log probabilities",
        tolerance=5e-5,
    )


def replay_ppo_update(
    before_state, optimizer_state, after_state, buffer, checks, label
):
    """Recompute the actual CPU Adam PPO update without the learner helper."""
    import torch

    class AuditedPolicy(torch.nn.Module):
        def __init__(self):
            super().__init__()

            def network(output):
                return torch.nn.Sequential(
                    torch.nn.Linear(9, 64),
                    torch.nn.Tanh(),
                    torch.nn.Linear(64, 64),
                    torch.nn.Tanh(),
                    torch.nn.Linear(64, output),
                )

            self.actor, self.critic = network(5), network(1)
            self.logstd = torch.nn.Parameter(torch.zeros(5))

    for group in optimizer_state["param_groups"]:
        for key, value in (
            ("lr", 0.0003),
            ("eps", 1e-8),
            ("weight_decay", 0),
            ("amsgrad", False),
            ("maximize", False),
        ):
            checks.equal(group.get(key), value, label + "/Adam parameter/" + key)
        checks.equal(list(group["betas"]), [0.9, 0.999], label + "/Adam betas")
    policy = AuditedPolicy().cpu()
    policy.load_state_dict(before_state)
    optimizer = torch.optim.Adam(policy.parameters(), lr=0.0003)
    optimizer.load_state_dict(optimizer_state)
    obs = torch.as_tensor(np.asarray(buffer["obs"]), dtype=torch.float32)
    latent = torch.as_tensor(np.asarray(buffer["latent"]), dtype=torch.float32)
    old = torch.as_tensor(np.asarray(buffer["old_logp"]), dtype=torch.float32).detach()
    rewards = torch.as_tensor(np.asarray(buffer["reward"]), dtype=torch.float32)
    with torch.no_grad():
        advantage = rewards - policy.critic(obs).squeeze(-1)
        advantage = (advantage - advantage.mean()) / (
            advantage.std(unbiased=False) + 1e-6
        )
    losses = []
    for _ in range(6):
        distribution = torch.distributions.Normal(
            policy.actor(obs), policy.logstd.exp()
        )
        ratio = (distribution.log_prob(latent).sum(-1) - old).exp()
        actor = -torch.minimum(
            ratio * advantage, ratio.clamp(0.8, 1.2) * advantage
        ).mean()
        critic = ((policy.critic(obs).squeeze(-1) - rewards) ** 2).mean()
        loss = actor + 0.5 * critic - 0.001 * distribution.entropy().sum(-1).mean()
        checks.require(
            bool(torch.isfinite(loss)), label + "/finite independent PPO loss"
        )
        optimizer.zero_grad()
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        checks.require(
            bool(torch.isfinite(norm)), label + "/finite independent PPO gradient"
        )
        optimizer.step()
        with torch.no_grad():
            policy.logstd.clamp_(-3.0, -0.7)
        losses.append(float(loss.detach()))
    reproduced = policy.state_dict()
    checks.equal(set(reproduced), set(after_state), label + "/updated parameter keys")
    for key, value in reproduced.items():
        checks.numeric(
            numpy_state({key: value})[key],
            numpy_state({key: after_state[key]})[key],
            label + "/replayed PPO parameter/" + key,
            tolerance=2e-6,
        )
    return dict(losses=losses, reproduced_weights_sha256=weights_sha256(reproduced))


def lines(path):
    return [
        json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()
    ]


def check_manifest(directory, checks, *, ignored=()):
    root = Path(directory)
    manifest = read(root / "manifest.json")
    checks.require(
        isinstance(manifest, dict) and bool(manifest),
        "Nonempty complete manifest/" + root.name,
    )
    for name, expected in manifest.items():
        checks.equal(
            sha(contained(root, name)),
            expected,
            "Artifact byte identity/" + root.name + "/" + name,
        )
    actual = {str(p.relative_to(root)) for p in root.rglob("*") if p.is_file()}
    actual -= {"manifest.json", *ignored}
    checks.equal(set(manifest), actual, "Manifest covers all evidence/" + root.name)
    return manifest


def stats_equal(actual, rows, checks, label):
    expected = counts(rows)
    for key, value in expected.items():
        if key == "mean_reward":
            checks.numeric(actual.get(key), value, label + "/" + key)
        else:
            checks.equal(actual.get(key), value, label + "/" + key)


def actor_module_sha256(state):
    return weights_sha256(
        {
            key.removeprefix("actor."): value
            for key, value in state.items()
            if key.startswith("actor.")
        }
    )


def tensor_object_digest(value):
    """Stable optimizer-state identity without pickle serialization details."""

    def converted(item):
        if hasattr(item, "detach") or isinstance(item, np.ndarray):
            array = numpy_state({"array": item})["array"]
            return dict(
                dtype=str(array.dtype),
                shape=list(array.shape),
                bytes_sha256=hashlib.sha256(array.tobytes()).hexdigest(),
            )
        if isinstance(item, dict):
            return {str(k): converted(v) for k, v in item.items()}
        if isinstance(item, (list, tuple)):
            return [converted(v) for v in item]
        return item

    return digest(converted(value))


def load_checkpoint(path, checks, label):
    import torch

    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    checks.equal(
        checkpoint["weight_sha256"],
        weights_sha256(checkpoint["policy"]),
        label + "/model weight identity",
    )
    checks.equal(
        checkpoint["actor_sha256"],
        actor_module_sha256(checkpoint["policy"]),
        label + "/actor weight identity",
    )
    return checkpoint


def independent_sweep_prediction(state, context, anchor):
    nearest = min(
        state["train"],
        key=lambda r: (
            float(
                np.linalg.norm(np.asarray(r["bucket_xy_m"]) - context["bucket_xy_m"])
            ),
            r["id"],
        ),
    )
    source = state["best"].get(nearest["id"])
    residual = np.zeros(5) if source is None else np.asarray(source["residual"])
    return np.asarray(anchor["values"]) + residual * RESIDUAL_SCALE, dict(
        nearest_training_context_id=nearest["id"],
        source_training_ids=[] if source is None else [source["id"]],
        normalized_residual=residual.tolist(),
        search_at_inference=False,
    )


def reconstruct_sweep(seed, train, rows, checks, *, snapshots=None):
    state = dict(
        version="sircus-guided-residual-sweep-v1",
        seed=seed,
        train=copy.deepcopy(train),
        counts={},
        best={},
    )
    snapshots = {} if snapshots is None else snapshots
    if 0 in snapshots:
        checks.equal(snapshots[0], state, "Sweep zero-budget archive")
    for offset in range(0, len(rows), 32):
        batch = rows[offset : offset + 32]
        for row in batch:
            key = row["context_id"]
            index = state["counts"].get(key, 0)
            state["counts"][key] = index + 1
            code = f"{seed}:{key}:{index}"
            generator = np.random.default_rng(
                int(hashlib.sha256(code.encode()).hexdigest()[:16], 16)
            )
            dependencies = []
            if index == 0:
                residual, strategy = np.zeros(5), "zero_shared_anchor"
            elif index <= 10:
                residual = np.zeros(5)
                residual[(index - 1) // 2] = -0.5 if index % 2 else 0.5
                strategy = "axis_sweep"
            elif index <= 20:
                residual, strategy = (
                    generator.uniform(-0.8, 0.8, 5),
                    "seeded_joint_space_fill",
                )
            else:
                source = state["best"].get(key)
                center = (
                    np.zeros(5) if source is None else np.asarray(source["residual"])
                )
                dependencies = [] if source is None else [source["id"]]
                residual = np.tanh(
                    np.arctanh(np.clip(center, -0.999999, 0.999999))
                    + generator.normal(0, 0.35, 5)
                )
                strategy = "train_best_refinement"
            label = row["id"]
            checks.numeric(
                row["normalized_residual"],
                residual,
                label + "/independent TRAIN sweep action",
            )
            checks.equal(row["strategy"], strategy, label + "/sweep proposal family")
            checks.equal(
                row["proposal_index"],
                index,
                label + "/sweep per-context proposal count",
            )
            checks.equal(
                row["source_training_ids"],
                dependencies,
                label + "/sweep TRAIN-only dependencies",
            )
        # No outcome in this batch may affect a proposal from the same batch.
        for row in batch:
            key = row["context_id"]
            rank = [
                int(
                    bool(
                        row.get("native_release")
                        and row.get("route_success")
                        and not row.get("invalid")
                    )
                ),
                row_reward(row),
            ]
            old = state["best"].get(key)
            if old is None or tuple(rank) > tuple(old["rank"]):
                state["best"][key] = dict(
                    id=row["id"],
                    context_id=key,
                    residual=row["normalized_residual"],
                    rank=rank,
                    values=row["values"],
                )
        milestone = offset + len(batch)
        if milestone in snapshots:
            checks.equal(
                snapshots[milestone], state, f"Sweep frozen TRAIN archive/{milestone}"
            )
    return state


def check_native_provenance(directory, rows, prior, home_sha, checks):
    from sircus.mjlab.audit_grid_study import (
        check_motion,
        check_local_path,
    )

    native = Path(directory) / "native"
    raw = lines(native / "shots.jsonl")
    checks.equal(
        [r["id"] for r in raw],
        [r["id"] for r in rows],
        "Every raw native/admission request is retained once",
    )
    cache, anchors_checked, local_checked = {}, set(), set()

    def artifact(name, expected):
        if name not in cache:
            path = contained(native, name)
            checks.equal(sha(path), expected, "Native proof file hash/" + name)
            cache[name] = (expected, read(path))
        checks.equal(cache[name][0], expected, "Stable repeated proof binding/" + name)
        return cache[name][1]

    for source, row in zip(raw, rows):
        label = row["id"]
        for key, value in source.items():
            if key == "invalid":
                checks.equal(
                    bool(row.get(key)), bool(value), label + "/raw rejection preserved"
                )
            else:
                checks.equal(
                    row.get(key), value, label + "/immutable native field/" + key
                )
        checks.equal(
            row["adapter_directory"], "native", label + "/native evidence directory"
        )
        hint = source_anchor(row["bucket_xy_m"], prior["records"])
        checks.equal(
            row["source_hint"], hint, label + "/same geometry-only historical source"
        )
        proposal = row["proposal"]
        checks.equal(
            proposal["source_hint"],
            hint,
            label + "/native geometry source has no hidden labels",
        )
        preparation = artifact(row["preparation_file"], row["preparation_sha256"])
        checks.equal(
            row["preparation_file"],
            proposal["preparation_file"],
            label + "/preparation path binding",
        )
        checks.equal(
            row["preparation_sha256"],
            proposal["preparation_sha256"],
            label + "/preparation byte binding",
        )
        index = proposal["preparation_row"]
        request, candidate = (
            preparation["requests"][index],
            preparation["candidates"][index],
        )
        checks.equal(request["id"], label, label + "/prepared request ID")
        checks.equal(
            preparation["phase"],
            row["role"],
            label + "/native phase matches split role",
        )
        checks.equal(
            preparation["common_home_sha256"],
            home_sha,
            label + "/common staging identity",
        )
        for key in (
            "bucket_xy_m",
            "values",
            "source_hint",
            "command_values",
            "perturbation_mixed_units",
            "normalized_residual",
        ):
            if key in request:
                checks.equal(request[key], row[key], label + "/prepared command/" + key)
        for key in (
            "id",
            "bucket_xy_m",
            "values",
            "q10",
            "q14",
            "geometry_eligible",
            "endpoint_eligible",
            "geometry_row",
            "anchor_admitted",
        ):
            checks.equal(candidate[key], row[key], label + "/prepared candidate/" + key)
        checks.equal(
            row["anchor_file"], proposal["anchor_file"], label + "/anchor path"
        )
        checks.equal(
            row["anchor_sha256"], proposal["anchor_sha256"], label + "/anchor bytes"
        )
        anchor = artifact(row["anchor_file"], row["anchor_sha256"])
        expected_anchor = dict(bucket_xy_m=row["bucket_xy_m"], source_hint=hint)
        checks.equal(
            anchor["request"],
            expected_anchor,
            label + "/query-specific historical anchor",
        )
        checks.equal(
            anchor["key"],
            digest(expected_anchor),
            label + "/method-independent anchor key",
        )
        checks.equal(
            anchor["passed"],
            row["anchor_admitted"],
            label + "/anchor admission retained",
        )
        if anchor["passed"]:
            checks.require(
                anchor["home_geometry"]["passed"], label + "/anchor world clearance"
            )
            if row["anchor_file"] not in anchors_checked:
                checks.require(
                    check_motion(
                        anchor["path"],
                        anchor["resolved"],
                        row["bucket_xy_m"],
                        home_sha,
                        checks,
                        label + "/anchor route",
                    ),
                    label + "/anchor route admitted",
                )
                anchors_checked.add(row["anchor_file"])
            geometry_index = row["geometry_row"]
            checks.equal(
                preparation["geometry_request_indices"][geometry_index],
                index,
                label + "/geometry subset maps to requested row",
            )
            detail = preparation["geometry"]
            checks.equal(detail["summary"]["starts"], 16, label + "/strict IK starts")
            checks.equal(
                detail["summary"]["iterations"], 180, label + "/strict IK iterations"
            )
            checks.equal(
                detail["reasons"][geometry_index][0] == 0,
                row["endpoint_eligible"],
                label + "/strict endpoint result",
            )
            if row["endpoint_eligible"]:
                checks.require(
                    detail["minimum_clearance_lower_bound_m"][geometry_index][0]
                    >= 0.005 - 1e-10,
                    label + "/strict endpoint clearance",
                )
                checks.require(
                    row["motion_evaluated"],
                    label + "/admitted endpoint still needs local route",
                )
        else:
            checks.require(
                row["endpoint_eligible"] is None
                and row["geometry_row"] is None
                and not row["motion_evaluated"],
                label + "/failed anchor has no hidden endpoint retry",
            )
        if row["motion_evaluated"]:
            local = artifact(row["local_path_file"], row["local_path_sha256"])
            checks.equal(
                local["request"],
                dict(
                    anchor_sha256=row["anchor_sha256"],
                    bucket_xy_m=row["bucket_xy_m"],
                    **{k: row[k] for k in ("values", "q10", "q14")},
                ),
                label + "/exact local-path source",
            )
            checks.equal(
                bool(local["proof"]["passed"]),
                row["motion_admitted"],
                label + "/local proof result retained",
            )
            if local["proof"]["passed"] and row["local_path_file"] not in local_checked:
                checks.require(
                    check_local_path(
                        local["proof"],
                        row,
                        row["bucket_xy_m"],
                        anchor["path"],
                        checks,
                        label + "/local route",
                    ),
                    label + "/local route admitted",
                )
                local_checked.add(row["local_path_file"])
        else:
            checks.require(
                not row["motion_admitted"],
                label + "/unmeasured motion cannot be admitted",
            )
        checks.equal(
            bool(row["native_release"]),
            bool(row["geometry_eligible"] and row["motion_admitted"]),
            label + "/no native execution outside shared gates",
        )
    return dict(
        raw_rows=len(raw),
        anchors=len(anchors_checked),
        local_paths=len(local_checked),
    )


def check_resources(directory, run_report, checks):
    events = lines(Path(directory) / "resource-samples.jsonl")
    snapshots = [e["values"] for e in events if e["event"] == "snapshot"]
    observed = [
        e["values"]["used_vram_mib"]
        for e in events
        if e["event"] == "nvidia_process_sample"
        and e["values"]["used_vram_mib"] is not None
    ]
    for resource in [r["resources"] for r in run_report["curves"]] + [
        run_report["resources"]
    ]:
        checks.require(
            resource in snapshots,
            "Resource summary is an actual logged process observation",
        )
        for key, value in resource.items():
            if value is not None:
                checks.require(
                    isinstance(value, (int, float))
                    and not isinstance(value, bool)
                    and np.isfinite(value)
                    and value >= 0,
                    "Finite nonnegative observed resource/" + key,
                )
        n = resource["vram_samples"]
        checks.require(
            isinstance(n, int) and n <= len(observed),
            "NVIDIA sample count comes from recorded process samples",
        )
        checks.equal(
            resource["peak_vram_mb"],
            max(observed[:n]) if n else None,
            "NVIDIA memory is the sampled process peak, not zero for missing",
        )
        checks.require(
            resource["gpu_reservation_s"] + 1e-4
            >= resource["wall_s"] + resource["evaluation_wall_s"],
            "Reservation time retains evaluation cost separately",
        )
        checks.require(
            resource["wall_s"] + 1e-4 >= resource["gpu_s"],
            "Native elapsed time is inside measured training phase",
        )
        if (
            resource["torch_allocated_mb"] is not None
            and resource["torch_reserved_mb"] is not None
        ):
            checks.require(
                resource["torch_reserved_mb"] >= resource["torch_allocated_mb"],
                "PyTorch reserved and allocated peaks distinguished",
            )
    batches = lines(Path(directory) / "batches.jsonl")
    for point in run_report["curves"]:
        prefix = [b for b in batches if b["samples"] <= point["samples"]]
        for key in ("geometry_s", "gpu_s"):
            checks.numeric(
                point["resources"][key],
                sum(b["adapter_timing"][key] for b in prefix),
                f"Training-only cumulative adapter timing/{point['samples']}/{key}",
            )
    return dict(nvidia_samples=len(observed), snapshots=len(snapshots))


def audit_run(directory, contract, prior, scene, home_sha, checks):
    from sircus.mjlab.audit_grid_study import rescore_rows

    directory = Path(directory)
    check_manifest(directory, checks)
    report = read(directory / "report.json")
    fingerprint(report, checks, directory.name + "/run report")
    checks.equal(report["status"], "completed", directory.name + "/worker completed")
    checks.equal(
        report["contract_fingerprint"],
        contract["fingerprint"],
        directory.name + "/contract binding",
    )
    method, seed, protocol = report["method"], report["seed"], contract["protocol"]
    checks.require(
        method in METHODS and seed in protocol["seeds"],
        directory.name + "/declared method and seed",
    )
    checks.equal(
        directory.name,
        f"{method}-seed{seed}",
        "One independent directory per method/seed",
    )
    rows = lines(directory / "rows.jsonl")
    check_row_invariants(rows, checks)
    layout_map = {
        r["id"]: (role, r) for role, group in contract["splits"].items() for r in group
    }
    train = [r for r in rows if r["role"] == "train"]
    budget = protocol["sample_budget"]
    checks.equal(
        len(train),
        budget,
        directory.name + "/every requested training decision retained",
    )
    schedule = expected_schedule(seed, budget)
    checks.equal(
        read(directory / "schedule.json"),
        schedule,
        directory.name + "/common balanced layout schedule",
    )
    for index, row in enumerate(train):
        expected_context = contract["splits"]["train"][schedule[index]]
        checks.equal(row["train_index"], index, row["id"] + "/training prefix position")
        checks.equal(
            row["samples"], index + 1, row["id"] + "/all training decisions charged"
        )
        checks.equal(
            row["context_id"],
            expected_context["id"],
            row["id"] + "/balanced TRAIN context",
        )
        checks.equal(
            row["id"],
            f"{method}-{seed}-train-{index:05d}",
            row["id"] + "/method-specific training ID",
        )
        checks.equal(
            row["perturbation_mixed_units"],
            [0.0] * 5,
            row["id"] + "/no unreported training noise",
        )
    for row in rows:
        label = row["id"]
        checks.equal(row["method"], method, label + "/method lineage")
        checks.equal(row["seed"], seed, label + "/seed lineage")
        checks.require(row["context_id"] in layout_map, label + "/declared layout")
        role, context = layout_map[row["context_id"]]
        checks.equal(row["role"], role, label + "/no split relabeling")
        checks.equal(
            row["bucket_xy_m"], context["bucket_xy_m"], label + "/bucket remains fixed"
        )
        anchor = source_anchor(context["bucket_xy_m"], prior["records"])
        checks.equal(
            row["source_teacher_id"], anchor["id"], label + "/nearest source anchor"
        )
        if role == "train":
            checks.numeric(
                row["observation"],
                expected_observation(context["bucket_xy_m"], anchor),
                label + "/geometry-only observations",
                1e-7,
            )
        residual = np.asarray(row["normalized_residual"], float)
        checks.require(
            residual.shape == (5,)
            and np.isfinite(residual).all()
            and (np.abs(residual) <= 1 + 1e-12).all(),
            label + "/bounded continuous residual",
        )
        if method == "ppo":
            latent = row["latent"] if role == "train" else row["latent_mean"]
            command = (
                np.asarray(anchor["values"])
                + np.tanh(np.asarray(latent, dtype=float)) * RESIDUAL_SCALE
            )
            checks.numeric(
                residual,
                np.tanh(np.asarray(latent, dtype=np.float32)),
                label + "/continuous latent, not candidate index",
                1e-7,
            )
        else:
            command = np.asarray(anchor["values"]) + residual * RESIDUAL_SCALE
        checks.numeric(
            row["command_values"], command, label + "/unprojected command values", 2e-7
        )
        checks.numeric(
            row["values"],
            np.asarray(row["command_values"]) + row["perturbation_mixed_units"],
            label + "/realized requested geometry",
            1e-12,
        )

    frozen, sweep_states = {}, {}
    for samples in protocol["milestones"]:
        relative = (
            f"checkpoints/milestone-{samples:04d}.pt"
            if method == "ppo"
            else f"sweep/milestone-{samples:04d}.json"
        )
        path = contained(directory, relative)
        state = (
            load_checkpoint(path, checks, f"{directory.name}/{samples}")
            if method == "ppo"
            else read(path)
        )
        for key, value in (
            ("contract_fingerprint", contract["fingerprint"]),
            ("method", method),
            ("seed", seed),
            ("samples", samples),
        ):
            checks.equal(state[key], value, relative + "/declared checkpoint/" + key)
        frozen[samples] = dict(state=state, path=relative, sha256=sha(path))
        if method == "sweep":
            sweep_states[samples] = state["state"]
    if method == "sweep":
        checks.require(
            not (directory / "updates.jsonl").exists(),
            "Sweep has no disguised optimizer update",
        )
        reconstruct_sweep(
            seed, contract["splits"]["train"], train, checks, snapshots=sweep_states
        )
    else:
        audit_ppo_training(directory, contract, train, frozen, checks)

    curves = {p["samples"]: p for p in report["curves"]}
    checks.equal(
        len(curves), len(report["curves"]), directory.name + "/unique curve milestones"
    )
    checks.equal(
        set(curves),
        set(protocol["milestones"]),
        directory.name + "/all curve milestones",
    )
    for samples, point in curves.items():
        checks.equal(point["method"], method, "Curve method identity")
        checks.equal(point["seed"], seed, "Curve seed identity")
        checks.equal(
            point["artifact_sha256"],
            frozen[samples]["sha256"],
            "Curve frozen artifact identity",
        )
        cohort = [
            r for r in rows if r["role"] == "validation" and r["samples"] == samples
        ]
        checks.equal(len(cohort), 36, "Every validation layout and repeat retained")
        stats_equal(
            point["validation"], cohort, checks, f"Validation/{method}/{seed}/{samples}"
        )
    selected_samples = max(
        curves,
        key=lambda n: (
            curves[n]["validation"]["captures"],
            curves[n]["validation"]["mean_reward"],
            -n,
        ),
    )
    selection = read(directory / "selected.json")
    artifact = dict(
        path=frozen[selected_samples]["path"],
        sha256=frozen[selected_samples]["sha256"],
        samples=selected_samples,
    )
    checks.equal(
        selection["selected_samples"],
        selected_samples,
        "Validation-only checkpoint selection",
    )
    checks.equal(
        selection["artifact"], artifact, "Selected artifact frozen before test"
    )
    checks.equal(
        selection["validation_key"],
        [
            curves[selected_samples]["validation"]["captures"],
            curves[selected_samples]["validation"]["mean_reward"],
            -selected_samples,
        ],
        "Same checkpoint selection and tie rule",
    )
    checks.require(
        selection["frozen_before_test"] is True, "Test does not select its checkpoint"
    )
    checks.equal(report["selected_samples"], selected_samples, "Run selected milestone")
    checks.equal(
        report["checkpoint_sha256"], artifact["sha256"], "Run selected artifact"
    )
    tests = [r for r in rows if r["role"] == "test"]
    checks.equal(len(tests), 72, "Every frozen test layout and repeat retained")
    stats_equal(report["final_test"]["counts"], tests, checks, "Frozen test counts")
    checks.equal(
        report["final_test"]["frozen_artifact_sha256"],
        artifact["sha256"],
        "Final test selected artifact",
    )
    checks.require(
        report["final_test"]["search_at_inference"] is False,
        "No hidden final-test search",
    )
    audit_predictions(
        directory,
        contract,
        prior,
        rows,
        frozen,
        sweep_states,
        selected_samples,
        method,
        checks,
    )
    checks.equal(
        len(rows),
        budget + len(protocol["milestones"]) * 36 + 72,
        "Exact training plus evaluation budget",
    )
    stats_equal(report["train_counts"], train, checks, "Training request counts")
    checks.equal(report["train_requests"], budget, "Equal requested decision budget")
    checks.equal(
        report["train_native_releases"],
        counts(train)["native_releases"],
        "Actual native work kept separate",
    )
    provenance = check_native_provenance(directory, rows, prior, home_sha, checks)
    traces = rescore_rows(
        directory / "native", rows, scene, checks, directory.name + "/native"
    )
    checks.equal(
        len(traces),
        counts(rows)["native_releases"],
        "Every native release independently scalar rescored",
    )
    resource = check_resources(directory, report, checks)
    return (
        report,
        rows,
        dict(provenance=provenance, resources=resource, native_releases=len(traces)),
    )


def audit_ppo_training(directory, contract, train, frozen, checks):
    budget = contract["protocol"]["sample_budget"]
    zero = numpy_state(frozen[0]["state"]["policy"])
    checks.require(
        not zero["actor.4.weight"].any() and not zero["actor.4.bias"].any(),
        "PPO starts at exactly the common anchor",
    )
    checks.numeric(zero["logstd"], np.full(5, -1.5), "Declared initial exploration", 0)
    updates = lines(directory / "updates.jsonl")
    checks.equal(
        len(updates), budget // 32, "Every PPO behavior batch has an actual update"
    )
    previous, actor_changes = frozen[0]["state"], 0
    for index, update in enumerate(updates):
        label = f"{directory.name}/update-{index:04d}"
        checks.equal(update["update"], index, label + "/update sequence")
        checks.equal(
            update["samples"], (index + 1) * 32, label + "/equal requested batch prefix"
        )
        batch = train[index * 32 : (index + 1) * 32]
        checks.equal(
            update["train_ids"],
            [r["id"] for r in batch],
            label + "/TRAIN-only batch IDs",
        )
        for field in ("behavior", "after"):
            checks.equal(
                sha(contained(directory, update[field]["path"])),
                update[field]["sha256"],
                label + "/" + field + " bytes",
            )
        checks.equal(
            sha(contained(directory, update["buffer"])),
            update["buffer_sha256"],
            label + "/rollout buffer bytes",
        )
        before = load_checkpoint(
            contained(directory, update["behavior"]["path"]), checks, label + "/before"
        )
        after = load_checkpoint(
            contained(directory, update["after"]["path"]), checks, label + "/after"
        )
        for checkpoint, count in ((before, index * 32), (after, (index + 1) * 32)):
            checks.equal(
                checkpoint["contract_fingerprint"],
                contract["fingerprint"],
                label + "/checkpoint contract",
            )
            checks.equal(checkpoint["samples"], count, label + "/exact TRAIN prefix")
        checks.equal(
            before["weight_sha256"],
            previous["weight_sha256"],
            label + "/unbroken model update chain",
        )
        checks.equal(
            tensor_object_digest(before["optimizer"]),
            tensor_object_digest(previous["optimizer"]),
            label + "/unbroken optimizer update chain",
        )
        checks.equal(
            update["log"]["before"],
            before["weight_sha256"],
            label + "/declared before weights",
        )
        checks.equal(
            update["log"]["after"],
            after["weight_sha256"],
            label + "/declared after weights",
        )
        checks.equal(
            update["actor_before"],
            actor_module_sha256(before["policy"]),
            label + "/actor before",
        )
        checks.equal(
            update["actor_after"],
            actor_module_sha256(after["policy"]),
            label + "/actor after",
        )
        actor_changes += update["actor_before"] != update["actor_after"]
        with np.load(
            contained(directory, update["buffer"]), allow_pickle=False
        ) as stored:
            buffer = {k: stored[k] for k in stored.files}
        checks.equal(
            buffer["ids"].tolist(),
            [r["id"] for r in batch],
            label + "/buffer cannot contain evaluation or other-method rows",
        )
        checks.numeric(
            buffer["observation"],
            [r["observation"] for r in batch],
            label + "/observation buffer",
            0,
        )
        checks.numeric(
            buffer["latent"], [r["latent"] for r in batch], label + "/latent buffer", 0
        )
        checks.numeric(
            buffer["old_logp"],
            [r["old_logp"] for r in batch],
            label + "/behavior likelihood buffer",
            0,
        )
        checks.numeric(
            buffer["reward"],
            [row_reward(r) for r in batch],
            label + "/native reward buffer",
            1e-6,
        )
        check_behavior_likelihood(
            before["policy"],
            buffer["observation"],
            buffer["latent"],
            buffer["old_logp"],
            checks,
            label,
        )
        for row in batch:
            checks.equal(
                row["behavior_checkpoint_sha256"],
                update["behavior"]["sha256"],
                row["id"] + "/on-policy behavior checkpoint",
            )
            checks.equal(
                row["source_training_ids"],
                [],
                row["id"] + "/no cross-method imitation labels",
            )
        replay_ppo_update(
            before["policy"],
            before["optimizer"],
            after["policy"],
            dict(
                obs=buffer["observation"],
                **{k: buffer[k] for k in ("latent", "old_logp", "reward")},
            ),
            checks,
            label,
        )
        previous = after
        if after["samples"] in frozen:
            milestone = frozen[after["samples"]]["state"]
            checks.equal(
                milestone["weight_sha256"],
                after["weight_sha256"],
                label + "/milestone is its final update",
            )
            checks.equal(
                tensor_object_digest(milestone["optimizer"]),
                tensor_object_digest(after["optimizer"]),
                label + "/milestone optimizer",
            )
    checks.require(
        actor_changes > 0, "Observed-reward PPO changed actual actor weights"
    )


def audit_predictions(
    directory,
    contract,
    prior,
    rows,
    frozen,
    sweep_states,
    selected_samples,
    method,
    checks,
):
    prediction_files = []
    for role, checkpoints in (
        ("validation", contract["protocol"]["milestones"]),
        ("test", [selected_samples]),
    ):
        for samples in checkpoints:
            filename = f"predictions/{role}-{samples:04d}.json"
            prediction_files.append(filename)
            prediction = read(directory / filename)
            checks.equal(prediction["role"], role, filename + "/split role")
            checks.equal(prediction["samples"], samples, filename + "/frozen milestone")
            checks.equal(
                prediction["artifact"],
                dict(
                    path=frozen[samples]["path"],
                    sha256=frozen[samples]["sha256"],
                    samples=samples,
                ),
                filename + "/fixed inference artifact",
            )
            checks.require(
                prediction["selection_or_learning"] is False,
                filename + "/no evaluation outcome feedback",
            )
            contexts = contract["splits"][role]
            checks.equal(
                [p["context_id"] for p in prediction["rows"]],
                [r["id"] for r in contexts],
                filename + "/one deterministic base action per layout",
            )
            cohort = [r for r in rows if r["role"] == role and r["samples"] == samples]
            checks.equal(
                len(cohort), 3 * len(contexts), filename + "/exact paired-repeat budget"
            )
            for context, base in zip(contexts, prediction["rows"]):
                anchor = source_anchor(context["bucket_xy_m"], prior["records"])
                checks.equal(
                    base["source_teacher_id"],
                    anchor["id"],
                    filename + "/inference anchor rule",
                )
                checks.equal(
                    base["bucket_xy_m"],
                    context["bucket_xy_m"],
                    filename + "/fixed queried bucket",
                )
                checks.numeric(
                    base["observation"],
                    expected_observation(context["bucket_xy_m"], anchor),
                    filename + "/geometry-only inference inputs",
                    1e-7,
                )
                checks.equal(
                    base["artifact_sha256"],
                    frozen[samples]["sha256"],
                    filename + "/base action frozen artifact",
                )
                if method == "ppo":
                    expected_latent = network_forward(
                        frozen[samples]["state"]["policy"], [base["observation"]]
                    )[0]
                    checks.numeric(
                        base["latent_mean"],
                        expected_latent,
                        filename + "/frozen direct actor inference",
                        2e-6,
                    )
                    values = (
                        np.asarray(anchor["values"])
                        + np.tanh(expected_latent.astype(float)) * RESIDUAL_SCALE
                    )
                    checks.equal(
                        base["source_training_ids"],
                        [],
                        filename + "/no evaluation-time outcome teachers",
                    )
                else:
                    values, metadata = independent_sweep_prediction(
                        sweep_states[samples], context, anchor
                    )
                    for key, value in metadata.items():
                        checks.equal(
                            base[key], value, filename + "/TRAIN-only retrieval/" + key
                        )
                checks.numeric(
                    base["command_values"],
                    values,
                    filename + "/unsearched base action",
                    2e-6,
                )
                members = [r for r in cohort if r["context_id"] == context["id"]]
                checks.equal(
                    sorted(r["repeat"] for r in members),
                    [0, 1, 2],
                    filename + "/paired repeats including failures",
                )
                for row in members:
                    checks.equal(
                        row["command_values"],
                        base["command_values"],
                        row["id"] + "/single base prediction, not best-of-three",
                    )
                    checks.equal(
                        row["frozen_artifact_sha256"],
                        frozen[samples]["sha256"],
                        row["id"] + "/frozen inference artifact",
                    )
                    checks.numeric(
                        row["perturbation_mixed_units"],
                        expected_perturbation(context["id"], row["repeat"]),
                        row["id"] + "/paired fixed placement noise",
                        0,
                    )
            if role == "test":
                checks.equal(
                    prediction["query_s"],
                    read(directory / "report.json")["final_test"]["query_s"],
                    "Final query time is its recorded fixed-layout prediction time",
                )
    checks.equal(
        set(prediction_files),
        {
            str(p.relative_to(directory))
            for p in (directory / "predictions").glob("*.json")
        },
        "No hidden extra evaluation or test-query files",
    )


def execution_identity(repository, ref):
    from sircus.mjlab.corrected_contract import PHYSICS

    source_hashes = dict(ref["source_hashes"])
    for name in (
        "benchmark_contract.py",
        "benchmark_adapter.py",
        "benchmark_policy.py",
        "benchmark_resources.py",
        "benchmark_run.py",
        "benchmark_study.py",
        "learning_policy.py",
    ):
        relative = "sircus/mjlab/" + name
        source_hashes[relative] = sha(Path(repository) / relative)
    return dict(
        contract_fingerprint=ref["contract"]["fingerprint"],
        scene_sha256=digest(ref["scene"]),
        asset_sha256=ref["asset_hashes"],
        source_sha256=source_hashes,
        physics=PHYSICS,
    )


def audit_controls(study, reference, assets, *, repository):
    """Audit the quarantined TRAIN-only engineering controls on CPU."""
    import torch
    from sircus.mjlab.sweep_worker import load_reference
    from sircus.mjlab.audit_grid_study import rescore_rows

    begin, checks = time.monotonic(), Checks()
    study, repository = Path(study), Path(repository)
    check_manifest(study, checks)
    report = read(study / "report.json")
    prior = verify_prior_bundle(repository, checks=checks)
    source_prior = {
        k: prior[k]
        for k in (
            "records",
            "records_sha256",
            "bundle_manifest_sha256",
            "recipes",
            "latest_study_native_releases",
            "latest_study_requested_trials",
            "full_historical_cost_known",
        )
    }
    source_prior["scope"] = (
        "Shared historical starting information, not 17 free cold-start labels. "
        "Latest two studies cost 4234 native releases; predecessor acquisition, "
        "contact tuning and human design costs are not fully quantified."
    )
    checks.equal(
        report["prior_sha256"],
        digest(source_prior),
        "Engineering control exact shared prior",
    )
    ref = load_reference(reference, assets, repository=repository)
    checks.equal(
        report["identity"],
        execution_identity(repository, ref),
        "Engineering control exact critical source and physics",
    )
    checks.equal(
        report["source_sha256"],
        sha(repository / "sircus/mjlab/benchmark_smoke.py"),
        "Engineering control source",
    )
    for key, value in (
        ("status", "passed"),
        ("requests", 50),
        ("controls_captured", 17),
        ("deliberate_rejects", 1),
        ("discarded_ppo_requests", 32),
        ("actor_changed", True),
    ):
        checks.equal(report[key], value, "Separate control gate/" + key)
    raw = lines(study / "native/shots.jsonl")
    checks.equal(len(raw), 50, "All quarantined control requests retained")
    controls = read(study / "controls.json")["rows"]
    smoke = read(study / "ppo-smoke.json")
    bad = [r for r in raw if r["id"] == "deliberate-invalid-global-bounds"]
    checks.equal(
        [r["id"] for r in controls],
        ["control-" + r["id"] for r in source_prior["records"]],
        "Exactly17 source controls, no heldouts",
    )
    checks.require(
        len(bad) == 1 and not bad[0]["native_release"] and bool(bad[0]["invalid"]),
        "Deliberate bad control is a rejected requested decision",
    )
    enriched = {r["id"]: r for r in controls + smoke["rows"]}
    rows = []
    for original in raw:
        row = copy.deepcopy(enriched.get(original["id"], original))
        if original["id"] == "deliberate-invalid-global-bounds":
            # The adapter's rejection text is enriched after its durable raw row.
            # Restore only saved immutable candidate metadata from its preparation.
            proposal = original["proposal"]
            preparation = read(study / "native" / proposal["preparation_file"])
            candidate = preparation["candidates"][proposal["preparation_row"]]
            row.update(
                {
                    k: candidate[k]
                    for k in (
                        "endpoint_eligible",
                        "anchor_admitted",
                        "geometry_eligible",
                        "geometry_row",
                    )
                }
            )
            row.update(
                preparation_file=proposal["preparation_file"],
                preparation_sha256=proposal["preparation_sha256"],
                anchor_file=proposal["anchor_file"],
                anchor_sha256=proposal["anchor_sha256"],
                motion_evaluated=False,
                motion_admitted=False,
            )
        row.update(
            role=original["phase"],
            source_hint=original["proposal"]["source_hint"],
            adapter_directory="native",
        )
        rows.append(row)
    check_row_invariants(rows, checks)
    checks.require(
        all(
            r["native_release"] and r["route_success"] and not r.get("invalid")
            for r in controls
        ),
        "Every exact historical control captures",
    )
    check_native_provenance(
        study, rows, source_prior, ref["contract"]["common_home"]["sha256"], checks
    )
    traces = rescore_rows(
        study / "native", rows, ref["scene"], checks, "engineering controls"
    )
    checks.equal(len(traces), report["native_releases"], "Control native release count")
    before = torch.load(study / "ppo-before.pt", map_location="cpu", weights_only=True)
    after = torch.load(study / "ppo-after.pt", map_location="cpu", weights_only=True)
    with np.load(study / "ppo-buffer.npz", allow_pickle=False) as stored:
        buffer = {k: stored[k] for k in stored.files}
    checks.equal(buffer["observation"].shape, (32, 9), "Discarded PPO smoke batch size")
    checks.numeric(
        buffer["reward"],
        [row_reward(r) for r in smoke["rows"]],
        "Discarded PPO smoke native reward",
        1e-6,
    )
    checks.require(
        np.ptp(buffer["reward"]) > 1e-6
        and counts(smoke["rows"])["native_releases"] > 0,
        "Smoke has real native work and nonconstant feedback",
    )
    check_behavior_likelihood(
        before["policy"],
        buffer["observation"],
        buffer["latent"],
        buffer["old_logp"],
        checks,
        "discarded PPO smoke",
    )
    replay_ppo_update(
        before["policy"],
        before["optimizer"],
        after["policy"],
        dict(
            obs=buffer["observation"],
            **{k: buffer[k] for k in ("latent", "old_logp", "reward")},
        ),
        checks,
        "discarded PPO smoke",
    )
    checks.equal(
        smoke["actor_before"],
        actor_module_sha256(before["policy"]),
        "Smoke actor before identity",
    )
    checks.equal(
        smoke["actor_after"],
        actor_module_sha256(after["policy"]),
        "Smoke actor after identity",
    )
    checks.require(
        smoke["actor_before"] != smoke["actor_after"],
        "Smoke actually changes actor weights",
    )
    return dict(
        version=VERSION,
        status="passed",
        checks=checks.count,
        failures=[],
        report_sha256=sha(study / "report.json"),
        identity=report["identity"],
        requested_trials=50,
        native_releases=len(traces),
        auditor_sha256=sha(Path(__file__)),
        elapsed_s=time.monotonic() - begin,
        scope="Independent CPU audit of separately quarantined TRAIN-only engineering controls, native scalar scores, geometry/path proof bindings and discarded PPO optimizer update; no new rollout or benchmark training data.",
    )


def audit(study, reference, assets, *, repository):
    from sircus.mjlab.sweep_worker import load_reference

    begin, checks = time.monotonic(), Checks()
    study, repository = Path(study), Path(repository)
    manifest = check_manifest(
        study, checks, ignored=("controller.lock", "active-worker.json")
    )
    contract, report, prior = (
        read(study / name) for name in ("contract.json", "report.json", "prior.json")
    )
    checked_prior = verify_prior_bundle(repository, prior, checks=checks)
    check_contract_protocol(contract, prior, checks)
    fingerprint(report, checks, "Overall report")
    checks.equal(report["status"], "completed", "Overall report frozen before audit")
    checks.equal(report["phase"], "completed", "Every finite worker completed")
    checks.equal(
        report["contract_fingerprint"],
        contract["fingerprint"],
        "Overall report contract binding",
    )
    checks.equal(
        report["protocol"], contract["protocol"], "Overall report protocol unchanged"
    )
    checks.equal(
        read(study / "splits.json"),
        contract["splits"],
        "Stored split manifest unchanged",
    )
    ref = load_reference(reference, assets, repository=repository)
    expected_identity = execution_identity(repository, ref)
    source_hashes = expected_identity["source_sha256"]
    checks.equal(
        contract["identity"],
        expected_identity,
        "Executed source, scene, assets and corrected physics identity",
    )
    checks.equal(
        contract["packages"], ref["packages"], "Pinned native package versions"
    )
    checks.equal(
        sha(study / "preflight.json"),
        contract["preflight"]["report_sha256"],
        "Archived engineering-control bytes",
    )
    preflight = read(study / "preflight.json")
    checks.equal(preflight["status"], "passed", "Separate native/PPO smoke passed")
    checks.equal(
        preflight["identity"],
        contract["identity"],
        "Engineering controls use the same critical source",
    )
    checks.equal(
        preflight["prior_sha256"],
        contract["prior_sha256"],
        "Engineering controls use the same shared prior",
    )
    checks.equal(
        preflight["source_sha256"],
        sha(repository / "sircus/mjlab/benchmark_smoke.py"),
        "Engineering-control source byte identity",
    )
    for key, value in (
        ("requests", 50),
        ("controls_captured", 17),
        ("deliberate_rejects", 1),
        ("discarded_ppo_requests", 32),
        ("actor_changed", True),
    ):
        checks.equal(preflight[key], value, "Separate quarantined control gate/" + key)
    checks.require(
        preflight["behavior_logp_max_error"] <= 1e-6,
        "Engineering-control behavior likelihood gate",
    )
    for key in ("requests", "native_releases"):
        checks.equal(
            preflight[key],
            contract["preflight"][key],
            "Control cost retained separately/" + key,
        )
    checks.require(
        17 <= preflight["native_releases"] <= 49,
        "Control native work is separate and finite",
    )
    p = contract["protocol"]
    expected_pairs = {(method, seed) for method in METHODS for seed in p["seeds"]}
    actual_directories = {
        directory.name for directory in (study / "runs").iterdir() if directory.is_dir()
    }
    checks.equal(
        actual_directories,
        {f"{method}-seed{seed}" for method, seed in expected_pairs},
        "All and only declared worker directories",
    )
    run_results, all_rows, summaries = {}, [], []
    for method, seed in sorted(expected_pairs):
        directory = study / "runs" / f"{method}-seed{seed}"
        run_report, rows, summary = audit_run(
            directory,
            contract,
            prior,
            ref["scene"],
            ref["contract"]["common_home"]["sha256"],
            checks,
        )
        run_results[(method, seed)] = run_report
        all_rows.extend(rows)
        summaries.append(
            dict(
                method=method,
                seed=seed,
                requested=counts(rows)["attempts"],
                training_requests=p["sample_budget"],
                native_releases=counts(rows)["native_releases"],
                **summary,
            )
        )
    checks.equal(
        len(all_rows),
        len({r["id"] for r in all_rows}),
        "Trial IDs are unique across methods, seeds and roles",
    )
    reported_pairs = [(r["method"], r["seed"]) for r in report["runs"]]
    checks.equal(
        len(reported_pairs),
        len(expected_pairs),
        "Every overall run summary occurs once",
    )
    checks.equal(
        set(reported_pairs),
        expected_pairs,
        "Overall run summaries match every method/seed",
    )
    for run_report in report["runs"]:
        checks.equal(
            run_report,
            run_results[(run_report["method"], run_report["seed"])],
            "Overall summary retains exact signed worker report",
        )
    expected_curves = {
        (r["method"], r["seed"], point["samples"]): point
        for r in run_results.values()
        for point in r["curves"]
    }
    curve_keys = [(r["method"], r["seed"], r["samples"]) for r in report["curves"]]
    checks.equal(
        len(curve_keys),
        len(expected_curves),
        "Every learning-curve point retained once",
    )
    checks.equal(
        set(curve_keys),
        set(expected_curves),
        "No omitted or extra learning-curve seeds/budgets",
    )
    for point in report["curves"]:
        checks.equal(
            point,
            expected_curves[(point["method"], point["seed"], point["samples"])],
            "Learning curve copied from fixed validation cohort",
        )
    finals = {(r["method"], r["seed"]): r["final_test"] for r in run_results.values()}
    final_pairs = [(r["method"], r["seed"]) for r in report["final_test"]]
    checks.equal(
        len(final_pairs), len(expected_pairs), "Every final test cohort appears once"
    )
    checks.equal(
        set(final_pairs), expected_pairs, "All declared final test cohorts retained"
    )
    for final in report["final_test"]:
        checks.equal(
            final,
            finals[(final["method"], final["seed"])],
            "Final test retains exact frozen worker outcome",
        )
    expected_requests = len(expected_pairs) * (
        p["sample_budget"] + 36 * len(p["milestones"]) + 72
    )
    checks.equal(
        len(all_rows),
        expected_requests,
        "Total new-decision and evaluation budgets, separate from controls/history",
    )
    return dict(
        version=VERSION,
        status="passed",
        checks=checks.count,
        failures=[],
        contract_fingerprint=contract["fingerprint"],
        report_sha256=sha(study / "report.json"),
        report_fingerprint=report["fingerprint"],
        study_manifest_sha256=sha(study / "manifest.json"),
        auditor_sha256=sha(Path(__file__)),
        source_sha256=source_hashes,
        prior_records_sha256=prior["records_sha256"],
        prior_published_files=len(checked_prior["files_sha256"]),
        requested_trials=len(all_rows),
        native_releases=counts(all_rows)["native_releases"],
        role_counts={
            role: counts([r for r in all_rows if r["role"] == role])
            for role in ("train", "validation", "test")
        },
        controls=dict(
            requests=preflight["requests"],
            native_releases=preflight["native_releases"],
            used_for_training=False,
        ),
        runs=summaries,
        maximum_numeric_error=checks.numeric_error,
        elapsed_s=time.monotonic() - begin,
        scope="Independent CPU identity/budget/split/source-prior checks, TRAIN-only sweep replay, behavior likelihoods and actual Adam/PPO update replay, frozen direct-inference reconstruction, scalar rescoring of every saved native trajectory, and resource-observation consistency. No new physics rollout.",
        limitations=[
            "Source-bound recorded endpoint/common-home/local-path proofs checked; geometry was not independently rerun.",
            "NVIDIA VRAM is a sampled process peak. PyTorch allocator peaks are separate. Process CPU time excludes external telemetry subprocess cost.",
            "Training/setup wall time excludes evaluation; lazy native initialization is recorded separately. Reservation time includes evaluation.",
            "Stored engineering-control report is source-bound; its raw trajectories are kept separately from this benchmark.",
            "Current fixed-ball, geometry-observation, restricted residual shot selection with held endpoints; no grasp, driven motion, compliance or physical validation.",
        ],
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("study", "reference", "assets", "repository", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument(
        "--controls-only",
        action="store_true",
        help="Audit separately quarantined TRAIN-only engineering controls",
    )
    args = parser.parse_args()
    if args.output.resolve().is_relative_to(args.study.resolve()):
        parser.error("Audit output must be outside immutable study evidence")
    try:
        checker = audit_controls if args.controls_only else audit
        result = checker(
            args.study, args.reference, args.assets, repository=args.repository
        )
    except Exception as exc:
        result = dict(
            version=VERSION,
            status="failed",
            checks=0,
            failures=[f"{type(exc).__name__}: {exc}"],
        )
        for name, key in (
            ("report.json", "report_sha256"),
            ("contract.json", "contract_sha256"),
        ):
            path = args.study / name
            if path.is_file():
                result[key] = sha(path)
        contract = args.study / "contract.json"
        if contract.is_file():
            try:
                result["contract_fingerprint"] = read(contract).get("fingerprint")
            except ValueError:
                pass
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                k: result.get(k)
                for k in (
                    "status",
                    "checks",
                    "failures",
                    "requested_trials",
                    "native_releases",
                )
            }
        )
    )
    if result["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
