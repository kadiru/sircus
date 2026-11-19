"""Versioned, candidate-conditioned contextual-bandit PPO study utilities.

No Isaac, hardware, Torch or rendering dependency is initialized on import.
Candidate qualification and release scoring are separate from learned ranking.
"""

import copy
import hashlib
import json
import math

import numpy as np

VERSION = "sircus-candidate-shot-learning-v1"
REWARD_VERSION = "sircus-graded-shot-reward-v1"
SEED = 20261110
PARAMETERS = ("azimuth_offset_deg", "radius_m", "height_m", "gap_m", "tilt_deg")
FEATURE_NAMES = (
    "bearing_sin",
    "bearing_cos",
    "radius_normalized",
    "board_height_normalized",
    "drop_gap_normalized",
    "tilt_normalized",
)


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, allow_nan=False).encode()
    ).hexdigest()


def study_contract(spec, bank_sha256, train_batches=128, eval_repeats=16, seed=SEED):
    if (
        spec.get("version") != "sircus-expanded-cylinder-film-v1"
        or spec.get("num_envs") != 35
        or len(spec.get("cells", [])) != 35
        or len({tuple(c["bucket_xy_m"]) for c in spec["cells"]}) != 35
    ):
        raise ValueError("Expected the unchanged 35 distinct expanded bucket sites")
    if (
        type(train_batches) is not int
        or not 8 <= train_batches <= 512
        or train_batches % 2
        or type(eval_repeats) is not int
        or not 2 <= eval_repeats <= 64
        or type(seed) is not int
    ):
        raise ValueError("Study budget must be bounded, even and reproducible")
    value = dict(
        version=VERSION,
        source_fingerprint=spec["fingerprint"],
        offline_bank_sha256=bank_sha256,
        seed=seed,
        num_envs=35,
        training_batches=train_batches,
        training_throws=train_batches * 35,
        batches_per_update=2,
        evaluation_repeats=eval_repeats,
        best_setting_repeats=4,
        observation_dim=5,
        candidate_feature_names=list(FEATURE_NAMES),
        actor_input_dim=11,
        action="Masked categorical index into that cell's native-accepted geometric candidate bank; selected candidate supplies five physical shot parameters.",
        actor="Shared 11->96->96->1 tanh MLP; no cell IDs or per-cell learned parameters; zero-initialized final layer gives uniform valid-action exploration.",
        critic="Shared 5->64->64->1 tanh MLP over bucket geometry.",
        policy_inputs="Five privileged bucket features plus six candidate geometry features; no camera, point cloud, outcome label or cell index.",
        reward_version=REWARD_VERSION,
        reward_scope="Graded proximity/retention shaping remains after external contact with a 0.8 penalty. Existing closest-distance and retention may include post-contact states; they are not precontact measurements. Capture bonus requires unchanged route_success.",
        lr=0.0003,
        clip=0.2,
        epochs=4,
        value_coef=0.5,
        entropy_coef=0.01,
        grad_norm=1.0,
        terminal=True,
        bootstrap=False,
        evaluation_selection="Final checkpoint after the predeclared budget; no evaluation-based checkpoint selection or optimization.",
        evaluation_scope="Fresh simulator episodes at the same 35 development bucket coordinates. Stochastic repeats sample qualified configurations; each deterministic result is one greedy episode, not a reliability, physical-noise robustness or unseen-layout estimate.",
        menu_selection="Every native-accepted candidate remains available; captures/rewards do not remove failures or choose the training action bank.",
        setup_scope="Direct endpoint initialization for headless learning, unchanged native measured settling and 240 Hz capture/route scorer; no learned grasp, joint path or driven transit certificate.",
    )
    value["fingerprint"] = digest(value)
    return value


def shot_features(action, low, high):
    values = np.asarray([action[k] for k in PARAMETERS], dtype=float)
    low, high = np.asarray(low, dtype=float), np.asarray(high, dtype=float)
    if (
        values.shape != (5,)
        or low.shape != (5,)
        or high.shape != (5,)
        or not np.isfinite(values).all()
        or np.any(high <= low)
        or np.any(values < low - 1e-9)
        or np.any(values > high + 1e-9)
    ):
        raise ValueError("Invalid candidate shot feature values or bounds")
    angle = math.radians(values[0])
    scaled = 2 * (values[1:] - low[1:]) / (high[1:] - low[1:]) - 1
    return np.r_[math.sin(angle), math.cos(angle), scaled].astype(np.float32)


def candidate_tensor_data(cells, menus, low, high):
    from cylinder_farm import observation

    if len(cells) != len(menus) or any(not options for options in menus):
        raise ValueError("Every bucket needs a nonempty native-qualified bank")
    width = max(map(len, menus))
    features = np.zeros((len(cells), width, 11), dtype=np.float32)
    mask = np.zeros((len(cells), width), dtype=bool)
    observations = np.array([observation(c) for c in cells], dtype=np.float32)
    for i, options in enumerate(menus):
        for j, action in enumerate(options):
            features[i, j] = np.r_[observations[i], shot_features(action, low, high)]
            mask[i, j] = True
    return observations, features, mask


def choose_actions(menus, indices):
    raw = np.asarray(indices)
    if raw.shape != (len(menus),) or not np.issubdtype(raw.dtype, np.integer):
        raise ValueError("Expected one integer action index per table")
    if any(not 0 <= int(j) < len(m) for m, j in zip(menus, raw)):
        raise ValueError("Policy chose a padded or unavailable candidate")
    return [copy.deepcopy(m[int(j)]) for m, j in zip(menus, raw)]


def learning_reward(row):
    """Keep graded misses and qualified capture bonuses under a separate version."""
    if row.get("invalid"):
        return -1.0
    external = bool(row["external_contact_steps"])
    if not row["board_bounce"]:
        return -0.25 - 0.5 * external
    distance = float(row["closest_opening_distance_m"])
    retention = float(row["max_retention_s"])
    if not math.isfinite(distance) or distance < 0 or not math.isfinite(retention):
        raise ValueError("Invalid measured distance/retention in reward")
    if row["route_success"] and external:
        raise ValueError("Route scorer cannot qualify an external-contact capture")
    return float(
        0.15
        + 0.40 * math.exp(-distance / 0.12)
        + 0.25 * min(max(retention, 0.0), 1.0)
        - 0.80 * external
        + float(row["route_success"])
        + 0.10 * bool(row["route_success"] and row["route_quality"] == 2)
        + 0.05 * bool(row["single_bounce_success"])
    )


def summary(rows):
    from farm_contract import reward as historical_reward

    return dict(
        attempts=len(rows),
        releases=sum(not r["invalid"] for r in rows),
        invalid=sum(bool(r["invalid"]) for r in rows),
        captures=sum(bool(r["route_success"]) and not r["invalid"] for r in rows),
        clean=sum(r["route_quality"] == 2 and not r["invalid"] for r in rows),
        single_bounce=sum(
            bool(r["single_bounce_success"]) and not r["invalid"] for r in rows
        ),
        mean_learning_reward=(
            float(np.mean([learning_reward(r) for r in rows])) if rows else None
        ),
        mean_historical_reward=(
            float(np.mean([historical_reward(r) for r in rows])) if rows else None
        ),
    )


def per_cell(cells, rows):
    return [
        dict(
            cell_id=c["id"],
            bucket_xy_m=list(c["bucket_xy_m"]),
            **summary([r for r in rows if r["cell_id"] == c["id"]]),
        )
        for c in cells
    ]


def make_policy():
    """Lazy Torch import keeps contract/reward inspection CPU dependency-light."""
    import torch

    class CandidatePolicy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.actor = torch.nn.Sequential(
                torch.nn.Linear(11, 96),
                torch.nn.Tanh(),
                torch.nn.Linear(96, 96),
                torch.nn.Tanh(),
                torch.nn.Linear(96, 1),
            )
            self.critic = torch.nn.Sequential(
                torch.nn.Linear(5, 64),
                torch.nn.Tanh(),
                torch.nn.Linear(64, 64),
                torch.nn.Tanh(),
                torch.nn.Linear(64, 1),
            )
            torch.nn.init.zeros_(self.actor[-1].weight)
            torch.nn.init.zeros_(self.actor[-1].bias)

        def distribution(self, features, mask):
            if (
                features.ndim != 3
                or features.shape[-1] != 11
                or mask.shape != features.shape[:2]
                or not bool(mask.any(dim=-1).all())
            ):
                raise ValueError("Invalid candidate policy features/mask")
            logits = self.actor(features).squeeze(-1)
            return torch.distributions.Categorical(
                logits=logits.masked_fill(~mask, -torch.inf)
            )

    return CandidatePolicy()


def ppo_update(
    policy,
    optimizer,
    observations,
    features,
    masks,
    indices,
    old_logp,
    returns,
    contract,
):
    import torch
    from farm_policy import weight_hash

    with torch.no_grad():
        advantage = returns - policy.critic(observations).squeeze(-1)
        advantage = (advantage - advantage.mean()) / (
            advantage.std(unbiased=False) + 1e-6
        )
    old_logp = old_logp.detach()
    before = weight_hash(policy)
    epochs = []
    for _ in range(contract["epochs"]):
        distribution = policy.distribution(features, masks)
        logp = distribution.log_prob(indices)
        ratio = (logp - old_logp).exp()
        actor_loss = -torch.minimum(
            ratio * advantage,
            ratio.clamp(1 - contract["clip"], 1 + contract["clip"]) * advantage,
        ).mean()
        critic_loss = ((policy.critic(observations).squeeze(-1) - returns) ** 2).mean()
        entropy = distribution.entropy().mean()
        loss = (
            actor_loss
            + contract["value_coef"] * critic_loss
            - contract["entropy_coef"] * entropy
        )
        if not torch.isfinite(loss):
            raise RuntimeError("Nonfinite categorical PPO loss")
        optimizer.zero_grad()
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(
            policy.parameters(), contract["grad_norm"]
        )
        if not torch.isfinite(norm):
            raise RuntimeError("Nonfinite categorical PPO gradient")
        optimizer.step()
        epochs.append(
            dict(
                loss=float(loss.detach()),
                actor_loss=float(actor_loss.detach()),
                critic_loss=float(critic_loss.detach()),
                categorical_entropy=float(entropy.detach()),
                gradient_norm=float(norm),
                clip_fraction=float(
                    ((ratio - 1).abs() > contract["clip"]).float().mean().detach()
                ),
            )
        )
    return dict(before=before, after=weight_hash(policy), epochs=epochs)
