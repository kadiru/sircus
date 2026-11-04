"""Shared categorical PPO over frozen native-qualified shot geometries.

This policy learns which endpoint parameters to select from fresh native rewards.
Candidate geometry is an input; candidate IDs and stored outcomes never are.
Physics, scoring and scripted articulation transit belong to the native runner.
"""

import copy
import hashlib
import re
from pathlib import Path

import numpy as np

try:
    import torch
except ModuleNotFoundError:
    torch = None

POLICY_VERSION = "sircus-categorical-shot-policy-v1"
OPTIONS = 24
OBSERVATION_DIM = 5
GEOMETRY_DIM = 5
CANDIDATE_DIM = 2 * GEOMETRY_DIM
ARCHITECTURE = dict(
    version=POLICY_VERSION,
    observation_dim=OBSERVATION_DIM,
    candidate_dim=CANDIDATE_DIM,
    options=OPTIONS,
    actor_input_dim=OBSERVATION_DIM + CANDIDATE_DIM,
    hidden=[64, 64],
    activation="tanh",
    actor="shared geometry-conditioned candidate scorer; zero output initialization",
    critic="shared state-only scalar critic",
)
PPO_DEFAULTS = dict(
    epochs=4, clip=0.2, value_coef=0.5, entropy_coef=0.01, grad_norm=1.0
)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _torch():
    if torch is None:
        raise ImportError("Categorical shot PPO requires the configured PyTorch environment")
    return torch


def _sha(value, label):
    _require(isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value),
             f"Invalid {label} SHA256")
    return value


def _counts(updates, training_batches):
    _require(type(updates) is int and type(training_batches) is int
             and 0 <= updates <= 64 and 0 <= training_batches <= 128
             and updates == training_batches // 2,
             "Checkpoint counters differ from the two-batch, 64-update study")


def build_inputs(cells, menus):
    """Return privileged state and geometry features without consulting outcomes.

    Each candidate has five global coordinates in [-1,1], followed by the same
    five coordinates normalized to its cell's qualified menu range. A constant
    dimension has local feature zero. The local features resolve small physical
    differences that would otherwise disappear within the broad global bounds.
    """
    from expanded_farm import ACTION_NAMES, HIGH, LOW, observation, validate_action

    _require(0 < len(cells) == len(menus), "Expected one qualified menu per cell")
    observations, candidates = [], []
    for cell, options in zip(cells, menus):
        _require(len(options) == OPTIONS, "Every cell requires exactly 24 qualified options")
        values = np.asarray([validate_action(cell, option) for option in options], dtype=float)
        _require(values.shape == (OPTIONS, GEOMETRY_DIM)
                 and len({tuple(v) for v in values}) == OPTIONS,
                 "Qualified menu must contain 24 distinct physical geometries")
        # Keep the feature ordering bound to the five-axis native task contract.
        _require(tuple(ACTION_NAMES) == ("azimuth_offset_deg", "radius_m", "height_m", "gap_m", "tilt_deg"),
                 "Unsupported shot geometry feature ordering")
        global_values = 2 * (values - LOW) / (HIGH - LOW) - 1
        lower, upper = values.min(axis=0), values.max(axis=0)
        half_span, midpoint = (upper - lower) / 2, (upper + lower) / 2
        local_values = np.zeros_like(values)
        np.divide(values - midpoint, half_span, out=local_values, where=half_span > 1e-12)
        state = np.asarray(observation(cell), dtype=float)
        _require(state.shape == (OBSERVATION_DIM,) and np.isfinite(state).all(),
                 "Invalid privileged shot observation")
        observations.append(state)
        candidates.append(np.concatenate([global_values, local_values], axis=-1))
    obs, features = np.asarray(observations, dtype=np.float32), np.asarray(candidates, dtype=np.float32)
    _require(np.isfinite(features).all() and np.max(np.abs(features)) <= 1 + 1e-6,
             "Invalid normalized candidate geometry")
    return obs, features


def actions_from_indices(cells, menus, indices):
    """Decode selected prototypes exactly, retaining their scene and trial IDs."""
    from expanded_farm import validate_action

    if torch is not None and isinstance(indices, torch.Tensor):
        indices = indices.detach().cpu().numpy()
    selected = np.asarray(indices)
    _require(selected.shape == (len(cells),) and len(menus) == len(cells)
             and np.issubdtype(selected.dtype, np.integer)
             and not np.issubdtype(selected.dtype, np.bool_)
             and ((selected >= 0) & (selected < OPTIONS)).all(),
             "Expected one integer categorical action in [0,24) per cell")
    actions = []
    for cell, menu, index in zip(cells, menus, selected):
        _require(len(menu) == OPTIONS, "Categorical action menu size changed")
        action = copy.deepcopy(menu[int(index)])
        validate_action(cell, action)
        actions.append(action)
    return actions


_BaseModule = torch.nn.Module if torch is not None else object


class MenuPolicy(_BaseModule):
    """Permutation-equivariant candidate actor and privileged-state critic."""

    def __init__(self):
        _torch()
        super().__init__()

        def mlp(inputs, outputs):
            return torch.nn.Sequential(
                torch.nn.Linear(inputs, 64), torch.nn.Tanh(),
                torch.nn.Linear(64, 64), torch.nn.Tanh(),
                torch.nn.Linear(64, outputs),
            )

        self.actor = mlp(OBSERVATION_DIM + CANDIDATE_DIM, 1)
        self.critic = mlp(OBSERVATION_DIM, 1)
        # Exactly uniform initial action probabilities. No source policy, action
        # outcome, successful setting or cell-specific learned bias is loaded.
        torch.nn.init.zeros_(self.actor[-1].weight)
        torch.nn.init.zeros_(self.actor[-1].bias)

    @staticmethod
    def _observations(observations):
        _require(observations.ndim == 2 and observations.shape[1] == OBSERVATION_DIM
                 and torch.isfinite(observations).all().item(), "Invalid policy observations")

    def logits(self, observations, candidates):
        self._observations(observations)
        _require(candidates.shape == (len(observations), OPTIONS, CANDIDATE_DIM)
                 and torch.isfinite(candidates).all().item(), "Invalid policy candidates")
        states = observations[:, None, :].expand(-1, OPTIONS, -1)
        return self.actor(torch.cat([states, candidates], dim=-1)).squeeze(-1)

    def distribution(self, observations, candidates):
        return torch.distributions.Categorical(logits=self.logits(observations, candidates))

    def value(self, observations):
        self._observations(observations)
        return self.critic(observations).squeeze(-1)


def sample(policy, observations, candidates, generator=None):
    """Sample on-policy actions; a separate generator isolates assessment RNG."""
    _torch()
    with torch.no_grad():
        distribution = policy.distribution(observations, candidates)
        indices = torch.multinomial(distribution.probs, 1, generator=generator).squeeze(-1)
        return dict(indices=indices, logp=distribution.log_prob(indices),
                    entropy=distribution.entropy(), value=policy.value(observations))


def actor_weight_hash(policy):
    """Prove actor changes separately from value-function-only updates."""
    _torch()
    result = hashlib.sha256()
    for name, value in sorted(policy.actor.state_dict().items()):
        result.update(name.encode())
        result.update(value.detach().cpu().contiguous().numpy().tobytes())
    return result.hexdigest()


def weight_hash(policy):
    _torch()
    result = hashlib.sha256()
    for name, value in sorted(policy.state_dict().items()):
        result.update(name.encode())
        result.update(value.detach().cpu().contiguous().numpy().tobytes())
    return result.hexdigest()


def update(policy, optimizer, observations, candidates, indices, old_logp, returns, spec):
    """Clipped categorical PPO for terminal one-shot contextual episodes.

    Every attempted slot remains in the buffer, including admission rejections.
    Returns are the unchanged native scalar reward, without bootstrap or GAE.
    Diagnostics distinguish policy changes, entropy, clipping and value fitting.
    """
    _torch()
    config = {name: spec[name] for name in PPO_DEFAULTS}
    _require(type(config["epochs"]) is int and 1 <= config["epochs"] <= 16,
             "Unbounded PPO epoch count")
    _require(all(np.isfinite(v) for v in config.values())
             and 0 < config["clip"] < 1 and config["value_coef"] >= 0
             and config["entropy_coef"] >= 0 and config["grad_norm"] > 0,
             "Invalid PPO hyperparameters")
    count = len(observations)
    _require(count > 1 and indices.shape == old_logp.shape == returns.shape == (count,)
             and indices.dtype == torch.long and torch.isfinite(old_logp).all().item()
             and torch.isfinite(returns).all().item(), "Invalid terminal PPO buffer")
    with torch.no_grad():
        advantages = returns - policy.value(observations)
        advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-6)
    old_logp = old_logp.detach()
    before, actor_before, logs = weight_hash(policy), actor_weight_hash(policy), []
    for _ in range(config["epochs"]):
        distribution = policy.distribution(observations, candidates)
        logp = distribution.log_prob(indices)
        log_ratio = logp - old_logp
        ratio = log_ratio.exp()
        actor = -torch.minimum(ratio * advantages,
                               ratio.clamp(1-config["clip"], 1+config["clip"]) * advantages).mean()
        value = ((policy.value(observations) - returns) ** 2).mean()
        entropy = distribution.entropy().mean()
        loss = actor + config["value_coef"] * value - config["entropy_coef"] * entropy
        if not torch.isfinite(loss):
            raise RuntimeError("Nonfinite categorical PPO loss")
        optimizer.zero_grad()
        loss.backward()
        grad = torch.nn.utils.clip_grad_norm_(policy.parameters(), config["grad_norm"])
        if not torch.isfinite(grad):
            raise RuntimeError("Nonfinite categorical PPO gradient")
        optimizer.step()
        logs.append(dict(loss=float(loss.detach()), actor_loss=float(actor.detach()),
                         critic_loss=float(value.detach()), categorical_entropy=float(entropy.detach()),
                         clip_fraction=float(((ratio-1).abs() > config["clip"]).float().mean().detach()),
                         approximate_kl=float(((ratio-1)-log_ratio).mean().detach()),
                         gradient_norm=float(grad), samples=count))
    after = weight_hash(policy)
    return dict(before=before, after=after, actor_before=actor_before,
                actor_after=actor_weight_hash(policy), samples=count, epochs=logs,
                methodology="Terminal categorical PPO; no bootstrap; all attempted slots retained")


def _cpu(value):
    if torch is not None and isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _cpu(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu(item) for item in value)
    return copy.deepcopy(value)


def checkpoint_payload(policy, optimizer, *, study_fingerprint, geometry_fingerprint,
                       menu_sha256, updates, training_batches, seed):
    _torch()
    _sha(study_fingerprint, "study")
    _sha(geometry_fingerprint, "geometry")
    _sha(menu_sha256, "native menu")
    _counts(updates, training_batches)
    _require(type(seed) is int and 0 <= seed < 2**63, "Invalid training seed")
    return dict(policy_version=POLICY_VERSION, architecture=copy.deepcopy(ARCHITECTURE),
                study_fingerprint=study_fingerprint, geometry_fingerprint=geometry_fingerprint,
                menu_sha256=menu_sha256, updates=updates, training_batches=training_batches,
                seed=seed, weight_hash=weight_hash(policy), actor_weight_hash=actor_weight_hash(policy),
                state_dict=_cpu(policy.state_dict()),
                optimizer=_cpu(optimizer.state_dict()) if optimizer is not None else None)


def load_checkpoint(path, *, study_fingerprint, geometry_fingerprint, menu_sha256,
                    device="cpu", expected_updates=None, expected_training_batches=None):
    """Load a versioned policy only; assessments do not restore an optimizer."""
    _torch()
    saved = torch.load(Path(path), map_location="cpu", weights_only=True)
    _require(isinstance(saved, dict) and saved.get("policy_version") == POLICY_VERSION
             and saved.get("architecture") == ARCHITECTURE, "Checkpoint policy architecture mismatch")
    for key, expected in (("study_fingerprint", study_fingerprint),
                          ("geometry_fingerprint", geometry_fingerprint),
                          ("menu_sha256", menu_sha256)):
        _sha(expected, key)
        _require(saved.get(key) == expected, f"Checkpoint {key} mismatch")
    _counts(saved.get("updates"), saved.get("training_batches"))
    if expected_updates is not None:
        _require(saved["updates"] == expected_updates, "Checkpoint update count mismatch")
    if expected_training_batches is not None:
        _require(saved["training_batches"] == expected_training_batches,
                 "Checkpoint training-batch count mismatch")
    # Constructing the temporary model must not perturb the training RNG when
    # an assessment reloads a checkpoint between on-policy batches. All module
    # initialization is on CPU; moving existing tensors to CUDA draws no noise.
    with torch.random.fork_rng(devices=[]):
        policy = MenuPolicy()
    policy = policy.to(device)
    policy.load_state_dict(saved["state_dict"], strict=True)
    _require(weight_hash(policy) == saved.get("weight_hash"), "Checkpoint policy weight hash mismatch")
    _require(actor_weight_hash(policy) == saved.get("actor_weight_hash"),
             "Checkpoint actor weight hash mismatch")
    return policy, saved
