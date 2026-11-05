"""Continuous one-shot PPO over five bounded physical shot parameters.

This is a terminal contextual policy, not a motor-level trajectory controller.
Every proposed shot is retained, including native admission failures. The policy
never sees a successful-shot menu and never searches or projects at inference.
"""

import copy
import hashlib
import math
import re
from pathlib import Path

import numpy as np

try:
    import torch
except ModuleNotFoundError:
    torch = None

POLICY_VERSION = "sircus-continuous-shot-ppo-v1"
OBSERVATION_DIM = 5
ACTION_NAMES = ("bearing_offset_rad", "radius_m", "height_m", "gap_m", "tilt_rad")
ACTION_DIM = len(ACTION_NAMES)
# Prespecified geometry-only starting residual: offset 0, board .35 m,
# gap .24 m and tilt 29 deg. Radius has zero learned residual around the
# state-conditioned bucket-to-REFERENCE distance; no successful-shot labels.
INITIAL_NORMALIZED = [0.0, 0.0, 2*(.35-.14)/(.60-.14)-1,
                      2*(.24-.08)/(.56-.08)-1, 2*(29-12)/(52-12)-1]
ARCHITECTURE = dict(
    version=POLICY_VERSION,
    observation_dim=OBSERVATION_DIM,
    action_names=list(ACTION_NAMES),
    action_dim=ACTION_DIM,
    hidden=[64, 64],
    activation="tanh",
    actor="state-conditioned independent Gaussian; tanh-squashed continuous actions",
    critic="state-only terminal return prediction",
    initial_logstd=-0.9,
    initial_actor_residual_normalized=INITIAL_NORMALIZED,
    initial_radius_prior=dict(observation_feature=2, physical_domain_m=[.28, 1.20],
                              prior_clip_m=[.29, 1.19],
                              meaning="Geometry-only bucket-to-reference distance; no search/outcome labels"),
    logstd_bounds=[-3.0, 1.0],
)
PPO_DEFAULTS = dict(
    epochs=4, clip=0.2, value_coef=0.5, entropy_coef=0.005, grad_norm=1.0
)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _torch():
    if torch is None:
        raise ImportError("Continuous shot PPO requires the configured PyTorch environment")
    return torch


def _sha(value, label):
    _require(isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value),
             f"Invalid {label} SHA256")
    return value


def _counts(update_count, attempted_shots):
    _require(type(update_count) is int and type(attempted_shots) is int
             and 0 <= update_count <= attempted_shots,
             "Invalid continuous policy update/attempt counters")


def build_observations(cells):
    """Deterministic privileged geometry only, with no IDs/outcomes in features."""
    from cylinder_farm import observation

    _require(len(cells) > 0, "Expected nonempty bucket cells")
    values = np.asarray([observation(cell) for cell in cells], dtype=np.float32)
    _require(values.shape == (len(cells), OBSERVATION_DIM) and np.isfinite(values).all(),
             "Invalid privileged bucket observations")
    return values


def physical_actions(normalized, low, high):
    """Affine decode only. No projection, clipping or rejection resampling."""
    if torch is not None and isinstance(normalized, torch.Tensor):
        normalized = normalized.detach().cpu().numpy()
    values = np.asarray(normalized, dtype=float)
    lower, upper = np.asarray(low, dtype=float), np.asarray(high, dtype=float)
    _require(values.ndim == 2 and values.shape[1] == ACTION_DIM
             and np.isfinite(values).all() and (np.abs(values) <= 1).all(),
             "Normalized actions must be finite N by 5 values in [-1,1]")
    _require(lower.shape == upper.shape == (ACTION_DIM,)
             and np.isfinite(lower).all() and np.isfinite(upper).all()
             and (upper > lower).all(), "Invalid physical action domain")
    return lower + (values + 1) * (upper - lower) / 2


_BaseModule = torch.nn.Module if torch is not None else object


class ContinuousPolicy(_BaseModule):
    """State-only actor/critic with auditable latent-to-action likelihoods."""

    def __init__(self):
        _torch()
        super().__init__()

        def mlp(outputs):
            return torch.nn.Sequential(
                torch.nn.Linear(OBSERVATION_DIM, 64), torch.nn.Tanh(),
                torch.nn.Linear(64, 64), torch.nn.Tanh(),
                torch.nn.Linear(64, outputs),
            )

        self.actor = mlp(ACTION_DIM)
        self.critic = mlp(1)
        self.logstd = torch.nn.Parameter(torch.full((ACTION_DIM,), -0.9))
        # Every bucket starts at the declared geometric prior plus zero learned radius residual.
        # There is no fitted checkpoint, per-cell warm start or success catalogue.
        torch.nn.init.zeros_(self.actor[-1].weight)
        with torch.no_grad():
            self.actor[-1].bias.copy_(torch.tensor(INITIAL_NORMALIZED).atanh())

    @staticmethod
    def _observations(observations):
        _require(observations.ndim == 2 and observations.shape[1] == OBSERVATION_DIM
                 and torch.isfinite(observations).all().item(), "Invalid policy observations")

    def distribution(self, observations):
        self._observations(observations)
        mean = self.actor(observations)
        # observation[2] is metric bucket-to-overlap-reference distance. The
        # declared prior centers board XY at that reference when radius bounds
        # permit it; the actor can learn arbitrary residuals across full bounds.
        radius_m = observations[:, 2].clamp(.29, 1.19)
        normalized = 2*(radius_m-.28)/(1.20-.28)-1
        prior = torch.zeros_like(mean)
        prior[:, 1] = torch.atanh(normalized)
        return torch.distributions.Normal(mean + prior, self.logstd.clamp(-3, 1).exp())

    def value(self, observations):
        self._observations(observations)
        return self.critic(observations).squeeze(-1)

    def log_prob(self, observations, latent):
        """Density in normalized action coordinates, computed from saved latent.

        The stable tanh Jacobian stays finite when float32 tanh rounds to +/-1.
        A fixed affine map to physical coordinates adds a constant log-density
        offset which cancels in PPO ratios, so it is deliberately not applied.
        """
        distribution = self.distribution(observations)
        _require(latent.shape == distribution.loc.shape and torch.isfinite(latent).all().item(),
                 "Invalid Gaussian action latent")
        jacobian = 2 * (math.log(2) - latent - torch.nn.functional.softplus(-2 * latent))
        return (distribution.log_prob(latent) - jacobian).sum(dim=-1)

    def entropy_estimate(self, observations, generator=None):
        """Reparameterized Monte Carlo entropy of the squashed distribution."""
        distribution = self.distribution(observations)
        noise = torch.randn(distribution.loc.shape, dtype=distribution.loc.dtype,
                            device=distribution.loc.device, generator=generator)
        latent = distribution.loc + distribution.scale * noise
        return -self.log_prob(observations, latent)


def sample(policy, observations, generator=None, deterministic=False):
    """One direct inference call; independent generators isolate assessment RNG."""
    _torch()
    _require(type(deterministic) is bool, "Expected Boolean deterministic flag")
    with torch.no_grad():
        distribution = policy.distribution(observations)
        if deterministic:
            latent = distribution.loc.clone()
        else:
            noise = torch.randn(distribution.loc.shape, dtype=distribution.loc.dtype,
                                device=distribution.loc.device, generator=generator)
            latent = distribution.loc + distribution.scale * noise
        logp = policy.log_prob(observations, latent)
        return dict(latent=latent, actions_normalized=latent.tanh(), logp=logp,
                    entropy_estimate=-logp, value=policy.value(observations),
                    deterministic=deterministic)


def _hash_items(items):
    result = hashlib.sha256()
    for name, value in sorted(items):
        result.update(name.encode())
        result.update(str(value.dtype).encode())
        result.update(str(tuple(value.shape)).encode())
        result.update(value.detach().cpu().contiguous().numpy().tobytes())
    return result.hexdigest()


def actor_weight_hash(policy):
    _torch()
    return _hash_items([(f"actor.{k}", v) for k, v in policy.actor.state_dict().items()]
                       + [("logstd", policy.logstd)])


def weight_hash(policy):
    _torch()
    return _hash_items(policy.state_dict().items())


def update(policy, optimizer, observations, latent, old_logp, returns, spec):
    """Clipped terminal PPO. No bootstrap/GAE, menu or failed-shot filtering."""
    _torch()
    config = {name: spec[name] for name in PPO_DEFAULTS}
    _require(type(config["epochs"]) is int and 1 <= config["epochs"] <= 16,
             "Unbounded PPO epoch count")
    _require(all(np.isfinite(v) for v in config.values())
             and 0 < config["clip"] < 1 and config["value_coef"] >= 0
             and config["entropy_coef"] >= 0 and config["grad_norm"] > 0,
             "Invalid PPO hyperparameters")
    policy._observations(observations)
    count = len(observations)
    _require(count > 1 and latent.shape == (count, ACTION_DIM)
             and old_logp.shape == returns.shape == (count,)
             and torch.isfinite(latent).all().item()
             and torch.isfinite(old_logp).all().item()
             and torch.isfinite(returns).all().item(), "Invalid terminal PPO buffer")
    with torch.no_grad():
        advantages = returns - policy.value(observations)
        advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-6)
    latent, old_logp, returns = latent.detach(), old_logp.detach(), returns.detach()
    before, actor_before, logs = weight_hash(policy), actor_weight_hash(policy), []
    for _ in range(config["epochs"]):
        logp = policy.log_prob(observations, latent)
        log_ratio = logp - old_logp
        ratio = log_ratio.exp()
        actor = -torch.minimum(ratio * advantages,
                               ratio.clamp(1-config["clip"], 1+config["clip"]) * advantages).mean()
        value = ((policy.value(observations) - returns) ** 2).mean()
        entropy = policy.entropy_estimate(observations).mean()
        loss = actor + config["value_coef"] * value - config["entropy_coef"] * entropy
        if not torch.isfinite(loss):
            raise RuntimeError("Nonfinite continuous PPO loss")
        optimizer.zero_grad()
        loss.backward()
        grad = torch.nn.utils.clip_grad_norm_(policy.parameters(), config["grad_norm"])
        if not torch.isfinite(grad):
            raise RuntimeError("Nonfinite continuous PPO gradient")
        optimizer.step()
        logs.append(dict(loss=float(loss.detach()), actor_loss=float(actor.detach()),
                         critic_loss=float(value.detach()), squashed_entropy_estimate=float(entropy.detach()),
                         clip_fraction=float(((ratio-1).abs() > config["clip"]).float().mean().detach()),
                         approximate_kl=float(((ratio-1)-log_ratio).mean().detach()),
                         gradient_norm=float(grad), samples=count))
    return dict(before=before, after=weight_hash(policy), actor_before=actor_before,
                actor_after=actor_weight_hash(policy), samples=count, epochs=logs,
                methodology="Terminal continuous tanh-Gaussian PPO; all attempted shots retained")


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
                       domain_fingerprint, update_count, attempted_shots, seed):
    _torch()
    _counts(update_count, attempted_shots)
    for name, value in (("study", study_fingerprint), ("geometry", geometry_fingerprint),
                        ("action domain", domain_fingerprint)):
        _sha(value, name)
    _require(type(seed) is int and 0 <= seed < 2**63, "Invalid training seed")
    return dict(policy_version=POLICY_VERSION, architecture=copy.deepcopy(ARCHITECTURE),
                study_fingerprint=study_fingerprint, geometry_fingerprint=geometry_fingerprint,
                domain_fingerprint=domain_fingerprint, update_count=update_count,
                attempted_shots=attempted_shots, seed=seed,
                weight_hash=weight_hash(policy), actor_weight_hash=actor_weight_hash(policy),
                state_dict=_cpu(policy.state_dict()),
                optimizer=_cpu(optimizer.state_dict()) if optimizer is not None else None)


def load_checkpoint(path, *, study_fingerprint, geometry_fingerprint, domain_fingerprint,
                    device="cpu", expected_updates=None, expected_attempted_shots=None):
    """Load frozen policy without touching training RNG or restoring optimizer."""
    _torch()
    saved = torch.load(Path(path), map_location="cpu", weights_only=True)
    _require(isinstance(saved, dict) and saved.get("policy_version") == POLICY_VERSION
             and saved.get("architecture") == ARCHITECTURE, "Checkpoint policy architecture mismatch")
    for key, expected in (("study_fingerprint", study_fingerprint),
                          ("geometry_fingerprint", geometry_fingerprint),
                          ("domain_fingerprint", domain_fingerprint)):
        _sha(expected, key)
        _require(saved.get(key) == expected, f"Checkpoint {key} mismatch")
    _counts(saved.get("update_count"), saved.get("attempted_shots"))
    if expected_updates is not None:
        _require(saved["update_count"] == expected_updates, "Checkpoint update count mismatch")
    if expected_attempted_shots is not None:
        _require(saved["attempted_shots"] == expected_attempted_shots,
                 "Checkpoint attempted-shot count mismatch")
    with torch.random.fork_rng(devices=[]):
        policy = ContinuousPolicy()
    policy = policy.to(device)
    policy.load_state_dict(saved["state_dict"], strict=True)
    _require(weight_hash(policy) == saved.get("weight_hash"), "Checkpoint policy weight hash mismatch")
    _require(actor_weight_hash(policy) == saved.get("actor_weight_hash"),
             "Checkpoint actor weight hash mismatch")
    return policy, saved
