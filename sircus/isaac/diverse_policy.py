"""Hybrid categorical-region / continuous-residual terminal PPO.

The joint action includes an explicit region identity, even when physical boxes
overlap. PPO uses its categorical likelihood plus the conditional squashed
Gaussian density. Entropy is in this joint region/normalized-residual space,
not physical shot-volume entropy. Deterministic inference is argmax region
followed by its residual mean, not the expectation of the physical mixture.
"""

import copy
import math
from pathlib import Path

import numpy as np

import continuous_policy as base
from continuous_feasibility import HIGH, LOW
from diverse_support import validate_support, SCHEMA_V1

torch = base.torch
POLICY_VERSION_V1 = "sircus-diverse-hybrid-shot-ppo-v1"
ARCHITECTURE_V1 = dict(
    version=POLICY_VERSION_V1, observation_dim=5, action_dim=5, latent_dim=6,
    latent_layout=["categorical_region_index", "bearing_latent", "radius_latent", "height_latent", "gap_latent", "tilt_latent"],
    action_names=list(base.ACTION_NAMES), hidden=[64, 64], activation="tanh",
    actor="Shared geometry-conditioned categorical region scorer and conditional diagonal Gaussian residual actor",
    region_features="5 privileged bucket features + 5 normalized box-center + 5 normalized box-width features",
    critic="state-only terminal return prediction", initial_region_distribution="uniform over available geometry regions",
    initial_logstd=list(base.INITIAL_LOGSTD), initial_actor_residual_normalized=[0.]*5,
    logstd_bounds=[-3., 1.], physical_decoder="explicit geometry region index plus affine five-dimensional residual box",
    entropy_coordinates="joint categorical region and normalized residual; not physical shot-volume entropy",
    deterministic="argmax region logits (geometry-prior region ordering breaks ties), then conditional Gaussian mean",
)
POLICY_VERSION = "sircus-diverse-hybrid-shot-ppo-v2"
ARCHITECTURE = dict(copy.deepcopy(ARCHITECTURE_V1), version=POLICY_VERSION,
    initial_region_distribution="uniform over retained local geometry regions and one unqualified full-domain exploration escape per supported context",
    physical_decoder="explicit local/global geometry region index plus affine five-dimensional residual box",
    exploration_scope="All local anchors retained plus original global action domain; global requests are not prequalified and may fail unchanged native gates")
build_observations = base.build_observations
weight_hash = base.weight_hash


def actor_weight_hash(policy):
    return base._hash_items([(f"region_actor.{k}", v) for k, v in policy.region_actor.state_dict().items()]
        + [(f"residual_actor.{k}", v) for k, v in policy.residual_actor.state_dict().items()]
        + [("logstd", policy.logstd)])


def _geometry(support):
    validate_support(support)
    cells = [dict(id=r["cell_id"], bucket_xy_m=r["bucket_xy_m"]) for r in support["cells"]]
    observations = build_observations(cells)
    count = max(1, support["max_regions_per_cell"])
    mask = np.zeros((35, count), dtype=bool)
    features = np.zeros((35, count, 10), dtype=np.float32)
    for i, row in enumerate(support["cells"]):
        if not row["regions"]:
            mask[i, 0] = True  # dummy sampled action remains rejected native data
        for j, region in enumerate(row["regions"]):
            mask[i, j] = True
            low, high = np.asarray(region["low"]), np.asarray(region["high"])
            center = (low+high)/2
            features[i, j, :5] = (2*(center-LOW)/(HIGH-LOW)-1).astype(np.float32)
            features[i, j, 5:] = ((high-low)/(HIGH-LOW)).astype(np.float32)
    return observations, mask, features


_Module = torch.nn.Module if torch is not None else object


class DiversePolicy(_Module):
    def __init__(self, support, cells=None):
        base._torch()
        super().__init__()
        validate_support(support, cells)
        self.policy_version = POLICY_VERSION_V1 if support["schema"] == SCHEMA_V1 else POLICY_VERSION
        self.architecture = copy.deepcopy(ARCHITECTURE_V1 if support["schema"] == SCHEMA_V1 else ARCHITECTURE)
        observations, mask, features = _geometry(support)
        self.support_fingerprint = support["fingerprint"]
        self.register_buffer("context_observations", torch.tensor(observations))
        self.register_buffer("region_mask", torch.tensor(mask))
        self.register_buffer("region_features", torch.tensor(features))
        def mlp(inputs, outputs):
            return torch.nn.Sequential(torch.nn.Linear(inputs, 64), torch.nn.Tanh(),
                torch.nn.Linear(64, 64), torch.nn.Tanh(), torch.nn.Linear(64, outputs))
        self.region_actor = mlp(15, 1)
        self.residual_actor = mlp(15, 5)
        self.critic = mlp(5, 1)
        self.logstd = torch.nn.Parameter(torch.tensor(base.INITIAL_LOGSTD, dtype=torch.float32))
        for module in (self.region_actor, self.residual_actor):
            torch.nn.init.zeros_(module[-1].weight)
            torch.nn.init.zeros_(module[-1].bias)

    @staticmethod
    def _observations(observations):
        base.ContinuousPolicy._observations(observations)

    def _features(self, observations):
        self._observations(observations)
        # Exact configured float32 feature identity, not nearest-context lookup.
        expected = self.context_observations.to(dtype=observations.dtype)
        matches = (observations[:, None, :] == expected[None, :, :]).all(-1)
        base._require((matches.sum(-1) == 1).all().item(), "Unknown or ambiguous diverse policy bucket context")
        indices = matches.to(torch.int64).argmax(-1)
        features = self.region_features[indices].to(dtype=observations.dtype)
        inputs = torch.cat((observations[:, None, :].expand(-1, features.shape[1], -1), features), dim=-1)
        return inputs, self.region_mask[indices]

    def distributions(self, observations):
        inputs, mask = self._features(observations)
        logits = self.region_actor(inputs).squeeze(-1).masked_fill(~mask, -torch.inf)
        regions = torch.distributions.Categorical(logits=logits)
        residuals = torch.distributions.Normal(self.residual_actor(inputs), self.logstd.clamp(-3, 1).exp())
        return regions, residuals, mask

    def _components(self, observations, latent):
        regions, residuals, mask = self.distributions(observations)
        base._require(latent.shape == (len(observations), 6) and torch.isfinite(latent).all().item(),
                      "Expected region index plus five Gaussian latents")
        raw_regions, residual = latent[:, 0], latent[:, 1:]
        indices = raw_regions.to(torch.int64)
        base._require(((raw_regions == indices) & (indices >= 0) & (indices < mask.shape[1])).all().item(),
                      "Invalid categorical region indices")
        batch = torch.arange(len(observations), device=observations.device)
        base._require(mask[batch, indices].all().item(), "Unavailable categorical region")
        distribution = torch.distributions.Normal(residuals.loc[batch, indices], residuals.scale[batch, indices])
        jacobian = 2*(math.log(2)-residual-torch.nn.functional.softplus(-2*residual))
        region_logp = regions.log_prob(indices)
        residual_logp = (distribution.log_prob(residual)-jacobian).sum(-1)
        return region_logp, residual_logp

    def log_prob(self, observations, latent):
        region, residual = self._components(observations, latent)
        return region+residual

    def value(self, observations):
        self._observations(observations)
        return self.critic(observations).squeeze(-1)

    def entropy_estimate(self, observations, generator=None):
        regions, residuals, _ = self.distributions(observations)
        noise = torch.randn(residuals.loc.shape, dtype=residuals.loc.dtype,
                            device=residuals.loc.device, generator=generator)
        latent = residuals.loc+residuals.scale*noise
        jacobian = 2*(math.log(2)-latent-torch.nn.functional.softplus(-2*latent))
        conditional = -(residuals.log_prob(latent)-jacobian).sum(-1)
        # Exact categorical expectation of one reparameterized entropy sample
        # per region, so gradients include probability changes of each region.
        return regions.entropy()+(regions.probs*conditional).sum(-1)


def sample(policy, observations, generator=None, deterministic=False):
    base._require(type(deterministic) is bool, "Expected Boolean deterministic flag")
    with torch.no_grad():
        regions, residuals, _ = policy.distributions(observations)
        indices = regions.logits.argmax(-1) if deterministic else torch.multinomial(regions.probs, 1, generator=generator).squeeze(-1)
        batch = torch.arange(len(observations), device=observations.device)
        mean, scale = residuals.loc[batch, indices], residuals.scale[batch, indices]
        noise = torch.zeros_like(mean) if deterministic else torch.randn(mean.shape, dtype=mean.dtype,
            device=mean.device, generator=generator)
        residual_latent = mean+scale*noise
        latent = torch.cat((indices[:, None].to(mean.dtype), residual_latent), dim=-1)
        region_logp, residual_logp = policy._components(observations, latent)
        logp = region_logp+residual_logp
        return dict(latent=latent, actions_normalized=residual_latent.tanh(), region_indices=indices,
                    logp=logp, region_logp=region_logp, residual_logp=residual_logp,
                    entropy_estimate=-logp, value=policy.value(observations), deterministic=deterministic)


def update(policy, optimizer, observations, latent, old_logp, returns, spec):
    config = {name: spec[name] for name in base.PPO_DEFAULTS}
    base._require(type(config["epochs"]) is int and 1 <= config["epochs"] <= 16
        and all(np.isfinite(v) for v in config.values()) and 0 < config["clip"] < 1
        and config["value_coef"] >= 0 and config["entropy_coef"] >= 0 and config["grad_norm"] > 0,
        "Invalid hybrid PPO hyperparameters")
    policy._observations(observations)
    count = len(observations)
    base._require(count > 1 and latent.shape == (count, 6) and old_logp.shape == returns.shape == (count,)
        and torch.isfinite(latent).all().item() and torch.isfinite(old_logp).all().item()
        and torch.isfinite(returns).all().item(), "Invalid hybrid terminal PPO buffer")
    with torch.no_grad():
        advantages = returns-policy.value(observations)
        advantages = (advantages-advantages.mean())/(advantages.std(unbiased=False)+1e-6)
    latent, old_logp, returns = latent.detach(), old_logp.detach(), returns.detach()
    before, actor_before, logs = weight_hash(policy), actor_weight_hash(policy), []
    for _ in range(config["epochs"]):
        logp = policy.log_prob(observations, latent)
        log_ratio = logp-old_logp
        ratio = log_ratio.exp()
        actor = -torch.minimum(ratio*advantages,
            ratio.clamp(1-config["clip"], 1+config["clip"])*advantages).mean()
        value = ((policy.value(observations)-returns)**2).mean()
        entropy = policy.entropy_estimate(observations).mean()
        loss = actor+config["value_coef"]*value-config["entropy_coef"]*entropy
        if not torch.isfinite(loss):
            raise RuntimeError("Nonfinite hybrid PPO loss")
        optimizer.zero_grad()
        loss.backward()
        grad = torch.nn.utils.clip_grad_norm_(policy.parameters(), config["grad_norm"])
        if not torch.isfinite(grad):
            raise RuntimeError("Nonfinite hybrid PPO gradient")
        optimizer.step()
        logs.append(dict(loss=float(loss.detach()), actor_loss=float(actor.detach()), critic_loss=float(value.detach()),
            joint_entropy_estimate=float(entropy.detach()), clip_fraction=float(((ratio-1).abs() > config["clip"]).float().mean().detach()),
            approximate_kl=float(((ratio-1)-log_ratio).mean().detach()), gradient_norm=float(grad), samples=count))
    return dict(before=before, after=weight_hash(policy), actor_before=actor_before,
        actor_after=actor_weight_hash(policy), samples=count, epochs=logs,
        methodology="Terminal hybrid region/residual PPO; joint likelihood, all attempted shots retained")


def checkpoint_payload(policy, optimizer, *, study_fingerprint, geometry_fingerprint,
                       domain_fingerprint, support_fingerprint, update_count, attempted_shots, seed):
    base._require(isinstance(policy, DiversePolicy) and policy.support_fingerprint == support_fingerprint,
                  "Expected matching diverse hybrid policy")
    base._counts(update_count, attempted_shots)
    for label, value in (("study", study_fingerprint), ("geometry", geometry_fingerprint),
                         ("domain", domain_fingerprint), ("support", support_fingerprint)):
        base._sha(value, label)
    base._require(type(seed) is int and 0 <= seed < 2**63, "Invalid hybrid training seed")
    return dict(policy_version=policy.policy_version, architecture=copy.deepcopy(policy.architecture),
        study_fingerprint=study_fingerprint, geometry_fingerprint=geometry_fingerprint,
        domain_fingerprint=domain_fingerprint, support_fingerprint=support_fingerprint,
        update_count=update_count, attempted_shots=attempted_shots, seed=seed,
        weight_hash=weight_hash(policy), actor_weight_hash=actor_weight_hash(policy),
        state_dict=base._cpu(policy.state_dict()),
        optimizer=base._cpu(optimizer.state_dict()) if optimizer is not None else None)


def load_checkpoint(path, *, support, study_fingerprint, geometry_fingerprint,
                    domain_fingerprint, support_fingerprint, device="cpu",
                    expected_updates=None, expected_attempted_shots=None):
    validate_support(support)
    base._require(support["fingerprint"] == support_fingerprint
        and support["source_fingerprint"] == geometry_fingerprint, "Diverse checkpoint support/geometry mismatch")
    saved = base._torch().load(Path(path), map_location="cpu", weights_only=True)
    expected_version = POLICY_VERSION_V1 if support["schema"] == SCHEMA_V1 else POLICY_VERSION
    expected_architecture = ARCHITECTURE_V1 if support["schema"] == SCHEMA_V1 else ARCHITECTURE
    base._require(isinstance(saved, dict) and saved.get("policy_version") == expected_version
        and saved.get("architecture") == expected_architecture, "Hybrid checkpoint architecture mismatch")
    for key, expected in (("study_fingerprint", study_fingerprint), ("geometry_fingerprint", geometry_fingerprint),
                          ("domain_fingerprint", domain_fingerprint), ("support_fingerprint", support_fingerprint)):
        base._sha(expected, key)
        base._require(saved.get(key) == expected, "Hybrid checkpoint "+key+" mismatch")
    base._counts(saved.get("update_count"), saved.get("attempted_shots"))
    if expected_updates is not None:
        base._require(saved["update_count"] == expected_updates, "Hybrid update count mismatch")
    if expected_attempted_shots is not None:
        base._require(saved["attempted_shots"] == expected_attempted_shots, "Hybrid attempted-shot count mismatch")
    with torch.random.fork_rng(devices=[]):
        policy = DiversePolicy(support)
    for key in ("context_observations", "region_mask", "region_features"):
        base._require(torch.equal(policy.state_dict()[key], saved["state_dict"][key]),
                      "Hybrid checkpoint geometry buffer mismatch")
    policy = policy.to(device)
    policy.load_state_dict(saved["state_dict"], strict=True)
    base._require(weight_hash(policy) == saved.get("weight_hash"), "Hybrid weight hash mismatch")
    base._require(actor_weight_hash(policy) == saved.get("actor_weight_hash"), "Hybrid actor hash mismatch")
    return policy, saved
