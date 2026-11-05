"""Separate PPO checkpoint contract for geometry-assisted continuous residuals.

The fixed per-context affine support transform has a constant Jacobian per
context, cancelling exactly in PPO likelihood ratios. Logged entropy is entropy
in normalized residual coordinates, not physical-action volume entropy.
"""

import copy
from pathlib import Path

import continuous_policy as base

POLICY_VERSION = "sircus-feasibility-guided-shot-ppo-v1"
ARCHITECTURE = dict(
    version=POLICY_VERSION, observation_dim=5, action_dim=5,
    action_names=list(base.ACTION_NAMES), hidden=[64, 64], activation="tanh",
    actor="Independent Gaussian; tanh-squashed normalized local continuous residuals",
    critic="state-only terminal return prediction",
    initial_logstd=list(base.INITIAL_LOGSTD), initial_actor_residual_normalized=[0.]*5,
    logstd_bounds=[-3., 1.],
    physical_decoder="Geometry-only fixed per-context affine local support box",
    entropy_coordinates="normalized residual; not physical action volume",
)


class GuidedPolicy(base.ContinuousPolicy):
    def __init__(self):
        super().__init__()
        with base.torch.no_grad():
            self.actor[-1].bias.zero_()

    def distribution(self, observations):
        self._observations(observations)
        return base.torch.distributions.Normal(self.actor(observations), self.logstd.clamp(-3, 1).exp())


FeasibilityGuidedPolicy = GuidedPolicy
sample = base.sample
update = base.update
weight_hash = base.weight_hash
actor_weight_hash = base.actor_weight_hash
build_observations = base.build_observations


def checkpoint_payload(policy, optimizer, *, study_fingerprint, geometry_fingerprint,
                       domain_fingerprint, support_fingerprint, update_count, attempted_shots, seed):
    base._require(isinstance(policy, GuidedPolicy), "Expected guided continuous policy")
    payload = base.checkpoint_payload(policy, optimizer, study_fingerprint=study_fingerprint,
        geometry_fingerprint=geometry_fingerprint, domain_fingerprint=domain_fingerprint,
        update_count=update_count, attempted_shots=attempted_shots, seed=seed)
    base._sha(support_fingerprint, "guided support")
    payload.update(policy_version=POLICY_VERSION, architecture=copy.deepcopy(ARCHITECTURE),
                   support_fingerprint=support_fingerprint)
    return payload


def load_checkpoint(path, *, study_fingerprint, geometry_fingerprint, domain_fingerprint,
                    support_fingerprint, device="cpu", expected_updates=None, expected_attempted_shots=None):
    torch = base._torch()
    saved = torch.load(Path(path), map_location="cpu", weights_only=True)
    base._require(isinstance(saved, dict) and saved.get("policy_version") == POLICY_VERSION
                  and saved.get("architecture") == ARCHITECTURE, "Guided checkpoint architecture mismatch")
    for key, expected in (("study_fingerprint", study_fingerprint), ("geometry_fingerprint", geometry_fingerprint),
                          ("domain_fingerprint", domain_fingerprint), ("support_fingerprint", support_fingerprint)):
        base._sha(expected, key)
        base._require(saved.get(key) == expected, "Guided checkpoint " + key + " mismatch")
    base._counts(saved.get("update_count"), saved.get("attempted_shots"))
    if expected_updates is not None:
        base._require(saved["update_count"] == expected_updates, "Guided update count mismatch")
    if expected_attempted_shots is not None:
        base._require(saved["attempted_shots"] == expected_attempted_shots, "Guided attempted-shot count mismatch")
    with torch.random.fork_rng(devices=[]):
        policy = GuidedPolicy()
    policy = policy.to(device)
    policy.load_state_dict(saved["state_dict"], strict=True)
    base._require(weight_hash(policy) == saved.get("weight_hash"), "Guided weight hash mismatch")
    base._require(actor_weight_hash(policy) == saved.get("actor_weight_hash"), "Guided actor hash mismatch")
    return policy, saved
