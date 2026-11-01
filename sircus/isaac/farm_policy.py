"""Shared Gaussian actor/critic and clipped PPO for terminal one-shot episodes."""

import hashlib

import numpy as np
import torch
from farm_contract import HIGH, LOW, WARM


class Policy(torch.nn.Module):
    def __init__(self):
        super().__init__()

        def mlp(out):
            return torch.nn.Sequential(
                torch.nn.Linear(5, 64),
                torch.nn.Tanh(),
                torch.nn.Linear(64, 64),
                torch.nn.Tanh(),
                torch.nn.Linear(64, out),
            )

        self.actor, self.critic = mlp(3), mlp(1)
        self.logstd = torch.nn.Parameter(torch.full((3,), -1.2))
        torch.nn.init.zeros_(self.actor[-1].weight)
        with torch.no_grad():
            self.actor[-1].bias.copy_(
                torch.tensor(
                    np.arctanh(2 * (WARM - LOW) / (HIGH - LOW) - 1), dtype=torch.float32
                )
            )

    def distribution(self, x):
        return torch.distributions.Normal(self.actor(x), self.logstd.exp())


def weight_hash(policy):
    h = hashlib.sha256()
    for k, v in sorted(policy.state_dict().items()):
        h.update(k.encode())
        h.update(v.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def update(policy, optimizer, observations, latents, old_logp, returns, spec):
    """The tanh Jacobian cancels in PPO's ratio for a stored latent action.

    Every throw terminates; returns are observed rewards (no next-value/GAE).
    The entropy term is explicitly latent-Gaussian entropy, not action entropy.
    """
    with torch.no_grad():
        adv = returns - policy.critic(observations).squeeze(-1)
        adv = (adv - adv.mean()) / (adv.std(unbiased=False) + 1e-6)
    old_logp = old_logp.detach()
    before = weight_hash(policy)
    logs = []
    for _ in range(spec["epochs"]):
        dist = policy.distribution(observations)
        logp = dist.log_prob(latents).sum(-1)
        ratio = (logp - old_logp).exp()
        actor = -torch.minimum(
            ratio * adv, ratio.clamp(1 - spec["clip"], 1 + spec["clip"]) * adv
        ).mean()
        value = ((policy.critic(observations).squeeze(-1) - returns) ** 2).mean()
        entropy = dist.entropy().sum(-1).mean()
        loss = actor + spec["value_coef"] * value - spec["entropy_coef"] * entropy
        if not torch.isfinite(loss):
            raise RuntimeError("Nonfinite PPO loss")
        optimizer.zero_grad()
        loss.backward()
        grad = torch.nn.utils.clip_grad_norm_(policy.parameters(), spec["grad_norm"])
        if not torch.isfinite(grad):
            raise RuntimeError("Nonfinite PPO gradient")
        optimizer.step()
        with torch.no_grad():
            policy.logstd.clamp_(-2.8, -0.2)
        logs.append(
            dict(
                loss=float(loss.detach()),
                actor_loss=float(actor.detach()),
                critic_loss=float(value.detach()),
                latent_entropy=float(entropy.detach()),
                clip_fraction=float(
                    ((ratio - 1).abs() > spec["clip"]).float().mean().detach()
                ),
                gradient_norm=float(grad),
            )
        )
    return dict(before=before, after=weight_hash(policy), epochs=logs)
