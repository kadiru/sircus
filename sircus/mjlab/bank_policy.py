"""Masked categorical shot selection trained from native contextual-bandit returns.

The fixed TRAIN-derived proposal bank is not a policy. This network learns which
geometry-feasible proposal to select for a query; it never looks up a query's
recorded outcome. All-invalid queries stay unresolved and do not enter PPO.
"""

import hashlib

import torch

VERSION = "sircus-mjlab-relative-bank-ppo-v1"
FEATURE_NAMES = (
    "query_x",
    "query_y",
    "relative_board_x",
    "relative_board_y",
    "absolute_board_x",
    "absolute_board_y",
    "board_height",
    "gap",
    "tilt",
)


def features(bucket, candidates):
    if (
        bucket.ndim != 2
        or bucket.shape[1] != 2
        or candidates.ndim != 3
        or candidates.shape[0] != len(bucket)
        or candidates.shape[2] != 5
    ):
        raise ValueError("Expected bucket[N,2] and candidates[N,K,5]")
    xy = bucket[:, None, :].expand(-1, candidates.shape[1], -1)
    query = (xy - bucket.new_tensor([0.67, -0.20])) / bucket.new_tensor([0.6, 0.5])
    relative = (candidates[:, :, :2] - xy) / 0.8
    absolute = (candidates[:, :, :2] - bucket.new_tensor([0.67, -0.04])) / 0.4
    hgt = (candidates[:, :, 2:3] - 0.35) / 0.2
    gap = (candidates[:, :, 4:5] - 0.24) / 0.3
    tilt = (candidates[:, :, 3:4] - 30) / 25
    return torch.cat((query, relative, absolute, hgt, gap, tilt), dim=-1)


class Policy(torch.nn.Module):
    def __init__(self):
        super().__init__()

        def net(inputs, outputs):
            return torch.nn.Sequential(
                torch.nn.Linear(inputs, 64),
                torch.nn.Tanh(),
                torch.nn.Linear(64, 64),
                torch.nn.Tanh(),
                torch.nn.Linear(64, outputs),
            )

        self.actor, self.critic = net(9, 1), net(2, 1)
        torch.nn.init.zeros_(self.actor[-1].weight)
        torch.nn.init.zeros_(self.actor[-1].bias)

    def distribution(self, bucket, candidates, mask):
        if mask.shape != candidates.shape[:2] or not candidates.shape[1]:
            raise ValueError("Candidate mask must have shape [N,K] with K>0")
        mask = mask.bool()
        eligible = mask.any(dim=1)
        safe = mask.clone()
        # Categorical cannot normalize all -inf. A dummy distribution exists
        # solely so callers can retain query order; eligible=False forbids use.
        safe[~eligible, 0] = True
        logits = self.actor(features(bucket, candidates)).squeeze(-1)
        return (
            torch.distributions.Categorical(logits=logits.masked_fill(~safe, -1e9)),
            eligible,
        )

    def value(self, bucket):
        obs = (bucket - bucket.new_tensor([0.67, -0.20])) / bucket.new_tensor(
            [0.6, 0.5]
        )
        return self.critic(obs).squeeze(-1)


def weight_hash(policy):
    value = hashlib.sha256()
    for key, tensor in sorted(policy.state_dict().items()):
        value.update(key.encode())
        value.update(tensor.detach().cpu().numpy().tobytes())
    return value.hexdigest()


def predict(policy, bucket, candidates, mask, *, sample=False, generator=None):
    """One frozen selection; no simulation search or policy update occurs here.

    ``sample=True`` supports the predeclared uniform-valid initial baseline.
    Deployment uses deterministic argmax. Candidate metadata stays in the caller.
    """
    with torch.no_grad():
        distribution, eligible = policy.distribution(bucket, candidates, mask)
        indices = (
            torch.multinomial(distribution.probs, 1, generator=generator).squeeze(-1)
            if sample
            else distribution.probs.argmax(dim=-1)
        )
        values = candidates[torch.arange(len(bucket), device=bucket.device), indices]
        return dict(
            indices=indices,
            eligible=eligible,
            values=values,
            logp=distribution.log_prob(indices),
            probability=distribution.probs.gather(1, indices[:, None]).squeeze(-1),
        )


def update(
    policy,
    optimizer,
    bucket,
    candidates,
    mask,
    selected,
    old_logp,
    returns,
    *,
    epochs=6
):
    """Terminal PPO: actual observed rewards, no bootstrapping or GAE."""
    eligible = mask.bool().any(dim=1)
    before = weight_hash(policy)
    if not bool(eligible.any()):
        return dict(
            before=before,
            after=before,
            optimizer_rows=0,
            excluded_all_invalid=len(bucket),
            epochs=[],
        )
    bucket, candidates, mask = bucket[eligible], candidates[eligible], mask[eligible]
    selected, old_logp, returns = (
        selected[eligible].detach(),
        old_logp[eligible].detach(),
        returns[eligible].detach(),
    )
    if not bool(mask.gather(1, selected[:, None]).all()):
        raise RuntimeError("PPO cannot update from a masked-out candidate")
    with torch.no_grad():
        advantage = returns - policy.value(bucket)
        advantage = (advantage - advantage.mean()) / (
            advantage.std(unbiased=False) + 1e-6
        )
    logs = []
    for _ in range(epochs):
        distribution, _ = policy.distribution(bucket, candidates, mask)
        logp = distribution.log_prob(selected)
        ratio = (logp - old_logp).exp()
        actor = -torch.minimum(
            ratio * advantage, ratio.clamp(0.8, 1.2) * advantage
        ).mean()
        value = ((policy.value(bucket) - returns) ** 2).mean()
        entropy = distribution.entropy().mean()
        loss = actor + 0.5 * value - 0.001 * entropy
        if not bool(torch.isfinite(loss)):
            raise RuntimeError("Nonfinite categorical PPO loss")
        optimizer.zero_grad()
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        if not bool(torch.isfinite(norm)):
            raise RuntimeError("Nonfinite categorical PPO gradient")
        optimizer.step()
        with torch.no_grad():
            approximate_kl = ((ratio - 1) - (logp - old_logp)).mean()
        logs.append(
            dict(
                loss=float(loss.detach()),
                actor_loss=float(actor.detach()),
                critic_loss=float(value.detach()),
                entropy=float(entropy.detach()),
                gradient_norm=float(norm),
                approximate_kl=float(approximate_kl),
                clip_fraction=float(((ratio - 1).abs() > 0.2).float().mean()),
            )
        )
    return dict(
        before=before,
        after=weight_hash(policy),
        optimizer_rows=len(bucket),
        excluded_all_invalid=int((~eligible).sum()),
        epochs=logs,
    )
