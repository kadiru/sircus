"""Continuous residual PPO and a matched, training-only guided sweep."""

import copy
import hashlib

import numpy as np
import torch

from .benchmark_contract import action_from_residual
from .learning_contract import reward
from .learning_policy import update, weight_hash


class ResidualPolicy(torch.nn.Module):
    def __init__(self):
        super().__init__()

        def net(outputs):
            return torch.nn.Sequential(
                torch.nn.Linear(9, 64),
                torch.nn.Tanh(),
                torch.nn.Linear(64, 64),
                torch.nn.Tanh(),
                torch.nn.Linear(64, outputs),
            )

        self.actor, self.critic = net(5), net(1)
        self.logstd = torch.nn.Parameter(torch.full((5,), -1.5))
        torch.nn.init.zeros_(self.actor[-1].weight)
        torch.nn.init.zeros_(self.actor[-1].bias)

    def distribution(self, observation):
        return torch.distributions.Normal(self.actor(observation), self.logstd.exp())


class GuidedSweep:
    """Search at TRAIN contexts, then transfer a frozen nearest-context residual."""

    def __init__(self, seed, train):
        self.seed, self.train = int(seed), copy.deepcopy(train)
        self.counts, self.best = {}, {}

    def propose(self, context, anchor):
        key = context["id"]
        index = self.counts.get(key, 0)
        self.counts[key] = index + 1
        code = f"{self.seed}:{key}:{index}"
        rng = np.random.default_rng(
            int(hashlib.sha256(code.encode()).hexdigest()[:16], 16)
        )
        dependencies = []
        if index == 0:
            residual, kind = np.zeros(5), "zero_shared_anchor"
        elif index <= 10:
            residual = np.zeros(5)
            residual[(index - 1) // 2] = -0.5 if index % 2 else 0.5
            kind = "axis_sweep"
        elif index <= 20:
            residual, kind = rng.uniform(-0.8, 0.8, 5), "seeded_joint_space_fill"
        else:
            source = self.best.get(key)
            center = np.zeros(5) if source is None else np.asarray(source["residual"])
            dependencies = [] if source is None else [source["id"]]
            residual = np.tanh(
                np.arctanh(np.clip(center, -0.999999, 0.999999))
                + rng.normal(0, 0.35, 5)
            )
            kind = "train_best_refinement"
        return residual, dict(
            strategy=kind, proposal_index=index, source_training_ids=dependencies
        )

    def observe(self, row):
        if row.get("role") != "train":
            raise ValueError("Sweep may learn only from TRAIN records")
        key = row["context_id"]
        rank = (
            int(
                bool(
                    row.get("native_release")
                    and row.get("route_success")
                    and not row.get("invalid")
                )
            ),
            reward(row),
        )
        old = self.best.get(key)
        if old is None or rank > tuple(old["rank"]):
            self.best[key] = dict(
                id=row["id"],
                context_id=key,
                residual=row["normalized_residual"],
                rank=list(rank),
                values=row["values"],
            )

    def predict(self, context, anchor):
        nearest = min(
            self.train,
            key=lambda r: (
                float(
                    np.linalg.norm(
                        np.asarray(r["bucket_xy_m"]) - context["bucket_xy_m"]
                    )
                ),
                r["id"],
            ),
        )
        source = self.best.get(nearest["id"])
        residual = np.zeros(5) if source is None else np.asarray(source["residual"])
        return action_from_residual(anchor["values"], residual), dict(
            nearest_training_context_id=nearest["id"],
            source_training_ids=[] if source is None else [source["id"]],
            normalized_residual=residual.tolist(),
            search_at_inference=False,
        )

    def state(self):
        return dict(
            version="sircus-guided-residual-sweep-v1",
            seed=self.seed,
            train=self.train,
            counts=self.counts,
            best=self.best,
        )

    @classmethod
    def from_state(cls, state):
        obj = cls(state["seed"], state["train"])
        obj.counts, obj.best = copy.deepcopy(state["counts"]), copy.deepcopy(
            state["best"]
        )
        return obj


__all__ = ["ResidualPolicy", "GuidedSweep", "update", "weight_hash"]
