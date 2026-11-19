"""Bounded continuation updates; no command line, I/O or native execution.

Constructing an engine is an explicit learning operation. Production callers
must first verify the parent and TRAIN export and persist their execution
authorization separately. Importing this module never creates an optimizer.
"""

from collections import Counter
import copy
from dataclasses import asdict
import json

from candidate_continuation_checkpoint import state_hash
from candidate_continuation_contract import (PROTOCOL_SHA256, SEED, DEMONSTRATIONS,
    DEMO_BATCH, IMITATION_EPOCHS, IMITATION_MAX_STEPS, PPO_BUFFER, PPO_EPOCHS,
    PPO_MAX_UPDATES, PPO_MAX_STEPS, protocol)
from candidate_continuation_policy import CandidateContinuationPolicy, CandidateDistribution, FrozenCandidateSupport
from candidate_continuation_losses import imitation_objective, freeze_terminal_rollout, categorical_ppo_loss
from continuous_policy import ContinuousPolicy, weight_hash, actor_weight_hash, torch

SCHEMA = "sircus-candidate-continuation-optimizer-update-v1"
COUNTER_KEYS = ("imitation_steps", "imitation_presentations", "ppo_updates", "ppo_steps", "ppo_samples")
OLD_LOGP_TOLERANCE = 1e-4
OLD_VALUE_TOLERANCE = 3e-6


def require(condition, message):
    if not condition:
        raise ValueError(message)


def _cpu(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key:_cpu(item) for key,item in value.items()}
    if isinstance(value, tuple):
        return tuple(_cpu(item) for item in value)
    if isinstance(value, list):
        return [_cpu(item) for item in value]
    return copy.deepcopy(value)


class LearningEngine:
    """Actor-only imitation then categorical PPO on immutable TRAIN supports.

    Native request/release accounting, persistence and wall limits belong to the
    runner. No held-out geometry/outcome is accepted here. A failed update halts
    this engine; completed optimizer steps are never rolled back or repeated.
    """

    def __init__(self, policy, observations, supports, demonstrations, successes_by_context):
        require(torch is not None, "Continuation updates require configured PyTorch")
        require(isinstance(policy, ContinuousPolicy), "Expected strictly loaded ContinuousPolicy")
        require(isinstance(observations, torch.Tensor) and observations.dtype == torch.float32
                and observations.shape == (35, 5) and torch.isfinite(observations).all().item(),
                "Expected 35 finite float32 TRAIN observations")
        require(type(supports) in (tuple, list) and len(supports) == 35
                and all(isinstance(s, FrozenCandidateSupport) and s.context_id.startswith("train-") for s in supports)
                and len({s.context_id for s in supports}) == 35, "Expected 35 distinct TRAIN supports")
        require(all(p.dtype == torch.float32 and p.device == observations.device for p in policy.parameters()),
                "Parent dtype/device changed")
        self.supports = tuple(supports)
        self.context_ids = tuple(s.context_id for s in supports)
        self._lookup = {context:i for i,context in enumerate(self.context_ids)}
        require(type(demonstrations) in (tuple, list) and len(demonstrations) == DEMONSTRATIONS,
                "Expected all 1534 successful TRAIN demonstrations")
        pairs, keys = set(), set()
        for demo in demonstrations:
            require(type(demo) is dict and set(demo) == {"context_id", "candidate_id", "index", "success_key", "clean"},
                    "Demonstration fields changed")
            context, index = demo["context_id"], demo["index"]
            require(context in self._lookup and type(index) is int and 0 <= index < len(supports[self._lookup[context]].candidate_ids),
                    "Demonstration must index its exact TRAIN support")
            require(supports[self._lookup[context]].candidate_ids[index] == demo["candidate_id"],
                    "Demonstration candidate/index changed")
            pair = (context, demo["candidate_id"])
            require(pair not in pairs and type(demo["success_key"]) is str and demo["success_key"]
                    and demo["success_key"] not in keys and type(demo["clean"]) is bool,
                    "Repeated or malformed successful demonstration")
            pairs.add(pair); keys.add(demo["success_key"])
        counts = Counter(d["context_id"] for d in demonstrations)
        require(type(successes_by_context) is dict and dict(counts) == successes_by_context and len(counts) == 21
                and all(type(v) is int for v in successes_by_context.values())
                and sum(d["clean"] for d in demonstrations) == 485,
                "Audited demonstration context/clean counts changed")
        self._demonstrations = tuple(copy.deepcopy(demonstrations))
        self._success_counts = dict(counts)
        self._observations = observations.detach().clone()
        self.policy = policy
        self.adapter = CandidateContinuationPolicy(policy)
        self._parent = copy.deepcopy(policy).eval()
        for parameter in self._parent.parameters():
            parameter.requires_grad_(False)
        self._parent_weight_hash = weight_hash(self._parent)
        self._parent_adapter = CandidateContinuationPolicy(self._parent)
        with torch.no_grad():
            self._anchor = self._parent_adapter.distribution(self._observations, self.supports)
        self.shuffle_generator = torch.Generator(device="cpu").manual_seed(SEED)
        self.sample_generator = torch.Generator(device="cpu").manual_seed(SEED)
        self._counters = dict.fromkeys(COUNTER_KEYS, 0)
        self.stage = "imitation"
        self._permutation = None
        self._final_buffer = False
        for name, parameter in policy.named_parameters():
            parameter.requires_grad_(name.startswith("actor."))
            parameter.grad = None
        self.imitation_optimizer = self._adam(policy.actor.parameters(), 1e-4)
        self.ppo_optimizer = None
        self.optimizer = self.imitation_optimizer
        self._expected_hash = weight_hash(policy)

    @staticmethod
    def _adam(parameters, learning_rate):
        settings = protocol()["adam"]
        return torch.optim.Adam(parameters, lr=learning_rate, betas=tuple(settings["betas"]),
            eps=settings["eps"], weight_decay=settings["weight_decay"], amsgrad=False, foreach=False, fused=False)

    def snapshot_counters(self):
        return dict(self._counters)

    @property
    def observations(self):
        return self._observations.detach().clone()

    @property
    def demonstrations(self):
        return copy.deepcopy(self._demonstrations)

    @property
    def successes_by_context(self):
        return dict(self._success_counts)

    def state_dict(self):
        return dict(stage=self.stage, counters=self.snapshot_counters(),
            rng_states={"shuffle":self.shuffle_generator.get_state().clone(),
                        "sample":self.sample_generator.get_state().clone()},
            optimizer_states={"imitation":_cpu(self.imitation_optimizer.state_dict()),
                              "ppo":None if self.ppo_optimizer is None else _cpu(self.ppo_optimizer.state_dict())},
            weight_hash=weight_hash(self.policy), parent_weight_hash=self._parent_weight_hash)

    def optimizer_state_hash(self):
        return state_hash(self.optimizer.state_dict())

    def _check(self, stage):
        require(self.stage == stage, "Learning engine stage is not " + stage)
        require(weight_hash(self.policy) == self._expected_hash, "Policy changed outside this learning engine")
        require(weight_hash(self._parent) == self._parent_weight_hash, "Frozen parent anchor changed")

    def _batch(self, context_ids):
        require(type(context_ids) in (tuple, list) and 0 < len(context_ids) <= PPO_BUFFER
                and all(type(c) is str and c in self._lookup for c in context_ids),
                "Expected only known TRAIN context IDs, at most 140")
        indices = [self._lookup[c] for c in context_ids]
        return self._observations[indices], tuple(self.supports[i] for i in indices)

    def distribution(self, context_ids):
        self._check("ppo")
        require(not self._final_buffer, "Final PPO buffer already consumed")
        observations, supports = self._batch(context_ids)
        with torch.no_grad():
            return self.adapter.distribution(observations, supports)

    def sample(self, context_ids):
        require(0 < len(context_ids) <= 35, "Sampling accepts one native batch of at most 35 TRAIN contexts")
        distribution = self.distribution(context_ids)
        # Keep one dedicated CPU stream even when the supplied model is on CUDA.
        indices = [int(torch.multinomial(row.detach().cpu().exp(), 1, generator=self.sample_generator).item())
                   for row in distribution.log_probabilities]
        return distribution, torch.tensor(indices, dtype=torch.int64, device=self._observations.device)

    def _hashes(self):
        return dict(weight_hash=weight_hash(self.policy), actor_hash=actor_weight_hash(self.policy),
            critic_hash=state_hash(self.policy.critic.state_dict()), logstd_hash=state_hash(self.policy.logstd),
            optimizer_hash=self.optimizer_state_hash(),
            rng_hash=state_hash({"shuffle":self.shuffle_generator.get_state(), "sample":self.sample_generator.get_state()}))

    def _receipt(self, before, counters_before, **extra):
        after = self._hashes()
        receipt = dict(schema=SCHEMA, protocol_sha256=PROTOCOL_SHA256, stage=self.stage,
            counters_before=counters_before, counters_after=self.snapshot_counters(), **extra)
        for name,value in before.items():
            receipt[name+"_before"] = value
            receipt[name+"_after"] = after[name]
        return json.loads(json.dumps(receipt, allow_nan=False))

    def _step(self, loss, *, clipped):
        require(torch.isfinite(loss).item(), "Nonfinite learning loss")
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        parameters = [p for p in self.policy.parameters() if p.requires_grad]
        require(all(p.grad is not None and torch.isfinite(p.grad).all().item() for p in parameters),
                "Nonfinite or missing parameter gradient")
        norm = torch.nn.utils.clip_grad_norm_(parameters, 1. if clipped else float("inf"), error_if_nonfinite=True)
        self.optimizer.step()
        if self.stage == "ppo":
            self._counters["ppo_steps"] += 1
            with torch.no_grad():
                self.policy.logstd.clamp_(-3., 1.)
        else:
            self._counters["imitation_steps"] += 1
        # Validate actual parameter and Adam moments; a failure is fatal, not retried.
        self._expected_hash = weight_hash(self.policy)
        state_hash(self.policy.state_dict()); state_hash(self.optimizer.state_dict())
        return float(norm)

    def imitation_step(self):
        self._check("imitation")
        require(self._counters["imitation_steps"] < IMITATION_MAX_STEPS, "Imitation schedule is complete")
        before, counters_before = self._hashes(), self.snapshot_counters()
        step = self._counters["imitation_steps"]
        epoch, minibatch = divmod(step, (DEMONSTRATIONS+DEMO_BATCH-1)//DEMO_BATCH)
        require(epoch < IMITATION_EPOCHS, "Imitation epoch cap reached")
        if minibatch == 0:
            self._permutation = torch.randperm(DEMONSTRATIONS, generator=self.shuffle_generator).tolist()
        indices = self._permutation[minibatch*DEMO_BATCH:(minibatch+1)*DEMO_BATCH]
        demonstrations = [self._demonstrations[i] for i in indices]
        observations, supports = self._batch([d["context_id"] for d in demonstrations])
        try:
            current = self.adapter.distribution(self._observations, self.supports)
            demos = self.adapter.distribution(observations, supports)
            objective = imitation_objective(demos, [d["index"] for d in demonstrations],
                self._success_counts, self._anchor, current)
            gradient_norm = self._step(objective.total, clipped=False)
            self._counters["imitation_presentations"] += len(indices)
            after = self._hashes()
            require(before["critic_hash"] == after["critic_hash"] and before["logstd_hash"] == after["logstd_hash"],
                    "Imitation changed frozen critic or log scales")
            return self._receipt(before, counters_before, index=step, epoch=epoch, minibatch=minibatch,
                demo_indices=indices, frozen_critic_and_logstd=True,
                metrics=dict(total=float(objective.total.detach()), demonstration_nll=float(objective.demonstration_nll.detach()),
                    forward_kl=float(objective.forward_kl.detach()), gradient_norm=gradient_norm))
        except Exception:
            self.stage = "halted"
            raise

    def start_ppo(self):
        self._check("imitation")
        require(self._counters["imitation_steps"] == IMITATION_MAX_STEPS
                and self._counters["imitation_presentations"] == IMITATION_EPOCHS*DEMONSTRATIONS,
                "PPO requires the exact completed imitation schedule")
        for parameter in self.policy.parameters():
            parameter.requires_grad_(True)
            parameter.grad = None
        self.ppo_optimizer = self._adam(self.policy.parameters(), 3e-4)
        self.optimizer = self.ppo_optimizer
        require(not self.optimizer.state, "PPO Adam state must be fresh")
        self.stage = "ppo"

    def ppo_update(self, context_ids, selected_indices, terminal_returns, old_distribution, *, final=False):
        self._check("ppo")
        require(type(final) is bool and not self._final_buffer, "Final PPO buffer already consumed or malformed")
        observations, supports = self._batch(context_ids)
        count = len(context_ids)
        require(count == PPO_BUFFER or final, "Only a final PPO buffer may be partial")
        require(self._counters["ppo_samples"]+count <= 1750
                and self._counters["ppo_updates"] < PPO_MAX_UPDATES
                and self._counters["ppo_steps"]+PPO_EPOCHS <= PPO_MAX_STEPS, "PPO request/update cap exceeded")
        require(isinstance(old_distribution, CandidateDistribution), "Expected categorical sampling distribution")
        current = self.distribution(context_ids)
        require(current.context_ids == old_distribution.context_ids
                and current.support_fingerprints == old_distribution.support_fingerprints
                and current.candidate_ids == old_distribution.candidate_ids, "Stale PPO support or context order")
        # Native sampling uses <=35 rows; this buffer may have 140. Different
        # matrix multiplication batch shapes can change last bits even with the
        # exact same weights. Keep the sampled values, never replace them here.
        require(all(torch.allclose(a, b, atol=OLD_LOGP_TOLERANCE, rtol=0)
                    for a,b in zip(current.log_probabilities, old_distribution.log_probabilities))
                and torch.allclose(current.values, old_distribution.values, atol=OLD_VALUE_TOLERANCE, rtol=0),
                "Stale PPO policy probabilities or old values")
        require(all(not x.requires_grad for x in old_distribution.log_probabilities)
                and not old_distribution.values.requires_grad, "PPO old distribution must be detached")
        try:
            rollout = freeze_terminal_rollout(old_distribution, selected_indices, terminal_returns)
        except Exception:
            self.stage = "halted"
            raise
        before, counters_before = self._hashes(), self.snapshot_counters()
        epochs = []
        try:
            for epoch in range(PPO_EPOCHS):
                objective = categorical_ppo_loss(self.adapter.distribution(observations, supports), rollout)
                gradient_norm = self._step(objective.total, clipped=True)
                epochs.append(dict(epoch=epoch, total=float(objective.total.detach()),
                    actor_loss=float(objective.actor_loss.detach()), critic_loss=float(objective.critic_loss.detach()),
                    entropy=float(objective.entropy.detach()), gradient_norm=gradient_norm,
                    ratios_min=float(objective.ratios.detach().min()), ratios_max=float(objective.ratios.detach().max())))
            index = self._counters["ppo_updates"]
            self._counters["ppo_updates"] += 1
            self._counters["ppo_samples"] += count
            self._final_buffer = final
            return self._receipt(before, counters_before, index=index, request_count=count, final=final,
                                 rollout=asdict(rollout), epochs=epochs)
        except Exception:
            self.stage = "halted"
            raise

    def finish_ppo(self):
        self._check("ppo")
        require(self._final_buffer and 1400 <= self._counters["ppo_samples"] <= 1750,
                "Final freeze requires a cap-stopped TRAIN history and final buffer")
        self.optimizer.zero_grad(set_to_none=True)
        for parameter in self.policy.parameters():
            parameter.requires_grad_(False)
        self.stage = "final"
