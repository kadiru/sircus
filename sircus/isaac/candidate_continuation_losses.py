"""Pure fixed-bank continuation objectives; no optimizer or training entry point.

These functions calculate losses and may build a synthetic backward graph. They
never change model weights, execute candidates, or accept continuous PPO buffers.
"""

from dataclasses import dataclass
import math
import re

import candidate_continuation_policy as categorical

torch = categorical.torch

DEMONSTRATIONS = 1534
SUCCESS_CONTEXTS = 21
TRAIN_CONTEXTS = 35
KL_COEFFICIENT = 0.05
PPO_CLIP = 0.2
PPO_VALUE_COEFFICIENT = 0.5
PPO_ENTROPY_COEFFICIENT = 0.005
ROLLOUT_SCHEMA = "sircus-candidate-continuation-terminal-rollout-v1"


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _metadata(distribution):
    """Check the categorical contract before any loss arithmetic."""
    _require(isinstance(distribution, categorical.CandidateDistribution),
             "Expected fixed-support categorical distribution, not a legacy continuous buffer")
    _require(distribution.schema == categorical.DISTRIBUTION_SCHEMA
             and distribution.probability_law == categorical.PROBABILITY_LAW,
             "Categorical probability law/schema mismatch")
    count = len(distribution.context_ids)
    _require(count > 0 and isinstance(distribution.context_ids, tuple)
             and all(isinstance(value, str) and value for value in distribution.context_ids),
             "Missing categorical context IDs")
    _require(isinstance(distribution.support_fingerprints, tuple)
             and len(distribution.support_fingerprints) == count
             and all(isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value)
                     for value in distribution.support_fingerprints),
             "Missing categorical support fingerprints")
    _require(isinstance(distribution.candidate_ids, tuple)
             and len(distribution.candidate_ids) == count,
             "Missing categorical candidate support IDs")
    for ids in distribution.candidate_ids:
        _require(isinstance(ids, tuple) and len(ids) > 0
                 and all(isinstance(value, str) and value for value in ids)
                 and len(set(ids)) == len(ids), "Invalid categorical candidate support IDs")
    return count


def _same_support(first, second):
    _metadata(first)
    _metadata(second)
    _require(first.context_ids == second.context_ids
             and first.support_fingerprints == second.support_fingerprints
             and first.candidate_ids == second.candidate_ids,
             "Categorical context, support fingerprints or ordered candidate IDs changed")


def _indices(distribution, selected_indices):
    count = _metadata(distribution)
    if torch is not None and isinstance(selected_indices, torch.Tensor):
        _require(selected_indices.ndim == 1 and selected_indices.shape[0] == count
                 and selected_indices.dtype in (torch.int8, torch.int16, torch.int32, torch.int64),
                 "Expected one integer categorical selection per context")
        indices = tuple(selected_indices.detach().cpu().tolist())
    else:
        _require(isinstance(selected_indices, (tuple, list)) and len(selected_indices) == count,
                 "Expected one integer categorical selection per context")
        indices = tuple(selected_indices)
    _require(all(type(index) is int and 0 <= index < len(ids)
                 for index, ids in zip(indices, distribution.candidate_ids)),
             "Categorical selection outside immutable candidate support")
    return indices


def _success_counts(successes_by_context):
    _require(isinstance(successes_by_context, dict)
             and len(successes_by_context) == SUCCESS_CONTEXTS
             and all(isinstance(key, str) and key and type(value) is int and value > 0
                     for key, value in successes_by_context.items())
             and sum(successes_by_context.values()) == DEMONSTRATIONS,
             "Expected audited counts for 1534 demonstrations across exactly 21 contexts")


def context_balanced_demo_nll(distribution, selected_indices, successes_by_context):
    """Weighted exact-choice NLL, averaged over the actual minibatch length.

    Counts describe all audited successful choices, not just this minibatch.
    Global weights are never renormalized within a minibatch.
    """
    _metadata(distribution)
    _success_counts(successes_by_context)
    indices = _indices(distribution, selected_indices)
    _require(all(context in successes_by_context for context in distribution.context_ids),
             "Demonstration context has no audited success target")
    selected = distribution.log_prob(indices)
    weights = selected.new_tensor([DEMONSTRATIONS / (SUCCESS_CONTEXTS * successes_by_context[context])
                                   for context in distribution.context_ids])
    loss = -(weights * selected).mean()
    _require(torch.isfinite(loss).item(), "Nonfinite demonstration NLL")
    return loss


def exact_forward_kl(parent_distribution, current_distribution):
    """Exact detached-parent KL, equally averaged over all 35 TRAIN contexts."""
    _same_support(parent_distribution, current_distribution)
    _require(len(parent_distribution.context_ids) == TRAIN_CONTEXTS
             and len(set(parent_distribution.context_ids)) == TRAIN_CONTEXTS,
             "KL anchor requires all 35 distinct TRAIN contexts exactly once")
    terms = []
    for parent, current in zip(parent_distribution.log_probabilities,
                               current_distribution.log_probabilities):
        frozen_parent = parent.detach().to(device=current.device)
        terms.append((frozen_parent.exp() * (frozen_parent - current)).sum())
    loss = torch.stack(terms).mean()
    _require(torch.isfinite(loss).item(), "Nonfinite categorical forward KL")
    return loss


@dataclass(frozen=True)
class ImitationLoss:
    total: object
    demonstration_nll: object
    forward_kl: object


def imitation_objective(demonstration_distribution, selected_indices, successes_by_context,
                        parent_context_distribution, current_context_distribution):
    """One minibatch NLL plus the once-per-step full-context 0.05 KL anchor."""
    _same_support(parent_context_distribution, current_context_distribution)
    _metadata(demonstration_distribution)
    _success_counts(successes_by_context)
    full = {context: (fingerprint, ids) for context, fingerprint, ids in zip(
        current_context_distribution.context_ids, current_context_distribution.support_fingerprints,
        current_context_distribution.candidate_ids)}
    _require(set(successes_by_context).issubset(full),
             "Audited successes include a context outside the TRAIN anchor")
    for context, fingerprint, ids in zip(demonstration_distribution.context_ids,
                                         demonstration_distribution.support_fingerprints,
                                         demonstration_distribution.candidate_ids):
        _require(full.get(context) == (fingerprint, ids),
                 "Demonstration categorical support differs from TRAIN anchor")
    anchor = exact_forward_kl(parent_context_distribution, current_context_distribution)
    nll = context_balanced_demo_nll(demonstration_distribution, selected_indices, successes_by_context)
    total = nll + KL_COEFFICIENT * anchor
    _require(torch.isfinite(total).item(), "Nonfinite imitation objective")
    return ImitationLoss(total, nll, anchor)


@dataclass(frozen=True)
class TerminalCategoricalRollout:
    """Immutable detached terminal targets and categorical sampling identities.

    Numeric snapshots are tuples, so later in-place tensor changes cannot alter
    old probabilities, values, rewards or the advantages reused across epochs.
    This is an objective input, not a native execution ledger.
    """
    context_ids: tuple
    support_fingerprints: tuple
    candidate_ids: tuple
    selected_indices: tuple
    old_selected_log_probabilities: tuple
    old_values: tuple
    terminal_returns: tuple
    advantages: tuple
    schema: str = ROLLOUT_SCHEMA
    probability_law: str = categorical.PROBABILITY_LAW


def freeze_terminal_rollout(distribution, selected_indices, terminal_returns):
    """Detach R-V_old once; normalize population std for N>=2, keep singleton raw."""
    count = _metadata(distribution)
    indices = _indices(distribution, selected_indices)
    if torch is None:
        raise ImportError("Categorical continuation losses require the configured PyTorch environment")
    returns = torch.as_tensor(terminal_returns, dtype=torch.float64,
                              device=distribution.values.device).detach().clone()
    _require(returns.shape == (count,) and torch.isfinite(returns).all().item(),
             "Expected one finite terminal return for every requested categorical action")
    old_values = distribution.values.detach().to(dtype=torch.float64).clone()
    old_selected = distribution.log_prob(indices).detach().clone()
    advantages = returns - old_values
    if count >= 2:
        advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-6)
    _require(torch.isfinite(advantages).all().item(), "Nonfinite terminal categorical advantages")
    snapshots = [tuple(value.detach().cpu().tolist())
                 for value in (old_selected, old_values, returns, advantages)]
    return TerminalCategoricalRollout(distribution.context_ids, distribution.support_fingerprints,
                                      distribution.candidate_ids, indices, *snapshots)


def _validate_rollout(distribution, rollout):
    count = _metadata(distribution)
    _require(isinstance(rollout, TerminalCategoricalRollout)
             and rollout.schema == ROLLOUT_SCHEMA
             and rollout.probability_law == categorical.PROBABILITY_LAW,
             "Expected new categorical rollout; legacy continuous probability law/buffer rejected")
    _require(distribution.context_ids == rollout.context_ids
             and distribution.support_fingerprints == rollout.support_fingerprints
             and distribution.candidate_ids == rollout.candidate_ids,
             "Rollout categorical context, support fingerprints or ordered candidate IDs changed")
    _indices(distribution, rollout.selected_indices)
    for values in (rollout.old_selected_log_probabilities, rollout.old_values,
                   rollout.terminal_returns, rollout.advantages):
        _require(isinstance(values, tuple) and len(values) == count
                 and all(type(value) in (int, float) and math.isfinite(value) for value in values),
                 "Invalid detached categorical rollout targets")
    _require(all(value <= 1e-12 for value in rollout.old_selected_log_probabilities),
             "Old selected categorical log probabilities must not be continuous densities")
    old_values = torch.tensor(rollout.old_values, dtype=torch.float64)
    expected = torch.tensor(rollout.terminal_returns, dtype=torch.float64) - old_values
    if count >= 2:
        expected = (expected - expected.mean()) / (expected.std(unbiased=False) + 1e-6)
    _require(torch.allclose(torch.tensor(rollout.advantages, dtype=torch.float64), expected,
                            rtol=1e-10, atol=1e-12),
             "Categorical advantages differ from detached terminal return minus rollout old value")


@dataclass(frozen=True)
class CategoricalPPOLoss:
    total: object
    actor_loss: object
    critic_loss: object
    entropy: object
    ratios: object
    advantages: object
    old_selected_log_probabilities: object
    new_selected_log_probabilities: object


def categorical_ppo_loss(current_distribution, rollout):
    """Inherited clipped terminal PPO objective on the same fixed categorical law.

    Every rollout row contributes, including rejected actions. This function has
    no optimizer, clipping of gradients, update loop, or parameter mutation.
    """
    _validate_rollout(current_distribution, rollout)
    selected = current_distribution.log_prob(rollout.selected_indices)
    old_selected = selected.new_tensor(rollout.old_selected_log_probabilities)
    advantages = selected.new_tensor(rollout.advantages)
    returns = current_distribution.values.new_tensor(rollout.terminal_returns)
    ratios = (selected - old_selected).exp()
    actor = -torch.minimum(ratios * advantages,
                           ratios.clamp(1 - PPO_CLIP, 1 + PPO_CLIP) * advantages).mean()
    critic = (current_distribution.values - returns).square().mean()
    entropy = current_distribution.entropy().mean()
    total = actor + PPO_VALUE_COEFFICIENT * critic - PPO_ENTROPY_COEFFICIENT * entropy
    _require(all(torch.isfinite(value).all().item()
                 for value in (ratios, actor, critic, entropy, total)),
             "Nonfinite categorical PPO objective")
    return CategoricalPPOLoss(total, actor, critic, entropy, ratios, advantages,
                              old_selected, selected)
