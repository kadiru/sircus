"""Differentiable finite choices on immutable candidate encodings, without updates.

The supplied ``ContinuousPolicy`` must already have passed its original strict
checkpoint loader. This wrapper neither loads weights nor constructs an optimizer.
Support qualification and checkpoint/bank lineage are caller responsibilities;
freezing a support does not authorize native execution or a physical release.
"""

from dataclasses import dataclass
import hashlib
import json
import math

import numpy as np

from continuous_policy import ACTION_DIM, ACTION_NAMES, ContinuousPolicy, torch

SUPPORT_SCHEMA = "sircus-fixed-candidate-support-v1"
DISTRIBUTION_SCHEMA = "sircus-candidate-continuation-distribution-v1"
PROBABILITY_LAW = "categorical-normalized-fixed-latent-normal-density-v1"
TIE_TOLERANCE = 1e-12
# Original physical affine domain. Keep these immutable and local so importing
# this adapter does not import qualification/native/simulator dependencies.
LOW = (-math.pi, .28, .14, .08, math.radians(12))
HIGH = (math.pi, 1.20, .60, .56, math.radians(52))


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _torch():
    if torch is None:
        raise ImportError("Candidate continuation requires the configured PyTorch environment")
    return torch


def _json(value):
    return json.dumps(value, sort_keys=True, allow_nan=False)


def _hash(value):
    return hashlib.sha256(_json(value).encode()).hexdigest()


def _values(action):
    _require(isinstance(action, dict), "Expected an exact physical action mapping")
    _require(all(type(action.get(name)) in (int, float) for name in ACTION_NAMES),
             "Expected five numeric physical action parameters")
    vector = np.asarray([action[name] for name in ACTION_NAMES], dtype=np.float64)
    _require(np.isfinite(vector).all() and (vector >= LOW).all() and (vector <= HIGH).all(),
             "Physical action outside original bounds")
    return vector


class NoQualifiedCandidates(ValueError):
    """Explicit empty-support decline; there is no sampled placeholder action."""

    def __init__(self, context_id):
        self.context_id = context_id
        self.reason = "no_qualified_candidate_support"
        super().__init__(self.reason + ": " + context_id)


@dataclass(frozen=True)
class FrozenCandidateSupport:
    """Immutable ordered identity, exact JSON actions and original latent vectors."""

    context_id: str
    candidate_ids: tuple
    latents: tuple
    parent_index: int | None
    _actions_json: tuple
    support_fingerprint: str
    schema: str = SUPPORT_SCHEMA
    probability_law: str = PROBABILITY_LAW

    def __post_init__(self):
        _require(isinstance(self.context_id, str) and self.context_id, "Missing support context identity")
        if not self.candidate_ids:
            raise NoQualifiedCandidates(self.context_id)
        count = len(self.candidate_ids)
        _require(type(self.candidate_ids) is type(self.latents) is type(self._actions_json) is tuple,
                 "Support storage must be immutable tuples")
        _require(len(self.latents) == len(self._actions_json) == count
                 and len(set(self.candidate_ids)) == count
                 and all(isinstance(name, str) and name for name in self.candidate_ids),
                 "Invalid ordered candidate identities")
        _require(self.parent_index is None or (type(self.parent_index) is int
                 and 0 <= self.parent_index < count), "Invalid exact-parent index")
        _require(self.schema == SUPPORT_SCHEMA and self.probability_law == PROBABILITY_LAW,
                 "Unexpected fixed support schema or probability law")
        _require(all(type(row) is tuple and len(row) == ACTION_DIM
                     and all(type(value) in (int, float) for value in row) for row in self.latents)
                 and np.isfinite(np.asarray(self.latents, dtype=np.float64)).all(),
                 "Expected immutable finite five-dimensional candidate latents")
        _require(all(isinstance(row, str) for row in self._actions_json), "Invalid exact action storage")
        for action in self.actions:
            _values(action)
        _require(self.support_fingerprint == _hash(self._identity()), "Fixed support fingerprint mismatch")

    def _identity(self):
        return dict(schema=self.schema, probability_law=self.probability_law,
                    context_id=self.context_id, candidate_ids=self.candidate_ids,
                    actions=self.actions, latents=self.latents, parent_index=self.parent_index)

    @property
    def actions(self):
        """Return independent exact-action copies; callers cannot mutate the support."""
        return tuple(json.loads(row) for row in self._actions_json)

    def action(self, index):
        _require(type(index) is int and 0 <= index < len(self.candidate_ids), "Candidate index out of bounds")
        return json.loads(self._actions_json[index])

    def to_dict(self):
        """Independent JSON-ready identity for an externally hash-bound export."""
        return json.loads(_json(dict(self._identity(), support_fingerprint=self.support_fingerprint)))

    @classmethod
    def from_dict(cls, value):
        _require(isinstance(value, dict) and set(value) == {
            "schema", "probability_law", "context_id", "candidate_ids", "actions", "latents",
            "parent_index", "support_fingerprint"}, "Fixed support export fields changed")
        _require(isinstance(value["candidate_ids"], (list, tuple))
                 and isinstance(value["actions"], (list, tuple))
                 and isinstance(value["latents"], (list, tuple))
                 and all(isinstance(row, (list, tuple)) for row in value["latents"]),
                 "Malformed fixed support export")
        return cls(value["context_id"], tuple(value["candidate_ids"]),
                   tuple(tuple(row) for row in value["latents"]), value["parent_index"],
                   tuple(_json(action) for action in value["actions"]), value["support_fingerprint"],
                   value["schema"], value["probability_law"])


def freeze_support(context_id, candidates, parent_action, parent_latent):
    """Freeze caller-qualified choices in their supplied order, without filtering.

    The exact parent's *original saved* latent is copied once, independently of
    any future actor mean. Other actions use the original affine inverse/atanh.
    A saturated nonparent boundary fails closed rather than being clipped. Empty
    supports raise ``NoQualifiedCandidates`` with an explicit decline reason.
    """
    _require(isinstance(context_id, str) and context_id, "Missing support context identity")
    _require(isinstance(candidates, (list, tuple)), "Expected an ordered candidate sequence")
    if not candidates:
        raise NoQualifiedCandidates(context_id)
    parent_values = _values(parent_action)
    original_latent = np.asarray(parent_latent, dtype=np.float64)
    _require(original_latent.shape == (ACTION_DIM,) and np.isfinite(original_latent).all(),
             "Invalid original frozen-parent latent")
    ids, latents, actions, parent_indices, physical = [], [], [], [], set()
    for index, row in enumerate(candidates):
        _require(isinstance(row, dict) and isinstance(row.get("candidate_id"), str)
                 and row["candidate_id"], "Missing exact candidate identity")
        _require("cell_id" not in row or row["cell_id"] == context_id,
                 "Candidate context identity changed")
        vector = _values(row.get("action"))
        identity = tuple(vector.tolist())
        _require(identity not in physical, "Duplicate exact physical candidate")
        physical.add(identity)
        if np.array_equal(vector, parent_values):
            latent = original_latent.copy()
            parent_indices.append(index)
        else:
            normalized = 2*(vector-np.asarray(LOW))/(np.asarray(HIGH)-np.asarray(LOW))-1
            _require((normalized > -1).all() and (normalized < 1).all(),
                     "Qualified action is on a saturated latent boundary; explicit new protocol required")
            latent = np.arctanh(normalized)
        ids.append(row["candidate_id"])
        latents.append(tuple(latent.tolist()))
        # JSON round-tripping keeps all exact supplied action fields, while
        # removing aliases to mutable bank dictionaries.
        actions.append(_json(row["action"]))
    _require(len(set(ids)) == len(ids), "Duplicate candidate identity")
    parent_index = parent_indices[0] if parent_indices else None
    identity = dict(schema=SUPPORT_SCHEMA, probability_law=PROBABILITY_LAW,
                    context_id=context_id, candidate_ids=tuple(ids),
                    actions=tuple(json.loads(row) for row in actions),
                    latents=tuple(latents), parent_index=parent_index)
    return FrozenCandidateSupport(context_id, tuple(ids), tuple(latents), parent_index,
                                  tuple(actions), _hash(identity))


@dataclass(frozen=True)
class CandidateDistribution:
    """Ragged categorical batch with explicit immutable support identities.

    All probabilities and log probabilities are finite-support quantities;
    they are never the old squashed-Gaussian density or its entropy estimate.
    ``values`` retains the supplied critic's weight dtype.
    """

    context_ids: tuple
    support_fingerprints: tuple
    candidate_ids: tuple
    scores: tuple
    log_probabilities: tuple
    values: object
    parent_indices: tuple = ()
    schema: str = DISTRIBUTION_SCHEMA
    probability_law: str = PROBABILITY_LAW

    def __post_init__(self):
        _torch()
        count = len(self.context_ids)
        _require(count > 0 and all(type(rows) is tuple for rows in
                 (self.context_ids, self.support_fingerprints, self.candidate_ids,
                  self.scores, self.log_probabilities, self.parent_indices)),
                 "Expected nonempty immutable categorical batch metadata")
        _require(all(len(rows) == count for rows in
                 (self.support_fingerprints, self.candidate_ids, self.scores, self.log_probabilities))
                 and (not self.parent_indices or len(self.parent_indices) == count),
                 "Categorical batch coverage changed")
        _require(all(isinstance(identity, str) and identity for identity in
                     self.context_ids+self.support_fingerprints), "Missing categorical context/support identity")
        _require(isinstance(self.values, torch.Tensor) and self.values.is_floating_point()
                 and self.values.shape == (count,)
                 and torch.isfinite(self.values).all().item(), "Invalid categorical critic values")
        for index, (names, score, logp) in enumerate(zip(self.candidate_ids, self.scores, self.log_probabilities)):
            _require(type(names) is tuple and names and len(names) == len(set(names))
                     and all(isinstance(name, str) and name for name in names), "Invalid categorical candidate IDs")
            _require(all(isinstance(row, torch.Tensor) and row.dtype == torch.float64
                         and row.shape == (len(names),) and torch.isfinite(row).all().item()
                         and row.device == self.values.device for row in (score, logp)),
                     "Expected finite float64 categorical scores/log probabilities")
            _require(torch.allclose(logp.logsumexp(0), logp.new_zeros(()), atol=1e-12, rtol=0),
                     "Categorical log probabilities are not normalized")
            _require(torch.allclose(logp, torch.log_softmax(score, dim=0), atol=1e-12, rtol=0),
                     "Categorical log probabilities disagree with finite density scores")
            if self.parent_indices:
                parent = self.parent_indices[index]
                _require(parent is None or (type(parent) is int and 0 <= parent < len(names)),
                         "Invalid categorical exact-parent index")

    @property
    def probabilities(self):
        return tuple(row.exp() for row in self.log_probabilities)

    def entropy(self):
        """Exact categorical entropy for every context, including singleton zero."""
        return torch.stack([-(row.exp()*row).sum() for row in self.log_probabilities])

    def _indices(self, indices):
        if isinstance(indices, torch.Tensor):
            _require(indices.ndim == 1 and indices.dtype in
                     (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8),
                     "Expected one integer candidate index per context")
            indices = indices.detach().cpu().tolist()
        _require(isinstance(indices, (list, tuple)) and len(indices) == len(self.context_ids)
                 and all(type(index) is int and 0 <= index < len(names)
                         for index, names in zip(indices, self.candidate_ids)),
                 "Candidate index out of bounds or batch coverage changed")
        return indices

    def log_prob(self, indices):
        return torch.stack([row[index] for row, index in zip(self.log_probabilities, self._indices(indices))])

    def sample(self, generator=None):
        """Draw exactly once per context from this same finite categorical law."""
        return torch.stack([torch.multinomial(row.exp(), 1, generator=generator).squeeze(0)
                            for row in self.log_probabilities])

    def deterministic(self):
        """Argmax with frozen parent then lexical-ID priority within 1e-12."""
        selected = []
        for batch_index, (names, score) in enumerate(zip(self.candidate_ids, self.scores)):
            numbers = score.detach().cpu().tolist()
            threshold = max(numbers)-TIE_TOLERANCE
            tied = [i for i, number in enumerate(numbers) if number >= threshold]
            parent = self.parent_indices[batch_index] if self.parent_indices else None
            selected.append(parent if parent in tied else min(tied, key=lambda i: names[i]))
        return torch.tensor(selected, dtype=torch.int64, device=self.values.device)


_BaseModule = torch.nn.Module if torch is not None else object


class CandidateContinuationPolicy(_BaseModule):
    """Reuse externally strict-loaded parent weights for a new finite action law."""

    def __init__(self, parent_policy):
        _torch()
        super().__init__()
        _require(isinstance(parent_policy, ContinuousPolicy), "Expected an externally strict-loaded ContinuousPolicy")
        self.parent_policy = parent_policy

    def distribution(self, observations, supports):
        _require(isinstance(supports, (list, tuple)) and supports
                 and all(isinstance(support, FrozenCandidateSupport) for support in supports),
                 "Expected nonempty fixed candidate supports")
        _require(isinstance(observations, torch.Tensor) and observations.ndim == 2
                 and len(observations) == len(supports), "Observation/support batch coverage changed")
        # This call is essential: actor(observations) alone drops the inherited
        # state-dependent radius prior. Cast outputs, never the parent weights.
        normal = self.parent_policy.distribution(observations)
        scores, log_probabilities = [], []
        for index, support in enumerate(supports):
            latent = torch.tensor(support.latents, dtype=torch.float64, device=normal.loc.device)
            gaussian = torch.distributions.Normal(normal.loc[index].double(), normal.scale[index].double())
            score = gaussian.log_prob(latent).sum(dim=-1)
            scores.append(score)
            log_probabilities.append(torch.log_softmax(score, dim=0))
        return CandidateDistribution(
            context_ids=tuple(s.context_id for s in supports),
            support_fingerprints=tuple(s.support_fingerprint for s in supports),
            candidate_ids=tuple(s.candidate_ids for s in supports), scores=tuple(scores),
            log_probabilities=tuple(log_probabilities), values=self.parent_policy.value(observations),
            parent_indices=tuple(s.parent_index for s in supports))
