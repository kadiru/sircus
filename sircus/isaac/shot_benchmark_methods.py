"""CPU proposal/prediction methods; no simulator, GPU, IK or outcome oracle.

The sweep spends one attempt per proposed action. Its frozen predictor transfers
a training residual, while PPO is the existing generic state-only actor. Neither
performs ball-flight search for a new bucket coordinate.
"""

import copy
import json
import math
from pathlib import Path

import numpy as np
from scipy.special import ndtri
from scipy.stats import qmc

from shot_benchmark_contract import (SEEDS, TRAIN_BUDGET, LOW, HIGH, REFERENCE,
    action_domain, canonical_hash, make_protocol, require, signed, validate_fingerprint)

VERSION = "sircus-prior-sobol-refinement-v1"
STATE_VERSION = "sircus-guided-shot-library-v1"
INITIAL_NORMALIZED = np.array([0., 0., 2 * (.35 - .14) / (.60 - .14) - 1,
    2 * (.24 - .08) / (.56 - .08) - 1, 2 * (29 - 12) / (52 - 12) - 1])
INITIAL_LOGSTD = np.array([-2.2, -2., -1.5, -1.5, -1.2])


def _xy(cell):
    point = np.asarray(cell.get("bucket_xy_m"), dtype=float)
    require(point.shape == (2,) and np.isfinite(point).all(), "Expected finite bucket XY")
    return point


def geometry_prior(cells):
    """Exact float64 recipe underlying the unchanged ContinuousPolicy prior."""
    require(len(cells) > 0, "Expected nonempty geometry contexts")
    latent = np.tile(np.arctanh(INITIAL_NORMALIZED), (len(cells), 1))
    for i, cell in enumerate(cells):
        radius = np.clip(np.linalg.norm(_xy(cell) - REFERENCE), .29, 1.19)
        latent[i, 1] = np.arctanh(2 * (radius - .28) / (1.20 - .28) - 1)
    return dict(latent=latent, actions_normalized=np.tanh(latent))


def decode_normalized(actions):
    values = np.asarray(actions, dtype=float)
    require(values.ndim == 2 and values.shape[1] == 5 and np.isfinite(values).all()
            and (np.abs(values) <= 1).all(), "Expected finite normalized N by 5 actions")
    return LOW + (values + 1) * (HIGH - LOW) / 2


def _train_cells(cells):
    require(isinstance(cells, (list, tuple)) and len(cells) == 35, "Benchmark requires exactly 35 TRAIN contexts")
    result = []
    for cell in cells:
        require(isinstance(cell.get("id"), str) and cell["id"] and cell.get("split") == "train",
                "Sweep storage accepts TRAIN contexts only")
        result.append(dict(id=cell["id"], split="train", bucket_xy_m=_xy(cell).tolist()))
    require(len({c["id"] for c in result}) == 35 and len({tuple(c["bucket_xy_m"]) for c in result}) == 35,
            "TRAIN contexts must have distinct IDs and coordinates")
    return result


def _reward(value):
    require(type(value) in (int, float) and math.isfinite(value), "Expected finite observed TRAIN reward")
    return float(value)


class GuidedSweep:
    """Bounded training search with a serializable, direct unseen-XY predictor."""

    def __init__(self, train_cells, seed, protocol_fingerprint=None):
        require(type(seed) is int and seed in SEEDS, "Unknown benchmark seed")
        self.protocol_fingerprint = make_protocol()["fingerprint"]
        if protocol_fingerprint is not None:
            require(protocol_fingerprint == self.protocol_fingerprint, "Benchmark protocol mismatch")
        self.train = _train_cells(train_cells)
        self.seed = seed
        self.counts = {c["id"]: 0 for c in self.train}
        self.observed_counts = dict(self.counts)
        self.issued = self.observed = 0
        self.best, self.pending = {}, {}
        self.history_hash = canonical_hash(dict(version=VERSION, seed=seed, train=self.train,
            protocol_fingerprint=self.protocol_fingerprint))
        self._sobol_cache = {}

    def _context(self, cell):
        require(cell.get("split") == "train", "Held-out outcomes/proposals cannot train the sweep")
        found = next((c for c in self.train if c["id"] == cell.get("id")), None)
        require(found is not None and np.array_equal(_xy(cell), found["bucket_xy_m"]),
                "Unknown or changed TRAIN context")
        return found

    def _unit(self, cell, index):
        key = cell["id"]
        if key not in self._sobol_cache:
            seed = int(canonical_hash(dict(seed=self.seed, xy=cell["bucket_xy_m"]))[:8], 16)
            self._sobol_cache[key] = qmc.Sobol(d=5, scramble=True, seed=seed, bits=30).random_base2(9) + 2.**-31
        return self._sobol_cache[key][index]

    def _propose(self, cell, global_index):
        index = self.counts[cell["id"]]
        require(index < 286, "Per-site budget exceeds the declared paired schedule")
        prior = geometry_prior([cell])["latent"][0]
        unit = self._unit(cell, index)
        source = None
        scale = None
        if index == 0:
            latent, phase = prior, "shared_geometry_prior"
        elif index <= 10:
            delta = np.zeros(5)
            axis = (index - 1) // 2
            delta[axis] = (1 if index % 2 == 0 else -1) * 2 * np.exp(INITIAL_LOGSTD[axis])
            latent, phase = prior + delta, "signed_prior_axis"
        elif index <= 42:
            latent = prior + ndtri(unit) * 2 * np.exp(INITIAL_LOGSTD)
            phase = "local_sobol_prior"
        elif (index - 43) % 4 == 0:
            latent, phase = np.arctanh(2 * unit - 1), "global_domain_sobol"
        else:
            best = self.best.get(cell["id"])
            source = best["proposal_id"] if best is not None else None
            center = prior if best is None else prior + np.asarray(best["latent_residual"])
            scale = 1.0 if global_index < 2500 else .5 if global_index < 5000 else .25
            latent = center + ndtri(unit) * 4 * np.exp(INITIAL_LOGSTD) * scale
            phase = "train_best_refinement"
        normalized = np.tanh(latent)
        metadata = dict(cell_id=cell["id"], bucket_xy_m=cell["bucket_xy_m"],
            context_proposal_index=index, global_request_index=global_index,
            phase=phase, refinement_scale=scale, source_training_proposal_id=source,
            latent=latent.tolist(), actions_normalized=normalized.tolist())
        metadata["proposal_id"] = canonical_hash(dict(version=VERSION, seed=self.seed,
            protocol_fingerprint=self.protocol_fingerprint, **metadata))
        return metadata

    def propose(self, cells):
        """Issue each requested candidate once; the caller must observe rejects too."""
        require(len(cells) > 0 and self.issued + len(cells) <= TRAIN_BUDGET,
                "Empty request or exceeded paid TRAIN budget")
        canonical = [self._context(c) for c in cells]
        require(len({c["id"] for c in canonical}) == len(canonical), "Duplicate context in native batch")
        pending_ids = {p["cell_id"] for p in self.pending.values()}
        require(not any(c["id"] in pending_ids for c in canonical), "Previous candidate has not been observed")
        metadata = [self._propose(c, self.issued + i) for i, c in enumerate(canonical)]
        for item in metadata:
            self.pending[item["proposal_id"]] = copy.deepcopy(item)
            self.counts[item["cell_id"]] += 1
            self.history_hash = canonical_hash(dict(previous=self.history_hash, issued=item))
        self.issued += len(metadata)
        return dict(actions_normalized=np.asarray([p["actions_normalized"] for p in metadata]),
                    proposal_metadata=copy.deepcopy(metadata))

    def observe(self, rows):
        """Only issued TRAIN rows can enter storage; no reward-based filtering."""
        require(len(rows) > 0, "Empty TRAIN observation")
        require(len({r.get("proposal_id") for r in rows}) == len(rows), "Duplicate TRAIN observation")
        checked = []
        for row in rows:
            pending = self.pending.get(row.get("proposal_id"))
            require(pending is not None, "Unknown, duplicate or unissued TRAIN observation")
            require(row.get("cell_id") == pending["cell_id"]
                    and np.array_equal(_xy(row), pending["bucket_xy_m"])
                    and np.array_equal(np.asarray(row.get("actions_normalized")), pending["actions_normalized"]),
                    "Observed action/context differs from its paid proposal")
            require(row.get("split", "train") == "train", "Held-out reward cannot enter sweep storage")
            checked.append((pending, _reward(row.get("reward"))))
        for proposal, reward in checked:
            cell_id = proposal["cell_id"]
            old = self.best.get(cell_id)
            if old is None or reward > old["reward"]:
                prior = geometry_prior([dict(bucket_xy_m=proposal["bucket_xy_m"])])["latent"][0]
                self.best[cell_id] = dict(proposal_id=proposal["proposal_id"], reward=reward,
                    global_request_index=proposal["global_request_index"],
                    actions_normalized=proposal["actions_normalized"],
                    latent=proposal["latent"], latent_residual=(np.asarray(proposal["latent"]) - prior).tolist())
            self.observed_counts[cell_id] += 1
            self.observed += 1
            del self.pending[proposal["proposal_id"]]
            self.history_hash = canonical_hash(dict(previous=self.history_hash,
                observed_proposal_id=proposal["proposal_id"], reward=reward))

    def predict(self, cells):
        """One frozen nearest-TRAIN transfer, with zero query-time outcome search."""
        require(len(cells) > 0, "Expected prediction contexts")
        prior = geometry_prior(cells)["latent"]
        output, metadata = [], []
        state_hash = self.state()["fingerprint"]
        for i, cell in enumerate(cells):
            point = _xy(cell)
            nearest = min(self.train, key=lambda c: (float(np.linalg.norm(_xy(c) - point)), c["id"]))
            best = self.best.get(nearest["id"])
            latent = prior[i] if best is None else prior[i] + np.asarray(best["latent_residual"])
            output.append(np.tanh(latent))
            metadata.append(dict(nearest_training_cell_id=nearest["id"],
                source_training_proposal_id=best["proposal_id"] if best is not None else None,
                state_fingerprint=state_hash, search_at_inference=False,
                ball_flight_queries=0, deterministic=True, latent=latent.tolist()))
        return dict(actions_normalized=np.asarray(output), prediction_metadata=metadata)

    def state(self):
        return signed(dict(schema=STATE_VERSION, method_version=VERSION, seed=self.seed,
            protocol_fingerprint=self.protocol_fingerprint, domain_fingerprint=action_domain()["fingerprint"],
            train=copy.deepcopy(self.train), counts=dict(self.counts), observed_counts=dict(self.observed_counts),
            issued=self.issued, observed=self.observed, best=copy.deepcopy(self.best),
            pending=copy.deepcopy(self.pending), history_hash=self.history_hash))

    @classmethod
    def from_state(cls, state, *, protocol_fingerprint=None, train_cells=None):
        validate_fingerprint(state)
        expected_keys = {"schema", "method_version", "seed", "protocol_fingerprint", "domain_fingerprint",
            "train", "counts", "observed_counts", "issued", "observed", "best", "pending", "history_hash", "fingerprint"}
        require(set(state) == expected_keys and state["schema"] == STATE_VERSION
                and state["method_version"] == VERSION, "Unknown sweep state contract")
        obj = cls(state["train"], state["seed"], protocol_fingerprint or state["protocol_fingerprint"])
        require(state["protocol_fingerprint"] == obj.protocol_fingerprint
                and state["domain_fingerprint"] == action_domain()["fingerprint"], "Sweep domain/protocol mismatch")
        if train_cells is not None:
            require(obj.train == _train_cells(train_cells), "Sweep TRAIN geometry changed")
        ids = set(obj.counts)
        require(set(state["counts"]) == set(state["observed_counts"]) == ids,
                "Sweep count context membership changed")
        for key in ids:
            n, seen = state["counts"][key], state["observed_counts"][key]
            require(type(n) is int and type(seen) is int and 0 <= seen <= n <= 286, "Invalid sweep counters")
        require(type(state["issued"]) is int and type(state["observed"]) is int
                and 0 <= state["observed"] <= state["issued"] <= TRAIN_BUDGET
                and sum(state["counts"].values()) == state["issued"]
                and sum(state["observed_counts"].values()) == state["observed"]
                and len(state["pending"]) == state["issued"] - state["observed"], "Inconsistent sweep accounting")
        require(isinstance(state["history_hash"], str) and len(state["history_hash"]) == 64
                and set(state["history_hash"]).issubset("0123456789abcdef"),
                "Invalid sweep history binding")
        require(set(state["best"]).issubset(ids), "Non-TRAIN evidence in sweep library")
        for cell_id, best in state["best"].items():
            require(state["observed_counts"][cell_id] > 0, "Unobserved best TRAIN candidate")
            require(set(best) == {"proposal_id", "reward", "global_request_index", "actions_normalized", "latent", "latent_residual"}
                    and isinstance(best["proposal_id"], str) and len(best["proposal_id"]) == 64
                    and set(best["proposal_id"]).issubset("0123456789abcdef"), "Invalid best TRAIN evidence fields")
            _reward(best["reward"])
            latent = np.asarray(best["latent"], dtype=float)
            residual = np.asarray(best["latent_residual"], dtype=float)
            normalized = np.asarray(best["actions_normalized"], dtype=float)
            point = next(c for c in obj.train if c["id"] == cell_id)
            require(latent.shape == residual.shape == normalized.shape == (5,)
                    and np.isfinite([latent, residual, normalized]).all()
                    and np.allclose(np.tanh(latent), normalized, rtol=0, atol=1e-14)
                    and np.allclose(latent - geometry_prior([point])["latent"][0], residual, rtol=0, atol=1e-12)
                    and type(best["global_request_index"]) is int
                    and 0 <= best["global_request_index"] < state["issued"], "Changed sweep best action")
        require(set(state["best"]) == {key for key in ids if state["observed_counts"][key]},
                "A paid TRAIN result was silently dropped")
        obj.counts, obj.observed_counts = copy.deepcopy(state["counts"]), copy.deepcopy(state["observed_counts"])
        obj.issued, obj.observed = state["issued"], state["observed"]
        obj.best, obj.pending = copy.deepcopy(state["best"]), copy.deepcopy(state["pending"])
        obj.history_hash = state["history_hash"]
        # Pending payloads are regenerated from the pre-issue count and compared;
        # only the geometry/phase/proposal recipe may survive serialization.
        pending_cells = [p["cell_id"] for p in obj.pending.values()]
        require(len(set(pending_cells)) == len(pending_cells) and set(pending_cells).issubset(ids),
                "Invalid pending TRAIN membership")
        require(all(obj.counts[key] - obj.observed_counts[key] == int(key in pending_cells) for key in ids),
                "Pending TRAIN counters changed context")
        for proposal_id, proposal in obj.pending.items():
            require(type(proposal.get("global_request_index")) is int
                    and 0 <= proposal["global_request_index"] < obj.issued, "Invalid pending request ordinal")
            cell = next(c for c in obj.train if c["id"] == proposal["cell_id"])
            index = obj.counts[cell["id"]]
            obj.counts[cell["id"]] -= 1
            expected = obj._propose(cell, proposal["global_request_index"])
            obj.counts[cell["id"]] = index
            require(expected == proposal and proposal_id == proposal["proposal_id"], "Changed pending sweep proposal")
        return obj

    def save(self, path):
        require(not self.pending and self.issued == self.observed,
                "Cannot freeze a sweep with unobserved paid candidates")
        with Path(path).open("x") as handle:
            json.dump(self.state(), handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")

    @classmethod
    def load(cls, path, *, protocol_fingerprint=None, train_cells=None):
        return cls.from_state(json.loads(Path(path).read_text()),
            protocol_fingerprint=protocol_fingerprint, train_cells=train_cells)


def make_cpu_ppo(seed):
    """Fresh original generic actor, without touching CUDA or caller RNG state."""
    require(type(seed) is int and seed in SEEDS, "Unknown benchmark seed")
    import torch
    from continuous_policy import ContinuousPolicy
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        return ContinuousPolicy().cpu()


def predict_cpu_ppo(policy, cells):
    import torch
    from continuous_policy import build_observations, sample
    require(all(p.device.type == "cpu" for p in policy.parameters()), "CPU benchmark facade refuses GPU model")
    observations = torch.as_tensor(build_observations(cells), dtype=torch.float32)
    picked = sample(policy, observations, deterministic=True)
    return dict(actions_normalized=picked["actions_normalized"].detach().numpy(),
        latent=picked["latent"].detach().numpy(), logp=picked["logp"].detach().numpy(),
        search_at_inference=False, ball_flight_queries=0, deterministic=True)
