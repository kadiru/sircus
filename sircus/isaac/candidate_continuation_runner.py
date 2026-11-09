"""Bounded TRAIN driver, without native dispatch or an executable training CLI.

Explicit callers supply a verified LearningEngine and external terminal
receipts. Reservations are durable before choices are returned. This driver
cannot command arms, start Isaac, bypass native guards, or open held-out inputs.
Its learning audit does not replace an independent native scoring/guard audit.
"""

import copy
import hashlib
import math
from pathlib import Path
import re
import shutil
import time

from candidate_continuation_contract import PROTOCOL_SHA256, PPO_BUFFER, RUN_SCHEMA
from candidate_continuation_journal import Journal, write_new_json, _sync_directory
from candidate_continuation_ledger import Ledger, Outcome, canonical_hash
from candidate_continuation_policy import CandidateDistribution, torch
from candidate_continuation_checkpoint import checkpoint_payload, save_checkpoint_new
from continuous_policy import weight_hash

STORAGE_FLOOR_BYTES = 6 * 1024**3
PARENT_WEIGHT_HASH = "ea9a69ec063e3d71cda2b92f1da64939f644244d4b4dbd70b500be132632e549"
RESULT_FIELDS = {"request_id", "released", "decline_reason", "source_sha256", "return", "captured", "clean"}


def require(value, message):
    if not value:
        raise ValueError(message)


def source_fingerprint():
    """Bind the exact source snapshot, including unchanged native guard sources."""
    source = Path(__file__).resolve().parent
    return canonical_hash({p.name:hashlib.sha256(p.read_bytes()).hexdigest()
                           for p in sorted(source.glob("*.py"))})


def _join(distributions):
    return CandidateDistribution(
        context_ids=tuple(c for d in distributions for c in d.context_ids),
        support_fingerprints=tuple(c for d in distributions for c in d.support_fingerprints),
        candidate_ids=tuple(c for d in distributions for c in d.candidate_ids),
        scores=tuple(c for d in distributions for c in d.scores),
        log_probabilities=tuple(c for d in distributions for c in d.log_probabilities),
        values=torch.cat([d.values for d in distributions]),
        parent_indices=tuple(c for d in distributions for c in d.parent_indices))


class ContinuationRunner:
    """Single-use driver: imitation -> reserved TRAIN requests -> final artifact.

    Merely constructing this object performs no optimizer step or shot. There
    is deliberately no resume, native callback, assessment or launch method.
    A future native producer must separately prove preparation, source guards,
    actual GPU ownership, startup diagnostics and full outcome evidence.
    """

    def __init__(self, directory, engine, ledger, bindings, *, synthetic,
                 clock=time.monotonic, free_bytes=None):
        require(torch is not None, "Configured CPU PyTorch is required")
        require(type(synthetic) is bool, "Explicit synthetic evidence label required")
        require(type(ledger) is Ledger and ledger.phase == "train" and not ledger.batches
                and ledger.stopped_reason is None, "A new empty TRAIN ledger is required")
        require(engine.stage == "imitation" and all(v == 0 for v in engine.snapshot_counters().values()),
                "An untouched learning engine is required")
        require(tuple(s.context_id for s in engine.supports) == ledger.train_context_ids,
                "Engine and ledger TRAIN contexts differ")
        require(type(bindings) is dict and all(type(v) is str and re.fullmatch(r"[0-9a-f]{64}", v)
                for v in bindings.values()), "Expected external SHA256 bindings")
        require(bindings.get("parent_checkpoint_sha256") == ledger.bindings.parent_checkpoint_sha256
                and bindings.get("learning_protocol_sha256") == PROTOCOL_SHA256
                and bindings.get("support_sha256") == canonical_hash([s.to_dict() for s in engine.supports])
                and bindings.get("demonstrations_sha256") == canonical_hash(list(engine.demonstrations))
                and bindings.get("observations_sha256") == canonical_hash(engine.observations.detach().cpu().tolist())
                and bindings.get("parent_weight_hash") == weight_hash(engine.policy)
                and bindings.get("source_sha256") == source_fingerprint()
                and "input_sha256" in bindings, "Run source/input/support/parent identity changed")
        require(synthetic or bindings["parent_weight_hash"] == PARENT_WEIGHT_HASH,
                "Real continuation must initialize from exact seed-22 tensors")
        require(synthetic or (ledger.bindings.complete and ledger.validation_context_ids is not None
                and ledger.test_context_ids is not None
                and all(bindings.get(k) == getattr(ledger.bindings, k) for k in
                    ("fresh_split_manifest_sha256", "fresh_support_sha256", "guard_protocol_sha256",
                     "scorer_sha256", "producer_source_sha256", "train_bank_sha256", "train_evidence_sha256"))),
                "Real learning requires externally verified preparation and producer bindings")
        # These two fields are the minimum training-only links; incomplete fresh
        # preparation remains visible and cannot authorize a native launch.
        require(ledger.bindings.train_support_sha256 == bindings["support_sha256"],
                "Ledger TRAIN support binding differs")
        self.engine, self.ledger = engine, ledger
        self.bindings, self.synthetic = copy.deepcopy(bindings), synthetic
        self.directory = Path(directory)
        self.directory.mkdir()
        _sync_directory(self.directory.parent)
        self.clock = clock
        self.free_bytes = free_bytes or (lambda:shutil.disk_usage(self.directory).free)
        self.started = self.phase_started = self.clock()
        self.cpu_started = self.phase_cpu_started = time.process_time()
        self.phase_costs = {}
        self.failed = self.finished = False
        self.pending_choices = []
        self.pending_distribution = None
        self.buffer_distributions, self.buffer_choices, self.buffer_results = [], [], []
        self.last_weight = weight_hash(engine.policy)
        self.journal = Journal(self.directory/"journal")
        self.journal.append("start", dict(bindings=self.bindings,
            learning_protocol_sha256=PROTOCOL_SHA256, ledger=self.ledger.to_dict(), synthetic=synthetic))

    def _guard(self):
        require(not self.failed and not self.finished, "Runner stopped; continuation cannot resume")
        require(self.bindings["source_sha256"] == source_fingerprint(), "Producer source changed during run")
        require(tuple(s.context_id for s in self.engine.supports) == self.ledger.train_context_ids
                and canonical_hash([s.to_dict() for s in self.engine.supports]) == self.bindings["support_sha256"],
                "Frozen TRAIN support changed during run")
        require(self.free_bytes() >= STORAGE_FLOOR_BYTES, "Original 6 GiB storage floor reached")
        limit = 600 if self.engine.stage == "imitation" else 7200
        require(self.clock()-self.phase_started < limit, "Learning phase wall limit reached")
        require(weight_hash(self.engine.policy) == self.last_weight, "Model changed outside the bounded updater")

    def _fail(self, error):
        if self.failed or self.finished:
            return
        self.failed = True
        if self.ledger.stopped_reason is None:
            self.ledger = self.ledger.halt(type(error).__name__+": "+str(error))
        if not self.journal.failed:
            self.journal.append("failure", dict(reason=self.ledger.stopped_reason,
                ledger=self.ledger.to_dict(), counters=self.engine.snapshot_counters()))

    def imitation_step(self):
        try:
            self._guard()
            receipt = self.engine.imitation_step()
            self.last_weight = weight_hash(self.engine.policy)
            self.journal.append("imitation_update", receipt)
            self._guard()
            return copy.deepcopy(receipt)
        except BaseException as error:
            self._fail(error)
            raise

    def start_ppo(self):
        try:
            self._guard()
            self.engine.start_ppo()
            self.phase_costs["imitation"] = dict(wall_s=self.clock()-self.phase_started,
                process_cpu_s=time.process_time()-self.phase_cpu_started)
            self.phase_started = self.clock()
            self.phase_cpu_started = time.process_time()
            self.journal.append("ppo_start", dict(counters=self.engine.snapshot_counters()))
        except BaseException as error:
            self._fail(error)
            raise

    def reserve_train_batch(self):
        """Return choices only after their request reservation is fsynced."""
        try:
            self._guard()
            require(self.engine.stage == "ppo", "PPO phase is not open")
            require(self.ledger.pending is None, "Reserved requests still await outcomes")
            batch = self.ledger.next_batch()
            require(batch is not None, "TRAIN cap reached; freeze the final artifact")
            require(len(self.buffer_results)+len(batch.requests) <= PPO_BUFFER,
                    "Native batch would straddle a frozen PPO buffer")
            contexts = tuple(r.context_id for r in batch.requests)
            distribution, indices = self.engine.sample(contexts)
            indices = distribution._indices(indices)
            support_by_id = {s.context_id:s for s in self.engine.supports}
            choices = []
            for i, (request, index) in enumerate(zip(batch.requests, indices)):
                support = support_by_id[request.context_id]
                choices.append(dict(request_id=request.request_id, context_id=request.context_id,
                    candidate_id=support.candidate_ids[index], index=index,
                    support_fingerprint=support.support_fingerprint, action=support.action(index),
                    latent=list(support.latents[index]), old_log_probability=float(distribution.log_probabilities[i][index]),
                    old_value=float(distribution.values[i])))
            reserved = self.ledger.reserve_batch(batch)
            self.journal.append("reserve", dict(ledger=reserved.to_dict(), choices=choices,
                policy_weight_hash=self.last_weight))
            self.ledger = reserved
            self.pending_choices, self.pending_distribution = choices, distribution
            return copy.deepcopy(choices)
        except BaseException as error:
            self._fail(error)
            raise

    def record_results(self, results):
        """Persist an ordered terminal prefix; rejected selections remain samples."""
        try:
            self._guard()
            require(self.ledger.pending is not None, "No reserved request awaits a result")
            require(type(results) in (list, tuple) and results, "Expected a nonempty terminal prefix")
            offset = len(self.ledger.pending.outcomes)
            remaining = self.ledger.pending.requests[offset:]
            require(len(results) <= len(remaining), "More outcomes than reserved requests")
            outcomes = []
            for row, request in zip(results, remaining):
                require(type(row) is dict and set(row) == RESULT_FIELDS, "Terminal receipt schema changed")
                require(row["request_id"] == request.request_id, "Terminal order or request identity changed")
                require(all(type(row[k]) is bool for k in ("released", "captured", "clean")),
                        "Terminal flags must be Boolean")
                require(not row["clean"] or row["captured"], "Clean capture lacks capture")
                require(not row["captured"] or row["released"], "Rejected request cannot capture")
                require(type(row["return"]) in (float, int) and math.isfinite(row["return"]), "Nonfinite terminal return")
                require(row["released"] or row["return"] == -1., "Pre-release rejection must keep original invalid reward")
                require(not row["captured"] or row["return"] == (1.6 if row["clean"] else 1.5),
                        "Capture return differs from unchanged reward")
                require(not row["released"] or row["captured"] or -.8 <= row["return"] <= 0.,
                        "Released miss return exceeds original reward bounds")
                require(type(row["source_sha256"]) is str and re.fullmatch(r"[0-9a-f]{64}", row["source_sha256"]),
                        "Full terminal source bytes must be bound")
                outcomes.append(Outcome(row["request_id"], row["released"], row["decline_reason"], row["source_sha256"]))
            updated = self.ledger.record_outcomes(outcomes)
            self.journal.append("outcomes", dict(ledger=updated.to_dict(), results=copy.deepcopy(list(results))))
            self.ledger = updated
            self.buffer_results.extend(copy.deepcopy(results))
            if updated.pending is None:
                self.buffer_choices.extend(copy.deepcopy(self.pending_choices))
                self.buffer_distributions.append(self.pending_distribution)
                self.pending_choices, self.pending_distribution = [], None
                at_cap = self.ledger.next_batch() is None
                if len(self.buffer_results) == PPO_BUFFER or at_cap:
                    self._flush(final=at_cap)
            return self.ledger.summary()
        except BaseException as error:
            self._fail(error)
            raise

    def _flush(self, *, final):
        require(self.ledger.pending is None and self.buffer_results, "Cannot update unresolved requests")
        require(len(self.buffer_results) == len(self.buffer_choices), "Selected requests missing terminal outcomes")
        contexts = tuple(c["context_id"] for c in self.buffer_choices)
        indices = tuple(c["index"] for c in self.buffer_choices)
        receipt = self.engine.ppo_update(contexts, indices, [r["return"] for r in self.buffer_results],
            _join(self.buffer_distributions), final=final)
        self.last_weight = weight_hash(self.engine.policy)
        receipt = dict(receipt, request_ids=[c["request_id"] for c in self.buffer_choices])
        self.journal.append("ppo_update", receipt)
        self.buffer_distributions, self.buffer_choices, self.buffer_results = [], [], []
        self._guard()

    def finish(self):
        """Freeze one new final model after the first TRAIN cap, never assessment."""
        try:
            self._guard()
            require(self.engine.stage == "ppo" and self.ledger.pending is None,
                    "TRAIN has unresolved requests or is in the wrong phase")
            require(self.ledger.next_batch() is None and not self.buffer_results,
                    "TRAIN cap and final buffer flush are required")
            state = self.engine.state_dict()
            payload = checkpoint_payload(self.engine.policy, stage="final",
                counters=self.engine.snapshot_counters(), bindings=self.bindings,
                rng_states=state["rng_states"], optimizer_states=state["optimizer_states"],
                ledger_fingerprint=self.ledger.to_dict()["fingerprint"], journal_fingerprint=self.journal.head)
            checkpoint_sha = save_checkpoint_new(self.directory/"final.pt", payload)
            self._guard()
            self.engine.finish_ppo()
            self.journal.append("final", dict(checkpoint_sha256=checkpoint_sha,
                ledger=self.ledger.to_dict(), counters=self.engine.snapshot_counters(), weight_hash=self.last_weight))
            result = dict(schema=RUN_SCHEMA, status="learning_complete_audit_pending", synthetic=self.synthetic,
                bindings=self.bindings, journal_head_sha256=self.journal.head,
                checkpoint_sha256=checkpoint_sha, counters=self.engine.snapshot_counters(),
                ledger=self.ledger.to_dict(), wall_s=self.clock()-self.started,
                process_cpu_s=time.process_time()-self.cpu_started,
                phase_costs=dict(self.phase_costs, ppo_including_external_waits=dict(
                    wall_s=self.clock()-self.phase_started,
                    process_cpu_s=time.process_time()-self.phase_cpu_started)),
                resource_scope="Driver process CPU and wall; PPO includes external receipt waits and final checkpoint serialization. Native setup/flight, initialization, GPU memory and device occupancy require the separate producer; no isolated native cost is inferred here.",
                native_dispatch_implemented=False, held_out_shot_files_read=0,
                execution_authorized=False, release_authorized=False,
                scope="TRAIN learning receipts only; native guard/scoring audit and fresh assessment remain separate.")
            result["fingerprint"] = canonical_hash(result)
            write_new_json(self.directory/"result.json", result)
            self.finished = True
            return copy.deepcopy(result)
        except BaseException as error:
            self._fail(error)
            raise
