"""Pure, immutable accounting for the proposed seed-22 continuation.

This module performs no I/O and imports no policy, optimizer or runtime. A
reservation consumes requests immediately; unresolved reservations cannot be
refunded, replaced or retried. Terminal receipts consume releases separately.
The fixed proposal limits are protocol identity, never caller configuration.

Typical use is ``Ledger.new(train_ids)`` followed by ``next_batch()``,
``reserve_batch(batch)`` and ``record_batch(outcomes)``. Every operation returns
a new ledger. Persist ``to_dict()`` through a separately reviewed producer;
``from_dict()`` verifies its fingerprint and replays accounting invariants.
Source hashes default to explicit unbound placeholders. Neither a populated
source binding nor a ledger receipt grants execution or release authorization.
"""

from dataclasses import asdict, dataclass, replace
import hashlib
import json
import random
import re


SCHEMA = "sircus-candidate-continuation-request-ledger-v1"
PROTOCOL_SCHEMA = "sircus-seed22-continuation-accounting-protocol-v1"
PARENT_CHECKPOINT_SHA256 = "57d7658e9ed2f72ad09cf4e8523ab497ccbe0708364f1b8ec4a4c5c1aef1fe9f"
TRAIN_REQUEST_LIMIT = 1750
TRAIN_RELEASE_LIMIT = 1400
NATIVE_BATCH_SIZE = 35
ASSESSMENT_REQUEST_LIMIT = 420
ASSESSMENT_RELEASE_LIMIT = 420
CONTINUATION_SEED = 20261125
METHODS = ("parent", "candidate_bank", "continued")
# Each entry is one 35-position native cohort. The two TEST halves use
# different prescribed rotations and keep every position in the denominator.
ASSESSMENT_COHORTS = (
    ("train", 0, "parent"), ("train", 0, "candidate_bank"), ("train", 0, "continued"),
    ("validation", 0, "candidate_bank"), ("validation", 0, "continued"), ("validation", 0, "parent"),
    ("test", 0, "continued"), ("test", 0, "parent"), ("test", 0, "candidate_bank"),
    ("test", 35, "parent"), ("test", 35, "candidate_bank"), ("test", 35, "continued"),
)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def canonical_hash(value):
    """Hash finite canonical JSON; mutable callers receive no stored reference."""
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(raw).hexdigest()


def protocol():
    """Return a fresh copy of fixed accounting rules, not a launch config."""
    return dict(schema=PROTOCOL_SCHEMA, parent_checkpoint_sha256=PARENT_CHECKPOINT_SHA256,
        train=dict(request_limit=TRAIN_REQUEST_LIMIT, release_limit=TRAIN_RELEASE_LIMIT,
            context_count=35, schedule="seeded-permutation-per-logical-round",
            schedule_rng="python-random-MT19937", schedule_seed=CONTINUATION_SEED),
        assessment=dict(request_limit=ASSESSMENT_REQUEST_LIMIT, release_limit=ASSESSMENT_RELEASE_LIMIT,
            positions_per_method=dict(train=35, validation=35, test=70), methods=list(METHODS),
            cohorts=[dict(split=s, offset=o, method=m) for s, o, m in ASSESSMENT_COHORTS],
            repeat_or_replacement_allowed=False, final_checkpoint_frozen_before_assessment=True),
        maximum_new_task_requests=2170, maximum_new_task_releases=1820,
        native_batch_size=NATIVE_BATCH_SIZE,
        active_batch_rule="min(35, remaining-phase-requests, remaining-phase-releases)",
        request_charge="reservation", release_charge="terminal-released-true",
        unresolved_requests_refundable=False, inactive_padding_counts=False,
        inherited_parent_requests=10000, prior_comparison_requests=70,
        startup_diagnostics="separate-non-task-accounting-not-implemented-here",
        operational_wall_storage_and_runtime_checks="not-implemented-by-this-ledger",
        execution_authorized=False, release_authorized=False)


PROTOCOL_SHA256 = canonical_hash(protocol())


def _text(value, label):
    _require(type(value) is str and bool(value.strip()), "Expected nonempty " + label)


def _integer(value, label, minimum=0):
    _require(type(value) is int and value >= minimum, "Expected integer " + label)


def _sha(value, label, optional=False):
    if optional and value is None:
        return
    _require(type(value) is str and re.fullmatch(r"[0-9a-f]{64}", value) is not None,
             "Expected SHA256 " + label)


def _contexts(value, count, label, optional=False):
    if optional and value is None:
        return
    _require(type(value) is tuple and len(value) == count, "Expected " + label + " context count")
    for context in value:
        _text(context, label + " context ID")
    _require(len(set(value)) == count, "Duplicate " + label + " context ID")


@dataclass(frozen=True, slots=True)
class SourceBindings:
    """Hash placeholders for evidence checked by a future independent producer.

    ``None`` means missing, not verified or implicitly inherited. The exact
    original parent identity cannot be replaced. Final checkpoint identity is
    frozen separately when entering assessment, after TRAIN has stopped.
    """

    parent_checkpoint_sha256: str = PARENT_CHECKPOINT_SHA256
    train_bank_sha256: str | None = None
    train_evidence_sha256: str | None = None
    train_support_sha256: str | None = None
    fresh_split_manifest_sha256: str | None = None
    fresh_support_sha256: str | None = None
    guard_protocol_sha256: str | None = None
    scorer_sha256: str | None = None
    producer_source_sha256: str | None = None

    def __post_init__(self):
        _require(self.parent_checkpoint_sha256 == PARENT_CHECKPOINT_SHA256,
                 "Original parent checkpoint identity changed")
        for name, value in asdict(self).items():
            _sha(value, name, optional=name != "parent_checkpoint_sha256")

    @property
    def fingerprint(self):
        return canonical_hash(asdict(self))

    @property
    def complete(self):
        return all(value is not None for value in asdict(self).values())


@dataclass(frozen=True, slots=True)
class Request:
    """An immutable scheduled task request; indices are local to its phase."""

    request_id: str
    request_index: int
    phase: str
    split: str
    method: str
    context_id: str
    protocol_sha256: str
    schedule_sha256: str
    source_binding_sha256: str
    final_checkpoint_sha256: str | None = None

    def __post_init__(self):
        for name in ("request_id", "phase", "split", "method", "context_id"):
            _text(getattr(self, name), name)
        _integer(self.request_index, "request index")
        _require(self.phase in ("train", "assessment"), "Unknown request phase")
        _require(self.split in ("train", "validation", "test"), "Unknown request split")
        _require(self.method in METHODS, "Unknown request method")
        _require(self.protocol_sha256 == PROTOCOL_SHA256, "Request protocol fingerprint changed")
        _sha(self.schedule_sha256, "request schedule")
        _sha(self.source_binding_sha256, "request source binding")
        _sha(self.final_checkpoint_sha256, "final checkpoint", optional=self.phase == "train")
        _require(self.phase != "train" or (self.split == "train" and self.method == "continued"
            and self.final_checkpoint_sha256 is None), "TRAIN request identity changed")


@dataclass(frozen=True, slots=True)
class Outcome:
    """One terminal outcome; every guard decline costs one request, no release.

    Missing ``source_sha256`` is explicitly unbound synthetic/bookkeeping data.
    Capture, reward and PPO probability auditing belong to separate receipts.
    """

    request_id: str
    released: bool
    decline_reason: str | None = None
    source_sha256: str | None = None

    def __post_init__(self):
        _text(self.request_id, "outcome request ID")
        _require(type(self.released) is bool, "Released status must be Boolean")
        if self.released:
            _require(self.decline_reason is None, "Released outcome cannot be a pre-release decline")
        else:
            _text(self.decline_reason, "decline reason")
        _sha(self.source_sha256, "outcome source", optional=True)


@dataclass(frozen=True, slots=True)
class Batch:
    """Active requests only; ``native_slots`` exposes explicit None padding."""

    batch_index: int
    phase: str
    requests: tuple[Request, ...]
    outcomes: tuple[Outcome, ...] = ()

    def __post_init__(self):
        _integer(self.batch_index, "batch index")
        _require(self.phase in ("train", "assessment"), "Unknown batch phase")
        _require(type(self.requests) is tuple and 0 < len(self.requests) <= NATIVE_BATCH_SIZE,
                 "Batch requires 1 to 35 active requests")
        _require(all(type(x) is Request and x.phase == self.phase for x in self.requests),
                 "Batch request phase/type changed")
        _require(type(self.outcomes) is tuple and len(self.outcomes) <= len(self.requests)
            and all(type(x) is Outcome for x in self.outcomes), "Invalid batch outcomes")
        _require(len({r.request_id for r in self.requests}) == len(self.requests), "Duplicate request ID")
        _require(tuple(o.request_id for o in self.outcomes) ==
                 tuple(r.request_id for r in self.requests[:len(self.outcomes)]),
                 "Terminal outcomes must follow reserved request order without duplicates")

    @property
    def native_slots(self):
        return self.requests + (None,) * (NATIVE_BATCH_SIZE - len(self.requests))

    @property
    def complete(self):
        return len(self.outcomes) == len(self.requests)


@dataclass(frozen=True, slots=True)
class Ledger:
    """Append-only logical ledger, returning new immutable values on change.

    Only one batch may remain unresolved. A producer must persist reservations
    before any execution; this pure value object cannot enforce disk durability.
    Reconstructing arbitrary values revalidates the complete scheduling history.
    """

    train_context_ids: tuple[str, ...]
    validation_context_ids: tuple[str, ...] | None = None
    test_context_ids: tuple[str, ...] | None = None
    bindings: SourceBindings = SourceBindings()
    batches: tuple[Batch, ...] = ()
    phase: str = "train"
    final_checkpoint_sha256: str | None = None
    stopped_reason: str | None = None
    schema: str = SCHEMA
    protocol_sha256: str = PROTOCOL_SHA256

    @classmethod
    def new(cls, train_context_ids, validation_context_ids=None, test_context_ids=None, bindings=None):
        for value in (train_context_ids, validation_context_ids, test_context_ids):
            _require(value is None or type(value) in (list, tuple), "Expected explicit context ID lists")
        _require(train_context_ids is not None, "TRAIN contexts are required")
        return cls(tuple(train_context_ids),
            None if validation_context_ids is None else tuple(validation_context_ids),
            None if test_context_ids is None else tuple(test_context_ids),
            SourceBindings() if bindings is None else bindings)

    def __post_init__(self):
        _require(self.schema == SCHEMA and self.protocol_sha256 == PROTOCOL_SHA256,
                 "Ledger schema/protocol fingerprint changed")
        _contexts(self.train_context_ids, 35, "TRAIN")
        _contexts(self.validation_context_ids, 35, "validation", optional=True)
        _contexts(self.test_context_ids, 70, "TEST", optional=True)
        all_ids = self.train_context_ids + (self.validation_context_ids or ()) + (self.test_context_ids or ())
        _require(len(set(all_ids)) == len(all_ids), "Context IDs overlap across splits")
        _require(type(self.bindings) is SourceBindings, "Expected immutable source bindings")
        _require(type(self.batches) is tuple and all(type(b) is Batch for b in self.batches),
                 "Expected immutable batch history")
        _require(self.phase in ("train", "assessment"), "Unknown ledger phase")
        if self.stopped_reason is not None:
            _text(self.stopped_reason, "stop reason")
        if self.phase == "train":
            _require(self.final_checkpoint_sha256 is None, "Final checkpoint cannot be frozen during TRAIN")
        else:
            _require(self.validation_context_ids is not None and self.test_context_ids is not None,
                     "Fresh held-out context preparation is missing")
            _sha(self.final_checkpoint_sha256, "frozen final checkpoint")
        counts = dict(train=[0, 0], assessment=[0, 0])
        assessment_started = False
        train_order = self._train_order()
        for index, batch in enumerate(self.batches):
            _require(batch.batch_index == index, "Batch indices must be contiguous and ordered")
            _require(index == 0 or self.batches[index - 1].complete,
                     "Unresolved requests cannot be refunded, replaced or retried")
            phase = batch.phase
            if phase == "assessment":
                assessment_started = True
                _require(self.phase == "assessment" and self._train_limit_reached(*counts["train"]),
                         "Assessment cannot precede the TRAIN stopping rule")
            else:
                _require(not assessment_started, "TRAIN cannot resume after assessment")
            requests, releases = counts[phase]
            allowed = self._capacity(phase, requests, releases)
            _require(len(batch.requests) == allowed and allowed > 0,
                     "Batch size changed or would overshoot a request/release limit")
            expected = self._requests(phase, requests, allowed, train_order)
            _require(batch.requests == expected, "Request IDs/order/context/source binding changed")
            counts[phase][0] += len(batch.requests)
            counts[phase][1] += sum(o.released for o in batch.outcomes)
        if self.phase == "assessment":
            _require(self._train_limit_reached(*counts["train"]), "TRAIN stopping rule has not been reached")
            _require(self.pending is None or self.pending.phase == "assessment",
                     "Reserved TRAIN requests still need terminal outcomes")

    @property
    def schedule_sha256(self):
        return canonical_hash(dict(protocol_sha256=self.protocol_sha256,
            train_context_ids=self.train_context_ids, validation_context_ids=self.validation_context_ids,
            test_context_ids=self.test_context_ids, train_request_order_sha256=canonical_hash(self._train_order())))

    def _train_order(self):
        rng = random.Random(CONTINUATION_SEED)
        result = []
        while len(result) < TRAIN_REQUEST_LIMIT:
            round_ids = list(self.train_context_ids)
            rng.shuffle(round_ids)
            result.extend(round_ids)
        return tuple(result[:TRAIN_REQUEST_LIMIT])

    @staticmethod
    def _train_limit_reached(requests, releases):
        return requests == TRAIN_REQUEST_LIMIT or releases == TRAIN_RELEASE_LIMIT

    @staticmethod
    def _capacity(phase, requests, releases):
        request_cap, release_cap = ((TRAIN_REQUEST_LIMIT, TRAIN_RELEASE_LIMIT) if phase == "train"
            else (ASSESSMENT_REQUEST_LIMIT, ASSESSMENT_RELEASE_LIMIT))
        return min(NATIVE_BATCH_SIZE, request_cap - requests, release_cap - releases)

    def _requests(self, phase, start, count, train_order=None):
        result = []
        schedule_sha = self.schedule_sha256
        source_sha = self.bindings.fingerprint
        if phase == "train" and train_order is None:
            train_order = self._train_order()
        for index in range(start, start + count):
            if phase == "train":
                split, method, context = "train", "continued", train_order[index]
            else:
                split, offset, method = ASSESSMENT_COHORTS[index // NATIVE_BATCH_SIZE]
                contexts = {"train": self.train_context_ids, "validation": self.validation_context_ids,
                            "test": self.test_context_ids}[split]
                _require(contexts is not None, "Fresh held-out context preparation is missing")
                context = contexts[offset + index % NATIVE_BATCH_SIZE]
            result.append(Request(request_id=f"{phase}-{index:06d}", request_index=index,
                phase=phase, split=split, method=method, context_id=context,
                protocol_sha256=self.protocol_sha256, schedule_sha256=schedule_sha,
                source_binding_sha256=source_sha,
                final_checkpoint_sha256=self.final_checkpoint_sha256 if phase == "assessment" else None))
        return tuple(result)

    def _counts(self, phase):
        batches = tuple(b for b in self.batches if b.phase == phase)
        return (sum(len(b.requests) for b in batches), sum(sum(o.released for o in b.outcomes) for b in batches))

    @property
    def pending(self):
        return self.batches[-1] if self.batches and not self.batches[-1].complete else None

    def next_batch(self):
        """Plan the full permitted active batch, or None at the phase cap."""
        _require(self.stopped_reason is None, "Ledger has stopped and cannot resume")
        _require(self.pending is None, "Reserved requests still need terminal outcomes")
        count = self._capacity(self.phase, *self._counts(self.phase))
        if count == 0:
            return None
        return Batch(len(self.batches), self.phase,
                     self._requests(self.phase, self._counts(self.phase)[0], count))

    def reserve_batch(self, batch):
        """Charge the planned requests immediately, before any external work."""
        _require(type(batch) is Batch and not batch.outcomes, "Expected an unexecuted planned batch")
        expected = self.next_batch()
        _require(expected is not None and batch == expected, "Batch is stale, duplicated or out of order")
        return replace(self, batches=self.batches + (batch,))

    def record_outcomes(self, outcomes):
        """Append an ordered terminal prefix; unresolved requests stay charged."""
        _require(self.stopped_reason is None, "Ledger has stopped and cannot resume")
        _require(self.pending is not None, "No reserved request awaits an outcome")
        values = tuple(outcomes)
        _require(bool(values) and all(type(x) is Outcome for x in values), "Expected terminal outcomes")
        updated = replace(self.pending, outcomes=self.pending.outcomes + values)
        return replace(self, batches=self.batches[:-1] + (updated,))

    def record_batch(self, outcomes):
        """Complete the pending batch, optionally including explicit None padding.

        An external 35-slot result may append only None for inactive slots.
        Inactive padding is never a request, release or terminal sample.
        ``record_outcomes`` is available for preserving a partial terminal prefix.
        """
        _require(self.pending is not None, "No reserved batch awaits outcomes")
        remaining = len(self.pending.requests) - len(self.pending.outcomes)
        values = tuple(outcomes)
        if len(values) == NATIVE_BATCH_SIZE and remaining < NATIVE_BATCH_SIZE:
            _require(all(x is None for x in values[remaining:]), "Inactive padding must have no outcome")
            values = values[:remaining]
        _require(len(values) == remaining, "Batch outcomes must match every remaining active request")
        return self.record_outcomes(values)

    def begin_assessment(self, final_checkpoint_sha256):
        """Freeze one final artifact after the cap stop, before any final outcome."""
        _require(self.stopped_reason is None, "Ledger has stopped and cannot resume")
        _require(self.phase == "train", "Assessment already opened; final checkpoint cannot change")
        _require(self.pending is None, "Reserved TRAIN requests still need terminal outcomes")
        _require(self._train_limit_reached(*self._counts("train")), "TRAIN stopping rule has not been reached")
        return replace(self, phase="assessment", final_checkpoint_sha256=final_checkpoint_sha256)

    def halt(self, reason):
        """Irreversibly stop this history, preserving every consumed reservation."""
        _require(self.stopped_reason is None, "Ledger has already stopped")
        _text(reason, "stop reason")
        return replace(self, stopped_reason=reason)

    def summary(self):
        result = {}
        for phase in ("train", "assessment"):
            batches = tuple(b for b in self.batches if b.phase == phase)
            requested, releases = self._counts(phase)
            outcomes = tuple(o for b in batches for o in b.outcomes)
            request_cap, release_cap = ((TRAIN_REQUEST_LIMIT, TRAIN_RELEASE_LIMIT) if phase == "train"
                else (ASSESSMENT_REQUEST_LIMIT, ASSESSMENT_RELEASE_LIMIT))
            contexts = (self.train_context_ids if phase == "train" else self.train_context_ids
                        + (self.validation_context_ids or ()) + (self.test_context_ids or ()))
            exposure = dict.fromkeys(contexts, 0)
            for batch in batches:
                for request in batch.requests:
                    key = request.context_id
                    exposure[key] = exposure.get(key, 0) + 1
            result[phase] = dict(request_limit=request_cap, release_limit=release_cap,
                requested=requested, terminal=len(outcomes), unresolved=requested - len(outcomes),
                released=releases, declined=sum(not o.released for o in outcomes),
                remaining_requests=request_cap - requested, remaining_releases=release_cap - releases,
                request_limit_reached=requested == request_cap, release_limit_reached=releases == release_cap,
                request_cap_release_target_shortfall=(release_cap - releases
                    if requested == request_cap and len(outcomes) == requested else None),
                context_exposure=exposure)
        result["total_new_task_requests"] = sum(result[p]["requested"] for p in ("train", "assessment"))
        result["total_new_task_releases"] = sum(result[p]["released"] for p in ("train", "assessment"))
        result["assessment_complete"] = (result["assessment"]["requested"] == ASSESSMENT_REQUEST_LIMIT
            and result["assessment"]["unresolved"] == 0 and self.stopped_reason is None)
        return result

    def to_dict(self):
        """Return a detached, fingerprinted receipt with explicit missing evidence."""
        value = asdict(self)
        # Use JSON's list representation consistently for persisted round trips.
        value = json.loads(json.dumps(value, allow_nan=False))
        value.update(schedule_sha256=self.schedule_sha256, source_binding_sha256=self.bindings.fingerprint,
            source_bindings_complete=self.bindings.complete, execution_authorized=False,
            release_authorized=False, counters=self.summary())
        value["fingerprint"] = canonical_hash(value)
        return value

    @classmethod
    def from_dict(cls, receipt):
        """Verify a receipt and replay every reservation; no files are read."""
        _require(type(receipt) is dict, "Expected ledger receipt object")
        value = dict(receipt)
        fingerprint = value.pop("fingerprint", None)
        _require(type(fingerprint) is str and canonical_hash(value) == fingerprint,
                 "Ledger receipt fingerprint changed")
        fields = ("train_context_ids", "validation_context_ids", "test_context_ids", "bindings",
                  "batches", "phase", "final_checkpoint_sha256", "stopped_reason", "schema", "protocol_sha256")
        expected_keys = set(fields) | {"schedule_sha256", "source_binding_sha256", "source_bindings_complete",
                                     "execution_authorized", "release_authorized", "counters"}
        _require(set(value) == expected_keys, "Ledger receipt schema fields changed")
        try:
            for name in ("train_context_ids", "validation_context_ids", "test_context_ids"):
                _require(type(value[name]) is list or (name != "train_context_ids" and value[name] is None),
                         "Expected serialized context ID lists")
            _require(type(value["batches"]) is list, "Expected serialized batch list")
            batches = []
            for batch in value["batches"]:
                _require(type(batch) is dict and set(batch) == {"batch_index", "phase", "requests", "outcomes"},
                         "Batch receipt schema changed")
                _require(type(batch["requests"]) is list and type(batch["outcomes"]) is list,
                         "Expected serialized request/outcome lists")
                batches.append(Batch(batch["batch_index"], batch["phase"],
                    tuple(Request(**r) for r in batch["requests"]), tuple(Outcome(**o) for o in batch["outcomes"])))
            result = cls(**{name: value[name] for name in ("phase", "final_checkpoint_sha256", "stopped_reason",
                "schema", "protocol_sha256")}, train_context_ids=tuple(value["train_context_ids"]),
                validation_context_ids=(None if value["validation_context_ids"] is None else tuple(value["validation_context_ids"])),
                test_context_ids=(None if value["test_context_ids"] is None else tuple(value["test_context_ids"])),
                bindings=SourceBindings(**value["bindings"]), batches=tuple(batches))
        except (TypeError, KeyError) as error:
            raise ValueError("Invalid ledger receipt schema") from error
        _require(result.to_dict()["fingerprint"] == fingerprint, "Ledger derived counters or identity changed")
        return result
