"""Explicit runner orchestration; importing this module executes nothing.

The same orchestration drives an injected native bridge or clearly synthetic
test engine. Native startup/ownership is external and must be verified before
native operations. This module neither starts an app nor launches a process.
"""

import copy
import math
from pathlib import Path
import re
import time

from candidate_continuation_journal import write_new_json, _sync_directory
from location_comparison import digest, PARENT_SHA, require
from location_comparison_bounds import Bounds, ARMS

SCHEMA = "sircus-location-comparison-trace-v1"
PREPARED_FINGERPRINT = "440cbcca29612820d3b36005010edd90fd4f8172646bd8ea95e57f154775f4e6"
MODELS = ("parent", *ARMS)
STATE_FIELDS = ("model_id", "weight_hash", "optimizer_hash", "optimizer_steps",
                "updates", "attempts", "parent_checkpoint_sha256")
RNG_FIELDS = ("action_rng_hash", "optimizer_rng_hash")


def _state(value):
    require(type(value) is dict and all(key in value for key in STATE_FIELDS), "Incomplete model state")
    require(type(value["model_id"]) is str and value["model_id"], "Missing model identity")
    for key in ("weight_hash", "optimizer_hash", "parent_checkpoint_sha256"):
        require(type(value[key]) is str and re.fullmatch(r"[a-f0-9]{64}", value[key]), "Invalid model digest")
    for key in ("optimizer_steps", "updates", "attempts"):
        require(type(value[key]) is int and value[key] >= 0, "Invalid model counter")
    for key in RNG_FIELDS:
        if key in value:
            require(type(value[key]) is str and re.fullmatch(r"[a-f0-9]{64}",value[key]), "Invalid RNG identity")
    require(value["parent_checkpoint_sha256"] == PARENT_SHA, "Wrong seed22 checkpoint")
    return copy.deepcopy(value)


def _same_learning_state(a, b):
    return all(a[key] == b[key] for key in STATE_FIELDS)


def _same_rng(a, b, keys=RNG_FIELDS):
    return all((key in a) == (key in b) and a.get(key) == b.get(key)
               for key in keys)


def _finite(value):
    return type(value) in (int, float) and math.isfinite(value)


class Journal:
    """New-only durable events. A failed write permanently poisons the writer."""

    def __init__(self, directory=None, writer=write_new_json):
        self.directory = None if directory is None else Path(directory)
        if self.directory is not None:
            self.directory.mkdir(parents=False, exist_ok=False)
            _sync_directory(self.directory.parent)
        self.writer, self.events, self.failed = writer, [], False

    def append(self, kind, payload):
        require(not self.failed, "Journal write failed; no retry")
        event = dict(seq=len(self.events), kind=kind, payload=copy.deepcopy(payload),
                     previous=self.events[-1]["fingerprint"] if self.events else "0" * 64)
        event["fingerprint"] = digest(event)
        try:
            if self.directory is not None:
                self.writer(self.directory / f"event-{len(self.events):06d}.json", event)
        except BaseException:
            self.failed = True
            raise
        self.events.append(event)
        return event


class ComparisonRunner:
    """One use only. Failed requests remain charged; incomplete runs never TEST."""

    def __init__(self, plan, engine, bounds, journal, *, clock=time.monotonic,
                 execution_authorized=False, external_supervisor=None):
        self.plan, self.engine = copy.deepcopy(plan), engine
        self.bounds, self.journal, self.clock = bounds, journal, clock
        self.mode = "synthetic" if getattr(engine, "synthetic", None) is True else "native"
        self.external_supervisor = external_supervisor
        require(plan.get("fingerprint") == digest({k:v for k,v in plan.items() if k != "fingerprint"}),
                "Changed prepared comparison")
        require(bounds.fingerprint == plan["fingerprint"], "Bounds belong to another plan")
        if self.mode == "native":
            require(plan["fingerprint"] == PREPARED_FINGERPRINT,
                    "Production requires the exact prepared coordinate package")
            require(execution_authorized is True and callable(external_supervisor)
                    and journal.directory is not None,
                    "Native execution requires explicit authorization, durable journal and external ownership/watchdog")
        self.used, self.batch_index = False, 0
        self.checkpoints = {}
        self.counts = dict(train={arm:dict(requested=0, completed=0, captures=0,
            clean_captures=0, rejected=0, released_misses=0) for arm in ARMS},
            assessment=dict(requested=0, completed=0, captures=0, clean_captures=0,
                            rejected=0, released_misses=0))

    def _guard(self):
        self.bounds.check()
        if self.external_supervisor is not None:
            require(self.external_supervisor(self.bounds.snapshot()) is True,
                    "External process ownership or watchdog check failed")

    def _emit(self, kind, payload):
        return self.journal.append(kind, payload)

    def _checkpoint(self, name, label):
        self._guard()
        before = _state(self.engine.state(name))
        began = self.clock()
        receipt = self.engine.checkpoint(name, label)
        require(type(receipt) is dict and re.fullmatch(r"[0-9a-f]{64}", receipt.get("sha256", ""))
                and type(receipt.get("file")) is str and receipt["file"], "Invalid checkpoint receipt")
        after = _state(self.engine.state(name))
        require(_same_learning_state(before, _state(receipt["state"]))
                and _same_learning_state(before, after) and _same_rng(before,after)
                and _same_rng(before,receipt["state"]), "Checkpoint changed model state or randomness")
        self._emit("checkpoint", dict(model=name, label=label, receipt=receipt,
                                      checkpoint_wall_s=self.clock()-began))
        self._guard()
        return copy.deepcopy(receipt)

    def _cells(self, rows, phase, name, milestone):
        split = "train" if phase == "train" else "validation" if phase == "monitor" else "test"
        return [dict(id=row["id"], target_id=row["id"],
            request_id=f"{phase}:{name}:{milestone}:{row['id']}", split=split, cohort=phase,
            bucket_xy_m=copy.deepcopy(row["bucket_xy_m"]), physical_slot=index,
            **({"original_id":row["original_id"]} if "original_id" in row else {}))
            for index,row in enumerate(rows)]

    def _batch(self, name, rows, phase, milestone, *, film=False):
        self._guard()
        cells = self._cells(rows, phase, name, milestone)
        deterministic = phase != "train"
        require(1 <= len(cells) <= 35, "Invalid active native batch size")
        arm = name if self.bounds.phase != "shared" else None
        reservation = self.bounds.reserve("train" if phase == "train" else "assessment", len(cells), arm)
        counter = self.counts["train"][name] if phase == "train" else self.counts["assessment"]
        counter["requested"] += len(cells)
        batch_id = f"batch-{self.batch_index:05d}"
        self.batch_index += 1
        # Persist request IDs and all coordinates before proposing or dispatching.
        self._emit("reserve", dict(batch_id=batch_id, model=name, phase=phase,
            milestone=milestone, cells=cells, deterministic=deterministic, film=film,
            reservation_id=reservation["index"]))
        self._guard()  # publish the charged reservation before a possibly blocked proposal
        before = _state(self.engine.state(name))
        began = self.clock()
        packet = self.engine.propose(name, cells, deterministic)
        proposal_wall = self.clock() - began
        require(packet.get("model_id") == before["model_id"] and packet.get("deterministic") is deterministic,
                "Proposal model or deterministic mode changed")
        proposals = packet["rows"]
        require(len(proposals) == len(cells), "Proposal dropped a requested target")
        for cell, proposed in zip(cells, proposals):
            require(all(proposed.get(key) == cell[key] for key in ("request_id", "target_id")),
                    "Proposal target association changed")
            for key in ("observation", "latent", "actions_normalized"):
                require(type(proposed.get(key)) is list and len(proposed[key]) == 5
                        and all(_finite(x) for x in proposed[key]), "Malformed policy proposal")
            require(_finite(proposed.get("logp")), "Invalid proposal log probability")
        self._emit("proposal", dict(batch_id=batch_id, packet=packet))
        self._guard()
        began = self.clock()
        terminal = self.engine.execute(cells, packet, film=film)
        episode_wall = self.clock() - began
        outcomes = terminal["rows"]
        require(len(outcomes) == len(cells), "Native result omitted requested targets")
        for cell, row in zip(cells, outcomes):
            require(all(row.get(key) == cell[key] for key in ("request_id", "target_id")),
                    "Native target association changed")
            require(row.get("outcome") in ("capture", "miss", "rejected")
                    and type(row.get("clean")) is bool and _finite(row.get("reward")), "Invalid native outcome")
            require(not row["clean"] or row["outcome"] == "capture", "Only capture can be clean")
            require((type(row.get("rejection_reason")) is str and bool(row["rejection_reason"]))
                    if row["outcome"] == "rejected" else row.get("rejection_reason") is None,
                    "Outcome/rejection reason mismatch")
        after = _state(self.engine.state(name))
        require(_same_learning_state(before, after), "Proposing or executing changed learning state")
        require(not deterministic or _same_rng(before,after), "Assessment consumed training randomness")
        costs = dict(terminal.get("costs", {}), proposal_wall_s=proposal_wall,
                     episode_wall_s=episode_wall)
        require(all(_finite(costs[k]) and costs[k] >= 0 for k in ("proposal_wall_s", "episode_wall_s")),
                "Invalid measured call cost")
        self._emit("terminal", dict(batch_id=batch_id, rows=outcomes,
            evidence=terminal["evidence"], costs=costs, state_before=before, state_after=after))
        for row in outcomes:
            counter["completed"] += 1
            counter[{"capture":"captures", "miss":"released_misses", "rejected":"rejected"}[row["outcome"]]] += 1
            counter["clean_captures"] += int(row["clean"])
        # Durably recorded terminal evidence precedes completion. Even a late
        # completion is counted; a post-call deadline prevents the next request.
        self.bounds.complete(reservation)
        self._guard()
        return [dict(proposal=copy.deepcopy(p), outcome=copy.deepcopy(o))
                for p,o in zip(proposals, outcomes)]

    def _learn(self, name, examples):
        require(len(examples) == 140, "PPO update must consume exactly 140 fresh requests")
        self._guard()
        before = _state(self.engine.state(name))
        payload = copy.deepcopy(examples)
        sha = digest(payload)
        began = self.clock()
        after = _state(self.engine.learn(name, payload))
        require(digest(payload) == sha, "Learner mutated its raw proposal/reward records")
        require(after["model_id"] == before["model_id"]
                and after["attempts"] == before["attempts"] + 140
                and after["updates"] == before["updates"] + 1
                and after["optimizer_steps"] == before["optimizer_steps"] + 4,
                "PPO update changed group or accounting")
        require(after == self.engine.state(name), "Learner returned inconsistent state")
        require(_same_rng(before,after,("action_rng_hash",)), "Optimizer consumed action randomness")
        self._emit("update", dict(model=name,
            request_ids=[example["proposal"]["request_id"] for example in examples],
            examples_sha256=sha, before=before, after=after, update_wall_s=self.clock()-began))
        self._guard()

    def run(self):
        require(not self.used, "Runner is one-use; no resume or retry")
        self.used = True
        status, reason = "incomplete_no_retry", None
        try:
            self._guard()
            self._emit("start", dict(plan_fingerprint=self.plan["fingerprint"], mode=self.mode,
                seeds=self.plan["seeds"], bounds=self.bounds.snapshot()))
            restored = {}
            for name in MODELS:
                self._guard()
                began = self.clock()
                state = _state(self.engine.restore(name, self.plan["seeds"]["action_noise"],
                                                  self.plan["seeds"]["optimizer"]))
                require((state["attempts"], state["updates"], state["optimizer_steps"]) == (10000,73,292),
                        "Restored the wrong parent training state")
                if self.mode == "native":
                    require(all(type(state.get(key)) is str and re.fullmatch(r"[0-9a-f]{64}",state[key])
                                for key in RNG_FIELDS), "Missing native randomness identities")
                restored[name] = state
                self._emit("restore", dict(model=name, state=state, restore_wall_s=self.clock()-began))
            require(len({s["model_id"] for s in restored.values()}) == 3, "Models alias each other")
            require(len({(s["weight_hash"],s["optimizer_hash"]) for s in restored.values()}) == 1,
                    "Parent weights or Adam state differ between groups")
            require(all(_same_rng(restored["parent"],state) for state in restored.values()),
                    "Restored exploration or optimizer randomness is not paired")
            for name in MODELS:
                self.checkpoints[name] = self._checkpoint(name, "initial")
            self._batch("parent", self.plan["cohorts"]["monitor"], "monitor", 0)
            for name in ARMS:
                self.bounds.enter_phase(name)
                examples = []
                schedule = self.plan["schedules"][name]
                require(len(schedule) == 8960, "Wrong TRAIN schedule size")
                for begin in range(0, 8960, 35):
                    examples.extend(self._batch(name, schedule[begin:begin+35], "train", begin,
                        film=begin in (0,2240,4480,6720)))
                    completed = begin+35
                    if len(examples) == 140:
                        self._learn(name, examples)
                        examples.clear()
                    if completed in (2240,4480,6720,8960):
                        self._checkpoint(name, f"milestone-{completed:05d}")
                        self._batch(name, self.plan["cohorts"]["monitor"], "monitor", completed)
                require(not examples, "Uncommitted training buffer at final checkpoint")
                self.checkpoints[name] = self._checkpoint(name, "final")
            self.bounds.enter_phase("shared")
            require(_same_learning_state(restored["parent"], _state(self.engine.state("parent"))),
                    "Untouched parent changed during training")
            self._emit("frozen", dict(checkpoints=self.checkpoints))
            for phase in ("test_broad", "test_local"):
                targets = self.plan["cohorts"][phase]
                for rotation in range(3):
                    for model_index, name in enumerate(MODELS):
                        subset = [row for index,row in enumerate(targets) if (index+rotation) % 3 == model_index]
                        for begin in range(0, len(subset), 35):
                            self._batch(name, subset[begin:begin+35], phase, rotation,
                                film=rotation == 0 and model_index == 0 and begin == 0)
            require(self.bounds.assessment_completed == self.plan["protocol"]["total_assessment_requests"],
                    "Final assessment count mismatch")
            for name in MODELS:
                require(_same_learning_state(self.checkpoints[name]["state"], _state(self.engine.state(name))),
                        "Final TEST changed a frozen model")
            status = "complete_audit_pending"
        except Exception as error:
            reason = type(error).__name__ + ": " + str(error)
            self.bounds.fail(reason)
        finally:
            # Remove only the bridge's own instance hooks. The outer owner
            # retains responsibility for app/process shutdown and the lock.
            try:
                close = getattr(self.engine, "close", None)
                if callable(close):
                    close()
            except Exception as error:
                status = "incomplete_no_retry"
                reason = "Bridge cleanup failed: " + type(error).__name__ + ": " + str(error)
                self.bounds.fail(reason)
            self._emit("finish", dict(status=status, reason=reason,
                counts=self.counts, bounds=self.bounds.snapshot()))
        return dict(schema=SCHEMA, plan_fingerprint=self.plan["fingerprint"], mode=self.mode,
                    events=copy.deepcopy(self.journal.events))
