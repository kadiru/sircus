"""Read-only structural audit of a bounded candidate-learning journal.

This independently replays reservations, outcome coverage and fixed learning
schedules. It does not replay optimizer arithmetic or native physics and cannot
certify a reported capture, reward or actual weight update from hashes alone.
An externally recorded journal head and bindings are required; a self-signed
journal is not evidence of execution permission.
"""

import math
import re

from candidate_continuation_contract import (AUDIT_SCHEMA, JOURNAL_SCHEMA,
    PROTOCOL_SHA256, IMITATION_EPOCHS, DEMONSTRATIONS, DEMO_BATCH,
    IMITATION_MAX_STEPS, PPO_BUFFER, PPO_EPOCHS, PPO_MAX_UPDATES, PPO_MAX_STEPS, PROBABILITY_LAW)
from candidate_continuation_ledger import Ledger, Outcome, canonical_hash, PARENT_CHECKPOINT_SHA256
from candidate_continuation_policy import FrozenCandidateSupport

COUNTERS = ("imitation_steps", "imitation_presentations", "ppo_updates", "ppo_steps", "ppo_samples")
ZERO = "0" * 64


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(value, label):
    require(type(value) is str and re.fullmatch(r"[0-9a-f]{64}", value), "Invalid " + label + " SHA256")


def finite(value, label):
    require(type(value) in (int, float) and math.isfinite(value), "Nonfinite or invalid " + label)


def counters(value):
    require(type(value) is dict and set(value) == set(COUNTERS)
        and all(type(n) is int and n >= 0 for n in value.values()), "Invalid learning counters")
    require(value["imitation_steps"] <= IMITATION_MAX_STEPS
        and value["imitation_presentations"] <= IMITATION_EPOCHS * DEMONSTRATIONS
        and value["ppo_updates"] <= PPO_MAX_UPDATES and value["ppo_steps"] <= PPO_MAX_STEPS
        and value["ppo_samples"] <= 1750, "Learning counter cap exceeded")
    return value


def finite_tree(value):
    if type(value) in (int, float):
        finite(value, "receipt metric")
    elif type(value) is dict:
        for child in value.values():
            finite_tree(child)
    elif type(value) in (list, tuple):
        for child in value:
            finite_tree(child)


def stopped_at_cap(ledger):
    summary = ledger.summary()["train"]
    return summary["request_limit_reached"] or summary["release_limit_reached"]


def _support_inputs(train_context_ids, supports, demonstrations):
    ids = tuple(train_context_ids)
    require(len(ids) == len(set(ids)) == 35 and all(type(x) is str and x for x in ids),
            "Expected exactly 35 distinct TRAIN context IDs")
    supports = tuple(FrozenCandidateSupport.from_dict(s) if type(s) is dict else s for s in supports)
    require(len(supports) == 35 and all(type(s) is FrozenCandidateSupport for s in supports)
            and tuple(s.context_id for s in supports) == ids, "TRAIN support/context order changed")
    lookup = {s.context_id:s for s in supports}
    require(type(demonstrations) in (list, tuple) and len(demonstrations) == DEMONSTRATIONS,
            "Expected exactly 1534 original successful demonstrations")
    keys, pairs, success_contexts = set(), set(), set()
    for row in demonstrations:
        context, index = row["context_id"], row["index"]
        require(context in lookup and type(index) is int and 0 <= index < len(lookup[context].candidate_ids),
                "Demonstration outside fixed TRAIN support")
        require(lookup[context].candidate_ids[index] == row["candidate_id"], "Demonstration candidate changed")
        key, pair = row["success_key"], (context, row["candidate_id"])
        require(type(key) is str and key and key not in keys and pair not in pairs,
                "Repeated demonstration witness or candidate")
        require(type(row["clean"]) is bool, "Invalid clean demonstration label")
        keys.add(key); pairs.add(pair); success_contexts.add(context)
    require(len(success_contexts) == 21, "Expected successful labels at exactly 21 TRAIN contexts")
    require(sum(row["clean"] for row in demonstrations) == 485,
            "Expected exactly 485 original clean demonstrations")
    return ids, supports, lookup


def interrupted_counters(committed, reported, phase, ledger, buffer):
    """Bound one unjournaled optimizer operation; never turn it into completion.

    Counters are charged immediately after an Adam step, before all later
    validation/receipt writes. A terminal exception may therefore expose part
    of that single operation. Missing optimizer receipts remain unverified.
    """
    counters(reported)
    delta = {key:reported[key]-committed[key] for key in COUNTERS}
    require(all(value >= 0 for value in delta.values()), "Failure counters moved backwards")
    if not any(delta.values()):
        return None
    if phase == "imitation":
        step = committed["imitation_steps"]
        minibatch = step % math.ceil(DEMONSTRATIONS/DEMO_BATCH)
        count = min(DEMO_BATCH, DEMONSTRATIONS-minibatch*DEMO_BATCH)
        require(step < IMITATION_MAX_STEPS and delta["imitation_steps"] == 1
            and delta["imitation_presentations"] in (0,count)
            and all(delta[key] == 0 for key in ("ppo_updates","ppo_steps","ppo_samples")),
            "Failure hides more than one interrupted imitation step")
    else:
        size = len(buffer)
        require(phase == "ppo" and ledger.pending is None and size > 0
            and (size == PPO_BUFFER or (size < PPO_BUFFER and stopped_at_cap(ledger)))
            and delta["imitation_steps"] == delta["imitation_presentations"] == 0
            and 1 <= delta["ppo_steps"] <= PPO_EPOCHS
            and delta["ppo_updates"] in (0,1) and delta["ppo_samples"] in (0,size)
            and (delta["ppo_updates"] == 0 or delta["ppo_steps"] == PPO_EPOCHS)
            and (delta["ppo_samples"] == 0 or delta["ppo_updates"] == 1),
            "Failure hides more than one interrupted PPO update or ineligible buffer")
    return dict(stage=phase, counter_deltas=delta,
        verification="Uncommitted interrupted operation; not a verified optimizer update and never resumable.")


def _chain(events, expected_head_sha256):
    sha(expected_head_sha256, "externally bound journal head")
    require(type(events) in (list, tuple) and bool(events), "Empty learning journal")
    previous = ZERO
    for seq, event in enumerate(events):
        require(type(event) is dict and set(event) ==
            {"schema", "seq", "previous_sha256", "kind", "payload", "fingerprint"}, "Journal event fields changed")
        require(event["schema"] == JOURNAL_SCHEMA and type(event["seq"]) is int and event["seq"] == seq
                and event["previous_sha256"] == previous, "Journal sequence or hash chain changed")
        unsigned = {k:v for k,v in event.items() if k != "fingerprint"}
        require(event["fingerprint"] == canonical_hash(unsigned), "Journal event fingerprint changed")
        require(type(event["payload"]) is dict, "Invalid journal payload")
        finite_tree(event["payload"])
        previous = event["fingerprint"]
    require(previous == expected_head_sha256, "Journal differs from externally bound head")


def audit_events(events, *, expected_bindings, expected_head_sha256, train_context_ids,
                 supports, demonstrations):
    """Validate a complete or explicitly failed TRAIN-only software history.

    ``passed`` means the journal is structurally consistent; it is not a native
    result or an independent numerical replay of the reported optimizer steps.
    A failed run is reported as ``incomplete`` with all reservations retained.
    """
    _chain(events, expected_head_sha256)
    ids, supports, support_by_id = _support_inputs(train_context_ids, supports, demonstrations)
    require(type(expected_bindings) is dict
        and expected_bindings.get("parent_checkpoint_sha256") == PARENT_CHECKPOINT_SHA256
        and expected_bindings.get("learning_protocol_sha256") == PROTOCOL_SHA256
        and expected_bindings.get("support_sha256") == canonical_hash([s.to_dict() for s in supports])
        and expected_bindings.get("demonstrations_sha256") == canonical_hash(demonstrations),
        "Expected source/parent/protocol/support/demonstration bindings changed")
    for key, value in expected_bindings.items():
        sha(value, key)
    sha(expected_bindings.get("parent_weight_hash"), "initial parent weights")
    start = events[0]
    require(start["kind"] == "start" and start["payload"].get("bindings") == expected_bindings
        and start["payload"].get("learning_protocol_sha256") == PROTOCOL_SHA256
        and type(start["payload"].get("synthetic")) is bool, "Missing bound run start")
    synthetic = start["payload"]["synthetic"]
    if not synthetic:
        require(expected_bindings["parent_weight_hash"] ==
            "ea9a69ec063e3d71cda2b92f1da64939f644244d4b4dbd70b500be132632e549",
            "Real run did not start from frozen seed-22 weights")
    ledger = Ledger.from_dict(start["payload"]["ledger"])
    require(ledger.train_context_ids == ids and not ledger.batches and ledger.phase == "train"
        and ledger.stopped_reason is None, "Run must begin with an empty TRAIN ledger")
    require(synthetic or ledger.bindings.complete, "Real run has incomplete source/guard/fresh-support bindings")
    current = dict.fromkeys(COUNTERS, 0)
    phase, final, weight, optimizer = "imitation", None, expected_bindings["parent_weight_hash"], None
    sampled, results, buffer, used, epoch_indices = {}, {}, [], set(), set()
    capture_count = clean_count = 0
    reported_counters, interrupted = None, None
    for event in events[1:]:
        kind, payload = event["kind"], event["payload"]
        require(final is None, "Journal continued after final or failure")
        if kind in ("imitation_update", "ppo_update"):
            require(payload.get("schema") == "sircus-candidate-continuation-optimizer-update-v1"
                and payload.get("protocol_sha256") == PROTOCOL_SHA256,
                "Optimizer receipt schema/protocol changed")
            expected_index = current["imitation_steps"] if kind == "imitation_update" else current["ppo_updates"]
            require(type(payload.get("index")) is int and payload["index"] == expected_index,
                    "Optimizer receipt index changed")
            require(payload["counters_before"] == current, "Update counters do not follow history")
            after = counters(payload["counters_after"])
            for field in ("weight_hash_before", "weight_hash_after", "actor_hash_before", "actor_hash_after",
                          "critic_hash_before", "critic_hash_after", "logstd_hash_before", "logstd_hash_after",
                          "optimizer_hash_before", "optimizer_hash_after", "rng_hash_before", "rng_hash_after"):
                sha(payload[field], field)
            require(weight is None or payload["weight_hash_before"] == weight, "Policy changed between recorded updates")
            require(optimizer is None or payload["optimizer_hash_before"] == optimizer,
                    "Optimizer changed between recorded updates")
            weight = payload["weight_hash_after"]
            optimizer = payload["optimizer_hash_after"]
        if kind == "imitation_update":
            require(phase == "imitation" and payload.get("stage") == "imitation", "Imitation update out of phase")
            step = current["imitation_steps"]
            epoch, minibatch = divmod(step, math.ceil(DEMONSTRATIONS / DEMO_BATCH))
            require(payload["epoch"] == epoch and payload["minibatch"] == minibatch,
                    "Imitation epoch/minibatch order changed")
            chosen = payload["demo_indices"]
            count = min(DEMO_BATCH, DEMONSTRATIONS - minibatch*DEMO_BATCH)
            require(type(chosen) is list and len(chosen) == count and len(set(chosen)) == count
                and all(type(i) is int and 0 <= i < DEMONSTRATIONS and i not in epoch_indices for i in chosen),
                "Imitation repeats or omits the scheduled demonstrations")
            epoch_indices.update(chosen)
            if minibatch == math.ceil(DEMONSTRATIONS / DEMO_BATCH)-1:
                require(len(epoch_indices) == DEMONSTRATIONS, "Incomplete imitation epoch")
                epoch_indices.clear()
            expected = dict(current, imitation_steps=step+1,
                imitation_presentations=current["imitation_presentations"]+count)
            require(after == expected and payload["critic_hash_before"] == payload["critic_hash_after"]
                and payload["logstd_hash_before"] == payload["logstd_hash_after"]
                and payload.get("frozen_critic_and_logstd") is True,
                "Imitation counters or frozen parameters changed")
            metrics = payload["metrics"]
            for key in ("total", "demonstration_nll", "forward_kl", "gradient_norm"):
                finite(metrics[key], key)
            require(metrics["gradient_norm"] >= 0, "Negative imitation gradient norm")
            current = dict(after)
        elif kind == "ppo_start":
            require(phase == "imitation" and current["imitation_steps"] == IMITATION_MAX_STEPS
                and current["imitation_presentations"] == IMITATION_EPOCHS*DEMONSTRATIONS
                and payload["counters"] == current, "PPO started before complete fixed imitation")
            phase = "ppo"
            optimizer = None
        elif kind == "reserve":
            require(phase == "ppo" and ledger.pending is None and len(buffer) < PPO_BUFFER,
                    "Reservation out of phase or before pending buffer update")
            batch = ledger.next_batch()
            require(batch is not None and payload["ledger"] == ledger.reserve_batch(batch).to_dict(),
                    "Reservation does not follow independent ledger schedule")
            require(payload["policy_weight_hash"] == weight, "Sampled policy differs from current recorded weights")
            choices = payload["choices"]
            require(len(choices) == len(batch.requests) and len(buffer)+len(choices) <= PPO_BUFFER,
                    "Selected choices omit requests or exceed PPO buffer")
            for request, choice in zip(batch.requests, choices):
                support = support_by_id[request.context_id]
                index = choice["index"]
                require(choice["request_id"] == request.request_id and request.request_id not in sampled
                    and choice["context_id"] == request.context_id and type(index) is int
                    and 0 <= index < len(support.candidate_ids), "Sampled request/index/context changed")
                require(choice["candidate_id"] == support.candidate_ids[index]
                    and choice["support_fingerprint"] == support.support_fingerprint
                    and choice["action"] == support.action(index)
                    and tuple(choice["latent"]) == support.latents[index], "Sampled immutable candidate changed")
                finite(choice["old_log_probability"], "old categorical probability")
                finite(choice["old_value"], "old critic value")
                require(choice["old_log_probability"] <= 1e-12, "Old categorical log probability is positive")
                sampled[request.request_id] = choice
            ledger = ledger.reserve_batch(batch)
        elif kind == "outcomes":
            require(phase == "ppo" and ledger.pending is not None, "Outcomes without prior reservation")
            rows = payload["results"]
            require(type(rows) is list and bool(rows), "Empty terminal outcome prefix")
            outcomes = []
            for row in rows:
                identifier = row["request_id"]
                require(identifier in sampled and identifier not in results, "Unreserved or repeated terminal outcome")
                finite(row["return"], "terminal return")
                require(type(row["released"]) is type(row["captured"]) is type(row["clean"]) is bool
                    and (not row["captured"] or row["released"]) and (not row["clean"] or row["captured"]),
                    "Capture/clean/release flags are inconsistent")
                require((not row["released"] and row["return"] == -1.)
                    or (row["captured"] and row["return"] == (1.6 if row["clean"] else 1.5))
                    or (row["released"] and not row["captured"] and -.8 <= row["return"] <= 0.),
                    "Terminal return violates the unchanged reward constants/bounds")
                if not synthetic:
                    sha(row["source_sha256"], "native terminal source")
                outcomes.append(Outcome(identifier, row["released"], row["decline_reason"], row["source_sha256"]))
                results[identifier] = row
                buffer.append(identifier)
                capture_count += int(row["captured"]); clean_count += int(row["clean"])
            ledger = ledger.record_outcomes(tuple(outcomes))
            require(payload["ledger"] == ledger.to_dict(), "Outcome ledger differs from independent replay")
        elif kind == "ppo_update":
            require(phase == "ppo" and ledger.pending is None and payload.get("stage") == "ppo",
                    "PPO update before every reserved terminal outcome")
            count = len(buffer)
            require(payload["request_ids"] == buffer and count > 0 and not (set(buffer)&used),
                    "PPO request coverage changed or request reused")
            is_final = payload["final"]
            require(type(is_final) is bool and count <= PPO_BUFFER
                and (count == PPO_BUFFER or (is_final and stopped_at_cap(ledger)))
                and (not is_final or stopped_at_cap(ledger)), "Premature partial PPO buffer")
            require(payload["request_count"] == count and len(payload["epochs"]) == PPO_EPOCHS,
                    "PPO sample/epoch count changed")
            for epoch, metric in enumerate(payload["epochs"]):
                require(type(metric["epoch"]) is int and metric["epoch"] == epoch,
                        "PPO optimizer epochs changed")
                for key in ("total", "actor_loss", "critic_loss", "entropy", "gradient_norm", "ratios_min", "ratios_max"):
                    finite(metric[key], key)
                require(metric["critic_loss"] >= 0 and metric["gradient_norm"] >= 0
                    and metric["entropy"] >= -1e-12 and 0 <= metric["ratios_min"] <= metric["ratios_max"],
                    "Invalid PPO objective metric")
            rollout = payload["rollout"]
            require(rollout.get("schema") == "sircus-candidate-continuation-terminal-rollout-v1"
                and rollout.get("probability_law") == PROBABILITY_LAW, "PPO categorical rollout law changed")
            expected = dict(context_ids=[sampled[k]["context_id"] for k in buffer],
                support_fingerprints=[sampled[k]["support_fingerprint"] for k in buffer],
                candidate_ids=[list(support_by_id[sampled[k]["context_id"]].candidate_ids) for k in buffer],
                selected_indices=[sampled[k]["index"] for k in buffer],
                old_selected_log_probabilities=[sampled[k]["old_log_probability"] for k in buffer],
                old_values=[sampled[k]["old_value"] for k in buffer],
                terminal_returns=[results[k]["return"] for k in buffer])
            require(all(rollout.get(key) == value for key,value in expected.items()),
                    "PPO rollout differs from sampled choices or terminal returns")
            advantages = [r-v for r,v in zip(expected["terminal_returns"], expected["old_values"])]
            if count >= 2:
                mean = math.fsum(advantages)/count
                deviation = math.sqrt(math.fsum((x-mean)**2 for x in advantages)/count)
                advantages = [(x-mean)/(deviation+1e-6) for x in advantages]
            actual_advantages = rollout.get("advantages")
            require(type(actual_advantages) is list and len(actual_advantages) == count
                and all(type(a) in (int,float) and math.isclose(a,b,rel_tol=1e-9,abs_tol=1e-10)
                        for a,b in zip(actual_advantages, advantages)), "PPO detached advantages changed")
            expected_counts = dict(current, ppo_updates=current["ppo_updates"]+1,
                ppo_steps=current["ppo_steps"]+PPO_EPOCHS, ppo_samples=current["ppo_samples"]+count)
            require(after == expected_counts, "PPO counters changed")
            current = dict(after); used.update(buffer); buffer.clear()
        elif kind == "final":
            sha(payload["checkpoint_sha256"], "final checkpoint")
            require(phase == "ppo" and stopped_at_cap(ledger) and ledger.pending is None and not buffer
                and set(results) == used == set(sampled) and payload["ledger"] == ledger.to_dict()
                and payload["counters"] == current and payload["weight_hash"] == weight,
                "Final artifact precedes caps, complete outcomes or complete PPO coverage")
            final = "complete"
        elif kind == "failure":
            require(type(payload["reason"]) is str and bool(payload["reason"]), "Missing stop reason")
            require(payload["ledger"] == ledger.halt(payload["reason"]).to_dict(),
                    "Failure refunded or changed consumed requests")
            interrupted = interrupted_counters(current,payload["counters"],phase,ledger,buffer)
            reported_counters = dict(payload["counters"])
            ledger = ledger.halt(payload["reason"]); final = "incomplete"
        else:
            raise ValueError("Unsupported learning event or held-out phase: " + str(kind))
    require(final is not None, "Journal lacks a terminal final/failure record")
    report = dict(schema=AUDIT_SCHEMA, status="passed" if final == "complete" else "incomplete",
        journal_head_sha256=expected_head_sha256, bindings=dict(expected_bindings), synthetic=synthetic,
        counters=reported_counters or current, committed_counters=current,
        interrupted_counters=reported_counters if interrupted is not None else None,
        interrupted_update=interrupted,
        accounting=ledger.summary(), reported_train_captures=capture_count,
        reported_train_clean=clean_count, terminal_requests=len(results), ppo_covered_requests=len(used),
        unresolved_requests=ledger.summary()["train"]["unresolved"],
        scope="Independent structural TRAIN learning/accounting audit; no optimizer or native physics replay.",
        limitations=["Reported weight hashes are checked for continuity, not independently recomputed here.",
            "Reported rewards and captures require separate native scorer/guard/evidence audit.",
            "Observation bytes are externally bound by caller; this function receives no observation file.",
            "No validation/TEST assessment is accepted or certified by this audit."],
        execution_authorized=False, release_authorized=False)
    report["fingerprint"] = canonical_hash(report)
    return report
