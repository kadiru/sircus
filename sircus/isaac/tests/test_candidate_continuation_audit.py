"""Fabricated journal audit tests: no optimizer, checkpoint or native execution."""

import ast
import copy
from functools import lru_cache
import math
from pathlib import Path
import sys
import unittest

import numpy as np

SOURCE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOURCE))
import candidate_continuation_audit as audit
import candidate_continuation_contract as contract
from candidate_continuation_ledger import Ledger, Outcome, canonical_hash, PARENT_CHECKPOINT_SHA256
from candidate_continuation_policy import ACTION_NAMES, LOW, HIGH, freeze_support

TRAIN = tuple(f"train-{i:03d}" for i in range(35))
H = "a"*64


@lru_cache(maxsize=1)
def inputs():
    supports, demos = [], []
    for cell, context in enumerate(TRAIN):
        count = (74 if cell == 0 else 73) if cell < 21 else 1
        rows = []
        for index in range(count):
            latent = [index/100, 0, 0, 0, 0]
            physical = np.asarray(LOW)+(np.tanh(latent)+1)*(np.asarray(HIGH)-LOW)/2
            action = dict(zip(ACTION_NAMES, physical.tolist()))
            rows.append(dict(candidate_id=f"{context}-candidate-{index}", action=action))
            if cell < 21:
                demos.append(dict(context_id=context, candidate_id=rows[-1]["candidate_id"], index=index,
                    success_key=f"historical-{len(demos)}", clean=len(demos)<485))
        supports.append(freeze_support(context, rows, rows[0]["action"], [0.]*5))
    bindings = dict(parent_checkpoint_sha256=PARENT_CHECKPOINT_SHA256,
        learning_protocol_sha256=contract.PROTOCOL_SHA256,
        support_sha256=canonical_hash([s.to_dict() for s in supports]), input_sha256="b"*64,
        demonstrations_sha256=canonical_hash(demos),
        parent_weight_hash=H,
        source_sha256="c"*64)
    return tuple(supports), demos, bindings


def append(events, kind, payload):
    event = dict(schema=contract.JOURNAL_SCHEMA, seq=len(events),
        previous_sha256=events[-1]["fingerprint"] if events else audit.ZERO,
        kind=kind, payload=payload)
    event["fingerprint"] = canonical_hash(event)
    events.append(event)


def rechain(events):
    result = []
    for event in events:
        append(result, event["kind"], event["payload"])
    return result


def hashes():
    return {name:H for name in ("weight_hash_before", "weight_hash_after", "actor_hash_before", "actor_hash_after",
        "critic_hash_before", "critic_hash_after", "logstd_hash_before", "logstd_hash_after",
        "optimizer_hash_before", "optimizer_hash_after", "rng_hash_before", "rng_hash_after")}


def start_and_imitation():
    _, _, bindings = inputs()
    ledger = Ledger.new(TRAIN)
    events = []
    current = dict.fromkeys(audit.COUNTERS, 0)
    append(events, "start", dict(bindings=bindings, learning_protocol_sha256=contract.PROTOCOL_SHA256,
        synthetic=True, ledger=ledger.to_dict()))
    for epoch in range(20):
        for minibatch, start in enumerate(range(0, 1534, 128)):
            chosen = list(range(start, min(start+128, 1534)))
            after = dict(current, imitation_steps=current["imitation_steps"]+1,
                imitation_presentations=current["imitation_presentations"]+len(chosen))
            append(events, "imitation_update", dict(schema="sircus-candidate-continuation-optimizer-update-v1",
                protocol_sha256=contract.PROTOCOL_SHA256, index=current["imitation_steps"],
                stage="imitation", epoch=epoch, minibatch=minibatch,
                demo_indices=chosen, counters_before=current, counters_after=after,
                frozen_critic_and_logstd=True,
                metrics=dict(total=1., demonstration_nll=1., forward_kl=0., gradient_norm=1.), **hashes()))
            current = after
    append(events, "ppo_start", dict(counters=current))
    return events, ledger, current


def add_reservation(events, ledger):
    supports, _, _ = inputs()
    lookup = {s.context_id:s for s in supports}
    batch = ledger.next_batch()
    ledger = ledger.reserve_batch(batch)
    choices = []
    for request in batch.requests:
        support = lookup[request.context_id]
        choices.append(dict(request_id=request.request_id, context_id=request.context_id,
            candidate_id=support.candidate_ids[0], index=0, support_fingerprint=support.support_fingerprint,
            action=support.action(0), latent=list(support.latents[0]), old_log_probability=0., old_value=.1))
    append(events, "reserve", dict(ledger=ledger.to_dict(), choices=choices, policy_weight_hash=H))
    return ledger, batch, choices


@lru_cache(maxsize=2)
def completed(released=True):
    events, ledger, current = start_and_imitation()
    supports, _, _ = inputs()
    lookup = {s.context_id:s for s in supports}
    buffer, buffered_choices, buffered_returns = [], [], []
    while ledger.next_batch() is not None:
        ledger, batch, choices = add_reservation(events, ledger)
        rows = [dict(request_id=r.request_id, released=released,
            decline_reason=None if released else "guard_declined", source_sha256=None,
            **{"return": 1.5 if released else -1.}, captured=released, clean=False) for r in batch.requests]
        ledger = ledger.record_batch(tuple(Outcome(r["request_id"], released, r["decline_reason"]) for r in rows))
        append(events, "outcomes", dict(ledger=ledger.to_dict(), results=rows))
        buffer.extend(r.request_id for r in batch.requests)
        buffered_choices.extend(choices); buffered_returns.extend(r["return"] for r in rows)
        if len(buffer) == 140 or ledger.next_batch() is None:
            count = len(buffer)
            advantages = [r-.1 for r in buffered_returns]
            if count >= 2:
                mean = math.fsum(advantages)/count
                std = math.sqrt(math.fsum((x-mean)**2 for x in advantages)/count)
                advantages = [(x-mean)/(std+1e-6) for x in advantages]
            rollout = dict(schema="sircus-candidate-continuation-terminal-rollout-v1",
                probability_law=contract.PROBABILITY_LAW,
                context_ids=[c["context_id"] for c in buffered_choices],
                support_fingerprints=[c["support_fingerprint"] for c in buffered_choices],
                candidate_ids=[list(lookup[c["context_id"]].candidate_ids) for c in buffered_choices],
                selected_indices=[0]*count, old_selected_log_probabilities=[0.]*count,
                old_values=[.1]*count, terminal_returns=buffered_returns, advantages=advantages)
            after = dict(current, ppo_updates=current["ppo_updates"]+1,
                ppo_steps=current["ppo_steps"]+4, ppo_samples=current["ppo_samples"]+count)
            append(events, "ppo_update", dict(schema="sircus-candidate-continuation-optimizer-update-v1",
                protocol_sha256=contract.PROTOCOL_SHA256, index=current["ppo_updates"],
                stage="ppo", counters_before=current, counters_after=after,
                request_ids=buffer, request_count=count, final=ledger.next_batch() is None,
                epochs=[dict(epoch=i,total=1., actor_loss=1., critic_loss=0., entropy=0., gradient_norm=0.,
                    ratios_min=1.,ratios_max=1.) for i in range(4)], rollout=rollout, **hashes()))
            current=after; buffer=[]; buffered_choices=[]; buffered_returns=[]
    append(events, "final", dict(checkpoint_sha256="f"*64, ledger=ledger.to_dict(), counters=current, weight_hash=H))
    return events


def run(events, **overrides):
    supports, demos, bindings = inputs()
    values = dict(expected_bindings=bindings, expected_head_sha256=events[-1]["fingerprint"],
        train_context_ids=TRAIN, supports=supports, demonstrations=demos)
    values.update(overrides)
    return audit.audit_events(events, **values)


def failed_prefix(events, reported):
    events=copy.deepcopy(events)
    ledger=Ledger.from_dict(next(e["payload"]["ledger"] for e in reversed(events) if "ledger" in e["payload"]))
    append(events,"failure",dict(ledger=ledger.halt("interrupted_update").to_dict(),
        reason="interrupted_update",counters=reported))
    return events


class ContinuationAuditTests(unittest.TestCase):
    def test_complete_release_cap_history_and_scope(self):
        result = run(completed(True))
        self.assertEqual(result["status"], "passed")
        self.assertEqual(result["counters"], dict(imitation_steps=240, imitation_presentations=30680,
            ppo_updates=10, ppo_steps=40, ppo_samples=1400))
        self.assertEqual(result["ppo_covered_requests"], 1400)
        self.assertIn("no optimizer or native physics replay", result["scope"])
        self.assertFalse(result["release_authorized"])

    def test_request_cap_includes_every_decline_and_final_partial_buffer(self):
        result = run(completed(False))
        self.assertEqual((result["terminal_requests"],result["ppo_covered_requests"]), (1750,1750))
        self.assertEqual((result["counters"]["ppo_updates"],result["counters"]["ppo_steps"]), (13,52))
        self.assertEqual(result["accounting"]["train"]["released"],0)
        self.assertEqual(result["accounting"]["train"]["declined"],1750)
        self.assertEqual([e["payload"]["request_count"] for e in completed(False) if e["kind"]=="ppo_update"][-1],70)

    def test_external_head_hash_and_sequence_are_required(self):
        events = completed(True)
        with self.assertRaisesRegex(ValueError,"externally bound head"):
            run(events,expected_head_sha256="d"*64)
        changed = copy.deepcopy(events)
        changed[1]["payload"]["metrics"]["total"] = 2.
        with self.assertRaisesRegex(ValueError,"fingerprint"):
            run(changed)
        changed[1]["seq"] = True
        with self.assertRaisesRegex(ValueError,"sequence"):
            run(changed)

    def test_no_early_ppo_or_repeated_imitation_demonstrations(self):
        events, ledger, counts = start_and_imitation()
        with self.assertRaisesRegex(ValueError,"before complete"):
            run(rechain([events[0],events[-1]]))
        changed = copy.deepcopy(events)
        changed[2]["payload"]["demo_indices"][0] = 0
        with self.assertRaisesRegex(ValueError,"repeats or omits"):
            run(rechain(changed))

    def test_imitation_frozen_parameters_optimizer_and_counter_continuity(self):
        events,_,_=start_and_imitation()
        for key in ("critic_hash_after","logstd_hash_after","optimizer_hash_before"):
            changed=copy.deepcopy(events)
            changed[2]["payload"][key]="d"*64
            with self.subTest(key=key),self.assertRaisesRegex(ValueError,"frozen parameters|Optimizer changed"):
                run(rechain(changed))
        changed=copy.deepcopy(events);changed[1]["payload"]["counters_after"]["ppo_steps"]=1
        with self.assertRaisesRegex(ValueError,"counters or frozen"):
            run(rechain(changed))

    def test_changed_action_identity_and_reservation_are_rejected(self):
        events,ledger,_=start_and_imitation();add_reservation(events,ledger)
        changed=copy.deepcopy(events);changed[-1]["payload"]["choices"][0]["latent"][0] += .1
        with self.assertRaisesRegex(ValueError,"immutable candidate"):
            run(rechain(changed))
        changed=copy.deepcopy(events);changed[-1]["payload"]["choices"].pop()
        with self.assertRaisesRegex(ValueError,"omit requests"):
            run(rechain(changed))
        changed=copy.deepcopy(events);changed[-1]["payload"]["policy_weight_hash"]="e"*64
        with self.assertRaisesRegex(ValueError,"Sampled policy"):
            run(rechain(changed))

    def test_failed_reserved_batch_keeps_unresolved_requests(self):
        events,ledger,counts=start_and_imitation()
        ledger,_,_=add_reservation(events,ledger)
        append(events,"failure",dict(ledger=ledger.halt("producer_failed").to_dict(),reason="producer_failed",counters=counts))
        result=run(events)
        self.assertEqual(result["status"],"incomplete")
        self.assertEqual((result["unresolved_requests"],result["terminal_requests"]),(35,0))
        self.assertEqual(result["accounting"]["train"]["requested"],35)

    def test_failed_prefix_never_refunds_consumed_reservations(self):
        events,ledger,counts=start_and_imitation()
        ledger,_,_=add_reservation(events,ledger)
        append(events,"failure",dict(ledger=Ledger.new(TRAIN).halt("producer_failed").to_dict(),reason="producer_failed",counters=counts))
        with self.assertRaisesRegex(ValueError,"refunded"):
            run(events)

    def test_dropped_request_changed_rollout_or_advantage_is_rejected(self):
        base=completed(True)
        position=next(i for i,e in enumerate(base) if e["kind"]=="ppo_update")
        mutations=[lambda p:p["request_ids"].pop(),
            lambda p:p["rollout"]["old_values"].__setitem__(0,.2),
            lambda p:p["rollout"]["terminal_returns"].__setitem__(0,-.5),
            lambda p:p["rollout"]["advantages"].__setitem__(0,1.)]
        for mutate in mutations:
            events=copy.deepcopy(base);mutate(events[position]["payload"])
            with self.subTest(mutate=mutate),self.assertRaisesRegex(ValueError,"coverage|rollout differs|advantages"):
                run(rechain(events))

    def test_premature_final_or_partial_update_is_rejected(self):
        events,ledger,counts=start_and_imitation()
        append(events,"final",dict(checkpoint_sha256=H,ledger=ledger.to_dict(),counters=counts,weight_hash=H))
        with self.assertRaisesRegex(ValueError,"Final artifact precedes"):
            run(events)
        base=copy.deepcopy(completed(True));i=next(i for i,e in enumerate(base) if e["kind"]=="ppo_update")
        base[i]["payload"]["final"]=True
        with self.assertRaisesRegex(ValueError,"Premature partial"):
            run(rechain(base))

    def test_capture_flags_held_out_event_and_nonfinite_metrics_fail(self):
        base=completed(False);i=next(i for i,e in enumerate(base) if e["kind"]=="outcomes")
        events=copy.deepcopy(base);events[i]["payload"]["results"][0]["captured"]=True
        with self.assertRaisesRegex(ValueError,"flags are inconsistent"):
            run(rechain(events))
        events=copy.deepcopy(base);events[1]["kind"]="validation"
        with self.assertRaisesRegex(ValueError,"held-out phase"):
            run(rechain(events))
        events=copy.deepcopy(base);events[1]["payload"]["metrics"]["total"]=float("inf")
        with self.assertRaises(ValueError):
            run(events)

    def test_reward_constants_are_fixed_for_rejection_and_capture(self):
        for released in (True,False):
            base=completed(released);i=next(i for i,e in enumerate(base) if e["kind"]=="outcomes")
            events=copy.deepcopy(base);events[i]["payload"]["results"][0]["return"] = .5 if released else -.5
            with self.subTest(released=released),self.assertRaisesRegex(ValueError,"unchanged reward"):
                run(rechain(events))
        events=copy.deepcopy(completed(True));i=next(i for i,e in enumerate(events) if e["kind"]=="outcomes")
        events[i]["payload"]["results"][0]["clean"]=True
        with self.assertRaisesRegex(ValueError,"unchanged reward"):
            run(rechain(events))

    def test_source_bindings_context_order_and_demo_identity_are_external(self):
        events=completed(True);supports,demos,bindings=inputs()
        changed=dict(bindings,input_sha256="f"*64)
        with self.assertRaisesRegex(ValueError,"bound run start"):
            run(events,expected_bindings=changed)
        with self.assertRaisesRegex(ValueError,"context order"):
            run(events,train_context_ids=tuple(reversed(TRAIN)))
        changed=copy.deepcopy(demos);changed[-1]["success_key"]=changed[0]["success_key"]
        with self.assertRaisesRegex(ValueError,"Repeated demonstration"):
            run(events,demonstrations=changed)

    def test_real_run_cannot_use_synthetic_missing_source_placeholders(self):
        _,_,bindings=inputs()
        bindings=dict(bindings,parent_weight_hash="ea9a69ec063e3d71cda2b92f1da64939f644244d4b4dbd70b500be132632e549")
        ledger=Ledger.new(TRAIN);events=[]
        append(events,"start",dict(bindings=bindings,learning_protocol_sha256=contract.PROTOCOL_SHA256,
            synthetic=False,ledger=ledger.to_dict()))
        append(events,"failure",dict(ledger=ledger.halt("not_started").to_dict(),reason="not_started",
            counters=dict.fromkeys(audit.COUNTERS,0)))
        with self.assertRaisesRegex(ValueError,"incomplete source/guard"):
            run(events,expected_bindings=bindings)

    def test_interrupted_imitation_step_counters_are_preserved_but_never_pass(self):
        base,_,_=start_and_imitation()
        committed=dict.fromkeys(audit.COUNTERS,0)
        for presentations in (0,128):
            actual=dict(committed,imitation_steps=1,imitation_presentations=presentations)
            result=run(failed_prefix(base[:1],actual))
            self.assertEqual(result["status"],"incomplete")
            self.assertEqual(result["committed_counters"],committed)
            self.assertEqual(result["interrupted_counters"],actual)
            self.assertEqual(result["interrupted_update"]["stage"],"imitation")
        prefix=base[:12]
        committed=prefix[-1]["payload"]["counters_after"]
        actual=dict(committed,imitation_steps=12,imitation_presentations=1534)
        result=run(failed_prefix(prefix,actual))
        self.assertEqual(result["interrupted_update"]["counter_deltas"]["imitation_presentations"],126)

    def test_failure_cannot_hide_extra_imitation_steps_or_presentation_counts(self):
        base,_,_=start_and_imitation();zero=dict.fromkeys(audit.COUNTERS,0)
        for changes in (dict(imitation_steps=2,imitation_presentations=256),
                        dict(imitation_presentations=128),dict(imitation_steps=1,imitation_presentations=1),
                        dict(ppo_steps=1)):
            with self.subTest(changes=changes),self.assertRaisesRegex(ValueError,"interrupted imitation"):
                run(failed_prefix(base[:1],dict(zero,**changes)))
        with self.assertRaisesRegex(ValueError,"backwards"):
            run(failed_prefix(base[:2],zero))

    def test_interrupted_ppo_epochs_and_post_update_counters_are_incomplete(self):
        base=completed(True);i=next(i for i,e in enumerate(base) if e["kind"]=="ppo_update")
        prefix=base[:i];committed=base[i]["payload"]["counters_before"]
        for steps,updates,samples in ((1,0,0),(4,0,0),(4,1,0),(4,1,140)):
            actual=dict(committed,ppo_steps=steps,ppo_updates=updates,ppo_samples=samples)
            result=run(failed_prefix(prefix,actual))
            self.assertEqual(result["status"],"incomplete")
            self.assertEqual(result["committed_counters"],committed)
            self.assertEqual(result["interrupted_counters"],actual)
            self.assertEqual(result["ppo_covered_requests"],0)
            self.assertEqual(result["terminal_requests"],140)
        base=completed(False);i=max(i for i,e in enumerate(base) if e["kind"]=="ppo_update")
        actual=base[i]["payload"]["counters_after"]
        result=run(failed_prefix(base[:i],actual))
        self.assertEqual(result["status"],"incomplete")
        self.assertEqual(result["interrupted_update"]["counter_deltas"]["ppo_samples"],70)
        self.assertEqual(result["committed_counters"]["ppo_steps"],48)
        self.assertEqual(result["counters"]["ppo_steps"],52)

    def test_failure_cannot_hide_multiple_ppo_updates_or_ineligible_partial_buffer(self):
        base=completed(True);i=next(i for i,e in enumerate(base) if e["kind"]=="ppo_update")
        committed=base[i]["payload"]["counters_before"]
        for changes in (dict(ppo_steps=5),dict(ppo_steps=4,ppo_updates=2),
                        dict(ppo_steps=3,ppo_updates=1),dict(ppo_steps=1,ppo_samples=140),
                        dict(ppo_steps=1,imitation_steps=241)):
            actual=dict(committed,**changes)
            with self.subTest(changes=changes),self.assertRaisesRegex(ValueError,"interrupted PPO|cap exceeded"):
                run(failed_prefix(base[:i],actual))
        j=next(i for i,e in enumerate(base) if e["kind"]=="outcomes")
        actual=dict(committed,ppo_steps=1)
        with self.assertRaisesRegex(ValueError,"ineligible buffer"):
            run(failed_prefix(base[:j+1],actual))

    def test_clean_demonstration_count_is_original_and_exact(self):
        supports,demos,bindings=inputs()
        demos=copy.deepcopy(demos);demos[0]["clean"]=False
        bindings=dict(bindings,demonstrations_sha256=canonical_hash(demos))
        with self.assertRaisesRegex(ValueError,"485 original clean"):
            run(completed(True),demonstrations=demos,expected_bindings=bindings)

    def test_auditor_has_no_optimizer_update_engine_or_native_entry_point(self):
        tree=ast.parse((SOURCE/"candidate_continuation_audit.py").read_text())
        imports=[n.module for n in ast.walk(tree) if isinstance(n,ast.ImportFrom)]
        self.assertFalse(any(x and any(v in x for v in ("updates","native","checkpoint")) for x in imports))
        attrs=[n.func.attr for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)]
        self.assertFalse(set(attrs)&{"step","backward","save","shot_batch"})


if __name__=="__main__":
    unittest.main()
