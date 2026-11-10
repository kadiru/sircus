"""Prepare a separate TEST-only evaluation of the closed V4 model.

No model is loaded, no inference/optimization occurs and no executable or
native launcher is imported. The old failed assessment is evidence, never a
resumable ledger. Preparation fixes its untouched TEST suffix under a new
identity. Only the separate synthetic mock runner is currently enabled.
"""

import copy
from dataclasses import replace
import re

from candidate_continuation_contract import JOURNAL_SCHEMA
from candidate_continuation_ledger import (
    ASSESSMENT_COHORTS, Ledger, Outcome, PARENT_CHECKPOINT_SHA256, canonical_hash,
)

SCHEMA = "sircus-v4-test-evaluation-plan-v1"
V4_CHECKPOINT_SHA256 = "15020a948d295fe5e7258a732145b89aa61fda2c385a0d6966a9e4cc7e81bd41"
V4_PRODUCER_SHA256 = "a8c6955f25377d5e404e710c5b685b2fc82725e3b1379b86f285fb31f0d358c2"
V4_PLAN_RAW_SHA256 = "88f0b350a9de5a025237d12cad8ad5da5ea40b47718aa8d4aa4a407ccb35b94f"
V4_PLAN_FINGERPRINT = "b04ca95e2e597cf2f3eaa5787e88f2903631a85e730e30e35f9c7fa77124d96e"
V4_JOURNAL_HEAD = "b44af2865339a28096db750df361b81434c90ef65e7ead3fc58618d0e11ae8a8"
V4_INVENTORY_SHA256 = "513fb5cdb235b2179544d01e5148923e11319a913d59703c0cf5709a66d1c32f"
V4_FAILURE_SHA256 = "e86dc36749c23e6024ca0cf3403192bc142c5d92b3abb8445497ab6e81844faa"
V4_COMPONENT_SHA256 = "f2c7339e3ba511a58f78e7de1851a91d2cc651df39bface8f3a43a87555f16bf"
TEST_COHORTS = tuple(ASSESSMENT_COHORTS[6:])
MIN_FREE_BYTES = 6 * 1024 ** 3


def require(value, message):
    if not value:
        raise ValueError(message)


def is_sha(value):
    return type(value) is str and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def protocol():
    return dict(schema="sircus-v4-test-evaluation-protocol-v1", requests=210,
        positions=70, methods=["parent", "candidate_bank", "continued"], batch_size=35,
        phases=dict(initialization=600, evaluation=1800, audit=600), wall_limit_s=3000,
        min_free_bytes=MIN_FREE_BYTES, native_starts_max=1, optimizer_steps=0,
        new_train_requests=0, new_validation_requests=0, repeats=0, retries=0,
        checkpoint_selection=False, inference_searches=0, model_promotion=False,
        original_assessment_completed=False, execution_authorized=False,
        release_authorized=False, native_execution_ready=False,
        scope="Separate untouched TEST-only evaluation; no old request is replaced, refunded, repeated or resumed.")


def _fingerprint(value, label):
    require(type(value) is dict and value.get("fingerprint") == canonical_hash(
        {k:v for k,v in value.items() if k != "fingerprint"}), label + " fingerprint changed")


def _original(plan):
    _fingerprint(plan, "Original pre-frozen plan")
    require(plan.get("schema") == "sircus-candidate-continuation-assessment-plan-v1"
        and plan.get("requests") == 420 and type(plan.get("optimizer_updates")) is int
        and plan["optimizer_updates"] == 0 and plan.get("held_out_outcomes_observed") == 0
        and plan.get("execution_authorized") is plan.get("release_authorized") is False,
        "Original fixed final-only assessment contract changed")
    require(plan.get("final_checkpoint_sha256") == V4_CHECKPOINT_SHA256
        and plan.get("parent_checkpoint_sha256") == PARENT_CHECKPOINT_SHA256,
        "Evaluation must reuse the saved V4 and original parent checkpoints")
    cells = plan["cells_by_split"]
    require(set(cells) == {"train", "validation", "test"}, "Original split set changed")
    ids = {}
    for split, size in (("train",35),("validation",35),("test",70)):
        rows = cells[split]
        require(type(rows) is list and len(rows) == size and all(
            type(c) is dict and c.get("split") == split and type(c.get("id")) is str for c in rows),
            "Original split coverage changed")
        ids[split] = [c["id"] for c in rows]
    require(len(set(sum(ids.values(), []))) == 140, "Repeated original position")
    require(type(plan.get("choices")) is list and len(plan["choices"]) == 420
        and type(plan.get("cohorts")) is list and len(plan["cohorts"]) == 12,
        "Original cohort coverage changed")
    for index, (split, offset, method) in enumerate(ASSESSMENT_COHORTS):
        choices = plan["choices"][index*35:index*35+35]
        c = plan["cohorts"][index]
        require(c == dict(index=index,split=split,offset=offset,method=method,
            request_ids=[x["request_id"] for x in choices],choice_sha256=canonical_hash(choices)),
            "Original cohort order or choice hash changed")
        for j, choice in enumerate(choices, index*35):
            require(choice.get("request_index") == j and choice.get("request_id") == f"assessment-{j:06d}"
                and choice.get("cohort") == index and choice.get("method") == method
                and choice.get("split") == split and choice.get("context_id") == ids[split][offset+j%35]
                and type(choice.get("enabled")) is bool, "Original frozen request association changed")
            if not choice["enabled"]:
                require(method != "parent" and choice.get("decline_reason") == "no_qualified_candidate_support"
                    and all(choice.get(k) is None for k in ("candidate_id","index","latent","scores","ik_seeds","support_fingerprint")),
                    "Original empty support was replaced")
            else:
                require(choice.get("decline_reason") is None, "Enabled choice has an empty-support decline")
    return ids


def _prior(plan, evidence):
    """Replay only exposure/accounting; never score or select from old outcomes."""
    events = evidence["events"]
    require(type(events) is list and [e.get("kind") for e in events] ==
        ["start","reserve","outcomes","reserve","outcomes","reserve","outcomes","reserve","failure"],
        "Prior assessment is not the preserved closed 140-request prefix")
    previous = "0"*64
    for seq,event in enumerate(events):
        _fingerprint(event,"Prior journal")
        require(event.get("schema") == JOURNAL_SCHEMA and event.get("seq") == seq
            and event.get("previous_sha256") == previous, "Prior journal chain changed")
        previous = event["fingerprint"]
    require(previous == evidence["journal_head_sha256"], "Prior externally bound journal head changed")
    start = events[0]["payload"]
    ledger = Ledger.from_dict(start["ledger"])
    require(ledger.phase == "assessment" and ledger.pending is None
        and ledger.to_dict()["fingerprint"] == plan["initial_assessment_ledger_fingerprint"]
        and start["plan_fingerprint"] == plan["fingerprint"]
        and start["final_checkpoint_sha256"] == plan["final_checkpoint_sha256"]
        and start["model_hashes"] == plan["model_hashes"], "Prior start did not freeze the same model and plan")
    for event in events[1:-1]:
        payload = event["payload"]
        if event["kind"] == "reserve":
            batch = ledger.next_batch()
            require(batch is not None, "Prior reservation after cap")
            ledger = ledger.reserve_batch(batch)
            begin = batch.requests[0].request_index
            require(begin < 140 and payload["choices"] == plan["choices"][begin:begin+35]
                and all(x["split"] != "test" for x in payload["choices"]),
                "Previously charged TEST or changed frozen choices")
        else:
            results = payload["results"]
            require(type(results) is list and len(results) == 35, "Prior terminal cohort incomplete")
            ledger = ledger.record_outcomes(tuple(Outcome(r["request_id"],r["released"],
                r["decline_reason"],r["source_sha256"]) for r in results))
        require(payload["ledger"] == ledger.to_dict(), "Prior journal accounting changed")
    failed = Ledger.from_dict(events[-1]["payload"]["ledger"])
    require(failed.stopped_reason and replace(ledger,stopped_reason=failed.stopped_reason).to_dict() == failed.to_dict(),
        "Prior failure rewrote accounting")
    counts = failed.summary()["assessment"]
    require(all(counts[k] == n for k,n in dict(requested=140,terminal=105,unresolved=35,released=96,declined=9).items())
        and all(counts["context_exposure"][c["id"]] == 0 for c in plan["cells_by_split"]["test"]),
        "Prior TEST exposure or partial counts changed")
    expected_files = {"assessment-plan.json",*(f"journal/event-{i:06d}.json" for i in range(9))}
    for i in range(4):
        stem=f"native/assessment-{i:06d}"
        expected_files.update(stem+s for s in ((".dispatch.json",".json.gz",".receipt.json",".npz")
            if i < 3 else (".dispatch.json",".failure.json",".npz")))
    inventory = evidence["inventory"]
    require(type(inventory) is dict and set(inventory) == expected_files
        and all(type(v) is dict and set(v)=={"sha256","bytes"} and is_sha(v["sha256"])
            and type(v["bytes"]) is int and v["bytes"] > 0 for v in inventory.values()),
        "Prior evidence inventory has missing, repeated, partial or TEST files")
    require(inventory["assessment-plan.json"]["sha256"] == evidence["original_plan_sha256"]
        and canonical_hash(inventory) == evidence["inventory_sha256"], "Prior inventory bytes changed")
    return dict(requested=140,terminal=105,unresolved=35,recorded_releases=96,recorded_declines=9,
        uncommitted_physical_work=True,test_requests=0,status="incomplete_no_retry")


def _choices(original):
    cohorts, flat = [], []
    for index,(split,offset,method) in enumerate(TEST_COHORTS):
        local=[]
        for source in original["choices"][(index+6)*35:(index+7)*35]:
            choice=copy.deepcopy(source)
            choice.update(original_request_id=source["request_id"],original_request_index=source["request_index"],
                original_cohort=source["cohort"],request_id=f"test-evaluation-{len(flat):06d}",
                request_index=len(flat),cohort=index)
            local.append(choice);flat.append(choice)
        cohorts.append(dict(index=index,original_cohort=index+6,split=split,offset=offset,method=method,choices=local))
    return cohorts,flat


def prepare_evaluation(original_plan, prior_evidence, *, evaluation_source_sha256, synthetic):
    """Pure preparation. Real inputs can be bound, but native execution stays off."""
    require(type(synthetic) is bool and is_sha(evaluation_source_sha256), "Explicit preparation provenance required")
    original=copy.deepcopy(original_plan);evidence=copy.deepcopy(prior_evidence)
    _original(original);partial=_prior(original,evidence)
    for key in ("original_plan_sha256","inventory_sha256","journal_head_sha256","failure_sha256","component_summary_sha256"):
        require(is_sha(evidence.get(key)), "Missing closed evidence identity: "+key)
    if not synthetic:
        require(evidence["original_plan_sha256"] == V4_PLAN_RAW_SHA256
            and original["fingerprint"] == V4_PLAN_FINGERPRINT
            and evidence["inventory_sha256"] == V4_INVENTORY_SHA256
            and evidence["journal_head_sha256"] == V4_JOURNAL_HEAD
            and evidence["failure_sha256"] == V4_FAILURE_SHA256
            and evidence["component_summary_sha256"] == V4_COMPONENT_SHA256
            and original["bindings"]["producer_source_sha256"] == V4_PRODUCER_SHA256,
            "Actual V4 closed evidence differs from independently recorded identities")
    cohorts,choices=_choices(original)
    bindings=dict(original_plan_sha256=evidence["original_plan_sha256"],
        prior_evidence_inventory_sha256=evidence["inventory_sha256"],
        prior_journal_head_sha256=evidence["journal_head_sha256"],
        prior_failure_sha256=evidence["failure_sha256"],component_summary_sha256=evidence["component_summary_sha256"],
        producer_source_sha256=original["bindings"]["producer_source_sha256"],
        evaluation_source_sha256=evaluation_source_sha256)
    result=dict(schema=SCHEMA,status="prepared_not_executed",synthetic=synthetic,
        protocol=protocol(),requests=210,batch_size=35,cohorts=cohorts,choices=choices,
        original_plan=original,prior_evidence=evidence,prior_assessment=partial,bindings=bindings,
        final_checkpoint_sha256=V4_CHECKPOINT_SHA256,parent_checkpoint_sha256=PARENT_CHECKPOINT_SHA256,
        model_hashes=copy.deepcopy(original["model_hashes"]),
        independent_inference_status="pending_pinned_cpu_forward_reconstruction",
        native_guard_scoring_startup_status="pending_separate_production_preflight_and_trace_audit",
        execution_authorized=False,release_authorized=False,native_execution_ready=False,
        original_assessment_completed=False,model_promoted=False,
        scope="Prepared untouched TEST choices from the pre-frozen V4 plan. No TRAIN/validation repeat, optimizer, model selection or new task work. Preparation does not certify model inference or native execution.")
    result["fingerprint"]=canonical_hash(result)
    return result


def validate_prepared(plan):
    _fingerprint(plan,"Prepared evaluation")
    expected=prepare_evaluation(plan["original_plan"],plan["prior_evidence"],
        evaluation_source_sha256=plan["bindings"]["evaluation_source_sha256"],synthetic=plan["synthetic"])
    require(plan == expected,"Prepared evaluation differs from preserved original TEST suffix")
    return copy.deepcopy(expected)
