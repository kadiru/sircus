"""Independent TEST-only evaluation accounting checks for synthetic mocks.

This reader certifies software bookkeeping only. It cannot certify a real
assessment, authorize execution, replay physics or turn V4's interrupted
420-request assessment into a completed result. No model, optimizer, native
runtime or process-control module is imported here.
"""

from collections import Counter
import copy
import hashlib
import json
from pathlib import Path
import re

from candidate_continuation_contract import JOURNAL_SCHEMA
from candidate_v4_evaluation import validate_prepared


SCHEMA = "sircus-v4-test-evaluation-mock-audit-v1"
RESULT_SCHEMA = "sircus-v4-test-evaluation-result-v1"
V4_CHECKPOINT_SHA256 = "15020a948d295fe5e7258a732145b89aa61fda2c385a0d6966a9e4cc7e81bd41"
PARENT_CHECKPOINT_SHA256 = "57d7658e9ed2f72ad09cf4e8523ab497ccbe0708364f1b8ec4a4c5c1aef1fe9f"
METHODS = ("parent", "candidate_bank", "continued")
# Independent literals, rather than the producer's schedule or comparison code.
COHORTS = ((6, 0, "continued"), (7, 0, "parent"), (8, 0, "candidate_bank"),
           (9, 35, "parent"), (10, 35, "candidate_bank"), (11, 35, "continued"))
RESULT_FIELDS = {"request_id", "released", "decline_reason", "captured", "clean", "source_sha256", "return"}
COUNTS = ("requested", "terminal", "released", "rejected", "captures", "clean_captures", "unresolved")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _sha(value):
    return type(value) is str and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _json(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, "Duplicate audit JSON key")
            result[key] = value
        return result
    return json.loads(raw, object_pairs_hook=pairs,
        parse_constant=lambda _: (_ for _ in ()).throw(ValueError("Nonfinite audit JSON")))


def _journal(directory):
    directory = Path(directory)
    require(directory.is_dir() and not directory.is_symlink(), "Missing or linked evaluation journal")
    paths = sorted(directory.iterdir())
    require(2 <= len(paths) <= 512, "Incomplete or unbounded evaluation journal")
    result, head = [], "0" * 64
    for i, path in enumerate(paths):
        require(path.name == f"event-{i:06d}.json" and path.is_file() and not path.is_symlink(),
            "Unexpected, missing, partial or linked evaluation event")
        event = _json(path.read_bytes())
        require(type(event) is dict and set(event) == {"schema", "seq", "previous_sha256", "kind", "payload", "fingerprint"}
            and event["schema"] == JOURNAL_SCHEMA and type(event["seq"]) is int and event["seq"] == i
            and event["previous_sha256"] == head and type(event["payload"]) is dict
            and event["fingerprint"] == canonical_hash({k:v for k,v in event.items() if k != "fingerprint"}),
            "Evaluation journal chain changed")
        head = event["fingerprint"]
        result.append(event)
    return result


def _plan(plan):
    """Independently check suffix membership and every unchanged shot field."""
    validate_prepared(plan)
    require(type(plan) is dict and plan.get("schema") == "sircus-v4-test-evaluation-plan-v1"
        and plan.get("requests") == 210 and plan.get("final_checkpoint_sha256") == V4_CHECKPOINT_SHA256
        and plan.get("parent_checkpoint_sha256") == PARENT_CHECKPOINT_SHA256,
        "Wrong fixed V4 checkpoint or TEST-only evaluation scope")
    require(plan.get("fingerprint") == canonical_hash({k:v for k,v in plan.items() if k != "fingerprint"}),
        "Evaluation plan fingerprint changed")
    original = plan["original_plan"]
    require(type(original) is dict and len(original.get("choices", [])) == 420
        and len(plan.get("choices", [])) == 210 and len(plan.get("cohorts", [])) == 6,
        "Evaluation must preserve the original plan and select exactly its untouched TEST suffix")
    contexts, seen, disabled = [], set(), []
    expected_contexts = [c["context_id"] for c in original["choices"][210:245]]
    expected_contexts += [c["context_id"] for c in original["choices"][315:350]]
    require(len(expected_contexts) == len(set(expected_contexts)) == 70, "Repeated or missing original TEST positions")
    for cohort, (old_cohort, offset, method) in enumerate(COHORTS):
        current = plan["cohorts"][cohort]
        require(current.get("index") == cohort and current.get("original_cohort") == old_cohort
            and current.get("split") == "test" and current.get("offset") == offset and current.get("method") == method
            and current.get("choices") == plan["choices"][cohort * 35:(cohort + 1) * 35],
            "TEST cohort membership, rotation or order changed")
        for slot, choice in enumerate(current["choices"]):
            index, old_index = cohort * 35 + slot, old_cohort * 35 + slot
            previous = copy.deepcopy(original["choices"][old_index])
            expected = dict(previous, request_id=f"test-evaluation-{index:06d}", request_index=index,
                original_request_id=previous["request_id"], original_request_index=previous["request_index"],
                cohort=cohort, original_cohort=old_cohort)
            require(choice == expected and previous.get("request_id") == f"assessment-{old_index:06d}"
                and previous.get("request_index") == old_index and choice.get("split") == "test"
                and choice.get("method") == method and choice.get("context_id") == expected_contexts[offset + slot],
                "A frozen TEST action, context, original identity or inference choice changed")
            require(choice["request_id"] not in seen and type(choice["enabled"]) is bool, "Repeated or malformed TEST request")
            seen.add(choice["request_id"])
            contexts.append((method, choice["context_id"]))
            if not choice["enabled"]:
                require(method != "parent" and choice["decline_reason"] == "no_qualified_candidate_support",
                    "Empty support cannot become an enabled replacement")
                disabled.append((method, choice["context_id"]))
    require(set(contexts) == {(m,c) for m in METHODS for c in expected_contexts}
        and len(contexts) == len(set(contexts)) == 210, "TEST grid contains repeated, missing or foreign members")
    empty = {c for m,c in disabled}
    require(len(disabled) == 4 and len(empty) == 2
        and set(disabled) == {(m,c) for m in ("candidate_bank", "continued") for c in empty},
        "The two empty TEST positions must remain four charged declines")
    return expected_contexts


def _frozen(plan):
    models = plan["original_plan"]["model_hashes"]
    require(type(models) is dict and set(models) == {"parent_weight_hash", "continued_weight_hash"}
        and all(_sha(value) for value in models.values()), "Malformed immutable model identities")
    return dict(final_checkpoint_sha256=plan["final_checkpoint_sha256"],
        parent_checkpoint_sha256=plan["parent_checkpoint_sha256"], bindings=copy.deepcopy(plan["bindings"]),
        model_hashes=copy.deepcopy(models))


def _zero_work(value):
    require(all(type(value.get(key)) is int and value[key] == 0
        for key in ("training_requests", "optimizer_steps", "native_requests_executed")),
        "Mock work boundary must retain exact integer zero counters")
    if "native_application_starts" in value:
        require(type(value["native_application_starts"]) is int and value["native_application_starts"] == 0,
            "Mock evidence cannot contain native application starts")
    require(all(value.get(key, False) is False for key in
        ("execution_authorized", "release_authorized", "model_promoted", "full_native_audit",
         "original_assessment_completed", "native_execution_ready")),
        "Mock evidence cannot grant execution, promotion or a native audit")


def _outcome(row, choice):
    require(type(row) is dict and set(row) == RESULT_FIELDS and row["request_id"] == choice["request_id"]
        and all(type(row[k]) is bool for k in ("released", "captured", "clean"))
        and (not row["clean"] or row["captured"]) and (not row["captured"] or row["released"])
        and _sha(row["source_sha256"]), "Synthetic outcome identity, source or Boolean flags changed")
    require((row["released"] and row["decline_reason"] is None)
        or (not row["released"] and type(row["decline_reason"]) is str and bool(row["decline_reason"])),
        "Synthetic release/rejection reason changed")
    value = row["return"]
    require(type(value) in (float, int) and ((not row["released"] and value == -1.)
        or (row["captured"] and value == (1.6 if row["clean"] else 1.5))
        or (row["released"] and not row["captured"] and -.8 <= value <= 0.)),
        "Synthetic outcome violates the unchanged reward rules")
    require(choice["enabled"] or (not row["released"] and not row["captured"] and not row["clean"]
        and row["decline_reason"] == "no_qualified_candidate_support"),
        "An empty support was retried, replaced, released or dropped from the denominator")


def _comparison(plan, outcomes, contexts):
    rows = {(c["method"], c["context_id"]):r for c,r in zip(plan["choices"], outcomes)}
    methods = {}
    for method in METHODS:
        local = [rows[method, context] for context in contexts]
        released = sum(row["released"] for row in local)
        captures, clean = sum(row["captured"] for row in local), sum(row["clean"] for row in local)
        methods[method] = dict(requested=70, released=released, rejected=70 - released,
            captures=captures, clean_captures=clean, capture_rate=captures / 70., clean_capture_rate=clean / 70.,
            rejection_reasons=dict(sorted(Counter(r["decline_reason"] for r in local if not r["released"]).items())))
    pairs = {}
    for control in ("candidate_bank", "parent"):
        pair = {}
        for field in ("captured", "clean"):
            gains = [c for c in contexts if rows["continued", c][field] and not rows[control, c][field]]
            losses = [c for c in contexts if rows[control, c][field] and not rows["continued", c][field]]
            pair[field] = dict(gains=len(gains), losses=len(losses), ties=70 - len(gains) - len(losses),
                difference_count=len(gains) - len(losses), difference_rate=(len(gains) - len(losses)) / 70.,
                gain_context_ids=gains, loss_context_ids=losses)
        pairs["continued_minus_" + control] = pair
    return dict(methods=methods, test_pairs=pairs, synthetic=True,
        primary_comparison="continued_minus_candidate_bank",
        primary_metric="Captures divided by all 70 requested TEST positions per method, including all declines",
        scope="Fabricated outcomes verify reporting arithmetic only; no model performance, physics or generalization is established.")


def audit_mock_evaluation(directory, plan, summary):
    """Recompute complete mock accounting; real evaluation always fails closed."""
    contexts = _plan(plan)
    require(plan.get("synthetic") is True and type(summary) is dict and summary.get("synthetic") is True
        and summary.get("mock_only") is True, "Real evaluation needs separately implemented independent native/startup/inference proofs")
    directory = Path(directory)
    require(directory.is_absolute() and directory == directory.resolve()
        and directory.is_dir() and not directory.is_symlink() and not (directory / "native").exists(),
        "Mock-only audit cannot consume a native evidence directory")
    known_files = {"evaluation-plan.json", "mock-audit.json", "evaluation-result.json"}
    entries = list(directory.iterdir())
    require({p.name for p in entries}.issubset(known_files | {"journal"})
        and all(not p.is_symlink() and (p.is_dir() if p.name == "journal" else p.is_file()) for p in entries),
        "Interrupted, foreign, partial or linked evaluation artifacts cannot pass")
    saved_plan = directory / "evaluation-plan.json"
    require(saved_plan.is_file() and not saved_plan.is_symlink() and _json(saved_plan.read_bytes()) == plan,
        "Durable prepared evaluation plan is missing or changed")
    events = _journal(directory / "journal")
    frozen = _frozen(plan)
    require(events[0]["kind"] == "start" and events[0]["payload"] == dict(
        plan_fingerprint=plan["fingerprint"], synthetic=True, mock_only=True, frozen=frozen),
        "Wrong mock start or immutable checkpoint/source/model identities")
    outcomes, pending, next_cohort, requested = [], None, 0, 0
    final = None
    for event in events[1:]:
        kind, value = event["kind"], event["payload"]
        require(final is None, "Evaluation continued after its final receipt")
        if kind == "reserve":
            require(pending is None and next_cohort < 6 and set(value) == {
                "cohort", "original_cohort", "choices", "requested", "terminal", "plan_fingerprint"},
                "Repeated, pending or malformed evaluation reservation")
            expected = plan["choices"][next_cohort * 35:(next_cohort + 1) * 35]
            requested += 35
            require(value == dict(cohort=next_cohort, original_cohort=COHORTS[next_cohort][0],
                choices=expected, requested=requested, terminal=len(outcomes), plan_fingerprint=plan["fingerprint"]),
                "Reservation changed a frozen action, source cohort, count or order")
            pending = dict(cohort=next_cohort, choices=expected, done=0)
            next_cohort += 1
        elif kind == "outcomes":
            require(pending is not None and set(value) == {"cohort", "results", "requested", "terminal"}
                and value["cohort"] == pending["cohort"] and type(value["results"]) is list
                and 0 < len(value["results"]) <= 35 - pending["done"],
                "Outcomes lack an exact unpaid reservation prefix")
            for row in value["results"]:
                _outcome(row, pending["choices"][pending["done"]])
                outcomes.append(row)
                pending["done"] += 1
            require(value["requested"] == requested and value["terminal"] == len(outcomes),
                "Charged or terminal counts differ from independent replay")
            if pending["done"] == 35:
                pending = None
        elif kind == "final":
            require(pending is None and next_cohort == 6 and requested == len(outcomes) == 210,
                "Incomplete or interrupted TEST evaluation cannot pass")
            final = value
        else:
            raise ValueError("Unexpected or failed evaluation event; no resume, retry or partial pass")
    require(final is not None, "Evaluation has no complete final receipt")
    counts = dict(requested=210, terminal=210, released=sum(r["released"] for r in outcomes),
        rejected=sum(not r["released"] for r in outcomes), captures=sum(r["captured"] for r in outcomes),
        clean_captures=sum(r["clean"] for r in outcomes), unresolved=0)
    expected_final = dict(plan_fingerprint=plan["fingerprint"], frozen=frozen, counts=counts,
        synthetic=True, mock_only=True, training_requests=0, optimizer_steps=0, native_requests_executed=0)
    _zero_work(final)
    _zero_work(summary)
    require(type(final.get("counts")) is dict and all(type(final["counts"].get(key)) is int for key in COUNTS)
        and type(summary.get("counts")) is dict and all(type(summary["counts"].get(key)) is int for key in COUNTS),
        "Mock count fields must be exact integers")
    require(final == expected_final, "Final synthetic identity, count or zero-work boundary changed")
    require(summary.get("schema") == RESULT_SCHEMA and summary.get("status") == "mock_evaluation_complete_audit_pending"
        and summary.get("journal_head_sha256") == events[-1]["fingerprint"]
        and all(summary.get(k) == v for k,v in expected_final.items())
        and summary.get("model_promoted", False) is False and summary.get("full_native_audit", False) is False,
        "Mock result identity, provenance, completed counts or work boundary changed")
    result = dict(schema=SCHEMA, status="passed_mock_only", synthetic=True, mock_only=True,
        full_native_audit=False, plan_fingerprint=plan["fingerprint"], journal_head_sha256=events[-1]["fingerprint"],
        counts=counts, frozen=frozen, comparison=_comparison(plan, outcomes, contexts),
        training_requests=0, optimizer_steps=0, native_requests_executed=0,
        original_assessment_status="incomplete_no_retry", original_assessment_completed=False,
        model_promoted=False, execution_authorized=False, release_authorized=False,
        scope="Independent synthetic journal/accounting/reporting check only. Real frozen inference, native guards, physics, startup and source-file verification are not certified.")
    result["fingerprint"] = canonical_hash(result)
    # The first audit happens before publication. A later readback may contain
    # these two known immutable publications, but never a coexisting failure.
    audit_path, result_path = directory / "mock-audit.json", directory / "evaluation-result.json"
    if audit_path.exists():
        require(_json(audit_path.read_bytes()) == result, "Saved mock audit differs from independent replay")
    if result_path.exists():
        published = _json(result_path.read_bytes())
        _zero_work(published)
        require(published.get("schema") == RESULT_SCHEMA and published.get("status") == summary["status"]
            and published.get("fingerprint") == canonical_hash({k:v for k,v in published.items() if k != "fingerprint"})
            and published.get("audit") == result and published.get("journal_head_sha256") == events[-1]["fingerprint"]
            and all(published.get(k) == v for k,v in expected_final.items()),
            "Saved evaluation result differs from independent mock accounting")
    return result
