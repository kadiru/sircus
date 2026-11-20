"""Independent accounting audit for the location-comparison runner.

Standard-library only: no optimizer, model, simulator or geometry search is
imported. A valid hash chain authenticates internal joins, not physical truth.
Native traces remain physics-pending unless a caller supplies an independent
evidence verifier for each completed request batch.
"""

from collections import Counter, defaultdict
import copy
import hashlib
import json
import math
from pathlib import Path
import re

SCHEMA = "sircus-location-comparison-trace-v1"
AUDIT_SCHEMA = "sircus-location-comparison-accounting-audit-v1"
PARENT_SHA = "57d7658e9ed2f72ad09cf4e8523ab497ccbe0708364f1b8ec4a4c5c1aef1fe9f"
MODELS = ("parent", "original_usable", "broader")
CORE_STATE = ("model_id", "weight_hash", "optimizer_hash", "optimizer_steps", "updates",
              "attempts", "parent_checkpoint_sha256")
RNG_STATE = ("action_rng_hash", "optimizer_rng_hash")
MILESTONES = (2240, 4480, 6720, 8960)


def require(value, message):
    if not value:
        raise ValueError(message)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def _sha(value):
    return type(value) is str and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def _state(value):
    require(type(value) is dict and all(key in value for key in CORE_STATE), "Missing model state")
    require(type(value["model_id"]) is str and value["model_id"], "Missing model instance ID")
    require(all(_sha(value[key]) for key in ("weight_hash", "optimizer_hash", "parent_checkpoint_sha256")),
            "Invalid state hash")
    require(value["parent_checkpoint_sha256"] == PARENT_SHA, "Different parent checkpoint")
    require(all(type(value[key]) is int and value[key] >= 0
                for key in ("optimizer_steps", "updates", "attempts")), "Invalid model counters")
    require(all(_sha(value[key]) for key in RNG_STATE if key in value), "Invalid optional RNG state hash")
    return {key: value[key] for key in CORE_STATE}


def _same_rng(left, right, fields=RNG_STATE):
    """Compare supplied RNG receipts without requiring old mocks to fabricate them."""
    for key in fields:
        if key in left or key in right:
            require(_sha(left.get(key)) and _sha(right.get(key)) and left[key] == right[key],
                    "RNG state changed outside its permitted operation: " + key)


def _rng_membership(left, right):
    require({key for key in RNG_STATE if key in left} == {key for key in RNG_STATE if key in right},
            "RNG state receipt disappeared or appeared mid-operation")


def _expected_batches(plan):
    """Reconstruct expected work independently, without runner scheduling code."""
    result = []

    def add(model, phase, milestone, cells, film=False):
        for start in range(0, len(cells), 35):
            result.append(dict(model=model, phase=phase, milestone=milestone,
                               cells=cells[start:start + 35], film=film and start == 0))

    add("parent", "monitor", 0, plan["cohorts"]["monitor"])
    for model in MODELS[1:]:
        schedule = plan["schedules"][model]
        require(len(schedule) == 8960, "Different TRAIN cap")
        for start in range(0, 8960, 35):
            add(model, "train", start, schedule[start:start + 35], film=start in (0, *MILESTONES[:-1]))
            completed = start + 35
            if completed in MILESTONES:
                add(model, "monitor", completed, plan["cohorts"]["monitor"])
    for phase in ("test_broad", "test_local"):
        cohort = plan["cohorts"][phase]
        for rotation in range(3):
            for model_index, model in enumerate(MODELS):
                subset = [cell for index, cell in enumerate(cohort)
                          if (index + rotation) % 3 == model_index]
                add(model, phase, rotation, subset, film=rotation == 0 and model == "parent")
    return result


def _observations(xy):
    x, y = xy
    dx, dy = .67 - x, -.10 - y
    bearing = math.atan2(dy, dx)
    return [(x - .67) / .9, (y + .25) / .625, math.hypot(dx, dy),
            math.sin(bearing), math.cos(bearing)]


def _check_proposals(reservation, packet):
    require(type(packet) is dict and packet.get("deterministic") is reservation["deterministic"],
            "Proposal inference mode changed")
    rows = packet.get("rows")
    require(type(rows) is list and len(rows) == len(reservation["cells"]), "Proposal coverage changed")
    for cell, row in zip(reservation["cells"], rows):
        require(row.get("request_id") == cell["request_id"] and row.get("target_id") == cell["target_id"],
                "Proposal/request identity changed")
        for key in ("observation", "latent", "actions_normalized"):
            require(type(row.get(key)) is list and len(row[key]) == 5 and all(map(_finite, row[key])),
                    "Invalid proposal vector: " + key)
        require(_finite(row.get("logp")), "Invalid raw likelihood")
        require(all(abs(actual - expected) <= 2e-6 for actual, expected in
                    zip(row["observation"], _observations(cell["bucket_xy_m"]))),
                "Observation does not describe the requested bucket")
        require(all(abs(action) <= 1 and abs(action - math.tanh(latent)) <= 2e-6
                    for action, latent in zip(row["actions_normalized"], row["latent"])),
                "Normalized action is not the original latent's tanh")


def _check_outcomes(reservation, rows):
    require(type(rows) is list and len(rows) == len(reservation["cells"]), "Terminal request coverage changed")
    for cell, row in zip(reservation["cells"], rows):
        require(row.get("request_id") == cell["request_id"] and row.get("target_id") == cell["target_id"],
                "Terminal/request identity changed")
        outcome, clean, reason, reward = (row.get(key) for key in ("outcome", "clean", "rejection_reason", "reward"))
        require(outcome in ("capture", "miss", "rejected") and type(clean) is bool and _finite(reward),
                "Invalid terminal outcome")
        require(not clean or outcome == "capture", "Clean label without capture")
        if outcome == "rejected":
            require(type(reason) is str and bool(reason) and reward == -1 and not clean,
                    "Rejected request was refunded or labelled successful")
        else:
            require(reason is None, "Released outcome has a rejection reason")
            if outcome == "capture":
                require(abs(reward - (1.6 if clean else 1.5)) < 1e-7, "Capture reward changed")
            else:
                require(reward == -.8 or -.7 <= reward <= 0, "Miss reward outside original rule")


def _summary(rows):
    counts = Counter(row["outcome"] for row in rows)
    return dict(requests=len(rows), captures=counts["capture"], clean=sum(row["clean"] for row in rows),
                rejected=counts["rejected"], misses=counts["miss"],
                rejection_reasons=dict(sorted(Counter(row["rejection_reason"] for row in rows
                    if row["outcome"] == "rejected").items())))


def audit_trace(plan, trace, *, native_verifier=None):
    """Audit a complete trace or a preserved prefix; never authorize resumption.

    `plan` must already be the caller's independently bound prepared artifact.
    `native_verifier(reservation, proposal, terminal)` must verify external raw
    evidence and model likelihoods, or raise. Merely supplying a truthy result
    cannot convert a synthetic trace into native evidence.
    """
    require(plan.get("fingerprint") == digest({k: v for k, v in plan.items() if k != "fingerprint"}),
            "Prepared plan fingerprint changed")
    require(plan.get("bindings", {}).get("parent_checkpoint") == PARENT_SHA, "Prepared parent changed")
    require(type(trace) is dict and trace.get("schema") == SCHEMA and trace.get("mode") in ("synthetic", "native")
            and trace.get("plan_fingerprint") == plan["fingerprint"], "Trace/plan identity changed")
    require(native_verifier is None or callable(native_verifier), "Expected an independent native evidence verifier")
    require(len(plan["cohorts"]["monitor"]) == 35 and len(plan["cohorts"]["test_broad"]) == 70,
            "Monitoring or primary TEST size changed")
    expected = _expected_batches(plan)
    events = trace.get("events")
    require(type(events) is list and bool(events), "Missing trace events")
    states, restorations, checkpoint_files, checkpoints = {}, {}, set(), {}
    full_states, initial_full_states = {}, {}
    pending_examples = defaultdict(list)
    outcomes = defaultdict(list)
    requested = Counter()
    completed = Counter()
    costs = defaultdict(float)
    per_model_phase_costs = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
    update_counts = Counter()
    request_ids, batch_ids = set(), set()
    reserve = proposal = None
    checkpoint_freeze = finish = None
    previous = "0" * 64
    batch_index = 0
    native_checked = 0

    def record_costs(model, phase, measurements):
        require(type(measurements) is dict, "Invalid cost measurements")
        for key, value in measurements.items():
            if type(value) not in (int, float):
                continue
            require(_finite(value) and value >= 0, "Invalid resource measurement")
            peak = any(word in key for word in ("peak", "rss", "gpu_memory"))
            destination = per_model_phase_costs[model][phase]
            if peak:
                costs[key] = max(costs[key], value)
                destination[key] = max(destination[key], value)
            else:
                costs[key] += value
                destination[key] += value

    for index, event in enumerate(events):
        require(type(event) is dict and set(event) == {"seq", "kind", "payload", "previous", "fingerprint"}
                and type(event["seq"]) is int and event["seq"] == index and event["previous"] == previous
                and event["fingerprint"] == digest({k: v for k, v in event.items() if k != "fingerprint"}),
                "Trace chain, order or contents changed")
        require(finish is None, "Events after terminal finish")
        previous = event["fingerprint"]
        kind, payload = event["kind"], event["payload"]
        require(type(payload) is dict, "Invalid event payload")
        if kind == "start":
            require(index == 0 and payload.get("plan_fingerprint") == plan["fingerprint"]
                    and payload.get("mode") == trace["mode"] and payload.get("seeds") == plan["seeds"],
                    "Changed start binding")
        else:
            require(index > 0 and events[0]["kind"] == "start", "Missing start event")
        if kind == "restore":
            model, state = payload["model"], _state(payload["state"])
            require(model in MODELS and model not in states and batch_index == 0 and reserve is None,
                    "Repeated, late or unknown parent restoration")
            require((state["attempts"], state["updates"], state["optimizer_steps"]) == (10000, 73, 292),
                    "Original learning state not restored")
            for key in ("action_seed", "optimizer_seed"):
                require(payload["state"].get(key) == plan["seeds"][key.removesuffix("_seed")
                    if key == "optimizer_seed" else "action_noise"], "Unpaired randomness seed")
            if restorations:
                first = next(iter(restorations.values()))
                require(state["model_id"] not in {s["model_id"] for s in states.values()}, "Shared mutable model instance")
                require(all(state[key] == first[key] for key in CORE_STATE if key != "model_id"),
                        "Groups did not start from identical weights and Adam state")
                _same_rng(payload["state"], next(iter(initial_full_states.values())))
            states[model] = state
            restorations[model] = copy.deepcopy(state)
            full_states[model] = copy.deepcopy(payload["state"])
            initial_full_states[model] = copy.deepcopy(payload["state"])
            if "restore_wall_s" in payload:
                record_costs(model, "restore", {"restore_wall_s": payload["restore_wall_s"]})
        elif kind == "reserve":
            require(reserve is None and set(states) == set(MODELS) and batch_index < len(expected),
                    "Overlapping, uninitialized or excess request reservation")
            require(all((model, "initial") in checkpoints for model in MODELS),
                    "Request before all initial checkpoint identities were saved")
            wanted = expected[batch_index]
            require(all(payload.get(key) == wanted[key] for key in ("model", "phase", "milestone", "film")),
                    "Batch/model/phase/milestone/film sequence changed")
            require(type(payload.get("deterministic")) is bool
                    and payload["deterministic"] is (wanted["phase"] != "train"), "Wrong inference mode")
            require(type(payload.get("batch_id")) is str and payload["batch_id"] not in batch_ids,
                    "Duplicate or missing batch ID")
            require(type(payload.get("reservation_id")) is int and payload["reservation_id"] == batch_index,
                    "Missing durable reservation identity")
            cells = payload.get("cells")
            require(type(cells) is list and len(cells) == len(wanted["cells"]), "Changed paid batch size")
            for slot, (cell, source) in enumerate(zip(cells, wanted["cells"])):
                require(cell.get("id") == cell.get("target_id") == source["id"]
                        and cell.get("bucket_xy_m") == source["bucket_xy_m"]
                        and cell.get("physical_slot") == slot and cell.get("cohort") == wanted["phase"]
                        and cell.get("split") == ("train" if wanted["phase"] == "train" else
                                                  "validation" if wanted["phase"] == "monitor" else "test"),
                        "Requested target, coordinates or physical slot changed")
                require(type(cell.get("request_id")) is str and cell["request_id"] not in request_ids,
                        "Repeated paid request ID")
                if "original_id" in source:
                    require(cell.get("original_id") == source["original_id"], "Original target provenance changed")
                request_ids.add(cell["request_id"])
            if wanted["phase"].startswith("test_"):
                require(checkpoint_freeze is not None, "Final TEST before all final models were frozen")
            else:
                require(checkpoint_freeze is None, "TRAIN or monitoring after TEST checkpoint freeze")
            require(not pending_examples[payload["model"]] or len(pending_examples[payload["model"]]) < 140,
                    "PPO batch boundary skipped")
            reserve = copy.deepcopy(payload)
            proposal = None
            batch_ids.add(payload["batch_id"])
            requested[(payload["model"], payload["phase"])] += len(cells)
        elif kind == "proposal":
            require(reserve is not None and proposal is None and payload.get("batch_id") == reserve["batch_id"],
                    "Unreserved, repeated or mismatched proposal")
            _check_proposals(reserve, payload["packet"])
            require(payload["packet"].get("model_id") == states[reserve["model"]]["model_id"],
                    "Proposal from wrong restored model")
            proposal = copy.deepcopy(payload["packet"])
        elif kind == "terminal":
            require(reserve is not None and proposal is not None and payload.get("batch_id") == reserve["batch_id"],
                    "Unreserved or mismatched terminal outcome")
            model, phase = reserve["model"], reserve["phase"]
            _check_outcomes(reserve, payload["rows"])
            require(_state(payload["state_before"]) == _state(payload["state_after"]) == states[model],
                    "Proposal/rollout/assessment changed model learning state")
            _same_rng(payload["state_before"], full_states[model])
            _rng_membership(payload["state_before"], payload["state_after"])
            _same_rng(payload["state_before"], payload["state_after"],
                      RNG_STATE if reserve["deterministic"] else ("optimizer_rng_hash",))
            full_states[model] = copy.deepcopy(payload["state_after"])
            evidence = payload.get("evidence")
            require(type(evidence) is dict and evidence.get("synthetic") is (trace["mode"] == "synthetic"),
                    "Synthetic/native evidence label changed")
            record_costs(model, phase, payload.get("costs", {}))
            if trace["mode"] == "native" and native_verifier is not None:
                require(native_verifier(copy.deepcopy(reserve), copy.deepcopy(proposal), copy.deepcopy(payload)) is True,
                        "Independent native evidence verification failed")
                native_checked += 1
            rows = payload["rows"]
            outcomes[(model, phase, reserve["milestone"])].extend(copy.deepcopy(rows))
            completed[(model, phase)] += len(rows)
            if phase == "train":
                pending_examples[model].extend(dict(proposal=copy.deepcopy(p), outcome=copy.deepcopy(r))
                    for p, r in zip(proposal["rows"], rows))
            reserve = proposal = None
            batch_index += 1
        elif kind == "update":
            require(reserve is None and checkpoint_freeze is None, "Learning during unresolved work or after final freeze")
            model = payload.get("model")
            require(model in MODELS[1:] and len(pending_examples[model]) == 140, "Wrong learning split/buffer size")
            examples = pending_examples[model]
            require(payload.get("request_ids") == [x["outcome"]["request_id"] for x in examples]
                    and payload.get("examples_sha256") == digest(examples),
                    "PPO input differs from original TRAIN latents/logp/rewards or omits rejects")
            before, after = _state(payload["before"]), _state(payload["after"])
            require(before == states[model] and after["model_id"] == before["model_id"]
                    and after["attempts"] == before["attempts"] + 140
                    and after["updates"] == before["updates"] + 1
                    and after["optimizer_steps"] == before["optimizer_steps"] + 4,
                    "PPO update state chain or fixed epochs changed")
            _same_rng(payload["before"], full_states[model])
            _rng_membership(payload["before"], payload["after"])
            _same_rng(payload["before"], payload["after"], ("action_rng_hash",))
            states[model] = after
            full_states[model] = copy.deepcopy(payload["after"])
            pending_examples[model] = []
            update_counts[model] += 1
            if "update_wall_s" in payload:
                record_costs(model, "update", {"update_wall_s": payload["update_wall_s"]})
        elif kind == "checkpoint":
            require(reserve is None and checkpoint_freeze is None, "Checkpoint during unresolved work or after freeze")
            model, receipt = payload.get("model"), payload.get("receipt")
            require(model in states and type(receipt) is dict and _state(receipt["state"]) == states[model]
                    and _sha(receipt.get("sha256")), "Checkpoint/model binding changed")
            _same_rng(receipt["state"], full_states[model])
            filename = receipt.get("file")
            require(type(filename) is str and filename and not Path(filename).is_absolute()
                    and ".." not in Path(filename).parts and filename not in checkpoint_files,
                    "Checkpoint overwrite or escaped path")
            require(type(payload.get("label")) is str and payload["label"], "Checkpoint label missing")
            require((model, payload["label"]) not in checkpoints, "Repeated checkpoint label")
            if payload["label"] == "final":
                require(model in MODELS[1:] and completed[(model, "train")] == 8960
                        and update_counts[model] == 64 and not pending_examples[model],
                        "Final checkpoint before exact TRAIN budget")
            checkpoint_files.add(filename)
            checkpoints[(model, payload["label"])] = copy.deepcopy(receipt)
            if "checkpoint_wall_s" in payload:
                record_costs(model, "checkpoint", {"checkpoint_wall_s": payload["checkpoint_wall_s"]})
        elif kind == "frozen":
            require(reserve is None and checkpoint_freeze is None and all(not x for x in pending_examples.values())
                    and all(update_counts[model] == 64 for model in MODELS[1:]),
                    "Final freeze before exact complete learning budgets")
            frozen = payload.get("checkpoints")
            require(type(frozen) is dict and set(frozen) == set(MODELS), "Missing final checkpoint member")
            for model in MODELS:
                label = "initial" if model == "parent" else "final"
                require(frozen[model] == checkpoints.get((model, label))
                        and _state(frozen[model]["state"]) == states[model],
                        "Frozen TEST checkpoint is not the actual terminal state")
                _same_rng(frozen[model]["state"], full_states[model])
            checkpoint_freeze = copy.deepcopy(frozen)
        elif kind == "finish":
            require(payload.get("status") in ("complete_audit_pending", "incomplete_no_retry"), "Unknown finish status")
            finish = payload
            def reported_group(model=None):
                selected = [row for (m, phase, _), rows in outcomes.items()
                            if model is not None and m == model and phase == "train" for row in rows]
                if model is None:
                    selected = [row for (_, phase, _), rows in outcomes.items() if phase != "train" for row in rows]
                summary = _summary(selected)
                reserved_count = requested[(model, "train")] if model else sum(
                    count for (_, phase), count in requested.items() if phase != "train")
                return dict(requested=reserved_count, completed=summary["requests"],
                    captures=summary["captures"], clean_captures=summary["clean"],
                    rejected=summary["rejected"], released_misses=summary["misses"])
            reconstructed = dict(train={model: reported_group(model) for model in MODELS[1:]},
                                 assessment=reported_group())
            require(payload.get("counts") == reconstructed, "Reported counts differ from charged rows")
            bounds = payload.get("bounds")
            require(type(bounds) is dict and bounds.get("comparison_fingerprint") == plan["fingerprint"]
                    and bounds.get("train_requested") == {m: requested[(m, "train")] for m in MODELS[1:]}
                    and bounds.get("train_completed") == {m: completed[(m, "train")] for m in MODELS[1:]}
                    and bounds.get("assessment_requested") == reconstructed["assessment"]["requested"]
                    and bounds.get("assessment_completed") == reconstructed["assessment"]["completed"],
                    "Bounded accounting differs from independent request counts")
            began = events[0]["payload"]["bounds"].get("started_monotonic")
            observed = bounds.get("observed_monotonic")
            elapsed = bounds.get("phase_elapsed_s")
            require(_finite(began) and _finite(observed) and observed >= began
                    and bounds.get("started_monotonic") == began
                    and type(elapsed) is dict and set(elapsed) == {"shared", *MODELS[1:]}
                    and all(_finite(value) and value >= 0 for value in elapsed.values())
                    and math.isclose(sum(elapsed.values()), observed - began, rel_tol=0, abs_tol=1e-6),
                    "Phase wall accounting changed")
            if finish["status"] == "complete_audit_pending":
                require(reserve is None and batch_index == len(expected) and checkpoint_freeze is not None
                        and requested == completed and all(not x for x in pending_examples.values())
                        and all(update_counts[model] == 64 for model in MODELS[1:]),
                        "Incomplete work labelled complete")
                require(states["parent"] == restorations["parent"], "Untouched parent changed")
                _same_rng(full_states["parent"], initial_full_states["parent"])
                require(bounds.get("pending") is None and bounds.get("stopped") is None
                        and bounds.get("phase_index") == 3 and bounds.get("active_phase") == "shared",
                        "Complete run has unresolved or stopped bounds")
                require(all(value < (3600 if phase == "shared" else 28800) for phase, value in elapsed.items())
                        and type(bounds.get("free_storage_bytes")) is int
                        and bounds["free_storage_bytes"] >= 6 * 1024**3,
                        "Complete run exceeded fixed wall/storage bounds")
        elif kind != "start":
            raise ValueError("Unknown trace event kind: " + str(kind))

    complete = finish is not None and finish["status"] == "complete_audit_pending"
    status = ("synthetic_verified" if trace["mode"] == "synthetic" else
              "native_accounting_verified_physics_pending") if complete else "incomplete_preserved_no_retry"
    if complete and trace["mode"] == "native" and native_verifier is not None:
        status = "native_accounting_and_external_evidence_verified"
    train = {model: _summary([row for (m, phase, _), rows in outcomes.items()
                             if m == model and phase == "train" for row in rows]) for model in MODELS[1:]}
    assessments = [dict(model=model, phase=phase, milestone=milestone, **_summary(rows))
                   for (model, phase, milestone), rows in outcomes.items() if phase != "train"]
    final_tests = {model: {phase: _summary([row for (m, p, _), rows in outcomes.items()
                                         if m == model and p == phase for row in rows])
                          for phase in ("test_broad", "test_local")} for model in MODELS}
    report = dict(schema=AUDIT_SCHEMA, status=status, synthetic=trace["mode"] == "synthetic",
        plan_fingerprint=plan["fingerprint"], trace_head=previous, event_count=len(events),
        requested=sum(requested.values()), completed=sum(completed.values()),
        unresolved=sum(requested.values()) - sum(completed.values()), train=train,
        updates=dict(update_counts), assessments=assessments, final_tests=final_tests, terminal_batches=batch_index,
        expected_batches=len(expected), native_evidence_batches_checked=native_checked,
        costs=dict(costs), per_model_phase_costs={model: {phase: dict(values) for phase, values in phases.items()}
                                               for model, phases in per_model_phase_costs.items()},
        observed_phase_wall_s=copy.deepcopy(finish["bounds"]["phase_elapsed_s"]) if finish else None,
        cost_scope="Only recorded scopes; nested geometry is not added to rollout wall. Restore/checkpoint/update costs stay separate. Memory is a sampled/lifetime peak, not a phase-local peak. Fabricated clocks are not measured compute.",
        native_physics_verified=complete and trace["mode"] == "native" and native_verifier is not None,
        checkpoint_bytes_independently_verified=False,
        limitations="Accounting validates recorded state and joins; independent checkpoint-byte, policy-likelihood, physics/geometry and watchdog checks remain necessary for native claims.")
    report["fingerprint"] = digest(report)
    return report


def read_trace(path):
    """Reject duplicate keys and nonfinite constants before checking a trace."""
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, "Duplicate trace JSON key")
            result[key] = value
        return result
    return json.loads(Path(path).read_bytes(), object_pairs_hook=pairs,
        parse_constant=lambda _: (_ for _ in ()).throw(ValueError("Nonfinite trace JSON")))
