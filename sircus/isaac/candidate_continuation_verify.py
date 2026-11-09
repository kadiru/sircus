"""Read-only learning-journal and final-checkpoint cross-check.

No updater, optimizer, producer or native runtime is imported. A passed result
means structural learning integrity and actual saved-model binding only. It
does not certify gradient arithmetic, native guards, physics, or capture labels.
"""

import hashlib
import json
import math
from pathlib import Path

from candidate_continuation_audit import audit_events
from candidate_continuation_checkpoint import load_continuation
from candidate_continuation_contract import RUN_SCHEMA, SEED, DEMONSTRATIONS, DEMO_BATCH, IMITATION_EPOCHS
from candidate_continuation_journal import read_events
from candidate_continuation_ledger import Ledger, canonical_hash
from continuous_policy import weight_hash, torch


def verify_run(directory, *, expected_result_sha256, expected_bindings,
               train_context_ids, supports, demonstrations):
    """Bind externally trusted result bytes to journal, counters and model bytes."""
    directory = Path(directory)
    raw = (directory/"result.json").read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_result_sha256:
        raise ValueError("External run-result SHA changed")
    result = json.loads(raw, parse_constant=lambda _: (_ for _ in ()).throw(ValueError("Nonfinite run JSON")))
    unsigned = {k:v for k,v in result.items() if k != "fingerprint"}
    if result.get("schema") != RUN_SCHEMA or result.get("status") != "learning_complete_audit_pending" \
            or canonical_hash(unsigned) != result.get("fingerprint") \
            or result.get("bindings") != expected_bindings \
            or result.get("execution_authorized") is not False or result.get("release_authorized") is not False:
        raise ValueError("Run result identity changed")
    events = read_events(directory/"journal")
    audit = audit_events(events, expected_bindings=expected_bindings,
        expected_head_sha256=result["journal_head_sha256"], train_context_ids=train_context_ids,
        supports=supports, demonstrations=demonstrations)
    if audit["status"] != "passed" or events[-1]["kind"] != "final":
        raise ValueError("Final learning history did not pass structural audit")
    final = events[-1]["payload"]
    if result["synthetic"] is not events[0]["payload"]["synthetic"] \
            or final["checkpoint_sha256"] != result["checkpoint_sha256"] \
            or final["counters"] != result["counters"] or final["ledger"] != result["ledger"]:
        raise ValueError("Run result differs from the committed final event")
    ledger = Ledger.from_dict(result["ledger"])
    model, saved = load_continuation(directory/"final.pt",
        expected_sha256=result["checkpoint_sha256"], expected_bindings=expected_bindings)
    if saved["stage"] != "final" or saved["counters"] != result["counters"] \
            or saved["ledger_fingerprint"] != ledger.to_dict()["fingerprint"] \
            or saved["journal_fingerprint"] != events[-1]["previous_sha256"] \
            or weight_hash(model) != final["weight_hash"]:
        raise ValueError("Final saved model differs from learning journal or ledger")
    # Replay dedicated CPU RNG consumption, without constructing any optimizer
    # or reproducing a weight update. The reported outcome cannot select an epoch.
    shuffle = torch.Generator(device="cpu").manual_seed(SEED)
    batches = [e["payload"]["demo_indices"] for e in events if e["kind"] == "imitation_update"]
    expected_batches = []
    for _ in range(IMITATION_EPOCHS):
        permutation = torch.randperm(DEMONSTRATIONS, generator=shuffle).tolist()
        expected_batches.extend(permutation[i:i+DEMO_BATCH] for i in range(0, DEMONSTRATIONS, DEMO_BATCH))
    sample = torch.Generator(device="cpu").manual_seed(SEED)
    sizes = {s.context_id:len(s.candidate_ids) for s in supports}
    for event in events:
        if event["kind"] == "reserve":
            for choice in event["payload"]["choices"]:
                size = sizes[choice["context_id"]]
                torch.multinomial(torch.ones(size, dtype=torch.float64)/size, 1, generator=sample)
        elif event["kind"] == "imitation_update":
            metric = event["payload"]["metrics"]
            if not math.isclose(metric["total"], metric["demonstration_nll"]+.05*metric["forward_kl"], rel_tol=1e-6, abs_tol=1e-7):
                raise ValueError("Reported imitation objective terms disagree")
        elif event["kind"] == "ppo_update":
            for metric in event["payload"]["epochs"]:
                if not math.isclose(metric["total"], metric["actor_loss"]+.5*metric["critic_loss"]-.005*metric["entropy"], rel_tol=1e-6, abs_tol=1e-7):
                    raise ValueError("Reported PPO objective terms disagree")
    if batches != expected_batches or not torch.equal(shuffle.get_state(), saved["rng_states"]["shuffle"]) \
            or not torch.equal(sample.get_state(), saved["rng_states"]["sample"]):
        raise ValueError("Dedicated seeded schedule or RNG consumption changed")
    verified = dict(audit, checkpoint_bytes_verified=True, checkpoint_ledger_journal_bound=True,
        result_sha256=expected_result_sha256, checkpoint_sha256=result["checkpoint_sha256"],
        seeded_demo_order_verified=True, dedicated_rng_consumption_verified=True,
        synthetic=result["synthetic"], optimizer_arithmetic_replayed=False,
        native_physics_or_capture_verified=False)
    verified.pop("fingerprint", None)
    verified["fingerprint"] = canonical_hash(verified)
    return verified
