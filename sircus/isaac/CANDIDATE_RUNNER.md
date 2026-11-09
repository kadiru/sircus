# Candidate continuation runner: implemented, real training off

This adds bounded learning and persistent accounting to the
[candidate adapter](CANDIDATE_ADAPTER.md). The original seed-22 checkpoint,
completed benchmark, TRAIN bank and existing viewers remain unchanged.
There is no training launch command or native dispatch in these modules.

## What the software now does

| Component | Implemented behavior |
|---|---|
| [Learning engine](candidate_continuation_updates.py) | Actor-only imitation, then categorical PPO, with fresh Adam states and dedicated CPU RNG streams |
| [Fixed learning contract](candidate_continuation_contract.py) | Hashes the proposed schedules, learning rates, objective coefficients and inherited counters |
| [Durable journal](candidate_continuation_journal.py) | Publishes new-only, complete, fsynced hash-linked events; an I/O failure poisons the writer |
| [TRAIN driver](candidate_continuation_runner.py) | Reserves requests before returning exact choices, records terminal prefixes, includes rejections, flushes buffers and freezes one final artifact |
| [New checkpoint schema](candidate_continuation_checkpoint.py) | Strict tensor/Adam/RNG/counter/source identity; atomic new-file publication and model-only loading |
| [Structural audit](candidate_continuation_audit.py) | Independently replays the learning schedule, ledger, chosen support, terminal coverage, detached advantages and failure accounting |
| [Saved-run verification](candidate_continuation_verify.py) | Binds externally trusted result bytes to the actual final model, ledger and preceding journal head; checks seeded demonstration order and dedicated RNG consumption |

Importing modules never trains a model. Explicit engine construction creates a
fresh optimizer but performs no update. The driver has no simulator callback,
arm command, assessment entry point or resume operation. A future caller must
verify original source/preparation evidence before constructing a real run.
Real runs require the exact known seed-22 tensor hash and complete external
preparation/producer bindings. These hashes provide identity, not permission or
independent proof that the native guards actually ran.

## Fixed learning and budget limits

Imitation visits every one of the 1,534 exact successful actions once per epoch,
for 20 epochs, 30,680 presentations and 240 actor-only steps. Batches have at most
128 examples; critic and log scales remain frozen. Context-balanced likelihood
and the fixed parent KL anchor use the adapter's existing categorical law.

PPO unfreezes the original actor, critic and log scales, starts fresh Adam, and
uses four epochs per 140-request buffer. It preserves the original reward and
clamp, fixed learning settings and detached terminal advantages. The driver
keeps every supplied rejection, enforcing the original -1 invalid reward, and
cannot omit a request from a committed buffer. Final partial buffers flush only
at a TRAIN stopping cap. There are at most 13 updates / 52 PPO optimizer steps.
The inherited 10,000 requests / 73 updates remain separate.

The caps remain **1,400 actual releases or 1,750 requested slots**, whichever
comes first. The existing ledger is explicitly versioned to v2: an active TRAIN
batch now also stops at the next 140-request buffer boundary, in addition to
35 slots and the remaining request/release allowances. Near the release cap,
several smaller rejected batches could otherwise straddle that boundary. The
refinement prevents stale spillover or premature updates; context order, caps
and all physical guards are unchanged. Assessment accounting remains separate
at 420 requests, but this TRAIN driver does not execute assessment.

The changed ledger source hash requires a new compact TRAIN export. The
candidate supports and demonstration records are identical to the prior export;
historical receipts remain preserved under their original source snapshot.

## Persistence and failure behavior

The driver requires a new output directory. It fsyncs directory creation,
writes each complete receipt to a temporary file, fsyncs its bytes, publishes
without replacement and fsyncs the directory. Only then can a reservation
return choices to an external producer. No file is overwritten on retry.
Partial outcomes remain charged in the ledger. Missing outcomes and failed
optimizer operations cannot be refunded, replayed or resumed.

A failed operation stops the driver. The audit distinguishes committed update
counters from at most one bounded interrupted operation and reports incomplete
work. A saved final model is allowed only after a cap, complete terminal
coverage and the last PPO flush. Its new schema cannot be loaded as an old
continuous-policy checkpoint. The loader never resumes an optimizer or RNG.

The driver enforces the 6 GiB storage floor, a 10-minute imitation wall limit
and a 2-hour PPO wall limit at operation boundaries. It cannot interrupt an
external/native operation while waiting for its receipt. A separate bounded
native supervisor is still necessary. Reported process CPU and wall costs are
driver scopes; they do not infer native setup/flight or GPU memory costs.

## What this audit proves

The independent reader checks the external journal head, exact actions/latents,
teacher membership and per-epoch coverage, phase/counter/hash continuity, native
request reservations, terminal coverage, four-epoch PPO receipts, original reward
constants/bounds and independently recomputed advantages. Saved-run verification
also reads the actual new checkpoint and validates tensor hashes, Adam moments,
counter steps, source bindings, final ledger identity and seeded shuffle/RNG
consumption. Objective term sums are checked independently.

This is **structural learning and saved-state verification**. It does not replay
the actual Adam arithmetic from each gradient, derive rewards from full native
trajectories, verify sampled choices against every intermediate weight tensor,
or certify IK, path, settling, release guards, physics or capture labels.
Reported captures remain producer claims until the separate native evidence
audit passes. A structurally valid journal alone cannot establish improvement.

## Execution boundary

The [compact verification evidence](../../baselines/isaac-candidate-runner-20261109/README.md)
records 103/103 CPU cases passed without skips, including the full synthetic
240-step imitation / 1,400-release mock PPO path and final saved-run audit.
The actual parent separately retained 35/35 choice parity and identical model
hashes without gradients or updates. Standard project mock/numerical checks
also passed; their optional Torch skips are covered by the real CPU suite.

Validation uses fabricated policies and outcomes; synthetic optimizer steps
test the code path. The real seed-22 checkpoint is only read by a separate
forward-only compatibility process. No imitation/PPO step on it, new shot,
held-out evaluation, package installation into existing environments, service
restart or physical action belongs to this implementation task.

Before any real learning, finish outcome-blind fresh split/support preparation
and its independent audit, connect the unchanged native guards and full trace
receipts, implement the one-start diagnostic/GPU/time supervisor, extend the
native scoring/guard audit, and review the final-only 420-request assessment.
The [proposed continuation plan](CANDIDATE_CONTINUATION_PLAN.md) still governs
those gates. No learning or native performance improvement is claimed here.
