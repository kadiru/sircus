# Seed 22 candidate adapter: implemented, training off

This is the no-update implementation step after the
[proposed continuation plan](CANDIDATE_CONTINUATION_PLAN.md). It connects the
saved policy's preferences to immutable candidate choices and supplies pure
learning objectives and separate request/release accounting. It does not start
imitation, PPO, native simulation or physical execution.

## Implemented components

| Component | Responsibility | Execution boundary |
|---|---|---|
| [Policy adapter](candidate_continuation_policy.py) | Fixed candidate encodings; float64 categorical probabilities, sampling, exact entropy and deterministic choice | Receives a previously verified parent model; no optimizer or arm commands |
| [Input exporter](candidate_continuation_inputs.py) | Verifies the approved bank and original successful TRAIN evidence; binds exact action IDs and teacher targets | No new geometry proposal, solver, model inference or native trial |
| [Learning objectives](candidate_continuation_losses.py) | Context-balanced demonstration likelihood, parent KL anchor and categorical terminal PPO loss | Returns losses only; no optimizer or update loop |
| [Budget ledger](candidate_continuation_ledger.py) | Fixed 1,750-request / 1,400-release TRAIN caps and separate 420-request final assessment; immutable reservations and receipts | Pure standard library; no I/O, timers, launch permission or release permission |
| [Readiness check](candidate_continuation_readiness.py) | Strict CPU parent loading, frozen-selector prediction parity, teacher-data alignment and no-update proof | Forward calculation only; CUDA must be hidden; no backward on the real parent |

The original continuous policy/updater, frozen selector, geometry/native guards,
completed benchmark, source archives, bank and viewer services remain unchanged.
The adapter is a new action-law version. Its availability does not mean the
continuation experiment is ready to launch.

## What is preserved

The original actor and critic have the same shapes and weights. The wrapper
uses `ContinuousPolicy.distribution`, preserving the state-dependent radius
prior and saved five log scales. It scores each fixed latent candidate under
those Gaussian parameters and normalizes a finite categorical distribution.
There is no continuous density/Jacobian substituted for the categorical log
probability, no moving parent latent, and no forced parent winner after a
higher-scoring choice emerges. Deterministic ties retain the prescribed
parent-first / stable-ID convention within `1e-12`.

Supports contain exact actions and immutable original latent vectors. Their
fingerprints and the compact export's independently supplied raw SHA bind those
values. The original parent latent is checked exactly. For other candidates,
recomputing affine-inverse/atanh is a consistency check with `1e-12` absolute
roundoff tolerance, never a replacement for the stored vector. A cross-runtime
check measured a maximum difference of `4.44e-16`; all non-latent fields agreed.
This affects no IK, collision, settling, release or capture threshold.

Empty support is an explicit decline without a fabricated action. A singleton
has zero selection entropy/gradient while still providing a critic target.
Different candidate counts are represented without treating padded entries as
choices. Categorical rollout receipts freeze the support identity, selected
index, old probability/value, terminal return and advantage across PPO epochs.
They reject legacy continuous buffers or changed support/order.

## Teaching and accounting

The exporter maps every one of the original 1,534 successful paid TRAIN actions,
including 485 clean, to an admitted exact candidate. They cover 21 of 35 TRAIN
positions; the other 14 receive no invented successful label. A teacher loss
uses global context weights and exact candidate probabilities, not an average
of different board settings. The parent KL term is equally averaged across
all 35 contexts. The PPO objective includes every supplied selected request,
including rejections, and uses the original reward weights and declared
categorical entropy. The future producer must join the rollout to the ledger
so requests cannot be omitted before reaching this objective.

The ledger charges a request at reservation and a release only when its
terminal receipt says a release occurred. Missing/partial outcomes stay in the
consumed denominator. Batch capacity is the minimum of 35, remaining requests
and remaining releases; inactive padding never consumes a sample. Assessment
cannot begin until a TRAIN stopping cap and a frozen final checkpoint are
recorded, and its three-method rotation and 420-request schedule are fixed.
A ledger default has explicit unbound source placeholders; neither completed
bindings nor a receipt grants execution permission. A future producer must
persist a reservation atomically before execution; this pure object cannot
guarantee disk durability or enforce operational wall/storage limits.

## No-update validation

The [recorded CPU verification](../../baselines/isaac-candidate-adapter-20261109/README.md)
passed all 53 focused cases without skips. On the actual saved seed-22 parent,
all 35 deterministic choices match the frozen selector. The largest reference
log-probability error is `9.95e-6`; parent-mean error is `2.38e-7`, within the
declared `1e-4` and `3e-6` tolerances. Checkpoint and tensor hashes before and
after match; no real-parent gradients, optimizer steps, new shots or held-out
shot reads occurred. The forward-only check took 0.188 s on the existing CPU
runtime; it is not a training-speed measurement.

The dedicated CPU workflow installs pinned PyTorch 2.7.0 CPU wheels using the
[official distribution index](https://pytorch.org/get-started/previous-versions/)
and requires all adapter cases to run without skips. Synthetic backward tests
check derivatives and numerical objectives while never constructing an
optimizer or taking a step. These are software tests, not policy training.
The standard project suite continues to cover existing mock/numerical behavior.

A separate readiness CLI accepts only an existing exported input manifest,
its externally recorded SHA, the exact parent checkpoint and a new receipt
path. It cannot launch training or native execution:

```bash
CUDA_VISIBLE_DEVICES='' python sircus/isaac/candidate_continuation_readiness.py \
  --inputs /ignored/task/train-inputs.json \
  --expected-inputs-sha256 <recorded-export-sha256> \
  --checkpoint /read-only/original/checkpoints/milestone-10000.pt \
  --output /ignored/task/new-readiness-receipt.json
```

It reports candidate-ID parity over all 35 existing TRAIN contexts, finite
categorical/teacher calculations, CPU/source numerical tolerances, original
checkpoint/tensor hashes before and after, no parameter gradients, unchanged
global CPU RNG, zero optimizer steps and zero native requests. These checks do
not measure a new capture rate. The previous native comparison remains
13/35 versus 16/35 captures; no learning improvement is claimed here.

## Still deferred

The training loop and optimizer, new continuation-checkpoint writer, fresh
split/support preparation, atomic native producer, wall/storage/GPU supervisor,
independent training/update auditor and final native assessment remain to be
integrated and reviewed. In particular, the existing continuous PPO updater
and old two-cohort native runner are not shortcuts for those components.
No held-out positions or outcomes are read by this adapter check. No packages
are installed into existing Mac or Ubuntu environments for this task.

Training, new native shots and physical work remain off. This implementation
supplies tested components for the separately reviewed proposal; it does not
authorize its proposed budgets or change the original experiment.
