# Saved V4 model: separate bounded TEST evaluation preparation

This task prepares and rehearses software with **mocks only**. It does not load
the real model, start Isaac, run training or execute a task shot. Native
execution is deliberately disabled. The original frozen 60,000-request
benchmark, its scores, its reader and all V4 evidence stay unchanged.

## Why 210 requests, rather than another 420

V4 saved its final trained checkpoint before assessment:
`15020a948d295fe5e7258a732145b89aa61fda2c385a0d6966a9e4cc7e81bd41`.
Its original assessment stopped with **140 charged, 105 terminal and 35
unresolved** requests. The unresolved validation-bank cohort performed physical
work and cannot be refunded, replaced or retried. The old assessment remains
`incomplete_no_retry`; this preparation does not reopen its ledger.

The preserved journal and native-file inventory show **zero TEST requests**.
The new proposal takes only original cohorts 6–11: **70 untouched TEST positions
times three frozen methods = 210 requests**. It omits all TRAIN and validation
requests, including the 70 unused validation slots. Consequently it cannot
complete the old 420-request assessment or establish a validation result.

| Cohort | Original cohort | TEST positions | Frozen method |
| --- | ---: | --- | --- |
| 0 | 6 | First 35 | Continued V4 |
| 1 | 7 | First 35 | Original seed 22 |
| 2 | 8 | First 35 | Original seed-22 bank selector |
| 3 | 9 | Last 35 | Original seed 22 |
| 4 | 10 | Last 35 | Original seed-22 bank selector |
| 5 | 11 | Last 35 | Continued V4 |

Every command comes from the original pre-frozen V4 plan. The new wrapper adds
new evaluation request IDs while retaining the original request/cohort IDs.
Action values, context, candidate identity, score, latent, IK seeds and enabled
state remain exact copies. There is no new candidate search, retuning or
checkpoint selection. Two empty TEST supports remain **four charged declines**
for the two bank methods; all 70 positions remain in each method's denominator.

## Measurements and fixed bounds

The primary comparison is continued V4 versus the original seed-22 bank
selector, using captures divided by **all 70 requested TEST positions**. The
original direct seed-22 policy is a second control. Report clean captures,
rejections by cause, releases, and paired capture gains/losses as separate
metrics. A geometry check passing does not guarantee a quiet release, a clear
flight or a capture. Rejections and empty supports count as failures.

| Limit | Proposal |
| --- | ---: |
| TRAIN / validation requests | 0 / 0 |
| TEST requests | Exactly 210 maximum; one per method/position |
| Optimizer steps / adaptive inference searches | 0 / 0 |
| Batch size | 35 |
| Native starts, if separately authorized later | At most 1 |
| Initialization / evaluation / CPU audit | 600 / 1,800 / 600 seconds |
| Total application wall limit | 3,000 seconds |
| Free-space floor | 6 GiB |
| Retry, resume, replacement or model promotion | None |

Charge a cohort durably before exposing its commands to the executor. Preserve
any terminal prefix and every unresolved reservation on failure. A new output
must not overlap old evidence, source or benchmark directories. Report new
initialization/evaluation/audit wall and CPU costs separately from inherited V4
training and earlier failed-attempt costs. No new videos are part of this
unchanged shot protocol; a future saved-state replay would need explicit labels.

## Implemented and checked in this preparation

- [Pure planner](candidate_v4_evaluation.py) checks the original 420-choice plan,
  replays the closed reservation/accounting chain, protects the exact 25-file
  partial-assessment inventory and extracts only its untouched TEST suffix.
  Actual V4 plan, journal head, inventory, failed receipt and completed-TRAIN
  component summary have separately recorded immutable identities. The model's
  original producer source remains distinct from new evaluation source.
- [Mock runner](candidate_v4_evaluation_runner.py) enforces new-only output,
  fixed accounting, checkpoint/source/model identity and wall/storage checks at
  callback boundaries. It accepts only explicit synthetic mocks and rejects a
  runtime with a training API. It cannot preempt a blocked callback; it is not a
  production watchdog or native launcher.
- [Independent mock reader](candidate_v4_evaluation_audit.py) reconstructs the
  exact six rotations, 210 request/action joins, journal accounting, four empty
  declines, 70-position denominators and paired reporting arithmetic. Mock
  completion is labelled synthetic and cannot certify physics or model quality.

The actual closed metadata was read without simulator startup, model loading or
inference. It passes the preparation checks. Raw plans, positions, journals,
media, host paths and runtime configuration remain private, outside Git.

[Compact preparation evidence](../../baselines/isaac-v4-test-preparation-20261109/summary.json) records immutable identities and checked bounds, with no raw positions or observations.

## Software validation

The 47 new synthetic cases passed without skips: ten preparation/exposure
checks, 20 bounded-runner checks and 17 independent-reader checks. They cover
exact suffix preservation, charged partial prefixes, empty supports, changed
model/action/source identities, time/storage failures, no retry, stored-plan and
failed-artifact integrity, and false native/completion claims. Independent code
and scope review passed. These are software results, not capture measurements.

Repository coverage also completed: 1,214 Isaac cases (121 optional skips),
34 physical mocks, five simulation cases, 59 Node cases and nine numerical
check scripts. The first sandboxed run blocked two existing localhost socket
tests; only those two were rerun with loopback permission and passed. No code
change was needed for those environment errors. The initial failure log remains
preserved alongside the separate successful recheck logs.

## Remaining production gate

No actual evaluation is authorized by this task. Before a later native run:

1. Verify exact parent/V4 checkpoint bytes and the unchanged original training
   bindings. Use the pinned producer-container CPU forward path and independent
   selector reconstruction to reproduce the old frozen plan. Do not rewrite a
   checkpoint's old source identity to match new code.
2. Wire and mock-test an evaluation-only production adapter and independent
   watchdog with the bounds above. Do not call the old continuation initializer:
   it creates a learning engine and runs imitation/PPO. A genuine new native
   initialization must pass its startup diagnostics and the original shot guards.
3. Require separate independent native trace/scorer/startup/inference proofs for
   these 210 requests. The existing old 420-request completeness gate remains
   unchanged; the new result must have its own schema and cannot promote V4.
4. Check current 4090 and shared scene-lock ownership, storage, pinned runtime,
   immutable source, untouched output and explicit execution authorization.
   Leave mjlab, the 4070, readers, containers and hardware alone.

## Learnings reviewed from `sircus-mjlab`

The recent chat and its completed results were reviewed during this preparation:

- The 97-recipe imitation refit reduced fitting error but scored **38/70 versus
  40/70** for its predecessor, gaining four positions and losing six. Better
  training fit alone is insufficient evidence of improved task performance.
- Its regression diagnosis separated three IK refusals, two ball–robot contacts
  and one rim miss. Preserve successful recipes and distinguish setup, flight
  and containment failures when reviewing this evaluation.
- Its wrist-rotation test removed robot contact in **6/6 drops**, but recovered
  only **3/6 captures**, with 18 controls and an independent audit. Clearing an
  obstruction is different from landing the ball. The existing policy has no
  gripper-yaw action, so rotation belongs to an explicitly versioned execution
  behavior before any later policy-action extension.

Those studies used their own MuJoCo geometry, scoring and 64-substep protocol.
Their commands, wrist behavior, drop-gap values and scores are not transplanted
into this fixed Isaac evaluation. Such a change would be a separate experiment.
See [imitation comparison PR 112](https://github.com/kadiru/sircus/pull/112) and
[wrist-rotation PR 114](https://github.com/kadiru/sircus/pull/114).
