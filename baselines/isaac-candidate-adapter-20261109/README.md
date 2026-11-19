# Candidate adapter: CPU verification, training off

The saved seed-22 policy can now score immutable choices from the qualified
TRAIN bank using differentiable categorical probabilities. This change adds
software components, not a newly trained model or a new native result.

| Check | Result |
|---|---:|
| Focused CPU cases executed / passed / skipped | 53 / 53 / 0 |
| Existing TRAIN positions with exact frozen-selector choice parity | 35 / 35 |
| Qualified candidates covered | 1,814 |
| Historical successful actions mapped exactly | 1,534 |
| Clean historical demonstrations | 485 |
| Positions with successful demonstration labels | 21 / 35 |
| Real-parent backward calls / optimizer steps | 0 / 0 |
| New native requests / releases / held-out shot files read | 0 / 0 / 0 |

[Readiness receipt](readiness.json) records exact input, checkpoint and source
hashes; the 35 selected candidate/support identities; numerical differences;
unchanged before/after weight hashes and global CPU RNG; no created gradients;
and the existing CPU runtime. [Verification receipt](verification.json) binds
the tested ten modules and four test files, raw log hashes, original files and
frozen reader report. It contains derived evidence only. Raw bank actions,
observations, checkpoint bytes and logs remain outside Git.

The real-parent forward check took 0.188 s, and the synthetic CPU suite took
12.814 s. Maximum CPU/reference log-probability error was `9.95e-6` against the
declared `1e-4` tolerance; mean error was `2.38e-7` against `3e-6`. Every
deterministic candidate identity matched. These are compatibility and software
cost measurements, not training speed or a new capture rate.

An earlier compact-input check failed before loading the model because inverse
math differed by at most `4.44e-16` across NumPy versions. The preserved receipt
and inputs were not overwritten. The correction keeps the exact signed stored
candidate encodings and externally supplied raw SHA, allowing `1e-12` only when
recomputing their consistency. Original parent latents stay exact; supports and
demonstrations are byte-for-byte identical between exports. No geometry,
IK, motion, release or scoring threshold changed.

The standard mock/numerical project suite also passed locally. Its Isaac portion
ran 805 cases with 96 environment-dependent skips before the last three input
hardening cases were added; the final focused CPU suite above ran all 53 without
skips. The dedicated workflow requires real CPU PyTorch coverage and no adapter
skips on future changes. Synthetic derivative tests create gradients only on
fabricated data, never an optimizer or weight update.

See the [implementation and remaining boundaries](../../sircus/isaac/CANDIDATE_ADAPTER.md)
and [proposed continuation protocol](../../sircus/isaac/CANDIDATE_CONTINUATION_PLAN.md).
The optimizer/update loop, continuation checkpoint writer, fresh split/support
preparation, durable native producer, operational supervisor and independent
training auditor remain deferred. Execution budgets are proposals, not launch
permission. Training remains off.

The previous [audited native comparison](../isaac-candidate-native-20261109/README.md)
remains 13/35 original captures versus 16/35 bank-selector captures, with no
learning improvement claimed here. The completed benchmark, original policy,
candidate bank and existing viewer services remain unchanged.
