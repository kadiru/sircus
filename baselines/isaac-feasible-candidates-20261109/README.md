# Seed 22 candidate-list preservation audit

Built a separate TRAIN-only candidate bank and checked it against **all 1,534
successful paid TRAIN shots**, including the **485 clean captures**, from the
unchanged original seed-20261122 run.

**All 1,534 survive exactly. Zero successful TRAIN actions are excluded.**
All 485 clean examples are retained and qualified too. This verifies action
availability and kinematic/path admission, not another successful ball flight.

| Candidate source | Proposed | Qualified | Declined |
|---|---:|---:|---:|
| Exact final-parent choices | 35 | 28 | 7 |
| Samples around learned parent preferences | 560 | 232 | 328 |
| Broad samples across original bounds | 2,240 | 20 | 2,220 |
| Historical successful paid TRAIN shots | 1,534 | 1,534 | 0 |
| **Total** | **4,369** | **1,814** | **2,555** |

All 4,369 proposals are distinct exact context/five-parameter vectors. No
rounding or nearest-action replacement occurred. All **35 TRAIN positions**
have at least one qualified candidate. The 485 clean examples are a subset of
the 1,534 historical successes, not additional source entries.

## Why other candidates were declined

| Primary reason | Count |
|---|---:|
| Requested flange beyond a necessary arm reach bound | 2,166 |
| Dropper endpoint IK unresolved | 197 |
| Board endpoint IK unresolved | 164 |
| Board/bucket enclosing-volume overlap proxy | 24 |
| Board-arm/table capsule overlap proxy | 4 |
| **Total** | **2,555** |

These are candidate-specific, short-circuited primary reasons. Numerical IK
failure does not prove a bucket target impossible. The proxies can reject safe
configurations. The seven declined parent mean actions were two reach-bound,
two dropper IK and three board IK failures; these were not historical captures.
The original final known-TRAIN evaluation already rejected those seven choices.

## What was preserved and checked

- Original seed-22 model SHA:
  `57d7658e9ed2f72ad09cf4e8523ab497ccbe0708364f1b8ec4a4c5c1aef1fe9f`.
- Original source, experiment, splits, scene and independent audit receipt.
  All **289 paid TRAIN packets and 10,000 request slots** were hash-verified.
- Exact parent mean actions/latents from the final **known-TRAIN** frozen
  prediction packet; learned Gaussian scales from CPU-only checkpoint reading.
- Fixed union: one exact mean, 16 parent-distribution samples and 64 broad
  Sobol proposals per TRAIN position, plus every paid TRAIN capture. Seed
  20261124; no outcome ranking or anchor-box shrinkage.
- Historical successful joint solutions independently verified, then used as
  warm starts to the original robust solver. Same geometry/FK/joint-limit and
  16-sample common-start path gates for every candidate; no weakened guards.
- Independent proposal regeneration, seed association and numerical gate audit
  passed. Successful and clean TRAIN denominators checked separately.

The [compact summary](summary.json) and [independent audit](audit.json) bind the
generated bank and complete retention CSV by SHA256. Full candidate actions,
joint fits and native source evidence remain in ignored local storage. An
independent clean-example accounting check covers all 485 clean demonstrations.

**Zero optimizer updates, new native shots or held-out shot files read.** No
simulation or existing service was started or restarted. The model, original
benchmark and its published performance remain unchanged. This bank does not
establish improved capture, native execution safety or physical validation.

[Preparation method and command](../../sircus/isaac/CANDIDATE_BANK.md) ·
[Frozen benchmark](../isaac-sweep-ppo-10k-20261106/README.md)
