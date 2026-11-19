# Strict-retention contact-boundary diagnostic — November 3, 2026

This is a post hoc explanation of six selected **validation** failures in the
first matched-budget sweep run. It is not a new capture evaluator, a held-out
performance estimate, or a change to training, scoring or historical results.
No final-test outcomes were analyzed and no rollouts were added.

The unchanged scorer reproduced all six original results. Each ball entered
whole early, remained below the rim after entry, and finished whole inside at
four seconds. Brief failures of the nominal wall-clearance predicate reset its
continuous one-second retention clock. Every reset lasted one 240 Hz sample
(4.167 ms); there were no post-entry rim-height or interior-floor violations.

| Validation layout / repeat | Entry (s) | Longest retention (s) | Wall resets | Maximum nominal overlap (mm) |
|---|---:|---:|---:|---:|
| 06 / 0 | 0.546 | 0.883 | 30 | 0.150 |
| 06 / 1 | 0.463 | 0.975 | 7 | 0.260 |
| 08 / 0 | 0.554 | 0.850 | 6 | 0.033 |
| 10 / 0 | 0.400 | 0.833 | 21 | 0.115 |
| 10 / 1 | 0.396 | 0.696 | 15 | 0.122 |
| 10 / 2 | 0.400 | 0.517 | 34 | 0.724 |

The current wall epsilon is 0.001 mm. These overlaps are computed from the sphere
and nominal polygon-wall planes; they are **not measured engine contact depths**.
Compiled collision geometry, soft contact and integration contributions remain
to be separated. [MuJoCo's solver documentation](https://mujoco.readthedocs.io/en/stable/modeling.html#solver-parameters)
explains why soft constraints can permit contact penetration.

Across the complete 36-request validation cohorts at 0/256/512 training requests,
original captures were 17/12/16. Each cohort had the same 15 pose rejections and
21 native releases. The six cases above were selected because they lost capture
at 256; they do not estimate failure prevalence. Original labels stay unchanged.
The formal metric is reproducible strict geometric containment. These results
expose a material limitation for ordinary bucket-capture/adaptation rankings.

## Evidence and timing

`validation-wall-diagnostic.json` binds the original contract, run report,
source files, native cache files and extracted traces by SHA256. Raw arrays and
local operational paths remain outside Git. The diagnostic is separate from
the full study/provenance audit and does not prove continuous collision geometry.

Native scoring has 960 samples at 240 Hz over four seconds. The canonical
`trajectory_dt_s=1/60` field belongs to the display trail. Legacy native NPZ files
have no timestamps: scoring cadence is established from the frozen physics,
hash-bound producer code and dimensions, rather than independently measured
timestamps. Scoring stops on its original terminal condition; scored-prefix and
full-recorded-trace end states are reported separately.

Diagnostic code: commit `34e0c0d`, version
`sircus-validation-wall-diagnostic-v2`. The v1 helper refused the real row's
display cadence before producing output; v2 corrects that diagnostic-only guard.
Execution-critical benchmark code and evidence were never changed.

## Reproduction and next step

With the matching original reference and raw benchmark evidence available:

```bash
CUDA_VISIBLE_DEVICES= python -B -m sircus.mjlab.benchmark_wall_diagnostic \
  --run STUDY/runs/sweep-seed20261311 --reference REFERENCE \
  --milestones 0 256 512 --output NEW_EXTERNAL.json
```

Repeated `--request-id` arguments restrict extraction to the six IDs listed in
the JSON while preserving full cohort counts. Without them, all requests from
the selected complete validation cohorts are analyzed. Output must be new and
outside the immutable study/reference. Verify this compact bundle using
`sha256sum --check SHA256SUMS` (Ubuntu) or `shasum -a 256 -c SHA256SUMS` (Mac).

Before a corrected benchmark, freeze independent contact controls: central and
wall rest/tangency, retained wall impacts, rim-assisted captures, genuine partial
exits/re-entry and outside negatives. Record compiled geometry/contact distances
and velocities. Define any contact envelope without method outcomes, then
version the scorer/reward/contract and evaluate afresh. Do not choose a tolerance
to recover these six failures. The current [benchmark protocol](../../sircus/mjlab/SWEEP_RL_BENCHMARK.md)
continues unchanged and its results retain their original definition.
