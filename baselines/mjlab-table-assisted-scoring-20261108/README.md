# Table-assisted captures: saved-shot rescoring

Completed November 8, 2026. The user accepts board → table → bucket when the
whole ball stays inside for one continuous second. Version 3 changes the task
route and reward gate; the original v2 containment, physics checks, files,
labels and checkpoints remain intact.

## Same recorded trajectories, changed objective

- **8,037 authenticated native trajectories** were evaluated
  by production and independent-oracle scorers under both definitions.
- Accepted task captures: **4,409 → 4,532**;
  **123** recorded captures now receive credit
  because table contact is permitted. No previously accepted capture was lost.
  Newly credited attempts: **115 training, 6 validation and 2 test** across
  their separately identified historical studies.
- **3,018 compatible requests** remain setup rejections
  or withheld requests; they have no observed flight. **11**
  native outcomes retain their invalid-physics classification.
- **71,155 unsupported historical index entries** remain excluded.
  They are not known unique trials and are not converted into policy failures.
  776 authenticated duplicate index rows were removed.
- **64,883 audit checks passed**, with no evidence
  contradictions. **Zero new physics trials, training requests or model updates.**

These aggregate counts mix training, evaluation, controls and diagnostic shots.
They are **not a new policy success rate** and do not show that any trajectory or
model improved. The complete scalar snapshot keeps study, method, phase and
64/128-substep groups separate. Different historical experiments are not a
fair head-to-head comparison merely because their counts share this report.

## Study totals

The table below is an inventory of the same saved outcomes, aggregating each
study's phases and timesteps only for compact accounting. Use the explicit
groups in [results.json](results.json) or the expanded UI breakdown for policy
comparisons; do not pool substeps for a numerical reliability claim.

| Study | Native trajectories | Old captures | New captures | Newly credited |
| --- | ---: | ---: | ---: | ---: |
| arm-clearance-v1 | 88 | 42 | 42 | 0 |
| calibrated-benchmark-v2 | 5467 | 3164 | 3277 | 113 |
| clean650-fine-v2 | 106 | 19 | 19 | 0 |
| clean650-raised-v3 | 120 | 17 | 17 | 0 |
| clean650-v1 | 155 | 17 | 17 | 0 |
| clearance-correction-v1 | 27 | 12 | 12 | 0 |
| clearance-horizon-v1 | 15 | 6 | 6 | 0 |
| consistent-flight-v2 | 116 | 64 | 65 | 1 |
| gap-coverage-v1 | 96 | 54 | 54 | 0 |
| gap-flight-correction-v1 | 30 | 15 | 15 | 0 |
| impact650-v1 | 146 | 32 | 32 | 0 |
| physics-ensemble-v1 | 204 | 113 | 113 | 0 |
| policy64-diagnosis-v1 | 64 | 44 | 44 | 0 |
| policy64-v1 | 1103 | 723 | 732 | 9 |
| replay650-v1b | 18 | 9 | 9 | 0 |
| tilt30-650-v1 | 54 | 27 | 27 | 0 |
| xy650-v1 | 228 | 51 | 51 | 0 |

The existing fixed64 policy64 TEST counts remain **20/50 for imitation** and
**18/50 for PPO**, including their original setup rejections. The new definition
does not change those particular held-out outcomes.

## Concrete example

`ppo-20261411-train-00221` was a historical PPO **training** attempt, not a
held-out result. Its measured contact sequence is board **0.137s**,
table **0.453s**, first whole-ball entry
**1.146s**, completed retention **2.146s**.
Its task reward changes **0.025282 → 4.125282**.
The original recorded replay remains linked in the preceding evidence panel.
Other examples use table-force presence to label a route; that label alone does
not prove a causal table rebound. First entry may be followed by an exit before
a later continuous retention interval.

## Scope and next step

The [v3 scoring contract](../../sircus/mjlab/SCORING_TABLE_ASSISTED.md) and explicit
native adapter are ready for a future experiment. Current models and the active
shot solver were not retrained or switched. A future trainer must use this same
version for reward, selection and evaluation, with fresh held-out positions.
No 128-substep verification, noise experiment or new physics run was initiated.
No 4070, Isaac, hardware or service configuration was changed or restarted.

Open `http://127.0.0.1:18781/#tableAssistedScoring` for the audited embedded
snapshot. It is installed additively without restarting the viewer.

## Provenance and validation

- Immutable producer commit: `8eb5452418238eece6fb531ec0e446c496d55ee2`.
- CPU-only execution: two workers; CUDA hidden and BLAS/OMP each limited to one
  thread. Input metadata, contracts and dense archives were checksum-bound and
  rechecked after scoring. Original evidence was not edited.
- The full derivative records, native traces and full audit remain outside Git.
  This report contains only sanitized scalar counts, examples and hash bindings.
  Source hashes are retained in `results.json`.
- Local follow-up indexing excludes recognized prior derivative reports so a
  future root-wide scan cannot count its own previous output as new trials.
  This does not alter the initial producer or its saved results.
- Repository mock regression suite, targeted scoring/identity/denominator tests,
  UI tests and repository file/credential checks passed. This is retrospective
  numerical evidence, not physical validation or an independent re-audit of the
  original simulator execution.
