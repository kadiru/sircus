# Drop-gap restoration: completed two-case comparison

Both inspected flight regressions recovered when the drop gap was reduced to
their matched successful parent value: **6/6 corrected captures**, versus
**0/6 unchanged recovery captures**. The exact parent recipes still captured
**6/6** and remain preserved. Each result comprises three nominal repeats per
case, under the same fixed 64-substep simulation and version 3 scoring.

This is a bounded posthoc development experiment, not another training run.
The original recovery-imitation benchmark remains **37/70**; do not relabel
these six corrected requests as fresh held-out evaluation or increment that
historical score. [Compact evidence](../baselines/mjlab-regression-drop-gap-20261109/README.md)
and the [frozen protocol](../sircus/mjlab/REGRESSION_DROP_GAP.md) are canonical.

## What changed and what was observed

Action coordinate 4 (`drop_gap_m`) was restored from the matched captured parent
recipe onto the **original recovery action**. The other four coordinates and
the bucket position stayed unchanged. The previous height correction was not
combined with this one. Absolute release height decreased by the same gap delta.

| Context | Bucket X/Y mm | Board X/Y mm | Board height mm | Tilt | Original → corrected gap mm | Actual release-center Z mm | Captures |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `r0-c5-n1` | 1053.282, -619.288 | 648.164, -196.698 | 424.899 | 31.478° | 133.998 → 103.431 | 565.830 | 3/3 |
| `r0-c6-n1` | 1124.875, -593.199 | 652.624, -198.229 | 443.245 | 30.546° | 147.678 → 109.932 | 590.677 | 3/3 |

The full context names begin with `recovery-test-`. Parameters above are rounded
for reading, not replacement commands. Full-precision values and the source
record identities are in the compact JSON. The table's release Z is the measured
initial native ball center; board height is the requested parameter in the
existing simulation frame.

Fresh strict IK admission was retained. In every corrected repeat, the actual
board-arm joint vector, board center and board normal were identical to the
saved recovery source. Thus there was no observed bouncer-branch or settled-board
pose change in this test. Dropper joint vectors changed, as expected for a lower
release; the maximum first-repeat joint changes were 0.105979 and 0.170286 radians.
No arm contact occurred in any diagnostic release.

All six corrected captures used **board → table → bucket** routes. Whole-ball
entry and one continuous second of retention determined success; the table
bounce was allowed by the unchanged version 3 scoring. Representative first
repeats show the following measured sequence, in seconds after release:

| Context | Board | Table rebound | First bucket contact | Whole-ball entry | One-second capture |
| --- | ---: | ---: | ---: | ---: | ---: |
| `r0-c5-n1` | 0.138477 | 0.474935 | 0.689323 | 0.872396 | 1.872396 |
| `r0-c6-n1` | 0.143490 | 0.490039 | 0.720117 | 0.918424 | 1.918424 |

The corrected first bucket contacts occurred at ball-center Z **218.161 mm**
and **217.436 mm**, compared with **185.570 mm** and **192.247 mm** in the original
recovery misses. The shorter initial drop changed the board/table rebound path
and approached the bucket higher. This is a concrete local result; it is not a
rule that reducing drop height improves every shot.

## Accounting, controls and independent audit

- Exactly **36 requested slots / 36 native releases** completed: two cases ×
  three variants × three repeats, plus nine formal controls before and after.
- Parent exact: **6/6 captures**; recovery exact: **0/6**; corrected: **6/6**.
  No setup declines, cached refusals or invalid-physics attempts occurred.
- Each formal block contained three direct captures, three table-assisted
  captures and three expected misses. Both blocks passed. The six robot-contact
  misses belong to intentional negative controls, not the diagnostic recipes.
- The independent CPU audit passed **16,072 checks**, including original
  admission branches, fresh changed-action proofs, native reset and timing,
  dense trajectory rescoring and realized board-pose comparisons.
- Native worker elapsed time was **170.198 seconds**. This is wall time, not
  isolated GPU compute time. Only the 4090 ran the native experiment.

The frozen request list had no outcome-dependent search, replacement or retry.
Formal controls retained their original 17-source proof namespace; diagnostic
shots retained the 19-source prior. Strict admission, geometry, timestep and
scoring were not relaxed to obtain the captures. The exact parent recipes remain
available independently of the corrected recipes.

## Evidence and reproduction identities

- Producer commit: `1e3707058b1e903913a9c93e85a1a9fc41924441`.
- Separate UI publication commit: `335d273d2cb1e26926c482a802fce93ffd9a1ef6`.
- Contract fingerprint: `103355e9de55b139eca5797804b015e53baad5256567330657951caec45b6ea6`.
- Final report SHA-256: `dc18617048d35a60665974d9733e9efb3d66e5682a55c61ed3139bcbda64fd35`.
- Independent audit SHA-256: `8c11bc5b12371c128df092725fe1f1e0c62c5ad42defadbb80d699e5cbd1c252`.
- Branch: `codex/macbook-regression-drop-gap`.
- [Draft PR #110](https://github.com/kadiru/sircus/pull/110), stacked on draft #109.

The task uses separate immutable source/evidence directories on Ubuntu;
raw native traces, full proofs, launch receipts and generated runtime files
remain outside Git. The local ignored evidence copy is under
`.runtime/drop-gap/` in the task worktree. The compact summary is bound to the
report and independent audit hashes above, and includes exact parameters,
per-case counts, control gates and first-repeat realization comparisons.

## Interface and integration boundary

The additive `#regressionDropGap` panel shows all six variant rows, exact
settings, complete request denominators, control/audit status and actual saved
ball samples. The first repeat of every variant is available, rather than
selecting the best-looking flight. A separate posture table reports admitted
board joint-vector and settled-pose differences. Setup declines would receive
no fabricated flight.

The published snapshot is **62,097 bytes**. All **57 existing display-report
keys and the report bytes** were preserved; historical HTML/CSS outside the
new markers, including the prior 60-second movie, were unchanged. Publication
did not require a service restart or a solver/policy switch.

The existing 4070 workload and Isaac setup were left untouched. No physical
hardware, Viam configuration, perception, grasp or transit behavior was used.
No new model fitting, policy activation, noise or 128-substep study took place.

At closeout, browser verification confirmed both corrected outcomes and saved-sample playback; the worker and supervisor had exited, the 4090 was idle, all original report hashes matched, and viewer PID 6671 and 4070 PID 3750988 retained their original start identities.

Validation passed: the complete local `scripts/test.sh`, 16 focused worker/audit
checks, nine publisher tests and 12 JavaScript UI tests. The local mjlab suite
ran 1,701 tests with 87 optional-dependency skips; the Isaac suite ran 128 with
one skip. GitHub CI passed the UI revision `335d273`. Skipped local backend tests
are not native qualification; the separate 16,072-check Ubuntu audit covers this
actual fixed-64 experiment.

## Next decision

Retain both validated parent recipes and these corrected recipes as development
examples. Together with the previous height test, three of the five inspected
capture losses now have local corrections; two finite IK declines remain.
Nominal 3/3 repetition does not establish a robust success interval. Before
claiming policy improvement, choose a separate training update or frozen selector
rule, then evaluate it on new bucket positions that have not been inspected.
No additional run or automatic policy promotion is implied by this result.
