# Wider drop-gap proposals: completed 50-position comparison

**No observed capture gain:** original and wider-gap selectors both captured
24/50 (48%) on the same fresh bucket positions. Clean captures decreased from
13 to 11. One gained capture was offset by one lost capture. This does not
establish equivalence or reliability from one nominal shot per method/layout.

| Frozen selector | Requested | Native releases | Complete captures | Clean | Setup rejects | Native misses |
|---|---:|---:|---:|---:|---:|---:|
| Original flight-consistent | 50 | 42 | 24 | 13 | 8 | 18 |
| Wider gap proposals | 50 | 42 | 24 | 11 | 8 | 18 |

All 112 requests completed, including 12 controls. There were 96 native releases,
six positive-control captures, six negative-control misses and zero invalid
physics. The separate v2 audit passed **1,441,075 checks**. No model was trained
or activated, and no new shots were run to repair the audit.

## What changed

Only candidate slots 08/09 changed: imitation gap ±36 mm became anchor gap
±120 mm, still using the imitation board pose. Fourteen other proposals, all
model weights, ranking, sixteen-candidate/top-four geometry budgets, native64
physics and complete-route scoring remained fixed. Re-ranking changed selected
actions at 19/50 positions; this does not mean only gap changed in all 19 actions.

- 23 positions captured by both selectors; 25 by neither.
- Case07, bucket XY (666.3, −530.3) mm: original miss, wider-proposal capture.
  Selected gap 188.34 → 277.10 mm; board height 316.06 → 323.45 mm and tilt
  21.90° → 23.47° also changed because a different candidate was selected.
- Case38, bucket XY (170.9, 146.3) mm: original capture, replacement miss.
  Selected gap 265.33 → 284.87 mm; other selected parameters also changed.

The 50 layouts were frozen before native outcomes, seed 20261801, within
X 100–1240 mm / Y −650–250 mm and separated from declared ancestral layouts
by at least 10 mm. They must become development data if used for further tuning.

## Audit history

The original v1 proof failed on case19's actor reconstruction. Independent
NumPy and the saved Torch computation differed by 0.0000021046° of tilt, or
0.0000002631 in normalized action units. Saved actions were not changed.

V2 retains the independent 2e-6 latent/residual tolerance, checks actions in
normalized units, and separately verifies saved tanh/action conversion at
1e-12. It authenticates the original frozen producer source and rescores all
native traces with the unchanged evaluator. The original failed proof remains
preserved alongside the new proof. This was numerical audit repair, not a
capture-rule change.

- [summary.json](summary.json): curated scalar configurations, paired outcomes,
  frozen source/model hashes, original failed proof and successful v2 audit metadata.
- [checksums.json](checksums.json): hashes for these curated files.
- [Protocol and runtime boundary](../../sircus/mjlab/GAP_COVERAGE.md).
- [Live read-only results](http://127.0.0.1:18781/#flightResidualDiagnosis).

Raw trajectories/contact arrays, credentials, host configuration, checkpoints
and logs remain outside Git. This is synthetic privileged-geometry selection
with held arm endpoints at 64 substeps, not driven transit, grasping, perception,
noise robustness, 128-substep qualification or physical validation. Provisional
geometry and historical failures remain unchanged. The original selector stays
the baseline; no wider-gap policy was deployed. The 4070 and Isaac setup were
left untouched.
