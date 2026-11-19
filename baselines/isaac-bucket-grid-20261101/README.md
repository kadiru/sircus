# Spatial bucket grid — completed November 1, 2026

**Synthetic development search, not a trained policy or physical validation.**
RTX4090 run `bucket-grid-20261101T193317Z` completed 689 parameter attempts in
21.5 minutes: 566 releases and 123 invalid endpoint/settling attempts. The release
plane stayed fixed; this study maps feasible shots before further RL training.

## Frozen settings and fresh confirmation

Board/release XY is (670, −100) mm. All heights below are above the modeled
tabletop. H is board-center height; gap is the vertical distance to the ball
bottom; tilt is measured from horizontal. Board heading faces the bucket.
Each selected setting received 30 independent trials with H/gap ±2 mm and tilt
±0.5°, without per-shot retuning. See [settings.csv](settings.csv) for exact values.

| Bucket XY mm | Horizontal distance mm | H mm | Tilt ° | Gap mm | Ball-bottom Z mm | Board-only captures | Exactly one board bounce | Clean / assisted |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 470, −400 | 360.56 | 350 | 24 | 240 | 590 | 30/30 | 30/30 | 28 / 2 |
| 670, −400 | 300 | 350 | 24 | 200 | 550 | 30/30 | 30/30 | 29 / 1 |
| 870, −400 | 360.56 | 350 | 24 | 240 | 590 | 30/30 | 30/30 | 29 / 1 |
| 470, −550 | 492.44 | — | — | — | — | None found in 49 search attempts | — | — |
| 670, −550 | 450 | 350 | 30 | 240 | 590 | 18/30 | 18/30 | 0 / 18 |
| 870, −550 | 492.44 | 450 | 16.5 | 170 | 620 | 6/30 | 0/30 | 0 / 6 |
| 470, −700 | 632.46 | — | — | — | — | None found in 50 search attempts | — | — |
| 670, −700 | 600 | — | — | — | — | None found in 50 search attempts | — | — |
| 870, −700 | 632.46 | — | — | — | — | None found in 50 search attempts | — | — |

All 150 fresh confirmation attempts passed endpoint/settling checks. At the
right-middle target, the historical capture metric scored 8/30, but only 6/30
qualified under the new route rule; none used exactly one board bounce.
The 30/30 near results have an approximate 95% Wilson interval of 88.6–100%.
They support local repeatability under the specified small perturbations, not
guaranteed reliability. No setting found means this finite search found none;
it does not establish physical impossibility or a zero success probability.
These searched cells are development data, not held-out policy evaluation.

Ball-center height adds the provisional 37.5 mm ball radius. The provisional
ball-center-to-flange offset adds a further 95 mm; neither is a taught robot pose.
Measured board dimensions remain 300 × 200 × 5 mm, landscape. Bucket inner top /
bottom diameters are 177 / 150 mm; outer top / bottom are 200 / 158 mm. Its 175 mm
reported height still has unconfirmed overall-height versus interior-depth
meaning; placement and material/contact properties remain provisional.

## Contact evidence and historical compatibility

The additive `sircus-board-route-v1` keeps the original board-bounce, whole-ball
entry and one-continuous-second retention requirement, and rejects detected
table/floor/robot/unattributed contact during flight. Bucket rim contacts remain
allowed. Exactly one board-contact episode is reported separately. Entry tiers
are 2 clean, 1 assisted/marginal, 0 failed route; invalid attempts remain distinct.

Five attribution probes passed (bucket floor, rim, table, floor and robot).
The old 400 mm control captures with two board contact episodes and no external
contact. The old 600 mm control captures after a measured table contact (~235 N).
Those results keep their original definitions. This stricter audit explains why
farther historical capture counts cannot be assumed to demonstrate the intended
board-to-bucket route.

Contacts use normal-force samples at 240 Hz, threshold 0.01 N, and a 50 ms quiet
gap between board episodes. Subthreshold or simultaneous canceling contacts can
escape detection. Table/floor are stationary kinematic supports in this opt-in
mode so filtered attribution works; geometry and other launch modes are unchanged.
Failed diagnostic smokes remain local. The final distinction between board-only
and exactly-one-bounce capture was frozen before this full search.

Across all released trials, maximum settled board position error was 0.848 mm,
normal error 0.104°, and joint speed 0.000182 rad/s. These validate endpoint
initialization only, not collision-free arm transit, grasping, perception or
physical transfer. No new policy checkpoint was trained or promoted.

## Reproducibility and verification

- [Protocol, launch/export commands and RL compute examples](../../sircus/isaac/BUCKET_GRID.md).
- Immutable source snapshot on Ubuntu: `~/sircus-isaac/studies/bucket-grid-v3`.
- Raw run: `~/sircus-isaac/output/bucket-grid-20261101T193317Z` (outside Git).
- Original contract fingerprint: `a70d7f581bdbb922b285b08f8ef1e43d1276d84be25a2a80ebd275cac9d854da`.
- Exporter matched all 689 raw/compact trial records, 3 evidence artifact hashes,
  and 43 Python source-snapshot hashes. [results.json](results.json) retains
  compact trial scores/parameters, seeds, protocol, provenance and verification.
- Original stored fingerprint is verified exactly. Only recomputed derived
  headings/distances tolerate cross-platform libm rounding; regenerated seeded
  perturbations tolerate 8 floating-point ULPs. Nominal actions, seeds and raw /
  compact equality stay exact. This does not loosen physical or replay guards.
- [SHA256SUMS](SHA256SUMS) covers the committed report, results and CSV. No raw
  trajectories, joints, observations, credentials, runtime config or logs included.

The finite 4090 process exited naturally. Read-only preview port18860 (forwarded
to Mac18780) shows `#bucket-grid`, including per-cell settings, uncertainty and a
verified CSV. Its source is `~/sircus-isaac/studies/grid-viewer-v1/preview.py`;
PID2125076 was observed after completion. The existing 4070 compute process976650
was untouched. Recheck current runtime state before future actions; these are
point-in-time records. No hardware, Viam, audio or physical-dashboard changes.

## Code and interface checks

Full `scripts/test.sh` passed: 34 physical mocks, 5 simulation tests, 83 Isaac
unit tests, 27 JavaScript tests and the legacy physics/scoring/orientation/study
checks. Report lint and repository credential/path checks passed. The completed
live dashboard table and per-cell uncertainty/details were checked in the browser;
served JSON/CSV hashes matched the baseline exports. The preview's display-only
JavaScript was refreshed without restarting a simulator or GPU workload.

## Decision for the next study

Use the three near settings as a starting curriculum. Middle positions remain
challenging and far positions have no qualifying setting in this search. Widen
the feasible-parameter investigation before declaring them unreachable. Profile
32/64 parallel environments against this serial contact/settling reference before
spending a long PPO budget. Train on balanced spatial development cells, select on
separate validation cells, then evaluate a frozen checkpoint on untouched cells
and continuous positions. No per-scene optimization during that final inference.

This serial run averaged 1.87 wall seconds per attempted throw, including invalid
attempts and setup overhead. At the same mix, 10,000 attempts would take about
5.2 hours before additional training/evaluation overhead. That is an extrapolation,
not a measured vectorized training rate or a guarantee of learning success.
