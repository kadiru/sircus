# Seven-hour-budget coverage study — November 9, 2026

All **2,757 planned requests** finished in **235.1 worker minutes**. Seven hours
was a maximum, not a required duration. Independent CPU verification passed **857,507 checks**, including native traces,
training/optimizer replay, frozen predictions and all controls.

On the same **70 new bucket positions**, the selected new PPO policy did **not**
improve capture over new imitation:

| Frozen model | Captures / all 70 | Capture rate | Setup rejections | Released misses | Clean captures |
| --- | ---: | ---: | ---: | ---: | ---: |
| Previous imitation weights | 28/70 | 40.0% | 26 | 16 | 8 |
| Previous PPO weights | 31/70 | 44.3% | 27 | 12 | 9 |
| New imitation | 28/70 | 40.0% | 22 | 20 | 8 |
| Selected new PPO | 28/70 | 40.0% | 22 | 20 | 8 |

New PPO and imitation captured at 27 common positions. PPO gained one capture
at bucket **(215.643, −379.034) mm**, and lost one at **(995.033, −377.948) mm**.
Both failed at the other 41 positions. The previous PPO weights had the highest
observed count, three above the new models; this small single-test difference
does not establish a reliable advantage. No model was promoted.

## What ran

- **1,260 guided search requests** across 105 TRAIN layouts produced 532
  captures and one successful teacher for each of 68 layouts. Imitation fits
  these fresh teachers; the critic also uses failed TRAIN attempts.
- **Two PPO seeds**, 512 new requests and 16 updates each, started from the
  same new imitation checkpoint. All 1,024 requests completed.
- **35 validation positions** selected the checkpoint before any TEST results.
  Imitation captured 22/35. PPO seed 2026110902 scored 22 then 19 at 256/512
  attempts; seed 2026110903 scored 24 then 25. The latter 512-attempt checkpoint
  was selected. Its validation gain did not carry over to the unseen TEST set.
- **280 TEST requests:** four frozen models, one prediction and one attempt
  per model at each of 70 new positions. No outcome-based TEST retry or search.
- **18 controls:** direct captures, table-assisted captures and expected misses
  each passed three repeats before and after the experiment.

TRAIN/validation/TEST layouts span a 7×5 grid, with two distinct TEST positions
per cell. Their coordinates were frozen before feasibility checks. All models
share the expanded 19-pose geometry prior. Previous weights are unchanged, but
their new prior/split means these are new scores, not a reproduction of their
historical results. This is not an equal-total-data comparison.

## Meaning and limits

A capture requires a board bounce, whole-ball entry and one continuous second
inside. Rim and table-assisted routes count; robot/floor/unattributed contacts
do not. All setup rejections stay in the denominator. There were no invalid
physics TEST shots. Twelve invalid physics shots occurred during search/training
(seven/five) and remain failed attempts, not removed data.

Both new policies still failed at 42/70 positions: 22 setup rejections and 20
released misses each. Selected-PPO rejection reasons were finite IK unresolved
13, endpoint clearance/pose 4, fixed-base clearance 3 and action bounds 2.
Finite search failure is not proof a position is impossible. Fewer rejections
than the previous policies did not translate into more captures.

This studies privileged-geometry shot parameters at fixed 64 substeps with held
arm endpoints. It does not train perception, grasping, carrying or arm transit,
establish timestep convergence, or prove reliability at arbitrary table points.
One attempt per sampled position is not a repeated-shot reliability test.

Next: inspect the 22 setup declines and 20 misses alongside the 37 TRAIN layouts
without successful teachers. Use development data to improve demonstrations or
feasible candidate selection, then freeze another evaluation. More unchanged PPO
training is not supported by this comparison alone.

## Evidence

- [Compact summary and provenance](summary.json)
- [All 70 positions and four model outcomes](test-positions.csv)
- [Frozen protocol](../../sircus/mjlab/COVERAGE_TRAINING7H.md)
- [Launch, verification and follow-up](../../docs/MJLAB_COVERAGE_TRAINING7H_20261109.md)
- UI: `http://127.0.0.1:18781/#coverageTraining7h`

Frozen producer: `1952dd5993f3d3dd0fdb52ea4eff2dbc72d36862`.
Report SHA-256: `1d9f91dce8aa264120eaefdb532b3dc9b4fc4122b27939de387d8d1e723b0c21`.
Raw trajectories, checkpoints and host runtime files remain outside Git.

The original audit failed a mixed-unit tolerance check: an accepted float32
network difference became about 0.000002° after scaling the tilt output. The
corrected checker compares independent outputs in normalized action units and
checks the saved action transform at 1e−12. Maximum observed tilt difference was
0.000002384°; position/gap difference was 0.0000429 mm. All 455 saved transforms
were exact. Original failed proof, producer, data and checkpoint bytes are
preserved; the correction adds no native requests or training.

Corrected audit SHA-256:
`ac54eeede0a1670e802dac028ef6b0a7a9d6b34f4561e558cc53698a8ec5594a`.

## Recorded comparison

The UI includes a **39.27-second, 1280×720 replay** of the two changed outcomes:
imitation miss → PPO capture at the gained position, then imitation capture →
PPO miss at the lost position. It uses saved native ball centers and held arm
poses, rendered on CPU with no new physics or invented arm transit. All 1,178
frames passed full MP4 decoding. This selected comparison is illustrative;
the complete 70-position table above remains the performance measure.
