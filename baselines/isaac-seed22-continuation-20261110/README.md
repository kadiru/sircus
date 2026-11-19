# Direct seed-22 continuation: audited native result

The continued continuous PPO model caught **28/70 unseen TEST requests**,
versus **26/70 for the original model with the same geometry checker**.
Both made eight clean captures. Rejections increased from nine to 20.
The frozen continuation's independent CPU audit passed; this is a small capture gain with
worse admission, not an overall improvement or a promoted model.

The [protocol](../../sircus/isaac/SEED22_CONTINUATION.md) restored original seed
`20261122` weights and Adam state, then trained direct continuous proposals
through a fixed geometry wrapper. No candidate-bank lookup or imitation was
used. Original physics, scoring, controller and release guards were retained.
The completed six-run sweep/PPO benchmark and separate V4 history remain intact.

## Unseen final comparison

Each version requested the same 70 fresh bucket positions once, with frozen
predictions and no outcome search. Rejections count as failed requests;
clean captures are a subset of captures. A finite refusal does not prove that
the bucket position is physically impossible.

| Version | Captures | Clean captures | Rejected | Released without capture |
|---|---:|---:|---:|---:|
| Original seed 22, directly | 25/70 (35.7%) | 8/70 (11.4%) | 14/70 (20.0%) | 31/70 |
| Original seed 22 + checker | 26/70 (37.1%) | 8/70 (11.4%) | 9/70 (12.9%) | 35/70 |
| Continued seed 22 + checker | 28/70 (40.0%) | 8/70 (11.4%) | 20/70 (28.6%) | 22/70 |

Against the original-plus-checker version, continuation gained captures at
eight positions and lost them at six: **net +2 captures, +11 rejections**.
It gained three clean captures and lost three: no net clean improvement.
The checker alone gained one capture without losing a capture against the raw
parent. These are paired descriptive results from one continued seed, not a
repeated-seed reliability result, an equal-budget sweep comparison or hardware
validation. The arms still use scripted motion; this does not train joint control.

## Experience and monitoring

- Retained the **35 original TRAIN positions**. New shot proposals increase
  experience at those positions, not the number of TRAIN target locations.
- Used **35 new monitoring positions and 70 new TEST positions**, frozen
  without outcome or reachability filtering and separated by at least 25 mm
  from discovered historical coordinates and each other.
- Completed **8,960 additional TRAIN requests**, including all 2,200 rejections:
  2,876 captures, 485 clean captures and 3,884 released noncaptures.
- Completed **67 additional PPO update rounds / 268 Adam steps**. Including the
  inherited 10,000 requests / 73 rounds / 292 steps, totals are **18,960 requests,
  140 rounds and 560 steps**. The prescribed time stop reserved final assessment
  and cleanup inside the eight-hour cap; the 30,000-request cap was not reached.

| Additional TRAIN requests | 0 | 1,000 | 2,000 | 5,000 | 8,960, final |
|---|---:|---:|---:|---:|---:|
| Monitoring captures / 35 | 16 | 14 | 17 | 12 | 20 |
| Monitoring clean / 35 | 5 | 5 | 6 | 5 | 5 |
| Monitoring rejected / 35 | 1 | 3 | 6 | 6 | 7 |

The zero-request checker point uses the unchanged parent network. The raw
parent monitoring baseline was 13 captures / four clean / seven rejections.
Monitoring did not select the final model; the last completed checkpoint was
frozen before TEST. [Aggregate evidence](publication-summary.json) includes all
259 scalar exploratory TRAIN curve points, monitoring and paired TEST counts.

## Actual computing cost

| Scope | Measured cost |
|---|---:|
| Whole native run, including earlier startup attempts and shutdown | **7 h 59 min 33.736 s** |
| Earlier closed startup attempts, included above | 4.024 s |
| TRAIN episodes, setup through recording | 27,211.523 s wall |
| Assessment episodes, setup through recording | 1,042.367 s wall |
| Geometry, already inside episode wall | 6,692.595 s wall / 25,031.637 process CPU s |
| Network proposals / updates / checkpoints | 0.515 / 1.031 / 1.614 s wall |
| Native process lifetime peak RAM | 7,416.543 MiB |
| Peak sampled whole-device GPU occupancy | 6,217.438 MiB |
| Separate independent CPU audit | **20 min 52.897 s wall** |
| Audit user / system CPU | 1,398.129 / 529.484 s |
| Audit maximum RSS | 1,056,468 KiB |

Episode wall already includes geometry; these are not additive independent
costs. GPU occupancy includes the device's physics/rendering allocations and is
sampled, not a policy-only peak. RAM is a lifetime high-water measurement, not
phase-local memory. The CPU audit uses a pinned `python_os_wait4` collector for
the actual audit command and its waited-for descendants; maximum RSS is a
high-water maximum, not summed simultaneous memory. Audit costs are separate
from the eight-hour native budget. Energy and monetary cost were not measured.

## Evidence, recordings and closure

- [Publication summary](publication-summary.json), SHA256
  `51337efd86ce7e169f2cf441b0f8dc76582cfe63fcd41f0c8f434c4b4a8749ce`:
  an allowlisted derivative bound to exact original result, audit, checkpoint,
  watchdog, learning-curve and resource bytes. Raw observations and coordinates
  remain private; this does not replace the original hash-bound evidence.
- [Media summary](media-summary.json): four prescheduled native clips, with
  browser playback verified. TRAIN cohorts after 0 / 1,000 / 5,000 new requests
  contain **35 / 15 / five paid requests** respectively. The final clip shows
  **the first 35 of the 70 continued-model TEST requests**, not the full TEST set.
  Clip selection was scheduled before outcomes; clips are not extra trials.
- Producer commit `77c57cf0921c694a3321a4d97faae3d7af345d27`;
  tree `d5df8156d0fe8b932f4819dd61cf2123bcb6b27a`;
  archive SHA256 `5a1eb91b5eeb2866f7ed92543d9f687e9896d2ef346cb4bb8bbc8d9efa12437d`.
- Parent checkpoint SHA256
  `57d7658e9ed2f72ad09cf4e8523ab497ccbe0708364f1b8ec4a4c5c1aef1fe9f`;
  saved final checkpoint SHA256
  `79932bc9e0399584ed0ac6b80c95dbc35a5f159f6fe3178beba713d05fe4b7a6`.
- Protocol fingerprint
  `47dad0235944c0e3d0ba8c4f4cee5d461deb18829c547f85a9885aaf1e0dba50`;
  independent audit SHA256
  `b800b230cd5516257d80168c31edb56376fcb155f7e733c147d405ad038804ee`;
  unchanged auditor source SHA256
  `a8e748d7547d85540836bd8eba530b1145e8bfdbad56e192420ec96636adb54a`.

After all learning, assessments and final evidence completed, `App.close()`
hung. One SIGTERM to the exact verified owned process produced exit code zero
and a safe lock release before the native cap. No work was repeated. The
container lacked `/usr/bin/time`; a CPU-only resource-wrapper fix enabled the
sole original audit invocation. Earlier failed observer receipts were preserved;
the producer, physics guards and independent auditor were not modified.

Software verification covered 1,516 Isaac cases with 121 optional-dependency
skips; two sandbox-blocked localhost cases passed their permission-enabled
recheck. The 34 physical mocks, five simulation cases, 59 Node cases, nine
numerical stages and 40 private publication checks passed. These software checks
are separate from the passed native audit. No model promotion or further
experiment follows automatically.
