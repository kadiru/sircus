# Guarded board-position fallback

## Problem and correction

The preceding [50-position comparison](BOARD_FALLBACK_STUDY.md) captured 41/50
with the current planner and 40/50 with an experimental board-pose fallback.
At the lost target, the fallback interrupted the normal search after 12.47 s,
before it could find the clear shot found by the current planner at 16.64 s.
The fallback selected an alternative with predicted negative arm clearance;
the native ball then hit the robot. Those original results remain unchanged.

This version makes two specified changes:

1. Let the original hybrid search finish naturally. Its existing reference-path
   stall handling remains unchanged; there is no additional 12-second stop.
   A board-pose fallback can use only time left within the original absolute
   30-second decision budget, after normal planning returns a risky checked shot.
2. Replace that shot only when the new shot has a finite predicted ball-to-arm
   clearance of at least 5 mm for its predicted route and a strictly better
   finite flight score. Full endpoint IK and existing path checks still apply.

If no qualifying replacement finishes before the deadline, retain the last
checked shot from the original search. This guards replacements; it does not
claim every original shot or predicted clearance will succeed in simulation.
The flight predictor, candidate geometry menu, models, contact settings,
scorer and original interactive planner are unchanged.

## Frozen test

- Two methods: current `StalledReferencePlanner` and the new
  `GuardedBoardFallbackPlanner`, using separate reusable CPU workers and the
  same frozen learned ordering and 30-second decision allowance.
- Twenty fresh seeded target positions: eight interior, eight edge and four
  near fixed robot bases. Exclude the previous 50 targets, all recorded prior
  training/development/evaluation positions and current interactive history.
  This is a stratified comparison, not a uniform table-wide estimate.
- Each method gets one choice and at most one drop per fresh target. A setup
  decline remains unsuccessful in its /20 denominator.
- Separately, independently plan the known regression target
  **(948.252, −97.816) mm** with each method, then repeat each chosen shot three
  times. These diagnostic repetitions do not enter the fresh-position score.
- Freeze all 42 planning decisions before any new native drop. Use no observed
  outcome to rerank, retry, repair or choose a shot during this comparison.
- Execute at 64 substeps on the 4090, under the existing shared GPU lease,
  with a 30-minute native lease cap and at most 64 releases: 40 fresh, six
  diagnostic, and 18 controls. Controls bracket native execution.

Success requires a board bounce, whole-ball entry and one continuous second
inside the bucket. Direct and table-assisted captures count. Robot-assisted
retentions remain separately reported. Historical scoring is not rewritten.

## Verification and interpretation

An independent CPU audit reconstructs target separation, frozen choices and
deadlines, natural completion before board search, replacement conditions,
full pose/path evidence, native contacts and retention scores. Report setup
availability, captures out of 20, paired gains/losses, route counts, and mean,
median and 90th-percentile planning request times. Show diagnostic repeats
separately. Full request time includes worker startup and cleanup; it can
exceed the decision limit and excludes GPU waiting and drop simulation.

This is a planner correction test, not new imitation or PPO training. It uses
simulator geometry, not camera or point-cloud observations. The native ball
flies with arms held at checked endpoints; sampled geometric paths are not
driven-arm dynamics or physical validation. The live planner and seven known
successful settings are preserved. No service restart, 4070 workload or
Isaac setup change is part of this task.

## Completed results

The two methods captured at the **same 15 of 20 fresh targets**. Each released
all 20 checked shots: 12 direct captures, three table-assisted captures and five
misses. No position was gained or lost; no arm-assisted retention or invalid
physics outcome occurred. These 20 new targets differ from the preceding
50-position test, so 15/20 must not be interpreted as a decline from 41/50.

| Measurement | Current planner | Guarded fallback |
| --- | ---: | ---: |
| First-drop captures | 15/20 | 15/20 |
| Checked shots released | 20/20 | 20/20 |
| Mean full planning request | 12.936 s | 15.052 s |
| Median full planning request | 8.419 s | 10.045 s |
| 90th-percentile full planning request | 30.043 s | 30.067 s |
| Separate known-target captures | 3/3 | 3/3 |

All six known-target repeats were direct captures. The guarded fallback
activated at one fresh position and replaced no shots. The known-target
regression did not recur in these three repeats; this is not a demonstrated
capture benefit on fresh positions. **Retain the
current interactive planner**; the extra fallback is not promoted.

The saved plans explain the lack of improvement: all five missed targets used
the full 30-second original search, so the guarded board search never started
there. Those targets are (942.6, −31.7), (1193.0, 147.4), (966.7, −40.9),
(1209.6, −160.6) and (1127.6, −38.3) mm. This is a planning-time diagnosis;
their post-release contact causes have not been newly diagnosed here.

The sole fallback activation was at (1224.6, −175.0) mm, where the original
shot already captured. Of 16 additional setup checks, six failed pose/clearance
checks; four passed setup but lacked 5 mm predicted flight clearance; six
cleared the arms but had worse predicted aiming scores. Keeping the original
was correct. The next correction should make alternate board poses available
earlier or concurrently while retaining the best checked original shot, first
using these five positions as development cases. No such experiment has been
started, and they would not count as fresh evaluation positions again.

Exactly **64 native releases** completed: 40 comparison drops, six diagnostic
repeats and 18 controls. Each control block produced three direct and three
table-assisted captures, with zero captures from its three negative controls.
The preparation audit passed **57,015 checks**, and the full independent audit
passed **71,175**. Native execution used **490.827 seconds** of the shared 4090
lease. Waiting for another workload to finish was separate; it was not stopped.
The 4070, Isaac setup and existing UI service were unchanged.

[Sanitized results and provenance](../../baselines/mjlab-guarded-board-fallback-20261123/summary.json)
include all paired outcomes, diagnostic results, timings and source hashes.
The published UI section is `#guardedBoardFallback`, with earlier results and
videos preserved. Original evidence remains immutable.

## Implementation checks

Implementation is frozen at `f8206420f23bd7f3fb498b683538284d7fb1fb53`.
All 41 targeted tests pass on pinned Ubuntu; GitHub CI and the UI checks pass.
A separate CPU-only real-geometry check at the known target preserved the
original +6.3044 mm predicted arm-clearance shot, completing its decision in
12.4156 s. Exact prepared-joint replay passed without CUDA initialization or
native release. This is setup evidence, not a new capture.

Draft [PR #159](https://github.com/kadiru/sircus/pull/159) is stacked on #158.
Later documentation and result publication do not change the computational
producer or the original report.

## Reproduction

Frozen runtime:
`/mnt/robot-training/sircus-mjlab/studies/guarded-board-fallback-v1`.
Frozen evidence:
`/mnt/robot-training/sircus-mjlab/output/guarded-board-fallback-v1`.
The runtime's `.producer-commit` and `.study-args.json` preserve source and
exact arguments. Its source is a Git archive of the producer commit. Outputs
use the larger data disk; earlier system-disk evidence is read without changes.

Pinned environment: mjlab 1.6.0, MuJoCo 3.11.0, mujoco-warp 3.11.0,
Warp 1.14.0, Torch 2.7.0, NumPy 1.26.4 and SciPy 1.15.3. Python executable:
`/home/kadir/sircus-mjlab/venv/bin/python`.

Run stages in order: `initialize`, `prepare`, `preflight`, `run`, `finalize`,
`audit`. Use a new output directory for a new experiment; never overwrite this
one. `preflight` and `audit` require new `--audit-output` paths outside the
evidence directory. Preflight verifies all CPU choices and proofs but does not
claim native contact, retention or control validation. Hide CUDA for CPU
stages. For `run`, select only 4090 UUID
`GPU-ab1950b0-01dd-0fab-407f-2d17936b27f4` and retain the shared lease at
`/home/kadir/sircus-isaac/scene.lock`. Full source/choice/manifest hashes bind
the final receipt; keep logs and audit receipts outside the original evidence.
