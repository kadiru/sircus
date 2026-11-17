# Reused workers and earlier checked-shot return

The CPU comparison completed on **50 fresh bucket positions**, with three
versions of the same frozen learned selector. Native drop tests and their
independent audit are pending; planning speed alone does not establish which
version should be used for the interactive demo.

## What changed

The fresh-worker version initializes a CPU process for every request. The
reused-worker version keeps a clean process and its initialized resources for
the next request. The early-return version also stops at the first completely
checked candidate whose predicted margin is at least 5 mm. All three retain
the same 30-second decision limit, weights, shot libraries, arm checks and
physics scoring. No model was retrained.

The margin is the minimum predicted separation from the arms and bucket
aperture; table-assisted shots also include the table edge. It is a prediction,
not a measured capture. [The frozen protocol](FAST_PLANNING_STUDY.md) records
the methods, seeds, exclusions, checks and reproduction stages.

## Planning results

| Measurement | Fresh worker | Reused worker | Reused + early return |
|---|---:|---:|---:|
| Checked shot available | 39/50 | 39/50 | 40/50 |
| Mean planning request | 26.47 s | 25.86 s | 22.87 s |
| Median planning request | 28.90 s | 29.87 s | 24.48 s |
| 90th-percentile planning request | 31.52 s | 31.49 s | 31.52 s |
| Mean decision time | 25.01 s | 25.09 s | 22.19 s |
| Requests reusing a worker | 0 | 25 | 29 |
| Requests reaching the deadline | 23 | 24 | 20 |
| Returns at first qualifying shot | 0 | 0 | 26 |

The measured request includes worker startup/reuse through verified return.
It excludes caller archive writes, network and browser work, so it is not
total UI latency. All three CPU methods run in parallel lanes on the same
host, with a barrier between bucket positions. Timings are specific to this
run and its host load.

Reuse alone saves **0.61 seconds per request on average**. Early return saves
another **2.99 seconds** compared with reuse alone, or **3.60 seconds** compared
with fresh workers. The slowest requests still approach the deadline.

## What makes planning slow

Finding arm joint configurations and verifying collision-free setup paths
consume **98.8–99.1% of decision time**. The neural network's ordering consumes
only **0.06–0.07%**. Reusing initialization cannot remove the main cost.

In the reused-worker run, completed intermediate-pose searches spent **355.42 s
checking paths** and **27.21 s solving IK**. Those records account for 383.53 s
of the 1,242.32 s arm-check total; candidate checks and unfinished work account
for the rest. Only 47/50 initial searches saved complete timing records, so
this is not an exact breakdown of all interrupted work.

For example, at bucket **(803.4, 107.1) mm**, the early-return planner's path
certificate contains a **−515.0° board-arm wrist turn** between intermediate
and requested joint configurations, contributing to 3,662 collision samples
in the full forward/reverse proof. This is a CPU-checked geometric path,
not motion observed during the endpoint-held native drops. Trying shorter
equivalent revolute-joint angles, then rechecking the complete path, is a
concrete next optimization; it is not part of this frozen comparison.

Timed-out or failed workers are discarded; a later request initializes a
clean one. Consequently, the reused versions still needed 25 and 21 process
starts. Query-specific caches clear between positions. Only initialized,
immutable resources are retained.

## Native comparison and interpretation

Pending: **118 checked shots**, one drop at most for each method and position,
at 64 substeps, with 18 matched controls. Requests with no checked shot remain
in the 50-position capture denominator. All 150 choices were frozen before
any native drop.

Capture requires a board rebound, whole-ball entry and one continuous second
inside the bucket. Rim and table assistance count; robot-assisted outcomes
remain separate. Inputs are simulator geometry and the native drops hold
settled arm endpoints. This does not validate perception, driven arm transit,
or physical robots.

The live planner and its service have not changed. The comparison panel is
`#fastPlanningStudy`; final capture rates require completion and a passed
independent audit. The 4070 workload and Isaac setup are untouched.

## Evidence and validation

- Frozen computational source commit: `2265f3547eb051258951fda2d304f288c631d03b`.
- Study: `/home/kadir/sircus-mjlab/output/fast-planning-v2`.
- Independent audit destination: `/home/kadir/sircus-mjlab/output/fast-planning-v2-audit.json`.
- Local and pinned-Ubuntu validation: 69 Python tests and 49 interface tests passed.
- A two-position real-model CPU smoke test preserved exact original/fresh/reused
  commands, verified reuse, and passed an independent 4,036-check audit without
  native releases. These smoke checks are not capture results.

The original attempt stopped when the shared data disk fell below the existing
8 GiB reserve, with 132 plans reported and 133 preparation files saved, and no
native releases. Its evidence remains unchanged in
`/mnt/robot-training/sircus-mjlab/fast-planning-v1`. The complete retry uses the
system disk with identical frozen inputs, source and layouts. No original-run
timings are pooled into this comparison; the reserve was not lowered and no
other workload or evidence was removed.
