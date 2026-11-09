# Current PPO: feasibility decomposition, November 8, 2026

The reusable [coupled-arm layer](../../sircus/isaac/JOINT_FEASIBILITY.md) matched
the original geometry/path admission on **420/420 saved final PPO requests**.
The three frozen checkpoints, actions and original native outcomes are unchanged.
This is retrospective analysis, not a new capture-performance experiment.

## Final unseen TEST, 70 requested positions per seed

| Original PPO seed | Preflight declined | Execution declined after preflight | Released | Released failures | Captures | Clean | Capture / all requests |
|---|---:|---:|---:|---:|---:|---:|---:|
| 20261121 | 17 | 1 | 52 | 24 | 28 | 10 | 40.0% |
| 20261122 | 17 | 0 | 53 | 25 | 28 | 7 | 40.0% |
| 20261123 | 41 | 1 | 28 | 0 | 28 | 14 | 40.0% |

These are the same 70 TEST coordinates across seeds, not 210 independent
positions. Every requested position stays in the denominator. No rejected
action was replaced and no historical failure was rescored as success.

Seed 20261122 remains the proposed parent, based on its previously observed
validation result of **12/35 captures, 5 clean, 9 rejected**, rather than TEST.
Its TEST decomposition is **17 preflight declines +25 released failures +28
captures =70 requests**. Admission is 53/70 =75.7%; conditional capture is
28/53 =52.8%; overall capture remains **28/70 =40.0%**. Its clean rate remains
7/70 =10.0%. The new layer does not turn that into a 52.8% overall score.

For that seed, the 17 TEST preflight declines were 12 board endpoint IK
unresolved, one dropper endpoint IK unresolved, and four necessary outer-reach
bound violations. These describe specific proposed configurations, not proof
that those bucket positions have no feasible alternative shot.

Seed 20261123 released only 28/70 positions and captured all 28. Its 100%
conditional rate therefore coexists with the same 40% overall rate. This is why
coverage and released-shot accuracy must stay separate.

## What was checked

- Three prescribed final 10,000-request checkpoints, each on 35 known-TRAIN
  diagnostic, 35 validation and 70 TEST requests: **420 exact requests total**.
- Original passed audits and experiment, split, scene, checkpoint, proof and
  trial hashes; exact position coverage; saved unresolved/successful IK answers
  independently checked with FK; original geometry and 16-sample path checks.
- **420/420 original admission decisions matched**, with zero changed actions.
- First ten final known-TRAIN requests of seed 22 also passed **10/10 parity**
  using fresh CPU robust solves: eight admitted and two declined. No flight.
- **Zero new native shots, optimizer updates, model changes or services started.**

[summary.json](summary.json) binds the compact counts, input receipts and current
layer/analyzer source hashes. Detailed decisions and raw native evidence remain
in ignored local storage. The shadow replay took about seven seconds on the
Mac; that excludes IK optimization and is not a benchmark of full preparation
or saved GPU time. The ten fresh CPU solver checks took about one second.

The controller source now uses this shared admission API. Its existing motion,
correction, settling and release checks remain mandatory. Pure/mock regression
and saved-data parity do not establish new native execution or physical safety.
No retraining or board-goal interface was added. All original comparison and
compute measurements remain in the [completed benchmark](../isaac-sweep-ppo-10k-20261106/README.md).
