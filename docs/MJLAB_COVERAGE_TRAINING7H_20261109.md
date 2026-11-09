# Bounded coverage study launch — November 9, 2026

The corrected v2 experiment is running, not a completed learning result.
[Protocol](../sircus/mjlab/COVERAGE_TRAINING7H.md),
[draft PR #102](https://github.com/kadiru/sircus/pull/102),
[local UI](http://127.0.0.1:18781/#coverageTraining7h).

Frozen producer: `1952dd5993f3d3dd0fdb52ea4eff2dbc72d36862`.
Archive SHA-256: `66a821f65f338d8e844309a886fa5cd025270e1391e7b3fe7e10326776baef7c`.
Launched **01:11:37 PST**; absolute deadline remains **07:59:00 PST**,
including the failed original startup and retry. The pipeline may finish earlier.

Initial matched controls passed: three direct captures, three table-assisted
captures, three expected misses, no invalid physics. The original v1 launch
stopped after nine controls and zero TRAIN requests when the expanded geometry
bank changed a control wrist branch. Its data is retained separately. V2 binds
original control proofs/joints; all four policies still use the same expanded
19-pose starting bank. No physics, scoring or setup guard was relaxed.

The planned maximum is 2,757 new requests: 1,260 search, 1,024 PPO across two
seeds, 175 validation, 280 TEST and 18 controls. Search spans 105 training
positions; validation has 35 positions; TEST has 70 unseen positions. Rejections
consume attempts. Frozen results and recorded examples remain provisional until
independent audit. No policy is automatically activated.

## Read-only follow-up

Paths are relative to the Ubuntu `sircus-mjlab` workspace:

- Source: `studies/coverage-training7h-v2/`.
- Evidence: `output/coverage-training7h-v2/`.
- Supervisor: `output/coverage-training7h-v2.pipeline.json` and `.pipeline.log`.
- Budget: `output/coverage-training7h-v2-budget.json`.
- Audit: `output/coverage-training7h-v2-audit.json` when complete.
- Optional replay: `output/coverage-training7h-v2-video/replay.mp4` and `replay.json`.
- Display snapshot: `coverage-training7h-v2-snapshot.json`.
- Publication receipt: `coverage-training7h-v2-publication-receipt.json`.
- Launch receipts: `launch-receipts/coverage-training7h-v2.json` and
  `coverage-training7h-v2-publication.json`.

Check actual process and GPU state before acting; recorded launch PIDs are not
proof a job is still running. Do not start a duplicate run or edit the frozen
source/evidence. The finite publisher updates the existing UI without restarting
the viewer. Original v1 evidence and its stopped snapshot remain preserved.

Full repository regression tests, 53 pinned Ubuntu CPU tests and 13 browser-JS
unit tests passed. Actual original control proof/joint hydration was also checked
on CPU before native execution. GitHub CI passed for producer `1952dd5`.
The running 4070 fountain process and existing viewer retained their original
PIDs/start times. No Isaac setup, physical hardware or service restart was used.

Next review: confirm full request accounting and audit; compare capture rates,
setup declines and flight misses for current BC/PPO and new BC/validation-selected
model across all 70 TEST positions. Report any truncated budget or failure before
claiming improvement. Saved video clips illustrate examples, not aggregate success.
