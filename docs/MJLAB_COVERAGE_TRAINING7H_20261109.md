# Completed bounded coverage study — November 9, 2026

The v2 experiment completed all **2,757 requests** in **235.1 worker minutes**.
New imitation and selected PPO each captured **28/70 (40%)** on unseen positions;
PPO gained one and lost one. Previous PPO weights scored 31/70 on this same split
and expanded prior. [Complete results](../baselines/mjlab-coverage-training-20261109/README.md).
Independent corrected CPU audit passed **857,507 checks**; no policy promoted.
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

## Verification follow-up

The original pipeline ended with `audit_failed` after native training/evaluation
completed. Its auditor compared a tilt value in degrees with the same absolute
tolerance used for metre-valued commands. NumPy/Torch float32 forward passes
agreed within the existing normalized tolerance, but one scaled tilt difference
exceeded the mixed-unit check. Saved commands were not changed.

Revision `normalized-command-units-v1` keeps independent normalized tolerance
2e−6, adds 1e−12 checks of recorded latent → tanh → physical command, and verifies
the original auditor source and failed proof by exact hashes. The original
pipeline state, failed audits, source, records and checkpoints remain unchanged.
The full correction passed 857,507 checks in 189.8 CPU seconds; all bootstrap/PPO
weight, loss and sample replays had zero discrepancy. All 455 saved action
transforms were exact. Nineteen pinned CPU unit tests and 17 JS tests passed.

Additional Ubuntu paths (relative to `sircus-mjlab`):

- Corrected auditor and reproducible invocation:
  `verification/coverage-training7h-v2-normalized-command-v2/launch.json`.
- Corrected proof: `output/coverage-training7h-v2-audit-normalized-command-v2.json`.
- Publication source: `verification/coverage-training7h-completion-publication-v1/`.
- Separate publication state and receipt:
  `output/coverage-training7h-v2-followup-20261109.publication-state.json` and
  `.publication-receipt.json`. These record correction/publication time separately
  from original launch, finish and deadline; they do not rewrite worker history.

The current study's final data is published once to the existing UI. Original
failed states remain evidence. Read-only follow-up should use the corrected proof
and publication receipt, rather than interpreting the original stopped publisher
as a still-running training task. No new GPU requests, model fitting, hardware
access, Isaac edits or service restarts were added by verification/publication.
