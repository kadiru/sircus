# Coverage recovery handoff — November 9, 2026

Status: running. The nine startup controls passed with six expected captures
and three expected misses. Search and final comparison are not yet complete.
[Protocol](../sircus/mjlab/COVERAGE_RECOVERY.md) specifies a maximum of 1,000
requests, a two-hour acquired 4090 slot, and a separate three-hour pipeline cap.

## Immutable parent

The parent is `coverage-training7h-v2`, with completed 2,757-request native
study and corrected independent audit. No parent file is overwritten.

- Parent producer: `1952dd5993f3d3dd0fdb52ea4eff2dbc72d36862`.
- Parent result commit: `531d78bf5ea673e901348225e49e02517e9bf2e8`.
- Parent report SHA-256:
  `1d9f91dce8aa264120eaefdb532b3dc9b4fc4122b27939de387d8d1e723b0c21`.
- Corrected proof SHA-256:
  `ac54eeede0a1670e802dac028ef6b0a7a9d6b34f4561e558cc53698a8ec5594a`.
- Parent contract fingerprint:
  `ea0e2cab154c6725b477dc84683f9c8eedee3f0dbfb1cd1ffa7ab166785c1eb7`.

Use the corrected proof `coverage-training7h-v2-audit-normalized-command-v2.json`.
The original pipeline and audit remain failed historical artifacts; they were
not relabeled after verification. The independent normalized-unit correction
changed no commands, model weights, scoring or simulation evidence.

## Scientific boundary

This follow-up explicitly reclassifies the parent's inspected TEST failures as
development cases. It preserves their historical score and excludes those bucket
coordinates from the new 70-position TEST set. All four frozen models use the
same 19-source geometry prior. Only recovery imitation receives additional
confirmed development demonstrations; this is not equal-training-budget PPO
versus imitation.

Setup rejections remain in the TEST denominator. The three fixed-base blockers
are reported as blocked by the current geometry/admission rule; the search does
not relax those rules. Search exhaustion is not a proof that no solution exists.
Every final prediction is frozen before outcomes, without per-position flight
optimization. Successful table-assisted and rim-assisted routes retain credit
under v3 scoring when whole-ball retention lasts one second and forbidden
robot/floor/unattributed contacts are absent.

## Runtime isolation

Preflight found the RTX 4090 idle and the 4070 running the existing fountain
workload. The existing mjlab viewer was already serving port 18781. Neither
process was restarted. Historical PIDs in launch receipts are evidence of a
particular check, not proof of current process state; inspect before acting.
The pinned mjlab/MuJoCo/Warp/Torch environment was verified without installation
or upgrades. Free disk was 19 GB; a minimum 8 GiB reserve is enforced. Raw traces,
checkpoints, logs and host-specific configuration stay outside Git.

Source synchronization does not deploy a different interactive selector. No
hardware access, Isaac code/config change, 4070 work or service restart is part
of this task. A later agent must check source, actual running state, pipeline
status and matching audit before reporting any new numerical result.

## Frozen launch and follow-up

Frozen producer commit: `522409306b836e39d2108be91c5d57fcf1fd484b`.
Source archive SHA-256:
`69917c81ef5c9b0cb0130a552b47981caee096ecd7b972acd33c51f362b6cbaa`.
Draft [PR #106](https://github.com/kadiru/sircus/pull/106) is stacked on #102.
No merge or policy promotion is requested.

Launched at **09:15:43 PST** on November 9. The worker has a maximum two-hour
acquired slot; the pipeline has a three-hour absolute deadline. Check live state
rather than inferring completion from elapsed time. The supervisor automatically
runs the independent CPU audit after a successful complete worker result.

Paths relative to the Ubuntu `sircus-mjlab` workspace:

- Frozen source: `studies/coverage-recovery-v1/`.
- Evidence and models: `output/coverage-recovery-v1/`.
- Supervisor state/log: `output/coverage-recovery-v1.pipeline.json` / `.pipeline.log`.
- Worker budget: `output/coverage-recovery-v1-budget.json`.
- Matching independent proof when complete: `output/coverage-recovery-v1-audit.json`.
- Launch receipt: `launch-receipts/coverage-recovery-v1.json`.
- Read-only UI snapshot: `coverage-recovery-v1-snapshot.json`.
- Publication receipt/log: `coverage-recovery-v1-publication-receipt.json` / `.log`.
- Publication launch and pre-update backups: `launch-receipts/coverage-recovery-v1-publication.json`
  and `launch-receipts/coverage-recovery-v1-ui-before/`.
- Saved parent diagnosis: `coverage-recovery-v1-parent-diagnosis.json`.

The additive [recovery UI](http://127.0.0.1:18781/#coverageRecovery) preserves
historical panels and refreshes saved state without restarting the viewer.
A finite publisher stops after terminal pipeline state or its own deadline.
Do not start a duplicate study, overwrite its evidence, or edit frozen source.

Validation before launch: full repository suite passed, including 1,648 mjlab
checks (expected dependency skips) and 281 JavaScript checks. Focused publication
checks passed 12 Python and 14 JavaScript tests. Pinned Ubuntu CPU checks passed
17 producer/controller tests, 10 independent-auditor tests, 1,553 actual-parent
checks and 160 actual-model checks. Saved setup/control hydration verified 55
parent connections plus three controls without initializing CUDA.

The 4070 fountain's PID changed externally during preparation. This task did not
signal or restart it; the launch receipt records the current process identity.
The original mjlab viewer remained running. Neither change in a historical PID
nor a source push should be interpreted as deployment by this task.
