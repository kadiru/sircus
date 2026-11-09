# Coverage recovery handoff — November 9, 2026

Status: implementation and preflight. No new result is claimed by this document.
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
